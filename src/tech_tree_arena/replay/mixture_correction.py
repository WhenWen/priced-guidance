"""Post-hoc mixture correction for repeated Judge rounds over several ideas.

Recorded runs price a passing submission with ``submission_bits + repeat_bits``:
the mass of ideas that pass in any round, times the fraction of rounds with any
pass. For a submission with several ideas and several rounds this can overstate
the acceptance probability of an unguided generator, which submits one idea
drawn from the submitted probabilities. The unbiased terminal score is
``sum_i p_i * l_i / k`` (see ``contract.scoring.mixture_pass_rate``).

This module recomputes that score from recorded events only. It never edits a
recorded file: corrections are returned as new records so existing replays,
hashes, and readers stay valid. It depends only on the standard library so it
can run against run directories produced by older checkouts.
"""

from __future__ import annotations

import json
import math
from decimal import Decimal
from pathlib import Path
from typing import Any, Iterable

CORRECTION_VERSION = "mixture-judge-v1"
K_TOLERANCE = 1e-6


def load_events(run_dir: str | Path) -> list[dict[str, Any]]:
    path = Path(run_dir) / "events.private.jsonl"
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _bits(probability: Decimal) -> float:
    return -math.log2(float(probability))


def terminal_correction(events: Iterable[dict[str, Any]]) -> dict[str, Any] | None:
    """Correct the final passing submission recorded in ``events``.

    Returns ``None`` when the run has no passing submission. The returned
    record keeps the recorded values next to the corrected ones.
    """

    events = list(events)
    passing = [
        event
        for event in events
        if event.get("kind") == "submission_judged" and event.get("status") == "pass"
    ]
    if not passing:
        return None
    judged = passing[-1]
    submissions = [
        event
        for event in events
        if event.get("kind") == "submission" and event.get("attempt") == judged.get("attempt")
    ]
    if len(submissions) != 1:
        raise ValueError(f"expected one submission for attempt {judged.get('attempt')!r}")
    ideas = submissions[0]["submission"]["ideas"]
    idea_ids = [idea["idea_id"] for idea in ideas]
    probabilities = [Decimal(str(idea["probability"])) for idea in ideas]
    total = sum(probabilities, Decimal(0))
    rounds = judged["verdict_rounds"]
    passes = dict.fromkeys(idea_ids, 0)
    for verdicts in rounds:
        by_id = {verdict["idea_id"]: verdict for verdict in verdicts}
        if set(by_id) != set(idea_ids) or len(by_id) != len(verdicts):
            raise ValueError("recorded verdict round does not match the submitted ideas")
        for idea_id in idea_ids:
            passes[idea_id] += int(by_id[idea_id]["passed"] is True)
    mixture = sum(
        (p * passes[i] for i, p in zip(idea_ids, probabilities)), Decimal(0)
    ) / (total * len(rounds))
    recorded_bits = float(judged.get("submission_bits") or 0.0) + float(
        judged.get("repeat_bits") or 0.0
    )
    corrected_bits = _bits(mixture)
    return {
        "version": CORRECTION_VERSION,
        "attempt": judged.get("attempt"),
        "path_k": float(judged["path_k"]),
        "ideas": len(idea_ids),
        "judge_repeats": len(rounds),
        "idea_passes": passes,
        "probabilities": {i: str(p / total) for i, p in zip(idea_ids, probabilities)},
        "recorded_passing_mass": judged.get("passing_mass"),
        "recorded_judge_pass_rate": judged.get("judge_pass_rate"),
        "recorded_terminal_bits": recorded_bits,
        "mixture_pass_rate": str(mixture),
        "mixture_terminal_bits": corrected_bits,
        "delta_bits": corrected_bits - recorded_bits,
    }


def corrected_k(recorded_k: float, correction: dict[str, Any] | None) -> float:
    """Apply a correction to a recorded ``K``, checking the recorded identity.

    Recorded runs satisfy ``K = path_k + submission_bits + repeat_bits``; a
    mismatch means the correction does not describe this ``K``.
    """

    if correction is None:
        return recorded_k
    expected = correction["path_k"] + correction["recorded_terminal_bits"]
    if abs(recorded_k - expected) > K_TOLERANCE:
        raise ValueError(f"recorded K {recorded_k} != path_k + terminal bits {expected}")
    return recorded_k + correction["delta_bits"]
