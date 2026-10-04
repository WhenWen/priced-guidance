import itertools
import math
import random
from decimal import Decimal

import pytest

from tech_tree_arena import Idea, IdeaVerdict, Submission
from tech_tree_arena.contract import IDEA_RECOVERY_V1
from tech_tree_arena.contract.scoring import mixture_pass_rate
from tech_tree_arena.replay.mixture_correction import corrected_k, terminal_correction


def _submission(*probabilities: str) -> Submission:
    return Submission(
        tuple(Idea(f"idea-{i}", f"candidate {i}", p) for i, p in enumerate(probabilities))
    )


def _rounds(submission: Submission, table: list[list[bool]]):
    return tuple(
        tuple(
            IdeaVerdict(idea.idea_id, passed)
            for idea, passed in zip(submission.ideas, row, strict=True)
        )
        for row in table
    )


def _recorded_score(submission: Submission, rounds) -> float:
    outcome, _ = IDEA_RECOVERY_V1.score_repeated_submission(submission, rounds, path_k=0.0)
    return 0.0 if outcome.status == "fail" else 2.0 ** (-outcome.k)


def test_single_idea_matches_recorded_repeat_pricing() -> None:
    submission = _submission("1")
    rounds = _rounds(submission, [[False], [True], [False]])

    assert mixture_pass_rate(submission, rounds) == Decimal(1) / Decimal(3)
    assert float(mixture_pass_rate(submission, rounds)) == pytest.approx(
        _recorded_score(submission, rounds)
    )


def test_single_round_matches_recorded_passing_mass() -> None:
    submission = _submission("0.25", "0.75")
    rounds = _rounds(submission, [[False, True]])

    assert mixture_pass_rate(submission, rounds) == Decimal("0.75")
    assert float(mixture_pass_rate(submission, rounds)) == pytest.approx(
        _recorded_score(submission, rounds)
    )


def test_ideas_passing_in_different_rounds_are_not_double_counted() -> None:
    submission = _submission("0.5", "0.5")
    rounds = _rounds(submission, [[True, False], [False, True], [False, False]])

    assert mixture_pass_rate(submission, rounds) == Decimal(1) / Decimal(3)
    assert _recorded_score(submission, rounds) == pytest.approx(2 / 3)


def test_mixture_never_exceeds_recorded_score() -> None:
    rng = random.Random(0)
    for ideas, repeats in itertools.product((1, 2, 3, 8), (1, 2, 3)):
        for _ in range(50):
            weights = [rng.randint(1, 9) for _ in range(ideas)]
            submission = _submission(*(str(Decimal(w) / sum(weights)) for w in weights))
            table = [[rng.random() < 0.4 for _ in range(ideas)] for _ in range(repeats)]
            rounds = _rounds(submission, table)
            mixture = float(mixture_pass_rate(submission, rounds))
            assert mixture <= _recorded_score(submission, rounds) + 1e-12


def _events(submission: Submission, rounds, *, path_k: float):
    outcome, _ = IDEA_RECOVERY_V1.score_repeated_submission(submission, rounds, path_k=path_k)
    return [
        {"kind": "submission", "attempt": 1, "submission": {"ideas": [
            {"idea_id": idea.idea_id, "probability": idea.probability, "content": {}}
            for idea in submission.ideas
        ]}},
        {
            "kind": "submission_judged",
            "attempt": 1,
            "status": outcome.status,
            "path_k": path_k,
            "passing_mass": str(outcome.passing_mass),
            "submission_bits": outcome.submission_bits,
            "repeat_bits": outcome.repeat_bits,
            "judge_pass_rate": str(outcome.judge_pass_rate),
            "verdict_rounds": [
                [{"idea_id": v.idea_id, "passed": v.passed} for v in verdicts]
                for verdicts in rounds
            ],
        },
    ], outcome


def test_terminal_correction_from_recorded_events() -> None:
    submission = _submission("0.5", "0.5")
    rounds = _rounds(submission, [[True, False], [False, True], [False, False]])
    events, outcome = _events(submission, rounds, path_k=10.0)

    correction = terminal_correction(events)

    assert correction is not None
    assert correction["ideas"] == 2 and correction["judge_repeats"] == 3
    assert correction["idea_passes"] == {"idea-0": 1, "idea-1": 1}
    assert Decimal(correction["mixture_pass_rate"]) == mixture_pass_rate(submission, rounds)
    assert correction["delta_bits"] == pytest.approx(1.0)
    assert corrected_k(outcome.k, correction) == pytest.approx(10.0 + math.log2(3))


def test_terminal_correction_is_identity_for_single_round() -> None:
    submission = _submission("0.25", "0.75")
    rounds = _rounds(submission, [[True, True]])
    events, outcome = _events(submission, rounds, path_k=4.0)

    correction = terminal_correction(events)

    assert correction["delta_bits"] == pytest.approx(0.0)
    assert corrected_k(outcome.k, correction) == pytest.approx(outcome.k)


def test_terminal_correction_ignores_runs_without_a_pass() -> None:
    submission = _submission("1")
    events, _ = _events(submission, _rounds(submission, [[False]]), path_k=1.0)

    assert terminal_correction(events) is None


def test_corrected_k_rejects_inconsistent_recorded_cost() -> None:
    submission = _submission("1")
    events, _ = _events(submission, _rounds(submission, [[True]]), path_k=1.0)

    with pytest.raises(ValueError):
        corrected_k(5.0, terminal_correction(events))
