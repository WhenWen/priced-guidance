"""Trusted per-submission judge interfaces."""

from __future__ import annotations

import json
from typing import Any, Callable, Protocol

from ..contract.messages import Idea, IdeaVerdict
from ..errors import ProtocolError

__all__ = [
    "FunctionJudge",
    "IdeaVerdict",
    "Judge",
    "aggregate_repeated_verdicts",
]


class Judge(Protocol):
    def evaluate(self, target: Any, ideas: tuple[Idea, ...]) -> tuple[IdeaVerdict, ...]: ...


def aggregate_repeated_verdicts(
    ideas: tuple[Idea, ...],
    verdict_rounds: tuple[tuple[IdeaVerdict, ...], ...],
) -> tuple[IdeaVerdict, ...]:
    """Collapse repeated independent Judge rounds with an any-pass rule.

    A single round is returned byte-for-byte so Judges and historical runs
    that do not opt into repetition keep their existing private diagnostics.
    For repeated rounds, every idea records the ordered boolean outcomes and
    reasons in an Arena-authored JSON diagnostic.
    """

    if not verdict_rounds:
        raise ProtocolError("repeat Judge must produce at least one verdict round")
    idea_ids = tuple(idea.idea_id for idea in ideas)
    normalized: list[tuple[IdeaVerdict, ...]] = []
    for verdicts in verdict_rounds:
        verdicts = tuple(verdicts)
        if (
            len(verdicts) != len(idea_ids)
            or any(not isinstance(verdict, IdeaVerdict) for verdict in verdicts)
            or any(type(verdict.passed) is not bool for verdict in verdicts)
            or len({verdict.idea_id for verdict in verdicts}) != len(verdicts)
            or {verdict.idea_id for verdict in verdicts} != set(idea_ids)
        ):
            raise ProtocolError("judge returned an invalid verdict set")
        normalized.append(verdicts)
    if len(normalized) == 1:
        return normalized[0]

    by_round = [
        {verdict.idea_id: verdict for verdict in verdicts}
        for verdicts in normalized
    ]
    aggregated: list[IdeaVerdict] = []
    for idea_id in idea_ids:
        values = [round_by_id[idea_id] for round_by_id in by_round]
        pass_count = sum(verdict.passed for verdict in values)
        reason = json.dumps(
            {
                "judge_passes": pass_count,
                "judge_repeats": len(values),
                "rounds": [
                    {
                        "passed": verdict.passed,
                        "private_reason": verdict.private_reason,
                    }
                    for verdict in values
                ],
            },
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        aggregated.append(IdeaVerdict(idea_id, pass_count >= 1, reason))
    return tuple(aggregated)


class FunctionJudge:
    def __init__(self, predicate: Callable[[Any, Idea], bool]) -> None:
        self.predicate = predicate

    def evaluate(self, target: Any, ideas: tuple[Idea, ...]) -> tuple[IdeaVerdict, ...]:
        return tuple(
            IdeaVerdict(idea.idea_id, bool(self.predicate(target, idea))) for idea in ideas
        )
