"""Non-semantic terminal judge for persisted idea sampling runs."""

from __future__ import annotations

from typing import Any

from ..contract.messages import Idea, IdeaVerdict


class SampleSubmissionJudge:
    """Accept the first protocol-authorized Submission without evaluating it."""

    def evaluate(self, target: Any, ideas: tuple[Idea, ...]) -> tuple[IdeaVerdict, ...]:
        return tuple(IdeaVerdict(idea.idea_id, True) for idea in ideas)
