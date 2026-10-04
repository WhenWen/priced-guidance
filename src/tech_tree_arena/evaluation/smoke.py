"""Offline-only evaluation probe."""

from __future__ import annotations

from typing import Any

from ..contract.messages import Idea
from .base import FunctionJudge


class SmokeAnswerJudge(FunctionJudge):
    """Deterministic wiring probe for targets with an ``answer`` field."""

    def __init__(self) -> None:
        super().__init__(self._matches)

    @staticmethod
    def _matches(target: Any, idea: Idea) -> bool:
        expected = target.get("answer") if isinstance(target, dict) else None
        content = idea.content.get("answer") if isinstance(idea.content, dict) else idea.content
        return content == expected
