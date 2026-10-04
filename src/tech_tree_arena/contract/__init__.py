"""Public `idea-recovery-v1` contract.

Scoring objects are loaded lazily so importing participant message classes does
not import trusted evaluation or runtime implementation modules.
"""

from importlib import import_module
from typing import Any

PROTOCOL_NAME = "idea-recovery-v1"
PROTOCOL_VERSION = 1

__all__ = [
    "IDEA_RECOVERY_V1",
    "PROTOCOL_NAME",
    "PROTOCOL_VERSION",
    "IdeaRecoveryV1Contract",
    "ResolvedChoice",
    "SubmissionOutcome",
]


def __getattr__(name: str) -> Any:
    if name in {
        "IDEA_RECOVERY_V1",
        "IdeaRecoveryV1Contract",
        "ResolvedChoice",
        "SubmissionOutcome",
    }:
        return getattr(import_module(".scoring", __name__), name)
    raise AttributeError(name)
