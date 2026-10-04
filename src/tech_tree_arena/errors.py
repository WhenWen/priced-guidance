"""Arena error taxonomy.

Participant-facing callers receive stable error codes rather than implementation
exceptions.  Detailed causes remain in the private audit log.
"""

from __future__ import annotations


class ArenaError(Exception):
    """Base class for trusted arena failures."""

    code = "arena_error"


class ValidationError(ArenaError):
    code = "validation_error"


class ProtocolError(ArenaError):
    code = "protocol_error"


class InvalidChoice(ProtocolError):
    code = "invalid_choice"


class InvalidCheckout(ProtocolError):
    code = "invalid_checkout"


class ResourceLimitExceeded(ArenaError):
    code = "resource_limit_exceeded"


class RecordedRunFailure(ArenaError):
    """A failed operation that left a durable, potentially resumable run."""

    def __init__(self, message: str, *, run_dir: str, error_code: str) -> None:
        super().__init__(f"{message}; run_dir={run_dir}")
        self.run_dir = run_dir
        self.code = error_code


class ReplayDivergence(ArenaError):
    code = "replay_divergence"


class ParticipantFailure(ArenaError):
    code = "participant_failure"

    def __init__(
        self,
        message: str,
        *,
        phase: str | None = None,
        error_type: str | None = None,
        private_detail: str | None = None,
    ) -> None:
        super().__init__(message)
        self.phase = phase
        self.error_type = error_type
        self.private_detail = private_detail
