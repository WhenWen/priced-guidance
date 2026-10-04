from typing import Any

from participant.pair import _GUIDE_ACTION_SCHEMA, _GUIDE_SYSTEM_PROMPT, Guide


class AgentGuide(Guide):
    """Oracle whose decisions run on the evaluation-profile agent backend.

    Selected by the Arena when a run passes ``--oracle-agent`` (claude-code,
    codex, or human): the CLI swaps the oracle entrypoint to this class and
    injects the matching agent-session backend. Every decision carries the
    same full-context prompt as the base Oracle -- MATCH_INIT verbatim, the
    Oracle's own last state_summary, and the current event -- but travels
    through ``services.agent_turn`` instead of ``structured_model``.

    Each turn runs as a fresh backend session (``session_id=None``). That
    keeps the v1.11 transport property -- nothing about the Oracle's context
    is decided by an auto-compacting CLI conversation -- while a human
    backend receives one self-contained request file per turn.
    """

    def _decide(self, user: str) -> Any:
        response = self.services.agent_turn(
            system_prompt=_GUIDE_SYSTEM_PROMPT,
            user=user,
            schema=_GUIDE_ACTION_SCHEMA,
            schema_name="oracle_action",
            session_id=None,
        )
        if not isinstance(response, dict):
            raise TypeError("agent backend returned no turn object")
        return response.get("output")


__all__ = ["AgentOracle", "Oracle"]


# Legacy Python names remain available for existing submissions.
AgentOracle = AgentGuide

# Legacy imported names.
_ORACLE_ACTION_SCHEMA = _GUIDE_ACTION_SCHEMA
_ORACLE_SYSTEM_PROMPT = _GUIDE_SYSTEM_PROMPT
Oracle = Guide

__all__ += ['AgentGuide', 'Guide']
