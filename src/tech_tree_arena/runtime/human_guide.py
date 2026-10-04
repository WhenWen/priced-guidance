"""A file-driven Oracle backend played by a human.

Each turn writes the full oracle-visible text to
``<workspace>/turn-NNNN.request.md`` and then blocks, polling for
``<workspace>/turn-NNNN.answer.json``. The answer file must contain the
ordinary five-field oracle action::

    {"action": "choose", "option_id": "mc-3"}
    {"action": "checkout", "question_id": "q00000004.abcd..."}

``reasoning`` and ``state_summary`` may be included; when omitted they are
filled with explicit human-oracle placeholders so the participant's
non-empty-field validation passes. Usage is recorded as zero: a human turn
spends no model tokens. Journals, checkpoints, and replay treat the turn
exactly like any other agent turn.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any


class HumanGuideBackend:
    """AgentSessionBackend implementation backed by a person and two files."""

    def __init__(
        self,
        *,
        working_directory: str | Path,
        timeout_seconds: float = 86_400.0,
        poll_seconds: float = 2.0,
    ) -> None:
        if timeout_seconds <= 0:
            raise ValueError("human oracle timeout must be positive")
        self.working_directory = Path(working_directory).resolve()
        self.working_directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.timeout_seconds = float(timeout_seconds)
        self.poll_seconds = float(poll_seconds)
        self.turn_index = 0
        self._system_prompt_shown = False
        self._last_metadata: dict[str, Any] = {}
        # Cumulative totals must survive a backend swap: on a promotion fork
        # the checkpoint carries the source oracle's (codex) real spend, and
        # the resume validator reconciles these totals against the service
        # journal. A human turn adds zero tokens on top of the inherited base.
        self._usage: dict[str, Any] = {
            "input_tokens": 0,
            "cached_input_tokens": 0,
            "cache_write_input_tokens": 0,
            "cache_write_5m_input_tokens": 0,
            "cache_write_1h_input_tokens": 0,
            "output_tokens": 0,
            "cost_usd": 0.0,
            "turns": 0,
        }

    # -- bookkeeping interface shared with AgentCLIBackend -------------------
    def last_call_metadata(self) -> dict[str, Any]:
        return dict(self._last_metadata)

    def usage_totals(self) -> dict[str, Any]:
        return dict(self._usage)

    def restore_usage(self, usage: dict[str, Any]) -> None:
        for key in self._usage:
            if key in usage:
                self._usage[key] = (
                    float(usage[key]) if key == "cost_usd" else int(usage[key])
                )
        self.turn_index = int(self._usage.get("turns", 0) or 0)

    def export_state(self) -> dict[str, Any]:
        return {"usage": self.usage_totals()}

    def restore_state(self, state: dict[str, Any]) -> None:
        usage = state.get("usage")
        if isinstance(usage, dict):
            self.restore_usage(usage)

    # -- the turn -------------------------------------------------------------
    def turn(
        self,
        *,
        system_prompt: str,
        user: str,
        schema: dict[str, Any],
        schema_name: str,
        session_id: str | None = None,
    ) -> dict[str, Any]:
        if not isinstance(user, str) or not user.strip():
            raise ValueError("human oracle turn requires a non-empty user message")
        self.turn_index += 1
        started = time.monotonic()
        request_path = self.working_directory / f"turn-{self.turn_index:04d}.request.md"
        answer_path = self.working_directory / f"turn-{self.turn_index:04d}.answer.json"
        answer_path.unlink(missing_ok=True)
        body = ""
        if session_id is None or not self._system_prompt_shown:
            # On a promotion/resume fork the participant carries a prior
            # (codex) session id; the human still needs the system prompt
            # once. Earlier history lives in the source run's journals.
            body += (
                "# SYSTEM PROMPT (shown on this backend's first turn)\n\n"
                + system_prompt
                + "\n\n"
            )
            self._system_prompt_shown = True
        body += f"# TURN {self.turn_index}\n\n{user}\n"
        body += (
            f"\n\n---\nAnswer by writing {answer_path.name} next to this file, e.g.\n"
            '{"action": "choose", "option_id": "<exact id>"} or\n'
            '{"action": "checkout", "question_id": "<exact eligible id>"}\n'
        )
        request_path.write_text(body, encoding="utf-8")

        deadline = started + self.timeout_seconds
        while True:
            if answer_path.exists():
                try:
                    raw = answer_path.read_text(encoding="utf-8").strip()
                    if raw:
                        answer = json.loads(raw)
                        if isinstance(answer, dict) and answer.get("action"):
                            break
                except (OSError, json.JSONDecodeError):
                    pass  # partially written; keep polling
            if time.monotonic() > deadline:
                raise TimeoutError(
                    f"human oracle gave no answer for turn {self.turn_index} "
                    f"within {self.timeout_seconds:.0f}s"
                )
            time.sleep(self.poll_seconds)

        output = {
            "reasoning": str(
                answer.get("reasoning") or "human oracle decision"
            ),
            "state_summary": str(
                answer.get("state_summary") or "human oracle plays this match"
            ),
            "action": str(answer.get("action")),
            "option_id": answer.get("option_id"),
            "question_id": answer.get("question_id"),
        }
        latency = time.monotonic() - started
        self._usage["turns"] = int(self._usage.get("turns", 0)) + 1
        self._last_metadata = {
            "agent_backend": "human",
            "cli_version": "human-oracle",
            "model": "human",
            "latency_s": latency,
            "usage": {
                "input_tokens": 0,
                "cached_input_tokens": 0,
                "cache_write_input_tokens": 0,
                "cache_write_5m_input_tokens": 0,
                "cache_write_1h_input_tokens": 0,
                "output_tokens": 0,
                "cost_usd": 0.0,
                "turns": 1,
            },
        }
        return {
            "session_id": "human",
            "output": output,
            "reasoning": [],
            "raw_events": [],
            "raw_non_json": [],
            "stderr": "",
            "usage": dict(self._last_metadata["usage"]),
        }


# Legacy Python names remain available for existing submissions.
HumanOracleBackend = HumanGuideBackend
