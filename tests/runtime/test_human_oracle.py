"""HumanOracleBackend: file-driven turn protocol and usage inheritance."""

from __future__ import annotations

import json
import threading
from pathlib import Path

import pytest

from tech_tree_arena.cli import (
    _make_oracle_agent_backend,
    _oracle_agent_config,
    _run_submission,
)
from tech_tree_arena.errors import ArenaError
from tech_tree_arena.runtime.human_oracle import HumanOracleBackend


def _answer_when_request_appears(workspace, turn, payload):
    request = workspace / f"turn-{turn:04d}.request.md"
    answer = workspace / f"turn-{turn:04d}.answer.json"

    def writer() -> None:
        while not request.exists():
            pass
        answer.write_text(json.dumps(payload), encoding="utf-8")

    thread = threading.Thread(target=writer, daemon=True)
    thread.start()
    return thread


def test_turn_round_trip_writes_request_and_reads_answer(tmp_path):
    backend = HumanOracleBackend(
        working_directory=tmp_path, timeout_seconds=30.0, poll_seconds=0.01
    )
    thread = _answer_when_request_appears(
        tmp_path,
        1,
        {"action": "choose", "option_id": "mc-3", "reasoning": "obvious"},
    )
    response = backend.turn(
        system_prompt="SYSTEM RULES",
        user="pick one",
        schema={"type": "object"},
        schema_name="oracle_action",
        session_id=None,
    )
    thread.join(timeout=5)
    assert response["session_id"] == "human"
    assert response["output"]["action"] == "choose"
    assert response["output"]["option_id"] == "mc-3"
    assert response["output"]["reasoning"] == "obvious"
    assert response["output"]["state_summary"]  # placeholder filled in
    assert response["usage"]["cost_usd"] == 0.0
    request_text = (tmp_path / "turn-0001.request.md").read_text(encoding="utf-8")
    assert "SYSTEM RULES" in request_text
    assert "pick one" in request_text
    assert backend.usage_totals()["turns"] == 1
    assert backend.last_call_metadata()["agent_backend"] == "human"


def test_system_prompt_shown_once_even_with_inherited_session(tmp_path):
    backend = HumanOracleBackend(
        working_directory=tmp_path, timeout_seconds=30.0, poll_seconds=0.01
    )
    thread = _answer_when_request_appears(tmp_path, 1, {"action": "choose", "option_id": "a"})
    backend.turn(
        system_prompt="SYSTEM RULES",
        user="first",
        schema={},
        schema_name="oracle_action",
        session_id="codex-inherited-session",
    )
    thread.join(timeout=5)
    assert "SYSTEM RULES" in (tmp_path / "turn-0001.request.md").read_text(encoding="utf-8")
    thread = _answer_when_request_appears(tmp_path, 2, {"action": "choose", "option_id": "b"})
    backend.turn(
        system_prompt="SYSTEM RULES",
        user="second",
        schema={},
        schema_name="oracle_action",
        session_id="human",
    )
    thread.join(timeout=5)
    assert "SYSTEM RULES" not in (tmp_path / "turn-0002.request.md").read_text(encoding="utf-8")


def test_restore_usage_keeps_inherited_spend(tmp_path):
    backend = HumanOracleBackend(working_directory=tmp_path, poll_seconds=0.01)
    backend.restore_state(
        {"usage": {"input_tokens": 1000, "output_tokens": 50, "cost_usd": 12.5, "turns": 7}}
    )
    totals = backend.usage_totals()
    assert totals["input_tokens"] == 1000
    assert totals["cost_usd"] == 12.5
    assert totals["turns"] == 7
    thread = _answer_when_request_appears(tmp_path, 8, {"action": "checkout", "question_id": "q1"})
    backend.turn(
        system_prompt="s", user="u", schema={}, schema_name="oracle_action", session_id="x"
    )
    thread.join(timeout=5)
    totals = backend.usage_totals()
    assert totals["turns"] == 8
    assert totals["cost_usd"] == 12.5  # human turns add zero spend


def test_timeout_raises(tmp_path):
    backend = HumanOracleBackend(
        working_directory=tmp_path, timeout_seconds=0.05, poll_seconds=0.01
    )
    with pytest.raises(TimeoutError):
        backend.turn(
            system_prompt="s", user="u", schema={}, schema_name="oracle_action", session_id=None
        )


def test_cli_config_builds_human_backend(tmp_path):
    config = _oracle_agent_config(
        "human",
        model=None,
        executable=None,
        reasoning_effort="high",
        timeout_seconds=600.0,
        max_budget_usd_per_turn=None,
    )
    assert config == {
        "backend": "human",
        "model": "human",
        "executable": None,
        "reasoning_effort": "high",
        "timeout_seconds": 86_400.0,
        "max_budget_usd_per_turn": None,
    }
    backend = _make_oracle_agent_backend(config, tmp_path / "oracle")
    assert isinstance(backend, HumanOracleBackend)
    assert backend.working_directory == (tmp_path / "oracle" / "human").resolve()


def test_oracle_agent_fails_fast_on_a_submission_without_an_agent_oracle(tmp_path):
    """Selecting an agent backend a pair cannot use must error, not silently
    fall back to the ordinary model Oracle (minimal_pair defines no
    AgentOracle)."""

    root = Path(__file__).resolve().parents[2]
    with pytest.raises(ArenaError, match="AgentOracle"):
        _run_submission(
            root / "submissions" / "examples" / "minimal_pair",
            "smoke",
            None,
            1,
            runs_dir=tmp_path,
            oracle_agent="human",
        )
