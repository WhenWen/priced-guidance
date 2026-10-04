from pathlib import Path

import pytest

from tech_tree_arena.runtime.agent_cli import AgentCLIBackend
from tech_tree_arena.runtime.services import ServiceEvent, ServiceFactory
from tech_tree_arena.replay.recorder import RunRecorder, verify_hash_chain


_SCHEMA = {
    "type": "object",
    "properties": {
        "reasoning": {"type": "string"},
        "state_summary": {"type": "string"},
        "action": {"type": "string"},
        "option_id": {"type": ["string", "null"]},
        "question_id": {"type": ["string", "null"]},
    },
    "required": ["reasoning", "state_summary", "action", "option_id", "question_id"],
    "additionalProperties": False,
}

_ACTION = {
    "reasoning": "The first option is the cheapest faithful answer.",
    "state_summary": "One faithful answer has been communicated.",
    "action": "choose",
    "option_id": "a",
    "question_id": None,
}


def _fake_cli(tmp_path: Path) -> Path:
    script = tmp_path / "fake-agent-cli"
    script.write_text(
        """#!/usr/bin/env python3
import json
import pathlib
import sys

ACTION = {
    "reasoning": "The first option is the cheapest faithful answer.",
    "state_summary": "One faithful answer has been communicated.",
    "action": "choose",
    "option_id": "a",
    "question_id": None,
}

args = sys.argv[1:]
if "--version" in args:
    print("fake-agent-cli 1.0")
    raise SystemExit(0)
_ = sys.stdin.read()

if "-p" in args:
    if "--resume" in args:
        session_id = args[args.index("--resume") + 1]
    else:
        session_id = args[args.index("--session-id") + 1]
    print(json.dumps({
        "type": "assistant",
        "session_id": session_id,
        "message": {"content": [{"type": "thinking", "thinking": "claude exposed reasoning"}]},
    }))
    print(json.dumps({
        "type": "result",
        "session_id": session_id,
        "structured_output": ACTION,
        "usage": {"input_tokens": 11, "output_tokens": 7, "cache_read_input_tokens": 3},
        "total_cost_usd": 0.02,
        "debug_args": args,
    }))
    raise SystemExit(0)

if "fork" in args:
    print(json.dumps({
        "type": "thread.started",
        "thread_id": "codex-session-fork-1",
        "debug_args": args,
    }))
    raise SystemExit(0)

is_resume = "resume" in args
session_id = "codex-session-1" if not is_resume else args[-2]
output_path = pathlib.Path(args[args.index("--output-last-message") + 1])
output_path.write_text(json.dumps(ACTION), encoding="utf-8")
print(json.dumps({"type": "thread.started", "thread_id": session_id, "debug_args": args}))
print(json.dumps({
    "type": "item.completed",
    "item": {
        "type": "reasoning",
        "content": [{"type": "reasoning_text", "text": "codex exposed reasoning"}],
    },
}))
print(json.dumps({
    "type": "item.completed",
    "item": {"type": "agent_message", "text": json.dumps(ACTION)},
}))
print(json.dumps({
    "type": "turn.completed",
    "usage": (
        {"input_tokens": 21, "cached_input_tokens": 8, "cache_write_input_tokens": 4,
         "output_tokens": 15}
        if is_resume else
        {"input_tokens": 13, "cached_input_tokens": 5, "cache_write_input_tokens": 2,
         "output_tokens": 9}
    ),
}))
""",
        encoding="utf-8",
    )
    script.chmod(0o700)
    return script


def test_claude_code_backend_resumes_and_preserves_reasoning(tmp_path: Path) -> None:
    executable = _fake_cli(tmp_path)
    backend = AgentCLIBackend(
        "claude-code",
        working_directory=tmp_path / "claude-workspace",
        executable=str(executable),
        model="opus",
    )

    first = backend.turn(
        system_prompt="oracle system",
        user="first event",
        schema=_SCHEMA,
        schema_name="action",
    )
    second = backend.turn(
        system_prompt="oracle system",
        user="second event",
        schema=_SCHEMA,
        schema_name="action",
        session_id=first["session_id"],
    )

    assert first["output"] == _ACTION
    assert first["reasoning"] == [
        {"source": "structured_output", "text": _ACTION["reasoning"]},
        {"source": "thinking", "text": "claude exposed reasoning"},
    ]
    result_event = next(event for event in second["raw_events"] if event["type"] == "result")
    assert result_event["debug_args"][result_event["debug_args"].index("--resume") + 1] == first[
        "session_id"
    ]
    assert backend.usage_totals() == {
        "input_tokens": 22,
        "cached_input_tokens": 6,
        "cache_write_input_tokens": 0,
        "cache_write_5m_input_tokens": 0,
        "cache_write_1h_input_tokens": 0,
        "output_tokens": 14,
        "cost_usd": 0.04,
        "turns": 2,
    }


def test_codex_backend_resumes_and_preserves_reasoning(tmp_path: Path) -> None:
    executable = _fake_cli(tmp_path)
    backend = AgentCLIBackend(
        "codex",
        working_directory=tmp_path / "codex-workspace",
        executable=str(executable),
        model="gpt-5.6-sol",
    )

    first = backend.turn(
        system_prompt="oracle developer instructions",
        user="first event",
        schema=_SCHEMA,
        schema_name="action",
    )
    second = backend.turn(
        system_prompt="oracle developer instructions",
        user="second event",
        schema=_SCHEMA,
        schema_name="action",
        session_id=first["session_id"],
    )

    assert first["session_id"] == "codex-session-1"
    assert first["output"] == _ACTION
    assert first["reasoning"] == [
        {"source": "structured_output", "text": _ACTION["reasoning"]},
        {"source": "reasoning_text", "text": "codex exposed reasoning"},
    ]
    started = next(event for event in second["raw_events"] if event["type"] == "thread.started")
    assert "resume" in started["debug_args"]
    assert first["session_id"] in started["debug_args"]
    assert second["usage"] == {
        "input_tokens": 8,
        "cached_input_tokens": 3,
        "cache_write_input_tokens": 2,
        "cache_write_5m_input_tokens": 0,
        "cache_write_1h_input_tokens": 0,
        "output_tokens": 6,
        "cost_usd": 0.0,
        "turns": 1,
    }
    assert second["cumulative_usage"] == {
        "input_tokens": 21,
        "cached_input_tokens": 8,
        "cache_write_input_tokens": 4,
        "output_tokens": 15,
        "turns": 1,
    }
    assert backend.usage_totals() == {
        "input_tokens": 21,
        "cached_input_tokens": 8,
        "cache_write_input_tokens": 4,
        "cache_write_5m_input_tokens": 0,
        "cache_write_1h_input_tokens": 0,
        "output_tokens": 15,
        "cost_usd": 0.0,
        "turns": 2,
    }


def test_codex_usage_snapshot_survives_backend_restore(tmp_path: Path) -> None:
    executable = _fake_cli(tmp_path)
    first_backend = AgentCLIBackend(
        "codex",
        working_directory=tmp_path / "first-workspace",
        executable=str(executable),
    )
    first = first_backend.turn(
        system_prompt="oracle developer instructions",
        user="first event",
        schema=_SCHEMA,
        schema_name="action",
    )

    resumed_backend = AgentCLIBackend(
        "codex",
        working_directory=tmp_path / "resumed-workspace",
        executable=str(executable),
    )
    resumed_backend.restore_state(first_backend.export_state())
    second = resumed_backend.turn(
        system_prompt="oracle developer instructions",
        user="second event",
        schema=_SCHEMA,
        schema_name="action",
        session_id=first["session_id"],
    )

    assert second["usage"]["input_tokens"] == 8
    assert resumed_backend.usage_totals()["input_tokens"] == 21


def test_codex_inherited_session_is_forked_before_first_write(tmp_path: Path) -> None:
    executable = _fake_cli(tmp_path)
    first_backend = AgentCLIBackend(
        "codex",
        working_directory=tmp_path / "first-workspace",
        executable=str(executable),
    )
    first = first_backend.turn(
        system_prompt="oracle developer instructions",
        user="first event",
        schema=_SCHEMA,
        schema_name="action",
    )

    resumed_backend = AgentCLIBackend(
        "codex",
        working_directory=tmp_path / "resumed-workspace",
        executable=str(executable),
        fork_inherited_sessions=True,
    )
    resumed_backend.restore_state(first_backend.export_state())
    second = resumed_backend.turn(
        system_prompt="oracle developer instructions",
        user="second event",
        schema=_SCHEMA,
        schema_name="action",
        session_id=first["session_id"],
    )

    # The write went to the fork, never to the inherited thread.
    started = next(
        event for event in second["raw_events"] if event["type"] == "thread.started"
    )
    assert "resume" in started["debug_args"]
    assert "codex-session-fork-1" in started["debug_args"]
    assert first["session_id"] not in started["debug_args"][started["debug_args"].index("resume"):]
    assert second["session_id"] == "codex-session-fork-1"
    # Usage delta stays a delta: the fork inherits the source thread baseline.
    assert second["usage"]["input_tokens"] == 8

    # A session created by this backend is not forked.
    third = resumed_backend.turn(
        system_prompt="oracle developer instructions",
        user="third event",
        schema=_SCHEMA,
        schema_name="action",
        session_id=second["session_id"],
    )
    third_started = next(
        event for event in third["raw_events"] if event["type"] == "thread.started"
    )
    assert "resume" in third_started["debug_args"]
    assert "codex-session-fork-1" in third_started["debug_args"]
    assert "fork" not in third_started["debug_args"]


def test_agent_backend_rejects_an_observed_tool_call(tmp_path: Path) -> None:
    executable = tmp_path / "fake-tool-using-cli"
    executable.write_text(
        """#!/usr/bin/env python3
import json
import sys

if "--version" in sys.argv:
    print("fake-tool-using-cli 1.0")
    raise SystemExit(0)
_ = sys.stdin.read()
print(json.dumps({"type": "thread.started", "thread_id": "bad-session"}))
print(json.dumps({
    "type": "item.started",
    "item": {"type": "command_execution", "command": "pwd"},
}))
""",
        encoding="utf-8",
    )
    executable.chmod(0o700)
    backend = AgentCLIBackend(
        "codex",
        working_directory=tmp_path / "tool-workspace",
        executable=str(executable),
    )

    with pytest.raises(RuntimeError, match="forbidden tool use"):
        backend.turn(
            system_prompt="oracle developer instructions",
            user="arena event",
            schema=_SCHEMA,
            schema_name="action",
        )

    metadata = backend.last_call_metadata()
    assert metadata["tool_policy_violation"] == [
        {"type": "command_execution", "name": "pwd"}
    ]


def test_agent_turn_and_reasoning_are_replayable_without_a_live_backend() -> None:
    class Backend:
        def __init__(self) -> None:
            self.calls = 0
            self.usage = {"input_tokens": 0, "output_tokens": 0, "cost_usd": 0.0}

        def turn(self, **_request):
            self.calls += 1
            self.usage = {"input_tokens": 5, "output_tokens": 3, "cost_usd": 0.01}
            return {
                "session_id": "session-1",
                "output": _ACTION,
                "reasoning": [{"source": "reasoning", "text": "preserved reasoning"}],
                "raw_events": [{"type": "reasoning", "text": "preserved reasoning"}],
                "usage": dict(self.usage),
            }

        def usage_totals(self):
            return dict(self.usage)

        def last_call_metadata(self):
            return {"usage": dict(self.usage), "session_id": "session-1"}

    request = {
        "system_prompt": "oracle system",
        "user": "arena event",
        "schema": _SCHEMA,
        "schema_name": "action",
        "session_id": None,
    }
    backend = Backend()
    original = ServiceFactory(seed=9, agent_backend=backend).create()
    expected = original.agent_turn(**request)
    event = original.export_tape()[0]

    replay = ServiceFactory(seed=9).create((event,))
    actual = replay.agent_turn(**request)

    assert backend.calls == 1
    assert event.kind == "agent.session_turn"
    assert event.response["reasoning"][0]["text"] == "preserved reasoning"
    assert event.response["raw_events"][0]["type"] == "reasoning"
    assert actual == expected
    assert replay.replay_remaining() == 0


def test_run_recorder_writes_a_dedicated_private_reasoning_transcript(tmp_path: Path) -> None:
    recorder = RunRecorder(tmp_path, "reasoning-run", {})
    event = ServiceEvent(
        "agent.session_turn",
        "hash",
        {
            "session_id": "session-1",
            "backend": "codex",
            "model": "gpt-5.6-sol",
            "cli_version": "codex-cli test",
            "output": _ACTION,
            "reasoning": [{"source": "reasoning", "text": "saved reasoning"}],
            "raw_events": [{"type": "reasoning", "text": "raw saved reasoning"}],
            "usage": {"input_tokens": 2, "output_tokens": 1},
        },
        request={
            "system_prompt": "system",
            "user": "arena event",
            "schema": _SCHEMA,
            "schema_name": "action",
            "session_id": None,
        },
    )

    recorder.record_service("oracle", event)

    records = verify_hash_chain(
        recorder.root / "oracle-reasoning.private.jsonl"
    )
    assert len(records) == 1
    assert records[0]["session_id"] == "session-1"
    assert records[0]["user"] == "arena event"
    assert records[0]["reasoning"][0]["text"] == "saved reasoning"
    assert records[0]["raw_events"][0]["text"] == "raw saved reasoning"


def test_codex_anthropic_proxy_injects_provider_and_prices_usage(tmp_path: Path) -> None:
    executable = _fake_cli(tmp_path)
    backend = AgentCLIBackend(
        "codex",
        working_directory=tmp_path / "codex-proxy-workspace",
        executable=str(executable),
        model="claude-fable-5",
        anthropic_proxy_base_url="http://127.0.0.1:50704",
    )

    first = backend.turn(
        system_prompt="oracle developer instructions",
        user="first event",
        schema=_SCHEMA,
        schema_name="action",
    )

    started = next(
        event for event in first["raw_events"] if event["type"] == "thread.started"
    )
    joined = " ".join(started["debug_args"])
    assert 'model_providers.arena_anthropic.base_url="http://127.0.0.1:50704"' in joined
    assert 'model_providers.arena_anthropic.env_key="ANTHROPIC_API_KEY"' in joined
    assert 'model_providers.arena_anthropic.wire_api="responses"' in joined
    assert 'model_provider="arena_anthropic"' in joined
    assert "claude-fable-5" in started["debug_args"]

    # The adapter's tokens are real Anthropic spend, priced against Fable 5's
    # published sheet ($/MTok): uncached input $10, cache read $1, 5m cache
    # write $12.50, 1h cache write $20, output $50.  Fake first turn: input 13
    # (cached 5, write 2 -> uncached 6), output 9.  The CLI relays only the flat
    # write total, so those 2 tokens must be charged at the 1h rate the adapter
    # configures -- not silently dropped, and not discounted to the 5m rate.
    expected = (6 * 10.00 + 5 * 1.00 + 2 * 20.00 + 9 * 50.00) / 1e6
    assert first["usage"]["cost_usd"] == pytest.approx(expected)
    assert backend.usage_totals()["cost_usd"] == pytest.approx(expected)
    # Pricing the cache-inclusive total at the base input rate -- the convention
    # this replaced -- overstates a cache-read-dominated session.
    flat = (13 * 10.00 + 9 * 50.00) / 1e6
    assert flat > expected


def test_anthropic_proxy_is_codex_only(tmp_path: Path) -> None:
    executable = _fake_cli(tmp_path)
    with pytest.raises(ValueError, match="codex backend only"):
        AgentCLIBackend(
            "claude-code",
            working_directory=tmp_path / "claude-proxy-workspace",
            executable=str(executable),
            model="claude-fable-5",
            anthropic_proxy_base_url="http://127.0.0.1:50704",
        )


def test_failed_agent_turn_never_reuses_the_previous_turns_usage(tmp_path: Path) -> None:
    executable = tmp_path / "fake-flaky-cli"
    executable.write_text(
        """#!/usr/bin/env python3
import json
import pathlib
import sys

args = sys.argv[1:]
if "--version" in args:
    print("fake-flaky-cli 1.0")
    raise SystemExit(0)
_ = sys.stdin.read()
if "resume" in args:
    raise SystemExit(9)
output_path = pathlib.Path(args[args.index("--output-last-message") + 1])
output_path.write_text(json.dumps({"ok": True}), encoding="utf-8")
print(json.dumps({"type": "thread.started", "thread_id": "flaky-1"}))
print(json.dumps({"type": "item.completed",
                  "item": {"type": "agent_message", "text": json.dumps({"ok": True})}}))
print(json.dumps({"type": "turn.completed",
                  "usage": {"input_tokens": 40, "cached_input_tokens": 10,
                            "cache_write_input_tokens": 5, "output_tokens": 7}}))
""",
        encoding="utf-8",
    )
    executable.chmod(0o700)
    backend = AgentCLIBackend(
        "codex",
        working_directory=tmp_path / "flaky-workspace",
        executable=str(executable),
        model="claude-fable-5",
        anthropic_proxy_base_url="http://127.0.0.1:1",
    )
    schema = {"type": "object", "properties": {"ok": {"type": "boolean"}},
              "required": ["ok"], "additionalProperties": False}

    first = backend.turn(
        system_prompt="oracle developer instructions",
        user="first event",
        schema=schema,
        schema_name="action",
    )
    assert first["usage"]["input_tokens"] == 40
    totals_after_first = backend.usage_totals()

    with pytest.raises(RuntimeError):
        backend.turn(
            system_prompt="oracle developer instructions",
            user="second event",
            schema=schema,
            schema_name="action",
            session_id=first["session_id"],
        )

    metadata = backend.last_call_metadata()
    # The failed turn reports zero usage instead of replaying turn 1's delta.
    assert metadata.get("usage") == {}
    assert metadata.get("error_type")
    assert backend.usage_totals() == totals_after_first
