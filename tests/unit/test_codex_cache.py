import json

import pytest

from tech_tree_arena.runtime import codex_rpc
from tech_tree_arena.runtime.codex_prompts import DEFAULT_VERSION


def test_cache_classes_charge_only_child_suffix():
    parent = {"usage_total": {"inputTokens": 1000, "cachedInputTokens": 500,
                             "cacheWriteInputTokens": 200, "outputTokens": 100}}
    total = {"inputTokens": 2200, "cachedInputTokens": 1400,
             "cacheWriteInputTokens": 400, "outputTokens": 110}
    usage, result = codex_rpc.usage_delta(
        [{"method": "thread/tokenUsage/updated", "params": {"tokenUsage": {"total": total}}}], parent)
    assert result == total
    assert usage["input_tokens"] == 1200
    assert usage["cached_input_tokens"] == 900
    assert usage["cache_write_input_tokens"] == 200
    assert usage["output_tokens"] == 10
    assert codex_rpc.standard_credit_equivalent(usage) == pytest.approx(0.1225)


@pytest.mark.parametrize("value", ["0", "-1", "nan", "inf", "invalid"])
def test_native_timeout_rejects_invalid_operator_values(monkeypatch, value):
    monkeypatch.setenv("IDEA_ARENA_CODEX_TURN_TIMEOUT_S", value)
    with pytest.raises(ValueError):
        codex_rpc.turn_timeout_seconds()


def test_codex_outer_step_allows_multiple_long_preview_calls(monkeypatch):
    from tech_tree_arena.cli import _participant_actor_timeout
    monkeypatch.delenv("IDEA_ARENA_ACTOR_TIMEOUT_S", raising=False)
    assert _participant_actor_timeout(codex=True) == 6000
    assert _participant_actor_timeout() == 900
    monkeypatch.setenv("IDEA_ARENA_ACTOR_TIMEOUT_S", "7200")
    assert _participant_actor_timeout(codex=True) == 7200


@pytest.mark.parametrize("payload", [
    {"options": []},  # incomplete eager preview
    {"options": [{"label": "left"}, {"label": "right"}], "unrequested": True},
    {"options": [{"label": "left"}, {"label": 7}]},
])
def test_invalid_inner_schema_fails_without_extra_call_but_keeps_usage(tmp_path, monkeypatch, payload):
    calls = []
    class RPC:
        def __init__(self, *args, **kwargs):
            self.events = []
        def send(self, value):
            pass
        def close(self):
            pass
        def call(self, method, params):
            calls.append(method)
            if method == "thread/start":
                return {"model": "gpt-6-astra", "thread": {"id": "child", "path": "synthetic"}}
            if method == "turn/start":
                self.events.extend([
                    {"method": "thread/tokenUsage/updated", "params": {"tokenUsage": {"total": {
                        "inputTokens": 1200, "cachedInputTokens": 900, "outputTokens": 50}}}},
                    {"method": "turn/completed", "params": {"turn": {"id": "t", "status": "completed",
                        "items": [{"type": "agentMessage", "text": json.dumps({"payload_json": json.dumps(payload)})}]}}}])
                return {"turn": {"id": "t"}}
            return {}
    monkeypatch.setattr(codex_rpc, "CodexRPC", RPC)
    schema = {"type": "object", "properties": {"options": {"type": "array", "minItems": 2, "maxItems": 2,
              "items": {"type": "object", "properties": {"label": {"type": "string"}},
                        "required": ["label"], "additionalProperties": False}}},
              "required": ["options"], "additionalProperties": False}
    with pytest.raises(ValueError) as error:
        codex_rpc.run_turn(tmp_path / "codex", tmp_path, {"model": "gpt-6-astra", "developer": "Full preview",
            "user": "Synthetic", "schema": schema}, None, DEFAULT_VERSION, cache_key="b" * 64)
    assert calls.count("turn/start") == 1
    assert error.value.codex_usage["cached_input_tokens"] == 900
    assert error.value.codex_usage["output_tokens"] == 50


@pytest.mark.parametrize("model", ["gpt-6-astra", "gpt-5.6-sol"])
@pytest.mark.parametrize("connected", [False, True])
def test_checkpoint_reconnect_resumes_and_live_continuation_avoids_fork(tmp_path, monkeypatch, connected, model):
    monkeypatch.delenv("IDEA_ARENA_CODEX_TURN_TIMEOUT_S", raising=False)
    clock = [1000.0]
    monkeypatch.setattr(codex_rpc.time, "monotonic", lambda: clock[0])
    calls = []
    class RPC:
        initialized = connected
        events = []
        closed = False
        def send(self, value):
            pass
        def close(self):
            self.closed = True
        def call(self, method, params):
            calls.append((method, params))
            if method == "thread/resume":
                return {"model": model, "thread": {"id": "parent", "path": "synthetic"}}
            if method == "turn/start":
                # A valid native reasoning turn is still alive after the old
                # 300-second cutoff, including a reused RPC from a prior turn.
                clock[0] += 400
                assert self.deadline == 2500
                assert self.deadline > codex_rpc.time.monotonic()
                self.events.extend([
                    {"method": "thread/tokenUsage/updated", "params": {"threadId": "parent", "tokenUsage": {"total": {
                        "inputTokens": 1010, "cachedInputTokens": 500, "outputTokens": 35}}}},
                    {"method": "turn/completed", "params": {"turn": {"id": "new", "status": "completed",
                        "items": [{"type": "agentMessage", "text": '{"payload_json":"{\\"ok\\":true}"}'}]}}}])
                return {"turn": {"id": "new"}}
            return {}
    rpc = RPC()
    parent = {"thread_id": "parent", "turn_id": "old", "usage_total": {"inputTokens": 10, "outputTokens": 20}}
    result = codex_rpc.run_turn(tmp_path / "codex", tmp_path,
        {"model": model, "developer": "Return ok=true", "user": "Continue.", "timeout": 300,
         "schema": {"type": "object", "properties": {"ok": {"type": "boolean"}}, "required": ["ok"]}},
        parent, DEFAULT_VERSION, cache_key="c" * 64, live_rpc=rpc,
        live_thread={"id": "parent", "path": "synthetic"} if connected else None)
    methods = [method for method, _ in calls]
    assert "thread/fork" not in methods
    assert ("thread/resume" in methods) == (not connected)
    assert ("initialize" in methods) == (not connected)
    assert result["thread_id"] == "parent"
    assert result["thread_action"] == ("continue" if connected else "resume")
    assert result["usage"]["input_tokens"] == 1000
    assert result["output"] == {"ok": True}
    assert not rpc.closed
