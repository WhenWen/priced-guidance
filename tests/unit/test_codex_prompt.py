import json

import pytest

from tech_tree_arena.runtime import codex_rpc
from tech_tree_arena.runtime.codex_prompts import DEFAULT_VERSION, EAGER_VERSION, LEGACY_VERSION, instructions


@pytest.mark.parametrize("fork", [False, True])
@pytest.mark.parametrize("version", [DEFAULT_VERSION, EAGER_VERSION, LEGACY_VERSION])
def test_native_start_and_fork_receive_pinned_prompt_and_original_request(tmp_path, monkeypatch, fork, version):
    calls = []
    parent = {"thread_id": "parent", "turn_id": "previous", "usage_total": {"inputTokens": 100, "outputTokens": 20}} if fork else None
    baseline = parent["usage_total"] if parent else {}
    request = {"model": "gpt-6-astra", "developer": "Author the current route only.",
               "user": "The public draft and ledger for this call.", "reasoning_effort": "high",
               "schema": {"type": "object", "properties": {"options": {"type": "array"}}}}

    class RPC:
        def __init__(self, *args, **kwargs):
            self.events = []

        def call(self, method, params):
            calls.append((method, params))
            if method in {"thread/start", "thread/fork"}:
                return {"model": request["model"], "thread": {"id": "child", "path": str(tmp_path / "rollout")}}
            if method == "turn/start":
                self.events.extend([
                    {"method": "thread/tokenUsage/updated", "params": {"tokenUsage": {"total": {
                        "inputTokens": baseline.get("inputTokens", 0) + 3,
                        "outputTokens": baseline.get("outputTokens", 0) + 1}}}},
                    {"method": "turn/completed", "params": {"turn": {"id": "turn", "status": "completed",
                        "items": [{"type": "agentMessage", "text": json.dumps(
                            {"payload_json": json.dumps({"options": []})} if version == DEFAULT_VERSION else {"options": []})}]}}},
                ])
                return {"turn": {"id": "turn"}}
            return {}

        def send(self, value):
            pass

        def close(self):
            pass

    monkeypatch.setattr(codex_rpc, "CodexRPC", RPC)
    result = codex_rpc.run_turn(tmp_path / "codex", tmp_path, request, parent, version,
                                cache_key="a" * 64 if version == DEFAULT_VERSION else None)
    thread_method, thread_params = calls[1]
    assert thread_method == ("thread/fork" if fork else "thread/start")
    assert thread_params["baseInstructions"] == instructions(version)
    if version != DEFAULT_VERSION:
        assert thread_params["developerInstructions"] == request["developer"]
    turn_params = calls[2][1]
    assert turn_params["input"] == [{"type": "text", "text": request["user"]}]
    if version == DEFAULT_VERSION:
        assert turn_params["outputSchema"] == codex_rpc.ENVELOPE_SCHEMA
        task = turn_params["additionalContext"]["idea_arena_call"]
        assert task["kind"] == "application"
        assert request["developer"] in task["value"]
        assert json.dumps(request["schema"], sort_keys=True) in task["value"]
    else:
        assert turn_params["outputSchema"] == request["schema"]
    assert result["usage"]["input_tokens"] == 3
    assert result["output"] == {"options": []}
