import copy
import json

import pytest

from tech_tree_arena.errors import ReplayDivergence
from tech_tree_arena.replay.artifacts import ArtifactStore
from tech_tree_arena.runtime.common_memory import CommonMemoryBackend, configuration as memory_configuration
from tech_tree_arena.runtime.memory_output_budget import OutputBudgetBackend, configuration, validate_metadata
from tech_tree_arena.runtime.native_api_generator import NativeAPIGeneratorBackend


@pytest.mark.parametrize("model", ["anthropic/claude-fable-5-1", "anthropic/claude-fable-5", "anthropic/claude-opus-5", "together/zai-org/GLM-5.3"])
def test_native_summary_and_formatter_get_128k_without_changing_prompts(tmp_path, monkeypatch, model):
    artifacts = ArtifactStore(tmp_path)
    native = NativeAPIGeneratorBackend(model, artifacts, attempts=1)
    bodies = []
    def send(body):
        bodies.append(copy.deepcopy(body))
        summary = "common_memory_summary" in body["messages"][-1]["content"]
        output = {"memory": "Keep the exact pending question."} if summary else {"answer": "ok"}
        if model.startswith("anthropic/"):
            return {"stop_reason": "end_turn", "usage": {"input_tokens": 20, "output_tokens": 51000},
                    "content": [{"type": "thinking", "thinking": "fixture", "signature": "opaque"},
                                {"type": "text", "text": json.dumps({"payload_json": json.dumps(output)})}]}
        formatting = not body["reasoning"]["enabled"]
        message = {"role": "assistant", "content": json.dumps(output) if formatting or summary else "Answer: ok"}
        if not formatting:
            message["reasoning_content"] = "fixture preserved thinking"
        return {"choices": [{"finish_reason": "stop", "message": message}],
                "usage": {"prompt_tokens": 20, "completion_tokens": 51000 if not formatting else 10}}
    monkeypatch.setattr(native, "_send", send)
    base = CommonMemoryBackend(memory_configuration(model, compact_calls=1), artifacts, transport=native)
    backend = OutputBudgetBackend(base, configuration(model))
    request = {"model": model, "developer": "Original task prompt", "user": "Original state",
               "schema_name": "answer", "schema": {"type": "object"}, "max_output_tokens": 50000}
    original = copy.deepcopy(request)
    backend.structured(**request)
    parent = backend.last_call_metadata()["native_context"]
    first_history = artifacts.load_json(parent["native"]["history"])
    result = backend.structured_in_context(parent, **request)
    metadata = backend.last_call_metadata()
    assert request == original
    assert metadata["parent_context"] == parent
    assert artifacts.load_json(parent["native"]["history"]) == first_history
    assert all(body["max_tokens"] == 128000 for body in bodies)
    assert metadata["operations"][-1]["request"]["developer"] == original["developer"]
    if model.startswith("together/"):
        assert any(not body["reasoning"]["enabled"] for body in bodies)
        assert all("fixture preserved thinking" not in json.dumps(body) for body in bodies if not body["reasoning"]["enabled"])
    service = {"request": request, "response": result, "metadata": metadata}
    validate_metadata(service, artifacts)
    tampered = copy.deepcopy(service)
    tampered["metadata"]["operations"][-1]["request"]["max_output_tokens"] = 50000
    with pytest.raises(ReplayDivergence):
        validate_metadata(tampered, artifacts)


def test_astra_and_unsupported_allowances_cannot_be_changed():
    with pytest.raises(ValueError):
        configuration("gpt-6-astra")
    for tokens in (True, 49999, 128001):
        with pytest.raises(ValueError):
            configuration("anthropic/claude-fable-5-1", tokens)
