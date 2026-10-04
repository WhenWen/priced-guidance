import copy
import json

import pytest

from tech_tree_arena.errors import ReplayDivergence
from tech_tree_arena.replay.artifacts import ArtifactStore
from tech_tree_arena.runtime.common_memory import CommonMemoryBackend, configuration, validate_metadata
from tech_tree_arena.runtime.native_api_generator import NativeAPIGeneratorBackend
from tech_tree_arena.runtime.services import ServiceFactory


MODEL = "anthropic/claude-fable-5-1"
SCHEMA = {"type": "object", "properties": {"answer": {"type": "string"}}, "required": ["answer"], "additionalProperties": False}


def request(user="initial"):
    return {"model": MODEL, "developer": "Test task", "user": user,
            "schema_name": "test", "schema": SCHEMA, "max_output_tokens": 50000}


def reply(output):
    return {"id": "synthetic", "stop_reason": "end_turn", "input_transformations": [],
            "content": [{"type": "thinking", "thinking": "synthetic private state", "signature": "opaque-signature"},
                        {"type": "text", "text": json.dumps({"payload_json": json.dumps(output)})}],
            "usage": {"input_tokens": 20, "output_tokens": 10, "cache_read_input_tokens": 8, "cache_creation_input_tokens": 0}}


@pytest.fixture
def setup(tmp_path, monkeypatch):
    store = ArtifactStore(tmp_path)
    transport = NativeAPIGeneratorBackend(MODEL, store, attempts=1)
    bodies = []
    def send(body):
        bodies.append(copy.deepcopy(body))
        if "common_memory_summary" in body["messages"][-1]["content"]:
            return reply({"memory": "Remember the original anchor and pending exact Question."})
        return reply({"answer": "ok"})
    monkeypatch.setattr(transport, "_send", send)
    backend = CommonMemoryBackend(configuration(MODEL, compact_calls=1), store, transport=transport)
    return backend, transport, store, bodies


def test_summary_inherits_signed_history_and_new_window_is_clean(setup):
    backend, transport, store, bodies = setup
    backend.structured(**request())
    first = backend.last_call_metadata()["native_context"]
    before = store.load_json(first["native"]["history"])
    backend.structured_in_context(first, **request("next"))
    meta = backend.last_call_metadata()
    assert len(bodies) == 3
    assert bodies[1]["messages"][:-1] == before["messages"]
    assert bodies[1]["system"] == bodies[0]["system"] == bodies[2]["system"]
    assert len(bodies[2]["messages"]) == 1
    assert "Remember the original anchor" in bodies[2]["messages"][0]["content"]
    assert meta["operations"][0]["metadata"]["reasoning_blocks_inherited"] == 1
    assert store.load_json(first["native"]["history"]) == before
    assert meta["native_context"]["window"] == 1
    assert backend.usage_totals()["physical_calls"] == 3
    assert backend.usage_totals()["calls"] == 2
    validate_metadata({"request": request("next"), "response": {"answer": "ok"}, "metadata": meta})


def test_replay_and_checkout_never_see_abandoned_branch(setup):
    backend, _, _, bodies = setup
    resources = {"generator_memory_policy": {"deterministic_calls": True, "timeout": 1500}}
    factory = ServiceFactory(seed=3, model_backend=backend, model_name=MODEL, public_resources=resources)
    services = factory.create()
    services.structured_model(**request("anchor"))
    tape = services.export_tape()
    parent = services.export_state()["model_context"]
    services.structured_model(**request("ABANDONED"))
    branch = factory.create(tape)
    previous = len(bodies)
    branch.structured_model(**request("anchor"))
    assert len(bodies) == previous
    branch.finish_replay("branch")
    branch.structured_model(**request("alternative"))
    meta = backend.last_call_metadata()
    assert meta["parent_context"] == parent
    assert "ABANDONED" not in json.dumps(bodies[-2:])


def test_failed_summary_does_not_commit_context_and_accounts_attempts(setup, monkeypatch):
    backend, transport, _, _ = setup
    backend.structured(**request())
    parent = backend.last_call_metadata()["native_context"]
    def bad(body):
        return reply({"memory": "x" * 12001})
    monkeypatch.setattr(transport, "_send", bad)
    with pytest.raises(RuntimeError):
        backend.structured_in_context(parent, **request("next"))
    meta = backend.last_call_metadata()
    assert "native_context" not in meta
    assert meta["usage"]["physical_calls"] == 3
    assert meta["usage"]["output_tokens"] == 30
    assert parent["window"] == 0


def test_policy_and_accounting_tampering_are_rejected(setup):
    backend, _, _, _ = setup
    output = backend.structured(**request())
    meta = backend.last_call_metadata()
    context = copy.deepcopy(meta["native_context"])
    context["policy_sha256"] = "changed"
    with pytest.raises(ReplayDivergence):
        backend.structured_in_context(context, **request())
    meta["usage"]["output_tokens"] = 0
    with pytest.raises(ReplayDivergence):
        validate_metadata({"request": request(), "response": output, "metadata": meta})


def test_cap_transition_preserves_parent_receipt_and_rejects_unknown_ancestor(setup):
    from tech_tree_arena.runtime.memory_policy_transition import ContinuedMemoryBackend
    from tech_tree_arena.runtime.common_memory import digest
    old, transport, store, _ = setup
    old.structured(**request())
    parent = old.last_call_metadata()["native_context"]
    original = copy.deepcopy(parent)
    config = {**old.config, "memory_chars": 14000}
    new = ContinuedMemoryBackend(config, store, ancestors=[old.config], transport=transport)
    for context in (parent, None, parent):
        output = new.structured_in_context(context, **request("continued"))
        meta = new.last_call_metadata()
        assert meta["parent_context"] == context
        assert meta["native_context"]["policy_sha256"] == digest(config)
        validate_metadata({"request": request("continued"), "response": output, "metadata": meta}, store)
    assert parent == original
    with pytest.raises(ReplayDivergence, match="Unrecorded"):
        new.structured_in_context({**parent, "policy_sha256": "unlisted"}, **request())


@pytest.mark.parametrize("cap,length", [(16000, 15592), (20000, 16639), (20000, 20000)])
def test_summary_target_is_12k_with_lenient_hard_cap(setup, monkeypatch, cap, length):
    from tech_tree_arena.runtime.memory_summary_target import TargetedMemoryBackend
    old, transport, store, bodies = setup
    old.structured(**request())
    parent = old.last_call_metadata()["native_context"]
    def send(body):
        bodies.append(copy.deepcopy(body))
        return reply({"memory": "x" * length} if "common_memory_summary" in body["messages"][-1]["content"] else {"answer": "ok"})
    monkeypatch.setattr(transport, "_send", send)
    backend = TargetedMemoryBackend({**old.config, "memory_chars": cap}, store,
                                    ancestors=[old.config], target=12000, transport=transport)
    result = backend.structured_in_context(parent, **request("next"))
    meta = backend.last_call_metadata()
    summary = meta["operations"][0]
    assert "12000" in summary["request"]["developer"]
    assert str(cap) not in summary["request"]["developer"]
    assert summary["request"]["schema"]["properties"]["memory"]["maxLength"] == cap
    assert len(summary["response"]["memory"]) == length
    assert meta["parent_context"] == parent
    assert meta["operations"][-1]["request"]["developer"] == request()["developer"]
    validate_metadata({"request": request("next"), "response": result, "metadata": meta}, store)
    backend.structured_in_context(meta["native_context"], **request("again"))


def test_glm_formatter_is_separate_and_preserves_exact_main_history(tmp_path, monkeypatch):
    model = "together/zai-org/GLM-5.3"
    store = ArtifactStore(tmp_path)
    backend = NativeAPIGeneratorBackend(model, store, attempts=1)
    bodies = []
    def send(body):
        bodies.append(copy.deepcopy(body))
        formatter = not body["reasoning"]["enabled"]
        message = {"role": "assistant", "content": json.dumps({"answer": "ok"}) if formatter
                   else "Final answer: ok"}
        if not formatter:
            message["reasoning_content"] = "exact native reasoning"
        return {"choices": [{"finish_reason": "stop", "message": message}],
                "usage": {"prompt_tokens": 20, "completion_tokens": 10}}
    monkeypatch.setattr(backend, "_send", send)
    req = {**request(), "model": model}
    backend.structured(**req)
    context = backend.last_call_metadata()["native_context"]
    backend.structured_in_context(context, **{**req, "user": "next"})
    assert bodies[2]["messages"][2]["reasoning_content"] == "exact native reasoning"
    assert bodies[2]["chat_template_kwargs"] == {"clear_thinking": False}
    assert "response_format" not in bodies[2]
    assert len(bodies) == 4
    assert "exact native reasoning" not in json.dumps(bodies[1])
    assert "exact native reasoning" not in json.dumps(bodies[3])
    history = store.load_json(backend.last_call_metadata()["native_context"]["history"])["messages"]
    assert len(history) == 4
    assert all(m["content"] == "Final answer: ok" for m in history if m["role"] == "assistant")
    usage = backend.usage_totals()
    assert usage["physical_calls"] == 4
    assert usage["thinking_calls"] == usage["format_calls"] == 2


def test_glm_formatter_retries_do_not_rerun_thinking(tmp_path, monkeypatch):
    model = "together/zai-org/GLM-5.3"
    transport = NativeAPIGeneratorBackend(model, ArtifactStore(tmp_path), attempts=1)
    backend = CommonMemoryBackend(configuration(model), transport.artifacts, transport=transport)
    bodies = []
    monkeypatch.setattr("tech_tree_arena.runtime.native_api_generator.time.sleep", lambda _: None)
    def send(body):
        bodies.append(body)
        value = {"answer": "ok"} if body["reasoning"]["enabled"] else {"answer": 17}
        return {"choices": [{"finish_reason": "stop", "message": {"content": "Final answer: ok" if body["reasoning"]["enabled"] else json.dumps(value),
                  "reasoning_content": "private" if body["reasoning"]["enabled"] else None}}],
                "usage": {"prompt_tokens": 20, "completion_tokens": 10}}
    monkeypatch.setattr(transport, "_send", send)
    with pytest.raises(RuntimeError, match="format failed"):
        backend.structured(**{**request(), "model": model})
    assert len(bodies) == 4
    assert sum(b["reasoning"]["enabled"] for b in bodies) == 1
    assert "native_context" not in backend.last_call_metadata()
    assert backend.usage_totals()["physical_calls"] == 4


@pytest.mark.parametrize("fence", ["```json\n", "```\r\n"])
@pytest.mark.parametrize("envelope", [False, True, "label", "empty_actions"])
def test_glm_fenced_probabilities_skip_formatter_and_keep_history(tmp_path, monkeypatch, fence, envelope):
    from tech_tree_arena.runtime.native_api_generator import validate_artifacts
    model = "together/zai-org/GLM-5.3"
    backend = NativeAPIGeneratorBackend(model, ArtifactStore(tmp_path))
    schema = {"type": "object", "properties": {"ideas": {"type": "array", "items": {
        "type": "object", "properties": {"draft": {"type": "string"},
        "prob": {"type": "number", "exclusiveMinimum": 0}}, "required": ["draft", "prob"]}}},
        "required": ["ideas"], "additionalProperties": False}
    payload = {"ideas": [{"draft": "one", "prob": 0.16}, {"draft": "two", "prob": 0.000013}]}
    value = {"payload_json": json.dumps(payload)} if envelope is True else payload
    if envelope == "empty_actions":
        value = {**payload, "guesses": [], "options": [], "question": "", "reasoning": "",
                 "prob_all_incorrect": 0, "prob_ask_different": 0, "prob_none": 0}
    text = fence + ("payload_json: " if envelope == "label" else "") + json.dumps(value) + "\n```"
    calls = []
    def send(body):
        assert body["reasoning"]["enabled"], "Valid fenced JSON must never call the formatter"
        calls.append(body)
        return {"choices": [{"finish_reason": "stop", "message": {
            "content": text, "reasoning_content": "unchanged reasoning"}}],
            "usage": {"prompt_tokens": 20, "completion_tokens": 10}}
    monkeypatch.setattr(backend, "_send", send)
    req = {**request(), "model": model, "schema": schema}
    assert backend.structured(**req) == payload
    meta = backend.last_call_metadata()
    assert meta["formatting"] == {"mode": "local_validated", "physical_calls": 0}
    history = backend.artifacts.load_json(meta["native_context"]["history"])["messages"]
    assert history[-1]["content"] == text
    assert history[-1]["reasoning_content"] == "unchanged reasoning"
    validate_artifacts(meta, req, backend.artifacts)
    assert len(calls) == 1


@pytest.mark.parametrize("text", [
    '```json\n{"answer": 0}\n```',
    'Extra prose\n```json\n{"answer": "ok"}\n```',
    '```json\n{"answer": "ok"}\n```\nExtra prose',
    'payload_json: {"answer": "ok"} Extra prose',
    'payload_json: {"answer": 0}',
])
def test_local_fence_parsing_does_not_relax_schema_or_extract_prose(text):
    from tech_tree_arena.runtime.native_api_generator import local_payload
    assert local_payload(text, SCHEMA) is None


@pytest.mark.parametrize("extra", [{"question": "active question"}, {"options": [{}]},
    {"prob_none": 0.1}, {"prob_none": False}, {"unknown": []}])
def test_ideas_cleanup_never_discards_nonempty_or_unknown_fields(extra):
    from tech_tree_arena.runtime.native_api_generator import local_payload
    schema = {"type": "object", "properties": {"ideas": {"type": "array"}},
              "required": ["ideas"], "additionalProperties": False}
    assert local_payload(json.dumps({"ideas": [], **extra}), schema) is None


def test_ideas_cleanup_still_rejects_zero_probabilities():
    from tech_tree_arena.runtime.native_api_generator import local_payload
    schema = {"type": "object", "properties": {"ideas": {"type": "array", "items": {
        "type": "object", "properties": {"prob": {"type": "number", "exclusiveMinimum": 0}},
        "required": ["prob"]}}}, "required": ["ideas"], "additionalProperties": False}
    assert local_payload(json.dumps({"ideas": [{"prob": 0}], "options": []}), schema) is None


def test_exact_pre_fence_fix_configuration_remains_replayable(tmp_path):
    config = configuration("together/zai-org/GLM-5.3")
    config["source_sha256"].update({
        "common_memory.py": "7bb7bb2edb6ce39e94f1964c519e577c77c033f31c2959d23d1b6b3c593c47db",
        "native_api_generator.py": "265d99b9a8777deda83f110f3a2d4ab3a73b0a15a6635a06dfe8d1510547effe"})
    CommonMemoryBackend(config, ArtifactStore(tmp_path))
    config["source_sha256"]["native_api_generator.py"] = "unreviewed"
    with pytest.raises(ReplayDivergence):
        CommonMemoryBackend(config, ArtifactStore(tmp_path / "bad"))


def test_refusal_is_metered_and_not_retried(setup, monkeypatch):
    backend, transport, _, bodies = setup
    def refuse(body):
        bodies.append(body)
        result = reply({})
        result.update(content=[], stop_reason="refusal", stop_details={"category": "reasoning_extraction"})
        return result
    monkeypatch.setattr(transport, "_send", refuse)
    with pytest.raises(RuntimeError, match="reasoning_extraction"):
        backend.structured(**request())
    assert len(bodies) == 1
    assert backend.usage_totals()["physical_calls"] == 1
    assert backend.usage_totals()["output_tokens"] == 10


def test_worker_disables_preview_races_only_for_common_policy():
    from tech_tree_arena.runtime.worker import RemoteServices
    assert RemoteServices(None, {}).supports_concurrent_calls
    assert not RemoteServices(None, {"generator_memory_policy": {"deterministic_calls": True}}).supports_concurrent_calls


def test_valid_glm_json_never_needs_a_paid_rewrite(tmp_path, monkeypatch):
    backend = NativeAPIGeneratorBackend("together/zai-org/GLM-5.3", ArtifactStore(tmp_path))
    metadata = {}
    def forbidden(body):
        pytest.fail("already valid JSON must not be rewritten")
    monkeypatch.setattr(backend, "_send", forbidden)
    text = json.dumps({"payload_json": json.dumps({"answer": "full original answer"})})
    assert backend._format(text, request(), metadata) == {"answer": "full original answer"}
    assert metadata["formatting"]["mode"] == "local_validated"


def test_native_journal_and_canonical_history_are_verified(setup):
    backend, _, store, _ = setup
    req = request()
    output = backend.structured(**req)
    meta = backend.last_call_metadata()
    validate_metadata({"request": req, "response": output, "metadata": meta}, store)
    history = meta["native_context"]["native"]["history"]
    forged = store.load_json(history)
    forged["messages"][-1]["content"][-1]["text"] = "formatter overwrote original"
    forged_ref = store.put_json(forged)
    meta["native_context"]["native"]["history"] = forged_ref
    meta["operations"][-1]["metadata"]["native_context"]["history"] = forged_ref
    with pytest.raises(ReplayDivergence, match="rewritten"):
        validate_metadata({"request": req, "response": output, "metadata": meta}, store)


def test_provider_cannot_silently_fallback_to_a_different_model(setup, monkeypatch):
    backend, transport, _, _ = setup
    def send(body):
        payload = reply({"answer": "ok"})
        payload["model"] = "unexpected-fallback-model"
        return payload
    monkeypatch.setattr(transport, "_send", send)
    with pytest.raises(RuntimeError, match="different model"):
        backend.structured(**request())
    assert backend.usage_totals()["physical_calls"] == 1


@pytest.mark.parametrize("field", ["reasoning_content", "reasoning"])
def test_streamed_glm_retains_provider_reasoning_field(tmp_path, monkeypatch, field):
    from tech_tree_arena.runtime import provider_client as pc
    chunks = [
        {"id": "call", "model": "zai-org/GLM-5.3", "choices": [{"delta": {field: "first "}}]},
        {"choices": [{"delta": {field: "second", "content": "hello "}}]},
        {"choices": [{"delta": {"content": "world"}, "finish_reason": "stop"}],
         "usage": {"prompt_tokens": 20, "completion_tokens": 10}},
    ]
    class Response:
        def __enter__(self):
            return iter(("data: " + json.dumps(c) + "\n").encode() for c in chunks)
        def __exit__(self, *args):
            pass
    monkeypatch.setattr(pc, "_load_together_key", lambda: "synthetic-key")
    requests = []
    def open_request(req, timeout):
        requests.append(json.loads(req.data))
        return Response()
    monkeypatch.setattr(pc.urllib.request, "urlopen", open_request)
    response = pc._together_chat_complete_streaming({"model": "zai-org/GLM-5.3"}, 1500,
                                                  preserve_reasoning_fields=True)
    message = response["choices"][0]["message"]
    assert message[field] == "first second"
    assert ({"reasoning", "reasoning_content"} - {field}).isdisjoint(message)
    assert requests[0]["stream"] is True
    assert requests[0]["stream_options"] == {"include_usage": True}
    assert response["usage"]["completion_tokens"] == 10


def test_anthropic_sdk_parse_helpers_are_not_sent_back_as_api_fields(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from anthropic.types import ParsedMessage, ParsedTextBlock
    from tech_tree_arena.runtime import provider_client as pc
    message = ParsedMessage.model_construct(content=[ParsedTextBlock.model_construct(
        type="text", text='{"payload_json":"{}"}', parsed_output={"payload_json": "{}"})])
    class Stream:
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def get_final_message(self): return message
    monkeypatch.setattr(pc, "anthropic_client", lambda: SimpleNamespace(
        messages=SimpleNamespace(stream=lambda **kwargs: Stream())))
    backend = NativeAPIGeneratorBackend(MODEL, ArtifactStore(tmp_path))
    payload = backend._send({})
    assert payload["content"][0]["text"] == '{"payload_json":"{}"}'
    assert "parsed_output" not in payload["content"][0]


def test_common_envelope_does_not_add_a_research_strategy():
    from tech_tree_arena.runtime.codex_prompts import COMMON_VERSION, instructions
    from tech_tree_arena.runtime.native_api_generator import task_text
    prompt = instructions(COMMON_VERSION)
    assert "eager" not in prompt and "hypothes" not in prompt and "research" not in prompt
    task = request()
    assert task["developer"] in task_text(task)
    assert task["user"] in task_text(task)
