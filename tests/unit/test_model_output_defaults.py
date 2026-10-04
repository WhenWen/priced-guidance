"""Future-run output allowances; all provider responses here are synthetic."""

from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest

from tech_tree_arena.evaluation.research import JUDGE_MAX_TOKENS
from tech_tree_arena.errors import ReplayDivergence
from tech_tree_arena.model_defaults import DEFAULT_MAX_OUTPUT_TOKENS
from tech_tree_arena.replay.recorder import RunRecorder, reconcile_provider_attempt_journal
from tech_tree_arena.runtime import provider_client
from tech_tree_arena.runtime.providers import ModelProviderBackend
from tech_tree_arena.runtime.services import ServiceEvent, ServiceFactory, ServiceLimits, _request_hash


class CaptureBackend:
    def __init__(self):
        self.requests = []

    def structured(self, **request):
        self.requests.append(request)
        return {"ok": True}


@pytest.mark.parametrize("requested, expected", [(None, 50_000), (1_000, 1_000), (80_000, 50_000)])
def test_service_default_and_explicit_output_requests(requested, expected):
    backend = CaptureBackend()
    factory = ServiceFactory(seed=1, model_backend=backend, model_name="gpt-6-astra")
    services = factory.create()
    kwargs = {} if requested is None else {"max_output_tokens": requested}

    assert services.structured_model(**kwargs) == {"ok": True}
    assert backend.requests[0]["max_output_tokens"] == expected
    assert backend.requests[0]["timeout"] == 300.0

    # Resolved allowances are taped, so replay does not repeat provider calls.
    replay = factory.create(services.export_tape())
    assert replay.structured_model(**kwargs) == {"ok": True}
    replay.finish_replay("output-default-test")
    assert len(backend.requests) == 1


def test_custom_service_limit_remains_authoritative():
    backend = CaptureBackend()
    services = ServiceFactory(
        seed=1, model_backend=backend,
        limits=ServiceLimits(max_output_tokens_per_call=123),
    ).create()
    services.structured_model()
    assert backend.requests[0]["max_output_tokens"] == 123


class FakeOpenAIClient:
    def __init__(self, empty_responses=0):
        self.responses = self
        self.requests = []
        self.empty_responses = empty_responses

    def with_options(self, **_kwargs):
        return self

    def create(self, **body):
        self.requests.append(body)
        empty = len(self.requests) <= self.empty_responses
        text = "" if empty else '{"ok":true}'
        response_id = f"resp-synthetic-{len(self.requests)}"
        payload = {
            "id": response_id,
            "status": "incomplete" if empty else "completed",
            "incomplete_details": {"reason": "max_output_tokens"} if empty else None,
            "output": [] if empty else [{"type": "message", "content": [
                {"type": "output_text", "text": text},
            ]}],
        }
        return SimpleNamespace(
            id=response_id, output_text=text, model_dump=lambda: payload,
            usage=SimpleNamespace(input_tokens=12, output_tokens=3),
        )


_SCHEMA = {
    "type": "object", "properties": {"ok": {"type": "boolean"}},
    "required": ["ok"], "additionalProperties": False,
}


@pytest.mark.parametrize("model", ["gpt-6-astra", "gpt-5.6-sol", "gpt-5.5"])
@pytest.mark.parametrize("requested", [None, 1_000])
def test_provider_wire_receives_default_or_explicit_allowance(monkeypatch, model, requested):
    client = FakeOpenAIClient()
    monkeypatch.setattr(provider_client, "client", lambda: client)
    monkeypatch.setattr(provider_client, "_TRACE_PATH", None)
    kwargs = {} if requested is None else {"max_output_tokens": requested}

    result = provider_client.structured(
        model=model, developer="Return the schema.", user="Synthetic test.",
        schema=_SCHEMA, schema_name="output_default", **kwargs,
    )
    assert result == {"ok": True}
    assert client.requests[0]["max_output_tokens"] == (requested or 50_000)


@pytest.mark.parametrize("initial, expected", [
    (16_000, [16_000, 24_000, 36_000]),
    (40_000, [40_000, 50_000, 50_000]),
    (50_000, [50_000, 50_000, 50_000]),
    (64_000, [64_000, 64_000, 64_000]),
])
def test_openai_parse_retries_do_not_silently_exceed_default_ceiling(monkeypatch, initial, expected):
    client = FakeOpenAIClient(empty_responses=2)
    monkeypatch.setattr(provider_client, "client", lambda: client)
    _, parsed, attempts = provider_client._create_parse_retry(
        {"model": "gpt-6-astra", "max_output_tokens": initial},
        timeout=300.0, schema_name="synthetic_retry",
    )
    assert parsed == {"ok": True}
    assert [body["max_output_tokens"] for body in client.requests] == expected
    assert [row["status"] for row in attempts] == ["parse_invalid", "parse_invalid", "ok"]
    assert all(row["usage_available"] for row in attempts)


@pytest.mark.parametrize("variant", ["reference_pair", "reference_pair_fable51"])
def test_every_reference_model_call_uses_the_frozen_50k_policy(variant):
    root = Path(__file__).resolve().parents[2] / "submissions" / variant
    pair = ast.parse((root / "participant/pair.py").read_text())
    value = next(
        ast.literal_eval(node.value) for node in pair.body
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "MODEL_MAX_OUTPUT_TOKENS"
                for target in node.targets)
    )
    assert value == DEFAULT_MAX_OUTPUT_TOKENS == 50_000
    calls = []
    for path in (root / "participant").rglob("*.py"):
        for call in ast.walk(ast.parse(path.read_text())):
            if not (isinstance(call, ast.Call) and isinstance(call.func, ast.Attribute)
                    and call.func.attr == "structured_model"):
                continue
            cap = next(kw.value for kw in call.keywords if kw.arg == "max_output_tokens")
            assert ast.unparse(cap) in {"MODEL_MAX_OUTPUT_TOKENS", "_pair.MODEL_MAX_OUTPUT_TOKENS"}
            calls.append(call)
    assert len(calls) == 16


def test_judge_and_service_defaults_agree():
    assert JUDGE_MAX_TOKENS == ServiceLimits().max_output_tokens_per_call == 50_000


@pytest.mark.parametrize("legacy", [False, True])
@pytest.mark.parametrize("tamper", [False, True])
def test_accounting_binds_resolved_defaults_without_reinterpreting_old_records(
    monkeypatch, tmp_path, legacy, tamper,
):
    client = FakeOpenAIClient()
    monkeypatch.setattr(provider_client, "client", lambda: client)
    monkeypatch.setattr(provider_client, "_TRACE_PATH", None)
    recorder = RunRecorder(tmp_path, "output-default", {"budgets": {}})
    backend = ModelProviderBackend(
        attempt_sink=lambda attempt: recorder.record_provider_attempt("generator", attempt),
    )
    request = {
        "model": "gpt-6-astra", "developer": "Return the schema.",
        "user": "Synthetic accounting test.", "schema": _SCHEMA,
        "schema_name": "output_default_accounting",
    }
    # Emulate the old provider's 8k default, then its old metadata shape.
    output = backend.structured(**request, **({"max_output_tokens": 8000} if legacy else {}))
    metadata = backend.last_call_metadata()
    if legacy:
        metadata.pop("max_output_tokens")
    else:
        assert metadata["max_output_tokens"] == 50_000
    if tamper:
        metadata["max_output_tokens"] = 49_999
    recorder.record_service("generator", ServiceEvent(
        kind="model.structured", request_hash=_request_hash("model.structured", request),
        request=request, response=output, metadata=metadata,
    ))
    paths = (recorder.root / "provider-attempts.private.jsonl",
             recorder.root / "service-calls.private.jsonl")
    if tamper:
        with pytest.raises(ReplayDivergence, match="logical hash diverges"):
            reconcile_provider_attempt_journal(*paths)
    else:
        assert reconcile_provider_attempt_journal(*paths)["logical_successes"] == 1
