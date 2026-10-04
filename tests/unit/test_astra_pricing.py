"""Astra/Sol billing used by the directional experiment's per-role budget."""

from types import SimpleNamespace

import pytest

from tech_tree_arena.runtime import provider_client
from tech_tree_arena.runtime.providers import ModelProviderBackend
from tech_tree_arena.runtime.services import ServiceEvent, _request_hash
from tech_tree_arena.replay.recorder import RunRecorder, reconcile_provider_attempt_journal


@pytest.mark.parametrize("input_tokens, expected", [(272_000, 3.04), (272_001, 6.05502)])
@pytest.mark.parametrize("model, scale", [("gpt-6-astra", 1), ("gpt-5.6-sol", 0.4)])
def test_astra_cache_classes_and_long_context_boundary(input_tokens, expected, model, scale):
    usage = SimpleNamespace(
        input_tokens=input_tokens,
        output_tokens=1_000,
        input_tokens_details=SimpleNamespace(cached_tokens=20_000, cache_write_tokens=180_000),
    )
    # At 272K: 72K uncached ($.72), 20K reads ($.02), 180K writes
    # ($2.25), and 1K output ($.05). Above it, input doubles and output x1.5.
    assert provider_client._openai_attempt_cost(model, usage) == pytest.approx(expected * scale)


@pytest.mark.parametrize("model, expected", [("gpt-6-astra", 0.0285), ("gpt-5.6-sol", 0.0114)])
def test_astra_provider_attempt_records_billed_usage(monkeypatch, model, expected):
    response = SimpleNamespace(
        id="resp_astra_test",
        usage=SimpleNamespace(
            input_tokens=3_000,
            output_tokens=100,
            input_tokens_details=SimpleNamespace(cached_tokens=1_000, cache_write_tokens=1_000),
        ),
    )
    client = SimpleNamespace(responses=SimpleNamespace(create=lambda **kwargs: response))
    client.with_options = lambda **kwargs: client
    monkeypatch.setattr(provider_client, "client", lambda: client)
    monkeypatch.setattr(provider_client, "_provider_response_sha", lambda response: "test-sha")
    _, attempts = provider_client._openai_response_create({"model": model}, timeout=1)
    assert attempts[0]["cost_usd"] == pytest.approx(expected)
    assert attempts[0]["cache_read_input_tokens"] == 1_000
    assert attempts[0]["cache_write_input_tokens"] == 1_000


def test_existing_openai_budget_accounting_is_preserved():
    usage = SimpleNamespace(input_tokens=3_000, output_tokens=100)
    assert provider_client._openai_attempt_cost("gpt-5.5", usage) == pytest.approx(0.018)


@pytest.mark.parametrize("model, expected", [("gpt-6-astra", 0.0285), ("gpt-5.6-sol", 0.0114)])
def test_astra_cache_cost_survives_durable_journal_validation(tmp_path, monkeypatch, model, expected):
    response = SimpleNamespace(
        id="resp_astra_journal", output_text='{"ok":true}',
        usage=SimpleNamespace(input_tokens=3_000, output_tokens=100,
            input_tokens_details=SimpleNamespace(cached_tokens=1_000, cache_write_tokens=1_000)),
    )
    client = SimpleNamespace(responses=SimpleNamespace(create=lambda **kwargs: response))
    client.with_options = lambda **kwargs: client
    monkeypatch.setattr(provider_client, "client", lambda: client)
    monkeypatch.setattr(provider_client, "_provider_response_sha", lambda response: "a" * 64)
    recorder = RunRecorder(tmp_path, "astra-cache", {"budgets": {}})
    backend = ModelProviderBackend(attempt_sink=lambda a: recorder.record_provider_attempt("generator", a))
    request = dict(model=model, developer="Return JSON", user="Return true", reasoning_effort="max",
                   schema={"type": "object", "properties": {"ok": {"type": "boolean"}},
                           "required": ["ok"], "additionalProperties": False}, schema_name="astra_cache")
    result = backend.structured(**request)
    recorder.record_service("generator", ServiceEvent(
        kind="model.structured", request_hash=_request_hash("model.structured", request),
        request=request, response=result, metadata=backend.last_call_metadata(),
    ))
    reconciled = reconcile_provider_attempt_journal(
        recorder.root / "provider-attempts.private.jsonl", recorder.root / "service-calls.private.jsonl")
    assert reconciled["provider_calls"] == 1
    assert backend.usage_totals()["cost_usd"] == pytest.approx(expected)
