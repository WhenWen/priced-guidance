from __future__ import annotations

import copy
from fractions import Fraction
import json
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from tech_tree_arena.errors import ReplayDivergence, ResourceLimitExceeded
from tech_tree_arena.replay.recorder import (
    HashChainWriter,
    RunRecorder,
    _validate_completed_service_accounting,
    reconcile_provider_attempt_journal,
    validate_provider_attempt_journal,
    verify_hash_chain,
)
from tech_tree_arena.runtime import provider_client
from tech_tree_arena.runtime.provider_client import _req_sha
from tech_tree_arena.runtime.providers import ModelProviderBackend
from tech_tree_arena.runtime.services import (
    ReplayableServices,
    ServiceEvent,
    ServiceLimits,
)
from tech_tree_arena.runtime.services import _request_hash


def _service_payload(event: ServiceEvent) -> dict:
    return {
        "kind": event.kind,
        "request_hash": event.request_hash,
        "response": event.response,
        "error": event.error,
        "request": event.request,
        "error_message": event.error_message,
        "started_at": event.started_at,
        "finished_at": event.finished_at,
        "metadata": event.metadata,
    }


@pytest.mark.parametrize('cost', [0, 1])
def test_zero_call_budget_admission_reconciles_without_physical_attempt(tmp_path, cost):
    provider = tmp_path / 'provider.jsonl'
    provider.write_text('')
    service = tmp_path / 'service.jsonl'
    request = dict(model='anthropic/claude-opus-4-6', developer='Return JSON.',
                   user='Synthetic request', schema_name='admission', schema={'type': 'object'},
                   max_output_tokens=50000, reasoning_effort='high')
    metadata = dict(model=request['model'], schema_name='admission', error_type='BudgetBlocked',
                    request_sha=_req_sha(request['model'], request['developer'], request['user'],
                                         'admission', request['schema'], 50000, 'high'),
                    max_output_tokens=50000,
                    usage=dict(calls=0, provider_calls=0, input_tokens=0, output_tokens=0,
                               cost_usd=cost, unknown_provider_attempts=0))
    HashChainWriter(service).append({'kind': 'service_call', 'role': 'oracle', 'service': {
        'kind': 'model.structured', 'request_hash': _request_hash('model.structured', request),
        'request': request, 'response': None, 'error': 'BudgetBlocked', 'metadata': metadata}})
    if cost:
        with pytest.raises(ReplayDivergence, match='nonzero'):
            reconcile_provider_attempt_journal(provider, service)
    else:
        reconcile_provider_attempt_journal(provider, service)


@pytest.mark.parametrize("record_success", [True, False])
def test_budget_admission_then_same_request_retry_accounts_for_every_attempt(
    tmp_path, monkeypatch, record_success
):
    payload = {"id": "synthetic-retry", "content": [{"type": "text", "text": '{"ok":true}'}],
               "usage": {"input_tokens": 10, "output_tokens": 5}}
    monkeypatch.setattr(provider_client, "anthropic_client", lambda: SimpleNamespace(
        messages=SimpleNamespace(create=lambda **kw: SimpleNamespace(model_dump=lambda: payload))))
    recorder = RunRecorder(tmp_path, "retry", {"budgets": {}})
    request = dict(model="anthropic/claude-opus-4-6", developer="Return JSON.",
                   user="Synthetic retry", schema_name="admission", reasoning_effort="low",
                   max_output_tokens=50000, schema={"type": "object", "properties": {
                       "ok": {"type": "boolean"}}, "required": ["ok"], "additionalProperties": False})
    metadata = dict(model=request["model"], schema_name="admission", error_type="BudgetBlocked",
                    request_sha=_req_sha(request["model"], request["developer"], request["user"],
                                         "admission", request["schema"], 50000, "low"),
                    max_output_tokens=50000, usage=dict(calls=0, provider_calls=0,
                    input_tokens=0, output_tokens=0, cost_usd=0, unknown_provider_attempts=0))
    recorder.record_service("oracle", ServiceEvent(kind="model.structured",
        request_hash=_request_hash("model.structured", request), request=request,
        response=None, error="BudgetBlocked", metadata=metadata))
    backend = ModelProviderBackend(attempt_sink=lambda a: recorder.record_provider_attempt("oracle", a))
    response = backend.structured(**request)
    if record_success:
        recorder.record_service("oracle", ServiceEvent(kind="model.structured",
            request_hash=_request_hash("model.structured", request), request=request,
            response=response, metadata=backend.last_call_metadata()))
    provider = recorder.root / "provider-attempts.private.jsonl"
    service = recorder.root / "service-calls.private.jsonl"
    if record_success:
        result = reconcile_provider_attempt_journal(provider, service)
        assert result["provider_calls"] == result["logical_calls"] == 1
    else:
        with pytest.raises(ReplayDivergence, match="journal and service metadata diverge"):
            reconcile_provider_attempt_journal(provider, service)


def test_run_recorder_attempt_chain_reconciles_and_copies_on_resume(
    tmp_path: Path, monkeypatch
) -> None:
    class RepairingMessages:
        def __init__(self) -> None:
            self.calls = 0

        def create(self, **_request):
            self.calls += 1
            payload = {
                "id": f"journal-{self.calls}",
                "content": [{
                    "type": "text",
                    "text": '{"wrong":true}' if self.calls == 1 else '{"ok":true}',
                }],
                "usage": {"input_tokens": 10 + self.calls, "output_tokens": 4 + self.calls},
            }
            return SimpleNamespace(model_dump=lambda: payload)

    messages = RepairingMessages()
    monkeypatch.setattr(
        provider_client,
        "anthropic_client",
        lambda: SimpleNamespace(messages=messages),
    )
    recorder = RunRecorder(tmp_path, "source-run", {"budgets": {}})
    backend = ModelProviderBackend(
        attempt_sink=lambda attempt: recorder.record_provider_attempt(
            "generator", attempt
        )
    )
    schema = {
        "type": "object",
        "properties": {"ok": {"type": "boolean"}},
        "required": ["ok"],
        "additionalProperties": False,
    }
    request = {
        "model": "anthropic/claude-opus-4-6",
        "developer": "Return the schema.",
        "user": "Exercise the durable journal.",
        "schema": schema,
        "schema_name": "provider_attempt_journal_test",
        "reasoning_effort": "low",
    }
    response = backend.structured(**request)
    service = ServiceEvent(
        kind="model.structured",
        request_hash=_request_hash("model.structured", request),
        response=response,
        request=request,
        metadata=backend.last_call_metadata(),
    )
    recorder.record_service("generator", service)

    provider_path = recorder.root / "provider-attempts.private.jsonl"
    service_path = recorder.root / "service-calls.private.jsonl"
    assert validate_provider_attempt_journal(provider_path)["provider_calls"] == 2
    reconciled = reconcile_provider_attempt_journal(provider_path, service_path)
    assert reconciled["logical_calls"] == 1
    assert reconciled["provider_calls"] == 2
    assert recorder.provider_attempt_journal.sequence == 4

    forged_service_path = recorder.root / "forged-service.private.jsonl"
    forged_service = _service_payload(service)
    forged_request = copy.deepcopy(request)
    forged_request["model"] = "gpt-5.5"
    forged_service["request"] = forged_request
    forged_service["request_hash"] = _request_hash(
        "model.structured", forged_request
    )
    forged_service["metadata"]["model"] = "gpt-5.5"
    forged_service["metadata"]["request_sha"] = _req_sha(
        "gpt-5.5",
        forged_request["developer"],
        forged_request["user"],
        forged_request["schema_name"],
        forged_request["schema"],
        8000,
        "low",
    )
    HashChainWriter(forged_service_path).append({
        "kind": "service_call",
        "role": "generator",
        "service": forged_service,
    })
    with pytest.raises(ReplayDivergence, match="diverges"):
        reconcile_provider_attempt_journal(provider_path, forged_service_path)

    derived = RunRecorder(tmp_path, "derived-run", {"budgets": {}})
    for record in verify_hash_chain(service_path):
        payload = {
            key: value
            for key, value in record.items()
            if key not in {"sequence", "previous_hash", "event_hash", "recorded_at"}
        }
        derived.copy_service_record(payload)
    for record in verify_hash_chain(provider_path):
        payload = {
            key: value
            for key, value in record.items()
            if key not in {"sequence", "previous_hash", "event_hash", "recorded_at"}
        }
        derived.copy_provider_attempt_record(payload)
    checkpoint = {
        "schema_version": 3,
        "run_id": "source-run",
        "private_event_count": 0,
        "public_event_count": 0,
        "service_event_count": 1,
        "provider_attempt_event_count": 4,
        "engine": {"phase": "generator_output", "k": 0.0},
        "branches": {"run_id": "source-run", "nodes": {}},
        "resume_committed_judge_tape": [{"kind": "model.structured"}],
        "resume_judge_state": {"meter": {"model_calls": 1}},
    }
    derived.seed_resume_checkpoint(checkpoint)
    copied_checkpoint = json.loads(
        (derived.root / "checkpoint.private.json").read_text(encoding="utf-8")
    )
    assert copied_checkpoint["provider_attempt_event_count"] == 4
    assert copied_checkpoint["service_event_count"] == 1
    assert copied_checkpoint["run_id"] == "derived-run"
    assert "resume_committed_judge_tape" not in copied_checkpoint
    assert "resume_judge_state" not in copied_checkpoint


def test_failed_service_usage_must_reconcile(tmp_path: Path, monkeypatch) -> None:
    class InvalidMessages:
        def __init__(self) -> None:
            self.calls = 0

        def create(self, **_request):
            self.calls += 1
            payload = {
                "id": f"failed-{self.calls}",
                "content": [{"type": "text", "text": '{"wrong":true}'}],
                "usage": {"input_tokens": 10, "output_tokens": 5},
            }
            return SimpleNamespace(model_dump=lambda: payload)

    invalid_messages = InvalidMessages()
    monkeypatch.setattr(
        provider_client,
        "anthropic_client",
        lambda: SimpleNamespace(messages=invalid_messages),
    )
    recorder = RunRecorder(tmp_path, "failed-source", {"budgets": {}})
    backend = ModelProviderBackend(
        attempt_sink=lambda attempt: recorder.record_provider_attempt(
            "generator", attempt
        )
    )
    schema = {
        "type": "object",
        "properties": {"ok": {"type": "boolean"}},
        "required": ["ok"],
        "additionalProperties": False,
    }
    request = {
        "model": "anthropic/claude-opus-4-6",
        "developer": "Return the schema.",
        "user": "Remain invalid.",
        "schema": schema,
        "schema_name": "failed_provider_attempt_journal_test",
        "reasoning_effort": "low",
    }
    with pytest.raises(ValueError) as caught:
        backend.structured(**request)
    service = ServiceEvent(
        kind="model.structured",
        request_hash=_request_hash("model.structured", request),
        response=None,
        error="ValueError",
        request=request,
        error_message=str(caught.value),
        metadata=backend.last_call_metadata(),
    )
    recorder.record_service("generator", service)
    provider_path = recorder.root / "provider-attempts.private.jsonl"
    service_path = recorder.root / "service-calls.private.jsonl"
    assert reconcile_provider_attempt_journal(
        provider_path, service_path
    )["logical_successes"] == 0

    forged_path = recorder.root / "forged-failed-service.private.jsonl"
    forged = _service_payload(service)
    forged["metadata"]["usage"] = {
        "calls": 0,
        "provider_calls": 0,
        "unknown_provider_attempts": 0,
        "input_tokens": 0,
        "output_tokens": 0,
        "cost_usd": 0.0,
    }
    HashChainWriter(forged_path).append({
        "kind": "service_call",
        "role": "generator",
        "service": forged,
    })
    with pytest.raises(ReplayDivergence, match="usage delta diverges"):
        reconcile_provider_attempt_journal(provider_path, forged_path)


def test_backend_last_call_metadata_is_bound_to_calling_thread(monkeypatch) -> None:
    totals = {
        "calls": 0,
        "provider_calls": 0,
        "unknown_provider_attempts": 0,
        "input_tokens": 0,
        "output_tokens": 0,
        "cost_usd": 0.0,
    }
    provider_metadata = threading.local()

    def fake_usage_totals() -> dict:
        return dict(totals)

    def fake_structured(**request):
        totals["calls"] += 1
        totals["provider_calls"] += 1
        totals["input_tokens"] += 1
        totals["output_tokens"] += 1
        totals["cost_usd"] += 0.01
        provider_metadata.value = {
            "request_sha": request["user"],
            "provider_request_id": f"id-{request['user']}",
        }
        return {"ok": True}

    monkeypatch.setattr(provider_client, "usage_totals", fake_usage_totals)
    monkeypatch.setattr(provider_client, "structured", fake_structured)
    monkeypatch.setattr(
        provider_client,
        "last_call_metadata",
        lambda: dict(provider_metadata.value),
    )

    backend = ModelProviderBackend()
    barrier = threading.Barrier(2)
    observed: dict[str, str] = {}
    failures: list[BaseException] = []

    def call(label: str) -> None:
        try:
            backend.structured(
                model="anthropic/claude-opus-4-6",
                schema_name="thread_binding",
                user=label,
            )
            barrier.wait(timeout=5)
            observed[label] = str(backend.last_call_metadata()["request_sha"])
        except BaseException as exc:  # surface worker failures in the main thread
            failures.append(exc)

    workers = [threading.Thread(target=call, args=(label,)) for label in ("A", "B")]
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join(timeout=10)
    assert not any(worker.is_alive() for worker in workers)
    assert not failures
    assert observed == {"A": "A", "B": "B"}


def test_reconcile_allows_concurrent_service_completion_order(
    tmp_path: Path, monkeypatch
) -> None:
    class ValidMessages:
        def __init__(self) -> None:
            self.calls = 0

        def create(self, **_request):
            self.calls += 1
            payload = {
                "id": f"reordered-{self.calls}",
                "content": [{"type": "text", "text": '{"ok":true}'}],
                "usage": {"input_tokens": 10 + self.calls, "output_tokens": 5},
            }
            return SimpleNamespace(model_dump=lambda: payload)

    messages = ValidMessages()
    monkeypatch.setattr(
        provider_client,
        "anthropic_client",
        lambda: SimpleNamespace(messages=messages),
    )
    recorder = RunRecorder(tmp_path, "reordered-run", {"budgets": {}})
    backend = ModelProviderBackend(
        attempt_sink=lambda attempt: recorder.record_provider_attempt(
            "judge", attempt
        )
    )
    schema = {
        "type": "object",
        "properties": {"ok": {"type": "boolean"}},
        "required": ["ok"],
        "additionalProperties": False,
    }
    events: list[ServiceEvent] = []
    for label in ("A", "B"):
        request = {
            "model": "anthropic/claude-opus-4-6",
            "developer": "Return the schema.",
            "user": label,
            "schema": schema,
            "schema_name": "reordered_attempt_test",
            "reasoning_effort": "low",
        }
        response = backend.structured(**request)
        events.append(ServiceEvent(
            kind="model.structured",
            request_hash=_request_hash("model.structured", request),
            response=response,
            request=request,
            metadata=backend.last_call_metadata(),
        ))

    # Provider A then B is durable, but concurrent service scheduling is allowed
    # to commit B then A.
    recorder.record_service("judge", events[1])
    recorder.record_service("judge", events[0])
    summary = reconcile_provider_attempt_journal(
        recorder.root / "provider-attempts.private.jsonl",
        recorder.root / "service-calls.private.jsonl",
    )
    assert summary["logical_calls"] == 2
    assert summary["logical_successes"] == 2
    assert summary["provider_calls"] == 2


def test_hash_chain_writer_serializes_concurrent_appends(
    tmp_path: Path, monkeypatch
) -> None:
    from tech_tree_arena.replay import recorder as recorder_module

    real_time = time.time

    def widened_time() -> float:
        # Widen the interval after an unlocked writer reads its shared cursor.
        time.sleep(0.002)
        return real_time()

    monkeypatch.setattr(recorder_module.time, "time", widened_time)
    path = tmp_path / "concurrent.jsonl"
    writer = HashChainWriter(path)
    start = threading.Barrier(12)
    failures: list[BaseException] = []

    def append(index: int) -> None:
        try:
            start.wait(timeout=5)
            writer.append({"kind": "concurrent", "index": index})
        except BaseException as exc:
            failures.append(exc)

    workers = [threading.Thread(target=append, args=(index,)) for index in range(12)]
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join(timeout=10)
    assert not any(worker.is_alive() for worker in workers)
    assert not failures
    records = verify_hash_chain(path)
    assert len(records) == 12
    assert [record["sequence"] for record in records] == list(range(12))
    assert {record["index"] for record in records} == set(range(12))
    assert writer.sequence == 12
    assert writer.previous_hash == records[-1]["event_hash"]


def test_audit_tape_and_durable_sink_share_concurrent_order() -> None:
    sink_order: list[str] = []
    first_in_sink = threading.Event()
    release_first = threading.Event()

    def sink(event: ServiceEvent) -> None:
        label = str(event.response)
        if label == "A":
            first_in_sink.set()
            assert release_first.wait(timeout=5)
        sink_order.append(label)

    services = ReplayableServices(seed=1, event_sink=sink)

    def record(label: str) -> None:
        services._record_event(  # exercise the atomic audit/sink commit boundary
            ServiceEvent(
                kind="model.structured",
                request_hash=label * 64,
                response=label,
            ),
            audit=True,
        )

    first = threading.Thread(target=record, args=("A",))
    second = threading.Thread(target=record, args=("B",))
    first.start()
    assert first_in_sink.wait(timeout=5)
    second.start()
    time.sleep(0.02)
    release_first.set()
    first.join(timeout=5)
    second.join(timeout=5)
    assert not first.is_alive() and not second.is_alive()
    audit_order = [str(event.response) for event in services.export_tape()]
    assert audit_order == ["A", "B"]
    assert sink_order == audit_order


def test_provider_error_journal_never_echoes_exception_text(
    tmp_path: Path, monkeypatch
) -> None:
    secret = "credential-like-secret-value"

    class FailingMessages:
        def create(self, **_request):
            raise RuntimeError(secret + ("X" * 10_000))

    monkeypatch.setattr(
        provider_client,
        "anthropic_client",
        lambda: SimpleNamespace(messages=FailingMessages()),
    )
    recorder = RunRecorder(tmp_path, "sanitized-error", {"budgets": {}})
    backend = ModelProviderBackend(
        attempt_sink=lambda attempt: recorder.record_provider_attempt(
            "generator", attempt
        )
    )
    services = ReplayableServices(
        seed=7,
        model_backend=backend,
        event_sink=lambda event: recorder.record_service("generator", event),
    )
    with pytest.raises(RuntimeError):
        services.structured_model(
            model="anthropic/claude-opus-4-6",
            developer="Return JSON.",
            user="Synthetic public request.",
            schema={"type": "object"},
            schema_name="sanitized_error",
            reasoning_effort="low",
        )
    path = recorder.root / "provider-attempts.private.jsonl"
    raw = path.read_text(encoding="utf-8")
    assert secret not in raw
    assert ("X" * 100) not in raw
    service_raw = (recorder.root / "service-calls.private.jsonl").read_text(
        encoding="utf-8"
    )
    assert secret not in service_raw
    assert ("X" * 100) not in service_raw
    assert "service call failed" in service_raw
    records = verify_hash_chain(path)
    finished = records[-1]["attempt"]
    assert finished["error_message"] == "provider request failed"
    assert len(finished["error_message"].encode("utf-8")) <= 160


def test_completed_native_service_requires_provider_journal(tmp_path: Path) -> None:
    run_id = "missing-attempt-journal"
    service_path = tmp_path / "service-calls.private.jsonl"
    request = {
        "model": "anthropic/claude-opus-4-6",
        "developer": "Return JSON.",
        "user": "Synthetic public request.",
        "schema": {"type": "object"},
        "schema_name": "missing_attempt_journal",
    }
    HashChainWriter(service_path).append({
        "kind": "service_call",
        "role": "generator",
        "service": {
            "kind": "model.structured",
            "request_hash": _request_hash("model.structured", request),
            "response": {},
            "error": None,
            "request": request,
            "error_message": None,
            "started_at": 1.0,
            "finished_at": 2.0,
            "metadata": {
                "provider_attempts": [{"attempt": 1}],
                "usage": {"provider_calls": 1},
            },
        },
    })
    (tmp_path / "service-accounting.private.json").write_text(
        json.dumps({
            "run_id": run_id,
            "service_event_count": 1,
            "provider_attempt_event_count": 0,
        }),
        encoding="utf-8",
    )
    (tmp_path / "usage.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ReplayDivergence, match="lack a durable provider-attempt"):
        _validate_completed_service_accounting(
            tmp_path,
            {"run_id": run_id},
            seed=1,
        )


def test_new_logical_call_resets_attempt_after_exhausted_repairs(
    tmp_path: Path, monkeypatch
) -> None:
    class ExhaustThenRecover:
        def __init__(self) -> None:
            self.calls = 0

        def create(self, **_request):
            self.calls += 1
            text = '{"wrong":true}' if self.calls <= 4 else '{"ok":true}'
            payload = {
                "id": f"reset-{self.calls}",
                "content": [{"type": "text", "text": text}],
                "usage": {"input_tokens": 11, "output_tokens": 5},
            }
            return SimpleNamespace(model_dump=lambda: payload)

    messages = ExhaustThenRecover()
    monkeypatch.setattr(
        provider_client,
        "anthropic_client",
        lambda: SimpleNamespace(messages=messages),
    )
    recorder = RunRecorder(tmp_path, "attempt-reset", {"budgets": {}})
    backend = ModelProviderBackend(
        attempt_sink=lambda attempt: recorder.record_provider_attempt(
            "generator", attempt
        )
    )
    schema = {
        "type": "object",
        "properties": {"ok": {"type": "boolean"}},
        "required": ["ok"],
        "additionalProperties": False,
    }

    def request(label: str) -> dict:
        return {
            "model": "anthropic/claude-opus-4-6",
            "developer": "Return the schema.",
            "user": label,
            "schema": schema,
            "schema_name": "attempt_reset",
            "reasoning_effort": "low",
        }

    first_request = request("identical-request")
    with pytest.raises(ValueError) as caught:
        backend.structured(**first_request)
    recorder.record_service("generator", ServiceEvent(
        kind="model.structured",
        request_hash=_request_hash("model.structured", first_request),
        response=None,
        error="ValueError",
        request=first_request,
        error_message="service call failed",
        metadata=backend.last_call_metadata(),
    ))

    # A separate logical service call may repeat the exact request hash.  Its
    # provider-attempt numbering must still restart at one.
    second_request = request("identical-request")
    response = backend.structured(**second_request)
    recorder.record_service("generator", ServiceEvent(
        kind="model.structured",
        request_hash=_request_hash("model.structured", second_request),
        response=response,
        request=second_request,
        metadata=backend.last_call_metadata(),
    ))
    assert str(caught.value)
    summary = reconcile_provider_attempt_journal(
        recorder.root / "provider-attempts.private.jsonl",
        recorder.root / "service-calls.private.jsonl",
    )
    assert summary["logical_calls"] == 2
    assert summary["logical_successes"] == 1
    assert summary["provider_calls"] == 5


def test_anthropic_cache_usage_is_priced_per_class_and_reconciles(
    tmp_path: Path, monkeypatch
) -> None:
    """A cached direct-path response is exactly priced, not rejected.

    This used to fail closed: the cache classes had no frozen rates, so the only safe
    move was to refuse the response.  Now each class is priced from the published
    sheet, which is what makes caching usable on the generator/judge path at all.
    """

    class CachedMessages:
        def create(self, **_request):
            payload = {
                "id": "cached-usage",
                "content": [{"type": "text", "text": '{"ok":true}'}],
                "usage": {
                    "input_tokens": 10,
                    "output_tokens": 5,
                    "cache_creation_input_tokens": 7,
                    "cache_read_input_tokens": 11,
                },
            }
            return SimpleNamespace(model_dump=lambda: payload)

    monkeypatch.setattr(
        provider_client,
        "anthropic_client",
        lambda: SimpleNamespace(messages=CachedMessages()),
    )
    recorder = RunRecorder(tmp_path, "cache-usage", {"budgets": {}})
    backend = ModelProviderBackend(
        attempt_sink=lambda attempt: recorder.record_provider_attempt(
            "generator", attempt
        )
    )
    result = backend.structured(
        model="anthropic/claude-opus-4-6",
        developer="Return JSON.",
        user="Synthetic public request.",
        schema={
            "type": "object",
            "properties": {"ok": {"type": "boolean"}},
            "required": ["ok"],
            "additionalProperties": False,
        },
        schema_name="cache_usage",
        reasoning_effort="low",
    )
    assert result == {"ok": True}

    path = recorder.root / "provider-attempts.private.jsonl"
    attempt = verify_hash_chain(path)[-1]["attempt"]
    assert attempt["schema_version"] == "arena-provider-attempt-v2"
    assert attempt["status"] == "ok"
    assert attempt["usage_available"] is True
    # input_tokens is the cache-inclusive total; the classes break it down.
    assert attempt["input_tokens"] == 10 + 11 + 7
    assert attempt["cache_read_input_tokens"] == 11
    # No cache_creation breakdown in the response, so the writes are charged to the
    # TTL this client configured rather than defaulting to the cheaper class.
    assert (
        attempt["cache_write_5m_input_tokens"],
        attempt["cache_write_1h_input_tokens"],
    ) == ((7, 0) if provider_client._ANTHROPIC_CACHE_TTL == "5m" else (0, 7))

    # Opus 4.6: $5 base in, $6.25 5m write, $10 1h write, $0.50 read, $25 out per MTok.
    write_rate = 6.25 if provider_client._ANTHROPIC_CACHE_TTL == "5m" else 10.0
    expected = (10 * 5.0 + 11 * 0.5 + 7 * write_rate + 5 * 25.0) / 1e6
    assert attempt["cost_usd"] == pytest.approx(expected)
    # Charging the cache-inclusive total at the base rate -- what a class-blind
    # estimator does -- overstates this response.
    assert (28 * 5.0 + 5 * 25.0) / 1e6 > attempt["cost_usd"]

    summary = validate_provider_attempt_journal(path)
    assert summary["usage_complete"] is True
    assert backend.usage_totals()["unknown_provider_attempts"] == 0
    assert backend.usage_totals()["cost_usd"] == pytest.approx(expected)


def test_anthropic_direct_path_caches_the_system_block(
    tmp_path: Path, monkeypatch
) -> None:
    """The breakpoint must sit on the system block, not on the volatile user turn.

    A breakpoint after the user message would be a fresh prefix on every call and would
    never be read back, turning caching into a pure 1.25x surcharge.
    """

    seen: list[dict] = []

    class RecordingMessages:
        def create(self, **request):
            seen.append(request)
            payload = {
                "id": "cache-placement",
                "content": [{"type": "text", "text": '{"ok":true}'}],
                "usage": {"input_tokens": 4, "output_tokens": 5,
                          "cache_creation_input_tokens": 6,
                          "cache_read_input_tokens": 0},
            }
            return SimpleNamespace(model_dump=lambda: payload)

    monkeypatch.setattr(
        provider_client,
        "anthropic_client",
        lambda: SimpleNamespace(messages=RecordingMessages()),
    )
    backend = ModelProviderBackend()
    backend.structured(
        model="anthropic/claude-fable-5",
        developer="Stable developer prompt reused across every call.",
        user="Volatile per-call question.",
        schema={
            "type": "object",
            "properties": {"ok": {"type": "boolean"}},
            "required": ["ok"],
            "additionalProperties": False,
        },
        schema_name="cache_placement",
        reasoning_effort="low",
    )

    system = seen[0]["system"]
    assert isinstance(system, list) and len(system) == 1
    assert system[0]["cache_control"] == {
        "type": "ephemeral",
        "ttl": provider_client._ANTHROPIC_CACHE_TTL,
    }
    assert system[0]["text"].startswith("Stable developer prompt")
    # The volatile turn stays after the breakpoint and carries no breakpoint of its own.
    for message in seen[0]["messages"]:
        assert "cache_control" not in message


def test_post_provider_budget_failure_preserves_paid_attempt(
    tmp_path: Path, monkeypatch
) -> None:
    class ValidMessages:
        def create(self, **_request):
            payload = {
                "id": "paid-before-budget-failure",
                "content": [{"type": "text", "text": '{"ok":true}'}],
                "usage": {"input_tokens": 10, "output_tokens": 5},
            }
            return SimpleNamespace(model_dump=lambda: payload)

    monkeypatch.setattr(
        provider_client,
        "anthropic_client",
        lambda: SimpleNamespace(messages=ValidMessages()),
    )
    recorder = RunRecorder(tmp_path, "post-provider-budget", {"budgets": {}})
    backend = ModelProviderBackend(
        attempt_sink=lambda attempt: recorder.record_provider_attempt(
            "generator", attempt
        )
    )
    services = ReplayableServices(
        seed=5,
        model_backend=backend,
        limits=ServiceLimits(max_model_cost_usd=1e-12),
        event_sink=lambda event: recorder.record_service("generator", event),
    )
    with pytest.raises(ResourceLimitExceeded):
        services.structured_model(
            model="anthropic/claude-opus-4-6",
            developer="Return JSON.",
            user="Synthetic public request.",
            schema={
                "type": "object",
                "properties": {"ok": {"type": "boolean"}},
                "required": ["ok"],
                "additionalProperties": False,
            },
            schema_name="post_provider_budget",
            reasoning_effort="low",
        )
    summary = reconcile_provider_attempt_journal(
        recorder.root / "provider-attempts.private.jsonl",
        recorder.root / "service-calls.private.jsonl",
    )
    assert summary["provider_calls"] == 1
    assert summary["logical_successes"] == 0
    assert summary["provider_successes"] == 1
    service = verify_hash_chain(
        recorder.root / "service-calls.private.jsonl"
    )[0]["service"]
    assert service["error"] == "ResourceLimitExceeded"
    assert service["metadata"]["usage"]["calls"] == 1


def test_reconcile_allows_interleaved_concurrent_provider_attempts(
    tmp_path: Path, monkeypatch
) -> None:
    barrier = threading.Barrier(2)

    class ConcurrentMessages:
        def __init__(self) -> None:
            self.calls = 0
            self._lock = threading.Lock()

        def create(self, **_request):
            with self._lock:
                self.calls += 1
                call = self.calls
            # Hold both requests open so their started records interleave in
            # the durable journal before either finished record lands.
            barrier.wait(timeout=10)
            payload = {
                "id": f"interleaved-{call}",
                "content": [{"type": "text", "text": '{"ok":true}'}],
                "usage": {"input_tokens": 10 + call, "output_tokens": 5},
            }
            return SimpleNamespace(model_dump=lambda: payload)

    messages = ConcurrentMessages()
    monkeypatch.setattr(
        provider_client,
        "anthropic_client",
        lambda: SimpleNamespace(messages=messages),
    )
    recorder = RunRecorder(tmp_path, "interleaved-run", {"budgets": {}})
    backend = ModelProviderBackend(
        attempt_sink=lambda attempt: recorder.record_provider_attempt(
            "generator", attempt
        )
    )
    schema = {
        "type": "object",
        "properties": {"ok": {"type": "boolean"}},
        "required": ["ok"],
        "additionalProperties": False,
    }
    events: dict[str, ServiceEvent] = {}
    errors: list[BaseException] = []

    def run_call(label: str) -> None:
        request = {
            "model": "anthropic/claude-opus-4-6",
            "developer": "Return the schema.",
            "user": label,
            "schema": schema,
            "schema_name": "interleaved_attempt_test",
            "reasoning_effort": "low",
        }
        try:
            response = backend.structured(**request)
            # Metadata is thread-bound: each call sees only its own attempts.
            metadata = backend.last_call_metadata()
            assert metadata["usage"]["provider_calls"] == 1
            assert metadata["usage"]["input_tokens"] in (11, 12)
            events[label] = ServiceEvent(
                kind="model.structured",
                request_hash=_request_hash("model.structured", request),
                response=response,
                request=request,
                metadata=metadata,
            )
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=run_call, args=(label,)) for label in ("A", "B")]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=30)
    assert not errors
    assert set(events) == {"A", "B"}

    journal = verify_hash_chain(tmp_path / "interleaved-run" / "provider-attempts.private.jsonl")
    kinds = [record["attempt"]["kind"] for record in (json.loads(json.dumps(r)) if isinstance(r, dict) else r for r in map(dict, journal))]
    # Both requests genuinely interleaved: two starts precede the finishes.
    assert kinds[:2] == ["provider_attempt_started", "provider_attempt_started"]

    for label in ("A", "B"):
        recorder.record_service("generator", events[label])
    summary = reconcile_provider_attempt_journal(
        recorder.root / "provider-attempts.private.jsonl",
        recorder.root / "service-calls.private.jsonl",
    )
    assert summary["logical_calls"] == 2
    assert summary["logical_successes"] == 2
    assert summary["provider_calls"] == 2


def test_legacy_v1_attempt_journal_still_validates(tmp_path: Path) -> None:
    """Runs recorded before direct-path caching must stay replayable.

    v1 has no cache-class fields and its input_tokens counts only uncached input, so it
    is priced at the base rate on that total.  Rejecting it would strand every existing
    trace.
    """

    recorder = RunRecorder(tmp_path, "legacy-v1", {"budgets": {}})
    common = {
        "schema_version": "arena-provider-attempt-v1",
        "provider": "anthropic",
        "model": "anthropic/claude-opus-4-6",
        "logical_request_sha": "a" * 24,
        "attempt": 1,
        "request_sha256": "b" * 64,
        "started_at": 1.0,
    }
    recorder.record_provider_attempt(
        "generator", {"kind": "provider_attempt_started", **common}
    )
    recorder.record_provider_attempt("generator", {
        "kind": "provider_attempt_finished",
        **common,
        "finished_at": 2.0,
        "status": "ok",
        "provider_request_id": "req_legacy",
        "response_sha256": "c" * 64,
        "input_tokens": 1_000,
        "output_tokens": 200,
        # Opus 4.6 at $5 in / $25 out per MTok, base rate on the whole input.
        "cost_usd": (1_000 * 5.0 + 200 * 25.0) / 1e6,
        "usage_available": True,
        "error_type": None,
        "error_message": None,
        "status_code": None,
    })

    path = recorder.root / "provider-attempts.private.jsonl"
    summary = validate_provider_attempt_journal(path)
    assert summary["usage_complete"] is True
    assert summary["input_tokens"] == 1_000
    assert summary["cost_usd"] == pytest.approx((1_000 * 5.0 + 200 * 25.0) / 1e6)


def test_v1_record_may_not_carry_cache_classes(tmp_path: Path) -> None:
    """The version tag decides the key set, so a v1 row with v2 fields is a divergence."""

    recorder = RunRecorder(tmp_path, "mixed-schema", {"budgets": {}})
    common = {
        "schema_version": "arena-provider-attempt-v1",
        "provider": "anthropic",
        "model": "anthropic/claude-opus-4-6",
        "logical_request_sha": "a" * 24,
        "attempt": 1,
        "request_sha256": "b" * 64,
        "started_at": 1.0,
    }
    recorder.record_provider_attempt(
        "generator", {"kind": "provider_attempt_started", **common}
    )
    recorder.record_provider_attempt("generator", {
        "kind": "provider_attempt_finished",
        **common,
        "finished_at": 2.0,
        "status": "ok",
        "provider_request_id": "req_mixed",
        "response_sha256": "c" * 64,
        "input_tokens": 1_000,
        "cache_read_input_tokens": 500,
        "cache_write_5m_input_tokens": 0,
        "cache_write_1h_input_tokens": 0,
        "output_tokens": 200,
        "cost_usd": 1.0,
        "usage_available": True,
        "error_type": None,
        "error_message": None,
        "status_code": None,
    })
    with pytest.raises(ReplayDivergence, match="invalid keys"):
        validate_provider_attempt_journal(recorder.root / "provider-attempts.private.jsonl")


def test_unknown_attempt_schema_version_is_rejected(tmp_path: Path) -> None:
    recorder = RunRecorder(tmp_path, "future-schema", {"budgets": {}})
    recorder.record_provider_attempt("generator", {
        "kind": "provider_attempt_started",
        "schema_version": "arena-provider-attempt-v99",
        "provider": "anthropic",
        "model": "anthropic/claude-opus-4-6",
        "logical_request_sha": "a" * 24,
        "attempt": 1,
        "request_sha256": "b" * 64,
        "started_at": 1.0,
    })
    with pytest.raises(ReplayDivergence, match="unknown schema version"):
        validate_provider_attempt_journal(
            recorder.root / "provider-attempts.private.jsonl", require_complete=False
        )



def test_failed_request_does_not_block_a_complete_usage_audit(tmp_path: Path) -> None:
    """A 529 must not make a paid run permanently unresumable.

    A request that failed outright carries no response, no provider request
    ID, and no tokens, so it is fully accounted at zero. Only a delivered
    response whose usage cannot be priced leaves a real gap in the ledger.
    """

    recorder = RunRecorder(tmp_path, "overloaded", {"budgets": {}})
    started = {
        "kind": "provider_attempt_started",
        "schema_version": "arena-provider-attempt-v2",
        "provider": "anthropic",
        "model": "anthropic/claude-opus-5",
        "logical_request_sha": "a" * 24,
        "attempt": 1,
        "request_sha256": "b" * 64,
        "started_at": 1.0,
    }
    recorder.record_provider_attempt("generator", started)
    recorder.record_provider_attempt("generator", {
        **started,
        "kind": "provider_attempt_finished",
        "finished_at": 2.0,
        "status": "error",
        "status_code": 529,
        "error_type": "OverloadedError",
        "error_message": "provider request failed",
        "input_tokens": 0,
        "output_tokens": 0,
        "cost_usd": 0.0,
        "usage_available": False,
        "response_sha256": None,
        "provider_request_id": None,
        "cache_read_input_tokens": 0,
        "cache_write_5m_input_tokens": 0,
        "cache_write_1h_input_tokens": 0,
    })
    path = recorder.root / "provider-attempts.private.jsonl"
    summary = validate_provider_attempt_journal(path, require_complete=True)
    assert summary["provider_calls"] == 1
    # The metric still reports the attempt never carried usage; only the
    # resume gate stopped treating that as an accounting hole.
    assert summary["usage_complete"] is False


def test_unpriceable_delivered_response_still_blocks_a_complete_audit(
    tmp_path: Path,
) -> None:
    recorder = RunRecorder(tmp_path, "unpriceable", {"budgets": {}})
    started = {
        "kind": "provider_attempt_started",
        "schema_version": "arena-provider-attempt-v2",
        "provider": "anthropic",
        "model": "anthropic/claude-opus-5",
        "logical_request_sha": "c" * 24,
        "attempt": 1,
        "request_sha256": "d" * 64,
        "started_at": 1.0,
    }
    recorder.record_provider_attempt("generator", started)
    recorder.record_provider_attempt("generator", {
        **started,
        "kind": "provider_attempt_finished",
        "finished_at": 2.0,
        "status": "usage_unavailable",
        "status_code": None,
        "error_type": "ProviderUsageUnavailable",
        "error_message": "provider response usage cannot be priced exactly",
        "input_tokens": 10,
        "output_tokens": 0,
        "cost_usd": 0.0,
        "usage_available": False,
        "response_sha256": "e" * 64,
        "provider_request_id": "msg_synthetic",
        "cache_read_input_tokens": 0,
        "cache_write_5m_input_tokens": 0,
        "cache_write_1h_input_tokens": 0,
    })
    path = recorder.root / "provider-attempts.private.jsonl"
    with pytest.raises(ReplayDivergence, match="unknown usage"):
        validate_provider_attempt_journal(path, require_complete=True)


def _together_response(payload: dict, *, response_id: str = "tog_1") -> dict:
    return {
        "id": response_id,
        "choices": [{"message": {"content": json.dumps(payload)}}],
        "usage": {"prompt_tokens": 120, "completion_tokens": 30},
    }


def test_together_call_records_an_auditable_provider_attempt(
    tmp_path: Path, monkeypatch
) -> None:
    """together/* runs must be promotable, which requires attempt records."""

    monkeypatch.setenv("TOGETHER_INPUT_USD_PER_MTOK", "1.4")
    monkeypatch.setenv("TOGETHER_OUTPUT_USD_PER_MTOK", "4.4")
    monkeypatch.setattr(
        provider_client,
        "_TOGETHER_DEFAULT_RATE",
        (1.4, 4.4),
        raising=False,
    )
    monkeypatch.setattr(
        provider_client,
        "_together_chat_complete",
        lambda body, timeout: _together_response({"ok": True}),
    )
    recorder = RunRecorder(tmp_path, "together-audit", {"budgets": {}})
    backend = ModelProviderBackend(
        attempt_sink=lambda attempt: recorder.record_provider_attempt(
            "generator", attempt
        )
    )
    services = ReplayableServices(
        seed=3,
        model_backend=backend,
        event_sink=lambda event: recorder.record_service("generator", event),
    )
    services.structured_model(
        model="together/test-model",
        developer="Return JSON.",
        user="Synthetic public request.",
        schema={"type": "object", "properties": {"ok": {"type": "boolean"}}},
        schema_name="together_probe",
        reasoning_effort="low",
    )
    path = recorder.root / "provider-attempts.private.jsonl"
    summary = validate_provider_attempt_journal(path, require_complete=True)
    assert summary["provider_calls"] == 1
    assert summary["usage_complete"] is True
    finished = verify_hash_chain(path)[-1]["attempt"]
    assert finished["provider"] == "together"
    assert finished["model"] == "together/test-model"
    assert finished["status"] == "ok"
    assert (finished["input_tokens"], finished["output_tokens"]) == (120, 30)
    assert finished["cost_usd"] == pytest.approx(120 / 1e6 * 1.4 + 30 / 1e6 * 4.4)


@pytest.mark.parametrize(
    ("model", "reasoning_timeout"),
    [
        ("together/zai-org/GLM-5.2", 900.0),
        ("together/zai-org/GLM-5.3", 900.0),
        ("together/moonshotai/Kimi-K3", 1800.0),
    ],
)
def test_together_models_use_max_reasoning_with_a_50k_cap(
    tmp_path: Path, monkeypatch, model: str, reasoning_timeout: float
) -> None:
    seen_bodies: list[dict] = []
    seen_timeouts: list[float] = []
    responses = [
        {
            "id": "tog_glm_reasoning",
            "choices": [{"message": {
                "reasoning_content": "The answer should be true.",
                "content": '{"ok":true}',
            }}],
            "usage": {"prompt_tokens": 100, "completion_tokens": 500},
        },
        _together_response({"ok": True}, response_id="tog_glm_good"),
        {
            "id": "tog_glm_reasoning_repeat",
            "choices": [{"message": {
                "reasoning_content": "The repeated answer should be true.",
                "content": '{"ok":true}',
            }}],
            "usage": {"prompt_tokens": 100, "completion_tokens": 500},
        },
        _together_response({"ok": True}, response_id="tog_glm_good_repeat"),
    ]

    def complete(body, timeout):
        seen_bodies.append(copy.deepcopy(body))
        seen_timeouts.append(timeout)
        return responses.pop(0)

    monkeypatch.setattr(provider_client, "_together_chat_complete", complete)
    monkeypatch.setattr(
        provider_client, "_together_chat_complete_streaming", complete
    )
    monkeypatch.setattr(
        provider_client,
        "_TOGETHER_DEFAULT_RATE",
        (1.4, 4.4),
        raising=False,
    )
    recorder = RunRecorder(tmp_path, "together-glm-two-stage", {"budgets": {}})
    backend = ModelProviderBackend(
        attempt_sink=lambda attempt: recorder.record_provider_attempt(
            "generator", attempt
        )
    )
    services = ReplayableServices(
        seed=3,
        model_backend=backend,
        event_sink=lambda event: recorder.record_service("generator", event),
    )
    parsed = services.structured_model(
        model=model,
        developer="Return JSON.",
        user="Synthetic public request.",
        schema={
            "type": "object",
            "properties": {"ok": {"type": "boolean"}},
            "required": ["ok"],
            "additionalProperties": False,
        },
        schema_name="together_glm_reasoning_probe",
        reasoning_effort="high",
    )
    parsed_repeat = services.structured_model(
        model=model,
        developer="Return JSON.",
        user="Synthetic public request.",
        schema={
            "type": "object",
            "properties": {"ok": {"type": "boolean"}},
            "required": ["ok"],
            "additionalProperties": False,
        },
        schema_name="together_glm_reasoning_probe",
        reasoning_effort="high",
    )

    assert parsed == {"ok": True}
    assert parsed_repeat == {"ok": True}
    assert len(seen_bodies) == 4
    assert seen_timeouts == [
        reasoning_timeout,
        300.0,
        reasoning_timeout,
        300.0,
    ]
    reasoning_body, formatting_body = seen_bodies[:2]
    assert reasoning_body["max_tokens"] == 50_000
    assert reasoning_body["temperature"] == 1.0
    assert reasoning_body["top_p"] == 0.95
    assert reasoning_body["reasoning"] == {"enabled": True}
    assert reasoning_body["reasoning_effort"] == "max"
    assert "response_format" not in reasoning_body

    assert formatting_body["max_tokens"] == 50_000
    assert formatting_body["temperature"] == 0
    assert formatting_body["reasoning"] == {"enabled": False}
    assert formatting_body["response_format"]["type"] == "json_schema"
    assert "The answer should be true." in formatting_body["messages"][1]["content"]
    attempts = backend.last_call_metadata()["provider_attempts"]
    assert [attempt["attempt"] for attempt in attempts] == [1, 2]
    assert sum(attempt["output_tokens"] for attempt in attempts) == 530
    provider_path = recorder.root / "provider-attempts.private.jsonl"
    service_path = recorder.root / "service-calls.private.jsonl"
    assert validate_provider_attempt_journal(
        provider_path, require_complete=True
    )["provider_calls"] == 4
    reconciled = reconcile_provider_attempt_journal(
        provider_path, service_path, require_complete=True
    )
    assert reconciled["logical_calls"] == 2
    assert reconciled["logical_successes"] == 2


def test_openrouter_call_charges_role_budget_and_reconciles(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(
        provider_client,
        "_openrouter_chat_complete",
        lambda body, timeout: _together_response(
            {"ok": True}, response_id="openrouter_1"
        ),
    )
    recorder = RunRecorder(tmp_path, "openrouter-audit", {"budgets": {}})
    backend = ModelProviderBackend(
        attempt_sink=lambda attempt: recorder.record_provider_attempt(
            "generator", attempt
        )
    )
    services = ReplayableServices(
        seed=3,
        model_backend=backend,
        event_sink=lambda event: recorder.record_service("generator", event),
    )

    result = services.structured_model(
        model="openrouter/meta/llama-4",
        developer="Return JSON.",
        user="Synthetic public request.",
        schema={
            "type": "object",
            "properties": {"ok": {"type": "boolean"}},
            "required": ["ok"],
            "additionalProperties": False,
        },
        schema_name="openrouter_probe",
        reasoning_effort="low",
    )

    assert result == {"ok": True}
    usage = backend.usage_totals()
    assert usage["provider_calls"] == 1
    assert usage["input_tokens"] == 120
    assert usage["output_tokens"] == 30
    assert usage["cost_usd"] == pytest.approx(
        120 / 1e6 * 5.0 + 30 / 1e6 * 30.0
    )
    provider_path = recorder.root / "provider-attempts.private.jsonl"
    service_path = recorder.root / "service-calls.private.jsonl"
    summary = validate_provider_attempt_journal(
        provider_path, require_complete=True
    )
    assert summary["provider_calls"] == 1
    assert reconcile_provider_attempt_journal(
        provider_path, service_path, require_complete=True
    )["logical_successes"] == 1


def test_openrouter_usage_enforces_role_monetary_budget(monkeypatch) -> None:
    monkeypatch.setattr(
        provider_client,
        "_openrouter_chat_complete",
        lambda body, timeout: _together_response(
            {"ok": True}, response_id="openrouter_budget"
        ),
    )
    backend = ModelProviderBackend()
    services = ReplayableServices(
        seed=3,
        model_backend=backend,
        limits=ServiceLimits(max_model_cost_usd=0.001),
    )

    with pytest.raises(ResourceLimitExceeded, match="monetary budget"):
        services.structured_model(
            model="openrouter/meta/llama-4",
            developer="Return JSON.",
            user="Enforce this route's budget.",
            schema={
                "type": "object",
                "properties": {"ok": {"type": "boolean"}},
                "required": ["ok"],
                "additionalProperties": False,
            },
            schema_name="openrouter_budget_probe",
            reasoning_effort="low",
        )

    assert backend.usage_totals()["cost_usd"] > 0.001


def test_together_transport_retry_is_one_audited_attempt_per_request(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv("TOGETHER_HTTP_RETRIES", "2")
    monkeypatch.setenv("TOGETHER_INPUT_USD_PER_MTOK", "1")
    monkeypatch.setenv("TOGETHER_OUTPUT_USD_PER_MTOK", "1")
    monkeypatch.setattr(provider_client.time, "sleep", lambda _seconds: None)
    calls = 0

    def complete(body, timeout):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise provider_client._ProviderTransportError(
                "synthetic transient failure", retryable=True
            )
        return _together_response({"ok": True}, response_id="tog-after-retry")

    monkeypatch.setattr(provider_client, "_together_chat_complete", complete)
    recorder = RunRecorder(tmp_path, "together-http-retry", {"budgets": {}})
    backend = ModelProviderBackend(
        attempt_sink=lambda attempt: recorder.record_provider_attempt(
            "generator", attempt
        )
    )
    services = ReplayableServices(
        seed=3,
        model_backend=backend,
        event_sink=lambda event: recorder.record_service("generator", event),
    )

    services.structured_model(
        model="together/test-model",
        developer="Return JSON.",
        user="Retry once.",
        schema={
            "type": "object",
            "properties": {"ok": {"type": "boolean"}},
            "required": ["ok"],
            "additionalProperties": False,
        },
        schema_name="together_http_retry_probe",
        reasoning_effort="low",
    )

    assert calls == 2
    assert backend.usage_totals()["provider_calls"] == 2
    provider_path = recorder.root / "provider-attempts.private.jsonl"
    service_path = recorder.root / "service-calls.private.jsonl"
    finished = [
        row["attempt"]
        for row in verify_hash_chain(provider_path)
        if row["attempt"]["kind"] == "provider_attempt_finished"
    ]
    assert [row["status"] for row in finished] == ["error", "ok"]
    assert reconcile_provider_attempt_journal(
        provider_path, service_path, require_complete=True
    )["provider_calls"] == 2


def test_together_http_failure_is_recorded_as_a_zero_cost_attempt(
    tmp_path: Path, monkeypatch
) -> None:
    def boom(body, timeout):
        raise RuntimeError("Together API HTTP 503: overloaded")

    monkeypatch.setattr(provider_client, "_together_chat_complete", boom)
    recorder = RunRecorder(tmp_path, "together-outage", {"budgets": {}})
    backend = ModelProviderBackend(
        attempt_sink=lambda attempt: recorder.record_provider_attempt(
            "generator", attempt
        )
    )
    services = ReplayableServices(
        seed=3,
        model_backend=backend,
        event_sink=lambda event: recorder.record_service("generator", event),
    )
    with pytest.raises(RuntimeError):
        services.structured_model(
            model="together/zai-org/GLM-5.2",
            developer="Return JSON.",
            user="Synthetic public request.",
            schema={"type": "object"},
            schema_name="together_outage",
            reasoning_effort="low",
        )
    path = recorder.root / "provider-attempts.private.jsonl"
    finished = verify_hash_chain(path)[-1]["attempt"]
    assert finished["status"] == "error"
    assert finished["cost_usd"] == 0.0
    assert finished["provider_request_id"] is None
    # A failed request is fully accounted at zero, so it must not strand the run.
    validate_provider_attempt_journal(path, require_complete=True)


def test_together_schema_retry_records_a_parse_invalid_attempt(
    tmp_path: Path, monkeypatch
) -> None:
    """A retried schema failure is a priced attempt, not a successful one."""

    monkeypatch.setenv("TOGETHER_INPUT_USD_PER_MTOK", "1.4")
    monkeypatch.setenv("TOGETHER_OUTPUT_USD_PER_MTOK", "4.4")
    responses = [
        _together_response({"wrong": 1}, response_id="tog_bad"),
        _together_response({"ok": True}, response_id="tog_good"),
    ]
    monkeypatch.setattr(
        provider_client,
        "_together_chat_complete",
        lambda body, timeout: responses.pop(0),
    )
    recorder = RunRecorder(tmp_path, "together-retry", {"budgets": {}})
    backend = ModelProviderBackend(
        attempt_sink=lambda attempt: recorder.record_provider_attempt(
            "generator", attempt
        )
    )
    services = ReplayableServices(
        seed=3,
        model_backend=backend,
        event_sink=lambda event: recorder.record_service("generator", event),
    )
    services.structured_model(
        model="together/test-model",
        developer="Return JSON.",
        user="Synthetic public request.",
        schema={
            "type": "object",
            "properties": {"ok": {"type": "boolean"}},
            "required": ["ok"],
            "additionalProperties": False,
        },
        schema_name="together_retry",
        reasoning_effort="low",
    )
    path = recorder.root / "provider-attempts.private.jsonl"
    finished = [
        record["attempt"]
        for record in verify_hash_chain(path)
        if record["attempt"]["kind"] == "provider_attempt_finished"
    ]
    assert [item["status"] for item in finished] == ["parse_invalid", "ok"]
    assert finished[0]["error_type"] and finished[0]["error_message"]
    assert finished[1]["error_type"] is None
    # Both attempts consumed tokens, so both are priced and the ledger holds.
    assert all(item["cost_usd"] > 0 for item in finished)
    reconcile_provider_attempt_journal(
        path,
        recorder.root / "service-calls.private.jsonl",
        require_complete=True,
    )


def test_together_read_timeout_is_retried_not_fatal(monkeypatch) -> None:
    """A read timeout arrives bare, outside URLError, and must not end a run."""

    monkeypatch.setenv("TOGETHER_API_KEY", "synthetic")
    monkeypatch.setenv("TOGETHER_HTTP_RETRIES", "3")
    monkeypatch.setattr(provider_client.time, "sleep", lambda _seconds: None)
    calls: list[int] = []

    class _Response:
        def __enter__(self):
            return self

        def __exit__(self, *_exc):
            return False

        def read(self):
            calls.append(1)
            if len(calls) == 1:
                raise TimeoutError("The read operation timed out")
            return json.dumps({
                "id": "tog",
                "choices": [{"message": {"content": '{"ok":true}'}}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 3},
            }).encode()

    monkeypatch.setattr(
        provider_client.urllib.request, "urlopen", lambda *a, **k: _Response()
    )
    durable_events = []
    with provider_client.provider_attempt_sink(durable_events.append):
        payload, parsed, attempts = provider_client._together_create_parse_retry(
            {"model": "m", "messages": [], "max_tokens": 100},
            5.0,
            "together_timeout_probe",
            {
                "type": "object",
                "properties": {"ok": {"type": "boolean"}},
                "required": ["ok"],
                "additionalProperties": False,
            },
            model="together/m",
            logical_request_sha="timeout-probe",
        )
    assert payload["id"] == "tog"
    assert parsed == {"ok": True}
    assert len(calls) == 2
    assert [attempt["status"] for attempt in attempts] == ["error", "ok"]
    assert [event["kind"] for event in durable_events] == [
        "provider_attempt_started",
        "provider_attempt_finished",
        "provider_attempt_started",
        "provider_attempt_finished",
    ]


def test_participant_actor_timeout_is_an_operator_knob(monkeypatch) -> None:
    from tech_tree_arena.cli import _participant_actor_timeout

    monkeypatch.delenv("IDEA_ARENA_ACTOR_TIMEOUT_S", raising=False)
    assert _participant_actor_timeout() == 900.0
    monkeypatch.setenv("IDEA_ARENA_ACTOR_TIMEOUT_S", "2400")
    assert _participant_actor_timeout() == 2400.0
    # A malformed or non-positive value falls back rather than disabling the wall.
    monkeypatch.setenv("IDEA_ARENA_ACTOR_TIMEOUT_S", "not-a-number")
    assert _participant_actor_timeout() == 900.0
    monkeypatch.setenv("IDEA_ARENA_ACTOR_TIMEOUT_S", "0")
    assert _participant_actor_timeout() == 900.0


def test_abandoned_call_is_metered_but_not_charged_to_the_actor() -> None:
    """A retried call's cost stays in the ledger, not in the actor's usage.

    --retry-interrupted-call keeps the failed call's cost and audit record,
    but the actor is rebuilt from the replayable tape, which excludes it. The
    checkpoint audit must therefore expect the actor's usage to match the
    replayable events while its meter still counts the attempt -- otherwise no
    run resumed through that flag could ever be promoted.
    """

    from tech_tree_arena.replay.recorder import (
        _branch_rng_state,
        _validate_service_state,
    )
    from tech_tree_arena.runtime.services import ServiceEvent

    def _event(cost: float) -> ServiceEvent:
        return ServiceEvent(
            kind="model.structured",
            request_hash="a" * 64,
            response={},
            error=None,
            error_message=None,
            metadata={"usage": {"cost_usd": cost, "turns": 1}},
        )

    completed = (_event(1.0), _event(2.0))
    abandoned = _event(0.5)
    state = {
        # The meter counted all three attempts; only the two that completed
        # reached the actor's usage.
        "meter": {"model_calls": 3, "random_calls": 0},
        "model_usage": {"cost_usd": 3.0, "turns": 2},
        "rng_state": _branch_rng_state(4, "root"),
    }
    # An agent CLI backend counts only completed turns.
    _validate_service_state(
        state,
        [],
        source="checkpoint current oracle",
        root_seed=4,
        branch_id="root",
        committed_events=(*completed, abandoned),
        usage_events=completed,
    )
    # A direct provider backend accumulates every attempt it made, so the same
    # audit must accept the abandoned call's tokens in the actor's usage.
    every_attempt = dict(state)
    every_attempt["model_usage"] = {"cost_usd": 3.5, "turns": 3}
    _validate_service_state(
        every_attempt,
        [],
        source="checkpoint current generator",
        root_seed=4,
        branch_id="root",
        committed_events=(*completed, abandoned),
        usage_events=completed,
    )
    # Anything that is neither exact total is still a divergence.
    drifted = dict(state)
    drifted["model_usage"] = {"cost_usd": 3.25, "turns": 2}
    with pytest.raises(ReplayDivergence, match="model usage disagrees"):
        _validate_service_state(
            drifted,
            [],
            source="checkpoint current oracle",
            root_seed=4,
            branch_id="root",
            committed_events=(*completed, abandoned),
            usage_events=completed,
        )


def test_proxy_service_usage_snapshot_is_not_double_counted() -> None:
    """A judge proxy may expose actor usage without spending another turn."""

    from tech_tree_arena.replay.recorder import (
        _branch_rng_state,
        _validate_service_state,
    )
    from tech_tree_arena.runtime.services import ServiceEvent

    model_event = ServiceEvent(
        kind="model.structured",
        request_hash="a" * 64,
        response={},
        error=None,
        error_message=None,
        metadata={"usage": {"cost_usd": 1.25, "calls": 1}},
    )
    proxy_event = ServiceEvent(
        kind="judge.evaluate",
        request_hash="b" * 64,
        response={},
        error=None,
        error_message=None,
        # This is the prior actor call's usage snapshot, not incremental use.
        metadata={"usage": {"cost_usd": 1.25, "calls": 1}},
    )
    state = {
        "meter": {"model_calls": 1, "random_calls": 0},
        "model_usage": {"cost_usd": 1.25, "calls": 1},
        "rng_state": _branch_rng_state(4, "root"),
    }

    _validate_service_state(
        state,
        [],
        source="checkpoint current oracle",
        root_seed=4,
        branch_id="root",
        committed_events=(model_event, proxy_event),
        usage_events=(model_event, proxy_event),
    )


def test_anthropic_overload_is_retried_and_every_attempt_is_journaled(
    tmp_path: Path, monkeypatch
) -> None:
    """A 529 must not end a paid run; it is the same class as an OpenAI 503."""

    class Overloaded(Exception):
        status_code = 529

    calls: list[int] = []

    class Messages:
        def create(self, **_request):
            calls.append(1)
            if len(calls) == 1:
                raise Overloaded("Overloaded")
            return SimpleNamespace(
                model_dump=lambda: {
                    "id": "msg_retry",
                    "content": [{"type": "text", "text": json.dumps({"ok": True})}],
                    "usage": {"input_tokens": 100, "output_tokens": 20},
                }
            )

    monkeypatch.setattr(provider_client.time, "sleep", lambda _s: None)
    monkeypatch.setattr(
        provider_client, "anthropic_client", lambda: SimpleNamespace(messages=Messages())
    )
    recorder = RunRecorder(tmp_path, "overload-retry", {"budgets": {}})
    backend = ModelProviderBackend(
        attempt_sink=lambda attempt: recorder.record_provider_attempt(
            "generator", attempt
        )
    )
    services = ReplayableServices(
        seed=5,
        model_backend=backend,
        event_sink=lambda event: recorder.record_service("generator", event),
    )
    services.structured_model(
        model="anthropic/claude-opus-4-6",
        developer="Return JSON.",
        user="Synthetic public request.",
        schema={"type": "object", "properties": {"ok": {"type": "boolean"}}},
        schema_name="overload_retry",
        reasoning_effort="low",
    )
    assert len(calls) == 2
    finished = [
        record["attempt"]
        for record in verify_hash_chain(
            recorder.root / "provider-attempts.private.jsonl"
        )
        if record["attempt"]["kind"] == "provider_attempt_finished"
    ]
    # The failed attempt stays in the ledger at zero cost beside the one that
    # served the call, so the retry is auditable rather than invisible.
    assert [item["status"] for item in finished] == ["error", "ok"]
    assert finished[0]["status_code"] == 529
    assert finished[0]["cost_usd"] == 0.0


def test_anthropic_non_transient_error_still_fails_immediately(
    tmp_path: Path, monkeypatch
) -> None:
    class BadRequest(Exception):
        status_code = 400

    calls: list[int] = []

    class Messages:
        def create(self, **_request):
            calls.append(1)
            raise BadRequest("invalid_request_error")

    monkeypatch.setattr(provider_client.time, "sleep", lambda _s: None)
    monkeypatch.setattr(
        provider_client, "anthropic_client", lambda: SimpleNamespace(messages=Messages())
    )
    recorder = RunRecorder(tmp_path, "bad-request", {"budgets": {}})
    backend = ModelProviderBackend(
        attempt_sink=lambda attempt: recorder.record_provider_attempt(
            "generator", attempt
        )
    )
    services = ReplayableServices(
        seed=5,
        model_backend=backend,
        event_sink=lambda event: recorder.record_service("generator", event),
    )
    with pytest.raises(Exception):
        services.structured_model(
            model="anthropic/claude-opus-4-6",
            developer="Return JSON.",
            user="Synthetic public request.",
            schema={"type": "object"},
            schema_name="bad_request",
            reasoning_effort="low",
        )
    assert len(calls) == 1


def _expectations(**overrides: object) -> "_ActorProtocolExpectations":
    from tech_tree_arena.replay.recorder import _ActorProtocolExpectations

    base: dict[str, object] = {
        "generator_branches": {"root": ()},
        "generator_branch_order": ("root",),
        "node_generator_refs": {},
        "current_generator_branch": "root",
        "generator_pending": False,
        "pending_generator_input": None,
        "guide_calls": (),
        "guide_pending": False,
        "pending_guide_input": None,
        "judge_pending": False,
    }
    base.update(overrides)
    return _ActorProtocolExpectations(**base)  # type: ignore[arg-type]


def _presented() -> object:
    from tech_tree_arena.contract.messages import (
        Option,
        PresentedQuestion,
        Question,
    )

    question = Question("which axis?", (Option("mode-mc", None, Fraction(1, 2)),))
    return PresentedQuestion("q1", question)


def test_an_interrupted_guide_judge_call_is_a_recognised_protocol_position() -> None:
    """The Oracle's own Judge call is a Judge call, with nothing committed.

    ``judge_pending`` tracks only the formal judgment of a committed
    submission. The Oracle may reach the Judge at any point in its turn, so a
    run that dies mid-call must still be resumable with the in-flight call
    retried; recognising ``judge_pending`` alone would reject it.
    """

    from tech_tree_arena.replay.recorder import _judge_call_is_pending

    mid_turn = _expectations(
        guide_pending=True,
        pending_guide_input=_presented(),
    )
    assert _judge_call_is_pending(mid_turn) is True


def test_no_judge_call_is_outstanding_outside_those_two_seats() -> None:
    """Between turns, and with nothing submitted, the Judge seat is empty."""

    from tech_tree_arena.replay.recorder import _judge_call_is_pending

    assert _judge_call_is_pending(_expectations()) is False
    assert _judge_call_is_pending(_expectations(judge_pending=True)) is True


def test_an_agent_guide_wall_outwaits_its_own_turn_budget() -> None:
    """Raising the oracle's turn budget must actually raise its wall.

    The actor wall used to be a flat participant default for every backend
    except ``human``, so a longer ``--oracle-agent-timeout-seconds`` was
    silently overridden: the wall killed the turn before the backend budget
    it was meant to permit.
    """

    from tech_tree_arena.cli import _guide_actor_timeout

    long_turn = {"backend": "codex", "timeout_seconds": 1800.0}
    assert _guide_actor_timeout(long_turn) >= 1800.0

    short_turn = {"backend": "codex", "timeout_seconds": 120.0}
    assert _guide_actor_timeout(short_turn) == _guide_actor_timeout(None)

    human = {"backend": "human", "timeout_seconds": 86_400.0}
    assert _guide_actor_timeout(human) > 86_400.0


def test_an_unpriceable_response_is_a_hard_stop_not_a_retry(
    tmp_path: Path, monkeypatch
) -> None:
    """Retrying an unpriceable response could only burn the same tokens again.

    The backend refuses a logical success whose attempts contain usage it
    cannot price, so an in-process retry after one can never rescue the call.
    The call fails once, the attempt stays in the ledger, and recovery is left
    to resume, where --retry-interrupted-call makes an operator own the gap.
    """

    calls: list[int] = []

    class Messages:
        def create(self, **_request):
            calls.append(1)
            return SimpleNamespace(
                model_dump=lambda: {
                    "id": f"msg_{len(calls)}",
                    "content": [{"type": "text", "text": json.dumps({"ok": True})}],
                }
            )

    monkeypatch.setattr(provider_client.time, "sleep", lambda _s: None)
    monkeypatch.setattr(
        provider_client, "anthropic_client", lambda: SimpleNamespace(messages=Messages())
    )
    recorder = RunRecorder(tmp_path, "unpriceable-stop", {"budgets": {}})
    backend = ModelProviderBackend(
        attempt_sink=lambda attempt: recorder.record_provider_attempt(
            "generator", attempt
        )
    )
    with pytest.raises(ValueError, match="auditable provider usage"):
        backend.structured(
            model="anthropic/claude-opus-4-6",
            developer="Return JSON.",
            user="Synthetic public request.",
            schema={"type": "object", "properties": {"ok": {"type": "boolean"}}},
            schema_name="unpriceable_stop",
            reasoning_effort="low",
        )
    assert len(calls) == 1

    finished = [
        record["attempt"]
        for record in verify_hash_chain(
            recorder.root / "provider-attempts.private.jsonl"
        )
        if record["attempt"]["kind"] == "provider_attempt_finished"
    ]
    assert [item["status"] for item in finished] == ["usage_unavailable"]


def test_anthropic_zero_output_after_invalid_json_can_repair_and_reconcile(tmp_path, monkeypatch):
    """Regression for the dev40 Oracle: parse error, zero output, valid repair."""
    calls = []

    class Messages:
        def create(self, **request):
            calls.append(request)
            index = len(calls)
            payload = {
                "id": f"zero-repair-{index}",
                "content": [{"type": "text", "text": ["invalid JSON", "", '{"ok":true}'][index - 1]}],
                "usage": {"input_tokens": 10, "cache_read_input_tokens": 7,
                          "output_tokens": [5, 0, 3][index - 1]},
            }
            return SimpleNamespace(model_dump=lambda: payload)

    monkeypatch.setattr(provider_client, "anthropic_client", lambda: SimpleNamespace(messages=Messages()))
    recorder = RunRecorder(tmp_path, "zero-repair", {"budgets": {}})
    backend = ModelProviderBackend(attempt_sink=lambda a: recorder.record_provider_attempt("oracle", a))
    services = ReplayableServices(seed=1, model_backend=backend,
        event_sink=lambda event: recorder.record_service("oracle", event))
    output = services.structured_model(model="anthropic/claude-opus-4-6", developer="Return JSON.",
        user="Synthetic request.", schema={"type": "object", "required": ["ok"],
            "properties": {"ok": {"type": "boolean"}}}, schema_name="zero_repair", reasoning_effort="low")
    assert output == {"ok": True}
    attempts = backend.last_call_metadata()["provider_attempts"]
    assert [a["status"] for a in attempts] == ["parse_invalid", "parse_invalid", "ok"]
    assert attempts[1]["output_tokens"] == 0 and attempts[1]["usage_fields_valid"] is True
    assert all(a["usage_available"] for a in attempts)
    summary = reconcile_provider_attempt_journal(recorder.root / "provider-attempts.private.jsonl",
                                                recorder.root / "service-calls.private.jsonl")
    assert summary["provider_calls"] == 3 and summary["logical_successes"] == 1
    assert summary["cost_usd"] == pytest.approx((3 * (10 * 5 + 7 * .5) + 8 * 25) / 1e6)
    assert backend.last_call_metadata()["usage"]["unknown_provider_attempts"] == 0


@pytest.mark.parametrize("output", ["missing", None, -1, False, "0", {}])
def test_invalid_output_usage_is_not_relabelled_as_reported_zero(tmp_path, monkeypatch, output):
    calls = []

    class Messages:
        def create(self, **request):
            calls.append(request)
            usage = {"input_tokens": 10}
            if output != "missing":
                usage["output_tokens"] = output
            payload = {"id": "unknown-output", "content": [{"type": "text", "text": '{"ok":true}'}],
                       "usage": usage}
            return SimpleNamespace(model_dump=lambda: payload)

    monkeypatch.setattr(provider_client, "anthropic_client", lambda: SimpleNamespace(messages=Messages()))
    recorder = RunRecorder(tmp_path, "unknown-output", {"budgets": {}})
    backend = ModelProviderBackend(attempt_sink=lambda a: recorder.record_provider_attempt("oracle", a))
    with pytest.raises(ValueError, match="auditable provider usage"):
        backend.structured(model="anthropic/claude-opus-4-6", developer="JSON.", user="Synthetic.",
            schema={"type": "object"}, schema_name="unknown_output", reasoning_effort="low")
    assert len(calls) == 1
    metadata = backend.last_call_metadata()
    assert metadata["usage"]["unknown_provider_attempts"] == 1
    assert metadata["provider_attempts"][0]["usage_fields_valid"] is False
    with pytest.raises(ReplayDivergence, match="unknown usage"):
        validate_provider_attempt_journal(recorder.root / "provider-attempts.private.jsonl")
