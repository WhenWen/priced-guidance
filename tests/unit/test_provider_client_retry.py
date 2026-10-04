"""Tests for Together transport retry behavior."""

from __future__ import annotations

import io
import json
import urllib.error
from types import SimpleNamespace

from tech_tree_arena.runtime import provider_client


class _Response:
    def __init__(self, payload: bytes) -> None:
        self._payload = payload

    def __enter__(self) -> "_Response":
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def read(self) -> bytes:
        return self._payload


class _StreamingResponse:
    def __init__(self, lines: list[bytes]) -> None:
        self._lines = lines

    def __enter__(self) -> "_StreamingResponse":
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def __iter__(self):
        return iter(self._lines)


def test_together_retry_honors_retry_after_header(monkeypatch) -> None:
    attempts = 0

    def fake_urlopen(*_args, **_kwargs):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise urllib.error.HTTPError(
                "https://api.together.xyz/v1/chat/completions",
                429,
                "Too Many Requests",
                {"Retry-After": "7"},
                io.BytesIO(b'{"error":{"message":"rate limited"}}'),
            )
        return _Response(json.dumps({
            "id": "tog-retried",
            "choices": [{"message": {"content": '{"ok":true}'}}],
            "usage": {"prompt_tokens": 12, "completion_tokens": 4},
        }).encode())

    sleeps: list[float] = []
    monkeypatch.setenv("TOGETHER_HTTP_RETRIES", "2")
    monkeypatch.setattr(provider_client, "_load_together_key", lambda: "test-key")
    monkeypatch.setattr(provider_client.urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(provider_client.time, "sleep", sleeps.append)
    monkeypatch.setattr(provider_client.random, "uniform", lambda _start, _stop: 0.0)

    durable_events = []
    schema = {
        "type": "object",
        "properties": {"ok": {"type": "boolean"}},
        "required": ["ok"],
        "additionalProperties": False,
    }
    with provider_client.provider_attempt_sink(durable_events.append):
        response, parsed, records = provider_client._together_create_parse_retry(
            {"model": "test", "messages": [], "max_tokens": 100},
            timeout=1,
            schema_name="retry_after_probe",
            schema=schema,
            model="together/test",
            logical_request_sha="logical-retry-probe",
        )

    assert response["id"] == "tog-retried"
    assert parsed == {"ok": True}
    assert attempts == 2
    assert sleeps == [7.0]
    assert [record["status"] for record in records] == ["error", "ok"]
    assert [event["kind"] for event in durable_events] == [
        "provider_attempt_started",
        "provider_attempt_finished",
        "provider_attempt_started",
        "provider_attempt_finished",
    ]


def test_together_streaming_rebuilds_reasoning_response(monkeypatch) -> None:
    seen_request = None

    def fake_urlopen(request, **_kwargs):
        nonlocal seen_request
        seen_request = json.loads(request.data.decode())
        events = [
            {
                "id": "tog-streamed",
                "model": "moonshotai/Kimi-K3",
                "choices": [{
                    "delta": {"reasoning_content": "think "},
                    "finish_reason": None,
                }],
            },
            {
                "id": "tog-streamed",
                "model": "moonshotai/Kimi-K3",
                "choices": [{
                    "delta": {"content": "answer"},
                    "finish_reason": "stop",
                }],
            },
            {
                "id": "tog-streamed",
                "model": "moonshotai/Kimi-K3",
                "choices": [],
                "usage": {"prompt_tokens": 12, "completion_tokens": 34},
            },
        ]
        lines = [
            f"data: {json.dumps(event)}\n\n".encode()
            for event in events
        ] + [b"data: [DONE]\n\n"]
        return _StreamingResponse(lines)

    monkeypatch.setattr(provider_client, "_load_together_key", lambda: "test-key")
    monkeypatch.setattr(provider_client.urllib.request, "urlopen", fake_urlopen)

    response, parsed, records = provider_client._together_create_parse_retry(
        {"model": "moonshotai/Kimi-K3", "messages": [], "max_tokens": 100},
        timeout=1,
        schema_name="streaming_probe",
        schema=None,
        model="together/moonshotai/Kimi-K3",
        logical_request_sha="logical-streaming-probe",
        retries=1,
        streaming=True,
    )

    assert seen_request["stream"] is True
    assert seen_request["stream_options"] == {"include_usage": True}
    assert response["choices"][0]["message"] == {
        "role": "assistant",
        "content": "answer",
        "reasoning_content": "think ",
    }
    assert parsed == response
    assert records[0]["status"] == "ok"
    assert records[0]["input_tokens"] == 12
    assert records[0]["output_tokens"] == 34


def test_openai_connection_failure_is_retried_and_audited(monkeypatch) -> None:
    class APIConnectionError(Exception):
        pass

    response = SimpleNamespace(
        id="resp-test",
        usage=SimpleNamespace(input_tokens=11, output_tokens=7),
    )

    class FakeClient:
        def __init__(self) -> None:
            self.calls = 0
            self.responses = self

        def with_options(self, **_kwargs):
            return self

        def create(self, **_kwargs):
            self.calls += 1
            if self.calls == 1:
                raise APIConnectionError("temporary disconnect")
            return response

    fake = FakeClient()
    sleeps: list[float] = []
    monkeypatch.setenv("OPENAI_HTTP_RETRIES", "3")
    monkeypatch.setenv("OPENAI_RETRY_MAX_DELAY_S", "0")
    monkeypatch.setattr(provider_client, "client", lambda: fake)
    monkeypatch.setattr(provider_client.time, "sleep", sleeps.append)

    actual, attempts = provider_client._openai_response_create({"model": "test"}, timeout=1)

    assert actual is response
    assert fake.calls == 2
    assert sleeps == [0.0]
    assert attempts[0]["error_type"] == "APIConnectionError"
    assert attempts[1]["provider_request_id"] == "resp-test"
    assert attempts[1]["input_tokens"] == 11


def test_openai_http_and_parse_retries_are_all_accounted(monkeypatch) -> None:
    class APIConnectionError(Exception):
        pass

    invalid = SimpleNamespace(
        id="resp-invalid",
        output_text="{",
        usage=SimpleNamespace(input_tokens=13, output_tokens=3),
    )
    valid = SimpleNamespace(
        id="resp-valid",
        output_text='{"ok":true}',
        usage=SimpleNamespace(input_tokens=17, output_tokens=5),
    )

    class FakeClient:
        def __init__(self) -> None:
            self.calls = 0
            self.responses = self

        def with_options(self, **_kwargs):
            return self

        def create(self, **_kwargs):
            self.calls += 1
            if self.calls == 1:
                raise APIConnectionError("retryable transport")
            return invalid if self.calls == 2 else valid

    fake = FakeClient()
    monkeypatch.setenv("OPENAI_HTTP_RETRIES", "2")
    monkeypatch.setenv("OPENAI_RETRY_MAX_DELAY_S", "0")
    monkeypatch.setattr(provider_client, "client", lambda: fake)
    monkeypatch.setattr(provider_client.time, "sleep", lambda _seconds: None)
    schema = {
        "type": "object",
        "properties": {"ok": {"type": "boolean"}},
        "required": ["ok"],
        "additionalProperties": False,
    }
    before = provider_client.usage_totals()
    durable_events = []
    with provider_client.provider_attempt_sink(durable_events.append):
        result = provider_client.structured(
            model="gpt-5.5",
            developer="Return the schema.",
            user="Exercise HTTP and parse retry accounting.",
            schema=schema,
            schema_name="openai_attempt_accounting_test",
            reasoning_effort="low",
        )
    after = provider_client.usage_totals()
    metadata = provider_client.last_call_metadata()

    assert result == {"ok": True}
    assert fake.calls == 3
    assert after["provider_calls"] - before["provider_calls"] == 3
    assert (
        after["unknown_provider_attempts"]
        - before["unknown_provider_attempts"]
        == 1
    )
    assert after["calls"] - before["calls"] == 1
    attempts = metadata["provider_attempts"]
    assert [item["status"] for item in attempts] == ["error", "parse_invalid", "ok"]
    assert [item["provider_request_id"] for item in attempts] == [
        None,
        "resp-invalid",
        "resp-valid",
    ]
    assert metadata["input_tokens"] == 30
    assert metadata["output_tokens"] == 8
    assert all(len(item["request_sha256"]) == 64 for item in attempts)
    assert [item["kind"] for item in durable_events] == [
        "provider_attempt_started",
        "provider_attempt_finished",
        "provider_attempt_started",
        "provider_attempt_finished",
        "provider_attempt_started",
        "provider_attempt_finished",
    ]
    assert [item["status"] for item in durable_events[1::2]] == [
        "error",
        "parse_invalid",
        "ok",
    ]
