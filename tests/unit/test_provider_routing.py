import sys
from types import SimpleNamespace

import pytest

from tech_tree_arena.runtime import provider_client
from tech_tree_arena.runtime.provider_client import (
    _anthropic_model_name,
    _anthropic_output_config,
    _anthropic_uses_adaptive_thinking,
    _is_anthropic_model,
    _is_openrouter_model,
    _openrouter_max_tokens,
    _openrouter_model_name,
    _openrouter_schema,
    _rates,
    _validate_json_schema,
    anthropic_class_rates,
    anthropic_turn_cost,
)
from tech_tree_arena.runtime.providers import ModelProviderBackend


def test_every_routed_anthropic_model_is_priced() -> None:
    """A routed Anthropic model must never fall through to the gpt-tier default.

    `_DEFAULT_RATE` is $5/$30, which is below Fable's $10/$50, so a missed match
    silently under-reports spend instead of erring on the safe side.
    """

    # Bare ids: the codex adapter used to be the only caller that remembered to
    # prepend "anthropic/", and without it Fable priced as $5/$30.
    assert _rates("claude-fable-5") == (10.0, 50.0)
    assert _rates("claude-code/claude-fable-5") == (10.0, 50.0)
    # Date-suffixed snapshots price as their family, not as the fallback.
    assert _rates("anthropic/claude-opus-4-6-20260514") == (5.0, 25.0)
    # Models that previously had no entry at all.
    assert _rates("anthropic/claude-opus-5") == (5.0, 25.0)
    assert _rates("anthropic/claude-opus-4-8") == (5.0, 25.0)
    assert _rates("anthropic/claude-sonnet-5") == (2.0, 10.0)
    assert _rates("anthropic/claude-haiku-4-5") == (1.0, 5.0)
    # OpenRouter spells 4.6 with a dot; both spellings must agree.
    assert _rates("openrouter/anthropic/claude-opus-4.6") == _rates(
        "anthropic/claude-opus-4-6"
    )
    # An Anthropic model released after the table was frozen prices as the
    # priciest known tier, so it over-reports rather than under-reports.
    assert _rates("anthropic/claude-opus-9-unreleased") == (10.0, 50.0)
    # Non-Anthropic routing is untouched.
    assert _rates("openrouter/meta/llama-4") == (5.0, 30.0)
    assert _rates("together/some-model") == (0.0, 0.0)


def test_anthropic_class_rates_match_the_published_sheet() -> None:
    # platform.claude.com/docs/en/about-claude/pricing, verified Aug 2026:
    # (base_input, 5m write, 1h write, cache read, output) in $/MTok.
    assert anthropic_class_rates("anthropic/claude-fable-5") == (
        10.0, 12.5, 20.0, 1.0, 50.0
    )
    assert anthropic_class_rates("anthropic/claude-opus-5") == (
        5.0, 6.25, 10.0, 0.5, 25.0
    )
    assert anthropic_class_rates("anthropic/claude-haiku-4-5") == (
        1.0, 1.25, 2.0, 0.1, 5.0
    )
    # A 5m write is 1.25x base, a 1h write 2x base, a read 0.1x base.
    for model in ("anthropic/claude-fable-5", "anthropic/claude-sonnet-5"):
        base, write_5m, write_1h, read, _out = anthropic_class_rates(model)
        assert write_5m == pytest.approx(base * 1.25)
        assert write_1h == pytest.approx(base * 2.0)
        assert read == pytest.approx(base * 0.1)


def test_anthropic_turn_cost_prices_each_class_separately() -> None:
    cost = anthropic_turn_cost(
        "anthropic/claude-fable-5",
        uncached_input_tokens=1_000,
        cache_read_input_tokens=1_000_000,
        cache_write_5m_input_tokens=100_000,
        cache_write_1h_input_tokens=200_000,
        output_tokens=10_000,
    )
    expected = (
        1_000 * 10.0 + 1_000_000 * 1.0 + 100_000 * 12.5 + 200_000 * 20.0
        + 10_000 * 50.0
    ) / 1e6
    assert cost == pytest.approx(expected)

    # The two write classes are not interchangeable: charging 1h writes at the
    # 5m rate under-reports them by 1.6x.
    as_5m = anthropic_turn_cost(
        "anthropic/claude-fable-5",
        uncached_input_tokens=0,
        cache_read_input_tokens=0,
        cache_write_5m_input_tokens=200_000,
        cache_write_1h_input_tokens=0,
        output_tokens=0,
    )
    as_1h = anthropic_turn_cost(
        "anthropic/claude-fable-5",
        uncached_input_tokens=0,
        cache_read_input_tokens=0,
        cache_write_5m_input_tokens=0,
        cache_write_1h_input_tokens=200_000,
        output_tokens=0,
    )
    assert as_1h == pytest.approx(as_5m * 1.6)

    # On the read-dominated mix a real persistent oracle session produces
    # (~90% of input tokens are cache reads), pricing the cache-inclusive total
    # at the base input rate overstates the bill by more than 3x.
    read, write, out = 20_050_424, 2_167_921, 69_084
    exact = anthropic_turn_cost(
        "anthropic/claude-fable-5",
        uncached_input_tokens=0,
        cache_read_input_tokens=read,
        cache_write_5m_input_tokens=0,
        cache_write_1h_input_tokens=write,
        output_tokens=out,
    )
    flat = ((read + write) * 10.0 + out * 50.0) / 1e6
    assert flat / exact > 3.0


def test_provider_routing_and_prices() -> None:
    assert _rates("gpt-5.6-terra") == (2.5, 15.0)
    assert _rates("anthropic/claude-fable-5") == (10.0, 50.0)
    assert _rates("anthropic/claude-opus-4-6") == (5.0, 25.0)
    assert _rates("openrouter/anthropic/claude-fable-5") == (10.0, 50.0)
    assert _is_anthropic_model("anthropic/claude-opus-4-6")
    assert _anthropic_model_name("anthropic/claude-opus-4-6") == "claude-opus-4-6"
    assert _anthropic_uses_adaptive_thinking("anthropic/claude-fable-5")
    assert _anthropic_uses_adaptive_thinking("anthropic/claude-opus-5-20260801")
    assert not _anthropic_uses_adaptive_thinking("anthropic/claude-opus-4-6")
    output_config = _anthropic_output_config(
        "anthropic/claude-fable-5",
        {"type": "array", "minItems": 2, "maxItems": 4, "items": {"type": "string"}},
        "high",
    )
    assert output_config["effort"] == "high"
    assert output_config["format"]["type"] == "json_schema"
    assert "minItems" not in output_config["format"]["schema"]
    assert "maxItems" not in output_config["format"]["schema"]
    assert _is_openrouter_model("openrouter/anthropic/claude-opus-4.6")
    assert _openrouter_model_name("openrouter/anthropic/claude-opus-4.6") == "anthropic/claude-opus-4.6"
    assert _openrouter_max_tokens(4_000, "high") == 49_152


def test_anthropic_sdk_retries_are_disabled(monkeypatch) -> None:
    calls = []

    def fake_anthropic(**kwargs):
        calls.append(kwargs)
        return object()

    monkeypatch.setenv("ANTHROPIC_API_KEY", "synthetic-key-for-unit-test")
    monkeypatch.setitem(
        sys.modules,
        "anthropic",
        SimpleNamespace(Anthropic=fake_anthropic),
    )
    monkeypatch.setattr(provider_client, "_ANTHROPIC_CLIENT", None)
    provider_client.anthropic_client()
    assert calls == [{"max_retries": 0}]
    monkeypatch.setattr(provider_client, "_ANTHROPIC_CLIENT", None)


def test_openai_sdk_retries_are_disabled(monkeypatch) -> None:
    calls = []

    def fake_openai(**kwargs):
        calls.append(kwargs)
        return object()

    monkeypatch.setenv("OPENAI_API_KEY", "synthetic-key-for-unit-test")
    monkeypatch.setattr(provider_client, "OpenAI", fake_openai)
    monkeypatch.setattr(provider_client, "_CLIENT", None)
    provider_client.client()
    assert calls == [{"max_retries": 0}]
    monkeypatch.setattr(provider_client, "_CLIENT", None)


def test_openrouter_schema_relaxation() -> None:
    schema = {
        "type": "array",
        "minItems": 2,
        "maxItems": 5,
        "items": {
            "type": "number",
            "exclusiveMinimum": 0,
            "exclusiveMaximum": 1,
        },
    }
    relaxed = _openrouter_schema(schema)
    assert "minItems" not in relaxed and "maxItems" not in relaxed
    assert "exclusiveMinimum" not in relaxed["items"]
    assert "exclusiveMaximum" not in relaxed["items"]
    assert schema["minItems"] == 2 and schema["maxItems"] == 5

    _validate_json_schema([0.25, 0.75], schema)
    with pytest.raises(ValueError, match="expected value > 0"):
        _validate_json_schema([0, 0.75], schema)
    with pytest.raises(ValueError, match="expected value < 1"):
        _validate_json_schema([0.25, 1], schema)


def test_host_schema_validator_enforces_bounded_strings_and_numbers() -> None:
    schema = {
        "type": "object",
        "properties": {
            "memo": {"type": "string", "minLength": 1, "maxLength": 4},
            "count": {"type": "integer", "minimum": 1, "maximum": 3},
        },
        "required": ["memo", "count"],
        "additionalProperties": False,
    }

    _validate_json_schema({"memo": "safe", "count": 2}, schema)
    with pytest.raises(ValueError, match="at most 4 characters"):
        _validate_json_schema({"memo": "unbounded", "count": 2}, schema)
    with pytest.raises(ValueError, match="value >= 1"):
        _validate_json_schema({"memo": "safe", "count": 0}, schema)
    with pytest.raises(ValueError, match="value <= 3"):
        _validate_json_schema({"memo": "safe", "count": 4}, schema)


def test_anthropic_retained_reasoning_round_trips_signed_blocks(monkeypatch) -> None:
    class FakeMessages:
        def __init__(self) -> None:
            self.requests = []

        def create(self, **request):
            self.requests.append(request)
            index = len(self.requests)
            payload = {
                "id": f"msg-{index}",
                "content": [
                    {
                        "type": "thinking",
                        "thinking": "",
                        "signature": f"signed-thinking-{index}",
                    },
                    {"type": "text", "text": '{"ok":true}'},
                ],
                "usage": {"input_tokens": 10, "output_tokens": 5},
            }
            return SimpleNamespace(model_dump=lambda: payload)

    fake_messages = FakeMessages()
    fake_client = SimpleNamespace(messages=fake_messages)
    monkeypatch.setattr(provider_client, "anthropic_client", lambda: fake_client)
    schema = {
        "type": "object",
        "properties": {"ok": {"type": "boolean"}},
        "required": ["ok"],
        "additionalProperties": False,
    }

    first = provider_client.structured(
        model="anthropic/claude-fable-5",
        developer="Keep reasoning across turns.",
        user="First turn",
        schema=schema,
        schema_name="retained_reasoning_test",
        return_conversation_state=True,
    )
    second = provider_client.structured(
        model="anthropic/claude-fable-5",
        developer="Keep reasoning across turns.",
        user="Second turn",
        schema=schema,
        schema_name="retained_reasoning_test",
        conversation_state=first["conversation_state"],
        return_conversation_state=True,
    )

    retained = fake_messages.requests[1]["messages"][1]["content"][0]
    assert retained == {
        "type": "thinking",
        "thinking": "",
        "signature": "signed-thinking-1",
    }
    assert second["output"] == {"ok": True}
    assert len(second["conversation_state"]["messages"]) == 4


def test_anthropic_schema_retry_returns_signed_blocks_with_correction(monkeypatch) -> None:
    class RepairingMessages:
        def __init__(self) -> None:
            self.requests = []

        def create(self, **request):
            self.requests.append(request)
            attempt = len(self.requests)
            text = '{"items":[1,2,3,4]}' if attempt == 1 else '{"items":[1,2,3]}'
            payload = {
                "id": f"repair-{attempt}",
                "content": [
                    {
                        "type": "thinking",
                        "thinking": "",
                        "signature": f"repair-signature-{attempt}",
                    },
                    {"type": "text", "text": text},
                ],
                "usage": {"input_tokens": 10, "output_tokens": 5},
            }
            return SimpleNamespace(model_dump=lambda: payload)

    repairing_messages = RepairingMessages()
    monkeypatch.setattr(
        provider_client,
        "anthropic_client",
        lambda: SimpleNamespace(messages=repairing_messages),
    )
    schema = {
        "type": "object",
        "properties": {
            "items": {
                "type": "array",
                "minItems": 3,
                "maxItems": 3,
                "items": {"type": "integer"},
            }
        },
        "required": ["items"],
        "additionalProperties": False,
    }

    result = provider_client.structured(
        model="anthropic/claude-fable-5",
        developer="Return three items.",
        user="Repair locally invalid structured output.",
        schema=schema,
        schema_name="anthropic_repair_test",
    )

    assert result == {"items": [1, 2, 3]}
    retry_messages = repairing_messages.requests[1]["messages"]
    assert retry_messages[1]["role"] == "assistant"
    assert retry_messages[1]["content"][0]["signature"] == "repair-signature-1"
    assert retry_messages[2]["role"] == "user"
    assert "expected at most 3 items" in retry_messages[2]["content"]


@pytest.mark.parametrize("provider_calls", [1, 2, 3, 4])
def test_anthropic_every_repair_attempt_is_accounted(monkeypatch, provider_calls: int) -> None:
    class RepairingMessages:
        def __init__(self) -> None:
            self.calls = 0

        def create(self, **_request):
            self.calls += 1
            valid = self.calls == provider_calls
            payload = {
                "id": f"attempt-{self.calls}",
                "content": [
                    {
                        "type": "text",
                        "text": '{"ok":true}' if valid else '{"wrong":true}',
                    }
                ],
                "usage": {
                    "input_tokens": 10 + self.calls,
                    "output_tokens": 5 + self.calls,
                },
            }
            return SimpleNamespace(model_dump=lambda: payload)

    messages = RepairingMessages()
    monkeypatch.setattr(
        provider_client,
        "anthropic_client",
        lambda: SimpleNamespace(messages=messages),
    )
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
            model="anthropic/claude-opus-4-6",
            developer="Return the schema.",
            user="Exercise bounded repair.",
            schema=schema,
            schema_name="anthropic_attempt_accounting_test",
            reasoning_effort="low",
        )
    after = provider_client.usage_totals()
    metadata = provider_client.last_call_metadata()

    assert result == {"ok": True}
    assert after["provider_calls"] - before["provider_calls"] == provider_calls
    assert after["calls"] - before["calls"] == 1
    assert metadata["input_tokens"] == sum(10 + index for index in range(1, provider_calls + 1))
    assert metadata["output_tokens"] == sum(5 + index for index in range(1, provider_calls + 1))
    attempts = metadata["provider_attempts"]
    assert len(attempts) == provider_calls
    assert [item["attempt"] for item in attempts] == list(range(1, provider_calls + 1))
    assert [item["status"] for item in attempts] == [
        *("parse_invalid" for _ in range(provider_calls - 1)),
        "ok",
    ]
    assert all(len(item["request_sha256"]) == 64 for item in attempts)
    assert [item["provider_request_id"] for item in attempts] == [
        f"attempt-{index}" for index in range(1, provider_calls + 1)
    ]
    assert [item["kind"] for item in durable_events] == [
        kind
        for _ in range(provider_calls)
        for kind in ("provider_attempt_started", "provider_attempt_finished")
    ]
    for started, finished in zip(durable_events[::2], durable_events[1::2], strict=True):
        assert started["request_sha256"] == finished["request_sha256"]
        assert started["attempt"] == finished["attempt"]


def test_failed_anthropic_repairs_remain_billed_in_backend(monkeypatch) -> None:
    class AlwaysInvalidMessages:
        def __init__(self) -> None:
            self.calls = 0

        def create(self, **_request):
            self.calls += 1
            payload = {
                "id": f"invalid-{self.calls}",
                "content": [{"type": "text", "text": '{"wrong":true}'}],
                "usage": {"input_tokens": 9, "output_tokens": 4},
            }
            return SimpleNamespace(model_dump=lambda: payload)

    messages = AlwaysInvalidMessages()
    monkeypatch.setattr(
        provider_client,
        "anthropic_client",
        lambda: SimpleNamespace(messages=messages),
    )
    backend = ModelProviderBackend()
    schema = {
        "type": "object",
        "properties": {"ok": {"type": "boolean"}},
        "required": ["ok"],
        "additionalProperties": False,
    }

    with pytest.raises(ValueError, match="after 4 attempts"):
        backend.structured(
            model="anthropic/claude-opus-4-6",
            developer="Return the schema.",
            user="Remain invalid for the accounting test.",
            schema=schema,
            schema_name="anthropic_failed_attempt_accounting_test",
            reasoning_effort="low",
        )

    usage = backend.usage_totals()
    metadata = backend.last_call_metadata()
    assert messages.calls == 4
    assert usage["calls"] == 0
    assert usage["provider_calls"] == 4
    assert usage["input_tokens"] == 36
    assert usage["output_tokens"] == 16
    assert [item["status"] for item in metadata["provider_attempts"]] == [
        "parse_invalid",
        "parse_invalid",
        "parse_invalid",
        "parse_invalid",
    ]


def test_interrupted_anthropic_attempt_has_durable_unfinished_start(monkeypatch) -> None:
    class InterruptedMessages:
        def create(self, **_request):
            raise KeyboardInterrupt

    monkeypatch.setattr(
        provider_client,
        "anthropic_client",
        lambda: SimpleNamespace(messages=InterruptedMessages()),
    )
    schema = {
        "type": "object",
        "properties": {"ok": {"type": "boolean"}},
        "required": ["ok"],
        "additionalProperties": False,
    }
    durable_events = []
    with pytest.raises(KeyboardInterrupt):
        with provider_client.provider_attempt_sink(durable_events.append):
            provider_client.structured(
                model="anthropic/claude-opus-4-6",
                developer="Return the schema.",
                user="Interrupt the synthetic provider.",
                schema=schema,
                schema_name="anthropic_interruption_accounting_test",
                reasoning_effort="low",
            )
    assert len(durable_events) == 1
    assert durable_events[0]["kind"] == "provider_attempt_started"
    assert durable_events[0]["provider"] == "anthropic"


def test_backend_accounts_finish_before_sink_interrupt(monkeypatch) -> None:
    class ValidMessages:
        def create(self, **_request):
            payload = {
                "id": "finished-before-interrupt",
                "content": [{"type": "text", "text": '{"ok":true}'}],
                "usage": {"input_tokens": 12, "output_tokens": 6},
            }
            return SimpleNamespace(model_dump=lambda: payload)

    monkeypatch.setattr(
        provider_client,
        "anthropic_client",
        lambda: SimpleNamespace(messages=ValidMessages()),
    )

    def interrupt_on_finish(event):
        if event["kind"] == "provider_attempt_finished":
            raise KeyboardInterrupt

    backend = ModelProviderBackend(attempt_sink=interrupt_on_finish)
    schema = {
        "type": "object",
        "properties": {"ok": {"type": "boolean"}},
        "required": ["ok"],
        "additionalProperties": False,
    }
    with pytest.raises(KeyboardInterrupt):
        backend.structured(
            model="anthropic/claude-opus-4-6",
            developer="Return the schema.",
            user="Interrupt after the provider finish.",
            schema=schema,
            schema_name="post_finish_interrupt_accounting_test",
            reasoning_effort="low",
        )
    usage = backend.usage_totals()
    assert usage["provider_calls"] == 1
    assert usage["input_tokens"] == 12
    assert usage["output_tokens"] == 6
    assert usage["cost_usd"] > 0.0
