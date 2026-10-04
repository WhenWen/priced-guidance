import json

import pytest

from tech_tree_arena.runtime.anthropic_responses_proxy import (
    TranslationError,
    anthropic_to_response,
    response_sse,
    responses_to_anthropic,
)


_SCHEMA = {
    "type": "object",
    "properties": {"action": {"type": "string"}},
    "required": ["action"],
    "additionalProperties": False,
}

_SPAWN_INPUT = {
    "message": "Registered role: generator_mc",
    "fork_context": False,
}

_COLLABORATION_NAMESPACE = {
    "type": "namespace",
    "name": "multi_agent_v1",
    "description": "Tools for spawning and managing sub-agents.",
    "tools": [
        {
            "type": "function",
            "name": name,
            "description": f"Use {name}.",
            "strict": False,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
                "additionalProperties": False,
            },
        }
        for name, properties, required in (
            (
                "spawn_agent",
                {"message": {"type": "string"}},
                ["message"],
            ),
            (
                "wait_agent",
                {"targets": {"type": "array", "items": {"type": "string"}}},
                ["targets"],
            ),
            (
                "close_agent",
                {"target": {"type": "string"}},
                ["target"],
            ),
            (
                "send_input",
                {"target": {"type": "string"}, "message": {"type": "string"}},
                ["target"],
            ),
        )
    ],
}


def test_responses_request_translates_to_cached_anthropic_messages() -> None:
    previous = anthropic_to_response(
        {"model": "claude-fable-5"},
        {
            "id": "msg_previous",
            "model": "claude-fable-5",
            "content": [
                {"type": "thinking", "thinking": "summary", "signature": "signed"},
                {"type": "text", "text": '{"action":"first"}'},
            ],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 10, "output_tokens": 4},
        },
    )
    request = {
        "model": "anthropic/claude-fable-5",
        "instructions": "base instructions",
        "input": [
            {
                "type": "message",
                "role": "developer",
                "content": [{"type": "input_text", "text": "oracle policy"}],
            },
            {"type": "message", "role": "user", "content": "first"},
            previous["output"][0],
            previous["output"][1],
            {"type": "message", "role": "user", "content": "second"},
        ],
        "reasoning": {"effort": "xhigh"},
        "text": {
            "format": {
                "type": "json_schema",
                "name": "oracle_action",
                "strict": True,
                "schema": _SCHEMA,
            }
        },
        "tools": [{"type": "function", "name": "forbidden"}],
    }

    translated, diagnostics = responses_to_anthropic(request, cache_ttl="1h")

    assert translated["model"] == "claude-fable-5"
    assert translated["system"] == "base instructions\n\noracle policy"
    assert translated["cache_control"] == {"type": "ephemeral", "ttl": "1h"}
    assert translated["thinking"] == {"type": "adaptive", "display": "summarized"}
    assert translated["output_config"] == {
        "effort": "xhigh",
        "format": {"type": "json_schema", "schema": _SCHEMA},
    }
    assert [message["role"] for message in translated["messages"]] == [
        "user",
        "assistant",
        "user",
    ]
    assert translated["messages"][1]["content"][0] == {
        "type": "thinking",
        "thinking": "summary",
        "signature": "signed",
    }
    assert translated["messages"][1]["content"][1]["text"] == '{"action":"first"}'
    assert diagnostics["ignored_codex_tools"] == 1
    assert diagnostics["json_schema_relaxed"] is False
    assert diagnostics["json_schema_upstream"] is True
    assert diagnostics["json_schema_prompt_only"] is False


def test_anthropic_output_schema_relaxes_provider_only_array_bounds() -> None:
    schema = {
        "type": "object",
        "properties": {
            "candidates": {
                "type": "array",
                "minItems": 6,
                "maxItems": 6,
                "items": {
                    "type": "number",
                    "exclusiveMinimum": 0,
                    "exclusiveMaximum": 1,
                },
            }
        },
        "required": ["candidates"],
        "additionalProperties": False,
    }
    request = {
        "input": [{"type": "message", "role": "user", "content": "generate"}],
        "text": {
            "format": {
                "type": "json_schema",
                "name": "fixed_candidates",
                "strict": True,
                "schema": schema,
            }
        },
    }

    translated, diagnostics = responses_to_anthropic(request)

    relaxed = translated["output_config"]["format"]["schema"]
    candidates = relaxed["properties"]["candidates"]
    assert "minItems" not in candidates
    assert "maxItems" not in candidates
    assert "exclusiveMinimum" not in candidates["items"]
    assert "exclusiveMaximum" not in candidates["items"]
    assert schema["properties"]["candidates"]["minItems"] == 6
    assert diagnostics["json_schema_relaxed"] is True
    assert diagnostics["json_schema_upstream"] is True


def test_large_open_ended_schema_moves_to_prompt_when_collaboration_is_enabled() -> None:
    schema = {
        "type": "object",
        "properties": {
            "results": {
                "type": "array",
                "minItems": 6,
                "maxItems": 6,
                "items": {"type": "object"},
            }
        },
        "required": ["results"],
        "additionalProperties": False,
    }
    translated, diagnostics = responses_to_anthropic({
        "instructions": "orchestrate safely",
        "input": [{"type": "message", "role": "user", "content": "delegate"}],
        "tools": [_COLLABORATION_NAMESPACE],
        "text": {
            "format": {
                "type": "json_schema",
                "name": "generator_turn_result",
                "strict": True,
                "schema": schema,
            }
        },
    })

    assert "format" not in translated["output_config"]
    assert translated["output_config"]["effort"] == "high"
    assert '<OUTPUT_SCHEMA encoding="json">' in translated["system"]
    assert json.dumps(
        schema, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ) in translated["system"]
    assert diagnostics["json_schema"] is True
    assert diagnostics["json_schema_upstream"] is False
    assert diagnostics["json_schema_prompt_only"] is True


def test_anthropic_response_maps_reasoning_cache_usage_and_sse() -> None:
    response = anthropic_to_response(
        {
            "model": "claude-fable-5",
            "reasoning": {"effort": "high"},
            "text": {"format": {"type": "json_schema", "schema": _SCHEMA}},
        },
        {
            "id": "msg_live",
            "model": "claude-fable-5",
            "content": [
                {
                    "type": "thinking",
                    "thinking": "provider-visible reasoning summary",
                    "signature": "signature",
                },
                {"type": "text", "text": '{"action":"choose"}'},
            ],
            "stop_reason": "end_turn",
            "usage": {
                "input_tokens": 7,
                "cache_read_input_tokens": 100,
                "cache_creation_input_tokens": 9,
                "output_tokens": 20,
                "output_tokens_details": {"thinking_tokens": 13},
            },
        },
    )

    assert response["output"][0]["type"] == "reasoning"
    assert response["output"][0]["content"][0]["text"] == (
        "provider-visible reasoning summary"
    )
    assert response["output"][0]["encrypted_content"].startswith("anthropic-v1:")
    assert response["output"][1]["content"][0]["text"] == '{"action":"choose"}'
    assert response["usage"] == {
        "input_tokens": 116,
        "input_tokens_details": {
            "cached_tokens": 100,
            "cache_write_tokens": 9,
        },
        "cached_input_tokens": 100,
        "cache_write_input_tokens": 9,
        "cache_write_5m_input_tokens": 0,
        "cache_write_1h_input_tokens": 9,
        "output_tokens": 20,
        "output_tokens_details": {"reasoning_tokens": 13},
        "total_tokens": 136,
    }
    encoded = response_sse(response).decode("utf-8")
    events = [
        json.loads(line.removeprefix("data: "))
        for line in encoded.splitlines()
        if line.startswith("data: {")
    ]
    assert [event["sequence_number"] for event in events] == list(range(len(events)))
    assert any(event["type"] == "response.reasoning_text.done" for event in events)
    assert any(event["type"] == "response.output_text.done" for event in events)
    assert events[-1]["type"] == "response.completed"
    assert events[-1]["response"] == response
    assert encoded.endswith("data: [DONE]\n\n")


def test_only_collaboration_namespace_tools_cross_to_anthropic() -> None:
    translated, diagnostics = responses_to_anthropic({
        "input": [{"type": "message", "role": "user", "content": "delegate"}],
        "tools": [
            {"type": "function", "name": "update_plan", "parameters": {}},
            _COLLABORATION_NAMESPACE,
            {"type": "web_search", "external_web_access": True},
        ],
    })

    assert [tool["name"] for tool in translated["tools"]] == [
        "arena_collab__spawn_agent",
        "arena_collab__wait_agent",
        "arena_collab__close_agent",
    ]
    assert translated["tool_choice"] == {"type": "auto"}
    assert set(translated["tools"][0]["input_schema"]["properties"]) == {
        "message",
        "fork_context",
    }
    assert translated["tools"][0]["input_schema"]["required"] == [
        "message",
        "fork_context",
    ]
    assert translated["tools"][0]["input_schema"]["additionalProperties"] is False
    assert diagnostics["allowed_collaboration_tools"] == 3
    assert diagnostics["ignored_codex_tools"] == 3


def test_anthropic_collaboration_tool_use_maps_to_namespaced_responses_call() -> None:
    request = {
        "model": "claude-fable-5",
        "tools": [_COLLABORATION_NAMESPACE],
        "tool_choice": "auto",
        "parallel_tool_calls": True,
    }
    response = anthropic_to_response(
        request,
        {
            "id": "msg_tools",
            "model": "claude-fable-5",
            "content": [
                {"type": "thinking", "thinking": "delegate", "signature": "sig"},
                {
                    "type": "tool_use",
                    "id": "toolu_spawn_1",
                    "name": "arena_collab__spawn_agent",
                    "input": _SPAWN_INPUT,
                },
            ],
            "stop_reason": "tool_use",
            "usage": {"input_tokens": 10, "output_tokens": 5},
        },
    )

    call = next(item for item in response["output"] if item["type"] == "function_call")
    assert call["namespace"] == "multi_agent_v1"
    assert call["name"] == "spawn_agent"
    assert call["call_id"] == "toolu_spawn_1"
    assert json.loads(call["arguments"]) == _SPAWN_INPUT
    events = [
        json.loads(line.removeprefix("data: "))
        for line in response_sse(response).decode("utf-8").splitlines()
        if line.startswith("data: {")
    ]
    assert any(
        event["type"] == "response.function_call_arguments.done"
        for event in events
    )


def test_adapter_rejects_forbidden_collaboration_argument_fields() -> None:
    with pytest.raises(TranslationError, match="forbidden fields"):
        anthropic_to_response(
            {"tools": [_COLLABORATION_NAMESPACE]},
            {
                "id": "msg_bad_tool",
                "content": [{
                    "type": "tool_use",
                    "id": "toolu_bad",
                    "name": "arena_collab__spawn_agent",
                    "input": {
                        "message": "child task",
                        "items": [{"type": "local_image", "path": "/secret"}],
                    },
                }],
                "stop_reason": "tool_use",
                "usage": {},
            },
        )


def test_collaboration_call_and_result_history_round_trip_to_anthropic() -> None:
    request = {
        "input": [
            {"type": "message", "role": "user", "content": "delegate"},
            {
                "type": "function_call",
                "namespace": "multi_agent_v1",
                "name": "spawn_agent",
                "call_id": "toolu_spawn_1",
                "arguments": json.dumps({
                    "message": "child task",
                    "fork_context": False,
                }),
            },
            {
                "type": "function_call_output",
                "call_id": "toolu_spawn_1",
                "output": '{"agent_id":"child-1"}',
            },
        ],
        "tools": [_COLLABORATION_NAMESPACE],
    }

    translated, _ = responses_to_anthropic(request)

    assert translated["messages"][1] == {
        "role": "assistant",
        "content": [{
            "type": "tool_use",
            "id": "toolu_spawn_1",
            "name": "arena_collab__spawn_agent",
            "input": {
                "message": "child task",
                "fork_context": False,
            },
        }],
    }
    assert translated["messages"][2] == {
        "role": "user",
        "content": [{
            "type": "tool_result",
            "tool_use_id": "toolu_spawn_1",
            "content": '{"agent_id":"child-1"}',
        }],
    }


@pytest.mark.parametrize(
    "item",
    [
        {
            "type": "function_call",
            "namespace": "multi_agent_v1",
            "name": "send_input",
            "call_id": "call-1",
            "arguments": "{}",
        },
        {
            "type": "function_call",
            "namespace": "codex_app",
            "name": "spawn_agent",
            "call_id": "call-1",
            "arguments": "{}",
        },
        {
            "type": "function_call_output",
            "call_id": "unbound",
            "output": "{}",
        },
    ],
)
def test_adapter_rejects_non_collaboration_or_unbound_tool_history(
    item: dict,
) -> None:
    with pytest.raises(TranslationError):
        responses_to_anthropic({
            "input": [
                {"type": "message", "role": "user", "content": "delegate"},
                item,
                {"type": "message", "role": "user", "content": "continue"},
            ]
        })


def test_direct_adapter_rejects_tool_history() -> None:
    with pytest.raises(TranslationError, match="tool history"):
        responses_to_anthropic({
            "input": [
                {"type": "function_call", "name": "shell", "arguments": "{}"},
                {"type": "message", "role": "user", "content": "continue"},
            ]
        })


def test_models_listing_uses_native_codex_catalog_shape() -> None:
    import json as _json
    import threading
    import urllib.request
    from tech_tree_arena.runtime.anthropic_responses_proxy import _AdapterServer

    server = _AdapterServer(
        ("127.0.0.1", 0),
        anthropic_base_url="https://api.anthropic.com/v1",
        cache_ttl="1h",
        log_file=None,
        timeout_seconds=5.0,
        default_max_tokens=1024,
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        with urllib.request.urlopen(
            f"http://127.0.0.1:{server.server_port}/models", timeout=5
        ) as response:
            status = response.status
            payload = _json.loads(response.read())
    finally:
        server.shutdown()
        server.server_close()

    assert status == 200
    assert [model["slug"] for model in payload["models"]] == ["claude-fable-5"]
    assert payload["models"][0]["experimental_supported_tools"] == []


def test_http_adapter_round_trips_only_collaboration_tool_calls() -> None:
    import threading
    import urllib.request
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    from tech_tree_arena.runtime.anthropic_responses_proxy import _AdapterServer

    captured: list[dict] = []

    class UpstreamHandler(BaseHTTPRequestHandler):
        def log_message(self, _format: str, *_args: object) -> None:
            return

        def do_POST(self) -> None:  # noqa: N802
            length = int(self.headers.get("Content-Length") or 0)
            captured.append(json.loads(self.rfile.read(length)))
            payload = json.dumps({
                "id": "msg_http_tools",
                "model": "claude-fable-5",
                "content": [{
                    "type": "tool_use",
                    "id": "toolu_http_spawn",
                    "name": "arena_collab__spawn_agent",
                    "input": _SPAWN_INPUT,
                }],
                "stop_reason": "tool_use",
                "usage": {"input_tokens": 10, "output_tokens": 4},
            }).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

    upstream = ThreadingHTTPServer(("127.0.0.1", 0), UpstreamHandler)
    upstream_thread = threading.Thread(target=upstream.serve_forever, daemon=True)
    upstream_thread.start()
    adapter = _AdapterServer(
        ("127.0.0.1", 0),
        anthropic_base_url=f"http://127.0.0.1:{upstream.server_port}/v1",
        cache_ttl="1h",
        log_file=None,
        timeout_seconds=5.0,
        default_max_tokens=1024,
    )
    adapter_thread = threading.Thread(target=adapter.serve_forever, daemon=True)
    adapter_thread.start()
    try:
        body = json.dumps({
            "model": "claude-fable-5",
            "input": [{
                "type": "message",
                "role": "user",
                "content": "delegate",
            }],
            "tools": [
                {"type": "function", "name": "update_plan", "parameters": {}},
                _COLLABORATION_NAMESPACE,
                {"type": "web_search", "external_web_access": True},
            ],
        }).encode("utf-8")
        request = urllib.request.Request(
            f"http://127.0.0.1:{adapter.server_port}/responses",
            data=body,
            headers={
                "Authorization": "Bearer test-only-key",
                "Content-Type": "application/json",
            },
        )
        with urllib.request.urlopen(request, timeout=5) as response:
            encoded = response.read().decode("utf-8")
    finally:
        adapter.shutdown()
        adapter.server_close()
        upstream.shutdown()
        upstream.server_close()

    assert [tool["name"] for tool in captured[0]["tools"]] == [
        "arena_collab__spawn_agent",
        "arena_collab__wait_agent",
        "arena_collab__close_agent",
    ]
    events = [
        json.loads(line.removeprefix("data: "))
        for line in encoded.splitlines()
        if line.startswith("data: {")
    ]
    completed = events[-1]["response"]
    call = next(
        item for item in completed["output"] if item["type"] == "function_call"
    )
    assert call["namespace"] == "multi_agent_v1"
    assert call["name"] == "spawn_agent"
    assert call["call_id"] == "toolu_http_spawn"


def test_cache_write_classes_are_reported_separately() -> None:
    """5m and 1h cache writes cost 1.25x and 2x base, so they cannot be merged."""

    from tech_tree_arena.runtime.anthropic_responses_proxy import _anthropic_usage

    # When Anthropic reports the split, it is passed through verbatim.
    usage = _anthropic_usage({
        "input_tokens": 2,
        "cache_read_input_tokens": 1_800,
        "cache_creation_input_tokens": 248,
        "cache_creation": {
            "ephemeral_5m_input_tokens": 148,
            "ephemeral_1h_input_tokens": 100,
        },
        "output_tokens": 503,
    })
    assert usage["cache_write_5m_input_tokens"] == 148
    assert usage["cache_write_1h_input_tokens"] == 100
    assert usage["cache_write_input_tokens"] == 248
    assert usage["input_tokens"] == 2 + 1_800 + 248

    # Anthropic only emits the breakdown when 1h caching is in play.  With just
    # the flat total, the writes are attributed to the TTL this proxy asked for
    # -- never dropped, and never discounted to the cheaper class by default.
    for cache_ttl, expected_5m, expected_1h in (("1h", 0, 248), ("5m", 248, 0)):
        usage = _anthropic_usage(
            {
                "input_tokens": 2,
                "cache_read_input_tokens": 0,
                "cache_creation_input_tokens": 248,
                "output_tokens": 10,
            },
            cache_ttl=cache_ttl,
        )
        assert usage["cache_write_5m_input_tokens"] == expected_5m
        assert usage["cache_write_1h_input_tokens"] == expected_1h

    # A partial breakdown still has to add up to the authoritative flat total.
    usage = _anthropic_usage(
        {
            "input_tokens": 0,
            "cache_read_input_tokens": 0,
            "cache_creation_input_tokens": 500,
            "cache_creation": {"ephemeral_5m_input_tokens": 200},
            "output_tokens": 0,
        },
        cache_ttl="1h",
    )
    assert usage["cache_write_5m_input_tokens"] == 200
    assert usage["cache_write_1h_input_tokens"] == 300

    # A missing usage object must not raise.
    assert _anthropic_usage(None)["cache_write_1h_input_tokens"] == 0


def test_fully_specified_oracle_schema_stays_enforced_upstream() -> None:
    """A closed contract must not be downgraded to a prompt request.

    Prompt-only means the exact shape is asked for, not guaranteed: the model
    can answer well and still land off-shape, which the CLI reports as no
    structured action and the Arena treats as a fatal participant failure.
    """

    schema = {
        "type": "object",
        "additionalProperties": False,
        "required": [
            "reasoning",
            "state_summary",
            "action",
            "option_id",
            "question_id",
        ],
        "properties": {
            "reasoning": {"type": "string"},
            "state_summary": {"type": "string"},
            "action": {"type": "string", "enum": ["choose", "checkout"]},
            "option_id": {"type": ["string", "null"]},
            "question_id": {"type": ["string", "null"]},
        },
    }
    translated, diagnostics = responses_to_anthropic({
        "instructions": "answer as the Oracle",
        "input": [{"type": "message", "role": "user", "content": "pick one"}],
        "tools": [_COLLABORATION_NAMESPACE],
        "text": {
            "format": {
                "type": "json_schema",
                "name": "persistent_oracle_action",
                "strict": True,
                "schema": schema,
            }
        },
    })
    assert "format" in translated["output_config"]
    assert diagnostics["json_schema_upstream"] is True
    assert diagnostics["json_schema_prompt_only"] is False
    assert diagnostics["json_schema_open_ended"] is False


def test_schema_breadth_not_size_decides_prompt_only() -> None:
    from tech_tree_arena.runtime.anthropic_responses_proxy import _schema_is_open_ended

    closed = {
        "type": "object",
        "properties": {"a": {"type": "string"}},
        "required": ["a"],
        "additionalProperties": False,
    }
    assert _schema_is_open_ended(closed) is False
    # Tiny but unconstrained: arbitrary JSON hides inside the array items.
    assert _schema_is_open_ended(
        {"type": "object", "properties": {"results": {"type": "array", "items": {"type": "object"}}}}
    ) is True
    # Nested one level deeper, and behind a union.
    assert _schema_is_open_ended(
        {"type": "object", "properties": {"x": {"anyOf": [closed, {"type": "object"}]}}}
    ) is True
