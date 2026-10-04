"""OpenAI Responses compatibility adapter for Anthropic's Messages API.

Codex custom providers speak the Responses wire protocol. This localhost-only
adapter translates the text-only subset used by the Arena Oracle plus a strict
collaboration-only tool subset. Only ``spawn_agent``, ``wait_agent``, and
``close_agent`` cross the adapter; all other Codex tools remain unavailable to
Anthropic. It returns standard Responses SSE events,
preserves Anthropic thinking blocks opaquely across Codex resume calls, and
never logs prompts, responses, tool arguments, or credentials.
"""

from __future__ import annotations

import argparse
import base64
import copy
import hashlib
import http.client
import json
import signal
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit


_ANTHROPIC_REASONING_PREFIX = "anthropic-v1:"
_MODEL_ID = "claude-fable-5"
_COLLABORATION_NAMESPACE = "multi_agent_v1"
_COLLABORATION_TOOLS = frozenset({"spawn_agent", "wait_agent", "close_agent"})
_ANTHROPIC_TOOL_PREFIX = "arena_collab__"
_COLLABORATION_INPUT_SCHEMAS: dict[str, dict[str, Any]] = {
    "spawn_agent": {
        "type": "object",
        "properties": {
            "message": {"type": "string", "minLength": 1, "maxLength": 1_000_000},
            "fork_context": {"type": "boolean", "enum": [False]},
        },
        "required": ["message", "fork_context"],
        "additionalProperties": False,
    },
    "wait_agent": {
        "type": "object",
        "properties": {
            "targets": {
                "type": "array",
                "items": {"type": "string", "minLength": 1},
                "minItems": 1,
                "maxItems": 16,
            },
            "timeout_ms": {
                "type": "integer",
                "minimum": 10_000,
                "maximum": 3_600_000,
            },
        },
        "required": ["targets"],
        "additionalProperties": False,
    },
    "close_agent": {
        "type": "object",
        "properties": {
            "target": {"type": "string", "minLength": 1},
        },
        "required": ["target"],
        "additionalProperties": False,
    },
}


def _model_catalog() -> dict[str, Any]:
    """Return the minimal native Codex catalog for the explicit Fable model.

    Codex refreshes ``/models`` asynchronously even when ``--model`` is
    explicit.  This endpoint uses Codex's native ``ModelsResponse`` shape so a
    successful turn cannot later be invalidated by catalog decoding errors.
    It grants no tools or capabilities; those remain fixed by the CLI policy.
    """

    return {
        "models": [{
            "slug": _MODEL_ID,
            "display_name": "Claude Fable 5",
            "description": "Arena-local Anthropic model transport.",
            "default_reasoning_level": "high",
            "supported_reasoning_levels": [{
                "effort": effort,
                "description": effort,
            } for effort in ("low", "medium", "high", "xhigh")],
            "shell_type": "shell_command",
            "visibility": "list",
            "minimal_client_version": "0.0.1",
            "supported_in_api": True,
            "priority": 1,
            "upgrade": None,
            "base_instructions": "",
            "support_verbosity": False,
            "default_verbosity": None,
            "apply_patch_tool_type": None,
            "truncation_policy": {"mode": "bytes", "limit": 10_000},
            "supports_parallel_tool_calls": True,
            "supports_image_detail_original": False,
            "context_window": 200_000,
            "max_context_window": 200_000,
            "auto_compact_token_limit": None,
            "experimental_supported_tools": [],
        }]
    }


class TranslationError(ValueError):
    """The Responses request uses a feature outside the adapter-safe subset."""


def _anthropic_tool_name(name: str) -> str:
    return _ANTHROPIC_TOOL_PREFIX + name


def _collaboration_tool_name(name: Any) -> str | None:
    value = str(name or "")
    if not value.startswith(_ANTHROPIC_TOOL_PREFIX):
        return None
    candidate = value.removeprefix(_ANTHROPIC_TOOL_PREFIX)
    return candidate if candidate in _COLLABORATION_TOOLS else None


def _validate_collaboration_arguments(name: str, value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise TranslationError("collaboration tool arguments must be an object")
    allowed = {
        "spawn_agent": {"message", "fork_context"},
        "wait_agent": {"targets", "timeout_ms"},
        "close_agent": {"target"},
    }[name]
    if set(value) - allowed:
        raise TranslationError("collaboration tool arguments contain forbidden fields")
    if name == "spawn_agent":
        message = value.get("message")
        if (
            not isinstance(message, str)
            or not message
            or len(message) > 1_000_000
            or value.get("fork_context") is not False
        ):
            raise TranslationError(
                "spawn_agent requires one registered fresh Generator child"
            )
    elif name == "wait_agent":
        targets = value.get("targets")
        if (
            not isinstance(targets, list)
            or not 1 <= len(targets) <= 16
            or any(not isinstance(target, str) or not target for target in targets)
        ):
            raise TranslationError("wait_agent requires bounded string targets")
        timeout = value.get("timeout_ms")
        if timeout is not None and (
            not isinstance(timeout, int) or not 10_000 <= timeout <= 3_600_000
        ):
            raise TranslationError("wait_agent timeout is outside its allowed range")
    else:
        target = value.get("target")
        if not isinstance(target, str) or not target:
            raise TranslationError("close_agent requires one target")
    return copy.deepcopy(value)


def _anthropic_output_schema(value: Any) -> Any:
    """Relax only JSON-Schema constraints rejected by Anthropic.

    The original Responses schema is preserved in the response envelope and is
    still enforced by Codex/the Arena after generation.  Anthropic structured
    outputs reject fixed-size arrays expressed with ``minItems > 1`` and reject
    several other numeric bounds, so sending those constraints upstream would
    fail before Fable can produce a result.
    """

    if isinstance(value, list):
        return [_anthropic_output_schema(item) for item in value]
    if not isinstance(value, dict):
        return value
    return {
        key: _anthropic_output_schema(item)
        for key, item in value.items()
        if not (
            (key == "minItems" and isinstance(item, int) and item > 1)
            or (key == "maxItems" and isinstance(item, int))
            or key in {"exclusiveMinimum", "exclusiveMaximum"}
        )
    }


def _schema_is_open_ended(schema: Any) -> bool:
    """Does the schema leave a region the upstream grammar cannot pin down?

    An object with no declared properties, or an array of such objects, admits
    arbitrary JSON. That is what makes a contract too broad to compile beside
    the collaboration tools -- not its byte size. A fully specified contract,
    however small or large, stays enforceable.
    """

    if isinstance(schema, list):
        return any(_schema_is_open_ended(item) for item in schema)
    if not isinstance(schema, dict):
        return False
    declared = schema.get("type")
    types = declared if isinstance(declared, list) else [declared]
    if "object" in types and not isinstance(schema.get("properties"), dict):
        return True
    for key in ("properties", "patternProperties", "$defs", "definitions"):
        value = schema.get(key)
        if isinstance(value, dict) and any(
            _schema_is_open_ended(item) for item in value.values()
        ):
            return True
    for key in ("items", "additionalProperties", "not"):
        if _schema_is_open_ended(schema.get(key)):
            return True
    for key in ("anyOf", "oneOf", "allOf", "prefixItems"):
        value = schema.get(key)
        if isinstance(value, list) and any(
            _schema_is_open_ended(item) for item in value
        ):
            return True
    return False


def _translated_collaboration_tools(
    raw_tools: Any,
) -> tuple[list[dict[str, Any]], int]:
    """Flatten only the approved Codex namespace into Anthropic function tools."""

    tools = raw_tools if isinstance(raw_tools, list) else []
    translated: list[dict[str, Any]] = []
    ignored = 0
    for outer in tools:
        if not isinstance(outer, dict):
            ignored += 1
            continue
        if (
            outer.get("type") != "namespace"
            or outer.get("name") != _COLLABORATION_NAMESPACE
        ):
            ignored += 1
            continue
        nested = outer.get("tools")
        if not isinstance(nested, list):
            ignored += 1
            continue
        for tool in nested:
            if not isinstance(tool, dict):
                ignored += 1
                continue
            name = str(tool.get("name") or "")
            parameters = tool.get("parameters")
            if (
                tool.get("type") != "function"
                or name not in _COLLABORATION_TOOLS
                or not isinstance(parameters, dict)
            ):
                ignored += 1
                continue
            translated.append({
                "name": _anthropic_tool_name(name),
                "description": str(tool.get("description") or ""),
                "input_schema": copy.deepcopy(
                    _COLLABORATION_INPUT_SCHEMAS[name]
                ),
            })
    return translated, ignored


def _text_from_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    chunks: list[str] = []
    for block in content:
        if isinstance(block, str):
            chunks.append(block)
            continue
        if not isinstance(block, dict):
            continue
        kind = str(block.get("type") or "")
        if kind in {"input_text", "output_text", "text", "reasoning_text"}:
            text = block.get("text")
            if isinstance(text, str):
                chunks.append(text)
        elif kind in {"input_image", "input_file", "image", "document"}:
            raise TranslationError(
                f"direct Anthropic Codex Oracle does not accept {kind} content"
            )
    return "\n".join(chunk for chunk in chunks if chunk)


def _decode_reasoning(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, str) or not value.startswith(_ANTHROPIC_REASONING_PREFIX):
        return []
    encoded = value.removeprefix(_ANTHROPIC_REASONING_PREFIX)
    try:
        padding = "=" * (-len(encoded) % 4)
        decoded = base64.urlsafe_b64decode(encoded + padding)
        blocks = json.loads(decoded)
    except (ValueError, json.JSONDecodeError, UnicodeDecodeError):
        return []
    if not isinstance(blocks, list):
        return []
    return [
        copy.deepcopy(block)
        for block in blocks
        if isinstance(block, dict)
        and block.get("type") in {"thinking", "redacted_thinking"}
    ]


def _encode_reasoning(blocks: list[dict[str, Any]]) -> str | None:
    if not blocks:
        return None
    raw = json.dumps(
        blocks, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    encoded = base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")
    return _ANTHROPIC_REASONING_PREFIX + encoded


def _append_message(
    messages: list[dict[str, Any]], role: str, content: list[dict[str, Any]]
) -> None:
    if not content:
        return
    if messages and messages[-1]["role"] == role:
        messages[-1]["content"].extend(content)
    else:
        messages.append({"role": role, "content": content})


def responses_to_anthropic(
    body: dict[str, Any],
    *,
    cache_ttl: str = "1h",
    default_max_tokens: int = 65_536,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Translate a Codex Responses request to an Anthropic Messages request."""

    if cache_ttl not in {"5m", "1h"}:
        raise TranslationError("cache_ttl must be '5m' or '1h'")
    model = str(body.get("model") or _MODEL_ID)
    if model.startswith("anthropic/"):
        model = model.removeprefix("anthropic/")

    system_parts: list[str] = []
    instructions = body.get("instructions")
    instruction_text = _text_from_content(instructions)
    if instruction_text:
        system_parts.append(instruction_text)

    input_value = body.get("input")
    items = input_value if isinstance(input_value, list) else []
    if isinstance(input_value, str):
        items = [{"type": "message", "role": "user", "content": input_value}]

    messages: list[dict[str, Any]] = []
    pending_reasoning: list[dict[str, Any]] = []
    collaboration_call_ids: set[str] = set()
    for item in items:
        if not isinstance(item, dict):
            continue
        kind = str(item.get("type") or "message")
        if kind == "reasoning":
            pending_reasoning = _decode_reasoning(item.get("encrypted_content"))
            continue
        if kind == "function_call":
            namespace = str(item.get("namespace") or "")
            name = str(item.get("name") or "")
            call_id = str(item.get("call_id") or item.get("id") or "")
            if (
                namespace != _COLLABORATION_NAMESPACE
                or name not in _COLLABORATION_TOOLS
                or not call_id
            ):
                raise TranslationError("non-collaboration tool history is forbidden")
            raw_arguments = item.get("arguments")
            if isinstance(raw_arguments, str):
                try:
                    arguments = json.loads(raw_arguments)
                except json.JSONDecodeError as exc:
                    raise TranslationError(
                        "collaboration tool arguments are not valid JSON"
                    ) from exc
            else:
                arguments = raw_arguments
            arguments = _validate_collaboration_arguments(name, arguments)
            content = list(pending_reasoning)
            pending_reasoning = []
            content.append({
                "type": "tool_use",
                "id": call_id,
                "name": _anthropic_tool_name(name),
                "input": arguments,
            })
            _append_message(messages, "assistant", content)
            collaboration_call_ids.add(call_id)
            continue
        if kind == "function_call_output":
            call_id = str(item.get("call_id") or "")
            if not call_id or call_id not in collaboration_call_ids:
                raise TranslationError(
                    "collaboration tool output has no bound tool call"
                )
            raw_output = item.get("output")
            if isinstance(raw_output, str):
                output_text = raw_output
            elif isinstance(raw_output, list):
                output_text = _text_from_content(raw_output)
            else:
                output_text = json.dumps(
                    raw_output, ensure_ascii=False, separators=(",", ":")
                )
            _append_message(messages, "user", [{
                "type": "tool_result",
                "tool_use_id": call_id,
                "content": output_text,
            }])
            continue
        if kind not in {"message", "input_text"} and not item.get("role"):
            if kind in {
                "custom_tool_call",
                "custom_tool_call_output",
            }:
                raise TranslationError("non-collaboration tool history is forbidden")
            continue
        role = str(item.get("role") or "user")
        text = _text_from_content(item.get("content") or item.get("text"))
        if role in {"developer", "system"}:
            if text:
                system_parts.append(text)
            continue
        if role not in {"user", "assistant"}:
            raise TranslationError(f"unsupported Responses message role {role!r}")
        content: list[dict[str, Any]] = []
        if role == "assistant" and pending_reasoning:
            content.extend(pending_reasoning)
            pending_reasoning = []
        if text:
            content.append({"type": "text", "text": text})
        _append_message(messages, role, content)

    if not messages or messages[-1]["role"] != "user":
        raise TranslationError("Anthropic request must end with a user message")

    translated_tools, ignored_tools = _translated_collaboration_tools(
        body.get("tools")
    )
    reasoning = body.get("reasoning")
    effort = reasoning.get("effort") if isinstance(reasoning, dict) else None
    if effort not in {"low", "medium", "high", "xhigh", "max"}:
        effort = "high"
    output_config: dict[str, Any] = {"effort": effort}
    text_config = body.get("text")
    response_format = (
        text_config.get("format") if isinstance(text_config, dict) else None
    )
    if isinstance(response_format, dict) and response_format.get("type") == "json_schema":
        schema = response_format.get("schema")
        if not isinstance(schema, dict):
            raise TranslationError("Responses json_schema format has no schema")
        anthropic_schema = _anthropic_output_schema(schema)
        output_config["format"] = {
            "type": "json_schema",
            "schema": anthropic_schema,
        }
    else:
        schema = None
        anthropic_schema = None

    schema_open_ended = _schema_is_open_ended(schema)
    schema_prompt_only = bool(
        translated_tools
        and schema is not None
        # Only a contract that actually blows the combined grammar gives up
        # upstream enforcement. A fully specified contract -- the Oracle action
        # is five named fields -- compiles alongside the collaboration tools and
        # must stay enforced: prompt-only means the exact shape is merely
        # requested, so a model that answers well but off-shape kills the run.
        and schema_open_ended
    )
    if schema_prompt_only:
        # Anthropic compiles function schemas and the response schema into one
        # grammar.  An open-ended contract can be broad enough that the
        # combined grammar exceeds Anthropic's limit.  Keep the collaboration
        # tools available and move the exact response contract into the trusted
        # system text; Codex/the Arena still receive and enforce the original
        # Responses schema after Fable returns its final JSON object.
        output_config.pop("format", None)
        system_parts.append(
            "After completing all collaboration tool calls, return exactly one "
            "JSON object matching the following host-owned output schema. Return "
            "no Markdown fences or commentary outside the JSON.\n"
            "<OUTPUT_SCHEMA encoding=\"json\">\n"
            + json.dumps(
                schema, ensure_ascii=False, sort_keys=True, separators=(",", ":")
            )
            + "\n</OUTPUT_SCHEMA>"
        )

    system_text = "\n\n".join(system_parts)

    requested_max = body.get("max_output_tokens")
    max_tokens = (
        int(requested_max)
        if isinstance(requested_max, int) and requested_max > 0
        else int(default_max_tokens)
    )
    request = {
        "model": model,
        "max_tokens": min(max_tokens, 128_000),
        "system": system_text,
        "messages": messages,
        "thinking": {"type": "adaptive", "display": "summarized"},
        "output_config": output_config,
        "cache_control": {"type": "ephemeral", "ttl": cache_ttl},
        "stream": False,
    }
    if translated_tools:
        request["tools"] = translated_tools
        request["tool_choice"] = {"type": "auto"}
    diagnostics = {
        "model": model,
        "input_items": len(items),
        "messages": len(messages),
        "system_chars": len(request["system"]),
        "effort": effort,
        "json_schema": schema is not None,
        "json_schema_upstream": "format" in output_config,
        "json_schema_open_ended": schema_open_ended,
        "json_schema_prompt_only": schema_prompt_only,
        "json_schema_relaxed": bool(
            schema is not None and anthropic_schema != schema
        ),
        "allowed_collaboration_tools": len(translated_tools),
        "ignored_codex_tools": ignored_tools,
        "cache_ttl": cache_ttl,
    }
    return request, diagnostics


def _anthropic_usage(usage: Any, *, cache_ttl: str = "1h") -> dict[str, Any]:
    """Map one Anthropic usage object onto the Responses shape, keeping billing classes.

    5m and 1h cache writes are billed at different multiples of the base input rate
    (1.25x vs 2x), so they are reported separately rather than as one ``cache_write``
    total.  Anthropic splits them in ``usage.cache_creation`` whenever 1h caching is in
    play; when only the flat total is present it is attributed to the TTL this proxy
    actually asked for, so the split can never silently collapse onto the cheaper class.
    """

    raw = usage if isinstance(usage, dict) else {}

    def integer(name: str) -> int:
        value = raw.get(name)
        return int(value) if isinstance(value, (int, float)) else 0

    regular = integer("input_tokens")
    cache_read = integer("cache_read_input_tokens")
    cache_write = integer("cache_creation_input_tokens")
    creation = raw.get("cache_creation")
    if isinstance(creation, dict):

        def creation_integer(name: str) -> int:
            value = creation.get(name)
            return int(value) if isinstance(value, (int, float)) else 0

        write_5m = creation_integer("ephemeral_5m_input_tokens")
        write_1h = creation_integer("ephemeral_1h_input_tokens")
    else:
        write_5m = write_1h = 0
    if write_5m + write_1h != cache_write:
        # The flat total is authoritative for the bill; charge whatever it does not
        # account for to the TTL this proxy configured (the pricier class by default).
        remainder = cache_write - (write_5m + write_1h)
        if cache_ttl == "5m":
            write_5m += remainder
        else:
            write_1h += remainder
        write_5m = max(0, write_5m)
        write_1h = max(0, write_1h)
    output = integer("output_tokens")
    details = raw.get("output_tokens_details")
    reasoning = (
        int(details.get("thinking_tokens") or 0) if isinstance(details, dict) else 0
    )
    total_input = regular + cache_read + cache_write
    return {
        "input_tokens": total_input,
        "input_tokens_details": {
            "cached_tokens": cache_read,
            "cache_write_tokens": cache_write,
        },
        "cached_input_tokens": cache_read,
        "cache_write_input_tokens": cache_write,
        "cache_write_5m_input_tokens": write_5m,
        "cache_write_1h_input_tokens": write_1h,
        "output_tokens": output,
        "output_tokens_details": {"reasoning_tokens": reasoning},
        "total_tokens": total_input + output,
    }


def anthropic_to_response(
    request_body: dict[str, Any], message: dict[str, Any], *, cache_ttl: str = "1h"
) -> dict[str, Any]:
    """Build one completed OpenAI Response object from an Anthropic Message."""

    content = message.get("content")
    blocks = content if isinstance(content, list) else []
    thinking_blocks = [
        copy.deepcopy(block)
        for block in blocks
        if isinstance(block, dict)
        and block.get("type") in {"thinking", "redacted_thinking"}
    ]
    reasoning_text = "\n".join(
        str(block.get("thinking") or "")
        for block in thinking_blocks
        if block.get("type") == "thinking" and block.get("thinking")
    )
    output_text = "\n".join(
        str(block.get("text") or "")
        for block in blocks
        if isinstance(block, dict) and block.get("type") == "text"
    )
    advertised_tools, _ = _translated_collaboration_tools(request_body.get("tools"))
    advertised_names = {str(tool["name"]) for tool in advertised_tools}
    tool_uses = [
        block
        for block in blocks
        if isinstance(block, dict) and block.get("type") == "tool_use"
    ]

    message_id = str(message.get("id") or "msg_anthropic")
    output: list[dict[str, Any]] = []
    if thinking_blocks:
        output.append({
            "id": "rs_" + hashlib.sha256(message_id.encode("utf-8")).hexdigest()[:24],
            "type": "reasoning",
            "summary": [],
            "content": (
                [{"type": "reasoning_text", "text": reasoning_text}]
                if reasoning_text
                else []
            ),
            "encrypted_content": _encode_reasoning(thinking_blocks),
        })
    if output_text or not tool_uses:
        output.append({
            "id": message_id,
            "type": "message",
            "status": "completed",
            "role": "assistant",
            "content": [{
                "type": "output_text",
                "text": output_text,
                "annotations": [],
                "logprobs": [],
            }],
        })
    for index, block in enumerate(tool_uses):
        anthropic_name = str(block.get("name") or "")
        name = _collaboration_tool_name(anthropic_name)
        call_id = str(block.get("id") or "")
        arguments = block.get("input")
        if (
            name is None
            or anthropic_name not in advertised_names
            or not call_id
        ):
            raise TranslationError(
                "Anthropic returned an unadvertised collaboration tool call"
            )
        arguments = _validate_collaboration_arguments(name, arguments)
        output.append({
            "id": "fc_"
            + hashlib.sha256(
                f"{message_id}:{call_id}:{index}".encode("utf-8")
            ).hexdigest()[:24],
            "type": "function_call",
            "status": "completed",
            "name": name,
            "namespace": _COLLABORATION_NAMESPACE,
            "arguments": json.dumps(
                arguments, ensure_ascii=False, separators=(",", ":")
            ),
            "call_id": call_id,
        })

    stop_reason = str(message.get("stop_reason") or "end_turn")
    incomplete = stop_reason in {"max_tokens", "model_context_window_exceeded"}
    now = int(time.time())
    return {
        "id": "resp_" + message_id,
        "object": "response",
        "created_at": now,
        "completed_at": now,
        "status": "incomplete" if incomplete else "completed",
        "error": None,
        "incomplete_details": (
            {"reason": "max_output_tokens"} if incomplete else None
        ),
        "instructions": request_body.get("instructions"),
        "max_output_tokens": request_body.get("max_output_tokens"),
        "model": message.get("model") or request_body.get("model"),
        "output": output,
        "parallel_tool_calls": bool(request_body.get("parallel_tool_calls", True)),
        "previous_response_id": request_body.get("previous_response_id"),
        "reasoning": request_body.get("reasoning"),
        "store": False,
        "temperature": 1.0,
        "text": request_body.get("text") or {"format": {"type": "text"}},
        "tool_choice": request_body.get("tool_choice") or "auto",
        "tools": copy.deepcopy(request_body.get("tools") or []),
        "top_p": 1.0,
        "truncation": request_body.get("truncation") or "disabled",
        "usage": _anthropic_usage(message.get("usage"), cache_ttl=cache_ttl),
        "metadata": {},
    }


def response_sse(response: dict[str, Any]) -> bytes:
    """Render the completed response as the item events Codex consumes."""

    events: list[dict[str, Any]] = []
    sequence = 0

    def add(kind: str, **fields: Any) -> None:
        nonlocal sequence
        events.append({"type": kind, **fields, "sequence_number": sequence})
        sequence += 1

    for output_index, completed_item in enumerate(response.get("output") or []):
        item = copy.deepcopy(completed_item)
        item_type = item.get("type")
        if item_type == "reasoning":
            started = copy.deepcopy(item)
            started["content"] = []
            started["status"] = "in_progress"
            add("response.output_item.added", output_index=output_index, item=started)
            for content_index, part in enumerate(item.get("content") or []):
                text = str(part.get("text") or "")
                if text:
                    add(
                        "response.reasoning_text.delta",
                        item_id=item["id"],
                        output_index=output_index,
                        content_index=content_index,
                        delta=text,
                    )
                add(
                    "response.reasoning_text.done",
                    item_id=item["id"],
                    output_index=output_index,
                    content_index=content_index,
                    text=text,
                )
            item["status"] = "completed"
            add("response.output_item.done", output_index=output_index, item=item)
            continue

        if item_type == "message":
            started = copy.deepcopy(item)
            started["content"] = []
            started["status"] = "in_progress"
            add("response.output_item.added", output_index=output_index, item=started)
            for content_index, part in enumerate(item.get("content") or []):
                text = str(part.get("text") or "")
                empty_part = {"type": "output_text", "text": "", "annotations": []}
                add(
                    "response.content_part.added",
                    item_id=item["id"],
                    output_index=output_index,
                    content_index=content_index,
                    part=empty_part,
                )
                if text:
                    add(
                        "response.output_text.delta",
                        item_id=item["id"],
                        output_index=output_index,
                        content_index=content_index,
                        delta=text,
                        logprobs=[],
                    )
                add(
                    "response.output_text.done",
                    item_id=item["id"],
                    output_index=output_index,
                    content_index=content_index,
                    text=text,
                    logprobs=[],
                )
                add(
                    "response.content_part.done",
                    item_id=item["id"],
                    output_index=output_index,
                    content_index=content_index,
                    part=part,
                )
            item["status"] = "completed"
            add("response.output_item.done", output_index=output_index, item=item)
            continue

        if item_type == "function_call":
            started = copy.deepcopy(item)
            arguments = str(started.get("arguments") or "")
            started["arguments"] = ""
            started["status"] = "in_progress"
            add("response.output_item.added", output_index=output_index, item=started)
            if arguments:
                add(
                    "response.function_call_arguments.delta",
                    item_id=item["id"],
                    output_index=output_index,
                    delta=arguments,
                )
            add(
                "response.function_call_arguments.done",
                item_id=item["id"],
                output_index=output_index,
                arguments=arguments,
            )
            item["status"] = "completed"
            add("response.output_item.done", output_index=output_index, item=item)

    terminal_type = (
        "response.completed" if response.get("status") == "completed" else "response.incomplete"
    )
    add(terminal_type, response=response)
    chunks = []
    for event in events:
        data = json.dumps(event, ensure_ascii=False, separators=(",", ":"))
        chunks.append(f"event: {event['type']}\ndata: {data}\n\n")
    chunks.append("data: [DONE]\n\n")
    return "".join(chunks).encode("utf-8")


class _AdapterServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(
        self,
        server_address: tuple[str, int],
        *,
        anthropic_base_url: str,
        cache_ttl: str,
        log_file: Path | None,
        timeout_seconds: float,
        default_max_tokens: int,
    ) -> None:
        super().__init__(server_address, _AdapterHandler)
        parsed = urlsplit(anthropic_base_url.rstrip("/"))
        if parsed.scheme not in {"http", "https"} or not parsed.hostname:
            raise ValueError("anthropic_base_url must be an http(s) URL")
        self.anthropic = parsed
        self.cache_ttl = cache_ttl
        self.log_file = log_file
        self.timeout_seconds = timeout_seconds
        self.default_max_tokens = default_max_tokens
        self.log_lock = threading.Lock()

    def audit(self, row: dict[str, Any]) -> None:
        if self.log_file is None:
            return
        line = json.dumps(
            {"timestamp": time.time(), **row},
            ensure_ascii=False,
            separators=(",", ":"),
        ) + "\n"
        with self.log_lock:
            with self.log_file.open("a", encoding="utf-8") as stream:
                stream.write(line)


class _AdapterHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server: _AdapterServer

    def log_message(self, _format: str, *_args: Any) -> None:
        return

    def do_GET(self) -> None:  # noqa: N802
        if self.path.split("?", 1)[0].rstrip("/") != "/models":
            self._json_error(404, "not_found", "unsupported adapter endpoint")
            return
        self._send_json(200, _model_catalog())

    def do_POST(self) -> None:  # noqa: N802
        if self.path.split("?", 1)[0].rstrip("/") != "/responses":
            self._json_error(404, "not_found", "unsupported adapter endpoint")
            return
        length = int(self.headers.get("Content-Length") or 0)
        raw_body = self.rfile.read(length)
        try:
            body = json.loads(raw_body)
            if not isinstance(body, dict):
                raise TranslationError("Responses body must be an object")
            anthropic_request, diagnostics = responses_to_anthropic(
                body,
                cache_ttl=self.server.cache_ttl,
                default_max_tokens=self.server.default_max_tokens,
            )
        except (json.JSONDecodeError, UnicodeDecodeError, TranslationError) as exc:
            self._json_error(400, "invalid_request_error", str(exc))
            return

        api_key = self._api_key()
        if not api_key:
            self._json_error(401, "authentication_error", "missing Anthropic API key")
            return
        diagnostics["request_bytes"] = length
        status, message = self._call_anthropic(api_key, anthropic_request)
        diagnostics["upstream_status"] = status
        if status != 200:
            error = message.get("error") if isinstance(message, dict) else None
            detail = error.get("message") if isinstance(error, dict) else str(message)
            # The CLI may swallow this error, so the audit log must keep it.
            diagnostics["upstream_error"] = str(detail)[:2000]
            self.server.audit(diagnostics)
            self._json_error(status, "anthropic_api_error", detail)
            return

        try:
            response = anthropic_to_response(
                body, message, cache_ttl=self.server.cache_ttl
            )
        except TranslationError as exc:
            diagnostics["upstream_error"] = str(exc)[:2000]
            self.server.audit(diagnostics)
            self._json_error(502, "translation_error", str(exc))
            return
        usage = response["usage"]
        diagnostics.update({
            "input_tokens": usage["input_tokens"],
            "cached_input_tokens": usage["cached_input_tokens"],
            "cache_write_input_tokens": usage["cache_write_input_tokens"],
            "cache_write_5m_input_tokens": usage["cache_write_5m_input_tokens"],
            "cache_write_1h_input_tokens": usage["cache_write_1h_input_tokens"],
            "output_tokens": usage["output_tokens"],
            "stop_reason": message.get("stop_reason"),
        })
        self.server.audit(diagnostics)
        payload = response_sse(response)
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(payload)
        self.close_connection = True

    def _api_key(self) -> str:
        bearer = self.headers.get("Authorization") or ""
        if bearer.startswith("Bearer "):
            return bearer.removeprefix("Bearer ").strip()
        return (self.headers.get("x-api-key") or "").strip()

    def _call_anthropic(
        self, api_key: str, body: dict[str, Any]
    ) -> tuple[int, dict[str, Any]]:
        upstream = self.server.anthropic
        path = upstream.path.rstrip("/") + "/messages"
        payload = json.dumps(body, ensure_ascii=False, separators=(",", ":")).encode(
            "utf-8"
        )
        connection_type = (
            http.client.HTTPSConnection
            if upstream.scheme == "https"
            else http.client.HTTPConnection
        )
        connection = connection_type(
            upstream.hostname, upstream.port, timeout=self.server.timeout_seconds
        )
        try:
            connection.request(
                "POST",
                path,
                body=payload,
                headers={
                    "x-api-key": api_key,
                    "anthropic-version": "2023-06-01",
                    "content-type": "application/json",
                    "accept": "application/json",
                },
            )
            result = connection.getresponse()
            raw = result.read()
            try:
                decoded = json.loads(raw)
            except (json.JSONDecodeError, UnicodeDecodeError):
                decoded = {"error": {"message": raw.decode("utf-8", errors="replace")}}
            return result.status, decoded if isinstance(decoded, dict) else {}
        except (OSError, http.client.HTTPException) as exc:
            return 502, {"error": {"message": str(exc)}}
        finally:
            connection.close()

    def _json_error(self, status: int, kind: str, message: str) -> None:
        self._send_json(status, {"error": {"type": kind, "message": message}})

    def _send_json(self, status: int, body: dict[str, Any]) -> None:
        payload = json.dumps(body, ensure_ascii=False, separators=(",", ":")).encode(
            "utf-8"
        )
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(payload)
        self.close_connection = True


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port-file", type=Path, required=True)
    parser.add_argument("--anthropic-base-url", default="https://api.anthropic.com/v1")
    parser.add_argument("--cache-ttl", choices=("5m", "1h"), default="1h")
    parser.add_argument("--log-file", type=Path)
    parser.add_argument("--timeout-seconds", type=float, default=900.0)
    parser.add_argument("--default-max-tokens", type=int, default=65_536)
    args = parser.parse_args(argv)

    server = _AdapterServer(
        ("127.0.0.1", 0),
        anthropic_base_url=args.anthropic_base_url,
        cache_ttl=args.cache_ttl,
        log_file=args.log_file,
        timeout_seconds=args.timeout_seconds,
        default_max_tokens=args.default_max_tokens,
    )
    args.port_file.write_text(str(server.server_port), encoding="ascii")

    def stop(_signum: int, _frame: Any) -> None:
        threading.Thread(target=server.shutdown, daemon=True).start()

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    try:
        server.serve_forever(poll_interval=0.1)
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
