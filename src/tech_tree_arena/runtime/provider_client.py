"""Arena-owned structured-model provider client.

Every structured() call -- generator, oracle, and judge alike -- is one
recorded conversation: full developer + user prompts, schema, parsed response,
token usage, cost, and a request hash. With set_trace(path), each call appends
one JSONL line for audit and debugging.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import random
import re
import threading
import time
import urllib.error
import urllib.request
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Iterator

from openai import OpenAI

from ..model_defaults import DEFAULT_MAX_OUTPUT_TOKENS

_CLIENT: OpenAI | None = None
_ANTHROPIC_CLIENT: Any | None = None
_TOGETHER_CLIENT_READY = False
_LOCK = threading.Lock()
_TRACE_PATH: Path | None = None
_TOTALS = {
    "calls": 0,
    "provider_calls": 0,
    "unknown_provider_attempts": 0,
    "input_tokens": 0,
    "output_tokens": 0,
    "cost_usd": 0.0,
}
_LAST_CALL: dict[str, Any] = {}
_ATTEMPT_CONTEXT = threading.local()
# Per-call accounting must stay attributable when multiple logical calls run
# concurrently, so the in-flight record lives in thread-local storage; the
# module-global mirror is kept for legacy single-threaded readers.
_CALL_LOCAL = threading.local()


class _ProviderTransportError(RuntimeError):
    """One failed stdlib HTTP request with retry metadata for the caller."""

    def __init__(
        self,
        message: str,
        *,
        retryable: bool,
        status_code: int | None = None,
        headers: Any = None,
    ) -> None:
        super().__init__(message)
        self.retryable = retryable
        self.status_code = status_code
        self.headers = headers


def _set_call_record(record: dict[str, Any]) -> None:
    global _LAST_CALL
    _CALL_LOCAL.record = record
    _LAST_CALL = record


def _mutate_call_record(**fields: Any) -> None:
    global _LAST_CALL
    record = getattr(_CALL_LOCAL, "record", None)
    if record is not None:
        record.update(fields)
        _LAST_CALL = record
    else:
        _LAST_CALL = {**_LAST_CALL, **fields}

# Global cap on concurrent in-flight API calls. Nested thread pools (e.g. best-of-N rung
# attempts, each running an 8-worker judge) can otherwise burst 24+ simultaneous requests and
# stall the connection pool / rate limit. The semaphore bounds total concurrency regardless of
# nesting; override via env LLM_MAX_CONCURRENCY.
_MAX_CONCURRENCY = max(1, int(os.environ.get("LLM_MAX_CONCURRENCY", "10")))
_SEM = threading.BoundedSemaphore(_MAX_CONCURRENCY)
# Together's serverless rate limits are substantially lower than the aggregate
# nested-game concurrency. Bound only active Together HTTP calls; retries release
# this slot while sleeping so unrelated work can still make progress.
_TOGETHER_MAX_CONCURRENCY = max(1, int(os.environ.get("TOGETHER_MAX_CONCURRENCY", "5")))
_TOGETHER_SEM = threading.BoundedSemaphore(_TOGETHER_MAX_CONCURRENCY)

# Per-model ($/1M tokens) rates (OpenAI list pricing, base tier; verified Jun 2026):
#   gpt-5.4 -> $2.50 in / $15 out   gpt-5.5 -> $5.00 in / $30 out
# reasoning_tokens are already folded into usage.output_tokens, so output billing covers them.
# Except for Astra and Sol (priced by reported cache classes), OpenAI cached input
# is billed at the FULL input rate (over-counts ->
# conservative); Anthropic's cache classes have published rates and are priced exactly
# instead -- see _ANTHROPIC_CLASS_RATES.
_PRICING = {
    "gpt-6-astra": (10.00, 50.00),
    "gpt-5.6-sol": (4.00, 20.00),
    "gpt-5.6-terra": (2.50, 15.00),
    "gpt-5.6": (5.00, 30.00),
    "gpt-5.5": (5.00, 30.00),
    "gpt-5.4": (2.50, 15.00),
}
# Fallback for unknown non-Anthropic models. It is the gpt-5.5 tier, NOT a global upper
# bound -- Anthropic's Fable tier costs more, which is why unknown claude-* models fall
# back to _ANTHROPIC_FALLBACK_CLASS_RATE (the priciest known Anthropic tier) instead.
_DEFAULT_RATE = (5.00, 30.00)
_TOGETHER_DEFAULT_RATE = (
    float(os.environ.get("TOGETHER_INPUT_USD_PER_MTOK", "0.0")),
    float(os.environ.get("TOGETHER_OUTPUT_USD_PER_MTOK", "0.0")),
)
_TOGETHER_TWO_STAGE_REASONING_MODELS = frozenset({
    "together/zai-org/GLM-5.2",
    "together/zai-org/GLM-5.3",
    "together/moonshotai/Kimi-K3",
})
_TOGETHER_STREAMING_REASONING_MODELS = frozenset({
    "together/moonshotai/Kimi-K3",
})
_TOGETHER_REASONING_MAX_TOKENS = 50_000
_TOGETHER_GLM_REASONING_TIMEOUT_SECONDS = 900.0
_TOGETHER_STREAMING_REASONING_TIMEOUT_SECONDS = 1_800.0
_OPENROUTER_DEFAULT_RATE = (5.00, 30.00)

# Anthropic bills five distinct token classes at published per-class rates, so a single
# blended input rate cannot reproduce a bill: on a cache-read-dominated agent session
# charging reads at the base rate overstates spend severalfold, while charging 1h cache
# writes at the base rate understates them twofold.  ($/1M tokens, from
# platform.claude.com/docs/en/about-claude/pricing, verified Aug 2026.)
#   (base_input, cache_write_5m, cache_write_1h, cache_read, output)
_ANTHROPIC_CLASS_RATES = {
    "claude-fable-5":    (10.00, 12.50, 20.00, 1.00, 50.00),
    "claude-mythos-5":   (10.00, 12.50, 20.00, 1.00, 50.00),
    "claude-opus-5":     ( 5.00,  6.25, 10.00, 0.50, 25.00),
    "claude-opus-4-8":   ( 5.00,  6.25, 10.00, 0.50, 25.00),
    "claude-opus-4-7":   ( 5.00,  6.25, 10.00, 0.50, 25.00),
    "claude-opus-4-6":   ( 5.00,  6.25, 10.00, 0.50, 25.00),
    "claude-sonnet-5":   ( 2.00,  2.50,  4.00, 0.20, 10.00),
    "claude-sonnet-4-6": ( 3.00,  3.75,  6.00, 0.30, 15.00),
    "claude-haiku-4-5":  ( 1.00,  1.25,  2.00, 0.10,  5.00),
}
# An unpriced claude-* model is priced as the most expensive known Anthropic tier, so a
# model released after this table was frozen over-reports rather than under-reports.
_ANTHROPIC_FALLBACK_CLASS_RATE = _ANTHROPIC_CLASS_RATES["claude-fable-5"]
_ANTHROPIC_ROUTE_PREFIXES = ("openrouter/anthropic/", "anthropic/", "claude-code/")

# Direct-path prompt caching.  Generator and judge calls reuse one system block
# (developer prompt + schema instruction) across every call in a run while only the
# user message varies, so an explicit breakpoint on the system block is read back on
# every later call.  A 5m write costs 1.25x base against a 0.1x read, so it pays for
# itself on the first hit and each hit refreshes the TTL.
#
# Measured on a 259-call Fable run (paper-2606.09967, 30 distinct prefixes): 5m saves
# 9.6% on 66 writes / 192 reads, 1h saves 10.4% on 30 writes / 228 reads.  1h is
# marginally better on a long run but doubles the write, so a short run that never
# reads a prefix back pays a full 1x surcharge instead of 0.25x -- hence 5m by default,
# with the override for long runs.  Fail closed on a bad value rather than pay 2x by
# accident.
#
# The user message is deliberately NOT cached: consecutive user messages on this run
# share only 17-24% of their text, so writing one at 1.25x to reclaim ~20% at 0.1x is
# a net loss.
_ANTHROPIC_CACHE_TTL = os.environ.get("ARENA_ANTHROPIC_CACHE_TTL", "5m")
if _ANTHROPIC_CACHE_TTL not in {"5m", "1h"}:
    raise ValueError("ARENA_ANTHROPIC_CACHE_TTL must be '5m' or '1h'")


def _split_cache_writes(flat_total: int, split_5m: int, split_1h: int, ttl: str):
    """Resolve Anthropic's cache-write total into its 5m and 1h billing classes.

    ``usage.cache_creation`` carries the split only when 1h caching is in play; the flat
    ``cache_creation_input_tokens`` is always authoritative for the bill.  Anything the
    split does not account for is charged to the TTL this client asked for, so a missing
    breakdown can never discount 1h writes to the cheaper 5m rate.  (The Responses
    adapter applies the same rule to its own usage mapping -- see
    anthropic_responses_proxy._anthropic_usage.)
    """

    if split_5m + split_1h == flat_total:
        return split_5m, split_1h
    if ttl == "5m":
        return max(0, flat_total - split_1h), min(split_1h, flat_total)
    return min(split_5m, flat_total), max(0, flat_total - split_5m)


def _anthropic_model_key(model: str) -> str | None:
    """Normalize a routed model string to an _ANTHROPIC_CLASS_RATES key, or None.

    Accepts bare ids as well as routed ones: a missing ``anthropic/`` prefix used to fall
    through to _DEFAULT_RATE and silently halve Fable's input rate.  Also folds the dotted
    spelling OpenRouter uses (``claude-opus-4.6``) and any trailing date snapshot.
    """

    name = str(model)
    for prefix in _ANTHROPIC_ROUTE_PREFIXES:
        if name.startswith(prefix):
            name = name[len(prefix):]
            break
    name = name.replace(".", "-")
    if not name.startswith("claude-"):
        return None
    if name in _ANTHROPIC_CLASS_RATES:
        return name
    # Longest matching family prefix wins, so claude-opus-4-6-20260514 prices as 4.6
    # rather than falling back to the Fable tier.
    matches = [key for key in _ANTHROPIC_CLASS_RATES if name.startswith(key)]
    return max(matches, key=len) if matches else ""


def anthropic_class_rates(model: str) -> tuple[float, float, float, float, float]:
    """Per-class $/1M rates for one Anthropic model.

    Returns (base_input, cache_write_5m, cache_write_1h, cache_read, output).
    """

    key = _anthropic_model_key(model)
    if key is None or not key:
        return _ANTHROPIC_FALLBACK_CLASS_RATE
    return _ANTHROPIC_CLASS_RATES[key]


def anthropic_turn_cost(
    model: str,
    *,
    uncached_input_tokens: int,
    cache_read_input_tokens: int,
    cache_write_5m_input_tokens: int,
    cache_write_1h_input_tokens: int,
    output_tokens: int,
) -> float:
    """Cost of one Anthropic turn, each token class at its own published rate."""

    base, write_5m, write_1h, read, out = anthropic_class_rates(model)
    return (
        max(0, int(uncached_input_tokens)) * base
        + max(0, int(cache_read_input_tokens)) * read
        + max(0, int(cache_write_5m_input_tokens)) * write_5m
        + max(0, int(cache_write_1h_input_tokens)) * write_1h
        + max(0, int(output_tokens)) * out
    ) / 1e6


def _rates(model: str) -> tuple[float, float]:
    """(input, output) $/1M for the uncached-input accounting used per provider call."""

    if str(model).startswith("together/"):
        # Read at call time: the operator supplies these rates through the
        # environment, and a run's cost audit must not depend on whether the
        # env file was loaded before or after this module was imported.
        return (
            float(os.environ.get("TOGETHER_INPUT_USD_PER_MTOK", _TOGETHER_DEFAULT_RATE[0])),
            float(os.environ.get("TOGETHER_OUTPUT_USD_PER_MTOK", _TOGETHER_DEFAULT_RATE[1])),
        )
    key = _anthropic_model_key(model)
    if key is not None:
        base, _w5, _w1, _read, out = (
            _ANTHROPIC_CLASS_RATES[key] if key else _ANTHROPIC_FALLBACK_CLASS_RATE
        )
        return (base, out)
    if str(model).startswith("openrouter/"):
        return _OPENROUTER_DEFAULT_RATE
    for prefix, rate in _PRICING.items():
        if str(model).startswith(prefix):
            return rate
    return _DEFAULT_RATE


def _provider_request_sha(provider: str, request: dict[str, Any]) -> str:
    """Hash one exact low-level provider request without retaining its plaintext."""

    blob = json.dumps(
        request,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(provider.encode("utf-8") + b"\0" + blob).hexdigest()


def _provider_response_sha(response: Any) -> str:
    def jsonable(value: Any) -> Any:
        if value is None or isinstance(value, (bool, int, float, str)):
            return value
        if isinstance(value, dict):
            return {str(key): jsonable(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [jsonable(item) for item in value]
        if hasattr(value, "model_dump"):
            return jsonable(value.model_dump())
        if hasattr(value, "__dict__"):
            return jsonable(vars(value))
        return {"type": type(value).__name__, "text": str(value)}

    payload = jsonable(response)
    blob = json.dumps(
        payload,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def _attempt_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    in_per_m, out_per_m = _rates(model)
    return input_tokens / 1e6 * in_per_m + output_tokens / 1e6 * out_per_m


def _openai_attempt_cost(model: str, usage: Any) -> float:
    """Price Astra/Sol cache classes; retain existing accounting for older models.

    https://developers.openai.com/api/docs/models/gpt-6-astra (2026-09-04).
    https://developers.openai.com/api/docs/pricing (Sol, 2026-09-04).
    Input totals include both cache reads and writes. Long-context multipliers
    apply to the entire request once total input exceeds 272,000 tokens.
    """
    input_tokens = int(getattr(usage, "input_tokens", 0) or 0)
    output_tokens = int(getattr(usage, "output_tokens", 0) or 0)
    if not model.startswith(("gpt-6-astra", "gpt-5.6-sol")):
        return _attempt_cost(model, input_tokens, output_tokens)
    details = getattr(usage, "input_tokens_details", None)
    reads = int(getattr(details, "cached_tokens", 0) or 0)
    writes = int(getattr(details, "cache_write_tokens", 0) or 0)
    uncached = max(0, input_tokens - reads - writes)
    input_rate, output_rate = _rates(model)
    input_cost = (uncached + reads * 0.1 + writes * 1.25) * input_rate / 1e6
    output_cost = output_tokens * output_rate / 1e6
    if input_tokens > 272_000:
        input_cost *= 2
        output_cost *= 1.5
    return input_cost + output_cost


def _bounded_error_fields(
    exc: BaseException,
    *,
    message: str = "provider request failed",
) -> tuple[str, str, int | None]:
    """Return non-echoing, bounded metadata safe for the durable attempt log."""

    raw_type = type(exc).__name__
    error_type = re.sub(r"[^A-Za-z0-9_.-]", "_", raw_type)[:64]
    if not error_type or not error_type[0].isalpha():
        error_type = "ProviderError"
    status = getattr(exc, "status_code", None)
    status_code = status if type(status) is int and 100 <= status <= 599 else None
    return error_type, message[:160], status_code


def _commit_provider_attempt(record: dict[str, Any]) -> None:
    """Commit one SDK-level request to cumulative and current-call accounting.

    Anthropic and OpenAI SDK retries are disabled, so each record corresponds
    to exactly one outbound provider request.  The record contains only a hash
    of the request and provider response accounting, never prompt plaintext.
    """

    with _LOCK:
        _TOTALS["provider_calls"] += 1
        if record.get("usage_available") is not True:
            _TOTALS["unknown_provider_attempts"] += 1
        _TOTALS["input_tokens"] += int(record.get("input_tokens") or 0)
        _TOTALS["output_tokens"] += int(record.get("output_tokens") or 0)
        _TOTALS["cost_usd"] += float(record.get("cost_usd") or 0.0)
        current = getattr(_CALL_LOCAL, "record", None)
        if current is None:
            current = _LAST_CALL
        attempts = list(current.get("provider_attempts") or [])
        attempts.append(copy.deepcopy(record))
        _mutate_call_record(provider_attempts=attempts)
    _emit_provider_attempt({"kind": "provider_attempt_finished", **record})


def _emit_provider_attempt(record: dict[str, Any], request: dict | None = None) -> None:
    guard = getattr(_ATTEMPT_CONTEXT, "guard", None)
    if guard is not None:
        guard.provider_event(record, request)
    sink = getattr(_ATTEMPT_CONTEXT, "sink", None)
    if callable(sink):
        sink(copy.deepcopy(record))


@contextmanager
def provider_attempt_guard(guard):
    previous = getattr(_ATTEMPT_CONTEXT, "guard", None)
    _ATTEMPT_CONTEXT.guard = guard
    try:
        yield
    finally:
        _ATTEMPT_CONTEXT.guard = previous


@contextmanager
def provider_attempt_sink(
    sink: Callable[[dict[str, Any]], None] | None,
) -> Iterator[None]:
    """Install the durable attempt journal for one serialized logical call."""

    previous = getattr(_ATTEMPT_CONTEXT, "sink", None)
    _ATTEMPT_CONTEXT.sink = sink
    try:
        yield
    finally:
        _ATTEMPT_CONTEXT.sink = previous


def _is_anthropic_model(model: str) -> bool:
    return str(model).startswith("anthropic/")


def _anthropic_model_name(model: str) -> str:
    return str(model).removeprefix("anthropic/")


def _is_openrouter_model(model: str) -> bool:
    return str(model).startswith("openrouter/")


def _openrouter_model_name(model: str) -> str:
    return str(model).removeprefix("openrouter/")


def _openrouter_max_tokens(max_output_tokens: int, reasoning_effort: str | None) -> int:
    """Leave sufficient completion room after high-reasoning inference for JSON output."""
    return max(max_output_tokens, 49_152) if reasoning_effort == "high" else max_output_tokens


def _openrouter_schema(value: Any) -> Any:
    """Relax provider-rejected constraints; local validation stays strict.

    Both OpenRouter-routed Anthropic models and Anthropic's native structured
    output support only a subset of JSON Schema.  Keep the participant-facing
    schema unchanged and remove only constraints that the provider rejects;
    :func:`_validate_json_schema` enforces them after parsing.
    """
    if isinstance(value, list):
        return [_openrouter_schema(item) for item in value]
    if not isinstance(value, dict):
        return value
    return {
        key: _openrouter_schema(item)
        for key, item in value.items()
        if not (
            (key == "minItems" and isinstance(item, int) and item > 1)
            or (key == "maxItems" and isinstance(item, int))
            or key in {"exclusiveMinimum", "exclusiveMaximum"}
        )
    }


def client() -> OpenAI:
    global _CLIENT
    if _CLIENT is None:
        # OPENAI_API_KEY is expected in the environment.
        if not os.environ.get("OPENAI_API_KEY") and os.environ.get("OPENAI_KEY"):
            os.environ["OPENAI_API_KEY"] = os.environ["OPENAI_KEY"]
        # Arena owns retry/deadline accounting; disable the SDK's hidden retry layer.
        _CLIENT = OpenAI(max_retries=0)
    return _CLIENT


def anthropic_client() -> Any:
    global _ANTHROPIC_CLIENT
    if _ANTHROPIC_CLIENT is None:
        if not os.environ.get("ANTHROPIC_API_KEY"):
            raise RuntimeError("ANTHROPIC_API_KEY is required for anthropic/* models")
        from anthropic import Anthropic
        # Arena owns every retry.  SDK retries are opaque to the service ledger,
        # so keeping them enabled would under-count provider attempts and cost.
        _ANTHROPIC_CLIENT = Anthropic(max_retries=0)
    return _ANTHROPIC_CLIENT


def _is_together_model(model: str) -> bool:
    return str(model).startswith("together/")


def _together_model_name(model: str) -> str:
    return str(model).removeprefix("together/")


def _load_together_key() -> str:
    key = os.environ.get("TOGETHER_API_KEY")
    if not key:
        raise RuntimeError("TOGETHER_API_KEY is required for together/* models")
    return key


def _together_retry_delay(attempt: int, headers: Any = None) -> float:
    """Respect Together's reset header and desynchronize concurrent retries."""
    reset = None
    if headers:
        for name in ("Retry-After", "X-RateLimit-Reset"):
            value = headers.get(name)
            if value is None:
                continue
            match = re.fullmatch(r"\s*(\d+(?:\.\d+)?)\s*(ms|s|m)?\s*", str(value), re.I)
            if not match:
                continue
            reset = float(match.group(1))
            unit = (match.group(2) or "s").lower()
            if unit == "ms":
                reset /= 1000.0
            elif unit == "m":
                reset *= 60.0
            break
    cap = float(os.environ.get("TOGETHER_RETRY_MAX_DELAY_S", "60"))
    delay = reset if reset is not None else min(cap, 2.0 * (2 ** attempt))
    delay = min(cap, max(0.0, delay))
    return delay + random.uniform(0.0, min(1.0, delay * 0.2))


def _together_chat_complete(body: dict[str, Any], timeout: float) -> dict[str, Any]:
    """Issue exactly one Together HTTP request.

    Retry ownership belongs to ``_together_create_parse_retry`` so each
    physical request receives its own durable provider-attempt record.
    """
    key = _load_together_key()
    data = json.dumps(body).encode("utf-8")
    url = os.environ.get("TOGETHER_CHAT_COMPLETIONS_URL", "https://api.together.xyz/v1/chat/completions")
    req = urllib.request.Request(
        url,
        data=data,
        headers={
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
            "User-Agent": "tech-tree-repro/1.0",
        },
        method="POST",
    )
    try:
        with _TOGETHER_SEM, urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise _ProviderTransportError(
            f"Together API HTTP {exc.code}: {detail[:1000]}",
            retryable=exc.code in {408, 409, 425, 429, 500, 502, 503, 504},
            status_code=exc.code,
            headers=exc.headers,
        ) from exc
    except urllib.error.URLError as exc:
        raise _ProviderTransportError(
            f"Together API URL error: {exc}", retryable=True
        ) from exc
    except TimeoutError as exc:
        # A read timeout arrives bare after a connection was established.
        raise _ProviderTransportError(
            f"Together API read timeout: {exc}", retryable=True
        ) from exc


def _together_chat_complete_streaming(
    body: dict[str, Any], timeout: float, *, preserve_reasoning_fields: bool = False
) -> dict[str, Any]:
    """Issue one streamed Together request and rebuild a chat response.

    Kimi K3 max-reasoning calls can run longer than Together's non-streaming
    Cloudflare Worker allows.  Consuming SSE deltas keeps that edge request
    alive while preserving the same auditable response and usage fields used
    by the non-streaming path.
    """
    key = _load_together_key()
    request_body = {
        **body,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    data = json.dumps(request_body).encode("utf-8")
    url = os.environ.get(
        "TOGETHER_CHAT_COMPLETIONS_URL",
        "https://api.together.xyz/v1/chat/completions",
    )
    req = urllib.request.Request(
        url,
        data=data,
        headers={
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
            "User-Agent": "tech-tree-repro/1.0",
        },
        method="POST",
    )
    response_id: str | None = None
    response_model: str | None = None
    content_parts: list[str] = []
    reasoning_parts: list[str] = []
    native_reasoning_parts: dict[str, list[str]] = {}
    finish_reason: str | None = None
    usage: dict[str, Any] = {}
    try:
        with _TOGETHER_SEM, urllib.request.urlopen(req, timeout=timeout) as resp:
            for raw_line in resp:
                line = raw_line.decode("utf-8", errors="replace").strip()
                if not line.startswith("data:"):
                    continue
                event_data = line.removeprefix("data:").strip()
                if not event_data or event_data == "[DONE]":
                    continue
                event = json.loads(event_data)
                if isinstance(event.get("id"), str) and event["id"]:
                    response_id = event["id"]
                if isinstance(event.get("model"), str) and event["model"]:
                    response_model = event["model"]
                if isinstance(event.get("usage"), dict):
                    usage = event["usage"]
                for choice in event.get("choices") or []:
                    if choice.get("finish_reason") is not None:
                        finish_reason = choice["finish_reason"]
                    delta = choice.get("delta") or {}
                    content = delta.get("content")
                    if isinstance(content, str):
                        content_parts.append(content)
                    reasoning = (
                        delta.get("reasoning_content")
                        or delta.get("reasoning")
                    )
                    if isinstance(reasoning, str):
                        reasoning_parts.append(reasoning)
                    for field in ("reasoning_content", "reasoning"):
                        if isinstance(delta.get(field), str):
                            native_reasoning_parts.setdefault(field, []).append(delta[field])
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise _ProviderTransportError(
            f"Together API HTTP {exc.code}: {detail[:1000]}",
            retryable=exc.code in {408, 409, 425, 429, 500, 502, 503, 504},
            status_code=exc.code,
            headers=exc.headers,
        ) from exc
    except urllib.error.URLError as exc:
        raise _ProviderTransportError(
            f"Together API URL error: {exc}", retryable=True
        ) from exc
    except TimeoutError as exc:
        raise _ProviderTransportError(
            f"Together API read timeout: {exc}", retryable=True
        ) from exc

    return {
        "id": response_id,
        "model": response_model,
        "choices": [{
            "message": {
                "role": "assistant",
                "content": "".join(content_parts),
                **({field: "".join(parts) for field, parts in native_reasoning_parts.items()}
                   if preserve_reasoning_fields else {"reasoning_content": "".join(reasoning_parts)}),
            },
            "finish_reason": finish_reason,
        }],
        "usage": usage,
    }


def _schema_instruction(schema_name: str, schema: dict[str, Any]) -> str:
    return (
        "Return ONLY valid JSON. The JSON must conform to this JSON Schema named "
        f"{schema_name}:\n{json.dumps(schema, ensure_ascii=False, sort_keys=True)}"
    )


def _validate_json_schema(value: Any, schema: dict[str, Any], path: str = "$") -> None:
    """Small strict-enough JSON-schema validator for the schemas used by this repo."""
    typ = schema.get("type")
    if typ == "object":
        if not isinstance(value, dict):
            raise ValueError(f"{path}: expected object")
        props = schema.get("properties") or {}
        for key in schema.get("required") or []:
            if key not in value:
                raise ValueError(f"{path}: missing required key {key!r}")
        if schema.get("additionalProperties") is False:
            extra = sorted(set(value) - set(props))
            if extra:
                raise ValueError(f"{path}: unexpected keys {extra}")
        for key, sub in props.items():
            if key in value:
                _validate_json_schema(value[key], sub, f"{path}.{key}")
    elif typ == "array":
        if not isinstance(value, list):
            raise ValueError(f"{path}: expected array")
        if "minItems" in schema and len(value) < int(schema["minItems"]):
            raise ValueError(f"{path}: expected at least {schema['minItems']} items")
        if "maxItems" in schema and len(value) > int(schema["maxItems"]):
            raise ValueError(f"{path}: expected at most {schema['maxItems']} items")
        sub = schema.get("items")
        if isinstance(sub, dict):
            for i, item in enumerate(value):
                _validate_json_schema(item, sub, f"{path}[{i}]")
    elif typ == "string":
        if not isinstance(value, str):
            raise ValueError(f"{path}: expected string")
        if "minLength" in schema and len(value) < int(schema["minLength"]):
            raise ValueError(f"{path}: expected at least {schema['minLength']} characters")
        if "maxLength" in schema and len(value) > int(schema["maxLength"]):
            raise ValueError(f"{path}: expected at most {schema['maxLength']} characters")
    elif typ == "integer":
        if not isinstance(value, int) or isinstance(value, bool):
            raise ValueError(f"{path}: expected integer")
    elif typ == "number":
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise ValueError(f"{path}: expected number")
    elif typ == "boolean":
        if not isinstance(value, bool):
            raise ValueError(f"{path}: expected boolean")
    if "enum" in schema and value not in schema["enum"]:
        raise ValueError(f"{path}: expected one of {schema['enum']}, got {value!r}")
    if "exclusiveMinimum" in schema and value <= schema["exclusiveMinimum"]:
        raise ValueError(f"{path}: expected value > {schema['exclusiveMinimum']}")
    if "exclusiveMaximum" in schema and value >= schema["exclusiveMaximum"]:
        raise ValueError(f"{path}: expected value < {schema['exclusiveMaximum']}")
    if "minimum" in schema and value < schema["minimum"]:
        raise ValueError(f"{path}: expected value >= {schema['minimum']}")
    if "maximum" in schema and value > schema["maximum"]:
        raise ValueError(f"{path}: expected value <= {schema['maximum']}")


def _chat_create_parse_retry(
    body: dict[str, Any],
    timeout: float,
    schema_name: str,
    schema: dict[str, Any] | None,
    *,
    provider: str,
    model: str,
    logical_request_sha: str,
    chat_complete: Callable[[dict[str, Any], float], dict[str, Any]],
    http_retries: int,
    retry_delay: Callable[[int, Any], float],
    retries: int = 4,
    max_tokens_cap: int | None = None,
    attempt_offset: int = 0,
) -> tuple[dict[str, Any], Any, list[dict[str, Any]]]:
    """Run one chat route with one durable attempt per physical HTTP request."""

    request, last_error = dict(body), None
    attempts: list[dict[str, Any]] = []
    for _schema_attempt in range(retries):
        missing_response = object()
        response: Any = missing_response
        base: dict[str, Any] | None = None
        for http_index in range(max(1, http_retries)):
            request_sha256 = hashlib.sha256(
                json.dumps(
                    request, sort_keys=True, ensure_ascii=False
                ).encode("utf-8")
            ).hexdigest()
            started = {
                "kind": "provider_attempt_started",
                "schema_version": "arena-provider-attempt-v2",
                "provider": provider,
                "model": model,
                "logical_request_sha": logical_request_sha,
                "attempt": attempt_offset + len(attempts) + 1,
                "request_sha256": request_sha256,
                "started_at": time.time(),
            }
            _emit_provider_attempt(started, request)
            base = {key: value for key, value in started.items() if key != "kind"}
            try:
                response = chat_complete(request, timeout)
            except Exception as exc:  # noqa: BLE001
                error_type, error_message, status_code = _bounded_error_fields(exc)
                record = {
                    **base,
                    "finished_at": time.time(),
                    "status": "error",
                    "provider_request_id": None,
                    "response_sha256": None,
                    "input_tokens": 0,
                    "cache_read_input_tokens": 0,
                    "cache_write_5m_input_tokens": 0,
                    "cache_write_1h_input_tokens": 0,
                    "output_tokens": 0,
                    "cost_usd": 0.0,
                    "usage_available": False,
                    "error_type": error_type,
                    "error_message": error_message,
                    "status_code": status_code,
                }
                attempts.append(record)
                _commit_provider_attempt(record)
                if (
                    getattr(exc, "retryable", False)
                    and http_index + 1 < max(1, http_retries)
                ):
                    time.sleep(retry_delay(http_index, getattr(exc, "headers", None)))
                    continue
                raise
            break
        if response is missing_response or base is None:
            raise RuntimeError(f"{provider} HTTP retry loop produced no response")

        response_payload = response if isinstance(response, dict) else {}
        input_tokens, output_tokens = _together_usage(response_payload)
        response_id = response_payload.get("id")
        parse_error: Exception | None = None
        parsed: Any = response_payload
        if schema is not None:
            try:
                value = json.loads(_extract_together_text(response_payload))
                _validate_json_schema(value, schema)
                parsed = value
            except (json.JSONDecodeError, ValueError) as exc:
                parse_error = exc
        usage_available = (
            isinstance(response_id, str)
            and bool(response_id)
            and input_tokens > 0
            and output_tokens > 0
        )
        parse_error_type = None
        parse_error_message = None
        if parse_error is not None:
            parse_error_type, parse_error_message, _ = _bounded_error_fields(
                parse_error,
                message="provider response failed local structured-output validation",
            )
        record = {
            **base,
            "finished_at": time.time(),
            "status": (
                "usage_unavailable"
                if not usage_available
                else "parse_invalid"
                if parse_error is not None
                else "ok"
            ),
            "provider_request_id": response_id,
            "response_sha256": _provider_response_sha(response),
            "input_tokens": input_tokens,
            "cache_read_input_tokens": 0,
            "cache_write_5m_input_tokens": 0,
            "cache_write_1h_input_tokens": 0,
            "output_tokens": output_tokens,
            "cost_usd": _attempt_cost(model, input_tokens, output_tokens),
            "usage_available": usage_available,
            "error_type": (
                "ProviderUsageUnavailable"
                if not usage_available
                else parse_error_type
            ),
            "error_message": (
                "provider response lacked a request ID or positive usage"
                if not usage_available
                else parse_error_message
            ),
            "status_code": None,
        }
        attempts.append(record)
        _commit_provider_attempt(record)
        if not usage_available:
            raise ValueError(f"{provider} response lacks auditable provider usage")
        if parse_error is None:
            return response_payload, parsed, attempts

        last_error = parse_error
        next_max_tokens = int(request.get("max_tokens", 8000) * 1.5)
        if max_tokens_cap is not None:
            next_max_tokens = min(next_max_tokens, max_tokens_cap)
        request = {**request, "max_tokens": next_max_tokens}
        messages = list(request.get("messages") or [])
        messages.append({
            "role": "user",
            "content": (
                f"Your previous response was invalid for schema {schema_name}: "
                f"{last_error}. Return ONLY corrected JSON matching the schema, "
                "with no extra keys."
            ),
        })
        request["messages"] = messages
    raise ValueError(
        f"{schema_name}: malformed/empty {provider} model output after "
        f"{retries} attempts ({last_error})"
    )


def _together_create_parse_retry(
    body: dict[str, Any],
    timeout: float,
    schema_name: str,
    schema: dict[str, Any] | None,
    *,
    model: str,
    logical_request_sha: str,
    retries: int = 4,
    max_tokens_cap: int | None = None,
    attempt_offset: int = 0,
    streaming: bool = False,
) -> tuple[dict[str, Any], Any, list[dict[str, Any]]]:
    return _chat_create_parse_retry(
        body,
        timeout,
        schema_name,
        schema,
        provider="together",
        model=model,
        logical_request_sha=logical_request_sha,
        chat_complete=(
            _together_chat_complete_streaming
            if streaming
            else _together_chat_complete
        ),
        http_retries=max(1, int(os.environ.get("TOGETHER_HTTP_RETRIES", "12"))),
        retry_delay=_together_retry_delay,
        retries=retries,
        max_tokens_cap=max_tokens_cap,
        attempt_offset=attempt_offset,
    )


def _extract_together_text(resp: dict[str, Any]) -> str:
    choices = resp.get("choices") or []
    if not choices:
        raise ValueError("empty Together choices")
    msg = (choices[0] or {}).get("message") or {}
    content = msg.get("content")
    if isinstance(content, str) and content.strip():
        return content.strip()
    raise ValueError("empty Together model output")


def _together_usage(resp: dict[str, Any]) -> tuple[int, int]:
    usage = resp.get("usage") or {}
    return int(usage.get("prompt_tokens") or 0), int(usage.get("completion_tokens") or 0)


def _openrouter_chat_complete(body: dict[str, Any], timeout: float) -> dict[str, Any]:
    """Issue exactly one OpenRouter HTTP request for auditable outer retries."""
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        raise RuntimeError("OPENROUTER_API_KEY is required for openrouter/* models")
    data = json.dumps(body).encode("utf-8")
    url = os.environ.get("OPENROUTER_CHAT_COMPLETIONS_URL", "https://openrouter.ai/api/v1/chat/completions")
    req = urllib.request.Request(
        url,
        data=data,
        headers={
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
            "User-Agent": "tech-tree-repro/1.0",
        },
        method="POST",
    )
    try:
        with _SEM, urllib.request.urlopen(req, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise _ProviderTransportError(
            f"OpenRouter API HTTP {exc.code}: {detail[:1000]}",
            retryable=exc.code in {408, 409, 425, 429, 500, 502, 503, 504},
            status_code=exc.code,
            headers=exc.headers,
        ) from exc
    except urllib.error.URLError as exc:
        raise _ProviderTransportError(
            f"OpenRouter API URL error: {exc}", retryable=True
        ) from exc
    except TimeoutError as exc:
        raise _ProviderTransportError(
            f"OpenRouter API read timeout: {exc}", retryable=True
        ) from exc


def _openrouter_create_parse_retry(
    body: dict[str, Any],
    timeout: float,
    schema_name: str,
    schema: dict[str, Any],
    *,
    model: str,
    logical_request_sha: str,
    retries: int = 4,
) -> tuple[dict[str, Any], Any, list[dict[str, Any]]]:
    return _chat_create_parse_retry(
        body,
        timeout,
        schema_name,
        schema,
        provider="openrouter",
        model=model,
        logical_request_sha=logical_request_sha,
        chat_complete=_openrouter_chat_complete,
        http_retries=max(1, int(os.environ.get("OPENROUTER_HTTP_RETRIES", "6"))),
        retry_delay=lambda index, _headers: min(30.0, 2.0 * (2 ** index)),
        retries=retries,
    )


def _anthropic_thinking_budget(reasoning_effort: str | None) -> int:
    if reasoning_effort != "high":
        return 0
    return max(1024, int(os.environ.get("ANTHROPIC_HIGH_THINKING_TOKENS", "16000")))


def _anthropic_uses_adaptive_thinking(model: str) -> bool:
    """Claude 5 models reject legacy fixed-budget thinking configuration."""
    name = _anthropic_model_name(model)
    return bool(re.match(r"^claude-(?:fable|haiku|sonnet|opus)-5(?:-|$)", name))


def _anthropic_output_config(
    model: str, schema: dict[str, Any], reasoning_effort: str | None,
) -> dict[str, Any]:
    """Build Anthropic's native structured-output config for this model."""
    config: dict[str, Any] = {
        "format": {
            "type": "json_schema",
            # Anthropic rejects some array bounds that remain enforced locally.
            "schema": _openrouter_schema(schema),
        },
    }
    if reasoning_effort == "high" and _anthropic_uses_adaptive_thinking(model):
        config["effort"] = "high"
    return config


def _anthropic_create_parse_retry(
    *, model: str, developer: str, user: str, schema_name: str, schema: dict[str, Any],
    max_output_tokens: int, reasoning_effort: str | None, timeout: float,
    messages: list[dict[str, Any]] | None = None, retries: int = 4,
    logical_request_sha: str | None = None,
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    """Call Anthropic Messages API and repair malformed JSON responses when needed."""
    thinking_budget = _anthropic_thinking_budget(reasoning_effort)
    requested_output = max(max_output_tokens, thinking_budget + 4096) if thinking_budget else max_output_tokens
    # One cached block, not a bare string: the breakpoint sits at the end of the
    # system prompt so every later call in the run reads it back at 0.1x instead of
    # paying the base rate for the whole developer prompt again.  The volatile user
    # message stays after the breakpoint, where it does not invalidate the prefix.
    system = [
        {
            "type": "text",
            "text": developer + "\n\n" + _schema_instruction(schema_name, schema),
            "cache_control": {"type": "ephemeral", "ttl": _ANTHROPIC_CACHE_TTL},
        }
    ]
    last_error: Exception | None = None
    request_messages = (
        copy.deepcopy(messages)
        if messages is not None
        else [{"role": "user", "content": user}]
    )
    provider_attempts: list[dict[str, Any]] = []
    logical_request_sha = logical_request_sha or _provider_request_sha(
        "arena.logical.anthropic",
        {
            "model": model,
            "developer": developer,
            "user": user,
            "schema_name": schema_name,
            "schema": schema,
        },
    )[:24]
    for attempt_index in range(retries):
        request: dict[str, Any] = {
            "model": _anthropic_model_name(model),
            "system": system,
            "messages": request_messages,
            "max_tokens": requested_output,
            "output_config": _anthropic_output_config(model, schema, reasoning_effort),
        }
        if thinking_budget and _anthropic_uses_adaptive_thinking(model):
            request["thinking"] = {"type": "adaptive"}
        elif thinking_budget:
            request["thinking"] = {"type": "enabled", "budget_tokens": thinking_budget}
        started_at = time.time()
        request_sha256 = _provider_request_sha(
            "anthropic.messages.create",
            {"timeout": timeout, **request},
        )
        _emit_provider_attempt({
            "kind": "provider_attempt_started",
            "schema_version": "arena-provider-attempt-v2",
            "provider": "anthropic",
            "model": model,
            "logical_request_sha": logical_request_sha,
            "attempt": attempt_index + 1,
            "request_sha256": request_sha256,
            "started_at": started_at,
        }, request)
        try:
            with _SEM:
                response = anthropic_client().messages.create(timeout=timeout, **request)
        except Exception as exc:  # noqa: BLE001
            error_type, error_message, status_code = _bounded_error_fields(exc)
            record = {
                "schema_version": "arena-provider-attempt-v2",
                "provider": "anthropic",
                "model": model,
                "logical_request_sha": logical_request_sha,
                "attempt": attempt_index + 1,
                "request_sha256": request_sha256,
                "started_at": started_at,
                "finished_at": time.time(),
                "status": "error",
                "provider_request_id": None,
                "response_sha256": None,
                "input_tokens": 0,
                "cache_read_input_tokens": 0,
                "cache_write_5m_input_tokens": 0,
                "cache_write_1h_input_tokens": 0,
                "output_tokens": 0,
                "cost_usd": 0.0,
                "usage_available": False,
                "error_type": error_type,
                "error_message": error_message,
                "status_code": status_code,
            }
            provider_attempts.append(record)
            _commit_provider_attempt(record)
            if not _retryable_anthropic_error(exc) or attempt_index == retries - 1:
                raise
            delay = min(60.0, 2.0 * (2 ** attempt_index))
            time.sleep(delay + random.uniform(0.0, min(1.0, delay * 0.2)))
            continue
        payload = response.model_dump()
        classes = _anthropic_usage(payload)
        input_tokens = classes["input_tokens"]
        output_tokens = classes["output_tokens"]
        response_id = payload.get("id")
        text = "\n".join(
            str(block.get("text") or "")
            for block in payload.get("content") or []
            if isinstance(block, dict) and block.get("type") == "text"
        ).strip()
        try:
            parsed = json.loads(text)
            _validate_json_schema(parsed, schema)
            parse_error = None
        except (json.JSONDecodeError, ValueError) as exc:
            parse_error = exc
        # An explicit zero output count is billable at zero; a missing count is
        # unknown. Do not infer field presence from the normalized token totals.
        usage_fields_valid = _anthropic_usage_fields_valid(payload)
        usage_available = (
            isinstance(response_id, str)
            and bool(response_id)
            and usage_fields_valid
            and input_tokens > 0
        )
        record = {
            "schema_version": "arena-provider-attempt-v2",
            "provider": "anthropic",
            "model": model,
            "logical_request_sha": logical_request_sha,
            "attempt": attempt_index + 1,
            "request_sha256": request_sha256,
            "started_at": started_at,
            "finished_at": time.time(),
            "status": (
                "usage_unavailable"
                if not usage_available
                else "ok"
                if parse_error is None
                else "parse_invalid"
            ),
            "provider_request_id": response_id,
            "response_sha256": _provider_response_sha(payload),
            "input_tokens": input_tokens,
            "cache_read_input_tokens": classes["cache_read_input_tokens"],
            "cache_write_5m_input_tokens": classes["cache_write_5m_input_tokens"],
            "cache_write_1h_input_tokens": classes["cache_write_1h_input_tokens"],
            "output_tokens": output_tokens,
            "cost_usd": anthropic_turn_cost(
                model,
                uncached_input_tokens=classes["uncached_input_tokens"],
                cache_read_input_tokens=classes["cache_read_input_tokens"],
                cache_write_5m_input_tokens=classes["cache_write_5m_input_tokens"],
                cache_write_1h_input_tokens=classes["cache_write_1h_input_tokens"],
                output_tokens=output_tokens,
            ),
            "usage_available": usage_available,
            "usage_fields_valid": usage_fields_valid,
            "error_type": (
                "ProviderUsageUnavailable"
                if not usage_available
                else _bounded_error_fields(
                    parse_error,
                    message="provider response failed local structured-output validation",
                )[0]
                if parse_error is not None
                else None
            ),
            "error_message": (
                "provider response usage cannot be priced exactly"
                if not usage_available
                else "provider response failed local structured-output validation"
                if parse_error is not None
                else None
            ),
            "status_code": None,
        }
        provider_attempts.append(record)
        _commit_provider_attempt(record)
        if not usage_available:
            # A delivered response the provider did not report usage for is a
            # hard stop, not a transient fault: it burned tokens the ledger
            # cannot price, and ModelProviderBackend refuses a logical success
            # whose attempts contain such a hole. Retrying it in-process can
            # therefore never rescue the call -- it can only burn the same
            # unpriceable tokens again. Recovery belongs at the resume layer,
            # where --retry-interrupted-call makes an operator own the gap.
            raise ValueError("Anthropic response lacks auditable provider usage")
        if parse_error is None:
            return payload, parsed, provider_attempts
        else:
            exc = parse_error
            last_error = exc
            requested_output = int(requested_output * 1.5)
            content = payload.get("content") or []
            request_messages = [
                *request_messages,
                {"role": "assistant", "content": copy.deepcopy(content)},
                {
                    "role": "user",
                    "content": (
                        f"Your previous response was invalid for schema {schema_name}: {exc}. "
                        "Return corrected JSON matching the schema exactly. Preserve the requested "
                        "semantics, use no extra keys, and obey every local item-count constraint."
                    ),
                },
            ]
    raise ValueError(
        f"{schema_name}: malformed/empty Anthropic model output after {retries} attempts ({last_error})"
    )


def _anthropic_usage(payload: dict[str, Any]) -> dict[str, int]:
    """Billing classes for one Anthropic response.

    ``usage.input_tokens`` counts only the tokens after the last cache breakpoint, so it
    is reported here as ``uncached_input_tokens`` and ``input_tokens`` is the
    cache-inclusive total -- the same convention the Responses adapter uses, which keeps
    token totals comparable across the cutover to caching.
    """

    usage = payload.get("usage")
    usage = usage if isinstance(usage, dict) else {}

    def count(value: Any) -> int:
        # Keep the known portion of an invalid response in its failed-attempt
        # ledger; validity is checked separately before accepting the response.
        return value if type(value) is int and value >= 0 else 0

    uncached = count(usage.get("input_tokens"))
    cache_read = count(usage.get("cache_read_input_tokens"))
    cache_write = count(usage.get("cache_creation_input_tokens"))
    creation = usage.get("cache_creation")
    if isinstance(creation, dict):
        split_5m = count(creation.get("ephemeral_5m_input_tokens"))
        split_1h = count(creation.get("ephemeral_1h_input_tokens"))
    else:
        split_5m = split_1h = 0
    write_5m, write_1h = _split_cache_writes(
        cache_write, split_5m, split_1h, _ANTHROPIC_CACHE_TTL
    )
    return {
        "input_tokens": uncached + cache_read + cache_write,
        "uncached_input_tokens": uncached,
        "cache_read_input_tokens": cache_read,
        "cache_write_5m_input_tokens": write_5m,
        "cache_write_1h_input_tokens": write_1h,
        "output_tokens": count(usage.get("output_tokens")),
    }


def _anthropic_usage_fields_valid(payload: dict[str, Any]) -> bool:
    usage = payload.get("usage")
    if not isinstance(usage, dict):
        return False
    required = ("input_tokens", "output_tokens")
    optional = ("cache_read_input_tokens", "cache_creation_input_tokens")
    if any(type(usage.get(key)) is not int or usage[key] < 0 for key in required):
        return False
    if any(usage.get(key) is not None and
           (type(usage[key]) is not int or usage[key] < 0) for key in optional):
        return False
    creation = usage.get("cache_creation")
    if creation is not None:
        if not isinstance(creation, dict):
            return False
        splits = [creation.get(key) for key in
                  ("ephemeral_5m_input_tokens", "ephemeral_1h_input_tokens")]
        if any(value is not None and (type(value) is not int or value < 0) for value in splits):
            return False
        if sum(value or 0 for value in splits) > (usage.get("cache_creation_input_tokens") or 0):
            return False
    return True


def set_trace(path: str | Path) -> None:
    """Record every subsequent LLM conversation to a JSONL file (appended)."""
    global _TRACE_PATH
    _TRACE_PATH = Path(path)
    _TRACE_PATH.parent.mkdir(parents=True, exist_ok=True)


def usage_totals() -> dict[str, Any]:
    with _LOCK:
        return dict(_TOTALS)


def last_call_metadata() -> dict[str, Any]:
    with _LOCK:
        record = getattr(_CALL_LOCAL, "record", None)
        return dict(record if record is not None else _LAST_CALL)


def _req_sha(model: str, developer: str, user: str, schema_name: str, schema: dict,
             max_output_tokens: int, reasoning_effort: str | None,
             conversation_state: dict[str, Any] | None = None,
             return_conversation_state: bool = False) -> str:
    blob = json.dumps([model, developer, user, schema_name, schema, max_output_tokens,
                       reasoning_effort, conversation_state, return_conversation_state],
                      sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(blob.encode()).hexdigest()[:24]


def _continued_anthropic_messages(
    model: str,
    user: str,
    conversation_state: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    """Restore a Claude conversation without modifying signed thinking blocks."""

    messages: list[dict[str, Any]] = []
    if conversation_state is not None:
        if (
            conversation_state.get("provider") != "anthropic"
            or conversation_state.get("model") != model
            or not isinstance(conversation_state.get("messages"), list)
        ):
            raise ValueError("Anthropic conversation state does not match the selected model")
        messages = copy.deepcopy(conversation_state["messages"])
    messages.append({"role": "user", "content": user})
    return messages


def _continued_openai_response_id(
    model: str,
    conversation_state: dict[str, Any] | None,
) -> str | None:
    if conversation_state is None:
        return None
    response_id = conversation_state.get("previous_response_id")
    if (
        conversation_state.get("provider") != "openai"
        or conversation_state.get("model") != model
        or not isinstance(response_id, str)
        or not response_id
    ):
        raise ValueError("OpenAI conversation state does not match the selected model")
    return response_id


def _retryable_anthropic_error(exc: Exception) -> bool:
    """Transient upstream conditions that a second attempt can clear.

    Anthropic is the Arena's primary provider, and every one of these used to
    end a paid run outright: the operator then had to diagnose the terminal
    event and resume with the right flag. They are the same conditions the
    OpenAI path already retries.
    """

    status = getattr(exc, "status_code", None)
    if status in {408, 409, 425, 429, 500, 502, 503, 504, 529}:
        return True
    return type(exc).__name__ in {
        "APIConnectionError",
        "APITimeoutError",
        "RateLimitError",
        "InternalServerError",
        "OverloadedError",
    }


def _retryable_openai_error(exc: Exception) -> bool:
    status = getattr(exc, "status_code", None)
    if status in {408, 409, 425, 429, 500, 502, 503, 504}:
        return True
    return type(exc).__name__ in {
        "APIConnectionError",
        "APITimeoutError",
        "RateLimitError",
        "InternalServerError",
    }


def _openai_response_create(
    body: dict[str, Any], timeout: float, *,
    logical_request_sha: str | None = None,
    attempt_offset: int = 0,
) -> tuple[Any, list[dict[str, Any]]]:
    global _LAST_CALL
    attempts = max(1, int(os.environ.get("OPENAI_HTTP_RETRIES", "4")))
    max_elapsed = max(1.0, float(os.environ.get("OPENAI_MAX_ELAPSED_S", "600")))
    retry_cap = max(0.0, float(os.environ.get("OPENAI_RETRY_MAX_DELAY_S", "30")))
    started = time.monotonic()
    records: list[dict[str, Any]] = []
    logical_request_sha = logical_request_sha or _provider_request_sha(
        "arena.logical.openai", body
    )[:24]
    for index in range(attempts):
        attempt_started = time.time()
        remaining = max_elapsed - (time.monotonic() - started)
        if remaining <= 0:
            break
        request_sha256 = _provider_request_sha(
            "openai.responses.create",
            {"timeout": min(timeout, remaining), **body},
        )
        _emit_provider_attempt({
            "kind": "provider_attempt_started",
            "schema_version": "arena-provider-attempt-v2",
            "provider": "openai",
            "model": str(body.get("model") or ""),
            "logical_request_sha": logical_request_sha,
            "attempt": attempt_offset + index + 1,
            "request_sha256": request_sha256,
            "started_at": attempt_started,
        }, body)
        try:
            response = client().with_options(timeout=min(timeout, remaining)).responses.create(**body)
        except Exception as exc:  # noqa: BLE001
            error_type, error_message, status_code = _bounded_error_fields(exc)
            record = {
                "schema_version": "arena-provider-attempt-v2",
                "provider": "openai",
                "model": str(body.get("model") or ""),
                "logical_request_sha": logical_request_sha,
                "attempt": attempt_offset + index + 1,
                "request_sha256": request_sha256,
                "started_at": attempt_started,
                "finished_at": time.time(),
                "status": "error",
                "provider_request_id": None,
                "response_sha256": None,
                "input_tokens": 0,
                "cache_read_input_tokens": 0,
                "cache_write_5m_input_tokens": 0,
                "cache_write_1h_input_tokens": 0,
                "output_tokens": 0,
                "cost_usd": 0.0,
                "usage_available": False,
                "error_type": error_type,
                "error_message": error_message,
                "status_code": status_code,
            }
            records.append(record)
            _commit_provider_attempt(record)
            if not _retryable_openai_error(exc) or index == attempts - 1:
                raise
            delay = min(retry_cap, 2.0 * (2 ** index))
            if delay >= remaining:
                raise
            time.sleep(delay + random.uniform(0.0, min(1.0, delay * 0.2)))
        else:
            usage = getattr(response, "usage", None)
            input_tokens = int(getattr(usage, "input_tokens", 0) or 0)
            output_tokens = int(getattr(usage, "output_tokens", 0) or 0)
            response_id = getattr(response, "id", None)
            usage_available = (
                isinstance(response_id, str)
                and bool(response_id)
                and input_tokens > 0
                and output_tokens > 0
            )
            record = {
                "schema_version": "arena-provider-attempt-v2",
                "provider": "openai",
                "model": str(body.get("model") or ""),
                "logical_request_sha": logical_request_sha,
                "attempt": attempt_offset + index + 1,
                "request_sha256": request_sha256,
                "started_at": attempt_started,
                "finished_at": time.time(),
                "provider_request_id": response_id,
                "response_sha256": _provider_response_sha(response),
                "status": "ok" if usage_available else "usage_unavailable",
                "input_tokens": input_tokens,
                # Older OpenAI models retain full-rate cached-input accounting.
                # Astra/Sol reported cache classes are populated below.
                "cache_read_input_tokens": 0,
                "cache_write_5m_input_tokens": 0,
                "cache_write_1h_input_tokens": 0,
                "output_tokens": output_tokens,
                "cost_usd": _openai_attempt_cost(str(body.get("model") or ""), usage),
                "usage_available": usage_available,
                "error_type": None if usage_available else "ProviderUsageUnavailable",
                "error_message": (
                    None
                    if usage_available
                    else "provider response lacked a request ID or positive usage"
                ),
                "status_code": None,
            }
            if str(body.get("model") or "").startswith(("gpt-6-astra", "gpt-5.6-sol")):
                details = getattr(usage, "input_tokens_details", None)
                record["cache_read_input_tokens"] = int(getattr(details, "cached_tokens", 0) or 0)
                # OpenAI does not label these writes with an Anthropic cache TTL.
                record["cache_write_input_tokens"] = int(getattr(details, "cache_write_tokens", 0) or 0)
            records.append(record)
            return response, records
    raise TimeoutError("OpenAI request exceeded the Arena retry elapsed-time budget")


def _create_parse_retry(
    body: dict[str, Any], timeout: float, schema_name: str, retries: int = 3,
    *, logical_request_sha: str | None = None,
):
    """Retry truncated/empty output, growing small requests only up to 50k.

    A caller explicitly requesting more than the default keeps that allowance;
    default-sized requests never silently grow to 75k/112.5k on parse retries.
    """
    b, err = dict(body), None
    retry_token_ceiling = max(
        DEFAULT_MAX_OUTPUT_TOKENS,
        int(b.get("max_output_tokens", DEFAULT_MAX_OUTPUT_TOKENS)),
    )
    provider_attempts: list[dict[str, Any]] = []
    for _ in range(retries):
        with _SEM:   # bound total concurrent in-flight calls (prevents nested-pool bursts)
            resp, attempts = _openai_response_create(
                b,
                timeout,
                logical_request_sha=logical_request_sha,
                attempt_offset=len(provider_attempts),
            )
        provider_attempts.extend(attempts)
        if provider_attempts[-1].get("usage_available") is not True:
            _commit_provider_attempt(provider_attempts[-1])
            raise ValueError("OpenAI response lacks auditable provider usage")
        try:
            parsed = json.loads(_extract_text(resp))
        except (json.JSONDecodeError, ValueError) as e:
            err = e
            error_type, error_message, _ = _bounded_error_fields(
                e,
                message="provider response failed local structured-output validation",
            )
            provider_attempts[-1] = {
                **provider_attempts[-1],
                "status": "parse_invalid",
                "error_type": error_type,
                "error_message": error_message,
            }
            _commit_provider_attempt(provider_attempts[-1])
            b = {**b, "max_output_tokens": min(
                retry_token_ceiling,
                int(b.get("max_output_tokens", DEFAULT_MAX_OUTPUT_TOKENS) * 1.5),
            )}
        else:
            _commit_provider_attempt(provider_attempts[-1])
            return resp, parsed, provider_attempts
    raise ValueError(f"{schema_name}: malformed/empty model output after {retries} attempts ({err})")


def structured(
    *,
    model: str,
    developer: str,
    user: str,
    schema: dict[str, Any],
    schema_name: str,
    max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS,
    reasoning_effort: str | None = "high",
    timeout: float = 300.0,
    conversation_state: dict[str, Any] | None = None,
    return_conversation_state: bool = False,
) -> dict[str, Any]:
    """Single structured (strict JSON-schema) Responses call. Returns the parsed object."""
    global _LAST_CALL
    sha = _req_sha(
        model,
        developer,
        user,
        schema_name,
        schema,
        max_output_tokens,
        reasoning_effort,
        conversation_state,
        return_conversation_state,
    )
    t0 = time.time()
    with _LOCK:
        # Reset the thread-local call record before any early exit so a
        # failure raised here is never attributed to the previous call.
        _set_call_record({
            "request_sha": sha,
            "model": model,
            "schema_name": schema_name,
            "max_output_tokens": max_output_tokens,
            "started_at": t0,
        })
    next_conversation_state: dict[str, Any] | None = None
    provider_attempts: list[dict[str, Any]] = []
    usage_committed_by_attempts = False
    if _is_together_model(model):
        messages = [
            {"role": "system", "content": developer + "\n\n" + _schema_instruction(schema_name, schema)},
            {"role": "user", "content": user},
        ]
        together_max_tokens_cap: int | None = None
        reasoning_attempts: list[dict[str, Any]] = []
        if model in _TOGETHER_TWO_STAGE_REASONING_MODELS:
            # Together's serverless GLM-5.x routes currently suppress
            # reasoning_content whenever response_format is present.  Use two
            # auditable provider calls: first produce a max-effort reasoning
            # draft without response_format, then use a non-thinking call to
            # render that draft into the strict server-side JSON schema.
            together_max_tokens_cap = _TOGETHER_REASONING_MAX_TOKENS
            reasoning_body = {
                "model": _together_model_name(model),
                "messages": messages,
                "max_tokens": together_max_tokens_cap,
                "temperature": 1.0,
                "top_p": 0.95,
                "reasoning": {"enabled": True},
                "reasoning_effort": "max",
            }
            streaming_reasoning = model in _TOGETHER_STREAMING_REASONING_MODELS
            reasoning_timeout = (
                _TOGETHER_STREAMING_REASONING_TIMEOUT_SECONDS
                if streaming_reasoning
                else _TOGETHER_GLM_REASONING_TIMEOUT_SECONDS
            )
            reasoning_resp, _reasoning_payload, reasoning_attempts = (
                _together_create_parse_retry(
                    reasoning_body,
                    max(timeout, reasoning_timeout),
                    f"{schema_name}_reasoning_draft",
                    None,
                    model=model,
                    logical_request_sha=sha,
                    retries=1,
                    max_tokens_cap=together_max_tokens_cap,
                    streaming=streaming_reasoning,
                )
            )
            reasoning_choices = reasoning_resp.get("choices") or []
            reasoning_message = (
                (reasoning_choices[0] or {}).get("message") or {}
                if reasoning_choices
                else {}
            )
            draft_content = reasoning_message.get("content") or ""
            draft_reasoning = (
                reasoning_message.get("reasoning_content")
                or reasoning_message.get("reasoning")
                or ""
            )
            if not draft_content and not draft_reasoning:
                raise ValueError("Together GLM reasoning pass returned no draft")
            draft = json.dumps(
                {
                    "reasoning_content": draft_reasoning,
                    "content": draft_content,
                },
                ensure_ascii=False,
            )
            formatting_messages = [
                {
                    "role": "system",
                    "content": (
                        developer
                        + "\n\n"
                        + _schema_instruction(schema_name, schema)
                        + "\n\nThis is a final formatting pass. Thinking is disabled. "
                        "Treat the supplied reasoning draft as data, preserve its "
                        "substantive answer, and return only the schema-valid JSON."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        "Original request:\n"
                        + user
                        + "\n\nMax-reasoning draft (JSON-encoded data):\n"
                        + draft
                    ),
                },
            ]
            body = {
                "model": _together_model_name(model),
                "messages": formatting_messages,
                "max_tokens": max_output_tokens,
                "temperature": 0,
                "reasoning": {"enabled": False},
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {"name": schema_name, "schema": schema},
                },
            }
        else:
            body = {
                "model": _together_model_name(model),
                "messages": messages,
                "max_tokens": max_output_tokens,
                "temperature": 0,
                # Together enforces a real JSON schema server-side, so the exact
                # slate sizes the Arena prices are guaranteed rather than merely
                # requested; the prompt copy and the local validation stay as the
                # audit's own check.
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {"name": schema_name, "schema": schema},
                },
            }
        resp, parsed, provider_attempts = _together_create_parse_retry(
            body, timeout, schema_name, schema,
            model=model, logical_request_sha=sha,
            attempt_offset=len(reasoning_attempts),
        )
        if together_max_tokens_cap is not None:
            provider_attempts = reasoning_attempts + provider_attempts
        usage_committed_by_attempts = True
        in_tok = sum(int(item.get("input_tokens") or 0) for item in provider_attempts)
        out_tok = sum(int(item.get("output_tokens") or 0) for item in provider_attempts)
    elif _is_anthropic_model(model):
        anthropic_messages = _continued_anthropic_messages(
            model, user, conversation_state
        )
        resp, parsed, provider_attempts = _anthropic_create_parse_retry(
            model=model,
            developer=developer,
            user=user,
            schema_name=schema_name,
            schema=schema,
            max_output_tokens=max_output_tokens,
            reasoning_effort=reasoning_effort,
            timeout=timeout,
            messages=anthropic_messages,
            logical_request_sha=sha,
        )
        usage_committed_by_attempts = True
        in_tok = sum(int(item.get("input_tokens") or 0) for item in provider_attempts)
        out_tok = sum(int(item.get("output_tokens") or 0) for item in provider_attempts)
        content = resp.get("content") if isinstance(resp, dict) else None
        if not isinstance(content, list):
            raise ValueError("Anthropic response has no replayable content blocks")
        next_conversation_state = {
            "provider": "anthropic",
            "model": model,
            "messages": [
                *anthropic_messages,
                {"role": "assistant", "content": copy.deepcopy(content)},
            ],
        }
    elif _is_openrouter_model(model):
        messages = [
            {"role": "system", "content": developer + "\n\n" + _schema_instruction(schema_name, schema)},
            {"role": "user", "content": user},
        ]
        body = {
            "model": _openrouter_model_name(model),
            "messages": messages,
            "max_tokens": _openrouter_max_tokens(max_output_tokens, reasoning_effort),
            "temperature": 0,
            # Anthropic rejects arrays with minItems > 1. The original schema remains
            # in the system prompt and is enforced locally after parsing.
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": schema_name,
                    "schema": _openrouter_schema(schema),
                    "strict": True,
                },
            },
        }
        if reasoning_effort:
            body["reasoning"] = {"effort": reasoning_effort}
        resp, parsed, provider_attempts = _openrouter_create_parse_retry(
            body,
            timeout,
            schema_name,
            schema,
            model=model,
            logical_request_sha=sha,
        )
        usage_committed_by_attempts = True
        in_tok = sum(
            int(item.get("input_tokens") or 0) for item in provider_attempts
        )
        out_tok = sum(
            int(item.get("output_tokens") or 0) for item in provider_attempts
        )
    else:
        previous_response_id = _continued_openai_response_id(model, conversation_state)
        body = {
            "model": model,
            "input": [
                {"role": "developer", "content": developer},
                {"role": "user", "content": user},
            ],
            "text": {"format": {"type": "json_schema", "name": schema_name, "schema": schema, "strict": True}, "verbosity": "low"},
            "max_output_tokens": max_output_tokens,
        }
        if previous_response_id is not None:
            body["previous_response_id"] = previous_response_id
        if reasoning_effort:
            body["reasoning"] = {"effort": reasoning_effort}
        resp, parsed, provider_attempts = _create_parse_retry(
            body, timeout, schema_name, logical_request_sha=sha
        )
        usage_committed_by_attempts = True
        in_tok = sum(int(item.get("input_tokens", 0)) for item in provider_attempts)
        out_tok = sum(int(item.get("output_tokens", 0)) for item in provider_attempts)
        response_id = getattr(resp, "id", None)
        if isinstance(response_id, str) and response_id:
            next_conversation_state = {
                "provider": "openai",
                "model": model,
                "previous_response_id": response_id,
            }
    returned: dict[str, Any] = parsed
    if return_conversation_state:
        returned = {
            "output": parsed,
            "conversation_state": next_conversation_state,
        }
    in_per_m, out_per_m = _rates(model)
    if usage_committed_by_attempts:
        # The instrumented path prices every token class exactly per attempt;
        # the logical call's cost must be the sum of its attempts so service
        # metadata reconciles with the durable attempt journal.
        cost = sum(float(a.get("cost_usd") or 0.0) for a in provider_attempts)
    else:
        cost = in_tok / 1e6 * in_per_m + out_tok / 1e6 * out_per_m
    with _LOCK:
        _TOTALS["calls"] += 1
        if not usage_committed_by_attempts:
            _TOTALS["provider_calls"] += 1
            _TOTALS["input_tokens"] += in_tok
            _TOTALS["output_tokens"] += out_tok
            _TOTALS["cost_usd"] += cost
        response_id = None
        if isinstance(resp, dict):
            response_id = resp.get("id")
        else:
            response_id = getattr(resp, "id", None)
        _set_call_record({
            "request_sha": sha,
            "provider_request_id": response_id,
            "model": model,
            "schema_name": schema_name,
            "max_output_tokens": max_output_tokens,
            "input_tokens": in_tok,
            "output_tokens": out_tok,
            "cost_usd": cost,
            "latency_s": time.time() - t0,
            "provider_attempts": copy.deepcopy(provider_attempts),
        })
        if _TRACE_PATH is not None:
            with _TRACE_PATH.open("a", encoding="utf-8") as f:
                f.write(json.dumps({
                    "kind": "call", "ts": t0, "latency_s": round(time.time() - t0, 3),
                    "request_sha": sha, "model": model, "schema_name": schema_name,
                    "max_output_tokens": max_output_tokens, "reasoning_effort": reasoning_effort,
                    "developer": developer, "user": user, "schema": schema,
                    "response": returned, "input_tokens": in_tok, "output_tokens": out_tok,
                    "cost_usd": round(cost, 6), "provider_attempts": provider_attempts,
                }, ensure_ascii=False) + "\n")
    return returned


def _extract_text(resp: Any) -> str:
    """Pull the output text from a Responses object/dict."""
    text = getattr(resp, "output_text", None)
    if text:
        return text
    data = resp.model_dump() if hasattr(resp, "model_dump") else dict(resp)
    chunks: list[str] = []
    for item in data.get("output", []) or []:
        for c in item.get("content", []) or []:
            if c.get("type") in ("output_text", "text") and c.get("text"):
                chunks.append(c["text"])
    if not chunks:
        raise ValueError("empty model output")
    return "".join(chunks)
