"""Arena-owned model backends.

Participants receive only :class:`ReplayableServices`; credentials and provider
selection stay on the trusted side of this adapter.
"""

from __future__ import annotations

import threading
import time
from typing import Any, Callable


class ModelProviderBackend:
    """Use the Arena-owned structured-model provider client."""

    _source_usage_lock = threading.RLock()

    def __init__(
        self,
        *,
        initial_usage: dict[str, Any] | None = None,
        attempt_sink: Callable[[dict[str, Any]], None] | None = None,
    ) -> None:
        self._usage = {
            "calls": 0,
            "provider_calls": 0,
            "unknown_provider_attempts": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "cost_usd": 0.0,
        }
        for key, value in (initial_usage or {}).items():
            if key in self._usage:
                self._usage[key] = value
        self._last_call: dict[str, Any] = {}
        self._thread_call = threading.local()
        self._attempt_sink = attempt_sink
        self.attempt_guard = None
        self._lock = threading.RLock()

    def set_attempt_sink(
        self, sink: Callable[[dict[str, Any]], None] | None
    ) -> None:
        with self._lock:
            self._attempt_sink = sink

    @staticmethod
    def _call_usage(details: dict[str, Any], *, succeeded: bool) -> dict[str, Any]:
        """Exact per-call usage from this call's own provider attempts.

        The provider client keeps the in-flight call record in thread-local
        storage, so concurrent logical calls stay individually attributable
        without serializing the underlying HTTP requests.
        """

        attempts = [
            attempt
            for attempt in (details.get("provider_attempts") or [])
            if isinstance(attempt, dict)
        ]
        return {
            "calls": 1 if succeeded else 0,
            "provider_calls": len(attempts),
            # A request that failed outright carries no response and no
            # tokens: it is fully accounted at zero, so it is not an unknown.
            # Only a delivered response whose usage cannot be priced leaves a
            # hole, and that is what must block a logical success.
            "unknown_provider_attempts": sum(
                attempt.get("usage_available") is not True
                and attempt.get("status") != "error"
                for attempt in attempts
            ),
            "input_tokens": sum(
                int(attempt.get("input_tokens") or 0) for attempt in attempts
            ),
            "output_tokens": sum(
                int(attempt.get("output_tokens") or 0) for attempt in attempts
            ),
            "cost_usd": sum(
                float(attempt.get("cost_usd") or 0.0) for attempt in attempts
            ),
        }

    def structured(self, **request: Any) -> Any:
        from . import provider_client

        started_at = time.time()
        try:
            with provider_client.provider_attempt_sink(self._attempt_sink), provider_client.provider_attempt_guard(self.attempt_guard):
                response = provider_client.structured(**request)
        except BaseException as exc:
            details_fn = getattr(provider_client, "last_call_metadata", None)
            details = details_fn() if callable(details_fn) else {}
            delta = self._call_usage(details, succeeded=False)
            with self._lock:
                for key, value in delta.items():
                    self._usage[key] += value
            error_type, error_message, _ = provider_client._bounded_error_fields(exc)
            metadata = {
                "model": request.get("model"),
                "schema_name": request.get("schema_name"),
                "latency_s": time.time() - started_at,
                "usage": delta,
                "error_type": error_type,
                "error_message": error_message,
                **details,
            }
            with self._lock:
                self._last_call = metadata
            self._thread_call.metadata = dict(metadata)
            raise
        else:
            details_fn = getattr(provider_client, "last_call_metadata", None)
            details = details_fn() if callable(details_fn) else {}
            delta = self._call_usage(details, succeeded=True)
            with self._lock:
                for key, value in delta.items():
                    self._usage[key] += value
            metadata = {
                "model": request.get("model"),
                "schema_name": request.get("schema_name"),
                "latency_s": time.time() - started_at,
                "usage": delta,
                **details,
            }
            with self._lock:
                self._last_call = metadata
            self._thread_call.metadata = dict(metadata)
            if int(delta.get("unknown_provider_attempts", 0)):
                raise RuntimeError(
                    "provider attempt usage is incomplete; refusing logical success"
                )
            return response

    def usage_totals(self) -> dict[str, Any]:
        with self._lock:
            return dict(self._usage)

    def last_call_metadata(self) -> dict[str, Any]:
        with self._lock:
            metadata = getattr(self._thread_call, "metadata", None)
            return dict(metadata if isinstance(metadata, dict) else self._last_call)

    def restore_usage(self, usage: dict[str, Any]) -> None:
        with self._lock:
            for key in self._usage:
                self._usage[key] = usage.get(key, 0)


class ScriptedModelBackend:
    """Small deterministic backend useful for offline adapter tests."""

    def __init__(self, responses: list[dict[str, Any]]) -> None:
        self._responses = list(responses)
        self.requests: list[dict[str, Any]] = []

    def structured(self, **request: Any) -> Any:
        self.requests.append(request)
        if not self._responses:
            raise RuntimeError("scripted model response tape is exhausted")
        return self._responses.pop(0)
