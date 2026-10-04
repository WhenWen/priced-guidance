"""Replayable services injected into participant actors."""

from __future__ import annotations

import copy
import hashlib
import json
import os
import random
import re
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from typing import Any, Callable, Protocol

from ..errors import ReplayDivergence, ResourceLimitExceeded
from ..model_defaults import DEFAULT_MAX_OUTPUT_TOKENS


class StructuredModelBackend(Protocol):
    def structured(self, **request: Any) -> Any: ...


class AgentSessionBackend(Protocol):
    def turn(self, **request: Any) -> Any: ...


class DisabledModelBackend:
    def structured(self, **request: Any) -> Any:
        raise RuntimeError("this run has no live model backend")


@dataclass(frozen=True, slots=True)
class ServiceEvent:
    kind: str
    request_hash: str
    response: Any
    error: str | None = None
    request: Any | None = None
    error_message: str | None = None
    started_at: float | None = None
    finished_at: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ServiceLimits:
    max_model_calls: int = 256
    max_random_calls: int = 100_000
    max_output_tokens_per_call: int = DEFAULT_MAX_OUTPUT_TOKENS
    max_total_tokens: int = 10_000_000
    max_model_cost_usd: float = 1_000.0
    max_request_bytes: int = 2_000_000


@dataclass(slots=True)
class ServiceMeter:
    """Factory-shared counters which survive actor reconstruction and checkout."""

    model_calls: int = 0
    random_calls: int = 0
    lock: threading.RLock = field(default_factory=threading.RLock, repr=False)


def _request_hash(kind: str, request: Any) -> str:
    try:
        encoded = json.dumps(
            request,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except (TypeError, ValueError, UnicodeError) as exc:
        raise ReplayDivergence("service request is not canonical JSON") from exc
    return hashlib.sha256(kind.encode("utf-8") + b"\0" + encoded).hexdigest()


_BUILTIN_REPLAY_ERROR_TYPES: dict[str, type[Exception]] = {
    "ArithmeticError": ArithmeticError,
    "ConnectionError": ConnectionError,
    "IndexError": IndexError,
    "LookupError": LookupError,
    "RuntimeError": RuntimeError,
    "TimeoutError": TimeoutError,
    "TypeError": TypeError,
    "ValueError": ValueError,
}

# Provider clients expose these stable error names in recorded service calls.
# Recreate only inert, dependency-free exception classes from this explicit
# whitelist; never import or instantiate a class named by an untrusted tape.
_PROVIDER_REPLAY_ERROR_BASES: dict[str, type[Exception]] = {
    "NativeCallError": RuntimeError,
    "OpenAIError": RuntimeError,
    "APIError": RuntimeError,
    "APIConnectionError": ConnectionError,
    "APITimeoutError": TimeoutError,
    "APIStatusError": RuntimeError,
    "APIResponseValidationError": RuntimeError,
    "BadRequestError": RuntimeError,
    "AuthenticationError": RuntimeError,
    "PermissionDeniedError": RuntimeError,
    "NotFoundError": RuntimeError,
    "ConflictError": RuntimeError,
    "UnprocessableEntityError": RuntimeError,
    "RateLimitError": RuntimeError,
    "InternalServerError": RuntimeError,
    "OverloadedError": RuntimeError,
}
_PROVIDER_REPLAY_ERROR_TYPES = {
    name: type(name, (base,), {"__module__": __name__})
    for name, base in _PROVIDER_REPLAY_ERROR_BASES.items()
}
_REPLAY_ERROR_TYPES: dict[str, type[Exception]] = {
    **_BUILTIN_REPLAY_ERROR_TYPES,
    **_PROVIDER_REPLAY_ERROR_TYPES,
}
SAFE_REPLAY_ERROR_NAMES = frozenset(
    {"ResourceLimitExceeded", "BudgetBlocked", "ReplayDivergence", *_REPLAY_ERROR_TYPES}
)


def _bounded_service_error(
    exc: BaseException, *, sensitive: bool
) -> tuple[str, str]:
    """Describe a live failure without persisting provider/request plaintext."""

    error_type = re.sub(r"[^A-Za-z0-9_.-]", "_", type(exc).__name__)[:64]
    if not error_type or not error_type[0].isalpha():
        error_type = "RuntimeError"
    if sensitive:
        return error_type, "service call failed"
    message = str(exc)
    if len(message.encode("utf-8")) > 160:
        message = "local service call failed"
    return error_type, message or "local service call failed"


def validate_service_tape_errors(
    events: tuple[ServiceEvent, ...], *, source: str
) -> None:
    """Reject replay events whose exception type cannot be recreated safely."""

    for index, event in enumerate(events):
        if not isinstance(event, ServiceEvent):
            raise ReplayDivergence(f"{source} event {index} is malformed")
        if event.error is not None and event.error not in SAFE_REPLAY_ERROR_NAMES:
            raise ReplayDivergence(
                f"unsupported replayed service error type at {source} event {index}: "
                f"{event.error}"
            )


class ReplayableServices:
    """Actor-local model and randomness facade with exact response replay."""

    def __init__(
        self,
        *,
        seed: int,
        model_backend: StructuredModelBackend | None = None,
        agent_backend: AgentSessionBackend | None = None,
        public_resources: dict[str, Any] | None = None,
        replay_tape: tuple[ServiceEvent, ...] = (),
        committed_tape: tuple[ServiceEvent, ...] = (),
        limits: ServiceLimits | None = None,
        meter: ServiceMeter | None = None,
        audit_events: list[ServiceEvent] | None = None,
        audit_lock: threading.RLock | None = None,
        model_name: str | None = None,
        reasoning_effort: str | None = None,
        event_sink: Callable[[ServiceEvent], None] | None = None,
        judge_call: Callable[[Any], Any] | None = None,
    ) -> None:
        validate_service_tape_errors(replay_tape, source="replay tape")
        validate_service_tape_errors(committed_tape, source="committed tape")
        self._root_seed = int(seed)
        self._rng = random.Random(self._root_seed)
        self._branch_id = "root"
        self._model_backend = model_backend or DisabledModelBackend()
        self._agent_backend = agent_backend
        self.public_resources = copy.deepcopy(public_resources or {})
        self._replay_tape = tuple(replay_tape)
        self._replay_cursor = 0
        self._replay_consumed: set[int] = set()
        # A resumed non-actor service (the Arena Judge) has already-consumed
        # history plus, possibly, a successful post-checkpoint call that must
        # still be replayed. Keep those concepts separate: committed history
        # is exported but never offered as the next replay response.
        self._events: list[ServiceEvent] = list(copy.deepcopy(committed_tape))
        self._limits = limits or ServiceLimits()
        self._meter = meter or ServiceMeter()
        self._audit_events = audit_events if audit_events is not None else []
        self._audit_lock = audit_lock or threading.RLock()
        self._model_name = model_name
        self._reasoning_effort = reasoning_effort
        self._judge_call = judge_call
        self._event_sink = event_sink
        self._lock = threading.RLock()
        self._model_context_lock = threading.RLock()
        self._model_context: dict[str, Any] | None = None
        self._local_only_depth = 0

    @contextmanager
    def local_only(self):
        """Temporarily reject every participant service before it is attempted."""

        with self._lock:
            self._local_only_depth += 1
        try:
            yield
        finally:
            with self._lock:
                self._local_only_depth -= 1

    def _call(
        self,
        kind: str,
        request: Any,
        live: Callable[[], Any],
        *,
        metadata_backend: Any | None = None,
    ) -> Any:
        with self._lock:
            if self._local_only_depth:
                raise ResourceLimitExceeded(
                    "participant services are disabled during a stage transition"
                )
        digest = _request_hash(kind, request)
        with self._lock:
            replaying = self._replay_cursor < len(self._replay_tape)
            if replaying:
                index = self._replay_cursor
                event = self._replay_tape[index]
                if event.kind != kind or event.request_hash != digest:
                    index = self._find_concurrent_model_event(kind, digest)
                    event = self._replay_tape[index] if index is not None else event
                if index is None or event.kind != kind or event.request_hash != digest:
                    _debug = os.environ.get("ARENA_REPLAY_DIVERGENCE_DUMP")
                    if _debug:
                        with open(_debug, "a", encoding="utf-8") as handle:
                            json.dump(
                                {
                                    "cursor": self._replay_cursor,
                                    "incoming_kind": kind,
                                    "incoming_hash": digest,
                                    "incoming_request": request,
                                    "recorded_kind": event.kind,
                                    "recorded_hash": event.request_hash,
                                    "recorded_request": getattr(event, "request", None),
                                },
                                handle,
                                ensure_ascii=False,
                                default=str,
                            )
                            handle.write("\n")
                    raise ReplayDivergence(
                        f"service request diverged at event {self._replay_cursor}"
                    )
                self._replay_consumed.add(index)
                # Replayed events are flushed to the exported tape in RECORDED
                # order, not consumption order: a sequential re-execution of a
                # concurrently recorded batch may legitimately consume events
                # out of order, but the exported tape must stay byte-identical
                # to the source journal.
                while self._replay_cursor in self._replay_consumed:
                    self._events.append(
                        copy.deepcopy(self._replay_tape[self._replay_cursor])
                    )
                    self._replay_cursor += 1
                accepted = copy.deepcopy(event)
                # Failed service calls are part of the branch-local execution
                # history too.  Enter their recorded RNG epoch before raising
                # so a caught failure followed by a live random call continues
                # on the same branch as the source run.
                self._enter_replayed_branch(accepted)
                if accepted.error is None:
                    self._advance_replayed_random(kind, request, accepted.response)
        if not replaying:
            started_at = time.time()
            try:
                response = live()
            except Exception as exc:  # noqa: BLE001
                metadata_fn = getattr(metadata_backend, "last_call_metadata", None)
                metadata = metadata_fn() if callable(metadata_fn) else {}
                error_type, error_message = _bounded_service_error(
                    exc,
                    sensitive=kind in {"model.structured", "agent.session_turn"},
                )
                accepted = ServiceEvent(
                    kind,
                    digest,
                    None,
                    error_type,
                    copy.deepcopy(request),
                    error_message,
                    started_at,
                    time.time(),
                    copy.deepcopy(metadata),
                )
                self._record_event(accepted, audit=True)
                raise
            metadata_fn = getattr(metadata_backend, "last_call_metadata", None)
            metadata = metadata_fn() if callable(metadata_fn) else {}
            accepted = ServiceEvent(
                kind,
                digest,
                copy.deepcopy(response),
                request=copy.deepcopy(request),
                started_at=started_at,
                finished_at=time.time(),
                metadata=copy.deepcopy(metadata),
            )
        if not replaying:
            self._record_event(accepted, audit=True)
        # Native Codex history is actor-local state. Derive it from the
        # committed tape order (which can differ from concurrent replay order),
        # never from a backend-global "last session" or an abandoned branch.
        if kind == "model.structured":
            with self._lock:
                for recorded in reversed(self._events):
                    context = recorded.metadata.get("native_context") or recorded.metadata.get("codex_context")
                    if recorded.error is None and isinstance(context, dict):
                        self._model_context = copy.deepcopy(context)
                        break
        if accepted.error is not None:
            self._raise_replayed_error(accepted.error, accepted.error_message)
        return copy.deepcopy(accepted.response)

    def _find_concurrent_model_event(self, kind: str, digest: str) -> int | None:
        """Match reordered calls only inside one contiguous model-call batch."""
        if kind != "model.structured" or self._replay_tape[self._replay_cursor].kind != kind:
            return None
        for index in range(self._replay_cursor + 1, len(self._replay_tape)):
            event = self._replay_tape[index]
            if event.kind != kind:
                break
            if index not in self._replay_consumed and event.request_hash == digest:
                return index
        return None

    def _advance_replayed_random(self, kind: str, request: Any, response: Any) -> None:
        """Keep RNG state aligned when a post-checkpoint random result is replayed."""
        if kind == "random.random":
            actual = self._rng.random()
        elif kind == "random.randint":
            actual = self._rng.randint(int(request["start"]), int(request["stop"]))
        elif kind == "random.choice":
            actual = self._rng.choice(request["values"])
        else:
            return
        if actual != response:
            raise ReplayDivergence("recorded random response does not match the restored RNG state")

    def _enter_replayed_branch(self, event: ServiceEvent) -> None:
        branch_id = (event.metadata or {}).get("service_branch_id")
        if not isinstance(branch_id, str) or not branch_id or branch_id == self._branch_id:
            return
        branch_seed = hashlib.sha256(
            f"{self._root_seed}:{branch_id}".encode("utf-8")
        ).digest()
        self._rng = random.Random(int.from_bytes(branch_seed[:8], "big"))
        self._branch_id = branch_id

    def _record_event(self, event: ServiceEvent, *, audit: bool) -> None:
        if audit:
            with self._meter.lock:
                metadata = {
                    **(event.metadata or {}),
                    "service_meter": {
                        "model_calls": self._meter.model_calls,
                        "random_calls": self._meter.random_calls,
                    },
                    "service_branch_id": self._branch_id,
                    "rng_state_after": copy.deepcopy(self._rng.getstate()),
                }
            event = replace(event, metadata=metadata)
        if audit:
            # The shared audit tape and its durable sink define one replay
            # order.  Keep them under the same shared lock so concurrent Judge
            # completions cannot publish A,B in memory but B,A on disk.
            with self._audit_lock:
                with self._lock:
                    self._events.append(copy.deepcopy(event))
                self._audit_events.append(copy.deepcopy(event))
                if self._event_sink is not None:
                    self._event_sink(copy.deepcopy(event))
        else:
            with self._lock:
                self._events.append(copy.deepcopy(event))

    @staticmethod
    def _raise_replayed_error(error: str, message: str | None = None) -> None:
        if error == "BudgetBlocked":
            from .run_budget import BudgetBlocked
            raise BudgetBlocked(message or "replayed cumulative budget exhaustion")
        if error == "ResourceLimitExceeded":
            raise ResourceLimitExceeded(message or "replayed service budget exhaustion")
        if error == "ReplayDivergence":
            raise ReplayDivergence(message or "replayed service divergence")
        error_type = _REPLAY_ERROR_TYPES.get(error)
        if error_type is None:
            raise ReplayDivergence(f"unsupported replayed service error type: {error}")
        raise error_type(message or f"replayed service failure ({error})")

    def structured_model(self, **request: Any) -> Any:
        if getattr(self._model_backend, "supports_context_chain", False):
            # One linear history across Generator channels. This intentionally
            # serializes their model calls while preserving eager slate authoring.
            with self._model_context_lock:
                return self._structured_model(**request)
        return self._structured_model(**request)

    def _structured_model(self, **request: Any) -> Any:
        request = dict(request)
        if self._model_name is not None:
            # The submitted role cannot select a provider or credential.  Its
            # model field is an ignored hint; the evaluation profile wins.
            request["model"] = self._model_name
        requested_tokens = int(request.get("max_output_tokens", DEFAULT_MAX_OUTPUT_TOKENS))
        if requested_tokens < 1:
            raise ResourceLimitExceeded("invalid model output-token request")
        request["max_output_tokens"] = min(requested_tokens, self._limits.max_output_tokens_per_call)
        memory_policy = self.public_resources.get("generator_memory_policy") or {}
        timeout_cap = float(memory_policy.get("timeout", 300.0))
        request["timeout"] = min(max(float(request.get("timeout", timeout_cap)), 1.0), timeout_cap)
        configured_effort = self._reasoning_effort or getattr(self._model_backend, "reasoning_effort", None)
        if configured_effort is not None:
            request["reasoning_effort"] = configured_effort
        # The evaluation profile selects the model above; allow the requested
        # maximum effort only on these explicitly supported OpenAI models.
        max_effort_supported = (
            request.get("reasoning_effort") == "max"
            and (str(request.get("model", "")).startswith(("gpt-6-astra", "gpt-5.6-sol"))
                 or (configured_effort == "max" and str(request.get("model", "")).startswith(("claude-", "anthropic/claude-")))
                 or (configured_effort == "max" and memory_policy and request.get("model") == "together/zai-org/GLM-5.3"))
        )
        native_xhigh = (request.get("reasoning_effort") == "xhigh"
                        and (self._reasoning_effort == "xhigh"
                             or getattr(self._model_backend, "supports_context_chain", False)))
        if request.get("reasoning_effort") not in {None, "low", "medium", "high"} and not max_effort_supported and not native_xhigh:
            raise ResourceLimitExceeded("model reasoning effort is not allowed by the profile")
        request_size = len(json.dumps(request, ensure_ascii=False, allow_nan=False).encode("utf-8"))
        if request_size > self._limits.max_request_bytes:
            raise ResourceLimitExceeded("model request exceeds the size budget")

        def live_model_call() -> Any:
            with self._meter.lock:
                if self._meter.model_calls >= self._limits.max_model_calls:
                    raise ResourceLimitExceeded("model-call budget exhausted")
                self._meter.model_calls += 1
            contextual = getattr(self._model_backend, "structured_in_context", None)
            response = (
                contextual(copy.deepcopy(self._model_context), **request)
                if callable(contextual)
                else self._model_backend.structured(**request)
            )
            usage_fn = getattr(self._model_backend, "usage_totals", None)
            if callable(usage_fn):
                usage = usage_fn()
                total_tokens = int(usage.get("input_tokens", 0)) + int(
                    usage.get("output_tokens", 0)
                )
                if total_tokens > self._limits.max_total_tokens:
                    raise ResourceLimitExceeded("model-token budget exhausted")
                if float(usage.get("cost_usd", 0.0)) > self._limits.max_model_cost_usd:
                    raise ResourceLimitExceeded("model monetary budget exhausted")
            return response

        response = self._call(
            "model.structured",
            request,
            live_model_call,
            metadata_backend=self._model_backend,
        )
        return response

    def judge_evaluate(self, ideas: Any) -> Any:
        """Run the Arena Judge over a candidate idea set.

        The Judge is Arena-owned and target-aware, so its model spend lands on
        the Judge's own budget rather than the caller's. The call is taped here
        only so that a replay resolves it without asking the Judge again.
        """

        if self._judge_call is None:
            raise ResourceLimitExceeded("this role cannot call the Judge")
        request = {"ideas": ideas}

        def live_judge_call() -> Any:
            return self._judge_call(ideas)

        return self._call("judge.evaluate", request, live_judge_call)

    def agent_turn(self, **request: Any) -> Any:
        """Run one turn of an evaluation-profile-selected persistent CLI agent.

        Unlike ``structured_model``, this service preserves a provider session ID
        across calls.  The participant supplies only prompts, schema, and the
        previously recorded session ID; executable and model selection remain in
        the evaluation profile.
        """
        request = dict(request)
        required = ("system_prompt", "user", "schema", "schema_name")
        if any(key not in request for key in required):
            raise ValueError("agent turn is missing a required field")
        if request.get("session_id") is not None and not isinstance(
            request.get("session_id"), str
        ):
            raise TypeError("agent session ID must be a string")
        request_size = len(
            json.dumps(request, ensure_ascii=False, allow_nan=False).encode("utf-8")
        )
        if request_size > self._limits.max_request_bytes:
            raise ResourceLimitExceeded("agent request exceeds the size budget")

        def live_agent_call() -> Any:
            if self._agent_backend is None:
                raise RuntimeError("this evaluation profile has no agent-session backend")
            with self._meter.lock:
                if self._meter.model_calls >= self._limits.max_model_calls:
                    raise ResourceLimitExceeded("model-call budget exhausted")
                self._meter.model_calls += 1
            response = self._agent_backend.turn(**request)
            # Agentic Generator responses account for the main model plus its
            # child turns. Reserve the logical service call above, then charge
            # the remaining model turns reported by the audited backend.
            response_usage = response.get("usage") if isinstance(response, dict) else None
            model_turns = (
                int(response_usage.get("turns", 1))
                if isinstance(response_usage, dict)
                else 1
            )
            additional_turns = max(0, model_turns - 1)
            if additional_turns:
                with self._meter.lock:
                    if self._meter.model_calls + additional_turns > self._limits.max_model_calls:
                        raise ResourceLimitExceeded("model-call budget exhausted")
                    self._meter.model_calls += additional_turns
            usage_fn = getattr(self._agent_backend, "usage_totals", None)
            if callable(usage_fn):
                usage = usage_fn()
                total_tokens = int(usage.get("input_tokens", 0)) + int(
                    usage.get("output_tokens", 0)
                )
                if total_tokens > self._limits.max_total_tokens:
                    raise ResourceLimitExceeded("model-token budget exhausted")
                if float(usage.get("cost_usd", 0.0)) > self._limits.max_model_cost_usd:
                    raise ResourceLimitExceeded("model monetary budget exhausted")
            return response

        return self._call(
            "agent.session_turn",
            request,
            live_agent_call,
            metadata_backend=self._agent_backend,
        )

    def random(self) -> float:
        return float(self._call("random.random", {}, lambda: self._random_call(self._rng.random)))

    def randint(self, start: int, stop: int) -> int:
        request = {"start": start, "stop": stop}
        return int(
            self._call(
                "random.randint",
                request,
                lambda: self._random_call(lambda: self._rng.randint(start, stop)),
            )
        )

    def choice(self, values: list[Any] | tuple[Any, ...]) -> Any:
        request = {"values": list(values)}
        return self._call(
            "random.choice", request, lambda: self._random_call(lambda: self._rng.choice(values))
        )

    def export_tape(self) -> tuple[ServiceEvent, ...]:
        with self._lock:
            return tuple(copy.deepcopy(self._events))

    def event_count(self) -> int:
        """Return the committed actor-local tape length without copying it."""
        with self._lock:
            return len(self._events)

    def usage(self) -> dict[str, Any]:
        with self._meter.lock, self._lock:
            value: dict[str, Any] = {
                "model_calls": self._meter.model_calls,
                "random_calls": self._meter.random_calls,
                "service_events": len(self._events),
            }
        combined_usage = self._combined_model_usage()
        if combined_usage:
            value["model_usage"] = combined_usage
        if callable(getattr(self._agent_backend, "usage_totals", None)):
            value["agent_usage"] = self._agent_backend.usage_totals()
        return value

    def _random_call(self, function: Callable[[], Any]) -> Any:
        with self._meter.lock:
            if self._meter.random_calls >= self._limits.max_random_calls:
                raise ResourceLimitExceeded("random-service budget exhausted")
            self._meter.random_calls += 1
        return function()

    def finish_replay(self, branch_id: str) -> None:
        with self._lock:
            if self._replay_cursor != len(self._replay_tape):
                raise ReplayDivergence("actor replay consumed a different number of service events")
            branch_seed = hashlib.sha256(
                f"{self._root_seed}:{branch_id}".encode("utf-8")
            ).digest()
            self._rng = random.Random(int.from_bytes(branch_seed[:8], "big"))
            self._branch_id = branch_id

    def replay_remaining(self) -> int:
        with self._lock:
            return len(self._replay_tape) - len(self._replay_consumed)

    def export_state(self) -> dict[str, Any]:
        """Private durable state needed to continue after actor reconstruction."""
        with self._meter.lock, self._lock:
            export_agent_state = getattr(self._agent_backend, "export_state", None)
            return {
                "rng_state": copy.deepcopy(self._rng.getstate()),
                "meter": {
                    "model_calls": self._meter.model_calls,
                    "random_calls": self._meter.random_calls,
                },
                "model_usage": self._combined_model_usage(),
                **({"model_context": copy.deepcopy(self._model_context)} if self._model_context is not None else {}),
                "agent_usage": (
                    self._agent_backend.usage_totals()
                    if callable(getattr(self._agent_backend, "usage_totals", None))
                    else {}
                ),
                "agent_backend_state": (
                    export_agent_state() if callable(export_agent_state) else {}
                ),
            }

    def restore_state(self, state: dict[str, Any]) -> None:
        def tuples(value: Any) -> Any:
            if isinstance(value, list):
                return tuple(tuples(item) for item in value)
            return value

        with self._lock:
            if "model_context" in state and state["model_context"] != self._model_context:
                raise ReplayDivergence("native model context differs from replayed checkpoint")
            if "rng_state" in state:
                self._rng.setstate(tuples(state["rng_state"]))
            restore_agent_state = getattr(self._agent_backend, "restore_state", None)
            agent_state = state.get("agent_backend_state")
            if (
                callable(restore_agent_state)
                and isinstance(agent_state, dict)
                and agent_state
            ):
                restore_agent_state(agent_state)
            else:
                restore_agent_usage = getattr(self._agent_backend, "restore_usage", None)
                agent_usage = state.get("agent_usage")
                if callable(restore_agent_usage) and isinstance(agent_usage, dict):
                    restore_agent_usage(agent_usage)

    def _combined_model_usage(self) -> dict[str, int | float]:
        combined: dict[str, int | float] = {}
        for backend in (self._model_backend, self._agent_backend):
            usage_fn = getattr(backend, "usage_totals", None)
            if not callable(usage_fn):
                continue
            for key, amount in usage_fn().items():
                if isinstance(amount, bool) or not isinstance(amount, (int, float)):
                    continue
                combined[key] = combined.get(key, 0) + amount
        return combined


class ServiceFactory:
    def __init__(
        self,
        *,
        seed: int,
        model_backend: StructuredModelBackend | None = None,
        agent_backend: AgentSessionBackend | None = None,
        public_resources: dict[str, Any] | None = None,
        limits: ServiceLimits | None = None,
        model_name: str | None = None,
        reasoning_effort: str | None = None,
        event_sink: Callable[[ServiceEvent], None] | None = None,
        initial_model_calls: int = 0,
        initial_random_calls: int = 0,
        judge_call: Callable[[Any], Any] | None = None,
    ) -> None:
        self.seed = seed
        self.model_backend = model_backend
        self.agent_backend = agent_backend
        self.public_resources = public_resources or {}
        self.limits = limits or ServiceLimits()
        self.meter = ServiceMeter(initial_model_calls, initial_random_calls)
        self.audit_events: list[ServiceEvent] = []
        self.audit_lock = threading.RLock()
        self.model_name = model_name
        self.reasoning_effort = reasoning_effort
        self.event_sink = event_sink
        self.judge_call = judge_call

    def create(
        self,
        replay_tape: tuple[ServiceEvent, ...] = (),
        *,
        committed_tape: tuple[ServiceEvent, ...] = (),
    ) -> ReplayableServices:
        return ReplayableServices(
            seed=self.seed,
            model_backend=self.model_backend,
            agent_backend=self.agent_backend,
            public_resources=self.public_resources,
            replay_tape=replay_tape,
            committed_tape=committed_tape,
            limits=self.limits,
            meter=self.meter,
            audit_events=self.audit_events,
            audit_lock=self.audit_lock,
            model_name=self.model_name,
            reasoning_effort=self.reasoning_effort,
            event_sink=self.event_sink,
            judge_call=self.judge_call,
        )

    def export_audit_tape(self) -> tuple[ServiceEvent, ...]:
        with self.audit_lock:
            return tuple(copy.deepcopy(self.audit_events))
