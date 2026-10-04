"""Hash-chained run storage and deterministic protocol/actor replay."""

from __future__ import annotations

from .._compat import legacy_fields, legacy_keywords

import copy
import ast
import hashlib
import json
import math
import os
import random
import re
import threading
import time
from collections import Counter
from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable

from ..contract import IDEA_RECOVERY_V1
from ..contract.messages import (
    Checkout,
    Choice,
    Idea,
    IdeaVerdict,
    Option,
    PresentedQuestion,
    Question,
    Submission,
    SubmissionFeedback,
    StageReady,
    StageTransition,
    SubmitOption,
)
from ..contract.validation import message_hash
from ..contract.recovery import (
    CURRENT_ACCOUNTING, LEGACY_ACCOUNTING, choice_surcharge, validate_accounting_version,
)
from ..errors import ProtocolError, ReplayDivergence, ValidationError
from ..evaluation.base import aggregate_repeated_verdicts
from ..runtime.actor import ActorCall, ActorCheckpoint, ActorRuntime
from ..runtime.services import (
    ServiceEvent,
    ServiceFactory,
    validate_service_tape_errors,
)
from ..runtime.wire import decode_message, encode_message
from ..runtime.subprocess_actor import SubprocessActorFactory
from ..submission_io.manifest import build_dependency_paths, load_manifest, validate_entrypoint_sources
from ..targets.loader import load_target_pack
from .artifacts import ArtifactStore, hash_tree
from .accounting import reprice_events


_TOGETHER_MULTI_STAGE_MODELS = frozenset({
    "together/zai-org/GLM-5.2",
    "together/zai-org/GLM-5.3",
    "together/moonshotai/Kimi-K3",
})


def _jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    return value


def _canonical_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            _jsonable(value), ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ValidationError("run event is not canonical JSON") from exc


class HashChainWriter:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.previous_hash = "0" * 64
        self.sequence = 0
        self._lock = threading.RLock()

    def append(self, payload: dict[str, Any]) -> str:
        # Judge work may complete concurrently.  Sequence allocation, hashing,
        # durable append, and the in-memory cursor form one atomic operation.
        with self._lock:
            body = {
                "sequence": self.sequence,
                "previous_hash": self.previous_hash,
                "recorded_at": time.time(),
                **_jsonable(payload),
            }
            event_hash = hashlib.sha256(
                bytes.fromhex(self.previous_hash) + _canonical_bytes(body)
            ).hexdigest()
            record = {**body, "event_hash": event_hash}
            with self.path.open("a", encoding="utf-8") as stream:
                stream.write(
                    json.dumps(
                        record,
                        ensure_ascii=False,
                        sort_keys=True,
                        separators=(",", ":"),
                    )
                    + "\n"
                )
                stream.flush()
                os.fsync(stream.fileno())
            self.previous_hash = event_hash
            self.sequence += 1
            return event_hash


def verify_hash_chain(
    path: Path,
    *,
    max_records: int | None = None,
    tolerate_truncated_tail: bool = False,
) -> tuple[dict[str, Any], ...]:
    previous = "0" * 64
    records: list[dict[str, Any]] = []
    try:
        raw_text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ValidationError(f"could not read replay file {path.name}") from exc
    lines = raw_text.splitlines()
    has_partial_tail = bool(raw_text) and not raw_text.endswith("\n")
    if max_records is not None:
        if max_records < 0 or len(lines) < max_records:
            raise ReplayDivergence(f"{path.name}: hash chain is shorter than its durable cursor")
        lines = lines[:max_records]
    for sequence, line in enumerate(lines):
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            if (
                tolerate_truncated_tail
                and sequence == len(lines) - 1
                and has_partial_tail
            ):
                break
            raise ReplayDivergence(f"{path.name}: invalid JSON at event {sequence}") from exc
        if not isinstance(record, dict):
            raise ReplayDivergence(f"{path.name}: event {sequence} is not an object")
        if (
            type(record.get("sequence")) is not int
            or record.get("sequence") != sequence
            or record.get("previous_hash") != previous
        ):
            raise ReplayDivergence(f"{path.name}: broken chain at event {sequence}")
        supplied = record.pop("event_hash", None)
        expected = hashlib.sha256(bytes.fromhex(previous) + _canonical_bytes(record)).hexdigest()
        if supplied != expected:
            raise ReplayDivergence(f"{path.name}: event hash mismatch at event {sequence}")
        record["event_hash"] = supplied
        previous = expected
        records.append(record)
    return tuple(records)


def validate_provider_attempt_journal(
    path: str | Path,
    *,
    max_records: int | None = None,
    require_complete: bool = True,
) -> dict[str, Any]:
    """Validate and summarize the durable low-level provider-attempt chain."""

    records = verify_hash_chain(Path(path), max_records=max_records)
    allowed_roles = {"generator", "oracle", "judge"}
    started_keys = {
        "kind", "schema_version", "provider", "model", "logical_request_sha",
        "attempt", "request_sha256", "started_at",
    }
    finished_keys = started_keys | {
        "finished_at", "status", "provider_request_id", "input_tokens",
        "output_tokens", "cost_usd", "usage_available", "error_type",
        "error_message", "status_code", "response_sha256",
    }
    # v1 predates direct-path prompt caching: it has no cache-class fields, its
    # input_tokens counts only uncached input (there was never any other kind), and its
    # cost is the base rate on that total.  v2 records the cache-inclusive total plus
    # the class breakdown and prices each class at its own rate.  Both must keep
    # validating so already-recorded runs stay replayable.
    _CACHE_CLASS_KEYS = (
        "cache_read_input_tokens",
        "cache_write_5m_input_tokens",
        "cache_write_1h_input_tokens",
    )
    finished_keys_by_version = {
        "arena-provider-attempt-v1": finished_keys,
        "arena-provider-attempt-v2": finished_keys | set(_CACHE_CLASS_KEYS),
    }
    # Concurrent logical calls legitimately interleave their started/finished
    # records, so pairing and repair-sequence checks are keyed by the logical
    # request rather than assuming one globally serial in-flight attempt.
    pending: dict[tuple[str, str, int], dict[str, Any]] = {}
    finished: list[tuple[str, dict[str, Any]]] = []
    provider_ids: list[str] = []
    previous_finished_by_call: dict[tuple[str, str], dict[str, Any]] = {}
    successful_attempts_by_call: dict[tuple[str, str], int] = {}
    for record in records:
        payload = _event_payload(record)
        if set(payload) != {"kind", "role", "attempt"}:
            raise ReplayDivergence("provider-attempt journal row has invalid keys")
        if payload.get("kind") != "provider_attempt" or payload.get("role") not in allowed_roles:
            raise ReplayDivergence("provider-attempt journal row has invalid role or kind")
        role = str(payload["role"])
        attempt = payload.get("attempt")
        if not isinstance(attempt, dict):
            raise ReplayDivergence("provider-attempt journal payload is malformed")
        kind = attempt.get("kind")
        schema_version = attempt.get("schema_version")
        if schema_version not in finished_keys_by_version:
            raise ReplayDivergence("provider-attempt record has an unknown schema version")
        expected_keys = (
            started_keys
            if kind == "provider_attempt_started"
            else finished_keys_by_version[schema_version]
        )
        openai_native_cache = (
            attempt.get("provider") == "openai"
            and str(attempt.get("model", "")).startswith(("gpt-6-astra", "gpt-5.6-sol"))
            and schema_version == "arena-provider-attempt-v2"
            and "cache_write_input_tokens" in attempt
        )
        if openai_native_cache and kind == "provider_attempt_finished":
            expected_keys = expected_keys | {"cache_write_input_tokens"}
        anthropic_usage_evidence = (
            attempt.get("provider") == "anthropic"
            and schema_version == "arena-provider-attempt-v2"
            and kind == "provider_attempt_finished"
            and "usage_fields_valid" in attempt
        )
        if anthropic_usage_evidence:
            expected_keys = expected_keys | {"usage_fields_valid"}
            if type(attempt["usage_fields_valid"]) is not bool:
                raise ReplayDivergence("provider usage field evidence is invalid")
        if set(attempt) != expected_keys:
            raise ReplayDivergence("provider-attempt record has invalid keys")
        if (
            attempt.get("provider")
            not in {"anthropic", "openai", "openrouter", "together"}
            or not isinstance(attempt.get("model"), str)
            or not attempt["model"]
            or not isinstance(attempt.get("logical_request_sha"), str)
            or len(attempt["logical_request_sha"]) != 24
            or type(attempt.get("attempt")) is not int
            or int(attempt["attempt"]) < 1
            or not isinstance(attempt.get("request_sha256"), str)
            or len(attempt["request_sha256"]) != 64
            or not isinstance(attempt.get("started_at"), (int, float))
        ):
            raise ReplayDivergence("provider-attempt record has invalid common fields")
        try:
            bytes.fromhex(attempt["request_sha256"])
            bytes.fromhex(attempt["logical_request_sha"])
        except ValueError as exc:
            raise ReplayDivergence("provider-attempt request hash is invalid") from exc
        attempt_key = (
            role,
            str(attempt["logical_request_sha"]),
            int(attempt["attempt"]),
        )
        if kind == "provider_attempt_started":
            if attempt_key in pending:
                raise ReplayDivergence("provider-attempt starts overlap")
            pending[attempt_key] = attempt
            continue
        start = pending.pop(attempt_key, None)
        if start is None:
            raise ReplayDivergence("provider-attempt finish has no start")
        for field in (
            "provider", "model", "logical_request_sha", "attempt", "request_sha256",
            "started_at",
        ):
            if attempt.get(field) != start.get(field):
                raise ReplayDivergence("provider-attempt start/finish diverge")
        attempt_number = int(attempt["attempt"])
        call_key = (role, str(attempt["logical_request_sha"]))
        if attempt_number == 1:
            # The logical SHA identifies request contents, not a unique call.
            # A participant may issue the exact same structured request again,
            # and each physical sequence correctly restarts at attempt one.
            # Do not let successful stages from an earlier identical call make
            # this call's reasoning -> formatter transition look like a third
            # success in one repair sequence.
            previous_finished_by_call.pop(call_key, None)
            successful_attempts_by_call[call_key] = 0
        else:
            previous = previous_finished_by_call.get(call_key)
            if previous is None:
                raise ReplayDivergence("provider repair attempt lacks predecessor")
            multi_stage_transition = (
                attempt.get("provider") == "together"
                and attempt.get("model") in _TOGETHER_MULTI_STAGE_MODELS
                and previous.get("status") == "ok"
                and successful_attempts_by_call.get(call_key, 0) == 1
            )
            if (
                previous.get("provider") != attempt.get("provider")
                or previous.get("model") != attempt.get("model")
                or int(previous.get("attempt", 0)) + 1 != attempt_number
                # A repair attempt may follow any attempt the client is allowed
                # to retry. That includes a delivered response whose usage the
                # provider did not report: the client retries it because the
                # condition is usually transient, and the unpriceable tokens it
                # burned are what keeps require_complete rejecting the journal.
                or (
                    previous.get("status")
                    not in {"parse_invalid", "error", "usage_unavailable"}
                    and not multi_stage_transition
                )
            ):
                raise ReplayDivergence("provider repair attempt sequence diverges")
        status = attempt.get("status")
        if status not in {"ok", "parse_invalid", "usage_unavailable", "error"}:
            raise ReplayDivergence("provider-attempt status is invalid")
        error_type = attempt.get("error_type")
        error_message = attempt.get("error_message")
        if error_type is not None and (
            not isinstance(error_type, str)
            or not re.fullmatch(r"[A-Za-z][A-Za-z0-9_.-]{0,63}", error_type)
        ):
            raise ReplayDivergence("provider-attempt error type is not bounded")
        if error_message is not None and (
            not isinstance(error_message, str)
            or not error_message
            or len(error_message.encode("utf-8")) > 160
        ):
            raise ReplayDivergence("provider-attempt error message is not bounded")
        if not isinstance(attempt.get("finished_at"), (int, float)) or float(
            attempt["finished_at"]
        ) < float(attempt["started_at"]):
            raise ReplayDivergence("provider-attempt timestamps are invalid")
        status_code = attempt.get("status_code")
        if status_code is not None and (
            type(status_code) is not int or not 100 <= status_code <= 599
        ):
            raise ReplayDivergence("provider-attempt status code is invalid")
        provider_request_id = attempt.get("provider_request_id")
        if provider_request_id is not None and (
            not isinstance(provider_request_id, str)
            or not provider_request_id
            or len(provider_request_id.encode("utf-8")) > 256
        ):
            raise ReplayDivergence("provider request ID is invalid")
        if status in {"ok", "parse_invalid"}:
            request_id = provider_request_id
            if (
                attempt.get("usage_available") is not True
                or not isinstance(request_id, str)
                or not request_id
                or not isinstance(attempt.get("response_sha256"), str)
                or len(attempt["response_sha256"]) != 64
                or type(attempt.get("input_tokens")) is not int
                or type(attempt.get("output_tokens")) is not int
                or int(attempt["input_tokens"]) <= 0
                or int(attempt["output_tokens"]) < 0
                or (int(attempt["output_tokens"]) == 0 and not (
                    anthropic_usage_evidence and attempt["usage_fields_valid"] is True
                ))
                or (anthropic_usage_evidence and attempt["usage_fields_valid"] is not True)
                or not isinstance(attempt.get("cost_usd"), (int, float))
                or not math.isfinite(float(attempt["cost_usd"]))
                or float(attempt["cost_usd"]) <= 0.0
                or attempt.get("status_code") is not None
            ):
                raise ReplayDivergence("completed provider attempt lacks exact usage")
            try:
                bytes.fromhex(attempt["response_sha256"])
            except ValueError as exc:
                raise ReplayDivergence("provider response hash is invalid") from exc
            if status == "ok" and (
                attempt.get("error_type") is not None
                or attempt.get("error_message") is not None
            ):
                raise ReplayDivergence("successful provider attempt contains an error")
            if status == "parse_invalid" and (
                not isinstance(attempt.get("error_type"), str)
                or not attempt["error_type"]
                or not isinstance(attempt.get("error_message"), str)
                or not attempt["error_message"]
            ):
                raise ReplayDivergence("parse-invalid provider attempt lacks an error")
            from ..runtime.provider_client import _openai_attempt_cost, _rates, anthropic_turn_cost

            cache_read, write_5m, write_1h = (
                int(attempt.get(key) or 0) for key in _CACHE_CLASS_KEYS
            )
            if schema_version != "arena-provider-attempt-v1":
                if any(
                    type(attempt.get(key)) is not int or int(attempt[key]) < 0
                    for key in _CACHE_CLASS_KEYS
                ):
                    raise ReplayDivergence("provider-attempt cache classes are invalid")
                # input_tokens is the cache-inclusive total, so the classes can never
                # add up to more than it.
                if cache_read + write_5m + write_1h > int(attempt["input_tokens"]):
                    raise ReplayDivergence("provider-attempt cache classes exceed input")
                if openai_native_cache:
                    writes = attempt["cache_write_input_tokens"]
                    if type(writes) is not int or writes < 0 or cache_read + writes > int(attempt["input_tokens"]):
                        raise ReplayDivergence("OpenAI native cache classes are invalid")
                    if write_5m or write_1h:
                        raise ReplayDivergence("OpenAI attempt reports Anthropic cache TTLs")
                # Older OpenAI records retain their full-rate input accounting.
                if attempt["provider"] != "anthropic" and not openai_native_cache and (
                    cache_read or write_5m or write_1h
                ):
                    raise ReplayDivergence("non-Anthropic attempt reports cache classes")
            if attempt["provider"] == "anthropic" and (
                cache_read or write_5m or write_1h
            ):
                expected_cost = anthropic_turn_cost(
                    str(attempt["model"]),
                    uncached_input_tokens=int(attempt["input_tokens"])
                    - cache_read
                    - write_5m
                    - write_1h,
                    cache_read_input_tokens=cache_read,
                    cache_write_5m_input_tokens=write_5m,
                    cache_write_1h_input_tokens=write_1h,
                    output_tokens=int(attempt["output_tokens"]),
                )
            elif openai_native_cache:
                expected_cost = _openai_attempt_cost(str(attempt["model"]), SimpleNamespace(
                    input_tokens=int(attempt["input_tokens"]),
                    output_tokens=int(attempt["output_tokens"]),
                    input_tokens_details=SimpleNamespace(
                        cached_tokens=cache_read,
                        cache_write_tokens=attempt["cache_write_input_tokens"],
                    ),
                ))
            else:
                input_rate, output_rate = _rates(str(attempt["model"]))
                expected_cost = (
                    int(attempt["input_tokens"]) / 1e6 * input_rate
                    + int(attempt["output_tokens"]) / 1e6 * output_rate
                )
            if not math.isclose(
                float(attempt["cost_usd"]), expected_cost, rel_tol=1e-12, abs_tol=1e-12
            ):
                raise ReplayDivergence("provider-attempt cost disagrees with frozen rates")
            provider_ids.append(request_id)
        else:
            if (
                attempt.get("usage_available") is not False
                or not isinstance(attempt.get("error_type"), str)
                or not attempt["error_type"]
                or not isinstance(attempt.get("error_message"), str)
                or not attempt["error_message"]
                or type(attempt.get("input_tokens")) is not int
                or type(attempt.get("output_tokens")) is not int
                or int(attempt["input_tokens"]) < 0
                or int(attempt["output_tokens"]) < 0
                or not isinstance(attempt.get("cost_usd"), (int, float))
                or not math.isfinite(float(attempt["cost_usd"]))
                or float(attempt["cost_usd"]) < 0.0
                or (
                    status == "error"
                    and (
                        attempt.get("response_sha256") is not None
                        or provider_request_id is not None
                    )
                )
                or (
                    status == "usage_unavailable"
                    and (
                        not isinstance(attempt.get("response_sha256"), str)
                        or len(attempt["response_sha256"]) != 64
                    )
                )
            ):
                raise ReplayDivergence("unaccounted provider attempt is malformed")
            if status == "usage_unavailable":
                try:
                    bytes.fromhex(str(attempt["response_sha256"]))
                except ValueError as exc:
                    raise ReplayDivergence("provider response hash is invalid") from exc
            if require_complete and status != "error":
                # A request that failed outright (5xx, timeout) carries no
                # response, no provider request ID, and no tokens: it is
                # fully accounted at zero, so it is not an accounting gap.
                # Only a delivered response whose usage cannot be priced
                # ("usage_unavailable") leaves the ledger incomplete.
                raise ReplayDivergence("provider attempt has unknown usage")
        finished.append((role, attempt))
        previous_finished_by_call[call_key] = attempt
        if status == "ok":
            successful_attempts_by_call[call_key] = (
                successful_attempts_by_call.get(call_key, 0) + 1
            )
    if pending and require_complete:
        raise ReplayDivergence("provider-attempt journal ends with an unfinished request")
    if len(provider_ids) != len(set(provider_ids)):
        raise ReplayDivergence("provider request IDs are not globally unique")
    return {
        "records": len(records),
        "provider_calls": len(finished),
        "unfinished": len(pending),
        "usage_complete": not pending and all(
            attempt.get("usage_available") is True for _, attempt in finished
        ),
        "input_tokens": sum(int(attempt.get("input_tokens") or 0) for _, attempt in finished),
        "output_tokens": sum(int(attempt.get("output_tokens") or 0) for _, attempt in finished),
        "cost_usd": sum(float(attempt.get("cost_usd") or 0.0) for _, attempt in finished),
        "roles": {
            role: sum(1 for item_role, _ in finished if item_role == role)
            for role in sorted(allowed_roles)
        },
    }


def _branch_data(branches: Any | None) -> dict[str, Any]:
    if branches is None:
        return {
            "head": None,
            "order": [],
            "nodes": {},
            "continuation_counts": {},
            "option_counts": {},
            "checkout_pair_counts": {},
            "checkout_audit": [],
        }
    return {
        "head": branches.head,
        "accounting_version": branches.accounting_version,
        "order": branches.order,
        "nodes": {
            question_id: {
                "question_id": node.question_id,
                "parent_question_id": node.parent_question_id,
                "path_k": node.path_k,
                "created_index": node.created_index,
                "integrity_hash": node.integrity_hash,
                "question_hash": message_hash(node.question),
            }
            for question_id, node in branches.nodes.items()
        },
        "continuation_counts": branches.continuation_counts,
        "option_counts": {
            f"{key[0]}\0{key[1]}": value for key, value in branches.option_counts.items()
        },
        "checkout_pair_counts": {
            f"{key[0]}\0{key[1]}": value
            for key, value in branches.checkout_pair_counts.items()
        },
        "checkout_audit": [_jsonable(value) for value in branches.checkout_audit],
    }


def _public_event(
    event: dict[str, Any],
    disclosure: str,
    *,
    after_submission: bool = False,
) -> dict[str, Any] | None:
    kind = event.get("kind")
    if disclosure == "hidden":
        if kind == "run_started":
            return {
                "kind": kind,
                "run_id": event.get("run_id"),
                "time_travel": event.get("time_travel"),
            }
        if kind == "run_finished":
            return {"kind": kind, "status": event.get("status"), "score": event.get("score")}
        return None
    if disclosure != "development":
        raise ReplayDivergence("manifest has an unknown disclosure profile")
    # Once an attempt starts, an append-only public timeline cannot expose later
    # recovery control flow without also revealing that the attempt was rejected.
    # Seal that suffix and publish only the sanitized terminal aggregate.
    if kind in {"submission", "submission_judged", "judge_previewed"}:
        return None
    if after_submission and kind != "run_finished":
        return None
    if kind == "stage_transition":
        return {
            "kind": kind,
            "role": event.get("role"),
            "branch_id": event.get("branch_id"),
            "from_stage": event.get("from_stage"),
            "to_stage": event.get("to_stage"),
            "handoff_sha256": event.get("handoff_sha256"),
        }
    if kind == "run_finished":
        verdicts = []
        for verdict in event.get("verdicts", ()):
            if isinstance(verdict, dict):
                verdicts.append({"idea_id": verdict.get("idea_id"), "passed": verdict.get("passed")})
            else:
                verdicts.append({"idea_id": verdict.idea_id, "passed": verdict.passed})
        public = {
            "kind": kind,
            "status": event.get("status"),
            "score": event.get("score"),
            "k": event.get("k"),
            "passing_mass": event.get("passing_mass"),
            "submission_bits": event.get("submission_bits"),
            "matched_idea_ids": event.get("matched_idea_ids", ()),
            "verdicts": verdicts,
        }
        for field in ("judge_repeats", "judge_passes", "judge_pass_rate", "repeat_bits"):
            if field in event:
                public[field] = event[field]
        return public
    return event


def _redacted_public_events(
    events: tuple[dict[str, Any], ...] | list[dict[str, Any]],
    disclosure: str,
) -> tuple[dict[str, Any], ...]:
    """Apply the stateful append-only disclosure policy to a private trace."""

    after_submission = False
    result: list[dict[str, Any]] = []
    for event in events:
        if event.get("kind") == "submission":
            after_submission = True
        redacted = _public_event(
            event,
            disclosure,
            after_submission=after_submission,
        )
        if redacted is not None:
            result.append(_jsonable(redacted))
    return tuple(result)


def _legacy_public_events(
    events: tuple[dict[str, Any], ...] | list[dict[str, Any]],
    disclosure: str,
) -> tuple[dict[str, Any], ...]:
    """Disclosure used by protocol-event schema 2 run artifacts."""

    result: list[dict[str, Any]] = []
    for event in events:
        kind = event.get("kind")
        if disclosure == "hidden":
            if kind == "run_started":
                result.append({
                    "kind": kind,
                    "run_id": event.get("run_id"),
                    "time_travel": event.get("time_travel"),
                })
            elif kind == "run_finished":
                result.append({
                    "kind": kind,
                    "status": event.get("status"),
                    "score": event.get("score"),
                })
            continue
        if disclosure != "development":
            raise ReplayDivergence("manifest has an unknown disclosure profile")
        if kind == "submission_judged":
            continue
        if kind == "run_finished":
            verdicts = []
            for verdict in event.get("verdicts", ()):
                if isinstance(verdict, dict):
                    verdicts.append({
                        "idea_id": verdict.get("idea_id"),
                        "passed": verdict.get("passed"),
                    })
                else:
                    verdicts.append({
                        "idea_id": verdict.idea_id,
                        "passed": verdict.passed,
                    })
            result.append(_jsonable({
                "kind": kind,
                "status": event.get("status"),
                "score": event.get("score"),
                "k": event.get("k"),
                "passing_mass": event.get("passing_mass"),
                "submission_bits": event.get("submission_bits"),
                "matched_idea_ids": event.get("matched_idea_ids", ()),
                "verdicts": verdicts,
                "submission_attempts": event.get("submission_attempts"),
                "failed_attempt_bits": event.get("failed_attempt_bits"),
            }))
        else:
            result.append(_jsonable(event))
    return tuple(result)


def _event_payload(record: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in record.items()
        if key not in {"sequence", "previous_hash", "event_hash", "recorded_at"}
    }


def reconcile_provider_attempt_journal(
    provider_path: str | Path,
    service_path: str | Path,
    *,
    max_provider_records: int | None = None,
    max_service_records: int | None = None,
    require_complete: bool = True,
) -> dict[str, Any]:
    """Bind physical provider attempts to logical structured service rows."""

    summary = validate_provider_attempt_journal(
        provider_path,
        max_records=max_provider_records,
        require_complete=require_complete,
    )
    provider_records = verify_hash_chain(
        Path(provider_path), max_records=max_provider_records
    )
    durable_finished: list[dict[str, Any]] = []
    for record in provider_records:
        payload = _event_payload(record)
        attempt = payload.get("attempt") or {}
        if attempt.get("kind") == "provider_attempt_finished":
            durable_finished.append({
                "role": payload.get("role"),
                "attempt": {
                    key: copy.deepcopy(value)
                    for key, value in attempt.items()
                    if key != "kind"
                },
            })

    service_records = verify_hash_chain(
        Path(service_path), max_records=max_service_records
    )
    service_finished: list[dict[str, Any]] = []
    logical_calls = 0
    logical_successes = 0
    provider_successes = 0
    for record in service_records:
        payload = _event_payload(record)
        service = payload.get("service")
        if not isinstance(service, dict) or service.get("kind") != "model.structured":
            continue
        role = payload.get("role")
        metadata = service.get("metadata")
        request = service.get("request")
        if isinstance(metadata, dict) and metadata.get("backend") == "common-memory-v1":
            from ..runtime.memory_output_budget import validate_metadata
            validate_metadata(service, ArtifactStore(Path(service_path).parent.parent))
            continue
        if isinstance(metadata, dict) and metadata.get("backend") == "codex-source-v1":
            from ..runtime.codex_generator import validate_metadata
            validate_metadata(service)
            continue
        if isinstance(metadata, dict) and metadata.get("backend") == "claude-code-native-v1":
            from ..runtime.claude_generator import validate_metadata
            validate_metadata(service)
            continue
        if (
            role not in {"generator", "oracle", "judge"}
            or not isinstance(metadata, dict)
            or not isinstance(request, dict)
        ):
            raise ReplayDivergence("structured service row lacks provider metadata")
        from ..runtime.provider_client import _req_sha
        from ..runtime.services import _request_hash

        expected_service_hash = _request_hash("model.structured", request)
        if service.get("request_hash") != expected_service_hash:
            raise ReplayDivergence("structured service request hash diverges")
        try:
            # Direct backend calls may omit this parameter. New records carry
            # the resolved allowance; historical records used the 8k default.
            # Never reinterpret a historical omitted value using today's 50k.
            resolved_output_tokens = int(request.get(
                "max_output_tokens", metadata.get("max_output_tokens", 8000)
            ))
            if (
                "max_output_tokens" in metadata
                and metadata["max_output_tokens"] != resolved_output_tokens
            ):
                raise ReplayDivergence("structured service output allowance diverges")
            expected_logical_sha = _req_sha(
                str(request["model"]),
                str(request["developer"]),
                str(request["user"]),
                str(request["schema_name"]),
                request["schema"],
                resolved_output_tokens,
                request.get("reasoning_effort", "high"),
                request.get("conversation_state"),
                bool(request.get("return_conversation_state", False)),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ReplayDivergence("structured service request is malformed") from exc
        requested_model = str(request["model"])
        expected_provider = (
            "anthropic"
            if requested_model.startswith("anthropic/")
            else "together"
            if requested_model.startswith("together/")
            else "openrouter"
            if requested_model.startswith("openrouter/")
            else "openai"
        )
        if (
            metadata.get("model") != requested_model
            or metadata.get("schema_name") != request.get("schema_name")
            or metadata.get("request_sha") != expected_logical_sha
        ):
            raise ReplayDivergence("structured service route or logical hash diverges")
        attempts = metadata.get("provider_attempts")
        if (attempts in (None, []) and service.get("error") == "BudgetBlocked"
                and metadata.get("error_type") == "BudgetBlocked"
                and service.get("response") is None):
            usage = metadata.get("usage") or {}
            required_zero = ("calls", "provider_calls", "input_tokens", "output_tokens",
                             "cost_usd", "unknown_provider_attempts")
            if not all(key in usage and usage[key] == 0 for key in required_zero):
                raise ReplayDivergence("budget admission failure has nonzero or missing usage")
            # A resumed attempt can have the same logical hash as this earlier
            # zero-call admission failure. Reconcile every physical attempt
            # against the nonzero service rows below using the exact multiset;
            # an unmatched or hidden attempt still causes divergence there.
            continue
        if not isinstance(attempts, list) or not attempts:
            raise ReplayDivergence("structured service row lacks physical attempts")
        logical_calls += 1
        for index, attempt in enumerate(attempts, 1):
            if not isinstance(attempt, dict) or attempt.get("attempt") != index:
                raise ReplayDivergence("structured service attempt order is invalid")
            if (
                attempt.get("logical_request_sha") != expected_logical_sha
                or attempt.get("model") != requested_model
                or attempt.get("provider") != expected_provider
            ):
                raise ReplayDivergence("structured service logical request hash diverges")
            service_finished.append({"role": role, "attempt": copy.deepcopy(attempt)})
        success = service.get("error") is None and service.get("error_message") is None
        statuses = [str(item.get("status")) for item in attempts]
        input_tokens = sum(int(item.get("input_tokens") or 0) for item in attempts)
        output_tokens = sum(int(item.get("output_tokens") or 0) for item in attempts)
        cost_usd = sum(float(item.get("cost_usd") or 0.0) for item in attempts)
        unknown_attempts = sum(
            item.get("usage_available") is not True
            and item.get("status") != "error"
            for item in attempts
        )
        usage = metadata.get("usage")
        provider_succeeded = statuses[-1] == "ok"
        expected_logical_successes = 1 if provider_succeeded else 0
        if (
            not isinstance(usage, dict)
            or usage.get("calls") != expected_logical_successes
            or usage.get("provider_calls") != len(attempts)
            or usage.get("unknown_provider_attempts") != unknown_attempts
            or usage.get("input_tokens") != input_tokens
            or usage.get("output_tokens") != output_tokens
            or not isinstance(usage.get("cost_usd"), (int, float))
            or not math.isclose(
                float(usage["cost_usd"]), cost_usd, rel_tol=1e-12, abs_tol=1e-12
            )
        ):
            raise ReplayDivergence("structured service usage delta diverges")
        if provider_succeeded:
            provider_successes += 1
            if not success and service.get("error") != "ResourceLimitExceeded":
                raise ReplayDivergence(
                    "successful provider attempt has an invalid service failure"
                )
            if success:
                logical_successes += 1
            expected_successes = (
                2 if requested_model in _TOGETHER_MULTI_STAGE_MODELS else 1
            )
            if (
                statuses[-1] != "ok"
                or statuses.count("ok") != expected_successes
                or any(
                    status not in {"ok", "parse_invalid", "error"}
                    for status in statuses[:-1]
                )
            ):
                raise ReplayDivergence("successful service has invalid provider-attempt path")
            last_id = attempts[-1].get("provider_request_id")
            if (
                metadata.get("provider_request_id") != last_id
                or metadata.get("input_tokens") != input_tokens
                or metadata.get("output_tokens") != output_tokens
                or not isinstance(metadata.get("cost_usd"), (int, float))
                or not math.isclose(
                    float(metadata["cost_usd"]), cost_usd, rel_tol=1e-12, abs_tol=1e-12
                )
            ):
                raise ReplayDivergence("structured service totals diverge from attempts")
    # Structured calls may finish in a different order from the corresponding
    # provider requests (notably concurrent Judge work).  Pairing is exact by
    # the complete, already-validated attempt payload and role, but logical
    # calls are deliberately order-insensitive here.  Per-call attempt order
    # remains enforced above by each service row's 1..N ordinals.
    def _attempt_multiset(rows: list[dict[str, Any]]) -> Counter[str]:
        try:
            return Counter(
                json.dumps(
                    row,
                    ensure_ascii=False,
                    allow_nan=False,
                    separators=(",", ":"),
                    sort_keys=True,
                )
                for row in rows
            )
        except (TypeError, ValueError, UnicodeError) as exc:
            raise ReplayDivergence(
                "provider-attempt reconciliation payload is not canonical JSON"
            ) from exc

    if _attempt_multiset(durable_finished) != _attempt_multiset(service_finished):
        raise ReplayDivergence("provider-attempt journal and service metadata diverge")
    return {
        **summary,
        "logical_calls": logical_calls,
        "logical_successes": logical_successes,
        "provider_successes": provider_successes,
    }


def _atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        stream.write(json.dumps(_jsonable(value), ensure_ascii=False, indent=2, sort_keys=True))
        stream.flush()
        os.fsync(stream.fileno())
    try:
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        temporary.unlink(missing_ok=True)


def _service_event_data(event: ServiceEvent) -> dict[str, Any]:
    return _jsonable(asdict(event))


def _decode_service_event(item: dict[str, Any]) -> ServiceEvent:
    return ServiceEvent(
        item["kind"],
        item["request_hash"],
        item.get("response"),
        item.get("error"),
        item.get("request"),
        item.get("error_message"),
        item.get("started_at"),
        item.get("finished_at"),
        item.get("metadata") or {},
    )


def decode_service_tape(records: list[dict[str, Any]]) -> tuple[ServiceEvent, ...]:
    return tuple(_decode_service_event(item) for item in records)


def _actor_stream_data(checkpoint: ActorCheckpoint) -> dict[str, Any]:
    return {
        "branch_id": checkpoint.branch_id,
        "calls": [
            {
                "message": encode_message(call.message),
                "output_hash": call.output_hash,
                "service_event_count": call.service_event_count,
            }
            for call in checkpoint.calls
        ],
        "service_tape": [_service_event_data(event) for event in checkpoint.service_tape],
    }


def _actor_checkpoint_reference(checkpoint: ActorCheckpoint) -> dict[str, Any]:
    return {
        "branch_id": checkpoint.branch_id,
        "call_count": len(checkpoint.calls),
        "service_event_count": len(checkpoint.service_tape),
        "service_state": checkpoint.service_state,
    }


def _expand_actor_checkpoint_data(
    value: dict[str, Any], streams: dict[str, Any] | None,
) -> dict[str, Any]:
    """Expand a v2 cursor recipe, while accepting v1 inline checkpoints."""
    if "call_count" not in value and "service_event_count" not in value:
        return copy.deepcopy(value)
    try:
        branch_id = str(value["branch_id"])
        stream = (streams or {})[branch_id]
        call_count = int(value["call_count"])
        service_count = int(value["service_event_count"])
        calls = stream["calls"]
        service_tape = stream["service_tape"]
        if (
            call_count < 0
            or service_count < 0
            or call_count > len(calls)
            or service_count > len(service_tape)
        ):
            raise ValueError("actor cursor exceeds its branch stream")
        return {
            "branch_id": branch_id,
            "calls": copy.deepcopy(calls[:call_count]),
            "service_tape": copy.deepcopy(service_tape[:service_count]),
            "service_state": copy.deepcopy(value.get("service_state") or {}),
        }
    except (KeyError, TypeError, ValueError) as exc:
        raise ReplayDivergence("durable actor stream reference is invalid") from exc


def _decode_actor_checkpoint(
    value: dict[str, Any], factory: Any, streams: dict[str, Any] | None = None,
) -> ActorCheckpoint:
    try:
        value = _expand_actor_checkpoint_data(value, streams)
        calls = tuple(
            ActorCall(
                decode_message(record["message"]),
                str(record["output_hash"]),
                (
                    int(record["service_event_count"])
                    if record.get("service_event_count") is not None
                    else None
                ),
            )
            for record in value.get("calls", [])
        )
        tape = tuple(_decode_service_event(item) for item in value.get("service_tape", []))
        return ActorCheckpoint(
            factory,
            calls,
            tape,
            str(value.get("branch_id") or "root"),
            value.get("service_state") or {},
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ReplayDivergence("durable actor checkpoint is invalid") from exc


class RunRecorder:
    """Write one append-only run directory.

    Full events go to the private chain.  The public chain is independently
    redacted according to the disclosure profile.
    """

    def __init__(self, root: Path, run_id: str, manifest: dict[str, Any], *, disclosure: str = "development") -> None:
        if disclosure not in {"development", "hidden"}:
            raise ValidationError("unknown disclosure profile")
        self.root = root.resolve() / run_id
        self.root.mkdir(parents=True, exist_ok=False, mode=0o700)
        os.chmod(self.root, 0o700)
        (self.root / "logs").mkdir(mode=0o700)
        for role in ("generator", "oracle", "judge"):
            (self.root / "logs" / f"{role}.log").write_text("", encoding="utf-8")
        self.disclosure = disclosure
        self._public_after_submission = False
        self.public = HashChainWriter(self.root / "events.public.jsonl")
        self.private = HashChainWriter(self.root / "events.private.jsonl")
        self.service_journal = HashChainWriter(self.root / "service-calls.private.jsonl")
        self.provider_attempt_journal = HashChainWriter(
            self.root / "provider-attempts.private.jsonl"
        )
        self.guide_reasoning = HashChainWriter(
            self.root / "oracle-reasoning.private.jsonl"
        )
        self.manifest = {"schema_version": 1, "run_id": run_id, "disclosure": disclosure, **manifest}
        _atomic_json(self.root / "manifest.json", self.manifest)
        self._write_status(
            "running",
            phase="initializing",
            resumable=False,
            created_at=time.time(),
            pid=os.getpid(),
        )

    def record(self, event: dict[str, Any]) -> None:
        self.private.append(event)
        if event.get("kind") in {"run_started", "accounting_updated"}:
            self.manifest["accounting_version"] = event.get(
                "accounting_version", LEGACY_ACCOUNTING
            )
            _atomic_json(self.root / "manifest.json", self.manifest)
        if event.get("kind") == "stage_transition":
            path = self.root / "stage-handoff.private.json"
            existing: dict[str, Any] = {}
            if path.is_file():
                try:
                    loaded = json.loads(path.read_text(encoding="utf-8"))
                    if isinstance(loaded, dict):
                        existing = loaded
                except (OSError, json.JSONDecodeError):
                    existing = {}
            roles = dict(existing.get("roles") or {})
            roles[str(event.get("role") or "unknown")] = {
                "branch_id": event.get("branch_id"),
                "handoff_sha256": event.get("handoff_sha256"),
                "ready": _jsonable(event.get("ready")),
            }
            _atomic_json(
                path,
                {
                    "schema_version": 1,
                    "from_stage": event.get("from_stage"),
                    "to_stage": event.get("to_stage"),
                    "roles": roles,
                },
            )
        if event.get("kind") == "submission":
            self._public_after_submission = True
        public = _public_event(
            event,
            self.disclosure,
            after_submission=self._public_after_submission,
        )
        if public is not None:
            self.public.append(public)

    def record_service(self, role: str, event: ServiceEvent) -> None:
        """Durably record each live service attempt before the enclosing actor step finishes."""
        service = _service_event_data(event)
        self.service_journal.append({
            "kind": "service_call",
            "role": role,
            "service": service,
        })
        self._record_guide_reasoning(role, service, journal_kind="service_call")

    def record_provider_attempt(self, role: str, attempt: dict[str, Any]) -> None:
        """Durably record provider request start/finish before service completion."""

        self.provider_attempt_journal.append({
            "kind": "provider_attempt",
            "role": role,
            "attempt": copy.deepcopy(attempt),
        })

    def copy_provider_attempt_record(self, payload: dict[str, Any]) -> None:
        """Copy one already-validated provider-attempt payload on resume."""

        if (
            not isinstance(payload, dict)
            or set(payload) != {"kind", "role", "attempt"}
            or payload.get("kind") != "provider_attempt"
        ):
            raise ReplayDivergence("resume source has a malformed provider-attempt row")
        self.provider_attempt_journal.append(copy.deepcopy(payload))

    def _record_guide_reasoning(
        self,
        role: str,
        service: dict[str, Any],
        *,
        journal_kind: str,
    ) -> None:
        if role != "oracle" or service.get("kind") != "agent.session_turn":
            return
        request = service.get("request") if isinstance(service.get("request"), dict) else {}
        response = service.get("response") if isinstance(service.get("response"), dict) else {}
        metadata = service.get("metadata") if isinstance(service.get("metadata"), dict) else {}
        self.guide_reasoning.append({
            "kind": "oracle_agent_turn",
            "journal_kind": journal_kind,
            "request_hash": service.get("request_hash"),
            "session_id_requested": request.get("session_id"),
            "user": request.get("user"),
            "session_id": response.get("session_id"),
            "backend": response.get("backend") or metadata.get("agent_backend"),
            "model": response.get("model") or metadata.get("model"),
            "cli_version": response.get("cli_version") or metadata.get("cli_version"),
            "output": response.get("output"),
            "reasoning": response.get("reasoning") or [],
            "raw_events": response.get("raw_events") or metadata.get("raw_events") or [],
            "usage": response.get("usage") or metadata.get("usage") or {},
            "tool_policy_violation": metadata.get("tool_policy_violation") or [],
            "error": service.get("error"),
            "error_message": service.get("error_message"),
        })

    def copy_service_record(
        self,
        payload: dict[str, Any],
        *,
        abandon_from_run: str | None = None,
        source_sequence: int | None = None,
    ) -> None:
        """Copy a validated service record into a derived run.

        ``service_abandoned`` means the attempt remains charged and auditable,
        but is deliberately not offered to participant control flow as a
        replay response.  Only the explicit interrupted-call retry policy may
        create such a record.
        """
        role = str(payload.get("role") or "")
        kind = str(payload.get("kind") or "")
        service = payload.get("service")
        if (
            role not in {"generator", "oracle", "judge"}
            or kind not in {"service_call", "service_abandoned"}
            or not isinstance(service, dict)
        ):
            raise ReplayDivergence("resume source has a malformed service record")
        copied = copy.deepcopy(payload)
        if abandon_from_run is not None:
            copied.update({
                "kind": "service_abandoned",
                "abandoned_from_run": abandon_from_run,
                "source_sequence": source_sequence,
            })
        self.service_journal.append(copied)
        self._record_guide_reasoning(role, service, journal_kind=str(copied["kind"]))

    def _write_status(self, status: str, **fields: Any) -> None:
        existing: dict[str, Any] = {}
        path = self.root / "status.json"
        if path.is_file():
            try:
                loaded = json.loads(path.read_text(encoding="utf-8"))
                if isinstance(loaded, dict):
                    existing = loaded
            except (OSError, json.JSONDecodeError):
                existing = {}
        _atomic_json(
            path,
            {
                **existing,
                "schema_version": 1,
                "run_id": self.manifest["run_id"],
                "status": status,
                "updated_at": time.time(),
                **fields,
            },
        )

    def checkpoint(self, state: dict[str, Any], runner: Any, judge: Any) -> None:
        """Atomically persist a resumable Arena boundary.

        The checkpoint is private because it contains participant transcripts,
        service responses, the BranchStore capability secret, and RNG state.
        """
        if runner.last_generator is None or runner.last_guide is None or runner.last_branches is None:
            return
        generator = runner.runtime.checkpoint(runner.last_generator)
        guide = runner.runtime.checkpoint(runner.last_guide)
        role_checkpoints = {
            "generator": [
                runner.runtime.checkpoint(handle) for _, handle in runner.generator_history
            ],
            "oracle": [
                runner.runtime.checkpoint(handle) for _, handle in runner.guide_history
            ],
        }
        actor_streams = {
            role: {
                checkpoint.branch_id: _actor_stream_data(checkpoint)
                for checkpoint in checkpoints
            }
            for role, checkpoints in role_checkpoints.items()
        }
        encoded_state = dict(state)
        for key in (
            "output",
            "decision",
            "submission",
            "submission_feedback",
        ):
            value = encoded_state.get(key)
            if value is not None:
                encoded_state[key] = encode_message(value)
        transitions = encoded_state.get("stage_transitions")
        if transitions is not None:
            encoded_state["stage_transitions"] = [
                encode_message(value) for value in transitions
            ]
        judge_services = getattr(judge, "services", None)
        value = {
            "schema_version": 3,
            "run_id": self.manifest["run_id"],
            "private_event_count": self.private.sequence,
            "public_event_count": self.public.sequence,
            "service_event_count": self.service_journal.sequence,
            "provider_attempt_event_count": self.provider_attempt_journal.sequence,
            "engine": encoded_state,
            "actors": {
                "generator": _actor_checkpoint_reference(generator),
                "oracle": _actor_checkpoint_reference(guide),
            },
            "actor_streams": actor_streams,
            "actor_history": {
                "generator": [
                    _actor_checkpoint_reference(checkpoint)
                    for checkpoint in role_checkpoints["generator"]
                ],
                "oracle": [
                    _actor_checkpoint_reference(checkpoint)
                    for checkpoint in role_checkpoints["oracle"]
                ],
            },
            "branches": runner.last_branches.export_state(_actor_checkpoint_reference),
            "judge_service_tape": (
                [_service_event_data(event) for event in judge_services.export_tape()]
                if judge_services is not None else []
            ),
            "judge_service_state": (
                judge_services.export_state() if judge_services is not None else {}
            ),
            "updated_at": time.time(),
        }
        _atomic_json(self.root / "checkpoint.private.json", value)
        self._write_status(
            "running",
            phase=str(state.get("phase") or "unknown"),
            resumable=True,
            questions=len(runner.last_branches.nodes),
            oracle_decisions=int(state.get("decisions", 0)),
            checkouts=int(state.get("checkouts", 0)),
            submissions=int(state.get("submission_attempts", 0)),
            K=float(state.get("k", 0.0)),
        )

    def seed_resume_checkpoint(self, source: dict[str, Any]) -> None:
        """Rebase one already-validated source boundary onto this derived run.

        The derived service journal may already contain a write-ahead suffix
        from the interrupted source call.  Keep the source's committed service
        cursor so that suffix remains checkpoint-external until the continued
        call consumes it; a normal runtime checkpoint will replace this seed at
        the next durable Arena boundary.
        """
        value = copy.deepcopy(source.get("_resume_seed_checkpoint", source))
        try:
            service_cursor = int(value["service_event_count"])
            provider_attempt_cursor = int(value.get("provider_attempt_event_count", 0))
            state = value["engine"]
            branches = value["branches"]
        except (KeyError, TypeError, ValueError) as exc:
            raise ReplayDivergence("resume seed checkpoint is malformed") from exc
        if (
            not isinstance(state, dict)
            or not isinstance(branches, dict)
            or service_cursor < 0
            or service_cursor > self.service_journal.sequence
            or provider_attempt_cursor < 0
            or provider_attempt_cursor > self.provider_attempt_journal.sequence
        ):
            raise ReplayDivergence("resume seed checkpoint has an invalid cursor or state")
        run_id = self.manifest["run_id"]
        value.update({
            "run_id": run_id,
            "private_event_count": self.private.sequence,
            "public_event_count": self.public.sequence,
            "service_event_count": service_cursor,
            "provider_attempt_event_count": provider_attempt_cursor,
            "updated_at": time.time(),
        })
        branches["run_id"] = run_id
        for key in (
            "resume_judge_tape",
            "resume_judge_meter",
            "resume_judge_usage",
            "resume_committed_judge_tape",
            "resume_judge_state",
        ):
            value.pop(key, None)
        _atomic_json(self.root / "checkpoint.private.json", value)
        self._write_status(
            "running",
            phase=str(state.get("phase") or "unknown"),
            resumable=True,
            questions=len((branches.get("nodes") or {})),
            oracle_decisions=int(state.get("decisions", 0)),
            checkouts=int(state.get("checkouts", 0)),
            submissions=int(state.get("submission_attempts", 0)),
            K=float(state.get("k", 0.0)),
        )

    def _public_event(self, event: dict[str, Any]) -> dict[str, Any] | None:
        return _public_event(
            event,
            self.disclosure,
            after_submission=self._public_after_submission,
        )

    def finalize(self, result: Any, runner: Any, judge: Any) -> None:
        branches = result.branch_store
        branch_data = _branch_data(branches)
        _atomic_json(self.root / "branches.json", branch_data)
        score = {
            "status": result.status,
            "accounting_version": branches.accounting_version,
            "score": result.score,
            "K": result.k,
            "matched_idea_ids": list(result.matched_idea_ids),
            "questions": result.question_count,
            "oracle_decisions": result.guide_decision_count,
            "checkouts": result.checkout_count,
            "submission_attempts": result.submission_attempt_count,
            "judge_repeats": result.judge_repeats,
            "judge_passes": result.judge_passes,
            "judge_pass_rate": result.judge_pass_rate,
            "repeat_bits": result.repeat_bits,
        }
        _atomic_json(self.root / "score.json", score)
        self._write_actor("generator", runner.last_generator)
        self._write_actor("oracle", runner.last_guide)
        self._write_actor_history("generator", runner.generator_history)
        self._write_actor_history("oracle", runner.guide_history)
        services = getattr(judge, "services", None)
        if services is not None:
            self._write_service_tape("judge", services.export_tape())
        usage = {
            "generator": runner.last_generator.services.usage(),
            "oracle": runner.last_guide.services.usage(),
            "judge_service_events": len(services.export_tape()) if services is not None else 0,
        }
        if services is not None:
            usage["judge"] = services.usage()
        _atomic_json(self.root / "usage.json", usage)
        accounting = {
            "schema_version": 1,
            "run_id": self.manifest["run_id"],
            "service_event_count": self.service_journal.sequence,
            "provider_attempt_event_count": self.provider_attempt_journal.sequence,
            "actors": {
                "generator": {
                    "branch_id": runner.last_generator.branch_id,
                    "service_state": runner.last_generator.services.export_state(),
                },
                "oracle": {
                    "branch_id": runner.last_guide.branch_id,
                    "service_state": runner.last_guide.services.export_state(),
                },
            },
            "judge_service_state": (
                services.export_state() if services is not None else {}
            ),
        }
        _atomic_json(self.root / "service-accounting.private.json", accounting)
        self._write_status(
            "completed",
            result_status=result.status,
            resumable=False,
            score=result.score,
            K=result.k,
        )
        checkpoint_path = self.root / "checkpoint.private.json"
        promotion_path = self.root / "promotion-checkpoint.private.json"
        if result.status == "pass" and checkpoint_path.is_file():
            # A successful run normally ends at the Judge call immediately
            # following a durable ``before_judge`` boundary. Retain that
            # private boundary so an explicitly stronger evaluation rung can
            # fork the same paid information state without replaying or
            # double-counting the weaker Judge's passing-mass pointer.
            try:
                promotion_checkpoint = json.loads(
                    checkpoint_path.read_text(encoding="utf-8")
                )
            except (OSError, json.JSONDecodeError) as exc:
                raise ReplayDivergence(
                    "completed pass has no valid promotion checkpoint"
                ) from exc
            if (promotion_checkpoint.get("engine") or {}).get("phase") != "before_judge":
                raise ReplayDivergence(
                    "completed pass promotion checkpoint is not before_judge"
                )
            _atomic_json(promotion_path, promotion_checkpoint)
        else:
            promotion_path.unlink(missing_ok=True)
        # Mark completion before removing recovery state. A crash between these
        # operations may leave a harmless stale checkpoint, never a false
        # resumable status without the checkpoint it promises.
        checkpoint_path.unlink(missing_ok=True)

    def finalize_failure(self, error_code: str, runner: Any, judge: Any | None = None) -> None:
        score = {"status": "error", "score": 0.0, "error_code": error_code,
                 "accounting_version": self.manifest.get("accounting_version", LEGACY_ACCOUNTING)}
        _atomic_json(self.root / "score.json", score)
        for role in ("generator", "oracle"):
            handle = getattr(runner, f"last_{role}", None)
            if handle is not None:
                self._write_actor(role, handle)
            history = getattr(runner, f"{role}_history", ())
            if history:
                self._write_actor_history(role, history)
        branches = getattr(runner, "last_branches", None)
        _atomic_json(self.root / "branches.json", _branch_data(branches))
        usage = {}
        for role in ("generator", "oracle"):
            handle = getattr(runner, f"last_{role}", None)
            if handle is not None:
                usage[role] = handle.services.usage()
        judge_services = getattr(judge, "services", None)
        if judge_services is not None:
            self._write_service_tape("judge", judge_services.export_tape())
            usage["judge"] = judge_services.usage()
            usage["judge_service_events"] = len(judge_services.export_tape())
        _atomic_json(self.root / "usage.json", usage)
        self._write_status(
            "error",
            error_code=error_code,
            resumable=(self.root / "checkpoint.private.json").is_file(),
        )

    def _write_actor(self, role: str, handle: Any) -> None:
        records = [
            {
                "message": _encode_actor_message(call.message),
                "output_hash": call.output_hash,
                "service_event_count": call.service_event_count,
            }
            for call in handle.calls
        ]
        _atomic_json(self.root / f"actor-calls.{role}.private.json", records)
        self._write_service_tape(role, handle.services.export_tape())
        factory = handle.factory.service_factory
        errors = [event for event in factory.export_audit_tape() if event.error is not None]
        if errors:
            _atomic_json(
                self.root / f"service-errors.{role}.private.json",
                [_jsonable(event) for event in errors],
            )

    def _write_actor_history(self, role: str, history: list[tuple[str, Any]]) -> None:
        if len(history) <= 1:
            (self.root / f"actor-branches.{role}.private.json").unlink(missing_ok=True)
            return
        # Multi-branch replay consumes the history file; remove the duplicate
        # current-branch call and service tapes written by _write_actor().
        (self.root / f"actor-calls.{role}.private.json").unlink(missing_ok=True)
        (self.root / f"service-tape.{role}.private.json").unlink(missing_ok=True)
        branches = []
        for branch_id, handle in history:
            branches.append({
                "branch_id": branch_id,
                "calls": [
                    {
                        "message": _encode_actor_message(call.message),
                        "output_hash": call.output_hash,
                        "service_event_count": call.service_event_count,
                    }
                    for call in handle.calls
                ],
                "service_tape": [_jsonable(event) for event in handle.services.export_tape()],
            })
        _atomic_json(self.root / f"actor-branches.{role}.private.json", branches)

    def _write_service_tape(self, role: str, tape: tuple[ServiceEvent, ...]) -> None:
        _atomic_json(
            self.root / f"service-tape.{role}.private.json",
            [_jsonable(event) for event in tape],
        )


def _encode_actor_message(message: Any) -> dict[str, Any]:
    try:
        return encode_message(message)
    except TypeError as exc:
        raise ValidationError(
            f"unsupported actor replay message {type(message).__name__}"
        ) from exc


def _decode_question(value: dict[str, Any]) -> Question:
    def decode_option(item: dict[str, Any]) -> Option | SubmitOption:
        if item.get("kind") == "submit":
            return SubmitOption(
                item["option_id"], item.get("public_payload"), item["probability"]
            )
        return Option(item["option_id"], item.get("public_payload"), item["probability"])

    return Question(
        value["question"],
        tuple(decode_option(item) for item in value["options"]),
    )


def _decode_actor_message(record: dict[str, Any]) -> Any:
    try:
        return decode_message(record)
    except (KeyError, TypeError, ValueError) as exc:
        raise ReplayDivergence(
            f"unsupported stored actor message type {record.get('type')!r}"
        ) from exc


def _load_calls(path: Path) -> tuple[ActorCall, ...]:
    try:
        records = json.loads(path.read_text(encoding="utf-8"))
        return tuple(
            ActorCall(
                _decode_actor_message(record["message"]),
                record["output_hash"],
                (
                    int(record["service_event_count"])
                    if record.get("service_event_count") is not None
                    else None
                ),
            )
            for record in records
        )
    except (OSError, KeyError, TypeError, json.JSONDecodeError) as exc:
        raise ReplayDivergence(f"invalid actor call tape {path.name}") from exc


def _load_service_tape(path: Path) -> tuple[ServiceEvent, ...]:
    try:
        records = json.loads(path.read_text(encoding="utf-8"))
        return tuple(_decode_service_event(item) for item in records)
    except (OSError, KeyError, TypeError, json.JSONDecodeError) as exc:
        raise ReplayDivergence(f"invalid service tape {path.name}") from exc


def _decode_call_records(records: list[dict[str, Any]], source: str) -> tuple[ActorCall, ...]:
    try:
        return tuple(
            ActorCall(
                _decode_actor_message(record["message"]),
                record["output_hash"],
                (
                    int(record["service_event_count"])
                    if record.get("service_event_count") is not None
                    else None
                ),
            )
            for record in records
        )
    except (KeyError, TypeError) as exc:
        raise ReplayDivergence(f"invalid actor calls in {source}") from exc


def _decode_service_records(records: list[dict[str, Any]], source: str) -> tuple[ServiceEvent, ...]:
    try:
        return tuple(_decode_service_event(item) for item in records)
    except (KeyError, TypeError) as exc:
        raise ReplayDivergence(f"invalid service tape in {source}") from exc


def _load_actor_history(path: Path) -> tuple[tuple[str, tuple[ActorCall, ...], tuple[ServiceEvent, ...]], ...]:
    try:
        records = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(records, list):
            raise TypeError("history is not a list")
        result = []
        seen: set[str] = set()
        for record in records:
            branch_id = str(record["branch_id"])
            if not branch_id or branch_id in seen:
                raise TypeError("branch ID is missing or duplicated")
            seen.add(branch_id)
            result.append((
                branch_id,
                _decode_call_records(record["calls"], path.name),
                _decode_service_records(record["service_tape"], path.name),
            ))
        return tuple(result)
    except (OSError, KeyError, TypeError, json.JSONDecodeError) as exc:
        if isinstance(exc, ReplayDivergence):
            raise
        raise ReplayDivergence(f"invalid actor branch history {path.name}") from exc


def _decode_submission(value: dict[str, Any]) -> Submission:
    return Submission(
        tuple(
            Idea(item["idea_id"], item.get("content"), item["probability"])
            for item in value["ideas"]
        )
    )


@legacy_fields(oracle_calls='guide_calls', oracle_pending='guide_pending', pending_oracle_input='pending_guide_input')
@dataclass(slots=True)
class _ActorProtocolExpectations:
    generator_branches: dict[str, tuple[ActorCall, ...]]
    generator_branch_order: tuple[str, ...]
    node_generator_refs: dict[str, tuple[str, int]]
    current_generator_branch: str
    generator_pending: bool
    pending_generator_input: Any
    guide_calls: tuple[ActorCall, ...]
    guide_pending: bool
    pending_guide_input: Any
    judge_pending: bool


def _decode_protocol_decision(value: Any) -> Choice | Checkout:
    if not isinstance(value, dict):
        raise ReplayDivergence("oracle decision is malformed")
    if set(value) == {"option_id", "question_id", "public_payload"}:
        return Choice(
            value["option_id"],
            value["question_id"],
            value["public_payload"],
        )
    if set(value) == {"question_id"}:
        return Checkout(value["question_id"])
    raise ReplayDivergence("oracle decision does not match its wire shape")


def _derive_actor_expectations(
    private: tuple[dict[str, Any], ...],
) -> _ActorProtocolExpectations:
    """Derive trusted actor transcripts solely from committed Arena events."""

    start = next((item for item in private if item.get("kind") == "run_started"), {})
    if start.get("protocol_event_schema") not in {2, 3, 4}:
        raise ReplayDivergence("actor binding requires a submit-aware event trace")
    time_travel = bool(start.get("time_travel"))
    questions: dict[str, Question] = {}
    question_order: list[str] = []
    generator: dict[str, list[ActorCall]] = {"root": []}
    generator_order = ["root"]
    current_branch = "root"
    node_refs: dict[str, tuple[str, int]] = {}
    generator_pending = True
    pending_generator_input: Any = None
    guide_calls: list[ActorCall] = []
    guide_pending = False
    pending_guide_input: Any = None
    active_submission: Submission | None = None
    judge_pending = False
    checkout_index = 0

    for record in private:
        kind = record.get("kind")
        if kind == "stage_transition":
            try:
                transition_value = record["transition"]
                ready_value = record["ready"]
                transition = StageTransition(
                    str(transition_value["from_stage"]),
                    str(transition_value["to_stage"]),
                )
                ready = StageReady(
                    str(ready_value["stage"]), ready_value.get("handoff")
                )
                role = str(record["role"])
                branch_id = str(record["branch_id"])
            except (KeyError, TypeError, ValueError) as exc:
                raise ReplayDivergence("stage transition event is malformed") from exc
            if ready.stage != transition.to_stage:
                raise ReplayDivergence("stage transition acknowledgement has the wrong stage")
            call = ActorCall(transition, message_hash(ready))
            if role == "generator":
                if branch_id != current_branch:
                    raise ReplayDivergence("stage transition targets the wrong Generator branch")
                generator[current_branch].append(call)
            elif role == "oracle":
                if branch_id != "root":
                    raise ReplayDivergence("stage transition targets a non-root Oracle")
                guide_calls.append(call)
            else:
                raise ReplayDivergence("stage transition has an unknown actor role")
            continue
        if kind == "question":
            if not generator_pending:
                raise ReplayDivergence("question has no matching Generator call")
            question = _decode_question(record["question"])
            question_id = str(record["question_id"])
            generator[current_branch].append(
                ActorCall(pending_generator_input, message_hash(question))
            )
            generator_pending = False
            pending_generator_input = None
            questions[question_id] = question
            question_order.append(question_id)
            node_refs[question_id] = (
                current_branch,
                len(generator[current_branch]),
            )
            pending_guide_input = PresentedQuestion(question_id, question)
            guide_pending = True
            continue

        if kind == "oracle_decision":
            if not guide_pending:
                raise ReplayDivergence("oracle decision has no trusted actor input")
            decision = _decode_protocol_decision(record.get("decision"))
            guide_calls.append(
                ActorCall(pending_guide_input, message_hash(decision))
            )
            guide_pending = False
            pending_guide_input = None
            continue

        if kind == "choice_cost":
            source_id = str(record.get("question_id"))
            question = questions.get(source_id)
            if question is None:
                raise ReplayDivergence("choice cost has no trusted source question")
            option_id = str(record.get("option_id"))
            option = next(
                (item for item in question.options if item.option_id == option_id),
                None,
            )
            if option is None:
                raise ReplayDivergence("choice cost has no trusted source option")
            pending_generator_input = Choice(
                option.option_id,
                public_payload=option.public_payload,
            )
            generator_pending = True
            continue

        if kind == "submission":
            if not generator_pending:
                raise ReplayDivergence("submission has no matching Generator call")
            active_submission = _decode_submission(record["submission"])
            generator[current_branch].append(
                ActorCall(
                    pending_generator_input,
                    message_hash(active_submission),
                )
            )
            generator_pending = False
            pending_generator_input = None
            judge_pending = True
            continue

        if kind == "submission_judged":
            if active_submission is None or not judge_pending:
                raise ReplayDivergence("judgment has no trusted submission")
            judge_pending = False
            if record.get("status") == "fail":
                verdicts = tuple(
                    IdeaVerdict(
                        value["idea_id"],
                        value["passed"],
                        str(value.get("private_reason") or ""),
                    )
                    for value in record.get("verdicts", [])
                )
                source_id = str(record.get("source_question_id"))
                source = questions.get(source_id)
                if source is None:
                    raise ReplayDivergence("feedback source question is missing")
                source_index = question_order.index(source_id)
                valid_ids = (
                    tuple((*question_order[:source_index], source_id))
                    if time_travel
                    else (source_id,)
                )
                feedback = SubmissionFeedback(
                    PresentedQuestion(source_id, source),
                    active_submission,
                    verdicts,
                    valid_ids,
                )
                IDEA_RECOVERY_V1.validate_submission_feedback(feedback)
                pending_guide_input = feedback
                guide_pending = True
            continue

        if kind == "checkout":
            target_id = str(record.get("target_question_id"))
            target_ref = node_refs.get(target_id)
            target_question = questions.get(target_id)
            if target_ref is None or target_question is None:
                raise ReplayDivergence("checkout target has no trusted Generator prefix")
            checkout_index += 1
            branch_id = f"branch-{checkout_index:08d}"
            source_branch, call_count = target_ref
            generator[branch_id] = list(generator[source_branch][:call_count])
            generator_order.append(branch_id)
            current_branch = branch_id
            generator_pending = False
            pending_generator_input = None
            pending_guide_input = PresentedQuestion(target_id, target_question)
            guide_pending = True
            if record.get("context") == "submission_recovery":
                active_submission = None
            continue

    return _ActorProtocolExpectations(
        generator_branches={key: tuple(value) for key, value in generator.items()},
        generator_branch_order=tuple(generator_order),
        node_generator_refs=node_refs,
        current_generator_branch=current_branch,
        generator_pending=generator_pending,
        pending_generator_input=pending_generator_input,
        guide_calls=tuple(guide_calls),
        guide_pending=guide_pending,
        pending_guide_input=pending_guide_input,
        judge_pending=judge_pending,
    )


def _judge_call_is_pending(
    expectations: _ActorProtocolExpectations,
) -> bool:
    """Whether the protocol position admits an in-flight Judge ``evaluate``.

    The Judge is called from two seats, not one: the formal judgment of a
    committed submission, and the Oracle's own judge_evaluate service, which it
    may reach for at any point in its turn. Only the first sets
    ``judge_pending``, so recognising it alone would reject an interruption of
    the second.
    """

    if expectations.judge_pending:
        return True
    return bool(expectations.guide_pending)


def _assert_actor_calls_bound(
    actual: tuple[ActorCall, ...],
    expected: tuple[ActorCall, ...],
    *,
    role: str,
    allow_extra_input: Any = None,
    allow_extra: bool = False,
) -> bool:
    """Bind a stored transcript to trusted messages; return whether one extra call exists."""

    extra = len(actual) == len(expected) + 1
    if len(actual) != len(expected) and not (allow_extra and extra):
        raise ReplayDivergence(f"{role} actor call count disagrees with protocol events")
    for index, (stored, trusted) in enumerate(zip(actual, expected, strict=False)):
        if stored.message != trusted.message or stored.output_hash != trusted.output_hash:
            raise ReplayDivergence(
                f"{role} actor call {index} is not bound to protocol events"
            )
    if extra and actual[-1].message != allow_extra_input:
        raise ReplayDivergence(f"{role} trailing actor call has an invalid input")
    return extra


def _checkpoint_calls(value: dict[str, Any], source: str) -> tuple[ActorCall, ...]:
    calls = value.get("calls")
    if not isinstance(calls, list):
        raise ReplayDivergence(f"{source} actor calls are malformed")
    return _decode_call_records(calls, source)


def _validate_actor_service_cursors(
    value: dict[str, Any],
    *,
    source: str,
    require_cursors: bool = False,
) -> None:
    calls = _checkpoint_calls(value, source)
    tape = value.get("service_tape")
    if not isinstance(tape, list):
        raise ReplayDivergence(f"{source} actor service tape is malformed")
    prior = 0
    for index, call in enumerate(calls):
        cursor = call.service_event_count
        if cursor is None:
            if require_cursors:
                raise ReplayDivergence(
                    f"{source} actor call {index} has no service cursor"
                )
            # Legacy schema-2 checkpoints did not bind per-call service cursors.
            continue
        if cursor < prior or cursor > len(tape):
            raise ReplayDivergence(
                f"{source} actor service cursor {index} is inconsistent"
            )
        prior = cursor
    if calls and calls[-1].service_event_count is not None:
        if calls[-1].service_event_count != len(tape):
            raise ReplayDivergence(
                f"{source} actor tape extends beyond its last committed call"
            )


def _service_key(value: ServiceEvent | dict[str, Any]) -> bytes:
    return _canonical_bytes(
        _service_event_data(value) if isinstance(value, ServiceEvent) else value
    )


def _branch_rng_state(root_seed: int, branch_id: str) -> Any:
    if branch_id == "root":
        rng = random.Random(root_seed)
    else:
        branch_seed = hashlib.sha256(
            f"{root_seed}:{branch_id}".encode("utf-8")
        ).digest()
        rng = random.Random(int.from_bytes(branch_seed[:8], "big"))
    return _jsonable(rng.getstate())


def _parse_service_meter(value: Any, *, source: str) -> dict[str, int]:
    if not isinstance(value, dict) or set(value) != {
        "model_calls",
        "random_calls",
    }:
        raise ReplayDivergence(f"{source} service meter is malformed")
    result: dict[str, int] = {}
    for key in ("model_calls", "random_calls"):
        counter = value.get(key)
        if isinstance(counter, bool) or not isinstance(counter, int) or counter < 0:
            raise ReplayDivergence(f"{source} service meter is malformed")
        result[key] = counter
    return result


def _is_model_service(event: ServiceEvent) -> bool:
    return event.kind in {"model.structured", "agent.session_turn"}


def _accumulate_model_usage(
    initial: dict[str, int | float],
    events: Iterable[ServiceEvent],
) -> dict[str, int | float]:
    """Add only service events that actually spend a model turn.

    Proxy calls such as ``judge.evaluate`` may carry the caller backend's
    latest cumulative usage snapshot for diagnostics. They are not model
    turns and counting that snapshot again corrupts resumed accounting.
    """

    result = dict(initial)
    for event in events:
        if not _is_model_service(event):
            continue
        for key, value in ((event.metadata or {}).get("usage") or {}).items():
            result[key] = result.get(key, 0) + value
    return result


def _model_service_units(event: ServiceEvent, *, source: str) -> int:
    """Return the audited model-turn units consumed by one service event.

    Ordinary structured calls and failed agent turns reserve one unit.  A
    successful agentic Generator call can additionally charge its completed
    child turns; those are reported in ``metadata.usage.turns`` and are bound
    to the same service event.
    """

    if not _is_model_service(event) or event.kind != "agent.session_turn":
        return 1
    metadata = event.metadata or {}
    usage = metadata.get("usage")
    if usage is None:
        return 1
    if not isinstance(usage, dict):
        raise ReplayDivergence(f"{source} agent usage is malformed")
    turns = usage.get("turns")
    if turns is None:
        return 1
    if isinstance(turns, bool) or not isinstance(turns, int) or turns < 1:
        raise ReplayDivergence(f"{source} agent turn count is malformed")
    return turns


def _advance_service_meter(
    prior: dict[str, int],
    event: ServiceEvent,
    *,
    source: str,
    require_metadata: bool,
) -> dict[str, int]:
    """Validate one journaled attempt and return its bound global meter."""

    raw = (event.metadata or {}).get("service_meter")
    if raw is None and not require_metadata:
        # Legacy schema-2 journals did not bind counters. Preserve their old
        # best-effort reconstruction behavior.
        result = dict(prior)
        if _is_model_service(event):
            result["model_calls"] += 1
        elif event.kind.startswith("random."):
            result["random_calls"] += 1
        return result
    current = _parse_service_meter(raw, source=source)
    model_delta = current["model_calls"] - prior["model_calls"]
    random_delta = current["random_calls"] - prior["random_calls"]
    expected_counter = (
        "model_calls"
        if _is_model_service(event)
        else "random_calls"
        if event.kind.startswith("random.")
        else None
    )
    if expected_counter is None:
        if model_delta != 0 or random_delta != 0:
            raise ReplayDivergence(f"{source} service meter changes on an unknown event")
        return current
    active_delta = current[expected_counter] - prior[expected_counter]
    inactive_counter = (
        "random_calls" if expected_counter == "model_calls" else "model_calls"
    )
    inactive_delta = current[inactive_counter] - prior[inactive_counter]
    # An agentic Generator success consumes the main turn plus its completed
    # child turns. A ResourceLimitExceeded attempt may be rejected by the
    # pre-call check and consume zero, or fail after some/all reported turns.
    units = (
        _model_service_units(event, source=source)
        if expected_counter == "model_calls"
        else 1
    )
    allowed = (
        set(range(units + 1))
        if event.error == "ResourceLimitExceeded"
        else {units}
    )
    if active_delta not in allowed or inactive_delta != 0:
        raise ReplayDivergence(f"{source} service meter is not monotonic per attempt")
    return current


def _journal_service_meter(
    events: tuple[ServiceEvent, ...] | list[ServiceEvent],
    *,
    source: str,
    initial: dict[str, int] | None = None,
    require_metadata: bool,
) -> dict[str, int]:
    meter = dict(initial or {"model_calls": 0, "random_calls": 0})
    if not require_metadata:
        for index, event in enumerate(events):
            meter = _advance_service_meter(
                meter,
                event,
                source=f"{source} event {index}",
                require_metadata=False,
            )
        return meter

    # Judge model calls may run concurrently. The shared meter is reserved
    # when each call starts, but journal records are appended when calls
    # finish. A valid completion-ordered tape can therefore contain snapshots
    # such as 1, 3, 3 (or even 3, 3, 2 if append-lock acquisition is reordered).
    # Validate conservation across the complete batch instead of requiring
    # every completion record to advance its snapshot by exactly one.
    observed: list[tuple[ServiceEvent, dict[str, int]]] = []
    required = {"model_calls": 0, "random_calls": 0}
    possible = {"model_calls": 0, "random_calls": 0}
    for index, event in enumerate(events):
        event_source = f"{source} event {index}"
        current = _parse_service_meter(
            (event.metadata or {}).get("service_meter"), source=event_source
        )
        for counter in ("model_calls", "random_calls"):
            if current[counter] < meter[counter]:
                raise ReplayDivergence(
                    f"{event_source} service meter predates the journal boundary"
                )
        counter = (
            "model_calls"
            if _is_model_service(event)
            else "random_calls"
            if event.kind.startswith("random.")
            else None
        )
        if counter is not None:
            units = (
                _model_service_units(event, source=event_source)
                if counter == "model_calls"
                else 1
            )
            possible[counter] += units
            if event.error != "ResourceLimitExceeded":
                required[counter] += units
                if current[counter] <= meter[counter]:
                    raise ReplayDivergence(
                        f"{event_source} service meter does not include its attempt"
                    )
        observed.append((event, current))

    final = {
        counter: max(
            [meter[counter], *(current[counter] for _, current in observed)]
        )
        for counter in ("model_calls", "random_calls")
    }
    for counter in ("model_calls", "random_calls"):
        delta = final[counter] - meter[counter]
        if not required[counter] <= delta <= possible[counter]:
            raise ReplayDivergence(
                f"{source} service meter disagrees with its attempt count"
            )
    return final


def _parse_model_usage(value: Any, *, source: str) -> dict[str, int | float]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ReplayDivergence(f"{source} model usage is malformed")
    result: dict[str, int | float] = {}
    for key, amount in value.items():
        if (
            not isinstance(key, str)
            or isinstance(amount, bool)
            or not isinstance(amount, (int, float))
            or not math.isfinite(float(amount))
            or amount < 0
        ):
            raise ReplayDivergence(f"{source} model usage is malformed")
        result[key] = amount
    return result


def _validate_service_tail_state(
    events: tuple[ServiceEvent, ...],
    *,
    initial_state: dict[str, Any],
    source: str,
    branch_id: str,
    require_metadata: bool,
) -> tuple[dict[str, int], dict[str, int | float]]:
    """Validate an uncommitted service suffix without executing participant code."""

    # Apply the same explicit safe-exception policy as ReplayableServices
    # before interpreting counters, cursors, or a Judge retry sequence.  An
    # unjudged failure tail is not necessarily executed during actor replay,
    # so constructor-time tape validation alone cannot protect this path.
    validate_service_tape_errors(events, source=source)
    meter = _parse_service_meter(initial_state.get("meter"), source=source)
    usage = _parse_model_usage(initial_state.get("model_usage") or {}, source=source)

    def tuples(value: Any) -> Any:
        if isinstance(value, list):
            return tuple(tuples(item) for item in value)
        return value

    rng = random.Random()
    try:
        rng.setstate(tuples(initial_state["rng_state"]))
    except (KeyError, TypeError, ValueError) as exc:
        raise ReplayDivergence(f"{source} RNG state is malformed") from exc

    # Concurrent model calls reserve the shared meter when they start but are
    # journaled when they finish, so their snapshots may interleave out of
    # increment order. Validate model-call accounting by batch conservation,
    # exactly like ``_journal_service_meter``; random calls remain serial and
    # keep exact per-event deltas because RNG verification depends on them.
    boundary = dict(meter)
    max_seen = dict(meter)
    required = {"model_calls": 0, "random_calls": 0}
    possible = {"model_calls": 0, "random_calls": 0}
    last_random = meter["random_calls"]
    for index, event in enumerate(events):
        event_source = f"{source} event {index}"
        metadata = event.metadata
        if not isinstance(metadata, dict):
            raise ReplayDivergence(f"{event_source} metadata is malformed")
        if require_metadata:
            current = _parse_service_meter(
                metadata.get("service_meter"), source=event_source
            )
            for counter in ("model_calls", "random_calls"):
                if current[counter] < boundary[counter]:
                    raise ReplayDivergence(
                        f"{event_source} service meter predates the journal boundary"
                    )
                max_seen[counter] = max(max_seen[counter], current[counter])
            counter = (
                "model_calls"
                if _is_model_service(event)
                else "random_calls"
                if event.kind.startswith("random.")
                else None
            )
            if counter is not None:
                units = (
                    _model_service_units(event, source=event_source)
                    if counter == "model_calls"
                    else 1
                )
                possible[counter] += units
                if event.error != "ResourceLimitExceeded":
                    required[counter] += units
                    if current[counter] <= boundary[counter]:
                        raise ReplayDivergence(
                            f"{event_source} service meter does not include its attempt"
                        )
            if event.kind.startswith("random."):
                consumed = current["random_calls"] - last_random
                last_random = current["random_calls"]
            else:
                consumed = 0
            meter = max_seen
        else:
            prior_meter = meter
            meter = _advance_service_meter(
                meter,
                event,
                source=event_source,
                require_metadata=False,
            )
            consumed = meter["random_calls"] - prior_meter["random_calls"]
        if require_metadata and metadata.get("service_branch_id") != branch_id:
            raise ReplayDivergence(f"{event_source} branch metadata diverges")

        delta_usage = _parse_model_usage(metadata.get("usage") or {}, source=event_source)
        if delta_usage and not _is_model_service(event):
            if event.kind == "judge.evaluate":
                # The proxy can carry the caller backend's latest cumulative
                # snapshot for diagnostics; it did not itself spend a model
                # turn and must not be added again.
                delta_usage = {}
            else:
                raise ReplayDivergence(
                    f"{event_source} attaches model usage to a non-model call"
                )
        for key, amount in delta_usage.items():
            usage[key] = usage.get(key, 0) + amount

        if event.kind.startswith("random."):
            if consumed == 1:
                try:
                    if event.kind == "random.random":
                        actual = rng.random()
                    elif event.kind == "random.randint":
                        actual = rng.randint(
                            int(event.request["start"]), int(event.request["stop"])
                        )
                    elif event.kind == "random.choice":
                        actual = rng.choice(event.request["values"])
                    else:
                        raise ReplayDivergence(f"{event_source} has an unknown random service kind")
                except ReplayDivergence:
                    raise
                except Exception as exc:  # noqa: BLE001
                    if event.error != type(exc).__name__ or event.response is not None:
                        raise ReplayDivergence(
                            f"{event_source} random failure disagrees with RNG state"
                        ) from exc
                else:
                    if event.error is not None or _canonical_bytes(actual) != _canonical_bytes(
                        event.response
                    ):
                        raise ReplayDivergence(
                            f"{event_source} random response disagrees with RNG state"
                        )
        elif not _is_model_service(event) and event.kind != "judge.evaluate":
            raise ReplayDivergence(f"{event_source} has an unknown service kind")

        if require_metadata and metadata.get("rng_state_after") != _jsonable(rng.getstate()):
            raise ReplayDivergence(f"{event_source} RNG metadata diverges")
    if require_metadata:
        for counter in ("model_calls", "random_calls"):
            delta = max_seen[counter] - boundary[counter]
            if not required[counter] <= delta <= possible[counter]:
                raise ReplayDivergence(
                    f"{source} service meter disagrees with its attempt count"
                )
        meter = max_seen
    return meter, usage


def _validate_service_state(
    state: Any,
    tape: list[dict[str, Any]],
    *,
    source: str,
    root_seed: int,
    branch_id: str,
    committed_events: tuple[ServiceEvent, ...] | None = None,
    usage_events: tuple[ServiceEvent, ...] | None = None,
    require_metadata: bool = False,
) -> None:
    if not isinstance(state, dict):
        raise ReplayDivergence(f"{source} service state is malformed")
    meter = state.get("meter")
    if meter is None and not require_metadata:
        meter = {"model_calls": 0, "random_calls": 0}
    actual_meter = _parse_service_meter(meter, source=source)

    if committed_events is not None:
        expected_meter = _journal_service_meter(
            committed_events,
            source=f"{source} journal",
            require_metadata=require_metadata,
        )
        if actual_meter != expected_meter:
            raise ReplayDivergence(f"{source} service meter disagrees with its journal")

        # An abandoned call's tokens land in the actor's own usage or not,
        # depending on which backend spent them: a direct provider backend
        # accumulates every attempt it made, while an agent CLI backend counts
        # only turns that completed. Both are exact, so the actor's usage must
        # equal one of the two totals -- never a value in between, and never
        # anything else.
        def _sum(events: tuple[ServiceEvent, ...]) -> dict[str, int | float]:
            total: dict[str, int | float] = {}
            for event in events:
                # Proxy services such as ``judge.evaluate`` can carry the
                # latest actor usage snapshot for auditability, but they do
                # not consume another model turn.  Counting that snapshot
                # here would double-charge the actor and make an otherwise
                # valid promotion checkpoint impossible to restore.
                if not _is_model_service(event):
                    continue
                for key, value in ((event.metadata or {}).get("usage") or {}).items():
                    total[key] = total.get(key, 0) + value
            return total

        actual_usage = state.get("model_usage") or {}
        if not isinstance(actual_usage, dict) or any(
            not isinstance(value, (int, float)) or value < 0
            for value in actual_usage.values()
        ):
            raise ReplayDivergence(f"{source} model usage is malformed")

        candidates = [_sum(committed_events)]
        if usage_events is not None and usage_events != committed_events:
            candidates.append(_sum(usage_events))
        if not any(
            all(
                math.isclose(
                    float(actual_usage.get(key, 0)),
                    float(candidate.get(key, 0)),
                    rel_tol=1e-10,
                    abs_tol=1e-10,
                )
                for key in set(actual_usage) | set(candidate)
            )
            for candidate in candidates
        ):
            raise ReplayDivergence(f"{source} model usage disagrees with its journal")

    expected_rng: Any | None = None
    rng_events: Any = (
        reversed(committed_events) if committed_events is not None else reversed(tape)
    )
    for item in rng_events:
        metadata = (
            (item.metadata or {})
            if isinstance(item, ServiceEvent)
            else item.get("metadata") or {}
        )
        if metadata.get("service_branch_id") == branch_id and "rng_state_after" in metadata:
            expected_rng = metadata["rng_state_after"]
            break
    if expected_rng is None:
        if require_metadata and tape and branch_id == "root":
            # A metadata-bound root tape must bind its latest RNG state. A
            # forked branch may legitimately inherit a tape with no event in
            # its new epoch.
            raise ReplayDivergence(f"{source} service tape has no RNG binding")
        expected_rng = _branch_rng_state(root_seed, branch_id)
    if state.get("rng_state") != expected_rng:
        raise ReplayDivergence(f"{source} RNG state disagrees with its service tape")


def _close(actual: Any, expected: float, field: str, *, tolerance: float = 1e-10) -> None:
    try:
        value = float(actual)
    except (TypeError, ValueError) as exc:
        raise ReplayDivergence(f"recorded {field} is not numeric") from exc
    if not math.isfinite(value) or not math.isclose(value, expected, rel_tol=tolerance, abs_tol=tolerance):
        raise ReplayDivergence(f"recorded {field} does not match replayed accounting")


def _replay_protocol_events_v1(
    private: tuple[dict[str, Any], ...], *, limits: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Recompute the trusted protocol state from recorded actions."""
    nodes: dict[str, dict[str, Any]] = {}
    order: list[str] = []
    continuation_counts: dict[str, int] = {}
    option_counts: dict[tuple[str, str], int] = {}
    checkout_pair_counts: dict[tuple[str, str], int] = {}
    checkout_audit: list[dict[str, Any]] = []
    current_id: str | None = None
    expected_parent: str | None = None
    pending_decision: dict[str, Any] | None = None
    submission: Submission | None = None
    k = 0.0
    decisions = 0
    checkouts = 0
    started = False
    start_event: dict[str, Any] | None = None
    terminal: dict[str, Any] | None = None
    budget_violations: list[str] = []
    budgets = limits or {}

    for position, record in enumerate(private):
        kind = record.get("kind")
        if kind == "run_started":
            if started or position != 0:
                raise ReplayDivergence("run_started is missing or out of order")
            started = True
            start_event = record
            continue
        if not started:
            raise ReplayDivergence("protocol event appears before run_started")

        if kind == "question":
            if pending_decision is not None or submission is not None:
                raise ReplayDivergence("question appears at an invalid protocol position")
            question = _decode_question(record["question"])
            IDEA_RECOVERY_V1.validate_question(question)
            question_id = record.get("question_id")
            if not isinstance(question_id, str) or not question_id or question_id in nodes:
                raise ReplayDivergence("question ID is missing or duplicated")
            parent = record.get("parent_question_id")
            if parent is not None and not isinstance(parent, str):
                raise ReplayDivergence("question parent is malformed")
            if parent != expected_parent:
                raise ReplayDivergence("question parent does not match the replayed path")
            _close(record.get("path_k"), k, "question path_k")
            created_index = len(order)
            integrity = message_hash({
                "question_id": question_id,
                "parent_question_id": parent,
                "question": question,
                "path_k": float(record["path_k"]),
                "failed_attempt_bits": float(record.get("failed_attempt_bits", 0.0)),
                "created_index": created_index,
            })
            if record.get("integrity_hash") != integrity:
                raise ReplayDivergence("question integrity hash does not match its recorded contents")
            nodes[question_id] = {
                "question": question,
                "parent": parent,
                "path_k": float(record["path_k"]),
                "failed_attempt_bits": float(record.get("failed_attempt_bits", 0.0)),
                "created_index": created_index,
                "integrity_hash": integrity,
            }
            order.append(question_id)
            max_questions = int(budgets.get("questions", 0) or 0)
            if max_questions and len(order) > max_questions:
                budget_violations.append("question limit")
            depth = 1
            ancestor = parent
            while ancestor is not None:
                depth += 1
                ancestor = nodes[ancestor]["parent"]
            max_depth = int(budgets.get("question_depth", 0) or 0)
            if max_depth and depth > max_depth:
                budget_violations.append("question-depth limit")
            current_id = question_id
            expected_parent = None
            continue

        if kind == "oracle_decision":
            if current_id is None or pending_decision is not None or submission is not None:
                raise ReplayDivergence("oracle decision appears at an invalid protocol position")
            if record.get("question_id") != current_id:
                raise ReplayDivergence("oracle decision refers to a different current question")
            decision = record.get("decision")
            if not isinstance(decision, dict):
                raise ReplayDivergence("oracle decision is malformed")
            pending_decision = decision
            decisions += 1
            max_decisions = int(budgets.get("oracle_decisions", 0) or 0)
            if max_decisions and decisions > max_decisions:
                budget_violations.append("oracle-decision limit")
            continue

        if kind == "choice_cost":
            if current_id is None or pending_decision is None or "option_id" not in pending_decision:
                raise ReplayDivergence("choice cost has no matching oracle choice")
            if record.get("question_id") != current_id:
                raise ReplayDivergence("choice cost refers to a different question")
            choice = Choice(
                pending_decision["option_id"],
                pending_decision.get("question_id"),
                pending_decision.get("public_payload"),
            )
            if choice.question_id not in (None, current_id):
                raise ReplayDivergence("oracle choice refers to a different question")
            question = nodes[current_id]["question"]
            try:
                resolved = IDEA_RECOVERY_V1.resolve_choice(choice, question)
            except ProtocolError as exc:
                raise ReplayDivergence("recorded choice violates the arena contract") from exc
            option = resolved.option
            probability = resolved.probability
            continuation_index = continuation_counts.get(current_id, 0) + 1
            continuation_counts[current_id] = continuation_index
            option_key = (current_id, option.option_id)
            option_index_count = option_counts.get(option_key, 0) + 1
            option_counts[option_key] = option_index_count
            information_bits = resolved.information_bits
            branch_bits = math.log2(continuation_index) + math.log2(option_index_count)
            k += information_bits + branch_bits
            if str(record.get("option_id")) != option.option_id:
                raise ReplayDivergence("choice-cost option does not match the oracle choice")
            if str(record.get("probability")) != str(probability):
                raise ReplayDivergence("recorded normalized choice probability diverges")
            _close(record.get("information_bits"), information_bits, "choice information bits")
            _close(record.get("branch_bits"), branch_bits, "choice branch bits")
            _close(record.get("path_k"), k, "choice path_k")
            max_bits = float(budgets.get("information_bits", 0.0) or 0.0)
            if max_bits and k > max_bits:
                budget_violations.append("information-cost limit")
            expected_parent = current_id
            pending_decision = None
            continue

        if kind == "checkout":
            if current_id is None or pending_decision is None or "option_id" in pending_decision:
                raise ReplayDivergence("checkout has no matching oracle checkout")
            if not bool((start_event or {}).get("time_travel")):
                raise ReplayDivergence("checkout appears in a run with time travel disabled")
            source_id = record.get("source_question_id")
            target_id = record.get("target_question_id")
            decision_target = pending_decision.get("question_id")
            if not all(isinstance(value, str) for value in (source_id, target_id, decision_target)):
                raise ReplayDivergence("checkout source or target is malformed")
            if source_id != current_id or decision_target != target_id:
                raise ReplayDivergence("checkout source or target disagrees with the oracle decision")
            source_index = nodes[source_id]["created_index"]
            eligible = [question_id for question_id in order if nodes[question_id]["created_index"] < source_index]
            if target_id not in eligible:
                raise ReplayDivergence("checkout target was not eligible")
            max_targets = int(budgets.get("checkout_targets", 0) or 0)
            if max_targets and len(eligible) > max_targets:
                budget_violations.append("checkout-target limit")
            max_rewind = int(budgets.get("checkout_rewind", 0) or 0)
            if max_rewind and source_index - nodes[target_id]["created_index"] > max_rewind:
                budget_violations.append("checkout-rewind limit")
            pair = (source_id, target_id)
            pair_index = checkout_pair_counts.get(pair, 0) + 1
            checkout_pair_counts[pair] = pair_index
            branch_bits = math.log2(len(eligible)) + math.log2(pair_index)
            k = nodes[target_id]["path_k"] + branch_bits
            _close(record.get("branch_bits"), branch_bits, "checkout branch bits")
            _close(record.get("path_k"), k, "checkout path_k")
            max_bits = float(budgets.get("information_bits", 0.0) or 0.0)
            if max_bits and k > max_bits:
                budget_violations.append("information-cost limit")
            checkout_audit.append({
                "source_question_id": source_id,
                "target_question_id": target_id,
                "target_count": len(eligible),
                "pair_index": pair_index,
                "branch_bits": branch_bits,
            })
            current_id = target_id
            expected_parent = None
            pending_decision = None
            checkouts += 1
            max_checkouts = int(budgets.get("checkouts", 0) or 0)
            if max_checkouts and checkouts > max_checkouts:
                budget_violations.append("checkout limit")
            continue

        if kind == "submission":
            if pending_decision is not None or submission is not None:
                raise ReplayDivergence("submission appears at an invalid protocol position")
            _close(record.get("path_k"), k, "submission path_k")
            submission = _decode_submission(record["submission"])
            IDEA_RECOVERY_V1.validate_submission(submission)
            continue

        if kind == "stage_transition":
            if (
                record.get("role") not in {"generator", "oracle"}
                or not isinstance(record.get("branch_id"), str)
                or not isinstance(record.get("from_stage"), str)
                or not isinstance(record.get("to_stage"), str)
                or not isinstance(record.get("transition"), dict)
                or not isinstance(record.get("ready"), dict)
                or not isinstance(record.get("handoff_sha256"), str)
            ):
                raise ReplayDivergence("stage transition event is malformed")
            if record["transition"] != {
                "from_stage": record["from_stage"],
                "to_stage": record["to_stage"],
            }:
                raise ReplayDivergence("stage transition fields disagree")
            if record["ready"].get("stage") != record["to_stage"]:
                raise ReplayDivergence("stage transition acknowledgement disagrees")
            if message_hash(record["ready"].get("handoff")) != record["handoff_sha256"]:
                raise ReplayDivergence("stage transition handoff hash disagrees")
            continue

        if kind == "run_finished":
            if position != len(private) - 1 or terminal is not None:
                raise ReplayDivergence("run_finished is duplicated or not terminal")
            terminal = record
            continue

        raise ReplayDivergence(f"unknown private protocol event {kind!r}")

    if terminal is None:
        raise ReplayDivergence("private event trace has no terminal result")
    status = terminal.get("status")
    if status not in {"pass", "fail", "error"}:
        raise ReplayDivergence("terminal result has an unknown status")
    if status == "error":
        _close(terminal.get("score", 0.0), 0.0, "error score")
        result_k = None
        matched: list[str] = []
    else:
        if pending_decision is not None:
            raise ReplayDivergence("completed result has an unresolved oracle decision")
        if submission is None:
            raise ReplayDivergence("completed result has no submission")
        verdicts = terminal.get("verdicts")
        if not isinstance(verdicts, list) or len(verdicts) != len(submission.ideas):
            raise ReplayDivergence("terminal verdict set is malformed")
        if any(
            not isinstance(value, dict)
            or not isinstance(value.get("idea_id"), str)
            or type(value.get("passed")) is not bool
            for value in verdicts
        ):
            raise ReplayDivergence("terminal verdict set is malformed")
        try:
            outcome = IDEA_RECOVERY_V1.score_submission(
                submission,
                tuple(
                    IdeaVerdict(
                        value["idea_id"],
                        value["passed"],
                        str(value.get("private_reason") or ""),
                    )
                    for value in verdicts
                ),
                path_k=k,
            )
        except ProtocolError as exc:
            raise ReplayDivergence("terminal verdict IDs do not match the submission") from exc
        matched = list(outcome.matched_idea_ids)
        recorded_matched = terminal.get("matched_idea_ids")
        if not isinstance(recorded_matched, list) or recorded_matched != matched:
            raise ReplayDivergence("terminal matched idea IDs diverge")
        if matched:
            result_k = outcome.k
            if status != "pass":
                raise ReplayDivergence("passing verdicts produced a non-pass result")
            if str(terminal.get("passing_mass")) != str(outcome.passing_mass):
                raise ReplayDivergence("terminal passing mass diverges")
            _close(terminal.get("submission_bits"), outcome.submission_bits, "submission bits")
            _close(terminal.get("k"), result_k, "terminal K")
            _close(terminal.get("score"), outcome.score, "terminal score")
            max_bits = float(budgets.get("information_bits", 0.0) or 0.0)
            if max_bits and result_k > max_bits:
                budget_violations.append("information-cost limit")
        else:
            result_k = outcome.k
            if status != "fail" or float(terminal.get("score", -1.0)) != 0.0:
                raise ReplayDivergence("rejected submission produced an invalid result")
            _close(terminal.get("k"), result_k, "terminal K")

    if status != "error" and budget_violations:
        raise ReplayDivergence(
            f"completed run exceeded its recorded {budget_violations[0]}"
        )

    return {
        "run_id": None if start_event is None else start_event.get("run_id"),
        "seed": None if start_event is None else start_event.get("seed"),
        "time_travel": None if start_event is None else start_event.get("time_travel"),
        "status": status,
        "error_code": terminal.get("error_code") if status == "error" else None,
        "score": float(terminal.get("score", 0.0)),
        "K": result_k,
        "matched_idea_ids": matched,
        "questions": len(order),
        "oracle_decisions": decisions,
        "checkouts": checkouts,
        "branch_state": {
            "head": current_id,
            "order": order,
            "nodes": nodes,
            "continuation_counts": continuation_counts,
            "option_counts": option_counts,
            "checkout_pair_counts": checkout_pair_counts,
            "checkout_audit": checkout_audit,
        },
    }


def _replay_protocol_events(
    private: tuple[dict[str, Any], ...],
    *,
    limits: dict[str, Any] | None = None,
    allow_incomplete: bool = False,
) -> dict[str, Any]:
    """Recompute the submit-aware protocol from its append-only event trace.

    Runs recorded before Oracle-controlled submissions did not have an explicit
    ``submission_judged`` event. Keep their parser available for diagnostics;
    every new run uses the state machine below.
    """

    start_schema = next(
        (
            record.get("protocol_event_schema")
            for record in private
            if record.get("kind") == "run_started"
        ),
        None,
    )
    if start_schema not in (None, 2, 3, 4):
        raise ReplayDivergence("run declares an unsupported protocol event schema")
    active_path_accounting = start_schema == 4
    accounting_version = validate_accounting_version(
        private[0].get("accounting_version", LEGACY_ACCOUNTING) if private else LEGACY_ACCOUNTING
    )
    if accounting_version == CURRENT_ACCOUNTING and not active_path_accounting:
        raise ReplayDivergence("occurrence accounting requires active-path event schema 4")
    submit_aware = (
        start_schema in {2, 3, 4}
        or any(record.get("kind") == "submission_judged" for record in private)
        or any(
            record.get("kind") == "question" and "failed_attempt_bits" in record
            for record in private
        )
        or any(
            record.get("kind") == "submission" and "attempt" in record
            for record in private
        )
    )
    if not submit_aware:
        if allow_incomplete:
            raise ReplayDivergence(
                "legacy protocol prefixes cannot be used as durable v2 checkpoints"
            )
        return _replay_protocol_events_v1(private, limits=limits)

    nodes: dict[str, dict[str, Any]] = {}
    order: list[str] = []
    continuation_counts: dict[str, int] = {}
    option_counts: dict[tuple[str, str], int] = {}
    checkout_pair_counts: dict[tuple[str, str], int] = {}
    checkout_audit: list[dict[str, Any]] = []
    current_id: str | None = None
    expected_parent: str | None = None
    pending_decision: dict[str, Any] | None = None
    pending_context = "question"
    authorized_submit = False
    submit_source_id: str | None = None
    submit_option_id: str | None = None
    submit_choice_bits = 0.0
    submission: Submission | None = None
    awaiting_generator_output = True
    submission_judged = False
    submission_attempts = 0
    failed_attempt_bits = 0.0
    awaiting_recovery = False
    successful_outcome: Any | None = None
    successful_verdicts: tuple[IdeaVerdict, ...] = ()
    k = 0.0
    decisions = 0
    checkouts = 0
    started = False
    start_event: dict[str, Any] | None = None
    terminal: dict[str, Any] | None = None
    budget_violations: list[str] = []
    resource_exhausted = False
    configured_judge_repeats = 1
    budgets = limits or {}

    def limit(name: str) -> int:
        return int(budgets.get(name, 0) or 0)

    def decode_verdicts(values: Any, expected: Submission) -> tuple[IdeaVerdict, ...]:
        if not isinstance(values, list) or len(values) != len(expected.ideas):
            raise ReplayDivergence("submission verdict set is malformed")
        verdicts: list[IdeaVerdict] = []
        for value in values:
            if (
                not isinstance(value, dict)
                or not isinstance(value.get("idea_id"), str)
                or type(value.get("passed")) is not bool
            ):
                raise ReplayDivergence("submission verdict set is malformed")
            verdicts.append(
                IdeaVerdict(
                    value["idea_id"],
                    value["passed"],
                    str(value.get("private_reason") or ""),
                )
            )
        return tuple(verdicts)

    def decode_verdict_rounds(
        record: dict[str, Any],
        expected: Submission,
        aggregated: tuple[IdeaVerdict, ...],
    ) -> tuple[tuple[IdeaVerdict, ...], ...]:
        values = record.get("verdict_rounds")
        if values is None:
            return (aggregated,)
        if not isinstance(values, list) or not values:
            raise ReplayDivergence("repeat Judge verdict rounds are malformed")
        return tuple(decode_verdicts(value, expected) for value in values)

    for position, record in enumerate(private):
        kind = record.get("kind")
        if resource_exhausted and kind != "run_finished":
            raise ReplayDivergence(
                "protocol continued after its information-cost limit was exhausted"
            )
        if kind == "run_started":
            if started or position != 0:
                raise ReplayDivergence("run_started is missing or out of order")
            started = True
            start_event = record
            configured_judge_repeats = int(record.get("judge_repeats") or 1)
            if not 1 <= configured_judge_repeats <= 64:
                raise ReplayDivergence("run has an invalid Judge repeat count")
            continue
        if not started:
            raise ReplayDivergence("protocol event appears before run_started")

        if kind == "accounting_updated":
            if (not active_path_accounting or accounting_version != LEGACY_ACCOUNTING
                    or successful_outcome is not None
                    or record.get("from_version") != LEGACY_ACCOUNTING
                    or record.get("accounting_version") != CURRENT_ACCOUNTING):
                raise ReplayDivergence("invalid accounting upgrade event")
            update = reprice_events(private[:position])
            _close(record.get("previous_path_k"), k, "accounting previous K")
            _close(record.get("path_k"), update["path_k"], "accounting updated K")
            if record.get("node_costs_sha256") != message_hash(update["node_costs"]):
                raise ReplayDivergence("accounting upgrade node costs diverge")
            for key, node in nodes.items():
                node["path_k"] = update["node_costs"][key]
                node["integrity_hash"] = message_hash({
                    "question_id": key, "parent_question_id": node["parent"],
                    "question": node["question"], "path_k": node["path_k"],
                    "created_index": node["created_index"],
                })
            k = update["path_k"]
            submit_choice_bits = update["submission_choice_bits"] if submit_source_id else 0.0
            accounting_version = CURRENT_ACCOUNTING
            if float(budgets.get("information_bits", 0.0) or 0.0) and k > float(budgets["information_bits"]):
                resource_exhausted = True
            continue

        if kind == "question":
            if (
                not awaiting_generator_output
                or pending_decision is not None
                or submission is not None
                or awaiting_recovery
            ):
                raise ReplayDivergence("question appears at an invalid protocol position")
            if authorized_submit:
                raise ReplayDivergence("SubmitOption produced a Question instead of a Submission")
            question = _decode_question(record["question"])
            IDEA_RECOVERY_V1.validate_question(question)
            question_id = record.get("question_id")
            if not isinstance(question_id, str) or not question_id or question_id in nodes:
                raise ReplayDivergence("question ID is missing or duplicated")
            parent = record.get("parent_question_id")
            if parent is not None and not isinstance(parent, str):
                raise ReplayDivergence("question parent is malformed")
            if parent != expected_parent:
                raise ReplayDivergence("question parent does not match the replayed path")
            _close(record.get("path_k"), k, "question path_k")
            if active_path_accounting and "failed_attempt_bits" in record:
                raise ReplayDivergence(
                    "active-path question contains legacy failed-attempt debt"
                )
            if not active_path_accounting:
                _close(
                    record.get("failed_attempt_bits", 0.0),
                    failed_attempt_bits,
                    "question failed-attempt bits",
                )
            created_index = len(order)
            integrity_payload = {
                "question_id": question_id,
                "parent_question_id": parent,
                "question": question,
                "path_k": float(record["path_k"]),
                "created_index": created_index,
            }
            if not active_path_accounting:
                integrity_payload["failed_attempt_bits"] = failed_attempt_bits
            integrity = message_hash(integrity_payload)
            if record.get("integrity_hash") != integrity:
                raise ReplayDivergence("question integrity hash does not match its recorded contents")
            node = {
                "question": question,
                "parent": parent,
                "path_k": float(record["path_k"]),
                "created_index": created_index,
                "integrity_hash": integrity,
            }
            if not active_path_accounting:
                node["failed_attempt_bits"] = failed_attempt_bits
            nodes[question_id] = node
            order.append(question_id)
            if limit("questions") and len(order) > limit("questions"):
                raise ReplayDivergence("question event exceeds the recorded question limit")
            depth = 1
            ancestor = parent
            while ancestor is not None:
                depth += 1
                ancestor = nodes[ancestor]["parent"]
            if limit("question_depth") and depth > limit("question_depth"):
                raise ReplayDivergence(
                    "question event exceeds the recorded question-depth limit"
                )
            current_id = question_id
            expected_parent = None
            authorized_submit = False
            awaiting_generator_output = False
            continue

        if kind == "judge_previewed":
            # Historical advisory previews are audit-only. Current execution
            # has no such event or participant message.
            if (accounting_version != LEGACY_ACCOUNTING or current_id is None
                    or record.get("question_id") != current_id
                    or awaiting_generator_output or pending_decision is not None
                    or submission is not None or awaiting_recovery):
                raise ReplayDivergence("historical judge preview appears at an invalid position")
            _close(record.get("path_k"), k, "historical judge preview K")
            preview_submission = _decode_submission(record["submission"])
            preview_verdicts = decode_verdicts(record.get("verdicts"), preview_submission)
            outcome = IDEA_RECOVERY_V1.score_submission(preview_submission, preview_verdicts, path_k=k)
            if record.get("status") != outcome.status:
                raise ReplayDivergence("historical judge preview status disagrees with verdicts")
            continue

        if kind == "oracle_decision":
            if pending_decision is not None or awaiting_generator_output:
                raise ReplayDivergence("oracle decision is duplicated")
            if "context" not in record:
                context = "question"
            elif record.get("context") == "submission_feedback":
                context = "submission_feedback"
            else:
                raise ReplayDivergence("oracle decision has an invalid context")
            if context == "submission_feedback":
                if not awaiting_recovery or submission is None or submit_source_id is None:
                    raise ReplayDivergence("submission recovery decision is out of order")
                if record.get("question_id") != submit_source_id:
                    raise ReplayDivergence("submission recovery refers to another source")
            else:
                if awaiting_recovery or submission is not None or current_id is None:
                    raise ReplayDivergence("oracle decision appears at an invalid protocol position")
                if record.get("question_id") != current_id:
                    raise ReplayDivergence("oracle decision refers to a different current question")
            decision = record.get("decision")
            if not isinstance(decision, dict):
                raise ReplayDivergence("oracle decision is malformed")
            pending_decision = decision
            pending_context = context
            decisions += 1
            if limit("oracle_decisions") and decisions > limit("oracle_decisions"):
                raise ReplayDivergence(
                    "oracle event exceeds the recorded decision limit"
                )
            continue

        if kind == "choice_cost":
            if (
                current_id is None
                or pending_decision is None
                or pending_context != "question"
                or "option_id" not in pending_decision
                or submission is not None
                or awaiting_generator_output
            ):
                raise ReplayDivergence("choice cost has no matching oracle choice")
            if record.get("question_id") != current_id:
                raise ReplayDivergence("choice cost refers to a different question")
            choice = Choice(
                pending_decision["option_id"],
                pending_decision.get("question_id"),
                pending_decision.get("public_payload"),
            )
            if choice.question_id not in (None, current_id):
                raise ReplayDivergence("oracle choice refers to a different question")
            try:
                resolved = IDEA_RECOVERY_V1.resolve_choice(choice, nodes[current_id]["question"])
            except ProtocolError as exc:
                raise ReplayDivergence("recorded choice violates the arena contract") from exc
            option = resolved.option
            continuation_index = continuation_counts.get(current_id, 0) + 1
            continuation_counts[current_id] = continuation_index
            option_key = (current_id, option.option_id)
            option_index = option_counts.get(option_key, 0) + 1
            option_counts[option_key] = option_index
            branch_bits = choice_surcharge(continuation_index, option_index, version=accounting_version)
            if accounting_version == CURRENT_ACCOUNTING:
                if (record.get("accounting_version") != CURRENT_ACCOUNTING
                        or type(record.get("continuation_index")) is not int
                        or record["continuation_index"] != continuation_index
                        or type(record.get("option_index")) is not int
                        or record["option_index"] != option_index):
                    raise ReplayDivergence("choice accounting version or counters diverge")
            choice_bits = resolved.information_bits
            total_choice_bits = choice_bits + branch_bits
            k += total_choice_bits
            if str(record.get("option_id")) != option.option_id:
                raise ReplayDivergence("choice-cost option does not match the oracle choice")
            if str(record.get("probability")) != str(resolved.probability):
                raise ReplayDivergence("recorded normalized choice probability diverges")
            _close(record.get("information_bits"), choice_bits, "choice information bits")
            _close(record.get("branch_bits"), branch_bits, "choice branch bits")
            _close(record.get("path_k"), k, "choice path_k")
            if float(budgets.get("information_bits", 0.0) or 0.0) and k > float(
                budgets["information_bits"]
            ):
                resource_exhausted = True
            authorized_submit = isinstance(option, SubmitOption)
            submit_source_id = current_id if authorized_submit else None
            submit_option_id = option.option_id if authorized_submit else None
            submit_choice_bits = total_choice_bits if authorized_submit else 0.0
            expected_parent = current_id
            pending_decision = None
            pending_context = "question"
            awaiting_generator_output = True
            continue

        if kind == "submission":
            if (
                pending_decision is not None
                or submission is not None
                or not authorized_submit
                or submit_source_id is None
                or not awaiting_generator_output
            ):
                raise ReplayDivergence("submission was not authorized by a SubmitOption")
            if record.get("source_question_id") != submit_source_id:
                raise ReplayDivergence("submission source does not match its SubmitOption")
            if record.get("option_id") != submit_option_id:
                raise ReplayDivergence("submission option does not match its SubmitOption")
            _close(record.get("path_k"), k, "submission path_k")
            if active_path_accounting and "failed_attempt_bits" in record:
                raise ReplayDivergence(
                    "active-path submission contains legacy failed-attempt debt"
                )
            if not active_path_accounting:
                _close(
                    record.get("failed_attempt_bits", 0.0),
                    failed_attempt_bits,
                    "submission failed-attempt bits",
                )
            submission = _decode_submission(record["submission"])
            IDEA_RECOVERY_V1.validate_submission(submission)
            awaiting_generator_output = False
            submission_judged = False
            submission_attempts += 1
            if int(record.get("attempt", -1)) != submission_attempts:
                raise ReplayDivergence("submission attempt index diverges")
            if limit("submission_attempts") and submission_attempts > limit("submission_attempts"):
                raise ReplayDivergence(
                    "submission event exceeds the recorded attempt limit"
                )
            continue

        if kind == "submission_judged":
            if (
                submission is None
                or not authorized_submit
                or awaiting_recovery
                or awaiting_generator_output
                or submission_judged
            ):
                raise ReplayDivergence("submission judgment appears out of order")
            if int(record.get("attempt", -1)) != submission_attempts:
                raise ReplayDivergence("submission judgment attempt index diverges")
            if record.get("source_question_id") != submit_source_id:
                raise ReplayDivergence("submission judgment source diverges")
            verdicts = decode_verdicts(record.get("verdicts"), submission)
            verdict_rounds = decode_verdict_rounds(record, submission, verdicts)
            try:
                outcome, expected_verdicts = IDEA_RECOVERY_V1.score_repeated_submission(
                    submission,
                    verdict_rounds,
                    path_k=k,
                )
            except ProtocolError as exc:
                raise ReplayDivergence("submission judgment violates the contract") from exc
            if verdicts != expected_verdicts:
                raise ReplayDivergence("repeat Judge aggregate verdicts diverge")
            if record.get("status") != outcome.status:
                raise ReplayDivergence("submission judgment status diverges")
            if list(record.get("matched_idea_ids") or []) != list(outcome.matched_idea_ids):
                raise ReplayDivergence("submission judgment matched IDs diverge")
            if str(record.get("passing_mass")) != str(outcome.passing_mass):
                raise ReplayDivergence("submission judgment passing mass diverges")
            if outcome.submission_bits is None:
                if record.get("submission_bits") is not None:
                    raise ReplayDivergence("rejected submission has pointer bits")
            else:
                _close(
                    record.get("submission_bits"),
                    outcome.submission_bits,
                    "submission pointer bits",
                )
            repeat_fields = {
                "judge_repeats": outcome.judge_repeats,
                "judge_passes": outcome.judge_passes,
                "judge_pass_rate": str(outcome.judge_pass_rate),
                "repeat_bits": outcome.repeat_bits,
            }
            for field, expected_value in repeat_fields.items():
                if field in record:
                    if field == "repeat_bits" and expected_value is not None:
                        _close(record[field], expected_value, "repeat Judge bits")
                    elif record[field] != expected_value:
                        raise ReplayDivergence(f"submission judgment {field} diverges")
            if active_path_accounting:
                if "failed_attempt_bits_after" in record:
                    raise ReplayDivergence(
                        "active-path judgment contains legacy failed-attempt debt"
                    )
            else:
                debt_after = failed_attempt_bits + (
                    submit_choice_bits if outcome.status == "fail" else 0.0
                )
                _close(
                    record.get("failed_attempt_bits_after", debt_after),
                    debt_after,
                    "post-judgment failed-attempt bits",
                )
            if outcome.status == "pass":
                successful_outcome = outcome
                successful_verdicts = verdicts
                if (
                    float(budgets.get("information_bits", 0.0) or 0.0)
                    and outcome.k > float(budgets["information_bits"])
                ):
                    resource_exhausted = True
            else:
                if not active_path_accounting:
                    failed_attempt_bits = debt_after
                awaiting_recovery = True
            submission_judged = True
            continue

        if kind == "checkout":
            if (
                current_id is None
                or pending_decision is None
                or "option_id" in pending_decision
                or awaiting_generator_output
            ):
                raise ReplayDivergence("checkout has no matching oracle checkout")
            source_id = record.get("source_question_id")
            target_id = record.get("target_question_id")
            decision_target = pending_decision.get("question_id")
            if not all(isinstance(value, str) for value in (source_id, target_id, decision_target)):
                raise ReplayDivergence("checkout source or target is malformed")
            recovery = pending_context == "submission_feedback"
            if recovery:
                if record.get("context") != "submission_recovery":
                    raise ReplayDivergence(
                        "submission recovery checkout has an invalid context"
                    )
            elif "context" in record:
                raise ReplayDivergence("ordinary checkout has an invalid context")
            expected_source = submit_source_id if recovery else current_id
            if source_id != expected_source or decision_target != target_id:
                raise ReplayDivergence("checkout source or target disagrees with the oracle decision")
            source_index = nodes[source_id]["created_index"]
            time_travel = bool((start_event or {}).get("time_travel"))
            if recovery and not time_travel:
                eligible = [source_id]
            else:
                eligible = [
                    question_id
                    for question_id in order
                    if nodes[question_id]["created_index"] < source_index
                    or (recovery and question_id == source_id)
                ]
            if target_id not in eligible:
                raise ReplayDivergence("checkout target was not eligible")
            if not recovery and not time_travel:
                raise ReplayDivergence("checkout appears in a run with time travel disabled")
            if recovery and target_id != source_id and not time_travel:
                raise ReplayDivergence("recovery rewound history with time travel disabled")
            if limit("checkout_targets") and len(eligible) > limit("checkout_targets"):
                raise ReplayDivergence(
                    "checkout event exceeds the recorded target limit"
                )
            if limit("checkout_rewind") and source_index - nodes[target_id]["created_index"] > limit(
                "checkout_rewind"
            ):
                raise ReplayDivergence(
                    "checkout event exceeds the recorded rewind limit"
                )
            pair = (source_id, target_id)
            pair_index = checkout_pair_counts.get(pair, 0) + 1
            checkout_pair_counts[pair] = pair_index
            branch_bits = 0.0
            if active_path_accounting:
                if "failed_attempt_bits" in record:
                    raise ReplayDivergence(
                        "active-path checkout contains legacy failed-attempt debt"
                    )
                k = nodes[target_id]["path_k"] + branch_bits
            else:
                k = (
                    nodes[target_id]["path_k"]
                    + failed_attempt_bits
                    - nodes[target_id]["failed_attempt_bits"]
                )
            _close(record.get("branch_bits"), branch_bits, "checkout branch bits")
            if not active_path_accounting:
                _close(record.get("failed_attempt_bits"), failed_attempt_bits, "checkout debt")
            _close(record.get("path_k"), k, "checkout path_k")
            checkout_audit.append({
                "source_question_id": source_id,
                "target_question_id": target_id,
                "target_count": len(eligible),
                "pair_index": pair_index,
                "branch_bits": branch_bits,
            })
            current_id = target_id
            expected_parent = None
            pending_decision = None
            pending_context = "question"
            checkouts += 1
            if limit("checkouts") and checkouts > limit("checkouts"):
                raise ReplayDivergence("checkout event exceeds the recorded checkout limit")
            if recovery:
                submission = None
                awaiting_recovery = False
                authorized_submit = False
                submit_source_id = None
                submit_option_id = None
                submit_choice_bits = 0.0
                successful_outcome = None
                successful_verdicts = ()
                submission_judged = False
            continue

        if kind == "stage_transition":
            if (
                record.get("role") not in {"generator", "oracle"}
                or not isinstance(record.get("branch_id"), str)
                or not isinstance(record.get("from_stage"), str)
                or not isinstance(record.get("to_stage"), str)
                or not isinstance(record.get("transition"), dict)
                or not isinstance(record.get("ready"), dict)
                or not isinstance(record.get("handoff_sha256"), str)
            ):
                raise ReplayDivergence("stage transition event is malformed")
            if record["transition"] != {
                "from_stage": record["from_stage"],
                "to_stage": record["to_stage"],
            }:
                raise ReplayDivergence("stage transition fields disagree")
            if record["ready"].get("stage") != record["to_stage"]:
                raise ReplayDivergence("stage transition acknowledgement disagrees")
            if message_hash(record["ready"].get("handoff")) != record["handoff_sha256"]:
                raise ReplayDivergence("stage transition handoff hash disagrees")
            continue

        if kind == "run_finished":
            if position != len(private) - 1 or terminal is not None:
                raise ReplayDivergence("run_finished is duplicated or not terminal")
            if active_path_accounting and "failed_attempt_bits" in record:
                raise ReplayDivergence(
                    "active-path result contains legacy failed-attempt debt"
                )
            terminal = record
            continue

        raise ReplayDivergence(f"unknown private protocol event {kind!r}")

    if terminal is None:
        if not allow_incomplete:
            raise ReplayDivergence("private event trace has no terminal result")
        status = "running"
        result_k = None
        matched: list[str] = []
    else:
        status = terminal.get("status")
        if status not in {"pass", "error"}:
            raise ReplayDivergence("submit-aware terminal result has an unknown status")
        if resource_exhausted and (
            status != "error" or terminal.get("error_code") != "resource_limit_exceeded"
        ):
            raise ReplayDivergence(
                "information-cost exhaustion did not terminate with a resource error"
            )
        if status == "error":
            _close(terminal.get("score", 0.0), 0.0, "error score")
            result_k = None
            matched = []
        else:
            if successful_outcome is None or submission is None or awaiting_recovery:
                raise ReplayDivergence("passing run has no successful judged submission")
            result_k = successful_outcome.k
            matched = list(successful_outcome.matched_idea_ids)
            if list(terminal.get("matched_idea_ids") or []) != matched:
                raise ReplayDivergence("terminal matched idea IDs diverge")
            terminal_verdicts = decode_verdicts(terminal.get("verdicts"), submission)
            if terminal_verdicts != successful_verdicts:
                raise ReplayDivergence("terminal verdicts differ from the judged submission")
            if str(terminal.get("passing_mass")) != str(successful_outcome.passing_mass):
                raise ReplayDivergence("terminal passing mass diverges")
            _close(
                terminal.get("submission_bits"),
                successful_outcome.submission_bits,
                "terminal submission bits",
            )
            terminal_repeat_fields = {
                "judge_repeats": successful_outcome.judge_repeats,
                "judge_passes": successful_outcome.judge_passes,
                "judge_pass_rate": float(successful_outcome.judge_pass_rate),
                "repeat_bits": successful_outcome.repeat_bits,
            }
            for field, expected_value in terminal_repeat_fields.items():
                if field not in terminal:
                    continue
                if field in {"judge_pass_rate", "repeat_bits"}:
                    _close(terminal[field], expected_value, f"terminal {field}")
                elif terminal[field] != expected_value:
                    raise ReplayDivergence(f"terminal {field} diverges")
            _close(terminal.get("k"), result_k, "terminal K")
            _close(terminal.get("score"), successful_outcome.score, "terminal score")
            if int(terminal.get("submission_attempts", -1)) != submission_attempts:
                raise ReplayDivergence("terminal submission-attempt count diverges")
            if not active_path_accounting:
                _close(
                    terminal.get("failed_attempt_bits"),
                    failed_attempt_bits,
                    "terminal failed-attempt bits",
                )

    if status != "error" and budget_violations:
        raise ReplayDivergence(f"completed run exceeded its recorded {budget_violations[0]}")

    result = {
        "run_id": None if start_event is None else start_event.get("run_id"),
        "accounting_version": accounting_version,
        "seed": None if start_event is None else start_event.get("seed"),
        "time_travel": None if start_event is None else start_event.get("time_travel"),
        "status": status,
        "error_code": (
            terminal.get("error_code")
            if terminal is not None and status == "error"
            else None
        ),
        "score": float(terminal.get("score", 0.0)) if terminal is not None else 0.0,
        "K": result_k,
        "matched_idea_ids": matched,
        "questions": len(order),
        "oracle_decisions": decisions,
        "checkouts": checkouts,
        "submission_attempts": submission_attempts,
        "judge_repeats": (
            successful_outcome.judge_repeats if successful_outcome is not None else None
        ),
        "judge_passes": (
            successful_outcome.judge_passes if successful_outcome is not None else None
        ),
        "judge_pass_rate": (
            float(successful_outcome.judge_pass_rate)
            if successful_outcome is not None
            else None
        ),
        "repeat_bits": (
            successful_outcome.repeat_bits if successful_outcome is not None else None
        ),
        "branch_state": {
            "head": current_id,
            "order": order,
            "nodes": nodes,
            "continuation_counts": continuation_counts,
            "option_counts": option_counts,
            "checkout_pair_counts": checkout_pair_counts,
            "checkout_audit": checkout_audit,
        },
    }
    if not active_path_accounting:
        result["failed_attempt_bits"] = failed_attempt_bits
    if allow_incomplete:
        result["resume_protocol_state"] = {
            "path_k": k,
            "current_question_id": current_id,
            "expected_parent_question_id": expected_parent,
            "pending_decision": pending_decision,
            "pending_context": pending_context,
            "awaiting_generator_output": awaiting_generator_output,
            "authorized_submit": authorized_submit,
            "submission_present": submission is not None,
            "submission_judged": submission_judged,
            "awaiting_recovery": awaiting_recovery,
            "submission_source_question_id": submit_source_id,
            "submission_option_id": submit_option_id,
            "submission_choice_bits": submit_choice_bits,
        }
    return result


def protocol_replay(run_directory: str | Path) -> dict[str, Any]:
    root = Path(run_directory).resolve()
    try:
        manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ReplayDivergence("manifest file is missing or invalid") from exc
    public = verify_hash_chain(root / "events.public.jsonl")
    private = verify_hash_chain(root / "events.private.jsonl")
    private_payloads = [_event_payload(record) for record in private]
    event_schema = next(
        (
            value.get("protocol_event_schema")
            for value in private_payloads
            if value.get("kind") == "run_started"
        ),
        None,
    )
    expected_public = (
        _redacted_public_events(private_payloads, manifest.get("disclosure"))
        if event_schema in {3, 4}
        else _legacy_public_events(private_payloads, manifest.get("disclosure"))
    )
    actual_public = tuple(_event_payload(record) for record in public)
    if actual_public != expected_public:
        raise ReplayDivergence("public event trace is not the declared redaction of the private trace")
    try:
        replayed = _replay_protocol_events(private, limits=manifest.get("budgets"))
    except ReplayDivergence:
        raise
    except (ArithmeticError, KeyError, TypeError, ValueError, ValidationError) as exc:
        raise ReplayDivergence("private protocol event trace is malformed") from exc
    if replayed["run_id"] != manifest.get("run_id"):
        raise ReplayDivergence("manifest run ID disagrees with the private trace")
    if replayed["seed"] != manifest.get("seed"):
        raise ReplayDivergence("manifest seed disagrees with the private trace")
    try:
        score = json.loads((root / "score.json").read_text(encoding="utf-8"))
        branches = json.loads((root / "branches.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ReplayDivergence("score or branch file is missing or invalid") from exc
    current_score = score
    if "repriced_from" in current_score:
        try:
            score = json.loads((root / "score.recorded.json").read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ReplayDivergence("repriced score has no recorded-score archive") from exc
    if score.get("accounting_version", LEGACY_ACCOUNTING) != replayed.get("accounting_version", LEGACY_ACCOUNTING):
        raise ReplayDivergence("recorded score accounting version diverges")
    for field in (
        "status",
        "error_code",
        "matched_idea_ids",
        "questions",
        "oracle_decisions",
        "checkouts",
        "submission_attempts",
        "judge_repeats",
        "judge_passes",
    ):
        if field in score and score[field] != replayed[field]:
            raise ReplayDivergence(f"score file disagrees on {field}")
    for field in ("judge_pass_rate", "repeat_bits"):
        if field in score and replayed[field] is not None:
            _close(score[field], replayed[field], f"score-file {field}")
    if event_schema == 4 and "failed_attempt_bits" in score:
        raise ReplayDivergence(
            "active-path score file contains legacy failed-attempt debt"
        )
    _close(score.get("score"), replayed["score"], "score-file score")
    if replayed["K"] is not None:
        _close(score.get("K"), float(replayed["K"]), "score-file K")
        if event_schema != 4:
            _close(
                score.get("failed_attempt_bits", 0.0),
                float(replayed.get("failed_attempt_bits", 0.0)),
                "score-file failed-attempt bits",
            )
    branch_state = replayed["branch_state"]
    if branches.get("accounting_version", LEGACY_ACCOUNTING) != replayed.get("accounting_version", LEGACY_ACCOUNTING):
        raise ReplayDivergence("branch accounting version disagrees with its event prefix")
    if branches.get("head") != branch_state["head"]:
        raise ReplayDivergence("branch head disagrees with protocol replay")
    if branches.get("order") != branch_state["order"]:
        raise ReplayDivergence("branch order disagrees with replayed questions")
    replayed_nodes = {}
    for question_id, node in branch_state["nodes"].items():
        replayed_node = {
            "question_id": question_id,
            "parent_question_id": node["parent"],
            "path_k": node["path_k"],
            "created_index": node["created_index"],
            "integrity_hash": node["integrity_hash"],
            "question_hash": message_hash(node["question"]),
        }
        if event_schema != 4:
            replayed_node["failed_attempt_bits"] = node.get("failed_attempt_bits", 0.0)
        replayed_nodes[question_id] = replayed_node
    if branches.get("nodes") != replayed_nodes:
        raise ReplayDivergence("stored branch nodes disagree with protocol replay")
    if branches.get("continuation_counts", {}) != branch_state["continuation_counts"]:
        raise ReplayDivergence("continuation counters disagree with protocol replay")
    encoded_options = {f"{key[0]}\0{key[1]}": value for key, value in branch_state["option_counts"].items()}
    if branches.get("option_counts", {}) != encoded_options:
        raise ReplayDivergence("option counters disagree with protocol replay")
    encoded_pairs = {
        f"{key[0]}\0{key[1]}": value for key, value in branch_state["checkout_pair_counts"].items()
    }
    if branches.get("checkout_pair_counts", {}) != encoded_pairs:
        raise ReplayDivergence("checkout counters disagree with protocol replay")
    if branches.get("checkout_audit", []) != branch_state["checkout_audit"]:
        raise ReplayDivergence("checkout audit disagrees with protocol replay")
    recorded_result = {key: value for key, value in replayed.items() if key != "branch_state"}
    result = dict(recorded_result)
    if "repriced_from" in current_score:
        from .accounting import repriced_score
        expected = repriced_score(score, recorded_result, private)
        if current_score != expected:
            raise ReplayDivergence("repriced score disagrees with current accounting or provenance")
        result.update({key: current_score[key] for key in ("K", "score", "accounting_version")})
    return {
        "status": "replayed",
        "run_id": manifest["run_id"],
        "public_events": len(public),
        "private_events": len(private),
        "result": result,
        "recorded_result": recorded_result,
    }


def resume_event_prefix(
    run_directory: str | Path,
    *,
    checkpoint_name: str = "checkpoint.private.json",
) -> tuple[dict[str, Any], ...]:
    root = Path(run_directory).resolve()
    try:
        checkpoint = json.loads((root / checkpoint_name).read_text(encoding="utf-8"))
        count = int(checkpoint["private_event_count"])
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ReplayDivergence("run has no valid durable checkpoint") from exc
    if count < 1:
        raise ReplayDivergence("durable checkpoint has an invalid event cursor")
    records = verify_hash_chain(root / "events.private.jsonl", max_records=count)
    return tuple(_event_payload(record) for record in records[:count])


def _validate_resume_checkpoint(
    checkpoint: dict[str, Any],
    manifest: dict[str, Any],
    prefix: tuple[dict[str, Any], ...],
    service_prefix: tuple[dict[str, Any], ...] | None = None,
    *,
    require_active_path: bool = False,
) -> None:
    """Bind a private durable checkpoint to its committed protocol prefix."""

    event_schema = next(
        (
            item.get("protocol_event_schema")
            for item in prefix
            if item.get("kind") == "run_started"
        ),
        None,
    )
    if require_active_path and event_schema != 4:
        raise ReplayDivergence(
            "resume requires active-path protocol event schema 4; start a new run"
        )
    active_path_accounting = event_schema == 4
    replayed = _replay_protocol_events(
        prefix,
        limits=manifest.get("budgets"),
        allow_incomplete=True,
    )
    if replayed["status"] != "running":
        raise ReplayDivergence("durable checkpoint includes a terminal run event")
    if replayed["run_id"] != manifest.get("run_id"):
        raise ReplayDivergence("durable checkpoint event prefix has a different run ID")
    if replayed["seed"] != manifest.get("seed"):
        raise ReplayDivergence("durable checkpoint event prefix has a different seed")
    if int(checkpoint.get("private_event_count", -1)) != len(prefix):
        raise ReplayDivergence("durable checkpoint private event cursor diverges")
    expected_public_count = len(
        _redacted_public_events(list(prefix), manifest.get("disclosure"))
        if event_schema in {3, 4}
        else _legacy_public_events(list(prefix), manifest.get("disclosure"))
    )
    if int(checkpoint.get("public_event_count", -1)) != expected_public_count:
        raise ReplayDivergence("durable checkpoint public event cursor diverges")
    if service_prefix is not None and int(
        checkpoint.get("service_event_count", -1)
    ) != len(service_prefix):
        raise ReplayDivergence("durable checkpoint service event cursor diverges")

    engine = checkpoint.get("engine")
    branches = checkpoint.get("branches")
    if not isinstance(engine, dict) or not isinstance(branches, dict):
        raise ReplayDivergence("durable checkpoint engine or branch state is malformed")
    expected_checkpoint_schema = 3 if active_path_accounting else 2
    if checkpoint.get("schema_version") != expected_checkpoint_schema:
        raise ReplayDivergence("durable checkpoint declares an unsupported schema")
    protocol = replayed["resume_protocol_state"]
    if active_path_accounting:
        if "failed_attempt_bits" in engine:
            raise ReplayDivergence(
                "active-path checkpoint contains legacy failed-attempt debt"
            )
    else:
        _close(
            engine.get("failed_attempt_bits", 0.0),
            float(replayed.get("failed_attempt_bits", 0.0)),
            "checkpoint engine failed-attempt bits",
        )
    for field, expected in (("k", protocol["path_k"]),):
        _close(engine.get(field, 0.0), float(expected), f"checkpoint engine {field}")
    for field, expected in (
        ("decisions", replayed["oracle_decisions"]),
        ("checkouts", replayed["checkouts"]),
        ("submission_attempts", replayed["submission_attempts"]),
    ):
        if int(engine.get(field, 0)) != int(expected):
            raise ReplayDivergence(f"checkpoint engine {field} disagrees with its event prefix")

    branch_state = replayed["branch_state"]
    if branches.get("accounting_version", LEGACY_ACCOUNTING) != replayed.get("accounting_version", LEGACY_ACCOUNTING):
        raise ReplayDivergence("checkpoint accounting version disagrees with its event prefix")
    if branches.get("head") != branch_state["head"]:
        raise ReplayDivergence("checkpoint branch head disagrees with its event prefix")
    if branches.get("order") != branch_state["order"]:
        raise ReplayDivergence("checkpoint branch order disagrees with its event prefix")
    raw_nodes = branches.get("nodes") or {}
    if not isinstance(raw_nodes, dict) or set(raw_nodes) != set(branch_state["nodes"]):
        raise ReplayDivergence("checkpoint branch nodes disagree with its event prefix")
    for question_id, expected in branch_state["nodes"].items():
        item = raw_nodes[question_id]
        try:
            question = decode_message(item["question"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ReplayDivergence("checkpoint branch question is malformed") from exc
        if not isinstance(question, Question) or message_hash(question) != message_hash(
            expected["question"]
        ):
            raise ReplayDivergence("checkpoint branch question disagrees with its event prefix")
        if (
            item.get("question_id") != question_id
            or item.get("parent_question_id") != expected["parent"]
            or int(item.get("created_index", -1)) != expected["created_index"]
            or item.get("integrity_hash") != expected["integrity_hash"]
        ):
            raise ReplayDivergence("checkpoint branch node metadata disagrees with its event prefix")
        _close(item.get("path_k"), expected["path_k"], "checkpoint node path K")
        if active_path_accounting:
            if "failed_attempt_bits" in item:
                raise ReplayDivergence(
                    "active-path checkpoint node contains legacy failed-attempt debt"
                )
        else:
            _close(
                item.get("failed_attempt_bits", 0.0),
                expected.get("failed_attempt_bits", 0.0),
                "checkpoint node failed-attempt bits",
            )

    checkpoint_continuations = {
        str(key): int(value)
        for key, value in (branches.get("continuation_counts") or {}).items()
    }
    if checkpoint_continuations != branch_state["continuation_counts"]:
        raise ReplayDivergence("checkpoint continuation counters disagree with its event prefix")
    try:
        checkpoint_options = {
            (str(question_id), str(option_id)): int(count)
            for question_id, option_id, count in branches.get("option_counts", [])
        }
        checkpoint_pairs = {
            (str(source_id), str(target_id)): int(count)
            for source_id, target_id, count in branches.get("checkout_pair_counts", [])
        }
    except (TypeError, ValueError) as exc:
        raise ReplayDivergence("checkpoint branch counters are malformed") from exc
    if checkpoint_options != branch_state["option_counts"]:
        raise ReplayDivergence("checkpoint option counters disagree with its event prefix")
    if checkpoint_pairs != branch_state["checkout_pair_counts"]:
        raise ReplayDivergence("checkpoint checkout counters disagree with its event prefix")
    if (branches.get("checkout_audit") or []) != branch_state["checkout_audit"]:
        raise ReplayDivergence("checkpoint checkout audit disagrees with its event prefix")

    phase = str(engine.get("phase") or "")
    raw_stage_chain = engine.get("stage_transitions") or []
    try:
        checkpoint_stage_chain = tuple(
            decode_message(value) for value in raw_stage_chain
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ReplayDivergence("checkpoint stage-transition chain is malformed") from exc
    if any(not isinstance(value, StageTransition) for value in checkpoint_stage_chain):
        raise ReplayDivergence("checkpoint stage-transition chain is malformed")
    event_stage_chain: list[StageTransition] = []
    for event in prefix:
        if event.get("kind") != "stage_transition" or event.get("role") != "oracle":
            continue
        value = event.get("transition")
        if not isinstance(value, dict):
            raise ReplayDivergence("stage transition event is malformed")
        event_stage_chain.append(
            StageTransition(str(value.get("from_stage") or ""), str(value.get("to_stage") or ""))
        )
    if checkpoint_stage_chain != tuple(event_stage_chain):
        raise ReplayDivergence(
            "checkpoint stage-transition chain disagrees with its event prefix"
        )
    if any(
        prior.to_stage != following.from_stage
        for prior, following in zip(
            checkpoint_stage_chain,
            checkpoint_stage_chain[1:],
            strict=False,
        )
    ):
        raise ReplayDivergence("checkpoint stage-transition chain is discontinuous")
    last = next((event for event in reversed(prefix) if event.get("kind") != "accounting_updated"), {})
    last_kind = last.get("kind")
    phase_position_ok = {
        "actors_ready": last_kind == "run_started",
        "generator_output": last_kind in {"run_started", "choice_cost"},
        "await_oracle": last_kind in {
            "question",
            "checkout",
            "stage_transition",
        },
        "after_oracle": last_kind == "oracle_decision"
        and str(last.get("context") or "question") == "question",
        "before_judge": last_kind in {"submission", "stage_transition"},
        "await_submission_recovery": last_kind == "submission_judged"
        and last.get("status") == "fail",
        "after_submission_recovery": last_kind == "oracle_decision"
        and last.get("context") == "submission_feedback",
    }.get(phase, False)
    if last_kind == "stage_transition" and last.get("role") not in {
        "generator",
        "oracle",
    }:
        phase_position_ok = False
    if not phase_position_ok:
        raise ReplayDivergence("durable checkpoint phase disagrees with its event prefix")

    expected_current = (
        protocol["current_question_id"]
        if phase in {"await_oracle", "after_oracle"}
        else None
    )
    if engine.get("current_question_id") != expected_current:
        raise ReplayDivergence("checkpoint current question disagrees with its phase")
    if engine.get("submission_source_question_id") != protocol[
        "submission_source_question_id"
    ]:
        raise ReplayDivergence("checkpoint submission source disagrees with its event prefix")
    if engine.get("submission_option_id") != protocol["submission_option_id"]:
        raise ReplayDivergence("checkpoint submit option disagrees with its event prefix")
    _close(
        engine.get("submission_choice_bits", 0.0),
        float(protocol["submission_choice_bits"]),
        "checkpoint submit-choice bits",
    )

    wire_expectations = {
        "after_oracle": ("decision", {"choice", "checkout"}),
        "before_judge": ("submission", {"submission"}),
        "await_submission_recovery": (
            "submission_feedback",
            {"submission_feedback"},
        ),
        "after_submission_recovery": ("decision", {"checkout"}),
    }
    if phase in wire_expectations:
        key, allowed_types = wire_expectations[phase]
        value = engine.get(key)
        if not isinstance(value, dict) or value.get("type") not in allowed_types:
            raise ReplayDivergence("checkpoint phase-local message is missing or malformed")
    elif phase == "generator_output":
        # ActorRuntime records any canonical wire return before the Arena can
        # enforce the Generator's role contract.  Keep that exact return in a
        # durable phase-local checkpoint even when it is, for example, a
        # Choice rather than a Question/Submission.  The actor input, output
        # hash, call cursor, and terminal pending-call marker are checked
        # independently below and by actor_replay.
        value = engine.get("output")
        if not isinstance(value, dict):
            raise ReplayDivergence("checkpoint phase-local message is missing or malformed")
        try:
            decode_message(value)
        except (KeyError, TypeError, ValueError) as exc:
            raise ReplayDivergence(
                "checkpoint phase-local message is missing or malformed"
            ) from exc

    actor_streams = checkpoint.get("actor_streams") or {}

    def actor_last_call(role: str) -> dict[str, Any]:
        try:
            current = _expand_actor_checkpoint_data(
                checkpoint["actors"][role],
                (actor_streams.get(role) or {}),
            )
            calls = current.get("calls") or []
            for call in reversed(calls):
                message = decode_message(call["message"])
                if not isinstance(message, StageTransition):
                    return call
            raise IndexError
        except (IndexError, KeyError, TypeError, ValueError) as exc:
            raise ReplayDivergence(
                f"checkpoint {role} output cursor is malformed"
            ) from exc

    def actor_output_hash(role: str) -> str:
        return str(actor_last_call(role)["output_hash"])

    def actor_input(role: str) -> Any:
        try:
            return decode_message(actor_last_call(role)["message"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ReplayDivergence(
                f"checkpoint {role} input cursor is malformed"
            ) from exc

    def decoded_engine_message(key: str) -> Any:
        try:
            return decode_message(engine[key])
        except (KeyError, TypeError, ValueError) as exc:
            raise ReplayDivergence(
                f"checkpoint {key} message is malformed"
            ) from exc

    actor_expectations = _derive_actor_expectations(prefix)
    require_service_cursors = next(
        (
            item.get("protocol_event_schema") in {3, 4}
            for item in prefix
            if item.get("kind") == "run_started"
        ),
        False,
    )
    # Normal records are actor/Judge replay transcripts. Abandoned records are
    # deliberately absent from those tapes, but remain part of the global
    # meter, provider usage, and per-branch RNG history.
    replayable_by_role: dict[str, tuple[ServiceEvent, ...]] = {
        "generator": (),
        "oracle": (),
        "judge": (),
    }
    accounted_by_role = dict(replayable_by_role)
    if service_prefix is not None:
        replayable: dict[str, list[ServiceEvent]] = {
            "generator": [],
            "oracle": [],
            "judge": [],
        }
        accounted: dict[str, list[ServiceEvent]] = {
            "generator": [],
            "oracle": [],
            "judge": [],
        }
        for payload in service_prefix:
            role = str(payload.get("role") or "")
            if (
                payload.get("kind") not in {"service_call", "service_abandoned"}
                or role not in accounted
                or not isinstance(payload.get("service"), dict)
            ):
                raise ReplayDivergence("committed service journal is malformed")
            event = _decode_service_event(payload["service"])
            accounted[role].append(event)
            if payload.get("kind") == "service_call":
                replayable[role].append(event)
        replayable_by_role = {
            role: tuple(events) for role, events in replayable.items()
        }
        accounted_by_role = {
            role: tuple(events) for role, events in accounted.items()
        }
        for role in ("generator", "oracle"):
            journal_keys = {_service_key(event) for event in replayable_by_role[role]}
            stream_keys: set[bytes] = set()
            streams = actor_streams.get(role) or {}
            if not isinstance(streams, dict):
                raise ReplayDivergence(f"checkpoint {role} actor streams are malformed")
            for stream in streams.values():
                tape = stream.get("service_tape") if isinstance(stream, dict) else None
                if not isinstance(tape, list):
                    raise ReplayDivergence(f"checkpoint {role} actor stream tape is malformed")
                keys = {_service_key(item) for item in tape}
                if not keys.issubset(journal_keys):
                    raise ReplayDivergence(
                        f"checkpoint {role} tape contains an unjournaled service event"
                    )
                stream_keys.update(keys)
            if stream_keys != journal_keys:
                raise ReplayDivergence(
                    f"checkpoint {role} streams do not cover their service journal"
                )
        judge_tape = checkpoint.get("judge_service_tape") or []
        if not isinstance(judge_tape, list) or [
            _service_key(item) for item in judge_tape
        ] != [_service_key(event) for event in replayable_by_role["judge"]]:
            raise ReplayDivergence("checkpoint Judge tape disagrees with its service journal")
    expected_generator = dict(actor_expectations.generator_branches)
    if phase == "generator_output":
        if not actor_expectations.generator_pending:
            raise ReplayDivergence("checkpoint has an unexpected Generator output")
        phase_output = decoded_engine_message("output")
        branch_calls = list(
            expected_generator[actor_expectations.current_generator_branch]
        )
        branch_calls.append(
            ActorCall(
                actor_expectations.pending_generator_input,
                message_hash(phase_output),
            )
        )
        expected_generator[actor_expectations.current_generator_branch] = tuple(
            branch_calls
        )

    current_actor_data: dict[str, dict[str, Any]] = {}
    for role in ("generator", "oracle"):
        try:
            data = _expand_actor_checkpoint_data(
                checkpoint["actors"][role],
                actor_streams.get(role) or {},
            )
        except (KeyError, TypeError) as exc:
            raise ReplayDivergence(f"checkpoint {role} actor reference is malformed") from exc
        _validate_actor_service_cursors(
            data,
            source=f"checkpoint current {role}",
            require_cursors=require_service_cursors,
        )
        _validate_service_state(
            data.get("service_state") or {},
            data.get("service_tape") or [],
            source=f"checkpoint current {role}",
            root_seed=int(manifest.get("seed", 0)) * 2 + (1 if role == "generator" else 2),
            branch_id=str(data.get("branch_id") or "root"),
            committed_events=(
                accounted_by_role[role] if service_prefix is not None else None
            ),
            usage_events=(
                replayable_by_role[role] if service_prefix is not None else None
            ),
            require_metadata=require_service_cursors,
        )
        current_actor_data[role] = data

    if current_actor_data["generator"].get("branch_id") != actor_expectations.current_generator_branch:
        raise ReplayDivergence("checkpoint current Generator branch disagrees with events")
    _assert_actor_calls_bound(
        _checkpoint_calls(current_actor_data["generator"], "checkpoint current generator"),
        expected_generator[actor_expectations.current_generator_branch],
        role="Generator",
    )
    if current_actor_data["oracle"].get("branch_id") != "root":
        raise ReplayDivergence("checkpoint Oracle branch must be root")
    _assert_actor_calls_bound(
        _checkpoint_calls(current_actor_data["oracle"], "checkpoint current oracle"),
        actor_expectations.guide_calls,
        role="Oracle",
    )

    history = checkpoint.get("actor_history") or {}
    generator_history = history.get("generator") or [checkpoint["actors"]["generator"]]
    if [str(item.get("branch_id") or "") for item in generator_history] != list(
        actor_expectations.generator_branch_order
    ):
        raise ReplayDivergence("checkpoint Generator branch history disagrees with events")
    for item in generator_history:
        data = _expand_actor_checkpoint_data(item, actor_streams.get("generator") or {})
        branch_id = str(data.get("branch_id") or "")
        _validate_actor_service_cursors(
            data,
            source=f"checkpoint generator {branch_id}",
            require_cursors=require_service_cursors,
        )
        _validate_service_state(
            data.get("service_state") or {},
            data.get("service_tape") or [],
            source=f"checkpoint generator {branch_id}",
            root_seed=int(manifest.get("seed", 0)) * 2 + 1,
            branch_id=branch_id,
            committed_events=(
                accounted_by_role["generator"] if service_prefix is not None else None
            ),
            usage_events=(
                replayable_by_role["generator"] if service_prefix is not None else None
            ),
            require_metadata=require_service_cursors,
        )
        _assert_actor_calls_bound(
            _checkpoint_calls(data, f"checkpoint generator {branch_id}"),
            expected_generator[branch_id],
            role=f"Generator {branch_id}",
        )
    guide_history = history.get("oracle") or [checkpoint["actors"]["oracle"]]
    if len(guide_history) != 1 or guide_history[0].get("branch_id") != "root":
        raise ReplayDivergence("checkpoint Oracle history disagrees with events")
    guide_history_data = _expand_actor_checkpoint_data(
        guide_history[0], actor_streams.get("oracle") or {}
    )
    _validate_actor_service_cursors(
        guide_history_data,
        source="checkpoint oracle root",
        require_cursors=require_service_cursors,
    )
    _validate_service_state(
        guide_history_data.get("service_state") or {},
        guide_history_data.get("service_tape") or [],
        source="checkpoint oracle root",
        root_seed=int(manifest.get("seed", 0)) * 2 + 2,
        branch_id="root",
        committed_events=(
            accounted_by_role["oracle"] if service_prefix is not None else None
        ),
        usage_events=(
            replayable_by_role["oracle"] if service_prefix is not None else None
        ),
        require_metadata=require_service_cursors,
    )
    _assert_actor_calls_bound(
        _checkpoint_calls(guide_history_data, "checkpoint oracle root"),
        actor_expectations.guide_calls,
        role="Oracle root",
    )

    for question_id, (branch_id, call_count) in actor_expectations.node_generator_refs.items():
        try:
            reference = raw_nodes[question_id]["generator_checkpoint"]
            data = _expand_actor_checkpoint_data(
                reference,
                actor_streams.get("generator") or {},
            )
        except (KeyError, TypeError) as exc:
            raise ReplayDivergence("checkpoint question actor reference is malformed") from exc
        if data.get("branch_id") != branch_id or len(data.get("calls") or []) != call_count:
            raise ReplayDivergence(
                "checkpoint question Generator cursor disagrees with events"
            )
        _validate_actor_service_cursors(
            data,
            source=f"checkpoint question {question_id}",
            require_cursors=require_service_cursors,
        )
        _validate_service_state(
            data.get("service_state") or {},
            data.get("service_tape") or [],
            source=f"checkpoint question {question_id}",
            root_seed=int(manifest.get("seed", 0)) * 2 + 1,
            branch_id=branch_id,
            require_metadata=require_service_cursors,
        )
        _assert_actor_calls_bound(
            _checkpoint_calls(data, f"checkpoint question {question_id}"),
            actor_expectations.generator_branches[branch_id][:call_count],
            role=f"Generator question {question_id}",
        )

    if service_prefix is not None:
        judge_state = checkpoint.get("judge_service_state") or {}
        judge_tape = checkpoint.get("judge_service_tape") or []
        if judge_state or judge_tape or accounted_by_role["judge"]:
            _validate_service_state(
                judge_state,
                judge_tape,
                source="checkpoint Judge",
                root_seed=int(manifest.get("seed", 0)) * 2 + 3,
                branch_id="root",
                committed_events=accounted_by_role["judge"],
                require_metadata=require_service_cursors,
            )

    if phase == "generator_output":
        output = decoded_engine_message("output")
        if actor_output_hash("generator") != message_hash(output):
            raise ReplayDivergence(
                "checkpoint Generator output is not bound to its actor cursor"
            )
        if engine.get("parent_question_id") != protocol["expected_parent_question_id"]:
            raise ReplayDivergence("checkpoint Generator parent is inconsistent")
        if last_kind == "run_started":
            expected_generator_input = None
        else:
            source_id = protocol["expected_parent_question_id"]
            source = branch_state["nodes"].get(source_id)
            option_id = last.get("option_id")
            option = next(
                (
                    value
                    for value in source["question"].options
                    if value.option_id == option_id
                ),
                None,
            ) if source is not None else None
            if option is None:
                raise ReplayDivergence("checkpoint Generator choice source is missing")
            expected_generator_input = Choice(
                option.option_id,
                public_payload=option.public_payload,
            )
        if actor_input("generator") != expected_generator_input:
            raise ReplayDivergence(
                "checkpoint Generator input is not bound to its choice event"
            )

    if phase == "before_judge":
        submitted = decoded_engine_message("submission")
        latest_submission = next(
            (record for record in reversed(prefix) if record.get("kind") == "submission"),
            None,
        )
        if (
            not isinstance(submitted, Submission)
            or latest_submission is None
            or engine["submission"]
            != {"type": "submission", "value": latest_submission.get("submission")}
            or actor_output_hash("generator") != message_hash(submitted)
        ):
            raise ReplayDivergence(
                "checkpoint submission is not bound to its event and Generator output"
            )
        source = branch_state["nodes"].get(protocol["submission_source_question_id"])
        option = next(
            (
                value
                for value in source["question"].options
                if value.option_id == protocol["submission_option_id"]
            ),
            None,
        ) if source is not None else None
        if option is None or actor_input("generator") != Choice(
            option.option_id,
            public_payload=option.public_payload,
        ):
            raise ReplayDivergence(
                "checkpoint submission input is not bound to its SubmitOption"
            )

    if phase in {"after_oracle", "after_submission_recovery"}:
        decision = decoded_engine_message("decision")
        raw_decision = last.get("decision")
        if isinstance(decision, Choice):
            expected_decision = {"type": "choice", "value": raw_decision}
        elif isinstance(decision, Checkout):
            expected_decision = {"type": "checkout", "value": raw_decision}
        else:
            raise ReplayDivergence("checkpoint Oracle decision has an invalid type")
        if engine["decision"] != expected_decision:
            raise ReplayDivergence("checkpoint Oracle decision disagrees with its event")
        if actor_output_hash("oracle") != message_hash(decision):
            raise ReplayDivergence(
                "checkpoint Oracle decision is not bound to its actor cursor"
            )
        if phase == "after_oracle":
            source_id = protocol["current_question_id"]
            expected_guide_input = PresentedQuestion(
                source_id,
                branch_state["nodes"][source_id]["question"],
            )
            if actor_input("oracle") != expected_guide_input:
                raise ReplayDivergence(
                    "checkpoint Oracle input is not bound to its presented question"
                )

    if phase in {"await_submission_recovery", "after_submission_recovery"}:
        latest_submission = next(
            (record for record in reversed(prefix) if record.get("kind") == "submission"),
            None,
        )
        latest_judgment = next(
            (
                record
                for record in reversed(prefix)
                if record.get("kind") == "submission_judged"
            ),
            None,
        )
        if latest_submission is None or latest_judgment is None:
            raise ReplayDivergence("checkpoint recovery has no judged submission")
        source_id = latest_judgment.get("source_question_id")
        source_node = branch_state["nodes"].get(source_id)
        if source_node is None:
            raise ReplayDivergence("checkpoint recovery source is missing")
        try:
            submitted = _decode_submission(latest_submission["submission"])
            verdicts = tuple(
                IdeaVerdict(
                    value["idea_id"],
                    value["passed"],
                    str(value.get("private_reason") or ""),
                )
                for value in latest_judgment.get("verdicts", [])
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ReplayDivergence("checkpoint recovery verdicts are malformed") from exc
        source_index = source_node["created_index"]
        valid_ids = (
            tuple(
                question_id
                for question_id in branch_state["order"]
                if branch_state["nodes"][question_id]["created_index"] < source_index
                or question_id == source_id
            )
            if bool(replayed["time_travel"])
            else (source_id,)
        )
        expected_feedback = SubmissionFeedback(
            PresentedQuestion(source_id, source_node["question"]),
            submitted,
            verdicts,
            valid_ids,
        )
        IDEA_RECOVERY_V1.validate_submission_feedback(expected_feedback)
        if engine.get("submission_feedback") != encode_message(expected_feedback):
            raise ReplayDivergence(
                "checkpoint private submission feedback disagrees with trusted events"
            )
        if (
            phase == "after_submission_recovery"
            and actor_input("oracle") != expected_feedback
        ):
            raise ReplayDivergence(
                "checkpoint Oracle recovery input is not bound to trusted feedback"
            )


def _retry_interrupted_call_marker(
    root: Path,
    checkpoint: dict[str, Any],
    service_records: tuple[dict[str, Any], ...],
) -> dict[str, str]:
    """Return the exact pending provider call eligible for a live retry."""
    private = verify_hash_chain(root / "events.private.jsonl")
    terminal = _event_payload(private[-1]) if private else {}
    marker = terminal.get("interrupted_call")
    if (
        terminal.get("kind") != "run_finished"
        or terminal.get("status") != "error"
        or terminal.get("error_code") not in {
            "participant_failure",
            "match_failure",
            "resource_limit_exceeded",
            "run_budget_blocked",
        }
        or not isinstance(marker, dict)
        or set(marker) != {"role", "branch_id", "operation"}
        or marker.get("role") not in {"generator", "oracle", "judge"}
    ):
        raise ReplayDivergence(
            "--retry-interrupted-call requires an exact failed pending-call marker"
        )
    cursor = int(checkpoint.get("service_event_count", 0))
    tail = tuple(_event_payload(record) for record in service_records[cursor:]
                 if _event_payload(record).get("kind") != "service_abandoned")
    role = str(marker["role"])
    if not tail:
        raise ReplayDivergence(
            "--retry-interrupted-call found no uncommitted provider attempt"
        )
    decoded_tail: list[tuple[str, ServiceEvent]] = []
    for payload in tail:
        event_role = str(payload.get("role") or "")
        if (
            payload.get("kind") != "service_call"
            or event_role not in ({role, "judge"} if role == "oracle" else {role})
            or not isinstance(payload.get("service"), dict)
        ):
            raise ReplayDivergence(
                "--retry-interrupted-call service tail is not one exact pending call"
            )
        decoded_tail.append((event_role, _decode_service_event(payload["service"])))
    # An Oracle can return a syntactically valid but unavailable option ID.
    # Retry that exact uncommitted Oracle turn, retaining its paid service row.
    if (role == "oracle" and terminal.get("error_type") == "ValueError"
            and terminal.get("error_message") == "Oracle selected an unavailable option ID"
            and len(decoded_tail) == 1 and decoded_tail[0][0] == "oracle"
            and decoded_tail[0][1].kind == "model.structured"
            and decoded_tail[0][1].error is None):
        return {key: str(value) for key, value in marker.items()}
    failures = [(index, event) for index, (_, event) in enumerate(decoded_tail)
                if event.error is not None]
    # Judge.evaluate can catch a rejected request and then encounter admission
    # failure on its next internal retry. Both belong to the same uncommitted
    # Judge turn; preserve their accounting and retry from that checkpoint.
    if (role == "judge" and failures
            and all(event_role == "judge" and event.kind == "model.structured"
                    for event_role, event in decoded_tail)
            and decoded_tail[-1][1].error is not None
            and terminal.get("error_type") in {decoded_tail[-1][1].error, "ParticipantFailure"}):
        return {key: str(value) for key, value in marker.items()}
    if len(failures) == 1 and failures[0][0] != len(decoded_tail) - 1:
        # Eager preview workers can finish a successful sibling after one fails.
        # There is no safe single-service suffix to replay in that case: restore
        # the committed Generator checkpoint and repeat this uncommitted turn,
        # retaining every abandoned service and its usage in the derived journal.
        failed = failures[0][1]
        wrapped_error = (terminal.get("error_type") == "RuntimeError" and
                         str(terminal.get("error_message") or "").startswith(f"{failed.error}:"))
        if (role == "generator" and all(event_role == role and event.kind == "model.structured"
                                       for event_role, event in decoded_tail)
                and (terminal.get("error_type") in {failed.error, "ParticipantFailure"} or wrapped_error)):
            return {key: str(value) for key, value in marker.items()}
        raise ReplayDivergence("failed service with a completed sibling is not an auditable Generator turn")
    # A participant turn may complete replayable subcalls before its terminal
    # provider failure. Oracle judge previews are the important case: the
    # target-aware Judge records its model calls, then the Oracle records one
    # judge.evaluate proxy response, and only a later Oracle model call fails.
    # Preserve that successful prefix and retry only the final failed service.
    if (role == "oracle" and _bound_failed_judge_tail(tail)
            and decoded_tail[-1][0] == "oracle"
            and decoded_tail[-1][1].kind == "judge.evaluate"
            and terminal.get("error_type") in {decoded_tail[-1][1].error, "ParticipantFailure"}):
        return {key: str(value) for key, value in marker.items()}
    pending_nested_judge = False
    for event_role, event in decoded_tail[:-1]:
        if event.error is not None:
            raise ReplayDivergence(
                "--retry-interrupted-call has a non-terminal failed service"
            )
        if event_role == "judge":
            pending_nested_judge = True
            continue
        if pending_nested_judge:
            if role != "oracle" or event.kind != "judge.evaluate":
                raise ReplayDivergence(
                    "--retry-interrupted-call has an unbound nested Judge prefix"
                )
            pending_nested_judge = False
    if pending_nested_judge:
        raise ReplayDivergence(
            "--retry-interrupted-call has an unfinished nested Judge prefix"
        )
    final = _decode_service_event(tail[-1]["service"])
    if (
        decoded_tail[-1][0] != role
        or final.error is None
        or terminal.get("error_type") not in {
        final.error,
        # Runs written before ActorRuntime preserved ParticipantFailure detail
        # lose the underlying service exception at this boundary. The exact
        # pending-call marker plus a terminal failed service record is the
        # strongest compatible proof available for those immutable artifacts.
        "ParticipantFailure",
        }
    ):
        raise ReplayDivergence(
            "--retry-interrupted-call is only valid when the terminal service error escaped"
        )
    return {key: str(value) for key, value in marker.items()}


def _bound_failed_judge_tail(records) -> bool:
    """Recognize failed nested Judge calls closed by their failed Oracle proxy."""
    pending = []
    found = False
    for record in records:
        payload = _event_payload(record)
        # Earlier retries retain these for accounting, not the current proxy.
        # Their provenance is validated separately by the resume journal audit.
        if payload.get("kind") == "service_abandoned":
            continue
        if payload.get("kind") != "service_call":
            return False
        role = payload.get("role")
        event = _decode_service_event(payload["service"])
        if role == "judge":
            pending.append(event)
        elif role == "oracle" and event.kind == "judge.evaluate" and pending:
            failures = [e for e in pending if e.error is not None]
            if failures:
                if event.error is None or event.error != pending[-1].error:
                    return False
                found = True
            elif event.error is not None:
                return False
            pending = []
        elif pending or event.error is not None or role != "oracle":
            return False
    return found and not pending


def retry_interrupted_service_indexes(
    root: Path, checkpoint: dict[str, Any], service_records: tuple[dict[str, Any], ...],
) -> tuple[int, ...]:
    """Validated abandoned-service selection, shared by recording and restore."""
    _retry_interrupted_call_marker(root, checkpoint, service_records)
    cursor = int(checkpoint.get("service_event_count", 0))
    failures = [index for index in range(cursor, len(service_records))
                if _event_payload(service_records[index]).get("kind") == "service_call"
                and _event_payload(service_records[index])["service"].get("error") is not None]
    if failures == [len(service_records) - 1]:
        return (failures[0],)
    return tuple(index for index in range(cursor, len(service_records))
                 if _event_payload(service_records[index]).get("kind") == "service_call")


def _patch_checkpoint_service_state(
    checkpoint: dict[str, Any],
    *,
    role: str,
    branch_id: str,
    meter: dict[str, int],
    usage: dict[str, int | float],
    rng_state: Any,
) -> None:
    """Carry abandoned-call accounting into every current/history reference."""
    references = [checkpoint["actors"][role]]
    references.extend((checkpoint.get("actor_history") or {}).get(role) or [])
    seen: set[int] = set()
    for reference in references:
        if id(reference) in seen:
            continue
        seen.add(id(reference))
        state = reference.setdefault("service_state", {})
        state["meter"] = copy.deepcopy(meter)
        state["model_usage"] = copy.deepcopy(usage)
        if str(reference.get("branch_id") or "root") == branch_id:
            state["rng_state"] = copy.deepcopy(rng_state)


def _partition_resume_service_tail(
    service_records: tuple[dict[str, Any], ...],
    cursor: int,
    *,
    retry_interrupted_call: bool,
    discard_service_tail: bool,
    retry_service_indexes: tuple[int, ...] | None = None,
) -> tuple[
    dict[str, list[ServiceEvent]],
    dict[str, list[ServiceEvent]],
    list[ServiceEvent],
]:
    """Split a write-ahead suffix into accounting, replay, and Judge history.

    A replayed Oracle ``judge.evaluate`` proxy returns its recorded response
    without executing the nested Arena Judge again. Contiguous Judge rows
    immediately preceding that proxy are therefore already-consumed history.
    Unbound Judge rows still belong to an unfinished proxy or direct Judge call
    and remain replayable.
    """

    recorded: dict[str, list[ServiceEvent]] = {
        "generator": [], "oracle": [], "judge": [],
    }
    replay: dict[str, list[ServiceEvent]] = {
        "generator": [], "oracle": [], "judge": [],
    }
    if discard_service_tail:
        return recorded, replay, []

    decoded: list[tuple[str, ServiceEvent]] = []
    previously_abandoned = set()
    for index, record in enumerate(service_records[cursor:], cursor):
        payload = _event_payload(record)
        role = str(payload.get("role") or "")
        if role not in recorded or payload.get("kind") not in {"service_call", "service_abandoned"}:
            raise ReplayDivergence("service journal contains an invalid role or event")
        if payload.get("kind") == "service_abandoned":
            previously_abandoned.add(index)
        event = _decode_service_event(payload["service"])
        recorded[role].append(event)
        decoded.append((role, event))

    abandoned = set(retry_service_indexes if retry_service_indexes is not None
                    else (len(service_records) - 1,)) if retry_interrupted_call else set()
    abandoned.update(previously_abandoned)
    replayable = [item for index, item in enumerate(decoded, cursor) if index not in abandoned]
    pending_nested_judge: list[ServiceEvent] = []
    committed_nested_judge: list[ServiceEvent] = []

    def flush_pending_nested_judge() -> None:
        replay["judge"].extend(pending_nested_judge)
        pending_nested_judge.clear()

    for role, event in replayable:
        if role == "judge":
            pending_nested_judge.append(event)
            continue
        if role == "oracle" and event.kind == "judge.evaluate" and pending_nested_judge:
            committed_nested_judge.extend(pending_nested_judge)
            pending_nested_judge.clear()
        else:
            flush_pending_nested_judge()
        replay[role].append(event)
    flush_pending_nested_judge()
    return recorded, replay, committed_nested_judge


def read_resume_checkpoint(root: Path, name: str = "checkpoint.private.json") -> dict[str, Any]:
    """Read a boundary, repairing only the exact historical retry-seed bug in memory.

    A legacy seed can include the pending service suffix in its accounting
    cursor before the actor returned. Reconstruct its byte-equivalent expected
    seed from the verified parent; never edit either run or discard the suffix.
    """
    raw = json.loads((root / name).read_text())
    manifest = json.loads((root / "manifest.json").read_text())
    if name != "checkpoint.private.json" or manifest.get("resume_policy") != "retry_interrupted_call":
        return raw
    parent_id = manifest.get("resumed_from", "")
    if not isinstance(parent_id, str) or len(parent_id) != 32 or any(c not in "0123456789abcdef" for c in parent_id):
        return raw
    # A promotion inherits the source manifest's old retry_policy, but its
    # immediate parent is a completed run with a promotion checkpoint only.
    # It cannot be the legacy failed-call retry seed handled below.
    if manifest.get("promoted_from") == parent_id:
        return raw
    parent = root.parent / parent_id
    original = json.loads((parent / name).read_text())
    services = verify_hash_chain(parent / "service-calls.private.jsonl")
    cursor = original.get("service_event_count", 0)
    if (raw.get("service_event_count") != len(services) or cursor >= len(services)
            or raw.get("actor_streams") != original.get("actor_streams")):
        return raw
    # Validate the parent independently before trusting any of its state.
    actor_replay(parent)
    indexes = retry_interrupted_service_indexes(parent, original, services)
    events = verify_hash_chain(parent / "events.private.jsonl")
    marker = _event_payload(events[-1]).get("interrupted_call") or {}
    if marker.get("role") != "generator":
        return raw  # Historical nested-Judge seeds require separate treatment.
    suffix = [_event_payload(x) for x in services[cursor:]]
    if any(x.get("role") != "generator" for x in suffix):
        return raw
    current = verify_hash_chain(root / "service-calls.private.jsonl")
    if len(current) <= len(services):
        return raw
    for i, (before, after) in enumerate(zip(services, current)):
        before, after = _event_payload(before), _event_payload(after)
        if before["service"] != after["service"] or before["role"] != after["role"]:
            raise ReplayDivergence("legacy retry seed changed a parent service")
        if i in indexes:
            if after.get("kind") != "service_abandoned" or after.get("abandoned_from_run") != parent_id:
                raise ReplayDivergence("legacy retry seed abandoned-service provenance differs")
        elif before.get("kind") != after.get("kind"):
            raise ReplayDivergence("legacy retry seed changed a retained service")
    private = verify_hash_chain(root / "events.private.jsonl")
    prefix = tuple(_event_payload(x) for x in events[:original["private_event_count"]])
    rebased = []
    for e in prefix:
        e = dict(e)
        if e.get("kind") == "run_started":
            e.update(run_id=manifest["run_id"], resumed_from=parent_id)
        rebased.append(e)
    if tuple(rebased) != tuple(_event_payload(x) for x in private[:len(prefix)]):
        raise ReplayDivergence("legacy retry seed changed its committed protocol prefix")
    normalized = copy.deepcopy(original)
    normalized.update(run_id=manifest["run_id"], private_event_count=len(prefix),
                      public_event_count=raw["public_event_count"], updated_at=raw["updated_at"])
    normalized["branches"]["run_id"] = manifest["run_id"]
    expected = copy.deepcopy(normalized)
    decoded = tuple(_decode_service_event(x["service"]) for x in suffix)
    meter, usage = _validate_service_tail_state(decoded,
        initial_state=original["actors"]["generator"]["service_state"],
        source="legacy retry seed suffix", branch_id=marker["branch_id"], require_metadata=True)
    _patch_checkpoint_service_state(expected, role="generator", branch_id=marker["branch_id"],
        meter=meter, usage=usage, rng_state=decoded[-1].metadata["rng_state_after"])
    expected["service_event_count"] = len(services)
    if expected != raw:
        raise ReplayDivergence("legacy retry seed differs from its exact parent reconstruction")
    return normalized


@legacy_keywords(oracle_factory="guide_factory")
def load_resume_state(
    run_directory: str | Path,
    *,
    generator_factory: Any,
    guide_factory: Any,
    runtime: ActorRuntime,
    seed: int,
    retry_interrupted_call: bool = False,
    checkpoint_name: str = "checkpoint.private.json",
    discard_service_tail: bool = False,
) -> tuple[Any, dict[str, Any], dict[str, Any]]:
    """Restore participant and Arena state from the last committed boundary."""
    from ..runtime.branch import BranchStore
    from ..runtime.engine import EngineResumeState

    root = Path(run_directory).resolve()
    try:
        checkpoint = read_resume_checkpoint(root, checkpoint_name)
        manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ReplayDivergence("run has no valid durable checkpoint or manifest") from exc
    if checkpoint.get("run_id") != manifest.get("run_id"):
        raise ReplayDivergence("durable checkpoint belongs to a different run")
    prefix = resume_event_prefix(root, checkpoint_name=checkpoint_name)
    if any(event.get("kind") == "judge_previewed" for event in prefix):
        raise ReplayDivergence(
            "historical JudgePreview records support protocol replay and repricing; "
            "actor resume requires their historical participant protocol"
        )
    service_records: tuple[dict[str, Any], ...] = ()
    service_path = root / "service-calls.private.jsonl"
    if service_path.is_file():
        service_records = verify_hash_chain(service_path, tolerate_truncated_tail=True)
    cursor = int(checkpoint.get("service_event_count", 0))
    if cursor < 0 or cursor > len(service_records):
        raise ReplayDivergence("durable checkpoint has an invalid service cursor")
    committed_service_prefix = tuple(
        _event_payload(record) for record in service_records[:cursor]
    )
    provider_attempt_path = root / "provider-attempts.private.jsonl"
    provider_attempt_records: tuple[dict[str, Any], ...] = ()
    if provider_attempt_path.is_file():
        provider_attempt_records = verify_hash_chain(
            provider_attempt_path, tolerate_truncated_tail=True
        )
    provider_attempt_cursor = int(checkpoint.get("provider_attempt_event_count", 0))
    if (
        provider_attempt_cursor < 0
        or provider_attempt_cursor > len(provider_attempt_records)
    ):
        raise ReplayDivergence("durable checkpoint has an invalid provider-attempt cursor")
    if provider_attempt_cursor:
        # ``--retry-interrupted-call`` resumes a run whose committed prefix may
        # legitimately end with a fully accounted provider failure; only that
        # explicit recovery path relaxes the completeness gate here.
        reconcile_provider_attempt_journal(
            provider_attempt_path,
            service_path,
            max_provider_records=provider_attempt_cursor,
            max_service_records=cursor,
            require_complete=not retry_interrupted_call,
        )
    _validate_resume_checkpoint(
        checkpoint,
        manifest,
        prefix,
        committed_service_prefix,
        require_active_path=True,
    )
    seed_checkpoint = copy.deepcopy(checkpoint)
    event_schema = next(
        (
            item.get("protocol_event_schema")
            for item in prefix
            if item.get("kind") == "run_started"
        ),
        None,
    )
    require_service_metadata = event_schema in {3, 4}

    retry_marker = (
        _retry_interrupted_call_marker(root, checkpoint, service_records)
        if retry_interrupted_call
        else None
    )
    retry_indexes = (retry_interrupted_service_indexes(root, checkpoint, service_records)
                     if retry_interrupted_call else ())

    # The write-ahead suffix is one ordered control-flow transcript. Replaying
    # failures as failures is necessary: dropping them can move a later success
    # to an earlier request or skip participant exception/retry state entirely.
    recorded_services, replay_services, committed_nested_judge = (
        _partition_resume_service_tail(
            service_records,
            cursor,
            retry_interrupted_call=retry_interrupted_call,
            discard_service_tail=discard_service_tail,
            retry_service_indexes=retry_indexes,
        )
    )
    # Committed Judge calls are already-consumed history. The exact post-checkpoint
    # suffix (successes and failures) is queued separately for the in-flight call.
    checkpoint["resume_judge_tape"] = [
        _service_event_data(event) for event in replay_services["judge"]
    ]
    committed_judge_events = [
        _decode_service_event(value)
        for value in (checkpoint.get("judge_service_tape") or [])
    ]
    judge_state = dict(checkpoint.get("judge_service_state") or {})
    judge_meter = dict(judge_state.get("meter") or {})
    judge_usage = dict(judge_state.get("model_usage") or {})
    if not judge_state:
        # Backward compatibility for schema-2 checkpoints written before Judge
        # RNG/meter state was explicit.
        judge_meter = {
            "model_calls": sum(
                _is_model_service(event) for event in committed_judge_events
            ),
            "random_calls": sum(
                event.kind.startswith("random.") for event in committed_judge_events
            ),
        }
        judge_usage = _accumulate_model_usage(judge_usage, committed_judge_events)
    judge_initial_meter = {
        "model_calls": int(judge_meter.get("model_calls", 0)),
        "random_calls": int(judge_meter.get("random_calls", 0)),
    }
    checkpoint["resume_judge_meter"] = _journal_service_meter(
        recorded_services["judge"],
        source="resume Judge service tail",
        initial=judge_initial_meter,
        require_metadata=require_service_metadata,
    )
    judge_usage = _accumulate_model_usage(
        judge_usage, recorded_services["judge"]
    )
    checkpoint["resume_judge_usage"] = judge_usage
    if committed_nested_judge:
        # The Oracle proxy response is replayed directly, so its already-run
        # nested Judge calls must be committed history rather than queued as
        # future Judge responses. Otherwise the next live preview/formal
        # judgment would consume stale results from the interrupted turn.
        nested_meter, nested_usage = _validate_service_tail_state(
            tuple(committed_nested_judge),
            initial_state=judge_state,
            source="completed nested Judge resume prefix",
            branch_id="root",
            require_metadata=require_service_metadata,
        )
        committed_judge_events.extend(committed_nested_judge)
        judge_state.update({
            "meter": nested_meter,
            "model_usage": nested_usage,
            "rng_state": copy.deepcopy(
                committed_nested_judge[-1].metadata["rng_state_after"]
            ),
        })
        augmented_tape = [
            _service_event_data(event) for event in committed_judge_events
        ]
        if retry_interrupted_call:
            # Retry recovery advances the global service cursor through the
            # source tail before the derived checkpoint is seeded, so this
            # history is durable immediately.
            checkpoint["judge_service_tape"] = augmented_tape
            checkpoint["judge_service_state"] = judge_state
        else:
            # A strict resume keeps the source cursor at its old boundary until
            # the in-flight Oracle call finishes. Expose the nested history to
            # the live Judge now, but do not put it in that older durable
            # checkpoint: a second interruption before the next boundary must
            # still leave a checkpoint whose tapes match its cursor.
            checkpoint["resume_committed_judge_tape"] = augmented_tape
            checkpoint["resume_judge_state"] = copy.deepcopy(judge_state)
    if retry_marker is not None and retry_marker["role"] == "judge":
        judge_meter, judge_usage = _validate_service_tail_state(
            tuple(recorded_services["judge"]),
            initial_state=judge_state,
            source="abandoned Judge provider tail",
            branch_id="root",
            require_metadata=require_service_metadata,
        )
        judge_state.update({
            "meter": judge_meter,
            "model_usage": judge_usage,
            "rng_state": copy.deepcopy(
                recorded_services["judge"][-1].metadata["rng_state_after"]
            ),
        })
        checkpoint["judge_service_state"] = judge_state
        checkpoint["resume_judge_meter"] = copy.deepcopy(judge_meter)
        checkpoint["resume_judge_usage"] = copy.deepcopy(judge_usage)

    factories = {"generator": generator_factory, "oracle": guide_factory}
    actor_streams = checkpoint.get("actor_streams") or {}
    restored_histories: dict[str, list[tuple[str, Any]]] = {}
    current_handles: dict[str, Any] = {}
    for role, factory in factories.items():
        role_streams = actor_streams.get(role) or {}
        current_data = _expand_actor_checkpoint_data(
            checkpoint["actors"][role], role_streams,
        )
        current_data.setdefault("service_tape", []).extend(
            _service_event_data(event) for event in replay_services[role]
        )
        service_state = current_data.get("service_state") or {}
        meter = service_state.get("meter") or {}
        initial_meter = {
            "model_calls": int(meter.get("model_calls", 0)),
            "random_calls": int(meter.get("random_calls", 0)),
        }
        resumed_meter = _journal_service_meter(
            recorded_services[role],
            source=f"resume {role} service tail",
            initial=initial_meter,
            require_metadata=require_service_metadata,
        )
        with factory.service_factory.meter.lock:
            factory.service_factory.meter.model_calls = resumed_meter["model_calls"]
            factory.service_factory.meter.random_calls = resumed_meter["random_calls"]
        usage = _accumulate_model_usage(
            dict(service_state.get("model_usage") or {}),
            recorded_services[role],
        )
        if retry_marker is not None and retry_marker["role"] == role:
            branch_id = retry_marker["branch_id"]
            resumed_meter, usage = _validate_service_tail_state(
                tuple(recorded_services[role]),
                initial_state=service_state,
                source=f"abandoned {role} provider tail",
                branch_id=branch_id,
                require_metadata=require_service_metadata,
            )
            _patch_checkpoint_service_state(
                checkpoint,
                role=role,
                branch_id=branch_id,
                meter=resumed_meter,
                usage=usage,
                rng_state=recorded_services[role][-1].metadata["rng_state_after"],
            )
            current_data = _expand_actor_checkpoint_data(
                checkpoint["actors"][role], role_streams,
            )
            current_data.setdefault("service_tape", []).extend(
                _service_event_data(event) for event in replay_services[role]
            )
            service_state = current_data.get("service_state") or {}
        agent_backend = getattr(factory.service_factory, "agent_backend", None)
        usage_backends = (
            (agent_backend,)
            if agent_backend is not None
            else (factory.service_factory.model_backend,)
        )
        for backend in usage_backends:
            restore_usage = getattr(backend, "restore_usage", None)
            if callable(restore_usage):
                restore_usage(usage)

        history_data = (checkpoint.get("actor_history") or {}).get(role) or [current_data]
        history: list[tuple[str, Any]] = []
        current = None
        current_signature = (
            str(current_data.get("branch_id") or "root"),
            tuple(item.get("output_hash") for item in current_data.get("calls", [])),
        )
        for item in history_data:
            actor_checkpoint = _decode_actor_checkpoint(item, factory, role_streams)
            handle = runtime.restore(actor_checkpoint)
            history.append((actor_checkpoint.branch_id, handle))
            signature = (
                actor_checkpoint.branch_id,
                tuple(call.output_hash for call in actor_checkpoint.calls),
            )
            if signature == current_signature and not replay_services[role]:
                current = handle
        if current is None:
            actor_checkpoint = _decode_actor_checkpoint(current_data, factory)
            if replay_services[role]:
                current = runtime.replay_prefix(actor_checkpoint)
                current.services.restore_state(actor_checkpoint.service_state)
            else:
                current = runtime.restore(actor_checkpoint)
            history = [item for item in history if item[0] != actor_checkpoint.branch_id]
            history.append((actor_checkpoint.branch_id, current))
        restored_histories[role] = history
        current_handles[role] = current

    branches = BranchStore.from_state(
        checkpoint["branches"],
        seed=seed,
        decode_checkpoint=lambda value: _decode_actor_checkpoint(
            value,
            generator_factory,
            actor_streams.get("generator") or {},
        ),
    )
    engine = dict(checkpoint.get("engine") or {})
    for key in (
        "output",
        "decision",
        "submission",
        "submission_feedback",
    ):
        if engine.get(key) is not None:
            engine[key] = decode_message(engine[key])
    raw_transitions = engine.get("stage_transitions") or []
    try:
        stage_transitions = tuple(decode_message(value) for value in raw_transitions)
    except (KeyError, TypeError, ValueError) as exc:
        raise ReplayDivergence("checkpoint stage-transition chain is malformed") from exc
    if any(not isinstance(value, StageTransition) for value in stage_transitions):
        raise ReplayDivergence("checkpoint stage-transition chain is malformed")
    state = EngineResumeState(
        phase=str(engine["phase"]),
        branches=branches,
        generator=current_handles["generator"],
        guide=current_handles["oracle"],
        k=float(engine.get("k", 0.0)),
        decisions=int(engine.get("decisions", 0)),
        checkouts=int(engine.get("checkouts", 0)),
        submission_attempts=int(engine.get("submission_attempts", 0)),
        parent_question_id=engine.get("parent_question_id"),
        current_question_id=engine.get("current_question_id"),
        output=engine.get("output"),
        decision=engine.get("decision"),
        submission=engine.get("submission"),
        submission_feedback=engine.get("submission_feedback"),
        submission_source_question_id=engine.get("submission_source_question_id"),
        submission_option_id=engine.get("submission_option_id"),
        submission_choice_bits=float(engine.get("submission_choice_bits", 0.0)),
        generator_history=restored_histories["generator"],
        guide_history=restored_histories["oracle"],
        stage_transitions=stage_transitions,
        accounting_update=(
            reprice_events(prefix) if branches.accounting_version == LEGACY_ACCOUNTING else None
        ),
    )
    if retry_interrupted_call:
        checkpoint["service_event_count"] = len(service_records)
    checkpoint["_resume_seed_checkpoint"] = seed_checkpoint
    return state, checkpoint, manifest


def _validate_completed_service_accounting(
    root: Path,
    manifest: dict[str, Any],
    *,
    seed: int,
) -> None:
    """Bind a completed run's tapes and usage to its full service journal."""
    path = root / "service-accounting.private.json"
    if not path.is_file():
        # Compatibility for completed runs written before this audit snapshot
        # existed. New runs always contain it, including retry-derived runs.
        return
    try:
        accounting = json.loads(path.read_text(encoding="utf-8"))
        usage_data = json.loads((root / "usage.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ReplayDivergence("completed service accounting is missing or invalid") from exc
    if accounting.get("run_id") != manifest.get("run_id"):
        raise ReplayDivergence("completed service accounting belongs to another run")
    service_path = root / "service-calls.private.jsonl"
    records = verify_hash_chain(service_path) if service_path.is_file() else ()
    if int(accounting.get("service_event_count", -1)) != len(records):
        raise ReplayDivergence("completed service accounting cursor diverges")
    provider_path = root / "provider-attempts.private.jsonl"
    provider_records = (
        verify_hash_chain(provider_path) if provider_path.is_file() else ()
    )
    if int(accounting.get("provider_attempt_event_count", 0)) != len(
        provider_records
    ):
        raise ReplayDivergence("completed provider-attempt accounting cursor diverges")
    instrumented_structured_calls = 0
    for record in records:
        payload = _event_payload(record)
        service = payload.get("service")
        request = service.get("request") if isinstance(service, dict) else None
        metadata = service.get("metadata") if isinstance(service, dict) else None
        if isinstance(metadata, dict) and metadata.get("backend") == "common-memory-v1":
            from ..runtime.memory_output_budget import validate_metadata
            validate_metadata(service, ArtifactStore(root.parent))
            continue
        if isinstance(metadata, dict) and metadata.get("backend") == "codex-source-v1":
            from ..runtime.codex_generator import validate_metadata
            validate_metadata(service)
        if isinstance(metadata, dict) and metadata.get("backend") == "claude-code-native-v1":
            from ..runtime.claude_generator import validate_metadata
            validate_metadata(service)
        usage = metadata.get("usage") if isinstance(metadata, dict) else None
        attempts = metadata.get("provider_attempts") if isinstance(metadata, dict) else None
        model = request.get("model") if isinstance(request, dict) else None
        native_route = (
            isinstance(model, str)
            and bool(model)
            and not model.startswith(("together/", "openrouter/"))
        )
        backend_binding = (
            isinstance(metadata, dict)
            and metadata.get("model") == model
            and isinstance(metadata.get("request_sha"), str)
        )
        if (
            isinstance(service, dict)
            and service.get("kind") == "model.structured"
            and (
                isinstance(attempts, list) and bool(attempts)
                or (
                    isinstance(usage, dict)
                    and type(usage.get("provider_calls")) is int
                    and usage["provider_calls"] > 0
                )
                or native_route and backend_binding
            )
        ):
            instrumented_structured_calls += 1
    if instrumented_structured_calls and not provider_records:
        raise ReplayDivergence(
            "instrumented structured services lack a durable provider-attempt journal"
        )
    if provider_records:
        reconcile_provider_attempt_journal(
            provider_path,
            service_path,
            require_complete=True,
        )

    accounted: dict[str, list[ServiceEvent]] = {
        "generator": [], "oracle": [], "judge": [],
    }
    replayable: dict[str, list[ServiceEvent]] = {
        "generator": [], "oracle": [], "judge": [],
    }
    for record in records:
        payload = _event_payload(record)
        role = str(payload.get("role") or "")
        kind = payload.get("kind")
        if (
            role not in accounted
            or kind not in {"service_call", "service_abandoned"}
            or not isinstance(payload.get("service"), dict)
        ):
            raise ReplayDivergence("completed service journal is malformed")
        if kind == "service_abandoned" and (
            manifest.get("resume_policy") != "retry_interrupted_call"
            or not isinstance(payload.get("abandoned_from_run"), str)
            or not isinstance(payload.get("source_sequence"), int)
        ):
            raise ReplayDivergence("abandoned service record has no retry provenance")
        event = _decode_service_event(payload["service"])
        accounted[role].append(event)
        if kind == "service_call":
            replayable[role].append(event)

    actors = accounting.get("actors") or {}
    for role in ("generator", "oracle"):
        history_path = root / f"actor-branches.{role}.private.json"
        history = (
            _load_actor_history(history_path)
            if history_path.is_file()
            else ((
                "root",
                _load_calls(root / f"actor-calls.{role}.private.json"),
                _load_service_tape(root / f"service-tape.{role}.private.json"),
            ),)
        )
        journal_keys = {_service_key(event) for event in replayable[role]}
        tape_keys = {
            _service_key(event)
            for _, _, tape in history
            for event in tape
        }
        if tape_keys != journal_keys:
            raise ReplayDivergence(
                f"completed {role} tapes disagree with the service journal"
            )
        row = actors.get(role) or {}
        branch_id = str(row.get("branch_id") or "")
        current_tape = next(
            (tape for candidate, _, tape in history if candidate == branch_id),
            None,
        )
        if current_tape is None:
            raise ReplayDivergence(f"completed {role} accounting branch is missing")
        _validate_service_state(
            row.get("service_state") or {},
            [_service_event_data(event) for event in current_tape],
            source=f"completed {role}",
            root_seed=seed * 2 + (1 if role == "generator" else 2),
            branch_id=branch_id,
            committed_events=tuple(accounted[role]),
            require_metadata=True,
        )
        state = row.get("service_state") or {}
        meter = _parse_service_meter(state.get("meter"), source=f"completed {role}")
        summary = usage_data.get(role) or {}
        expected_usage = _parse_model_usage(
            state.get("model_usage") or {}, source=f"completed {role}"
        )
        if (
            int(summary.get("model_calls", -1)) != meter["model_calls"]
            or int(summary.get("random_calls", -1)) != meter["random_calls"]
            or int(summary.get("service_events", -1)) != len(current_tape)
            or _parse_model_usage(
                summary.get("model_usage") or {}, source=f"completed {role} usage"
            ) != expected_usage
        ):
            raise ReplayDivergence(
                f"completed {role} usage disagrees with service accounting"
            )

    judge_tape_path = root / "service-tape.judge.private.json"
    judge_tape = _load_service_tape(judge_tape_path) if judge_tape_path.is_file() else ()
    if [_service_key(event) for event in judge_tape] != [
        _service_key(event) for event in replayable["judge"]
    ]:
        raise ReplayDivergence("completed Judge tape disagrees with the service journal")
    judge_state = accounting.get("judge_service_state") or {}
    if judge_state or accounted["judge"] or judge_tape:
        _validate_service_state(
            judge_state,
            [_service_event_data(event) for event in judge_tape],
            source="completed Judge",
            root_seed=seed * 2 + 3,
            branch_id="root",
            committed_events=tuple(accounted["judge"]),
            require_metadata=True,
        )
        meter = _parse_service_meter(judge_state.get("meter"), source="completed Judge")
        summary = usage_data.get("judge") or {}
        expected_usage = _parse_model_usage(
            judge_state.get("model_usage") or {}, source="completed Judge"
        )
        if (
            int(summary.get("model_calls", -1)) != meter["model_calls"]
            or int(summary.get("random_calls", -1)) != meter["random_calls"]
            or int(summary.get("service_events", -1)) != len(judge_tape)
            or _parse_model_usage(
                summary.get("model_usage") or {}, source="completed Judge usage"
            ) != expected_usage
        ):
            raise ReplayDivergence(
                "completed Judge usage disagrees with service accounting"
            )


def actor_replay(run_directory: str | Path, *, model_backend: Any = None) -> dict[str, Any]:
    root = Path(run_directory).resolve()
    # Actor determinism is an additional replay layer, never a substitute for
    # protocol validation. This also prevents legacy fallbacks or intact actor
    # tapes from masking an impossible/tampered Arena event sequence.
    protocol_replay(root)
    manifest_data = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    sample_mode = manifest_data.get("run_kind") == "sample_ideas"
    artifacts = ArtifactStore(root.parent)
    submission_root = (
        artifacts.resolve_tree(manifest_data["submission_snapshot"])
        if manifest_data.get("submission_snapshot") else Path(manifest_data["submission_path"])
    )
    submission = load_manifest(submission_root)
    if hash_tree(submission.root) != manifest_data["submission_sha256"]:
        raise ReplayDivergence("submission files changed since the run")
    validate_entrypoint_sources(submission)
    dependency_paths = (
        tuple(artifacts.resolve_tree(item) for item in manifest_data.get("dependency_snapshots", []))
        if manifest_data.get("dependency_snapshots") is not None
        else build_dependency_paths(submission.root)
    )
    target: dict[str, Any] = {}
    if not sample_mode:
        pack_root = (
            artifacts.resolve_tree(manifest_data["target_pack_snapshot"])
            if manifest_data.get("target_pack_snapshot")
            else manifest_data.get("target_pack_path") or manifest_data["target_pack"]
        )
        pack = load_target_pack(pack_root)
        if hash_tree(pack.root) != manifest_data.get("target_pack_sha256"):
            raise ReplayDivergence("target pack changed since the run")
        target = pack.load(manifest_data["target_id"])
    if message_hash(target) != manifest_data["target_sha256"]:
        raise ReplayDivergence("target changed since the run")
    seed = int(manifest_data["seed"])
    generator_resources = (
        artifacts.load_json(manifest_data["generator_public_resources_ref"])
        if manifest_data.get("generator_public_resources_ref")
        else manifest_data.get("generator_public_resources") or {}
    )
    guide_resources = (
        artifacts.load_json(manifest_data["oracle_public_resources_ref"])
        if manifest_data.get("oracle_public_resources_ref")
        else manifest_data.get("oracle_public_resources") or {}
    )

    def replay_only_judge_call(_ideas: Any) -> Any:
        # Oracle Judge previews are actor-local proxy service events.  They
        # must resolve from the recorded Oracle tape during actor replay, but
        # ``ReplayableServices.judge_evaluate`` still requires a configured
        # capability before it consults that tape.  Supply a fail-closed
        # capability here: a complete tape never invokes it, while a missing
        # or divergent event cannot silently execute a fresh Judge call.
        raise ReplayDivergence(
            "actor replay attempted an unrecorded live Oracle Judge call"
        )

    factories = {
        "generator": SubprocessActorFactory(
            submission.root,
            submission.generator,
            service_factory=ServiceFactory(
                seed=seed * 2 + 1,
                model_backend=model_backend,
                model_name=(manifest_data.get("models") or {}).get("generator"),
                reasoning_effort=(manifest_data.get("generator_memory") or manifest_data.get("generator_claude") or manifest_data.get("generator_codex") or {}).get("reasoning_effort"),
                public_resources=generator_resources,
            ),
            dependency_paths=dependency_paths,
            sandbox_generator=bool(manifest_data.get("generator_memory") or manifest_data.get("generator_codex") or manifest_data.get("generator_claude")),
        ),
        "oracle": SubprocessActorFactory(
            submission.root,
            str(manifest_data.get("oracle_entrypoint") or submission.guide),
            constructor_args=() if sample_mode else (target,),
            service_factory=ServiceFactory(
                seed=seed * 2 + 2,
                model_backend=model_backend,
                model_name=(manifest_data.get("models") or {}).get("oracle"),
                public_resources=guide_resources,
                judge_call=replay_only_judge_call,
            ),
            dependency_paths=dependency_paths,
        ),
    }
    runtime = ActorRuntime()
    counts: dict[str, int] = {}
    branch_counts: dict[str, int] = {}
    unreplayed_service_events: dict[str, int] = {}
    try:
        score_data = json.loads((root / "score.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ReplayDivergence("score file is missing or invalid") from exc
    failed_run = score_data.get("status") == "error"
    if not failed_run:
        _validate_completed_service_accounting(
            root,
            manifest_data,
            seed=seed,
        )
    private = verify_hash_chain(root / "events.private.jsonl")
    private_payloads = tuple(_event_payload(record) for record in private)
    actor_expectations = _derive_actor_expectations(private_payloads)
    terminal = next(
        (record for record in private_payloads if record.get("kind") == "run_finished"),
        None,
    )
    interrupted = terminal.get("interrupted_call") if terminal is not None else None
    if interrupted is not None:
        if (
            not isinstance(interrupted, dict)
            or set(interrupted) != {"role", "branch_id", "operation"}
            or interrupted.get("role") not in {"generator", "oracle", "judge"}
        ):
            raise ReplayDivergence("terminal interrupted-call marker is malformed")
        role = interrupted["role"]
        expected_branch = (
            actor_expectations.current_generator_branch
            if role == "generator"
            else "root"
            if role == "oracle"
            else "judge"
        )
        pending = {
            "generator": actor_expectations.generator_pending,
            "oracle": actor_expectations.guide_pending,
            "judge": _judge_call_is_pending(actor_expectations),
        }[role]
        if interrupted.get("branch_id") != expected_branch or not pending:
            raise ReplayDivergence(
                "interrupted-call marker disagrees with the protocol position"
            )
        expected_operation = "evaluate" if role == "judge" else "step"
        if interrupted.get("operation") != expected_operation:
            raise ReplayDivergence("interrupted-call marker has an invalid operation")

    expected_generator = dict(actor_expectations.generator_branches)
    checkpoint_bound_pending: tuple[str, str] | None = None
    failed_service_context: dict[str, Any] | None = None
    checkpoint_path = root / "checkpoint.private.json"
    if failed_run and checkpoint_path.is_file():
        try:
            durable = read_resume_checkpoint(root)
            cursor = int(durable.get("private_event_count", -1))
            service_cursor = int(durable.get("service_event_count", 0))
        except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise ReplayDivergence("failed run has an invalid durable checkpoint") from exc
        service_records: tuple[dict[str, Any], ...] = ()
        service_path = root / "service-calls.private.jsonl"
        if service_path.is_file():
            service_records = verify_hash_chain(
                service_path,
                tolerate_truncated_tail=True,
            )
        if service_cursor < 0 or service_cursor > len(service_records):
            raise ReplayDivergence("durable checkpoint has an invalid service cursor")
        committed_service_prefix = tuple(
            _event_payload(record) for record in service_records[:service_cursor]
        )
        if not 0 < cursor <= len(private_payloads):
            raise ReplayDivergence("failed run has an invalid durable event cursor")
        _validate_resume_checkpoint(
            durable,
            manifest_data,
            private_payloads[:cursor],
            committed_service_prefix,
        )
        if (
            cursor == len(private_payloads) - 1
            and (durable.get("engine") or {}).get("phase") == "generator_output"
        ):
            if interrupted is None or interrupted.get("role") != "generator":
                raise ReplayDivergence(
                    "uncommitted Generator output has no exact pending-call marker"
                )
            output = decode_message(durable["engine"]["output"])
            calls = list(
                expected_generator[actor_expectations.current_generator_branch]
            )
            calls.append(
                ActorCall(
                    actor_expectations.pending_generator_input,
                    message_hash(output),
                )
            )
            expected_generator[actor_expectations.current_generator_branch] = tuple(calls)
            checkpoint_bound_pending = (
                "generator",
                actor_expectations.current_generator_branch,
            )

        tail_by_role: dict[str, list[ServiceEvent]] = {
            "generator": [], "oracle": [], "judge": [],
        }
        accounted_tail = {role: [] for role in tail_by_role}
        for record in service_records[service_cursor:]:
            payload = _event_payload(record)
            role = str(payload.get("role") or "")
            if (
                payload.get("kind") not in {"service_call", "service_abandoned"}
                or role not in tail_by_role
                or not isinstance(payload.get("service"), dict)
            ):
                raise ReplayDivergence("post-checkpoint service journal is malformed")
            event = _decode_service_event(payload["service"])
            accounted_tail[role].append(event)
            if payload["kind"] == "service_call":
                tail_by_role[role].append(event)
        if any(tail_by_role.values()) and interrupted is None:
            raise ReplayDivergence("post-checkpoint service tail has no interrupted call")
        for role, events in tail_by_role.items():
            nested_guide_judge = bool(
                interrupted is not None
                and interrupted.get("role") == "oracle"
                and role == "judge"
                and (all(event.error is None for event in events)
                     or _bound_failed_judge_tail(service_records[service_cursor:]))
            )
            if events and interrupted.get("role") != role and not nested_guide_judge:
                raise ReplayDivergence("post-checkpoint service tail belongs to another role")

        actor_streams = durable.get("actor_streams") or {}
        checkpoint_tapes: dict[str, dict[str, tuple[ServiceEvent, ...]]] = {
            "generator": {}, "oracle": {},
        }
        checkpoint_states: dict[str, dict[str, Any]] = {}
        for role in ("generator", "oracle"):
            streams = actor_streams.get(role) or {}
            checkpoint_tapes[role] = {
                str(branch_id): _decode_service_records(
                    stream.get("service_tape") or [], f"checkpoint {role} {branch_id}"
                )
                for branch_id, stream in streams.items()
            }
            current_data = _expand_actor_checkpoint_data(durable["actors"][role], streams)
            checkpoint_states[role] = dict(current_data.get("service_state") or {})

        event_schema = next(
            (
                item.get("protocol_event_schema")
                for item in private_payloads[:cursor]
                if item.get("kind") == "run_started"
            ),
            None,
        )
        require_metadata = event_schema in {3, 4}
        expected_meter: dict[str, dict[str, int]] = {}
        expected_usage: dict[str, dict[str, int | float]] = {}
        for role in ("generator", "oracle"):
            branch_id = (
                str(interrupted.get("branch_id"))
                if interrupted is not None and interrupted.get("role") == role
                else str(durable["actors"][role].get("branch_id") or "root")
            )
            state = dict(checkpoint_states[role])
            if role == "generator" and branch_id not in checkpoint_tapes[role]:
                state["rng_state"] = _branch_rng_state(seed * 2 + 1, branch_id)
            expected_meter[role], expected_usage[role] = _validate_service_tail_state(
                tuple(accounted_tail[role]),
                initial_state=state,
                source=f"post-checkpoint {role} service tail",
                branch_id=branch_id,
                require_metadata=require_metadata,
            )

        judge_state = dict(durable.get("judge_service_state") or {})
        judge_prefix = _decode_service_records(
            durable.get("judge_service_tape") or [], "checkpoint Judge"
        )
        if judge_state or judge_prefix or tail_by_role["judge"]:
            expected_meter["judge"], expected_usage["judge"] = _validate_service_tail_state(
                tuple(accounted_tail["judge"]),
                initial_state=judge_state,
                source="post-checkpoint Judge service tail",
                branch_id="root",
                require_metadata=require_metadata,
            )
        failed_service_context = {
            "durable": durable,
            "checkpoint_tapes": checkpoint_tapes,
            "tail_by_role": {key: tuple(value) for key, value in tail_by_role.items()},
            "expected_meter": expected_meter,
            "expected_usage": expected_usage,
            "judge_prefix": judge_prefix,
        }

    for role, factory in factories.items():
        history_path = root / f"actor-branches.{role}.private.json"
        if history_path.is_file():
            history = _load_actor_history(history_path)
        else:
            history = ((
                "root",
                _load_calls(root / f"actor-calls.{role}.private.json"),
                _load_service_tape(root / f"service-tape.{role}.private.json"),
            ),)
        if role == "generator":
            if tuple(item[0] for item in history) != actor_expectations.generator_branch_order:
                raise ReplayDivergence(
                    "Generator actor branches disagree with checkout events"
                )
            trusted_by_branch = expected_generator
        else:
            if tuple(item[0] for item in history) != ("root",):
                raise ReplayDivergence("Oracle actor history must contain only root")
            trusted_by_branch = {"root": actor_expectations.guide_calls}

        total_calls = 0
        current_service_event_count: int | None = None
        for branch_id, calls, tape in history:
            trusted = trusted_by_branch.get(branch_id)
            if trusted is None:
                raise ReplayDivergence(f"{role} actor branch is not present in events")
            allow_pending_call = bool(
                failed_run
                and interrupted is not None
                and interrupted.get("role") == role
                and interrupted.get("branch_id") == branch_id
            )
            bound_pending_call = checkpoint_bound_pending == (role, branch_id)
            pending_input = (
                actor_expectations.pending_generator_input
                if role == "generator"
                else actor_expectations.pending_guide_input
            )
            extra_pending_call = _assert_actor_calls_bound(
                calls,
                trusted,
                role=f"{role} {branch_id}",
                # A phase-local checkpoint has already consumed the marker by
                # binding the completed return into ``trusted``.  Only an
                # artifact without that checkpoint (for example a derived
                # resume failure) may use the same marker to explain one extra
                # completed ActorCall.
                allow_extra=allow_pending_call and not bound_pending_call,
                allow_extra_input=pending_input,
            )
            pending_call_count = int(bound_pending_call) + int(extra_pending_call)
            if pending_call_count > 1:
                raise ReplayDivergence(
                    f"{role} {branch_id} pending-call marker was consumed more than once"
                )
            if failed_service_context is not None:
                prefixes = failed_service_context["checkpoint_tapes"][role]
                suffix = failed_service_context["tail_by_role"][role]
                interrupted_branch = bool(
                    interrupted is not None
                    and interrupted.get("role") == role
                    and interrupted.get("branch_id") == branch_id
                )
                prefix = prefixes.get(branch_id)
                if prefix is None:
                    if role != "generator" or not interrupted_branch:
                        raise ReplayDivergence(
                            f"{role} final service tape has an unexplained branch"
                        )
                    base_length = len(tape) - len(suffix)
                    if base_length < 0:
                        raise ReplayDivergence("Generator final service tape is truncated")
                    prefix = tape[:base_length]
                    streams = (
                        failed_service_context["durable"].get("actor_streams") or {}
                    ).get("generator") or {}
                    allowed_prefixes = []
                    durable_nodes = (
                        (failed_service_context["durable"].get("branches") or {}).get("nodes")
                        or {}
                    )
                    for node in durable_nodes.values():
                        expanded = _expand_actor_checkpoint_data(
                            node.get("generator_checkpoint") or {}, streams
                        )
                        allowed_prefixes.append(_decode_service_records(
                            expanded.get("service_tape") or [], "checkpoint Generator node"
                        ))
                    if prefix not in allowed_prefixes:
                        raise ReplayDivergence(
                            "Generator fork service prefix is not checkpoint-bound"
                        )
                expected_tape = prefix + (suffix if interrupted_branch else ())
                if tape != expected_tape:
                    raise ReplayDivergence(
                        f"{role} post-checkpoint service tail disagrees with its journal"
                    )
                current_branch = (
                    actor_expectations.current_generator_branch if role == "generator" else "root"
                )
                if branch_id == current_branch:
                    current_service_event_count = len(tape)
            checkpoint = ActorCheckpoint(factory, calls, tape, branch_id)
            allow_tail = allow_pending_call
            if allow_tail:
                handle = runtime.replay_prefix(checkpoint)
                remaining = handle.services.replay_remaining()
                if pending_call_count and remaining:
                    runtime.close(handle)
                    raise ReplayDivergence(
                        f"{role} {branch_id} completed pending call has trailing service events"
                    )
                if remaining:
                    unreplayed_service_events[role] = remaining
            else:
                handle = runtime.fork(checkpoint, f"replay-{role}-{branch_id}")
            runtime.close(handle)
            total_calls += len(calls)
        counts[role] = total_calls
        branch_counts[role] = len(history)
        if failed_service_context is not None:
            try:
                usage_data = json.loads((root / "usage.json").read_text(encoding="utf-8"))
                usage_row = usage_data[role]
            except (OSError, KeyError, TypeError, json.JSONDecodeError) as exc:
                raise ReplayDivergence("failed run usage summary is invalid") from exc
            meter = failed_service_context["expected_meter"][role]
            usage = _parse_model_usage(
                usage_row.get("model_usage") or {}, source=f"{role} usage summary"
            )
            try:
                summary_matches = (
                    int(usage_row.get("model_calls", -1)) == meter["model_calls"]
                    and int(usage_row.get("random_calls", -1)) == meter["random_calls"]
                    and int(usage_row.get("service_events", -1))
                    == current_service_event_count
                    and usage == failed_service_context["expected_usage"][role]
                )
            except (TypeError, ValueError) as exc:
                raise ReplayDivergence("failed run usage summary is malformed") from exc
            if not summary_matches:
                raise ReplayDivergence(
                    f"{role} usage summary disagrees with its service journal"
                )
    judge_calls = 0
    judge_replayed = False
    submission_records = [
        _event_payload(record) for record in private if record.get("kind") == "submission"
    ]
    judged_records = [
        _event_payload(record)
        for record in private
        if record.get("kind") == "submission_judged"
    ]
    if not failed_run and (not submission_records or terminal is None):
        raise ReplayDivergence("completed run is missing submission or terminal events")

    # Legacy completed traces stored the one trusted verdict only on
    # run_finished. Submit-aware traces record every judgment, including
    # rejected attempts. Error runs may have an unjudged trailing submission,
    # but every committed judgment must still be independently replayed.
    expected_records = judged_records or (
        [terminal] if not failed_run and terminal is not None else []
    )
    submissions_by_attempt: dict[int, dict[str, Any]] = {}
    for index, record in enumerate(submission_records, 1):
        attempt = int(record.get("attempt", index))
        if attempt in submissions_by_attempt:
            raise ReplayDivergence("submission attempt ID is duplicated")
        submissions_by_attempt[attempt] = record
    replay_pairs: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for index, expected in enumerate(expected_records, 1):
        attempt = int(expected.get("attempt", index))
        submission_record = submissions_by_attempt.get(attempt)
        if submission_record is None:
            raise ReplayDivergence("judgment has no matching submission attempt")
        replay_pairs.append((submission_record, expected))
    if not failed_run and len(replay_pairs) != len(submission_records):
        raise ReplayDivergence("submission and judgment attempt counts diverge")

    # Judge previews share the authoritative Judge service tape with formal
    # submission judgments. Replay both in their original protocol order so
    # model responses and cost accounting remain deterministic.
    formal_pairs_by_attempt = {
        int(expected.get("attempt", index)): (_decode_submission(submission["submission"]), expected)
        for index, (submission, expected) in enumerate(replay_pairs, 1)
    }
    # A promoted run's copied prefix contains judgments made under the
    # PREVIOUS rung's Judge, so the "active" profile is time-dependent. Walking
    # stage_transition events proved unreliable: a chained promotion
    # (directional -> essence -> strict) copies and re-inserts transition
    # records, so the stream carries duplicated, reordered transitions. The
    # Judge service tape is the ground truth instead -- each recorded call's
    # schema_name names the era that actually produced it, and pairs consume
    # the tape strictly in order, so each pair's era is read from the next
    # unconsumed tape entry at replay time.
    # Merge formal judgments with Oracle previews in durable service-journal
    # order. A preview's ``judge.evaluate`` proxy immediately follows the
    # underlying Judge calls. A formal evaluation instead ends when the journal
    # reaches another role (or EOF). This service-local boundary is stable when
    # resume re-hashes a prefix with new ``recorded_at`` values; wall-clock
    # timestamps are not.
    judge_replay_pairs: list[
        tuple[Submission, dict[str, Any], str, int | None]
    ] = []
    formal_judgments: list[tuple[Submission, dict[str, Any], str]] = []
    for raw_record in private:
        record = _event_payload(raw_record)
        if record.get("kind") != "submission_judged":
            continue
        pair = formal_pairs_by_attempt.get(int(record.get("attempt", -1)))
        if pair is None:
            raise ReplayDivergence("formal judgment is missing its replay pair")
        formal_judgments.append((*pair, "active"))

    service_journal_path = root / "service-calls.private.jsonl"
    service_journal = (
        verify_hash_chain(service_journal_path, tolerate_truncated_tail=failed_run)
        if service_journal_path.is_file()
        else ()
    )
    judge_cursor = 0
    pending_judge_group = False

    def finish_formal_group() -> None:
        nonlocal pending_judge_group
        if not pending_judge_group:
            return
        if formal_judgments:
            judge_replay_pairs.append((*formal_judgments.pop(0), judge_cursor))
        pending_judge_group = False

    for raw_record in service_journal:
        record = _event_payload(raw_record)
        service_data = record.get("service")
        # Retried requests can split one formal evaluation across resume.
        # An abandoned attempt remains auditable but does not end that group.
        if record.get("kind") == "service_abandoned":
            continue
        if record.get("kind") != "service_call" or not isinstance(service_data, dict):
            finish_formal_group()
            continue
        if record.get("role") == "judge":
            judge_cursor += 1
            pending_judge_group = True
            continue
        preview_event = (
            _decode_service_event(service_data)
            if record.get("role") == "oracle"
            else None
        )
        if preview_event is None or preview_event.kind != "judge.evaluate":
            finish_formal_group()
            continue
        if preview_event.error is not None:
            finish_formal_group()
            continue
        try:
            preview_ideas = tuple(
                Idea(
                    idea_id=str(value["idea_id"]),
                    content=value.get("content"),
                    probability=value.get("probability", 1),
                )
                for value in (preview_event.request.get("ideas") or ())
            )
            preview_verdicts = list(preview_event.response)
        except (KeyError, TypeError, ValueError) as exc:
            raise ReplayDivergence("Oracle Judge preview record is malformed") from exc
        judge_replay_pairs.append(
            (
                Submission(preview_ideas),
                {"verdicts": preview_verdicts},
                "active",
                judge_cursor,
            )
        )
        pending_judge_group = False
    finish_formal_group()

    # A deterministic/test Judge can emit a formal verdict without any Judge
    # service events. Such rounds have no tape ordering to reconstruct and are
    # replayed after the service-backed groups.
    judge_replay_pairs.extend(
        (*formal, judge_cursor) for formal in formal_judgments
    )
    formal_judgments.clear()

    # Legacy traces can carry their only trusted judgment on run_finished,
    # without a timestamped submission_judged marker.
    if replay_pairs and not any(
        _event_payload(value).get("kind") == "submission_judged" for value in private
    ):
        if len(replay_pairs) != 1:
            raise ReplayDivergence("legacy judgments have no unambiguous replay order")
        replayed_submission, expected = (
            _decode_submission(replay_pairs[0][0]["submission"]),
            replay_pairs[0][1],
        )
        judge_replay_pairs.append(
            (replayed_submission, expected, "active", judge_cursor)
        )

    if failed_service_context is not None:
        judge_path = root / "service-tape.judge.private.json"
        final_judge_tape = _load_service_tape(judge_path) if judge_path.is_file() else ()
        expected_judge_tape = (
            failed_service_context["judge_prefix"]
            + failed_service_context["tail_by_role"]["judge"]
        )
        if final_judge_tape != expected_judge_tape:
            raise ReplayDivergence(
                "Judge post-checkpoint service tail disagrees with its journal"
            )
        if "judge" in failed_service_context["expected_meter"]:
            try:
                usage_data = json.loads((root / "usage.json").read_text(encoding="utf-8"))
                usage_row = usage_data["judge"]
                meter = failed_service_context["expected_meter"]["judge"]
                summary_matches = (
                    int(usage_row.get("model_calls", -1)) == meter["model_calls"]
                    and int(usage_row.get("random_calls", -1)) == meter["random_calls"]
                    and int(usage_row.get("service_events", -1)) == len(final_judge_tape)
                    and _parse_model_usage(
                        usage_row.get("model_usage") or {}, source="Judge usage summary"
                    ) == failed_service_context["expected_usage"]["judge"]
                )
            except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                raise ReplayDivergence("failed run Judge usage summary is invalid") from exc
            if not summary_matches:
                raise ReplayDivergence(
                    "Judge usage summary disagrees with its service journal"
                )

    if judge_replay_pairs:
        judge_name = str(manifest_data.get("judge") or "")
        if judge_name == "sample-accept-all":
            from ..evaluation.sample import SampleSubmissionJudge

            replay_judge = SampleSubmissionJudge()
            replay_judges = {"active": replay_judge, "essence": replay_judge}
            judge_services = None
        elif judge_name == "smoke-answer-probe":
            from ..evaluation.smoke import SmokeAnswerJudge

            replay_judge = SmokeAnswerJudge()
            replay_judges = {"active": replay_judge, "essence": replay_judge}
            judge_services = None
        else:
            from ..evaluation.research import ResearchJudge

            judge_tape = _load_service_tape(root / "service-tape.judge.private.json")
            judge_services = ServiceFactory(
                seed=seed * 2 + 3,
                model_backend=model_backend,
                model_name=(manifest_data.get("models") or {}).get("judge"),
            ).create(judge_tape)
            judge_config = manifest_data.get("judge_config") or {}
            final_mode = judge_name.removeprefix("research-")
            # Reasons are a property of the recording, not of this run's
            # config: an inherited era may itself have been recorded with
            # coarse reasons on (a promoted run whose source already carried
            # the flag) or off (pre-flag runs). Both variants exist per mode
            # and the per-call tape peek picks the one that matches.
            era_judges = {
                (mode, reasons): ResearchJudge(
                    judge_services,
                    mode=mode,
                    fmn_m=judge_config.get("m") if mode == final_mode else None,
                    fmn_n=int(judge_config.get("n") or 0) if mode == final_mode else 0,
                    coarse_reasons=reasons,
                )
                for mode in {final_mode, "directional", "essence", "fmn"}
                for reasons in (False, True)
            }
            replay_judges = {
                "active": era_judges[
                    (final_mode, bool(judge_config.get("coarse_reasons")))
                ],
                "essence": era_judges[("essence", False)],
                "_by_era": era_judges,
            }

        def reason_value(value: Any) -> Any:
            text = str(value or "")
            for parser in (json.loads, ast.literal_eval):
                try:
                    return parser(text)
                except (ValueError, SyntaxError, json.JSONDecodeError):
                    continue
            return text

        def _next_recorded_era() -> tuple[str, bool] | None:
            """(era, coarse_reasons) of the next unconsumed Judge tape entry."""

            if judge_services is None:
                return None
            tape = getattr(judge_services, "_replay_tape", ())
            consumed = getattr(judge_services, "_replay_consumed", set())
            cursor = getattr(judge_services, "_replay_cursor", 0)
            while cursor in consumed:
                cursor += 1
            if cursor >= len(tape):
                return None
            request = getattr(tape[cursor], "request", None)
            if not isinstance(request, dict):
                return None
            schema = str(request.get("schema_name") or "")
            reasons = '"reason"' in json.dumps(request.get("schema") or {}, sort_keys=True)
            if "directional" in schema:
                return ("directional", reasons)
            if "essence" in schema:
                return ("essence", reasons)
            if "fmn" in schema or "motivation" in schema or "faithful" in schema:
                return ("fmn", reasons)
            return None

        for replayed_submission, expected, judge_profile, target_cursor in judge_replay_pairs:
            replayed_round_rows: list[tuple[IdeaVerdict, ...]] = []
            if judge_services is not None and target_cursor is not None:
                consumed = len(judge_tape) - judge_services.replay_remaining()
                if target_cursor < consumed:
                    raise ReplayDivergence("Judge replay markers are out of order")
                while consumed < target_cursor:
                    if judge_profile == "active":
                        era_judges = replay_judges.get("_by_era")
                        era_key = _next_recorded_era()
                        selected_judge = (
                            era_judges.get(era_key, replay_judges.get("active"))
                            if isinstance(era_judges, dict) and era_key is not None
                            else replay_judges.get("active")
                        )
                    else:
                        selected_judge = replay_judges.get(judge_profile)
                    if selected_judge is None:
                        raise ReplayDivergence(
                            f"judge preview uses unknown profile {judge_profile!r}"
                        )
                    replayed_round_rows.append(
                        tuple(selected_judge.evaluate(target, replayed_submission.ideas))
                    )
                    advanced = len(judge_tape) - judge_services.replay_remaining()
                    if advanced <= consumed or advanced > target_cursor:
                        raise ReplayDivergence(
                            "Judge evaluation crossed its recorded replay boundary"
                        )
                    consumed = advanced
                # Deterministic or test Judges may not use participant model
                # services at all, so their durable boundary legitimately has
                # zero Judge tape events. Replay the recorded number of rounds
                # directly in that case.
                if not replayed_round_rows:
                    selected_judge = replay_judges.get(judge_profile)
                    if selected_judge is None:
                        raise ReplayDivergence(
                            f"judge preview uses unknown profile {judge_profile!r}"
                        )
                    repeats = int(
                        expected.get("judge_repeats")
                        or (manifest_data.get("judge_config") or {}).get("repeats")
                        or 1
                    )
                    if not 1 <= repeats <= 64:
                        raise ReplayDivergence("recorded Judge repeat count is invalid")
                    replayed_round_rows.extend(
                        tuple(selected_judge.evaluate(target, replayed_submission.ideas))
                        for _ in range(repeats)
                    )
            else:
                selected_judge = replay_judges.get(judge_profile)
                if selected_judge is None:
                    raise ReplayDivergence(
                        f"judge preview uses unknown profile {judge_profile!r}"
                    )
                repeats = int(
                    expected.get("judge_repeats")
                    or (manifest_data.get("judge_config") or {}).get("repeats")
                    or 1
                )
                if not 1 <= repeats <= 64:
                    raise ReplayDivergence("recorded Judge repeat count is invalid")
                replayed_round_rows.extend(
                    tuple(selected_judge.evaluate(target, replayed_submission.ideas))
                    for _ in range(repeats)
                )
            replayed_rounds = tuple(replayed_round_rows)
            recorded_repeats = expected.get("judge_repeats")
            if recorded_repeats is not None and len(replayed_rounds) != int(recorded_repeats):
                raise ReplayDivergence("repeat Judge round count diverged during actor replay")
            replayed_verdicts = aggregate_repeated_verdicts(
                replayed_submission.ideas,
                replayed_rounds,
            )
            replayed_signatures = [
                (value.idea_id, value.passed, reason_value(value.private_reason))
                for value in replayed_verdicts
            ]
            recorded_signatures = [
                (
                    str(value.get("idea_id")),
                    bool(value.get("passed")),
                    reason_value(value.get("private_reason")),
                )
                for value in (expected.get("verdicts") or [])
            ]
            if replayed_signatures != recorded_signatures:
                raise ReplayDivergence("judge verdicts diverged during actor replay")
            recorded_rounds = expected.get("verdict_rounds")
            if recorded_rounds is not None:
                replayed_round_signatures = [
                    [
                        (value.idea_id, value.passed, reason_value(value.private_reason))
                        for value in verdicts
                    ]
                    for verdicts in replayed_rounds
                ]
                recorded_round_signatures = [
                    [
                        (
                            str(value.get("idea_id")),
                            bool(value.get("passed")),
                            reason_value(value.get("private_reason")),
                        )
                        for value in verdicts
                    ]
                    for verdicts in recorded_rounds
                ]
                if replayed_round_signatures != recorded_round_signatures:
                    raise ReplayDivergence("repeat Judge rounds diverged during actor replay")
        if judge_services is not None:
            judge_calls = len(judge_tape)
            remaining = judge_services.replay_remaining()
            allow_judge_tail = bool(
                failed_run
                and interrupted is not None
                and (interrupted.get("role") == "judge"
                     or (interrupted.get("role") == "oracle"
                         and failed_service_context is not None
                         and _bound_failed_judge_tail(service_records[service_cursor:])
                         and remaining == len(failed_service_context["tail_by_role"]["judge"])))
            )
            if allow_judge_tail and remaining:
                unreplayed_service_events["judge"] = remaining
            else:
                judge_services.finish_replay("judge-replay")
        judge_replayed = True
    elif failed_run:
        judge_path = root / "service-tape.judge.private.json"
        if judge_path.is_file():
            judge_tape = _load_service_tape(judge_path)
            judge_calls = len(judge_tape)
            if judge_tape:
                bound_guide_tail = bool(
                    interrupted is not None and interrupted.get("role") == "oracle"
                    and failed_service_context is not None
                    and _bound_failed_judge_tail(service_records[service_cursor:])
                    and len(judge_tape) == len(failed_service_context["tail_by_role"]["judge"]))
                if interrupted is None or (interrupted.get("role") != "judge" and not bound_guide_tail):
                    raise ReplayDivergence(
                        "Judge service tail has no interrupted Judge call"
                    )
                unreplayed_service_events["judge"] = len(judge_tape)
    return {
        "status": "actor-replayed",
        "run_id": manifest_data["run_id"],
        "calls": counts,
        "branches": branch_counts,
        "unreplayed_service_events": unreplayed_service_events,
        "judge_replayed": judge_replayed,
        "judge_calls": judge_calls,
    }
