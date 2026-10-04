#!/usr/bin/env python3
"""Audit eager dispatch and account usage from recorded evidence; print no target text."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from decimal import Decimal, localcontext
import hashlib
import json
import threading
from pathlib import Path
from types import SimpleNamespace

from tech_tree_arena.contract.pricing import validate_distribution
from tech_tree_arena.errors import ValidationError
from tech_tree_arena.replay.recorder import verify_hash_chain
from tech_tree_arena.replay.artifacts import ArtifactStore
from tech_tree_arena.submission_io.manifest import load_manifest, load_participant_classes


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                   separators=(",", ":")).encode()).hexdigest()


def normalize_question(value):
    return {"question": value["question"], "options": [
        {**option, "kind": option.get("kind", "ordinary")} for option in value["options"]]}


def valid_preview_distribution(rows):
    # Use the same serialization tolerance as the frozen question contract.
    try:
        with localcontext() as precision:
            precision.prec = 100
            validate_distribution(SimpleNamespace(probability=row["probability"]) for row in rows)
        return True
    except (ValidationError, KeyError, TypeError):
        return False


def valid_submit8_preview(preview, stage):
    # The frozen submit8 participant expands only Directional submissions.
    expected = {"directional": 8, "essence": 1, "strict": 1}.get(stage)
    return expected is not None and len(preview.get("submission", {}).get("ideas", [])) == expected


def audit_common_native(metadata, artifacts, expected_model, expected_effort):
    checks, problems = [], []
    for op in metadata.get("operations", []):
        inner = op["metadata"]
        check = {"role": op["role"], "error": op.get("error"), "backend": inner.get("backend")}
        if inner.get("backend") == "native-api-history-v1":
            body = artifacts.load_json(inner["request_body"])
            check.update(reasoning_blocks_inherited=inner.get("reasoning_blocks_inherited", 0),
                reasoning_blocks_returned=sum(a.get("reasoning_blocks_returned", 0) for a in inner.get("attempts", [])),
                actual_model=body.get("model"), actual_effort=(body.get("output_config") or {}).get("effort"))
            if expected_model.startswith("anthropic/") and (check["actual_model"] != expected_model.removeprefix("anthropic/")
                    or check["actual_effort"] != expected_effort
                    or body.get("thinking", {}).get("block_binding", {}).get("prefix_mismatch_behavior") != "error"):
                problems.append("Native API model, effort or thinking binding differs")
        elif inner.get("codex_context"):
            context, parent = inner["codex_context"], inner.get("parent_context")
            def records(ctx):
                return [json.loads(line) for line in artifacts.load_json(ctx["rollout"])["rollout"].splitlines()]
            native = records(context)
            def reasoning(rows):
                return Counter(digest(r["payload"]["encrypted_content"]) for r in rows
                    if r.get("type") == "response_item" and r.get("payload", {}).get("type") == "reasoning"
                    and r["payload"].get("encrypted_content"))
            before, after = reasoning(records(parent)) if parent else Counter(), reasoning(native)
            turns = [r["payload"] for r in native if r.get("type") == "turn_context"]
            turn = turns[-1] if turns else {}
            compacted = any(r.get("type") == "compacted" or r.get("payload", {}).get("type") == "context_compacted" for r in native)
            complete = any(r.get("type") == "event_msg" and r.get("payload", {}).get("type") == "task_complete"
                           and r["payload"].get("turn_id") == context["turn_id"] for r in native[-5:])
            check.update(thread_action=inner.get("thread_action"), parent_reasoning_items=sum(before.values()),
                native_reasoning_items=sum(after.values()), parent_reasoning_preserved=not bool(before - after),
                cache_lineage_preserved=not parent or parent.get("cache_lineage") == context.get("cache_lineage"),
                completed_turn_checkpoint=complete, actual_model=turn.get("model"), actual_effort=turn.get("effort"),
                native_compacted=compacted)
            if (before - after or not check["cache_lineage_preserved"] or not complete or compacted
                    or turn.get("model") != expected_model or turn.get("effort") != expected_effort):
                problems.append("Codex native history, checkpoint, model or effort differs")
        checks.append(check)
    return checks, problems


_AUDIT_LOCK = threading.RLock()


def activation_cursors(root):
    """Use actor service cursors, which survive resume timestamp rewriting."""
    checkpoint_path = root / "checkpoint.private.json"
    if checkpoint_path.exists():
        checkpoint = json.loads(checkpoint_path.read_text())
        streams = checkpoint.get("actor_streams", {}).get("generator", {}).values()
    else:
        path = root / "actor-branches.generator.private.json"
        streams = json.loads(path.read_text()) if path.exists() else []
    result = defaultdict(list)
    for stream in streams:
        previous = 0
        for call in stream.get("calls", []):
            count = call["service_event_count"]
            message = call.get("message") or {}
            value = message.get("value") or {}
            payload = value.get("public_payload") or {}
            if message.get("type") == "choice" and payload.get("kind") == "dispatch":
                result[(payload.get("bundle_id"), value.get("option_id"))].append(count - previous)
            previous = count
    return result


def audit(root: Path, **kwargs):
    # Participant loading replaces global sys.modules entries and sys.path.
    # Concurrent audits in one supervisor must not mutate that state together.
    with _AUDIT_LOCK:
        return _audit(root, **kwargs)


def _audit(root: Path, *, expected_effort="xhigh", expected_model="gpt-6-astra",
           expected_backend="codex-source-v1"):
    manifest = json.loads((root / "manifest.json").read_text())
    events = verify_hash_chain(root / "events.private.jsonl", tolerate_truncated_tail=True)
    path = root / "service-calls.private.jsonl"
    services = verify_hash_chain(path, tolerate_truncated_tail=True) if path.exists() else []
    # This is the frozen, target-independent formatter used in the actual run.
    participant_manifest = load_manifest(manifest["submission_path"])
    generator, _ = load_participant_classes(participant_manifest)
    render = generator.__init__.__globals__["_render_options"]
    questions = {event["question_id"]: event for event in events if event["kind"] == "question"}
    dispatches = {}
    problems = []
    route_sizes = defaultdict(list)
    retry_only = []
    guide_visible = 0
    for qid, event in questions.items():
        options = event["question"]["options"]
        if not options or (options[0].get("public_payload") or {}).get("kind") != "dispatch":
            continue
        dispatches[qid] = event
        first = options[0]["public_payload"]
        hashes = {}
        for option in options:
            payload = option["public_payload"]
            mode = payload["mode"]
            if mode == "submit":
                continue
            preview = payload.get("preview") or {}
            rows = preview.get("options") or []
            route_sizes[mode].append(len(rows))
            if len(rows) == 1:
                retry_only.append({"question_id": qid, "mode": mode})
            if (not preview.get("question") or not rows
                    or len({r["option_id"] for r in rows}) != len(rows)
                    or not valid_preview_distribution(rows)):
                problems.append(f"incomplete or invalid {mode} preview at {qid}")
            hashes[mode] = digest(preview)
        source = {key: first[key] for key in ("source_stage", "source_draft", "fact_ledger_hash")}
        source["question_hashes"] = hashes
        submit_options = [o for o in options if o["public_payload"]["mode"] == "submit"]
        for option in submit_options:
            preview = option["public_payload"].get("preview") or {}
            if "submission" in preview:
                source["submission_hash"] = digest(preview["submission"])
                if participant_manifest.name == "reference-pair-submit8" and not valid_submit8_preview(preview, first.get("source_stage")):
                    problems.append(f"submit preview idea count differs from stage policy at {qid}")
        if any(option["public_payload"]["bundle_id"] != digest(source) for option in options):
            problems.append(f"bundle binding mismatch at {qid}")
        guide_calls = [row for row in services if row["role"] == "oracle"
                        and f"question_id: {qid}\n" in str(row["service"].get("request", {}).get("user", ""))]
        if guide_calls:
            # The ordinary routes cannot be hidden by the failed-submit guard.
            visible = all(any(render([{"option_id": option["option_id"],
                "probability": option["probability"], "option_type": "answer",
                "public_payload": option["public_payload"]}]) in row["service"]["request"]["user"]
                for row in guide_calls) for option in options if option["public_payload"]["mode"] != "submit")
            if not visible:
                problems.append(f"Oracle did not receive exact complete previews at {qid}")
            else:
                guide_visible += 1

    activations = []
    cursor_counts = activation_cursors(root)
    for index, event in enumerate(events):
        if event["kind"] != "choice_cost" or event["question_id"] not in dispatches:
            continue
        qid = event["question_id"]
        option = next(o for o in dispatches[qid]["question"]["options"] if o["option_id"] == event["option_id"])
        payload = option["public_payload"]
        if payload["mode"] == "submit":
            expected = (payload.get("preview") or {}).get("submission")
            if expected is None:
                continue  # Legacy one-idea versions did not precompute slates.
            submitted = next((e for e in events[index + 1:] if e["kind"] == "submission"
                              and e.get("source_question_id") == qid), None)
            if submitted is not None:
                extra = [s for s in services if s["role"] == "generator"
                         and event["recorded_at"] <= s["recorded_at"] <= submitted["recorded_at"]]
                counts = cursor_counts.get((payload.get("bundle_id"), event["option_id"]))
                extra_count = max(counts) if counts else len(extra)
                equal = submitted["submission"] == expected
                activations.append({"dispatch_id": qid, "mode": "submit", "idea_count": len(expected["ideas"]),
                    "exact_cached_submission": equal, "generator_calls_during_activation": extra_count,
                    "call_count_source": "actor_cursor" if counts else "timestamp"})
                if not equal or extra_count:
                    problems.append(f"cached submission activation failed at {qid}")
            continue
        downstream = next((e for e in events[index + 1:] if e["kind"] == "question"
                           and e.get("parent_question_id") == qid), None)
        if downstream is None:
            continue  # A live run may still be inside activation.
        equal = normalize_question(downstream["question"]) == normalize_question(payload["preview"])
        extra_calls = [s for s in services if s["role"] == "generator"
                       and event["recorded_at"] <= s["recorded_at"] <= downstream["recorded_at"]]
        counts = cursor_counts.get((payload.get("bundle_id"), event["option_id"]))
        extra_count = max(counts) if counts else len(extra_calls)
        row = {"dispatch_id": qid, "mode": payload["mode"], "downstream_id": downstream["question_id"],
               "exact_cached_question": equal, "generator_calls_during_activation": extra_count,
               "call_count_source": "actor_cursor" if counts else "timestamp"}
        activations.append(row)
        if not equal or extra_count:
            problems.append(f"cached preview activation failed at {qid}")

    checkpoint_path = root / "checkpoint.private.json"
    checkpoint = json.loads(checkpoint_path.read_text()) if checkpoint_path.exists() else {}
    cursor_activations = []
    seen = set()
    for stream in checkpoint.get("actor_streams", {}).get("generator", {}).values():
        previous = 0
        for call in stream["calls"]:
            count = call["service_event_count"]
            message = call.get("message") or {}
            value = message.get("value") or {}
            payload = value.get("public_payload") or {}
            if (message.get("type") == "choice" and payload.get("kind") == "dispatch"
                    and (payload.get("mode") != "submit" or "submission" in (payload.get("preview") or {}))):
                identity = (payload.get("bundle_id"), value.get("option_id"), call["output_hash"])
                if identity not in seen:
                    seen.add(identity)
                    cursor_activations.append(count - previous)
                    if count != previous:
                        problems.append("actor cursor records a model call while activating an eager preview")
            previous = count

    usage = defaultdict(lambda: defaultdict(float))
    schemas = Counter()
    native_calls = 0
    native_efforts = Counter()
    native_context_checks = []
    common_native_checks = []
    compactions = set()
    per_call = []
    usage_by_action = defaultdict(lambda: defaultdict(float))
    latest_quota = None
    artifacts = ArtifactStore(root.parent)
    for row in services:
        service = row["service"]
        for key, value in (service.get("metadata", {}).get("usage") or {}).items():
            if isinstance(value, (int, float)):
                usage[row["role"]][key] += value
        if row["role"] != "generator":
            continue
        request, metadata = service["request"], service["metadata"]
        if expected_backend == "common-memory-v1":
            from tech_tree_arena.runtime.memory_output_budget import validate_metadata
            validate_metadata(service, artifacts)
            checks, errors = audit_common_native(metadata, artifacts, expected_model, expected_effort)
            common_native_checks.extend(checks)
            problems.extend(errors)
        schemas[request.get("schema_name")] += 1
        native_calls += 1
        call_usage = metadata.get("usage") or {}
        action = metadata.get("thread_action", "unknown")
        for key, value in call_usage.items():
            if isinstance(value, (int, float)):
                usage_by_action[action][key] += value
        per_call.append({"index": native_calls, "recorded_at": row["recorded_at"],
                         "schema": request.get("schema_name"), "thread_action": action,
                         "error_type": metadata.get("error_type"),
                         "latency_s": metadata.get("latency_s"),
                         "native_turn_timeout_seconds": metadata.get("native_turn_timeout_seconds", 300),
                         "usage": call_usage,
                         "cache_hit_percent": round(100 * call_usage.get("cached_input_tokens", 0)
                             / call_usage["input_tokens"], 2) if call_usage.get("input_tokens") else 0})
        for event in metadata.get("codex_events", []):
            if event.get("method") == "account/rateLimits/updated":
                latest_quota = event["params"]["rateLimits"]
        context = metadata.get("codex_context")
        if context:
            snapshot = artifacts.load_json(context["rollout"])
            native = [json.loads(line) for line in snapshot["rollout"].splitlines()]
            compactions.update(digest(r) for r in native if r.get("type") == "compacted"
                               or r.get("payload", {}).get("type") == "context_compacted")
            turn_contexts = [r["payload"] for r in native if r.get("type") == "turn_context"]
            actual = turn_contexts[-1] if turn_contexts else {}
            native_efforts[actual.get("effort", "missing")] += 1
            if actual.get("effort") != expected_effort or actual.get("model") != "gpt-6-astra":
                problems.append("native rollout model or effort differs from frozen protocol")
            def reasoning_fingerprints(records):
                return Counter(digest(r["payload"]["encrypted_content"]) for r in records
                               if r.get("type") == "response_item"
                               and r.get("payload", {}).get("type") == "reasoning"
                               and r["payload"].get("encrypted_content"))
            child_reasoning = reasoning_fingerprints(native)
            parent = metadata.get("parent_context")
            parent_reasoning = Counter()
            if parent:
                parent_snapshot = artifacts.load_json(parent["rollout"])
                parent_reasoning = reasoning_fingerprints(
                    [json.loads(line) for line in parent_snapshot["rollout"].splitlines()])
            inherited = not (parent_reasoning - child_reasoning)
            same_lineage = not parent or parent.get("cache_lineage") == context.get("cache_lineage")
            complete = any(r.get("type") == "event_msg"
                           and r.get("payload", {}).get("type") == "task_complete"
                           and r["payload"].get("turn_id") == context["turn_id"] for r in native[-5:])
            native_context_checks.append({"index": native_calls,
                "parent_encrypted_reasoning_items": sum(parent_reasoning.values()),
                "native_encrypted_reasoning_items": sum(child_reasoning.values()),
                "parent_reasoning_preserved": inherited, "cache_lineage_preserved": same_lineage,
                "completed_turn_checkpoint": complete})
            if not inherited or not same_lineage or not complete:
                problems.append("native context inheritance or completed checkpoint check failed")
        if (metadata.get("backend") != expected_backend or request.get("model") != expected_model
                or (expected_backend == "codex-source-v1" and request.get("reasoning_effort") != expected_effort)):
            problems.append("Generator model, effort, or backend differs from frozen protocol")
        for operation in metadata.get("operations", []):
            if operation.get("request", {}).get("reasoning_effort") != expected_effort:
                problems.append("Native operation effort differs from frozen protocol")
    generated = usage["generator"]
    from tech_tree_arena.runtime.codex_rpc import standard_credit_equivalent
    credit_estimate = standard_credit_equivalent(generated) if expected_model == "gpt-6-astra" else None
    score = json.loads((root / "score.json").read_text()) if (root / "score.json").exists() else None
    return {"run_id": manifest["run_id"], "target_id": manifest.get("target_id"),
            "status": score.get("status") if score else "running", "K": score.get("K") if score else None,
            "generator_native_calls": native_calls, "generator_schemas": dict(schemas),
            "native_rollout_efforts": dict(native_efforts), "latest_account_quota": latest_quota,
            "native_context_checks": native_context_checks,
            "common_native_checks": common_native_checks,
            "native_compaction_records": len(compactions),
            "usage_by_role": {role: dict(value) for role, value in usage.items()},
            "generator_calls": per_call,
            "generator_usage_by_thread_action": {action: dict(value) for action, value in usage_by_action.items()},
            "generator_cache_hit_percent": round(100 * generated.get("cached_input_tokens", 0)
                / generated["input_tokens"], 2) if generated.get("input_tokens") else 0,
            "estimated_standard_credits": credit_estimate,
            "credit_estimate_source": "https://learn.chatgpt.com/docs/pricing#what-are-tokens-and-credits",
            "credit_estimate_note": "Token-rate equivalent, not purchased-credit deduction; reported cache writes use the API guide's 1.25x input premium as an estimate. Account snapshots are separate.",
            "dispatches": len(dispatches), "dispatches_with_complete_previews_in_oracle_prompt": guide_visible,
            "preview_sizes_by_route": dict(route_sizes), "retry_only_previews": retry_only,
            "activations": activations, "actor_cursor_activation_service_deltas": cursor_activations,
            "pending_dispatches_without_oracle_call": len(dispatches) - guide_visible,
            "problems": problems}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--expected-model", default="gpt-6-astra")
    parser.add_argument("--expected-effort", default="xhigh")
    parser.add_argument("--expected-backend", default="codex-source-v1")
    args = parser.parse_args()
    report = audit(args.run_dir.resolve(), expected_model=args.expected_model,
                   expected_effort=args.expected_effort, expected_backend=args.expected_backend)
    text = json.dumps(report, indent=2) + "\n"
    if args.output:
        args.output.write_text(text)
    print(text, end="")
    return int(bool(report["problems"]))


if __name__ == "__main__":
    raise SystemExit(main())
