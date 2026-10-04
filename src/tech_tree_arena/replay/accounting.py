"""Reprice validated active-path traces without changing their event chains."""

from __future__ import annotations

from collections import Counter
from decimal import Decimal
from typing import Any, Iterable

from ..contract.pricing import information_cost
from ..contract.recovery import CURRENT_ACCOUNTING, LEGACY_ACCOUNTING, choice_surcharge
from ..errors import ReplayDivergence


def reprice_events(events: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Project an already validated schema-4 trace into current accounting.

    Snapshots retain the exact earlier edge occurrences. Counters describe the
    chronological request stream and are never rewound by checkout. This is an
    accounting projection, not a verifier of stochastic-kernel assumptions.
    """
    events = tuple(events)
    if not events or events[0].get("protocol_event_schema") != 4:
        raise ReplayDivergence("repricing requires active-path protocol event schema 4")
    continuations: Counter[str] = Counter()
    options: Counter[tuple[str, str]] = Counter()
    snapshots: dict[str, tuple[float, tuple[dict[str, Any], ...]]] = {}
    active: tuple[dict[str, Any], ...] = ()
    path_k = 0.0
    submission_choice_bits = 0.0
    final_k = None
    status = "running"
    for position, event in enumerate(events):
        kind = event.get("kind")
        if kind == "question":
            snapshots[event["question_id"]] = (path_k, active)
            submission_choice_bits = 0.0
        elif kind == "choice_cost":
            question_id, option_id = event["question_id"], event["option_id"]
            continuations[question_id] += 1
            options[question_id, option_id] += 1
            i, j = continuations[question_id], options[question_id, option_id]
            bits = information_cost(Decimal(event["probability"]))
            surcharge = choice_surcharge(i, j)
            submission_choice_bits = bits + surcharge
            path_k += submission_choice_bits
            active = (*active, {
                "event_index": position, "question_id": question_id,
                "option_id": option_id, "continuation_index": i, "option_index": j,
                "information_bits": bits, "branch_bits": surcharge,
            })
        elif kind == "checkout":
            path_k, active = snapshots[event["target_question_id"]]
            submission_choice_bits = 0.0
        elif kind == "submission_judged" and event.get("status") == "pass":
            final_k = path_k + float(event["submission_bits"]) + float(event.get("repeat_bits") or 0.0)
        elif kind == "run_finished":
            status = event["status"]
            if status != "pass":
                final_k = None
    return {
        "accounting_version": CURRENT_ACCOUNTING,
        "status": status, "K": final_k,
        "score": 0.0 if final_k is None else 2.0 ** -final_k,
        "path_k": path_k,
        "node_costs": {key: value[0] for key, value in snapshots.items()},
        "submission_choice_bits": submission_choice_bits,
        "active_choices": list(active),
    }


def repriced_score(original: dict[str, Any], recorded: dict[str, Any],
                   events: tuple[dict[str, Any], ...]) -> dict[str, Any]:
    """Canonical score document derived exclusively from verified evidence."""
    priced = reprice_events(events)
    return {
        **original,
        "K": priced["K"], "score": priced["score"],
        "accounting_version": CURRENT_ACCOUNTING,
        "repriced_from": {
            "accounting_version": recorded.get("accounting_version", LEGACY_ACCOUNTING),
            "K": recorded["K"], "score": recorded["score"],
            "private_event_hash": events[-1]["event_hash"],
        },
    }


def migrate_run(run_directory: Any, *, write: bool = False) -> dict[str, Any]:
    """Verify and atomically replace a completed run's canonical score.

    The original score, events, branches and checkpoints remain available for
    historical protocol/actor replay. Repeated migrations verify the prior
    result and do not apply another surcharge. Live runs are never migrated.
    """
    import json
    import os
    from pathlib import Path

    from .recorder import _atomic_json, protocol_replay, verify_hash_chain

    root = Path(run_directory).resolve()
    lock = root / ".accounting-migration.lock"
    if write:
        # Separate migrations must not interleave score archival/replacement.
        with lock.open("x", encoding="utf-8") as stream:
            stream.write(str(os.getpid()))
    try:
        score_path = root / "score.json"
        original_bytes = score_path.read_bytes()
        stored = json.loads(original_bytes)
        status_path = root / "status.json"
        status = json.loads(status_path.read_text()) if status_path.is_file() else {}
        if status.get("state") == "running" or status.get("status") == "running":
            raise ReplayDivergence("cannot migrate a running record")
        replay = protocol_replay(root)
        recorded = replay["recorded_result"]
        events = verify_hash_chain(root / "events.private.jsonl")
        if events[0].get("protocol_event_schema") != 4:
            raise ReplayDivergence("repricing requires active-path protocol event schema 4")
        if recorded["status"] not in {"pass", "error"}:
            raise ReplayDivergence("repricing requires a terminal record")
        archived_path = root / "score.recorded.json"
        original = (
            json.loads(archived_path.read_text()) if "repriced_from" in stored else stored
        )
        changed = stored.get("accounting_version") != CURRENT_ACCOUNTING
        new_score = repriced_score(original, recorded, events) if changed else stored
        projected = reprice_events(events)
        if write and changed:
            # Detect concurrent writes before replacing any canonical data.
            if score_path.read_bytes() != original_bytes:
                raise ReplayDivergence("score changed during migration")
            if archived_path.exists():
                if archived_path.read_bytes() != original_bytes:
                    raise ReplayDivergence("recorded score archive already differs")
            else:
                with archived_path.open("xb") as stream:
                    stream.write(original_bytes)
                    stream.flush()
                    os.fsync(stream.fileno())
            _atomic_json(score_path, new_score)
            protocol_replay(root)
        old_k, new_k = recorded["K"], projected["K"]
        return {
            "run_dir": str(root), "run_id": replay["run_id"],
            "status": recorded["status"],
            "action": "updated" if write and changed else "would_update" if changed else "current",
            "accounting_version": CURRENT_ACCOUNTING,
            "recorded_K": old_k, "K": new_k,
            "delta_bits": None if old_k is None or new_k is None else new_k - old_k,
            "retained_choices": len(projected["active_choices"]),
            "has_generator_stage_transition": any(
                event.get("kind") == "stage_transition" and event.get("role") == "generator"
                for event in events
            ),
            "interpretation": "accounting projection; stochastic-kernel and evaluation assumptions require separate verification",
        }
    finally:
        if write:
            lock.unlink()


def migrate_tree(root_directory: Any, *, write: bool = False) -> dict[str, Any]:
    """Batch migration with explicit per-record failures, defaulting to dry run."""
    import json
    from pathlib import Path

    root = Path(root_directory).resolve()
    paths = [root / "score.json"] if (root / "score.json").is_file() else sorted(root.rglob("score.json"))
    results, errors = [], []
    for path in paths:
        if not (path.parent / "events.private.jsonl").is_file():
            continue
        try:
            results.append(migrate_run(path.parent, write=write))
        except (OSError, ValueError, KeyError, ReplayDivergence, json.JSONDecodeError) as exc:
            errors.append({"run_dir": str(path.parent), "error": str(exc)})
    return {"write": write, "accounting_version": CURRENT_ACCOUNTING, "runs": results, "errors": errors}
