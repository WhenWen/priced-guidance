import json
import math
from functools import partial
from pathlib import Path

import pytest

from tech_tree_arena.cli import _run_submission
from tech_tree_arena.contract.recovery import (
    CURRENT_ACCOUNTING, LEGACY_ACCOUNTING, choice_surcharge, occurrence_bits,
)
from tech_tree_arena.errors import ReplayDivergence
from tech_tree_arena.replay.accounting import migrate_run, reprice_events
from tech_tree_arena.replay.recorder import (
    HashChainWriter, _event_payload, _replay_protocol_events,
    actor_replay, protocol_replay, verify_hash_chain,
)
from tech_tree_arena.runtime import engine
from tech_tree_arena.runtime.branch import BranchStore


ROOT = Path(__file__).resolve().parents[2]


def test_prior_is_normalized_with_exact_tail():
    for n in (1, 2, 100, 10000):
        mass = math.fsum(2.0 ** -occurrence_bits(j) for j in range(1, n + 1))
        assert mass + 0.05 / n == pytest.approx(1.0, abs=1e-14)
    assert occurrence_bits(1) == pytest.approx(-math.log2(0.95))
    assert occurrence_bits(2) == pytest.approx(math.log2(40))
    assert math.isfinite(occurrence_bits(10**1000))


@pytest.mark.parametrize("index", [0, -1, 1.0, True, None])
def test_occurrence_index_is_a_positive_integer(index):
    with pytest.raises(ValueError):
        occurrence_bits(index)


def test_new_surcharge_replaces_the_old_formula_without_a_floor():
    assert choice_surcharge(100, 1) == occurrence_bits(1)
    assert choice_surcharge(100, 1) < choice_surcharge(100, 1, version=LEGACY_ACCOUNTING)
    assert choice_surcharge(100, 2) == choice_surcharge(2, 2)
    with pytest.raises(ValueError):
        choice_surcharge(1, 2)
    with pytest.raises(ValueError):
        choice_surcharge(1, 1, version="unknown")


def _choice(q, a, probability="1"):
    return {"kind": "choice_cost", "question_id": q, "option_id": a, "probability": probability}


def test_checkout_restores_the_exact_saved_occurrence_not_the_latest_edge():
    events = [
        {"kind": "run_started", "protocol_event_schema": 4},
        {"kind": "question", "question_id": "root"},
        _choice("root", "a"),
        {"kind": "question", "question_id": "old_child"},
        _choice("old_child", "b"),
        {"kind": "question", "question_id": "old_grandchild"},
        {"kind": "checkout", "target_question_id": "root"},
        _choice("root", "a"),
        {"kind": "question", "question_id": "new_child"},
        # Return to a checkpoint on an abandoned branch after reusing its
        # ancestor option. Its saved prefix still contains occurrence one.
        {"kind": "checkout", "target_question_id": "old_child"},
        _choice("old_child", "b"),
    ]
    priced = reprice_events(events)
    assert [edge["option_index"] for edge in priced["active_choices"]] == [1, 2]
    assert priced["path_k"] == pytest.approx(occurrence_bits(1) + occurrence_bits(2))
    assert priced["node_costs"]["new_child"] == pytest.approx(occurrence_bits(2))
    assert priced["node_costs"]["old_grandchild"] == pytest.approx(2 * occurrence_bits(1))


def _legacy_smoke(tmp_path, monkeypatch):
    with monkeypatch.context() as patch:
        patch.setattr(engine, "CURRENT_ACCOUNTING", LEGACY_ACCOUNTING)
        patch.setattr(engine, "BranchStore", partial(BranchStore, accounting_version=LEGACY_ACCOUNTING))
        result = _run_submission(
            ROOT / "submissions/examples/minimal_pair", "smoke", None, 1,
            runs_dir=tmp_path,
        )
    return Path(result["run_dir"])


def test_migration_is_verified_idempotent_and_preserves_audit_evidence(tmp_path, monkeypatch):
    root = _legacy_smoke(tmp_path, monkeypatch)
    paths = [root / name for name in (
        "events.private.jsonl", "events.public.jsonl", "branches.json",
        "promotion-checkpoint.private.json", "manifest.json",
    )]
    evidence = {path: path.read_bytes() for path in paths}
    before = (root / "score.json").read_bytes()
    preview = migrate_run(root)
    assert preview["action"] == "would_update"
    assert preview["delta_bits"] == pytest.approx(occurrence_bits(1))
    assert (root / "score.json").read_bytes() == before
    report = migrate_run(root, write=True)
    assert report["action"] == "updated"
    assert (root / "score.recorded.json").read_bytes() == before
    assert all(path.read_bytes() == value for path, value in evidence.items())
    after = (root / "score.json").read_bytes()
    assert migrate_run(root, write=True)["action"] == "current"
    assert (root / "score.json").read_bytes() == after
    replayed = protocol_replay(root)
    assert replayed["result"]["K"] == pytest.approx(1 + occurrence_bits(1))
    assert replayed["recorded_result"]["K"] == 1
    assert actor_replay(root)["status"] == "actor-replayed"

    tampered = json.loads(after)
    tampered["repriced_from"]["private_event_hash"] = "0" * 64
    (root / "score.json").write_text(json.dumps(tampered))
    with pytest.raises(ReplayDivergence, match="provenance"):
        protocol_replay(root)


def test_migration_rejects_tampering_before_writing(tmp_path, monkeypatch):
    root = _legacy_smoke(tmp_path, monkeypatch)
    before = (root / "score.json").read_bytes()
    path = root / "events.private.jsonl"
    path.write_text(path.read_text().replace('"branch_bits":0.0', '"branch_bits":1.0', 1))
    with pytest.raises(ReplayDivergence, match="hash mismatch"):
        migrate_run(root, write=True)
    assert (root / "score.json").read_bytes() == before
    assert not (root / "score.recorded.json").exists()
    assert not (root / ".accounting-migration.lock").exists()


def test_native_v2_run_needs_no_migration(tmp_path):
    result = _run_submission(ROOT / "submissions/examples/minimal_pair", "smoke", None, 1, runs_dir=tmp_path)
    root = Path(result["run_dir"])
    assert migrate_run(root, write=True)["action"] == "current"
    assert not (root / "score.recorded.json").exists()
    assert protocol_replay(root)["result"]["accounting_version"] == CURRENT_ACCOUNTING


def test_historical_preview_is_validated_and_remains_private(tmp_path, monkeypatch):
    root = _legacy_smoke(tmp_path, monkeypatch)
    path = root / "events.private.jsonl"
    events = [_event_payload(event) for event in verify_hash_chain(path)]
    question = next(event for event in events if event["kind"] == "question")
    submitted = next(event for event in events if event["kind"] == "submission")
    judged = next(event for event in events if event["kind"] == "submission_judged")
    preview = {
        "kind": "judge_previewed", "question_id": question["question_id"],
        "path_k": question["path_k"], "submission": submitted["submission"],
        "verdicts": judged["verdicts"], "status": "pass", "judge_profile": "active",
    }
    index = next(i for i, event in enumerate(events) if event["kind"] == "oracle_decision")
    events.insert(index, preview)
    # Only a test fixture is rewritten. The existing public chain stays valid
    # because historical previews were always private.
    path.unlink()
    writer = HashChainWriter(path)
    for event in events:
        writer.append(event)
    assert protocol_replay(root)["status"] == "replayed"
    assert migrate_run(root, write=True)["action"] == "updated"
    preview["status"] = "fail"
    with pytest.raises(ReplayDivergence, match="preview status"):
        _replay_protocol_events(tuple(events))


def test_migration_refuses_running_records(tmp_path, monkeypatch):
    root = _legacy_smoke(tmp_path, monkeypatch)
    status_path = root / "status.json"
    status = json.loads(status_path.read_text())
    status["status"] = "running"
    status_path.write_text(json.dumps(status))
    with pytest.raises(ReplayDivergence, match="running record"):
        migrate_run(root, write=True)
    assert not (root / "score.recorded.json").exists()
