"""The auditor must recognize the submit8 bundle and reject changed cached slates."""
import copy
import importlib.util
import json
from pathlib import Path

import pytest

from tech_tree_arena.replay.recorder import HashChainWriter

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("submit8_audit", ROOT / "tools/audit_codex_eager_run.py")
audit_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit_module)


@pytest.mark.parametrize("change", [None, "submitted_content", "bundle_hash", "candidate_count"])
def test_exact_submit8_activation_and_tampering(tmp_path, change):
    root = tmp_path / "run"
    root.mkdir()
    (root / "manifest.json").write_text(json.dumps({"run_id": "synthetic",
        "submission_path": str(ROOT / "submissions/reference_pair_submit8")}))
    slate = {"ideas": [{"idea_id": str(i), "content": {"setting_and_object": f"Synthetic proposal {i}",
        "findings": []}, "probability": "0.125"} for i in range(8)]}
    if change == "candidate_count":
        slate["ideas"].pop()
    source = {"source_stage": "directional", "source_draft": "synthetic", "fact_ledger_hash": "ledger"}
    bundle = audit_module.digest({**source, "question_hashes": {}, "submission_hash": audit_module.digest(slate)})
    payload = {"kind": "dispatch", "mode": "submit", **source,
        "bundle_id": "changed" if change == "bundle_hash" else bundle, "preview": {"submission": slate}}
    writer = HashChainWriter(root / "events.private.jsonl")
    writer.append({"kind": "question", "question_id": "dispatch", "question": {"question": "Choose route",
        "options": [{"option_id": "submit", "probability": "1", "kind": "submit", "public_payload": payload}]}})
    writer.append({"kind": "choice_cost", "question_id": "dispatch", "option_id": "submit"})
    submitted = copy.deepcopy(slate)
    if change == "submitted_content":
        submitted["ideas"][0]["content"]["setting_and_object"] = "Changed after Oracle selection"
    writer.append({"kind": "submission", "source_question_id": "dispatch", "submission": submitted})
    report = audit_module.audit(root, expected_backend="common-memory-v1")
    assert bool(report["problems"]) == bool(change)
    if change is None:
        assert report["activations"] == [{"dispatch_id": "dispatch", "mode": "submit", "idea_count": 8,
            "exact_cached_submission": True, "generator_calls_during_activation": 0,
            "call_count_source": "timestamp"}]
