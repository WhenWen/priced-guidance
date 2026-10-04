from pathlib import Path

from tech_tree_arena.replay.recorder import RunRecorder, verify_hash_chain


def test_hidden_public_trace_releases_only_start_and_aggregate_finish(tmp_path: Path) -> None:
    recorder = RunRecorder(
        tmp_path,
        "run",
        {"target_id": "secret-target", "target_sha256": "private"},
        disclosure="hidden",
    )
    recorder.record({"kind": "run_started", "run_id": "run", "time_travel": True, "seed": 7})
    recorder.record({"kind": "question", "question": "secret-looking participant text"})
    recorder.record({"kind": "oracle_decision", "decision": "private"})
    recorder.record({"kind": "run_finished", "status": "pass", "score": 0.5, "k": 1.0})

    public = verify_hash_chain(recorder.root / "events.public.jsonl")
    assert [event["kind"] for event in public] == ["run_started", "run_finished"]
    assert "seed" not in public[0] and "k" not in public[1]
    assert "secret" not in (recorder.root / "events.public.jsonl").read_text()
