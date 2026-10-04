"""Native common-memory calls through the OS worker, tape replay, and recovery."""
import copy
import json
import shutil
from pathlib import Path

import pytest

from tech_tree_arena.cli import _resume_submission, _run_submission
from tech_tree_arena.errors import ParticipantFailure
from tech_tree_arena.replay.recorder import actor_replay, protocol_replay
from tech_tree_arena.runtime.common_memory import configuration
from tech_tree_arena.runtime.native_api_generator import NativeAPIGeneratorBackend, NativeCallError

ROOT = Path(__file__).resolve().parents[2]
MODEL = "anthropic/claude-fable-5-1"

@pytest.fixture
def pair(tmp_path, monkeypatch):
    root = tmp_path / "pair"
    shutil.copytree(ROOT / "submissions/examples/minimal_pair", root)
    path = root / "participant/generator.py"
    path.write_text(path.read_text() + '''
BaseGenerator = Generator
class Generator(BaseGenerator):
    def step(self, message):
        self.services.structured_model(developer="synthetic", user=str(message),
            schema_name="synthetic", schema={"type":"object"})
        return super().step(message)
''')
    bodies = []
    failures = set()
    def send(self, body):
        bodies.append(copy.deepcopy(body))
        if len(bodies) in failures:
            raise NativeCallError("synthetic unavailable credential", retryable=False)
        output = {"memory": "Synthetic previous work"} if "common_memory_summary" in body["messages"][-1]["content"] else {"ok": True}
        return {"content": [{"type": "thinking", "thinking": "synthetic", "signature": "signature"},
                            {"type": "text", "text": json.dumps({"payload_json": json.dumps(output)})}],
                "stop_reason": "end_turn", "usage": {"input_tokens": 20, "output_tokens": 10}}
    monkeypatch.setattr(NativeAPIGeneratorBackend, "_send", send)
    return root, bodies, failures


def test_common_memory_recording_replays_without_calls(pair, tmp_path):
    root, bodies, _ = pair
    result = _run_submission(root, "smoke", None, 1, runs_dir=tmp_path / "runs",
                             generator_memory=configuration(MODEL, compact_calls=1))
    before = len(bodies)
    assert result["status"] == "pass" and before >= 3
    assert protocol_replay(result["run_dir"])["status"] == "replayed"
    assert actor_replay(result["run_dir"])["status"] == "actor-replayed"
    assert len(bodies) == before


@pytest.mark.parametrize("model", ["anthropic/claude-fable-5", "anthropic/claude-opus-5"])
def test_added_claude_models_complete_and_replay_with_compaction(pair, tmp_path, model):
    root, bodies, _ = pair
    result = _run_submission(root, "smoke", None, 1, runs_dir=tmp_path / "runs",
                             generator_memory=configuration(model, compact_calls=1))
    assert result["status"] == "pass"
    assert len(bodies) >= 3
    assert all(b["model"] == model.removeprefix("anthropic/") and b["max_tokens"] == 128000 for b in bodies)
    count = len(bodies)
    assert protocol_replay(result["run_dir"])["status"] == "replayed"
    assert actor_replay(result["run_dir"])["status"] == "actor-replayed"
    assert len(bodies) == count


def test_common_memory_resume_restores_exact_native_history(pair, tmp_path):
    root, bodies, failures = pair
    failures.add(2)
    with pytest.raises(ParticipantFailure):
        _run_submission(root, "smoke", None, 1, runs_dir=tmp_path / "runs",
                        generator_memory=configuration(MODEL))
    source = next(path.parent for path in (tmp_path / "runs").glob("*/manifest.json"))
    failures.clear()
    result = _resume_submission(source, retry_interrupted_call=True)
    assert result["status"] == "pass"
    assert bodies[1] == bodies[2]
    assert actor_replay(result["run_dir"])["status"] == "actor-replayed"


def test_output_budget_continuation_changes_only_live_allowance(pair, tmp_path):
    root, bodies, failures = pair
    failures.add(2)
    with pytest.raises(ParticipantFailure):
        _run_submission(root, "smoke", None, 1, runs_dir=tmp_path / "runs",
                        generator_memory=configuration(MODEL), generator_output_tokens=None)
    source = next(path.parent for path in (tmp_path / "runs").glob("*/manifest.json"))
    original_files = {p: p.read_bytes() for p in source.glob("*.json*")}
    failed_request = copy.deepcopy(bodies[-1])
    assert failed_request["max_tokens"] == 50000
    failures.clear()
    count = len(bodies)
    result = _resume_submission(source, retry_interrupted_call=True, generator_output_tokens=128000,
                                generator_memory_chars=20000, generator_memory_target_chars=12000)
    assert result["status"] == "pass"
    assert bodies[count] == {**failed_request, "max_tokens": 128000}
    assert all(p.read_bytes() == value for p, value in original_files.items())
    manifest = json.loads((Path(result["run_dir"]) / "manifest.json").read_text())
    assert manifest["generator_output_budget"]["max_output_tokens"] == 128000
    count = len(bodies)
    assert actor_replay(result["run_dir"])["status"] == "actor-replayed"
    assert protocol_replay(result["run_dir"])["status"] == "replayed"
    assert len(bodies) == count


@pytest.mark.parametrize("cap,target", [(14000, None), (16000, 12000)])
def test_summary_cap_continuation_keeps_native_history_and_replays(pair, tmp_path, monkeypatch, cap, target):
    root, bodies, _ = pair
    original_send = NativeAPIGeneratorBackend._send
    def oversized(self, body):
        response = original_send(self, body)
        if "common_memory_summary" in body["messages"][-1]["content"]:
            response["content"][-1]["text"] = json.dumps({"payload_json": json.dumps({"memory": "x" * 13000})})
        return response
    monkeypatch.setattr(NativeAPIGeneratorBackend, "_send", oversized)
    with pytest.raises(ParticipantFailure):
        _run_submission(root, "smoke", None, 1, runs_dir=tmp_path / "runs",
                        generator_memory=configuration(MODEL, compact_calls=1, attempts=1))
    source = next(p.parent for p in (tmp_path / "runs").glob("*/manifest.json"))
    originals = {p: p.read_bytes() for p in source.glob("*.json*")}
    old_summary_body = copy.deepcopy(bodies[-1])
    before = len(bodies)
    result = _resume_submission(source, retry_interrupted_call=True, generator_memory_chars=cap,
                                generator_memory_target_chars=target)
    assert result["status"] == "pass"
    assert bodies[before]["messages"][:-1] == old_summary_body["messages"][:-1]
    assert str(cap) in bodies[before]["messages"][-1]["content"]
    manifest = json.loads((Path(result["run_dir"]) / "manifest.json").read_text())
    assert manifest["generator_memory"]["memory_chars"] == cap
    assert manifest["generator_memory_transition"].get("target_chars") == target
    assert manifest["generator_memory_transition"]["ancestors"][0]["memory_chars"] == 12000
    assert all(p.read_bytes() == value for p, value in originals.items())
    count = len(bodies)
    assert actor_replay(result["run_dir"])["status"] == "actor-replayed"
    assert protocol_replay(result["run_dir"])["status"] == "replayed"
    assert len(bodies) == count


def test_summary_cap_transition_rejects_other_policy_changes():
    from tech_tree_arena.runtime.memory_policy_transition import validate_ancestors
    from tech_tree_arena.errors import ReplayDivergence
    old = configuration(MODEL)
    validate_ancestors({**old, "memory_chars": 14000}, [old])
    with pytest.raises(ReplayDivergence):
        validate_ancestors({**old, "memory_chars": 14000, "compact_calls": 1}, [old])


def test_promotion_with_inherited_retry_policy_does_not_require_parent_retry_checkpoint(tmp_path):
    from tech_tree_arena.replay.recorder import read_resume_checkpoint
    parent = "a" * 32
    root = tmp_path / ("b" * 32)
    root.mkdir()
    (root / "manifest.json").write_text(json.dumps({"resume_policy": "retry_interrupted_call",
        "resumed_from": parent, "promoted_from": parent}))
    checkpoint = {"service_event_count": 10, "actor_streams": {}}
    (root / "checkpoint.private.json").write_text(json.dumps(checkpoint))
    assert read_resume_checkpoint(root) == checkpoint


@pytest.mark.parametrize("legacy_seed", [False, True])
def test_repeated_retry_retains_successful_uncommitted_native_calls(pair, tmp_path, monkeypatch, legacy_seed):
    root, bodies, failures = pair
    path = root / "participant/generator.py"
    path.write_text(path.read_text().replace('return super().step(message)', '''self.services.structured_model(developer="synthetic sibling", user=str(message),
            schema_name="sibling", schema={"type":"object"})
        return super().step(message)'''))
    failures.update({4, 5})
    with pytest.raises(ParticipantFailure):
        _run_submission(root, "smoke", None, 1, runs_dir=tmp_path / "runs",
                        generator_memory=configuration(MODEL))
    source = next(p.parent for p in (tmp_path / "runs").glob("*/manifest.json"))
    from tech_tree_arena.replay.recorder import RunRecorder, read_resume_checkpoint
    from tech_tree_arena.errors import ReplayDivergence
    with monkeypatch.context() as patch:
        if legacy_seed:
            seed = RunRecorder.seed_resume_checkpoint
            patch.setattr(RunRecorder, "seed_resume_checkpoint", lambda self, value:
                seed(self, {k: v for k, v in value.items() if k != "_resume_seed_checkpoint"}))
        with pytest.raises(ParticipantFailure):
            _resume_submission(source, retry_interrupted_call=True)
    derived = next(p.parent for p in (tmp_path / "runs").glob("*/manifest.json")
                   if p.parent != source)
    assert len(bodies) == 5
    checkpoint = derived / "checkpoint.private.json"
    original = checkpoint.read_bytes()
    if legacy_seed:
        damaged = json.loads(original)
        damaged["engine"]["k"] += 1
        checkpoint.write_text(json.dumps(damaged))
        with pytest.raises(ReplayDivergence, match="exact parent reconstruction"):
            read_resume_checkpoint(derived)
        checkpoint.write_bytes(original)
    assert actor_replay(derived)["status"] == "actor-replayed"
    result = _resume_submission(derived, retry_interrupted_call=True)
    assert result["status"] == "pass"
    assert len(bodies) == 6  # Successful sibling was never sent again.
    assert bodies[3] == bodies[4] == bodies[5]
    assert actor_replay(result["run_dir"])["status"] == "actor-replayed"
    records = [json.loads(x) for x in (Path(result["run_dir"]) / "service-calls.private.jsonl").read_text().splitlines()]
    assert sum(x["kind"] == "service_abandoned" for x in records) == 2
    assert checkpoint.read_bytes() == original


def test_budget_blocks_first_send_and_failure_replays(pair, tmp_path):
    from tech_tree_arena.runtime.run_budget import Ledger
    root, bodies, _ = pair
    ledger = Ledger(tmp_path / 'budget.json', create=True)
    ledger.seed('smoke-blue', {'generator': 50}, origin=str((tmp_path / 'runs').resolve()))
    with pytest.raises(Exception):
        _run_submission(root, 'smoke', None, 1, runs_dir=tmp_path / 'runs',
                        generator_memory=configuration(MODEL), budget_ledger=ledger.path)
    assert bodies == []
    source = next(p.parent for p in (tmp_path / 'runs').glob('*/manifest.json'))
    assert protocol_replay(source)['status'] == 'replayed'
    assert actor_replay(source)['status'] == 'actor-replayed'


def test_two_128k_truncations_stop_before_third_physical_send(pair, tmp_path, monkeypatch):
    root, bodies, _ = pair
    send = NativeAPIGeneratorBackend._send
    def truncate(self, body):
        response = send(self, body)
        response['stop_reason'] = 'max_tokens'
        response['usage']['output_tokens'] = 128000
        return response
    monkeypatch.setattr(NativeAPIGeneratorBackend, '_send', truncate)
    monkeypatch.setattr('tech_tree_arena.runtime.common_memory.time.sleep', lambda _: None)
    with pytest.raises(Exception):
        _run_submission(root, 'smoke', None, 1, runs_dir=tmp_path / 'runs',
                        generator_memory=configuration(MODEL))
    assert len(bodies) == 2
    source = next(p.parent for p in (tmp_path / 'runs').glob('*/manifest.json'))
    assert json.loads((source / 'status.json').read_text())['error_code'] == 'run_budget_blocked'
    assert protocol_replay(source)['status'] == 'replayed'
    assert actor_replay(source)['status'] == 'actor-replayed'
    with pytest.raises(Exception):
        _resume_submission(source, retry_interrupted_call=True)
    assert len(bodies) == 2


def test_fresh_summary_target_keeps_hard_cap_and_replays(pair, tmp_path):
    root, bodies, _ = pair
    result = _run_submission(root, 'smoke', None, 1, runs_dir=tmp_path / 'runs',
        generator_memory=configuration(MODEL, compact_calls=1, memory_chars=20000),
        generator_memory_target_chars=12000)
    summaries = [b for b in bodies if 'common_memory_summary' in b['messages'][-1]['content']]
    assert summaries and 'at most 12000 characters' in summaries[0]['messages'][-1]['content']
    assert result['status'] == 'pass'
    assert actor_replay(result['run_dir'])['status'] == 'actor-replayed'
