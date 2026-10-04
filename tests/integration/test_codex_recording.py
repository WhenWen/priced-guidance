"""Exercise Codex service journals and recovery without a network or credentials."""
import copy
import json
import sys
from pathlib import Path

import pytest

from tech_tree_arena.cli import _resume_submission, _run_submission
from tech_tree_arena.errors import ParticipantFailure
from tech_tree_arena.replay.recorder import actor_replay, protocol_replay
from tech_tree_arena.runtime import codex_generator
from tech_tree_arena.runtime.services import _request_hash

ROOT = Path(__file__).resolve().parents[2]
pytestmark = pytest.mark.skipif(sys.platform not in {"darwin", "linux"}, reason="requires OS sandbox")


class FakeNativeBackend:
    supports_context_chain = True
    parents = []
    fail_on = None

    def __init__(self, config, artifacts):
        self.artifacts = artifacts
        self.usage = {"calls": 0, "input_tokens": 0, "output_tokens": 0, "cost_usd": 0.0}
        self.metadata = {}

    def structured_in_context(self, context, **request):
        type(self).parents.append(copy.deepcopy(context))
        call = len(type(self).parents)
        self.metadata = {"backend": "codex-source-v1", "model": request["model"], "schema_name": request["schema_name"],
            "auth_mode": "chatgpt", "sandbox_profile": "codex-seatbelt-v1", "parent_context": copy.deepcopy(context),
            "request_hash": _request_hash("model.structured", request), "usage": {"cost_usd": 0.0}}
        if call == self.fail_on:
            raise RuntimeError("synthetic interrupted native turn")
        total = (context or {}).get("usage_total") or {"inputTokens": 0, "outputTokens": 0}
        total = {"inputTokens": total["inputTokens"] + 11, "outputTokens": total["outputTokens"] + 2}
        reference = self.artifacts.put_json({"rollout": json.dumps({"synthetic": call}) + "\n"})
        self.metadata.update({"codex_context": {"thread_id": str(call), "turn_id": "turn", "rollout": reference, "usage_total": total},
            "codex_events": [{"method": "thread/tokenUsage/updated", "params": {"tokenUsage": {"total": total}}}],
            "usage": {"calls": 1, "input_tokens": 11, "output_tokens": 2, "cost_usd": 0.0}})
        for key, value in self.metadata["usage"].items():
            self.usage[key] += value
        return {"ok": True}

    def last_call_metadata(self):
        return copy.deepcopy(self.metadata)

    def usage_totals(self):
        return dict(self.usage)

    def restore_usage(self, value):
        self.usage = {key: value.get(key, 0) for key in self.usage}


@pytest.fixture
def native_pair(tmp_path, monkeypatch):
    import shutil
    root = tmp_path / "pair"
    shutil.copytree(ROOT / "submissions/examples/minimal_pair", root)
    path = root / "participant/generator.py"
    path.write_text(path.read_text() + '''

BaseGenerator = Generator
class Generator(BaseGenerator):
    def step(self, message):
        self.services.structured_model(model="gpt-6-astra", developer="synthetic", user=str(message),
            schema_name="synthetic", schema={"type":"object"})
        return super().step(message)
''')
    FakeNativeBackend.parents = []
    FakeNativeBackend.fail_on = None
    monkeypatch.setattr(codex_generator, "CodexGeneratorBackend", FakeNativeBackend)
    return root


@pytest.mark.parametrize("effort", [None, "xhigh"])
def test_recorded_codex_calls_replay_offline(native_pair, tmp_path, effort):
    result = _run_submission(native_pair, "smoke", None, 1, runs_dir=tmp_path / "runs",
                             generator_codex={"backend": "codex-source-v1",
                                              **({"reasoning_effort": effort} if effort else {})})
    before = len(FakeNativeBackend.parents)
    assert before >= 2
    assert result["status"] == "pass"
    assert protocol_replay(result["run_dir"])["status"] == "replayed"
    assert actor_replay(result["run_dir"])["status"] == "actor-replayed"
    assert len(FakeNativeBackend.parents) == before


def test_resume_inherits_recorded_native_parent(native_pair, tmp_path):
    FakeNativeBackend.fail_on = 2
    with pytest.raises(ParticipantFailure):
        _run_submission(native_pair, "smoke", None, 1, runs_dir=tmp_path / "runs",
                        generator_codex={"backend": "codex-source-v1"})
    source = next(path.parent for path in (tmp_path / "runs").glob("*/manifest.json"))
    FakeNativeBackend.fail_on = None
    result = _resume_submission(source, retry_interrupted_call=True)
    assert result["status"] == "pass"
    assert [parent["thread_id"] if parent else None for parent in FakeNativeBackend.parents] == [None, "1", "1"]
    assert actor_replay(result["run_dir"])["status"] == "actor-replayed"
