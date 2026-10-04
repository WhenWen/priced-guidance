import json
import copy
import shutil
from functools import partial
from pathlib import Path

import pytest

from tech_tree_arena import IdeaVerdict
from tech_tree_arena.contract.recovery import LEGACY_ACCOUNTING, occurrence_bits
from tech_tree_arena.runtime import engine
from tech_tree_arena.runtime.branch import BranchStore
from tech_tree_arena.replay.accounting import migrate_run
from tech_tree_arena.cli import _resume_submission, _run_submission
from tech_tree_arena.errors import ArenaError, ResourceLimitExceeded
from tech_tree_arena.evaluation.research import ResearchJudge


def test_common_memory_survives_promotion_and_essence_checkout(tmp_path, monkeypatch):
    from tech_tree_arena.runtime.common_memory import configuration
    from tech_tree_arena.runtime.native_api_generator import NativeAPIGeneratorBackend
    pair, pack = tmp_path / "pair", tmp_path / "pack"
    _write_modular_pair(pair, essence_marker="essence", strict_marker="strict")
    _write_pack(pack)
    generator = pair / "participant/generator.py"
    generator.write_text(generator.read_text().replace(
        "    def step(self, value):\n",
        "    def step(self, value):\n"
        "        if not isinstance(value, StageTransition):\n"
        "            self.services.structured_model(developer='synthetic', user=self.stage, "
        "schema_name='stage_probe', schema={'type': 'object'})\n"))
    bodies = []
    def send(self, body):
        bodies.append(copy.deepcopy(body))
        return {"content": [{"type": "thinking", "thinking": "synthetic", "signature": "signature"},
            {"type": "text", "text": '{"payload_json":"{}"}'}], "stop_reason": "end_turn",
            "usage": {"input_tokens": 20, "output_tokens": 10}}
    monkeypatch.setattr(NativeAPIGeneratorBackend, "_send", send)
    essence_calls = 0
    def evaluate(self, target, ideas):
        nonlocal essence_calls
        if self.mode == "essence":
            essence_calls += 1
        # Reject the inherited submission on all three repetitions. The actor
        # must then restore a directional checkpoint, reapply the transition,
        # and continue with that checkpoint's exact native history.
        passed = self.mode != "essence" or essence_calls > 3
        return tuple(IdeaVerdict(idea.idea_id, passed, self.mode) for idea in ideas)
    monkeypatch.setattr(ResearchJudge, "evaluate", evaluate)
    directional = _run_submission(pair, str(pack), "synthetic", 1,
        runs_dir=tmp_path / "runs", judge_mode="directional",
        generator_memory=configuration("anthropic/claude-fable-5-1"))
    initial = len(bodies)
    essence = _resume_submission(Path(directional["run_dir"]), promote_judge="essence")
    assert essence["status"] == "pass" and essence["checkouts"] == 1
    assert len(bodies) > initial
    child = bodies[initial]
    assert child["messages"][0] == bodies[0]["messages"][0]
    assert child["messages"][1]["content"][0]["signature"] == "signature"
    assert "essence" in child["messages"][-1]["content"]
    before_replay = len(bodies)
    essence_calls = 0
    assert actor_replay(essence["run_dir"])["status"] == "actor-replayed"
    assert len(bodies) == before_replay
from tech_tree_arena.replay.recorder import actor_replay, protocol_replay


def _write_pair(root: Path) -> None:
    participant = root / "participant"
    participant.mkdir(parents=True)
    (root / "submission.toml").write_text(
        """schema_version = 1
name = "promotion-fixture"
version = "0.1.0"
protocol = "idea-recovery-v1"
generator = "participant.generator:Generator"
oracle = "participant.oracle:Oracle"
""",
        encoding="utf-8",
    )
    (participant / "__init__.py").write_text("", encoding="utf-8")
    (participant / "generator.py").write_text(
        """from tech_tree_arena import Choice, Idea, Question, Submission, SubmitOption

class Generator:
    def __init__(self, services):
        self.services = services

    def step(self, choice: Choice | None):
        if choice is None:
            return Question("choose", (
                SubmitOption("submit-a", {"candidate": "a"}, "0.25"),
                SubmitOption("submit-b", {"candidate": "b"}, "0.75"),
            ))
        return Submission((Idea(
            "idea",
            {"setting_and_object": "A neutral synthetic contribution.", "findings": []},
            "1",
        ),))
""",
        encoding="utf-8",
    )
    (participant / "oracle.py").write_text(
        """from tech_tree_arena import Choice

class Oracle:
    def __init__(self, target, services):
        self.services = services

    def step(self, presented):
        return Choice("submit-a")
""",
        encoding="utf-8",
    )


def _write_pack(root: Path) -> None:
    gold = root / "secret" / "gold"
    gold.mkdir(parents=True)
    (root / "pack.toml").write_text(
        """schema_version = 1
name = "promotion-targets"
kind = "development"
protocol = "idea-recovery-v1"
target_count = 1
""",
        encoding="utf-8",
    )
    (gold / "synthetic.json").write_text(
        json.dumps(
            {
                "arxiv_id": "synthetic",
                "summary": {
                    "title": "Synthetic target",
                    "setting_and_object": {
                        "category": "Method",
                        "groups": ["A neutral synthetic contribution."],
                    },
                    "concrete_detailed_setting": [],
                    "key_findings": [],
                },
            }
        ),
        encoding="utf-8",
    )


def _write_modular_pair(root: Path, *, essence_marker: str, strict_marker: str) -> None:
    participant = root / "participant"
    stages = participant / "stages"
    stages.mkdir(parents=True)
    (root / "submission.toml").write_text(
        """schema_version = 1
name = "modular-promotion-fixture"
version = "0.1.0"
protocol = "idea-recovery-v1"
generator = "participant.generator:Generator"
oracle = "participant.oracle:Oracle"

[modules]
shared = ["participant/__init__.py", "participant/generator.py", "participant/oracle.py", "participant/stages/__init__.py"]
directional = ["participant/stages/directional.py"]
essence = ["participant/stages/essence.py"]
strict = ["participant/stages/strict.py"]
""",
        encoding="utf-8",
    )
    (participant / "__init__.py").write_text("", encoding="utf-8")
    (stages / "__init__.py").write_text("", encoding="utf-8")
    (stages / "directional.py").write_text("MARKER = 'directional-v1'\n", encoding="utf-8")
    (stages / "essence.py").write_text(
        f"MARKER = {essence_marker!r}\n", encoding="utf-8"
    )
    (stages / "strict.py").write_text(
        f"MARKER = {strict_marker!r}\n", encoding="utf-8"
    )
    (participant / "generator.py").write_text(
        """from importlib import import_module
from tech_tree_arena import Choice, Idea, Question, StageReady, StageTransition, Submission, SubmitOption

class Generator:
    def __init__(self, services):
        self.services = services
        self.stage = services.public_resources["active_stage"]
        self.whiteboard = ["synthetic directional anchor"]

    def step(self, value):
        if isinstance(value, StageTransition):
            assert value.from_stage == self.stage
            self.stage = value.to_stage
            marker = import_module(f"participant.stages.{self.stage}").MARKER
            return StageReady(self.stage, {
                "idea": "A neutral synthetic contribution.",
                "whiteboard": list(self.whiteboard),
                "module_marker": marker,
            })
        if value is None:
            return Question("choose", (SubmitOption("submit", {"stage": self.stage}, "1"),))
        assert isinstance(value, Choice)
        return Submission((Idea(
            "idea",
            {"setting_and_object": "A neutral synthetic contribution.", "findings": []},
            "1",
        ),))
""",
        encoding="utf-8",
    )
    (participant / "oracle.py").write_text(
        """from importlib import import_module
from tech_tree_arena import Checkout, Choice, StageReady, StageTransition, SubmissionFeedback

class Oracle:
    def __init__(self, target, services):
        self.services = services
        self.stage = services.public_resources["active_stage"]

    def step(self, value):
        if isinstance(value, StageTransition):
            assert value.from_stage == self.stage
            self.stage = value.to_stage
            marker = import_module(f"participant.stages.{self.stage}").MARKER
            return StageReady(self.stage, {"module_marker": marker})
        if isinstance(value, SubmissionFeedback):
            return Checkout(value.source.question_id)
        return Choice("submit")
""",
        encoding="utf-8",
    )


@pytest.mark.parametrize("legacy", [False, True])
def test_completed_pass_can_promote_directional_to_essence_to_strict(
    tmp_path: Path, monkeypatch, legacy
) -> None:
    pair = tmp_path / "pair"
    pack = tmp_path / "pack"
    runs = tmp_path / "runs"
    _write_pair(pair)
    _write_pack(pack)

    def always_pass(self, _target, ideas):
        return tuple(IdeaVerdict(idea.idea_id, True, self.mode) for idea in ideas)

    monkeypatch.setattr(ResearchJudge, "evaluate", always_pass)

    with monkeypatch.context() as pricing:
        if legacy:
            pricing.setattr(engine, "CURRENT_ACCOUNTING", LEGACY_ACCOUNTING)
            pricing.setattr(engine, "BranchStore", partial(BranchStore, accounting_version=LEGACY_ACCOUNTING))
        directional = _run_submission(
            pair,
            str(pack),
            "synthetic",
            7,
            runs_dir=runs,
            judge_mode="directional",
            max_cost_usd_per_role=1,
        )
    directional_root = Path(directional["run_dir"])
    directional_manifest = json.loads(
        (directional_root / "manifest.json").read_text()
    )

    assert directional["K"] == pytest.approx(2.0 + (0.0 if legacy else occurrence_bits(1)))
    assert not (directional_root / "checkpoint.private.json").exists()
    assert (directional_root / "promotion-checkpoint.private.json").is_file()
    if legacy:
        migrate_run(directional_root, write=True)
        with pytest.raises(ResourceLimitExceeded):
            _resume_submission(
                directional_root, promote_judge="essence", max_cost_usd_per_role=1,
                max_information_bits=2.01,
            )
        failed = [path.parent for path in runs.rglob("score.json")
                  if json.loads(path.read_text()).get("status") == "error"]
        assert len(failed) == 1
        assert protocol_replay(failed[0])["result"]["status"] == "error"
        assert actor_replay(failed[0])["status"] == "actor-replayed"

    essence = _resume_submission(
        directional_root,
        promote_judge="essence",
        max_cost_usd_per_role=1,
    )
    essence_root = Path(essence["run_dir"])
    essence_manifest = json.loads((essence_root / "manifest.json").read_text())

    assert essence["status"] == "pass"
    assert essence["K"] == pytest.approx(2.0 + occurrence_bits(1))
    assert essence["judge_repeats"] == 3
    assert essence["judge_passes"] == 3
    assert essence["promotion_base_K"] == pytest.approx(2.0 + occurrence_bits(1))
    assert essence_manifest["judge"] == "research-essence"
    assert essence_manifest["judge_config"]["repeats"] == 3
    assert essence_manifest["promotion_source_judge"] == "research-directional"
    assert (
        essence_manifest["generator_public_resources_ref"]
        == directional_manifest["generator_public_resources_ref"]
    )
    assert (
        essence_manifest["oracle_public_resources_ref"]
        == directional_manifest["oracle_public_resources_ref"]
    )
    assert (essence_root / "promotion-checkpoint.private.json").is_file()
    assert protocol_replay(essence_root)["status"] == "replayed"
    assert actor_replay(essence_root)["status"] == "actor-replayed"

    strict = _resume_submission(
        essence_root,
        promote_judge="strict",
        max_cost_usd_per_role=1,
    )
    strict_manifest = json.loads(
        (Path(strict["run_dir"]) / "manifest.json").read_text()
    )

    assert strict["status"] == "pass"
    assert strict["K"] == pytest.approx(2.0 + occurrence_bits(1))
    assert strict["promotion_base_K"] == pytest.approx(2.0 + occurrence_bits(1))
    assert strict_manifest["judge"] == "research-fmn"
    assert strict_manifest["judge_config"] == {
        "m": None,
        "mode": "fmn",
        "n": 0,
        "repeats": 1,
        "coarse_reasons": True,
    }
    assert strict_manifest["promotion_source_judge"] == "research-essence"

    with pytest.raises(ArenaError, match="unsupported Judge promotion"):
        _resume_submission(
            directional_root,
            promote_judge="unknown",
            max_cost_usd_per_role=1,
        )


def test_modular_promotion_freezes_prefix_and_activates_only_the_new_stage(
    tmp_path: Path, monkeypatch
) -> None:
    original = tmp_path / "original"
    essence_replacement = tmp_path / "essence-replacement"
    strict_replacement = tmp_path / "strict-replacement"
    pack = tmp_path / "pack"
    runs = tmp_path / "runs"
    _write_modular_pair(
        original, essence_marker="essence-v1", strict_marker="strict-v1"
    )
    _write_pack(pack)

    strict_calls = 0

    def always_pass(self, _target, ideas):
        nonlocal strict_calls
        passed = True
        if self.mode == "fmn":
            strict_calls += 1
            passed = strict_calls > 1
        return tuple(IdeaVerdict(idea.idea_id, passed, self.mode) for idea in ideas)

    monkeypatch.setattr(ResearchJudge, "evaluate", always_pass)
    directional = _run_submission(
        original,
        str(pack),
        "synthetic",
        9,
        runs_dir=runs,
        judge_mode="directional",
        max_cost_usd_per_role=1,
    )
    directional_root = Path(directional["run_dir"])
    directional_manifest = json.loads(
        (directional_root / "manifest.json").read_text(encoding="utf-8")
    )

    shutil.copytree(original, essence_replacement)
    (essence_replacement / "participant/stages/essence.py").write_text(
        "MARKER = 'essence-v2'\n", encoding="utf-8"
    )
    essence = _resume_submission(
        directional_root,
        promote_judge="essence",
        compatible_submission=essence_replacement,
        max_cost_usd_per_role=1,
    )
    essence_root = Path(essence["run_dir"])
    essence_manifest = json.loads(
        (essence_root / "manifest.json").read_text(encoding="utf-8")
    )
    handoff = json.loads(
        (essence_root / "stage-handoff.private.json").read_text(encoding="utf-8")
    )

    assert essence["K"] == directional["K"] == pytest.approx(occurrence_bits(1))
    assert essence_manifest["active_stage"] == "essence"
    assert essence_manifest["stage_prefix_proof"]["frozen_through"] == "directional"
    assert (
        essence_manifest["stage_module_hashes"]["directional"]
        == directional_manifest["stage_module_hashes"]["directional"]
    )
    assert handoff["roles"]["generator"]["ready"]["handoff"]["whiteboard"] == [
        "synthetic directional anchor"
    ]
    assert (
        handoff["roles"]["generator"]["ready"]["handoff"]["module_marker"]
        == "essence-v2"
    )
    assert protocol_replay(essence_root)["status"] == "replayed"
    assert actor_replay(essence_root)["status"] == "actor-replayed"
    service_path = essence_root / "service-calls.private.jsonl"
    assert not service_path.exists() or not service_path.read_text(encoding="utf-8")

    shutil.copytree(essence_replacement, strict_replacement)
    (strict_replacement / "participant/stages/strict.py").write_text(
        "MARKER = 'strict-v2'\n", encoding="utf-8"
    )
    strict = _resume_submission(
        essence_root,
        promote_judge="strict",
        compatible_submission=strict_replacement,
        max_cost_usd_per_role=1,
    )
    strict_handoff = json.loads(
        (Path(strict["run_dir"]) / "stage-handoff.private.json").read_text(
            encoding="utf-8"
        )
    )
    assert strict["status"] == "pass"
    assert strict["K"] == pytest.approx(occurrence_bits(2))
    assert (
        strict_handoff["roles"]["generator"]["ready"]["handoff"]["module_marker"]
        == "strict-v2"
    )
    strict_calls = 0
    assert actor_replay(Path(strict["run_dir"]))["status"] == "actor-replayed"
    private_events = [
        json.loads(line)
        for line in (Path(strict["run_dir"]) / "events.private.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    recovery_transitions = [
        event
        for event in private_events
        if event.get("kind") == "stage_transition"
        and event.get("role") == "generator"
        and event.get("branch_id") != "root"
    ]
    assert [event["to_stage"] for event in recovery_transitions] == [
        "essence",
        "strict",
    ]

    frozen_tamper = tmp_path / "frozen-tamper"
    shutil.copytree(original, frozen_tamper)
    (frozen_tamper / "participant/stages/directional.py").write_text(
        "MARKER = 'tampered'\n", encoding="utf-8"
    )
    with pytest.raises(ArenaError, match="frozen prefix modules: directional"):
        _resume_submission(
            directional_root,
            promote_judge="essence",
            compatible_submission=frozen_tamper,
            max_cost_usd_per_role=1,
        )


def test_compatible_resume_records_hashes_matching_its_own_snapshot(
    tmp_path: Path, monkeypatch
) -> None:
    """A plain compatible-submission resume must stay resumable itself.

    The fork swaps in the replacement tree as its snapshot, so the module
    hashes it records have to describe that tree. Inheriting the source's
    hashes leaves the fork disagreeing with its own snapshot, and every later
    resume of it fails validation.
    """

    from tech_tree_arena.replay.artifacts import ArtifactStore
    from tech_tree_arena.submission_io.manifest import (
        load_manifest,
        stage_module_hashes,
    )

    original = tmp_path / "original"
    replacement = tmp_path / "replacement"
    pack = tmp_path / "pack"
    runs = tmp_path / "runs"
    marker = tmp_path / "allow-finish"
    _write_modular_pair(
        original, essence_marker="essence-v1", strict_marker="strict-v1"
    )
    _write_pack(pack)
    generator_path = original / "participant/generator.py"
    source = generator_path.read_text(encoding="utf-8")
    generator_path.write_text(
        source.replace(
            "        assert isinstance(value, Choice)\n",
            "        assert isinstance(value, Choice)\n"
            "        import os\n"
            "        from pathlib import Path\n"
            "        if not Path(os.environ['PROMOTION_FIXTURE_MARKER']).exists():\n"
            "            raise RuntimeError('synthetic participant failure')\n",
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("PROMOTION_FIXTURE_MARKER", str(marker))
    monkeypatch.setattr(
        ResearchJudge,
        "evaluate",
        lambda self, _target, ideas: tuple(
            IdeaVerdict(idea.idea_id, True, self.mode) for idea in ideas
        ),
    )

    with pytest.raises(Exception):
        _run_submission(
            original,
            str(pack),
            "synthetic",
            9,
            runs_dir=runs,
            judge_mode="directional",
            max_cost_usd_per_role=1,
        )
    failed_root = max(
        (path for path in runs.iterdir() if (path / "manifest.json").is_file()),
        key=lambda path: path.stat().st_mtime,
    )
    failed_status = json.loads(
        (failed_root / "status.json").read_text(encoding="utf-8")
    )
    assert failed_status["status"] == "error" and failed_status["resumable"]

    shutil.copytree(original, replacement)
    (replacement / "participant/stages/strict.py").write_text(
        "MARKER = 'strict-v2'\n", encoding="utf-8"
    )
    marker.write_text("ok\n", encoding="utf-8")
    resumed = _resume_submission(
        failed_root,
        compatible_submission=replacement,
        max_cost_usd_per_role=1,
    )
    resumed_root = Path(resumed["run_dir"])
    resumed_manifest = json.loads(
        (resumed_root / "manifest.json").read_text(encoding="utf-8")
    )

    artifacts = ArtifactStore(runs)
    snapshot_hashes = stage_module_hashes(
        load_manifest(artifacts.resolve_tree(resumed_manifest["submission_snapshot"]))
    )
    assert resumed_manifest["stage_module_hashes"] == snapshot_hashes
    source_manifest = json.loads(
        (failed_root / "manifest.json").read_text(encoding="utf-8")
    )
    assert (
        resumed_manifest["stage_module_hashes"]["strict"]
        != source_manifest["stage_module_hashes"]["strict"]
    )
