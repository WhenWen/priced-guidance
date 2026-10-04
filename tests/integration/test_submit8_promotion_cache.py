"""Exercise real submit8 activation across a compatible stage replacement."""
import json
import shutil
from pathlib import Path

from tech_tree_arena import IdeaVerdict
from tech_tree_arena.cli import _resume_submission, _run_submission
from tech_tree_arena.evaluation.research import ResearchJudge
from tech_tree_arena.replay.recorder import actor_replay
from tools.audit_codex_eager_run import audit

ROOT = Path(__file__).resolve().parents[2]


def test_compatible_essence_fix_preserves_inherited_slate_on_checkout(tmp_path, monkeypatch):
    source = tmp_path / "source"
    fixed = tmp_path / "fixed"
    shutil.copytree(ROOT / "submissions/reference_pair_submit8", source)
    # Synthetic route generation and Oracle choices use no model service.
    # Submission activation and stage transitions use the production methods.
    (source / "participant/generator.py").write_text('''
from tech_tree_arena import Option, Question
from participant.pair import Generator as Base, _serialized_question, _stable_hash

class Generator(Base):
    def step(self, value):
        return super().step(value)

    def _field_question(self, role):
        self.state['current_draft'] = 'A synthetic public draft.'
        return self._directional_dispatch_question()

    def _build_directional_previews(self, modes):
        result = {}
        for mode in modes:
            q = Question('Synthetic ' + mode, (Option('retry', {'kind': 'retry'}, '1'),))
            result[mode] = {'question': q, 'question_hash': _stable_hash(_serialized_question(q))}
        return result

    def _build_submission_preview(self, draft):
        return {'ideas': [{'idea_id': str(i), 'content': {
            'setting_and_object': 'Synthetic alternative ' + str(i), 'findings': []},
            'probability': '0.125'} for i in range(8)]}
''')
    (source / "participant/guide.py").write_text('''
from tech_tree_arena import Checkout, Choice, StageReady, StageTransition, SubmissionFeedback

class Guide:
    def __init__(self, target, services):
        self.stage = services.public_resources['active_stage']

    def step(self, value):
        if isinstance(value, StageTransition):
            self.stage = value.to_stage
            return StageReady(self.stage, {})
        if isinstance(value, SubmissionFeedback):
            return Checkout(value.source.question_id)
        return Choice('mode-submit')
''')
    shutil.copytree(source, fixed)
    # Reproduce the frozen pre-fix source, then replace only its future stage.
    shutil.copyfile(ROOT / "submissions/reference_pair/participant/stages/essence.py",
                    source / "participant/stages/essence.py")
    pack = tmp_path / "pack"
    gold = pack / "secret/gold"
    gold.mkdir(parents=True)
    (pack / "pack.toml").write_text('''schema_version = 1
name = "synthetic-promotion"
kind = "development"
protocol = "idea-recovery-v1"
target_count = 1
''')
    (gold / "synthetic.json").write_text(json.dumps({"arxiv_id": "synthetic", "summary": {
        "title": "Synthetic", "setting_and_object": {"category": "Method", "groups": ["Synthetic"]},
        "concrete_detailed_setting": [], "key_findings": [],
    }}))
    essence_calls = 0

    def evaluate(self, target, ideas):
        nonlocal essence_calls
        assert len(ideas) == 8  # Also checks the post-checkout submission.
        if self.mode == "essence":
            essence_calls += 1
        passed = self.mode != "essence" or essence_calls > 3
        return tuple(IdeaVerdict(idea.idea_id, passed, "synthetic") for idea in ideas)

    monkeypatch.setattr(ResearchJudge, "evaluate", evaluate)
    directional = _run_submission(source, str(pack), "synthetic", 1,
                                   runs_dir=tmp_path / "runs", judge_mode="directional")
    promoted = _resume_submission(Path(directional["run_dir"]), promote_judge="essence",
                                  compatible_submission=fixed)
    run = Path(promoted["run_dir"])
    assert promoted["status"] == "pass"
    assert promoted["checkouts"] == 1
    report = audit(run, expected_backend="common-memory-v1")
    assert report["problems"] == []
    assert all(a["exact_cached_submission"] and a["generator_calls_during_activation"] == 0
               for a in report["activations"])
    essence_calls = 0
    assert actor_replay(run)["status"] == "actor-replayed"
