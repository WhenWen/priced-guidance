from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from tech_tree_arena import (
    Option,
    PresentedQuestion,
    ProbabilitySamplingGuide,
    Question,
    SubmitOption,
    resume_sample_ideas,
    sample_ideas,
)
from tech_tree_arena.errors import RecordedRunFailure
from tech_tree_arena.replay.recorder import actor_replay, protocol_replay
from tech_tree_arena.runtime.providers import ScriptedModelBackend
from tech_tree_arena.sampling import _draft_from_question


ROOT = Path(__file__).resolve().parents[2]


class FixedRandomServices:
    def __init__(self, draw: float) -> None:
        self.draw = draw
        self.calls = 0

    def random(self) -> float:
        self.calls += 1
        return self.draw


def test_probability_sampling_guide_uses_generator_distribution() -> None:
    question = PresentedQuestion(
        "q1",
        Question(
            "pick",
            (
                Option("first", None, "0.2"),
                Option("second", None, "0.3"),
                SubmitOption("last", None, "0.5"),
            ),
        ),
    )

    assert ProbabilitySamplingGuide(FixedRandomServices(0.199)).step(question).option_id == "first"
    assert ProbabilitySamplingGuide(FixedRandomServices(0.2)).step(question).option_id == "second"
    assert ProbabilitySamplingGuide(FixedRandomServices(0.999)).step(question).option_id == "last"


def test_draft_preview_reads_generator_committed_option_payloads() -> None:
    question = Question(
        "next",
        (
            Option("continue", {"belief": {"top_idea": "Current full draft"}}, "0.5"),
            SubmitOption("submit", {"preview": {"top_idea": "Current full draft"}}, "0.5"),
        ),
    )

    assert _draft_from_question(question) == "Current full draft"


def _write_pair(
    root: Path, *, never_submit: bool = False, model_call: bool = False
) -> None:
    participant = root / "participant"
    participant.mkdir(parents=True)
    (root / "submission.toml").write_text(
        """schema_version = 1
name = "sampling-test-pair"
version = "0.1.0"
protocol = "idea-recovery-v1"
generator = "participant.generator:Generator"
oracle = "participant.oracle:Oracle"
""",
        encoding="utf-8",
    )
    (participant / "__init__.py").write_text("", encoding="utf-8")
    if never_submit:
        options = 'Option("continue", {"answer": "again"}, "1")'
    else:
        options = """SubmitOption("low", {"answer": "low"}, "0.25"),
                SubmitOption("high", {"answer": "high"}, "0.75")"""
    model_setup = "self.called_model = False" if model_call else ""
    model_step = (
        """if not self.called_model:
            self.services.structured_model(
                developer="test",
                user="test",
                schema={"type": "object", "properties": {}},
                schema_name="sampling_progress",
                max_output_tokens=10,
            )
            self.called_model = True"""
        if model_call
        else ""
    )
    (participant / "generator.py").write_text(
        f"""from tech_tree_arena import Choice, Idea, Option, Question, Submission, SubmitOption

class Generator:
    def __init__(self, services):
        self.services = services
        {model_setup}

    def step(self, choice: Choice | None):
        {model_step}
        if choice is not None and choice.option_id in {{"low", "high"}}:
            return Submission((Idea("sampled", choice.public_payload, "1"),))
        return Question("sample", ({options},))
""",
        encoding="utf-8",
    )
    (participant / "oracle.py").write_text(
        """class Oracle:
    def __init__(self, target, services):
        raise RuntimeError("the original Oracle must not be constructed")

    def step(self, message):
        raise RuntimeError("the original Oracle must not be called")
""",
        encoding="utf-8",
    )


def test_sample_ideas_replaces_original_guide_and_returns_submission(tmp_path: Path) -> None:
    pair = tmp_path / "pair"
    _write_pair(pair)

    runs = tmp_path / "runs"
    first = sample_ideas(pair, target_pack=None, seed=1, runs_dir=runs)
    second = sample_ideas(pair, target_pack=None, seed=2, runs_dir=runs)

    assert first.ideas[0].content == {"answer": "low"}
    assert first.choices[0].probability == "0.25"
    assert first.guide_usage["random_calls"] == 1
    assert second.ideas[0].content == {"answer": "high"}
    assert second.choices[0].probability == "0.75"
    manifest = json.loads((Path(first.run_dir) / "manifest.json").read_text())
    assert manifest["run_kind"] == "sample_ideas"
    assert "target_pack_snapshot" not in manifest
    assert protocol_replay(first.run_dir)["status"] == "replayed"
    assert actor_replay(first.run_dir)["status"] == "actor-replayed"


def test_sample_ideas_streams_progress_and_model_boundaries(tmp_path: Path) -> None:
    pair = tmp_path / "pair"
    _write_pair(pair, model_call=True)
    events: list[dict[str, object]] = []

    result = sample_ideas(
        pair,
        target_pack=None,
        seed=1,
        model_backend=ScriptedModelBackend([{}]),
        event_sink=events.append,
        runs_dir=tmp_path / "runs",
    )

    kinds = [event["kind"] for event in events]
    assert result.ideas[0].content == {"answer": "low"}
    assert kinds == [
        "sampling_started",
        "generator_step_started",
        "model_call_started",
        "model_call_finished",
        "generator_step_finished",
        "question",
        "choice",
        "generator_step_started",
        "generator_step_finished",
        "submission",
        "sampling_finished",
    ]
    assert events[2]["schema_name"] == "sampling_progress"


def test_sample_ideas_fails_if_sampled_path_never_submits(tmp_path: Path) -> None:
    pair = tmp_path / "pair"
    _write_pair(pair, never_submit=True)

    with pytest.raises(RecordedRunFailure) as caught:
        sample_ideas(
            pair,
            target_pack=None,
            seed=1,
            max_questions=2,
            runs_dir=tmp_path / "runs",
        )

    assert caught.value.code == "resource_limit_exceeded"
    status = json.loads((Path(caught.value.run_dir) / "status.json").read_text())
    assert status["status"] == "error"
    assert status["resumable"] is True


def _write_resumable_pair(root: Path) -> None:
    _write_pair(root)
    (root / "participant" / "generator.py").write_text(
        """from tech_tree_arena import Choice, Idea, Question, Submission, SubmitOption

class Generator:
    def __init__(self, services):
        self.services = services

    def step(self, choice: Choice | None):
        if choice is None:
            return Question("sample", (SubmitOption("submit", {"draft": "ready"}, "1"),))
        self.services.structured_model(
            developer="test",
            user="test",
            schema={"type": "object", "properties": {}},
            schema_name="resume_sample",
            max_output_tokens=10,
        )
        return Submission((Idea("resumed", {"answer": "recovered"}, "1"),))
""",
        encoding="utf-8",
    )


def test_sample_ideas_resumes_interrupted_model_call(tmp_path: Path) -> None:
    pair = tmp_path / "pair"
    _write_resumable_pair(pair)
    runs = tmp_path / "runs"

    with pytest.raises(RecordedRunFailure) as caught:
        sample_ideas(
            pair,
            target_pack=None,
            seed=7,
            model_backend=ScriptedModelBackend([]),
            runs_dir=runs,
        )

    failed_dir = Path(caught.value.run_dir)
    failed_status = json.loads((failed_dir / "status.json").read_text())
    assert failed_status["resumable"] is True
    assert actor_replay(failed_dir)["status"] == "actor-replayed"

    resumed = resume_sample_ideas(
        failed_dir,
        retry_interrupted_call=True,
        model_backend=ScriptedModelBackend([{}]),
    )

    assert resumed.resumed_from == failed_dir.name
    assert resumed.ideas[0].content == {"answer": "recovered"}
    assert [choice.option_id for choice in resumed.choices] == ["submit"]
    assert protocol_replay(resumed.run_dir)["status"] == "replayed"
    assert actor_replay(resumed.run_dir)["status"] == "actor-replayed"


def test_sample_ideas_cli_outputs_ideas_as_json(tmp_path: Path) -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "tech_tree_arena.cli",
            "sample-ideas",
            str(ROOT / "submissions" / "examples" / "minimal_pair"),
            "--target-pack",
            "smoke",
            "--seed",
            "4",
            "--runs-dir",
            str(tmp_path / "runs"),
        ],
        check=True,
        text=True,
        capture_output=True,
    )

    result = json.loads(completed.stdout)
    assert result["pair"] == "minimal-pair"
    assert result["questions"] == 1
    assert result["choices"][0]["submit"] is True
    assert result["ideas"][0]["idea_id"] == "selected-color"
    assert "sampling started pair=minimal-pair" in completed.stderr
    assert "random choice question=1" in completed.stderr
    assert "submission ready ideas=1" in completed.stderr
