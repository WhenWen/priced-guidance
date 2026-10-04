import json
import math
import os
import signal
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pytest

from tech_tree_arena import (
    Checkout,
    Choice,
    Idea,
    IdeaVerdict,
    Option,
    Question,
    Submission,
    SubmitOption,
)
from tech_tree_arena import cli as arena_cli
from tech_tree_arena.cli import _run_submission
from tech_tree_arena.contract.validation import message_hash
from tech_tree_arena.contract.recovery import CURRENT_ACCOUNTING, LEGACY_ACCOUNTING, occurrence_bits
from tech_tree_arena.errors import (
    InvalidCheckout,
    ParticipantFailure,
    RecordedRunFailure,
    ReplayDivergence,
    ResourceLimitExceeded,
)
from tech_tree_arena.evaluation.smoke import SmokeAnswerJudge
from tech_tree_arena.replay.recorder import (
    HashChainWriter,
    RunRecorder,
    _accumulate_model_usage,
    _decode_question,
    _event_payload,
    _jsonable,
    _partition_resume_service_tail,
    _redacted_public_events,
    _replay_protocol_events,
    _service_event_data,
    actor_replay,
    protocol_replay,
    verify_hash_chain,
)
from tech_tree_arena.runtime.actor import ActorFactory
from tech_tree_arena.runtime.engine import ArenaRunner, RunLimits
from tech_tree_arena.runtime.services import ServiceEvent, ServiceFactory


ROOT = Path(__file__).resolve().parents[2]


def _rewrite_hash_chain(path: Path, payloads) -> None:
    path.unlink()
    writer = HashChainWriter(path)
    for payload in payloads:
        writer.append(dict(payload))


def test_durable_hash_chain_cursor_ignores_only_uncommitted_tail(tmp_path: Path) -> None:
    path = tmp_path / "events.jsonl"
    writer = HashChainWriter(path)
    writer.append({"kind": "one"})
    writer.append({"kind": "two"})
    with path.open("a", encoding="utf-8") as stream:
        stream.write('{"partial"')

    assert len(verify_hash_chain(path, max_records=2)) == 2
    assert len(verify_hash_chain(path, tolerate_truncated_tail=True)) == 2
    with pytest.raises(ReplayDivergence, match="invalid JSON"):
        verify_hash_chain(path)


def _service_record(role: str, kind: str, label: str) -> dict[str, object]:
    return {
        "kind": "service_call",
        "role": role,
        "service": _service_event_data(
            ServiceEvent(kind, label, {"label": label}, request={"label": label})
        ),
    }


def test_resume_tail_commits_nested_judge_calls_bound_to_replayed_guide_proxy() -> None:
    records = (
        _service_record("generator", "random.random", "committed-prefix"),
        _service_record("judge", "model.structured", "old-coarse"),
        _service_record("judge", "model.structured", "old-strict"),
        _service_record("oracle", "judge.evaluate", "old-preview"),
        _service_record("oracle", "model.structured", "old-choice"),
    )

    recorded, replay, committed_judge = _partition_resume_service_tail(
        records,
        1,
        retry_interrupted_call=False,
        discard_service_tail=False,
    )

    assert [event.request_hash for event in recorded["judge"]] == [
        "old-coarse", "old-strict",
    ]
    assert replay["judge"] == []
    assert [event.request_hash for event in committed_judge] == [
        "old-coarse", "old-strict",
    ]
    assert [event.request_hash for event in replay["oracle"]] == [
        "old-preview", "old-choice",
    ]


def test_resume_tail_keeps_unbound_nested_judge_calls_replayable() -> None:
    records = (
        _service_record("judge", "model.structured", "unfinished-preview"),
        _service_record("oracle", "judge.evaluate", "failed-proxy"),
    )

    _, replay, committed_judge = _partition_resume_service_tail(
        records,
        0,
        retry_interrupted_call=True,
        discard_service_tail=False,
    )

    assert [event.request_hash for event in replay["judge"]] == [
        "unfinished-preview"
    ]
    assert replay["oracle"] == []
    assert committed_judge == []


def test_discard_service_tail_neither_accounts_nor_replays_suffix() -> None:
    records = (
        _service_record("judge", "model.structured", "old-coarse"),
        _service_record("oracle", "judge.evaluate", "old-preview"),
    )

    recorded, replay, committed_judge = _partition_resume_service_tail(
        records,
        0,
        retry_interrupted_call=False,
        discard_service_tail=True,
    )

    assert recorded == {"generator": [], "oracle": [], "judge": []}
    assert replay == {"generator": [], "oracle": [], "judge": []}
    assert committed_judge == []


def test_resume_usage_ignores_diagnostic_snapshot_on_judge_proxy() -> None:
    model = ServiceEvent(
        "model.structured",
        "model-call",
        {},
        metadata={"usage": {"calls": 1, "cost_usd": 0.25}},
    )
    proxy = ServiceEvent(
        "judge.evaluate",
        "judge-proxy",
        {},
        metadata={"usage": {"calls": 1, "cost_usd": 99.0}},
    )

    assert _accumulate_model_usage(
        {"calls": 4, "cost_usd": 1.0},
        (proxy, model),
    ) == {"calls": 5, "cost_usd": 1.25}


def test_recorded_smoke_run_supports_protocol_and_actor_replay(tmp_path: Path) -> None:
    run = subprocess.run(
        [
            sys.executable, "-m", "tech_tree_arena.cli", "run",
            str(ROOT / "submissions" / "examples" / "minimal_pair"),
            "--target-pack", "smoke", "--runs-dir", str(tmp_path),
        ],
        check=True,
        text=True,
        capture_output=True,
    )
    run_directory = json.loads(run.stdout)["run_dir"]
    for extra, expected in (([], "replayed"), (["--actor"], "actor-replayed")):
        replay = subprocess.run(
            [sys.executable, "-m", "tech_tree_arena.cli", "replay", run_directory, *extra],
            check=True,
            text=True,
            capture_output=True,
        )
        assert json.loads(replay.stdout)["status"] == expected

    private = verify_hash_chain(Path(run_directory) / "events.private.jsonl")
    public = verify_hash_chain(Path(run_directory) / "events.public.jsonl")
    assert any(record["kind"] == "submission_judged" for record in private)
    assert all(record["kind"] != "submission_judged" for record in public)
    tampered = tuple(dict(record) for record in private)
    choice_cost = next(record for record in tampered if record["kind"] == "choice_cost")
    choice_cost["information_bits"] += 0.25
    with pytest.raises(ReplayDivergence, match="information bits"):
        _replay_protocol_events(tampered)

    judged = next(record for record in private if record["kind"] == "submission_judged")
    duplicated_judgment = tuple((*private[:-1], dict(judged), private[-1]))
    with pytest.raises(ReplayDivergence, match="judgment appears out of order"):
        _replay_protocol_events(duplicated_judgment)

    # Schema 3 artifacts retain their historical debt-bearing hashes and remain
    # replayable even though new schema-4 runs use pure active-path accounting.
    legacy_directory = tmp_path / "legacy-schema-3"
    shutil.copytree(run_directory, legacy_directory)
    legacy_private = [_event_payload(record) for record in private]
    legacy_private[0]["protocol_event_schema"] = 3
    legacy_private[0].pop("accounting_version")
    legacy_branches_path = legacy_directory / "branches.json"
    legacy_branches = json.loads(legacy_branches_path.read_text(encoding="utf-8"))
    legacy_branches.pop("accounting_version")
    question_index = 0
    removed_bits = 0.0
    for event in legacy_private:
        if event["kind"] == "choice_cost":
            removed_bits += event["branch_bits"]
            event["branch_bits"] = 0.0
            for field in ("accounting_version", "continuation_index", "option_index"):
                event.pop(field)
        if "path_k" in event:
            event["path_k"] -= removed_bits
        if event["kind"] == "question":
            event["failed_attempt_bits"] = 0.0
            event["integrity_hash"] = message_hash(
                {
                    "question_id": event["question_id"],
                    "parent_question_id": event["parent_question_id"],
                    "question": _decode_question(event["question"]),
                    "path_k": event["path_k"],
                    "failed_attempt_bits": 0.0,
                    "created_index": question_index,
                }
            )
            node = legacy_branches["nodes"][event["question_id"]]
            node["failed_attempt_bits"] = 0.0
            node["path_k"] = event["path_k"]
            node["integrity_hash"] = event["integrity_hash"]
            question_index += 1
        elif event["kind"] == "submission":
            event["failed_attempt_bits"] = 0.0
        elif event["kind"] == "submission_judged":
            event["failed_attempt_bits_after"] = 0.0
        elif event["kind"] == "run_finished":
            event["failed_attempt_bits"] = 0.0
            event["k"] -= removed_bits
            event["score"] = 2.0 ** -event["k"]
    legacy_score_path = legacy_directory / "score.json"
    legacy_score = json.loads(legacy_score_path.read_text())
    legacy_score.pop("accounting_version")
    legacy_score["K"] = legacy_private[-1]["k"]
    legacy_score["score"] = legacy_private[-1]["score"]
    legacy_score_path.write_text(json.dumps(legacy_score))
    legacy_branches_path.write_text(json.dumps(legacy_branches), encoding="utf-8")
    _rewrite_hash_chain(
        legacy_directory / "events.private.jsonl",
        legacy_private,
    )
    _rewrite_hash_chain(
        legacy_directory / "events.public.jsonl",
        _redacted_public_events(legacy_private, "development"),
    )
    assert protocol_replay(legacy_directory)["status"] == "replayed"


def test_actor_replay_replays_guide_judge_preview_without_live_capability(
    tmp_path: Path,
) -> None:
    submission = tmp_path / "oracle-preview-pair"
    participant = submission / "participant"
    participant.mkdir(parents=True)
    (submission / "submission.toml").write_text(
        """schema_version = 1
name = "oracle-preview-pair"
version = "0.1.0"
protocol = "idea-recovery-v1"
generator = "participant.generator:Generator"
oracle = "participant.oracle:Oracle"
""",
        encoding="utf-8",
    )
    (participant / "__init__.py").write_text("", encoding="utf-8")
    (participant / "generator.py").write_text(
        """from tech_tree_arena import Idea, Question, Submission, SubmitOption
class Generator:
    def __init__(self, services): self.services = services
    def step(self, choice):
        if choice is None:
            return Question("submit the answer", (SubmitOption("submit", "submit", "1"),))
        return Submission((Idea("answer", {"answer": "blue"}, "1"),))
""",
        encoding="utf-8",
    )
    (participant / "oracle.py").write_text(
        """from tech_tree_arena import Choice
class Oracle:
    def __init__(self, target, services): self.services = services
    def step(self, presented):
        verdicts = self.services.judge_evaluate([
            {"idea_id": "preview", "content": {"answer": "blue"}, "probability": "1"}
        ])
        assert verdicts[0]["passed"] is True
        return Choice("submit")
""",
        encoding="utf-8",
    )

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "tech_tree_arena.cli",
            "run",
            str(submission),
            "--target-pack",
            "smoke",
            "--runs-dir",
            str(tmp_path / "runs"),
        ],
        check=True,
        text=True,
        capture_output=True,
    )
    run_directory = Path(json.loads(completed.stdout)["run_dir"])

    replayed = subprocess.run(
        [
            sys.executable,
            "-m",
            "tech_tree_arena.cli",
            "replay",
            str(run_directory),
            "--actor",
        ],
        check=True,
        text=True,
        capture_output=True,
    )
    assert json.loads(replayed.stdout)["status"] == "actor-replayed"


def test_actor_replay_binds_outputs_to_the_question_events(tmp_path: Path) -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "tech_tree_arena.cli",
            "run",
            str(ROOT / "submissions" / "examples" / "minimal_pair"),
            "--target-pack",
            "smoke",
            "--runs-dir",
            str(tmp_path / "runs"),
        ],
        check=True,
        text=True,
        capture_output=True,
    )
    source = Path(json.loads(completed.stdout)["run_dir"])
    tampered = source.parent / "tampered-question"
    shutil.copytree(source, tampered)
    private = [
        _event_payload(record)
        for record in verify_hash_chain(tampered / "events.private.jsonl")
    ]
    question_event = next(event for event in private if event["kind"] == "question")
    question_event["question"]["question"] = "a different trusted question"
    decoded = _decode_question(question_event["question"])
    question_event["integrity_hash"] = message_hash(
        {
            "question_id": question_event["question_id"],
            "parent_question_id": question_event["parent_question_id"],
            "question": decoded,
            "path_k": question_event["path_k"],
            "created_index": 0,
        }
    )
    branches_path = tampered / "branches.json"
    branches = json.loads(branches_path.read_text(encoding="utf-8"))
    node = branches["nodes"][question_event["question_id"]]
    node["integrity_hash"] = question_event["integrity_hash"]
    node["question_hash"] = message_hash(decoded)
    branches_path.write_text(json.dumps(branches), encoding="utf-8")
    _rewrite_hash_chain(tampered / "events.private.jsonl", private)
    _rewrite_hash_chain(
        tampered / "events.public.jsonl",
        _redacted_public_events(private, "development"),
    )

    # Protocol replay accepts the internally consistent trusted trace. Actor
    # replay must additionally reject the unchanged participant output tape.
    assert protocol_replay(tampered)["status"] == "replayed"
    replayed = subprocess.run(
        [sys.executable, "-m", "tech_tree_arena.cli", "replay", str(tampered), "--actor"],
        check=False,
        text=True,
        capture_output=True,
    )
    assert replayed.returncode == 1
    assert "not bound to protocol events" in replayed.stderr


class TwoStepSubmissionGenerator:
    def __init__(self, services):
        self.services = services

    def step(self, choice):
        if choice is None:
            return Question(
                "first",
                (
                    SubmitOption("premature", None, "0.5"),
                    Option("continue", None, "0.5"),
                ),
            )
        if choice.option_id == "continue":
            return Question("second", (SubmitOption("submit", None, "1"),))
        return Submission((Idea("right", {"answer": "blue"}, "1"),))


class TwoStepSubmissionGuide:
    def __init__(self, _target, services):
        self.services = services

    def step(self, presented):
        return Choice("continue" if presented.question.question == "first" else "submit")


def test_protocol_replay_rejects_a_second_decision_while_generator_is_pending() -> None:
    events = []
    ArenaRunner(event_sink=events.append).run(
        generator_factory=ActorFactory(
            TwoStepSubmissionGenerator,
            service_factory=ServiceFactory(seed=1),
        ),
        guide_factory=ActorFactory(
            TwoStepSubmissionGuide,
            constructor_args=({"answer": "blue"},),
            service_factory=ServiceFactory(seed=2),
        ),
        target={"answer": "blue"},
        judge=SmokeAnswerJudge(),
        run_id="double-decision-replay-test",
    )
    trace = [_jsonable(event) for event in events]
    first_question = next(event for event in trace if event["kind"] == "question")
    first_guide_index = next(
        index for index, event in enumerate(trace) if event["kind"] == "oracle_decision"
    )
    injected = [
        {
            "kind": "oracle_decision",
            "question_id": first_question["question_id"],
            "decision": {
                "option_id": "premature",
                "question_id": None,
                "public_payload": None,
            },
        },
        {
            "kind": "choice_cost",
            "question_id": first_question["question_id"],
            "option_id": "premature",
            "probability": "0.5",
            "information_bits": 1.0,
            "branch_bits": occurrence_bits(1),
            "path_k": 1.0 + occurrence_bits(1),
            "accounting_version": CURRENT_ACCOUNTING,
            "continuation_index": 1,
            "option_index": 1,
        },
    ]
    invalid = trace[:first_guide_index] + injected + trace[first_guide_index:]

    # Account for the extra paid choice so every later numeric/hash field is
    # self-consistent. The only invalidity is asking the Oracle again before
    # consuming the selected SubmitOption with a Generator output.
    existing_choice = next(
        event
        for event in invalid[first_guide_index + len(injected):]
        if event["kind"] == "choice_cost"
    )
    existing_choice["branch_bits"] = 1.0
    existing_choice["path_k"] = 3.0
    second_question = [event for event in invalid if event["kind"] == "question"][1]
    second_question["path_k"] = 3.0
    decoded_second = _decode_question(second_question["question"])
    second_question["integrity_hash"] = message_hash(
        {
            "question_id": second_question["question_id"],
            "parent_question_id": second_question["parent_question_id"],
            "question": decoded_second,
            "path_k": 3.0,
            "created_index": 1,
        }
    )
    final_choice = [event for event in invalid if event["kind"] == "choice_cost"][-1]
    final_choice["path_k"] = 3.0
    submission = next(event for event in invalid if event["kind"] == "submission")
    submission["path_k"] = 3.0
    judged = next(event for event in invalid if event["kind"] == "submission_judged")
    judged["path_k"] = 3.0
    finished = invalid[-1]
    finished["k"] = 3.0
    finished["score"] = 0.125

    with pytest.raises(ReplayDivergence, match="oracle decision"):
        _replay_protocol_events(tuple(invalid))


def test_actor_replay_reconstructs_abandoned_and_surviving_generator_branches(
    tmp_path: Path,
) -> None:
    submission = tmp_path / "branching-pair"
    participant = submission / "participant"
    participant.mkdir(parents=True)
    (submission / "submission.toml").write_text(
        """schema_version = 1
name = "branching-pair"
version = "0.1.0"
protocol = "idea-recovery-v1"
generator = "participant.generator:Generator"
oracle = "participant.oracle:Oracle"
""",
        encoding="utf-8",
    )
    (participant / "__init__.py").write_text("", encoding="utf-8")
    (participant / "generator.py").write_text(
        """from tech_tree_arena import Choice, Idea, Option, Question, Submission, SubmitOption

class Generator:
    def __init__(self, services):
        self.services = services
        self.phase = "start"

    def step(self, choice: Choice | None):
        if choice is None:
            self.phase = "first"
            return Question("first", (
                Option("red", "red", "0.5"), SubmitOption("blue", "blue", "0.5")
            ))
        if self.phase == "first" and choice.option_id == "red":
            self.phase = "second"
            return Question("rewind", (Option("continue", "continue", "1"),))
        return Submission((Idea("answer", {"answer": choice.public_payload}, "1"),))
""",
        encoding="utf-8",
    )
    (participant / "oracle.py").write_text(
        """from tech_tree_arena import Checkout, Choice

class Oracle:
    def __init__(self, target, services):
        self.services = services
        self.first_id = None

    def step(self, presented):
        if presented.question.question == "first":
            if self.first_id is None:
                self.first_id = presented.question_id
                return Choice("red")
            return Choice("blue")
        return Checkout(self.first_id)
""",
        encoding="utf-8",
    )

    run = subprocess.run(
        [
            sys.executable,
            "-m",
            "tech_tree_arena.cli",
            "run",
            str(submission),
            "--target-pack",
            "smoke",
            "--runs-dir",
            str(tmp_path / "runs"),
        ],
        check=True,
        text=True,
        capture_output=True,
    )
    run_directory = json.loads(run.stdout)["run_dir"]
    replay = subprocess.run(
        [sys.executable, "-m", "tech_tree_arena.cli", "replay", run_directory, "--actor"],
        check=True,
        text=True,
        capture_output=True,
    )
    replayed = json.loads(replay.stdout)
    assert replayed["branches"] == {"generator": 2, "oracle": 1}
    assert replayed["calls"] == {"generator": 4, "oracle": 3}

    private = [
        _event_payload(record)
        for record in verify_hash_chain(Path(run_directory) / "events.private.jsonl")
    ]
    explicit_ordinary_context = [dict(event) for event in private]
    ordinary_decision = next(
        event for event in explicit_ordinary_context if event["kind"] == "oracle_decision"
    )
    ordinary_decision["context"] = "question"
    with pytest.raises(ReplayDivergence, match="invalid context"):
        _replay_protocol_events(tuple(explicit_ordinary_context))

    contextual_checkout = [dict(event) for event in private]
    checkout = next(event for event in contextual_checkout if event["kind"] == "checkout")
    checkout["context"] = "submission_recovery"
    with pytest.raises(ReplayDivergence, match="ordinary checkout"):
        _replay_protocol_events(tuple(contextual_checkout))


def test_rejected_submission_recovery_is_recorded_and_actor_replayable(
    tmp_path: Path,
) -> None:
    submission = tmp_path / "retry-pair"
    participant = submission / "participant"
    participant.mkdir(parents=True)
    (submission / "submission.toml").write_text(
        """schema_version = 1
name = "retry-pair"
version = "0.1.0"
protocol = "idea-recovery-v1"
generator = "participant.generator:Generator"
oracle = "participant.oracle:Oracle"
""",
        encoding="utf-8",
    )
    (participant / "__init__.py").write_text("", encoding="utf-8")
    (participant / "generator.py").write_text(
        """from tech_tree_arena import Idea, Option, Question, Submission, SubmitOption

class Generator:
    def __init__(self, services): self.services = services
    def step(self, choice):
        if choice is None:
            self.services.random()
            return Question("first", (
                SubmitOption("try", "wrong", "0.25"),
                Option("revise", "revise", "0.75"),
            ))
        if choice.option_id == "try":
            return Submission((Idea("wrong", {"answer": "red"}, "1"),))
        if choice.option_id == "revise":
            return Question("second", (SubmitOption("submit", "right", "1"),))
        return Submission((Idea("right", {"answer": "blue"}, "1"),))
""",
        encoding="utf-8",
    )
    (participant / "oracle.py").write_text(
        """from tech_tree_arena import Checkout, Choice, SubmissionFeedback

class Oracle:
    def __init__(self, target, services): self.failed = False
    def step(self, message):
        if isinstance(message, SubmissionFeedback):
            assert [verdict.passed for verdict in message.verdicts] == [False]
            assert message.valid_checkout_question_ids == (message.source.question_id,)
            self.failed = True
            return Checkout(message.source.question_id)
        if message.question.question == "first":
            return Choice("revise" if self.failed else "try")
        return Choice("submit")
""",
        encoding="utf-8",
    )

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "tech_tree_arena.cli",
            "run",
            str(submission),
            "--target-pack",
            "smoke",
            "--runs-dir",
            str(tmp_path / "runs"),
        ],
        check=True,
        text=True,
        capture_output=True,
    )
    result = json.loads(completed.stdout)
    assert result["status"] == "pass"
    assert result["submission_attempts"] == 2
    assert "failed_attempt_bits" not in result
    assert result["checkouts"] == 1
    assert result["K"] == pytest.approx(-math.log2(0.75) + 2 * occurrence_bits(1))

    run_directory = Path(result["run_dir"])
    private = verify_hash_chain(run_directory / "events.private.jsonl")
    assert [
        record["status"]
        for record in private
        if record["kind"] == "submission_judged"
    ] == ["fail", "pass"]
    public = verify_hash_chain(run_directory / "events.public.jsonl")
    public_payloads = [_event_payload(record) for record in public]
    assert all(
        record.get("kind") not in {"submission", "submission_judged", "checkout"}
        for record in public_payloads
    )
    assert all("context" not in record for record in public_payloads)
    public_terminal = public_payloads[-1]
    assert public_terminal["kind"] == "run_finished"
    assert "submission_attempts" not in public_terminal
    assert "failed_attempt_bits" not in public_terminal
    assert protocol_replay(run_directory)["status"] == "replayed"

    replay = subprocess.run(
        [
            sys.executable,
            "-m",
            "tech_tree_arena.cli",
            "replay",
            str(run_directory),
            "--actor",
        ],
        check=True,
        text=True,
        capture_output=True,
    )
    replayed = json.loads(replay.stdout)
    assert replayed["status"] == "actor-replayed"
    assert replayed["judge_replayed"] is True
    assert replayed["branches"] == {"generator": 2, "oracle": 1}


def test_resume_from_rejected_submission_feedback_preserves_attempt_state(
    tmp_path: Path,
) -> None:
    submission = tmp_path / "resumable-retry-pair"
    participant = submission / "participant"
    participant.mkdir(parents=True)
    marker = tmp_path / "allow-feedback"
    (submission / "submission.toml").write_text(
        """schema_version = 1
name = "resumable-retry-pair"
version = "0.1.0"
protocol = "idea-recovery-v1"
generator = "participant.generator:Generator"
oracle = "participant.oracle:Oracle"
""",
        encoding="utf-8",
    )
    (participant / "__init__.py").write_text("", encoding="utf-8")
    (participant / "generator.py").write_text(
        """from tech_tree_arena import Idea, Option, Question, Submission, SubmitOption
class Generator:
    def __init__(self, services): self.services = services
    def step(self, choice):
        if choice is None:
            self.services.random()
            return Question("first", (
                SubmitOption("try", "wrong", "0.25"),
                Option("revise", "revise", "0.75"),
            ))
        if choice.option_id == "try":
            return Submission((Idea("wrong", {"answer": "red"}, "1"),))
        if choice.option_id == "revise":
            return Question("second", (SubmitOption("submit", "right", "1"),))
        return Submission((Idea("right", {"answer": "blue"}, "1"),))
""",
        encoding="utf-8",
    )
    (participant / "oracle.py").write_text(
        """import os
from pathlib import Path
from tech_tree_arena import Checkout, Choice, SubmissionFeedback
class Oracle:
    def __init__(self, target, services): self.failed = False
    def step(self, message):
        if isinstance(message, SubmissionFeedback):
            if not Path(os.environ["IDEA_ARENA_TEST_RESUME_MARKER"]).is_file():
                raise RuntimeError("intentional crash before recovery checkout")
            self.failed = True
            return Checkout(message.source.question_id)
        if message.question.question == "first":
            return Choice("revise" if self.failed else "try")
        return Choice("submit")
""",
        encoding="utf-8",
    )

    runs = tmp_path / "runs"
    environment = {**os.environ, "IDEA_ARENA_TEST_RESUME_MARKER": str(marker)}
    failed = subprocess.run(
        [
            sys.executable,
            "-m",
            "tech_tree_arena.cli",
            "run",
            str(submission),
            "--target-pack",
            "smoke",
            "--runs-dir",
            str(runs),
        ],
        check=False,
        text=True,
        capture_output=True,
        env=environment,
    )
    assert failed.returncode == 1
    source = next(path for path in runs.iterdir() if path.name != ".artifacts")
    status = json.loads((source / "status.json").read_text(encoding="utf-8"))
    assert status["phase"] == "await_submission_recovery"
    assert status["resumable"] is True
    checkpoint = json.loads(
        (source / "checkpoint.private.json").read_text(encoding="utf-8")
    )
    assert checkpoint["engine"]["submission_attempts"] == 1
    assert "failed_attempt_bits" not in checkpoint["engine"]
    assert checkpoint["engine"]["submission_feedback"]["type"] == "submission_feedback"
    failed_actor_replay = subprocess.run(
        [
            sys.executable,
            "-m",
            "tech_tree_arena.cli",
            "replay",
            str(source),
            "--actor",
        ],
        check=True,
        text=True,
        capture_output=True,
        env=environment,
    )
    assert json.loads(failed_actor_replay.stdout)["judge_replayed"] is True

    def path_k_tamper(value):
        value["engine"]["k"] = 0.0

    def feedback_tamper(value):
        value["engine"]["submission_feedback"]["value"][
            "valid_checkout_question_ids"
        ] = []

    def capability_tamper(value):
        value["branches"]["capability_secret"] = "00" * 32

    def node_cursor_tamper(value):
        node = next(iter(value["branches"]["nodes"].values()))
        node["generator_checkpoint"]["call_count"] = 0

    def guide_cursor_tamper(value):
        value["actors"]["oracle"]["call_count"] = 0

    def service_meter_tamper(value):
        value["actors"]["generator"]["service_state"]["meter"][
            "random_calls"
        ] = 0

    def service_call_cursor_tamper(value):
        value["actor_streams"]["oracle"]["root"]["calls"][0][
            "service_event_count"
        ] = 1

    for name, mutate in (
        ("tampered-path-k", path_k_tamper),
        ("tampered-feedback", feedback_tamper),
        ("tampered-capability", capability_tamper),
        ("tampered-node-cursor", node_cursor_tamper),
        ("tampered-oracle-cursor", guide_cursor_tamper),
        ("tampered-service-meter", service_meter_tamper),
        ("tampered-service-call-cursor", service_call_cursor_tamper),
    ):
        tampered = runs / name
        shutil.copytree(source, tampered)
        tampered_checkpoint_path = tampered / "checkpoint.private.json"
        tampered_checkpoint = json.loads(
            tampered_checkpoint_path.read_text(encoding="utf-8")
        )
        mutate(tampered_checkpoint)
        tampered_checkpoint_path.write_text(
            json.dumps(tampered_checkpoint),
            encoding="utf-8",
        )
        rejected = subprocess.run(
            [sys.executable, "-m", "tech_tree_arena.cli", "resume", str(tampered)],
            check=False,
            text=True,
            capture_output=True,
            env=environment,
        )
        assert rejected.returncode == 1
        assert rejected.stderr.startswith("idea-arena:")
        if name == "tampered-service-meter":
            actor_rejected = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "tech_tree_arena.cli",
                    "replay",
                    str(tampered),
                    "--actor",
                ],
                check=False,
                text=True,
                capture_output=True,
                env=environment,
            )
            assert actor_rejected.returncode == 1
            assert "service meter disagrees with its journal" in actor_rejected.stderr

    marker.write_text("ok\n", encoding="utf-8")
    resumed = subprocess.run(
        [sys.executable, "-m", "tech_tree_arena.cli", "resume", str(source)],
        check=True,
        text=True,
        capture_output=True,
        env=environment,
    )
    result = json.loads(resumed.stdout)
    assert result["status"] == "pass"
    assert result["submission_attempts"] == 2
    assert "failed_attempt_bits" not in result
    assert result["checkouts"] == 1
    assert result["K"] == pytest.approx(-math.log2(0.75) + 2 * occurrence_bits(1))

    replayed = subprocess.run(
        [
            sys.executable,
            "-m",
            "tech_tree_arena.cli",
            "replay",
            result["run_dir"],
            "--actor",
        ],
        check=True,
        text=True,
        capture_output=True,
        env=environment,
    )
    assert json.loads(replayed.stdout)["judge_replayed"] is True


class OneQuestionGenerator:
    def __init__(self, services):
        self.services = services

    def step(self, _choice):
        return Question("only", (Option("yes", "yes", "1"),))


class InvalidCheckoutGuide:
    def __init__(self, _target, services):
        self.services = services

    def step(self, _question):
        return Checkout("missing-question")


class MultiTargetFailureGenerator:
    def __init__(self, services):
        self.phase = "root"

    def step(self, choice):
        if choice is None:
            self.phase = "first"
            return Question(
                "first",
                (
                    Option("continue", None, "0.5"),
                    Option("crash", None, "0.5"),
                ),
            )
        if choice.option_id == "crash":
            raise RuntimeError("intentional post-checkout failure")
        if self.phase == "first":
            self.phase = "second"
            return Question("second", (Option("continue-2", None, "1"),))
        self.phase = "third"
        return Question("third", (Option("unused", None, "1"),))


class MultiTargetFailureGuide:
    def __init__(self, _target, services):
        self.first_id = None
        self.revisited = False

    def step(self, presented):
        if presented.question.question == "first":
            if self.first_id is None:
                self.first_id = presented.question_id
                return Choice("continue")
            self.revisited = True
            return Choice("crash")
        if presented.question.question == "second":
            return Choice("continue-2")
        return Checkout(self.first_id)


def test_submit_aware_error_before_any_judgment_replays_free_checkout(
    tmp_path: Path,
) -> None:
    recorder = RunRecorder(
        tmp_path,
        "pre-judgment-error",
        {
            "seed": 13,
            "budgets": {
                "questions": 256,
                "oracle_decisions": 256,
                "checkouts": 64,
                "checkout_targets": 128,
                "checkout_rewind": 256,
                "question_depth": 256,
                "submission_attempts": 256,
                "information_bits": 1_024.0,
            },
        },
    )
    runner = ArenaRunner(event_sink=recorder.record)
    judge = SmokeAnswerJudge()
    with pytest.raises(ParticipantFailure) as failure:
        runner.run(
            generator_factory=ActorFactory(
                MultiTargetFailureGenerator,
                service_factory=ServiceFactory(seed=1),
            ),
            guide_factory=ActorFactory(
                MultiTargetFailureGuide,
                constructor_args=({},),
                service_factory=ServiceFactory(seed=2),
            ),
            target={},
            judge=judge,
            seed=13,
            run_id="pre-judgment-error",
        )
    error_code = getattr(failure.value, "code", "match_failure")
    recorder.record(
        {
            "kind": "run_finished",
            "status": "error",
            "score": 0.0,
            "error_code": error_code,
        }
    )
    recorder.finalize_failure(error_code, runner, judge)

    replayed = protocol_replay(recorder.root)
    assert replayed["result"]["status"] == "error"
    assert replayed["result"]["checkouts"] == 1
    branches = json.loads((recorder.root / "branches.json").read_text(encoding="utf-8"))
    assert branches["checkout_audit"][-1]["target_count"] == 2
    assert branches["checkout_audit"][-1]["branch_bits"] == 0.0


class PointerBudgetGenerator:
    def __init__(self, services):
        self.services = services

    def step(self, choice):
        if choice is None:
            return Question("submit", (SubmitOption("submit", None, "1"),))
        return Submission(
            (
                Idea("match", {"answer": "blue"}, "0.25"),
                Idea("miss", {"answer": "red"}, "0.75"),
            )
        )


class PointerBudgetGuide:
    def __init__(self, _target, services):
        self.services = services

    def step(self, _message):
        return Choice("submit")


class OneOfThreeJudge:
    def __init__(self):
        self.calls = 0

    def evaluate(self, _target, ideas):
        self.calls += 1
        return tuple(
            IdeaVerdict(idea.idea_id, self.calls == 2 and idea.idea_id == "match")
            for idea in ideas
        )


def test_repeat_judge_cost_is_durable_and_protocol_replayable(tmp_path: Path) -> None:
    budgets = {
        "questions": 256,
        "oracle_decisions": 256,
        "checkouts": 64,
        "checkout_targets": 128,
        "checkout_rewind": 256,
        "question_depth": 256,
        "submission_attempts": 256,
        "information_bits": 1_024.0,
    }
    recorder = RunRecorder(
        tmp_path,
        "repeat-judge-replay",
        {"seed": 19, "budgets": budgets, "judge_config": {"repeats": 3}},
    )
    judge = OneOfThreeJudge()
    runner = ArenaRunner(judge_repeats=3, event_sink=recorder.record)
    result = runner.run(
        generator_factory=ActorFactory(
            PointerBudgetGenerator,
            service_factory=ServiceFactory(seed=1),
        ),
        guide_factory=ActorFactory(
            PointerBudgetGuide,
            constructor_args=({"answer": "blue"},),
            service_factory=ServiceFactory(seed=2),
        ),
        target={"answer": "blue"},
        judge=judge,
        seed=19,
        run_id="repeat-judge-replay",
    )
    recorder.finalize(result, runner, judge)

    replayed = protocol_replay(recorder.root)["result"]
    assert replayed["status"] == "pass"
    assert replayed["judge_repeats"] == 3
    assert replayed["judge_passes"] == 1
    assert replayed["K"] == pytest.approx(2.0 + math.log2(3) + occurrence_bits(1))

    private = tuple(
        _event_payload(record)
        for record in verify_hash_chain(recorder.root / "events.private.jsonl")
    )
    tampered = tuple(dict(record) for record in private)
    judged = next(record for record in tampered if record["kind"] == "submission_judged")
    judged["judge_passes"] = 2
    with pytest.raises(ReplayDivergence, match="judge_passes"):
        _replay_protocol_events(tampered, limits=budgets)


class ChoiceBudgetGenerator:
    def __init__(self, services):
        self.services = services

    def step(self, choice):
        if choice is None:
            return Question(
                "submit",
                (
                    SubmitOption("submit", None, "0.5"),
                    Option("continue", None, "0.5"),
                ),
            )
        raise AssertionError("an unaffordable choice must not reach the Generator")


def test_choice_budget_failure_exports_replayable_branch_counters(
    tmp_path: Path,
) -> None:
    budgets = {
        "questions": 256,
        "oracle_decisions": 256,
        "checkouts": 64,
        "checkout_targets": 128,
        "checkout_rewind": 256,
        "question_depth": 256,
        "submission_attempts": 256,
        "information_bits": 0.5,
    }
    recorder = RunRecorder(
        tmp_path,
        "choice-budget-error",
        {"seed": 17, "budgets": budgets},
    )
    runner = ArenaRunner(
        event_sink=recorder.record,
        limits=RunLimits(max_bits=0.5),
    )
    judge = SmokeAnswerJudge()
    with pytest.raises(ResourceLimitExceeded) as failure:
        runner.run(
            generator_factory=ActorFactory(
                ChoiceBudgetGenerator,
                service_factory=ServiceFactory(seed=1),
            ),
            guide_factory=ActorFactory(
                PointerBudgetGuide,
                constructor_args=({"answer": "blue"},),
                service_factory=ServiceFactory(seed=2),
            ),
            target={"answer": "blue"},
            judge=judge,
            seed=17,
            run_id="choice-budget-error",
        )
    recorder.record({
        "kind": "run_finished",
        "status": "error",
        "score": 0.0,
        "error_code": failure.value.code,
    })
    recorder.finalize_failure(failure.value.code, runner, judge)

    assert runner.last_branches.continuation_counts == {}
    assert runner.last_branches.option_counts == {}
    assert protocol_replay(recorder.root)["result"]["status"] == "error"


def test_protocol_replay_enforces_pointer_bits_on_error_and_pass_traces(
    tmp_path: Path,
) -> None:
    budgets = {
        "questions": 256,
        "oracle_decisions": 256,
        "checkouts": 64,
        "checkout_targets": 128,
        "checkout_rewind": 256,
        "question_depth": 256,
        "submission_attempts": 256,
        "information_bits": 1.0,
    }
    recorder = RunRecorder(
        tmp_path,
        "pointer-budget-error",
        {"seed": 17, "budgets": budgets},
    )
    runner = ArenaRunner(
        event_sink=recorder.record,
        limits=RunLimits(max_bits=1.0),
    )
    judge = SmokeAnswerJudge()
    with pytest.raises(ResourceLimitExceeded) as failure:
        runner.run(
            generator_factory=ActorFactory(
                PointerBudgetGenerator,
                service_factory=ServiceFactory(seed=1),
            ),
            guide_factory=ActorFactory(
                PointerBudgetGuide,
                constructor_args=({"answer": "blue"},),
                service_factory=ServiceFactory(seed=2),
            ),
            target={"answer": "blue"},
            judge=judge,
            seed=17,
            run_id="pointer-budget-error",
        )
    recorder.record(
        {
            "kind": "run_finished",
            "status": "error",
            "score": 0.0,
            "error_code": failure.value.code,
        }
    )
    recorder.finalize_failure(failure.value.code, runner, judge)
    assert protocol_replay(recorder.root)["result"]["status"] == "error"

    private = [dict(record) for record in verify_hash_chain(
        recorder.root / "events.private.jsonl"
    )]
    judged = next(record for record in private if record["kind"] == "submission_judged")
    terminal = private[-1]
    terminal.update(
        status="pass",
        score=0.25,
        k=2.0,
        matched_idea_ids=["match"],
        verdicts=judged["verdicts"],
        passing_mass="0.25",
        submission_bits=2.0,
        submission_attempts=1,
    )
    with pytest.raises(ReplayDivergence, match="information-cost exhaustion"):
        _replay_protocol_events(tuple(private), limits=budgets)


def test_partially_completed_failure_run_is_protocol_replayable(tmp_path: Path) -> None:
    recorder = RunRecorder(tmp_path, "failed-run", {"seed": 11, "budgets": {}})
    runner = ArenaRunner(event_sink=recorder.record)
    judge = SmokeAnswerJudge()
    with pytest.raises(InvalidCheckout) as failure:
        runner.run(
            generator_factory=ActorFactory(
                OneQuestionGenerator, service_factory=ServiceFactory(seed=1)
            ),
            guide_factory=ActorFactory(
                InvalidCheckoutGuide,
                constructor_args=({},),
                service_factory=ServiceFactory(seed=2),
            ),
            target={},
            judge=judge,
            seed=11,
            run_id="failed-run",
        )
    recorder.record(
        {
            "kind": "run_finished",
            "status": "error",
            "score": 0.0,
            "error_code": failure.value.code,
        }
    )
    recorder.finalize_failure(failure.value.code, runner, judge)

    replayed = protocol_replay(recorder.root)
    assert replayed["result"]["status"] == "error"
    usage = json.loads((recorder.root / "usage.json").read_text(encoding="utf-8"))
    assert set(usage) == {"generator", "oracle"}


def test_failed_subprocess_run_resumes_from_last_durable_boundary(tmp_path: Path) -> None:
    submission = tmp_path / "resumable-pair"
    participant = submission / "participant"
    participant.mkdir(parents=True)
    marker = tmp_path / "allow-finish"
    (submission / "submission.toml").write_text(
        """schema_version = 1
name = "resumable-pair"
version = "0.1.0"
protocol = "idea-recovery-v1"
generator = "participant.generator:Generator"
oracle = "participant.oracle:Oracle"
""",
        encoding="utf-8",
    )
    (participant / "__init__.py").write_text("", encoding="utf-8")
    (participant / "generator.py").write_text(
        """import os
from pathlib import Path
from tech_tree_arena import Idea, Question, Submission, SubmitOption

class Generator:
    def __init__(self, services):
        self.services = services
    def step(self, choice):
        if choice is None:
            return Question("recover", (SubmitOption("blue", "blue", "1"),))
        if not Path(os.environ["IDEA_ARENA_TEST_RESUME_MARKER"]).is_file():
            raise RuntimeError("intentional crash before submission")
        return Submission((Idea("answer", {"answer": choice.public_payload}, "1"),))
""",
        encoding="utf-8",
    )
    (participant / "oracle.py").write_text(
        """from tech_tree_arena import Choice
class Oracle:
    def __init__(self, target, services): self.services = services
    def step(self, presented): return Choice("blue")
""",
        encoding="utf-8",
    )
    runs = tmp_path / "runs"
    environment = {**os.environ, "IDEA_ARENA_TEST_RESUME_MARKER": str(marker)}
    failed = subprocess.run(
        [
            sys.executable, "-m", "tech_tree_arena.cli", "run", str(submission),
            "--target-pack", "smoke", "--runs-dir", str(runs),
        ],
        check=False,
        text=True,
        capture_output=True,
        env=environment,
    )
    assert failed.returncode == 1
    source = next(path for path in runs.iterdir() if path.is_dir() and path.name != ".artifacts")
    source_status = json.loads((source / "status.json").read_text(encoding="utf-8"))
    assert source_status["status"] == "error"
    assert source_status["phase"] == "after_oracle"
    assert source_status["resumable"] is True
    assert (source / "logs" / "generator.log").stat().st_size > 0
    durable = json.loads((source / "checkpoint.private.json").read_text(encoding="utf-8"))
    assert durable["schema_version"] == 3
    assert set(durable["actor_streams"]) == {"generator", "oracle"}
    node_checkpoint = next(iter(durable["branches"]["nodes"].values()))[
        "generator_checkpoint"
    ]
    assert "call_count" in node_checkpoint
    assert "calls" not in node_checkpoint
    failed_replay = subprocess.run(
        [sys.executable, "-m", "tech_tree_arena.cli", "replay", str(source), "--actor"],
        check=True,
        text=True,
        capture_output=True,
        env=environment,
    )
    failed_replay_result = json.loads(failed_replay.stdout)
    assert failed_replay_result["status"] == "actor-replayed"
    assert failed_replay_result["judge_replayed"] is False

    marker.write_text("ok\n", encoding="utf-8")
    resumed = subprocess.run(
        [sys.executable, "-m", "tech_tree_arena.cli", "resume", str(source)],
        check=True,
        text=True,
        capture_output=True,
        env=environment,
    )
    result = json.loads(resumed.stdout)
    assert result["status"] == "pass"
    assert result["questions"] == 1
    assert result["oracle_decisions"] == 1
    assert result["resumed_from"] == source.name
    replayed = subprocess.run(
        [sys.executable, "-m", "tech_tree_arena.cli", "replay", result["run_dir"], "--actor"],
        check=True,
        text=True,
        capture_output=True,
        env=environment,
    )
    assert json.loads(replayed.stdout)["status"] == "actor-replayed"


def test_resume_replays_failed_then_successful_service_tail_in_order(
    tmp_path: Path,
) -> None:
    submission = tmp_path / "service-tail-pair"
    participant = submission / "participant"
    participant.mkdir(parents=True)
    marker = tmp_path / "allow-service-tail-finish"
    (submission / "submission.toml").write_text(
        """schema_version = 1
name = "service-tail-pair"
version = "0.1.0"
protocol = "idea-recovery-v1"
generator = "participant.generator:Generator"
oracle = "participant.oracle:Oracle"
""",
        encoding="utf-8",
    )
    (participant / "__init__.py").write_text("", encoding="utf-8")
    (participant / "generator.py").write_text(
        """import os
from pathlib import Path
from tech_tree_arena import Idea, Question, Submission, SubmitOption

class Generator:
    def __init__(self, services): self.services = services
    def step(self, choice):
        if choice is None:
            return Question("service tail", (SubmitOption("submit", None, "1"),))
        try:
            self.services.choice([])
        except Exception:
            pass
        draw = self.services.random()
        if not Path(os.environ["IDEA_ARENA_TEST_RESUME_MARKER"]).is_file():
            raise RuntimeError("intentional failure after service retry")
        return Submission((Idea("answer", {"answer": "blue", "draw": draw}, "1"),))
""",
        encoding="utf-8",
    )
    (participant / "oracle.py").write_text(
        """from tech_tree_arena import Choice
class Oracle:
    def __init__(self, target, services): self.services = services
    def step(self, presented): return Choice("submit")
""",
        encoding="utf-8",
    )

    runs = tmp_path / "runs"
    environment = {**os.environ, "IDEA_ARENA_TEST_RESUME_MARKER": str(marker)}
    failed = subprocess.run(
        [
            sys.executable,
            "-m",
            "tech_tree_arena.cli",
            "run",
            str(submission),
            "--target-pack",
            "smoke",
            "--runs-dir",
            str(runs),
        ],
        check=False,
        text=True,
        capture_output=True,
        env=environment,
    )
    assert failed.returncode == 1
    source = next(path for path in runs.iterdir() if path.name != ".artifacts")
    source_services = [
        _event_payload(record)
        for record in verify_hash_chain(source / "service-calls.private.jsonl")
    ]
    assert [record["service"]["error"] for record in source_services] == [
        "IndexError",
        None,
    ]
    assert [
        record["service"]["metadata"]["service_meter"]["random_calls"]
        for record in source_services
    ] == [1, 2]

    replayed_failure = subprocess.run(
        [sys.executable, "-m", "tech_tree_arena.cli", "replay", str(source), "--actor"],
        check=True,
        text=True,
        capture_output=True,
        env=environment,
    )
    assert json.loads(replayed_failure.stdout)["unreplayed_service_events"] == {
        "generator": 2
    }

    # The same tail is not legal merely because the run failed. Removing the
    # exact in-flight call marker turns it into unexplained trailing data.
    unmarked = runs / "unmarked-service-tail"
    shutil.copytree(source, unmarked)
    unmarked_private = [
        _event_payload(record)
        for record in verify_hash_chain(unmarked / "events.private.jsonl")
    ]
    unmarked_private[-1].pop("interrupted_call")
    _rewrite_hash_chain(unmarked / "events.private.jsonl", unmarked_private)
    _rewrite_hash_chain(
        unmarked / "events.public.jsonl",
        _redacted_public_events(unmarked_private, "development"),
    )
    rejected_tail = subprocess.run(
        [sys.executable, "-m", "tech_tree_arena.cli", "replay", str(unmarked), "--actor"],
        check=False,
        text=True,
        capture_output=True,
        env=environment,
    )
    assert rejected_tail.returncode == 1
    assert "post-checkpoint service tail" in rejected_tail.stderr

    # The actor-local tail is not an alternative source of truth. Every full
    # post-checkpoint service record must also exist in the write-ahead journal.
    missing_journal = runs / "missing-service-journal-tail"
    shutil.copytree(source, missing_journal)
    (missing_journal / "service-calls.private.jsonl").write_text("", encoding="utf-8")
    rejected_missing_journal = subprocess.run(
        [
            sys.executable,
            "-m",
            "tech_tree_arena.cli",
            "replay",
            str(missing_journal),
            "--actor",
        ],
        check=False,
        text=True,
        capture_output=True,
        env=environment,
    )
    assert rejected_missing_journal.returncode == 1
    assert "disagrees with its journal" in rejected_missing_journal.stderr

    # Even mutually edited journal/tape copies cannot invent valid resource
    # accounting or RNG/branch metadata for the interrupted call.
    for name in ("meter", "branch", "rng", "usage"):
        tampered = runs / f"tampered-service-tail-{name}"
        shutil.copytree(source, tampered)
        journal = [
            _event_payload(record)
            for record in verify_hash_chain(tampered / "service-calls.private.jsonl")
        ]
        actor_tape_path = tampered / "service-tape.generator.private.json"
        actor_tape = json.loads(actor_tape_path.read_text(encoding="utf-8"))
        journal_service = journal[0]["service"]
        tape_service = actor_tape[0]
        if name == "meter":
            journal_service["metadata"]["service_meter"]["random_calls"] = 2
            tape_service["metadata"]["service_meter"]["random_calls"] = 2
        elif name == "branch":
            journal_service["metadata"]["service_branch_id"] = "forged-branch"
            tape_service["metadata"]["service_branch_id"] = "forged-branch"
        elif name == "rng":
            journal_service["metadata"]["rng_state_after"] = []
            tape_service["metadata"]["rng_state_after"] = []
        else:
            journal_service["metadata"]["usage"] = {"input_tokens": 1}
            tape_service["metadata"]["usage"] = {"input_tokens": 1}
        _rewrite_hash_chain(tampered / "service-calls.private.jsonl", journal)
        actor_tape_path.write_text(json.dumps(actor_tape), encoding="utf-8")
        rejected_metadata = subprocess.run(
            [
                sys.executable,
                "-m",
                "tech_tree_arena.cli",
                "replay",
                str(tampered),
                "--actor",
            ],
            check=False,
            text=True,
            capture_output=True,
            env=environment,
        )
        assert rejected_metadata.returncode == 1
        assert rejected_metadata.stderr.startswith("idea-arena:")

    # A derived resume must own a durable baseline before its first continued
    # participant call.  With the marker still absent, that very first call
    # consumes the copied failure/success tail and fails again.
    existing_runs = set(runs.iterdir())
    resumed_failure = subprocess.run(
        [sys.executable, "-m", "tech_tree_arena.cli", "resume", str(source)],
        check=False,
        text=True,
        capture_output=True,
        env=environment,
    )
    assert resumed_failure.returncode == 1
    derived_failure = next(
        path
        for path in set(runs.iterdir()) - existing_runs
        if path.is_dir() and path.name != ".artifacts"
    )
    source_checkpoint = json.loads(
        (source / "checkpoint.private.json").read_text(encoding="utf-8")
    )
    derived_checkpoint = json.loads(
        (derived_failure / "checkpoint.private.json").read_text(encoding="utf-8")
    )
    assert derived_checkpoint["run_id"] == derived_failure.name
    assert derived_checkpoint["branches"]["run_id"] == derived_failure.name
    assert (
        derived_checkpoint["service_event_count"]
        == source_checkpoint["service_event_count"]
    )
    replayed_derived_failure = subprocess.run(
        [
            sys.executable,
            "-m",
            "tech_tree_arena.cli",
            "replay",
            str(derived_failure),
            "--actor",
        ],
        check=True,
        text=True,
        capture_output=True,
        env=environment,
    )
    assert json.loads(replayed_derived_failure.stdout)["unreplayed_service_events"] == {
        "generator": 2
    }

    derived_missing_journal = runs / "derived-missing-service-journal"
    shutil.copytree(derived_failure, derived_missing_journal)
    (derived_missing_journal / "service-calls.private.jsonl").write_text(
        "", encoding="utf-8"
    )
    rejected_derived_journal = subprocess.run(
        [
            sys.executable,
            "-m",
            "tech_tree_arena.cli",
            "replay",
            str(derived_missing_journal),
            "--actor",
        ],
        check=False,
        text=True,
        capture_output=True,
        env=environment,
    )
    assert rejected_derived_journal.returncode == 1
    assert "disagrees with its journal" in rejected_derived_journal.stderr

    derived_empty_tape = runs / "derived-empty-generator-tape"
    shutil.copytree(derived_failure, derived_empty_tape)
    (derived_empty_tape / "service-tape.generator.private.json").write_text(
        "[]", encoding="utf-8"
    )
    rejected_derived_tape = subprocess.run(
        [
            sys.executable,
            "-m",
            "tech_tree_arena.cli",
            "replay",
            str(derived_empty_tape),
            "--actor",
        ],
        check=False,
        text=True,
        capture_output=True,
        env=environment,
    )
    assert rejected_derived_tape.returncode == 1
    assert "disagrees with its journal" in rejected_derived_tape.stderr

    marker.write_text("ok\n", encoding="utf-8")
    resumed = subprocess.run(
        [sys.executable, "-m", "tech_tree_arena.cli", "resume", str(source)],
        check=True,
        text=True,
        capture_output=True,
        env=environment,
    )
    result = json.loads(resumed.stdout)
    assert result["status"] == "pass"
    derived = Path(result["run_dir"])
    derived_services = [
        _event_payload(record)
        for record in verify_hash_chain(derived / "service-calls.private.jsonl")
    ]
    assert derived_services == source_services
    replayed = subprocess.run(
        [sys.executable, "-m", "tech_tree_arena.cli", "replay", str(derived), "--actor"],
        check=True,
        text=True,
        capture_output=True,
        env=environment,
    )
    assert json.loads(replayed.stdout)["status"] == "actor-replayed"


def test_resume_can_retry_an_uncaught_service_failure_without_replaying_it(
    tmp_path: Path,
) -> None:
    submission = tmp_path / "retry-interrupted-pair"
    participant = submission / "participant"
    participant.mkdir(parents=True)
    marker = tmp_path / "provider-recovered"
    (submission / "submission.toml").write_text(
        """schema_version = 1
name = "retry-interrupted-pair"
version = "0.1.0"
protocol = "idea-recovery-v1"
generator = "participant.generator:Generator"
oracle = "participant.oracle:Oracle"
""",
        encoding="utf-8",
    )
    (participant / "__init__.py").write_text("", encoding="utf-8")
    (participant / "generator.py").write_text(
        """import os
from pathlib import Path
from tech_tree_arena import Idea, Question, Submission, SubmitOption

class Generator:
    def __init__(self, services): self.services = services
    def step(self, choice):
        if choice is None:
            return Question("retry", (SubmitOption("submit", None, "1"),))
        values = ["recovered"] if Path(os.environ["IDEA_ARENA_TEST_RETRY_MARKER"]).is_file() else []
        value = self.services.choice(values)
        return Submission((Idea("answer", {"answer": "blue", "value": value}, "1"),))
""",
        encoding="utf-8",
    )
    (participant / "oracle.py").write_text(
        """from tech_tree_arena import Choice
class Oracle:
    def __init__(self, target, services): self.services = services
    def step(self, presented): return Choice("submit")
""",
        encoding="utf-8",
    )

    runs = tmp_path / "runs"
    environment = {**os.environ, "IDEA_ARENA_TEST_RETRY_MARKER": str(marker)}
    failed = subprocess.run(
        [
            sys.executable,
            "-m",
            "tech_tree_arena.cli",
            "run",
            str(submission),
            "--target-pack",
            "smoke",
            "--runs-dir",
            str(runs),
        ],
        check=False,
        text=True,
        capture_output=True,
        env=environment,
    )
    assert failed.returncode == 1
    source = next(path for path in runs.iterdir() if path.name != ".artifacts")
    source_records = verify_hash_chain(source / "service-calls.private.jsonl")
    assert len(source_records) == 1
    assert source_records[0]["service"]["error"] == "IndexError"

    # Strict resume remains the exact deterministic replay mode.
    strict = subprocess.run(
        [sys.executable, "-m", "tech_tree_arena.cli", "resume", str(source)],
        check=False,
        text=True,
        capture_output=True,
        env=environment,
    )
    assert strict.returncode == 1

    # A fresh retry can itself fail. Its derived checkpoint must already bind
    # the abandoned source attempt plus the newly failed live attempt.
    existing_runs = set(runs.iterdir())
    failed_retry = subprocess.run(
        [
            sys.executable,
            "-m",
            "tech_tree_arena.cli",
            "resume",
            str(source),
            "--retry-interrupted-call",
        ],
        check=False,
        text=True,
        capture_output=True,
        env=environment,
    )
    assert failed_retry.returncode == 1
    failed_retry_run = next(
        path
        for path in set(runs.iterdir()) - existing_runs
        if path.is_dir() and path.name != ".artifacts"
    )
    replayed_failed_retry = subprocess.run(
        [
            sys.executable,
            "-m",
            "tech_tree_arena.cli",
            "replay",
            str(failed_retry_run),
            "--actor",
        ],
        check=True,
        text=True,
        capture_output=True,
        env=environment,
    )
    assert json.loads(replayed_failed_retry.stdout)["unreplayed_service_events"] == {
        "generator": 1
    }

    marker.write_text("ok\n", encoding="utf-8")
    retried = subprocess.run(
        [
            sys.executable,
            "-m",
            "tech_tree_arena.cli",
            "resume",
            str(source),
            "--retry-interrupted-call",
        ],
        check=True,
        text=True,
        capture_output=True,
        env=environment,
    )
    result = json.loads(retried.stdout)
    assert result["status"] == "pass"
    derived = Path(result["run_dir"])
    service_records = [
        _event_payload(record)
        for record in verify_hash_chain(derived / "service-calls.private.jsonl")
    ]
    assert [record["kind"] for record in service_records] == [
        "service_abandoned",
        "service_call",
    ]
    assert service_records[0]["abandoned_from_run"] == source.name
    usage = json.loads((derived / "usage.json").read_text(encoding="utf-8"))
    assert usage["generator"]["random_calls"] == 2
    assert usage["generator"]["service_events"] == 1
    replayed = subprocess.run(
        [sys.executable, "-m", "tech_tree_arena.cli", "replay", str(derived), "--actor"],
        check=True,
        text=True,
        capture_output=True,
        env=environment,
    )
    assert json.loads(replayed.stdout)["status"] == "actor-replayed"

    tampered = runs / "retry-missing-abandoned-accounting"
    shutil.copytree(derived, tampered)
    _rewrite_hash_chain(
        tampered / "service-calls.private.jsonl",
        service_records[1:],
    )
    rejected = subprocess.run(
        [sys.executable, "-m", "tech_tree_arena.cli", "replay", str(tampered), "--actor"],
        check=False,
        text=True,
        capture_output=True,
        env=environment,
    )
    assert rejected.returncode == 1
    assert "accounting cursor" in rejected.stderr


def test_actor_replay_uses_content_addressed_submission_snapshot(tmp_path: Path) -> None:
    submission = tmp_path / "submission"
    shutil.copytree(ROOT / "submissions" / "examples" / "minimal_pair", submission)
    runs = tmp_path / "runs"
    completed = subprocess.run(
        [
            sys.executable, "-m", "tech_tree_arena.cli", "run", str(submission),
            "--target-pack", "smoke", "--runs-dir", str(runs),
        ],
        check=True,
        text=True,
        capture_output=True,
    )
    run_directory = json.loads(completed.stdout)["run_dir"]
    shutil.rmtree(submission)
    replayed = subprocess.run(
        [sys.executable, "-m", "tech_tree_arena.cli", "replay", run_directory, "--actor"],
        check=True,
        text=True,
        capture_output=True,
    )
    assert json.loads(replayed.stdout)["status"] == "actor-replayed"


def test_actor_replay_binds_contract_invalid_generator_return_to_pending_call(
    tmp_path: Path,
) -> None:
    submission = tmp_path / "invalid-generator-return-pair"
    participant = submission / "participant"
    participant.mkdir(parents=True)
    (submission / "submission.toml").write_text(
        """schema_version = 1
name = "invalid-generator-return-pair"
version = "0.1.0"
protocol = "idea-recovery-v1"
generator = "participant.generator:Generator"
oracle = "participant.oracle:Oracle"
""",
        encoding="utf-8",
    )
    (participant / "__init__.py").write_text("", encoding="utf-8")
    (participant / "generator.py").write_text(
        """from tech_tree_arena import Choice
class Generator:
    def __init__(self, services): self.services = services
    def step(self, message): return Choice("not-a-generator-output")
""",
        encoding="utf-8",
    )
    (participant / "oracle.py").write_text(
        """from tech_tree_arena import Choice
class Oracle:
    def __init__(self, target, services): self.services = services
    def step(self, message): return Choice("unused")
""",
        encoding="utf-8",
    )

    runs = tmp_path / "runs"
    failed = subprocess.run(
        [
            sys.executable,
            "-m",
            "tech_tree_arena.cli",
            "run",
            str(submission),
            "--target-pack",
            "smoke",
            "--runs-dir",
            str(runs),
        ],
        check=False,
        text=True,
        capture_output=True,
    )
    assert failed.returncode == 1
    source = next(
        path for path in runs.iterdir()
        if path.is_dir() and path.name != ".artifacts"
    )
    private = [
        _event_payload(record)
        for record in verify_hash_chain(source / "events.private.jsonl")
    ]
    assert [record["kind"] for record in private] == ["run_started", "run_finished"]
    assert private[-1]["error_code"] == "protocol_error"
    assert private[-1]["interrupted_call"] == {
        "role": "generator",
        "branch_id": "root",
        "operation": "step",
    }
    checkpoint = json.loads(
        (source / "checkpoint.private.json").read_text(encoding="utf-8")
    )
    assert checkpoint["engine"]["phase"] == "generator_output"
    assert checkpoint["engine"]["output"]["type"] == "choice"

    replayed = subprocess.run(
        [sys.executable, "-m", "tech_tree_arena.cli", "replay", str(source), "--actor"],
        check=True,
        text=True,
        capture_output=True,
    )
    replay_result = json.loads(replayed.stdout)
    assert replay_result["status"] == "actor-replayed"
    assert replay_result["calls"] == {"generator": 1, "oracle": 0}

    # Resuming this checkpoint deterministically reaches the same protocol
    # error without making a new participant call.  The derived artifact must
    # carry the pending marker and remain replayable even though it does not
    # inherit the source checkpoint file.
    resumed_failure = subprocess.run(
        [sys.executable, "-m", "tech_tree_arena.cli", "resume", str(source)],
        check=False,
        text=True,
        capture_output=True,
    )
    assert resumed_failure.returncode == 1
    derived = next(
        path
        for path in runs.iterdir()
        if path.is_dir()
        and path.name not in {".artifacts", source.name}
        and json.loads((path / "manifest.json").read_text(encoding="utf-8")).get(
            "resumed_from"
        )
        == source.name
    )
    derived_private = [
        _event_payload(record)
        for record in verify_hash_chain(derived / "events.private.jsonl")
    ]
    assert derived_private[-1]["interrupted_call"] == private[-1]["interrupted_call"]
    replayed_derived = subprocess.run(
        [sys.executable, "-m", "tech_tree_arena.cli", "replay", str(derived), "--actor"],
        check=True,
        text=True,
        capture_output=True,
    )
    assert json.loads(replayed_derived.stdout)["status"] == "actor-replayed"

    # The checkpoint is the only trusted binding for the returned value that
    # never became a question/submission event.  It cannot be changed while
    # retaining the original ActorCall tape.
    tampered = runs / "tampered-invalid-generator-return"
    shutil.copytree(source, tampered)
    tampered_checkpoint_path = tampered / "checkpoint.private.json"
    tampered_checkpoint = json.loads(
        tampered_checkpoint_path.read_text(encoding="utf-8")
    )
    tampered_checkpoint["engine"]["output"]["value"]["option_id"] = "different"
    tampered_checkpoint_path.write_text(
        json.dumps(tampered_checkpoint),
        encoding="utf-8",
    )
    rejected_tamper = subprocess.run(
        [sys.executable, "-m", "tech_tree_arena.cli", "replay", str(tampered), "--actor"],
        check=False,
        text=True,
        capture_output=True,
    )
    assert rejected_tamper.returncode == 1
    assert "not bound" in rejected_tamper.stderr

    # The pending marker is already consumed by the phase-local checkpoint's
    # bound return.  Duplicating that deterministic ActorCall must not let the
    # same marker authorize a second call on the same input.
    duplicated_call = runs / "duplicated-invalid-generator-call"
    shutil.copytree(source, duplicated_call)
    actor_calls_path = duplicated_call / "actor-calls.generator.private.json"
    actor_calls = json.loads(actor_calls_path.read_text(encoding="utf-8"))
    actor_calls.append(dict(actor_calls[-1]))
    actor_calls_path.write_text(json.dumps(actor_calls), encoding="utf-8")
    rejected_duplicate = subprocess.run(
        [
            sys.executable,
            "-m",
            "tech_tree_arena.cli",
            "replay",
            str(duplicated_call),
            "--actor",
        ],
        check=False,
        text=True,
        capture_output=True,
    )
    assert rejected_duplicate.returncode == 1
    assert "call count disagrees" in rejected_duplicate.stderr

    # A failed run cannot use a phase-local checkpoint to invent an otherwise
    # unexplained completed call: the exact pending-call marker is required.
    unmarked = runs / "unmarked-invalid-generator-return"
    shutil.copytree(source, unmarked)
    unmarked_private = [
        _event_payload(record)
        for record in verify_hash_chain(unmarked / "events.private.jsonl")
    ]
    unmarked_private[-1].pop("interrupted_call")
    _rewrite_hash_chain(unmarked / "events.private.jsonl", unmarked_private)
    _rewrite_hash_chain(
        unmarked / "events.public.jsonl",
        _redacted_public_events(unmarked_private, "development"),
    )
    rejected_unmarked = subprocess.run(
        [sys.executable, "-m", "tech_tree_arena.cli", "replay", str(unmarked), "--actor"],
        check=False,
        text=True,
        capture_output=True,
    )
    assert rejected_unmarked.returncode == 1
    assert "no exact pending-call marker" in rejected_unmarked.stderr


def test_sigkill_interrupted_run_resumes_without_repeating_committed_prefix(tmp_path: Path) -> None:
    submission = tmp_path / "killable-pair"
    participant = submission / "participant"
    participant.mkdir(parents=True)
    marker = tmp_path / "skip-sleep"
    (submission / "submission.toml").write_text(
        """schema_version = 1
name = "killable-pair"
version = "0.1.0"
protocol = "idea-recovery-v1"
generator = "participant.generator:Generator"
oracle = "participant.oracle:Oracle"
""",
        encoding="utf-8",
    )
    (participant / "__init__.py").write_text("", encoding="utf-8")
    (participant / "generator.py").write_text(
        """import os, time
from pathlib import Path
from tech_tree_arena import Idea, Question, Submission, SubmitOption
class Generator:
    def __init__(self, services): self.services = services
    def step(self, choice):
        if choice is None:
            return Question("recover", (SubmitOption("blue", "blue", "1"),))
        if not Path(os.environ["IDEA_ARENA_TEST_RESUME_MARKER"]).is_file():
            time.sleep(60)
        return Submission((Idea("answer", {"answer": choice.public_payload}, "1"),))
""",
        encoding="utf-8",
    )
    (participant / "oracle.py").write_text(
        """from tech_tree_arena import Choice
class Oracle:
    def __init__(self, target, services): self.services = services
    def step(self, presented): return Choice("blue")
""",
        encoding="utf-8",
    )
    runs = tmp_path / "runs"
    environment = {**os.environ, "IDEA_ARENA_TEST_RESUME_MARKER": str(marker)}
    process = subprocess.Popen(
        [
            sys.executable, "-m", "tech_tree_arena.cli", "run", str(submission),
            "--target-pack", "smoke", "--runs-dir", str(runs),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=environment,
        start_new_session=True,
    )
    source = None
    deadline = time.monotonic() + 10
    try:
        while time.monotonic() < deadline:
            candidates = [
                path for path in runs.iterdir()
                if path.is_dir() and path.name != ".artifacts"
            ] if runs.is_dir() else []
            if candidates:
                candidate = candidates[0]
                status_path = candidate / "status.json"
                if status_path.is_file():
                    status = json.loads(status_path.read_text(encoding="utf-8"))
                    if status.get("phase") == "after_oracle":
                        source = candidate
                        break
            time.sleep(0.05)
        assert source is not None
    finally:
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=5)

    interrupted = json.loads((source / "status.json").read_text(encoding="utf-8"))
    assert interrupted["status"] == "running"
    assert interrupted["phase"] == "after_oracle"
    assert interrupted["resumable"] is True
    inspected = subprocess.run(
        [sys.executable, "-m", "tech_tree_arena.cli", "status", str(source)],
        check=True,
        text=True,
        capture_output=True,
        env=environment,
    )
    inspected_status = json.loads(inspected.stdout)
    assert inspected_status["process_alive"] is False
    assert inspected_status["effective_status"] == "interrupted"
    marker.write_text("ok\n", encoding="utf-8")
    resumed = subprocess.run(
        [sys.executable, "-m", "tech_tree_arena.cli", "resume", str(source)],
        check=True,
        text=True,
        capture_output=True,
        env=environment,
    )
    result = json.loads(resumed.stdout)
    assert result["status"] == "pass"
    assert result["questions"] == 1
    assert result["oracle_decisions"] == 1


def test_sigint_pauses_parent_without_killing_participant_worker(tmp_path: Path) -> None:
    submission = tmp_path / "sigint-pair"
    participant = submission / "participant"
    participant.mkdir(parents=True)
    (submission / "submission.toml").write_text(
        """schema_version = 1
name = "sigint-pair"
version = "0.1.0"
protocol = "idea-recovery-v1"
generator = "participant.generator:Generator"
oracle = "participant.oracle:Oracle"
""",
        encoding="utf-8",
    )
    (participant / "__init__.py").write_text("", encoding="utf-8")
    (participant / "generator.py").write_text(
        """import time
from tech_tree_arena import Idea, Question, Submission, SubmitOption
class Generator:
    def __init__(self, services): self.services = services
    def step(self, choice):
        if choice is None:
            return Question("pause", (SubmitOption("submit", None, "1"),))
        time.sleep(2)
        return Submission((Idea("answer", {"answer": "blue"}, "1"),))
""",
        encoding="utf-8",
    )
    (participant / "oracle.py").write_text(
        """from tech_tree_arena import Choice
class Oracle:
    def __init__(self, target, services): self.services = services
    def step(self, presented): return Choice("submit")
""",
        encoding="utf-8",
    )
    runs = tmp_path / "runs"
    process = subprocess.Popen(
        [
            sys.executable, "-m", "tech_tree_arena.cli", "run", str(submission),
            "--target-pack", "smoke", "--runs-dir", str(runs),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    source = None
    deadline = time.monotonic() + 10
    try:
        while time.monotonic() < deadline:
            candidates = [
                path for path in runs.iterdir()
                if path.is_dir() and path.name != ".artifacts"
            ] if runs.is_dir() else []
            if candidates:
                candidate = candidates[0]
                status_path = candidate / "status.json"
                if status_path.is_file():
                    status = json.loads(status_path.read_text(encoding="utf-8"))
                    if status.get("phase") == "after_oracle":
                        source = candidate
                        break
            time.sleep(0.05)
        assert source is not None
        os.killpg(process.pid, signal.SIGINT)
        process.wait(timeout=10)
    finally:
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=5)

    assert process.returncode == 1
    status = json.loads((source / "status.json").read_text(encoding="utf-8"))
    assert status["error_code"] == "interrupted"
    private = [
        _event_payload(record)
        for record in verify_hash_chain(source / "events.private.jsonl")
    ]
    assert private[-1]["error_code"] == "interrupted"
    assert private[-1]["error_type"] == "KeyboardInterrupt"
    assert private[-1]["interrupted_call"] == {
        "role": "generator",
        "branch_id": "root",
        "operation": "step",
    }
    replayed = subprocess.run(
        [sys.executable, "-m", "tech_tree_arena.cli", "replay", str(source), "--actor"],
        check=True,
        text=True,
        capture_output=True,
    )
    assert json.loads(replayed.stdout)["status"] == "actor-replayed"


def test_failure_drains_inflight_service_before_final_accounting(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    submission = tmp_path / "inflight-failure-pair"
    participant = submission / "participant"
    participant.mkdir(parents=True)
    marker = tmp_path / "provider-call-started"
    monkeypatch.setenv("IDEA_ARENA_TEST_PROVIDER_MARKER", str(marker))
    (submission / "submission.toml").write_text(
        """schema_version = 1
name = "inflight-failure-pair"
version = "0.1.0"
protocol = "idea-recovery-v1"
generator = "participant.generator:Generator"
oracle = "participant.oracle:Oracle"
""",
        encoding="utf-8",
    )
    (participant / "__init__.py").write_text("", encoding="utf-8")
    (participant / "generator.py").write_text(
        """import os, threading, time
from pathlib import Path
class Generator:
    def __init__(self, services): self.services = services
    def step(self, choice):
        def call_model():
            self.services.structured_model(
                developer="test", user="test", schema_name="inflight",
                schema={"type": "object"}, max_output_tokens=1,
            )
        threading.Thread(target=call_model, daemon=True).start()
        marker = Path(os.environ["IDEA_ARENA_TEST_PROVIDER_MARKER"])
        while not marker.is_file():
            time.sleep(0.01)
        os._exit(9)
""",
        encoding="utf-8",
    )
    (participant / "oracle.py").write_text(
        """class Oracle:
    def __init__(self, target, services): pass
    def step(self, presented): raise AssertionError("oracle must not run")
""",
        encoding="utf-8",
    )

    class SlowBackend:
        def structured(self, **request):
            marker.write_text("started\n", encoding="utf-8")
            # Keep the provider call in flight until after the worker has exited.
            time.sleep(0.25)
            return {"ok": True}

    runs = tmp_path / "runs"
    with pytest.raises(RecordedRunFailure) as failure:
        _run_submission(
            submission,
            None,
            None,
            1,
            runs_dir=runs,
            sample_mode=True,
            sample_max_questions=1,
            model_backends={"generator": SlowBackend()},
        )
    assert arena_cli._runner_active is False

    run = Path(failure.value.run_dir)
    actor_tape = json.loads(
        (run / "service-tape.generator.private.json").read_text(encoding="utf-8")
    )
    journal = [
        _event_payload(record)
        for record in verify_hash_chain(run / "service-calls.private.jsonl")
    ]
    generator_journal = [
        record["service"]
        for record in journal
        if record.get("role") == "generator"
    ]
    usage = json.loads((run / "usage.json").read_text(encoding="utf-8"))

    assert len(actor_tape) == 1
    assert generator_journal == actor_tape
    assert usage["generator"]["model_calls"] == 1
    assert usage["generator"]["service_events"] == 1
    replayed = actor_replay(run)
    assert replayed["status"] == "actor-replayed"


def test_resume_preserves_checkout_capabilities_and_branch_history(tmp_path: Path) -> None:
    submission = tmp_path / "branch-resume-pair"
    participant = submission / "participant"
    participant.mkdir(parents=True)
    marker = tmp_path / "allow-branch-finish"
    (submission / "submission.toml").write_text(
        """schema_version = 1
name = "branch-resume-pair"
version = "0.1.0"
protocol = "idea-recovery-v1"
generator = "participant.generator:Generator"
oracle = "participant.oracle:Oracle"
""",
        encoding="utf-8",
    )
    (participant / "__init__.py").write_text("", encoding="utf-8")
    (participant / "generator.py").write_text(
        """import os
from pathlib import Path
from tech_tree_arena import Idea, Option, Question, Submission, SubmitOption
class Generator:
    def __init__(self, services): self.phase = "start"
    def step(self, choice):
        if choice is None:
            self.phase = "first"
            return Question("first", (Option("red", "red", "0.5"), SubmitOption("blue", "blue", "0.5")))
        if self.phase == "first" and choice.option_id == "red":
            self.phase = "second"
            return Question("rewind", (Option("continue", "continue", "1"),))
        if not Path(os.environ["IDEA_ARENA_TEST_RESUME_MARKER"]).is_file():
            raise RuntimeError("intentional failure after checkout")
        return Submission((Idea("answer", {"answer": choice.public_payload}, "1"),))
""",
        encoding="utf-8",
    )
    (participant / "oracle.py").write_text(
        """from tech_tree_arena import Checkout, Choice
class Oracle:
    def __init__(self, target, services): self.first = None
    def step(self, presented):
        if presented.question.question == "first":
            if self.first is None:
                self.first = presented.question_id
                return Choice("red")
            return Choice("blue")
        return Checkout(self.first)
""",
        encoding="utf-8",
    )
    runs = tmp_path / "runs"
    environment = {**os.environ, "IDEA_ARENA_TEST_RESUME_MARKER": str(marker)}
    failed = subprocess.run(
        [
            sys.executable, "-m", "tech_tree_arena.cli", "run", str(submission),
            "--target-pack", "smoke", "--runs-dir", str(runs),
        ],
        check=False,
        text=True,
        capture_output=True,
        env=environment,
    )
    assert failed.returncode == 1
    source = next(path for path in runs.iterdir() if path.is_dir() and path.name != ".artifacts")
    marker.write_text("ok\n", encoding="utf-8")
    resumed = subprocess.run(
        [sys.executable, "-m", "tech_tree_arena.cli", "resume", str(source)],
        check=True,
        text=True,
        capture_output=True,
        env=environment,
    )
    result = json.loads(resumed.stdout)
    assert result["status"] == "pass"
    assert result["questions"] == 2
    assert result["oracle_decisions"] == 3
    assert result["checkouts"] == 1
    replayed = subprocess.run(
        [sys.executable, "-m", "tech_tree_arena.cli", "replay", result["run_dir"], "--actor"],
        check=True,
        text=True,
        capture_output=True,
        env=environment,
    )
    assert json.loads(replayed.stdout)["branches"]["generator"] == 2


def test_failed_nested_judge_tail_requires_matching_proxy():
    from tech_tree_arena.replay.recorder import _bound_failed_judge_tail
    records = [_service_record("judge", "model.structured", "failed"),
               _service_record("oracle", "judge.evaluate", "proxy")]
    for record in records:
        record["service"]["error"] = "BudgetBlocked"
    assert _bound_failed_judge_tail(records)
    assert not _bound_failed_judge_tail(records[:1])
    records[-1]["service"]["error"] = "RateLimitError"
    assert not _bound_failed_judge_tail(records)
    records[-1]["service"]["error"] = None
    assert not _bound_failed_judge_tail(records)


def test_retry_judge_rejection_followed_by_budget_stop(tmp_path):
    from tech_tree_arena.replay.recorder import _retry_interrupted_call_marker
    marker = dict(role='judge', branch_id='judge', operation='evaluate')
    HashChainWriter(tmp_path / 'events.private.jsonl').append(dict(kind='run_finished',
        status='error', error_code='run_budget_blocked', error_type='BudgetBlocked', interrupted_call=marker))
    records = [_service_record('judge','model.structured',str(i)) for i in range(3)]
    records[1]['service']['error']='RateLimitError'
    records[2]['service']['error']='BudgetBlocked'
    assert _retry_interrupted_call_marker(tmp_path, {'service_event_count':0}, tuple(records)) == marker
    records[2]['service']['error']=None
    with pytest.raises(ReplayDivergence):
        _retry_interrupted_call_marker(tmp_path, {'service_event_count':0}, tuple(records))


def test_failed_nested_judge_tail_after_an_earlier_retry():
    from tech_tree_arena.replay.recorder import _bound_failed_judge_tail
    previous = [_service_record('judge', 'model.structured', 'old'),
                _service_record('oracle', 'judge.evaluate', 'old-proxy')]
    for record in previous:
        record.update(kind='service_abandoned', abandoned_from_run='parent')
        record['service']['error'] = 'RateLimitError'
    current = [_service_record('judge', 'model.structured', 'new'),
               _service_record('oracle', 'judge.evaluate', 'new-proxy')]
    for record in current:
        record['service']['error'] = 'BudgetBlocked'
    assert _bound_failed_judge_tail(previous + current)
    assert not _bound_failed_judge_tail(previous)
    assert not _bound_failed_judge_tail(previous + current[:1])
    current[-1]['service']['error'] = 'RateLimitError'
    assert not _bound_failed_judge_tail(previous + current)
