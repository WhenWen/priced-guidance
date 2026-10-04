import math

import pytest
from tech_tree_arena.contract.recovery import occurrence_bits

from tech_tree_arena import (
    Checkout,
    Choice,
    Idea,
    IdeaVerdict,
    Option,
    PresentedQuestion,
    Question,
    Submission,
    SubmissionFeedback,
    SubmitOption,
)
from tech_tree_arena.errors import ProtocolError, ReplayDivergence, ResourceLimitExceeded
from tech_tree_arena.evaluation.smoke import SmokeAnswerJudge
from tech_tree_arena.runtime.actor import ActorFactory, ActorRuntime
from tech_tree_arena.runtime.engine import ArenaRunner, RunLimits
from tech_tree_arena.runtime.services import ServiceFactory


class BranchingGenerator:
    def __init__(self, services):
        self.services = services
        self.phase = "start"

    def step(self, choice):
        if choice is None:
            self.phase = "first"
            return Question(
                "first",
                (Option("a", "a", "0.5"), SubmitOption("b", "b", "0.5")),
            )
        if self.phase == "first" and choice.option_id == "a":
            self.phase = "second"
            return Question("second", (Option("x", "x", "0.5"), Option("y", "y", "0.5")))
        return Submission((Idea("answer", {"answer": choice.public_payload}, "1"),))


class TimeTravelOracle:
    def __init__(self, target, services):
        self.services = services
        self.first_question_id = None
        self.revisited = False

    def step(self, question: PresentedQuestion):
        if question.question.question == "first":
            if self.first_question_id is None:
                self.first_question_id = question.question_id
                return Choice("a")
            self.revisited = True
            return Choice("b")
        return Checkout(self.first_question_id)


def test_time_travel_restores_generator_but_not_oracle() -> None:
    runner = ArenaRunner(allow_time_travel=True)
    result = runner.run(
        generator_factory=ActorFactory(
            BranchingGenerator, service_factory=ServiceFactory(seed=1)
        ),
        oracle_factory=ActorFactory(
            TimeTravelOracle,
            constructor_args=({"answer": "b"},),
            service_factory=ServiceFactory(seed=2),
        ),
        target={"answer": "b"},
        judge=SmokeAnswerJudge(),
        seed=3,
        run_id="time-travel-test",
    )

    assert result.status == "pass"
    assert result.checkout_count == 1
    assert result.question_count == 2
    # A is abandoned. B pays its bid and its own first-occurrence surcharge.
    assert result.k == pytest.approx(1.0 + occurrence_bits(1))
    assert result.score == pytest.approx(0.5 * 0.95)
    assert len(result.branch_store.checkout_audit) == 1


def test_time_travel_can_be_disabled_without_changing_participant_api() -> None:
    runner = ArenaRunner(allow_time_travel=False)
    with pytest.raises(Exception, match="disabled"):
        runner.run(
            generator_factory=ActorFactory(
                BranchingGenerator, service_factory=ServiceFactory(seed=1)
            ),
            oracle_factory=ActorFactory(
                TimeTravelOracle,
                constructor_args=({"answer": "b"},),
                service_factory=ServiceFactory(seed=2),
            ),
            target={"answer": "b"},
            judge=SmokeAnswerJudge(),
            seed=3,
            run_id="no-time-travel-test",
        )


@pytest.mark.parametrize("max_bits", [0.5, 1.05])
def test_information_limit_rejects_choice_before_charging_or_stepping_generator(max_bits) -> None:
    events = []
    runner = ArenaRunner(
        limits=RunLimits(max_bits=max_bits),
        event_sink=events.append,
    )

    with pytest.raises(ResourceLimitExceeded, match="information-cost"):
        runner.run(
            generator_factory=ActorFactory(
                BranchingGenerator, service_factory=ServiceFactory(seed=1)
            ),
            oracle_factory=ActorFactory(
                TimeTravelOracle,
                constructor_args=({"answer": "b"},),
                service_factory=ServiceFactory(seed=2),
            ),
            target={"answer": "b"},
            judge=SmokeAnswerJudge(),
            seed=3,
            run_id="precharge-information-limit-test",
        )

    assert [event["kind"] for event in events][-1] == "oracle_decision"
    assert all(event["kind"] != "choice_cost" for event in events)
    assert len(runner.last_generator.calls) == 1
    assert runner.last_branches.continuation_counts == {}
    assert runner.last_branches.option_counts == {}


class FailingForkRuntime(ActorRuntime):
    def fork(self, checkpoint, branch_id):
        raise ReplayDivergence("intentional reconstruction failure")


def test_failed_generator_fork_does_not_commit_a_ghost_checkout() -> None:
    events = []
    runner = ArenaRunner(runtime=FailingForkRuntime(), event_sink=events.append)
    with pytest.raises(ReplayDivergence, match="reconstruction failure"):
        runner.run(
            generator_factory=ActorFactory(
                BranchingGenerator,
                service_factory=ServiceFactory(seed=1),
            ),
            oracle_factory=ActorFactory(
                TimeTravelOracle,
                constructor_args=({"answer": "b"},),
                service_factory=ServiceFactory(seed=2),
            ),
            target={"answer": "b"},
            judge=SmokeAnswerJudge(),
            run_id="atomic-checkout-test",
        )

    assert runner.last_branches.checkout_audit == []
    assert runner.last_branches.checkout_pair_counts == {}
    assert runner.last_branches.head == runner.last_branches.order[-1]
    assert all(event["kind"] != "checkout" for event in events)


class ImmediateSubmissionGenerator:
    def __init__(self, services):
        self.services = services

    def step(self, choice):
        if choice is None:
            return Question("submit?", (SubmitOption("submit", None, "1"),))
        return Submission(
            (
                Idea("match", {"answer": "blue"}, "0.25"),
                Idea("miss", {"answer": "red"}, "0.75"),
            )
        )


class SubmitOracle:
    def __init__(self, _target, services):
        self.services = services

    def step(self, _question):
        return Choice("submit")


def test_terminal_submission_cost_obeys_information_limit() -> None:
    runner = ArenaRunner(limits=RunLimits(max_bits=1.0))
    with pytest.raises(ResourceLimitExceeded, match="information-cost"):
        runner.run(
            generator_factory=ActorFactory(
                ImmediateSubmissionGenerator, service_factory=ServiceFactory(seed=1)
            ),
            oracle_factory=ActorFactory(
                SubmitOracle,
                constructor_args=({"answer": "blue"},),
                service_factory=ServiceFactory(seed=2),
            ),
            target={"answer": "blue"},
            judge=SmokeAnswerJudge(),
            run_id="terminal-budget-test",
        )


class SequencedRepeatJudge:
    def __init__(self):
        self.round = 0

    def evaluate(self, _target, ideas):
        self.round += 1
        return tuple(
            IdeaVerdict(
                idea.idea_id,
                self.round == 2 and idea.idea_id == "match",
                f"round {self.round}",
            )
            for idea in ideas
        )


def test_runner_repeats_judge_and_charges_negative_log_pass_fraction() -> None:
    events = []
    judge = SequencedRepeatJudge()

    result = ArenaRunner(judge_repeats=3, event_sink=events.append).run(
        generator_factory=ActorFactory(
            ImmediateSubmissionGenerator,
            service_factory=ServiceFactory(seed=1),
        ),
        oracle_factory=ActorFactory(
            SubmitOracle,
            constructor_args=({"answer": "blue"},),
            service_factory=ServiceFactory(seed=2),
        ),
        target={"answer": "blue"},
        judge=judge,
        run_id="repeat-judge-test",
    )

    assert judge.round == 3
    assert result.status == "pass"
    assert result.judge_repeats == 3
    assert result.judge_passes == 1
    assert result.repeat_bits == pytest.approx(math.log2(3))
    # The matched idea has mass 0.25 (two pointer bits), and one of three
    # independent judgments passed (log2(3) repeat bits).
    assert result.k == pytest.approx(2.0 + math.log2(3) + occurrence_bits(1))
    judged = next(event for event in events if event["kind"] == "submission_judged")
    assert float(judged["judge_pass_rate"]) == pytest.approx(1 / 3)
    assert len(judged["verdict_rounds"]) == 3


class TolerantSubmissionGenerator:
    def __init__(self, services):
        self.services = services

    def step(self, choice):
        if choice is None:
            return Question("submit?", (SubmitOption("submit", None, "1"),))
        return Submission(
            (
                Idea("first", {"answer": "blue"}, "0.5000000004"),
                Idea("second", {"answer": "blue"}, "0.5000000004"),
            )
        )


def test_tolerated_probability_roundoff_cannot_create_mass_above_one() -> None:
    result = ArenaRunner().run(
        generator_factory=ActorFactory(
            TolerantSubmissionGenerator, service_factory=ServiceFactory(seed=1)
        ),
        oracle_factory=ActorFactory(
            SubmitOracle,
            constructor_args=({"answer": "blue"},),
            service_factory=ServiceFactory(seed=2),
        ),
        target={"answer": "blue"},
        judge=SmokeAnswerJudge(),
        run_id="tolerant-mass-test",
    )
    assert result.k == pytest.approx(occurrence_bits(1))
    assert result.score == pytest.approx(0.95)


class RetryAfterRejectedSubmissionGenerator:
    def __init__(self, services):
        self.services = services

    def step(self, choice):
        if choice is None:
            return Question(
                "first",
                (
                    SubmitOption("submit-wrong", "wrong", "0.25"),
                    Option("revise", "revise", "0.75"),
                ),
            )
        if choice.option_id == "submit-wrong":
            return Submission((Idea("wrong", {"answer": "red"}, "1"),))
        if choice.option_id == "revise":
            return Question(
                "second",
                (SubmitOption("submit-right", "right", "1"),),
            )
        if choice.option_id == "submit-right":
            return Submission((Idea("right", {"answer": "blue"}, "1"),))
        raise AssertionError(f"unexpected choice: {choice.option_id}")


class RetryAfterRejectedSubmissionOracle:
    def __init__(self, _target, services):
        self.services = services
        self.root_question_id = None
        self.recovered = False
        self.saw_private_feedback = False

    def step(self, message):
        if isinstance(message, SubmissionFeedback):
            self.saw_private_feedback = True
            assert tuple(verdict.passed for verdict in message.verdicts) == (False,)
            assert message.valid_checkout_question_ids == (message.source.question_id,)
            self.recovered = True
            return Checkout(message.source.question_id)
        if message.question.question == "first":
            if self.root_question_id is None:
                self.root_question_id = message.question_id
                return Choice("submit-wrong")
            assert self.recovered
            assert message.question_id == self.root_question_id
            return Choice("revise")
        return Choice("submit-right")


def test_rejected_submission_is_private_recoverable_and_rewindable() -> None:
    events = []
    runner = ArenaRunner(event_sink=events.append)
    result = runner.run(
        generator_factory=ActorFactory(
            RetryAfterRejectedSubmissionGenerator,
            service_factory=ServiceFactory(seed=1),
        ),
        oracle_factory=ActorFactory(
            RetryAfterRejectedSubmissionOracle,
            constructor_args=({"answer": "blue"},),
            service_factory=ServiceFactory(seed=2),
        ),
        target={"answer": "blue"},
        judge=SmokeAnswerJudge(),
        run_id="rejected-submission-recovery-test",
    )

    assert result.status == "pass"
    assert result.submission_attempt_count == 2
    assert result.checkout_count == 1
    assert result.oracle_decision_count == 4
    assert result.question_count == 2
    # The rejected 0.25 SubmitOption is abandoned. The surviving path pays
    # revise (0.75), then submit, both with their first-occurrence mass 0.95.
    assert result.k == pytest.approx(-math.log2(0.75) + 2 * occurrence_bits(1))
    assert result.score == pytest.approx(0.75 * 0.95**2)
    checkout = next(event for event in events if event["kind"] == "checkout")
    assert checkout["path_k"] == pytest.approx(0.0)
    assert all("failed_attempt_bits" not in event for event in events)
    assert all("failed_attempt_bits_after" not in event for event in events)
    assert runner.last_oracle.actor.saw_private_feedback
    assert all(
        not isinstance(call.message, SubmissionFeedback)
        for _, handle in runner.generator_history
        for call in handle.calls
    )
    assert [
        event["status"]
        for event in events
        if event["kind"] == "submission_judged"
    ] == ["fail", "pass"]


class NoTimeTravelRetryGenerator:
    def __init__(self, services):
        self.services = services

    def step(self, choice):
        if choice is None:
            return Question("root", (Option("continue", None, "1"),))
        if choice.option_id == "continue":
            return Question(
                "attempt",
                (
                    SubmitOption("wrong", None, "0.25"),
                    Option("revise", None, "0.75"),
                ),
            )
        if choice.option_id == "wrong":
            return Submission((Idea("wrong", {"answer": "red"}, "1"),))
        if choice.option_id == "revise":
            return Question("ready", (SubmitOption("right", None, "1"),))
        return Submission((Idea("right", {"answer": "blue"}, "1"),))


class NoTimeTravelRetryOracle:
    def __init__(self, _target, services):
        self.failed = False
        self.attempt_id = None

    def step(self, message):
        if isinstance(message, SubmissionFeedback):
            assert self.attempt_id is not None
            assert message.source.question_id == self.attempt_id
            assert message.valid_checkout_question_ids == (self.attempt_id,)
            self.failed = True
            return Checkout(self.attempt_id)
        if message.question.question == "root":
            return Choice("continue")
        if message.question.question == "attempt":
            self.attempt_id = message.question_id
            return Choice("revise" if self.failed else "wrong")
        return Choice("right")


def test_rejected_submission_without_time_travel_offers_only_self_recovery() -> None:
    result = ArenaRunner(allow_time_travel=False).run(
        generator_factory=ActorFactory(
            NoTimeTravelRetryGenerator,
            service_factory=ServiceFactory(seed=1),
        ),
        oracle_factory=ActorFactory(
            NoTimeTravelRetryOracle,
            constructor_args=({"answer": "blue"},),
            service_factory=ServiceFactory(seed=2),
        ),
        target={"answer": "blue"},
        judge=SmokeAnswerJudge(),
        run_id="no-time-travel-recovery-test",
    )

    assert result.status == "pass"
    assert result.submission_attempt_count == 2
    assert result.checkout_count == 1
    assert result.branch_store.checkout_audit[-1].target_count == 1


class UnauthorizedSubmissionGenerator:
    def __init__(self, services):
        self.services = services

    def step(self, _choice):
        return Submission((Idea("guess", {"answer": "blue"}, "1"),))


def test_generator_cannot_submit_without_oracle_authorization() -> None:
    with pytest.raises(ProtocolError, match="only after the oracle selects"):
        ArenaRunner().run(
            generator_factory=ActorFactory(
                UnauthorizedSubmissionGenerator,
                service_factory=ServiceFactory(seed=1),
            ),
            oracle_factory=ActorFactory(
                SubmitOracle,
                constructor_args=({"answer": "blue"},),
                service_factory=ServiceFactory(seed=2),
            ),
            target={"answer": "blue"},
            judge=SmokeAnswerJudge(),
            run_id="unauthorized-submission-test",
        )


class UnboundedDepthGenerator:
    def __init__(self, services):
        self.services = services

    def step(self, _choice):
        return Question("continue", (Option("yes", "yes", "1"),))


class AlwaysChooseOracle:
    def __init__(self, _target, services):
        self.services = services

    def step(self, _question):
        return Choice("yes")


def test_question_depth_is_bounded_independently_of_question_count() -> None:
    runner = ArenaRunner(limits=RunLimits(max_questions=10, max_depth=2))
    with pytest.raises(ResourceLimitExceeded, match="question-depth"):
        runner.run(
            generator_factory=ActorFactory(
                UnboundedDepthGenerator, service_factory=ServiceFactory(seed=1)
            ),
            oracle_factory=ActorFactory(
                AlwaysChooseOracle,
                constructor_args=({},),
                service_factory=ServiceFactory(seed=2),
            ),
            target={},
            judge=SmokeAnswerJudge(),
            run_id="depth-limit-test",
        )
