import pytest

from tech_tree_arena.contract import IDEA_RECOVERY_V1
from tech_tree_arena.errors import ProtocolError
from tech_tree_arena import (
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
from tech_tree_arena.contract.validation import message_hash, validate_submission_feedback
from tech_tree_arena.runtime.wire import decode_message, encode_message
from tech_tree_arena.verification import verify_installation


def test_contract_uses_generator_committed_payload_and_prices_the_choice() -> None:
    question = Question(
        "pick",
        (
            Option("a", {"source": "generator"}, "0.25"),
            Option("b", None, "0.75"),
        ),
    )

    resolved = IDEA_RECOVERY_V1.resolve_choice(
        Choice("a", public_payload={"source": "oracle"}),
        question,
    )

    assert resolved.generator_choice.public_payload == {"source": "generator"}
    assert resolved.probability == pytest.approx(0.25)
    assert resolved.information_bits == pytest.approx(2.0)


def test_contract_scores_total_semantically_passing_mass() -> None:
    submission = Submission(
        (
            Idea("first", "same semantics, one wording", "0.2"),
            Idea("second", "same semantics, another wording", "0.3"),
            Idea("miss", "different semantics", "0.5"),
        )
    )
    outcome = IDEA_RECOVERY_V1.score_submission(
        submission,
        (
            IdeaVerdict("first", True),
            IdeaVerdict("second", True),
            IdeaVerdict("miss", False),
        ),
        path_k=1.0,
    )

    assert outcome.matched_idea_ids == ("first", "second")
    assert outcome.passing_mass == pytest.approx(0.5)
    assert outcome.k == pytest.approx(2.0)
    assert outcome.score == pytest.approx(0.25)


def test_contract_rejects_an_incomplete_judge_verdict_set() -> None:
    submission = Submission((Idea("idea", "content", "1"),))
    with pytest.raises(ProtocolError, match="verdict set"):
        IDEA_RECOVERY_V1.score_submission(
            submission,
            (),
            path_k=0.0,
        )


def test_repeated_judge_accepts_any_pass_and_prices_empirical_pass_rate() -> None:
    submission = Submission((Idea("idea", "candidate", "1"),))

    outcome, verdicts = IDEA_RECOVERY_V1.score_repeated_submission(
        submission,
        (
            (IdeaVerdict("idea", False, "miss one"),),
            (IdeaVerdict("idea", True, "pass"),),
            (IdeaVerdict("idea", False, "miss two"),),
        ),
        path_k=2.0,
    )

    assert outcome.status == "pass"
    assert outcome.judge_repeats == 3
    assert outcome.judge_passes == 1
    assert float(outcome.judge_pass_rate) == pytest.approx(1 / 3)
    assert outcome.repeat_bits == pytest.approx(-__import__("math").log2(1 / 3))
    assert outcome.k == pytest.approx(2.0 - __import__("math").log2(1 / 3))
    assert verdicts[0].passed is True
    assert '"judge_passes":1' in verdicts[0].private_reason

    two_pass_outcome, _ = IDEA_RECOVERY_V1.score_repeated_submission(
        submission,
        (
            (IdeaVerdict("idea", True),),
            (IdeaVerdict("idea", False),),
            (IdeaVerdict("idea", True),),
        ),
        path_k=0.0,
    )
    assert two_pass_outcome.judge_passes == 2
    assert two_pass_outcome.repeat_bits == pytest.approx(
        -__import__("math").log2(2 / 3)
    )


def test_repeated_judge_all_misses_reject_without_repeat_cost() -> None:
    submission = Submission((Idea("idea", "candidate", "1"),))

    outcome, verdicts = IDEA_RECOVERY_V1.score_repeated_submission(
        submission,
        (
            (IdeaVerdict("idea", False, "miss one"),),
            (IdeaVerdict("idea", False, "miss two"),),
        ),
        path_k=2.0,
    )

    assert outcome.status == "fail"
    assert outcome.k == pytest.approx(2.0)
    assert outcome.repeat_bits is None
    assert verdicts[0].passed is False


def test_installation_verification_is_behavioral() -> None:
    report = verify_installation()

    assert report["verified"] is True
    assert report["basis"] == "behavioral-contract"


def test_submit_option_is_distinct_in_hashes_and_wire_round_trips() -> None:
    ordinary = Question("next", (Option("go", {"mode": "go"}, "1"),))
    submit = Question("next", (SubmitOption("go", {"mode": "go"}, "1"),))

    encoded = encode_message(submit)

    assert encoded["value"]["options"][0]["type"] == "submit_option"
    assert decode_message(encoded) == submit
    assert message_hash(ordinary) != message_hash(submit)


def test_submission_feedback_is_immutable_validated_and_wire_round_trips() -> None:
    source = PresentedQuestion(
        "q-source",
        Question(
            "choose",
            (
                Option("continue", "continue", "0.75"),
                SubmitOption("submit", "submit", "0.25"),
            ),
        ),
    )
    submission = Submission((Idea("idea", {"answer": "candidate"}, "1"),))
    feedback = SubmissionFeedback(
        source,
        submission,
        [IdeaVerdict("idea", False, "missing mechanism")],
        ["q-source", "q-earlier"],
    )

    validate_submission_feedback(feedback)
    decoded = decode_message(encode_message(feedback))

    assert isinstance(feedback.verdicts, tuple)
    assert isinstance(feedback.valid_checkout_question_ids, tuple)
    assert decoded == feedback


def test_individually_legal_large_messages_fit_the_feedback_envelope() -> None:
    source = PresentedQuestion(
        "q-large",
        Question(
            "large but legal source",
            (SubmitOption("submit", "q" * 120_000, "1"),),
        ),
    )
    submission = Submission((Idea("idea", "s" * 120_000, "1"),))
    feedback = SubmissionFeedback(
        source,
        submission,
        (IdeaVerdict("idea", False, "r" * 120_000),),
        ("q-large",),
    )

    # Question, Submission, and verdict vector each satisfy their component
    # cap even though their Arena-authored recovery envelope is larger than a
    # participant message.
    IDEA_RECOVERY_V1.validate_question(source.question)
    IDEA_RECOVERY_V1.validate_submission(submission)
    validate_submission_feedback(feedback)


def test_evaluation_base_reexports_the_contract_verdict_type() -> None:
    from tech_tree_arena.contract.messages import IdeaVerdict as ContractIdeaVerdict
    from tech_tree_arena.evaluation.base import IdeaVerdict as EvaluationIdeaVerdict

    assert EvaluationIdeaVerdict is ContractIdeaVerdict
