import pytest

from tech_tree_arena.errors import ValidationError
from tech_tree_arena import (
    Checkout,
    Idea,
    IdeaVerdict,
    Option,
    PresentedQuestion,
    Question,
    Submission,
    SubmissionFeedback,
    SubmitOption,
)
from tech_tree_arena.contract.validation import (
    validate_checkout,
    validate_question,
    validate_submission_feedback,
)
from tech_tree_arena.runtime.wire import decode_message


def test_negative_zero_payload_fails_closed() -> None:
    with pytest.raises(ValidationError, match="negative zero"):
        validate_question(Question("q", (Option("x", -0.0, "1"),)))


def test_excessive_payload_nesting_fails_closed() -> None:
    payload = None
    for _ in range(40):
        payload = [payload]
    with pytest.raises(ValidationError, match="nesting"):
        validate_question(Question("q", (Option("x", payload, "1"),)))


@pytest.mark.parametrize("question_id", [[], "", "x" * 257, "bad\nhandle"])
def test_malformed_checkout_handles_fail_schema_validation(question_id) -> None:
    with pytest.raises(ValidationError, match="question_id"):
        validate_checkout(Checkout(question_id))


def test_submission_wire_rejects_fields_outside_the_contract() -> None:
    with pytest.raises(TypeError, match="wire contract"):
        decode_message(
            {
                "type": "submission",
                "value": {
                    "ideas": [
                        {"idea_id": "idea", "content": "content", "probability": "1"}
                    ],
                    "metadata": {"undeclared": True},
                },
            }
        )


def test_submit_option_wire_rejects_an_unknown_discriminator() -> None:
    with pytest.raises(TypeError, match="option type"):
        decode_message(
            {
                "type": "question",
                "value": {
                    "question": "q",
                    "options": [
                        {
                            "type": "magic_submit",
                            "value": {
                                "option_id": "submit",
                                "public_payload": None,
                                "probability": "1",
                            },
                        }
                    ],
                },
            }
        )


def test_feedback_verdict_ids_must_match_the_submission() -> None:
    feedback = SubmissionFeedback(
        PresentedQuestion(
            "source",
            Question("q", (SubmitOption("submit", None, "1"),)),
        ),
        Submission((Idea("idea", "candidate", "1"),)),
        (IdeaVerdict("different", False, "miss"),),
        ("source",),
    )

    with pytest.raises(ValidationError, match="verdict IDs"):
        validate_submission_feedback(feedback)


def test_feedback_must_allow_checkout_to_its_source_question() -> None:
    feedback = SubmissionFeedback(
        PresentedQuestion(
            "source",
            Question(
                "q",
                (
                    Option("continue", None, "0.5"),
                    SubmitOption("submit", None, "0.5"),
                ),
            ),
        ),
        Submission((Idea("idea", "candidate", "1"),)),
        (IdeaVerdict("idea", False, "miss"),),
        ("earlier",),
    )

    with pytest.raises(ValidationError, match="source question"):
        validate_submission_feedback(feedback)
