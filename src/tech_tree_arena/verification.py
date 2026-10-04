"""Behavioral installation checks for the normative arena contract."""

from __future__ import annotations

import math

from .contract import IDEA_RECOVERY_V1
from .errors import ValidationError
from .contract.messages import (
    Choice,
    Idea,
    Option,
    PresentedQuestion,
    Question,
    Submission,
    SubmissionFeedback,
    SubmitOption,
)
from .evaluation.base import IdeaVerdict


def verify_installation() -> dict[str, object]:
    """Exercise contract behavior without comparing implementation source files."""

    question = Question(
        "contract probe",
        (
            SubmitOption("chosen", {"committed": "generator payload"}, "0.25"),
            Option("other", None, "0.75"),
        ),
    )
    resolved = IDEA_RECOVERY_V1.resolve_choice(
        Choice("chosen", public_payload={"committed": "oracle payload"}),
        question,
    )
    submission = Submission(
        (
            Idea("accepted", {"idea": "semantic candidate"}, "0.4"),
            Idea("rejected", {"idea": "alternative"}, "0.6"),
        )
    )
    outcome = IDEA_RECOVERY_V1.score_submission(
        submission,
        (
            IdeaVerdict("accepted", True),
            IdeaVerdict("rejected", False),
        ),
        path_k=resolved.information_bits,
    )
    feedback = SubmissionFeedback(
        PresentedQuestion("q-probe", question),
        submission,
        (
            IdeaVerdict("accepted", False, "private probe"),
            IdeaVerdict("rejected", False, "private probe"),
        ),
        ("q-probe",),
    )
    IDEA_RECOVERY_V1.validate_submission_feedback(feedback)

    checks = {
        "choice_pricing": math.isclose(resolved.information_bits, 2.0),
        "oracle_submit_authority": isinstance(resolved.option, SubmitOption),
        "generator_committed_payload": resolved.generator_choice.public_payload
        == {"committed": "generator payload"},
        "oracle_private_recovery_feedback": feedback.valid_checkout_question_ids
        == ("q-probe",),
        "passing_mass": str(outcome.passing_mass) == "0.4",
        "passing_attempt_scoring": outcome.status == "pass"
        and math.isclose(outcome.score, 0.1)
        and math.isclose(outcome.k, -math.log2(0.1)),
    }
    if not all(checks.values()):
        failed = ", ".join(name for name, passed in checks.items() if not passed)
        raise ValidationError(f"arena contract verification failed: {failed}")
    return {
        "verified": True,
        "protocol": IDEA_RECOVERY_V1.name,
        "version": IDEA_RECOVERY_V1.version,
        "basis": "behavioral-contract",
        "checks": checks,
    }
