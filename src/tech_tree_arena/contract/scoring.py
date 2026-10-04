"""Normative behavior for the ``idea-recovery-v1`` arena protocol.

The contract owns participant-message validation, information pricing, verdict
validation, attempt scoring, and final-pass scoring. Implementations and reference strategies may
change without changing these rules.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import Iterable, Literal

from ..errors import ProtocolError
from ..evaluation.base import IdeaVerdict, aggregate_repeated_verdicts
from .messages import Checkout, Choice, Question, QuestionOption, Submission, SubmissionFeedback
from .pricing import accepted_mass, information_cost
from .validation import (
    validate_checkout,
    validate_choice,
    validate_question,
    validate_submission,
    validate_submission_feedback,
)
from . import PROTOCOL_NAME, PROTOCOL_VERSION


@dataclass(frozen=True, slots=True)
class ResolvedChoice:
    """A validated oracle choice and the arena-authored generator response."""

    option: QuestionOption
    probability: Decimal
    information_bits: float
    generator_choice: Choice


@dataclass(frozen=True, slots=True)
class SubmissionOutcome:
    """One judged attempt; only a passing outcome is terminal for the run."""

    status: Literal["pass", "fail"]
    score: float
    k: float
    matched_idea_ids: tuple[str, ...]
    passing_mass: Decimal
    submission_bits: float | None
    judge_repeats: int = 1
    judge_passes: int = 0
    judge_pass_rate: Decimal = Decimal(0)
    repeat_bits: float | None = None


class IdeaRecoveryV1Contract:
    """Executable form of the public ``idea-recovery-v1`` contract."""

    name = PROTOCOL_NAME
    version = PROTOCOL_VERSION

    @staticmethod
    def validate_question(question: Question) -> tuple[Decimal, ...]:
        return validate_question(question)

    @staticmethod
    def validate_checkout(checkout: Checkout) -> None:
        validate_checkout(checkout)

    @staticmethod
    def validate_submission(submission: Submission) -> tuple[Decimal, ...]:
        return validate_submission(submission)

    @staticmethod
    def validate_submission_feedback(feedback: SubmissionFeedback) -> None:
        validate_submission_feedback(feedback)

    @staticmethod
    def resolve_choice(choice: Choice, question: Question) -> ResolvedChoice:
        option, option_index = validate_choice(choice, question)
        probabilities = validate_question(question)
        probability = probabilities[option_index] / accepted_mass(probabilities)
        return ResolvedChoice(
            option=option,
            probability=probability,
            information_bits=information_cost(probability),
            # Oracle-authored payloads are never forwarded.  Only the payload
            # committed by the generator before the oracle chose is visible.
            generator_choice=Choice(
                option_id=option.option_id,
                public_payload=option.public_payload,
            ),
        )

    @staticmethod
    def score_submission(
        submission: Submission,
        verdicts: Iterable[IdeaVerdict],
        *,
        path_k: float,
    ) -> SubmissionOutcome:
        probabilities = validate_submission(submission)
        verdicts = tuple(verdicts)
        idea_ids = tuple(idea.idea_id for idea in submission.ideas)
        if (
            len(verdicts) != len(idea_ids)
            or any(not isinstance(verdict, IdeaVerdict) for verdict in verdicts)
            or any(type(verdict.passed) is not bool for verdict in verdicts)
            or len({verdict.idea_id for verdict in verdicts}) != len(verdicts)
            or {verdict.idea_id for verdict in verdicts} != set(idea_ids)
        ):
            raise ProtocolError("judge returned an invalid verdict set")

        by_id = {verdict.idea_id: verdict for verdict in verdicts}
        matched = tuple(idea_id for idea_id in idea_ids if by_id[idea_id].passed)
        if not matched:
            return SubmissionOutcome(
                status="fail",
                score=0.0,
                k=path_k,
                matched_idea_ids=(),
                passing_mass=Decimal(0),
                submission_bits=None,
                judge_repeats=1,
                judge_passes=0,
                judge_pass_rate=Decimal(0),
                repeat_bits=None,
            )

        accepted = (
            probability
            for idea_id, probability in zip(idea_ids, probabilities, strict=True)
            if idea_id in matched
        )
        # Validation allows a tiny serialization tolerance. Normalize against
        # the submitted total so accepted mass can never exceed one.
        passing_mass = accepted_mass(accepted) / accepted_mass(probabilities)
        submission_bits = information_cost(passing_mass)
        final_k = path_k + submission_bits
        return SubmissionOutcome(
            status="pass",
            score=2.0 ** (-final_k),
            k=final_k,
            matched_idea_ids=matched,
            passing_mass=passing_mass,
            submission_bits=submission_bits,
            judge_repeats=1,
            judge_passes=1,
            judge_pass_rate=Decimal(1),
            repeat_bits=0.0,
        )

    @staticmethod
    def score_repeated_submission(
        submission: Submission,
        verdict_rounds: Iterable[Iterable[IdeaVerdict]],
        *,
        path_k: float,
    ) -> tuple[SubmissionOutcome, tuple[IdeaVerdict, ...]]:
        """Score fixed independent Judge repetitions for one submission.

        A repetition passes when any submitted idea passes in that round. If
        ``l`` of ``k`` repetitions pass and ``l >= 1``, the submission is
        terminal and pays ``-log2(l / k)`` in addition to its ordinary pointer
        cost. All-zero repetitions reject without charging either cost.
        """

        rounds = tuple(tuple(verdicts) for verdicts in verdict_rounds)
        aggregated = aggregate_repeated_verdicts(submission.ideas, rounds)
        round_outcomes = tuple(
            IdeaRecoveryV1Contract.score_submission(
                submission,
                verdicts,
                path_k=path_k,
            )
            for verdicts in rounds
        )
        repeats = len(round_outcomes)
        passes = sum(outcome.status == "pass" for outcome in round_outcomes)
        if passes == 0:
            return (
                SubmissionOutcome(
                    status="fail",
                    score=0.0,
                    k=path_k,
                    matched_idea_ids=(),
                    passing_mass=Decimal(0),
                    submission_bits=None,
                    judge_repeats=repeats,
                    judge_passes=0,
                    judge_pass_rate=Decimal(0),
                    repeat_bits=None,
                ),
                aggregated,
            )

        base = IdeaRecoveryV1Contract.score_submission(
            submission,
            aggregated,
            path_k=path_k,
        )
        pass_rate = Decimal(passes) / Decimal(repeats)
        repeat_bits = information_cost(pass_rate)
        final_k = base.k + repeat_bits
        return (
            SubmissionOutcome(
                status="pass",
                score=2.0 ** (-final_k),
                k=final_k,
                matched_idea_ids=base.matched_idea_ids,
                passing_mass=base.passing_mass,
                submission_bits=base.submission_bits,
                judge_repeats=repeats,
                judge_passes=passes,
                judge_pass_rate=pass_rate,
                repeat_bits=repeat_bits,
            ),
            aggregated,
        )


def mixture_pass_rate(
    submission: Submission,
    verdict_rounds: Iterable[Iterable[IdeaVerdict]],
) -> Decimal:
    """Estimate the pass rate of one idea drawn from ``submission``.

    An unguided generator submits a single idea sampled with the submitted
    probabilities, so its acceptance probability is ``sum_i p_i * P(i passes)``.
    With ``k`` independent Judge rounds, ``sum_i p_i * l_i / k`` estimates it
    without bias, where ``l_i`` counts the rounds in which idea ``i`` passes.

    ``score_repeated_submission`` instead multiplies the mass of ideas that pass
    in any round by the fraction of rounds with any pass. That product is never
    smaller than this estimate and equals it for one idea or one round. Runtime
    scoring is unchanged; this function supports post-hoc correction.
    """

    probabilities = validate_submission(submission)
    idea_ids = tuple(idea.idea_id for idea in submission.ideas)
    rounds = tuple(tuple(verdicts) for verdicts in verdict_rounds)
    if not rounds:
        raise ProtocolError("mixture pass rate needs at least one verdict round")
    passes = dict.fromkeys(idea_ids, 0)
    for verdicts in rounds:
        by_id = {verdict.idea_id: verdict for verdict in verdicts}
        if set(by_id) != set(idea_ids) or len(by_id) != len(verdicts):
            raise ProtocolError("judge returned an invalid verdict set")
        for idea_id in idea_ids:
            passes[idea_id] += int(by_id[idea_id].passed is True)
    total = accepted_mass(probabilities)
    weighted = sum(
        (
            probability * passes[idea_id]
            for idea_id, probability in zip(idea_ids, probabilities, strict=True)
        ),
        Decimal(0),
    )
    return weighted / (total * len(rounds))


IDEA_RECOVERY_V1 = IdeaRecoveryV1Contract()
