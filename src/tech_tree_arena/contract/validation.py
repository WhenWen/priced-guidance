"""Validation for the bundled ``idea-recovery-v1`` protocol."""

from __future__ import annotations

import json
import math
from dataclasses import fields, is_dataclass
from decimal import Decimal
from hashlib import sha256
from typing import Any

from ..errors import InvalidChoice, ValidationError
from .messages import (
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
from .pricing import validate_distribution

MAX_QUESTION_CHARS = 16_384
MAX_OPTIONS = 1_024
MAX_PAYLOAD_BYTES = 262_144
# Feedback is an Arena-authored envelope containing an already validated
# Question, Submission, verdict vector, and checkout list. Giving the envelope
# its own bound prevents two individually legal participant messages from
# becoming impossible to return after a rejection.
MAX_FEEDBACK_BYTES = 5 * MAX_PAYLOAD_BYTES
MAX_ID_CHARS = 256
MAX_SUBMISSION_IDEAS = 256
MAX_NESTING = 32
MAX_CONTAINER_ITEMS = 4_096
MAX_STRING_CHARS = 131_072
MAX_INTEGER = 2**63 - 1


def _validate_id(value: str, field: str) -> None:
    if not isinstance(value, str) or not value or len(value) > MAX_ID_CHARS:
        raise ValidationError(f"{field} must be a non-empty bounded string")
    if any(ord(char) < 32 for char in value):
        raise ValidationError(f"{field} contains a control character")


def _json_value(value: Any, depth: int = 0) -> Any:
    if depth > MAX_NESTING:
        raise ValidationError("message exceeds the nesting limit")
    # ``Option`` and ``SubmitOption`` intentionally have the same public
    # fields. Preserve their semantic distinction in canonical hashes even
    # when they are nested inside a Question dataclass.
    if isinstance(value, SubmitOption):
        return {
            "__arena_type__": "submit_option",
            **{
                item.name: _json_value(getattr(value, item.name), depth + 1)
                for item in fields(value)
            },
        }
    if is_dataclass(value):
        return {
            item.name: _json_value(getattr(value, item.name), depth + 1)
            for item in fields(value)
        }
    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, int):
        if abs(value) > MAX_INTEGER:
            raise ValidationError("message integer exceeds the protocol range")
        return value
    if isinstance(value, str):
        if len(value) > MAX_STRING_CHARS:
            raise ValidationError("message string exceeds the protocol limit")
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValidationError("message contains a non-finite number")
        if value == 0.0 and math.copysign(1.0, value) < 0:
            raise ValidationError("message contains ambiguous negative zero")
        return value
    if isinstance(value, Decimal):
        if not value.is_finite():
            raise ValidationError("message contains a non-finite decimal")
        if value.is_zero() and value.is_signed():
            raise ValidationError("message contains ambiguous negative zero")
        return str(value)
    if isinstance(value, (list, tuple)):
        if len(value) > MAX_CONTAINER_ITEMS:
            raise ValidationError("message container exceeds the item limit")
        return [_json_value(item, depth + 1) for item in value]
    if isinstance(value, dict):
        if not all(isinstance(key, str) for key in value):
            raise ValidationError("message object keys must be strings")
        if len(value) > MAX_CONTAINER_ITEMS:
            raise ValidationError("message container exceeds the item limit")
        return {key: _json_value(item, depth + 1) for key, item in value.items()}
    raise ValidationError("message contains a non-canonical JSON value")


def canonical_json(value: Any, *, max_bytes: int = MAX_PAYLOAD_BYTES) -> bytes:
    value = _json_value(value)
    try:
        encoded = json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except (TypeError, ValueError, UnicodeError) as exc:
        raise ValidationError("message contains a non-canonical JSON value") from exc
    if len(encoded) > max_bytes:
        raise ValidationError("message exceeds the protocol payload limit")
    return encoded


def message_hash(value: Any) -> str:
    return sha256(canonical_json(value)).hexdigest()


def validate_question(question: Question) -> tuple:
    if not isinstance(question, Question):
        raise ValidationError("generator must return Question or Submission")
    if not isinstance(question.question, str) or not question.question.strip():
        raise ValidationError("question text must not be empty")
    if len(question.question) > MAX_QUESTION_CHARS:
        raise ValidationError("question text exceeds the protocol limit")
    if not 1 <= len(question.options) <= MAX_OPTIONS:
        raise ValidationError("question has an invalid number of options")
    seen: set[str] = set()
    for option in question.options:
        if not isinstance(option, (Option, SubmitOption)):
            raise ValidationError("every question option must be an Option or SubmitOption")
        _validate_id(option.option_id, "option_id")
        if option.option_id in seen:
            raise ValidationError("question option IDs must be unique")
        seen.add(option.option_id)
        canonical_json(option.public_payload)
    probabilities = validate_distribution(question.options)
    canonical_json(question)
    return probabilities


def validate_choice(choice: Choice, question: Question) -> tuple[Option | SubmitOption, int]:
    if not isinstance(choice, Choice):
        raise InvalidChoice("oracle must return Choice or Checkout")
    _validate_id(choice.option_id, "option_id")
    for index, option in enumerate(question.options):
        if option.option_id == choice.option_id:
            return option, index
    raise InvalidChoice("oracle selected an option not present in the question")


def validate_checkout(checkout: Checkout) -> None:
    if not isinstance(checkout, Checkout):
        raise InvalidChoice("oracle must return Choice or Checkout")
    _validate_id(checkout.question_id, "checkout question_id")


def validate_submission(submission: Submission) -> tuple:
    if not isinstance(submission, Submission):
        raise ValidationError("generator must return Question or Submission")
    if not 1 <= len(submission.ideas) <= MAX_SUBMISSION_IDEAS:
        raise ValidationError("submission has an invalid number of ideas")
    seen: set[str] = set()
    for idea in submission.ideas:
        if not isinstance(idea, Idea):
            raise ValidationError("every submission item must be an Idea")
        _validate_id(idea.idea_id, "idea_id")
        if idea.idea_id in seen:
            raise ValidationError("submission idea IDs must be unique")
        seen.add(idea.idea_id)
        canonical_json(idea.content)
    probabilities = validate_distribution(submission.ideas)
    canonical_json(submission)
    return probabilities


def validate_submission_feedback(feedback: SubmissionFeedback) -> None:
    """Validate Arena-authored feedback before it crosses into the Oracle."""

    if not isinstance(feedback, SubmissionFeedback):
        raise ValidationError("oracle feedback must be a SubmissionFeedback")
    if not isinstance(feedback.source, PresentedQuestion):
        raise ValidationError("submission feedback source must be a PresentedQuestion")
    _validate_id(feedback.source.question_id, "feedback source question_id")
    validate_question(feedback.source.question)
    validate_submission(feedback.submission)

    idea_ids = tuple(idea.idea_id for idea in feedback.submission.ideas)
    if len(feedback.verdicts) != len(idea_ids):
        raise ValidationError("submission feedback must contain one verdict per idea")
    verdict_ids: list[str] = []
    for verdict in feedback.verdicts:
        if not isinstance(verdict, IdeaVerdict):
            raise ValidationError("submission feedback contains an invalid idea verdict")
        _validate_id(verdict.idea_id, "feedback verdict idea_id")
        if type(verdict.passed) is not bool:
            raise ValidationError("feedback verdict passed must be a boolean")
        if not isinstance(verdict.private_reason, str):
            raise ValidationError("feedback verdict private_reason must be a string")
        canonical_json(verdict.private_reason)
        verdict_ids.append(verdict.idea_id)
    if len(set(verdict_ids)) != len(verdict_ids) or set(verdict_ids) != set(idea_ids):
        raise ValidationError("submission feedback verdict IDs must match the submission")
    # Bound the trusted diagnostic vector independently so the larger envelope
    # allowance cannot be used to smuggle an unbounded set of reasons.
    canonical_json(feedback.verdicts)

    checkout_ids = feedback.valid_checkout_question_ids
    if not checkout_ids:
        raise ValidationError("submission feedback must provide a valid checkout target")
    seen_checkout_ids: set[str] = set()
    for question_id in checkout_ids:
        _validate_id(question_id, "feedback checkout question_id")
        if question_id in seen_checkout_ids:
            raise ValidationError("submission feedback checkout IDs must be unique")
        seen_checkout_ids.add(question_id)
    if feedback.source.question_id not in seen_checkout_ids:
        raise ValidationError("submission feedback must allow checkout to its source question")
    canonical_json(checkout_ids)
    canonical_json(feedback, max_bytes=MAX_FEEDBACK_BYTES)
