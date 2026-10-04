"""Canonical JSON wire representation for local actor processes."""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from typing import Any

from ..contract.messages import (
    Checkout,
    Choice,
    Idea,
    IdeaVerdict,
    Option,
    PresentedQuestion,
    Question,
    Submission,
    SubmissionFeedback,
    StageReady,
    StageTransition,
    SubmitOption,
)


def _object(
    value: Any,
    *,
    name: str,
    required: frozenset[str],
) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != required:
        raise TypeError(f"{name} does not match the arena wire contract")
    return value


def _array(value: Any, *, name: str) -> list[Any]:
    if not isinstance(value, list):
        raise TypeError(f"{name} must be an array")
    return value


def json_value(value: Any) -> Any:
    if is_dataclass(value):
        return json_value(asdict(value))
    if isinstance(value, (list, tuple)):
        return [json_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): json_value(item) for key, item in value.items()}
    return value


def _option_value(value: Option | SubmitOption) -> dict[str, Any]:
    result = {
        "option_id": value.option_id,
        "public_payload": json_value(value.public_payload),
        "probability": value.probability,
    }
    if isinstance(value, SubmitOption):
        result["kind"] = value.kind
    return result


def _option_record(value: Option | SubmitOption) -> dict[str, Any]:
    return {
        "type": "submit_option" if isinstance(value, SubmitOption) else "option",
        "value": _option_value(value),
    }


def _question_value(value: Question) -> dict[str, Any]:
    return {
        "question": value.question,
        "options": [_option_record(option) for option in value.options],
    }


def _presented_question_value(value: PresentedQuestion) -> dict[str, Any]:
    return {
        "question_id": value.question_id,
        "question": _question_value(value.question),
    }


def _idea_value(value: Idea) -> dict[str, Any]:
    return {
        "idea_id": value.idea_id,
        "content": json_value(value.content),
        "probability": value.probability,
    }


def _submission_value(value: Submission) -> dict[str, Any]:
    return {"ideas": [_idea_value(idea) for idea in value.ideas]}


def _verdict_value(value: IdeaVerdict) -> dict[str, Any]:
    return {
        "idea_id": value.idea_id,
        "passed": value.passed,
        "private_reason": value.private_reason,
    }


def encode_message(value: Any) -> dict[str, Any]:
    if value is None:
        return {"type": "none"}
    if isinstance(value, SubmitOption):
        return {"type": "submit_option", "value": _option_value(value)}
    if isinstance(value, Option):
        return {"type": "option", "value": _option_value(value)}
    if isinstance(value, Question):
        return {"type": "question", "value": _question_value(value)}
    if isinstance(value, PresentedQuestion):
        return {"type": "presented_question", "value": _presented_question_value(value)}
    if isinstance(value, Choice):
        return {"type": "choice", "value": json_value(value)}
    if isinstance(value, Checkout):
        return {"type": "checkout", "value": json_value(value)}
    if isinstance(value, Idea):
        return {"type": "idea", "value": _idea_value(value)}
    if isinstance(value, Submission):
        return {"type": "submission", "value": _submission_value(value)}
    if isinstance(value, IdeaVerdict):
        return {"type": "idea_verdict", "value": _verdict_value(value)}
    if isinstance(value, SubmissionFeedback):
        return {
            "type": "submission_feedback",
            "value": {
                "source": _presented_question_value(value.source),
                "submission": _submission_value(value.submission),
                "verdicts": [_verdict_value(verdict) for verdict in value.verdicts],
                "valid_checkout_question_ids": list(value.valid_checkout_question_ids),
            },
        }
    if isinstance(value, StageTransition):
        return {"type": "stage_transition", "value": json_value(value)}
    if isinstance(value, StageReady):
        return {"type": "stage_ready", "value": json_value(value)}
    raise TypeError(f"unsupported wire message {type(value).__name__}")


def _option(value: dict[str, Any]) -> Option:
    value = _object(
        value,
        name="option",
        required=frozenset({"option_id", "public_payload", "probability"}),
    )
    return Option(value["option_id"], value["public_payload"], value["probability"])


def _submit_option(value: dict[str, Any]) -> SubmitOption:
    value = _object(
        value,
        name="submit option",
        required=frozenset({"option_id", "public_payload", "probability", "kind"}),
    )
    if value["kind"] != "submit":
        raise TypeError("submit option has an invalid kind discriminator")
    return SubmitOption(value["option_id"], value["public_payload"], value["probability"])


def _question_option(value: Any) -> Option | SubmitOption:
    # Decode legacy Option-only question records as well as the discriminated
    # representation emitted by the submit-aware protocol.
    if isinstance(value, dict) and set(value) == {"type", "value"}:
        kind = value.get("type")
        if kind == "option":
            return _option(value["value"])
        if kind == "submit_option":
            return _submit_option(value["value"])
        raise TypeError(f"unsupported question option type {kind!r}")
    return _option(value)


def _question(value: dict[str, Any]) -> Question:
    value = _object(
        value,
        name="question",
        required=frozenset({"question", "options"}),
    )
    return Question(
        value["question"],
        tuple(
            _question_option(option)
            for option in _array(value["options"], name="question options")
        ),
    )


def _idea(value: dict[str, Any]) -> Idea:
    value = _object(
        value,
        name="idea",
        required=frozenset({"idea_id", "content", "probability"}),
    )
    return Idea(value["idea_id"], value["content"], value["probability"])


def _submission(value: dict[str, Any]) -> Submission:
    value = _object(
        value,
        name="submission",
        required=frozenset({"ideas"}),
    )
    return Submission(tuple(_idea(item) for item in _array(value["ideas"], name="ideas")))


def _presented_question(value: dict[str, Any]) -> PresentedQuestion:
    if not isinstance(value, dict) or set(value) != {"question_id", "question"}:
        raise TypeError("presented question does not match the arena wire contract")
    return PresentedQuestion(value["question_id"], _question(value["question"]))


def _idea_verdict(value: dict[str, Any]) -> IdeaVerdict:
    value = _object(
        value,
        name="idea verdict",
        required=frozenset({"idea_id", "passed", "private_reason"}),
    )
    return IdeaVerdict(value["idea_id"], value["passed"], value["private_reason"])


def decode_message(record: dict[str, Any]) -> Any:
    if not isinstance(record, dict) or not isinstance(record.get("type"), str):
        raise TypeError("wire message does not match the arena contract")
    kind = record["type"]
    if kind == "none":
        _object(record, name="none message", required=frozenset({"type"}))
        return None
    record = _object(
        record,
        name="wire message",
        required=frozenset({"type", "value"}),
    )
    value = record["value"]
    if kind == "option":
        return _option(value)
    if kind == "submit_option":
        return _submit_option(value)
    if kind == "question":
        return _question(value)
    if kind == "presented_question":
        return _presented_question(value)
    if kind == "choice":
        value = _object(
            value,
            name="choice",
            required=frozenset({"option_id", "question_id", "public_payload"}),
        )
        return Choice(value["option_id"], value["question_id"], value["public_payload"])
    if kind == "checkout":
        value = _object(
            value,
            name="checkout",
            required=frozenset({"question_id"}),
        )
        return Checkout(value["question_id"])
    if kind == "idea":
        return _idea(value)
    if kind == "submission":
        return _submission(value)
    if kind == "idea_verdict":
        return _idea_verdict(value)
    if kind == "submission_feedback":
        value = _object(
            value,
            name="submission feedback",
            required=frozenset(
                {
                    "source",
                    "submission",
                    "verdicts",
                    "valid_checkout_question_ids",
                }
            ),
        )
        checkout_ids = _array(
            value["valid_checkout_question_ids"],
            name="valid checkout question IDs",
        )
        return SubmissionFeedback(
            _presented_question(value["source"]),
            _submission(value["submission"]),
            tuple(
                _idea_verdict(item)
                for item in _array(value["verdicts"], name="idea verdicts")
            ),
            tuple(checkout_ids),
        )
    if kind == "stage_transition":
        value = _object(
            value,
            name="stage transition",
            required=frozenset({"from_stage", "to_stage"}),
        )
        return StageTransition(value["from_stage"], value["to_stage"])
    if kind == "stage_ready":
        value = _object(
            value,
            name="stage ready",
            required=frozenset({"stage", "handoff"}),
        )
        return StageReady(value["stage"], value["handoff"])
    raise TypeError(f"unsupported wire message type {kind!r}")
