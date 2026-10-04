"""The complete participant-visible message surface."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, TypeAlias

JsonValue: TypeAlias = None | bool | int | float | str | list["JsonValue"] | dict[str, "JsonValue"]


@dataclass(frozen=True, slots=True)
class Option:
    option_id: str
    public_payload: JsonValue
    probability: int | float | str


@dataclass(frozen=True, slots=True)
class SubmitOption:
    """A priced option that asks the generator to produce a submission.

    This is deliberately a sibling of :class:`Option`, rather than an option
    whose payload happens to contain a magic value.  The wire protocol and
    canonical hashing preserve that distinction.
    """

    option_id: str
    public_payload: JsonValue
    probability: int | float | str
    kind: Literal["submit"] = field(default="submit", init=False)


@dataclass(frozen=True, slots=True)
class Question:
    question: str
    options: tuple[Option | SubmitOption, ...] | list[Option | SubmitOption]

    def __post_init__(self) -> None:
        object.__setattr__(self, "options", tuple(self.options))


@dataclass(frozen=True, slots=True)
class PresentedQuestion:
    question_id: str
    question: Question


@dataclass(frozen=True, slots=True)
class Choice:
    """An oracle choice.

    The oracle needs to set only ``option_id``.  The arena fills ``question_id``
    and ``public_payload`` in the copy delivered to the generator.
    """

    option_id: str
    question_id: str | None = None
    public_payload: JsonValue = None


@dataclass(frozen=True, slots=True)
class Checkout:
    question_id: str


@dataclass(frozen=True, slots=True)
class Idea:
    idea_id: str
    content: JsonValue
    probability: int | float | str


@dataclass(frozen=True, slots=True)
class Submission:
    ideas: tuple[Idea, ...] | list[Idea]

    def __post_init__(self) -> None:
        object.__setattr__(self, "ideas", tuple(self.ideas))


@dataclass(frozen=True, slots=True)
class IdeaVerdict:
    """One Arena-authored semantic verdict over a submitted idea."""

    idea_id: str
    passed: bool
    private_reason: str = ""


@dataclass(frozen=True, slots=True)
class SubmissionFeedback:
    """Private failed-submission feedback delivered only to the Oracle."""

    source: PresentedQuestion
    submission: Submission
    verdicts: tuple[IdeaVerdict, ...] | list[IdeaVerdict]
    valid_checkout_question_ids: tuple[str, ...] | list[str]

    def __post_init__(self) -> None:
        object.__setattr__(self, "verdicts", tuple(self.verdicts))
        object.__setattr__(
            self,
            "valid_checkout_question_ids",
            tuple(self.valid_checkout_question_ids),
        )


@dataclass(frozen=True, slots=True)
class StageTransition:
    """Arena-authored, zero-information switch between frozen policy modules.

    The message is delivered only after an earlier stage has been
    deterministically reconstructed.  Participants must handle it locally:
    using a model, random service, or agent service during a transition is a
    protocol error.  The actor's reply can expose a private, JSON-serializable
    handoff for the next stage without changing K.
    """

    from_stage: str
    to_stage: str


@dataclass(frozen=True, slots=True)
class StageReady:
    """Participant acknowledgement and private state handoff for a stage."""

    stage: str
    handoff: JsonValue = None


QuestionOption: TypeAlias = Option | SubmitOption
# Participant step traffic. On a Judge promotion the Arena sends both
# participants a StageTransition, which each must acknowledge by returning a
# StageReady without calling any participant service (see runtime.engine);
# the Generator additionally receives one on every checkout that rebuilds it
# behind the active stage. OracleDecision stays the priced-decision subset.
GeneratorOutput: TypeAlias = Question | Submission | StageReady
GuideInput: TypeAlias = PresentedQuestion | SubmissionFeedback | StageTransition
GuideDecision: TypeAlias = Choice | Checkout


# Legacy Python names remain available for existing submissions.
OracleInput = GuideInput
OracleDecision = GuideDecision
