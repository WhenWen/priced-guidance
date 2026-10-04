"""Standalone generator-oracle arena."""

from .contract.validation import canonical_json
from .contract.messages import (
    Checkout,
    Choice,
    GeneratorOutput,
    Idea,
    IdeaVerdict,
    JsonValue,
    Option,
    OracleDecision,
    OracleInput,
    PresentedQuestion,
    Question,
    QuestionOption,
    Submission,
    SubmissionFeedback,
    StageReady,
    StageTransition,
    SubmitOption,
)

__all__ = [
    "canonical_json",
    "Checkout",
    "Choice",
    "GeneratorOutput",
    "Idea",
    "IdeaVerdict",
    "JsonValue",
    "Option",
    "OracleDecision",
    "OracleInput",
    "PresentedQuestion",
    "Question",
    "QuestionOption",
    "Submission",
    "SubmissionFeedback",
    "StageReady",
    "StageTransition",
    "SubmitOption",
    "ProbabilitySamplingOracle",
    "SampledChoice",
    "SampledIdeas",
    "resume_sample_ideas",
    "sample_ideas",
]


def __getattr__(name: str):
    if name in {
        "ProbabilitySamplingOracle",
        "SampledChoice",
        "SampledIdeas",
        "resume_sample_ideas",
        "sample_ideas",
    }:
        from . import sampling

        return getattr(sampling, name)
    raise AttributeError(name)

__version__ = "0.1.0"
