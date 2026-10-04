"""High-level idea sampling with a probability-weighted random Oracle."""

from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Any, Callable, Mapping

from .contract import IDEA_RECOVERY_V1
from .contract.messages import (
    Checkout,
    Choice,
    Idea,
    PresentedQuestion,
    Question,
    Submission,
    SubmissionFeedback,
)
from .errors import ProtocolError
from .runtime.services import ReplayableServices


SamplingEventSink = Callable[[dict[str, Any]], None]


def _draft_from_question(question: Question) -> str | None:
    """Read a Generator-authored draft already committed to a public payload."""

    candidates: list[str] = []
    for option in question.options:
        payload = option.public_payload
        if not isinstance(payload, dict):
            continue
        direct = payload.get("draft")
        if isinstance(direct, str) and direct.strip():
            candidates.append(direct.strip())
        for container_name in ("belief", "preview"):
            container = payload.get(container_name)
            if not isinstance(container, dict):
                continue
            top_idea = container.get("top_idea")
            if isinstance(top_idea, str) and top_idea.strip():
                candidates.append(top_idea.strip())
    return max(candidates, key=len) if candidates else None


class _ProgressModelBackend:
    """Report model-call boundaries without exposing prompts or responses."""

    def __init__(self, backend: Any, event_sink: SamplingEventSink | None) -> None:
        self.backend = backend
        self.event_sink = event_sink
        self.calls = 0
        self.lock = threading.Lock()

    def _emit(self, kind: str, **fields: Any) -> None:
        if self.event_sink is not None:
            self.event_sink({"kind": kind, **fields})

    def structured(self, **request: Any) -> Any:
        return self._structured(None, **request)

    @property
    def supports_context_chain(self) -> bool:
        return bool(getattr(self.backend, "supports_context_chain", False))

    @property
    def reasoning_effort(self):
        return getattr(self.backend, "reasoning_effort", None)

    def structured_in_context(self, context: dict | None, **request: Any) -> Any:
        return self._structured(context, **request)

    def _structured(self, context: dict | None, **request: Any) -> Any:
        with self.lock:
            self.calls += 1
            call = self.calls
        started_at = time.monotonic()
        public_request = {
            "call": call,
            "model": request.get("model"),
            "schema_name": request.get("schema_name"),
            "max_output_tokens": request.get("max_output_tokens"),
        }
        self._emit("model_call_started", **public_request)
        try:
            contextual = getattr(self.backend, "structured_in_context", None)
            response = contextual(context, **request) if callable(contextual) else self.backend.structured(**request)
        except Exception as exc:
            self._emit(
                "model_call_finished",
                **public_request,
                latency_s=time.monotonic() - started_at,
                error=type(exc).__name__,
            )
            raise

        metadata_fn = getattr(self.backend, "last_call_metadata", None)
        metadata = metadata_fn() if callable(metadata_fn) else {}
        usage = metadata.get("usage") or {}
        self._emit(
            "model_call_finished",
            **public_request,
            latency_s=time.monotonic() - started_at,
            cost_usd=float(usage.get("cost_usd", 0.0)),
            input_tokens=int(usage.get("input_tokens", 0)),
            output_tokens=int(usage.get("output_tokens", 0)),
        )
        return response

    def usage_totals(self) -> dict[str, Any]:
        usage_fn = getattr(self.backend, "usage_totals", None)
        return dict(usage_fn()) if callable(usage_fn) else {}

    def last_call_metadata(self) -> dict[str, Any]:
        metadata_fn = getattr(self.backend, "last_call_metadata", None)
        return dict(metadata_fn()) if callable(metadata_fn) else {}

    def restore_usage(self, usage: dict[str, Any]) -> None:
        restore_fn = getattr(self.backend, "restore_usage", None)
        if callable(restore_fn):
            restore_fn(usage)


class ProbabilitySamplingOracle:
    """Select Generator options according to their declared probabilities.

    Randomness comes exclusively from the injected service, so a fixed seed
    produces a reproducible option path when Generator outputs are unchanged.
    """

    def __init__(self, services: ReplayableServices) -> None:
        self.services = services

    def step(self, message: PresentedQuestion | SubmissionFeedback) -> Choice | Checkout:
        if isinstance(message, SubmissionFeedback):
            # There is no Generator-authored distribution over recovery handles.
            # The source question is always valid and is the least surprising
            # recovery behavior when this Oracle is used in a full arena run.
            return Checkout(message.source.question_id)
        if not isinstance(message, PresentedQuestion):
            raise ProtocolError("probability-sampling oracle requires a PresentedQuestion")

        probabilities = IDEA_RECOVERY_V1.validate_question(message.question)
        total = sum(probabilities, Decimal(0))
        draw = Decimal(str(self.services.random())) * total
        cumulative = Decimal(0)
        for option, probability in zip(
            message.question.options, probabilities, strict=True
        ):
            cumulative += probability
            if draw < cumulative:
                return Choice(option.option_id)

        # random.Random.random() is strictly below one, but retaining the final
        # option makes the sampler robust to another conforming service backend
        # returning the upper endpoint after decimal conversion.
        return Choice(message.question.options[-1].option_id)


@dataclass(frozen=True, slots=True)
class SampledChoice:
    question_index: int
    option_id: str
    probability: str
    submit: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "question_index": self.question_index,
            "option_id": self.option_id,
            "probability": self.probability,
            "submit": self.submit,
        }


@dataclass(frozen=True, slots=True)
class SampledIdeas:
    """A Generator submission and the random path that produced it."""

    run_id: str
    run_dir: str
    pair: str
    target_pack: str | None
    seed: int
    submission: Submission
    choices: tuple[SampledChoice, ...]
    generator_usage: Mapping[str, Any]
    oracle_usage: Mapping[str, Any]
    resumed_from: str | None = None

    @property
    def ideas(self) -> tuple[Idea, ...]:
        return self.submission.ideas

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "run_dir": self.run_dir,
            "resumed_from": self.resumed_from,
            "pair": self.pair,
            "target_pack": self.target_pack,
            "seed": self.seed,
            "questions": len(self.choices),
            "choices": [choice.to_dict() for choice in self.choices],
            "ideas": [
                {
                    "idea_id": idea.idea_id,
                    "content": idea.content,
                    "probability": idea.probability,
                }
                for idea in self.ideas
            ],
            "usage": {
                "generator": dict(self.generator_usage),
                "oracle": dict(self.oracle_usage),
            },
        }


def sample_ideas(
    pair: str | Path,
    *,
    target_pack: str | Path | None = "smoke",
    seed: int = 1,
    max_questions: int = 256,
    max_model_cost_usd: float = 1000.0,
    model_name: str | None = None,
    model_backend: Any | None = None,
    generator_codex: dict[str, Any] | None = None,
    generator_claude: dict[str, Any] | None = None,
    generator_memory: dict[str, Any] | None = None,
    generator_output_tokens: int | None = 128_000,
    budget_ledger: str | Path | None = None,
    public_resources: Mapping[str, Any] | None = None,
    event_sink: SamplingEventSink | None = None,
    runs_dir: str | Path | None = None,
) -> SampledIdeas:
    """Persistently run a Generator against a probability-weighted random Oracle.

    The pair's original Oracle is never constructed and no private target is
    loaded. Public resources are loaded from ``target_pack`` when provided.
    The first valid Submission reached through a sampled SubmitOption is
    returned without semantic evaluation. The run uses the same snapshots,
    checkpoints, service journal, replay, status, and resume machinery as a
    scored Arena run.
    """

    # Imported lazily to keep participant-facing ``tech_tree_arena`` imports
    # free of CLI orchestration and to avoid a module cycle at process startup.
    from .cli import _run_submission

    raw = _run_submission(
        Path(pair),
        str(target_pack) if target_pack is not None else None,
        None,
        seed,
        runs_dir=Path(runs_dir) if runs_dir is not None else None,
        allow_time_travel=False,
        max_cost_usd_per_role=max_model_cost_usd,
        sample_mode=True,
        sample_max_questions=max_questions,
        sample_model_name=model_name,
        sample_model_backend=model_backend,
        generator_codex=generator_codex,
        generator_claude=generator_claude,
        generator_memory=generator_memory,
        generator_output_tokens=generator_output_tokens,
        budget_ledger=budget_ledger,
        sample_public_resources=dict(public_resources or {}),
        sample_event_sink=event_sink,
    )
    return _sampled_ideas_from_raw(raw)


def resume_sample_ideas(
    run_directory: str | Path,
    *,
    max_model_cost_usd: float | None = None,
    retry_interrupted_call: bool = False,
    compatible_pair: str | Path | None = None,
    model_backend: Any | None = None,
    event_sink: SamplingEventSink | None = None,
) -> SampledIdeas:
    """Continue a failed sample run from its last durable checkpoint.

    The resumed run is a new, fully audited run whose ``resumed_from`` field
    points at ``run_directory``. Use ``retry_interrupted_call=True`` to retry a
    provider call that failed after the last committed checkpoint.
    """

    source = Path(run_directory).expanduser().resolve()
    try:
        manifest = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ProtocolError("resume source has no valid manifest") from exc
    if manifest.get("run_kind") != "sample_ideas":
        raise ProtocolError("resume source is not a sample-ideas run")

    from .cli import _resume_submission

    raw = _resume_submission(
        source,
        max_cost_usd_per_role=max_model_cost_usd,
        retry_interrupted_call=retry_interrupted_call,
        compatible_submission=compatible_pair,
        model_backend=model_backend,
        sample_event_sink=event_sink,
    )
    return _sampled_ideas_from_raw(raw)


def _sampled_ideas_from_raw(raw: Mapping[str, Any]) -> SampledIdeas:
    submission = Submission(
        tuple(
            Idea(item["idea_id"], item["content"], item["probability"])
            for item in raw["ideas"]
        )
    )
    return SampledIdeas(
        run_id=str(raw["run_id"]),
        run_dir=str(raw["run_dir"]),
        resumed_from=(
            str(raw["resumed_from"]) if raw.get("resumed_from") is not None else None
        ),
        pair=str(raw["submission"]),
        target_pack=(
            str(raw["target_pack"]) if raw.get("target_pack") is not None else None
        ),
        seed=int(raw["seed"]),
        submission=submission,
        choices=tuple(
            SampledChoice(
                int(item["question_index"]),
                str(item["option_id"]),
                str(item["probability"]),
                bool(item["submit"]),
            )
            for item in raw["choices"]
        ),
        generator_usage=dict((raw.get("usage") or {}).get("generator") or {}),
        oracle_usage=dict((raw.get("usage") or {}).get("oracle") or {}),
    )
