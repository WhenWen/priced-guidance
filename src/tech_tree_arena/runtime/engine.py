"""Minimal generator-oracle game loop with optional time travel."""

from __future__ import annotations

from .._compat import legacy_fields, legacy_keywords

import math
import uuid
from dataclasses import dataclass
from typing import Any, Callable

from .actor import ActorFactory, ActorRuntime
from .branch import BranchStore, QuestionNode
from ..contract import IDEA_RECOVERY_V1, IdeaRecoveryV1Contract
from ..contract.validation import message_hash
from ..contract.recovery import CURRENT_ACCOUNTING, LEGACY_ACCOUNTING
from ..contract.messages import (
    Checkout,
    Choice,
    IdeaVerdict,
    PresentedQuestion,
    Question,
    Submission,
    SubmissionFeedback,
    StageReady,
    StageTransition,
    SubmitOption,
)
from ..errors import InvalidCheckout, ProtocolError, ResourceLimitExceeded
from ..evaluation.base import Judge


@legacy_fields(max_oracle_decisions='max_guide_decisions')
@dataclass(frozen=True, slots=True)
class RunLimits:
    max_questions: int = 256
    max_guide_decisions: int = 256
    max_checkouts: int = 64
    max_checkout_targets: int = 128
    max_checkout_rewind: int = 256
    max_depth: int = 256
    max_submission_attempts: int = 256
    max_bits: float = 1_024.0


@legacy_fields(oracle_decision_count='guide_decision_count')
@dataclass(frozen=True, slots=True)
class RunResult:
    run_id: str
    status: str
    score: float
    k: float
    matched_idea_ids: tuple[str, ...]
    verdicts: tuple[IdeaVerdict, ...]
    judge_repeats: int
    judge_passes: int
    judge_pass_rate: float
    repeat_bits: float
    question_count: int
    guide_decision_count: int
    checkout_count: int
    submission_attempt_count: int
    branch_store: BranchStore


@legacy_fields(oracle='guide', oracle_history='guide_history')
@dataclass(slots=True)
class EngineResumeState:
    phase: str
    branches: BranchStore
    generator: Any
    guide: Any
    k: float = 0.0
    decisions: int = 0
    checkouts: int = 0
    submission_attempts: int = 0
    parent_question_id: str | None = None
    current_question_id: str | None = None
    output: Any = None
    decision: Any = None
    submission: Submission | None = None
    submission_feedback: SubmissionFeedback | None = None
    submission_source_question_id: str | None = None
    submission_option_id: str | None = None
    submission_choice_bits: float = 0.0
    generator_history: list[tuple[str, Any]] | None = None
    guide_history: list[tuple[str, Any]] | None = None
    stage_transitions: tuple[StageTransition, ...] = ()
    accounting_update: dict[str, Any] | None = None


class ArenaRunner:
    def __init__(
        self,
        *,
        allow_time_travel: bool = True,
        limits: RunLimits | None = None,
        runtime: ActorRuntime | None = None,
        event_sink: Callable[[dict[str, Any]], None] | None = None,
        checkpoint_sink: Callable[[dict[str, Any], Any, Judge], None] | None = None,
        contract: IdeaRecoveryV1Contract = IDEA_RECOVERY_V1,
        should_interrupt: Callable[[], bool] | None = None,
        judge_repeats: int = 1,
    ) -> None:
        if type(judge_repeats) is not int or not 1 <= judge_repeats <= 64:
            raise ValueError("judge_repeats must be an integer from 1 to 64")
        self.allow_time_travel = allow_time_travel
        self.limits = limits or RunLimits()
        self.runtime = runtime or ActorRuntime()
        self.event_sink = event_sink
        self.checkpoint_sink = checkpoint_sink
        self.contract = contract
        self.should_interrupt = should_interrupt
        self.judge_repeats = judge_repeats
        self.last_generator = None
        self.last_guide = None
        self.last_branches = None
        self.generator_history: list[tuple[str, Any]] = []
        self.guide_history: list[tuple[str, Any]] = []
        self.interrupted_call: dict[str, str] | None = None
        self.active_stage_transitions: tuple[StageTransition, ...] = ()

    def _checkpoint(self, judge: Judge, phase: str, **state: Any) -> None:
        if self.checkpoint_sink is not None:
            if self.active_stage_transitions:
                state["stage_transitions"] = self.active_stage_transitions
            self.checkpoint_sink({"phase": phase, **state}, self, judge)

    def _event(self, kind: str, **fields: Any) -> None:
        if self.event_sink is not None:
            self.event_sink({"kind": kind, **fields})

    def _actor_call(self, role: str, handle: Any, message: Any) -> Any:
        marker = {"role": role, "branch_id": handle.branch_id, "operation": "step"}
        # Keep the marker until the returned value is represented by a durable
        # protocol event.  A participant can return successfully and still
        # fail role/contract validation before such an event exists; that
        # completed ActorCall must remain distinguishable from transcript
        # tampering during actor replay.
        self.interrupted_call = marker
        return self.runtime.call(handle, message)

    def _judge_call(self, judge: Judge, target: Any, ideas: Any) -> tuple[IdeaVerdict, ...]:
        marker = {"role": "judge", "branch_id": "judge", "operation": "evaluate"}
        self.interrupted_call = marker
        return tuple(judge.evaluate(target, ideas))

    def _apply_stage_transitions(self, role: str, handle: Any) -> None:
        """Bring one reconstructed actor to the run's active policy stage.

        Historical Generator checkpoints may predate one or more promotions.
        Reapplying only the missing suffix makes checkout into an earlier stage
        behave like the current stage without rerunning any model service.
        """

        applied = tuple(
            call.message
            for call in handle.calls
            if isinstance(call.message, StageTransition)
        )
        expected_prefix = self.active_stage_transitions[: len(applied)]
        if applied != expected_prefix:
            raise ProtocolError("actor contains an incompatible stage-transition history")
        for transition in self.active_stage_transitions[len(applied) :]:
            before = handle.services.event_count()
            with handle.services.local_only():
                ready = self._actor_call(role, handle, transition)
            after = handle.services.event_count()
            if after != before:
                raise ProtocolError("stage transition may not call participant services")
            if not isinstance(ready, StageReady) or ready.stage != transition.to_stage:
                raise ProtocolError("participant returned an invalid stage-transition acknowledgement")
            self._event(
                "stage_transition",
                role=role,
                branch_id=handle.branch_id,
                from_stage=transition.from_stage,
                to_stage=transition.to_stage,
                transition=transition,
                ready=ready,
                handoff_sha256=message_hash(ready.handoff),
            )
            self.interrupted_call = None

    # Stored run roles still use the v1 "oracle" spelling.
    @property
    def last_oracle(self):
        return self.last_guide

    @property
    def oracle_history(self):
        return self.guide_history

    @legacy_keywords(oracle_factory='guide_factory')
    def run(
        self,
        *,
        generator_factory: ActorFactory,
        guide_factory: ActorFactory,
        target: Any,
        judge: Judge,
        seed: int = 0,
        run_id: str | None = None,
        resume_state: EngineResumeState | None = None,
        stage_transition: StageTransition | None = None,
    ) -> RunResult:
        run_id = run_id or uuid.uuid4().hex
        self.interrupted_call = None
        if resume_state is None:
            if stage_transition is not None:
                raise ProtocolError("a stage transition requires a resumed run")
            self.active_stage_transitions = ()
            branches = BranchStore(run_id=run_id, seed=seed)
            self.last_branches = branches
            self.last_generator = None
            self.last_guide = None
            self.generator_history = []
            self.guide_history = []
            self._event(
                "run_started",
                run_id=run_id,
                seed=seed,
                time_travel=self.allow_time_travel,
                judge_repeats=self.judge_repeats,
                protocol_event_schema=4,
                accounting_version=CURRENT_ACCOUNTING,
            )
            generator = self.runtime.start(generator_factory)
            self.last_generator = generator
            self.generator_history.append(("root", generator))
            guide = self.runtime.start(guide_factory)
            self.last_guide = guide
            self.guide_history.append(("root", guide))
            k = 0.0
            decisions = 0
            checkouts = 0
            submission_attempts = 0
            parent_question_id: str | None = None
            current: QuestionNode | None = None
            output: Any = None
            decision: Any = None
            submission: Submission | None = None
            submission_feedback: SubmissionFeedback | None = None
            submission_source_question_id: str | None = None
            submission_option_id: str | None = None
            submission_choice_bits = 0.0
            phase = "actors_ready"
            self._checkpoint(
                judge, phase, k=k, decisions=decisions, checkouts=checkouts,
                submission_attempts=submission_attempts,
                parent_question_id=parent_question_id, current_question_id=None,
            )
        else:
            self.active_stage_transitions = tuple(resume_state.stage_transitions)
            phase = resume_state.phase
            branches = resume_state.branches
            branches.run_id = run_id
            generator = resume_state.generator
            guide = resume_state.guide
            k = float(resume_state.k)
            decisions = int(resume_state.decisions)
            checkouts = int(resume_state.checkouts)
            submission_attempts = int(resume_state.submission_attempts)
            parent_question_id = resume_state.parent_question_id
            output = resume_state.output
            decision = resume_state.decision
            submission = resume_state.submission
            submission_feedback = resume_state.submission_feedback
            submission_source_question_id = resume_state.submission_source_question_id
            submission_option_id = resume_state.submission_option_id
            submission_choice_bits = float(resume_state.submission_choice_bits)
            current = (
                branches.nodes.get(resume_state.current_question_id)
                if resume_state.current_question_id is not None else None
            )
            self.last_branches = branches
            self.last_generator = generator
            self.last_guide = guide
            self.generator_history = resume_state.generator_history or [(generator.branch_id, generator)]
            self.guide_history = resume_state.guide_history or [(guide.branch_id, guide)]
            if resume_state.accounting_update is not None:
                update = resume_state.accounting_update
                previous_k = k
                branches.upgrade_accounting(update["node_costs"])
                k = update["path_k"]
                submission_choice_bits = (
                    update["submission_choice_bits"] if submission_source_question_id else 0.0
                )
                if current is not None:
                    current = branches.get(current.question_id)
                self._event(
                    "accounting_updated", from_version=LEGACY_ACCOUNTING,
                    accounting_version=CURRENT_ACCOUNTING, previous_path_k=previous_k,
                    path_k=k, node_costs_sha256=message_hash(update["node_costs"]),
                )
                self._enforce_information_limit(k)
            elif branches.accounting_version != CURRENT_ACCOUNTING:
                raise ProtocolError("legacy resume requires a verified accounting upgrade")
            if phase == "generator_output":
                # The restored ActorCall returned before the source checkpoint
                # but still has no matching question/submission event.  Carry
                # the same pending-call proof into the derived run until that
                # output is either committed or rejected by the contract.
                self.interrupted_call = {
                    "role": "generator",
                    "branch_id": generator.branch_id,
                    "operation": "step",
                }
            if stage_transition is not None:
                if (
                    self.active_stage_transitions
                    and self.active_stage_transitions[-1].to_stage
                    != stage_transition.from_stage
                ):
                    raise ProtocolError("new stage does not continue the active stage chain")
                self.active_stage_transitions = (
                    *self.active_stage_transitions,
                    stage_transition,
                )
                self._apply_stage_transitions("generator", generator)
                self._apply_stage_transitions("oracle", guide)
            if stage_transition is not None or resume_state.accounting_update is not None:
                self._checkpoint(
                    judge,
                    phase,
                    k=k,
                    decisions=decisions,
                    checkouts=checkouts,
                    submission_attempts=submission_attempts,
                    parent_question_id=parent_question_id,
                    current_question_id=(current.question_id if current is not None else None),
                    output=output,
                    decision=decision,
                    submission=submission,
                    submission_feedback=submission_feedback,
                    submission_source_question_id=submission_source_question_id,
                    submission_option_id=submission_option_id,
                    submission_choice_bits=submission_choice_bits,
                )
        while True:
            if self.should_interrupt is not None and self.should_interrupt():
                # A pause requested mid-phase (e.g. SIGTERM) surfaces here, at
                # the loop boundary, where no concurrent service batch is in
                # flight -- so the finalize path writes an untorn journal tail.
                raise KeyboardInterrupt("pause requested at loop boundary")
            if phase == "actors_ready":
                output = self._actor_call("generator", generator, None)
                phase = "generator_output"
                self._checkpoint(
                    judge, phase, k=k, decisions=decisions, checkouts=checkouts,
                    submission_attempts=submission_attempts,
                    parent_question_id=parent_question_id, current_question_id=None,
                    output=output,
                )

            if phase == "generator_output":
                if isinstance(output, Submission):
                    if submission_source_question_id is None:
                        raise ProtocolError(
                            "generator may submit only after the oracle selects a SubmitOption"
                        )
                    self.contract.validate_submission(output)
                    submission = output
                    submission_attempts += 1
                    self._enforce_limits(
                        k, decisions, checkouts, submission_attempts=submission_attempts
                    )
                    self._event(
                        "submission",
                        attempt=submission_attempts,
                        source_question_id=submission_source_question_id,
                        option_id=submission_option_id,
                        submission=submission,
                        path_k=k,
                    )
                    self.interrupted_call = None
                    phase = "before_judge"
                    self._checkpoint(
                        judge, phase, k=k, decisions=decisions, checkouts=checkouts,
                        submission_attempts=submission_attempts,
                        parent_question_id=parent_question_id, current_question_id=None,
                        submission=submission,
                        submission_source_question_id=submission_source_question_id,
                        submission_option_id=submission_option_id,
                        submission_choice_bits=submission_choice_bits,
                    )
                    continue
                if submission_source_question_id is not None:
                    raise ProtocolError(
                        "generator must return Submission after a selected SubmitOption"
                    )
                if not isinstance(output, Question):
                    raise ProtocolError("generator returned neither Question nor Submission")
                self.contract.validate_question(output)
                if len(branches.nodes) >= self.limits.max_questions:
                    raise ResourceLimitExceeded("question limit exhausted")
                if self._question_depth(branches, parent_question_id) > self.limits.max_depth:
                    raise ResourceLimitExceeded("question-depth limit exhausted")
                current = branches.add_question(
                    output,
                    checkpoint=self.runtime.checkpoint(generator),
                    path_k=k,
                    parent_question_id=parent_question_id,
                )
                self._event(
                    "question",
                    question_id=current.question_id,
                    parent_question_id=current.parent_question_id,
                    path_k=current.path_k,
                    integrity_hash=current.integrity_hash,
                    question=current.question,
                )
                self.interrupted_call = None
                phase = "await_oracle"
                self._checkpoint(
                    judge, phase, k=k, decisions=decisions, checkouts=checkouts,
                    submission_attempts=submission_attempts,
                    parent_question_id=parent_question_id,
                    current_question_id=current.question_id,
                )
                continue

            if phase == "await_oracle":
                if current is None:
                    raise ProtocolError("resume checkpoint has no current question")
                self._enforce_limits(
                    k,
                    decisions,
                    checkouts,
                    submission_attempts=submission_attempts,
                    before_guide_call=True,
                )
                decision = self._actor_call(
                    "oracle",
                    guide,
                    PresentedQuestion(
                        current.question_id, current.question
                    ),
                )
                decisions += 1
                self._event("oracle_decision", question_id=current.question_id, decision=decision)
                self.interrupted_call = None
                phase = "after_oracle"
                self._checkpoint(
                    judge, phase, k=k, decisions=decisions, checkouts=checkouts,
                    submission_attempts=submission_attempts,
                    parent_question_id=parent_question_id,
                    current_question_id=current.question_id, decision=decision,
                )
                continue

            if phase == "after_oracle":
                if current is None:
                    raise ProtocolError("resume checkpoint has no current question")
                if isinstance(decision, Checkout):
                    self.contract.validate_checkout(decision)
                    if not self.allow_time_travel:
                        raise InvalidCheckout("time travel is disabled for this run")
                    if checkouts >= self.limits.max_checkouts:
                        raise ResourceLimitExceeded("checkout limit exhausted")
                    eligible = branches.valid_checkout_targets(current)
                    if len(eligible) > self.limits.max_checkout_targets:
                        raise ResourceLimitExceeded("checkout-target limit exhausted")
                    target_candidate = branches.get(decision.question_id)
                    if target_candidate not in eligible:
                        raise InvalidCheckout("checkout target is not a resumable question")
                    if (
                        current.created_index - target_candidate.created_index
                        > self.limits.max_checkout_rewind
                    ):
                        raise ResourceLimitExceeded("checkout rewind limit exhausted")
                    preview = branches.preview_checkout(current, decision.question_id)
                    target_node = preview.target
                    branch_bits = preview.branch_bits
                    previous_generator = generator
                    candidate_generator = self.runtime.fork(
                        target_node.generator_checkpoint, preview.branch_id
                    )
                    next_k = target_node.path_k + branch_bits
                    try:
                        self._event(
                            "checkout",
                            source_question_id=current.question_id,
                            target_question_id=target_node.question_id,
                            branch_bits=branch_bits,
                            path_k=next_k,
                        )
                        branches.commit_checkout(preview, commit_branch=True)
                    except Exception:
                        self.runtime.close(candidate_generator)
                        raise
                    checkouts += 1
                    generator = candidate_generator
                    self._apply_stage_transitions("generator", generator)
                    self.generator_history.append((preview.branch_id, generator))
                    self.last_generator = generator
                    k = next_k
                    current = target_node
                    self.runtime.close(previous_generator)
                    phase = "await_oracle"
                    self._checkpoint(
                        judge, phase, k=k, decisions=decisions, checkouts=checkouts,
                        submission_attempts=submission_attempts,
                        parent_question_id=parent_question_id,
                        current_question_id=current.question_id,
                    )
                    continue

                if not isinstance(decision, Choice):
                    raise ProtocolError("oracle returned neither Choice nor Checkout")
                if decision.question_id not in (None, current.question_id):
                    raise ProtocolError("choice refers to a different question")
                resolved = self.contract.resolve_choice(decision, current.question)
                option = resolved.option
                choice_preview = branches.preview_choice(current, option.option_id)
                branch_bits = choice_preview.branch_bits
                choice_bits = resolved.information_bits
                selected_choice_bits = choice_bits + branch_bits
                next_k = k + selected_choice_bits
                # Refuse an unaffordable edge before recording or applying it.
                # This keeps the durable path at or below the configured cap and
                # leaves the after_oracle checkpoint resumable with a larger cap.
                self._enforce_information_limit(next_k)
                self._event(
                    "choice_cost",
                    question_id=current.question_id,
                    option_id=option.option_id,
                    probability=str(resolved.probability),
                    information_bits=choice_bits,
                    branch_bits=branch_bits,
                    continuation_index=choice_preview.continuation_index,
                    option_index=choice_preview.option_index,
                    accounting_version=branches.accounting_version,
                    path_k=next_k,
                )
                branches.commit_choice(choice_preview)
                k = next_k
                self._enforce_limits(
                    k, decisions, checkouts, submission_attempts=submission_attempts
                )
                if isinstance(option, SubmitOption):
                    submission_source_question_id = current.question_id
                    submission_option_id = option.option_id
                    submission_choice_bits = selected_choice_bits
                else:
                    submission_source_question_id = None
                    submission_option_id = None
                    submission_choice_bits = 0.0
                output = self._actor_call(
                    "generator", generator, resolved.generator_choice
                )
                parent_question_id = current.question_id
                current = None
                phase = "generator_output"
                self._checkpoint(
                    judge, phase, k=k, decisions=decisions, checkouts=checkouts,
                    submission_attempts=submission_attempts,
                    parent_question_id=parent_question_id, current_question_id=None,
                    output=output,
                    submission_source_question_id=submission_source_question_id,
                    submission_option_id=submission_option_id,
                    submission_choice_bits=submission_choice_bits,
                )
                continue

            if phase == "before_judge":
                if not isinstance(submission, Submission):
                    raise ProtocolError("resume checkpoint has no submission")
                if submission_source_question_id is None:
                    raise ProtocolError("resume checkpoint has no submission source")
                verdict_rounds = tuple(
                    self._judge_call(judge, target, submission.ideas)
                    for _ in range(self.judge_repeats)
                )
                outcome, verdicts = self.contract.score_repeated_submission(
                    submission,
                    verdict_rounds,
                    path_k=k,
                )
                self._event(
                    "submission_judged",
                    attempt=submission_attempts,
                    source_question_id=submission_source_question_id,
                    status=outcome.status,
                    matched_idea_ids=outcome.matched_idea_ids,
                    verdicts=verdicts,
                    verdict_rounds=verdict_rounds,
                    path_k=k,
                    passing_mass=str(outcome.passing_mass),
                    submission_bits=outcome.submission_bits,
                    judge_repeats=outcome.judge_repeats,
                    judge_passes=outcome.judge_passes,
                    judge_pass_rate=str(outcome.judge_pass_rate),
                    repeat_bits=outcome.repeat_bits,
                )
                self.interrupted_call = None
                if outcome.status == "pass":
                    self._enforce_limits(
                        outcome.k,
                        decisions,
                        checkouts,
                        submission_attempts=submission_attempts,
                    )
                    self.last_generator, self.last_guide = generator, guide
                    return self._finish(
                        run_id,
                        outcome,
                        verdicts,
                        decisions,
                        checkouts,
                        submission_attempts,
                        branches,
                    )

                # A rejected proposal has no pointer cost. Its SubmitOption is
                # part of the abandoned branch and is unwound by the mandatory
                # recovery checkout just like an ordinary abandoned choice.
                source_node = branches.get(submission_source_question_id)
                recovery_targets = (
                    branches.valid_checkout_targets(source_node, include_source=True)
                    if self.allow_time_travel
                    else (source_node,)
                )
                if len(recovery_targets) > self.limits.max_checkout_targets:
                    raise ResourceLimitExceeded("checkout-target limit exhausted")
                submission_feedback = SubmissionFeedback(
                    PresentedQuestion(source_node.question_id, source_node.question),
                    submission,
                    verdicts,
                    tuple(node.question_id for node in recovery_targets),
                )
                self.contract.validate_submission_feedback(submission_feedback)
                phase = "await_submission_recovery"
                self._checkpoint(
                    judge,
                    phase,
                    k=k,
                    decisions=decisions,
                    checkouts=checkouts,
                    submission_attempts=submission_attempts,
                    parent_question_id=parent_question_id,
                    current_question_id=None,
                    submission=submission,
                    submission_feedback=submission_feedback,
                    submission_source_question_id=submission_source_question_id,
                    submission_option_id=submission_option_id,
                    submission_choice_bits=submission_choice_bits,
                )
                continue

            if phase == "await_submission_recovery":
                if not isinstance(submission_feedback, SubmissionFeedback):
                    raise ProtocolError("resume checkpoint has no submission feedback")
                self._enforce_limits(
                    k,
                    decisions,
                    checkouts,
                    submission_attempts=submission_attempts,
                    before_guide_call=True,
                )
                decision = self._actor_call("oracle", guide, submission_feedback)
                decisions += 1
                self._event(
                    "oracle_decision",
                    context="submission_feedback",
                    question_id=submission_feedback.source.question_id,
                    decision=decision,
                )
                self.interrupted_call = None
                phase = "after_submission_recovery"
                self._checkpoint(
                    judge,
                    phase,
                    k=k,
                    decisions=decisions,
                    checkouts=checkouts,
                    submission_attempts=submission_attempts,
                    parent_question_id=parent_question_id,
                    current_question_id=None,
                    submission=submission,
                    submission_feedback=submission_feedback,
                    submission_source_question_id=submission_source_question_id,
                    submission_option_id=submission_option_id,
                    submission_choice_bits=submission_choice_bits,
                    decision=decision,
                )
                continue

            if phase == "after_submission_recovery":
                if not isinstance(submission_feedback, SubmissionFeedback):
                    raise ProtocolError("resume checkpoint has no submission feedback")
                if not isinstance(decision, Checkout):
                    raise ProtocolError(
                        "oracle must checkout after a rejected submission"
                    )
                self.contract.validate_checkout(decision)
                source_node = branches.get(submission_feedback.source.question_id)
                if decision.question_id not in submission_feedback.valid_checkout_question_ids:
                    raise InvalidCheckout("rejected-submission checkout target is not recoverable")
                if not self.allow_time_travel and decision.question_id != source_node.question_id:
                    raise InvalidCheckout("time travel is disabled for this run")
                if checkouts >= self.limits.max_checkouts:
                    raise ResourceLimitExceeded("checkout limit exhausted")
                target_candidate = branches.get(decision.question_id)
                if (
                    source_node.created_index - target_candidate.created_index
                    > self.limits.max_checkout_rewind
                ):
                    raise ResourceLimitExceeded("checkout rewind limit exhausted")
                preview = branches.preview_checkout(
                    source_node,
                    decision.question_id,
                    include_source=True,
                    allow_earlier=self.allow_time_travel,
                )
                target_node = preview.target
                branch_bits = preview.branch_bits
                previous_generator = generator
                candidate_generator = self.runtime.fork(
                    target_node.generator_checkpoint, preview.branch_id
                )
                next_k = target_node.path_k + branch_bits
                try:
                    self._event(
                        "checkout",
                        context="submission_recovery",
                        source_question_id=source_node.question_id,
                        target_question_id=target_node.question_id,
                        branch_bits=branch_bits,
                        path_k=next_k,
                    )
                    branches.commit_checkout(preview, commit_branch=True)
                except Exception:
                    self.runtime.close(candidate_generator)
                    raise
                checkouts += 1
                generator = candidate_generator
                self._apply_stage_transitions("generator", generator)
                self.generator_history.append((preview.branch_id, generator))
                self.last_generator = generator
                k = next_k
                current = target_node
                self.runtime.close(previous_generator)
                parent_question_id = target_node.parent_question_id
                submission = None
                submission_feedback = None
                submission_source_question_id = None
                submission_option_id = None
                submission_choice_bits = 0.0
                output = None
                decision = None
                phase = "await_oracle"
                self._checkpoint(
                    judge,
                    phase,
                    k=k,
                    decisions=decisions,
                    checkouts=checkouts,
                    submission_attempts=submission_attempts,
                    parent_question_id=parent_question_id,
                    current_question_id=current.question_id,
                )
                continue

            raise ProtocolError(f"unknown durable run phase {phase!r}")

    @staticmethod
    def _question_depth(branches: BranchStore, parent_question_id: str | None) -> int:
        depth = 1
        current = parent_question_id
        while current is not None:
            depth += 1
            current = branches.nodes[current].parent_question_id
        return depth

    def _enforce_limits(
        self,
        k: float,
        decisions: int,
        checkouts: int,
        *,
        submission_attempts: int,
        before_guide_call: bool = False,
    ) -> None:
        self._enforce_information_limit(k)
        if decisions > self.limits.max_guide_decisions or (
            before_guide_call and decisions >= self.limits.max_guide_decisions
        ):
            raise ResourceLimitExceeded("oracle-decision limit exhausted")
        if checkouts > self.limits.max_checkouts:
            raise ResourceLimitExceeded("checkout limit exhausted")
        if submission_attempts > self.limits.max_submission_attempts:
            raise ResourceLimitExceeded("submission-attempt limit exhausted")

    def _enforce_information_limit(self, k: float) -> None:
        if not math.isfinite(k) or k > self.limits.max_bits:
            raise ResourceLimitExceeded("information-cost limit exhausted")

    def _finish(
        self,
        run_id: str,
        outcome: Any,
        verdicts: tuple[IdeaVerdict, ...],
        decisions: int,
        checkouts: int,
        submission_attempts: int,
        branches: BranchStore,
    ) -> RunResult:
        result = RunResult(
            run_id,
            outcome.status,
            outcome.score,
            outcome.k,
            outcome.matched_idea_ids,
            verdicts,
            outcome.judge_repeats,
            outcome.judge_passes,
            float(outcome.judge_pass_rate),
            float(outcome.repeat_bits or 0.0),
            len(branches.nodes),
            decisions,
            checkouts,
            submission_attempts,
            branches,
        )
        event = {
            "status": result.status,
            "score": result.score,
            "k": result.k,
            "matched_idea_ids": result.matched_idea_ids,
            "verdicts": result.verdicts,
            "judge_repeats": result.judge_repeats,
            "judge_passes": result.judge_passes,
            "judge_pass_rate": result.judge_pass_rate,
            "repeat_bits": result.repeat_bits,
            "submission_attempts": result.submission_attempt_count,
        }
        if outcome.submission_bits is not None:
            event.update(
                passing_mass=str(outcome.passing_mass),
                submission_bits=outcome.submission_bits,
            )
        self._event("run_finished", **event)
        return result
