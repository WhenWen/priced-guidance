"""Command-line player workflow."""

from __future__ import annotations

from ._compat import legacy_keywords

import argparse
import copy
import hashlib
import signal
import json
import math
import shutil
import sqlite3
import sys
import uuid
import os
import re
from importlib.resources import files
from pathlib import Path
from typing import Any, Callable

from . import __version__
from .contract import IDEA_RECOVERY_V1
from .contract.messages import Idea, Question, StageTransition, SubmitOption
from .runtime.engine import ArenaRunner, RunLimits
from .errors import ArenaError, RecordedRunFailure
from .submission_io.environment import require_runner
from .evaluation.base import aggregate_repeated_verdicts
from .evaluation.research import ResearchJudge
from .evaluation.sample import SampleSubmissionJudge
from .evaluation.smoke import SmokeAnswerJudge
from .contract.validation import message_hash
from .runtime.providers import ModelProviderBackend
from .runtime.agent_cli import AgentCLIBackend
from .runtime.human_guide import HumanGuideBackend
from .replay.artifacts import ArtifactStore, hash_tree
from .replay.recorder import (
    RunRecorder,
    actor_replay,
    decode_service_tape,
    load_resume_state,
    protocol_replay,
    reconcile_provider_attempt_journal,
    resume_event_prefix,
    retry_interrupted_service_indexes,
    verify_hash_chain,
)
from .resources import arena_home
from .sampling import (
    _ProgressModelBackend,
    _draft_from_question,
    sample_ideas,
)
from .runtime.services import ServiceFactory, ServiceLimits
from .runtime.actor import ActorRuntime
from .submission_io.manifest import (
    STAGE_ORDER,
    build_dependency_paths,
    entrypoint_is_defined,
    load_manifest,
    stage_module_hashes,
    validate_entrypoint_sources,
)
from .runtime.subprocess_actor import SubprocessActorFactory
from .targets.loader import load_target_pack
from .trajectory_html import render_trajectory_html
from .verification import verify_installation
from .web import serve


_ENV_ASSIGNMENT = re.compile(r"^(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*)$")


def _load_env_file(path: str | Path) -> None:
    source = Path(path).expanduser().resolve()
    try:
        lines = source.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise ArenaError(f"could not read env file {source}") from exc
    for number, raw in enumerate(lines, 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        match = _ENV_ASSIGNMENT.fullmatch(line)
        if not match:
            raise ArenaError(f"env file has an invalid assignment on line {number}")
        name, value = match.groups()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        os.environ[name] = value


def _progress_event(event: dict[str, object]) -> None:
    kind = event.get("kind")
    if kind == "question":
        print(
            f"idea-arena: question={event.get('question_id')} K={float(event.get('path_k', 0.0)):.3f}",
            file=sys.stderr,
            flush=True,
        )
    elif kind == "oracle_decision":
        print(f"idea-arena: oracle decision for {event.get('question_id')}", file=sys.stderr, flush=True)
    elif kind == "submission":
        print("idea-arena: submission ready; starting judge", file=sys.stderr, flush=True)
    elif kind == "run_finished":
        print(
            f"idea-arena: finished status={event.get('status')} score={event.get('score')}",
            file=sys.stderr,
            flush=True,
        )


def _progress_service(role: str, event: object) -> None:
    metadata = getattr(event, "metadata", {}) or {}
    usage = metadata.get("usage") or {}
    error = getattr(event, "error", None)
    suffix = f" error={error}" if error else ""
    print(
        "idea-arena: "
        f"role={role} service={getattr(event, 'kind', 'unknown')} "
        f"latency={float(metadata.get('latency_s', 0.0)):.1f}s "
        f"cost=${float(usage.get('cost_usd', 0.0)):.4f}{suffix}",
        file=sys.stderr,
        flush=True,
    )


def _sampling_progress_event(event: dict[str, object]) -> None:
    kind = event.get("kind")
    if kind == "sampling_started":
        print(
            "idea-arena: sampling started "
            f"pair={event.get('pair')} seed={event.get('seed')}",
            file=sys.stderr,
            flush=True,
        )
    elif kind == "generator_step_started":
        suffix = (
            f" after option={event.get('after_option_id')}"
            if event.get("after_option_id") is not None
            else ""
        )
        print(
            f"idea-arena: generator step={event.get('step')} started{suffix}",
            file=sys.stderr,
            flush=True,
        )
    elif kind == "model_call_started":
        print(
            "idea-arena: generator model call="
            f"{event.get('call')} schema={event.get('schema_name')} started",
            file=sys.stderr,
            flush=True,
        )
    elif kind == "model_call_finished":
        suffix = (
            f" error={event.get('error')}" if event.get("error") else ""
        )
        print(
            "idea-arena: generator model call="
            f"{event.get('call')} finished "
            f"latency={float(event.get('latency_s') or 0.0):.1f}s "
            f"cost=${float(event.get('cost_usd') or 0.0):.4f}{suffix}",
            file=sys.stderr,
            flush=True,
        )
    elif kind == "question":
        print(
            "idea-arena: sampled question="
            f"{event.get('question_index')} options={event.get('option_count')} "
            f"submit_mass={event.get('submit_probability')}",
            file=sys.stderr,
            flush=True,
        )
    elif kind == "draft_changed":
        preview = " ".join(str(event.get("draft") or "").split())
        if len(preview) > 360:
            preview = preview[:357] + "..."
        print(
            "idea-arena: current draft "
            f"question={event.get('question_index')} chars={event.get('chars')} "
            f"preview={preview}",
            file=sys.stderr,
            flush=True,
        )
    elif kind == "choice":
        print(
            "idea-arena: random choice "
            f"question={event.get('question_index')} "
            f"option={event.get('option_id')} p={event.get('probability')} "
            f"submit={bool(event.get('submit'))}",
            file=sys.stderr,
            flush=True,
        )
    elif kind == "submission":
        print(
            f"idea-arena: submission ready ideas={event.get('idea_count')}",
            file=sys.stderr,
            flush=True,
        )
    elif kind == "sampling_failed":
        print(
            "idea-arena: sampling failed "
            f"error={event.get('error_type')} questions={event.get('questions')}",
            file=sys.stderr,
            flush=True,
        )


def _sample_protocol_progress(
    event: dict[str, object],
    event_sink: Callable[[dict[str, Any]], None] | None,
    question_indices: dict[str, int],
    submit_options: dict[tuple[str, str], bool],
    current_draft: list[str | None],
) -> None:
    """Translate persisted Arena events into the live sample progress surface."""

    def emit(payload: dict[str, Any]) -> None:
        if event_sink is not None:
            event_sink(payload)

    kind = event.get("kind")
    if kind == "question" and isinstance(event.get("question"), Question):
        question = event["question"]
        question_id = str(event["question_id"])
        question_index = len(question_indices) + 1
        question_indices[question_id] = question_index
        probabilities = IDEA_RECOVERY_V1.validate_question(question)
        total = sum(probabilities)
        submit_mass = sum(
            probability
            for option, probability in zip(question.options, probabilities, strict=True)
            if isinstance(option, SubmitOption)
        )
        for option in question.options:
            submit_options[(question_id, option.option_id)] = isinstance(
                option, SubmitOption
            )
        emit({
            "kind": "generator_step_finished",
            "step": question_index - 1,
            "output_type": "Question",
        })
        emit({
            "kind": "question",
            "question_index": question_index,
            "option_count": len(question.options),
            "submit_probability": (
                "0" if submit_mass == 0 else str(submit_mass / total)
            ),
        })
        draft = _draft_from_question(question)
        if draft is not None and draft != current_draft[0]:
            current_draft[0] = draft
            emit({
                "kind": "draft_changed",
                "question_index": question_index,
                "draft": draft,
                "chars": len(draft),
            })
    elif kind == "choice_cost":
        question_id = str(event.get("question_id") or "")
        option_id = str(event.get("option_id") or "")
        question_index = question_indices.get(question_id, len(question_indices))
        is_submit = submit_options.get((question_id, option_id), False)
        emit({
            "kind": "choice",
            "question_index": question_index,
            "option_id": option_id,
            "probability": str(event.get("probability")),
            "submit": is_submit,
        })
        emit({
            "kind": "generator_step_started",
            "step": question_index,
            "after_option_id": option_id,
        })
    elif kind == "submission":
        submission = event.get("submission")
        idea_count = len(submission.ideas) if hasattr(submission, "ideas") else 0
        emit({
            "kind": "generator_step_finished",
            "step": len(question_indices),
            "output_type": "Submission",
        })
        emit({
            "kind": "submission",
            "question_index": len(question_indices),
            "idea_count": idea_count,
        })


def _sample_artifacts(run_dir: Path) -> dict[str, Any]:
    """Derive the public sample result from the authoritative private trace."""

    records = verify_hash_chain(run_dir / "events.private.jsonl")
    question_indices: dict[str, int] = {}
    submit_options: dict[tuple[str, str], bool] = {}
    choices: list[dict[str, Any]] = []
    ideas: list[dict[str, Any]] | None = None
    for record in records:
        kind = record.get("kind")
        if kind == "question":
            question_id = str(record["question_id"])
            question_indices[question_id] = len(question_indices) + 1
            for option in (record.get("question") or {}).get("options") or []:
                submit_options[(question_id, str(option.get("option_id") or ""))] = (
                    option.get("kind") == "submit"
                )
        elif kind == "choice_cost":
            question_id = str(record.get("question_id") or "")
            option_id = str(record.get("option_id") or "")
            choices.append({
                "question_index": question_indices.get(
                    question_id, len(question_indices)
                ),
                "option_id": option_id,
                "probability": str(record.get("probability")),
                "submit": submit_options.get((question_id, option_id), False),
            })
        elif kind == "submission":
            ideas = list((record.get("submission") or {}).get("ideas") or [])
    if ideas is None:
        raise ArenaError("completed sample run has no recorded Submission")
    try:
        usage = json.loads((run_dir / "usage.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ArenaError("completed sample run has no valid usage record") from exc
    return {"choices": choices, "ideas": ideas, "usage": usage}


def _implementation_hash(*names: str) -> str:
    root = Path(__file__).resolve().parent
    digest = hashlib.sha256()
    for name in names:
        data = (root / name).read_bytes()
        digest.update(name.encode("utf-8") + b"\0" + data)
    return digest.hexdigest()


def _guide_agent_config(
    backend: str | None,
    *,
    model: str | None,
    executable: str | None,
    reasoning_effort: str,
    timeout_seconds: float,
    max_budget_usd_per_turn: float | None,
) -> dict[str, object] | None:
    if backend is None:
        return None
    if backend not in {"claude-code", "codex", "human"}:
        raise ArenaError("--oracle-agent must be claude-code, codex, or human")
    if timeout_seconds <= 0:
        raise ArenaError("--oracle-agent-timeout-seconds must be positive")
    if max_budget_usd_per_turn is not None and max_budget_usd_per_turn <= 0:
        raise ArenaError("--oracle-agent-max-budget-usd-per-turn must be positive")
    if backend == "human":
        return {
            "backend": "human",
            "model": "human",
            "executable": None,
            "reasoning_effort": reasoning_effort,
            # A person answers on human time; never let the CLI's 600s
            # default kill a turn they are still thinking about.
            "timeout_seconds": max(float(timeout_seconds), 86_400.0),
            "max_budget_usd_per_turn": None,
        }
    selected_model = (
        model
        or os.environ.get("IDEA_ARENA_GUIDE_AGENT_MODEL")
        or os.environ.get("IDEA_ARENA_ORACLE_AGENT_MODEL")
        or ("opus" if backend == "claude-code" else "gpt-5.6-sol")
    )
    requested_executable = (
        executable
        or os.environ.get("IDEA_ARENA_GUIDE_AGENT_EXECUTABLE")
        or os.environ.get("IDEA_ARENA_ORACLE_AGENT_EXECUTABLE")
        or ("claude" if backend == "claude-code" else "codex")
    )
    selected_executable = shutil.which(requested_executable)
    if selected_executable is None:
        raise ArenaError(
            f"could not find {backend} executable {requested_executable!r}; "
            "install it or pass --oracle-agent-executable"
        )
    return {
        "backend": backend,
        "model": selected_model,
        "executable": selected_executable,
        "reasoning_effort": reasoning_effort,
        "timeout_seconds": float(timeout_seconds),
        "max_budget_usd_per_turn": max_budget_usd_per_turn,
    }


def _codex_wants_anthropic_proxy(config: dict[str, object]) -> bool:
    model = str(config.get("model") or "")
    return str(config.get("backend")) == "codex" and (
        model.startswith("anthropic/") or model.startswith("claude")
    )


def _spawn_anthropic_responses_proxy(
    run_root: Path,
    *,
    log_name: str = "anthropic-adapter.requests.jsonl",
) -> str:
    """Start the per-run OpenAI-Responses -> Anthropic adapter; return its URL.

    The adapter is localhost-only and dies with this process (atexit); its
    audit log lands next to the run's other private journals.
    """
    import atexit
    import subprocess
    import tempfile
    import time as _time

    port_file = Path(tempfile.mkstemp(prefix="anthropic-adapter-", suffix=".port")[1])
    port_file.unlink(missing_ok=True)
    process = subprocess.Popen(
        [
            sys.executable,
            "-B",
            "-m",
            "tech_tree_arena.runtime.anthropic_responses_proxy",
            "--port-file",
            str(port_file),
            "--log-file",
            str(run_root / log_name),
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    atexit.register(process.terminate)
    deadline = _time.monotonic() + 10.0
    while _time.monotonic() < deadline:
        if process.poll() is not None:
            raise ArenaError("the Anthropic responses adapter exited during startup")
        try:
            port = int(port_file.read_text(encoding="ascii"))
            port_file.unlink(missing_ok=True)
            return f"http://127.0.0.1:{port}"
        except (OSError, ValueError):
            _time.sleep(0.05)
    process.terminate()
    raise ArenaError("the Anthropic responses adapter did not announce a port")


def _participant_actor_timeout(*, codex: bool = False) -> float:
    """Wall clock for one participant step.

    A slow third-party generator can spend minutes per call, and the eager
    dispatch builds every route's preview before the step returns, so the
    default ceiling is an operator knob rather than a constant.
    """

    default = 6000.0 if codex else 900.0
    try:
        value = float(os.environ.get("IDEA_ARENA_ACTOR_TIMEOUT_S", str(default)))
    except ValueError:
        return default
    return value if value > 0 else default


def _guide_actor_timeout(agent_config: dict[str, object] | None) -> float:
    """The oracle subprocess must outwait its agent backend.

    A human oracle blocks the participant inside agent_turn for as long as
    the person thinks; the actor's wall clock has to sit above the backend
    timeout or the arena kills the run mid-thought. An agent-CLI oracle is
    no different: raising its turn budget is meaningless while the actor
    wall stays at a flat default, because the wall kills the turn first.
    """
    if agent_config is None:
        return _participant_actor_timeout()
    if str(agent_config.get("backend")) == "human":
        return float(agent_config.get("timeout_seconds") or 86_400.0) + 900.0
    return max(
        _participant_actor_timeout(),
        float(agent_config.get("timeout_seconds") or 600.0) + 300.0,
    )


def _make_guide_agent_backend(
    config: dict[str, object] | None,
    working_directory: Path,
    *,
    fork_inherited_sessions: bool = False,
) -> AgentCLIBackend | HumanGuideBackend | None:
    if config is None:
        return None
    if str(config.get("backend")) == "human":
        return HumanGuideBackend(
            working_directory=working_directory / "human",
            timeout_seconds=float(config.get("timeout_seconds") or 86_400.0),
        )
    try:
        proxy_base_url = None
        if _codex_wants_anthropic_proxy(config):
            if not os.environ.get("ANTHROPIC_API_KEY"):
                raise ArenaError(
                    "codex with an Anthropic oracle model needs ANTHROPIC_API_KEY "
                    "in the environment (load it with --env-file)"
                )
            proxy_base_url = _spawn_anthropic_responses_proxy(working_directory.parent.parent)
        return AgentCLIBackend(
            str(config["backend"]),
            working_directory=working_directory,
            executable=str(config["executable"]),
            model=str(config["model"]),
            reasoning_effort=str(config.get("reasoning_effort") or "high"),
            timeout_seconds=float(config.get("timeout_seconds") or 600.0),
            max_budget_usd_per_turn=(
                float(config["max_budget_usd_per_turn"])
                if config.get("max_budget_usd_per_turn") is not None
                else None
            ),
            anthropic_proxy_base_url=proxy_base_url,
            fork_inherited_sessions=fork_inherited_sessions,
        )
    except (OSError, TypeError, ValueError, RuntimeError) as exc:
        raise ArenaError(str(exc)) from exc


@legacy_keywords(oracle_agent='guide_agent', oracle_agent_model='guide_agent_model', oracle_agent_executable='guide_agent_executable', oracle_agent_reasoning_effort='guide_agent_reasoning_effort', oracle_agent_timeout_seconds='guide_agent_timeout_seconds', oracle_agent_max_budget_usd_per_turn='guide_agent_max_budget_usd_per_turn')
def _run_submission(
    path: Path,
    target_pack: str | None,
    target_id: str | None,
    seed: int,
    *,
    runs_dir: Path | None = None,
    allow_time_travel: bool = True,
    judge_mode: str = "fmn",
    judge_repeats: int | None = None,
    fmn_m: int | None = None,
    fmn_n: int = 0,
    disclosure: str = "development",
    runner_profile: str = "local",
    progress: bool = False,
    max_cost_usd_per_role: float = 1000.0,
    max_information_bits: float = 1024.0,
    guide_agent: str | None = None,
    guide_agent_model: str | None = None,
    guide_agent_executable: str | None = None,
    guide_agent_reasoning_effort: str = "high",
    guide_agent_timeout_seconds: float = 600.0,
    guide_agent_max_budget_usd_per_turn: float | None = None,
    html_report_on_failure: bool = False,
    sample_mode: bool = False,
    sample_max_questions: int = 256,
    sample_model_name: str | None = None,
    sample_model_backend: Any | None = None,
    model_backends: dict[str, Any] | None = None,
    generator_codex: dict[str, Any] | None = None,
    generator_claude: dict[str, Any] | None = None,
    generator_memory: dict[str, Any] | None = None,
    generator_output_tokens: int | None = 128_000,
    generator_memory_target_chars: int | None = None,
    budget_ledger: str | Path | None = None,
    sample_public_resources: dict[str, Any] | None = None,
    sample_event_sink: Callable[[dict[str, Any]], None] | None = None,
) -> dict:
    _install_terminate_handler()
    from .runtime.memory_output_budget import MODELS as output_budget_models, configuration as output_configuration
    output_policy = None
    if generator_memory and generator_memory["model"] in output_budget_models and generator_output_tokens is not None:
        output_policy = output_configuration(generator_memory["model"], generator_output_tokens)
    if generator_memory and generator_memory.get("codex"):
        generator_codex = generator_memory["codex"]
    if generator_codex and generator_claude:
        raise ArenaError("Select only one native Generator backend")
    if max_cost_usd_per_role <= 0:
        raise ArenaError("--max-cost-usd-per-role must be positive")
    if not math.isfinite(max_information_bits) or max_information_bits <= 0:
        raise ArenaError("--max-information-bits must be positive and finite")
    if fmn_m is not None and fmn_m < 0:
        raise ArenaError("--fmn-m must be non-negative")
    if fmn_n < 0:
        raise ArenaError("--fmn-n must be non-negative")
    if fmn_m is not None and fmn_n > fmn_m:
        raise ArenaError("--fmn-n cannot exceed --fmn-m")
    if judge_mode != "fmn" and (fmn_m is not None or fmn_n != 0):
        raise ArenaError("--fmn-m/--fmn-n require --judge fmn")
    resolved_judge_repeats = (
        3 if judge_mode == "essence" else 1
    ) if judge_repeats is None else judge_repeats
    if (
        type(resolved_judge_repeats) is not int
        or not 1 <= resolved_judge_repeats <= 64
    ):
        raise ArenaError("--judge-repeats must be an integer from 1 to 64")
    if sample_max_questions <= 0:
        raise ArenaError("--max-questions must be positive")
    judge_config = {
        "mode": judge_mode,
        "m": fmn_m if judge_mode == "fmn" else None,
        "n": fmn_n if judge_mode == "fmn" else None,
        "repeats": resolved_judge_repeats,
        # Coarse-screen rejections carry a one-sentence reason. Recorded in
        # the manifest so replay rebuilds the same judge requests; absent on
        # older runs, which therefore replay without reasons, byte-identical.
        "coarse_reasons": True,
    }
    manifest = load_manifest(path)
    validate_entrypoint_sources(manifest)
    module_hashes = stage_module_hashes(manifest)
    active_stage = {
        "directional": "directional",
        "essence": "essence",
        "fmn": "strict",
    }[judge_mode]
    dependency_paths = build_dependency_paths(manifest.root)
    pack = load_target_pack(target_pack) if target_pack is not None else None
    if sample_mode:
        selected = None
        target: dict[str, Any] = {}
        public_resources = pack.public_resources() if pack is not None else {}
        public_resources.update(sample_public_resources or {})
        allow_time_travel = False
    else:
        if pack is None:
            raise ArenaError("evaluation runs require a target pack")
        require_runner(pack, runner_profile)
        ids = pack.target_ids()
        if not ids:
            raise ArenaError("target pack is empty")
        selected = target_id or ids[0]
        target = pack.load(selected)
        public_resources = pack.public_resources()
    run_id = uuid.uuid4().hex
    agent_config = (
        None
        if sample_mode
        else _guide_agent_config(
            guide_agent,
            model=guide_agent_model,
            executable=guide_agent_executable,
            reasoning_effort=guide_agent_reasoning_effort,
            timeout_seconds=guide_agent_timeout_seconds,
            max_budget_usd_per_turn=guide_agent_max_budget_usd_per_turn,
        )
    )
    guide_entrypoint = (
        "tech_tree_arena.sampling:ProbabilitySamplingGuide"
        if sample_mode
        else ("participant.guide:AgentGuide"
              if entrypoint_is_defined(manifest, "participant.guide:AgentGuide")
              else "participant.oracle:AgentOracle")
        if agent_config is not None
        else manifest.guide
    )
    if agent_config is not None and not entrypoint_is_defined(
        manifest, guide_entrypoint
    ):
        raise ArenaError(
            "--oracle-agent requires the submission to define "
            f"{guide_entrypoint!r} (an Oracle whose decisions go through "
            "services.agent_turn); this submission does not"
        )
    backends = {role: ModelProviderBackend() for role in ("generator", "oracle", "judge")}
    if model_backends is not None:
        unknown_roles = set(model_backends) - set(backends)
        if unknown_roles:
            raise ArenaError(
                "unknown model-backend roles: " + ", ".join(sorted(unknown_roles))
            )
        backends.update(model_backends)
    if sample_model_backend is not None:
        backends["generator"] = sample_model_backend
    generator_model = sample_model_name or os.environ.get(
        "IDEA_ARENA_GENERATOR_MODEL", "claude-fable-5-1" if generator_claude else "gpt-6-astra" if generator_codex else "gpt-5.4"
    )
    generator_resources = {
        **public_resources,
        "judge_mode": judge_mode,
        "fmn_m": fmn_m,
        "fmn_n": fmn_n,
        # Public arena configuration; the Generator may shape its question
        # menus around whether ordinary time travel exists (e.g. priced
        # abandon rows are only needed when it does not).
        "time_travel_enabled": allow_time_travel,
        "offline_smoke": pack is not None and pack.name == "smoke",
        "reference_offline_smoke": (
            pack is not None
            and pack.name == "smoke"
            and manifest.name == "reference-pair"
        ),
        **({"active_stage": active_stage} if module_hashes else {}),
    }
    public_resources = {
        **public_resources,
        # Public arena configuration; the participant may shape its question
        # menus around whether ordinary time travel exists (e.g. priced
        # abandon rows are only needed when it does not).
        "time_travel_enabled": allow_time_travel,
    }
    guide_resources = {
        **public_resources,
        "judge_mode": judge_mode,
        "fmn_m": fmn_m,
        "fmn_n": fmn_n,
        "time_travel_enabled": allow_time_travel,
        "offline_smoke": pack is not None and pack.name == "smoke",
        "oracle_agent_backend": agent_config.get("backend") if agent_config else None,
        **({"active_stage": active_stage} if module_hashes else {}),
    }
    model_mapping = {
        "generator": generator_model,
        "oracle": (
            "probability-rng"
            if sample_mode
            else f"{agent_config['backend']}/{agent_config['model']}"
            if agent_config
            else (os.environ.get("IDEA_ARENA_GUIDE_MODEL")
                  or os.environ.get("IDEA_ARENA_ORACLE_MODEL", "gpt-5.5"))
        ),
        "judge": (
            "sample-accept-all"
            if sample_mode
            else os.environ.get("IDEA_ARENA_JUDGE_MODEL", "gpt-5.5")
        ),
    }
    if generator_memory:
        if generator_claude:
            raise ArenaError("Common memory requires Codex subscription or native API, not Claude Code")
        generator_model = generator_memory["model"]
        model_mapping["generator"] = generator_model
        generator_resources["generator_memory_policy"] = {
            "deterministic_calls": True, "timeout": generator_memory["timeout"],
        }
    profile_hash = message_hash({
        "runner": "local-unverified",
        "disclosure": disclosure,
        "time_travel": allow_time_travel,
        "run_kind": "sample_ideas" if sample_mode else "evaluation",
        "judge": (
            "sample-accept-all"
            if sample_mode
            else "smoke-answer-probe"
            if pack is not None and pack.name == "smoke"
            else judge_config
        ),
        "models": model_mapping,
        "oracle_agent": agent_config,
        "oracle_entrypoint": guide_entrypoint,
        **({"generator_codex": generator_codex} if generator_codex else {}),
        **({"generator_claude": generator_claude} if generator_claude else {}),
        **({"generator_memory": generator_memory} if generator_memory else {}),
        **({"generator_output_budget": output_policy} if output_policy else {}),
    })
    runs_root = (runs_dir or arena_home() / "runs").resolve()
    artifacts = ArtifactStore(runs_root)
    if generator_codex:
        from .runtime.codex_generator import CodexGeneratorBackend
        if sample_model_backend is not None or model_backends and "generator" in model_backends:
            raise ArenaError("Cannot combine Codex Generator with another Generator backend")
        backends["generator"] = CodexGeneratorBackend(generator_codex, artifacts)
    if generator_claude:
        from .runtime.claude_generator import ClaudeGeneratorBackend
        if sample_model_backend is not None or model_backends and "generator" in model_backends:
            raise ArenaError("Cannot combine Claude Generator with another Generator backend")
        backends["generator"] = ClaudeGeneratorBackend(generator_claude, artifacts)
    if generator_memory:
        from .runtime.common_memory import CommonMemoryBackend
        backends["generator"] = CommonMemoryBackend(generator_memory, artifacts,
            transport=backends["generator"] if generator_codex else None)
        if generator_memory_target_chars is not None:
            from .runtime.memory_summary_target import TargetedMemoryBackend
            backends["generator"] = TargetedMemoryBackend(generator_memory, artifacts,
                ancestors=[], target=generator_memory_target_chars,
                transport=backends["generator"].transport)
    if output_policy:
        from .runtime.memory_output_budget import OutputBudgetBackend
        backends["generator"] = OutputBudgetBackend(backends["generator"], output_policy)
    submission_snapshot = artifacts.snapshot_tree(manifest.root)
    target_pack_snapshot = (
        None
        if sample_mode or pack is None
        else artifacts.snapshot_tree(pack.root)
    )
    dependency_snapshots = [artifacts.snapshot_tree(path) for path in dependency_paths]
    public_resources_ref = artifacts.put_json(public_resources)
    generator_resources_ref = artifacts.put_json(generator_resources)
    guide_resources_ref = artifacts.put_json(guide_resources)
    run_manifest: dict[str, Any] = {
        "run_kind": "sample_ideas" if sample_mode else "evaluation",
        "protocol": manifest.protocol,
        "protocol_version": IDEA_RECOVERY_V1.version,
        "protocol_sha256": _implementation_hash(
            "contract/messages.py",
            "contract/validation.py",
            "contract/pricing.py",
            "contract/scoring.py",
            "runtime/branch.py",
            "runtime/engine.py",
        ),
        "arena_version": __version__,
        "submission_name": manifest.name,
        "submission_version": manifest.version,
        "oracle_entrypoint": guide_entrypoint,
        "oracle_agent": agent_config,
        "submission_path": str(manifest.root),
        **({"generator_codex": generator_codex} if generator_codex else {}),
        **({"generator_claude": generator_claude} if generator_claude else {}),
        **({"generator_memory": generator_memory} if generator_memory else {}),
        **({"generator_output_budget": output_policy} if output_policy else {}),
        "submission_sha256": hash_tree(manifest.root),
        "submission_snapshot": submission_snapshot,
        **(
            {
                "active_stage": active_stage,
                "stage_module_hashes": module_hashes,
                "stage_order": list(STAGE_ORDER),
            }
            if module_hashes
            else {}
        ),
        "dependency_paths": [str(path) for path in dependency_paths],
        "dependency_snapshots": dependency_snapshots,
        "target_pack": pack.name if pack is not None else None,
        # Public resources are content-addressed artifacts, not Arena protocol
        # messages. Reuse the artifact digest so a valid large taxonomy is not
        # rejected by the 256 KiB participant message bound.
        "target_pack_public_sha256": public_resources_ref["sha256"],
        "target_id": selected,
        "target_sha256": message_hash(target),
        "judge": (
            "sample-accept-all"
            if sample_mode
            else "smoke-answer-probe"
            if pack is not None and pack.name == "smoke"
            else f"research-{judge_mode}"
        ),
        "judge_config": judge_config,
        "judge_sha256": _implementation_hash(
            "evaluation/sample.py"
            if sample_mode
            else "evaluation/smoke.py"
            if pack is not None and pack.name == "smoke"
            else "evaluation/research.py"
        ),
        "scorer": "idea-recovery-v1-score-v2-repeat-judge",
        "models": model_mapping,
        "generator_public_resources_ref": generator_resources_ref,
        "oracle_public_resources_ref": guide_resources_ref,
        "seed": seed,
        "runner": "local-unverified",
        "budgets": {
            "oracle_decisions": sample_max_questions if sample_mode else 256,
            "questions": sample_max_questions if sample_mode else 256,
            "submission_attempts": 256,
            "checkouts": 64,
            "checkout_targets": 128,
            "checkout_rewind": 256,
            "question_depth": sample_max_questions if sample_mode else 256,
            "information_bits": max_information_bits,
            "model_calls_per_role": 256,
            "model_tokens_per_role": 10000000,
            "model_cost_usd_per_role": max_cost_usd_per_role,
        },
        "evaluation_profile_sha256": profile_hash,
    }
    if not sample_mode and pack is not None:
        run_manifest.update({
            "target_pack_path": str(pack.root),
            "target_pack_sha256": hash_tree(pack.root),
            "target_pack_snapshot": target_pack_snapshot,
        })
    from .runtime.subscription_api_budget import setup as setup_run_budget
    if generator_memory and generator_memory_target_chars is not None:
        from .runtime.memory_summary_target import source_hash as summary_source_hash
        run_manifest['generator_memory_transition'] = {
            'kind': 'summary-target-v2', 'source_sha256': summary_source_hash(),
            'ancestors': [], 'target_ancestors': [], 'target_chars': generator_memory_target_chars}
    setup_run_budget(backends, run_manifest, runs_root, path=budget_ledger)
    recorder = RunRecorder(
        runs_root,
        run_id,
        run_manifest,
        disclosure=disclosure,
    )
    for role, backend in backends.items():
        if isinstance(backend, ModelProviderBackend):
            backend.set_attempt_sink(
                lambda attempt, role=role: recorder.record_provider_attempt(role, attempt)
            )
    sample_question_indices: dict[str, int] = {}
    sample_submit_options: dict[tuple[str, str], bool] = {}
    sample_current_draft: list[str | None] = [None]

    def record_event(event: dict[str, object]) -> None:
        recorder.record(event)
        if sample_mode:
            _sample_protocol_progress(
                event,
                sample_event_sink,
                sample_question_indices,
                sample_submit_options,
                sample_current_draft,
            )
        if progress:
            _progress_event(event)

    def service_sink(role: str):
        def sink(event: object) -> None:
            recorder.record_service(role, event)
            if progress:
                _progress_service(role, event)
        return sink

    service_limits = ServiceLimits(max_model_cost_usd=max_cost_usd_per_role)
    guide_agent_backend = (
        None
        if sample_mode
        else _make_guide_agent_backend(
            agent_config,
            recorder.root / "agent-workspace" / "oracle",
        )
    )
    if sample_mode:
        judge = SampleSubmissionJudge()
    elif pack is not None and pack.name == "smoke":
        judge = SmokeAnswerJudge()
    else:
        judge_services = ServiceFactory(
            seed=seed * 2 + 3,
            model_backend=backends["judge"],
            model_name=os.environ.get("IDEA_ARENA_JUDGE_MODEL", "gpt-5.5"),
            event_sink=service_sink("judge"),
            limits=service_limits,
        ).create()
        judge = ResearchJudge(
            judge_services,
            mode=judge_mode,
            fmn_m=fmn_m,
            fmn_n=fmn_n,
            coarse_reasons=bool(judge_config.get("coarse_reasons")),
        )
    guide_judge = _GuideJudgeHandle()
    guide_judge.bind(judge, target, repeats=resolved_judge_repeats)
    run_budgets = run_manifest["budgets"]
    runner = ArenaRunner(
        allow_time_travel=allow_time_travel,
        limits=RunLimits(
            max_questions=int(run_budgets["questions"]),
            max_guide_decisions=int(run_budgets["oracle_decisions"]),
            max_checkouts=int(run_budgets["checkouts"]),
            max_checkout_targets=int(run_budgets["checkout_targets"]),
            max_checkout_rewind=int(run_budgets["checkout_rewind"]),
            max_depth=int(run_budgets["question_depth"]),
            max_submission_attempts=int(run_budgets["submission_attempts"]),
            max_bits=float(run_budgets["information_bits"]),
        ),
        event_sink=record_event,
        checkpoint_sink=recorder.checkpoint,
        should_interrupt=_pause_requested,
        judge_repeats=resolved_judge_repeats,
    )
    global _runner_active
    _runner_active = True
    if sample_mode and sample_event_sink is not None:
        sample_event_sink({
            "kind": "sampling_started",
            "pair": manifest.name,
            "target_pack": pack.name if pack is not None else None,
            "seed": seed,
            "max_questions": sample_max_questions,
            "run_id": run_id,
            "run_dir": str(recorder.root),
        })
        sample_event_sink({
            "kind": "generator_step_started",
            "step": 0,
            "after_option_id": None,
        })
    generator_backend = backends["generator"]
    if sample_mode:
        generator_backend = _ProgressModelBackend(
            generator_backend, sample_event_sink
        )
    try:
        result = runner.run(
            generator_factory=SubprocessActorFactory(
                manifest.root,
                manifest.generator,
                service_factory=ServiceFactory(
                    seed=seed * 2 + 1,
                    model_backend=generator_backend,
                    model_name=generator_model,
                    reasoning_effort=(generator_memory or generator_claude or generator_codex or {}).get("reasoning_effort"),
                    public_resources=generator_resources,
                    event_sink=service_sink("generator"),
                    limits=service_limits,
                ),
                dependency_paths=dependency_paths,
                log_path=recorder.root / "logs" / "generator.log",
                sandbox_generator=bool(generator_memory or generator_codex or generator_claude),
                timeout_seconds=_participant_actor_timeout(codex=bool(generator_memory or generator_codex or generator_claude)),
            ),
            guide_factory=SubprocessActorFactory(
                manifest.root,
                guide_entrypoint,
                constructor_args=() if sample_mode else (target,),
                service_factory=ServiceFactory(
                    seed=seed * 2 + 2,
                    model_backend=backends["oracle"],
                    agent_backend=guide_agent_backend,
                    model_name=os.environ.get("IDEA_ARENA_ORACLE_MODEL", "gpt-5.5"),
                    public_resources=guide_resources,
                    event_sink=service_sink("oracle"),
                    judge_call=guide_judge,
                    limits=service_limits,
                ),
                dependency_paths=dependency_paths,
                log_path=recorder.root / "logs" / "oracle.log",
                timeout_seconds=_guide_actor_timeout(agent_config),
            ),
            target=target,
            judge=judge,
            seed=seed,
            run_id=run_id,
        )
    except (Exception, KeyboardInterrupt) as exc:  # noqa: BLE001
        _runner_active = False
        error_code = (
            "interrupted" if isinstance(exc, KeyboardInterrupt)
            else getattr(exc, "code", "match_failure")
        )
        record_event({
            "kind": "run_finished",
            "status": "error",
            "score": 0.0,
            "error_code": error_code,
            "error_type": getattr(exc, "error_type", None) or type(exc).__name__,
            "error_message": getattr(exc, "private_detail", None) or str(exc),
            "error_phase": getattr(exc, "phase", None),
            "interrupted_call": runner.interrupted_call,
        })
        _close_runner_handles(runner, backends["generator"] if generator_memory or generator_codex or generator_claude else None)
        recorder.finalize_failure(error_code, runner, judge)
        if sample_mode and sample_event_sink is not None:
            sample_event_sink({
                "kind": "sampling_failed",
                "error_type": getattr(exc, "error_type", None) or type(exc).__name__,
                "error_message": getattr(exc, "private_detail", None) or str(exc),
                "questions": len(sample_question_indices),
                "run_id": run_id,
                "run_dir": str(recorder.root),
            })
        if html_report_on_failure:
            failure_result: dict[str, object] = {
                "run_dir": str(recorder.root),
                "submission": manifest.name,
                "target_id": selected,
            }
            _attach_default_html_reports(failure_result)
            report = failure_result.get("trajectory_report")
            if report:
                print(f"idea-arena: trajectory report {report}", file=sys.stderr)
            elif failure_result.get("trajectory_report_error"):
                print(
                    "idea-arena: trajectory report failed: "
                    f"{failure_result['trajectory_report_error']}",
                    file=sys.stderr,
                )
        if sample_mode:
            raise RecordedRunFailure(
                "sample run failed",
                run_dir=str(recorder.root),
                error_code=str(error_code),
            ) from exc
        if isinstance(exc, ArenaError):
            raise
        raise ArenaError("match failed; inspect the private run record") from exc
    _runner_active = False
    _close_runner_handles(runner, backends["generator"] if generator_memory or generator_codex or generator_claude else None)
    recorder.finalize(result, runner, judge)
    output = {
        "run_kind": "sample_ideas" if sample_mode else "evaluation",
        "run_id": result.run_id,
        "seed": seed,
        "submission": manifest.name,
        "target_pack": pack.name if pack is not None else None,
        "target_id": selected,
        "status": result.status,
        "score": result.score,
        "K": result.k,
        "accounting_version": result.branch_store.accounting_version,
        "matched_idea_ids": list(result.matched_idea_ids),
        "questions": result.question_count,
        "oracle_decisions": result.guide_decision_count,
        "checkouts": result.checkout_count,
        "submission_attempts": result.submission_attempt_count,
        "judge_repeats": result.judge_repeats,
        "judge_passes": result.judge_passes,
        "judge_pass_rate": result.judge_pass_rate,
        "repeat_bits": result.repeat_bits,
        "run_dir": str(recorder.root),
        "evaluation_profile_sha256": profile_hash,
        "judge_config": judge_config,
    }
    if sample_mode:
        output.update(_sample_artifacts(recorder.root))
        if sample_mode and sample_event_sink is not None:
            sample_event_sink({
                "kind": "sampling_finished",
                "questions": result.question_count,
                "idea_count": len(output["ideas"]),
                "run_id": result.run_id,
                "run_dir": str(recorder.root),
                "usage": output["usage"],
            })
    return output


_terminate_delivered = False
_terminate_requested = False
_runner_active = False


def _terminate_as_interrupt(signum: int, frame: Any) -> None:
    """Route the first termination signal into a clean pause; ignore repeats.

    While the engine loop is active the signal only sets a flag: the loop
    raises KeyboardInterrupt at its next boundary, where no concurrent
    service batch is in flight, so the finalize path writes an untorn
    journal tail (raising mid-batch left a tail whose meters disagreed with
    the attempt journal and blocked every resume path). Outside the engine
    loop (setup, teardown) the signal raises immediately so the process
    still exits. Process supervisors often forward TERM more than once; a
    second delivery mid-finalize would abort the finalize it triggered, so
    only the first one acts.
    """

    global _terminate_delivered, _terminate_requested
    if _terminate_delivered:
        return
    _terminate_requested = True
    if _runner_active:
        return
    _terminate_delivered = True
    raise KeyboardInterrupt(f"terminated by signal {signum}")



class _GuideJudgeHandle:
    """Late-bound Arena Judge access for the Oracle.

    The Judge is Arena-owned and holds the target, so the Oracle reaches it
    through the Arena rather than grading with a model of its own: a preview
    the Oracle acts on has to come from the same ruler that scores the run.
    The resume path rebuilds the Oracle factory before it rebuilds the Judge,
    so the service closes over this holder instead of over a Judge.
    """

    def __init__(self) -> None:
        self.judge: Any = None
        self.target: Any = None
        self.repeats = 1

    def bind(self, judge: Any, target: Any, *, repeats: int = 1) -> None:
        self.judge = judge
        self.target = target
        self.repeats = repeats

    def __call__(self, raw_ideas: Any) -> Any:
        if self.judge is None:
            raise ArenaError("the Judge is not available to the Oracle")
        ideas = tuple(
            Idea(
                idea_id=str(item.get("idea_id")),
                content=item.get("content"),
                probability=item.get("probability", 1),
            )
            for item in (raw_ideas or ())
        )
        verdict_rounds = tuple(
            tuple(self.judge.evaluate(self.target, ideas))
            for _ in range(self.repeats)
        )
        verdicts = aggregate_repeated_verdicts(ideas, verdict_rounds)
        return [
            {
                "idea_id": verdict.idea_id,
                "passed": bool(verdict.passed),
                "private_reason": verdict.private_reason,
            }
            for verdict in verdicts
        ]


def _pause_requested() -> bool:
    global _terminate_delivered
    if _terminate_requested and not _terminate_delivered:
        _terminate_delivered = True
        return True
    return False


def _install_terminate_handler() -> None:
    global _terminate_delivered, _terminate_requested, _runner_active
    # A CLI process normally executes one run, but embedded callers and tests
    # may execute several sequentially.  Signal state belongs to one run only.
    _terminate_delivered = False
    _terminate_requested = False
    _runner_active = False
    try:
        signal.signal(signal.SIGTERM, _terminate_as_interrupt)
        signal.signal(signal.SIGINT, _terminate_as_interrupt)
    except ValueError:
        # Not the main thread (e.g. embedded callers); leave the default.
        pass


def _close_runner_handles(runner: ArenaRunner, codex_backend: Any | None = None) -> None:
    seen: set[int] = set()
    for history in (runner.generator_history, runner.guide_history):
        for _, handle in history:
            if id(handle) not in seen:
                seen.add(id(handle))
                runner.runtime.close(handle)
    for handle in (runner.last_generator, runner.last_guide):
        if handle is not None and id(handle) not in seen:
            seen.add(id(handle))
            runner.runtime.close(handle)
    close = getattr(codex_backend, "close", None)
    if callable(close):
        close()


@legacy_keywords(oracle_agent_override='guide_agent_override')
def _resume_submission(
    run_directory: Path,
    *,
    progress: bool = False,
    max_cost_usd_per_role: float | None = None,
    max_information_bits: float | None = None,
    generator_memory_chars: int | None = None,
    generator_memory_target_chars: int | None = None,
    generator_output_tokens: int | None = None,
    budget_ledger: str | Path | None = None,
    retry_interrupted_call: bool = False,
    discard_service_tail: bool = False,
    compatible_submission: str | Path | None = None,
    promote_judge: str | None = None,
    guide_agent_override: dict[str, object] | None = None,
    html_report_on_failure: bool = False,
    model_backend: Any | None = None,
    sample_event_sink: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, object]:
    _install_terminate_handler()
    source_root = run_directory.expanduser().resolve()
    from .runtime.refusal_recovery import require_no_provider_refusal
    require_no_provider_refusal(source_root)
    try:
        source_manifest = json.loads((source_root / "manifest.json").read_text(encoding="utf-8"))
        source_status = json.loads((source_root / "status.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ArenaError("resume source has no valid manifest or status") from exc
    promotion = promote_judge is not None
    if retry_interrupted_call and discard_service_tail:
        raise ArenaError(
            "--retry-interrupted-call and --discard-service-tail are exclusive"
        )
    if promotion and discard_service_tail:
        raise ArenaError("--discard-service-tail does not apply to promotions")
    if promotion and retry_interrupted_call:
        raise ArenaError("Judge promotion cannot retry an interrupted call")
    if promotion:
        if (
            source_status.get("status") != "completed"
            or source_status.get("result_status") != "pass"
        ):
            raise ArenaError("Judge promotion requires a completed passing source run")
        checkpoint_name = "promotion-checkpoint.private.json"
    else:
        checkpoint_name = "checkpoint.private.json"
    run_kind = str(source_manifest.get("run_kind") or "evaluation")
    if run_kind not in {"evaluation", "sample_ideas"}:
        raise ArenaError(f"unsupported run kind {run_kind!r}")
    sample_mode = run_kind == "sample_ideas"
    if not promotion and (
        source_status.get("status") == "completed" or not source_status.get("resumable")
    ):
        raise ArenaError("run is complete or has no resumable checkpoint")
    try:
        from .replay.recorder import read_resume_checkpoint
        source_checkpoint = read_resume_checkpoint(source_root, checkpoint_name)
        source_service_cursor = int(source_checkpoint.get("service_event_count", 0))
        source_provider_attempt_cursor = int(
            source_checkpoint.get("provider_attempt_event_count", 0)
        )
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ArenaError("resume source has no valid durable checkpoint") from exc
    source_services_path = source_root / "service-calls.private.jsonl"
    source_service_records = (
        verify_hash_chain(source_services_path, tolerate_truncated_tail=True)
        if source_services_path.is_file()
        else ()
    )
    if source_service_cursor < 0 or source_service_cursor > len(source_service_records):
        raise ArenaError("resume source has an invalid service cursor")
    source_provider_attempts_path = source_root / "provider-attempts.private.jsonl"
    source_provider_attempt_records = (
        verify_hash_chain(
            source_provider_attempts_path, tolerate_truncated_tail=True
        )
        if source_provider_attempts_path.is_file()
        else ()
    )
    if (
        source_provider_attempt_cursor < 0
        or source_provider_attempt_cursor > len(source_provider_attempt_records)
    ):
        raise ArenaError("resume source has an invalid provider-attempt cursor")
    if source_provider_attempt_records:
        try:
            # An uncaught provider failure legitimately leaves a fully
            # accounted non-ok attempt (or an unfinished tail) in the source
            # journal. ``--retry-interrupted-call`` exists to resume exactly
            # that state, so only that flag relaxes the completeness gate;
            # hash chains, binding, and frozen-rate cost checks still run.
            reconcile_provider_attempt_journal(
                source_provider_attempts_path,
                source_services_path,
                max_provider_records=(
                    source_provider_attempt_cursor
                    if discard_service_tail
                    else None
                ),
                max_service_records=(
                    source_service_cursor if discard_service_tail else None
                ),
                require_complete=not (retry_interrupted_call or discard_service_tail),
            )
        except Exception as exc:  # noqa: BLE001
            raise ArenaError("resume source provider-attempt accounting is invalid") from exc
    retry_service_indexes: tuple[int, ...] = ()
    if retry_interrupted_call:
        # The recovery policy deliberately declines deterministic service-tail
        # replay, so first require the source's complete actor/service audit to
        # prove that this really is one uncommitted provider failure.
        actor_replay(source_root)
        # Validate recovery before creating a derived run. An unsupported retry
        # must not leave an initializing record that looks like an audit failure.
        retry_service_indexes = retry_interrupted_service_indexes(
            source_root, source_checkpoint, source_service_records,
        )

    runs_root = source_root.parent
    artifacts = ArtifactStore(runs_root)
    source_submission_root = (
        artifacts.resolve_tree(source_manifest["submission_snapshot"])
        if source_manifest.get("submission_snapshot")
        else Path(source_manifest["submission_path"])
    )
    source_submission = load_manifest(source_submission_root)
    source_module_hashes = stage_module_hashes(source_submission)
    recorded_module_hashes = source_manifest.get("stage_module_hashes")
    if bool(recorded_module_hashes) != bool(source_module_hashes) or (
        source_module_hashes and recorded_module_hashes != source_module_hashes
    ):
        raise ArenaError("resume source stage-module hashes do not match its snapshot")
    submission_override: dict[str, object] | None = None
    replacement_module_hashes: dict[str, str] = source_module_hashes
    stage_transition: StageTransition | None = None
    if compatible_submission is not None:
        override_root = Path(compatible_submission).expanduser().resolve()
        override_manifest = load_manifest(override_root)
        compatibility_fields = ("name", "protocol", "generator", "oracle")
        mismatches = [
            field
            for field in compatibility_fields
            if getattr(override_manifest, field) != getattr(source_submission, field)
        ]
        if mismatches:
            raise ArenaError(
                "compatible submission changes immutable interface fields: "
                + ", ".join(mismatches)
            )
        replacement_module_hashes = stage_module_hashes(override_manifest)
        if bool(source_module_hashes) != bool(replacement_module_hashes):
            raise ArenaError(
                "compatible submission cannot add or remove the stage-module contract"
            )
        submission_root = override_manifest.root
        submission_override = {
            "source_sha256": source_manifest.get("submission_sha256"),
            "replacement_sha256": hash_tree(override_manifest.root),
            "replacement_path": str(override_manifest.root),
        }
    else:
        submission_root = source_submission_root
    manifest = load_manifest(submission_root)
    validate_entrypoint_sources(manifest)
    dependency_paths = (
        tuple(artifacts.resolve_tree(item) for item in source_manifest.get("dependency_snapshots", []))
        if source_manifest.get("dependency_snapshots") is not None
        else build_dependency_paths(manifest.root)
    )
    pack = None
    target: dict[str, Any] = {}
    if not sample_mode:
        target_pack_root = (
            artifacts.resolve_tree(source_manifest["target_pack_snapshot"])
            if source_manifest.get("target_pack_snapshot")
            else source_manifest.get("target_pack_path")
            or source_manifest["target_pack"]
        )
        pack = load_target_pack(target_pack_root)
        target = pack.load(str(source_manifest["target_id"]))
    if (
        compatible_submission is None
        and hash_tree(manifest.root) != source_manifest.get("submission_sha256")
    ):
        raise ArenaError("resume submission snapshot does not match the source run")
    if (
        not sample_mode
        and pack is not None
        and hash_tree(pack.root) != source_manifest.get("target_pack_sha256")
    ):
        raise ArenaError("resume target-pack snapshot does not match the source run")
    if message_hash(target) != source_manifest.get("target_sha256"):
        raise ArenaError("resume target does not match the source run")

    generator_resources = (
        artifacts.load_json(source_manifest["generator_public_resources_ref"])
        if source_manifest.get("generator_public_resources_ref")
        else source_manifest.get("generator_public_resources") or {}
    )
    guide_resources = (
        artifacts.load_json(source_manifest["oracle_public_resources_ref"])
        if source_manifest.get("oracle_public_resources_ref")
        else source_manifest.get("oracle_public_resources") or {}
    )
    source_judge_name = str(source_manifest.get("judge") or "")
    active_judge_mode = source_judge_name.removeprefix("research-")
    active_judge_config = dict(source_manifest.get("judge_config") or {})
    if promotion:
        if promote_judge not in {"essence", "strict"}:
            raise ArenaError(f"unsupported Judge promotion {promote_judge!r}")
        expected_source, active_judge_mode = {
            "essence": ("research-directional", "essence"),
            "strict": ("research-essence", "fmn"),
        }[promote_judge]
        if source_judge_name != expected_source:
            raise ArenaError(
                f"cannot promote {source_judge_name or 'unknown Judge'} to {promote_judge}"
            )
        active_judge_config = {
            "mode": active_judge_mode,
            "m": None,
            "n": 0,
            "repeats": 3 if active_judge_mode == "essence" else 1,
            "coarse_reasons": True,
        }
        if source_module_hashes:
            source_stage = str(source_manifest.get("active_stage") or "")
            target_stage = str(promote_judge)
            try:
                source_index = STAGE_ORDER.index(source_stage)
            except ValueError as exc:
                raise ArenaError("promotion source has no valid active stage") from exc
            if source_index + 1 >= len(STAGE_ORDER) or STAGE_ORDER[source_index + 1] != target_stage:
                raise ArenaError("Judge promotion does not match the next policy module")
            if compatible_submission is None:
                replacement_module_hashes = source_module_hashes
            frozen_groups = ("shared", *STAGE_ORDER[: source_index + 1])
            changed = [
                group
                for group in frozen_groups
                if replacement_module_hashes.get(group) != source_module_hashes.get(group)
            ]
            if changed:
                raise ArenaError(
                    "compatible submission changes frozen prefix modules: "
                    + ", ".join(changed)
                )
            stage_transition = StageTransition(source_stage, target_stage)
    if isinstance(source_manifest.get("generator_agent"), dict):
        # The agent-Generator lane was removed; its recorded turns cannot be
        # continued live. Replays of such runs still verify from the tape.
        raise ArenaError(
            "this build no longer supports agent Generators; the source run "
            "used one and cannot be resumed live"
        )
    raw_agent_config = source_manifest.get("oracle_agent")
    agent_config = raw_agent_config if isinstance(raw_agent_config, dict) else None
    if guide_agent_override is not None:
        if agent_config is None:
            raise ArenaError(
                "--oracle-agent override requires a source run that already "
                "used an agent oracle"
            )
        agent_config = guide_agent_override
    guide_entrypoint = str(source_manifest.get("oracle_entrypoint") or manifest.guide)
    models = source_manifest.get("models") or {}
    if guide_agent_override is not None:
        models = {
            **models,
            "oracle": f"{guide_agent_override['backend']}/{guide_agent_override['model']}",
        }
    seed = int(source_manifest["seed"])
    prefix = resume_event_prefix(source_root, checkpoint_name=checkpoint_name)
    if prefix[0].get("protocol_event_schema") != 4:
        raise ArenaError(
            "resume requires active-path protocol event schema 4; start a new run"
        )
    allow_time_travel = bool(prefix[0].get("time_travel"))
    disclosure = str(source_manifest.get("disclosure") or "development")
    budget = float(
        max_cost_usd_per_role
        if max_cost_usd_per_role is not None
        else (source_manifest.get("budgets") or {}).get("model_cost_usd_per_role", 1000.0)
    )
    if budget <= 0:
        raise ArenaError("--max-cost-usd-per-role must be positive")
    information_budget = float(
        max_information_bits
        if max_information_bits is not None
        else (source_manifest.get("budgets") or {}).get("information_bits", 1024.0)
    )
    if not math.isfinite(information_budget) or information_budget <= 0:
        raise ArenaError("--max-information-bits must be positive and finite")
    run_id = uuid.uuid4().hex
    inherited_manifest = {
        key: value for key, value in source_manifest.items()
        if key not in {"schema_version", "run_id", "disclosure"}
    }
    inherited_manifest["budgets"] = {
        **(source_manifest.get("budgets") or {}),
        "model_cost_usd_per_role": budget,
        "information_bits": information_budget,
    }
    inherited_manifest.update({
        "resumed_from": source_manifest["run_id"],
        "resume_checkpoint_phase": (source_checkpoint.get("engine") or {}).get(
            "phase"
        ),
    })
    if guide_agent_override is not None:
        # The fork's manifest must say who actually played oracle from here on.
        inherited_manifest.update({
            "oracle_agent": guide_agent_override,
            "models": models,
            "oracle_agent_overridden_from": (
                raw_agent_config.get("backend")
                if isinstance(raw_agent_config, dict)
                else None
            ),
        })
    if promotion:
        inherited_manifest.update({
            "judge": f"research-{active_judge_mode}",
            "judge_config": active_judge_config,
            "evaluation_profile_sha256": message_hash({
                "promoted_from": source_manifest.get("evaluation_profile_sha256"),
                "judge": active_judge_config,
            }),
            "promoted_from": source_manifest["run_id"],
            "promotion_source_judge": source_judge_name,
            "promotion_target_judge": f"research-{active_judge_mode}",
            "promotion_base_k": float(
                (source_checkpoint.get("engine") or {}).get("k", 0.0)
            ),
            "promotion_source_service_cursor": source_service_cursor,
        })
        if stage_transition is not None:
            inherited_manifest.update({
                "active_stage": stage_transition.to_stage,
                "stage_order": list(STAGE_ORDER),
                "stage_module_hashes": replacement_module_hashes,
                "stage_prefix_proof": {
                    "source_run_id": source_manifest["run_id"],
                    "frozen_through": stage_transition.from_stage,
                    "frozen_hashes": {
                        group: source_module_hashes[group]
                        for group in (
                            "shared",
                            *STAGE_ORDER[
                                : STAGE_ORDER.index(stage_transition.from_stage) + 1
                            ],
                        )
                    },
                    "activated_module": stage_transition.to_stage,
                    "activated_sha256": replacement_module_hashes[
                        stage_transition.to_stage
                    ],
                },
            })
    if submission_override is not None:
        replacement_snapshot = artifacts.snapshot_tree(manifest.root)
        inherited_manifest.update({
            "submission_path": str(manifest.root),
            "submission_sha256": submission_override["replacement_sha256"],
            "submission_snapshot": replacement_snapshot,
            "submission_version": manifest.version,
            "resume_compatible_submission": submission_override,
        })
        if replacement_module_hashes:
            # The recorded module hashes describe the snapshot this run
            # actually played. A promotion already rewrote them alongside its
            # stage transition; a plain compatible-submission resume must do
            # the same, or the fork's manifest disagrees with its own
            # snapshot and no later resume of it can validate.
            inherited_manifest["stage_module_hashes"] = replacement_module_hashes
    if discard_service_tail:
        inherited_manifest["resume_policy"] = "discard_service_tail"
    if retry_interrupted_call:
        inherited_manifest["resume_policy"] = "retry_interrupted_call"
        inherited_manifest["retry_interrupted_call_history"] = [
            *(source_manifest.get("retry_interrupted_call_history") or []),
            {
                "source_run_id": source_manifest["run_id"],
                "source_service_cursor": source_service_cursor,
                # Successful subcalls earlier in the interrupted actor turn
                # remain replayable. Only its terminal failed service is
                # abandoned and retried live.
                "abandoned_service_events": len(retry_service_indexes),
                "retry_scope": "service" if len(retry_service_indexes) == 1 else "uncommitted_generator_turn",
            },
        ]
    from .runtime.memory_output_budget import configuration as output_configuration, validate_policy
    output_policy = source_manifest.get("generator_output_budget")
    if output_policy:
        validate_policy(output_policy)
    if generator_output_tokens is not None:
        memory_model = (source_manifest.get("generator_memory") or {}).get("model")
        previous = (output_policy or {}).get("max_output_tokens", 50_000)
        if generator_output_tokens < previous:
            raise ArenaError("Continuation output allowance cannot decrease")
        try:
            output_policy = output_configuration(memory_model, generator_output_tokens)
        except ValueError as exc:
            raise ArenaError(str(exc)) from exc
    if output_policy:
        inherited_manifest["generator_output_budget"] = output_policy
    memory_transition = source_manifest.get("generator_memory_transition")
    if memory_transition or generator_memory_chars is not None or generator_memory_target_chars is not None:
        from .runtime.memory_policy_transition import source_hash, validate_ancestors
        from .runtime.memory_summary_target import source_hash as target_source_hash
        config = copy.deepcopy(source_manifest.get("generator_memory"))
        if not config:
            raise ArenaError("Summary cap override requires a common-memory source")
        old_target = (memory_transition or {}).get("target_chars")
        if memory_transition:
            expected_kind = "summary-target-v2" if old_target is not None else "summary-cap-increase-v1"
            expected_source = target_source_hash() if old_target is not None else source_hash()
            if memory_transition.get("kind") != expected_kind or memory_transition.get("source_sha256") != expected_source:
                raise ArenaError("Summary policy transition implementation changed")
        ancestors = copy.deepcopy((memory_transition or {}).get("ancestors", []))
        target_ancestors = copy.deepcopy((memory_transition or {}).get("target_ancestors", []))
        summary_target = generator_memory_target_chars if generator_memory_target_chars is not None else old_target
        if old_target is not None and (generator_memory_chars not in (None, config["memory_chars"]) or summary_target != old_target):
            target_ancestors.append({"configuration": copy.deepcopy(config), "target": old_target})
        if generator_memory_chars is not None and generator_memory_chars != config["memory_chars"]:
            if not config["memory_chars"] < generator_memory_chars <= 64000:
                raise ArenaError("Continuation summary cap must increase and be at most 64000")
            ancestors.append(copy.deepcopy(config))
            config["memory_chars"] = generator_memory_chars
        validate_ancestors(config, ancestors)
        inherited_manifest["generator_memory"] = config
        inherited_manifest["generator_memory_transition"] = {
            "kind": "summary-cap-increase-v1", "source_sha256": source_hash(),
            "ancestors": ancestors,
        }
        if summary_target is not None:
            if not 256 <= summary_target <= config["memory_chars"]:
                raise ArenaError("Summary target must be between 256 and the hard cap")
            inherited_manifest["generator_memory_transition"].update(
                kind="summary-target-v2", source_sha256=target_source_hash(),
                target_chars=summary_target, target_ancestors=target_ancestors)
    from .runtime.subscription_api_budget import setup as setup_run_budget
    setup_run_budget(None, inherited_manifest, runs_root, source_root=source_root, path=budget_ledger)
    recorder = RunRecorder(
        runs_root,
        run_id,
        inherited_manifest,
        disclosure=disclosure,
    )
    sample_question_indices: dict[str, int] = {}
    sample_submit_options: dict[tuple[str, str], bool] = {}
    sample_current_draft: list[str | None] = [None]
    for event in prefix:
        event = dict(event)
        if event.get("kind") == "run_started":
            event["run_id"] = run_id
            event["resumed_from"] = source_manifest["run_id"]
        recorder.record(event)
        if sample_mode and event.get("kind") == "question":
            question_id = str(event.get("question_id") or "")
            sample_question_indices[question_id] = len(sample_question_indices) + 1
            question_data = event.get("question") or {}
            if isinstance(question_data, dict):
                for option in question_data.get("options") or []:
                    if isinstance(option, dict):
                        sample_submit_options[
                            (question_id, str(option.get("option_id") or ""))
                        ] = option.get("kind") == "submit"
    copied_source_service_records = (
        source_service_records[:source_service_cursor]
        if promotion or discard_service_tail
        else source_service_records
    )
    for index, record in enumerate(copied_source_service_records):
        payload = {
            key: value
            for key, value in record.items()
            if key not in {"sequence", "previous_hash", "event_hash", "recorded_at"}
        }
        recorder.copy_service_record(
            payload,
            abandon_from_run=(
                str(source_manifest["run_id"])
                if retry_interrupted_call
                and index in retry_service_indexes
                else None
            ),
            source_sequence=(
                int(record["sequence"])
                if retry_interrupted_call and index >= source_service_cursor
                else None
            ),
        )
    copied_source_provider_attempt_records = (
        source_provider_attempt_records[:source_provider_attempt_cursor]
        if promotion or discard_service_tail
        else source_provider_attempt_records
    )
    for record in copied_source_provider_attempt_records:
        recorder.copy_provider_attempt_record({
            key: value
            for key, value in record.items()
            if key not in {"sequence", "previous_hash", "event_hash", "recorded_at"}
        })

    service_limits = ServiceLimits(max_model_cost_usd=budget)

    def record_event(event: dict[str, object]) -> None:
        recorder.record(event)
        if sample_mode:
            _sample_protocol_progress(
                event,
                sample_event_sink,
                sample_question_indices,
                sample_submit_options,
                sample_current_draft,
            )
        elif progress:
            _progress_event(event)

    def service_sink(role: str):
        def sink(event: object) -> None:
            recorder.record_service(role, event)
            if progress:
                _progress_service(role, event)
        return sink

    backends = {
        role: ModelProviderBackend(
            attempt_sink=lambda attempt, role=role: recorder.record_provider_attempt(
                role, attempt
            )
        )
        for role in ("generator", "oracle", "judge")
    }
    if model_backend is not None:
        backends["generator"] = model_backend
    generator_codex = source_manifest.get("generator_codex")
    generator_claude = source_manifest.get("generator_claude")
    generator_memory = inherited_manifest.get("generator_memory")
    if generator_codex:
        from .runtime.codex_generator import CodexGeneratorBackend
        if model_backend is not None:
            raise ArenaError("Cannot replace a recorded Codex Generator backend on resume")
        backends["generator"] = CodexGeneratorBackend(generator_codex, artifacts)
    if generator_claude:
        from .runtime.claude_generator import ClaudeGeneratorBackend
        if model_backend is not None:
            raise ArenaError("Cannot replace a recorded Claude Generator backend on resume")
        backends["generator"] = ClaudeGeneratorBackend(generator_claude, artifacts)
    if generator_memory:
        from .runtime.common_memory import CommonMemoryBackend
        if model_backend is not None:
            raise ArenaError("Cannot replace a recorded common-memory backend on resume")
        transition = inherited_manifest.get("generator_memory_transition")
        if transition and transition.get("target_chars") is not None:
            from .runtime.memory_summary_target import TargetedMemoryBackend
            backends["generator"] = TargetedMemoryBackend(generator_memory, artifacts,
                ancestors=transition["ancestors"], target=transition["target_chars"],
                target_ancestors=transition["target_ancestors"],
                transport=backends["generator"] if generator_codex else None)
        elif transition:
            from .runtime.memory_policy_transition import ContinuedMemoryBackend
            backends["generator"] = ContinuedMemoryBackend(generator_memory, artifacts,
                ancestors=transition["ancestors"],
                transport=backends["generator"] if generator_codex else None)
        else:
            backends["generator"] = CommonMemoryBackend(generator_memory, artifacts,
                transport=backends["generator"] if generator_codex else None)
    if output_policy:
        from .runtime.memory_output_budget import OutputBudgetBackend
        backends["generator"] = OutputBudgetBackend(backends["generator"], output_policy)
    from .runtime.subscription_api_budget import setup as setup_run_budget
    setup_run_budget(backends, inherited_manifest, runs_root, source_root=source_root, path=budget_ledger)
    generator_backend = backends["generator"]
    if sample_mode:
        generator_backend = _ProgressModelBackend(
            generator_backend, sample_event_sink
        )
    guide_agent_backend = (
        None
        if sample_mode
        else _make_guide_agent_backend(
            agent_config,
            recorder.root / "agent-workspace" / "oracle",
            # A resumed or promoted run inherits the source run's oracle
            # thread; fork it so the source stays resumable and concurrent
            # forks of one lineage stop colliding on codex's single-writer
            # thread store.
            fork_inherited_sessions=True,
        )
    )
    runtime = ActorRuntime()
    generator_factory = SubprocessActorFactory(
        manifest.root,
        manifest.generator,
        service_factory=ServiceFactory(
            seed=seed * 2 + 1,
            model_backend=generator_backend,
            model_name=models.get("generator"),
            reasoning_effort=(generator_memory or generator_claude or generator_codex or {}).get("reasoning_effort"),
            public_resources=generator_resources,
            event_sink=service_sink("generator"),
            limits=service_limits,
        ),
        dependency_paths=dependency_paths,
        log_path=recorder.root / "logs" / "generator.log",
        sandbox_generator=bool(generator_memory or generator_codex or generator_claude),
        timeout_seconds=_participant_actor_timeout(codex=bool(generator_memory or generator_codex or generator_claude)),
    )
    guide_judge = _GuideJudgeHandle()
    guide_factory = SubprocessActorFactory(
        manifest.root,
        guide_entrypoint,
        constructor_args=() if sample_mode else (target,),
        service_factory=ServiceFactory(
            seed=seed * 2 + 2,
            model_backend=backends["oracle"],
            agent_backend=guide_agent_backend,
            model_name=models.get("oracle"),
            public_resources=guide_resources,
            event_sink=service_sink("oracle"),
            judge_call=guide_judge,
            limits=service_limits,
        ),
        dependency_paths=dependency_paths,
        log_path=recorder.root / "logs" / "oracle.log",
        timeout_seconds=_guide_actor_timeout(agent_config),
    )
    resume_state, checkpoint, _ = load_resume_state(
        source_root,
        generator_factory=generator_factory,
        guide_factory=guide_factory,
        runtime=runtime,
        seed=seed,
        retry_interrupted_call=retry_interrupted_call,
        checkpoint_name=checkpoint_name,
        # A hard interrupt can tear the post-checkpoint tail (meters counted
        # for calls whose events never landed); --discard-service-tail drops
        # that tail wholesale and re-runs from the durable checkpoint live.
        discard_service_tail=promotion or discard_service_tail,
    )
    resume_state.branches.run_id = run_id
    if promotion:
        # Copied lineage events may contain an earlier accounting upgrade.
        # Bind this promotion's base to its own verified source checkpoint.
        from .replay.recorder import _atomic_json
        recorder.manifest["promotion_base_k"] = (
            resume_state.accounting_update["path_k"]
            if resume_state.accounting_update is not None else resume_state.k
        )
        if resume_state.accounting_update is not None:
            recorder.manifest["promotion_recorded_base_k"] = resume_state.k
        else:
            recorder.manifest.pop("promotion_recorded_base_k", None)
        _atomic_json(recorder.root / "manifest.json", recorder.manifest)
    # Establish a derived-run recovery boundary before any continuation call.
    # The source checkpoint has already been fully validated by
    # ``load_resume_state``; rebasing preserves its committed service cursor,
    # leaving any copied write-ahead suffix attributable to the pending call.
    recorder.seed_resume_checkpoint(checkpoint)

    if sample_mode:
        judge = SampleSubmissionJudge()
    elif pack is not None and pack.name == "smoke":
        judge = SmokeAnswerJudge()
    else:
        committed_judge_tape = decode_service_tape(
            checkpoint.get("resume_committed_judge_tape")
            or checkpoint.get("judge_service_tape")
            or []
        )
        judge_tape = decode_service_tape(checkpoint.get("resume_judge_tape") or [])
        judge_meter = dict(checkpoint.get("resume_judge_meter") or {})
        judge_usage = dict(checkpoint.get("resume_judge_usage") or {})
        backends["judge"].restore_usage(judge_usage)
        judge_services = ServiceFactory(
            seed=seed * 2 + 3,
            model_backend=backends["judge"],
            model_name=models.get("judge"),
            event_sink=service_sink("judge"),
            limits=service_limits,
            initial_model_calls=int(judge_meter.get("model_calls", 0)),
            initial_random_calls=int(judge_meter.get("random_calls", 0)),
        ).create(judge_tape, committed_tape=committed_judge_tape)
        judge_services.restore_state(
            checkpoint.get("resume_judge_state")
            or checkpoint.get("judge_service_state")
            or {}
        )
        judge = ResearchJudge(
            judge_services,
            mode=active_judge_mode,
            fmn_m=active_judge_config.get("m"),
            fmn_n=int(active_judge_config.get("n") or 0),
            coarse_reasons=bool(active_judge_config.get("coarse_reasons")),
        )

    active_judge_repeats = int(active_judge_config.get("repeats") or 1)
    if not 1 <= active_judge_repeats <= 64:
        raise ArenaError("recorded Judge repeat count must be from 1 to 64")
    guide_judge.bind(judge, target, repeats=active_judge_repeats)
    budgets = inherited_manifest.get("budgets") or {}
    runner = ArenaRunner(
        allow_time_travel=allow_time_travel,
        limits=(
            RunLimits(
                max_questions=int(budgets.get("questions", 256)),
                max_guide_decisions=int(budgets.get("oracle_decisions", 256)),
                max_checkouts=int(budgets.get("checkouts", 64)),
                max_checkout_targets=int(budgets.get("checkout_targets", 128)),
                max_checkout_rewind=int(budgets.get("checkout_rewind", 256)),
                max_depth=int(budgets.get("question_depth", 256)),
                max_submission_attempts=int(budgets.get("submission_attempts", 256)),
                max_bits=float(budgets.get("information_bits", 1024.0)),
            )
            if sample_mode
            else RunLimits(
                max_questions=int(budgets.get("questions", 256)),
                max_guide_decisions=int(budgets.get("oracle_decisions", 256)),
                max_checkouts=int(budgets.get("checkouts", 64)),
                max_checkout_targets=int(budgets.get("checkout_targets", 128)),
                max_checkout_rewind=int(budgets.get("checkout_rewind", 256)),
                max_depth=int(budgets.get("question_depth", 256)),
                max_submission_attempts=int(budgets.get("submission_attempts", 256)),
                max_bits=float(budgets.get("information_bits", 1024.0)),
            )
        ),
        runtime=runtime,
        event_sink=record_event,
        checkpoint_sink=recorder.checkpoint,
        should_interrupt=_pause_requested,
        judge_repeats=active_judge_repeats,
    )
    global _runner_active
    _runner_active = True
    if sample_mode and sample_event_sink is not None:
        sample_event_sink({
            "kind": "sampling_started",
            "pair": manifest.name,
            "target_pack": source_manifest.get("target_pack"),
            "seed": seed,
            "max_questions": budgets.get("questions", 256),
            "run_id": run_id,
            "run_dir": str(recorder.root),
            "resumed_from": source_manifest["run_id"],
        })
        sample_event_sink({
            "kind": "generator_step_started",
            "step": len(sample_question_indices),
            "after_option_id": None,
        })
    try:
        result = runner.run(
            generator_factory=generator_factory,
            guide_factory=guide_factory,
            target=target,
            judge=judge,
            seed=seed,
            run_id=run_id,
            resume_state=resume_state,
            stage_transition=stage_transition,
        )
    except (Exception, KeyboardInterrupt) as exc:  # noqa: BLE001
        _runner_active = False
        error_code = (
            "interrupted" if isinstance(exc, KeyboardInterrupt)
            else getattr(exc, "code", "match_failure")
        )
        record_event({
            "kind": "run_finished",
            "status": "error",
            "score": 0.0,
            "error_code": error_code,
            "error_type": getattr(exc, "error_type", None) or type(exc).__name__,
            "error_message": getattr(exc, "private_detail", None) or str(exc),
            "error_phase": getattr(exc, "phase", None),
            "interrupted_call": runner.interrupted_call,
        })
        _close_runner_handles(runner, backends["generator"] if generator_memory or generator_codex or generator_claude else None)
        recorder.finalize_failure(error_code, runner, judge)
        if sample_mode and sample_event_sink is not None:
            sample_event_sink({
                "kind": "sampling_failed",
                "error_type": getattr(exc, "error_type", None) or type(exc).__name__,
                "error_message": getattr(exc, "private_detail", None) or str(exc),
                "questions": len(sample_question_indices),
                "run_id": run_id,
                "run_dir": str(recorder.root),
                "resumed_from": source_manifest["run_id"],
            })
        if html_report_on_failure:
            failure_result: dict[str, object] = {
                "run_dir": str(recorder.root),
                "submission": manifest.name,
                "target_id": source_manifest["target_id"],
            }
            _attach_default_html_reports(failure_result)
            report = failure_result.get("trajectory_report")
            if report:
                print(f"idea-arena: trajectory report {report}", file=sys.stderr)
            elif failure_result.get("trajectory_report_error"):
                print(
                    "idea-arena: trajectory report failed: "
                    f"{failure_result['trajectory_report_error']}",
                    file=sys.stderr,
                )
        if sample_mode:
            raise RecordedRunFailure(
                "resumed sample run failed",
                run_dir=str(recorder.root),
                error_code=str(error_code),
            ) from exc
        if isinstance(exc, ArenaError):
            raise
        raise ArenaError("resumed match failed; inspect the private run record") from exc
    _runner_active = False
    _close_runner_handles(runner, backends["generator"] if generator_memory or generator_codex or generator_claude else None)
    recorder.finalize(result, runner, judge)
    output: dict[str, object] = {
        "run_kind": run_kind,
        "run_id": result.run_id,
        "seed": seed,
        "resumed_from": source_manifest["run_id"],
        "submission": manifest.name,
        "target_pack": (
            pack.name if pack is not None else source_manifest.get("target_pack")
        ),
        "target_id": source_manifest["target_id"],
        "status": result.status,
        "score": result.score,
        "K": result.k,
        "accounting_version": result.branch_store.accounting_version,
        "matched_idea_ids": list(result.matched_idea_ids),
        "questions": result.question_count,
        "oracle_decisions": result.guide_decision_count,
        "checkouts": result.checkout_count,
        "submission_attempts": result.submission_attempt_count,
        "judge_repeats": result.judge_repeats,
        "judge_passes": result.judge_passes,
        "judge_pass_rate": result.judge_pass_rate,
        "repeat_bits": result.repeat_bits,
        "run_dir": str(recorder.root),
        "evaluation_profile_sha256": inherited_manifest.get(
            "evaluation_profile_sha256"
        ),
    }
    if promotion:
        output.update({
            "promoted_from": source_manifest["run_id"],
            "promotion_source_judge": source_judge_name,
            "promotion_target_judge": f"research-{active_judge_mode}",
            "promotion_base_K": float(
                recorder.manifest["promotion_base_k"]
            ),
        })
    if sample_mode:
        output.update(_sample_artifacts(recorder.root))
        if sample_mode and sample_event_sink is not None:
            sample_event_sink({
                "kind": "sampling_finished",
                "questions": result.question_count,
                "idea_count": len(output["ideas"]),
                "run_id": result.run_id,
                "run_dir": str(recorder.root),
                "resumed_from": source_manifest["run_id"],
                "usage": output["usage"],
            })
    return output


def _cmd_doctor(args: argparse.Namespace) -> int:
    packs = {}
    for name in ("smoke", "development40", "test87"):
        pack = load_target_pack(name)
        target_ids = pack.target_ids()
        for target_id in target_ids:
            pack.load(target_id)
        packs[name] = len(target_ids)
    result: dict[str, object] = {
        "status": "ok",
        "target_packs": packs,
        "contract": verify_installation(),
    }
    if args.live:
        from .runtime.provider_client import structured, usage_totals

        schema = {
            "type": "object",
            "properties": {"ok": {"type": "boolean"}},
            "required": ["ok"],
            "additionalProperties": False,
        }
        live = {}
        models = {
            os.environ.get("IDEA_ARENA_GENERATOR_MODEL", "gpt-5.4"),
            os.environ.get("IDEA_ARENA_ORACLE_MODEL", "gpt-5.5"),
            os.environ.get("IDEA_ARENA_JUDGE_MODEL", "gpt-5.5"),
        }
        for model in sorted(models):
            response = structured(
                model=model,
                developer="Return the requested JSON.",
                user="Set ok to true.",
                schema=schema,
                schema_name="doctor_live_probe",
                max_output_tokens=128,
                reasoning_effort="low",
                timeout=60,
            )
            live[model] = {"status": "ok", "parsed": response == {"ok": True}}
        result["live_models"] = live
        result["live_usage"] = usage_totals()
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


def _cmd_validate(args: argparse.Namespace) -> int:
    result = _run_submission(Path(args.path), "smoke", None, 1)
    if result["status"] != "pass":
        raise ArenaError("submission failed the offline smoke match")
    print(json.dumps({"status": "valid", "smoke": result}, indent=2, sort_keys=True))
    return 0


def _attach_default_html_reports(result: dict[str, object]) -> None:
    """Best-effort public and private reports for interactive run commands."""
    run_dir = Path(str(result["run_dir"])).expanduser().resolve()
    title = "Idea Arena · {submission} · {target}".format(
        submission=result.get("submission") or "submission",
        target=result.get("target_id") or "target",
    )
    reports = (
        ("trajectory_report", "trajectory-report.html", False),
        ("trajectory_report_private", "trajectory-report.private.html", True),
    )
    for result_key, filename, private in reports:
        try:
            output = render_trajectory_html(
                [run_dir],
                run_dir / filename,
                private=private,
                title=title,
            )
        except (ArenaError, OSError) as exc:
            # Rendering is observability, not part of the scored protocol.
            # Keep failures independent so one report cannot suppress the other.
            result[f"{result_key}_error"] = str(exc)
        else:
            result[result_key] = str(output)


def _cmd_run(args: argparse.Namespace) -> int:
    result = _run_submission(
        Path(args.path),
        args.target_pack,
        args.target,
        args.seed,
        runs_dir=Path(args.runs_dir) if args.runs_dir else None,
        allow_time_travel=not args.no_time_travel,
        judge_mode=args.judge,
        judge_repeats=args.judge_repeats,
        fmn_m=args.fmn_m,
        fmn_n=args.fmn_n,
        disclosure=args.disclosure,
        runner_profile=args.runner,
        progress=args.progress,
        max_cost_usd_per_role=args.max_cost_usd_per_role,
        max_information_bits=args.max_information_bits,
        guide_agent=args.guide_agent,
        guide_agent_model=args.guide_agent_model,
        guide_agent_executable=args.guide_agent_executable,
        guide_agent_reasoning_effort=args.guide_agent_reasoning_effort,
        guide_agent_timeout_seconds=args.guide_agent_timeout_seconds,
        guide_agent_max_budget_usd_per_turn=args.guide_agent_max_budget_usd_per_turn,
        html_report_on_failure=not args.no_html_report,
        generator_codex=_generator_codex_config(args),
        generator_claude=_generator_claude_config(args),
        generator_memory=_generator_memory_config(args),
        generator_output_tokens=args.generator_output_tokens,
        budget_ledger=args.budget_ledger,
        generator_memory_target_chars=args.generator_memory_target_chars,
        sample_model_name=args.generator_model,
    )
    if not args.no_html_report:
        _attach_default_html_reports(result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "pass" else 2


def _cmd_sample_ideas(args: argparse.Namespace) -> int:
    result = sample_ideas(
        Path(args.path),
        target_pack=args.target_pack,
        seed=args.seed,
        max_questions=args.max_questions,
        max_model_cost_usd=args.max_model_cost_usd,
        model_name=args.generator_model,
        generator_codex=_generator_codex_config(args),
        generator_claude=_generator_claude_config(args),
        generator_memory=_generator_memory_config(args),
        generator_output_tokens=args.generator_output_tokens,
        budget_ledger=args.budget_ledger,
        event_sink=None if args.quiet else _sampling_progress_event,
        runs_dir=Path(args.runs_dir) if args.runs_dir else None,
    )
    print(json.dumps(result.to_dict(), indent=2, sort_keys=True))
    return 0


def _cmd_new(args: argparse.Namespace) -> int:
    destination = Path(args.path).resolve()
    if destination.exists():
        raise ArenaError("destination already exists")
    template = "reference_pair" if args.reference else "minimal_pair"
    packaged = Path(str(files("tech_tree_arena.data").joinpath("submission_templates", template)))
    source_root = Path(__file__).resolve().parents[2] / "submissions"
    checkout = (
        source_root / "reference_pair"
        if args.reference
        else source_root / "examples" / "minimal_pair"
    )
    source = packaged if packaged.is_dir() else checkout
    if not source.is_dir():
        raise ArenaError(f"submission template {template!r} is not installed")
    shutil.copytree(source, destination)
    print(destination)
    return 0


def _cmd_replay(args: argparse.Namespace) -> int:
    result = actor_replay(args.run_dir) if args.actor else protocol_replay(args.run_dir)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


def _cmd_reprice(args: argparse.Namespace) -> int:
    from .replay.accounting import migrate_tree
    report = migrate_tree(args.run_dir, write=args.write)
    if args.report:
        from .replay.recorder import _atomic_json
        _atomic_json(Path(args.report).resolve(), report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 1 if report["errors"] else 0


def _cmd_resume(args: argparse.Namespace) -> int:
    try:
        resume_manifest = json.loads(
            (Path(args.run_dir).expanduser().resolve() / "manifest.json").read_text(
                encoding="utf-8"
            )
        )
    except (OSError, json.JSONDecodeError):
        resume_manifest = {}
    sample_progress = (
        _sampling_progress_event
        if args.progress and resume_manifest.get("run_kind") == "sample_ideas"
        else None
    )
    guide_agent_override = _guide_agent_config(
        args.guide_agent,
        model=args.guide_agent_model,
        executable=args.guide_agent_executable,
        reasoning_effort=args.guide_agent_reasoning_effort,
        timeout_seconds=args.guide_agent_timeout_seconds,
        max_budget_usd_per_turn=args.guide_agent_max_budget_usd_per_turn,
    )
    result = _resume_submission(
        Path(args.run_dir),
        progress=args.progress,
        max_cost_usd_per_role=args.max_cost_usd_per_role,
        max_information_bits=args.max_information_bits,
        retry_interrupted_call=args.retry_interrupted_call,
        generator_memory_chars=args.generator_memory_chars,
        generator_memory_target_chars=args.generator_memory_target_chars,
        generator_output_tokens=args.generator_output_tokens,
        budget_ledger=args.budget_ledger,
        discard_service_tail=args.discard_service_tail,
        compatible_submission=args.compatible_submission,
        promote_judge=args.promote_judge,
        guide_agent_override=guide_agent_override,
        html_report_on_failure=not args.no_html_report,
        sample_event_sink=sample_progress,
    )
    if not args.no_html_report:
        _attach_default_html_reports(result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "pass" else 2


def _cmd_status(args: argparse.Namespace) -> int:
    root = Path(args.run_dir).expanduser().resolve()
    try:
        status = json.loads((root / "status.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ArenaError("run has no valid status record") from exc
    score_path = root / "score.json"
    if status.get("status") != "running" and score_path.is_file():
        canonical = json.loads(score_path.read_text(encoding="utf-8"))
        if "repriced_from" in canonical:
            protocol_replay(root)
            status["recorded_K"] = status.get("K")
            status.update({key: canonical[key] for key in ("K", "score", "accounting_version")})
    usage_path = root / "usage.json"
    if usage_path.is_file():
        status["usage"] = json.loads(usage_path.read_text(encoding="utf-8"))
    service_path = root / "service-calls.private.jsonl"
    if service_path.is_file():
        live_usage: dict[str, dict[str, float | int]] = {}
        last_service_at: float | None = None
        for record in verify_hash_chain(
            service_path,
            tolerate_truncated_tail=status.get("status") == "running",
        ):
            last_service_at = float(record.get("recorded_at", 0.0))
            role = str(record.get("role") or "unknown")
            service = record.get("service") or {}
            row = live_usage.setdefault(role, {"attempts": 0, "errors": 0, "cost_usd": 0.0})
            row["attempts"] += 1
            row["errors"] += int(bool(service.get("error")))
            metadata = service.get("metadata") or {}
            row["cost_usd"] += float((metadata.get("usage") or {}).get("cost_usd", 0.0))
        status["live_service_usage"] = live_usage
        status["last_service_at"] = last_service_at
    if status.get("status") == "running":
        try:
            os.kill(int(status["pid"]), 0)
        except (KeyError, TypeError, ValueError, ProcessLookupError):
            process_alive = False
        except PermissionError:
            process_alive = True
        else:
            process_alive = True
        status["process_alive"] = process_alive
        if not process_alive and status.get("resumable"):
            status["effective_status"] = "interrupted"
    status["run_dir"] = str(root)
    print(json.dumps(status, indent=2, sort_keys=True))
    return 0


def _leaderboard_path() -> Path:
    path = arena_home() / "arena.sqlite3"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _record_tournament(row: dict) -> None:
    with sqlite3.connect(_leaderboard_path()) as database:
        _ensure_leaderboard_schema(database)
        database.execute(
            """INSERT INTO tournaments
               (submission, target_pack, judge, profile, run_count, pass_rate, mean_score, mean_k)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                row["submission"], row["target_pack"], row["judge"], row["profile"], row["run_count"],
                row["pass_rate"], row["mean_score"], row["mean_K"],
            ),
        )


def _ensure_leaderboard_schema(database: sqlite3.Connection) -> None:
    database.execute(
        """CREATE TABLE IF NOT EXISTS tournaments (
            id INTEGER PRIMARY KEY, created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            submission TEXT NOT NULL, target_pack TEXT NOT NULL, judge TEXT NOT NULL,
            profile TEXT NOT NULL DEFAULT '',
            run_count INTEGER NOT NULL, pass_rate REAL NOT NULL, mean_score REAL NOT NULL,
            mean_k REAL
        )"""
    )
    columns = {row[1] for row in database.execute("PRAGMA table_info(tournaments)")}
    if "profile" not in columns:
        database.execute("ALTER TABLE tournaments ADD COLUMN profile TEXT NOT NULL DEFAULT ''")


def _cmd_tournament(args: argparse.Namespace) -> int:
    if args.runs < 1:
        raise ArenaError("--runs must be positive")
    pack = load_target_pack(args.target_pack)
    results = []
    for target_id in pack.target_ids():
        for repeat in range(args.runs):
            results.append(
                _run_submission(
                    Path(args.path), str(pack.root), target_id, args.seed + repeat,
                    runs_dir=Path(args.runs_dir) if args.runs_dir else None,
                    allow_time_travel=not args.no_time_travel,
                    judge_mode=args.judge,
                    judge_repeats=args.judge_repeats,
                    fmn_m=args.fmn_m,
                    fmn_n=args.fmn_n,
                    runner_profile=args.runner,
                    progress=args.progress,
                    max_cost_usd_per_role=args.max_cost_usd_per_role,
                    max_information_bits=args.max_information_bits,
                    guide_agent=args.guide_agent,
                    guide_agent_model=args.guide_agent_model,
                    guide_agent_executable=args.guide_agent_executable,
                    guide_agent_reasoning_effort=args.guide_agent_reasoning_effort,
                    guide_agent_timeout_seconds=args.guide_agent_timeout_seconds,
                    guide_agent_max_budget_usd_per_turn=(
                        args.guide_agent_max_budget_usd_per_turn
                    ),
                )
            )
    passed = [result for result in results if result["status"] == "pass"]
    manifest = load_manifest(args.path)
    row = {
        "submission": manifest.name,
        "target_pack": pack.name,
        "judge": "smoke-answer-probe" if pack.name == "smoke" else args.judge,
        "profile": results[0]["evaluation_profile_sha256"],
        "run_count": len(results),
        "pass_rate": len(passed) / len(results),
        "mean_score": sum(result["score"] for result in results) / len(results),
        "mean_K": (sum(result["K"] for result in passed) / len(passed)) if passed else None,
        "runs": [result["run_id"] for result in results],
    }
    _record_tournament(row)
    print(json.dumps(row, indent=2, sort_keys=True))
    return 0


def _cmd_leaderboard(_: argparse.Namespace) -> int:
    path = _leaderboard_path()
    with sqlite3.connect(path) as database:
        database.row_factory = sqlite3.Row
        _ensure_leaderboard_schema(database)
        rows = [dict(row) for row in database.execute(
            "SELECT * FROM tournaments ORDER BY mean_score DESC, pass_rate DESC, id ASC"
        )]
    print(json.dumps({"leaderboard": rows}, indent=2, sort_keys=True))
    return 0


def _cmd_serve(args: argparse.Namespace) -> int:
    if args.host not in {"127.0.0.1", "localhost", "::1"} and not args.allow_remote:
        raise ArenaError("non-localhost binding requires --allow-remote")
    print(f"Idea Arena listening on http://{args.host}:{args.port}", file=sys.stderr)
    serve(args.host, args.port)
    return 0


def _cmd_render(args: argparse.Namespace) -> int:
    output = render_trajectory_html(
        args.inputs,
        args.output,
        private=args.private,
        title=args.title,
    )
    print(json.dumps({"output": str(output), "private": bool(args.private)}, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="idea-arena")
    subcommands = parser.add_subparsers(dest="command", required=True)
    doctor = subcommands.add_parser("doctor", help="verify the local installation")
    doctor.add_argument("--live", action="store_true", help="make a minimal call to every configured model")
    doctor.add_argument("--env-file")
    doctor.set_defaults(handler=_cmd_doctor)
    validate = subcommands.add_parser("validate", help="validate a pair with an offline smoke match")
    validate.add_argument("path")
    validate.set_defaults(handler=_cmd_validate)
    run = subcommands.add_parser("run", help="run one local match")
    run.add_argument("path")
    run.add_argument("--target-pack", default="smoke")
    run.add_argument("--target")
    run.add_argument("--seed", type=int, default=1)
    run.add_argument("--runs-dir")
    run.add_argument("--no-time-travel", action="store_true")
    run.add_argument("--judge", choices=("fmn", "essence", "directional"), default="fmn")
    run.add_argument(
        "--judge-repeats",
        type=int,
        help="independent Judge repetitions (default: 3 for essence, 1 otherwise)",
    )
    run.add_argument("--fmn-m", type=int, help="number of leading gold findings in F(m,n)")
    run.add_argument("--fmn-n", type=int, default=0, help="findings required among the leading m")
    run.add_argument("--disclosure", choices=("development", "hidden"), default="development")
    run.add_argument("--runner", choices=("local", "hardened"), default="local")
    run.add_argument("--env-file")
    run.add_argument("--progress", action="store_true")
    run.add_argument("--max-cost-usd-per-role", type=float, default=1000.0)
    run.add_argument("--max-information-bits", type=float, default=1024.0)
    run.add_argument(
        "--no-html-report",
        action="store_true",
        help="do not generate the default public and private trajectory reports",
    )
    _add_guide_agent_arguments(run)
    _add_generator_codex_arguments(run)
    run.add_argument('--generator-memory-target-chars', type=int,
                     help='summary prompt target, distinct from the hard character ceiling')
    run.add_argument("--generator-model")
    run.set_defaults(handler=_cmd_run)
    sample = subcommands.add_parser(
        "sample-ideas",
        help="sample Generator options by probability and return its first submission",
    )
    sample.add_argument("path")
    sample.add_argument("--target-pack", default="smoke")
    sample.add_argument("--seed", type=int, default=1)
    sample.add_argument("--max-questions", type=int, default=256)
    sample.add_argument("--max-model-cost-usd", type=float, default=1000.0)
    sample.add_argument("--generator-model")
    _add_generator_codex_arguments(sample)
    sample.add_argument("--runs-dir")
    sample.add_argument("--env-file")
    sample.add_argument(
        "--quiet",
        action="store_true",
        help="suppress streaming progress on stderr",
    )
    sample.set_defaults(handler=_cmd_sample_ideas)
    new = subcommands.add_parser("new", help="create a minimal submission")
    new.add_argument("path")
    new.add_argument("--reference", action="store_true", help="create the bundled source-compatible pair")
    new.set_defaults(handler=_cmd_new)
    replay = subcommands.add_parser("replay", help="verify a recorded run")
    replay.add_argument("run_dir")
    replay.add_argument("--actor", action="store_true", help="also reconstruct both participant actors")
    replay.set_defaults(handler=_cmd_replay)
    reprice = subcommands.add_parser("reprice", help="verify and reprice completed records with the current occurrence prior")
    reprice.add_argument("run_dir", help="one run directory or a tree of runs")
    reprice.add_argument("--write", action="store_true", help="archive original scores and update score.json (default is dry run)")
    reprice.add_argument("--report", help="save the complete JSON migration report")
    reprice.set_defaults(handler=_cmd_reprice)
    resume = subcommands.add_parser("resume", help="continue from a durable failed/interrupted checkpoint")
    resume.add_argument("run_dir")
    resume.add_argument("--env-file")
    resume.add_argument("--progress", action="store_true")
    resume.add_argument("--max-cost-usd-per-role", type=float)
    resume.add_argument(
        "--discard-service-tail",
        action="store_true",
        help=(
            "drop the post-checkpoint service tail and re-run it live; for "
            "runs whose tail was torn by a hard interrupt"
        ),
    )
    resume.add_argument("--max-information-bits", type=float)
    resume.add_argument("--budget-ledger", help="shared cumulative Fable two-stage billing ledger")
    resume.add_argument("--generator-output-tokens", type=int,
                        help="record a live common Claude/GLM output allowance increase (up to 128000)")
    resume.add_argument("--generator-memory-chars", type=int,
                        help="explicitly increase the common summary cap for this continuation")
    resume.add_argument("--generator-memory-target-chars", type=int,
                        help="requested summary length, independently of the hard cap")
    resume.add_argument(
        "--no-html-report",
        action="store_true",
        help="do not generate the default public and private trajectory reports",
    )
    resume.add_argument(
        "--retry-interrupted-call",
        action="store_true",
        help=(
            "retry an uncaught provider failure live while retaining its cost and audit record"
        ),
    )
    resume.add_argument(
        "--compatible-submission",
        help=(
            "resume with an audited bug-fixed participant tree; name, protocol, "
            "and actor entrypoints must match the source snapshot; Judge promotion "
            "also freezes shared and all completed policy modules"
        ),
    )
    resume.add_argument(
        "--promote-judge",
        choices=("essence", "strict"),
        help=(
            "fork a completed passing run from its private before-judge boundary "
            "and continue under the next stronger Judge"
        ),
    )
    _add_guide_agent_arguments(resume)
    resume.set_defaults(handler=_cmd_resume)
    status = subcommands.add_parser("status", help="show durable run progress and resumability")
    status.add_argument("run_dir")
    status.set_defaults(handler=_cmd_status)
    tournament = subcommands.add_parser("tournament", help="run a pair over every target")
    tournament.add_argument("path")
    tournament.add_argument("--target-pack", default="smoke")
    tournament.add_argument("--runs", type=int, default=1)
    tournament.add_argument("--seed", type=int, default=1)
    tournament.add_argument("--runs-dir")
    tournament.add_argument("--no-time-travel", action="store_true")
    tournament.add_argument("--judge", choices=("fmn", "essence", "directional"), default="fmn")
    tournament.add_argument(
        "--judge-repeats",
        type=int,
        help="independent Judge repetitions (default: 3 for essence, 1 otherwise)",
    )
    tournament.add_argument("--fmn-m", type=int, help="number of leading gold findings in F(m,n)")
    tournament.add_argument("--fmn-n", type=int, default=0, help="findings required among the leading m")
    tournament.add_argument("--runner", choices=("local", "hardened"), default="local")
    tournament.add_argument("--env-file")
    tournament.add_argument("--progress", action="store_true")
    tournament.add_argument("--max-cost-usd-per-role", type=float, default=1000.0)
    tournament.add_argument("--max-information-bits", type=float, default=1024.0)
    _add_guide_agent_arguments(tournament)
    tournament.set_defaults(handler=_cmd_tournament)
    leaderboard = subcommands.add_parser("leaderboard", help="show local tournament results")
    leaderboard.set_defaults(handler=_cmd_leaderboard)
    serve_parser = subcommands.add_parser("serve", help="serve the local development UI")
    serve_parser.add_argument("--host", default="127.0.0.1")
    serve_parser.add_argument("--port", type=int, default=8765)
    serve_parser.add_argument("--allow-remote", action="store_true")
    serve_parser.set_defaults(handler=_cmd_serve)
    render_parser = subcommands.add_parser(
        "render", help="render run trajectories as a self-contained HTML report"
    )
    render_parser.add_argument(
        "inputs", nargs="+", help="run directories or JSON/JSONL trajectory files"
    )
    render_parser.add_argument("--output", "-o", default="trajectory-report.html")
    render_parser.add_argument("--title", default="Idea Arena trajectories")
    render_parser.add_argument(
        "--private",
        action="store_true",
        help="read private event streams; the generated HTML may contain secrets",
    )
    render_parser.set_defaults(handler=_cmd_render)
    return parser


def _add_generator_codex_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--generator-backend", choices=("api", "codex", "claude-code"), default="api")
    parser.add_argument("--generator-memory", choices=("common",), help="reasoning-preserving history and common self-summary")
    parser.add_argument("--generator-memory-effort", choices=("low", "medium", "high", "xhigh", "max"))
    parser.add_argument("--generator-memory-compact-tokens", type=int, default=100000)
    parser.add_argument("--generator-memory-compact-calls", type=int, default=0)
    parser.add_argument("--budget-ledger", help="shared cumulative Fable two-stage billing ledger")
    parser.add_argument("--generator-output-tokens", type=int, default=128000,
                        help="common Claude/GLM output allowance; Codex routes remain unchanged")
    parser.add_argument("--generator-memory-chars", type=int, default=12000)
    parser.add_argument("--generator-claude-install", help="receipt from tools/install_claude.py")
    parser.add_argument("--generator-claude-auth-home", help="independent Claude subscription login directory")
    parser.add_argument("--generator-claude-effort", choices=("low", "medium", "high", "xhigh", "max"), default="max")
    parser.add_argument("--generator-codex-build", help="source-build receipt from tools/build_codex.py")
    parser.add_argument("--generator-codex-auth-home", help="independent account directory created by tools/codex_account.py")
    parser.add_argument("--generator-codex-effort", choices=("low", "medium", "high", "xhigh", "max"),
                        help="override all Generator semantic calls with this Codex reasoning effort")


def _generator_codex_config(args: argparse.Namespace) -> dict | None:
    if args.generator_backend != "codex":
        return None
    from .runtime.codex_generator import configuration
    try:
        return configuration(args.generator_codex_build, args.generator_codex_auth_home,
                             reasoning_effort=args.generator_codex_effort)
    except (OSError, ValueError, RuntimeError) as exc:
        raise ArenaError("Codex Generator setup failed; run python3 tools/build_codex.py: " + str(exc)) from exc


def _generator_memory_config(args: argparse.Namespace) -> dict | None:
    if not getattr(args, "generator_memory", None):
        return None
    from .runtime.common_memory import configuration
    model = args.generator_model or ("gpt-6-astra" if args.generator_backend == "codex" else None)
    if not model or args.generator_backend == "claude-code":
        raise ArenaError("Common memory requires an explicit API model or --generator-backend codex")
    if model in {"gpt-6-astra", "gpt-5.6-sol"} and args.generator_backend != "codex":
        raise ArenaError("Common-memory Astra/Sol require --generator-backend codex (subscription)")
    try:
        return configuration(model, effort=args.generator_memory_effort or args.generator_codex_effort,
            compact_tokens=args.generator_memory_compact_tokens, compact_calls=args.generator_memory_compact_calls,
            memory_chars=args.generator_memory_chars, codex=_generator_codex_config(args))
    except (OSError, ValueError, RuntimeError) as exc:
        raise ArenaError("Common-memory setup failed: " + str(exc)) from exc


def _generator_claude_config(args: argparse.Namespace) -> dict | None:
    if args.generator_backend != "claude-code":
        return None
    from .runtime.claude_generator import configuration
    try:
        return configuration(args.generator_claude_install, args.generator_claude_auth_home,
                             reasoning_effort=args.generator_claude_effort)
    except (OSError, ValueError, RuntimeError) as exc:
        raise ArenaError("Claude Generator setup failed; run tools/install_claude.py and tools/claude_account.py: " + str(exc)) from exc


def _add_guide_agent_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--guide-agent", "--oracle-agent",
        choices=("claude-code", "codex", "human"),
        help=(
            "run the match with participant.guide:AgentGuide (or legacy AgentOracle), routing "
            "every Guide decision through the selected backend via "
            "services.agent_turn; 'human' blocks each turn on a person "
            "answering via run_dir/agent-workspace/oracle/human/ files "
            "(on resume, overrides the source run's guide backend)"
        ),
    )
    parser.add_argument("--guide-agent-model", "--oracle-agent-model")
    parser.add_argument("--guide-agent-executable", "--oracle-agent-executable")
    parser.add_argument(
        "--guide-agent-reasoning-effort", "--oracle-agent-reasoning-effort",
        choices=("low", "medium", "high", "xhigh", "max"),
        default="high",
    )
    parser.add_argument("--guide-agent-timeout-seconds", "--oracle-agent-timeout-seconds", type=float, default=600.0)
    parser.add_argument("--guide-agent-max-budget-usd-per-turn", "--oracle-agent-max-budget-usd-per-turn", type=float)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if getattr(args, "env_file", None):
            _load_env_file(args.env_file)
        return int(args.handler(args))
    except ArenaError as exc:
        print(f"idea-arena: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())


# Legacy Python names remain available for existing submissions.
_oracle_agent_config = _guide_agent_config
_oracle_actor_timeout = _guide_actor_timeout
_make_oracle_agent_backend = _make_guide_agent_backend
_OracleJudgeHandle = _GuideJudgeHandle
_add_oracle_agent_arguments = _add_guide_agent_arguments

# Legacy imported names.
HumanOracleBackend = HumanGuideBackend
