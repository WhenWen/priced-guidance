"""Source-pinned Codex Generator with immutable native conversation branches."""
from __future__ import annotations

import copy
import atexit
from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import secrets
import tempfile
import threading
import time
from typing import Any

from ..errors import ReplayDivergence, ResourceLimitExceeded
from ..replay.artifacts import ArtifactStore
from .codex_rpc import CodexRPC, run_turn, turn_timeout_seconds
from .codex_prompts import CACHE_VERSIONS, DEFAULT_VERSION, LEGACY_VERSION, prompt_sha
from .codex_sandbox import PROFILE_VERSION, SUPPORTED_PROFILES, environment, profile

ROOT = Path(__file__).resolve().parents[3]
SOURCE_LOCK = ROOT / "tools/codex/source.lock.json"
CACHE_SOURCE_LOCK = ROOT / "tools/codex/cache-source.lock.json"
BACKEND = "codex-source-v1"


def file_sha(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def configuration(receipt_path: str | Path | None = None,
                  auth_home: str | Path | None = None, *,
                  prompt_version: str = DEFAULT_VERSION,
                  reasoning_effort: str | None = None) -> dict:
    if reasoning_effort not in {None, "low", "medium", "high", "xhigh", "max"}:
        raise ValueError("Unsupported Codex Generator reasoning effort")
    cache_transport = prompt_version in CACHE_VERSIONS
    default_build = "codex-cache-v1" if cache_transport else "codex"
    receipt_path = Path(receipt_path or ROOT / f".idea-arena/{default_build}/build-receipt.json").expanduser().resolve()
    receipt = json.loads(receipt_path.read_text())
    source_lock = CACHE_SOURCE_LOCK if cache_transport else SOURCE_LOCK
    lock = json.loads(source_lock.read_text())
    if (receipt.get("source") != lock or receipt.get("source_lock_sha256") != file_sha(source_lock)
            or receipt.get("patch_sha256") != {name: file_sha(source_lock.parent / name) for name in lock["patches"]}):
        raise RuntimeError("Codex build receipt does not match the versioned source lock and patches")
    executable = Path(receipt["executable"]).resolve()
    if file_sha(executable) != receipt["executable_sha256"]:
        raise RuntimeError("Codex executable differs from its source-build receipt")
    account = Path(auth_home or Path.home() / ".local/share/idea-arena/codex-account").expanduser().resolve()
    return {"backend": BACKEND, "build_receipt": str(receipt_path), "auth_home": str(account),
            "source_commit": lock["commit"], "executable_sha256": receipt["executable_sha256"],
            "sandbox_profile": PROFILE_VERSION,
            "context_mode": "native-live-session-checkpoint-forks" if cache_transport else "native-fork-every-call",
            "auth_mode": "chatgpt",
            "prompt_version": prompt_version, "prompt_sha256": prompt_sha(prompt_version),
            **({"reasoning_effort": reasoning_effort} if reasoning_effort is not None else {})}


def import_account(account: Path, source: Path | None = None) -> None:
    account.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(account, 0o700)
    destination = account / "auth.json"
    if not destination.exists():
        source = source or Path.home() / ".codex/auth.json"
        value = json.loads(source.read_text())
        if not value.get("tokens") or value.get("OPENAI_API_KEY"):
            raise RuntimeError("A saved ChatGPT/Codex account login is required; API-key auth is refused")
        descriptor = os.open(destination, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, "w") as stream:
            json.dump(value, stream)
    value = json.loads(destination.read_text())
    if not value.get("tokens") or value.get("OPENAI_API_KEY"):
        raise RuntimeError("Codex generator requires account credentials, not an API key")


def require_account(account: Path) -> None:
    path = account / "auth.json"
    if not path.is_file():
        raise RuntimeError("No independent Codex account login. Run: uv run python tools/codex_account.py --auth-home "
                           + str(account) + " login")
    value = json.loads(path.read_text())
    if not value.get("tokens") or value.get("OPENAI_API_KEY"):
        raise RuntimeError("Codex generator requires account credentials, not an API key")


def validate_metadata(service: dict) -> None:
    """Keep Codex subscription accounting distinct from physical API attempts."""
    from .services import _request_hash
    metadata, request = service.get("metadata") or {}, service.get("request") or {}
    if metadata.get("backend") != BACKEND:
        raise ReplayDivergence("invalid Codex backend marker")
    if (metadata.get("model") != request.get("model")
            or metadata.get("schema_name") != request.get("schema_name")
            or metadata.get("request_hash") != _request_hash("model.structured", request)
            or metadata.get("auth_mode") != "chatgpt"
            or metadata.get("sandbox_profile") not in SUPPORTED_PROFILES):
        raise ReplayDivergence("Codex service binding diverges")
    if metadata.get("provider_attempts"):
        raise ReplayDivergence("Codex service must not claim API provider attempts")
    if "prompt_version" in metadata or "prompt_sha256" in metadata:
        try:
            expected_prompt = prompt_sha(metadata.get("prompt_version"))
        except RuntimeError as exc:
            raise ReplayDivergence("unknown Codex prompt version") from exc
        if metadata.get("prompt_sha256") != expected_prompt:
            raise ReplayDivergence("Codex prompt hash diverges")
    usage = metadata.get("usage") or {}
    if usage.get("cost_usd") != 0.0 or any(type(value) is not int or value < 0
            for key, value in usage.items() if key != "cost_usd"):
        raise ReplayDivergence("invalid Codex account usage")
    if not service.get("error"):
        context = metadata.get("codex_context") or {}
        if not context.get("thread_id") or not context.get("turn_id") or not context.get("rollout"):
            raise ReplayDivergence("Codex success is missing its durable native context")
        usages = [event["params"]["tokenUsage"]["total"] for event in metadata.get("codex_events", [])
                  if event.get("method") == "thread/tokenUsage/updated"]
        baseline = (metadata.get("parent_context") or {}).get("usage_total") or {}
        if (not usages or usage.get("input_tokens") != usages[-1].get("inputTokens", 0) - baseline.get("inputTokens", 0)
                or usage.get("output_tokens") != usages[-1].get("outputTokens", 0) - baseline.get("outputTokens", 0)
                or usage.get("cached_input_tokens", 0) != usages[-1].get("cachedInputTokens", 0) - baseline.get("cachedInputTokens", 0)):
            raise ReplayDivergence("Codex usage differs from its native event stream")
        if metadata.get("prompt_version") in CACHE_VERSIONS:
            lineage = metadata.get("cache_lineage")
            parent_lineage = (metadata.get("parent_context") or {}).get("cache_lineage", lineage)
            if (not isinstance(lineage, str) or len(lineage) != 64
                    or any(c not in "0123456789abcdef" for c in lineage)
                    or context.get("cache_lineage") != lineage or parent_lineage != lineage):
                raise ReplayDivergence("Codex cache lineage differs from its native checkpoint")


class CodexGeneratorBackend:
    """Native history belongs to each ReplayableServices actor, never the backend.

    A completed call archives its entire native rollout. Linear calls reuse the
    live session; checkpoint restoration forks its exact immutable archive.
    """
    supports_context_chain = True

    def __init__(self, config: dict, artifacts: ArtifactStore, *, trace_requests: bool = False):
        checked = configuration(config["build_receipt"], config["auth_home"],
                                prompt_version=config.get("prompt_version", LEGACY_VERSION),
                                reasoning_effort=config.get("reasoning_effort"))
        # Existing recorded runs retain their original prompt on resume.
        # New runs bind the prompt version and hash into the run profile.
        if "prompt_version" not in config and "prompt_sha256" not in config:
            checked.pop("prompt_version")
            checked.pop("prompt_sha256")
        if checked != config:
            raise RuntimeError("Recorded Codex configuration changed; refusing to resume")
        self.config = checked
        self.trace_requests = trace_requests
        self._native_workspace = None
        self._native_rpc = None
        self._native_lineage = None
        self._active_context = None
        self._active_thread = None
        atexit.register(self.close)
        receipt = json.loads(Path(config["build_receipt"]).read_text())
        self.executable = Path(receipt["executable"]).resolve()
        self.account = Path(config["auth_home"])
        self.artifacts = artifacts
        # Fail on unsupported hosts before copying any credentials.
        profile(Path(tempfile.gettempdir()) / "arena-profile-check", self.executable)
        require_account(self.account)
        self._lock = threading.RLock()
        self._thread = threading.local()
        self._usage = {"calls": 0, "input_tokens": 0, "cached_input_tokens": 0,
                       "cache_write_input_tokens": 0,
                       "output_tokens": 0, "reasoning_output_tokens": 0,
                       "unknown_account_turns": 0, "cost_usd": 0.0}

    def structured(self, **request):
        return self.structured_in_context(None, **request)

    def close(self):
        """Release the idle native session; archived checkpoints stay immutable."""
        rpc, self._native_rpc = getattr(self, "_native_rpc", None), None
        workspace, self._native_workspace = getattr(self, "_native_workspace", None), None
        try:
            if rpc is not None:
                rpc.close()
        finally:
            if workspace is not None:
                workspace.cleanup()
            self._active_context = self._active_thread = self._native_lineage = None

    @contextmanager
    def _turn_workspace(self, cache_key):
        if cache_key is None:
            with tempfile.TemporaryDirectory(prefix="arena-codex-") as folder:
                yield Path(folder).resolve()
            return
        existing = getattr(self, "_native_workspace", None)
        if existing is not None:
            auth = Path(existing.name) / "codex/auth.json"
            if (self._native_lineage != cache_key
                    or file_sha(auth) != file_sha(self.account / "auth.json")):
                self.close()
        if getattr(self, "_native_workspace", None) is None:
            self._native_workspace = tempfile.TemporaryDirectory(prefix="arena-codex-")
            self._native_lineage = cache_key
        yield Path(self._native_workspace.name).resolve()

    @property
    def reasoning_effort(self):
        return self.config.get("reasoning_effort")

    def structured_in_context(self, context: dict | None, **request: Any):
        from .services import _request_hash
        started = time.monotonic()
        prompt_version = self.config.get("prompt_version", LEGACY_VERSION)
        metadata = {"backend": BACKEND, "auth_mode": "chatgpt", "billing": "subscription; no API-dollar estimate",
                    "sandbox_profile": PROFILE_VERSION, "source_commit": self.config["source_commit"],
                    "executable_sha256": self.config["executable_sha256"],
                    "prompt_version": prompt_version, "prompt_sha256": prompt_sha(prompt_version),
                    "model": request["model"], "schema_name": request["schema_name"],
                    "reasoning_effort": request.get("reasoning_effort", "high"),
                    "request_hash": _request_hash("model.structured", request),
                    "native_turn_timeout_seconds": turn_timeout_seconds(),
                    "parent_context": copy.deepcopy(context), "usage": {"cost_usd": 0.0}}
        self._thread.metadata = metadata
        if request.get("conversation_state") or request.get("return_conversation_state"):
            metadata.update(error_type="ValueError", usage={"cost_usd": 0.0},
                            latency_s=time.monotonic() - started)
            raise ValueError("API conversation state cannot be mixed with native Codex history")
        cache_key = None
        if prompt_version in CACHE_VERSIONS:
            cache_key = context.get("cache_lineage") if context else secrets.token_hex(32)
            if (not isinstance(cache_key, str) or len(cache_key) != 64
                    or any(c not in "0123456789abcdef" for c in cache_key)):
                raise ValueError("Native checkpoint is missing a valid cache lineage")
            metadata["cache_lineage"] = cache_key
        # Serializing account access prevents concurrent refresh-token rotation.
        # Actor-local ordering is separately enforced by ReplayableServices.
        with self._lock, (self.account / "auth.lock").open("a") as account_lock:
            fcntl.flock(account_lock, fcntl.LOCK_EX)
            with self._turn_workspace(cache_key) as workspace:
                env = environment(workspace)
                auth = Path(env["CODEX_HOME"]) / "auth.json"
                shutil.copyfile(self.account / "auth.json", auth)
                os.chmod(auth, 0o600)
                (Path(env["CODEX_HOME"]) / "config.toml").write_text(
                    'forced_login_method = "chatgpt"\ncli_auth_credentials_store = "file"\n'
                    'web_search = "disabled"\napproval_policy = "never"\n'
                    'sandbox_mode = "read-only"\n'
                    '[features]\nshell_tool = false\nplugins = false\napps = false\n'
                    'memories = false\nmulti_agent = false\nmulti_agent_v2 = false\n'
                    '[analytics]\nenabled = false\n[feedback]\nenabled = false\n'
                )
                model_started = False
                failed = False
                try:
                    if context:
                        snapshot = self.artifacts.load_json(context["rollout"])
                        (workspace / "parent.jsonl").write_text(snapshot["rollout"], encoding="utf-8")
                    model_started = True
                    if cache_key and getattr(self, "_native_rpc", None) is None:
                        self._native_rpc = CodexRPC(self.executable, workspace,
                            timeout=turn_timeout_seconds(), cache_key=cache_key,
                            trace_requests=self.trace_requests)
                    trace_path = workspace / "prompt-requests.jsonl"
                    trace_path.unlink(missing_ok=True)
                    result = run_turn(self.executable, workspace, request, context, prompt_version,
                                      **({"cache_key": cache_key, "trace_requests": self.trace_requests,
                                          "live_rpc": self._native_rpc,
                                          "live_thread": (self._active_thread if context is not None
                                                          and context == self._active_context else None)}
                                         if cache_key else {}))
                    metadata["usage"] = result["usage"]
                    metadata["codex_events"] = result["events"]
                    metadata["thread_action"] = result.get("thread_action", "fork" if context else "start")
                    # Paths come from trusted Codex, but must stay in the
                    # allowlisted workspace before the host reads anything.
                    path = Path(result["rollout"] or "").resolve()
                    if not path.is_relative_to(workspace) or not path.is_file():
                        raise RuntimeError("Codex did not persist a local native rollout")
                    rollout = path.read_text(encoding="utf-8")
                    if cache_key:
                        # The persistent process stays open. Wait for the final
                        # ordered rollout event instead of relying on shutdown
                        # to flush an asynchronous writer before checkpointing.
                        deadline = time.monotonic() + 5
                        while True:
                            try:
                                tail = [json.loads(line) for line in rollout.splitlines()[-5:]]
                                complete = any(row.get("type") == "event_msg"
                                    and row.get("payload", {}).get("type") == "task_complete"
                                    and row["payload"].get("turn_id") == result["turn_id"] for row in tail)
                            except json.JSONDecodeError:
                                complete = False
                            if complete:
                                break
                            if time.monotonic() >= deadline:
                                raise RuntimeError("Codex native checkpoint did not persist the completed turn")
                            time.sleep(0.05)
                            rollout = path.read_text(encoding="utf-8")
                    reference = self.artifacts.put_json({"rollout": rollout})
                    metadata["codex_context"] = {"rollout": reference,
                        "thread_id": result["thread_id"], "turn_id": result["turn_id"],
                        "usage_total": result["usage_total"],
                        **({"cache_lineage": cache_key} if cache_key else {})}
                    if int(result["usage"]["output_tokens"]) > int(request.get("max_output_tokens", 8000)):
                        raise ResourceLimitExceeded("Codex output exceeded the requested token budget (post-response check)")
                    if cache_key:
                        self._active_context = copy.deepcopy(metadata["codex_context"])
                        self._active_thread = {"id": result["thread_id"], "path": str(path)}
                    return result["output"]
                except BaseException as exc:
                    failed = True
                    metadata["error_type"] = type(exc).__name__
                    metadata["codex_events"] = getattr(exc, "codex_events", metadata.get("codex_events", []))
                    stderr = workspace / "stderr.log"
                    metadata["stderr"] = stderr.read_text(errors="replace")[-8000:] if stderr.is_file() else ""
                    if not metadata["usage"].get("calls"):
                        metadata["usage"] = {**getattr(exc, "codex_usage", {}),
                                             "unknown_account_turns": int(model_started), "cost_usd": 0.0}
                    raise
                finally:
                    trace = workspace / "prompt-requests.jsonl"
                    if trace.is_file():
                        metadata["request_trace"] = self.artifacts.put_json(
                            {"requests": [json.loads(line) for line in trace.read_text().splitlines()]})
                    metadata["latency_s"] = time.monotonic() - started
                    for key, value in metadata["usage"].items():
                        self._usage[key] = self._usage.get(key, 0) + value
                    # Only the auth cache is persisted to this dedicated login
                    # directory; no personal sessions/config are ever imported.
                    updated = self.account / "auth.pending"
                    shutil.copyfile(auth, updated)
                    os.chmod(updated, 0o600)
                    os.replace(updated, self.account / "auth.json")
                    if failed and cache_key:
                        self.close()

    def last_call_metadata(self):
        return copy.deepcopy(getattr(self._thread, "metadata", {}))

    def usage_totals(self):
        with self._lock:
            return dict(self._usage)

    def restore_usage(self, usage):
        with self._lock:
            self._usage = {key: usage.get(key, 0) for key in self._usage}
