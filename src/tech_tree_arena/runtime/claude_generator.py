"""Pinned Claude Code account backend, with immutable native transcript ancestors.

Only the broker sees credentials and transcript artifacts. Actor services own
the current context, so checkpoint restoration cannot inherit a discarded tail.
"""
from __future__ import annotations

import atexit
import copy
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import threading
import time
import uuid

from ..errors import ReplayDivergence, ResourceLimitExceeded
from .claude_sandbox import PROFILE_VERSION, command, environment
from .codex_prompts import EAGER_VERSION, instructions

ROOT = Path(__file__).resolve().parents[3]
RELEASE_LOCK = ROOT / "tools/claude/release.lock.json"
BACKEND = "claude-code-native-v1"
PROMPT_VERSION = "claude-eager-v2"
# Native resume owns history. Instructions concern visible proposals and public
# evidence only; the model is never asked to disclose or reconstruct reasoning.
PROMPT = (instructions(EAGER_VERSION).replace("developer instructions", "task instructions")
    .replace("Native conversation history includes your earlier reasoning and may include previews for several mutually exclusive routes.",
             "Conversation history contains earlier Generator outputs, including previews for several mutually exclusive routes.")
    .replace("Retain useful earlier reasoning as hypotheses", "Treat earlier candidate proposals as hypotheses")) + """
The latest user message is a broker-authored idea_arena_call containing the
current task instructions, user input and payload schema. Return only a JSON
object with one string field payload_json, whose value is the serialized JSON
payload matching that schema. No Markdown fences. Do not call tools. Preserve
all eager preview content inside the payload. This transport envelope is fixed
across calls; task-specific instructions belong only to the latest call.
"""
PROMPT_SHA = hashlib.sha256(PROMPT.encode()).hexdigest()


def file_sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def installation(receipt_path=None):
    path = Path(receipt_path or ROOT / ".idea-arena/claude-code/install-receipt.json").expanduser().resolve()
    receipt = json.loads(path.read_text())
    if receipt.get("release") != json.loads(RELEASE_LOCK.read_text()) or receipt.get("release_lock_sha256") != file_sha(RELEASE_LOCK):
        raise RuntimeError("Claude installation differs from the versioned release lock")
    executable = Path(receipt["executable"]).resolve()
    if file_sha(executable) != receipt.get("executable_sha256"):
        raise RuntimeError("Claude executable checksum differs from installation receipt")
    return path, receipt


def configuration(receipt_path=None, auth_home=None, *, reasoning_effort="max"):
    if sys.platform != "linux":
        raise RuntimeError("Claude Generator currently requires Linux bubblewrap")
    if reasoning_effort not in {"low", "medium", "high", "xhigh", "max"}:
        raise ValueError("Unsupported Claude effort")
    path, receipt = installation(receipt_path)
    account = Path(auth_home or Path.home() / ".local/share/idea-arena/claude-account").expanduser().resolve()
    return {"backend": BACKEND, "install_receipt": str(path), "auth_home": str(account),
            "version": receipt["release"]["version"], "executable_sha256": receipt["executable_sha256"],
            "sandbox_profile": PROFILE_VERSION, "auth_mode": "claude-subscription",
            "context_mode": "native-resume-immutable-transcript-forks", "reasoning_effort": reasoning_effort,
            "model_policy": "exact-model-no-fallback-v1",
            "prompt_version": PROMPT_VERSION, "prompt_sha256": PROMPT_SHA}


def account_document(account):
    path = Path(account) / ".credentials.json"
    if path.is_symlink() or not path.is_file():
        raise RuntimeError("Independent Claude login required: python tools/claude_account.py login")
    doc = json.loads(path.read_text())
    oauth = doc.get("claudeAiOauth") or {}
    if not oauth.get("accessToken") or not oauth.get("subscriptionType"):
        raise RuntimeError("Claude subscription credentials required; API-key fallback is disabled")
    if oauth.get("expiresAt", 0) <= time.time() * 1000 and not oauth.get("refreshToken"):
        raise RuntimeError("Claude login expired; log in again with tools/claude_account.py")
    # No settings, projects, hooks, MCP servers or personal memory are imported.
    return {"claudeAiOauth": oauth}


def token_usage(result):
    raw = result.get("usage")
    if not isinstance(raw, dict) or not all(type(raw.get(k)) is int and raw[k] >= 0 for k in ("input_tokens", "output_tokens")):
        raise RuntimeError("Claude result did not report token usage")
    cached, written = raw.get("cache_read_input_tokens", 0), raw.get("cache_creation_input_tokens", 0)
    if any(type(x) is not int or x < 0 for x in (cached, written)):
        raise RuntimeError("Invalid Claude cache usage")
    # Anthropic input excludes cached reads/writes; Arena input includes both.
    return {"calls": 1, "input_tokens": raw["input_tokens"] + cached + written,
            "cached_input_tokens": cached, "cache_write_input_tokens": written,
            "output_tokens": raw["output_tokens"], "cost_usd": 0.0}


def validate_metadata(service):
    from .services import _request_hash
    m, request = service.get("metadata") or {}, service.get("request") or {}
    if (m.get("backend") != BACKEND or m.get("sandbox_profile") != PROFILE_VERSION
            or m.get("auth_mode") != "claude-subscription" or m.get("model") != request.get("model")
            or m.get("request_hash") != _request_hash("model.structured", request)
            or m.get("prompt_version") != PROMPT_VERSION or m.get("prompt_sha256") != PROMPT_SHA
            or m.get("provider_attempts")):
        raise ReplayDivergence("Claude service binding diverges")
    usage = m.get("usage") or {}
    if usage.get("cost_usd") != 0.0 or any(type(v) is not int or v < 0 for k, v in usage.items() if k != "cost_usd"):
        raise ReplayDivergence("Invalid Claude account usage")
    if not service.get("error"):
        context = m.get("native_context") or {}
        if (context.get("backend") != BACKEND or not context.get("session_id") or not context.get("transcript")
                or context.get("cwd") != "/arena" or not m.get("claude_result")):
            raise ReplayDivergence("Claude success lacks its native checkpoint")
        try:
            if usage != token_usage(m["claude_result"]):
                raise ValueError("usage mismatch")
        except (RuntimeError, ValueError) as exc:
            raise ReplayDivergence("Claude usage differs from recorded result") from exc


def run_cli(executable, workspace, args, prompt, timeout):
    """No shell, inherited environment, hidden retries, or unsandboxed fallback."""
    env = environment(workspace)
    proc = subprocess.Popen(command(workspace, executable, args), env=env, cwd=workspace,
                            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            start_new_session=True, text=True)
    try:
        stdout, stderr = proc.communicate(prompt, timeout=timeout)
    except BaseException:
        os.killpg(proc.pid, signal.SIGTERM)
        try:
            proc.communicate(timeout=10)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
            proc.communicate()
        raise
    events = []
    for line in stdout.splitlines():
        try:
            events.append(json.loads(line))
        except ValueError:
            raise RuntimeError("Claude returned a malformed event stream") from None
    return proc.returncode, events


class ClaudeGeneratorBackend:
    supports_context_chain = True

    def __init__(self, config, artifacts):
        checked = configuration(config["install_receipt"], config["auth_home"], reasoning_effort=config["reasoning_effort"])
        if checked != config:
            raise RuntimeError("Recorded Claude configuration changed; refusing to resume")
        self.config, self.artifacts = checked, artifacts
        self.executable = Path(installation(config["install_receipt"])[1]["executable"])
        self.account = Path(config["auth_home"])
        account_document(self.account)
        self._workspace = None
        self._active_context = None
        self._lock = threading.RLock()
        self._thread = threading.local()
        self._usage = {"calls": 0, "input_tokens": 0, "cached_input_tokens": 0,
                       "cache_write_input_tokens": 0, "output_tokens": 0,
                       "unknown_account_turns": 0, "cost_usd": 0.0}
        atexit.register(self.close)

    @property
    def reasoning_effort(self):
        return self.config["reasoning_effort"]

    def close(self):
        if self._workspace is not None:
            self._workspace.cleanup()
        self._workspace = self._active_context = None

    def structured(self, **request):
        return self.structured_in_context(None, **request)

    def structured_in_context(self, context, **request):
        from .services import _request_hash
        from .provider_client import _validate_json_schema
        start = time.monotonic()
        metadata = {"backend": BACKEND, "auth_mode": "claude-subscription", "sandbox_profile": PROFILE_VERSION,
                    "prompt_version": PROMPT_VERSION, "prompt_sha256": PROMPT_SHA,
                    "version": self.config["version"], "executable_sha256": self.config["executable_sha256"],
                    "model": request["model"], "reasoning_effort": self.reasoning_effort,
                    "schema_name": request["schema_name"], "request_hash": _request_hash("model.structured", request),
                    "parent_context": copy.deepcopy(context), "usage": {"cost_usd": 0.0}}
        self._thread.metadata = metadata
        if request.get("conversation_state") or request.get("return_conversation_state"):
            raise ValueError("API conversation state cannot be mixed with Claude native history")
        model = request["model"].removeprefix("anthropic/")
        if not model.startswith("claude-"):
            raise ValueError("Select an explicit Claude model ID, not a moving alias")
        timeout = float(os.environ.get("IDEA_ARENA_CLAUDE_TURN_TIMEOUT_S", "1500"))
        if not 1 <= timeout <= 86400:
            raise ValueError("Invalid Claude turn timeout")
        metadata["native_turn_timeout_seconds"] = timeout
        with self._lock, (self.account / "auth.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            forward = context is not None and context == self._active_context and self._workspace is not None
            if not forward:
                self.close()
                self._workspace = tempfile.TemporaryDirectory(prefix="arena-claude-")
            workspace = Path(self._workspace.name).resolve()
            environment(workspace)
            credentials = workspace / "claude/.credentials.json"
            credentials.write_text(json.dumps(account_document(self.account)))
            credentials.chmod(0o600)
            (workspace / "system.txt").write_text(PROMPT)
            # Safe mode preserves OAuth, unlike --bare, which ignores OAuth.
            args = ["--print", "--output-format", "stream-json", "--verbose", "--safe-mode",
                    "--tools", "", "--strict-mcp-config", "--mcp-config", '{"mcpServers":{}}',
                    "--setting-sources", "", "--settings", json.dumps({"disableAllHooks": True,
                        "autoMemoryEnabled": False, "switchModelsOnFlag": False,
                        "fallbackModel": [], "availableModels": [model]}),
                    "--disable-slash-commands", "--no-chrome", "--model", model,
                    "--effort", self.reasoning_effort, "--max-turns", "1",
                    "--system-prompt-file", "/arena/system.txt", "--system-prompt-snapshot", "on"]
            if context:
                if context.get("backend") != BACKEND or context.get("cwd") != "/arena":
                    raise ValueError("Invalid Claude checkpoint")
                session_id = str(uuid.UUID(context["session_id"]))
                if not forward:
                    snapshot = self.artifacts.load_json(context["transcript"])
                    parent = workspace / "claude/projects/-arena" / f"{session_id}.jsonl"
                    parent.parent.mkdir(parents=True, exist_ok=True)
                    parent.write_text(snapshot["transcript"])
                args += ["--resume", session_id]
                if not forward:
                    args += ["--fork-session"]
            else:
                args += ["--session-id", str(uuid.uuid4())]
            metadata["thread_action"] = "continue" if forward else "fork" if context else "start"
            call = {"idea_arena_call": {"instructions": request.get("developer", ""),
                    "input": request.get("user", ""), "schema_name": request["schema_name"], "schema": request["schema"]}}
            began, failed = False, True
            try:
                began = True
                code, events = run_cli(self.executable, workspace, args, json.dumps(call), timeout)
                metadata["native_model_diagnostics"] = {
                    "initial_models": [e["model"] for e in events if e.get("type") == "system" and e.get("model")],
                    "assistant_models": sorted({e["message"]["model"] for e in events
                        if e.get("type") == "assistant" and (e.get("message") or {}).get("model")}),
                    "notices": [{k: e[k] for k in ("subtype", "message", "level", "error") if k in e}
                        for e in events if e.get("type") == "system" and e.get("subtype") != "init"],
                }
                metadata["rate_limit_events"] = [e for e in events if e.get("type") == "rate_limit_event"]
                results = [e for e in events if e.get("type") == "result"]
                if len(results) != 1:
                    raise RuntimeError("Claude did not return exactly one terminal result")
                result = results[0]
                # Store usage even if local schema validation or native turn failed.
                metadata["usage"] = token_usage(result)
                metadata["claude_result"] = result
                if code or result.get("is_error") or result.get("subtype") != "success":
                    raise RuntimeError("Claude turn failed (stop_reason=" + str(result.get("stop_reason"))
                        + ", terminal_reason=" + str(result.get("terminal_reason")) + "): "
                        + str(result.get("result", result.get("subtype", "unknown")))[:1200])
                if any(e.get("type") == "assistant" and any(c.get("type") == "tool_use" for c in e.get("message", {}).get("content", [])) for e in events):
                    raise RuntimeError("Claude unexpectedly used a tool")
                used_models = set(result.get("modelUsage") or {})
                if not used_models or any(m != model for m in used_models):
                    raise RuntimeError("Claude model differed from the pinned model; result is not accepted")
                session = str(uuid.UUID(result["session_id"]))
                if forward and session != context["session_id"]:
                    raise RuntimeError("Native continuation changed session ID")
                if context and not forward and session == context["session_id"]:
                    raise RuntimeError("Native fork did not create a separate session")
                path = workspace / "claude/projects/-arena" / f"{session}.jsonl"
                if path.is_symlink() or not path.resolve().is_relative_to(workspace) or not path.is_file():
                    raise RuntimeError("Native session transcript was not persisted")
                transcript = path.read_text()
                records = [json.loads(line) for line in transcript.splitlines()]
                if not records or not any(r.get("type") == "assistant" for r in records):
                    raise RuntimeError("Native transcript has no completed assistant turn")
                last = next(r for r in reversed(records) if r.get("type") == "assistant")
                text = "\n".join(c.get("text", "") for c in last.get("message", {}).get("content", [])
                                 if isinstance(c, dict) and c.get("type") == "text")
                if text.strip() != result.get("result", "").strip():
                    raise RuntimeError("Native checkpoint does not contain the returned assistant output")
                reference = self.artifacts.put_json({"transcript": transcript})
                metadata["native_context"] = {"backend": BACKEND, "session_id": session, "cwd": "/arena", "transcript": reference}
                metadata["compaction_count"] = sum(r.get("subtype") == "compact_boundary" for r in records)
                if metadata["usage"]["output_tokens"] > request.get("max_output_tokens", 50000):
                    raise ResourceLimitExceeded("Claude output exceeded token budget (post-response check)")
                envelope = json.loads(result.get("result", ""))
                _validate_json_schema(envelope, {"type": "object", "properties": {"payload_json": {"type": "string"}},
                                                 "required": ["payload_json"], "additionalProperties": False})
                output = json.loads(envelope["payload_json"])
                _validate_json_schema(output, request["schema"])
                self._active_context = copy.deepcopy(metadata["native_context"])
                failed = False
                return output
            except BaseException as exc:
                metadata["error_type"] = type(exc).__name__
                if not metadata["usage"].get("calls"):
                    metadata["usage"] = {"unknown_account_turns": int(began), "cost_usd": 0.0}
                raise
            finally:
                metadata["latency_s"] = time.monotonic() - start
                for key, value in metadata["usage"].items():
                    self._usage[key] = self._usage.get(key, 0) + value
                # Retain refreshes only in the selected account, never in artifacts.
                if credentials.is_file() and not credentials.is_symlink():
                    updated = account_document(credentials.parent)
                    pending = self.account / ".credentials.pending"
                    pending.write_text(json.dumps(updated)); pending.chmod(0o600)
                    pending.replace(self.account / ".credentials.json")
                if failed:
                    self.close()

    def last_call_metadata(self):
        return copy.deepcopy(getattr(self._thread, "metadata", {}))

    def usage_totals(self):
        return dict(self._usage)

    def restore_usage(self, usage):
        self._usage = {key: usage.get(key, 0) for key in self._usage}
