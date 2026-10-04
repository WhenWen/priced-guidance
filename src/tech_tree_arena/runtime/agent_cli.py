"""Persistent Claude Code / Codex CLI sessions for experimental Oracle runs.

The CLI process is invoked once per Arena turn.  Conversation state lives in the
CLI's persisted session and is addressed by the session ID returned from the
first turn.  The complete JSONL event stream is returned to ReplayableServices,
which makes it part of the private service tape.
"""

from __future__ import annotations

import copy
import json
import os
import shutil
import subprocess
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any


_BACKENDS = {"claude-code", "codex"}
_EFFORTS = {"low", "medium", "high", "xhigh", "max"}



class AgentCLIBackend:
    """One persistent, tool-restricted CLI conversation exposed turn by turn."""

    def __init__(
        self,
        backend: str,
        *,
        working_directory: str | Path,
        executable: str | None = None,
        model: str | None = None,
        reasoning_effort: str = "high",
        timeout_seconds: float = 600.0,
        max_budget_usd_per_turn: float | None = None,
        max_output_bytes: int = 16 * 1024 * 1024,
        anthropic_proxy_base_url: str | None = None,
        fork_inherited_sessions: bool = False,
    ) -> None:
        if backend not in _BACKENDS:
            raise ValueError(f"unknown agent CLI backend {backend!r}")
        if reasoning_effort not in _EFFORTS:
            raise ValueError(f"unsupported agent reasoning effort {reasoning_effort!r}")
        if timeout_seconds <= 0:
            raise ValueError("agent CLI timeout must be positive")
        if max_budget_usd_per_turn is not None and max_budget_usd_per_turn <= 0:
            raise ValueError("agent CLI per-turn budget must be positive")

        requested = executable or ("claude" if backend == "claude-code" else "codex")
        resolved = shutil.which(requested)
        if resolved is None:
            raise RuntimeError(
                f"could not find {backend} executable {requested!r}; "
                "install it or pass --oracle-agent-executable"
            )
        self.backend = backend
        self.executable = resolved
        self.model = (model or "").strip() or None
        self.reasoning_effort = reasoning_effort
        self.timeout_seconds = float(timeout_seconds)
        self.max_budget_usd_per_turn = max_budget_usd_per_turn
        self.max_output_bytes = int(max_output_bytes)
        if anthropic_proxy_base_url is not None and backend != "codex":
            raise ValueError("the Anthropic responses proxy applies to the codex backend only")
        self.anthropic_proxy_base_url = anthropic_proxy_base_url
        self.working_directory = Path(working_directory).resolve()
        self.working_directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        os.chmod(self.working_directory, 0o700)
        self.cli_version = self._read_version()
        self._usage: dict[str, Any] = {
            "input_tokens": 0,
            "cached_input_tokens": 0,
            "cache_write_input_tokens": 0,
            "cache_write_5m_input_tokens": 0,
            "cache_write_1h_input_tokens": 0,
            "output_tokens": 0,
            "cost_usd": 0.0,
            "turns": 0,
        }
        # `codex exec resume` reports usage accumulated across the whole Codex
        # thread.  Keep the last provider snapshot per thread so the Arena
        # meter records a turn delta instead of summing cumulative totals.
        self._codex_session_usage: dict[str, dict[str, Any]] = {}
        self._pending_codex_usage_baseline: dict[str, Any] | None = None
        self._last_metadata: dict[str, Any] = {}
        # Codex threads allow one active writer, so a resumed run must not
        # write into the source run's thread: with fork_inherited_sessions a
        # session id this process did not create is forked (a pure local
        # history copy, no model call) and the fork is resumed instead. The
        # participant-visible id changes to the forked id on the next
        # response, so the alias is only needed for the first live turn.
        self.fork_inherited_sessions = bool(fork_inherited_sessions)
        self._codex_sessions_created_here: set[str] = set()
        self._codex_session_aliases: dict[str, str] = {}

    def _read_version(self) -> str:
        try:
            completed = subprocess.run(
                [self.executable, "--version"],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                timeout=10,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired):
            return "unknown"
        return completed.stdout.strip()[:500] or "unknown"

    def last_call_metadata(self) -> dict[str, Any]:
        return copy.deepcopy(self._last_metadata)

    def usage_totals(self) -> dict[str, Any]:
        return copy.deepcopy(self._usage)

    def restore_usage(self, usage: dict[str, Any]) -> None:
        for key in self._usage:
            if key in usage:
                self._usage[key] = (
                    float(usage[key]) if key == "cost_usd" else int(usage[key])
                )
        if self.backend == "codex":
            # Backward-compatible recovery for checkpoints written before
            # per-session snapshots were persisted.  An AgentCLIBackend serves
            # one active Oracle thread, so its aggregate token totals are also
            # the best available baseline for the first resumed call.
            self._pending_codex_usage_baseline = self._usage_snapshot(self._usage)

    def export_state(self) -> dict[str, Any]:
        return {
            "usage": self.usage_totals(),
            "codex_session_usage": copy.deepcopy(self._codex_session_usage),
        }

    def restore_state(self, state: dict[str, Any]) -> None:
        usage = state.get("usage")
        if isinstance(usage, dict):
            self.restore_usage(usage)
        snapshots = state.get("codex_session_usage")
        if isinstance(snapshots, dict):
            self._codex_session_usage = {
                str(session_id): self._usage_snapshot(value)
                for session_id, value in snapshots.items()
                if isinstance(session_id, str) and isinstance(value, dict)
            }
            self._pending_codex_usage_baseline = None

    def turn(
        self,
        *,
        system_prompt: str,
        user: str,
        schema: dict[str, Any],
        schema_name: str,
        session_id: str | None = None,
    ) -> dict[str, Any]:
        if not isinstance(system_prompt, str) or not system_prompt.strip():
            raise ValueError("agent turn requires a non-empty system prompt")
        if not isinstance(user, str) or not user.strip():
            raise ValueError("agent turn requires a non-empty user message")
        if not isinstance(schema, dict):
            raise TypeError("agent turn schema must be an object")
        if not isinstance(schema_name, str) or not schema_name:
            raise ValueError("agent turn requires a schema name")
        if session_id is not None and (not isinstance(session_id, str) or not session_id):
            raise ValueError("agent session ID must be a non-empty string")

        started = time.monotonic()
        # Never let a failed turn inherit the previous turn's metadata: a
        # stale usage delta on an errored service event double-counts spend
        # and breaks replay reconciliation against the durable journals.
        self._last_metadata = {}
        try:
            if self.backend == "claude-code":
                response = self._claude_turn(system_prompt, user, schema, session_id)
            else:
                response = self._codex_turn(system_prompt, user, schema, session_id)
        except BaseException as exc:
            if not self._last_metadata:
                # _reject_tool_use records its own violation metadata; every
                # other failure gets a truthful zero-usage record.
                self._last_metadata = {
                    "agent_backend": self.backend,
                    "cli_version": self.cli_version,
                    "model": self.model,
                    "latency_s": time.monotonic() - started,
                    "error_type": type(exc).__name__,
                    "usage": {},
                }
            raise
        latency = time.monotonic() - started
        raw_usage = response.get("usage") or {}
        if self.backend == "codex":
            response["cumulative_usage"] = copy.deepcopy(raw_usage)
            usage = self._codex_usage_delta(
                str(response.get("session_id") or session_id or ""),
                raw_usage,
                is_resume=session_id is not None,
            )
            # The agentic Generator reports one logical model turn for the
            # main plus each audited child. Ordinary Codex Oracle responses
            # omit this field (or report one), preserving historical usage.
            usage["turns"] = max(1, int(raw_usage.get("turns", 1)))
            if self.anthropic_proxy_base_url is not None and not usage.get("cost_usd"):
                # Codex reports no dollar cost; through the Anthropic adapter
                # the tokens are real API spend, so price them with Anthropic's
                # exact sheet, each class at its own published rate. A
                # persistent session is cache-read dominated, so the full-rate
                # convention used for per-call API accounting would overstate
                # the oracle role's spend severalfold and distort its
                # fail-closed budget; conversely the 1h cache writes this
                # adapter defaults to cost 2x base, not the 1.25x a 5m write
                # costs, so the two write classes are billed separately rather
                # than collapsed. Agent usage is not bound by the
                # provider-attempt reconciler, so the exact sheet is safe here.
                from .provider_client import anthropic_turn_cost

                total_input = int(usage.get("input_tokens", 0))
                cache_read = int(usage.get("cached_input_tokens", 0))
                cache_write = int(usage.get("cache_write_input_tokens", 0))
                write_5m = int(usage.get("cache_write_5m_input_tokens", 0))
                write_1h = int(usage.get("cache_write_1h_input_tokens", 0))
                if write_5m + write_1h != cache_write:
                    # The adapter splits the two write classes, but the CLI in
                    # between only has to relay the flat total.  Charge whatever
                    # the split does not account for at the 1h rate -- the TTL
                    # the adapter defaults to and the pricier of the two -- so a
                    # dropped field can never make cache writes free.
                    write_1h = max(0, cache_write - write_5m)
                    write_5m = min(write_5m, cache_write)
                usage["cost_usd"] = anthropic_turn_cost(
                    self.model or "",
                    uncached_input_tokens=max(
                        0, total_input - cache_read - cache_write
                    ),
                    cache_read_input_tokens=cache_read,
                    cache_write_5m_input_tokens=write_5m,
                    cache_write_1h_input_tokens=write_1h,
                    output_tokens=int(usage.get("output_tokens", 0)),
                )
        else:
            usage = self._normalize_usage(raw_usage)
        for key in (
            "input_tokens",
            "cached_input_tokens",
            "cache_write_input_tokens",
            "cache_write_5m_input_tokens",
            "cache_write_1h_input_tokens",
            "output_tokens",
            "turns",
        ):
            self._usage[key] += int(usage.get(key, 0))
        self._usage["cost_usd"] += float(usage.get("cost_usd", 0.0))
        response["usage"] = usage
        response.update({
            "backend": self.backend,
            "model": self.model,
            "cli_version": self.cli_version,
        })
        self._last_metadata = {
            "agent_backend": self.backend,
            "cli_version": self.cli_version,
            "model": self.model,
            "session_id": response.get("session_id"),
            "latency_s": latency,
            "usage": usage,
        }
        return response

    def _claude_turn(
        self,
        system_prompt: str,
        user: str,
        schema: dict[str, Any],
        session_id: str | None,
    ) -> dict[str, Any]:
        current_session = session_id or str(uuid.uuid4())
        command = [
            self.executable,
            "-p",
            "--output-format",
            "stream-json",
            "--input-format",
            "text",
            "--verbose",
            "--safe-mode",
            "--no-chrome",
            "--permission-mode",
            "dontAsk",
            "--tools",
            "",
            "--disallowedTools",
            "mcp__*",
            "--strict-mcp-config",
            "--mcp-config",
            '{"mcpServers":{}}',
            "--max-turns",
            "1",
            "--system-prompt",
            system_prompt,
            "--json-schema",
            json.dumps(schema, ensure_ascii=False, separators=(",", ":")),
        ]
        if session_id is None:
            command.extend(["--session-id", current_session])
        else:
            command.extend(["--resume", session_id])
        if self.model:
            command.extend(["--model", self.model])
        command.extend(["--effort", self.reasoning_effort])
        if self.max_budget_usd_per_turn is not None:
            command.extend(["--max-budget-usd", repr(self.max_budget_usd_per_turn)])

        events, non_json, stderr = self._run_jsonl(command, user)
        self._reject_tool_use(events, stderr)
        result_event = next(
            (event for event in reversed(events) if event.get("type") == "result"),
            None,
        )
        if result_event is None:
            raise RuntimeError("Claude Code returned no result event")
        output = result_event.get("structured_output")
        if not isinstance(output, dict):
            output = _parse_json_object(result_event.get("result"))
        if not isinstance(output, dict):
            output = _last_assistant_json(events)
        if not isinstance(output, dict):
            raise RuntimeError("Claude Code returned no structured Oracle action")
        returned_session = str(result_event.get("session_id") or current_session)
        usage = dict(result_event.get("usage") or {})
        if result_event.get("total_cost_usd") is not None:
            usage["cost_usd"] = result_event["total_cost_usd"]
        usage["turns"] = 1
        return {
            "session_id": returned_session,
            "output": output,
            "reasoning": _reasoning_blocks(events, output),
            "raw_events": events,
            "raw_non_json": non_json,
            "stderr": stderr,
            "usage": usage,
        }

    def _codex_turn(
        self,
        system_prompt: str,
        user: str,
        schema: dict[str, Any],
        session_id: str | None,
    ) -> dict[str, Any]:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            suffix=".schema.json",
            prefix="oracle-",
            dir=self.working_directory,
            delete=False,
        ) as schema_file:
            json.dump(schema, schema_file, ensure_ascii=False, separators=(",", ":"))
            schema_path = Path(schema_file.name)
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            suffix=".output.json",
            prefix="oracle-",
            dir=self.working_directory,
            delete=False,
        ) as output_file:
            output_path = Path(output_file.name)
        try:
            command = [
                self.executable,
                "-a",
                "never",
                "-s",
                "read-only",
                "-C",
                str(self.working_directory),
                "exec",
            ]
            if session_id is not None:
                command.append("resume")
            command.extend([
                "--json",
                "--skip-git-repo-check",
                "--ignore-user-config",
                "--ignore-rules",
                "--output-schema",
                str(schema_path),
                "--output-last-message",
                str(output_path),
                "-c",
                "developer_instructions=" + json.dumps(system_prompt, ensure_ascii=False),
                "-c",
                "model_reasoning_effort=" + json.dumps(self.reasoning_effort),
            ])
            if self.anthropic_proxy_base_url is not None:
                # Route the codex session through the local OpenAI-Responses ->
                # Anthropic Messages adapter so the Oracle model is Claude.
                command.extend([
                    "-c", 'model_providers.arena_anthropic.name="Arena Anthropic Adapter"',
                    "-c", "model_providers.arena_anthropic.base_url="
                    + json.dumps(self.anthropic_proxy_base_url),
                    "-c", 'model_providers.arena_anthropic.env_key="ANTHROPIC_API_KEY"',
                    "-c", 'model_providers.arena_anthropic.wire_api="responses"',
                    "-c", 'model_provider="arena_anthropic"',
                ])
            if self.model:
                command.extend(["--model", self.model])
            resume_id = session_id
            if session_id is not None and self.backend == "codex":
                resume_id = self._resolve_codex_session(session_id)
            if resume_id is not None:
                command.extend([resume_id, "-"])
            else:
                command.append("-")

            events, non_json, stderr = self._run_jsonl(command, user)
            self._reject_tool_use(events, stderr)
            thread_id = _codex_thread_id(events) or resume_id
            if not thread_id:
                raise RuntimeError("Codex returned no thread ID")
            try:
                output_text = output_path.read_text(encoding="utf-8").strip()
            except OSError:
                output_text = ""
            output = _parse_json_object(output_text) or _last_assistant_json(events)
            if not isinstance(output, dict):
                raise RuntimeError("Codex returned no structured Oracle action")
            usage = _codex_usage(events)
            usage["turns"] = 1
            self._codex_sessions_created_here.add(str(thread_id))
            return {
                "session_id": thread_id,
                "output": output,
                "reasoning": _reasoning_blocks(events, output),
                "raw_events": events,
                "raw_non_json": non_json,
                "stderr": stderr,
                "usage": usage,
            }
        finally:
            schema_path.unlink(missing_ok=True)
            output_path.unlink(missing_ok=True)

    def _resolve_codex_session(self, session_id: str) -> str:
        """Map an inherited session onto a private fork before writing to it."""

        alias = self._codex_session_aliases.get(session_id)
        if alias is not None:
            return alias
        if (
            not self.fork_inherited_sessions
            or session_id in self._codex_sessions_created_here
        ):
            return session_id
        forked = self._fork_codex_session(session_id)
        self._codex_session_aliases[session_id] = forked
        self._codex_sessions_created_here.add(forked)
        baseline = self._codex_session_usage.get(session_id)
        if baseline is not None:
            # If the forked thread reports cumulative usage including the
            # copied history this yields the true delta; if it restarts at
            # zero the negative-delta guard falls back to the raw turn usage.
            self._codex_session_usage.setdefault(forked, copy.deepcopy(baseline))
        return forked

    def _fork_codex_session(self, session_id: str) -> str:
        command = [
            self.executable,
            "-a",
            "never",
            "-s",
            "read-only",
            "-C",
            str(self.working_directory),
            "exec",
            "fork",
            "--json",
            "--skip-git-repo-check",
            "--ignore-user-config",
            "--ignore-rules",
        ]
        if self.anthropic_proxy_base_url is not None:
            command.extend([
                "-c", 'model_providers.arena_anthropic.name="Arena Anthropic Adapter"',
                "-c", "model_providers.arena_anthropic.base_url="
                + json.dumps(self.anthropic_proxy_base_url),
                "-c", 'model_providers.arena_anthropic.env_key="ANTHROPIC_API_KEY"',
                "-c", 'model_providers.arena_anthropic.wire_api="responses"',
                "-c", 'model_provider="arena_anthropic"',
            ])
        if self.model:
            command.extend(["--model", self.model])
        command.append(session_id)
        events, _, stderr = self._run_jsonl(command, "")
        forked = _codex_thread_id(events)
        if not forked or forked == session_id:
            raise RuntimeError(
                f"codex failed to fork session {session_id}: {stderr[-2000:]}"
            )
        return forked

    def _run_jsonl(
        self, command: list[str], user: str
    ) -> tuple[list[dict[str, Any]], list[str], str]:
        try:
            completed = subprocess.run(
                command,
                input=user,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                cwd=self.working_directory,
                timeout=self.timeout_seconds,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise TimeoutError(
                f"{self.backend} Oracle turn exceeded {self.timeout_seconds:.0f}s"
            ) from exc
        except OSError as exc:
            raise RuntimeError(f"could not launch {self.backend} Oracle CLI") from exc
        size = len(completed.stdout.encode("utf-8")) + len(completed.stderr.encode("utf-8"))
        if size > self.max_output_bytes:
            raise RuntimeError(f"{self.backend} Oracle CLI output exceeded the size limit")
        events: list[dict[str, Any]] = []
        non_json: list[str] = []
        for line in completed.stdout.splitlines():
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                non_json.append(line)
                continue
            if isinstance(value, dict):
                events.append(value)
            else:
                non_json.append(line)
        if completed.returncode != 0:
            detail = (completed.stderr or "\n".join(non_json) or completed.stdout).strip()
            raise RuntimeError(
                f"{self.backend} Oracle CLI exited {completed.returncode}: {detail[-4000:]}"
            )
        return events, non_json, completed.stderr

    def _reject_tool_use(self, events: list[dict[str, Any]], stderr: str) -> None:
        violations = _tool_use_blocks(events)
        if not violations:
            return
        self._last_metadata = {
            "agent_backend": self.backend,
            "cli_version": self.cli_version,
            "model": self.model,
            "tool_policy_violation": violations,
            "raw_events": events,
            "stderr": stderr,
        }
        kinds = ", ".join(sorted({str(row.get("type")) for row in violations}))
        raise RuntimeError(f"{self.backend} Oracle attempted forbidden tool use: {kinds}")

    @staticmethod
    def _normalize_usage(raw: dict[str, Any]) -> dict[str, Any]:
        def number(*names: str) -> int:
            for name in names:
                if raw.get(name) is not None:
                    return int(raw[name])
            return 0

        return {
            "input_tokens": number("input_tokens", "inputTokens"),
            "cached_input_tokens": number(
                "cached_input_tokens", "cache_read_input_tokens", "cachedInputTokens"
            ),
            "cache_write_input_tokens": number(
                "cache_write_input_tokens",
                "cache_creation_input_tokens",
                "cacheWriteInputTokens",
            ),
            "cache_write_5m_input_tokens": number("cache_write_5m_input_tokens"),
            "cache_write_1h_input_tokens": number("cache_write_1h_input_tokens"),
            "output_tokens": number("output_tokens", "outputTokens"),
            "cost_usd": float(raw.get("cost_usd") or raw.get("total_cost_usd") or 0.0),
            "turns": number("turns") or 1,
        }

    @classmethod
    def _usage_snapshot(cls, raw: dict[str, Any]) -> dict[str, Any]:
        snapshot = cls._normalize_usage(raw)
        snapshot.pop("turns", None)
        return snapshot

    def _codex_usage_delta(
        self, session_id: str, raw: dict[str, Any], *, is_resume: bool
    ) -> dict[str, Any]:
        current = self._usage_snapshot(raw)
        previous = self._codex_session_usage.get(session_id)
        if (
            previous is None
            and is_resume
            and self._pending_codex_usage_baseline is not None
        ):
            previous = self._pending_codex_usage_baseline
            self._pending_codex_usage_baseline = None
        previous = previous or {}
        delta: dict[str, Any] = {}
        for key, value in current.items():
            old = previous.get(key, 0)
            amount = value - old
            # A provider/session reset must never produce negative accounting.
            delta[key] = value if amount < 0 else amount
        delta["turns"] = 1
        if session_id:
            self._codex_session_usage[session_id] = current
        return delta


def _parse_json_object(value: Any) -> dict[str, Any] | None:
    if isinstance(value, dict):
        return value
    if not isinstance(value, str) or not value.strip():
        return None
    text = value.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        text = "\n".join(lines[1:-1]).strip() if len(lines) >= 3 else text
        if text.startswith("json\n"):
            text = text[5:]
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def _walk_content(value: Any):
    if isinstance(value, dict):
        yield value
        for child in value.values():
            yield from _walk_content(child)
    elif isinstance(value, list):
        for child in value:
            yield from _walk_content(child)


def _last_assistant_json(events: list[dict[str, Any]]) -> dict[str, Any] | None:
    candidates: list[str] = []
    for event in events:
        for item in _walk_content(event):
            kind = str(item.get("type") or "")
            if kind in {"agent_message", "text"}:
                text = item.get("text") or item.get("content")
                if isinstance(text, str):
                    candidates.append(text)
    for candidate in reversed(candidates):
        parsed = _parse_json_object(candidate)
        if parsed is not None:
            return parsed
    return None


def _reasoning_blocks(
    events: list[dict[str, Any]], output: dict[str, Any]
) -> list[dict[str, str]]:
    blocks: list[dict[str, str]] = []
    explicit = output.get("reasoning")
    if isinstance(explicit, str) and explicit.strip():
        blocks.append({"source": "structured_output", "text": explicit.strip()})
    for event in events:
        for item in _walk_content(event):
            kind = str(item.get("type") or "")
            if kind not in {
                "thinking",
                "reasoning",
                "reasoning_summary",
                "reasoning_text",
                "summary_text",
            }:
                continue
            text = item.get("thinking") or item.get("text") or item.get("summary")
            if isinstance(text, list):
                text = "\n".join(str(part) for part in text)
            if isinstance(text, str) and text.strip():
                row = {"source": kind, "text": text.strip()}
                if row not in blocks:
                    blocks.append(row)
    return blocks


def _codex_thread_id(events: list[dict[str, Any]]) -> str | None:
    for event in events:
        if event.get("type") == "thread.started":
            value = event.get("thread_id") or event.get("session_id")
            if isinstance(value, str) and value:
                return value
    for event in events:
        value = event.get("thread_id") or event.get("session_id")
        if isinstance(value, str) and value:
            return value
    return None


def _codex_usage(events: list[dict[str, Any]]) -> dict[str, Any]:
    for event in reversed(events):
        usage = event.get("usage")
        if isinstance(usage, dict):
            return dict(usage)
    return {}


def _tool_use_blocks(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    forbidden = {
        "tool_use",
        "server_tool_use",
        "command_execution",
        "file_change",
        "mcp_tool_call",
        "web_search",
        "dynamic_tool_call",
        "collab_agent_spawn",
    }
    violations: list[dict[str, Any]] = []
    for event in events:
        for item in _walk_content(event):
            kind = str(item.get("type") or "")
            if kind not in forbidden:
                continue
            name = item.get("name") or item.get("command") or item.get("tool_name")
            # Recent Claude Code CLIs deliver --json-schema output through an
            # internal StructuredOutput tool call. That is the sanctioned
            # output channel, not an environment action.
            if kind == "tool_use" and str(name) == "StructuredOutput":
                continue
            violations.append({"type": kind, "name": name})
    return violations
