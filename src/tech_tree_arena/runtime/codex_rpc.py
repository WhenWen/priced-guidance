"""Bounded stdio app-server transport for the pinned, source-built Codex."""
from __future__ import annotations

import json
import math
import os
import queue
import signal
import subprocess
import threading
import time
from pathlib import Path
from typing import Any

from .codex_sandbox import command, environment
from .codex_prompts import CACHE_VERSIONS, DEFAULT_VERSION, instructions

ENVELOPE_SCHEMA = {"type": "object", "properties": {"payload_json": {"type": "string"}},
                   "required": ["payload_json"], "additionalProperties": False}
MAX_TASK_CONTEXT_BYTES = 32_000


def turn_timeout_seconds() -> float:
    """Host-owned native wall budget, independent of the API request timeout.

    Preserve the logical service request for replay and cache identity. Native
    reasoning turns can exceed the API adapter's 300-second request ceiling.
    """
    value = float(os.environ.get("IDEA_ARENA_CODEX_TURN_TIMEOUT_S", "1500"))
    if not math.isfinite(value) or value <= 0:
        raise ValueError("Codex turn timeout must be finite and positive")
    return value


def usage_delta(events: list[dict], parent: dict | None, thread_id: str | None = None) -> tuple[dict, dict]:
    reports = [event["params"]["tokenUsage"] for event in events
               if event.get("method") == "thread/tokenUsage/updated"
               and (thread_id is None or event["params"].get("threadId", thread_id) == thread_id)]
    if not reports:
        raise RuntimeError("Codex returned no token accounting")
    total = reports[-1]["total"]
    baseline = (parent or {}).get("usage_total") or {}
    if any(total.get(key, 0) < value for key, value in baseline.items()):
        raise RuntimeError("Codex fork reset cumulative usage unexpectedly")
    raw = {key: value - baseline.get(key, 0) for key, value in total.items()}
    return {"calls": 1, "input_tokens": int(raw["inputTokens"]),
            "cached_input_tokens": int(raw.get("cachedInputTokens", 0)),
            "cache_write_input_tokens": int(raw.get("cacheWriteInputTokens", 0)),
            "output_tokens": int(raw["outputTokens"]),
            "reasoning_output_tokens": int(raw.get("reasoningOutputTokens", 0)), "cost_usd": 0.0}, total


def standard_credit_equivalent(usage: dict) -> float:
    """Astra Standard token-rate equivalent, including cache-write premium."""
    reads = usage.get("cached_input_tokens", 0)
    writes = usage.get("cache_write_input_tokens", 0)
    plain = usage.get("input_tokens", 0) - reads - writes
    if min(plain, reads, writes) < 0:
        raise ValueError("Invalid native cache usage classes")
    return (plain * 250 + reads * 25 + writes * 312.5 + usage.get("output_tokens", 0) * 1250) / 1_000_000


class CodexRPC:
    def __init__(self, executable: Path, workspace: Path, *, timeout: float,
                 cache_key: str | None = None, trace_requests: bool = False):
        self.events: list[dict[str, Any]] = []
        self.deadline = time.monotonic() + timeout
        self.inbox: queue.Queue = queue.Queue()
        self.serial = 0
        self.size = 0
        self.stderr = (workspace / "stderr.log").open("wb")
        env = environment(workspace)
        if cache_key is not None:
            if len(cache_key) != 64 or any(c not in "0123456789abcdef" for c in cache_key):
                raise ValueError("Invalid broker-owned cache lineage")
            env["IDEA_ARENA_PROMPT_CACHE_KEY"] = cache_key
        if trace_requests:
            env["IDEA_ARENA_TRACE_PROMPT"] = "1"
        self.process = subprocess.Popen(
            command(workspace, executable, ["app-server"]),
            cwd=workspace, env=env, stdin=subprocess.PIPE,
            stdout=subprocess.PIPE, stderr=self.stderr, start_new_session=True,
        )
        self.reader = threading.Thread(target=self._read, daemon=True)
        self.reader.start()

    def _read(self):
        try:
            while line := self.process.stdout.readline(32 * 1024 * 1024 + 1):
                self.size += len(line)
                if self.size > 32 * 1024 * 1024:
                    raise RuntimeError("Codex event stream exceeds 32 MiB")
                self.inbox.put(json.loads(line))
        except BaseException as exc:
            self.inbox.put(exc)
        finally:
            self.inbox.put(RuntimeError("Codex app-server closed its output"))

    def send(self, value):
        self.process.stdin.write((json.dumps(value, ensure_ascii=False) + "\n").encode())
        self.process.stdin.flush()

    def next(self):
        remaining = self.deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("Codex generator turn timed out")
        try:
            value = self.inbox.get(timeout=remaining)
        except queue.Empty as exc:
            raise TimeoutError("Codex generator turn timed out") from exc
        if isinstance(value, BaseException):
            raise value
        if "method" in value and "id" in value:
            # No tool, approval, auth-refresh callback, or arbitrary capability
            # can turn this trusted broker into a data access service.
            self.send({"id": value["id"], "error": {"code": -32601, "message": "Generator capability disabled"}})
            raise RuntimeError("Codex attempted a disabled client capability")
        self.events.append(value)
        if value.get("method") == "model/rerouted":
            raise RuntimeError("Codex rerouted to a different model")
        if value.get("method") == "item/started":
            item = value.get("params", {}).get("item", {})
            if item.get("type") not in {"userMessage", "agentMessage", "reasoning", "contextCompaction"}:
                raise RuntimeError("Codex attempted a disabled tool")
        return value

    def call(self, method, params):
        self.serial += 1
        request_id = self.serial
        self.send({"id": request_id, "method": method, "params": params})
        while True:
            value = self.next()
            if value.get("id") == request_id:
                if "error" in value:
                    raise RuntimeError(f"Codex {method} failed: {value['error'].get('message', 'unknown error')}")
                return value["result"]

    def close(self):
        if self.process.stdin and not self.process.stdin.closed:
            self.process.stdin.close()
        try:
            self.process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            os.killpg(self.process.pid, signal.SIGTERM)
            try:
                self.process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                os.killpg(self.process.pid, signal.SIGKILL)
                self.process.wait()
        self.stderr.close()


def run_turn(executable: Path, workspace: Path, request: dict, parent: dict | None,
             prompt_version: str = DEFAULT_VERSION, *, cache_key: str | None = None,
             trace_requests: bool = False, live_rpc: CodexRPC | None = None,
             live_thread: dict | None = None):
    cached_transport = prompt_version in CACHE_VERSIONS
    if cached_transport and not cache_key:
        raise ValueError("Cache-aware transport requires a broker-owned lineage")
    task = (request["developer"] + "\n\nCurrent payload JSON Schema (" + request.get("schema_name", "payload")
            + "):\n" + json.dumps(request["schema"], ensure_ascii=False, sort_keys=True)
            + "\nSerialize the entire matching payload in the envelope field payload_json.")
    if cached_transport and len(task.encode("utf-8")) > MAX_TASK_CONTEXT_BYTES:
        raise ValueError("Codex task developer context exceeds 32,000 bytes; refusing to truncate the schema")
    timeout = turn_timeout_seconds()
    rpc = live_rpc or CodexRPC(executable, workspace, timeout=timeout,
                               cache_key=cache_key, trace_requests=trace_requests)
    rpc.deadline = time.monotonic() + timeout
    rpc.events = []
    rpc.size = 0
    reconnecting = live_rpc is not None and not getattr(rpc, "initialized", False)
    thread = None
    try:
        if not getattr(rpc, "initialized", False):
            rpc.call("initialize", {"clientInfo": {"name": "idea_arena", "version": "0.1.0"},
                                    "capabilities": {"experimentalApi": True}})
            rpc.send({"method": "initialized"})
            rpc.initialized = True
        params = {"model": request["model"], "cwd": str(workspace),
                  "approvalPolicy": "never", "sandbox": "read-only",
                  "baseInstructions": instructions(prompt_version),
                  "developerInstructions": ("Follow the latest idea_arena_call developer message."
                                            if cached_transport else request["developer"])}
        if live_thread is not None:
            if parent is None or live_thread["id"] != parent["thread_id"]:
                raise ValueError("Live Codex thread does not match its checkpoint")
            result = {"model": request["model"], "thread": live_thread}
            action = "continue"
        elif parent and reconnecting:
            result = rpc.call("thread/resume", {**params, "threadId": parent["thread_id"],
                "path": str(workspace / "parent.jsonl"), "excludeTurns": True})
            action = "resume"
        elif parent:
            result = rpc.call("thread/fork", {**params, "threadId": parent["thread_id"],
                "path": str(workspace / "parent.jsonl"), "lastTurnId": parent["turn_id"],
                "excludeTurns": True})
            action = "fork"
        else:
            result = rpc.call("thread/start", {**params, "environments": [],
                                               "selectedCapabilityRoots": [], "dynamicTools": [],
                                               "historyMode": "legacy"})
            action = "start"
        if result.get("model") != request["model"]:
            raise RuntimeError("Codex substituted a different model")
        thread = result["thread"]
        started = rpc.call("turn/start", {"threadId": thread["id"], "environments": [],
                    "input": [{"type": "text", "text": request["user"]}],
                    "model": request["model"], "effort": request.get("reasoning_effort", "high"),
                    "outputSchema": ENVELOPE_SCHEMA if cached_transport else request["schema"],
                    **({"additionalContext": {"idea_arena_call": {"kind": "application", "value": task}}}
                       if cached_transport else {})})
        turn_id = started["turn"]["id"]
        # The server may complete a tiny turn before the turn/start response.
        finished = next((event for event in rpc.events if event.get("method") == "turn/completed"
                         and event.get("params", {}).get("turn", {}).get("id") == turn_id), None)
        while finished is None:
            event = rpc.next()
            if event.get("method") == "turn/completed" and event.get("params", {}).get("turn", {}).get("id") == turn_id:
                finished = event
        if finished["params"]["turn"]["status"] != "completed":
            raise RuntimeError("Codex generator turn failed: " + json.dumps(finished["params"]["turn"].get("error")))
        candidates = [event["params"]["item"]["text"] for event in rpc.events
            if event.get("method") == "item/completed"
            and event.get("params", {}).get("item", {}).get("type") == "agentMessage"]
        if not candidates:
            candidates = [item["text"] for item in finished["params"]["turn"].get("items", []) if item.get("type") == "agentMessage"]
        # Forks inherit cumulative usage with native history. Charge only the
        # new suffix, including any model-side compaction within this turn.
        usage, total = usage_delta(rpc.events, parent, thread["id"])
        output = json.loads(candidates[-1])
        if cached_transport:
            from .provider_client import _validate_json_schema
            _validate_json_schema(output, ENVELOPE_SCHEMA)
            output = json.loads(output["payload_json"])
            _validate_json_schema(output, request["schema"])
        if not isinstance(output, dict):
            raise RuntimeError("Codex structured response must be an object")
        return {"output": output, "usage": usage, "events": rpc.events,
                "thread_id": thread["id"], "turn_id": turn_id, "rollout": thread.get("path"),
                "usage_total": total, "thread_action": action}
    except Exception as exc:
        exc.codex_events = rpc.events
        try:
            exc.codex_usage, _ = usage_delta(rpc.events, parent, thread["id"] if thread else None)
        except RuntimeError:
            exc.codex_usage = {}
        raise
    finally:
        if live_rpc is None:
            rpc.close()
