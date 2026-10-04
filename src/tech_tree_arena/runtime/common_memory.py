"""Provider-neutral, reasoning-aware self-compaction outside Codex.

The participant protocol and Question artifacts are untouched. One logical
service call may contain a summary plus a new-window call; both are metered and
recorded, while checkpoints bind only fully committed logical calls.
"""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import threading
import time
from typing import Any

from ..errors import ReplayDivergence
from ..replay.artifacts import ArtifactStore
from .native_api_generator import NativeAPIGeneratorBackend, MODELS as NATIVE_MODELS, add_usage, digest
from .services import _request_hash

BACKEND = "common-memory-v1"
SUMMARY_PROMPT = """Create a concise continuation memory for yourself at a context-window boundary.
Summarize the task state established in the conversation: results, decisions, and pending work.
Record working hypotheses, conclusions and their evidence references, rejected paths,
unresolved questions, and next steps. Keep proposals distinct from confirmed Oracle
choices. Unselected eager previews are hypotheses, never confirmation or rejection.
The supplied current public state and immutable Question artifacts remain authoritative.
Do not solve a new task, generate new candidate slates, or add scientific claims.
Return only the requested JSON payload containing memory. This memory will replace
the older conversation in a fresh window. Preserve useful task knowledge concisely.
"""
SEED_PREFIX = """Continuation memory from your previous context window follows. Treat it as
fallible working memory. The current call's public state and exact cached Questions
are authoritative. Speculation is not Oracle evidence.
"""


def configuration(model: str, *, effort: str | None = None, compact_tokens: int = 100_000,
                  compact_calls: int = 0, memory_chars: int = 12_000,
                  timeout: float = 1500, attempts: int = 3, codex: dict | None = None) -> dict:
    if model not in {"gpt-6-astra", "gpt-5.6-sol", *NATIVE_MODELS}:
        raise ValueError("Unsupported common-memory model")
    effort = effort or ("xhigh" if model in {"gpt-6-astra", "gpt-5.6-sol"} else "max")
    allowed = {"low", "high", "max"} if model.startswith("together/") else {"low", "medium", "high", "xhigh", "max"}
    if effort not in allowed or compact_tokens < 1000 or compact_calls < 0 or not 256 <= memory_chars <= 64_000:
        raise ValueError("Invalid common-memory policy")
    if not 1 <= attempts <= 5 or not 1 <= timeout <= 1500:
        raise ValueError("Invalid common-memory retry/timeout policy")
    if model in {"gpt-6-astra", "gpt-5.6-sol"} and codex is None:
        from .codex_generator import configuration as codex_configuration
        codex = codex_configuration(reasoning_effort=effort)
    if model not in {"gpt-6-astra", "gpt-5.6-sol"} and codex is not None:
        raise ValueError("Only Astra and Sol use the Codex subscription transport")
    if codex is not None:
        from .codex_generator import configuration as codex_configuration
        from .codex_prompts import COMMON_VERSION
        codex = codex_configuration(codex["build_receipt"], codex["auth_home"],
                                    prompt_version=COMMON_VERSION, reasoning_effort=effort)
    sources = ("common_memory.py", "native_api_generator.py", "codex_prompts.py",
               "codex_generator.py", "codex_rpc.py", "provider_client.py")
    return {"backend": BACKEND, "model": model, "reasoning_effort": effort,
            "compact_tokens": compact_tokens, "compact_calls": compact_calls,
            "memory_chars": memory_chars, "timeout": timeout, "attempts": attempts,
            "summary_prompt_sha256": digest(SUMMARY_PROMPT), "codex": codex,
            "source_sha256": {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest() for name in sources}}


class CommonMemoryBackend:
    supports_context_chain = True
    deterministic_call_order = True

    def __init__(self, config: dict, artifacts: ArtifactStore, *, transport: Any = None):
        expected = configuration(config["model"], effort=config["reasoning_effort"],
            compact_tokens=config["compact_tokens"], compact_calls=config["compact_calls"],
            memory_chars=config["memory_chars"], timeout=config["timeout"], attempts=config["attempts"], codex=config["codex"])
        # Reviewed legacy histories remain usable with an external monetary
        # guard. The physical wire requests and prompts are unchanged.
        legacy_sources = [
            ("80c12052488aaad9a8f4bafd7d3ae121c625c8ba1b3d6097c390d0a3e778fe6f",
             "6348787b758495f6eb2ec729eeb26a335da934b8cfd8ea8b934695d80b8f837f",
             {"gpt-6-astra", "anthropic/claude-fable-5-1", "together/zai-org/GLM-5.3"}),
            ("9c446c17000e4ed3c72e973e88e572052689a410371e78330cc2bc860f9564e2",
             "01d3098bf1284f3977a7fd848f49c3896bb4e9bf464965e66501574b76b6ba1b",
             {"gpt-6-astra", "gpt-5.6-sol", *NATIVE_MODELS}),
        ]
        accepted = [expected]
        # Removing only empty unused action fields preserves idea values and
        # canonical history. Bind resume compatibility to the exact prior code.
        compatible = copy.deepcopy(expected)
        compatible["source_sha256"].update({
            "common_memory.py": "43b4a64525994c40f742d8f3b8b7ee3f424d8a00f37f801dff975c440b095043",
            "native_api_generator.py": "b8bf1e2078c6a1da1bd0d5feb6a0f82c42a09f7b6f548d6029f8d68b1f1aee48"})
        accepted.append(compatible)
        # Whole-answer JSON-fence parsing preserves the original payload and
        # all wire prompts. Accept this exact pre-fix configuration for resume.
        compatible = copy.deepcopy(expected)
        compatible["source_sha256"].update({
            "common_memory.py": "7bb7bb2edb6ce39e94f1964c519e577c77c033f31c2959d23d1b6b3c593c47db",
            "native_api_generator.py": "265d99b9a8777deda83f110f3a2d4ab3a73b0a15a6635a06dfe8d1510547effe"})
        accepted.append(compatible)
        # The next parser-only repair also recognizes the explicit payload_json
        # label; accept the exact fenced-JSON parser revision as an ancestor.
        compatible = copy.deepcopy(expected)
        compatible["source_sha256"].update({
            "common_memory.py": "1037420b6b2cde7d2f3b8caeec5b8f0bfbe7df0f0e954bfa5119dbc210435fa8",
            "native_api_generator.py": "8fb5eee1ad3a43befed3acf1891f7fb9c32b00529da136fbb5c33887d1c70044"})
        accepted.append(compatible)
        # These exact histories differ only in native artifact validation.
        for native_sha in ("d9512020920ff52d05afbbc64ae60068d56e01cc839400e54d418963b4f80de2",
                           "265d99b9a8777deda83f110f3a2d4ab3a73b0a15a6635a06dfe8d1510547effe"):
            compatible = copy.deepcopy(expected)
            compatible["source_sha256"].update({
                "common_memory.py": "a9c3b0831c7617e9bf6e254821b24e7f25af2acc2bbd778a395ea385ade3d415",
                "native_api_generator.py": native_sha})
            accepted.append(compatible)
        for common_sha, native_sha, models in legacy_sources:
            if config["model"] in models:
                compatible = copy.deepcopy(expected)
                compatible["source_sha256"].update({"common_memory.py": common_sha,
                    "native_api_generator.py": native_sha,
                    "provider_client.py": "388e7a2de1c612c716f6e638eea0df19fccda47ec2134eee13b24444e39c0239"})
                accepted.append(compatible)
        if config not in accepted:
            raise ReplayDivergence("Common-memory source or policy differs from recorded configuration")
        self.config, self.artifacts = copy.deepcopy(config), artifacts
        self.policy_hash = digest(config)
        self.reasoning_effort = config["reasoning_effort"]
        self.request_timeout_seconds = config["timeout"]
        self._lock = threading.RLock()
        self._local = threading.local()
        self._usage = {"calls": 0, "physical_calls": 0, "input_tokens": 0, "output_tokens": 0,
                       "cached_input_tokens": 0, "cost_usd": 0.0, "compactions": 0}
        if transport is not None:
            self.transport = transport
        elif config["codex"]:
            from .codex_generator import CodexGeneratorBackend
            self.transport = CodexGeneratorBackend(config["codex"], artifacts)
        else:
            self.transport = NativeAPIGeneratorBackend(config["model"], artifacts,
                effort=self.reasoning_effort, timeout=config["timeout"], attempts=1)

    def structured(self, **request):
        return self.structured_in_context(None, **request)

    def structured_in_context(self, context: dict | None, **request):
        with self._lock:
            return self._structured(context, request)

    def _structured(self, context: dict | None, request: dict):
        if request["model"] != self.config["model"]:
            raise ReplayDivergence("Common-memory model changed")
        if context and context.get("policy_sha256") != self.policy_hash:
            raise ReplayDivergence("Common-memory checkpoint policy changed")
        parent = copy.deepcopy(context)
        metadata = {"backend": BACKEND, "model": request["model"], "schema_name": request["schema_name"],
                    "request_hash": _request_hash("model.structured", request), "policy_sha256": self.policy_hash,
                    "parent_context": parent, "operations": [], "usage": {"cost_usd": 0.0}}
        self._local.metadata = metadata
        native = (parent or {}).get("native")
        calls = (parent or {}).get("window_calls", 0)
        window = (parent or {}).get("window", 0)
        summary_ref = (parent or {}).get("summary")
        active = (parent or {}).get("active_tokens", 0)
        # Conservative byte estimate for the next input; cumulative billed tokens
        # are not a context-size estimate. Keep output room before Codex's limit.
        incoming = len(json.dumps({k: request[k] for k in (
            "user", "developer", "schema", "schema_name")}, ensure_ascii=False).encode())
        should_compact = bool(native) and (
            active + incoming >= self.config["compact_tokens"]
            or self.config["compact_calls"] > 0 and calls >= self.config["compact_calls"])
        forwarded = copy.deepcopy(request)
        try:
            if should_compact:
                summary_request = {**request, "schema_name": "common_memory_summary",
                    "developer": SUMMARY_PROMPT + f"\nMemory must be at most {self.config['memory_chars']} characters.",
                    "user": "Summarize the completed work in this session for continuation.",
                    "schema": {"type": "object", "properties": {"memory": {"type": "string", "minLength": 1,
                                "maxLength": self.config["memory_chars"]}}, "required": ["memory"], "additionalProperties": False}}
                summary, _ = self._invoke(native, summary_request, "summary", metadata)
                summary_ref = self.artifacts.put_json({"memory": summary["memory"], "source_context": parent})
                forwarded["user"] = SEED_PREFIX + json.dumps(summary["memory"], ensure_ascii=False) + "\n\nCurrent call:\n" + request["user"]
                native, calls, window = None, 0, window + 1
                metadata["compaction"] = {"summary": summary_ref, "source_context": parent,
                                           "reason": "calls" if self.config["compact_calls"] and parent["window_calls"] >= self.config["compact_calls"] else "tokens"}
            output, transport_metadata = self._invoke(native, forwarded, "generate", metadata)
            native = transport_metadata.get("native_context") or transport_metadata.get("codex_context")
            if not isinstance(native, dict):
                raise RuntimeError("Native transport omitted its checkpoint")
            active = int(native.get("active_tokens", 0))
            if transport_metadata.get("backend") == "codex-source-v1":
                reports = [x["params"]["tokenUsage"] for x in transport_metadata.get("codex_events", []) if x.get("method") == "thread/tokenUsage/updated"]
                if reports:
                    last = reports[-1].get("last") or {}
                    active = int(last.get("inputTokens", 0)) + int(last.get("outputTokens", 0))
            metadata["native_context"] = {"backend": BACKEND, "model": request["model"],
                "policy_sha256": self.policy_hash, "native": native, "window": window,
                "window_calls": calls + 1, "active_tokens": active,
                "summary": summary_ref, "parent_sha256": digest(parent)}
            metadata["usage"]["calls"] = 1
            metadata["usage"]["compactions"] = int(should_compact)
            return output
        finally:
            add_usage(self._usage, metadata["usage"])

    def _invoke(self, native: dict | None, request: dict, role: str, metadata: dict):
        for attempt in range(self.config["attempts"]):
            operation = {"role": role, "request": copy.deepcopy(request), "response": None, "error": None}
            try:
                if getattr(self.transport, "budget_guard", None) is not None:
                    self.transport.budget_checkpoint = digest({"parent": metadata["parent_context"],
                        "request": metadata["request_hash"], "operation": role})
                operation["response"] = self.transport.structured_in_context(copy.deepcopy(native), **request)
                inner = self.transport.last_call_metadata()
                if any(x.get("method") in {"item/started", "item/completed"}
                       and x.get("params", {}).get("item", {}).get("type") == "contextCompaction"
                       for x in inner.get("codex_events", [])):
                    raise ReplayDivergence("Unexpected native compaction violates common-memory policy")
                if role == "summary" and (not isinstance(operation["response"].get("memory"), str)
                        or not 0 < len(operation["response"]["memory"]) <= self.config["memory_chars"]):
                    raise ValueError("Summary violates memory length limit")
                return operation["response"], inner
            except BaseException as exc:
                operation["error"] = type(exc).__name__
                if not isinstance(exc, Exception):
                    raise
                if (isinstance(exc, ReplayDivergence) or not getattr(exc, "retryable", True)
                        or attempt + 1 == self.config["attempts"]):
                    raise
            finally:
                operation["metadata"] = self.transport.last_call_metadata()
                operation["request_hash"] = _request_hash("model.structured", request)
                metadata["operations"].append(operation)
                delta = dict(operation["metadata"].get("usage") or {})
                # Logical calls and K are owned by Arena, not summary operations.
                delta.pop("calls", None)
                if operation["metadata"].get("backend") == "codex-source-v1":
                    delta["physical_calls"] = 1
                add_usage(metadata["usage"], delta)
            time.sleep(min(8, 2 ** attempt))

    def last_call_metadata(self):
        return copy.deepcopy(getattr(self._local, "metadata", {}))

    def usage_totals(self):
        with self._lock:
            return dict(self._usage)

    def restore_usage(self, usage):
        with self._lock:
            self._usage = copy.deepcopy(usage)

    def close(self):
        self.transport.close()


def validate_metadata(service: dict, artifacts: ArtifactStore | None = None) -> None:
    """Bind replay to the actual native operations, including summary overhead."""
    metadata, request = service.get("metadata") or {}, service.get("request") or {}
    if (metadata.get("backend") != BACKEND or metadata.get("model") != request.get("model")
            or metadata.get("request_hash") != _request_hash("model.structured", request)):
        raise ReplayDivergence("Common-memory service binding diverges")
    operations = metadata.get("operations") or []
    total: dict = {"cost_usd": 0.0}
    parent = metadata.get("parent_context")
    expected_native = (parent or {}).get("native")
    summaries = 0
    for op in operations:
        inner = op.get("metadata") or {}
        if op.get("request_hash") != _request_hash("model.structured", op["request"]):
            raise ReplayDivergence("Common-memory operation request changed")
        if inner.get("request_hash") != op["request_hash"] or inner.get("model") != request["model"]:
            raise ReplayDivergence("Common-memory operation binding diverges")
        if inner.get("parent_context") != expected_native:
            raise ReplayDivergence("Common-memory native ancestor changed")
        if op.get("role") not in {"summary", "generate"}:
            raise ReplayDivergence("Unknown common-memory operation role")
        if inner.get("backend") == "codex-source-v1":
            from .codex_generator import validate_metadata as validate_codex
            validate_codex(op)
        elif inner.get("backend") == "native-api-history-v1":
            if artifacts is not None:
                from .native_api_generator import validate_artifacts
                validate_artifacts(inner, op["request"], artifacts)
            attempt_usage: dict = {"cost_usd": 0.0}
            for attempt in inner.get("attempts", []):
                add_usage(attempt_usage, attempt["usage"])
            for key, value in attempt_usage.items():
                if inner.get("usage", {}).get(key) != value:
                    raise ReplayDivergence("Native API attempt accounting diverges")
        else:
            raise ReplayDivergence("Unknown common-memory transport")
        delta = dict(inner.get("usage") or {})
        delta.pop("calls", None)
        if inner.get("backend") == "codex-source-v1":
            delta["physical_calls"] = 1
        add_usage(total, delta)
        if op["role"] == "summary" and not op.get("error"):
            summaries += 1
            expected_native = None
    for key, value in total.items():
        if metadata.get("usage", {}).get(key, 0) != value:
            raise ReplayDivergence("Common-memory usage omitted a native operation")
    if not service.get("error"):
        context = metadata.get("native_context") or {}
        if not operations or operations[-1].get("role") != "generate" or operations[-1].get("error"):
            raise ReplayDivergence("Common-memory success has no completed generation")
        if context.get("policy_sha256") != metadata.get("policy_sha256"):
            raise ReplayDivergence("Common-memory context policy diverges")
        last = operations[-1]["metadata"]
        if (summaries > 1 or context.get("parent_sha256") != digest(parent)
                or context.get("native") != (last.get("native_context") or last.get("codex_context"))
                or context.get("window") != (parent or {}).get("window", 0) + summaries
                or context.get("window_calls") != (0 if summaries else (parent or {}).get("window_calls", 0)) + 1
                or metadata.get("usage", {}).get("compactions") != summaries):
            raise ReplayDivergence("Common-memory checkpoint transition diverges")
        expected = copy.deepcopy(request)
        if summaries:
            summary = next(op["response"]["memory"] for op in operations if op["role"] == "summary" and not op.get("error"))
            expected["user"] = SEED_PREFIX + json.dumps(summary, ensure_ascii=False) + "\n\nCurrent call:\n" + request["user"]
            if artifacts and artifacts.load_json(context["summary"]) != {"memory": summary, "source_context": parent}:
                raise ReplayDivergence("Summary artifact differs from native summary")
        if operations[-1]["request"] != expected:
            raise ReplayDivergence("Fresh window contains input other than summary and current state")
        if operations[-1].get("response") != service.get("response"):
            raise ReplayDivergence("Common-memory response differs from native generation")
