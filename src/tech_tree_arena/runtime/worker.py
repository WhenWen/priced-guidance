"""Participant-side worker for the local JSON actor protocol."""

from __future__ import annotations

import importlib
import itertools
import json
import queue
import signal
import sys
import threading
import traceback
from pathlib import Path
from typing import Any

from .wire import decode_message, encode_message


class RemoteServices:
    """Thread-safe service facade over the id-multiplexed wire protocol.

    Participant code may call services from concurrent threads (for example
    to build several dispatch previews at once). Each request carries a unique
    id; a single reader thread routes broker responses back to their waiters
    and forwards non-service frames (step/close) to the main loop.
    """

    supports_concurrent_calls = True

    def __init__(self, protocol_out: Any, public_resources: dict[str, Any]) -> None:
        self._out = protocol_out
        self.public_resources = public_resources
        if (public_resources.get("generator_memory_policy") or {}).get("deterministic_calls"):
            self.supports_concurrent_calls = False
        self._write_lock = threading.Lock()
        self._ids = itertools.count(1)
        self._waiters: dict[int, dict[str, Any]] = {}
        self._waiters_lock = threading.Lock()
        self._disconnected = False
        self.inbox: queue.SimpleQueue = queue.SimpleQueue()
        self._reader: threading.Thread | None = None

    def start_reader(self) -> None:
        if self._reader is not None:
            return
        self._reader = threading.Thread(target=self._read_loop, daemon=True)
        self._reader.start()

    def _read_loop(self) -> None:
        for line in sys.stdin:
            try:
                frame = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(frame, dict):
                continue
            operation = frame.get("op")
            if operation in {"service_result", "service_error"}:
                request_id = frame.get("id")
                with self._waiters_lock:
                    waiter = self._waiters.pop(request_id, None)
                if waiter is not None:
                    waiter["frame"] = frame
                    waiter["event"].set()
                continue
            self.inbox.put(frame)
        # EOF: the broker disconnected. Wake every waiter and the main loop.
        with self._waiters_lock:
            self._disconnected = True
            waiters = list(self._waiters.values())
            self._waiters.clear()
        for waiter in waiters:
            waiter["frame"] = None
            waiter["event"].set()
        self.inbox.put(None)

    def _call(self, method: str, request: dict[str, Any]) -> Any:
        request_id = next(self._ids)
        waiter: dict[str, Any] = {"event": threading.Event(), "frame": None}
        with self._waiters_lock:
            if self._disconnected:
                raise RuntimeError("arena service broker disconnected")
            self._waiters[request_id] = waiter
        with self._write_lock:
            self._out.write(json.dumps(
                {"op": "service", "id": request_id, "method": method, "request": request},
                ensure_ascii=False, separators=(",", ":"),
            ) + "\n")
            self._out.flush()
        if self._reader is None:
            # Initialization happens before the reader thread exists; consume
            # the single synchronous response inline.
            line = sys.stdin.readline()
            if not line:
                raise RuntimeError("arena service broker disconnected")
            waiter["frame"] = json.loads(line)
        else:
            waiter["event"].wait()
        response = waiter["frame"]
        if response is None:
            raise RuntimeError("arena service broker disconnected")
        if response.get("op") == "service_error":
            error_type = str(response.get("error_type") or response.get("error") or "ServiceError")
            detail = str(response.get("message") or "arena service call failed")
            raise RuntimeError(f"{error_type}: {detail}")
        if response.get("op") != "service_result":
            raise RuntimeError("arena service broker returned an invalid frame")
        return response.get("result")

    def structured_model(self, **request: Any) -> Any:
        return self._call("structured_model", request)

    def agent_turn(self, **request: Any) -> Any:
        return self._call("agent_turn", request)

    def random(self) -> float:
        return float(self._call("random", {}))

    def randint(self, start: int, stop: int) -> int:
        return int(self._call("randint", {"start": start, "stop": stop}))

    def judge_evaluate(self, ideas: Any) -> Any:
        return self._call("judge_evaluate", {"ideas": ideas})

    def choice(self, values: list[Any] | tuple[Any, ...]) -> Any:
        return self._call("choice", {"values": list(values)})


def _entrypoint(root: Path, value: str) -> type:
    module_name, separator, attribute = value.partition(":")
    if not separator:
        raise ValueError("invalid entrypoint")
    sys.path.insert(0, str(root))
    module = importlib.import_module(module_name)
    loaded = getattr(module, attribute)
    if not isinstance(loaded, type):
        raise TypeError("entrypoint is not a class")
    return loaded


def _ignore_parent_owned_signals() -> None:
    """Let the Arena parent turn terminal interrupts into durable pauses."""

    # Shells and interactive terminals deliver SIGTERM/SIGINT to the whole process
    # group. The worker must not die mid-step from that broadcast -- the
    # parent pauses at its engine-loop boundary and then closes workers
    # deliberately; an uncoordinated worker death finalizes the run as
    # participant_failure with a torn usage summary.
    for signum in (signal.SIGTERM, signal.SIGINT):
        try:
            signal.signal(signum, signal.SIG_IGN)
        except (ValueError, OSError):
            pass


def main() -> int:
    _ignore_parent_owned_signals()
    protocol_out = sys.stdout
    # Keep ordinary participant prints out of the framing channel.
    sys.stdout = sys.stderr
    first = sys.stdin.readline()
    if not first:
        return 1
    try:
        init = json.loads(first)
        if init.get("op") != "init":
            raise ValueError("expected init")
        services = RemoteServices(protocol_out, init.get("public_resources") or {})
        for dependency_path in reversed(init.get("dependency_paths") or []):
            sys.path.insert(0, str(dependency_path))
        actor_class = _entrypoint(Path(init["root"]), init["entrypoint"])
        actor = actor_class(*(init.get("constructor_args") or []), services)
        if not callable(getattr(actor, "step", None)):
            raise TypeError("actor has no step method")
        protocol_out.write('{"op":"ready"}\n')
        protocol_out.flush()
    except Exception as exc:  # noqa: BLE001
        traceback.print_exc(file=sys.stderr)
        protocol_out.write(json.dumps({
            "op": "error",
            "phase": "initialization",
            "error_type": type(exc).__name__,
            "message": str(exc),
        }, ensure_ascii=False, separators=(",", ":")) + "\n")
        protocol_out.flush()
        return 1

    services.start_reader()
    while True:
        frame = services.inbox.get()
        if frame is None:
            return 0
        try:
            if frame.get("op") == "close":
                return 0
            if frame.get("op") != "step":
                raise ValueError("expected step")
            output = actor.step(decode_message(frame["message"]))
            response = {"op": "result", "message": encode_message(output)}
        except Exception as exc:  # noqa: BLE001
            traceback.print_exc(file=sys.stderr)
            response = {
                "op": "error",
                "phase": "step",
                "error_type": type(exc).__name__,
                "message": str(exc),
            }
        with services._write_lock:
            protocol_out.write(json.dumps(response, ensure_ascii=False, separators=(",", ":")) + "\n")
            protocol_out.flush()
if __name__ == "__main__":
    raise SystemExit(main())
