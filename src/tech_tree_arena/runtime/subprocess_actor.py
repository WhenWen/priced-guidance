"""JSON-RPC proxy for the convenient, explicitly unverified local runner."""

from __future__ import annotations

import json
import select
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from ..errors import ParticipantFailure, ReplayDivergence, ResourceLimitExceeded
from .services import ReplayableServices, ServiceEvent, ServiceFactory
from .wire import decode_message, encode_message, json_value

_SERVICE_WORKERS = 8
_FATAL_POLL_SECONDS = 0.25


class SubprocessActor:
    def __init__(
        self,
        *,
        root: Path,
        entrypoint: str,
        constructor_args: tuple[Any, ...],
        services: ReplayableServices,
        timeout_seconds: float = 900.0,
        dependency_paths: tuple[Path, ...] = (),
        log_path: Path | None = None,
        sandbox_generator: bool = False,
    ) -> None:
        self.services = services
        self.timeout_seconds = timeout_seconds
        self.log_path = log_path
        self._stderr_stream = log_path.open("a", encoding="utf-8") if log_path else None
        self._last_service_error: tuple[str, str] | None = None
        self._send_lock = threading.Lock()
        self._state_lock = threading.Lock()
        self._fatal_service_error: BaseException | None = None
        # Service requests are dispatched concurrently so participant threads
        # can overlap model calls; the executor is created lazily on first use.
        self._executor: ThreadPoolExecutor | None = None
        self._sandbox_directory = None
        launch: dict[str, Any] = {"args": [sys.executable, "-B", "-m", "tech_tree_arena.runtime.worker"]}
        if sandbox_generator:
            from .generator_worker_sandbox import stage_worker
            self._sandbox_directory, root, dependency_paths, launch = stage_worker(root, dependency_paths)
        self.process = subprocess.Popen(
            **launch,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=self._stderr_stream or subprocess.DEVNULL,
            text=True,
            encoding="utf-8",
            bufsize=1,
        )
        self._send({
            "op": "init",
            "root": str(root),
            "entrypoint": entrypoint,
            "constructor_args": json_value(constructor_args),
            "public_resources": json_value(services.public_resources),
            "dependency_paths": [str(path) for path in dependency_paths],
        })
        while True:
            response = self._read()
            if response.get("op") == "ready":
                break
            if response.get("op") != "service":
                self.close()
                raise ParticipantFailure(
                    "participant subprocess failed to initialize",
                    phase=str(response.get("phase") or "initialization"),
                    error_type=str(response.get("error_type") or "ParticipantError"),
                    private_detail=str(response.get("message") or ""),
                )
            # Initialization-phase services stay synchronous: the worker is
            # single-threaded until its reader starts after ready.
            self._handle_service_frame(response, synchronous=True)
            self._raise_fatal_if_set()

    def _send(self, value: dict[str, Any]) -> None:
        if self.process.stdin is None:
            raise ParticipantFailure("participant subprocess input is unavailable")
        with self._send_lock:
            self.process.stdin.write(
                json.dumps(value, ensure_ascii=False, separators=(",", ":")) + "\n"
            )
            self.process.stdin.flush()

    def _read(self, timeout: float | None = None) -> dict[str, Any]:
        """Read one frame, polling for fatal service failures while waiting."""
        if self.process.stdout is None:
            raise ParticipantFailure("participant subprocess output is unavailable")
        deadline = time.monotonic() + (
            self.timeout_seconds if timeout is None else max(timeout, 0.0)
        )
        while True:
            self._raise_fatal_if_set()
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                self.close()
                raise ParticipantFailure("participant subprocess timed out")
            ready, _, _ = select.select(
                [self.process.stdout], [], [], min(remaining, _FATAL_POLL_SECONDS)
            )
            if ready:
                break
        line = self.process.stdout.readline()
        if not line:
            self._raise_fatal_if_set()
            raise ParticipantFailure("participant subprocess terminated")
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ParticipantFailure("participant subprocess emitted an invalid frame") from exc
        if not isinstance(value, dict):
            raise ParticipantFailure("participant subprocess emitted an invalid frame")
        return value

    def _raise_fatal_if_set(self) -> None:
        with self._state_lock:
            fatal = self._fatal_service_error
        if fatal is not None:
            # Drain in-flight service tasks before the engine captures failure
            # state: every event that reaches the durable journal must also be
            # present in the actor's exported service tape. Queued-but-unstarted
            # tasks are cancelled and record nothing.
            executor = self._executor
            self._executor = None
            if executor is not None:
                executor.shutdown(wait=True, cancel_futures=True)
            self.close()
            raise fatal

    def _handle_service_frame(
        self, frame: dict[str, Any], *, synchronous: bool
    ) -> None:
        request_id = frame.get("id")
        method = str(frame.get("method"))
        request = frame.get("request") or {}
        if synchronous:
            self._execute_service(request_id, method, request)
            return
        if self._executor is None:
            self._executor = ThreadPoolExecutor(
                max_workers=_SERVICE_WORKERS,
                thread_name_prefix="arena-service",
            )
        self._executor.submit(self._execute_service, request_id, method, request)

    def _execute_service(
        self, request_id: Any, method: str, request: dict[str, Any]
    ) -> None:
        try:
            result = self._service(method, request)
        except (ReplayDivergence, ResourceLimitExceeded) as exc:
            # Fail closed exactly like the historical synchronous path: the
            # run-level failure propagates from step()/init and the worker is
            # torn down without a reply.
            with self._state_lock:
                if self._fatal_service_error is None:
                    self._fatal_service_error = exc
            return
        except Exception as exc:  # noqa: BLE001
            with self._state_lock:
                self._last_service_error = (type(exc).__name__, str(exc))
            self._send({
                "op": "service_error",
                "id": request_id,
                "error_type": type(exc).__name__,
                "message": str(exc),
            })
            return
        with self._state_lock:
            self._last_service_error = None
        self._send({"op": "service_result", "id": request_id, "result": json_value(result)})

    def step(self, message: Any) -> Any:
        self._send({"op": "step", "message": encode_message(message)})
        deadline = time.monotonic() + self.timeout_seconds
        while True:
            response = self._read(deadline - time.monotonic())
            operation = response.get("op")
            if operation == "result":
                return decode_message(response["message"])
            if operation == "error":
                self._raise_fatal_if_set()
                error_type = str(response.get("error_type") or "ParticipantError")
                detail = str(response.get("message") or "")
                with self._state_lock:
                    if self._last_service_error is not None:
                        error_type, detail = self._last_service_error
                raise ParticipantFailure(
                    "participant subprocess step failed",
                    phase=str(response.get("phase") or "step"),
                    error_type=error_type,
                    private_detail=detail,
                )
            if operation != "service":
                raise ParticipantFailure("participant subprocess used an unknown operation")
            self._handle_service_frame(response, synchronous=False)

    def _service(self, method: str, request: dict[str, Any]) -> Any:
        if method == "structured_model":
            return self.services.structured_model(**request)
        if method == "agent_turn":
            return self.services.agent_turn(**request)
        if method == "random":
            return self.services.random()
        if method == "randint":
            return self.services.randint(int(request["start"]), int(request["stop"]))
        if method == "choice":
            return self.services.choice(request["values"])
        if method == "judge_evaluate":
            return self.services.judge_evaluate(request.get("ideas"))
        raise ParticipantFailure("participant requested an unsupported service")

    def close(self) -> None:
        executor = self._executor
        self._executor = None
        if executor is not None:
            # A service call records to both the actor-local replay tape and the
            # run-wide journal.  Do not let teardown race recorder finalization:
            # drain work that already started, while cancelling requests that
            # never began, so both accounting views end at the same boundary.
            executor.shutdown(wait=True, cancel_futures=True)
        try:
            if self.process.poll() is None:
                try:
                    self._send({"op": "close"})
                    self.process.wait(timeout=2)
                except Exception:  # noqa: BLE001
                    self.process.kill()
                    self.process.wait(timeout=2)
        finally:
            # Explicitly close parent pipe wrappers.  If the worker died, an
            # implicit TextIOWrapper finalizer can otherwise try to flush its
            # broken stdin later and emit an unraisable BrokenPipeError.
            for stream in (
                self.process.stdin,
                self.process.stdout,
                self._stderr_stream,
            ):
                if stream is not None and not stream.closed:
                    try:
                        stream.close()
                    except OSError:
                        pass
            if self._sandbox_directory is not None:
                self._sandbox_directory.cleanup()
                self._sandbox_directory = None

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:  # noqa: BLE001
            pass


class SubprocessActorFactory:
    """ActorFactory-compatible entrypoint for separate local role processes."""

    def __init__(
        self,
        root: str | Path,
        entrypoint: str,
        *,
        constructor_args: tuple[Any, ...] = (),
        service_factory: ServiceFactory,
        timeout_seconds: float = 900.0,
        dependency_paths: tuple[Path, ...] = (),
        log_path: str | Path | None = None,
        sandbox_generator: bool = False,
    ) -> None:
        self.root = Path(root).resolve()
        self.entrypoint = entrypoint
        self.constructor_args = constructor_args
        self.service_factory = service_factory
        self.timeout_seconds = timeout_seconds
        self.dependency_paths = dependency_paths
        self.log_path = Path(log_path) if log_path is not None else None
        self.sandbox_generator = sandbox_generator

    def create(self, replay_tape: tuple[ServiceEvent, ...] = ()) -> tuple[SubprocessActor, ReplayableServices]:
        services = self.service_factory.create(replay_tape)
        actor = SubprocessActor(
            root=self.root,
            entrypoint=self.entrypoint,
            constructor_args=self.constructor_args,
            services=services,
            timeout_seconds=self.timeout_seconds,
            dependency_paths=self.dependency_paths,
            log_path=self.log_path,
            sandbox_generator=self.sandbox_generator,
        )
        return actor, services
