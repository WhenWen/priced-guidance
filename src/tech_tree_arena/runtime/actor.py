"""Opaque stateful actors with deterministic reconstruction checkpoints."""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any

from ..contract.validation import message_hash
from ..errors import ParticipantFailure, ReplayDivergence, ResourceLimitExceeded
from .services import ReplayableServices, ServiceEvent, ServiceFactory


@dataclass(frozen=True, slots=True)
class ActorCall:
    message: Any
    output_hash: str
    # Number of actor-local service events committed when this call returned.
    # This binds durable call cursors to the exact service-tape prefix they used.
    service_event_count: int | None = None


class ActorFactory:
    """Constructs one participant class and its isolated service facade."""

    def __init__(
        self,
        actor_class: type,
        *,
        constructor_args: tuple[Any, ...] = (),
        service_factory: ServiceFactory,
    ) -> None:
        self.actor_class = actor_class
        self.constructor_args = constructor_args
        self.service_factory = service_factory

    def create(
        self, replay_tape: tuple[ServiceEvent, ...] = ()
    ) -> tuple[Any, ReplayableServices]:
        services = self.service_factory.create(replay_tape)
        try:
            actor = self.actor_class(*copy.deepcopy(self.constructor_args), services)
        except Exception as exc:  # noqa: BLE001
            raise ParticipantFailure("participant constructor failed") from exc
        if not callable(getattr(actor, "step", None)):
            raise ParticipantFailure("participant class does not define step(message)")
        return actor, services


@dataclass(slots=True)
class ActorHandle:
    factory: ActorFactory
    actor: Any
    services: ReplayableServices
    calls: list[ActorCall] = field(default_factory=list)
    branch_id: str = "root"


@dataclass(frozen=True, slots=True)
class ActorCheckpoint:
    """Runtime-owned reconstruction recipe; never passed to participant code."""

    factory: ActorFactory
    calls: tuple[ActorCall, ...]
    service_tape: tuple[ServiceEvent, ...]
    branch_id: str = "root"
    service_state: dict[str, Any] = field(default_factory=dict)


class ActorRuntime:
    def start(self, factory: ActorFactory, *, branch_id: str = "root") -> ActorHandle:
        actor, services = factory.create()
        return ActorHandle(factory, actor, services, branch_id=branch_id)

    def call(self, handle: ActorHandle, message: Any) -> Any:
        try:
            output = handle.actor.step(copy.deepcopy(message))
        except (ParticipantFailure, ReplayDivergence, ResourceLimitExceeded):
            raise
        except Exception as exc:  # noqa: BLE001
            raise ParticipantFailure("participant step failed") from exc
        # Participant-owned output may retain references to mutable actor state.
        # Copy it at the trust boundary before hashing or storing it in the arena.
        trusted_output = copy.deepcopy(output)
        handle.calls.append(
            ActorCall(
                copy.deepcopy(message),
                message_hash(trusted_output),
                handle.services.event_count(),
            )
        )
        return trusted_output

    def checkpoint(self, handle: ActorHandle) -> ActorCheckpoint:
        return ActorCheckpoint(
            factory=handle.factory,
            calls=tuple(copy.deepcopy(handle.calls)),
            service_tape=handle.services.export_tape(),
            branch_id=handle.branch_id,
            service_state=handle.services.export_state(),
        )

    def fork(self, checkpoint: ActorCheckpoint, branch_id: str) -> ActorHandle:
        actor, services = checkpoint.factory.create(checkpoint.service_tape)
        handle = ActorHandle(checkpoint.factory, actor, services, branch_id=branch_id)
        for index, record in enumerate(checkpoint.calls):
            try:
                output = actor.step(copy.deepcopy(record.message))
            except Exception as exc:  # noqa: BLE001
                raise ReplayDivergence(
                    f"actor failed while replaying call {index}: "
                    f"{type(exc).__name__}: {exc}"
                ) from exc
            digest = message_hash(output)
            if digest != record.output_hash:
                raise ReplayDivergence(f"actor output diverged while replaying call {index}")
            if (
                record.service_event_count is not None
                and len(checkpoint.service_tape) - services.replay_remaining()
                != record.service_event_count
            ):
                raise ReplayDivergence(
                    f"actor service cursor diverged while replaying call {index}"
                )
            handle.calls.append(copy.deepcopy(record))
        services.finish_replay(branch_id)
        return handle

    def restore(self, checkpoint: ActorCheckpoint) -> ActorHandle:
        """Reconstruct a durable checkpoint and continue from its exact service state."""
        actor, services = checkpoint.factory.create(checkpoint.service_tape)
        handle = ActorHandle(
            checkpoint.factory,
            actor,
            services,
            branch_id=checkpoint.branch_id,
        )
        for index, record in enumerate(checkpoint.calls):
            try:
                output = actor.step(copy.deepcopy(record.message))
            except Exception as exc:  # noqa: BLE001
                raise ReplayDivergence(
                    f"actor failed while restoring call {index}: "
                    f"{type(exc).__name__}: {exc}"
                ) from exc
            if message_hash(output) != record.output_hash:
                raise ReplayDivergence(f"actor output diverged while restoring call {index}")
            if (
                record.service_event_count is not None
                and len(checkpoint.service_tape) - services.replay_remaining()
                != record.service_event_count
            ):
                raise ReplayDivergence(
                    f"actor service cursor diverged while restoring call {index}"
                )
            handle.calls.append(copy.deepcopy(record))
        services.finish_replay(checkpoint.branch_id)
        services.restore_state(checkpoint.service_state)
        return handle

    def replay_prefix(self, checkpoint: ActorCheckpoint) -> ActorHandle:
        """Verify completed calls while allowing a failed step's trailing service events."""
        actor, services = checkpoint.factory.create(checkpoint.service_tape)
        handle = ActorHandle(
            checkpoint.factory,
            actor,
            services,
            branch_id=checkpoint.branch_id,
        )
        for index, record in enumerate(checkpoint.calls):
            try:
                output = actor.step(copy.deepcopy(record.message))
            except Exception as exc:  # noqa: BLE001
                raise ReplayDivergence(f"actor failed while replaying prefix call {index}") from exc
            if message_hash(output) != record.output_hash:
                raise ReplayDivergence(f"actor output diverged while replaying prefix call {index}")
            if (
                record.service_event_count is not None
                and len(checkpoint.service_tape) - services.replay_remaining()
                != record.service_event_count
            ):
                raise ReplayDivergence(
                    f"actor service cursor diverged while replaying prefix call {index}"
                )
            handle.calls.append(copy.deepcopy(record))
        return handle

    @staticmethod
    def close(handle: ActorHandle) -> None:
        close = getattr(handle.actor, "close", None)
        if callable(close):
            close()
