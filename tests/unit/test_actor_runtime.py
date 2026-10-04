import json

from tech_tree_arena import Choice, Option, Question
from tech_tree_arena.runtime.actor import ActorFactory, ActorRuntime
import pytest

from tech_tree_arena.errors import ReplayDivergence, ResourceLimitExceeded
from tech_tree_arena.contract.validation import message_hash
from tech_tree_arena.replay.recorder import (
    _journal_service_meter,
    _validate_service_tail_state,
)
from tech_tree_arena.runtime.services import (
    ReplayableServices,
    ServiceEvent,
    ServiceFactory,
    ServiceLimits,
    ServiceMeter,
)


class StatefulActor:
    def __init__(self, services):
        self.services = services
        self.seen = []

    def step(self, message):
        self.seen.append(message)
        draw = self.services.randint(1, 100)
        return Question(str(len(self.seen)), (Option("x", draw, "1"),))


def test_fork_reconstructs_private_state_and_replays_service_prefix() -> None:
    runtime = ActorRuntime()
    factory = ActorFactory(StatefulActor, service_factory=ServiceFactory(seed=11))
    actor = runtime.start(factory)

    first = runtime.call(actor, None)
    checkpoint = runtime.checkpoint(actor)
    runtime.call(actor, Choice("x"))

    fork = runtime.fork(checkpoint, "alternative")
    assert fork.actor.seen == [None]
    assert runtime.call(fork, Choice("y")).question == "2"
    assert first.question == "1"


def test_durable_restore_continues_exact_rng_state() -> None:
    runtime = ActorRuntime()
    factory = ActorFactory(StatefulActor, service_factory=ServiceFactory(seed=19))
    actor = runtime.start(factory)
    runtime.call(actor, None)
    checkpoint = runtime.checkpoint(actor)
    expected = runtime.call(actor, Choice("x"))

    restored = runtime.restore(checkpoint)
    actual = runtime.call(restored, Choice("x"))

    assert actual == expected


def test_durable_restore_replays_the_rng_epoch_of_a_forked_branch() -> None:
    runtime = ActorRuntime()
    factory = ActorFactory(StatefulActor, service_factory=ServiceFactory(seed=29))
    root = runtime.start(factory)
    runtime.call(root, None)
    root_checkpoint = runtime.checkpoint(root)

    branch = runtime.fork(root_checkpoint, "branch-rng")
    runtime.call(branch, Choice("first-branch-call"))
    branch_checkpoint = runtime.checkpoint(branch)
    expected = runtime.call(branch, Choice("second-branch-call"))

    restored = runtime.restore(branch_checkpoint)
    actual = runtime.call(restored, Choice("second-branch-call"))

    assert actual == expected


def test_service_budget_does_not_rewind_with_actor_state() -> None:
    runtime = ActorRuntime()
    factory = ActorFactory(
        StatefulActor,
        service_factory=ServiceFactory(seed=11, limits=ServiceLimits(max_random_calls=2)),
    )
    actor = runtime.start(factory)
    runtime.call(actor, None)
    checkpoint = runtime.checkpoint(actor)
    runtime.call(actor, Choice("x"))

    fork = runtime.fork(checkpoint, "alternative")
    with pytest.raises(ResourceLimitExceeded, match="budget"):
        runtime.call(fork, Choice("y"))
    assert factory.service_factory.meter.random_calls == 2


def test_resource_limit_service_tail_uses_the_recorded_zero_delta_meter() -> None:
    events = []
    services = ServiceFactory(
        seed=31,
        limits=ServiceLimits(max_random_calls=0),
        event_sink=events.append,
    ).create()

    with pytest.raises(ResourceLimitExceeded, match="budget"):
        services.random()

    assert len(events) == 1
    assert events[0].error == "ResourceLimitExceeded"
    assert _journal_service_meter(
        events,
        source="test RLE tail",
        initial={"model_calls": 0, "random_calls": 0},
        require_metadata=True,
    ) == {"model_calls": 0, "random_calls": 0}


def test_local_only_boundary_rejects_services_without_cost_or_tape_event() -> None:
    services = ServiceFactory(seed=32).create()

    with services.local_only():
        with pytest.raises(ResourceLimitExceeded, match="stage transition"):
            services.random()

    assert services.event_count() == 0
    assert services.usage()["random_calls"] == 0


def test_journal_service_meter_accepts_concurrent_completion_snapshots() -> None:
    events = [
        ServiceEvent(
            "model.structured",
            str(index),
            {"ok": True},
            metadata={
                "service_meter": {"model_calls": meter, "random_calls": 0}
            },
        )
        for index, meter in enumerate((1, 3, 3), start=1)
    ]

    assert _journal_service_meter(
        events,
        source="concurrent Judge batch",
        require_metadata=True,
    ) == {"model_calls": 3, "random_calls": 0}


def test_journal_service_meter_rejects_unaccounted_concurrent_attempt() -> None:
    events = [
        ServiceEvent(
            "model.structured",
            str(index),
            {"ok": True},
            metadata={
                "service_meter": {"model_calls": meter, "random_calls": 0}
            },
        )
        for index, meter in enumerate((1, 3), start=1)
    ]

    with pytest.raises(ReplayDivergence, match="attempt count"):
        _journal_service_meter(
            events,
            source="tampered concurrent Judge batch",
            require_metadata=True,
        )


def test_journal_service_meter_accounts_for_agentic_child_turns() -> None:
    event = ServiceEvent(
        "agent.session_turn",
        "generator-call",
        {"ok": True},
        metadata={
            "service_meter": {"model_calls": 4, "random_calls": 0},
            "usage": {"turns": 4},
        },
    )

    assert _journal_service_meter(
        [event],
        source="agentic Generator batch",
        require_metadata=True,
    ) == {"model_calls": 4, "random_calls": 0}


def test_service_tail_accepts_agentic_success_then_failed_main_turn() -> None:
    services = ReplayableServices(seed=38)
    initial_state = services.export_state()
    initial_state["meter"] = {"model_calls": 4, "random_calls": 0}
    rng_state = json.loads(json.dumps(initial_state["rng_state"]))
    events = (
        ServiceEvent(
            "agent.session_turn",
            "state-synthesis",
            {"ok": True},
            metadata={
                "service_meter": {"model_calls": 6, "random_calls": 0},
                "service_branch_id": "root",
                "rng_state_after": rng_state,
                "usage": {"turns": 2},
            },
        ),
        ServiceEvent(
            "agent.session_turn",
            "timed-out-generator",
            None,
            error="TimeoutError",
            error_message="service call failed",
            metadata={
                "service_meter": {"model_calls": 7, "random_calls": 0},
                "service_branch_id": "root",
                "rng_state_after": rng_state,
                "usage": {},
            },
        ),
    )

    meter, _ = _validate_service_tail_state(
        events,
        initial_state=initial_state,
        source="agentic Generator service tail",
        branch_id="root",
        require_metadata=True,
    )

    assert meter == {"model_calls": 7, "random_calls": 0}


def test_failed_replay_enters_recorded_branch_and_preserves_exception_type() -> None:
    seed = 37
    branch_id = "failed-service-branch"
    source = ReplayableServices(seed=seed)
    source.finish_replay(branch_id)

    with pytest.raises(IndexError):
        source.choice([])
    failed_event = source.export_tape()[0]
    expected = source.random()

    resumed = ReplayableServices(
        seed=seed,
        replay_tape=(failed_event,),
        meter=ServiceMeter(random_calls=1),
    )
    with pytest.raises(IndexError):
        resumed.choice([])
    actual = resumed.random()

    assert actual == expected
    assert resumed.export_tape()[-1].metadata["service_branch_id"] == branch_id


@pytest.mark.parametrize("tape_name", ("replay_tape", "committed_tape"))
def test_unknown_replayed_exception_type_fails_closed(tape_name) -> None:
    source = ReplayableServices(seed=41)
    with pytest.raises(IndexError):
        source.choice([])
    failed_event = source.export_tape()[0]
    unknown_event = ServiceEvent(
        kind=failed_event.kind,
        request_hash=failed_event.request_hash,
        response=None,
        error="ParticipantDefinedError",
        request=failed_event.request,
        error_message="opaque participant exception",
        metadata=failed_event.metadata,
    )
    with pytest.raises(ReplayDivergence, match="unsupported replayed service error type"):
        ReplayableServices(seed=41, **{tape_name: (unknown_event,)})


def test_unknown_post_checkpoint_service_tail_fails_closed_before_retry() -> None:
    source = ReplayableServices(seed=42)
    initial_state = source.export_state()
    with pytest.raises(IndexError):
        source.choice([])
    failed_event = source.export_tape()[0]
    unknown_event = ServiceEvent(
        kind=failed_event.kind,
        request_hash=failed_event.request_hash,
        response=None,
        error="OpaqueProviderError",
        request=failed_event.request,
        error_message="unknown Judge provider failure",
        metadata=failed_event.metadata,
    )

    with pytest.raises(ReplayDivergence, match="unsupported replayed service error type"):
        _validate_service_tail_state(
            (unknown_event,),
            initial_state=initial_state,
            source="post-checkpoint Judge service tail",
            branch_id="root",
            require_metadata=True,
        )


def test_common_provider_exception_type_remains_replayable() -> None:
    source = ReplayableServices(seed=43)
    with pytest.raises(IndexError):
        source.choice([])
    failed_event = source.export_tape()[0]
    provider_event = ServiceEvent(
        kind=failed_event.kind,
        request_hash=failed_event.request_hash,
        response=None,
        error="APIConnectionError",
        request=failed_event.request,
        error_message="temporary provider connection failure",
        metadata=failed_event.metadata,
    )
    resumed = ReplayableServices(seed=43, replay_tape=(provider_event,))

    with pytest.raises(ConnectionError) as failure:
        resumed.choice([])
    assert type(failure.value).__name__ == "APIConnectionError"


def test_openai_base_exception_type_remains_replayable() -> None:
    source = ReplayableServices(seed=44)
    with pytest.raises(IndexError):
        source.choice([])
    failed_event = source.export_tape()[0]
    provider_event = ServiceEvent(
        kind=failed_event.kind,
        request_hash=failed_event.request_hash,
        response=None,
        error="OpenAIError",
        request=failed_event.request,
        error_message="missing or invalid OpenAI configuration",
        metadata=failed_event.metadata,
    )
    resumed = ReplayableServices(seed=44, replay_tape=(provider_event,))

    with pytest.raises(RuntimeError) as failure:
        resumed.choice([])
    assert type(failure.value).__name__ == "OpenAIError"


class RetainedOutputActor:
    def __init__(self, services):
        self.services = services
        self.payload = {"value": "original"}

    def step(self, message):
        if message is None:
            return Question("first", (Option("x", self.payload, "1"),))
        self.payload["value"] = "mutated"
        return Question("second", (Option("x", self.payload, "1"),))


def test_actor_output_is_copied_at_the_trust_boundary() -> None:
    runtime = ActorRuntime()
    factory = ActorFactory(RetainedOutputActor, service_factory=ServiceFactory(seed=1))
    actor = runtime.start(factory)

    first = runtime.call(actor, None)
    checkpoint = runtime.checkpoint(actor)
    runtime.call(actor, Choice("x"))

    assert first.options[0].public_payload == {"value": "original"}
    assert actor.calls[0].output_hash == message_hash(first)
    fork = runtime.fork(checkpoint, "copy-regression")
    assert fork.calls == list(checkpoint.calls)
