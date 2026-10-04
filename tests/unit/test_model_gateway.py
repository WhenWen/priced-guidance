from concurrent.futures import ThreadPoolExecutor

import pytest

from tech_tree_arena.errors import ResourceLimitExceeded
from tech_tree_arena.runtime.services import ServiceFactory, ServiceLimits


class CaptureBackend:
    def __init__(self) -> None:
        self.requests = []

    def structured(self, **request):
        self.requests.append(request)
        return {"ok": True}


def test_evaluation_profile_overrides_participant_model_and_caps_output() -> None:
    backend = CaptureBackend()
    services = ServiceFactory(
        seed=1,
        model_backend=backend,
        model_name="arena-model",
        limits=ServiceLimits(max_model_calls=1, max_output_tokens_per_call=123),
    ).create()
    response = services.structured_model(model="participant-endpoint", max_output_tokens=99_999)
    assert response == {"ok": True}
    assert backend.requests[0]["model"] == "arena-model"
    assert backend.requests[0]["max_output_tokens"] == 123
    with pytest.raises(ResourceLimitExceeded, match="budget"):
        services.structured_model(model="anything")


def test_parallel_model_calls_cannot_oversubscribe_the_shared_budget() -> None:
    backend = CaptureBackend()
    factory = ServiceFactory(
        seed=1,
        model_backend=backend,
        limits=ServiceLimits(max_model_calls=4),
    )
    services = factory.create()

    def call(index):
        try:
            services.structured_model(schema_name=f"request-{index}")
            return "accepted"
        except ResourceLimitExceeded:
            return "rejected"

    with ThreadPoolExecutor(max_workers=12) as executor:
        outcomes = list(executor.map(call, range(24)))

    assert outcomes.count("accepted") == 4
    assert outcomes.count("rejected") == 20
    assert factory.meter.model_calls == 4
    assert len(backend.requests) == 4
    assert len(services.export_tape()) == 24
    assert sum(event.error == "ResourceLimitExceeded" for event in services.export_tape()) == 20


def test_service_failures_are_part_of_the_replay_tape() -> None:
    factory = ServiceFactory(seed=7, limits=ServiceLimits(max_random_calls=1))
    services = factory.create()
    first = services.random()
    with pytest.raises(ResourceLimitExceeded):
        services.random()

    replay = ServiceFactory(seed=7).create(services.export_tape())
    assert replay.random() == first
    with pytest.raises(ResourceLimitExceeded, match="random-service budget exhausted"):
        replay.random()
    replay.finish_replay("failure-tape")


def test_non_model_services_do_not_inherit_stale_provider_usage() -> None:
    class MetadataBackend(CaptureBackend):
        @staticmethod
        def last_call_metadata():
            return {"usage": {"calls": 1, "cost_usd": 0.25}}

    services = ServiceFactory(
        seed=9,
        model_backend=MetadataBackend(),
        judge_call=lambda ideas: [{"idea_id": "probe", "passed": True}],
    ).create()

    services.structured_model(schema_name="paid-model-call")
    services.random()
    services.judge_evaluate([{"idea_id": "probe"}])

    model_event, random_event, judge_proxy = services.export_tape()
    assert model_event.metadata["usage"]["cost_usd"] == 0.25
    assert "usage" not in random_event.metadata
    assert "usage" not in judge_proxy.metadata


def test_replayed_post_checkpoint_random_call_advances_restored_rng() -> None:
    original = ServiceFactory(seed=23).create()
    first = original.randint(1, 1_000_000)
    checkpoint_state = original.export_state()
    second = original.randint(1, 1_000_000)
    expected_next = original.randint(1, 1_000_000)

    resumed = ServiceFactory(seed=23).create(original.export_tape()[:2])
    assert resumed.randint(1, 1_000_000) == first
    resumed.restore_state(checkpoint_state)
    assert resumed.randint(1, 1_000_000) == second
    assert resumed.randint(1, 1_000_000) == expected_next


def test_concurrent_model_tape_accepts_reordered_distinct_requests() -> None:
    backend = CaptureBackend()
    original = ServiceFactory(seed=1, model_backend=backend).create()
    first = original.structured_model(schema_name="first")
    second = original.structured_model(schema_name="second")

    replay = ServiceFactory(seed=1).create(original.export_tape())
    assert replay.structured_model(schema_name="second") == second
    assert replay.structured_model(schema_name="first") == first
    replay.finish_replay("concurrent-model-batch")


def test_committed_service_history_is_exported_but_not_replayed_again() -> None:
    original_backend = CaptureBackend()
    original = ServiceFactory(seed=31, model_backend=original_backend).create()
    original.structured_model(schema_name="committed")
    expected_tail = original.structured_model(schema_name="tail")
    committed, tail = original.export_tape()

    resumed_backend = CaptureBackend()
    resumed_factory = ServiceFactory(
        seed=31,
        model_backend=resumed_backend,
        initial_model_calls=2,
    )
    resumed = resumed_factory.create((tail,), committed_tape=(committed,))

    assert resumed.structured_model(schema_name="tail") == expected_tail
    assert resumed_backend.requests == []
    resumed.structured_model(schema_name="new-live-call")
    assert [event.request_hash for event in resumed.export_tape()[:2]] == [
        committed.request_hash,
        tail.request_hash,
    ]
    assert len(resumed_backend.requests) == 1
    assert resumed_factory.meter.model_calls == 3
