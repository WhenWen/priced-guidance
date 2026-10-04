import copy
import threading
from concurrent.futures import ThreadPoolExecutor

import pytest

from tech_tree_arena.errors import ReplayDivergence
from tech_tree_arena.runtime.actor import ActorFactory, ActorRuntime
from tech_tree_arena.runtime.services import ServiceFactory


class NativeHistoryBackend:
    supports_context_chain = True

    def __init__(self):
        self.parents = []
        self.local = threading.local()

    def structured_in_context(self, context, **request):
        self.parents.append(copy.deepcopy(context))
        node = {"thread_id": str(len(self.parents)), "turn_id": "turn", "rollout": {"sha256": str(len(self.parents))}}
        self.local.metadata = {"codex_context": node, "usage": {"input_tokens": 1, "output_tokens": 1, "cost_usd": 0.0}}
        return {"node": node["thread_id"]}

    def last_call_metadata(self):
        return self.local.metadata


class Generator:
    def __init__(self, services):
        self.services = services

    def step(self, message):
        return self.services.structured_model(model="gpt-6-astra", developer="d", user=str(message), schema={}, schema_name="test")


def test_checkout_inherits_checkpoint_thinking_without_abandoned_branch():
    backend = NativeHistoryBackend()
    factory = ActorFactory(Generator, service_factory=ServiceFactory(seed=1, model_backend=backend))
    runtime = ActorRuntime()
    actor = runtime.start(factory)
    assert runtime.call(actor, "first") == {"node": "1"}
    first = runtime.checkpoint(actor)
    assert runtime.call(actor, "abandoned") == {"node": "2"}
    fork = runtime.fork(first, "alternate")
    assert runtime.call(fork, "alternative") == {"node": "3"}
    assert [None if p is None else p["thread_id"] for p in backend.parents] == [None, "1", "1"]
    assert runtime.checkpoint(fork).service_state["model_context"]["thread_id"] == "3"
    assert runtime.checkpoint(actor).service_state["model_context"]["thread_id"] == "2"


def test_restore_replays_without_model_call_and_preserves_native_context():
    backend = NativeHistoryBackend()
    runtime = ActorRuntime()
    actor = runtime.start(ActorFactory(Generator, service_factory=ServiceFactory(seed=1, model_backend=backend)))
    runtime.call(actor, "first")
    checkpoint = runtime.checkpoint(actor)
    restored = runtime.restore(checkpoint)
    assert len(backend.parents) == 1
    runtime.call(restored, "second")
    assert backend.parents[-1]["thread_id"] == "1"


def test_concurrent_channels_form_one_ordered_native_history():
    backend = NativeHistoryBackend()
    services = ServiceFactory(seed=1, model_backend=backend).create()
    generator = Generator(services)
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(generator.step, range(8)))
    assert [None if p is None else p["thread_id"] for p in backend.parents] == [None, *map(str, range(1, 8))]
    assert services.export_state()["model_context"]["thread_id"] == "8"


def test_checkpoint_cannot_replace_native_context():
    services = ServiceFactory(seed=1).create()
    with pytest.raises(ReplayDivergence, match="native model context"):
        services.restore_state({"model_context": {"thread_id": "abandoned"}})


def test_profile_overrides_submitted_effort_and_replays_offline():
    backend = NativeHistoryBackend()
    live = ServiceFactory(seed=1, model_backend=backend, reasoning_effort="xhigh").create()
    Generator(live).step("preview")
    tape = live.export_tape()
    assert tape[0].request["reasoning_effort"] == "xhigh"
    replay = ServiceFactory(seed=1, reasoning_effort="xhigh").create(tape)
    assert Generator(replay).step("preview") == {"node": "1"}
    assert len(backend.parents) == 1


def test_reordered_replay_failure_flushes_later_success_context():
    class FailingBackend(NativeHistoryBackend):
        def structured_in_context(self, context, **request):
            if request["user"] == "failure":
                self.local.metadata = {}
                raise RuntimeError("synthetic failure")
            return super().structured_in_context(context, **request)

    original = ServiceFactory(seed=1, model_backend=FailingBackend()).create()
    with pytest.raises(RuntimeError):
        Generator(original).step("failure")
    Generator(original).step("success")
    replay = ServiceFactory(seed=1).create(original.export_tape())
    # Concurrent channels can consume the second response before the first.
    Generator(replay).step("success")
    assert "model_context" not in replay.export_state()
    with pytest.raises(RuntimeError):
        Generator(replay).step("failure")
    assert replay.export_state()["model_context"] == original.export_state()["model_context"]
