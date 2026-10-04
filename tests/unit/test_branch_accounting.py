import math

import pytest

from tech_tree_arena import Option, Question
from tech_tree_arena.contract.recovery import occurrence_bits
from tech_tree_arena.runtime.actor import ActorFactory, ActorRuntime
from tech_tree_arena.runtime.branch import BranchStore
from tech_tree_arena.runtime.services import ServiceFactory


class Actor:
    def __init__(self, services):
        pass

    def step(self, message):
        return Question("q", (Option("a", "a", "0.5"), Option("b", "b", "0.5")))


def _checkpoint():
    runtime = ActorRuntime()
    actor = runtime.start(ActorFactory(Actor, service_factory=ServiceFactory(seed=1)))
    runtime.call(actor, None)
    return runtime.checkpoint(actor)


def test_choice_repetition_counters_never_rewind() -> None:
    store = BranchStore(run_id="r", seed=1)
    question = Question("q", (Option("a", "a", "0.5"), Option("b", "b", "0.5")))
    node = store.add_question(question, checkpoint=_checkpoint(), path_k=0, parent_question_id=None)
    assert store.choice_branch_cost(node, "a") == pytest.approx(-math.log2(0.95))
    assert store.choice_branch_cost(node, "b") == pytest.approx(-math.log2(0.95))
    assert store.choice_branch_cost(node, "a") == pytest.approx(occurrence_bits(2))


def test_checkout_is_free_but_audited_and_pair_counts_do_not_rewind() -> None:
    store = BranchStore(run_id="r", seed=1)
    question = Question("q", (Option("a", "a", "1"),))
    first = store.add_question(question, checkpoint=_checkpoint(), path_k=0, parent_question_id=None)
    second = store.add_question(question, checkpoint=_checkpoint(), path_k=1, parent_question_id=first.question_id)
    third = store.add_question(question, checkpoint=_checkpoint(), path_k=2, parent_question_id=second.question_id)

    first_preview = store.preview_checkout(third, first.question_id)
    store.commit_checkout(first_preview)
    repeated_preview = store.preview_checkout(third, first.question_id)
    store.commit_checkout(repeated_preview)
    assert first_preview.branch_bits == 0.0
    assert repeated_preview.branch_bits == 0.0
    assert store.checkout_pair_counts[(third.question_id, first.question_id)] == 2
    assert [item.pair_index for item in store.checkout_audit] == [1, 2]
