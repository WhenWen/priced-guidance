import pytest

from tech_tree_arena.errors import ReplayDivergence, ResourceLimitExceeded
from tech_tree_arena.evaluation.research import _judge_call


@pytest.mark.parametrize(
    "failure",
    (
        ReplayDivergence("tampered replay tape"),
        ResourceLimitExceeded("judge service budget exhausted"),
    ),
)
def test_judge_retry_does_not_swallow_arena_control_errors(failure) -> None:
    attempts = 0

    def fail():
        nonlocal attempts
        attempts += 1
        raise failure

    with pytest.raises(type(failure), match=str(failure)):
        _judge_call(fail)
    assert attempts == 1
