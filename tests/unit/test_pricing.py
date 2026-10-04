from decimal import Decimal

import pytest

from tech_tree_arena.errors import ValidationError
from tech_tree_arena import Option
from tech_tree_arena.contract.pricing import information_cost, validate_distribution


def test_direct_probability_pricing() -> None:
    assert information_cost("0.5") == pytest.approx(1.0)
    assert information_cost(Decimal("0.25")) == pytest.approx(2.0)


def test_distribution_is_strictly_positive_and_normalized() -> None:
    options = (Option("a", None, "0.25"), Option("b", None, "0.75"))
    assert validate_distribution(options) == (Decimal("0.25"), Decimal("0.75"))

    with pytest.raises(ValidationError, match="strictly positive"):
        validate_distribution((Option("a", None, 0), Option("b", None, 1)))
    with pytest.raises(ValidationError, match="not one"):
        validate_distribution((Option("a", None, "0.2"), Option("b", None, "0.2")))


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -1, True])
def test_invalid_probabilities_fail_closed(bad) -> None:
    with pytest.raises(ValidationError):
        information_cost(bad)
