"""Trusted probability and information-cost accounting."""

from __future__ import annotations

from decimal import Decimal, InvalidOperation, localcontext
from typing import Iterable, Protocol

from ..errors import ValidationError

DEFAULT_TOLERANCE = Decimal("1e-9")


class HasProbability(Protocol):
    probability: int | float | str


def as_probability(value: int | float | str | Decimal) -> Decimal:
    """Parse one participant probability without inheriting binary float math."""

    if isinstance(value, bool):
        raise ValidationError("probability must be numeric, not boolean")
    try:
        probability = value if isinstance(value, Decimal) else Decimal(str(value))
    except (InvalidOperation, ValueError, TypeError) as exc:
        raise ValidationError("probability is not a valid decimal") from exc
    if not probability.is_finite():
        raise ValidationError("probability must be finite")
    if probability <= 0:
        raise ValidationError("probability must be strictly positive")
    if probability > 1:
        raise ValidationError("probability must not exceed one")
    return probability


def validate_distribution(
    items: Iterable[HasProbability], *, tolerance: Decimal = DEFAULT_TOLERANCE
) -> tuple[Decimal, ...]:
    probabilities = tuple(as_probability(item.probability) for item in items)
    if not probabilities:
        raise ValidationError("a probability distribution must not be empty")
    total = sum(probabilities, Decimal(0))
    if abs(total - Decimal(1)) > tolerance:
        raise ValidationError(f"probabilities sum to {total}, not one")
    return probabilities


def information_cost(probability: int | float | str | Decimal) -> float:
    p = as_probability(probability)
    with localcontext() as ctx:
        ctx.prec = 50
        bits = -(p.ln() / Decimal(2).ln())
    return float(bits)


def accepted_mass(probabilities: Iterable[Decimal]) -> Decimal:
    return sum(probabilities, Decimal(0))
