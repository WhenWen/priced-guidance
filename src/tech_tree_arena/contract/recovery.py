"""Versioned, target-independent pricing of repeated continuations."""

from __future__ import annotations

import math


LEGACY_ACCOUNTING = "branch-product-v1"
CURRENT_ACCOUNTING = "occurrence-prior-v2"
OCCURRENCE_EPSILON = 0.05


def occurrence_bits(index: int) -> float:
    """Code length for pi(1)=.95, pi(j)=.05/[j(j-1)] for j >= 2."""
    if type(index) is not int or index < 1:
        raise ValueError("occurrence index must be a positive integer")
    if index == 1:
        return -math.log2(1.0 - OCCURRENCE_EPSILON)
    return math.log2(index) + math.log2(index - 1) - math.log2(OCCURRENCE_EPSILON)


def choice_surcharge(continuation_index: int, option_index: int, *,
                     version: str = CURRENT_ACCOUNTING) -> float:
    """One pricing entry point for live execution and historical replay."""
    if (type(continuation_index) is not int or type(option_index) is not int
            or not 1 <= option_index <= continuation_index):
        raise ValueError("invalid continuation counters")
    if version == CURRENT_ACCOUNTING:
        return occurrence_bits(option_index)
    if version == LEGACY_ACCOUNTING:
        return math.log2(continuation_index) + math.log2(option_index)
    raise ValueError(f"unsupported accounting version {version!r}")


def validate_accounting_version(version: str) -> str:
    choice_surcharge(1, 1, version=version)
    return version
