"""Evaluation-profile trust checks shared by runner frontends."""

from __future__ import annotations

from ..errors import ArenaError
from ..targets.loader import TargetPack


def require_runner(pack: TargetPack, runner: str) -> None:
    if runner == "local" and pack.manifest.get("kind") == "hidden":
        raise ArenaError("hidden target packs are refused by the unverified local runner")
    if runner == "hardened":
        # A local process or ordinary container cannot honestly satisfy the
        # hidden-profile boundary.  Fail closed until an independently reviewed
        # external worker is configured and attested.
        raise ArenaError(
            "no attested hardened worker is configured; use the local runner only with public/development targets"
        )
    if runner != "local":
        raise ArenaError(f"unknown runner profile {runner!r}")
