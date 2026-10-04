"""Working-directory-independent paths for Arena state."""

from __future__ import annotations

import os
from pathlib import Path


def arena_home() -> Path:
    configured = os.environ.get("IDEA_ARENA_HOME")
    if configured:
        return Path(configured).expanduser().resolve()
    return Path.home() / ".local" / "share" / "idea-arena"
