from __future__ import annotations

import json
import stat
from pathlib import Path

import pytest

from tech_tree_arena.errors import ReplayDivergence
from tech_tree_arena.replay.artifacts import ArtifactStore


def test_content_addressed_artifacts_ignore_untrusted_reference_paths(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "participant.py").write_text("VALUE = 1\n", encoding="utf-8")
    store = ArtifactStore(tmp_path / "runs")

    assert stat.S_IMODE(store.root.stat().st_mode) == 0o700
    assert stat.S_IMODE(store.root.parent.stat().st_mode) == 0o700

    tree = store.snapshot_tree(source)
    data = store.put_json({"answer": 42})
    tree["path"] = "../../outside"
    data["path"] = "../../outside.json"

    assert (store.resolve_tree(tree) / "participant.py").read_text(encoding="utf-8") == "VALUE = 1\n"
    assert store.load_json(data) == {"answer": 42}


def test_content_addressed_artifact_corruption_is_rejected(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path / "runs")
    reference = store.put_json({"answer": 42})
    artifact = store.root / f"{reference['sha256']}.json"
    artifact.write_text(json.dumps({"answer": 0}), encoding="utf-8")

    with pytest.raises(ReplayDivergence, match="corrupt"):
        store.load_json(reference)
