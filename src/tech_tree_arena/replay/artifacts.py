"""Content-addressed private artifacts shared by run records."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

from ..errors import ReplayDivergence


def hash_tree(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        relative = path.relative_to(root).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        data = path.read_bytes()
        digest.update(len(data).to_bytes(8, "big"))
        digest.update(data)
    return digest.hexdigest()


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


class ArtifactStore:
    def __init__(self, runs_root: str | Path) -> None:
        self.runs_root = Path(runs_root).resolve()
        self.root = self.runs_root / ".artifacts" / "sha256"
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        os.chmod(self.runs_root / ".artifacts", 0o700)
        os.chmod(self.root, 0o700)

    def snapshot_tree(self, source: str | Path) -> dict[str, Any]:
        source = Path(source).resolve()
        digest = hash_tree(source)
        destination = self.root / digest
        if not destination.is_dir():
            temporary = Path(tempfile.mkdtemp(prefix=f".{digest[:12]}-", dir=self.root))
            try:
                shutil.rmtree(temporary)
                shutil.copytree(source, temporary)
                try:
                    os.replace(temporary, destination)
                except FileExistsError:
                    shutil.rmtree(temporary, ignore_errors=True)
            finally:
                if temporary.exists():
                    shutil.rmtree(temporary, ignore_errors=True)
        return {
            "kind": "tree",
            "sha256": digest,
            "path": str(destination.relative_to(self.runs_root)),
        }

    def put_json(self, value: Any) -> dict[str, Any]:
        data = _canonical_json(value)
        digest = hashlib.sha256(data).hexdigest()
        destination = self.root / f"{digest}.json"
        if not destination.is_file():
            temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
            temporary.write_bytes(data)
            try:
                os.replace(temporary, destination)
            finally:
                temporary.unlink(missing_ok=True)
        return {
            "kind": "json",
            "sha256": digest,
            "size": len(data),
            "path": str(destination.relative_to(self.runs_root)),
        }

    def resolve_tree(self, reference: dict[str, Any]) -> Path:
        digest = reference.get("sha256")
        if reference.get("kind") != "tree" or not isinstance(digest, str) or len(digest) != 64:
            raise ReplayDivergence("content-addressed tree reference is invalid")
        path = self.root / digest
        if not path.is_dir() or hash_tree(path) != digest:
            raise ReplayDivergence("content-addressed tree artifact is missing or corrupt")
        return path

    def load_json(self, reference: dict[str, Any]) -> Any:
        digest = reference.get("sha256")
        if reference.get("kind") != "json" or not isinstance(digest, str) or len(digest) != 64:
            raise ReplayDivergence("content-addressed JSON reference is invalid")
        path = self.root / f"{digest}.json"
        try:
            data = path.read_bytes()
        except OSError as exc:
            raise ReplayDivergence("content-addressed JSON artifact is missing") from exc
        if hashlib.sha256(data).hexdigest() != digest:
            raise ReplayDivergence("content-addressed JSON artifact is corrupt")
        try:
            return json.loads(data)
        except json.JSONDecodeError as exc:
            raise ReplayDivergence("content-addressed JSON artifact is invalid") from exc
