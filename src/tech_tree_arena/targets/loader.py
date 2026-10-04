"""Target-pack discovery and integrity validation."""

from __future__ import annotations

import hashlib
import json
import os
import re
import tomllib
from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path
from typing import Any

from ..contract import PROTOCOL_NAME
from ..errors import ValidationError

_TARGET_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


@dataclass(frozen=True, slots=True)
class TargetPack:
    name: str
    root: Path
    manifest: dict[str, Any]

    def target_ids(self) -> tuple[str, ...]:
        gold = self.root / "secret" / "gold"
        return tuple(sorted(path.stem for path in gold.glob("*.json") if _TARGET_ID.fullmatch(path.stem)))

    def load(self, target_id: str) -> dict[str, Any]:
        if not _TARGET_ID.fullmatch(target_id):
            raise ValidationError("invalid target ID")
        path = self.root / "secret" / "gold" / f"{target_id}.json"
        if not path.is_file():
            raise ValidationError(f"target {target_id!r} is not in pack {self.name!r}")
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValidationError("target is not valid JSON") from exc
        if not isinstance(value, dict):
            raise ValidationError("target record must be an object")
        if self.manifest.get("kind") == "development":
            summary = value.get("summary")
            if value.get("arxiv_id") != target_id or not isinstance(summary, dict):
                raise ValidationError("development target has an invalid identity or summary")
            setting = summary.get("setting_and_object")
            if (
                not isinstance(summary.get("title"), str)
                or not isinstance(setting, dict)
                or not isinstance(setting.get("category"), str)
                or not isinstance(setting.get("groups"), list)
                or not isinstance(summary.get("key_findings"), list)
            ):
                raise ValidationError("development target does not match the summary schema")
        return value

    def public_resources(self) -> dict[str, Any]:
        """Load JSON values explicitly published by this target pack."""
        public = self.root / "public"
        if not public.is_dir():
            return {}
        result: dict[str, Any] = {}
        for path in sorted(public.glob("*.json")):
            if path.is_symlink() or not path.is_file() or not _TARGET_ID.fullmatch(path.stem):
                raise ValidationError("public resources contain an unsupported entry")
            try:
                result[path.stem] = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise ValidationError(f"public resource {path.name!r} is not valid JSON") from exc
        return result


def _candidate_roots(name: str) -> tuple[Path, ...]:
    candidates: list[Path] = []
    arena_home = os.environ.get("IDEA_ARENA_HOME")
    if arena_home:
        candidates.append(Path(arena_home) / "target-packs" / name)
    package_root = files("tech_tree_arena.data").joinpath("target_packs", name)
    candidates.append(Path(str(package_root)))
    return tuple(candidates)


def load_target_pack(name_or_path: str | Path) -> TargetPack:
    requested = Path(name_or_path)
    roots = (requested,) if requested.is_dir() else _candidate_roots(str(name_or_path))
    root = next((candidate.resolve() for candidate in roots if candidate.is_dir()), None)
    if root is None:
        raise ValidationError(f"target pack {str(name_or_path)!r} was not found")
    manifest_path = root / "pack.toml"
    if not manifest_path.is_file():
        raise ValidationError("target pack is missing pack.toml")
    try:
        manifest = tomllib.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise ValidationError("target pack manifest is invalid") from exc
    name = manifest.get("name")
    if not isinstance(name, str) or not name:
        raise ValidationError("target pack manifest has no name")
    pack = TargetPack(name, root, manifest)
    if manifest.get("schema_version") != 1 or manifest.get("protocol") != PROTOCOL_NAME:
        raise ValidationError("target pack uses an unsupported schema or protocol")
    declared = manifest.get("target_count")
    if declared is not None and declared != len(pack.target_ids()):
        raise ValidationError("target pack count does not match its manifest")
    expected_taxonomy = manifest.get("taxonomy_sha256")
    if expected_taxonomy and file_sha256(root / "public" / "taxonomy.json") != expected_taxonomy:
        raise ValidationError("target pack taxonomy hash does not match its manifest")
    expected_rankings = manifest.get("taxonomy_rankings_sha256")
    if expected_rankings and file_sha256(root / "public" / "taxonomy_rankings.json") != expected_rankings:
        raise ValidationError("target pack taxonomy rankings hash does not match its manifest")
    expected_gold = manifest.get("gold_tree_sha256")
    if expected_gold and gold_tree_sha256(root / "secret" / "gold") != expected_gold:
        raise ValidationError("target pack gold-tree hash does not match its manifest")
    return pack


def file_sha256(path: Path) -> str:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError as exc:
        raise ValidationError(f"target-pack resource {path.name!r} is missing") from exc


def gold_tree_sha256(root: Path) -> str:
    """Reproduce the provenance hash: SHA-256 of sorted ``shasum`` lines."""
    lines = []
    for path in sorted(root.glob("*.json")):
        if path.is_symlink() or not path.is_file():
            raise ValidationError("gold directory contains an unsupported entry")
        lines.append(f"{file_sha256(path)}  ./{path.name}\n")
    return hashlib.sha256("".join(lines).encode("utf-8")).hexdigest()
