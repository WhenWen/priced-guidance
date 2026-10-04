"""Build a validated development target pack from a selected paper JSONL."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import re
from pathlib import Path
from typing import Any


def _atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value, encoding="utf-8")
    temporary.replace(path)


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not rows:
        raise ValueError(f"empty selection: {path}")
    return rows


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _gold_tree_sha256(root: Path) -> str:
    lines = [
        f"{_file_sha256(path)}  ./{path.name}\n"
        for path in sorted(root.glob("*.json"))
    ]
    return hashlib.sha256("".join(lines).encode("utf-8")).hexdigest()


def _validate_summary(record: dict[str, Any], arxiv_id: str) -> dict[str, Any]:
    summary = record.get("summary")
    if record.get("arxiv_id") != arxiv_id or not isinstance(summary, dict):
        raise ValueError(f"invalid gold identity or summary for {arxiv_id}")
    setting = summary.get("setting_and_object")
    if (
        not isinstance(summary.get("title"), str)
        or not isinstance(setting, dict)
        or not isinstance(setting.get("category"), str)
        or not isinstance(setting.get("groups"), list)
        or not isinstance(summary.get("key_findings"), list)
    ):
        raise ValueError(f"gold summary does not match development schema: {arxiv_id}")
    return {
        "arxiv_id": arxiv_id,
        "model": record.get("model"),
        "summary": summary,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", required=True, type=Path)
    parser.add_argument("--gold-dir", required=True, type=Path)
    parser.add_argument("--public-source-pack", required=True, type=Path)
    parser.add_argument("--name", required=True)
    parser.add_argument("--out-dir", required=True, type=Path)
    args = parser.parse_args(argv)

    # A fresh destination prevents accidentally replacing an established benchmark.
    if args.out_dir.exists():
        raise ValueError("output already exists; choose a new pack directory")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", args.name):
        raise ValueError("pack name must be a simple identifier")
    rows = _load_jsonl(args.selection)
    ids = [str(row["arxiv_id"]) for row in rows]
    if any(not re.fullmatch(r"[0-9]{4}\.[0-9]{4,5}(?:v[0-9]+)?", value) for value in ids):
        raise ValueError("expected modern arXiv IDs, for example 2604.27351")
    if len(ids) != len(set(ids)):
        raise ValueError("selection contains duplicate arxiv_id values")

    taxonomy_source = args.public_source_pack / "public" / "taxonomy.json"
    rankings_source = (
        args.public_source_pack / "public" / "taxonomy_rankings.json"
    )
    for source in (taxonomy_source, rankings_source):
        if not source.is_file():
            raise FileNotFoundError(f"missing public pack resource: {source}")

    for arxiv_id in ids:
        source = args.gold_dir / "json" / f"{arxiv_id}.json"
        _validate_summary(json.loads(source.read_text(encoding="utf-8")), arxiv_id)

    public_dir = args.out_dir / "public"
    secret_dir = args.out_dir / "secret" / "gold"
    public_dir.mkdir(parents=True, exist_ok=True)
    secret_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(taxonomy_source, public_dir / taxonomy_source.name)
    shutil.copy2(rankings_source, public_dir / rankings_source.name)

    expected_names = {f"{arxiv_id}.json" for arxiv_id in ids}
    for stale in secret_dir.glob("*.json"):
        if stale.name not in expected_names:
            stale.unlink()
    for arxiv_id in ids:
        source = args.gold_dir / "json" / f"{arxiv_id}.json"
        if not source.is_file():
            raise FileNotFoundError(f"missing gold record: {source}")
        record = json.loads(source.read_text(encoding="utf-8"))
        compact = _validate_summary(record, arxiv_id)
        _atomic_text(
            secret_dir / f"{arxiv_id}.json",
            json.dumps(compact, ensure_ascii=False, indent=2) + "\n",
        )

    taxonomy = public_dir / "taxonomy.json"
    rankings = public_dir / "taxonomy_rankings.json"
    manifest = "\n".join(
        (
            "schema_version = 1",
            f'name = "{args.name}"',
            'kind = "development"',
            'protocol = "idea-recovery-v1"',
            f"target_count = {len(ids)}",
            f'taxonomy_sha256 = "{_file_sha256(taxonomy)}"',
            f'taxonomy_rankings_sha256 = "{_file_sha256(rankings)}"',
            f'gold_tree_sha256 = "{_gold_tree_sha256(secret_dir)}"',
            "",
        )
    )
    _atomic_text(args.out_dir / "pack.toml", manifest)
    print(
        json.dumps(
            {
                "status": "complete",
                "name": args.name,
                "target_count": len(ids),
                "out_dir": str(args.out_dir.resolve()),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
