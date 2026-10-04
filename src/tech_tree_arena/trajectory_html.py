"""Self-contained HTML rendering for one or more Arena trajectories."""

from __future__ import annotations

import argparse
import html
import json
import os
from datetime import UTC, datetime
from importlib.resources import files
from pathlib import Path
from typing import Any

from .errors import ArenaError


class TrajectoryRenderError(ArenaError):
    """Raised when a trajectory input cannot be rendered safely."""


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise TrajectoryRenderError(f"could not read trajectory input {path}") from exc
    except json.JSONDecodeError as exc:
        raise TrajectoryRenderError(f"trajectory input is not valid JSON: {path}") from exc


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise TrajectoryRenderError(f"could not read trajectory input {path}") from exc
    for line_number, raw in enumerate(lines, 1):
        if not raw.strip():
            continue
        try:
            row = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise TrajectoryRenderError(
                f"invalid JSON on line {line_number} of {path}"
            ) from exc
        if not isinstance(row, dict):
            raise TrajectoryRenderError(
                f"line {line_number} of {path} is not an event object"
            )
        rows.append(row)
    return rows


def _optional_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    value = _read_json(path)
    return value if isinstance(value, dict) else {}


def _normalize_bundle(
    value: dict[str, Any], *, source: Path, fallback_id: str, privacy: str
) -> dict[str, Any]:
    events = value.get("events")
    if not isinstance(events, list) or not all(isinstance(row, dict) for row in events):
        raise TrajectoryRenderError(f"trajectory bundle in {source} has no event-object list")
    manifest = value.get("manifest") if isinstance(value.get("manifest"), dict) else {}
    status = value.get("status") if isinstance(value.get("status"), dict) else {}
    score = value.get("score") if isinstance(value.get("score"), dict) else {}
    usage = value.get("usage") if isinstance(value.get("usage"), dict) else {}
    trajectory_id = str(
        value.get("id")
        or manifest.get("run_id")
        or status.get("run_id")
        or fallback_id
    )
    declared_privacy = str(value.get("privacy") or "public").casefold()
    effective_privacy = (
        "private" if privacy == "private" or declared_privacy == "private" else "public"
    )
    return {
        "id": trajectory_id,
        "label": str(value.get("label") or trajectory_id),
        "source": str(source),
        "privacy": effective_privacy,
        "manifest": manifest,
        "status": status,
        "score": score,
        "usage": usage,
        "events": events,
    }


def load_trajectory_input(path: str | Path, *, private: bool = False) -> list[dict[str, Any]]:
    """Load a run directory, JSONL event stream, or JSON trajectory bundle."""
    source = Path(path).expanduser().resolve()
    privacy = "private" if private else "public"
    if source.is_dir():
        event_name = "events.private.jsonl" if private else "events.public.jsonl"
        event_path = source / event_name
        if not event_path.is_file():
            raise TrajectoryRenderError(f"run directory has no {event_name}: {source}")
        manifest = _optional_json(source / "manifest.json")
        status = _optional_json(source / "status.json")
        score = _optional_json(source / "score.json")
        usage = _optional_json(source / "usage.json")
        run_id = str(manifest.get("run_id") or status.get("run_id") or source.name)
        return [
            {
                "id": run_id,
                "label": run_id,
                "source": str(source),
                "privacy": privacy,
                "manifest": manifest,
                "status": status,
                "score": score,
                "usage": usage,
                "events": _read_jsonl(event_path),
            }
        ]
    if not source.is_file():
        raise TrajectoryRenderError(f"trajectory input does not exist: {source}")
    if source.suffix.casefold() == ".jsonl":
        return [
            {
                "id": source.stem,
                "label": source.stem,
                "source": str(source),
                "privacy": privacy,
                "manifest": {},
                "status": {},
                "score": {},
                "usage": {},
                "events": _read_jsonl(source),
            }
        ]
    value = _read_json(source)
    if isinstance(value, list):
        if not all(isinstance(row, dict) for row in value):
            raise TrajectoryRenderError(f"trajectory array contains non-object values: {source}")
        return [
            {
                "id": source.stem,
                "label": source.stem,
                "source": str(source),
                "privacy": privacy,
                "manifest": {},
                "status": {},
                "score": {},
                "usage": {},
                "events": value,
            }
        ]
    if not isinstance(value, dict):
        raise TrajectoryRenderError(f"trajectory JSON must be an object or array: {source}")
    bundles = value.get("trajectories")
    if isinstance(bundles, list):
        normalized: list[dict[str, Any]] = []
        for index, bundle in enumerate(bundles, 1):
            if not isinstance(bundle, dict):
                raise TrajectoryRenderError(
                    f"trajectory {index} in {source} is not an object"
                )
            normalized.append(
                _normalize_bundle(
                    bundle,
                    source=source,
                    fallback_id=f"{source.stem}-{index}",
                    privacy=privacy,
                )
            )
        return normalized
    return [
        _normalize_bundle(
            value,
            source=source,
            fallback_id=source.stem,
            privacy=privacy,
        )
    ]


def _script_json(value: Any) -> str:
    # Prevent user-controlled text from terminating the inert JSON script tag.
    return (
        json.dumps(value, ensure_ascii=False, separators=(",", ":"))
        .replace("<", "\\u003c")
        .replace("\u2028", "\\u2028")
        .replace("\u2029", "\\u2029")
    )


def render_trajectory_html(
    inputs: list[str | Path] | tuple[str | Path, ...],
    output: str | Path,
    *,
    private: bool = False,
    title: str = "Idea Arena trajectories",
) -> Path:
    """Render inputs to one portable HTML file and return its absolute path."""
    if not inputs:
        raise TrajectoryRenderError("at least one trajectory input is required")
    trajectories = [
        trajectory
        for source in inputs
        for trajectory in load_trajectory_input(source, private=private)
    ]
    template = (
        files("tech_tree_arena")
        .joinpath("templates", "trajectory.html")
        .read_text(encoding="utf-8")
    )
    generated_at = datetime.now(UTC).isoformat(timespec="seconds")
    document = (
        template.replace("{{TITLE}}", html.escape(title))
        .replace("{{GENERATED_AT}}", html.escape(generated_at))
        .replace("{{TRAJECTORY_DATA}}", _script_json({"trajectories": trajectories}))
    )
    destination = Path(output).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    try:
        temporary.write_text(document, encoding="utf-8")
        os.chmod(temporary, 0o600)
        temporary.replace(destination)
    except OSError as exc:
        raise TrajectoryRenderError(f"could not write HTML report {destination}") from exc
    return destination


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="render-trajectories",
        description="Render Arena run directories or trajectory JSON/JSONL as one HTML report.",
    )
    parser.add_argument("inputs", nargs="+", help="run directories or JSON/JSONL trajectory files")
    parser.add_argument("--output", "-o", default="trajectory-report.html")
    parser.add_argument("--title", default="Idea Arena trajectories")
    parser.add_argument(
        "--private",
        action="store_true",
        help="read events.private.jsonl from run directories; output may contain secrets",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        output = render_trajectory_html(
            args.inputs,
            args.output,
            private=args.private,
            title=args.title,
        )
    except ArenaError as exc:
        raise SystemExit(str(exc)) from exc
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
