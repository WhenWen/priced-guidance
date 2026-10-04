"""Small localhost-only, server-rendered development arena."""

from __future__ import annotations

import html
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib.resources import files
from typing import Any

from .resources import arena_home


def public_runs() -> list[dict[str, Any]]:
    root = arena_home() / "runs"
    rows: list[dict[str, Any]] = []
    if not root.is_dir():
        return rows
    for directory in sorted(root.iterdir(), key=lambda path: path.stat().st_mtime, reverse=True):
        try:
            manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
            score = json.loads((directory / "score.json").read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        row = {
            "run_id": manifest.get("run_id"),
            "submission": manifest.get("submission_name"),
            "target_pack": manifest.get("target_pack"),
            "disclosure": manifest.get("disclosure"),
            "status": score.get("status"),
            "score": score.get("score"),
        }
        if manifest.get("disclosure") != "hidden":
            row.update({"target_id": manifest.get("target_id"), "K": score.get("K")})
        rows.append(row)
    return rows


def render_home() -> str:
    template = files("tech_tree_arena").joinpath("templates", "index.html").read_text(encoding="utf-8")
    rows = []
    for run in public_runs():
        rows.append(
            "<tr>"
            f"<td><code>{html.escape(str(run.get('run_id', '')))}</code></td>"
            f"<td>{html.escape(str(run.get('submission', '')))}</td>"
            f"<td>{html.escape(str(run.get('target_pack', '')))}</td>"
            f"<td>{html.escape(str(run.get('status', '')))}</td>"
            f"<td>{html.escape(str(run.get('score', '')))}</td>"
            f"<td>{html.escape(str(run.get('K', 'hidden')))}</td>"
            "</tr>"
        )
    return template.replace("{{RUN_ROWS}}", "\n".join(rows) or '<tr><td colspan="6">No runs yet.</td></tr>')


class ArenaHandler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:  # noqa: N802
        if self.path == "/":
            payload = render_home().encode("utf-8")
            content_type = "text/html; charset=utf-8"
        elif self.path == "/api/runs":
            payload = json.dumps({"runs": public_runs()}, sort_keys=True).encode("utf-8")
            content_type = "application/json"
        else:
            self.send_error(404)
            return
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Content-Security-Policy", "default-src 'none'; style-src 'unsafe-inline'")
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, format: str, *args: Any) -> None:
        return


def serve(host: str = "127.0.0.1", port: int = 8765) -> None:
    ThreadingHTTPServer((host, port), ArenaHandler).serve_forever()
