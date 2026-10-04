"""Tiny local adapter that enables Anthropic prompt caching on OpenRouter.

Codex speaks the OpenAI Responses wire format and emits ``prompt_cache_key``,
but OpenRouter's Anthropic routes require a top-level ``cache_control`` object.
This proxy adds that provider-specific field without changing Codex itself.  It
also maps Codex's cache key to OpenRouter's ``session_id`` for sticky routing.

The proxy never logs prompts, credentials, or response bodies.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import http.client
import json
import signal
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit


_HOP_BY_HOP_HEADERS = {
    "connection",
    "content-length",
    "host",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailer",
    "transfer-encoding",
    "upgrade",
}


def prepare_openrouter_body(
    body: dict[str, Any], *, cache_ttl: str = "1h"
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return an OpenRouter-compatible request and non-sensitive diagnostics."""

    prepared = copy.deepcopy(body)
    cache_control: dict[str, str] = {"type": "ephemeral"}
    if cache_ttl == "1h":
        cache_control["ttl"] = "1h"
    elif cache_ttl not in {"5m", "ephemeral"}:
        raise ValueError("cache_ttl must be '5m', 'ephemeral', or '1h'")
    prepared.setdefault("cache_control", cache_control)

    cache_key = prepared.get("prompt_cache_key")
    if not isinstance(prepared.get("session_id"), str) or not prepared["session_id"]:
        if isinstance(cache_key, str) and cache_key:
            material = cache_key
        else:
            # This fallback remains stable as a resumed conversation grows:
            # instructions, model, and the first input item do not change.
            input_items = prepared.get("input")
            first_input = (
                input_items[0]
                if isinstance(input_items, list) and input_items
                else None
            )
            material = json.dumps(
                {
                    "model": prepared.get("model"),
                    "instructions": prepared.get("instructions"),
                    "first_input": first_input,
                },
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
        prepared["session_id"] = "codex-" + hashlib.sha256(
            material.encode("utf-8")
        ).hexdigest()

    input_value = prepared.get("input")
    diagnostics = {
        "model": prepared.get("model"),
        "cache_control": copy.deepcopy(prepared.get("cache_control")),
        "prompt_cache_key_present": isinstance(cache_key, str) and bool(cache_key),
        "session_id_sha256": hashlib.sha256(
            str(prepared["session_id"]).encode("utf-8")
        ).hexdigest(),
        "input_items": len(input_value) if isinstance(input_value, list) else None,
    }
    return prepared, diagnostics


class _ProxyServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(
        self,
        server_address: tuple[str, int],
        *,
        upstream_base_url: str,
        cache_ttl: str,
        log_file: Path | None,
        timeout_seconds: float,
    ) -> None:
        super().__init__(server_address, _ProxyHandler)
        parsed = urlsplit(upstream_base_url.rstrip("/"))
        if parsed.scheme not in {"http", "https"} or not parsed.hostname:
            raise ValueError("upstream_base_url must be an http(s) URL")
        self.upstream = parsed
        self.cache_ttl = cache_ttl
        self.log_file = log_file
        self.timeout_seconds = timeout_seconds
        self.log_lock = threading.Lock()

    def audit(self, row: dict[str, Any]) -> None:
        if self.log_file is None:
            return
        record = {"timestamp": time.time(), **row}
        line = json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n"
        with self.log_lock:
            with self.log_file.open("a", encoding="utf-8") as stream:
                stream.write(line)


class _ProxyHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server: _ProxyServer

    def log_message(self, _format: str, *_args: Any) -> None:
        return

    def do_POST(self) -> None:  # noqa: N802
        length = int(self.headers.get("Content-Length") or 0)
        raw_body = self.rfile.read(length)
        diagnostics: dict[str, Any] = {"path": self.path, "request_bytes": length}
        if self.path.rstrip("/").endswith("/responses"):
            try:
                decoded = json.loads(raw_body)
            except (json.JSONDecodeError, UnicodeDecodeError):
                decoded = None
            if isinstance(decoded, dict):
                prepared, cache_diagnostics = prepare_openrouter_body(
                    decoded, cache_ttl=self.server.cache_ttl
                )
                raw_body = json.dumps(
                    prepared, ensure_ascii=False, separators=(",", ":")
                ).encode("utf-8")
                diagnostics.update(cache_diagnostics)
        self._forward("POST", raw_body, diagnostics)

    def do_GET(self) -> None:  # noqa: N802
        self._forward("GET", None, {"path": self.path, "request_bytes": 0})

    def _forward(
        self, method: str, body: bytes | None, diagnostics: dict[str, Any]
    ) -> None:
        upstream = self.server.upstream
        base_path = upstream.path.rstrip("/")
        request_path = base_path + (
            self.path if self.path.startswith("/") else "/" + self.path
        )
        headers = {
            key: value
            for key, value in self.headers.items()
            if key.lower() not in _HOP_BY_HOP_HEADERS
        }
        if body is not None:
            headers["Content-Length"] = str(len(body))
        connection_type = (
            http.client.HTTPSConnection
            if upstream.scheme == "https"
            else http.client.HTTPConnection
        )
        connection = connection_type(
            upstream.hostname, upstream.port, timeout=self.server.timeout_seconds
        )
        status = 502
        try:
            connection.request(method, request_path, body=body, headers=headers)
            response = connection.getresponse()
            status = response.status
            self.send_response(response.status, response.reason)
            for key, value in response.getheaders():
                if key.lower() not in _HOP_BY_HOP_HEADERS:
                    self.send_header(key, value)
            self.send_header("Connection", "close")
            self.end_headers()
            while True:
                chunk = response.read(64 * 1024)
                if not chunk:
                    break
                self.wfile.write(chunk)
                self.wfile.flush()
        except (OSError, http.client.HTTPException) as exc:
            if not self.wfile.closed:
                message = json.dumps({"error": {"message": str(exc)}}).encode("utf-8")
                try:
                    self.send_response(502)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(message)))
                    self.send_header("Connection", "close")
                    self.end_headers()
                    self.wfile.write(message)
                except OSError:
                    pass
        finally:
            diagnostics["upstream_status"] = status
            self.server.audit(diagnostics)
            self.close_connection = True
            connection.close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port-file", type=Path, required=True)
    parser.add_argument(
        "--upstream-base-url", default="https://openrouter.ai/api/v1"
    )
    parser.add_argument("--cache-ttl", choices=("5m", "1h"), default="1h")
    parser.add_argument("--log-file", type=Path)
    parser.add_argument("--timeout-seconds", type=float, default=900.0)
    args = parser.parse_args(argv)

    server = _ProxyServer(
        ("127.0.0.1", 0),
        upstream_base_url=args.upstream_base_url,
        cache_ttl=args.cache_ttl,
        log_file=args.log_file,
        timeout_seconds=args.timeout_seconds,
    )
    args.port_file.write_text(str(server.server_port), encoding="ascii")

    def stop(_signum: int, _frame: Any) -> None:
        threading.Thread(target=server.shutdown, daemon=True).start()

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    try:
        server.serve_forever(poll_interval=0.1)
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
