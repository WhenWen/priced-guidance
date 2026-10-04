"""Rootless namespaces with a narrow, host-owned TLS CONNECT relay.

The sandbox has no host network interface, home, repository, or host /proc.
Its only egress is a Unix socket that accepts CONNECT to exact OpenAI hosts
on port 443. TLS remains end-to-end; the relay never sees auth or model text.
This module is also an isolated stdlib-only supervisor/inner relay executable.
"""
from __future__ import annotations

import ipaddress
import json
import os
from pathlib import Path
import select
import shutil
import socket
import socketserver
import subprocess
import sys
import tempfile
import threading

ALLOWED_HOSTS = frozenset({"chatgpt.com", "api.openai.com", "auth.openai.com", "ab.chatgpt.com"})
PROVIDER_HOSTS = {"openai": ALLOWED_HOSTS,
                  "anthropic": frozenset({"api.anthropic.com", "claude.ai", "platform.claude.com",
                                           "console.anthropic.com", "platform.anthropic.com"})}


def specification(workspace, executable, *, network=True, read_roots=(), network_provider="openai", workspace_target=None):
    if network_provider not in PROVIDER_HOSTS:
        raise ValueError("Unknown sandbox network provider")
    bwrap = os.environ.get("IDEA_ARENA_BWRAP") or shutil.which("bwrap")
    if not bwrap or not Path(bwrap).is_file():
        raise RuntimeError("Linux Generator requires bubblewrap; set IDEA_ARENA_BWRAP. No unsafe fallback.")
    return {"bwrap": str(Path(bwrap).resolve()), "workspace": str(Path(workspace).resolve()),
            "executable": str(Path(executable).resolve()), "network": bool(network),
            "read_roots": [str(Path(p).resolve()) for p in read_roots],
            "python": str(Path(sys.executable).resolve()), "python_prefix": str(Path(sys.base_prefix).resolve()),
            "network_provider": network_provider,
            **({"workspace_target": workspace_target} if workspace_target else {})}


def tunnel(left, right):
    while True:
        ready, _, _ = select.select([left, right], [], [], 60)
        for source in ready:
            data = source.recv(65536)
            if not data:
                return
            (right if source is left else left).sendall(data)


def connect_destination(header, allowed_hosts=ALLOWED_HOSTS):
    """Reject arbitrary hosts, ports, URL forms, and non-public DNS answers."""
    first = header.split(b"\r\n", 1)[0].decode("ascii")
    method, destination, protocol = first.split(" ")
    if method != "CONNECT" or protocol not in {"HTTP/1.0", "HTTP/1.1"}:
        raise ValueError("Only TLS CONNECT is allowed")
    host, port = destination.rsplit(":", 1)
    if host not in allowed_hosts or port != "443":
        raise ValueError("Destination is not allowed")
    addresses = socket.getaddrinfo(host, 443, family=socket.AF_INET, type=socket.SOCK_STREAM)
    if not addresses or any(not ipaddress.ip_address(a[4][0]).is_global for a in addresses):
        raise ValueError("Non-public address rejected")
    return addresses


class Egress(socketserver.BaseRequestHandler):
    def handle(self):
        try:
            self.request.settimeout(15)
            header = bytearray()
            # Read exactly the CONNECT header; do not swallow early TLS bytes.
            while not header.endswith(b"\r\n\r\n"):
                chunk = self.request.recv(1)
                if not chunk or len(header) >= 8192:
                    raise ValueError("Invalid CONNECT header")
                header.extend(chunk)
            addresses = connect_destination(bytes(header), self.server.allowed_hosts)
            upstream = None
            for family, kind, proto, _, address in addresses:
                attempt = socket.socket(family, kind, proto)
                attempt.settimeout(15)
                try:
                    attempt.connect(address)
                    upstream = attempt
                    break
                except OSError:
                    attempt.close()
            if upstream is None:
                raise OSError("Upstream unavailable")
            with upstream:
                self.request.sendall(b"HTTP/1.1 200 Connection established\r\n\r\n")
                self.request.settimeout(None)
                upstream.settimeout(None)
                tunnel(self.request, upstream)
        except (OSError, ValueError, UnicodeError):
            try:
                self.request.sendall(b"HTTP/1.1 403 Forbidden\r\nConnection: close\r\n\r\n")
            except OSError:
                pass


class UnixServer(socketserver.ThreadingUnixStreamServer):
    allowed_hosts = ALLOWED_HOSTS
    daemon_threads = True
    block_on_close = False
    def handle_error(self, request, address):
        pass  # Never log request headers or TLS payloads.


class TCPServer(socketserver.ThreadingTCPServer):
    daemon_threads = True
    block_on_close = False
    def handle_error(self, request, address):
        pass


def inner(socket_path, executable, args):
    class Bridge(socketserver.BaseRequestHandler):
        def handle(self):
            try:
                with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as upstream:
                    upstream.connect(socket_path)
                    tunnel(self.request, upstream)
            except OSError:
                pass
    with TCPServer(("127.0.0.1", 18080), Bridge) as relay:
        threading.Thread(target=relay.serve_forever, daemon=True).start()
        env = dict(os.environ)
        for key in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"):
            env[key] = "http://127.0.0.1:18080"
        env["NO_PROXY"] = env["no_proxy"] = ""
        try:
            return subprocess.call([executable, *args], env=env)
        finally:
            relay.shutdown()


def supervise(spec):
    workspace = Path(spec["workspace"])
    invocation = [spec["bwrap"], "--unshare-user", "--unshare-pid", "--unshare-net",
                  "--unshare-ipc", "--unshare-uts", "--disable-userns", "--die-with-parent",
                  "--new-session", "--cap-drop", "ALL", "--proc", "/proc", "--dev", "/dev",
                  "--tmpfs", "/tmp"]
    # Shared libraries only; /usr/bin, home, repository, sockets and host /proc
    # are absent. Python's standalone prefix is needed for the wire worker and
    # inner network relay. All host runtime mounts are read-only.
    roots = {Path("/usr/lib"), Path("/usr/lib64"), Path(spec["executable"]),
             *(Path(p) for p in spec["read_roots"])}
    if spec["network"]:
        roots.update({Path(spec["python_prefix"]), Path(spec["python"]),
                      Path("/etc/ssl/certs"), Path(__file__).resolve()})
    for root in sorted(roots):
        if root.exists():
            invocation += ["--ro-bind", str(root), str(root)]
    for name in ("lib", "lib64"):
        if Path("/" + name).is_symlink():
            invocation += ["--symlink", os.readlink("/" + name), "/" + name]
        elif Path("/" + name).is_dir():
            invocation += ["--ro-bind", "/" + name, "/" + name]
    target = spec.get("workspace_target", str(workspace))
    invocation += ["--bind", str(workspace), target, "--chdir", target]
    if not spec["network"]:
        return subprocess.call([*invocation, spec["executable"], *spec["args"]])
    # The socket lives outside the writable workspace and is mounted alone.
    # AF_UNIX paths are capped at 108 bytes; nested per-run TMPDIRs can exceed
    # that limit. The host-owned 0700 directory is not mounted into the child.
    with tempfile.TemporaryDirectory(prefix="arena-egress-", dir="/tmp") as directory:
        socket_path = str(Path(directory) / "relay.sock")
        with UnixServer(socket_path, Egress) as relay:
            relay.allowed_hosts = PROVIDER_HOSTS[spec.get("network_provider", "openai")]
            os.chmod(socket_path, 0o600)
            invocation += ["--ro-bind", socket_path, "/run/arena-egress.sock"]
            threading.Thread(target=relay.serve_forever, daemon=True).start()
            try:
                return subprocess.call([*invocation, spec["python"], "-I", "-S", str(Path(__file__).resolve()),
                                        "--inner", "/run/arena-egress.sock", spec["executable"], *spec["args"]])
            finally:
                relay.shutdown()


if __name__ == "__main__":
    if sys.argv[1] == "--inner":
        raise SystemExit(inner(sys.argv[2], sys.argv[3], sys.argv[4:]))
    raise SystemExit(supervise(json.loads(sys.argv[1])))
