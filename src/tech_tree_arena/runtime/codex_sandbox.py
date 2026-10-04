"""A deny-by-default OS boundary around the entire Codex process.

This development profile is not the attested hidden-target runner. No fallback
to an unsandboxed process is allowed on unsupported hosts.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

PROFILE_VERSION = "codex-linux-bwrap-proxy-v1" if sys.platform == "linux" else "codex-seatbelt-v1"
SUPPORTED_PROFILES = {"codex-seatbelt-v1", "codex-linux-bwrap-proxy-v1"}


def profile(workspace: Path, executable: Path, *, network: bool = True,
            read_roots: tuple[Path, ...] = ()) -> str:
    if sys.platform == "linux":
        from .linux_sandbox import specification
        return json.dumps(specification(workspace, executable, network=network, read_roots=read_roots), sort_keys=True)
    if sys.platform != "darwin" or not Path("/usr/bin/sandbox-exec").is_file():
        raise RuntimeError("Codex generator isolation currently requires macOS sandbox-exec; no unsafe fallback")
    workspace, executable = workspace.resolve(), executable.resolve()
    q = lambda value: json.dumps(str(value), ensure_ascii=False)
    lines = ["(version 1)", "(deny default)",
             "(allow file-read* (literal \"/\"))",
             "(allow file-read-metadata (literal \"/etc\") (literal \"/var\") (literal \"/tmp\"))",
             "(allow sysctl-read)", "(allow process-info* (target same-sandbox))",
             "(allow signal (target same-sandbox))", "(allow process-fork)",
             f"(allow process-exec (literal {q(executable)}))",
             f"(allow file-map-executable (literal {q(executable)}) (subpath \"/System\") (subpath \"/usr/lib\"))",
             "(allow file-read* file-map-executable (subpath \"/private/var/db/dyld\"))",
             "(allow system-mac-syscall (mac-policy-name \"vnguard\"))",
             "(allow system-mac-syscall (require-all (mac-policy-name \"Sandbox\") (mac-syscall-number 67)))",
             "(allow mach-lookup (global-name \"com.apple.system.opendirectoryd.libinfo\") (global-name \"com.apple.trustd.agent\") (global-name \"com.apple.SystemConfiguration.configd\"))",
             "(allow file-read* (subpath \"/System\") (subpath \"/usr/lib\") (subpath \"/usr/share\") (subpath \"/private/etc\"))",
             "(allow file-read* file-write* (literal \"/dev/null\"))",
             "(allow file-read* (literal \"/dev/urandom\") (literal \"/dev/random\"))",
             f"(allow file-read* (literal {q(executable)}))",
             f"(allow file-read* file-write* (subpath {q(workspace)}))"]
    # Directory traversal metadata only: never enumerate or read the parent.
    parents = set(workspace.parents) | set(executable.parents)
    for root in read_roots:
        root = root.resolve()
        parents.update(root.parents)
        kind = "subpath" if root.is_dir() else "literal"
        lines.append(f"(allow file-read* ({kind} {q(root)}))")
        lines.append(f"(allow file-map-executable ({kind} {q(root)}))")
    for parent in sorted(parents):
        lines.append(f"(allow file-read-metadata (literal {q(parent)}))")
    if network:
        # TLS is needed by the trusted Codex transport. No tools or local
        # service ports are permitted; loopback remains denied even on 443.
        lines.extend(["(allow mach-lookup (global-name \"com.apple.SystemConfiguration.DNSConfiguration\") (global-name \"com.apple.networkd\") (global-name \"com.apple.SecurityServer\"))",
                      "(allow system-socket (require-all (socket-domain AF_SYSTEM) (socket-protocol 2)))",
                      "(allow network-outbound (literal \"/private/var/run/mDNSResponder\"))",
                      "(allow network-outbound (remote tcp \"*:443\"))",
                      "(deny network-outbound (remote tcp \"localhost:*\"))",
                      "(allow network-outbound (remote udp \"*:53\"))"])
    return "\n".join(lines) + "\n"


def environment(workspace: Path) -> dict[str, str]:
    # No API keys, proxy settings, inherited CODEX_HOME, PYTHONPATH, DYLD_*,
    # personal config, or session identifiers cross the process boundary.
    for name in ("home", "tmp", "codex"):
        (workspace / name).mkdir(mode=0o700, exist_ok=True)
    return {"HOME": str(workspace / "home"), "CODEX_HOME": str(workspace / "codex"),
            "TMPDIR": str(workspace / "tmp"), "PATH": "/usr/bin:/bin",
            "LANG": "en_US.UTF-8", "RUST_LOG": "error"}


def command(workspace: Path, executable: Path, args: list[str], **kwargs) -> list[str]:
    if sys.platform == "linux":
        from .linux_sandbox import specification
        spec = specification(workspace, executable, **kwargs)
        spec["args"] = args
        return [str(Path(sys.executable).resolve()), "-I", "-S", str(Path(__file__).with_name("linux_sandbox.py")), json.dumps(spec)]
    policy = profile(workspace, executable, **kwargs)
    return ["/usr/bin/sandbox-exec", "-p", policy, str(executable), *args]
