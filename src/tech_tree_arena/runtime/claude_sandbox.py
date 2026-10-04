"""Linux-only Claude Code isolation, with a stable namespace-local cwd."""
import json
from pathlib import Path
import sys

from .linux_sandbox import specification

PROFILE_VERSION = "claude-linux-bwrap-proxy-v1"


def command(workspace, executable, args, *, network=True, read_roots=()):
    if sys.platform != "linux":
        raise RuntimeError("Claude Generator currently requires Linux bubblewrap; no unsafe fallback")
    spec = specification(workspace, executable, network=network,
                         network_provider="anthropic", workspace_target="/arena", read_roots=read_roots)
    spec["args"] = args
    return [str(Path(sys.executable).resolve()), "-I", "-S",
            str(Path(__file__).with_name("linux_sandbox.py")), json.dumps(spec)]


def environment(workspace):
    for name in ("home", "tmp", "claude"):
        (workspace / name).mkdir(exist_ok=True, mode=0o700)
    return {"HOME": "/arena/home", "CLAUDE_CONFIG_DIR": "/arena/claude", "TMPDIR": "/arena/tmp",
            "PATH": "/usr/bin:/bin", "LANG": "C.UTF-8", "DISABLE_AUTOUPDATER": "1",
            "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1", "CLAUDE_CODE_DISABLE_FEEDBACK_SURVEY": "1",
            "CLAUDE_CODE_DISABLE_AUTO_MEMORY": "1", "CLAUDE_CODE_SAFE_MODE": "1",
            "CLAUDE_CODE_MAX_OUTPUT_TOKENS": "50000"}
