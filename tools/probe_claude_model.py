#!/usr/bin/env python3
"""Inspect an explicit model through Claude Code's local /model command."""
import argparse
import fcntl
import json
from pathlib import Path
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from tech_tree_arena.runtime.claude_generator import account_document, installation, run_cli
from tech_tree_arena.runtime.claude_sandbox import environment


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--auth-home", type=Path, required=True)
    p.add_argument("--model", default="claude-fable-5-1")
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if not args.model.startswith("claude-") or any(c.isspace() for c in args.model):
        p.error("Select an explicit Claude model ID")
    args.output.mkdir(parents=True, exist_ok=False)
    executable = installation()[1]["executable"]
    with (args.auth_home / "auth.lock").open("a") as lock, tempfile.TemporaryDirectory(prefix="arena-claude-model-") as directory:
        fcntl.flock(lock, fcntl.LOCK_EX)
        workspace = Path(directory)
        environment(workspace)
        credentials = workspace / "claude/.credentials.json"
        credentials.write_text(json.dumps(account_document(args.auth_home)))
        credentials.chmod(0o600)
        # /model is a documented local command in print mode, not a model prompt.
        command = ["--print", "--output-format", "stream-json", "--verbose", "--safe-mode",
                   "--tools", "", "--strict-mcp-config", "--mcp-config", '{"mcpServers":{}}',
                   "--setting-sources", "", "--no-chrome", "--max-turns", "1"]
        try:
            code, events = run_cli(executable, workspace, command, "/model " + args.model, 120)
            (args.output / "events.json").write_text(json.dumps(events, indent=2) + "\n")
            print(json.dumps({"exit_code": code, "events": events}, indent=2))
            return code
        finally:
            updated = account_document(credentials.parent)
            pending = args.auth_home / ".credentials.pending"
            pending.write_text(json.dumps(updated))
            pending.chmod(0o600)
            pending.replace(args.auth_home / ".credentials.json")


if __name__ == "__main__":
    raise SystemExit(main())
