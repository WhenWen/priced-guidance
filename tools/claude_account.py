#!/usr/bin/env python3
"""Manage one explicit Linux Claude login, separate from ~/.claude and API keys."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from tech_tree_arena.runtime.claude_generator import account_document, installation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--install-receipt")
    parser.add_argument("--auth-home", type=Path, default=Path.home() / ".local/share/idea-arena/claude-account")
    parser.add_argument("action", choices=("login", "status"))
    parser.add_argument("--email")
    args = parser.parse_args()
    if sys.platform != "linux":
        parser.error("Use this independent file-based login on Linux; it does not import macOS Keychain credentials")
    os.umask(0o077)
    account = args.auth_home.expanduser().resolve()
    account.mkdir(parents=True, exist_ok=True, mode=0o700)
    if args.action == "status":
        try:
            doc = account_document(account)["claudeAiOauth"]
            print(json.dumps({"logged_in": True, "subscription_type": doc.get("subscriptionType"),
                              "expires_at_ms": doc.get("expiresAt"), "auth_home": str(account)}))
            return 0
        except (OSError, ValueError, RuntimeError):
            print(json.dumps({"logged_in": False, "auth_home": str(account)}))
            return 1
    _, receipt = installation(args.install_receipt)
    for name in ("login-home", "tmp"):
        (account / name).mkdir(exist_ok=True, mode=0o700)
    env = {"HOME": str(account / "login-home"), "CLAUDE_CONFIG_DIR": str(account),
           "TMPDIR": str(account / "tmp"), "PATH": "/usr/bin:/bin", "LANG": "C.UTF-8",
           "DISABLE_AUTOUPDATER": "1", "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1"}
    command = [receipt["executable"], "auth", "login", "--claudeai"]
    if args.email:
        command += ["--email", args.email]
    with (account / "auth.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        # Official interactive login only. No model process or data is involved.
        code = subprocess.call(command, cwd=account / "login-home", env=env)
        if code == 0:
            account_document(account)
            (account / ".credentials.json").chmod(0o600)
        return code


if __name__ == "__main__":
    raise SystemExit(main())
