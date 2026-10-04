#!/usr/bin/env python3
"""Log in independently using the source-built Codex, without touching ~/.codex.

The login-only process is not an agent and does not run in the Generator
sandbox. Model execution always uses the separately enforced sandbox.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import fcntl
import shutil
import tempfile
from datetime import datetime, timezone

from tech_tree_arena.runtime.codex_generator import configuration, import_account, require_account


def quota_exhausted(limits):
    bucket = (limits.get("rateLimitsByLimitId") or {}).get("codex") or limits.get("rateLimits") or {}
    return bool(bucket.get("rateLimitReachedType") or any(
        (bucket.get(window) or {}).get("usedPercent", 0) >= 100 for window in ("primary", "secondary")))


def maybe_reset(rpc, identity, limits, reset_key, expected_email):
    if not reset_key:
        return None
    if not expected_email or identity.get("email") != expected_email:
        raise RuntimeError("Reset account identity differs from the authorized experiment")
    if not quota_exhausted(limits):
        return {"outcome": "notNeeded"}
    return rpc.call("account/rateLimitResetCredit/consume", {"idempotencyKey": reset_key})


def inspect_account(config, model=None, thread_id=None, *, reset_key=None, expected_email=None,
                    include_token_usage=True, refresh=False):
    """Read this independent account through the source-built sandboxed server."""
    from tech_tree_arena.runtime.codex_rpc import CodexRPC
    from tech_tree_arena.runtime.codex_sandbox import environment
    account = Path(config["auth_home"])
    require_account(account)
    executable = Path(json.loads(Path(config["build_receipt"]).read_text())["executable"])
    with (account / "auth.lock").open("a") as lock, tempfile.TemporaryDirectory(prefix="arena-account-") as directory:
        fcntl.flock(lock, fcntl.LOCK_EX)
        workspace = Path(directory).resolve()
        env = environment(workspace)
        auth = Path(env["CODEX_HOME"]) / "auth.json"
        shutil.copyfile(account / "auth.json", auth)
        os.chmod(auth, 0o600)
        (auth.parent / "config.toml").write_text('forced_login_method="chatgpt"\ncli_auth_credentials_store="file"\n')
        rpc = CodexRPC(executable, workspace, timeout=60)
        try:
            rpc.call("initialize", {"clientInfo": {"name": "idea_arena", "version": "0.1.0"},
                                    "capabilities": {"experimentalApi": True}})
            rpc.send({"method": "initialized"})
            identity = rpc.call("account/read", {"refreshToken": refresh}).get("account") or {}
            limits = rpc.call("account/rateLimits/read", {})
            reset = maybe_reset(rpc, identity, limits, reset_key, expected_email)
            if reset is not None:
                limits = rpc.call("account/rateLimits/read", {})
            limits.pop("accountId", None)
            limits.pop("rateLimitUpsell", None)
            resets = limits.get("rateLimitResetCredits")
            if resets:
                limits["rateLimitResetCredits"] = {"availableCount": resets.get("availableCount")}
            result = {"captured_at": datetime.now(timezone.utc).isoformat(),
                      "email": identity.get("email"), "plan_type": identity.get("planType"),
                      "rate_limits": limits}
            if reset is not None:
                result["reset"] = reset
            if model:
                models = []
                cursor = None
                while True:
                    page = rpc.call("model/list", {"includeHidden": True, "limit": 100, "cursor": cursor})
                    models.extend(row for row in page["data"] if row.get("model") == model)
                    cursor = page.get("nextCursor")
                    if not cursor:
                        break
                result["models"] = models
            try:
                if not include_token_usage:
                    return result
                params = {"threadId": thread_id} if thread_id else {}
                if thread_id:
                    result["requested_thread_id"] = thread_id
                result["token_usage"] = rpc.call("account/usage/read", params)
            except RuntimeError as exc:
                result["token_usage_unavailable"] = str(exc)[:500]
            return result
        finally:
            rpc.close()
            pending = account / "auth.pending"
            shutil.copyfile(auth, pending)
            os.chmod(pending, 0o600)
            os.replace(pending, account / "auth.json")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--auth-home", type=Path)
    parser.add_argument("--build-receipt", type=Path)
    commands = parser.add_subparsers(dest="action", required=True)
    login = commands.add_parser("login", help="sign in to a separate ChatGPT/Codex account")
    login.add_argument("--device-auth", action="store_true", help="use device code instead of browser callback")
    commands.add_parser("status", help="show login method without printing credentials")
    inspect = commands.add_parser("inspect", help="read independent account usage and optional model capabilities")
    inspect.add_argument("--model")
    inspect.add_argument("--thread-id", help="read backend-estimated credits and tokens for a native Codex thread")
    inspect.add_argument("--output", type=Path)
    imported = commands.add_parser("import-existing", help="explicitly copy a saved ChatGPT auth cache")
    imported.add_argument("--source", type=Path, default=Path.home() / ".codex/auth.json")
    args = parser.parse_args()
    config = configuration(args.build_receipt, args.auth_home)
    account = Path(config["auth_home"])
    account.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(account, 0o700)
    if args.action == "inspect":
        value = json.dumps(inspect_account(config, args.model, args.thread_id), indent=2) + "\n"
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(value)
        print(value, end="")
        return
    if args.action == "import-existing":
        if (account / "auth.json").exists():
            raise SystemExit("This account directory already has credentials; select a new --auth-home")
        import_account(account, args.source.expanduser().resolve())
        print("Imported account credentials into " + str(account))
        return
    receipt = json.loads(Path(config["build_receipt"]).read_text())
    # Inherit only a small login environment; always remove inference keys and
    # route all auth and configuration writes to the selected account directory.
    env = {key: value for key, value in os.environ.items()
           if key in {"HOME", "PATH", "LANG", "TERM", "TMPDIR", "DISPLAY"}}
    env["CODEX_HOME"] = str(account)
    command = [receipt["executable"], "-c", 'forced_login_method="chatgpt"',
               "-c", 'cli_auth_credentials_store="file"', "login"]
    if args.action == "status":
        command.append("status")
    elif args.device_auth:
        command.append("--device-auth")
    raise SystemExit(subprocess.call(command, env=env, cwd=account))


if __name__ == "__main__":
    main()
