#!/usr/bin/env python3
"""Build the pinned OSS Codex; never resolve an installed codex from PATH."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
LOCK = ROOT / "tools/codex/cache-source.lock.json"


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-dir", type=Path, default=ROOT / ".idea-arena/codex-cache-v1")
    parser.add_argument("--source-lock", type=Path, default=LOCK)
    parser.add_argument("--jobs", type=int, default=6)
    args = parser.parse_args()
    root = args.build_dir.resolve()
    root.mkdir(parents=True, exist_ok=True)
    lock_path = args.source_lock.resolve()
    lock = json.loads(lock_path.read_text())
    source = root / "source"

    def run(*command: str, cwd: Path = root, env=None) -> None:
        subprocess.run(command, cwd=cwd, env=env, check=True)

    if not source.exists():
        run("git", "clone", "--depth", "1", "--branch", lock["tag"], lock["repository"], str(source))
    source = source.resolve()
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=source, text=True).strip()
    if commit != lock["commit"]:
        raise SystemExit(f"Source commit differs from {lock_path}")
    for name in lock["patches"]:
        patch = lock_path.parent / name
        applied = subprocess.run(["git", "apply", "--reverse", "--check", str(patch)], cwd=source, capture_output=True)
        if applied.returncode:
            run("git", "apply", "--check", str(patch), cwd=source)
            run("git", "apply", str(patch), cwd=source)
    expected_diff = b"".join((lock_path.parent / name).read_bytes() for name in lock["patches"])
    actual_diff = subprocess.check_output(["git", "diff", "--binary", "HEAD"], cwd=source)
    if actual_diff != expected_diff or subprocess.check_output(["git", "ls-files", "--others", "--exclude-standard"], cwd=source):
        raise SystemExit("Source has unrecorded changes; refusing to create a build receipt")
    env = {**os.environ, "CARGO_HOME": str((root / "cargo").resolve()), "RUSTUP_HOME": str((root / "rustup").resolve()),
           "RUSTUP_TOOLCHAIN": lock["rust_toolchain"], "CARGO_TARGET_DIR": str((root / "target").resolve()),
           "CARGO_PROFILE_DEV_DEBUG": "0", "CARGO_BUILD_JOBS": str(args.jobs)}
    env["PATH"] = str(root / "cargo/bin") + os.pathsep + os.environ.get("PATH", "")
    cargo = root / "cargo/bin/cargo"
    if not cargo.exists():
        installer = root / "rustup-init.sh"
        urllib.request.urlretrieve("https://sh.rustup.rs", installer)
        run("sh", str(installer), "-y", "--profile", "minimal", "--default-toolchain", lock["rust_toolchain"], "--no-modify-path", env=env)
    run(str(cargo), "build", "--locked", "--bin", "codex", cwd=source / "codex-rs", env=env)
    # Isolate the runtime executable from compiler artifacts and the source tree.
    binary = root / "bin/codex"
    binary.parent.mkdir(exist_ok=True)
    shutil.copy2(root / "target/debug/codex", binary)
    receipt = {"schema_version": 1, "source": lock, "source_lock_sha256": digest(lock_path),
               "patch_sha256": {name: digest(lock_path.parent / name) for name in lock["patches"]},
               "cargo_lock_sha256": digest(source / "codex-rs/Cargo.lock"),
               "executable": str(binary), "executable_sha256": digest(binary),
               "version": subprocess.check_output([str(binary), "--version"], text=True).strip()}
    (root / "build-receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(root / "build-receipt.json")


if __name__ == "__main__":
    main()
