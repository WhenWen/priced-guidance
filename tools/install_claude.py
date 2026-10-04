#!/usr/bin/env python3
"""Install a checksum-pinned official Claude Code distribution, never from PATH."""
import argparse
import base64
import hashlib
import io
import json
import os
from pathlib import Path
import platform
import subprocess
import tarfile
import urllib.request

ROOT = Path(__file__).resolve().parents[1]


def install(destination):
    lock_path = ROOT / "tools/claude/release.lock.json"
    lock = json.loads(lock_path.read_text())
    arch = {"aarch64": "arm64", "arm64": "arm64", "x86_64": "x64"}.get(platform.machine())
    key = platform.system().lower() + "-" + str(arch)
    package = lock["platforms"][key]
    destination.mkdir(parents=True, exist_ok=True)
    executable = destination / "bin/claude"
    receipt_path = destination / "install-receipt.json"
    if receipt_path.exists():
        receipt = json.loads(receipt_path.read_text())
        if receipt["release"] != lock or hashlib.sha256(executable.read_bytes()).hexdigest() != receipt["executable_sha256"]:
            raise RuntimeError("Existing install differs; select a new install directory")
        return receipt_path
    if executable.exists():
        raise RuntimeError("Refusing to overwrite an unreceipted executable")
    archive = urllib.request.urlopen(package["url"], timeout=120).read()
    integrity = "sha512-" + base64.b64encode(hashlib.sha512(archive).digest()).decode()
    if integrity != package["integrity"]:
        raise RuntimeError("Official package checksum mismatch")
    # Extract just one ordinary executable; never extract archive paths/links.
    with tarfile.open(fileobj=io.BytesIO(archive), mode="r:gz") as tar:
        matches = [m for m in tar.getmembers() if m.isfile() and Path(m.name).name == "claude"]
        if len(matches) != 1:
            raise RuntimeError("Expected exactly one native claude executable")
        binary = tar.extractfile(matches[0]).read()
    executable.parent.mkdir(exist_ok=True)
    executable.write_bytes(binary)
    executable.chmod(0o755)
    version = subprocess.check_output([str(executable), "--version"], text=True, timeout=30).strip()
    if not version.startswith(lock["version"] + " "):
        raise RuntimeError("Installed version differs from release lock")
    receipt = {"release": lock, "platform": key, "executable": str(executable),
               "executable_sha256": hashlib.sha256(binary).hexdigest(),
               "release_lock_sha256": hashlib.sha256(lock_path.read_bytes()).hexdigest(), "version": version}
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")
    return receipt_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--install-dir", type=Path, default=ROOT / ".idea-arena/claude-code")
    args = parser.parse_args()
    os.umask(0o077)
    print(install(args.install_dir.expanduser().resolve()))
