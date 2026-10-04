"""Stage only the participant, its declared dependencies, and the wire contract."""
from __future__ import annotations

from pathlib import Path
import shutil
import sys
import tempfile

from .codex_sandbox import command, environment


def stage_worker(root: Path, dependency_paths: tuple[Path, ...]):
    temporary = tempfile.TemporaryDirectory(prefix="arena-generator-")
    workspace = Path(temporary.name).resolve()
    try:
        for tree in (root, *dependency_paths):
            if any(path.is_symlink() for path in tree.rglob("*")):
                raise RuntimeError("Sandboxed Generator trees must not contain symlinks")
        participant = workspace / "participant-source"
        shutil.copytree(root, participant)
        dependencies = []
        for index, tree in enumerate(dependency_paths):
            destination = workspace / f"dependency-{index}"
            shutil.copytree(tree, destination)
            dependencies.append(destination)
        source = Path(__file__).resolve().parents[1]
        package = workspace / "runtime/tech_tree_arena"
        (package / "runtime").mkdir(parents=True)
        for name in ("__init__.py", "errors.py"):
            shutil.copyfile(source / name, package / name)
        shutil.copytree(source / "contract", package / "contract", ignore=shutil.ignore_patterns("__pycache__"))
        (package / "runtime/__init__.py").write_text('"""Isolated participant wire runtime."""\n')
        for name in ("worker.py", "wire.py"):
            shutil.copyfile(source / "runtime" / name, package / "runtime" / name)
        executable = Path(sys.executable).resolve()
        bootstrap = "import sys,runpy;sys.path.insert(0," + repr(str(workspace / "runtime")) + ");runpy.run_module('tech_tree_arena.runtime.worker',run_name='__main__')"
        invocation = command(workspace, executable, ["-I", "-S", "-B", "-c", bootstrap],
                             network=False, read_roots=(Path(sys.base_prefix),))
        return temporary, participant, tuple(dependencies), {
            "args": invocation, "cwd": workspace, "env": environment(workspace)}
    except BaseException:
        temporary.cleanup()
        raise
