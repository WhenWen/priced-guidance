"""Local submission manifests and entry-point loading."""

from __future__ import annotations

from .._compat import legacy_fields

import importlib
import ast
import hashlib
import io
import os
import shutil
import stat
import subprocess
import sys
import sysconfig
import tempfile
import tomllib
import zipfile
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

from ..contract import PROTOCOL_NAME
from ..errors import ValidationError
from ..resources import arena_home

MAX_ARCHIVE_FILES = 2_048
MAX_ARCHIVE_BYTES = 64 * 1024 * 1024
MAX_FILE_BYTES = 16 * 1024 * 1024
MAX_PATH_DEPTH = 16
STAGE_ORDER = ("directional", "essence", "strict")


@legacy_fields(oracle='guide')
@dataclass(frozen=True, slots=True)
class SubmissionManifest:
    root: Path
    name: str
    version: str
    protocol: str
    generator: str
    guide: str
    modules: dict[str, tuple[str, ...]] | None = None


def _dependency_environment_digest(
    lock_bytes: bytes, *, runtime_identity: str | None = None
) -> str:
    """Bind a locked environment cache to dependency data and Python ABI."""

    identity = runtime_identity or "\0".join(
        (
            sys.implementation.name,
            str(sys.implementation.cache_tag or ""),
            ".".join(str(part) for part in sys.version_info[:3]),
            sysconfig.get_platform(),
        )
    )
    digest = hashlib.sha256()
    digest.update(lock_bytes)
    digest.update(b"\0idea-arena-runtime\0")
    digest.update(identity.encode("utf-8"))
    return digest.hexdigest()


def _module_paths(value: object, *, group: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not value:
        raise ValidationError(f"submission module group {group!r} must be a non-empty array")
    result: list[str] = []
    for raw in value:
        if not isinstance(raw, str):
            raise ValidationError(f"submission module group {group!r} contains a non-string path")
        relative = _safe_relative(raw)
        if relative.as_posix() == "submission.toml":
            raise ValidationError("submission.toml cannot belong to a policy module")
        result.append(relative.as_posix())
    if len(set(result)) != len(result):
        raise ValidationError(f"submission module group {group!r} contains duplicate paths")
    return tuple(result)


def _module_files(manifest: SubmissionManifest) -> dict[str, tuple[Path, ...]]:
    """Expand and validate one complete, non-overlapping module partition."""

    if manifest.modules is None:
        return {}
    expanded: dict[str, tuple[Path, ...]] = {}
    owners: dict[Path, str] = {}
    for group, declared in manifest.modules.items():
        files: list[Path] = []
        for raw in declared:
            candidate = manifest.root / raw
            if not candidate.exists() or candidate.is_symlink():
                raise ValidationError(
                    f"submission module group {group!r} references a missing or unsafe path"
                )
            selected = (
                tuple(path for path in candidate.rglob("*") if path.is_file())
                if candidate.is_dir()
                else (candidate,)
            )
            if not selected:
                raise ValidationError(
                    f"submission module group {group!r} references an empty directory"
                )
            for path in selected:
                relative = path.relative_to(manifest.root)
                prior = owners.get(relative)
                if prior is not None:
                    raise ValidationError(
                        f"submission file {relative.as_posix()!r} belongs to both "
                        f"{prior!r} and {group!r}"
                    )
                owners[relative] = group
                files.append(relative)
        expanded[group] = tuple(sorted(files, key=lambda path: path.as_posix()))

    all_files = {
        path.relative_to(manifest.root)
        for path in manifest.root.rglob("*")
        if path.is_file() and path != manifest.root / "submission.toml"
    }
    unowned = sorted(all_files - set(owners), key=lambda path: path.as_posix())
    if unowned:
        raise ValidationError(
            "modular submission leaves files outside the module partition: "
            + ", ".join(path.as_posix() for path in unowned[:8])
        )
    required_shared = {
        path
        for path in (Path("participant/__init__.py"), Path("pyproject.toml"), Path("uv.lock"))
        if (manifest.root / path).is_file()
    }
    for entrypoint in (manifest.generator, manifest.guide):
        module_name = entrypoint.partition(":")[0]
        relative = Path(*module_name.split("."))
        candidates = (relative.with_suffix(".py"), relative / "__init__.py")
        required_shared.add(next(path for path in candidates if (manifest.root / path).is_file()))
    misplaced = sorted(
        (path for path in required_shared if owners.get(path) != "shared"),
        key=lambda path: path.as_posix(),
    )
    if misplaced:
        raise ValidationError(
            "modular submission must keep entrypoints and dependency manifests shared: "
            + ", ".join(path.as_posix() for path in misplaced)
        )
    _validate_module_import_boundaries(manifest.root, owners)
    return expanded


def _python_module_file(root: Path, module_name: str) -> Path | None:
    """Resolve a local Python module name without importing participant code."""

    if not module_name:
        return None
    relative = Path(*module_name.split("."))
    for candidate in (relative.with_suffix(".py"), relative / "__init__.py"):
        if (root / candidate).is_file():
            return candidate
    return None


def _module_name(relative: Path) -> tuple[str, str]:
    """Return the import name and package for one local Python source file."""

    if relative.name == "__init__.py":
        parts = relative.parent.parts
        name = ".".join(parts)
        return name, name
    parts = relative.with_suffix("").parts
    name = ".".join(parts)
    return name, ".".join(parts[:-1])


def _imported_local_files(root: Path, relative: Path) -> set[Path]:
    """Return statically named local modules imported by ``relative``.

    Literal imports are part of a module's executable dependency boundary.  A
    modular policy may dynamically dispatch to the currently active stage, but
    it may not hide a shared dispatcher or a stage dependency behind a module
    group whose hash is allowed to change later.
    """

    try:
        tree = ast.parse(
            (root / relative).read_text(encoding="utf-8"),
            filename=relative.as_posix(),
        )
    except (OSError, UnicodeError, SyntaxError) as exc:
        raise ValidationError(
            f"modular submission contains invalid Python source {relative.as_posix()!r}"
        ) from exc

    _, package = _module_name(relative)
    imported_names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_names.update(alias.name for alias in node.names)
            continue
        if not isinstance(node, ast.ImportFrom):
            continue
        if node.level:
            package_parts = package.split(".") if package else []
            remove = node.level - 1
            if remove > len(package_parts):
                continue
            base_parts = package_parts[: len(package_parts) - remove]
            if node.module:
                base_parts.extend(node.module.split("."))
            base = ".".join(base_parts)
        else:
            base = node.module or ""
        if base:
            imported_names.add(base)
        for alias in node.names:
            if alias.name != "*":
                imported_names.add(".".join(part for part in (base, alias.name) if part))

    imported: set[Path] = set()
    for name in imported_names:
        resolved = _python_module_file(root, name)
        if resolved is not None:
            imported.add(resolved)
        parts = name.split(".")
        for index in range(1, len(parts)):
            package_init = Path(*parts[:index]) / "__init__.py"
            if (root / package_init).is_file():
                imported.add(package_init)
    return imported


def _validate_module_import_boundaries(root: Path, owners: dict[Path, str]) -> None:
    """Ensure local static imports cannot bypass frozen stage hashes."""

    for relative, owner in owners.items():
        if relative.suffix != ".py":
            continue
        allowed = {"shared"} if owner == "shared" else {"shared", owner}
        for dependency in _imported_local_files(root, relative):
            dependency_owner = owners.get(dependency)
            if dependency_owner not in allowed:
                raise ValidationError(
                    "modular submission import crosses a frozen module boundary: "
                    f"{relative.as_posix()} ({owner}) imports "
                    f"{dependency.as_posix()} ({dependency_owner or 'unowned'})"
                )


def stage_module_hashes(manifest: SubmissionManifest) -> dict[str, str]:
    """Return content hashes for the shared and three ordered policy modules."""

    groups = _module_files(manifest)
    result: dict[str, str] = {}
    for group, files in groups.items():
        digest = hashlib.sha256()
        for relative in files:
            digest.update(relative.as_posix().encode("utf-8"))
            digest.update(b"\0")
            digest.update(hashlib.sha256((manifest.root / relative).read_bytes()).digest())
        result[group] = digest.hexdigest()
    return result


def _safe_relative(name: str) -> Path:
    if not name or "\x00" in name or "\\" in name:
        raise ValidationError("submission archive contains an invalid path")
    path = Path(name)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise ValidationError("submission archive contains path traversal")
    if len(path.parts) > MAX_PATH_DEPTH:
        raise ValidationError("submission archive path is too deep")
    return path


def _validate_tree(root: Path) -> None:
    seen: set[str] = set()
    files = 0
    total = 0
    for path in root.rglob("*"):
        relative = path.relative_to(root)
        if "__pycache__" in relative.parts or path.suffix.casefold() in {".pyc", ".pyo"}:
            raise ValidationError("submission may not contain compiled Python bytecode")
        folded = relative.as_posix().casefold()
        if folded in seen:
            raise ValidationError("submission contains a case-folding path collision")
        seen.add(folded)
        if path.is_symlink():
            raise ValidationError("submission may not contain symbolic links")
        mode = path.stat(follow_symlinks=False).st_mode
        if path.is_dir():
            continue
        if not stat.S_ISREG(mode):
            raise ValidationError("submission may contain only regular files and directories")
        files += 1
        size = path.stat().st_size
        total += size
        if files > MAX_ARCHIVE_FILES or total > MAX_ARCHIVE_BYTES or size > MAX_FILE_BYTES:
            raise ValidationError("submission exceeds the file or size limit")
        if len(relative.parts) > MAX_PATH_DEPTH:
            raise ValidationError("submission path is too deep")
    if (root / "pyproject.toml").is_file() and not (root / "uv.lock").is_file():
        raise ValidationError("submissions with dependencies require uv.lock")


def _validate_archive_infos(infos: list[zipfile.ZipInfo]) -> None:
    """Reject unsafe archive metadata before creating anything in the build cache."""
    if len(infos) > MAX_ARCHIVE_FILES:
        raise ValidationError("submission archive contains too many entries")
    total = 0
    seen: set[str] = set()
    for info in infos:
        relative = _safe_relative(info.filename.rstrip("/"))
        folded = relative.as_posix().casefold()
        if folded in seen:
            raise ValidationError(
                "submission archive contains a duplicate or case-folding collision"
            )
        seen.add(folded)
        mode = (info.external_attr >> 16) & 0xFFFF
        file_type = stat.S_IFMT(mode)
        if file_type and not (stat.S_ISREG(mode) or stat.S_ISDIR(mode)):
            raise ValidationError("submission archive contains a link or special file")
        if info.flag_bits & 0x1:
            raise ValidationError("encrypted submission archives are unsupported")
        total += info.file_size
        if info.file_size > MAX_FILE_BYTES or total > MAX_ARCHIVE_BYTES:
            raise ValidationError("submission archive exceeds the size limit")
        if info.compress_size and info.file_size / info.compress_size > 1_000:
            raise ValidationError("submission archive has an unsafe compression ratio")


def prepare_submission(path: str | Path) -> Path:
    """Validate a directory or safely materialize a zip into the arena build cache."""
    requested = Path(path).expanduser().resolve()
    if requested.is_dir():
        _validate_tree(requested)
        return requested
    if not requested.is_file() or requested.suffix.casefold() != ".zip":
        raise ValidationError("submission must be a directory or zip archive")
    archive_bytes = requested.read_bytes()
    digest = hashlib.sha256(archive_bytes).hexdigest()
    try:
        with zipfile.ZipFile(io.BytesIO(archive_bytes)) as archive:
            _validate_archive_infos(archive.infolist())
    except (OSError, zipfile.BadZipFile) as exc:
        raise ValidationError("submission archive is invalid") from exc
    destination = arena_home() / "builds" / digest
    if destination.is_dir():
        _validate_tree(destination)
        return destination
    builds = destination.parent
    builds.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f"{digest[:12]}-", dir=builds))
    try:
        with zipfile.ZipFile(io.BytesIO(archive_bytes)) as archive:
            infos = archive.infolist()
            _validate_archive_infos(infos)
            for info in infos:
                relative = _safe_relative(info.filename.rstrip("/"))
                output = temporary / relative
                if info.is_dir():
                    output.mkdir(parents=True, exist_ok=True)
                else:
                    output.parent.mkdir(parents=True, exist_ok=True)
                    with archive.open(info) as source, output.open("xb") as target:
                        shutil.copyfileobj(source, target)
        roots = [temporary]
        if not (temporary / "submission.toml").is_file():
            children = [child for child in temporary.iterdir() if child.is_dir()]
            files = [child for child in temporary.iterdir() if child.is_file()]
            roots = children if len(children) == 1 and not files else []
        if len(roots) != 1 or not (roots[0] / "submission.toml").is_file():
            raise ValidationError("archive must contain one submission.toml at its root")
        source_root = roots[0]
        _validate_tree(source_root)
        if source_root == temporary:
            temporary.replace(destination)
        else:
            source_root.replace(destination)
            temporary.rmdir()
        return destination
    except (OSError, zipfile.BadZipFile) as exc:
        raise ValidationError("submission archive is invalid") from exc
    finally:
        if temporary.exists():
            shutil.rmtree(temporary, ignore_errors=True)


def build_dependency_paths(root: str | Path) -> tuple[Path, ...]:
    """Build locked third-party dependencies for the unverified local runner.

    Participant source itself is never installed and no target has been loaded at
    this point. Official hidden builds belong in the external hardened builder.
    """
    root = Path(root).resolve()
    if not (root / "pyproject.toml").is_file():
        return ()
    uv = shutil.which("uv")
    if uv is None:
        raise ValidationError("uv is required to build a submission with dependencies")
    digest = _dependency_environment_digest((root / "uv.lock").read_bytes())
    environment = arena_home() / "build-envs" / digest
    marker = environment / ".idea-arena-ready"
    if not marker.is_file():
        command = [
            uv,
            "sync",
            "--frozen",
            "--no-dev",
            "--no-install-project",
            "--project",
            str(root),
            "--python",
            sys.executable,
        ]
        if os.environ.get("IDEA_ARENA_OFFLINE_BUILD") == "1":
            command.append("--offline")
        environment.parent.mkdir(parents=True, exist_ok=True)
        process_environment = os.environ.copy()
        process_environment["UV_PROJECT_ENVIRONMENT"] = str(environment)
        try:
            completed = subprocess.run(
                command,
                env=process_environment,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=900,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise ValidationError("locked dependency build failed") from exc
        if completed.returncode != 0:
            raise ValidationError("locked dependency build failed")
        marker.write_text(digest + "\n", encoding="utf-8")
    candidates = [
        environment / "Lib" / "site-packages",
        environment
        / "lib"
        / f"python{sys.version_info.major}.{sys.version_info.minor}"
        / "site-packages",
    ]
    paths = tuple(path.resolve() for path in candidates if path.is_dir())
    if not paths:
        raise ValidationError("locked dependency environment has no site-packages directory")
    return paths


def load_manifest(root: str | Path) -> SubmissionManifest:
    root = prepare_submission(root)
    path = root / "submission.toml"
    if not root.is_dir() or not path.is_file():
        raise ValidationError("submission directory must contain submission.toml")
    try:
        data = tomllib.loads(path.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise ValidationError("submission.toml is invalid") from exc
    if data.get("schema_version") != 1:
        raise ValidationError("unsupported submission schema_version")
    if "guide" in data:
        if "oracle" in data and data["oracle"] != data["guide"]:
            raise ValidationError("submission manifest has conflicting guide and oracle entrypoints")
        data["oracle"] = data.pop("guide")
    required = ("name", "version", "protocol", "generator", "oracle")
    if set(data) - {"schema_version", *required, "modules"}:
        raise ValidationError("submission manifest contains unknown fields")
    if any(not isinstance(data.get(key), str) or not data[key] for key in required):
        raise ValidationError("submission manifest is missing a required string")
    if data["protocol"] != PROTOCOL_NAME:
        raise ValidationError(f"this build supports protocol {PROTOCOL_NAME}")
    for entrypoint in (data["generator"], data["oracle"]):
        module = entrypoint.partition(":")[0]
        if module != "participant" and not module.startswith("participant."):
            raise ValidationError("participant entry points must live under participant/")
    raw_modules = data.get("modules")
    modules = None
    if raw_modules is not None:
        if not isinstance(raw_modules, dict) or set(raw_modules) != {
            "shared",
            *STAGE_ORDER,
        }:
            raise ValidationError(
                "submission modules must define shared, directional, essence, and strict"
            )
        modules = {
            group: _module_paths(raw_modules[group], group=group)
            for group in ("shared", *STAGE_ORDER)
        }
    manifest = SubmissionManifest(root, *(data[key] for key in required), modules)
    _module_files(manifest)
    return manifest


@contextmanager
def _submission_import_path(root: Path) -> Iterator[None]:
    old_path = list(sys.path)
    old_dont_write = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(root))
    try:
        yield
    finally:
        sys.path[:] = old_path
        sys.dont_write_bytecode = old_dont_write


def _load_entrypoint(root: Path, value: str) -> type:
    module_name, separator, attribute = value.partition(":")
    if not separator or not module_name or not attribute:
        raise ValidationError("entry points must use module:Class syntax")
    try:
        module = importlib.import_module(module_name)
        loaded = getattr(module, attribute)
    except (ImportError, AttributeError) as exc:
        raise ValidationError(f"could not import entry point {value!r}") from exc
    if not isinstance(loaded, type) or not callable(getattr(loaded, "step", None)):
        raise ValidationError(f"entry point {value!r} is not a participant class")
    return loaded


def load_participant_classes(manifest: SubmissionManifest) -> tuple[type, type]:
    prior = {name: module for name, module in sys.modules.items() if name == "participant" or name.startswith("participant.")}
    for name in prior:
        sys.modules.pop(name, None)
    try:
        with _submission_import_path(manifest.root):
            generator = _load_entrypoint(manifest.root, manifest.generator)
            guide = _load_entrypoint(manifest.root, manifest.guide)
        return generator, guide
    finally:
        for name in list(sys.modules):
            if name == "participant" or name.startswith("participant."):
                sys.modules.pop(name, None)
        sys.modules.update(prior)


def entrypoint_is_defined(manifest: SubmissionManifest, entrypoint: str) -> bool:
    """Whether the entrypoint's module binds the attribute, without executing it.

    Used to fail fast on optional entrypoints the run profile selects (for
    example the agent oracle behind ``--oracle-agent``) before any actor
    subprocess is spawned. Accepts a top-level class definition, a top-level
    assignment to the name, or an explicit import alias.
    """

    module_name, _, attribute = entrypoint.partition(":")
    if not module_name or not attribute:
        return False
    relative = Path(*module_name.split("."))
    candidates = (manifest.root / relative.with_suffix(".py"), manifest.root / relative / "__init__.py")
    source = next((path for path in candidates if path.is_file()), None)
    if source is None:
        return False
    try:
        tree = ast.parse(source.read_text(encoding="utf-8"), filename=source.name)
    except (OSError, UnicodeError, SyntaxError):
        return False
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == attribute:
            return True
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == attribute
            for target in node.targets
        ):
            return True
        if (
            isinstance(node, ast.AnnAssign)
            and isinstance(node.target, ast.Name)
            and node.target.id == attribute
            and node.value is not None
        ):
            return True
        if isinstance(node, ast.ImportFrom) and any(
            (alias.asname or alias.name) == attribute for alias in node.names
        ):
            return True
    return False


def validate_entrypoint_sources(manifest: SubmissionManifest) -> None:
    """Validate entrypoint shape without executing participant module code."""
    for entrypoint in (manifest.generator, manifest.guide):
        module_name, _, attribute = entrypoint.partition(":")
        relative = Path(*module_name.split("."))
        candidates = (manifest.root / relative.with_suffix(".py"), manifest.root / relative / "__init__.py")
        source = next((path for path in candidates if path.is_file()), None)
        if source is None:
            raise ValidationError(f"entrypoint module {module_name!r} has no Python source file")
        try:
            tree = ast.parse(source.read_text(encoding="utf-8"), filename=source.name)
        except (OSError, UnicodeError, SyntaxError) as exc:
            raise ValidationError(f"entrypoint module {module_name!r} is not valid Python source") from exc
        class_match = any(
            isinstance(node, ast.ClassDef)
            and node.name == attribute
            and any(isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)) and child.name == "step"
                    for child in node.body)
            for node in tree.body
        )
        alias_match = any(
            isinstance(node, ast.ImportFrom)
            and any((alias.asname or alias.name) == attribute for alias in node.names)
            for node in tree.body
        )
        if not (class_match or alias_match):
            raise ValidationError(
                f"entrypoint {entrypoint!r} must define a class with step() or import an explicit class alias"
            )
