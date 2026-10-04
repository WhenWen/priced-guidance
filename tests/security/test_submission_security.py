import os
import zipfile
from pathlib import Path

import pytest

from tech_tree_arena.errors import ValidationError
from tech_tree_arena.submission_io.manifest import (
    load_manifest,
    prepare_submission,
    stage_module_hashes,
)


def _write_modular_fixture(root: Path) -> None:
    participant = root / "participant"
    stages = participant / "stages"
    stages.mkdir(parents=True)
    (root / "submission.toml").write_text(
        """schema_version = 1
name = "modular-fixture"
version = "1"
protocol = "idea-recovery-v1"
generator = "participant.generator:Generator"
oracle = "participant.oracle:Oracle"

[modules]
shared = ["participant/__init__.py", "participant/generator.py", "participant/oracle.py"]
directional = ["participant/stages/directional.py"]
essence = ["participant/stages/essence.py"]
strict = ["participant/stages/strict.py"]
""",
        encoding="utf-8",
    )
    (participant / "__init__.py").write_text("", encoding="utf-8")
    (participant / "generator.py").write_text("class Generator: pass\n", encoding="utf-8")
    (participant / "oracle.py").write_text("class Oracle: pass\n", encoding="utf-8")
    for stage in ("directional", "essence", "strict"):
        (stages / f"{stage}.py").write_text(f"NAME = {stage!r}\n", encoding="utf-8")


def test_zip_path_traversal_fails_closed(tmp_path: Path) -> None:
    archive = tmp_path / "bad.zip"
    with zipfile.ZipFile(archive, "w") as output:
        output.writestr("../escape", "bad")
        output.writestr("submission.toml", "schema_version = 1")
    with pytest.raises(ValidationError, match="traversal"):
        prepare_submission(archive)


def test_zip_case_collision_fails_closed(tmp_path: Path) -> None:
    archive = tmp_path / "bad.zip"
    with zipfile.ZipFile(archive, "w") as output:
        output.writestr("submission.toml", "schema_version = 1")
        output.writestr("participant/A.py", "")
        output.writestr("participant/a.py", "")
    with pytest.raises(ValidationError, match="collision"):
        prepare_submission(archive)


def test_directory_symlink_fails_closed(tmp_path: Path) -> None:
    (tmp_path / "submission.toml").write_text("schema_version = 1")
    os.symlink(tmp_path / "submission.toml", tmp_path / "linked")
    with pytest.raises(ValidationError, match="symbolic"):
        prepare_submission(tmp_path)


def test_manifest_unknown_fields_fail_closed(tmp_path: Path) -> None:
    (tmp_path / "submission.toml").write_text(
        """schema_version = 1
name = "x"
version = "1"
protocol = "idea-recovery-v1"
generator = "participant.generator:Generator"
oracle = "participant.oracle:Oracle"
judge = "participant.judge:Judge"
"""
    )
    with pytest.raises(ValidationError, match="unknown"):
        load_manifest(tmp_path)


def test_modular_manifest_requires_a_complete_file_partition(tmp_path: Path) -> None:
    _write_modular_fixture(tmp_path)
    (tmp_path / "participant" / "unowned.txt").write_text("hidden prompt", encoding="utf-8")

    with pytest.raises(ValidationError, match="outside the module partition"):
        load_manifest(tmp_path)


def test_stage_hash_changes_only_for_the_edited_module(tmp_path: Path) -> None:
    _write_modular_fixture(tmp_path)
    before = stage_module_hashes(load_manifest(tmp_path))
    (tmp_path / "participant/stages/essence.py").write_text(
        "NAME = 'essence-v2'\n", encoding="utf-8"
    )
    after = stage_module_hashes(load_manifest(tmp_path))

    assert after["essence"] != before["essence"]
    assert after["shared"] == before["shared"]
    assert after["directional"] == before["directional"]
    assert after["strict"] == before["strict"]


def test_modular_entrypoint_cannot_import_dispatcher_from_mutable_stage(
    tmp_path: Path,
) -> None:
    _write_modular_fixture(tmp_path)
    manifest_path = tmp_path / "submission.toml"
    manifest_path.write_text(
        manifest_path.read_text(encoding="utf-8").replace(
            'shared = ["participant/__init__.py", "participant/generator.py", "participant/oracle.py"]',
            'shared = ["participant/__init__.py", "participant/generator.py", "participant/oracle.py"]\n'
            'directional = ["participant/stages/directional.py"]\n'
            'essence = ["participant/pair.py", "participant/stages/essence.py"]\n'
            'strict = ["participant/stages/strict.py"]',
        ).replace(
            'directional = ["participant/stages/directional.py"]\n'
            'essence = ["participant/stages/essence.py"]\n'
            'strict = ["participant/stages/strict.py"]',
            "",
        ),
        encoding="utf-8",
    )
    (tmp_path / "participant/generator.py").write_text(
        "from participant.pair import Generator\n", encoding="utf-8"
    )
    (tmp_path / "participant/pair.py").write_text(
        "class Generator: pass\n", encoding="utf-8"
    )

    with pytest.raises(ValidationError, match="crosses a frozen module boundary"):
        load_manifest(tmp_path)


def test_stage_cannot_import_dependency_from_a_later_mutable_stage(tmp_path: Path) -> None:
    _write_modular_fixture(tmp_path)
    manifest_path = tmp_path / "submission.toml"
    manifest_path.write_text(
        manifest_path.read_text(encoding="utf-8").replace(
            'essence = ["participant/stages/essence.py"]',
            'essence = ["participant/stages/essence.py"]\n'
            'strict = ["participant/strict_helper.py", "participant/stages/strict.py"]',
        ).replace('strict = ["participant/stages/strict.py"]\n', "", 1),
        encoding="utf-8",
    )
    (tmp_path / "participant/stages/essence.py").write_text(
        "from participant.strict_helper import VALUE\n", encoding="utf-8"
    )
    (tmp_path / "participant/strict_helper.py").write_text("VALUE = 1\n", encoding="utf-8")

    with pytest.raises(ValidationError, match="crosses a frozen module boundary"):
        load_manifest(tmp_path)
