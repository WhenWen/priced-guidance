import pytest

from tech_tree_arena.submission_io.manifest import (
    SubmissionManifest,
    _dependency_environment_digest,
    entrypoint_is_defined,
)


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("AgentOracle: type = BaseOracle\n", True),
        ("AgentOracle: type\n", False),
    ],
)
def test_entrypoint_detection_handles_annotated_assignments(
    tmp_path, source: str, expected: bool
) -> None:
    (tmp_path / "oracle.py").write_text(source, encoding="utf-8")
    manifest = SubmissionManifest(
        root=tmp_path,
        name="test",
        version="1",
        protocol="test",
        generator="generator:Generator",
        guide="oracle:Oracle",
    )

    assert entrypoint_is_defined(manifest, "oracle:AgentOracle") is expected


def test_dependency_environment_digest_includes_runtime_identity() -> None:
    lock = b"version = 1\n"

    first = _dependency_environment_digest(
        lock, runtime_identity="cpython-3.12-macos-arm64"
    )
    second = _dependency_environment_digest(
        lock, runtime_identity="cpython-3.13-macos-arm64"
    )

    assert first != second
    assert first == _dependency_environment_digest(
        lock, runtime_identity="cpython-3.12-macos-arm64"
    )
