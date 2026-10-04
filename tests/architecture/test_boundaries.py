import ast
from pathlib import Path

from tech_tree_arena.submission_io.manifest import load_manifest, load_participant_classes
from tech_tree_arena.targets.loader import load_target_pack


ROOT = Path(__file__).resolve().parents[2]


def test_runtime_and_submissions_do_not_import_legacy_core() -> None:
    assert not (ROOT / "src" / "core").exists()
    offenders = []
    for source_root in (ROOT / "src", ROOT / "submissions"):
        for path in source_root.rglob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in ast.walk(tree):
                names = []
                if isinstance(node, ast.Import):
                    names = [alias.name for alias in node.names]
                elif isinstance(node, ast.ImportFrom) and node.module:
                    names = [node.module]
                if any(name == "core" or name.startswith("core.") for name in names):
                    offenders.append(str(path.relative_to(ROOT)))
    assert offenders == []


def test_reference_pair_owns_real_participant_classes() -> None:
    manifest = load_manifest(ROOT / "submissions" / "reference_pair")
    generator, oracle = load_participant_classes(manifest)

    assert generator.__module__ == "participant.pair"
    assert oracle.__module__ == "participant.pair"
    assert not (ROOT / "src" / "tech_tree_arena" / "reference.py").exists()


def test_reference_pair_imports_only_the_public_arena_surface() -> None:
    offenders = []
    participant = ROOT / "submissions" / "reference_pair" / "participant"
    for path in participant.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                if node.module.startswith("tech_tree_arena."):
                    offenders.append((str(path.relative_to(ROOT)), node.module))
            elif isinstance(node, ast.Import):
                offenders.extend(
                    (str(path.relative_to(ROOT)), alias.name)
                    for alias in node.names
                    if alias.name.startswith("tech_tree_arena.")
                )
    assert offenders == []


def test_reference_template_has_one_checked_in_source() -> None:
    assert (ROOT / "submissions" / "reference_pair" / "submission.toml").is_file()
    assert not (ROOT / "src" / "tech_tree_arena" / "data" / "submission_templates").exists()
    source_manifest = (ROOT / "MANIFEST.in").read_text(encoding="utf-8")
    assert "graft submissions/reference_pair" in source_manifest
    assert "graft submissions/examples/minimal_pair" in source_manifest


def test_taxonomy_is_a_public_target_pack_resource() -> None:
    resources = load_target_pack("development40").public_resources()

    assert "deep learning" in resources["taxonomy"]
    assert "deep learning" in resources["taxonomy_rankings"]
