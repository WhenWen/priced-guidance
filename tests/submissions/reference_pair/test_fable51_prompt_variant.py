from __future__ import annotations

import ast
from pathlib import Path

from tech_tree_arena.submission_io import load_manifest


ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT / "submissions" / "reference_pair"
VARIANT = ROOT / "submissions" / "reference_pair_fable51"


def _literal_assignment(path: Path, name: str):
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == name
            for target in node.targets
        ):
            return ast.literal_eval(node.value)
        if (
            isinstance(node, ast.AnnAssign)
            and isinstance(node.target, ast.Name)
            and node.target.id == name
        ):
            return ast.literal_eval(node.value)
    raise AssertionError(f"missing literal assignment {name}")


def test_fable51_variant_is_a_valid_standalone_submission() -> None:
    manifest = load_manifest(VARIANT)

    assert manifest.name == "reference-pair-fable51"
    assert manifest.version == "1.17.5"
    assert manifest.generator == "participant.generator:Generator"
    assert manifest.guide == "participant.guide:Guide"


def test_fable51_variant_changes_only_guide_output_wording() -> None:
    base_files = {
        path.relative_to(BASE)
        for path in BASE.rglob("*")
        if path.is_file()
    }
    variant_files = {
        path.relative_to(VARIANT)
        for path in VARIANT.rglob("*")
        if path.is_file()
    }
    assert base_files == variant_files

    expected_differences = {
        Path("README.md"),
        Path("submission.toml"),
        Path("participant/pair.py"),
    }
    actual_differences = {
        relative
        for relative in base_files
        if (BASE / relative).read_bytes() != (VARIANT / relative).read_bytes()
    }
    assert actual_differences == expected_differences


def test_fable51_guide_contract_avoids_reasoning_extraction_language() -> None:
    pair_path = VARIANT / "participant" / "pair.py"
    prompt = _literal_assignment(pair_path, "_GUIDE_SYSTEM_PROMPT")
    schema = _literal_assignment(pair_path, "_GUIDE_ACTION_SCHEMA")

    forbidden = (
        "<concise private rationale>",
        "<compact durable private state>",
        "not hidden chain-of-thought",
        "saved for the ablation transcript",
        "Concise private rationale for this exact action.",
        "Compact private belief, gaps, failed paths, and next plan.",
    )
    rendered_contract = prompt + repr(schema)
    assert not any(phrase in rendered_contract for phrase in forbidden)

    assert schema["required"] == [
        "reasoning",
        "state_summary",
        "action",
        "option_id",
        "question_id",
    ]
    assert "brief justification" in prompt
    assert "compact summary" in prompt
    assert schema["properties"]["reasoning"]["description"] == (
        "Brief justification for this exact action."
    )
    assert schema["properties"]["state_summary"]["description"] == (
        "Compact summary of prior choices, gaps, and next plan."
    )
