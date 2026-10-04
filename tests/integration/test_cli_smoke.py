import json
import subprocess
import sys
from pathlib import Path
import pytest
from tech_tree_arena.contract.recovery import occurrence_bits


ROOT = Path(__file__).resolve().parents[2]


def test_cli_runs_offline_smoke_match() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "tech_tree_arena.cli",
            "run",
            str(ROOT / "submissions" / "examples" / "minimal_pair"),
            "--target-pack",
            "smoke",
            "--seed",
            "4",
        ],
        check=True,
        text=True,
        capture_output=True,
    )
    result = json.loads(completed.stdout)
    assert result["status"] == "pass"
    assert result["K"] == pytest.approx(1.0 + occurrence_bits(1))


def test_generator_resources_carry_the_time_travel_flag(tmp_path) -> None:
    """Both participants read the arena's time-travel setting.

    The Generator shapes its priced menus around it (e.g. keyword abandon
    rows), so it must land in the Generator's public resources too -- it was
    once added only after the generator resource snapshot was taken, leaving
    the Generator permanently on its missing-key compatibility branch.
    """

    from tech_tree_arena.cli import _run_submission
    from tech_tree_arena.replay.artifacts import ArtifactStore

    result = _run_submission(
        ROOT / "submissions" / "examples" / "minimal_pair",
        "smoke",
        None,
        4,
        runs_dir=tmp_path,
        allow_time_travel=False,
    )
    manifest = json.loads(
        (Path(result["run_dir"]) / "manifest.json").read_text(encoding="utf-8")
    )
    store = ArtifactStore(tmp_path)
    generator_resources = store.load_json(manifest["generator_public_resources_ref"])
    assert generator_resources["time_travel_enabled"] is False
    oracle_resources = store.load_json(manifest["oracle_public_resources_ref"])
    assert oracle_resources["time_travel_enabled"] is False


def test_essence_defaults_to_three_repeat_judgments(tmp_path) -> None:
    from tech_tree_arena.cli import _run_submission

    result = _run_submission(
        ROOT / "submissions" / "examples" / "minimal_pair",
        "smoke",
        None,
        4,
        runs_dir=tmp_path,
        judge_mode="essence",
    )
    manifest = json.loads(
        (Path(result["run_dir"]) / "manifest.json").read_text(encoding="utf-8")
    )

    assert result["judge_repeats"] == 3
    assert result["judge_passes"] == 3
    assert result["repeat_bits"] == 0.0
    assert manifest["judge_config"]["repeats"] == 3
