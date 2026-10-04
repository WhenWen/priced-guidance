import json
import stat

import pytest

from tech_tree_arena.cli import main as cli_main
from tech_tree_arena.trajectory_html import (
    TrajectoryRenderError,
    load_trajectory_input,
    render_trajectory_html,
)


def _write_json(path, value) -> None:
    path.write_text(json.dumps(value), encoding="utf-8")


def _write_jsonl(path, rows) -> None:
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )


def _events(question_text="Which mechanism?"):
    return [
        {
            "sequence": 1,
            "kind": "question",
            "question_id": "q1.capability",
            "path_k": 0.0,
            "question": {
                "question": question_text,
                "options": [
                    {
                        "option_id": "answer-a",
                        "probability": "0.75",
                        "public_payload": {"label": "Mechanism A", "kind": "mc"},
                    },
                    {
                        "option_id": "answer-b",
                        "probability": "0.25",
                        "public_payload": {"label": "Mechanism B", "kind": "mc"},
                    },
                ],
            },
        },
        {
            "sequence": 2,
            "kind": "oracle_decision",
            "question_id": "q1.capability",
            "decision": {"option_id": "answer-a"},
        },
        {
            "sequence": 3,
            "kind": "choice_cost",
            "question_id": "q1.capability",
            "option_id": "answer-a",
            "probability": "0.75",
            "information_bits": 0.415,
            "branch_bits": 0.0,
            "path_k": 0.415,
        },
    ]


def test_render_run_directory_defaults_to_public_events_and_is_self_contained(tmp_path) -> None:
    run = tmp_path / "run-1"
    run.mkdir()
    _write_json(
        run / "manifest.json",
        {
            "run_id": "run-1",
            "submission_name": "pair",
            "target_id": "paper-1",
            "judge": "research-essence",
            "protocol": "idea-recovery-v1",
        },
    )
    _write_json(run / "score.json", {"status": "pass", "K": 0.415})
    _write_jsonl(run / "events.public.jsonl", _events())
    _write_jsonl(run / "events.private.jsonl", _events("PRIVATE SECRET"))
    output = tmp_path / "report.html"

    rendered = render_trajectory_html([run], output)
    document = rendered.read_text(encoding="utf-8")

    assert rendered == output.resolve()
    assert "Which mechanism?" in document
    assert "Mechanism A" in document
    assert "PRIVATE SECRET" not in document
    assert "https://" not in document
    assert "public event stream" not in document  # Filled by the embedded app at runtime.
    assert stat.S_IMODE(output.stat().st_mode) == 0o600


def test_private_render_is_explicit_and_script_terminators_are_neutralized(tmp_path) -> None:
    run = tmp_path / "run-private"
    run.mkdir()
    _write_jsonl(run / "events.public.jsonl", _events("public"))
    _write_jsonl(run / "events.private.jsonl", _events("secret </script><script>alert(1)</script>"))
    output = tmp_path / "private.html"

    render_trajectory_html([run], output, private=True)
    document = output.read_text(encoding="utf-8")

    assert "secret \\u003c/script>\\u003cscript>alert(1)\\u003c/script>" in document
    assert "secret </script>" not in document
    assert '"privacy":"private"' in document


def test_json_bundle_can_contain_multiple_named_trajectories(tmp_path) -> None:
    source = tmp_path / "trajectories.json"
    _write_json(
        source,
        {
            "trajectories": [
                {"id": "one", "label": "First run", "events": _events("First?")},
                {"id": "two", "label": "Second run", "events": _events("Second?")},
            ]
        },
    )

    trajectories = load_trajectory_input(source)

    assert [row["id"] for row in trajectories] == ["one", "two"]
    assert [row["label"] for row in trajectories] == ["First run", "Second run"]


def test_private_flag_cannot_be_downgraded_by_bundle_metadata(tmp_path) -> None:
    source = tmp_path / "trajectory.json"
    _write_json(source, {"privacy": "public", "events": _events()})

    [trajectory] = load_trajectory_input(source, private=True)

    assert trajectory["privacy"] == "private"


def test_invalid_jsonl_fails_with_a_line_number(tmp_path) -> None:
    source = tmp_path / "broken.jsonl"
    source.write_text('{"kind":"question"}\nnot-json\n', encoding="utf-8")

    with pytest.raises(TrajectoryRenderError, match="line 2"):
        load_trajectory_input(source)


def test_cli_render_command_writes_the_requested_report(tmp_path, capsys) -> None:
    source = tmp_path / "events.jsonl"
    output = tmp_path / "cli-report.html"
    _write_jsonl(source, _events())

    assert cli_main(["render", str(source), "--output", str(output)]) == 0

    response = json.loads(capsys.readouterr().out)
    assert response == {"output": str(output.resolve()), "private": False}
    assert output.is_file()


def test_cli_run_writes_public_and_private_reports_in_the_run_directory(
    tmp_path, capsys, monkeypatch
) -> None:
    run = tmp_path / "run-auto-report"
    run.mkdir()
    _write_jsonl(run / "events.public.jsonl", _events("Auto-rendered?"))
    _write_jsonl(run / "events.private.jsonl", _events("PRIVATE DETAILS"))

    def fake_run_submission(*_args, **_kwargs):
        return {
            "run_id": "run-auto-report",
            "submission": "pair",
            "target_id": "paper-1",
            "status": "pass",
            "run_dir": str(run),
        }

    monkeypatch.setattr(
        "tech_tree_arena.cli._run_submission", fake_run_submission
    )

    assert cli_main(["run", "unused-submission"]) == 0

    response = json.loads(capsys.readouterr().out)
    report = run / "trajectory-report.html"
    private_report = run / "trajectory-report.private.html"
    assert response["trajectory_report"] == str(report.resolve())
    assert response["trajectory_report_private"] == str(private_report.resolve())
    assert report.is_file()
    assert private_report.is_file()
    assert "Auto-rendered?" in report.read_text(encoding="utf-8")
    assert "PRIVATE DETAILS" not in report.read_text(encoding="utf-8")
    assert "PRIVATE DETAILS" in private_report.read_text(encoding="utf-8")
