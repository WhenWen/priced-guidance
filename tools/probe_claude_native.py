#!/usr/bin/env python3
"""Four synthetic turns: continue, rollback/fork, continue; no target data."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from tech_tree_arena.replay.artifacts import ArtifactStore
from tech_tree_arena.runtime.claude_generator import ClaudeGeneratorBackend, configuration


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--install-receipt")
    p.add_argument("--auth-home", required=True)
    p.add_argument("--model", default="claude-fable-5-1")
    p.add_argument("--effort", default="max")
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output / "started.json").exists():
        raise RuntimeError("Choose a fresh probe output directory; prior calls must stay recorded")
    config = configuration(args.install_receipt, args.auth_home, reasoning_effort=args.effort)
    (args.output / "started.json").write_text(json.dumps(config, indent=2) + "\n")
    backend = ClaudeGeneratorBackend(config, ArtifactStore(args.output))
    def call(context, prompt, index):
        try:
            result = backend.structured_in_context(context, model=args.model, reasoning_effort=args.effort,
                developer="Synthetic session-isolation probe. Follow the latest task exactly.", user=prompt,
                schema_name="probe", schema={"type":"object","properties":{"answer":{"type":"string"}},
                "required":["answer"],"additionalProperties":False}, max_output_tokens=50000)
            return result, backend.last_call_metadata()["native_context"]
        finally:
            (args.output / f"call-{index}.json").write_text(json.dumps(backend.last_call_metadata(), indent=2) + "\n")
    try:
        a, first = call(None, "The synthetic ledger is initially empty. Reply with answer empty.", 1)
        b, second = call(first, "Add marker BRANCH_ORANGE to the synthetic ledger. Reply with answer BRANCH_ORANGE.", 2)
        c, fork = call(first, "Read the current synthetic ledger. Reply with its marker, or empty if no marker was added.", 3)
        d, last = call(fork, "Set marker BRANCH_BLUE in the synthetic ledger. Reply with answer BRANCH_BLUE.", 4)
        report = {"outputs": [a,b,c,d], "correct": [a,b,c,d] == [{"answer":"empty"},{"answer":"BRANCH_ORANGE"},{"answer":"empty"},{"answer":"BRANCH_BLUE"}],
            "same_forward_session": first["session_id"] == second["session_id"],
            "separate_fork": first["session_id"] != fork["session_id"],
            "fork_continues": fork["session_id"] == last["session_id"], "usage":backend.usage_totals()}
        (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, indent=2))
        return 0 if all(report[k] for k in ("correct","same_forward_session","separate_fork","fork_continues")) else 1
    finally:
        backend.close()


if __name__ == "__main__":
    raise SystemExit(main())
