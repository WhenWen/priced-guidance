#!/usr/bin/env python3
"""A small live account test of native thinking inheritance and branch isolation.

Uses synthetic nonces only; never loads an evaluation target. This consumes
four Codex account turns when successful. Run with `uv run python ...`.
"""
import argparse
import hashlib
import json
from pathlib import Path
import secrets

from tech_tree_arena.replay.artifacts import ArtifactStore
from tech_tree_arena.runtime.codex_generator import CodexGeneratorBackend, configuration
from tech_tree_arena.runtime.services import ServiceFactory


def encrypted_fingerprints(rollout):
    found = set()
    def visit(value):
        if isinstance(value, dict):
            encrypted = value.get("encrypted_content")
            if value.get("type") == "reasoning" and encrypted:
                found.add(hashlib.sha256(encrypted.encode()).hexdigest())
            for child in value.values():
                visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)
    for line in rollout.splitlines():
        visit(json.loads(line))
    return found


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="gpt-6-astra")
    parser.add_argument("--auth-home", type=Path)
    parser.add_argument("--output", type=Path, default=Path(".idea-arena/codex-probe"))
    args = parser.parse_args()
    store = ArtifactStore(args.output)
    config = configuration(auth_home=args.auth_home)
    backend = CodexGeneratorBackend(config, store)
    factory = ServiceFactory(seed=17, model_backend=backend, model_name=args.model)
    services = factory.create()
    anchor, abandoned = secrets.token_hex(12), secrets.token_hex(12)
    request = {"developer": "Maintain the private memory established in this conversation. Answer only the requested JSON. Do not use tools.",
               "schema_name": "native_history_probe", "reasoning_effort": "high", "timeout": 180,
               "max_output_tokens": 5000,
               "schema": {"type": "object", "properties": {"answer": {"type": "string"}, "forbidden_seen": {"type": "boolean"}},
                          "required": ["answer", "forbidden_seen"], "additionalProperties": False}}
    first_prompt = (f"Remember ANCHOR={anchor}. Compute the exact coefficient of x^18 in "
                    "(1+x+x^2+x^3+x^4)^9. Return that integer as a decimal string in answer "
                    "and forbidden_seen=false. Solve it carefully without tools.")
    services.structured_model(**request, user=first_prompt)
    tape = services.export_tape()
    parent = services.export_state()["model_context"]
    parent_rollout = store.load_json(parent["rollout"])["rollout"]
    services.structured_model(**request, user=f"Remember FORBIDDEN={abandoned}. Return answer=ready and forbidden_seen=true.")
    calls_before = backend.usage_totals()["calls"]
    fork = factory.create(tape)
    fork.structured_model(**request, user=first_prompt)
    fork.finish_replay("alternative")
    replay_calls = backend.usage_totals()["calls"] - calls_before
    output = fork.structured_model(**request, user="What is ANCHOR? Put its exact value in answer. Set forbidden_seen to whether a FORBIDDEN value was established in this conversation.")
    child = fork.export_state()["model_context"]
    child_rollout = store.load_json(child["rollout"])["rollout"]
    parent_reasoning = encrypted_fingerprints(parent_rollout)
    child_reasoning = encrypted_fingerprints(child_rollout)
    continued = fork.structured_model(**request, user="Again report the exact ANCHOR value in answer, and whether FORBIDDEN has been established.")
    continued_context = fork.export_state()["model_context"]
    continued_rollout = store.load_json(continued_context["rollout"])["rollout"]
    report = {"model": args.model, "source_commit": config["source_commit"], "auth": "chatgpt",
              "native_fork": parent["thread_id"] != child["thread_id"],
              "remembers_anchor": output["answer"] == anchor, "abandoned_branch_absent": not output["forbidden_seen"] and abandoned not in child_rollout,
              "parent_immutable": store.load_json(parent["rollout"])["rollout"] == parent_rollout,
              "replay_model_calls": replay_calls, "parent_encrypted_reasoning_items": len(parent_reasoning),
              "inherited_encrypted_reasoning_items": len(parent_reasoning & child_reasoning),
              "continued_session": continued_context["thread_id"] == child["thread_id"] and continued["answer"] == anchor
                                and not continued["forbidden_seen"] and abandoned not in continued_rollout,
              "usage": backend.usage_totals()}
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)
    assert report["native_fork"] and report["remembers_anchor"] and report["abandoned_branch_absent"] and report["parent_immutable"] and report["continued_session"]
    assert replay_calls == 0
    assert parent_reasoning and parent_reasoning.issubset(child_reasoning), "Native encrypted reasoning was not retained"


if __name__ == "__main__":
    main()
