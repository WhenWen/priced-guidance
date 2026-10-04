#!/usr/bin/env python3
"""Paid, target-free native-history/compaction smoke; no private paper input."""
import argparse
import json
from pathlib import Path
import secrets
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tech_tree_arena.cli import _load_env_file
from tech_tree_arena.replay.artifacts import ArtifactStore
from tech_tree_arena.runtime.common_memory import CommonMemoryBackend, configuration
from tech_tree_arena.runtime.services import ServiceFactory
from tech_tree_arena.runtime.memory_output_budget import MODELS, OutputBudgetBackend, configuration as output_configuration, validate_metadata


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--env-file", type=Path)
    p.add_argument("--effort")
    p.add_argument("--build-receipt", type=Path)
    p.add_argument("--auth-home", type=Path)
    p.add_argument("--submit8", action="store_true", help="exercise the exact submit8 participant; route Questions use synthetic fixtures")
    p.add_argument("--submit8-only", action="store_true", help="actual 8-idea call and cached activation; no memory probe")
    args = p.parse_args()
    args.submit8 = args.submit8 or args.submit8_only
    if args.env_file:
        _load_env_file(args.env_file)
    codex = None
    if args.model in {"gpt-6-astra", "gpt-5.6-sol"}:
        from tech_tree_arena.runtime.codex_generator import configuration as cc
        codex = cc(args.build_receipt, args.auth_home, reasoning_effort=args.effort or "xhigh")
    config = configuration(args.model, effort=args.effort, compact_calls=1, codex=codex)
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    store = ArtifactStore(args.output)
    backend = CommonMemoryBackend(config, store)
    if args.model in MODELS:
        output_policy = output_configuration(args.model)
        (args.output / "output-policy.json").write_text(json.dumps(output_policy, indent=2) + "\n")
        backend = OutputBudgetBackend(backend, output_policy)
    resources = {"generator_memory_policy": {"deterministic_calls": True, "timeout": 1500}}
    services = ServiceFactory(seed=1, model_backend=backend, model_name=args.model,
                              public_resources=resources).create()
    anchor = secrets.token_hex(12)
    request = {"developer": "Perform the specified target-free memory test. Preserve the exact anchor for later turns.",
               "schema_name": "common_memory_smoke", "max_output_tokens": 50000,
               "schema": {"type": "object", "properties": {"answer": {"type": "string"}}, "required": ["answer"], "additionalProperties": False}}
    report = {"model": args.model, "status": "running", "target_data_used": False}
    try:
        if args.submit8:
            import copy
            from tech_tree_arena import Option, Question
            from tech_tree_arena.submission_io.manifest import load_manifest, load_participant_classes
            from tech_tree_arena.contract.validation import validate_question
            manifest = load_manifest(Path(__file__).resolve().parents[1] / "submissions/reference_pair_submit8")
            Generator, _ = load_participant_classes(manifest)
            generator = Generator(services)
            generator.state["stage"] = "directional"
            generator.state["current_draft"] = (
                "Synthetic hypothesis: learn a compact representation of a sensor time series "
                "using prediction of missing intervals and uncertainty estimates. "
                + ("" if args.submit8_only else f"Keep the synthetic memory marker ANCHOR={anchor} for the next call."))
            helpers = Generator.__init__.__globals__
            def previews(modes):
                result = {}
                for mode in modes:
                    question = Question(f"Synthetic {mode}", (Option(mode + "-retry", {"kind": "retry"}, "1"),))
                    result[mode] = {"question": question, "question_hash": helpers["_stable_hash"](helpers["_serialized_question"](question))}
                return result
            generator._build_directional_previews = previews
            facts = copy.deepcopy(generator.state["facts"])
            question = generator._directional_dispatch_question()
            validate_question(question)
            submit = next(o for o in question.options if o.option_id == "mode-submit")
            expected = submit.public_payload["preview"]["submission"]
            calls_before = backend.usage_totals()["physical_calls"]
            restored = Generator(services)
            restored.state = copy.deepcopy(generator.state)
            restored._validate_directional_submit(submit.public_payload)
            actual = helpers["_submission_snapshot"](restored._shared_submission())
            assert len(actual["ideas"]) == 8 and actual == expected
            assert backend.usage_totals()["physical_calls"] == calls_before
            assert generator.state["facts"] == facts
            report.update(submit8_candidates=8, cached_submission_unchanged=True,
                          activation_paid_calls=0, route_questions="synthetic fixtures")
            store.put_json({"question": helpers["_serialized_question"](question)})
        else:
            first = services.structured_model(**request, user=f"Remember ANCHOR={anchor}. Compute 37*43. Return the product as answer.")
            assert first["answer"] == "1591"
        print(json.dumps({"event": "first_completed", "model": args.model, "submit8": args.submit8}), flush=True)
        if args.submit8_only:
            for event in services.export_tape():
                validate_metadata({"request": event.request, "response": event.response, "error": event.error,
                                   "metadata": event.metadata}, store)
            report.update(status="passed", memory_probe=False)
            return
        second = services.structured_model(**request, user="Return the exact ANCHOR from before as answer.")
        tape = services.export_tape()
        (args.output / "tape.json").write_text(json.dumps([
            {"request": e.request, "response": e.response, "error": e.error, "metadata": e.metadata}
            for e in tape], indent=2) + "\n")
        for event in tape:
            validate_metadata({"request": event.request, "response": event.response, "error": event.error, "metadata": event.metadata}, store)
        meta = backend.last_call_metadata()
        summary_meta = meta["operations"][0]["metadata"]
        inherited = summary_meta.get("reasoning_blocks_inherited")
        if args.model in {"gpt-6-astra", "gpt-5.6-sol"}:
            from tools.probe_codex_generator import encrypted_fingerprints
            parent = meta["parent_context"]["native"]
            child = summary_meta["codex_context"]
            before = encrypted_fingerprints(store.load_json(parent["rollout"])["rollout"])
            after = encrypted_fingerprints(store.load_json(child["rollout"])["rollout"])
            inherited = len(before & after)
            assert before and before <= after
        report.update(status="passed", anchor_preserved=second["answer"] in {anchor, "ANCHOR=" + anchor},
                      summary_inherited_reasoning_blocks=inherited,
                      new_window=meta["native_context"]["window"] == 1,
                      operations=[x["role"] for x in meta["operations"]])
        assert report["anchor_preserved"] and inherited and report["new_window"]
    except BaseException as exc:
        report.update(status="failed" if isinstance(exc, Exception) else "interrupted",
                      error_type=type(exc).__name__, error=str(exc)[:500])
        raise
    finally:
        report["usage"] = backend.usage_totals()
        (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
        store.put_json({"last_call_metadata": backend.last_call_metadata()})
        backend.close()
        print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
