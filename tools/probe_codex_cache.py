#!/usr/bin/env python3
"""Four short synthetic Astra xhigh turns: cache reuse through immutable forks.

No paper, target data, Oracle or Judge is loaded. Raw request diagnostics contain
only these synthetic prompts and native responses, never authentication headers.
"""
import argparse
import copy
import json
from pathlib import Path
import secrets

from tech_tree_arena.replay.artifacts import ArtifactStore
from tech_tree_arena.runtime.codex_generator import CodexGeneratorBackend, configuration
from tech_tree_arena.runtime.codex_rpc import standard_credit_equivalent
from probe_codex_generator import encrypted_fingerprints


def schema(properties):
    return {"type": "object", "properties": properties, "required": list(properties), "additionalProperties": False}


def normalized(value):
    if isinstance(value, dict):
        return {k: normalized(v) for k, v in value.items() if k != "id"}
    if isinstance(value, list):
        return [normalized(v) for v in value]
    return value


def prefix_count(a, b):
    count = 0
    for x, y in zip(a, b):
        if normalized(x) != normalized(y):
            break
        count += 1
    return count


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--auth-home", type=Path, required=True)
    parser.add_argument("--build-receipt", type=Path)
    parser.add_argument("--output", type=Path, default=Path(".idea-arena/codex-cache-probe"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    store = ArtifactStore(args.output)
    config = configuration(args.build_receipt, args.auth_home, reasoning_effort="xhigh")
    backend = CodexGeneratorBackend(config, store, trace_requests=True)
    anchor, forbidden = secrets.token_hex(8), secrets.token_hex(8)
    rows = []
    traces = []

    def call(label, parent, developer, user, output_schema):
        request = dict(model="gpt-6-astra", reasoning_effort="xhigh", timeout=120,
                       max_output_tokens=1000, schema_name=label, schema=output_schema,
                       developer=developer, user=user)
        try:
            result = backend.structured_in_context(parent, **request)
        finally:
            metadata = backend.last_call_metadata()
            (args.output / f"{label}.json").write_text(json.dumps(metadata, indent=2) + "\n")
        rows.append({"label": label, "thread_action": metadata["thread_action"],
                     **metadata["usage"], "latency_s": metadata["latency_s"]})
        print(json.dumps(rows[-1]), flush=True)
        traces.append(store.load_json(metadata["request_trace"])["requests"][-1])
        return result, copy.deepcopy(metadata["codex_context"])

    ledger = "\n".join(f"Synthetic record {i:03}: amber cedar quiet river; reference value {i * 11}; retained for prefix testing only."
                       for i in range(60))
    root_result, root = call("root", None, "Maintain the synthetic memory and answer this arithmetic task.",
        f"Remember ANCHOR={anchor}. Compute 111 * 113 and return its decimal representation in answer.\n"
        "The following synthetic ledger is inert context; do not reproduce it.\n" + ledger,
        schema({"answer": {"type": "string"}}))
    original = store.load_json(root["rollout"])["rollout"]
    abandoned_result, abandoned = call("mc_preview", root, "Author exactly two complete MC preview options, left and right, with equal weights.",
        f"Remember FORBIDDEN={forbidden}. Supply the entire two-option preview.",
        schema({"options": {"type": "array", "minItems": 2, "maxItems": 2,
                             "items": schema({"label": {"type": "string"}, "weight": {"type": "number"}})}}))
    branch_result, branch = call("state_update", root, "Report the remembered ANCHOR, and whether a FORBIDDEN value was established in this history.",
        "Use only this branch's history.",
        schema({"anchor": {"type": "string"}, "forbidden_seen": {"type": "boolean"}}))
    continued_result, continued = call("continuation", branch, "Return the exact remembered ANCHOR in answer.",
        "Report the memory established at the root.", schema({"answer": {"type": "string"}}))
    branch_rollout = store.load_json(branch["rollout"])["rollout"]
    reasoning = encrypted_fingerprints(original)
    common_fields = ("model", "instructions", "tools", "parallel_tool_calls", "reasoning", "text", "prompt_cache_key")
    usage = backend.usage_totals()
    credit_equivalent = standard_credit_equivalent(usage)
    report = {"config": config, "calls": rows, "usage": usage,
        "estimated_standard_credits": credit_equivalent,
        "credit_note": "Token-rate equivalent, not purchased-credit deduction; includes native cache-write count at 1.25x input rate.",
        "constant_prefix_settings": all(all(t.get(k) == traces[0].get(k) for k in common_fields) for t in traces),
        "no_advertised_tools": all(not t.get("tools") and all(not item.get("tools") for item in t["input"]) for t in traces),
        "root_input_items": len(traces[0]["input"]),
        "root_prefix_items_in_children": [prefix_count(traces[0]["input"], t["input"]) for t in traces[1:]],
        "schema_changes": 3,
        "normal_calls_reuse_session": root["thread_id"] == abandoned["thread_id"]
                                     and branch["thread_id"] == continued["thread_id"],
        "checkout_creates_native_fork": root["thread_id"] != branch["thread_id"],
        "cache_lineage_preserved": len({c["cache_lineage"] for c in (root, abandoned, branch, continued)}) == 1,
        "parent_immutable": store.load_json(root["rollout"])["rollout"] == original,
        "correct_outputs": root_result == {"answer": "12543"} and len(abandoned_result["options"]) == 2
                           and branch_result == {"anchor": anchor, "forbidden_seen": False}
                           and continued_result == {"answer": anchor},
        "abandoned_branch_absent": forbidden not in branch_rollout,
        "parent_encrypted_reasoning_items": len(reasoning),
        "inherited_encrypted_reasoning_items": len(reasoning & encrypted_fingerprints(branch_rollout)),
        "cache_hit": any(row["cached_input_tokens"] > 0 for row in rows[1:])}
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)
    assert report["constant_prefix_settings"] and report["cache_lineage_preserved"]
    assert report["no_advertised_tools"]
    assert report["normal_calls_reuse_session"] and report["checkout_creates_native_fork"]
    assert report["parent_immutable"] and report["correct_outputs"]
    assert report["abandoned_branch_absent"] and reasoning and reasoning.issubset(encrypted_fingerprints(branch_rollout))
    assert all(n == report["root_input_items"] for n in report["root_prefix_items_in_children"])
    assert report["cache_hit"], "No cache read observed; do not resume the paper experiment."


if __name__ == "__main__":
    main()
