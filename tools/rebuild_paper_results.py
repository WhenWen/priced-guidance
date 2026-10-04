"""Rebuild current test87 costs from frozen arena costs and promotion charges."""
from __future__ import annotations
import csv
import json
import math
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "paper/results/test87"

def rebuild():
    rows = list(csv.DictReader((DATA / "arena_costs.csv").open()))
    promotions = {(row["series"], row["target_id"]): float(row["bits"])
                  for row in csv.DictReader((DATA / "promotion_costs.csv").open())}
    values = {}
    for row in rows:
        key = row["criterion"], row["model"], row["paper"]
        if key in values:
            raise ValueError(f"duplicate measurement: {key}")
        passed = row["validated_or_positive_ensemble_score"].lower() == "true"
        cost = float(row["K_bits"]) if passed else math.inf
        if key[0] == "essence" and key[1] != "ensemble" and math.isfinite(cost):
            cost += promotions[key[1], key[2]]
        values[key] = cost
    ids = sorted({key[2] for key in values})
    if len(ids) != 87:
        raise ValueError("expected all 87 test targets")
    for target in ids:
        costs = [values["essence", model, target] for model in ("fable", "opus", "astra")]
        lowest = min(costs)
        values["essence", "ensemble", target] = (lowest - math.log2(
            math.fsum(2 ** (lowest - cost) for cost in costs) / 3)
            if math.isfinite(lowest) else math.inf)
    summary = {"accounting": ["occurrence-prior-v2", "mixture-judge-v1", "promotion-cost-v1"],
               "denominator": 87, "ensemble_members": ["fable", "opus", "astra"], "series": {}}
    with (DATA / "compression.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["criterion", "model", "paper", "compression_bits", "passed"])
        for (stage, model, target), cost in sorted(values.items()):
            writer.writerow([stage, model, target, cost, math.isfinite(cost)])
    for stage in ("directional", "essence"):
        summary["series"][stage] = {}
        for model in sorted({key[1] for key in values}):
            costs = sorted(values[stage, model, target] for target in ids)
            finite = lambda value: value if math.isfinite(value) else None
            summary["series"][stage][model] = {"passes": sum(math.isfinite(v) for v in costs),
                "median_cost_bits": finite(costs[43]), "p80_cost_bits": finite(costs[69])}
    (DATA / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    return summary

if __name__ == "__main__":
    print(json.dumps(rebuild(), indent=2))
