import importlib.util
import json
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    "eager_audit", Path(__file__).resolve().parents[2] / "tools/audit_codex_eager_run.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_recurring_decimal_distribution_is_valid():
    assert module.valid_preview_distribution([{"probability": "0." + "3" * 62}] * 3)


@pytest.mark.parametrize("values", [[], ["0", "1"], ["-0.1", "1.1"],
    ["NaN"], ["Infinity"], ["0.4", "0.5"], ["0.5", "0.500000002"], [True]])
def test_invalid_distributions_still_fail(values):
    assert not module.valid_preview_distribution([{"probability": p} for p in values])


@pytest.mark.parametrize("stage,count,valid", [
    ("directional", 8, True), ("directional", 1, False),
    ("essence", 1, True), ("essence", 8, False),
    ("strict", 1, True), ("strict", 0, False), (None, 8, False),
])
def test_submit8_frozen_stage_policy(stage, count, valid):
    assert module.valid_submit8_preview({"submission": {"ideas": [{}] * count}}, stage) is valid


@pytest.mark.parametrize("checkpoint", [False, True])
@pytest.mark.parametrize("delta", [0, 1])
def test_activation_cursors_ignore_old_calls_but_count_new_calls(tmp_path, checkpoint, delta):
    stream = {"calls": [
        {"message": {"type": "none"}, "service_event_count": 12},
        {"message": {"type": "choice", "value": {"option_id": "keyword",
            "public_payload": {"kind": "dispatch", "bundle_id": "bundle"}}},
         "service_event_count": 12 + delta},
    ]}
    if checkpoint:
        path = tmp_path / "checkpoint.private.json"
        value = {"actor_streams": {"generator": {"root": stream}}}
    else:
        path = tmp_path / "actor-branches.generator.private.json"
        value = [stream]
    path.write_text(json.dumps(value))
    assert module.activation_cursors(tmp_path)[("bundle", "keyword")] == [delta]
