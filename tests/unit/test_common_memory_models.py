import copy

import pytest

from tech_tree_arena.cli import build_parser, _generator_memory_config
from tech_tree_arena.errors import ArenaError, ReplayDivergence
from tech_tree_arena.replay.artifacts import ArtifactStore
from tech_tree_arena.runtime.common_memory import CommonMemoryBackend, configuration
from tech_tree_arena.runtime.memory_output_budget import configuration as output_configuration, validate_policy


def test_sol_routes_to_codex_and_rejects_api(monkeypatch, tmp_path):
    from tech_tree_arena.runtime import codex_generator
    calls = []
    def fake_configuration(receipt=None, auth=None, **kwargs):
        calls.append(kwargs)
        return {"build_receipt": "fixture", "auth_home": "fixture", **kwargs}
    monkeypatch.setattr(codex_generator, "configuration", fake_configuration)
    config = configuration("gpt-5.6-sol")
    assert config["reasoning_effort"] == "xhigh"
    assert config["codex"]["reasoning_effort"] == "xhigh"
    assert config["codex"]["prompt_version"]
    backend = CommonMemoryBackend(config, ArtifactStore(tmp_path), transport=object())
    assert backend.config["model"] == "gpt-5.6-sol"
    args = build_parser().parse_args(["run", "submissions/reference_pair_submit8",
        "--generator-memory", "common", "--generator-model", "gpt-5.6-sol"])
    with pytest.raises(ArenaError, match="codex"):
        _generator_memory_config(args)
    args.generator_backend = "codex"
    assert _generator_memory_config(args)["model"] == "gpt-5.6-sol"


@pytest.mark.parametrize("model", ["anthropic/claude-fable-5", "anthropic/claude-opus-5"])
def test_new_claude_cli_and_provider_route(model, tmp_path):
    args = build_parser().parse_args(["run", "submissions/reference_pair_submit8",
        "--generator-memory", "common", "--generator-model", model])
    config = _generator_memory_config(args)
    assert config["model"] == model and config["reasoning_effort"] == "max"
    assert config["codex"] is None
    backend = CommonMemoryBackend(config, ArtifactStore(tmp_path))
    assert backend.transport.provider == "anthropic"
    body = backend.transport._request([], {"max_output_tokens": 128000})
    assert body["model"] == model.removeprefix("anthropic/")
    assert body["max_tokens"] == 128000
    assert body["thinking"]["block_binding"]["prefix_mismatch_behavior"] == "error"
    assert body["output_config"]["effort"] == "max"


def test_exact_old_policy_receipts_remain_usable_but_tampering_fails(tmp_path):
    model = "anthropic/claude-fable-5-1"
    config = configuration(model)
    config["source_sha256"].update({
        "provider_client.py": "388e7a2de1c612c716f6e638eea0df19fccda47ec2134eee13b24444e39c0239",
        "common_memory.py": "80c12052488aaad9a8f4bafd7d3ae121c625c8ba1b3d6097c390d0a3e778fe6f",
        "native_api_generator.py": "unreviewed"})
    # Unknown source hashes must fail closed.
    with pytest.raises(ReplayDivergence):
        CommonMemoryBackend(config, ArtifactStore(tmp_path), transport=object())
    config["source_sha256"]["native_api_generator.py"] = "6348787b758495f6eb2ec729eeb26a335da934b8cfd8ea8b934695d80b8f837f"
    assert CommonMemoryBackend(config, ArtifactStore(tmp_path), transport=object()).config == config
    bad = copy.deepcopy(config)
    bad["source_sha256"]["provider_client.py"] = "unreviewed"
    with pytest.raises(ReplayDivergence):
        CommonMemoryBackend(bad, ArtifactStore(tmp_path), transport=object())
    old = output_configuration(model)
    old["source_sha256"] = "717f14bf37bcca9946c1bca04ea74b4fecf865abbf753906db0e500def031305"
    validate_policy(old)
    with pytest.raises(ReplayDivergence):
        validate_policy({**old, "model": "anthropic/claude-opus-5"})
