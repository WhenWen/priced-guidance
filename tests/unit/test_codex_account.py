import json
from pathlib import Path
import threading

import pytest

from tech_tree_arena.runtime import codex_generator


def test_generator_requires_explicit_independent_login(tmp_path, monkeypatch):
    personal = tmp_path / ".codex"
    personal.mkdir()
    original = json.dumps({"tokens": {"access_token": "synthetic-personal-token"}})
    (personal / "auth.json").write_text(original)
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    account = tmp_path / "experiment-account"
    with pytest.raises(RuntimeError, match="independent Codex account login"):
        codex_generator.require_account(account)
    assert not account.exists()
    assert (personal / "auth.json").read_text() == original


def test_explicit_account_import_is_independent_and_rejects_api_key(tmp_path):
    source = tmp_path / "source.json"
    source.write_text(json.dumps({"tokens": {"access_token": "synthetic"}}))
    account = tmp_path / "account"
    codex_generator.import_account(account, source)
    codex_generator.require_account(account)
    assert (account / "auth.json").stat().st_mode & 0o777 == 0o600
    (account / "auth.json").write_text(json.dumps({"tokens": {"access_token": "refreshed"}}))
    assert json.loads(source.read_text())["tokens"]["access_token"] == "synthetic"
    source.write_text(json.dumps({"OPENAI_API_KEY": "synthetic-api-key"}))
    with pytest.raises(RuntimeError, match="API-key auth is refused"):
        codex_generator.import_account(tmp_path / "api-account", source)


def _build_receipt(tmp_path, monkeypatch):
    binary = tmp_path / "codex"
    binary.write_text("synthetic source-built executable")
    lock = tmp_path / "source.lock.json"
    lock.write_text(json.dumps({"commit": "f" * 40, "patches": []}))
    receipt = tmp_path / "receipt.json"
    receipt.write_text(json.dumps({"source": json.loads(lock.read_text()),
        "source_lock_sha256": codex_generator.file_sha(lock), "patch_sha256": {},
        "executable": str(binary), "executable_sha256": codex_generator.file_sha(binary)}))
    monkeypatch.setattr(codex_generator, "SOURCE_LOCK", lock)
    monkeypatch.setattr(codex_generator, "CACHE_SOURCE_LOCK", lock)
    return binary, receipt


def test_build_receipt_detects_binary_replacement(tmp_path, monkeypatch):
    binary, receipt = _build_receipt(tmp_path, monkeypatch)
    assert codex_generator.configuration(receipt)["executable_sha256"] == codex_generator.file_sha(binary)
    binary.write_text("replaced installed binary")
    with pytest.raises(RuntimeError, match="executable differs"):
        codex_generator.configuration(receipt)


def test_prompt_binding_rejects_changes_and_preserves_legacy_resume(tmp_path, monkeypatch):
    from tech_tree_arena.replay.artifacts import ArtifactStore
    from tech_tree_arena.runtime.codex_prompts import DEFAULT_VERSION, prompt_sha
    _, receipt = _build_receipt(tmp_path, monkeypatch)
    account = tmp_path / "account"
    account.mkdir()
    (account / "auth.json").write_text(json.dumps({"tokens": {"access_token": "synthetic"}}))
    monkeypatch.setattr(codex_generator, "profile", lambda *args: "")
    artifacts = ArtifactStore(tmp_path / "artifacts")
    config = codex_generator.configuration(receipt, account)
    assert config["prompt_version"] == DEFAULT_VERSION
    assert config["prompt_sha256"] == prompt_sha(DEFAULT_VERSION)
    assert codex_generator.CodexGeneratorBackend(config, artifacts).config == config
    with pytest.raises(RuntimeError, match="configuration changed"):
        codex_generator.CodexGeneratorBackend({**config, "prompt_sha256": "modified"}, artifacts)
    legacy = {key: value for key, value in config.items() if key not in {"prompt_version", "prompt_sha256"}}
    legacy["context_mode"] = "native-fork-every-call"
    assert codex_generator.CodexGeneratorBackend(legacy, artifacts).config == legacy


def _backend_without_transport(tmp_path):
    from tech_tree_arena.replay.artifacts import ArtifactStore
    backend = object.__new__(codex_generator.CodexGeneratorBackend)
    backend.config = {"source_commit": "synthetic", "executable_sha256": "synthetic"}
    backend.executable = tmp_path / "unused-codex"
    backend.account = tmp_path / "account"
    backend.account.mkdir()
    (backend.account / "auth.json").write_text(json.dumps({"tokens": {"access_token": "synthetic"}}))
    backend.artifacts = ArtifactStore(tmp_path / "artifacts")
    backend._thread = threading.local()
    backend._lock = threading.RLock()
    backend._usage = {"calls": 0, "cost_usd": 0.0}
    return backend


def test_preflight_failure_cannot_reuse_previous_call_metadata(tmp_path):
    backend = _backend_without_transport(tmp_path)
    backend._thread.metadata = {"codex_context": {"thread_id": "previous"}, "usage": {"calls": 1}}
    with pytest.raises(ValueError, match="API conversation state"):
        backend.structured(model="gpt-6-astra", schema_name="synthetic", return_conversation_state=True)
    metadata = backend.last_call_metadata()
    assert "codex_context" not in metadata
    assert metadata["usage"] == {"cost_usd": 0.0}
    assert backend.usage_totals()["calls"] == 0


def test_transport_failure_without_stderr_preserves_original_error(tmp_path, monkeypatch):
    backend = _backend_without_transport(tmp_path)

    def fail(*args):
        raise RuntimeError("synthetic transport initialization failure")

    monkeypatch.setattr(codex_generator, "run_turn", fail)
    with pytest.raises(RuntimeError, match="synthetic transport initialization failure"):
        backend.structured(model="gpt-6-astra", schema_name="synthetic")
    metadata = backend.last_call_metadata()
    assert metadata["stderr"] == ""
    assert metadata["usage"]["unknown_account_turns"] == 1
    assert "codex_context" not in metadata


def test_cache_identity_follows_checkpoint_and_distinguishes_new_roots(tmp_path, monkeypatch):
    from tech_tree_arena.runtime.codex_prompts import DEFAULT_VERSION
    backend = _backend_without_transport(tmp_path)
    backend.config["prompt_version"] = DEFAULT_VERSION
    backend.trace_requests = False
    seen = []
    class RPC:
        def __init__(self, *args, **kwargs):
            pass
        def close(self):
            pass
    monkeypatch.setattr(codex_generator, "CodexRPC", RPC)
    def turn(executable, workspace, request, parent, version, **options):
        seen.append((parent, options["cache_key"]))
        path = workspace / "rollout.jsonl"
        path.write_text(json.dumps({"type": "event_msg", "payload": {"type": "task_complete", "turn_id": "turn"}}))
        return {"output": {"ok": True}, "usage": {"calls": 1, "output_tokens": 1, "cost_usd": 0.0},
                "events": [], "thread_id": str(len(seen)), "turn_id": "turn",
                "rollout": str(path), "usage_total": {}}
    monkeypatch.setattr(codex_generator, "run_turn", turn)
    request = {"model": "gpt-6-astra", "schema_name": "synthetic"}
    backend.structured(**request)
    root = backend.last_call_metadata()["codex_context"]
    backend.structured_in_context(root, **request)
    backend.structured_in_context(root, **request)
    backend.structured(**request)
    assert seen[0][1] == seen[1][1] == seen[2][1] == root["cache_lineage"]
    assert seen[3][1] != root["cache_lineage"]
    assert seen[1][0]["thread_id"] == seen[2][0]["thread_id"] == "1"
    bad = {key: value for key, value in root.items() if key != "cache_lineage"}
    with pytest.raises(ValueError, match="cache lineage"):
        backend.structured_in_context(bad, **request)
    assert len(seen) == 4
    backend.close()
