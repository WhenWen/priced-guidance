import copy
import json
import time
import uuid

import pytest

from tech_tree_arena.errors import ReplayDivergence
from tech_tree_arena.replay.artifacts import ArtifactStore
from tech_tree_arena.runtime import claude_generator as cg
from tech_tree_arena.runtime.claude_sandbox import environment
from tech_tree_arena.runtime.linux_sandbox import connect_destination, PROVIDER_HOSTS


def test_claude_cache_accounting_includes_reads_and_writes_once():
    assert cg.token_usage({"usage":{"input_tokens":10,"output_tokens":3,"cache_read_input_tokens":90,"cache_creation_input_tokens":20}}) == {
        "calls":1,"input_tokens":120,"output_tokens":3,"cached_input_tokens":90,"cache_write_input_tokens":20,"cost_usd":0.0}


def test_claude_environment_does_not_inherit_keys_or_configs(tmp_path, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "synthetic-secret")
    monkeypatch.setenv("CLAUDE_CODE_OAUTH_TOKEN", "synthetic-secret")
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", "/private-account")
    env = environment(tmp_path)
    assert "ANTHROPIC_API_KEY" not in env and "CLAUDE_CODE_OAUTH_TOKEN" not in env
    assert env["CLAUDE_CONFIG_DIR"] == "/arena/claude"
    assert env["HOME"] == "/arena/home"


def test_provider_network_allowlists_do_not_mix(monkeypatch):
    monkeypatch.setattr("socket.getaddrinfo", lambda *a, **k: pytest.fail("must reject before DNS"))
    with pytest.raises(ValueError):
        connect_destination(b"CONNECT api.anthropic.com:443 HTTP/1.1\r\n\r\n")
    with pytest.raises(ValueError):
        connect_destination(b"CONNECT chatgpt.com:443 HTTP/1.1\r\n\r\n", PROVIDER_HOSTS["anthropic"])


@pytest.fixture
def backend(tmp_path, monkeypatch):
    account = tmp_path / "account"; account.mkdir()
    (account / ".credentials.json").write_text(json.dumps({"claudeAiOauth":{"accessToken":"synthetic", "subscriptionType":"max", "expiresAt":(time.time()+3600)*1000}}))
    config = {"install_receipt":"synthetic", "auth_home":str(account),"reasoning_effort":"max","version":"synthetic","executable_sha256":"synthetic"}
    monkeypatch.setattr(cg, "configuration", lambda *a, **k: config)
    monkeypatch.setattr(cg, "installation", lambda *a: (None,{"executable":"/synthetic"}))
    b = cg.ClaudeGeneratorBackend(config, ArtifactStore(tmp_path / "artifacts"))
    parents=[]
    def native(executable, workspace, args, prompt, timeout):
        assert args[args.index("--tools")+1] == ""
        assert "--safe-mode" in args and "--bare" not in args
        assert args[args.index("--effort")+1] == "max"
        settings = json.loads(args[args.index("--settings")+1])
        assert settings["switchModelsOnFlag"] is False and settings["fallbackModel"] == []
        assert settings["availableModels"] == ["claude-fable-5-1"]
        directory = workspace / "claude/projects/-arena"; directory.mkdir(parents=True,exist_ok=True)
        prior = ""
        if "--resume" in args:
            sid=args[args.index("--resume")+1];prior=(directory/f"{sid}.jsonl").read_text()
            if "--fork-session" in args:sid=str(uuid.uuid4())
        else:sid=args[args.index("--session-id")+1]
        parents.append(prior)
        response=json.dumps({"payload_json":json.dumps({"ok":True})})
        row={"type":"assistant","message":{"content":[{"type":"thinking","thinking":"synthetic","signature":str(len(parents))},{"type":"text","text":response}]}}
        (directory/f"{sid}.jsonl").write_text(prior+json.dumps(row)+"\n")
        return 0,[{"type":"result","subtype":"success","is_error":False,"session_id":sid,
                   "usage":{"input_tokens":11,"output_tokens":2,"cache_read_input_tokens":10},
                   "modelUsage":{"claude-fable-5-1":{}},"result":response}]
    monkeypatch.setattr(cg,"run_cli",native)
    yield b,parents
    b.close()


def request(b,context=None):
    return b.structured_in_context(context,model="claude-fable-5-1",reasoning_effort="max",developer="synthetic",
            user="test",schema_name="test",schema={"type":"object","properties":{"ok":{"type":"boolean"}},"required":["ok"]})


def test_rollback_restores_exact_native_ancestor_and_all_costs(backend):
    b,parents=backend
    request(b);first=copy.deepcopy(b.last_call_metadata()["native_context"])
    original=b.artifacts.load_json(first["transcript"])
    request(b,first);second=copy.deepcopy(b.last_call_metadata()["native_context"])
    assert first["session_id"]==second["session_id"]
    request(b,first);fork=b.last_call_metadata()["native_context"]
    assert fork["session_id"]!=first["session_id"]
    assert parents[1]==parents[2]==original["transcript"]
    assert b.artifacts.load_json(first["transcript"])==original
    assert b.usage_totals()["calls"]==3
    b.close()
    request(b,second)
    assert len(parents[-1].splitlines())==2


def test_failed_payload_keeps_usage_and_does_not_advance_context(backend,monkeypatch):
    b,_=backend
    original=cg.run_cli
    def broken(*args):
        code,events=original(*args);events[-1]['result']='invalid JSON';return code,events
    monkeypatch.setattr(cg,'run_cli',broken)
    with pytest.raises((ValueError,RuntimeError)):request(b)
    assert b.last_call_metadata()['usage']['calls']==1
    assert b._active_context is None


def test_replay_rejects_tampered_usage(backend):
    b,_=backend;request(b)
    m=b.last_call_metadata();req={"model":"claude-fable-5-1","reasoning_effort":"max","developer":"synthetic","user":"test","schema_name":"test","schema":{"type":"object","properties":{"ok":{"type":"boolean"}},"required":["ok"]}}
    cg.validate_metadata({"metadata":m,"request":req})
    m['usage']['cached_input_tokens']+=1
    with pytest.raises(ReplayDivergence):cg.validate_metadata({"metadata":m,"request":req})


def test_unexpected_native_model_is_rejected_and_usage_retained(backend, monkeypatch):
    b, _ = backend
    original = cg.run_cli
    def fallback(*args):
        code, events = original(*args)
        events[-1]["modelUsage"] = {"claude-opus-5": {}}
        return code, [{"type": "system", "subtype": "init", "model": "claude-opus-5"}, *events]
    monkeypatch.setattr(cg, "run_cli", fallback)
    with pytest.raises(RuntimeError, match="model differed"):
        request(b)
    assert b.last_call_metadata()["usage"]["calls"] == 1
    assert b.last_call_metadata()["native_model_diagnostics"]["initial_models"] == ["claude-opus-5"]
    assert b._active_context is None
