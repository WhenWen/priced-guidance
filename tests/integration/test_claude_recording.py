"""Claude native context through actor checkpoint, resume and offline replay."""
import copy
import json
from pathlib import Path
import shutil
import sys
import uuid

import pytest

from tech_tree_arena.cli import _run_submission, _resume_submission
from tech_tree_arena.errors import ParticipantFailure
from tech_tree_arena.replay.recorder import actor_replay, protocol_replay
from tech_tree_arena.runtime import claude_generator as cg
from tech_tree_arena.runtime.services import _request_hash

pytestmark = pytest.mark.skipif(sys.platform not in {"darwin", "linux"}, reason="requires OS sandbox")


class FakeNative:
    supports_context_chain = True
    parents = []
    fail_on = None

    def __init__(self, config, artifacts):
        self.artifacts = artifacts
        self.usage = {"calls":0,"input_tokens":0,"output_tokens":0,"cached_input_tokens":0,"cache_write_input_tokens":0,"cost_usd":0.0}
        self.metadata = {}

    def structured_in_context(self, context, **request):
        self.parents.append(copy.deepcopy(context));n=len(self.parents)
        self.metadata = {"backend":cg.BACKEND,"auth_mode":"claude-subscription","sandbox_profile":cg.PROFILE_VERSION,
            "prompt_version":cg.PROMPT_VERSION,"prompt_sha256":cg.PROMPT_SHA,"model":request['model'],
            "parent_context":copy.deepcopy(context),"request_hash":_request_hash('model.structured',request),"usage":{"cost_usd":0.0}}
        if n == self.fail_on:raise RuntimeError('synthetic interrupted call')
        result={"usage":{"input_tokens":11,"output_tokens":2},"session_id":str(uuid.uuid4())}
        transcript=self.artifacts.put_json({'transcript':json.dumps({'synthetic':n})+'\n'})
        self.metadata.update(native_context={'backend':cg.BACKEND,'session_id':result['session_id'],'cwd':'/arena','transcript':transcript},claude_result=result,usage=cg.token_usage(result))
        for k,v in self.metadata['usage'].items():self.usage[k]+=v
        return {'ok':True}

    def last_call_metadata(self):return copy.deepcopy(self.metadata)
    def usage_totals(self):return dict(self.usage)
    def restore_usage(self,usage):self.usage={k:usage.get(k,0) for k in self.usage}


@pytest.fixture
def pair(tmp_path,monkeypatch):
    root=tmp_path/'pair';shutil.copytree(Path(__file__).resolve().parents[2]/'submissions/examples/minimal_pair',root)
    source=root/'participant/generator.py'
    source.write_text(source.read_text()+'''
BaseGenerator = Generator
class Generator(BaseGenerator):
    def step(self, message):
        self.services.structured_model(model="claude-fable-5-1",developer="synthetic",user=str(message),schema_name="synthetic",schema={"type":"object"})
        return super().step(message)
''')
    FakeNative.parents=[];FakeNative.fail_on=None
    monkeypatch.setattr(cg,'ClaudeGeneratorBackend',FakeNative)
    return root


def test_claude_max_records_and_replays_without_new_model_calls(pair,tmp_path):
    result=_run_submission(pair,'smoke',None,1,runs_dir=tmp_path/'runs',generator_claude={'backend':cg.BACKEND,'reasoning_effort':'max'})
    calls=len(FakeNative.parents)
    assert calls>=2 and result['status']=='pass'
    assert protocol_replay(result['run_dir'])['status']=='replayed'
    assert actor_replay(result['run_dir'])['status']=='actor-replayed'
    assert len(FakeNative.parents)==calls


def test_claude_recovery_preserves_parent_and_usage(pair,tmp_path):
    FakeNative.fail_on=2
    with pytest.raises(ParticipantFailure):
        _run_submission(pair,'smoke',None,1,runs_dir=tmp_path/'runs',generator_claude={'backend':cg.BACKEND,'reasoning_effort':'max'})
    source=next((tmp_path/'runs').glob('*/manifest.json')).parent
    FakeNative.fail_on=None
    result=_resume_submission(source,retry_interrupted_call=True)
    assert result['status']=='pass'
    assert FakeNative.parents[0] is None and FakeNative.parents[1]==FakeNative.parents[2]
    assert actor_replay(result['run_dir'])['status']=='actor-replayed'
