import json
import pytest
from tech_tree_arena.cli import _resume_submission
from tech_tree_arena.replay.recorder import HashChainWriter
from tech_tree_arena.runtime.refusal_recovery import provider_refusal, ProviderRefusalBlocked


@pytest.mark.parametrize('category', ['cyber', 'reasoning_extraction', 'refusal'])
def test_explicit_refusal_blocks_source_and_descendant_before_new_run(tmp_path, category):
    source, child = tmp_path / 'source', tmp_path / 'child'
    source.mkdir(); child.mkdir()
    (source / 'manifest.json').write_text('{}')
    (child / 'manifest.json').write_text(json.dumps({'resumed_from': 'source'}))
    HashChainWriter(source / 'events.private.jsonl').append({'kind': 'run_finished',
        'error_type': 'NativeCallError', 'error_code': 'participant_failure',
        'error_message': f'Provider refused request ({category})'})
    for root in (source, child):
        assert provider_refusal(root)['run_dir'] == str(source)
        with pytest.raises(ProviderRefusalBlocked):
            _resume_submission(root, retry_interrupted_call=True)
    assert sorted(p.name for p in tmp_path.iterdir()) == ['child', 'source']


def test_transient_failure_remains_retryable(tmp_path):
    (tmp_path / 'manifest.json').write_text('{}')
    HashChainWriter(tmp_path / 'events.private.jsonl').append({'kind': 'run_finished',
        'error_type': 'NativeCallError', 'error_message': 'Native anthropic thinking failed (TimeoutError, status=None)'})
    assert provider_refusal(tmp_path) is None
