"""Budget admission tests use synthetic billing only; never call a provider."""
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import json
import time
from types import SimpleNamespace

import pytest

from tech_tree_arena.runtime.run_budget import (
    AttemptGuard, BudgetBlocked, Ledger, import_existing, micros, policy,
    request_reservation, setup,
)


def ledger_at(tmp_path, baseline=None):
    ledger = Ledger(tmp_path / 'budget.json', create=True)
    ledger.seed('paper', baseline or {}, origin='runs')
    return ledger


def test_cumulative_roles_shared_total_and_no_resume_reset(tmp_path):
    ledger = ledger_at(tmp_path, {'generator': 40, 'oracle': 14, 'judge': 6})
    ledger.seed('paper', {}, origin='runs')
    ledger.reserve('essence', 'paper', 'judge', micros(4))
    ledger.settle('essence', cost=4, known=True)
    restored = Ledger(ledger.path)
    assert sum(restored.totals(restored.snapshot()).values()) == micros(64)
    with pytest.raises(BudgetBlocked):
        restored.reserve('judge-over', 'paper', 'judge', 1)
    with pytest.raises(BudgetBlocked):
        restored.reserve('generator-over', 'paper', 'generator', micros(11))
    restored.reserve('generator-exact', 'paper', 'generator', micros(10))
    assert sum(restored.totals(restored.snapshot()).values()) == micros(74)


def _process_reserve(args):
    path, paper = args
    ledger = Ledger(path)
    try:
        ledger.reserve(paper, paper, 'generator', micros(30), timeout=0)
        return True
    except BudgetBlocked:
        return False


def test_40_processes_share_batch_reservations(tmp_path):
    ledger = Ledger(tmp_path / 'budget.json', create=True)
    for i in range(40):
        ledger.seed(str(i), {}, origin='batch')
    with ProcessPoolExecutor(max_workers=10) as pool:
        accepted = list(pool.map(_process_reserve, [(str(ledger.path), str(i)) for i in range(40)]))
    assert sum(accepted) == 33
    assert sum(ledger.totals(ledger.snapshot()).values()) == micros(990)
    assert len(ledger.snapshot()['alerts']) == 1


def test_authorized_batch_increase_preserves_old_policy_and_all_balances(tmp_path):
    from tech_tree_arena.runtime.run_budget import deployed_policy
    ledger = Ledger(tmp_path / 'budget.json', create=True)
    for i in range(21):
        ledger.seed(str(i), {'generator': 50} if i < 20 else {}, origin='batch')
    state = ledger.snapshot()
    state['policy'] = deployed_policy()
    ledger.path.write_text(json.dumps(state))
    ledger = Ledger(ledger.path, expected=deployed_policy())
    with pytest.raises(BudgetBlocked):
        ledger.reserve('over', '20', 'generator', micros(1))
    before = ledger.snapshot()
    ledger.authorize_batch_limit(1200, reason='User raised this cohort to USD 1200')
    after = ledger.snapshot()
    for key in ('papers', 'attempts', 'policy'):
        assert after[key] == before[key]
    restored = Ledger(ledger.path, expected=deployed_policy())
    restored.reserve('new', '20', 'generator', micros(50))
    assert restored.batch_limit(restored.snapshot()) == micros(1200)
    with pytest.raises(BudgetBlocked):
        restored.reserve('role-over', '0', 'generator', 1)
    with pytest.raises(BudgetBlocked):
        restored.authorize_batch_limit(1000, reason='invalid decrease')


def test_parallel_judge_waits_for_reservation_settlement(tmp_path):
    ledger = ledger_at(tmp_path)
    ledger.reserve('first', 'paper', 'judge', micros(8))
    with ThreadPoolExecutor() as pool:
        future = pool.submit(ledger.reserve, 'second', 'paper', 'judge', micros(8), timeout=2)
        time.sleep(.1)
        assert not future.done()
        ledger.settle('first', cost=1, known=True)
        future.result(timeout=2)
    assert ledger.totals(ledger.snapshot())[('paper', 'judge')] == micros(9)


def test_unknown_billing_and_imported_hold_never_become_free(tmp_path):
    ledger = ledger_at(tmp_path)
    ledger.reserve('lost', 'paper', 'generator', micros(8))
    ledger.settle('lost', cost=0, known=False)
    ledger.seed('old', {}, origin='old', holds=[{'id': 'unfinished', 'role': 'generator', 'reserve': micros(9)}])
    state = Ledger(ledger.path).snapshot()
    assert sum(ledger.totals(state, include_pending=False).values()) == micros(17)
    assert state['attempts']['lost']['known'] is False
    with pytest.raises(BudgetBlocked):
        ledger.settle('lost', cost=0, known=True)


def test_checkpoint_truncation_breaker_survives_restart(tmp_path):
    ledger = ledger_at(tmp_path)
    for i in range(2):
        ledger.reserve(str(i), 'paper', 'generator', micros(7), 'checkpoint', 128000)
        ledger.settle(str(i), cost=6.4, known=True, truncated=True)
    with pytest.raises(BudgetBlocked, match='exhausted 128k twice'):
        Ledger(ledger.path).reserve('third', 'paper', 'generator', micros(7), 'checkpoint', 128000)
    ledger.reserve('different', 'paper', 'generator', micros(7), 'different checkpoint', 128000)


def test_reservation_full_output_and_unknown_model_rejected():
    body = {'model': 'claude-fable-5-1', 'max_tokens': 128000, 'messages': []}
    assert request_reservation(body, 'anthropic') > micros(6.4)
    with pytest.raises(BudgetBlocked):
        request_reservation({**body, 'model': 'claude-future-9'}, 'anthropic')


def test_recorded_ledger_cannot_be_deleted_to_reset(tmp_path):
    manifest = {'target_id': 'paper', 'generator_memory': {'model': policy()['model']}}
    setup(None, manifest, tmp_path)
    from pathlib import Path
    Path(manifest['run_budget']['ledger']).unlink()
    with pytest.raises(BudgetBlocked, match='missing'):
        setup(None, manifest, tmp_path)


def test_import_deduplicates_copied_provider_attempts_and_all_stages(tmp_path):
    from tech_tree_arena.replay.recorder import HashChainWriter
    runs = tmp_path / 'runs'
    first, second = runs / 'directional', runs / 'essence'
    first.mkdir(parents=True)
    second.mkdir()
    (second / 'manifest.json').write_text(json.dumps({'target_id': 'paper'}))
    attempt = {'kind': 'provider_attempt_finished', 'provider': 'anthropic', 'model': 'claude-opus-5',
               'logical_request_sha': 'one', 'attempt': 0, 'started_at': 1,
               'usage_available': True, 'cost_usd': 6}
    for directory in (first, second):
        HashChainWriter(directory / 'provider-attempts.private.jsonl').append({'role': 'judge', 'attempt': attempt})
    HashChainWriter(second / 'extra-unused.jsonl')  # no side effects
    ledger = Ledger(tmp_path / 'budget.json', create=True)
    import_existing(ledger, second)
    assert ledger.totals(ledger.snapshot())[('paper', 'judge')] == micros(6)
    import_existing(ledger, second)
    assert ledger.totals(ledger.snapshot())[('paper', 'judge')] == micros(6)


def test_provider_parse_retry_is_billed_and_readmitted(tmp_path, monkeypatch):
    from tech_tree_arena.runtime import provider_client as pc
    from tech_tree_arena.runtime.providers import ModelProviderBackend
    class Messages:
        calls = 0
        def create(self, **request):
            self.calls += 1
            payload = {'id': f'synthetic-{self.calls}', 'content': [{'type': 'text', 'text': '{"wrong":true}' if self.calls == 1 else '{"ok":true}'}],
                       'usage': {'input_tokens': 100, 'output_tokens': 20}}
            return SimpleNamespace(model_dump=lambda: payload)
    messages = Messages()
    monkeypatch.setattr(pc, 'anthropic_client', lambda: SimpleNamespace(messages=messages))
    ledger = ledger_at(tmp_path)
    backend = ModelProviderBackend()
    backend.attempt_guard = AttemptGuard(ledger, 'paper', 'judge')
    response = backend.structured(model='anthropic/claude-opus-5', developer='synthetic', user='synthetic',
        schema_name='test', schema={'type': 'object', 'properties': {'ok': {'type': 'boolean'}},
                                  'required': ['ok'], 'additionalProperties': False}, max_output_tokens=8000)
    assert response == {'ok': True} and messages.calls == 2
    attempts = list(ledger.snapshot()['attempts'].values())
    assert len(attempts) == 2 and all(a['charge'] and a['known'] for a in attempts)
    ledger.seed('blocked', {'judge': 10}, origin='other')
    backend.attempt_guard = AttemptGuard(ledger, 'blocked', 'judge')
    with pytest.raises(BudgetBlocked):
        backend.structured(model='anthropic/claude-opus-5', developer='x', user='x', schema_name='test',
                           schema={'type': 'object'}, max_output_tokens=8000)
    assert messages.calls == 2


def test_openai_http_and_parse_retries_each_reserve_and_settle(tmp_path, monkeypatch):
    from tech_tree_arena.runtime import provider_client as pc
    class APIConnectionError(Exception):
        pass
    class Client:
        calls = 0
        def with_options(self, **kwargs):
            return self
        @property
        def responses(self):
            return self
        def create(self, **body):
            self.calls += 1
            if self.calls == 1:
                raise APIConnectionError('synthetic')
            return SimpleNamespace(id=f'response-{self.calls}', output_text='{' if self.calls == 2 else '{"ok":true}',
                                   usage=SimpleNamespace(input_tokens=100, output_tokens=10))
    client = Client()
    ledger = ledger_at(tmp_path)
    monkeypatch.setattr(pc, 'client', lambda: client)
    monkeypatch.setenv('OPENAI_RETRY_MAX_DELAY_S', '0')
    with pc.provider_attempt_guard(AttemptGuard(ledger, 'paper', 'oracle')):
        result = pc.structured(model='gpt-5.5', developer='synthetic', user='synthetic',
                               schema={'type': 'object'}, schema_name='test', max_output_tokens=8000)
    assert result == {'ok': True} and client.calls == 3
    attempts = list(ledger.snapshot()['attempts'].values())
    assert len(attempts) == 3 and all(a['charge'] is not None for a in attempts)
    assert sum(not a['known'] for a in attempts) == 1
    unknown = next(a for a in attempts if not a['known'])
    assert unknown['charge'] == unknown['reserve'] > 0


def test_tightening_migrates_old_ledger_without_forgiving_spend(tmp_path):
    from tech_tree_arena.runtime.run_budget import previous_policy
    ledger = ledger_at(tmp_path, {'generator': 49})
    ledger.reserve('inflight', 'paper', 'generator', micros(1))
    old = ledger.snapshot()
    old['policy'] = previous_policy()
    ledger.path.write_text(json.dumps(old))
    upgraded = Ledger(ledger.path, expected=previous_policy())
    after = upgraded.snapshot()
    assert after['papers'] == old['papers']
    assert after['attempts'] == old['attempts']
    assert after['alerts'] == old['alerts']
    assert after['policy']['role_usd']['generator'] == 50
    assert len(after['policy_history']) == 1
    with pytest.raises(BudgetBlocked):
        upgraded.reserve('next', 'paper', 'generator', 1, timeout=0)
    upgraded.settle('inflight', cost=.5, known=True)
    upgraded.reserve('fits', 'paper', 'generator', micros(.5))
    assert len(Ledger(ledger.path).snapshot()['policy_history']) == 1
    bad = previous_policy()
    bad['role_usd']['generator'] = 500
    with pytest.raises(BudgetBlocked):
        Ledger(ledger.path, expected=bad)
