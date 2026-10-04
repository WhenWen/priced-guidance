import pytest
from types import SimpleNamespace
from tech_tree_arena.runtime.comparison_budget import ComparisonLedger, policy, setup, ExcludedGeneratorGuard
from tech_tree_arena.runtime.run_budget import BudgetBlocked, micros

@pytest.mark.parametrize('model,limit', [('anthropic/claude-opus-5',1200),('gpt-5.6-sol',1000),('together/zai-org/GLM-5.3',1000)])
def test_budget_persists_and_rejects_other_model(tmp_path,model,limit):
    path=tmp_path/'ledger.json'
    ledger=ComparisonLedger(path,model=model,create=True)
    ledger.seed('paper',{},origin='runs')
    ledger.reserve('one','paper','oracle',micros(1))
    ledger.settle('one',cost=.4,known=True)
    restored=ComparisonLedger(path,model=model,expected=policy(model))
    assert restored.totals(restored.snapshot())[('paper','oracle')]==micros(.4)
    assert restored.batch_limit(restored.snapshot())==micros(limit)
    with pytest.raises(BudgetBlocked):
        ComparisonLedger(path,model='gpt-5.6-sol' if model!='gpt-5.6-sol' else 'anthropic/claude-opus-5')
    with pytest.raises(BudgetBlocked):
        restored.reserve('two','paper','oracle',micros(15),timeout=0)

def test_glm_excludes_cost_but_preserves_truncation_breaker(tmp_path):
    ledger=ComparisonLedger(tmp_path/'ledger.json',model='together/zai-org/GLM-5.3',create=True)
    ledger.seed('p',{},origin='runs')
    guard=ExcludedGeneratorGuard(ledger,'p','generator')
    for i in range(2):
        guard.native_start({'attempt_id':str(i)},{'max_tokens':128000},'together','checkpoint')
        guard.native_finish({'attempt_id':str(i),'usage':{'cost_usd':100},'stop_reason':'length','status':'error'})
    assert ledger.totals(ledger.snapshot())[('p','generator')]==0
    with pytest.raises(BudgetBlocked,match='exhausted 128k twice'):
        guard.native_start({'attempt_id':'3'},{'max_tokens':128000},'together','checkpoint')

def test_sol_setup_and_resume_keeps_balance(tmp_path):
    from tech_tree_arena.runtime.subscription_api_budget import setup as route
    manifest={'generator_memory':{'model':'gpt-5.6-sol'},'target_id':'p'}
    backends={r:SimpleNamespace() for r in ['generator','oracle','judge']}
    route(backends,manifest,tmp_path/'runs',path=tmp_path/'ledger.json')
    assert not hasattr(backends['generator'],'attempt_guard')
    ledger=backends['oracle'].attempt_guard.ledger
    ledger.reserve('one','p','oracle',micros(1));ledger.settle('one',cost=.5,known=True)
    route(backends,manifest,tmp_path/'runs',source_root=tmp_path/'parent')
    assert backends['oracle'].attempt_guard.ledger.totals(ledger.snapshot())[('p','oracle')]==micros(.5)
