from types import SimpleNamespace
import pytest
from tech_tree_arena.runtime.subscription_api_budget import SubscriptionLedger, setup
from tech_tree_arena.runtime.run_budget import BudgetBlocked, Ledger, micros


def test_subscription_keeps_cash_roles_separate_and_resume_never_resets(tmp_path):
    backends={role:SimpleNamespace() for role in ('generator','oracle','judge')}
    manifest={'target_id':'paper','generator_memory':{'model':'gpt-6-astra'}}
    setup(backends,manifest,tmp_path,path=tmp_path/'budget.json')
    assert not hasattr(backends['generator'],'attempt_guard')
    ledger=backends['oracle'].attempt_guard.ledger
    ledger.reserve('first','paper','oracle',micros(15))
    ledger.settle('first',cost=14,known=True)
    setup(backends,manifest,tmp_path,source_root=tmp_path/'source')
    with pytest.raises(BudgetBlocked):
        ledger.reserve('next','paper','oracle',micros(2))
    assert ledger.expected['role_usd']=={'oracle':15,'judge':10}
    with pytest.raises(BudgetBlocked):
        ledger.reserve('bad','paper','generator',1)


def test_fable_ledger_cannot_be_used_for_subscription(tmp_path):
    path=tmp_path/'budget.json';Ledger(path,create=True)
    with pytest.raises(BudgetBlocked):
        SubscriptionLedger(path)


def test_deployed_policy_survives_locked_read_fix(tmp_path):
    import json
    path=tmp_path/'budget.json';ledger=SubscriptionLedger(path,create=True)
    ledger.seed('paper',{'oracle':3},origin='source')
    state=ledger.snapshot();state['policy']['source_sha256']='2a632b24446c4e7560d2abe548f1a940f215236854a59dddf06c1b815e8bbf32'
    path.write_text(json.dumps(state))
    restored=SubscriptionLedger(path,expected=state['policy'])
    assert restored.snapshot()==state
