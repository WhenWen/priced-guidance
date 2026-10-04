"""API-side cumulative admission for subscription-backed Generator experiments."""
import hashlib
import fcntl
import json
from pathlib import Path

from .run_budget import Ledger, AttemptGuard, BudgetBlocked, policy as fable_policy, setup as fable_setup


def policy():
    return {**fable_policy(), 'kind': 'subscription-oracle-judge-budget-v1', 'model': 'gpt-6-astra',
            'role_usd': {'oracle': 15, 'judge': 10},
            'generator_meter': 'Codex subscription quota, not priced in USD',
            'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'ledger_source_sha256': fable_policy()['source_sha256']}


class SubscriptionLedger(Ledger):
    def __init__(self, path, *, create=False, expected=None):
        self.path = Path(path).expanduser().resolve()
        self.expected = policy()
        legacy = {**self.expected, 'source_sha256': '2a632b24446c4e7560d2abe548f1a940f215236854a59dddf06c1b815e8bbf32'}
        accepted = (self.expected, legacy)
        if expected is not None and expected not in accepted:
            raise BudgetBlocked('Subscription API budget policy changed')
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # NFS can fail a path lookup while another process atomically replaces
        # the ledger. Admission and this initial policy read share one lock.
        with self.path.with_suffix(self.path.suffix + '.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            if self.path.exists():
                recorded = json.loads(self.path.read_text())['policy']
                if recorded not in accepted:
                    raise BudgetBlocked('Cannot reuse another model budget ledger')
                self.expected = recorded
        with self._locked(create=create):
            pass


def setup(backends, manifest, runs_root, *, source_root=None, path=None):
    model = (manifest.get('generator_memory') or {}).get('model')
    from .comparison_budget import MODELS, setup as comparison_setup
    if model in MODELS and (path or manifest.get('comparison_budget')):
        return comparison_setup(backends, manifest, runs_root, source_root=source_root, path=path)
    recorded = manifest.get('subscription_api_budget')
    if model != 'gpt-6-astra' or not (path or recorded):
        return fable_setup(backends, manifest, runs_root, source_root=source_root, path=path)
    if source_root and not recorded:
        raise BudgetBlocked('Historical subscription runs need explicit billing import; no reset allowed')
    if recorded and path and Path(path).resolve() != Path(recorded['ledger']).resolve():
        raise BudgetBlocked('Cannot change subscription API ledger during recovery')
    ledger = SubscriptionLedger((recorded or {}).get('ledger') or path,
                                create=not bool(recorded), expected=(recorded or {}).get('policy'))
    paper, origin = str(manifest['target_id']), str(Path(runs_root).resolve())
    if recorded:
        if paper not in ledger.snapshot()['papers']:
            raise BudgetBlocked('Recorded API balance missing')
    else:
        ledger.seed(paper, {}, origin=origin)
    manifest['subscription_api_budget'] = {'ledger': str(ledger.path), 'policy': ledger.expected,
                                           'paper': paper, 'origin': origin}
    for role, backend in (backends or {}).items():
        if role in ledger.expected['role_usd']:
            backend.attempt_guard = AttemptGuard(ledger, paper, role)
