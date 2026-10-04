"""Durable, process-shared two-stage API admission and checkpoint circuit breaker."""
from contextlib import contextmanager
from decimal import Decimal, ROUND_CEILING
import copy
import fcntl
import hashlib
import json
import os
from pathlib import Path
import threading
import time

from ..errors import ResourceLimitExceeded


class BudgetBlocked(ResourceLimitExceeded):
    code = 'run_budget_blocked'
    retryable = False


def micros(value):
    value = Decimal(str(value))
    if not value.is_finite() or value < 0:
        raise ValueError('Invalid monetary amount')
    return int((value * 1_000_000).to_integral_value(rounding=ROUND_CEILING))


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def policy(model='anthropic/claude-fable-5-1'):
    if model != 'anthropic/claude-fable-5-1':
        raise ValueError('The calibrated cumulative budget currently applies to Fable 5.1 only')
    return {'kind': 'common-cumulative-budget-v1', 'model': model,
            'role_usd': {'generator': 50, 'oracle': 15, 'judge': 10},
            'paper_usd': 100, 'batch_usd': 1000, 'alert_usd': 800,
            'max_checkpoint_truncations': 2, 'truncation_output_tokens': 128000,
            'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}


def previous_policy():
    """Exact deployed $80 policy, eligible for the authorized $50 tightening."""
    return {**policy(), 'role_usd': {'generator': 80, 'oracle': 15, 'judge': 10},
            'source_sha256': '7d7448e4469a21b71d2eae3953bd03829a6039ad8283da696b372e47f513142e'}


def deployed_policy():
    return {**policy(), 'source_sha256': '4d8cf111dfe7bb26d6d1267390dda6475520c902849628301c21f86207214a2e'}


def request_reservation(body, provider):
    """Conservative text-only input bound + full requested output, no cache discount."""
    from . import provider_client as pc
    model = body.get('model', '')
    output = body.get('max_output_tokens', body.get('max_tokens'))
    if type(output) is not int or output <= 0:
        raise BudgetBlocked('Budget admission requires an explicit output ceiling')
    # Include every serialized field and protocol-overhead headroom. This avoids
    # depending on an optimistic chars/token ratio or assuming a cache hit.
    input_bound = len(json.dumps(body, ensure_ascii=False).encode()) + 4096
    if provider == 'anthropic':
        # The existing billing table prices the verified 5.1 route as Fable 5.
        rate_model = 'claude-fable-5' if model == 'claude-fable-5-1' else model
        rates = pc._ANTHROPIC_CLASS_RATES.get(rate_model)
        if rates is None:
            raise BudgetBlocked('No verified Anthropic reservation rate')
        in_rate, out_rate = max(rates[:4]), rates[4]
    elif provider == 'openai':
        if model not in pc._PRICING:
            raise BudgetBlocked('No verified OpenAI reservation rate')
        in_rate, out_rate = pc._PRICING[model]
        if model in {'gpt-6-astra', 'gpt-5.6-sol'}:
            in_rate *= 1.25  # Most expensive reported input class: cache writes.
        if input_bound > 272000:
            in_rate *= 2
            out_rate *= 1.5
    else:
        raise BudgetBlocked('Uncalibrated provider in the Fable cumulative budget')
    return micros((Decimal(input_bound) * Decimal(str(in_rate)) + Decimal(output) * Decimal(str(out_rate))) / 1_000_000)


class Ledger:
    def __init__(self, path, *, create=False, expected=None):
        self.path = Path(path).expanduser().resolve()
        self.expected = policy()
        if expected is not None and expected not in (self.expected, previous_policy(), deployed_policy()):
            raise BudgetBlocked('Budget policy or implementation changed')
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._locked(create=create):
            pass

    @contextmanager
    def _locked(self, *, create=False):
        with self.path.with_suffix(self.path.suffix + '.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            if self.path.exists():
                state = json.loads(self.path.read_text())
            elif create:
                state = {'policy': self.expected, 'papers': {}, 'attempts': {}, 'alerts': []}
            else:
                raise BudgetBlocked('Shared budget ledger missing; refusing to reset spend')
            if state['policy'] == previous_policy():
                state.setdefault('policy_history', []).append({
                    'from': state['policy'], 'to': self.expected, 'time': time.time(),
                    'reason': 'User lowered cumulative Generator ceiling from $80 to $50'})
                state['policy'] = copy.deepcopy(self.expected)
            if state['policy'] != self.expected:
                if self.expected in (policy(), deployed_policy()) and state['policy'] == deployed_policy():
                    # Preserve the deployed policy so in-flight old workers can
                    # still settle their reservations during a rolling update.
                    self.expected = state['policy']
                else:
                    raise BudgetBlocked('Shared ledger policy differs')
            yield state
            temp = self.path.with_suffix(self.path.suffix + f'.{os.getpid()}.{threading.get_ident()}.tmp')
            with temp.open('w') as f:
                json.dump(state, f, sort_keys=True)
                f.flush()
                os.fsync(f.fileno())
            os.replace(temp, self.path)
            # Persist rename before admitting an outbound request.
            fd = os.open(self.path.parent, os.O_RDONLY)
            try:
                os.fsync(fd)
            finally:
                os.close(fd)

    def seed(self, paper, role_costs, *, origin, holds=()):
        """One-time baseline import; recoveries cannot lower or reset charges."""
        baseline = {role: micros(role_costs.get(role, 0)) for role in self.expected['role_usd']}
        with self._locked() as state:
            if paper in state['papers']:
                if state['papers'][paper]['origin'] != origin:
                    raise BudgetBlocked('Paper budget already belongs to another trajectory')
                return
            state['papers'][paper] = {'baseline': baseline, 'origin': origin}
            for hold in holds:
                key = 'import:' + hold['id']
                state['attempts'][key] = {'paper': paper, 'role': hold['role'], 'reserve': hold['reserve'],
                    'charge': hold['reserve'], 'known': False, 'checkpoint': None, 'truncated': False,
                    'successful': False, 'output_tokens': hold.get('output_tokens', 50000)}
            self._alert(state)

    @staticmethod
    def totals(state, *, include_pending=True):
        amounts = {(paper, role): cost for paper, row in state['papers'].items() for role, cost in row['baseline'].items()}
        for attempt in state['attempts'].values():
            amount = attempt['charge']
            if amount is None:
                amount = attempt['reserve'] if include_pending else 0
            key = (attempt['paper'], attempt['role'])
            amounts[key] = amounts.get(key, 0) + amount
        return amounts

    def _alert(self, state):
        total = sum(self.totals(state).values())
        if total >= micros(self.expected['alert_usd']) and not state['alerts']:
            state['alerts'].append({'kind': 'batch_budget_warning', 'threshold_usd': self.expected['alert_usd'],
                                    'committed_and_reserved_usd': total / 1e6, 'time': time.time()})

    def batch_limit(self, state):
        override = state.get('authorized_batch_limit')
        if override is None:
            return micros(self.expected['batch_usd'])
        if (self.expected['kind'] != 'common-cumulative-budget-v1'
                or override.get('model') != self.expected['model']
                or not override.get('reason')
                or micros(override['usd']) < micros(self.expected['batch_usd'])):
            raise BudgetBlocked('Invalid authorized batch budget')
        return micros(override['usd'])

    def authorize_batch_limit(self, usd, *, reason):
        """Record an explicit user-authorized increase without rewriting balances."""
        with self._locked() as state:
            if self.expected['kind'] != 'common-cumulative-budget-v1' or not reason:
                raise BudgetBlocked('Batch increase requires Fable policy and authorization reason')
            if micros(usd) < self.batch_limit(state):
                raise BudgetBlocked('This operation only increases the batch limit')
            change = {'usd': float(usd), 'model': self.expected['model'],
                      'reason': reason, 'time': time.time()}
            state.setdefault('batch_limit_history', []).append(change)
            state['authorized_batch_limit'] = change

    def reserve(self, key, paper, role, amount, checkpoint=None, output_tokens=0, timeout=1500):
        deadline = time.monotonic() + timeout
        while True:
            wait = False
            with self._locked() as state:
                if state.get('halt'):
                    raise BudgetBlocked(state['halt'])
                if paper not in state['papers'] or role not in self.expected['role_usd']:
                    raise BudgetBlocked('Unregistered paper or role')
                if key in state['attempts']:
                    raise BudgetBlocked('Attempt already reserved; do not resend it')
                streak = 0
                for a in sorted(state['attempts'].values(), key=lambda a: a.get('sequence', -1)):
                    if checkpoint and (a['paper'], a['role'], a['checkpoint']) == (paper, role, checkpoint):
                        if a['successful']:
                            streak = 0
                        elif a['truncated'] and a['output_tokens'] >= self.expected['truncation_output_tokens']:
                            streak += 1
                if streak >= self.expected['max_checkpoint_truncations']:
                    raise BudgetBlocked('Same checkpoint exhausted 128k twice; automatic retry disabled')
                def fits(pending):
                    values = self.totals(state, include_pending=pending)
                    return (values.get((paper, role), 0) + amount <= micros(self.expected['role_usd'][role])
                        and sum(v for (p, _), v in values.items() if p == paper) + amount <= micros(self.expected['paper_usd'])
                        and sum(values.values()) + amount <= self.batch_limit(state))
                if not fits(False):
                    raise BudgetBlocked('Cumulative role/paper/batch budget cannot fund the full next request')
                if not fits(True):
                    # Other workers hold reservations: queue rather than fail
                    # an otherwise affordable request or oversubscribe the cap.
                    wait = True
                else:
                    state['attempts'][key] = {'paper': paper, 'role': role, 'reserve': amount,
                        'charge': None, 'known': False, 'checkpoint': checkpoint, 'truncated': False,
                        'successful': False, 'output_tokens': output_tokens, 'sequence': len(state['attempts'])}
                    self._alert(state)
            if not wait:
                return
            if time.monotonic() >= deadline:
                raise BudgetBlocked('Budget reservations still pending; no new request was sent')
            time.sleep(.05)

    def settle(self, key, *, cost, known, truncated=False, successful=False):
        with self._locked() as state:
            a = state['attempts'][key]
            charge = micros(cost) if known else a['reserve']
            if a['charge'] is not None:
                if a['charge'] != charge or a['known'] != known:
                    raise BudgetBlocked('Conflicting settlement of a physical request')
                return  # A copied journal/parse update is not another bill.
            a.update(charge=charge, known=bool(known), truncated=bool(truncated), successful=bool(successful))
            if charge > a['reserve']:
                state['halt'] = 'Provider charge exceeded conservative reservation; reconcile pricing before more calls'
            self._alert(state)

    def snapshot(self):
        with self._locked() as state:
            return copy.deepcopy(state)


class AttemptGuard:
    def __init__(self, ledger, paper, role):
        self.ledger, self.paper, self.role = ledger, paper, role

    def native_start(self, record, body, provider, checkpoint):
        self.ledger.reserve('native:' + record['attempt_id'], self.paper, self.role,
            request_reservation(body, provider), checkpoint, body['max_tokens'])

    def native_finish(self, record):
        u = record['usage']
        self.ledger.settle('native:' + record['attempt_id'], cost=u.get('cost_usd', 0),
            known=not u.get('unknown_usage_calls'), truncated=record.get('stop_reason') in {'max_tokens', 'length'},
            successful=record.get('status') == 'ok')

    @staticmethod
    def provider_key(record):
        return 'provider:' + digest({k: record[k] for k in ('provider', 'model', 'logical_request_sha', 'attempt', 'started_at')})

    def provider_event(self, record, body=None):
        key = self.provider_key(record)
        if record['kind'] == 'provider_attempt_started':
            if body is None:
                raise BudgetBlocked('Missing physical request for admission')
            self.ledger.reserve(key, self.paper, self.role, request_reservation(body, record['provider']),
                output_tokens=body.get('max_output_tokens', body.get('max_tokens', 0)))
        elif record['kind'] == 'provider_attempt_finished':
            self.ledger.settle(key, cost=record.get('cost_usd', 0), known=record.get('usage_available') is True,
                               successful=record.get('status') == 'ok')


def default_path(runs_root):
    runs_root = Path(runs_root).resolve()
    # The multi-paper driver explicitly supplies this path too. Detect its
    # frozen experiment layout for one-shot/manual recoveries of older runs.
    for parent in runs_root.parents:
        if (parent / 'experiment.json').is_file():
            return parent / 'budgets' / 'fable-cumulative-v1.json'
    return runs_root / '.budgets' / 'fable-cumulative-v1.json'


def import_existing(ledger, source_root):
    """Seed all known physical charges, including discarded branches/retries.

    No stage subtraction: Directional and Essence share the same paper balance.
    Journals contain metadata only; request bodies are read solely to reserve
    unfinished requests. Native history or target contents are never exported.
    """
    source_root = Path(source_root).resolve()
    manifest = json.loads((source_root / 'manifest.json').read_text())
    paper = str(manifest['target_id'])
    origin = str(source_root.parent)
    if paper in ledger.snapshot()['papers']:
        if ledger.snapshot()['papers'][paper]['origin'] != origin:
            raise BudgetBlocked('Paper already registered to a different run collection')
        return
    native, nstart, provider, pstart = {}, {}, {}, {}
    for path in source_root.parent.glob('.artifacts/sha256/native-attempts-*.jsonl'):
        for line in path.read_text().splitlines():
            row = json.loads(line)
            key = row['attempt_id']
            if row['kind'] == 'attempt_started':
                nstart[key] = row
            elif row['kind'] == 'attempt_finished':
                if key in native and native[key] != row:
                    raise BudgetBlocked('Conflicting native billing records')
                native[key] = row
    for path in source_root.parent.glob('*/provider-attempts.private.jsonl'):
        for line in path.read_text().splitlines():
            row = json.loads(line)
            a = row['attempt']
            key = row['role'] + ':' + AttemptGuard.provider_key(a)
            if a['kind'] == 'provider_attempt_started':
                pstart[key] = (row['role'], a)
            elif a['kind'] == 'provider_attempt_finished':
                if key in provider and provider[key] != (row['role'], a):
                    raise BudgetBlocked('Conflicting provider billing records')
                provider[key] = (row['role'], a)
    costs = {role: Decimal(0) for role in ledger.expected['role_usd']}
    for role, row, usage, known in [('generator', r, r['usage'], not r['usage'].get('unknown_usage_calls')) for r in native.values()] + [
            (role, r, r, r.get('usage_available') is True) for role, r in provider.values()]:
        if not known:
            # The historical journal lacks a trustworthy maximum for generic
            # unknown calls. Explicit reconciliation is required, never zero it.
            raise BudgetBlocked('Historical request has unknown billing; cannot initialize a reliable balance')
        costs[role] += Decimal(str(usage.get('cost_usd') or 0))
    holds = []
    from ..replay.artifacts import ArtifactStore
    artifacts = ArtifactStore(source_root.parent)
    for key, row in nstart.items():
        if key not in native:
            body = artifacts.load_json(row['request_body'])
            holds.append({'id': key, 'role': 'generator', 'reserve': request_reservation(body, 'anthropic'),
                          'output_tokens': body['max_tokens']})
    if any(key not in provider for key in pstart):
        raise BudgetBlocked('Historical provider request still lacks final billing; reconcile before enabling')
    ledger.seed(paper, {k: str(v) for k, v in costs.items()}, origin=origin, holds=holds)


def import_batch(ledger, batch):
    """Initialize the whole paused cohort before admitting any one paper."""
    registered = ledger.snapshot()['papers']
    for runs in sorted((Path(batch) / 'fable').glob('*/runs')):
        if runs.parent.name in registered:
            continue
        manifests = sorted(runs.glob('*/manifest.json'))
        if manifests:
            import_existing(ledger, manifests[-1].parent)


def setup(backends, manifest, runs_root, *, source_root=None, path=None):
    """Attach the calibrated policy to Fable 5.1 before any live model call."""
    model = (manifest.get('generator_memory') or {}).get('model')
    recorded = manifest.get('run_budget')
    if model != 'anthropic/claude-fable-5-1':
        if path or recorded:
            raise BudgetBlocked('This calibrated budget is for Fable 5.1')
        return
    if recorded and path and Path(path).resolve() != Path(recorded['ledger']).resolve():
        raise BudgetBlocked('Cannot change the shared ledger during recovery')
    location = (recorded or {}).get('ledger') or path or default_path(runs_root)
    ledger = Ledger(location, create=not bool(recorded), expected=(recorded or {}).get('policy'))
    for parent in Path(runs_root).resolve().parents:
        if (parent / 'experiment.json').is_file():
            import_batch(ledger, parent)
            break
    paper, origin = str(manifest['target_id']), str(Path(runs_root).resolve())
    if source_root and not recorded:
        import_existing(ledger, source_root)
    elif not recorded:
        ledger.seed(paper, {}, origin=origin)
    elif paper not in ledger.snapshot()['papers']:
        raise BudgetBlocked('Recorded paper balance missing; refusing a reset')
    manifest['run_budget'] = {'ledger': str(ledger.path), 'policy': ledger.expected, 'paper': paper, 'origin': origin}
    for role, backend in (backends or {}).items():
        guard = AttemptGuard(ledger, paper, role)
        if role == 'generator':
            backend.transport.budget_guard = guard
        else:
            backend.attempt_guard = guard
    if ledger.snapshot()['alerts']:
        import warnings
        warnings.warn('Cumulative batch spend plus reservations reached the $800 warning threshold', RuntimeWarning)
