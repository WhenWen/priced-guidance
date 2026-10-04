"""Explicit budgets for the Opus/Sol/GLM comparison; existing runs stay frozen."""
import hashlib
from pathlib import Path
from .run_budget import Ledger, AttemptGuard, BudgetBlocked

MODELS = {"anthropic/claude-opus-5", "gpt-5.6-sol", "together/zai-org/GLM-5.3"}

def policy(model):
    if model not in MODELS:
        raise BudgetBlocked("Unsupported comparison model")
    opus = model == "anthropic/claude-opus-5"
    return dict(kind="comparison-cumulative-budget-v1", model=model,
        role_usd={"generator": 50, "oracle": 15, "judge": 10} if opus else {"generator": 0, "oracle": 15, "judge": 10} if model.startswith("together/") else {"oracle": 15, "judge": 10},
        paper_usd=100, batch_usd=1200 if opus else 1000, alert_usd=800,
        max_checkpoint_truncations=2, truncation_output_tokens=128000,
        generator_meter="API USD" if opus else "Codex subscription" if model == "gpt-5.6-sol" else "Excluded from budget by user; usage retained",
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        ledger_source_sha256=hashlib.sha256(Path(__file__).with_name("run_budget.py").read_bytes()).hexdigest())

class ComparisonLedger(Ledger):
    def __init__(self, path, *, model, create=False, expected=None):
        self.path = Path(path).expanduser().resolve()
        self.expected = policy(model)
        if expected is not None and expected != self.expected:
            raise BudgetBlocked("Comparison budget policy changed")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._locked(create=create):
            pass

class ExcludedGeneratorGuard(AttemptGuard):
    """Keep retry/truncation protection while excluding Generator dollars."""
    def native_start(self, record, body, provider, checkpoint):
        self.ledger.reserve("native:" + record["attempt_id"], self.paper, self.role,
                            0, checkpoint, body["max_tokens"])

    def native_finish(self, record):
        self.ledger.settle("native:" + record["attempt_id"], cost=0, known=True,
            truncated=record.get("stop_reason") in {"max_tokens", "length"},
            successful=record.get("status") == "ok")

def setup(backends, manifest, runs_root, *, source_root=None, path=None):
    model = manifest["generator_memory"]["model"]
    recorded = manifest.get("comparison_budget")
    if not (path or recorded):
        raise BudgetBlocked("Comparison requires an explicit budget ledger")
    if source_root and not recorded:
        raise BudgetBlocked("Historical comparison requires billing import")
    if recorded and path and Path(path).resolve() != Path(recorded["ledger"]).resolve():
        raise BudgetBlocked("Cannot replace comparison ledger on resume")
    ledger = ComparisonLedger((recorded or {}).get("ledger") or path, model=model,
        create=not bool(recorded), expected=(recorded or {}).get("policy"))
    paper, origin = str(manifest["target_id"]), str(Path(runs_root).resolve())
    if recorded:
        if paper not in ledger.snapshot()["papers"]:
            raise BudgetBlocked("Recorded balance missing")
    else:
        ledger.seed(paper, {}, origin=origin)
    manifest["comparison_budget"] = dict(ledger=str(ledger.path), policy=ledger.expected, paper=paper, origin=origin)
    for role, backend in (backends or {}).items():
        if role not in ledger.expected["role_usd"]:
            continue
        guard_type = ExcludedGeneratorGuard if role == "generator" and model.startswith("together/") else AttemptGuard
        guard = guard_type(ledger, paper, role)
        if role == "generator":
            backend.transport.budget_guard = guard
        else:
            backend.attempt_guard = guard
