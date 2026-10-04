# Occurrence accounting

The Arena uses `occurrence-prior-v2` for live choices, budgets and canonical scores. Its fixed prior is

```text
epsilon = 0.05
pi(1) = 0.95
pi(j) = 0.05 / (j * (j - 1))   for j >= 2
choice_cost = -log2(p) - log2(pi(j))
```

Here `j` counts committed choices of the same option at the same exact stored question. These counters never rewind. The total continuation index remains available for auditing and has no separate charge. Every choice, including an authorized submission, pays the occurrence surcharge. Abandoned edge costs disappear when checkout restores a saved prefix. Accepted-mass pointer and fixed-repeat Judge costs remain unchanged.

The prior is normalized because the tail telescopes to 0.05. Its first code length is about 0.074 bits and its second is about 5.322 bits. Repricing replaces the earlier charge directly, so some records become cheaper.

## Reprice recorded runs

```sh
idea-arena reprice RUN_DIRECTORY_OR_ROOT --report accounting-dry-run.json
idea-arena reprice RUN_DIRECTORY_OR_ROOT --write --report accounting-migration.json
```

The default is read-only. With `--write`, each completed, valid schema-4 record receives a canonical `score.json` using the same pricing function as live execution. The byte-for-byte original is archived as `score.recorded.json`. Canonical scores carry `accounting_version` and `repriced_from`, which binds the original score and terminal private event hash. Event chains, branch snapshots, manifests and promotion checkpoints remain unchanged. Repeating the command verifies the stored result and makes no further pricing change.

Protocol replay validates the recorded chain and historical accounting, then recomputes and checks any migrated canonical score. Its `result` uses canonical pricing and `recorded_result` exposes the historical result. Arithmetic repricing follows chronological counters and immutable path snapshots, including restoration to an abandoned branch. It never reconstructs an old prefix from the latest choice or final counter values.

Migration refuses running records, unsupported event schemas, broken chains, inconsistent scores and inconsistent branch state. The batch report lists each excluded record. It does not infer repairs to source evidence. A per-run lock prevents competing migrations; a stale lock left after a killed migration requires inspection before removal. `score.json` is atomically replaced after its original archive is durable.

The historical advisory-preview protocol is supported for protocol-level accounting verification. Resuming or reconstructing its actors requires the historical participant protocol. Current runs never emit those preview events.

## Resume and promotion

A schema-4 checkpoint using the earlier pricing is verified before loading. At the start of its derived continuation, an explicit `accounting_updated` event installs current prices for all saved nodes, the active prefix and any pending submit choice. Its node-cost digest is recomputed by replay. Counters and participant states are preserved. The current information budget is checked before continued execution. Native v2 resumes need no accounting transition.

Both the manifest and durable branch state record the accounting version. A promotion's `promotion_base_k` follows the current price; when an upgrade occurs, `promotion_recorded_base_k` retains the historical base. No historical event chain is edited to pretend it originally used the new prices.

## Interpretation and report scope

The output certifies deterministic accounting consistency with the fixed prior. A successful stochastic run supplies one observation of the recovery theorem's expectation term. It does not by itself certify an unconditional fresh-run probability. The common-kernel, public-state and evaluation assumptions must still hold, and choosing a prior after inspecting historical outcomes needs a valid selection argument. In particular, Generator stage transitions in promotion records require a separate kernel/state audit; the migration report marks their presence.

Canonical run scores are updated locally. Exported experiment tables, standalone result caches, published figures and remote copies must be regenerated or migrated from the corresponding run directories. Historical timeline events retain their recorded costs and must be distinguished from a migrated headline score.
