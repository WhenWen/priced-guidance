# Experiment title

Describe the question, strategy, and main finding in a few sentences.

## Google Drive data (required)

- Folder URL:
- Sharing: reviewers can view and download (recommended: Anyone with the link — Viewer).
- Data follows [the reference layout](main-scaffold/README.md): catalog, all outcomes,
  run index/configuration, sanitized trajectories, provenance, and checksums.
- Validation command and outcome:

## Reproduce

- Runtime commit and pair path/version/hash:
- Target pack, manifest/hash, and exact target IDs:
- Development/tuning cases and held-out evaluation cases:
- Generator model, backend, effort, and memory/compaction settings:
- Oracle model and effort:
- Judge model, criterion, and repetitions:
- Seeds, planned trials, and run dates:
- Information, USD-per-role, question, and attempt limits:
- Accounting version and Essence promotion-cost treatment:
- Provider/model substitutions or infrastructure limitations:

```bash
# Paste the exact run and promotion commands here.
```

## Results

Add `results.csv` with these columns (extend them if useful):

```csv
target_id,seed,stage,status,compression_bits,api_cost_usd,run_id
```

Include every planned trial. Use `inf` for unsuccessful compression cost; explain
missing USD values and distinguish failed, budget-limited, and interrupted runs.
Report cohort size, successes, coverage, P50/P80 over the whole cohort, and a
baseline with matching settings. Show individual ensemble members and weights.
Describe repetitions, retries, exclusions, and uncertainty.

## Validation

Record protocol replay and actor replay commands/outcomes, target-pack validation,
and code tests. Link reviewed evidence included in this PR. Explain any missing
historical provider tapes or replay limitation.
