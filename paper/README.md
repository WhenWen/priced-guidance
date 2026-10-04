# Paper configurations and results

The released cohorts are available as `development40` and `test87` through
`idea-arena`; their complete targets and public taxonomies are package data.
The 40-paper cohort supports the development ablations; the 87-paper cohort
supports final model comparisons. They are the paper's supplied cohorts, not
newly screened targets.

## Frozen participants

| Directory under `paper/participants/` | Reported condition |
| --- | --- |
| `legacy/` | Earlier single-candidate reference scaffold |
| `one_candidate/` | One-candidate scaffold for the native-context comparison |
| `eight_candidates/` | Eight-candidate native-context ablation |
| `main_directional/` | Main common-memory Directional participant |
| `main_essence/` | Compatible Essence participant for promotion |

These are historical participant snapshots. The maintained runtime is shipped
in `src/`; not every historical runtime/backend revision or provider state is
frozen here. Existing `strict` stage modules are retained for manifest and replay
compatibility; Strict experiments are not part of the reported comparison.
`submissions/reference_pair_fable51` preserves the reported development guide
prompt variant; it should not be confused with the main reference condition.

To run a new Opus measurement with the paper participants, use the README command
with `paper/participants/main_directional` as the pair. Promote the selected
passing run with:

```bash
idea-arena resume RUN_DIR --promote-judge essence \
  --compatible-submission paper/participants/main_essence --progress
```

The main study uses eight Directional candidates; after promotion, the inherited
submission is judged first and later submissions contain one idea. Common memory
retains reasoning and compacts history. Use Opus 5 at high effort, or Fable 5.1 and
GLM 5.3 at max effort, as recorded in the paper. Astra and Sol use the pinned
Codex backend at xhigh effort; [native backend setup](../docs/codex-generator.md)
requires your own build and account. A smaller budget or a different backend is
a different condition, even when the model name matches.

## Test-set tables

`results/test87/compression.csv` and `summary.json` contain the current manuscript's
costs: occurrence-prior-v2, mixture-judge-v1, and the promotion-cost-v1 addition
for Essence. `arena_costs.csv` and `arena_summary.json` preserve the earlier arena
measurement inputs. `promotion_costs.csv` contains the 354 successful individual
Essence runs' extra charges, with original numerical fields and host paths removed.
All failed outcomes remain in the 87-target denominator. Summary JSON uses `null`
for a non-finite quantile; the CSV uses `inf`.

Rebuild the current tables offline:

```bash
python tools/rebuild_paper_results.py
```

The uniform ensemble averages the recovery probabilities of Fable, Opus, and
Astra, with failed members contributing zero and the denominator fixed at three.
P50/P80 are the 44th/70th ordered costs over all 87 targets.

## Promotion accounting

A raw arena Essence score does not include the manuscript's additional charge
for choosing the Directional-to-Essence switch. `tools/promotion_cost.py` exports
`promotion_bits(events)` for a verified, complete schema-4 event chain using the
paper's staged protocol. It charges the repeated/promoting Submit or the 1%
fallback switch, plus the cost of continuing through preceding Directional
questions. Add the returned `bits` to a successful raw Essence cost exactly once.
The function expects the paper's `mode-submit` scaffold; a custom switching scheme
requires its own documented valid accounting rather than blindly reusing this
helper. An unsuccessful recovery still has infinite compression cost.

Retain the full event chain when computing fresh promotion costs and replay it
before accounting. The frozen promotion table makes the released paper results
rebuildable without remote machines or provider calls. This compact code release
does not distribute all historical private event/provider tapes, so it does not
claim to independently replay the entire historical study.

## Provenance

`source-provenance.json` records source revisions and hashes for the initial
selected export. Source hashes describe bytes before documented portability
edits. `release-manifest.json` at the repository root records final distributed
file hashes. `docs/RELEASE_SCOPE.md` explains the selection. No original research
Git history is included.
