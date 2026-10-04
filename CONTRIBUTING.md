# Contributing

We welcome new cases, Generator–Oracle pairs, bug fixes, and compression results.
A negative result or a careful replication is welcome. Open a pull request with
a short description of what changed and enough evidence to check the result.

## New test cases

Follow the Opus 5 workflow in the [README](README.md#1-create-a-new-test-case-with-an-opus-5-summary).
Commit the selected-paper JSONL, reviewed structured gold records, `pack.toml`,
and public taxonomy resources under `target-packs/YOUR_PACK/`. Include a README
with public paper links, source versions/dates, exact summary model and effort,
prompt/source hashes if available, any manual changes, and the memorization
screening procedure. Keep the raw summary-generation records locally; share a
reviewed provenance record without account IDs, absolute machine paths, or keys.
Do not copy full source archives into the PR by default.

Run `python tools/validate_target_pack.py target-packs/YOUR_PACK`.
Use a new pack name for a changed benchmark so earlier results remain identifiable.
Do not place target summaries or paper-specific hints in Generator code or
public taxonomy resources. Gold answers belong only in `secret/gold/`.

## Submission pairs

Create a new folder in `submissions/` rather than replacing a reference pair.
Give the manifest a unique name and version. Include a brief pair README
explaining the strategy, changes from the reference, and how to run it.
Use injected services for nondeterminism and keep participant state on the
actor instance. Include targeted offline tests for changed protocol behavior.

```bash
idea-arena validate submissions/YOUR_PAIR
python -m pytest -q
```

Do not tune a Generator on held-out gold answers. Disclose which cases you used
while developing the pair, and evaluate improvements on a separate cohort.

## Compression results

Copy [results/TEMPLATE.md](results/TEMPLATE.md) to
`results/YOUR_EXPERIMENT/README.md` and add a CSV with one row per target, seed,
and judge stage. Report every planned trial, including unsuccessful and
budget-limited runs. Use `inf` for unsuccessful compression cost and a separate
status to distinguish a scientific miss, budget stop, and infrastructure failure.
Do not silently mix retries or cherry-pick the best seed.

Record the exact runtime/pair revision, target-pack hashes, model identifiers,
reasoning efforts, memory/backend, judge model/repeats, seed, limits, sample
size, and accounting convention. Publish the commands and a baseline using the
same settings. If live provider behavior changed, say so.

For a cohort of N targets, unsuccessful runs count as infinite cost for coverage
quantiles. For M ensemble members, use
`K = -log2(sum(2**(-K_i)) / M)`; unsuccessful members contribute zero probability
and still count in M. Keep USD spending separate from information cost in bits.
Directional-to-Essence promotion needs the manuscript's additional selection
accounting to support a paper-comparable claim; see [paper/README.md](paper/README.md).

Run `idea-arena replay RUN_DIR` and, when supported, `idea-arena replay RUN_DIR --actor`
on the originals. Report both outcomes and any limitation. Include reviewed
`score.json`/manifest excerpts or a public report if needed to substantiate the
table. Avoid copying private provider logs, native histories, checkpoints, account
files, or entire artifact stores into Git. Keep the originals for local audit.

## Pull request checklist

- Explain the contribution and link the relevant result note or case README.
- Include all planned targets and seeds, including failures, for empirical claims.
- State validation commands and outcomes; do not present unrun paid checks as tested.
- Check that the diff contains no credentials, account files, or local run stores.
- Contribute code under the repository's MIT license and preserve third-party notices.
