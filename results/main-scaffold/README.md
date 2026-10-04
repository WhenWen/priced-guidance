# Main-scaffold results used by the Results Explorer

**[Download the Google Drive data](https://drive.google.com/drive/folders/1_Q5Y0L0OU_st2m90m8-ky4f4MvontswA).**
The folder is readable and downloadable by anyone with the link.

This is the website's current main test-set dataset: 87 papers, five individual
models and their reported ensemble, both Directional and Essence. It contains
1,044 outcomes and 838 independent trajectories. Failed, budget-blocked, refused,
and unpromoted cases remain in the catalog. The eight development-ablation
datasets are outside this main-scaffold bundle.

## Download and verify

Download `metadata.zip`, the five `trajectories-MODEL.zip` files, and `SHA256SUMS`.
The folder's `README.md` explains the format. Verify downloaded archives with
`sha256sum -c SHA256SUMS` (or `shasum -a 256 -c SHA256SUMS` on macOS); this also
checks the separately downloaded README. Extract all six ZIPs into the same
folder, then run `python verify.py` there. Only the Python standard library is needed.

| Extracted file | Purpose |
| --- | --- |
| `catalog.json` | The website's paper/model metadata and all main-test outcomes |
| `results.csv` | Flat outcome table, including failures and ensemble member IDs |
| `run-index.csv` | Exact model IDs, effort, seed, judge repetitions, archive, and source event hash |
| `promotion_costs.json` | The 354 successful individual Essence promotion charges |
| `provenance.json` | Website/manuscript revisions and source hashes |
| `validation.json` | Cohort, endpoint, ensemble, and credential-scan results |
| `manifest.json` and `verify.py` | Checksums, trajectory completeness, and endpoint verification |
| `public/data/raw/RUN_ID.json.gz` | Sanitized event and exact Guide-message exports |
| `public/data/runs/RUN_ID.json.gz` | Browser trajectory records with targets and corrected display costs |

Catalog rows retain the site's fields: `paper`, `model`, `stage`, `status`,
`cost`, `id`, `dataset`, and `members`. A JSON null cost denotes no finite
successful recovery. A missing run ID is explicit. Ensembles reference the
individual run IDs and have no independent trajectory file.

The files in `raw/` and `runs/` are byte-identical to the site's selected exports.
“Raw” here means sanitized scientific exports; it does not mean private provider
tapes or full replay hash chains. The original exported event charges remain
unchanged. Browser `displayCost` includes occurrence-prior-v2, mixture-judge-v1,
and promotion-cost-v1, so do not add the promotion charges a second time.

## Use this format for a submission

Every compression-result submission must provide a **Google Drive folder** with
this logical layout and a README containing its exact commands and configuration.
Include all planned targets and seeds, statuses for unsuccessful trials, model
and judge settings, the submission/runtime revision, and replay-validation outcomes.
Include only your experiment's models; using five archives is not a requirement.
Large trajectory collections may be divided into ZIPs as above.

The folder must permit reviewers to view and download the files; “Anyone with
the link — Viewer” is recommended for open-source result submissions. Put the
Drive link in the PR description and in `results/YOUR_EXPERIMENT/README.md`.
Keep API keys, account files, private service tapes, and native histories out of
the shared folder. Use the sanitized website-export schema for the trajectories.

If you already have a compatible Results Explorer checkout, package its main
test dataset from the repository root with:

```bash
python tools/package_site_results.py \
  --site-root /path/to/results-site \
  --out-dir generated/my-drive-results
```

The packager reads the source without changing it, checks complete cohorts and
reported costs, and produces the model archives and metadata bundle. It expects
the existing website's `lib/catalog.json`, `public/data/{raw,runs}/`, and accounting
provenance. It does not turn an arbitrary private run directory into a sanitized
export automatically.

## Validation and publication record

The website's complete data test passed for all 1,509 trajectories and 33,802
Guide-choice links, including all 547 selected promotion charges across the main
and development datasets. This main-only export separately passed cohort,
endpoint, ensemble and promotion checks, with 18,462 priced choices and no
credential-pattern matches. All 1,684 extracted files passed checksum verification.
After upload, all eight Drive files' sizes matched the local files and their
inherited public read-only permissions were verified.

[drive.json](drive.json) records the actual uploaded file links, byte sizes,
SHA-256 checksums, and dataset scope. No manuscript or website files were changed.
