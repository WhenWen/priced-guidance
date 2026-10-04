# Local release validation

Validated on macOS with standalone CPython 3.13.5, without paid model calls.

| Check | Outcome |
| --- | --- |
| Full offline suite (`python -m pytest -q`) | 613 passed, 5 Linux-only tests skipped |
| `idea-arena doctor` | Contract checks and all three packaged cohorts pass |
| Development/test target validation | All 40 development and 87 test records load and match integrity hashes |
| New-case pipeline | Mocked Opus structured response writes summary JSON, builds a pack, and loads the target; real API inference was not run |
| New-pair workflow | Minimal and reference templates created; reference validation passed |
| Offline smoke run | Passed; protocol and actor replay passed |
| Source/wheel build | Both distributions built successfully |
| Isolated wheel installation | Doctor, both template commands, reference validation, and smoke execution passed outside the source checkout |
| Current paper tables | Rebuilt from all 87 targets, including failures and Essence promotion charges; reported main-model medians/P80 match the current manuscript |
| Release scan | No credential-pattern or known source-key matches in selected files; no broken local Markdown links or tracked files over 10 MiB |

Nine new authoring tests cover the Opus response-to-pack workflow, reusing an
existing destination, target tampering, malformed/duplicate IDs, unsafe archive
members, TeX includes outside the source tree, and standard/batch argument parsing.
Existing offline tests cover pricing, boundaries, recording, promotion, model
transport mocks, native-history handling, and reference participants.

An initial run with Homebrew CPython 3.14 failed 20 sandbox-dependent tests because
the framework launcher tried to spawn an executable outside the permitted launch
path. The full suite passes with standalone CPython 3.13. The README recommends
that environment; the sandbox was not weakened to accommodate the launcher.
Linux namespace/egress integration checks require a Linux host with bubblewrap;
they have not been executed as part of this local preparation.

The first GitHub Actions run selected the runner's preinstalled framework Python
on both Python 3.11 and 3.13, causing 20 sandbox-dependent failures per version
(593 passed, 5 skipped). The workflow now requires uv-managed standalone Python
with `uv sync --managed-python`; it preserves the sandbox restrictions and runs
both Python versions independently.

Live Opus summaries, provider-backed compression experiments, new native-backend
builds, and every historical trajectory are not claimed as rerun. The tests use
recorded fixtures or synthetic provider responses. No API credentials are included.

## Release manifest

From a Git checkout, `python tools/check_release.py` verifies the snapshot hashes,
credential patterns, prohibited local state, file sizes, and local Markdown links.
It checks tracked files; review and stage intended files before a new release.
After intentional changes, maintainers can refresh the release manifest with
`python tools/check_release.py --write-manifest` and stage `release-manifest.json`.
The manifest describes a reviewed release snapshot, not generated experiment output.

## Guide terminology

The terminology refactor keeps model prompts and schemas unchanged. Before/after
AST comparison of all 36 participant files confirmed identical logic after
normalizing renamed identifiers and excluding compatibility aliases. Regression
fixtures capture complete model-request hashes from commit `9e9c198` for all three
reference pairs in Directional, Essence, and Strict, including a follow-up turn.
Legacy manifest keys, entrypoints, CLI flags, and Python keyword aliases are
covered by compatibility checks. A smoke run recorded before the rename passes
both protocol replay and actor replay with the renamed runtime.
