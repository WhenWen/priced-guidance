# Priced Guidance

**How much information does a language model need to recover a research idea?**
Priced Guidance measures this as a compression cost in bits. A Generator proposes
questions and possible answers; a Guide, called the **Oracle** in the code, knows
the target paper and selects answers. Each answer has a price determined by the
Generator's probability assignment. An independent judge checks whether the
resulting idea recovers the target at the Directional or Essence level.

This repository contains the Idea Arena runtime, reference Generator–Oracle
pairs, the paper's 40-paper development and 87-paper test cohorts, and tools for
creating new cases. The installable package and command-line tool are both named `idea-arena`. See [the release scope](docs/RELEASE_SCOPE.md) for the retained
experiments and [the paper assets](paper/README.md) for historical configurations.

We welcome **pull requests with new test cases, new submission pairs, or
compression results**. You do not need to beat a reference result to contribute;
a well-documented comparison or unsuccessful run is useful too.

## Install and try an offline run

Python 3.13 is recommended (Python 3.11+ is required). From the repository root:

```bash
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev,providers]'
idea-arena doctor
idea-arena validate submissions/examples/minimal_pair
idea-arena run submissions/examples/minimal_pair \
  --target-pack smoke --seed 1 --runs-dir generated/smoke --no-html-report
```

If Python 3.13 is not installed, `uv venv --python 3.13 .venv` can create the
environment instead. On Linux, common-memory/native Generator runs and their
isolation tests also require `bubblewrap` and enabled unprivileged user namespaces
(for example, `sudo apt-get install bubblewrap` on a compatible Ubuntu host).
On macOS, use a standalone Python such as `uv`'s Python 3.13: the Homebrew
Python 3.14 framework launcher is not supported by the current sandbox profile.

The smoke example uses no model APIs. Its output includes a `run_dir`, status,
and score. All commands below assume this environment is active and that you
are at the repository root. The offline test suite is `python -m pytest -q`.

## 1. Create a new test case with an Opus 5 summary

A test case is a paper's structured **gold summary**, generated from its arXiv
LaTeX source. The summary is visible to the Oracle and judge; the Generator only
receives public taxonomy resources and the information purchased through choices.

**Choose papers.** Create a JSONL file with one object per paper. For example:

```bash
mkdir -p generated/my-cases
cat > generated/my-cases/papers.jsonl <<'EOF'
{"arxiv_id": "2604.27351"}
EOF
```

Replace the example ID with your paper IDs. A `title` field is optional for
summary generation, but required for a title-only memorization audit. Optional
`daily_paper_date` and `dataset_split` fields preserve collection metadata.

**Generate summaries with Claude Opus 5.** Set `ANTHROPIC_API_KEY` in your shell,
then run:

```bash
python tools/target_packs/opus5_batch_summary.py \
  --papers generated/my-cases/papers.jsonl \
  --out-dir generated/my-cases/summaries \
  --model claude-opus-5 --effort high --service-tier standard
```

This downloads arXiv source and makes paid Anthropic API calls. It writes
`json/ARXIV_ID.json`, readable `markdown/ARXIV_ID.md`, and aggregate usage reports.
For a larger collection, use `--service-tier batch`; add `--no-wait` to return
after submission and rerun the same command to collect results. Reuse the same
inputs, model, and output directory when resuming. Use a new output directory
when changing the configuration, because completed summaries are reused.

The code preserves the study's model ID and summary prompt. Availability depends
on your provider account. If you substitute a model, record its exact ID and
label the cases accordingly. USD reports use the script's historical price table,
not a current billing quote. The summarizer requires LaTeX source and truncates
flattened source at 140,000 characters; inspect source metadata and the rendered
summary before accepting a case.

**Review and build the pack.** Check that the summary identifies the setting,
main object, and novel findings faithfully. Then package the cases:

```bash
python tools/target_packs/build_development_pack.py \
  --selection generated/my-cases/papers.jsonl \
  --gold-dir generated/my-cases/summaries \
  --public-source-pack src/tech_tree_arena/data/target_packs/development40 \
  --name my-cases --out-dir target-packs/my-cases
python tools/validate_target_pack.py target-packs/my-cases
```

The builder creates `pack.toml`, public taxonomy files, and
`secret/gold/ARXIV_ID.json`, with integrity hashes. Here `secret` means
**evaluator-only benchmark answers**, not credentials. `kind = "development"`
in the file format means the pack can run in the local evaluator, even when its
scientific split is test. The local runner is intended for trusted participants;
it is not a hosted hidden-test service.

Run a reference pair on a new case using the next section's command, replacing
`--target-pack test87 --target 2604.27351` with
`--target-pack target-packs/my-cases --target YOUR_ARXIV_ID`.

For a benchmark intended to measure unseen-paper recovery, audit title-only
recall before treating the cases as held out. The tool
`tools/target_packs/title_recall_memorization.py --help` describes the inputs.
Report the tested models, title treatment, screening threshold, and exclusions;
the screening model must not receive the gold summary while producing its recall.
See [contributing test cases](CONTRIBUTING.md#new-test-cases) for PR contents.

## 2. Add a submission pair

A **submission pair** consists of a Generator and an Oracle, a manifest, and
any private strategy modules owned by those participants. Start with either:

```bash
# Small offline pair: easiest way to learn the protocol.
idea-arena new submissions/my_pair

# Or start a separate pair from the full one-candidate reference scaffold.
idea-arena new submissions/my_reference_pair --reference
```

You can also create your own pair following the structure of
`submissions/reference_pair_submit8`, our eight-candidate Directional reference.
Give your pair a unique `name` and version in `submission.toml`, and implement
`participant/generator.py`, `participant/oracle.py`, and any strategy modules.
The reference entrypoints delegate most strategy code to `participant/stages/`.

```toml
schema_version = 1
name = "my-pair"
version = "0.1.0"
protocol = "idea-recovery-v1"
generator = "participant.generator:Generator"
oracle = "participant.oracle:Oracle"
```

Keep the reference template's `[modules]` section if you use its staged scaffold.
Use injected `services` for model calls and randomness so the run can be replayed.
The Generator authors finite option distributions; the Oracle may choose an
option or check out an earlier question, but cannot send unpriced text back to
the Generator. A submission is allowed only after the Oracle selects a
`SubmitOption`. [The player guide](docs/PLAYER_GUIDE.md) includes runnable class
examples, the service interface, manifests, and stage transitions.

Validate the pair before running it:

```bash
idea-arena validate submissions/my_pair
idea-arena run submissions/my_pair \
  --target-pack smoke --seed 1 --runs-dir generated/my-pair-smoke --no-html-report
```

The second command applies to the minimal offline template. A full research
reference pair needs model credentials and a research pack with taxonomy resources.
If you add third-party participant dependencies, include both `pyproject.toml`
and an exact `uv.lock` in the submission folder.

## Run a compression experiment

Set your provider credentials in the shell, or copy `.env.example` to an ignored
`.env` and pass `--env-file .env`. This example needs Anthropic for the Generator
and Oracle, and OpenAI for the judge:

```bash
export IDEA_ARENA_ORACLE_MODEL=anthropic/claude-fable-5
export IDEA_ARENA_JUDGE_MODEL=gpt-5.5
idea-arena run submissions/reference_pair_submit8 \
  --target-pack test87 --target 2604.27351 --seed 1 \
  --judge directional --generator-model anthropic/claude-opus-5 \
  --generator-memory common --generator-memory-effort high \
  --max-information-bits 1024 --max-cost-usd-per-role 25 \
  --runs-dir generated/compression --no-html-report --progress
```

Replace the submission path to evaluate your own pair. The USD limit is a
**per-role ceiling**. This small
example budget can stop before recovery; disclose changed budgets in comparisons.
Use development40 for method development and reserve test87 for final evaluation.

To continue a passing Directional run under the Essence judge, use the actual
`run_dir` printed by the first command:

```bash
idea-arena resume generated/compression/RUN_ID --promote-judge essence --progress
idea-arena replay generated/compression/RUN_ID
idea-arena replay generated/compression/RUN_ID --actor
```

Promotion creates a derived run and prints its own directory. Replay that new
directory to check the Essence result. Retain the source run and artifact store
locally. [The paper guide](paper/README.md) identifies the frozen stage-specific
participants and the additional promotion-cost accounting used in the manuscript.
A fresh run is a new measurement; current providers and maintained participants
need not reproduce a historical trajectory exactly.

## Share compression results

**Interested in improving idea compression? Submit a PR.** Add your pair under
`submissions/`, and a result note under `results/YOUR_EXPERIMENT/` using the
[results template](results/TEMPLATE.md). Include exact model IDs, efforts,
Generator memory/backend, judge settings, target IDs, seeds, budgets, code
revision, and per-target compression costs, including failures. State which
accounting version you used and whether Essence includes promotion costs.

Compare on the same cohort and settings. Keep failed targets in the denominator;
do not report only successful cases or reroll failures silently. For ensembles,
average recovery probabilities before converting back to bits. Replay original
runs locally and include the validation outcome. Share a reviewed result table
and, if useful, a public trajectory report rather than entire run directories.

[CONTRIBUTING.md](CONTRIBUTING.md) describes the PR workflow and required evidence.
No paid runs are needed to contribute code or run the test suite.

## Repository map

| Path | Contents |
| --- | --- |
| `src/tech_tree_arena/` | Contract, pricing, judges, runtime, recording and replay |
| `submissions/` | Minimal example and maintained reference pairs |
| `src/tech_tree_arena/data/target_packs/` | Smoke, development40, and test87 packs |
| `tools/target_packs/` | Opus summary generation, pack authoring, recall audit |
| `paper/` | Frozen paper participants, curated results, and provenance |
| `results/` | Community compression results and reporting template |
| `tests/` | Offline protocol, runtime, isolation, and participant tests |

The code is [MIT licensed](LICENSE). Paper titles, identifiers, source papers,
and upstream software retain their respective attribution and licenses; see
[NOTICE](NOTICE).
