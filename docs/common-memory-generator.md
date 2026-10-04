# Common-memory Generator (implementation and validation)

This adapter keeps the participant policy from `reference-pair-submit8` 1.18.0
(commit `2829b494`, imported through Git unchanged). It adds a common memory
policy outside the source-built Codex runtime. It does not alter Oracle choices,
Judge criteria, probabilities, or K scoring. The existing API and native CLI paths
remain available when `--generator-memory common` is omitted.

The common system prompt now contains only transport-format instructions. All
participant developer/user task text is forwarded without rewriting. No additional
eager-preview or research-strategy instruction is prepended.

## Native histories and GLM formatting

- Astra and GPT-5.6 Sol use the pinned, receipt-verified open-source Codex build and the separate
  ChatGPT account directory. Normal continuation retains the native session and
  its cache lineage; rollback restores the selected immutable checkpoint.
- Fable 5.1, Fable 5, and Opus 5 use Anthropic Messages with adaptive thinking, max effort, an unchanged
  system prefix, append-only messages, and original signed blocks. The SDK-only
  `parsed_output` helper is excluded when serializing message blocks for the API. Binding checks
  use `thinking-binding-controls-2026-08-01` with mismatch behavior `error`.
- GLM 5.3 uses streaming Together requests with max effort and `clear_thinking: false`. The canonical
  assistant message keeps the original final answer and the provider's original
  reasoning field. No `response_format` is sent in this thinking stream.
- If the GLM final answer already satisfies the payload schema, parsing is local.
  Otherwise a separate stateless GLM call, with thinking disabled, converts only
  that final answer to schema-constrained JSON. It receives no native reasoning
  or prior conversation. It is instructed to preserve content, order, and weights.
  The formatter's answer never replaces the canonical assistant message. A valid
  JSON answer is never sent for a paid rewrite. Formatting retries reuse the same
  primary answer, and formatter exhaustion does not rerun thinking.

Schema validity alone cannot establish semantic equivalence for a malformed draft
that needed an LLM formatter. This is a transport difference to disclose in model
comparisons; formatter call/token counts are reported separately.

## Common compaction

Before a new semantic call, estimate the next input from the latest native context
size plus the UTF-8 byte length of the current request (including its schema).
The default trigger is 100,000; bytes are a conservative heuristic, not an exact
shared tokenizer. `--generator-memory-compact-calls` can force a boundary in a
smoke test and defaults to zero in normal runs.

At the boundary, make a summary call using the complete original native session,
including its native reasoning blocks. Request at most 12,000 characters of task
memory: working hypotheses, evidence references, rejected paths, and unresolved
work. Then begin a fresh native session with that summary and the current public
state. No old signed or encrypted reasoning block is transplanted onto the new
prefix. The summary operation itself inherits the original native history.

Cached Questions and eight-candidate slates remain exact participant state;
compaction does not summarize or regenerate those artifacts. Eager previews are
authored in a deterministic serial order within one paper. Parallel papers can
still run independently. Summary and failed attempts count toward resource usage,
but do not create priced Oracle choices or change K. Unexpected native Codex
compaction is detected and fails as a protocol mismatch.

The common trigger and memory budget define policy fairness. Native tokenizer,
reasoning effort scales, server cache behavior, and hidden state representations
remain provider-specific; this is not a claim of identical computation.

## Supported model routes (September 6, 2026)

| Model | Exact model ID | Transport | Default effort | Live output allowance |
|---|---|---|---|---|
| Astra | `gpt-6-astra` | Source-built Codex subscription | xhigh | Existing 50K post-response check |
| Sol | `gpt-5.6-sol` | Source-built Codex subscription | xhigh | Existing 50K post-response check |
| Fable 5.1 | `anthropic/claude-fable-5-1` | Anthropic API | max | 128K |
| Fable 5 | `anthropic/claude-fable-5` | Anthropic API | max | 128K |
| Opus 5 | `anthropic/claude-opus-5` | Anthropic API | max | 128K |
| GLM 5.3 | `together/zai-org/GLM-5.3` | Together API | max | 128K |

Sol must select `--generator-backend codex`, with the same independent account
and source-build receipt options as Astra. This extension does not add an OpenAI
API common-memory transport. Claude variants use their own existing price table
entries; no Fable 5.1 pricing is substituted for Opus. Model changes within a
checkpoint are rejected: each evaluation starts its own model-bound history.

`run`, `sample-ideas`, and the standalone `probe_common_memory.py` accept these
models. The old two-cohort dev40 launcher retains its Astra/Fable 5.1 experiment
layout. New model cohorts should use the explicit CLI model IDs above.

Anthropic's [preserved-thinking documentation](https://platform.claude.com/docs/en/build-with-claude/preserved-thinking)
says older models also accept the binding-control object and beta header. The
[Fable 5](https://platform.claude.com/docs/en/models/fable-5/overview) and
[Opus 5](https://platform.claude.com/docs/en/models/opus-5/overview) model pages
list 128K output. Task prompts, native thinking preservation, refusal handling,
compaction and eager-preview semantics remain unchanged.

The exact previous common-memory/native-transport source pair and output-budget
source receipt are accepted for the previously supported models only. Every
other dependency hash and policy field must still match. Unknown hashes and
using an old receipt for a newly added model fail closed. This preserves recovery
of the paused dev40 experiments without weakening model binding.

The three new routes were checked with synthetic transport responses, including
compaction and actor/protocol replay. No paid API/subscription probe or paper run
was started as part of this model-list extension; live account availability has
not been re-verified.

## Running

```sh
# Together API
.venv/bin/idea-arena run submissions/reference_pair_submit8 \
  --target-pack smoke --judge directional --env-file .env --generator-memory common \
  --generator-model together/zai-org/GLM-5.3 --generator-memory-effort max

# Anthropic API
.venv/bin/idea-arena run submissions/reference_pair_submit8 \
  --target-pack smoke --judge directional --env-file .env --generator-memory common \
  --generator-model anthropic/claude-fable-5-1 --generator-memory-effort max

# Source-built Codex + separate subscription account
.venv/bin/idea-arena run submissions/reference_pair_submit8 \
  --target-pack smoke --judge directional --generator-backend codex --generator-memory common \
  --generator-model gpt-6-astra --generator-memory-effort xhigh
```

Replace the target pack and select a paper for an evaluation. The `smoke` pack is
for plumbing, not scientific performance. Credentials are loaded by the trusted
host transport; the participant worker cannot read data, account files, or the
network. API transports expose no tools to the model. Codex has its existing OS
sandbox as well. Timeout is 1500 seconds per provider request (streaming transports also have
per-read timeout semantics). The participant step retains its existing 6000-second
ceiling, configurable with `IDEA_ARENA_ACTOR_TIMEOUT_S`. For Codex, the reported
`physical_calls` counter counts native turns; it does not enumerate internal HTTP
retries inside Codex.

Resume uses the recorded policy and source hashes. Changing the common-memory
implementation rejects resume rather than silently using different experimental
conditions. Private artifacts contain native histories and must not be published.
Native API attempts have a durable hash-chain journal written before each HTTP
request. Completed service records bind to native requests, responses, histories,
and summary overhead. A process killed during a request can leave an attempt with
unknown usage; the journal must be retained, and zero reported cost must not be
interpreted as proof that no provider charge occurred.

## Validation scope

The tests cover immutable ancestors, abandoned-branch isolation, signed-history
input to summaries, fresh windows, formatter isolation/retries, exact cached
8-idea activation, K semantics, OS sandbox denial, offline actor/protocol replay,
and recovery from a failed call. Linux-only namespace tests are skipped on macOS.

Live checks use only synthetic public input and do not read paper targets. The
memory probe is a diagnostic (a random marker, summary, then recall), not part of
the participant or a dev40 run. The submit8 probe invokes the actual eight-idea
expansion and cached submission code; other route Questions are explicit synthetic
fixtures, so it does not establish full-paper directional-pass performance.

GLM pricing must be configured using the existing Together rate environment
variables. `unpriced_calls > 0` means `cost_usd` is incomplete, not free usage.
Astra's zero API-dollar field represents subscription routing and is not a credit
estimate. Anthropic refusals before any output report input usage but are unbilled,
according to the [provider documentation](https://platform.claude.com/docs/en/build-with-claude/refusals-and-fallback).

The local release checks are recorded in [VALIDATION.md](VALIDATION.md). Historical
paid diagnostic artifacts are not included in this export.
