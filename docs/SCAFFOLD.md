# The Modular Three-Stage Scaffold (reference pair)

Status date: 2026-08-30. Source of truth: `submissions/reference_pair`
(submission name `reference-pair`, version `1.17.1`). This document describes
the architecture for review; campaign results and target-specific evidence
live in `docs/research-handoffs/` and must never be copied into participant
code.

---

## 1. Problem and arena contract

The arena plays an idea-recovery game over a hidden target paper:

- The **Generator** is target-blind. It authors finite Questions (option slates
  with declared probabilities) and eventually a Submission.
- The **Guide** holds the private gold summary. It communicates with the
  Generator ONLY by selecting one displayed option per Question. Selecting an
  option with displayed probability `p` costs roughly `-log2(p)` bits; the run
  score `K` is the accumulated information cost along the active path.
- The **Judge** (GPT-5.5, evaluation-owned) grades Submissions against gold.
  Judge feedback is private to the Guide and must never enter Generator state.
- Three Judge rungs form a ladder: **directional** (same problem/area/
  contribution type + plausible mechanism) -> **essence** (recognizable defining
  mechanism, coarse) -> **strict/fmn** (exact defining mechanism, faithful
  scope, no overclaims). A pass at rung N can be promoted to rung N+1 by
  forking the run at its private before-judge boundary.

Models: the Generator and Guide each make ordinary structured model calls
through the injected services (`IDEA_ARENA_GENERATOR_MODEL` /
`IDEA_ARENA_GUIDE_MODEL`); GPT-5.5 is the Judge. There is no persistent CLI
conversation in the default profile — see §5 for why, and §5.4 for the
optional `--guide-agent` lane (including a human guide).

## 2. Package layout

```
submissions/reference_pair/
  submission.toml            # entrypoints + module map (name/version/protocol)
  README.md                  # public participant description (info-boundary safe)
  participant/
    generator.py             # entrypoint: re-exports Generator from pair.py
    guide.py                # entrypoints: Guide (default) and AgentGuide
    pair.py                  # SHARED: stable actor shell, ledger/toolkit, dynamic dispatcher
    stages/
      directional.py          # complete Directional actor policy + prompts/menu
      directional_channels.py  # stage-owned keyword-channel implementation
      essence.py               # complete Essence actor policy
      essence_channels.py      #   "
      strict.py                # complete Strict actor policy
      strict_channels.py       #   "
```

`submission.toml` maps four hash-tracked module groups: `shared` (pair.py,
entrypoints, README), `directional`, `essence`, `strict` — each stage group
holds the stage module plus its channels file. Promotion freezes `shared` plus
every completed stage group (§7.3). Shared owns only behavior that is truly
frozen across the whole ladder; every replaceable Generator/Guide decision
enters through the active stage group's function interface.

## 3. Stage modules: the only stage-specific surface

Each stage module carries its actor entry functions and policy data, plus a
stage-owned keyword channels file in the same hash group:

| Attribute | Role |
|---|---|
| `JUDGE_CRITERION` | Exact current-rung criterion rendered into this stage's Guide prompt; an earlier prompt never reads a later module |
| `generator_step` / `guide_step` | Complete replaceable message-handling boundary for the two actors; a stage may wrap or replace the shared default locally |
| `generator_on_enter` / `guide_on_enter` | Service-free destination-stage initialization hooks; new keys use `setdefault` so the shared constructor remains frozen |
| `dispatch_question`, `mc_question`, `candidate_question`, `run_state_update`, `correction_*`, `submission` | Fine-grained Generator seams, allowing one later-stage mechanism to change without copying the whole dispatcher |
| `STAGE_GOAL` | Stage-aware goal injected into every Generator prompt: names the ladder rung, the current rung's public Judge criterion, and what earlier rungs established |
| `DROPPED_ROUTES` | Routes this stage does not display (see §4.3); displayed probabilities renormalize over the remaining menu |
| `MC_PROMPT` | Atomic multiple-choice channel instructions |
| `KEYWORD_CATEGORY_PROMPT` / `KEYWORD_PROMPT` / `KEYWORD_GUESS_PROMPT` | Category-first identity-phrase machinery (v1.8): a paid 16-way semantic-category slate gates the 64-phrase slate + prefix completion; directional targets the central contribution name, essence the defining mechanism, strict the exact missing detail |
| `EXPLORE_PROMPT` / `AUDIT_PROMPT` | Whole-claim candidate slates (where the stage displays them) with the diversity-axis and cross-draft-synthesis contracts |
| `DIFFERENTIATE_PROMPT` | Core-preserving single-relation additions (where displayed) |
| `*_COUNT`, `*_ACTION` | Slate sizes and apply semantics (`replace`/`add`/`rewrite`) |
| `*_channels.py` | The complete keyword-channel implementation for that stage (questions, prices, validators, state transitions); the shared dispatcher only forwards to it |

## 4. Generator architecture

### 4.1 State ledger

A single JSON-serializable `state` dict: `stage`, `current_draft`, `facts`
(append-only ledger; each fact has id/text/active/source `stage:channel`),
`whiteboard` (model-authored prose belief state), `primary_field`/
`secondary_field` taxonomy paths, `category`, negative-evidence stores
(`rejected_candidate_regions` as compact fingerprints, `rejected_mc_questions`,
`rejected_keywords`), `insufficient_directional_drafts` (stable draft IDs + the
recovery mode that declared them insufficient), and the pending slates +
`active_dispatch_binding`. `_compact_state` renders the Generator-visible view
(bounded) into every model prompt. Judge text and gold never enter this state.

### 4.2 Bootstrap

Before any dispatch the Generator walks the public research taxonomy one priced
edge at a time (primary path), then asks the contribution-category Question.
The first draft is written by the state-update turn from the top idea after
any fact lands, so no whole-draft channel is needed to bootstrap.

### 4.3 Dispatch: exact eager previews

The current implementation authors and caches every displayed route's complete
downstream Question before the Guide chooses. Each dispatch option contains
that exact serialized Question, including its option IDs, payloads and final
probabilities. The bundle is bound to the source stage, current draft and fact
ledger hash. Selecting a route activates its cached Question without another
model call; stale or modified bindings are rejected. An oversized bundle fails
instead of publishing a partial preview.

Preview builders work on isolated copies of pending state. Unselected slates
do not mutate the live draft, fact ledger, whiteboard or paid negative evidence.
The optional source-built Codex backend preserves this eager behavior and
serializes the semantic calls into one native reasoning history. Its prompt
explicitly distinguishes remembered previews from purchased evidence. Further
model calls still occur when a channel requires them, such as after a keyword
category or correction axis is purchased. The older route-digest design is not
the current implementation.

Each stage displays only the routes its own measured usage justifies
(`DROPPED_ROUTES` in the stage module); the fixed mode probabilities
renormalize over the displayed subset, so every surviving route gets cheaper:

- **directional**: `mc`, `keyword`, `submit` (once a draft exists).
- **essence**: `mc`, `keyword`, `correct`, `submit`.
- **strict**: `mc`, `keyword`, `audit`, `correct`, `submit`.

`secondary_field` is displayed at no stage in the current tree, and `explore`
/ `differentiate` are dropped everywhere on measured usage. Selecting a
recovery route while a draft exists records that draft as insufficient (public
negative evidence); `correct` does not, because a repair is not a claim that
the draft is wrong.

### 4.4 Route catalog

- **mc** — one Generator-authored atomic clarification: 8 concrete values on
  one unresolved axis + `prob_all_incorrect` + ask-different-axis. A concrete
  selection adds exactly the deterministic fact `axis: value`. Rejection
  records the axis as priced negative evidence.
- **keyword** — the v0.4 machinery with the v1.8 category layer in front. The
  keyword preview is a paid 16-way slate of semantic CATEGORIES of the missing
  phrase; buying one commits it as an ordinary confirmed fact and only then is
  the 64-phrase direct slate generated (honest weights + one paid entry into
  the two-letter prefix reveal + retry). The 676-way reveal is priced from the
  displayed candidates' own implied letter distribution (declared `prob_hint`
  gate x half-uniform blended first/second-letter masses with the v0.4 letter
  floor). After every reveal/extension: one 16-way completion slate for the
  exact prefix, whose declared `prob_none` prices the extend-by-one-character
  exit; the character slate is the frequency-weighted 38-way alphabet plus
  accept-the-exact-prefix. When ordinary time travel is enabled there is no
  priced abandon row — a zero-cost checkout is the exit; without time travel
  the priced abandon row returns. A missed guess slate is negative evidence
  for that prefix only. Selecting a phrase adds only that public phrase as one
  DIRECT fact. Prefix bounded at 24 chars.
- **audit** (strict) — whole-claim revision slates that prune or repair
  overclaims, with the diversity-axis contract.
- **correct** (essence, strict) — two-step semantic correction over a claim
  already present in the draft. Step 1 offers 16 complete, truth-evaluable
  `before_claim` propositions that the draft expresses or entails; they need
  not be exact sentences or substrings. Step 2 offers 16 complete semantic
  patches in one of three modes: `replace`, `refine`, or `delete`, together
  with the complete resulting draft. Replace/refine must carry a non-empty
  `after_claim`, which becomes a canonical paid fact even when the old claim
  was a Generator-authored inference; delete must carry an empty after-claim.
  Correction cannot insert an independent missing claim. Runtime constrains
  the draft rewrite to one localized region and rejects malformed mode/claim
  pairs, duplicate drafts, and unchanged drafts.
- **submit** — a SubmitOption carrying the current draft as the idea. The
  Guide gates it on its own Judge reading (§5.3); the runtime attaches
  nothing.

### 4.5 Pricing mechanics

All slates emit exact protocol Decimal strings summing to 1
(`_decimal_distribution`, 60-digit context, positive-residual check).
Candidate slates use Generator-declared weights with a declared retry mass
(`_weighted_probabilities`); keyword slates use the v0.4 half-uniform
smoothing (`KEYWORD_PRICE_MIX=0.5`); malformed rows fold their raw mass into
retry rather than renormalizing survivors. Retiring the sole active fact
deterministically clears the draft and returns to broad search — an empty
ledger is a valid state, never a provider error.

### 4.6 Stage transitions

`StageTransition` messages are Arena-authored, service-free, and add no bits.
The shared shell validates the edge, flips `state["stage"]`, invokes the
destination module's `generator_on_enter`, keeps the full ledger, and hands off
a private summary (`StageReady.handoff`). The Guide similarly invokes
`guide_on_enter`. Pending slates survive transitions so later-stage Guides
can checkout historical Questions; the newly active stage remains responsible
for consuming those frozen earlier payload shapes.

## 5. Guide architecture: full context, one call per turn (v1.11)

### 5.1 Design and why the CLI conversation was dropped

Through v1.10 the Guide ran as one persistent CLI conversation. Measuring
what that conversation actually did was alarming: the CLI compacted it on its
own, losing 40-84% of the context, as often inside a session as at a
deliberate restart, and the binary deciding when lived in an auto-updating
desktop app whose version changed underneath the experiments.

v1.11 keeps the Guide and drops the conversation. Every turn is one ordinary
structured model call carrying, in full: `MATCH_INIT` verbatim (gold target
JSON, the active rung's exact Judge criterion, arena rules, and a purely
descriptive `MODULAR_DISPATCH_NOTES` block explaining every route), the
Guide's own last `state_summary` inside `<RESUMED_PRIVATE_STATE>`, and the
current `ARENA_EVENT`. Gold, criterion, and rules can no longer be compacted
away; every call is priced through the Arena's own provider client, so no
turn can go unaccounted. Output is a strict 5-field JSON action (`choose` an
offered option or `checkout` an eligible question), validated fail-closed.

The decision policy keeps truthfulness as a hard constraint
(whole-proposition entailment; partly-true compounds are false), cost-aware arm
selection at dispatch, submit-gating on the exact Judge criterion,
checkout/time-travel rules, and mandatory post-rejection recovery. It also
requires conservative correction recovery: if an interpretation is unsupported
and no displayed replacement is target-supported, first delete the complete
unsupported proposition, mark its relation unresolved, and use keyword/MC at the
next dispatch to rediscover that relation rather than refining the same guess.

### 5.2 What replaced local bookkeeping

The old stateless per-Choice Guide's enforcement (just-rejected-channel
masking, exhausted-correction tracking) lives in the Guide's own
`state_summary`; taxonomy questions are answered as ordinary priced Choices
from gold.

### 5.3 Stage transitions and Judge previews

Transitions perform no service call; the criterion change is queued and
delivered as an ordinary `ARENA_EVENT` prefixed to the next paid turn.

Judge previews belong to the Guide: it calls the injected
`services.judge_evaluate(ideas)` itself, once per distinct draft (an unchanged
draft cannot earn a different verdict). The Arena used to attach previews on a
fixed schedule; that scheduling is removed. The Judge stays Arena-owned, its
spend lands on the Judge's own budget, and the call is taped for replay. A
profile that gives the Guide no Judge simply goes without.

### 5.4 Optional agent-backend lane (`--guide-agent`)

Passing `--guide-agent {claude-code,codex,human}` swaps the guide entrypoint
to `participant.guide:AgentGuide`, which sends the **identical** per-turn
prompt through `services.agent_turn` instead of `structured_model`:

1. **human** — `HumanGuideBackend` writes each turn to
   `run_dir/agent-workspace/oracle/human/turn-NNNN.request.md` and blocks until
   a person writes the matching `.answer.json`. Human turns record zero model
   usage. Timeouts default to a day; the actor wall clock is raised to match.
2. **claude-code** — Claude Code CLI with structured output.
3. **codex** — codex CLI; with an Anthropic model it is routed through the
   per-run localhost adapter `runtime/anthropic_responses_proxy.py` (audit log
   `anthropic-adapter.requests.jsonl`, terminated at exit) and priced with
   Anthropic's exact sheet.

Sessions are fresh per turn (the v1.11 full-context transport is preserved; no
CLI decides what the Guide remembers), and every turn is journaled and
replayable like any other service call. The chosen backend is pinned in the
run manifest; resume and promotion rebuild it from there (`--guide-agent` on
`resume` overrides it explicitly and records the override).

There is no Generator counterpart: the agent-Generator lane was removed
(see the TODO in the top-level README).

## 6. Judge

Judges are evaluation-owned (`research-directional/-essence/-fmn`, GPT-5.5).
A coarse pre-screen gates the full judgment. Formal submissions that fail
return private `SubmissionFeedback` (verdicts + mandatory checkout targets) to
the Guide only; the Guide-requested preview path is §5.3.

## 7. Runtime guarantees the scaffold depends on

### 7.1 Recording and replay

Every service call (model/agent/random) is journaled with hash chains;
provider attempts are separately journaled and reconciled per-call against
service metadata (multiset binding, exact frozen-rate cost checks). Actor
replay re-executes participants against the recorded tape; `idea-arena replay
--actor <run>` must pass for any run used as evidence. Concurrency support:
id-multiplexed worker wire protocol, thread-local per-call provider
accounting, interleaving-tolerant attempt pairing, conservation-based meter
validation, recorded-order tape flushing, fatal-path draining, and judge-era
tracking so promoted runs replay pre-promotion judgments under the Judge that
produced them.

### 7.2 Resume and escalation

`idea-arena resume <run>` forks a new run from the last durable checkpoint;
budget caps are lineage-cumulative (`--max-cost-usd-per-role`), so escalation
= resume with a higher cap. Runs interrupted by an uncaught provider failure
resume with `--retry-interrupted-call` (the completeness gates relax only
under that flag).

### 7.3 Promotion freeze

`resume --promote-judge <stage>` forks a completed passing run at its private
before-judge boundary under the next Judge. `shared` and every completed stage
module group are hash-frozen against the source snapshot; only later-stage
groups may change (via `--compatible-submission`). Consequence: **changing
pair.py or an entrypoint invalidates promotion of every existing pass**. The
three stage actor files are the intended iteration surface: after a Directional
pass, an Essence or Strict actor function may change without moving the shared
or Directional hashes; after an Essence pass, the same is true only for Strict.
The sibling channels file is covered by the same stage hash.

The shared dispatcher uses dynamic loading. A static import from shared into a
stage is rejected by manifest validation, and an earlier stage must not read a
later stage's route menu, criterion, or actor function while producing replayed
output. The Guide therefore renders only the active stage's criterion and
route sentence; changing Strict cannot perturb a Directional service request.

### 7.4 Ladder orchestration

`tools/run_judge_ladder.py` runs best-of-N per stage with exact nested prefix
curves, records per-target `ladder.json` (experiment identity includes the
submission hash, caps, and guide-agent config), auto-promotes passing rungs
(`--stop-after`), and passes through `--guide-agent*` flags.

## 8. Operations manual

```bash
# Full ladder for one paper (directional -> essence):
PYTHONDONTWRITEBYTECODE=1 uv run python tools/run_judge_ladder.py \
  submissions/reference_pair \
  --target-pack <pack> --target <id> \
  --attempts 1 --stop-after essence --seed 1 \
  --runs-dir <runs-dir> --env-file .env \
  --max-cost-usd-per-role 20 --threshold-K 200

# Escalate an exhausted lineage:
uv run idea-arena resume <run> --env-file .env \
  --max-cost-usd-per-role <higher> --no-html-report --retry-interrupted-call

# Promote a pass to the next rung (uses the run's own snapshot):
uv run idea-arena resume <pass-run> --env-file .env \
  --promote-judge <essence|strict> --max-cost-usd-per-role <cap> --no-html-report

# Verify any evidence run:
uv run idea-arena replay --actor <run>
```

## 9. Known issues and open questions

1. **Per-paper judge criteria are public text in the Guide prompt**; the
   Judge's actual rubric is richer (coarse screen + full grading). Divergence
   between the stated criterion and the real screen has shown up as
   "preview passes but formal judgment rejects".
2. **Submit gating rhythm**: the Guide reads the Judge once per distinct
   draft, so a draft completed mid-loop (e.g. final keyword acceptance) is
   preview-checked when it next appears in a dispatch question.
3. **Route menus are tuned on small samples** (tens of recorded dispatches per
   stage); `DROPPED_ROUTES` trades measured tails for cheaper menus and should
   be revisited as evidence accumulates.
