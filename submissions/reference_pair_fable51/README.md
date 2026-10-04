# Reference pair — Fable 5.1 prompt-safe variant

New runs use a 50,000-token model-output allowance for both roles in every
stage/channel. This includes reasoning tokens on OpenAI, is not a required
output length, and does not change monetary or timeout limits. This request
policy belongs to the frozen shared submission; use the original snapshot
for old trajectories rather than substituting this version during recovery.

This submission is a standalone fork of `reference-pair` v1.17.4. Its only
behavioral change is the Oracle structured-output wording: the fields still
carry a brief action justification and durable run summary, but the prompt no
longer describes them as private or hidden reasoning. This avoids Fable 5.1's
`reasoning_extraction` false positive without changing the Generator, stage
policies, option protocol, target data, Judge criteria, or JSON field names.

The reference pair separates its Generator and Oracle policies into three
replaceable stage actor modules, one per Judge rung:

1. `directional.py` + `directional_channels.py`: zero information to a
   recognizable contribution;
2. `essence.py` + `essence_channels.py`: a Directional draft to its defining
   mechanism or relation;
3. `strict.py` + `strict_channels.py`: an Essence draft to a faithful,
   sufficiently detailed draft.

Each stage's `.py` file owns the complete actor function surface
(`generator_step`, route builders, state update, submission, `oracle_step`,
and the two service-free `*_on_enter` hooks). Its sibling `_channels.py` is an
implementation file in the same stage hash group. The entrypoints and
`pair.py` are frozen `shared` files; `pair.py` is the stable actor shell,
ledger, validation/math toolkit, and dynamic dispatcher, not the owner of an
active stage's behavior. A stage pass can therefore be combined with a changed
later-stage module through Arena promotion:

```bash
uv run idea-arena resume <directional-pass-run> \
  --promote-judge essence \
  --compatible-submission submissions/reference_pair_fable51

uv run idea-arena resume <essence-pass-run> \
  --promote-judge strict \
  --compatible-submission submissions/reference_pair_fable51
```

The freeze boundary is prefix-shaped: after Directional, `shared` and
`directional` are immutable while Essence and Strict remain replaceable; after
Essence, only Strict remains replaceable. Shared code dynamically loads the
active stage instead of statically importing stage functions, because a static
shared -> mutable-stage import would violate the manifest dependency boundary.
Earlier-stage prompts read only the active stage's criterion and route menu, so
editing Strict cannot change reconstruction of a paid Directional turn.

## Generator

Every Generator prompt opens with the first-principles researcher preamble.
The whiteboard is a model-authored prose belief state (leading hypothesis,
established mechanisms with one-line explanations, ranked open uncertainties);
after every applied oracle answer or channel-level rejection a dedicated
state-update turn folds the buffered priced events into a rewritten whiteboard
and top idea (the draft). The update turn is target-blind and unpriced; every
fact it integrates still traces to a paid Choice. The Generator-visible
context is rendered as a readable narrative (field label, confirmed facts,
whiteboard, top idea, prior draft IDs, priced negative evidence) rather than
compact JSON.

The mandatory preamble is priced traversal of the public research taxonomy —
the primary field walk one edge at a time, then the contribution-category
choice. The Generator builds its finite codebooks from whichever labels have
actually been confirmed. No target text or Judge feedback enters Generator
state.

**Dispatch (route digest).** Before each dispatch the Generator authors one
route digest in a single model call: for every model-backed displayed route it
states concretely what it would ask if that route were opened (the MC axis and
candidate values, the keyword categories in rank order, each correction axis
with its exact current span). The Oracle routes on the digest and pays the
ordinary route Choice; the real downstream Question is generated only after a
route is bought. Digest rows are a plan, never facts. (Earlier versions built
every route's complete downstream Question eagerly; that preview was ~93% of
Generator spend and is gone.)

**Per-stage route menus.** Each stage displays only the routes its own
measured usage justifies, and the fixed mode probabilities renormalize over
the displayed subset:

- **directional**: `mc`, `keyword`, and `submit` once a draft exists;
- **essence**: `mc`, `keyword`, `correct`, `submit`;
- **strict**: `mc`, `keyword`, `audit`, `correct`, `submit`.

`secondary_field`, `explore`, and `differentiate` are displayed at no stage in
the current tree (each stage's `DROPPED_ROUTES` documents the measured usage
behind its menu). Selecting a recovery route while a draft exists records that
draft as insufficient (public negative evidence); `correct` does not, because
a repair is not a claim that the draft is wrong.

**MC.** One Generator-authored atomic clarification: 8 concrete values on one
unresolved axis plus explicit all-incorrect and different-axis controls. A
concrete selection adds exactly the deterministic fact `axis: value`; it does
not accept a whole draft.

**Keyword.** The historical v0.4 keyword machinery with a category layer in
front. The keyword preview is a paid 16-way slate of open-ended semantic
CATEGORIES of the missing phrase; buying one is an ordinary priced Choice
recorded as a confirmed fact, and only then is the 64-phrase slate generated —
so the phrase slate, the reveal price, and every completion slate below it are
conditioned on the category through the ordinary fact ledger. The phrase slate
exposes 64 contribution-neutral candidates with honest relative weights plus
one paid entry into the two-letter prefix reveal. The 676-way reveal is priced
from the distribution implied by the displayed candidates themselves (declared
`prob_hint` gate times half-uniform-blended first/second-letter masses with
the v0.4 letter floor), so no price ever depends on the hidden target. After
every paid reveal or extension the Generator authors one 16-way completion
slate for the exact prefix, with its declared `prob_none` pricing the
extend-by-one-character exit; the extension characters come from the
frequency-weighted 38-way alphabet, and every character slate carries the
accept-the-exact-prefix exit. A missed guess slate is priced negative evidence
for that prefix only. When the arena declares ordinary time travel enabled
there is no priced abandon row — a zero-cost checkout to the dispatch is the
exit; without time travel the priced abandon row returns. Selecting a phrase
adds only the public phrase as one DIRECT fact. The prefix is bounded at 24
characters.

**Correct** (essence, strict). A two-step semantic correction channel over an
existing claim. The target-blind Generator first authors 16 complete
`before_claim` propositions already expressed or entailed by the draft; a
claim may summarize wording spread across clauses and need not quote an exact
sentence. After the Oracle selects one, the Generator authors 16 full patches:
`replace`, `refine`, or `delete`, each with the complete resulting draft.
Replace/refine carries a complete non-empty `after_claim` that becomes the
canonical paid fact; delete carries no after-claim. `insert` is deliberately
absent: correction cannot add an independent missing claim. Runtime keeps the
rewrite localized to one region and rejects malformed mode/claim pairs,
duplicate or unchanged drafts, folding their probability mass into retry
without renormalizing survivors. Rejecting either finite slate makes `correct`
ineligible for the unchanged draft.

**Audit** (strict). Whole-claim revision slates that prune or repair
overclaims under the stage's faithfulness-first instructions, with enforced
slate-diversity axes.

All slates emit exact protocol Decimal strings summing to 1. Retiring the sole
active fact deterministically clears the draft and returns to broad search: an
empty fact ledger is a valid search state, not a participant failure.

Rejected Directional slates retain compact semantic region fingerprints rather
than every long draft sentence; selected drafts receive stable
Generator-local IDs when a later priced dispatch declares them insufficient.
The IDs, drafts, and route labels all arise from public paid Choices.

`StageReady.handoff` privately records the current draft, whiteboard, active
and retired facts, and fact provenance. Stage switching itself performs no
model or random call and adds no information bits.

## Oracle

Every Oracle turn is one ordinary structured model call carrying the complete
context: `MATCH_INIT` verbatim (the private gold target, the active stage's
exact Judge criterion, the arena rules, and a purely descriptive
`MODULAR_DISPATCH_NOTES` block), the Oracle's own last `state_summary`, and
the current event. Nothing about the Oracle's context is decided by an
auto-compacting external conversation — earlier versions ran a persistent CLI
session, and measurement showed the CLI compacted it unpredictably. Its policy
keeps truthfulness as a hard constraint, cost-aware arm selection at dispatch,
submit-gating on the exact Judge criterion, and mandatory post-rejection
recovery. When several labels are true, taxonomy, contribution-category, and
later finite-slate choices follow a semantic-bottleneck policy: anchor on the
title's main clause and headline novelty, then choose the label that represents
the main contribution and most tightly constrains its distinctive mechanism,
instead of a cheaper generic application or artifact label. Semantic correction
is deliberately conservative: when the current
interpretation is unsupported and no displayed replacement is target-supported,
the Oracle first chooses a complete deletion, records the relation as unresolved,
and returns to keyword/MC to recover it instead of refining the same guess.

The Oracle takes its own same-stage Judge reading through the injected
`services.judge_evaluate` service, once per distinct draft: an unchanged draft
cannot earn a different verdict. Judge text never reaches the Generator.

`participant.oracle:AgentOracle` is the optional agent-backend variant the
Arena selects when a run passes `--oracle-agent {claude-code,codex,human}`: it
sends the identical per-turn prompt through `services.agent_turn` (one fresh
backend session per turn), so a human can play the Oracle through the
`run_dir/agent-workspace/oracle/human/` request/answer files, or a CLI agent
can play it under the same journaled, replayable transport. Default runs use
the plain `Oracle`.

## StageTransition handling

`StageTransition` messages are Arena-authored and service-free. The Generator
flips its stage, keeps its full ledger, and invokes the destination module's
`generator_on_enter`; the Oracle invokes `oracle_on_enter`, updates its
criterion, and reports the new rung as an ordinary `ARENA_EVENT` on its next
paid turn. The hooks may initialize later-stage keys with `setdefault`, but
must not call an injected service. Both actors then reply with `StageReady`.
Pending slates deliberately survive the transition, so a later-stage checkout
can still select a historical option; the active stage function owns that
backward-compatible payload handling.
