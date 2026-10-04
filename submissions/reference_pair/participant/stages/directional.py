"""Zero information -> Directional actor policy."""

JUDGE_CRITERION = (
    "Pass when the idea is in the same research direction as the gold: same core problem "
    "and goal, area, and contribution type, with a concrete sensible peer-quality "
    "mechanism. It need not recover the gold's exact defining mechanism."
)

STAGE_GOAL = (
    "You are in the DIRECTIONAL stage, the first rung of a three-stage Judge "
    "ladder (directional -> essence -> strict). The Directional Judge passes "
    "when the draft is in the same research direction as the hidden paper: "
    "same core problem and goal, area, and contribution type, with a concrete "
    "sensible peer-quality mechanism; it does not require the exact defining "
    "mechanism yet. Later rungs will demand the recognizable defining mechanism "
    "(essence) and then exact-mechanism faithfulness (strict), so prefer facts "
    "that will remain useful there. Find a short, recognizable statement of the "
    "paper's main contribution. Prefer the correct contribution category, main "
    "object, and broad direction; do not invent implementation details or "
    "quantitative results."
)

MC_PROMPT = """
Ask exactly one bare-object question about the highest-value unresolved finite
axis in the Generator-visible state. Produce exactly 8 concrete values at one
semantic level. Values must be mutually exclusive enough that selecting one is
an atomic clarification, not acceptance of a complete draft. Cover distant
plausible values rather than synonyms or cosmetic variants. Do not ask for a
paper title, author, exact metric, dataset name, or trivia already implied by a
confirmed field/category/fact. Runtime derives the public fact deterministically
as `axis: value`; there is no second model-authored fact, label, explanation,
mechanism, consequence, evaluation result, or second claim. Give honest positive
relative weights plus separate positive masses
for all concrete values being wrong and for asking a fundamentally different
axis. Use the identical policy and schema for every contribution category.
"""

KEYWORD_CATEGORY_PROMPT = """
Before proposing phrases, map the space of what could still be missing.
Propose exactly 16 mutually distinct SEMANTIC CATEGORIES of the still-missing central component of the hidden paper. A
category names a KIND of fact -- a kind of data or data domain, a kind of
operation, a kind of architectural component, a kind of augmentation or
invariance treatment, a kind of training-schedule property, a kind of scope or
coverage claim -- not a specific value. Word every category open-endedly: it
must admit values beyond those your confirmed facts already name, never
restating your current draft's enumerations as closed lists. Cover the full
space a reconstruction of a research paper could still lack, including
categories far from your current best guesses: honest calibration spreads mass
over regions you have NOT yet explored, because the confirmed facts imply what
is missing is something you have repeatedly failed to name. Assign each
category your honest probability that the missing central component belongs to it; the weights need not sum to
one. Rank most-likely first. Also report prob_none: your honest probability
that no displayed category contains it. The same policy applies to every
contribution category.
"""

KEYWORD_PROMPT = """
Propose exactly 64 distinct candidate identity phrases (each 1-3 words) naming
the missing central component of the hidden paper. The missing component may be
a local refinement of a known idea, a new mechanism, a recombination, an
analysis concept, or an artifact. Infer its likely scope and granularity from
the confirmed facts instead of assuming it must be incremental or surprising.
Include both nearby refinements and more orthogonal alternatives when the
evidence leaves both plausible. Phrase each as a pure technical term, not a
vague area name. Assign each a probability reflecting your honest belief it is
the paper's central contribution; they need not sum to one. Rank most-likely
first, and let the set range from the obvious to the genuinely surprising.
Avoid redundancy: for each candidate, ask whether confirming it would earn new
information beyond what the confirmed facts already imply. Do not merely rename
an already-confirmed proposition. You have only 64 slots, so avoid duplicate
rephrasings and filler; candidates should represent meaningfully different
remaining hypotheses, and the probability mass should reflect the actual
posterior implied by the facts. Also report prob_hint: your honest estimate of
how much a two-letter reveal of the single most important still-missing phrase
would help — high when none of your candidates feels likely to be the missing
construct, low when you are confident one of them is right. The same count,
semantics, and coverage rule applies to all contribution categories.
"""

KEYWORD_GUESS_PROMPT = """
The Oracle paid to reveal that the single most important still-missing identity
phrase begins with the given prefix. Propose exactly 16 distinct complete
phrases (each 1-3 words) that begin literally with that prefix — the spelled-out
name of the missing central component, not an acronym. Infer its likely scope
and granularity from the confirmed facts. When a confirmed fact states the semantic category the missing
phrase belongs to, that category was paid for and is settled: every
completion must name a member of it, and a phrase outside it only wastes
a row. Include both nearby refinements and
more orthogonal alternatives when the evidence leaves both plausible. Rank
most-likely first. Avoid duplicate rephrasings; the completions should represent
meaningfully different remaining hypotheses consistent with the prefix. Also
report prob_none: your honest probability that none of your 16 completions is
the intended phrase (the true phrase begins with the prefix but is not among
your list); it prices the extend-by-one-character outcome, so calibrate it
honestly. Apply the same rule for every contribution category.
"""

EXPLORE_PROMPT = """
Propose exactly 12 mutually distinct one-sentence Directional drafts consistent
with the confirmed primary field, optional complementary field, and selected
contribution category. Each draft must identify one main object and one broad
contribution direction. The selected taxonomy paths are priced navigation hints,
not confirmed claims about the central object, scope, or abstraction level.
Partition the remaining semantic space across distant object families and
contribution directions; explicitly label both axes for every candidate, use at
least four substantively different values on each axis, and do not cluster around
one fashionable exemplar. Also label each candidate's relation to the selected
taxonomy paths with exactly one of: central (the paths name its central object),
context (the paths are only substrate or background), broadened (the contribution
is centered at a higher abstraction or wider scope), or adjacent (the paths point
nearby but not at the main object). Every slate must contain all four relations.
Axis and relation labels describe the candidate; they are not extra factual claims.
A candidate must state the central contribution, not merely a true subordinate
fact whose role is incidental to that contribution.
The rejected-candidate regions are priced negative evidence that none of those
whole drafts was faithful: do not paraphrase, specialize, or revisit those
regions. Stay at the contribution-identity level: no paper titles, author names,
model names, exact metrics, dataset names, or speculative implementation details.
The options are a finite codebook, so avoid synonyms and cosmetic paraphrases.
Give an honest positive prior weight for each complete draft and an honest retry
probability for the event that none of the twelve drafts is faithful.
Previously insufficient Directional drafts may contain true clauses, but neither
their completeness nor recognizability is confirmed. A draft followed by a paid
atomic replacement is unsafe as an unchanged component; the corrected result is
a new draft. Merely opening `correct` or rejecting its axis/value slate does not
endorse a replacement. A previously selected
draft followed by `audit`, `explore`, or `differentiate` is supported but
insufficient: when at
least two such draft IDs exist, at least four candidates must cite and synthesize
two or more distinct compatible source_draft_ids. Give those synthesis candidates
different source-ID sets whenever enough combinations exist, and always give
them different semantic axes. Preserve only clauses
already present in the cited drafts; do not infer target-specific details merely
to connect them. The remaining candidates should still explore distant regions.
"""

AUDIT_PROMPT = """
Propose exactly 8 conservative alternatives to the current Directional draft.
Each alternative must change the object/category/scope in a substantively
different way, replace a misidentified contribution direction, or shorten the
draft to only its recognizable object-level core. Explicitly label object family
and contribution direction, with at least four substantively different values on
each axis across the slate. Treat the selected taxonomy paths as fallible navigation
hints and include central, context, broadened, and adjacent relations across the
slate. Do not preserve and locally vary an assumed mechanism,
and do not add new mechanism details. retire_fact_ids may name Generator facts contradicted by
the rewrite. Give an honest positive prior weight for each whole alternative
and an honest probability that every displayed repair is wrong.
When at least two previously selected draft IDs have recovery_mode
audit/explore/differentiate,
at least four alternatives must cite and synthesize two or more compatible
source_draft_ids, using a different source-ID set for each. Never cite a draft
whose recovery_mode is correct. When only one source-ID combination exists,
reuse that pair across substantively different semantic axes. Synthesis may
combine only clauses already
present in the cited Generator-authored drafts; it cannot add guessed details.
"""

DIFFERENTIATE_PROMPT = """
Propose exactly 10 complete Directional alternatives that preserve the exact
current core and add exactly one short identity-bearing relation. Return only the
new standalone relation clause in identity_relation; runtime will append it after
the exact current core, so do not restate, edit, weaken, or expand the core. Each
relation must be a single independently checkable clause of at most 24 words,
with no second proposition, explanation, example, result, or coordinated claim.
Bare and/or may appear only inside one compound subject, object, input set, or
value set governed by one predicate; never use them to coordinate predicates or
propositions. Do not use internal commas.
Omit sentence-final punctuation. Runtime may remove malformed rows and transfer
their probability mass to retry, so every raw proposal should satisfy the rule.
Name a contribution-neutral facet_family for each relation. Use at least six
substantively different model-authored facet families across the slate, with the
same prompt, count, and schema for every contribution category. Do not branch on
category and do not use paper titles, author names, model names, dataset names,
exact metrics, or domain-specific exemplars. Every candidate must cite exactly
the displayed current source_draft_id and no other draft ID. Its probability is
an honest positive relative weight for the resulting whole draft (preserved core
plus this one relation) as the faithful Directional identity. retry_prob is the honest
probability that none of the ten complete resulting drafts is faithful.
"""

EXPLORE_COUNT = 12
AUDIT_COUNT = 8
EXPLORE_ACTION = "replace"
AUDIT_ACTION = "rewrite"
DIFFERENTIATE_COUNT = 10
DIFFERENTIATE_ACTION = "add"

# Directional experiment: the minimal live menu.
#
# Across 29 recorded Directional dispatches (v1.8 three-paper campaign, the
# video-reasoning ladder, and the v1.10 dLLM run) the Oracle chose mc 15 times,
# submit 5, keyword 4, explore 3, differentiate 1 and secondary_field 1, and it
# chose audit and correct exactly zero times -- with a draft present in 22 of
# those 29, so those two were live options it declined, not dead ones.
#
# Dropping the unused routes is not only a cheaper dispatch: the displayed
# probabilities are renormalised over the routes that remain, so every
# surviving route gets cheaper. mc falls 2.56 -> 1.44 bits, keyword 2.64 ->
# 1.52, submit 2.94 -> 1.82.
#
# The bootstrap still works without explore: the first draft is written by the
# state update from the top idea after any fact lands, which is exactly what
# both dLLM Directional runs did -- the draft appeared right after the first mc
# purchase, before any whole-draft channel was ever opened.
DROPPED_ROUTES = ("explore", "differentiate", "audit", "correct", "secondary_field")


# Actor-policy interface ----------------------------------------------------
#
# pair.py is a frozen shell/toolkit.  These functions are the replaceable
# ownership boundary: replay of a Directional prefix executes only this stage's
# functions, while a future Essence/Strict file can wrap or replace any one of
# them without changing the shared or Directional hashes.

def generator_on_enter(generator, transition) -> None:
    """Initialize stage-local state without calling an injected service."""


def generator_step(generator, value):
    return generator._shared_generator_step(value)


def dispatch_question(generator):
    return generator._shared_dispatch_question()


def mc_question(generator):
    return generator._shared_mc_question()


def candidate_question(generator, channel):
    return generator._shared_candidate_question(channel)


def run_state_update(generator) -> None:
    generator._shared_run_state_update()


def correction_axis_question(generator):
    return generator._shared_correction_axis_question()


def correction_replacement_question(generator, payload):
    return generator._shared_correction_replacement_question(payload)


def submission(generator):
    return generator._shared_submission()


def guide_on_enter(guide, transition) -> None:
    """Initialize stage-local Oracle state without calling a service."""


def guide_step(guide, value):
    return guide._shared_guide_step(value)


# Legacy Python names remain available for existing submissions.
oracle_on_enter = guide_on_enter
oracle_step = guide_step
