"""Essence -> Strict actor policy."""

JUDGE_CRITERION = (
    "Pass only when the idea recovers the gold setting and main object with the exact "
    "defining mechanism and faithful scope. It must not overclaim or assert anything "
    "false of the paper; secondary configuration details are unnecessary unless defining."
)

STAGE_GOAL = (
    "You are in the STRICT stage, the final rung of a three-stage Judge ladder "
    "(directional -> essence -> strict). Your current draft and fact ledger "
    "already passed the Directional and Essence Judges, so the setting, main "
    "object, and a coarse defining mechanism are established. The Strict Judge "
    "passes only when the draft recovers the setting and main object with the "
    "EXACT defining mechanism and faithful scope: it must not overclaim or "
    "assert anything false of the paper, while secondary configuration details "
    "remain unnecessary unless defining. Produce a faithful and sufficiently "
    "detailed reconstruction. First remove, weaken, or correct claims not "
    "entailed by the target; only then add the smallest missing defining detail "
    "needed for Strict recognition."
)

# v1.14: audit comes back to Strict. v1.12 dropped it saying its job was folded
# into correct, and a dLLM Strict run measured what that cost: correct spent
# 37.47 bits over six rounds deleting the same overclaims one dLLM Strict pass
# had removed for 2.74 bits in a single audit pick -- 77 of 146 words across
# eight separate hunks. That is not correct being used badly. correct repairs
# one localized semantic claim by construction, and an overclaim can be a framing spread
# across a draft, so no number of correct rounds expresses one audit revision.
# The two channels do different jobs: audit rewrites, correct amends a value.
#
# Re-offering it costs every route 0.26 bits (correct 2.61 -> 2.87, keyword
# 1.78 -> 2.04, mc 1.69 -> 1.96, submit 2.08 -> 2.34), about 3 bits over a
# Strict run's dispatches, against the ~48 the measured chain wasted.
#
# explore and differentiate stay dropped on their own measured usage: across 33
# recorded Strict dispatches the Oracle chose correct 12 times against explore 3
# and differentiate 1.
DROPPED_ROUTES = ("secondary_field", "explore", "differentiate")

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
Propose exactly 16 mutually distinct SEMANTIC CATEGORIES of the still-missing
exact defining detail. A category names a KIND of fact -- a kind of data or
data domain, a kind of operation, a kind of architectural component, a kind of
augmentation or invariance treatment, a kind of training-schedule property, a
kind of scope or coverage claim -- not a specific value. Word every category
open-endedly: it must admit values beyond those your confirmed facts already
name, never restating your current draft's enumerations as closed lists.
Cover the full space a strict reconstruction of a research paper could still
lack, including categories far from your current best guesses: honest
calibration spreads mass over regions you have NOT yet explored, because the
confirmed facts imply the missing detail is something you have repeatedly
failed to name. Assign each category your honest probability that the single
most valuable missing defining detail belongs to it; the weights need not sum
to one. Rank most-likely first. Also report prob_none: your honest probability
that no displayed category contains it. The same policy applies to every
contribution category.
"""

KEYWORD_PROMPT = """
Propose exactly 64 distinct candidate identity phrases (each 1-3 words) naming
the missing EXACT defining detail of the hidden paper's already-recovered
mechanism: the precise construction, operation, quantity, interface, or scope
term that Strict recognition still lacks. The setting, main object, and coarse
mechanism are established by the confirmed facts; the phrases must supply the
exact missing term at the right specificity - neither a vague area name nor
invented trivia. Assign each a probability reflecting your honest belief it is
the exact missing defining detail; they need not sum to one. Rank most-likely
first. Avoid redundancy: do not rename an already-confirmed proposition, and
avoid duplicate rephrasings; candidates should represent meaningfully different
remaining hypotheses. Also report prob_hint: your honest estimate of how much a
two-letter reveal of the single most important still-missing phrase would help.
The same count, semantics, and coverage rule applies to all contribution
categories.
"""

KEYWORD_GUESS_PROMPT = """
The Oracle paid to reveal that the single most important still-missing
defining-detail phrase begins with the given prefix. Propose exactly 16
distinct complete phrases (each 1-3 words) that begin literally with that
prefix - the spelled-out exact term, not an acronym. Infer its likely scope and
granularity from the confirmed facts. When a confirmed fact states the
semantic category the missing detail belongs to, that category was paid for
and is settled: every completion must name a member of it, and a phrase
outside it only wastes a row. Rank most-likely first. Avoid duplicate
rephrasings; the completions should represent meaningfully different remaining
hypotheses consistent with the prefix. Also report prob_none: your honest
probability that none of your 16 completions is the intended phrase; it prices
the extend-by-one-character outcome, so calibrate it honestly. Apply the same
rule for every contribution category.
"""

EXPLORE_PROMPT = """
The current draft is intended to be faithful already. Propose exactly 12
distinct candidates for one missing Strict-level atomic detail. Each candidate
must contain exactly one independently checkable fact and a resulting draft
that adds only that fact. Prefer defining functional behavior, interface,
training/evaluation relation, or scope only when appropriate to the current
contribution category. Do not pad, enumerate examples, invent metrics, or add a
second proposition. Explicitly label each candidate's object_family (the kind
of detail the fact supplies) and contribution_direction (what role that detail
plays), with at least four substantively different values on each axis across
the slate. Also label each candidate's relation to the selected taxonomy paths
with exactly one of: central, context, broadened, or adjacent; every slate must
contain all four relations. Axis and relation labels describe the candidate;
they are not extra factual claims. The rejected-candidate regions are priced
negative evidence: do not paraphrase or revisit those regions. When at least
two previously selected draft IDs have recovery_mode audit/explore/differentiate,
at least four candidates must cite and synthesize two or more distinct
compatible source_draft_ids, combining only clauses already present in the
cited drafts. Give an honest positive prior weight for the whole candidate and
an honest retry probability that no displayed candidate is faithful.
"""

AUDIT_PROMPT = """
Perform a faithfulness-first claim audit. Propose exactly 8 substantively
different conservative revisions of the current draft. Across the slate cover:
scope or quantifier narrowing; deletion of unsupported examples or evaluation
context; abstraction to a directly supported defining relation; splitting a
compound claim; removal of unsupported causal or comparative language; and the
shortest recognizable draft. Each option must list every active fact ID it
retires. Do not add any new factual claim. Explicitly label object_family and
contribution_direction for each revision, with at least four substantively
different values on each axis across the slate, and include central, context,
broadened, and adjacent taxonomy relations across the slate; the labels
describe the revision and are not extra factual claims. When at least two
previously selected draft IDs have recovery_mode audit/explore/differentiate,
at least four revisions must cite and synthesize two or more compatible
source_draft_ids, combining only clauses already present in the cited drafts;
never cite a draft whose recovery_mode is correct. Give an honest positive
prior weight for each complete revised draft and an honest probability that
every displayed revision is wrong.
"""

DIFFERENTIATE_PROMPT = """
Propose exactly 10 complete Strict alternatives that preserve the exact current
core and add exactly one short exact-defining-detail relation. Return only the
new standalone relation clause in identity_relation; runtime will append it
after the exact current core, so do not restate, edit, weaken, or expand the
core. Each relation must be a single independently checkable clause of at most
24 words, with no second proposition, explanation, example, result, or
coordinated claim. Bare and/or may appear only inside one compound subject,
object, input set, or value set governed by one predicate; never use them to
coordinate predicates or propositions. Do not use internal commas. Omit
sentence-final punctuation. Runtime may remove malformed rows and transfer
their probability mass to retry, so every raw proposal should satisfy the rule.
Name a contribution-neutral facet_family for each relation. Use at least six
substantively different model-authored facet families across the slate, with
the same prompt, count, and schema for every contribution category. Do not
branch on category and do not use paper titles, author names, model names,
dataset names, exact metrics, or domain-specific exemplars. Every candidate
must cite exactly the displayed current source_draft_id and no other draft ID.
Its probability is an honest positive relative weight for the resulting whole
draft as the faithful Strict identity. retry_prob is the honest probability
that none of the ten complete resulting drafts is faithful.
"""

EXPLORE_COUNT = 12
AUDIT_COUNT = 8
EXPLORE_ACTION = "add"
AUDIT_ACTION = "rewrite"
DIFFERENTIATE_COUNT = 10
DIFFERENTIATE_ACTION = "add"


# Actor-policy interface ----------------------------------------------------
# Strict is intentionally free to replace any function below after an Essence
# pass.  The stable actor shell and the earlier stage files remain byte-frozen.

def generator_on_enter(generator, transition) -> None:
    """Initialize new Strict-only keys with setdefault when needed."""


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
    """Initialize new Strict-only Oracle state without a service call."""


def guide_step(guide, value):
    return guide._shared_guide_step(value)


# Legacy Python names remain available for existing submissions.
oracle_on_enter = guide_on_enter
oracle_step = guide_step
