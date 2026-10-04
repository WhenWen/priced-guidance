"""Directional -> Essence actor policy."""

JUDGE_CRITERION = (
    "Pass when the idea conveys the same setting and the recognizable essence of the "
    "gold's main object, including its defining mechanism coarsely. Secondary datasets, "
    "sizes, configurations, and exact parameters may be omitted or approximated."
)

STAGE_GOAL = (
    "You are in the ESSENCE stage, the second rung of a three-stage Judge "
    "ladder (directional -> essence -> strict). Your current draft and fact "
    "ledger already passed the weaker Directional Judge, which required only "
    "the right problem, area, and contribution type with a plausible mechanism. "
    "The Essence Judge is stricter: it passes only when the draft conveys the "
    "same setting AND the recognizable essence of the paper's main object, "
    "including its defining mechanism at least coarsely; secondary datasets, "
    "sizes, configurations, and exact parameters may be omitted or "
    "approximated. The gap to close is therefore the defining mechanism, "
    "relation, or organizing principle that distinguishes the contribution - "
    "recover it while keeping the Directional core faithful."
)

# v1.12: Essence and Strict run the same four routes as each other.
#
# correct comes back to Essence, where it was the only fact-repair channel
# missing, and audit leaves Strict because its job is folded into correct.
# explore and differentiate leave Strict on measured usage: across 33 recorded
# Strict dispatches the Oracle chose correct 12 times against explore 3 and
# differentiate 1. Both of those uses were productive rather than dead ends,
# and the single differentiate was the row that turned a Utonia Strict preview
# from fail to pass, so this trades a real tail for a cheaper menu.
#
# Renormalising over four routes discounts all of them: at Strict correct falls
# 3.38 -> 2.61 bits, keyword 2.55 -> 1.78, submit 2.85 -> 2.08.
DROPPED_ROUTES = ("secondary_field", "explore", "differentiate", "audit")

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
Propose exactly 16 mutually distinct SEMANTIC CATEGORIES of the still-missing defining mechanism or its decisive
component. A
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
category your honest probability that the missing mechanism belongs to it; the weights need not sum to
one. Rank most-likely first. Also report prob_none: your honest probability
that no displayed category contains it. The same policy applies to every
contribution category.
"""

KEYWORD_PROMPT = """
Propose exactly 64 distinct candidate identity phrases (each 1-3 words) naming
the missing DEFINING MECHANISM, relation, or organizing principle of the hidden
paper's already-located main contribution. The contribution's setting and broad
direction are established by the confirmed facts; the phrases must name how it
centrally works: the mechanism, construction, training or interaction scheme,
representation, or organizing relation that distinguishes it. Phrase each as a
pure technical term, not a vague area name. Assign each a probability
reflecting your honest belief it names the defining mechanism; they need not
sum to one. Rank most-likely first, and let the set range from the obvious to
the genuinely surprising. Avoid redundancy: do not merely rename an
already-confirmed proposition, and avoid duplicate rephrasings and filler;
candidates should represent meaningfully different remaining hypotheses. Also
report prob_hint: your honest estimate of how much a two-letter reveal of the
single most important still-missing phrase would help - high when none of your
candidates feels likely, low when you are confident one is right. The same
count, semantics, and coverage rule applies to all contribution categories.
"""

KEYWORD_GUESS_PROMPT = """
The Oracle paid to reveal that the single most important still-missing
mechanism-naming phrase begins with the given prefix. Propose exactly 16
distinct complete phrases (each 1-3 words) that begin literally with that
prefix - the spelled-out name of the defining mechanism or its decisive
component, not an acronym. Infer its likely scope and granularity from the
confirmed facts. When a confirmed fact states the semantic category the missing
phrase belongs to, that category was paid for and is settled: every
completion must name a member of it, and a phrase outside it only wastes
a row. Include both nearby refinements and more orthogonal
alternatives when the evidence leaves both plausible. Rank most-likely first.
Avoid duplicate rephrasings; the completions should represent meaningfully
different remaining hypotheses consistent with the prefix. Also report
prob_none: your honest probability that none of your 16 completions is the
intended phrase; it prices the extend-by-one-character outcome, so calibrate
it honestly. Apply the same rule for every contribution category.
"""

EXPLORE_PROMPT = """
Propose exactly 12 distinct candidates for the single most useful missing
Essence-level fact. Each candidate must contain one atomic fact and a concise
resulting draft that incorporates exactly that fact into the current draft.
Focus on a defining mechanism, relation, abstraction, or organizing principle;
do not add examples, metrics, model names, datasets, or a bundle of details.
The atomic fact must be independently removable later. Explicitly label each
candidate's object_family (the kind of mechanism or construct the fact
attributes) and contribution_direction (what role that mechanism plays), with
at least four substantively different values on each axis across the slate,
and do not cluster around one fashionable exemplar. Also label each
candidate's relation to the selected taxonomy paths with exactly one of:
central, context, broadened, or adjacent; every slate must contain all four
relations. Axis and relation labels describe the candidate; they are not extra
factual claims. The rejected-candidate regions are priced negative evidence:
do not paraphrase, specialize, or revisit those regions. Previously
insufficient drafts may contain true clauses, but neither their completeness
nor recognizability is confirmed; when at least two such draft IDs exist, at
least four candidates must cite and synthesize two or more distinct compatible
source_draft_ids, combining only clauses already present in the cited drafts.
Give an honest positive prior weight for the complete candidate and an honest
retry probability for the event that no displayed candidate is faithful.
"""

AUDIT_PROMPT = """
Propose exactly 8 distinct conservative repairs of the current Essence draft.
Each repair may delete or weaken unsupported scope, split a compound claim, or
remove one unsupported active fact. It must preserve every directly supported
core claim and must not introduce replacement factual content or a new mechanism
guess; use the separately priced two-step correct route when one retained axis
needs a different value. Explicitly label object_family and
contribution_direction for each repair, with at least four substantively
different values on each axis across the slate, and include central, context,
broadened, and adjacent taxonomy relations across the slate; the labels
describe the repair and are not extra factual claims. When at least two
previously selected draft IDs have recovery_mode audit/explore/differentiate,
at least four repairs must cite and synthesize two or more compatible
source_draft_ids, combining only clauses already present in the cited drafts;
never cite a draft whose recovery_mode is correct. List the fact IDs retired
by the repair, give an honest positive prior weight for the complete repair,
and give an honest probability that every displayed repair is wrong.
"""

DIFFERENTIATE_PROMPT = """
Propose exactly 10 complete Essence alternatives that preserve the exact
current core and add exactly one short defining-mechanism-bearing relation.
Return only the new standalone relation clause in identity_relation; runtime
will append it after the exact current core, so do not restate, edit, weaken,
or expand the core. Each relation must be a single independently checkable
clause of at most 24 words, with no second proposition, explanation, example,
result, or coordinated claim. Bare and/or may appear only inside one compound
subject, object, input set, or value set governed by one predicate; never use
them to coordinate predicates or propositions. Do not use internal commas.
Omit sentence-final punctuation. Runtime may remove malformed rows and transfer
their probability mass to retry, so every raw proposal should satisfy the rule.
Name a contribution-neutral facet_family for each relation. Use at least six
substantively different model-authored facet families across the slate, with
the same prompt, count, and schema for every contribution category. Do not
branch on category and do not use paper titles, author names, model names,
dataset names, exact metrics, or domain-specific exemplars. Every candidate
must cite exactly the displayed current source_draft_id and no other draft ID.
Its probability is an honest positive relative weight for the resulting whole
draft (preserved core plus this one relation) as the recognizable Essence
identity. retry_prob is the honest probability that none of the ten complete
resulting drafts is faithful.
"""

EXPLORE_COUNT = 12
AUDIT_COUNT = 8
EXPLORE_ACTION = "add"
AUDIT_ACTION = "rewrite"
DIFFERENTIATE_COUNT = 10
DIFFERENTIATE_ACTION = "add"


# Actor-policy interface ----------------------------------------------------
# Editing these functions changes only the Essence module hash.  The shared
# actor shell replays the frozen Directional prefix, applies a service-free
# transition, and begins calling this interface only after Essence activates.

def generator_on_enter(generator, transition) -> None:
    """Initialize new Essence-only keys with setdefault when needed."""


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
    # Checkout can restore a Directional dispatch before the runtime reapplies
    # the Essence transition. Its published slate remains the binding contract;
    # the active stage must not replace it with the current single draft.
    bundle = generator.state.get("pending_dispatch_previews") or {}
    if bundle.get("source_stage") == "directional":
        import copy
        import hashlib
        import json

        from tech_tree_arena import Idea, Submission

        preview = bundle.get("submission_preview")
        preview_hash = hashlib.sha256(json.dumps(
            preview, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
        ).encode("utf-8")).hexdigest()
        if (
            generator.state["stage"] != "essence"
            or not isinstance(preview, dict)
            or len(preview.get("ideas", [])) != 8
            or preview_hash != bundle.get("submission_hash")
            or bundle.get("source_draft")
            != " ".join(str(generator.state["current_draft"] or "").split())
            or bundle.get("fact_ledger_hash") != generator._fact_ledger_hash()
        ):
            raise ValueError("Essence submission requires the exact inherited cached slate")
        return Submission(tuple(
            Idea(row["idea_id"], copy.deepcopy(row["content"]), row["probability"])
            for row in preview["ideas"]
        ))
    return generator._shared_submission()


def oracle_on_enter(oracle, transition) -> None:
    """Initialize new Essence-only Oracle state without a service call."""


def oracle_step(oracle, value):
    return oracle._shared_oracle_step(value)
