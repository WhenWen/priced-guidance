"""Arena-owned semantic judges over the Standard Paper Summary Format.

Given a GOLD paper summary ({setting_and_object,
concrete_detailed_setting, key_findings}) and a candidate generated idea, we score
how much of the paper the idea recovers, as a *nested* hierarchy of judges:

  * Motivation-judge (== Finding-0): does the idea recover the SETTING and MAIN
    OBJECT (section 1), regardless of any findings?
  * Finding-k judge (k = 1..K): does the idea recover the first..k-th findings
    (sections-1 setting AND gold findings 1..k)?

Strictness hierarchy (guaranteed by construction): we evaluate the motivation and
each finding INDEPENDENTLY (one boolean each), then define

    level_pass[0] = motivation_recovered
    level_pass[k] = motivation_recovered AND findings 1..k all recovered

Because level_pass[k] is a conjunction over a strict superset of the conditions in
level_pass[k-1], the set of ideas passing Finding-k is a (weakly) strict subset of
those passing Finding-(k-1). The reported `max_level` is the largest k whose
level passes (= number of leading consecutively-recovered findings, or -1 if even
the motivation fails).

For every judge the candidate need NOT match the gold *concrete detailed setting*
(exact datasets / sizes / configs) -- those secondary specifics matter only when a
particular detail is crucial to the specific finding being judged.

Candidate idea format (the "generation format", mirrors the summary so the judge
can compare component-by-component):

    {"setting_and_object": "<prose: problem + system + object + goal>",
     "findings": ["<claimed finding 1>", "<claimed finding 2>", ...]}

"""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable

from ..contract.messages import Idea
from ..errors import ReplayDivergence, ResourceLimitExceeded
from ..model_defaults import DEFAULT_MAX_OUTPUT_TOKENS
from .base import IdeaVerdict

JUDGE_MAX_TOKENS = DEFAULT_MAX_OUTPUT_TOKENS


def _judge_call(model_call: Callable[..., Any], **kw: Any) -> Any:
    """Call the Arena-injected judge service with bounded retries."""
    import time
    last = None
    for i in range(4):
        try:
            return model_call(**kw)
        except (ReplayDivergence, ResourceLimitExceeded):
            raise
        except Exception as exc:  # noqa: BLE001
            last = exc
            time.sleep(2.0 * (i + 1))
    raise last


def _submit(executor: ThreadPoolExecutor, function: Callable[..., Any], *args: Any):
    return executor.submit(function, *args)

VERDICT_SCHEMA: dict = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "recovered": {"type": "boolean"},
        "reason": {"type": "string", "description": "One sentence justification."},
    },
    "required": ["recovered", "reason"],
}

# --------------------------------------------------------------------------- #
# Formatting gold pieces for the prompts
# --------------------------------------------------------------------------- #


def _fmt_groups(groups: list[dict]) -> str:
    out = []
    for g in groups:
        out.append(f"- {g['aspect']}: " + "; ".join(g["details"]))
    return "\n".join(out)


def _fmt_idea(idea: dict) -> str:
    findings = idea.get("findings", []) or []
    fl = "\n".join(f"  {i}. {f}" for i, f in enumerate(findings, 1)) or "  (none)"
    return (f"Setting & main object: {idea['setting_and_object']}\n"
            f"Claimed findings:\n{fl}")


# --------------------------------------------------------------------------- #
# Component judges
# --------------------------------------------------------------------------- #

_MOTIVATION_DEV = """\
You are a strict but fair research judge. You are given a GOLD setting + main \
object (the crux of a real paper) and a CANDIDATE research idea. Decide whether \
the candidate recovers the SAME setting and, crucially, the SAME MAIN OBJECT -- \
INCLUDING THE DEFINING MECHANISM that makes that object what it is.

RECOVERING THE MAIN OBJECT REQUIRES THE DEFINING MECHANISM, NOT JUST THE DIRECTION. \
The candidate must capture WHAT the central object is AND the core mechanism the \
gold states defines it (the key operation / rule / quantity -- HOW it actually \
works). Being in the same general area, the same family of methods, or pursuing the \
same high-level goal is NOT sufficient. If the candidate substitutes a DIFFERENT \
mechanism, or gets the defining mechanism wrong, it does NOT recover the main object \
-- EVEN IF the problem, the goal, and the broad approach all match. "Same rough \
direction" FAILS. In particular, a candidate does NOT recover the object when it: \
applies the rule at a DIFFERENT SCOPE than the gold (e.g. a single global / aggregate \
quantity vs a per-element / per-group / per-coordinate one, or vice versa); uses a \
DIFFERENT underlying quantity, signal, or operation to achieve the same goal; or \
swaps the gold's mechanism for another mechanism in the same family. Treat the \
defining mechanism, and its scope, as part of the ESSENCE -- not as optional detail.

Judge on ESSENCE/crux, not wording or names. The candidate need NOT mention any \
secondary specifics (exact datasets, model sizes, hyperparameters, configs); their \
absence is never a reason to fail. Require the mechanism the gold ACTUALLY states, at \
the level it states it -- no more, no less: do NOT invent extra sub-details the gold \
omits and then fail the candidate for missing them, but DO fail a candidate whose \
stated mechanism is incompatible with, or merely adjacent to, the gold one. Ignore \
the candidate's specific findings here -- judge only whether the setting and main \
object (with its defining mechanism) match.

FAITHFULNESS IS REQUIRED (do not be lenient), and it is checked against the FULL \
paper (you are also shown ALL of the paper's key findings as CONTEXT). Set \
recovered=true ONLY IF BOTH: (a) the candidate is about the SAME setting and main \
object in essence, AND (b) the candidate asserts NOTHING that is false of or at \
odds with the paper -- its setting, its concrete details, OR ANY of its key \
findings. If the candidate claims the paper studies, benchmarks, or builds \
something it does NOT (e.g. claims fine-tuning is benchmarked when only pretraining \
is), set recovered=false EVEN IF the rest matches. A candidate that is broader \
than, or inconsistent with, the actual setting/object does NOT recover it -- \
partial overlap is not enough. (Do NOT require the candidate to state the findings; \
they are faithfulness context only.)

The NAME or acronym the candidate gives the method/system/benchmark is IRRELEVANT \
-- judge the substance only; never accept or reject based on what it is called.\
"""

_FINDING_DEV = """\
You are a strict but fair research judge. You are given a GOLD finding from a \
real paper (with its supporting evidence and the gold setting for context) and a \
CANDIDATE research idea. Decide whether the candidate RECOVERS this specific \
finding: does the candidate clearly assert the same core claim / takeaway as the \
gold finding?

Judge on ESSENCE, not wording, and not on whether other findings are present. \
The candidate does NOT need to match the gold concrete detailed setting (exact \
datasets, sizes, configs) UNLESS a particular detail is genuinely crucial to the \
MEANING of THIS finding -- in that case the candidate must convey that crucial \
detail.

FAITHFULNESS IS REQUIRED (do not be lenient), and it is checked against the FULL \
paper. You are also shown ALL of the paper's key findings as CONTEXT. Set \
recovered=true ONLY IF BOTH: (a) the candidate clearly conveys THIS finding's \
essential claim, AND (b) the candidate asserts NOTHING that contradicts or is at \
odds with ANY part of the paper -- its setting, its concrete details, OR ANY of \
its other key findings. CRUCIAL: a candidate can convey THIS finding yet still be \
unfaithful by over-claiming in a way another finding refutes (e.g. claiming "no \
new optimizer ever beats the baseline" when another finding says a specific class \
of optimizers DOES beat it) -- that is at odds with the paper, so recovered=false. \
Do NOT, however, REQUIRE the candidate to state the other findings; they are \
context for the faithfulness check only, not recovery targets. A candidate that \
asserts the opposite of this finding, omits its core claim, or contradicts the \
paper does NOT recover it. The NAME or acronym used for any method or construct is \
IRRELEVANT -- judge the substance only.\
"""


def _all_findings_context(gold: dict) -> str:
    fl = "\n".join(f"  {i}. {f['finding']}" for i, f in enumerate(gold["key_findings"], 1))
    return ("FULL PAPER -- ALL KEY FINDINGS (faithfulness context only; the candidate "
            "must not contradict ANY of these, but you do NOT require it to state them):\n"
            f"{fl}")


def _judge_motivation(model_call: Callable[..., Any], gold: dict, idea: dict) -> dict:
    so = gold["setting_and_object"]
    user = (
        f"GOLD SETTING & MAIN OBJECT (category: {so['category']}):\n"
        f"{_fmt_groups(so['groups'])}\n\n"
        f"{_all_findings_context(gold)}\n\n"
        f"CANDIDATE IDEA:\n{_fmt_idea(idea)}\n\n"
        "Does the candidate recover the gold setting and main object?"
    )
    return _judge_call(
        model_call,
        developer=_MOTIVATION_DEV, user=user,
        schema=VERDICT_SCHEMA, schema_name="motivation_verdict",
        max_output_tokens=JUDGE_MAX_TOKENS, reasoning_effort="high",
    )


def _judge_finding(model_call: Callable[..., Any], gold: dict, idea: dict, j: int) -> dict:
    so = gold["setting_and_object"]
    f = gold["key_findings"][j]
    user = (
        f"GOLD SETTING & MAIN OBJECT (category: {so['category']}):\n"
        f"{_fmt_groups(so['groups'])}\n\n"
        "GOLD CONCRETE DETAILED SETTING (require a detail only if crucial to the "
        f"finding below):\n{_fmt_groups(gold['concrete_detailed_setting'])}\n\n"
        f"{_all_findings_context(gold)}\n\n"
        f"GOLD FINDING #{j + 1} TO RECOVER (this is the ONLY recovery target):\n{f['finding']}\n"
        f"Supporting evidence: {f['evidence']}\n\n"
        f"CANDIDATE IDEA:\n{_fmt_idea(idea)}\n\n"
        f"Does the candidate recover the essence of GOLD FINDING #{j + 1} WITHOUT "
        "contradicting any other part of the paper?"
    )
    return _judge_call(
        model_call,
        developer=_FINDING_DEV, user=user,
        schema=VERDICT_SCHEMA, schema_name="finding_verdict",
        max_output_tokens=JUDGE_MAX_TOKENS, reasoning_effort="high",
    )


# --------------------------------------------------------------------------- #
# Shared schema for coarse batch screens
# --------------------------------------------------------------------------- #


def _coarse_schema(n: int, *, with_reasons: bool = False) -> dict:
    item_properties: dict = {"idea_number": {"type": "integer"},
                             "recovered": {"type": "boolean"}}
    required = ["idea_number", "recovered"]
    if with_reasons:
        # One short sentence naming the decisive gap (or match). Without it a
        # coarse rejection reaches the Oracle as a bare FAIL and it probes
        # blind; the reason is private Judge text like any full verdict.
        item_properties["reason"] = {"type": "string", "maxLength": 400}
        required = ["idea_number", "recovered", "reason"]
    return {
        "type": "object", "additionalProperties": False,
        "properties": {
            "verdicts": {
                "type": "array", "minItems": n, "maxItems": n,
                "items": {
                    "type": "object", "additionalProperties": False,
                    "properties": item_properties,
                    "required": required,
                },
            },
        },
        "required": ["verdicts"],
    }
# --------------------------------------------------------------------------- #
# F(m, n) judge: among the TOP-m gold findings, are at least n recovered?
#
# This replaces the strict-prefix climb (F1->F2->...). For a fixed window m of the
# m most-important gold findings, F(m, n) passes iff the idea recovers AT LEAST n of
# them -- in ANY order -- AND the idea contradicts nothing in the paper. The climb
# hill-climbs n = 0..m for that fixed m; we never walk the diagonal F(1,1)->F(2,2).
# --------------------------------------------------------------------------- #

_FAITHFUL_DEV = """\
You are a STRICT faithfulness judge. You are given the FULL paper (its setting + \
main object, its concrete detailed setting, and ALL of its key findings) and a \
CANDIDATE research idea (a setting + a list of claimed findings). Decide one thing: \
is EVERY claim the candidate makes ENTAILED by the paper at the stated STRENGTH and \
SCOPE? Set faithful=true ONLY IF so.

The candidate is UNFAITHFUL (faithful=false) in ANY of these cases:
1. CONTRADICTION: a claim is the opposite of, or incompatible with, the paper's \
setting, concrete details, or ANY key finding (e.g. "no method beats the baseline" \
when a finding says some do; claiming the paper studies/benchmarks/builds something \
it does not). This INCLUDES misstating the MAIN OBJECT'S DEFINING MECHANISM: \
describing it with a different mechanism, a different underlying quantity/operation, \
or a different SCOPE than the paper (e.g. a per-element / per-group / per-coordinate \
rule when the paper's is a single global / aggregate one, or vice versa) is a \
contradiction -- a same-family or "rough direction" restatement that alters the \
defining mechanism is UNFAITHFUL, not a faithful rephrasing.
2. OVER-GENERALIZATION / OVER-SCOPING: a claim asserted more BROADLY or UNIVERSALLY \
than the paper's evidence supports -- words like "always / across all / for every / \
consistently / in general / universally / regardless of scale" when the paper shows \
the effect only under SPECIFIC conditions, regimes, scales, or budgets, or shows it \
WEAKENING / SHRINKING / VARYING. The paper supporting a claim in a NARROW regime does \
NOT license the candidate's broad version. (E.g. "X is fastest across all scales" \
when the paper shows X's advantage decreases with scale and nearly vanishes at the \
largest scale -- UNFAITHFUL, even though the paper does say X is the fastest class.)
3. DROPPED ESSENTIAL QUALIFIER: stating a finding WITHOUT a qualifier that is \
essential to its truth, so the bare claim reads stronger/broader than the true \
qualified one (the qualifier is not optional flavor -- omitting it makes the claim \
overstated).
4. OVER-STRENGTHENING: asserting a larger magnitude, wider applicability, or higher \
certainty than the paper establishes.
5. UNSUPPORTED ADDITION / FABRICATED SPECIFIC: asserting a concrete setting component, \
mechanism, sub-rule, technique, or design detail that the paper does NOT state -- EVEN \
IF it is plausible, consistent with the paper, or a reasonable inference from it. Every \
claim must be ENTAILED by the paper, not merely COMPATIBLE with it. Adding a specific \
the paper never states (e.g. how a sub-case or auxiliary part is handled, an extra \
component/step, or a named technique the paper does not mention) is UNFAITHFUL, because \
"not stated" is not "true". Do NOT fill gaps with plausible detail; if the paper is \
silent on a specific, the candidate must not assert it. EXCEPTION -- a detail that is the \
STANDARD, INTRINSIC realization of a method/construct the paper DOES name is ENTAILED by \
that name and is NOT an unsupported addition (see "What is STILL fine"): naming a method \
implies its well-known defining implementation.

Bar: if the paper supports only a WEAKER or NARROWER version of a claim, the \
candidate's stronger/broader version is UNFAITHFUL -- do NOT give it the benefit of \
the doubt. A claim that is "mostly right but overstated" is UNFAITHFUL. Likewise a claim \
that is "plausible but not stated by the paper" is UNFAITHFUL.

What is STILL fine (do NOT penalize): pure OMISSION of a separate finding (saying \
nothing about a topic is not a contradiction); a claim stated at or BELOW the paper's \
supported strength/scope (a correctly-qualified or appropriately-hedged claim); \
differences in NAME/acronym/wording; faithfully RE-PHRASING or ABSTRACTING a component \
the paper DOES state (restating a stated component in other words, or more generally, is \
fine -- case 5 is only about ADDING a substantive component/mechanism/detail the paper \
never states, not about wording a stated one differently). A faithful abstraction must stay CONSISTENT with the \
stated mechanism and its scope, though -- abstracting is fine, but silently CHANGING the scope (global \
vs per-element) or SUBSTITUTING a different mechanism is case 1, not rephrasing. ALSO STILL fine: \
spelling out the STANDARD / INTRINSIC implementation of a method or construct the paper DOES name -- if \
the paper names a method (by its established name or as a stated mechanism), giving that method's \
well-known DEFINING implementation detail is ENTAILED by the name, NOT a fabricated addition (case 5 \
does not apply). E.g. if the paper states the update is "Muon" / an "orthogonalized-momentum" / \
"Orth(M_t)" update, describing the orthogonalization as "Newton-Schulz" is FAITHFUL, because \
Newton-Schulz is the standard way that named method computes it. Case 5 targets substantive components \
the paper neither states NOR entails through a named/stated mechanism -- not the textbook realization \
of something it does name. Judge on ESSENCE, not \
phrasing -- but treat scope, strength, and unsupported additions as part of the essence.\
"""

_FAITHFUL_SCHEMA: dict = {
    "type": "object", "additionalProperties": False,
    "properties": {
        "faithful": {"type": "boolean"},
        "reason": {"type": "string", "description": "One sentence; if false, name the contradicted part."},
    },
    "required": ["faithful", "reason"],
}


def _judge_faithful(model_call: Callable[..., Any], gold: dict, idea: dict) -> dict:
    so = gold["setting_and_object"]
    user = (
        f"GOLD SETTING & MAIN OBJECT (category: {so['category']}):\n"
        f"{_fmt_groups(so['groups'])}\n\n"
        "GOLD CONCRETE DETAILED SETTING:\n"
        f"{_fmt_groups(gold['concrete_detailed_setting'])}\n\n"
        f"{_all_findings_context(gold)}\n\n"
        f"CANDIDATE IDEA:\n{_fmt_idea(idea)}\n\n"
        "Does the candidate contradict ANY part of the paper? (faithful=true iff it "
        "contradicts nothing.)"
    )
    return _judge_call(
        model_call,
        developer=_FAITHFUL_DEV, user=user,
        schema=_FAITHFUL_SCHEMA, schema_name="faithful_verdict",
        max_output_tokens=JUDGE_MAX_TOKENS, reasoning_effort="high",
    )


def judge_fmn(model_call: Callable[..., Any], gold: dict, idea: dict, m: int, *, need_n: int | None = None,
              workers: int = 8) -> dict:
    """F(m, n) judge. Among the TOP-m gold findings, how many does the idea recover
    (in ANY order), is the setting/motivation recovered, and is the idea globally
    faithful (contradicts nothing in the paper)?

    Returns ``achieved_n`` = number of top-m findings recovered IF the idea is both
    on-topic (motivation) and globally faithful, else 0. F(m, n) passes iff
    ``achieved_n >= n``. (n=0 == F(m,0) == faithful setting, no findings required.)

    SHORT-CIRCUIT (saves gpt-5.5 calls): motivation + faithfulness are judged FIRST
    (2 calls). If either fails the idea recovers nothing creditable, so the m
    per-finding judges are SKIPPED. They are also skipped when ``need_n == 0`` (the
    setting rung needs no findings at all). Pass ``need_n`` from the caller to enable
    the setting-rung skip; leave it None to always score all findings (e.g. for jumps).
    """
    total = len(gold["key_findings"])
    m = max(0, min(m, total))
    # Stage 1 -- gate: motivation + faithfulness (2 calls in parallel).
    with ThreadPoolExecutor(max_workers=2) as ex:
        mot_fut = _submit(ex, _judge_motivation, model_call, gold, idea)
        faith_fut = _submit(ex, _judge_faithful, model_call, gold, idea)
        motivation = mot_fut.result()
        faithful = faith_fut.result()
    mot_ok = bool(motivation["recovered"])
    faith_ok = bool(faithful["faithful"])

    # Stage 2 -- per-finding judges, ONLY if the gate passed AND findings are needed.
    findings: list[dict] = []
    gate_ok = mot_ok and faith_ok
    if gate_ok and (need_n is None or need_n >= 1):
        with ThreadPoolExecutor(max_workers=workers) as ex:
            find_futs = [
                _submit(ex, _judge_finding, model_call, gold, idea, j) for j in range(m)
            ]
            findings = [f.result() for f in find_futs]

    recovered = [bool(f["recovered"]) for f in findings]   # per top-m finding (empty if skipped)
    count = sum(recovered)
    # An idea that is off-topic or unfaithful recovers nothing creditable.
    achieved_n = count if gate_ok else 0
    return {
        "m": m,
        "total_findings": total,
        "motivation": motivation,
        "faithful": faithful,
        "findings": findings,
        "recovered": recovered,       # which of the top-m are recovered
        "count": count,               # raw count among top-m (ignoring gates)
        "motivation_ok": mot_ok,
        "faithful_ok": faith_ok,
        "achieved_n": achieved_n,     # creditable n (gated by motivation & faithful)
    }


_COARSE_FMN_DEV = """\
You are a COARSE research judge doing a fast BATCH PRE-FILTER. You are given the GOLD \
setting + main object, a numbered list of the paper's TOP-m key findings, and the \
FULL paper's findings as faithfulness context, plus several CANDIDATE research \
ideas. A stricter judge will fully confirm whatever you flag, so your job is to be \
PRECISE: flag a candidate ONLY when it would plausibly survive that strict judge, \
and DROP candidates that clearly would not -- every wrong flag wastes an expensive \
confirmation.

For EACH candidate independently, mark recovered=true iff BOTH:
(a) it conveys the SAME setting + main object AND CLEARLY recovers (in ANY order) AT \
LEAST {need_n} of the listed top-m findings -- "recovers" means the candidate \
actually ASSERTS that finding's core claim, not merely names its topic or gestures \
at the area; and
(b) it is FAITHFUL: contradicts nothing in the paper AND does not OVER-CLAIM -- i.e. \
no claim stated more broadly/universally ("all / always / consistently / regardless \
of scale") than the paper supports, no dropped essential qualifier, no over-stated \
magnitude or scope. An over-claimed or over-scoped candidate is a NON-match (mark \
false) even if its topic is right.

Mark recovered=FALSE for a candidate that merely TOUCHES the topic, only PARTIALLY \
overlaps a finding, recovers FEWER than {need_n} findings, OR over-claims -- do NOT \
flag it. Only when genuinely unsure whether a clearly-faithful candidate crosses the \
{need_n} bar should you lean toward flagging (the fine judge decides the borderline). \
Order does NOT matter -- any {need_n} of the m count. Judge on ESSENCE, not wording, \
but treat SCOPE and STRENGTH as part of the essence; secondary specifics (datasets, \
sizes, configs) are not required unless crucial to a counted finding; the \
NAME/acronym a candidate gives a method is IRRELEVANT. One verdict per candidate.\
"""


def coarse_screen_fmn(model_call: Callable[..., Any], gold: dict, ideas: list[dict], m: int, need_n: int,
                      *, reasoning: str = "medium", with_reasons: bool = False,
                      reasons_out: list | None = None) -> list[bool]:
    """ONE chunked call: for each candidate, does it recover the setting AND at least
    ``need_n`` of the TOP-m gold findings (any order) without contradicting the paper?
    Cheap inclusive filter; confirm flagged candidates with judge_fmn. One bool each."""
    if not ideas:
        return []
    total = len(gold["key_findings"])
    m = max(0, min(m, total))
    need_n = max(0, min(need_n, m))
    so = gold["setting_and_object"]
    finds = gold["key_findings"][:m]
    gold_block = (f"GOLD SETTING & MAIN OBJECT (category: {so['category']}):\n"
                  f"{_fmt_groups(so['groups'])}\n\n")
    if finds:
        gold_block += (f"TOP-{m} GOLD FINDINGS (recover ANY {need_n} of these {m}):\n"
                       + "\n".join(f"{i}. {f['finding']}" for i, f in enumerate(finds, 1)))
    else:
        gold_block += "No findings in the window -- screen the setting & main object only."
    gold_block += f"\n\n{_all_findings_context(gold)}"
    cand_block = "\n\n".join(f"CANDIDATE {i}:\n{_fmt_idea(c)}" for i, c in enumerate(ideas, 1))
    user = (f"{gold_block}\n\n=== {len(ideas)} CANDIDATE IDEAS ===\n{cand_block}\n\n"
            f"Return one verdict per candidate (recovered=true iff it recovers the "
            f"setting AND at least {need_n} of the {m} listed findings, faithfully).")
    if with_reasons:
        user += (" For each candidate, also give reason: one short sentence naming the"
                 " decisive missing or mismatched element (or the decisive match).")
    out = _judge_call(model_call, developer=_COARSE_FMN_DEV.format(need_n=need_n),
                      user=user, schema=_coarse_schema(len(ideas), with_reasons=with_reasons),
                      schema_name="coarse_screen_fmn",
                      max_output_tokens=JUDGE_MAX_TOKENS, reasoning_effort=reasoning)
    flags = [False] * len(ideas)
    reasons = [""] * len(ideas)
    for v in out.get("verdicts", []):
        num = int(v.get("idea_number", 0))
        if 1 <= num <= len(ideas):
            flags[num - 1] = bool(v.get("recovered"))
            reasons[num - 1] = str(v.get("reason") or "")
    if reasons_out is not None:
        reasons_out.extend(reasons)
    return flags


# --------------------------------------------------------------------------- #
# Directional judge (lenient complement of the strict motivation judge)
#
# The motivation judge REQUIRES the gold's exact defining mechanism ("same rough
# direction FAILS"). The directional judge is its deliberate opposite: it does NOT
# require the same mechanism and does NOT check faithfulness to the paper's actual
# claims. It asks only two questions:
#   (1) direction_aligned  -- does the candidate pursue the SAME research direction
#       (same problem/goal/area/contribution-type) as the gold, even via a DIFFERENT
#       concrete mechanism?
#   (2) equally_sensible   -- is the candidate, on its own merits, a PEER-QUALITY
#       idea that makes about as much research sense as the gold (concrete, coherent,
#       well-motivated, non-trivial -- not a vague topic, not degenerate, not absurd)?
# aligned == direction_aligned AND equally_sensible.
# --------------------------------------------------------------------------- #

_DIRECTIONAL_DEV = """\
You are a research-TASTE judge. You are given the GOLD setting + main object (the crux \
of a real paper) and a CANDIDATE research idea. This is NOT a recovery or faithfulness \
judge: you do NOT require the candidate to reproduce the gold's exact defining \
mechanism, and you do NOT penalize it for differing from, or not matching, the paper's \
actual claims or results. Judge PURELY the two things below, and set the two booleans \
INDEPENDENTLY.

1. direction_aligned -- does the candidate pursue the SAME RESEARCH DIRECTION as the \
gold? This requires BOTH: (a) it attacks the gold's SPECIFIC core problem / bottleneck -- \
the exact problem named in the gold's problem & motivation, not merely some problem in the \
same broad area; and (b) it embodies the gold's OWN advance -- the specific new \
contribution that distinguishes the gold from the prior work it builds on -- not just the \
pre-existing method, backbone, or family the gold shares with that prior work. A candidate \
that targets the gold's specific problem with a DIFFERENT concrete mechanism is still \
aligned -- a different-but-on-target approach is exactly what this criterion accepts. It is \
NOT aligned if it attacks a different problem, pursues a different goal, sits in a \
different subfield, is a different TYPE of contribution (e.g. a benchmark when the gold is \
a method), OR amounts to nothing more than the established prior method the gold improves \
upon -- i.e. it restates a famous backbone (even the very one the gold itself uses) without \
committing to the gold's distinctive new contribution.

Operational test for (b): identify the prior work the gold extends and the gold's DELTA \
over it. If your justification for "aligned" would be just as true of that prior work \
alone -- if the candidate would describe the bare baseline equally well -- then the \
candidate has NOT captured the gold's direction: set direction_aligned=false. Sitting in \
the same method family as the gold is NECESSARY but NOT SUFFICIENT; the candidate must \
commit to the gold's specific advance, not merely the shared starting point.

2. equally_sensible -- taken on its OWN merits, does the candidate make AS MUCH RESEARCH \
SENSE as the gold idea? It need not BE the gold idea, but it must be a PEER-QUALITY idea \
in that direction: a concrete, coherent, well-motivated, non-trivial contribution that a \
competent researcher would judge about as plausible-to-work and as worth doing as the \
gold. Set equally_sensible=false if the candidate is merely a vague topic or aspiration \
rather than a concrete idea, is internally incoherent or self-contradictory, is a trivial \
or degenerate variant, rests on an implausible or nonsensical premise, or would clearly \
make a substantially weaker or worse-motivated paper than the gold. You are judging the \
idea's INTRINSIC quality and plausibility, NOT whether it reproduces the paper's reported \
findings.

Restating a famous prior method the gold builds on, with nothing of the gold's own advance \
added, is the single most common false-positive -- guard against it above all.

Judge on ESSENCE, not wording; the NAME/acronym a candidate gives anything is IRRELEVANT. \
Secondary specifics (exact datasets, model sizes, hyperparameters, configs) are \
irrelevant and their absence is never a reason to fail either criterion. Give ONE reason \
that, if either boolean is false, names which criterion failed and why.\
"""

_DIRECTIONAL_SCHEMA: dict = {
    "type": "object", "additionalProperties": False,
    "properties": {
        "direction_aligned": {
            "type": "boolean",
            "description": "Same research direction (problem/goal/area/contribution-type) "
                           "as the gold, even if via a different mechanism.",
        },
        "equally_sensible": {
            "type": "boolean",
            "description": "The candidate is a concrete, coherent, peer-quality idea that "
                           "makes about as much research sense as the gold idea.",
        },
        "reason": {"type": "string",
                   "description": "One sentence; if either is false, name which and why."},
    },
    "required": ["direction_aligned", "equally_sensible", "reason"],
}


def judge_directional(model_call: Callable[..., Any], gold: dict, idea: dict) -> dict:
    """Lenient direction+taste judge. Returns the model's two booleans plus a derived
    ``aligned`` = direction_aligned AND equally_sensible. Does NOT require the gold's
    defining mechanism and does NOT check faithfulness to the paper's claims."""
    so = gold["setting_and_object"]
    user = (
        f"GOLD SETTING & MAIN OBJECT (category: {so['category']}):\n"
        f"{_fmt_groups(so['groups'])}\n\n"
        f"CANDIDATE IDEA:\n{_fmt_idea(idea)}\n\n"
        "Does the candidate pursue the same research DIRECTION as the gold, and does it "
        "make as much research SENSE as the gold idea (on its own merits)?"
    )
    out = _judge_call(
        model_call,
        developer=_DIRECTIONAL_DEV, user=user,
        schema=_DIRECTIONAL_SCHEMA, schema_name="directional_verdict",
        max_output_tokens=JUDGE_MAX_TOKENS, reasoning_effort="high",
    )
    out["aligned"] = bool(out["direction_aligned"]) and bool(out["equally_sensible"])
    return out


_COARSE_DIRECTIONAL_DEV = """\
You are a COARSE research-TASTE judge doing a fast BATCH PRE-FILTER. You are given the \
GOLD setting + main object and several CANDIDATE research ideas. This is NOT a recovery \
or faithfulness screen: do NOT require the gold's exact defining mechanism and do NOT \
penalize a candidate for differing from the paper's claims. For EACH candidate \
independently, mark recovered=true iff BOTH: (a) it pursues the SAME RESEARCH DIRECTION \
as the gold -- same core problem/goal, same broad area, same contribution TYPE -- even \
via a DIFFERENT concrete mechanism; AND (b) it is EQUALLY SENSIBLE -- a concrete, \
coherent, well-motivated, non-trivial idea that makes about as much research sense as \
the gold (NOT a vague topic/aspiration, NOT incoherent, NOT degenerate, NOT obviously \
weaker-motivated than the gold). A stricter judge confirms whatever you flag, so be \
PRECISE: drop candidates in a different problem/area/contribution-type, and drop vague or \
under-specified ones. Judge on ESSENCE, not wording; the NAME/acronym is irrelevant; \
secondary specifics (datasets, sizes, configs) are irrelevant. One verdict per candidate.\
"""


_ESSENCE_DEV = """\
You are a research judge with MIDDLE strictness -- tighter than a "same direction" judge, looser \
than a strict faithfulness judge. You are given the GOLD setting + main object of a real paper and \
a CANDIDATE research idea. Decide whether the candidate recovers the ESSENCE of the gold's setting \
and main object -- its defining idea/mechanism stated at the level that makes the contribution \
recognizably THIS one -- while ALLOWING it to omit or differ on secondary concrete details.

REQUIRE (the essence): the candidate must capture the gold's essential mechanism/idea -- what the \
object essentially does, the core operation, rule, or quantity that defines it. Pursuing the same \
goal or area with a DIFFERENT essential mechanism is NOT enough: if the candidate substitutes a \
different defining mechanism, gets the core wrong, or merely gestures at the topic, it does NOT \
recover the essence. (This is where it is stricter than a directional judge.)

LET GO (secondary concrete details): exact datasets, model sizes, hyperparameters, configurations, \
numerical constants, the precise formula form, exact protocol parameters, or how a secondary \
sub-case is handled. Their absence, approximation, or minor divergence is NOT a reason to fail. A \
candidate that states the defining mechanism a bit more coarsely than the gold, or omits a fine \
qualifier, still recovers the essence. Do NOT fail for missing specifics the way a strict judge \
would. (This is where it is looser than the strict judge.)

Judge on ESSENCE, not wording; the NAME/acronym is irrelevant. Set recovered=true iff the candidate \
conveys the same setting AND the essential main object. Still set recovered=false if the candidate \
CONTRADICTS the paper's core or its setting, or replaces/mis-states the defining mechanism -- a \
wrong core is a miss even when the details are vague.\
"""


def judge_essence(model_call: Callable[..., Any], gold: dict, idea: dict) -> dict:
    """Middle-strictness judge: requires the ESSENCE of the setting + main object (the defining
    mechanism/idea), but tolerates missing/approximate secondary concrete details. Returns
    {recovered, reason}. Stricter than judge_directional, looser than judge_fmn."""
    so = gold["setting_and_object"]
    user = (
        f"GOLD SETTING & MAIN OBJECT (category: {so['category']}):\n"
        f"{_fmt_groups(so['groups'])}\n\n"
        f"CANDIDATE IDEA:\n{_fmt_idea(idea)}\n\n"
        "Does the candidate recover the ESSENCE of the gold setting and main object (its defining "
        "mechanism), allowing missing or approximate secondary concrete details?"
    )
    return _judge_call(
        model_call,
        developer=_ESSENCE_DEV, user=user,
        schema=VERDICT_SCHEMA, schema_name="essence_verdict",
        max_output_tokens=JUDGE_MAX_TOKENS, reasoning_effort="high",
    )


_COARSE_ESSENCE_DEV = """\
You are a COARSE research judge doing a fast BATCH PRE-FILTER at MIDDLE strictness. You are given the \
GOLD setting + main object and several CANDIDATE ideas. For EACH candidate independently, mark \
recovered=true iff it conveys the SAME setting and the ESSENCE of the main object -- its defining \
mechanism/idea -- judged on essence, not wording. It is STRICTER than a directional screen (a \
different or wrong defining mechanism, or a mere topic gesture, is a NON-match) but LOOSER than a \
strict screen (missing or approximate secondary concrete details -- datasets, sizes, configs, exact \
formula, protocol parameters -- are fine and never a reason to drop). A stricter judge confirms \
whatever you flag, so flag plausible essence matches and drop candidates that miss or contradict the \
defining mechanism or the setting. One verdict per candidate.\
"""


def coarse_essence(model_call: Callable[..., Any], gold: dict, ideas: list[dict], *, reasoning: str = "medium", with_reasons: bool = False, reasons_out: list | None = None) -> list[bool]:
    """ONE chunked call: for each candidate, does it recover the ESSENCE of the gold setting+object
    (defining mechanism), tolerating missing concrete details? Cheap filter; confirm with
    judge_essence. One bool per candidate."""
    if not ideas:
        return []
    so = gold["setting_and_object"]
    gold_block = (f"GOLD SETTING & MAIN OBJECT (category: {so['category']}):\n"
                  f"{_fmt_groups(so['groups'])}")
    cand_block = "\n\n".join(f"CANDIDATE {i}:\n{_fmt_idea(c)}" for i, c in enumerate(ideas, 1))
    user = (f"{gold_block}\n\n=== {len(ideas)} CANDIDATE IDEAS ===\n{cand_block}\n\n"
            "Return one verdict per candidate (recovered=true iff it recovers the essence of the "
            "setting + main object, tolerating missing concrete details).")
    if with_reasons:
        user += (" For each candidate, also give reason: one short sentence naming the"
                 " decisive missing or mismatched element (or the decisive match).")
    out = _judge_call(model_call, developer=_COARSE_ESSENCE_DEV, user=user,
                      schema=_coarse_schema(len(ideas), with_reasons=with_reasons), schema_name="coarse_essence",
                      max_output_tokens=JUDGE_MAX_TOKENS, reasoning_effort=reasoning)
    flags = [False] * len(ideas)
    reasons = [""] * len(ideas)
    for v in out.get("verdicts", []):
        num = int(v.get("idea_number", 0))
        if 1 <= num <= len(ideas):
            flags[num - 1] = bool(v.get("recovered"))
            reasons[num - 1] = str(v.get("reason") or "")
    if reasons_out is not None:
        reasons_out.extend(reasons)
    return flags


def coarse_directional(model_call: Callable[..., Any], gold: dict, ideas: list[dict], *, reasoning: str = "medium", with_reasons: bool = False, reasons_out: list | None = None) -> list[bool]:
    """ONE chunked call: for each candidate, is it direction-aligned AND equally sensible
    vs the gold (lenient; no mechanism/faithfulness requirement)? Cheap inclusive filter;
    confirm flagged candidates with judge_directional. Returns one bool per candidate."""
    if not ideas:
        return []
    so = gold["setting_and_object"]
    gold_block = (f"GOLD SETTING & MAIN OBJECT (category: {so['category']}):\n"
                  f"{_fmt_groups(so['groups'])}")
    cand_block = "\n\n".join(f"CANDIDATE {i}:\n{_fmt_idea(c)}" for i, c in enumerate(ideas, 1))
    user = (f"{gold_block}\n\n=== {len(ideas)} CANDIDATE IDEAS ===\n{cand_block}\n\n"
            "Return one verdict per candidate (recovered=true iff it is direction-aligned "
            "AND equally sensible vs the gold).")
    if with_reasons:
        user += (" For each candidate, also give reason: one short sentence naming the"
                 " decisive missing or mismatched element (or the decisive match).")
    out = _judge_call(model_call, developer=_COARSE_DIRECTIONAL_DEV, user=user,
                      schema=_coarse_schema(len(ideas), with_reasons=with_reasons), schema_name="coarse_directional",
                      max_output_tokens=JUDGE_MAX_TOKENS, reasoning_effort=reasoning)
    flags = [False] * len(ideas)
    reasons = [""] * len(ideas)
    for v in out.get("verdicts", []):
        num = int(v.get("idea_number", 0))
        if 1 <= num <= len(ideas):
            flags[num - 1] = bool(v.get("recovered"))
            reasons[num - 1] = str(v.get("reason") or "")
    if reasons_out is not None:
        reasons_out.extend(reasons)
    return flags


# --------------------------------------------------------------------------- #
# Arena judge adapter
# --------------------------------------------------------------------------- #


def _gold(target: Any) -> dict[str, Any]:
    if isinstance(target, dict) and isinstance(target.get("summary"), dict):
        return target["summary"]
    if not isinstance(target, dict):
        raise TypeError("research target must be an object")
    return target


class ResearchJudge:
    """Candidate-separable attempt judge for research-idea target packs.

    The judge receives only validated idea content.  Participant-declared
    probabilities stay in the scoring contract and never enter a model prompt.
    """

    def __init__(
        self,
        services: Any,
        mode: str = "fmn",
        *,
        fmn_m: int | None = None,
        fmn_n: int = 0,
        coarse_reasons: bool = False,
    ) -> None:
        if mode not in {"directional", "essence", "fmn"}:
            raise ValueError("unknown research judge mode")
        if fmn_m is not None and fmn_m < 0:
            raise ValueError("fmn_m must be non-negative")
        if fmn_n < 0:
            raise ValueError("fmn_n must be non-negative")
        if fmn_m is not None and fmn_n > fmn_m:
            raise ValueError("fmn_n cannot exceed fmn_m")
        self.services = services
        self.mode = mode
        self.fmn_m = fmn_m
        self.fmn_n = fmn_n
        self.coarse_reasons = bool(coarse_reasons)

    def evaluate(self, target: Any, ideas: tuple[Idea, ...]) -> tuple[IdeaVerdict, ...]:
        gold = _gold(target)
        verdicts: list[IdeaVerdict] = []
        model_call = self.services.structured_model
        for idea in ideas:
            if isinstance(idea.content, dict):
                source = {
                    "setting_and_object": str(
                        idea.content.get("setting_and_object") or idea.content
                    ),
                    "findings": list(idea.content.get("findings") or []),
                }
            else:
                source = {"setting_and_object": str(idea.content), "findings": []}

            coarse_reason_rows: list = []
            if self.mode == "directional":
                flagged = bool(coarse_directional(
                    model_call, gold, [source],
                    with_reasons=self.coarse_reasons,
                    reasons_out=coarse_reason_rows,
                )[0])
            elif self.mode == "essence":
                flagged = bool(coarse_essence(
                    model_call, gold, [source],
                    with_reasons=self.coarse_reasons,
                    reasons_out=coarse_reason_rows,
                )[0])
            else:
                m = (
                    self.fmn_m
                    if self.fmn_m is not None
                    else len(gold.get("key_findings", []))
                )
                flagged = bool(
                    coarse_screen_fmn(
                        model_call,
                        gold,
                        [source],
                        m=m,
                        need_n=self.fmn_n,
                        with_reasons=self.coarse_reasons,
                        reasons_out=coarse_reason_rows,
                    )[0]
                )
            if not flagged:
                reason = "coarse screen rejected"
                if coarse_reason_rows and coarse_reason_rows[0]:
                    reason = f"coarse screen rejected: {coarse_reason_rows[0]}"
                verdicts.append(IdeaVerdict(idea.idea_id, False, reason))
                continue

            if self.mode == "directional":
                value = judge_directional(model_call, gold, source)
                passed = bool(value.get("aligned"))
            elif self.mode == "essence":
                value = judge_essence(model_call, gold, source)
                passed = bool(value.get("recovered"))
            else:
                value = judge_fmn(
                    model_call,
                    gold,
                    source,
                    m,
                    need_n=self.fmn_n,
                )
                passed = bool(
                    value.get("motivation_ok")
                    and value.get("faithful_ok")
                    and int(value.get("achieved_n") or 0) >= self.fmn_n
                )
            verdicts.append(IdeaVerdict(
                idea.idea_id,
                passed,
                json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")),
            ))
        return tuple(verdicts)
