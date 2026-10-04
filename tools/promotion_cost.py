"""Promotion cost of a Directional-to-Essence run (accounting layer promotion-cost-v1).

The arena prices the answers on a run's final active path. A promoted run's
generator also switches from the Directional to the Essence stage, and the
guide chooses where: after the promotion, a checkout may restore an earlier
Directional question, and the engine re-applies the switch there. The replay
generator of Appendix A prices this choice:

* At every Directional-stage question it switches without an answer with
  probability FALLBACK_EPSILON; otherwise it continues.
* Selecting Submit at a Directional meta question promotes the run. The
  generator then asks the same meta question again, with the same
  probabilities, in the Essence stage. Selecting Submit there sends the
  prepared slate to the Essence judge.

The extra bits relative to the repriced arena cost are

* reused slate (no Essence-stage answer on the path): the Submit selected
  again in the re-asked question, -log2 p_S - log2 pi(1);
* switch at the promoting Submit question: the promoting Submit, which the
  checkout removed from the path, -log2 p_S - log2 pi(j_S);
* switch at any other question: the fallback switch, -log2 FALLBACK_EPSILON;

plus -log2(1 - FALLBACK_EPSILON) for every Directional-stage question passed
without switching. The Directional judge acts only as a private guide preview
and adds nothing.
"""

from __future__ import annotations

import math
from collections import Counter
from typing import Any, Iterable

VERSION = "promotion-cost-v1"
FALLBACK_EPSILON = 0.01
OCCURRENCE_EPSILON = 0.05


def occurrence_bits(j: int) -> float:
    """Code length of occurrence index j under the occurrence prior (Eq. fixed-occurrence-prior)."""
    if j == 1:
        return -math.log2(1 - OCCURRENCE_EPSILON)
    return -math.log2(OCCURRENCE_EPSILON / (j * (j - 1)))


def promotion_bits(events: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Extra promotion bits for one verified schema-4 event chain.

    Returns case "not_promoted" with zero bits for runs without a Generator
    Directional-to-Essence transition, and case "no_essence_pass" when no
    Essence submission passed (the run's cost is infinite either way).
    """
    events = list(events)
    t0 = next((i for i, e in enumerate(events)
               if e.get("kind") == "stage_transition" and e.get("role") == "generator"
               and e.get("from_stage") == "directional" and e.get("to_stage") == "essence"),
              None)
    if t0 is None:
        return {"version": VERSION, "case": "not_promoted", "bits": 0.0}
    fork = max(i for i in range(t0) if events[i].get("kind") == "submission")
    fork_q = events[fork]["source_question_id"]

    options: Counter[tuple[str, str]] = Counter()
    snapshots: dict[str, tuple] = {}
    active: tuple = ()
    final = None
    promoting_submit = None
    for i, e in enumerate(events):
        kind = e.get("kind")
        if kind == "question":
            snapshots[e["question_id"]] = active
        elif kind == "choice_cost":
            key = (e["question_id"], e["option_id"])
            options[key] += 1
            choice = {"event_index": i, "question_id": key[0], "option_id": key[1],
                      "probability": float(e["probability"]), "option_index": options[key]}
            active = (*active, choice)
            if i < t0 and key[0] == fork_q:
                promoting_submit = choice
        elif kind == "checkout":
            active = snapshots[e["target_question_id"]]
        elif kind == "submission_judged" and e.get("status") == "pass" and i > t0:
            final = (active, e.get("source_question_id"))
    if final is None:
        return {"version": VERSION, "case": "no_essence_pass", "bits": 0.0}
    if promoting_submit is None or promoting_submit["option_id"] != "mode-submit":
        raise ValueError("promotion is not preceded by a Directional Submit at its source question")

    path, final_q = final
    directional = [c for c in path if c["event_index"] < t0]
    essence = [c for c in path if c["event_index"] > t0]
    stay = -math.log2(1 - FALLBACK_EPSILON)
    submit_p = promoting_submit["probability"]
    if not essence:
        if final_q != fork_q or directional[-1]["question_id"] != fork_q:
            raise ValueError("reused-slate path does not end at the promoting Submit")
        case = "reused_slate"
        stays = len(directional)
        switch = -math.log2(submit_p) + occurrence_bits(1)
    elif essence[0]["question_id"] == fork_q:
        case = "switch_at_submit"
        stays = len(directional) + 1
        switch = -math.log2(submit_p) + occurrence_bits(promoting_submit["option_index"])
    else:
        case = "switch_elsewhere"
        stays = len(directional)
        switch = -math.log2(FALLBACK_EPSILON)
    return {
        "version": VERSION, "case": case, "bits": stays * stay + switch,
        "directional_questions": stays, "switch_bits": switch,
        "submit_probability": submit_p,
        "submit_occurrence_index": promoting_submit["option_index"],
        "fallback_epsilon": FALLBACK_EPSILON,
    }
