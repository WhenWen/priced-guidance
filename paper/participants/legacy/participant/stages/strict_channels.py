"""Stage-owned keyword channel implementation (strict).

This file belongs to the strict module group in submission.toml, so it can
change freely without invalidating promotions whose frozen prefix ends
before this stage. The shared dispatcher (participant/pair.py) only
forwards to the functions below; every question, price, validator, and
state transition of the keyword channel lives here. The sibling stages
carry their own independent copies.

The mechanism is the historical v0.4 keyword search -- a 64-candidate direct
phrase slate, a paid two-letter prefix reveal priced from the displayed
candidates' own implied letter distribution, one Generator-authored 16-way
completion slate after every revealed character, and a frequency-weighted
character slate -- with one Strict-stage layer in front of it: the keyword
preview is a paid slate of semantic categories of the missing detail, and
the phrase slate is generated only after a category is bought. The chosen
category is committed as an ordinary confirmed fact, so every step below it
(phrase slate, reveal pricing, completions) is conditioned through the fact
ledger the v0.4 machinery already renders, leaving that machinery untouched.
With ordinary time travel there is no abandon row; a checkout is the exit.
"""

from __future__ import annotations

import string
from typing import Any

from tech_tree_arena import Option, Question

from participant import pair as _pair
from participant.stages import strict as _STAGE_MODULE

GENERATOR_PREAMBLE = _pair.GENERATOR_PREAMBLE
_generator_context = _pair._generator_context
_decimal_distribution = _pair._decimal_distribution
_STAGE = _STAGE_MODULE


def handle(generator, kind: str, payload: dict) -> "Question | None":
    """Route one keyword-channel payload kind.

    Returns the next in-channel Question, or None when the channel has
    concluded (fact recorded or channel-level rejection) and control returns
    to dispatch.
    """

    if kind in {"keyword", "keyword_guess", "keyword_prefix_accept"}:
        apply_keyword(generator, payload)
        return None
    if kind == "keyword_category":
        return category_terms_question(generator, str(payload.get("category") or ""))
    if kind == "keyword_category_retry":
        record_keyword_category_retry(generator)
        return None
    if kind == "keyword_prefix_request":
        return prefix_reveal_question(generator)
    if kind == "keyword_prefix_reveal":
        return guess_question(generator, str(payload.get("prefix") or ""))
    if kind == "keyword_guess_extend":
        prefix = str(payload.get("prefix") or "")
        record_keyword_guess_miss(generator, prefix)
        return hint_character_question(generator, prefix)
    if kind == "keyword_hint_character":
        return guess_question(generator, str(payload.get("prefix") or ""))
    if kind == "keyword_prefix_abandon":
        generator.state["pending_keyword_prefix"] = (
            str(payload.get("prefix") or "") or None
        )
        record_keyword_retry(generator, kind)
        return None
    if kind == "keyword_retry":
        record_keyword_retry(generator, kind)
        return None
    raise ValueError(f"unknown keyword payload kind {kind!r}")


# The keyword channel is a faithful port of the historical v0.4 mechanism:
# a 64-candidate direct slate whose two-letter prefix reveal is priced from
# the distribution implied by the candidates themselves, then one
# Generator-authored 16-way guess slate after every revealed character, a
# frequency-weighted hint alphabet, and an accept-the-exact-prefix exit.
# Constants below reproduce the v0.4 values exactly.
DIRECTIONAL_KEYWORD_COUNT = 64
DIRECTIONAL_KEYWORD_GUESS_COUNT = 16
KEYWORD_ALL_INCORRECT_MASS = 0.05
KEYWORD_PRICE_MIX = 0.5
KEYWORD_LETTER_FLOOR = 0.5 / 26.0
KEYWORD_HINT_ALPHABET = string.ascii_lowercase + string.digits + " -"
KEYWORD_HINT_CHARACTER_WEIGHTS = {
    **{
        character: weight
        for character, weight in zip(
            string.ascii_lowercase,
            (
                8.2, 1.5, 2.8, 4.3, 12.7, 2.2, 2.0, 6.1, 7.0, 0.15, 0.77,
                4.0, 2.4, 6.7, 7.5, 1.9, 0.095, 6.0, 6.3, 9.1, 2.8, 0.98,
                2.4, 0.15, 2.0, 0.074,
            ),
            strict=True,
        )
    },
    **{digit: 0.35 for digit in string.digits},
    " ": 8.0,
    "-": 2.0,
}
# Engineering additions over v0.4, both documented in the README: a length
# bound so a Choice-only Oracle always has a terminating exit, and an abandon
# row standing in for v0.4's checkout-based abandonment.
DIRECTIONAL_KEYWORD_PREFIX_MAX_CHARS = 24
KEYWORD_ABANDON_MASS = 0.5


def _keyword_blended_probability(value: float, count: int) -> float:
    """v0.4 smoothing: half the declared mass, half uniform over the slate."""

    return (1.0 - KEYWORD_PRICE_MIX) * min(max(float(value), 0.0), 1.0) + (
        KEYWORD_PRICE_MIX / max(count, 1)
    )


def _keyword_smoothed_probabilities(raw: list[float]) -> list[float]:
    """v0.4 final smoothing: normalize, then blend half with uniform."""

    values = [max(float(value), 1e-12) for value in raw]
    total = sum(values) or 1.0
    values = [value / total for value in values]
    count = len(values)
    return [
        (1.0 - KEYWORD_PRICE_MIX) * value + KEYWORD_PRICE_MIX / count
        for value in values
    ]


KEYWORD_CATEGORY_COUNT = 16


def _keyword_category_schema(count: int) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "reasoning": {"type": "string"},
            "categories": {
                "type": "array",
                "minItems": count,
                "maxItems": count,
                "items": {
                    "type": "object",
                    "properties": {
                        "category": {"type": "string"},
                        "prob": {"type": "number"},
                    },
                    "required": ["category", "prob"],
                    "additionalProperties": False,
                },
            },
            "prob_none": {"type": "number"},
        },
        "required": ["reasoning", "categories", "prob_none"],
        "additionalProperties": False,
    }


def _sanitize_category(raw_value: Any) -> str | None:
    raw = str(raw_value or "")
    if "\n" in raw or "\r" in raw:
        return None
    value = " ".join(raw.split())
    # A category is one displayed line, so it takes the same generous bound as
    # every other validated row in this pair. The first calibration used 120
    # characters and starved the whole route on the first paper whose
    # Generator wrote categories with illustrative tails: all sixteen rows were
    # dropped, three attempts burned, and the keyword preview collapsed to a
    # retry-only row that still cost three model calls per dispatch.
    if not value or len(value) > 500 or len(value.split()) > 80:
        return None
    return value


def _directional_keyword_schema(count: int) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "reasoning": {"type": "string"},
            "keywords": {
                "type": "array",
                "minItems": count,
                "maxItems": count,
                "items": {
                    "type": "object",
                    "properties": {
                        "term": {"type": "string"},
                        "prob": {"type": "number"},
                    },
                    "required": ["term", "prob"],
                    "additionalProperties": False,
                },
            },
            "prob_hint": {"type": "number"},
        },
        "required": [
            "reasoning",
            "keywords",
            "prob_hint",
        ],
        "additionalProperties": False,
    }


def _directional_keyword_guess_schema(count: int) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "reasoning": {"type": "string"},
            "guesses": {
                "type": "array",
                "minItems": count,
                "maxItems": count,
                "items": {
                    "type": "object",
                    "properties": {
                        "term": {"type": "string"},
                        "prob": {"type": "number"},
                    },
                    "required": ["term", "prob"],
                    "additionalProperties": False,
                },
            },
            "prob_none": {"type": "number"},
        },
        "required": ["reasoning", "guesses", "prob_none"],
        "additionalProperties": False,
    }


def _sanitize_identity_term(raw_value: Any, *, prefix: str = "") -> str | None:
    raw_term = str(raw_value or "")
    term = " ".join(raw_term.split())
    folded = term.casefold()
    if (
        not term
        or len(term) > 500
        or len(term.split()) > 80
        or any(character in raw_term for character in ("\n", "\r"))
        or any(ord(character) < 32 for character in term)
        or not any(character.isalnum() for character in term)
        or any(character in term for character in (".", ",", ";", ":", "?", "!", "。"))
        or (prefix and not folded.startswith(prefix.casefold()))
    ):
        return None
    return term


def _validate_keyword_prefix(prefix: str) -> None:
    if len(prefix) < 2:
        raise ValueError("keyword prefix starts from the paid two-letter reveal")
    if len(prefix) > DIRECTIONAL_KEYWORD_PREFIX_MAX_CHARS:
        raise ValueError("keyword prefix exceeds its bounded length")
    if (
        prefix[0] not in string.ascii_lowercase
        or prefix[1] not in string.ascii_lowercase
    ):
        raise ValueError("keyword prefix must start with two lowercase letters")
    if any(character not in KEYWORD_HINT_ALPHABET for character in prefix):
        raise ValueError("keyword prefix contains an invalid character")
    if "  " in prefix:
        raise ValueError("keyword prefix must not contain consecutive spaces")


def _keyword_prefix_masses(generator) -> dict[str, float]:
    """v0.4 implied-distribution pricing for every two-letter reveal.

    Masses derive from the Generator's own displayed candidate slate: a
    gate on the declared prob_hint, times the blended first-letter mass,
    times the blended conditional second-letter mass.
    """

    candidates = generator.state["pending_keyword_slate"]
    hint = float(generator.state.get("pending_keyword_hint") or 1e-3)
    values = [max(float(row.get("prob") or 0.0), 1e-12) for row in candidates]
    total = sum(values) or 1.0
    values = [value / total for value in values]
    gate = _keyword_blended_probability(hint, 2)
    masses: dict[str, float] = {}
    for first in string.ascii_lowercase:
        first_mass = sum(
            value
            for row, value in zip(candidates, values, strict=True)
            if str(row.get("term", "")).casefold().startswith(first)
        )
        for second in string.ascii_lowercase:
            prefix = first + second
            second_numerator = sum(
                value
                for row, value in zip(candidates, values, strict=True)
                if str(row.get("term", "")).casefold().startswith(prefix)
            )
            second_mass = second_numerator / first_mass if first_mass else 0.0
            masses[prefix] = (
                gate
                * _keyword_blended_probability(
                    max(first_mass, KEYWORD_LETTER_FLOOR), 26
                )
                * _keyword_blended_probability(
                    max(second_mass, KEYWORD_LETTER_FLOOR), 26
                )
            )
    return masses

def _rebind_after_paid_fact(generator) -> None:
    """Refresh this channel's own binding after it commits a paid fact.

    The dispatch binding pins the fact-ledger hash so a cached preview can
    never be replayed against state that moved underneath it. A category
    purchase moves the ledger from inside the bound channel itself, and that
    new ledger is exactly what every later step of this channel must read, so
    the binding is refreshed rather than treated as stale.
    """

    binding = generator.state.get("active_dispatch_binding")
    if isinstance(binding, dict):
        binding["fact_ledger_hash"] = generator._fact_ledger_hash()


def direct_question(generator) -> Question:
    """Keyword preview: the paid category slate only.

    The term list is not built here. Choosing a category is an ordinary
    priced Choice; only then is the phrase slate generated, conditioned on
    the confirmed category through the ordinary fact ledger.
    """

    module = _STAGE
    developer = (
        GENERATOR_PREAMBLE
        + "\n\n"
        + "You are the target-blind Generator in a finite-choice research-idea "
        "recovery game. Before any specific phrase is proposed, the paid "
        "keyword route first asks WHERE the missing exact defining detail "
        "lives: you propose semantic categories, one is selected and paid "
        "for as an ordinary priced Choice, and only then do you enumerate "
        "phrases inside it. Each category is a single line with no "
        "explanation or hidden second claim. prob is your honest belief the "
        "missing detail belongs to that category; the weights need not sum "
        "to one. You do not see gold or Judge text."
        "\n\n"
        + str(module.STAGE_GOAL)
        + "\n\n"
        + str(module.KEYWORD_CATEGORY_PROMPT)
    )
    user = _generator_context(generator.state)
    for _ in range(3):
        output = generator.services.structured_model(
            developer=developer,
            user=user,
            schema=_keyword_category_schema(KEYWORD_CATEGORY_COUNT),
            schema_name=f"{generator.state['stage']}_keyword_categories",
            max_output_tokens=9000,
            reasoning_effort="high",
        )
        rows = output.get("categories") if isinstance(output, dict) else None
        if not isinstance(rows, list) or len(rows) != KEYWORD_CATEGORY_COUNT:
            continue
        sanitized: list[dict[str, Any]] = []
        seen: set[str] = set()
        for raw in rows:
            if not isinstance(raw, dict):
                continue
            category = _sanitize_category(raw.get("category"))
            if category is None:
                continue
            folded = category.casefold()
            if folded in seen:
                continue
            seen.add(folded)
            try:
                weight = float(raw.get("prob"))
            except (TypeError, ValueError):
                continue
            if not (weight > 0.0) or weight != weight or weight == float("inf"):
                continue
            sanitized.append({"term": category, "prob": weight, "is_category": True})
        if len(sanitized) < 8:
            continue
        try:
            none_mass = float(output.get("prob_none"))
        except (TypeError, ValueError):
            continue
        none_mass = min(max(none_mass, 1e-3), 0.5)
        # Dispatch builds previews on clones and restores only the state the
        # shared kernel knows, so the category slate rides the keyword slate
        # itself; is_category marks the rows for the handlers below.
        generator.state["pending_keyword_slate"] = [
            dict(row) for row in sanitized
        ]
        generator.state["pending_keyword_hint"] = None
        smoothed = _keyword_smoothed_probabilities(
            [row["prob"] for row in sanitized] + [none_mass]
        )
        probabilities = _decimal_distribution(smoothed)
        options = [
            Option(
                f"keyword-category-{index}",
                {"kind": "keyword_category", "category": row["term"]},
                probabilities[index - 1],
            )
            for index, row in enumerate(sanitized, 1)
        ]
        options.append(
            Option(
                "keyword-category-retry",
                {"kind": "keyword_category_retry"},
                probabilities[-1],
            )
        )
        return Question(
            "Choose the semantic category containing the single most valuable "
            "missing defining detail; select retry only when no displayed "
            "category contains it. A category confirms category membership "
            "only, never a specific value.",
            tuple(options),
        )
    return generator._retry_only_preview(
        "keyword", " ".join(str(generator.state["current_draft"] or "").split())
    )


def category_terms_question(generator, category: str) -> Question:
    """Record the paid category, then run the unchanged v0.4 phrase slate.

    The category is committed as an ordinary confirmed fact, so it reaches
    every downstream call -- this slate, the two-letter reveal priced from
    it, and each completion slate -- through the generator context the v0.4
    machinery already renders. Nothing below this line differs from the flat
    channel; the conditioning is carried by the ledger, not by new prompts.
    """

    generator._validate_active_dispatch_binding()
    module = _STAGE
    selected = next(
        (
            row
            for row in (generator.state.get("pending_keyword_slate") or [])
            if row.get("is_category") and row.get("term") == category
        ),
        None,
    )
    if selected is None:
        raise ValueError("category does not match the activated finite slate")
    generator.state["pending_keyword_slate"] = []
    binding = generator.state.get("active_dispatch_binding") or {}
    generator._add_fact(
        f"Missing defining detail lies in: {category}",
        "keyword",
        stage=str(binding.get("source_stage") or generator.state["stage"]),
    )
    _rebind_after_paid_fact(generator)
    generator._note_event(
        f"DIRECT confirmed missing-detail category: {category}"
    )
    developer = (
        GENERATOR_PREAMBLE
        + "\n\n"
        + "You are the target-blind Generator in a finite-choice research-idea "
        "recovery game. Propose identity-bearing technical names or phrases "
        "that could name the hidden paper's central contribution at the right "
        "abstraction. The same policy applies symmetrically to methods, theory "
        "or analysis results, benchmarks, datasets, architectures, mechanism "
        "studies, scaling laws, empirical phenomena, evaluation protocols, "
        "software or systems, and applications. A phrase may name an object, "
        "construct, result, artifact, protocol, phenomenon, or application; do "
        "not assume it is a method or mechanism. Each option contains only the "
        "phrase itself, with no explanation or hidden second claim. prob is "
        "your honest belief the phrase is the central contribution; the "
        "weights need not sum to one. prob_hint is your honest estimate of "
        "how much a paid two-letter reveal of the single most important "
        "still-missing phrase would help: high when none of your candidates "
        "feels likely to be the missing construct, low when you are confident "
        "one of them is right. The reveal's price is derived from your own "
        "displayed candidate distribution. You do not see gold or Judge text."
        "\n\n"
        + str(module.STAGE_GOAL)
        + "\n\n"
        + str(module.KEYWORD_PROMPT)
    )
    user = _generator_context(generator.state)
    for _ in range(3):
        output = generator.services.structured_model(
            developer=developer,
            user=user,
            schema=_directional_keyword_schema(DIRECTIONAL_KEYWORD_COUNT),
            schema_name="directional_identity_keywords",
            max_output_tokens=16000,
            reasoning_effort="high",
        )
        rows = output.get("keywords") if isinstance(output, dict) else None
        if not isinstance(rows, list) or len(rows) != DIRECTIONAL_KEYWORD_COUNT:
            continue
        sanitized: list[dict[str, Any]] = []
        seen: set[str] = set()
        for raw in rows:
            if not isinstance(raw, dict):
                continue
            term = _sanitize_identity_term(raw.get("term"))
            if term is None:
                continue
            folded = term.casefold()
            if folded in seen:
                continue
            seen.add(folded)
            try:
                weight = float(raw.get("prob"))
            except (TypeError, ValueError):
                continue
            if not (weight > 0.0) or weight != weight or weight == float("inf"):
                continue
            sanitized.append({"term": term, "prob": weight})
        if len(sanitized) < 8:
            continue
        try:
            hint = float(output.get("prob_hint"))
        except (TypeError, ValueError):
            continue
        hint = min(max(hint, 1e-3), 1.0 - 1e-3)
        generator.state["pending_keyword_slate"] = [dict(row) for row in sanitized]
        generator.state["pending_keyword_hint"] = hint
        direct_smoothed = _keyword_smoothed_probabilities(
            [row["prob"] for row in sanitized] + [KEYWORD_ALL_INCORRECT_MASS]
        )
        gate_total = sum(_keyword_prefix_masses(generator).values())
        probabilities = _decimal_distribution(
            direct_smoothed[:-1] + [gate_total, direct_smoothed[-1]]
        )
        options = [
            Option(
                f"keyword-{index}",
                {"kind": "keyword", "term": row["term"]},
                probabilities[index - 1],
            )
            for index, row in enumerate(sanitized, 1)
        ]
        options.extend(
            (
                Option(
                    "keyword-prefix-request",
                    {"kind": "keyword_prefix_request"},
                    probabilities[-2],
                ),
                Option(
                    "keyword-retry",
                    {"kind": "keyword_retry"},
                    probabilities[-1],
                ),
            )
        )
        return Question(
            "Within the confirmed category, choose a phrase only when it "
            "identifies the central contribution, not when it is merely true "
            "or incidental. Otherwise pay for a two-letter reveal of the most "
            "useful missing phrase, or retry.",
            tuple(options),
        )
    return generator._retry_only_preview(
        "keyword", " ".join(str(generator.state["current_draft"] or "").split())
    )


def record_keyword_category_retry(generator) -> None:
    generator._validate_active_dispatch_binding()
    categories = [
        str(row["term"])
        for row in (generator.state.get("pending_keyword_slate") or [])
        if row.get("is_category")
    ]
    generator.state["rejected_keywords"].append(
        {"phase": "category", "prefix": None, "phrases": categories}
    )
    generator.state["rejected_keywords"] = generator.state["rejected_keywords"][-24:]
    generator._note_event(
        "DIRECT paid keyword outcome: no displayed semantic category contains "
        "the missing defining detail."
    )
    generator.state["pending_keyword_slate"] = []
    generator.state["pending_keyword_hint"] = None
    generator.state["pending_keyword_prefix"] = None
    generator.state["pending_keyword_guesses"] = []
    generator._clear_active_dispatch_binding()


def prefix_reveal_question(generator) -> Question:
    generator._validate_active_dispatch_binding()
    if not generator.state["pending_keyword_slate"]:
        raise ValueError("keyword reveal requires the displayed candidate slate")
    masses = _keyword_prefix_masses(generator)
    prefixes = list(masses)
    probabilities = _decimal_distribution([masses[prefix] for prefix in prefixes])
    return Question(
        "Reveal the first two letters of the single most useful missing "
        "identity phrase. Each reveal is priced from the displayed candidate "
        "slate's own implied letter distribution.",
        tuple(
            Option(
                f"keyword-reveal-{prefix}",
                {"kind": "keyword_prefix_reveal", "prefix": prefix},
                probabilities[index],
            )
            for index, prefix in enumerate(prefixes)
        ),
    )

def guess_question(generator, prefix: str) -> Question:
    generator._validate_active_dispatch_binding()
    prefix = prefix.casefold()
    _validate_keyword_prefix(prefix)
    generator.state["pending_keyword_prefix"] = prefix
    module = _STAGE
    developer = (
        GENERATOR_PREAMBLE
        + "\n\n"
        + "You are the target-blind Generator completing one identity-bearing "
        "technical name after the Oracle paid to reveal that the single most "
        "important still-missing phrase begins with the given prefix. "
        "Generate exactly 16 distinct complete phrases beginning literally "
        "with that prefix. Phrases rejected in earlier guess slates of this "
        "search are priced negative evidence; do not repeat them. Apply the "
        "same contribution-neutral identity semantics used by the keyword "
        "channel. Each option is only a phrase, and prob is its positive "
        "relative weight. prob_none is your honest probability that none of "
        "your 16 completions is the intended phrase; it prices the displayed "
        "extend-by-one-character outcome. You do not see gold or Judge "
        "feedback."
        "\n\n"
        + str(module.KEYWORD_GUESS_PROMPT)
    )
    user = (
        _generator_context(generator.state)
        + f"\n\nRevealed prefix the missing identity phrase starts with: '{prefix}'"
    )
    at_bound = len(prefix) >= DIRECTIONAL_KEYWORD_PREFIX_MAX_CHARS
    omit_tail = at_bound and generator.omit_keyword_abandon
    tail_kind = "keyword_prefix_abandon" if at_bound else "keyword_guess_extend"
    tail_id = "keyword-guess-abandon" if at_bound else "keyword-guess-extend"
    tail_text = (
        "abandon the search"
        if at_bound
        else "extend the exact prefix by one character"
    )
    if omit_tail:
        tail_text = "checkout to an earlier question if none fits"
    for _ in range(3):
        output = generator.services.structured_model(
            developer=developer,
            user=user,
            schema=_directional_keyword_guess_schema(
                DIRECTIONAL_KEYWORD_GUESS_COUNT
            ),
            schema_name="directional_keyword_guesses",
            max_output_tokens=12000,
            reasoning_effort="high",
        )
        rows = output.get("guesses") if isinstance(output, dict) else None
        if not isinstance(rows, list) or len(rows) != DIRECTIONAL_KEYWORD_GUESS_COUNT:
            continue
        sanitized = []
        seen = set()
        for raw in rows:
            if not isinstance(raw, dict):
                continue
            term = _sanitize_identity_term(raw.get("term"), prefix=prefix)
            if term is None:
                continue
            folded = term.casefold()
            if folded in seen:
                continue
            seen.add(folded)
            try:
                weight = float(raw.get("prob"))
            except (TypeError, ValueError):
                continue
            if not (weight > 0.0) or weight != weight or weight == float("inf"):
                continue
            sanitized.append({"term": term, "prob": weight})
        if len(sanitized) < 4:
            continue
        try:
            none_mass = float(output.get("prob_none"))
        except (TypeError, ValueError):
            continue
        none_mass = min(max(none_mass, 1e-3), 1.0 - 1e-3)
        generator.state["pending_keyword_guesses"] = [dict(row) for row in sanitized]
        weights = [row["prob"] for row in sanitized]
        if not omit_tail:
            weights = weights + [none_mass]
        probabilities = _decimal_distribution(
            _keyword_smoothed_probabilities(weights)
        )
        options = [
            Option(
                f"keyword-guess-{index}",
                {"kind": "keyword_guess", "term": row["term"]},
                probabilities[index - 1],
            )
            for index, row in enumerate(sanitized, 1)
        ]
        if not omit_tail:
            options.append(
                Option(tail_id, {"kind": tail_kind, "prefix": prefix}, probabilities[-1])
            )
        return Question(
            f"Choose the missing identity phrase beginning with {prefix!r}, "
            f"or {tail_text}.",
            tuple(options),
        )
    # Fail closed: an unusable guess response abandons the search and
    # returns to dispatch as an ordinary channel rejection.
    return Question(
        f"No valid guess slate survived for prefix {prefix!r}; abandon the "
        "prefix search and return to dispatch.",
        (
            Option(
                "keyword-guess-abandon",
                {"kind": "keyword_prefix_abandon", "prefix": prefix},
                "1",
            ),
        ),
    )

def hint_character_question(generator, prefix: str) -> Question:
    generator._validate_active_dispatch_binding()
    prefix = prefix.casefold()
    _validate_keyword_prefix(prefix)
    if len(prefix) >= DIRECTIONAL_KEYWORD_PREFIX_MAX_CHARS:
        raise ValueError("keyword prefix reached its bound; no further character")
    rows: list[tuple[dict[str, Any], float]] = [
        (
            {"kind": "keyword_hint_character", "prefix": prefix + character},
            KEYWORD_HINT_CHARACTER_WEIGHTS[character],
        )
        for character in KEYWORD_HINT_ALPHABET
        if not (character == " " and prefix.endswith(" "))
    ]
    rows.append(({"kind": "keyword_prefix_accept", "prefix": prefix}, 1.0))
    if not generator.omit_keyword_abandon:
        rows.append(
            ({"kind": "keyword_prefix_abandon", "prefix": prefix}, KEYWORD_ABANDON_MASS)
        )
    probabilities = _decimal_distribution(
        _keyword_smoothed_probabilities([weight for _payload, weight in rows])
    )
    character_ids = {" ": "space", "-": "hyphen"}
    options: list[Option] = []
    for index, (payload, _weight) in enumerate(rows):
        if payload["kind"] == "keyword_hint_character":
            suffix = payload["prefix"][-1]
            option_id = (
                f"keyword-hint-{len(prefix) + 1}-"
                f"{character_ids.get(suffix, suffix)}"
            )
        elif payload["kind"] == "keyword_prefix_accept":
            option_id = f"keyword-accept-{len(prefix)}"
        else:
            option_id = f"keyword-abandon-{len(prefix)}"
        options.append(Option(option_id, payload, probabilities[index]))
    exit_text = (
        "accept it as the exact phrase (checkout to rewind if the search is wrong)"
        if generator.omit_keyword_abandon
        else "accept it as the exact phrase, or abandon the search"
    )
    return Question(
        f"Extend the exact prefix {prefix!r} by one frequency-weighted "
        f"character, or {exit_text}.",
        tuple(options),
    )


def apply_keyword(generator, payload: dict[str, Any]) -> None:
    generator._validate_active_dispatch_binding()
    kind = str(payload.get("kind") or "")
    if kind == "keyword_prefix_accept":
        term = str(payload.get("prefix") or "")
        _validate_keyword_prefix(term)
        if term != str(generator.state.get("pending_keyword_prefix") or ""):
            raise ValueError("accepted prefix does not match the paid search state")
    else:
        term = " ".join(str(payload.get("term") or "").split())
        source = (
            generator.state["pending_keyword_guesses"]
            if kind == "keyword_guess"
            else generator.state["pending_keyword_slate"]
        )
        selected = next(
            (
                row
                for row in source
                if row.get("term") == term and not row.get("is_category")
            ),
            None,
        )
        if selected is None:
            raise ValueError("keyword does not match the activated finite slate")
    binding = generator.state.get("active_dispatch_binding") or {}
    generator._add_fact(
        f"Central identity phrase: {term}",
        "keyword",
        stage=str(binding.get("source_stage") or generator.state["stage"]),
    )
    generator._note_event(f"DIRECT selected identity phrase: {term}")
    generator.state["pending_keyword_slate"] = []
    generator.state["pending_keyword_hint"] = None
    generator.state["pending_keyword_prefix"] = None
    generator.state["pending_keyword_guesses"] = []
    generator._clear_active_dispatch_binding()

def record_keyword_guess_miss(generator, prefix: str) -> None:
    """Record one missed guess slate while the prefix search continues.

    The keyword channel stays active and bound: the miss is priced negative
    evidence for this exact prefix, not a channel-level rejection.
    """

    generator._validate_active_dispatch_binding()
    _validate_keyword_prefix(prefix)
    phrases = [
        str(row["term"]) for row in generator.state["pending_keyword_guesses"]
    ]
    generator.state["rejected_keywords"].append(
        {"phase": "guess", "prefix": prefix, "phrases": phrases}
    )
    generator.state["rejected_keywords"] = generator.state["rejected_keywords"][-24:]
    generator._note_event(
        "DIRECT paid guess outcome: no displayed completion for prefix "
        f"{prefix!r} identified the central contribution; the prefix search continues."
    )
    generator.state["pending_keyword_guesses"] = []

def record_keyword_retry(generator, kind: str) -> None:
    generator._validate_active_dispatch_binding()
    if kind == "keyword_prefix_abandon":
        phrases: list[str] = []
        phase = "prefix_abandon"
    else:
        phrases = [
            str(row["term"]) for row in generator.state["pending_keyword_slate"]
        ]
        phase = "direct"
    generator.state["rejected_keywords"].append(
        {
            "phase": phase,
            "prefix": generator.state.get("pending_keyword_prefix"),
            "phrases": phrases,
        }
    )
    generator.state["rejected_keywords"] = generator.state["rejected_keywords"][-24:]
    generator._note_event(
        "DIRECT paid keyword outcome: the constructive prefix search was "
        "abandoned before naming the central contribution."
        if kind == "keyword_prefix_abandon"
        else "DIRECT paid keyword outcome: none of the displayed phrases "
        "identified the central contribution."
    )
    generator.state["pending_keyword_slate"] = []
    generator.state["pending_keyword_hint"] = None
    generator.state["pending_keyword_prefix"] = None
    generator.state["pending_keyword_guesses"] = []
    generator._clear_active_dispatch_binding()
