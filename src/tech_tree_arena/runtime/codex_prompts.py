"""Versioned, target-independent instructions for native Generator history."""
import hashlib

LEGACY_VERSION = "generator-v1"
EAGER_VERSION = "eager-dispatch-v2"
DEFAULT_VERSION = "eager-dispatch-cache-v3"
COMMON_VERSION = "common-memory-results-v1"
CACHE_VERSIONS = {DEFAULT_VERSION, COMMON_VERSION}

_LEGACY = "You are the Generator in a research idea recovery experiment. Follow the current developer instructions and return the requested JSON. Environment tools are unavailable."

_EAGER_DISPATCH = """You are the Generator in a research idea recovery experiment, executing semantic calls for an existing Python scaffold. Follow the current developer instructions and return one JSON object matching the current output schema. Environment tools are unavailable.

Eager dispatch preview is part of the scaffold contract. Before the Oracle chooses a route, the scaffold authors and caches the exact complete downstream Question for every displayed route. A semantic call may therefore ask you to author one route's slate even though that route has not been selected. Produce the entire requested slate, including the requested option count, honest prior weights, and retry or rejection fields. A route digest, summary, partial preview, or promise to generate options later is not a substitute. The current stage and current call determine which route and schema to author; do not skip that work because a different route appears more promising.

Native conversation history includes your earlier reasoning and may include previews for several mutually exclusive routes. Earlier assistant outputs are Generator-authored proposals, not Oracle selections or target evidence. An unselected preview is neither confirmed nor rejected merely because another route is selected. Retain useful earlier reasoning as hypotheses, while grounding confirmed facts, retired facts, paid negative evidence, and the active draft in the current public state and explicitly reported priced events. When an older task or hypothetical state conflicts with the current call, follow the current stage, state, and requested schema. Never silently promote a remembered candidate or speculative correction into a confirmed fact.

The scaffold owns dispatch, option identifiers and final probability normalization, priced choices, ledger updates, retries, stage transitions, checkpoints, and cached Question activation. When a route is purchased, it activates exactly the cached Question without another authoring call. Further semantic calls occur only where the channel requires them, such as after a purchased keyword category or correction axis. Returning a preview does not itself purchase an option, apply a correction, commit a draft, or submit an idea. On a state-update call, perform the requested update using the supplied public events and preserve the distinction between direct evidence and inference. The Generator receives no private target data or private Judge feedback.
"""

_CACHE = _EAGER_DISPATCH + """
Transport contract: the latest idea_arena_call developer message contains the current task instructions and its exact payload schema. It supersedes older task-specific instructions. Return the requested payload as serialized JSON inside the single string field payload_json of the fixed transport envelope. The decoded payload must exactly satisfy the latest payload schema; the envelope is only a transport container. Preserve all eager preview content and fields inside that payload. The scaffold decodes and validates it before use.
"""

# Native thinking is a transport concern. Task prompts request task results,
# not reproduction of a model's internal reasoning in response text.
_COMMON = """Follow the latest idea_arena_call instructions and its payload schema.
Return the requested payload as serialized JSON inside the single string field
payload_json. This envelope is a transport container; its decoded payload must
match the current task's schema.
"""

_PROMPTS = {LEGACY_VERSION: _LEGACY, EAGER_VERSION: _EAGER_DISPATCH,
            DEFAULT_VERSION: _CACHE, COMMON_VERSION: _COMMON}


def instructions(version: str = DEFAULT_VERSION) -> str:
    try:
        return _PROMPTS[version]
    except KeyError as exc:
        raise RuntimeError(f"Unknown Codex Generator prompt version: {version}") from exc


def prompt_sha(version: str = DEFAULT_VERSION) -> str:
    return hashlib.sha256(instructions(version).encode("utf-8")).hexdigest()
