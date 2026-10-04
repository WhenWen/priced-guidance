from pathlib import Path
from decimal import Decimal, localcontext
from types import SimpleNamespace
import json

import pytest

from tech_tree_arena import (
    Checkout,
    Choice,
    Idea,
    IdeaVerdict,
    Option,
    PresentedQuestion,
    Question,
    StageReady,
    StageTransition,
    Submission,
    SubmissionFeedback,
    SubmitOption,
)
from tech_tree_arena.runtime.actor import ActorFactory, ActorRuntime
from tech_tree_arena.runtime.engine import ArenaRunner
from tech_tree_arena.runtime.providers import ScriptedModelBackend
from tech_tree_arena.runtime.services import ServiceFactory
from tech_tree_arena.contract.validation import canonical_json, validate_question
from tech_tree_arena.submission_io.manifest import (
    load_manifest,
    load_participant_classes,
    stage_module_hashes,
)


_ROOT = Path(__file__).resolve().parents[3]
_PAIR = _ROOT / "submissions" / "reference_pair"
ModularGenerator, ModularOracle = load_participant_classes(load_manifest(_PAIR))
_PAIR_GLOBALS = ModularGenerator.__init__.__globals__


def _judge_call(judge):
    """Bind a test Judge the way the Arena binds the real one for the Oracle."""

    def call(raw_ideas):
        ideas = tuple(
            Idea(item["idea_id"], item.get("content"), item.get("probability", 1))
            for item in raw_ideas or ()
        )
        return [
            {
                "idea_id": verdict.idea_id,
                "passed": bool(verdict.passed),
                "private_reason": verdict.private_reason or "",
            }
            for verdict in judge.evaluate(None, ideas)
        ]

    return call


def _snapshot_passes(services, presented) -> bool:
    """Whether the Judge accepts the draft the Generator published this turn."""

    snapshot = next(
        (
            payload["generator_idea_snapshot"]
            for option in presented.question.options
            if isinstance(payload := option.public_payload, dict)
            and payload.get("generator_idea_snapshot")
        ),
        None,
    )
    if not snapshot:
        return False
    verdicts = services.judge_evaluate(snapshot.get("ideas") or [])
    return any(verdict["passed"] for verdict in verdicts)
_CHANNEL_GLOBALS = vars(_PAIR_GLOBALS["_stage_channels"]("directional"))


def test_actor_edits_stay_inside_their_stage_hash_group(tmp_path) -> None:
    """Editing a later actor function must not move shared/earlier hashes.

    This is the promotion contract that makes policy iteration compatible
    with --compatible-submission: a directional pass freezes shared and
    directional only, so essence/strict actor implementations remain
    swappable.
    """

    import shutil

    baseline = stage_module_hashes(load_manifest(_PAIR))
    clone = tmp_path / "clone"
    shutil.copytree(_PAIR, clone)
    policy = clone / "participant" / "stages" / "essence.py"
    policy.write_text(
        policy.read_text(encoding="utf-8")
        + "\ndef generator_on_enter(generator, transition):\n"
        + "    generator.state.setdefault('essence_revision', 2)\n",
        encoding="utf-8",
    )
    edited = stage_module_hashes(load_manifest(clone))

    assert edited["shared"] == baseline["shared"]
    assert edited["directional"] == baseline["directional"]
    assert edited["strict"] == baseline["strict"]
    assert edited["essence"] != baseline["essence"]

    EditedGenerator, _ = load_participant_classes(load_manifest(clone))
    services = ServiceFactory(
        seed=31,
        public_resources={"active_stage": "directional"},
    ).create()
    generator = EditedGenerator(services)
    generator.step(StageTransition("directional", "essence"))

    assert generator.state["essence_revision"] == 2


def test_manifest_has_independent_stage_hashes() -> None:
    hashes = stage_module_hashes(load_manifest(_PAIR))

    assert set(hashes) == {"shared", "directional", "essence", "strict"}
    assert len(set(hashes.values())) == 4


_STAGE_ACTOR_FUNCTIONS = {
    "generator_on_enter",
    "generator_step",
    "dispatch_question",
    "mc_question",
    "candidate_question",
    "run_state_update",
    "correction_axis_question",
    "correction_replacement_question",
    "submission",
    "oracle_on_enter",
    "oracle_step",
}


def test_each_stage_owns_the_complete_actor_function_surface() -> None:
    actor_module = _PAIR_GLOBALS["_stage_actor_module"]

    for stage in ("directional", "essence", "strict"):
        module = actor_module(stage)
        assert all(callable(getattr(module, name, None)) for name in _STAGE_ACTOR_FUNCTIONS)
        assert isinstance(module.JUDGE_CRITERION, str) and module.JUDGE_CRITERION


def test_actor_shell_calls_the_active_stage_functions(monkeypatch) -> None:
    directional = _PAIR_GLOBALS["_stage_actor_module"]("directional")
    original_step = directional.generator_step
    calls = []

    def recording_step(generator, value):
        calls.append((generator.state["stage"], value))
        return original_step(generator, value)

    monkeypatch.setattr(directional, "generator_step", recording_step)
    services = ServiceFactory(
        seed=19,
        public_resources={"active_stage": "directional", "taxonomy": {}},
    ).create()
    generator = ModularGenerator(services)

    generator.step(None)

    assert calls == [("directional", None)]


def test_target_stage_owns_service_free_transition_initialization(monkeypatch) -> None:
    essence = _PAIR_GLOBALS["_stage_actor_module"]("essence")

    def initialize(generator, transition):
        generator.state["pending_events"].append(
            f"Essence policy revision entered from {transition.from_stage}."
        )

    monkeypatch.setattr(essence, "generator_on_enter", initialize)
    services = ServiceFactory(
        seed=29,
        public_resources={"active_stage": "directional"},
    ).create()
    generator = ModularGenerator(services)
    service_events_before = services.event_count()

    ready = generator.step(StageTransition("directional", "essence"))

    assert ready.handoff["pending_events"][-1] == (
        "Essence policy revision entered from directional."
    )
    assert services.event_count() == service_events_before


def test_generator_transition_preserves_private_ledger_without_service_calls() -> None:
    services = ServiceFactory(
        seed=17,
        public_resources={"active_stage": "directional"},
    ).create()
    generator = ModularGenerator(services)
    generator.state.update(
        {
            "category": "a new algorithm or method",
            "current_draft": "A concise synthetic contribution.",
            "facts": [
                {
                    "fact_id": "F0001",
                    "text": "One selected atomic relation.",
                    "active": True,
                    "source": "directional:explore",
                }
            ],
            "whiteboard": "DIRECT category; OPEN mechanism",
            "pending_correction_axes": [{"axis_id": "CAX01"}],
            "pending_correction_axis": {"axis_id": "CAX01"},
            "pending_correction_replacements": [{"replacement_id": "CREP01"}],
            "next_fact_index": 2,
        }
    )

    ready = generator.step(StageTransition("directional", "essence"))

    assert isinstance(ready, StageReady)
    assert ready.stage == "essence"
    assert ready.handoff["current_draft"] == "A concise synthetic contribution."
    assert ready.handoff["active_facts"][0]["fact_id"] == "F0001"
    assert ready.handoff["pending_events"][-1].endswith("directional -> essence.")
    assert generator.state["pending_correction_axes"] == [{"axis_id": "CAX01"}]
    assert generator.state["pending_correction_axis"] == {"axis_id": "CAX01"}
    assert generator.state["pending_correction_replacements"] == [
        {"replacement_id": "CREP01"}
    ]
    assert services.event_count() == 0


def test_strict_dispatch_exposes_the_full_eager_route_set_with_active_preview() -> None:
    class StrictServices(_EagerServices):
        public_resources = {"active_stage": "strict", "taxonomy": {}}

    services = StrictServices()
    generator = ModularGenerator(services)
    generator.state["stage"] = "strict"
    generator.state["category"] = "a theory or analysis result"
    generator.state["current_draft"] = "A concise synthetic contribution."
    generator.state["secondary_field_complete"] = True
    generator.state["facts"] = [
        {
            "fact_id": "F0001",
            "text": "A visible synthetic fact.",
            "active": True,
            "source": "strict:explore",
        }
    ]
    generator.state["insufficient_directional_drafts"] = [
        {
            "draft_id": "D0001",
            "draft": "A concise synthetic contribution.",
            "recovery_mode": "audit",
        }
    ]

    question = generator._dispatch_question()
    payloads = [option.public_payload for option in question.options]

    # Every stage now exposes the full eager route set.
    assert [payload["mode"] for payload in payloads] == [
        "mc",
        "keyword",
        "audit",
        "correct",
        "submit",
    ]
    assert all(payload["stage"] == "strict" for payload in payloads)
    assert all(payload["source_stage"] == "strict" for payload in payloads)
    assert isinstance(question.options[-1], SubmitOption)
    assert all(
        payload["generator_idea_snapshot"]["ideas"][0]["content"]
        ["setting_and_object"]
        == "A concise synthetic contribution."
        for payload in payloads
    )
    # The route carries the exact cached slate, and opening it is free.
    keyword_payload = next(p for p in payloads if p["mode"] == "keyword")
    request_count = len(services.requests)
    downstream = generator.step(Choice("mode-keyword", public_payload=keyword_payload))
    assert keyword_payload["preview"] == _PAIR_GLOBALS["_serialized_question"](
        downstream
    )
    assert len(services.requests) == request_count
    validate_question(downstream)


def test_strict_stage_goal_routes_faithfulness_before_missing_detail() -> None:
    # The ordering has to live in STAGE_GOAL, the only stage text a prompt reads.
    # It used to be asserted against ORACLE_POLICY, which no prompt consumed, so
    # the test passed while the Oracle never saw a word of it.
    goal = _PAIR_GLOBALS["_stage_module"]("strict").STAGE_GOAL.casefold()

    assert "faithful" in goal
    assert goal.index("first remove") < goal.index("only then add")


def test_oracle_route_lists_name_only_routes_the_stage_displays(monkeypatch) -> None:
    # A hand-written route list once told the Oracle that explore and
    # differentiate were displayed at every stage. Strict has never displayed
    # either, and a dLLM Strict run paid for a correction-axis retry whose
    # stated purpose was to reach differentiate.
    notes_for = _PAIR_GLOBALS["_modular_dispatch_notes"]
    sentence_for = _PAIR_GLOBALS["_dispatch_routes_sentence"]
    stage_routes = _PAIR_GLOBALS["_stage_dispatch_routes"]

    for stage in ("directional", "essence", "strict"):
        displayed = stage_routes(stage)
        notes = notes_for(stage)
        sentence = sentence_for(stage)
        assert "@@DISPATCH_ROUTES@@" not in notes
        assert f"active {stage} stage displays " + ", ".join(displayed) in notes
        for route in _PAIR_GLOBALS["_BASE_DISPATCH_ROUTES"]:
            if route not in displayed:
                assert route not in sentence
    assert "explore" not in " ".join(stage_routes("strict"))

    # A replacement Strict menu must not alter replay of a paid Directional
    # Oracle turn. Earlier versions rendered all three menus into every prompt.
    directional_before = notes_for("directional")
    strict = _PAIR_GLOBALS["_stage_module"]("strict")
    monkeypatch.setattr(strict, "DROPPED_ROUTES", ())
    assert notes_for("directional") == directional_before
    assert notes_for("strict") != directional_before


def test_oracle_prompt_deletes_unsupported_interpretation_before_rediscovery() -> None:
    prompt = " ".join(_PAIR_GLOBALS["_ORACLE_SYSTEM_PROMPT"].split())
    delete_policy = "prefer a delete patch that completely retracts"
    rediscovery_policy = "at the next dispatch prefer keyword or MC"

    assert "no offered replace/refine patch is directly supported" in prompt
    assert delete_policy in prompt
    assert "If no offered delete does so, reject the replacement slate" in prompt
    assert rediscovery_policy in prompt
    assert prompt.index(delete_policy) < prompt.index(rediscovery_policy)


def test_oracle_prompt_uses_title_mechanism_as_taxonomy_bottleneck() -> None:
    prompt = " ".join(_PAIR_GLOBALS["_ORACLE_SYSTEM_PROMPT"].split())

    assert "treat taxonomy and category choices as search commitments" in prompt
    assert "paper title's main clause" in prompt
    assert "most tightly constrains its distinctive mechanism" in prompt
    assert "Apply that semantic-bottleneck rule throughout the run" in prompt
    assert "A cheap generic fact is not strategically comparable" in prompt
    assert "visible mechanism-level option resolves the central gap" in prompt


def test_directional_starts_with_priced_public_taxonomy_traversal() -> None:
    services = ServiceFactory(
        seed=23,
        public_resources={
            "active_stage": "directional",
            "taxonomy": {
                "deep learning": [
                    {"name": "representations"},
                    {"name": "optimization"},
                ],
                "deep learning / representations": [
                    {"name": "structured inputs"},
                ],
            },
        },
    ).create()
    generator = ModularGenerator(services)

    root = generator.step(None)
    representation = next(
        option
        for option in root.options
        if option.public_payload["name"] == "representations"
    )
    child = generator.step(
        Choice(
            representation.option_id,
            public_payload=representation.public_payload,
        )
    )
    stop = next(option for option in child.options if option.public_payload["stop"])
    category = generator.step(
        Choice(stop.option_id, public_payload=stop.public_payload)
    )

    assert root.options[0].public_payload["kind"] == "field"
    assert root.options[0].public_payload["role"] == "primary"
    assert "primary field: representations" in generator.state["pending_events"][-1]
    assert all(option.public_payload["kind"] == "category" for option in category.options)
    assert services.event_count() == 0


def test_directional_retry_records_the_entire_rejected_semantic_region() -> None:
    services = ServiceFactory(
        seed=29,
        public_resources={"active_stage": "directional"},
    ).create()
    generator = ModularGenerator(services)
    generator.state["pending_candidate_slate"] = [
        {
            "channel": "explore",
            "label": "region-a",
            "draft": "A neutral draft.",
            "object_family": "family-a",
            "contribution_direction": "direction-a",
            "taxonomy_relation": "central",
        },
        {
            "channel": "explore",
            "label": "region-b",
            "draft": "Another neutral draft.",
            "object_family": "family-b",
            "contribution_direction": "direction-b",
            "taxonomy_relation": "broadened",
        },
    ]

    generator._record_rejected_slate("explore")

    rejected = generator.state["rejected_candidate_regions"][-1]
    assert rejected["semantic_regions"] == [
        {
            "object_family": "family-a",
            "contribution_direction": "direction-a",
            "taxonomy_relation": "central",
        },
        {
            "object_family": "family-b",
            "contribution_direction": "direction-b",
            "taxonomy_relation": "broadened",
        },
    ]
    assert "drafts" not in rejected
    assert generator.state["pending_candidate_slate"] == []


def test_candidate_retry_mass_is_generator_declared_and_normalized() -> None:
    probabilities = _PAIR_GLOBALS["_weighted_probabilities"](
        [2.0, 1.0], retry_mass="0.25"
    )

    assert probabilities[-1] == "0.25"
    assert sum(map(float, probabilities)) == 1.0
    schema = _PAIR_GLOBALS["_candidate_schema"](3)
    assert "retry_prob" in schema["required"]


def test_directional_schema_and_runtime_require_two_axis_diversity(monkeypatch) -> None:
    monkeypatch.setitem(
        _PAIR_GLOBALS,
        "_stage_module",
        lambda stage: SimpleNamespace(STAGE_GOAL="Synthetic stage goal."),
    )
    schema = _PAIR_GLOBALS["_candidate_schema"](
        4, require_directional_axes=True
    )
    required = schema["properties"]["candidates"]["items"]["required"]
    properties = schema["properties"]["candidates"]["items"]["properties"]
    assert "object_family" in required
    assert "contribution_direction" in required
    assert "taxonomy_relation" in required
    assert "source_draft_ids" in required
    assert properties["object_family"]["maxLength"] == 500
    assert properties["contribution_direction"]["maxLength"] == 500

    def slate(*, clustered: bool) -> dict:
        return {
            "reasoning": "Synthetic slate.",
            "retry_prob": 0.2,
            "candidates": [
                {
                    "label": f"candidate-{index}",
                    "draft": f"A synthetic object {index} changes relation {index}.",
                    "atomic_fact": f"Object {index} changes relation {index}.",
                    "retire_fact_ids": [],
                    "prob": 1,
                    "object_family": (
                        "same family" if clustered else f"family-{index}"
                    ),
                    "contribution_direction": f"direction-{index}",
                    "taxonomy_relation": (
                        "central",
                        "context",
                        "broadened",
                        "adjacent",
                    )[index - 1],
                    "source_draft_ids": [],
                }
                for index in range(1, 5)
            ],
        }

    class Services:
        public_resources = {"active_stage": "directional"}

        def __init__(self) -> None:
            self.requests = []

        def structured_model(self, **request):
            self.requests.append(request)
            if len(self.requests) == 1:
                return slate(clustered=True)
            if len(self.requests) == 2:
                overlong = slate(clustered=False)
                overlong["candidates"][0]["object_family"] = "x" * 500 + "a"
                overlong["candidates"][1]["object_family"] = "x" * 500 + "b"
                return overlong
            return slate(clustered=False)

    services = Services()
    generator = ModularGenerator(services)
    rows, retry_probability = generator._generate_candidates(
        "Synthetic prompt.", 4, "explore", "replace"
    )

    assert len(services.requests) == 3
    assert retry_probability == 0.2
    assert len({row["object_family"] for row in rows}) == 4
    assert len({row["contribution_direction"] for row in rows}) == 4


def test_priced_directional_recovery_records_draft_without_private_feedback(
    monkeypatch,
) -> None:
    monkeypatch.setitem(
        _PAIR_GLOBALS,
        "_stage_module",
        lambda stage: SimpleNamespace(
            STAGE_GOAL="Synthetic stage goal.",
            EXPLORE_COUNT=12,
            AUDIT_COUNT=8,
            EXPLORE_ACTION="replace",
            AUDIT_ACTION="rewrite",
            EXPLORE_PROMPT="Synthetic explore prompt.",
            AUDIT_PROMPT="Synthetic audit prompt.",
        ),
    )
    class Services:
        public_resources = {"active_stage": "directional"}

        def __init__(self) -> None:
            self.requests = []

        def structured_model(self, **request):
            self.requests.append(request)
            return {
                "reasoning": "Synthetic alternatives.",
                "retry_prob": 0.2,
                "candidates": [
                    {
                        "label": f"repair-{index}",
                        "draft": f"A conservative synthetic core {index}.",
                        "atomic_fact": "",
                        "retire_fact_ids": [],
                        "prob": 1,
                        "object_family": f"family-{index % 4}",
                        "contribution_direction": f"direction-{index}",
                        "taxonomy_relation": (
                            "central",
                            "context",
                            "broadened",
                            "adjacent",
                        )[(index - 1) % 4],
                        "source_draft_ids": [],
                    }
                    for index in range(1, 9)
                ],
            }

    services = Services()
    generator = ModularGenerator(services)
    generator.state["current_draft"] = "An insufficient synthetic hypothesis."
    question = generator.step(
        Choice(
            "mode-audit",
            public_payload={
                "kind": "dispatch",
                "mode": "audit",
                "private_judge_marker": "must never cross",
            },
        )
    )

    assert generator.state["insufficient_directional_drafts"] == [
        {
            "draft_id": "D0001",
            "draft": "An insufficient synthetic hypothesis.",
            "recovery_mode": "audit",
        }
    ]
    assert len(question.options) == 9
    request_text = services.requests[0]["developer"] + services.requests[0]["user"]
    assert "must never cross" not in request_text
    assert "An insufficient synthetic hypothesis." in request_text
    assert "Supported but insufficient prior drafts" in request_text


def test_directional_requires_paid_safe_cross_draft_synthesis(monkeypatch) -> None:
    monkeypatch.setitem(
        _PAIR_GLOBALS,
        "_stage_module",
        lambda stage: SimpleNamespace(STAGE_GOAL="Synthetic stage goal."),
    )

    safe_ids = ("D0001", "D0002", "D0003")

    def slate(source_sets: list[list[str]]) -> dict:
        return {
            "reasoning": "Synthetic component synthesis.",
            "retry_prob": 0.2,
            "candidates": [
                {
                    "label": f"synthesis-{index}",
                    "draft": f"A synthetic synthesis {index}.",
                    "atomic_fact": f"A synthetic synthesis {index}.",
                    "retire_fact_ids": [],
                    "prob": 1,
                    "object_family": f"family-{index}",
                    "contribution_direction": f"direction-{index}",
                    "taxonomy_relation": (
                        "central",
                        "context",
                        "broadened",
                        "adjacent",
                    )[index - 1],
                    "source_draft_ids": source_sets[index - 1],
                }
                for index in range(1, 5)
            ],
        }

    class Services:
        public_resources = {"active_stage": "directional"}

        def __init__(self) -> None:
            self.requests = []

        def structured_model(self, **request):
            self.requests.append(request)
            if len(self.requests) == 1:
                return slate(
                    [["D0001", "D0002"], ["D0001", "D0003"], ["D0002", "D0003"], []]
                )
            if len(self.requests) == 2:
                return slate(
                    [["D0001", "D0002"], ["D0001", "D0003"], ["D0002", "D0003"], ["D0001", "D9999"]]
                )
            return slate(
                [
                    ["D0001", "D0002"],
                    ["D0001", "D0003"],
                    ["D0002", "D0003"],
                    ["D0001", "D0002", "D0003"],
                ]
            )

    services = Services()
    generator = ModularGenerator(services)
    generator.state["insufficient_directional_drafts"] = [
        {
            "draft_id": draft_id,
            "draft": f"Safe synthetic component {index}.",
            "recovery_mode": "audit" if index % 2 else "explore",
        }
        for index, draft_id in enumerate(safe_ids, 1)
    ] + [
        {
            "draft_id": "D9999",
            "draft": "Unsafe synthetic component.",
            "recovery_mode": "correct",
        }
    ]

    rows, retry_probability = generator._generate_candidates(
        "Synthetic synthesis prompt.", 4, "audit", "rewrite"
    )

    assert len(services.requests) == 3
    assert retry_probability == 0.2
    assert all(len(row["source_draft_ids"]) >= 2 for row in rows)
    assert all("D9999" not in row["source_draft_ids"] for row in rows)
    assert {
        frozenset(row["source_draft_ids"]) for row in rows
    } == {
        frozenset(("D0001", "D0002")),
        frozenset(("D0001", "D0003")),
        frozenset(("D0002", "D0003")),
        frozenset(("D0001", "D0002", "D0003")),
    }


def test_directional_component_safety_only_downgrades() -> None:
    services = ServiceFactory(
        seed=43,
        public_resources={"active_stage": "directional"},
    ).create()
    generator = ModularGenerator(services)
    draft = "A stable synthetic component."
    generator.state["current_draft"] = draft

    generator._record_insufficient_directional_draft("explore")
    generator._record_insufficient_directional_draft("correct")
    generator._record_insufficient_directional_draft("audit")

    assert generator.state["insufficient_directional_drafts"] == [
        {
            "draft_id": "D0001",
            "draft": draft,
            "recovery_mode": "correct",
        }
    ]
    assert "downgraded by correction" in generator.state["pending_events"][-1]


def test_synthesis_validation_uses_exact_generator_visible_window(monkeypatch) -> None:
    monkeypatch.setitem(
        _PAIR_GLOBALS,
        "_stage_module",
        lambda stage: SimpleNamespace(STAGE_GOAL="Synthetic stage goal."),
    )

    class Services:
        public_resources = {"active_stage": "directional"}

        def __init__(self) -> None:
            self.requests = []

        def structured_model(self, **request):
            self.requests.append(request)
            return {
                "reasoning": "No visible safe components.",
                "retry_prob": 0.2,
                "candidates": [
                    {
                        "label": f"candidate-{index}",
                        "draft": f"A visible-window candidate {index}.",
                        "atomic_fact": f"A visible-window candidate {index}.",
                        "retire_fact_ids": [],
                        "prob": 1,
                        "object_family": f"family-{index}",
                        "contribution_direction": f"direction-{index}",
                        "taxonomy_relation": (
                            "central",
                            "context",
                            "broadened",
                            "adjacent",
                        )[index - 1],
                        "source_draft_ids": [],
                    }
                    for index in range(1, 5)
                ],
            }

    services = Services()
    generator = ModularGenerator(services)
    generator.state["insufficient_directional_drafts"] = [
        {
            "draft_id": f"D{index:04d}",
            "draft": f"Synthetic archived draft {index}.",
            "recovery_mode": "explore" if index <= 2 else "correct",
        }
        for index in range(1, 19)
    ]

    rows, _ = generator._generate_candidates(
        "Synthetic visible-window prompt.", 4, "explore", "replace"
    )

    assert len(services.requests) == 1
    assert "D0001" not in services.requests[0]["user"]
    assert "D0002" not in services.requests[0]["user"]
    assert all(row["source_draft_ids"] == [] for row in rows)


def test_directional_runtime_rejects_missing_taxonomy_relation_coverage(
    monkeypatch,
) -> None:
    monkeypatch.setitem(
        _PAIR_GLOBALS,
        "_stage_module",
        lambda stage: SimpleNamespace(STAGE_GOAL="Synthetic stage goal."),
    )

    class Services:
        public_resources = {"active_stage": "directional"}

        def __init__(self) -> None:
            self.requests = []

        def structured_model(self, **request):
            self.requests.append(request)
            return {
                "reasoning": "Synthetic but unbalanced slate.",
                "retry_prob": 0.2,
                "candidates": [
                    {
                        "label": f"candidate-{index}",
                        "draft": f"A distinct synthetic draft {index}.",
                        "atomic_fact": f"A distinct synthetic fact {index}.",
                        "retire_fact_ids": [],
                        "prob": 1,
                        "object_family": f"family-{index}",
                        "contribution_direction": f"direction-{index}",
                        "taxonomy_relation": "central",
                        "source_draft_ids": [],
                    }
                    for index in range(1, 5)
                ],
            }

    services = Services()
    generator = ModularGenerator(services)

    with pytest.raises(ValueError, match="valid finite candidate slate"):
        generator._generate_candidates(
            "Synthetic prompt.", 4, "explore", "replace"
        )
    assert len(services.requests) == 3


def test_rejected_repair_slate_returns_to_dispatch_with_fresh_eager_previews() -> None:
    services = _EagerServices()
    generator = ModularGenerator(services)
    generator.state["current_draft"] = "A stable synthetic draft."
    generator.state["insufficient_directional_drafts"] = [
        {
            "draft_id": "D0001",
            "draft": "A stable synthetic draft.",
            "recovery_mode": "audit",
        }
    ]
    generator.state["pending_candidate_slate"] = [
        {
            "channel": "audit",
            "label": "synthetic-repair",
            "draft": "A distinct synthetic repair.",
            "object_family": "synthetic family",
            "contribution_direction": "synthetic direction",
            "taxonomy_relation": "adjacent",
        }
    ]
    question = generator.step(
        Choice(
            "audit-retry",
            public_payload={"kind": "retry", "channel": "audit"},
        )
    )

    assert [option.public_payload["mode"] for option in question.options] == ["mc", "keyword", "submit"]
    # The two active Directional routes are both authored as exact previews.
    assert {request["schema_name"] for request in services.requests} == {
        "state_update",
        "directional_atomic_mc",
        "directional_keyword_categories",
    }
    assert generator.state["rejected_candidate_regions"][-1]["channel"] == "audit"


def _synthetic_correction_fact() -> str:
    values = " ".join(f"old-neutral-value-{index:02d}" for index in range(1, 17))
    return f"The neutral system records {values} for one synthetic task"


def _synthetic_correction_axis_response(owned: bool = True) -> dict:
    # A before-claim is a proposition entailed by the draft, not necessarily a
    # sentence or exact span. source_fact_id stays optional for synthesized
    # inferences and overclaims.
    return {
        "reasoning": "Synthetic target-blind before-claims.",
        "retry_prob": 0.2,
        "axes": [
            {
                "before_claim": (
                    "The neutral system records "
                    f"old-neutral-value-{index:02d} for one synthetic task."
                ),
                "evidence": f"old-neutral-value-{index:02d}",
                "source_fact_id": "F0001" if owned else "",
                "prob": 1,
            }
            for index in range(1, 17)
        ],
    }


def _synthetic_correction_replacement_response(
    request: dict | None = None,
    fact: str | None = None,
    owned: bool = True,
) -> dict:
    """A semantic patch carries its complete after-claim and resulting draft.

    Each row rewrites exactly one of the sixteen neutral values, so every row
    differs from the source in a single continuous stretch.
    """

    source_draft = f"{_synthetic_correction_fact()}."
    if request is not None:
        # single-hunk repairs only exist relative to the real draft, so read it
        # back out of the narrative context the Generator actually sent
        narrative = json.loads(request["user"])["generator_visible_state"]
        marker = "Your current top idea (the exact draft a submission would carry):"
        if marker in narrative:
            tail = narrative.split(marker, 1)[1].lstrip("\n")
            source_draft = " ".join(tail.split("\n", 1)[0].split())
    rows = []
    for index in range(1, 17):
        old = f"old-neutral-value-{index:02d}"
        new = f"new-neutral-value-{index:02d}"
        rows.append({
            "mode": "replace",
            "after_claim": (
                f"The neutral system records {new} for one synthetic task."
            ),
            "draft": source_draft.replace(old, new, 1),
            "prob": 1,
        })
    return {
        "reasoning": "Synthetic target-blind repairs.",
        "retry_prob": 0.2,
        "replacements": rows,
    }


def test_two_step_semantic_correction_commits_after_claim_and_preserves_other_facts() -> None:
    class Services:
        public_resources = {"active_stage": "directional"}

        def __init__(self) -> None:
            self.requests = []

        def structured_model(self, **request):
            self.requests.append(request)
            if request["schema_name"].endswith("correction_axes"):
                return _synthetic_correction_axis_response()
            return _synthetic_correction_replacement_response(request)

    services = Services()
    generator = ModularGenerator(services)
    source_fact = _synthetic_correction_fact()
    preserved_fact = "A separate supported relation remains active"
    original_draft = f"{source_fact}. {preserved_fact}."
    generator.state["current_draft"] = original_draft
    generator.state["facts"] = [
        {
            "fact_id": "F0001",
            "text": source_fact,
            "active": True,
            "source": "directional:explore",
        },
        {
            "fact_id": "F0002",
            "text": preserved_fact,
            "active": True,
            "source": "directional:differentiate",
        },
    ]
    generator.state["next_fact_index"] = 3
    generator.state["insufficient_directional_drafts"] = [
        {"draft_id": "D0001", "draft": original_draft, "recovery_mode": "audit"}
    ]
    generator.state["next_directional_draft_index"] = 2

    axes = generator.step(
        Choice(
            "mode-correct",
            public_payload={"kind": "dispatch", "mode": "correct"},
        )
    )
    assert len(axes.options) == 17
    assert float(axes.options[2].probability) == pytest.approx(0.05)
    assert float(axes.options[-1].probability) == pytest.approx(0.2)

    axis = axes.options[2]
    axis_payload = dict(axis.public_payload)
    axis_payload["private_judge_marker"] = "MUST_NOT_ENTER_GENERATOR"
    replacements = generator.step(
        Choice(axis.option_id, public_payload=axis_payload)
    )
    assert len(replacements.options) == 17
    assert "MUST_NOT_ENTER_GENERATOR" not in repr(services.requests)

    replacement = replacements.options[4]
    generator._apply_correction_replacement(replacement.public_payload)

    assert len(services.requests) == 2
    assert generator.state["current_draft"] == original_draft.replace(
        "old-neutral-value-05", "new-neutral-value-05", 1
    )
    active = [row for row in generator.state["facts"] if row["active"]]
    assert [row["fact_id"] for row in active] == ["F0002", "F0003"]
    assert active[0]["text"] == preserved_fact
    assert active[1]["text"] == (
        "The neutral system records new-neutral-value-05 for one synthetic task."
    )
    assert generator.state["insufficient_directional_drafts"][0][
        "recovery_mode"
    ] == "correct"


def test_semantic_correction_forbids_insert_and_validates_mode_claim_pairs() -> None:
    schema = _PAIR_GLOBALS["_correction_replacement_schema"](16)
    mode_schema = schema["properties"]["replacements"]["items"]["properties"][
        "mode"
    ]
    assert mode_schema["enum"] == ["replace", "refine", "delete"]

    class Services:
        public_resources = {"active_stage": "directional"}

        def structured_model(self, **request):
            if request["schema_name"].endswith("correction_axes"):
                return _synthetic_correction_axis_response()
            response = _synthetic_correction_replacement_response(request)
            # Defense in depth even if a provider fails to enforce the schema.
            response["replacements"][0]["mode"] = "insert"
            response["replacements"][1]["after_claim"] = ""
            response["replacements"][2]["mode"] = "delete"
            # A real deletion carries no after-claim and removes only the
            # selected existing proposition from the draft.
            row = response["replacements"][3]
            row["mode"] = "delete"
            row["after_claim"] = ""
            row["draft"] = " ".join(
                row["draft"].replace("new-neutral-value-04", "", 1).split()
            )
            return response

    generator = ModularGenerator(Services())
    source_fact = _synthetic_correction_fact()
    generator.state["current_draft"] = f"{source_fact}."
    generator.state["facts"] = [
        {
            "fact_id": "F0001",
            "text": source_fact,
            "active": True,
            "source": "directional:explore",
        }
    ]

    claims = generator._correction_axis_question()
    patches = generator.step(
        Choice(
            claims.options[0].option_id,
            public_payload=claims.options[0].public_payload,
        )
    )

    displayed = [option.public_payload for option in patches.options[:-1]]
    assert all(payload["mode"] != "insert" for payload in displayed)
    assert all(
        (payload["mode"] == "delete") == (payload["after_claim"] == "")
        for payload in displayed
    )
    delete_patch = next(
        option for option in patches.options if option.public_payload.get("mode") == "delete"
    )
    generator._apply_correction_replacement(delete_patch.public_payload)
    assert not any(row["active"] for row in generator.state["facts"])


def test_unowned_semantic_refinement_adds_its_complete_after_claim() -> None:
    class Services:
        public_resources = {"active_stage": "directional"}

        def structured_model(self, **request):
            if request["schema_name"].endswith("correction_axes"):
                return _synthetic_correction_axis_response(owned=False)
            response = _synthetic_correction_replacement_response(
                request, owned=False
            )
            for row in response["replacements"]:
                row["mode"] = "refine"
            return response

    generator = ModularGenerator(Services())
    source_draft = f"{_synthetic_correction_fact()}."
    generator.state["current_draft"] = source_draft

    claims = generator._correction_axis_question()
    patches = generator.step(
        Choice(
            claims.options[0].option_id,
            public_payload=claims.options[0].public_payload,
        )
    )
    selected = patches.options[0]
    generator._apply_correction_replacement(selected.public_payload)

    active = [row for row in generator.state["facts"] if row["active"]]
    assert [row["text"] for row in active] == [
        selected.public_payload["after_claim"]
    ]


def test_invalid_semantic_claim_slates_retry_three_times_then_fail_closed() -> None:
    class Services:
        public_resources = {"active_stage": "directional"}

        def __init__(self) -> None:
            self.requests = []

        def structured_model(self, **request):
            self.requests.append(request)
            response = _synthetic_correction_axis_response()
            for row in response["axes"][:13]:
                row["source_fact_id"] = "F9999"  # names no active fact
            return response

    services = Services()
    generator = ModularGenerator(services)
    source_fact = _synthetic_correction_fact()
    generator.state["current_draft"] = f"{source_fact}."
    generator.state["facts"] = [
        {
            "fact_id": "F0001",
            "text": source_fact,
            "active": True,
            "source": "directional:explore",
        }
    ]

    question = generator._correction_axis_question()

    assert len(services.requests) == 3
    assert [option.option_id for option in question.options] == [
        "correction-axis-retry"
    ]
    assert question.options[0].probability == "1"
    assert generator.state["current_draft"] == f"{source_fact}."
    assert generator.state["pending_correction_axes"] == []


def test_invalid_atomic_replacement_slates_retry_three_times_without_mutation() -> None:
    class Services:
        public_resources = {"active_stage": "directional"}

        def __init__(self) -> None:
            self.requests = []

        def structured_model(self, **request):
            self.requests.append(request)
            if request["schema_name"].endswith("correction_axes"):
                return _synthetic_correction_axis_response()
            response = _synthetic_correction_replacement_response(request)
            for index, row in enumerate(response["replacements"][:13], 1):
                # two separate edits in one repair: not atomic
                row["draft"] = (
                    row["draft"]
                    .replace("old-neutral-value-01", "x", 1)
                    .replace("for one synthetic task", "for two synthetic tasks", 1)
                )
            return response

    services = Services()
    generator = ModularGenerator(services)
    source_fact = _synthetic_correction_fact()
    original_draft = f"{source_fact}."
    generator.state["current_draft"] = original_draft
    generator.state["facts"] = [
        {
            "fact_id": "F0001",
            "text": source_fact,
            "active": True,
            "source": "directional:explore",
        }
    ]

    axes = generator._correction_axis_question()
    selected = axes.options[0]
    replacements = generator.step(
        Choice(selected.option_id, public_payload=selected.public_payload)
    )

    assert len(services.requests) == 4
    assert [option.option_id for option in replacements.options] == [
        "correction-value-retry"
    ]
    assert replacements.options[0].probability == "1"
    assert generator.state["current_draft"] == original_draft
    assert generator.state["facts"][0]["active"] is True
    assert generator.state["pending_correction_replacements"] == []


def test_semantic_claim_sanitizer_drops_duplicate_evidence_and_folds_mass() -> None:
    class Services:
        public_resources = {"active_stage": "directional"}

        def structured_model(self, **_request):
            response = _synthetic_correction_axis_response()
            # A second before-claim grounded in the same evidence is ambiguous.
            response["axes"][0]["evidence"] = response["axes"][1]["evidence"]
            return response

    generator = ModularGenerator(Services())
    source_fact = _synthetic_correction_fact()
    generator.state["current_draft"] = f"{source_fact}."
    generator.state["facts"] = [
        {
            "fact_id": "F0001",
            "text": source_fact,
            "active": True,
            "source": "directional:explore",
        }
    ]

    question = generator._correction_axis_question()

    assert len(question.options) == 16
    assert all(
        Decimal(option.probability) == Decimal("0.05")
        for option in question.options[:-1]
    )
    assert Decimal(question.options[-1].probability) == Decimal("0.25")
    assert sum(Decimal(option.probability) for option in question.options) == 1
    displayed = [
        option.public_payload["evidence"] for option in question.options[:-1]
    ]
    assert "old-neutral-value-01" not in displayed


def test_atomic_span_rejects_partial_token_cross_clause_and_shared_fact_ownership() -> None:
    span = _PAIR_GLOBALS["_atomic_correction_span"]

    assert span("A transformer is used", "form", {"F0001": "transformer"}, "F0001") is None
    assert (
        span(
            "A wrong value. Another claim remains",
            "wrong value. Another claim",
            {"F0001": "wrong value. Another claim"},
            "F0001",
        )
        is None
    )
    assert (
        span(
            "A system uses sharedvalue",
            "sharedvalue",
            {"F0001": "uses sharedvalue", "F0002": "also uses sharedvalue"},
            "F0001",
        )
        is None
    )


@pytest.mark.parametrize(
    "value",
    [
        "old-neutral-value-01 and adds claim",
        "new value, with another clause",
        "new value: extra claim",
        "new value because it changes behavior",
        "new value while adding behavior",
        "new value plus another mechanism",
        "new value which also adds behavior",
    ],
)
def test_atomic_replacement_rejects_restatement_punctuation_and_connectors(
    value: str,
) -> None:
    validate = _PAIR_GLOBALS["_is_atomic_replacement_value"]
    assert not validate(value, "old-neutral-value-01")


def test_semantic_patch_sanitizer_folds_dropped_mass_without_renormalizing() -> None:
    class Services:
        public_resources = {"active_stage": "directional"}

        def structured_model(self, **request):
            if request["schema_name"].endswith("correction_axes"):
                return _synthetic_correction_axis_response()
            response = _synthetic_correction_replacement_response(request)
            response["replacements"][0]["draft"] = (
                "old-neutral-value-01 and adds claim"
            )
            return response

    generator = ModularGenerator(Services())
    source_fact = _synthetic_correction_fact()
    generator.state["current_draft"] = f"{source_fact}."
    generator.state["facts"] = [
        {
            "fact_id": "F0001",
            "text": source_fact,
            "active": True,
            "source": "directional:explore",
        }
    ]
    axes = generator._correction_axis_question()
    replacements = generator.step(
        Choice(
            axes.options[0].option_id,
            public_payload=axes.options[0].public_payload,
        )
    )

    assert len(replacements.options) == 16
    assert all(
        Decimal(option.probability) == Decimal("0.05")
        for option in replacements.options[:-1]
    )
    assert Decimal(replacements.options[-1].probability) == Decimal("0.25")
    assert sum(Decimal(option.probability) for option in replacements.options) == 1
    assert all(
        "adds claim" not in str(option.public_payload)
        for option in replacements.options
    )


def test_pending_semantic_claim_survives_checkpoint_and_fork() -> None:
    class CorrectionGenerator(ModularGenerator):
        def __init__(self, services):
            super().__init__(services)
            source_fact = _synthetic_correction_fact()
            self.state["current_draft"] = f"{source_fact}."
            self.state["facts"] = [
                {
                    "fact_id": "F0001",
                    "text": source_fact,
                    "active": True,
                    "source": "directional:explore",
                }
            ]

    backend = ScriptedModelBackend(
        [
            _synthetic_correction_axis_response(),
            _synthetic_correction_replacement_response(),
            {"whiteboard": "Synthetic belief state.", "top_idea": {"setting_and_object": ""}},
        ]
    )
    runtime = ActorRuntime()
    actor = runtime.start(
        ActorFactory(
            CorrectionGenerator,
            service_factory=ServiceFactory(seed=43, model_backend=backend),
        )
    )
    axes = runtime.call(
        actor,
        Choice(
            "mode-correct",
            public_payload={"kind": "dispatch", "mode": "correct"},
        ),
    )
    checkpoint = runtime.checkpoint(actor)
    fork = runtime.fork(checkpoint, "atomic-correction-fork")

    assert fork.actor.state["pending_correction_axes"] == actor.actor.state[
        "pending_correction_axes"
    ]
    selected = axes.options[4]
    replacements = runtime.call(
        fork,
        Choice(selected.option_id, public_payload=selected.public_payload),
    )

    assert len(replacements.options) == 17
    assert fork.actor.state["pending_correction_axis"]["axis_id"] == "CAX05"
    assert len(fork.actor.state["pending_correction_replacements"]) == 16


def test_historical_semantic_claim_and_patch_survive_stage_transitions() -> None:
    class Services(_EagerServices):
        def __init__(self) -> None:
            super().__init__()
            self.scripted = [
                _synthetic_correction_axis_response(),
                _synthetic_correction_replacement_response(),
            ]

        def structured_model(self, **request):
            if self.scripted:
                return self.scripted.pop(0)
            self.requests.append(request)
            return _synthetic_eager_response(request)

    generator = ModularGenerator(Services())
    source_fact = _synthetic_correction_fact()
    generator.state["current_draft"] = f"{source_fact}."
    generator.state["facts"] = [
        {
            "fact_id": "F0001",
            "text": source_fact,
            "active": True,
            "source": "directional:explore",
        }
    ]
    generator.state["next_fact_index"] = 2
    generator.state["insufficient_directional_drafts"] = [
        {
            "draft_id": "D0001",
            "draft": f"{source_fact}.",
            "recovery_mode": "audit",
        }
    ]
    generator.state["next_directional_draft_index"] = 2

    axes = generator._correction_axis_question()
    generator.step(StageTransition("directional", "essence"))
    axis = axes.options[2]
    replacements = generator.step(
        Choice(axis.option_id, public_payload=axis.public_payload)
    )
    generator.step(StageTransition("essence", "strict"))
    replacement = replacements.options[4]
    dispatch = generator.step(
        Choice(replacement.option_id, public_payload=replacement.public_payload)
    )

    assert generator.state["stage"] == "strict"
    assert generator.state["current_draft"] == f"{source_fact}.".replace(
        "old-neutral-value-05", "new-neutral-value-05", 1
    )
    assert generator.state["insufficient_directional_drafts"][0][
        "recovery_mode"
    ] == "correct"
    # v1.14: Strict runs Essence's routes plus audit.
    assert [option.public_payload["mode"] for option in dispatch.options] == [
        "mc",
        "keyword",
        "audit",
        "correct",
        "submit",
    ]


def test_semantic_correction_path_prices_mode_claim_patch_and_submit_without_leak() -> None:
    secret_reason = "PRIVATE_SYNTHETIC_CORRECTION_DIAGNOSIS"

    class FlowGenerator(ModularGenerator):
        def __init__(self, services):
            super().__init__(services)
            source_fact = _synthetic_correction_fact()
            self.state["category"] = "a new algorithm or method"
            self.state["current_draft"] = f"{source_fact}."
            self.state["facts"] = [
                {
                    "fact_id": "F0001",
                    "text": source_fact,
                    "active": True,
                    "source": "directional:explore",
                }
            ]
            self.state["next_fact_index"] = 2

        def _directional_dispatch_question(self):
            return _simple_directional_dispatch(self)

        def step(self, value):
            if value is None:
                return self._dispatch_question()
            return super().step(value)

    class FlowOracle:
        def __init__(self, _target, services):
            self.services = services

        def step(self, presented):
            assert isinstance(presented, PresentedQuestion)
            if _snapshot_passes(self.services, presented):
                return Choice("mode-submit")
            kind = presented.question.options[0].public_payload.get("kind")
            if kind == "dispatch":
                return Choice("mode-correct")
            if kind == "correction_axis":
                return Choice("correction-axis-3")
            if kind == "correction_replacement":
                return Choice("correction-value-5")
            raise AssertionError(f"unexpected correction question kind {kind!r}")

    class FlowJudge:
        def evaluate(self, _target, ideas):
            return tuple(
                IdeaVerdict(
                    idea.idea_id,
                    "new-neutral-value-05"
                    in str(idea.content.get("setting_and_object"))
                    and "old-neutral-value-05"
                    not in str(idea.content.get("setting_and_object")),
                    None
                    if "new-neutral-value-05"
                    in str(idea.content.get("setting_and_object"))
                    else secret_reason,
                )
                for idea in ideas
            )

    backend = ScriptedModelBackend(
        [
            _synthetic_correction_axis_response(),
            _synthetic_correction_replacement_response(),
            {"whiteboard": "Synthetic belief state.", "top_idea": {"setting_and_object": ""}},
        ]
    )
    events = []
    judge = FlowJudge()
    result = ArenaRunner(
        event_sink=events.append,
    ).run(
        generator_factory=ActorFactory(
            FlowGenerator,
            service_factory=ServiceFactory(seed=47, model_backend=backend),
        ),
        oracle_factory=ActorFactory(
            FlowOracle,
            constructor_args=({"summary": {"synthetic": True}},),
            service_factory=ServiceFactory(
                seed=53, judge_call=_judge_call(judge)
            ),
        ),
        target={"summary": {"synthetic": True}},
        judge=judge,
        run_id="synthetic-atomic-correction-contract",
    )

    assert result.status == "pass"
    charged = [
        event
        for event in events
        if event["kind"] == "choice_cost" and event["information_bits"] > 0
    ]
    assert [event["option_id"] for event in charged] == [
        "mode-correct",
        "correction-axis-3",
        "correction-value-5",
        "mode-submit",
    ]
    expected_k = -__import__("math").log2(0.09 * 0.05 * 0.05 * 0.13 * 0.95**4)
    assert result.k == pytest.approx(expected_k)
    assert len(backend.requests) == 3
    assert secret_reason not in repr(backend.requests)


def _synthetic_differentiation_response(*, invalid_all_relations: bool = False) -> dict:
    rows = []
    for index in range(1, 11):
        relation = f"The contribution specifies synthetic relation {index}"
        if invalid_all_relations:
            relation += " because it includes an unrelated explanation"
        relation += "."
        rows.append(
            {
                "identity_relation": relation,
                "facet_family": f"neutral-facet-{index}",
                "source_draft_id": "D0001",
                "prob": 1,
            }
        )
    return {
        "reasoning": "Synthetic target-blind alternatives.",
        "retry_prob": 0.2,
        "candidates": rows,
    }


def _synthetic_eager_response(request: dict) -> dict:
    schema_name = request["schema_name"]
    if schema_name == "directional_atomic_mc":
        return {
            "reasoning": "Choose one neutral unresolved axis.",
            "question": "Which neutral value identifies the unresolved axis?",
            "axis": "neutral axis",
            "options": [
                {
                    "value": f"neutral-value-{index}",
                    "prob": 1,
                }
                for index in range(1, 9)
            ],
            "prob_all_incorrect": 0.1,
            "prob_ask_different": 0.1,
        }
    if schema_name == "directional_identity_keywords":
        count = request["schema"]["properties"]["keywords"]["minItems"]
        return {
            "reasoning": "Cover neutral identity names.",
            "keywords": [
                {"term": f"neutral identity {index}", "prob": 1}
                for index in range(1, count + 1)
            ],
            "prob_hint": 0.1,
        }
    if schema_name.endswith("_keyword_categories"):
        count = request["schema"]["properties"]["categories"]["minItems"]
        return {
            "reasoning": "Map the space of missing-detail kinds.",
            "categories": [
                {"category": f"neutral missing-detail kind {index}", "prob": 1}
                for index in range(1, count + 1)
            ],
            "prob_none": 0.1,
        }
    if schema_name == "directional_keyword_guesses":
        marker = "Revealed prefix the missing identity phrase starts with: '"
        prefix = request["user"].split(marker, 1)[1].split("'", 1)[0]
        count = request["schema"]["properties"]["guesses"]["minItems"]
        return {
            "reasoning": "Complete the paid neutral prefix once.",
            "guesses": [
                {"term": f"{prefix} identity {index}", "prob": 1}
                for index in range(1, count + 1)
            ],
            "prob_none": 0.2,
        }
    if schema_name == "directional_differentiate_candidates":
        return _synthetic_differentiation_response()
    if schema_name.endswith("_candidates"):
        count = request["schema"]["properties"]["candidates"]["minItems"]
        return {
            "reasoning": "Cover neutral whole-draft regions.",
            "retry_prob": 0.2,
            "candidates": [
                {
                    "label": f"neutral-candidate-{index}",
                    "draft": f"A neutral object {index} supports direction {index}.",
                    "atomic_fact": f"Neutral object {index} supports direction {index}",
                    "retire_fact_ids": [],
                    "prob": 1,
                    "object_family": f"neutral-family-{index}",
                    "contribution_direction": f"neutral-direction-{index}",
                    "taxonomy_relation": (
                        "central",
                        "context",
                        "broadened",
                        "adjacent",
                    )[(index - 1) % 4],
                    "source_draft_ids": [],
                }
                for index in range(1, count + 1)
            ],
        }
    if schema_name == "state_update":
        return {
            "whiteboard": "Synthetic belief state.",
            "top_idea": {"setting_and_object": ""},
        }
    if schema_name.endswith("correction_axes"):
        return _synthetic_correction_axis_response()
    if schema_name.endswith("correction_replacements"):
        return _synthetic_correction_replacement_response(request)
    raise AssertionError(f"unexpected synthetic schema {schema_name}")


class _EagerServices:
    public_resources = {"active_stage": "directional"}

    def __init__(self) -> None:
        self.requests = []

    def structured_model(self, **request):
        self.requests.append(request)
        return _synthetic_eager_response(request)


def _simple_directional_dispatch(generator) -> Question:
    draft = str(generator.state["current_draft"])
    preview = _PAIR_GLOBALS["_preview_payload"](draft)
    probabilities = _PAIR_GLOBALS["DIRECTIONAL_MODE_PROBABILITIES"]
    return Question(
        "Synthetic route-only dispatch.",
        tuple(
            (SubmitOption if mode == "submit" else Option)(
                f"mode-{mode}",
                {
                    "kind": "dispatch",
                    "mode": mode,
                    "current_draft": draft,
                    "generator_idea_snapshot": preview,
                },
                probability,
            )
            for mode, probability in probabilities.items()
        ),
    )


def _buy_first_category(generator, preview):
    """Enter the phrase stage: every keyword preview is now a category slate."""

    chosen = preview.options[0]
    assert chosen.public_payload["kind"] == "keyword_category"
    return generator.step(
        Choice(chosen.option_id, public_payload=chosen.public_payload)
    )


def _eager_generator_with_draft():
    services = _EagerServices()
    generator = ModularGenerator(services)
    generator.state["category"] = "a theory or analysis result"
    generator.state["current_draft"] = "A neutral supported core."
    generator.state["insufficient_directional_drafts"] = [
        {
            "draft_id": "D0001",
            "draft": "A neutral supported core.",
            "recovery_mode": "audit",
        }
    ]
    return generator, services


def test_eager_dispatch_previews_are_exact_cached_questions_and_activation_is_free() -> None:
    expected_modes = ["mc", "keyword", "submit"]
    for mode in expected_modes[:-1]:
        generator, services = _eager_generator_with_draft()
        dispatch = generator._dispatch_question()
        validate_question(dispatch)
        payload = next(
            option.public_payload
            for option in dispatch.options
            if option.public_payload["mode"] == mode
        )
        request_count = len(services.requests)
        downstream = generator.step(
            Choice(f"mode-{mode}", public_payload=payload)
        )

        assert payload["preview"] == _PAIR_GLOBALS["_serialized_question"](
            downstream
        )
        assert len(services.requests) == request_count
        assert generator.state["last_channel"] == mode
        assert generator.state["pending_dispatch_previews"] == {}
    assert [
        option.public_payload["mode"]
        for option in _eager_generator_with_draft()[0]._dispatch_question().options
    ] == expected_modes


@pytest.mark.parametrize("stage", ["directional", "essence", "strict"])
@pytest.mark.parametrize("provider", ["codex", "claude"])
def test_native_history_preserves_eager_preview_activation(stage, provider) -> None:
    import copy
    import threading

    class NativeBackend:
        supports_context_chain = True

        def __init__(self):
            self.parents = []
            self.local = threading.local()

        def structured_in_context(self, context, **request):
            self.parents.append(copy.deepcopy(context))
            node = {"thread_id": str(len(self.parents)), "turn_id": "turn",
                    "rollout": {"sha256": str(len(self.parents))}}
            if provider == "claude":
                node = {"backend": "claude-code-native-v1", "cwd": "/arena",
                        "session_id": str(len(self.parents)),
                        "transcript": {"sha256": str(len(self.parents))}}
            self.local.metadata = {"codex_context" if provider == "codex" else "native_context": node}
            return _synthetic_eager_response(request)

        def last_call_metadata(self):
            return self.local.metadata

    backend = NativeBackend()
    services = ServiceFactory(seed=17, model_backend=backend,
        public_resources={"active_stage": stage, "taxonomy": {}}).create()
    generator = ModularGenerator(services)
    generator.state["category"] = "a theory or analysis result"
    generator.state["current_draft"] = f"{_synthetic_correction_fact()}."
    generator.state["secondary_field_complete"] = True
    evidence_keys = ("current_draft", "facts", "whiteboard", "pending_events",
                     "rejected_candidate_regions", "rejected_mc_questions", "rejected_keywords")
    before = {key: copy.deepcopy(generator.state[key]) for key in evidence_keys}
    dispatch = generator._dispatch_question()
    validate_question(dispatch)
    assert {key: generator.state[key] for key in evidence_keys} == before
    count = len(backend.parents)
    assert count >= len(dispatch.options) - 1
    id_key = "thread_id" if provider == "codex" else "session_id"
    assert [None if p is None else p[id_key] for p in backend.parents] == [None, *map(str, range(1, count))]
    native_context = services.export_state()["model_context"]
    for option in dispatch.options:
        if isinstance(option, SubmitOption):
            continue
        # Open every route from the same cached dispatch boundary.
        branch = copy.copy(generator)
        branch.state = copy.deepcopy(generator.state)
        downstream = branch.step(Choice(option.option_id, public_payload=option.public_payload))
        assert option.public_payload["preview"] == _PAIR_GLOBALS["_serialized_question"](downstream)
        validate_question(downstream)
        assert len(backend.parents) == count
        assert services.export_state()["model_context"] == native_context


def test_eager_dispatch_is_bound_to_bundle_draft_and_fact_ledger() -> None:
    generator, _services = _eager_generator_with_draft()
    dispatch = generator._dispatch_question()
    keyword = next(
        option for option in dispatch.options if option.option_id == "mode-keyword"
    )
    tampered = dict(keyword.public_payload)
    tampered["bundle_id"] = "tampered"

    with pytest.raises(ValueError, match="stale or tampered"):
        generator.step(Choice(keyword.option_id, public_payload=tampered))

    generator, _services = _eager_generator_with_draft()
    dispatch = generator._dispatch_question()
    keyword = next(
        option for option in dispatch.options if option.option_id == "mode-keyword"
    )
    generator.state["facts"].append(
        {
            "fact_id": "F9999",
            "text": "A late synthetic mutation.",
            "active": True,
            "source": "directional:test",
        }
    )
    with pytest.raises(ValueError, match="stale or tampered"):
        generator.step(
            Choice(keyword.option_id, public_payload=keyword.public_payload)
        )


def test_unselected_eager_previews_do_not_mutate_epistemic_state() -> None:
    generator, services = _eager_generator_with_draft()
    before = {
        "draft": generator.state["current_draft"],
        "facts": list(generator.state["facts"]),
        "whiteboard": str(generator.state["whiteboard"]),
        "pending_events": list(generator.state["pending_events"]),
        "rejected_regions": list(generator.state["rejected_candidate_regions"]),
        "rejected_mc": list(generator.state["rejected_mc_questions"]),
        "rejected_keywords": list(generator.state["rejected_keywords"]),
    }

    dispatch = generator._dispatch_question()

    assert generator.state["current_draft"] == before["draft"]
    assert generator.state["facts"] == before["facts"]
    assert generator.state["whiteboard"] == before["whiteboard"]
    assert generator.state["pending_events"] == before["pending_events"]
    assert generator.state["rejected_candidate_regions"] == before["rejected_regions"]
    assert generator.state["rejected_mc_questions"] == before["rejected_mc"]
    assert generator.state["rejected_keywords"] == before["rejected_keywords"]
    # Directional currently exposes exactly two model-backed routes. Both
    # exact previews are authored without mutating the live ledger.
    assert len(services.requests) == 2
    assert len(canonical_json(dispatch)) < 262_144


def test_eager_dispatch_stays_bounded_with_long_fact_ledger_and_valid_maximal_drafts() -> None:
    class LongServices(_EagerServices):
        def structured_model(self, **request):
            self.requests.append(request)
            response = _synthetic_eager_response(request)
            if (
                request["schema_name"].endswith("_candidates")
                and request["schema_name"] != "directional_differentiate_candidates"
            ):
                for index, row in enumerate(response["candidates"], 1):
                    token = f"candidate{index:02d}" + "x" * 6
                    row["draft"] = " ".join([token, *(["x" * 15] * 139)])
                    row["atomic_fact"] = f"f{index:02d}" + "x" * 450
                    row["label"] = f"label{index:02d}" + "x" * 220
            return response

    services = LongServices()
    generator = ModularGenerator(services)
    generator.state["category"] = "a theory or analysis result"
    generator.state["current_draft"] = "A neutral supported core."
    generator.state["insufficient_directional_drafts"] = [
        {
            "draft_id": "D0001",
            "draft": "A neutral supported core.",
            "recovery_mode": "audit",
        }
    ]
    generator.state["facts"] = [
        {
            "fact_id": f"F{index:04d}",
            "text": f"ledger-{index:04d}-" + "x" * 290,
            "active": True,
            "source": "directional:synthetic",
        }
        for index in range(1, 97)
    ]

    dispatch = generator._dispatch_question()

    validate_question(dispatch)
    assert len(canonical_json(dispatch)) < _PAIR_GLOBALS[
        "DIRECTIONAL_DISPATCH_SOFT_CAP"
    ]
    assert all(
        "active_facts" not in option.public_payload for option in dispatch.options
    )
    context = _PAIR_GLOBALS["_generator_context"](generator.state)
    facts_section = context.split("Facts confirmed so far", 1)[1]
    facts_section = facts_section.split("Your current whiteboard:", 1)[0]
    assert facts_section.count("- [F") == 64


def test_oversized_exact_dispatch_fails_closed_instead_of_publishing_partial_preview(
    monkeypatch,
) -> None:
    monkeypatch.setitem(_PAIR_GLOBALS, "DIRECTIONAL_DISPATCH_SOFT_CAP", 1)
    generator, _services = _eager_generator_with_draft()

    with pytest.raises(ValueError, match="exact dispatch preview bundle"):
        generator._dispatch_question()


def test_mc_schema_has_no_second_free_form_fact_codeword() -> None:
    generator, services = _eager_generator_with_draft()
    dispatch = generator._dispatch_question()
    mc_option = next(
        option for option in dispatch.options if option.option_id == "mode-mc"
    )
    # v1.10 authors the MC slate when the MC route is opened, so its schema is
    # requested there rather than at dispatch.
    mc_question = generator.step(
        Choice("mode-mc", public_payload=mc_option.public_payload)
    )
    request = next(
        row for row in services.requests if row["schema_name"] == "directional_atomic_mc"
    )
    item = request["schema"]["properties"]["options"]["items"]
    assert set(item["properties"]) == {"value", "prob"}
    assert set(item["required"]) == {"value", "prob"}

    for option in mc_question.options[:-2]:
        payload = option.public_payload
        assert payload["fact"] == f"{payload['axis']}: {payload['value']}"


def test_atomic_mc_and_keyword_commit_only_the_selected_public_fact() -> None:
    mc_generator, _services = _eager_generator_with_draft()
    mc_dispatch = mc_generator._dispatch_question()
    mc_mode = next(
        option for option in mc_dispatch.options if option.option_id == "mode-mc"
    )
    mc_question = mc_generator.step(
        Choice(mc_mode.option_id, public_payload=mc_mode.public_payload)
    )
    mc_answer = mc_question.options[2]
    original_draft = mc_generator.state["current_draft"]
    mc_generator._apply_mc_answer(mc_answer.public_payload)

    assert mc_generator.state["current_draft"] == original_draft
    assert [row["text"] for row in mc_generator.state["facts"] if row["active"]] == [
        "neutral axis: neutral-value-3"
    ]
    assert "private" not in repr(mc_generator.state).casefold()

    keyword_generator, _services = _eager_generator_with_draft()
    keyword_dispatch = keyword_generator._dispatch_question()
    keyword_mode = next(
        option for option in keyword_dispatch.options if option.option_id == "mode-keyword"
    )
    keyword_question = _buy_first_category(
        keyword_generator,
        keyword_generator.step(
            Choice(keyword_mode.option_id, public_payload=keyword_mode.public_payload)
        ),
    )
    keyword = keyword_question.options[4]
    _CHANNEL_GLOBALS["apply_keyword"](
        keyword_generator, keyword.public_payload
    )

    assert [row["text"] for row in keyword_generator.state["facts"] if row["active"]] == [
        "Missing defining detail lies in: neutral missing-detail kind 1",
        "Central identity phrase: neutral identity 5",
    ]






def test_the_complementary_field_walk_is_no_longer_offered_anywhere() -> None:
    """v1.11 retires the walk instead of pricing it.

    Across 29 recorded Directional dispatches the Oracle opened it once, and
    the one time it paid for a walk on Utonia it spent 10.80 bits to reach
    "point clouds" -- a phrase the keyword channel sells for 6.05.
    """

    taxonomy = {
        "deep learning": [
            {"name": "representations"},
            {"name": "optimization"},
        ],
    }

    class Services(_EagerServices):
        public_resources = {"active_stage": "directional", "taxonomy": taxonomy}

    generator = ModularGenerator(Services())
    generator.state["primary_field"] = ["deep learning", "representations"]
    generator.state["primary_field_complete"] = True

    dispatch = generator.step(
        Choice(
            "category-1",
            public_payload={
                "kind": "category",
                "category": "a new algorithm or method",
            },
        )
    )

    modes = [option.public_payload["mode"] for option in dispatch.options]
    assert modes == ["mc", "keyword"]
    for stage in ("directional", "essence", "strict"):
        assert "secondary_field" in _PAIR_GLOBALS["_dropped_routes"](stage)


def test_historical_directional_bundle_survives_stage_transition() -> None:
    generator, services = _eager_generator_with_draft()
    dispatch = generator._dispatch_question()
    keyword = next(
        option for option in dispatch.options if option.option_id == "mode-keyword"
    )
    request_count = len(services.requests)
    generator.step(StageTransition("directional", "essence"))

    categories = generator.step(
        Choice(keyword.option_id, public_payload=keyword.public_payload)
    )
    # The historical bundle still binds and replays the exact route it priced.
    assert len(services.requests) == request_count
    question = _buy_first_category(generator, categories)
    selected = question.options[0]
    _CHANNEL_GLOBALS["apply_keyword"](generator, selected.public_payload)

    # Only the phrase slate needs a new call; the category slate was cached.
    assert len(services.requests) == request_count + 1
    assert generator.state["stage"] == "essence"
    assert generator.state["facts"][-1]["source"] == "directional:keyword"


# Removed in v1.12: the priced route -> candidate -> submit flow ran through explore, and a route that no stage offers has no public
# contract to assert. The dispatch-menu tests cover its absence instead.


# Removed in v1.12: differentiate is retired at every stage, and a route that no stage offers has no public
# contract to assert. The dispatch-menu tests cover its absence instead.


def test_paid_differentiate_builds_atomic_single_core_slate_without_private_marker(
    monkeypatch,
) -> None:
    monkeypatch.setitem(
        _PAIR_GLOBALS,
        "_stage_module",
        lambda stage: SimpleNamespace(
            STAGE_GOAL="Synthetic stage goal.",
            DIFFERENTIATE_COUNT=10,
            DIFFERENTIATE_ACTION="add",
            DIFFERENTIATE_PROMPT="Synthetic differentiation prompt.",
        ),
    )
    class Services:
        public_resources = {"active_stage": "directional"}

        def __init__(self) -> None:
            self.requests = []

        def structured_model(self, **request):
            self.requests.append(request)
            return _synthetic_differentiation_response(
                invalid_all_relations=len(self.requests) == 1
            )

    services = Services()
    generator = ModularGenerator(services)
    core = "A neutral supported core."
    generator.state["current_draft"] = core
    generator.state["facts"] = [
        {
            "fact_id": "F0001",
            "text": core,
            "active": True,
            "source": "directional:explore",
        }
    ]
    generator.state["next_fact_index"] = 2

    question = generator.step(
        Choice(
            "mode-differentiate",
            public_payload={
                "kind": "dispatch",
                "mode": "differentiate",
                "private_judge_marker": "MUST_NOT_ENTER_GENERATOR",
            },
        )
    )

    assert len(services.requests) == 2
    assert "MUST_NOT_ENTER_GENERATOR" not in repr(services.requests)
    assert services.requests[-1]["schema_name"] == "directional_differentiate_candidates"
    differentiation_item = services.requests[-1]["schema"]["properties"][
        "candidates"
    ]["items"]
    assert "label" not in differentiation_item["properties"]
    assert "label" not in differentiation_item["required"]
    concrete = question.options[:-1]
    assert len(concrete) == 10
    assert len({option.public_payload["facet_family"] for option in concrete}) >= 6
    for option in concrete:
        payload = option.public_payload
        assert payload["draft"].startswith(core)
        assert payload["draft"] == f"{core} {payload['identity_relation']}."
        assert payload["atomic_fact"] == payload["identity_relation"]
        assert payload["source_draft_ids"] == ["D0001"]
    assert question.options[-1].option_id == "differentiate-retry"
    assert "other gaps may remain" in question.question
    assert sum(float(option.probability) for option in question.options) == pytest.approx(1)

    selected = concrete[0]
    generator._apply_candidate(selected.public_payload)
    active = [row for row in generator.state["facts"] if row["active"]]
    assert [row["text"] for row in active] == [
        core,
        selected.public_payload["identity_relation"],
    ]
    assert generator.state["current_draft"] == selected.public_payload["draft"]
    assert generator.state["next_fact_index"] == 3
    assert generator.state["pending_events"][-1].endswith(
        selected.public_payload["identity_relation"]
    )


def test_differentiate_folds_dropped_row_mass_into_retry_without_leaking_text(
    monkeypatch,
) -> None:
    monkeypatch.setitem(
        _PAIR_GLOBALS,
        "_stage_module",
        lambda stage: SimpleNamespace(STAGE_GOAL="Synthetic stage goal."),
    )
    response = _synthetic_differentiation_response()
    dropped_marker = "SYNTHETIC_DROPPED_PRIVATE_MARKER"
    response["candidates"][0]["identity_relation"] = (
        f"{dropped_marker} because it adds an explanation."
    )
    response["candidates"][1]["source_draft_id"] = "D9999"
    response["candidates"][2]["identity_relation"] = response["candidates"][3][
        "identity_relation"
    ]
    newline_marker = "SYNTHETIC_NEWLINE_DROP_MARKER"
    response["candidates"][4]["identity_relation"] = (
        f"{newline_marker}\nA second proposition"
    )
    compound_relation = "Spatial and temporal inputs share one projection."
    response["candidates"][5]["identity_relation"] = compound_relation

    class Services:
        public_resources = {"active_stage": "directional"}

        def __init__(self) -> None:
            self.requests = []

        def structured_model(self, **request):
            self.requests.append(request)
            return response

    services = Services()
    generator = ModularGenerator(services)
    core = "A neutral supported core."
    generator.state["current_draft"] = core
    generator.state["insufficient_directional_drafts"] = [
        {"draft_id": "D0001", "draft": core, "recovery_mode": "audit"}
    ]

    rows, retry_probability = generator._generate_differentiations(
        "Synthetic prompt.", 10
    )
    probabilities = _PAIR_GLOBALS["_weighted_probabilities"](
        [row["prob"] for row in rows],
        retry_mass=str(retry_probability),
    )

    assert len(rows) == 6
    assert retry_probability == Decimal("0.52")
    assert all(float(value) == pytest.approx(0.08) for value in probabilities[:-1])
    assert float(probabilities[-1]) == pytest.approx(0.52)
    assert sum(map(float, probabilities)) == pytest.approx(1)
    assert dropped_marker not in repr(rows)
    assert newline_marker not in repr(rows)
    assert dropped_marker not in _PAIR_GLOBALS["_generator_context"](generator.state)
    assert compound_relation.removesuffix(".") in {
        row["identity_relation"] for row in rows
    }
    compound_row = next(
        row
        for row in rows
        if row["identity_relation"] == compound_relation.removesuffix(".")
    )
    assert Decimal(str(compound_row["prob"])) == Decimal("1")
    assert all(not row["identity_relation"].endswith(".") for row in rows)
    assert all(".." not in row["draft"] for row in rows)


@pytest.mark.parametrize(
    "relation",
    [
        "Spatial and temporal inputs share one projection",
        "The operation maps inputs and values into one code",
        "The operation accepts projection or normalization values",
    ],
)
def test_differentiate_atomic_filter_accepts_one_predicate_with_compound_argument(
    relation,
) -> None:
    assert _PAIR_GLOBALS["_is_atomic_identity_relation"](relation)


@pytest.mark.parametrize(
    "relation",
    [
        "The contribution changes one relation while adding another relation",
        "The contribution changes one relation whereas another remains fixed",
        "The contribution changes one relation because it improves another",
        "The contribution changes one relation thereby improving another",
        "The contribution changes one relation so that another becomes possible",
        "The contribution changes one relation in order to improve another",
        "The contribution changes one relation which also affects another",
        "The contribution changes one relation, and another process changes outputs",
        "The contribution changes one relation; another follows",
        "The contribution changes one relation..",
        "The contribution changes one relation,",
    ],
)
def test_differentiate_atomic_filter_rejects_bundled_or_malformed_relation(
    relation,
) -> None:
    assert not _PAIR_GLOBALS["_is_atomic_identity_relation"](relation)


def test_differentiate_invalid_weight_retries_entire_model_response(
    monkeypatch,
) -> None:
    monkeypatch.setitem(
        _PAIR_GLOBALS,
        "_stage_module",
        lambda stage: SimpleNamespace(STAGE_GOAL="Synthetic stage goal."),
    )
    class Services:
        public_resources = {"active_stage": "directional"}

        def __init__(self) -> None:
            self.requests = []

        def structured_model(self, **request):
            self.requests.append(request)
            response = _synthetic_differentiation_response()
            if len(self.requests) == 1:
                response["candidates"][0]["prob"] = 0
            return response

    services = Services()
    generator = ModularGenerator(services)
    core = "A neutral supported core."
    generator.state["current_draft"] = core
    generator.state["insufficient_directional_drafts"] = [
        {"draft_id": "D0001", "draft": core, "recovery_mode": "audit"}
    ]

    rows, retry_probability = generator._generate_differentiations(
        "Synthetic prompt.", 10
    )

    assert len(services.requests) == 2
    assert len(rows) == 10
    assert retry_probability == Decimal("0.2")


def test_differentiate_probability_math_handles_finite_extreme_weights(
    monkeypatch,
) -> None:
    monkeypatch.setitem(
        _PAIR_GLOBALS,
        "_stage_module",
        lambda stage: SimpleNamespace(STAGE_GOAL="Synthetic stage goal."),
    )
    large = _synthetic_differentiation_response()
    for row in large["candidates"]:
        row["prob"] = 1e308
    mixed = _synthetic_differentiation_response()
    for index, row in enumerate(mixed["candidates"]):
        if index < 6:
            row["identity_relation"] = (
                f"Synthetic invalid relation {index} because it adds an explanation."
            )
            row["prob"] = 1e308
        else:
            row["prob"] = 1e-308

    class Services:
        public_resources = {"active_stage": "directional"}

        def __init__(self) -> None:
            self.responses = [large, mixed]

        def structured_model(self, **request):
            return self.responses.pop(0)

    generator = ModularGenerator(Services())
    core = "A neutral supported core."
    generator.state["current_draft"] = core
    generator.state["insufficient_directional_drafts"] = [
        {"draft_id": "D0001", "draft": core, "recovery_mode": "audit"}
    ]

    large_rows, large_retry = generator._generate_differentiations(
        "Synthetic prompt.", 10
    )
    assert len(large_rows) == 10
    assert large_retry == Decimal("0.2")

    mixed_rows, mixed_retry = generator._generate_differentiations(
        "Synthetic prompt.", 10
    )
    probabilities = _PAIR_GLOBALS["_weighted_probabilities"](
        [row["prob"] for row in mixed_rows],
        retry_mass=str(mixed_retry),
    )
    decimals = tuple(Decimal(value) for value in probabilities)

    assert len(mixed_rows) == 4
    assert Decimal(0) < mixed_retry < Decimal(1)
    assert all(Decimal(0) < value < Decimal(1) for value in decimals)
    assert sum(decimals, Decimal(0)) == Decimal(1)


def test_differentiate_probability_math_preserves_smallest_float_retry(
    monkeypatch,
) -> None:
    monkeypatch.setitem(
        _PAIR_GLOBALS,
        "_stage_module",
        lambda stage: SimpleNamespace(STAGE_GOAL="Synthetic stage goal."),
    )
    response = _synthetic_differentiation_response()
    response["retry_prob"] = 5e-324

    class Services:
        public_resources = {"active_stage": "directional"}

        def structured_model(self, **request):
            return response

    generator = ModularGenerator(Services())
    core = "A neutral supported core."
    generator.state["current_draft"] = core
    generator.state["insufficient_directional_drafts"] = [
        {"draft_id": "D0001", "draft": core, "recovery_mode": "audit"}
    ]

    rows, retry_probability = generator._generate_differentiations(
        "Synthetic prompt.", 10
    )
    probabilities = _PAIR_GLOBALS["_weighted_probabilities"](
        [row["prob"] for row in rows],
        retry_mass=str(retry_probability),
    )
    decimals = tuple(Decimal(value) for value in probabilities)

    assert len(rows) == 10
    assert retry_probability == Decimal(str(5e-324))
    assert all(value > 0 for value in decimals)
    assert sum(decimals, Decimal(0)) == Decimal(1)


def test_three_unusable_differentiation_calls_return_retry_only(
    monkeypatch,
) -> None:
    monkeypatch.setitem(
        _PAIR_GLOBALS,
        "_stage_module",
        lambda stage: SimpleNamespace(
            STAGE_GOAL="Synthetic stage goal.",
            DIFFERENTIATE_COUNT=10,
            DIFFERENTIATE_ACTION="add",
            DIFFERENTIATE_PROMPT="Synthetic differentiation prompt.",
        ),
    )

    class Services:
        public_resources = {"active_stage": "directional"}

        def __init__(self) -> None:
            self.requests = []

        def structured_model(self, **request):
            self.requests.append(request)
            return _synthetic_differentiation_response(invalid_all_relations=True)

    services = Services()
    generator = ModularGenerator(services)
    core = "A neutral supported core."
    generator.state["current_draft"] = core
    generator.state["insufficient_directional_drafts"] = [
        {"draft_id": "D0001", "draft": core, "recovery_mode": "audit"}
    ]

    question = generator._candidate_question("differentiate")

    assert len(services.requests) == 3
    assert [option.option_id for option in question.options] == [
        "differentiate-retry"
    ]
    assert question.options[0].probability == "1"
    assert generator.state["pending_candidate_slate"] == []


def test_differentiate_retry_returns_to_full_mode_eager_dispatch() -> None:
    services = _EagerServices()
    generator = ModularGenerator(services)
    generator.state["current_draft"] = "A neutral supported core."
    generator.state["insufficient_directional_drafts"] = [
        {
            "draft_id": "D0001",
            "draft": "A neutral supported core.",
            "recovery_mode": "audit",
        }
    ]
    generator.state["pending_candidate_slate"] = [
        {
            "channel": "differentiate",
            "label": "synthetic relation",
            "draft": "A neutral supported core. A synthetic relation.",
            "identity_relation": "A synthetic relation",
            "facet_family": "neutral facet",
            "source_draft_ids": ["D0001"],
        }
    ]

    question = generator.step(
        Choice(
            "differentiate-retry",
            public_payload={"kind": "retry", "channel": "differentiate"},
        )
    )

    assert [option.public_payload["mode"] for option in question.options] == ["mc", "keyword", "submit"]
    assert services.requests
    assert generator.state["rejected_candidate_regions"][-1]["channel"] == "differentiate"
    # The buffered rejection event was folded by the state-update turn.
    assert generator.state["pending_events"] == []
    assert generator.state["whiteboard"] == "Synthetic belief state."


def test_differentiate_rejects_an_unsafe_corrected_core_before_model_call() -> None:
    class Services:
        public_resources = {"active_stage": "directional"}

        def __init__(self) -> None:
            self.requests = []

        def structured_model(self, **request):
            self.requests.append(request)
            return _synthetic_differentiation_response()

    services = Services()
    generator = ModularGenerator(services)
    generator.state["current_draft"] = "An unsafe synthetic core."
    generator.state["insufficient_directional_drafts"] = [
        {
            "draft_id": "D0001",
            "draft": "An unsafe synthetic core.",
            "recovery_mode": "correct",
        }
    ]

    with pytest.raises(ValueError, match="current safe Directional core"):
        generator._generate_differentiations("Synthetic prompt.", 10)
    assert services.requests == []


def test_correction_axis_retry_then_paid_differentiate_preserves_safe_core(
    monkeypatch,
) -> None:
    directional_source = (
        _PAIR / "participant/stages/directional.py"
    ).read_text(encoding="utf-8")
    assert "atomic replacement is unsafe as an unchanged component" in directional_source
    assert "rejecting its axis/value slate does not" in directional_source
    monkeypatch.setitem(
        _PAIR_GLOBALS,
        "_stage_module",
        lambda stage: SimpleNamespace(
            STAGE_GOAL="Synthetic stage goal.",
            DIFFERENTIATE_COUNT=10,
            DIFFERENTIATE_ACTION="add",
            DIFFERENTIATE_PROMPT="Synthetic differentiation prompt.",
        ),
    )

    class Services:
        public_resources = {"active_stage": "directional"}

        def __init__(self) -> None:
            self.requests = []

        def structured_model(self, **request):
            self.requests.append(request)
            return _synthetic_differentiation_response()

    services = Services()
    generator = ModularGenerator(services)
    core = "A safe synthetic core."
    generator.state["current_draft"] = core
    generator.state["facts"] = [
        {
            "fact_id": "F0001",
            "text": core,
            "active": True,
            "source": "directional:explore",
        }
    ]
    generator.state["next_fact_index"] = 2
    generator.state["insufficient_directional_drafts"] = [
        {"draft_id": "D0001", "draft": core, "recovery_mode": "audit"}
    ]
    generator.state["next_directional_draft_index"] = 2

    generator.state["pending_correction_axes"] = [
        {
            "axis_id": "CAX01",
            "before_claim": "The draft entails a synthetic claim.",
            "evidence": "safe",
            "source_fact_id": "F0001",
            "prob": 1,
        }
    ]
    generator._record_correction_retry("axis")
    assert generator.state["insufficient_directional_drafts"][0][
        "recovery_mode"
    ] == "audit"
    question = generator.step(
        Choice(
            "mode-differentiate",
            public_payload={"kind": "dispatch", "mode": "differentiate"},
        )
    )

    assert len(question.options) == 11
    assert len(services.requests) == 1
    assert generator.state["insufficient_directional_drafts"][0]["recovery_mode"] == "audit"


def test_correction_retries_record_neutral_paid_evidence() -> None:
    services = ServiceFactory(
        seed=105,
        public_resources={"active_stage": "directional"},
    ).create()
    generator = ModularGenerator(services)
    generator.state["current_draft"] = "A stable synthetic core."
    generator.state["pending_correction_axis"] = {
        "axis_id": "CAX01",
        "before_claim": "The draft entails a synthetic claim.",
        "evidence": "stable",
        "source_fact_id": "F0001",
    }
    generator.state["pending_correction_replacements"] = [
        {"replacement_id": "CREP01", "draft": "revised"}
    ]

    generator._record_correction_retry("replacement")

    assert generator.state["current_draft"] == "A stable synthetic core."
    assert generator.state["rejected_corrections"][-1] == {
        "phase": "replacement",
        "before_claim": {
            "before_claim": "The draft entails a synthetic claim.",
            "evidence": "stable",
            "source_fact_id": "F0001",
        },
        "rejected_repairs": 1,
    }
    assert generator.state["pending_events"][-1] == (
        "Paid correction outcome: no displayed repair was faithful."
    )


def test_actual_differentiate_path_charges_mode_whole_draft_and_submit_without_leak(
    monkeypatch,
) -> None:
    secret_reason = "PRIVATE_SYNTHETIC_JUDGE_DIAGNOSIS"
    monkeypatch.setitem(
        _PAIR_GLOBALS,
        "_stage_module",
        lambda stage: SimpleNamespace(
            STAGE_GOAL="Synthetic stage goal.",
            DIFFERENTIATE_COUNT=10,
            DIFFERENTIATE_ACTION="add",
            DIFFERENTIATE_PROMPT="Synthetic differentiation prompt.",
            ORACLE_POLICY="Synthetic Oracle policy.",
        ),
    )

    class FlowGenerator(ModularGenerator):
        def __init__(self, services):
            super().__init__(services)
            core = "A neutral supported core."
            self.state["category"] = "a theory or analysis result"
            self.state["current_draft"] = core
            self.state["facts"] = [
                {
                    "fact_id": "F0001",
                    "text": core,
                    "active": True,
                    "source": "directional:explore",
                }
            ]
            self.state["insufficient_directional_drafts"] = [
                {
                    "draft_id": "D0001",
                    "draft": core,
                    "recovery_mode": "audit",
                }
            ]
            self.state["next_directional_draft_index"] = 2

        def _directional_dispatch_question(self):
            return _simple_directional_dispatch(self)

        def step(self, value):
            if value is None:
                return self._dispatch_question()
            return super().step(value)

    class FlowOracle:
        def __init__(self, _target, services):
            self.services = services

        def step(self, presented):
            assert isinstance(presented, PresentedQuestion)
            if presented.question.options[0].public_payload.get("kind") == "candidate":
                return Choice("differentiate-1")
            if _snapshot_passes(self.services, presented):
                return Choice("mode-submit")
            return Choice("mode-differentiate")

    class FlowJudge:
        def evaluate(self, _target, ideas):
            return tuple(
                IdeaVerdict(
                    idea.idea_id,
                    "synthetic relation 1" in str(idea.content.get("setting_and_object")),
                    None
                    if "synthetic relation 1" in str(idea.content.get("setting_and_object"))
                    else secret_reason,
                )
                for idea in ideas
            )

    backend = ScriptedModelBackend([
        _synthetic_differentiation_response(),
        {"whiteboard": "Synthetic belief state.", "top_idea": {"setting_and_object": ""}},
    ])
    events = []
    judge = FlowJudge()
    result = ArenaRunner(
        event_sink=events.append,
    ).run(
        generator_factory=ActorFactory(
            FlowGenerator,
            service_factory=ServiceFactory(seed=107, model_backend=backend),
        ),
        oracle_factory=ActorFactory(
            FlowOracle,
            constructor_args=({"summary": {"synthetic": True}},),
            service_factory=ServiceFactory(
                seed=109, judge_call=_judge_call(judge)
            ),
        ),
        target={"summary": {"synthetic": True}},
        judge=judge,
        run_id="synthetic-differentiate-contract",
    )

    assert result.status == "pass"
    charged = [
        event
        for event in events
        if event["kind"] == "choice_cost" and event["information_bits"] > 0
    ]
    assert [event["option_id"] for event in charged] == [
        "mode-differentiate",
        "differentiate-1",
        "mode-submit",
    ]
    expected_k = -__import__("math").log2(0.13 * 0.08 * 0.13 * 0.95**3)
    assert result.k == pytest.approx(expected_k)
    assert secret_reason not in repr(backend.requests)


# ---------------------------------------------------------------------------
# Persistent agent Oracle (ported from the reference pair)
# ---------------------------------------------------------------------------


class _AgentServices:
    public_resources = {"active_stage": "directional", "time_travel_enabled": True}

    judge_reason = "PRIVATE_REASON"

    def __init__(self, outputs) -> None:
        self.requests = []
        self.outputs = list(outputs)

    def structured_model(self, **request):
        self.requests.append(request)
        return self.outputs.pop(0)

    def judge_evaluate(self, ideas):
        return [
            {
                "idea_id": idea["idea_id"],
                "passed": False,
                "private_reason": self.judge_reason,
            }
            for idea in ideas
        ]


def _agent_output(
    action="choose",
    option_id=None,
    question_id=None,
    session="s1",
    state_summary="Synthetic durable state.",
):
    # v1.11: one structured call per turn, so the action is the whole response.
    del session
    return {
        "reasoning": "Synthetic private rationale.",
        "state_summary": state_summary,
        "action": action,
        "option_id": option_id,
        "question_id": question_id,
    }


def _two_option_question(question_id):
    return PresentedQuestion(
        question_id,
        Question(
            "Synthetic finite question.",
            (
                Option("option-a", {"kind": "dispatch", "mode": "mc"}, "0.5"),
                Option("option-b", {"kind": "dispatch", "mode": "audit"}, "0.5"),
            ),
        ),
    )


def test_agent_oracle_rendering_is_canonical_under_journal_round_trip() -> None:
    """Live payloads arrive insertion-ordered; replayed ones decode key-sorted.

    The rendered oracle user text must be byte-identical for both, or the
    agent_turn request hash diverges under replay (the exact failures seen on
    runs 7bf3a459 and 5fa5fdf3)."""

    def scrambled(value):
        if isinstance(value, dict):
            return {k: scrambled(value[k]) for k in reversed(list(value))}
        if isinstance(value, list):
            return [scrambled(item) for item in value]
        return value

    def journaled(value):
        return json.loads(json.dumps(value, ensure_ascii=False, sort_keys=True))

    draft_content = {
        "setting_and_object": "A benchmark comparing optimizers.",
        "findings": [],
        "nested": {"zeta": 1, "alpha": {"b": 2, "a": 3}},
    }
    question = PresentedQuestion(
        "q-0001",
        Question(
            "Pick a route.",
            (
                Option(
                    "mode-mc",
                    {
                        "kind": "dispatch",
                        "mode": "mc",
                        "preview": {
                            "question": "Axis?",
                            "options": [
                                {
                                    "option_id": "mc-1",
                                    "probability": "0.25",
                                    "public_payload": {"axis": "a", "value": "v", "kind": "mc_answer"},
                                }
                            ],
                        },
                    },
                    "0.5",
                ),
                Option(
                    "mode-explore",
                    {
                        "kind": "dispatch",
                        "mode": "explore",
                        "generator_idea_snapshot": {
                            "ideas": [
                                {
                                    "idea_id": "current-draft",
                                    "content": draft_content,
                                    "probability": "1",
                                }
                            ]
                        },
                        "generator_whiteboard": "WB_MARKER belief text.",
                    },
                    "0.5",
                ),
            ),
        ),
    )
    feedback = SubmissionFeedback(
        question,
        Submission((Idea("current-draft", draft_content, "1"),)),
        (IdeaVerdict("current-draft", False, "PRIVATE_GAP"),),
        ("q-0001",),
    )
    target = {"summary": {"title": "T", "groups": [{"aspect": "z", "details": ["d"]}]}}

    def user_texts(target_value, question_value, feedback_value):
        services = _AgentServices(
            [
                _agent_output(option_id="mode-mc", session="s1"),
                _agent_output(action="checkout", question_id="q-0001", session="s1"),
            ]
        )
        oracle = ModularOracle(target_value, services)
        oracle.step(question_value)
        oracle.step(feedback_value)
        return [row["user"] for row in services.requests]

    def rebuild_question(payload_transform):
        options = tuple(
            Option(o.option_id, payload_transform(o.public_payload), o.probability)
            for o in question.question.options
        )
        ideas = (Idea("current-draft", payload_transform(draft_content), "1"),)
        q = PresentedQuestion("q-0001", Question("Pick a route.", options))
        fb = SubmissionFeedback(
            q, Submission(ideas), feedback.verdicts, ("q-0001",)
        )
        return q, fb

    live_q, live_fb = rebuild_question(scrambled)
    replay_q, replay_fb = rebuild_question(journaled)
    live = user_texts(scrambled(target), live_q, live_fb)
    replayed = user_texts(journaled(target), replay_q, replay_fb)
    assert live == replayed
    # The snapshot is named once by the protocol notes and never rendered per
    # option; the Oracle reads the draft through its own Judge call instead.
    assert live[0].count("generator_idea_snapshot") == 1
    # The whiteboard renders exactly once, in its own section, not per option.
    assert live[0].count("WB_MARKER belief text.") == 1
    assert "generator's whiteboard" in live[0]


def test_oracle_rebuilds_its_whole_context_on_every_turn() -> None:
    """No turn inherits anything the Oracle did not write down itself.

    v1.10 kept a CLI conversation and restarted it every twelve turns. The CLI
    also compacted that conversation on its own, losing 40-84% of it as often
    inside a session as at a restart, so what survived between turns was not
    ours to decide. Every turn now carries MATCH_INIT verbatim and the Oracle's
    own last state_summary, in the same words v1.10 used when it restarted.
    """

    turns = 4
    services = _AgentServices([
        _agent_output(option_id="option-a", state_summary=f"summary after turn {i + 1}")
        for i in range(turns)
    ])
    oracle = ModularOracle({"summary": {"synthetic": True}}, services)
    for index in range(turns):
        oracle.step(_two_option_question(f"q-{index:04d}"))

    assert len(services.requests) == turns
    assert not any("session_id" in request for request in services.requests)
    for index, request in enumerate(services.requests):
        user = request["user"]
        assert "<MATCH_INIT>" in user
        assert _PAIR_GLOBALS["_stage_criterion"]("directional") in user
        if index == 0:
            assert "<RESUMED_PRIVATE_STATE>" not in user
        else:
            assert "<RESUMED_PRIVATE_STATE>" in user
            assert f"summary after turn {index}" in user
            assert f"summary after turn {index + 1}" not in user


def test_agent_oracle_match_init_carries_gold_criterion_and_dispatch_notes() -> None:
    services = _AgentServices(
        [
            _agent_output(option_id="option-b"),
            _agent_output(option_id="option-a", session="s1"),
        ]
    )
    oracle = ModularOracle({"summary": {"synthetic": "PRIVATE_GOLD_MARKER"}}, services)

    first = oracle.step(_two_option_question("q-0001"))

    assert first.option_id == "option-b"
    init_request = services.requests[0]
    assert "session_id" not in init_request
    assert init_request["schema_name"] == "oracle_action"
    assert "persistent multi-turn conversation" in init_request["developer"]
    user = init_request["user"]
    assert "<MATCH_INIT>" in user
    assert "PRIVATE_GOLD_MARKER" in user
    assert _PAIR_GLOBALS["_stage_criterion"]("directional") in user
    assert "<MODULAR_DISPATCH_NOTES>" in user
    assert "676-way two-letter prefix reveal" in user
    assert "eligible_checkout_questions" in user

    second = oracle.step(_two_option_question("q-0002"))

    assert second.option_id == "option-a"
    followup = services.requests[1]
    # the follow-up repeats MATCH_INIT rather than relying on a session for it
    assert "<MATCH_INIT>" in followup["user"]
    assert "<RESUMED_PRIVATE_STATE>" in followup["user"]
    body = followup["user"].split("<ARENA_EVENT>\n", 1)[1].rsplit("\n</ARENA_EVENT>", 1)[0]
    assert body.startswith("=== ARENA EVENT: presented_question ===")
    assert "question_id: q-0002" in body
    checkout_section = body.split("eligible_checkout_questions", 1)[1]
    assert "- q-0001: Synthetic finite question." in checkout_section


def test_agent_oracle_validates_actions_against_displayed_options() -> None:
    services = _AgentServices([_agent_output(option_id="option-nonexistent")])
    oracle = ModularOracle({"summary": {"synthetic": True}}, services)
    with pytest.raises(ValueError, match="unavailable option ID"):
        oracle.step(_two_option_question("q-0001"))

    services = _AgentServices(
        [
            _agent_output(option_id="option-a"),
            _agent_output(action="checkout", question_id="q-elsewhere", session="s1"),
        ]
    )
    oracle = ModularOracle({"summary": {"synthetic": True}}, services)
    oracle.step(_two_option_question("q-0001"))
    with pytest.raises(ValueError, match="unavailable checkout ID"):
        oracle.step(_two_option_question("q-0002"))

    services = _AgentServices(
        [
            _agent_output(option_id="option-a"),
            _agent_output(option_id="option-a", session="s1"),
            _agent_output(action="checkout", question_id="q-0001", session="s1"),
        ]
    )
    oracle = ModularOracle({"summary": {"synthetic": True}}, services)
    oracle.step(_two_option_question("q-0001"))
    oracle.step(_two_option_question("q-0002"))
    rewind = oracle.step(_two_option_question("q-0003"))
    assert isinstance(rewind, Checkout)
    assert rewind.question_id == "q-0001"


def test_agent_oracle_does_not_hide_judge_preview_failures() -> None:
    class FailingJudgeServices(_AgentServices):
        def judge_evaluate(self, ideas):
            raise RuntimeError("synthetic Judge transport failure")

    services = FailingJudgeServices([_agent_output(option_id="mode-mc")])
    oracle = ModularOracle({"summary": {"synthetic": True}}, services)
    snapshot = {
        "ideas": [
            {
                "idea_id": "current-draft",
                "content": {"setting_and_object": "A draft.", "findings": []},
                "probability": "1",
            }
        ]
    }
    presented = PresentedQuestion(
        "q-preview-failure",
        Question(
            "Pick one route.",
            (
                Option(
                    "mode-mc",
                    {
                        "kind": "dispatch",
                        "mode": "mc",
                        "generator_idea_snapshot": snapshot,
                    },
                    "1",
                ),
            ),
        ),
    )

    with pytest.raises(RuntimeError, match="synthetic Judge transport failure"):
        oracle.step(presented)

    assert services.requests == []


def test_agent_oracle_maps_a_unique_exact_preview_row_to_its_dispatch_route() -> None:
    services = _AgentServices([_agent_output(option_id="keyword-category-15")])
    oracle = ModularOracle({"summary": {"synthetic": True}}, services)
    presented = PresentedQuestion(
        "q-dispatch",
        Question(
            "Pick one route.",
            (
                Option(
                    "mode-mc",
                    {
                        "kind": "dispatch",
                        "mode": "mc",
                        "preview": {
                            "question": "Pick an axis value.",
                            "options": [
                                {"option_id": "mc-1", "kind": "ordinary"}
                            ],
                        },
                    },
                    "0.5",
                ),
                Option(
                    "mode-keyword",
                    {
                        "kind": "dispatch",
                        "mode": "keyword",
                        "preview": {
                            "question": "Pick a category.",
                            "options": [
                                {
                                    "option_id": "keyword-category-15",
                                    "kind": "ordinary",
                                }
                            ],
                        },
                    },
                    "0.5",
                ),
            ),
        ),
    )

    choice = oracle.step(presented)

    assert isinstance(choice, Choice)
    assert choice.option_id == "mode-keyword"


def test_agent_oracle_routes_decisions_through_agent_turn() -> None:
    """--oracle-agent (human/claude-code/codex) must actually be consulted.

    The CLI swaps the oracle entrypoint to participant.oracle:AgentOracle and
    injects an agent-session backend; the participant reaches that backend
    only through services.agent_turn. v1.11-v1.13 left AgentOracle as a bare
    alias of Oracle, so a selected human or CLI oracle silently fell back to
    the ordinary model provider.
    """

    from dataclasses import replace

    _, agent_oracle_cls = load_participant_classes(
        replace(load_manifest(_PAIR), oracle="participant.oracle:AgentOracle")
    )

    class AgentTurnServices(_AgentServices):
        def __init__(self, outputs):
            super().__init__(outputs)
            self.agent_requests = []

        def structured_model(self, **request):
            raise AssertionError("AgentOracle must not call structured_model")

        def agent_turn(self, **request):
            self.agent_requests.append(request)
            return {"session_id": "human", "output": self.outputs.pop(0)}

    services = AgentTurnServices(
        [
            _agent_output(option_id="option-a"),
            _agent_output(option_id="option-b"),
        ]
    )
    oracle = agent_oracle_cls({"summary": {"synthetic": True}}, services)
    first = oracle.step(_two_option_question("q-0001"))
    second = oracle.step(_two_option_question("q-0002"))
    assert isinstance(first, Choice) and first.option_id == "option-a"
    assert isinstance(second, Choice) and second.option_id == "option-b"
    assert len(services.agent_requests) == 2
    for request in services.agent_requests:
        assert request["schema_name"] == "oracle_action"
        # One fresh backend session per turn: the full context rides the
        # prompt, so no auto-compacting CLI conversation decides what the
        # Oracle remembers, and a human gets one self-contained request file.
        assert request["session_id"] is None
        assert "<MATCH_INIT>" in request["user"]
        assert "persistent multi-turn conversation" in request["system_prompt"]


def test_agent_oracle_stage_transition_is_service_free_and_updates_criterion() -> None:
    services = _AgentServices(
        [
            _agent_output(option_id="option-a"),
            _agent_output(option_id="option-b", session="s1"),
        ]
    )
    oracle = ModularOracle({"summary": {"synthetic": True}}, services)
    oracle.step(_two_option_question("q-0001"))
    request_count = len(services.requests)

    ready = oracle.step(StageTransition("directional", "essence"))

    assert isinstance(ready, StageReady)
    assert ready.stage == "essence"
    assert len(services.requests) == request_count

    oracle.step(_two_option_question("q-0002"))
    followup = services.requests[-1]["user"]
    assert "=== ARENA EVENT: stage_transition" in followup
    assert _PAIR_GLOBALS["_stage_criterion"]("essence") in followup

    with pytest.raises(ValueError, match="does not match"):
        oracle.step(StageTransition("directional", "strict"))


def test_oracle_refreshes_preview_and_blocks_unchanged_failure_after_promotion() -> None:
    """A Directional PASS must never authorize the same draft in Essence.

    This reproduces the GLM-5V loop: promotion judges the carried draft under
    Essence, recovery checks out to the historical Directional dispatch, and
    the Oracle sees that exact snapshot again. The preview must be refreshed
    for Essence and the failed draft's submit option must be unavailable.
    """

    class StagePreviewServices(_AgentServices):
        def __init__(self, outputs):
            super().__init__(outputs)
            self.judge_calls = 0

        def judge_evaluate(self, ideas):
            self.judge_calls += 1
            passed = self.judge_calls == 1
            reason = (
                "DIRECTIONAL_PREVIEW_PASS"
                if passed
                else "ESSENCE_PREVIEW_FAIL"
            )
            return [
                {
                    "idea_id": idea["idea_id"],
                    "passed": passed,
                    "private_reason": reason,
                }
                for idea in ideas
            ]

    services = StagePreviewServices(
        [
            _agent_output(option_id="mode-submit"),
            _agent_output(action="checkout", question_id="q-directional"),
            _agent_output(option_id="mode-keyword"),
        ]
    )
    oracle = ModularOracle({"summary": {"synthetic": True}}, services)
    content = {
        "setting_and_object": "A carried Directional draft.",
        "findings": [],
    }
    snapshot = {
        "ideas": [
            {
                "idea_id": "current-draft",
                "content": content,
                "probability": "1",
            }
        ]
    }
    payload = {
        "kind": "dispatch",
        "generator_idea_snapshot": snapshot,
    }
    presented = PresentedQuestion(
        "q-directional",
        Question(
            "Pick a recovery route.",
            (
                Option("mode-keyword", {**payload, "mode": "keyword"}, "0.5"),
                SubmitOption("mode-submit", {**payload, "mode": "submit"}, "0.5"),
            ),
        ),
    )

    first = oracle.step(presented)
    assert isinstance(first, Choice) and first.option_id == "mode-submit"
    assert services.judge_calls == 1
    assert "DIRECTIONAL_PREVIEW_PASS" in services.requests[0]["user"]

    ready = oracle.step(StageTransition("directional", "essence"))
    assert ready.stage == "essence"
    feedback = SubmissionFeedback(
        presented,
        Submission((Idea("current-draft", content, "1"),)),
        (IdeaVerdict("current-draft", False, "ESSENCE_FORMAL_FAIL"),),
        ("q-directional",),
    )
    recovery = oracle.step(feedback)
    assert isinstance(recovery, Checkout)

    next_choice = oracle.step(presented)
    assert isinstance(next_choice, Choice)
    assert next_choice.option_id == "mode-keyword"
    assert services.judge_calls == 2
    replay_prompt = services.requests[-1]["user"]
    assert "=== ACTIVE JUDGE (stage: essence) ===" in replay_prompt
    assert "ESSENCE_PREVIEW_FAIL" in replay_prompt
    assert "DIRECTIONAL_PREVIEW_PASS" not in replay_prompt
    assert "\n- option_id=mode-keyword" in replay_prompt
    assert "\n- option_id=mode-submit" not in replay_prompt
    assert "submit is unavailable: this exact draft already failed" in replay_prompt


def test_agent_oracle_prefixed_transition_before_first_turn_sets_match_init_stage() -> None:
    class EssenceServices(_AgentServices):
        public_resources = {"active_stage": "essence", "time_travel_enabled": True}

    services = EssenceServices([_agent_output(option_id="option-a")])
    oracle = ModularOracle({"summary": {"synthetic": True}}, services)
    ready = oracle.step(StageTransition("essence", "strict"))

    assert ready.stage == "strict"
    oracle.step(_two_option_question("q-0001"))
    user = services.requests[0]["user"]
    assert "=== ACTIVE JUDGE (stage: strict) ===" in user
    assert _PAIR_GLOBALS["_stage_criterion"]("strict") in user
    assert '"event_type":"stage_transition"' not in user


def test_agent_oracle_receives_judge_preview_and_feedback_events() -> None:
    class PreviewServices(_AgentServices):
        judge_reason = "PRIVATE_PREVIEW_REASON"

    services = PreviewServices(
        [
            _agent_output(option_id="option-a"),
            _agent_output(action="checkout", question_id="q-0001", session="s1"),
        ]
    )
    oracle = ModularOracle({"summary": {"synthetic": True}}, services)
    base = _two_option_question("q-0001").question
    snapshot = {
        "ideas": [
            {
                "idea_id": "current-draft",
                "content": {"draft": "A synthetic core."},
                "probability": "1",
            }
        ]
    }
    options = tuple(
        Option(
            option.option_id,
            {**option.public_payload, "generator_idea_snapshot": snapshot},
            option.probability,
        )
        for option in base.options
    )
    presented = PresentedQuestion(
        "q-0001", Question(base.question, options)
    )
    oracle.step(presented)

    user = services.requests[0]["user"]
    assert "PRIVATE_PREVIEW_REASON" in user
    assert "FAIL -- PRIVATE_PREVIEW_REASON" in user

    feedback = SubmissionFeedback(
        presented,
        Submission((Idea("current-draft", {"draft": "A synthetic core."}, "1"),)),
        (IdeaVerdict("current-draft", False, "PRIVATE_JUDGE_DIAGNOSIS"),),
        ("q-0001",),
    )
    recovery = oracle.step(feedback)

    assert isinstance(recovery, Checkout)
    assert recovery.question_id == "q-0001"
    feedback_user = services.requests[1]["user"]
    assert "=== ARENA EVENT: submission_feedback" in feedback_user
    assert "PRIVATE_JUDGE_DIAGNOSIS" in feedback_user
    assert "valid_checkout_question_ids" in feedback_user
    assert "- q-0001: Synthetic finite question." in feedback_user.split(
        "valid_checkout_question_ids", 1
    )[1]


def test_oracle_requires_reasoning_and_state_summary() -> None:
    """state_summary is the only memory now, so an empty one is fatal."""

    for field in ("state_summary", "reasoning"):
        output = _agent_output(option_id="option-a")
        output[field] = ""
        services = _AgentServices([output])
        oracle = ModularOracle({"summary": {"synthetic": True}}, services)
        with pytest.raises(ValueError, match="reasoning or state_summary"):
            oracle.step(_two_option_question("q-0001"))


def test_keyword_reveal_guess_and_character_loop_follows_the_v0_4_rhythm() -> None:
    generator, services = _eager_generator_with_draft()
    dispatch = generator._dispatch_question()
    keyword_mode = next(
        option for option in dispatch.options if option.option_id == "mode-keyword"
    )
    keyword_question = _buy_first_category(
        generator,
        generator.step(
            Choice(keyword_mode.option_id, public_payload=keyword_mode.public_payload)
        ),
    )
    validate_question(keyword_question)
    # Direct slate: 64 candidate phrases plus the paid reveal entry and retry.
    keyword_count = _CHANNEL_GLOBALS["DIRECTIONAL_KEYWORD_COUNT"]
    assert len(keyword_question.options) == keyword_count + 2
    request_prefix = next(
        option
        for option in keyword_question.options
        if option.option_id == "keyword-prefix-request"
    )
    request_count = len(services.requests)
    reveal = generator.step(
        Choice(request_prefix.option_id, public_payload=request_prefix.public_payload)
    )
    validate_question(reveal)

    # The 676-way two-letter reveal is static: no model call, and every prefix
    # is priced from the displayed candidate slate's implied distribution.
    assert len(services.requests) == request_count
    assert len(reveal.options) == 676
    assert all(
        option.public_payload["kind"] == "keyword_prefix_reveal"
        for option in reveal.options
    )
    reveal_ne = next(
        option for option in reveal.options if option.option_id == "keyword-reveal-ne"
    )
    guesses = generator.step(
        Choice(reveal_ne.option_id, public_payload=reveal_ne.public_payload)
    )
    validate_question(guesses)

    # One Generator call per paid reveal: 16 completions plus the extend exit.
    guess_count = _CHANNEL_GLOBALS["DIRECTIONAL_KEYWORD_GUESS_COUNT"]
    assert len(services.requests) == request_count + 1
    assert len(guesses.options) == guess_count + 1
    assert all(
        option.public_payload["term"].casefold().startswith("ne")
        for option in guesses.options
        if option.public_payload["kind"] == "keyword_guess"
    )
    extend = guesses.options[-1]
    assert extend.public_payload == {"kind": "keyword_guess_extend", "prefix": "ne"}

    # A missed guess slate is priced negative evidence for this exact prefix;
    # the channel stays bound and moves to the frequency-weighted character
    # Choice without another model call.
    request_count = len(services.requests)
    characters = generator.step(
        Choice(extend.option_id, public_payload=extend.public_payload)
    )
    validate_question(characters)

    assert len(services.requests) == request_count
    assert generator.state["rejected_keywords"][-1] == {
        "phase": "guess",
        "prefix": "ne",
        "phrases": [
            option.public_payload["term"]
            for option in guesses.options
            if option.public_payload["kind"] == "keyword_guess"
        ],
    }
    assert generator.state["active_dispatch_binding"] is not None
    # 26 letters + 10 digits + space + hyphen, plus accept and abandon.
    assert len(characters.options) == 40
    accept = next(
        option
        for option in characters.options
        if option.public_payload["kind"] == "keyword_prefix_accept"
    )
    assert accept.option_id == "keyword-accept-2"
    assert accept.public_payload["prefix"] == "ne"

    # Paying the next character immediately re-arms one fresh guess slate for
    # the extended exact prefix; selecting a guess commits exactly one DIRECT
    # identity fact and closes the search.
    character_u = next(
        option
        for option in characters.options
        if option.option_id == "keyword-hint-3-u"
    )
    assert character_u.public_payload == {
        "kind": "keyword_hint_character",
        "prefix": "neu",
    }
    rearmed = generator.step(
        Choice(character_u.option_id, public_payload=character_u.public_payload)
    )

    assert len(services.requests) == request_count + 1
    selected = next(
        option
        for option in rearmed.options
        if option.public_payload["kind"] == "keyword_guess"
    )
    assert selected.public_payload["term"].casefold().startswith("neu")
    generator.step(Choice(selected.option_id, public_payload=selected.public_payload))

    facts = [row for row in generator.state["facts"] if row["active"]]
    assert facts[-1]["text"] == f"Central identity phrase: {selected.public_payload['term']}"
    assert generator.state["active_dispatch_binding"] is None

def test_time_travel_enabled_removes_priced_abandon_rows() -> None:
    max_chars = _CHANNEL_GLOBALS["DIRECTIONAL_KEYWORD_PREFIX_MAX_CHARS"]
    generator, _services = _eager_generator_with_draft()
    generator.omit_keyword_abandon = True
    generator.state["active_dispatch_binding"] = {
        "bundle_id": "synthetic",
        "source_stage": "directional",
        "source_draft": generator.state["current_draft"],
        "fact_ledger_hash": generator._fact_ledger_hash(),
        "mode": "keyword",
    }

    # Character slate: accept stays, priced abandon disappears.
    chars = _CHANNEL_GLOBALS["hint_character_question"](generator, "ne")
    ids = [option.option_id for option in chars.options]
    assert any(option_id.startswith("keyword-accept-") for option_id in ids)
    assert not any("abandon" in option_id for option_id in ids)
    assert "checkout" in chars.question

    # Guess slate at the bound: no tail row at all; checkout is the exit.
    full = "n" * max_chars
    bounded = _CHANNEL_GLOBALS["guess_question"](generator, full)
    bounded_ids = [option.option_id for option in bounded.options]
    assert not any("abandon" in option_id for option_id in bounded_ids)
    assert all(option_id.startswith("keyword-guess-") for option_id in bounded_ids)

def test_keyword_prefix_search_is_bounded_and_abandonment_rejects_the_channel() -> None:
    max_chars = _CHANNEL_GLOBALS["DIRECTIONAL_KEYWORD_PREFIX_MAX_CHARS"]
    generator, _services = _eager_generator_with_draft()
    generator.state["active_dispatch_binding"] = {
        "bundle_id": "synthetic",
        "source_stage": "directional",
        "source_draft": generator.state["current_draft"],
        "fact_ledger_hash": generator._fact_ledger_hash(),
        "mode": "keyword",
    }
    full = "n" * max_chars
    bounded = _CHANNEL_GLOBALS["guess_question"](generator, full)

    # At the bound the guess slate's exit row is the explicitly priced abandon
    # instead of extend-by-one-character, and no further character exists.
    assert bounded.options[-1].option_id == "keyword-guess-abandon"
    assert bounded.options[-1].public_payload == {
        "kind": "keyword_prefix_abandon",
        "prefix": full,
    }
    with pytest.raises(ValueError, match="bound"):
        _CHANNEL_GLOBALS["hint_character_question"](generator, full)

    # Abandonment at the bound is a channel-level rejection returning to
    # dispatch, exactly like the direct-slate retry.
    exit_question = generator.step(
        Choice(
            bounded.options[-1].option_id,
            public_payload=bounded.options[-1].public_payload,
        )
    )
    assert exit_question.options[0].public_payload["kind"] == "dispatch"
    assert generator.state["rejected_keywords"][-1] == {
        "phase": "prefix_abandon",
        "prefix": full,
        "phrases": [],
    }
    assert generator.state["active_dispatch_binding"] is None

    # Abandonment anywhere in the loop is the same channel-level rejection.
    generator, _services = _eager_generator_with_draft()
    generator.state["active_dispatch_binding"] = {
        "bundle_id": "synthetic",
        "source_stage": "directional",
        "source_draft": generator.state["current_draft"],
        "fact_ledger_hash": generator._fact_ledger_hash(),
        "mode": "keyword",
    }
    abandoned = generator.step(
        Choice(
            "keyword-abandon-2",
            public_payload={"kind": "keyword_prefix_abandon", "prefix": "ne"},
        )
    )
    assert abandoned.options[0].public_payload["kind"] == "dispatch"
    assert generator.state["rejected_keywords"][-1] == {
        "phase": "prefix_abandon",
        "prefix": "ne",
        "phrases": [],
    }
    assert generator.state["active_dispatch_binding"] is None

def test_keyword_prefix_validation_bounds_the_search_alphabet() -> None:
    validate_prefix = _CHANNEL_GLOBALS["_validate_keyword_prefix"]

    with pytest.raises(ValueError, match="two-letter reveal"):
        validate_prefix("n")
    with pytest.raises(ValueError, match="bounded length"):
        validate_prefix(
            "n" * (_CHANNEL_GLOBALS["DIRECTIONAL_KEYWORD_PREFIX_MAX_CHARS"] + 1)
        )
    with pytest.raises(ValueError, match="two lowercase letters"):
        validate_prefix("1e")
    with pytest.raises(ValueError, match="invalid character"):
        validate_prefix("ne!")
    with pytest.raises(ValueError, match="consecutive spaces"):
        validate_prefix("ne  a")

    # A trailing separator never offers a second consecutive space; the exact
    # spelled prefix stays acceptable as the complete phrase.
    generator, _services = _eager_generator_with_draft()
    generator.state["active_dispatch_binding"] = {
        "bundle_id": "synthetic",
        "source_stage": "directional",
        "source_draft": generator.state["current_draft"],
        "fact_ledger_hash": generator._fact_ledger_hash(),
        "mode": "keyword",
    }
    question = _CHANNEL_GLOBALS["hint_character_question"](generator, "ne ")

    kinds = [option.public_payload["kind"] for option in question.options]
    assert kinds.count("keyword_hint_character") == 37  # space is suppressed
    assert kinds[-2:] == ["keyword_prefix_accept", "keyword_prefix_abandon"]
    assert not any(
        option.public_payload["kind"] == "keyword_hint_character"
        and option.public_payload["prefix"].endswith("  ")
        for option in question.options
    )
    assert any(option.option_id == "keyword-hint-4-hyphen" for option in question.options)

def test_keyword_drops_malformed_or_duplicate_rows_from_the_direct_slate() -> None:
    keyword_count = _CHANNEL_GLOBALS["DIRECTIONAL_KEYWORD_COUNT"]

    class Services(_EagerServices):
        def structured_model(self, **request):
            self.requests.append(request)
            response = _synthetic_eager_response(request)
            if request["schema_name"] == "directional_identity_keywords":
                response["keywords"][0]["term"] = "duplicate identity"
                response["keywords"][1]["term"] = "DUPLICATE IDENTITY"
                response["keywords"][2]["term"] = "compound sentence. extra claim"
                response["keywords"][3]["term"] = "line\nbreak"
            return response

    generator = ModularGenerator(Services())
    generator.state["category"] = "a theory or analysis result"
    generator.state["current_draft"] = "A neutral supported core."
    generator.state["insufficient_directional_drafts"] = [
        {
            "draft_id": "D0001",
            "draft": "A neutral supported core.",
            "recovery_mode": "audit",
        }
    ]
    dispatch = generator._dispatch_question()
    keyword_mode = next(
        option for option in dispatch.options if option.option_id == "mode-keyword"
    )
    question = _buy_first_category(
        generator,
        generator.step(
            Choice(keyword_mode.option_id, public_payload=keyword_mode.public_payload)
        ),
    )
    probabilities = [Decimal(option.probability) for option in question.options]

    # 61 surviving phrases plus the paid reveal entry and retry.
    assert len(question.options) == keyword_count - 3 + 2
    assert question.options[-2].option_id == "keyword-prefix-request"
    assert question.options[-1].option_id == "keyword-retry"
    # Every surviving phrase declared the same weight, so the displayed direct
    # probabilities are equal and strictly positive.
    direct = probabilities[:-2]
    assert len(set(direct)) == 1
    assert direct[0] > 0
    # The reveal entry carries the summed per-prefix mass; the retry row keeps
    # only the smoothed all-incorrect mass, below one direct candidate.
    assert probabilities[-2] > direct[0]
    assert Decimal(0) < probabilities[-1] < direct[0]
    with localcontext() as context:
        context.prec = 1000
        assert sum(probabilities, Decimal(0)) == Decimal(1)
    public = repr(question).casefold()
    assert "extra claim" not in public
    assert "line" not in public


def test_keyword_guess_slate_drops_control_characters_and_prefix_mismatches() -> None:
    guess_count = _CHANNEL_GLOBALS["DIRECTIONAL_KEYWORD_GUESS_COUNT"]

    class Services(_EagerServices):
        def structured_model(self, **request):
            self.requests.append(request)
            response = _synthetic_eager_response(request)
            response["guesses"][0]["term"] = "ne\x00utral identity"
            response["guesses"][1]["term"] = "wrong prefix identity"
            return response

    services = Services()
    generator = ModularGenerator(services)
    generator.state["active_dispatch_binding"] = {
        "bundle_id": "synthetic",
        "source_stage": "directional",
        "source_draft": "",
        "fact_ledger_hash": generator._fact_ledger_hash(),
        "mode": "keyword",
    }
    question = _CHANNEL_GLOBALS["guess_question"](generator, "ne")
    probabilities = [Decimal(option.probability) for option in question.options]

    # 14 surviving completions plus the extend-by-one-character exit.
    assert len(question.options) == guess_count - 2 + 1
    assert "\x00" not in repr(question)
    assert "wrong prefix" not in repr(question)
    assert question.options[-1].public_payload["kind"] == "keyword_guess_extend"
    assert all(
        option.public_payload["term"].casefold().startswith("ne")
        for option in question.options
        if option.public_payload["kind"] == "keyword_guess"
    )
    with localcontext() as context:
        context.prec = 1000
        assert sum(probabilities, Decimal(0)) == Decimal(1)


def test_draft_length_is_bounded_only_by_what_the_wire_can_carry() -> None:
    """One ceiling, derived from the protocol, identical at every stage.

    Stages used to own their budgets (2400 / 3600 / 4800) and anything longer
    was discarded, so a fact the Oracle had paid for could stop short of the
    text the Judge reads. The remaining bound exists only so a slate cannot
    exceed the Arena's 262,144-byte message limit.
    """

    max_draft_chars = _PAIR_GLOBALS["_max_draft_chars"]
    budgets = {stage: max_draft_chars(stage) for stage in ("directional", "essence", "strict")}
    assert len(set(budgets.values())) == 1
    bound = budgets["directional"]
    assert bound == _PAIR_GLOBALS["MAX_DRAFT_CHARS"]
    assert bound > 4_800  # larger than every budget it replaced
    assert bound * 12 < 262_144  # a full slate still fits one message
    assert "_max_draft_words" not in _PAIR_GLOBALS



def test_a_long_draft_starves_nothing_and_no_paid_fact_is_dropped(
    monkeypatch,
) -> None:
    """Regression: length must never silently remove content or options.

    A draft that outgrew its stage budget used to lose two things in silence.
    Differentiate collapsed to retry-only, because every appended relation
    pushed the result past the cap. And a state update that produced a longer
    draft was discarded, so a fact the Oracle had already bought -- in one
    Utonia run "rotary position embedding", at 7.24 bits -- reached the ledger
    and the whiteboard but never the draft the Judge reads, with nothing to
    signal it. Neither gate exists now.
    """

    monkeypatch.setitem(
        _PAIR_GLOBALS,
        "_stage_module",
        lambda stage: SimpleNamespace(
            STAGE_GOAL="Synthetic stage goal.",
            DIFFERENTIATE_COUNT=10,
            DIFFERENTIATE_ACTION="add",
            DIFFERENTIATE_PROMPT="Synthetic differentiation prompt.",
        ),
    )

    class Services:
        public_resources = {"active_stage": "directional"}

        def structured_model(self, **request):
            return _synthetic_differentiation_response()

    generator = ModularGenerator(Services())
    filler = "A neutral supported core sentence that keeps growing longer."
    core = " ".join([filler] * 41)[:2395].rsplit(" ", 1)[0] + "."
    assert 2_360 < len(core) <= 2_400  # the length that used to starve it
    generator.state["current_draft"] = core
    generator.state["facts"] = [
        {
            "fact_id": "F0001",
            "text": core,
            "active": True,
            "source": "directional:explore",
        }
    ]
    generator.state["next_fact_index"] = 2

    slate = generator.step(
        Choice(
            "mode-differentiate",
            public_payload={"kind": "dispatch", "mode": "differentiate"},
        )
    )
    assert len(slate.options) == 11
    assert slate.options[-1].option_id == "differentiate-retry"

    # and a state update that grows the draft past the old cap is kept
    longer = core + " " + " ".join([filler] * 8)
    assert len(longer) > 2_400
    generator.state["pending_events"] = ["synthetic event"]
    generator.services = SimpleNamespace(
        public_resources={"active_stage": "directional"},
        structured_model=lambda **_r: {
            "whiteboard": "synthetic whiteboard",
            "top_idea": {"setting_and_object": longer},
        },
    )
    generator._run_state_update()
    assert generator.state["current_draft"] == longer



STAGES_WITH_CATEGORY_KEYWORD = ("directional", "essence", "strict")


def _stage_generator_with_draft(stage: str):
    class StageServices(_EagerServices):
        public_resources = {"active_stage": stage}

    services = StageServices()
    generator = ModularGenerator(services)
    generator.state["category"] = "a new algorithm or method"
    generator.state["current_draft"] = f"A neutral supported {stage} core."
    generator.state["insufficient_directional_drafts"] = [
        {
            "draft_id": "D0001",
            "draft": f"A neutral supported {stage} core.",
            "recovery_mode": "audit",
        }
    ]
    return generator, services


@pytest.mark.parametrize("stage", STAGES_WITH_CATEGORY_KEYWORD)
def test_keyword_is_category_first_then_terms_then_prefix_fallback(stage: str) -> None:
    generator, services = _stage_generator_with_draft(stage)
    channels = vars(_PAIR_GLOBALS["_stage_channels"](stage))
    dispatch = generator._dispatch_question()
    keyword_mode = next(
        option for option in dispatch.options if option.option_id == "mode-keyword"
    )
    categories = generator.step(
        Choice(keyword_mode.option_id, public_payload=keyword_mode.public_payload)
    )
    validate_question(categories)

    # The preview is the category slate alone: no phrase rows are built yet.
    category_count = channels["KEYWORD_CATEGORY_COUNT"]
    assert len(categories.options) == category_count + 1
    assert all(
        option.public_payload["kind"] == "keyword_category"
        for option in categories.options[:-1]
    )
    assert categories.options[-1].public_payload == {"kind": "keyword_category_retry"}
    assert sum(float(option.probability) for option in categories.options) == pytest.approx(1)
    # Dispatch restores only the state the shared kernel knows, so the category
    # rows ride the keyword slate under an explicit marker.
    assert all(row["is_category"] for row in generator.state["pending_keyword_slate"])
    assert f"{stage}_keyword_categories" in {
        request["schema_name"] for request in services.requests
    }

    # Buying a category commits it as an ordinary fact and only then generates
    # the unchanged v0.4 phrase slate.
    chosen = categories.options[0]
    category = chosen.public_payload["category"]
    request_count = len(services.requests)
    terms = generator.step(
        Choice(chosen.option_id, public_payload=chosen.public_payload)
    )
    validate_question(terms)
    assert len(services.requests) == request_count + 1
    term_request = services.requests[-1]
    assert term_request["schema_name"] == "directional_identity_keywords"
    # The conditioning travels through the fact ledger, not through a new
    # prompt: the stock v0.4 developer text is unchanged, and the category
    # reaches the model in the rendered generator context.
    assert category not in term_request["developer"]
    assert category in term_request["user"]
    active = [row for row in generator.state["facts"] if row["active"]]
    assert active[-1]["text"] == f"Missing defining detail lies in: {category}"

    term_count = channels["DIRECTIONAL_KEYWORD_COUNT"]
    assert len(terms.options) == term_count + 2
    assert terms.options[-2].option_id == "keyword-prefix-request"
    assert terms.options[-1].option_id == "keyword-retry"
    assert sum(float(option.probability) for option in terms.options) == pytest.approx(1)
    assert len(generator.state["pending_keyword_slate"]) == term_count
    assert not any(
        row.get("is_category") for row in generator.state["pending_keyword_slate"]
    )

    # The v0.4 prefix machinery still runs verbatim, and the completion call
    # inherits the same category conditioning through the ledger.
    request_prefix = terms.options[-2]
    reveal = generator.step(
        Choice(request_prefix.option_id, public_payload=request_prefix.public_payload)
    )
    validate_question(reveal)
    assert len(reveal.options) == 676
    reveal_ne = next(
        option for option in reveal.options if option.option_id == "keyword-reveal-ne"
    )
    guesses = generator.step(
        Choice(reveal_ne.option_id, public_payload=reveal_ne.public_payload)
    )
    validate_question(guesses)
    assert services.requests[-1]["schema_name"] == "directional_keyword_guesses"
    assert category in services.requests[-1]["user"]
    guess = next(
        option
        for option in guesses.options
        if option.public_payload["kind"] == "keyword_guess"
    )
    generator.step(Choice(guess.option_id, public_payload=guess.public_payload))
    active = [row for row in generator.state["facts"] if row["active"]]
    assert active[-1]["text"] == (
        f"Central identity phrase: {guess.public_payload['term']}"
    )


@pytest.mark.parametrize("stage", STAGES_WITH_CATEGORY_KEYWORD)
def test_keyword_term_choice_cannot_resolve_against_a_category_row(stage: str) -> None:
    generator, _ = _stage_generator_with_draft(stage)
    channels = _PAIR_GLOBALS["_stage_channels"](stage)
    dispatch = generator._dispatch_question()
    keyword_mode = next(
        option for option in dispatch.options if option.option_id == "mode-keyword"
    )
    categories = generator.step(
        Choice(keyword_mode.option_id, public_payload=keyword_mode.public_payload)
    )
    category = categories.options[0].public_payload["category"]
    with pytest.raises(ValueError):
        channels.apply_keyword(generator, {"kind": "keyword", "term": category})


@pytest.mark.parametrize("stage", STAGES_WITH_CATEGORY_KEYWORD)
def test_keyword_category_retry_rejects_the_category_slate(stage: str) -> None:
    generator, _ = _stage_generator_with_draft(stage)
    dispatch = generator._dispatch_question()
    keyword_mode = next(
        option for option in dispatch.options if option.option_id == "mode-keyword"
    )
    categories = generator.step(
        Choice(keyword_mode.option_id, public_payload=keyword_mode.public_payload)
    )
    retry = categories.options[-1]
    result = generator.step(
        Choice(retry.option_id, public_payload=retry.public_payload)
    )
    validate_question(result)
    # Control returns to dispatch, whose eager rebuild repopulates the
    # category preview; the rejection itself is durably recorded.
    assert {option.public_payload.get("mode") for option in result.options} >= {
        "mc",
        "keyword",
    }
    rejected = generator.state["rejected_keywords"][-1]
    assert rejected["phase"] == "category"
    assert len(rejected["phrases"]) == 16
    # The rejection note was flushed into the state update on the way back to
    # dispatch; the durable record above is the channel's contract.


@pytest.mark.parametrize("stage", STAGES_WITH_CATEGORY_KEYWORD)
def test_category_sanitizer_bounds_shape_and_rejects_multiline(stage: str) -> None:
    channels = vars(_PAIR_GLOBALS["_stage_channels"](stage))
    sanitize = channels["_sanitize_category"]
    assert sanitize("  kinds of data   domains ") == "kinds of data domains"
    assert sanitize("a\nb") is None
    assert sanitize("x" * 501) is None
    assert sanitize(" ".join(["word"] * 81)) is None
    assert sanitize("") is None
    # Regression: the first fresh paper produced sixteen categories of this
    # shape, every one of them over the original 120-character bound, which
    # dropped the entire slate and left the route retry-only.
    illustrative = (
        "A specific non-standard or emerging model architecture family whose "
        "internals are dissected (e.g., state-space models, mixture-of-experts "
        "routers, diffusion language models, or recurrent-attention hybrids)"
    )
    assert len(illustrative) > 120
    assert sanitize(illustrative) == illustrative


def test_every_stage_carries_the_same_category_layer() -> None:
    """The improvement is uniform: no stage keeps the flat phrase preview."""

    for stage in STAGES_WITH_CATEGORY_KEYWORD:
        channels = vars(_PAIR_GLOBALS["_stage_channels"](stage))
        assert channels["KEYWORD_CATEGORY_COUNT"] == 16
        module = _PAIR_GLOBALS["_stage_module"](stage)
        assert "SEMANTIC CATEGORIES" in module.KEYWORD_CATEGORY_PROMPT
        # A bought category is binding on the completion slate.
        assert "is settled" in module.KEYWORD_GUESS_PROMPT
        generator, _ = _stage_generator_with_draft(stage)
        dispatch = generator._dispatch_question()
        keyword_mode = next(
            option for option in dispatch.options if option.option_id == "mode-keyword"
        )
        preview = generator.step(
            Choice(keyword_mode.option_id, public_payload=keyword_mode.public_payload)
        )
        kinds = {option.public_payload["kind"] for option in preview.options}
        assert kinds == {"keyword_category", "keyword_category_retry"}


def test_pruned_stages_price_their_surviving_routes_more_cheaply() -> None:
    """Dropping a dead route is a discount on every route that remains.

    Displayed probabilities are renormalised over what a stage still offers, so
    Directional -- down to mc, keyword and submit -- prices those well below the
    same names at strict, where the full set is live.
    """

    prices = {}
    for stage in STAGES_WITH_CATEGORY_KEYWORD:
        generator, _ = _stage_generator_with_draft(stage)
        dispatch = generator._dispatch_question()
        modes = [option.public_payload["mode"] for option in dispatch.options]
        assert "secondary_field" not in modes, stage
        prices[stage] = {
            option.public_payload["mode"]: float(option.probability)
            for option in dispatch.options
        }
    assert set(prices["directional"]) == {"mc", "keyword", "submit"}
    for mode in ("mc", "keyword", "submit"):
        assert prices["directional"][mode] > prices["strict"][mode], mode
