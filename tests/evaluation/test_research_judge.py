import json
import threading

from tech_tree_arena import Idea
from tech_tree_arena.evaluation import research
from tech_tree_arena.runtime.services import ServiceFactory


class SchemaBackend:
    def __init__(self) -> None:
        self.schemas = []
        self.lock = threading.Lock()

    def structured(self, **request):
        with self.lock:
            self.schemas.append(request["schema_name"])
        if request["schema_name"] == "motivation_verdict":
            return {"recovered": True, "reason": "ok"}
        if request["schema_name"] == "faithful_verdict":
            return {"faithful": True, "reason": "ok"}
        raise AssertionError("unexpected schema")


def test_parallel_fmn_judge_uses_the_explicit_arena_service() -> None:
    backend = SchemaBackend()
    services = ServiceFactory(seed=1, model_backend=backend).create()
    gold = {
        "setting_and_object": {
            "category": "test",
            "groups": [{"aspect": "problem", "details": ["recover it"]}],
        },
        "concrete_detailed_setting": [],
        "key_findings": [],
    }
    idea = {"setting_and_object": "recover it", "findings": [], "prob": 1.0}

    verdict = research.judge_fmn(services.structured_model, gold, idea, 0, need_n=0)

    assert verdict["motivation_ok"] is True
    assert verdict["faithful_ok"] is True
    assert sorted(backend.schemas) == ["faithful_verdict", "motivation_verdict"]
    assert services.usage()["model_calls"] == 2


class FmnOneOfThreeBackend:
    def __init__(self) -> None:
        self.schemas = []
        self.lock = threading.Lock()

    def structured(self, **request):
        with self.lock:
            self.schemas.append(request["schema_name"])
        name = request["schema_name"]
        if name == "coarse_screen_fmn":
            return {"verdicts": [{"idea_number": 1, "recovered": True}]}
        if name == "motivation_verdict":
            return {"recovered": True, "reason": "exact object"}
        if name == "faithful_verdict":
            return {"faithful": True, "reason": "no contradiction"}
        if name == "finding_verdict":
            return {
                "recovered": "GOLD FINDING #2 TO RECOVER" in request["user"],
                "reason": "only the second finding is present",
            }
        raise AssertionError(f"unexpected schema {name}")


def test_research_judge_supports_f_three_one() -> None:
    backend = FmnOneOfThreeBackend()
    services = ServiceFactory(seed=1, model_backend=backend).create()
    gold = {
        "summary": {
            "setting_and_object": {
                "category": "test",
                "groups": [{"aspect": "problem", "details": ["recover it"]}],
            },
            "concrete_detailed_setting": [],
            "key_findings": [
                {"finding": f"finding {index}", "evidence": "evidence"}
                for index in range(1, 5)
            ],
        }
    }
    judge = research.ResearchJudge(services, mode="fmn", fmn_m=3, fmn_n=1)

    verdict = judge.evaluate(
        gold,
        (Idea("idea-1", {"setting_and_object": "recover it", "findings": ["finding 2"]}, "1"),),
    )[0]

    assert verdict.passed is True
    private = json.loads(verdict.private_reason)
    assert private["m"] == 3
    assert private["achieved_n"] == 1
    assert backend.schemas.count("finding_verdict") == 3
