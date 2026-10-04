from pathlib import Path

import pytest
from tech_tree_arena.contract.recovery import occurrence_bits

from tech_tree_arena.evaluation.smoke import SmokeAnswerJudge
from tech_tree_arena.runtime.engine import ArenaRunner
from tech_tree_arena.runtime.services import ServiceFactory
from tech_tree_arena.runtime.subprocess_actor import SubprocessActorFactory


def test_time_travel_reconstructs_generator_in_a_fresh_role_process(tmp_path: Path) -> None:
    participant = tmp_path / "participant"
    participant.mkdir()
    (participant / "__init__.py").write_text("")
    (participant / "generator.py").write_text(
        """from tech_tree_arena import Choice, Idea, Option, Question, Submission, SubmitOption
class Generator:
    def __init__(self, services): self.phase = 'start'
    def step(self, choice):
        if choice is None:
            self.phase = 'first'
            return Question('first', (Option('a','a','0.5'), SubmitOption('b','b','0.5')))
        if self.phase == 'first' and choice.option_id == 'a':
            self.phase = 'second'
            return Question('second', (Option('x','x','0.5'), Option('y','y','0.5')))
        return Submission((Idea('answer', {'answer': choice.public_payload}, '1'),))
"""
    )
    (participant / "oracle.py").write_text(
        """from tech_tree_arena import Checkout, Choice
class Oracle:
    def __init__(self, target, services):
        self.first = None
    def step(self, question):
        if question.question.question == 'first':
            if self.first is None:
                self.first = question.question_id
                return Choice('a')
            return Choice('b')
        return Checkout(self.first)
"""
    )

    runner = ArenaRunner()
    result = runner.run(
        generator_factory=SubprocessActorFactory(
            tmp_path, "participant.generator:Generator", service_factory=ServiceFactory(seed=1)
        ),
        oracle_factory=SubprocessActorFactory(
            tmp_path,
            "participant.oracle:Oracle",
            constructor_args=({"answer": "b"},),
            service_factory=ServiceFactory(seed=2),
        ),
        target={"answer": "b"},
        judge=SmokeAnswerJudge(),
        seed=3,
    )
    try:
        assert result.status == "pass"
        assert result.checkout_count == 1
        assert result.k == pytest.approx(1.0 + occurrence_bits(1))
    finally:
        runner.runtime.close(runner.last_generator)
        runner.runtime.close(runner.last_oracle)


def test_a_subprocess_oracle_reaches_the_judge_through_its_services(
    tmp_path: Path,
) -> None:
    """The Oracle's Judge access has to survive the process boundary.

    judge_evaluate is the seat the Arena's scheduled preview used to occupy.
    It is exercised in-process elsewhere; a real run puts the Oracle in its own
    process, so the frame has to round-trip through the service broker too.
    """

    participant = tmp_path / "participant"
    participant.mkdir()
    (participant / "__init__.py").write_text("")
    (participant / "generator.py").write_text(
        """from tech_tree_arena import Idea, Option, Question, Submission, SubmitOption
class Generator:
    def __init__(self, services): pass
    def step(self, choice):
        if choice is None:
            return Question('first', (Option('a','a','0.5'), SubmitOption('b','b','0.5')))
        return Submission((Idea('answer', {'answer': 'b'}, '1'),))
"""
    )
    (participant / "oracle.py").write_text(
        """from tech_tree_arena import Choice
class Oracle:
    def __init__(self, target, services):
        self.services = services
    def step(self, question):
        verdicts = self.services.judge_evaluate(
            [{"idea_id": "probe", "content": {"answer": "b"}, "probability": "1"}]
        )
        assert [v["idea_id"] for v in verdicts] == ["probe"]
        assert verdicts[0]["passed"] is True
        return Choice('b')
"""
    )

    judge = SmokeAnswerJudge()
    target = {"answer": "b"}

    def judge_call(raw_ideas):
        from tech_tree_arena import Idea

        ideas = tuple(
            Idea(item["idea_id"], item["content"], item["probability"])
            for item in raw_ideas
        )
        return [
            {
                "idea_id": verdict.idea_id,
                "passed": bool(verdict.passed),
                "private_reason": verdict.private_reason or "",
            }
            for verdict in judge.evaluate(target, ideas)
        ]

    runner = ArenaRunner()
    result = runner.run(
        generator_factory=SubprocessActorFactory(
            tmp_path, "participant.generator:Generator", service_factory=ServiceFactory(seed=1)
        ),
        oracle_factory=SubprocessActorFactory(
            tmp_path,
            "participant.oracle:Oracle",
            constructor_args=(target,),
            service_factory=ServiceFactory(seed=2, judge_call=judge_call),
        ),
        target=target,
        judge=judge,
        seed=3,
    )
    try:
        assert result.status == "pass"
    finally:
        runner.runtime.close(runner.last_generator)
        runner.runtime.close(runner.last_oracle)
