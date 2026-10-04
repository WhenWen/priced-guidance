from tech_tree_arena import Choice, Idea, Question, Submission, SubmitOption
from tech_tree_arena.contract.validation import message_hash
from tech_tree_arena.evaluation.smoke import SmokeAnswerJudge
from tech_tree_arena.runtime.actor import ActorFactory
from tech_tree_arena.runtime.engine import ArenaRunner
from tech_tree_arena.runtime.services import ServiceFactory


class Generator:
    def __init__(self, services):
        pass

    def step(self, choice):
        if choice is None:
            return Question("q", (SubmitOption("fixed", "same", "1"),))
        return Submission((Idea("i", {"answer": "same"}, "1"),))


class RecordedDecisionOracle:
    def __init__(self, target, services):
        # Deliberately hold decisions fixed across targets.
        self.target = target

    def step(self, question):
        return Choice("fixed")


def _visible_prefix(target):
    events = []
    runner = ArenaRunner(event_sink=events.append)
    runner.run(
        generator_factory=ActorFactory(Generator, service_factory=ServiceFactory(seed=1)),
        oracle_factory=ActorFactory(
            RecordedDecisionOracle,
            constructor_args=(target,),
            service_factory=ServiceFactory(seed=2),
        ),
        target=target,
        judge=SmokeAnswerJudge(),
        seed=3,
        run_id="same-run",
    )
    visible = []
    for event in events:
        if event["kind"] == "question":
            visible.append({"kind": "question", "question": event["question"], "path_k": event["path_k"]})
        elif event["kind"] == "oracle_decision":
            visible.append({"kind": "oracle_decision", "decision": event["decision"]})
        elif event["kind"] == "choice_cost":
            visible.append({key: value for key, value in event.items() if key != "question_id"})
        elif event["kind"] == "submission":
            visible.append(
                {key: value for key, value in event.items() if key != "source_question_id"}
            )
    return [message_hash(event) for event in visible]


def test_target_change_cannot_affect_generator_visible_prefix_when_decisions_are_fixed() -> None:
    assert _visible_prefix({"answer": "same", "secret": "A"}) == _visible_prefix(
        {"answer": "same", "secret": "B"}
    )
