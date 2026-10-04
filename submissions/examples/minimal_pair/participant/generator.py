from tech_tree_arena import Choice, Idea, Question, Submission, SubmitOption


class Generator:
    def __init__(self, services):
        self.services = services

    def step(self, choice: Choice | None):
        if choice is None:
            return Question(
                question="Which hidden color is correct?",
                options=(
                    SubmitOption("red", "red", "0.5"),
                    SubmitOption("blue", "blue", "0.5"),
                ),
            )
        return Submission((Idea("selected-color", {"answer": choice.public_payload}, "1"),))
