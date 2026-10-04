from tech_tree_arena import Choice, PresentedQuestion


class Oracle:
    def __init__(self, target, services):
        self.answer = target["answer"]
        self.services = services

    def step(self, question: PresentedQuestion):
        return Choice(self.answer)

