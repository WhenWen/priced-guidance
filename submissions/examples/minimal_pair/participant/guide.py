from tech_tree_arena import Choice, PresentedQuestion


class Guide:
    def __init__(self, target, services):
        self.answer = target["answer"]
        self.services = services

    def step(self, question: PresentedQuestion):
        return Choice(self.answer)



# Legacy Python names remain available for existing submissions.
Oracle = Guide
