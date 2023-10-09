from abc import ABC, abstractmethod


class Node(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def get_true_answer(self):
        pass

    @abstractmethod
    def get_private_answer(self):
        pass


class NaiveNode(Node):
    def __init__(self, ids, query):
        super().__init__()
        self.ids = ids
        self.query = query
        self.true_answer = self.query.get_true_answer(self.ids)
        self.private_answer = self.query.get_private_answer(self.ids)

    def get_true_answer(self):
        return self.true_answer

    def get_private_answer(self):
        return self.private_answer


class RestartNode(Node):
    def __init__(self):
        super().__init__()

    def get_true_answer(self):
        pass

    def get_private_answer(self):
        pass

    def restart(self):
        pass
