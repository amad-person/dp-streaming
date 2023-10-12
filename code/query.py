from abc import ABC, abstractmethod
import numpy as np


class Query(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def set_privacy_parameters(self, epsilon, delta=None):
        pass

    @abstractmethod
    def get_true_answer(self, *args):
        pass

    @abstractmethod
    def get_private_answer(self, *args):
        pass


class CountQuery(Query):
    def __init__(self, sensitivity=None, epsilon=None):
        super().__init__()
        self.sensitivity = sensitivity
        self.epsilon = epsilon

    def set_privacy_parameters(self, epsilon, delta=None):
        self.epsilon = epsilon

    def get_true_answer(self, ids):
        if ids is not None:
            return len(ids)
        else:
            return 0

    def get_private_answer(self, ids):
        if ids is not None:
            true_answer = self.get_true_answer(ids)
            laplace_noise = np.random.laplace(loc=0, scale=(self.sensitivity / self.epsilon))
            return true_answer + laplace_noise
        else:
            return 0


class PredicateQuery(Query):
    def __init__(self, dataset=None, predicate=None, sensitivity=None, epsilon=None):
        super().__init__()
        self.dataset = dataset
        self.predicate = predicate
        self.sensitivity = sensitivity
        self.epsilon = epsilon

    def set_privacy_parameters(self, epsilon, delta=None):
        self.epsilon = epsilon

    def get_true_answer(self, ids):
        if ids is not None:
            df = self.dataset.select_rows_from_ids(ids)
            return df.query(self.predicate).shape[0]
        else:
            return 0

    def get_private_answer(self, ids):
        if ids is not None:
            true_answer = self.get_true_answer(ids)
            laplace_noise = np.random.laplace(loc=0, scale=(self.sensitivity / self.epsilon))
            return true_answer + laplace_noise
        else:
            return 0
