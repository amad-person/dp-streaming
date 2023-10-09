from abc import ABC, abstractmethod
import numpy as np


class Query(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def get_true_answer(self, *args):
        pass

    @abstractmethod
    def get_private_answer(self, *args):
        pass


class CountQuery(Query):
    def __init__(self, epsilon, sensitivity):
        super().__init__()
        self.epsilon = epsilon
        self.sensitivity = sensitivity

    def get_true_answer(self, ids):
        return len(ids)

    def get_private_answer(self, ids):
        true_answer = self.get_true_answer(ids)
        laplace_noise = np.random.laplace(loc=0, scale=(self.sensitivity/self.epsilon))
        return true_answer + laplace_noise


class PredicateQuery(Query):
    def __init__(self, dataset, predicate, epsilon, sensitivity):
        super().__init__()
        self.dataset = dataset
        self.predicate = predicate
        self.epsilon = epsilon
        self.sensitivity = sensitivity

    def get_true_answer(self, ids):
        df = self.dataset.select_rows_from_ids(ids)
        return df.query(self.predicate).shape[0]

    def get_private_answer(self, ids):
        true_answer = self.get_true_answer(ids)
        laplace_noise = np.random.laplace(loc=0, scale=(self.sensitivity/self.epsilon))
        return true_answer + laplace_noise
