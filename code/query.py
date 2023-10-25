from abc import ABC, abstractmethod
import numpy as np
import utils


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
    def __init__(self, sensitivity=None, epsilon=None, rng=None):
        super().__init__()
        self.sensitivity = sensitivity
        self.epsilon = epsilon
        self.delta = None
        if rng is None:
            self.rng = np.random.default_rng(1000)
        else:
            self.rng = rng

    def set_privacy_parameters(self, epsilon, delta=None):
        self.epsilon = epsilon
        if delta is not None:
            self.delta = delta

    def get_true_answer(self, ids):
        if ids is not None:
            return len(ids)
        else:
            return 0

    def get_private_answer(self, ids):
        if ids is not None:
            true_answer = self.get_true_answer(ids)
            laplace_noise = self.rng.laplace(loc=0, scale=(self.sensitivity / self.epsilon))
            return true_answer + laplace_noise
        else:
            return 0


class PredicateQuery(Query):
    def __init__(self, dataset=None, predicate=None, sensitivity=None, epsilon=None, rng=None):
        super().__init__()
        self.dataset = dataset
        self.predicate = predicate
        self.sensitivity = sensitivity
        self.epsilon = epsilon
        self.delta = None
        if rng is None:
            self.rng = np.random.default_rng(1000)
        else:
            self.rng = rng

    def set_privacy_parameters(self, epsilon, delta=None):
        self.epsilon = epsilon
        if delta is not None:
            self.delta = delta

    def get_true_answer(self, ids):
        if ids is not None:
            df = self.dataset.select_rows_from_ids(ids)
            return df.query(self.predicate).shape[0]
        else:
            return 0

    def get_private_answer(self, ids):
        if ids is not None:
            true_answer = self.get_true_answer(ids)
            laplace_noise = self.rng.laplace(loc=0, scale=(self.sensitivity / self.epsilon))
            return true_answer + laplace_noise
        else:
            return 0


class PmwQuery(Query):
    def __init__(self, dataset=None, workload=None, sensitivity=None,
                 epsilon=None, delta=None, iterations=None, rng=None):
        super().__init__()
        self.dataset = dataset
        self.workload = workload
        self.sensitivity = sensitivity
        self.epsilon = epsilon
        self.delta = delta
        self.iterations = iterations
        if rng is None:
            self.rng = np.random.default_rng(1000)
        else:
            self.rng = rng
        self.synthetic_dataset = None

    def set_privacy_parameters(self, epsilon, delta=None):
        self.epsilon = epsilon
        if delta is not None:
            self.delta = delta

    def get_true_answer(self, ids):
        if ids is not None:
            df = self.dataset.select_rows_from_ids(ids)
            answers = []
            for predicate in self.workload:
                answers.append(df.query(predicate).shape[0])
            return answers
        else:
            return [0] * len(self.workload)

    def get_private_answer(self, ids):
        if ids is not None:
            self.generate_synthetic_dataset(ids)
            answers = []
            for predicate in self.workload:
                answers.append(self.synthetic_dataset.query(predicate).shape[0])
            return answers
        else:
            return [0] * len(self.workload)

    def select_query_using_exponential_mech(self, ids, synthetic_hist, epsilon):
        true_answers = np.array(self.get_true_answer(ids))
        errors = [0] * len(self.workload)
        for idx, predicate in enumerate(self.workload):
            histogram_predicate = utils.get_histogram_predicate(predicate, self.dataset.get_domain())
            synthetic_answer = 0  # TODO: fill this out
            errors[idx] = epsilon * (np.abs(true_answers[idx] - synthetic_answer) / 2.0)
        return 0

    # TODO: fill this out
    def update_synthetic_hist(self, synthetic_hist, measurements):
        print(measurements, self.synthetic_dataset)
        return synthetic_hist

    def generate_synthetic_dataset(self, ids):
        synthetic_hist = utils.init_synthetic_histogram(self.dataset.get_domain())

        measurements = {}  # dict of query idx -> answer on synthetic dataset
        for iteration in range(self.iterations):
            epsilon_for_exponential = (self.epsilon / (2 * self.iterations))
            selected_query_idx = self.select_query_using_exponential_mech(ids, synthetic_hist,
                                                                          epsilon_for_exponential)
            while selected_query_idx in measurements:
                selected_query_idx = self.select_query_using_exponential_mech(ids, synthetic_hist,
                                                                              epsilon_for_exponential)

            histogram_predicate = utils.get_histogram_predicate(self.workload[selected_query_idx],
                                                                self.dataset.get_domain())
            # TODO: fill this out
            measurements[selected_query_idx] = 0 + self.rng.laplace(loc=0, scale=(2 * self.iterations / self.epsilon))

            synthetic_hist = self.update_synthetic_hist(synthetic_hist, measurements)

        self.synthetic_dataset = utils.get_synthetic_dataset_from_histogram(synthetic_hist, self.dataset.get_domain())
