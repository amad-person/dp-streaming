from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
import random

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
        """
        :param sensitivity:
        :param epsilon:
        :param rng:
        """
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
        """
        :param dataset:
        :param predicate:
        :param sensitivity:
        :param epsilon:
        :param rng:
        """
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
    def __init__(self, dataset=None, predicates=None, k=None,
                 sensitivity=None, epsilon=None, delta=None,
                 iterations=10, repetitions=10,
                 noisy_max_budget=0.5, rng=None):
        """
        :param dataset:
        :param predicates:
        :param k:
        :param sensitivity:
        :param epsilon:
        :param delta:
        :param iterations:
        :param repetitions:
        :param noisy_max_budget:
        :param rng:
        """
        super().__init__()
        self.dataset = dataset
        self.predicates = predicates
        self.workload = None
        self._create_workload(k)
        self.sensitivity = sensitivity
        self.epsilon = epsilon
        self.delta = delta
        self.iterations = iterations
        self.repetitions = repetitions
        self.noisy_max_budget = noisy_max_budget
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
            for predicate in self.predicates:
                answers.append(df.query(predicate).shape[0])
            return answers
        else:
            return [0] * len(self.predicates)

    def get_private_answer(self, ids):
        if ids is not None:
            self._mwem(ids)  # learn histogram using MWEM and generate synthetic dataset
            answers = []
            for predicate in self.predicates:
                answers.append(self.synthetic_dataset.query(predicate).shape[0])  # answers from synthetic dataset
            return answers
        else:
            return [0] * len(self.predicates)

    # TODO: fill this out
    def _create_workload(self, k):
        workload_hist = []
        self.workload = workload_hist

    @staticmethod
    def _evaluate_query_on_hist(query, histogram):
        return np.dot(query, histogram)

    @staticmethod
    def _evaluate_workload_on_hist(workload, histogram):
        return np.dot(workload @ workload.T, histogram)

    def _get_true_hist_answers(self, ids):
        true_hist = self.dataset.get_hist_repr(ids)
        return self._evaluate_workload_on_hist(self.workload, true_hist)

    def _select_query_using_noisy_max(self, true_hist_answers, synthetic_hist, measurements, scale_for_noisy_max):
        synthetic_answers = self._evaluate_workload_on_hist(self.workload, synthetic_hist)
        errors = true_hist_answers - synthetic_answers
        for prev_measured_query in measurements.keys():  # to ignore previously measured queries
            errors[prev_measured_query] = 0.0
        noisy_errors = np.abs(errors) + self.rng.laplace(loc=0, scale=scale_for_noisy_max, size=len(errors))
        return np.argmax(noisy_errors)

    def _update_synthetic_hist(self, query_idx, synthetic_hist, measurements):
        query = self.workload[query_idx]
        error = measurements[query_idx] - self._evaluate_query_on_hist(query, synthetic_hist)
        for i in range(len(synthetic_hist)):
            synthetic_hist[i] *= np.exp((error * query[i]) / 2.0)
        synthetic_hist /= synthetic_hist.sum()  # normalize
        return synthetic_hist

    def _mwem(self, ids):
        true_hist_answers = self._get_true_hist_answers(ids)

        # initialize histogram as a uniform distribution
        synthetic_hist = np.ones(shape=self.dataset.get_hist_repr_dim(),  # dimensions = product of domains
                                 dtype=np.float32)
        synthetic_hist /= synthetic_hist.sum()  # normalize

        epsilon_for_each_iteration = self.epsilon / self.iterations
        measurements = {}  # dict of query idx -> answer on synthetic dataset
        for iteration in range(self.iterations):
            # distribute epsilon for iteration for query selection and noisy measurement of query
            scale = 2.0 / (epsilon_for_each_iteration * len(ids))
            scale_for_noisy_max = scale / self.noisy_max_budget
            scale_for_noisy_measurement = scale / (1 - self.noisy_max_budget)

            # select new query to measure
            selected_query_idx = self._select_query_using_noisy_max(true_hist_answers,
                                                                    synthetic_hist,
                                                                    measurements,
                                                                    scale_for_noisy_max)

            # noisy ground truth answer for selected query
            measurements[selected_query_idx] = (true_hist_answers[selected_query_idx] +
                                                self.rng.laplace(loc=0, scale=scale_for_noisy_measurement))

            # update synthetic histogram using selected query
            self._update_synthetic_hist(selected_query_idx, synthetic_hist, measurements)

            # update synthetic histogram using noisy measurements of all previously selected queries
            for _ in range(self.repetitions):
                query_indices = list(measurements.keys())
                random.shuffle(query_indices)
                for query_idx in query_indices:
                    self._update_synthetic_hist(query_idx, synthetic_hist, measurements)

        # create a tabular dataset from the histogram
        self.synthetic_dataset = self._create_synthetic_dataset(synthetic_hist, num_records=len(ids))

    def _create_synthetic_dataset(self, synthetic_hist, num_records):
        flattened_synthetic_hist = synthetic_hist.flatten()
        samples = self.rng.choice(a=np.array(range(len(flattened_synthetic_hist))),
                                  size=num_records,
                                  p=flattened_synthetic_hist)
        dim = self.dataset.get_hist_repr_dim()
        data = np.zeros(shape=(num_records, dim))
        for idx, sample in enumerate(samples):
            data[idx] = np.flip(np.base_repr(sample, base=2, padding=dim), axis=0)
        synthetic_df_ohe = pd.DataFrame(data, columns=self.dataset.get_hist_repr_columns())
        synthetic_df = pd.from_dummies(synthetic_df_ohe)
        return synthetic_df
