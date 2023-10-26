from abc import ABC, abstractmethod
import numpy as np
import pandas as pd

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
        self.workload_hist = None
        self._create_workload_hist(workload)
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
            self._mwem(ids)  # learn histogram using MWEM and generate synthetic dataset
            answers = []
            for predicate in self.workload:
                answers.append(self.synthetic_dataset.query(predicate).shape[0])  # answers from synthetic dataset
            return answers
        else:
            return [0] * len(self.workload)

    def _create_workload_hist(self, workload):
        feat_names = list(self.dataset.get_domain().keys())
        num_feats = len(feat_names)
        workload_hist = []
        for predicate in workload:
            # predicate is a str in the format of a pandas query, e.g., feat1 == val & feat2 == val
            # TODO: add check for valid predicates
            conditions = predicate.split(" & ")
            predicate_hist = [None] * num_feats
            for cond in conditions:
                feat, val = cond.split(" == ")
                predicate_hist[feat_names.index(feat)] = int(val)
            workload_hist.append(tuple(predicate_hist))
        self.workload_hist = workload_hist

    def _get_true_hist_answers(self, ids):
        true_hist, _ = self.dataset.get_hist_repr(ids)
        true_hist /= true_hist.flatten().sum()  # normalize

        # compute answers on the true dataset
        true_hist_answers = []
        for predicate_hist in self.workload_hist:
            true_hist_answers.append(true_hist[predicate_hist].sum())
        return true_hist_answers

    def _select_predicate_using_exponential_mech(self, true_hist_answers, synthetic_hist, epsilon):
        errors = [0] * len(self.workload)
        for idx, predicate_hist in enumerate(self.workload_hist):
            synthetic_hist_answer = synthetic_hist[predicate_hist].sum()
            errors[idx] = epsilon * (np.abs(true_hist_answers[idx] - synthetic_hist_answer) / 2.0)
        return 0

    def _update_synthetic_hist(self, synthetic_hist, measurements):
        total = synthetic_hist.sum()
        for _ in range(1):
            for pred_idx in measurements.keys():
                error = measurements[pred_idx] - synthetic_hist[self.workload_hist[pred_idx]].sum()

                mask = np.ones_like(synthetic_hist)
                mask[self.workload_hist[pred_idx]] = 1
                factor = np.exp((mask * error) / (2 * total))

                synthetic_hist *= factor
                synthetic_hist /= synthetic_hist.sum()  # re-normalize
        return synthetic_hist

    def _mwem(self, ids):
        true_hist_answers = self._get_true_hist_answers(ids)

        # initialize histogram as a uniform distribution
        synthetic_hist = np.ones(self.dataset.get_synthetic_hist_shape(),  # dimensions = product of domains
                                 dtype=np.float32)
        synthetic_hist /= synthetic_hist.sum()  # normalize

        measurements = {}  # dict of query idx -> answer on synthetic dataset
        for iteration in range(self.iterations):
            epsilon_for_exponential = (self.epsilon / (2 * self.iterations))

            # select new predicate to measure
            selected_pred_idx = self._select_predicate_using_exponential_mech(true_hist_answers,
                                                                              synthetic_hist,
                                                                              epsilon_for_exponential)
            while selected_pred_idx in measurements:
                selected_pred_idx = self._select_predicate_using_exponential_mech(ids, synthetic_hist,
                                                                                  epsilon_for_exponential)

            # noisy ground truth answer for selected predicate
            measurements[selected_pred_idx] = (true_hist_answers[selected_pred_idx] +
                                               self.rng.laplace(loc=0, scale=(2 * self.iterations / self.epsilon)))

            # update weights of histogram using noisy ground truth answers of all previously selected predicates
            synthetic_hist = self._update_synthetic_hist(synthetic_hist, measurements)

        # create a tabular dataset from the histogram
        self.synthetic_dataset = self._create_synthetic_dataset(synthetic_hist, num_records=len(ids))

    def _create_synthetic_dataset(self, synthetic_hist, num_records):
        def _reverse_hist_index(index, domain_sizes):
            new_index = [0] * len(domain_sizes)
            for i, s in enumerate(domain_sizes):
                new_index[i] += index % s
                index -= index % s
                index //= s
            return new_index

        flattened_synthetic_hist = synthetic_hist.flatten()
        samples = self.rng.choice(a=np.array(range(len(flattened_synthetic_hist))),
                                  size=num_records,
                                  p=flattened_synthetic_hist)
        data = []
        dom_sizes = self.dataset.get_synthetic_hist_shape()
        for idx in samples:
            row = _reverse_hist_index(idx, dom_sizes)
            data.append(row)
        return pd.DataFrame(data, columns=list(self.dataset.get_domain().keys()))
