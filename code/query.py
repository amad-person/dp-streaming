import random
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd

from mbi import Dataset, Domain
import utils


class Query(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def set_privacy_parameters(self, epsilon, delta=None):
        pass

    @abstractmethod
    def get_true_answer(self, *args) -> list:
        pass

    @abstractmethod
    def get_private_answer(self, *args) -> list:
        pass

    @abstractmethod
    def __repr__(self):
        pass


class CountQuery(Query):
    """
    Counts the number of records in the specified dataset.
    """

    def __init__(self, sensitivity=None, epsilon=None, rng=None):
        """
        :param sensitivity: Sensitivity of the query to be used by the Laplace Mechanism.
        :param epsilon: Privacy budget for the Laplace Mechanism.
        :param rng: Numpy random generator for experiment reproducibility.
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
            return [len(ids)]
        else:
            return [0]

    def get_private_answer(self, ids, rerun=True):
        if ids is not None:
            true_answer = self.get_true_answer(ids)[0]
            laplace_noise = self.rng.laplace(loc=0, scale=(self.sensitivity / self.epsilon))
            noisy_answer = true_answer + laplace_noise
            return [noisy_answer]
        else:
            return [0]

    def __repr__(self):
        return f"Count Query:\tEpsilon: {self.epsilon}\tDelta: {self.delta}\tSensitivity: {self.sensitivity}"


class PredicateQuery(Query):
    """
    Counts the number of records that match a predicate in the specified dataset.
    """

    def __init__(self, dataset=None, predicate=None, sensitivity=None, epsilon=None, rng=None):
        """
        :param dataset: Dataset to evaluate the predicate query on.
        :param predicate: Set of conditions that records in the dataset must match.
        :param sensitivity: Sensitivity of the query to be used by the Laplace Mechanism.
        :param epsilon: Privacy budget for the Laplace Mechanism.
        :param rng: Numpy random generator for experiment reproducibility.
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
            return [df.query(self.predicate).shape[0]]
        else:
            return [0]

    def get_private_answer(self, ids, rerun=True):
        if ids is not None:
            true_answer = self.get_true_answer(ids)[0]
            laplace_noise = self.rng.laplace(loc=0, scale=(self.sensitivity / self.epsilon))
            noisy_answer = true_answer + laplace_noise
            return [noisy_answer]
        else:
            return [0]

    def __repr__(self):
        return f"Predicate Query:\tEpsilon: {self.epsilon}\tDelta: {self.delta}\tSensitivity: {self.sensitivity}"


class PmwQuery(Query):
    """
    Generates a synthetic dataset using k-way predicates for the specified dataset, and
    returns the answers for predicates using the synthetic dataset.
    Uses the MWEM algorithm to generate the synthetic dataset.
    """

    def __init__(self, dataset=None, predicates=None, k=None,
                 sensitivity=None, epsilon=None, delta=None,
                 iterations=10, repetitions=10,
                 noisy_max_budget=0.5, rng=None):
        """
        :param dataset: Dataset for which a synthetic dataset is generated.
        :param predicates: Queries that need to be answered using the synthetic dataset. Each predicate is a
            set of conditions that records in the dataset must match.
        :param k: Upper bound on the number of conditions a predicate can have.
        :param sensitivity:
        :param epsilon: Privacy budget for Multiplicative Weights Exponential Mechanism (MWEM).
        :param iterations: Number of times to run MWEM.
        :param repetitions: Each iteration of MWEM updates the synthetic dataset for "repetitions" number of times.
        :param noisy_max_budget: Fraction of the privacy budget to be used by the query selection Exponential Mechanism.
        :param rng: Numpy random generator for experiment reproducibility.

        Reference: Hardt, Moritz, Katrina Ligett, and Frank McSherry. "A simple and practical algorithm for
            differentially private data release." Advances in neural information processing systems 25 (2012).
        Code adapted from: https://github.com/mrtzh/PrivateMultiplicativeWeights.jl
        """
        super().__init__()
        self.dataset = dataset
        self.predicates = predicates
        self.k = k
        self.workload = None
        self._create_workload()
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
        if ids is not None and len(ids) > 0:
            df = self.dataset.select_rows_from_ids(ids)
            answers = []
            for predicate in self.predicates:
                answers.append(df.query(predicate).shape[0])
            return answers
        else:
            return [0] * len(self.predicates)

    def get_private_answer(self, ids, rerun=True):
        if ids is not None and len(ids) > 0:
            if rerun:  # check if previously generated synthetic histogram can't be used
                self._mwem(ids)  # learn histogram using MWEM and generate synthetic dataset
            answers = []
            for predicate in self.predicates:
                answers.append(self.synthetic_dataset.query(predicate).shape[0])  # answers from synthetic dataset
            return answers
        else:
            return [0] * len(self.predicates)

    def _create_workload(self):
        self.workload = utils.get_parity_queries(self.dataset.get_hist_repr_dim(), self.k)

    @staticmethod
    def _evaluate_query_on_hist(query, histogram):
        return np.dot(query, histogram)

    @staticmethod
    def _evaluate_workload_on_hist(workload, histogram):
        return np.dot(workload, histogram)

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
        synthetic_hist = np.ones(shape=2 ** self.dataset.get_hist_repr_dim(),  # dimensions = product of domains
                                 dtype=np.float64)  # use high precision
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
        # numerical instability: remove invalid values (NaN), slightly increase weight of 0s, and normalize
        synthetic_hist = np.nan_to_num(synthetic_hist)
        synthetic_hist[synthetic_hist == 0] = 1e-32
        synthetic_hist /= synthetic_hist.sum()

        # sample 'num_records' rows for the synthetic dataset
        # each row is a record type from the domain 2^dim
        samples = self.rng.choice(a=np.array(range(len(synthetic_hist))),
                                  size=num_records,
                                  p=synthetic_hist)

        # convert record type to encoded row
        dim = self.dataset.get_hist_repr_dim()
        data = np.zeros(shape=(num_records, dim))
        for i in range(num_records):
            binary_str = format(samples[i], f'0{dim}b')
            data[i] = [int(bit) for bit in binary_str]

        # convert encoded dataset to original form
        synthetic_df_encoded = pd.DataFrame(data, columns=self.dataset.get_hist_repr_columns())
        hist_repr_type = self.dataset.get_hist_repr_type()
        if hist_repr_type == "ohe":
            return utils.ohe_to_dataset(encoded_df=synthetic_df_encoded)
        elif hist_repr_type == "binarized":
            return utils.binarized_to_dataset(encoded_df=synthetic_df_encoded, domain=self.dataset.get_domain())

    def __repr__(self):
        return f"PMW Query:\tEpsilon: {self.epsilon}\tDelta: {self.delta}\tSensitivity: {self.sensitivity}"


class MstQuery(Query):
    """
    Generates a synthetic dataset using k-way predicates for the specified dataset, and
    returns the answers for predicates using the synthetic dataset.
    Uses the MST algorithm to generate the synthetic dataset. MST is workload-agnostic.
    Depends on the private-pgm package: https://github.com/ryan112358/private-pgm
    """

    def __init__(self, dataset=None, predicates=None,
                 sensitivity=None, epsilon=None, delta=1e-9,
                 rng=None):
        super().__init__()
        self.dataset = dataset
        self.predicates = predicates
        self.mst_dataset = None
        self.mst_domain = None
        self._create_mst_domain()
        self.sensitivity = sensitivity
        self.epsilon = epsilon
        self.delta = delta
        if rng is None:
            self.rng = np.random.default_rng(1000)
        else:
            self.rng = rng
        self.synthetic_dataset = None

    def set_privacy_parameters(self, epsilon, delta=None):
        self.epsilon = epsilon
        if delta is not None:
            self.delta = delta

    def get_true_answer(self, ids) -> list:
        if ids is not None and len(ids) > 0:
            df = self.dataset.select_rows_from_ids(ids)
            answers = []
            for predicate in self.predicates:
                answers.append(df.query(predicate).shape[0])
            return answers
        else:
            return [0] * len(self.predicates)

    def get_private_answer(self, ids, rerun=True) -> list:
        if ids is not None and len(ids) > 0:
            if rerun:  # check if previously generated synthetic histogram can't be used
                self._mst(ids)  # learn histogram using MST and generate synthetic dataset
            answers = []
            for predicate in self.predicates:
                answers.append(self.synthetic_dataset.query(predicate).shape[0])  # answers from synthetic dataset
            return answers
        else:
            return [0] * len(self.predicates)

    def _create_mst_domain(self):
        # create MST compatible domain
        our_domain = self.dataset.get_domain()
        mst_domain_dict = {}
        for feature in our_domain.keys():
            feature_domain = our_domain[feature]
            if isinstance(feature_domain, int):
                mst_domain_dict[feature] = feature_domain  # number of categories is already given as int
            else:
                mst_domain_dict[feature] = len(feature_domain)  # number of categories for discrete variable
        self.mst_domain = Domain.fromdict(mst_domain_dict)

    def _mst(self, ids):
        # create MST compatible dataset with selected IDs
        self.mst_dataset = Dataset(df=self.dataset.select_rows_from_ids(ids),
                                   domain=self.mst_domain)

        # learn synthetic dataset using MST
        self.synthetic_dataset = utils.MST(self.mst_dataset, self.epsilon, self.delta).df

    def __repr__(self):
        return f"MST Query:\tEpsilon: {self.epsilon}\tDelta: {self.delta}\tSensitivity: {self.sensitivity}"


class MwemPgmQuery(Query):
    """
    Generates a synthetic dataset using k-way predicates for the specified dataset, and
    returns the answers for predicates using the synthetic dataset.
    Uses the MWEM-PGM algorithm to generate the synthetic dataset.
    Depends on the private-pgm package: https://github.com/ryan112358/private-pgm
    """

    def __init__(self, dataset=None, predicates=None, k=None,
                 sensitivity=None, epsilon=None, delta=1e-9,
                 rng=None):
        super().__init__()
        self.dataset = dataset
        self.predicates = predicates
        self.k = k
        self.mwem_pgm_dataset = None
        self.mwem_pgm_domain = None
        self._create_mwem_pgm_domain()
        self.sensitivity = sensitivity
        self.epsilon = epsilon
        self.delta = delta
        if rng is None:
            self.rng = np.random.default_rng(1000)
        else:
            self.rng = rng
        self.synthetic_dataset = None

    def set_privacy_parameters(self, epsilon, delta=None):
        self.epsilon = epsilon
        if delta is not None:
            self.delta = delta

    def get_true_answer(self, ids) -> list:
        if ids is not None and len(ids) > 0:
            df = self.dataset.select_rows_from_ids(ids)
            answers = []
            for predicate in self.predicates:
                answers.append(df.query(predicate).shape[0])
            return answers
        else:
            return [0] * len(self.predicates)

    def get_private_answer(self, ids, rerun=True) -> list:
        if ids is not None and len(ids) > 0:
            if rerun:  # check if previously generated synthetic histogram can't be used
                self._mwem_pgm(ids)  # learn histogram using MST and generate synthetic dataset
            answers = []
            for predicate in self.predicates:
                answers.append(self.synthetic_dataset.query(predicate).shape[0])  # answers from synthetic dataset
            return answers
        else:
            return [0] * len(self.predicates)

    def _create_mwem_pgm_domain(self):
        # create MWEM-PGM compatible domain
        our_domain = self.dataset.get_domain()
        mst_domain_dict = {}
        for feature in our_domain.keys():
            feature_domain = our_domain[feature]
            if isinstance(feature_domain, int):
                mst_domain_dict[feature] = feature_domain  # number of categories is already given as int
            else:
                mst_domain_dict[feature] = len(feature_domain)  # number of categories for discrete variable
        self.mwem_pgm_domain = Domain.fromdict(mst_domain_dict)

    def _mwem_pgm(self, ids):
        # create MWEM-PGM compatible dataset with selected IDs
        self.mwem_pgm_dataset = Dataset(df=self.dataset.select_rows_from_ids(ids),
                                        domain=self.mwem_pgm_domain)

        # learn synthetic dataset using MWEM-PGM
        self.synthetic_dataset = utils.mwem_pgm(
            self.mwem_pgm_dataset, self.epsilon, self.delta, self.k
        ).df

    def __repr__(self):
        return f"MWEM-PGM Query:\tEpsilon: {self.epsilon}\tDelta: {self.delta}\tSensitivity: {self.sensitivity}"


def initialize_answer_var(query: Query):
    if isinstance(query, PmwQuery):
        answer = np.zeros(shape=len(query.predicates))
    elif isinstance(query, MstQuery):
        answer = np.zeros(shape=len(query.predicates))
    elif isinstance(query, MwemPgmQuery):
        answer = np.zeros(shape=len(query.predicates))
    else:
        answer = np.array([0.0])
    return answer


def initialize_answer_vars(query: Query):
    if isinstance(query, PmwQuery):
        true_answer = np.zeros(shape=len(query.predicates))
        private_answer = np.zeros(shape=len(query.predicates))
    elif isinstance(query, MstQuery):
        true_answer = np.zeros(shape=len(query.predicates))
        private_answer = np.zeros(shape=len(query.predicates))
    elif isinstance(query, MwemPgmQuery):
        true_answer = np.zeros(shape=len(query.predicates))
        private_answer = np.zeros(shape=len(query.predicates))
    else:
        true_answer = np.array([0.0])
        private_answer = np.array([0.0])
    return true_answer, private_answer
