import itertools
import math
import os.path

import dill
import networkx as nx
import numpy as np
import pandas as pd
from disjoint_set import DisjointSet
from mbi import FactoredInference, Domain, Dataset, GraphicalModel
from scipy import sparse
from scipy.special import logsumexp, softmax


def get_time_offset_obj(time_int_type, time_int_quantity):
    time_offset_obj = None
    if time_int_type == "days":
        time_offset_obj = pd.DateOffset(days=time_int_quantity)
    elif time_int_type == "hours":
        time_offset_obj = pd.DateOffset(hours=time_int_quantity)
    elif time_int_type == "minutes":
        time_offset_obj = pd.DateOffset(minutes=time_int_quantity)
    return time_offset_obj


def get_first_non_zero_lsb(binary_num):
    binary_str = str(binary_num)[::-1]
    return binary_str.index("1")


def get_binary_repr(num):
    return bin(num)[2:]


def get_tree_idx(num):
    binary_repr = get_binary_repr(num)
    return len(binary_repr)


def get_tree_height(num):
    return get_tree_idx(num)


def get_interval_tree_level(num):
    if num % 2 == 1:
        # odd-numbered nodes are always at level = 1
        level = 1
    else:
        # even-numbered nodes are at level = 1 + number of times node ID can be divided by 2
        level = 1
        while num % 2 == 0:
            level += 1
            num = num / 2
    return level


def get_interval_tree_lowest_ancestor(num):
    with open(f"../caching/lowest_ancestor_map.pkl", "rb") as anc_f:
        lowest_ancestor_map = dill.load(anc_f)["lowest_ancestor_map"]
    return lowest_ancestor_map[num]


def get_interval_tree_nodes_on_rtl_path(num):
    with open(f"../caching/rtl_path_map.pkl", "rb") as rtl_f:
        rtl_path_map = dill.load(rtl_f)["rtl_path_map"]
    return rtl_path_map[num]


# source: https://stackoverflow.com/a/57025941
def is_power_of_two(num):
    return (num & (num - 1) == 0) and num != 0


def dataset_to_ohe(df, domain):
    # convert all columns to one-hot encoding
    df[df.columns] = df[df.columns].astype("category")
    for feature in domain.keys():
        feature_domain = domain[feature]
        if isinstance(feature_domain, int):
            num_categories = feature_domain
        else:
            num_categories = len(feature_domain)
        # to account for missing categories in the dataset, we explicitly set all possible ones for the feature
        df[feature] = df[feature].cat.set_categories(range(num_categories))
    return pd.get_dummies(df, dtype=int)


def ohe_to_dataset(encoded_df):
    # TODO: this needs a more graceful failure method when a variable has more than one 1 in its OHE.
    return pd.from_dummies(encoded_df)


def dataset_to_binarized(df, domain):
    encoded_df = pd.DataFrame({})
    for feature in domain.keys():
        feature_domain = domain[feature]
        if feature_domain == "Binarized":
            r = df[feature].max() - df[feature].min()  # range for continuous variable
            dim = int(np.ceil(np.log2(r)))  # length of binarized representation
        else:
            num_categories = len(feature_domain)  # number of categories for discrete variable
            dim = int(np.ceil(np.log2(num_categories)))  # length of binarized representation

        # create binarized representation
        # source: https://stackoverflow.com/a/22227898
        original_column = df[feature].to_numpy()
        binarized_columns = [f"{feature}_{idx}" for idx in range(dim)]
        encoded_df[binarized_columns] = pd.DataFrame(
            (np.fliplr(original_column[:, None] & (1 << np.arange(dim))) > 0).astype(int)
        )
    return encoded_df


def binarized_to_dataset(encoded_df, domain):
    df = pd.DataFrame({})
    for feature in domain.keys():
        # get binarized columns for the feature
        binarized_columns = [col for col in encoded_df.columns if col.startswith(f"{feature}_")]
        dim = len(binarized_columns)

        # decode binarized representation
        # source: https://stackoverflow.com/a/15506055
        binarized_arr = encoded_df[binarized_columns].to_numpy()
        df[feature] = binarized_arr.dot(1 << np.arange(dim - 1, -1, -1)).astype(int)
    return df


# source: https://github.com/mrtzh/PrivateMultiplicativeWeights.jl/blob/master/src/gosper.jl
def gosper(n, k):
    assert 0 < k <= n <= 62
    return gosper_generator(n, k)


# source: https://github.com/mrtzh/PrivateMultiplicativeWeights.jl/blob/master/src/gosper.jl
def gosper_generator(n, k):
    state = (2 ** k - 1, 2 ** n - 1)
    while state[0] <= state[1]:
        o = state[0]
        u = state[0] & -state[0]
        v = u + state[0]
        y = v + (((v ^ state[0]) // u) >> 2)
        state = (y, state[1])
        yield o


# source: https://github.com/mrtzh/PrivateMultiplicativeWeights.jl/blob/master/src/gosper.jl
def binary(d, indices):
    assert 0 < d <= 62
    if isinstance(indices, int):
        _, indices = norm_1_with_indices(indices)
    return binary_generator(d, indices, 0)


# source: https://github.com/mrtzh/PrivateMultiplicativeWeights.jl/blob/master/src/gosper.jl
def binary_generator(indices, init_value=0):
    for state in range(2 ** len(indices)):
        sub_binary_number = init_value
        temp_state = state
        for i in range(len(indices)):
            if temp_state == 0:
                break
            if temp_state % 2 == 1:
                sub_binary_number += 1 << (indices[i] - 1)
            temp_state >>= 1
        yield sub_binary_number


# source: https://github.com/mrtzh/PrivateMultiplicativeWeights.jl/blob/master/src/gosper.jl
def norm_1_with_indices(alpha):
    assert alpha >= 0
    count = 0
    index = 1
    idx = []
    while alpha > 0:
        if alpha % 2 == 1:
            count += 1
            idx.append(index)
        alpha >>= 1
        index += 1
    return count, idx


# source: https://github.com/mrtzh/PrivateMultiplicativeWeights.jl/blob/master/src/parities.jl
def hadamard_basis_vector(index, dim):
    hadamard = np.zeros(1 << dim, dtype=float)  # vector has length = 2^dim
    hadamard[0] = 1.0
    for i in range(dim):  # build vector according to the bits in index
        sign = -1.0 if (index & (1 << i)) > 0 else 1.0  # sign is negative if bit is 1, else it is positive
        for j in range(1 << i):
            hadamard[j + (1 << i)] = sign * hadamard[j]
    return hadamard


# source: https://github.com/mrtzh/PrivateMultiplicativeWeights.jl/blob/master/src/parities.jl
def get_parity_queries(dim, k):
    parity_queries_filepath = f"../caching/parity_queries_dim{dim}_k{k}"
    if os.path.exists(parity_queries_filepath):
        queries = np.load(f"{parity_queries_filepath}.npz")["arr_0"]
    else:
        parities = set()
        parities.add(0)
        for seq in gosper(dim, k):
            for alpha in binary(dim, seq):
                parities.add(alpha + 1)
        parities = sorted(parities)
        queries = [hadamard_basis_vector(parity, dim) for parity in parities]
        queries = np.array(queries)
        np.savez(parity_queries_filepath, queries)
    return queries


# source: https://github.com/ryan112358/private-pgm/blob/master/mechanisms/cdp2adp.py
def cdp_delta(rho, eps):
    assert rho >= 0
    assert eps >= 0
    if rho == 0:
        return 0  # degenerate case

    # search for best alpha
    # Note that any alpha in (1, infty) yields a valid upper bound on delta
    # Thus if this search is slightly "incorrect" it will only result in larger delta (still valid)
    # This code has two "hacks".
    # First the binary search is run for a pre-specified length.
    # 1000 iterations should be sufficient to converge to a good solution.
    # Second we set a minimum value of alpha to avoid numerical stability issues.
    # Note that the optimal alpha is at least (1 + eps/rho)/2. Thus, we only hit this constraint
    # when eps <= rho or close to it. This is not an interesting parameter regime, as you will
    # inherently get large delta in this regime.
    amin = 1.01  # don't let alpha be too small, due to numerical stability
    amax = (eps + 1) / (2 * rho) + 2
    for i in range(1000):  # should be enough iterations
        alpha = (amin + amax) / 2
        derivative = (2 * alpha - 1) * rho - eps + math.log1p(-1.0 / alpha)
        if derivative < 0:
            amin = alpha
        else:
            amax = alpha
    # now calculate delta
    delta = math.exp((alpha - 1) * (alpha * rho - eps) + alpha * math.log1p(-1 / alpha)) / (alpha - 1.0)
    return min(delta, 1.0)  # delta <=1 always


# source: https://github.com/ryan112358/private-pgm/blob/master/mechanisms/cdp2adp.py
def cdp_rho(eps, delta):
    assert eps >= 0
    assert delta > 0
    if delta >= 1:
        return 0.0  # if delta >= 1 anything goes
    rhomin = 0.0  # maintain cdp_delta(rho,eps) <= delta
    rhomax = eps + 1  # maintain cdp_delta(rhomax,eps) > delta
    for i in range(1000):
        rho = (rhomin + rhomax) / 2
        if cdp_delta(rho, eps) <= delta:
            rhomin = rho
        else:
            rhomax = rho
    return rhomin


# source: https://github.com/ryan112358/private-pgm/blob/master/mechanisms/mst.py
def MST(data, epsilon, delta):
    rho = cdp_rho(epsilon, delta)
    sigma = np.sqrt(3 / (2 * rho))
    cliques = [(col,) for col in data.domain]
    log1 = measure(data, cliques, sigma)
    data, log1, undo_compress_fn = compress_domain(data, log1)
    cliques = select(data, rho / 3.0, log1)
    log2 = measure(data, cliques, sigma)
    engine = FactoredInference(data.domain, iters=5000)
    est = engine.estimate(log1 + log2)
    synth = est.synthetic_data(rows=len(data.df))
    return undo_compress_fn(synth)


# source: https://github.com/ryan112358/private-pgm/blob/master/mechanisms/mst.py
def measure(data, cliques, sigma, weights=None):
    if weights is None:
        weights = np.ones(len(cliques))
    weights = np.array(weights) / np.linalg.norm(weights)
    measurements = []
    for proj, wgt in zip(cliques, weights):
        x = data.project(proj).datavector()
        y = x + np.random.normal(loc=0, scale=sigma / wgt, size=x.size)
        Q = sparse.eye(x.size)
        measurements.append((Q, y, sigma / wgt, proj))
    return measurements


# source: https://github.com/ryan112358/private-pgm/blob/master/mechanisms/mst.py
def compress_domain(data, measurements):
    supports = {}
    new_measurements = []
    for Q, y, sigma, proj in measurements:
        col = proj[0]
        sup = y >= 3 * sigma
        supports[col] = sup
        if supports[col].sum() == y.size:
            new_measurements.append((Q, y, sigma, proj))
        else:  # need to re-express measurement over the new domain
            y2 = np.append(y[sup], y[~sup].sum())
            I2 = np.ones(y2.size)
            I2[-1] = 1.0 / np.sqrt(y.size - y2.size + 1.0)
            y2[-1] /= np.sqrt(y.size - y2.size + 1.0)
            I2 = sparse.diags(I2)
            new_measurements.append((I2, y2, sigma, proj))
    undo_compress_fn = lambda data: reverse_data(data, supports)
    return transform_data(data, supports), new_measurements, undo_compress_fn


# source: https://github.com/ryan112358/private-pgm/blob/master/mechanisms/mst.py
def exponential_mechanism(q, eps, sensitivity, prng=np.random, monotonic=False):
    coef = 1.0 if monotonic else 0.5
    scores = coef * eps / sensitivity * q
    probas = np.exp(scores - logsumexp(scores))
    return prng.choice(q.size, p=probas)


# source: https://github.com/ryan112358/private-pgm/blob/master/mechanisms/mst.py
def select(data, rho, measurement_log, cliques=[]):
    engine = FactoredInference(data.domain, iters=1000)
    est = engine.estimate(measurement_log)

    weights = {}
    candidates = list(itertools.combinations(data.domain.attrs, 2))
    for a, b in candidates:
        xhat = est.project([a, b]).datavector()
        x = data.project([a, b]).datavector()
        weights[a, b] = np.linalg.norm(x - xhat, 1)

    T = nx.Graph()
    T.add_nodes_from(data.domain.attrs)
    ds = DisjointSet()

    for e in cliques:
        T.add_edge(*e)
        ds.union(*e)

    r = len(list(nx.connected_components(T)))
    epsilon = np.sqrt(8 * rho / (r - 1))
    for i in range(r - 1):
        candidates = [e for e in candidates if not ds.connected(*e)]
        wgts = np.array([weights[e] for e in candidates])
        idx = exponential_mechanism(wgts, epsilon, sensitivity=1.0)
        e = candidates[idx]
        T.add_edge(*e)
        ds.union(*e)

    return list(T.edges)


# source: https://github.com/ryan112358/private-pgm/blob/master/mechanisms/mst.py
def transform_data(data, supports):
    df = data.df.copy()
    newdom = {}
    for col in data.domain:
        support = supports[col]
        size = support.sum()
        newdom[col] = int(size)
        if size < support.size:
            newdom[col] += 1
        mapping = {}
        idx = 0
        for i in range(support.size):
            mapping[i] = size
            if support[i]:
                mapping[i] = idx
                idx += 1
        assert idx == size
        df[col] = df[col].map(mapping)
    newdom = Domain.fromdict(newdom)
    return Dataset(df, newdom)


# source: https://github.com/ryan112358/private-pgm/blob/master/mechanisms/mst.py
def reverse_data(data, supports):
    df = data.df.copy()
    newdom = {}
    for col in data.domain:
        support = supports[col]
        mx = support.sum()
        newdom[col] = int(support.size)
        idx, extra = np.where(support)[0], np.where(~support)[0]
        mask = df[col] == mx
        if extra.size == 0:
            pass
        else:
            df.loc[mask, col] = np.random.choice(extra, mask.sum())
        df.loc[~mask, col] = idx[df.loc[~mask, col]]
    newdom = Domain.fromdict(newdom)
    return Dataset(df, newdom)


# source: https://github.com/ryan112358/private-pgm/blob/master/mechanisms/mwem%2Bpgm.py
def worst_approximated(workload_answers, est, workload, eps, penalty=True, bounded=False):
    errors = np.array([])
    for cl in workload:
        bias = est.domain.size(cl) if penalty else 0
        x = workload_answers[cl]
        xest = est.project(cl).datavector()
        errors = np.append(errors, np.abs(x - xest).sum() - bias)
    sensitivity = 2.0 if bounded else 1.0
    prob = softmax(0.5 * eps / sensitivity * (errors - errors.max()))
    # prevent numerical instability errors
    prob = np.nan_to_num(prob)
    prob[prob == 0] = 1e-32
    prob = prob / prob.sum()
    key = np.random.choice(len(errors), p=prob)
    return workload[key]


# source: https://github.com/ryan112358/private-pgm/blob/master/mechanisms/mwem%2Bpgm.py
def mwem_pgm(data, epsilon, delta=0.0,
             k=2, workload=None, rounds=None,
             maxsize_mb=25, pgm_iters=1000,
             noise='gaussian', bounded=False, alpha=0.9):
    if workload is None:
        workload = list(itertools.combinations(data.domain, k))
    if rounds is None:
        rounds = len(data.domain)

    if noise == 'laplace':
        eps_per_round = epsilon / rounds
        sigma = 1.0 / (alpha * eps_per_round)
        exp_eps = (1 - alpha) * eps_per_round
        marginal_sensitivity = 2 if bounded else 1.0
    else:
        rho = cdp_rho(epsilon, delta)
        rho_per_round = rho / rounds
        sigma = np.sqrt(0.5 / (alpha * rho_per_round))
        exp_eps = np.sqrt(8 * (1 - alpha) * rho_per_round)
        marginal_sensitivity = np.sqrt(2) if bounded else 1.0

    domain = data.domain
    total = data.records if bounded else None

    def size(cliques):
        return GraphicalModel(domain, cliques).size * 8 / 2 ** 20

    workload_answers = {cl: data.project(cl).datavector() for cl in workload}

    engine = FactoredInference(data.domain, log=False, iters=pgm_iters, warm_start=True)
    measurements = []
    est = engine.estimate(measurements, total)
    cliques = []
    for i in range(1, rounds + 1):
        # [New] Only consider candidates that keep the model sufficiently small
        candidates = [cl for cl in workload if size(cliques + [cl]) <= maxsize_mb * i / rounds]
        ax = worst_approximated(workload_answers, est, candidates, exp_eps)
        # print('Round', i, 'Selected', ax, 'Model Size (MB)', est.size * 8 / 2 ** 20)
        n = domain.size(ax)
        x = data.project(ax).datavector()
        if noise == 'laplace':
            y = x + np.random.laplace(loc=0, scale=marginal_sensitivity * sigma, size=n)
        else:
            y = x + np.random.normal(loc=0, scale=marginal_sensitivity * sigma, size=n)
        Q = sparse.eye(n)
        measurements.append((Q, y, 1.0, ax))
        est = engine.estimate(measurements, total)
        cliques.append(ax)

    return est.synthetic_data(rows=len(data.df))
