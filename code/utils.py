from functools import partial

import numpy as np
import multiprocessing


def get_first_non_zero_lsb(binary_num):
    binary_str = str(binary_num)[::-1]
    return binary_str.index('1')


def get_binary_repr(num):
    return bin(num)[2:]


def get_tree_idx(num):
    binary_repr = get_binary_repr(num)
    return len(binary_repr)


def get_tree_height(num):
    return get_tree_idx(num)


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
def binary_generator(d, indices, init_value=0):
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
    print("Generating parity queries", dim, k)
    parities = set()
    parities.add(0)
    for seq in gosper(dim, k):
        for alpha in binary(dim, seq):
            parities.add(alpha + 1)
    parities = sorted(parities)
    print("Parities to hadamard", parities)
    queries = [hadamard_basis_vector(parity, dim) for parity in parities]
    print("Queries final shape", np.array(queries).shape)
    return np.array(queries)
