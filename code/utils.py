import os.path

import numpy as np
import pandas as pd


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


# Testing
if __name__ == "__main__":
    dummy_df = pd.DataFrame({
        "age": [2, 15, 19, 25, 20, 34, 18, 55],
        "sex": [1, 0, 1, 1, 0, 1, 0, 0]
    })
    dummy_domain = {
        "age": "Binarized",
        "sex": ["Male", "Female"]
    }

    enc_df = dataset_to_binarized(df=dummy_df, domain=dummy_domain)
    dec_df = binarized_to_dataset(enc_df, domain=dummy_domain)

    assert np.all(dummy_df.to_numpy() == dec_df.to_numpy())
