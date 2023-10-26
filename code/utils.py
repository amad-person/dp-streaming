import numpy as np
import pandas as pd


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
