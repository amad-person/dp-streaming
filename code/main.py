import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from dataset import Dataset
from query import CountQuery
from query_engine import NaiveBinaryQueryEngine, BinaryRestartsQueryEngine

warnings.filterwarnings("ignore")

if __name__ == "__main__":
    batch_size = 1000
    window_size = 3
    dataset_prefix = "adult_small"
    dataset_name = f"{dataset_prefix}_batch{batch_size}_window{window_size}"

    time_int = pd.DateOffset(days=1)
    time_int_str = "1day"
    data_encoding_type = "binarized"
    dataset = Dataset.load_from_path(f"../data/{dataset_name}_{data_encoding_type}.csv",
                                     domain_path=f"../data/{dataset_prefix}_{data_encoding_type}_domain.json",
                                     id_col="Person ID",
                                     insertion_time_col="Insertion Time",
                                     deletion_time_col="Deletion Time",
                                     time_interval=time_int,
                                     hist_repr_type=data_encoding_type)
    dataset.save_to_path(f"../data/{dataset_name}_{data_encoding_type}_batched_{time_int_str}.csv")

    query_type = "count"
    epsilon = 10.0
    delta = None
    privstr = "eps" + str(epsilon).replace(".", "_")
    if delta:
        privstr += "del" + str(delta).replace(".", "_").replace("^", "_")
    num_runs = 1
    org_seed = 1000
    exp_save_dir = Path(f"../save/{dataset_name}_nb_vs_br_{query_type}_{privstr}_{num_runs}runs_{org_seed}oseed")
    if not Path.is_dir(exp_save_dir):
        os.mkdir(exp_save_dir)
    num_threads = 4
    start_from_batch_num = 3
    num_batches = 6

    # run mechanisms on the same dataset NUM_RUNS number of times
    for run in range(num_runs):
        print("On run number:", run)
        seed = org_seed + run
        rng = np.random.default_rng(seed)

        print("Running Naive Binary Mechanism")
        nb_query = CountQuery(sensitivity=1, rng=rng)
        naive_binary_query_engine = NaiveBinaryQueryEngine(dataset, nb_query, epsilon, delta,
                                                           save_path_prefix=f"{exp_save_dir}/run{run}_nb",
                                                           num_threads=num_threads)
        nb_true_ans, nb_private_ans = naive_binary_query_engine.run(
            num_batches=num_batches,
            start_from_batch_num=start_from_batch_num
        )
        print("True Answers:", nb_true_ans.tolist())
        print("Private Answers:", nb_private_ans.tolist())
        np.savez(f"{exp_save_dir}/nb_true_ans_run{run}", np.array(nb_true_ans))
        np.savez(f"{exp_save_dir}/nb_private_ans_run{run}", np.array(nb_private_ans))

        print("Running Binary Restarts Mechanism")
        br_query = CountQuery(sensitivity=1, rng=rng)
        binary_restarts_query_engine = BinaryRestartsQueryEngine(dataset, br_query, epsilon, delta,
                                                                 save_path_prefix=f"{exp_save_dir}/run{run}_br",
                                                                 num_threads=num_threads)
        br_true_ans, br_private_ans = binary_restarts_query_engine.run(
            num_batches=num_batches,
            start_from_batch_num=start_from_batch_num
        )
        print("True Answers:", br_true_ans.tolist())
        print("Private Answers:", br_private_ans.tolist())
        np.savez(f"{exp_save_dir}/br_true_ans_run{run}", np.array(br_true_ans))
        np.savez(f"{exp_save_dir}/br_private_ans_run{run}", np.array(br_private_ans))
