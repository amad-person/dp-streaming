import os
from pathlib import Path

import numpy as np
import pandas as pd

from data.preprocessor import create_fake_ins_after_del_dataset
from dataset import Dataset
from query import CountQuery
from query_engine import NaiveBinaryQueryEngine, BinaryRestartsQueryEngine

if __name__ == "__main__":
    n_ins = 1000
    n_repeats = 10
    create_fake_ins_after_del_dataset(path=f"../data/fake_ins_after_del_dataset_{n_ins}_{n_repeats}.csv",
                                      domain_path=f"../data/fake_ins_after_del_dataset_{n_ins}_{n_repeats}_domain"
                                                  f".json",
                                      num_ins=n_ins,
                                      num_repeats=n_repeats)
    time_int = pd.DateOffset(days=1)
    time_int_str = "1day"
    dataset = Dataset.load_from_path(f"../data/fake_ins_after_del_dataset_{n_ins}_{n_repeats}.csv",
                                     domain_path=f"../data/fake_ins_after_del_dataset_{n_ins}_{n_repeats}_domain"
                                                 f".json",
                                     id_col="Person ID",
                                     insertion_time_col="Insertion Time",
                                     deletion_time_col="Deletion Time",
                                     time_interval=time_int)
    dataset.save_to_path(f"../data/fake_ins_after_del_dataset_{n_ins}_{n_repeats}_batched_{time_int_str}.csv")

    query_type = "count"
    epsilon = 10.0
    delta = None
    privstr = "eps" + str(epsilon).replace(".", "_")
    if delta:
        privstr += "del" + str(delta).replace(".", "_").replace("^", "_")
    num_runs = 10
    org_seed = 1234
    exp_save_dir = Path(f"../save/nb_vs_br_{query_type}_{privstr}_{num_runs}runs_{org_seed}oseed")
    if not Path.is_dir(exp_save_dir):
        os.mkdir(exp_save_dir)

    # run mechanisms on the same dataset NUM_RUNS number of times
    for run in range(num_runs):
        seed = org_seed + run
        rng = np.random.default_rng(seed)

        nb_query = CountQuery(sensitivity=1, rng=rng)
        naive_binary_query_engine = NaiveBinaryQueryEngine(dataset, nb_query, epsilon, delta)
        nb_true_ans, nb_private_ans = naive_binary_query_engine.run()
        print("Naive Binary", nb_true_ans, nb_private_ans)
        np.savez(f"{exp_save_dir}/nb_true_ans_run{run}", np.array(nb_true_ans))
        np.savez(f"{exp_save_dir}/nb_private_ans_run{run}", np.array(nb_private_ans))

        br_query = CountQuery(sensitivity=1, rng=rng)
        binary_restarts_query_engine = BinaryRestartsQueryEngine(dataset, br_query, epsilon, delta)
        br_true_ans, br_private_ans = binary_restarts_query_engine.run()
        print("Binary Restarts", br_true_ans, br_private_ans)
        np.savez(f"{exp_save_dir}/br_true_ans_run{run}", np.array(br_true_ans))
        np.savez(f"{exp_save_dir}/br_private_ans_run{run}", np.array(br_private_ans))