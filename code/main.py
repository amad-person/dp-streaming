import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from dataset import Dataset
from query import MwemPgmQuery
from query_engine import NaiveBinaryQueryEngine, BinaryRestartsQueryEngine, IntervalRestartsQueryEngine

warnings.filterwarnings("ignore")

# testing
if __name__ == "__main__":
    for batch_size in [None]:
        print("Batch Size:", batch_size)
        for window_size in [None]:
            print("Window Size:", window_size)
            dataset_prefix = "ny_taxi_medium"
            dataset_name = f"{dataset_prefix}_batch{batch_size}_window{window_size}"

            time_int = pd.DateOffset(days=1)
            time_int_str = "10mins"
            data_encoding_type = "binarized"
            dataset = Dataset.load_from_path(f"../data/{dataset_name}_{data_encoding_type}.csv",
                                             domain_path=f"../data/{dataset_prefix}_{data_encoding_type}_domain.json",
                                             id_col="Person ID",
                                             insertion_time_col="Insertion Time",
                                             deletion_time_col="Deletion Time",
                                             time_interval=time_int,
                                             hist_repr_type=data_encoding_type)
            dataset.df = dataset.df[dataset.df["insertion_batch"] != dataset.df["deletion_batch"]]
            dataset.save_to_path(f"../data/{dataset_name}_{data_encoding_type}_batched_{time_int_str}.csv")

            query_type = "mwem_pgm"
            comparison_type = "all"
            epsilon = 10.0
            delta = 1e-9
            privstr = "eps" + str(epsilon).replace(".", "_")
            if delta:
                privstr += "del" + str(delta).replace(".", "_").replace("^", "_")
            num_runs = 3
            org_seed = 1234
            exp_save_dir = Path(f"../save/{dataset_name}_{comparison_type}_{query_type}"
                                f"_{privstr}_{num_runs}runs_{org_seed}oseed")
            if not Path.is_dir(exp_save_dir):
                os.mkdir(exp_save_dir)
            start_from_batch_num = None
            num_batches = 25
            predicates = [
                "VendorID == 0",
                "VendorID == 1",
                "store_and_fwd_flag == 1",
                "store_and_fwd_flag == 0",
                "payment_type == 0",
                "payment_type == 1",
                "payment_type == 2",
                "payment_type == 3",
                "payment_type == 4",
                "payment_type == 5",
                "payment_type == 6",
                "passenger_count == 1 & VendorID == 0",
                "passenger_count == 1 & VendorID == 1",
                "passenger_count == 2 & VendorID == 0",
                "passenger_count == 2 & VendorID == 1",
                "passenger_count == 3 & VendorID == 0",
                "passenger_count == 3 & VendorID == 1",
                "RatecodeID == 1 & payment_type == 0 & store_and_fwd_flag == 1",
                "RatecodeID == 1 & payment_type == 1 & store_and_fwd_flag == 0",
                "RatecodeID == 2 & payment_type == 0 & store_and_fwd_flag == 1",
                "RatecodeID == 3 & payment_type == 1 & store_and_fwd_flag == 0"
            ]
            num_threads = 4

            # run mechanisms on the same dataset NUM_RUNS number of times
            for run in range(num_runs):
                print("On run number:", run)
                seed = org_seed + run
                rng = np.random.default_rng(seed)

                print("Running Naive Binary Mechanism")
                nb_query = MwemPgmQuery(dataset=dataset, predicates=predicates, k=3, rng=rng)
                naive_binary_query_engine = NaiveBinaryQueryEngine(dataset, nb_query,
                                                                   epsilon, delta,
                                                                   save_path_prefix=f"{exp_save_dir}/run{run}_nb",
                                                                   num_threads=num_threads)
                nb_true_ans, nb_private_ans = naive_binary_query_engine.run(num_batches=num_batches,
                                                                            start_from_batch_num=start_from_batch_num)
                print("True Answers:", nb_true_ans.tolist())
                print("Private Answers:", nb_private_ans.tolist())
                np.savez(f"{exp_save_dir}/nb_true_ans_run{run}", np.array(nb_true_ans))
                np.savez(f"{exp_save_dir}/nb_private_ans_run{run}", np.array(nb_private_ans))

            for run in range(num_runs):
                print("On run number:", run)
                seed = org_seed + run
                rng = np.random.default_rng(seed)

                print("Running Binary Restarts Mechanism")
                br_query = MwemPgmQuery(dataset=dataset, predicates=predicates, k=3, rng=rng)
                binary_restarts_query_engine = BinaryRestartsQueryEngine(dataset, br_query,
                                                                         epsilon, delta,
                                                                         save_path_prefix=f"{exp_save_dir}/run{run}_br",
                                                                         num_threads=num_threads)
                br_true_ans, br_private_ans = binary_restarts_query_engine.run(num_batches=num_batches,
                                                                               start_from_batch_num=start_from_batch_num)
                print("True Answers:", br_true_ans.tolist())
                print("Private Answers:", br_private_ans.tolist())
                np.savez(f"{exp_save_dir}/br_true_ans_run{run}", np.array(br_true_ans))
                np.savez(f"{exp_save_dir}/br_private_ans_run{run}", np.array(br_private_ans))

            for run in range(num_runs):
                print("On run number:", run)
                seed = org_seed + run
                rng = np.random.default_rng(seed)

                print("Running Interval Restarts Mechanism")
                int_query = MwemPgmQuery(dataset=dataset, predicates=predicates, k=3, rng=rng)
                interval_restarts_query_engine = IntervalRestartsQueryEngine(dataset, int_query,
                                                                             epsilon, delta,
                                                                             save_path_prefix=f"{exp_save_dir}/run{run}_int",
                                                                             num_threads=num_threads)
                int_true_ans, int_private_ans = interval_restarts_query_engine.run(num_batches=num_batches,
                                                                                   start_from_batch_num=start_from_batch_num)
                print("True Answers:", int_true_ans.tolist())
                print("Private Answers:", int_private_ans.tolist())
                np.savez(f"{exp_save_dir}/int_true_ans_run{run}", np.array(int_true_ans))
                np.savez(f"{exp_save_dir}/int_private_ans_run{run}", np.array(int_private_ans))

