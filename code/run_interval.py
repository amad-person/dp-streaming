import argparse
import json
import os
import warnings
from pathlib import Path

import numpy as np
import yaml

from dataset import Dataset
from query import MwemPgmQuery
from query_engine import IntervalRestartsQueryEngine
from utils import get_time_offset_obj

warnings.filterwarnings("ignore")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config", help="Path to config file")
    args = parser.parse_args()

    exp_config_path = args.config

    with open(exp_config_path, "r") as config_file:
        exp_config = yaml.safe_load(config_file)

    for batch_size in exp_config["batch_size"]:
        print("Batch Size:", batch_size)
        for window_size in exp_config["batch_size"]:
            print("Window Size:", window_size)
            dataset_prefix = exp_config["dataset_prefix"]
            dataset_name = f"{dataset_prefix}_batch{batch_size}_window{window_size}"

            time_int_type = exp_config["time_int_type"]
            time_int_quantity = exp_config["time_int_quantity"]
            time_int = get_time_offset_obj(time_int_type, time_int_quantity)
            time_int_str = exp_config["time_int_str"]

            data_encoding_type = exp_config["data_encoding_type"]
            dataset = Dataset.load_from_path(f"../data/{dataset_name}_{data_encoding_type}.csv",
                                             domain_path=f"../data/{dataset_prefix}_{data_encoding_type}_domain.json",
                                             id_col="Person ID",
                                             insertion_time_col="Insertion Time",
                                             deletion_time_col="Deletion Time",
                                             time_interval=time_int,
                                             hist_repr_type=data_encoding_type)
            dataset.df = dataset.df[dataset.df["insertion_batch"] != dataset.df["deletion_batch"]]
            dataset.save_to_path(f"../data/{dataset_name}_{data_encoding_type}_batched_{time_int_str}.csv")

            query_type = exp_config["query_type"]
            comparison_type = exp_config["comparison_type"]
            epsilon = exp_config["epsilon"]
            delta = float(exp_config["delta"])
            privstr = "eps" + str(epsilon).replace(".", "_")
            if delta:
                privstr += "del" + str(delta).replace(".", "_").replace("^", "_")
            num_runs = exp_config["num_runs"]
            org_seed = exp_config["org_seed"]
            exp_save_dir = Path(f"../save/{dataset_name}_{comparison_type}_{query_type}"
                                f"_{privstr}_{num_runs}runs_{org_seed}oseed")
            if not Path.is_dir(exp_save_dir):
                os.mkdir(exp_save_dir)
            start_from_batch_num = exp_config["start_from_batch_num"]
            num_batches = exp_config["num_batches"]

            with open(Path(__file__).parent / f"../predicates/{exp_config['predicates_filename']}") as predicates_file:
                predicates = json.load(predicates_file)["predicates"]

            num_threads = exp_config["num_threads"]

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

