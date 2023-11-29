from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import yaml

if __name__ == "__main__":
    plots_config_path = f"./plots_config.yaml"
    with open(plots_config_path, "r") as config_file:
        plots_config = yaml.safe_load(config_file)

    dataset_prefix = plots_config["dataset_prefix"]
    batch_size = plots_config["batch_size"]
    window_size = plots_config["window_size"]
    dataset_name = f"{dataset_prefix}_batch{batch_size}_window{window_size}"

    query_type = plots_config["query_type"]
    comparison_type = plots_config["comparison_type"]
    epsilon = plots_config["epsilon"]
    delta = plots_config["delta"]
    privstr = "eps" + str(epsilon).replace(".", "_")
    if delta:
        privstr += "del" + str(delta).replace(".", "_").replace("^", "_")

    num_runs = plots_config["num_runs"]
    org_seed = plots_config["org_seed"]
    exp_save_dir = Path(f"../save/{dataset_name}_{comparison_type}_{query_type}"
                        f"_{privstr}_{num_runs}runs_{org_seed}oseed")

    batches = plots_config["batches"]

    # create data for plot
    mechanism_labels, batch_nums, error_values = [], [], []
    for run in range(num_runs):
        # load error values for Naive Binary
        nb_true_ans_path = f"{exp_save_dir}/nb_true_ans_run{run}"
        if batches is not None:
            nb_true_ans_path += f"_batches{batches}"
        nb_true_ans_path += ".npz"
        nb_true_ans = np.load(nb_true_ans_path)["arr_0"]
        nb_num_batches, nb_num_queries = nb_true_ans.shape

        nb_priv_ans_path = f"{exp_save_dir}/nb_private_ans_run{run}"
        if batches is not None:
            nb_priv_ans_path += f"_batches{batches}"
        nb_priv_ans_path += ".npz"
        nb_priv_ans = np.load(nb_priv_ans_path)["arr_0"]
        for query_idx in range(nb_num_queries):
            query_true_answers = nb_true_ans[:, query_idx]  # query answers are stored in columns
            query_nb_answers = nb_priv_ans[:, query_idx]  # query answers are stored in columns
            error_values += (np.abs(query_nb_answers - query_true_answers + 1e-32) /
                             (query_true_answers + 1e-32)).tolist()
            mechanism_labels += ["Naive Binary"] * nb_num_batches
            batch_nums += list(range(nb_num_batches))

        # load error values for Binary Restarts
        br_true_ans_path = f"{exp_save_dir}/br_true_ans_run{run}"
        if batches is not None:
            br_true_ans_path += f"_batches{batches}"
        br_true_ans_path += ".npz"
        br_true_ans = np.load(br_true_ans_path)["arr_0"]
        br_num_batches, br_num_queries = br_true_ans.shape

        br_priv_ans_path = f"{exp_save_dir}/br_private_ans_run{run}"
        if batches is not None:
            br_priv_ans_path += f"_batches{batches}"
        br_priv_ans_path += ".npz"
        br_priv_ans = np.load(br_priv_ans_path)["arr_0"]
        for query_idx in range(br_num_queries):
            query_true_answers = br_true_ans[:, query_idx]
            query_br_answers = br_priv_ans[:, query_idx]
            error_values += (np.abs(query_br_answers - query_true_answers + 1e-32) /
                             (query_true_answers + 1e-32)).tolist()
            mechanism_labels += ["Binary Restarts"] * br_num_batches
            batch_nums += list(range(br_num_batches))

        if comparison_type == "all":
            # load error values for Interval Restarts
            int_true_ans_path = f"{exp_save_dir}/int_true_ans_run{run}"
            if batches is not None:
                int_true_ans_path += f"_batches{batches}"
            int_true_ans_path += ".npz"
            int_true_ans = np.load(int_true_ans_path)["arr_0"]
            int_num_batches, int_num_queries = int_true_ans.shape

            int_priv_ans_path = f"{exp_save_dir}/int_private_ans_run{run}"
            if batches is not None:
                int_priv_ans_path += f"_batches{batches}"
            int_priv_ans_path += ".npz"
            int_priv_ans = np.load(int_priv_ans_path)["arr_0"]
            for query_idx in range(int_num_queries):
                query_true_answers = int_true_ans[:, query_idx]
                query_int_answers = int_priv_ans[:, query_idx]
                error_values += (np.abs(query_int_answers - query_true_answers + 1e-32) /
                                 (query_true_answers + 1e-32)).tolist()
                mechanism_labels += ["Interval Restarts"] * int_num_batches
                batch_nums += list(range(int_num_batches))

    data_dict = {
        "Mechanism": mechanism_labels,
        "Relative Error": error_values,
        "Batch Number": batch_nums
    }
    df = pd.DataFrame(data_dict)
    plt.title(f"Query Type: {query_type}\nBatch Size: {batch_size}\nWindow Size: {window_size}")
    sns.lineplot(data=df, x="Batch Number", y="Relative Error", hue="Mechanism")
    plt.tight_layout()
    plt.savefig(f"{exp_save_dir}/rel_error_vs_batch.png", dpi=1000)
    # plt.show()
