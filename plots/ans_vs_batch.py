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
    epsilon = plots_config["epsilon"]
    delta = plots_config["delta"]
    privstr = "eps" + str(epsilon).replace(".", "_")
    if delta:
        privstr += "del" + str(delta).replace(".", "_").replace("^", "_")

    num_runs = plots_config["num_runs"]
    org_seed = plots_config["org_seed"]
    exp_save_dir = Path(f"../save/{dataset_name}_nb_vs_br_{query_type}_{privstr}_{num_runs}runs_{org_seed}oseed")

    batches = plots_config["batches"]

    # create data for plot
    mechanism_labels, batch_nums, answer_values = [], [], []
    for run in range(num_runs):
        true_ans_path = f"{exp_save_dir}/nb_true_ans_run{run}"
        if batches is not None:
            true_ans_path += f"_batches{batches}"
        true_ans_path += ".npz"
        true_ans = np.load(true_ans_path)["arr_0"]
        num_batches, num_queries = true_ans.shape

        # load answer values for Naive Binary
        nb_priv_ans_path = f"{exp_save_dir}/nb_private_ans_run{run}"
        if batches is not None:
            nb_priv_ans_path += f"_batches{batches}"
        nb_priv_ans_path += ".npz"
        nb_priv_ans = np.load(nb_priv_ans_path)["arr_0"]
        for query_idx in range(num_queries):
            query_nb_answers = nb_priv_ans[:, query_idx]  # query answers are stored in columns
            answer_values += query_nb_answers.tolist()
            mechanism_labels += ["Naive Binary"] * num_batches
            batch_nums += list(range(num_batches))

        # load answer values for Binary Restarts
        br_priv_ans_path = f"{exp_save_dir}/br_private_ans_run{run}"
        if batches is not None:
            br_priv_ans_path += f"_batches{batches}"
        br_priv_ans_path += ".npz"
        br_priv_ans = np.load(br_priv_ans_path)["arr_0"]
        for query_idx in range(num_queries):
            query_br_answers = br_priv_ans[:, query_idx]
            answer_values += query_br_answers.tolist()
            mechanism_labels += ["Binary Restarts"] * num_batches
            batch_nums += list(range(num_batches))

    data_dict = {
        "Mechanism": mechanism_labels,
        "Answer": answer_values,
        "Batch Number": batch_nums
    }
    df = pd.DataFrame(data_dict)
    plt.title(f"Query Type: {query_type}\nBatch Size: {batch_size}\nWindow Size: {window_size}")
    sns.lineplot(data=df, x="Batch Number", y="Answer", hue="Mechanism")
    plt.tight_layout()
    plt.savefig(f"{exp_save_dir}/ans_vs_batch.png", dpi=1000)
    # plt.show()
