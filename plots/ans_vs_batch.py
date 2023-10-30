import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


if __name__ == "__main__":
    dataset_name = "adult_small"
    query_type = "pmw"
    epsilon = 10.0
    delta = None
    privstr = "eps" + str(epsilon).replace(".", "_")
    if delta:
        privstr += "del" + str(delta).replace(".", "_").replace("^", "_")
    num_runs = 3
    org_seed = 1234
    exp_save_dir = Path(f"../save/{dataset_name}_nb_vs_br_{query_type}_{privstr}_{num_runs}runs_{org_seed}oseed")

    # create data for plot
    mechanism_labels, batch_nums, answer_values = [], [], []
    for run in range(num_runs):
        true_ans = np.load(f"{exp_save_dir}/nb_true_ans_run{run}.npz")['arr_0']
        num_batches, num_queries = true_ans.shape

        # load answer values for Naive Binary
        nb_priv_ans = np.load(f"{exp_save_dir}/nb_private_ans_run{run}.npz")['arr_0']
        for query_idx in range(num_queries):
            query_nb_answers = nb_priv_ans[:, query_idx]  # query answers are stored in columns
            answer_values += query_nb_answers.tolist()
            mechanism_labels += ["Naive Binary"] * num_batches
            batch_nums += list(range(num_batches))

        # load answer values for Binary Restarts
        br_priv_ans = np.load(f"{exp_save_dir}/br_private_ans_run{run}.npz")['arr_0']
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
    plt.title(f"Query Type: {query_type}")
    sns.lineplot(data=df, x="Batch Number", y="Answer", hue="Mechanism")
    plt.savefig(f"{exp_save_dir}/ans_vs_batch.png")
    plt.show()
