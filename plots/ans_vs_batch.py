import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


if __name__ == "__main__":
    query_type = "count"
    epsilon = 10.0
    delta = None
    privstr = "eps" + str(epsilon).replace(".", "_")
    if delta:
        privstr += "del" + str(delta).replace(".", "_").replace("^", "_")
    num_runs = 10
    org_seed = 1234
    exp_save_dir = Path(f"../save/nb_vs_br_{query_type}_{privstr}_{num_runs}runs_{org_seed}oseed")

    # create data for plot
    mechanism_labels, batch_nums, answer_values = [], [], []
    for run in range(num_runs):
        true_ans = np.load(f"{exp_save_dir}/nb_true_ans_run{run}.npz")['arr_0']
        num_batches = len(true_ans)

        # load error values for Naive Binary
        nb_priv_ans = np.load(f"{exp_save_dir}/nb_private_ans_run{run}.npz")['arr_0']
        answer_values += nb_priv_ans.tolist()
        mechanism_labels += ["Naive Binary"] * num_batches
        batch_nums += list(range(num_batches))

        # load error values for Binary Restarts
        br_priv_ans = np.load(f"{exp_save_dir}/br_private_ans_run{run}.npz")['arr_0']
        answer_values += br_priv_ans.tolist()
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
