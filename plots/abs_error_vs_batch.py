from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


if __name__ == "__main__":
    batch_size = 1000
    window_size = 3
    dataset_name = f"adult_small_batch{batch_size}_window{window_size}"

    query_type = "pmw"
    epsilon = 10.0
    delta = None
    privstr = "eps" + str(epsilon).replace(".", "_")
    if delta:
        privstr += "del" + str(delta).replace(".", "_").replace("^", "_")
    num_runs = 3
    org_seed = 1000
    exp_save_dir = Path(f"../save/{dataset_name}_nb_vs_br_{query_type}_{privstr}_{num_runs}runs_{org_seed}oseed")

    batches = None

    # create data for plot
    mechanism_labels, batch_nums, error_values = [], [], []
    for run in range(num_runs):
        true_ans_path = f"{exp_save_dir}/nb_true_ans_run{run}"
        if batches is not None:
            true_ans_path += f"_batches{batches}"
        true_ans_path += ".npz"
        true_ans = np.load(true_ans_path)['arr_0']
        num_batches, num_queries = true_ans.shape

        # load error values for Naive Binary
        nb_priv_ans_path = f"{exp_save_dir}/nb_private_ans_run{run}"
        if batches is not None:
            nb_priv_ans_path += f"_batches{batches}"
        nb_priv_ans_path += ".npz"
        nb_priv_ans = np.load(nb_priv_ans_path)['arr_0']
        for query_idx in range(num_queries):
            query_true_answers = true_ans[:, query_idx]  # query answers are stored in columns
            query_nb_answers = nb_priv_ans[:, query_idx]  # query answers are stored in columns
            error_values += np.abs(query_nb_answers - query_true_answers).tolist()
            mechanism_labels += ["Naive Binary"] * num_batches
            batch_nums += list(range(num_batches))

        # load error values for Binary Restarts
        br_priv_ans_path = f"{exp_save_dir}/br_private_ans_run{run}"
        if batches is not None:
            br_priv_ans_path += f"_batches{batches}"
        br_priv_ans_path += ".npz"
        br_priv_ans = np.load(br_priv_ans_path)['arr_0']
        for query_idx in range(num_queries):
            query_true_answers = true_ans[:, query_idx]
            query_br_answers = br_priv_ans[:, query_idx]
            error_values += np.abs(query_br_answers - query_true_answers).tolist()
            mechanism_labels += ["Binary Restarts"] * num_batches
            batch_nums += list(range(num_batches))

    data_dict = {
        "Mechanism": mechanism_labels,
        "Absolute Error": error_values,
        "Batch Number": batch_nums
    }
    df = pd.DataFrame(data_dict)
    plt.title(f"Query Type: {query_type}")
    sns.lineplot(data=df, x="Batch Number", y="Absolute Error", hue="Mechanism")
    plt.savefig(f"{exp_save_dir}/abs_error_vs_batch.png")
    plt.show()


