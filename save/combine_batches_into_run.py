from pathlib import Path

import numpy as np
import yaml

if __name__ == "__main__":
    plots_config_path = f"../plots/plots_config.yaml"
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

    # combine data for each run
    for run in range(num_runs):
        nb_true_answers, nb_private_answers = [], []
        br_true_answers, br_private_answers = [], []
        int_true_answers, int_private_answers = [], []
        for batch_idx in range(batches):
            # load batch answers for Naive Binary
            nb_true_ans = np.load(f"{exp_save_dir}/run{run}_nb_true_ans_batch{batch_idx}.npz")['arr_0']
            nb_priv_ans = np.load(f"{exp_save_dir}/run{run}_nb_private_ans_batch{batch_idx}.npz")['arr_0']
            nb_true_answers.append(nb_true_ans)
            nb_private_answers.append(nb_priv_ans)

            # load batch answers for Binary Restarts
            br_true_ans = np.load(f"{exp_save_dir}/run{run}_br_true_ans_batch{batch_idx}.npz")['arr_0']
            br_priv_ans = np.load(f"{exp_save_dir}/run{run}_br_private_ans_batch{batch_idx}.npz")['arr_0']
            br_true_answers.append(br_true_ans)
            br_private_answers.append(br_priv_ans)

            if comparison_type == "all":
                int_true_ans = np.load(f"{exp_save_dir}/run{run}_int_true_ans_batch{batch_idx}.npz")['arr_0']
                int_priv_ans = np.load(f"{exp_save_dir}/run{run}_int_private_ans_batch{batch_idx}.npz")['arr_0']
                int_true_answers.append(br_true_ans)
                int_private_answers.append(br_priv_ans)

        # save combined results
        np.savez(f"{exp_save_dir}/nb_true_ans_run{run}_batches{batches}", np.array(nb_true_answers))
        np.savez(f"{exp_save_dir}/nb_private_ans_run{run}_batches{batches}", np.array(nb_private_answers))
        np.savez(f"{exp_save_dir}/br_true_ans_run{run}_batches{batches}", np.array(br_true_answers))
        np.savez(f"{exp_save_dir}/br_private_ans_run{run}_batches{batches}", np.array(br_private_answers))

        if comparison_type == "all":
            np.savez(f"{exp_save_dir}/int_true_ans_run{run}_batches{batches}",
                     np.array(int_true_answers))
            np.savez(f"{exp_save_dir}/int_private_ans_run{run}_batches{batches}",
                     np.array(int_private_answers))