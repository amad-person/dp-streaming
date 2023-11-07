from pathlib import Path
import numpy as np


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
    num_runs = 1
    org_seed = 1234
    exp_save_dir = Path(f"./{dataset_name}_nb_vs_br_{query_type}_{privstr}_{num_runs}runs_{org_seed}oseed")

    batches = 2

    # combine data for each run
    for run in range(num_runs):
        nb_true_answers, nb_private_answers = [], []
        br_true_answers, br_private_answers = [], []
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

        # save combined results
        np.savez(f"{exp_save_dir}/nb_true_ans_run{run}_batches{batches}", np.array(nb_true_answers))
        np.savez(f"{exp_save_dir}/nb_private_ans_run{run}_batches{batches}", np.array(nb_private_answers))
        np.savez(f"{exp_save_dir}/br_true_ans_run{run}_batches{batches}", np.array(br_true_answers))
        np.savez(f"{exp_save_dir}/br_private_ans_run{run}_batches{batches}", np.array(br_private_answers))