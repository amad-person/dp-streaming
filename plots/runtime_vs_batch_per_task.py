import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import yaml
import dill


def init_task_timing_dict(task_labels, num_runs, num_batches):
    task_timing_dict = {}
    for tl in task_labels:
        task_timing_dict[tl] = np.zeros(shape=(num_runs, num_batches))
    return task_timing_dict


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="./plots_config.yaml", help="Path to config file")
    args = parser.parse_args()

    plots_config_path = args.config

    with open(plots_config_path, "r") as config_file:
        plots_config = yaml.safe_load(config_file)

    for batch_size in plots_config["batch_size"]:
        for window_size in plots_config["window_size"]:
            dataset_prefix = plots_config["dataset_prefix"]
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

            runs_for_plot = plots_config["num_runs_for_plot"]
            batches_for_plot = plots_config["batches_for_plot"]

            task_labels = [
                "node_init_time",
                "update_trees_time",
                "propagate_deletions_time",
                "private_answers_time"
            ]

            # create data for plots
            mechanism_labels, batch_nums = [], []
            nb_task_timing_dict = init_task_timing_dict(task_labels, runs_for_plot, batches_for_plot)
            br_task_timing_dict = init_task_timing_dict(task_labels, runs_for_plot, batches_for_plot)
            int_task_timing_dict = init_task_timing_dict(task_labels, runs_for_plot, batches_for_plot)
            for run in range(runs_for_plot):
                for batch in range(batches_for_plot):
                    # load task time values for Naive Binary
                    nb_timing_path = f"{exp_save_dir}/run{run}_nb_timing_batch{batch}.pkl"
                    with open(nb_timing_path, "rb") as f:
                        obj = dill.load(f)
                        for tl in task_labels:
                            nb_task_timing_dict[tl][run][batch] = obj["time"][tl]

                    # load task time values for Binary Restarts
                    br_timing_path = f"{exp_save_dir}/run{run}_br_timing_batch{batch}.pkl"
                    with open(br_timing_path, "rb") as f:
                        obj = dill.load(f)
                        for tl in task_labels:
                            br_task_timing_dict[tl][run][batch] = obj["time"][tl]

                    # load task time values for Interval Restarts
                    int_timing_path = f"{exp_save_dir}/run{run}_int_timing_batch{batch}.pkl"
                    with open(int_timing_path, "rb") as f:
                        obj = dill.load(f)
                        for tl in task_labels:
                            int_task_timing_dict[tl][run][batch] = obj["time"][tl]
                mechanism_labels += ["Naive Binary"] * batches_for_plot
                mechanism_labels += ["Binary Restarts"] * batches_for_plot
                mechanism_labels += ["Interval Restarts"] * batches_for_plot
                for i in range(3):  # three mechanisms
                    batch_nums += list(range(batches_for_plot))

            # plot runtime vs batch number per task (averaged over runs)
            for tl in task_labels:
                time_for_task = nb_task_timing_dict[tl].flatten().tolist()
                time_for_task += br_task_timing_dict[tl].flatten().tolist()
                time_for_task += int_task_timing_dict[tl].flatten().tolist()
                data_dict = {
                    "Mechanism": mechanism_labels,
                    "Time": time_for_task,
                    "Batch Number": batch_nums
                }
                df = pd.DataFrame(data_dict)
                plt.title(f"Query Type: {query_type}\nBatch Size: {batch_size}\nWindow Size: {window_size}")
                ax = sns.lineplot(data=df, x="Batch Number", y="Time", hue="Mechanism")
                ax.set_ylabel(f"Time for {tl}")
                plt.tight_layout()
                plt.savefig(f"{exp_save_dir}/runtime_vs_batch_{tl}.png", dpi=1000)
                plt.close()
                # plt.show()

            # plot total time per task (averaged over runs)
            mechanism_labels_for_total, time_for_total, task_labels_for_total = [], [], []
            for tl in task_labels:
                time_for_total += nb_task_timing_dict[tl].sum(axis=1).tolist()
                mechanism_labels_for_total += ["Naive Binary"] * runs_for_plot

                time_for_total += br_task_timing_dict[tl].sum(axis=1).tolist()
                mechanism_labels_for_total += ["Binary Restarts"] * runs_for_plot

                time_for_total += int_task_timing_dict[tl].sum(axis=1).tolist()
                mechanism_labels_for_total += ["Interval Restarts"] * runs_for_plot

                task_labels_for_total += [tl] * (runs_for_plot * 3)

            data_dict = {
                "Mechanism": mechanism_labels_for_total,
                "Time": time_for_total,
                "Task": task_labels_for_total
            }
            df = pd.DataFrame(data_dict)
            plt.title(f"Query Type: {query_type}\nBatch Size: {batch_size}\nWindow Size: {window_size}")
            ax = sns.barplot(data=df, x="Task", y="Time", hue="Mechanism")
            plt.setp(ax.get_xticklabels(), rotation=15, ha='right')
            plt.tight_layout()
            plt.savefig(f"{exp_save_dir}/total_runtime_vs_task.png", dpi=1000)
            plt.close()
            # plt.show()