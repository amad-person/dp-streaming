import logging
import math
import os
import sys
import warnings
from abc import ABC, abstractmethod
from multiprocessing import Pool
from pathlib import Path

import dill
import numpy as np
import pandas as pd
from codetiming import Timer

import utils
from dataset import Dataset
from node import NaiveNode, RestartNode
from query import initialize_answer_vars, MwemPgmQuery

warnings.filterwarnings("ignore")

logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)

stdoutHandler = logging.StreamHandler(stream=sys.stdout)
fileHandler = logging.FileHandler("logs.txt")

logger.addHandler(stdoutHandler)
logger.addHandler(fileHandler)


class QueryEngine(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def run(self):
        pass


def true_answer_worker(node):
    return node.get_true_answer()


def private_answer_worker(node):
    return node.get_private_answer()


class NaiveBinaryQueryEngine(QueryEngine):
    def __init__(self, dataset, query, epsilon, delta, save_path_prefix, num_threads=8):
        super().__init__()
        self.dataset = dataset
        self.query = query
        self.epsilon = epsilon
        self.delta = delta
        self.naive_binary_insertions_map = {}  # stream to track insertions
        self.naive_binary_deletions_map = {}  # stream to track deletions
        self.save_path_prefix = save_path_prefix
        self.num_threads = num_threads

    def run(self, num_batches=None, start_from_batch_num=None, save_checkpts=False):
        true_answers, private_answers = [], []
        num_nodes, current_tree_idx = 0, 0
        for i, (ins_ids, del_ids) in enumerate(self.dataset.get_batches()):
            if num_batches is not None and i == num_batches:
                break

            # load saved answers until run_from_batch number is reached
            if start_from_batch_num is not None and i < start_from_batch_num:
                saved_true_ans = np.load(f"{self.save_path_prefix}_true_ans_batch{i}.npz")["arr_0"]
                true_answers.append(saved_true_ans)
                saved_private_ans = np.load(f"{self.save_path_prefix}_private_ans_batch{i}.npz")["arr_0"]
                private_answers.append(saved_private_ans)
                continue

            # load state from checkpoint
            if start_from_batch_num is not None and i == start_from_batch_num:
                prev_batch_num = start_from_batch_num - 1
                with open(f"{self.save_path_prefix}_checkpt_batch{prev_batch_num}.pkl", "rb") as f:
                    checkpt = dill.load(f)
                    self.naive_binary_insertions_map = checkpt["naive_binary_insertions_map"]
                    self.naive_binary_deletions_map = checkpt["naive_binary_deletions_map"]

            print("Batch number:", i)
            print("Insertion IDs:", ins_ids)
            print("Deletion IDs:", del_ids)

            node_i = i + 1
            logger.debug(f"Created Node ID: {node_i}")

            batch_timer = Timer(logger=None)

            # start building new tree
            tree_idx = utils.get_tree_idx(node_i)
            if tree_idx != current_tree_idx:
                num_nodes = 0
                current_tree_idx = tree_idx
            logger.debug(f"Current Tree ID: {current_tree_idx}")

            # build current nodes
            batch_timer.start()
            self.query.set_privacy_parameters(epsilon=self.epsilon / utils.get_tree_height(node_i),
                                              delta=self.delta)
            ins_node = NaiveNode(ins_ids, self.query)
            del_node = NaiveNode(del_ids, self.query)
            node_init_time = batch_timer.stop()

            num_nodes += 1
            logger.debug(f"Query for Node {node_i}: {self.query}")

            # add current nodes to maps
            ins_tree_nodes = self.naive_binary_insertions_map.get(current_tree_idx, [])
            del_tree_nodes = self.naive_binary_deletions_map.get(current_tree_idx, [])
            ins_tree_nodes.append(ins_node)
            del_tree_nodes.append(del_node)

            # update maps by merging nodes
            batch_timer.start()
            n = num_nodes
            while n > 0:
                if n % 2 == 0:
                    # remove and merge last two nodes in the list
                    second_ins_node = ins_tree_nodes.pop()  # ins_tree_nodes[-1]
                    merged_ins_node = ins_tree_nodes.pop()  # ins_tree_nodes[-2]
                    merged_ins_node.merge_node(second_ins_node)
                    ins_tree_nodes.append(merged_ins_node)

                    second_del_node = del_tree_nodes.pop()  # del_tree_nodes[-1]
                    merged_del_node = del_tree_nodes.pop()  # del_tree_nodes[-2]
                    merged_del_node.merge_node(second_del_node)
                    del_tree_nodes.append(merged_del_node)
                n = n / 2
            self.naive_binary_insertions_map[current_tree_idx] = ins_tree_nodes
            self.naive_binary_deletions_map[current_tree_idx] = del_tree_nodes
            update_trees_time = batch_timer.stop()
            logger.debug(f"Naive Binary Insertions Map: {self.naive_binary_insertions_map}")
            logger.debug(f"Naive Binary Deletions Map: {self.naive_binary_deletions_map}")

            # combine answers from all trees
            true_answer, private_answer = initialize_answer_vars(self.query)
            insertion_nodes = []
            for tree_idx, tree_nodes in self.naive_binary_insertions_map.items():
                for node in tree_nodes:
                    if tree_idx != current_tree_idx:
                        node.set_rerun(rerun=False)
                    insertion_nodes.append(node)

            insertion_pool = Pool(self.num_threads)
            insertion_true_results = insertion_pool.map(true_answer_worker, (node for node in insertion_nodes))
            insertion_pool.close()
            insertion_pool.join()

            batch_timer.start()
            insertion_pool = Pool(self.num_threads)
            insertion_private_results = insertion_pool.map(private_answer_worker, (node for node in insertion_nodes))
            insertion_pool.close()
            insertion_pool.join()
            insertion_private_answers_time = batch_timer.stop()

            deletion_nodes = []
            for tree_idx, tree_nodes in self.naive_binary_deletions_map.items():
                for node in tree_nodes:
                    if tree_idx != current_tree_idx:
                        node.set_rerun(rerun=False)
                    deletion_nodes.append(node)

            deletion_pool = Pool(self.num_threads)
            deletion_true_results = deletion_pool.map(true_answer_worker, (node for node in deletion_nodes))
            deletion_pool.close()
            deletion_pool.join()

            batch_timer.start()
            deletion_pool = Pool(self.num_threads)
            deletion_private_results = deletion_pool.map(private_answer_worker, (node for node in deletion_nodes))
            deletion_pool.close()
            deletion_pool.join()
            deletion_private_answers_time = batch_timer.stop()

            for result in insertion_true_results:
                true_answer += result
            for result in deletion_true_results:
                true_answer -= result

            for result in insertion_private_results:
                private_answer += result
            for result in deletion_private_results:
                private_answer -= result

            # save answers for current batch
            np.savez(f"{self.save_path_prefix}_true_ans_batch{i}", np.array(true_answer))
            np.savez(f"{self.save_path_prefix}_private_ans_batch{i}", np.array(private_answer))

            # save timing info for current batch
            with open(f"{self.save_path_prefix}_timing_batch{i}.pkl", "wb") as f:
                time_dict = {
                    "node_init_time": node_init_time,
                    "update_trees_time": update_trees_time,
                    "propagate_deletions_time": 0,
                    "private_answers_time": insertion_private_answers_time + deletion_private_answers_time,
                }
                logger.debug(f"Naive Binary Time: {time_dict}")
                dill.dump({
                    "time": time_dict
                }, f)

            # save state into checkpoint file
            if save_checkpts is True:
                with open(f"{self.save_path_prefix}_checkpt_batch{i}.pkl", "wb") as f:
                    dill.dump({
                        "naive_binary_insertions_map": self.naive_binary_insertions_map,
                        "naive_binary_deletions_map": self.naive_binary_deletions_map
                    }, f)

            true_answers.append(true_answer)
            private_answers.append(private_answer)

        return np.array(true_answers), np.array(private_answers)


class BinaryRestartsQueryEngine(QueryEngine):
    def __init__(self, dataset, query, epsilon, delta, save_path_prefix, num_threads=8):
        super().__init__()
        self.dataset = dataset
        self.query = query
        self.epsilon = epsilon
        self.delta = delta
        self.binary_restarts_map = {}
        self.save_path_prefix = save_path_prefix
        self.num_threads = num_threads

    def run(self, num_batches=None, start_from_batch_num=None, save_checkpts=False):
        true_answers, private_answers = [], []
        num_nodes, current_tree_idx = 0, 0
        for i, (ins_ids, del_ids) in enumerate(self.dataset.get_batches()):
            if num_batches is not None and i == num_batches:
                break

            # load saved answers until run_from_batch number is reached
            if start_from_batch_num is not None and i < start_from_batch_num:
                saved_true_ans = np.load(f"{self.save_path_prefix}_true_ans_batch{i}.npz")["arr_0"]
                true_answers.append(saved_true_ans)
                saved_private_ans = np.load(f"{self.save_path_prefix}_private_ans_batch{i}.npz")["arr_0"]
                private_answers.append(saved_private_ans)
                continue

            # load state from checkpoint
            if start_from_batch_num is not None and i == start_from_batch_num:
                prev_batch_num = start_from_batch_num - 1
                with open(f"{self.save_path_prefix}_checkpt_batch{prev_batch_num}.pkl", "rb") as f:
                    checkpt = dill.load(f)
                    self.binary_restarts_map = checkpt["binary_restarts_map"]

            print("Batch number:", i)
            print("Insertion IDs:", ins_ids)
            print("Deletion IDs:", del_ids)

            node_i = i + 1
            logger.debug(f"Created Node ID: {node_i}")

            batch_timer = Timer(logger=None)

            # start building new tree
            tree_idx = utils.get_tree_idx(node_i)
            if tree_idx != current_tree_idx:
                num_nodes = 0
                current_tree_idx = tree_idx
            logger.debug(f"Current Tree ID: {current_tree_idx}")

            # build current node
            batch_timer.start()
            epsilon_for_node = self.epsilon / utils.get_tree_height(node_i)
            self.query.set_privacy_parameters(epsilon=epsilon_for_node, delta=self.delta)
            node = RestartNode(ins_ids, self.query,
                               epsilon=epsilon_for_node, delta=self.delta,
                               num_threads=self.num_threads)
            node_init_time = batch_timer.stop()

            num_nodes += 1
            logger.debug(f"Query for Node {node_i}: {self.query}")

            # add current node to map
            tree_nodes = self.binary_restarts_map.get(current_tree_idx, [])
            tree_nodes.append(node)

            # update map by merging nodes
            batch_timer.start()
            n = num_nodes
            while n > 0:
                if n % 2 == 0:
                    # remove and merge last two nodes in the list
                    second_node = tree_nodes.pop()  # trees_node[-1]
                    merged_node = tree_nodes.pop()  # trees_node[-2]
                    merged_node.merge_node(second_node)
                    tree_nodes.append(merged_node)
                n = n / 2
            self.binary_restarts_map[current_tree_idx] = tree_nodes
            update_trees_time = batch_timer.stop()
            logger.debug(f"Binary Restarts Map: {self.binary_restarts_map}")

            # propagate deletions to all nodes
            batch_timer.start()
            for tree_idx, tree_nodes in self.binary_restarts_map.items():
                for node in tree_nodes:
                    node.process_deletions(del_ids)
            propagate_deletions_time = batch_timer.stop()

            # combine answers from all trees
            true_answer, private_answer = initialize_answer_vars(self.query)
            binary_nodes = []
            for tree_idx, tree_nodes in self.binary_restarts_map.items():
                for node in tree_nodes:
                    if tree_idx != current_tree_idx:
                        node.set_rerun(rerun=False)
                    binary_nodes.append(node)

            true_pool = Pool(self.num_threads)
            true_results = true_pool.map(true_answer_worker, (node for node in binary_nodes))
            true_pool.close()
            true_pool.join()

            batch_timer.start()
            private_pool = Pool(self.num_threads)
            private_results = private_pool.map(private_answer_worker, (node for node in binary_nodes))
            private_pool.close()
            private_pool.join()
            private_answers_time = batch_timer.stop()

            for result in true_results:
                true_answer += result

            for result in private_results:
                private_answer += result

            # save answers for current batch
            np.savez(f"{self.save_path_prefix}_true_ans_batch{i}", np.array(true_answer))
            np.savez(f"{self.save_path_prefix}_private_ans_batch{i}", np.array(private_answer))

            # save timing info for current batch
            with open(f"{self.save_path_prefix}_timing_batch{i}.pkl", "wb") as f:
                time_dict = {
                    "node_init_time": node_init_time,
                    "update_trees_time": update_trees_time,
                    "propagate_deletions_time": propagate_deletions_time,
                    "private_answers_time": private_answers_time
                }
                logger.debug(f"Binary Restarts Time: {time_dict}")
                dill.dump({
                    "time": time_dict
                }, f)

            # save state into checkpoint file
            if save_checkpts is True:
                with open(f"{self.save_path_prefix}_checkpt_batch{i}.pkl", "wb") as f:
                    dill.dump({
                        "binary_restarts_map": self.binary_restarts_map,
                    }, f)

            true_answers.append(true_answer)
            private_answers.append(private_answer)

        return np.array(true_answers), np.array(private_answers)


class IntervalRestartsQueryEngine(QueryEngine):
    def __init__(self, dataset, query, epsilon, delta, save_path_prefix, num_threads=8):
        super().__init__()
        self.dataset = dataset
        self.query = query
        self.epsilon = epsilon
        self.delta = delta
        self.interval_restarts_list = []
        self.current_ids = []
        self.save_path_prefix = save_path_prefix
        self.num_threads = num_threads

    def run(self, num_batches=None, start_from_batch_num=None, save_checkpts=False):
        true_answers, private_answers = [], []
        for i, (ins_ids, del_ids) in enumerate(self.dataset.get_batches()):
            if num_batches is not None and i == num_batches:
                break

            # load saved answers until run_from_batch number is reached
            if start_from_batch_num is not None and i < start_from_batch_num:
                saved_true_ans = np.load(f"{self.save_path_prefix}_true_ans_batch{i}.npz")["arr_0"]
                true_answers.append(saved_true_ans)
                saved_private_ans = np.load(f"{self.save_path_prefix}_private_ans_batch{i}.npz")["arr_0"]
                private_answers.append(saved_private_ans)
                continue

            # load state from checkpoint
            if start_from_batch_num is not None and i == start_from_batch_num:
                prev_batch_num = start_from_batch_num - 1
                with open(f"{self.save_path_prefix}_checkpt_batch{prev_batch_num}.pkl", "rb") as f:
                    checkpt = dill.load(f)
                    self.interval_restarts_list = checkpt["interval_restarts_map"]
                    self.current_ids = checkpt["current_ids"]

            print("Batch number:", i)
            print("Insertion IDs:", ins_ids)
            print("Deletion IDs:", del_ids)

            node_i = i + 1
            logger.debug(f"Created Node ID: {node_i}, Level: {utils.get_interval_tree_level(node_i)}")

            batch_timer = Timer(logger=None)

            # update current IDs (all IDs in dataset after the current batch is processed)
            batch_timer.start()
            self.current_ids.extend(ins_ids)
            for del_id in del_ids:
                if del_id in self.current_ids:
                    self.current_ids.remove(del_id)

            # calculate epsilon and delta for current node
            epsilon_for_node = (6 * self.epsilon / ((math.pi ** 2) * (utils.get_interval_tree_level(node_i) ** 2)))
            if self.delta is not None:
                delta_for_node = (6 * self.delta / ((math.pi ** 2) * (utils.get_interval_tree_level(node_i) ** 2)))
            else:
                delta_for_node = self.delta
            logger.debug(f"Node \tEpsilon: {epsilon_for_node}\tDelta: {delta_for_node}")

            # build current node
            if utils.is_power_of_two(node_i):
                # add all current IDs
                ids_for_node = self.current_ids
            else:
                # get the lowest ancestor node ID of current node
                ancestor_node_id = utils.get_interval_tree_lowest_ancestor(node_i)

                # find all current IDs that were added after the lowest ancestor node,
                # node IDs are 1-based, batch numbers are 0-based
                current_ids_df = self.dataset.select_rows_from_ids(self.current_ids)
                current_ids_df = current_ids_df.loc[current_ids_df["insertion_batch"] > (ancestor_node_id - 1)]
                ids_for_node = current_ids_df[self.dataset.id_col].tolist()
            logger.debug(f"IDs for Node {node_i}: {ids_for_node}")

            self.query.set_privacy_parameters(epsilon=epsilon_for_node,
                                              delta=delta_for_node)
            node = RestartNode(ids_for_node, self.query,
                               epsilon=epsilon_for_node, delta=delta_for_node,
                               num_threads=self.num_threads, is_interval=True)
            node_init_time = batch_timer.stop()

            logger.debug(f"Query for Node {node_i}: {self.query}")

            # add current node to map
            self.interval_restarts_list.append(node)
            logger.debug(f"Interval Restarts List: {self.interval_restarts_list}")

            # propagate deletions to all nodes
            batch_timer.start()
            for node in self.interval_restarts_list:
                node.process_deletions(del_ids)
            propagate_deletions_time = batch_timer.stop()

            # get answers
            true_answer, private_answer = initialize_answer_vars(self.query)
            interval_nodes = []
            for node_id in utils.get_interval_tree_nodes_on_rtl_path(node_i):
                # node IDs are 1-based, interval restarts list is 0-based
                interval_nodes.append(self.interval_restarts_list[node_id - 1])

            true_pool = Pool(self.num_threads)
            true_results = true_pool.map(true_answer_worker, (node for node in interval_nodes))
            true_pool.close()
            true_pool.join()

            batch_timer.start()
            private_pool = Pool(self.num_threads)
            private_results = private_pool.map(private_answer_worker, (node for node in interval_nodes))
            private_pool.close()
            private_pool.join()
            private_answers_time = batch_timer.stop()

            for result in true_results:
                true_answer += result

            for result in private_results:
                private_answer += result

            # save answers for current batch
            np.savez(f"{self.save_path_prefix}_true_ans_batch{i}", np.array(true_answer))
            np.savez(f"{self.save_path_prefix}_private_ans_batch{i}", np.array(private_answer))

            # save timing info for current batch
            with open(f"{self.save_path_prefix}_timing_batch{i}.pkl", "wb") as f:
                time_dict = {
                    "node_init_time": node_init_time,
                    "update_trees_time": 0,
                    "propagate_deletions_time": propagate_deletions_time,
                    "private_answers_time": private_answers_time
                }
                logger.debug(f"Interval Restarts Time: {time_dict}")
                dill.dump({
                    "time": time_dict
                }, f)

            # save state into checkpoint file
            if save_checkpts:
                with open(f"{self.save_path_prefix}_checkpt_batch{i}.pkl", "wb") as f:
                    dill.dump({
                        "interval_restarts_map": self.interval_restarts_list,
                        "current_ids": self.current_ids
                    }, f)

            true_answers.append(true_answer)
            private_answers.append(private_answer)

        return np.array(true_answers), np.array(private_answers)


# Testing on the Adult dataset
if __name__ == "__main__":
    for batch_size in [5]:
        print("Batch Size:", batch_size)
        for window_size in [10]:
            print("Window Size:", window_size)
            dataset_prefix = "adult_small"
            dataset_name = f"{dataset_prefix}_batch{batch_size}_window{window_size}"

            time_int = pd.DateOffset(days=1)
            time_int_str = "1day"
            data_encoding_type = "ohe"
            dataset = Dataset.load_from_path(f"../data/{dataset_name}_{data_encoding_type}.csv",
                                             domain_path=f"../data/{dataset_prefix}_{data_encoding_type}_domain.json",
                                             id_col="Person ID",
                                             insertion_time_col="Insertion Time",
                                             deletion_time_col="Deletion Time",
                                             time_interval=time_int,
                                             hist_repr_type=data_encoding_type)
            dataset.save_to_path(f"../data/{dataset_name}_{data_encoding_type}_batched_{time_int_str}.csv")

            query_type = "mwem_pgm"
            comparison_type = "all"
            epsilon = 10.0
            delta = 1e-9
            privstr = "eps" + str(epsilon).replace(".", "_")
            if delta:
                privstr += "del" + str(delta).replace(".", "_").replace("^", "_")
            num_runs = 1
            org_seed = 1234
            exp_save_dir = Path(f"../save/{dataset_name}_{comparison_type}_{query_type}"
                                f"_{privstr}_{num_runs}runs_{org_seed}oseed")
            if not Path.is_dir(exp_save_dir):
                os.mkdir(exp_save_dir)
            start_from_batch_num = None
            num_batches = 2
            predicates = ["sex == 0 & race == 0", "sex == 1 & race == 0",
                          "sex == 0 & race == 1", "sex == 1 & race == 1",
                          "sex == 0 & race == 2", "sex == 1 & race == 2",
                          "sex == 0 & race == 3", "sex == 1 & race == 3",
                          "sex == 0 & race == 4", "sex == 1 & race == 4",
                          "sex == 0 & income == 0", "sex == 1 & income == 0",
                          "sex == 0 & income == 1", "sex == 1 & income == 1",
                          "sex == 0", "sex == 1",
                          "race == 0", "race == 1", "race == 2", "race == 3", "race == 4",
                          "income == 0", "income == 1"]
            num_threads = 4

            # run mechanisms on the same dataset NUM_RUNS number of times
            for run in range(num_runs):
                print("On run number:", run)
                seed = org_seed + run
                rng = np.random.default_rng(seed)

                print("Running Naive Binary Mechanism")
                nb_query = MwemPgmQuery(dataset=dataset, predicates=predicates, k=2, rng=rng)
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

            # for run in range(num_runs):
            #     print("On run number:", run)
            #     seed = org_seed + run
            #     rng = np.random.default_rng(seed)
            #
            #     print("Running Binary Restarts Mechanism")
            #     br_query = MwemPgmQuery(dataset=dataset, predicates=predicates, k=2, rng=rng)
            #     binary_restarts_query_engine = BinaryRestartsQueryEngine(dataset, br_query,
            #                                                              epsilon, delta,
            #                                                              save_path_prefix=f"{exp_save_dir}/run{run}_br",
            #                                                              num_threads=num_threads)
            #     br_true_ans, br_private_ans = binary_restarts_query_engine.run(num_batches=num_batches,
            #                                                                    start_from_batch_num=start_from_batch_num)
            #     print("True Answers:", br_true_ans.tolist())
            #     print("Private Answers:", br_private_ans.tolist())
            #     np.savez(f"{exp_save_dir}/br_true_ans_run{run}", np.array(br_true_ans))
            #     np.savez(f"{exp_save_dir}/br_private_ans_run{run}", np.array(br_private_ans))

            # for run in range(num_runs):
            #     print("On run number:", run)
            #     seed = org_seed + run
            #     rng = np.random.default_rng(seed)
            #
            #     print("Running Interval Restarts Mechanism")
            #     int_query = MwemPgmQuery(dataset=dataset, predicates=predicates, k=2, rng=rng)
            #     interval_restarts_query_engine = IntervalRestartsQueryEngine(dataset, int_query,
            #                                                                  epsilon, delta,
            #                                                                  save_path_prefix=f"{exp_save_dir}/run{run}_int",
            #                                                                  num_threads=num_threads)
            #     int_true_ans, int_private_ans = interval_restarts_query_engine.run(num_batches=num_batches,
            #                                                                        start_from_batch_num=start_from_batch_num)
            #     print("True Answers:", int_true_ans.tolist())
            #     print("Private Answers:", int_private_ans.tolist())
            #     np.savez(f"{exp_save_dir}/int_true_ans_run{run}", np.array(int_true_ans))
            #     np.savez(f"{exp_save_dir}/int_private_ans_run{run}", np.array(int_private_ans))
