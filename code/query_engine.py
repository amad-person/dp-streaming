import os
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
import pandas as pd

import utils
from dataset import Dataset
from node import NaiveNode, RestartNode
from query import PmwQuery


class QueryEngine(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def run(self):
        pass


class NaiveBinaryQueryEngine(QueryEngine):
    def __init__(self, dataset, query, epsilon, delta):
        super().__init__()
        self.dataset = dataset
        self.query = query
        self.epsilon = epsilon
        self.delta = delta
        self.naive_binary_insertions_map = {}  # stream to track insertions
        self.naive_binary_deletions_map = {}  # stream to track deletions

    def run(self, num_batches=None):
        true_answers, private_answers = [], []
        num_nodes, current_tree_idx = 0, 0
        for i, (ins_ids, del_ids) in enumerate(self.dataset.get_batches()):
            if i == num_batches:
                break

            node_i = i + 1

            # start building new tree
            tree_idx = utils.get_tree_idx(node_i)
            if tree_idx != current_tree_idx:
                num_nodes = 0
                current_tree_idx = tree_idx

            # build current nodes
            self.query.set_privacy_parameters(epsilon=self.epsilon / utils.get_tree_height(node_i),
                                              delta=self.delta)
            ins_node = NaiveNode(ins_ids, self.query)
            del_node = NaiveNode(del_ids, self.query)
            num_nodes += 1

            # add current nodes to maps
            ins_tree_nodes = self.naive_binary_insertions_map.get(current_tree_idx, [])
            del_tree_nodes = self.naive_binary_deletions_map.get(current_tree_idx, [])
            ins_tree_nodes.append(ins_node)
            del_tree_nodes.append(del_node)

            # update maps by merging nodes
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

            # combine answers from all trees
            true_answer, private_answer = utils.initialize_answer_vars(self.query)
            for tree_idx, tree_nodes in self.naive_binary_insertions_map.items():
                for node in tree_nodes:
                    true_answer += node.get_true_answer()
                    private_answer += node.get_private_answer()
            for tree_idx, tree_nodes in self.naive_binary_deletions_map.items():
                for node in tree_nodes:
                    true_answer -= node.get_true_answer()
                    private_answer -= node.get_private_answer()

            true_answers.append(true_answer)
            private_answers.append(private_answer)

        return np.array(true_answers), np.array(private_answers)


class BinaryRestartsQueryEngine(QueryEngine):
    def __init__(self, dataset, query, epsilon, delta):
        super().__init__()
        self.dataset = dataset
        self.query = query
        self.epsilon = epsilon
        self.delta = delta
        self.binary_restarts_map = {}

    def run(self, num_batches=None):
        true_answers, private_answers = [], []
        num_nodes, current_tree_idx = 0, 0
        for i, (ins_ids, del_ids) in enumerate(self.dataset.get_batches()):
            if i == num_batches:
                break

            node_i = i + 1

            # start building new tree
            tree_idx = utils.get_tree_idx(node_i)
            if tree_idx != current_tree_idx:
                num_nodes = 0
                current_tree_idx = tree_idx

            # build current node
            self.query.set_privacy_parameters(epsilon=self.epsilon / utils.get_tree_height(node_i),
                                              delta=self.delta)
            node = RestartNode(ins_ids, self.query, epsilon=self.epsilon / utils.get_tree_height(node_i))
            num_nodes += 1

            # add current node to map
            tree_nodes = self.binary_restarts_map.get(current_tree_idx, [])
            tree_nodes.append(node)

            # update map by merging nodes
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

            # propagate deletions to all nodes
            for tree_idx, tree_nodes in self.binary_restarts_map.items():
                for node in tree_nodes:
                    node.process_deletions(del_ids)

            # combine answers from all trees
            true_answer, private_answer = utils.initialize_answer_vars(self.query)
            for tree_idx, tree_nodes in self.binary_restarts_map.items():
                for node in tree_nodes:
                    true_answer += node.get_true_answer()
                    private_answer += node.get_private_answer()

            true_answers.append(true_answer)
            private_answers.append(private_answer)

        return np.array(true_answers), np.array(private_answers)


# Testing on the Adult dataset
if __name__ == "__main__":
    dataset_name = "adult_small"
    time_int = pd.DateOffset(days=1)
    time_int_str = "1day"
    pmw_encoding_type = "binarized"
    dataset = Dataset.load_from_path(f"../data/{dataset_name}_{pmw_encoding_type}.csv",
                                     domain_path=f"../data/{dataset_name}_{pmw_encoding_type}_domain.json",
                                     id_col="Person ID",
                                     insertion_time_col="Insertion Time",
                                     deletion_time_col="Deletion Time",
                                     time_interval=time_int,
                                     hist_repr_type=pmw_encoding_type)
    dataset.save_to_path(f"../data/{dataset_name}_{pmw_encoding_type}_batched_{time_int_str}.csv")

    query_type = "pmw"
    epsilon = 10.0
    delta = None
    privstr = "eps" + str(epsilon).replace(".", "_")
    if delta:
        privstr += "del" + str(delta).replace(".", "_").replace("^", "_")
    num_runs = 3
    org_seed = 1234
    exp_save_dir = Path(f"../save/{dataset_name}_nb_vs_br_{query_type}_{privstr}_{num_runs}runs_{org_seed}oseed")
    if not Path.is_dir(exp_save_dir):
        os.mkdir(exp_save_dir)

    predicates = ['sex == 0 & race == 1', 'sex == 0']

    # run mechanisms on the same dataset NUM_RUNS number of times
    for run in range(num_runs):
        print("On run number:", run)
        seed = org_seed + run
        rng = np.random.default_rng(seed)

        print("Running Naive Binary Mechanism")
        nb_query = PmwQuery(dataset=dataset, predicates=predicates, k=2, iterations=25, rng=rng)
        naive_binary_query_engine = NaiveBinaryQueryEngine(dataset, nb_query, epsilon, delta)
        nb_true_ans, nb_private_ans = naive_binary_query_engine.run(num_batches=2)
        print("True Answers:", nb_true_ans.tolist())
        print("Private Answers:", nb_private_ans.tolist())
        np.savez(f"{exp_save_dir}/nb_true_ans_run{run}", np.array(nb_true_ans))
        np.savez(f"{exp_save_dir}/nb_private_ans_run{run}", np.array(nb_private_ans))

        print("Running Binary Restarts Mechanism")
        br_query = PmwQuery(dataset=dataset, predicates=predicates, k=2, iterations=25, rng=rng)
        binary_restarts_query_engine = BinaryRestartsQueryEngine(dataset, br_query, epsilon, delta)
        br_true_ans, br_private_ans = binary_restarts_query_engine.run(num_batches=2)
        print("True Answers:", br_true_ans.tolist())
        print("Private Answers:", br_private_ans.tolist())
        np.savez(f"{exp_save_dir}/br_true_ans_run{run}", np.array(br_true_ans))
        np.savez(f"{exp_save_dir}/br_private_ans_run{run}", np.array(br_private_ans))
