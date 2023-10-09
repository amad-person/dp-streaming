import numpy as np
import pandas as pd
from dataset import create_fake_dataset, Dataset
from query import CountQuery
from node import NaiveNode
import utils


if __name__ == "__main__":
    n_rows = 10000
    create_fake_dataset(n_rows)

    time_int = pd.DateOffset(days=1)
    time_int_str = "1day"
    fake_dataset = Dataset.load_from_path(f"fake_dataset_{n_rows}.csv",
                                          id_col="Person ID",
                                          insertion_time_col="Insertion Time",
                                          deletion_time_col="Deletion Time",
                                          time_interval=time_int)
    fake_dataset.save_to_path(f"fake_dataset_{n_rows}_batched_{time_int_str}.csv")

    epsilon = 10.0
    true_answers, private_answers = [], []

    naive_binary_insertions_map = {}
    naive_binary_deletions_map = {}
    num_nodes, current_tree_idx = 0, 0
    for i, (ins_ids, del_ids) in enumerate(fake_dataset.get_batches()):
        node_i = i + 1

        # start building new tree
        if utils.get_tree_idx(node_i) != current_tree_idx:
            num_nodes = 0
            current_tree_idx = utils.get_tree_idx(node_i)

        # build current node
        count_query = CountQuery(epsilon, sensitivity=utils.get_tree_height(node_i))
        insertions_node = NaiveNode(ins_ids, count_query)
        deletions_node = NaiveNode(del_ids, count_query)

        # TODO: update maps
        insertions_tree_nodes = naive_binary_insertions_map.get(current_tree_idx, {})
        deletions_tree_nodes = naive_binary_deletions_map.get(current_tree_idx, {})

        # combine answers from all trees
        true_answer = 0
        private_answer = 0
        for tree_idx, tree_nodes in naive_binary_insertions_map.items():
            for node in tree_nodes:
                true_answer += node.get_true_answer()
                private_answer += node.get_private_answer()
        for tree_idx, tree_nodes in naive_binary_deletions_map.items():
            for node in tree_nodes:
                true_answer -= node.get_true_answer()
                private_answer -= node.get_private_answer()

        true_answers.append(true_answer)
        private_answers.append(private_answer)