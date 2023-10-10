import pandas as pd
from dataset import create_toy_dataset, create_fake_dataset, Dataset
from query import CountQuery
from node import NaiveNode
import utils


# Testing
if __name__ == "__main__":
    # n_rows = 10000
    # create_fake_dataset(n_rows)
    # time_int = pd.DateOffset(days=1)
    # time_int_str = "1day"
    # dataset = Dataset.load_from_path(f"fake_dataset_{n_rows}.csv",
    #                                  id_col="Person ID",
    #                                  insertion_time_col="Insertion Time",
    #                                  deletion_time_col="Deletion Time",
    #                                  time_interval=time_int)
    # dataset.save_to_path(f"fake_dataset_{n_rows}_batched_{time_int_str}.csv")

    create_toy_dataset()
    time_int = pd.DateOffset(days=1)
    dataset = Dataset.load_from_path("toy_dataset.csv",
                                     id_col="Person ID",
                                     insertion_time_col="Insertion Time",
                                     deletion_time_col="Deletion Time",
                                     time_interval=time_int)
    dataset.save_to_path("toy_dataset_batched_1day.csv")

    epsilon = 10.0
    true_answers, private_answers = [], []

    naive_binary_insertions_map = {}
    naive_binary_deletions_map = {}
    num_nodes, current_tree_idx = 0, 0
    for i, (ins_ids, del_ids) in enumerate(dataset.get_batches()):
        node_i = i + 1
        print(f"Batch: {node_i}")

        # start building new tree
        if utils.get_tree_idx(node_i) != current_tree_idx:
            num_nodes = 0
            current_tree_idx = utils.get_tree_idx(node_i)

        # build current nodes
        count_query = CountQuery(epsilon, sensitivity=utils.get_tree_height(node_i))
        ins_node = NaiveNode(ins_ids, count_query)
        del_node = NaiveNode(del_ids, count_query)
        num_nodes += 1

        # add current nodes to maps
        ins_tree_nodes = naive_binary_insertions_map.get(current_tree_idx, [])
        del_tree_nodes = naive_binary_deletions_map.get(current_tree_idx, [])
        ins_tree_nodes.append(ins_node)
        del_tree_nodes.append(del_node)

        # update maps by merging nodes
        n = num_nodes
        while n > 0:
            if n % 2 == 0:
                # remove and merge last two nodes in the list
                second_ins_node = ins_tree_nodes.pop()  # ins_trees_node[-1]
                merged_ins_node = ins_tree_nodes.pop()  # ins_trees_node[-2]
                merged_ins_node.merge_node(second_ins_node)
                ins_tree_nodes.append(merged_ins_node)

                second_del_node = del_tree_nodes.pop()  # del_trees_node[-1]
                merged_del_node = del_tree_nodes.pop()  # del_trees_node[-2]
                merged_del_node.merge_node(second_del_node)
                del_tree_nodes.append(merged_del_node)
            n = n / 2
        naive_binary_insertions_map[current_tree_idx] = ins_tree_nodes
        naive_binary_deletions_map[current_tree_idx] = del_tree_nodes
        print(naive_binary_insertions_map)
        print(naive_binary_deletions_map)

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

        print(f"True Answer: {true_answer}")
        print(f"Private Answer: {private_answer}")

        true_answers.append(true_answer)
        private_answers.append(private_answer)
