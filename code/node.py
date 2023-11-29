import math
from abc import ABC, abstractmethod
from copy import deepcopy
from multiprocessing import Pool

import numpy as np

import utils
from query import initialize_answer_var, initialize_answer_vars, CountQuery


def private_answer_worker(node):
    return node.get_private_answer()


class Node(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def get_true_answer(self):
        pass

    @abstractmethod
    def get_private_answer(self):
        pass

    @abstractmethod
    def __repr__(self):
        pass


class NaiveNode(Node):
    def __init__(self, ids, query):
        super().__init__()
        self.ids = ids
        self.query = query
        self.true_answer = None
        self.private_answer = None

        # caching
        self.rerun = True

        self.compute_answers()

    def compute_answers(self):
        self.true_answer = self.query.get_true_answer(self.ids)
        self.private_answer = self.query.get_private_answer(self.ids, self.rerun)

    def merge_node(self, node):
        self.ids = self.ids + node.ids
        self.set_rerun(rerun=True)  # need to regenerate answers after merging
        self.compute_answers()

    def set_rerun(self, rerun):
        self.rerun = rerun

    def get_true_answer(self):
        return self.true_answer

    def get_private_answer(self):
        return self.private_answer

    def __repr__(self):
        return f"IDs: {self.ids}"


class RestartNode(Node):
    def __init__(self, ins_ids, query, epsilon, delta=None, beta=0.15, num_threads=8, is_interval=False):
        super().__init__()
        self.ins_ids = ins_ids
        self.query = query

        # set number of restarts
        self.num_restarts = 1

        # flag for updating parameters after restarts: Binary Restarts = false, Interval Restarts = true
        self.is_interval = is_interval

        # flag for halting Interval Restarts
        self.halt_node = False

        # privacy parameters to distribute over deletion streams
        if is_interval:
            self.epsilon = (3 * epsilon / 2 * (math.pi ** 2) * (self.num_restarts ** 2))
            if delta is not None:
                self.delta = (2 * delta / (math.pi ** 2) * (self.num_restarts ** 2))
            else:
                self.delta = delta
            self.beta = beta * (1 / (math.pi ** 2))
        else:
            self.epsilon = epsilon
            self.delta = delta
            self.beta = beta

        # noisy estimate of the current number of ids (every query returns a list, so we take the first value)
        self.num_ins_ids = CountQuery(sensitivity=1, epsilon=self.epsilon).get_private_answer(self.ins_ids)[0]

        # variables that store the final answers for the node
        self.true_answer = None
        self.ins_ids_private_answer = None
        self.private_answer = None

        # caching and optimization
        self.rerun = True
        self.num_threads = num_threads

        self.compute_answers()

        # initialize deletion streams
        self.del_ids = []
        self.naive_binary_deletions_map = {}
        self.current_tree_idx_nb_deletions, self.node_i_nb_deletions = 0, 0
        self.num_nodes_nb_deletions = 0
        self.naive_binary_count_deletions_map = {}
        self.current_tree_idx_nb_del_count, self.node_i_nb_del_count = 0, 0
        self.num_nodes_nb_del_count = 0

    def compute_answers(self):
        self.query.set_privacy_parameters(epsilon=self.epsilon, delta=self.delta)
        self.true_answer = self.query.get_true_answer(self.ins_ids)

        if self.is_interval and self.halt_node is True:
            self.private_answer = initialize_answer_vars(self.query)
        else:
            self.ins_ids_private_answer = self.query.get_private_answer(self.ins_ids, self.rerun)
            self.private_answer = self.ins_ids_private_answer

    def merge_node(self, node):
        self.ins_ids = self.ins_ids + node.ins_ids  # merge insertion ids

        # merge deletion streams (self.node is guaranteed to be the earlier node, so counters don't need to be updated)
        for tree_idx, tree_nodes in self.naive_binary_deletions_map.items():
            to_be_merged_nodes = node.naive_binary_deletions_map.get(tree_idx, [])
            for (node_1, node_2) in zip(tree_nodes, to_be_merged_nodes):  # zip will merge acc. length of shorter list
                node_1.merge_node(node_2)
        for tree_idx, tree_nodes in self.naive_binary_count_deletions_map.items():
            to_be_merged_nodes = node.naive_binary_count_deletions_map.get(tree_idx, [])
            for (node_1, node_2) in zip(tree_nodes, to_be_merged_nodes):  # zip will merge acc. length of shorter list
                node_1.merge_node(node_2)

        self.set_rerun(rerun=True)  # need to regenerate answers after merging

        self.compute_answers()  # recompute answers

    def set_rerun(self, rerun):
        self.rerun = rerun

    def process_deletions(self, received_del_ids):
        for del_id in received_del_ids:
            if del_id in self.ins_ids:
                # send del id to deletion streams
                self.del_ids.append(del_id)
                self.update_naive_binary_deletions_map(del_id)
                self.update_naive_binary_count_deletions_map(del_id)
            else:
                # no op sent to deletion streams
                self.update_naive_binary_deletions_map(None)
                self.update_naive_binary_count_deletions_map(None)

        # check if node can be restarted
        num_deletions = self.get_answer_from_naive_binary_count_deletions_map()[0]  # every query returns a list
        num_deletions_error = (1 / self.epsilon
                               * np.power(np.log2(max(self.num_nodes_nb_del_count, 1)), 1.5)  # TODO: check t
                               * np.log2(self.beta))
        if num_deletions > ((self.num_ins_ids / 2) + 2 * num_deletions_error):
            self.restart()
        else:
            del_ids_private_answer = self.get_answer_from_naive_binary_deletions_map()
            self.private_answer = self.ins_ids_private_answer - del_ids_private_answer

    def update_naive_binary_deletions_map(self, del_id):
        self.node_i_nb_deletions += 1

        # start building new tree
        tree_idx = utils.get_tree_idx(self.node_i_nb_deletions)
        if tree_idx != self.current_tree_idx_nb_deletions:
            self.num_nodes_nb_deletions = 0
            self.current_tree_idx_nb_deletions = tree_idx

        # add current node to map
        query = deepcopy(self.query)
        # TODO: Check epsilon
        query.set_privacy_parameters(epsilon=self.epsilon / utils.get_tree_height(self.node_i_nb_deletions))
        tree_nodes = self.naive_binary_deletions_map.get(self.current_tree_idx_nb_deletions, [])
        if del_id is not None:
            del_id_list = [del_id]
        else:
            del_id_list = []
        tree_nodes.append(NaiveNode(del_id_list, query))
        self.num_nodes_nb_deletions += 1

        # update tree by merging nodes
        n = self.num_nodes_nb_deletions
        while n > 0:
            if n % 2 == 0:
                # remove and merge last two nodes in the list
                second_node = tree_nodes.pop()  # trees_node[-1]
                merged_node = tree_nodes.pop()  # trees_node[-2]
                merged_node.merge_node(second_node)
                tree_nodes.append(merged_node)
            n = n / 2
        self.naive_binary_deletions_map[self.current_tree_idx_nb_deletions] = tree_nodes

    def update_naive_binary_count_deletions_map(self, del_id):
        self.node_i_nb_del_count += 1

        # start building new tree
        tree_idx = utils.get_tree_idx(self.node_i_nb_del_count)
        if tree_idx != self.current_tree_idx_nb_del_count:
            self.num_nodes_nb_del_count = 0
            self.current_tree_idx_nb_del_count = tree_idx

        # add current node to map
        # TODO: Check epsilon
        count_query = CountQuery(sensitivity=1,
                                 epsilon=self.epsilon / utils.get_tree_height(self.node_i_nb_deletions))
        tree_nodes = self.naive_binary_count_deletions_map.get(self.current_tree_idx_nb_del_count, [])
        if del_id is not None:
            del_id_list = [del_id]
        else:
            del_id_list = []
        tree_nodes.append(NaiveNode(del_id_list, count_query))
        self.num_nodes_nb_del_count += 1

        # update tree by merging nodes
        n = self.num_nodes_nb_del_count
        while n > 0:
            if n % 2 == 0:
                # remove and merge last two nodes in the list
                second_node = tree_nodes.pop()  # trees_node[-1]
                merged_node = tree_nodes.pop()  # trees_node[-2]
                merged_node.merge_node(second_node)
                tree_nodes.append(merged_node)
            n = n / 2
        self.naive_binary_count_deletions_map[self.current_tree_idx_nb_del_count] = tree_nodes

    def get_answer_from_naive_binary_deletions_map(self):
        answer = initialize_answer_var(self.query)
        nodes = []
        for tree_idx, tree_nodes in self.naive_binary_deletions_map.items():
            for node in tree_nodes:
                if tree_idx != self.current_tree_idx_nb_deletions:
                    node.set_rerun(rerun=False)
                nodes.append(node)

        pool = Pool(self.num_threads)
        pool_results = pool.map(private_answer_worker, (node for node in nodes))
        pool.close()
        pool.join()

        for result in pool_results:
            answer += result

        return answer

    def get_answer_from_naive_binary_count_deletions_map(self):
        answer = initialize_answer_var(CountQuery())
        nodes = []
        for tree_idx, tree_nodes in self.naive_binary_count_deletions_map.items():
            for node in tree_nodes:
                if tree_idx != self.current_tree_idx_nb_del_count:
                    node.set_rerun(rerun=False)
                nodes.append(node)

        pool = Pool(self.num_threads)
        pool_results = pool.map(private_answer_worker, (node for node in nodes))
        pool.close()
        pool.join()

        for result in pool_results:
            answer += result

        return answer

    def get_true_answer(self):
        return self.true_answer

    def get_private_answer(self):
        return self.private_answer

    def restart(self):
        # update number of restarts
        self.num_restarts = self.num_restarts + 1

        # remove previously deleted items from the insertion ids
        for del_id in self.del_ids:
            self.ins_ids.remove(del_id)

        # need to regenerate answers after merging
        self.rerun = True

        # set new values for privacy parameters and beta
        if self.is_interval:
            # Interval Restarts
            self.epsilon = (3 * self.epsilon / 2 * (math.pi ** 2) * (self.num_restarts ** 2))
            if self.delta:
                self.delta = (2 * self.delta / (math.pi ** 2) * (self.num_restarts ** 2))
            self.beta = (self.beta / (math.pi ** 2) * (self.num_restarts ** 2))
        else:
            # Binary Restarts
            self.epsilon = self.epsilon / utils.get_tree_height(self.node_i_nb_deletions)
            if self.delta:
                self.delta = self.delta / utils.get_tree_height(self.node_i_nb_deletions)
            self.beta = (self.beta / utils.get_tree_height(self.node_i_nb_deletions))

        # restart deletion streams
        self.del_ids = []
        self.naive_binary_deletions_map = {}
        self.current_tree_idx_nb_deletions, self.node_i_nb_deletions = 0, 0
        self.num_nodes_nb_deletions = 0
        self.naive_binary_count_deletions_map = {}
        self.current_tree_idx_nb_del_count, self.node_i_nb_del_count = 0, 0
        self.num_nodes_nb_del_count = 0

        # set new noisy estimate for the number of ids (every query returns a list, so we take the first value)
        self.num_ins_ids = CountQuery(sensitivity=1, epsilon=self.epsilon).get_private_answer(self.ins_ids)[0]

        # halt Interval Restarts for future calls if noisy estimate for the number of ids
        # is less than num_deletions_error
        num_deletions_error = (1 / self.epsilon
                               * np.power(np.log2(max(self.num_nodes_nb_del_count, 1)), 1.5)  # TODO: check t
                               * np.log2(self.beta))
        if self.num_ins_ids < 2 * num_deletions_error:
            print("Halted node")
            self.halt_node = True

        # recompute answers
        self.compute_answers()

    def __repr__(self):
        return f"IDs: {self.ins_ids}\tDel IDs: {self.del_ids}"
