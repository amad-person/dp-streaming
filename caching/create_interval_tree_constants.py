class TreeNode:
    def __init__(self, val):
        self.val = val
        self.left = None
        self.right = None


def create_tree(values, left, right):
    if right < left:
        return None

    mid_idx = int(left + ((right - left) / 2))

    node = TreeNode(values[mid_idx])

    node.left = create_tree(values, left, mid_idx - 1)
    node.right = create_tree(values, mid_idx + 1, right)

    return node


def print_binary_tree(root):
    def print_tree(node, depth=0, prefix="Root:"):
        if node is not None:
            print(" " * (depth * 4) + prefix, node.val)
            if node.left or node.right:
                if node.left:
                    print_tree(node.left, depth + 1, "L---")
                else:
                    print(" " * ((depth + 1) * 4) + "L---None")
                if node.right:
                    print_tree(node.right, depth + 1, "R---")
                else:
                    print(" " * ((depth + 1) * 4) + "R---None")

    print_tree(root)


def find_path_to_val(root, val):
    def dfs(node, path):
        if node is None:
            return None

        path.append(node.val)

        if node.val == val:
            return path

        left_path = dfs(node.left, path.copy())
        if left_path:
            return left_path

        right_path = dfs(node.right, path.copy())
        if right_path:
            return right_path

        return None

    path = dfs(root, [])
    return path


if __name__ == "__main__":
    max_value = 1023
    sorted_values = list(range(1, max_value + 1))
    root_node = create_tree(sorted_values, 0, len(sorted_values) - 1)

    # print_binary_tree(root_node)

    for val in sorted_values:
        path_to_val = find_path_to_val(root_node, val)
        print(val, path_to_val)

        ancestor = None
        for par in reversed(path_to_val):
            if par < val:
                ancestor = par
                break
        print(val, ancestor)
