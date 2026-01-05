import numpy as np
from cassiopeia.data import CassiopeiaTree
from scipy.cluster.hierarchy import dendrogram

from .newick import newick_to_networkx
# CassiopeiaTree.remove_leaves_and_prune_lineages => create new tree from old tree with only partial leaves 

def newick_to_CassiopeiaTree(newick_string: str, format=None) -> CassiopeiaTree:
    
    tree = CassiopeiaTree(
        tree=newick_to_networkx(
            newick_string,
            format=1 if format is None else format
        )
    )

    return tree


def make_ultrametric_and_get_linkage(tree: CassiopeiaTree):
    root_distance = {}
    bl = 1

    def calc_distances(node, dist=0.0):
        root_distance[node] = dist
        for child in tree.children(node):
            calc_distances(child, dist + bl)
    
    calc_distances(tree.root, 0.0)

    leaf_distances = {leaf: root_distance[leaf] for leaf in tree.leaves}
    max_distance = max(leaf_distances.values())

    ultrametric_root_distance = {}

    def adjust_to_ultrametric(node, target_distance):
        ultrametric_root_distance[node] = target_distance
        children = list(tree.children(node))
        if not children:
            return

        child_target = target_distance - 1.0
        for child in children:
            adjust_to_ultrametric(child, child_target)
    
    adjust_to_ultrametric(tree.root, max_distance + 1)

    leaves = list(tree.leaves)
    node_to_id = {leaf: i for i, leaf in enumerate(leaves)}
    next_internal_id = len(leaves)
    linkage_matrix = []
    node_to_linkage_id = {}

    def postorder(node):
        nonlocal next_internal_id
        if tree.is_leaf(node):
            node_id = node_to_id[node]
            node_to_linkage_id[node] = node_id
            return node_id

        children = list(tree.children(node))
        child_ids = [postorder(child) for child in children]

        merge_height = max(ultrametric_root_distance[child] for child in children)

        current_id = child_ids[0]
        for i in range(1, len(child_ids)):
            next_id = child_ids[i]
            id1, id2 = sorted([current_id, next_id])
            total_leaves = sum(len(tree.leaves_in_subtree(children[j])) for j in range(i+1))
            linkage_matrix.append([id1, id2, merge_height, total_leaves])
            current_id = next_internal_id
            next_internal_id += 1

        node_to_linkage_id[node] = current_id
        return current_id

    postorder(tree.root)
    return np.array(linkage_matrix).astype(float)

def get_tree_linkage(tree: CassiopeiaTree, plot=False):
    original_leaves = tree.leaves
    linkage_mat = make_ultrametric_and_get_linkage(tree)
    dendro = dendrogram(
        linkage_mat,
        labels=original_leaves,
        no_plot=not plot
    )
    ordered_leaves = [
        original_leaves[i] for i in dendro['leaves']
    ]
    
    return linkage_mat, original_leaves, ordered_leaves