import numpy as np
from cassiopeia.data import CassiopeiaTree
from scipy.cluster.hierarchy import dendrogram
from typing import Tuple, List

from .newick import newick_to_networkx
# CassiopeiaTree.remove_leaves_and_prune_lineages => create new tree from old tree with only partial leaves 

def newick_to_CassiopeiaTree(newick_string: str, format=None) -> CassiopeiaTree:
    """
    Convert a Newick-formatted tree (string or file path) into a CassiopeiaTree object.

    Parses the input using ete3 via newick_to_networkx, which supports both raw Newick strings
    and file paths. The resulting networkx graph is wrapped in a CassiopeiaTree for integration
    with downstream phylogenetic analysis tools in VAETracer.

    Args:
        newick_string (str): Either:
            - A string containing a Newick-formatted tree (e.g., "((A,B),C);"), OR
            - A file path to a text file containing a Newick tree.
        format (int, optional): Newick parsing format used by ete3.Tree constructor.
                                 Defaults to 1 if not provided.

    Returns:
        CassiopeiaTree: Initialized tree object with topology and branch lengths preserved.

    Raises:
        FileNotFoundError: If the provided path does not exist.
        ete3.parser.newick.NewickError: If the Newick content is malformed.
        ValueError: If the tree structure is invalid for CassiopeiaTree initialization.
    """
    tree = CassiopeiaTree(
        tree=newick_to_networkx(
            newick_string,
            format=1 if format is None else format
        )
    )

    return tree


def make_ultrametric_and_get_linkage(tree: CassiopeiaTree) -> np.ndarray:
    """
    Transform a CassiopeiaTree into an ultrametric representation and generate its linkage matrix.

    First computes root-to-node distances assuming unit branch length (bl=1). Then adjusts all paths 
    so that leaf nodes are equidistant from the root (ultrametric property), enabling hierarchical 
    clustering visualization. A scipy-compatible linkage matrix is constructed via postorder traversal, 
    where each merge event records cluster IDs, height, and size.

    Note:
        This method uses a simplified model: all branches are treated as length 1 regardless of original values.
        For applications requiring true branch length preservation, consider alternative scaling methods.

    Args:
        tree (CassiopeiaTree): Input phylogenetic tree.

    Returns:
        np.ndarray: A (n-1) x 4 linkage matrix compatible with scipy.cluster.hierarchy.dendrogram,
                    where n = number of leaves. Each row: [id1, id2, height, count].

    Side Effects:
        Internal node ordering is determined by postorder traversal; leaf IDs are assigned based on 
        their order in tree.leaves.
    """
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

def get_tree_linkage(tree: CassiopeiaTree, plot=False) -> Tuple[np.ndarray, List[str], List[str]]:
    """
    Generate a hierarchical clustering linkage matrix from a CassiopeiaTree and extract leaf orderings.

    Wraps make_ultrametric_and_get_linkage to produce a dendrogram-compatible structure.
    Optionally plots the dendrogram using scipy's dendrogram function.

    Args:
        tree (CassiopeiaTree): Input phylogenetic tree.
        plot (bool): If True, renders the dendrogram plot. Otherwise, only computes data.

    Returns:
        tuple:
            - np.ndarray: Linkage matrix (from make_ultrametric_and_get_linkage).
            - list: Original list of leaf names in CassiopeiaTree.leaves order.
            - list: Leaf name ordering induced by dendrogram traversal (useful for reordering matrices).

    Example:
        linkage_mat, orig_leaves, ordered_leaves = get_tree_linkage(ctree, plot=True)
        # Use `ordered_leaves` to reorder gene expression matrix for visualization
    """
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