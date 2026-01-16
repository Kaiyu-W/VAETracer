from collections import defaultdict
import ete3
import networkx as nx

def make_named_nodes_unique(tree: ete3.Tree) -> None:
    """
    Ensure non-empty node names are unique.
    The first occurrence (closest to root, via preorder) keeps the original name.
    Subsequent duplicates are renamed as {name}_1, {name}_2, ...
    Empty names ('') are left untouched.
    """
    seen = set()
    name_suffix = defaultdict(int)  # track next suffix for each base name

    # Use preorder traversal: root first, then children
    for node in tree.traverse(strategy="preorder"):
        if node.name == "":
            continue

        if node.name not in seen:
            # First time seeing this name -> keep it
            seen.add(node.name)
        else:
            # Already seen -> assign suffix
            base = node.name
            name_suffix[base] += 1
            node.name = f"{base}_{name_suffix[base]}"
            # Also add the new name to seen to avoid further conflicts
            seen.add(node.name)

def newick_to_networkx(newick_string: str, format=3) -> nx.DiGraph:
    tree = ete3.Tree(newick_string, format=format)
    make_named_nodes_unique(tree)
    return ete3_to_networkx(tree)

def ete3_to_networkx(tree: ete3.Tree) -> nx.DiGraph:
    g = nx.DiGraph()
    internal_node_iter = 0

    for n in tree.traverse():
        if n.name == "":
            n.name = f"cassiopeia_internal_node{internal_node_iter}"
            internal_node_iter += 1

        if not n.is_root():
            edge_length = n.dist
            if edge_length > 0:
                g.add_edge(n.up.name, n.name, length=edge_length)
            else:
                g.add_edge(n.up.name, n.name)

    return g

def networkx_to_ete3(g: nx.DiGraph) -> ete3.Tree:
    if not nx.is_directed_acyclic_graph(g):
        raise ValueError("Input must be a directed acyclic graph (DAG).")
    
    if not nx.is_tree(g):
        raise ValueError("Input must be a tree (i.e., connected and |E| = |V| - 1).")

    root_nodes = [n for n in g.nodes if g.in_degree(n) == 0]
    if len(root_nodes) != 1:
        raise ValueError(f"Expected exactly one root node, found {len(root_nodes)}: {root_nodes}")
    root_name = root_nodes[0]

    root = ete3.TreeNode(name=root_name)
    stack = [(root, root_name)]

    while stack:
        parent_node, parent_name = stack.pop()

        for child_name in g.successors(parent_name):
            edge_data = g.get_edge_data(parent_name, child_name)
            length = edge_data.get('length', 0.0) if edge_data else 0.0

            child_node = ete3.TreeNode(name=child_name, dist=length)

            parent_node.add_child(child_node)

            stack.append((child_node, child_name))

    return root

