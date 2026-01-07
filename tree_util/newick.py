from collections import defaultdict
import ete3
import networkx as nx

def make_named_nodes_unique(tree: ete3.Tree) -> None:
    """
    Modify the input ete3 tree in-place to ensure all non-empty node names are unique.

    Traverses the tree in preorder (root first). The first occurrence of a name retains it;
    subsequent duplicates are renamed with incremental suffixes (e.g., 'name_1', 'name_2').
    Empty node names ('') are preserved and not modified.

    Args:
        tree (ete3.Tree): An ete3 Tree object whose nodes may have duplicate names.
                         Modified in place.

    Side Effects:
        Alters the .name attribute of nodes in the input tree to enforce uniqueness.
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
    """
    Parse a Newick tree (either as string content or file path) into a directed NetworkX graph.

    Uses ete3.Tree to parse the input. If `newick_string` is a file path, ete3 will automatically 
    read and parse its contents. Otherwise, it treats the string as raw Newick text.

    After parsing, ensures node names are unique (via make_named_nodes_unique), then converts 
    the tree structure into a directed graph with branch lengths stored as edge attributes.

    Args:
        newick_string (str): Either:
            - A string containing a Newick-formatted tree (e.g., "((A,B),C);"), OR
            - A file path pointing to a text file containing a Newick tree.
        format (int, optional): Parsing format passed to ete3.Tree constructor. Default is 3.

    Returns:
        nx.DiGraph: A directed graph representation of the tree topology and branch lengths.
                    Edge attribute 'length' stores branch length (distance from parent).
                    Internal nodes are named uniquely if originally unnamed.

    Raises:
        ete3.parser.newick.NewickError: If the Newick content is malformed.
        FileNotFoundError: If a file path is provided but the file does not exist.
        ValueError: If the parsed tree has structural issues (e.g., missing root).
    """
    tree = ete3.Tree(newick_string, format=format)
    make_named_nodes_unique(tree)
    return ete3_to_networkx(tree)

def ete3_to_networkx(tree: ete3.Tree) -> nx.DiGraph:
    """
    Convert an ete3 Tree object into a directed NetworkX graph.

    Each node becomes a vertex; each parent-child link becomes a directed edge.
    Branch lengths (`.dist`) are stored as edge attribute 'length'. Unnamed nodes are assigned 
    unique placeholder names (e.g., cassiopeia_internal_node0).

    Args:
        tree (ete3.Tree): Input phylogenetic tree.

    Returns:
        nx.DiGraph: Directed graph representing the tree topology and branch lengths.
                    Nodes are identified by their (possibly auto-assigned) names.
    """
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
    """
    Convert a directed NetworkX tree into an ete3 Tree object.

    Validates that the input is a valid rooted tree (DAG, connected, single root),
    then constructs the corresponding ete3.Tree via depth-first traversal starting from the root.
    Edge 'length' attributes are used to set node `.dist` values.

    Args:
        g (nx.DiGraph): A directed acyclic graph representing a rooted tree.

    Returns:
        ete3.Tree: Reconstructed tree structure compatible with ete3 operations.

    Raises:
        ValueError: If the graph is not a DAG, not a tree, or does not have exactly one root.
    """
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

