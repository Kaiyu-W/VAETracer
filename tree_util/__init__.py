#!/usr/bin/env python

try:

    import cassiopeia as cas
    from .cas_tree import newick_to_CassiopeiaTree, get_tree_linkage
    from .newick import newick_to_networkx, ete3_to_networkx, networkx_to_ete3

except:

    import warnings
    warnings.warn('NO cassiopeia package! Cannot use this module!')