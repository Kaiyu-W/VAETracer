import re
import numpy as np
import matplotlib.pyplot as plt
from .typing import Tuple, Optional, Union, Dict, List, Deque
from collections import deque, defaultdict
from .log import logger

# generate silicon data
def _beta_sample(
    a: float,
    b: float,
    size: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Sample from a Beta distribution.

    Internal helper for generating mutation rate priors.

    Args:
        a: Alpha parameter.
        b: Beta parameter.
        size: Number of samples.
        rng: Random number generator.

    Returns:
        Array of beta-distributed values in (0,1).
    """

    samples = rng.beta(
        a=a, b=b, size=size
    )
    return samples

def _bibeta_sample(
    a1: float,
    b1: float,
    a2: float,
    b2: float,
    size: int,
    ratio1: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Sample from a mixture of two Beta distributions.

    Internal helper for bimodal mutation rate simulation.

    Args:
        a1: Alpha of first component.
        b1: Beta of first component.
        a2: Alpha of second component.
        b2: Beta of second component.
        size: Total number of samples.
        ratio1: Proportion from first component.
        rng: Random number generator.

    Returns:
        Shuffled array combining both components.
    """
    
    size1 = int(size * ratio1)
    size2 = size - size1
    samples1 = rng.beta(a1, b1, size=size1)
    samples2 = rng.beta(a2, b2, size=size2)
    samples = np.concatenate([samples1, samples2])
    rng.shuffle(samples)
    return samples

def sample_by_beta(
    a: float,
    b: float,
    a2: Optional[float] = None,
    b2: Optional[float] = None,
    size: int = 10000,
    ratio1: float = 0.5,
    seed: Optional[int] = 42,
    rng: Optional[np.random.Generator] = None,
    plot: bool = True,
) -> np.ndarray:
    """
    Sample mutation rate-like values from one or two Beta distributions.

    This function draws samples from either a single Beta(a,b) distribution or a mixture of 
    two Beta distributions with specified mixing ratio. It is primarily used to simulate 
    site-specific mutation probabilities in synthetic data generation.

    The output can be used as input for `mutation_rates` in downstream simulation functions.

    Args:
        a (float): Alpha parameter of the first Beta distribution.
        b (float): Beta parameter of the first Beta distribution.
        a2 (float, optional): Alpha parameter of the second Beta distribution. 
                              If None, only one Beta distribution is used.
        b2 (float, optional): Beta parameter of the second Beta distribution. 
                              Must also be None if a2 is None.
        size (int): Number of samples to draw. Default: 10000.
        ratio1 (float): Proportion of samples drawn from the first component (Beta(a,b)). 
                        The rest (1-ratio1) come from Beta(a2,b2) if provided. Default: 0.5.
        seed (int, optional): Random seed for reproducibility. Used only if `rng` is None. 
                              Default: 42.
        rng (np.random.Generator, optional): Custom random number generator. 
                                             If not provided, creates one from `seed`.
        plot (bool): If True, plots a histogram of the sampled values using matplotlib.

    Returns:
        np.ndarray: Array of shape (size,) containing sampled values in [0,1], 
                    representing simulated mutation probabilities per site.

    Example:
        >>> rates = sample_by_beta(0.1, 0.5, size=500)
        >>> mixed_rates = sample_by_beta(0.1, 0.5, a2=0.3, b2=0.7, ratio1=0.7, size=1000)

    Note:
        This function is designed for generating biologically plausible sparse mutation profiles,
        where most sites have low mutation probability (spike near 0), and a few are more likely.
    """
    if rng is None:
        rng = np.random.default_rng(seed)

    if a2 is None and b2 is None:
        label = f"Beta({a}, {b})"
        samples = _beta_sample(a, b, size, rng)
    else:
        assert ratio1>=0 and ratio1<=1
        label = f"{ratio1:.2f} Beta1({a}, {b}) + {1-ratio1:.2f} Beta2({a2}, {b2})"
        samples = _bibeta_sample(a, b, a2, b2, size, ratio1, rng)

    if plot:
        plt.hist(samples, bins=100, density=True, label=label)
        plt.title(label)

    return samples


def simulate_data(
    n_cells: int = 1000,
    n_sites: int = 500,
    mutation_beta_a: float = 0.1,
    mutation_beta_b: float = 0.5,
    mutation_beta_a2: Optional[float] = None,
    mutation_beta_b2: Optional[float] = None,
    ratio2: float = 0.5,
    generation_min: int = 0,
    generation_max: int = 100,
    noise_level: float = 0.0,
    seed: Optional[int] = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate synthetic scMut observation data under a cumulative mutation model without lineage structure.

    Creates a dataset where each cell has an integer "generation" value (N), and each genomic site 
    has a base mutation rate (P). The observed mutation status follows the formula:

        R = 1 - (1 - P)^N

    This simulates irreversible binary mutations accumulated over divisions.

    No tree structure is enforced; generations are assigned independently.

    Args:
        n_cells (int): Number of cells (rows in output matrix). Default: 1000.
        n_sites (int): Number of genomic sites / features (columns). Default: 500.
        mutation_beta_a (float): Alpha parameter for Beta distribution of site-level mutation rates.
        mutation_beta_b (float): Beta parameter for Beta distribution of site-level mutation rates.
        mutation_beta_a2 (float, optional): Second component alpha for bimodal mutation rate prior.
        mutation_beta_b2 (float, optional): Second component beta for bimodal mutation rate prior.
        ratio2 (float): Fraction of sites sampled from the second Beta component (if used).
        generation_min (int): Minimum cellular generation (inclusive). Default: 0.
        generation_max (int): Maximum cellular generation (inclusive). Default: 100.
        noise_level (float): Fraction of entries to mask as missing/noisy observations (set to 0 in mask).
                             Should be in [0, 1]. Default: 0.0 (no noise).
        seed (int, optional): Random seed for reproducibility. Default: 42.

    Returns:
        tuple:
            - mutation_rates (np.ndarray): True underlying mutation probabilities per site, shape (n_sites,).
            - generations (np.ndarray): Simulated generation count for each cell, shape (n_cells,), dtype=int8.
            - observed_matrix (np.ndarray): Observed mutation matrix, shape (n_cells, n_sites), values in [0,1].
              Computed as 1 - (1 - P)^N.
            - mask (np.ndarray): Binary mask indicating valid observations, shape (n_cells, n_sites), dtype=int8.
              1 = observed, 0 = masked (noise).

    Note:
        All random streams are isolated using separate generators seeded identically for reproducibility.
        Use this function when you want independent cells without phylogenetic constraints.
    """

    # Create isolated random generator
    # rng = np.random.default_rng(seed)
    rng_mutation = np.random.default_rng(seed)
    rng_generations = np.random.default_rng(seed)
    rng_noise = np.random.default_rng(seed)
    
    # 1. Generate mutation rates from Beta distribution
    mutation_rates = sample_by_beta(
        a=mutation_beta_a,
        b=mutation_beta_b,
        a2=mutation_beta_a2,
        b2=mutation_beta_b2,
        size=n_sites,
        ratio1=1-ratio2,
        rng=rng_mutation,
        plot=False
    ).astype(np.float16)
    
    # 2. Generate integer generations with inclusive upper bound
    generations = rng_generations.integers(
        low=generation_min,
        high=generation_max + 1,  # high is exclusive
        size=n_cells,
        dtype=np.int8
    )
    
    # 3. Calculate observation matrix
    eps = 1e-20
    safe_rates = np.clip(mutation_rates, eps, 1-eps)
    observed_matrix = 1 - np.power(
        1 - safe_rates,
        generations[:, np.newaxis]
    ).astype(np.float16)
    
    # 4. Generate noise mask using binomial distribution
    mask = rng_noise.binomial(
        n=1,
        p=noise_level,
        size=observed_matrix.shape
    ).astype(np.int8)
    
    return mutation_rates, generations, observed_matrix, mask


def simulate_lineage_data(
    n_cells: int = 1000,
    n_sites: int = 500,
    mutation_beta_a: float = 0.1,
    mutation_beta_b: float = 0.5,
    mutation_beta_a2: Optional[float] = None,
    mutation_beta_b2: Optional[float] = None,
    ratio2: float = 0.5,
    generation_max: int = 100,
    survival_rate: float = 0.8,
    noise_level: float = 0.0,
    seed: Optional[int] = 42,
    keep_whole_lineage: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Union[str, Tuple[str, Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]], np.ndarray]:
    """
    Generate synthetic single-cell mutation data with realistic lineage tree structure.

    Simulates a branching process starting from a root cell, where each division may introduce new mutations.
    A subsample of cells is then selected to mimic real scDNA-seq experiments.

    Enforces true ancestral relationships: mutations are irreversible and accumulate along branches.

    Args:
        n_cells (int): Target number of sampled cells to return. May be reduced if fewer exist.
        n_sites (int): Number of genomic sites (binary features).
        mutation_beta_a (float): Alpha parameter for Beta-distributed site mutation probabilities.
        mutation_beta_b (float): Beta parameter for same.
        mutation_beta_a2 (float, optional): Second component alpha for bimodal mutation prior.
        mutation_beta_b2 (float, optional): Second component beta.
        ratio2 (float): Proportion of sites from second component (if used).
        generation_max (int): Maximum allowed cell generation (tree depth).
        survival_rate (float): Probability that a cell survives beyond early generations (>3) to divide.
                               Controls tree bushiness.
        noise_level (float): Fraction of observations to corrupt via masking (0–1).
        seed (int, optional): Seed for all random number generators. Default: 42.
        keep_whole_lineage (bool): If True, returns additional intermediate node information as part of newick_result.

    Returns:
        tuple:
            - mutation_rates (np.ndarray): Base mutation probabilities per site, shape (n_sites,).
            - generations (np.ndarray): Generation level of each sampled cell, shape (n_cells,), dtype=int8.
            - observed_matrix (np.ndarray): Mutation observation matrix, shape (n_cells, n_sites), values ∈ {0,1}.
            - mask (np.ndarray): Observation validity mask, shape (n_cells, n_sites), dtype=int8.
            - newick_result (str or tuple):
                If keep_whole_lineage=False: Newick string of the pruned tree with virtual nodes.
                If True: A tuple of (newick_str, gens_v, mat_v, ids_v), where:
                    - newick_str: same as above
                    - gens_v: generation levels of virtual nodes
                    - mat_v: observed mutations of virtual nodes
                    - ids_v: their identifiers (e.g., ['V1', 'V2', ...])
                  Any element may be None if no virtual nodes exist.
            - cell_ids (np.ndarray): Cell identifiers (e.g., ['C0', 'C1', ...]), shape (n_cells,).

    Note:
        This function generates a full lineage tree and then samples cells from it, preserving biological realism.
        Virtual nodes ('V*') represent unobserved ancestors in the reconstructed tree.
    """

    # rng = np.random.default_rng(seed)
    rng_mutation = np.random.default_rng(seed)
    rng_mutation2 = np.random.default_rng(seed)
    rng_survival = np.random.default_rng(seed)
    rng_generations = np.random.default_rng(seed)
    rng_noise = np.random.default_rng(seed)

    # Step 1: Generate mutation rates using a beta distribution
    mutation_rates = sample_by_beta(
        a=mutation_beta_a,
        b=mutation_beta_b,
        a2=mutation_beta_a2,
        b2=mutation_beta_b2,
        size=n_sites,
        ratio1=1-ratio2,
        rng=rng_mutation,
        plot=False
    ).astype(np.float16)

    # Step 2: Initialize data structures to track lineage relationships
    child_dict: Dict[int, List[int]] = {}  # parent_id -> [child_ids]
    cell_data = {}  # cell_id -> (generation, mutations)
    queue: Deque[Tuple[int, np.ndarray, int]] = deque()  # Queue for processing cells
    sampled_ids = []  # List of sampled cell IDs
    next_id = 0  # Counter for assigning unique cell IDs

    # Initialize root cell with zero mutations
    root_mutations = np.zeros(n_sites, dtype=np.int8)
    cell_data[next_id] = (0, root_mutations)  # Root cell at generation 0
    queue.append((0, root_mutations, next_id))  # Add root cell to the queue
    next_id += 1  # Increment ID counter

    # Step 3: Dynamically generate lineage tree with survival rate control
    while queue:
        generation, mutations, parent_id = queue.popleft()
        if generation >= generation_max:
            continue  # Stop if maximum generation is reached
        
        # Survival and division logic
        if generation < 3 or rng_survival.random() < survival_rate:
            # Generate new mutations for offspring
            new_mut = rng_mutation2.binomial(n=1, p=mutation_rates).astype(np.int8)
            offspring_mut = np.maximum(mutations, new_mut)  # Combine parent and new mutations
            
            # Create two offspring cells
            for _ in range(2):
                child_id = next_id
                next_id += 1
                
                offspring_mutations = offspring_mut.copy()
                cell_data[child_id] = (generation + 1, offspring_mutations)
                child_dict.setdefault(parent_id, []).append(child_id)
                queue.append((generation + 1, offspring_mutations, child_id))

    # Step 4: Sample cells from the generated lineage tree
    all_cell_ids = list(cell_data.keys())
    if len(all_cell_ids) < n_cells:
        n_cells = len(all_cell_ids)
        logger.warning(f'Only got {n_cells} samples!')
    sampled_ids = rng_generations.choice(all_cell_ids, size=n_cells, replace=False).tolist()
    sorted_sampled_ids = sorted(
        sampled_ids,
        key=lambda x: (cell_data[x][0], x)  # Sort by generation and ID
    )

    # Step 5: Generate tree newick
    # Build parent mapping for all cells
    parent_dict = {}
    for pid, children in child_dict.items():
        for child in children:
            parent_dict[child] = pid

    pruned_child_dict = {}
    original_root = 0  # Root node ID is always 0
    for cid in [x for x in sampled_ids if x != original_root]:
        current = cid
        while True:
            # Find parent of current
            parent = parent_dict[current]
            pruned_child_dict.setdefault(parent, set()).add(current)
            current = parent
            if current == original_root or current in sampled_ids:
                break

    # Recursive function to build Newick string representation of the tree
    def build_newick(node_id: int, branch_length: int) -> str:
        children = pruned_child_dict.get(node_id, [])
        child_newicks = []
        for child in children:
            child_generation = cell_data[child][0]
            node_generation = cell_data[node_id][0]
            child_branch_length = child_generation - node_generation
            
            child_newick = build_newick(child, child_branch_length)
            if child_newick:
                child_newicks.append(child_newick)
        
        # If the current node is a sampled node, add it as a virtual leaf
        node_label = f"C{node_id}:{branch_length}"
        node_label_virtual = f"V{node_id}:{branch_length}"
        if node_id in sampled_ids:  # Otherwise, it's the root
            child_newicks.append(node_label)
        
        if len(child_newicks) > 1:
            return f"({','.join(child_newicks)}){node_label_virtual}"
        elif len(child_newicks) == 1:
            split_list = child_newicks[0].split(':')
            rest_list, branch_length_raw = split_list[:-1], int(split_list[-1])
            child_newick_new = ':'.join(
                rest_list + [str(branch_length_raw + branch_length)]
            )
            return child_newick_new
        else:
            return ""

    # Determine the root node and build the Newick string
    newick_str = build_newick(original_root, 0) + ";" # remove ':0'
    if original_root in sampled_ids:
        newick_str = f"(C{original_root}:1,{newick_str.rstrip(';')}:1);"

    if keep_whole_lineage:
        v_numbers = sorted([int(_num) for _num in re.findall(r'V(\d+)', newick_str)])

        if len(v_numbers) > 0:
            sorted_sampled_ids_v = np.array([f'V{x}' for x in v_numbers])
            generations_v = np.array([cell_data[i][0] for i in v_numbers], dtype=np.int8)
            observed_matrix_v = np.vstack([cell_data[i][1] for i in v_numbers]).astype(np.float16)
            newick_result = (
                newick_str, 
                generations_v,
                observed_matrix_v,
                sorted_sampled_ids_v
            )
        else:
            newick_result = (newick_str, None, None, None)
    else:
        newick_result = newick_str

    # Step 6: Prepare other output arrays
    generations = np.array([cell_data[i][0] for i in sorted_sampled_ids], dtype=np.int8)
    observed_matrix = np.vstack([cell_data[i][1] for i in sorted_sampled_ids]).astype(np.float16)
    mask = rng_noise.binomial(n=1, p=noise_level, size=observed_matrix.shape).astype(np.int8)
    cell_ids = np.array([f'C{x}' for x in sorted_sampled_ids])

    return (
        mutation_rates, 
        generations, 
        observed_matrix, 
        mask, 
        newick_result, 
        cell_ids
    )


def simulate_lineage_data_segment(
    n_cells: int = 1000,
    n_sites: int = 500,
    mutation_beta_a: float = 0.1,
    mutation_beta_b: float = 0.5,
    mutation_beta_a2: Optional[float] = None,
    mutation_beta_b2: Optional[float] = None,
    ratio2: float = 0.5,
    generation_max: int = 100,
    survival_rate: float = 0.8,
    noise_level: float = 0.0,
    seed: Optional[int] = 42,
    keep_whole_lineage: bool = False,
    segment_dict: Dict[Tuple[int, int], Union[float, int]] = {
        (5, 10): 0.1,
        (20,25): 0.3,
        (40,45): 0.6
    }
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Union[str, Tuple[str, Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]], np.ndarray]:
    """
    Generate synthetic lineage data with controlled sampling across specific generation intervals.

    Extends `simulate_lineage_data` by allowing precise control over how many cells are sampled from 
    user-defined generation ranges (segments). Useful for simulating staged differentiation processes.

    Args:
        n_cells (int): Total number of cells to sample.
        n_sites (int): Number of genomic sites.
        mutation_beta_a (float): Alpha parameter for mutation rate Beta distribution.
        mutation_beta_b (float): Beta parameter.
        mutation_beta_a2 (float, optional): Second component alpha (bimodal prior).
        mutation_beta_b2 (float, optional): Second component beta.
        ratio2 (float): Fraction of sites from second mutation component.
        generation_max (int): Maximum allowed cell generation.
        survival_rate (float): Division survival probability after generation 3.
        noise_level (float): Fraction of entries to mask as noise.
        seed (int, optional): Random seed. Default: 42.
        keep_whole_lineage (bool): Whether to return internal node data in newick_result.
        segment_dict (dict): Maps generation intervals `(start, end)` to proportions or absolute counts.
                             Example: `{(0,10): 0.3, (20,30): 0.7}` means 30% from early, 70% from late.
                             Values can be floats ∈ (0,1] (fractions) or integers ≥1 (absolute numbers).
                             Sum must equal 1.0 or `n_cells`.

    Returns:
        tuple:
            - mutation_rates (np.ndarray): Site-level mutation probabilities, shape (n_sites,).
            - generations (np.ndarray): Sampled cell generations, shape (n_cells,), dtype=int8.
            - observed_matrix (np.ndarray): Observed mutation states, shape (n_cells, n_sites).
            - mask (np.ndarray): Valid observation mask, shape (n_cells, n_sites).
            - newick_result (str or tuple): 
                If keep_whole_lineage=False: A Newick-format string representing the pruned lineage tree.
                If True: A tuple of (newick_str, gens_v, mat_v, ids_v), where:
                    - newick_str: same as above
                    - gens_v: generation levels of virtual nodes
                    - mat_v: observed mutations of virtual nodes
                    - ids_v: their identifiers (e.g., ['V1', 'V2', ...])
                  Any element may be None if no virtual nodes exist.
            - cell_ids (np.ndarray): Cell labels (e.g., ['C12', 'C45', ...]), shape (n_cells,).

    Raises:
        AssertionError: If any segment exceeds `generation_max`, or sum of ratios ≠ 1/n_cells.

    Example:
        To simulate 80% of cells coming from late generations:
        >>> seg_dict = {(0, 10): 0.2, (20, 40): 0.8}
        >>> simulate_lineage_data_segment(n_cells=500, segment_dict=seg_dict)

    Note:
        This function enables simulation of non-uniform developmental trajectories, such as burst-like expansion.
    """

    assert all([
        y<=generation_max 
        for x in segment_dict.keys() 
        for y in x
    ]) and sum(segment_dict.values()) in [1, n_cells]

    # rng = np.random.default_rng(seed)
    rng_mutation = np.random.default_rng(seed)
    rng_mutation2 = np.random.default_rng(seed)
    rng_survival = np.random.default_rng(seed)
    rng_generations = np.random.default_rng(seed)
    rng_noise = np.random.default_rng(seed)

    # Step 1: Generate mutation rates using a beta distribution
    mutation_rates = sample_by_beta(
        a=mutation_beta_a,
        b=mutation_beta_b,
        a2=mutation_beta_a2,
        b2=mutation_beta_b2,
        size=n_sites,
        ratio1=1-ratio2,
        rng=rng_mutation,
        plot=False
    ).astype(np.float16)

    # Step 2: Initialize data structures to track lineage relationships
    child_dict: Dict[int, List[int]] = {}  # parent_id -> [child_ids]
    cell_data = {}  # cell_id -> (generation, mutations)
    queue: Deque[Tuple[int, np.ndarray, int]] = deque()  # Queue for processing cells
    sampled_ids = []  # List of sampled cell IDs
    next_id = 0  # Counter for assigning unique cell IDs

    # Initialize root cell with zero mutations
    root_mutations = np.zeros(n_sites, dtype=np.int8)
    cell_data[next_id] = (0, root_mutations)  # Root cell at generation 0
    queue.append((0, root_mutations, next_id))  # Add root cell to the queue
    next_id += 1  # Increment ID counter

    # Step 3: Dynamically generate lineage tree with survival rate control
    while queue:
        generation, mutations, parent_id = queue.popleft()
        if generation >= generation_max:
            continue  # Stop if maximum generation is reached
        
        # Survival and division logic
        if generation < 3 or rng_survival.random() < survival_rate:
            # Generate new mutations for offspring
            new_mut = rng_mutation2.binomial(n=1, p=mutation_rates).astype(np.int8)
            offspring_mut = np.maximum(mutations, new_mut)  # Combine parent and new mutations
            
            # Create two offspring cells
            for _ in range(2):
                child_id = next_id
                next_id += 1
                
                offspring_mutations = offspring_mut.copy()
                cell_data[child_id] = (generation + 1, offspring_mutations)
                child_dict.setdefault(parent_id, []).append(child_id)
                queue.append((generation + 1, offspring_mutations, child_id))

    # Step 4: Sample cells from the generated lineage tree
    sampled_ids = []
    for _range, _ratio in segment_dict.items():
        _n_cells = int(np.around(n_cells * _ratio) if _ratio <= 1 else _ratio)
        _range_in = lambda x: x>=_range[0] and x<=_range[1]
        _cell_ids = [ x for x in cell_data.keys() if _range_in(cell_data[x][0]) ]
        if len(_cell_ids) < _n_cells:
            _n_cells = len(_cell_ids)
            logger.warning(f'Only got {_n_cells} samples for {_range}!')
        sampled_ids.extend(
            rng_generations.choice(_cell_ids, size=_n_cells, replace=False).tolist()
        )
    if len(sampled_ids) < n_cells:
        logger.warning(f'Only got {len(sampled_ids)} samples in total!')

    sorted_sampled_ids = sorted(
        sampled_ids,
        key=lambda x: (cell_data[x][0], x)  # Sort by generation and ID
    )

    # Step 5: Generate tree newick
    # Build parent mapping for all cells
    parent_dict = {}
    for pid, children in child_dict.items():
        for child in children:
            parent_dict[child] = pid

    pruned_child_dict = {}
    original_root = 0  # Root node ID is always 0
    for cid in [x for x in sampled_ids if x != original_root]:
        current = cid
        while True:
            # Find parent of current
            parent = parent_dict[current]
            pruned_child_dict.setdefault(parent, set()).add(current)
            current = parent
            if current == original_root or current in sampled_ids:
                break

    # Recursive function to build Newick string representation of the tree
    def build_newick(node_id: int, branch_length: int) -> str:
        children = pruned_child_dict.get(node_id, [])
        child_newicks = []
        for child in children:
            child_generation = cell_data[child][0]
            node_generation = cell_data[node_id][0]
            child_branch_length = child_generation - node_generation
            
            child_newick = build_newick(child, child_branch_length)
            if child_newick:
                child_newicks.append(child_newick)
        
        # If the current node is a sampled node, add it as a virtual leaf
        node_label = f"C{node_id}:{branch_length}"
        node_label_virtual = f"V{node_id}:{branch_length}"
        if node_id in sampled_ids:  # Otherwise, it's the root
            child_newicks.append(node_label)
        
        if len(child_newicks) > 1:
            return f"({','.join(child_newicks)}){node_label_virtual}"
        elif len(child_newicks) == 1:
            split_list = child_newicks[0].split(':')
            rest_list, branch_length_raw = split_list[:-1], int(split_list[-1])
            child_newick_new = ':'.join(
                rest_list + [str(branch_length_raw + branch_length)]
            )
            return child_newick_new
        else:
            return ""

    # Determine the root node and build the Newick string
    newick_str = build_newick(original_root, 0) + ";" # remove ':0'
    if original_root in sampled_ids:
        newick_str = f"(C{original_root}:1,{newick_str.rstrip(';')}:1);"

    if keep_whole_lineage:
        v_numbers = sorted([int(_num) for _num in re.findall(r'V(\d+)', newick_str)])

        if len(v_numbers) > 0:
            sorted_sampled_ids_v = np.array([f'V{x}' for x in v_numbers])
            generations_v = np.array([cell_data[i][0] for i in v_numbers], dtype=np.int8)
            observed_matrix_v = np.vstack([cell_data[i][1] for i in v_numbers]).astype(np.float16)
            newick_result = (
                newick_str, 
                generations_v,
                observed_matrix_v,
                sorted_sampled_ids_v
            )
        else:
            newick_result = (newick_str, None, None, None)
    else:
        newick_result = newick_str

    # Step 6: Prepare other output arrays
    generations = np.array([cell_data[i][0] for i in sorted_sampled_ids], dtype=np.int8)
    observed_matrix = np.vstack([cell_data[i][1] for i in sorted_sampled_ids]).astype(np.float16)
    mask = rng_noise.binomial(n=1, p=noise_level, size=observed_matrix.shape).astype(np.int8)
    cell_ids = np.array([f'C{x}' for x in sorted_sampled_ids])

    return (
        mutation_rates, 
        generations, 
        observed_matrix, 
        mask, 
        newick_result, 
        cell_ids
    )

