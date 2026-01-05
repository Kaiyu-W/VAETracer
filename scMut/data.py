import re
import numpy as np
import matplotlib.pyplot as plt
from .typing import Tuple, Optional, Union, Dict, List, Deque
from collections import deque, defaultdict
from .log import logger

# generate silicon data
def _beta_sample(a, b, size, rng):
    samples = rng.beta(
        a=a, b=b, size=size
    )
    return samples

def _bibeta_sample(a1, b1, a2, b2, size, ratio1, rng):
    size1 = int(size * ratio1)
    size2 = size - size1
    samples1 = rng.beta(a1, b1, size=size1)
    samples2 = rng.beta(a2, b2, size=size2)
    samples = np.concatenate([samples1, samples2])
    rng.shuffle(samples)
    return samples

def sample_by_beta(a, b, a2=None, b2=None, size=10000, ratio1=0.5, seed=42, rng=None, plot=True):
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
    Generate synthetic genetic observation data with controlled randomness
    
    Parameters:
    n_cells        : Number of cells (rows in matrix)
    n_sites        : Number of genomic sites (columns in matrix)
    mutation_beta_a: Alpha parameter for Beta distribution of mutation rates
    mutation_beta_b: Beta parameter for Beta distribution of mutation rates
    mutation_beta_a2: Alpha parameter for second Beta distribution of mutation rates
    mutation_beta_b2: Beta parameter for second Beta distribution of mutation rates
    ratio2: sampled ratio of second Beta distribution of mutation rates
    generation_min : Minimum cell generation number (inclusive)
    generation_max : Maximum cell generation number (inclusive)
    noise_level    : Proportion of observations to mask as noise (0.0-1.0)
    seed           : Random seed for reproducibility (None for random state)
    
    Returns:
    mutation_rates : Array of site-specific mutation probabilities (n_sites,)
    generations    : Array of integer cell generations (n_cells,)
    observed_matrix: Noisy observation matrix (n_cells, n_sites)
    mask           : Binary mask indicating valid observations (1=valid, 0=noise)
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
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, str, np.ndarray]:
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
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, str, np.ndarray]:
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

