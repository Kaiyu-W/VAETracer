import os
import gzip
import numpy as np
import pandas as pd
from scipy.sparse import (
    save_npz, 
    load_npz, 
    issparse, 
    coo_matrix,
    spmatrix
)
from typing import Tuple, List, Union

def save_matrix(
    AD: Union[np.ndarray, spmatrix],
    RD: Union[np.ndarray, spmatrix],
    cells: Union[List[str], pd.Index, np.ndarray],
    mutations: Union[List[str], pd.Index, np.ndarray],
    save_prefix: str,
    txt_gzip: bool = True
) -> None:
    """
    Save AD and RD matrices along with associated metadata to disk in appropriate formats.

    Matrices are saved in binary format: sparse as .npz, dense as .npy.
    List-like data (cells, mutations) are saved as plain text or gzipped text files.

    Parameters:
        AD (array or spmatrix): Allele depth matrix, shape (n_mut, n_cell)
        RD (array or spmatrix): Reference depth matrix, same shape
        cells (list-like of str): Cell barcode identifiers
        mutations (list-like of str): Mutation identifiers
        save_prefix (str): Base path and filename prefix for all output files
        txt_gzip (bool): If True, save text files with gzip compression (.txt.gz); otherwise as .txt

    Outputs:
        - {save_prefix}_AD.npz/.npy
        - {save_prefix}_RD.npz/.npy
        - {save_prefix}_cells.txt(.gz)
        - {save_prefix}_mutations.txt(.gz)
    """

    # Save matrices: sparse -> .npz, dense -> .npy
    ext_ad = 'npz' if issparse(AD) else 'npy'
    ext_rd = 'npz' if issparse(RD) else 'npy'
    (save_npz if issparse(AD) else np.save)(f"{save_prefix}_AD.{ext_ad}", AD)
    (save_npz if issparse(RD) else np.save)(f"{save_prefix}_RD.{ext_rd}", RD)

    # Save list-like data
    for data, name in [(cells, 'cells'), (mutations, 'mutations')]:
        lines = [str(item) for item in data]
        filename = f"{save_prefix}_{name}.txt"
        if txt_gzip:
            filename += '.gz'
            with gzip.open(filename, 'wt', encoding='utf-8') as f:
                f.write('\n'.join(lines))
        else:
            with open(filename, 'w', encoding='utf-8') as f:
                f.write('\n'.join(lines))

def load_matrix(
    save_prefix: str
) -> Tuple[Union[np.ndarray, spmatrix], Union[np.ndarray, spmatrix], List[str], List[str]]:
    """
    Load AD, RD matrices and associated metadata from disk.

    Automatically detects and loads:
        - Matrix storage format (.npz for sparse, .npy for dense)
        - Text file compression (.txt or .txt.gz)

    Parameters:
        save_prefix (str): Prefix used during saving; used to locate files

    Returns:
        AD (array or spmatrix): Allele depth matrix
        RD (array or spmatrix): Reference depth matrix
        cells (list of str): List of cell barcodes
        mutations (list of str): List of mutation names

    Raises:
        FileNotFoundError: If expected files are not found
        OSError: If matrix files cannot be read
    """

    def load_mat(base, key):
        sparse_path = f"{base}_{key}.npz"
        dense_path = f"{base}_{key}.npy"
        try:
            return load_npz(sparse_path)
        except OSError:
            return np.load(dense_path, allow_pickle=False)

    def load_text_file(base, key):
        txt_path = f"{base}_{key}.txt"
        txt_gz_path = f"{base}_{key}.txt.gz"
        if os.path.isfile(txt_gz_path):
            with gzip.open(txt_gz_path, 'rt', encoding='utf-8') as f:
                return [line.strip() for line in f if line.strip()]
        elif os.path.isfile(txt_path):
            with open(txt_path, 'r', encoding='utf-8') as f:
                return [line.strip() for line in f if line.strip()]
        else:
            raise FileNotFoundError(f"Text file not found: {txt_path} or {txt_gz_path}")

    AD = load_mat(save_prefix, 'AD')
    RD = load_mat(save_prefix, 'RD')
    cells = load_text_file(save_prefix, 'cells')
    mutations = load_text_file(save_prefix, 'mutations')

    return AD, RD, cells, mutations

def read_parquet(
    file: Union[str, os.PathLike]
) -> pd.DataFrame:
    """
    Read and preprocess a parquet-formatted mutation data file.

    Performs:
        - Filtering out entries with AF == 3 (indicating missing values)
        - Type conversion for categorical columns
        - Extraction of cell tag and true CB (cell barcode) from CellBarcode column

    Expects 'CellBarcode' in format '{cell}-{index}_{label}' (e.g., CTCCGAAGTAACGCGA-1_normal).

    Parameters:
        file (str or PathLike): Path to input parquet file

    Returns:
        pd.DataFrame with columns:
            Original columns + 'tag', 'CB'
            'CellBarcode' and 'Mutation' as category dtype
    """

    df = (
        pd.read_parquet(file)
        .loc[lambda x: x['AF'] != 3] # 3 means missing, avoid it although 3 shall not be saved
        .assign(
            tag=lambda df: df['CellBarcode'].str.split('_', n=1).str[1], # {cell}_{label} by upstream code
            CB=lambda df: df['CellBarcode'].str.split('-', n=1).str[0] # real CellBarcode for {CB}-\d+ by cellranger
        )
        .astype({
            'CellBarcode': 'category',
            'Mutation': 'category',
            'tag': 'category',
            'CB': 'category',
        })
        .reset_index(drop=True)
    )
    return df

def filter_mut(
    df: pd.DataFrame,
    min_cells: int = 10,
    min_mutations: int = 5,
    min_ALT: int = 1,
    n_iter: int = 10,
    verbose: bool = True,
    remove_cell: bool = True,
) -> pd.DataFrame:
    """
    Iteratively filter mutations and cells based on minimum observation thresholds.

    Removes:
        - Mutations observed in fewer than `min_cells` cells
        - Cells harboring fewer than `min_mutations` mutations (optional, controlled by `remove_cell`)

    Filtering is applied iteratively until convergence or `n_iter` is reached.
    When `remove_cell=False`, cell filtering is disabled: all cells are retained regardless of mutation count.

    This is useful when some cells have only REF>0 (not captured in ALT>0), which would otherwise be excluded
    during valid_df construction even if min_mutations=0.

    Parameters:
        df (pd.DataFrame): Mutation count data with 'CellBarcode', 'Mutation', 'ALT'
        min_cells (int): Minimum number of cells a mutation must appear in to be kept
        min_mutations (int): Minimum number of mutations a cell must have to be kept (only used if remove_cell=True)
        min_ALT (int): Minimum value of mutation (ALT)
        n_iter (int): Maximum number of iterative filtering rounds
        verbose (bool): If True, print progress information at each iteration
        remove_cell (bool): If True, apply cell filtering; if False, keep all cells from input

    Returns:
        pd.DataFrame: Filtered subset of input DataFrame satisfying the criteria

    Note:
        - Only rows with ALT > 0 are considered when counting mutation/cell support
        - Function terminates early if no changes occur between iterations
        - Returns empty DataFrame if filtering removes all data
        - Setting remove_cell=False preserves cells that may only have REF>0 observations
    """

    required_cols = {'CellBarcode', 'Mutation', 'ALT'}
    assert required_cols.issubset(df.columns), f"Missing columns: {required_cols - set(df.columns)}"

    prev_cell_set = set()
    prev_mut_set = set()
    n = 0

    # Extract full cell set before any filtering
    all_cells = set(df['CellBarcode'].unique())

    if verbose:
        n_cell_raw, n_mut_raw = df.agg({'CellBarcode':'nunique', 'Mutation':'nunique'}).tolist()
        print(f'Raw: Get {n_cell_raw} cells and {n_mut_raw} mutations')

    while n < n_iter:
        # Use only ALT > 0 entries for counting support
        valid_df = df[df['ALT'] >= min_ALT]
        if valid_df.empty:
            df = valid_df
            break

        if verbose and n==0:
            n_cell_valid, n_mut_valid = valid_df.agg(
                {'CellBarcode':'nunique', 'Mutation':'nunique'}
            ).tolist()
            print(f'Valid: Get {n_cell_valid} cells and {n_mut_valid} mutations')

        # Count how many cells each mutation appears in
        mut_cell_counts = valid_df.groupby('Mutation')['CellBarcode'].nunique()
        keep_muts = set(mut_cell_counts[mut_cell_counts >= min_cells].index)

        # Decide which cells to keep
        if remove_cell:
            cell_mut_counts = valid_df.groupby('CellBarcode')['Mutation'].nunique()
            keep_cells = set(cell_mut_counts[cell_mut_counts >= min_mutations].index)
        else:
            # Keep all originally observed cells
            keep_cells = all_cells

        # Early termination if no change
        if keep_cells == prev_cell_set and keep_muts == prev_mut_set:
            if verbose:
                print(f"Iter{n}: Converged. Final: {len(keep_cells)} cells, {len(keep_muts)} mutations")
            break

        prev_cell_set = keep_cells
        prev_mut_set = keep_muts
        n += 1
        if verbose:
            print(f'Iter{n}: Get {len(keep_cells)} cells and {len(keep_muts)} mutations')

        # Apply filters and reset low-confidence ALT
        mask = (
            df['CellBarcode'].isin(keep_cells) &
            df['Mutation'].isin(keep_muts) &
            ((df['ALT'] >= min_ALT) | (df['REF'] > 0))
        )
        df = df[mask].copy()  # Only one copy here
        if df.empty:
            if verbose:
                print("All data filtered out.")
            break

        low_idx = df['ALT'] < min_ALT
        df.loc[low_idx, 'ALT'] = 0  # In-place reset of low-confidence ALT
        if 'AF' in df.columns:
            df.loc[low_idx, 'AF'] = 0

    return df.reset_index(drop=True)

def to_AD_RD(
    df: pd.DataFrame,
    sparse: Union[bool, str] = False
) -> Tuple[spmatrix, spmatrix, List[str], List[str]]:
    """
    Convert a long-format DataFrame to allele and reference depth matrices.
    
    Expects columns: 'CellBarcode', 'Mutation', 'ALT', 'REF'.
    Only entries with ALT > 0 or REF > 0 are preserved (implicit in data model).
    
    Parameters:
        df (pd.DataFrame): Long-form mutation count data
            Columns: ['CellBarcode', 'Mutation', 'ALT', 'REF']
            Each row represents a non-zero observation.
        sparse (bool or str): Storage format for output matrices.
            If True or 'csc'/'csr'/'coo': returns sparse matrices (default 'csc').
            If False: returns dense np.ndarray.
    
    Returns:
        AD (spmatrix or ndarray): Allele depth matrix, shape (n_mut, n_cell)
        RD (spmatrix or ndarray): Reference depth matrix, same shape
        cells (list): Ordered list of cell barcodes
        mutations (list): Ordered list of mutation names
    
    Matrix indexing:
        AD[i, j] = ALT count for mutations[i] in cells[j]
    """

    required_cols = {'CellBarcode', 'Mutation', 'ALT', 'REF'}
    assert required_cols.issubset(df.columns), f"Missing columns: {required_cols - set(df.columns)}"
    assert sparse in [True, False, None, 'coo', 'csc', 'csr']
    if sparse == True:
        sparse = 'csc'
    
    cells = sorted(df['CellBarcode'].unique())
    mutations = sorted(df['Mutation'].unique())
    n_cell, n_mut = len(cells), len(mutations)
    print(f'Get {n_cell} cells and {n_mut} mutations')
    
    cell_to_idx = {c: i for i, c in enumerate(cells)}
    mut_to_idx = {m: i for i, m in enumerate(mutations)}
    row_indices = df['Mutation'].map(mut_to_idx).values
    col_indices = df['CellBarcode'].map(cell_to_idx).values
    alt_vals = df['ALT'].values
    ref_vals = df['REF'].values

    mask_alt = alt_vals > 0
    AD = coo_matrix(
        (alt_vals[mask_alt], (row_indices[mask_alt], col_indices[mask_alt])),
        shape=(n_mut, n_cell),
        dtype=np.int32
    )
    if sparse:
        if sparse == 'csc':
            AD = AD.tocsc()
        elif sparse == 'csr':
            AD = AD.tocsr()
        elif sparse == 'coo':
            pass
    else:
        AD = AD.toarray()

    mask_ref = ref_vals > 0
    RD = coo_matrix(
        (ref_vals[mask_ref], (row_indices[mask_ref], col_indices[mask_ref])),
        shape=(n_mut, n_cell),
        dtype=np.int32
    )
    if sparse:
        if sparse == 'csc':
            RD = RD.tocsc()
        elif sparse == 'csr':
            RD = RD.tocsr()
        elif sparse == 'coo':
            pass
    else:
        RD = RD.toarray()

    return AD, RD, cells, mutations

def to_AD_DP(
    df: pd.DataFrame,
    sparse: Union[bool, str] = True
) -> Tuple[spmatrix, spmatrix, List[str], List[str]]:
    """
    Convert a long-format DataFrame to allele depth and total depth matrices.
    
    Equivalent to: AD, RD = to_AD_RD(...); DP = AD + RD.
    Preserves only entries where ALT > 0 or REF > 0.
    
    Parameters:
        df (pd.DataFrame): Long-form mutation count data
            Columns: ['CellBarcode', 'Mutation', 'ALT', 'REF']
        sparse (bool or str): Output matrix format.
            If True or 'csc'/'csr'/'coo': returns sparse matrices (default 'csc').
            If False: returns dense arrays.
    
    Returns:
        AD (spmatrix or ndarray): Allele depth matrix, shape (n_mut, n_cell)
        DP (spmatrix or ndarray): Total depth matrix (ALT + REF), same shape
        cells (list): Ordered list of cell barcodes
        mutations (list): Ordered list of mutation names
    
    Note:
        RD can be recovered as DP - AD for use in downstream reconstruction.
    """

    AD, RD, cells, mutations = to_AD_RD(df=df, sparse=sparse)
    DP = AD + RD
    return AD, DP, cells, mutations

def from_AD_RD(
    AD: Union[np.ndarray, spmatrix],
    RD: Union[np.ndarray, spmatrix],
    cells: List[str],
    mutations: List[str]
) -> pd.DataFrame:
    """
    Reconstruct the original long-format DataFrame from AD and RD matrices.
    
    Inverse of to_AD_RD. Recovers all entries where ALT > 0 or REF > 0.
    Assumes no (ALT=0, REF=0) rows exist in original data (consistent with model).
    
    Parameters:
        AD (array or spmatrix): Allele depth matrix, shape (n_mut, n_cell)
        RD (array or spmatrix): Reference depth matrix, same shape
        cells (list): Ordered list of cell barcodes, length n_cell
        mutations (list): Ordered list of mutation names, length n_mut
    
    Returns:
        pd.DataFrame with columns ['CellBarcode', 'Mutation', 'ALT', 'REF']
            Each row corresponds to a non-zero observation.
            Order is not guaranteed; sort if needed.
    
    Note:
        - Input matrices can be dense (np.ndarray) or any sparse format.
        - Coordinates are extracted via COO representation for efficiency.
        - Fully compatible with large-scale data (e.g., 10k x 1M).
    """

    # Normalize to COO for unified access
    ad_coo = coo_matrix(AD)
    rd_coo = coo_matrix(RD)

    # Extract non-zero coordinates and values
    i_ad, j_ad, v_ad = ad_coo.row, ad_coo.col, ad_coo.data
    i_rd, j_rd, v_rd = rd_coo.row, rd_coo.col, rd_coo.data

    # Use structured array or dict for coordinate-based merge
    data = {}

    # Add ALT entries
    for i, j, val in zip(i_ad, j_ad, v_ad):
        data[(i, j)] = [val, 0]

    # Update REF; guaranteed no (ALT=0, REF=0) so we don't drop anything
    for i, j, val in zip(i_rd, j_rd, v_rd):
        if (i, j) in data:
            data[(i, j)][1] = val
        else:
            data[(i, j)] = [0, val]

    # Unpack
    coords, vals = zip(*data.items()) if data else ([], [])
    rows, cols = zip(*coords) if coords else ([], [])
    alts, refs = zip(*vals) if vals else ([], [])

    # Map back to labels
    cells_arr = np.array(cells)
    muts_arr = np.array(mutations)

    return pd.DataFrame({
        'Mutation': muts_arr[np.array(rows)],
        'CellBarcode': cells_arr[np.array(cols)],
        'ALT': alts,
        'REF': refs
    }).reset_index(drop=True)

def from_AD_DP(
    AD: Union[np.ndarray, spmatrix],
    DP: Union[np.ndarray, spmatrix],
    cells: List[str],
    mutations: List[str]
) -> pd.DataFrame:
    """
    Reconstruct the original DataFrame from AD and total depth (DP) matrices.
    
    Inverse of to_AD_DP. Computes RD = DP - AD, then uses from_AD_RD for recovery.
    
    Parameters:
        AD (array or spmatrix): Allele depth matrix, shape (n_mut, n_cell)
        DP (array or spmatrix): Total depth matrix (i.e., AD + REF), same shape
        cells (list): Ordered list of cell barcodes
        mutations (list): Ordered list of mutation names
    
    Returns:
        pd.DataFrame with ['CellBarcode', 'Mutation', 'ALT', 'REF']
            Contains all observed non-zero entries.
            Sort by Mutation/CellBarcode for deterministic order.
    
    Note:
        - Requires DP >= AD element-wise (holds biologically).
        - Supports dense and sparse inputs.
        - Leverages existing logic in from_AD_RD for correctness and consistency.
    """

    # Compute RD = DP - AD safely
    RD = DP - AD

    # Reuse the well-tested inverse function
    return from_AD_RD(AD, RD, cells, mutations)
