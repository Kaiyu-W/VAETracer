import numpy as np
import pandas as pd
from anndata import AnnData
from scipy.sparse import csr_matrix
from .typing import Dict, Optional, Union
import warnings

def save_model_to_adata(
    model_n: "MutModel",
    model_p: Optional["MutModel"] = None,
    adata_gex: Optional[AnnData] = None,
    cell_obs: Optional[pd.DataFrame] = None,
    var_names: Optional[pd.Index] = None,
    obsm_names: Optional[Dict[str, str]] = None,
    add_metrics: bool = True,
) -> AnnData:
    """
    Convert scMut model outputs into a comprehensive AnnData object.

    Integrates inferred cellular generations (N), mutation rates (P), latent spaces (Z),
    and reconstructed matrices into a single structured object compatible with Scanpy workflows.

    Designed for downstream analysis including clustering, UMAP visualization, and multi-omics integration.

    Args:
        model_n: Trained MutModel instance where train_transpose=False.
                 Used to infer N per cell.
        model_p: Optional MutModel instance where train_transpose=True.
                 Used to infer P per site. If not provided, uses model_n's P.
        adata_gex: Optional paired gene expression AnnData for metadata propagation.
                   Will copy .obs, .obsm if present.
        cell_obs: Optional DataFrame of cell-level annotations (e.g., 'celltype').
                  Must have index matching number of cells.
        var_names: Names for genomic sites (e.g., SNPs). Default: f"site_{i}".
        obsm_names: Mapping from internal keys to output .obsm layer names.
                    Example: {"Z": "latent_z", "X_mnn": "X_mnn_gex"}
        add_metrics: Whether to include training metrics in `.uns`

    Returns:
        Annotated AnnData object with the following structure:
            - .X: Observed binary mutation matrix (masked missing entries)
            - .layers:
                - 'raw': Original observed data (before masking)
                - 'Xhat_np': Reconstructed mutations using estimated N/P
                - 'posterior_call': Denoised calls based on posterior probability
            - .obs:
                - 'N': Estimated generation count per cell
                - 'N_ft': Integer-scaled version (if available)
                - Any additional metadata from adata_gex or cell_obs
            - .var:
                - 'P': Estimated mutation rate per site
                - 'P_n': Rate estimated via model_n (optional)
                - 'P_ft': Fine-tuned rate (if available)
            - .obsm:
                - 'Z': Latent representation from model_n
                - 'Z_p': Latent representation from model_p (if provided)
                - Other embeddings from adata_gex (e.g., X_umap_gex)
            - .varm:
                - 'R_expected': Expected R = 1 - (1-P)^N matrix (mean level)
            - .uns:
                - 'method': Method info
                - 'train_metrics': Training history (if enabled)

    Raises:
        ValueError: If dimension mismatches occur between models and inputs.
    
    Note:
        Requires the 'anndata' package. Install it via:
            pip install anndata

    Example:
        mdata = save_model_to_adata(
            model_n=model_n,
            model_p=model_p,
            adata_gex=adata_rna,
            cell_obs=pd.DataFrame({'batch': batch_ids}, index=adata.obs_names),
            obsm_names={"X_umap": "X_umap_gex"},
            add_metrics=True
        )
        mdata.write_h5ad("scMut_output.h5ad")
    """
    # --- Step 1: Extract core results from model(s) ---
    if model_p is not None:
        if not model_p.train_transpose:
            raise ValueError("model_p must be trained with train_transpose=True.")
        if model_p.N is None:
            raise ValueError("model_p has no inferred N. Run inference first.")
        if len(model_p.N) != model_n.input_dim:
            raise ValueError("model_p.N length does not match model_n.input_dim (n_sites).")
        P_from_p_model = model_p.P
    else:
        P_from_p_model = None

    # Ensure required estimates exist
    if model_n.N is None:
        raise ValueError("model_n has no inferred N. Run inference first.")
    if model_n.P is None:
        raise ValueError("model_n has no inferred P. Run inference first.")

    n_cells = len(model_n.N)
    n_sites = len(model_n.P)

    # Use consistent site names
    if var_names is None:
        var_names = pd.Index([f"site_{i}" for i in range(n_sites)])

    # --- Step 2: Build observation matrix (.X) ---
    # We assume X was stored as float tensor; convert back to int32 for sparsity
    try:
        X_tensor = model_n.X.cpu().detach()
        X_np = X_tensor.numpy()
    except Exception:
        X_np = model_n.X

    # Mask out missing values (do NOT impute yet)
    X_masked = np.where(X_np == model_n.miss_value, 0, X_np)
    X_sparse = csr_matrix(X_masked.astype(np.int8))

    # Create base AnnData
    adata = AnnData(
        X=X_sparse,
        obs=pd.DataFrame(index=[f"cell_{i}" for i in range(n_cells)]),
        var=pd.DataFrame(index=var_names)
    )

    # --- Step 3: Populate .obs (cell-level info) ---
    adata.obs["N"] = model_n.N
    if hasattr(model_n, 'N_ft') and model_n.N_ft is not None:
        adata.obs["N_ft"] = model_n.N_ft.astype(int)
    if hasattr(model_n, 'k_optimal'):
        adata.uns["scaling_factor_N"] = model_n.k_optimal

    # Add external annotations
    if adata_gex is not None:
        if adata_gex.n_obs == n_cells:
            common_cols = adata.obs.columns.intersection(adata_gex.obs.columns)
            if len(common_cols):
                warnings.warn(f"Overwriting existing columns: {list(common_cols)}")
            adata.obs = adata.obs.join(adata_gex.obs, how='left')
        else:
            warnings.warn("adata_gex.n_obs != model_n.N.size, skipping .obs merge.")
    elif cell_obs is not None:
        if cell_obs.shape[0] == n_cells:
            adata.obs = adata.obs.join(cell_obs, how='left', rsuffix='_input')
        else:
            raise ValueError("cell_obs must have same number of rows as model_n.N")

    # --- Step 4: Populate .var (site-level info) ---
    adata.var["P"] = model_n.P
    if P_from_p_model is not None:
        adata.var["P_from_model_p"] = P_from_p_model
    if hasattr(model_n, 'P_ft') and model_n.P_ft is not None:
        adata.var["P_ft"] = model_n.P_ft
    if hasattr(model_n, 'P_nmf'):
        adata.var["P_nmf"] = model_n.P_nmf

    # --- Step 5: Populate .layers (alternative representations) ---
    # Raw observations
    raw_data = np.where(X_np == model_n.miss_value, np.nan, X_np)
    adata.layers["raw"] = raw_data

    # Reconstructed from N/P
    if hasattr(model_n, 'Xhat_np') and model_n.Xhat_np is not None:
        Xhat_dense = model_n.Xhat_np
        adata.layers["Xhat_np"] = csr_matrix(Xhat_dense)

    # Posterior-based denoising
    if hasattr(model_n, 'posterior_X'):
        try:
            posterior_mtx = model_n.posterior_X(
                n=model_n.N,
                p=model_n.P,
                posterior_threshold=0.95
            )
            adata.layers["posterior_call"] = csr_matrix(posterior_mtx)
        except Exception as e:
            warnings.warn(f"Failed to compute posterior_call: {e}")

    # --- Step 6: Populate .obsm and .varm ---
    # Latent space from model_n
    if hasattr(model_n, 'Z') and model_n.Z is not None:
        adata.obsm["Z"] = model_n.Z

    # Latent space from model_p (if available)
    if model_p is not None and model_p.Z is not None:
        use_p = P_from_p_model < 0.8  # Filter high-noise sites
        adata.varm["Z_p"] = model_p.Z[use_p]
        adata.var["use_in_Zp"] = use_p

    # Copy other embeddings
    if adata_gex is not None and hasattr(adata_gex, 'obsm'):
        if obsm_names is None:
            obsm_names = {}
        for src_key, dst_key in obsm_names.items():
            if src_key in adata_gex.obsm:
                adata.obsm[dst_key] = adata_gex.obsm[src_key]

    # Expected R matrix (per-site expectation given N and P)
    R_expected = model_n.compute_R(model_n.N, model_n.P, sample=False)
    adata.varm["R_expected"] = R_expected.T  # shape: (n_sites, n_cells)

    # --- Step 7: Metadata and provenance ---
    adata.uns["method"] = {
        "project": "scMut",
        "version": "1.0",
        "description": "Single-cell somatic mutation inference using VAE+NMF"
    }
    if add_metrics and hasattr(model_n, 'train_metrics'):
        adata.uns["train_metrics"] = model_n.train_metrics

    # Final check
    assert adata.n_obs == n_cells, f"Obs mismatch: got {adata.n_obs}, expected {n_cells}"
    assert adata.n_vars == n_sites, f"Var mismatch: got {adata.n_vars}, expected {n_sites}"

    return adata

