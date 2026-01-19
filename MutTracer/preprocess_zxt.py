import re
import pickle
import scanpy as sc
from collections import defaultdict

import torch
import scvi


def process_h5ad_to_zxt(h5ad_path, output_path='adc_zx_dict.pt'):
    """
    Process a single-cell RNA-seq AnnData (.h5ad) file and extract transcriptional latent features (zxt)
    using a LinearSCVI model.

    Steps:
    1. Load the .h5ad file using Scanpy.
    2. Preprocess the data:
       - Copy raw counts to adata.layers["counts"].
       - Normalize total counts per cell to a target sum.
       - Log-transform the normalized data.
       - Save the raw preprocessed data to adata.raw for future reference.
    3. Select highly variable genes (HVGs) using Seurat v3 method.
    4. Prepare data for scVI:
       - Register the AnnData object with LinearSCVI using the counts layer.
    5. Train LinearSCVI model:
       - Initialize model with n_latent=20 dimensions.
       - Train for up to 250 epochs with learning rate 0.005.
       - Save the trained model to a pickle file for later use.
    6. Extract latent representation (zxt):
       - Obtain latent embedding for each cell via model.get_latent_representation().
    7. Organize latent features by group:
       - Define a helper function `extract_group_key` that assigns a group label
         based on the cell name suffix (e.g., '_ADC', '_Mix', '_SCC').
       - Collect latent vectors into a dictionary keyed by group.
    8. Convert lists of latent vectors to PyTorch tensors for each group.
    9. Save the zxt_dict to a .pt file.
    10. Print a summary of saved tensors and their shapes.

    Args:
        h5ad_path (str): Path to input .h5ad file containing scRNA-seq data.
        output_path (str): Path to save the zxt feature dictionary (.pt file).

    Returns:
        model (LinearSCVI): Trained LinearSCVI model.
    """
    adata = sc.read_h5ad(h5ad_path)
    
    adata.layers["counts"] = adata.X.copy()
    sc.pp.normalize_total(adata, target_sum=10e4)
    sc.pp.log1p(adata)
    adata.raw = adata
    
    sc.pp.highly_variable_genes(
        adata, 
        flavor="seurat_v3", 
        layer="counts", 
        n_top_genes=3000, 
        subset=True
    )

    scvi.model.LinearSCVI.setup_anndata(adata, layer="counts")
    model = scvi.model.LinearSCVI(adata, n_latent=20)
    model.train(max_epochs=250, plan_kwargs={"lr": 5e-3})

    print(f"Saved model to linear_scvi")
    with open("scvi_model.pkl", "wb") as f:
        pickle.dump(model, f)

    Z_hat = model.get_latent_representation()
    
    def extract_group_key(index):
        match = re.search(r'_(ADC|Mix|SCC)$', index)
        if match:
            return {'ADC': 1, 'Mix': 2, 'SCC': 3}[match.group(1)]
        return None

    zxt_dict = defaultdict(list)
    for i, idx in enumerate(adata.obs_names):
        group_key = extract_group_key(idx)
        if group_key is not None:
            zxt_dict[group_key].append(Z_hat[i])
    
    zxt_dict = {k: torch.stack(v) for k, v in zxt_dict.items()}

    torch.save(zxt_dict, output_path)
    print(f"Saved zxt features to {output_path}")
    print({k: v.shape for k, v in zxt_dict.items()})

    return model  
