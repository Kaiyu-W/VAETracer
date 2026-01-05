import scanpy as sc
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

# need to utilize!
def save_model_to_adata(model):
    use_p = model_dict['model_p'].P < 0.8
    mdata = sc.AnnData(
        X = csr_matrix(
            model_dict['model_n'].X.numpy()
        ),
        obs = pd.DataFrame(
            {
                'N': model_dict['model_n'].N,
                'N_p': model_dict['model_p'].N,
                'celltype': adata.obs['seurat_clusters_group3'].to_numpy()
            }, 
            index=adata.obs_names
        ),
        var = pd.DataFrame(
            {
                'P_n': model_dict['model_n'].P, 
                'P': model_dict['model_p'].P[use_p]
            }
        )
    )
    mdata.obsm['Z'] = model_dict['model_n'].Z
    mdata.obsm['X_mnn_gex'] = adata.obsm['X_mnn']
    mdata.obsm['X_umap_gex'] = adata.obsm['X_umap']
    mdata.varm['Z_p'] = model_dict['model_p'].Z[use_p,:]
    mdata


    # save X_hat
    mdata.layers['Xhat_np'] = csr_matrix(model_dict['model_n'].Xhat_np)
    # mdata.layers['Xhat'] = model_dict['model_n'].Xhat

    return mdata