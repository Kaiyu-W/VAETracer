import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scanpy as sc
from sklearn.preprocessing import QuantileTransformer
from sklearn.metrics import r2_score

import torch
from scvi.model import LinearSCVI


def load_scvi_model(model_path, adata=None):
    """
    Load a pretrained scVI-based linear model from disk.

    This function restores a serialized LinearSCVI model and optionally
    attaches an AnnData object for downstream inference or embedding usage.
    """
    return LinearSCVI.load(model_path, adata=adata)


def align_zxt_features(pred_dict, ref_dict, method='quantile'):
    """
    Align predicted auxiliary latent features to reference time-point distributions.

    For time points with available references, direct distribution alignment
    is applied. For unseen time points, features are aligned using temporally
    weighted interpolation from neighboring reference distributions.
    """

    aligned_dict = {}
    all_ref_times = sorted(ref_dict.keys())
    
    for t_pred, pred_data in pred_dict.items():

        if isinstance(pred_data, dict) and 'zxt_pred' in pred_data:
            pred = pred_data['zxt_pred']
        else:
            pred = pred_data  
            
        if torch.is_tensor(pred):
            pred = pred.detach().cpu().numpy()
            
        if t_pred in ref_dict:
            ref = ref_dict[t_pred]
            if torch.is_tensor(ref):
                ref = ref.detach().cpu().numpy()
            aligned_dict[t_pred] = robust_align(pred, ref, method)

        else:
            if t_pred < min(all_ref_times):
                ref_times = all_ref_times[:2]
            elif t_pred > max(all_ref_times):
                ref_times = all_ref_times[-2:]
            else:
                left = max([t for t in all_ref_times if t < t_pred], default=None)
                right = min([t for t in all_ref_times if t > t_pred], default=None)
                ref_times = [t for t in [left, right] if t is not None]

            available_refs = {t: ref_dict[t] for t in ref_times if t in ref_dict}
            
            aligned_dict[t_pred] = temporal_aware_align(
                pred, t_pred, available_refs, ref_times, method
            )
    
    return aligned_dict


def robust_align(pred, ref, method='quantile'):
    """
    Perform robust distribution alignment between predicted and reference features.

    This function supports multiple normalization strategies, including
    quantile normalization, z-score alignment, and robust z-score alignment,
    to match predicted feature distributions to reference statistics.
    """
    if torch.is_tensor(pred):
        pred = pred.detach().cpu().numpy()
    if torch.is_tensor(ref):
        ref = ref.detach().cpu().numpy()
    
    if method == 'zscore':
        pred_mean = np.nanmean(pred, axis=0)
        pred_std = np.nanstd(pred, axis=0)
        ref_mean = np.nanmean(ref, axis=0)
        ref_std = np.nanstd(ref, axis=0)
        aligned = (pred - pred_mean) / (pred_std + 1e-6) * ref_std + ref_mean
        return np.clip(aligned, -5, 5)
    
    elif method == 'quantile':
        qt = QuantileTransformer(n_quantiles=1000, output_distribution='normal')
        qt.fit(ref)
        return qt.transform(pred)
    
    elif method == 'robust_zscore':
        pred_median = np.median(pred, axis=0)
        pred_iqr = np.percentile(pred, 75, axis=0) - np.percentile(pred, 25, axis=0)
        ref_median = np.median(ref, axis=0)
        ref_iqr = np.percentile(ref, 75, axis=0) - np.percentile(ref, 25, axis=0)
        aligned = (pred - pred_median) / (pred_iqr + 1e-6) * ref_iqr + ref_median
        return np.clip(aligned, -5, 5)


def temporal_aware_align(pred, t_pred, ref_dict, ref_times=[1,2,3], method='quantile'):
    """
    Align predicted features using temporally weighted reference distributions.

    Predicted features are aligned by aggregating distribution-matched outputs
    from nearby reference time points, with weights determined by temporal
    proximity.
    """
    total_weight = 0
    aligned = np.zeros_like(pred)
    
    for t_ref in ref_times:
        if t_ref not in ref_dict:
            continue
            
        weight = np.exp(-abs(t_pred - t_ref))
        ref_feat = ref_dict[t_ref]
        
        if pred.shape[1] == ref_feat.shape[1]:
            aligned += weight * robust_align(pred, ref_feat, method=method)
            total_weight += weight
    
    return aligned / (total_weight + 1e-6)


def plot_combined_umap(adata, output_path):
    """
    Visualize real and predicted cells in a shared UMAP embedding.

    This function generates a combined UMAP plot colored by time points and
    annotated by data source (real versus predicted), enabling intuitive
    inspection of temporal trajectories and model extrapolations.
    """
    plt.figure(figsize=(8, 6))

    time_categories = adata.obs['Time'].astype('category')
    colors = sns.color_palette("husl", n_colors=len(time_categories.cat.categories))
    sc.pl.umap(
        adata,
        color='Time',
        palette=colors,
        show=False,
        title='Combined Trajectory (Real + Predicted)',
        frameon=False,
        size=20 if adata.n_obs < 5000 else 10,
        alpha=0.7,
        legend_loc='right margin'
    )
    
    ax = plt.gca()
    real_idx = np.where(adata.obs['Source'] == 'Real')[0]
    pred_idx = np.where(adata.obs['Source'] == 'Predicted')[0]
    
    umap_coords = adata.obsm['X_umap']
    
    if len(pred_idx) > 0:
        pred_scatter = ax.scatter(
            umap_coords[pred_idx, 0],
            umap_coords[pred_idx, 1],
            c=[colors[list(time_categories.cat.categories).index(t)] 
               for t in adata.obs.iloc[pred_idx]['Time']],
            s=20 if adata.n_obs < 5000 else 10,
            alpha=0.7,
            edgecolor='black',
            linewidth=0.5,
            marker='o'
        )
    
    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w', label='Real',
                  markerfacecolor='gray', markersize=8),
        plt.Line2D([0], [0], marker='o', color='w', label='Predicted',
                  markerfacecolor='gray', markersize=8, markeredgecolor='black', markeredgewidth=0.5)
    ]
    
    ax.legend(handles=legend_elements, frameon=False, bbox_to_anchor=(1.05, 1), loc='upper left')
    
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_linewidth(0.5)
    ax.spines['bottom'].set_linewidth(0.5)
    
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()


def analyze_zxt(
    model,
    adata, 
    original_zxt_dict,
    split_preds,
    real_times_keep=None,
    pred_times_keep=None,
    output_dir="training_plots",
    plot_format="pdf",
    ancestor_time=None
):
    """
    Comprehensive analysis and visualization of predicted auxiliary latent states.

    This function decodes predicted and real zxt representations into gene
    expression space, integrates them into a shared AnnData object, and performs
    dimensionality reduction, visualization, and differential expression
    analysis. It further evaluates prediction fidelity by comparing marker gene
    expression between real and predicted cells across time points and reports
    R² scores. All results, including UMAP plots and expression matrices, are
    saved for downstream inspection.
    """
    os.makedirs(output_dir, exist_ok=True)

    def zxt_to_expression(zxt_array):
        decoder = model.module.decoder if hasattr(model, 'module') else model.decoder
        zxt_tensor = torch.as_tensor(zxt_array, dtype=torch.float32)
        device = next(decoder.parameters()).device
        zxt_tensor = zxt_tensor.to(device)
        
        with torch.no_grad():
            px_scale = decoder.factor_regressor(zxt_tensor, None)
            library = torch.ones(zxt_tensor.shape[0], 1, device=device) * 1e4
            return (torch.exp(px_scale) * library).cpu().numpy()


    def add_time_prefix(t):
        return t if str(t).startswith('Time_') else f"Time_{t}"


    all_data, labels, sources = [], [], []
    

    for t in (real_times_keep if real_times_keep else original_zxt_dict.keys()):
        t_label = add_time_prefix(t)
        if t in original_zxt_dict:
            expr = zxt_to_expression(original_zxt_dict[t].cpu().numpy())
            expr = expr / expr.sum(axis=1, keepdims=True) * 1e4
            all_data.append(expr)
            labels.extend([t_label] * expr.shape[0])
            sources.extend(["Real"] * expr.shape[0])
    
    for t in (pred_times_keep if pred_times_keep else split_preds.keys()):
        t_label = add_time_prefix(t)
        if t in split_preds:
            expr = zxt_to_expression(split_preds[t])
            expr = expr / expr.sum(axis=1, keepdims=True) * 1e4
            all_data.append(expr)
            labels.extend([t_label] * expr.shape[0])
            sources.extend(["Predicted"] * expr.shape[0])


    combined_adata = sc.AnnData(np.concatenate(all_data))
    combined_adata.obs['Time'] = labels
    combined_adata.obs['Source'] = sources
    combined_adata.var_names = adata.var_names[:combined_adata.shape[1]] 
    

    sc.pp.normalize_total(combined_adata, target_sum=1e4)
    sc.tl.pca(combined_adata)
    sc.pp.neighbors(combined_adata)
    sc.tl.umap(combined_adata)

    def safe_plot(adata_sub, ax, title):
        if adata_sub.n_obs == 0:
            ax.text(0.5, 0.5, f"No data: {title}", ha='center')
        else:
            sc.pl.umap(adata_sub, color='Time', ax=ax, show=False, title=title)
    

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    

    real_mask = combined_adata.obs['Source'] == 'Real'
    safe_plot(combined_adata[real_mask], axes[0], 'Real')
    

    pred_mask = combined_adata.obs['Source'] == 'Predicted'
    if pred_times_keep:
        time_filter = [add_time_prefix(t) for t in pred_times_keep]
        if ancestor_time is not None:
            time_filter.append(f"Time_{ancestor_time}")
        pred_mask &= combined_adata.obs['Time'].isin(time_filter)
    safe_plot(combined_adata[pred_mask], axes[1], 'Predicted')
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/comparison.{plot_format}")
    plt.close()

    combined_adata.obs['comparison_group'] = (
        combined_adata.obs['Time'].astype(str) + '|' + 
        combined_adata.obs['Source'].astype(str)
    )
    

    sc.tl.rank_genes_groups(
        combined_adata, 
        groupby='comparison_group',
        method='wilcoxon',
        use_raw=False
    )
    

    markers = sc.get.rank_genes_groups_df(combined_adata, group=None)
    markers = markers[(markers['pvals_adj'] < 0.05) & (abs(markers['logfoldchanges']) > 1)]
    

    marker_dict = {
        grp: set(sub['names']) 
        for grp, sub in markers.groupby('group')
    }
    
    def calculate_r2(time_point):

        common_markers = (
            marker_dict.get(f"Time_{time_point}|Predicted", set()) &
            marker_dict.get(f"Time_{time_point}|Real", set())
        )
        
        if not common_markers:
            return None
            

        expr_pred = combined_adata[
            combined_adata.obs['comparison_group'] == f"Time_{time_point}|Predicted"
        ][:, list(common_markers)].X.mean(axis=0)
        
        expr_real = combined_adata[
            combined_adata.obs['comparison_group'] == f"Time_{time_point}|Real"
        ][:, list(common_markers)].X.mean(axis=0)
        

        r2 = r2_score(expr_real, expr_pred)
        

        plt.figure(figsize=(7, 6))
        plt.scatter(expr_real, expr_pred, color='lightblue', s=30, alpha=0.8)
        plt.plot([min(expr_real), max(expr_real)],
                 [min(expr_real), max(expr_real)],
                 color='#FFA6A6', linewidth=1, label='Ideal Fit')
        mean_expr = (expr_pred + expr_real) / 2
        top_idx = np.argsort(mean_expr)[-10:]
        for i in top_idx:
            plt.text(expr_real[i], expr_pred[i], list(common_markers)[i],
                     fontsize=5, ha='right', va='bottom')
        plt.xlabel(f"Real (Time_{time_point}|Real)")
        plt.ylabel(f"Predicted (Time_{time_point}|Predicted)")
        plt.title(f"Predicted vs Real (Time {time_point})\nR² = {r2:.3f}")
        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['right'].set_visible(False)
        plt.grid(False)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/r2_time_{time_point}.{plot_format}")
        plt.close()
        return r2
    

    time_points_to_calculate = []
    if pred_times_keep is not None:
        for t in pred_times_keep:
            if isinstance(t, str) and t.startswith('Time_'):
                time_points_to_calculate.append(int(t.split('_')[1]))
            else:
                time_points_to_calculate.append(int(t))
    else:
        time_points_to_calculate = [int(t) for t in split_preds.keys()]
    

    r2_results = {}
    for t in time_points_to_calculate:
        t = int(t)
        if f"Time_{t}|Real" not in marker_dict or f"Time_{t}|Predicted" not in marker_dict:

            continue
            
        r2 = calculate_r2(t)
        if r2 is not None:
            r2_results[f"Time_{t}"] = r2

    pd.DataFrame.from_dict(r2_results, orient='index', columns=['R2']).to_csv(
        f"{output_dir}/r2_scores.csv"
    )
    plot_combined_umap(combined_adata, f"{output_dir}/combined_umap.{plot_format}")

    def save_expression_matrices():

        os.makedirs(f"{output_dir}/expression_matrices", exist_ok=True)
        

        real_data = []
        real_times = []

        for t, zxt in original_zxt_dict.items():
            if real_times_keep is None or t in real_times_keep:
                expr = zxt_to_expression(zxt.cpu().numpy())
                real_data.append(expr)
                real_times.extend([f"Time_{t}"] * expr.shape[0])

        real_adata = sc.AnnData(np.concatenate(real_data))
        real_adata.obs["Time"] = real_times
        real_adata.var_names = adata.var_names[:real_adata.shape[1]]

        real_adata.write(f"{output_dir}/expression_matrices/real_expression.h5ad")

        pred_data = []
        pred_times = []
        for t, data in split_preds.items():
            if pred_times_keep is None or t in pred_times_keep:
                expr = zxt_to_expression(data)
                pred_data.append(expr)
                pred_times.extend([f"Time_{t}"] * expr.shape[0])
    
        pred_adata = sc.AnnData(np.concatenate(pred_data))
        pred_adata.obs['Time'] = pred_times
        pred_adata.var_names = adata.var_names[:pred_adata.shape[1]]
        pred_adata.write(f"{output_dir}/expression_matrices/predicted_expression.h5ad")

        combined_adata.write(f"{output_dir}/expression_matrices/combined_expression.h5ad")
    save_expression_matrices() 
