import os
import pickle
import logging
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.font_manager as fm

from scipy.stats import wasserstein_distance
from sklearn.metrics import r2_score
from sklearn.neighbors import NearestNeighbors
from sklearn.manifold import TSNE
import umap

import torch
import torch.nn.functional as F


"""
Visualization utilities for training and evaluation of the temporal latent state predictor.

This module provides functions for:
1. Standardizing matplotlib plotting style and font properties.
2. Saving figures in consistent directories and formats.
3. Smoothing and plotting training loss curves with moving averages.
4. Visualizing predicted and real latent states (zt and zxt) in 2D using t-SNE,
   enabling color-coded comparisons across time points and modalities.

These tools support model diagnostics, convergence monitoring, and
qualitative assessment of trajectory reconstruction in bidirectional temporal prediction tasks.
"""


def plt_set_default(font_path=None):

    plt.style.use('default')
    if font_path is None:
        font_path = os.path.join(os.path.dirname(__file__), "arial.ttf")

    if not os.path.exists(font_path):
        try:
            arial_font = fm.FontProperties(family='Arial')

        except:
            print(f"file not in: {font_path}")
    else:

        try:
            fm.fontManager.addfont(font_path)
            arial_font = fm.FontProperties(fname=font_path)
            print(f"load: {font_path}")
        except Exception as e:
            print(f"failed: {e}")
    
    mpl.rcParams['font.family'] = ['sans-serif']
    mpl.rcParams['font.sans-serif'] = ['Arial']
    mpl.rcParams['pdf.fonttype'] = 42 
    mpl.rcParams['axes.unicode_minus'] = False
    mpl.rcParams['figure.figsize'] = (3, 3)

    logging.getLogger("fontTools").setLevel(logging.ERROR)
    logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)


def get_plot_dir(args):

    plot_dir = os.path.join(args.save_dir, "training_plots")
    os.makedirs(plot_dir, exist_ok=True)
    return plot_dir


def save_plot(fig_name, args):

    plot_dir = get_plot_dir(args)
    out_path = os.path.join(plot_dir, f"{fig_name}.{args.plot_format}")
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()


def moving_average(data, window_size=10):
    if len(data) < window_size:
        return data
    return [sum(data[i-window_size:i])/window_size for i in range(window_size, len(data)+1)]


def plot_enhanced_loss(system, args):

    os.makedirs(get_plot_dir(args), exist_ok=True)
    plt.figure()
    for k, v in system.loss_history.items():
        if v and isinstance(v[0], (int, float)):
            smoothed = moving_average(v, window_size=10)
            plt.plot(range(len(smoothed)), smoothed, label=k)
    
    plt.title("Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()

    save_plot("enhanced_loss", args)


def plot_tsne_comparison_color(system, z_real_dict, zxt_dict, device, args):
    os.makedirs(get_plot_dir(args), exist_ok=True)

    zt_dim = z_real_dict[next(iter(z_real_dict))].shape[-1]
    zxt_dim = zxt_dict[next(iter(zxt_dict))].shape[-1]
    total_dim = zt_dim + zxt_dim

    all_data, all_labels = [], []

    for t, data in system.pred_output.items():
        if data["ground_truth"] is not None:
            gt = data["ground_truth"].cpu()
            if gt.shape[1] == total_dim:
                all_data.append(gt[:, :zt_dim])
                all_labels.extend([f"Real_zt_t{t}"] * len(gt))
                all_data.append(gt[:, zt_dim:])
                all_labels.extend([f"Real_zxt_t{t}"] * len(gt))

        pred = data["prediction"].cpu()
        if pred.shape[1] == total_dim:
            all_data.append(pred[:, :zt_dim])
            all_labels.extend([f"Pred_zt_t{t}"] * len(pred))
            all_data.append(pred[:, zt_dim:])
            all_labels.extend([f"Pred_zxt_t{t}"] * len(pred))

    if not all_data:
        print(" no data")
        return

    all_data = torch.cat(all_data).detach().numpy()
    all_labels = np.array(all_labels)

    from sklearn.manifold import TSNE
    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    tsne_results = tsne.fit_transform(all_data)

    plt.figure()
    color_map = {
        'Real_zt': 'red', 'Real_zxt': 'blue',
        'Pred_zt': 'orange', 'Pred_zxt': 'green'
    }

    for label_type in ['Real_zt', 'Real_zxt', 'Pred_zt', 'Pred_zxt']:
        mask = np.array([label_type in l for l in all_labels])
        if np.any(mask):
            plt.scatter(tsne_results[mask, 0], tsne_results[mask, 1],
                        c=color_map[label_type],
                        label=label_type,
                        alpha=0.7, s=40 if 'Real' in label_type else 25)

    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    plt.title("Latent Space Comparison (Real vs Pred)")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    save_plot("tsne_comparison_color", args)


def plot_tsne_comparison_all(system, z_real_dict, zxt_dict, device, args):

    plot_dir = os.path.join(args.save_dir, "training_plots")
    os.makedirs(plot_dir, exist_ok=True)
    output_path = os.path.join(plot_dir, f"tsne_comparison_all.{args.plot_format}")

    zt_dim = z_real_dict[next(iter(z_real_dict))].shape[-1]
    zxt_dim = zxt_dict[next(iter(zxt_dict))].shape[-1]
    total_dim = zt_dim + zxt_dim

    real_data, real_labels = [], []
    for t, z in z_real_dict.items():
        real_data.append(z.cpu()[:, :zt_dim])
        real_labels.extend([f"Real_zt_{t}"] * z.shape[0])
        real_data.append(zxt_dict[t].cpu()[:, :zxt_dim])
        real_labels.extend([f"Real_zxt_{t}"] * z.shape[0])

    pred_data, pred_labels = [], []
    for t, data in system.pred_output.items():
        pred = data["prediction"].cpu()
        if pred.shape[1] != total_dim:
            continue
        pred_data.append(pred[:, :zt_dim])
        pred_labels.extend([f"Pred_zt_{t}"] * pred.shape[0])
        pred_data.append(pred[:, zt_dim:])
        pred_labels.extend([f"Pred_zxt_{t}"] * pred.shape[0])

    if not real_data and not pred_data:
        print(" no data")
        return

    all_data = torch.cat(real_data + pred_data).detach().numpy()
    all_labels = np.array(real_labels + pred_labels)

    from sklearn.manifold import TSNE
    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    tsne_results = tsne.fit_transform(all_data)

    plt.figure()  
    unique_labels = np.unique(all_labels)
    colors = plt.cm.tab20(np.linspace(0, 1, len(unique_labels)))

    for label, color in zip(unique_labels, colors):
        idx = np.where(all_labels == label)
        plt.scatter(tsne_results[idx, 0], tsne_results[idx, 1],
                    c=[color], label=label, alpha=0.6)

    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    plt.title("Latent Space Comparison (zt and zxt)")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True)
    plt.tight_layout()


    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_tsne_comparison_separate(system, z_real_dict, zxt_dict, device, args):

    plot_dir = os.path.join(args.save_dir, "training_plots")
    os.makedirs(plot_dir, exist_ok=True)
    output_path = os.path.join(plot_dir, f"tsne_comparison_separate.{args.plot_format}")

    zt_dim = z_real_dict[next(iter(z_real_dict))].shape[-1]
    zxt_dim = zxt_dict[next(iter(zxt_dict))].shape[-1]

    color_map = {
        'Pred_zt': '#1f77b4', 'Pred_zxt': '#ff7f0e',
        'Real_zt': '#9467bd', 'Real_zxt': '#2ca02c'
    }


    real_zt_data, real_zt_labels = [], []
    real_zxt_data, real_zxt_labels = [], []
    for t, z in z_real_dict.items():
        real_zt_data.append(z.cpu())
        real_zt_labels.extend([f"Real_zt_{t}"] * z.shape[0])
        real_zxt_data.append(zxt_dict[t].cpu())
        real_zxt_labels.extend([f"Real_zxt_{t}"] * zxt_dict[t].shape[0])

    pred_zt_data, pred_zt_labels = [], []
    pred_zxt_data, pred_zxt_labels = [], []
    for t, data in system.pred_output.items():
        pred = data["prediction"].cpu()
        if pred.shape[1] != zt_dim + zxt_dim:
            continue
        pred_zt_data.append(pred[:, :zt_dim])
        pred_zt_labels.extend([f"Pred_zt_{t}"] * pred.shape[0])
        pred_zxt_data.append(pred[:, zt_dim:])
        pred_zxt_labels.extend([f"Pred_zxt_{t}"] * pred.shape[0])

    if not (real_zt_data or real_zxt_data or pred_zt_data or pred_zxt_data):
        print(" no data")
        return

    plt.figure()  
    plt.subplot(1, 2, 1)
    zt_data = torch.cat(real_zt_data + pred_zt_data).detach().numpy()
    zt_labels = np.array(real_zt_labels + pred_zt_labels)
    from sklearn.manifold import TSNE
    zt_results = TSNE(n_components=2, perplexity=30, random_state=42).fit_transform(zt_data)
    for label in np.unique(zt_labels):
        prefix = '_'.join(label.split('_')[:-1])
        color = color_map.get(prefix, '#333333')
        idx = np.where(zt_labels == label)
        plt.scatter(zt_results[idx, 0], zt_results[idx, 1], c=[color], label=label, alpha=0.6)
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    plt.title("zt Space Comparison")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True)

    plt.subplot(1, 2, 2)
    zxt_data = torch.cat(real_zxt_data + pred_zxt_data).detach().numpy()
    zxt_labels = np.array(real_zxt_labels + pred_zxt_labels)
    zxt_results = TSNE(n_components=2, perplexity=30, random_state=42).fit_transform(zxt_data)
    for label in np.unique(zxt_labels):
        prefix = '_'.join(label.split('_')[:-1])
        color = color_map.get(prefix, '#333333')
        idx = np.where(zxt_labels == label)
        plt.scatter(zxt_results[idx, 0], zxt_results[idx, 1], c=[color], label=label, alpha=0.6)
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    plt.title("zxt Space Comparison")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True)

    plt.tight_layout()

    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_full_timeline(system, z_real_dict, zxt_dict, device, args):

    plot_dir = os.path.join(args.save_dir, "training_plots")
    os.makedirs(plot_dir, exist_ok=True)
    output_path = os.path.join(plot_dir, f"full_timeline.{args.plot_format}")

    plt.figure()  

    all_t = sorted(system.pred_output.keys())
    real_zt, real_zxt, pred_zt, pred_zxt = [], [], [], []

    for t in all_t:
        gt = system.pred_output[t].get("ground_truth")
        pred = system.pred_output[t]["prediction"]
        if gt is not None:
            real_zt.append(gt[0, 0].item())
            real_zxt.append(gt[0, 10].item())
        else:
            real_zt.append(np.nan)
            real_zxt.append(np.nan)
        pred_zt.append(pred[0, 0].item())
        pred_zxt.append(pred[0, 10].item())

    plt.subplot(2, 1, 1)
    plt.plot(all_t, pred_zt, 'b--o', label='Predicted zt', markersize=8, alpha=0.8)
    valid_real_indices = [i for i, val in enumerate(real_zt) if not np.isnan(val)]
    if valid_real_indices:
        plt.scatter(np.array(all_t)[valid_real_indices],
                    np.array(real_zt)[valid_real_indices],
                    c='r', s=100, label='Real zt', zorder=3)
    plt.xlabel("Time Step")
    plt.ylabel("Latent Dimension 0 (zt)")
    plt.title("ZT Timeline (Red=Real, Blue=Predicted)")
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(all_t, pred_zxt, 'g--o', label='Predicted zxt', markersize=8, alpha=0.8)
    valid_real_indices = [i for i, val in enumerate(real_zxt) if not np.isnan(val)]
    if valid_real_indices:
        plt.scatter(np.array(all_t)[valid_real_indices],
                    np.array(real_zxt)[valid_real_indices],
                    c='orange', s=100, label='Real zxt', zorder=3)
    plt.xlabel("Time Step")
    plt.ylabel("Latent Dimension 0 (zxt)")
    plt.title("ZXT Timeline (Orange=Real, Green=Predicted)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_prediction_comparison(system, z_real_dict, zxt_dict, predict_time_points, args, filename="prediction_comparison"):

    plot_dir = os.path.join(args.save_dir, "training_plots")
    os.makedirs(plot_dir, exist_ok=True)
    output_path = os.path.join(plot_dir, f"{filename}.{args.plot_format}")
    valid_times = [t for t in predict_time_points 
                   if t in z_real_dict and t in system.pred_dict]
    if not valid_times:
        print("no data")
        return
    
    plt.figure() 
    plt.subplot(2, 2, 1)
    for t in valid_times:
        real = z_real_dict[t][:, 0].cpu().numpy()
        pred = system.pred_dict[t]["prediction"][:, 0].cpu().numpy()
        plt.scatter(real, pred, alpha=0.6, label=f't={t}')
    plt.plot([-3,3], [-3,3], 'k--')
    plt.xlabel("Real ZT")
    plt.ylabel("Pred ZT")
    plt.title("ZT Value Comparison")
    plt.legend()
    plt.grid(False) 

    plt.subplot(2, 2, 2)
    for t in valid_times:
        real = zxt_dict[t][:, 0].cpu().numpy()
        pred = system.pred_dict[t]["prediction"][:, 64].cpu().numpy()
        plt.scatter(real, pred, alpha=0.6, label=f't={t}')
    plt.plot([-3,3], [-3,3], 'k--')
    plt.xlabel("Real ZXT")
    plt.ylabel("Pred ZXT")
    plt.title("ZXT Value Comparison")
    plt.legend()
    plt.grid(False) 

    plt.subplot(2, 2, 3)
    zt_errors = []
    zxt_errors = []
    for t in valid_times:
        zt_errors.extend((z_real_dict[t][:, 0] - system.pred_dict[t]["prediction"][:, 0]).cpu().numpy())
        zxt_errors.extend((zxt_dict[t][:, 0] - system.pred_dict[t]["prediction"][:, 64]).cpu().numpy())
    plt.hist(zt_errors, bins=30, alpha=0.7, label='ZT Errors')
    plt.hist(zxt_errors, bins=30, alpha=0.7, label='ZXT Errors')
    plt.xlabel("Prediction Error")
    plt.ylabel("Frequency")
    plt.title("Error Distribution")
    plt.legend()
    plt.grid(False)  

    plt.subplot(2, 2, 4)
    zt_mae = [F.l1_loss(system.pred_dict[t]["prediction"][:, :64], z_real_dict[t]).item() 
              for t in valid_times]
    zxt_mae = [F.l1_loss(system.pred_dict[t]["prediction"][:, 64:], zxt_dict[t]).item() 
               for t in valid_times]
    x = range(len(valid_times))
    plt.bar(x, zt_mae, width=0.4, label='ZT MAE')
    plt.bar([i+0.4 for i in x], zxt_mae, width=0.4, label='ZXT MAE')
    plt.xticks([i+0.2 for i in x], valid_times)
    plt.xlabel("Time Point")
    plt.ylabel("MAE")
    plt.title("MAE per Time Point")
    plt.legend()
    plt.grid(False) 

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def visualize_predictions(
    original_z_real_dict, original_zxt_dict, pred_dict, 
    zt_dim=10, zxt_dim=10,  
    real_times_keep=None, pred_times_keep=None,
    args=None
):

    if args is not None and hasattr(args, 'save_dir'):
        output_dir = os.path.join(args.save_dir, "training_plots")
    else:
        output_dir = "training_plots"
    os.makedirs(output_dir, exist_ok=True)
    split_preds = split_prediction(pred_dict, zt_dim, zxt_dim)

    plot_format = getattr(args, 'plot_format', 'pdf') if args is not None else 'pdf'
    _generate_all_umaps(original_z_real_dict, original_zxt_dict, 
                        split_preds, real_times_keep, pred_times_keep,
                        output_dir, plot_format)


def _generate_all_umaps(
    real_z_dict, real_zxt_dict, split_preds,
    real_times_keep, pred_times_keep,
    output_dir, plot_format
):

    _plot_umap(
        real_dict=real_z_dict,
        pred_dict=split_preds,
        key='zt_pred',
        title='UMAP of zt (All Time Points)',
        filename='umap_all_zt',
        output_dir=output_dir,
        plot_format=plot_format
    )
    
    _plot_umap(
        real_dict=real_zxt_dict,
        pred_dict=split_preds,
        key='zxt_pred',
        title='UMAP of zxt (All Time Points)',
        filename='umap_all_zxt',
        output_dir=output_dir,
        plot_format=plot_format
    )
    
    if real_times_keep and pred_times_keep:
        _plot_umap(
            real_dict=real_z_dict,
            pred_dict=split_preds,
            key='zt_pred',
            title=f'UMAP of zt (Filtered)\nReal: {real_times_keep} Pred: {pred_times_keep}',
            filename=f'umap_filtered_zt_real_{"_".join(map(str, real_times_keep))}_pred_{"_".join(map(str, pred_times_keep))}',
            output_dir=output_dir,
            plot_format=plot_format,
            real_times_keep=real_times_keep,
            pred_times_keep=pred_times_keep
        )
        
        _plot_umap(
            real_dict=real_zxt_dict,
            pred_dict=split_preds,
            key='zxt_pred',
            title=f'UMAP of zxt (Filtered)\nReal: {real_times_keep} Pred: {pred_times_keep}',
            filename=f'umap_filtered_zxt_real_{"_".join(map(str, real_times_keep))}_pred_{"_".join(map(str, pred_times_keep))}',
            output_dir=output_dir,
            plot_format=plot_format,
            real_times_keep=real_times_keep,
            pred_times_keep=pred_times_keep
        )


def _plot_umap(
    real_dict, pred_dict, key, title, filename, 
    output_dir, plot_format, 
    real_times_keep=None, pred_times_keep=None
):

    os.makedirs(output_dir, exist_ok=True)
    embeddings = []
    labels = []

    for t, real in real_dict.items():
        if real_times_keep is None or t in real_times_keep:
            real_np = real.detach().cpu().numpy()
            embeddings.append(real_np)
            labels.extend([f"Real_{t}"] * len(real_np))
    
    for t, pred in pred_dict.items():
        if (pred_times_keep is None or t in pred_times_keep) and key in pred:
            pred_np = pred[key]
            embeddings.append(pred_np)
            labels.extend([f"Pred_{t}"] * len(pred_np))
    
    if not embeddings:
        return
    
    reducer = umap.UMAP(random_state=42)
    umap_results = reducer.fit_transform(np.vstack(embeddings))

    plt.figure()
    unique_labels = sorted(set(labels), key=lambda x: (x.split('_')[0], int(x.split('_')[1])))
    
    for label in unique_labels:
        idx = np.array(labels) == label
        plt.scatter(umap_results[idx, 0], umap_results[idx, 1], 
                   label=label, s=10, alpha=0.6)
    
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.title(title)
    ax = plt.gca()
    for spine in ax.spines.values():
        spine.set_visible(False)
    plt.grid(False)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/{filename}.{plot_format}", format=plot_format)
    plt.close()


def compute_compactness(embeddings_dict, time_keys=None):

    nn_dists = {}
    for t, emb in embeddings_dict.items():
        if time_keys is not None and t not in time_keys:
            continue
        emb_np = emb.detach().cpu().numpy() if torch.is_tensor(emb) else emb
        nbrs = NearestNeighbors(n_neighbors=2).fit(emb_np)
        distances, _ = nbrs.kneighbors(emb_np)
        nn_dists[t] = np.mean(distances[:, 1]) 
    return nn_dists


def analyze_and_visualize(
    original_z_real_dict, original_zxt_dict, pred_dict,
    zt_dim, zxt_dim, args=None, 
    real_times_keep=None, pred_times_keep=None
):

    if args is not None and hasattr(args, 'save_dir'):
        output_dir = os.path.join(args.save_dir, "training_plots")
    else:
        output_dir = "training_plots"
    os.makedirs(output_dir, exist_ok=True)

    plot_format = getattr(args, 'plot_format', 'pdf') if args is not None else 'pdf'

    split_preds = split_prediction(pred_dict, zt_dim, zxt_dim)
    
    negative_times = [t for t in split_preds.keys() if t <= 0]  
    if negative_times:
        pred_nn_dists = compute_compactness(
            {t: v['zt_pred'] for t, v in split_preds.items() if t in negative_times},
            time_keys=None
        )
        ancestor_time = min(pred_nn_dists, key=pred_nn_dists.get)

        if real_times_keep is None:
            real_times_keep = []
        if pred_times_keep is None:
            pred_times_keep = []
        
        if isinstance(pred_times_keep, set):
            pred_times_keep = list(pred_times_keep)
        pred_times_keep = list(set(pred_times_keep + [ancestor_time]))
    pred_times= [-10,-9,-8,-7,-6,-5,-4,-3,-2,-1,0]

    _plot_distance_metrics(
        split_preds, 
        pred_times,
        args
    )

    _generate_all_umaps(
        original_z_real_dict, original_zxt_dict,
        split_preds, real_times_keep, pred_times_keep,
        output_dir, plot_format
    )
    return ancestor_time
    

def _plot_distance_metrics(pred_dict, pred_times, args=None):

    if args is not None and hasattr(args, 'save_dir'):
        output_dir = os.path.join(args.save_dir, "training_plots")
    else:
        output_dir = "training_plots"
    os.makedirs(output_dir, exist_ok=True)

    plot_format = getattr(args, 'plot_format', 'pdf') if args is not None else 'pdf'

    negative_times = [t for t in pred_times if t < 0]

    pred_dists = compute_compactness(
        {t: v['zt_pred'] for t, v in pred_dict.items() if t in negative_times},
        negative_times
    )
    
    if not pred_dists:
        print("No negative time points found to plot.")
        return
    
    times_sorted = sorted(pred_dists.keys())
    dists_sorted = [pred_dists[t] for t in times_sorted]
    
    plt.figure()
    
    plt.plot(times_sorted, dists_sorted, marker='o', color='royalblue', 
             linewidth=2, markersize=8)
    

    ancestor_time = min(pred_dists, key=pred_dists.get)
    plt.scatter(ancestor_time, pred_dists[ancestor_time],
               s=150, c='gold', edgecolors='k', zorder=10,
               label=f'Ancestor (Time {ancestor_time})')
    
    plt.xlabel('Time Point', fontsize=12)
    plt.ylabel('Mean Nearest Neighbor Distance', fontsize=12)
    plt.title('Cell Compactness Across Time Points', fontsize=14)

    plt.legend(fontsize=10, frameon=False)
    
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(0.5)
    ax.spines['bottom'].set_linewidth(0.5)
    plt.grid(False)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/nn_distances.{plot_format}", 
               format=plot_format, dpi=300)
    plt.close()


def _plot_real_n_umap(decoder_n, real_dict, times_keep, args=None):

    if args is not None and hasattr(args, 'save_dir'):
        output_dir = os.path.join(args.save_dir, "training_plots")
    else:
        output_dir = "training_plots"
    os.makedirs(output_dir, exist_ok=True)

    plot_format = getattr(args, 'plot_format', 'pdf') if args is not None else 'pdf'

    z_all, n_all, times = _extract_z_and_n(real_dict, decoder_n, times_keep)
    
    reducer = umap.UMAP(random_state=42)
    z_umap = reducer.fit_transform(z_all)

    plt.figure()

    plt.subplot(121)
    for t in np.unique(times):
        idx = times == t
        plt.scatter(z_umap[idx, 0], z_umap[idx, 1], 
                   label=f'Time {t}', s=10, alpha=0.6)
    plt.legend()
    plt.title('UMAP of Real zt (by Time)')
    
    plt.subplot(122)
    sc = plt.scatter(z_umap[:, 0], z_umap[:, 1], 
                   c=n_all[:, 0], cmap='viridis', s=10, alpha=0.6)
    plt.colorbar(sc, label='n value (dim 0)')
    plt.title('UMAP of Real zt (colored by n)')

    plt.tight_layout()
    plt.savefig(f"{output_dir}/umap_real_n.{plot_format}", 
               format=plot_format, dpi=300)
    plt.close()


def _extract_z_and_n(real_dict, decoder_n, times_keep):

    z_list, n_list, times = [], [], []
    for t in sorted(real_dict.keys()):
        if t not in times_keep:
            continue
        device = next(decoder_n.parameters()).device
        z = real_dict[t].cpu()
        z = z.to(device)
        with torch.no_grad():
            n = F.softplus(decoder_n(z)).cpu()
        z_list.append(z)
        n_list.append(n)
        times.extend([t] * len(z))
    return (
        torch.cat(z_list).cpu().numpy(),
        torch.cat(n_list).cpu().numpy(),
        np.array(times))
    

def load_model_and_data(model_path, zt_path, zxt_path):

    with open(model_path, 'rb') as f:
        model_dict = pickle.load(f)
    
    if isinstance(model_dict, dict):
        # 如果是字典，遍历字典中的模型
        for key in model_dict:
            if hasattr(model_dict[key], 'eval'):
                model_dict[key].eval()
    else:
        # 如果不是字典，直接是模型对象
        if hasattr(model_dict, 'eval'):
            model_dict.eval()
    
    original_z_real_dict = torch.load(zt_path)
    original_zxt_dict = torch.load(zxt_path)
    
    original_z_real_dict = dict(sorted(original_z_real_dict.items()))
    original_zxt_dict = dict(sorted(original_zxt_dict.items()))
    
    return model_dict, original_z_real_dict, original_zxt_dict


def split_prediction(pred_dict, zt_dim, zxt_dim):

    split_results = {}
    for t, data in pred_dict.items():
        if not isinstance(data, dict):
            continue
            
        pred = data.get('prediction', None)
        if pred is None:
            continue
            
        pred = torch.tensor(pred) if not torch.is_tensor(pred) else pred
        split_results[t] = {
            'zt_pred': pred[:, :zt_dim].cpu().detach().numpy(),
            'zxt_pred': pred[:, zt_dim: zt_dim+zxt_dim].cpu().detach().numpy()
        }
    return split_results


def analyze_n_distribution(
    model, original_z_real_dict, split_pred_dict, 
    real_times_keep=[1,2,3], pred_times_keep=None,
    args=None
):
 
    if isinstance(model, dict):
        decoder_n = model['model_n'].model.decoder_n
    else:
        decoder_n = model.model.decoder_n
    
    _plot_real_n_umap(
        decoder_n, original_z_real_dict, 
        real_times_keep, args)

    if pred_times_keep:
        _plot_combined_n_umap(
            decoder_n, original_z_real_dict, split_pred_dict,
            real_times_keep, pred_times_keep,
            args)


def plot_distribution_comparison(
    real_dict, pred_dict, decoder_n, 
    predict_time=None,  
    args=None
):
    device = next(decoder_n.parameters()).device
    if args is not None and hasattr(args, 'save_dir'):
        output_dir = os.path.join(args.save_dir, "training_plots")
    else:
        output_dir = "training_plots"
    os.makedirs(output_dir, exist_ok=True)

    plot_format = getattr(args, 'plot_format', 'pdf') if args is not None else 'pdf'
    
    real_z = real_dict[predict_time].float().to(device)
    pred_data = pred_dict[predict_time]
    
    if isinstance(pred_data, dict):
        pred_z = torch.tensor(pred_data['zt_pred']).float().to(device)
    elif isinstance(pred_data, (np.ndarray, torch.Tensor)):
        pred_z = torch.tensor(pred_data).float().to(device)
    
    with torch.no_grad():
        real_n = F.softplus(decoder_n(real_z)).cpu().numpy().squeeze()
        pred_n = F.softplus(decoder_n(pred_z)).cpu().numpy().squeeze()
    
    from sklearn.metrics import r2_score
    r2 = r2_score(real_n[:len(pred_n)], pred_n)
    wd = wasserstein_distance(real_n, pred_n)

    plt.figure()

    plt.hist(real_n, bins=50, density=True, alpha=0.7, 
             label=f'Real Time {predict_time}', color='#1f78b4')
    plt.hist(pred_n, bins=50, density=True, alpha=0.7,
             label=f'Pred Time {predict_time} (R²={r2:.3f}, WD={wd:.3f})', 
             color='#e6550d')

    plt.xlabel('n value')
    plt.ylabel('Density')
    plt.title(f'n Distribution Comparison\n(Time {predict_time})')
    plt.legend()

    ax = plt.gca()
    for spine in ax.spines.values():
        spine.set_visible(False)
    plt.grid(False)
    plt.tight_layout()
    output_path = f"{output_dir}/n_compare_t{predict_time}.{plot_format}"
    plt.savefig(output_path, format=plot_format, dpi=300)
    plt.close()
    
    return r2, wd


def _extract_all_z_and_n(
    real_dict, split_pred_dict, decoder_n, 
    real_times_keep, pred_times_keep
):  
    z_list, n_list, labels, times = [], [], [], []

    for t in sorted(real_dict.keys()):
        if t not in real_times_keep:
            continue
        
        z = real_dict[t].cpu()
        device = next(decoder_n.parameters()).device
        z = z.to(device)
        with torch.no_grad():
            n = F.softplus(decoder_n(z)).cpu()
        z_list.append(z)
        n_list.append(n)
        labels.extend([f"Real_{t}"] * len(z))
        times.extend([t] * len(z))
    
    for t in sorted(split_pred_dict.keys()):
        if t not in pred_times_keep:
            continue
        if 'zt_pred' not in split_pred_dict[t]:
            continue
            
        z_pred = torch.tensor(split_pred_dict[t]['zt_pred']).float()
        with torch.no_grad():
            n_pred = F.softplus(decoder_n(z_pred)).cpu()
        
        z_list.append(z_pred)
        n_list.append(n_pred)
        labels.extend([f"Pred_{t}"] * len(z_pred))
        times.extend([t] * len(z_pred))
    
    return (
        torch.cat(z_list).cpu().numpy(),
        torch.cat(n_list).cpu().numpy(),
        np.array(labels),
        np.array(times))


def _plot_combined_n_umap(
    decoder_n, original_z_real_dict, split_pred_dict,
    real_times_keep, pred_times_keep,
    args=None
):

    if args is not None and hasattr(args, 'save_dir'):
        output_dir = os.path.join(args.save_dir, "training_plots")
    else:
        output_dir = "training_plots"
    os.makedirs(output_dir, exist_ok=True)

    plot_format = getattr(args, 'plot_format', 'pdf') if args is not None else 'pdf'
    z_all, n_all, labels, times = _extract_all_z_and_n(
        original_z_real_dict, 
        split_pred_dict,
        decoder_n,
        real_times_keep,
        pred_times_keep
    )
    
    reducer = umap.UMAP(random_state=42, n_jobs=1)
    z_umap = reducer.fit_transform(z_all)
    
    plt.figure()
    plt.subplot(121)
    unique_labels = sorted(set(labels), key=lambda x: int(x.split('_')[-1]))
    for label in unique_labels:
        color = '#ff7f0e' if 'Pred_-5' in label else None  
        idx = np.array(labels) == label
        plt.scatter(z_umap[idx, 0], z_umap[idx, 1], 
                   label=label, s=10, alpha=0.6, c=color)
    
    plt.title('UMAP of zt (Real & Predicted)')
    plt.legend(bbox_to_anchor=(1.05, 1))
    
    plt.subplot(122)
    sc = plt.scatter(z_umap[:, 0], z_umap[:, 1], 
                   c=n_all[:, 0], cmap='viridis', s=10, alpha=0.6)
    ancestor_idx = np.where([l.startswith('Pred_-5') for l in labels])[0]
    plt.scatter(z_umap[ancestor_idx, 0], z_umap[ancestor_idx, 1],
               s=50, edgecolors='red', facecolors='none',
               label='Ancestor (Time -5)')
    plt.colorbar(sc, label='n value (dim 0)')
    plt.title('UMAP colored by n (Ancestor Highlighted)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{output_dir}/umap_combined_n.{plot_format}", 
               format=plot_format, dpi=300)
    plt.close()


def plot_tsne_comparison(system, z_real_dict, zxt_dict, device, args, zt_dim, zxt_dim):

    plot_dir = os.path.join(args.save_dir, "training_plots") if hasattr(args, "save_dir") else "training_plots"
    os.makedirs(plot_dir, exist_ok=True)
    plot_format = getattr(args, "plot_format", "pdf")
    out_path = os.path.join(plot_dir, f"plot_tsne_comparison.{plot_format}")
    total_dim = zt_dim + zxt_dim
    all_data, all_labels = [], []

    for t, data in system.pred_output.items():
        gt = data.get("ground_truth", None)
        pred = data.get("prediction", None)

        if gt is not None and gt.shape[1] == total_dim:
            all_data.append(gt[:, :zt_dim].cpu())
            all_labels.extend([f"Real_zt_t{t}"] * gt.shape[0])
            all_data.append(gt[:, zt_dim:].cpu())
            all_labels.extend([f"Real_zxt_t{t}"] * gt.shape[0])

        if pred is not None and pred.shape[1] == total_dim:
            all_data.append(pred[:, :zt_dim].cpu())
            all_labels.extend([f"Pred_zt_t{t}"] * pred.shape[0])
            all_data.append(pred[:, zt_dim:].cpu())
            all_labels.extend([f"Pred_zxt_t{t}"] * pred.shape[0])

    all_data = torch.cat(all_data).detach().numpy()
    all_labels = np.array(all_labels)

    # TSNE
    from sklearn.manifold import TSNE
    tsne_results = TSNE(n_components=2, perplexity=30, random_state=42).fit_transform(all_data)

    plt.figure()
    unique_times = sorted(set(int(l.split("_t")[-1]) for l in all_labels))
    cols = min(3, len(unique_times))
    rows = (len(unique_times) + cols - 1) // cols

    for i, t in enumerate(unique_times):
        plt.subplot(rows, cols, i + 1)
        mask = np.array([f"_t{t}" in l for l in all_labels])
        t_data = tsne_results[mask]
        t_labels = all_labels[mask]

        for label_type in ['Real_zt', 'Real_zxt', 'Pred_zt', 'Pred_zxt']:
            type_mask = np.array([label_type in l for l in t_labels])
            if np.any(type_mask):
                color = 'red' if 'Real' in label_type else 'blue'
                marker = 'o' if 'zt' in label_type else 's'
                plt.scatter(t_data[type_mask, 0], t_data[type_mask, 1],
                            c=color, marker=marker, label=label_type,
                            alpha=0.7, s=100 if 'Real' in label_type else 60)
        plt.title(f"Time Step {t}")
        plt.xlabel("TSNE-1")
        plt.ylabel("TSNE-2")
        if i == 0:
            plt.legend()

    plt.suptitle("Latent Space Comparison by Time Step\n(Red=Real, Blue=Predicted | Circle=zt, Square=zxt)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_training_loss(system, args, filename="full_loss"):

    plot_dir = os.path.join(args.save_dir, "training_plots")
    os.makedirs(plot_dir, exist_ok=True)
    output_path = os.path.join(plot_dir, f"{filename}.{args.plot_format}")
    plt.figure()
    plt.plot(system.loss_history['total'], label='Total Loss', color='#2D5F8A', linewidth=2)
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Loss", fontsize=12)
    plt.title("Training Loss Curve", fontsize=14)
    plt.legend()
    plt.grid(False)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_benchmark1(results):
    plt.bar(results.keys(), results.values())
    plt.title("Model Comparison")
    plt.tight_layout()
    plt.savefig(f"training_plots/benchmark1.png")  
    plt.close()
