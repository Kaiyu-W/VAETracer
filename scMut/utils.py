import random
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn")
from sklearn.manifold import TSNE

from .log import logger
try:
    from umap import UMAP
except:
    logger.warning(
    	'UMAP cannot work for `umap` package is not installed. '
    	'Please use `pip install umap-learn` to install it.'
    )

import os
import multiprocessing
cpu_count = multiprocessing.cpu_count()
if cpu_count > 2:
    if 'NUMEXPR_MAX_THREADS' in os.environ and os.environ["NUMEXPR_MAX_THREADS"] == str(cpu_count-1):
        pass
    else:
        os.environ["NUMEXPR_MAX_THREADS"] = str(cpu_count-1)
        logger.info(f"(Import) Set NUMEXPR_MAX_THREADS to {cpu_count-1}.")

import torch
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
logger.info(f'(Import) Use {DEVICE} as default device!')


def _input_tensor(x, device=None, dtype=None):
    if isinstance(x, torch.Tensor):
        x = x.clone().detach()
        if device is not None:
            x=x.to(device)
        if dtype is not None:
            x=x.to(dtype)
    else:
        try:
            x = torch.tensor(x, device=device, dtype=dtype)
        except Exception as e:
            raise ValueError(f"Cannot input as torch.Tensor: {e}")
    return x

def _softplus_inverse(x):
    logit = x + torch.log1p(torch.exp(-x)) # torch.log(torch.exp(y) - 1)
    return logit

def set_seed(seed, device=DEVICE):
    random.seed(seed)
    np.random.seed(seed)
    if (
        isinstance(device, str) and 'cuda' in device
    ) or (
        isinstance(device, torch.device) and device.type == 'cuda'
    ):
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.manual_seed(seed)

def visualize_loss(losses, yscale=None, save=None, show=True):
    """
    Visualize training loss curve.
    
    Args:
        losses (list): List of loss values during training
        yscale (str, optional): Scale of y-axis ('linear', 'log', etc.). Defaults to None
    """

    plt.figure(figsize=(5,4))
    plt.plot(losses, label='Loss')
    plt.title('Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    if yscale:
        plt.yscale(yscale)
    if save is not None:
        plt.savefig(save, bbox_inches='tight')
    if show:
        plt.show()

def visualize_loss_k(loss_dict, k_optimal=None, smooth=False, save=None, show=True):
    keys = sorted(loss_dict.keys())
    values = [loss_dict[k] for k in keys]

    plt.figure(figsize=(5,4))
    if smooth:
        plt.plot(keys, values, linestyle='-', marker='o', markersize=0.1, color='black')
    else:
        plt.scatter(keys, values, s=0.5, c='black')

    if k_optimal is not None:
        plt.scatter(x=k_optimal, y=loss_dict[k_optimal], s=50, c='red')

    plt.xlabel('K')
    plt.ylabel('Loss')

    if save is not None:
        plt.savefig(save, bbox_inches='tight')
    if show:
        plt.show()

def plot_metrics(model, yscale=None, nrow=1, save=None, show=True):
    """
    Plot training and validation metrics over time.
    
    Args:
        model (AutoEncoderModel): Trained model with recorded metrics
        yscale (str, optional): Scale for y-axis (e.g., 'log')
        nrow (int, optional): Number of rows for subplots
    
    Notes:
        Plots three metrics if available:
        - total_loss
        - reconstruction_loss
        - kl_loss (for VAE only)
    """

    # Get available metrics from the first training record
    if not model.train_metrics:
        logger.warning("No training metrics available to plot")
        return
        
    metrics = list(model.train_metrics[0].keys())
    ncol = (len(metrics) + nrow - 1) // nrow  # Calculate number of columns
    fig, axes = plt.subplots(nrow, ncol, figsize=(5*ncol, 4*nrow))
    
    # Flatten axes array if necessary
    if nrow * ncol > 1:
        axes = axes.flatten()
    else:
        axes = [axes]
    
    for i, metric in enumerate(metrics):
        ax = axes[i]
        train_values = [epoch_metrics[metric] for epoch_metrics in model.train_metrics]
        ax.plot(train_values, label='Train')
        
        if model.valid_metrics:
            valid_values = [epoch_metrics[metric] for epoch_metrics in model.valid_metrics]
            ax.plot(valid_values, label='Valid')
        
        ax.set_title(metric)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Value')
        ax.legend()
        if yscale:
            ax.set_yscale(yscale)
    
    # Remove any unused subplots
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])
    
    plt.tight_layout()
    if save is not None:
        plt.savefig(save, bbox_inches='tight')
    if show:
        plt.show()

def plot_latent_space(model, data=None, labels=None, reduction='tsne', save=None, show=True, return_z=False, use_cpu=False, **embed_kwargs):
    """
    Visualize the latent space using dimensionality reduction if needed.
    
    Args:
        model (AutoEncoderModel): Trained model
        data (numpy.ndarray): Input data to encode. Defaults to None and use the latent space of trained model
        labels (numpy.ndarray, optional): Labels for coloring points. Defaults to None
        reduction (str): t-SNE or UMAP embedding method
    
    Notes:
        - If latent dimension > 2, uses t-SNE for visualization
        - If labels provided, points are colored according to their labels
    """
    # define embedding method
    _embed_kwargs = {}
    if reduction == 'umap':
        try:
            from umap import UMAP
        except ImportError as e:
            raise Exception('The `umap` package is not installed. Please use `pip install umap-learn` to install it.')

        embedding = UMAP
    elif reduction == 'tsne':
        embedding = TSNE
        _embed_kwargs['init'] = "pca"
        _embed_kwargs['learning_rate'] = "auto"
    else:
        raise ValueError('Only tsne and umap are available for reduction!')
    _embed_kwargs.update(embed_kwargs)

    # Get latent representations
    if data is None:
        Z = model.Z
    else:
        data = _input_tensor(data, dtype=model.dtype, device=torch.device('cpu') if use_cpu else model.device)
        model.model.eval()
        with torch.no_grad():
            outs = model.model.encoder(data) # (mu, logvar) for VAE; z for AE
            if model.model_type == 'AE':
                Z = outs
            else:
                Z = outs[0]
            Z = Z.cpu().numpy()
    
    # Use t-SNE for dimensionality reduction if z_dim > 2
    if Z.shape[1] > 2:
        z = embedding(n_components=2, random_state=42, **_embed_kwargs).fit_transform(Z)
    else:
        if Z.shape[1] < 2:
            return
        z = Z
    
    plt.figure(figsize=(6.4, 4.8))
    if labels is None:
        scatter = plt.scatter(z[:, 0], z[:, 1], c=None)
    elif isinstance(labels[0], (str, np.str_)):
        unique_labels = list(set(labels))
        colors = plt.cm.get_cmap("tab20", len(unique_labels))
        scatter = plt.scatter(z[:, 0], z[:, 1], c=[unique_labels.index(label) for label in labels], cmap=colors)
        plt.colorbar(scatter, ticks=range(len(unique_labels)), label='Categories')
        plt.clim(-0.5, len(unique_labels) - 0.5)
    else:
        scatter = plt.scatter(z[:, 0], z[:, 1], c=labels, cmap='viridis')
        plt.colorbar(scatter)
    plt.title('Latent Space Visualization')
    
    if save is not None:
        plt.savefig(save, bbox_inches='tight')
    if show:
        plt.show()
    if return_z:
        return z

def plot_regplot(data, x, y, lowess=False, save=None, show=True):
    plt.figure(figsize=(5,4))
    sns.regplot(
        x=data[x],y=data[y],lowess=lowess,
        scatter_kws=dict(s=0.1,alpha=1,color='black'), 
        line_kws=dict(color='red')
    )
    plt.xlabel(x)
    plt.ylabel(y)
    if save is not None:
        plt.savefig(save, bbox_inches='tight')
    if show:
        plt.show()