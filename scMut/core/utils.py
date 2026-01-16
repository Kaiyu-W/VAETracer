import random
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn")
from sklearn.manifold import TSNE

from .log import logger
from .typing import *
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


def _input_tensor(
    x: Union[np.ndarray, torch.Tensor, List],
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    """
    Convert input data to a detached tensor on specified device and dtype.

    If input is already a tensor, clones and detaches it. Otherwise converts from array/list.
    
    Args:
        x: Input data (array, list, or tensor).
        device: Target device ('cpu' or 'cuda'). Uses current if None.
        dtype: Target data type (e.g., torch.float32).

    Returns:
        A detached, cloned tensor on the specified device and dtype.
    """

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

def _softplus_inverse(x: torch.Tensor) -> torch.Tensor:
    """
    Compute the inverse of the softplus function in a numerically stable way.

    Given values that were transformed by softplus (a smooth function mapping real numbers to positive values),
    this function recovers the original input values before transformation.

    Useful for decoding outputs in probabilistic models where parameters are constrained to be positive.

    Args:
        x: A tensor of values that have been passed through a softplus operation, with values greater than zero.

    Returns:
        A tensor of the same shape, containing the recovered original values (which can be any real number).
    """

    logit = x + torch.log1p(torch.exp(-x)) # torch.log(torch.exp(y) - 1)
    return logit

def set_seed(seed: int, device: Union[str, torch.device] = DEVICE) -> None:
    """
    Set random seeds for reproducibility across libraries.

    Applies seed to Python's random, NumPy, and PyTorch (with CUDA determinism if available).

    Args:
        seed: Integer seed value.
        device: Device context; used to determine whether to enable CUDA determinism.
    """

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

def visualize_loss(
    losses: List[float],
    yscale: Optional[str] = None,
    save: Optional[str] = None,
    show: bool = True,
) -> None:
    """
    Visualize training loss curve over epochs.

    Plots scalar loss values with optional logarithmic scaling.

    Args:
        losses: Sequence of loss values per epoch.
        yscale: Y-axis scale ('linear', 'log', etc.). Default: None.
        save: File path to save figure. If None, not saved.
        show: Whether to display plot immediately.
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

def visualize_loss_k(
    loss_dict: Dict[int, float],
    k_optimal: Optional[int] = None,
    smooth: bool = False,
    save: Optional[str] = None,
    show: bool = True,
) -> None:
    """
    Plot model selection criterion (loss vs K) for hyperparameter tuning.

    Used to visualize elbow or minimum point in K-selection procedures.

    Args:
        loss_dict: Mapping from K (number of components) to corresponding loss.
        k_optimal: Highlight optimal K with a red dot.
        smooth: If True, connects points with line; otherwise scatter only.
        save: Path to save figure.
        show: Whether to render plot.
    """

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

def plot_metrics(
    model: "AutoEncoderModel",
    yscale: Optional[str] = None,
    nrow: int = 1,
    save: Optional[str] = None,
    show: bool = True,
) -> None:
    """
    Plot training and validation metrics recorded during model fitting.

    Supports plotting multiple metrics (e.g., total_loss, recon_loss, kl_loss) in subplots.

    Args:
        model: Trained model with `.train_metrics` and optionally `.valid_metrics`.
        yscale: Scale for y-axis (e.g., 'log').
        nrow: Number of subplot rows.
        save: Figure save path.
        show: Whether to display the plot.
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

def plot_latent_space(
    model: "AutoEncoderModel",
    data: Optional[Union[np.ndarray, torch.Tensor]] = None,
    labels: Optional[np.ndarray] = None,
    reduction: Literal["tsne", "umap"] = "tsne",
    save: Optional[str] = None,
    show: bool = True,
    return_z: bool = False,
    use_cpu: bool = False,
    **embed_kwargs,
) -> Optional[np.ndarray]:
    """
    Visualize model's latent space using t-SNE or UMAP dimensionality reduction.

    Projects high-dimensional latent vectors into 2D for visualization.
    Points can be colored by labels if provided.

    Args:
        model (AutoEncoderModel): Trained model with encoder and latent representation.
        data: Optional input data to encode; if None, uses stored Z.
        labels: Optional labels for coloring (categorical or continuous).
        reduction: Dimensionality reduction method ('tsne' or 'umap').
        save: Save path for figure.
        show: Whether to display plot.
        return_z: If True, returns the 2D embedding coordinates.
        use_cpu: If True, performs encoding on CPU (useful for large models).
        **embed_kwargs: Additional arguments passed to the embedding algorithm.

    Returns:
        If return_z is True, returns the 2D coordinates (n_samples, 2). Otherwise None.
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

def plot_regplot(
    data,
    x: str,
    y: str,
    lowess: bool = False,
    save: Optional[str] = None,
    show: bool = True,
) -> None:
    """
    Create a regression plot (scatter + fit line) using Seaborn.

    Useful for visualizing relationships between two variables.

    Args:
        data: DataFrame or dict-like object with column access.
        x: Column name for x-axis variable.
        y: Column name for y-axis variable.
        lowess: If True, uses non-parametric LOWESS smoothing instead of linear fit.
        save: Figure save path.
        show: Whether to display plot.
    """
    
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