"""
scMut – Single-Cell Mutation Parser

A deep generative modeling framework for inferring cellular lineage 
and mutation rates from sparse, noisy single-cell DNA/RNA sequencing data.

Core Features:
- VAE-based joint N/P estimation (MutModel)
- Support for tree-constrained simulation
- Multi-stage training with denoising and fine-tuning
- Integration-ready output via AnnData export

Public API:
- Data Simulation: simulate_data, simulate_lineage_data, simulate_lineage_data_segment
- Model Classes: AEModel, scVIModel, MutModel
- Visualization: plot_metrics, plot_latent_space, ...
- Export: save_model_to_adata, ...
"""

import warnings

from .log import (
    setup_logging,
    add_file_handler,
    remove_file_handler,
    cleanup_logging,
    logger
)

from .data import (
    sample_by_beta,
    simulate_data,
    simulate_lineage_data,
    simulate_lineage_data_segment
)

__all__ = [
    'setup_logging',
    'add_file_handler',
    'remove_file_handler',
    'cleanup_logging',
    'logger',
    'sample_by_beta',
    'simulate_data',
    'simulate_lineage_data',
    'simulate_lineage_data_segment'
]

try:
    from .utils import (
        set_seed,
        visualize_loss,
        visualize_loss_k,
        plot_metrics,
        plot_latent_space,
        plot_regplot
    )

    from .baseVAE import AutoEncoderModel as AEModel
    from .scVI import scVIModel
    from .scMut import MutModel
    from . import test

    __all__.extend([
        'set_seed',
        'visualize_loss',
        'visualize_loss_k',
        'plot_metrics',
        'plot_latent_space',
        'plot_regplot',
        'AEModel',
        'scVIModel',
        'MutModel',
        'test'
    ])
    
except Exception as e:
    warnings.warn(
        f'Error when import scMut: {e}'
        'Only functions from .data can be used',
        ImportWarning
    )

try:
    from .export import (
        save_model_to_adata,
        save_model_to_pickle,
        load_model_from_pickle,
        extract_latent_mu
    )

    __all__.extend([
        'save_model_to_adata',
        'save_model_to_pickle',
        'load_model_from_pickle',
        'extract_latent_mu'
    ])
    
except Exception as e:
    warnings.warn(
        f'Error when import scMut: {e}'
        "Optional dependency '.export' not loaded. "
        "To enable AnnData export, install anndata: pip install anndata",
        ImportWarning
    )
