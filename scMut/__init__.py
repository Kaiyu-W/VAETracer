#!/usr/bin/env python

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
    
except Exception as e:
    print(f'Error when import scMut: {e}')
    print('Only funcitons from .data can be used')