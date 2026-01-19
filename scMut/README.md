# scMut

`scMut` (single-cell Mutation Parser) decomposes the 2D mutation profile **M** into two biologically interpretable components:

- **N**: Cell generation index (lineage time)
- **P**: Site-specific mutation rate (mutation bias)

It consists of three submodules:

- `NMF`: generative Non-negative Matrix Factorization, for initial and best decomposition (gNMF in paper)

- `VAE`: Variational Autoencoder, with two operational modes:
  <pre>
  ● mode1-`np`: 
      ▪ Infers latent representation Z via encoder:        Z = encoder1(M)
      ▪ Encodes Z -> N through a learned transformation:   N = encoder2(Z)
      ▪ Learns P as site-specific parameters:              P = P_site
      ▪ Reconstructs M by combining N and P:               Mhat = f(N, P)

  ● mode2-`xhat`: 
      ▪ Uses standard encoder-latent-decoder structure:    Z = encoder(M)
      ▪ Reconstructs mutation matrix directly:             Mhat = decoder(Z)
  </pre>

- `FT`: Fine-tuning module, for post-hoc refinement of N and P estimates


## Python API
```python
# Add ../scMut to sys.path to enable direct import in the console.
import scMut

# or

from scMut import *

# here are some core function/class:

# 1. Simulate synthetic data
from scMut import simulate_data, simulate_lineage_data, simulate_lineage_data_segment

# 2. Core model class
from scMut import MutModel

# 3. Save model output to AnnData
from scMut import save_model_to_adata
# Packages inferred N, P, Z, etc. into an AnnData object for downstream analysis (e.g., UMAP, integration).

# 4. Test pipeline
from scMut.test import run_pipe

# 5. Save model for input of MutTracer
from scMut import extract_latent_mu    # save z_m (actually mu of z)
from scMut import save_model_to_pickle # save the whole MutModel

# Use help(func) in Python to view detailed documentation for each function.
```


## Setup environment

- Please set up the environment according to the dependency list in `requirements.txt`.

- Using `pip install .` (via `pyproject.toml`) installs this directory into the system environment, allowing direct import without configuring `sys.path`; however, be sure to prepare the correct environment beforehand, as automatic installation via pip may lead to unexpected issues if dependencies conflict or incompatible versions are installed.


