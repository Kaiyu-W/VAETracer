# VAETracer
Mutation-Guided Lineage Reconstruction and Generational State Inference from scRNA-seq

VAETracer is a computational framework that integrates somatic mutation profiles with single-cell RNA sequencing (scRNA-seq) data to reconstruct lineage relationships and infer generational dynamics. It consists of three modular components: `preprocess`, `scMut`, and `MutTracer`, implementing end-to-end analysis from raw sequencing data to lineage-aware expression modeling.


## 1. Environment Installation

### 1. preprocess: 
For upstream data processing (FASTQ generation, alignment, variant calling), we need to install core tools including STAR, GATK, samtools, vcftools, and Python packages pysam, pyarrow, pyranges.
```bash
conda create -n sc_preprocess -c conda-forge -c bioconda \
    gcc gxx pigz 'bash=5' 'python=3.7' \
    'samtools' \
    'vcftools' \
    'gatk4==4.2.3.0' \
    'star==2.7.6a' \
    'sra-tools==3.2.0' \
    pandas pysam pyarrow pyranges

conda activate sc_preprocess
```

Note:
- To enable SRA file conversion, please also install `sra-tools`.
- `cellranger` is required and must be downloaded and installed manually from the 10x Genomics website.
- `bash=5` is required for enhanced functionality of the `wait` command used in workflow scripts.
- `gcc`/`gxx`: in case the system's built-in compiler is too old
- `pigz`: parallelly gzip/gunzip
- We recommend **maintaining version consistency** across installations to avoid bugs caused by changes in command-line arguments or usage patterns.

### 2. scMut: 
For running scMut, only basic scientific Python packages and `PyTorch` are required.
```bash
# Create and activate environment
conda create -n vaetracer python=3.11 -c conda-forge
conda activate vaetracer

# Install PyTorch (example with CUDA 12.4)
conda install -c pytorch -c nvidia \
    pytorch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 pytorch-cuda=12.4

# Install core dependencies
conda install -c conda-forge numpy pandas scipy scikit-learn matplotlib seaborn tqdm anndata
# Alternatively, you can install via scanpy (which includes most dependencies):
conda install -c conda-forge scanpy

# Alternatively, pip works
```

### 3. MutTracer: 
To use MutTracer, which **additionally depends on `scvi-tools`**, a more specific environment with **Python 3.11+ and CUDA 12+** is required. 
```bash
# 1. install PyTorch, and JAX (with jaxlib) for scvi-tools (choose the correct CUDA version based on your hardware)
# 2. install scvi-tools
# 3. install scanpy 
    
# example (pytorch==2.5.1 and use cuda rather than cpu)
conda create -n vaetracer \
    -c conda-forge -c pytorch -c nvidia -c bioconda \
    'python=3.11' \
    pytorch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 pytorch-cuda=12.4 \
    scvi-tools "jaxlib=*=*cuda*" jax \
    scanpy

# Alternatively, pip works
```

### Integrated Analysis in the Paper
In our study, we integrated multiple analysis workflows, including single-cell RNA-seq analysis (`scanpy`), lineage tree reconstruction (`Cassiopeia`) and others. To manage complex and conflicting dependencies across these tools, we provide the script `env_split.sh`, which automatically sets up isolated Conda environments:

1) **`vaetracer_vcf`**
    - For upstream data processing (`preprocess`)
    - Dependencies:
`sra-tools`, `samtools`, `vcftools`, `gatk`, `STAR`, `pysam`, `pyarrow` and `pyranges` 
    - `cellranger` must be downloaded and installed manually
    - Recommended version consistency with scripts for compatibility

2) **`vaetracer_vae`** or **`vaetracer_vae_scvi`**
    - For modeling and decomposition (`scMut` or/and `MutTracer`)
    - Dependencies:
`pytorch`, `pyarrow`, `scipy`, `scikit-learn`, `umap-learn,` `scanpy` and others (`scvi-tools`)
    - `pyarrow` for reading output from `preprocess`
    - `scvi-tools` is only required for `MutTracer` and is challenging to install due to numerous dependencies and strict version requirements. Therefore, users should set up the environment according to their specific needs.

3) **`vaetracer_sc`**
    - For downstream analysis with omics data
    - Dependencies:
`scanpy` (for single-cell analysis), `cassiopeia` (for lineage tree construction), and others

This separation ensures reproducibility and flexibility, allowing users to activate the appropriate environment for each task.

### Notes
- We provide `pyproject.toml` for both `scMut` and `MutTracer`. After setting up the environment, install the package locally by running `pip install .` in each project root.

- Due to strict version requirements (e.g., `scvi-tools` requires `JAX` with CUDA 12-13 and Python 3.11+), we recommend using a Python 3.11+ environment and CUDA 12+ for `MutTracer`.

- Some packages, like `scikit-misc` (a scanpy dependency) and `Cassiopeia`, are prone to version conflicts. If issues arise, please refer to specific installation guides or consider manual version pinning.

- In theory, all three modules can be integrated in an environment with a **newer** Python version, but maintaining separate environments offers greater **flexibility** and helps users meet specific requirements. Therefore, users are encouraged to manually create and configure environments according to their needs.


## 2. Quick Start (Coming Soon)
A minimal end-to-end example to help you get started with VAETracer will be provided here, including:
- Synthetic data simulation
- Running `scMut` for mutation matrix decomposition
- Using `MutTracer` for lineage-aware expression prediction
Stay tuned — this section will be updated in the next release.


## 3. Usage of VAETracer

### 1) preprocess: Data Preprocessing Pipeline
The preprocessing module converts raw sequencing data into structured mutation profiles for downstream analysis. 
It is designed for command-line usage:
```bash
# Add VAETracer/preprocess to $PATH, then run:

# (optional) Convert SRA to FASTQ
bash SRAtoFastq.sh --help

# Fastqs to generate gene expression matrix (GEX) and BAM
bash RunCellranger.sh --help

# Align scRNA-seq maps using STAR
bash RunSTAR.sh --help

# Call variants using GATK best practices
bash RunGATK.sh --help

# Extract allele frequency (AF) matrix from VCF and single-cell BAM
python GetAF.py --help

# Note: 
# All bash scripts include the WAIT_FOR_DATA parameter, allowing users to launch the scripts simultaneously even if the required input data is still being generated. 
# The scripts will automatically wait for the data to become available before proceeding.
```

Alternatively, for GetAF.py, you can use it programmatically:
```python
from preprocess import GetAF

# Define arguments manually, then call the main function
GetAF.main(args)
```

### 2) scMut: Mutation Matrix Decomposition
The `scMut` (single-cell Mutation Parser) module decomposes the 2D mutation profile **M** into two biologically interpretable components:
- **N**: Cell generation index (lineage time)
- **P**: Site-specific mutation rate (mutation bias)

It consists of three submodules:
- `NMF`: Non-negative Matrix Factorization, for initial and best decomposition (gNMF in paper)
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

`scMut` is implemented as a Python API, so it can be used as follows:
```python
# after `pip install /path/to/VAETracer/scMut` or from current directory:
import scMut

# Alternatively, if the package is not installed in PYTHONPATH:
import sys
sys.path.append('/path/to/VAETracer')
import scMut

# Note: Users can also install the package locally by running `pip install .` in the scMut root directory, which allows direct imports without modifying `sys.path`.

# API
# Simulate synthetic data
from scMut import simulate_data, simulate_lineage_data, simulate_lineage_data_segment

# Core model class
from scMut import MutModel

# Run test pipeline
from scMut.test import run_pipe

# Use help(func) in Python to view detailed documentation for each function.
```

### 3) MutTracer: Lineage-Aware Expression Dynamics Modeling
`MutTracer` integrates inferred lineage information with gene expression to predict temporal gene expression patterns along lineages. 

`MutTracer` is implemented in Python and provides both a command-line interface and an API for flexible usage:
```bash
# Add VAETracer to $PYTHONPATH (export PYTHONPATH="$PYTHONPATH:/path/to/VAETracer"), then run:
python -m MutTracer.main --help
# or
PYTHONPATH=/path/to/VAETracer python -m MutTracer.main --help
```

It can also be used programmatically:
```python
# after `pip install /path/to/VAETracer/MutTracer` or from current directory:
import MutTracer as mt

# Alternatively, if the package is not installed in PYTHONPATH:
import sys
sys.path.append('/path/to/VAETracer')
import MutTracer as mt
```

#### Command-Line Usage
```bash
python -m MutTracer.main \
  --scmut_model_path <path_to_scmut_trained_model.pkl> \
  --zmt_path <path_to_zm_dictionary.pt> \
  --zxt_path <path_to_zx_dictionary.pt> \
  --input_times <list_of_input_timepoints> \
  --predict_times <list_of_timepoints_to_predict> \
  --epochs <num_training_epochs> \
  --adata_path <path_to_original_h5ad_file> \
  --scvi_model_path <path_to_scvi_model.pkl> \
  --save_dir <output_directory> \
  [--auto_ancestor] \
  [--real_times_keep <list_of_times>] \
  [--pred_times_keep <list_of_times>]
```

#### Parameter Descriptions
- `--scmut_model_path`  
  Path to a pre-trained scMut model `.pkl` file. Used to initialize the predictor weights.

- `--zmt_path`  
  Path to the mutational latent representation dictionary (`Zm`) saved in `.pt` format. Typically generated by `scMut`.

- `--zxt_path`  
  Path to the transcriptional latent representation dictionary (`Zx`) saved in `.pt` format. Typically generated by `scVI` on normalized scRNA-seq data.

- `--input_times`  
  List of time points to use as **input** for training or prediction. Only data from these time points will be used as the model's known states.

- `--predict_times`  
  List of time points for which MutTracer will predict latent and transcriptional states.

- `--epochs`  
  Number of training epochs to run during model fitting. Default is `500`. Increasing epochs may improve convergence but increases runtime.

- `--adata_path`  
  Path to the original `.h5ad` single-cell AnnData object, used for downstream gene expression alignment and analysis.

- `--scvi_model_path`  
  Path to a pre-trained scVI model `.pkl` file, used to obtain transcriptional latent embeddings and reconstruct expression profiles.

- `--save_dir`  
  Directory to save training results, plots, and predictions. If the folder does not exist, it will be created.

- `--auto_ancestor` (optional flag)  
  Automatically selects the most compact time point as the **ancestral state** for prediction.

- `--real_times_keep` (optional)  
  Subset of real time points to include in filtered visualizations (e.g., t-SNE or N distribution plots).

- `--pred_times_keep` (optional)  
  Subset of predicted time points to include in filtered visualizations.

#### Example Usage
```bash
python -m MutTracer.main \
  --scmut_model_path ./model.pkl \
  --zmt_path ./z_mt.pt \
  --zxt_path ./z_xt.pt \
  --input_times 2 3 \
  --predict_times 1 \
  --epochs 2000 \
  --real_times_keep 2 3 \
  --pred_times_keep 1 \
  --auto_ancestor \
  --adata_path ./scRNAlistep.h5ad \
  --scvi_model_path ./scvi_model.pkl \
  --save_dir ./output
```

This command will train MutTracer using time points 2 and 3 as input, predict states for time point 1, automatically select an ancestral state, and save all results and plots to the specified save_dir.

### 4) tree_util: Lineage tree utilities
Provides utility functions for lineage tree processing and format interoperability:
- Converts between `Newick` string format and `CassiopeiaTree` object
- Fixes common tree format issues to improve stability
- Other utilities: e.g. extracts tree linkage matrix


## 4. Notes and Recommendations
- Due to the complexity of deep learning dependencies (especially `PyTorch`), we recommend installing `CUDA` first using the appropriate command for your system (CPU/GPU) before installing other packages.

- In theory, `PyTorch` maintains backward compatibility, so you can install a `PyTorch` version suitable for your hardware. Here, we have confirmed that `PyTorch=1.12.0` works as expected. If a newer version causes incompatibilities, please downgrade accordingly.

- `Scanpy` has specific Python version requirements, and automatically installed versions (by `conda`) often lead to dependency conflicts. The best approach is to check version compatibility and **manually** specify the appropriate version during installation (for example, `conda install 'scanpy<1.10'` for Python 3.8). The same principle applies to other critical packages (such as `Cassiopeia`) as well.

- The dependency versions listed here are valid as of **January 2026**. Future updates to `scvi-tools` may change its installation requirements; please refer to the official documentation for the latest guidance.

- All codes and scripts have been tested and verified on `Linux` systems (Ubuntu and Red-Hat distributions). Compatibility with other operating systems is not guaranteed.


## 5. For Publication (Under Preparation)
If you are using VAETracer in a research manuscript, please contact the authors for citation details.
A formal citation and BibTeX entry will be available upon publication in a peer-reviewed journal.


For questions, please contact.
