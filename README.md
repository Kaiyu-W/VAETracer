# VAETracer
Mutation-Guided Lineage Reconstruction and Generational State Inference from scRNA-seq

VAETracer is a computational framework that integrates somatic mutation profiles with single-cell RNA equencing (scRNA-seq) data to reconstruct lineage relationships and infer generational dynamics. It consists of three modular components: `preprocess`, `scMut`, and `MutTracer`, implementing end-to-end analysis from raw sequencing data to lineage-aware expression modeling.


## 1. Components of VAETracer

### 1) preprocess: Data Preprocessing Pipeline

The preprocessing module converts raw sequencing data into structured mutation profiles for downstream analysis. 

#### Workflow
```bash
# (optional) Convert SRA to FASTQ
bash SRAtoFastq.sh

# Fastqs to generate gene expression matrix (GEX) and BAM
bash RunCellranger.sh 

# Align scRNA-seq maps using STAR
bash RunSTAR.sh

# Call variants using GATK best practices
bash RunGATK.sh

# Extract allele frequency (AF) matrix from VCF and single-cell BAM
python GetAF.py

# All scripts support --help for detailed usage instructions.
```

### 2) scMut: Mutation Matrix Decomposition

The `scMut` module decomposes the 2D mutation profile **M** into two biologically interpretable components:

- **N**: Cell generation index (lineage time)
- **P**: Site-specific mutation rate (mutation bias)

It consists of three submodules:

- `NMF`: Non-negative Matrix Factorization, for initial and best decomposition
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

#### Python API
```python
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

### 4) tree_util: Lineage tree utilities

Provides utility functions for lineage tree processing and format interoperability:

- Converts between `Newick` string format and `CassiopeiaTree` object
- Fixes common tree format issues to improve stability
- Other utilities: e.g. extracts tree linkage matrix


## 2. Environment Installation

Due to dependency conflicts among packages, we provide ``env_split.sh`` to automatically set up three isolated Conda environments:

1) **VAETracer_vcf**
    - For upstream data processing
    - Dependencies:
`sra-tools`, `samtools`, `vcftools`, `gatk`, `STAR`, `pysam`, `pyarrow` 
    - Recommended version consistency with scripts for compatibility

2) **VAETracer_vae**
    - For scMut modeling and decomposition
    - Dependencies:
`pytorch`, `pyarrow`, `scipy`, `scikit-learn`, `umap-learn,` `scanpy`

3) **VAETracer_sc**
    - For downstream analysis with scRNA-seq data
    - Dependencies:
`scanpy` (for single-cell analysis), `cassiopeia` (for lineage tree construction), and others

Alternatively, users may manually create the environments and configure the corresponding dependencies as needed.

## 3. Notes and Recommendations

- Ensure consistent software versions across pipeline steps to avoid compatibility issues.

For questions, please contact.
