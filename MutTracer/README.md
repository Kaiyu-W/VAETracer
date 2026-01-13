## MutTracer

MutTracer (Mutation-guided Temporal State Inference) is a unified framework for single-cell data that performs generation-aware joint modeling of mutational and transcriptional dynamics. It allows reconstruction of temporally resolved cellular lineages by predicting both ancestral and future cellular states while integrating mutational and transcriptional information.

MutTracer can order cells by generation (N) obtained from `scMut`, or by explicitly provided multi-timepoint annotations when available. It leverages both mutational and transcriptional latent representations to infer cellular states across generations.

---

## Framework Overview

### Input Data

- **Mutational latent representation (Zm)**: Obtained from `scMut` encoder, capturing lineage-specific features.  
- **Transcriptional latent representation (Zx)**: Extracted from the aligned expression matrix `X` using `scVI`.

### Feature Projection and Fusion

1. `Zm` is transformed via a Multi-Layer Perceptron (MLP) projection to emphasize lineage-related mutational features.  
2. `Zx` is mapped through a linear projection layer.  
3. The projected features are concatenated and fed into a bidirectional LSTM (BiLSTM) to model forward and backward temporal dependencies across generations.

### Dynamic Weighting and Modality Separation

1. The BiLSTM outputs a fused hidden representation.  
2. A learnable Weight Net generates dynamic modality-specific weights, separating the fused representation into mutational and transcriptional components.  
3. Each component is passed through a linear projection head to produce the predicted transcriptional and mutational states at each generation.

### Output

- Predicted `N` (generation index) and `X` (expression matrix) for each generation `t±1`.  
- Supports inference of both ancestral (`t−1`) and future (`t+1`) cellular states, enabling temporally resolved reconstruction of lineage dynamics.

---


# Setup environment

conda env create -f environment.yml
conda activate sc_vae_pytorch






