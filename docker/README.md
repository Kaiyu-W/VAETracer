# VAETracer Docker

## Available Images

| Image | Description | Min System Requirement | Python | PyTorch | CUDA |
|-------|-------------|----------------------|--------|---------|------|
| vaetracer/preprocess:0.0.1 | Upstream preprocessing | CentOS 7 | 3.7 | - | - |
| vaetracer/scmut:0.1.0 | Mutation matrix decomposition | CentOS 7 / CUDA >= 10.2 | 3.7 | 1.12.0 | 10.2 |
| vaetracer/muttracer:0.1.0 | Lineage-aware expression modeling | CentOS 7 / CUDA >= 12.4 | 3.11 | 2.5.1 | 12.4 |

### Compatibility Notes

- preprocess and scMut images are based on CentOS 7 (glibc 2.17) for maximum HPC compatibility, supporting legacy clusters with older kernels and system libraries.
- MutTracer requires Python 3.11+, PyTorch 2.5.1, JAX with CUDA >= 12.4, and scvi-tools. These dependencies require glibc >= 2.28 and are incompatible with CentOS 7 / RHEL 7. A system such as Rocky Linux 8+, Ubuntu 20.04+, or RHEL 8+ is required. Note that while the host OS must meet this requirement, conda's sysroot mechanism allows the container to run on systems where the native glibc is older, as long as the NVIDIA driver supports CUDA >= 12.4.

## Prerequisites

- Docker >= 19.03
- For GPU support: NVIDIA driver >= 440.x and nvidia-container-toolkit installed
- For preprocess: Cell Ranger installed locally (not included due to licensing)

## Build

Run from the project root directory (VAETracer/):

```bash
    # preprocess
    docker build -t vaetracer/preprocess:0.0.1 -f ./docker/preprocess/Dockerfile .

    # scMut
    docker build -t vaetracer/scmut:0.1.0 -f ./docker/scMut/Dockerfile .

    # MutTracer
    docker build -t vaetracer/muttracer:0.1.0 -f ./docker/MutTracer/Dockerfile .
```

## Pull

```bash
    docker pull vaetracer/preprocess:0.0.1
    docker pull vaetracer/scmut:0.1.0
    docker pull vaetracer/muttracer:0.1.0
```

## Verify Installation

### preprocess

```bash
    docker run --rm vaetracer/preprocess:0.0.1 bash -c "\
        bash --version | head -1 && \
        python --version && \
        STAR --version && \
        samtools --version | head -1 && \
        vcftools --version && \
        gatk --version && \
        preprocess10X --help"

    docker run --rm vaetracer/preprocess:0.0.1 python -c "
import pysam, pyarrow, sys
print('pysam', pysam.__version__)
print('pyarrow', pyarrow.__version__)
print('python', sys.version.split()[0])
try:
    import pyranges
except ImportError:
    import importlib_metadata as m, importlib
    importlib.metadata = m
    sys.modules['importlib.metadata'] = m
    import pyranges
print('pyranges', pyranges.__version__)"
```

### scMut

```bash
    docker run --rm --gpus all vaetracer/scmut:0.1.0 python -c "import torch， scMut; print(f'PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}', scMut {scMut.__version__})"
    
    docker run --rm --gpus all vaetracer/scmut:0.1.0 python -c "from scMut import test; test.run_pipe(run_model_method='nmf+vae+ft', n_repeat=1, n_cells=100, n_sites=100, train_transpose=False, beta_pairs=[(1,32,None,None)], model_params=dict(num_epochs=100,num_epochs_nmf=100,lr=1e-3,beta_kl=0.001,beta_best=0.001), train_params=dict(patience=45), load_params=dict(batch_size=1000,num_workers=0), cpu_time=True); print('TEST PASSED')"
```

### MutTracer

```bash
    docker run --rm --gpus all vaetracer/muttracer:0.1.0 bash -c "python --version && python -c \"import torch, jax, scvi, scanpy, scMut, MutTracer; print('PyTorch', torch.__version__); print('JAX', jax.__version__); print('scvi-tools', scvi.__version__); print('Scanpy', scanpy.__version__); print('All imports OK')\""
```

## Usage

### preprocess

Cell Ranger is not included in the image due to licensing. Mount your local installation via -v:

```bash
    docker run --rm \
        -v /path/to/cellranger:/opt/cellranger:ro \
        -v /path/to/data:/app/data:ro \
        -v /path/to/output:/app/output \
        vaetracer/preprocess:0.0.1 \
        preprocess10X RunCellranger --cellranger /opt/cellranger/cellranger <subcommand> [options]
```

Show available subcommands:

```bash
    docker run --rm vaetracer/preprocess:0.0.1 preprocess10X --help
```

### scMut

scMut is a Python library. Three usage modes:

#### Run a script

```bash
    docker run --rm \
        -v /path/to/my_script.py:/app/my_script.py:ro \
        -v /path/to/data:/app/data:ro \
        -v /path/to/output:/app/output \
        vaetracer/scmut:0.1.0 \
        python /app/my_script.py
```

#### Interactive Python REPL

```bash
    docker run --rm -it vaetracer/scmut:0.1.0 python
```

#### Jupyter Notebook

```bash
    docker run --rm -it -p 8888:8888 \
        -v /path/to/notebooks:/app/notebooks \
        -v /path/to/data:/app/data:ro \
        vaetracer/scmut:0.1.0 \
        jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser --allow-root
```

For GPU, add `--gpus all` to any of the above commands.

### MutTracer

MutTracer is invoked via its command-line entry point. Mount your data and output directories:

#### Run MutTracer

```bash
    docker run --rm --gpus all \
        -v /path/to/data:/app/data:ro \
        -v /path/to/output:/app/output \
        vaetracer/muttracer:0.1.0 \
        python -m MutTracer.main <arguments>
```

#### Show help

```bash
    docker run --rm vaetracer/muttracer:0.1.0 python -m MutTracer.main --help
```

#### Interactive shell (for debugging or custom scripts)

```bash
    docker run --rm -it --gpus all vaetracer/muttracer:0.1.0 bash
```

Note: MutTracer requires GPU. Always add `--gpus all`.

## Export and Transfer

For offline deployment to HPC clusters without Docker Hub access:

```bash
    # Save all images
    docker save vaetracer/preprocess:0.0.1 vaetracer/scmut:0.1.0 vaetracer/muttracer:0.1.0 \
      | gzip > vaetracer-images.tar.gz

    # Transfer to cluster
    scp vaetracer-images.tar.gz user@cluster:~/

    # Load on cluster
    gunzip -c ~/vaetracer-images.tar.gz | docker load
```

## Notes

- scMut uses PyTorch 1.12.0 + CUDA 10.2; confirmed compatible with CentOS 7 kernel 3.10.
- MutTracer uses PyTorch 2.5.1 + CUDA 12.4 + JAX + scvi-tools; requires host NVIDIA driver supporting CUDA >= 12.4.
- See main project README for full API documentation, test procedures, and citation information.
- The Dockerfiles use Chinese mirror sources (mirrors.aliyun.com) for CentOS vault repositories to accelerate downloads within mainland China. Users outside China should remove or replace the mirror configuration in the RUN sed commands at the beginning of each Dockerfile, or substitute with a geographically appropriate mirror.
