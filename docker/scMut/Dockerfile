FROM centos:7

# ============================================================
# CentOS 7 EOL vault mirror
# ============================================================
RUN sed -i 's/mirrorlist/#mirrorlist/g' /etc/yum.repos.d/CentOS-*.repo \
    && sed -i 's|#baseurl=http://mirror.centos.org|baseurl=https://mirrors.aliyun.com/centos-vault|g' /etc/yum.repos.d/CentOS-*.repo \
    && yum clean all \
    && yum makecache

# ============================================================
# System dependencies (required for gcc/gxx compilation)
# ============================================================
RUN yum install -y \
        bzip2 ca-certificates curl tar gzip which \
        gcc gcc-c++ make zlib-devel bzip2-devel xz-devel \
    && yum clean all && rm -rf /var/cache/yum

# ============================================================
# Miniconda + mamba
# ============================================================
ENV CONDA_DIR=/opt/conda
RUN curl -fsSL https://repo.anaconda.com/miniconda/Miniconda3-py38_4.12.0-Linux-x86_64.sh \
        -o /tmp/miniconda.sh \
    && bash /tmp/miniconda.sh -b -p $CONDA_DIR \
    && rm /tmp/miniconda.sh \
    && $CONDA_DIR/bin/conda install -y -n base -c conda-forge mamba \
    && $CONDA_DIR/bin/mamba clean -afy
ENV PATH="$CONDA_DIR/bin:$PATH"

# ============================================================
# Create vaetracer environment
# PyTorch 1.12.0 + CUDA 10.2 + Python 3.7
# ============================================================
RUN mamba create -n vaetracer -y -c conda-forge \
        'python=3.7' uv gcc gxx \
    && mamba install -y -n vaetracer \
        -c pytorch -c conda-forge -c bioconda \
        numpy pandas matplotlib seaborn pyarrow typing_extensions \
        scipy scikit-learn umap-learn \
        "pytorch==1.12.0" \
        "torchvision==0.13.0" \
        "torchaudio==0.12.0" \
        "cudatoolkit=10.2" \
        "mkl==2024.0" \
        "scanpy<=1.9.3" python-igraph leidenalg \
        ipykernel ipywidgets jupyter \
        tqdm \
    && mamba clean -afy
ENV PATH="/opt/conda/envs/vaetracer/bin:$PATH"

# ============================================================
# Install scMut package from source
# ============================================================
WORKDIR /app
COPY scMut/ /app/scMut/
COPY demo/ /app/demo/
RUN conda run -n vaetracer pip install /app/scMut/
# ENV PATH="/app:$PATH"

# ============================================================
# Entry point
# ============================================================
SHELL ["conda", "run", "-n", "vaetracer", "/bin/bash", "-c"]
ENTRYPOINT ["conda", "run", "-n", "vaetracer"]
CMD ["bash"]