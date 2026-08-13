FROM centos:7

# ============================================================
# CentOS 7 EOL vault mirror
# ============================================================
RUN sed -i 's/mirrorlist/#mirrorlist/g' /etc/yum.repos.d/CentOS-*.repo \
    && sed -i 's|#baseurl=http://mirror.centos.org|baseurl=https://mirrors.aliyun.com/centos-vault|g' /etc/yum.repos.d/CentOS-*.repo \
    && yum clean all \
    && yum makecache

# ============================================================
# System dependencies
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
# Create vaetracer_scvi environment
# Python 3.11 + PyTorch 2.5.1 + CUDA 12.4 + scvi-tools
# conda sysroot provides glibc >= 2.28 automatically
# ============================================================
RUN CONDA_OVERRIDE_CUDA="12.4" mamba create -n vaetracer_scvi -y \
        -c conda-forge -c pytorch -c nvidia -c bioconda \
        'python=3.11' uv gcc gxx \
        pytorch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 pytorch-cuda=12.4 \
        scvi-tools==1.4.2 "jaxlib==0.4.34=cuda120*" jax==0.4.35 \
        scanpy==1.11.5 scikit-misc==0.5.2 pyarrow \
        ipykernel ipywidgets jupyter \
        seaborn umap-learn python-igraph leidenalg \
        tqdm \
    && mamba clean -afy
ENV PATH="/opt/conda/envs/vaetracer_scvi/bin:$PATH"

# ============================================================
# Install MutTracer package from source
# ============================================================
WORKDIR /app
COPY scMut/ /app/scMut/
COPY MutTracer/ /app/MutTracer/
COPY demo/ /app/demo/
RUN conda run -n vaetracer_scvi pip install /app/scMut/
RUN conda run -n vaetracer_scvi pip install /app/MutTracer/ --no-deps # --no-deps avoids pip to install scikit-misc manually
# ENV PATH="/app:$PATH"
# ENV PYTHONPATH="/app"

# ============================================================
# Entry point
# ============================================================
SHELL ["conda", "run", "-n", "vaetracer_scvi", "/bin/bash", "-c"]
ENTRYPOINT ["conda", "run", "-n", "vaetracer_scvi"]
CMD ["bash"]