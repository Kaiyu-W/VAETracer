FROM centos:7

# ============================================================
# CentOS 7
# ============================================================
RUN sed -i 's/mirrorlist/#mirrorlist/g' /etc/yum.repos.d/CentOS-*.repo \
    && sed -i 's|#baseurl=http://mirror.centos.org|baseurl=https://mirrors.aliyun.com/centos-vault|g' /etc/yum.repos.d/CentOS-*.repo \
    && yum clean all \
    && yum makecache

# ============================================================
# System dependencies (required for pysam compilation and GATK Java runtime)
# ============================================================
RUN yum install -y \
        bzip2 ca-certificates curl tar gzip which \
        gcc gcc-c++ make zlib-devel bzip2-devel xz-devel \
        java-1.8.0-openjdk-headless \
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
# mamba create 
# ============================================================
RUN mamba create -n sc_preprocess -y \
        -c conda-forge -c bioconda \
        gcc gxx pigz 'bash=5' 'python=3.7' \
        'samtools>=1.4' \
        vcftools \
        'gatk4==4.2.3.0' \
        'star==2.7.6a' \
        'sra-tools==3.2.0' \
        pandas pysam pyarrow pyranges \
    && mamba clean -afy
ENV PATH="/opt/conda/envs/sc_preprocess/bin:$PATH"

# ============================================================
# copy scripts
# ============================================================
WORKDIR /app
COPY preprocess/ /app/preprocess/
COPY preprocess10X /app/preprocess10X
RUN chmod +x /app/preprocess10X
ENV PATH="/app:$PATH"

# ============================================================
# entry
# ============================================================
SHELL ["conda", "run", "-n", "sc_preprocess", "/bin/bash", "-c"]
ENTRYPOINT ["conda", "run", "-n", "sc_preprocess"]
CMD ["bash"]