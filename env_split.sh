#!/bin/bash

check_install_1() {
    local success=0
    echo "[check] cmd/package: "
    for cmd in fastq-dump.3.2.0 samtools vcftools gatk STAR; do
        which $cmd || { echo "$cmd not in this environment! "; success=1; }
    done
    for py_package in pysam pyarrow; do
        python -c "import $py_package; print(\"$py_package\", $py_package.__version__)" 2>/dev/null \
        || { echo "$py_package cannot be imported! "; success=1; }
    done
    return $success
}

check_install_2() {
    local success=0
    echo "[check] cmd/package: "
    for py_package in torch scanpy pyarrow; do
        python -c "import $py_package; print(\"$py_package\", $py_package.__version__); print(\"torch.cuda.is_available() =\",torch.cuda.is_available()) if \"$py_package\" == 'torch' else None" 2>/dev/null \
        || { echo "$py_package cannot be imported! "; success=1; }
    done
    return $success
}

check_install_2_scvi() {
    local success=0
    echo "[check] cmd/package: "
    for py_package in torch scvi scanpy pyarrow; do
        python -c "import $py_package; print(\"$py_package\", $py_package.__version__); print(\"torch.cuda.is_available() =\",torch.cuda.is_available()) if \"$py_package\" == 'torch' else None; print(\"scvi.settings.jax_preallocate_gpu_memory =\",scvi.settings.jax_preallocate_gpu_memory) if \"$py_package\" == 'scvi' else None" 2>/dev/null \
        || { echo "$py_package cannot be imported! "; success=1; }
    done
    return $success
}

check_install_3() {
    local success=0
    echo "[check] cmd/package: "
    for py_package in cassiopeia scanpy pyarrow; do
        python -c "import $py_package; print(\"$py_package\", $py_package.__version__);" 2>/dev/null \
        || { echo "$py_package cannot be imported! "; success=1; }
    done
    return $success
}


env_create_1() {
    local conda_bin=$1
    local mamba_avoid=$2
    local env_name=$3

    # initial conda
    eval "$($conda_bin shell.bash hook)"
    conda activate base # avoid already activate other env

    if $mamba_avoid; then
        run=conda
    else
        # initial mamba
        which mamba >/dev/null 2>&1 || conda install -n base -c conda-forge mamba --yes 1>&2 || return $?
        eval "$(mamba shell hook --shell bash)"
        run=mamba
    fi

    # create new environment
    $run create -n $env_name -c conda-forge \
        'python=3.7' gcc gxx pigz 'bash=5' --yes 1>&2 \
    && $run activate $env_name || return $?
    # gcc/gxx: In case the built-in compiler is too old
    # pigz: parallelly gzip/gunzip
    # bash=5: 'wait -n' needs bash>=5

    $run install -c pytorch -c conda-forge -c bioconda \
        'samtools' \
        'vcftools' \
        'gatk4==4.2.3.0' \
        'star==2.7.6a' \
        'sra-tools==3.2.0' \
        pandas pysam pyarrow pyranges \
        --yes 1>&2 || return $?

    # test
    check_install_1 && echo "Successfully create $env_name" || return $?
}

env_create_2() {
    local conda_bin=$1
    local mamba_avoid=$2
    local env_name=$3
    local pytorch_version=$4
    local cudatoolkit_version=$5


    local pytorch_1=$(echo $pytorch_version | cut -d'.' -f1)
    local pytorch_2=$(echo $pytorch_version | cut -d'.' -f2)
    local pytorch_3=$(echo $pytorch_version | cut -d'.' -f3)
    local torchvision_2=$((pytorch_2 + 1))
    local torchvision_version="0.${torchvision_2}.${pytorch_3}"
    local torchaudio_version="0.${pytorch_2}.${pytorch_3}"
    if [ "$pytorch_version" == 1.10.1 ]; then
        torchvision_version=0.11.2
        torchaudio_version=0.10.1
    fi
    [[ $pytorch_1 -ne 1 || $pytorch_2 -lt 10 || $pytorch_3 -lt 0 || $pytorch_3 -gt 1 ]] && {
        echo "Recommend to install pytorch with version 1.10.0~1.13.1, for package&environment has been validated for 1.10.1 and 1.12.0"
        echo "See https://pytorch.org/get-started/previous-versions/ to find the version combination of pytorch dependent-packages and modify this script manually!"
        return 1
    }


    # initial conda
    eval "$($conda_bin shell.bash hook)"
    conda activate base # avoid already activate other env

    if $mamba_avoid; then
        run=conda
    else
        # initial mamba
        which mamba >/dev/null 2>&1 || conda install -n base -c conda-forge mamba --yes 1>&2 || return $?
        eval "$(mamba shell hook --shell bash)"
        run=mamba
    fi

    # create new environment
    $run create -n $env_name -c conda-forge \
        'python=3.8' uv gcc gxx --yes 1>&2 \
    && $run activate $env_name || return $?
    # uv: accelerate pip
    # gcc/gxx: In case the built-in compiler is too old

    # packages for deep-learning framework (please use the correct pytorch version for your own system!)
    $run install -c pytorch -c conda-forge -c bioconda \
        numpy pandas matplotlib seaborn pyarrow typing_extensions \
        scipy scikit-learn umap-learn \
        "pytorch==$pytorch_version" \
        "torchvision==$torchvision_version" \
        "torchaudio==$torchaudio_version" \
        "cudatoolkit=$cudatoolkit_version" \
        "mkl==2024.0" \
        "scanpy<1.10" python-igraph leidenalg \
        ipykernel ipywidgets tqdm \
        --yes 1>&2 || return $?
    # scanpy 1.9.x supports python 3.8; scanpy 1.9.3 supports python 3.7

    # notebook kernal
    python -m ipykernel install --user --name $env_name --display-name "$env_name(python)" || return $?

    # test
    check_install_2 && echo "Successfully create $env_name" || return $?
}

env_create_2_scvi() {
    local conda_bin=$1
    local mamba_avoid=$2
    local env_name=$3

    # initial conda
    eval "$($conda_bin shell.bash hook)"
    conda activate base # avoid already activate other env

    if $mamba_avoid; then
        run=conda
    else
        # initial mamba
        which mamba >/dev/null 2>&1 || conda install -n base -c conda-forge mamba --yes 1>&2 || return $?
        eval "$(mamba shell hook --shell bash)"
        run=mamba
    fi

    # create new environment
    $run create -n $env_name \
        -c conda-forge -c pytorch -c nvidia -c bioconda \
        'python=3.11' uv gcc gxx \
        pytorch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 pytorch-cuda=12.4 \
        scvi-tools "jaxlib=*=*cuda*" jax \
        pyarrow scanpy ipykernel ipywidgets tqdm \
        --yes 1>&2 \
    && $run activate $env_name || return $?
    # uv: accelerate pip
    # gcc/gxx: In case the built-in compiler is too old
    # python 3.11 for scvi-tools only support 3.11-13 (till now 2026/1); if not install scvi-tools, python can be 3.7/3.8
    # PyTorch 2.5.1 is required for python 3.11. Due to changes in PyTorch's installation mechanism starting from version 2.x.x, the installation command is hardcoded here. For other versions, please refer to the official website.
    # scvi-tools depends on both JAX and PyTorch. While PyTorch's CUDA support is handled during installation, JAX requires careful checking of the system's CUDA driver compatibility (JAX needs CUDA ≥12.1). If GPU acceleration for JAX is not needed, avoid installing "jaxlib=*=*cuda*" and jax for scvi-tools.

    # notebook kernal
    python -m ipykernel install --user --name $env_name --display-name "$env_name(python)" || return $?

    # test
    check_install_2_scvi && echo "Successfully create $env_name" || return $?
}

env_create_3() {
    local conda_bin=$1
    local mamba_avoid=$2
    local env_name=$3
    local cassiopeia_github_folder=$4
    local python_version=$5 # vary with cassiopeia version

    # initial conda
    eval "$($conda_bin shell.bash hook)"
    conda activate base # avoid already activate other env

    if $mamba_avoid; then
        run=conda
    else
        # initial mamba
        which mamba >/dev/null 2>&1 || conda install -n base -c conda-forge mamba --yes 1>&2 || return $?
        eval "$(mamba shell hook --shell bash)"
        run=mamba
    fi

    # create new environment
    $run create -n $env_name -c conda-forge \
        "python=${python_version}" uv gcc gxx --yes 1>&2 \
    && $run activate $env_name || return $?
    # uv: accelerate pip
    # gcc/gxx: In case the built-in compiler is too old

    # packages for deep-learning framework (please use the correct pytorch version for your own system!)
    $run install -c conda-forge -c bioconda \
        "numpy==1.24.4" "numba==0.57.1" pandas \
        matplotlib seaborn pyarrow \
        scipy scikit-learn "statsmodels>=0.14.0" \
        statannotations adjustText \
        "scanpy==1.9.4" python-igraph leidenalg gseapy \
        "hits>=0.3.3" libxcrypt \
        ccphylo \
        vireosnp \
        ipykernel ipywidgets tqdm \
        --yes 1>&2 || return $?

    # packages for lineage-tree analysis
    # Use mirrors to accelerate
    export PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple 
    export UV_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple
    if [ -n "$cassiopeia_github_folder" ] && [ -d "$cassiopeia_github_folder" ]; then
        # download zipped folder from github
        cd $cassiopeia_github_folder
        uv pip install . 1>&2 || return $?
        cd - >/dev/null
    else
        uv pip install 'git+https://github.com/YosefLab/Cassiopeia@master#egg=cassiopeia-lineage' 1>&2 || return $?
    fi

    # packages for scRNA-seq analysis
    pip install scikit-misc "numpy==1.24.4" --no-build-isolation 1>&2 || return $?
    # here use --no-build-isolation, otherwise it will upgrade numpy to 2.x
    # here not use uv for uv will automatically uninstall and re-install all existing packages
    
    # notebook kernal
    python -m ipykernel install --user --name $env_name --display-name "$env_name(python)" || return $?

    # test
    check_install_3 && echo "Successfully create $env_name" || return $?

    # add ccphylo ini
    get_py_short() {
        local python_version="$1"
        local major minor
        IFS='.' read -r major minor _ <<< "$python_version"
        echo "$major.$minor"
    }
    ccphylo_path=$(dirname ${conda_bin})/../envs/${env_name}/bin/ccphylo
    cassiopeia_path=$(dirname ${conda_bin})/../envs/${env_name}/lib/python$(get_py_short $python_version)/site-packages/cassiopeia
    
    if [ -e $ccphylo_path ] && [ -d $cassiopeia_path ] && python -c "import cassiopeia, sys; from packaging.version import parse; sys.exit(0 if parse(cassiopeia.__version__) >= parse('2.1.0') else 1)"; then
        echo -e "[Paths]\nccphylo_path = ${ccphylo_path}" > ${cassiopeia_path}/config.ini
    else    
        echo "Error: Failed to configure ccphylo path for Cassiopeia (missing path or incompatible version)."
        # return 1
    fi

    # # if ScisTree2 is required
    # git clone https://github.com/yufengwudcs/ScisTree2.git
    # cd ScisTree2/; pip install .
}



conda_bin=/home/kaiyu/miniconda3/bin/conda
cassiopeia_github_folder='' # /mnt/c/Cassiopeia-master
pytorch_version=1.12.0
cudatoolkit_version=11.6 # 11.6 10.2
mamba_avoid=false

main() {
    local success=0

    env_create_1 $conda_bin $mamba_avoid vaetracer_vcf \
    || { echo 'Some error happened! Please install the package manually!'; success=$(( success+1 )); }

    env_create_2 $conda_bin $mamba_avoid vaetracer_vae $pytorch_version $cudatoolkit_version \
    || { echo 'Some error happened! Please install the package manually!'; success=$(( success+1 )); }

    env_create_2_scvi $conda_bin $mamba_avoid vaetracer_vae_scvi \
    || { echo 'Some error happened! Please install the package manually!'; success=$(( success+1 )); }

    env_create_3 $conda_bin $mamba_avoid vaetracer_sc "$cassiopeia_github_folder" "3.10" \
    || { echo 'Some error happened! Please install the package manually!'; success=$(( success+1 )); }

    return $success
}

if main; then
    echo 'Successfully create all environments!'
else
    echo "Fail to create $? environment(s)!"
    exit 1
fi