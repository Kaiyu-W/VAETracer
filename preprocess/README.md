# preprocess:

This suite of Bash and Python scripts provides a complete upstream pipeline for processing **10X Genomics single-cell RNA-seq data**, enabling both gene expression (GEX) profiling and mutation-aware analysis through allele frequency (AF) matrix generation.

## Overview

The pipeline integrates standard bioinformatics tools into a modular workflow:

*   **`SRAtoFastq`**: Downloads and converts SRA files to FASTQ format (optional)
*   **`RunCellranger.sh`**: Processes scRNA-seq data using CellRanger to generate GEX and BAM
*   **`RunSTAR.sh`**: Aligns reads using STAR
*   **`RunGATK.sh`**: Performs variant calling using GATK4 best practices
*   **`GetAF.py`**: Extracts cell-level allele frequencies from BAM/VCF

All scripts support **`--help`** for detailed usage.

## Workflow

Below is the complete upstream preprocessing pipeline for 10X single-cell RNA-seq data, from raw FASTQ to allele frequency (AF) matrix generation.
Simple example commands are provided below; replace paths and sample names as needed.

```bash
# ==============================================================================
# Step 0: (Optional) Convert SRA files to FASTQ
# ==============================================================================
bash SRAtoFastq.sh --help

# Define sample groups: map each output name to one or more SRR accessions
declare -A SAMPLE_GROUPS=(
    ["LM0-RNA"]="SRR13045969 SRR13045970"
    ["100k-RNA"]="SRR13045939 SRR13045940"
)
# Prepare input lists
SAMPLE_LIST="LM0-RNA,100k-RNA"
# or
SAMPLE_LIST=$(IFS=,; echo "${!SAMPLE_GROUPS[*]}")

# Run conversion
bash SRAtoFastq.sh \
    -i "$(declare -p SAMPLE_GROUPS)" \
    -o fastq_dir/ \
    -t $(nproc) \
    -w

# Notes:
#   - Outputs will follow 10X naming: {sample}_S1_L001_R[12]_001.fastq.gz
#   - Use `-w` to wait for stable input if downloading in parallel.
```

```bash
# ==============================================================================
# Step 1: Generate Gene Expression Matrix (GEX) and BAM using CellRanger
# ==============================================================================
bash RunCellranger.sh --help

# Process scRNA-seq data with pre-built reference
bash RunCellranger.sh \
    -i $SAMPLE_LIST \
    -f fastq_dir/ \
    -r /path/to/refdata-gex-GRCh38-2020-A \
    -o cr_out/ \
    -t $(nproc) \
    -w

# Notes:
#   - Reference can be downloaded from 10x Genomics website.
#   - Alternatively, use `-a <fasta>` and `-g <gtf>` to build reference automatically.
#   - Output BAM: cr_out/{sample}/outs/possorted_genome_bam.bam
```

```bash
# ==============================================================================
# Step 2: Align Reads Using STAR (for variant calling)
# ==============================================================================
bash RunSTAR.sh --help

# Align reads for mutation profiling
bash RunSTAR.sh \
    -i $SAMPLE_LIST \
    -f fastq_dir/ \
    -o vcf_out/ \
    -r star_ref/ \
    -a /path/to/refdata-gex-GRCh38-2020-A/fasta/genome.fa \
    -g /path/to/refdata-gex-GRCh38-2020-A/genes/genes.gtf \
    -t $(nproc) \
    -w

# Notes:
#   - Output: vcf_out/{sample}_star/Aligned.out.bam
#   - This BAM is used by downstream GATK for variant calling.
```

```bash
# ==============================================================================
# Step 3: Variant Calling Using GATK Best Practices
# ==============================================================================
bash RunGATK.sh --help

# Perform joint variant calling across all samples
bash RunGATK.sh \
    -i $SAMPLE_LIST \
    -o vcf_out/ \
    -a /path/to/refdata-gex-GRCh38-2020-A/fasta/genome.fa \
    -k /path/to/known.vcf.gz \
    --CHR_LIST $(echo chr{1..22} chrX chrY chrM | tr ' ' ',') \
    -m joint \
    -t $(nproc) \
    -w

# Notes:
#   - Make sure the output directory structure matches that of RunSTAR.
#   - --KNOWNs(-k) supports multiple files, separated by commas (e.g., file1.vcf.gz,file2.vcf.gz).
#   - For mouse: use chr{1..19},chrX,chrY,chrM; for human: use chr{1..22},chrX,chrY,chrM.
#   - For single-sample calling, set `-m single`.
#   - Joint mode outputs: vcf_out/merge09maf05.recode.vcf.gz
#   - Single mode outputs: vcf_out/{sample}.merge09maf05.recode.vcf.gz
```

```bash
# ==============================================================================
# Step 4: Extract Allele Frequency (AF) Matrix
# ==============================================================================
python3 GetAF.py --help

# Build BAM list: path,label pairs separated by semicolon
BAMs="$(for sample in ${SAMPLE_LIST//,/ }; do echo -n "cr_out/$sample/outs/possorted_genome_bam.bam,$sample;"; done)"

# Choose VCF input based on calling mode
VCFs="vcf_out/merge09maf05.recode.vcf.gz"  # for joint calling
# VCFs="$(for sample in ${SAMPLE_LIST//,/ }; do echo -n "vcf_out/${sample}.merge09maf05.recode.vcf.gz;"; done)"  # for single

# Run AF extraction
python3 GetAF.py \
    --bams "$BAMs" \
    --vcfs "$VCFs" \
    --outdir af_out/ \
    --gtf /path/to/refdata-gex-GRCh38-2020-A/genes/genes.gtf \
    --processes $(nproc) \

# Notes:
#   - Use `--method`, `--chunk-size`, `--segment-size` to control multi-core parallelism.
#   - High parallelism improves speed but increases memory usage.
```


## Note:

### 1. Input Format for `SAMPLE_LIST`

*   The `--SAMPLE_LIST` parameter used in `RunSTAR.sh`, `RunGATK.sh`, and `RunCellranger.sh` expects a **comma-separated string** (e.g., `Sample1,Sample2,Sample3`).
*   **Important:** Sample names must **not contain commas or spaces** to ensure correct parsing.

### 2. FASTQ File Naming Convention

*   This pipeline assumes FASTQ files follow the standard **10X Genomics naming convention**:
    *   `{sample}_S1_L001_R1_001.fastq.gz`
    *   `{sample}_S1_L001_R2_001.fastq.gz`

#### Multi-Lane (Multi-Batch) Data Handling:

The treatment of multiple sequencing batches differs across scripts due to tool capabilities and design goals:

- **RunCellranger.sh**:
   CellRanger natively supports multi-lane aggregation by automatically grouping FASTQs with the same sample name (`{sample}`). Therefore, you can directly place multi-batch data in `FASTQ_DIR`, and CellRanger will merge them during processing.
   - Supported out-of-the-box.

- **RunSTAR.sh**:
   Although STAR itself supports multiple input files via `--readFilesIn`, this script is designed for simplicity and consistency within the pipeline. It expects a single pair of FASTQs per sample.
   - Recommended workaround: Manually merge R1 and/or R2 files across lanes before running the script.
   - Advanced option (not recommended): You may modify the script to pass multiple FASTQ paths to `--readFilesIn`. This avoids the time and memory cost of pre-merging large FASTQ files but sacrifices portability and increases complexity.

*Best Practice*: For end-to-end reproducibility, we recommend preprocessing all FASTQs into a single batch format prior to using this pipeline, unless you are only using `RunCellranger.sh`. This ensures consistent behavior across all modules.

### 3. Input Format for `GetAF.py`

*   The `GetAF.py` script uses custom delimiters for its parameters:
    *   Use **semicolon (`;`)** to separate different `(BAM file, label)` pairs.
    *   Use **comma (`,`)** to separate the BAM filename and its label within each pair.
    *   Example: `--bams path/to/bam1,label1;path/to/bam2,label2`

### 4. Compatibility Note

*   These scripts utilize GNU-specific extensions for command-line tools.
*   **It is recommended to run these scripts in a GNU environment (e.g., standard Linux distributions like Ubuntu).**
*   They may not function correctly on systems with non-GNU core utilities (e.g., macOS without GNU coreutils installed via Homebrew, or Alpine Linux without explicit installation of GNU packages).

### 5. For Complex `SRAtoFastq` Inputs

*   The `SRAtoFastq` script accepts a complex `declare -A` associative array string via `--SAMPLE_GROUPS`.
*   If your input list is large or complex, consider generating this array programmatically from a table (e.g., using a `Python` or `awk` script) before passing it to the command.

### 6. others

__environment__:
- Required software:
  - `cellranger`: must be manually installed.
  - `gatk`, `STAR`, `samtools`, `vcftools`, `pigz`, `sra-tools`
- Python dependencies: `pysam`, `pyarrow`, `pyranges`
- Shell requirement: Bash 5+ (for enhanced `wait` functionality)

__database__:
mouse known sites can be downloaded from https://ftp.ncbi.nih.gov/snp/organisms/archive/mouse_10090/VCF/
