## preprocess : Data Preprocessing Pipeline

The preprocessing module converts raw sequencing data into structured mutation profiles for downstream analysis. 

## Workflow
```bash
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

# All scripts support --help for detailed usage instructions.
```

## Note:

__environment__:
- `cellranger` need to download and install by user!
- other dependency: `sra-tools`, `samtools`, `vcftools`, `gatk`, `STAR`, `pysam`, `pyarrow` and `pyranges` 
- `bash=5` for `wait`'s enhanced functionality

__database__:
mouse known sites can be downloaded from https://ftp.ncbi.nih.gov/snp/organisms/archive/mouse_10090/VCF/