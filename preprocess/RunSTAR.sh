#!/usr/bin/env bash

TimeNow() {
    date +"%Y-%m-%d %H:%M:%S"
}
echoStep() {
    # $1 refers to the content 
    # $2 refers to the step
    # $3 refers to the line-head, if $3 exists

    echo -e "${3}[${2}] $(TimeNow) ${1}"
}
echoError() {
    # $1 refers to the error content
    echoStep "${1}\n" "error" "\n" >&2
    exit 1
}
CmdCheck() {
    # $1 refers to the shell command
    command -v "${1}" >/dev/null 2>&1 || echoError "No command ${1}! "
}
FileCheck() {
    # $1 refers to the file
    [ -f "$1" ] || echoError "No file ${1}! "
}
DirCheck() {
    local dir="$1"
    if [ -n "$dir" ]; then
        [ -d "$dir" ] || echoError "No directory ${1}! "
    fi
}
DirAbspath() {
    # $1 refers to the directory
    readlink -m "$1" | sed "s/\/$//"
}
FileAbspath() {
    # $1 refers to the file
    readlink -m "$1"
}
CharGetVar() {
    # $1 refers to the character
    eval echo \$$1
}
check_file_stable() {
    local file="$1"
    local n="${2:-3}"
    local interval="${3:-60}"
    local last_size=0
    local stable_count=0
    [ $n -le 1 ] && n=2
    for ((i=1; i<=n; i++)); do
        sleep "$interval"
        [[ -s "$file" ]] || return 1

        local size=$(stat -c "%s" "$file" 2>/dev/null) || return 1
        if [[ "$size" -eq "$last_size" && "$size" -ne 0 ]]; then
            ((stable_count++))
        else
            stable_count=1
            last_size="$size"
        fi
    done
    [[ $stable_count -ge $n ]]
}
check_file_growing() {
    local file="$1"
    local n="${2:-3}"
    local interval="${3:-60}"

    (( n < 2 )) && n=2

    [[ -s "$file" ]] || return 1
    local initial_size=$(stat -c "%s" "$file" 2>/dev/null) || return 1
    local current_size=$initial_size

    for ((i=1; i<n; i++)); do
        sleep "$interval"
        [[ -s "$file" ]] || return 1

        current_size=$(stat -c "%s" "$file" 2>/dev/null) || return 1
    done

    [[ $current_size -gt $initial_size ]]
}
Help() {
    cat >&2 << EOF
Usage: RunSTAR.sh [options]
Options:
  --SAMPLE_LIST,-i <string>   Set sample groups (space-separated string)
  --OUTPUT_DIR,-o <dir>       Set output directory
  --FASTQ_DIR,-f <dir>        Set fastq input directory
  --REF_DIR,-r <dir>          Set STAR genome directory
  --REF_FASTA,-a <dir>        Set reference fasta
  --REF_GTF,-g <dir>          [optional] Set reference gtf (if REF_DIR not exist)
  --THREADS,-t <int>          [optional] Set total cpu cores (default: $(nproc))
  --TASK_THREADS,-T <int>     [optional] Set single task cores (default: 9)
  --WAIT_FOR_DATA,-w          [optional] Wait for input files to be stable before running.
  --CLEAN,-c                  [optional] Clean the temporary result files.
  --USE_FASTQ <string>        [optional] Set input fastq file(s), can be set as 'R2', 'R1' or 'R1_R2' (default R2)
  --STAR[=<path>]             [optional] Set STAR executable path/alias. (default: STAR)
  --STAR_OPTIONS[=<string>]   [optional] Set other STAR options.
  --SAMTOOLS[=<path>]         [optional] Set SAMTOOLS executable path/alias. (default: samtools)
  --help, -h                  Show this help message
EOF
}

SAMPLE_LIST=""
STAR=STAR
STAR_OPTIONS=""
SAMTOOLS=samtools
REF_FASTA=""
REF_GTF=""
REF_DIR=""
FASTQ_DIR=""
OUTPUT_DIR=""
THREADS=$(nproc)
TASK_THREADS=9
WAIT_FOR_DATA=0
CLEAN=0
USE_FASTQ=R2


TEMP=$(getopt \
-o i:o:f:r:a:g:t:T:wch \
-l SAMPLE_LIST:,OUTPUT_DIR:,FASTQ_DIR:,REF_DIR:,REF_FASTA:,REF_GTF:,THREADS:,TASK_THREADS:,WAIT_FOR_DATA,CLEAN,help \
-l STAR:,SAMTOOLS:,STAR_OPTIONS: \
-l USE_FASTQ: \
-- "$@")
[ $? -ne 0 ] && { echo "Error in command line arguments" >&2; exit 1; }
eval set -- "$TEMP"
while true; do
    case "$1" in
        --STAR ) STAR="$2"; shift 2 ;;
        --STAR_OPTIONS ) STAR_OPTIONS="$2"; shift 2 ;;
        --SAMTOOLS ) SAMTOOLS="$2"; shift 2 ;;
        --SAMPLE_LIST | -i ) SAMPLE_LIST="$2"; shift 2 ;;
        --OUTPUT_DIR | -o ) OUTPUT_DIR="$2"; shift 2 ;;
        --FASTQ_DIR | -f ) FASTQ_DIR="$2"; shift 2 ;;
        --REF_DIR | -r ) REF_DIR="$2"; shift 2 ;;
        --REF_FASTA | -a ) REF_FASTA="$2"; shift 2 ;;
        --REF_GTF | -g ) REF_GTF="$2"; shift 2 ;;
        --THREADS | -t ) THREADS="$2"; shift 2 ;;
        --TASK_THREADS | -T ) TASK_THREADS="$2"; shift 2 ;;
        --WAIT_FOR_DATA | -w ) WAIT_FOR_DATA=1; shift ;;
        --CLEAN | -c ) CLEAN=1; shift ;;
        --USE_FASTQ ) USE_FASTQ="$2"; shift 2 ;; 
        --help | -h ) Help; exit 0 ;;
        -- ) shift; break ;;
        * ) Help; exit 1 ;;
    esac
done

cmds_char="pigz $STAR $SAMTOOLS"
for cmd in $cmds_char; do CmdCheck $cmd; done
args_char="SAMPLE_LIST OUTPUT_DIR FASTQ_DIR REF_DIR REF_FASTA"
args_empty=""
for arg in $args_char; do [ -z "$(CharGetVar $arg)" ] && args_empty="${args_empty} --$arg"; done
[ -n "$args_empty" ] && Help && echoError "You must specify$args_empty"
[[ "$USE_FASTQ" =~ ^(R1|R2|R1_R2)$ ]] || echoError "USE_FASTQ must be one of (R1|R2|R1_R2)"
(( TASK_THREADS >= THREADS )) && TASK_THREADS=$THREADS


DirCheck "$FASTQ_DIR"
# DirCheck "$REF_DIR"
FileCheck "$REF_FASTA"
# FileCheck "$REF_GTF"
OUTPUT_DIR=$(DirAbspath $OUTPUT_DIR)
FASTQ_DIR=$(DirAbspath $FASTQ_DIR)
REF_DIR=$(DirAbspath $REF_DIR)
REF_FASTA=$(FileAbspath $REF_FASTA)
REF_GTF=$(FileAbspath $REF_GTF)
OUTPUT_REF=$OUTPUT_DIR/REF_
[ -d "$OUTPUT_DIR" ] || mkdir -p $OUTPUT_DIR
[ -d "$OUTPUT_REF" ] || mkdir -p $OUTPUT_REF
cd $OUTPUT_DIR


echoStep "Generate&check REF INDEX" star

# REF Index
REF_FASTA_TMP=$OUTPUT_REF/$(basename $REF_FASTA | sed "s/.gz$//")
[[ $REF_FASTA_TMP =~ \.f(ast)?a$ ]] || echoError "REF_FASTA should be .f(ast)a(.gz)! "
[ ! -e $REF_FASTA_TMP ] && {
    [[ $REF_FASTA =~ \.gz$ ]] && pigz -dcp $THREADS $REF_FASTA > $REF_FASTA_TMP || ln -s $REF_FASTA $REF_FASTA_TMP
}
[ -s "${REF_FASTA_TMP}.fai" ] || $SAMTOOLS faidx "$REF_FASTA_TMP"


# STAR Index
if [ ! -d "$REF_DIR" ] || [ -z "$(ls -A "$REF_DIR")" ]; then
    [ -z "$REF_GTF" ] && echoError "REF_DIR not found or empty, and REF_GTF not provided! Please provide --REF_GTF."
    FileCheck "$REF_GTF"
    mkdir -p $REF_DIR

    REF_GTF_TMP=$OUTPUT_REF/$(basename $REF_GTF | sed "s/.gz$//")
    [[ $REF_GTF =~ .gz$ ]] && pigz -dcp $THREADS $REF_GTF > $REF_GTF_TMP || ln -s $REF_GTF $REF_GTF_TMP

    echoStep "Generate genome for STAR" star
    $STAR --runMode genomeGenerate \
         --genomeDir $REF_DIR \
         --runThreadN $THREADS \
         --genomeFastaFiles "$REF_FASTA_TMP" \
         --sjdbGTFfile "$REF_GTF_TMP" 1> "${REF_DIR}.STAR_genomeGenerate.log" || exit 1

    # [ -s $REF_FASTA_TMP ] && rm $REF_FASTA_TMP
    [[ $CLEAN -eq 1 ]] && [ -s $REF_GTF_TMP ] && rm $REF_GTF_TMP
fi
[ -d "$REF_DIR" ] || exit 1


# STAR
job_cores=0
trap 'kill $(jobs -p) 2>/dev/null' EXIT
for sample in $SAMPLE_LIST; do
    if [[ -s "${sample}.erc.g.vcf" && -s "${sample}.erc.g.vcf.idx" ]]; then
        echoStep "Skipping $sample, for VCF's output already exists." star
        continue
    fi

    bam_out="${sample}_star/${sample}Aligned.out.bam"
    final_log="${sample}_star/${sample}Log.final.out"

    if [[ -s "$bam_out" && -s "$final_log" ]]; then
        echoStep "Skipping $sample, for BAM's output already exists." star
        continue
    fi

    if [[ -s "$bam_out" ]] && [[ ! -s "$final_log" ]]; then
        echoStep "BAM exists but Log.final.out missing for $sample. Checking if actively being written..." star
        if check_file_growing "$bam_out"; then
            echoStep "Detected active write on $bam_out. Another job may be running. Skipping $sample." star
            continue
        else
            echoStep "BAM file is not growing. Re-run STAR for ${sample}." star
        fi
    fi

    (
        echoStep "Running STAR for $sample" star
        log_out=$(FileAbspath "${sample}.STAR_out.log")
        log_step=$(FileAbspath "${sample}.STAR_step.log")
        > $log_out

        {
            echoStep "${sample}: check fastqs..." star
            fq1="$FASTQ_DIR/${sample}_S1_L001_R1_001.fastq.gz"
            fq2="$FASTQ_DIR/${sample}_S1_L001_R2_001.fastq.gz"
            [ $USE_FASTQ == R1 ] && input_fq=$fq1
            [ $USE_FASTQ == R2 ] && input_fq=$fq2
            [ $USE_FASTQ == R1_R2 ] && input_fq="$fq1 $fq2"
            {
                if [[ $WAIT_FOR_DATA -eq 1 ]]; then
                    for fq in $input_fq; do
                        echoStep "Waiting for $fq to be stable..." star
                        until check_file_stable "$fq"; do
                            echoStep "Still waiting for $fq to be stable..." star >&2
                        done
                        echoStep "Got stable $fq." star
                    done
                fi
                for fq in $input_fq; do
                    [[ -s "$fq" ]] || echoError "FASTQ file $fq not found! "
                done
            } &>> $log_out
            echoStep "${sample}: check fastqs over" star
            
            [ -d "${sample}_star" ] || mkdir -p "${sample}_star"; cd "${sample}_star"
            echoStep "${sample}: STAR will output in $OUTPUT_DIR/${sample}_star/" star
            # n_thread=$(( TASK_THREADS / 2 )); (( n_thread < 1 )) && n_thread=1

            echoStep "${sample}: STAR..." star
            $STAR --runThreadN "$TASK_THREADS" \
                --genomeDir "$REF_DIR" \
                --readFilesIn "$input_fq" \
                --outFileNamePrefix "${sample}" \
                --readFilesCommand "gunzip -c" \
                --outSAMtype BAM Unsorted \
                --quantMode GeneCounts $STAR_OPTIONS &>> $log_out
                # default: only process R2, since R1 is CB-UMI which has no info
                # pigz -dcp $n_thread
                # why use 'gunzip' but not 'pigz' is that the speed of read-file is faster that process-sequences
            echoStep "${sample}: STAR over" star
            
        } &> $log_step

        echoStep "$sample over" star
    ) &

    ((job_cores+=TASK_THREADS))
    while ((job_cores + TASK_THREADS > THREADS)); do
        wait -n
        ((job_cores-=TASK_THREADS))
    done
done
wait

trap - EXIT
echoStep "All STAR jobs finished." star
