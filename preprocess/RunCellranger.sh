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
Help() {
    cat >&2 << EOF
Usage: RunCellranger.sh [options]
Options:
  --SAMPLE_LIST,-i <string>   Set sample groups (space-separated string)
  --OUTPUT_DIR,-o <dir>       Set output directory
  --FASTQ_DIR,-f <dir>        Set fastq input directory
  --REF_DIR,-r <dir>          Set cellranger reference
  --REF_FASTA,-a <dir>        Set reference fasta
  --REF_GTF,-g <dir>          [optional] Set reference gtf (if REF_DIR not exist)
  --THREADS,-t <int>          [optional] Set total cpu cores (default: $(nproc))
  --TASK_THREADS,-T <int>     [optional] Set single task cores (default: 9)
  --WAIT_FOR_DATA,-w          [optional] Wait for input files to stabilize before running.
  --CELLRANGER[=<path>]       [optional] Set CELLRANGER executable path/alias. (default: cellranger)
  --CELLRANGER_OPTIONS[=<string>]
                              [optional] Set other CELLRANGER options.
  --help, -h                  Show this help message

Example:
  RunCellranger -i "LM0-RNA 100k-RNA" \\
                -f ./fastq_dir \\
                -r /path/to/refdata-gex-GRCh38-2020-A \\
                -o ./output_dir \\
                -w
EOF
}

SAMPLE_LIST=""
CELLRANGER=cellranger
CELLRANGER_OPTIONS=""
REF_DIR=""
REF_FASTA=""
REF_GTF=""
FASTQ_DIR=""
OUTPUT_DIR=""
THREADS=$(nproc)
TASK_THREADS=9
WAIT_FOR_DATA=0

TEMP=$(getopt \
  -o i:o:f:r:t:T:wh \
  -l SAMPLE_LIST:,OUTPUT_DIR:,FASTQ_DIR:,REF_DIR:,THREADS:,TASK_THREADS:,WAIT_FOR_DATA,help \
  -l CELLRANGER:,REF_FASTA:,REF_GTF:,CELLRANGER_OPTIONS: \
  -- "$@")
[ $? -ne 0 ] && { echo "Error in command line arguments" >&2; exit 1; }
eval set -- "$TEMP"
while true; do
    case "$1" in
        --CELLRANGER ) CELLRANGER="$2"; shift 2 ;;
        --CELLRANGER_OPTIONS ) CELLRANGER_OPTIONS="$2"; shift 2 ;;
        --SAMPLE_LIST | -i ) SAMPLE_LIST="$2"; shift 2 ;;
        --OUTPUT_DIR | -o ) OUTPUT_DIR="$2"; shift 2 ;;
        --FASTQ_DIR | -f ) FASTQ_DIR="$2"; shift 2 ;;
        --REF_DIR | -r ) REF_DIR="$2"; shift 2 ;;
        --REF_FASTA | -a ) REF_FASTA="$2"; shift 2 ;;
        --REF_GTF | -g ) REF_GTF="$2"; shift 2 ;;
        --THREADS | -t ) THREADS="$2"; shift 2 ;;
        --TASK_THREADS | -T ) TASK_THREADS="$2"; shift 2 ;;
        --WAIT_FOR_DATA | -w ) WAIT_FOR_DATA=1; shift ;;
        --help | -h ) Help; exit 0 ;;
        -- ) shift; break ;;
        * ) Help; exit 1 ;;
    esac
done

CmdCheck $CELLRANGER
CmdCheck pigz
args_char="SAMPLE_LIST OUTPUT_DIR FASTQ_DIR REF_DIR REF_FASTA"
args_empty=""
for arg in $args_char; do [ -z "$(CharGetVar $arg)" ] && args_empty="${args_empty} --$arg"; done
[ -n "$args_empty" ] && Help && echoError "You must specify$args_empty"
(( TASK_THREADS >= THREADS )) && TASK_THREADS=$THREADS

DirCheck "$FASTQ_DIR"
FileCheck "$REF_FASTA"
REF_FASTA=$(FileAbspath $REF_FASTA)
REF_GTF=$(FileAbspath $REF_GTF)
REF_DIR=$(DirAbspath $REF_DIR)
FASTQ_DIR=$(DirAbspath $FASTQ_DIR)
OUTPUT_DIR=$(DirAbspath $OUTPUT_DIR)
[ -d "$OUTPUT_DIR" ] || mkdir -p $OUTPUT_DIR


# cellranger index
if [ ! -d "$REF_DIR" ] || [ -z "$(ls -A "$REF_DIR")" ]; then
    FileCheck "$REF_GTF"
    mkdir -p $REF_DIR
    REF_FASTA_TMP=$OUTPUT_DIR/ref_tmp.fa
    REF_GTF_TMP=$OUTPUT_DIR/ref_tmp.gtf
    REF_DIR_NAME=$(basename $REF_DIR)

    [[ $REF_FASTA =~ .gz$ ]] && pigz -dcp $THREADS $REF_FASTA > $REF_FASTA_TMP || ln -s $REF_FASTA $REF_FASTA_TMP
    [[ $REF_GTF =~ .gz$ ]] && pigz -dcp $THREADS $REF_GTF > $REF_GTF_TMP || ln -s $REF_GTF $REF_GTF_TMP

    echoStep "Generate genome for cellranger" cellranger
    cd $REF_DIR/..
    
    [ -d "$REF_DIR_NAME" ] && rmdir "$REF_DIR_NAME" # dir shouldn't exist for cellranger mkref
    cellranger mkref \
        --genome="$REF_DIR_NAME" \
        --fasta="$REF_FASTA_TMP" \
        --genes="$REF_GTF_TMP" \
        --nthreads=$THREADS 1> "${REF_DIR}.mkref.log" || exit 1

    [ -s $REF_FASTA_TMP ] && rm $REF_FASTA_TMP
    [ -s $REF_GTF_TMP ] && rm $REF_GTF_TMP
    cd - > /dev/null
fi
[ -d "$REF_DIR" ] || exit 1

# cellranger count
job_cores=0
trap 'kill $(jobs -p) 2>/dev/null' EXIT
cd $OUTPUT_DIR
for sample in $SAMPLE_LIST; do
    if [[ -d "${sample}/outs/" && -s "${sample}/outs/possorted_genome_bam.bam" ]]; then
        echoStep "Skipping $sample, for BAM's output already exists." cellranger
        continue
    fi

    (
        echoStep "Running cellranger for $sample" cellranger
        {
            fq1="$FASTQ_DIR/${sample}_S1_L001_R1_001.fastq.gz"
            fq2="$FASTQ_DIR/${sample}_S1_L001_R2_001.fastq.gz"
            if [[ $WAIT_FOR_DATA -eq 1 ]]; then
                for fq in "$fq1" "$fq2"; do
                    echoStep "Waiting for $fq to be stable..." cellranger
                    until check_file_stable "$fq"; do
                        echoStep "Still waiting for $fq to be stable..." cellranger >&2
                    done
                    echoStep "Got stable $fq." cellranger
                done
            fi
            for fq in "$fq1" "$fq2"; do
                [ -s "$fq" ] || echoError "FASTQ file $fq not found! "
            done

            $CELLRANGER count \
                --id="$sample" \
                --transcriptome="$REF_DIR" \
                --fastqs="$FASTQ_DIR" \
                --sample="$sample" \
                --localcores="$TASK_THREADS" $CELLRANGER_OPTIONS
        } 1> "${sample}.count.log"

        echoStep "$sample over" cellranger
    ) &


    ((job_cores+=TASK_THREADS))
    while ((job_cores + TASK_THREADS > THREADS)); do
        wait -n
        ((job_cores-=TASK_THREADS))
    done
done
wait

trap - EXIT
echoStep "All cellranger jobs finished." cellranger