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
DeclareStringCheck() {
    input="$1"
    if [[ "$input" != declare\ -A\ SAMPLE_GROUPS=* ]]; then
        cat >&2 << 'EOF'
Invalid format for sample group! You should do like this:
declare -A SAMPLE_GROUPS=(
    ["LM0-RNA"]="SRR13045969 SRR13045970" # your sample1
    ["100k-RNA"]="SRR13045939 SRR13045940" # your sample2
)
SAMPLE_GROUPS_str=$(declare -p SAMPLE_GROUPS)
# then SAMPLE_GROUPS_str can be input:
split_sra_data -i "${SAMPLE_GROUPS_str}"

EOF
        exit 1
    fi
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
rename_file_prefix() {
    local sample="$1"
    local tag="$2"
    local use_10X_style="$3"
    local prefix_10X=${sample}_S1_L001_${tag}_001
    local prefix_normal=${sample}.${tag}
    [ "$use_10X_style" = "true" ] && echo $prefix_10X || echo $prefix_normal
}
Help() {
    cat >&2 << EOF
Usage: SRAtoFastq [options]

Options:
  --SAMPLE_GROUPS,-i <array string> 
                               Set sample groups using a 'declare -A' formatted string.
  --OUTPUT_DIR,-o <dir>        Set fastq-output directory.
  --DATA_DIR,-d <dir>          [optional] Set SRA-data directory. (default: same as input SRR)
  --THREADS,-t <int>           [optional] Total CPU cores available. (default: $(nproc))
  --TASK_THREADS,-T <int>      [optional] CPU cores allocated per sample group. (default: 9)
  --WAIT_FOR_DATA,-w           [optional] Wait for input files to stabilize before dumping.
  --SRATOOL[=<path>]           [optional] Set SRATOOL executable path/alias. (default: fastq-dump-orig.3.2.0)
  --VDB_VALIDATE[=<path>]      [optional] Set VDB_VALIDATE executable path/alias. (default: vdb-validate)
  --R1_TAG[=<int>]             [optional] Set R1 file tag. (default 1 for _1.fastq)
  --R2_TAG[=<int>]             [optional] Set R2 file tag. (default 2 for _2.fastq)
  --OUT_10X_STYLE <string>     [optional] Style to rename R1/R2 files. (default: true)
                                  <{sample}_S1_L001_{R12}_001.fastq.gz>(true) or <{sample}.{R12}.fastq.gz>(false)
                                  Only accepts "true" or "false" (lowercase)
  --help, -h                   Show this help message

Example:
  declare -A SAMPLE_GROUPS=(
      ["LM0-RNA"]="SRR13045969 SRR13045970"
      ["100k-RNA"]="SRR13045939 SRR13045940"
  )
  SAMPLE_GROUPS_str=\$(declare -p SAMPLE_GROUPS)
  SRAtoFastq -i "\${SAMPLE_GROUPS_str}" -o ./fastq_output -w
EOF
}

# set defaults
declare -A SAMPLE_GROUPS=()
SRATOOL=fastq-dump-orig.3.2.0
VDB_VALIDATE=vdb-validate
DATA_DIR=''
OUTPUT_DIR=''
THREADS=$(nproc)
TASK_THREADS=9
WAIT_FOR_DATA=0
R1_TAG=1
R2_TAG=2
OUT_10X_STYLE=true

# get options
TEMP=$(getopt \
  -o i:d:o:t:T:wh \
  -l SAMPLE_GROUPS:,DATA_DIR:,OUTPUT_DIR:,THREADS:,TASK_THREADS:,WAIT_FOR_DATA,help \
  -l SRATOOL:,VDB_VALIDATE:,R1_TAG:,R2_TAG:,OUT_10X_STYLE: \
  -- "$@")
  
[ $? -ne 0 ] && {
    echo "Error in command line arguments" >&2
    exit 1
}
eval set -- "$TEMP"
while true; do
  case "$1" in
    --SRATOOL ) SRATOOL="$2"; shift 2 ;;
    --VDB_VALIDATE ) VDB_VALIDATE="$2"; shift 2 ;;
    --R1_TAG ) R1_TAG="$2"; shift 2 ;;
    --R2_TAG ) R2_TAG="$2"; shift 2 ;;
    --SAMPLE_GROUPS | -i ) DeclareStringCheck "$2" && eval "$2"; shift 2 ;;
    --DATA_DIR | -d ) DATA_DIR="$2"; shift 2 ;;
    --OUTPUT_DIR | -o ) OUTPUT_DIR="$2"; shift 2 ;;
    --THREADS | -t ) THREADS="$2"; shift 2 ;;
    --TASK_THREADS | -T ) TASK_THREADS="$2"; shift 2 ;;
    --WAIT_FOR_DATA | -w ) WAIT_FOR_DATA=1; shift ;;
    --OUT_10X_STYLE ) OUT_10X_STYLE="$2"; shift 2 ;;
    --help | -h ) Help; exit 0 ;;
    -- ) shift; break ;;
    * ) Help ; exit 1 ;;
  esac
done

# check
cmds_char="pigz $SRATOOL $VDB_VALIDATE"
for cmd in $cmds_char; do CmdCheck $cmd; done
if [ -n "$DATA_DIR" ]; then
    DirCheck "$DATA_DIR"
    DATA_DIR=$(DirAbspath $DATA_DIR)
fi
args_empty=""
[[ ${#SAMPLE_GROUPS[@]} -eq 0 ]] && args_empty="${args_empty} --SAMPLE_GROUPS"
[ -z "$OUTPUT_DIR" ] && args_empty="${args_empty} --OUTPUT_DIR"
[ -n "$args_empty" ] && Help && echoError "You must specify$args_empty"
(( TASK_THREADS >= THREADS )) && TASK_THREADS=$THREADS
[ "$OUT_10X_STYLE" = 'true' ] || [ "$OUT_10X_STYLE" = 'false' ] || echoError "OUT_10X_STYLE must be true or false"

# configure
OUTPUT_DIR=$(DirAbspath $OUTPUT_DIR)
[ -d "$OUTPUT_DIR" ] || mkdir -p $OUTPUT_DIR

SrrValidate() {
    local file="$1"
    $VDB_VALIDATE $file &>/dev/null
    return $?
}

# SRA data split&merge
job_cores=0
trap 'kill $(jobs -p) 2>/dev/null' EXIT
for sample in "${!SAMPLE_GROUPS[@]}"; do

    (
        echoStep "Processing sample group: $sample" sra

        srr_list=${SAMPLE_GROUPS[$sample]}
        n_file=$(echo "$srr_list" | wc -w)
        n_thread=$(( TASK_THREADS / n_file )); (( n_thread < 1 )) && n_thread=1
        for srr in $srr_list; do
            R1=$OUTPUT_DIR/${srr}_${R1_TAG}.fastq
            R2=$OUTPUT_DIR/${srr}_${R2_TAG}.fastq
            if [[ -f "$R1" && -f "$R2" ]]; then # TO DO: check files' completeness
                echoStep "$srr already converted (all parts exist)." sra
                continue
            fi

            [ -n "$DATA_DIR" ] && input_srr="${DATA_DIR%/}/$srr" || input_srr=$srr
            srr_log="${OUTPUT_DIR}/${srr}.sra.log"
            > $srr_log

            {
                echoStep "${sample}: check srr..." sra
                {
                    SrrValidate "$input_srr" || {
                        if [[ $WAIT_FOR_DATA -eq 1 ]]; then
                            echoStep "Waiting for $input_srr to be stable..." sra

                            until check_file_stable "$input_srr"; do
                                echoStep "Still waiting for $input_srr to be stable..." sra >&2
                            done

                            echoStep "Got stable $input_srr." sra
                        fi
                        SrrValidate "$input_srr" || echoError "Database '${input_srr}' check failed! "
                    }
                } &>> $srr_log
                echoStep "${sample}: check srr over" sra

                echoStep "Converting $srr to FASTQ..." sra
                {
                    if [[ $SRATOOL =~ fasterq ]]; then
                        $SRATOOL --split-files --log-level 3 -e "$n_thread" -O "$OUTPUT_DIR" "${input_srr}"
                    else
                        $SRATOOL --split-files --log-level 3 -O "$OUTPUT_DIR" "${input_srr}" 
                    fi
                } &>> $srr_log

                echoStep "$srr over" sra
            } &
        done
        wait

        echoStep "Merging FASTQ files for $sample" sra
        n_thread=$(( TASK_THREADS / 2 )); (( n_thread < 1 )) && n_thread=1
        for d in $R1_TAG $R2_TAG; do
            [ $d -eq $R1_TAG ] && tag=R1
            [ $d -eq $R2_TAG ] && tag=R2
            {
                for srr in $srr_list; do 
                    file=$OUTPUT_DIR/${srr}_${d}.fastq
                    [ -s "$file" ] && cat $file
                done
            } | pigz -c -p $n_thread > $OUTPUT_DIR/$(rename_file_prefix $sample $tag $OUT_10X_STYLE).fastq.gz && {
                for srr in $srr_list; do
                    file=$OUTPUT_DIR/${srr}_${d}.fastq
                    [ -s "$file" ] && rm $file
                done
            } &
        done
        wait
        echoStep "$sample done." sra
    ) &

    ((job_cores+=TASK_THREADS))
    while ((job_cores + TASK_THREADS > THREADS)); do
        wait -n
        ((job_cores-=TASK_THREADS))
    done

done
wait

trap - EXIT
echoStep "All samples finished." sra

