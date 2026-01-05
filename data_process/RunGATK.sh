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
build_java_options() {
    local default_opts="$1"
    local user_opts="$2"

    [[ "$user_opts" == *"-Djava.util.concurrent.ForkJoinPool.common.parallelism="* ]] && \
    echoStep "User provided '-Djava.util.concurrent.ForkJoinPool.common.parallelism=...' option. Thread settings in default options will be overridden." WARNING

    if [[ -z "$user_opts" ]]; then
        echo "$default_opts"
    else
        echo "$default_opts $user_opts"
    fi
}
Help() {
    cat >&2 << EOF
Usage: RunGATK.sh [options]
Options:
  --SAMPLE_LIST,-i <string>   Set sample groups (space-separated string)
  --OUTPUT_DIR,-o <dir>       Set output directory
  --REF_FASTA,-a <dir>        Set reference fasta
  --KNOWNs,-k <vcf>           Set known sites vcf (space-separated string if multiple ones)
  --CALL_MODE,-m <string>     [optional] Set calling mode: 'joint' (default) or 'single'.
  --THREADS,-t <int>          [optional] Set total cpu cores (default: $(nproc))
  --TASK_THREADS,-T <int>     [optional] Set single task cores (default: 9)
  --WAIT_FOR_DATA,-w          [optional] Wait for input files to be stable before running.
  --CLEAN,-c                  [optional] Clean the temporary result files.
  --Xmx <int+g>               [optional] Max memory for JVM. (default: 32g)
  --GATK4_JAVA_OPTS <string>  [optional] Options for JVM-GATK4. (default: set parallelism to maximum available CPU threads)
  --CHR_LIST <string>         [optional] Set chromosome groups (space-separated string) (default: chr1-19+X/Y/M for mouse)
  --GATK4[=<path>]            [optional] Set GATK4 executable path/alias. (default: gatk)
  --SAMTOOLS[=<path>]         [optional] Set SAMTOOLS executable path/alias. (default: samtools)
  --VCFTOOLS[=<path>]         [optional] Set VCFTOOLS executable path/alias. (default: vcftools)
  --help, -h                  Show this help message
EOF
}

SAMPLE_LIST=""
CHR_LIST=$(eval echo "chr{1..19} chrX chrY chrM")
GATK4=gatk
SAMTOOLS=samtools
VCFTOOLS=vcftools
REF_FASTA=""
KNOWNs=""
OUTPUT_DIR=""
THREADS=$(nproc)
TASK_THREADS=9
WAIT_FOR_DATA=0
CLEAN=0
Xmx=32g
CALL_MODE="joint"  # default: joint calling


TEMP=$(getopt \
-o i:o:a:k:m:t:T:wch \
-l SAMPLE_LIST:,CHR_LIST:,OUTPUT_DIR:,REF_FASTA:,KNOWNs:,CALL_MODE:,THREADS:,TASK_THREADS:,WAIT_FOR_DATA,CLEAN,help \
-l GATK4:,SAMTOOLS:,VCFTOOLS: \
-l Xmx:,GATK4_JAVA_OPTS: \
-- "$@")
[ $? -ne 0 ] && { echo "Error in command line arguments" >&2; exit 1; }
eval set -- "$TEMP"
while true; do
    case "$1" in
        --GATK4 ) GATK4="$2"; shift 2 ;;
        --SAMTOOLS ) SAMTOOLS="$2"; shift 2 ;;
        --VCFTOOLS ) VCFTOOLS="$2"; shift 2 ;;
        --SAMPLE_LIST | -i ) SAMPLE_LIST="$2"; shift 2 ;;
        --OUTPUT_DIR | -o ) OUTPUT_DIR="$2"; shift 2 ;;
        --REF_FASTA | -a ) REF_FASTA="$2"; shift 2 ;;
        --KNOWNs | -k ) KNOWNs="$2"; shift 2 ;;
        --CALL_MODE | -m ) CALL_MODE="$2"; shift 2 ;;
        --THREADS | -t ) THREADS="$2"; shift 2 ;;
        --TASK_THREADS | -T ) TASK_THREADS="$2"; shift 2 ;;
        --WAIT_FOR_DATA | -w ) WAIT_FOR_DATA=1; shift ;;
        --CLEAN | -c ) CLEAN=1; shift ;;
        --Xmx ) Xmx="$2"; shift 2 ;;
        --GATK4_JAVA_OPTS ) GATK4_JAVA_OPTS="$2"; shift 2 ;;
        --CHR_LIST ) CHR_LIST="$2"; shift 2 ;;
        --help | -h ) Help; exit 0 ;;
        -- ) shift; break ;;
        * ) Help; exit 1 ;;
    esac
done

cmds_char="pigz bgzip tabix $GATK4 $SAMTOOLS $VCFTOOLS"
for cmd in $cmds_char; do CmdCheck $cmd; done
args_char="SAMPLE_LIST OUTPUT_DIR REF_FASTA KNOWNs"
args_empty=""
for arg in $args_char; do [ -z "$(CharGetVar $arg)" ] && args_empty="${args_empty} --$arg"; done
[ -n "$args_empty" ] && Help && echoError "You must specify$args_empty"
(( TASK_THREADS >= THREADS )) && TASK_THREADS=$THREADS

[[ "$CALL_MODE" =~ ^(joint|single)$ ]] || echoError "CALL_MODE must be 'joint' or 'single'! "
[[ -n "$GATK4_JAVA_OPTS" && "$GATK4_JAVA_OPTS" =~ ^--java-options ]] && echoError "GATK4_JAVA_OPTS should NOT have '--java-options' ! "
GATK4_JAVA_OPTS_TOTAL="-Xmx${Xmx} -Dsamjdk.use_async_io_read_samtools=true -Dsamjdk.use_async_io_write_samtools=true -Djava.util.concurrent.ForkJoinPool.common.parallelism=${THREADS}"
GATK4_JAVA_OPTS_TASK="-Xmx${Xmx} -Dsamjdk.use_async_io_read_samtools=true -Dsamjdk.use_async_io_write_samtools=true -Djava.util.concurrent.ForkJoinPool.common.parallelism=${TASK_THREADS}"
# Some options may not take effect (e.g., -Dsamjdk.use_async_io_read_samtools, -Dsamjdk.use_async_io_write_samtools)
# because they are hard-coded in the GATK Python launcher script.
GATK4_JAVA_OPTS_TOTAL=$(build_java_options "$GATK4_JAVA_OPTS_TOTAL" "$GATK4_JAVA_OPTS")
GATK4_JAVA_OPTS_TASK=$(build_java_options "$GATK4_JAVA_OPTS_TASK" "$GATK4_JAVA_OPTS")

echoStep "Configuration: THREADS=$THREADS, TASK_THREADS=$TASK_THREADS, Xmx=$Xmx" "config"
echoStep "Samples: $SAMPLE_LIST" "config"
echoStep "Chromosomes: $CHR_LIST" "config"

FileCheck "$REF_FASTA"
for KNOWN in $KNOWNs; do FileCheck "$KNOWN"; done
OUTPUT_DIR=$(DirAbspath $OUTPUT_DIR)
REF_FASTA=$(FileAbspath $REF_FASTA)
KNOWNs=$(for KNOWN in $KNOWNs; do FileAbspath "$KNOWN"; done)
OUTPUT_VCF=$OUTPUT_DIR/VCF_OUTPUT_
OUTPUT_REF=$OUTPUT_DIR/REF_
[ -d "$OUTPUT_DIR" ] || mkdir -p $OUTPUT_DIR
[ -d "$OUTPUT_REF" ] || mkdir -p $OUTPUT_REF
cd $OUTPUT_DIR


echoStep "Generate&check REF/KNOWNs INDEX for GATK" gatk

# REF Index
REF_FASTA_TMP=$OUTPUT_REF/$(basename $REF_FASTA | sed "s/.gz$//")
[[ $REF_FASTA_TMP =~ \.f(ast)?a$ ]] || echoError "REF_FASTA should be .f(ast)a(.gz)! "
[ ! -e $REF_FASTA_TMP ] && {
    [[ $REF_FASTA =~ \.gz$ ]] && pigz -dcp $THREADS $REF_FASTA > $REF_FASTA_TMP || ln -s $REF_FASTA $REF_FASTA_TMP
}
[ -s "${REF_FASTA_TMP}.fai" ] || $SAMTOOLS faidx "$REF_FASTA_TMP"
# for i in $CHR_LIST; do
#     [[ ${i/chr/} == M ]] && echo -e "${i}\tMT" || echo -e "${i}\t${i/chr/}" 
# done > ${REF_FASTA_TMP}_ALT_NAMES
REF_FASTA_TMP_DICT=$(echo $REF_FASTA_TMP | sed 's/\.\(fa\|fasta\)$/\.dict/')
[ -s ${REF_FASTA_TMP_DICT} ] || $GATK4 CreateSequenceDictionary -R $REF_FASTA_TMP -O ${REF_FASTA_TMP_DICT} &>/dev/null \
|| echoError "Failed to create dictionary for $REF_FASTA_TMP"
# --ALT_NAMES ${REF_FASTA_TMP}_ALT_NAMES

# KNOWN Index
bgzip_gz_file() {
    local gz_file="$1"
    local force=$2
    local n_thread=$(( THREADS / 2 )); (( n_thread < 1 )) && n_thread=1
    # $force should be string "true" or "false" (commands)
    if $force; then
        pigz -dcp $n_thread "$gz_file" | bgzip -@ $n_thread -c > "${gz_file}_tmp" && rm "$gz_file" && mv "${gz_file}_tmp" "$gz_file"
        return $?
    fi

    if bgzip -t "$gz_file" &>/dev/null; then
        return 0
    else
        pigz -dcp $n_thread "$gz_file" | bgzip -@ $n_thread -c > "${gz_file}_tmp" && rm "$gz_file" && mv "${gz_file}_tmp" "$gz_file"
        return $?
    fi
}
> "$OUTPUT_REF/KNOWN.index.log"
KNOWNs=$(
    for KNOWN in $KNOWNs; do
        KNOWN_TMP=$OUTPUT_REF/$(basename $KNOWN)
        {
            [ -e "$KNOWN_TMP" ] || ln -s "$KNOWN" "$KNOWN_TMP"
            [[ "$KNOWN_TMP" =~ \.gz$ ]] && bgzip_gz_file "$KNOWN_TMP" false || echoError "KNOWN should be bgzip file! ($KNOWN)"
            [ ! -e "${KNOWN_TMP}.tbi" ] && [ -s "${KNOWN}.tbi" ] && ln -s "${KNOWN}.tbi" "${KNOWN_TMP}.tbi" || {
                [ -s "${KNOWN_TMP}.tbi" ] || {
                    $GATK4 IndexFeatureFile -I $KNOWN_TMP || {
                        echoStep "forced to bgzip $KNOWN_TMP again and then index" gatk
                        bgzip_gz_file "$KNOWN_TMP" true && $GATK4 IndexFeatureFile -I $KNOWN_TMP # forced to bgzip again and then index
                    } 
                } &>>"$OUTPUT_REF/KNOWN.index.log" || echoError "Failed to index $KNOWN_TMP"
            }
        } &
        echo "$KNOWN_TMP"
    done
    wait
)
KNOWNs_options1=$(for KNOWN in $KNOWNs; do echo "--known-sites $KNOWN"; done)
KNOWNs_options2=$(for KNOWN in $KNOWNs; do echo "-resource:dbsnp,known=true,training=true,truth=true,prior=2.0 $KNOWN"; done)


# GATK
job_cores=0
trap 'kill $(jobs -p) 2>/dev/null' EXIT
for sample in $SAMPLE_LIST; do

    cd $OUTPUT_DIR
    if [[ "$CALL_MODE" == "single" ]]; then
        if [[ -s "${sample}.merge09maf05.recode.vcf.gz" && -s "${sample}.snps.VQSR.vcf.gz" ]]; then
            echoStep "Skipping ${sample}, output already exists." gatk
            continue
        fi
    fi

    echoStep "Running GATK for $sample" gatk
    log_out=$(FileAbspath "${sample}.GATK4_out.log")
    log_step=$(FileAbspath "${sample}.GATK4_step.log")
    [ -s $log_out ] && mv $log_out ${log_out}_old
    [ -s $log_step ] && mv $log_step ${log_step}_old
    > $log_out
    > $log_step

    {
        run_gatk_hap=true
        if [[ -s "${sample}.erc.g.vcf" && -s "${sample}.erc.g.vcf.idx" ]]; then
            echoStep "Skipping ${sample}-HaplotypeCaller, output already exists." gatk
            run_gatk_hap=false
        else
            if [[ -d "${sample}_gatk" ]] && [[ -s "${sample}_gatk/${sample}.erc.g.vcf" ]]; then
                echoStep "It seems ongoing HaplotypeCaller for $sample. Checking if actively being written..." gatk
                if check_file_growing "${sample}_gatk/${sample}.erc.g.vcf"; then
                    echoStep "Detected active write on ${sample}_gatk/${sample}.erc.g.vcf. Another job may be running. Waiting..." gatk

                    wait_seconds=0
                    time_limits=$(( 24 * 3600 ))  # 24 h
                    until [[ -s "${sample}.erc.g.vcf" && -s "${sample}.erc.g.vcf.idx" ]]; do
                        sleep 200
                        ((wait_seconds += 200))
                        echoStep "Still waiting... (${wait_seconds}s)" gatk
                        (( wait_seconds > time_limits )) && echoError "Timeout waiting for ${sample}.erc.g.vcf after 24 hours."
                    done

                    echoStep "Got HaplotypeCaller output." star
                    run_gatk_hap=false
                else
                    echoStep "Detected incomplete ${sample}_gatk/${sample}.erc.g.vcf. Re-run GATK for ${sample}." gatk
                fi
            fi
        fi

        if $run_gatk_hap; then
            (
                set -e
                export OMP_NUM_THREADS=${TASK_THREADS} # for gatk PairHMM
    
                cd $OUTPUT_DIR
                [ -d "${sample}_gatk" ] || mkdir -p "${sample}_gatk"; cd "${sample}_gatk"
                echoStep "${sample}: GATK will output in $OUTPUT_DIR/${sample}_gatk/" gatk

                echoStep "${sample}: check bam..." gatk
                star_bam="$OUTPUT_DIR/${sample}_star/${sample}Aligned.out.bam"
                {
                    if [[ $WAIT_FOR_DATA -eq 1 ]]; then
                        echoStep "Waiting for $star_bam to be stable..." gatk
                        until check_file_stable "$star_bam"; do
                            echoStep "Still waiting for $star_bam to be stable..." gatk >&2
                        done
                        echoStep "Got stable $star_bam." gatk
                    fi
                    [[ -s "$star_bam" ]] || echoError "BAM file $star_bam not found! "
                    $SAMTOOLS quickcheck "$star_bam" &>/dev/null || echoError "$star_bam failed quickcheck! "
                } &>> $log_out
                echoStep "${sample}: check bam over" gatk

                echoStep "${sample}: process BAM..." gatk
                { # use GATK to replace what PICARD can do
                    $SAMTOOLS sort -@ "$TASK_THREADS" -o "${sample}.sorted.bam" "$star_bam"
                    $SAMTOOLS index "${sample}.sorted.bam"
        
                    $GATK4 --java-options "$GATK4_JAVA_OPTS_TASK" AddOrReplaceReadGroups \
                        -I "${sample}.sorted.bam" \
                        -O "${sample}.bam" \
                        --RGID "$sample" \
                        --RGLB "lib$sample" \
                        --RGPL illumina \
                        --RGPU unit \
                        --RGSM "$sample"
                    rm "${sample}.sorted.bam"
                    $SAMTOOLS index "${sample}.bam"
        
                    $GATK4 --java-options "$GATK4_JAVA_OPTS_TASK" MarkDuplicates \
                        -I "${sample}.bam" \
                        -O "${sample}.markdup.bam" \
                        -M "${sample}.markdup.txt" \
                        --REMOVE_DUPLICATES true
                    rm "${sample}.bam"
                    $GATK4 --java-options "$GATK4_JAVA_OPTS_TASK" BuildBamIndex \
                        -I "${sample}.markdup.bam"

                    $SAMTOOLS flagstat -@ "$TASK_THREADS" "${sample}.markdup.bam" > "${sample}.markdup.stat"
                } &>> $log_out
                echoStep "${sample}: process BAM over" gatk

                echoStep "${sample}: gatk BaseRecalibrator..." gatk
                $GATK4 --java-options "$GATK4_JAVA_OPTS_TASK" BaseRecalibrator \
                    -R "$REF_FASTA_TMP" \
                    -I "${sample}.markdup.bam" \
                    $KNOWNs_options1 \
                    -O "recal_data_${sample}.table" &>> $log_out
                echoStep "${sample}: gatk BaseRecalibrator over" gatk

                echoStep "${sample}: gatk ApplyBQSR..." gatk
                $GATK4 --java-options "$GATK4_JAVA_OPTS_TASK" ApplyBQSR \
                    --bqsr-recal-file "recal_data_${sample}.table" \
                    -R "$REF_FASTA_TMP" \
                    -I "${sample}.markdup.bam" \
                    -O "${sample}.markdup.BQSR.bam" &>> $log_out
                rm "${sample}.markdup.bam" "recal_data_${sample}.table"
                echoStep "${sample}: gatk ApplyBQSR over" gatk

                echoStep "${sample}: gatk SplitNCigarReads..." gatk
                $GATK4 --java-options "$GATK4_JAVA_OPTS_TASK" SplitNCigarReads \
                    -R "$REF_FASTA_TMP" \
                    -I "${sample}.markdup.BQSR.bam" \
                    -O "${sample}.markdup.BQSR.split.bam" &>> $log_out
                rm "${sample}.markdup.BQSR.bam"
                echoStep "${sample}: gatk SplitNCigarReads over" gatk

                echoStep "${sample}: gatk HaplotypeCaller..." gatk
                $GATK4 --java-options "$GATK4_JAVA_OPTS_TASK" HaplotypeCaller \
                    -R "$REF_FASTA_TMP" \
                    -I "${sample}.markdup.BQSR.split.bam" \
                    -ERC GVCF \
                    -O "${sample}.erc.g.vcf" &>> $log_out
                rm "${sample}.markdup.BQSR.split.bam"
                echoStep "${sample}: gatk HaplotypeCaller over" gatk

                echoStep "${sample}: compress & move & clean..." gatk
                {
                    # bgzip -@ ${TASK_THREADS} -c "${sample}.erc.g.vcf" > "$OUTPUT_DIR/${sample}.erc.g.vcf.gz"
                    # rm "${sample}.erc.g.vcf"
                    mv "${sample}.erc.g.vcf" $OUTPUT_DIR
                    mv "${sample}.erc.g.vcf.idx" $OUTPUT_DIR

                    cd $OUTPUT_DIR; [[ $CLEAN -eq 1 ]] && [ -d ${sample}_gatk ] && rm -r ${sample}_gatk/
                }
                echoStep "${sample}: compress & move & clean over" gatk

            ) &>> $log_step
        fi
        
        if [[ "$CALL_MODE" == "single" ]]; then
            (
                set -e
                export OMP_NUM_THREADS=${TASK_THREADS} # for gatk PairHMM

                cd $OUTPUT_DIR
                [[ -s "${sample}.erc.g.vcf" && -s "${sample}.erc.g.vcf.idx" ]] || echoError "No HaplotypeCaller output for $sample! "
                [ -d "${sample}_vcf" ] || mkdir -p "${sample}_vcf"; cd "${sample}_vcf"
                echoStep "${sample}: VCF will output in $OUTPUT_DIR/${sample}_vcf/" vcf

                echoStep "${sample}: gatk GenotypeGVCFs (single sample)..." vcf
                $GATK4 --java-options "$GATK4_JAVA_OPTS_TASK" GenotypeGVCFs \
                    -R "$REF_FASTA_TMP" \
                    -V "../${sample}.erc.g.vcf" \
                    -O "${sample}.raw.vcf" &>> $log_out
                echoStep "${sample}: gatk GenotypeGVCFs over" vcf

                echoStep "${sample}: gatk VariantRecalibrator..." vcf
                $GATK4 --java-options "$GATK4_JAVA_OPTS_TASK" VariantRecalibrator \
                    -R "$REF_FASTA_TMP" \
                    -V "${sample}.raw.vcf" \
                    $KNOWNs_options2 \
                    -an DP -an QD -an FS -an SOR \
                    -mode SNP \
                    -tranche 100.0 -tranche 99.9 -tranche 99.0 -tranche 95.0 -tranche 90.0 \
                    -O "${sample}.snp.recal" \
                    --tranches-file "${sample}.snp.tranches" \
                    --rscript-file "${sample}.snp.plots.R" &>> $log_out
                echoStep "${sample}: gatk VariantRecalibrator over" vcf

                echoStep "${sample}: gatk ApplyVQSR..." vcf
                $GATK4 --java-options "$GATK4_JAVA_OPTS_TASK" ApplyVQSR \
                    -R "$REF_FASTA_TMP" \
                    --variant "${sample}.raw.vcf" \
                    --ts-filter-level 99.0 \
                    --tranches-file "${sample}.snp.tranches" \
                    --recal-file "${sample}.snp.recal" \
                    --mode SNP \
                    --output "${sample}.snps.VQSR.vcf" &>> $log_out
                rm "${sample}.raw.vcf" "${sample}.snp.tranches" "${sample}.snp.recal"
                echoStep "${sample}: gatk ApplyVQSR over" vcf

                echoStep "${sample}: vcftools filter..." vcf
                $VCFTOOLS --vcf "${sample}.snps.VQSR.vcf" \
                    --max-missing 0.9 \
                    --maf 0.05 \
                    --recode --recode-INFO-all \
                    --out "${sample}.merge09maf05" &>> $log_out
                echoStep "${sample}: vcftools filter over" gatk

                echoStep "${sample}: compress & move & clean..." vcf
                {
                    bgzip -@ ${TASK_THREADS} -c "$OUTPUT_DIR/${sample}.erc.g.vcf" > "$OUTPUT_DIR/${sample}.erc.g.vcf.gz"
                    rm "$OUTPUT_DIR/${sample}.erc.g.vcf"

                    bgzip -@ ${TASK_THREADS} -c "${sample}.merge09maf05.recode.vcf" > "$OUTPUT_DIR/${sample}.merge09maf05.recode.vcf.gz"
                    rm "${sample}.merge09maf05.recode.vcf"

                    bgzip -@ ${TASK_THREADS} -c "${sample}.snps.VQSR.vcf" > "$OUTPUT_DIR/${sample}.snps.VQSR.vcf.gz"
                    rm "${sample}.snps.VQSR.vcf"

                    tabix -p vcf "$OUTPUT_DIR/${sample}.merge09maf05.recode.vcf.gz"
                    [ -s "${sample}.snps.VQSR.vcf.idx" ] && mv "${sample}.snps.VQSR.vcf.idx" $OUTPUT_DIR

                    cd $OUTPUT_DIR; [[ $CLEAN -eq 1 ]] && [ -d ${sample}_vcf ] && rm -r ${sample}_vcf/
                }
                echoStep "${sample}: compress & move & clean over" vcf
            
            ) &>> $log_step
        fi

        echoStep "$sample over" gatk
    } &

    ((job_cores+=TASK_THREADS))
    while ((job_cores + TASK_THREADS > THREADS)); do
        wait -n
        ((job_cores-=TASK_THREADS))
    done
done
wait

trap - EXIT
echoStep "All GATK jobs finished." gatk


[[ "$CALL_MODE" == "single" ]] && exit 0


# Joint VCF
(
    set -e
    export OMP_NUM_THREADS=${THREADS} # for gatk PairHMM

    echoStep "Running Joint-VCF for all samples" vcf
    cd $OUTPUT_DIR
    vcf_log=$(FileAbspath "Joint_VCF_out.log")
    vcf_step=$(FileAbspath "Joint_VCF_step.log")
    [ -s $vcf_log ] && mv $vcf_log ${vcf_log}_old
    [ -s $vcf_step ] && mv $vcf_step ${vcf_step}_old
    > $vcf_log
    > $vcf_step
    [ -d "$OUTPUT_VCF" ] || mkdir -p $OUTPUT_VCF

    {
        > "$OUTPUT_VCF/input.list"
        > "$OUTPUT_VCF/chr.list"
        for sample in $SAMPLE_LIST; do
            gvcf="$OUTPUT_DIR/${sample}.erc.g.vcf"
            [ -s $gvcf ] && echo "$gvcf" >> "$OUTPUT_VCF/input.list"
        done
        for chr in $CHR_LIST; do
            echo "$OUTPUT_VCF/gvcfs_${chr}.vcf" >> "$OUTPUT_VCF/chr.list"
        done
        [ -s "$OUTPUT_VCF/input.list" ] || echoError "No valid GVCF files found for VCF steps! "


        echoStep "Processing chromosome..." vcf
        job_cores=0
        for chr in $CHR_LIST; do
            (
                echoStep "Processing chromosome: $chr" vcf
                {
                    $GATK4 --java-options "$GATK4_JAVA_OPTS_TOTAL" GenomicsDBImport \
                        -R "$REF_FASTA_TMP" \
                        -V "$OUTPUT_VCF/input.list" \
                        --genomicsdb-workspace-path "$OUTPUT_VCF/${chr}.db" \
                        -L "$chr"
            
                    $GATK4 --java-options "$GATK4_JAVA_OPTS_TOTAL" GenotypeGVCFs \
                        -R "$REF_FASTA_TMP" \
                        -V "gendb://${OUTPUT_VCF}/${chr}.db" \
                        -O "$OUTPUT_VCF/gvcfs_${chr}.vcf"
                } &> "$OUTPUT_VCF/${chr}.run.log"
            
                echoStep "$chr over" vcf
            ) &

            # Empirically, GenomicsDBImport/GenotypeGVCFs use low CPU (<2 cores per task),
            # so we allow up to THREADS concurrent jobs.
            ((job_cores+=1))
            if (( job_cores >= THREADS )); then
                wait -n
                ((job_cores-=1))
            fi
        done
        wait
        echoStep "Processing chromosome over" vcf

        echoStep "gatk MergeVcfs..." vcf
        $GATK4 --java-options "$GATK4_JAVA_OPTS_TOTAL" MergeVcfs \
            -I "$OUTPUT_VCF/chr.list" \
            -O "$OUTPUT_VCF/merged.vcf" &>> $vcf_log
        echoStep "gatk MergeVcfs over" vcf

        echoStep "gatk VariantRecalibrator..." vcf
        $GATK4 --java-options "$GATK4_JAVA_OPTS_TOTAL" VariantRecalibrator \
            -R "$REF_FASTA_TMP" \
            -V "$OUTPUT_VCF/merged.vcf" \
            $KNOWNs_options2 \
            -an DP -an QD -an FS -an SOR \
            -mode SNP \
            -tranche 100.0 -tranche 99.9 -tranche 99.0 -tranche 95.0 -tranche 90.0 \
            -O "$OUTPUT_VCF/snp.recal" \
            --tranches-file "$OUTPUT_VCF/snp.tranches" \
            --rscript-file "$OUTPUT_VCF/snp.plots.R" &>> $vcf_log
        echoStep "gatk VariantRecalibrator over" vcf

        echoStep "gatk ApplyVQSR..." vcf
        $GATK4 --java-options "$GATK4_JAVA_OPTS_TOTAL" ApplyVQSR \
            -R "$REF_FASTA_TMP" \
            --variant "$OUTPUT_VCF/merged.vcf" \
            --ts-filter-level 99.0 \
            --tranches-file "$OUTPUT_VCF/snp.tranches" \
            --recal-file "$OUTPUT_VCF/snp.recal" \
            --mode SNP \
            --output "$OUTPUT_VCF/snps.VQSR.vcf" &>> $vcf_log
        echoStep "gatk ApplyVQSR over" vcf

        echoStep "vcftools filter..." vcf
        $VCFTOOLS --vcf "$OUTPUT_VCF/snps.VQSR.vcf" \
            --max-missing 0.9 \
            --maf 0.05 \
            --recode --recode-INFO-all \
            --out "$OUTPUT_VCF/merge09maf05" &>> $vcf_log
        echoStep "vcftools filter over" vcf

        echoStep "compress & move & clean..." vcf
        {
            bgzip -@ ${THREADS} -c "$OUTPUT_VCF/merge09maf05.recode.vcf" > "$OUTPUT_DIR/merge09maf05.recode.vcf.gz"
            rm "${OUTPUT_VCF}/merge09maf05.recode.vcf"

            bgzip -@ ${THREADS} -c "$OUTPUT_VCF/snps.VQSR.vcf" > "$OUTPUT_DIR/snps.VQSR.vcf.gz"
            rm "${OUTPUT_VCF}/snps.VQSR.vcf"

            tabix -p vcf "$OUTPUT_DIR/merge09maf05.recode.vcf.gz"
            [ -s "$OUTPUT_VCF/snps.VQSR.vcf.idx" ] && mv "$OUTPUT_VCF/snps.VQSR.vcf.idx" $OUTPUT_DIR

            [[ $CLEAN -eq 1 ]] && [ -d $OUTPUT_VCF ] && rm -r ${OUTPUT_VCF}
        }
        echoStep "compress & move & clean over" vcf

        echoStep "Joint-VCF pipeline finished." vcf

    } &> $vcf_step
)
