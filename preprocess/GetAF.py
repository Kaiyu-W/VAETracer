#!/usr/bin/env python3

import os
import csv
import time
import argparse
import logging
import shutil
import subprocess
from contextlib import contextmanager
from multiprocessing import Pool
from functools import partial
from collections import defaultdict, OrderedDict

import pandas as pd
import pyarrow.parquet as pq
import pyarrow as pa
import pysam

import sys
if sys.version_info < (3, 8):
    try:
        import importlib_metadata as metadata
    except ImportError:
        raise ImportError("Please install importlib_metadata: pip install importlib_metadata")
    import importlib
    importlib.metadata = metadata
    sys.modules['importlib.metadata'] = metadata
import pyranges as pr


#  Constants
TOOL_NAME = "GetAF"
__version__ = "0.1.0"

PARQUET_SCHEMA = pa.schema([
    ('CellBarcode', pa.string()),
    ('Mutation', pa.string()),
    ('Gene', pa.string()),
    ('AF', pa.float32()),
    ('ALT', pa.int32()),
    ('REF', pa.int32())
])
DF_COLUMNS=["CellBarcode", "Mutation", "Gene", "AF", "ALT", "REF"]
CHUNK_PREFIX = "chunk_"
SEGMENT_PREFIX = "segment_"
DONE_SUFFIX = ".done"
PARQUET_SUFFIX = ".parquet"


# tools function
import multiprocessing
import multiprocessing.pool
class NoDaemonProcess(multiprocessing.Process):
    def _get_daemon(self):
        return False
    def _set_daemon(self, value):
        pass
    daemon = property(_get_daemon, _set_daemon)
class NoDaemonProcessPool(multiprocessing.pool.Pool):
    Process = NoDaemonProcess

def format_seconds(seconds):
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    
    parts = []
    if h > 0:
        parts.append(f"{h} hour" + ("s" if h != 1 else ""))
    if m > 0:
        parts.append(f"{m} minute" + ("s" if m != 1 else ""))
    if s > 0 or not parts:
        parts.append(f"{s} second" + ("s" if s != 1 else ""))
    
    return ", ".join(parts)

def detect_compression_method():
    try:
        if pa.Codec.is_available('zstd'):
            return 'zstd'
        else:
            logging.warning("ZSTD compression not supported by PyArrow, falling back to snappy")
            return 'snappy'
    except Exception as e:
        logging.warning(f"Unable to check ZSTD support: {e}, using snappy")
        return 'snappy'

def is_parquet_valid(file_path):
    """
    Verify if a Parquet file is complete and valid
    Returns: (bool, error_message)
    """
    try:
        # Quick size check (Parquet files have footer)
        if os.path.getsize(file_path) < 128:
            return False, "File too small"
        # Verify metadata
        pq.read_metadata(file_path)
        return True, ""
    except Exception as e:
        return False, str(e)

def data_to_table(data):
    df = pd.DataFrame(data, columns=DF_COLUMNS)
    table = pa.Table.from_pandas(df, schema=PARQUET_SCHEMA)
    return table


# vcf
@contextmanager
def get_stream(vcf_path):
    """Yield a stream that automatically handles bgzip process cleanup."""
    if vcf_path.endswith(".gz"):
        if not shutil.which("bgzip"):
            raise RuntimeError("bgzip not found in PATH. Please install htslib or provide a plain .vcf file.")
        
        logging.info(f"Reading and decompressing {vcf_path}...")
        num_threads = os.cpu_count()
        process = subprocess.Popen(
            ["bgzip", "-d", "-c", f"-@{num_threads}", vcf_path],
            stdout=subprocess.PIPE, text=True, bufsize=1024*1024
        )
        try:
            yield process.stdout
        finally:
            process.wait()
    else:
        with open(vcf_path, "r") as f:
            yield f

def read_vcf_mutations(vcf_path):
    mutations = []
    with get_stream(vcf_path) as stream:
        for line in stream:
            if line.startswith("##"):
                continue
            if line.startswith("#CHROM"):
                header = line.strip().lstrip("#").split("\t")
                break
        reader = csv.DictReader(stream, fieldnames=header, delimiter="\t")
        for row in reader:
            mutations.append({
                "chrom": row["CHROM"],
                "position": int(row["POS"]),
                "ref": row["REF"],
                "alt": row["ALT"]
            })

    logging.info(f"Read {len(mutations)} mutations from {vcf_path}")
    return mutations

def merge_vcf_mutations(vcf_paths):
    all_data = []
    for vcf_path in vcf_paths:
        logging.info(f"Processing {vcf_path}...")
        try:
            mutations = read_vcf_mutations(vcf_path)
            batch_data = [
                (mut["chrom"], mut["position"], mut["ref"], mut["alt"])
                for mut in mutations
            ]
            all_data.extend(batch_data)
        except Exception as e:
            logging.error(f"Failed to process {vcf_path}: {e}")
            raise

    if not all_data:
        logging.warning("No mutations were read from any VCF files.")
        return []

    df = pd.DataFrame(all_data, columns=["chrom", "position", "ref", "alt"])
    before_count = len(df)
    df.drop_duplicates(inplace=True)
    after_count = len(df)

    def _chrom_key(chrom):
        chrom_clean = chrom.replace("chr", "")
        return (0, int(chrom_clean)) if chrom_clean.isdigit() else (1, chrom_clean)
    df["chrom_sort_key"] = df["chrom"].map(_chrom_key)
    df.sort_values(["chrom_sort_key", "position"], inplace=True)
    df.drop(columns=["chrom_sort_key"], inplace=True)

    result = df.to_dict('records')
    logging.info(
        f"Combined {before_count} total mutations to {after_count} unique mutations "
        f"from {len(vcf_paths)} VCF files."
    )

    return result


# bam
def get_chromosomes(bam_file):
    with pysam.AlignmentFile(bam_file, "rb") as bam:
        return bam.references


# gtf
def read_gtf(gtf_path):
    gtf = pr.read_gtf(gtf_path)
    gtf_genes = gtf[gtf.Feature == "gene"]
    return gtf_genes

def annotate_positions_with_gene(chromosomes, positions, gtf_genes):
    if len(chromosomes) != len(positions):
        raise ValueError("Chromosomes and Positions must be of the same length")
    if not isinstance(gtf_genes, pr.PyRanges):
        raise ValueError("gtf_genes must be a PyRanges object")

    loci = [f"{c}:{p}" for c, p in zip(chromosomes, positions)]
    loci_df = pd.DataFrame({
        "Chromosome": chromosomes,
        "Start": [int(p) - 1 for p in positions],
        "End": [int(p) for p in positions],
        "locus": loci
    }).drop_duplicates()
    loci_pr = pr.PyRanges(loci_df)
    annotated = loci_pr.join(
        gtf_genes, apply_strand_suffix=False, nb_cpu=1
    )

    if len(annotated) == 0:
        logging.warning(f"No gene annotations found for any of the {len(chromosomes)} loci in {chromosomes[0]}")
        return [None] * len(chromosomes)

    locus_to_genes = defaultdict(list)
    for _, row in annotated.df.iterrows():
        locus = row["locus"]
        gene_name = row.get("gene_name", row.get("gene_id", None))
        if gene_name is not None:
            locus_to_genes[locus].append(gene_name)

    result_dict = {}
    for locus in loci_df['locus']:
        genes = locus_to_genes.get(locus, [])
        if genes:
            unique_sorted_genes = sorted(set(genes))
            result_dict[locus] = ",".join(unique_sorted_genes)
        else:
            result_dict[locus] = None

    return [ result_dict[locus] for locus in loci ]


# chunk / segment
def scan_task_state(task_tmp_dir, segment_size, max_seg_idx, tag, cleanup_incomplete=True):
    """
    Scan task state with integrity verification and row count tracking
    
    Parameters:
    -----------
    task_tmp_dir : str
        Temporary directory for this task
    segment_size : int
        Number of chunks per segment
    tag : str
        Logging tag (e.g., 'L7-chr5')
    cleanup_incomplete : bool
        If True, delete incomplete segment and orphaned chunk files.
        If False, only count completed chunks without modifying filesystem.
    
    Returns:
    --------
    (completed_chunks, valid_segments, total_rows)
    """
    # First, collect all segments and their expected chunks
    segments_info = defaultdict(lambda: {'chunks': [], 'row_count': 0})
    task_tmp_files = os.listdir(task_tmp_dir)
    completed_chunks = set()
    valid_segments = set()
    total_rows = 0

    # 1. Verify segment parquet files first
    for f in task_tmp_files:
        if f.startswith(SEGMENT_PREFIX) and f.endswith(PARQUET_SUFFIX):
            try:
                seg_idx = int(f.split("_")[1].split(".")[0])
                seg_done = f"{SEGMENT_PREFIX}{seg_idx}{DONE_SUFFIX}"
                
                valid, _ = is_parquet_valid(os.path.join(task_tmp_dir, f))
                if not valid:
                    # Delete invalid segment files only if cleanup_incomplete is True
                    if cleanup_incomplete:
                        for cleanup_file in [f, seg_done]:
                            full_path = os.path.join(task_tmp_dir, cleanup_file)
                            if os.path.exists(full_path):
                                os.remove(full_path)
                    continue
                
                # Parse segment done file for row counts and chunk info
                seg_done_path = os.path.join(task_tmp_dir, seg_done)
                if os.path.exists(seg_done_path):
                    with open(seg_done_path, "r") as df:
                        for line in df:
                            parts = line.strip().split(":")
                            if len(parts) == 2:
                                chunk_idx = int(parts[0])
                                row_count = int(parts[1])
                                segments_info[seg_idx]['chunks'].append(chunk_idx)
                                segments_info[seg_idx]['row_count'] += row_count
            except Exception as e:
                logging.warning(f"Error processing segment file {f}: {str(e)}")
                continue
    
    # Mark segments as invalid if they don't have the expected number of chunks
    for seg_idx, info in list(segments_info.items()):
        # ONLY consider a segment complete if it has exactly segment_size chunks
        # or it indeed is the last segment
        if len(info['chunks']) == segment_size or seg_idx == max_seg_idx:
            for chunk_idx in info['chunks']:
                completed_chunks.add(chunk_idx)
            total_rows += info['row_count']
            valid_segments.add(seg_idx)
        else:
            if cleanup_incomplete:
                # Segment is incomplete, cleanup
                logging.debug(f"Segment {seg_idx+1} is incomplete ({SEGMENT_PREFIX}{seg_idx}{PARQUET_SUFFIX} has {len(info['chunks'])} chunks), marking as invalid")
                
                # Delete segment files
                for ext in [PARQUET_SUFFIX, DONE_SUFFIX]:
                    seg_file = os.path.join(task_tmp_dir, f"{SEGMENT_PREFIX}{seg_idx}{ext}")
                    if os.path.exists(seg_file):
                        os.remove(seg_file)
    
    # 2. Verify orphaned chunk parquet files (not in any segment)
    for f in task_tmp_files:
        if f.startswith(CHUNK_PREFIX) and f.endswith(PARQUET_SUFFIX):
            try:
                chunk_idx = int(f.split("_")[1].split(".")[0])
                chunk_done = f"{CHUNK_PREFIX}{chunk_idx}{DONE_SUFFIX}"
                
                # Skip chunks already counted in segments
                if chunk_idx in completed_chunks:
                    continue
                    
                valid, _ = is_parquet_valid(os.path.join(task_tmp_dir, f))
                if not valid:
                    if cleanup_incomplete:
                        # Delete invalid chunk files
                        for cleanup_file in [f, chunk_done]:
                            full_path = os.path.join(task_tmp_dir, cleanup_file)
                            if os.path.exists(full_path):
                                os.remove(full_path)
                    continue
                
                # Read row count from .done file
                if os.path.exists(os.path.join(task_tmp_dir, chunk_done)):
                    try:
                        with open(os.path.join(task_tmp_dir, chunk_done), "r") as df:
                            row_count = int(df.read().strip())
                        completed_chunks.add(chunk_idx)
                        total_rows += row_count
                    except Exception as e:
                        logging.warning(f"Error reading chunk done file {chunk_done}: {str(e)}")
                        
            except Exception as e:
                logging.warning(f"Error processing chunk file {f}: {str(e)}")
                continue

    logging.debug(f"Found {len(valid_segments)} valid segments for {tag}, "
                 f"{len(completed_chunks)} completed chunks")
    return completed_chunks, valid_segments, total_rows

def create_segment_writer(task_tmp_dir, seg_idx, compression_method):
    """
    Create segment writer structure with paths (not actual writer object)
    
    Returns a dictionary with segment paths but no writer object
    """
    temp_path = os.path.join(task_tmp_dir, f"{SEGMENT_PREFIX}{seg_idx}{PARQUET_SUFFIX}.tmp")
    final_path = os.path.join(task_tmp_dir, f"{SEGMENT_PREFIX}{seg_idx}{PARQUET_SUFFIX}")
    
    # Cleanup possible leftovers
    for f in [temp_path, final_path]:
        if os.path.exists(f):
            os.remove(f)
    
    return {
        'temp_path': temp_path,
        'final_path': final_path,
        'seg_idx': seg_idx,
        'compression_method': compression_method
    }

def finalize_segment(segment, chunk_indices_with_counts, task_tmp_dir, tag, n_segments_total):
    """
    Finalize segment file by merging all chunk data
    
    This correctly merges all chunk Parquet files into a single segment file
    using pq.write_table() as in merge_parquets_for_label()
    
    chunk_indices_with_counts: list of (chunk_idx, row_count) tuples
    """
    # 1. Collect all chunk Parquet files
    chunk_files = []
    for chunk_idx, _ in chunk_indices_with_counts:
        chunk_file = os.path.join(task_tmp_dir, f"{CHUNK_PREFIX}{chunk_idx}{PARQUET_SUFFIX}")
        if os.path.exists(chunk_file):
            chunk_files.append(chunk_file)
    
    if not chunk_files:
        logging.warning(f"No valid chunk files to merge for segment {segment['seg_idx']}")
        return 
    
    # 2. Read and merge all chunk data
    dataset = pq.ParquetDataset(chunk_files, use_legacy_dataset=False)
    table = dataset.read(columns=DF_COLUMNS)
    
    # 3. Write merged table to segment file
    pq.write_table(table, segment['temp_path'], compression=segment['compression_method'])
    
    # 4. Atomic rename to final path
    os.rename(segment['temp_path'], segment['final_path'])
    
    # 5. Create segment marker
    seg_done = os.path.join(task_tmp_dir, f"{SEGMENT_PREFIX}{segment['seg_idx']}{DONE_SUFFIX}")
    total_segment_rows = 0
    with open(seg_done, "w") as f:
        for chunk_idx, row_count in chunk_indices_with_counts:
            f.write(f"{chunk_idx}:{row_count}\n")
            total_segment_rows += row_count
    
    logging.debug(f"Segment {segment['seg_idx']+1}/{n_segments_total} finalized with {len(chunk_indices_with_counts)} chunks for {tag}, "
                 f"{total_segment_rows} records")
    
    # 6. Cleanup chunk markers and files
    for chunk_idx, _ in chunk_indices_with_counts:
        for ext in [DONE_SUFFIX, PARQUET_SUFFIX]:
            chunk_file = os.path.join(task_tmp_dir, f"{CHUNK_PREFIX}{chunk_idx}{ext}")
            if os.path.exists(chunk_file):
                os.remove(chunk_file)

def process_single_chunk(
    chunk_idx, chrom_mutations, chunk_size, 
    bam, min_mapq, mutation_to_gene, label, task_tmp_dir
):
    """
    Process single chunk and write parquet file
    Returns: bool (success status)
    """
    try:
        start = chunk_idx * chunk_size
        end = min(start + chunk_size, len(chrom_mutations))
        current_chunk = chrom_mutations[start:end]
        
        result = defaultdict(lambda: defaultdict(lambda: {"ref": 0, "alt": 0, 'gene': None}))
        all_cells = set()
        all_mut_keys = []
        
        for mut in current_chunk:
            chrom_mut = mut["chrom"]
            pos_mut = mut["position"] - 1
            ref_mut = mut["ref"]
            alt_mut = mut["alt"]
            mut_key = f"{chrom_mut}:{mut['position']}:{ref_mut}>{alt_mut}"
            all_mut_keys.append(mut_key)
            gene = mutation_to_gene.get((mut["chrom"], mut["position"]), None)
            
            for pileupcolumn in bam.pileup(chrom_mut, pos_mut, pos_mut + 1, truncate=True):
                if pileupcolumn.pos != pos_mut:
                    continue
                
                for pileupread in pileupcolumn.pileups:
                    if pileupread.is_del or pileupread.is_refskip:
                        continue
                    aln = pileupread.alignment
                    if not aln.has_tag("CB") or aln.mapping_quality < min_mapq:
                        continue
                    
                    base = aln.query_sequence[pileupread.query_position]
                    cell = aln.get_tag("CB")
                    full_cell = f"{cell}_{label}"
                    all_cells.add(full_cell)
                    result[full_cell][mut_key]['gene'] = gene
                    
                    if base == alt_mut:
                        result[full_cell][mut_key]["alt"] += 1
                    elif base == ref_mut:
                        result[full_cell][mut_key]["ref"] += 1
        
        # Collect rows
        rows = []
        for cell in all_cells:
            for mut_key in all_mut_keys:
                data = result[cell].get(mut_key, {"ref": 0, "alt": 0, 'gene': None})
                ref = data["ref"]
                alt = data["alt"]
                gene = data['gene']
                if ref + alt > 0:
                    af = alt / (ref + alt)
                    rows.append((cell, mut_key, gene, af, alt, ref))
        
        # Write chunk parquet
        row_count = 0
        if rows:
            chunk_path = os.path.join(task_tmp_dir, f"{CHUNK_PREFIX}{chunk_idx}{PARQUET_SUFFIX}")
            table = data_to_table(rows)
            pq.write_table(table, chunk_path)
            row_count = len(rows)
        
        # Mark chunk done (only after successful write)
        with open(os.path.join(task_tmp_dir, f"{CHUNK_PREFIX}{chunk_idx}{DONE_SUFFIX}"), "w") as f:
            f.write(str(row_count))
        
        return row_count
    except Exception as e:
        logging.error(f"Chunk {chunk_idx} failed: {str(e)}")
        return -1

def process_segment(
    seg_idx,
    already_done,
    to_process,
    chrom_mutations,
    chunk_size,
    bam_file,
    min_mapq,
    mutation_to_gene,
    label,
    chrom,
    n_segments_total,
    task_tmp_dir,
    compression_method
):
    """Process a single segment. Trusts upstream state; no redundant checks."""
    try:
        # Step 1: Directly use the 'already_done' list from upstream.
        #         We assume their .done files are valid and readable.
        segment_chunks_done = []
        for chunk_idx in already_done:
            chunk_done_file = os.path.join(task_tmp_dir, f"{CHUNK_PREFIX}{chunk_idx}{DONE_SUFFIX}")
            if os.path.exists(chunk_done_file):
                with open(chunk_done_file, "r") as f:
                    row_count = int(f.read().strip())
                segment_chunks_done.append((chunk_idx, row_count))

        # If all chunks are already done, skip processing and finalize.
        n_process = len(to_process)
        if to_process:
            # Step 2: Process only the remaining chunks.
            with pysam.AlignmentFile(bam_file, "rb") as bam:
                for i, chunk_idx in enumerate(to_process):
                    row_count = process_single_chunk(
                        chunk_idx, chrom_mutations, chunk_size, 
                        bam, min_mapq, mutation_to_gene, label, task_tmp_dir
                    )
                    logging.debug(
                        f"Processing {label} - {chrom}: Segment {seg_idx+1}/{n_segments_total}, "
                        f"{i+1}/{n_process} remaining chunks"
                    )
                    if row_count >= 0:
                        segment_chunks_done.append((chunk_idx, row_count))

        # Step 3: Finalize the segment.
        current_segment = create_segment_writer(task_tmp_dir, seg_idx, compression_method)
        finalize_segment(current_segment, segment_chunks_done, task_tmp_dir, f'{label}-{chrom}', n_segments_total)
        return sum(row_count for _, row_count in segment_chunks_done)

    except Exception as e:
        logging.error(f"Error processing segment {seg_idx}: {str(e)}")
        return 0


# task
def process_task(
    task_info, chrom_mutations, mutation_to_gene,
    compression_method, chunk_size, segment_size, 
):
    """Main task processing with clean state management and time tracking"""
    bam_file = task_info["bam_file"]
    label = task_info["label"]
    chrom = task_info["chrom"]
    outdir = task_info["outdir"]
    min_mapq = task_info["min_mapq"]
    secondary_processes = task_info.get("secondary_processes", 1)
    
    start_time = time.time()
    logging.info(f"Processing {label} - {chrom} with {secondary_processes} secondary processes")
    
    global_done_file = os.path.join(outdir, f"{label}_{chrom}{DONE_SUFFIX}")
    if os.path.exists(global_done_file):
        logging.info(f"Skip completed task: {label} - {chrom} (already processed)")
        return
    
    total_chunks = (len(chrom_mutations) + chunk_size - 1) // chunk_size
    max_seg_idx = (total_chunks-1) // segment_size
    
    if not chrom_mutations:
        logging.warning(f"No mutations found for {label}-{chrom}. Skipping.")
        with open(global_done_file, "w") as f:
            f.write("done")
        return

    # Create task-specific temp directory
    task_tmp_dir = os.path.join(outdir, f"{label}_{chrom}_tmp")
    os.makedirs(task_tmp_dir, exist_ok=True)
    logging.debug(f"Using temp directory: {task_tmp_dir}")
    
    # Scan and verify state: get set of completed chunks and total row count
    completed_chunks, completed_segments, total_rows = scan_task_state(
        task_tmp_dir, segment_size, max_seg_idx, f'{label}-{chrom}', 
        cleanup_incomplete=True
    )

    # --- Identify segments that need processing ---
    # We only process a segment if it was NOT fully completed in scan_task_state
    # i.e., if any of its chunks is missing from completed_chunks
    segments_work = {}
    for chunk_idx in range(total_chunks):
        seg_idx = chunk_idx // segment_size

        if seg_idx not in segments_work:
            # Initialize the full chunk list for this segment
            start = seg_idx * segment_size
            end = min(start + segment_size, total_chunks)
            full_chunk_indices = list(range(start, end))
            segments_work[seg_idx] = {
                'full_chunk_indices': full_chunk_indices,
                'to_process': [],
                'already_done': []
            }
        
        if chunk_idx in completed_chunks:
            segments_work[seg_idx]['already_done'].append(chunk_idx)
        else:
            segments_work[seg_idx]['to_process'].append(chunk_idx)

    # Filter: Only keep segments that have work to do
    segments_to_process = {
        seg_idx: work for seg_idx, work in segments_work.items()
        if seg_idx not in completed_segments
    }

    n_segments_to_process = len(segments_to_process)
    n_segments_total = len(segments_work)
    base_args = (
        chrom_mutations,
        chunk_size,
        bam_file,
        min_mapq,
        mutation_to_gene,
        label,
        chrom,
        n_segments_total,
        task_tmp_dir,
        compression_method
    )

    if completed_chunks:
        logging.info(f"Found {len(completed_chunks)}/{total_chunks} chunks, {len(completed_segments)}/{n_segments_total} segments completed for {label}-{chrom}.")
    else:
        logging.debug(f"Found {len(completed_chunks)}/{total_chunks} chunks, {len(completed_segments)}/{n_segments_total} segments completed for {label}-{chrom}.")

    # Process segments
    try:
        if secondary_processes > 1 and n_segments_to_process > 1:
            logging.debug(f"Using {secondary_processes} processes for {label}-{chrom} task (processing {n_segments_to_process} segments)")
            
            segment_args = [
                (seg_idx, work['already_done'], work['to_process'], *base_args)
                for seg_idx, work in sorted(segments_to_process.items())
            ]

            with Pool(processes=secondary_processes) as pool:
                results = pool.starmap(process_segment, segment_args)
                new_rows = sum(results) if results else 0
        else:
            new_rows = 0
            for seg_idx, work in sorted(segments_to_process.items()):
                segment_rows = process_segment(
                    seg_idx, work['already_done'], work['to_process'], *base_args
                )
                new_rows += segment_rows
        
        total_rows += new_rows

        # Merge and mark task complete
        merge_parquets_for_task(f"{label}_{chrom}", outdir, compression_method)
        with open(global_done_file, "w") as f:
            f.write(f"total_rows:{total_rows}\n")
            f.write(f"total_chunks: {total_chunks}\n")
            f.write(f"total_segments: {n_segments_total}\n")
        
        elapsed = format_seconds(time.time() - start_time)
        logging.info(f"Finished {label} - {chrom}: Processed up to {total_chunks} chunks, "
                     f"{total_rows} total records ({elapsed})")
    
    except Exception as e:
        elapsed = format_seconds(time.time() - start_time)
        logging.error(f"Error processing {label} - {chrom} ({elapsed}): {str(e)}")
        raise

def merge_parquets_for_task(task, outdir, compression_method):
    """Merge all segment files for a task (label-chrom)"""
    # Collect all task temp directories
    task_tmp_dir = os.path.join(outdir, f"{task}_tmp")
    
    if not os.path.exists(task_tmp_dir):
        logging.warning(f"No task temp directory found for {task}")
        return
    
    segment_files = []
    for f in os.listdir(task_tmp_dir):
        if f.startswith(SEGMENT_PREFIX) and f.endswith(PARQUET_SUFFIX):
            segment_files.append(os.path.join(task_tmp_dir, f))
    
    if not segment_files:
        logging.warning(f"No segment files found for {task}")
        return
    
    # Merge all segment files
    dataset = pq.ParquetDataset(segment_files, use_legacy_dataset=False)
    table = dataset.read(columns=DF_COLUMNS)
    
    final_path = os.path.join(outdir, f"{task}{PARQUET_SUFFIX}")
    pq.write_table(table, final_path, compression=compression_method)
    
    # Cleanup temp directories
    try:
        shutil.rmtree(task_tmp_dir)
        logging.debug(f"Cleaned up temp directory: {task_tmp_dir}")
    except Exception as e:
        logging.warning(f"Failed to remove {task_tmp_dir}: {e}")
    
    logging.info(f"Merged {len(segment_files)} segments into {final_path}")

def merge_parquets_for_label(label, outdir, compression_method):
    """Merge all task files for a label"""
    all_files = os.listdir(outdir)
    matching_files = [
        f for f in all_files 
        if f.startswith(f"{label}_") and f.endswith(".parquet")
    ]
    if not matching_files:
        logging.warning(f"No parquet files found to merge for {label}")
        return

    file_paths = [os.path.join(outdir, f) for f in matching_files]
    dataset = pq.ParquetDataset(file_paths, use_legacy_dataset=False)
    table = dataset.read(columns=["CellBarcode", "Mutation", "Gene", "AF", "ALT", "REF"])
    
    final_path = os.path.join(outdir, f"Merged_{label}.parquet")
    pq.write_table(table, final_path, compression=compression_method)
    
    for f in file_paths:
        try:
            os.remove(f)
        except Exception as e:
            logging.warning(f"Failed to remove {f}: {e}")
            
    logging.info(f"Merged {len(matching_files)} files into {final_path}")

def filter_completed_tasks(tasks):
    """Filter out already completed tasks based on global done file"""
    pending_tasks = []
    for task in tasks:
        global_done_file = os.path.join(task['outdir'], f"{task['label']}_{task['chrom']}{DONE_SUFFIX}")
        if not os.path.exists(global_done_file):
            pending_tasks.append(task)
        else:
            logging.info(f"Skip completed task: {task['label']} - {task['chrom']} (already processed)")
    return pending_tasks

def get_remaining_workload(task_tmp_dir, total_chunks, segment_size, tag):
    """Calculate real-to-process workload"""
    if not os.path.exists(task_tmp_dir):
        return total_chunks
    
    max_seg_idx = (total_chunks-1) // segment_size
    completed_chunks, completed_segments, total_rows = scan_task_state(
        task_tmp_dir, segment_size, max_seg_idx, tag,
        cleanup_incomplete=False
    )
    done_count = len(completed_chunks)
    return max(0, total_chunks - done_count)

def distribute_secondary_processes(tasks, total_processes, method="balanced"):
    """
    Distribute secondary processes to pending tasks based on workload.
    
    Parameters:
    -----------
    tasks : List[Dict]
        List of task dicts with 'workload' and 'total_chunks' fields.
    total_processes : int
        Total number of secondary processes available (from --processes).
    method : str
        Allocation method:
        - "balanced": base = total_processes // n_tasks, then distribute remainder by weight
        - "proportional": each task gets at least 1, rest distributed by workload proportion
    
    Returns:
    --------
    None (modifies tasks in place, sets task['secondary_processes'])
    """
    n_tasks = len(tasks)
    if n_tasks == 0:
        return
    
    # Initialize with base of 1
    for task in tasks:
        task["secondary_processes"] = 1

    if total_processes < n_tasks:
        logging.warning(f"Not enough processes ({total_processes}) for {n_tasks} tasks. Assigning 1 each.")
        return
    if total_processes == n_tasks:
        logging.debug(f"Due to equal processes ({total_processes}) to {n_tasks} tasks, assigning 1 each.")
        return
    
    total_workload = sum(task["workload"] for task in tasks)
    if total_workload == 0:
        base = max(1, total_processes // n_tasks)
        remainder = total_processes - n_tasks * base
        for i, task in enumerate(tasks):
            task["secondary_processes"] = base + (1 if i < remainder else 0)
        return

    # We'll compute additional processes for each task
    if method == "proportional":
        # Each task gets at least 1 -> reserve n_tasks
        remaining_after_base = total_processes - n_tasks

        # Step 1: floor allocation based on weight
        weights = [task["workload"] / total_workload for task in tasks]
        raw_shares = [remaining_after_base * w for w in weights]
        additional = [int(s) for s in raw_shares]  # floor
        allocated = sum(additional)

        # Step 2: distribute the remainder by fractional part
        remainder = remaining_after_base - allocated
        if remainder > 0:
            # Sort by fractional part descending
            fractional_parts = [rs % 1 for rs in raw_shares]
            sorted_indices = sorted(
                range(len(fractional_parts)),
                key=lambda i: fractional_parts[i],
                reverse=True
            )
            for i in range(remainder):
                idx = sorted_indices[i]
                additional[idx] += 1

        additional_allocations = additional

    elif method == "balanced":
        # Base allocation: divide evenly
        base_processes = max(1, total_processes // n_tasks)
        for task in tasks:
            task["secondary_processes"] = base_processes

        remaining_after_base = total_processes - n_tasks * base_processes

        # Distribute remainder by weight (floor only, no round)
        weights = [task["workload"] / total_workload for task in tasks]
        raw_shares = [remaining_after_base * w for w in weights]
        additional = [int(s) for s in raw_shares]
        allocated = sum(additional)

        # Fill remainder by fractional part
        remainder = remaining_after_base - allocated
        if remainder > 0:
            fractional_parts = [rs % 1 for rs in raw_shares]
            sorted_indices = sorted(
                range(len(fractional_parts)),
                key=lambda i: fractional_parts[i],
                reverse=True
            )
            for i in range(remainder):
                idx = sorted_indices[i]
                additional[idx] += 1

        additional_allocations = additional

    else:
        raise ValueError(f"Unknown method: {method}. Use 'balanced' or 'proportional'.")

    # Apply additional allocations
    for task, add in zip(tasks, additional_allocations):
        task["secondary_processes"] += add

        # Safe clamp to valid range
        task["secondary_processes"] = min(task["secondary_processes"], total_processes)
        logging.debug(
            f"Task {task['label']}-{task['chrom']} allocated {task['secondary_processes']} "
            f"secondary processes (workload: {task['workload']}/{task['total_chunks']} chunks) [{method}]"
        )

    total_allocated = sum(task["secondary_processes"] for task in tasks)
    logging.debug(f"Process distribution complete. Total allocated: {total_allocated} (target: {total_processes}) [{method}]")


# main
def parse_args():
    """Parse command-line arguments for parallel BAM and mutation processing."""
    parser = argparse.ArgumentParser(
        description="Parallel BAM + Mutation Processing",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Input files
    parser.add_argument("--bams", required=True,
                        help="List of BAM files and labels. Format: file1,label1(;file2,label2)")
    parser.add_argument("--vcfs", required=True,
                        help="List of (bgzipped) VCF file containing mutations. Format: vcf1(;vcf2)")
    parser.add_argument("--gtf", default=None,
                        help="GTF annotation file (optional)")

    # Output
    parser.add_argument("--outdir", required=True,
                        help="Output directory for Parquet files")
    parser.add_argument("--log-file", default="GetAF_run.log",
                        help="Name of the log file, saved in --outdir")

    # Processing options
    parser.add_argument("--processes", type=int, default=os.cpu_count(),
                        help="Number of processes to use (default: number of CPUs)")
    parser.add_argument("--chunk-size", type=int, default=50,
                        help="Number of mutations per chunk (default: 50)")
    parser.add_argument("--segment-size", type=int, default=50,
                        help="Number of chunks per segment file (default: 50)")
    parser.add_argument("--min-mapq", type=int, default=30,
                        help="Minimum mapping quality for reads (default: 30)")
    parser.add_argument("--chromosomes", nargs='+', default=None,
                        help="List of chromosomes to process (optional)")
    parser.add_argument("--method", choices=["balanced", "proportional"], default="balanced",
                        help="Method for distributing secondary processes among tasks. "
                             "'balanced': base allocation then distribute remainder; "
                             "'proportional': each task gets at least 1, rest by workload proportion.")
    parser.add_argument("--dry-run", action="store_true",
                        help="If set, print task distribution without executing any processing.")
    parser.add_argument("--version", action="version", version=f"{TOOL_NAME} {__version__}")

    return parser.parse_args()

def setup_logging(outdir, log_file="GetAF_run.log"):
    """
    Configure logging to show INFO+ on screen and all logs (including DEBUG) in a file.
    """
    # Full path for log file
    log_path = os.path.join(outdir, log_file)

    # Create logger
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)  # Lowest level to capture everything

    # Formatter
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")

    # Console handler: only show INFO and above
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler: write all levels (DEBUG and above)
    try:
        file_handler = logging.FileHandler(log_path)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        logging.info(f"Logging to file: {log_path}")
    except Exception as e:
        logging.warning(f"Failed to create log file at {log_path}: {e}. Only logging to console.")

def main():
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    setup_logging(args.outdir, args.log_file)

    use_vcfs = [x for x in args.vcfs.split(";") if x]
    if len(use_vcfs) > 1:
        mutations = merge_vcf_mutations(use_vcfs)
    else:
        mutations = read_vcf_mutations(use_vcfs[0])

    mutations_by_chrom = {}
    for mut in mutations:
        chrom = mut["chrom"]
        pos = mut["position"] - 1
        key = (chrom, pos)
        mutations_by_chrom.setdefault(chrom, OrderedDict()).setdefault(key, []).append(mut)

    COMPRESSION_METHOD = detect_compression_method()
    logging.info(f"Using compression method: {COMPRESSION_METHOD}")

    mutation_to_gene = {}
    if args.gtf:
        if not os.path.exists(args.gtf):
            raise FileNotFoundError(f"GTF file not found: {args.gtf}")
        logging.info(f"Read GTF from {args.gtf}...")
        gtf = pr.read_gtf(args.gtf)
        gtf_genes = gtf[gtf.Feature == "gene"]

        logging.info(f"Annotate gene...")
        all_chroms = [mut["chrom"] for mut in mutations]
        all_positions = [mut["position"] for mut in mutations]
        genes_list = annotate_positions_with_gene(all_chroms, all_positions, gtf_genes)

        for mut, gene in zip(mutations, genes_list):
            key = (mut["chrom"], mut["position"])
            mutation_to_gene[key] = gene

        del gtf_genes
        import gc
        gc.collect()

    bam_files_info = [
        {"filename": p.split(",")[0], "label": p.split(",")[1]}
        for p in args.bams.split(";")
        if p
    ]

    # Generate all tasks
    tasks = []
    for info in bam_files_info:
        bam_file = info["filename"]
        label = info["label"]

        chromosomes = get_chromosomes(bam_file)
        if args.chromosomes:
            if len(args.chromosomes) == 1:
                args.chromosomes = args.chromosomes[0].split()
            chromosomes = [c for c in chromosomes if c in args.chromosomes]

        for chrom in chromosomes:
            if chrom in mutations_by_chrom and mutations_by_chrom[chrom]:
                tasks.append({
                    "bam_file": bam_file,
                    "label": label,
                    "chrom": chrom,
                    "outdir": args.outdir,
                    "min_mapq": args.min_mapq
                })
            else:
                logging.warning(f"Skipping chromosome {chrom} for {label}: No mutations found in VCF.")

    # Filter completed tasks and calculate workloads
    pending_tasks = filter_completed_tasks(tasks)
    logging.info(f"Found {len(pending_tasks)} pending tasks out of {len(tasks)} total tasks.")
    
    use_chroms_data = {}
    for chrom in set([task['chrom'] for task in pending_tasks]):
        chrom_mutations = [
            mut 
            for (c, p), mut_list in mutations_by_chrom[chrom].items()
            for mut in mut_list
        ]

        chrom_mutation_to_gene = {}
        for mut in chrom_mutations:
            key = (mut["chrom"], mut["position"])
            if key in mutation_to_gene:
                chrom_mutation_to_gene[key] = mutation_to_gene[key]

        use_chroms_data[chrom] = {
            'mutation': chrom_mutations,
            'gene': chrom_mutation_to_gene
        }

    # Calculate workloads for pending tasks
    logging.info(f"Calculating the remaining workload for each task...")
    pending_tasks_with_chromMut = []
    for task in pending_tasks:
        chrom_mutations = use_chroms_data[task['chrom']]['mutation']
        chrom_mutation_to_gene = use_chroms_data[task['chrom']]['gene']

        # Calculate workload for this task
        total_chunks = (len(chrom_mutations) + args.chunk_size - 1) // args.chunk_size
        task["total_chunks"] = total_chunks
        
        # Get remaining workload
        task_tmp_dir = os.path.join(task['outdir'], f"{task['label']}_{task['chrom']}_tmp")
        tag = f"{task['label']}_{task['chrom']}"
        remaining_workload = get_remaining_workload(
            task_tmp_dir, total_chunks, args.segment_size, tag
        )
        task["workload"] = remaining_workload

        pending_tasks_with_chromMut.append(
            (task, chrom_mutations, chrom_mutation_to_gene)
        ) # here task from pending_tasks_with_chromMut is the same quote as one from pending_tasks
        if remaining_workload == 0:
            logging.info(f"Task {task['label']}-{task['chrom']} has no remaining work, will be skipped.")

    # Calculate secondary processes for each pending task
    n_pending = len(pending_tasks)
    if n_pending > 0:
        logging.info(f"Running with {args.processes} processes using '{args.method}' allocation strategy.")
        distribute_secondary_processes(pending_tasks, args.processes, method=args.method)
    else:
        logging.info("No pending tasks to process.")
        return

    logging.info("Using 0-based segment indexing internally (e.g., segment_0.parquet). "
             "Progress logs show 1-based numbering for readability (e.g., Segment 1/100).")

    if args.dry_run:
        logging.info("Skipped processing and merging due to --dry-run. Exited.")
        return

    func = partial(
        process_task, 
        compression_method=COMPRESSION_METHOD,
        chunk_size=args.chunk_size,
        segment_size=args.segment_size,
    )
    with NoDaemonProcessPool(processes=args.processes) as pool:
        pool.starmap(func, pending_tasks_with_chromMut)

    for info in bam_files_info:
        label = info["label"]
        logging.info(f"Merge parquets for {label}...")
        merge_parquets_for_label(label, args.outdir, COMPRESSION_METHOD)

    logging.info("All tasks completed.")


if __name__ == "__main__":
    main()
