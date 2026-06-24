"""
Created on Sun Dec 21 16:23:18 2025

@author: Angelo Antonio Manzatto
"""

###############################################################################
# Libraries
###############################################################################
import json
import logging
import hashlib
from pathlib import Path
from tqdm import tqdm

from concurrent.futures import ProcessPoolExecutor

import tensorflow as tf

from dataclasses import dataclass

from typing import Any, Dict, List, Optional

from pipelines.ntp.common.io_utils import read_file
from pipelines.ntp.common.logger import setup_logger

from src.core.tokenizer.tokenizer import BBPETokenizer

###############################################################################
# Tokenization pipeline
###############################################################################

@dataclass(frozen=True)
class TokenizeJob:
    dataset_id: str
    tokenizer_id: str

    input_dir: Path              # e.g. data/ntp/processed/<dataset_id>
    output_dir: Path             # e.g. data/ntp/tokenized/<dataset_id>/<tokenizer_id>

    tokenizer_checkpoint: Path   # path to bbpe_tokenizer_*.pkl

    shard_size: int = 10_000
    workers: int = 6

    # Optional behaviors
    keep_raw_text: bool = False  # usually False for NTP, but you can enable for debugging
    

_TOKENIZER: Optional[BBPETokenizer] = None
_WORKER_CFG: Optional[TokenizeJob] = None
_JOB_TOKENIZER_CHECKPOINT: str = ""
_JOB_KEEP_RAW_TEXT: bool = False

def _init_worker(tokenizer_checkpoint: str, keep_raw_text: bool):
    """
    Runs once per process. Loads tokenizer into a per-process global.
    """
    global _TOKENIZER, _WORKER_CFG
    _TOKENIZER = BBPETokenizer.load(Path(tokenizer_checkpoint))
    # store small flags only (avoid pickling huge objects)
    _WORKER_CFG = {
        "keep_raw_text": keep_raw_text,
    }

def _calculate_metrics(text: str, tokens: List[int]) -> Dict[str, Any]:
    token_count = len(tokens)
    return {
        "token_count": token_count,
        "avg_chars_per_token": (len(text) / token_count) if token_count else 0.0,
    }

def _process_document(doc: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    global _TOKENIZER, _WORKER_CFG
    assert _TOKENIZER is not None and _WORKER_CFG is not None, "Worker not initialized"

    text = doc.get("content", "")
    if not isinstance(text, str) or not text.strip():
        return None

    tokens = _TOKENIZER.text_to_indices(text)

    # Dedup hash (content-based)
    doc_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()

    # Metrics
    metrics = doc.get("metrics", {}) or {}
    metrics.update(_calculate_metrics(text, tokens))

    lang = doc.get("lang") or "unknown"
    
    out = {
        "id": doc.get("id") or doc_hash,
        "title": doc.get("title", ""),
        "url": doc.get("url", ""),
        "source_file": doc.get("source_file"),
        "lang": lang,
        "content": tokens,
        "metrics": metrics,
    }

    if _WORKER_CFG["keep_raw_text"]:
        out["raw_content"] = text

    return out

def _split_into_chunks(data: List[Any], num_chunks: int) -> List[List[Any]]:
    k, m = divmod(len(data), num_chunks)
    return [data[i*k + min(i, m):(i+1)*k + min(i+1, m)] for i in range(num_chunks)]

def _process_batch(docs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for d in docs:
        r = _process_document(d)
        if r is not None:
            out.append(r)
    return out

def _process_input_shard(file_path: Path, *, workers: int, logger) -> List[Dict[str, Any]]:
    logger.info(f"📂 Processing shard: {file_path}")
    docs = read_file(file_path, encoding="utf-8")
    total = len(docs) if docs else 0
    if not docs:
        return []

    # Split docs across workers into big chunks (keeps overhead low)
    doc_chunks = _split_into_chunks(docs, workers)

    kept: List[Dict[str, Any]] = []
    with ProcessPoolExecutor(
        max_workers=workers,
        initializer=_init_worker,
        initargs=(str(_JOB_TOKENIZER_CHECKPOINT), _JOB_KEEP_RAW_TEXT),
    ) as ex:
        futures = [ex.submit(_process_batch, chunk) for chunk in doc_chunks]
        for f in tqdm(futures, desc=f"🔄 {file_path.name}", total=len(futures)):
            try:
                kept.extend(f.result())
            except Exception as e:
                logger.error(f"⚠️ Error in chunk: {e}")

    logger.success(f"✅ Shard done: {file_path.name} | Kept {len(kept)}/{total} docs")
    return kept

def tokenization_pipeline(job: TokenizeJob) -> None:
    """
    Tokenize all processed JSONL shards under job.input_dir and write tokenized JSONL shards
    under job.output_dir, with resumption support (skip existing output shards).
    """
    global _JOB_TOKENIZER_CHECKPOINT, _JOB_KEEP_RAW_TEXT
    _JOB_TOKENIZER_CHECKPOINT = str(job.tokenizer_checkpoint)
    _JOB_KEEP_RAW_TEXT = job.keep_raw_text

    logger = setup_logger(pipeline_name=f"tokenize_{job.dataset_id}", log_to_stdout=False)

    job.output_dir.mkdir(parents=True, exist_ok=True)

    existing = sorted(job.output_dir.glob("*.jsonl"))
    shard_index = len(existing)
    doc_counter = shard_index * job.shard_size
    logger.info(f"🔁 Resuming at shard {shard_index:05d} (starting from doc #{doc_counter})")

    input_files = sorted(job.input_dir.glob("*.jsonl"))
    if not input_files:
        logger.warning(f"⚠️ No input shards found in: {job.input_dir}")
        return

    shard_docs: List[Dict[str, Any]] = []

    for file_path in input_files:
        processed_docs = _process_input_shard(file_path, workers=job.workers, logger=logger)

        for doc in processed_docs:
            # resume logic
            if doc_counter < shard_index * job.shard_size:
                doc_counter += 1
                continue

            shard_docs.append(doc)
            doc_counter += 1

            if len(shard_docs) >= job.shard_size:
                out_path = job.output_dir / f"{job.dataset_id}_{shard_index:05d}.jsonl"
                with open(out_path, "w", encoding="utf-8") as f:
                    for d in shard_docs:
                        f.write(json.dumps(d, ensure_ascii=False) + "\n")
                logger.success(f"📝 Shard saved: {out_path.name} ({len(shard_docs)} docs)")
                shard_docs = []
                shard_index += 1

    if shard_docs:
        out_path = job.output_dir / f"{job.dataset_id}_{shard_index:05d}.jsonl"
        with open(out_path, "w", encoding="utf-8") as f:
            for d in shard_docs:
                f.write(json.dumps(d, ensure_ascii=False) + "\n")
        logger.success(f"📝 Final shard saved: {out_path.name} ({len(shard_docs)} docs)")

    logger.info("✅ All documents processed and sharded.")
    
###############################################################################
# Pack pipeline
###############################################################################

@dataclass(frozen=True)
class PackJob:
    dataset_id: str
    tokenizer_id: str

    input_dir: Path          # tokenized dir
    output_dir: Path         # packed dir

    tokenizer_checkpoint: Path

    max_tokens_per_shard: int = 1_024_000
    append_eos: bool = True  # if True, appends EOS after each document
    
def _save_tfrecord(flat_tokens: List[int], output_path: Path) -> None:
    example = tf.train.Example(
        features=tf.train.Features(
            feature={"tokens": tf.train.Feature(int64_list=tf.train.Int64List(value=flat_tokens))}
        )
    )
    with tf.io.TFRecordWriter(str(output_path)) as writer:
        writer.write(example.SerializeToString())
        
def _append_packed_index_line(index_path: Path, shard_path: Path, num_tokens: int) -> None:
    index_path.parent.mkdir(parents=True, exist_ok=True)
    rec = {"path": str(shard_path), "tokens": int(num_tokens)}
    with open(index_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        
def _read_single_tfrecord_token_count(tfrecord_path: Path) -> int:
    """
    Each packed TFRecord contains exactly one Example with VarLenFeature 'tokens'.
    """
    ds = tf.data.TFRecordDataset(str(tfrecord_path))

    feature_description = {"tokens": tf.io.VarLenFeature(tf.int64)}

    def _parse(ex):
        x = tf.io.parse_single_example(ex, feature_description)
        return tf.sparse.to_dense(x["tokens"])

    tokens = next(iter(ds.map(_parse).take(1)))
    return int(tokens.shape[0])

def _ensure_packed_index_consistent(output_dir: Path) -> Path:
    """
    Ensures output_dir/packed_index.jsonl exists and matches the current packed_*.tfrecord set.
    If missing or line count mismatches, rebuild from existing TFRecords.
    """
    index_path = output_dir / "packed_index.jsonl"
    shards = sorted(output_dir.glob("packed_*.tfrecord"))

    if not shards:
        return index_path  # nothing to index yet

    def _count_lines(p: Path) -> int:
        if not p.exists():
            return 0
        with open(p, "r", encoding="utf-8") as f:
            return sum(1 for _ in f)

    current_lines = _count_lines(index_path)
    if current_lines == len(shards):
        return index_path  # already consistent

    # Rebuild (safe, deterministic)
    tmp = output_dir / "packed_index.jsonl.tmp"
    if tmp.exists():
        tmp.unlink()

    with open(tmp, "w", encoding="utf-8") as f:
        for shard in shards:
            n_tokens = _read_single_tfrecord_token_count(shard)
            f.write(json.dumps({"path": str(shard), "tokens": n_tokens}, ensure_ascii=False) + "\n")

    tmp.replace(index_path)
    return index_path
        
def _write_or_check_tokenization_meta(output_dir: Path, dataset_id: str, tokenizer_id: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    meta_path = output_dir / "tokenization_meta.json"

    if not meta_path.exists():
        meta_path.write_text(
            json.dumps({"dataset_id": dataset_id, "tokenizer_id": tokenizer_id}, indent=2),
            encoding="utf-8",
        )
        return

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if meta.get("dataset_id") != dataset_id or meta.get("tokenizer_id") != tokenizer_id:
        raise RuntimeError(
            f"Tokenization meta mismatch in {meta_path}. Refusing to mix outputs.\n"
            f"Expected dataset_id={dataset_id}, tokenizer_id={tokenizer_id}\n"
            f"Found    dataset_id={meta.get('dataset_id')}, tokenizer_id={meta.get('tokenizer_id')}"
        )

def pack_pipeline(job: PackJob) -> None:
    if not job.tokenizer_id:
        raise ValueError("tokenizer_id is required to avoid mixing packed datasets.")

    # Safety: ensure the packed output dir is pinned to dataset+tokenizer and has meta
    _write_or_check_tokenization_meta(job.output_dir, job.dataset_id, job.tokenizer_id)

    # Load tokenizer only to get EOS id (and optionally for debug decoding)
    tokenizer = BBPETokenizer.load(job.tokenizer_checkpoint)
    eos_token = tokenizer.token_to_index.get("<EOS>")
    if job.append_eos and eos_token is None:
        raise RuntimeError("append_eos=True but tokenizer has no '<EOS>' token.")

    job.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Ensure index is present & consistent with any existing shards (resume-safe)
    index_path = _ensure_packed_index_consistent(job.output_dir)

    # Resume support: continue shard_index after existing shards
    existing = sorted(job.output_dir.glob("packed_*.tfrecord"))
    shard_index = len(existing)

    token_stream: List[int] = []

    input_files = sorted(job.input_dir.glob("*.jsonl"))
    if not input_files:
        raise FileNotFoundError(f"No tokenized .jsonl files found in {job.input_dir}")

    for file_path in tqdm(input_files, desc="Reading tokenized shards"):
        with open(file_path, "r", encoding="utf-8") as fin:
            for line in fin:
                obj = json.loads(line)
                tokens = obj.get("content", [])
                if not tokens:
                    logging.warning(f"No tokens in content: {file_path}")
                    continue

                token_stream.extend(tokens)
                if job.append_eos:
                    token_stream.append(int(eos_token))

                # Emit shards while we have enough tokens
                while len(token_stream) >= job.max_tokens_per_shard:
                    chunk = token_stream[:job.max_tokens_per_shard]
                    save_path = job.output_dir / f"packed_{shard_index:05d}.tfrecord"
                    _save_tfrecord(chunk, save_path)
                    _append_packed_index_line(index_path, save_path, len(chunk))
                    shard_index += 1
                    token_stream = token_stream[job.max_tokens_per_shard:]

    # Save remaining tokens
    if token_stream:
        save_path = job.output_dir / f"packed_{shard_index:05d}.tfrecord"
        _save_tfrecord(token_stream, save_path)
        _append_packed_index_line(index_path, save_path, len(chunk))

    print("✅ Packing complete.")

def read_tfrecord_sample(tfrecord_path: Path, tokenizer_checkpoint: Path, num_examples: int = 3) -> None:
    """
    Debug helper: reads a TFRecord shard and prints a small preview + decoded text.
    """
    tokenizer = BBPETokenizer.load(tokenizer_checkpoint)

    raw_dataset = tf.data.TFRecordDataset(str(tfrecord_path))
    feature_description = {"tokens": tf.io.VarLenFeature(tf.int64)}

    def _parse(example_proto):
        return tf.io.parse_single_example(example_proto, feature_description)

    parsed_dataset = raw_dataset.map(_parse)

    print(f"🔍 Inspecting: {tfrecord_path}")
    for i, example in enumerate(parsed_dataset.take(num_examples)):
        dense_tokens = tf.sparse.to_dense(example["tokens"])
        tokens = dense_tokens.numpy().tolist()

        preview = tokens[:50]
        print(f"Example {i+1}: shape={dense_tokens.shape}, first10={preview[:10]}...")
        print(f"Decoded preview: {tokenizer.indices_to_text(preview)}")