"""
Created on Thu Jun 26 07:52:55 2025

@author: Angelo Antonio Manzatto
"""

###############################################################################
# Libraries
###############################################################################

import json
import regex
import hashlib
from pathlib import Path

from tqdm import tqdm
import wikitextparser as wtp
from langdetect import detect
from concurrent.futures import ProcessPoolExecutor

from pipelines.ntp.common.text_processing.text_cleaning import clean_text
from pipelines.ntp.common.text_processing.text_filtering  import filter_text
from pipelines.ntp.common.text_processing.text_quality import compute_quality_metrics

from pipelines.ntp.common.logger import setup_logger
from pipelines.ntp.common.io_utils import read_file

###############################################################################
# Globals
###############################################################################

dataset_id = "wikipedia"

ALLOWED_LANGUAGES = {"en", "pt"}
shard_size = 10_000
workers = 6

logger = setup_logger(pipeline_name=dataset_id, log_to_stdout=False)

###############################################################################
# Files and Folders
###############################################################################

input_dir =  Path("data") / "ntp" / "extracted" / dataset_id 
output_dir = Path("data") / "ntp" / "processed" / dataset_id 
output_dir.mkdir(parents=True, exist_ok=True)

###############################################################################
# Worker Function
###############################################################################

def clean_wikipedia_markup(text, remove_templates=True, remove_links=True):
    parsed = wtp.parse(text)

    # Optionally remove templates
    if remove_templates:
        for tmpl in parsed.templates:
            text = text.replace(tmpl.string, "")
    
    # Optionally remove links like [[Page|label]]
    if remove_links:
        text = parsed.plain_text()  # removes all wiki markup, including links and formatting
    else:
        # Only remove templates and keep links
        text = text

    # Final cleanup: remove extra newlines/whitespace
    text = regex.sub(r'\n{3,}', '\n\n', text)
    text = text.strip()

    return text

def process_document(doc):
    """
    Process a single document:
    - Clean text
    - Detect language
    - Filter unwanted content
    - Compute quality metrics
    - Generate final document structure
    Returns None if document should be skipped.
    """
    text = doc.get("content", "")
    if not text.strip():
        return None
    
    # Clean markup (Wikipedia templates)
    parsed_text = clean_wikipedia_markup(text)

    # Clean
    cleaned = clean_text(parsed_text)

    # Language detection
    try:
        lang = detect(cleaned)
    except Exception:
        return None

    if lang not in ALLOWED_LANGUAGES:
        return None

    # Filtering
    if not filter_text(cleaned):
        return None

    # Metrics
    metrics = compute_quality_metrics(cleaned, lang=lang)

    # Hash for deduplication
    doc_hash = hashlib.sha256(cleaned.encode("utf-8")).hexdigest()

    # Final structure
    return {
        "id": doc.get("id", ""),
        "title": doc.get("title", ""),
        "url": doc.get("url", ""),
        "source_file": doc.get("source_file", ""),
        "lang": lang,
        "content": cleaned,
        "metrics": metrics,
        "hash": doc_hash
    }

# Chunking utility
def chunked(iterable, size):
    """Yield successive chunks of given size."""
    for i in range(0, len(iterable), size):
        yield iterable[i:i + size]

# Batch processor
def process_batch(docs):
    return [doc for doc in map(process_document, docs) if doc is not None]


# Main shard processor
def process_shard(file_path, workers=4, batch_size=10):
    """
    Processes a single .jsonl shard file using multiprocessing with batched documents.
    Returns a list of cleaned and filtered documents.
    """
    logger.info(f" Processing shard: {file_path}")
    file_path = Path(file_path)
    docs = read_file(file_path, encoding="utf-8")
    total = len(docs)

    kept_docs = []

    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = []
        for batch in chunked(docs, batch_size):
            futures.append(executor.submit(process_batch, batch))

        for f in tqdm(futures, desc=f" {file_path.name}", total=len(futures)):
            try:
                batch_result = f.result()
                kept_docs.extend(batch_result)
            except Exception as e:
                logger.error(f"⚠️ Error in batch: {e}")

    logger.success(f"✅ Shard done: {file_path.name} | Kept {len(kept_docs)}/{total} docs")
    
    return kept_docs

def run_pipeline(input_dir, output_dir, shard_size=1_000, workers=4, batch_size=10):
    """
    Processes all input .jsonl shards with multiprocessing and writes fixed-size output shards.
    Supports resumption by skipping already-written output shards.
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Detect already written shards
    existing_shards = sorted(output_dir.glob("*.jsonl"))
    shard_index = len(existing_shards)
    doc_counter = shard_index * shard_size
    logger.info(f" Resuming at shard {shard_index:05d} (starting from doc #{doc_counter})")

    # Gather input files
    input_files = sorted(input_dir.glob("*.jsonl"))

    # Initialize buffer
    shard_docs = []

    for file_path in input_files:
        processed_docs = process_shard(file_path, workers=workers, batch_size=batch_size)

        for doc in processed_docs:
            if doc_counter < shard_index * shard_size:
                doc_counter += 1
                continue  # Skip already written docs

            shard_docs.append(doc)
            doc_counter += 1

            if len(shard_docs) >= shard_size:
                # Save current shard
                shard_path = output_dir / f"{pipeline_name}_{shard_index:05d}.jsonl"
                with open(shard_path, "w", encoding="utf-8") as fout:
                    for d in shard_docs:
                        fout.write(json.dumps(d, ensure_ascii=False) + "\n")

                logger.success(f" Shard saved: {shard_path.name} ({len(shard_docs)} docs)")
                shard_docs = []
                shard_index += 1

    # Save final partial shard
    if shard_docs:
        shard_path = output_dir / f"{pipeline_name}_{shard_index:05d}.jsonl"
        with open(shard_path, "w", encoding="utf-8") as fout:
            for d in shard_docs:
                fout.write(json.dumps(d, ensure_ascii=False) + "\n")

        logger.success(f" Final shard saved: {shard_path.name} ({len(shard_docs)} docs)")

    logger.info("✅ All documents processed and sharded.")

if __name__ == "__main__":
    run_pipeline(
        input_dir=input_dir,
        output_dir=output_dir,
        shard_size=shard_size,
        workers=workers,
        batch_size=100
    )