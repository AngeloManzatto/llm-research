"""
Created on Sat Jun 21 21:33:33 2025

@author: recruta42
"""

###############################################################################
# Libraries
###############################################################################

import os
import json
import hashlib
import xml.etree.ElementTree as ET
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from threading import Lock
from pipelines.ntp.common.io_utils import open_compressed_file
from pipelines.ntp.common.logger import setup_logger

###############################################################################
# Globals
###############################################################################

dataset_id = "wikipedia"

logger = setup_logger(dataset_id, log_to_stdout=False)

global_shard_index = 0
max_threads = 6
shard_lock = Lock()

###############################################################################
# Files and Folders
###############################################################################

input_dir =  Path("data") / "ntp" / "raw"  / dataset_id  
output_dir = Path("data") / "ntp" / "extracted" / dataset_id 

###############################################################################
# Process zip
###############################################################################
def extract_documents(file_path, max_articles=None, log_interval=1000):
    with open_compressed_file(file_path) as xml_file:
        logger.info(f"📂 Extracting from: {file_path}")

        context = ET.iterparse(xml_file, events=("start", "end"))
        context = iter(context)
        _, root = next(context)

        yielded_count = 0

        for event, elem in context:
            if event == "end" and elem.tag.endswith("page"):
                try:
                    title = elem.findtext("./{*}title")
                    text = elem.findtext("./{*}revision/{*}text")
                    page_id = elem.findtext("./{*}id")

                    if title and text:
                        text = text.strip()
                        doc_hash = hashlib.md5(text.encode("utf-8")).hexdigest()

                        yield {
                            "id": page_id,
                            "title": title,
                            "content": text,
                            "hash": doc_hash,
                            "source_file": os.path.basename(file_path)
                        }

                        yielded_count += 1
                        if yielded_count % log_interval == 0:
                            logger.info(f"✅ {yielded_count} articles yielded...")

                        if max_articles and yielded_count >= max_articles:
                            break

                except Exception as e:
                    logger.error(f"⚠️ Error processing element: {e}")
                finally:
                    elem.clear()

        logger.info(f"✅ Done: {yielded_count} articles extracted from {file_path}")
        
###############################################################################
# Worker for multiprocessing
###############################################################################
def process_file(file_path: Path, output_dir: Path, shard_size=10_000):
    global global_shard_index
    docs = []
    doc_counter = 0

    for doc in extract_documents(file_path):
        docs.append(doc)
        doc_counter += 1

        if len(docs) >= shard_size:
            with shard_lock:
                shard_idx = global_shard_index
                global_shard_index += 1

            shard_path = output_dir / f"wikipedia_{shard_idx:05d}.jsonl"
            with open(shard_path, "w", encoding="utf-8") as fout:
                for d in docs:
                    fout.write(json.dumps(d, ensure_ascii=False) + "\n")
            logger.success(f"📝 Shard saved: {shard_path.name} ({len(docs)} docs)")
            docs = []

    if docs:
        with shard_lock:
            shard_idx = global_shard_index
            global_shard_index += 1

        shard_path = output_dir / f"wikipedia_{shard_idx:05d}.jsonl"
        with open(shard_path, "w", encoding="utf-8") as fout:
            for d in docs:
                fout.write(json.dumps(d, ensure_ascii=False) + "\n")
        logger.success(f"📝 Final shard saved: {shard_path.name} ({len(docs)} docs)")

    logger.info(f"✅ Finished processing file: {file_path.name} ({doc_counter} documents)")
    
def run_parallel_extraction(input_dir, output_dir, max_threads=4):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    file_paths = sorted(input_dir.glob("*.bz2"))

    with ThreadPoolExecutor(max_workers=max_threads) as executor:
        for file_path in file_paths:
            executor.submit(process_file, file_path, output_dir)
            
run_parallel_extraction(input_dir, output_dir, max_threads=max_threads)