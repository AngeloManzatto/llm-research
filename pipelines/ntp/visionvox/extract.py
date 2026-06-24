"""
Created on Sun Jun 22 08:16:30 2025

@author: Angelo Antonio Manzatto
"""

###############################################################################
# Libraries
###############################################################################

import re
import json

from pathlib import Path
import unicodedata
import hashlib

from pipelines.ntp.common.logger import setup_logger
from pipelines.ntp.common.io_utils import read_file

from concurrent.futures import ThreadPoolExecutor
from threading import Lock

###############################################################################
# Globals
###############################################################################

dataset_id = "visionvox"

global_shard_index = 0
shard_lock = Lock()

logger = setup_logger(pipeline_name="visionvox", log_to_stdout=False)

###############################################################################
# Files and Folders
###############################################################################

input_dir =  Path("data") / "ntp" / "raw"  / dataset_id  
output_dir = Path("data") / "ntp" / "extracted" / dataset_id 

###############################################################################
# Pipeline functions
###############################################################################

def sanitize_name(name):
    # 1. Normalize Unicode (remove accents)
    name = unicodedata.normalize("NFKD", name)
    name = name.encode("ascii", "ignore").decode("ascii")

    # 2. Replace non-word characters (excluding underscore) with underscore
    name = re.sub(r"[^\w]", "_", name)

    # 3. Collapse multiple underscores
    name = re.sub(r"_+", "_", name)

    # 4. Trim leading/trailing underscores
    name = name.strip("_")

    return name

def process_folder(folder_path: Path, output_dir: Path, shard_size=1_000):
    global global_shard_index

    logger.info(f"📂 Processing folder: {folder_path}")
    files = sorted(folder_path.rglob("*.txt"))
    docs = []
    doc_counter = 0

    for file_path in files:
        try:
            text = read_file(file_path, encoding="latin-1")
            text = text.replace("\r\n", "\n").strip()  # Normalize newlines + strip
            title = sanitize_name(file_path.stem)
            rel_path = file_path.relative_to(folder_path.parent)
            doc_id = hashlib.md5(str(rel_path).encode("utf-8")).hexdigest()
            doc_hash = hashlib.md5(text.encode("utf-8")).hexdigest()

            docs.append({
                "id": doc_id,
                "title": title,
                "content": text,
                "hash": doc_hash,
                "source_file": str(rel_path)
            })
            doc_counter += 1
            
            while len(docs) >= shard_size:
                with shard_lock:
                    shard_idx = global_shard_index
                    global_shard_index += 1
            
                shard_docs = docs[:shard_size]
                docs = docs[shard_size:]
            
                shard_path = output_dir / f"visionvox_{shard_idx:05d}.jsonl"
                with open(shard_path, "w", encoding="utf-8") as fout:
                    for d in shard_docs:
                        fout.write(json.dumps(d, ensure_ascii=False) + "\n")
            
                logger.success(f"📝 Shard saved: {shard_path.name} ({len(shard_docs)} docs)")


        except Exception as e:
            logger.error(f"⚠️ Error processing {file_path}: {e}")

    # Final partial shard
    if docs:
        with shard_lock:
            shard_idx = global_shard_index
            global_shard_index += 1

        shard_path = output_dir / f"visionvox_{shard_idx:05d}.jsonl"
        with open(shard_path, "w", encoding="utf-8") as fout:
            for d in docs:
                fout.write(json.dumps(d, ensure_ascii=False) + "\n")
        logger.success(f"📝 Final shard saved: {shard_path.name} ({len(docs)} docs)")

    logger.info(f"✅ Done: {doc_counter} documents from folder {folder_path.name}")

###############################################################################
# Parallel extraction
###############################################################################
def run_parallel_extraction(input_root, output_dir, max_threads=4):
    input_root = Path(input_root)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    folder_paths = sorted([p for p in input_root.iterdir() if p.is_dir()])

    with ThreadPoolExecutor(max_workers=max_threads) as executor:
        for folder in folder_paths:
            executor.submit(process_folder, folder, output_dir)
            
run_parallel_extraction(input_dir, output_dir)