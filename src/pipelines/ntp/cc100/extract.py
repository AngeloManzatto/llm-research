"""
Created on Sat Oct  4 17:55:03 2025

@author: Angelo Antonio Manzatto
"""

###############################################################################
# Libraries
###############################################################################
from datasets import load_dataset
import json, hashlib
from pathlib import Path
import re

###############################################################################
# Files and folders
###############################################################################
languages = ["en","pt"]
split = "train"
dataset_id = "cc100"

input_dir  = Path("data") / "ntp" / "raw" / dataset_id / split
output_dir = Path("data") / "ntp" / "extracted" / dataset_id / split

input_dir.mkdir(parents=True, exist_ok=True)
output_dir.mkdir(parents=True, exist_ok=True)

###############################################################################
# Create raw cache in disk
###############################################################################
def clean_text(t: str) -> str:
    return re.sub(r"\s+", " ", (t or "").strip())

def clean(t): return re.sub(r"\s+", " ", (t or "").strip())

def stream_cc100_to_jsonl(out_dir: Path, shard_docs=100_000, seed=42):
    out_dir.mkdir(parents=True, exist_ok=True)
    
    for language in languages:
    
        ds = load_dataset("cc100", language, split=split, streaming=True).shuffle(seed=seed, buffer_size=100_000)
    
        shard_idx, in_shard, total = 0, 0, 0
        fout = open(out_dir / f"cc100_pt_{shard_idx:05d}.jsonl", "w", encoding="utf-8")
        try:
            for row in ds:
                txt = clean(row.get("text"))
                if len(txt) < 100:  # skip trivial
                    continue
                _id = hashlib.sha256(txt.encode("utf-8")).hexdigest()
                
                rec = {
                    "id": _id, 
                    "lang": language, 
                    "content": txt
                    }
                
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                total += 1; in_shard += 1
                
                if in_shard >= shard_docs:
                    fout.close()
                    shard_idx += 1; in_shard = 0
                    fout = open(out_dir / f"cc100_{language}_{shard_idx:05d}.jsonl", "w", encoding="utf-8")
        finally:
            fout.close()
        print(f"✅ Streamed {total:,} docs into {shard_idx+1} jsonl shard(s) at {out_dir}")
    
stream_cc100_to_jsonl(output_dir)

