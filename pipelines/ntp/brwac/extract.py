"""
Created on Mon Dec 22 16:30:17 2025

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
# Settings
###############################################################################
dataset_id = "brwac"
split = "train"

###############################################################################
# Folders
###############################################################################
output_dir = Path("data") / "ntp" / "extracted" / dataset_id / split
output_dir.mkdir(parents=True, exist_ok=True)

###############################################################################
# Helpers
###############################################################################
def clean(t: str) -> str:
    return re.sub(r"\s+", " ", (t or "").strip())

###############################################################################
# Stream brWaC to JSONL
###############################################################################

def flatten_text_field(text_field) -> str:
    # Some BrWaC formats use {"paragraphs": [[sent1, sent2], ...]}
    if isinstance(text_field, str):
        return text_field
    if isinstance(text_field, dict) and "paragraphs" in text_field:
        paras = text_field["paragraphs"] or []
        parts = []
        for p in paras:
            if isinstance(p, list):
                parts.append(" ".join([s for s in p if isinstance(s, str)]))
            elif isinstance(p, str):
                parts.append(p)
        return "\n".join(parts)
    return ""

def stream_brwac_to_jsonl(
    out_dir: Path,
    *,
    shard_docs: int = 10_000,
    min_chars: int = 100,
    max_docs: int | None = None,
):
    ds = load_dataset(
        "dominguesm/brwac",
        split=split,
        streaming=True,
    )

    shard_idx, in_shard, total = 0, 0, 0
    fout = open(out_dir / f"{dataset_id}_pt_{shard_idx:05d}.jsonl", "w", encoding="utf-8")

    try:
        for row in ds:
            txt = clean(flatten_text_field(row.get("text")))
            if len(txt) < min_chars:
                continue

            _id = hashlib.sha256(txt.encode("utf-8")).hexdigest()
            rec = {"id": _id, "lang": "pt", "content": txt}

            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            total += 1
            in_shard += 1

            if max_docs is not None and total >= max_docs:
                break

            if in_shard >= shard_docs:
                fout.close()
                shard_idx += 1
                in_shard = 0
                fout = open(out_dir / f"{dataset_id}_pt_{shard_idx:05d}.jsonl", "w", encoding="utf-8")

    finally:
        fout.close()

    print(f"✅ Streamed {total:,} PT docs into {shard_idx+1} shard(s) at {out_dir}")

stream_brwac_to_jsonl(output_dir, shard_docs=10_000, max_docs=None)