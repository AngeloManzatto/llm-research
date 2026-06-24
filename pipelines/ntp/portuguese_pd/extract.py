"""
Created on Fri Jan  2 19:25:19 2026

@author: Angelo Antonio Manzatto
"""

###############################################################################
# Libraries
###############################################################################
import json, hashlib, re
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from huggingface_hub import HfFileSystem

###############################################################################
# Settings
###############################################################################
dataset_id = "portuguese_pd"
split = "train"

###############################################################################
# Output folders
###############################################################################
out_dir = Path("data") / "ntp" / "extracted" / dataset_id / split
out_dir.mkdir(parents=True, exist_ok=True)

###############################################################################
# Helpers
###############################################################################
def clean(t: str) -> str:
    return re.sub(r"\s+", " ", (t or "").strip())

###############################################################################
# Stream Portuguese-PD
###############################################################################
def stream_portuguese_pd_parquet_to_jsonl(
    out_dir: Path,
    shard_docs: int = 100_000,
    min_chars: int = 100,
    revision: str | None = None,  # e.g. "d8d45f37105a975b84ad8fe49a0d574a552178c8"
):
    fs = HfFileSystem()

    base = "datasets/PleIAs/Portuguese-PD"
    if revision:
        base = f"{base}@{revision}"

    # list parquet files in repo root (adjust if dataset stores them in subfolders)
    files = fs.ls(base, detail=False)
    parquet_files = sorted([p for p in files if p.endswith(".parquet")])

    if not parquet_files:
        raise RuntimeError(f"No .parquet files found under {base}. Got: {files[:20]}")

    shard_idx = 0
    in_shard = 0
    total = 0
    fout = open(out_dir / f"{dataset_id}_{shard_idx:05d}.jsonl", "w", encoding="utf-8")

    try:
        for p in parquet_files:
            with fs.open(p, "rb") as f:
                pf = pq.ParquetFile(f)

                # Read only 'text' (and optionally identifier-like columns if present)
                cols = set(pf.schema.names)
                read_cols = ["text"]
                if "identifier" in cols: read_cols.append("identifier")
                if "file_id" in cols: read_cols.append("file_id")
                if "id" in cols: read_cols.append("id")

                for batch in pf.iter_batches(batch_size=2048, columns=read_cols):
                    tbl = pa.Table.from_batches([batch])
                    data = tbl.to_pydict()

                    texts = data.get("text", [])
                    identifiers = (
                        data.get("identifier")
                        or data.get("file_id")
                        or data.get("id")
                        or [None] * len(texts)
                    )

                    for text, ident in zip(texts, identifiers):
                        text = clean(text)
                        if len(text) < min_chars:
                            continue

                        base_id = ident or text
                        doc_id = hashlib.sha256(str(base_id).encode("utf-8")).hexdigest()

                        rec = {
                            "id": doc_id,
                            "lang": "pt",
                            "source": "PleIAs/Portuguese-PD",
                            "content": text,
                        }

                        fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                        total += 1
                        in_shard += 1

                        if in_shard >= shard_docs:
                            fout.close()
                            shard_idx += 1
                            in_shard = 0
                            fout = open(out_dir / f"{dataset_id}_{shard_idx:05d}.jsonl", "w", encoding="utf-8")

    finally:
        fout.close()

    print(f"✅ Streamed {total:,} docs into {shard_idx + 1} shards at {out_dir}")

# If you want to pin the exact commit shown in your error, pass revision=...
stream_portuguese_pd_parquet_to_jsonl(out_dir, 
                                      shard_docs=100, 
                                      revision="d8d45f37105a975b84ad8fe49a0d574a552178c8")
