"""
Created on Mon Dec 29 10:02:15 2025

@author: Angelo Antonio Manzatto
"""
###############################################################################
# Libraries
###############################################################################
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence, Union, Tuple

###############################################################################
# Config
###############################################################################
@dataclass(frozen=True)
class PackedShard:
    path: Path
    tokens: int

###############################################################################
# Loading Packed Manifest Utility
###############################################################################
def _as_list(x: Union[str, Sequence[str]]) -> List[str]:
    return [x] if isinstance(x, str) else list(x)

def load_packed_index(index_sources: Union[str, Sequence[str]]) -> List[PackedShard]:
    shards: List[PackedShard] = []
    for src in _as_list(index_sources):
        p = Path(src)
        if not p.exists():
            raise FileNotFoundError(f"Index file not found: {p}")
        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                rec = json.loads(line)
                shard_path = Path(rec["path"])
                shards.append(PackedShard(path=shard_path, tokens=int(rec["tokens"])))
    # stable order
    shards.sort(key=lambda s: str(s.path))
    return shards

def shards_to_paths_and_total_tokens(shards: Iterable[PackedShard]) -> Tuple[List[str], int]:
    paths = []
    total = 0
    for s in shards:
        paths.append(str(s.path))
        total += int(s.tokens)
    return paths, total
