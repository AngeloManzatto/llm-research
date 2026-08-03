"""
Created on Sat Aug  1 12:57:42 2026

@author: Angelo Antonio Manzatto
"""

###############################################################################
# Libraries
###############################################################################

import json

from dataclasses import dataclass
from typing import Any
from pathlib import Path

###############################################################################
# Benchmark
###############################################################################

@dataclass(frozen=True)
class BenchmarkExample:
    id: str
    category: str
    language: str
    messages: list[dict[str, str]]   # [{"role": "user"|"assistant"|"system", "content": "..."}]
    meta: dict[str, Any] | None = None

@dataclass(frozen=True)
class Benchmark:
    benchmark_id: str
    root_dir: Path
    data_files: list[Path]
    examples: list[BenchmarkExample]
    category_metrics: dict[str, dict[str, dict[str, Any]]]
    default_decode: dict[str, Any]
 
###############################################################################
# Parse benchmark example
###############################################################################
 
_KNOWN_TOP_LEVEL_FIELDS = {"id", "category", "language", "messages"}
 
def parse_example(data: dict[str, Any]) -> BenchmarkExample:
    """
    Parses and validates one raw JSON row into a BenchmarkExample.
    Structural validation (required fields, message shape, roles) is
    unchanged from the original from_dict. What changed: expected_any
    and expected_stop_token are no longer required top-level keys or
    dedicated attributes -- they flow into meta like everything else,
    so a row simply omits them if a category's metrics don't need them.
    """
    required = ["id", "category", "language", "messages"]
    missing = [k for k in required if k not in data]
    if missing:
        raise ValueError(f"Benchmark example missing fields: {missing}")
 
    messages = data["messages"]
    if not isinstance(messages, list) or not messages:
        raise TypeError(f"Example {data.get('id', '?')} messages must be a non-empty list")
    for m in messages:
        if "role" not in m or "content" not in m:
            raise ValueError(f"Example {data.get('id', '?')} each message must have 'role' and 'content'")
        if m["role"] not in ("user", "assistant", "system"):
            raise ValueError(f"Example {data.get('id', '?')} unknown role: {m['role']!r}")
 
    if messages[-1]["role"] != "user":
        raise ValueError(f"Example {data.get('id', '?')} last message must be from 'user'")
 
    meta = {k: v for k, v in data.items() if k not in _KNOWN_TOP_LEVEL_FIELDS}
 
    return BenchmarkExample(
        id=str(data["id"]),
        category=str(data["category"]),
        language=str(data["language"]),
        messages=[{"role": m["role"], "content": m["content"]} for m in messages],
        meta=meta,
    )

###############################################################################
# Load benchmark
###############################################################################
 
def load_benchmark(benchmark_path: Path):
    
    benchmark_path = Path(benchmark_path)
    root_dir = benchmark_path.parent

    with benchmark_path.open("r", encoding="utf-8") as f:
        manifest = json.load(f)
    
    data_files = [root_dir / p for p in manifest["data_files"]]
    
    # Load benchmark examples
    examples = []
    
    for path in data_files:
        if not path.exists():
            raise FileNotFoundError(f"Benchmark data file not found: {path}")
        with path.open("r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                except json.JSONDecodeError as e:
                    raise ValueError(f"Invalid JSON in {path} line {line_no}: {e}") from e
                examples.append(parse_example(data))
    
    benchmark = Benchmark(
        benchmark_id=manifest["benchmark_id"], 
        root_dir=root_dir, 
        data_files=data_files, 
        examples=examples, 
        category_metrics=manifest["category_metrics"], 
        default_decode=manifest["default_decode"]
    )
    
    return benchmark