"""
Created on Sat Jun 27 23:55:17 2026

@author: Angelo Antonio Manzatto
"""

###############################################################################
# Libraries
###############################################################################

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

###############################################################################
# Benchmark Example
###############################################################################

@dataclass(frozen=True)
class BenchmarkExample:
    id: str
    category: str
    language: str
    messages: list[dict[str, str]]   # [{"role": "user"|"assistant"|"system", "content": "..."}]
    expected_any: list[str]
    expected_stop_token: str | None = None
    # Generic bucket for category-specific ground truth used by metrics
    # beyond expected_any/expected_stop_token (stated_value, corrected_value,
    # constraint_type, constraint_value, refusal_patterns, ...). Deliberately
    # untyped rather than one dataclass field per metric — each category
    # only populates the handful of keys its own metrics need, and adding a
    # new metric never requires touching this dataclass again.
    meta: dict[str, Any] | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "BenchmarkExample":
        
        # Required fields on each example
        required = ["id", "category", "language", "messages", "expected_any"]

        missing = [k for k in required if k not in data]
        if missing:
            raise ValueError(f"Benchmark example missing fields: {missing}")
        
        # Messages using CHatML format
        messages = data["messages"]
        if not isinstance(messages, list) or not messages:
            raise TypeError(f"Example {data.get('id', '?')} messages must be a non-empty list")
        for m in messages:
            if "role" not in m or "content" not in m:
                raise ValueError(f"Example {data.get('id', '?')} each message must have 'role' and 'content'")
            if m["role"] not in ("user", "assistant", "system"):
                raise ValueError(f"Example {data.get('id', '?')} unknown role: {m['role']!r}")
                
        # Assert that USER finished conversation 
        if messages[-1]["role"] != "user":
            raise ValueError(f"Example {data.get('id', '?')} last message must be from 'user'")
        
        # Need an expected assertion to validate against something
        if not isinstance(data["expected_any"], list):
            raise TypeError(f"Example {data.get('id', '?')} expected_any must be a list")

        known = {"id", "category", "language", "messages", "expected_any", "expected_stop_token"}
        meta = {k: v for k, v in data.items() if k not in known}

        return cls(
            id=str(data["id"]),
            category=str(data["category"]),
            language=str(data["language"]),
            messages=[{"role": m["role"], "content": m["content"]} for m in messages],
            expected_any=[str(x) for x in data["expected_any"]],
            expected_stop_token=(
                str(data["expected_stop_token"])
                if data.get("expected_stop_token") is not None
                else None
            ),
            meta=meta,
        )

###############################################################################
# Benchmark
###############################################################################

@dataclass(frozen=True)
class Benchmark:
    benchmark_id: str
    version: str
    description: str
    root_dir: Path
    data_files: list[Path]
    default_decode: dict[str, Any]
    examples: list[BenchmarkExample]
    # Which metric scores `passed` for each category. This is the single
    # source of truth — there is no separate global "scoring_metric" field
    # to keep in sync with it.
    category_scoring_metric: dict[str, str]
    # Metrics computed for every example regardless of category (e.g.
    # expected_stop_token, repetition — pooled across all categories per
    # Completion Criteria v1.1 §1).
    always_computed: list[str]
    # Category-level shared context, merged into every row of that
    # category at evaluation time (see evaluator.py's _build_context).
    # Solves the "hard link between category and pattern" problem: some
    # ground truth (e.g. uncertainty's refusal patterns) is genuinely the
    # SAME for every row in a category, not row-specific data. Duplicating
    # it into every row's own expected_any/meta means expanding the pattern
    # list requires editing hundreds of rows; defining it once here means
    # expanding it is a one-line manifest edit that every row in that
    # category picks up automatically. A row's own expected_any/meta still
    # takes precedence when it's genuinely row-specific (e.g.
    # knowledge_completion's actual fact) — see _build_context for the
    # precedence rule.
    category_shared_context: dict[str, dict[str, Any]]

    @classmethod
    def from_manifest(cls, manifest_path: Path) -> "Benchmark":
        manifest_path = Path(manifest_path)
        root_dir = manifest_path.parent

        with manifest_path.open("r", encoding="utf-8") as f:
            manifest = json.load(f)

        required = [
            "benchmark_id", "version", "description", "data_files",
            "default_decode", "category_scoring_metric", "always_computed",
        ]
        missing = [k for k in required if k not in manifest]
        if missing:
            raise ValueError(f"Benchmark manifest missing fields: {missing}")

        data_files = [root_dir / p for p in manifest["data_files"]]
        examples: list[BenchmarkExample] = []

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
                    examples.append(BenchmarkExample.from_dict(data))

        # category_scoring_metric may be a single string (applied to every
        # category found in the data — the common case, avoids repeating
        # the same metric name per category) or a full per-category dict.
        raw_scoring = manifest["category_scoring_metric"]
        if isinstance(raw_scoring, str):
            categories = {ex.category for ex in examples}
            category_scoring_metric = {cat: raw_scoring for cat in categories}
        else:
            category_scoring_metric = {str(k): str(v) for k, v in raw_scoring.items()}

        always_computed = [str(m) for m in manifest["always_computed"]]

        # Optional — absent manifests just get no shared context (fully
        # backward compatible; every row still works via its own
        # expected_any/meta exactly as before).
        category_shared_context = {
            str(cat): dict(ctx)
            for cat, ctx in manifest.get("category_shared_context", {}).items()
        }

        return cls(
            benchmark_id=str(manifest["benchmark_id"]),
            version=str(manifest["version"]),
            description=str(manifest["description"]),
            root_dir=root_dir,
            data_files=data_files,
            default_decode=dict(manifest["default_decode"]),
            examples=examples,
            category_scoring_metric=category_scoring_metric,
            always_computed=always_computed,
            category_shared_context=category_shared_context,
        )

    def __len__(self) -> int:
        return len(self.examples)

    def __iter__(self):
        return iter(self.examples)

    @property
    def categories(self) -> list[str]:
        return sorted({ex.category for ex in self.examples})

    def summary(self) -> dict[str, Any]:
        return {
            "benchmark_id": self.benchmark_id,
            "version": self.version,
            "description": self.description,
            "n_examples": len(self.examples),
            "categories": self.categories,
            "data_files": [str(p) for p in self.data_files],
            "default_decode": self.default_decode,
            "category_scoring_metric": self.category_scoring_metric,
            "always_computed": self.always_computed,
            "category_shared_context": self.category_shared_context,
        }