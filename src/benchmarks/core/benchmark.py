"""
Created on Sat Jun 27 23:55:17 2026

@author: Angelo Antonio Manzatto
"""

###############################################################################
# Libraries
###############################################################################

from __future__ import annotations

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
    prompt: str
    expected_any: list[str]
    expected_stop_token: str | None = None
    scoring: str = "contains"

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "BenchmarkExample":
        required = ["id", "category", "language", "prompt", "expected_any"]

        missing = [k for k in required if k not in data]
        if missing:
            raise ValueError(f"Benchmark example missing fields: {missing}")

        if not isinstance(data["expected_any"], list):
            raise TypeError(
                f"Example {data.get('id', '<unknown>')} expected_any must be a list"
            )

        return cls(
            id=str(data["id"]),
            category=str(data["category"]),
            language=str(data["language"]),
            prompt=str(data["prompt"]),
            expected_any=[str(x) for x in data["expected_any"]],
            expected_stop_token=(
                str(data["expected_stop_token"])
                if data.get("expected_stop_token") is not None
                else None
            ),
            scoring=str(data.get("scoring", "contains")),
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
    scoring_metric: str
    diagnostic_metrics: list[str]
    examples: list[BenchmarkExample]

    @classmethod
    def from_manifest(cls, manifest_path: Path) -> "Benchmark":
        manifest_path = Path(manifest_path)
        root_dir = manifest_path.parent

        with manifest_path.open("r", encoding="utf-8") as f:
            manifest = json.load(f)

        required = [
            "benchmark_id",
            "version",
            "description",
            "data_files",
            "default_decode",
            "scoring_metric",
            "diagnostic_metrics",
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
                        raise ValueError(
                            f"Invalid JSON in {path} line {line_no}: {e}"
                        ) from e

                    examples.append(BenchmarkExample.from_dict(data))

        return cls(
            benchmark_id=str(manifest["benchmark_id"]),
            version=str(manifest["version"]),
            description=str(manifest["description"]),
            root_dir=root_dir,
            data_files=data_files,
            default_decode=dict(manifest["default_decode"]),
            scoring_metric=str(manifest["scoring_metric"]),
            diagnostic_metrics=[str(m) for m in manifest["diagnostic_metrics"]],
            examples=examples,
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
            "scoring_metric": self.scoring_metric,
            "diagnostic_metrics": self.diagnostic_metrics,
        }