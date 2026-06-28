"""
Created on Sun Jun 28 08:44:40 2026

@author: Angelo Antonio Manzatto
"""

###############################################################################
# Libraries
###############################################################################

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from collections import defaultdict

from src.tasks.sft.conversation.evaluator import EvaluationResult

###############################################################################
# Evaluation Summary
###############################################################################

@dataclass
class EvaluationSummary:
    run_metadata: dict[str, Any]
    total: int = 0
    passed: int = 0
    metrics: dict[str, int] = field(default_factory=dict)
    by_category: dict[str, dict[str, int]] = field(
        default_factory=lambda: defaultdict(lambda: {
            "total": 0,
            "passed": 0,
        })
    )

    def update(self, result: EvaluationResult) -> None:
        category = result.category

        self.total += 1
        self.passed += int(result.passed)

        self.by_category[category]["total"] += 1
        self.by_category[category]["passed"] += int(result.passed)

        for metric_name, metric_value in result.metrics.items():
            # Only aggregate boolean metrics.
            if isinstance(metric_value, bool):
                self.metrics.setdefault(metric_name, 0)
                self.metrics[metric_name] += int(metric_value)

                self.by_category[category].setdefault(metric_name, 0)
                self.by_category[category][metric_name] += int(metric_value)

    def to_dict(self) -> dict[str, Any]:
        return {
            **self.run_metadata,
            "total": self.total,
            "passed": self.passed,
            **self.metrics,
            "by_category": dict(self.by_category),
        }

###############################################################################
# Report Writer
###############################################################################

class ReportWriter:
    def __init__(self, result_dir: Path):
        self.result_dir = Path(result_dir)
        self.result_dir.mkdir(parents=True, exist_ok=True)

        self.results_file = self.result_dir / "results.jsonl"
        self.summary_file = self.result_dir / "summary.json"

    def write_result(self, result: EvaluationResult) -> None:
        with self.results_file.open("a", encoding="utf-8") as f:
            f.write(json.dumps(result.to_dict(), ensure_ascii=False) + "\n")

    def write_summary(self, summary: EvaluationSummary) -> None:
        with self.summary_file.open("w", encoding="utf-8") as f:
            json.dump(summary.to_dict(), f, indent=3, ensure_ascii=False)

    def reset(self) -> None:
        if self.results_file.exists():
            self.results_file.unlink()
        if self.summary_file.exists():
            self.summary_file.unlink()