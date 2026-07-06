"""
Created on Sun Jun 28 08:44:40 2026

@author: Angelo Antonio Manzatto
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


###############################################################################
# Evaluation Summary
###############################################################################

def _cell() -> dict:
    return {"total": 0, "passed": 0, "word_count_sum": 0}


@dataclass
class EvaluationSummary:
    run_metadata: dict[str, Any]
    total:  int = 0
    passed: int = 0

    # Aggregate boolean diagnostic counts (e.g. wiki_talk_artifact, repetition)
    diagnostics: dict[str, int] = field(default_factory=dict)

    # category × language breakdown
    by_category_language: dict[str, dict[str, dict]] = field(
        default_factory=lambda: defaultdict(lambda: defaultdict(_cell))
    )

    def update(self, result) -> None:
        cat  = result.category
        lang = result.language

        self.total  += 1
        self.passed += int(result.passed)

        cell = self.by_category_language[cat][lang]
        cell["total"]  += 1
        cell["passed"] += int(result.passed)

        for name, value in result.metrics.items():
            if isinstance(value, bool):
                # Global diagnostic count
                self.diagnostics.setdefault(name, 0)
                self.diagnostics[name] += int(value)
                # Per-cell diagnostic count
                cell.setdefault(name, 0)
                cell[name] += int(value)
            elif isinstance(value, int):
                # Accumulate numeric metrics (e.g. word_count) for averaging
                key = f"{name}_sum"
                cell[key] = cell.get(key, 0) + value

    @property
    def pass_rate(self) -> float:
        return self.passed / self.total if self.total else 0.0

    def to_dict(self) -> dict[str, Any]:
        # Compute per-cell derived stats before serialising
        breakdown = {}
        for cat, langs in self.by_category_language.items():
            breakdown[cat] = {}
            for lang, cell in langs.items():
                c = dict(cell)
                c["pass_rate"] = round(
                    c["passed"] / c["total"] if c["total"] else 0.0, 4
                )
                # Convert word_count_sum → avg_word_count if present
                if "word_count_sum" in c:
                    c["avg_word_count"] = round(
                        c["word_count_sum"] / c["total"] if c["total"] else 0.0, 2
                    )
                    del c["word_count_sum"]
                breakdown[cat][lang] = c

        return {
            **self.run_metadata,
            "total":      self.total,
            "passed":     self.passed,
            "pass_rate":  round(self.pass_rate, 4),
            **self.diagnostics,
            "by_category_language": breakdown,
        }

    @classmethod
    def from_file(cls, summary_path: Path) -> "EvaluationSummary":
        """Reload a previously saved summary.json for comparison."""
        data = json.loads(Path(summary_path).read_text(encoding="utf-8"))
        # Reconstruct metadata (everything except the known aggregate keys)
        known = {"total", "passed", "pass_rate", "by_category_language"}
        diagnostic_keys = {
            "wiki_talk_artifact", "repetition", "too_long", "expected_stop_token"
        }
        metadata = {k: v for k, v in data.items()
                    if k not in known and k not in diagnostic_keys}
        summary = cls(run_metadata=metadata)
        summary.total       = data.get("total", 0)
        summary.passed      = data.get("passed", 0)
        summary.diagnostics = {k: data[k] for k in diagnostic_keys if k in data}
        return summary

    def compare(self, other: "EvaluationSummary") -> dict[str, Any]:
        """Return a diff of pass rates between self (baseline) and other (new)."""
        delta = round(other.pass_rate - self.pass_rate, 4)
        result = {
            "baseline_pass_rate": round(self.pass_rate, 4),
            "new_pass_rate":      round(other.pass_rate, 4),
            "delta":              delta,
            "improved":           delta > 0,
            "by_category_language": {},
        }
        all_cats  = set(self.by_category_language) | set(other.by_category_language)
        all_langs = {"en", "pt"}
        for cat in sorted(all_cats):
            result["by_category_language"][cat] = {}
            for lang in sorted(all_langs):
                b = self.by_category_language.get(cat, {}).get(lang, _cell())
                n = other.by_category_language.get(cat, {}).get(lang, _cell())
                b_rate = b["passed"] / b["total"] if b["total"] else 0.0
                n_rate = n["passed"] / n["total"] if n["total"] else 0.0
                result["by_category_language"][cat][lang] = {
                    "baseline": round(b_rate, 4),
                    "new":      round(n_rate, 4),
                    "delta":    round(n_rate - b_rate, 4),
                }
        return result

    def print_table(self) -> None:
        """Print a human-readable summary table to stdout."""
        d = self.to_dict()
        cats  = sorted(self.by_category_language)
        langs = sorted({l for c in self.by_category_language.values() for l in c})

        # Header
        col = 24
        header = f"{'Category':<{col}}" + "".join(f"  {l.upper():>8}" for l in langs) + f"  {'TOTAL':>8}"
        print("\n" + "=" * len(header))
        print(f"Benchmark: {d.get('benchmark_id','?')}  v{d.get('benchmark_version','?')}  "
              f"model={d.get('model_id','?')}")
        print(f"Overall: {self.passed}/{self.total} passed ({self.pass_rate:.1%})")
        if self.diagnostics:
            diag_str = "  ".join(f"{k}={v}" for k, v in self.diagnostics.items())
            print(f"Diagnostics: {diag_str}")
        print("=" * len(header))
        print(header)
        print("-" * len(header))

        for cat in cats:
            row = f"{cat:<{col}}"
            cat_total = cat_passed = 0
            for lang in langs:
                cell = self.by_category_language[cat].get(lang, _cell())
                p, t = cell["passed"], cell["total"]
                cat_total  += t
                cat_passed += p
                pct = p / t if t else 0.0
                row += f"  {p:>4}/{t:<3} ({pct:>5.1%})"
            cat_pct = cat_passed / cat_total if cat_total else 0.0
            row += f"  {cat_passed:>4}/{cat_total:<3} ({cat_pct:>5.1%})"
            print(row)

        print("=" * len(header))


###############################################################################
# Report Writer
###############################################################################

class ReportWriter:
    def __init__(self, result_dir: Path):
        self.result_dir   = Path(result_dir)
        self.result_dir.mkdir(parents=True, exist_ok=True)
        self.results_file = self.result_dir / "results.jsonl"
        self.summary_file = self.result_dir / "summary.json"

    def write_result(self, result) -> None:
        with self.results_file.open("a", encoding="utf-8") as f:
            f.write(json.dumps(result.to_dict(), ensure_ascii=False) + "\n")

    def write_summary(self, summary: EvaluationSummary) -> None:
        with self.summary_file.open("w", encoding="utf-8") as f:
            json.dump(summary.to_dict(), f, indent=3, ensure_ascii=False)

    def reset(self) -> None:
        for path in (self.results_file, self.summary_file):
            if path.exists():
                path.unlink()