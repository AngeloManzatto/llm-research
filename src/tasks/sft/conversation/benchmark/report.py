"""
Created on Sun Jun 28 08:44:40 2026

@author: Angelo Antonio Manzatto
"""

###############################################################################
# Libraries
###############################################################################

import json
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

###############################################################################
# Evaluation Summary
###############################################################################

def _cell() -> dict:
    return {"total": 0, "passed": 0}


@dataclass
class EvaluationSummary:
    run_metadata: dict[str, Any]
    total:  int = 0
    passed: int = 0

    # metric_id -> pass count, pooled across every example in every category
    # (per Completion Criteria v1.1 §1 — expected_stop_token/repetition are
    # computed for all categories, not just turn_taking's own rows).
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

        # result.metrics: dict[str, MetricResult]. Every metric's `passed`
        # (always "True = desired behavior", per metric.py's convention)
        # is tallied both globally and per category/language cell.
        # `.details` isn't aggregated here — it's preserved per-row in
        # results.jsonl already; nothing has needed a run-level aggregate
        # of it yet.
        for name, metric_result in result.metrics.items():
            self.diagnostics.setdefault(name, 0)
            self.diagnostics[name] += int(metric_result.passed)
            cell.setdefault(name, 0)
            cell[name] += int(metric_result.passed)

    @property
    def pass_rate(self) -> float:
        return self.passed / self.total if self.total else 0.0

    def to_dict(self) -> dict[str, Any]:
        breakdown = {}
        for cat, langs in self.by_category_language.items():
            breakdown[cat] = {}
            for lang, cell in langs.items():
                c = dict(cell)
                c["pass_rate"] = round(
                    c["passed"] / c["total"] if c["total"] else 0.0, 4
                )
                breakdown[cat][lang] = c

        return {
            **self.run_metadata,
            "total":       self.total,
            "passed":      self.passed,
            "pass_rate":   round(self.pass_rate, 4),
            "diagnostics": dict(self.diagnostics),
            "by_category_language": breakdown,
        }

    @classmethod
    def from_file(cls, summary_path: Path) -> "EvaluationSummary":
        """Reload a previously saved summary.json for comparison."""
        data = json.loads(Path(summary_path).read_text(encoding="utf-8"))
        known = {"total", "passed", "pass_rate", "diagnostics", "by_category_language"}
        metadata = {k: v for k, v in data.items() if k not in known}
        summary = cls(run_metadata=metadata)
        summary.total       = data.get("total", 0)
        summary.passed      = data.get("passed", 0)
        summary.diagnostics = dict(data.get("diagnostics", {}))
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
        """Print a human-readable summary table, including each category's
        own metric pass counts (Completion Criteria v1.1 §1 cross-tab)."""
        d = self.to_dict()
        cats  = sorted(self.by_category_language)
        langs = sorted({l for c in self.by_category_language.values() for l in c})

        col = 24
        header = f"{'Category':<{col}}" + "".join(f"  {l.upper():>8}" for l in langs) + f"  {'TOTAL':>8}"
        print("\n" + "=" * len(header))
        print(f"Benchmark: {d.get('benchmark_id','?')}  v{d.get('benchmark_version','?')}  "
              f"model={d.get('model_id','?')}")
        print(f"Overall: {self.passed}/{self.total} passed ({self.pass_rate:.1%})")
        if self.diagnostics:
            diag_str = "  ".join(
                f"{name}={count}/{self.total} ({count/self.total:.1%})"
                for name, count in self.diagnostics.items()
            )
            print(f"Diagnostics (pooled, all categories): {diag_str}")
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

            # Per-category metric breakdown — only metrics that actually
            # ran for this category (e.g. constraint_satisfied never runs
            # for turn_taking rows, so it shouldn't appear as a misleading
            # "0 passes" line there).
            metric_names_in_cat: set[str] = set()
            totals: dict[str, int] = {}
            for lang in langs:
                cell = self.by_category_language[cat].get(lang, _cell())
                for k, v in cell.items():
                    if k in ("total", "passed"):
                        continue
                    metric_names_in_cat.add(k)
                    totals[k] = totals.get(k, 0) + v
            if metric_names_in_cat:
                detail = "  ".join(f"{k}={totals[k]}/{cat_total}" for k in sorted(metric_names_in_cat))
                print(f"{'':<{col}}  └─ {detail}")

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