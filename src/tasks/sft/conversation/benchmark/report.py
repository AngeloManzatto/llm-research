"""
Created on Sun Jun 28 08:44:40 2026

@author: Angelo Antonio Manzatto
"""

###############################################################################
# Libraries
###############################################################################

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from src.tasks.sft.conversation.benchmark.evaluator import EvaluationResult, evaluation_result_to_dict

###############################################################################
# Summary
###############################################################################
 
def summarize_results(
    results: list[EvaluationResult],
    run_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    total  = len(results)
    passed = sum(1 for r in results if r.passed)
 
    # pooled, across every result regardless of category
    diagnostics: dict[str, int] = defaultdict(int)
    for r in results:
        for metric_name, metric_result in r.metrics.items():
            diagnostics[metric_name] += int(metric_result.passed)
 
    by_cat_lang: dict[str, dict[str, list[EvaluationResult]]] = defaultdict(lambda: defaultdict(list))
    for r in results:
        by_cat_lang[r.category][r.language].append(r)
 
    breakdown: dict[str, dict[str, Any]] = {}
    for category, by_lang in by_cat_lang.items():
        breakdown[category] = {}
        for language, rows in by_lang.items():
            cat_total  = len(rows)
            cat_passed = sum(1 for r in rows if r.passed)
 
            cell: dict[str, Any] = {
                "total": cat_total,
                "passed": cat_passed,
                "pass_rate": round(cat_passed / cat_total, 4) if cat_total else 0.0,
            }
            # only metrics that actually ran for THIS category's rows
            # appear here -- no misleading "0/N" for a metric this
            # category never uses
            cell_diagnostics: dict[str, int] = defaultdict(int)
            for r in rows:
                for metric_name, metric_result in r.metrics.items():
                    cell_diagnostics[metric_name] += int(metric_result.passed)
            cell.update(cell_diagnostics)
 
            breakdown[category][language] = cell
 
    summary = {
        **(run_metadata or {}),
        "total": total,
        "passed": passed,
        "pass_rate": round(passed / total, 4) if total else 0.0,
        "diagnostics": dict(diagnostics),
        "by_category_language": breakdown,
    }
    return summary

###############################################################################
# Print Summary
###############################################################################

def print_summary(
    summary: dict[str, Any],
    previous_summary: dict[str, Any] | None = None,
    step: int | None = None,
) -> None:
    header = f"Step {step}" if step is not None else "Summary"
    print(f"=== {header} ===")
 
    overall_line = f"Overall: {summary['passed']}/{summary['total']} passed ({summary['pass_rate']:.1%})"
    if previous_summary is not None:
        delta = summary["pass_rate"] - previous_summary["pass_rate"]
        overall_line += f"  [{_format_delta(delta)}]"
    print(overall_line)
 
    if summary.get("diagnostics"):

        diag_totals: dict[str, int] = {}
        for by_lang in summary["by_category_language"].values():
            for cell in by_lang.values():
                for name in summary["diagnostics"]:
                    if name in cell:
                        diag_totals[name] = diag_totals.get(name, 0) + cell["total"]
 
        diag_str = "  ".join(
            f"{name}={count}/{diag_totals.get(name, summary['total'])} "
            f"({count/diag_totals[name]:.1%})" if diag_totals.get(name) else f"{name}={count}/0"
            for name, count in summary["diagnostics"].items()
        )
        print(f"Diagnostics (pooled across categories that use each metric): {diag_str}")
    print()
 
    print(f"{'Category':<24}{'Lang':<6}{'Total':>7}{'Passed':>8}{'Rate':>8}{'Δ':>10}")
    for category in sorted(summary["by_category_language"]):
        for language in sorted(summary["by_category_language"][category]):
            stats = summary["by_category_language"][category][language]
 
            delta_str = ""
            if previous_summary is not None:
                prev_stats = previous_summary.get("by_category_language", {}).get(category, {}).get(language)
                if prev_stats is not None:
                    delta_str = _format_delta(stats["pass_rate"] - prev_stats["pass_rate"])
                else:
                    delta_str = "new"
 
            print(
                f"{category:<24}{language:<6}{stats['total']:>7}{stats['passed']:>8}"
                f"{stats['pass_rate']:>8.1%}{delta_str:>10}"
            )
 
            metric_names = sorted(
                k for k in stats if k not in ("total", "passed", "pass_rate")
            )
            if metric_names:
                detail = "  ".join(f"{name}={stats[name]}/{stats['total']}" for name in metric_names)
                print(f"{'':<24}  └─ {detail}")
 
 
def _format_delta(delta: float) -> str:
    if abs(delta) < 0.0001:
        return "→ 0.0%"
    arrow = "↑" if delta > 0 else "↓"
    return f"{arrow} {abs(delta):.1%}"

###############################################################################
# Evaluation Summary
###############################################################################

def write_result(result_dir: Path, result: EvaluationResult) -> None:
    """Appends one result to result_dir/results.jsonl."""
    results_file = Path(result_dir) / "results.jsonl"
    with results_file.open("a", encoding="utf-8") as f:
        f.write(json.dumps(evaluation_result_to_dict(result), ensure_ascii=False) + "\n")
 
 
def write_summary(result_dir: Path, summary: dict[str, Any]) -> None:
    """Overwrites result_dir/summary.json with the current summary."""
    summary_file = Path(result_dir) / "summary.json"
    with summary_file.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=3, ensure_ascii=False)
 
 
def reset_report(result_dir: Path) -> None:
    """Deletes both results.jsonl and summary.json in result_dir, if present."""
    result_dir = Path(result_dir)
    for name in ("results.jsonl", "summary.json"):
        path = result_dir / name
        if path.exists():
            path.unlink()