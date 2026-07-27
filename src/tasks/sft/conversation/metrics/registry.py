# -*- coding: utf-8 -*-
"""
Assembles the metric registry from each metric's own file, and defines
run_metric — the single entry point every caller (evaluator.py,
dedup_scan.py, ad hoc checks) actually uses.

Adding a new metric means: create metrics/your_metric.py with a
function following the (raw_answer, **kwargs) -> MetricResult
signature and a `.requires` tuple, then add one line here. Nothing
else in the codebase needs to change — this was the whole point of
the earlier category_scoring_metric refactor, now extended to the
file layout itself.
"""

from __future__ import annotations

from typing import Callable

from .base import MetricResult
from .expected_stop_token import expected_stop_token
from .repetition import repetition
from .contains_expected import contains_expected
from .constraint_satisfied import constraint_satisfied
from .coherence import coherence

METRICS: dict[str, Callable[..., MetricResult]] = {
    "expected_stop_token": expected_stop_token,
    "repetition":          repetition,
    "contains_expected":   contains_expected,
    "constraint_satisfied": constraint_satisfied,
    "coherence":           coherence,
}


def run_metric(metric_id: str, raw_answer: str, **kwargs) -> MetricResult:
    if metric_id not in METRICS:
        raise KeyError(f"Unknown metric: '{metric_id}'. Available: {sorted(METRICS)}")

    fn = METRICS[metric_id]
    missing = [k for k in getattr(fn, "requires", ()) if k not in kwargs]
    if missing:
        raise ValueError(f"'{metric_id}' missing required fields: {missing}")

    return fn(raw_answer, **kwargs)
