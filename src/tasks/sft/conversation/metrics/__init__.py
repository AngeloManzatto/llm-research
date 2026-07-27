# -*- coding: utf-8 -*-
"""
Stage 0 benchmark metrics — one file per metric (see Metrics Contract
v2.0 for the full design discussion behind each one).

This package replaces the old single metric.py. Every name that used
to be importable from `metric` is importable from `metrics` unchanged:

    from ...benchmark.metrics import MetricResult, run_metric, METRICS

Callers never need to know metrics live in separate files — that's an
internal organization detail, not part of the public surface.
"""

from .base import MetricResult
from .registry import METRICS, run_metric

from .expected_stop_token import expected_stop_token
from .repetition import repetition
from .contains_expected import contains_expected
from .constraint_satisfied import constraint_satisfied
from .coherence import coherence

__all__ = [
    "MetricResult",
    "METRICS",
    "run_metric",
    "expected_stop_token",
    "repetition",
    "contains_expected",
    "constraint_satisfied",
    "coherence",
]
