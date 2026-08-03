"""
Created on Sun Aug  2 22:23:39 2026

@author: Angelo Antonio Manzatto
"""

###############################################################################
# Libraries
###############################################################################

import json
from pathlib import Path

from dataclasses import dataclass
from typing import Any
from collections import defaultdict

from src.tasks.sft.conversation.core.special_tokens import TOKEN_BY_NAME
from src.tasks.sft.conversation.metrics import run_metric, MetricResult
from src.tasks.sft.conversation.metrics.utils import strip_trailing_tags

###############################################################################
# Evaluation Result
###############################################################################
 
@dataclass(frozen=True)
class EvaluationResult:
    id: str
    category: str
    language: str
    messages: list[dict[str, str]]
    raw_answer: str
    answer: str
    passed: bool
    metrics: dict[str, MetricResult]
    failed_checks: list[str]
    decode: dict[str, Any]
 
 
def evaluation_result_to_dict(result: EvaluationResult) -> dict[str, Any]:
    """Flattens an EvaluationResult into the dict shape written to results.jsonl."""
    flat: dict[str, Any] = {}
    for name, metric_result in result.metrics.items():
        flat[name] = metric_result.passed
        if metric_result.details:
            flat[f"{name}_details"] = metric_result.details
 
    return {
        "id": result.id,
        "category": result.category,
        "language": result.language,
        "messages": result.messages,
        "raw_answer": result.raw_answer,
        "answer": result.answer,
        "passed": result.passed,
        "failed_checks": result.failed_checks,
        **flat,
        "decode": result.decode,
    }
 
###############################################################################
# Build per-metric context
###############################################################################
 
def build_metric_context(example, benchmark, metric_id: str) -> dict[str, Any]:
    """
    Assembles the kwargs ONE specific metric needs for ONE specific row.
 
    Replaces the old _build_context, which built one context dict
    shared across every metric for a row, pulled from
    category_shared_context (uniform per category). That mechanism no
    longer exists -- category_metrics stores context per
    (category, metric_id) pair, since not every metric for a category
    needs the same injected context.
 
    Merge rule, carried over unchanged from the old shared-context
    logic: list-valued keys UNION (category context + row meta,
    deduplicated, order preserved); scalar keys let the row's own meta
    win when present.
    """
    category_config = benchmark.category_metrics.get(example.category, {})
    metric_config   = category_config.get(metric_id, {})
    shared          = metric_config.get("context", {})
    row_data        = dict(example.meta or {})
 
    context: dict[str, Any] = {}
    for key in set(shared) | set(row_data):
        shared_val = shared.get(key)
        row_val = row_data.get(key)
        if isinstance(shared_val, list) and isinstance(row_val, list):
            context[key] = list(dict.fromkeys(shared_val + row_val))
        elif row_val not in (None, [], ""):
            context[key] = row_val
        elif shared_val is not None:
            context[key] = shared_val
        else:
            context[key] = row_val
 
    stop_token_name = row_data.get("expected_stop_token")
    if stop_token_name is not None:
        context.setdefault("expected_token", TOKEN_BY_NAME[stop_token_name])
 
    context.setdefault("language", example.language)
 
    return context

###############################################################################
# Evaluate example
###############################################################################
 
def evaluate_example(example, benchmark, raw_answer: str) -> EvaluationResult:
    """
    benchmark.category_metrics drives everything: which metrics run for
    this category, whether each one gates `passed` or is diagnostic-
    only, and what extra context it needs. Changing any of that is a
    manifest edit, not a code change.
    """
    category_config = benchmark.category_metrics.get(example.category, {})
 
    metrics: dict[str, MetricResult] = {}
    for metric_id in category_config:
        context = build_metric_context(example, benchmark, metric_id)
        metrics[metric_id] = run_metric(metric_id, raw_answer, **context)
 
    failed_checks = [
        metric_id for metric_id, config in category_config.items()
        if config.get("role") == "gate" and not metrics[metric_id].passed
    ]
    passed = len(failed_checks) == 0
    
    result = EvaluationResult(
        id=example.id,
        category=example.category,
        language=example.language,
        messages=example.messages,
        raw_answer=raw_answer,
        answer=strip_trailing_tags(raw_answer),
        passed=passed,
        metrics=metrics,
        failed_checks=failed_checks,
        decode=benchmark.default_decode,
    )
 
    return result