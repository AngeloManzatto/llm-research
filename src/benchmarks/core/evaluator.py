"""
Created on Sun Jun 28 08:11:28 2026

@author: Angelo Antonio Manzatto
"""

###############################################################################
# Libraries
###############################################################################

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.benchmarks.core.benchmark import BenchmarkExample
from src.benchmarks.metrics.text import build_metric

###############################################################################
# Evaluation Result
###############################################################################

@dataclass(frozen=True)
class EvaluationResult:
    id: str
    category: str
    language: str
    prompt: str
    expected_any: list[str]
    full_text: str
    answer: str
    passed: bool
    metrics: dict[str, Any]
    decode: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "category": self.category,
            "language": self.language,
            "prompt": self.prompt,
            "expected_any": self.expected_any,
            "full_text": self.full_text,
            "answer": self.answer,
            "passed": self.passed,
            **self.metrics,
            "decode": self.decode,
        }

###############################################################################
# Evaluation example
###############################################################################

def extract_completion(full_text: str, prompt: str) -> str:
    if full_text.startswith(prompt):
        return full_text[len(prompt):].strip()
    return full_text.strip()

def evaluate_example(
    *,
    example: BenchmarkExample,
    full_text: str,
    decode: dict[str, Any],
    scoring_metric: str,
    diagnostic_metrics: list[str],
) -> EvaluationResult:
    answer = extract_completion(full_text, example.prompt)

    scoring = build_metric(scoring_metric)
    score_result = scoring.evaluate(answer=answer, example=example)
    passed = bool(score_result.value)

    metrics = {}
    for metric_id in diagnostic_metrics:
        metric = build_metric(metric_id)
        metric_result = metric.evaluate(answer=answer, example=example)
        metrics[metric_result.metric_id] = metric_result.value

    return EvaluationResult(
        id=example.id,
        category=example.category,
        language=example.language,
        prompt=example.prompt,
        expected_any=example.expected_any,
        full_text=full_text,
        answer=answer,
        passed=passed,
        metrics=metrics,
        decode=decode,
    )