"""
Created on Sun Jun 28 08:11:28 2026

@author: Angelo Antonio Manzatto
"""

###############################################################################
# Libraries
###############################################################################

from __future__ import annotations

import re

from dataclasses import dataclass
from typing import Any

from src.tasks.sft.conversation.core.special_tokens import STOP_TOKENS
from src.tasks.sft.conversation.benchmark.metric import build_metric
from src.tasks.sft.conversation.benchmark.benchmark import BenchmarkExample

###############################################################################
# Evaluation Result
###############################################################################

@dataclass(frozen=True)
class EvaluationResult:
    id: str
    category: str
    language: str
    messages: list[dict[str, str]]
    expected_any: list[str]
    raw_answer: str
    answer: str
    passed: bool
    metrics: dict[str, Any]
    decode: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "category": self.category,
            "language": self.language,
            "messages": self.messages,
            "expected_any": self.expected_any,
            "raw_answer": self.raw_answer,
            "answer": self.answer,
            "passed": self.passed,
            **self.metrics,
            "decode": self.decode,
        }

###############################################################################
# Normalize answer
###############################################################################

def strip_special_stop_tokens(text: str) -> str:
    for tok in STOP_TOKENS:
        text = text.replace(tok, "")
    return text

def normalize_answer(
    answer: str,
    *,
    strip_stop_tokens: bool = True,
    normalize_whitespace: bool = True,
) -> str:
    if strip_stop_tokens:
        answer = strip_special_stop_tokens(answer)

    if normalize_whitespace:
        answer = re.sub(r"\s+", " ", answer).strip()

    return answer

###############################################################################
# Evaluate example
###############################################################################

def evaluate_example(
    *,
    example: BenchmarkExample,
    generated: str,
    decode: dict[str, Any],
    scoring_metric: str,
    diagnostic_metrics: list[str],
) -> EvaluationResult:
    # greedy_decode returns only the generated portion (no prompt prefix to strip)
    raw_answer = generated.strip()
    answer     = normalize_answer(raw_answer, strip_stop_tokens=True, normalize_whitespace=True)

    passed = bool(build_metric(scoring_metric).evaluate(answer=answer, example=example).value)

    metrics = {
        metric_id: build_metric(metric_id).evaluate(
            answer=answer, raw_answer=raw_answer, example=example,
        ).value
        for metric_id in diagnostic_metrics
    }

    return EvaluationResult(
        id=example.id,
        category=example.category,
        language=example.language,
        messages=example.messages,
        expected_any=example.expected_any,
        raw_answer=raw_answer,
        answer=answer,
        passed=passed,
        metrics=metrics,
        decode=decode,
    )