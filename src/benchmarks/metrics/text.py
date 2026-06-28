"""
Created on Sun Jun 28 10:35:58 2026

@author: Angelo Antonio Manzatto
"""

###############################################################################
# Libraries
###############################################################################

from __future__ import annotations

import re
from typing import Any

from src.benchmarks.core.metric import Metric, MetricResult

###############################################################################
# Contains Expected
###############################################################################

class ContainsExpectedMetric(Metric):
    id = "contains_expected"
    description = "Returns True if the answer contains any expected string."
    output_type = bool

    def evaluate(self, *, answer: str, example: Any | None = None) -> MetricResult:
        if example is None:
            raise ValueError("ContainsExpectedMetric requires an example.")

        answer_lower = answer.lower()
        value = any(exp.lower() in answer_lower for exp in example.expected_any)

        return MetricResult(self.id, value, self.version)

###############################################################################
# Wiki Talsk Artifact
###############################################################################

class WikiTalkArtifactMetric(Metric):
    id = "wiki_talk_artifact"
    description = "Detects Wikipedia talk-page artifacts."
    output_type = bool

    patterns = [
        r"\(talk\)",
        r"\bUTC\b",
        r"unsigned comment",
        r"talk page",
        r"deletion review",
        r"please do not modify it",
    ]

    def evaluate(self, *, answer: str, example: Any | None = None) -> MetricResult:
        value = any(re.search(p, answer, flags=re.IGNORECASE) for p in self.patterns)
        return MetricResult(self.id, value, self.version)

###############################################################################
# Repetition
###############################################################################

class RepetitionMetric(Metric):
    id = "repetition"
    description = "Detects repeated sentence-like chunks."
    output_type = bool

    def evaluate(self, *, answer: str, example: Any | None = None) -> MetricResult:
        chunks = [c.strip().lower() for c in re.split(r"[.\n]", answer) if c.strip()]
        value = len(chunks) != len(set(chunks))
        return MetricResult(self.id, value, self.version)

###############################################################################
# Role Leakage
###############################################################################

class RoleLeakageMetric(Metric):
    id = "role_leakage"
    description = "Detects leaked User:/Assistant: role markers in generated answer."
    output_type = bool

    def evaluate(self, *, answer: str, example: Any | None = None) -> MetricResult:
        value = "User:" in answer or answer.count("Assistant:") > 0
        return MetricResult(self.id, value, self.version)

###############################################################################
# Word Count
###############################################################################

class WordCountMetric(Metric):
    id = "word_count"
    description = "Counts whitespace-separated words."
    output_type = int

    def evaluate(self, *, answer: str, example: Any | None = None) -> MetricResult:
        return MetricResult(self.id, len(answer.split()), self.version)

###############################################################################
# Too Long
###############################################################################

class TooLongMetric(Metric):
    id = "too_long"
    description = "Returns True if answer exceeds maximum word count."
    output_type = bool

    def __init__(self, max_words: int = 40):
        self.max_words = max_words

    def evaluate(self, *, answer: str, example: Any | None = None) -> MetricResult:
        value = len(answer.split()) > self.max_words
        return MetricResult(self.id, value, self.version)

###############################################################################
# Metric registry
###############################################################################

METRIC_REGISTRY = {
    "contains_expected": ContainsExpectedMetric,
    "wiki_talk_artifact": WikiTalkArtifactMetric,
    "repetition": RepetitionMetric,
    "role_leakage": RoleLeakageMetric,
    "word_count": WordCountMetric,
    "too_long": TooLongMetric,
}


def build_metric(metric_id: str) -> Metric:
    if metric_id not in METRIC_REGISTRY:
        raise KeyError(f"Unknown metric: {metric_id}")
    return METRIC_REGISTRY[metric_id]()