"""
Created on Sun Jun 28 10:34:27 2026

@author: Angelo Antonio Manzatto
"""

###############################################################################
# Libraries
###############################################################################

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

###############################################################################
# Metric Result
###############################################################################

@dataclass(frozen=True)
class MetricResult:
    metric_id: str
    value: Any
    version: str

###############################################################################
# Metric Base
###############################################################################

class Metric(ABC):
    id: str
    version: str = "1.0"
    description: str = ""
    deterministic: bool = True
    output_type: type = object

    @abstractmethod
    def evaluate(self, *, answer: str, raw_answer: str | None = None, example: Any | None = None) -> MetricResult:
        pass

###############################################################################
# Contains Expected
###############################################################################

class ContainsExpectedMetric(Metric):
    id = "contains_expected"
    description = "Returns True if the answer contains any expected string."
    output_type = bool

    def evaluate(self, *, answer: str, raw_answer: str | None = None, example: Any | None = None) -> MetricResult:
        if example is None:
            raise ValueError("ContainsExpectedMetric requires an example.")
        answer_lower = answer.lower()
        value = any(exp.lower() in answer_lower for exp in example.expected_any)
        return MetricResult(self.id, value, self.version)

###############################################################################
# Wiki Talk Artifact
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

    def evaluate(self, *, answer: str, raw_answer: str | None = None, example: Any | None = None) -> MetricResult:
        value = any(re.search(p, answer, flags=re.IGNORECASE) for p in self.patterns)
        return MetricResult(self.id, value, self.version)

###############################################################################
# Repetition
###############################################################################

class RepetitionMetric(Metric):
    id = "repetition"
    description = "Detects repeated sentence-like chunks."
    output_type = bool

    def evaluate(self, *, answer: str, raw_answer: str | None = None, example: Any | None = None) -> MetricResult:
        chunks = [c.strip().lower() for c in re.split(r"[.\n]", answer) if c.strip()]
        value = len(chunks) != len(set(chunks))
        return MetricResult(self.id, value, self.version)

###############################################################################
# Word Count
###############################################################################

class WordCountMetric(Metric):
    id = "word_count"
    description = "Counts whitespace-separated words."
    output_type = int

    def evaluate(self, *, answer: str, raw_answer: str | None = None, example: Any | None = None) -> MetricResult:
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

    def evaluate(self, *, answer: str, raw_answer: str | None = None, example: Any | None = None) -> MetricResult:
        value = len(answer.split()) > self.max_words
        return MetricResult(self.id, value, self.version)

###############################################################################
# Expected Stop Token
###############################################################################

class ExpectedStopTokenMetric(Metric):
    id = "expected_stop_token"
    description = (
        "Checks whether the raw answer ends with the expected stop token. "
        "Requires raw_answer (pre-normalization) since normalizer strips stop tokens. "
        "NOTE: raw_answer must preserve the literal stop token string (e.g. '<EOS>') "
        "as emitted by the tokenizer's decode path for special token IDs."
    )
    output_type = bool

    def evaluate(self, *, answer: str, raw_answer: str | None = None, example: Any | None = None) -> MetricResult:
        if example is None:
            raise ValueError("ExpectedStopTokenMetric requires an example.")

        expected_name = getattr(example, "expected_stop_token", None)

        if expected_name is None:
            return MetricResult(self.id, False, self.version)

        from src.benchmarks.core.special_tokens import TOKEN_BY_NAME
        expected_token = TOKEN_BY_NAME[expected_name].token

        # Check raw_answer (stop tokens are stripped from normalized answer).
        # Use endswith rather than `in` to ensure the stop token closes the turn
        # rather than appearing spuriously mid-completion.
        target = raw_answer if raw_answer is not None else answer
        value = target.rstrip().endswith(expected_token)

        return MetricResult(self.id, value, self.version)

###############################################################################
# Metric Registry
###############################################################################

METRIC_REGISTRY: dict[str, type[Metric]] = {
    "contains_expected":  ContainsExpectedMetric,
    "wiki_talk_artifact": WikiTalkArtifactMetric,
    "repetition":         RepetitionMetric,
    "word_count":         WordCountMetric,
    "too_long":           TooLongMetric,
    "expected_stop_token": ExpectedStopTokenMetric,
}

def build_metric(metric_id: str) -> Metric:
    if metric_id not in METRIC_REGISTRY:
        raise KeyError(f"Unknown metric: '{metric_id}'. Available: {sorted(METRIC_REGISTRY)}")
    return METRIC_REGISTRY[metric_id]()