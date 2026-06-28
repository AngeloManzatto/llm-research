"""
Created on Sat Jun 27 23:03:19 2026

@author: Angelo Antonio Manzatto
"""

###############################################################################
# Libraries
###############################################################################
from __future__ import annotations

import re
from typing import Callable, Any

###############################################################################
# Metrics function
###############################################################################
MetricFn = Callable[[str], Any]

###############################################################################
# Metrics heuristics
###############################################################################
def score_contains(answer: str, expected_any: list[str]) -> bool:
    answer_lower = answer.lower()
    return any(exp.lower() in answer_lower for exp in expected_any)

def has_wiki_talk_artifact(text: str) -> bool:
    patterns = [
        r"\(talk\)",
        r"\bUTC\b",
        r"unsigned comment",
        r"talk page",
        r"deletion review",
        r"please do not modify it",
    ]
    return any(re.search(p, text, flags=re.IGNORECASE) for p in patterns)

def has_repetition(text: str) -> bool:
    chunks = [c.strip().lower() for c in re.split(r"[.\n]", text) if c.strip()]
    return len(chunks) != len(set(chunks))

def has_role_leakage(text: str) -> bool:
    return "User:" in text or text.count("Assistant:") > 0

def word_count(text: str) -> int:
    return len(text.split())

def too_long(text: str, max_words: int = 40) -> bool:
    return word_count(text) > max_words

METRIC_REGISTRY: dict[str, MetricFn] = {
    "wiki_talk_artifact": has_wiki_talk_artifact,
    "repetition": has_repetition,
    "role_leakage": has_role_leakage,
    "word_count": word_count,
    "too_long": too_long,
}

def compute_metrics(answer: str, metric_names: list[str]) -> dict[str, Any]:
    metrics = {}

    for name in metric_names:
        if name not in METRIC_REGISTRY:
            raise KeyError(f"Unknown metric: {name}")

        metrics[name] = METRIC_REGISTRY[name](answer)

    return metrics