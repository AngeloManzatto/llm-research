# -*- coding: utf-8 -*-
from __future__ import annotations

from .base import MetricResult


def contains_expected(raw_answer: str, **kwargs) -> MetricResult:
    """
    True if raw_answer contains any one of the candidate strings in
    `expected_any` (case-insensitive substring match). `details` records
    which candidate actually matched, for diagnostics.

    Per the Metrics Contract v2.0: this metric gates `passed` only for
    the faithfulness categories (local_context, correction,
    instruction_following, uncertainty), where the correct value was
    genuinely given in the conversation. For the factual categories
    (knowledge_completion, turn_taking's factual subset) it stays
    computed and visible as a diagnostic, but does not gate — coherence
    (plus, eventually, the numeric type-match check) is the gate there.
    """
    expected_any = kwargs["expected_any"]
    answer_lower = raw_answer.lower()
    matched = next((exp for exp in expected_any if exp.lower() in answer_lower), None)
    return MetricResult(passed=matched is not None, details={"matched": matched} if matched else {})

contains_expected.requires = ("expected_any",)
