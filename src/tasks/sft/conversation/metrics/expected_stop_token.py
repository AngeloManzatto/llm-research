# -*- coding: utf-8 -*-
from __future__ import annotations

from .base import MetricResult


def expected_stop_token(raw_answer: str, **kwargs) -> MetricResult:
    """
    Pass if `raw_answer` ends with the expected stop token (e.g. "<EOS>").

    `raw_answer` must be the pre-normalization text — whatever normalizes
    the answer for display/scoring strips stop tokens out, so checking the
    normalized text would always fail here.

    The most foundational metric in the whole set — everything else
    (repetition, coherence, contains_expected...) implicitly assumes the
    sequence terminated somewhere sensible. This is the precondition
    that makes every other metric meaningful to even ask.
    """
    expected_token = kwargs["expected_token"]
    passed = raw_answer.rstrip().endswith(expected_token)
    return MetricResult(passed=passed, details={"expected_token": expected_token})

expected_stop_token.requires = ("expected_token",)
