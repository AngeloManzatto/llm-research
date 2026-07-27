# -*- coding: utf-8 -*-
"""
Shared base for every metric: the MetricResult dataclass every metric
returns, plus helpers used by more than one metric file. A helper used
by only ONE metric stays local to that metric's own file — this module
is deliberately kept minimal, not a dumping ground for anything vaguely
reusable.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field


@dataclass(frozen=True)
class MetricResult:
    """
    `passed` is always True = "the desired behavior happened" — no
    polarity flag needed, no exception to remember. `details` is
    metric-specific and may be empty; nothing downstream should assume
    particular keys exist without checking.
    """
    passed: bool
    details: dict = field(default_factory=dict)


def strip_trailing_tags(text: str) -> str:
    """Strip trailing special-token-like tags (e.g. '<EOS>') before a
    constraint or coherence check, since raw_answer includes them but
    those checks would otherwise be silently corrupted by the suffix.
    Shared by constraint_satisfied.py and coherence.py."""
    return re.sub(r"(<[^>\s]+>\s*)+$", "", text).strip()
