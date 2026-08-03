"""
Created on Sat Aug  1 12:50:03 2026

@author: Angelo Antonio Manzatto
"""

###############################################################################
# Libraries
###############################################################################

from dataclasses import dataclass, field

###############################################################################
# Metric Result
###############################################################################

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


