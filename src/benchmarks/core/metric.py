"""
Created on Sun Jun 28 10:34:27 2026

@author: Angelo Antonio Manzatto
"""

###############################################################################
# Libraries
###############################################################################

from __future__ import annotations

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
# Metric
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