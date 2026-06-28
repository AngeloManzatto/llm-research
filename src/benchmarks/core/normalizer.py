"""
Created on Sun Jun 28 17:50:07 2026

@author: Angelo Antonio Manzatto
"""

###############################################################################
# Libraries
###############################################################################

from __future__ import annotations

import re

from src.benchmarks.core.special_tokens import STOP_TOKENS

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