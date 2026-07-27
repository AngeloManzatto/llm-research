# -*- coding: utf-8 -*-
from __future__ import annotations

import re

from .base import MetricResult


def _tokenize_for_repetition(text: str) -> list[str]:
    """Tokenize while preserving contractions ("don't"); punctuation ignored."""
    return re.findall(r"\b[\w]+(?:['’][\w]+)?\b", text.lower())


def _detect_consecutive_repetition(text: str) -> dict:
    """
    Detect a word or phrase repeating consecutively, searching longest
    phrases first so the reported match is the most descriptive one
    (e.g. a whole repeated clause, not a fragment of it).

    Threshold is graduated by phrase length, not a single fixed number:

      - 1-2 word phrases need 3+ consecutive repeats to flag. A uniform
        threshold of 2 here flags ordinary English doubling as a bug —
        tested directly: "no no thanks", "very very good", and
        "I think I think that is right" all false-positive under a
        flat min_repetitions=2.
      - 3+ word phrases flag on 2 consecutive repeats, since a multi-word
        clause repeating even once more is already a strong degeneration
        signal (e.g. "I know about I know about").

    No cap on phrase length. A fixed max_ngram_size (tested at both 4 and
    8) misses any repeated clause longer than that cap regardless of
    repeat count — confirmed against a real failure case, "the woman saw
    it coming the woman saw it coming" (5 words), and against a
    constructed 9-word clause, both invisible to a capped search. This
    searches every phrase length up to half the token count.

    Returns a dict: {"repeated": bool, "phrase": str, "repetitions": int,
    "ngram_size": int}. phrase/repetitions/ngram_size are empty/0 when
    no repetition is found.
    """
    tokens = _tokenize_for_repetition(text)
    if not tokens:
        return {"repeated": False, "phrase": "", "repetitions": 0, "ngram_size": 0}

    max_ngram_size = len(tokens) // 2

    for ngram_size in range(max_ngram_size, 0, -1):
        min_repetitions = 3 if ngram_size <= 2 else 2
        for start in range(len(tokens) - ngram_size * min_repetitions + 1):
            phrase = tokens[start : start + ngram_size]
            repetitions = 1
            cursor = start + ngram_size
            while (
                cursor + ngram_size <= len(tokens)
                and tokens[cursor : cursor + ngram_size] == phrase
            ):
                repetitions += 1
                cursor += ngram_size

            if repetitions >= min_repetitions:
                return {
                    "repeated": True,
                    "phrase": " ".join(phrase),
                    "repetitions": repetitions,
                    "ngram_size": ngram_size,
                }

    return {"repeated": False, "phrase": "", "repetitions": 0, "ngram_size": 0}


def repetition(raw_answer: str, **kwargs) -> MetricResult:
    """
    passed=True means NO degenerate repetition was found — kept consistent
    with every other metric's "True = desired behavior" convention, unlike
    an earlier version of this metric where True meant a defect was found.
    """
    detail = _detect_consecutive_repetition(raw_answer)
    repeated = detail.pop("repeated")
    return MetricResult(passed=not repeated, details=detail if repeated else {})

repetition.requires = ()
