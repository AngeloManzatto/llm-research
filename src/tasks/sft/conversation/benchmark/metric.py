"""
Created on Sun Jun 28 10:34:27 2026

@author: Angelo Antonio Manzatto
"""

###############################################################################
# Libraries
###############################################################################

import re
from dataclasses import dataclass, field
from typing import Callable

###############################################################################
# Metric Result
###############################################################################

@dataclass(frozen=True)
class MetricResult:
    passed: bool
    details: dict = field(default_factory=dict)

###############################################################################
# Expected Stop Token
###############################################################################

def expected_stop_token(raw_answer: str, **kwargs) -> MetricResult:
    """
    Pass if `raw_answer` ends with the expected stop token (e.g. "<EOS>").

    `raw_answer` must be the pre-normalization text — whatever normalizes
    the answer for display/scoring strips stop tokens out, so checking the
    normalized text would always fail here.
    """
    expected_token = kwargs["expected_token"]
    passed = raw_answer.rstrip().endswith(expected_token)
    return MetricResult(passed=passed, details={"expected_token": expected_token})

expected_stop_token.requires = ("expected_token",)

###############################################################################
# Repetition
###############################################################################

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

###############################################################################
# Contains Expected
###############################################################################

def contains_expected(raw_answer: str, **kwargs) -> MetricResult:
    """
    True if raw_answer contains any one of the candidate strings in
    `expected_any` (case-insensitive substring match). `details` records
    which candidate actually matched, for diagnostics.
    """
    expected_any = kwargs["expected_any"]
    answer_lower = raw_answer.lower()
    matched = next((exp for exp in expected_any if exp.lower() in answer_lower), None)
    return MetricResult(passed=matched is not None, details={"matched": matched} if matched else {})

contains_expected.requires = ("expected_any",)

###############################################################################
# Constraint Satisfied
###############################################################################

def _strip_trailing_tags(text: str) -> str:
    """Strip trailing special-token-like tags (e.g. '<EOS>') before a
    constraint check, since raw_answer includes them but an exact-match or
    casing check would otherwise be silently corrupted by that suffix."""
    return re.sub(r"(<[^>\s]+>\s*)+$", "", text).strip()


def constraint_satisfied(raw_answer: str, **kwargs) -> MetricResult:
    """
    Checks the four constraint types actually used in the generated
    instruction_following data — not a speculative larger set:

      - "one_word": exactly one word.
      - "yes_no": answer is exactly yes/no (en or pt).
      - "uppercase": every alphabetic character is uppercase.
      - "exact_match": answer equals constraint_value exactly (used for
        both word-echo and count-sequence constraints — both are really
        the same check, "must equal this given string", so kept as one
        type rather than two).

    constraint_value is required only for "exact_match"; other types
    don't take one.
    """
    constraint_type = kwargs["constraint_type"]
    text = _strip_trailing_tags(raw_answer).rstrip(".!? ")

    if constraint_type == "one_word":
        passed = len(text.split()) == 1
    elif constraint_type == "yes_no":
        passed = text.lower() in ("yes", "no", "sim", "nao", "não")
    elif constraint_type == "uppercase":
        letters = [c for c in text if c.isalpha()]
        passed = bool(letters) and all(c.isupper() for c in letters)
    elif constraint_type == "exact_match":
        constraint_value = kwargs["constraint_value"]
        passed = text == str(constraint_value).rstrip(".!? ")
    else:
        raise ValueError(f"Unknown constraint_type: {constraint_type!r}")

    return MetricResult(passed=passed, details={"constraint_type": constraint_type})

constraint_satisfied.requires = ("constraint_type",)

###############################################################################
# Registry
###############################################################################

METRICS: dict[str, Callable[..., MetricResult]] = {
    "expected_stop_token": expected_stop_token,
    "repetition":          repetition,
    "contains_expected":   contains_expected,
    "constraint_satisfied": constraint_satisfied,
}

def run_metric(metric_id: str, raw_answer: str, **kwargs) -> MetricResult:
    if metric_id not in METRICS:
        raise KeyError(f"Unknown metric: '{metric_id}'. Available: {sorted(METRICS)}")

    fn = METRICS[metric_id]
    missing = [k for k in getattr(fn, "requires", ()) if k not in kwargs]
    if missing:
        raise ValueError(f"'{metric_id}' missing required fields: {missing}")

    return fn(raw_answer, **kwargs)
 