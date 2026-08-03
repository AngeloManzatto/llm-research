"""
Created on Sat Aug  1 12:57:42 2026

@author: Angelo Antonio Manzatto
"""

###############################################################################
# Libraries
###############################################################################

from src.tasks.sft.conversation.metrics.base import MetricResult
from src.tasks.sft.conversation.metrics.utils import strip_trailing_tags

###############################################################################
# Constainr Satisfied Metric
###############################################################################

def constraint_satisfied(raw_answer: str, **kwargs) -> MetricResult:
    """
    Checks the constraint types actually used in the generated
    instruction_following data — not a speculative larger set:

      - "one_word": exactly one word.
      - "yes_no": answer is exactly yes/no (en or pt).
      - "uppercase": every alphabetic character is uppercase.
      - "exact_match": answer equals constraint_value exactly (used for
        both word-echo and count-sequence constraints — both are really
        the same check, "must equal this given string", so kept as one
        type rather than two).
      - "reverse_word": answer equals constraint_value spelled backwards
        (e.g. constraint_value="trombeta" -> answer must be "atebmort").
        Reverses constraint_value itself rather than trusting a
        precomputed answer stored separately — the ground truth only
        needs the original word, so there's no way for the stored
        "correct" reversal to silently drift out of sync with it.

    constraint_value is required for "exact_match" and "reverse_word";
    other types don't take one.

    NOTE per Metrics Contract v2.0: this metric is ready, but the
    instruction_following ground-truth data isn't — rows still lack
    constraint_type/constraint_value tags in practice, so this has
    nothing real to check yet. Migrating the data is separate, still-
    open work.
    """
    constraint_type = kwargs["constraint_type"]
    text = strip_trailing_tags(raw_answer).rstrip(".!? ")

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
    elif constraint_type == "reverse_word":
        constraint_value = kwargs["constraint_value"]
        expected_reversed = str(constraint_value).strip()[::-1]
        passed = text.lower() == expected_reversed.lower()
    else:
        raise ValueError(f"Unknown constraint_type: {constraint_type!r}")

    return MetricResult(passed=passed, details={"constraint_type": constraint_type})

constraint_satisfied.requires = ("constraint_type",)
