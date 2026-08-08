"""
Created on Sat Aug  1 12:57:42 2026

@author: Angelo Antonio Manzatto
"""

###############################################################################
# Libraries
###############################################################################

import re
from src.tasks.sft.conversation.metrics.base import MetricResult
from src.tasks.sft.conversation.metrics.utils import strip_trailing_tags

###############################################################################
# Constainr Satisfied Metric
###############################################################################

def constraint_satisfied(raw_answer: str, **kwargs) -> MetricResult:
    """
    Checks the constraint types actually used in the generated
    instruction_following data.
 
      - "single_number": exactly one token, and that token is a valid
        integer (digits only, optional leading minus). Distinct from
        "one_word" -- confirmed as a real, large, separate constraint
        in actual data ("Answer with a single number only. What is 15
        plus 11?"), not a subset of it: a malformed single-word answer
        like "many" would incorrectly PASS one_word's check even though
        it violates "must be a number specifically". Requires the
        answer to be in digit form ("26"), not spelled out ("twenty-
        six") -- the instruction says "number", read as numeral form,
        a real interpretive choice worth knowing about if it ever
        needs revisiting.
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
    """
    constraint_type = kwargs["constraint_type"]
    text = strip_trailing_tags(raw_answer).rstrip(".!? ")
 
    if constraint_type == "single_number":
        parts = text.split()
        passed = len(parts) == 1 and bool(re.fullmatch(r"-?\d+", parts[0]))
    elif constraint_type == "lowercase":
        letters = [c for c in text if c.isalpha()]
        passed = bool(letters) and all(c.islower() for c in letters)
    elif constraint_type == "one_word":
        passed = len(text.split()) == 1
    elif constraint_type == "yes_no":
        passed = text.lower() in ("yes", "no", "sim", "nao", "não")
    elif constraint_type == "uppercase":
        letters = [c for c in text if c.isalpha()]
        passed = bool(letters) and all(c.isupper() for c in letters)
    elif constraint_type == "max_words":
        constraint_value = kwargs["constraint_value"]
        passed = len(text.split()) <= int(constraint_value)
    elif constraint_type == "exact_item_count":
        # Heuristic, stated honestly: splits on commas and "and"/"e" to
        # approximate item count in a free-text list answer ("apple,
        # banana and orange" -> 3). Genuinely imprecise for answers that
        # don't follow simple list punctuation -- a coarse signal, not
        # a guarantee, same spirit as coherence's known blind spots.
        constraint_value = int(kwargs["constraint_value"])
        segments = re.split(r",|\be\b|\band\b", text, flags=re.IGNORECASE)
        item_count = len([s for s in segments if s.strip()])
        passed = item_count == constraint_value
    elif constraint_type == "sentence_count":
        constraint_value = int(kwargs["constraint_value"])
        sentences = [s for s in re.split(r"[.!?]+", text) if s.strip()]
        passed = len(sentences) == constraint_value
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
 