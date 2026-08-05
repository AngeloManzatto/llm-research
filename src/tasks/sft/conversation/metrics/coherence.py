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
# Spell Checker
###############################################################################

try:
    from spellchecker import SpellChecker
    _SPELL_CHECKERS = {"en": SpellChecker(language="en"), "pt": SpellChecker(language="pt")}
except ImportError:
    _SPELL_CHECKERS = {}

_WORD_RE = re.compile(r"[a-zA-ZÀ-ÿ]+")

###############################################################################
# Coherece 
###############################################################################

def coherence(raw_answer: str, *, language: str, **kwargs) -> MetricResult:
    """
    A cheap, real (dictionary-based, not LLM-judged) structural check,
    distinct from repetition/expected_stop_token. Checks two things,
    deliberately narrow in scope:

      1. Near-empty output — answer reduces to fewer than 2 alphabetic
         characters after stripping tags. Catches the observed PT
         knowledge_completion collapse to ".", "a.", "ta.".
      2. Invented/non-dictionary words — lowercase alphabetic tokens
         (length >= 2) not found in a real EN/PT dictionary
         (pyspellchecker). Catches genuine gibberish confirmed by hand:
         "meish", "forbon", "paintup".

    KNOWN BLIND SPOT, stated honestly rather than papered over: a short
    fragment that happens to BE a real dictionary word in the target
    language (e.g. PT "ta", a valid informal contraction of "está")
    will pass even when used as a degenerate, contentless answer to a
    real question — dictionary + length checks alone can't distinguish
    "legitimately short answer" from "short garbage that happens to be
    a real word". This metric doesn't try to solve that; it catches
    non-words, not all degenerate short answers.

    Capitalized tokens and digits are deliberately EXCLUDED from the
    word check: they're usually proper nouns (names, places) that
    legitimately won't be in any dictionary, and flagging them would
    produce a large false-positive rate on entirely correct answers
    (Whiskers, Pemberton, Paris, Tóquio, ...).

    This does NOT check grammar, factual correctness, or topical
    relevance — those need much heavier tooling (a real parser, a
    fact-checker, a semantic judge; see the BLiMP/spaCy discussion in
    Metrics Contract v2.0) and are out of scope for a cheap,
    deterministic metric. A row can pass this and still be nonsensical
    in other ways; it can also fail due to a genuinely correct but
    rare word absent from the dictionary. Coarse, cheap signal — not a
    quality guarantee.

    Per Metrics Contract v2.0: this is the gate for the factual
    categories (knowledge_completion, turn_taking's factual subset),
    replacing contains_expected there — Stage 0 measures whether the
    response is coherently structured, not whether the specific fact
    is correct.
    """
    text = strip_trailing_tags(raw_answer).strip()
 
    # alphanumeric, not just alphabetic -- a purely numeric answer like
    # "1945" (a real, substantive year) has zero LETTERS but is not
    # remotely empty. Confirmed real false negative: "1945." for "In
    # what year did World War II end?" was failing near_empty before
    # this fix, purely because digits weren't counted as content.
    alnum_chars = [c for c in text if c.isalnum()]
    if len(alnum_chars) < 1:
        return MetricResult(passed=False, details={"reason": "near_empty", "text": text})
 
    checker = _SPELL_CHECKERS.get(language)
    if checker is None:
        # No dictionary for this language — the near-empty check above
        # still ran, but the word check can't. Pass rather than penalize
        # an unsupported language for a check it was never able to do.
        return MetricResult(passed=True, details={"reason": "no_dictionary_for_language"})
 
    tokens = _WORD_RE.findall(text)
    lowercase_tokens = [t for t in tokens if t.islower() and len(t) >= 2]
    unknown = checker.unknown(lowercase_tokens)
 
    if unknown:
        return MetricResult(passed=False, details={"reason": "invented_words", "words": sorted(unknown)})
 
    return MetricResult(passed=True, details={})

coherence.requires = ("language",)
