"""
Created on Sun Jun 15 22:38:38 2025

@author: recruta42
"""
###############################################################################
# Libraries
###############################################################################
import math
import zlib
import regex
from collections import Counter

###############################################################################
# Global filter config
###############################################################################
default_config = {
    "min_chars": 200,
    "max_chars": 1_000_000,
    "min_words": 30,
    "min_sentences": 3,
    "min_entropy": 3.0,
    "min_char_diversity": 0.3,
    "max_repetition": 30,
    "max_symbol_ratio": 0.3,
    "use_language_detection": True,
    "allowed_languages": ("en", "pt"),
    "use_compression_filter": True,
    "use_garbage_filter": True,
    "stopwords": None,
}

###############################################################################
# Filters
###############################################################################

def filter_by_length(text, min_chars=100, max_chars=1_000_000):
    """
    Filter out documents that are too short or too long.
    """
    length = len(text.strip())
    return min_chars <= length <= max_chars

def filter_by_stopword_ratio(text, stopwords, min_ratio=0.01, max_ratio=0.8):
    """
    Filters documents with abnormally low or high stopword ratios.
    """
    words = text.split()
    if not words:
        return False
    
    stop_count = sum(1 for w in words if w.lower() in stopwords)
    ratio = stop_count / len(words)
    return min_ratio <= ratio <= max_ratio

def filter_by_compression_ratio(text, min_ratio=1.1):
    """
    Filters out low-entropy text based on compression ratio.
    """
    if not text.strip():
        return False
    original_size = len(text.encode("utf-8"))
    compressed_size = len(zlib.compress(text.encode("utf-8")))
    ratio = original_size / compressed_size if compressed_size > 0 else 0
    return ratio >= min_ratio

def filter_garbage_patterns(text):
    """
    Filters out garbage patterns like long hexadecimal dumps or binary blobs.
    """
    hex_pattern = regex.compile(r'\b[0-9a-fA-F]{20,}\b')
    bin_pattern = regex.compile(r'\b[01]{20,}\b')
    return not (hex_pattern.search(text) or bin_pattern.search(text))

def filter_by_symbol_ratio(text, max_ratio=0.3):
    clean = regex.sub(r'\s+', '', text)
    total = len(clean)
    if total == 0:
        return False
    symbols = regex.findall(r'[\p{S}\p{P}]', clean)
    return (len(symbols) / total) <= max_ratio

def filter_by_entropy(text, min_entropy=3.0):
    if not text:
        return False
    counts = Counter(text)
    total = len(text)
    entropy = -sum((count / total) * math.log2(count / total)
                   for count in counts.values())
    return entropy >= min_entropy

def filter_by_sentence_count(text, min_sentences=3):
    # Rough approximation using punctuation
    sentence_count = len(regex.findall(r'[\p{Lu}][^.!?]*[.!?]', text))
    return sentence_count >= min_sentences

###############################################################################
# Filter pipeline
###############################################################################

def apply_filter(name, func, text, config, logger):
    if not func(text, config):
        if logger:
            logger.info(f"⛔ Rejected: {name}")
        return False
    return True

def filter_text(text, config=None, logger=None):
    """
    Applies all configured filters to the input text.
    
    Returns:
        bool: True if text passes all filters.
    """
    if config is None:
        config = default_config

    text = text.strip()
    if not text:
        return False

    filters = [
        ("length", lambda t, c: filter_by_length(t, c["min_chars"], c["max_chars"])),
        ("word_count", lambda t, c: len(t.split()) >= c["min_words"]),
        ("sentence_count", lambda t, c: filter_by_sentence_count(t, c["min_sentences"])),
        ("entropy", lambda t, c: filter_by_entropy(t, c["min_entropy"])),
        ("symbol_ratio", lambda t, c: filter_by_symbol_ratio(t, c["max_symbol_ratio"])),
        ("compression", lambda t, c: not c.get("use_compression_filter") or filter_by_compression_ratio(t)),
        ("garbage", lambda t, c: not c.get("use_garbage_filter") or filter_garbage_patterns(t)),
        ("stopword_ratio", lambda t, c: True if not c.get("stopwords") else filter_by_stopword_ratio(t, c["stopwords"])),
    ]

    for name, func in filters:
        if not apply_filter(name, func, text, config, logger):
            return False

    return True