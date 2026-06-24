"""
Created on Sun Jun 15 23:22:06 2025

@author: recruta42
"""
###############################################################################
# Libraries 
###############################################################################

import regex
import math
from collections import Counter
import pyphen
import zlib

from pipelines.ntp.common.text_processing.sentence_splitter import split_text_into_sentences

###############################################################################
# Stop Words
###############################################################################

ENGLISH_STOPWORDS = set("""
the of and to in a is that for it as was with on be by are this or from at an which
but not have has were their they had you his he she we been will would there what if
about can her all my one so up out who more when said them some
""".split())

PORTUGUESE_STOPWORDS = set("""
a à ao aos aonde aquilo abaixo acima após até atrás com como contra de do da dos das dum duma duns dumas
e é em entre era eram essa esse esta este estão foi foram fui há isso isto já lhe lhes mais mas me mesmo
mesma meus minhas minha meu na nas não nem no nos nós nossa nosso o os ou para pela pelo pelas pelos por
qual quando que quem se sem seu seus sob sobre sua suas também te tem têm tua tuas um uma umas uns vai vem
você vocês vos eu ele ela eles elas nós
""".split())

###############################################################################
# Libraries 
###############################################################################
def compute_entropy(text):
    if not text:
        return 0.0
    counts = Counter(text)
    total = len(text)
    return -sum((count / total) * math.log2(count / total) for count in counts.values())

# Compression ratio
def compute_compression_ratio(text):
    if not text:
        return 1.0
    original_size = len(text.encode("utf-8"))
    compressed_size = len(zlib.compress(text.encode("utf-8")))
    return compressed_size / original_size

# Character diversity
def compute_char_diversity(text):
    return len(set(text)) / max(1, len(text))

# Symbol ratio (non-alnum characters)
def compute_symbol_ratio(text):
    non_alnum = sum(1 for c in text if not c.isalnum() and not c.isspace())
    return non_alnum / max(1, len(text))

# Repetition percentage (3+ repeated chars)
def compute_repetition_percentage(text):
    repeated = regex.findall(r"(.)\1{2,}", text)
    return len(repeated) / max(1, len(set(text)))

# Punctuation anomalies
def detect_punctuation_anomalies(text):
    return bool(regex.search(r"([!?.,])\1{2,}", text))

# ASCII ratio
def compute_ascii_ratio(text):
    return sum(1 for c in text if ord(c) < 128) / max(1, len(text))

# Compute Flesch Szigrist score
def compute_flesch_day_szigrist_score(text, lang="en"):
    """
    Calculates a Flesch-style reading ease score using the same logic for both English and Portuguese.
    
    For Portuguese, this aligns with the Flesch–Day-Szigrist adaptation.
    For English, it mimics standard FRE but with consistent logic using Pyphen.
    
    Returns:
        float: The Flesch-style score (higher = easier)
    """
    lang = lang.lower()
    
    if lang == 'en':
        dic = pyphen.Pyphen(lang='en')
    elif lang == 'pt':
        dic = pyphen.Pyphen(lang='pt_BR')
    else:
        return -1.0
    
    # Unicode-safe word detection
    words = regex.findall(r"\p{L}+", text)
    total_words = len(words)
    if total_words == 0:
        return -1.0

    # Estimate syllables using Pyphen
    total_syllables = sum(len(dic.inserted(word).split("-")) for word in words)

    # Estimate sentence count
    sentences = split_text_into_sentences(text, language=lang)
    total_sentences = max(1, len(sentences))  # avoid division by zero

    # Apply Flesch–Day-Szigrist formula
    avg_words_per_sentence = total_words / total_sentences
    avg_syllables_per_word = total_syllables / total_words

    score = 206.835 - 1.015 * avg_words_per_sentence - 84.6 * avg_syllables_per_word
    
    return round(score, 2)

###############################################################################
# Quality Control (QC) 
###############################################################################
def compute_quality_metrics(text, lang):
    """
    Computes various quality control metrics for a document.
    Returns a dictionary of scores and booleans.
    """
    
    metrics = {}
    
    # Entropy metric
    metrics["entropy"] = compute_entropy(text)
    
    # Sentences count metrics
    sentences = split_text_into_sentences(text, language=lang)
    words = regex.findall(r"\w+", text)

    
    metrics["sentence_count"] = len(sentences)
    metrics["word_count"] = len(words)
    metrics["mean_sentence_length"] = len(words) / max(1, len(sentences))
    metrics["compression_ratio"] = compute_compression_ratio(text)
    metrics["char_diversity"] = compute_char_diversity(text)
    metrics["symbol_ratio"] = compute_symbol_ratio(text)
    metrics["repetition_percentage"] = compute_repetition_percentage(text)
    metrics["punctuation_anomalies"] = detect_punctuation_anomalies(text)
    metrics["ascii_ratio"] = compute_ascii_ratio(text)
    metrics["flesch_score"] = compute_flesch_day_szigrist_score(text, lang)
    
    # Stopword ratio
    metrics["stopword_ratio"]  = None
    metrics["stopword_coverage"] = None
    
    stopwords = None
    
    if lang == 'en':
        stopwords = ENGLISH_STOPWORDS
    if lang == 'pt':
        stopwords = PORTUGUESE_STOPWORDS
        
    if stopwords:
        word_list = [w.lower() for w in words]
        sw_count = sum(1 for w in word_list if w in stopwords)
        
        metrics["stopword_ratio"] = sw_count / max(1, len(word_list))
        metrics["stopword_coverage"] = int(sw_count > 0)

    return metrics