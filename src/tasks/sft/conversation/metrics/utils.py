"""
Created on Sat Aug  1 12:48:21 2026

@author: Angelo Antonio Manzatto
"""

###############################################################################
# Libraries
###############################################################################

import re 

###############################################################################
# Utility
###############################################################################

def strip_trailing_tags(text: str) -> str:
    """Strip trailing special-token-like tags (e.g. '<EOS>') before a
    constraint or coherence check, since raw_answer includes them but
    those checks would otherwise be silently corrupted by the suffix.
    Shared by constraint_satisfied.py and coherence.py."""
    return re.sub(r"(<[^>\s]+>\s*)+$", "", text).strip()
