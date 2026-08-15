"""
Created on Fri Aug 14 22:24:50 2026

@author: Angelo Antonio Manzatto
"""

###############################################################################
# Libraries
###############################################################################

"""
Portuguese article and contraction resolution.

This module contains pure grammatical rules only.

Inputs:
    grammatical gender: "Masc" or "Fem"

Outputs:
    article/contraction tokens such as:
    "o", "a", "um", "uma", "do", "da", "no", "na"

This module does not know about Nodes, Facts, templates,
KnowledgeBase, or dataset generation.
"""

###############################################################################
# Globals
###############################################################################

VALID_GENDERS = {"Masc", "Fem"}

_ARTICLE_FORMS = {
    "definite":       {"Masc": "o",  "Fem": "a"},
    "indefinite":     {"Masc": "um", "Fem": "uma"},
    "contraction_de": {"Masc": "do", "Fem": "da"},
    "contraction_em": {"Masc": "no", "Fem": "na"},
    "contraction_a" : {"Masc": "ao", "Fem": "à"},
}

###############################################################################
# Validate
###############################################################################

def _validate_gender(gender: str) -> None:
    if gender not in VALID_GENDERS:
        raise ValueError(
            f"Unsupported Portuguese gender {gender!r}. "
            f"Expected one of {sorted(VALID_GENDERS)!r}."
        )

###############################################################################
# Articles
###############################################################################

def definite_article(gender: str) -> str:
    """
    Return the singular definite article.

    Masc -> o
    Fem -> a
    """
    _validate_gender(gender)
    return _ARTICLE_FORMS["definite"][gender]

def indefinite_article(gender: str) -> str:
    """
    Return the singular indefinite article.

    Masc -> um
    Fem -> uma
    """
    _validate_gender(gender)
    return _ARTICLE_FORMS["indefinite"][gender]

def contraction_de(gender: str) -> str:
    """
    Return the contraction of 'de' + definite article.

    Masc -> do
    Fem -> da
    """
    _validate_gender(gender)
    return _ARTICLE_FORMS["contraction_de"][gender]


def contraction_em(gender: str) -> str:
    """
    Return the contraction of 'em' + definite article.

    Masc -> no
    Fem -> na
    """
    _validate_gender(gender)
    return _ARTICLE_FORMS["contraction_em"][gender]

def contraction_a(gender: str) -> str:
    """
    Return the contraction of 'a' + definite article.

    Masc -> ao
    Fem  -> à
    """
    _validate_gender(gender)
    return _ARTICLE_FORMS["contraction_a"][gender]