"""
Created on Sat Aug 15 08:52:51 2026

@author: Angelo Antonio Manzatto
"""

###############################################################################
# Libraries
###############################################################################


###############################################################################
# Globals
###############################################################################

VALID_GENDERS = {"Masc", "Fem"}

_POSSESSIVE_FORMS = {
    "possessive": {
        "Masc": "meu",
        "Fem": "minha",
    },
    "de_possessive": {
        "Masc": "do meu",
        "Fem": "da minha",
    },
    "em_possessive": {
        "Masc": "no meu",
        "Fem": "na minha",
    },
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
# Possessives
###############################################################################

def possessive(gender: str) -> str:
    """
    Return the singular first-person possessive determiner.

    Masc -> meu
    Fem  -> minha
    """
    _validate_gender(gender)
    return _POSSESSIVE_FORMS["possessive"][gender]

def contraction_de_possessive(gender: str) -> str:
    """
    Return 'de' + definite article + first-person possessive.

    Masc -> do meu
    Fem  -> da minha
    """
    _validate_gender(gender)
    return _POSSESSIVE_FORMS["de_possessive"][gender]

def contraction_em_possessive(gender: str) -> str:
    """
    Return 'em' + definite article + first-person possessive.

    Masc -> no meu
    Fem  -> na minha
    """
    _validate_gender(gender)
    return _POSSESSIVE_FORMS["em_possessive"][gender]