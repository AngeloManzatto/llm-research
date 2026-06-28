"""
Created on Sun Jun 28 16:24:33 2026

@author: Angelo Antonio Manzatto
"""

###############################################################################
# Libraries
###############################################################################

"""
Special token registry.

This module defines the semantic meaning of every reserved token used by the
LLM project. These tokens form the communication protocol between datasets,
models, benchmarks and inference.

Special tokens are intentionally versioned and documented here so their meaning
never depends on hardcoded strings scattered throughout the codebase.
"""

from dataclasses import dataclass

###############################################################################
# Token Definition
###############################################################################

@dataclass(frozen=True)
class SpecialToken:
    """
    Definition of one reserved special token.
    """

    name: str
    token: str
    description: str

    stop_generation: bool = False

###############################################################################
# Conversation Tokens
###############################################################################

END_OF_TURN = SpecialToken(
    name="END_OF_TURN",
    token="<SPECIAL-0>",
    description=(
        "Marks the end of a single assistant turn. "
        "Default terminator for every assistant turn, including the only "
        "assistant turn in a single-turn example."
    ),
    stop_generation=True,
)

END_OF_CONVERSATION = SpecialToken(
    name="END_OF_CONVERSATION",
    token="<SPECIAL-1>",
    description=(
        "Marks the end of a complete multi-turn conversation. "
        "Only used on the final assistant turn of an explicitly multi-turn "
        "training example (two or more user/assistant exchanges)."
    ),
    stop_generation=True,
)

###############################################################################
# Registry
###############################################################################

SPECIAL_TOKENS = (
    END_OF_TURN,
    END_OF_CONVERSATION,
)


TOKEN_BY_NAME = {
    token.name: token
    for token in SPECIAL_TOKENS
}

TOKEN_BY_STRING = {
    token.token: token
    for token in SPECIAL_TOKENS
}


STOP_TOKENS = tuple(
    token.token
    for token in SPECIAL_TOKENS
    if token.stop_generation
)