"""
Special token registry.

This module defines the semantic meaning of every reserved token used by the
LLM project. These tokens form the communication protocol between datasets,
models, benchmarks and inference.

Conversation Protocol
---------------------
Every turn — system, user, or assistant — opens with a role marker and closes
with EOS. This mirrors the ChatML convention but uses pre-reserved SPECIAL
slots so no embedding rows need to be added or resized.

Stage 0 (no system prompt):

    <SPECIAL-11>user text<EOS><SPECIAL-12>assistant text<EOS>

Stage 3+ (system prompt introduced):

    <SPECIAL-10>system text<EOS><SPECIAL-11>user text<EOS><SPECIAL-12>assistant text<EOS>

At inference, feed everything up to and including <SPECIAL-12> and generate
until EOS is emitted. Only one stop token ID is needed by the decode loop.

Reserved but unused
-------------------
<SPECIAL-0>, <SPECIAL-1> : reserved for future use (e.g. tool calls, function
    results, multi-agent roles). Do not assign meaning without updating this file.

<BOS> : no longer used for conversation structure. Retained in the vocabulary
    for raw NTP pretraining sequences where it already has a learned prior.
"""

from dataclasses import dataclass

###############################################################################
# Token Definition
###############################################################################

@dataclass(frozen=True)
class SpecialToken:
    name: str
    token: str
    description: str
    stop_generation: bool = False

###############################################################################
# Conversation tokens
###############################################################################

SYSTEM_TURN = SpecialToken(
    name="SYSTEM_TURN",
    token="<SPECIAL-10>",
    description=(
        "Opens a system prompt turn. "
        "Introduced at Stage 3. Not used in Stage 0 data."
    ),
)

USER_TURN = SpecialToken(
    name="USER_TURN",
    token="<SPECIAL-11>",
    description=(
        "Opens a user turn. "
        "Every user message is immediately preceded by this token, "
        "no separator between token and content."
    ),
)

ASSISTANT_TURN = SpecialToken(
    name="ASSISTANT_TURN",
    token="<SPECIAL-12>",
    description=(
        "Opens an assistant turn. "
        "Every assistant message is immediately preceded by this token. "
        "At inference this is the generation trigger: feed the prompt up to "
        "and including this token, then generate until EOS."
    ),
)

END_OF_TURN = SpecialToken(
    name="END_OF_TURN",
    token="<EOS>",
    description=(
        "Closes any turn (system, user, or assistant). "
        "The single stop signal for the decode loop. "
        "EOS is used here because the model already has a strong pretraining "
        "prior for it as a sequence-end signal."
    ),
    stop_generation=True,
)

###############################################################################
# Registry
###############################################################################

SPECIAL_TOKENS = (
    SYSTEM_TURN,
    USER_TURN,
    ASSISTANT_TURN,
    END_OF_TURN,
)

TOKEN_BY_NAME   = {t.name: t for t in SPECIAL_TOKENS}
TOKEN_BY_STRING = {t.token: t for t in SPECIAL_TOKENS}

STOP_TOKENS = tuple(t.token for t in SPECIAL_TOKENS if t.stop_generation)
ROLE_TOKENS = tuple(t.token for t in SPECIAL_TOKENS if not t.stop_generation)