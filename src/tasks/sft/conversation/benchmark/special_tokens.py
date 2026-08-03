"""
Created on Sun Aug  2 20:52:52 2026

@author: Angelo Antonio Manzatto
"""

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

###############################################################################
# Libraries
###############################################################################

from types import MappingProxyType
from typing import Any

TOKEN_BY_NAME = MappingProxyType({
    "SYSTEM_TURN":    "<SPECIAL-10>",
    "USER_TURN":      "<SPECIAL-11>",
    "ASSISTANT_TURN": "<SPECIAL-12>",
    "END_OF_TURN":    "<EOS>",
})
 

ROLE_TOKENS = MappingProxyType({
    "system":    TOKEN_BY_NAME["SYSTEM_TURN"],
    "user":      TOKEN_BY_NAME["USER_TURN"],
    "assistant": TOKEN_BY_NAME["ASSISTANT_TURN"],
})
 
###############################################################################
# Resolve Special Tokens
###############################################################################

def resolve_special_tokens(
    tokenizer,
    token_by_name: dict = TOKEN_BY_NAME,
    role_tokens: dict[str, str] = ROLE_TOKENS,
) -> dict[str, Any]:
    """
    Returns a plain dict with everything generate_batch/decode_loop/
    messages_to_ids need:
        {
            "eos_id": int,
            "asst_id": int,
            "pad_id": int,
            "role_token_ids": {"system": int, "user": int, "assistant": int},
            "stop_ids": {eos_id},
        }
    """
    eos_id  = tokenizer.token_to_index[token_by_name["END_OF_TURN"]]
    asst_id = tokenizer.token_to_index[token_by_name["ASSISTANT_TURN"]]
    pad_id  = tokenizer.token_to_index["<PAD>"]
 
    role_token_ids = {
        role: tokenizer.token_to_index[token]
        for role, token in role_tokens.items()
    }
 
    return {
        "eos_id": eos_id,
        "asst_id": asst_id,
        "pad_id": pad_id,
        "role_token_ids": role_token_ids,
        "stop_ids": {eos_id},
    }