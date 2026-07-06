"""
Created on Sun Jun 28 16:32:28 2026

@author: Angelo Antonio Manzatto
"""

###############################################################################
# Libraries
###############################################################################

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.core.model.generation import greedy_decode
from src.tasks.sft.conversation.core.special_tokens import TOKEN_BY_NAME

###############################################################################
# Role → special token name mapping
###############################################################################

ROLE_TOKEN_NAMES = {
    "system":    "SYSTEM_TURN",
    "user":      "USER_TURN",
    "assistant": "ASSISTANT_TURN",
}

###############################################################################
# Text Generator
###############################################################################

@dataclass
class TextGenerator:
    model: Any
    tokenizer: Any
    decode_config: dict[str, Any]

    def messages_to_ids(self, messages: list[dict[str, str]]) -> list[int]:
        """
        Convert a messages list into a flat token ID sequence ready for inference.

        Completed turns (every turn except the final user turn):
            [ROLE_ID] + text_ids + [EOS_ID]

        Final user turn (generation trigger):
            [USER_ID] + text_ids + [ASST_ID]

        The model generates from there until it emits EOS_ID.
        """
        EOS_ID  = self.tokenizer.token_to_index["<EOS>"]
        ASST_ID = self.tokenizer.token_to_index[TOKEN_BY_NAME["ASSISTANT_TURN"].token]

        ids: list[int] = []

        for i, message in enumerate(messages):
            role_id  = self.tokenizer.token_to_index[
                TOKEN_BY_NAME[ROLE_TOKEN_NAMES[message["role"]]].token
            ]
            text_ids = self.tokenizer.text_to_indices(message["content"])

            ids.append(role_id)
            ids.extend(text_ids)

            if i < len(messages) - 1:
                ids.append(EOS_ID)   # close completed turn
            else:
                ids.append(ASST_ID)  # open assistant slot → generation starts here

        return ids

    def generate(self, messages: list[dict[str, str]]) -> str:
        strategy = self.decode_config.get("strategy", "greedy")

        if strategy == "greedy":
            return greedy_decode(
                model=self.model,
                input_ids=self.messages_to_ids(messages),
                tokenizer=self.tokenizer,
                max_length=int(self.decode_config.get("max_length", 64)),
                stop_token_ids={self.tokenizer.token_to_index["<EOS>"]},
                verbose=bool(self.decode_config.get("verbose", False)),
            )

        raise ValueError(f"Unsupported decoding strategy: {strategy!r}")