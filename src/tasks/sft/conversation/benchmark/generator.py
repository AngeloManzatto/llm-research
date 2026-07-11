"""
Created on Sun Jun 28 16:32:28 2026

@author: Angelo Antonio Manzatto
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import tensorflow as tf

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

    # Resolved once at init time
    _eos_id:  int = field(init=False)
    _pad_id:  int = field(init=False)
    _asst_id: int = field(init=False)

    def __post_init__(self):
        self._eos_id  = self.tokenizer.token_to_index[TOKEN_BY_NAME["END_OF_TURN"].token]
        self._pad_id  = self.tokenizer.token_to_index["<PAD>"]
        self._asst_id = self.tokenizer.token_to_index[TOKEN_BY_NAME["ASSISTANT_TURN"].token]

    def messages_to_ids(self, messages: list[dict[str, str]]) -> list[int]:
        """Convert a messages list to a flat token ID sequence ending with ASST_ID."""
        ids: list[int] = []
        for i, message in enumerate(messages):
            role_id  = self.tokenizer.token_to_index[
                TOKEN_BY_NAME[ROLE_TOKEN_NAMES[message["role"]]].token
            ]
            text_ids = self.tokenizer.text_to_indices(message["content"])
            is_last  = (i == len(messages) - 1)

            ids.append(role_id)
            ids.extend(text_ids)
            if is_last:
                ids.append(self._asst_id)   # generation trigger
            else:
                ids.append(self._eos_id)    # close completed turn

        return ids

    def generate(self, messages: list[dict[str, str]]) -> str:
        """Generate a single completion. Thin wrapper around generate_batch."""
        return self.generate_batch([messages])[0]

    def generate_batch(self, batch_messages: list[list[dict[str, str]]]) -> list[str]:
        """
        Generate completions for a batch of message lists in a single
        batched forward pass loop — much faster than calling generate()
        in a Python loop.

        All prompt sequences are left-padded to the length of the longest
        one in the batch. Completed sequences (those that have emitted EOS)
        are masked out of subsequent forward passes.

        Returns a list of decoded completion strings, one per input.
        """
        max_length = int(self.decode_config.get("max_length", 20))
        PAD_ID     = self._pad_id
        EOS_ID     = self._eos_id

        # Build per-example prompt ID sequences
        all_ids = [self.messages_to_ids(msgs) for msgs in batch_messages]
        B        = len(all_ids)
        # Left-pad all sequences to the same length
        max_prompt_len = max(len(ids) for ids in all_ids)
        padded = np.full((B, max_prompt_len), PAD_ID, dtype=np.int32)
        for i, ids in enumerate(all_ids):
            padded[i, max_prompt_len - len(ids):] = ids  # left-pad

        prompt_len = max_prompt_len

        # Running sequences on GPU — shape [B, current_len]
        sequences = tf.constant(padded, dtype=tf.int32)

        # Track which sequences are still generating
        # [B] bool: True = still active (hasn't emitted EOS yet)
        active = tf.ones([B], dtype=tf.bool)

        # Store only the generated portion (after the prompt)
        generated = [[] for _ in range(B)]

        for _ in range(max_length):
            if not tf.reduce_any(active).numpy():
                break   # all sequences done

            logits   = self.model(sequences, training=False)  # [B, T, vocab]
            next_ids = tf.argmax(logits[:, -1, :], axis=-1,
                                 output_type=tf.int32)         # [B]

            next_ids_np = next_ids.numpy()
            active_np   = active.numpy()

            for i in range(B):
                if active_np[i]:
                    token = int(next_ids_np[i])
                    generated[i].append(token)
                    if token == EOS_ID:
                        active_np[i] = False

            active = tf.constant(active_np)

            # Append next tokens (use PAD for finished sequences)
            emit = np.where(active_np, next_ids_np, PAD_ID)
            sequences = tf.concat(
                [sequences, tf.constant(emit[:, None], dtype=tf.int32)],
                axis=1,
            )

        # Decode each generated sequence
        return [self.tokenizer.indices_to_text(g) for g in generated]