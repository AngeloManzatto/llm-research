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
from src.core.model.generation import (
    greedy_decode,
    top_k_decode,
    nucleus_decode,
    beam_search_decode,
)

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
            ids.append(self._asst_id if is_last else self._eos_id)

        return ids

    def generate(self, messages: list[dict[str, str]]) -> str:
        """Generate a single completion. Thin wrapper around generate_batch."""
        return self.generate_batch([messages])[0]

    def generate_batch(self, batch_messages: list[list[dict[str, str]]]) -> list[str]:
        """
        Generate completions for a batch of message lists.

        Strategy is read from decode_config["strategy"]:
            "greedy"      — batched, deterministic (default, best for benchmark)
            "top_k"       — batched, stochastic (best for qualitative inspection)
            "nucleus"     — batched, stochastic (best for qualitative inspection)
            "beam_search" — per-example (not batchable), deterministic

        greedy / top_k / nucleus run a single batched forward-pass loop.
        beam_search falls back to sequential per-example calls since beam
        candidates cannot be batched without significant complexity.
        """
        strategy   = self.decode_config.get("strategy", "greedy")
        max_length = int(self.decode_config.get("max_length", 20))
        stop_ids   = {self._eos_id}

        # --- beam search: sequential, one example at a time ---
        if strategy == "beam_search":
            beam_width = int(self.decode_config.get("beam_width", 3))
            return [
                beam_search_decode(
                    self.model, self.tokenizer,
                    input_ids=self.messages_to_ids(msgs),
                    stop_token_ids=stop_ids,
                    beam_width=beam_width,
                    max_length=max_length,
                )
                for msgs in batch_messages
            ]

        # --- batched strategies: greedy, top_k, nucleus ---
        PAD_ID = self._pad_id
        EOS_ID = self._eos_id

        all_ids        = [self.messages_to_ids(msgs) for msgs in batch_messages]
        B              = len(all_ids)
        max_prompt_len = max(len(ids) for ids in all_ids)

        # Left-pad all prompts to the same length
        padded = np.full((B, max_prompt_len), PAD_ID, dtype=np.int32)
        for i, ids in enumerate(all_ids):
            padded[i, max_prompt_len - len(ids):] = ids

        sequences = tf.constant(padded, dtype=tf.int32)
        active    = np.ones(B, dtype=bool)
        generated = [[] for _ in range(B)]

        # Strategy-specific config
        k = int(self.decode_config.get("k", 5))
        p = float(self.decode_config.get("p", 0.9))

        for _ in range(max_length):
            if not active.any():
                break

            logits = self.model(sequences, training=False)  # [B, T, vocab]
            last   = logits[:, -1, :]                       # [B, vocab]

            # --- select next token per strategy ---
            if strategy == "greedy":
                next_ids_np = tf.argmax(last, axis=-1,
                                        output_type=tf.int32).numpy()

            elif strategy == "top_k":
                top_probs, top_indices = tf.math.top_k(last, k=k)
                sampled     = tf.random.categorical(
                    tf.math.log(tf.nn.softmax(top_probs)), num_samples=1
                )                                           # [B, 1]
                next_ids_np = tf.gather(
                    top_indices, sampled, batch_dims=1
                ).numpy().squeeze(-1).astype(np.int32)

            elif strategy == "nucleus":
                # Per-example nucleus sampling (vectorised across batch)
                sorted_logits  = tf.sort(last, direction="DESCENDING")
                sorted_indices = tf.argsort(last, direction="DESCENDING")
                probs          = tf.nn.softmax(sorted_logits)
                cumulative     = tf.cumsum(probs, axis=-1)
                # Shift so the token that pushes us past p is included
                shifted        = tf.concat(
                    [tf.zeros_like(cumulative[:, :1]), cumulative[:, :-1]], axis=-1
                )
                nucleus_mask   = shifted < p
                neg_inf        = tf.fill(tf.shape(sorted_logits), float("-inf"))
                masked_logits  = tf.where(nucleus_mask, sorted_logits, neg_inf)
                sampled        = tf.random.categorical(
                    tf.cast(masked_logits, tf.float32), num_samples=1
                )                                           # [B, 1]
                next_ids_np    = tf.gather(
                    sorted_indices, sampled, batch_dims=1
                ).numpy().squeeze(-1).astype(np.int32)

            else:
                raise ValueError(
                    f"Unknown strategy {strategy!r}. "
                    "Choose: 'greedy', 'top_k', 'nucleus', 'beam_search'."
                )

            # --- update generated tokens and active flags ---
            for i in range(B):
                if active[i]:
                    token = int(next_ids_np[i])
                    generated[i].append(token)
                    if token == EOS_ID:
                        active[i] = False

            # Append next token (PAD for finished sequences)
            emit      = np.where(active, next_ids_np, PAD_ID)
            sequences = tf.concat(
                [sequences, tf.constant(emit[:, None], dtype=tf.int32)],
                axis=1,
            )

        return [self.tokenizer.indices_to_text(g) for g in generated]