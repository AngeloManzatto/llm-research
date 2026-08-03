"""
Created on Sun Jun 28 16:32:28 2026

@author: Angelo Antonio Manzatto
"""

###############################################################################
# Libraries
###############################################################################

from typing import Any, Callable
from functools import partial
import numpy as np
import tensorflow as tf

###############################################################################
# 1. Token selection -- pure functions, no model, no loop bookkeeping
###############################################################################
 
def select_next_token_greedy(logits: tf.Tensor) -> tf.Tensor:
    """
    logits: [B, vocab] -- raw logits for the next position, any batch size.
    Returns: [B] int32 -- the argmax token ID per row.
    """
    return tf.argmax(logits, axis=-1, output_type=tf.int32)
 
def select_next_token_top_k(logits: tf.Tensor, k: int) -> tf.Tensor:
    """
    logits: [B, vocab] -- raw logits for the next position, any batch size.
    Samples one token per row from the k highest-logit candidates.
 
    Passes the top-k logits directly to tf.random.categorical, which
    applies softmax internally. The previous version of this function
    (in the old generation.py) applied log() directly to the raw,
    unsoftmaxed top-k logit values -- confirmed to produce NaN whenever
    a negative logit lands in the top-k (routine for real model
    output), silently breaking sampling. Passing logits straight to
    categorical avoids this entirely and is mathematically equivalent
    to the intended distribution.
 
    Returns: [B] int32 -- the sampled token ID per row.
    """
    top_logits, top_indices = tf.math.top_k(logits, k=k)          # [B, k] each
    sampled_pos = tf.random.categorical(top_logits, num_samples=1)  # [B, 1]
    return tf.gather(top_indices, sampled_pos, batch_dims=1)[:, 0]  # [B]
 
def select_next_token_nucleus(logits: tf.Tensor, p: float) -> tf.Tensor:
    """
    logits: [B, vocab] -- raw logits for the next position, any batch size.
    Samples one token per row from the smallest set of highest-probability
    tokens whose cumulative probability reaches p (the "nucleus").
 
    Returns: [B] int32 -- the sampled token ID per row.
    """
    sorted_logits  = tf.sort(logits, direction="DESCENDING")            # [B, vocab]
    sorted_indices = tf.argsort(logits, direction="DESCENDING")          # [B, vocab]
 
    probs      = tf.nn.softmax(sorted_logits)
    cumulative = tf.cumsum(probs, axis=-1)
 
    shifted = tf.concat(
        [tf.zeros_like(cumulative[:, :1]), cumulative[:, :-1]], axis=-1
    )
    nucleus_mask = shifted < p
 
    neg_inf       = tf.fill(tf.shape(sorted_logits), float("-inf"))
    masked_logits = tf.where(nucleus_mask, sorted_logits, neg_inf)
 
    sampled_pos = tf.random.categorical(masked_logits, num_samples=1)     # [B, 1]
    return tf.gather(sorted_indices, sampled_pos, batch_dims=1)[:, 0]      # [B]
 
 
###############################################################################
# 2. Tokenization -- messages list -> flat token ID sequence
###############################################################################
 
def messages_to_ids(
    messages: list[dict[str, str]],
    *,
    text_to_indices: Callable[[str], list[int]],
    role_token_ids: dict[str, int],
    eos_id: int,
    asst_id: int,
) -> list[int]:
    """
    Each message becomes: [role_token, ...text_tokens, boundary_token].
    boundary_token is asst_id for the LAST message (the generation
    trigger), eos_id for every earlier message.
    """
    ids: list[int] = []
    for i, message in enumerate(messages):
        role_id  = role_token_ids[message["role"]]
        text_ids = text_to_indices(message["content"])
        is_last  = (i == len(messages) - 1)
 
        ids.append(role_id)
        ids.extend(text_ids)
        ids.append(asst_id if is_last else eos_id)
 
    return ids
 
 
###############################################################################
# 3. Decode loop -- the one shared generation loop
###############################################################################
 
def decode_loop(
    model,
    input_ids_batch: list[list[int]],
    select_fn: Callable[[tf.Tensor], tf.Tensor],
    pad_id: int,
    stop_ids: set[int],
    max_length: int,
) -> list[list[int]]:
    """
    stop_ids: any of these token IDs halts that row's generation. A
    single END_OF_TURN id AND a role-switch id (USER_TURN/SYSTEM_TURN,
    signaling the model has gone off the rails) can both be included --
    matches the manifest's own "stop_tokens" list and the set[int]
    stop_token_ids the original per-example functions already used.
 
    Returns: one list of generated token ids per example (prompt
    excluded), including whichever stop_ids member was emitted if one
    was.
    """
    B = len(input_ids_batch)
    max_prompt_len = max(len(ids) for ids in input_ids_batch)
 
    padded = np.full((B, max_prompt_len), pad_id, dtype=np.int32)
    for i, ids in enumerate(input_ids_batch):
        padded[i, max_prompt_len - len(ids):] = ids
 
    sequences = tf.constant(padded, dtype=tf.int32)
    active    = np.ones(B, dtype=bool)
    generated: list[list[int]] = [[] for _ in range(B)]
 
    for _ in range(max_length):
        if not active.any():
            break
 
        logits = model(sequences, training=False)  # [B, T, vocab]
        last   = logits[:, -1, :]                   # [B, vocab]
 
        next_ids = select_fn(last).numpy()
 
        for i in range(B):
            if active[i]:
                token = int(next_ids[i])
                generated[i].append(token)
                if token in stop_ids:
                    active[i] = False
 
        emit      = np.where(active, next_ids, pad_id)
        sequences = tf.concat(
            [sequences, tf.constant(emit[:, None], dtype=tf.int32)], axis=1,
        )
 
    return generated
 
 
###############################################################################
# 4. Top-level assembly
###############################################################################
 
def build_select_fn(decode_config: dict[str, Any]) -> Callable[[tf.Tensor], tf.Tensor]:
    """Maps decode_config["strategy"] to the corresponding selection
    function, partially applying its parameter (k or p) where needed."""
    strategy = decode_config.get("strategy", "greedy")
 
    if strategy == "greedy":
        return select_next_token_greedy
    elif strategy == "top_k":
        return partial(select_next_token_top_k, k=int(decode_config.get("k", 5)))
    elif strategy == "nucleus":
        return partial(select_next_token_nucleus, p=float(decode_config.get("p", 0.9)))
    else:
        raise ValueError(
            f"Unknown strategy {strategy!r} for batched generation. "
            "Use 'greedy', 'top_k', or 'nucleus' -- 'beam_search' is not "
            "batchable and needs beam_search_decode directly, below."
        )
 
def generate_batch(
    model,
    batch_messages: list[list[dict[str, str]]],
    *,
    text_to_indices: Callable[[str], list[int]],
    indices_to_text: Callable[[list[int]], str],
    role_token_ids: dict[str, int],
    eos_id: int,
    asst_id: int,
    pad_id: int,
    stop_ids: set[int],
    decode_config: dict[str, Any],
) -> list[str]:
    """The single entry point for greedy/top_k/nucleus generation, any
    batch size from 1 to N. For beam_search, call beam_search_decode
    directly instead (see section 5)."""
    select_fn  = build_select_fn(decode_config)
    max_length = int(decode_config.get("max_length", 20))
 
    input_ids_batch = [
        messages_to_ids(
            msgs, text_to_indices=text_to_indices, role_token_ids=role_token_ids,
            eos_id=eos_id, asst_id=asst_id,
        )
        for msgs in batch_messages
    ]
 
    generated_ids = decode_loop(model, input_ids_batch, select_fn, pad_id, stop_ids, max_length)
 
    return [indices_to_text(ids) for ids in generated_ids]
 
 
###############################################################################
# 5. Beam search -- kept structurally separate; doesn't fit the "pick one
# token from one distribution" shape the functions above share. Not
# batchable (each beam is its own forward pass). Unchanged in behavior
# from the original -- it already used log_softmax correctly and never
# had the naive-log bug that top_k_decode had.
###############################################################################
 
def beam_search_decode(
    model,
    tokenizer,
    input_ids: list[int],
    stop_token_ids: set[int] | None = None,
    beam_width: int = 3,
    max_length: int = 64,
) -> str:
    stop_ids   = set(stop_token_ids) if stop_token_ids is not None else set()
    prompt_len = len(input_ids)
 
    beams = [(list(input_ids), 0.0)]
 
    for _ in range(max_length):
        candidates = []
 
        for seq, score in beams:
            logits    = model(tf.constant([seq], dtype=tf.int32))
            log_probs = tf.nn.log_softmax(logits[:, -1, :])[0]
 
            top_log_probs, top_ids = tf.math.top_k(log_probs, k=beam_width)
 
            for i in range(beam_width):
                candidates.append((
                    seq + [int(top_ids[i])],
                    score + float(top_log_probs[i]),
                ))
 
        beams = sorted(candidates, key=lambda x: x[1], reverse=True)[:beam_width]
 
        if stop_ids and all(seq[-1] in stop_ids for seq, _ in beams):
            break
 
    best_seq      = beams[0][0]
    generated_ids = best_seq[prompt_len:]
 
    if generated_ids and stop_ids and generated_ids[-1] in stop_ids:
        generated_ids = generated_ids[:-1]
 
    return tokenizer.indices_to_text(generated_ids)

