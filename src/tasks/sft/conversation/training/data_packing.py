"""
Created on Sat Jul 11 09:49:49 2026

@author: Angelo Antonio Manzatto
"""

###############################################################################
# Libraries
###############################################################################

import numpy as np

import tensorflow as tf
from typing import Sequence

###############################################################################
# Pack examples
###############################################################################
 
def pack_examples(
    examples: Sequence[dict],
    seq_len: int,
    pad_id: int = 0,
    ignore_id: int = -100,
) -> dict:
    """
    Greedily pack a list of tokenised examples into a single sequence of
    length seq_len.
 
    Each example dict must have:
        input_ids      : np.int32[seq_len]   (from messages_to_tokens)
        labels         : np.int32[seq_len]   (from messages_to_tokens)
        attention_mask : np.int32[seq_len]   (1 for real tokens, 0 for pad)
 
    The examples are already padded to seq_len by messages_to_tokens, but
    here we use only their real tokens (where attention_mask == 1) and
    concatenate them into a dense packed sequence.
 
    Parameters
    ----------
    examples : list of dicts
        Pre-tokenised examples. Will be packed in the order given; the caller
        should shuffle before packing.
    seq_len : int
        Target packed sequence length (model's seq_len).
    pad_id : int
        Token ID used for padding any remaining space.
    ignore_id : int
        Label value used to mask padding positions.
 
    Returns
    -------
    dict with:
        input_ids      : np.int32[seq_len]
        labels         : np.int32[seq_len]
        attention_mask : np.int32[seq_len]
        segment_ids    : np.int32[seq_len]  ← NEW: which example each token
                                              belongs to (0-indexed). PAD
                                              tokens get segment id -1.
        n_packed       : int  number of examples successfully packed
    """
    packed_ids  = np.full(seq_len, pad_id,    dtype=np.int32)
    packed_labs = np.full(seq_len, ignore_id, dtype=np.int32)
    packed_mask = np.zeros(seq_len,           dtype=np.int32)
    packed_segs = np.full(seq_len, -1,        dtype=np.int32)
 
    cursor    = 0
    n_packed  = 0
 
    for seg_id, ex in enumerate(examples):
        # Extract only the real (non-pad) tokens
        real = ex["attention_mask"].astype(bool)
        ids  = ex["input_ids"][real]
        labs = ex["labels"][real]
        n    = len(ids)
 
        if cursor + n > seq_len:
            break  # no room for this example; stop packing
 
        packed_ids [cursor:cursor + n] = ids
        packed_labs[cursor:cursor + n] = labs
        packed_mask[cursor:cursor + n] = 1
        packed_segs[cursor:cursor + n] = seg_id
 
        cursor   += n
        n_packed += 1
 
    return {
        "input_ids":      packed_ids,
        "labels":         packed_labs,
        "attention_mask": packed_mask,
        "segment_ids":    packed_segs,
        "n_packed":       n_packed,
    }

###############################################################################
# Block-diagonal causal mask
###############################################################################
 
def packed_causal_mask(
    segment_ids: np.ndarray,
    dtype: tf.DType = tf.float32,
) -> tf.Tensor:
    """
    Build a block-diagonal causal attention mask from segment IDs.
 
    A query at position i can attend to key at position j iff:
        1. j <= i                  (causal: no future tokens)
        2. segment_ids[j] == segment_ids[i]   (same example)
        3. segment_ids[j] != -1    (j is not a PAD token)
 
    Positions where attention is forbidden receive -1e9 (additive mask
    convention matching the existing causal_mask() in transformer.py).
 
    Parameters
    ----------
    segment_ids : np.int32[T]
        Per-token segment assignment from pack_examples(). -1 for PAD.
    dtype : tf.DType
        Output dtype (should match model's compute dtype).
 
    Returns
    -------
    tf.Tensor of shape [1, 1, T, T]
        Additive mask broadcastable to [B, H, T, T].
        0.0 where attention is allowed, -1e9 where it is blocked.
    """
    T = len(segment_ids)
    NEG_INF = -1e9
 
    # Broadcast segment_ids to [T, T] for vectorised comparison
    seg_i = segment_ids[:, None]   # [T, 1]  (query positions)
    seg_j = segment_ids[None, :]   # [1, T]  (key positions)
 
    # Position indices for causal check
    pos_i = np.arange(T)[:, None]  # [T, 1]
    pos_j = np.arange(T)[None, :]  # [1, T]
 
    # Allowed = causal AND same segment AND key is not PAD
    allowed = (
        (pos_j <= pos_i) &          # causal
        (seg_j == seg_i) &          # same example
        (seg_j != -1)               # key is real token
    )                               # [T, T] bool
 
    # Convert to additive float mask
    mask_np = np.where(allowed, 0.0, NEG_INF).astype(np.float32)
 
    # Shape: [1, 1, T, T] to broadcast over [B, H, T, T]
    mask = tf.constant(mask_np, dtype=dtype)[None, None, :, :]
 
    return mask

###############################################################################
# Batch packer: turn a list of examples into a batch of packed sequences
###############################################################################
 
def make_packed_batch(
    examples: list[dict],
    seq_len: int,
    batch_size: int,
    pad_id: int = 0,
    ignore_id: int = -100,
    shuffle: bool = True,
    seed: int | None = None,
) -> dict:
    """
    Pack a shuffled list of examples into a batch of packed sequences.
 
    Greedily fills each batch element with as many examples as fit, then
    moves to the next. Returns TF tensors ready for a training step.
 
    Parameters
    ----------
    examples : list[dict]
        All tokenised examples for an epoch (from messages_to_tokens).
    seq_len : int
        Model's context length (e.g. 1024).
    batch_size : int
        Number of packed sequences per batch.
    pad_id, ignore_id : int
        Padding values.
    shuffle : bool
        Whether to shuffle examples before packing. Should be True for
        training, False for deterministic inspection.
    seed : int | None
        Random seed for shuffle reproducibility.
 
    Returns
    -------
    dict with tf.Tensor values, each [batch_size, seq_len]:
        input_ids, labels, attention_mask, segment_ids
    And:
        masks : tf.Tensor [batch_size, 1, seq_len, seq_len]
            Per-sequence block-diagonal causal masks.
        n_packed_per_seq : list[int]
            Number of examples packed into each sequence.
    """
    if shuffle:
        rng = np.random.default_rng(seed)
        indices = rng.permutation(len(examples))
        examples = [examples[i] for i in indices]
 
    batch_input_ids  = []
    batch_labels     = []
    batch_attn_masks = []
    batch_seg_ids    = []
    batch_masks      = []
    n_packed_list    = []
 
    cursor = 0
    for _ in range(batch_size):
        # Greedily pack starting from cursor
        packed = pack_examples(
            examples[cursor:],
            seq_len=seq_len,
            pad_id=pad_id,
            ignore_id=ignore_id,
        )
 
        cursor += packed["n_packed"]
 
        batch_input_ids.append(packed["input_ids"])
        batch_labels.append(packed["labels"])
        batch_attn_masks.append(packed["attention_mask"])
        batch_seg_ids.append(packed["segment_ids"])
        n_packed_list.append(packed["n_packed"])
 
        # Build block-diagonal causal mask for this sequence
        mask = packed_causal_mask(packed["segment_ids"])  # [1,1,T,T]
        batch_masks.append(mask)
 
        if cursor >= len(examples):
            # Ran out of examples — pad remaining batch elements
            empty = {
                "input_ids":      np.full(seq_len, pad_id,    dtype=np.int32),
                "labels":         np.full(seq_len, ignore_id, dtype=np.int32),
                "attention_mask": np.zeros(seq_len,           dtype=np.int32),
                "segment_ids":    np.full(seq_len, -1,        dtype=np.int32),
                "n_packed":       0,
            }
            for _ in range(batch_size - len(batch_input_ids)):
                batch_input_ids.append(empty["input_ids"])
                batch_labels.append(empty["labels"])
                batch_attn_masks.append(empty["attention_mask"])
                batch_seg_ids.append(empty["segment_ids"])
                n_packed_list.append(0)
                batch_masks.append(packed_causal_mask(empty["segment_ids"]))
            break
 
    return {
        "input_ids":       tf.constant(np.stack(batch_input_ids),  dtype=tf.int32),
        "labels":          tf.constant(np.stack(batch_labels),      dtype=tf.int32),
        "attention_mask":  tf.constant(np.stack(batch_attn_masks),  dtype=tf.int32),
        "segment_ids":     tf.constant(np.stack(batch_seg_ids),     dtype=tf.int32),
        "masks":           tf.concat(batch_masks, axis=0),          # [B,1,T,T]
        "n_packed_per_seq": n_packed_list,
    }