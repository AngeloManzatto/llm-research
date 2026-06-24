"""
Created on Wed Dec 24 14:43:34 2025

@author: Angelo Antonio Manzatto
"""
###############################################################################
# Libraries
###############################################################################

from __future__ import annotations

from dataclasses import dataclass

import tensorflow as tf

###############################################################################
# Loss
###############################################################################

@dataclass(frozen=True)
class NTPLossConfig:
    """
    NTP loss config.
    - cast_logits_to_fp32: recommended when using bf16/fp16 or when logits come from mixed dtypes
    - reduction_over_time: how to reduce token loss into per-example loss
    """
    cast_logits_to_fp32: bool = True
    reduction_over_time: str = "mean"  # "mean" or "sum"

def per_token_cross_entropy(
    y_true: tf.Tensor,
    logits: tf.Tensor,
    *,
    cast_logits_to_fp32: bool = True,
) -> tf.Tensor:
    """
    Token-level cross entropy.
    Returns: per-token loss of shape [B, T]
    """
    if cast_logits_to_fp32:
        logits = tf.cast(logits, tf.float32)

    # sparse CE expects y_true int tensor with same [B,T] shape
    per_tok = tf.keras.losses.sparse_categorical_crossentropy(
        y_true, logits, from_logits=True
    )  # [B,T]
    return per_tok

def cross_entropy_loss(
    y_true: tf.Tensor,
    logits: tf.Tensor,
    *,
    global_batch_size: int,
    cfg: NTPLossConfig = NTPLossConfig(),
) -> tf.Tensor:
    """
    Computes a scalar loss suitable for distributed training.

    Steps:
      1) CE per token: [B,T]
      2) reduce over T -> per-example loss: [B]
      3) compute_average_loss(per-example, global_batch_size)

    This matches your old pipeline behavior.
    """
    per_tok = per_token_cross_entropy(y_true, 
                                      logits, 
                                      cast_logits_to_fp32=cfg.cast_logits_to_fp32)

    if cfg.reduction_over_time == "mean":
        per_ex = tf.reduce_mean(per_tok, axis=-1)  # [B]
    elif cfg.reduction_over_time == "sum":
        per_ex = tf.reduce_sum(per_tok, axis=-1)   # [B]
    else:
        raise ValueError(f"Unknown reduction_over_time: {cfg.reduction_over_time}")

    # Scales correctly under tf.distribute strategies
    loss = tf.nn.compute_average_loss(per_ex, global_batch_size=global_batch_size)
    return loss