"""
Created on Tue Dec 30 15:02:43 2025

@author: Angelo Antonio Manzatto
"""

###############################################################################
# Libraries
###############################################################################

from __future__ import annotations

import random
from dataclasses import dataclass

import tensorflow as tf

from src.core.model.generation import (
    greedy_decode, 
    top_k_decode, 
    nucleus_decode, 
    beam_search_decode
    )

###############################################################################
# Config
###############################################################################

@dataclass(frozen=True)
class NTPTrainStepConfig:
    global_batch_size: int
    grad_clip_norm: float = 1.0
    
###############################################################################
# Build train step
###############################################################################

def build_train_step(
    model: tf.keras.Model,
    optimizer: tf.keras.optimizers.Optimizer,
    strategy: tf.distribute.Strategy,
    cfg: NTPTrainStepConfig,
):
    """
    Returns a callable:
        loss = train_step(dist_batch)
    where dist_batch is what you get from iter(dist_dataset).
    """

    def _replica_step(x, y):
        # x,y are per-replica tensors [B,T]
        
        with tf.GradientTape() as tape:
            
            logits = model(x, start_pos=0, training=True)  # [B,T,V]
            
            per_tok = tf.keras.losses.sparse_categorical_crossentropy(
                y, tf.cast(logits, tf.float32), from_logits=True
            )
            
            loss = tf.nn.compute_average_loss(
                tf.reduce_mean(per_tok, axis=-1),  # per-example loss (mean over T)
                global_batch_size=cfg.global_batch_size
            )

        # Trainable vars
        grads = tape.gradient(loss, model.trainable_variables)
        
        if cfg.grad_clip_norm is not None and cfg.grad_clip_norm > 0:
            grads, _ = tf.clip_by_global_norm(grads, cfg.grad_clip_norm)

        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        
        return loss

    @tf.function
    def train_step(dist_batch):
        # dist_batch comes from a distributed dataset iterator
        x, y = dist_batch
        per_replica_losses = strategy.run(_replica_step, args=(x, y))
        return strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses, axis=None)

    return train_step

###############################################################################
# Generation Monitor
###############################################################################

def build_generation_monitor(
    tokenizer,
    prompts,
    decoding_strategies=None,   # dict[str, callable(model, prompt) -> str]
    max_length=128,
    prompts_per_eval=1,
    seed=None,
    max_chars=500,
    print_fn=print,
):
    """
    Returns a function `run(model, prompt=None)` that ALWAYS runs generation.
    Scheduling (every N steps) should be handled outside (train loop).
    """
    rng = random.Random(seed)

    if decoding_strategies is None:
        def _greedy(m, p):  return greedy_decode(m, p, tokenizer, max_length=max_length)
        def _topk(m, p):    return top_k_decode(m, p, tokenizer, k=5, max_length=max_length)
        def _nucleus(m, p): return nucleus_decode(m, p, tokenizer, p=0.9, max_length=max_length)
        def _beam(m, p):    return beam_search_decode(m, p, tokenizer, beam_width=3, max_length=max_length)
        decoding_strategies = {
            "Greedy": _greedy,
            "Top-k (k=5)": _topk,
            "Nucleus (p=0.9)": _nucleus,
            "Beam Search (k=3)": _beam,
        }

    def _pick_prompts(n):
        if n >= len(prompts):
            return prompts[:]
        return rng.sample(prompts, n)

    def run(model, prompt=None) -> None:
        chosen_prompts = [prompt] if prompt is not None else _pick_prompts(prompts_per_eval)

        print_fn("\n" + "=" * 100)
        print_fn("** Text Generation Monitor **")
        print_fn("=" * 100)

        for prm in chosen_prompts:
            print_fn(f"\n**Prompt:** {prm}")
            for name, strategy in decoding_strategies.items():
                out = strategy(model, prm) or ""
                out = out[:max_chars]
                print_fn(f"\n**{name} Strategy:**\n{out}...")

        print_fn("\n" + "=" * 100)
        print_fn("** Finished Generation **")
        print_fn("=" * 100 + "\n")

    return run