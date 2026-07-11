"""
Created on Sat Jul 11 09:40:07 2026

@author: Angelo Antonio Manzatto
"""
###############################################################################
# Libraries
###############################################################################

import json
import time
from pathlib import Path
 
import numpy as np
import tensorflow as tf
 
from src.core.model.serialization import model_all_finite

from src.tasks.sft.conversation.benchmark.benchmark import Benchmark
from src.tasks.sft.conversation.benchmark.evaluator import evaluate_example
from src.tasks.sft.conversation.benchmark.generator import TextGenerator
from src.tasks.sft.conversation.benchmark.report import EvaluationSummary

from src.tasks.sft.conversation.training.data_packing import (
    make_packed_batch,
    packed_causal_mask,
)
 
###############################################################################
# Loss
###############################################################################

def compute_loss(
    logits,
    labels,
    global_batch_size: int,
    ignore_id: int = -100,
) -> tf.Tensor:
    """
    Cross-entropy over trainable tokens only (where labels != ignore_id),
    scaled correctly for distributed training via compute_average_loss.

    logits            : [B, T, vocab_size]  (any dtype — cast to fp32 internally)
    labels            : [B, T]              (int32, ignore_id where masked)
    global_batch_size : total batch size across all replicas
    """
    # Cast logits to fp32 before cross-entropy — prevents NaN from bf16/fp16
    # numerical instability, matches NTP pipeline's per_token_cross_entropy.
    logits = tf.cast(logits, tf.float32)

    mask        = tf.cast(labels != ignore_id, tf.float32)           # [B, T]
    safe_labels = tf.where(labels == ignore_id,
                           tf.zeros_like(labels), labels)             # [B, T]

    per_token = tf.keras.losses.sparse_categorical_crossentropy(
        safe_labels, logits, from_logits=True
    )                                                                  # [B, T]

    # Sum over trainable tokens within each example, producing [B]
    per_example = tf.reduce_sum(per_token * mask, axis=-1)            # [B]

    # Normalise by the number of trainable tokens in this example so
    # examples with more assistant tokens don't dominate the gradient.
    n_trainable  = tf.reduce_sum(mask, axis=-1)                       # [B]
    per_example  = per_example / (n_trainable + 1e-8)                 # [B]

    # compute_average_loss scales correctly across replicas and handles
    # uneven trainable-token counts between packed sequences.
    return tf.nn.compute_average_loss(
        per_example, global_batch_size=global_batch_size
    )

###############################################################################
# Distributed training step
###############################################################################

def make_train_step(
    model,
    optimizer,
    strategy,
    global_batch_size: int,
    grad_clip_norm: float = 1.0,
    ignore_id: int = -100,
):
    """
    Returns a compiled train_step function bound to model/optimizer/strategy.

    Changes vs previous version:
      - compute_loss now takes global_batch_size and uses compute_average_loss
      - logits are cast to fp32 inside compute_loss (NaN guard)
      - gradient clipping uses tf.clip_by_global_norm (clips the global norm
        across all tensors jointly, not per-tensor — matches NTP pipeline)
      - manual replica scaling removed (handled by compute_average_loss)
    """

    @tf.function
    def _step_fn(input_ids, labels, segment_ids):
        # Build block-diagonal causal mask from segment IDs on each replica
        attn_mask = tf.py_function(
            func=lambda s: tf.concat(
                [packed_causal_mask(s[i].numpy()) for i in range(s.shape[0])],
                axis=0,
            ),
            inp=[segment_ids],
            Tout=tf.float32,
        )
        attn_mask.set_shape([None, 1, None, None])

        with tf.GradientTape() as tape:
            logits = model(input_ids, attn_mask=attn_mask, training=True)
            loss   = compute_loss(logits, labels, global_batch_size, ignore_id)

        grads = tape.gradient(loss, model.trainable_variables)

        # Clip global gradient norm across all parameter tensors jointly.
        # Stronger than per-tensor clipnorm: prevents total gradient magnitude
        # from exploding even when individual tensors each look "safe".
        if grad_clip_norm is not None and grad_clip_norm > 0:
            grads, _ = tf.clip_by_global_norm(grads, grad_clip_norm)

        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        return loss

    @tf.function
    def train_step(input_ids, labels, segment_ids):
        per_replica = strategy.run(_step_fn, args=(input_ids, labels, segment_ids))
        # SUM is correct here: compute_average_loss already divided by
        # global_batch_size, so summing across replicas gives the right scalar.
        return strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica, axis=None)

    return train_step
 
 
###############################################################################
# Benchmark hook
###############################################################################
 
def run_benchmark(
    model,
    tokenizer,
    step: int,
    benchmark_dir: Path,
    batch_size: int = 32,   # number of examples to generate in parallel
) -> dict:
    """
    Run conversation_level0 benchmark with batched generation.
    Generates `batch_size` examples simultaneously — much faster than
    sequential generation on a GPU.
    """
    from src.tasks.sft.conversation.benchmark.benchmark import Benchmark
    from src.tasks.sft.conversation.benchmark.evaluator import evaluate_example
    from src.tasks.sft.conversation.benchmark.generator import TextGenerator
    from src.tasks.sft.conversation.benchmark.report import EvaluationSummary
 
    if not model_all_finite(model):
        print(f"⚠ Skipping benchmark at step {step} — model weights contain NaN/Inf")
        return {"step": step, "skipped": True, "reason": "nan_weights"}
 
    bm  = Benchmark.from_manifest(benchmark_dir / "benchmark.json")
    gen = TextGenerator(model=model, tokenizer=tokenizer,
                        decode_config=bm.default_decode)
 
    run_meta = {"benchmark_id": bm.benchmark_id,
                "benchmark_version": bm.version, "step": step}
    summary  = EvaluationSummary(run_metadata=run_meta)
 
    # Collect all examples then process in batches
    examples = list(bm)
 
    for i in range(0, len(examples), batch_size):
        chunk    = examples[i:i + batch_size]
        messages = [ex.messages for ex in chunk]
 
        # One batched forward pass loop for the whole chunk
        completions = gen.generate_batch(messages)
 
        for example, generated in zip(chunk, completions):
            result = evaluate_example(
                example=example,
                generated=generated,
                decode=bm.default_decode,
                scoring_metric=bm.scoring_metric,
                diagnostic_metrics=bm.diagnostic_metrics,
            )
            summary.update(result)
 
    summary.print_table()
    return summary.to_dict()
 
###############################################################################
# Training logger
###############################################################################
 
class TrainingLogger:
    def __init__(self, log_path: Path):
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
 
    def log(self, record: dict) -> None:
        with self.log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
 
###############################################################################
# Main training loop
###############################################################################
 
def train(
    *,
    model,
    tokenizer,
    optimizer,
    strategy,
    token_dicts: list[dict],
    cfg: dict,
    run_dir: Path,
    benchmark_dir: Path,
    run_baseline_benchmark: bool = True
):
    """
    Full Stage 0 SFT training loop.
 
    Parameters
    ----------
    model, tokenizer, optimizer, strategy : TF objects
    token_dicts : list[dict]
        Pre-tokenised examples from messages_to_tokens().
    cfg : dict
        Training config keys: SEQ_LEN, BATCH_SIZE, EPOCHS, LEARNING_RATE,
        WARMUP_STEPS, CHECKPOINT_EVERY, PAD_ID, IGNORE_ID, SFT_MODEL_ID.
    run_dir : Path
        Root output directory for this run (checkpoints + logs).
    benchmark_dir : Path
        Directory containing benchmark.json.
    """
    SEQ_LEN          = cfg["SEQ_LEN"]
    BATCH_SIZE       = cfg["BATCH_SIZE"]
    EPOCHS           = cfg["EPOCHS"]
    LEARNING_RATE    = cfg["LEARNING_RATE"]
    WARMUP_STEPS     = cfg["WARMUP_STEPS"]
    CHECKPOINT_EVERY = cfg["CHECKPOINT_EVERY"]
    PAD_ID           = cfg["PAD_ID"]
    IGNORE_ID        = cfg["IGNORE_ID"]
    SFT_MODEL_ID     = cfg["SFT_MODEL_ID"]
 
    # --- Steps estimate ---
    _dry          = make_packed_batch(token_dicts, SEQ_LEN, BATCH_SIZE, shuffle=True, seed=0)
    avg_packed    = sum(_dry["n_packed_per_seq"]) / BATCH_SIZE
    steps_per_epoch = max(1, int(len(token_dicts) / (avg_packed * BATCH_SIZE)))
    total_steps     = steps_per_epoch * EPOCHS
 
    print(f"Avg examples packed/seq: {avg_packed:.1f}")
    print(f"Steps per epoch: {steps_per_epoch}  |  Total: {total_steps}")
 
    # --- LR schedule ---
    lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=LEARNING_RATE,
        decay_steps=max(1, total_steps - WARMUP_STEPS),
        alpha=0.1,
    )
 
    def get_lr(step: int) -> float:
        if step < WARMUP_STEPS:
            return LEARNING_RATE * (step + 1) / WARMUP_STEPS
        return float(lr_schedule(step - WARMUP_STEPS))
 
    # --- Compiled train step ---
    train_step = make_train_step(
        model=model,
        optimizer=optimizer,
        strategy=strategy,
        global_batch_size=BATCH_SIZE,   # ← new required argument
        grad_clip_norm=1.0,
        ignore_id=IGNORE_ID,
    )
     
    # --- Checkpointing ---
    ckpt_dir = run_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt     = tf.train.Checkpoint(model=model, optimizer=optimizer)
    ckpt_mgr = tf.train.CheckpointManager(ckpt, str(ckpt_dir), max_to_keep=5)
 
    # --- Logging ---
    logger = TrainingLogger(run_dir / "training_log.jsonl")
 
    # --- State ---
    global_step       = 0
    benchmark_history = []
 
    print("=" * 100)
    print(f"Stage 0 SFT — {SFT_MODEL_ID}")
    print(f"Epochs: {EPOCHS}  |  Batch size: {BATCH_SIZE}  |  Peak LR: {LEARNING_RATE}")
    print("=" * 100)
 
    # Baseline benchmark
    if run_baseline_benchmark:
        print("\nBaseline benchmark (step 0)...")
        bm = run_benchmark(model, tokenizer, step=0, benchmark_dir=benchmark_dir)
        benchmark_history.append(bm)
        logger.log({**bm, "type": "benchmark"})
 
    for epoch in range(1, EPOCHS + 1):
        epoch_start  = time.time()
        epoch_losses = []
 
        # Shuffle + pack for this epoch
        shuffled = [token_dicts[i]
                    for i in np.random.default_rng(epoch).permutation(len(token_dicts))]
   
        # Generatorr:
        def epoch_batches(shuffled):
            cursor = 0
            while cursor < len(shuffled):
                batch = make_packed_batch(shuffled[cursor:], SEQ_LEN, BATCH_SIZE,
                                          shuffle=False, pad_id=PAD_ID, ignore_id=IGNORE_ID)
                consumed = sum(batch["n_packed_per_seq"])
                if consumed == 0:
                    break
                cursor += consumed
                yield batch
        
        print(f"\nEpoch {epoch}/{EPOCHS}")
         
        for batch in epoch_batches(shuffled):
            global_step += 1
            optimizer.learning_rate.assign(get_lr(global_step))
 
            loss_val = float(
                train_step(batch["input_ids"], batch["labels"],
                           batch["segment_ids"]).numpy()
            )
 
            # NaN guard — stop immediately rather than corrupt weights further
            if np.isnan(loss_val) or np.isinf(loss_val):
                print(f"\n⚠ NaN/Inf loss at step {global_step} — stopping.")
                print("Restore from last checkpoint and reduce LEARNING_RATE.")
                return
 
            epoch_losses.append(loss_val)
            logger.log({"type": "step", "epoch": epoch, "step": global_step,
                        "loss": round(loss_val, 6),
                        "lr":   round(get_lr(global_step), 8),
                        "n_packed": sum(batch["n_packed_per_seq"])})
 
            if global_step % 50 == 0:
                print(f"  step {global_step:5d} | loss {loss_val:.4f} "
                      f"| lr {get_lr(global_step):.2e}")
 
            if global_step % CHECKPOINT_EVERY == 0:
                ckpt_path = ckpt_mgr.save()
                print(f"\n  → checkpoint: {ckpt_path}")
                bm = run_benchmark(model, tokenizer, global_step, benchmark_dir)
                benchmark_history.append(bm)
                logger.log({**bm, "type": "benchmark"})
 
        epoch_loss = float(np.mean(epoch_losses))
        elapsed    = time.time() - epoch_start
        print(f"\nEpoch {epoch} — avg loss: {epoch_loss:.4f} | {elapsed:.0f}s")
        logger.log({"type": "epoch", "epoch": epoch,
                    "avg_loss": round(epoch_loss, 6),
                    "elapsed_s": round(elapsed, 1), "step": global_step})
 
    # Final checkpoint + benchmark
    ckpt_mgr.save()
    bm = run_benchmark(model, tokenizer, global_step, benchmark_dir)
    benchmark_history.append(bm)
    logger.log({**bm, "type": "benchmark"})
 
    # Summary
    print("\n" + "=" * 100)
    print(f"TRAINING COMPLETE — {SFT_MODEL_ID} — {global_step} steps")
    print("\nBenchmark progression:")
    for r in benchmark_history:
        print(f"  step {r.get('step',0):5d} | "
              f"pass_rate={r.get('pass_rate',0):.1%}")