"""
Created on Sat Jul 11 09:40:07 2026

@author: Angelo Antonio Manzatto
"""
###############################################################################
# Libraries
###############################################################################

import json
import time
from collections import deque
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
    packed_causal_mask_tf,
)
 
###############################################################################
# Training instability detection
#
# The old guard only checked np.isnan(loss_val) or np.isinf(loss_val).
# That is necessary but not sufficient: a real collapse was traced to loss
# going 0.04 -> 9.27 -> 6.69 across three steps — all finite values, none
# of which ever tripped that check. The model trained straight through the
# spike, got checkpointed in its already-collapsed state, and never
# recovered (degenerated to emitting the stop token immediately on every
# input). This widens detection to catch that shape of failure too.
###############################################################################

class InstabilityDetector:
    """
    Tracks a rolling window of recent losses and flags a step as unstable
    if:
      - loss is nan/inf (the original check), OR
      - loss is finite but far above the recent trailing average (a
        spike) — catches the collapse shape above, which the original
        check structurally cannot see since 9.27 and 6.69 are both
        ordinary finite floats.

    Spike detection only activates once enough history exists
    (`min_history` steps) — early training loss is naturally volatile,
    and comparing against a near-empty window would false-trigger on
    normal warmup behavior.
    """

    def __init__(self, window: int = 50, spike_multiplier: float = 5.0, min_history: int = 10):
        self.recent = deque(maxlen=window)
        self.spike_multiplier = spike_multiplier
        self.min_history = min_history

    def check(self, loss_val: float) -> str | None:
        """Returns a reason string if unstable, else None. Does not record
        the loss — call record() separately once you've decided whether
        this step's result should be kept."""
        if np.isnan(loss_val) or np.isinf(loss_val):
            return "nan_or_inf"
        if len(self.recent) >= self.min_history:
            trailing_mean = float(np.mean(self.recent))
            if trailing_mean > 0 and loss_val > self.spike_multiplier * trailing_mean:
                return "loss_spike"
        return None

    def record(self, loss_val: float) -> None:
        self.recent.append(loss_val)

    def reset(self) -> None:
        self.recent.clear()

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
        # Build block-diagonal causal mask directly from segment IDs using
        # pure TF ops — no tf.py_function. The previous version used
        # tf.py_function to call the numpy-based packed_causal_mask from
        # data_packing.py, which drops out of the traced graph into an
        # eager Python callback. That is NOT reliably supported inside a
        # tf.function step run via strategy.run() under MirroredStrategy
        # with >1 replica — this was the direct cause of the
        # "cond/Placeholder ... EagerPyFunc" crash under real multi-GPU
        # training, and why it never reproduced on a single-device run
        # (one replica means nothing for the eager callback to
        # desynchronize against). packed_causal_mask_tf is verified
        # bit-identical to the original numpy version across packed/
        # single/all-pad/many-tiny-example segment_ids patterns.
        attn_mask = packed_causal_mask_tf(segment_ids)

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
    result_dir: Path = None,
    batch_size: int = 32,
) -> dict:
    """
    Run conversation_level0 benchmark with batched generation.

    Saves per-example results and summary to:
        {result_dir}/step_{step:06d}/results.jsonl
        {result_dir}/step_{step:06d}/summary.json

    If result_dir is None, results are printed but not saved to disk.
    """
    if not model_all_finite(model):
        print(f"⚠ Skipping benchmark at step {step} — model weights contain NaN/Inf")
        return {"step": step, "skipped": True, "reason": "nan_weights"}

    bm  = Benchmark.from_manifest(benchmark_dir / "benchmark.json")
    gen = TextGenerator(model=model, tokenizer=tokenizer,
                        decode_config=bm.default_decode)

    run_meta = {"benchmark_id": bm.benchmark_id,
                "benchmark_version": bm.version, "step": step}
    summary  = EvaluationSummary(run_metadata=run_meta)
    results  = []   # collect for disk write

    examples_all = list(bm)

    for i in range(0, len(examples_all), batch_size):
        examples    = examples_all[i:i + batch_size]
        completions = gen.generate_batch([ex.messages for ex in examples])

        for example, generated in zip(examples, completions):
            result = evaluate_example(
                benchmark=bm,
                example=example,
                generated=generated,
                decode=bm.default_decode,
            )
            summary.update(result)
            results.append(result)

    summary.print_table()

    # Persist to disk if result_dir provided
    if result_dir is not None:
        step_dir = Path(result_dir) / f"step_{step:06d}"
        step_dir.mkdir(parents=True, exist_ok=True)

        # Per-example results
        with (step_dir / "results.jsonl").open("w", encoding="utf-8") as f:
            for r in results:
                f.write(json.dumps(r.to_dict(), ensure_ascii=False) + "\n")

        # Summary
        (step_dir / "summary.json").write_text(
            json.dumps(summary.to_dict(), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        print(f"  → benchmark results saved: {step_dir}")

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
    RESULT_DIR       = run_dir / "benchmark_results"
    
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

    # --- Force optimizer slot-variable creation before any real step ---
    # Adam creates its momentum/velocity slot variables lazily on the
    # FIRST call to apply_gradients. If that first call happens to
    # coincide with tracing ANY conditional op elsewhere in the graph
    # (e.g. a tf.debugging.assert_equal inside the model's attention
    # layer, which compiles to tf.cond — or, previously, the
    # tf.py_function this code used to build the packed causal mask)
    # under MirroredStrategy with >1 replica, variable creation and
    # conditional branching can desynchronize across replicas. That
    # produces exactly the "cond/Placeholder ... not fed" family of
    # errors seen from two different sources now. optimizer.build()
    # forces every slot variable to exist deterministically, outside any
    # conditional trace, before the real (conditional-laden) first step
    # ever runs — so there is no longer a first-call race for any
    # assert anywhere in the model to collide with.
    with strategy.scope():
        optimizer.build(model.trainable_variables)

    # --- Sharding helper ---
    # make_packed_batch produces one plain tensor of BATCH_SIZE packed
    # sequences (global, across all replicas) — NOT a tf.data.Dataset run
    # through strategy.experimental_distribute_dataset(). Handing a plain
    # tensor straight to strategy.run() does NOT shard it: every replica
    # receives the FULL tensor, identically. Confirmed this has a real
    # consequence beyond wasted compute: tf.keras optimizers SUM gradients
    # across replicas by default, under the assumption that each replica
    # computed its gradient from a DISTINCT shard (with loss already
    # divided by the GLOBAL batch size in compute_loss). With every
    # replica processing IDENTICAL unsharded data, summing their identical
    # gradients multiplies the correct gradient by num_replicas_in_sync —
    # equivalent to silently training at num_replicas x the configured
    # learning rate. This shards the packed batch per replica using
    # experimental_distribute_values_from_function, before strategy.run
    # ever sees it, so gradient summation reconstructs the correct total
    # instead of inflating it.
    num_replicas = strategy.num_replicas_in_sync
    assert BATCH_SIZE % num_replicas == 0, (
        f"BATCH_SIZE ({BATCH_SIZE}) must be evenly divisible by "
        f"num_replicas ({num_replicas}) for even per-GPU sharding."
    )
    per_replica_n = BATCH_SIZE // num_replicas

    def _shard_for_replicas(tensor):
        def value_fn(ctx):
            start = ctx.replica_id_in_sync_group * per_replica_n
            return tensor[start:start + per_replica_n]
        return strategy.experimental_distribute_values_from_function(value_fn)

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
    instability        = InstabilityDetector(window=50, spike_multiplier=5.0, min_history=10)
    FINITE_CHECK_EVERY = 25  # periodic weight-finiteness check, independent of loss value
 
    print("=" * 100)
    print(f"Stage 0 SFT — {SFT_MODEL_ID}")
    print(f"Epochs: {EPOCHS}  |  Batch size: {BATCH_SIZE}  |  Peak LR: {LEARNING_RATE}")
    print("=" * 100)
 
    # Baseline benchmark
    if run_baseline_benchmark:
        print("\nBaseline benchmark (step 0)...")
        bm = run_benchmark(model, tokenizer, step=0, benchmark_dir=benchmark_dir, result_dir=RESULT_DIR)
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
                train_step(_shard_for_replicas(batch["input_ids"]),
                           _shard_for_replicas(batch["labels"]),
                           _shard_for_replicas(batch["segment_ids"])).numpy()
            )
 
            # Instability check — widened per the InstabilityDetector docstring:
            # catches both literal nan/inf AND large-but-finite loss spikes,
            # which is what actually caused the real collapse this guard
            # failed to catch previously.
            reason = instability.check(loss_val)

            # Independent second check: weights can go non-finite without
            # that step's scalar loss reading as nan/inf/spike (e.g. a bad
            # value lands in a parameter that doesn't dominate this batch's
            # loss). Checked periodically rather than every step since
            # model_all_finite() walks every variable.
            if reason is None and global_step % FINITE_CHECK_EVERY == 0:
                if not model_all_finite(model):
                    reason = "nonfinite_weights"

            if reason is not None:
                print(f"\n⚠ Training instability at step {global_step} ({reason}); loss={loss_val:.4f}")
                latest = ckpt_mgr.latest_checkpoint
                if latest:
                    print(f"  Restoring from: {latest}")
                    ckpt.restore(latest)
                    # Roll back global_step to where the checkpoint was saved
                    global_step = int(optimizer.iterations.numpy())
                    # Rebuild train_step to clear any corrupted TF graph state
                    train_step = make_train_step(
                        model=model,
                        optimizer=optimizer,
                        strategy=strategy,
                        global_batch_size=BATCH_SIZE,
                        grad_clip_norm=1.0,
                        ignore_id=IGNORE_ID,
                    )
                    # Clear the rolling loss window — it describes the
                    # collapsed run, not the restored checkpoint's state.
                    instability.reset()
                    print(f"  Resumed from step {global_step} — skipping rest of epoch {epoch}")
                    logger.log({"type": "instability_recovery", "epoch": epoch,
                                "reason": reason, "loss_at_detection": round(loss_val, 6),
                                "detected_step": global_step, "restored_from": latest})
                else:
                    print("  No checkpoint to restore from — stopping.")
                    return
                break  # break out of epoch_batches, continue to next epoch

            instability.record(loss_val)
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
                if run_baseline_benchmark:
                    bm = run_benchmark(model, tokenizer, global_step, benchmark_dir, result_dir=RESULT_DIR)
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