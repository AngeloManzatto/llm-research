"""
Created on Thu Jul  2 22:21:09 2026

@author: Angelo Antonio Manzatto
"""

###############################################################################
# Libraries
###############################################################################

import json
from pathlib import Path

import numpy as np

import tensorflow as tf
from typing import Sequence

from src.core.loader import load_model_and_tokenizer
from src.benchmarks.core.special_tokens import TOKEN_BY_NAME

###############################################################################
# Files and Folders
###############################################################################

dataset_path = Path("data") / "sft" / "conversation" / "level0" / "raw"

###############################################################################
# GPU Strategy
###############################################################################

strategy = tf.distribute.MirroredStrategy()
print("-" * 100)
print(f"Number of devices (GPUs): {strategy.num_replicas_in_sync}")

###############################################################################
# Model / Tokenizer
###############################################################################

artifacts = load_model_and_tokenizer(
    Path("configs") / "artifacts" / "base_model_8x8x768x1024_tokenizer_bbpe32k.json",
    strategy,
    build_dummy_forward=True,
)

model     = artifacts.model
tokenizer = artifacts.tokenizer
cfg       = artifacts.transformer_cfg

base_model_id = (
    f"base_model_{cfg.n_layers}x{cfg.n_heads}x{cfg.d_model}x{cfg.seq_len}"
    f"_{Path(artifacts.tokenizer_checkpoint).parent.name}_ntp_v1"
)

print("-" * 100)
print(f"Model: {base_model_id}")
model.summary()

###############################################################################
# Special Tokens
###############################################################################

IGNORE_ID = -100
USER_ID = tokenizer.token_to_index[TOKEN_BY_NAME['USER_TURN'].token]
ASST_ID = tokenizer.token_to_index[TOKEN_BY_NAME['ASSISTANT_TURN'].token]
SYS_ID  = tokenizer.token_to_index[TOKEN_BY_NAME['SYSTEM_TURN'].token]
EOS_ID  = tokenizer.token_to_index[TOKEN_BY_NAME['END_OF_TURN'].token]
PAD_ID  = tokenizer.token_to_index['<PAD>']

###############################################################################
# Load dataset
###############################################################################

dataset = []

for dataset_file in  dataset_path.glob("**/*.jsonl"):
    
    with dataset_file.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            
            dataset.append(json.loads(line))
            
        
def messages_to_tokens(
    messages: list[dict],
    seq_len: int,
    pad_side: str = "left",
    truncate_from: str = "left",
):
    """
    Convert a messages list into LM training tensors.

    Token layout for each turn:
        user      : [USER_ID] + text_ids + [EOS_ID]
        assistant : [ASST_ID] + text_ids + [EOS_ID]   ← loss computed here
        system    : [SYS_ID]  + text_ids + [EOS_ID]   (Stage 3+, masked)

    Loss masking:
        IGNORE_ID on all role-marker tokens and on all user/system turn content.
        Loss computed only on assistant content tokens and their closing EOS.
        This teaches the model what to generate, not how to format the prompt.

    Truncation:
        The last (most recent) assistant turn is always preserved in full.
        If the sequence is too long, earlier turns are trimmed from the left.

    Parameters
    ----------
    messages : list[dict]
        Each dict has "role" ("user" | "assistant" | "system") and "content" (str).
        Must follow the Stage 0 contract:
          - starts with "user"
          - ends with "user" (the assistant completion is what we train on)
          - strictly alternating roles
    seq_len : int
        Fixed output length. Sequences are truncated and/or padded to this length.
    pad_side : "left" | "right"
        Where to add PAD tokens. "left" is standard for decoder-only models.
    truncate_from : "left" | "right"
        Which end to remove tokens from when truncating. "left" preserves the
        most recent context (recommended).

    Returns
    -------
    dict with:
        input_ids      : np.int32[seq_len]
        labels         : np.int32[seq_len]   (-1 / IGNORE_ID where loss is masked)
        attention_mask : np.int32[seq_len]   (1 for real tokens, 0 for pad)
        debug          : dict with token counts and mask statistics
    """

    # ------------------------------------------------------------------
    # 1. Build the flat token ID sequence and a parallel loss-mask vector
    # ------------------------------------------------------------------
    ROLE_IDS = {
        "user":      USER_ID,
        "assistant": ASST_ID,
        "system":    SYS_ID,
    }

    full   = []   # token IDs
    trainable = []  # True where loss should be computed (assistant content + EOS)

    for msg in messages:
        role    = msg["role"]
        content = msg["content"].strip()

        role_id  = ROLE_IDS[role]
        text_ids = tokenizer.text_to_indices(content)
        is_asst  = (role == "assistant")

        # Role marker — always masked
        full.append(role_id)
        trainable.append(False)

        # Content tokens — trainable only for assistant turns
        for tid in text_ids:
            full.append(tid)
            trainable.append(is_asst)

        # Closing EOS — trainable only for assistant turns
        # (teaching the model *when* to stop is part of what we train)
        full.append(EOS_ID)
        trainable.append(is_asst)

    # ------------------------------------------------------------------
    # 2. Build shifted labels (next-token prediction targets)
    #    label[i] = full[i+1] if trainable[i+1] else IGNORE_ID
    # ------------------------------------------------------------------
    n = len(full)

    # Shift: label at position i predicts the token at position i+1
    # The last position predicts nothing → IGNORE_ID
    labels_raw = full[1:] + [IGNORE_ID]

    labels = []
    for i in range(n):
        # We want to compute loss at position i only if the *target* token
        # (full[i+1]) is part of an assistant turn
        target_trainable = trainable[i + 1] if i + 1 < n else False
        labels.append(labels_raw[i] if target_trainable else IGNORE_ID)

    full   = np.array(full,   dtype=np.int32)
    labels = np.array(labels, dtype=np.int32)

    # ------------------------------------------------------------------
    # 3. Truncate if longer than seq_len
    # ------------------------------------------------------------------
    if len(full) > seq_len:
        if truncate_from == "left":
            full   = full[-seq_len:]
            labels = labels[-seq_len:]
        else:
            full   = full[:seq_len]
            labels = labels[:seq_len]

    # ------------------------------------------------------------------
    # 4. Pad to seq_len
    # ------------------------------------------------------------------
    pad_len = seq_len - len(full)
    if pad_len > 0:
        pad_ids  = np.full(pad_len, PAD_ID,    dtype=np.int32)
        pad_labs = np.full(pad_len, IGNORE_ID, dtype=np.int32)
        if pad_side == "left":
            full   = np.concatenate([pad_ids,  full],   axis=0)
            labels = np.concatenate([pad_labs, labels], axis=0)
        else:
            full   = np.concatenate([full,   pad_ids],  axis=0)
            labels = np.concatenate([labels, pad_labs], axis=0)

    attention_mask = (full != PAD_ID).astype(np.int32)

    # ------------------------------------------------------------------
    # 5. Debug info
    # ------------------------------------------------------------------
    n_trainable = int((labels != IGNORE_ID).sum())
    n_real      = int(attention_mask.sum())

    return {
        "input_ids":      full,
        "labels":         labels,
        "attention_mask": attention_mask,
        "debug": {
            "n_real_tokens":      n_real,
            "n_trainable_tokens": n_trainable,
            "n_pad_tokens":       seq_len - n_real,
            "trainable_ratio":    round(n_trainable / max(n_real, 1), 3),
        },
    }


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

###############################################################################
# Training configuration
###############################################################################
 
SEQ_LEN        = cfg.seq_len          # 1024
BATCH_SIZE     = 1
EPOCHS         = 4
LEARNING_RATE  = 3e-4
WARMUP_STEPS   = 100
CHECKPOINT_EVERY = 250               # steps between benchmark + checkpoint

SFT_MODEL_ID = base_model_id.replace("_ntp_v1", "_sft_stage0_v1")
 

###############################################################################
# Tokenise full dataset once (done outside epoch loop)
###############################################################################
 
print("-" * 100)
print(f"Tokenising {len(dataset)} examples...")
 
token_dicts = [
    messages_to_tokens(row["messages"], seq_len=SEQ_LEN)
    for row in dataset
]
 
n_trainable_total = sum(
    int((t["labels"] != IGNORE_ID).sum()) for t in token_dicts
)
print(f"Total trainable tokens across dataset: {n_trainable_total:,}")
print(f"Average trainable tokens per example:  {n_trainable_total / len(token_dicts):.1f}")

###############################################################################
# Estimate steps per epoch from packing
###############################################################################
 
# Dry-run one shuffle to estimate packed sequences per epoch
_dry = make_packed_batch(token_dicts, SEQ_LEN, BATCH_SIZE, shuffle=True, seed=0)
# Count how many examples were consumed in one batch
avg_packed_per_seq = sum(_dry["n_packed_per_seq"]) / BATCH_SIZE
steps_per_epoch    = max(1, int(len(token_dicts) / (avg_packed_per_seq * BATCH_SIZE)))
total_steps        = steps_per_epoch * EPOCHS
 
print(f"Avg examples packed per sequence: {avg_packed_per_seq:.1f}")
print(f"Estimated steps per epoch:        {steps_per_epoch}")
print(f"Total steps ({EPOCHS} epochs):       {total_steps}")
 
###############################################################################
# Optimizer with linear warmup + cosine decay
###############################################################################
 
lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
    initial_learning_rate=LEARNING_RATE,
    decay_steps=max(1, total_steps - WARMUP_STEPS),
    alpha=0.1,                        # minimum lr = 10% of peak
)
 
def get_lr(step: int) -> float:
    """Linear warmup then cosine decay."""
    if step < WARMUP_STEPS:
        return LEARNING_RATE * (step + 1) / WARMUP_STEPS
    return float(lr_schedule(step - WARMUP_STEPS))
 
with strategy.scope():
    optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE, clipnorm=1.0)
 

###############################################################################
# Loss function
###############################################################################
 
def compute_loss(logits, labels):
    """
    Cross-entropy loss over trainable tokens only.
 
    logits : [B, T, vocab_size]
    labels : [B, T]  (IGNORE_ID where masked)
    """
    mask = tf.cast(labels != IGNORE_ID, tf.float32)              # [B, T]
 
    # Replace IGNORE_ID with 0 so sparse_categorical_crossentropy doesn't error
    safe_labels = tf.where(labels == IGNORE_ID, tf.zeros_like(labels), labels)
 
    per_token_loss = tf.keras.losses.sparse_categorical_crossentropy(
        safe_labels, logits, from_logits=True
    )                                                             # [B, T]
 
    loss = tf.reduce_sum(per_token_loss * mask) / (tf.reduce_sum(mask) + 1e-8)
 
    return loss

###############################################################################
# Single training step (compiled for speed)
###############################################################################
 
@tf.function
def _step_fn(input_ids, labels, segment_ids):
    # Build the block-diagonal mask on each replica from its segment_ids slice
    # segment_ids: [B//n_replicas, T]
    # We need [B//n_replicas, 1, T, T] — build per-sequence and stack
    attn_mask = tf.py_function(
        func=lambda s: tf.concat(
            [packed_causal_mask(s[i].numpy()) for i in range(s.shape[0])], axis=0
        ),
        inp=[segment_ids],
        Tout=tf.float32,
    )
    attn_mask.set_shape([None, 1, None, None])

    with tf.GradientTape() as tape:
        logits = model(input_ids, attn_mask=attn_mask, training=True)
        loss   = compute_loss(logits, labels)
        scaled_loss = loss / tf.cast(strategy.num_replicas_in_sync, tf.float32)

    grads = tape.gradient(scaled_loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return loss


@tf.function
def train_step(input_ids, labels, segment_ids):
    per_replica_losses = strategy.run(_step_fn, args=(input_ids, labels, segment_ids))
    return strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses, axis=None)

###############################################################################
# Checkpoint setup
###############################################################################
 
checkpoint_dir = Path("runs") / "sft" / SFT_MODEL_ID / "checkpoints"
checkpoint_dir.mkdir(parents=True, exist_ok=True)
 
ckpt    = tf.train.Checkpoint(model=model, optimizer=optimizer)
ckpt_mgr = tf.train.CheckpointManager(ckpt, str(checkpoint_dir), max_to_keep=5)
 
###############################################################################
# Benchmark evaluation hook
###############################################################################
 
def run_benchmark(step: int) -> dict:
    """
    Run conversation_level0 benchmark and return summary dict.
    Imported inline to avoid circular dependency with training setup.
    """
    from src.benchmarks.core.benchmark import Benchmark
    from src.benchmarks.core.evaluator import evaluate_example
    from src.benchmarks.core.generator import TextGenerator
 
    bm_dir   = Path("benchmarks") / "conversation_level0"
    bm       = Benchmark.from_manifest(bm_dir / "benchmark.json")
    gen      = TextGenerator(model=model, tokenizer=tokenizer,
                             decode_config=bm.default_decode)
 
    total = passed = 0
    stop_ok = 0
 
    for example in bm:
        result = evaluate_example(
            example=example,
            generated=gen.generate(example.messages),
            decode=bm.default_decode,
            scoring_metric=bm.scoring_metric,
            diagnostic_metrics=bm.diagnostic_metrics,
        )
        total   += 1
        passed  += int(result.passed)
        stop_ok += int(result.metrics.get("expected_stop_token", False))
 
    summary = {
        "step":                step,
        "benchmark_total":     total,
        "benchmark_passed":    passed,
        "pass_rate":           round(passed / max(total, 1), 4),
        "stop_token_rate":     round(stop_ok / max(total, 1), 4),
    }
    print(f"\n  [benchmark @ step {step}]  "
          f"pass={passed}/{total} ({summary['pass_rate']:.1%})  "
          f"stop_token={stop_ok}/{total} ({summary['stop_token_rate']:.1%})")
    return summary
 
###############################################################################
# Training log
###############################################################################
 
log_path = Path("runs") / "sft" / SFT_MODEL_ID / "training_log.jsonl"
log_path.parent.mkdir(parents=True, exist_ok=True)
 
def log(record: dict):
    with log_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
 
###############################################################################
# Training loop
###############################################################################
 
import time
 
global_step    = 0
benchmark_history = []
 
print("=" * 100)
print(f"Starting Stage 0 SFT — {SFT_MODEL_ID}")
print(f"Epochs: {EPOCHS}  |  Batch size: {BATCH_SIZE}  |  Peak LR: {LEARNING_RATE}")
print("=" * 100)
 
# Baseline benchmark before any training
print("\nRunning baseline benchmark (step 0)...")
bm_result = run_benchmark(step=0)
benchmark_history.append(bm_result)
log({**bm_result, "type": "benchmark"})
 
for epoch in range(1, EPOCHS + 1):
    epoch_start = time.time()
    epoch_losses = []
 
    # Build packed batches for this epoch (new shuffle each epoch)
    batches = []
    cursor  = 0
    shuffled = list(np.random.default_rng(epoch).permutation(len(token_dicts)))
    shuffled_dicts = [token_dicts[i] for i in shuffled]
 
    while cursor < len(shuffled_dicts):
        batch = make_packed_batch(
            shuffled_dicts[cursor:],
            seq_len=SEQ_LEN,
            batch_size=BATCH_SIZE,
            shuffle=False,           # already shuffled above
            pad_id=PAD_ID,
            ignore_id=IGNORE_ID,
        )
        consumed = sum(batch["n_packed_per_seq"])
        if consumed == 0:
            break
        batches.append(batch)
        cursor += consumed
 
    print(f"\nEpoch {epoch}/{EPOCHS} — {len(batches)} steps")
 
    for batch in batches:
        global_step += 1
 
        # Update learning rate
        new_lr = get_lr(global_step)
        optimizer.learning_rate.assign(new_lr)
 
        # Training step
        loss = train_step(batch["input_ids"], batch["labels"], batch["segment_ids"])
 
        loss_val = float(loss.numpy())
        epoch_losses.append(loss_val)
 
        # Per-step log
        step_record = {
            "type":        "step",
            "epoch":       epoch,
            "step":        global_step,
            "loss":        round(loss_val, 6),
            "lr":          round(new_lr, 8),
            "n_packed":    sum(batch["n_packed_per_seq"]),
        }
        log(step_record)
 
        if global_step % 50 == 0:
            print(f"  step {global_step:5d} | loss {loss_val:.4f} | lr {new_lr:.2e}")
 
        # Checkpoint + benchmark
        if global_step % CHECKPOINT_EVERY == 0:
            ckpt_path = ckpt_mgr.save()
            print(f"\n  → checkpoint saved: {ckpt_path}")
            bm_result = run_benchmark(step=global_step)
            benchmark_history.append(bm_result)
            log({**bm_result, "type": "benchmark"})
 
    # End-of-epoch summary
    epoch_loss = float(np.mean(epoch_losses))
    elapsed    = time.time() - epoch_start
    print(f"\nEpoch {epoch} complete — avg loss: {epoch_loss:.4f} | time: {elapsed:.0f}s")
    log({"type": "epoch", "epoch": epoch, "avg_loss": round(epoch_loss, 6),
         "elapsed_s": round(elapsed, 1), "step": global_step})
 
# Final checkpoint + benchmark
ckpt_path = ckpt_mgr.save()
print(f"\nFinal checkpoint: {ckpt_path}")
bm_result = run_benchmark(step=global_step)
benchmark_history.append(bm_result)
log({**bm_result, "type": "benchmark"})
 
###############################################################################
# Training summary
###############################################################################
 
print("\n" + "=" * 100)
print("TRAINING COMPLETE")
print(f"Model: {SFT_MODEL_ID}")
print(f"Total steps: {global_step}")
print("\nBenchmark progression:")
for r in benchmark_history:
    print(f"  step {r['step']:5d} | pass {r['benchmark_passed']:3d}/{r['benchmark_total']} "
          f"({r['pass_rate']:.1%}) | stop_token {r['stop_token_rate']:.1%}")