"""
Created on Tue Dec 23 15:55:41 2025

@author: Angelo Antonio Manzatto
"""

###############################################################################
# Libraries
###############################################################################

import time
import json

from pathlib import Path

import tensorflow as tf

from src.core.model.config import TransformerConfig
from src.core.model.transformer import Transformer 
from src.core.model.optimizer import (
    OptimizerConfig,
    build_optimizer
    )
from src.core.model.serialization import ( 
    TransformerCheckpointManager, 
    model_all_finite, 
    optimizer_all_finite
    )

from src.core.tokenizer.tokenizer import BBPETokenizer

from src.tasks.ntp.dataloader import NTPDatasetConfig, build_dataset_from_sources
from src.tasks.ntp.train_utils import NTPTrainStepConfig, build_train_step, build_generation_monitor

from tensorflow.keras import mixed_precision
mixed_precision.set_global_policy("mixed_bfloat16")   

###############################################################################
# GPU Strategy
###############################################################################

strategy = tf.distribute.MirroredStrategy()
num_gpus = strategy.num_replicas_in_sync

print(100*"-")
print(f"Number of devices (GPUs): {num_gpus}")

###############################################################################
# Hyper parameters
###############################################################################

# Model
d_model     = 768
n_layers    = 8
n_heads     = 8
ffn_dim_multiplier = 1.5
multiple_of = 256
seq_len     = 1024
norm_eps    = 1e-5
use_cache   = False

# Tokenizer id
tokenizer_id = "tokenizer_bbpe32k_v1"

# Model id
base_model_id = f"base_model_{n_layers}x{n_heads}x{d_model}x{seq_len}_{tokenizer_id}_ntp_v1"

###############################################################################
# Paths and folders
###############################################################################

def make_run_dirs(*, base_model_id: str) -> dict:
    run_dir = Path("runs") / "ntp" / base_model_id

    ckpt_dir = run_dir / "checkpoints"
    samples_dir = run_dir / "samples"

    ckpt_dir.mkdir(parents=True, exist_ok=True)
    samples_dir.mkdir(parents=True, exist_ok=True)

    return {
        "run_dir": run_dir,
        "ckpt_dir": ckpt_dir,
        "samples_dir": samples_dir,
    }

dirs = make_run_dirs(base_model_id=base_model_id)
run_dir = dirs["run_dir"]
ckpt_dir = dirs["ckpt_dir"]
samples_dir = dirs["samples_dir"]

print(100*"-")
print("Train Directories\n")
print("📁 run_dir:", run_dir)
print("📁 ckpt_dir:", ckpt_dir)

###############################################################################
# Dataset files
###############################################################################

index_files = sorted(
    p for p in Path("data/ntp/packed").rglob("packed_index.jsonl")
    if p.parent.name == tokenizer_id
)

tfrecord_files = []
total_tokens = 0

for idx in index_files:
    with idx.open("r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            tfrecord_files.append(rec["path"])
            total_tokens += rec["tokens"]

print(100*"-")
print("Dataset files\n")
print(f"Total Number of files: {len(tfrecord_files)}")
print(f"Total Number of tokens: {total_tokens:,}")

if not tfrecord_files:
    raise RuntimeError("No TFRecord files found from packed_index.jsonl")
    
###############################################################################
# Load Tokenizer
###############################################################################

tokenizer_checkpoint = Path("src") / "core" / "tokenizer" / tokenizer_id / "bbpe_tokenizer_32000.pkl"

tokenizer = BBPETokenizer.load(Path(tokenizer_checkpoint))

vocab_size  = len(tokenizer.vocab)

print(100*"-")
print("Tokenizer\n")
print(f"Total Number of tokens in vocabulary: {vocab_size:,}")

###############################################################################
# Train Parameters
###############################################################################

global_batch_size = num_gpus * 16
per_replica_batch_size = global_batch_size // num_gpus

assert global_batch_size % num_gpus == 0

tokens_per_step = seq_len * global_batch_size

# One full pass over dataset
total_steps = max(1, total_tokens // tokens_per_step)

# Learning rate schedule
base_lr = 4.2e-4
min_lr  = 0.1 * base_lr
warmup_steps = max(min(int(0.015 * total_steps), 10_000),1)

grad_clip_norm = 1.0
print(100*"-")
print("Training Steps\n")
print(f"global batch size: {global_batch_size}")
print(f"per replica batch size: {per_replica_batch_size}")
print(f"tokens/step: {tokens_per_step:,}")
print(f"total_steps: {total_steps:,}")
print(f"warmup_steps: {warmup_steps:,}")

###############################################################################
# Load Model
###############################################################################

with strategy.scope():
    
    ###############################################################################
    # Load Model
    ###############################################################################
    
    tranformer_cfg = TransformerConfig(
                vocab_size=vocab_size,
                d_model=d_model,
                n_layers=n_layers,
                n_heads=n_heads,
                ffn_dim_multiplier=ffn_dim_multiplier,
                multiple_of=multiple_of,
                seq_len=seq_len,
                norm_eps=norm_eps,
                use_cache=use_cache,
            )
    
    model = Transformer(tranformer_cfg)
    _ = model(tf.zeros((1, seq_len), tf.int32), start_pos=0, training=False)
    
    print(100*"-")
    print("Model\n")
    model.summary()
    
    ###############################################################################
    # Load Optimizer
    ###############################################################################
    
    opt_cfg = OptimizerConfig(
        name="adamw",
        base_lr=base_lr,
        min_lr=min_lr,
        warmup_steps=warmup_steps,
        total_steps=total_steps,
        weight_decay=0.10,
        beta_1=0.9,
        beta_2=0.95,
        epsilon=1e-5,
    )
    
    optimizer = build_optimizer(model, opt_cfg)
    
    ###############################################################################
    # Load Checkpoint
    ###############################################################################
    
    step_var = optimizer.iterations

    ckpt_mgr = TransformerCheckpointManager(
        model=model,
        optimizer=optimizer,
        step_var=step_var,
        checkpoint_dir=ckpt_dir,
        model_config=tranformer_cfg,
        tokenizer_checkpoint=tokenizer_checkpoint,
        base_model_id=base_model_id,
    )
    
    restored_path = ckpt_mgr.restore_latest()
    if restored_path:
        print(f"RESUME MODE: {restored_path} | optimizer.iterations={int(optimizer.iterations.numpy())}")
    else:
        print("NEW RUN: no checkpoint found, starting from scratch")
        
    ###############################################################################
    # Load Dataset
    ###############################################################################
    
    dataset_config = NTPDatasetConfig(
        seq_len=seq_len,
        global_batch_size=global_batch_size,
        vocab_size=vocab_size,
        shuffle=True,
        shuffle_buffer=10_000,
        repeat=True,
        num_parallel_reads=tf.data.AUTOTUNE,
        num_parallel_calls=tf.data.AUTOTUNE,
        deterministic=False,
        assert_in_vocab=True
    )
    
    train_dataset = build_dataset_from_sources(
        tfrecord_files,
        cfg=dataset_config,
        shuffle_files=True,
        )
    
    got = 0
    for x, y in train_dataset.take(1):
        got += 1
    
        tf.debugging.assert_equal(tf.shape(x)[0], global_batch_size)
        tf.debugging.assert_equal(tf.shape(x)[1], seq_len)
        tf.debugging.assert_equal(tf.shape(y)[0], global_batch_size)
        tf.debugging.assert_equal(tf.shape(y)[1], seq_len)
    
        tf.debugging.assert_type(x, tf.int32)
        tf.debugging.assert_type(y, tf.int32)
    
        # y should be x shifted by 1 within each window
        tf.debugging.assert_equal(x[:, 1:], y[:, :-1])
        
        print(100*"-")
        print("Train Dataset\n")
    
        print(
            f"✅ Batch {got}: x={x.shape} y={y.shape} | "
            f"x[min,max]=({int(tf.reduce_min(x))},{int(tf.reduce_max(x))})"
        )
        
        print("input_ids:", tokenizer.indices_to_text(x[0].numpy()))
        print("labels   :", y[0].numpy())
        
    if got == 0:
        raise RuntimeError(
            "Dataset produced 0 batches. "
            "Likely seq_len too large for your records or empty TFRecords."
        )
        
    dist_ds = strategy.experimental_distribute_dataset(train_dataset)
    ds_iter = iter(dist_ds)
    
###############################################################################
# Train Controls
###############################################################################

LOG_EVERY           = 100       # print every N steps
SAVE_EVERY_STEPS    = 1000      # save every N steps (set None to disable)
MONITOR_EVERY_STEPS = 1000      # run generation monitor every N steps (set None to disable)
        
###############################################################################
# Train Step
###############################################################################

train_step = build_train_step(
    model=model,
    optimizer=optimizer,
    strategy=strategy,
    cfg=NTPTrainStepConfig(
        global_batch_size=global_batch_size,
        grad_clip_norm=grad_clip_norm,
    ),
)

###############################################################################
# Train Monitor
###############################################################################

gen_step = build_generation_monitor(
    tokenizer=tokenizer,
    prompts = [
        
    # Everyday / Narrative (EN)
    "The sun dipped below the horizon, painting the sky with hues of orange and purple.",
    "A sudden gust of wind swept through the trees, scattering leaves in every direction.",
    "She hummed a cheerful tune as she walked down the bustling city street.",
    "The old book lay open on the table, its pages filled with faded ink.",
    "He stared out the window, lost in thought, as the rain began to fall.",
    "The aroma of freshly baked bread filled the cozy kitchen.",
    "A playful kitten chased a ball of yarn across the floor.",
    "The distant sound of music drifted through the open window.",
    "She carefully watered the plants, ensuring each one received enough moisture.",
    "He took a deep breath of the crisp, morning air.",
    "The children laughed and played in the park, their voices echoing.",
    "The old clock on the wall ticked slowly, marking the passage of time.",
    "She wrote a letter to her friend, sharing news and memories.",
    "The stars twinkled brightly in the vast, dark sky.",
    "He sipped his coffee, enjoying the warmth and the quiet.",
    "The dog barked excitedly, eager for a walk in the park.",
    "She closed her eyes and imagined herself on a tropical beach.",
    "The fire crackled in the fireplace, providing warmth and comfort.",
    "He smiled, remembering a funny joke he had heard earlier.",

    # Cotidiano / Narrativa (PT)
    "A chuva dançava sobre o telhado.",
    "O gato dormia tranquilamente no sol.",
    "Ela leu um livro fascinante.",
    "O vento soprava forte hoje.",
    "Eles foram ao cinema ontem à noite.",
    "A flor desabrochou lindamente.",
    "O pássaro cantava alegremente na árvore.",
    "Nós comemos pizza deliciosa.",
    "Ele correu no parque pela manhã.",
    "A música tocava suavemente no rádio.",
    "Ela sorriu para ele.",
    "O carro era vermelho e rápido.",
    "Nós planejamos uma viagem emocionante.",
    "Ele gosta de jogar futebol.",
    "A criança brincava com seus brinquedos.",
    "A lua brilhava intensamente no céu.",
    "Nós assistimos a um filme engraçado.",
    "Ele comprou um presente para ela.",
    "A neve caía suavemente."
    ],
    max_length=128,
)

def current_lr(optimizer) -> float:
    lr = optimizer.learning_rate
    if callable(lr):
        return float(lr(optimizer.iterations).numpy())
    try:
        return float(tf.keras.backend.get_value(lr))
    except Exception:
        return float(lr.numpy())

running_loss = 0.0
start_step = int(optimizer.iterations.numpy())
end_step = start_step + total_steps

print(f"Starting training at global step {start_step} → {end_step}")

# Main loop
while int(optimizer.iterations.numpy()) < end_step:
    
    # Clock for each step
    start_time = time.time()
    
    # Update global step
    global_step = int(optimizer.iterations.numpy())
    
    # Fetch batch and train one step
    batch = next(ds_iter)
    
    # Calculate loss
    loss = train_step(batch)
    
    # Guard flags
    model_ok = model_all_finite(model)
    optmizer_ok = optimizer_all_finite(optimizer)
    
    if not model_ok or not optmizer_ok:
        print(f"⚠️ Step failed for model: {model_ok} or optimizer: {optmizer_ok}")
        print("↩️ Restoring latest checkpoint and continuing...")

        latest = ckpt_mgr.latest_checkpoint
        if latest:
            ckpt_mgr.restore(latest)
            running_loss = 0.0
        else:
            print("No checkpoint available to restore. Stopping (nothing to restore).")
            raise

        # rebuild train_step (helps after TF runtime issues)
        train_step = build_train_step(
            model=model,
            optimizer=optimizer,
            strategy=strategy,
            cfg=NTPTrainStepConfig(
                global_batch_size=global_batch_size,
                grad_clip_norm=grad_clip_norm,
            ),
        )
        continue

    # Metrics
    running_loss += float(loss.numpy())

    # Logging
    if (global_step == start_step) or ((global_step + 1) % LOG_EVERY == 0):
        dt = time.time() - start_time
        steps_done = (global_step - start_step + 1)
        avg = running_loss / max(1, steps_done)
        lr_now = current_lr(optimizer)
        print(
            f"step {global_step+1:8d}/{end_step} | "
            f"loss={float(loss):.4f} | avg={avg:.4f} | "
            f"lr={lr_now:.6g} | time={dt:.3f}s"
        )

    # Generation monitor 
    if MONITOR_EVERY_STEPS and ((global_step + 1) % MONITOR_EVERY_STEPS == 0):
        gen_step(model)

    # Periodic checkpoint 
    if SAVE_EVERY_STEPS and ((global_step + 1) % SAVE_EVERY_STEPS == 0):
        ckpt_mgr.save()

# Final save
ckpt_mgr.save()
print("✅ Training complete.")