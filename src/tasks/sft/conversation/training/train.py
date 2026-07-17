"""
Created on Fri Jul 10 07:58:22 2026

@author: Angelo Antonio Manzatto
"""

###############################################################################
# Libraries
###############################################################################

import os
from pathlib import Path
import tensorflow as tf
 
from src.core.loader import load_model_and_tokenizer
from src.core.model.serialization import restore_model_from_checkpoint, model_all_finite

from src.tasks.sft.conversation.training.data_loader import load_dataset
from src.tasks.sft.conversation.training.data_tokenizer import (
    resolve_token_ids,
    messages_to_tokens,
)
from src.tasks.sft.conversation.training.train_utils import train

from tensorflow.keras import mixed_precision
mixed_precision.set_global_policy("mixed_bfloat16")

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

###############################################################################
# GPU Strategy
###############################################################################

strategy = tf.distribute.MirroredStrategy()
num_gpus = strategy.num_replicas_in_sync

print(100*"-")
print(f"Number of devices (GPUs): {num_gpus}")
 
###############################################################################
# Training configuration
###############################################################################

global_batch_size = num_gpus * 12
per_replica_batch_size = global_batch_size // num_gpus

assert global_batch_size % num_gpus == 0

BATCH_SIZE       = global_batch_size
EPOCHS           = 4
LEARNING_RATE    = 3e-4      # reduced from 3e-4 after NaN at step 800
WARMUP_STEPS     = 100
CHECKPOINT_EVERY = 250

ARTIFACT_CFG  = Path("configs")    / "artifacts" / "base_model_8x8x768x1024_tokenizer_bbpe32k.json"
DATASET_DIR   = Path("data")       / "sft" / "conversation" / "level0" 
BENCHMARK_DIR = Path("benchmarks") / "conversation" / "level0"

###############################################################################
# GPU strategy
###############################################################################
 
strategy = tf.distribute.MirroredStrategy()
print("-" * 100)
print(f"GPUs: {strategy.num_replicas_in_sync}")
 
###############################################################################
# Model + tokenizer
###############################################################################
 
with strategy.scope():
    artifacts = load_model_and_tokenizer(ARTIFACT_CFG, strategy, build_dummy_forward=True)
    model     = artifacts.model
    tokenizer = artifacts.tokenizer
    cfg       = artifacts.transformer_cfg
    
BASE_MODEL_ID = (
    f"base_model_{cfg.n_layers}x{cfg.n_heads}x{cfg.d_model}x{cfg.seq_len}"
    f"_{Path(artifacts.tokenizer_checkpoint).parent.name}_ntp_v1"
)
SFT_MODEL_ID = BASE_MODEL_ID.replace("_ntp_v1", "_sft_stage0_v1")
 
print(f"Model: {BASE_MODEL_ID}")
model.summary()

###############################################################################
# Optimizer (must be inside strategy.scope())
###############################################################################
 
with strategy.scope():
    optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE, clipnorm=1.0)
    
###############################################################################
# Checkpoint setup
###############################################################################
 
checkpoint_dir = Path("runs") / "ntp" / BASE_MODEL_ID / "checkpoints"

restore_model_from_checkpoint(model, checkpoint_dir)

# Always verify after restoring
assert model_all_finite(model), "Checkpoint is corrupted — choose a different one"
 
###############################################################################
# Special token IDs
###############################################################################
 
token_ids = resolve_token_ids(tokenizer)
SEQ_LEN   = cfg.seq_len     # 1024

###############################################################################
# Dataset
###############################################################################
 
dataset     = load_dataset(DATASET_DIR, validate=True)
token_dicts = [
    messages_to_tokens(row["messages"], SEQ_LEN, tokenizer, token_ids)
    for row in dataset
]

n_trainable = sum(int((t["labels"] != token_ids["IGNORE_ID"]).sum()) for t in token_dicts)
print(f"Tokenised {len(token_dicts):,} examples | {n_trainable:,} trainable tokens")

###############################################################################
# Train
###############################################################################
 
train(
    model=model,
    tokenizer=tokenizer,
    optimizer=optimizer,
    strategy=strategy,
    token_dicts=token_dicts,
    cfg={
        "SEQ_LEN":          SEQ_LEN,
        "BATCH_SIZE":       BATCH_SIZE,
        "EPOCHS":           EPOCHS,
        "LEARNING_RATE":    LEARNING_RATE,
        "WARMUP_STEPS":     WARMUP_STEPS,
        "CHECKPOINT_EVERY": CHECKPOINT_EVERY,
        "PAD_ID":           token_ids["PAD_ID"],
        "IGNORE_ID":        token_ids["IGNORE_ID"],
        "SFT_MODEL_ID":     SFT_MODEL_ID,
    },
    run_dir=Path("runs") / "sft" / SFT_MODEL_ID,
    benchmark_dir=BENCHMARK_DIR,
    run_baseline_benchmark=True
)