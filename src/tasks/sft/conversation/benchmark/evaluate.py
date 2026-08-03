"""
Created on Sun Jul  5 13:39:18 2026

@author: Angelo Antonio Manzatto
"""

###############################################################################
# Libraries
###############################################################################

import json
from datetime import datetime, timezone
from pathlib import Path

import tensorflow as tf

from src.core.loader import load_model_and_tokenizer
from src.core.model.serialization import restore_model_from_checkpoint

from src.tasks.sft.conversation.benchmark.benchmark import load_benchmark
from src.tasks.sft.conversation.benchmark.generator import generate_batch
from src.tasks.sft.conversation.benchmark.special_tokens import resolve_special_tokens
from src.tasks.sft.conversation.benchmark.evaluator import evaluate_example
from src.tasks.sft.conversation.benchmark.report import (
    summarize_results, print_summary, write_result, write_summary,
)

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
# Restore Checkpoint
###############################################################################

checkpoint_path = restore_model_from_checkpoint(
    model,
    Path("runs") / "ntp" / base_model_id / "checkpoints",
)

###############################################################################
# Benchmark
###############################################################################

benchmark_path = Path("benchmarks") / "conversation" / "level0" / "benchmark.json"
benchmark      = load_benchmark(benchmark_path)

# resolve once -- everything downstream (generate_batch) takes plain ints,
# not the tokenizer object itself
resolved = resolve_special_tokens(tokenizer)

run_metadata = {
    "benchmark_id":    benchmark.benchmark_id,
    "model_id":        base_model_id,
    "checkpoint_path": str(checkpoint_path),
    "timestamp_utc":   datetime.now(timezone.utc).isoformat(),
    "decode":          benchmark.default_decode,
}

# benchmark_path is a FILE (benchmark.json), not a directory --
# .parent is required here, appending directly under a file was a
# latent bug in the original script.
result_path = (
    benchmark_path.parent / "results" / base_model_id
    / run_metadata["timestamp_utc"].replace(":", "-").replace("+00:00", "Z")
)
result_path.mkdir(parents=True, exist_ok=True)

###############################################################################
# Run
###############################################################################

BATCH_SIZE = 32
all_results = []

examples = benchmark.examples
for i in range(0, len(examples), BATCH_SIZE):
    batch = examples[i : i + BATCH_SIZE]

    raw_answers = generate_batch(
        model,
        [ex.messages for ex in batch],
        text_to_indices=tokenizer.text_to_indices,
        indices_to_text=tokenizer.indices_to_text,
        decode_config=benchmark.default_decode,
        **resolved,
    )

    for example, raw_answer in zip(batch, raw_answers):
        result = evaluate_example(example, benchmark, raw_answer)
        all_results.append(result)
        write_result(result_path, result)
        print(f"[{result.id}] passed={result.passed} | answer={result.answer!r}")

    print(f"-- batch {i // BATCH_SIZE + 1}/{-(-len(examples) // BATCH_SIZE)} done "
          f"({len(all_results)}/{len(examples)} examples) --")

###############################################################################
# Summary
###############################################################################

summary = summarize_results(all_results, run_metadata=run_metadata)
write_summary(result_path, summary)

print("=" * 80)
print_summary(summary)
print("=" * 80)
print(json.dumps(summary, indent=3, ensure_ascii=False))