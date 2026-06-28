"""
Created on Tue Dec 23 15:55:41 2025

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

from src.benchmarks.core.benchmark import Benchmark
from src.benchmarks.core.evaluator import evaluate_example
from src.benchmarks.core.report import EvaluationSummary, ReportWriter
from src.tasks.sft.conversation.generator import TextGenerator

###############################################################################
# GPU Strategy
###############################################################################

strategy = tf.distribute.MirroredStrategy()
num_gpus = strategy.num_replicas_in_sync

print(100 * "-")
print(f"Number of devices (GPUs): {num_gpus}")


###############################################################################
# Model / Tokenizer
###############################################################################

model_tokenizer_path = (
    Path("configs")
    / "artifacts"
    / "base_model_8x8x768x1024_tokenizer_bbpe32k.json"
)

artifacts = load_model_and_tokenizer(
    model_tokenizer_path,
    strategy,
    build_dummy_forward=True,
)

model = artifacts.model
tokenizer = artifacts.tokenizer

d_model = artifacts.transformer_cfg.d_model
n_layers = artifacts.transformer_cfg.n_layers
n_heads = artifacts.transformer_cfg.n_heads
seq_len = artifacts.transformer_cfg.seq_len

vocab_size = artifacts.transformer_cfg.vocab_size
tokenizer_id = Path(artifacts.tokenizer_checkpoint).parent.name

base_model_id = (
    f"base_model_{n_layers}x{n_heads}x{d_model}x{seq_len}"
    f"_{tokenizer_id}_ntp_v1"
)

print(100 * "-")
print(f"Tokenizer: {tokenizer_id}\n")
print(f"Total Number of tokens in vocabulary: {vocab_size:,}")

print(100 * "-")
print(f"Model: {base_model_id}\n")
model.summary()

###############################################################################
# Restore Model
###############################################################################

checkpoint_dir = Path("runs") / "ntp" / base_model_id / "checkpoints"
checkpoint_path = restore_model_from_checkpoint(model, checkpoint_dir)

###############################################################################
# Benchmark
###############################################################################

BENCHMARK_DIR = Path("benchmarks") / "conversation_level0"
MANIFEST_FILE = BENCHMARK_DIR / "benchmark.json"

benchmark = Benchmark.from_manifest(MANIFEST_FILE)
manifest = benchmark.summary()

###############################################################################
# Run Metadata
###############################################################################

run_metadata = {
    "benchmark_id": manifest["benchmark_id"],
    "benchmark_version": manifest["version"],
    "model_id": base_model_id,
    "checkpoint_path": str(checkpoint_path),
    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    "decode": benchmark.default_decode,
}

###############################################################################
# Text generator
###############################################################################

text_generator = TextGenerator(
    model=model,
    tokenizer=tokenizer,
    decode_config=benchmark.default_decode,
)

###############################################################################
# Result Directory
###############################################################################

safe_timestamp = (
    run_metadata["timestamp_utc"]
    .replace(":", "-")
    .replace("+00:00", "Z")
)

RESULT_DIR = (
    BENCHMARK_DIR
    / "results"
    / base_model_id
    / safe_timestamp
)


###############################################################################
# Pipeline Run
###############################################################################

def main():
    writer = ReportWriter(RESULT_DIR)
    writer.reset()

    summary = EvaluationSummary(run_metadata=run_metadata)

    for example in benchmark:
        
        full_text = text_generator.generate(example.prompt)
        
        result = evaluate_example(
            example=example,
            full_text=full_text,
            decode=benchmark.default_decode,
            scoring_metric=benchmark.scoring_metric,
            diagnostic_metrics=benchmark.diagnostic_metrics,
        )

        summary.update(result)
        writer.write_result(result)

        result_dict = result.to_dict()

        print("-" * 80)
        print("ID:", example.id)
        print("ANSWER:", result_dict["answer"])
        print("PASSED:", result_dict["passed"])

    writer.write_summary(summary)

    print("=" * 80)
    print("SUMMARY")
    print(json.dumps(summary.to_dict(), indent=3, ensure_ascii=False))


if __name__ == "__main__":
    main()