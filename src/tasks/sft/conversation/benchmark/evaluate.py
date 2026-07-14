"""
Created on Sun Jul  5 13:39:18 2026

@author: root
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

from src.tasks.sft.conversation.benchmark.benchmark import Benchmark
from src.tasks.sft.conversation.benchmark.evaluator import evaluate_example
from src.tasks.sft.conversation.benchmark.report import EvaluationSummary, ReportWriter
from src.tasks.sft.conversation.benchmark.generator import TextGenerator

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

benchmark_path = Path("benchmarks") / "conversation" / "level0"
benchmark      = Benchmark.from_manifest(benchmark_path / "benchmark.json")

run_metadata = {
    "benchmark_id":      benchmark.benchmark_id,
    "benchmark_version": benchmark.version,
    "model_id":          base_model_id,
    "checkpoint_path":   str(checkpoint_path),
    "timestamp_utc":     datetime.now(timezone.utc).isoformat(),
    "decode":            benchmark.default_decode,
}

text_generator = TextGenerator(
    model=model,
    tokenizer=tokenizer,
    decode_config=benchmark.default_decode,
)

result_path = (
    benchmark_path / "results" / base_model_id
    / run_metadata["timestamp_utc"].replace(":", "-").replace("+00:00", "Z")
)

###############################################################################
# Run
###############################################################################

writer  = ReportWriter(result_path)
summary = EvaluationSummary(run_metadata=run_metadata)

for example in benchmark:
    
    generated = text_generator.generate(example.messages)
    
    result = evaluate_example(
        benchmark=benchmark,
        example=example,
        generated=generated,
        decode=benchmark.default_decode,
    )
    
    summary.update(result)
    writer.write_result(result)
    print(f"[{result.id}] passed={result.passed} | answer={result.answer!r}")

writer.write_summary(summary)

print("=" * 80)
print(json.dumps(summary.to_dict(), indent=3, ensure_ascii=False))