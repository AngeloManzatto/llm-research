"""
Created on Tue Dec 23 15:55:41 2025

@author: Angelo Antonio Manzatto
"""

###############################################################################
# Libraries
###############################################################################

import json

from collections import defaultdict

from pathlib import Path

import tensorflow as tf

from src.core.loader import load_model_and_tokenizer
from src.core.model.serialization import restore_model_from_checkpoint
from src.core.model.generation import greedy_decode

###############################################################################
# GPU Strategy
###############################################################################

strategy = tf.distribute.MirroredStrategy()
num_gpus = strategy.num_replicas_in_sync

print(100*"-")
print(f"Number of devices (GPUs): {num_gpus}")

###############################################################################
# Model / Tokenizer
###############################################################################

model_tokenizer_path = Path("configs") / "artifacts" / "base_model_8x8x768x1024_tokenizer_bbpe32k.json"

artifacts = load_model_and_tokenizer(
    model_tokenizer_path,
    strategy,
    build_dummy_forward=True
)

model = artifacts.model
tokenizer = artifacts.tokenizer

# Model parameters
d_model  = artifacts.transformer_cfg.d_model
n_layers = artifacts.transformer_cfg.n_layers
n_heads  = artifacts.transformer_cfg.n_heads
seq_len  = artifacts.transformer_cfg.seq_len

# Tokenizer parameters
vocab_size = artifacts.transformer_cfg.vocab_size
tokenizer_id = Path(artifacts.tokenizer_checkpoint).parent.name

# Model id
base_model_id = f"base_model_{n_layers}x{n_heads}x{d_model}x{seq_len}_{tokenizer_id}_ntp_v1"

print(100*"-")
print(f"Tokenizer: {tokenizer_id}\n")
print(f"Total Number of tokens in vocabulary: {vocab_size:,}")

print(100*"-")
print(f"Model: {base_model_id}\n")
model.summary()

###############################################################################
# Restore Model
###############################################################################

checkpoint_dir = Path("runs") / "ntp" / base_model_id  / "checkpoints"

restore_model_from_checkpoint(model, checkpoint_dir)

###############################################################################
# Pipeline Run
###############################################################################

BENCHMARK_FILE = Path("benchmarks/conversation_level0/data/smoke_test.jsonl")
RESULT_FILE = Path("benchmarks/conversation_level0/results/smoke_base_ntp_greedy64.jsonl")

def extract_completion(full_text: str, prompt: str) -> str:
    if full_text.startswith(prompt):
        return full_text[len(prompt):].strip()
    return full_text.strip()

def score_contains(answer: str, expected_any: list[str]) -> bool:
    answer_lower = answer.lower()
    return any(exp.lower() in answer_lower for exp in expected_any)

def has_wiki_talk_artifact(text: str) -> bool:
    patterns = [
        "(talk)",
        "UTC",
        "unsigned comment",
        "talk page",
        "deletion review",
        "Please do not modify it",
    ]
    t = text.lower()
    return any(p.lower() in t for p in patterns)


def has_repetition(text: str) -> bool:
    chunks = [c.strip() for c in text.split(".") if c.strip()]
    return len(chunks) != len(set(chunks))

def main():
    
    summary = {
        "total": 0,
        "passed": 0,
        "wiki_talk_artifacts": 0,
        "repetition": 0,
        "decode": {
            "method": "greedy",
            "max_length": 64,
        },
        "by_category": defaultdict(lambda: {
            "total": 0,
            "passed": 0,
            "wiki_talk_artifacts": 0,
            "repetition": 0,
        }),
    }
    
    RESULT_FILE.parent.mkdir(parents=True, exist_ok=True)

    with BENCHMARK_FILE.open("r", encoding="utf-8") as f_in, \
         RESULT_FILE.open("w", encoding="utf-8") as f_out:

        for line in f_in:
            ex = json.loads(line)

            full_text = greedy_decode(
                model=model,
                prompt=ex["prompt"],
                tokenizer=tokenizer,
                max_length=64,
            )

            answer = extract_completion(full_text, ex["prompt"])
            passed = score_contains(answer, ex["expected_any"])
            
            wiki_talk_artifact = has_wiki_talk_artifact(answer)
            repetition = has_repetition(answer)
            result = {
                "id": ex["id"],
                "category": ex["category"],
                "language": ex["language"],
                "prompt": ex["prompt"],
                "expected_any": ex["expected_any"],
                "full_text": full_text,
                "answer": answer,
                "passed": passed,
                "wiki_talk_artifact": wiki_talk_artifact,
                "repetition": repetition,
                "decode": {
                    "method": "greedy",
                    "max_length": 64,
                }
            }
            
            summary["total"] += 1
            summary["passed"] += int(passed)
            summary["wiki_talk_artifacts"] += int(wiki_talk_artifact)
            summary["repetition"] += int(repetition)
            
            cat = ex["category"]
            summary["by_category"][cat]["total"] += 1
            summary["by_category"][cat]["passed"] += int(passed)
            summary["by_category"][cat]["wiki_talk_artifacts"] += int(wiki_talk_artifact)
            summary["by_category"][cat]["repetition"] += int(repetition)

            f_out.write(json.dumps(result, ensure_ascii=False) + "\n")

            print("-" * 80)
            print("ID:", ex["id"])
            print("ANSWER:", answer)
            print("PASSED:", passed)
            
        SUMMARY_FILE = RESULT_FILE.with_suffix(".summary.json")

        summary["by_category"] = dict(summary["by_category"])
        
        with SUMMARY_FILE.open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=3, ensure_ascii=False)
        
        print("=" * 80)
        print("SUMMARY")
        print(json.dumps(summary, indent=3, ensure_ascii=False))


if __name__ == "__main__":
    main()