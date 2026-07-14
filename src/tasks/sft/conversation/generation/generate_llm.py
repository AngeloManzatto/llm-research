"""
Created on Tue Jul 14 07:21:23 2026

@author: Angelo Antonio Manzatto
"""

"""
LLM-based Stage 0 data generator.

Run on your own machine — this calls api.anthropic.com and needs
ANTHROPIC_API_KEY set in the environment. Requires: pip install anthropic

Usage:
    python generate_llm.py --out-dir data/sft/conversation/level0/llm_v1

What this does, and does NOT do, deliberately:
  - Calls the standard Messages API in a loop (not the Batches API).
    Batches would be cheaper at real scale but needs polling/result-
    retrieval infrastructure this doesn't build yet — add it later if
    cost actually becomes a problem, not speculatively now.
  - Validates every row with the exact same structural rules as
    data_loader.py (validate.py mirrors them) — a row accepted here is
    guaranteed to pass the training pipeline's loader too.
  - Dedupes both within this run AND against any prior dataset you pass
    via --exclude-dir, using the same exclusion-set approach that caught
    the 51% overlap problem between the two earlier template-generated
    datasets. Oversamples per request and filters, same pattern.
  - Does NOT implement the "Self Review" LLM-judge step from the
    generation spec (naturalness, no hallucinated facts, etc.) — that's
    a real, separate piece of work (a second model call acting as
    reviewer) intentionally left out of this first version. What's
    here is the deterministic structural gate only; semantic review is
    a follow-up, not assumed to already exist.
"""

###############################################################################
# Libraries
###############################################################################


import argparse
import glob
import json
import os
import random
import sys
import time
from pathlib import Path

import anthropic

from prompts import SYSTEM_PROMPT, build_category_prompt
from validate import validate_row, content_key

MODEL = os.environ.get("GENERATION_MODEL", "claude-sonnet-5")
MAX_RETRIES_PER_BATCH = 3
REQUEST_BATCH_SIZE = 20          # examples requested per API call
MAX_API_CALLS_PER_TARGET = 15    # hard ceiling so a stuck category can't loop forever

# Dataset Targets, per protocol v2.1
CATEGORY_TARGETS = {
    "turn_taking":           1500,
    "knowledge_completion":  1500,
    "local_context":         1500,
    "correction":            1500,
    "instruction_following": 1000,
    "uncertainty":           500,
}

# Turn Depth Distribution, per protocol v2.1 (message count -> fraction)
TURN_DEPTH = {
    "local_context":         {4: 0.70, 6: 0.20, 8: 0.10},
    "correction":            {4: 0.70, 6: 0.20, 8: 0.10},
    "instruction_following": {2: 0.90, 4: 0.10},
    "uncertainty":           {2: 0.80, 4: 0.20},
    "turn_taking":           {2: 1.0},
    "knowledge_completion":  {2: 1.0},
}


def _extract_json_array(text: str) -> list[dict]:
    """The model is asked to return only a JSON array, but strip markdown
    fences defensively in case one slips through."""
    text = text.strip()
    if text.startswith("```"):
        text = text.split("```", 2)[1]
        if text.startswith("json"):
            text = text[4:]
        text = text.rsplit("```", 1)[0]
    return json.loads(text.strip())


def call_model(client: anthropic.Anthropic, category: str, language: str, n: int, turn_count: int) -> list[dict]:
    prompt = build_category_prompt(category, language, n, turn_count)
    response = client.messages.create(
        model=MODEL,
        max_tokens=4096,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": prompt}],
    )
    text = "".join(block.text for block in response.content if block.type == "text")
    try:
        rows = _extract_json_array(text)
    except json.JSONDecodeError as e:
        print(f"    ! JSON parse failed for {category}/{language}/{turn_count}msg: {e}")
        return []
    return rows if isinstance(rows, list) else []


def load_exclusion_keys(exclude_dir: str | None) -> set[str]:
    if not exclude_dir:
        return set()
    keys = set()
    for fp in glob.glob(os.path.join(exclude_dir, "*.jsonl")):
        with open(fp, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                keys.add(content_key(row["messages"]))
    print(f"Loaded {len(keys):,} exclusion keys from {exclude_dir}")
    return keys


def generate_category_language(
    client: anthropic.Anthropic,
    category: str,
    language: str,
    target_n: int,
    banned_keys: set[str],
) -> list[list[dict]]:
    """Returns a list of accepted `messages` lists (not yet ID'd)."""
    depth_dist = TURN_DEPTH[category]
    per_depth_targets = {
        depth: round(target_n * frac) for depth, frac in depth_dist.items()
    }
    # Rounding can leave a small gap/excess versus target_n — absorb it
    # into the first depth bucket rather than silently drifting off target.
    drift = target_n - sum(per_depth_targets.values())
    first_depth = next(iter(per_depth_targets))
    per_depth_targets[first_depth] += drift

    accepted: list[list[dict]] = []
    seen_local: set[str] = set()

    for turn_count, depth_target in per_depth_targets.items():
        got = 0
        calls = 0
        while got < depth_target and calls < MAX_API_CALLS_PER_TARGET:
            calls += 1
            request_n = min(REQUEST_BATCH_SIZE, max(5, (depth_target - got) + 5))
            try:
                raw_rows = call_model(client, category, language, request_n, turn_count)
            except anthropic.APIError as e:
                print(f"    ! API error ({category}/{language}/{turn_count}msg, call {calls}): {e}")
                time.sleep(2)
                continue

            batch_accepted = 0
            for row in raw_rows:
                messages = row.get("messages")
                if messages is None:
                    continue
                errors = validate_row(category, language, messages)
                if errors:
                    continue
                key = content_key(messages)
                if key in banned_keys or key in seen_local:
                    continue
                seen_local.add(key)
                accepted.append(messages)
                got += 1
                batch_accepted += 1
                if got >= depth_target:
                    break

            print(f"    {category}/{language}/{turn_count}msg call {calls}: "
                  f"+{batch_accepted} accepted ({got}/{depth_target})")

        if got < depth_target:
            print(f"    ! shortfall: {category}/{language}/{turn_count}msg got "
                  f"{got}/{depth_target} after {calls} calls (hit call ceiling)")

    return accepted


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--exclude-dir", default=None,
                         help="Directory of prior *.jsonl datasets to exclude exact-duplicate content against")
    parser.add_argument("--categories", nargs="*", default=list(CATEGORY_TARGETS),
                         help="Subset of categories to generate (default: all)")
    parser.add_argument("--languages", nargs="*", default=["en", "pt"])
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("ERROR: set ANTHROPIC_API_KEY in your environment before running this.")
        sys.exit(1)
    client = anthropic.Anthropic(api_key=api_key)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    banned_keys = load_exclusion_keys(args.exclude_dir)

    total_rows = 0
    total_errors_skipped = 0

    for category in args.categories:
        target_n = CATEGORY_TARGETS[category]
        per_lang_n = target_n // len(args.languages)

        for language in args.languages:
            print(f"\n=== {category} / {language} (target {per_lang_n}) ===")
            messages_list = generate_category_language(
                client, category, language, per_lang_n, banned_keys,
            )

            rows = []
            for i, messages in enumerate(messages_list, start=1):
                rows.append({
                    "id": f"{category}_{language}_sft_{i:05d}",
                    "category": category,
                    "language": language,
                    "stage": "stage0",
                    "messages": messages,
                })

            out_path = out_dir / f"{category}_{language}.jsonl"
            with out_path.open("w", encoding="utf-8") as f:
                for row in rows:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")

            total_rows += len(rows)
            print(f"  -> wrote {len(rows)} rows to {out_path}")

    print(f"\n{'='*70}")
    print(f"TOTAL ROWS WRITTEN: {total_rows}")
    print("NOTE: this is structural validation only. Semantic review "
          "(naturalness, hallucinated-fact check, category-skill match — "
          "generation spec Section 10) is not implemented here; run a "
          "sample audit before treating this as final.")


if __name__ == "__main__":
    main()