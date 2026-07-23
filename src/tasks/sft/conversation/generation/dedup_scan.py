"""
Created on Wed Jul 22 22:46:47 2026

@author: Angelo Antonio Manzatto

dedup_scan.py — duplication and near-duplication scanner for the Stage 0
training corpus. Runnable standalone in your own environment; no upload
needed.

Three checks, each answering a different question:

  1. EXACT DUPLICATES — the same full conversation (all messages, byte-
     identical) appears more than once across the corpus. Same method
     audit_dataset.py already used.

  2. ANSWER CONCENTRATION — for a given (category, language, question),
     how many DISTINCT final answers appear across the corpus, and how
     dominant is the most common one? This generalizes the specific
     pattern found by hand ("Qual é a cor do meu carro?" -> "vermelho"
     10 times, "qual o nome do meu cachorro" -> "Rex" 7 times): even
     without exact-duplicate rows, a question can collapse onto one
     "obvious" answer far more often than genuine scenario diversity
     would predict. This is the check that actually matters for
     local_context/correction/uncertainty, where the correct answer is
     supposed to vary per conversation, not converge on one default.

  3. BENCHMARK LEAK (optional, pass --benchmark) — for each benchmark
     row, does any training row ask the same question AND does its
     answer contain the benchmark's ground-truth value? This is the
     train/test contamination check.

Usage:
    python dedup_scan.py --data-dir /path/to/training/data
    python dedup_scan.py --data-dir /path/to/training/data --benchmark benchmark.jsonl
    python dedup_scan.py --data-dir /path/to/training/data --category local_context --min-group-size 3

Writes a human-readable report to stdout, and optionally a full JSON
report via --output.
"""

from __future__ import annotations
 
import glob
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any
 

###############################################################################
# Loading
###############################################################################
 
def load_corpus(data_dir: str, category: str | None = None) -> list[tuple[str, dict]]:
    """Returns [(source_file, row_dict), ...] for every *.jsonl under data_dir."""
    pattern = f"**/{category}_*.jsonl" if category else "**/*.jsonl"
    files = sorted(glob.glob(str(Path(data_dir) / pattern), recursive=True))
    rows = []
    for fp in files:
        with open(fp, encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as e:
                    print(f"  ! JSON parse error in {fp} line {line_no}: {e}")
                    continue
                rows.append((fp, row))
    return rows, files
 
 
def load_benchmark(path: str, category: str | None = None, manifest_path: str | None = None) -> list[dict]:
    """
    Loads benchmark rows. If manifest_path is given, also loads its
    category_shared_context and merges it into each row's expected_any
    (union, matching evaluator.py's _build_context precedence) — required
    for categories like uncertainty, whose ground truth intentionally
    lives in shared context rather than per-row expected_any. Without
    this, a category using shared context shows expected_any=[] for
    every row and the leak check can never fire — that's a broken
    check silently reporting "no leak found", not a real result.
    """
    shared_context: dict[str, dict] = {}
    if manifest_path:
        with open(manifest_path, encoding="utf-8") as f:
            manifest = json.load(f)
        shared_context = manifest.get("category_shared_context", {})
 
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if category is not None and row.get("category") != category:
                continue
 
            shared = shared_context.get(row.get("category"), {})
            shared_expected = shared.get("expected_any", [])
            if shared_expected:
                row_expected = row.get("expected_any", [])
                row["expected_any"] = list(dict.fromkeys(shared_expected + row_expected))
 
            rows.append(row)
    return rows
 
 
###############################################################################
# Helpers
###############################################################################
 
def normalize(text: str) -> str:
    text = text.strip().lower()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[.!?]+$", "", text)
    return text
 
 
def content_key(messages: list[dict]) -> str:
    return json.dumps(messages, ensure_ascii=False)
 
 
def train_final_question(row: dict) -> str | None:
    messages = row.get("messages", [])
    if len(messages) < 2:
        return None
    return normalize(messages[-2]["content"])
 
 
def train_final_answer(row: dict) -> str:
    return normalize(row["messages"][-1]["content"])
 
 
def bench_final_question(row: dict) -> str:
    return normalize(row["messages"][-1]["content"])
 
 
###############################################################################
# Check 1: exact duplicates
###############################################################################
 
def check_exact_duplicates(rows: list[tuple[str, dict]]) -> dict:
    key_to_files = defaultdict(list)
    for fp, row in rows:
        key_to_files[content_key(row.get("messages", []))].append(fp)
 
    dup_groups = {k: v for k, v in key_to_files.items() if len(v) > 1}
    total_extra = sum(len(v) - 1 for v in dup_groups.values())
 
    by_file = Counter()
    for files in dup_groups.values():
        for f in files:
            by_file[f] += 1
 
    return {
        "total_rows": len(rows),
        "duplicate_groups": len(dup_groups),
        "total_extra_copies": total_extra,
        "pct_of_corpus": round(100 * total_extra / len(rows), 2) if rows else 0.0,
        "top_offending_files": by_file.most_common(15),
    }
 
 
###############################################################################
# Check 2: answer concentration per question
###############################################################################
 
def check_answer_concentration(rows: list[tuple[str, dict]], min_group_size: int) -> dict:
    # group by (category, language, normalized question) -> list of (answer, source_file, row_id)
    groups: dict[tuple, list[tuple[str, str, str]]] = defaultdict(list)
    for fp, row in rows:
        q = train_final_question(row)
        if q is None:
            continue
        key = (row.get("category"), row.get("language"), q)
        answer = train_final_answer(row)
        groups[key].append((answer, fp, row.get("id", "?")))
 
    flagged = []
    by_category_stats = defaultdict(lambda: {"groups_checked": 0, "groups_flagged": 0})
 
    for (category, language, question), entries in groups.items():
        by_category_stats[category]["groups_checked"] += 1
        if len(entries) < min_group_size:
            continue
        answer_counts = Counter(a for a, _, _ in entries)
        top_answer, top_count = answer_counts.most_common(1)[0]
        concentration = top_count / len(entries)
        n_distinct = len(answer_counts)
 
        # Flag if a small number of distinct answers dominate a
        # question asked many times — the "Rex"/"vermelho" pattern.
        if concentration >= 0.5 or n_distinct <= 2:
            by_category_stats[category]["groups_flagged"] += 1
            flagged.append({
                "category": category,
                "language": language,
                "question": question,
                "n_occurrences": len(entries),
                "n_distinct_answers": n_distinct,
                "top_answer": top_answer,
                "top_answer_share": round(concentration, 2),
            })
 
    flagged.sort(key=lambda x: -x["n_occurrences"])
    return {"flagged": flagged, "by_category": dict(by_category_stats)}
 
 
###############################################################################
# Check 3: benchmark leak (optional)
###############################################################################
 
def check_benchmark_leak(bench_rows: list[dict], train_rows: list[tuple[str, dict]]) -> dict:
    train_index = defaultdict(list)
    for fp, t in train_rows:
        q = train_final_question(t)
        if q is None:
            continue
        train_index[(t.get("category"), t.get("language"), q)].append((fp, t))
 
    by_category = defaultdict(lambda: {"total": 0, "near": 0, "question_only": 0, "details": []})
 
    for b in bench_rows:
        cat = b["category"]
        by_category[cat]["total"] += 1
        bq = bench_final_question(b)
        matches = train_index.get((b["category"], b["language"], bq), [])
        if not matches:
            continue
        by_category[cat]["question_only"] += 1
 
        expected = [e.lower() for e in b.get("expected_any", [])]
        fact_matches = [(fp, t) for fp, t in matches
                         if any(exp in train_final_answer(t) for exp in expected)]
        if fact_matches:
            by_category[cat]["near"] += 1
            fp, t = fact_matches[0]
            by_category[cat]["details"].append({
                "benchmark_id": b["id"],
                "expected_any": b.get("expected_any"),
                "matched_file": fp,
                "matched_answer": t["messages"][-1]["content"],
            })
 
    return dict(by_category)
 
###############################################################################
# Report
###############################################################################
 
def print_report(exact: dict, concentration: dict, leak: dict | None, top_n: int) -> None:
    print("=" * 70)
    print("1. EXACT DUPLICATES")
    print("=" * 70)
    print(f"Total rows: {exact['total_rows']}")
    print(f"Duplicate-content groups: {exact['duplicate_groups']}")
    print(f"Extra (redundant) copies: {exact['total_extra_copies']} "
          f"({exact['pct_of_corpus']}% of corpus)")
    if exact["top_offending_files"]:
        print("Top files contributing duplicate rows:")
        for fp, c in exact["top_offending_files"]:
            print(f"    {c:5d}  {fp}")
 
    print()
    print("=" * 70)
    print("2. ANSWER CONCENTRATION (same question, answer collapses to one value)")
    print("=" * 70)
    print("Per-category groups checked / flagged:")
    for cat, stats in sorted(concentration["by_category"].items()):
        print(f"  {cat:22s} checked={stats['groups_checked']:5d}  "
              f"flagged={stats['groups_flagged']:5d}")
    print()
    print(f"Top {top_n} most concentrated questions (highest occurrence count):")
    for f in concentration["flagged"][:top_n]:
        print(f"  [{f['category']}/{f['language']}] {f['n_occurrences']}x  "
              f"'{f['question'][:60]}' -> {f['n_distinct_answers']} distinct answer(s), "
              f"top='{f['top_answer']}' ({f['top_answer_share']:.0%} share)")
 
    if leak is not None:
        print()
        print("=" * 70)
        print("3. BENCHMARK LEAK CHECK")
        print("=" * 70)
        for cat, r in sorted(leak.items()):
            total = r["total"]
            if not total:
                continue
            print(f"\n{cat}  (n={total})")
            print(f"  near match (same Q + same fact in training): {r['near']}/{total} "
                  f"({100*r['near']/total:.0f}%)")
            print(f"  question repeats (any fact): {r['question_only']}/{total} "
                  f"({100*r['question_only']/total:.0f}%)")
            for d in r["details"][:5]:
                print(f"    [{d['benchmark_id']}] expected={d['expected_any']} "
                      f"<-> {d['matched_answer']!r}  (from {d['matched_file']})")
 
###############################################################################
# Main
###############################################################################

def main():

    data_dir = Path("data") / "sft" / "conversation" / "level0"

    # Which benchmark to check leakage against. owner picks which provider's
    # benchmark file; both live under the same manifest, which supplies
    # category_shared_context (needed for uncertainty — without it,
    # expected_any is empty for every uncertainty row and the leak check
    # can never fire, silently reporting "no leak" instead of a real result).
    owner, api = "anthropic", "claude_core_0001"   # or: "openai", "openai_core_0001"
    benchmark_dir = Path("benchmarks") / "conversation" / "level0" / "data"
    benchmark = benchmark_dir / owner / f"{api}.jsonl"
    # Adjust this if your manifest lives somewhere else — it's what makes
    # the uncertainty check actually work, not optional.
    benchmark_manifest = Path("benchmarks") / "conversation" / "level0" / "benchmark.json"

    categories = [
        "turn_taking",
        "knowledge_completion",
        "local_context",
        "correction",
        "instruction_following",
        "uncertainty",
    ]
    min_group_size = 3
    top_n = 20

    output_dir = data_dir
    combined_report: dict[str, Any] = {}

    manifest_arg = benchmark_manifest if benchmark_manifest.exists() else None
    if benchmark.exists() and manifest_arg is None:
        print(f"! WARNING: {benchmark_manifest} not found — uncertainty's leak check "
              f"will silently report 0 regardless of the real answer. Fix the path above.\n")

    for category in categories:
        print("\n" + "#" * 70)
        print(f"# CATEGORY: {category}")
        print("#" * 70)

        rows, files = load_corpus(data_dir, category)
        print(f"Loaded {len(rows)} rows from {len(files)} files under {data_dir}\n")

        exact = check_exact_duplicates(rows)
        concentration = check_answer_concentration(rows, min_group_size)

        leak = None
        if benchmark.exists():
            bench_rows = load_benchmark(benchmark, category, manifest_arg)
            leak = check_benchmark_leak(bench_rows, rows)
        else:
            print(f"  ! benchmark file not found at {benchmark}, skipping leak check")

        print_report(exact, concentration, leak, top_n)

        report: dict[str, Any] = {"exact_duplicates": exact, "answer_concentration": concentration}
        if leak is not None:
            report["benchmark_leak"] = leak
        combined_report[category] = report

        out_path = output_dir / f"dup_diagnostic_{category}_{api}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        print(f"\nReport written to {out_path}")

    combined_path = output_dir / f"dup_diagnostic_ALL_{api}.json"
    with open(combined_path, "w", encoding="utf-8") as f:
        json.dump(combined_report, f, indent=2, ensure_ascii=False)
    print(f"\n\nCombined report for all categories written to {combined_path}")


if __name__ == "__main__":
    main()