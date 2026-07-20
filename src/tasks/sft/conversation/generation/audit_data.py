"""
Created on Sun Jul 19 09:35:41 2026

@author: Angelo Antonio Manzatto
"""

###############################################################################
# Libraries
###############################################################################

"""
Audit script for the Stage 0 dataset before training on it.


Checks, each independent of the others (one failing doesn't block the rest):
  1. Structural validity (same rules as data_loader.py's validate_example)
  2. Category/language balance vs protocol v2.1's Dataset Targets
  3. Turn-depth distribution vs protocol v2.1's Turn Depth Distribution
  4. Cross-file AND within-file exact-duplicate content (same method that
     caught the 51% overlap between the two earlier template datasets)
  5. Bare-acknowledgment detection on intermediate assistant turns — checks
     whether "restate the fact, don't just acknowledge" was actually
     followed, per file, so batches/providers that ignored the instruction
     are visible rather than averaged away
  6. Per-file assistant-content word count (proxy for trainable tokens,
     since the real BPE tokenizer isn't available here) — flags files/
     categories whose content is suspiciously short
  7. Random sample printer per file, for actual human reading

Nothing here is a semantic/naturalness judge (the Section 10 "Self Review"
step is still not implemented) — this is the deterministic, structural
half of the audit. Read the printed samples yourself; don't only trust
the numbers.
"""

import glob
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

###############################################################################
# Structural validation (mirrors data_loader.py / validate.py exactly)
###############################################################################

VALID_CATEGORIES = {
    "turn_taking", "knowledge_completion", "local_context",
    "correction", "instruction_following", "uncertainty",
}
VALID_LANGUAGES = {"en", "pt"}
VALID_ROLES = {"user", "assistant"}

FORBIDDEN_STRINGS = [
    "<EOS>", "<BOS>", "<PAD>", "<SEQ>", "<MASK>",
] + [f"<SPECIAL-{i}>" for i in range(200)]

CATEGORY_MSG_CONSTRAINTS = {
    "turn_taking":           (2, 2),
    "knowledge_completion":  (2, 2),
    "local_context":         (4, 8),
    "correction":            (4, 8),
    "instruction_following": (2, 6),
    "uncertainty":           (2, 4),
}

# Protocol v2.1 Dataset Targets (per-language; total is 2x this)
CATEGORY_TARGETS_PER_LANG = {
    "turn_taking":           1500,
    "knowledge_completion":  1500,
    "local_context":         1500,
    "correction":            1500,
    "instruction_following": 1000,
    "uncertainty":           500,
}

# Protocol v2.1 Turn Depth Distribution
TURN_DEPTH_TARGETS = {
    "local_context":         {4: 0.70, 6: 0.20, 8: 0.10},
    "correction":            {4: 0.70, 6: 0.20, 8: 0.10},
    "instruction_following": {2: 0.90, 4: 0.10},
    "uncertainty":           {2: 0.80, 4: 0.20},
}

# Bare-acknowledgment phrases the protocol explicitly asks to avoid as an
# intermediate turn's ONLY content (see protocol v2.1 Conversation Style).
BARE_ACK_PATTERNS = [
    r"^got it\.?$", r"^understood\.?$", r"^okay\.?$", r"^ok\.?$",
    r"^entendi\.?$", r"^certo\.?$", r"^combinado\.?$", r"^ta bom\.?$", r"^tá bom\.?$",
]
BARE_ACK_RE = re.compile("|".join(BARE_ACK_PATTERNS), re.IGNORECASE)


def validate_row(category: str, language: str, messages: list[dict]) -> list[str]:
    errors: list[str] = []
    if category not in VALID_CATEGORIES:
        errors.append(f"invalid category: {category!r}")
    if language not in VALID_LANGUAGES:
        errors.append(f"invalid language: {language!r}")
    if not isinstance(messages, list) or len(messages) == 0:
        errors.append("messages must be a non-empty list")
        return errors
    for i, msg in enumerate(messages):
        if not isinstance(msg, dict) or "role" not in msg or "content" not in msg:
            errors.append(f"messages[{i}] malformed")
            continue
        if msg["role"] not in VALID_ROLES:
            errors.append(f"messages[{i}] invalid role: {msg.get('role')!r}")
        if not isinstance(msg["content"], str) or not msg["content"].strip():
            errors.append(f"messages[{i}] content empty/non-string")
    if errors:
        return errors
    roles = [m["role"] for m in messages]
    if roles[0] != "user":
        errors.append(f"first message must be 'user'; got {roles[0]!r}")
    if roles[-1] != "assistant":
        errors.append(f"last message must be 'assistant'; got {roles[-1]!r}")
    for i in range(1, len(roles)):
        if roles[i] == roles[i - 1]:
            errors.append(f"roles must alternate at messages[{i}]")
    for i, msg in enumerate(messages):
        content = msg.get("content", "")
        for forbidden in FORBIDDEN_STRINGS:
            if forbidden in content:
                errors.append(f"messages[{i}] contains forbidden string {forbidden!r}")
                break
    if category in CATEGORY_MSG_CONSTRAINTS:
        lo, hi = CATEGORY_MSG_CONSTRAINTS[category]
        if not (lo <= len(messages) <= hi):
            errors.append(f"{category} requires {lo}-{hi} messages; got {len(messages)}")
    return errors


def content_key(messages: list[dict]) -> str:
    return json.dumps(messages, ensure_ascii=False)


###############################################################################
# Loading
###############################################################################

def load_all_rows(dataset_dir: Path) -> list[tuple[str, dict]]:
    """Returns [(source_file, row_dict), ...] across every *.jsonl found."""
    rows = []
    files = sorted(glob.glob(str(dataset_dir / "**" / "*.jsonl"), recursive=True))
    if not files:
        raise FileNotFoundError(f"No .jsonl files found under {dataset_dir}")
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
    return rows


###############################################################################
# Checks
###############################################################################

def check_structural(rows: list[tuple[str, dict]]) -> dict:
    total = len(rows)
    failed = 0
    failures_by_file = Counter()
    for fp, row in rows:
        errors = validate_row(
            row.get("category", ""), row.get("language", ""), row.get("messages", []),
        )
        if errors:
            failed += 1
            failures_by_file[fp] += 1
    return {"total": total, "failed": failed, "failures_by_file": failures_by_file}


def check_balance(rows: list[tuple[str, dict]]) -> dict:
    counts = Counter((r.get("category"), r.get("language")) for _, r in rows)
    report = {}
    for category, target in CATEGORY_TARGETS_PER_LANG.items():
        for lang in ("en", "pt"):
            actual = counts.get((category, lang), 0)
            report[(category, lang)] = {
                "actual": actual, "target": target,
                "pct_of_target": round(100 * actual / target, 1) if target else None,
            }
    return report


def check_turn_depth(rows: list[tuple[str, dict]]) -> dict:
    report = {}
    for category, dist in TURN_DEPTH_TARGETS.items():
        depths = Counter(
            len(r.get("messages", [])) for _, r in rows if r.get("category") == category
        )
        total = sum(depths.values())
        actual_dist = {d: round(100 * depths.get(d, 0) / total, 1) if total else 0.0 for d in dist}
        report[category] = {
            "total": total,
            "expected_pct": {d: round(100 * f, 1) for d, f in dist.items()},
            "actual_pct": actual_dist,
            "raw_counts": dict(depths),
        }
    return report


def check_duplicates(rows: list[tuple[str, dict]]) -> dict:
    key_to_files = defaultdict(list)
    for fp, row in rows:
        key_to_files[content_key(row.get("messages", []))].append(fp)

    dup_keys = {k: v for k, v in key_to_files.items() if len(v) > 1}
    total_dup_rows = sum(len(v) - 1 for v in dup_keys.values())  # extra copies beyond the first

    # Which file-pairs share the most duplicate content? Useful for spotting
    # e.g. two OpenAI batches that overlapped despite --exclude-dir.
    cross_file_pairs = Counter()
    within_file = Counter()
    for k, files in dup_keys.items():
        file_counts = Counter(files)
        for f, c in file_counts.items():
            if c > 1:
                within_file[f] += c - 1
        unique_files = sorted(set(files))
        if len(unique_files) > 1:
            for i in range(len(unique_files)):
                for j in range(i + 1, len(unique_files)):
                    cross_file_pairs[(unique_files[i], unique_files[j])] += 1

    return {
        "total_rows": len(rows),
        "duplicate_content_groups": len(dup_keys),
        "total_extra_copies": total_dup_rows,
        "within_file_extra_copies": dict(within_file),
        "top_cross_file_pairs": cross_file_pairs.most_common(10),
    }


def check_bare_acknowledgments(rows: list[tuple[str, dict]]) -> dict:
    """Checks intermediate assistant turns (not the final message) in
    multi-turn rows for bare-acknowledgment-only content."""
    by_file = Counter()
    by_file_total = Counter()
    for fp, row in rows:
        messages = row.get("messages", [])
        if len(messages) <= 2:
            continue  # no intermediate turns to check
        intermediate_assistant = [
            m for i, m in enumerate(messages[:-1]) if m.get("role") == "assistant"
        ]
        if not intermediate_assistant:
            continue
        by_file_total[fp] += len(intermediate_assistant)
        for m in intermediate_assistant:
            if BARE_ACK_RE.match(m.get("content", "").strip()):
                by_file[fp] += 1
    report = {}
    for fp in by_file_total:
        bare = by_file.get(fp, 0)
        total = by_file_total[fp]
        report[fp] = {"bare_ack": bare, "intermediate_total": total,
                       "pct": round(100 * bare / total, 1) if total else 0.0}
    return report


def check_content_length(rows: list[tuple[str, dict]]) -> dict:
    """Word-count proxy for trainable content, per category and per file —
    the real tokenizer isn't available here, but word count correlates
    well enough to flag anomalies."""
    by_category = defaultdict(list)
    by_file = defaultdict(list)
    for fp, row in rows:
        category = row.get("category")
        assistant_words = sum(
            len(m.get("content", "").split())
            for m in row.get("messages", []) if m.get("role") == "assistant"
        )
        by_category[category].append(assistant_words)
        by_file[fp].append(assistant_words)

    def summarize(values):
        if not values:
            return {}
        values = sorted(values)
        n = len(values)
        return {
            "mean": round(sum(values) / n, 1),
            "median": values[n // 2],
            "min": values[0],
            "max": values[-1],
        }

    return {
        "by_category": {c: summarize(v) for c, v in by_category.items()},
        "by_file": {f: summarize(v) for f, v in by_file.items()},
    }


def print_random_samples(rows: list[tuple[str, dict]], n_per_file: int, seed: int) -> None:
    rng = random.Random(seed)
    by_file = defaultdict(list)
    for fp, row in rows:
        by_file[fp].append(row)
    print("\n" + "=" * 70)
    print("RANDOM SAMPLES (read these yourself — nothing here checks naturalness)")
    print("=" * 70)
    for fp in sorted(by_file):
        sample = rng.sample(by_file[fp], min(n_per_file, len(by_file[fp])))
        print(f"\n--- {fp} ---")
        for row in sample:
            print(f"  [{row.get('id','?')}] ({row.get('category')}/{row.get('language')})")
            for m in row.get("messages", []):
                print(f"    {m['role']:>9}: {m['content']}")

###############################################################################
# Execut data audition
###############################################################################

def execute_audit_data(dataset_dir):

    dataset_dir = Path(dataset_dir)
    rows = load_all_rows(dataset_dir)
    files = sorted({fp for fp, _ in rows})
    print(f"Loaded {len(rows):,} rows from {len(files)} files under {dataset_dir}\n")

    print("=" * 70)
    print("1. STRUCTURAL VALIDITY")
    print("=" * 70)
    s = check_structural(rows)
    print(f"Total: {s['total']}  Failed: {s['failed']} ({100*s['failed']/s['total']:.2f}%)")
    if s["failures_by_file"]:
        for fp, c in s["failures_by_file"].most_common(10):
            print(f"  {fp}: {c} failures")

    print("\n" + "=" * 70)
    print("2. CATEGORY/LANGUAGE BALANCE vs protocol v2.1 targets")
    print("=" * 70)
    b = check_balance(rows)
    for (cat, lang), info in b.items():
        flag = "  <-- LOW" if info["pct_of_target"] is not None and info["pct_of_target"] < 80 else ""
        print(f"  {cat:22s} {lang}: {info['actual']:6d} / {info['target']:6d} "
              f"({info['pct_of_target']}%){flag}")

    print("\n" + "=" * 70)
    print("3. TURN-DEPTH DISTRIBUTION vs protocol v2.1 targets")
    print("=" * 70)
    td = check_turn_depth(rows)
    for cat, info in td.items():
        print(f"  {cat} (n={info['total']}):")
        print(f"    expected%: {info['expected_pct']}")
        print(f"    actual%:   {info['actual_pct']}")

    print("\n" + "=" * 70)
    print("4. DUPLICATE CONTENT (within-file and cross-file)")
    print("=" * 70)
    d = check_duplicates(rows)
    print(f"Total rows: {d['total_rows']}")
    print(f"Duplicate-content groups: {d['duplicate_content_groups']}")
    print(f"Total extra (redundant) copies: {d['total_extra_copies']} "
          f"({100*d['total_extra_copies']/d['total_rows']:.2f}% of dataset)")
    if d["within_file_extra_copies"]:
        print("Within-file extra copies (top offenders):")
        for fp, c in sorted(d["within_file_extra_copies"].items(), key=lambda x: -x[1])[:10]:
            print(f"    {fp}: {c}")
    if d["top_cross_file_pairs"]:
        print("Top cross-file duplicate-sharing pairs:")
        for (f1, f2), c in d["top_cross_file_pairs"]:
            print(f"    {c:4d} shared  <->  {f1}  |  {f2}")

    print("\n" + "=" * 70)
    print("5. BARE-ACKNOWLEDGMENT CHECK (intermediate assistant turns)")
    print("=" * 70)
    ba = check_bare_acknowledgments(rows)
    for fp, info in sorted(ba.items(), key=lambda x: -x[1]["pct"]):
        flag = "  <-- HIGH" if info["pct"] > 20 else ""
        print(f"  {fp}: {info['bare_ack']}/{info['intermediate_total']} "
              f"({info['pct']}%){flag}")

    print("\n" + "=" * 70)
    print("6. ASSISTANT CONTENT LENGTH (word-count proxy for trainable tokens)")
    print("=" * 70)
    cl = check_content_length(rows)
    print("By category:")
    for cat, stats in cl["by_category"].items():
        print(f"  {cat:22s} {stats}")
    print("By file:")
    for fp, stats in cl["by_file"].items():
        print(f"  {fp}: {stats}")

    print("\n" + "=" * 70)
    print("Audit complete. This covers structure, balance, duplication, and two")
    print("style proxies (bare-ack, content length) — NOT semantic quality or")
    print("naturalness. Read the random samples above before trusting the numbers.")
