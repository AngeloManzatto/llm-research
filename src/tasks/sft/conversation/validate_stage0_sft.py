"""
Created on Thu Jul  2 22:06:36 2026

@author: Angelo Antonio Manzatto
"""

"""
Stage 0 SFT Data Validator
===========================

Validates every row in a Stage 0 SFT JSONL file against the formal
structural contract defined below. Each rule maps directly to a check
in the validation logic. Designed to be run before any training run
to catch malformed data early.

Usage:
    python validate_sft.py <file.jsonl> [file2.jsonl ...]

Exit code 0 if all files pass, 1 if any error is found.

=============================================================================
FORMAL DEFINITIONS — Stage 0 SFT
=============================================================================

UNIVERSAL RULES (every row, every category)
--------------------------------------------
required fields : id, category, language, stage, messages
id format       : {category}_{language}_sft_{NNNNN}  (5-digit zero-padded)
category        : turn_taking | knowledge_completion | local_context |
                  correction | instruction_following | uncertainty
language        : en | pt
stage           : "stage0"
messages        : non-empty list of {"role": ..., "content": ...} dicts
role values     : user | assistant | system  (system reserved for Stage 3+)
alternation     : roles must strictly alternate; no two consecutive same roles
first message   : role must be "user" (system prompts not used until Stage 3+)
last message    : role must be "user" (the assistant turn is what we train on)
content         : non-empty string after stripping whitespace
no special      : content must not contain any literal special token strings
                  (<EOS>, <BOS>, <SPECIAL-N> for any N)

CATEGORY-SPECIFIC STRUCTURAL RULES
------------------------------------

turn_taking
    purpose   : teach the model to produce exactly one response and stop
    messages  : exactly 1 (single user turn, no prior context)
    content   : a simple question, greeting, statement, or short imperative

knowledge_completion
    purpose   : teach the model to recall a universally accepted fact
    messages  : exactly 1 (single user turn)
    content   : a factual question whose answer is deterministic and universal
                (no personal information, no opinion, no multi-step reasoning)

local_context
    purpose   : teach the model to use information stated earlier in the conversation
    messages  : minimum 3 (at least one prior user/assistant exchange + final question)
                maximum 7 (three prior exchanges + final question)
    structure : user, assistant, user [, assistant, user [, assistant, user]]
    constraint: the final user message must ask about something introduced in a
                prior user message (tests recall, not world knowledge)
    prior asst: prior assistant turns must contain non-empty acknowledgment text

correction
    purpose   : teach the model to replace outdated information after a correction
    messages  : minimum 3 (statement turn + acknowledgment + correction turn)
                maximum 7
    structure : user, assistant, user [, assistant, user...]
    constraint: the final user message must introduce a correction or updated fact
                (a prior user message must have stated something that is now corrected)

instruction_following
    purpose   : teach the model to comply with explicit formatting or response constraints
    messages  : minimum 1 (instruction embedded in the single user message)
                maximum 5 (one prior instruction-setting exchange + test question)
    constraint: the final user message, or a prior user message, must contain an
                explicit instruction (e.g. "answer in one word", "list three", "say X")

uncertainty
    purpose   : teach the model to recognise when it cannot answer
    messages  : minimum 1 (unanswerable question in the single user message)
                maximum 3 (one prior exchange providing context about a different
                           entity, then an unanswerable question)
    constraint: the final user message must ask about something that cannot be
                determined from the conversation (personal fact, false presupposition,
                unknowable future, or a different entity than what context describes)

=============================================================================
"""

from __future__ import annotations

import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

###############################################################################
# Constants
###############################################################################

VALID_CATEGORIES = {
    "turn_taking",
    "knowledge_completion",
    "local_context",
    "correction",
    "instruction_following",
    "uncertainty",
}

VALID_LANGUAGES = {"en", "pt"}
VALID_ROLES     = {"user", "assistant", "system"}

# Literal special token strings that must never appear inside content fields.
# These are injected at the index level by the data loader — storing them as
# text would corrupt the tokenization pipeline.
FORBIDDEN_STRINGS = [
    "<EOS>", "<BOS>", "<PAD>", "<SEQ>", "<MASK>",
] + [f"<SPECIAL-{i}>" for i in range(200)]

# Per-category message count constraints (min, max inclusive)
CATEGORY_MSG_CONSTRAINTS: dict[str, tuple[int, int]] = {
    "turn_taking":           (1, 1),
    "knowledge_completion":  (1, 1),
    "local_context":         (3, 7),
    "correction":            (3, 7),
    "instruction_following": (1, 5),
    "uncertainty":           (1, 3),
}

###############################################################################
# Per-example validation
###############################################################################

def validate_example(example: dict[str, Any], line_no: int) -> list[str]:
    errors: list[str] = []
    ex_id = example.get("id", f"<line {line_no}>")

    # ── Required fields ──────────────────────────────────────────────────────
    for field in ("id", "category", "language", "stage", "messages"):
        if field not in example:
            errors.append(f"missing required field: '{field}'")

    if errors:
        return errors  # can't continue without the basics

    category = str(example["category"])
    language = str(example["language"])
    stage    = str(example["stage"])
    messages = example["messages"]

    # ── Scalar field values ──────────────────────────────────────────────────
    if category not in VALID_CATEGORIES:
        errors.append(f"invalid category: {category!r}")

    if language not in VALID_LANGUAGES:
        errors.append(f"invalid language: {language!r}")

    if stage != "stage0":
        errors.append(f"stage must be 'stage0'; got {stage!r}")

    # ── ID format ─────────────────────────────────────────────────────────────
    expected_id_pattern = rf"^{re.escape(category)}_{re.escape(language)}_sft_\d{{5}}$"
    if not re.match(expected_id_pattern, str(example["id"])):
        errors.append(
            f"id must match '{category}_{language}_sft_NNNNN'; got {example['id']!r}"
        )

    # ── Messages list structure ───────────────────────────────────────────────
    if not isinstance(messages, list) or len(messages) == 0:
        errors.append("messages must be a non-empty list")
        return errors

    # Each message must have role and content
    for i, msg in enumerate(messages):
        if not isinstance(msg, dict):
            errors.append(f"messages[{i}] must be a dict")
            continue
        if "role" not in msg:
            errors.append(f"messages[{i}] missing 'role'")
        if "content" not in msg:
            errors.append(f"messages[{i}] missing 'content'")
            continue
        if msg.get("role") not in VALID_ROLES:
            errors.append(f"messages[{i}] invalid role: {msg.get('role')!r}")
        if not isinstance(msg["content"], str) or not msg["content"].strip():
            errors.append(f"messages[{i}] content is empty or non-string")

    if errors:
        return errors  # can't validate structure if messages are malformed

    # ── Role constraints ──────────────────────────────────────────────────────
    roles = [m["role"] for m in messages]

    # System prompts not used until Stage 3+
    if "system" in roles:
        errors.append("system role not permitted in Stage 0 data")

    # First message must be user
    if roles[0] != "user":
        errors.append(f"first message must be 'user'; got {roles[0]!r}")

    # Last message must be user (assistant turn is what the model generates)
    if roles[-1] != "user":
        errors.append(f"last message must be 'user'; got {roles[-1]!r}")

    # Roles must strictly alternate
    for i in range(1, len(roles)):
        if roles[i] == roles[i - 1]:
            errors.append(
                f"roles must alternate; messages[{i-1}] and messages[{i}] "
                f"are both {roles[i]!r}"
            )

    # ── Content must not contain special token strings ────────────────────────
    for i, msg in enumerate(messages):
        content = msg.get("content", "")
        for forbidden in FORBIDDEN_STRINGS:
            if forbidden in content:
                errors.append(
                    f"messages[{i}] content contains forbidden string {forbidden!r} "
                    f"(special tokens must be injected at index level, not stored as text)"
                )
                break  # one error per message is enough

    # ── Category-specific message count ──────────────────────────────────────
    if category in CATEGORY_MSG_CONSTRAINTS:
        min_msgs, max_msgs = CATEGORY_MSG_CONSTRAINTS[category]
        n = len(messages)
        if not (min_msgs <= n <= max_msgs):
            errors.append(
                f"{category} requires {min_msgs}–{max_msgs} messages; got {n}"
            )

    return errors

###############################################################################
# File validation
###############################################################################

def validate_file(path: Path) -> tuple[int, int]:
    """
    Validate a single JSONL file.
    Returns (total, failed) counts.
    """
    path = Path(path)
    if not path.exists():
        print(f"ERROR: file not found: {path}")
        return 0, 1

    total  = 0
    failed = 0
    seen_ids: set[str] = set()

    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue

            total += 1

            try:
                example = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"[line {line_no}] INVALID JSON: {e}")
                failed += 1
                continue

            ex_id = str(example.get("id", ""))
            if ex_id and ex_id in seen_ids:
                print(f"[line {line_no}] DUPLICATE ID: {ex_id!r}")
                failed += 1
                continue
            if ex_id:
                seen_ids.add(ex_id)

            errors = validate_example(example, line_no)
            if errors:
                failed += 1
                print(f"\n{'─'*70}")
                print(f"[line {line_no}] id={example.get('id', '?')!r}")
                for err in errors:
                    print(f"  ✗ {err}")

    print(f"\n{'='*70}")
    print(f"File    : {path}")
    print(f"Total   : {total}")
    print(f"Passed  : {total - failed}")
    print(f"Failed  : {failed}")

    return total, failed

###############################################################################
# Entry point
###############################################################################

def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python validate_sft.py <file.jsonl> [file2.jsonl ...]")
        raise SystemExit(1)

    total_all  = 0
    failed_all = 0

    for arg in sys.argv[1:]:
        total, failed = validate_file(Path(arg))
        total_all  += total
        failed_all += failed

    if len(sys.argv) > 2:
        print(f"\n{'='*70}")
        print(f"GRAND TOTAL : {total_all} rows, {failed_all} failed")

    raise SystemExit(0 if failed_all == 0 else 1)


if __name__ == "__main__":
    main()