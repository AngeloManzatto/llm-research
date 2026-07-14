"""
Created on Tue Jul 14 07:21:23 2026

@author: Angelo Antonio Manzatto
"""

"""
Structural validation for LLM-generated Stage 0 rows.

Deliberately identical to data_loader.py's validate_example rules — the
generation method changed (LLM instead of templates), the row-schema
contract did not. Reusing the same rules means a row that passes here
is guaranteed to also pass the training pipeline's own loader, with no
separate schema drifting into existence.
"""
###############################################################################
# Libraries
###############################################################################

import re

VALID_CATEGORIES = {
    "turn_taking", "knowledge_completion", "local_context",
    "correction", "instruction_following", "uncertainty",
}
VALID_LANGUAGES = {"en", "pt"}
VALID_ROLES = {"user", "assistant"}

FORBIDDEN_STRINGS = [
    "<EOS>", "<BOS>", "<PAD>", "<SEQ>", "<MASK>",
] + [f"<SPECIAL-{i}>" for i in range(200)]

CATEGORY_MSG_CONSTRAINTS: dict[str, tuple[int, int]] = {
    "turn_taking":           (2, 2),
    "knowledge_completion":  (2, 2),
    "local_context":         (4, 8),
    "correction":            (4, 8),
    "instruction_following": (2, 6),
    "uncertainty":           (2, 4),
}


def validate_row(category: str, language: str, messages: list[dict]) -> list[str]:
    """Returns a list of error strings; empty list means valid."""
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
            errors.append(f"messages[{i}] must be a dict with 'role' and 'content'")
            continue
        if msg["role"] not in VALID_ROLES:
            errors.append(f"messages[{i}] invalid role: {msg.get('role')!r}")
        if not isinstance(msg["content"], str) or not msg["content"].strip():
            errors.append(f"messages[{i}] content is empty or non-string")

    if errors:
        return errors

    roles = [m["role"] for m in messages]
    if roles[0] != "user":
        errors.append(f"first message must be 'user'; got {roles[0]!r}")
    if roles[-1] != "assistant":
        errors.append(f"last message must be 'assistant'; got {roles[-1]!r}")
    for i in range(1, len(roles)):
        if roles[i] == roles[i - 1]:
            errors.append(f"roles must alternate; messages[{i-1}] and messages[{i}] both {roles[i]!r}")

    for i, msg in enumerate(messages):
        content = msg.get("content", "")
        for forbidden in FORBIDDEN_STRINGS:
            if forbidden in content:
                errors.append(f"messages[{i}] contains forbidden string {forbidden!r}")
                break

    if category in CATEGORY_MSG_CONSTRAINTS:
        min_msgs, max_msgs = CATEGORY_MSG_CONSTRAINTS[category]
        n = len(messages)
        if not (min_msgs <= n <= max_msgs):
            errors.append(f"{category} requires {min_msgs}-{max_msgs} messages; got {n}")

    return errors


def content_key(messages: list[dict]) -> str:
    """Dedup key: exact content match, used both within a run and against
    any previously-accepted dataset passed in as an exclusion set."""
    import json
    return json.dumps(messages, ensure_ascii=False)