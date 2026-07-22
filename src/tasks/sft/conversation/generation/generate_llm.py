"""
Multi-provider LLM-based Stage 0 data generator.

Supports:
  - Anthropic Messages API
  - OpenAI Responses API

Required packages:
    pip install anthropic openai

Environment variables:
    ANTHROPIC_API_KEY
    OPENAI_API_KEY

Optional model overrides:
    ANTHROPIC_GENERATION_MODEL
    OPENAI_GENERATION_MODEL

Recommended execution from the project root:

    python -m src.tasks.sft.conversation.generation.generate_llm_multi \
        --provider openai \
        --batch-id 0002 \
        --out-dir data/sft/conversation/level0/llm_v1

Resume behavior:
    If an output file already exists, its valid rows are preserved and
    generation continues until the configured target is reached.

Overwrite behavior:
    Pass --overwrite to discard an existing output file and regenerate it.
"""

from __future__ import annotations

###############################################################################
# Libraries
###############################################################################

import argparse
import glob
import json
import math
import os
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Protocol

import anthropic
from openai import OpenAI

from src.tasks.sft.conversation.generation.prompts import (
    SYSTEM_PROMPT,
    build_category_prompt,
)
from src.tasks.sft.conversation.generation.validate import (
    content_key,
    validate_row,
)

###############################################################################
# Defaults
###############################################################################

DEFAULT_MODELS = {
    "anthropic": os.environ.get(
        "ANTHROPIC_GENERATION_MODEL",
        "claude-haiku-4-5-20251001",
    ),
    "openai": os.environ.get(
        "OPENAI_GENERATION_MODEL",
        "gpt-5-mini",
    ),
}

# Number of examples requested from the API in each call.
REQUEST_BATCH_SIZE = 20

# Used to estimate how many calls are likely to be necessary.
#
# For example:
#   20 requested examples × 0.70 acceptance = 14 accepted per API call.
EXPECTED_ACCEPTANCE_RATE = 0.70

# Additional room above the estimated number of calls.
CALL_SAFETY_FACTOR = 1.30

# Absolute lower and upper bounds for API calls per depth target.
MIN_API_CALLS_PER_TARGET = 3
MAX_API_CALLS_HARD_LIMIT = 250

# Delay after provider/API errors.
RETRY_SLEEP_SECONDS = 2.0

# Delay between successful API calls. Increase this if rate limits occur.
SUCCESS_SLEEP_SECONDS = 0.0

# Maximum generated tokens per provider request.
DEFAULT_MAX_OUTPUT_TOKENS = 8192

###############################################################################
# Dataset targets
###############################################################################

# Total across all requested languages.
#
# With the default languages ["en", "pt"], these become:
#   turn_taking:           750 per language
#   knowledge_completion:  750 per language
#   local_context:         750 per language
#   correction:            750 per language
#   instruction_following: 500 per language
#   uncertainty:           500 per language
CATEGORY_TARGETS = {
    "turn_taking": 1500,
    "knowledge_completion": 1500,
    "local_context": 1500,
    "correction": 1500,
    "instruction_following": 1000,
    "uncertainty": 1000,
}

# Message-count distribution.
TURN_DEPTH = {
    "local_context": {
        4: 0.70,
        6: 0.20,
        8: 0.10,
    },
    "correction": {
        4: 0.70,
        6: 0.20,
        8: 0.10,
    },
    "instruction_following": {
        2: 0.90,
        4: 0.10,
    },
    "uncertainty": {
        2: 0.80,
        4: 0.20,
    },
    "turn_taking": {
        2: 1.0,
    },
    "knowledge_completion": {
        2: 1.0,
    },
}

###############################################################################
# Provider abstraction
###############################################################################


class GenerationProvider(Protocol):
    """Interface implemented by each supported API provider."""

    name: str
    model: str

    def generate(
        self,
        *,
        prompt: str,
        max_output_tokens: int,
    ) -> str:
        """Return the generated plain-text response."""


class AnthropicProvider:
    """Anthropic Messages API implementation."""

    name = "anthropic"

    def __init__(
        self,
        *,
        api_key: str,
        model: str,
    ) -> None:
        self.model = model
        self.client = anthropic.Anthropic(api_key=api_key)
        # Set after each call — the model the API actually reports having
        # used. Compared against self.model by the caller (see
        # verify_model_match) to catch a mistyped/nonexistent --model
        # string silently resolving to something unintended. This is how
        # the "sonet-4.6" batch — a nonexistent model string with no
        # validation anywhere — went undetected until an audit of the
        # generated content weeks later.
        self.last_response_model: str | None = None

    def generate(
        self,
        *,
        prompt: str,
        max_output_tokens: int,
    ) -> str:
        response = self.client.messages.create(
            model=self.model,
            max_tokens=max_output_tokens,
            system=SYSTEM_PROMPT,
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
        )

        self.last_response_model = getattr(response, "model", None)

        return "".join(
            block.text
            for block in response.content
            if getattr(block, "type", None) == "text"
        )


class OpenAIProvider:
    """OpenAI Responses API implementation."""

    name = "openai"

    def __init__(
        self,
        *,
        api_key: str,
        model: str,
    ) -> None:
        self.model = model
        self.client = OpenAI(api_key=api_key)
        self.last_response_model: str | None = None

    def generate(
        self,
        *,
        prompt: str,
        max_output_tokens: int,
    ) -> str:
        response = self.client.responses.create(
            model=self.model,
            instructions=SYSTEM_PROMPT,
            input=prompt,
            max_output_tokens=max_output_tokens,
        )

        self.last_response_model = getattr(response, "model", None)

        return response.output_text


def build_provider(
    *,
    provider_name: str,
    model: str | None,
) -> GenerationProvider:
    """Construct the selected provider from environment credentials."""

    selected_model = model or DEFAULT_MODELS[provider_name]

    if provider_name == "anthropic":
        api_key = os.environ.get("ANTHROPIC_API_KEY")

        if not api_key:
            raise RuntimeError(
                "ANTHROPIC_API_KEY is not set."
            )

        return AnthropicProvider(
            api_key=api_key,
            model=selected_model,
        )

    if provider_name == "openai":
        api_key = os.environ.get("OPENAI_API_KEY")

        if not api_key:
            raise RuntimeError(
                "OPENAI_API_KEY is not set."
            )

        return OpenAIProvider(
            api_key=api_key,
            model=selected_model,
        )

    raise ValueError(
        f"Unsupported provider: {provider_name!r}"
    )

###############################################################################
# JSON response parsing
###############################################################################


def extract_json_array(
    text: str,
) -> list[dict[str, Any]]:
    """
    Parse the model response as a JSON array.

    The model is instructed to return only JSON, but Markdown fences are
    removed defensively.
    """

    text = text.strip()

    if text.startswith("```"):
        lines = text.splitlines()

        if lines and lines[0].strip().startswith("```"):
            lines = lines[1:]

        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]

        text = "\n".join(lines).strip()

        # Defensive handling in case the first remaining line is only "json".
        if text.lower().startswith("json\n"):
            text = text[5:].strip()

    parsed = json.loads(text)

    if not isinstance(parsed, list):
        raise TypeError(
            "The provider response must contain a JSON array; "
            f"received {type(parsed).__name__}."
        )

    return [
        row
        for row in parsed
        if isinstance(row, dict)
    ]


def call_model(
    *,
    provider: GenerationProvider,
    category: str,
    language: str,
    requested_examples: int,
    turn_count: int,
    max_output_tokens: int,
) -> list[dict[str, Any]]:
    """Build the category prompt, call the provider and parse its JSON."""

    prompt = build_category_prompt(
        category,
        language,
        requested_examples,
        turn_count,
    )

    text = provider.generate(
        prompt=prompt,
        max_output_tokens=max_output_tokens,
    )

    return extract_json_array(text)

###############################################################################
# Target calculations
###############################################################################


def calculate_depth_targets(
    *,
    category: str,
    target_n: int,
) -> dict[int, int]:
    """
    Convert fractional depth distributions into exact integer targets.

    Any rounding drift is absorbed by the first depth bucket.
    """

    distribution = TURN_DEPTH[category]

    targets = {
        depth: round(target_n * fraction)
        for depth, fraction in distribution.items()
    }

    drift = target_n - sum(targets.values())

    first_depth = next(iter(targets))
    targets[first_depth] += drift

    return targets


def calculate_max_calls(
    *,
    target_n: int,
    request_batch_size: int,
    expected_acceptance_rate: float,
    safety_factor: float,
) -> int:
    """
    Estimate an appropriate API-call ceiling.

    Example:
        target_n = 750
        request_batch_size = 20
        acceptance = 0.70
        safety = 1.30

        Expected accepted per call = 14
        Base calls = ceil(750 / 14) = 54
        Safe calls = ceil(54 * 1.30) = 71
    """

    if target_n <= 0:
        return 0

    if request_batch_size <= 0:
        raise ValueError(
            "request_batch_size must be greater than zero"
        )

    if not 0 < expected_acceptance_rate <= 1:
        raise ValueError(
            "expected_acceptance_rate must be between 0 and 1"
        )

    if safety_factor < 1:
        raise ValueError(
            "safety_factor must be at least 1"
        )

    expected_accepted_per_call = (
        request_batch_size * expected_acceptance_rate
    )

    estimated_calls = math.ceil(
        target_n / expected_accepted_per_call
    )

    safe_calls = math.ceil(
        estimated_calls * safety_factor
    )

    return min(
        MAX_API_CALLS_HARD_LIMIT,
        max(
            MIN_API_CALLS_PER_TARGET,
            safe_calls,
        ),
    )

###############################################################################
# Existing output and exclusion data
###############################################################################


def load_jsonl_rows(
    path: Path,
) -> list[dict[str, Any]]:
    """Load a JSONL file, raising a useful error for malformed rows."""

    if not path.exists():
        return []

    rows: list[dict[str, Any]] = []

    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()

            if not line:
                continue

            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON in {path}, line {line_no}: {exc}"
                ) from exc

            if not isinstance(row, dict):
                raise TypeError(
                    f"{path}, line {line_no}: expected a JSON object"
                )

            rows.append(row)

    return rows


def validate_existing_rows(
    *,
    rows: list[dict[str, Any]],
    category: str,
    language: str,
    source_path: Path,
) -> None:
    """Ensure a partial output file is safe to resume."""

    seen_ids: set[str] = set()
    seen_keys: set[str] = set()

    for row_index, row in enumerate(rows, start=1):
        row_id = row.get("id")

        if row.get("category") != category:
            raise ValueError(
                f"{source_path}, row {row_index}: category is "
                f"{row.get('category')!r}, expected {category!r}"
            )

        if row.get("language") != language:
            raise ValueError(
                f"{source_path}, row {row_index}: language is "
                f"{row.get('language')!r}, expected {language!r}"
            )

        if row.get("stage") != "stage0":
            raise ValueError(
                f"{source_path}, row {row_index}: stage is "
                f"{row.get('stage')!r}, expected 'stage0'"
            )

        messages = row.get("messages")

        if messages is None:
            raise ValueError(
                f"{source_path}, row {row_index}: missing messages"
            )

        errors = validate_row(
            category,
            language,
            messages,
        )

        if errors:
            raise ValueError(
                f"{source_path}, row {row_index}: existing row "
                f"failed validation: {errors}"
            )

        if not isinstance(row_id, str) or not row_id:
            raise ValueError(
                f"{source_path}, row {row_index}: missing or invalid id"
            )

        if row_id in seen_ids:
            raise ValueError(
                f"{source_path}: duplicate existing id {row_id!r}"
            )

        key = content_key(messages)

        if key in seen_keys:
            raise ValueError(
                f"{source_path}: duplicate existing conversation "
                f"at row {row_index}"
            )

        seen_ids.add(row_id)
        seen_keys.add(key)


def load_exclusion_keys(
    exclude_dirs: list[str] | None,
) -> set[str]:
    """
    Load conversation keys from previous datasets.

    Directories are searched recursively for *.jsonl files.
    """

    if not exclude_dirs:
        return set()

    keys: set[str] = set()
    files_read = 0
    malformed_rows = 0

    for exclude_dir in exclude_dirs:
        pattern = os.path.join(
            exclude_dir,
            "**",
            "*.jsonl",
        )

        for filepath in glob.glob(
            pattern,
            recursive=True,
        ):
            files_read += 1

            with open(
                filepath,
                encoding="utf-8",
            ) as handle:
                for line_no, line in enumerate(
                    handle,
                    start=1,
                ):
                    line = line.strip()

                    if not line:
                        continue

                    try:
                        row = json.loads(line)
                        messages = row["messages"]
                        keys.add(content_key(messages))
                    except (
                        json.JSONDecodeError,
                        KeyError,
                        TypeError,
                    ) as exc:
                        malformed_rows += 1

                        print(
                            "WARNING: skipped malformed exclusion row "
                            f"{filepath}:{line_no}: {exc}"
                        )

    print(
        f"Loaded {len(keys):,} exclusion keys from "
        f"{files_read:,} JSONL files."
    )

    if malformed_rows:
        print(
            f"WARNING: skipped {malformed_rows:,} malformed "
            "exclusion rows."
        )

    return keys


def count_existing_rows_by_depth(
    rows: list[dict[str, Any]],
) -> Counter[int]:
    """Count existing accepted rows by message count."""

    return Counter(
        len(row["messages"])
        for row in rows
    )

###############################################################################
# Rejection diagnostics
###############################################################################


def display_rejection_summary(
    rejection_counts: Counter[str],
    *,
    indent: str = "      ",
) -> None:
    """Print compact rejection diagnostics."""

    if not rejection_counts:
        print(f"{indent}rejected: none")
        return

    total = sum(
        count
        for reason, count in rejection_counts.items()
        if ":" not in reason
    )

    print(
        f"{indent}rejected total: {total}"
    )

    for reason, count in rejection_counts.most_common():
        print(
            f"{indent}- {reason}: {count}"
        )

###############################################################################
# Generation
###############################################################################


def generate_depth_partition(
    *,
    provider: GenerationProvider,
    category: str,
    language: str,
    turn_count: int,
    remaining_target: int,
    banned_keys: set[str],
    max_output_tokens: int,
    request_batch_size: int,
    expected_acceptance_rate: float,
    call_safety_factor: float,
) -> tuple[
    list[list[dict[str, str]]],
    Counter[str],
]:
    """
    Generate one category/language/message-depth partition.

    The shared banned_keys set is updated immediately when a conversation
    is accepted, preventing duplicates across the complete run.
    """

    if remaining_target <= 0:
        return [], Counter()

    max_calls = calculate_max_calls(
        target_n=remaining_target,
        request_batch_size=request_batch_size,
        expected_acceptance_rate=expected_acceptance_rate,
        safety_factor=call_safety_factor,
    )

    accepted: list[list[dict[str, str]]] = []
    rejection_counts: Counter[str] = Counter()

    got = 0
    calls = 0

    print(
        f"    depth target: {remaining_target}, "
        f"max calls: {max_calls}"
    )

    while got < remaining_target and calls < max_calls:
        calls += 1

        missing = remaining_target - got

        # Oversample slightly to compensate for invalid or duplicate rows.
        requested_examples = min(
            request_batch_size,
            max(
                5,
                missing + 5,
            ),
        )

        try:
            raw_rows = call_model(
                provider=provider,
                category=category,
                language=language,
                requested_examples=requested_examples,
                turn_count=turn_count,
                max_output_tokens=max_output_tokens,
            )

        except json.JSONDecodeError as exc:
            rejection_counts["json_parse_failure"] += 1

            print(
                f"    ! JSON parse failure "
                f"({category}/{language}/{turn_count}msg, "
                f"call {calls}/{max_calls}): {exc}"
            )

            time.sleep(RETRY_SLEEP_SECONDS)
            continue

        except TypeError as exc:
            rejection_counts["invalid_response_shape"] += 1

            print(
                f"    ! Invalid response shape "
                f"({category}/{language}/{turn_count}msg, "
                f"call {calls}/{max_calls}): {exc}"
            )

            time.sleep(RETRY_SLEEP_SECONDS)
            continue

        except Exception as exc:
            # Provider SDKs use different exception classes, so the common
            # generator catches ordinary exceptions here and logs the concrete
            # exception type. KeyboardInterrupt and SystemExit are unaffected.
            rejection_counts[
                f"provider_error:{type(exc).__name__}"
            ] += 1

            print(
                f"    ! Provider error "
                f"({category}/{language}/{turn_count}msg, "
                f"call {calls}/{max_calls}): "
                f"{type(exc).__name__}: {exc}"
            )

            time.sleep(RETRY_SLEEP_SECONDS)
            continue

        batch_accepted = 0

        if provider.last_response_model and provider.last_response_model != provider.model:
            print(
                f"    ! MODEL MISMATCH: requested {provider.model!r} but the "
                f"API reports it was served by {provider.last_response_model!r}. "
                f"This can happen with a mistyped or unrecognized model string "
                f"resolving to something unintended — verify before trusting "
                f"this batch's content."
            )

        if len(raw_rows) < requested_examples:
            rejection_counts["provider_returned_fewer_rows"] += (
                requested_examples - len(raw_rows)
            )

        for row in raw_rows:
            messages = row.get("messages")

            if messages is None:
                rejection_counts["missing_messages"] += 1
                continue

            errors = validate_row(
                category,
                language,
                messages,
            )

            if errors:
                rejection_counts["validation_failure"] += 1

                for error in errors:
                    rejection_counts[
                        f"validation:{error}"
                    ] += 1

                continue

            if len(messages) != turn_count:
                # This may already be checked by validate_row, but keeping
                # it explicit protects the requested depth distribution.
                rejection_counts["wrong_turn_depth"] += 1
                continue

            key = content_key(messages)

            if key in banned_keys:
                rejection_counts["duplicate_content"] += 1
                continue

            banned_keys.add(key)
            accepted.append(messages)

            got += 1
            batch_accepted += 1

            if got >= remaining_target:
                break

        print(
            f"    {category}/{language}/{turn_count}msg "
            f"call {calls}/{max_calls}: "
            f"requested={requested_examples}, "
            f"returned={len(raw_rows)}, "
            f"accepted=+{batch_accepted} "
            f"({got}/{remaining_target})"
        )

        if SUCCESS_SLEEP_SECONDS > 0:
            time.sleep(SUCCESS_SLEEP_SECONDS)

    if got < remaining_target:
        print(
            f"    ! shortfall: "
            f"{category}/{language}/{turn_count}msg got "
            f"{got}/{remaining_target} after {calls} calls"
        )

    display_rejection_summary(
        rejection_counts,
    )

    return accepted, rejection_counts


def generate_category_language(
    *,
    provider: GenerationProvider,
    category: str,
    language: str,
    target_n: int,
    existing_rows: list[dict[str, Any]],
    banned_keys: set[str],
    max_output_tokens: int,
    request_batch_size: int,
    expected_acceptance_rate: float,
    call_safety_factor: float,
) -> tuple[
    list[list[dict[str, str]]],
    Counter[str],
]:
    """
    Generate only the missing rows for one category/language output file.

    Existing rows are accounted for separately by message depth.
    """

    target_by_depth = calculate_depth_targets(
        category=category,
        target_n=target_n,
    )

    existing_by_depth = count_existing_rows_by_depth(
        existing_rows
    )

    accepted_messages: list[list[dict[str, str]]] = []
    total_rejections: Counter[str] = Counter()

    for turn_count, depth_target in target_by_depth.items():
        existing_count = existing_by_depth.get(
            turn_count,
            0,
        )

        remaining_target = max(
            0,
            depth_target - existing_count,
        )

        print(
            f"  Depth {turn_count}: target={depth_target}, "
            f"existing={existing_count}, "
            f"remaining={remaining_target}"
        )

        if remaining_target == 0:
            continue

        generated, rejections = generate_depth_partition(
            provider=provider,
            category=category,
            language=language,
            turn_count=turn_count,
            remaining_target=remaining_target,
            banned_keys=banned_keys,
            max_output_tokens=max_output_tokens,
            request_batch_size=request_batch_size,
            expected_acceptance_rate=expected_acceptance_rate,
            call_safety_factor=call_safety_factor,
        )

        accepted_messages.extend(generated)
        total_rejections.update(rejections)

    return accepted_messages, total_rejections

###############################################################################
# Row construction and output
###############################################################################


def build_new_rows(
    *,
    category: str,
    language: str,
    messages_list: list[list[dict[str, str]]],
    start_index: int,
) -> list[dict[str, Any]]:
    """Assign IDs to newly generated message lists."""

    return [
        {
            "id": (
                f"{category}_{language}_sft_"
                f"{row_index:05d}"
            ),
            "category": category,
            "language": language,
            "stage": "stage0",
            "messages": messages,
        }
        for row_index, messages in enumerate(
            messages_list,
            start=start_index,
        )
    ]


def write_jsonl_atomic(
    *,
    path: Path,
    rows: list[dict[str, Any]],
) -> None:
    """
    Write JSONL through a temporary file, then replace the destination.

    This reduces the risk of leaving a partially written batch if the
    process is interrupted during the write operation.
    """

    temporary_path = path.with_suffix(
        path.suffix + ".tmp"
    )

    with temporary_path.open(
        "w",
        encoding="utf-8",
    ) as handle:
        for row in rows:
            handle.write(
                json.dumps(
                    row,
                    ensure_ascii=False,
                )
                + "\n"
            )

    temporary_path.replace(path)


def write_generation_report(
    *,
    output_path: Path,
    provider: GenerationProvider,
    category: str,
    language: str,
    batch_id: str,
    target_n: int,
    existing_before: int,
    generated_now: int,
    final_rows: list[dict[str, Any]],
    rejection_counts: Counter[str],
) -> None:
    """Write a small JSON report beside each generated batch file."""

    rows_by_depth = Counter(
        len(row["messages"])
        for row in final_rows
    )

    report = {
        "provider": provider.name,
        "model_requested": provider.model,
        "model_actually_served": getattr(provider, "last_response_model", None),
        "category": category,
        "language": language,
        "batch_id": batch_id,
        "target_rows": target_n,
        "existing_rows_before_run": existing_before,
        "generated_rows_this_run": generated_now,
        "final_rows": len(final_rows),
        "complete": len(final_rows) >= target_n,
        "rows_by_message_count": {
            str(depth): count
            for depth, count in sorted(
                rows_by_depth.items()
            )
        },
        "rejections": dict(
            rejection_counts
        ),
    }

    report_path = output_path.with_name(
        output_path.stem + "_report.json"
    )

    report_path.write_text(
        json.dumps(
            report,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

###############################################################################
# CLI
###############################################################################


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate structurally validated Stage 0 SFT "
            "conversation batches."
        )
    )

    parser.add_argument(
        "--provider",
        choices=(
            "anthropic",
            "openai",
        ),
        required=True,
    )

    parser.add_argument(
        "--model",
        default=None,
        help=(
            "Optional model override. Otherwise uses the "
            "provider-specific environment/default model."
        ),
    )

    parser.add_argument(
        "--out-dir",
        required=True,
    )

    parser.add_argument(
        "--exclude-dir",
        action="append",
        default=None,
        help=(
            "Directory containing prior JSONL datasets to exclude "
            "exact duplicates against. May be passed multiple times."
        ),
    )

    parser.add_argument(
        "--categories",
        nargs="*",
        default=list(CATEGORY_TARGETS),
    )

    parser.add_argument(
        "--languages",
        nargs="*",
        default=[
            "en",
            "pt",
        ],
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--batch-id",
        default="0001",
        help=(
            "Four-digit generation batch identifier, "
            "for example 0002."
        ),
    )

    parser.add_argument(
        "--max-output-tokens",
        type=int,
        default=DEFAULT_MAX_OUTPUT_TOKENS,
    )

    parser.add_argument(
        "--request-batch-size",
        type=int,
        default=REQUEST_BATCH_SIZE,
    )

    parser.add_argument(
        "--expected-acceptance-rate",
        type=float,
        default=EXPECTED_ACCEPTANCE_RATE,
        help=(
            "Estimated fraction of requested examples that pass "
            "validation and deduplication."
        ),
    )

    parser.add_argument(
        "--call-safety-factor",
        type=float,
        default=CALL_SAFETY_FACTOR,
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help=(
            "Discard existing output files instead of resuming them."
        ),
    )

    return parser.parse_args()


def validate_args(
    args: argparse.Namespace,
) -> None:
    unknown_categories = (
        set(args.categories)
        - set(CATEGORY_TARGETS)
    )

    if unknown_categories:
        raise ValueError(
            "Unknown categories: "
            f"{sorted(unknown_categories)}"
        )

    unknown_languages = (
        set(args.languages)
        - {
            "en",
            "pt",
        }
    )

    if unknown_languages:
        raise ValueError(
            "Unknown languages: "
            f"{sorted(unknown_languages)}"
        )

    if not (
        len(args.batch_id) == 4
        and args.batch_id.isdigit()
    ):
        raise ValueError(
            "--batch-id must contain exactly four digits, "
            "for example 0002"
        )

    if args.request_batch_size <= 0:
        raise ValueError(
            "--request-batch-size must be greater than zero"
        )

    if not (
        0 < args.expected_acceptance_rate <= 1
    ):
        raise ValueError(
            "--expected-acceptance-rate must be between 0 and 1"
        )

    if args.call_safety_factor < 1:
        raise ValueError(
            "--call-safety-factor must be at least 1"
        )

    if args.max_output_tokens <= 0:
        raise ValueError(
            "--max-output-tokens must be greater than zero"
        )

###############################################################################
# Main
###############################################################################


def main() -> None:
    args = parse_args()
    validate_args(args)

    if args.seed is not None:
        random.seed(args.seed)

    try:
        provider = build_provider(
            provider_name=args.provider,
            model=args.model,
        )

    except RuntimeError as exc:
        print(f"ERROR: {exc}")
        sys.exit(1)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Load previous datasets supplied explicitly through --exclude-dir.
    banned_keys = load_exclusion_keys(
        args.exclude_dir
    )

    total_existing_before = 0
    total_generated_now = 0
    total_final_rows = 0
    incomplete_files: list[str] = []

    print(
        f"Provider: {provider.name}\n"
        f"Model: {provider.model}\n"
        f"Batch: {args.batch_id}\n"
        f"Request batch size: {args.request_batch_size}\n"
        f"Expected acceptance rate: "
        f"{args.expected_acceptance_rate:.2f}\n"
        f"Call safety factor: "
        f"{args.call_safety_factor:.2f}"
    )

    for category in args.categories:
        total_category_target = CATEGORY_TARGETS[
            category
        ]

        if (
            total_category_target
            % len(args.languages)
            != 0
        ):
            raise ValueError(
                f"Target {total_category_target} for "
                f"{category!r} cannot be evenly divided "
                f"across languages {args.languages}"
            )

        per_language_target = (
            total_category_target
            // len(args.languages)
        )

        for language in args.languages:
            output_path = out_dir / (
                f"{category}_{language}_"
                f"{args.batch_id}.jsonl"
            )

            print(
                "\n"
                + "=" * 70
            )
            print(
                f"{category} / {language} "
                f"(target {per_language_target})"
            )
            print(
                f"Output: {output_path}"
            )

            if args.overwrite and output_path.exists():
                print(
                    "  --overwrite enabled: removing "
                    "existing output."
                )

                output_path.unlink()

            existing_rows = load_jsonl_rows(
                output_path
            )

            if existing_rows:
                validate_existing_rows(
                    rows=existing_rows,
                    category=category,
                    language=language,
                    source_path=output_path,
                )

                print(
                    f"  Resuming from "
                    f"{len(existing_rows)} existing rows."
                )

            # Existing rows from the current batch must also be banned.
            for row in existing_rows:
                banned_keys.add(
                    content_key(
                        row["messages"]
                    )
                )

            existing_before = len(existing_rows)
            remaining_total = max(
                0,
                per_language_target - existing_before,
            )

            total_existing_before += existing_before

            if remaining_total == 0:
                print(
                    "  File is already complete; "
                    "no generation required."
                )

                write_generation_report(
                    output_path=output_path,
                    provider=provider,
                    category=category,
                    language=language,
                    batch_id=args.batch_id,
                    target_n=per_language_target,
                    existing_before=existing_before,
                    generated_now=0,
                    final_rows=existing_rows,
                    rejection_counts=Counter(),
                )

                total_final_rows += len(existing_rows)
                continue

            print(
                f"  Remaining total rows: "
                f"{remaining_total}"
            )

            messages_list, rejection_counts = (
                generate_category_language(
                    provider=provider,
                    category=category,
                    language=language,
                    target_n=per_language_target,
                    existing_rows=existing_rows,
                    banned_keys=banned_keys,
                    max_output_tokens=args.max_output_tokens,
                    request_batch_size=args.request_batch_size,
                    expected_acceptance_rate=(
                        args.expected_acceptance_rate
                    ),
                    call_safety_factor=(
                        args.call_safety_factor
                    ),
                )
            )

            new_rows = build_new_rows(
                category=category,
                language=language,
                messages_list=messages_list,
                start_index=existing_before + 1,
            )

            combined_rows = (
                existing_rows + new_rows
            )

            # Defensive cap: never exceed the configured target.
            combined_rows = combined_rows[
                :per_language_target
            ]

            write_jsonl_atomic(
                path=output_path,
                rows=combined_rows,
            )

            generated_now = (
                len(combined_rows)
                - existing_before
            )

            total_generated_now += generated_now
            total_final_rows += len(combined_rows)

            print(
                f"  -> preserved: {existing_before}"
            )
            print(
                f"  -> generated now: {generated_now}"
            )
            print(
                f"  -> final rows: "
                f"{len(combined_rows)}/"
                f"{per_language_target}"
            )
            print(
                f"  -> wrote: {output_path}"
            )

            write_generation_report(
                output_path=output_path,
                provider=provider,
                category=category,
                language=language,
                batch_id=args.batch_id,
                target_n=per_language_target,
                existing_before=existing_before,
                generated_now=generated_now,
                final_rows=combined_rows,
                rejection_counts=rejection_counts,
            )

            if len(combined_rows) < per_language_target:
                incomplete_files.append(
                    str(output_path)
                )

    print(
        "\n"
        + "=" * 70
    )
    print(
        "GENERATION SUMMARY"
    )
    print(
        f"Existing rows before run: "
        f"{total_existing_before:,}"
    )
    print(
        f"New rows generated: "
        f"{total_generated_now:,}"
    )
    print(
        f"Final rows across selected files: "
        f"{total_final_rows:,}"
    )

    if incomplete_files:
        print(
            "\nINCOMPLETE FILES:"
        )

        for filepath in incomplete_files:
            print(
                f"  - {filepath}"
            )

        print(
            "\nRun the same command again to resume "
            "only the missing rows."
        )

        sys.exit(2)

    print(
        "\nAll selected files reached their targets."
    )
    print(
        "NOTE: deterministic structural validation and exact "
        "deduplication were applied. Semantic review remains "
        "a separate step."
    )


if __name__ == "__main__":
    main()