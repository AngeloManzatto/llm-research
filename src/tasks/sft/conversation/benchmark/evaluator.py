"""
Created on Sun Jun 28 08:11:28 2026

@author: Angelo Antonio Manzatto
"""

###############################################################################
# Libraries
###############################################################################

import re
from dataclasses import dataclass
from typing import Any
 
from src.tasks.sft.conversation.core.special_tokens import STOP_TOKENS, TOKEN_BY_NAME

from src.tasks.sft.conversation.benchmark.metric import run_metric, MetricResult
from src.tasks.sft.conversation.benchmark.benchmark import Benchmark, BenchmarkExample
 
###############################################################################
# Evaluation Result
###############################################################################
 
@dataclass(frozen=True)
class EvaluationResult:
    id: str
    category: str
    language: str
    messages: list[dict[str, str]]
    expected_any: list[str]
    raw_answer: str
    answer: str
    passed: bool
    scoring_metric: str
    metrics: dict[str, MetricResult]
    decode: dict[str, Any]
 
    def to_dict(self) -> dict[str, Any]:
        flat: dict[str, Any] = {}
        for name, result in self.metrics.items():
            flat[name] = result.passed
            if result.details:
                flat[f"{name}_details"] = result.details
 
        return {
            "id": self.id,
            "category": self.category,
            "language": self.language,
            "messages": self.messages,
            "expected_any": self.expected_any,
            "raw_answer": self.raw_answer,
            "answer": self.answer,
            "passed": self.passed,
            "scoring_metric": self.scoring_metric,
            **flat,
            "decode": self.decode,
        }
 
###############################################################################
# Normalize answer
###############################################################################
 
def strip_special_stop_tokens(text: str) -> str:
    for tok in STOP_TOKENS:
        text = text.replace(tok, "")
    return text
 
def normalize_answer(
    answer: str,
    *,
    strip_stop_tokens: bool = True,
    normalize_whitespace: bool = True,
) -> str:
    if strip_stop_tokens:
        answer = strip_special_stop_tokens(answer)
    if normalize_whitespace:
        answer = re.sub(r"\s+", " ", answer).strip()
    return answer
 
###############################################################################
# Build per-example context
#
# One generic dict, not one branch per metric_id. Any metric can pull
# whatever key it needs out of this via **kwargs; metric.py's own
# `.requires` declaration is the single source of truth for what's
# required, checked once inside run_metric(). This function's only job
# is translating the small, fixed set of real BenchmarkExample fields
# into the same flat namespace as `meta` — it does not grow when a new
# metric is added, only when a genuinely new *kind* of real field is
# added to BenchmarkExample itself (rare, and separate from adding
# metrics).
###############################################################################
 
def _build_context(example: BenchmarkExample, benchmark: Benchmark) -> dict[str, Any]:
    """
    Merges category-level shared context with the row's own data, using
    type-aware rules rather than one blanket precedence order:

      - LIST-valued keys (expected_any, refusal_patterns, ...) are UNIONED
        -- category defaults plus whatever the row adds, de-duplicated,
        order preserved. There is no scenario where a row should EXCLUDE a
        category-level default acceptable answer, so merging can only make
        scoring more correct, never less. This also means a category's own
        per-row list (e.g. an older, narrower uncertainty refusal list)
        does not need to be blanked out for a broader shared list to take
        effect -- it simply merges in, harmlessly.
      - SCALAR-valued keys (constraint_type, stated_value, ...) still have
        the row's own value win outright when present -- there's no
        sensible way to "merge" two different single values, so this is
        genuine precedence, not a union.

    An earlier version used blanket "row overrides shared if non-empty"
    precedence for everything, including expected_any -- which silently
    discarded the shared refusal-pattern list for every uncertainty row,
    since each row still shipped its own narrow list. That required every
    future row author to remember to leave expected_any empty for a
    category using shared context, an easy-to-forget, easy-to-silently-
    break rule. Type-aware merging removes the need to remember it at all.
    """
    shared = benchmark.category_shared_context.get(example.category, {})

    row_data: dict[str, Any] = dict(example.meta or {})
    row_data.setdefault("expected_any", example.expected_any)

    context: dict[str, Any] = {}
    for key in set(shared) | set(row_data):
        shared_val = shared.get(key)
        row_val = row_data.get(key)
        if isinstance(shared_val, list) and isinstance(row_val, list):
            context[key] = list(dict.fromkeys(shared_val + row_val))  # union, de-duplicated, order preserved
        elif row_val not in (None, [], ""):
            context[key] = row_val
        elif shared_val is not None:
            context[key] = shared_val
        else:
            context[key] = row_val

    if example.expected_stop_token is not None:
        context.setdefault("expected_token", TOKEN_BY_NAME[example.expected_stop_token].token)

    return context
 
###############################################################################
# Evaluate example
###############################################################################
 
def evaluate_example(
    *,
    benchmark: Benchmark,
    example: BenchmarkExample,
    generated: str,
    decode: dict[str, Any],
) -> EvaluationResult:
    """
    `benchmark.category_scoring_metric` and `benchmark.always_computed`
    drive which metric scores this example and which are pooled across
    every category — changing either is a manifest edit, not a code
    change (see Benchmark.from_manifest).
    """
    # greedy_decode returns only the generated portion (no prompt prefix to strip)
    raw_answer = generated.strip()
    answer     = normalize_answer(raw_answer, strip_stop_tokens=True, normalize_whitespace=True)

    category_scoring_metric = benchmark.category_scoring_metric
    if example.category not in category_scoring_metric:
        raise KeyError(
            f"No scoring metric configured for category {example.category!r}. "
            f"Configured categories: {sorted(category_scoring_metric)}"
        )
    scoring_id = category_scoring_metric[example.category]

    context = _build_context(example, benchmark)

    metrics: dict[str, MetricResult] = {}
    for metric_id in {scoring_id, *benchmark.always_computed}:
        metrics[metric_id] = run_metric(metric_id, raw_answer, **context)

    passed = metrics[scoring_id].passed

    return EvaluationResult(
        id=example.id,
        category=example.category,
        language=example.language,
        messages=example.messages,
        expected_any=example.expected_any,
        raw_answer=raw_answer,
        answer=answer,
        passed=passed,
        scoring_metric=scoring_id,
        metrics=metrics,
        decode=decode,
    )