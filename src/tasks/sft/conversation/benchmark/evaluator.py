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

# TODO rename
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
    Merges three layers, in increasing precedence:

      1. Category-level shared context (benchmark.category_shared_context)
         -- e.g. uncertainty's refusal_patterns, defined ONCE for the whole
         category rather than duplicated into every row. Solves the "hard
         link between category and pattern" problem: expanding the
         refusal-pattern list is now a one-line manifest edit, not a
         rewrite of every uncertainty row.
      2. The row's own `meta` -- for anything genuinely row-specific
         (stated_value, corrected_value, constraint_type, ...).
      3. The row's own `expected_any`, but ONLY if it's non-empty. This
         keeps existing categories working exactly as before --
         knowledge_completion/local_context/correction rows have real,
         row-specific facts in expected_any and that must win. A category
         relying entirely on shared context (uncertainty, going forward)
         simply ships an empty expected_any per row and gets the shared
         refusal_patterns list instead.
    """
    context: dict[str, Any] = dict(benchmark.category_shared_context.get(example.category, {}))
    context.update(example.meta or {})

    if example.expected_any:
        context["expected_any"] = example.expected_any

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