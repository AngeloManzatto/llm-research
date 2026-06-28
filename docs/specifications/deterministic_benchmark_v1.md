# Deterministic Benchmark Specification v1.0

## Purpose

This specification defines how benchmarks should be designed, implemented, executed, and reported in this repository.

The goal is to evaluate small language models using deterministic, reproducible, scriptable methods without relying on human judgment, external language models, or subjective interpretation.

## Core Principle

Every primary benchmark score must be computable using only deterministic algorithms implemented inside this repository.

No benchmark may require another language model, human evaluator, or external service to determine whether an answer is correct.

## Design Principles

### 1. Determinism

Running the same model checkpoint with the same benchmark version and decoding configuration must produce the same score.

### 2. Independence

Benchmark evaluation must not depend on external LLMs, proprietary judges, human raters, or non-reproducible services.

### 3. Reproducibility

Every benchmark run must record:

* benchmark id
* benchmark version
* model id
* checkpoint path
* decoding configuration
* timestamp
* metric versions

### 4. Capability Isolation

Each benchmark example should test one primary capability whenever possible.

A local-context test should not require world knowledge.
A formatting test should not require reasoning.
An uncertainty test should not require factual recall.

### 5. Metric Transparency

Every metric must be implemented as a self-contained object with:

* metric id
* version
* description
* deterministic flag
* output type
* evaluation method

### 6. Separation of Concerns

The benchmark framework separates:

* benchmark representation
* model generation
* metric evaluation
* result aggregation
* report writing

### 7. Version Stability

Benchmarks and metrics must be versioned.
Changing a benchmark dataset or metric behavior requires a version update.

## Benchmark Manifest Schema

Each benchmark must define a `benchmark.json` manifest:

```json
{
  "benchmark_id": "conversation_level0",
  "version": "0.1.0",
  "description": "Deterministic Level 0 conversation benchmark.",
  "data_files": [
    "data/smoke_test.jsonl",
    "data/core_001.jsonl"
  ],
  "default_decode": {
    "method": "greedy",
    "max_length": 64
  },
  "scoring_metric": "contains_expected",
  "diagnostic_metrics": [
    "wiki_talk_artifact",
    "repetition",
    "role_leakage",
    "word_count",
    "too_long"
  ]
}
```

## Example Schema

Each JSONL row must contain:

```json
{
  "id": "ctx_001",
  "category": "local_context",
  "capability": "short_context_recall",
  "difficulty": "easy",
  "language": "en",
  "prompt": "User: I have a blue notebook.\nAssistant:",
  "expected_any": ["blue"],
  "scoring": "contains"
}
```

## Required Example Fields

| Field          | Meaning                         |
| -------------- | ------------------------------- |
| `id`           | Stable unique example id        |
| `category`     | Benchmark category              |
| `capability`   | Specific ability being tested   |
| `difficulty`   | easy, medium, or hard           |
| `language`     | en, pt, or mixed                |
| `prompt`       | Input prompt given to the model |
| `expected_any` | Accepted answer substrings      |
| `scoring`      | Example-level scoring style     |

## Level 0 Conversation Categories

### turn_taking

Tests whether the model responds after a `User:` / `Assistant:` prompt.

### knowledge_completion

Tests simple factual completion without chat-role structure.

### local_context

Tests whether the model uses information stated inside the prompt.

### correction

Tests whether the model updates an earlier statement after correction.

### instruction_following

Tests whether the model follows simple, scriptable instructions.

### uncertainty

Tests whether the model recognizes insufficient information.

## Metric Types

### Scoring Metric

Determines whether the example passed.

Example:

* `contains_expected`

### Diagnostic Metrics

Characterize failure modes but do not define pass/fail.

Examples:

* `wiki_talk_artifact`
* `repetition`
* `role_leakage`
* `word_count`
* `too_long`

## Non-Allowed Evaluation Methods

Primary benchmark scores must not use:

* LLM-as-judge
* human preference ranking
* subjective coherence scoring
* semantic similarity models
* embedding similarity
* probabilistic “maybe correct” scoring

These may be used later only as exploratory annotations, never as the primary score.

## Guiding Rule

If a Python script cannot deterministically score the example, the example does not belong in the benchmark yet.
