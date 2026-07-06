# Stage 0 — Learning the Mechanics of Conversation

## Status

**Design Phase**

---

# Objective

The objective of Stage 0 is to teach a pretrained next-token prediction (NTP) language model the fundamental mechanics of a conversation.

At the end of this stage, the model is not expected to be intelligent, helpful, or capable of reasoning. Instead, it should demonstrate that it understands the structure of a dialogue and can participate in simple conversational exchanges.

This stage intentionally focuses on conversational mechanics rather than knowledge acquisition.

---

# Motivation

The base model was trained exclusively with next-token prediction over large text corpora.

Although this provides linguistic knowledge and text completion ability, it does not explicitly teach:

* when a user has finished speaking;
* when the assistant should respond;
* how to maintain conversational context;
* how to update information after corrections;
* how to admit insufficient information;
* when to terminate a response.

Stage 0 introduces these conversational concepts through supervised fine-tuning.

---

# Scope

Stage 0 is intentionally limited.

## In Scope

* User / Assistant interaction
* Turn taking
* Short context retention
* Correction handling
* Basic uncertainty responses
* Response termination
* Conversation formatting

## Out of Scope

The following capabilities are explicitly **not** objectives of Stage 0:

* Mathematical reasoning
* Logical reasoning
* Tool use
* Long-context reasoning
* Planning
* Safety alignment
* Coding
* World knowledge expansion
* Multi-agent behavior

Performance changes in these areas are considered incidental.

---

# Benchmark

Stage 0 is evaluated exclusively using the deterministic benchmark:

```
conversation_level0
```

Current baseline:

```
Checkpoint:
ckpt-1017525

Pass Rate:
10 / 66
15.2%
```

This benchmark provides deterministic evaluation without requiring human judgment or external language models.

---

# Conversation Capabilities

Stage 0 targets the following conversational abilities.

## Turn Taking

Generate exactly one assistant response after a user prompt.

---

## Knowledge Completion

Answer simple factual prompts already represented by the pretrained model.

This category measures retention of existing language knowledge rather than new learning.

---

## Local Context

Use information presented earlier in the same conversation.

---

## Correction

Replace outdated information when corrected by the user.

---

## Instruction Following

Follow simple formatting constraints.

Examples:

* answer with one word;
* answer yes/no;
* answer with a number only.

---

## Uncertainty

Recognize when insufficient information exists.

Instead of hallucinating, the model should acknowledge that the requested information cannot be determined from the conversation.

---

# Conversation Protocol

Stage 0 introduces explicit conversation boundary tokens.

## END_OF_TURN

```
<SPECIAL-0>
```

Marks the end of a single assistant turn.

Every assistant response terminates with this token.

Inference stops generation when this token is produced.

---

## END_OF_CONVERSATION

```
<SPECIAL-1>
```

Marks the end of a complete multi-turn conversation.

This token is used only on the final assistant response of an explicitly multi-turn training example.

Single-turn examples never use this token.

---

# Dataset Format

Training examples are stored as structured JSON.

Example:

```json
{
    "id": "...",
    "stage": "stage0",
    "category": "...",
    "capability": "...",
    "language": "...",
    "turns": [
        ...
    ]
}
```

A formatting pipeline converts this representation into the textual format consumed by the tokenizer.

This separation keeps the dataset human-readable while allowing future formatting changes without modifying the original annotations.

---

# Training Philosophy

Stage 0 is intended to teach conversation mechanics rather than memorize answers.

The benchmark and the training dataset are intentionally independent.

The benchmark acts as an examination.

The training dataset acts as instruction.

Neither should be derived mechanically from the other.

---

# Exit Criteria

Stage 0 is considered complete when the model consistently demonstrates conversational behavior on the deterministic benchmark.

Target values will be established after the first supervised fine-tuning experiments.

---

# Research Questions

Stage 0 aims to answer the following questions:

1. How much conversational behavior can be learned through supervised fine-tuning alone?

2. How many examples are required before conversational performance begins to saturate?

3. Does explicit conversation structure improve response quality without modifying the transformer architecture?

4. Does introducing explicit stop tokens reduce response repetition?

5. Does multi-turn supervision improve single-turn conversational performance?

---

# Expected Deliverables

At the conclusion of Stage 0 the repository should contain:

* A deterministic conversation benchmark.
* A reproducible evaluation pipeline.
* A structured Stage 0 SFT dataset.
* A conversationally fine-tuned checkpoint.
* Benchmark comparisons before and after fine-tuning.
* Documentation describing the complete experimental procedure.

The completion of Stage 0 establishes the conversational foundation required for all subsequent stages of the project.
