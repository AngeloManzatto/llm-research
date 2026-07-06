# Conversation Level 0 Blueprint v1.0

## Purpose

This document defines the structure of the **Conversation Level 0 Benchmark** and the **Stage 0 Supervised Fine-Tuning (SFT) dataset**.

Its purpose is to establish a precise and reproducible specification for every conversational capability introduced during Stage 0.

This blueprint does **not** define how examples are generated. Dataset generation is described separately in **Data Generation Protocol v1.0**.

---

# Objective

Conversation Level 0 teaches a pretrained next-token prediction (NTP) model the fundamental mechanics of participating in a conversation.

The objective is **not** to improve reasoning, planning, coding, or world knowledge.

Instead, the model should learn:

* when to answer;
* how to answer;
* how to preserve short conversational context;
* how to update information after corrections;
* how to recognize insufficient information;
* how to terminate responses correctly.

---

# Benchmark Size

The Level 0 benchmark contains **600 deterministic examples**.

## Language Distribution

| Language   | Examples |
| ---------- | -------: |
| English    |      300 |
| Portuguese |      300 |
| **Total**  |  **600** |

---

## Category Distribution

| Category              | English | Portuguese |   Total |
| --------------------- | ------: | ---------: | ------: |
| Turn Taking           |      50 |         50 |     100 |
| Knowledge Completion  |      50 |         50 |     100 |
| Local Context         |      50 |         50 |     100 |
| Correction            |      50 |         50 |     100 |
| Instruction Following |      50 |         50 |     100 |
| Uncertainty           |      50 |         50 |     100 |
| **Total**             | **300** |    **300** | **600** |

---

## Category Single / Multi Turn Distribution

| Category                  | Single-turn | Multi-turn | Rationale                                                                                                                                                                                                                   |
| ------------------------- | :---------: | :--------: | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Turn Taking**           |   **100%**  |   **0%**   | Tests the fundamental conversation mechanic: one user message should produce exactly one assistant response. Multi-turn examples do not introduce additional behavior for this capability.                                  |
| **Knowledge Completion**  |   **100%**  |   **0%**   | Measures factual recall from the pretrained model. By definition, the answer must not depend on previous conversational context.                                                                                            |
| **Local Context**         |    **0%**   |  **100%**  | Requires information introduced earlier in the conversation. A single-turn example cannot evaluate context retention.                                                                                                       |
| **Correction**            |    **0%**   |  **100%**  | Requires an earlier statement that is subsequently corrected. The model must replace outdated information with the corrected one.                                                                                           |
| **Instruction Following** |   **90%**   |   **10%**  | Most formatting or response constraints are naturally expressed in a single turn. A small fraction of multi-turn examples verifies that instructions remain active across one conversational turn.                          |
| **Uncertainty**           |   **80%**   |   **20%**  | Most uncertainty cases involve standalone questions with insufficient information. A smaller portion uses conversational context to verify that the model recognizes information is still missing instead of hallucinating. |

---

# General Design Rules

Every example must satisfy the following principles.

* Test exactly one conversational capability.
* Be deterministic.
* Have one expected answer.
* Be concise.
* Use natural language.
* Avoid ambiguity.
* Avoid reasoning.
* Avoid opinion-based questions.
* Avoid subjective answers.
* Be independently understandable.

---

# Difficulty

All examples in Level 0 are classified as **Easy**.

Examples should not require:

* mathematical reasoning;
* logical reasoning;
* multi-step inference;
* domain expertise;
* long-term memory.

---

# Conversation Length

The benchmark emphasizes short conversations.

Approximate distribution:

* Single-turn conversations: **80%**
* Multi-turn conversations: **20%**

Multi-turn conversations should not exceed four user/assistant exchanges.

---

# Response Length

Assistant responses should be concise.

Recommended response length:

* Minimum: one word.
* Maximum: approximately twelve words.

Long explanations should be avoided.

---

# World Knowledge

Knowledge examples should rely only on universally accepted facts.

Examples:

✓ Capital of France

✓ Color of grass

✓ Water freezes into ice

Avoid:

* obscure historical facts;
* niche scientific knowledge;
* trivia;
* culturally dependent answers.

---

# Conversation Style

All examples should use:

* simple language;
* neutral tone;
* natural conversations;
* realistic interactions.

Avoid humor, sarcasm, roleplay, and emotional manipulation.

---

# Conversation Protocol

Assistant responses terminate using explicit conversation tokens.

## END_OF_TURN

```text
<SPECIAL-0>
```

Marks the end of a single assistant turn.

Used for:

* every assistant response inside a conversation;
* the only assistant response in a single-turn conversation.

---

## END_OF_CONVERSATION

```text
<SPECIAL-1>
```

Marks the final assistant response of an explicitly multi-turn conversation.

Never appears in single-turn conversations.

---

# Canonical Capability Examples

The following examples define the expected structure of each conversational capability.

These examples serve as canonical references for future dataset generation.

---

# 1. Turn Taking

## Purpose

Teach the model that a user message should be followed by exactly one assistant response.

## Example

```text
User:
What color is the sky on a clear day?

Assistant:
Blue.<SPECIAL-0>
```

### Expected Behavior

* Produce exactly one response.
* Do not continue the conversation.
* Terminate correctly.

---

# 2. Knowledge Completion

## Purpose

Teach the model to answer simple factual questions using existing pretrained knowledge.

## Example

```text
User:
What is the capital of France?

Assistant:
Paris.<SPECIAL-0>
```

### Expected Behavior

* Produce a concise factual answer.
* Avoid unnecessary explanations.

---

# 3. Local Context

## Purpose

Teach the model to use information introduced earlier in the same conversation.

## Example

```text
User:
I have a blue notebook.

Assistant:
Understood.<SPECIAL-0>

User:
What color is the notebook?

Assistant:
Blue.<SPECIAL-1>
```

### Expected Behavior

* Recall previous conversational information.
* Avoid hallucinating new facts.

---

# 4. Correction

## Purpose

Teach the model to replace outdated information after a correction.

## Example

```text
User:
My dog is called Bento.

Assistant:
Understood.<SPECIAL-0>

User:
Actually, Bento is my cat.

Assistant:
Bento is your cat.<SPECIAL-1>
```

### Expected Behavior

* Discard outdated information.
* Use only the corrected statement.

---

# 5. Instruction Following

## Purpose

Teach the model to satisfy simple formatting constraints.

## Example

```text
User:
Answer with one word only.

What is the capital of France?

Assistant:
Paris.<SPECIAL-0>
```

### Expected Behavior

* Respect the requested format.
* Do not generate additional text.

---

# 6. Uncertainty

## Purpose

Teach the model to acknowledge when the conversation does not provide enough information.

## Example

```text
User:
What is my mother's name?

Assistant:
I don't know.<SPECIAL-0>
```

### Expected Behavior

* Do not invent an answer.
* Explicitly acknowledge insufficient information.

---

# Exclusions

The following are outside the scope of Conversation Level 0.

* Mathematical reasoning
* Logical puzzles
* Coding
* Tool use
* Long-context reasoning
* Planning
* Creative writing
* Multiple valid answers
* Subjective opinions
* Open-ended conversations
* Conversations requiring external tools

---

# Success Criteria

The benchmark is designed to measure whether a pretrained NTP model has acquired the basic mechanics of conversation.

A successful Stage 0 model should demonstrate improvements across all six conversational capabilities while preserving deterministic behavior.

---

# Relationship to Stage 0 SFT

This blueprint defines the target behavior for both:

* the deterministic benchmark;
* the supervised fine-tuning dataset.

The benchmark evaluates these behaviors.

The Stage 0 dataset teaches these behaviors.

Although both follow the same specification, they are generated independently to avoid benchmark contamination.

---

# Schema Mapping

This section maps every conversational behavior defined above onto the actual
`BenchmarkExample` fields used by the benchmark pipeline. Generation must
satisfy both the behavior described above and the field rules below.

## Multi-Turn Stop Token Rule (explicit)

This rule is only implicit in the canonical examples above. Stated directly:

* Every assistant turn **except the last** in an example ends with
  `END_OF_TURN` (`<SPECIAL-0>`).
* The **last** assistant turn in a **single-turn** example ends with
  `END_OF_TURN` (`<SPECIAL-0>`).
* The **last** assistant turn in a **multi-turn** example ends with
  `END_OF_CONVERSATION` (`<SPECIAL-1>`).

`END_OF_CONVERSATION` appears at most once per example, always on the final
assistant turn, never anywhere else.

## Field Definitions

Scoring dispatch (which algorithm is used to grade an answer) is controlled
entirely at the benchmark-manifest level via `scoring_metric`, applied
uniformly to every example in the run. There is no per-example scoring
override field — see **Scoring** below.

| Field                  | Rule |
| ----------------------- | ---- |
| `id`                    | `{category}_{language}_{NNN}`, zero-padded 3 digits, unique per file. Example: `local_context_en_014`. |
| `category`               | One of: `turn_taking`, `knowledge_completion`, `local_context`, `correction`, `instruction_following`, `uncertainty`. |
| `language`               | `en` or `pt`. No mixed-language examples in this pass. |
| `prompt`                 | Full conversation text up to and including the final `Assistant:` marker, using the `User:` / `Assistant:` template shown in the canonical examples. For multi-turn examples, all prior turns are included with their correct stop tokens already in place. |
| `expected_any`           | A list of the **minimal deterministic substring(s)** that must appear in the final answer for it to be considered correct — not the full expected sentence. Case is ignored at scoring time, so casing in the list does not matter. Prefer a single short token where possible (`["Paris"]`, `["blue"]`, `["cat"]`); use multiple entries only when more than one phrasing is genuinely acceptable (`["I don't know", "not sure", "no information"]`). Never include a string so short or common it could match unrelated text. |
| `expected_stop_token`    | `"END_OF_TURN"` for every single-turn example. `"END_OF_CONVERSATION"` for every multi-turn example. This refers only to the **final** assistant turn's token — non-final turns are not separately scored by this field. |

## Worked Example: Mapping a Canonical Case to a Row

Canonical `correction` example maps to:

```json
{
  "id": "correction_en_001",
  "category": "correction",
  "language": "en",
  "prompt": "User:\nMy dog is called Bento.\n\nAssistant:\nUnderstood.<SPECIAL-0>\n\nUser:\nActually, Bento is my cat.\n\nAssistant:\n",
  "expected_any": ["cat"],
  "expected_stop_token": "END_OF_CONVERSATION"
}
```

Notes on this mapping:

* `prompt` includes the first assistant turn already completed (with `<SPECIAL-0>`), since the benchmark is testing the **second** assistant turn's behavior. The prompt ends right after the final `Assistant:` marker, with nothing after it — that is what the model must complete.
* `expected_any` is `["cat"]`, not `["Bento is your cat"]` — the minimal distinguishing substring is sufficient and more robust to harmless phrasing variation (e.g. "Your cat." / "Bento is your cat." / "It's a cat." all correctly pass).
* `expected_stop_token` is `"END_OF_CONVERSATION"` because this is the final turn of a two-turn example.
* There is no `scoring` field on the row. Scoring algorithm selection happens once, at the benchmark manifest level (see **Scoring** below), and applies to every example in the file.

## What `expected_any` Should Never Contain

* Full sentences when a single word would do.
* Generic words likely to appear by accident (`"a"`, `"is"`, `"the"`, `"yes"` on its own for non-yes/no questions).
* Anything not actually present in the canonical correct answer.

---

# Scoring

The benchmark manifest's `scoring_metric` field specifies the deterministic algorithm used to determine whether the model successfully answered an example. This is set once per benchmark file and applies uniformly to every example it contains — there is no per-example scoring override.

Scoring algorithms are implemented by the benchmark framework and are version-controlled. Future benchmark versions may introduce additional scoring methods without modifying previous datasets.

## Current Scoring Methods

### `contains_expected`

The generated assistant response is considered correct if it contains at least one entry from the `expected_any` list after normalization.

Normalization currently includes:

* case-insensitive comparison;
* whitespace normalization;
* removal of benchmark stop tokens.

The comparison is performed using substring matching.

Formally,

```text
PASS

if

∃ expected ∈ expected_any

such that

expected is contained in normalized_answer
```

Otherwise,

```text
FAIL
```

## Examples

### Example 1

```json
{
  "expected_any": ["Paris"]
}
```

(scored under a benchmark file whose manifest sets `"scoring_metric": "contains_expected"`)

Generated answer:

```text
Paris.
```

PASS

Generated answer:

```text
The capital is Paris.
```

PASS

Generated answer:

```text
PARIS
```

PASS (case-insensitive)

Generated answer:

```text
London
```

FAIL

Generated answer:

```text
I believe it is London.
```

FAIL

### Example 2

```json
{
  "expected_any": [
    "I don't know",
    "not sure",
    "cannot determine"
  ]
}
```

Generated answer:

```text
I don't know.
```

PASS

Generated answer:

```text
I'm not sure.
```

PASS

Generated answer:

```text
There isn't enough information to determine that.
```

FAIL

Although semantically similar, this exact phrase is not included in `expected_any`. If this phrasing should be accepted, it must be explicitly added to the list. The benchmark intentionally favors deterministic reproducibility over semantic similarity.

## Design Philosophy

Scoring algorithms are intentionally deterministic. The benchmark does not attempt to infer the intended meaning of an answer. Only explicitly defined acceptance rules determine whether an example passes.

This design avoids dependence on:

* human judgment;
* external language models;
* embedding similarity;
* probabilistic semantic matching.

Future scoring algorithms may extend the benchmark, but existing scoring methods remain unchanged to preserve reproducibility.