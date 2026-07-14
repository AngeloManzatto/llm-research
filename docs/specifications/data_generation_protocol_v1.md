# Stage 0 SFT Data Generation Protocol v1.1

## Purpose

This document is the authoritative prompt specification for generating
Stage 0 SFT training examples. It defines what to generate, how to
format it, and what to avoid. Every generated row must pass
`validate_sft.py` before use.

---

## Row Schema

```json
{
  "id":       "{category}_{language}_sft_{NNNNN}",
  "category": "turn_taking | knowledge_completion | local_context | correction | instruction_following | uncertainty",
  "language": "en | pt",
  "stage":    "stage0",
  "messages": [
    {"role": "user",      "content": "..."},
    {"role": "assistant", "content": "..."},
    {"role": "user",      "content": "..."},
    {"role": "assistant", "content": "..."}
  ]
}
```

**Rules that apply to every row:**

- `messages` always starts with `user`, always ends with `assistant`
- The final `assistant` message is the **training target** — it must be
  the actual answer to the immediately preceding `user` message, never
  an acknowledgment
- Roles strictly alternate — no two consecutive same roles
- `content` is plain natural language — never contains `<EOS>`, `<BOS>`,
  or any `<SPECIAL-N>` string (these are injected at index level by the data loader)
- `system` role is not used in Stage 0
- `id` is zero-padded 5 digits, unique per file

---

## Conversation Style

- Simple, neutral, natural language
- Concise assistant responses (1–12 words where possible)
- No humor, sarcasm, roleplay, or emotional content
- No reasoning, coding, math beyond basic arithmetic, or opinion questions
- Both `en` and `pt` examples use natural phrasing — not literal translation

---

## Category Specifications

### turn_taking
**Purpose:** one user message → one assistant response, nothing more.
**Structure:** exactly 2 messages (user, then the assistant answer — this is the training target).
**Diversity axes:** factual questions, yes/no questions, greetings, imperatives,
comparisons, simple how-to, counting/sequencing, statement + acknowledgment.

```json
[
  {"role": "user",      "content": "What color is the sky?"},
  {"role": "assistant", "content": "Blue."}
]
```

---

### knowledge_completion
**Purpose:** recall a universally accepted fact.
**Structure:** exactly 2 messages (user question, then the assistant answer).
**Content rules:** answer must be deterministic and universally known
(capitals, basic science, simple arithmetic, units, common measurements).
Avoid obscure facts, opinion, or reasoning.

```json
[
  {"role": "user",      "content": "What is the capital of France?"},
  {"role": "assistant", "content": "Paris."}
]
```

---

### local_context
**Purpose:** use information stated earlier in the same conversation.
**Structure:** 4–8 messages, always ending in the assistant's answer
(min one prior user/assistant exchange + final question + final answer).
**Content rules:**
- Prior user messages introduce a fact (color, name, quantity, location, etc.)
- Prior assistant turns acknowledge with a short phrase ("Understood.", "Got it.", etc.)
- The second-to-last user message asks about a fact stated in a prior user message
- The final assistant message answers that question using the earlier fact —
  this is the training target, never an acknowledgment
- The final question must be answerable from context alone — no world knowledge needed
- 4-message examples: introduce fact → acknowledge → ask about it → answer it
- 6/8-message examples: add distractor turns between introduction and question

```json
[
  {"role": "user",      "content": "I have a blue notebook."},
  {"role": "assistant", "content": "Understood."},
  {"role": "user",      "content": "What color is the notebook?"},
  {"role": "assistant", "content": "Blue."}
]
```

---

### correction
**Purpose:** replace a previously stated fact after a correction.
**Structure:** 4–8 messages, always ending in an assistant message that
acknowledges the corrected fact (this is the training target).
**Content rules:**
- A prior user message states a fact
- A prior assistant turn acknowledges it
- A later user message corrects the previously stated fact
- The final assistant message acknowledges the correction specifically
  (e.g. "Got it, Bento is your cat."), not a generic "Got it."
- Direct corrections: "Actually, X is Y."
- Indirect corrections (4-turn): user says "I was wrong about that" →
  assistant asks what's correct → user states the corrected fact →
  assistant acknowledges it
- 6/8-message examples: add an unrelated distractor turn between original
  statement and correction

```json
[
  {"role": "user",      "content": "My dog is called Bento."},
  {"role": "assistant", "content": "Got it."},
  {"role": "user",      "content": "Actually, Bento is my cat."},
  {"role": "assistant", "content": "Got it, Bento is your cat."}
]
```

---

### instruction_following
**Purpose:** comply with an explicit formatting or response constraint.
**Structure:** 2–6 messages, always ending in the assistant's constrained answer.
**Content rules:**
- Single-turn (2 messages): the instruction and the question appear in the same
  user message; the assistant reply obeys the constraint
- Multi-turn (4 messages): a prior user message sets a persistent instruction
  ("From now on, answer in one word."), assistant acknowledges, next user message
  is a plain question, final assistant message answers it under that constraint
- Instruction types: one-word answers, yes/no only, list N items, uppercase/lowercase,
  specific opening word, repeat/echo, count to N

```json
[
  {"role": "user",      "content": "Answer with one word only. What is the capital of France?"},
  {"role": "assistant", "content": "Paris."}
]
```

---

### uncertainty
**Purpose:** acknowledge that the question cannot be answered from available context.
**Structure:** 2–4 messages, always ending in the assistant's uncertainty
acknowledgment (this is the training target — never a fabricated answer).
**Content rules:**
- Single-turn (2 messages): the question is unanswerable by design; the
  assistant states it can't answer
- 4-message: a prior exchange establishes context about entity A;
  the next question asks about a different entity B (not in context);
  the final assistant message states it doesn't have that information
- Unanswerability types:
  - Personal/private facts ("What is my mother's name?")
  - False presuppositions ("What is the capital of the moon?")
  - Unknowable future/present ("What number am I thinking of?")
  - Wrong entity ("My dog is called Rex." → "What is my cat's name?")

```json
[
  {"role": "user",      "content": "What is my mother's name?"},
  {"role": "assistant", "content": "I don't have that information."}
]
```

## Conversation Length and Context Budget

The model has a maximum context length of 1024 tokens. Dataset length is
therefore controlled using the project's production tokenizer after role
markers and end-of-turn tokens have been added.

Generated conversations should span multiple length bands:

* Short: approximately 20–60 serialized tokens
* Medium: approximately 60–150 serialized tokens
* Long: approximately 150–300 serialized tokens
* Extended: approximately 300–550 serialized tokens

Most examples should remain below 550 serialized tokens. No accepted Stage 0
example may exceed 768 serialized tokens.

Conversation length should arise from meaningful content, additional turns,
contextual details, corrections, distractors, or persistent instructions.
Generators must not add irrelevant filler solely to reach a target length.

Longer examples should occur primarily in `local_context`, `correction`, and
multi-turn `instruction_following`. Shorter examples should remain common in
`turn_taking`, `knowledge_completion`, and `uncertainty`.

Word counts may guide generation, but acceptance and auditing must use the
actual serialized token count produced by the project's tokenizer.

---

## Dataset Targets

| Category              | Total | EN   | PT   |
|-----------------------|------:|-----:|-----:|
| turn_taking           | 3000  | 1500 | 1500 |
| knowledge_completion  | 3000  | 1500 | 1500 |
| local_context         | 3000  | 1500 | 1500 |
| correction            | 3000  | 1500 | 1500 |
| instruction_following | 2000  | 1000 | 1000 |
| uncertainty           | 1000  |  500 |  500 |
| **Total**             | **15000** | **7500** | **7500** |

---

## Turn Depth Distribution

For multi-turn categories, use approximately (counts include the final
assistant target message):

| Category              | 4-msg | 6-msg | 8-msg |
|-----------------------|------:|------:|------:|
| local_context         |  70%  |  20%  |  10%  |
| correction            |  70%  |  20%  |  10%  |
| instruction_following | 90% 2-msg, 10% 4-msg | — | — |
| uncertainty           | 80% 2-msg, 20% 4-msg | — | — |

---

## Contamination Boundary

This dataset and `core_0001.jsonl` (the benchmark) are generated
independently. Avoid reusing the same specific phrasings, names, or
entities that appear in the benchmark. The structural patterns
(templates) are necessarily similar — that is acceptable.

---

## Generation Method

Examples are produced using slot-based composition:
sentence frames with swappable filler pools (names, colors, objects,
numbers, cities, etc.) combined programmatically to ensure surface
diversity. Every generated file must be validated with `validate_sft.py`
before use. No file is accepted with validation errors.

---



## Changelog

**v1.1** — Fixed a structural bug: rows previously ended on a `user`
message with no corresponding assistant answer, so the actual response
to the final question was never present as a training target. All
categories now end with the assistant's real answer to the final user
message. Corresponding code change (out of scope for this doc): loss
masking in `messages_to_tokens` must compute loss only on the **final**
assistant turn, not on every assistant turn in the sequence — otherwise
intermediate acknowledgment turns ("Got it.", "Understood.") get equal
training weight to the actual answer, which is what caused the model to
collapse onto acknowledgment tokens.