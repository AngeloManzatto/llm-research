# Stage 0 SFT Data Generation Protocol v1.0

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
    {"role": "user",      "content": "..."}
  ]
}
```

**Rules that apply to every row:**

- `messages` always starts with `user`, always ends with `user`
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
**Structure:** exactly 1 message (user only — the assistant turn is what the model learns).
**Diversity axes:** factual questions, yes/no questions, greetings, imperatives,
comparisons, simple how-to, counting/sequencing, statement + acknowledgment.

```json
{"role": "user", "content": "What color is the sky?"}
```

---

### knowledge_completion
**Purpose:** recall a universally accepted fact.
**Structure:** exactly 1 message.
**Content rules:** answer must be deterministic and universally known
(capitals, basic science, simple arithmetic, units, common measurements).
Avoid obscure facts, opinion, or reasoning.

```json
{"role": "user", "content": "What is the capital of France?"}
```

---

### local_context
**Purpose:** use information stated earlier in the same conversation.
**Structure:** 3–7 messages (min one prior user/assistant exchange + final question).
**Content rules:**
- Prior user messages introduce a fact (color, name, quantity, location, etc.)
- Prior assistant turns acknowledge with a short phrase ("Understood.", "Got it.", etc.)
- The final user message asks about a fact stated in a prior user message
- The final question must be answerable from context alone — no world knowledge needed
- 3-message examples: introduce fact → acknowledge → ask about it
- 5/7-message examples: add distractor turns between introduction and question

```json
[
  {"role": "user",      "content": "I have a blue notebook."},
  {"role": "assistant", "content": "Understood."},
  {"role": "user",      "content": "What color is the notebook?"}
]
```

---

### correction
**Purpose:** replace a previously stated fact after a correction.
**Structure:** 3–7 messages.
**Content rules:**
- A prior user message states a fact
- A prior assistant turn acknowledges it
- The final user message corrects the previously stated fact
- Direct corrections: "Actually, X is Y."
- Indirect corrections (3-turn): user says "I was wrong about that" →
  assistant asks what's correct → user states the corrected fact
- 5/7-message examples: add an unrelated distractor turn between original
  statement and correction

```json
[
  {"role": "user",      "content": "My dog is called Bento."},
  {"role": "assistant", "content": "Got it."},
  {"role": "user",      "content": "Actually, Bento is my cat."}
]
```

---

### instruction_following
**Purpose:** comply with an explicit formatting or response constraint.
**Structure:** 1–5 messages.
**Content rules:**
- Single-turn: the instruction and the question appear in the same user message
- Multi-turn (3 messages): a prior user message sets a persistent instruction
  ("From now on, answer in one word."), assistant acknowledges, final user message
  is a plain question to be answered under that constraint
- Instruction types: one-word answers, yes/no only, list N items, uppercase/lowercase,
  specific opening word, repeat/echo, count to N

```json
{"role": "user", "content": "Answer with one word only. What is the capital of France?"}
```

---

### uncertainty
**Purpose:** acknowledge that the question cannot be answered from available context.
**Structure:** 1–3 messages.
**Content rules:**
- Single-turn: the question is unanswerable by design
- 3-message: a prior exchange establishes context about entity A;
  the final question asks about a different entity B (not in context)
- Unanswerability types:
  - Personal/private facts ("What is my mother's name?")
  - False presuppositions ("What is the capital of the moon?")
  - Unknowable future/present ("What number am I thinking of?")
  - Wrong entity ("My dog is called Rex." → "What is my cat's name?")

```json
{"role": "user", "content": "What is my mother's name?"}
```

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

For multi-turn categories, use approximately:

| Category              | 3-msg | 5-msg | 7-msg |
|-----------------------|------:|------:|------:|
| local_context         |  70%  |  20%  |  10%  |
| correction            |  70%  |  20%  |  10%  |
| instruction_following | 90% single-turn, 10% 3-msg | — | — |
| uncertainty           | 80% single-turn, 20% 3-msg | — | — |

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