# Canonical Training Examples

The following examples illustrate the intended behavior for each Stage 0 capability.

These examples serve as canonical references when creating new supervised training data.

The wording of future examples may vary, but they should preserve the same conversational objective.

---

## Turn Taking

### Goal

Teach the model that a user message should be followed by exactly one assistant response.

### Example

```
User:
What color is the sky on a clear day?

Assistant:
Blue.<SPECIAL-0>
```

Expected behavior:

* Respond once.
* Do not continue the conversation.
* Terminate with `END_OF_TURN`.

---

## Knowledge Completion

### Goal

Teach the model to complete simple factual prompts using existing knowledge.

### Example

```
User:
What is the capital of France?

Assistant:
Paris.<SPECIAL-0>
```

Expected behavior:

* Produce a concise factual answer.
* No additional explanation.
* End the assistant turn.

---

## Local Context

### Goal

Teach the model to use information stated earlier in the conversation.

### Example

```
User:
I have a blue notebook.

Assistant:
Understood.<SPECIAL-0>

User:
What color is the notebook?

Assistant:
Blue.<SPECIAL-1>
```

Expected behavior:

* Recall information from earlier turns.
* Do not invent new information.
* Correctly terminate the conversation.

---

## Correction

### Goal

Teach the model to replace outdated information after a correction.

### Example

```
User:
My dog is called Bento.

Assistant:
Understood.<SPECIAL-0>

User:
Actually, Bento is my cat.

Assistant:
Bento is your cat.<SPECIAL-1>
```

Expected behavior:

* Ignore the outdated statement.
* Use the corrected information.

---

## Instruction Following

### Goal

Teach the model to satisfy simple formatting constraints.

### Example

```
User:
Answer with one word only.
What is the capital of France?

Assistant:
Paris.<SPECIAL-0>
```

Expected behavior:

* Respect the requested format.
* Do not add explanations.

---

## Uncertainty

### Goal

Teach the model to recognize when the conversation does not provide enough information.

### Example

```
User:
What is my mother's name?

Assistant:
I don't know.<SPECIAL-0>
```

Expected behavior:

* Do not fabricate an answer.
* Explicitly acknowledge insufficient information.

---

# Dataset Design Rules

Every Stage 0 example should satisfy the following principles:

1. Test a single conversational capability.
2. Be deterministic.
3. Have one unambiguous expected response.
4. Avoid reasoning whenever possible.
5. Avoid unnecessary world knowledge.
6. Use the shortest conversation capable of testing the target behavior.
7. Terminate every assistant response with the appropriate special token.

---

# Conversation Termination Rules

For every assistant response:

* Intermediate assistant turns terminate with:

```
<SPECIAL-0>
```

* The final assistant turn of a multi-turn conversation terminates with:

```
<SPECIAL-1>
```

* Single-turn conversations always terminate with:

```
<SPECIAL-0>
```

This convention is used consistently throughout Stage 0.
