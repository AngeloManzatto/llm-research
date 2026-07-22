"""
Created on Tue Jul 14 07:21:23 2026

@author: Angelo Antonio Manzatto
"""

"""
Prompt templates for LLM-based Stage 0 data generation.

Derived directly from:
  - stage0_definition_v1.0.md (the 5 goals, non-goals)
  - data_generation_protocol_v2.1.md (row schema, category specs,
    conversation style, turn depth distribution, length budget)

Nothing here is invented fresh — every rule below traces back to a
decision already made and tested earlier in this project. The point of
switching to LLM generation was naturalness/diversity (Section 3 of the
generation spec explicitly lists "template-like wording" and "slot
substitution" as things to MINIMIZE) — so the prompt asks for genuinely
varied scenarios, but the structural rules (schema, alternation, turn
depth, no bare acknowledgments) are unchanged from the template
generator, because those rules were never the problem.
"""

SYSTEM_PROMPT = """You generate synthetic training conversations for Stage 0 of a \
bilingual (English/Portuguese) language model's supervised fine-tuning.

STAGE 0's PURPOSE (do not violate this):
Stage 0 teaches conversational MECHANICS, not knowledge, reasoning, or style. \
A model that completes Stage 0 should be no more knowledgeable or eloquent than \
before — just capable of bounded, role-consistent, context-aware dialogue. \
Do NOT write content that requires reasoning, multi-step problem solving, \
coding, opinions, or specialized domain knowledge. Keep facts genuinely simple \
and universally known when a category calls for facts at all.

ROW SCHEMA (every example you produce must satisfy this exactly):
- A JSON object: {"messages": [...]}. You will be told the category and \
language for each request; do not include category/language/id yourself — \
the caller adds those.
- "messages" is a list of {"role": "user"|"assistant", "content": "..."}.
- Starts with "user", ends with "assistant". Roles strictly alternate — \
no two consecutive same roles.
- The FINAL message is always a real assistant answer to the immediately \
preceding user message — never a bare acknowledgment, never a question.
- "content" is plain natural language. NEVER include the literal strings \
<EOS>, <BOS>, <PAD>, or anything shaped like <SPECIAL-N> — these are \
injected by the training pipeline, not part of the text.

CONVERSATION STYLE:
- Natural language, concise, no filler sentences that only exist to pad length.
- AVOID bare acknowledgments as intermediate assistant turns ("Got it.", \
"Understood.", "Okay.", "Entendi.", "Certo."). Since every assistant turn \
is used as a training target (not just the final one), an intermediate \
turn should restate or confirm the user's actual content — e.g. "Noted — \
your notebook is blue." instead of just "Got it." A small amount of plain \
acknowledgment is fine; it must not be your only pattern.
- Maximize genuine diversity: vary names, occupations, ages, relationships, \
locations, objects, scenarios, sentence openings, and emotional register \
across examples. Prefer realistic everyday situations — work, school, \
travel, shopping, calendars, hobbies, family, pets, cooking, technology, \
health logistics, entertainment, home organization — over generic or \
repeated templates. No two examples in a batch should share an opening \
sentence, a name, or a scenario.
- Also vary the ANSWER'S FORM, not just its content — this is a separate \
axis from scenario diversity above. Within a batch, mix bare-word answers \
("Paris."), short phrases ("The capital's Paris."), and full sentences \
("The capital of France is Paris.") — all equally correct and equally \
concise for what the category calls for, but structurally different. A \
batch where every answer is exactly one word (or exactly one sentence \
length) teaches a rigid response template just as surely as repeating the \
same phrasing would, even though the words themselves differ each time.
- Portuguese examples must be natural Brazilian Portuguese phrasing, not \
literal translations of English sentence structure.

OUTPUT FORMAT:
Return ONLY a JSON array of example objects, nothing else — no markdown \
fences, no preamble, no explanation. Each element: {"messages": [...]}."""


CATEGORY_INSTRUCTIONS = {
    "turn_taking": """Category: turn_taking.
Teaches bounded, role-consistent response to a single turn — nothing more.
Structure: exactly 2 messages (one user turn, one assistant answer).
Diversity axes to rotate across the batch: factual questions, yes/no \
questions, greetings, imperatives, comparisons, simple how-to, counting/\
sequencing, statement + confirmation.
Vary answer form across the batch too — some one-word answers, some short \
phrases, some full sentences — matched naturally to what each question \
calls for, not the same shape every time.
For yes/no questions specifically: the correct answer must be "No" (a \
false premise to deny) for roughly HALF of them, not just occasionally. \
A model trained on yes/no data skewed toward "Yes" learns a real bias \
toward agreeing with whatever is stated, including false premises — this \
was confirmed directly: a trained model affirmed "Yes, humans have \
feathers" and "Yes, a rock floats on water" rather than denying them. \
Include deliberately false, even absurd premises requiring a clear denial \
("Is the moon made of cheese?", "Do fish have fur?"), not only obviously \
true ones.""",

    "knowledge_completion": """Category: knowledge_completion.
Teaches falling back to general knowledge when there is no conversational \
state to draw on (dialogue state is empty).
Structure: exactly 2 messages.
The answer must be a deterministic, universally known fact — capitals, \
basic arithmetic, common units, basic science. Avoid obscure facts, \
opinion, or anything requiring reasoning.
Do not default to a single bare word for every answer in the batch — mix \
in some short phrases and full sentences too (e.g. "Paris." / "It's \
Paris." / "The capital of France is Paris." should all appear across the \
batch, not just the first form repeated every time).""",

    "local_context": """Category: local_context.
Teaches maintaining dialogue state across turns.
Structure: {turn_count} messages. A user message states a fact (color, \
name, quantity, location, ownership, schedule, etc.); the assistant \
confirms it by RESTATING the fact (not a bare acknowledgment); {distractor_note}\
a final user question asks about that earlier fact; the final assistant \
message answers using ONLY the earlier stated fact — no world knowledge \
needed, and the question must not be answerable without it.""",

    "correction": """Category: correction.
Teaches updating dialogue state when new information supersedes old.
Structure: {turn_count} messages. A user message states a fact; the \
assistant confirms it by restating it; {distractor_note}a later user \
message corrects that fact (either directly — "Actually, X is Y" — or \
indirectly — "I was wrong about that" then stating the correction); the \
final assistant message must explicitly reflect the CORRECTED fact, not \
a generic acknowledgment and not the original value.""",

    "instruction_following": """Category: instruction_following.
Teaches obeying an explicit constraint, including when it must persist \
across turns.
Structure: {turn_count} messages. {instruction_shape}
Constraint types to rotate across the batch: one-word answers, yes/no \
only, listing a specific number of items, uppercase/lowercase, exact \
word echo, counting a sequence, a word limit, answering in exactly one \
sentence. The assistant's final answer must actually satisfy the stated \
constraint — verify this yourself before including the example.""",

    "uncertainty": """Category: uncertainty.
Teaches declining to invent information rather than confabulating.
Structure: {turn_count} messages. {uncertainty_shape}
Unanswerability types to rotate across the batch: personal/private facts \
neither party could know ("What is my mother's name?"), false \
presuppositions ("What is the capital of the moon?"), unknowable future/\
present states ("What number am I thinking of?"), or wrong-entity \
questions (context establishes fact about entity A, question asks about \
unrelated entity B). The final assistant message must be an honest \
non-answer — never a fabricated specific claim.""",
}


def build_category_prompt(category: str, language: str, n: int, turn_count: int = 2) -> str:
    """Build the per-request user prompt for one category/language/turn-depth batch."""
    lang_name = "English" if language == "en" else "Brazilian Portuguese"

    if category in ("local_context", "correction"):
        distractor_note = (
            "no distractor turns; " if turn_count == 4 else
            "one unrelated distractor turn between the fact and the question/correction; " if turn_count == 6 else
            "two unrelated distractor turns between the fact and the question/correction; "
        )
        instructions = CATEGORY_INSTRUCTIONS[category].format(
            turn_count=turn_count, distractor_note=distractor_note
        )
    elif category == "instruction_following":
        if turn_count == 2:
            shape = "The instruction and the question appear in the SAME user message."
        else:
            shape = (
                "A user message sets a persistent instruction (e.g. \"From now on, "
                "answer every question with one word only.\"); the assistant "
                "confirms the constraint (restating it, not a bare \"Okay.\"); a "
                "later user message asks a plain question; the final assistant "
                "message must still honor the constraint set earlier."
            )
        instructions = CATEGORY_INSTRUCTIONS[category].format(
            turn_count=turn_count, instruction_shape=shape
        )
    elif category == "uncertainty":
        if turn_count == 2:
            shape = "The question is unanswerable by design from the first turn."
        else:
            shape = (
                "A prior exchange establishes a fact about entity A; the final "
                "question asks about a different, unrelated entity B not "
                "covered by that context."
            )
        instructions = CATEGORY_INSTRUCTIONS[category].format(
            turn_count=turn_count, uncertainty_shape=shape
        )
    else:
        instructions = CATEGORY_INSTRUCTIONS[category]

    return (
        f"{instructions}\n\n"
        f"Language: {lang_name} (language code: \"{language}\").\n"
        f"Generate exactly {n} examples for this category/language/structure "
        f"combination. Every example must be genuinely distinct from every "
        f"other one in this batch — different names, different scenarios, "
        f"different sentence openings. Return the JSON array now."
    )