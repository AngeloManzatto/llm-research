# Stage 0 (Conversation Foundation) — Definition v1.0

## Definition

Stage 0 is the supervised training stage whose objective is to teach the fundamental mechanics of dialogue through conversational state management.

At the end of Stage 0, the model should:

1. Interpret alternating user and assistant messages as a single evolving interaction, bounded by clear turn structure — each assistant response terminates appropriately and stays in role.
2. Maintain a coherent dialogue state.
3. Update that state when new information supersedes previous information.
4. Recognize the absence of information when appropriate.
5. Generate assistant responses consistent with the current dialogue state.

Stage 0 intentionally excludes reasoning, planning, tool use, alignment,
and domain specialization — it is exclusively the foundation for
coherent multi-turn conversation, not a demonstration of it.

---

## The Six Categories

| Category | Teaches | Measured by |
|---|---|---|
| **turn_taking** | Bounded, role-consistent response to a single turn | Response terminates at the correct stop token; no user-turn content generated; not truncated (`too_long`); no runaway repetition |
| **knowledge_completion** | State is empty → fall back to general knowledge correctly | Answer matches a fixed, deterministic ground truth |
| **local_context** | State maintenance across turns | Final answer matches the fact as stated earlier in *this* conversation, not a hallucinated or generic alternative |
| **correction** | State update — new info supersedes old | Final answer reflects the *corrected* fact, not the original one |
| **instruction_following** | An explicit constraint becomes part of the state and must persist | Response satisfies every stated constraint (format, casing, wording, length) |
| **uncertainty** | Absence recognition — no confabulation | Response contains an explicit non-answer/refusal-to-invent, not a fabricated fact |

Every category is checked automatically per row against ground truth or a structural rule — no human judgment call, no LLM-as-judge required for pass/fail. Each measurement is a deterministic yes/no.
