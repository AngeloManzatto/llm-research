# Stage 0 Metrics Contract v2.0

Closes out the metrics redesign discussion. This document is the
reference going forward — the same role stage0_definition_v1.0.md and
stage0_completion_criteria_v1.1.md play for the rest of the project.

## The core correction this contract makes

Earlier metric design implicitly required `knowledge_completion` and
`turn_taking`'s factual questions to be answered *correctly*, not just
*coherently*. That was never Stage 0's actual goal — the original
definition only ever asked for mechanics (bounded, role-consistent,
context-aware dialogue), not knowledge. This contract corrects that:
Stage 0 measures whether the model can produce a well-structured,
semantically appropriate response — a toddler learning to *say* the
capital of Brazil correctly-shaped, not learning geography.

This correction is backed by established hallucination-research
terminology, found independently of our own reasoning and matching it
closely:

- **Factual hallucination** — content contradicts real-world truth,
  independent of anything stated in the conversation. This is what
  `knowledge_completion` and `turn_taking`'s factual questions test.
  The field's own literature states LLMs "prioritize coherence and
  fluency over factual correctness" as an inherent probabilistic
  property of the model class, not a defect specific to a small model.
- **Faithfulness hallucination** — content contradicts what was
  actually *given* in the conversation. This is what `local_context`,
  `correction`, `instruction_following`, and `uncertainty` test. The
  literature treats this as the more serious failure, since the
  correct answer was never missing — it just wasn't used.

Standard practice tracks both axes **separately**, never discarding
factual-correctness measurement entirely — it demotes it to a visible,
non-gating diagnostic where it isn't the right gate. This contract
follows that practice exactly.

## Final metric table

| # | Metric | Checks | Applies to (gates `passed`) | Applies to (diagnostic only) | Status |
|---|---|---|---|---|---|
| 1 | `expected_stop_token` | Model produces the turn-boundary token | All categories | — | Built, tested. ~99-100% across recent clean runs — Goal 1 (bounded turns) is essentially solved. |
| 2 | `repetition` | No degenerate looping phrase within one answer | All categories | — | Built, tested. Independently verified against real collapse data (a row can pass #1 while failing this — proven, not assumed). |
| 3 | `coherence` | Real words (not invented gibberish), not near-empty | `knowledge_completion`, `turn_taking`'s factual subset | Could extend to all categories as a floor — open decision, not yet made | Built and tested this session (pyspellchecker, EN+PT). Documented, known blind spot: a short real dictionary word used as a degenerate answer (e.g. PT "ta") can still slip through. |
| 4 | *(numeric type-match — unbuilt)* | Does a "how many" question get a number-shaped answer | Same factual categories, alongside `coherence` | — | Designed, scoped deliberately to numbers only. Place/person/color type-matching explicitly shelved (see below). |
| 5 | `contains_expected` | Does the answer contain the value the conversation actually stated | `local_context`, `correction`, `instruction_following`, `uncertainty` | `knowledge_completion`, `turn_taking`'s factual subset (kept visible, never gates there) | Built long ago. Needs the manifest split + a new "diagnostic-only" bucket in evaluator.py (not yet built — see Open Engineering Work). |
| 6 | `constraint_satisfied` | Did the model obey the stated constraint (uppercase, one-word, reverse, exact match) | `instruction_following` only | — | Metric built and tested (including `reverse_word`). Ground truth data isn't ready — rows still lack `constraint_type`/`constraint_value` tags. |

## Category assignments — the faithfulness / factual split

**Factual (open question, no context given, pure recall from pretraining):**
- `knowledge_completion` — unambiguous, every row is open recall.
- `turn_taking`'s factual-question subset only — NOT the whole category.
  Greetings, imperatives, and yes/no social-convention questions don't
  cleanly fit either bucket and are deliberately left ungated by this
  split for now. Revisiting this needs a row-level tag (factual vs.
  social/procedural), not a category-wide decision — real authoring
  work, explicitly deferred.

**Faithfulness (the needed information was given, in the conversation
or the instruction itself):**
- `local_context`, `correction` — the stated fact was given; using it
  wrong isn't a knowledge gap, it's not using what's present.
- `instruction_following` — the constraint itself is given directly in
  the prompt, not recalled from pretraining. Same logic as
  local_context, applied to a rule instead of a fact.
- `uncertainty` — no stated value to be faithful to, but the correct
  behavior (decline) is fully determined by what's present or absent
  in the conversation, not by outside knowledge. Belongs with this
  group for that reason.

## Uncertainty redefinition

The category's original four subtypes are reduced to three. **False
presupposition** ("How many legs does a unicorn have?") is discarded
entirely — confirmed via a real benchmark row (`uncertainty_en_027`,
model answered "Uns 48." instead of declining) that this subtype
secretly required world knowledge (knowing unicorns are fictional),
smuggling a factual-hallucination dependency into a category built
specifically to avoid one. There is no way to reliably hand-pick
"safe enough" false premises — what counts as safely universal
knowledge depends on this specific model's actual pretraining
coverage, which drifts with every dataset change and can't be verified
in advance. Retained subtypes, all requiring zero world knowledge:

1. Personal/private facts neither party could know.
2. Unknowable present/future states.
3. Wrong-entity questions (context establishes fact about A, question
   asks about unrelated B).

Existing benchmark/training rows using the discarded subtype should be
replaced with one of the three retained ones when convenient — not
urgent, same zero-cost hand-authoring process used for the
local_context/correction benchmark fixes earlier this project.

## Explicitly shelved — real ideas, not needed at Level 0 right now

- **Distinct-n diversity metric** — measures whether the model gives
  varied answers across different questions (catches the "I don't have
  that information" verbatim-every-time collapse). Real, established
  (Li et al. 2016), but tests style/variety, not mechanics or
  coherence — a later-stage concern once Level 0's structural goals are
  solid.
- **Full answer-type matching** (place, person, color — beyond
  numbers) — needs a gazetteer or real NER tooling (spaCy identified as
  the concrete future option). Genuine infrastructure, not a cheap
  addition like the numeric case.
- **Four-label uncertainty classifier** (SUPPORTED_ANSWER /
  VALID_UNCERTAINTY / UNSUPPORTED_ANSWER / INVALID_REFUSAL) — the
  INVALID_REFUSAL pattern it would catch showed up for real this
  session, but as decline-bleed *into* `knowledge_completion`
  (historical facts called "fictional"), not as a problem with
  `uncertainty`'s own scoring. Real value, wrong target — belongs to a
  future cross-category interference investigation, not this contract.
- **BLiMP-style grammatical minimal-pair testing** — the strongest
  literature-backed candidate for a genuine "Level 1" of evaluation.
  Tests real grammatical competence (subject-verb agreement, word
  order) via likelihood comparison between a correct/incorrect sentence
  pair — a fundamentally different evaluation mechanism than our
  generate-and-score pipeline, not a metric-function addition. The
  BabyLM Challenge (real, ongoing, ACL/CoNLL-affiliated research on
  child-scale language model training) is the direct precedent;
  their own findings show small models can get within ~3% of human
  performance on this despite orders-of-magnitude less data than
  typical LLMs — external validation that structure-over-facts is a
  sound bet for a model this size.

## Open engineering work (not yet done — next coding session, after the file reorganization)

1. Add a `diagnostic_only` list per category in the manifest schema, and
   teach `evaluator.py` to compute those metrics without letting them
   gate `passed`. Currently only two buckets exist (the one scoring
   metric, and `always_computed` — both gate). This is required before
   `contains_expected` can be safely demoted for the factual categories
   without losing visibility into it.
2. Update `category_scoring_metric` in the manifest: `knowledge_completion`
   → `coherence`. Decide `turn_taking` later (mixed category, deferred).
3. Install `pyspellchecker`; add `coherence` to `always_computed` (or to
   the factual categories specifically — open decision on whether it
   should be universal).
4. Build the numeric type-match metric (regex-based, no new
   dependencies).
5. Finish the long-deferred `instruction_following` constraint-type
   migration so `constraint_satisfied` has real data to check.
