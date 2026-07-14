# Stage 0 Completion Criteria v1.1

Addendum to `stage0_definition_v1.0.md`. Supersedes v1.0. §1 and Goal 1
in §3 are updated to match the actual metric set in `metric.py`
(`stage0_metrics_v1.1.md`) — v1.0 referenced `wiki_talk_artifact`
(cut) and `no_stop`/`wrong_stop_token` as separate columns (collapsed
into one `expected_stop_token` metric). Thresholds in §2 and the
mapping in §4 are unchanged.

---

## 1. Cross-Tabulation Requirement

Diagnostics (`expected_stop_token`, `repetition`) must be reported
**per category**, not only pooled across all 600 examples. Pooled
diagnostics conflate mechanical failure (clause 1 — turn structure)
with content failure (each category's own clause), making the two
indistinguishable and any single category's `passed` rate unreliable as
a signal until this is separated.

Required report shape, per checkpoint:

| Category | passed | expected_stop_token | repetition |
|---|---|---|---|
| turn_taking | | | |
| knowledge_completion | | | |
| local_context | | | |
| correction | | | |
| instruction_following | | | |
| uncertainty | | | |

(v1.0 had four diagnostic columns: `repetition`, `too_long`,
`expected_stop_token`, `wiki_talk_artifact`. `too_long` was folded into
`expected_stop_token` — a response that runs on forever without
stopping already fails `expected_stop_token`, and Stage 0's definition
has no clause about brevity, so a standalone length ceiling would
measure something out of scope. `wiki_talk_artifact` is cut per
`stage0_metrics_v1.1.md`.)

---

## 2. Completion Thresholds

Unchanged from v1.0. Stage 0 is complete only when **all** of the
following hold simultaneously:

| Category | Threshold |
|---|---|
| turn_taking | ≥ 95% |
| knowledge_completion | ≥ 90% |
| local_context | ≥ 80% |
| correction | ≥ 75% |
| instruction_following | ≥ 80% |
| uncertainty | ≥ 75% |

Plus two binding conditions:

- **Floor**: no category may score below 60%, regardless of the average
  across categories. A high blended score does not offset one category
  collapsing — each goal is falsifiable independently.
- **Stability**: thresholds must hold on **2 consecutive evaluation
  checkpoints**, not a single snapshot. A single passing checkpoint is
  not sufficient evidence the behavior has settled rather than
  transiently appeared.

---

## 3. Per-Goal Measurement Definitions

### Goal 1 — Turn structure (bounded, role-consistent response)

**Primary signal**: `turn_taking` category `passed` rate.

**Supporting signal**: the two mechanical diagnostics, computed **per
category** (per §1) and then aggregated across *all* categories, since
turn structure is a cross-cutting precondition, not specific to one
category's content:

```
turn_structure_score = 1 − (fraction of ALL examples, across every
                            category, with either:
                            expected_stop_token = false
                            OR repetition = false)
```

(Recall `repetition`'s polarity: `repetition = false` means a
degenerate loop *was* found — see `stage0_metrics_v1.1.md`. `passed =
True` is always "good" for both metrics, so this formula is checking
for either one coming back `False`.)

A category can only be considered to have a meaningful `passed` rate
once its own `turn_structure_score` is high — a low `passed` rate
alongside a low `turn_structure_score` in the same category indicates
the failure is mechanical (Goal 1), not a failure of that category's
own goal.

**Not yet measurable**: role-consistency specifically (the model
leaving its own role, e.g. writing the next user turn) has no
implemented metric (`role_leakage` — deferred, see
`stage0_metrics_v1.1.md`). `turn_structure_score` as defined above
covers termination and repetition, not role leakage. Until that metric
exists, a clean `turn_structure_score` does not fully certify Goal 1 —
it certifies the two failure modes that are currently instrumented.

### Goal 2 — Maintain dialogue state

**Signal**: `local_context` category `passed` rate, via
`contains_expected` with `expected_any = [stated_value]`.

### Goal 3 — Update dialogue state (correction)

**Signal**: `correction` category `passed` rate, via `contains_expected`
with `expected_any = [corrected_value]`.

### Goal 4 — Recognize absence of information

**Signal**: `uncertainty` category `passed` rate, via `contains_expected`
with `expected_any = [refusal_pattern, ...]`.

### Goal 5 — Generate responses consistent with current state

**Primary signal**: `instruction_following` category `passed` rate, via
`constraint_satisfied`.

**Baseline signal**: `knowledge_completion` category `passed` rate, via
`contains_expected` with `expected_any = [ground_truth]` — the trivial
case where dialogue state is empty. This is the sanity floor for Goal
5: if this is failing, no other Goal 5 measurement can be trusted.

```
goal_5_score = min(instruction_following_passed_rate,
                    knowledge_completion_passed_rate)
```

Using `min` rather than an average is deliberate — the same
independent-falsifiability principle as §2's floor: strong
`instruction_following` performance does not compensate for a broken
empty-state baseline, and vice versa.

**Not yet measurable**: constraint *persistence* specifically (a
constraint set on an earlier turn still holding several turns later) —
`constraint_satisfied_on_final_turn` is deferred. Current
`instruction_following` `passed` rate reflects one-shot compliance,
which is a real but partial measurement of Goal 5's "consistent with
current state" claim for multi-turn rows.

---

## 4. Summary Mapping

| Goal | Category(ies) | Metric | Fully measured? |
|---|---|---|---|
| 1. Turn structure | turn_taking (+ all, for mechanics) | `passed` rate + `turn_structure_score` | Partial — role leakage not yet instrumented |
| 2. Maintain state | local_context | `passed` rate (`contains_expected`) | Yes |
| 3. Update state | correction | `passed` rate (`contains_expected`) | Yes |
| 4. Recognize absence | uncertainty | `passed` rate (`contains_expected`) | Yes |
| 5. Consistent response | instruction_following, knowledge_completion | `min(passed rates)` (`constraint_satisfied`, `contains_expected`) | Partial — persistence not yet instrumented |