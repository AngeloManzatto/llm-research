# Stage 0 Metrics — Definitions, Measurement, Examples (v1.1)

Supersedes v1.0. Rewritten to match `metric.py` as actually implemented,
not a speculative superset of it. Four metrics exist; everything else
below is explicitly marked as deferred, and deferred means **no code
exists for it yet**, not "implemented but untested."

---

## Metric Contract

Every metric is a plain function, not a class:

```
metric(raw_answer: str, **kwargs) -> MetricResult
```

- `raw_answer` is the pre-normalization generated text (stop token still
  attached, e.g. `"Paris.<EOS>"`).
- Anything else a specific metric needs is pulled from `**kwargs` by
  name. Required keys are declared on the function itself via a
  `.requires` tuple (e.g. `expected_stop_token.requires =
  ("expected_token",)`), so `run_metric()` can validate up front with one
  consistent error, instead of each metric failing differently deep
  inside its own body.
- `MetricResult(passed: bool, details: dict)` — `passed` always means
  "the desired behavior happened" (no polarity flag, no exception to
  remember). `details` is metric-specific diagnostic data, populated
  only when there's something informative to say (e.g. empty on a clean
  pass).

There is no `version` field, no `description` field, no `output_type`
field, and no class hierarchy — v1.0 had all of these and nothing read
any of them. Cut for that reason, not oversight.

---

## The Four Implemented Metrics

### `expected_stop_token`

**Measures**: the response terminates on the correct stop token.

**Requires**: `expected_token`.

**How it works**: `raw_answer.rstrip().endswith(expected_token)`.

**Example**
- Pass: `"Blue.<EOS>"` with `expected_token="<EOS>"`.
- Fail: `"Blue."` (no stop token at all) or a response ending on a
  different token.

**Note on scope**: v1.0 split this into two separate gating metrics,
`no_stop` (ran to `max_length` with no stop token) and
`wrong_stop_token` (stopped, but on the wrong token) — on the reasoning
that they're different causes. They were collapsed back into one
metric during implementation: both are just the two ways
`expected_stop_token` comes back `False`, not two independent failures,
and scoring them as two separate gating checks double-counts one
underlying problem. The cause-level distinction (`no_stop` vs
`wrong_stop_token`) is not currently captured in `details` — if that
distinction becomes useful in practice, it belongs in `details`, not as
a second gating metric.

---

### `repetition`

**Measures**: the response does NOT degenerate into a repeated word or
phrase. (`passed=True` means clean — no repetition found.)

**Requires**: nothing beyond `raw_answer`.

**How it works**: searches every possible repeat-unit length, longest
first, for a phrase repeating immediately and consecutively. Threshold
is graduated by phrase length:
- 1-2 word phrases need **3+** consecutive repeats to flag (so normal
  English doubling — "no no thanks", "very very good", "I think I think
  that is right" — does not false-positive).
- 3+ word phrases flag on **2** consecutive repeats (a repeated clause
  is already a strong degeneration signal on its own).

No cap on phrase length — searches up to half the token count. An
earlier version capped n-gram length at 4, then 8; both were proven
insufficient against real failure examples (a 5-word and a 9-word
repeated clause, respectively, are structurally invisible to any fixed
cap regardless of threshold tuning).

**Example**
- Fail (`passed=False`): `"the woman saw it coming the woman saw it
  coming"` → `details={"phrase": "the woman saw it coming",
  "repetitions": 2, "ngram_size": 5}`
- Fail: `"I dont I dont I dont"` → `details={"phrase": "i dont",
  "repetitions": 3, "ngram_size": 2}`
- Pass: `"no no thanks."` → `details={}`
- Pass: `"Blue."` → `details={}`

**v1.0's `repetition_loop` had the opposite polarity** (`True` = bad).
Fixed during implementation to match every other metric's "`True` =
desired behavior" convention — `passed=True` here means no loop was
detected.

---

### `contains_expected`

**Measures**: whether `raw_answer` contains any one of a list of
acceptable strings (case-insensitive substring match).

**Requires**: `expected_any` (list of strings).

**How it works**: `any(exp.lower() in raw_answer.lower() for exp in
expected_any)`. `details={"matched": <which one matched>}` on pass.

This single function, parameterized differently, currently covers four
of v1.0's proposed category-specific metrics — not because the
distinction between them is meaningless, but because none of them have
yet needed logic beyond "does the answer contain this string":

| v1.0 proposed metric | Category | How `contains_expected` covers it |
|---|---|---|
| `exact_or_normalized_match` | knowledge_completion | `expected_any = [ground_truth]` |
| `matches_stated_fact` | local_context | `expected_any = [stated_value]` |
| `reflects_correction` | correction | `expected_any = [corrected_value]` |
| `contains_fabricated_answer` | uncertainty | `expected_any = [refusal_pattern, ...]` |

**Polarity note for uncertainty specifically**: v1.0's
`contains_fabricated_answer` had `True` mean *fabrication detected*
(bad) — the metric doc's own flagged polarity bug from earlier
discussion. Reusing `contains_expected` against `refusal_patterns`
sidesteps that entirely: `passed=True` means *a refusal pattern was
matched*, which — for this category — is the desired behavior. No
inversion needed; the reuse happens to land on the correct polarity
for free.

**What `contains_expected` cannot yet distinguish** (deferred, not
implemented — see below): *why* a match failed. `local_context`'s
`used_prior_knowledge_instead`, `correction`'s `used_stale_fact` /
`used_unrelated_fact`, and `uncertainty`'s `refusal_phrase_diversity`
are all sub-codes describing failure *causes*, not pass/fail gates.
None of them have any code behind them yet.

---

### `constraint_satisfied`

**Measures**: whether the response satisfies an explicit instruction
constraint.

**Requires**: `constraint_type`; `constraint_value` additionally
required only when `constraint_type == "exact_match"`.

**How it works**: strips any trailing special-token tag from
`raw_answer` first (otherwise e.g. `"BANANA.<EOS>"`'s casing or exact
comparison would be corrupted by the tag), then branches on
`constraint_type`:

| `constraint_type` | Check |
|---|---|
| `one_word` | exactly one whitespace-separated token |
| `yes_no` | text is exactly "yes"/"no" (en) or "sim"/"não" (pt) |
| `uppercase` | every alphabetic character is uppercase |
| `exact_match` | text equals `constraint_value` exactly (used for both word-echo and count-sequence constraints — both are the same check, so kept as one type rather than two) |

**Example**
- `one_word`, `"Paris.<EOS>"` → pass.
- `one_word`, `"The answer is 12.<EOS>"` → fail.
- `uppercase`, `"TOKYO.<EOS>"` → pass. `"banana.<EOS>"` → fail.
- `exact_match`, `constraint_value="banana"`, `"banana.<EOS>"` → pass
  (trailing punctuation tolerated).

**What this does not yet check**: constraint *persistence* — v1.0's
`constraint_satisfied_on_final_turn`, for rows where a constraint is
set on an earlier turn and must still hold several turns later.
Currently `constraint_satisfied` only checks the turn it's given; there
is no code distinguishing "followed once" from "followed and
maintained."

---

## Explicitly Deferred (no code exists)

These appeared as planned metrics in v1.0. None are implemented. Each
is deferred for a stated reason, not by oversight:

| Item | Why deferred |
|---|---|
| `role_leakage` | Maps to a real Stage 0 clause (role consistency), but zero observed instances in actual training/generation logs so far. Building a detector for an unobserved failure mode repeats the speculative-metric mistake this file's history is a record of avoiding. |
| `constraint_satisfied_on_final_turn` | Real gap, not urgent — single-turn compliance had to exist first, which it now does. |
| `used_prior_knowledge_instead` | Sub-code enrichment on top of an already-working `local_context` gate, not a prerequisite for one. |
| `used_stale_fact` / `used_unrelated_fact` | Same — enrichment on top of a working `correction` gate. |
| `refusal_phrase_diversity` | Non-gating diversity stat; also arguably a `report.py`-level aggregate (computed across many rows) rather than a per-row metric at all. |

**`wiki_talk_artifact` is not deferred — it's cut.** It tested
pretraining-corpus formatting leakage, which is not a clause of the
Stage 0 definition. Out of scope, not "not yet built."

---

## Summary Table

| Metric (as implemented) | Category | Requires |
|---|---|---|
| `expected_stop_token` | turn_taking (+ all, pooled) | `expected_token` |
| `repetition` | turn_taking (+ all, pooled) | — |
| `contains_expected` | knowledge_completion, local_context, correction, uncertainty | `expected_any` |
| `constraint_satisfied` | instruction_following | `constraint_type` (+ `constraint_value` for `exact_match`) |