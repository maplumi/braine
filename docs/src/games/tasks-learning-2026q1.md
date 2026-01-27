# Learning Improvements Task List (2026 Q1)

This document turns the learning review into concrete, testable engineering tasks.

## Priority 0: Fix Spot ceiling

### Task 0.1 — Stop continuous imprinting during Spot
- Change the daemon’s Spot/SpotReversal/Bandit stimulus application to avoid one-shot imprinting for these simple discrete cues.
- Approach: apply sensor input via `apply_stimulus_inference()` and explicitly record the context symbol via `note_compound_symbol()`.

**Acceptance**
- In Spot, accuracy reliably exceeds 0.9 after ~100–200 trials with default settings.
- `births_last_step` should not show continuous births every trial in Spot.

### Task 0.2 — Add a deterministic Spot regression test
- Add a unit test that simulates a Spot-like loop (two stimuli, two actions, +/- reward), and asserts learning >0.9.

## Priority 1: Make meaning contexts learnable

### Task 1.1 — Replay context should be trial-specific
- Modify Replay’s `stimulus_key` so meaning contexts aren’t collapsed to `replay::<dataset>`.

**Acceptance**
- Built-in replay Spot dataset converges to >0.9 accuracy quickly.

### Task 1.2 — Pong add `dy_bucket` context
- Add a compact, decision-relevant context feature (`dy_bucket`, approach/visible bits) to reduce context fragmentation.

## Priority 2: Behavior tests from the checklist

### Task 2.1 — Bandit drift/flip schedule
- Add a periodic flip or slow drift in arm probabilities.

### Task 2.2 — SpotReversal uncued mode
- Add a mode without `spot_rev_ctx` / `::rev` context so unlearning is required.

### Task 2.3 — SpotXY action space reduction
- Keep action space small (2 or 4 actions) and make “grid” a context label rather than an action label.

## Priority 3: Experts (child brains) positioning

### Task 3.1 — Keep experts optional and bounded
- Maintain experts behind an explicit toggle; do not require them for core association learning.
- Improve docs/tests to show what experts add beyond neurogenesis + imprinting.

