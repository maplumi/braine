# Visualizer games (braine_viz)

This document describes each interactive environment in the `braine_viz` macroquad visualizer: how to run it, what it measures, and what to look for.

## Run
- Start the visualizer: `cargo run -p braine_viz`
- Pick a game using the top-row buttons.
- Pick **Human** or **Braine** control mode.

## Global controls / behavior
- **Human mode**: you directly control actions (varies by game).
- **Braine mode**: the brain selects actions; occasional exploration is injected to prevent lock-in.
- Persistence hotkeys:
  - `S` queues a save of the current brain image (`.bbi`).
  - `L` queues a load (replaces the current in-memory brain).
- Autosave: enabled by default (every ~2s). Configure via env/args documented in [brain-image.md](../architecture/brain-image.md).

## Logging and “what to look at”
- Metrics log file: `braine_viz_metrics.log`
- Most games periodically emit `FLIP_MARKER ...` events when the task regime flips.
- A simple way to read adaptation is: after a flip, how quickly does reward/accuracy recover?

---

## Pong
**How to run:** select **Pong**.

**Human controls:** Left/Right via arrows or `A`/`D`.

**Task:** keep the paddle under a falling ball. The environment includes a decoy/distractor ball.

**Measures**
- Closed-loop sensorimotor control and stabilization into a reliable policy.
- Distractor resistance (decoy sensor stream vs real ball stream).
- Switching/adaptation cost when the task regime flips.
- (If enabled in this build) “child brain” sandboxing: training a child on variants without corrupting the parent.

**Regime flips:** the reward mapping/context changes periodically; `FLIP_MARKER` is logged.

---

## Bandit
**How to run:** select **Bandit**.

**Human controls:** choose arm `A` or `B` (shown in the UI; usually via simple key selection in the existing Bandit implementation).

**Task:** a 2-armed bandit where the better arm changes over time.

**Measures**
- Simple value learning and unlearning under drifting reward.
- Switching/adaptation cost after flips.
- Whether the brain forms a stable preference under stationary reward.

**Regime flips:** which arm is good changes periodically; logged with `FLIP_MARKER`.

---

## Forage
**How to run:** select **Forage**.

**Human controls:** movement via arrows or `WASD`.

**Task:** move an agent in a 2D field toward the currently “good” target and away from the “bad” one. A cue determines which color is rewarded.

**Measures**
- Context-conditioned approach/avoid learning.
- Credit assignment from sparse outcomes (reward occurs only on contact).
- Switching/adaptation after cue flips.

**Regime flips:** which color is good flips every N outcomes; logged with `FLIP_MARKER`.

---

## Whack
**How to run:** select **Whack**.

**Human controls:** press `1`/`2`/`3` (mapped to actions `A`/`B`/`C`).

**Task:** a target appears in one of 3 lanes. You must choose the correct action corresponding to the lane. The label→lane mapping flips.

**Measures**
- Fast stimulus→action mapping under time pressure.
- Habit formation vs flexibility (mapping flips force unlearning/relearning).
- Sensitivity to sparse penalties (wrong hits are only occasionally penalized).

**Regime flips:** the mapping reverses periodically; logged with `FLIP_MARKER`.

---

## Beacon
**How to run:** select **Beacon**.

**Human controls:** movement via arrows or `WASD`.

**Task:** navigate toward the current target beacon while avoiding a distractor beacon. The target beacon identity (blue vs yellow) flips.

**Measures**
- Simple navigation policy conditioned on a contextual “which beacon matters” regime.
- Distractor suppression (being close to the non-target beacon is mildly penalized).
- Switching/adaptation after the target identity flips.

**Regime flips:** target beacon identity flips after N hits; logged with `FLIP_MARKER`.

---

## Sequence
**How to run:** select **Seq**.

**Human controls:** `1`/`2`/`3` predict next token `A`/`B`/`C`.

**Task:** the environment emits a token sequence (A/B/C). Your action is interpreted as a prediction of the *next* token. Reward is delayed by one step: the previous prediction is scored when the next token arrives.

**Measures**
- Short-horizon temporal credit assignment (reward depends on the next timestep).
- Sequence/pattern memory (learning a transition structure rather than a static mapping).
- Switching/adaptation when the underlying pattern changes.

**Regime flips:** alternates between two fixed patterns; logged with `FLIP_MARKER`.

---

## Appendix: interpreting logs

The visualizer writes a line-oriented log to `braine_viz_metrics.log`. It includes two main kinds of entries:

### 1) Events
- Look for lines containing `EVENT`.
- Most importantly: `FLIP_MARKER ...` indicates the environment regime flipped (reward mapping, cue identity, pattern, etc).

Use `FLIP_MARKER` as a ground-truth timestamp for “adaptation” questions.

### 2) Snapshots
- Each frame/step emits a snapshot-like record with:
  - the active `test` name
  - the `mode` (Human/Braine)
  - `reward` for that step
  - compact HUD strings (which include “top actions”, context name, and sometimes causal summaries)

### Practical metrics you can eyeball quickly

**Flip recovery time (steps)**
- After a `FLIP_MARKER`, count how many steps until reward/accuracy returns near its pre-flip baseline.
- This approximates “switching cost” and contextual flexibility.

**Stability / habit strength**
- In stable regimes, check whether the selected action becomes consistent and reward becomes less noisy.

**Sequence memory (Sequence game)**
- Because reward is delayed by one step, watch whether the brain learns a consistent next-token prediction.
- If it collapses to a single token, you’ll see persistently negative reward after the novelty wears off.

If you want a more formalized version of these ideas, see [metrics.md](../development/metrics.md).
