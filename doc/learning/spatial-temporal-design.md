# Spatial awareness + temporal structure (design note)

This note defines a *minimal, test-first* path to spatial awareness and temporal structure without changing the core substrate principles.
It is intentionally scoped to documentation and test definitions (no implementation yet).

---

## Goals (what we mean by “spatial awareness”)

1) **Spatial classification** (easy, dense reward)
- Input: continuous position $(x,y)$ (and optionally velocity).
- Output: discrete action (e.g., left/right or quadrant).
- Reward: immediate correctness.

2) **Spatial tracking / prediction** (harder, temporal)
- Input: $(x,y)$ over time (optionally with velocity cues).
- Output: action that depends on *history* (predict next position, intercept, etc).
- Reward: delayed and/or sparse.

We start with (1) because it isolates *representation + mapping* before temporal credit assignment.

---

## Option A: factorized population codes (recommended)

### Principle
Encode each continuous axis independently as a **population code** (a set of overlapping “bumps”).
The full observation is the *union* of all active bumps across axes.

This scales well:
- $d$ dimensions with $k$ bumps per axis → **$d \cdot k$ sensors**.
- Avoids $k^d$ combinatorial blow-up from a full grid.

### Concrete encoding spec (v1)
For each axis value $v \in [-1, +1]$:
- Choose $k$ evenly spaced centers: $c_i = -1 + 2\, i/(k-1)$ for $i=0..k-1$.
- Compute unnormalized bump activations:

$$
 a_i = \exp\left(-\frac{(v-c_i)^2}{2\sigma^2}\right)
$$

Recommended defaults:
- $k \in \{8, 16, 24\}$ (start with **16**)
- $\sigma = 2/(k-1)$ (roughly “two neighbors overlap”)

Normalize *per axis* so total energy stays stable:

$$
 \hat a_i = \frac{a_i}{\sum_j a_j + \epsilon}
$$

Then emit stimuli:
- X axis: `pos_x_00..pos_x_15` with strengths $\hat a_i$
- Y axis: `pos_y_00..pos_y_15` with strengths $\hat a_i$

Notes:
- This matches current system usage: multiple `apply_stimulus(Stimulus { name, strength })` calls per step.
- Keep names stable; do not embed raw floats in symbol names.

### Determinism requirements
- The encoder must be **pure and deterministic** given $(x,y)$ and config $(k,\sigma)$.
- If we later add random features (e.g., random Fourier features), the seed must come from the brain/config and be recorded in experiment logs.

---

## Scaling to higher dimensions

Adding an axis should be a **mechanical extension**:
- Add `pos_z_00..pos_z_15` for Z, etc.
- Keep the same normalization per axis.

What changes as dimensionality increases:
- **Sample complexity rises**: you need more experience to cover the space.
- **Aliasing risk rises** if sensors are too coarse (too small $k$).

Mitigations (in order of simplicity):
1) Increase $k$ modestly (16 → 24).
2) Keep tasks factorized early (rules that depend mostly on one axis).
3) Only later add limited cross-terms (e.g., a small set of conjunction sensors) if needed.

---

## Interaction model alignment (core principles stay unchanged)

This approach does not require new learning rules.
It only changes the *stimulus set* we present each step.

- Substrate remains local, sparse, continuous-time.
- Reward remains a scalar neuromodulator in $[-1,+1]$.
- Meaning/causal memory continues to record (stimuli, action, reward) and can learn conditional action-value.

---

## Minimal test battery (before implementation)

These are intended to become assays/experiments later.
They should be runnable with fixed seeds and produce a small metrics report.

### S1 — Encoder sanity
**Test:** for random $(x,y)$ samples:
- per-axis sums are ~1.0
- strengths are in [0,1]
- continuity: small $\Delta x$ → small change in sensor distribution (e.g., cosine similarity stays high)

**Pass criterion:** no NaNs; stable normalization; similarity degrades smoothly.

### S2 — SpotXY (2 actions)
**Task:** choose action based on sign of $x$.
- Action: `left` if $x<0$, else `right`.
- Reward: +1 for correct, -1 for incorrect.

**Pass criterion:** rolling accuracy (last 200) exceeds 0.85 within a reasonable step budget.

### S3 — SpotXY (4 actions)
**Task:** quadrant classification.
- Actions: NW/NE/SW/SE.

**Pass criterion:** rolling accuracy exceeds 0.70 (harder) and improves over baseline.

### S4 — SpotXY-Reversal
**Task:** after mastery, flip the rule (e.g., left/right swapped or quadrants rotated).

**Metrics:** time-to-recover rolling accuracy; compare first vs second reversal (savings effect).

### T1 — One-step prediction (discrete)
**Task:** discretize next-position into bins and predict the next bin.
- Inputs: current position and velocity (or provide previous position to infer velocity).
- Reward: delayed by one step (like Sequence game).

**Pass criterion:** prediction accuracy beats chance; improves with training.

---

## Metrics to log (so results are comparable)

- Seed(s): brain seed + encoder config.
- Task config: k, sigma, action set, reward schedule.
- Rolling accuracy (windowed) and lifetime accuracy.
- Flip markers for regime changes.
- Steps-to-recover after flip.

---

## Known risks / failure modes (watch for these)

1) **Energy drift**: without per-axis normalization, total stimulus energy varies with position and destabilizes learning.
2) **Too-sharp bumps**: if $\sigma$ is too small, adjacent positions look unrelated → no generalization.
3) **Too-wide bumps**: if $\sigma$ is too large, many positions look the same → aliasing.
4) **Context overwrite**: reversals or multi-tasking can overwrite meaning; prefer explicit context keys when needed (as already done for Spot Reversal).

---

## Proposed next step (still doc/test-first)

1) Add a short section to the experiment log template for “encoder config”.
2) Add an assay spec for S2 (2-action SpotXY) with fixed seed and printed metrics.
3) Only after those are stable, implement the minimal SpotXY experiment.
