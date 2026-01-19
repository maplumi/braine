# Stability–Plasticity Control Loop (Braine)

This document captures the **bounded adaptation** approach used in Braine’s core substrate and the 2026 update to align the implementation more closely with a control-theoretic framing:

- **Fast dynamics** (state update, bounded activity)
- **Medium plasticity** (local learning, gated by neuromodulation)
- **Slow homeostasis & governance** (boundedness guarantees and safe degradation)

The project invariant still holds:

- **Learning modifies state; inference uses state.**
- No backprop, no global losses.

## Background

A stable-yet-plastic substrate is fundamentally a control problem: the system must remain bounded and useful while continuing to change.

The guiding goal is **bounded adaptation**:

- bounded state (no runaway activity)
- bounded weights (no unbounded growth)
- reversible learning (new learning can be overwritten)
- retention under noise (attractors persist)
- graceful degradation (forgetting is controlled)

## What changes in this update

This update introduces a more explicit separation between:

- *local correlations that might matter later* (**eligibility**) and
- *whether learning is committed now* (**neuromodulator**).

This moves the substrate closer to a three-factor rule:

$$\Delta w_{ij} \propto m(t) \cdot e_{ij}(t)$$

Where $e_{ij}$ is a local eligibility trace, and $m(t)$ is the neuromodulator (reward/salience).

### Summary of behavioral impact

- The substrate continues to run dynamics every tick.
- **Eligibility traces update every tick** from local activity and phase alignment.
- **Weight updates are committed only when neuromodulation is present** (outside a small deadband).

This is designed to work with the daemon/game loop as-is:

- Games already provide reward pulses via `set_neuromodulator(reward)` and then `commit_observation()`.
- No changes are required to the *meaning* of game inputs/outputs.

## Fast loop: bounded dynamics

Braine uses leaky recurrent “wave” dynamics with:

- per-unit decay (leak)
- global inhibition (competition)
- noise (exploration)
- hard clamps on amplitude (saturation)

This provides the stability backbone (attractor-friendly but bounded).

## Medium loop: gated local plasticity

### Eligibility traces

Each synapse maintains an eligibility value (ephemeral, local state):

- decays slowly over time
- is driven by co-activity and phase alignment

Eligibility traces are **not a reward signal**. They only represent local credit to be potentially applied.

### Neuromodulator gating

The neuromodulator $m(t)$:

- is supplied by the Frame/game loop (typically reward in [-1, 1])
- gates whether eligibility is converted into actual weight updates
- allows negative values to suppress or reverse updates (punishment)

### Plasticity budget

When the neuromodulator triggers learning, updates are capped by a simple budget:

- maximum total $\sum |\Delta w|$ per step

This enforces boundedness under bursts of eligibility.

## Slow loop: homeostasis and governance

### Per-unit homeostasis

A slow bias adaptation nudges unit excitability so typical activity stays within a target band.

This reduces:

- sensitivity to hyperparameters
- saturation regimes
- silent-network regimes

### Stability monitors

The daemon can monitor cheap signals:

- max/mean activity
- weight norms / saturation proxies
- action entropy proxies

If unstable:

- temporarily reduce learning
- increase inhibition
- freeze plasticity

#### Learning monitor API

The core substrate exposes per-step learning monitors via:

- `Brain::learning_stats()` → `LearningStats`

These values are intended for dashboards/debugging (not as control signals by default):

- `plasticity_committed`: whether `abs(neuromod)` exceeded the deadband
- `plasticity_l1`: total $
	\sum |\Delta w|$ actually applied this step
- `plasticity_edges`: count of edges updated this step
- `plasticity_budget` / `plasticity_budget_used`: budget configuration and consumption
- `eligibility_l1`: total eligibility magnitude after update
- `homeostasis_bias_l1`: total bias change applied by homeostasis (0 if not run)

On the daemon, these are forwarded in the `brain_stats` snapshot as optional fields so
clients can visualize when learning is happening without changing the game contract.

## Persistence and compatibility

- Brain images remain loadable across versions.
- Eligibility traces are ephemeral and can safely reset on load.

## Game / input contract impact

No protocol-level changes are required for games.

- Existing games already produce rewards and context symbols.
- The learning mechanism becomes more robust to delayed reward.

If a game previously relied on *continuous* unmodulated Hebbian drift, it should instead:

- rely on eligibility accumulation during the trial
- rely on neuromodulator pulses at scoring time

This is closer to the intended RL-like loop.
