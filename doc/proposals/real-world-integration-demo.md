# Proposal: Real-world integration demo (Bridge task)

This proposal defines a **single, minimal “real integration” task** that demonstrates how Braine can be embedded in an external system.

The intent is not to be a full robotics stack; it’s a reproducible, end-to-end demo that:
- streams stimuli in
- streams actions out
- applies bounded scalar reward
- runs continuously

## Why this demo

Toy games are good for iteration, but a real integration example makes it obvious:
- what the interface contract is
- what minimum signals are required for learning
- what metrics to watch during adaptation

## Contract (problem-set style)

### Stimulus symbols
A small fixed set of named inputs, e.g.:
- `bridge_sensor_a`
- `bridge_sensor_b`
- `bridge_sensor_c`

These can represent anything: a thresholded telemetry signal, a user feedback bit, or a binned sensor reading.

### Action symbols
A small fixed set of discrete actions, e.g.:
- `bridge_action_0`
- `bridge_action_1`
- `bridge_action_2`

### Reward / neuromodulation
A scalar `reward` in a bounded range (recommended `[-1, +1]`).

Critical guideline: reward should be **interpretable and timely**. If reward is delayed, the integration should either:
- emit intermediate shaping rewards, or
- explicitly tag the reward to the trial/action that caused it.

### Temporal structure
Define a trial cadence in wall-clock time (e.g. 50–200ms) so the system is stable regardless of render FPS.

Per trial:
1) external host sends current stimuli
2) Braine selects an action
3) external host applies the action
4) external host sends reward (can be immediate or delayed)

## Minimal daemon API surface

The daemon already speaks newline-delimited JSON over TCP. The Bridge task can be implemented with a minimal set of additional messages:

- **Set bridge inputs**: provide a list of `{ name, strength }` stimuli for the next decision.
- **Read chosen action**: return the action name + score.
- **Apply reward**: provide scalar reward/neuromodulation and indicate whether this ends a trial.

A key goal is to keep this API stable and small.

## Observability

The dashboard for this demo should show:
- trials, last_100_rate (or reward rate), learning milestones
- unit/connection counts, pruning/births, saturation

See:
- [doc/development/dashboard.md](../development/dashboard.md)
- [doc/games/metrics-per-game.md](../games/metrics-per-game.md)

## Suggested demo host (reference implementation)

Provide a tiny host program (can live under `crates/core/src/bin/` or `scripts/`) that:
- connects to the daemon
- runs a simple closed loop (e.g., stabilize a drifting scalar with discrete up/down actions)
- logs metrics and shows adaptation over time

This gives a concrete “real world” loop without needing external hardware.
