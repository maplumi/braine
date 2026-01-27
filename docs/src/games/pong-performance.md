# Pong performance notes (why it can look "stuck" at low hit-rate)

This repo contains **two Pong-like tasks**:

- `pong-demo` (core CLI experiment): a 1D catch task with dense shaping + bootstrap.
- Daemon/Web Pong: a continuous 2D-ish paddle/ball sim with binned sensors and 3 actions (up/down/stay).

If you observe a low hit-rate (e.g. ~11%), the most common causes are control cadence, encoding, and reward timing.

## Likely causes

### 1) Trial cadence too slow for the physics
The sim updates continuously, but actions are only applied once per `Trial ms`.
If `Trial ms` is large (e.g. 500ms), the paddle may not be able to correct fast enough even with a good policy.

**Experiment:** try `Trial ms` in the 40–120ms range and compare hit-rate.

### 2) Reward is dominated by sparse hit/miss events
The sim emits +/-1 reward only at left-wall events (hit/miss), plus a small correctness shaping term.
Depending on dynamics, the agent may not see enough informative reward to learn a stable policy.

**Experiment:** add a small dense shaping term proportional to `-(ball_y - paddle_y).abs()` while the ball is visible.

### 3) Binning may be too coarse
Ball/paddle state is discretized into 8 bins. If bin boundaries are too coarse relative to paddle size/speed, the policy can be ambiguous.

**Experiment:** increase bins (e.g. 12 or 16) and/or encode `dy` directly (bucketed).

### 4) Alias between contexts (meaning key too compressed)
If the meaning context key collapses too many states, the learned action ranking can be noisy.

**Experiment:** ensure the meaning context key includes enough of `(ball_y, paddle_y, vy)` to separate regimes.

## A minimal debugging checklist

- Lower `Trial ms` (web default is relatively high); observe if the ceiling rises.
- Lower `Ball speed` and/or increase `Paddle speed` to make the task controllable.
- Check if exploration is too low (stuck) or too high (never stabilizes).
- Confirm that the reward signal is applied as neuromodulator and that learning writes are enabled (not in eval mode).

## Next engineering steps (if we want Pong to be a reliable capability test)

- Add an explicit `dy_bucket` sensor (state-conditional action learning becomes simpler).
- Add optional distractor ball sensors (capability checklist: ignore distractors).
- Add a reversal/axis-flip regime to test rapid adaptation.
