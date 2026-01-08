# Experiments

This repo deliberately keeps **core substrate code** separate from **experiments/assays**:

- Core “braine” library (what the brain *is*): `src/lib.rs` + `src/{substrate,causality,supervisor,prng}.rs`
- Experiments + demos (how we *probe it*): `src/experiments/*`
- CLI entrypoints (wires commands to experiments): `src/main.rs`

There are currently **no Rust unit tests**; instead the project uses reproducible **assays** and interactive-ish **demos** you run from the CLI.

## Current experiments

### `assays`
Run: `cargo run -- assays`

Objective: small, repeatable probes for:

- Association learning (food→approach, threat→avoid)
- Partial/noisy recall (reduced stimulus strength)
- Switching cost (how many steps to adapt after stimulus flips)
- Forgetting/retention (idle decay, then re-test)
- Policy comparison on a novel stimulus: habit-only vs meaning-guided stability

Output: a compact report with accuracies and a few stability/energy proxies.

### `pong-demo` (env_pong)
Run: `cargo run -- pong-demo`

This is a **1D catch task** (not full 2D Pong):

- A “ball” falls vertically from the top toward a fixed paddle row.
- The paddle moves horizontally: `left`, `right`, or `stay`.
- When the ball reaches the paddle row, it’s a **hit** if the paddle is within ±1 cell, otherwise a **miss**.

#### Objective
Maximize hits (and cumulative reward) by learning the state-conditional policy:

- If ball is left of paddle → move left
- If ball is right of paddle → move right
- If aligned → stay

#### Stimuli (what the brain receives)
The environment emits discrete sensor symbols each step:

- `pong_ctx_far_left`: ball is ≥4 cells left of paddle
- `pong_ctx_left`: ball is 1–3 cells left of paddle
- `pong_ctx_aligned`: ball and paddle share the same x
- `pong_ctx_right`: ball is 1–3 cells right of paddle
- `pong_ctx_far_right`: ball is ≥4 cells right of paddle
- `pong_ball_falling`: indicates the ball is currently falling (always true in the current task, but kept explicit)

Additionally, the demo records an **action-tagged context symbol**:

- `pair::<ctx>::<action>` (example: `pair::pong_ctx_left::left`)

This `pair::...` symbol is not an external sensor; it is used so the causal/meaning memory can learn *state-conditional* action value (“in this context, that action predicts reward”).

#### Reward schedule

- Dense shaping while the ball is falling:
	- small penalty proportional to distance to the ball
	- small bonus for being near-aligned
	- small bonus if the chosen action moves in the correct direction
- Sparse outcome reward at the catch event:
	- hit: `+0.7`
	- miss: `-0.7`

The final scalar reward is clamped to `[-1, 1]` and applied as the neuromodulator.

#### Bootstrap + self-retire

- First ~600 steps use a hand-coded reflex so the brain sees correct pairings early.
- After that it switches to meaning-guided control with some exploration.
- The demo can “self-retire” when the rolling hit-rate exceeds a configured threshold.

### `autonomy-demo`
Run: `cargo run -- autonomy-demo`

Objective: show the “child brain” loop:

- parent encounters unknown stimulus
- spawns a child sandbox with higher plasticity/exploration
- consolidates best child’s structure back into the parent
- verifies the parent now responds correctly to the new stimulus

## Log template
Copy/paste for each experiment:

- Date:
- Branch/commit:
- Hypothesis:
- Change:
- Determinism:
	- Brain seed:
	- Environment seed (if different):
	- Any other randomness sources:
- Protocol (stimuli/actions/reward schedule):
- Encoder config (if continuous/spatial):
	- Encoding family: (e.g., factorized population code)
	- Axes: (x, y, z, ...)
	- Bumps per axis (k):
	- Sigma (\sigma):
	- Normalization: (per-axis sum=1, global clamp, etc)
	- Stimulus naming scheme:
- Metrics collected:
- Result summary:
- Next step:

## Current backlog (minimal, high value)

1) Ensure sensor/action groups never overlap
- Reason: overlapping allocations create accidental coupling and muddy metrics.

2) Add deterministic seeding everywhere
- Reason: makes regressions and improvements measurable.

3) Add a small “task battery”
- 2–5 tasks only: association, partial recall, switching, forgetting.

4) Edge profile pass
- Move toward fixed-size buffers, avoid per-step allocations.

5) Persistence
- Save/restore couplings + unit states in a compact format.
