# Interaction model (inputs, outputs, basic tasks)

This project is a **closed-loop agent substrate**, not a text model.

## What counts as input

At the boundary, the system accepts **events**:

- **Stimulus**: `Stimulus { name, strength }`
  - Example: `("vision_food", 1.0)`
  - Meaning: excite a named sensor group.

- **Neuromodulator**: a scalar reward/salience in `[-1, +1]`
  - Example: `+0.7` = positive reward, `-0.4` = negative reward
  - Meaning: scales local plasticity and is also recorded into causal memory.

- **Time**: `brain.step()` advances the internal dynamics.

Internally, `brain.commit_observation()` records the current event set
(stimulus + action + reward) into the causality/meaning subsystem.

## What counts as output

The minimal output is an **action selection**:

- `brain.select_action(...) -> (action_name, score)`
  - Example: `( "avoid", +0.64 )`

Additionally, the system can provide a crude explanation-like hint:

- `brain.meaning_hint(stimulus_name) -> Option<(action_name, score)>`
  - Interpreted as: "given what I have seen, this action tends to cause reward".

## CLI usage

- Demo loop (prints periodic diagnostics):
  - `cargo run`

- Capability assays (repeatable small task battery):
  - `cargo run -- assays`

- Child-brain sandbox learning + consolidation demo:
  - `cargo run -- spawn-demo`

- Help:
  - `cargo run -- --help`

## Basic tasks (minimal autonomy indicators)

These tasks are intentionally tiny and cheap to run on an edge device.

### Task A: One-shot association
**Goal:** after one exposure, create a weak but usable link.

Protocol:
- Present a novel stimulus `S` once.
- Reward a target action `A`.
- Present `S` again and check whether `A` is selected.

Expected behavior:
- 1 exposure: weak preference.
- Repetition: stronger preference and higher action stability.

### Task B: Recall from partial cue
**Goal:** respond correctly even if the stimulus is weak/noisy.

Protocol:
- Train with `strength=1.0`.
- Test with `strength=0.3..0.5`.

Expected behavior:
- Correct action still emerges from the attractor.

### Task C: Switching / adaptation
**Goal:** rapidly switch when the environment changes.

Protocol:
- Drive stimulus A for K steps.
- Switch to stimulus B.

Expected behavior:
- steps-to-switch decreases with learning.

### Task D: Forgetting (relevance)
**Goal:** unused structure decays/prunes.

Protocol:
- Train associations.
- Run idle steps.

Expected behavior:
- accuracy decays over time.
- connection count decreases (pruning frees capacity).

### Task E: Child-brain autonomy pattern (sandboxing)
**Goal:** learn a new signal without destabilizing identity.

Protocol:
- Spawn child with higher plasticity.
- Train child on a novel stimulus.
- Consolidate only strong learned links back into parent.

Expected behavior:
- parent behavior stays stable during child exploration.
- parent gains new association after consolidation.

---

## Daemon games: input/output encoding

The daemon (`brained`) runs small “games” that convert environment state into a set of
`Stimulus { name, strength }` events, then scores an action and applies reward.

General rules:

- **Stimuli are named sensors** created at daemon init (or ensured on load). A “sensor” can have
  multiple units, but the daemon generally uses either:
  - **one-hot** stimulation (`strength=1.0`) for discrete state, or
  - **graded/population-coded** stimulation (`strength in [0,1]`) for continuous state.
- **Actions are named action groups**. The daemon asks the substrate to select one of the
  allowed action names for the current game.
- **Meaning conditioning key**: some games also attach a *symbol tag* (via
  `note_compound_symbol([stimulus_key])`) to provide a more specific context for causal/meaning
  credit assignment than the base stimulus name.

### Dev note: sensor channels vs task symbols

The substrate supports two stimulus application paths:

- `Brain::apply_stimulus(...)`: injects input **and** records the stimulus name as a symbol in the
  current observation (and may update stimulus telemetry/imprinting).
- `Brain::apply_stimulus_inference(...)`: injects input **only** (no symbol recording).

Use them consistently:

- **Sensor channels (high-cardinality, per-tick features)** should usually use
  `apply_stimulus_inference` so they don’t flood the bounded causal symbol set.
  Examples: bin/population-coded position channels, wall sensors, velocity sign bits.
- **Task symbols / context keys (low-cardinality, credit-assignment anchors)** should be recorded as
  symbols (via `apply_stimulus` and/or `note_*` symbol tagging) when you want the causality/meaning
  subsystem to attribute reward to a compact named key.
  Examples: `spotxy_bin_{...}`, `pong_b08_bx03_by05_py04_vxp_vyn`, regime/context tags.

For shared games, prefer the helper wrappers in `braine_games::brain_io` to make this explicit:

- `apply_sensor_channel(...)` (inference-only)
- `apply_task_symbol(...)` (records a symbol)

Below are the current encodings for each daemon game kind.

### `spot`

- **Actions:** `left`, `right`
- **Stimuli (one-hot, strength=1.0):** exactly one of
  - `spot_left` (spot appears on left)
  - `spot_right` (spot appears on right)

### `bandit`

- **Actions:** `left`, `right`
- **Stimuli:** constant context sensor
  - `bandit` (strength=1.0 every tick)

### `spot_reversal`

- **Actions:** `left`, `right`
- **Stimuli (one-hot, strength=1.0):** exactly one of
  - `spot_left`
  - `spot_right`
- **Extra context sensor (one-hot, strength=1.0):**
  - `spot_rev_ctx` is applied only while the reversal regime is active.
- **Meaning conditioning key (symbol tag):** while reversal is active, the daemon also conditions
  the context key as `spot_left::rev` or `spot_right::rev`.

### `spotxy`

- **Actions:**
  - Binary mode: `left`, `right`
  - Grid mode: `spotxy_cell_{n:02}_{ix:02}_{iy:02}` for all $0 \le ix,iy < n$
- **Stimuli (population-coded):**
  - `pos_x_00..15` (graded strengths)
  - `pos_y_00..15` (graded strengths)
- **Meaning conditioning key (symbol tag):**
  - Binary mode: `spotxy_xbin_{k:02}`
  - Grid mode: `spotxy_bin_{n:02}_{ix:02}_{iy:02}`

---

## Daemon games: Pong (3 actions)

The daemon (`brained`) includes a minimal `pong` task with **3 actions**: `up`, `down`, `stay`.

### Discrete-bin sensors (current)
The current implementation uses a small, discrete encoding:

- One-hot bins for position:
  - `pong_ball_x_00..07` (ball x in [0,1])
  - `pong_ball_y_00..07` (ball y in [-1,1])
  - `pong_paddle_y_00..07` (paddle y in [-1,1])
- Velocity sign bits:
  - `pong_vx_pos` / `pong_vx_neg`
  - `pong_vy_pos` / `pong_vy_neg`

For meaning/credit-assignment conditioning, the daemon also tags each timestep with a
symbol key like:

`pong_b08_bx03_by05_py04_vxp_vyn`

This lets the causal/meaning system learn context-conditioned effects without requiring
the symbol itself to be a sensor.

### Tunable parameters (UI)

The visualizer exposes a few runtime tunables:

- `paddle_speed` (units/sec in normalized y range [-1,1])
- `paddle_half_height` (half height in normalized y units)
- `ball_speed` (multiplier applied to ball velocity per step)

### Path to continuous encodings (planned)
No protocol change is needed to move beyond bins.
The substrate already supports real-valued `Stimulus { strength }`, so “continuous” can be expressed as:

- **Graded bins**: stimulate the nearest 2–3 bins with weights (e.g. linear or Gaussian falloff).
- **Population coding**: use fixed centers and compute normalized activations (as in SpotXY).

Both approaches preserve the same action space while increasing state resolution.
