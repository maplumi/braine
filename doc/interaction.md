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
