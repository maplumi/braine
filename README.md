# braine

A research prototype for a **brain-like** cognitive substrate based on **sparse local dynamics** (no matrices, no backprop, no transformers).

## Framing (vs familiar baselines)

This project behaves like a **continuous-time recurrent substrate** with **local plasticity** (Hebbian-ish) and a scalar **neuromodulator** used as reward/salience. That puts it closest to these reference frames:

### 1) Neuromodulated Hebbian RL (3-factor learning)
- **Overlap:** local co-activity updates (Hebb) scaled by a reward-like signal (neuromodulator).
- **Difference:** no explicit $Q(s,a)$ table, no policy gradient, and no backprop-through-time; “credit assignment” is structural/local.
- **Where you’ll see it:** action-group reinforcement in the interactive demos (Pong/Bandit).

### 2) Reservoir computing / Liquid State Machine (LSM)
- **Overlap:** rich recurrent dynamics + simple readout-like action selection; computation comes from the substrate’s evolving state.
- **Difference:** the substrate is *plastic* (connections can strengthen/forget/prune), so the “reservoir” is not fixed.

### 3) Predictive processing / Active inference (spirit, not implementation)
- **Overlap (spirit):** closed-loop interaction, continual adaptation, and internal structure that stabilizes into habits/attractors.
- **Difference:** no explicit generative model, no variational inference, and no prediction-error objective is being optimized.

If you want a longer version, see [doc/research-landscape.md](doc/research-landscape.md).

## Quick start
- Demo: `cargo run`
- Capability assays: `cargo run -- assays`
- Layman-visible environment: `cargo run -- pong-demo`

## 2D visualizer (macroquad)
- Run the interactive 2D demo: `cargo run -p braine_viz`

## Docs
- See [doc/README.md](doc/README.md)
- Interaction + I/O: [doc/interaction.md](doc/interaction.md)
