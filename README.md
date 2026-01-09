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

### Building and running
```bash
# Build and run core demos
cargo run                      # Basic food/threat demo
cargo run -- assays            # Capability assays
cargo run -- pong-demo         # Pong environment

# Build release binaries
cargo build --release

# Cross-compile for Windows (requires MinGW toolchain)
cargo build --release --target x86_64-pc-windows-gnu

# All-in-one: format, lint, build (Linux + Windows), package
./scripts/dev.sh
```

### Daemon + UI (Spot game)
The daemon-based architecture runs the brain as a persistent service:
```bash
# Start daemon (listens on 127.0.0.1:9876)
cargo run --release -p brained

# Launch UI (connects to daemon)
cargo run --release -p braine_viz

# CLI control (start/stop/status/save/load)
cargo run --release --bin braine-cli -- status
cargo run --release --bin braine-cli -- start
cargo run --release --bin braine-cli -- save
```

**Persistence**: Brain state auto-saves on `Stop` and persists to:
- Linux: `~/.local/share/braine/brain.bbi`
- Windows: `%APPDATA%\Braine\brain.bbi`
- MacOS: `~/Library/Application Support/Braine/brain.bbi`

The UI also supports **timestamped snapshots** (saved alongside the main brain image) so you can
save a point-in-time copy and load older/newer snapshots.
Snapshots live under the same data directory in `snapshots/`.

### Windows portable bundle
After building with `./scripts/dev.sh`, the Windows installer bundle is at:
- `dist/braine-portable.zip` — unzip and run `run_braine.bat` (starts daemon + UI)

See [doc/packaging.md](doc/packaging.md) for details.

## Feature flags

| Feature | Dependency | Use Case |
|---------|------------|----------|
| `parallel` | rayon | Multi-threaded execution for desktop/server |
| `simd` | wide | SIMD vectorization (ARM NEON, x86 SSE/AVX) |
| `gpu` | wgpu | GPU compute shaders for large substrates (10k+ units) |

```bash
# Build with all optimizations
cargo build --release --features "simd,parallel"

# Run benchmarks
cargo bench --features "simd,parallel"
```

## Execution tiers

The substrate supports tiered execution for scaling from MCUs to servers:

```rust
use braine::substrate::{Brain, ExecutionTier};

let mut brain = Brain::new(1024, 16, 42);
brain.set_execution_tier(ExecutionTier::Parallel);  // Use rayon
brain.set_execution_tier(ExecutionTier::Simd);      // Use SIMD
brain.set_execution_tier(ExecutionTier::Gpu);       // Use GPU compute
```

## 2D visualizer (macroquad)
- Run the interactive 2D demo: `cargo run -p braine_viz`

### Meta-modulation (temporal learning progress)
In `braine_viz` Pong, the demo uses a **meta-modulation** signal derived from **temporal learning progress**:
- It tracks the time/steps between successful hits and maintains a moving baseline (EMA).
- When performance gets *slower than its own baseline*, it temporarily increases exploration (it does **not** change the environment reward).
- When performance improves, the extra exploration decays back down.

This is a deliberate design choice: keep the environment reward as the substrate’s neuromodulator, and use “progress” only to modulate how aggressively the system explores/learns.

**Security note:** the visualizer depends on `macroquad`, which currently has a RustSec
informational advisory for unsoundness (no patched version is available as of 2026-01).
The core `braine` crate stays std-only; treat `braine_viz` as a demo tool rather than a
security-hardened component.

## Docs
- **New to braine?** Start with [How It Works](doc/how-it-works.md) for a comprehensive guide with detailed neurogenesis explanation
- See [doc/README.md](doc/README.md)
- Interaction + I/O: [doc/interaction.md](doc/interaction.md)
- Persistence + storage adapters: [doc/brain-image.md](doc/brain-image.md)
- Visualizer games (what each measures): [doc/visualizer-games.md](doc/visualizer-games.md)
- What this does that LLMs don’t (yet): [doc/what-llms-dont-do-yet.md](doc/what-llms-dont-do-yet.md)
- **Accelerated Learning**: [doc/accelerated-learning.md](doc/accelerated-learning.md)

### Learning actions (UI)
The visualizer exposes a few manual “accelerators”:
- **Dream**: offline replay/consolidation over recent structure.
- **Burst**: temporary learning-rate boost for rapid adaptation.
- **Sync**: phase-align sensor groups to improve coherent encoding.
- **Imprint**: one-shot association of the current context.

## Accelerated Learning Mechanisms (9 of 13 implemented)

| Mechanism | Status | Description |
|-----------|--------|-------------|
| Three-Factor Hebbian | ✅ | Core learning rule (pre × post × neuromodulator) |
| One-Shot Imprinting | ✅ | Instant concept formation on novel stimuli |
| Neurogenesis | ✅ | Dynamic capacity expansion when saturated |
| Pruning | ✅ | Structural forgetting of weak connections |
| Child Brains | ✅ | Parallel exploration via brain cloning |
| Attention Gating | ✅ | Focus learning on most active units |
| Burst-Mode Learning | ✅ | Enhanced plasticity on activity spikes |
| Dream Replay | ✅ | Offline memory consolidation |
| Forced Synchronization | ✅ | One-shot supervised learning (teacher mode) |
| STDP | 📋 | Spike-timing dependent plasticity |
| BCM Homeostasis | 📋 | Self-tuning activity thresholds |
| Meta-Learning | 📋 | Per-connection learning rate adaptation |
| Predictive Shortcut | 📋 | Direct wiring from causal memory |
