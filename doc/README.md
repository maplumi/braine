# braine docs

This folder documents the **braine** research project: a minimal, brain-like cognitive substrate built from **local scalar dynamics** (no matrices, no backprop, no transformers).

## Goals
- Build a system that learns continuously from interaction (not text prediction).
- **Ultra-low-power / edge-first**: sparse local updates, bounded memory, and natural forgetting.
- **Memory = storage = state**: knowledge lives in persistent internal structure (couplings + oscillatory state), not a separate database.
- Progressive capability growth: repeated behaviors become stable and consistent (habits/attractors).

## Non-goals
- Competing with LLMs on text benchmarks.
- Dense vector embeddings, matrix multiplications, gradient descent, or pretraining.

## How to navigate

**New to braine? Start here:**
- [How It Works](how-it-works.md): comprehensive guide to how braine operates, with detailed neurogenesis explanation.

**Core documentation:**
- [Architecture](architecture.md): core substrate + "body/frame" interface + execution tiers.
- (In Architecture) `brained` daemon + UI/CLI protocol, including FPS vs trial-period controls.
- [Experts / child brains](experts.md): general sandbox learning mechanism for all games (design contract; implemented later).
- [Performance](performance.md): execution tiers, SIMD, GPU, and benchmarking.
- [Accelerated Learning](accelerated-learning.md): neurogenesis, dream replay, and other speed-up mechanisms.
- [What LLMs don’t do (yet)](what-llms-dont-do-yet.md): closed-loop online learning, persistent adaptation, embodiment framing.
- [Spatial + temporal design note](spatial-temporal-design.md): population-coded spatial inputs, scaling to higher dimensions, and a test-first battery.
- [Research Comparison](research-comparison.md): how braine differs from mainstream AI, AGI analysis.
- [Research Landscape](research-landscape.md): what this resembles and how it differs.
- [Metrics](metrics.md): simple capability measurements to track progress.
- [Experiments](experiments.md): experiment log template + current backlog.
- [Brain image format](brain-image.md): custom persistence format + storage adapters.
- [Visualizer games](visualizer-games.md): how to run each game + what it measures.

## Reproducible assays
- Run the current assays with: `cargo run -- assays`
- Run the layman-visible environment demo with: `cargo run -- pong-demo`
- Run the interactive-ish demo with: `cargo run`

## Code separation
- Core brain code lives in `src/lib.rs` + `src/{substrate,causality,supervisor,prng}.rs`
- Demos/assays live in `src/experiments/*` and are wired by `src/main.rs`

## Working definitions
- **Unit**: a tiny dynamical element with amplitude + phase.
- **Connection**: sparse directed coupling between units.
- **Attractor**: a stable pattern the system repeatedly returns to.
- **Imprinting**: one-shot creation/strengthening of a concept link from perception.
- **Neuromodulator**: a scalar reward/salience signal that scales plasticity.
