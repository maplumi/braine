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
- [Architecture](architecture.md): core substrate + “body/frame” interface.
- [Metrics](metrics.md): simple capability measurements to track progress.
- [Experiments](experiments.md): experiment log template + current backlog.
- [Research landscape](research-landscape.md): what this resembles and how it differs.

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
