# Braine docs

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
- [How It Works](overview/how-it-works.md): comprehensive guide to how braine operates, with detailed neurogenesis explanation.

### Overview
- [How It Works](overview/how-it-works.md): comprehensive guide with detailed neurogenesis explanation
- [What LLMs don't do (yet)](overview/what-llms-dont-do-yet.md): closed-loop online learning, persistent adaptation, embodiment framing
- [Problem sets → Trials](overview/problem-sets.md): how to transform datasets/problems into closed-loop trial streams

### Architecture
- [Architecture](architecture/architecture.md): core substrate + "body/frame" interface + execution tiers
- (In Architecture) `brained` daemon + UI/CLI protocol, including FPS vs trial-period controls
- [Interaction](architecture/interaction.md): inputs, outputs, and basic interaction model
- [Brain Image Format](architecture/brain-image.md): custom persistence format + storage adapters
- [Performance](architecture/performance.md): execution tiers, SIMD, GPU, and benchmarking
- [Experts / Child Brains](architecture/experts.md): general sandbox learning mechanism for all games (design contract; implemented later)

### Learning Mechanisms
- [Accelerated Learning](learning/accelerated-learning.md): neurogenesis, dream replay, and other speed-up mechanisms
- [Spatial + Temporal Design](learning/spatial-temporal-design.md): population-coded spatial inputs, scaling to higher dimensions, and a test-first battery

### Research Context
- [Research Comparison](research/research-comparison.md): how braine differs from mainstream AI, AGI analysis
- [Research Landscape](research/research-landscape.md): what this resembles and how it differs
- [Research Questions](research/research-questions.md): open questions and research directions

### Games & Testing
- [Visualizer Games](games/visualizer-games.md): how to run each game + what it measures
- [Game Testing Guide](games/game-testing-guide.md): measuring brain capabilities with each game
- [Capabilities Checklist](games/capabilities-checklist.md): testable criteria for what braine should do
- [Experiments](games/experiments.md): experiment log template + current backlog
- [Pong performance notes](games/pong-performance.md): why hit-rate can look capped and what to try

### Web
- [Web vs Desktop parity](development/web-desktop-parity.md): what exists where (and why)

### Development
- [Packaging](development/packaging.md): how to build and package for distribution
- [Metrics](development/metrics.md): simple capability measurements to track progress

### Proposals & Roadmaps
- [Enhancement Proposal](proposals/enhancement-proposal.md): brain image + visualizer redesign proposals
- [Graph Visualization Roadmap](proposals/graph-visualization-roadmap.md): future visualization features

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
