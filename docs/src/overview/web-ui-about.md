# Web UI “About” (in-app summary)

The web app includes an in-app **About** view that summarizes how Braine works at a glance.
This page mirrors that content in the main docs so it lives alongside the rest of the project documentation.

If you’re looking for the full docs site (GitHub Pages), open: https://maplumi.github.io/braine/docs/

## What the in-app About covers

### Overview
Braine is a continuously running dynamical substrate with:
- sparse recurrent units (amplitude/phase)
- local plasticity (no global loss, no backprop)
- a scalar neuromodulator that gates learning

For the deeper architectural walk-through, start with:
- [How It Works](how-it-works.md)
- [Architecture](../architecture/architecture.md)

### Dynamics
The core idea is “computation via recurrent dynamics” — patterns stabilize into attractors/habits.

Relevant docs:
- [Performance](../architecture/performance.md)
- [Stability/plasticity control](../architecture/stability-plasticity-control.md)

### Learning
Learning is local and online. Each trial/observation typically:
1) applies stimuli
2) runs dynamics
3) chooses an action
4) applies reward/neuromodulation
5) commits or discards the observation

Relevant docs:
- [Accelerated learning](../learning/accelerated-learning.md)
- [Problem sets → Trials](problem-sets.md)

### Memory
Braine records symbols during an observation (actions, stimulus/action pairs, compound symbols) and updates bounded causal/meaning memory.

Relevant docs:
- [Interaction](../architecture/interaction.md)
- [Memory bottlenecks](../architecture/memory-bottlenecks.md)

### The “Math Behind” (embedded in the web UI)
The web UI embeds the repo’s math explainer as rendered Markdown.

- [The math behind](../maths/the-math-behind.md)

## Where to go next

- Want to integrate an external loop? Read [Daemon protocol](../architecture/daemon-protocol.md)
- Want to understand task definitions? Read [Language contract](../architecture/language-contract.md)
- Want to evaluate behavior? Read [Visualizer games](../games/visualizer-games.md)
