# Research landscape (and differentiation)

This project sits near several established areas.

## Closest related areas
- **Attractor networks**: memories as stable patterns that can be recalled from partial cues.
- **Continuous Hopfield-style dynamics**: energy/competition-driven convergence (conceptually).
- **Reservoir computing / liquid state machines**: rich dynamics + lightweight readouts.
- **Coupled oscillator models / neural oscillations**: phase relationships as a representational dimension.
- **Local plasticity (Hebb/STDP families)**: online adaptation without backprop.
- **Embodied cognition**: behavior emerges from closed-loop interaction, not static datasets.

## How we keep it distinct from LLM-style systems
- No token prediction objective.
- No gradient training on corpora.
- No dense embeddings/matrices.
- Online, local, scalar updates.
- Built for edge constraints (bounded compute + bounded memory growth + pruning).

## Novelty statement (careful)
It is unlikely any single ingredient here is “new” academically.
The intended novelty is:
- a **minimal** implementation path (Rust, sparse scalar ops)
- an edge-first constraint set (low power + high memory + continual learning)
- the specific integration of: oscillatory state + local plasticity + structural forgetting + embodiment frame

As we evolve, novelty should be demonstrated through:
- ablation comparisons (turn off imprinting / phase lock / pruning)
- capability metrics improving under strict compute/memory budgets
