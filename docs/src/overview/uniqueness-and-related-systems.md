# What makes Braine unique (and related systems)

This document summarizes what’s distinctive about **Braine** in this repo, and maps it to nearby families of systems that “rhyme” with the same goals.

Braine = **Biologically-Rooted Adaptive Intelligent Neural Engine**: a continuously running, closed-loop learning substrate built on sparse recurrent dynamics + local plasticity + scalar reward (neuromodulation).

## What makes Braine unique

### 1) Always-on, closed-loop learning substrate (not an offline model)
- The system is designed to run continuously (`brained` daemon owns long-lived `Brain` state).
- Learning happens online from **stimulus → action → reward** streams.
- Persistence is part of the contract (brain image + runtime state), so learning carries across sessions.

How this differs from many ML systems:
- Most ML stacks are “train → freeze → deploy”; Braine is “run → adapt → persist”.

### 2) Local plasticity with a scalar neuromodulator (3-factor learning)
- Plasticity is **local**: updates depend on local unit activity/phase alignment.
- A scalar global signal (reward/salience) gates commitment (three-factor Hebbian style).
- No backprop, no global loss, no gradients.

Why this matters:
- Credit assignment is structural/local and can operate continuously.

### 3) Sparse recurrent dynamics are the compute substrate
- Computation is performed by evolving recurrent state (oscillatory-ish dynamics), not by token prediction.
- Readouts select actions; behavior emerges from dynamics + learned couplings.

### 4) Capacity management is in-loop (forgetting, pruning, growth)
- The substrate is intended to remain bounded over long runtimes.
- Structural forgetting/pruning frees capacity.
- Optional growth mechanisms (e.g., neurogenesis / experts) are framed as adaptation tools rather than “bigger model = better”.

### 5) Observability hooks are first-class
- The daemon protocol and runtime stats are meant to make learning diagnosable: what’s improving, what’s saturating, whether commits are happening.
- This repo also treats “games” as instrumentation: small tasks that expose specific learning capabilities.

## Related systems (by category)

These categories describe *near neighbors*. Braine shares themes with them, but typically differs in one or more core invariants: online learning, local plasticity, persistent substrate, bounded resource operation, and closed-loop task framing.

### A) Reservoir computing / Liquid State Machines (LSM)
**Similarity**
- Uses rich recurrent dynamics as the computational core.
- Often uses a simple readout layer to select outputs.

**Key difference vs Braine**
- Classic reservoirs are usually **fixed** (train only the readout). Braine’s substrate is intended to be **plastic** and to persist across time.

When to compare
- If you want “dynamics as compute” but do not need lifetime structural learning, reservoirs are a close conceptual neighbor.

### B) Neuromodulated Hebbian learning / three-factor learning (eligibility traces)
**Similarity**
- Strong conceptual match: eligibility traces + scalar modulation gating plasticity.
- Often used to model dopamine-gated learning and biologically plausible reinforcement.

**Key difference vs Braine**
- Many implementations are research simulations or single-task demos; Braine packages this into a long-running service boundary with persistence, task loop, and observability.

When to compare
- If you care about biological plausibility of learning rules and online RL-like adaptation, this is the closest family.

### C) Spiking simulators with dopamine/STDP (Brian2 / NEST-style ecosystems)
**Similarity**
- Can express local rules like STDP and dopamine-like modulation.
- Can run closed-loop simulations with online adaptation.

**Key difference vs Braine**
- Those tools are general-purpose simulators; Braine is an opinionated substrate with a specific interaction contract, persistence format, and product-shaped daemon/UI/CLI.

When to compare
- If you want high biological fidelity (spikes, conductances, detailed neuron models), spiking simulators are the right comparison point.

### D) Predictive processing / active inference
**Similarity**
- Similar “spirit”: closed-loop interaction, continual adaptation, internal state that stabilizes into behaviors.

**Key difference vs Braine**
- Many predictive coding / active inference implementations optimize explicit prediction-error objectives or perform global inference.
- Braine is primarily framed around local plasticity gated by scalar neuromodulation, not variational inference.

When to compare
- If you want explicit generative models and inference-as-learning, active inference is the closer neighbor.

### E) HTM (Hierarchical Temporal Memory)
**Similarity**
- Online learning and continual adaptation are central.
- Focus on structure, not gradient descent.

**Key difference vs Braine**
- Underlying representations/mechanisms differ (HTM emphasizes sparse distributed representations and sequence memory; Braine emphasizes recurrent dynamics + local plasticity + neuromodulation + task loop).

When to compare
- If you want online anomaly detection/sequence memory with engineered sparsity, HTM is a relevant neighbor.

### F) Neuroevolution / evolutionary RL hybrids
**Similarity**
- Can avoid backprop and still produce adaptive agents.

**Key difference vs Braine**
- Often uses an outer-loop population search (evolution) rather than purely in-loop lifetime learning.
- Braine’s intent is “learn during operation” without needing large outer-loop optimization.

When to compare
- If you need robust optimization without differentiability and can afford population search, neuroevolution approaches are relevant.

## What Braine is *not*

- Not a transformer or LLM objective (no token prediction training loop).
- Not backprop-trained deep RL.
- Not a pure reservoir with a trained readout only.

## Where to look next

- Core framing: [How It Works](how-it-works.md)
- Research positioning: [Research comparison](../research/research-comparison.md)
- “What LLMs don’t do (yet)”: [What LLMs don't do yet](what-llms-dont-do-yet.md)
