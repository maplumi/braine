# Scaling plan: module-local learning + routing (Phase 1–2)

This document translates the scalability recommendations into a Braine-aligned implementation path.

**Non-negotiable invariant (project direction):**
- Scale should primarily come from **reusing the same recurrent dynamics and local learning rules**, not from spawning expert/child brains just because a task is larger.
- Child brains remain reserved for **novelty / distribution shift / collapse**, and should consolidate structure back into the parent.

The goal of Phase 1–2 is to make compute and interference predictable without turning the system into an MoE/LLM.

---

## Why modules?

As unit count grows, a single undifferentiated substrate tends to:
- entangle unrelated contingencies (interference),
- accumulate couplings that are expensive to update, and
- lose controllability (reward affects too much of the graph).

A **module** here means a *connectivity + learning boundary* inside a single `Brain`:
- intra-module edges are common,
- cross-module edges are sparse and must “pay rent” (utility / stability),
- learning can be **gated** per module without changing inference.

This differs from child brains/experts: modules are internal structure in one substrate.

---

## Phase 1: module-local budgets + learning active-set

### 1.1 Budgets (bounded work)
Add learning governance that is enforceable per module:
- per-module cap on committed plasticity: $\sum |\Delta w|$ per tick (local L1 budget)
- optional per-module cap on number of updated edges per tick

**Design intent:** keep the learning cost proportional to the amount of useful activity, not to total network size.

### 1.2 Learning active-set (cheap gating)
Introduce an activity-based filter so learning work concentrates on currently relevant units:
- define a “learning activity” scalar per unit (instant amp or activity-trace, depending on `activity_trace_decay`)
- if unit activity is below a threshold, skip eligibility/plasticity for its outgoing edges

**Important:** Phase 1 does *not* require skipping dynamics updates (amp/phase integration still runs on all units). The immediate win is bounding *learning work*.

---

## Phase 2: routing-as-attention (module gating)

### 2.1 Routing purpose
Routing decides **where learning is allowed** in the next step:
- inference continues to run across the full substrate
- only selected modules receive plasticity updates

This is routing-as-attention, not an expert-per-subproblem architecture.

### 2.2 Minimal router (meaning-driven)
Use the boundary symbol set from the last committed observation (stimulus/action/reward symbols) to select top-K modules:

- Maintain a small per-module **signature**: sparse association scores between module and symbols.
- Score modules by:

$$
\mathrm{score}(M_k) = \sum_{s \in S(t)} \mathrm{assoc}(M_k, s) + \beta \cdot \mathrm{reward\_ema}(M_k)
$$

- Activate top-K modules; if scores are uninformative (all near zero), fall back to “no gating” (learn everywhere).

### 2.3 Update policy
After committing an observation, update the signatures for the modules that were routed:
- decay old association strengths
- increment associations for symbols in $S(t)$
- update per-module reward EMA using the current neuromodulator

---

## Constraints / safety

- Defaults must preserve current behavior (routing/budgets off by default).
- New config fields must be **backwards compatible** with persisted images (append-only with safe defaults on read).
- Routing must not cause child brain spawning to become a scaling mechanism.

---

## Acceptance checks

**Phase 1 success:**
- learning time per tick scales with “active set” size, not total unit count
- no regressions in existing tests

**Phase 2 success:**
- on mixed-context workloads, plasticity concentrates into a subset of modules
- interference between unrelated contexts reduces (qualitative and via targeted tests/assays)

---

## Implementation notes (repo mapping)

Core loop touch points live in:
- `Brain::commit_observation` (symbol set available; place to compute routing for next step)
- `Brain::update_eligibility_scalar` (learning active-set gating)
- `Brain::apply_plasticity_scalar` (budget enforcement + module gating)

This phase intentionally does not change the daemon protocol or UI; knobs can be surfaced later once behavior is stable.
