# Scaling plan: module-local learning + routing (Phase 1–5)

This document translates the scalability recommendations into a Braine-aligned implementation path.

**Non-negotiable invariant (project direction):**
- Scale should primarily come from **reusing the same recurrent dynamics and local learning rules**, not from spawning expert/child brains just because a task is larger.
- Child brains remain reserved for **novelty / distribution shift / collapse**, and should consolidate structure back into the parent.

The goal of Phase 1–5 is to make compute and interference predictable without turning the system into an MoE/LLM.

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

## Phase 3: cross-module rent (governance)

Even with routing, cross-module edges can still accumulate and create unintended entanglement. Phase 3 introduces an explicit *rent* on cross-module couplings so that:

- within-module structure is cheap to form and retain,
- cross-module structure must “earn its keep”.

This is not a new learner and not expert spawning; it is a bias that shapes the same local plasticity + forgetting/pruning dynamics.

### Mechanisms

- **Cross-module plasticity scaling**: weight updates on edges crossing module boundaries are multiplied by a scale factor.
	- Config: `cross_module_plasticity_scale` ($\in [0,\infty)$, default `1.0`).
- **Cross-module extra forgetting**: cross-module edges decay faster during forgetting.
	- Config: `cross_module_forget_boost` ($\in [0,\infty)$, default `0.0`).
- **Cross-module stricter pruning**: cross-module edges are pruned using a higher threshold.
	- Config: `cross_module_prune_bonus` ($\in [0,\infty)$, default `0.0`).

These apply only when both endpoints are assigned to modules; unassigned nodes use baseline behavior.

---

## Phase 4: persisted latent modules

Phase 1–3 treat modules as a learning boundary derived mostly from sensor/action groups. Phase 4 adds **latent modules**: internal module definitions that:

- are persisted in the brain image,
- participate in routing/gating like other modules,
- are not exposed as sensors or actions.

This enables future work where the substrate can carve out reusable internal partitions without changing the IO contract.

### Mechanisms

- Add a persisted list of latent module groups (name + unit IDs).
- Include latent modules in routing module rebuild and learning gating.

Persistence is implemented as an **optional chunk** in the brain image (forward-compatible): older images load with no latent modules.

---

## Phase 5: latent module auto-formation + retirement

Phase 4 adds the *ability* to define internal partitions. Phase 5 makes that structure **self-organizing**:

- when routing is enabled but uninformative for a committed boundary symbol set, and
- when that symbol set appears novel (not associated with any module signature), and
- when reward salience is high enough,

the brain may auto-create a fresh latent module with a reserved unit budget. This is a *representation move* (a new internal partition), not a new learner.

Symmetrically, stale low-utility latent modules can be retired (their units are freed and the group is removed from persistence) when they have not been routed for a long time and their reward EMA remains near zero.

### Mechanisms

- **Auto-create** (off by default)
	- `latent_module_auto_create`: enable/disable
	- `latent_module_auto_width`: unit budget for each auto-created module (0 disables)
	- `latent_module_auto_cooldown_steps`: minimum steps between creations
	- `latent_module_auto_max_active`: cap active latent modules (0 = unlimited)
	- `latent_module_auto_reward_threshold`: requires `abs(neuromod) >= threshold`

- **Retire** (off by default)
	- `latent_module_retire_after_steps`: staleness window (0 disables)
	- `latent_module_retire_reward_threshold`: retire only if `abs(reward_ema)` is below this

### Safety notes

- Routing never selects empty modules (modules with zero assigned units are ignored).
- Defaults keep legacy behavior: no auto-create, no retirement.

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

**Phase 3 success:**
- cross-module couplings are measurably harder to grow/retain than within-module couplings under equal eligibility
- saved brains from older versions still load unchanged (defaults apply)

**Phase 4 success:**
- latent modules roundtrip through save/load
- latent modules can be selected by routing (e.g., via symbol seed-match)

**Phase 5 success:**
- when routing is uninformative under novel boundary symbol sets, a latent module can be auto-created and becomes routable
- stale, low-utility latent modules retire cleanly (units unassigned; group removed from persistence)

---

## Implementation notes (repo mapping)

Core loop touch points live in:
- `Brain::commit_observation` (symbol set available; place to compute routing for next step)
- `Brain::update_eligibility_scalar` (learning active-set gating)
- `Brain::apply_plasticity_scalar` (budget enforcement + module gating)

This phase intentionally does not change the daemon protocol or UI; knobs can be surfaced later once behavior is stable.
