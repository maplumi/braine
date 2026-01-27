# Braine Capabilities Checklist

This document tracks what braine should be able to do, with testable criteria.
Check items off as they are verified through experiments.

---

## Level 0: Fundamental Operations

These must work for anything else to work.

| # | Capability | Test | Status | Notes |
|---|-----------|------|--------|-------|
| 0.1 | Oscillators run | Units have changing phase/amplitude | ☑️ | Core step() function |
| 0.2 | Connections exist | Non-zero weights between units | ☑️ | CSR storage works |
| 0.3 | Hebbian learning fires | Co-active units strengthen connections | ☑️ | Tested in unit tests |
| 0.4 | Neuromodulation affects learning | Reward changes learning rate | ☑️ | 3-factor rule implemented |
| 0.5 | Sensors activate units | Stimulus → unit amplitude increase | ☑️ | apply_stimulus works |
| 0.6 | Actions readable | Can query which action group is most active | ☑️ | select_action works |

---

## Level 1: Basic Association Learning

Single stimulus → single response mappings.

| # | Capability | Test | Status | Notes |
|---|-----------|------|--------|-------|
| 1.1 | Learn one association | sensor A → action X after repeated reward | ☑️ | Spot: converges to high hit-rate in short runs |
| 1.2 | Learn two associations | A→X, B→Y simultaneously | ☑️ | Spot: left/right stimulus-action mappings learned together |
| 1.3 | Discriminate stimuli | Different sensors → different actions | ☑️ | Spot: distinct stimuli drive distinct actions (not a single reflex) |
| 1.4 | Retain over time | Association persists across 100+ steps | ☑️ | Observed stable performance over 100+ trials; state persists via braine.bbi + runtime.json |
| 1.5 | Imprint one-shot | Single strong exposure creates association | 🔄 | Mechanism exists (imprint_if_novel + imprint_rate), needs a dedicated one-shot experiment |

---

## Level 2: Reward-Based Learning

Learning from reinforcement signals.

| # | Capability | Test | Status | Notes |
|---|-----------|------|--------|-------|
| 2.1 | Prefer rewarded action | After +reward, action more likely | ⬜ | Bandit game |
| 2.2 | Avoid punished action | After -reward, action less likely | ⬜ | Forage game |
| 2.3 | Track changing values | Adapt when reward structure changes | ⬜ | Bandit with flip |
| 2.4 | Credit correct action | Only reinforce the chosen action | ⬜ | |
| 2.5 | Explore when uncertain | Try different actions initially | ⬜ | |

---

## Level 3: Adaptation and Unlearning

Responding to environmental changes.

| # | Capability | Test | Status | Notes |
|---|-----------|------|--------|-------|
| 3.1 | Unlearn old mapping | When A→X no longer rewarded, stop doing X | ⬜ | Pong axis flip |
| 3.2 | Learn new mapping | After unlearning, acquire A→Y | ⬜ | |
| 3.3 | Reversal learning | Complete A→X to A→Y switch | ⬜ | Whack game |
| 3.4 | Rapid re-adaptation | Faster second reversal than first | ⬜ | Savings effect |
| 3.5 | Regime detection | Behave differently in different regimes | ⬜ | Sequence game |

---

## Spatial / Continuous Track (Higher-D Inputs)

This track covers *spatial awareness* and scaling to higher-dimensional observations without changing the core principles.

Key idea: prefer **factorized population codes** (e.g., X-code + Y-code) over a full combinatorial grid.
This keeps sensor dimensionality roughly linear in the number of axes.

| # | Capability | Test | Status | Notes |
|---|-----------|------|--------|-------|
| S1 | Encode 2D position | Given (x,y), sensor vector is normalized and stable | ⬜ | Population code: e.g., 8–32 bumps per axis; deterministic via seed |
| S2 | Learn quadrant/side | Reward based on quadrant → choose correct action | ⬜ | Start with 4 actions (NE/NW/SE/SW) or 2 actions (left/right) |
| S3 | Generalize locally | Train on subset of positions; test on held-out nearby positions | ⬜ | Validates smooth sensor similarity → smooth policy |
| S4 | Context separation | Same (x,y) in two contexts yields different action mapping | ⬜ | Mirrors reversal-regime logic; prevents catastrophic overwrite |
| S5 | Scale to N dims | Add a new axis (z) with no redesign | ⬜ | Goal: add `Z` sensors and keep learning dynamics stable |

Suggested experiments (minimal → harder):
1) `SpotXY (2 actions)`: choose left/right based on x<0 vs x>=0.
2) `SpotXY (4 actions)`: quadrant classification.
3) `SpotXY-Reversal`: flip rule mid-run; measure recovery.

---

## Level 4: Temporal Processing

Handling time-extended patterns.

| # | Capability | Test | Status | Notes |
|---|-----------|------|--------|-------|
| 4.1 | Persist activation | Activity echoes for 5-10 steps | ⬜ | Phase persistence |
| 4.2 | Delayed response | Stimulus now → action later | ⬜ | Delayed Association game |
| 4.3 | Sequence completion | A-B-? → predict C | ⬜ | Sequence game |
| 4.4 | Temporal credit | Reward now credits action from 5 steps ago | ⬜ | Hard |
| 4.5 | Rhythm entrainment | Oscillators sync to periodic input | ⬜ | |

---

## Temporal Structure Track (Predict / Track)

This track is about learning from *sequences* rather than single-step labels.

| # | Capability | Test | Status | Notes |
|---|-----------|------|--------|-------|
| T1 | One-step prediction | Given (x,y) and velocity, predict next (x',y') bucket | ⬜ | Start discrete (bins), then continuous (population code) |
| T2 | Temporal aliasing handling | Same observation → different best action depending on history | ⬜ | Requires internal state or working memory-like dynamics |
| T3 | Delayed reward credit | Reward at t credits action at t-k | ⬜ | Measure performance vs delay length |

---

## Level 5: Attention and Selection

Filtering relevant from irrelevant.

| # | Capability | Test | Status | Notes |
|---|-----------|------|--------|-------|
| 5.1 | Ignore distractors | Learn with irrelevant sensors active | ⬜ | Pong decoy ball |
| 5.2 | Selective learning | Only high-amplitude units learn | ☑️ | Unit test: attention_gate_focuses_learning |
| 5.3 | Focus on rewarded | Increase attention to reward-predictive stimuli | ⬜ | |
| 5.4 | Filter by phase | Same-phase units bind, opposite don't | ⬜ | Phase binding |
| 5.5 | Attentional switch | Shift focus when target changes | ⬜ | Beacon game |

---

## Level 6: Memory Consolidation

Strengthening and organizing memories.

| # | Capability | Test | Status | Notes |
|---|-----------|------|--------|-------|
| 6.1 | Dream replay helps | Offline processing improves retention | ☑️ | Unit test: dream_consolidates_memories |
| 6.2 | Burst learning works | High-plasticity bursts accelerate learning | ☑️ | Unit test: burst_learning_detects_spikes |
| 6.3 | Pruning cleans up | Weak connections removed over time | ⚠️ | Pruning + diagnostics exist (pruned_last_step), but no explicit pruning-focused test/experiment yet |
| 6.4 | Consolidation transfers | Knowledge moves from fast to slow weights | ⬜ | Child brain? |
| 6.5 | Interference reduced | Learning B doesn't erase A | ⬜ | Catastrophic forgetting |

---

## Level 7: Generalization

Applying learning beyond exact training.

| # | Capability | Test | Status | Notes |
|---|-----------|------|--------|-------|
| 7.1 | Similar stimuli transfer | Learning A→X helps with A'→X | ⬜ | Need graded sensors |
| 7.2 | Novel combinations | A+B seen, A+C novel → reasonable response | ⬜ | |
| 7.3 | Interpolation | Train on extremes, test middle | ⬜ | |
| 7.4 | Abstraction | Learn "left" means left regardless of context | ⬜ | Very hard |

---

## Level 8: Multi-step Behavior

Chaining actions toward goals.

| # | Capability | Test | Status | Notes |
|---|-----------|------|--------|-------|
| 8.1 | Two-step sequence | Do A then B for reward | ⬜ | |
| 8.2 | Navigate to goal | Multiple movements to reach target | ⬜ | Beacon/Forage |
| 8.3 | Subgoal learning | Learn intermediate targets | ⬜ | |
| 8.4 | Planning | Represent future before acting | ⬜ | Probably impossible |

---

## Quick Test Protocol

### Minimal Test (5 minutes)
1. Run Spot game (Level 1)
2. Confirm hit rate > 0.8 after 100 trials

### Standard Test (30 minutes)
1. Run each game for 500 frames
2. Record final hit rate
3. Trigger one flip, measure recovery time

### Full Test (2 hours)
1. All games, multiple runs
2. With/without each accelerated learning mechanism
3. Statistical comparison

---

## Status Legend

| Symbol | Meaning |
|--------|---------|
| ⬜ | Not tested |
| 🔄 | In progress |
| ☑️ | Verified working |
| ⚠️ | Partially works |
| ❌ | Does not work |

---

## Changelog

| Date | Update |
|------|--------|
| 2026-01-07 | Initial checklist created |
| 2026-01-08 | Updated Level 1 (Spot) to verified; marked attention/dream/burst as verified by unit tests; clarified imprinting/pruning as not yet behaviorally validated |
| 2026-01-08 | Added Spatial/Continuous and Temporal-Structure tracking sections with concrete experiment/test targets |

