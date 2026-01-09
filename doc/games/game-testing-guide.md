# Game Testing Guide: Measuring Brain Capabilities

This document describes how each visualizer game tests specific brain capabilities,
what metrics indicate success, and how to measure learning rate.

---

## Overview: What Each Game Tests

| Game | Primary Capability | Secondary Capabilities |
|------|-------------------|----------------------|
| **Pong** | Sensorimotor mapping | Regime adaptation, distractor filtering |
| **Bandit** | Reward-based selection | Probability estimation, regime shift |
| **Forage** | Spatial navigation | Reward/punishment discrimination |
| **Whack** | Timed response | Mapping reversal adaptation |
| **Beacon** | Goal-directed movement | Target switching, distractor rejection |
| **Sequence** | Temporal prediction | Pattern memory, regime switching |

---

## Cross-cutting: Experts (child brains) across all games

This repo is adding a general expert/child-brain mechanism intended to work across every game. The canonical design contract lives in:
- [Expert / Child-Brain Mechanism](../architecture/experts.md)

### What stays the same
- Games still apply stimuli, read an action, and compute reward.
- With experts disabled, behavior should match today.

### What should change (observable)

When experts are enabled (once implemented):
- **Lower interference**: competence on an already-learned mapping should degrade less while exploring novelty.
- **Faster recovery**: after flips/regime changes, a child can adapt quickly, then consolidation improves the parent.
- **Bounded churn**: the parent’s structure changes less aggressively than the child’s.

### How to test (baseline vs experts)

For any game that has a flip/reversal/regime shift:
1. Run baseline (experts disabled). Record recovery time after a flip.
2. Run with experts enabled. Record:
   - whether a child spawns near the flip/novelty event,
   - whether recovery time improves,
   - whether performance remains stable outside the novel regime.

The mechanism is considered successful if:
- Disabling experts reproduces baseline.
- Enabling experts improves at least two games without destabilizing others.

---

## Game 1: Pong

### What It Tests
- **Sensorimotor mapping**: Linking visual context (ball position relative to paddle) to motor actions (left/right/stay)
- **Regime adaptation**: Sensor axis flips periodically, requiring the brain to unlearn and relearn mappings
- **Distractor filtering**: Decoy ball provides irrelevant sensory input that should be ignored

### Sensors & Actions
```
Sensors (5 context buckets):
  pong_ctx_far_left, pong_ctx_left, pong_ctx_aligned, pong_ctx_right, pong_ctx_far_right
  pong_ball_falling (binary: is ball moving downward?)
  
Distractors (decoy ball):
  pong_decoy_ctx_* (same buckets as above)
  pong_decoy_falling

Actions:
  left, right, stay
```

### Success Metrics

| Metric | Good | Excellent | How to Read |
|--------|------|-----------|-------------|
| **Hit Rate** | > 0.5 | > 0.7 | `hits / (hits + misses)` in HUD |
| **Recent Rate** | > 0.6 | > 0.8 | Rolling window of last N outcomes |
| **Score** | Increasing | Stable high | Cumulative reward |

### Learning Rate Measurement

1. **Initial Learning Speed**:
   - Frames to first 10 consecutive hits
   - Time to reach 0.5 hit rate from cold start

2. **Regime Adaptation Speed** (key metric):
   - `FLIP_RECOVERY` events in log show outcomes needed to recover after axis flip
   - Good: < 30 outcomes to recover
   - Excellent: < 15 outcomes

3. **Action Score Convergence**:
   - Watch `top_actions_ema` in HUD
   - When correct action consistently scores highest for each context = learned

### Testing Accelerated Learning

| Mechanism | How to Test | Expected Effect |
|-----------|------------|-----------------|
| **Attention Gating** | Enable at difficulty 3 with decoy | Should ignore decoy, focus on real ball |
| **Dream Replay** | Trigger after hitting 0.6 rate | Faster convergence to 0.7+ |
| **Burst Mode** | Enable during slump after flip | Faster recovery from regime shift |
| **Force Sync** | Trigger after flip | Immediate phase alignment for sensors |

### Difficulty Levels
- **Level 0**: No jitter, no decoy, easy (baseline learning test)
- **Level 1**: Slight jitter, no decoy
- **Level 2**: Moderate jitter + decoy (realistic challenge)
- **Level 3**: High jitter + fast decoy (stress test)

---

## Game 2: Bandit (Two-Armed Bandit)

### What It Tests
- **Value estimation**: Learning which arm (A or B) has higher reward probability
- **Exploration vs exploitation**: Balancing trying both arms vs. sticking with best
- **Regime adaptation**: Arm probabilities swap periodically

### Sensors & Actions
```
Sensors:
  bandit_ctx (constant context, always active)

Actions:
  A (arm A), B (arm B)
```

### Success Metrics

| Metric | Good | Excellent | Interpretation |
|--------|------|-----------|----------------|
| **Win Rate** | > 0.6 | > 0.75 | Should approach p(best_arm) |
| **Recent64** | > 0.65 | > 0.8 | Short-term decision quality |
| **Arm Selection** | Mostly optimal | >90% optimal | Which action scores highest |

### Learning Rate Measurement

1. **Initial Preference Formation**:
   - Steps to consistently prefer the 0.8 arm over 0.2 arm
   - Good: < 50 steps

2. **Regime Recovery**:
   - Steps to recover after probability swap
   - Look for `FLIP_RECOVERY` in logs
   - Good: < 40 steps
   - Excellent: < 20 steps

3. **Action Score Dominance**:
   - Watch `top_actions(ctx)` - winning arm should dominate by +0.3 or more

### Testing Accelerated Learning

| Mechanism | Test Approach | Expected Improvement |
|-----------|--------------|---------------------|
| **Dream Replay** | Trigger after 100 steps | Stronger arm preference |
| **Burst Mode** | Enable right after flip | Faster preference reversal |
| **Imprint** | Click when selecting correct arm | Immediate strong preference |

---

## Game 3: Forage

### What It Tests
- **Spatial navigation**: Moving toward targets in 2D space
- **Reward discrimination**: Green vs red items have opposite values (configurable)
- **Regime adaptation**: Which color is "good" flips periodically

### Sensors & Actions
```
Sensors (18 context combinations):
  forage_ctx_{green,red}_{L,C,R}{U,C,D}
  Example: forage_ctx_green_LU = green target is Left-Up

Poison sensors (distractor position):
  forage_poison_dx_{L,C,R}
  forage_poison_dy_{U,C,D}

Actions:
  up, down, left, right, stay
```

### Success Metrics

| Metric | Good | Excellent | Notes |
|--------|------|-----------|-------|
| **Score** | Increasing | High positive | +1 for good, -1 for bad |
| **Recent Rate** | > 0.6 | > 0.8 | Proportion of correct collections |
| **Flip Recovery** | < 30 outcomes | < 15 outcomes | Adapting to color swap |

### Learning Rate Measurement

1. **Navigation Learning**:
   - Does brain move toward active target?
   - Check if action matches direction (e.g., target is Left → action is "left")

2. **Discrimination Learning**:
   - After regime flip, how quickly does brain avoid previously-rewarded color?
   - Watch for initial performance drop then recovery

3. **Spatial Consistency**:
   - Same context should produce same action across episodes

### Testing Accelerated Learning

| Mechanism | Test Scenario |
|-----------|---------------|
| **Attention** | Set boost high when poison is nearby |
| **Dream Replay** | After successful foraging session |
| **Force Sync** | After color regime flip |

---

## Game 4: Whack (Whack-a-Mole)

### What It Tests
- **Reactive mapping**: Quick response to target appearance
- **Mapping reversal**: Lane-to-action mapping flips periodically
- **Timed response**: Must respond before target disappears

### Sensors & Actions
```
Sensors (6 context combinations):
  whack_ctx_map{0,1}_lane{0,1,2}
  Example: whack_ctx_map0_lane1 = mapping 0, target in lane 1

Actions:
  A, B, C (correspond to lanes 0, 1, 2 in map0; reversed in map1)

Mapping:
  map0: A→lane0, B→lane1, C→lane2
  map1: A→lane2, B→lane1, C→lane0 (reversed)
```

### Success Metrics

| Metric | Good | Excellent |
|--------|------|-----------|
| **Hit Rate** | > 0.6 | > 0.8 |
| **Recent50** | > 0.65 | > 0.85 |
| **Post-Flip Recovery** | < 20 outcomes | < 10 outcomes |

### Learning Rate Measurement

1. **Action-Lane Mapping**:
   - For each lane, does the correct action dominate?
   - map0_lane0 → A should be highest
   - map0_lane1 → B should be highest
   - etc.

2. **Reversal Learning**:
   - After map flip, how fast do action preferences reverse?
   - This is a pure credit assignment test

3. **Response Time** (implicit):
   - If brain frequently misses targets, it's not responding fast enough

### Testing Accelerated Learning

| Mechanism | Application |
|-----------|-------------|
| **Burst Mode** | Enable during map reversal period |
| **Imprint** | Click when successfully hitting target |
| **Force Sync** | After mapping flip |

---

## Game 5: Beacon

### What It Tests
- **Goal-directed navigation**: Moving toward a specified target
- **Target switching**: Active target alternates (blue/yellow)
- **Distractor rejection**: Non-target beacon should be ignored

### Sensors & Actions
```
Sensors (18 per color):
  beacon_ctx_{blue,yellow}_{L,C,R}{U,C,D}
  Example: beacon_ctx_blue_RD = blue beacon is Right-Down relative to agent

Distractor direction:
  beacon_distr_dx_{L,C,R}
  beacon_distr_dy_{U,C,D}

Actions:
  up, down, left, right, stay
```

### Success Metrics

| Metric | Good | Excellent |
|--------|------|-----------|
| **Hits** | Increasing steadily | Rapid collection |
| **Score** | High positive | Minimal penalties |
| **Target Switching** | Quick adaptation | < 5 hits to adapt |

### Learning Rate Measurement

1. **Directional Accuracy**:
   - When target is Left-Up, does brain move left and up?
   - Watch action selection vs context

2. **Target Discrimination**:
   - After target switch (blue→yellow), how many hits to adapt?
   - Good: < 10 hits
   - Excellent: < 5 hits

3. **Distractor Rejection**:
   - Brain should NOT navigate toward non-target beacon
   - Watch for incorrect movements toward distractor

### Testing Accelerated Learning

| Mechanism | Test |
|-----------|------|
| **Attention** | Boost to filter distractor signals |
| **Dream Replay** | After successful target collection streak |
| **Auto-triggers** | Enable "Dream on Flip" for automatic replay after target switch |

---

## Game 6: Sequence

### What It Tests
- **Temporal prediction**: Predicting next token in a sequence
- **Pattern memory**: Remembering fixed patterns (ABAC, ACBC)
- **Regime switching**: Active pattern changes periodically

### Sensors & Actions
```
Sensors:
  seq_token_A, seq_token_B, seq_token_C (current token shown)
  seq_regime_0, seq_regime_1 (which pattern is active)

Actions:
  A, B, C (prediction of next token)

Patterns:
  pattern0: A → B → A → C → (repeat)
  pattern1: A → C → B → C → (repeat)
```

### Success Metrics

| Metric | Good | Excellent |
|--------|------|-----------|
| **Accuracy** | > 0.5 | > 0.75 |
| **Recent Rate** | > 0.6 | > 0.8 |
| **Pattern Recognition** | Correct predictions | Near-perfect for known pattern |

### Learning Rate Measurement

1. **Sequence Learning Curve**:
   - Start: random (0.33 accuracy for 3 actions)
   - After 50 steps: should exceed 0.5
   - After 200 steps: should approach 0.7+

2. **Transition Probabilities**:
   - Brain should learn: "After A in pattern0, B is likely"
   - Check action scores when showing each token

3. **Regime Switch Adaptation**:
   - After pattern flip, accuracy drops then recovers
   - Recovery time indicates learning speed

### Testing Accelerated Learning

| Mechanism | Application |
|-----------|-------------|
| **Dream Replay** | Consolidate pattern memory after successful run |
| **Burst Mode** | Enable after pattern switch |
| **Force Sync** | Helps with regime indicator processing |

---

## Universal Measurement Framework

### Quantitative Metrics (from logs)

```
Look for these events in braine_viz_metrics.log:

FLIP_MARKER     → Records baseline before regime flip
FLIP_RECOVERY   → Records steps/outcomes to recovery
reinforce       → Shows learning signal magnitude
```

### Learning Rate Formula

```
Learning Rate = 1 / (outcomes_to_criterion)

Where criterion could be:
- Reaching 0.6 hit rate
- First 5 consecutive successes  
- Recovery to pre-flip performance

Higher = Faster learning
```

### Comparative Testing Protocol

1. **Baseline Run** (no accelerated learning):
   - Run game for 500 frames
   - Record: final hit rate, outcomes to first flip recovery, score

2. **Enhanced Run** (with accelerated learning):
   - Enable mechanism (e.g., Burst Mode)
   - Run game for 500 frames
   - Record same metrics

3. **Compare**:
   - Faster recovery = mechanism helps adaptation
   - Higher final rate = mechanism helps overall learning
   - Higher score = mechanism improves performance

### Recommended Test Sequence

1. **Cold Start Test**: New brain, run until first flip recovery
2. **Sustained Performance**: Continue for 3+ flips, measure recovery consistency
3. **Stress Test**: Maximum difficulty, enable all auto-triggers
4. **Ablation**: Disable mechanisms one by one to see individual contribution

---

## Quick Reference: Metric Locations in HUD

| Game | Key Metrics in HUD |
|------|-------------------|
| Pong | `hits=X misses=Y hit_rate=Z recent=W` |
| Bandit | `wins=X losses=Y win_rate=Z recent=W` |
| Forage | `score=X recent=Y flip_countdown=Z` |
| Whack | `outcomes=X recent50=Y flip_countdown=Z` |
| Beacon | `hits=X score=Y target={BLUE/YELLOW}` |
| Sequence | `outcomes=X score=Y pattern={0/1}` |

---

## Log File Analysis

The metrics log (`braine_viz_metrics.log`) contains detailed data for analysis:

```bash
# Find all flip events
grep "FLIP_MARKER" braine_viz_metrics.log

# Find all recovery events
grep "FLIP_RECOVERY" braine_viz_metrics.log

# Extract learning signals
grep "reinforce" braine_viz_metrics.log | tail -20
```

### Key Log Fields
- `frame`: Simulation frame number
- `game`: Which game produced the event
- `mode`: Human or Braine control
- `reward`: Last reward signal
- `recent*_before`: Performance before flip
- `recent*_after`: Performance after recovery
- `outcomes_since_flip`: Learning speed metric
