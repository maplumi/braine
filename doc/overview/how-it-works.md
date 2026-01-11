# How Braine Works

This document provides a comprehensive explanation of how the braine cognitive substrate operates, with special attention to neurogenesis and the learning mechanisms.

## Table of Contents
1. [Overview: What is Braine?](#overview-what-is-braine)
2. [Core Concepts](#core-concepts)
3. [The Brain Substrate](#the-brain-substrate)
4. [How Learning Happens](#how-learning-happens)
5. [Neurogenesis: Growing New Capacity](#neurogenesis-growing-new-capacity)
6. [Complete Learning Cycle](#complete-learning-cycle)
7. [Practical Examples](#practical-examples)

---

## Overview: What is Braine?

Braine is a **brain-like cognitive substrate** designed for continuous learning from interaction. Unlike neural networks trained with backpropagation or large language models that predict text, braine:

- **Learns online**: Updates happen immediately from experience, not from batch training
- **Uses local rules**: No global optimization or gradients—learning is purely local
- **Grows dynamically**: Can add new capacity (neurogenesis) when needed
- **Forgets structurally**: Prunes unused connections to free capacity
- **Runs on edge devices**: Designed for low-power, bounded-memory deployment

Think of it as a **continuously-running dynamical system** where:
- **Memory = Structure**: What the system "knows" is encoded in connection weights and oscillatory patterns
- **Learning = Local plasticity**: Connections strengthen when units are co-active during reward
- **Adaptation = Structural change**: The network physically grows and prunes over time

---

## Core Concepts

### Units

The fundamental building block is a **Unit**—a simple oscillator with just four scalar values:

```rust
struct Unit {
    amp: f32,      // Amplitude (activity level, 0.0-2.0 typical)
    phase: f32,    // Phase angle (-π to +π)
    bias: f32,     // Baseline excitability
    decay: f32,    // How quickly activity fades
}
```

Each unit is like a tiny neuron that:
- Oscillates continuously (has amplitude and phase)
- Can be excited by inputs (sensors, other units)
- Decays back to baseline without input
- Has an intrinsic bias (some units are naturally more excitable)

### Connections

Units influence each other through **sparse directed connections**:

```
Unit A ---[weight=0.5]---> Unit B
```

When Unit A is active, it pushes Unit B's amplitude up or down based on:
- The connection **weight** (stronger = more influence)
- Unit A's current **amplitude** (more active = more influence)
- Their relative **phase alignment** (in-phase = constructive, out-of-phase = destructive)

The entire network is stored in **CSR (Compressed Sparse Row)** format for efficiency:
- Most units connect to only a small fraction of others
- Pruned connections are marked as invalid but not removed immediately
- Periodic compaction rebuilds the CSR to reclaim memory

### Sensors and Actions

The brain communicates with the outside world through:

**Sensors**: Named groups of units that receive external stimuli
```rust
brain.define_sensor("vision", 8);  // 8 units for vision
brain.apply_stimulus(Stimulus { name: "vision", strength: 0.8 });
```

**Actions**: Named groups whose activity determines behavior
```rust
brain.define_action("move_left", 4);
brain.define_action("move_right", 4);
let (action, score) = brain.select_action(&["move_left", "move_right"]);
```

### Neuromodulator (Reward)

A single scalar value in `[-1, +1]` that represents:
- Positive reward (+1.0) = "that was good, strengthen what just happened"
- Negative reward (-1.0) = "that was bad, weaken what just happened"  
- Neutral (0.0) = "no signal, decay normally"

This is the brain's only global signal. Everything else is local.

---

## The Brain Substrate

### Dynamics: How the Brain "Thinks"

Every step, each unit updates its amplitude and phase based on:

1. **Neighbor influence**: Sum of inputs from connected units
   ```
   input = Σ (weight[i→j] × amp[i] × cos(phase[i] - phase[j]))
   ```

2. **Bias**: Intrinsic excitability
   ```
   input += bias[j]
   ```

3. **External stimulus**: If unit j is in a sensor group receiving input
   ```
   input += pending_input[j]
   ```

4. **Global inhibition**: Competition for activity
   ```
   inhibition = avg(all amplitudes) × inhibition_strength
   input -= inhibition
   ```

5. **Noise**: Small random perturbations for exploration
   ```
   input += random(-0.01, +0.01)
   ```

6. **Decay**: Fade toward zero
   ```
   amp[j] = amp[j] × (1 - decay[j])
   ```

The final update:
```rust
amp[j] += input × dt
amp[j] = clamp(amp[j], -2.0, +2.0)
phase[j] += (base_freq + amp[j] × freq_coupling) × dt
```

This creates **rich recurrent dynamics**:
- Stable patterns (attractors) emerge from repeated experiences
- Partial cues can trigger complete patterns (content-addressable memory)
- Different contexts produce different activity patterns
- The system never stops—it's always "thinking" even without input

### Action Selection: How the Brain "Decides"

To choose an action, the brain:

1. **Reads out** average amplitude from each action group:
   ```rust
   for each action_group:
       score = average amplitude of units in group
   ```

2. **Adds noise** for exploration:
   ```rust
   score += random_gaussian(0, exploration_noise)
   ```

3. **Picks the winner**:
   ```rust
   selected_action = action with highest score
   ```

Optionally, it can bias selection using **causal memory** (see below) to prefer actions that historically led to reward in similar contexts.

---

## How Learning Happens

Braine uses **three-factor Hebbian learning**—a local rule that requires:

1. **Pre-synaptic activity**: Unit A is active (amp > threshold)
2. **Post-synaptic activity**: Unit B is active (amp > threshold)
3. **Neuromodulator**: Reward signal is present

### The Learning Rule

```rust
if amp[A] > coactive_threshold && amp[B] > coactive_threshold {
    // Check phase alignment
    phase_diff = phase[A] - phase[B]
    alignment = cos(phase_diff)  // 1.0 = in phase, -1.0 = out of phase
    
    if alignment > phase_lock_threshold {
        // Strengthen: units are co-active and in-phase
        Δweight = hebb_rate × neuromodulator × alignment
    } else {
        // Weaken slightly: anti-Hebbian for specialization
        Δweight = -hebb_rate × 0.05
    }
    
    weight[A→B] += Δweight
}
```

**Key insights**:
- Learning only happens when **both units are active** (local rule)
- **Phase alignment** matters: in-phase connections strengthen, out-of-phase weaken
- **Neuromodulator scales learning**: more reward = faster learning
- Connections that don't contribute get weakened slightly (anti-Hebb)

### Forgetting and Pruning

Every step, all connections decay:
```rust
weight[i→j] *= (1 - forget_rate)  // typical forget_rate = 0.0001
```

When a weight becomes tiny (`|weight| < prune_below`, typically 0.01), the connection is marked for pruning:
```rust
if abs(weight[i→j]) < 0.01 {
    target[i→j] = INVALID_UNIT  // Mark as pruned
    weight[i→j] = 0.0
}
```

Periodically, the CSR structure is compacted to remove pruned connections and reclaim memory.

**Why this matters**:
- **Bounded memory**: The network doesn't grow without limit
- **Relevance**: Recent/important connections are strong, old unused ones disappear
- **Energy efficiency**: Fewer connections = less computation
- **Capacity recycling**: Pruning makes room for new learning

### Causal Memory (Meaning)

Beyond connection weights, braine maintains a lightweight **causal memory** that tracks symbolic relationships:

```
(stimulus_symbol, action_symbol) → reward_symbol
```

For example, after many steps of seeing "red light" → taking "brake" → getting "reward", the system learns:
```
P(brake | red_light) = 0.7
P(reward | brake) = 0.8
```

This is stored as:
- **Symbol table**: strings → integer IDs
- **Directed edges**: (prev_symbol → current_symbol) with count
- **Co-occurrence counts**: how often symbols appear together

The causal memory is used to:
1. **Bias action selection**: prefer actions that led to reward in similar contexts
2. **Provide hints**: "in this context, this action tends to work"
3. **Guide neurogenesis**: grow units connected to important sensor groups

Importantly, causal memory is **bounded and decaying**:
- Counts decay continuously: `count *= 0.999` each step
- Near-zero entries are pruned periodically
- This prevents unbounded growth during long-running deployment

---

## Neurogenesis: Growing New Capacity

**Neurogenesis** is the process of adding new units to the network when existing capacity is saturated. This is one of braine's key mechanisms for **continuous learning without catastrophic forgetting**.

### Why Neurogenesis?

As the brain learns, connection weights grow stronger. Eventually, the network becomes **saturated**:
- Most weights are near their maximum
- New learning would require "overwriting" existing knowledge
- The system can't easily encode new concepts

Neurogenesis solves this by **adding fresh capacity**—new units with weak connections that can specialize for new concepts.

### How Saturation is Detected

The brain computes the average absolute weight:
```rust
avg_weight = Σ |weight[i→j]| / total_connections
```

When `avg_weight > saturation_threshold` (typically 0.4-0.6), neurogenesis is triggered.

**Intuition**: If most connections are strong, the network has "used up" its capacity for representing new patterns cleanly.

### Growing a New Unit

When `brain.grow_unit(connectivity)` is called:

1. **Create the unit**:
   ```rust
   new_unit = Unit {
       amp: 0.0,              // Start quiet
       phase: random(-π, π),  // Random phase
       bias: 0.05,            // Slightly excitable
       decay: 0.12,           // Standard decay
   }
   ```

2. **Wire outgoing connections** (new unit → existing units):
   ```rust
   for _ in 0..connectivity {
       target = random_unit()
       weight = random(0.05, 0.15)  // Small positive weights
       add_connection(new_unit → target, weight)
   }
   ```

3. **Wire incoming connections** (existing units → new unit):
   ```rust
   incoming_count = connectivity / 2
   for _ in 0..incoming_count {
       source = random_unit()
       weight = random(0.05, 0.15)
       add_connection(source → new_unit, weight)
   }
   ```

4. **Extend auxiliary arrays**:
   ```rust
   reserved.push(false)
   learning_enabled.push(true)
   pending_input.push(0.0)
   sensor_member.push(false)
   group_member.push(false)
   ```

5. **Update unit count**:
   ```rust
   unit_count += 1
   births_last_step += 1
   ```

### Automatic Neurogenesis

The `maybe_neurogenesis()` method automates this process:

```rust
pub fn maybe_neurogenesis(
    &mut self,
    saturation_threshold: f32,  // Grow when avg_weight > this (0.3-0.6)
    growth_count: usize,        // How many units to add (4-16 typical)
    max_units: usize,           // Never exceed this limit
) -> usize
```

**Logic**:
```rust
if unit_count >= max_units {
    return 0;  // At capacity limit
}

if !should_grow(saturation_threshold) {
    return 0;  // Not saturated yet
}

// Grow up to growth_count units, but respect max_units
let to_add = min(growth_count, max_units - unit_count)
grow_units(to_add, connectivity)
return to_add
```

### Targeted Neurogenesis

You can also grow units specifically connected to a sensor/action group:

```rust
pub fn neurogenesis_for_group(
    &mut self,
    group_name: &str,
    count: usize,
    incoming_per_unit: usize,
    outgoing_per_unit: usize,
) -> Vec<UnitId>
```

This creates units that:
- Receive input from the named group (incoming connections)
- Project back to the group (outgoing connections)
- Start with small random weights

**Use case**: When you add a new sensor modality (e.g., a new camera), grow units specifically to process that input.

### Integration and Learning

Newly-grown units start **weak and quiet**:
- Amplitude near zero
- Small connection weights
- Not part of any existing attractors

They integrate into the network through **normal learning**:
1. Random activity occasionally activates them
2. If they happen to be active during reward, their connections strengthen (Hebbian learning)
3. Over time, some specialize for useful patterns, others remain dormant
4. Dormant units can be pruned if they never become useful

**Key insight**: Neurogenesis provides **capacity**, but new units still need **other learning mechanisms** to be useful quickly:
- **Imprinting**: One-shot concept formation
- **Burst learning**: Flashbulb memory for dramatic events
- **Dream replay**: Offline consolidation
- **Forced synchronization**: Teacher-guided learning
- **Attention gating**: Focus learning on most active units

### Neurogenesis + Pruning Cycle

In a long-running deployment, the network goes through cycles:

```
1. Network is sparse, weights small → Normal learning
2. Weights grow as patterns are learned → Saturation increases
3. avg_weight > threshold → Neurogenesis adds units
4. New units integrate via learning → Some specialize, some don't
5. Unused connections decay → Pruning removes weak links
6. Return to step 1 with updated structure
```

This creates a **self-maintaining system**:
- Capacity expands when needed (neurogenesis)
- Irrelevant structure is removed (pruning)
- Memory footprint stays bounded (max_units limit + pruning)
- Knowledge persists as stable attractors in active connections

### Example: Saturation → Growth → Integration

**Initial state** (32 units, avg_weight = 0.2):
```
Unit connections: ~128 (4 per unit)
Avg weight: 0.2
Saturation: NO
```

**After 1000 learning steps** (avg_weight = 0.55):
```
Unit connections: ~128
Avg weight: 0.55  ← SATURATED!
Action: Trigger neurogenesis
```

**Grow 8 new units**:
```
Old units: 32 (avg_weight = 0.55)
New units: 8 (avg_weight = 0.1, connections to random existing units)
Total units: 40
Overall avg_weight: 0.46  ← Below threshold again
```

**After 500 more learning steps**:
```
New units have specialized:
  - Units 32-35: learned to respond to "food" stimulus
  - Units 36-37: learned to respond to "threat" stimulus
  - Units 38-39: remain mostly dormant
  
Overall avg_weight: 0.42 (still below threshold)
```

**After 10000 more steps**:
```
Dormant units 38-39 pruned (never became useful)
Active units 32-37 integrated into attractors
Network has 38 units, ready for next growth if needed
```

---

## Complete Learning Cycle

Here's how all the pieces fit together in a typical interaction loop:

### 1. Perception (Stimulus)

```rust
// Environment provides input
brain.apply_stimulus(Stimulus { name: "vision_food", strength: 1.0 });
```

This injects current into the "vision_food" sensor group, exciting those units.

### 2. Dynamics (Step)

```rust
brain.step();
```

All units update based on:
- Neighbor connections
- External stimulus
- Intrinsic bias and decay
- Global competition
- Random noise

The network settles into a pattern shaped by the stimulus and its learned attractors.

### 3. Action Selection

```rust
let (action, score) = brain.select_action(&["approach", "avoid"]);
```

Read out activity from action groups, add exploration noise, pick winner.

### 4. Outcome (Reward)

```rust
// Environment provides feedback
brain.set_neuromodulator(0.8);  // Good outcome!
```

The neuromodulator is set based on whether the action was beneficial.

### 5. Symbol Recording

```rust
brain.note_action(action.clone());
brain.commit_observation();
```

Record the (stimulus, action, reward) triple into causal memory for future reference.

### 6. Learning (Automatic)

During the next `step()`, the learning rule applies:
- Co-active, phase-aligned units strengthen (scaled by neuromodulator)
- All connections decay slightly
- Weak connections are marked for pruning

### 7. Structural Adaptation (Periodic)

Every N steps (e.g., 1000):

```rust
// Check for saturation
let grown = brain.maybe_neurogenesis(0.5, 8, 512);
if grown > 0 {
    println!("Grew {} new units due to saturation", grown);
}

// Compact pruned connections
if step % 1000 == 0 {
    brain.rebuild_csr();
}

// Dream consolidation
if step % 5000 == 0 {
    brain.dream(100, 5.0, 3.0);  // 100 steps, 5× learning, 3× noise
}
```

### 8. Persistence (Save/Load)

```rust
// Save current state
brain.save("braine.bbi")?;

// Later, restore
let brain = Brain::load("braine.bbi")?;
```

The entire brain state (units, connections, symbols, causal memory) is serialized to disk. On reload, the system picks up exactly where it left off.

---

## Practical Examples

### Example 1: One-Shot Association

**Goal**: After seeing "red light" once with "brake" action and positive reward, associate them.

```rust
// Initial brain with no prior knowledge
let mut brain = Brain::new(BrainConfig::with_size(64, 4).with_seed(42));
brain.define_sensor("red_light", 4);
brain.define_action("brake", 4);

// Trial 1: Present stimulus, force action, reward
brain.apply_stimulus(Stimulus { name: "red_light", strength: 1.0 });
brain.step();
brain.note_action("brake");
brain.set_neuromodulator(1.0);  // Strong positive reward
brain.step();  // Learning happens here
brain.commit_observation();

// Trial 2: Present stimulus again, check if "brake" is selected
brain.apply_stimulus(Stimulus { name: "red_light", strength: 1.0 });
brain.step();
let (action, score) = brain.select_action(&["brake", "accelerate"]);

// Result: action is likely "brake" (not guaranteed due to noise, but probable)
```

**What happened**:
- During Trial 1's `step()`, units in "red_light" sensor and "brake" action that were co-active and phase-aligned had their connections strengthened
- The neuromodulator (1.0) scaled this learning strongly
- During Trial 2, the "red_light" stimulus reactivates a similar pattern, which now has stronger connections to "brake" units
- Action selection reads higher average amplitude from "brake" → selects it

### Example 2: Forgetting and Re-Learning

**Goal**: Show that unused associations decay, but re-learning is faster (savings).

```rust
// Learn "A" → "X"
for _ in 0..10 {
    brain.apply_stimulus(Stimulus { name: "A", strength: 1.0 });
    brain.step();
    brain.note_action("X");
    brain.set_neuromodulator(0.5);
    brain.step();
    brain.commit_observation();
}

// Now "A" reliably triggers "X"
brain.apply_stimulus(Stimulus { name: "A", strength: 1.0 });
brain.step();
let (action, _) = brain.select_action(&["X", "Y"]);
assert_eq!(action, "X");

// Run 10000 idle steps (no stimulus, no reward)
for _ in 0..10000 {
    brain.step();
}

// Connection weights have decayed
// Retry: "A" may no longer reliably select "X"
brain.apply_stimulus(Stimulus { name: "A", strength: 1.0 });
brain.step();
let (action, _) = brain.select_action(&["X", "Y"]);
// action might be "Y" now (or "X" with lower confidence)

// Re-learn: only takes 2-3 trials instead of 10 (savings effect)
for _ in 0..3 {
    brain.apply_stimulus(Stimulus { name: "A", strength: 1.0 });
    brain.step();
    brain.note_action("X");
    brain.set_neuromodulator(0.5);
    brain.step();
    brain.commit_observation();
}

// Now "A" → "X" is strong again
```

### Example 3: Neurogenesis in Action

**Goal**: Saturate the network, trigger neurogenesis, verify new units integrate.

```rust
let mut brain = Brain::new(BrainConfig::with_size(32, 4).with_seed(42));
brain.define_sensor("input", 8);
brain.define_action("output", 8);

// Artificially saturate the network
for w in brain.connections.weights.iter_mut() {
    *w = 0.7;  // High saturation
}

let diag = brain.diagnostics();
println!("Before: {} units, avg_weight = {:.2}", 
         diag.unit_count, diag.avg_weight);
// Output: "Before: 32 units, avg_weight = 0.70"

// Trigger neurogenesis
let grown = brain.maybe_neurogenesis(0.5, 8, 100);
println!("Grew {} units", grown);
// Output: "Grew 8 units"

let diag = brain.diagnostics();
println!("After: {} units, avg_weight = {:.2}", 
         diag.unit_count, diag.avg_weight);
// Output: "After: 40 units, avg_weight = 0.56"

// New units start quiet
for id in 32..40 {
    let amp = brain.units[id].amp;
    println!("Unit {}: amp = {:.3}", id, amp);
}
// Output: "Unit 32: amp = 0.000", "Unit 33: amp = 0.000", ...

// Run learning for 1000 steps
for step in 0..1000 {
    brain.apply_stimulus(Stimulus { name: "input", strength: 0.5 });
    brain.step();
    let (action, _) = brain.select_action(&["output"]);
    brain.note_action(action);
    brain.set_neuromodulator(0.3);
    brain.step();
    brain.commit_observation();
}

// Check which new units have specialized
for id in 32..40 {
    let amp = brain.units[id].amp;
    if amp > 0.1 {
        println!("Unit {} became active (amp = {:.3})", id, amp);
    }
}
// Output might show: "Unit 34 became active (amp = 0.421)"
```

---

## Summary

**Braine** is a cognitive substrate that:

1. **Represents knowledge as structure**: Connection weights and oscillatory patterns
2. **Learns locally and continuously**: Three-factor Hebbian rule applied every step
3. **Grows when saturated**: Neurogenesis adds fresh capacity for new concepts
4. **Prunes when idle**: Structural forgetting frees capacity and maintains relevance
5. **Persists its state**: Can save/load the entire brain and continue learning

**Neurogenesis specifically**:
- Detects saturation by monitoring average connection weight
- Adds new units with random weak connections
- New units integrate through normal learning (Hebbian + other mechanisms)
- Works in tandem with pruning to maintain bounded, relevant capacity
- Enables continuous learning without catastrophic forgetting

**Key design principles**:
- **Local**: No global optimization, no backprop, no gradients
- **Bounded**: Fixed memory limits enforced by pruning and max_units
- **Embodied**: Sensors → Dynamics → Actions → Reward
- **Persistent**: State is the knowledge; save/load is first-class
- **Edge-first**: Designed for low-power, always-on deployment

For more details:
- Implementation: `src/core/substrate.rs`
- Architecture: `../architecture/architecture.md`
- Learning mechanisms: `../learning/accelerated-learning.md`
- Research context: `../research/research-landscape.md`
