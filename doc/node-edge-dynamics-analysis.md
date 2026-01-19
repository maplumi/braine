# Node Size, Edge Length, and Memory Dynamics: Research Analysis

## Current Braine Memory Mechanisms

### 1. Edge-Based Memory (Current Implementation)

**Plasticity (Eligibility + Neuromodulated Commit)**:
- Each step updates an **eligibility trace** per connection based on recent co-activity/alignment (this accumulates “credit” but does not change weights yet).
- Weight updates are **committed only when** neuromodulator magnitude exceeds a **deadband**: `|neuromod| > learning_deadband`.
- When committing, the update uses the eligibility trace and the **signed** neuromodulator (negative values drive LTD):
   - `Δw ∝ hebb_rate * neuromod * eligibility`
- Optional governance knobs:
   - **Plasticity budget** caps total per-step weight change (approximately an L1 cap on committed updates).
   - **Homeostasis** can periodically nudge unit biases toward a target amplitude (stability support, separate from synaptic learning).
- Weights remain clamped to `[-1.5, 1.5]`.

**Forgetting & Pruning** (`forget_and_prune`):
- All weights decay: `w *= (1 - forget_rate)` 
- Default forget_rate: 0.0005 (very slow decay)
- Pruning: if `|w| < prune_below` (default 0.01), connection is removed
- Engram protection: sensor↔concept edges maintain minimal trace (never fully pruned)

**Causal Memory** (`CausalMemory`):
- Separate symbolic memory tracking temporal co-occurrence
- Decay: `count *= (1 - causal_decay)` each observation
- Directed edges: from previous symbols → current symbols
- Co-occurrence edges: same-tick symbol pairs

### 2. Unit Structure (Current)

```rust
pub struct Unit {
    pub amp: Amplitude,   // Current activation level
    pub phase: Phase,     // Oscillation phase
    pub bias: f32,        // Resting activation tendency
    pub decay: f32,       // How fast amplitude decays
}
```

**Not currently tracked:**
- Access frequency / activation history (separate from salience)
- Connectivity density (in/out degree)

---

## Proposed Enhancements

### A. Node Size as Frequent Access Indicator

**Concept:** Larger nodes = more frequently accessed/activated units

**Implementation Options:**

1. **Activation Accumulator** (implemented)
   ```rust
   pub struct Unit {
       // ... existing fields ...
       pub salience: f32,  // Accumulated importance
   }
   ```
   - Update: `salience = (1-decay)*salience + gain*max(0, trace - threshold)`
   - Trace is an EMA of nonnegative amplitude (`activity_trace`)
   - Decay slowly over time to forget old importance
   - Visualize as node size

2. **Access Counter**
   - Simple counter incremented when `amp > threshold`
   - Pros: Simple, cheap
   - Cons: No decay, requires periodic normalization

3. **Exponential Moving Average of Activation**
   ```rust
   pub activation_ema: f32,  // EMA of amplitude
   // Update: activation_ema = alpha * amp + (1-alpha) * activation_ema
   ```

**Mathematical Model:**
$$\text{salience}_i(t+1) = (1 - \lambda_s) \cdot \text{salience}_i(t) + \alpha_s \cdot \max(0, \tilde{a}_i(t) - \theta)$$

Where:
- $\lambda_s$ = salience decay rate (e.g., 0.001)
- $\alpha_s$ = salience gain on activation (e.g., 0.1)
- $\theta$ = activation threshold
- $\tilde{a}_i(t)$ = slow activity trace (EMA of $\max(0,a_i)$)

---

### B. Edge Length as Relation Proximity

**Concept:** Shorter edges = closer/more related concepts

**Challenge:** Current edges have no explicit "length" - they're abstract connections with weights.

**Options:**

1. **Derive from Co-activation Frequency**
   - Track how often two connected units are co-active
   - Higher co-activation → shorter visual distance
   - Already implicit in weight magnitude

2. **Explicit Distance Metric** (new field)
   ```rust
   struct EdgeMetrics {
       weight: f32,           // Current: connection strength
       distance: f32,         // NEW: conceptual distance
       last_coactive: u64,    // NEW: when last co-activated
   }
   ```
   - Distance decreases when co-active, increases otherwise
   - Mathematical model:
   $$d_{ij}(t+1) = \begin{cases}
   d_{ij}(t) \cdot (1 - \beta) & \text{if } a_i > \theta \wedge a_j > \theta \\
   d_{ij}(t) \cdot (1 + \gamma) & \text{otherwise}
   \end{cases}$$

3. **Inverse Weight as Distance** (simplest, visualization-only)
   - `visual_distance = base_length / (1 + abs(weight))`
   - No core changes, just rendering

**Recommendation:** Start with Option 3 for visualization, consider Option 2 for advanced semantic clustering.

---

### C. Edge Thickness as Connection Strength (Already Implemented!)

The current visualization already does this:
```rust
let absw = (w.abs() / max_abs_w).clamp(0.0, 1.0) as f64;
ctx.set_line_width(0.5 + 1.6 * absw);
```

Stronger connections = thicker lines.

---

## Comprehensive Memory Model

Combining all three dimensions:

| Property | Meaning | Update Rule | Visual |
|----------|---------|-------------|--------|
| **Edge Weight** | Connection strength | Hebbian + forgetting | Thickness |
| **Edge Distance** | Conceptual proximity | Shrinks on co-activation | Length |
| **Node Salience** | Access frequency | Grows on activation | Size |

### Unified Update Equations

**Per Step:**

1. **Dynamics update** (existing)
2. **Learning update** (existing Hebbian)
3. **Salience update** (NEW):
   $$s_i \leftarrow (1 - \lambda_s) s_i + \alpha_s \cdot \max(0, a_i - \theta)$$

4. **Distance update** (NEW, optional):
   $$d_{ij} \leftarrow d_{ij} \cdot \begin{cases}
   (1 - \beta) & \text{if co-active} \\
   (1 + \gamma) & \text{otherwise}
   \end{cases}$$
   Clamped to $[d_{min}, d_{max}]$

5. **Forgetting** (existing weight decay + NEW distance growth)

---

## Pros and Cons Analysis

### Adding Node Salience

**Pros:**
- Captures "importance" beyond instantaneous amplitude
- Enables visualization of frequently-used concepts
- Could inform pruning decisions (don't prune high-salience nodes)
- Low computational cost (O(N) per step)

**Cons:**
- Adds 4 bytes per unit (minor memory increase)
- Another hyperparameter (salience decay rate)
- Needs tuning to balance responsiveness vs. stability

### Adding Edge Distance

**Pros:**
- Richer representation of semantic relationships
- Could enable graph-based clustering/reasoning
- Natural fit with embedding-space interpretations

**Cons:**
- Significant memory increase (4+ bytes per edge, edges >> units)
- O(E) computation per step
- May conflict with sparse pruning (distance metric on pruned edges?)
- Adds complexity to image format

### Mathematical Consistency

The proposed model maintains key properties:
1. **Locality**: Updates only depend on local state (no global gradients)
2. **Continuous forgetting**: All metrics decay without use
3. **Activity-dependent strengthening**: Aligns with biological Hebbian principles
4. **Bounded dynamics**: All values are clamped

---

## Gaps Identified

1. **No explicit "recency" tracking**: When was a connection last used?
   - Could add `last_active_step` to edges

2. **No differentiation between learning/inference state in visualization**
   - Current: same rendering regardless of mode
   - Should: color differently when `learning_enabled` vs not

3. **Causal memory is separate from substrate**
   - No visual bridge between unit graph and symbol graph
   - Causal view should show symbol↔symbol edges, not unit↔unit

4. **Phase information not visualized**
   - Units have phase but it's not shown
   - Could use color hue to indicate phase

---

## Recommended Implementation Path

### Phase 1: Visualization Enhancements (Low Risk)
1. ✅ Disable auto-rotate, enable manual rotation + drag to rotate
2. ✅ Add vibration based on amplitude (activity-based animation)
3. ✅ Color nodes by learning state (learning=warm, inference=cool palette)
4. ✅ Implement causal view (symbol graph with directed edges)
5. ✅ Controls hint: "Drag to rotate | Shift+drag to pan | Scroll to zoom"
6. ⏭️ Edge length from inverse weight (skipped - weight already encodes proximity)

### Phase 2: Node Salience (Medium Risk) ✅ IMPLEMENTED
1. ✅ Add `salience: f32` to Unit struct
2. ✅ Update during step: `s = (1-λ)*s + α*max(0, amp-θ)` with configurable decay/gain
3. ✅ Persist in BBI format (separate SALI chunk for backwards compatibility)
4. ✅ Use for visualization size (larger nodes = higher salience = more frequently accessed)
5. ✅ Added `salience_decay` and `salience_gain` to BrainConfig (defaults: 0.001, 0.1)

### Phase 3: Edge Distance (Higher Risk)
1. Design edge metadata structure
2. Benchmark memory impact
3. Implement distance dynamics
4. Integrate with pruning decisions

---

## Conclusion

**Yes, this makes mathematical and conceptual sense.**

The proposed model:
- **Node size ~ access frequency (salience)**: Reflects cumulative importance
- **Edge length ~ conceptual proximity**: Inversely related to co-activation
- **Edge thickness ~ connection strength**: Already implemented

**Recommended approach:**
1. Start with visualization-only changes (no core modifications)
2. Add node salience to core (low risk, high value)
3. Defer edge distance until memory/performance impact is evaluated

The key insight is that **edge weight already encodes both strength AND proximity** in the current model - stronger connections between frequently co-active units. Adding explicit distance would create redundancy unless it captures something weight doesn't (e.g., temporal recency vs. cumulative strength).
