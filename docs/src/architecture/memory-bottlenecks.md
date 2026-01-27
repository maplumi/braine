# Memory Bottleneck Analysis: Braine Core

**Document Version**: 1.0  
**Date**: 2026-01-11  
**Scope**: Core substrate (`crates/core`) memory usage and optimization opportunities

## Executive Summary

Braine is designed as an **edge-first, continuously-running cognitive substrate** with bounded memory requirements. This document analyzes potential memory bottlenecks in the core system and provides concrete recommendations for optimization.

### Key Findings

1. **CSR Connection Storage** is the dominant memory consumer, scaling O(N×K) where N=unit_count, K=connectivity_per_unit
2. **Causal Memory** can grow unbounded with symbol/edge accumulation without aggressive pruning
3. **Neurogenesis** lacks hard caps, allowing potential memory exhaustion
4. **Symbol Tables** use inefficient String storage with no deduplication
5. **Child Brain Spawning** creates full memory copies without structural sharing

### Implementation Status (repo-aligned)

This document mixes analysis with proposed changes. As of **2026-01-12**, the repo already implements some of the “recommended” mitigations, and a few others are intentionally deferred because they would change core dynamics, determinism, or persistence compatibility.

Implemented (no functional changes):
- **CSR tombstone reuse is already implemented**: when adding/bumping a connection, the code first reuses a tombstone slot inside the unit’s CSR segment before falling back to an expensive CSR insert/rebuild.
- **CSR tombstone tracking + threshold compaction**: compaction now triggers not only periodically but also when tombstones exceed a fraction of the CSR storage.
- **Neurogenesis capacity reservation**: `grow_units` now reserves capacity across the parallel arrays and CSR buffers up-front to reduce peak realloc spikes.
- **Causal base_total maintenance**: pruning now updates `base_total` incrementally during retain, avoiding an O(|base|) sum during pruning passes.

Deferred / needs explicit approval (changes behavior or compatibility):
- **Hard caps / eviction in causal memory**: any cap changes long-horizon retention. Even “keep strongest edges” introduces an algorithmic bias. This should be behind a config knob with a clear evaluation plan.
- **Sparse `pending_input`**: switching from `Vec<f32>` to a hash-map changes iteration order and can introduce small nondeterministic floating-point differences unless carefully controlled.
- **f16 / quantized weights**: changes numeric precision and therefore the dynamical regime; should be treated as a new execution mode and validated per-task.
- **Copy-on-write / structural sharing for child brains**: large architectural change that touches persistence, mutability boundaries, and expert consolidation semantics.

### Critical Bottlenecks (Priority Order)

| Component | Severity | Memory Impact | Mitigation Complexity |
|-----------|----------|---------------|----------------------|
| CSR Connections | **HIGH** | O(N×K×8 bytes) | Medium |
| Causal Memory Edges | **HIGH** | Unbounded HashMap growth | Low |
| Neurogenesis | **MEDIUM** | Unbounded unit growth | Low |
| Symbol Tables | **MEDIUM** | O(symbols×string_len) | Medium |
| Child Brains | **MEDIUM** | Full state duplication | High |
| Telemetry Buffers | **LOW** | Small fixed overhead | Low |

---

## 1. CSR Connection Storage

### Current Implementation

The CSR (Compressed Sparse Row) format stores connections as three parallel arrays:

```rust
pub struct CsrConnections {
    pub targets: Vec<UnitId>,      // O(N×K) × sizeof(usize) = ~8 bytes each
    pub weights: Vec<Weight>,       // O(N×K) × sizeof(f32) = 4 bytes each
    pub offsets: Vec<usize>,        // O(N+1) × sizeof(usize) = ~8 bytes each
}
```

### Memory Footprint Analysis

For default configuration (256 units, 12 connections/unit):
- `targets`: 256 × 12 × 8 = **24,576 bytes**
- `weights`: 256 × 12 × 4 = **12,288 bytes**
- `offsets`: 257 × 8 = **2,056 bytes**
- **Total**: ~38 KB

For larger configurations (16K units, 64 connections/unit):
- `targets`: 16,384 × 64 × 8 = **8,388,608 bytes** (~8 MB)
- `weights`: 16,384 × 64 × 4 = **4,194,304 bytes** (~4 MB)
- `offsets`: 16,385 × 8 = **131,080 bytes** (~128 KB)
- **Total**: ~12.5 MB

### Bottleneck Analysis

#### 1.1. Tombstoning Memory Leak

**Issue**: Pruned connections are marked with `INVALID_UNIT` sentinel but storage is not reclaimed until periodic compaction.

**Location**: `substrate.rs:4407-4413`
```rust
if abs < prune_below {
    self.connections.targets[idx] = INVALID_UNIT;  // Tombstone, not freed
    self.connections.weights[idx] = 0.0;
    self.pruned_last_step += 1;
}
```

**Impact**: 
- Between compactions (every 1000 steps), tombstones accumulate
- Worst case: all connections pruned but vectors remain full size
- Memory waste: up to 100% of CSR storage

**Recommendation**:
```rust
// Option A: More frequent compaction (trade CPU for memory)
if self.age_steps % 100 == 0 {  // Compact every 100 steps instead of 1000
    self.compact_connections();
}

// Option B: Lazy compaction on memory pressure
if self.pruned_last_step > (self.connections.targets.len() / 10) {
    self.compact_connections();
}

// Option C: Track tombstone count and trigger threshold-based compaction
struct CsrConnections {
    tombstone_count: usize,  // Add field
    // ... existing fields
}

fn compact_if_needed(&mut self) {
    if self.connections.tombstone_count > self.connections.targets.len() / 4 {
        self.compact_connections();
        self.connections.tombstone_count = 0;
    }
}
```

#### 1.2. CSR Append Inefficiency

**Issue**: Adding new connections requires full CSR rebuild via `insert()`, causing O(N) memory reallocation.

**Location**: `substrate.rs:836-847`
```rust
fn append_connection(&mut self, from: UnitId, target: UnitId, weight: f32) {
    let insert_pos = self.connections.offsets[from + 1];
    self.connections.targets.insert(insert_pos, target);  // O(N) operation
    self.connections.weights.insert(insert_pos, weight);
    
    for i in (from + 1)..self.connections.offsets.len() {
        self.connections.offsets[i] += 1;  // Update all subsequent offsets
    }
}
```

**Impact**:
- Every new connection after tombstone exhaustion triggers array shifts
- Worst case: O(N²) for K sequential appends
- Memory spike during reallocation

**Recommendation**:
```rust
// Pre-allocate extra capacity with growth factor
impl CsrConnections {
    fn with_capacity(units: usize, conns_per_unit: usize) -> Self {
        let capacity = (units * conns_per_unit * 3) / 2;  // 50% overallocation
        Self {
            targets: Vec::with_capacity(capacity),
            weights: Vec::with_capacity(capacity),
            offsets: Vec::with_capacity(units + 1),
        }
    }
}

// Amortized append with batching
struct PendingConnection {
    from: UnitId,
    target: UnitId,
    weight: f32,
}

impl Brain {
    pending_connections: Vec<PendingConnection>,
    
    fn flush_pending_connections(&mut self) {
        if self.pending_connections.is_empty() { return; }
        
        // Sort by 'from' to minimize CSR rebuilds
        self.pending_connections.sort_by_key(|c| c.from);
        
        // Batch insert all connections in a single CSR rebuild
        // ... implementation details
    }
}
```

#### 1.3. Weight Precision Overhead

**Issue**: Using `f32` (4 bytes) for weights when many applications could use lower precision.

**Current**: `pub type Weight = f32;`

**Recommendation**:
```rust
// Option A: Use f16 (half precision) where supported
#[cfg(feature = "f16_weights")]
pub type Weight = half::f16;  // 2 bytes instead of 4
#[cfg(not(feature = "f16_weights"))]
pub type Weight = f32;

// Memory savings: 50% on weights array
// For 16K units × 64 conn: 4MB → 2MB

// Option B: Quantized weights with fixed-point arithmetic
pub type Weight = i8;  // 1 byte, range -128..127 → scale to -1.5..1.5

// Memory savings: 75% on weights array
// For 16K units × 64 conn: 4MB → 1MB
```

**Trade-offs**:
- **f16**: Good balance, hardware support on modern ARM/x86, ~2-3 decimal digits precision
- **i8**: Maximum memory savings, requires careful scaling, may lose precision on small updates

---

## 2. Causal Memory

### Current Implementation

Causal memory uses HashMap-based storage for symbols and edges:

```rust
pub struct CausalMemory {
    edges: HashMap<u64, EdgeStats>,     // Unbounded edge map
    base: HashMap<SymbolId, f32>,       // Unbounded symbol map
    base_total: f32,
    prev_symbols: Vec<SymbolId>,
    // ...
}
```

### Memory Footprint Analysis

**Per-edge overhead**:
- HashMap entry: ~24 bytes (key + value + metadata)
- `u64` key: 8 bytes (packed from/to SymbolIds)
- `EdgeStats`: 4 bytes (f32 count)
- **Total**: ~36 bytes per edge

**Growth characteristics**:
- Directed edges: O(|prev_symbols| × |current_symbols|) per observation
- Co-occurrence edges: O(|current_symbols|²) per observation
- Without aggressive pruning: unbounded growth

**Example scenario**:
- 1000 unique symbols
- Fully connected graph: 1000 × 1000 = 1M edges
- Memory: 1M × 36 bytes = **36 MB**

### Bottleneck Analysis

#### 2.1. Unbounded Edge Growth

**Issue**: Edges accumulate without hard limits, only soft pruning every 256 observations.

**Location**: `causality.rs:104-110`
```rust
if (self.observe_count & 0xFF) == 0 {  // Every 256 observations
    let thr = 0.001;
    self.base.retain(|_, v| *v > thr);
    self.edges.retain(|_, e| e.count > thr);
    self.base_total = self.base.values().sum::<f32>();
}
```

**Impact**:
- Between pruning cycles, edges can grow unbounded
- Decay rate alone may not prevent accumulation
- Long-running systems risk OOM

**Recommendation**:
```rust
// Option A: Add hard caps with LRU-style eviction
const MAX_EDGES: usize = 100_000;  // Configurable limit
const MAX_SYMBOLS: usize = 10_000;

impl CausalMemory {
    fn prune_to_capacity(&mut self) {
        if self.edges.len() > MAX_EDGES {
            // Keep only strongest edges
            let mut sorted: Vec<_> = self.edges.iter()
                .map(|(k, v)| (*k, v.count))
                .collect();
            sorted.sort_by(|a, b| b.1.total_cmp(&a.1));
            sorted.truncate(MAX_EDGES);
            
            let keep_set: HashSet<u64> = sorted.iter().map(|(k, _)| *k).collect();
            self.edges.retain(|k, _| keep_set.contains(k));
        }
    }
}

// Option B: Adaptive pruning threshold
fn adaptive_prune_threshold(&self) -> f32 {
    if self.edges.len() < MAX_EDGES / 2 {
        0.001  // Lenient when under capacity
    } else {
        // Increase threshold as we approach limit
        let ratio = self.edges.len() as f32 / MAX_EDGES as f32;
        0.001 + (ratio - 0.5) * 0.01  // 0.001 → 0.006 as we hit limit
    }
}

// Option C: Probabilistic sampling (reservoir sampling)
fn should_record_edge(&mut self, key: u64) -> bool {
    if self.edges.len() < MAX_EDGES {
        true  // Always record when under capacity
    } else {
        // Replace random edge with probability 1/N
        self.rng.gen_range_f32(0.0, 1.0) < (MAX_EDGES as f32 / self.edges.len() as f32)
    }
}
```

#### 2.2. HashMap Overhead

**Issue**: HashMap has significant per-entry overhead (~24 bytes metadata) for storing simple counters.

**Recommendation**:
```rust
// Option A: Use more compact data structures for hot paths
use hashbrown::HashMap;  // Already used in no_std, more compact than std

// Option B: Use integer map for symbol IDs
use fxhash::FxHashMap;  // Faster hashing for integer keys

// Option C: Use bloom filter for edge existence checks
struct CausalMemory {
    edge_bloom: BloomFilter,  // Quick existence check
    edges: HashMap<u64, EdgeStats>,  // Actual storage
}

fn has_edge_fast(&self, from: SymbolId, to: SymbolId) -> bool {
    let key = pack(from, to);
    self.edge_bloom.contains(&key)  // Fast negative lookup
}
```

#### 2.3. base_total Recomputation

**Issue**: On every 256th observation, `base_total` is recomputed by summing all base counts.

**Location**: `causality.rs:109`
```rust
self.base_total = self.base.values().sum::<f32>();  // O(|symbols|)
```

**Impact**: O(N) operation during pruning, blocks observation loop

**Recommendation**:
```rust
// Already implemented correctly: incremental maintenance
// Just verify pruning maintains invariant:

fn prune_symbols(&mut self, threshold: f32) {
    let mut removed_total = 0.0;
    self.base.retain(|_, v| {
        if *v <= threshold {
            removed_total += *v;
            false
        } else {
            true
        }
    });
    self.base_total -= removed_total;  // Maintain invariant
}
```

---

## 3. Neurogenesis Memory Growth

### Current Implementation

Neurogenesis allows unbounded unit growth:

```rust
pub fn grow_unit(&mut self, conns_per_unit: usize) -> UnitId {
    let new_id = self.units.len();
    self.units.push(/* ... */);
    self.reserved.push(false);
    self.learning_enabled.push(true);
    // ... more Vec::push operations
}

pub fn maybe_neurogenesis(&mut self, threshold: f32, grow_count: usize, max_units: usize) -> usize {
    if self.units.len() >= max_units {
        return 0;  // Respects max_units
    }
    // ... grow up to grow_count units
}
```

### Bottleneck Analysis

#### 3.1. No Global Memory Budget

**Issue**: `max_units` is checked per call but there's no global memory budget or system-wide limit.

**Location**: `substrate.rs:5255-5260`
```rust
let grown = brain.maybe_neurogenesis(0.5, 100, 20);
assert_eq!(grown, 4, "Should only grow up to max_units");
```

**Impact**:
- Application code must manually track memory
- No protection against gradual growth from multiple sources
- Risk: slow memory leak via neurogenesis

**Recommendation**:
```rust
// Add global memory budget tracking
pub struct Brain {
    cfg: BrainConfig,
    units: Vec<Unit>,
    // ... existing fields
    
    memory_budget: MemoryBudget,  // New field
}

pub struct MemoryBudget {
    max_bytes: usize,
    current_bytes: usize,
}

impl Brain {
    pub fn set_memory_budget(&mut self, max_bytes: usize) {
        self.memory_budget = MemoryBudget {
            max_bytes,
            current_bytes: self.estimate_memory_usage(),
        };
    }
    
    pub fn estimate_memory_usage(&self) -> usize {
        let units_size = self.units.len() * core::mem::size_of::<Unit>();
        let conns_size = self.connections.targets.len() * 
            (core::mem::size_of::<UnitId>() + core::mem::size_of::<Weight>());
        let offsets_size = self.connections.offsets.len() * core::mem::size_of::<usize>();
        let symbols_size = self.symbols_rev.iter().map(|s| s.len()).sum::<usize>();
        let causal_size = self.causal.edges.len() * 36; // Estimate
        
        units_size + conns_size + offsets_size + symbols_size + causal_size
    }
    
    fn can_grow_units(&self, count: usize) -> bool {
        let estimated_growth = count * (
            core::mem::size_of::<Unit>() +
            self.cfg.connectivity_per_unit * 12  // CSR overhead
        );
        
        self.memory_budget.current_bytes + estimated_growth <= self.memory_budget.max_bytes
    }
    
    pub fn grow_unit(&mut self, conns_per_unit: usize) -> UnitId {
        if !self.can_grow_units(1) {
            panic!("Memory budget exceeded");
        }
        // ... existing implementation
        self.memory_budget.current_bytes = self.estimate_memory_usage();
    }
}
```

#### 3.2. Vector Reallocation Spikes

**Issue**: Growing units triggers multiple vector reallocations for parallel arrays.

**Location**: `substrate.rs:3763-3773`
```rust
pub fn grow_unit(&mut self, conns_per_unit: usize) -> UnitId {
    self.units.push(/* ... */);           // Realloc 1
    self.reserved.push(false);             // Realloc 2
    self.learning_enabled.push(true);      // Realloc 3
    self.sensor_member.push(false);        // Realloc 4
    self.group_member.push(false);         // Realloc 5
    self.pending_input.push(0.0);          // Realloc 6
    // ... 6 separate reallocations!
}
```

**Impact**: Memory spikes up to 2× during Vec growth

**Recommendation**:
```rust
// Pre-reserve capacity in batches
pub fn grow_units(&mut self, count: usize, conns_per_unit: usize) -> Range<UnitId> {
    let start = self.units.len();
    
    // Reserve all capacity upfront
    self.units.reserve(count);
    self.reserved.reserve(count);
    self.learning_enabled.reserve(count);
    self.sensor_member.reserve(count);
    self.group_member.reserve(count);
    self.pending_input.reserve(count);
    
    for _ in 0..count {
        self.grow_unit_impl(conns_per_unit);  // Push without realloc
    }
    
    start..(start + count)
}

// Or use struct-of-arrays wrapper with single allocation
struct UnitArrays {
    count: usize,
    capacity: usize,
    data: Vec<u8>,  // Single flat allocation
}

impl UnitArrays {
    fn units(&self) -> &[Unit] { /* ... */ }
    fn reserved(&self) -> &[bool] { /* ... */ }
    // ... accessors with pointer arithmetic
}
```

---

## 4. Symbol Table Memory

### Current Implementation

```rust
pub struct Brain {
    symbols: HashMap<String, SymbolId>,   // String → ID
    symbols_rev: Vec<String>,              // ID → String
    // ...
}
```

### Bottleneck Analysis

#### 4.1. String Duplication

**Issue**: Symbol names are stored twice (in HashMap key and Vec), with full String allocation per symbol.

**Memory overhead**:
- HashMap entry: ~24 bytes overhead
- String in map: 24 bytes (ptr, len, cap) + heap allocation
- String in vec: 24 bytes + heap allocation
- **Total**: ~48 bytes + 2× heap allocations per symbol

**Example**:
- 1000 symbols, avg 20 chars/name
- HashMap: 1000 × (24 + 24 + 20) = 68 KB
- Vec: 1000 × (24 + 20) = 44 KB
- **Total**: 112 KB for 1000 symbols

**Recommendation**:
```rust
// Option A: Use string interning with Rc<str>
use std::rc::Rc;

pub struct Brain {
    symbols: HashMap<Rc<str>, SymbolId>,
    symbols_rev: Vec<Rc<str>>,  // Shared ownership, single allocation
}

fn intern_symbol(
    map: &mut HashMap<Rc<str>, SymbolId>,
    rev: &mut Vec<Rc<str>>,
    name: &str,
) -> SymbolId {
    if let Some(&id) = map.get(name) {
        return id;
    }
    let id = rev.len() as SymbolId;
    let shared: Rc<str> = Rc::from(name);
    rev.push(Rc::clone(&shared));
    map.insert(shared, id);
    id
}

// Memory savings: 50% (single allocation per symbol)

// Option B: Use a string arena with &'static str (unsafe)
struct StringArena {
    storage: Vec<u8>,
    offsets: Vec<usize>,
}

impl StringArena {
    fn intern(&mut self, s: &str) -> &'static str {
        let offset = self.storage.len();
        self.storage.extend_from_slice(s.as_bytes());
        self.offsets.push(offset);
        
        // SAFETY: Arena is never deallocated during Brain lifetime
        unsafe {
            std::str::from_utf8_unchecked(
                std::slice::from_raw_parts(
                    self.storage.as_ptr().add(offset),
                    s.len()
                )
            )
        }
    }
}

// Memory savings: 75% (single arena allocation, no HashMap overhead)

// Option C: Use fixed-size symbol table with numeric IDs only
const MAX_SYMBOLS: usize = 65536;  // Fits in u16
type SymbolId = u16;

pub struct SymbolTable {
    names: Vec<String>,  // ID → Name lookup only
    lookup: HashMap<String, SymbolId>,  // Name → ID (lazily populated)
}

// Memory: ~100KB for 1000 symbols (vs 112KB, minimal savings but simpler)
```

---

## 5. Child Brain Spawning

### Current Implementation

Child brains are created by cloning the parent:

```rust
pub fn spawn_child(&self, seed: u64, overrides: ChildConfigOverrides) -> Brain {
    let mut child = self.clone();  // Full deep copy!
    // ... apply overrides
    child
}
```

### Bottleneck Analysis

#### 5.1. Full State Duplication

**Issue**: Every child brain gets a complete copy of all parent state.

**Memory multiplication**:
- 1 parent (16K units): ~12.5 MB
- 10 children: 10 × 12.5 MB = **125 MB**
- 100 children: 1.25 **GB**

**What's duplicated**:
- All units (amp, phase, bias, decay, salience)
- All connections (CSR arrays)
- All symbols and causal memory
- All group definitions

**Recommendation**:
```rust
// Option A: Copy-on-write for connections
use std::sync::Arc;

pub struct CsrConnections {
    targets: Arc<Vec<UnitId>>,     // Shared until modified
    weights: Arc<Vec<Weight>>,     // Copy-on-write
    offsets: Arc<Vec<usize>>,
}

impl CsrConnections {
    fn make_mut_targets(&mut self) -> &mut Vec<UnitId> {
        Arc::make_mut(&mut self.targets)  // Clone only on write
    }
}

// Option B: Structural sharing via parent pointer
pub struct ChildBrain {
    parent_ref: Weak<Brain>,  // Reference to parent
    delta_units: Vec<Unit>,   // Only changed units
    delta_weights: HashMap<usize, Weight>,  // Only changed weights
}

impl ChildBrain {
    fn get_unit(&self, id: UnitId) -> &Unit {
        self.delta_units.get(id)
            .or_else(|| self.parent_ref.upgrade()?.units.get(id))
    }
}

// Option C: Lazy cloning with page-based COW
const PAGE_SIZE: usize = 256;

pub struct PagedUnits {
    pages: Vec<Arc<[Unit; PAGE_SIZE]>>,
}

impl PagedUnits {
    fn get_mut(&mut self, id: UnitId) -> &mut Unit {
        let page_idx = id / PAGE_SIZE;
        let page = Arc::make_mut(&mut self.pages[page_idx]);  // Clone page only
        &mut page[id % PAGE_SIZE]
    }
}

// Memory savings: 
// - Option A: ~50-90% if child changes are sparse
// - Option B: ~80-95% for read-heavy children
// - Option C: ~70-85% with moderate changes
```

#### 5.2. Redundant Symbol Tables

**Issue**: Each child has a full copy of symbol tables even though symbols are rarely added.

**Recommendation**:
```rust
// Share symbol tables across family trees
pub struct SymbolRegistry {
    symbols: Arc<HashMap<String, SymbolId>>,
    symbols_rev: Arc<Vec<String>>,
}

pub struct Brain {
    symbol_registry: SymbolRegistry,  // Shared across parent/children
    local_symbols: Vec<SymbolId>,     // Child-specific symbols only
}

// Memory: O(1) per child instead of O(symbols)
```

---

## 6. Miscellaneous Bottlenecks

### 6.1. Telemetry Buffers

**Current**: `telemetry.last_stimuli`, `last_actions`, etc. are pre-allocated Vecs.

**Location**: `substrate.rs:884-896`
```rust
if self.telemetry.last_stimuli.capacity() < 8 {
    self.telemetry.last_stimuli.reserve(8);
}
```

**Impact**: Minimal (~64 bytes per brain) but unnecessary in production

**Recommendation**:
```rust
// Compile-time feature flag
#[cfg(feature = "telemetry")]
struct Telemetry { /* ... */ }

#[cfg(not(feature = "telemetry"))]
struct Telemetry;  // Zero-sized type

// Or runtime disable with lazy allocation
impl Telemetry {
    fn record_stimulus(&mut self, sym: SymbolId) {
        if !self.enabled { return; }
        if self.last_stimuli.is_empty() {
            self.last_stimuli = Vec::with_capacity(8);  // Lazy alloc
        }
        self.last_stimuli.push(sym);
    }
}
```

### 6.2. Pending Input Array

**Current**: `pending_input: Vec<f32>` is same size as units, mostly zeros.

**Location**: `substrate.rs:695`

**Recommendation**:
```rust
// Use sparse map for pending inputs
pending_input: HashMap<UnitId, f32>,  // Only store non-zero inputs

fn apply_stimulus(&mut self, stimulus: Stimulus<'_>) {
    for &id in group_units {
        self.pending_input.insert(id, strength);  // Sparse
    }
}

fn step(&mut self) {
    for (&id, &input) in &self.pending_input {
        self.units[id].amp += input;
    }
    self.pending_input.clear();  // No iteration over full array
}

// Memory: O(active_inputs) instead of O(units)
// For 16K units with 4 active: 4×12 = 48 bytes vs 64 KB
```

---

## 7. Prioritized Recommendations

### Immediate (Low-Hanging Fruit)

1. **Add memory budget tracking** (Section 3.1)
   - Prevents unbounded growth
   - Complexity: Low
   - Impact: High (safety)

2. **Increase pruning frequency for causal memory** (Section 2.1)
   - Change: `if (self.observe_count & 0xFF) == 0` → `if (self.observe_count & 0x3F) == 0`
   - Complexity: Trivial
   - Impact: Medium (prevents unbounded growth)

3. **Add tombstone tracking in CSR** (Section 1.1)
   - Triggers compaction when >25% tombstones
   - Complexity: Low
   - Impact: Medium (reduces memory waste)

4. **Disable telemetry by default** (Section 6.1)
   - Feature flag or lazy allocation
   - Complexity: Trivial
   - Impact: Low (but good hygiene)

### Short-Term (Medium Effort)

5. **Use sparse pending_input** (Section 6.2)
   - HashMap instead of Vec
   - Complexity: Medium
   - Impact: Medium (saves O(units) space)

6. **Intern symbols with Rc<str>** (Section 4.1)
   - Eliminate duplication
   - Complexity: Medium
   - Impact: Medium (50% savings on symbol tables)

7. **Add hard caps to causal memory** (Section 2.1, Option A)
   - `MAX_EDGES`, `MAX_SYMBOLS` with LRU eviction
   - Complexity: Medium
   - Impact: High (prevents OOM)

8. **Pre-reserve capacity in grow_units** (Section 3.2)
   - Eliminate reallocation spikes
   - Complexity: Low
   - Impact: Medium (reduces peak memory)

### Long-Term (High Effort)

9. **Implement f16 weights** (Section 1.3)
   - Requires feature flag, careful testing
   - Complexity: High
   - Impact: High (50% savings on largest structure)

10. **Copy-on-write for child brains** (Section 5.1, Option A)
    - Arc-based structural sharing
    - Complexity: High (API changes)
    - Impact: Very High (10× memory savings for children)

11. **Paged units with COW** (Section 5.1, Option C)
    - Fine-grained structural sharing
    - Complexity: Very High
    - Impact: Very High (optimal memory usage)

---

## 8. Memory Profiling Recommendations

### Add Runtime Memory Tracking

```rust
pub struct MemoryStats {
    pub units_bytes: usize,
    pub connections_bytes: usize,
    pub causal_bytes: usize,
    pub symbols_bytes: usize,
    pub total_bytes: usize,
    pub peak_bytes: usize,  // Track high-water mark
}

impl Brain {
    pub fn memory_stats(&self) -> MemoryStats {
        MemoryStats {
            units_bytes: self.units.len() * core::mem::size_of::<Unit>(),
            connections_bytes: self.connections.targets.capacity() * 8 +
                              self.connections.weights.capacity() * 4 +
                              self.connections.offsets.capacity() * 8,
            causal_bytes: self.causal.edges.capacity() * 36 +
                         self.causal.base.capacity() * 12,
            symbols_bytes: self.symbols_rev.iter().map(|s| s.capacity()).sum(),
            total_bytes: /* sum of above */,
            peak_bytes: self.peak_memory_usage,  // Track via allocator hook
        }
    }
}
```

### Add Allocation Hooks (Optional)

```rust
use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering};

static ALLOCATED: AtomicUsize = AtomicUsize::new(0);
static PEAK: AtomicUsize = AtomicUsize::new(0);

struct TrackingAllocator;

unsafe impl GlobalAlloc for TrackingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let size = layout.size();
        let old = ALLOCATED.fetch_add(size, Ordering::Relaxed);
        let new = old + size;
        
        let mut peak = PEAK.load(Ordering::Relaxed);
        while new > peak {
            match PEAK.compare_exchange_weak(peak, new, Ordering::Relaxed, Ordering::Relaxed) {
                Ok(_) => break,
                Err(p) => peak = p,
            }
        }
        
        System.alloc(layout)
    }
    
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        ALLOCATED.fetch_sub(layout.size(), Ordering::Relaxed);
        System.dealloc(ptr, layout)
    }
}

#[global_allocator]
static GLOBAL: TrackingAllocator = TrackingAllocator;
```

---

## 9. Testing Strategy

### Memory Regression Tests

```rust
#[test]
fn memory_growth_bounded() {
    let mut brain = Brain::new(BrainConfig::with_size(256, 12));
    brain.set_memory_budget(10 * 1024 * 1024);  // 10 MB limit
    
    for _ in 0..10_000 {
        brain.step();
        assert!(brain.memory_stats().total_bytes <= 10 * 1024 * 1024);
    }
}

#[test]
fn neurogenesis_respects_budget() {
    let mut brain = Brain::new(BrainConfig::with_size(256, 12));
    brain.set_memory_budget(1 * 1024 * 1024);  // 1 MB limit
    
    // Saturate network
    for w in &mut brain.connections.weights {
        *w = 0.9;
    }
    
    let initial = brain.units.len();
    brain.maybe_neurogenesis(0.5, 1000, 10000);  // Try to grow 1000
    let grown = brain.units.len() - initial;
    
    // Should grow less than 1000 due to memory budget
    assert!(grown < 1000);
    assert!(brain.memory_stats().total_bytes <= 1 * 1024 * 1024);
}

#[test]
fn causal_memory_caps_enforced() {
    let mut mem = CausalMemory::new(0.001);
    mem.set_max_edges(1000);
    
    // Try to create 10,000 edges
    for i in 0..100 {
        for j in 0..100 {
            mem.observe(&[i]);
            mem.observe(&[j]);
        }
    }
    
    let stats = mem.stats();
    assert!(stats.edges <= 1000, "Should cap at 1000 edges");
}

#[test]
fn child_brain_memory_multiplier() {
    let parent = Brain::new(BrainConfig::with_size(256, 12));
    let parent_size = parent.memory_stats().total_bytes;
    
    let child = parent.spawn_child(42, ChildConfigOverrides::default());
    let child_size = child.memory_stats().total_bytes;
    
    // Without COW: child ≈ parent size
    assert!(child_size >= parent_size * 9 / 10);  // Allow 10% variance
    
    // With COW (future): child << parent size
    // assert!(child_size < parent_size / 10);
}
```

### Benchmarks

Add to `crates/core/benches/substrate.rs`:

```rust
use criterion::{black_box, Criterion};

fn bench_memory_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("memory");
    
    group.bench_function("csr_compact", |b| {
        let mut brain = Brain::new(BrainConfig::with_size(1024, 16));
        // Create tombstones
        for w in brain.connections.weights.iter_mut().step_by(4) {
            *w = 0.0;
        }
        
        b.iter(|| {
            black_box(brain.compact_connections());
        });
    });
    
    group.bench_function("causal_prune", |b| {
        let mut mem = CausalMemory::new(0.01);
        for i in 0..100 {
            mem.observe(&[i, i+1, i+2]);
        }
        
        b.iter(|| {
            black_box(mem.prune_to_capacity());
        });
    });
}
```

---

## 10. Configuration Recommendations

### Default Configs for Different Use Cases

```rust
impl BrainConfig {
    /// Minimal config for embedded/MCU targets (< 1MB)
    pub fn embedded() -> Self {
        Self {
            unit_count: 128,
            connectivity_per_unit: 6,
            // ... conservative settings
        }
    }
    
    /// Mobile/edge config (< 50MB)
    pub fn mobile() -> Self {
        Self {
            unit_count: 2048,
            connectivity_per_unit: 16,
            // ...
        }
    }
    
    /// Desktop config (< 500MB)
    pub fn desktop() -> Self {
        Self {
            unit_count: 16384,
            connectivity_per_unit: 32,
            // ...
        }
    }
    
    /// Server config (< 4GB)
    pub fn server() -> Self {
        Self {
            unit_count: 131072,  // 128K units
            connectivity_per_unit: 64,
            // ...
        }
    }
}
```

### Memory Budget Presets

```rust
pub enum MemoryProfile {
    Embedded,   // 1 MB
    Mobile,     // 50 MB
    Desktop,    // 500 MB
    Server,     // 4 GB
}

impl Brain {
    pub fn with_memory_profile(profile: MemoryProfile) -> Self {
        let (cfg, budget) = match profile {
            MemoryProfile::Embedded => (BrainConfig::embedded(), 1 * MB),
            MemoryProfile::Mobile => (BrainConfig::mobile(), 50 * MB),
            MemoryProfile::Desktop => (BrainConfig::desktop(), 500 * MB),
            MemoryProfile::Server => (BrainConfig::server(), 4 * GB),
        };
        
        let mut brain = Brain::new(cfg);
        brain.set_memory_budget(budget);
        brain
    }
}
```

---

## 11. Conclusion

Braine's memory usage is currently **predictable and bounded** for static configurations, but lacks **runtime safeguards** against:

1. Unbounded causal memory growth
2. Neurogenesis without global budgets
3. Child brain memory multiplication

The recommended improvements fall into three categories:

### Safety (Prevent OOM)
- Global memory budgets (#1, #7)
- Causal memory hard caps (#2, #7)
- Adaptive pruning thresholds

### Efficiency (Reduce Waste)
- Tombstone tracking (#3)
- Symbol interning (#6)
- Sparse pending inputs (#5)

### Scalability (Enable Large Deployments)
- f16 weights (#9)
- Copy-on-write children (#10)
- Paged allocations (#11)

**Next Steps**:
1. Implement immediate recommendations (1-4) in next sprint
2. Add memory profiling and regression tests
3. Benchmark before/after for each optimization
4. Document memory profiles for different deployment targets

---

## Appendix A: Memory Calculation Formulas

### Brain Base Memory

```
M_brain = M_units + M_connections + M_symbols + M_causal

M_units = N × sizeof(Unit)
        = N × (4×4 + 4)  // amp, phase, bias, decay, salience
        = N × 20 bytes

M_connections = (N × K) × (sizeof(UnitId) + sizeof(Weight)) + (N+1) × sizeof(usize)
              = (N × K) × (8 + 4) + (N+1) × 8
              = (N × K × 12) + (N × 8) bytes

M_symbols = S × (24 + 24 + avg_name_len × 2)  // HashMap + Vec duplication
          ≈ S × (48 + 40) bytes  // assume avg 20 chars
          ≈ S × 88 bytes

M_causal = E × 36 + S × 12  // edges + base counts
```

### Example Calculations

| Config | Units (N) | Conn/Unit (K) | M_units | M_connections | Total (no symbols/causal) |
|--------|-----------|---------------|---------|---------------|---------------------------|
| Embedded | 128 | 6 | 2.5 KB | 9.2 KB | ~12 KB |
| Mobile | 2,048 | 16 | 40 KB | 394 KB | ~434 KB |
| Desktop | 16,384 | 32 | 320 KB | 6.3 MB | ~6.6 MB |
| Server | 131,072 | 64 | 2.5 MB | 101 MB | ~103 MB |

### Child Brain Multiplication

```
M_total = M_parent + (N_children × M_child)

Without COW:
  M_child ≈ M_parent
  M_total = M_parent × (1 + N_children)

With COW (sparse changes):
  M_child ≈ 0.1 × M_parent  // 10% changes
  M_total = M_parent × (1 + 0.1 × N_children)
```

Example:
- Desktop config (6.6 MB parent)
- 10 children
- Without COW: 6.6 × 11 = **72.6 MB**
- With COW: 6.6 × 2 = **13.2 MB**

---

## Appendix B: Alternative Data Structures

### CSR Alternatives

1. **COO (Coordinate List)**
   - Storage: `Vec<(UnitId, UnitId, Weight)>`
   - Pros: Simple, no tombstones, easy append
   - Cons: Slow neighbor iteration (requires linear scan or index)

2. **CSC (Compressed Sparse Column)**
   - Storage: by target instead of source
   - Pros: Fast incoming edge queries
   - Cons: Same memory, different access pattern

3. **Hybrid CSR + Hash**
   - CSR for stable connections
   - HashMap for dynamic/sparse connections
   - Pros: Best of both worlds
   - Cons: Complexity

### Causal Memory Alternatives

1. **Trie-based Symbol Storage**
   - Share prefixes (e.g., "action::left", "action::right")
   - Pros: Memory-efficient for structured names
   - Cons: Slower lookup

2. **Bloom Filter + Sketch**
   - Bloom for existence, Count-Min Sketch for approximate counts
   - Pros: Bounded memory, probabilistic
   - Cons: No exact retrieval

3. **Time-window Bucketing**
   - Keep only recent N observations
   - Pros: Bounded memory, simple
   - Cons: Loses long-term patterns

---

**Document End**
