use std::collections::HashMap;

pub type SymbolId = u32;

#[derive(Debug, Clone, Default)]
struct EdgeStats {
    // Exponentially decayed co-occurrence counts.
    // This is a very cheap proxy for temporal causality on edge devices.
    count: f32,
}

#[derive(Debug, Clone)]
pub struct CausalMemory {
    decay: f32,

    // Directed temporal edges: previous symbols -> current symbols.
    // Key is packed (from,to) into u64.
    edges: HashMap<u64, EdgeStats>,

    // Unconditional (base) counts of symbols appearing.
    base: HashMap<SymbolId, f32>,

    prev_symbols: Vec<SymbolId>,
}

impl CausalMemory {
    pub fn new(decay: f32) -> Self {
        Self {
            decay: decay.clamp(0.0, 1.0),
            edges: HashMap::new(),
            base: HashMap::new(),
            prev_symbols: Vec::new(),
        }
    }

    pub fn observe(&mut self, current_symbols: &[SymbolId]) {
        // Apply decay.
        for v in self.base.values_mut() {
            *v *= 1.0 - self.decay;
        }
        for e in self.edges.values_mut() {
            e.count *= 1.0 - self.decay;
        }

        // Update base counts.
        for &s in current_symbols {
            *self.base.entry(s).or_default() += 1.0;
        }

        // Update directed edges from previous->current.
        for &a in &self.prev_symbols {
            for &b in current_symbols {
                let key = pack(a, b);
                self.edges.entry(key).or_default().count += 1.0;
            }
        }

        self.prev_symbols.clear();
        self.prev_symbols.extend_from_slice(current_symbols);
    }

    /// A cheap "causal strength" score: P(B|A) - P(B)
    /// - Positive means A increases likelihood of B.
    /// - Near zero means little relationship.
    pub fn causal_strength(&self, a: SymbolId, b: SymbolId) -> f32 {
        let base_a = *self.base.get(&a).unwrap_or(&0.0);
        let base_b = *self.base.get(&b).unwrap_or(&0.0);

        if base_a <= 0.001 {
            return 0.0;
        }

        let edge = self.edges.get(&pack(a, b)).map(|e| e.count).unwrap_or(0.0);

        // Approximate conditional probability and base probability.
        let p_b_given_a = (edge / base_a).clamp(0.0, 1.0);
        let total: f32 = self.base.values().sum::<f32>().max(1.0);
        let p_b = (base_b / total).clamp(0.0, 1.0);

        (p_b_given_a - p_b).clamp(-1.0, 1.0)
    }

    /// Merge edges from another memory into this one.
    /// `rate` controls how much of the other's counts are blended in.
    pub fn merge_from(&mut self, other: &CausalMemory, rate: f32) {
        let rate = rate.clamp(0.0, 1.0);

        for (&sym, &count) in other.base.iter() {
            let entry = self.base.entry(sym).or_insert(0.0);
            *entry = (1.0 - rate) * (*entry) + rate * count;
        }

        for (&key, stats) in other.edges.iter() {
            let entry = self.edges.entry(key).or_default();
            entry.count = (1.0 - rate) * entry.count + rate * stats.count;
        }
    }
}

fn pack(a: SymbolId, b: SymbolId) -> u64 {
    ((a as u64) << 32) | (b as u64)
}
