use std::collections::HashMap;

pub type SymbolId = u32;

#[derive(Debug, Clone, Copy, Default)]
pub struct CausalStats {
    pub base_symbols: usize,
    pub edges: usize,
    pub last_directed_edge_updates: usize,
    pub last_cooccur_edge_updates: usize,
}

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

    last_directed_edge_updates: usize,
    last_cooccur_edge_updates: usize,
}

impl CausalMemory {
    pub fn new(decay: f32) -> Self {
        Self {
            decay: decay.clamp(0.0, 1.0),
            edges: HashMap::new(),
            base: HashMap::new(),
            prev_symbols: Vec::new(),

            last_directed_edge_updates: 0,
            last_cooccur_edge_updates: 0,
        }
    }

    pub fn observe(&mut self, current_symbols: &[SymbolId]) {
        self.last_directed_edge_updates = 0;
        self.last_cooccur_edge_updates = 0;

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
                self.last_directed_edge_updates += 1;
            }
        }

        // Also record same-tick co-occurrence as a cheap proxy for immediate meaning links
        // (e.g. stimulus present and action selected in the same control cycle).
        // This keeps compute small because current_symbols is tiny.
        for (i, &a) in current_symbols.iter().enumerate() {
            for &b in current_symbols.iter().skip(i + 1) {
                if a == b {
                    continue;
                }
                let k1 = pack(a, b);
                let k2 = pack(b, a);
                self.edges.entry(k1).or_default().count += 0.5;
                self.edges.entry(k2).or_default().count += 0.5;
                self.last_cooccur_edge_updates += 2;
            }
        }

        self.prev_symbols.clear();
        self.prev_symbols.extend_from_slice(current_symbols);
    }

    pub fn stats(&self) -> CausalStats {
        CausalStats {
            base_symbols: self.base.len(),
            edges: self.edges.len(),
            last_directed_edge_updates: self.last_directed_edge_updates,
            last_cooccur_edge_updates: self.last_cooccur_edge_updates,
        }
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
