// no_std support
#[cfg(not(feature = "std"))]
extern crate alloc;

#[cfg(feature = "std")]
use std::collections::BinaryHeap;
#[cfg(feature = "std")]
use std::collections::HashMap;
#[cfg(feature = "std")]
use std::io::{self, Read, Write};

#[cfg(not(feature = "std"))]
use alloc::collections::BinaryHeap;
#[cfg(not(feature = "std"))]
use alloc::vec::Vec;
#[cfg(not(feature = "std"))]
use hashbrown::HashMap;

use core::cmp::{Ordering, Reverse};

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

#[cfg(feature = "std")]
use crate::storage;

pub type SymbolId = u32;

#[derive(Debug, Clone, Copy, Default)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct CausalStats {
    pub base_symbols: usize,
    pub edges: usize,
    pub last_directed_edge_updates: usize,
    pub last_cooccur_edge_updates: usize,
}

#[derive(Debug, Clone, Default)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
struct EdgeStats {
    // Exponentially decayed co-occurrence counts.
    // This is a very cheap proxy for temporal causality on edge devices.
    count: f32,
}

#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct CausalMemory {
    decay: f32,

    // Directed temporal edges: previous symbols -> current symbols.
    // Key is packed (from,to) into u64.
    edges: HashMap<u64, EdgeStats>,

    // Unconditional (base) counts of symbols appearing.
    base: HashMap<SymbolId, f32>,

    // Cached sum of `base` values, maintained incrementally.
    // This avoids O(|base|) work inside `causal_strength`, which is called frequently
    // when ranking edges for visualization.
    base_total: f32,

    prev_symbols: Vec<SymbolId>,

    // Counts how many observe() calls have occurred. Used to amortize pruning.
    observe_count: u64,

    last_directed_edge_updates: usize,
    last_cooccur_edge_updates: usize,
}

impl CausalMemory {
    pub fn new(decay: f32) -> Self {
        Self {
            decay: decay.clamp(0.0, 1.0),
            edges: HashMap::new(),
            base: HashMap::new(),
            base_total: 0.0,
            prev_symbols: Vec::new(),

            observe_count: 0,

            last_directed_edge_updates: 0,
            last_cooccur_edge_updates: 0,
        }
    }

    pub fn observe(&mut self, current_symbols: &[SymbolId]) {
        self.last_directed_edge_updates = 0;
        self.last_cooccur_edge_updates = 0;

        self.observe_count = self.observe_count.wrapping_add(1);

        // Apply decay.
        self.base_total *= 1.0 - self.decay;
        for v in self.base.values_mut() {
            *v *= 1.0 - self.decay;
        }
        for e in self.edges.values_mut() {
            e.count *= 1.0 - self.decay;
        }

        // Keep memory bounded: occasionally remove near-zero entries.
        // Amortized to avoid scanning large maps every tick.
        if (self.observe_count & 0xFF) == 0 {
            let thr = 0.001;
            self.prune_near_zero(thr);
        }

        // Update base counts.
        for &s in current_symbols {
            *self.base.entry(s).or_default() += 1.0;
            self.base_total += 1.0;
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

    #[must_use]
    pub fn prev_symbols(&self) -> &[SymbolId] {
        &self.prev_symbols
    }

    /// Observe current symbols with lagged directed edges.
    ///
    /// - lag 1 uses the internal `prev_symbols` (persisted in brain images)
    /// - lag >=2 are provided via `lag2_plus_history[0]=lag2`, `[1]=lag3`, ...
    /// - contributions decay geometrically by `lag_decay` (0<lag_decay<1)
    pub fn observe_lagged(
        &mut self,
        current_symbols: &[SymbolId],
        lag2_plus_history: &[Vec<SymbolId>],
        lag_decay: f32,
    ) {
        self.last_directed_edge_updates = 0;
        self.last_cooccur_edge_updates = 0;

        self.observe_count = self.observe_count.wrapping_add(1);

        // Apply decay.
        self.base_total *= 1.0 - self.decay;
        for v in self.base.values_mut() {
            *v *= 1.0 - self.decay;
        }
        for e in self.edges.values_mut() {
            e.count *= 1.0 - self.decay;
        }

        if (self.observe_count & 0xFF) == 0 {
            let thr = 0.001;
            self.prune_near_zero(thr);
        }

        // Update base counts.
        for &s in current_symbols {
            *self.base.entry(s).or_default() += 1.0;
            self.base_total += 1.0;
        }

        // Directed edges: lag 1 from persisted prev_symbols.
        for &a in &self.prev_symbols {
            for &b in current_symbols {
                let key = pack(a, b);
                self.edges.entry(key).or_default().count += 1.0;
                self.last_directed_edge_updates += 1;
            }
        }

        // Directed edges: lag >=2 from provided history.
        let mut w = lag_decay;
        if w.is_finite() {
            for prev in lag2_plus_history {
                if prev.is_empty() {
                    w *= lag_decay;
                    continue;
                }
                let weight = w;
                if weight <= 0.0 {
                    break;
                }
                for &a in prev {
                    for &b in current_symbols {
                        let key = pack(a, b);
                        self.edges.entry(key).or_default().count += weight;
                        self.last_directed_edge_updates += 1;
                    }
                }
                w *= lag_decay;
            }
        }

        // Same-tick co-occurrence as a cheap proxy for immediate meaning links.
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

    fn prune_near_zero(&mut self, threshold: f32) {
        let mut new_total = 0.0;
        self.base.retain(|_, v| {
            if *v <= threshold {
                false
            } else {
                new_total += *v;
                true
            }
        });
        self.base_total = new_total.max(0.0);

        self.edges.retain(|_, e| e.count > threshold);

        // Floating-point drift can accumulate after many retains.
        // Re-anchor occasionally; this is still amortized and keeps invariants tight.
        if (self.observe_count & 0x1FFF) == 0 {
            self.base_total = self.base.values().sum::<f32>();
        }
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
        let total: f32 = self.base_total.max(1.0);
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

        self.base_total = self.base.values().sum::<f32>();

        for (&key, stats) in other.edges.iter() {
            let entry = self.edges.entry(key).or_default();
            entry.count = (1.0 - rate) * entry.count + rate * stats.count;
        }
    }

    /// Return the strongest outgoing causal links from `a`.
    ///
    /// This scans existing edges (so it stays cheap) and ranks by `causal_strength(a,b)`.
    pub fn top_outgoing(&self, a: SymbolId, top_n: usize) -> Vec<(SymbolId, f32)> {
        let mut out: Vec<(SymbolId, f32)> = Vec::new();
        for (&key, _stats) in self.edges.iter() {
            let from = (key >> 32) as SymbolId;
            if from != a {
                continue;
            }
            let b = (key & 0xFFFF_FFFF) as SymbolId;
            let s = self.causal_strength(a, b);
            out.push((b, s));
        }

        out.sort_by(|x, y| y.1.total_cmp(&x.1));
        out.truncate(top_n);
        out
    }

    /// Return strongest outgoing edges from `a` to symbols in `candidates`.
    ///
    /// Useful for predicting next context when you have a known set of context symbol IDs.
    pub fn top_outgoing_filtered(
        &self,
        a: SymbolId,
        candidates: &[SymbolId],
        top_n: usize,
    ) -> Vec<(SymbolId, f32)> {
        let mut out: Vec<(SymbolId, f32)> = Vec::with_capacity(candidates.len());
        for &b in candidates {
            let s = self.causal_strength(a, b);
            if s.abs() > 0.001 {
                out.push((b, s));
            }
        }
        out.sort_by(|x, y| y.1.total_cmp(&x.1));
        out.truncate(top_n);
        out
    }

    /// Return all symbols with their base counts, sorted by count (descending).
    pub fn all_symbols_sorted(&self, top_n: usize) -> Vec<(SymbolId, f32)> {
        let mut out: Vec<(SymbolId, f32)> = self.base.iter().map(|(&s, &c)| (s, c)).collect();
        out.sort_by(|x, y| y.1.total_cmp(&x.1));
        out.truncate(top_n);
        out
    }

    /// Return the current (decayed) base count for a symbol.
    #[must_use]
    pub fn base_count(&self, sym: SymbolId) -> f32 {
        *self.base.get(&sym).unwrap_or(&0.0)
    }

    /// Return top N strongest causal edges in the entire graph.
    ///
    /// Returns (from_symbol, to_symbol, causal_strength).
    pub fn top_edges(&self, top_n: usize) -> Vec<(SymbolId, SymbolId, f32)> {
        if top_n == 0 {
            return Vec::new();
        }

        #[derive(Debug, Clone, Copy)]
        struct EdgeCandidate {
            abs: f32,
            from: SymbolId,
            to: SymbolId,
            s: f32,
        }

        impl PartialEq for EdgeCandidate {
            fn eq(&self, other: &Self) -> bool {
                self.abs.to_bits() == other.abs.to_bits()
                    && self.from == other.from
                    && self.to == other.to
                    && self.s.to_bits() == other.s.to_bits()
            }
        }

        impl Eq for EdgeCandidate {}

        impl PartialOrd for EdgeCandidate {
            fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
                Some(self.cmp(other))
            }
        }

        impl Ord for EdgeCandidate {
            fn cmp(&self, other: &Self) -> Ordering {
                // Total ordering on floats; ties broken by ids for stability.
                self.abs
                    .total_cmp(&other.abs)
                    .then_with(|| self.from.cmp(&other.from))
                    .then_with(|| self.to.cmp(&other.to))
            }
        }

        // Keep only the top N edges by |strength| using a bounded min-heap.
        let mut heap: BinaryHeap<Reverse<EdgeCandidate>> = BinaryHeap::with_capacity(top_n + 1);

        for &key in self.edges.keys() {
            let from = (key >> 32) as SymbolId;
            let to = (key & 0xFFFF_FFFF) as SymbolId;
            let s = self.causal_strength(from, to);
            let abs = s.abs();
            if abs <= 0.001 {
                continue;
            }

            let cand = EdgeCandidate { abs, from, to, s };
            if heap.len() < top_n {
                heap.push(Reverse(cand));
                continue;
            }

            if let Some(smallest) = heap.peek() {
                if cand.abs > smallest.0.abs {
                    heap.pop();
                    heap.push(Reverse(cand));
                }
            }
        }

        let mut out: Vec<(SymbolId, SymbolId, f32)> = heap
            .into_iter()
            .map(|r| (r.0.from, r.0.to, r.0.s))
            .collect();
        out.sort_by(|x, y| y.2.abs().total_cmp(&x.2.abs()));
        out
    }

    #[cfg(feature = "std")]
    pub(crate) fn image_payload_len_bytes(&self) -> u32 {
        let base_n = self.base.len() as u64;
        let edge_n = self.edges.len() as u64;
        let prev_n = self.prev_symbols.len() as u64;

        let mut len = 0u64;
        len += 4; // decay
        len += 4; // base count
        len += base_n * (4 + 4); // (sym,count)
        len += 4; // edge count
        len += edge_n * (8 + 4); // (key,count)
        len += 4; // prev count
        len += prev_n * 4; // prev symbols

        u32::try_from(len).unwrap_or(u32::MAX)
    }

    #[cfg(feature = "std")]
    pub(crate) fn write_image_payload<W: Write>(&self, w: &mut W) -> io::Result<()> {
        storage::write_f32_le(w, self.decay)?;

        storage::write_u32_le(w, self.base.len() as u32)?;
        for (&sym, &count) in self.base.iter() {
            storage::write_u32_le(w, sym)?;
            storage::write_f32_le(w, count)?;
        }

        storage::write_u32_le(w, self.edges.len() as u32)?;
        for (&key, stats) in self.edges.iter() {
            storage::write_u64_le(w, key)?;
            storage::write_f32_le(w, stats.count)?;
        }

        storage::write_u32_le(w, self.prev_symbols.len() as u32)?;
        for &sym in self.prev_symbols.iter() {
            storage::write_u32_le(w, sym)?;
        }

        Ok(())
    }

    #[cfg(feature = "std")]
    pub(crate) fn read_image_payload<R: Read>(r: &mut R) -> io::Result<Self> {
        let decay = storage::read_f32_le(r)?;

        let base_n = storage::read_u32_le(r)? as usize;
        let mut base: HashMap<SymbolId, f32> = HashMap::with_capacity(base_n);
        for _ in 0..base_n {
            let sym = storage::read_u32_le(r)?;
            let count = storage::read_f32_le(r)?;
            base.insert(sym, count);
        }

        let base_total = base.values().sum::<f32>();

        let edge_n = storage::read_u32_le(r)? as usize;
        let mut edges: HashMap<u64, EdgeStats> = HashMap::with_capacity(edge_n);
        for _ in 0..edge_n {
            let key = storage::read_u64_le(r)?;
            let count = storage::read_f32_le(r)?;
            edges.insert(key, EdgeStats { count });
        }

        let prev_n = storage::read_u32_le(r)? as usize;
        let mut prev_symbols: Vec<SymbolId> = Vec::with_capacity(prev_n);
        for _ in 0..prev_n {
            prev_symbols.push(storage::read_u32_le(r)?);
        }

        Ok(Self {
            decay: decay.clamp(0.0, 1.0),
            edges,
            base,
            base_total,
            prev_symbols,
            observe_count: 0,
            last_directed_edge_updates: 0,
            last_cooccur_edge_updates: 0,
        })
    }
}

fn pack(a: SymbolId, b: SymbolId) -> u64 {
    ((a as u64) << 32) | (b as u64)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn causal_memory_observe_updates_base_counts() {
        let mut mem = CausalMemory::new(0.0); // No decay for predictable testing

        mem.observe(&[1, 2, 3]);

        let stats = mem.stats();
        assert_eq!(stats.base_symbols, 3);
        assert!(stats.edges > 0); // Co-occurrence edges
    }

    #[test]
    fn causal_memory_directed_edges() {
        let mut mem = CausalMemory::new(0.0);

        // First observation sets prev_symbols
        mem.observe(&[1]);
        // Second observation creates directed edge 1->2
        mem.observe(&[2]);

        let stats = mem.stats();
        assert!(stats.last_directed_edge_updates > 0);

        // Causal strength should be positive (1 precedes 2)
        let strength = mem.causal_strength(1, 2);
        assert!(
            strength > 0.0,
            "Expected positive causal strength, got {}",
            strength
        );
    }

    #[test]
    fn causal_memory_lagged_directed_edges_decay_with_lag() {
        let mut mem = CausalMemory::new(0.0);

        // Establish base counts for lag symbols.
        mem.observe(&[30]);
        mem.observe(&[10]);

        // Now observe 20 with lag 1 = [10] (internal prev_symbols)
        // and lag 2 = [30] (provided history) with decay 0.5.
        mem.observe_lagged(&[20], &[vec![30]], 0.5);

        let stats = mem.stats();
        assert_eq!(
            stats.last_directed_edge_updates, 2,
            "Expected 2 directed updates (lag1 + lag2)"
        );

        let s_lag1 = mem.causal_strength(10, 20);
        let s_lag2 = mem.causal_strength(30, 20);
        assert!(s_lag1 > 0.0, "Expected positive lag1 strength");
        assert!(s_lag2 > 0.0, "Expected positive lag2 strength");
        assert!(
            s_lag1 > s_lag2,
            "Expected lag1 strength > lag2 strength (decayed)"
        );
    }

    #[test]
    fn causal_memory_decay() {
        let mut mem = CausalMemory::new(0.5); // 50% decay

        mem.observe(&[1]);
        mem.observe(&[2]);

        // After decay, base counts should be reduced
        let _stats1 = mem.stats();

        mem.observe(&[3]); // This applies decay

        // Strength of 1->2 should decrease after decay
        let strength_after = mem.causal_strength(1, 2);
        assert!(strength_after < 1.0, "Strength should decay");
    }

    #[test]
    fn causal_memory_merge() {
        let mut mem1 = CausalMemory::new(0.0);
        let mut mem2 = CausalMemory::new(0.0);

        mem1.observe(&[1]);
        mem1.observe(&[2]);

        mem2.observe(&[3]);
        mem2.observe(&[4]);

        // Merge mem2 into mem1 at 50% rate
        mem1.merge_from(&mem2, 0.5);

        let stats = mem1.stats();
        assert!(
            stats.base_symbols >= 2,
            "Should have symbols from both memories"
        );
    }

    #[test]
    fn causal_memory_top_outgoing() {
        let mut mem = CausalMemory::new(0.0);

        // Create multiple edges from symbol 1
        mem.observe(&[1]);
        mem.observe(&[2]);
        mem.observe(&[1]);
        mem.observe(&[3]);
        mem.observe(&[1]);
        mem.observe(&[2]); // 2 follows 1 twice, 3 follows once

        let top = mem.top_outgoing(1, 5);
        assert!(!top.is_empty(), "Should have outgoing edges");
    }

    #[test]
    fn causal_memory_serialization_roundtrip() {
        let mut mem = CausalMemory::new(0.1);
        mem.observe(&[1, 2]);
        mem.observe(&[3, 4]);
        mem.observe(&[1]);

        let stats_before = mem.stats();

        // Serialize
        let mut buf = Vec::new();
        mem.write_image_payload(&mut buf).unwrap();

        // Deserialize
        let mut cursor = std::io::Cursor::new(&buf);
        let mem2 = CausalMemory::read_image_payload(&mut cursor).unwrap();

        let stats_after = mem2.stats();
        assert_eq!(stats_before.base_symbols, stats_after.base_symbols);
        assert_eq!(stats_before.edges, stats_after.edges);
    }

    #[test]
    fn causal_strength_returns_zero_for_unknown() {
        let mem = CausalMemory::new(0.0);

        // Unknown symbols should return 0
        let strength = mem.causal_strength(999, 888);
        assert_eq!(strength, 0.0);
    }

    #[test]
    fn pack_unpack_roundtrip() {
        let a: SymbolId = 12345;
        let b: SymbolId = 67890;
        let packed = pack(a, b);

        let unpacked_a = (packed >> 32) as SymbolId;
        let unpacked_b = (packed & 0xFFFF_FFFF) as SymbolId;

        assert_eq!(a, unpacked_a);
        assert_eq!(b, unpacked_b);
    }

    #[test]
    fn base_total_tracks_base_sum_after_prune() {
        let mut mem = CausalMemory::new(0.0);

        // Create a bunch of symbols with varying counts.
        for i in 0..200u32 {
            mem.observe(&[i]);
        }

        // Manually force some near-zero entries, then prune.
        for (i, (_sym, v)) in mem.base.iter_mut().enumerate() {
            if i % 3 == 0 {
                *v = 0.0005;
            }
        }
        mem.base_total = mem.base.values().sum::<f32>();
        mem.prune_near_zero(0.001);

        let expected = mem.base.values().sum::<f32>();
        assert!((mem.base_total - expected).abs() < 1e-6);
    }
}
