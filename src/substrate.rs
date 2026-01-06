use std::collections::HashMap;

use crate::causality::{CausalMemory, SymbolId};
use crate::prng::Prng;

pub type UnitId = usize;

#[derive(Debug, Clone)]
pub struct Connection {
    pub target: UnitId,
    pub weight: f32,
}

#[derive(Debug, Clone)]
pub struct Unit {
    // "Wave" state: amplitude + phase.
    pub amp: f32,
    pub phase: f32,

    pub bias: f32,
    pub decay: f32,

    // Sparse local couplings.
    pub connections: Vec<Connection>,
}

#[derive(Debug, Clone, Copy)]
pub struct BrainConfig {
    pub unit_count: usize,
    pub connectivity_per_unit: usize,

    pub dt: f32,
    pub base_freq: f32,

    pub noise_amp: f32,
    pub noise_phase: f32,

    // Competition: subtract proportional inhibition from all units.
    pub global_inhibition: f32,

    // Local learning/forgetting.
    pub hebb_rate: f32,
    pub forget_rate: f32,
    pub prune_below: f32,

    pub coactive_threshold: f32,

    // If two units are active and phase-aligned, strengthen more.
    // Range ~ [0, 1]: higher means "must be more aligned".
    pub phase_lock_threshold: f32,

    // One-shot concept formation strength (imprinting).
    pub imprint_rate: f32,

    // If set, makes behavior reproducible for evaluation.
    pub seed: Option<u64>,

    // Causality/meaning memory decay (0..1). Higher means faster forgetting.
    pub causal_decay: f32,
}

#[derive(Debug, Clone, Copy)]
pub struct Stimulus<'a> {
    pub name: &'a str,
    pub strength: f32,
}

impl<'a> Stimulus<'a> {
    pub fn new(name: &'a str, strength: f32) -> Self {
        Self { name, strength }
    }
}

#[derive(Debug, Clone)]
pub enum ActionPolicy {
    Deterministic,
    EpsilonGreedy { epsilon: f32 },
}

#[derive(Debug, Clone)]
pub struct Diagnostics {
    pub unit_count: usize,
    pub connection_count: usize,
    pub pruned_last_step: usize,
    pub avg_amp: f32,
}

#[derive(Debug, Clone)]
struct NamedGroup {
    name: String,
    units: Vec<UnitId>,
}

pub struct Brain {
    cfg: BrainConfig,
    units: Vec<Unit>,

    rng: Prng,

    reserved: Vec<bool>,

    // If false, unit's outgoing connections do not undergo learning updates.
    // Used to protect a parent identity subset in child brains.
    learning_enabled: Vec<bool>,

    // External "sensor" input is just injected current to some units.
    sensor_groups: Vec<NamedGroup>,
    action_groups: Vec<NamedGroup>,

    pending_input: Vec<f32>,

    // Neuromodulator scales learning ("reward", "salience").
    neuromod: f32,

    // Boundary symbol table for causality/meaning.
    symbols: HashMap<String, SymbolId>,
    symbols_rev: Vec<String>,
    active_symbols: Vec<SymbolId>,
    causal: CausalMemory,

    reward_pos_symbol: SymbolId,
    reward_neg_symbol: SymbolId,

    pruned_last_step: usize,
}

impl Brain {
    pub fn new(cfg: BrainConfig) -> Self {
        let mut rng = Prng::new(cfg.seed.unwrap_or(1));

        let mut units = Vec::with_capacity(cfg.unit_count);
        for _ in 0..cfg.unit_count {
            units.push(Unit {
                amp: 0.0,
                phase: rng.gen_range_f32(-core::f32::consts::PI, core::f32::consts::PI),
                bias: 0.0,
                decay: 0.12,
                connections: Vec::new(),
            });
        }

        // Random sparse wiring (no matrices, no dense ops).
        for i in 0..cfg.unit_count {
            let mut conns = Vec::with_capacity(cfg.connectivity_per_unit);
            for _ in 0..cfg.connectivity_per_unit {
                let mut target = rng.gen_range_usize(0, cfg.unit_count);
                if target == i {
                    target = (target + 1) % cfg.unit_count;
                }
                let weight = rng.gen_range_f32(-0.15, 0.15);
                conns.push(Connection { target, weight });
            }
            units[i].connections = conns;
        }

        let pending_input = vec![0.0; cfg.unit_count];
        let reserved = vec![false; cfg.unit_count];
        let learning_enabled = vec![true; cfg.unit_count];

        let mut symbols: HashMap<String, SymbolId> = HashMap::new();
        let mut symbols_rev: Vec<String> = Vec::new();

        // Reserve reward symbols up front.
        let reward_pos_symbol = intern_symbol(&mut symbols, &mut symbols_rev, "reward_pos");
        let reward_neg_symbol = intern_symbol(&mut symbols, &mut symbols_rev, "reward_neg");

        let causal = CausalMemory::new(cfg.causal_decay);

        Self {
            cfg,
            units,
            sensor_groups: Vec::new(),
            action_groups: Vec::new(),
            pending_input,
            neuromod: 0.0,
            pruned_last_step: 0,
            rng,
            reserved,
            learning_enabled,

            symbols,
            symbols_rev,
            active_symbols: Vec::new(),
            causal,
            reward_pos_symbol,
            reward_neg_symbol,
        }
    }

    /// Ensure a sensor group exists; if missing, create it.
    pub fn ensure_sensor(&mut self, name: &str, width: usize) {
        if self.sensor_groups.iter().any(|g| g.name == name) {
            return;
        }
        self.define_sensor(name, width);
    }

    pub fn has_sensor(&self, name: &str) -> bool {
        self.sensor_groups.iter().any(|g| g.name == name)
    }

    /// Create a sandboxed child brain.
    ///
    /// Design intent:
    /// - child inherits structure (couplings + causal memory)
    /// - child can explore with different noise/plasticity
    /// - child cannot mutate a protected identity subset (action groups by default)
    pub fn spawn_child(
        &self,
        seed: u64,
        overrides: crate::supervisor::ChildConfigOverrides,
    ) -> Brain {
        let mut cfg = self.cfg;
        cfg.seed = Some(seed);
        cfg.noise_amp = overrides.noise_amp;
        cfg.noise_phase = overrides.noise_phase;
        cfg.hebb_rate = overrides.hebb_rate;
        cfg.forget_rate = overrides.forget_rate;

        let mut child = Brain::new(cfg);

        // Copy substrate state.
        child.units = self.units.clone();
        child.sensor_groups = self.sensor_groups.clone();
        child.action_groups = self.action_groups.clone();
        child.reserved = self.reserved.clone();

        // Copy symbol table + causal memory.
        child.symbols = self.symbols.clone();
        child.symbols_rev = self.symbols_rev.clone();
        child.reward_pos_symbol = self.reward_pos_symbol;
        child.reward_neg_symbol = self.reward_neg_symbol;
        child.causal = self.causal.clone();

        // Protect parent identity subset: action-group units.
        let mut mask = vec![true; child.units.len()];
        for g in &child.action_groups {
            for &id in &g.units {
                mask[id] = false;
            }
        }
        child.learning_enabled = mask;

        child
    }

    /// Consolidate structural/casual knowledge from a child back into self.
    /// Only merges strong, non-identity couplings.
    pub fn consolidate_from(&mut self, child: &Brain, policy: crate::supervisor::ConsolidationPolicy) {
        let thr = policy.weight_threshold;
        let rate = policy.merge_rate.clamp(0.0, 1.0);

        // Identity units are action group units.
        let mut protected = vec![false; self.units.len()];
        for g in &self.action_groups {
            for &id in &g.units {
                protected[id] = true;
            }
        }

        // Merge couplings.
        for i in 0..self.units.len() {
            if protected[i] {
                continue;
            }

            // For each child connection above threshold, pull parent weight toward child.
            for c_child in &child.units[i].connections {
                if c_child.weight.abs() < thr {
                    continue;
                }
                if c_child.target < protected.len() && protected[c_child.target] {
                    continue;
                }

                if let Some(c_parent) = self.units[i]
                    .connections
                    .iter_mut()
                    .find(|c| c.target == c_child.target)
                {
                    c_parent.weight = (1.0 - rate) * c_parent.weight + rate * c_child.weight;
                } else {
                    self.units[i].connections.push(c_child.clone());
                }
            }
        }

        // Merge causal memory: copy any strong edges from child.
        self.causal.merge_from(&child.causal, 0.25);
    }

    pub fn define_sensor(&mut self, name: &str, width: usize) {
        let units = self.allocate_units(width);
        self.sensor_groups.push(NamedGroup {
            name: name.to_string(),
            units,
        });

        self.intern(name);
    }

    pub fn define_action(&mut self, name: &str, width: usize) {
        let units = self.allocate_units(width);
        // Slight positive bias so actions can become stable attractors.
        for &id in &units {
            self.units[id].bias += 0.02;
        }
        self.action_groups.push(NamedGroup {
            name: name.to_string(),
            units,
        });

        self.intern(name);
    }

    pub fn apply_stimulus(&mut self, stimulus: Stimulus<'_>) {
        let group_units = self
            .sensor_groups
            .iter()
            .find(|g| g.name == stimulus.name)
            .map(|g| g.units.clone());

        if let Some(group_units) = group_units {
            for &id in &group_units {
                self.pending_input[id] += stimulus.strength;
            }

            self.note_symbol(stimulus.name);

            // One-shot imprinting: when a stimulus is present, create a new "concept" unit
            // connected to currently active units (including the sensor group itself).
            // This is the simplest "instant learning" mechanism without training loops.
            self.imprint_if_novel(&group_units, stimulus.strength);
        }
    }

    /// Record the selected action as an event for causality/meaning.
    pub fn note_action(&mut self, action: &str) {
        self.note_symbol(action);
    }

    /// Commit current perception/action/reward events into causal memory.
    /// Call this once per loop after:
    /// - apply_stimulus
    /// - step
    /// - select_action + note_action
    /// - (optional) reinforce_action
    pub fn commit_observation(&mut self) {
        // Map reward scalar to discrete events.
        if self.neuromod > 0.2 {
            self.active_symbols.push(self.reward_pos_symbol);
        } else if self.neuromod < -0.2 {
            self.active_symbols.push(self.reward_neg_symbol);
        }

        // Deduplicate cheaply (small vectors).
        self.active_symbols.sort_unstable();
        self.active_symbols.dedup();

        self.causal.observe(&self.active_symbols);
        self.active_symbols.clear();
    }

    /// Very small "meaning" query: which action is most causally linked to positive reward
    /// under the last seen stimulus symbol.
    pub fn meaning_hint(&self, stimulus: &str) -> Option<(String, f32)> {
        let s = self.symbol_id(stimulus)?;

        let mut best: Option<(String, f32)> = None;
        for g in &self.action_groups {
            let a = self.symbol_id(&g.name)?;
            let score = self.causal.causal_strength(a, self.reward_pos_symbol)
                - self.causal.causal_strength(a, self.reward_neg_symbol);
            if best.as_ref().map(|b| score > b.1).unwrap_or(true) {
                best = Some((g.name.clone(), score));
            }
        }

        // Also ensure stimulus is at least somewhat connected to the suggested action.
        best.and_then(|(act, sc)| {
            let a = self.symbol_id(&act)?;
            let link = self.causal.causal_strength(s, a);
            Some((act, sc * 0.7 + link * 0.3))
        })
    }

    pub fn set_neuromodulator(&mut self, value: f32) {
        // Clamp to a reasonable range.
        self.neuromod = value.clamp(-1.0, 1.0);
    }

    pub fn reinforce_action(&mut self, action: &str, delta_bias: f32) {
        if let Some(group) = self.action_groups.iter().find(|g| g.name == action) {
            for &id in &group.units {
                self.units[id].bias = (self.units[id].bias + delta_bias * 0.01).clamp(-0.5, 0.5);
            }
        }
    }

    pub fn step(&mut self) {
        self.pruned_last_step = 0;

        // Compute global inhibition target as mean activity.
        let avg_amp = self.units.iter().map(|u| u.amp).sum::<f32>() / self.units.len() as f32;
        let inhibition = self.cfg.global_inhibition * avg_amp;

        let mut next_amp = vec![0.0; self.units.len()];
        let mut next_phase = vec![0.0; self.units.len()];

        for i in 0..self.units.len() {
            let u = &self.units[i];
            let mut influence_amp = 0.0;
            let mut influence_phase = 0.0;

            for c in &u.connections {
                let v = &self.units[c.target];
                // Wave-flavored coupling:
                // - amplitude is pulled by neighbor amplitude
                // - phase is gently pulled toward neighbor phase
                // These are local scalar ops; no matrices, no global objective.
                influence_amp += c.weight * v.amp;
                influence_phase += c.weight * angle_diff(v.phase, u.phase);
            }

            let noise_a = self.rng.gen_range_f32(-self.cfg.noise_amp, self.cfg.noise_amp);
            let noise_p = self.rng.gen_range_f32(-self.cfg.noise_phase, self.cfg.noise_phase);

            let input = self.pending_input[i];

            // Continuous-time-ish update.
            let damp = u.decay * u.amp;
            let d_amp = (u.bias + input + influence_amp - inhibition - damp + noise_a) * self.cfg.dt;
            let d_phase = (self.cfg.base_freq + influence_phase + noise_p) * self.cfg.dt;

            next_amp[i] = (u.amp + d_amp).clamp(-2.0, 2.0);
            next_phase[i] = wrap_angle(u.phase + d_phase);
        }

        for i in 0..self.units.len() {
            self.units[i].amp = next_amp[i];
            self.units[i].phase = next_phase[i];
        }

        // Clear one-tick inputs.
        for x in &mut self.pending_input {
            *x = 0.0;
        }

        self.learn_hebbian();
        self.forget_and_prune();
    }

    pub fn select_action(&mut self, policy: &mut ActionPolicy) -> (String, f32) {
        let mut scores: Vec<(String, f32)> = self
            .action_groups
            .iter()
            .map(|g| (g.name.clone(), g.units.iter().map(|&id| self.units[id].amp).sum()))
            .collect();

        scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(core::cmp::Ordering::Equal));

        match policy {
            ActionPolicy::Deterministic => scores[0].clone(),
            ActionPolicy::EpsilonGreedy { epsilon } => {
                if self.rng.gen_range_f32(0.0, 1.0) < *epsilon {
                    let idx = self.rng.gen_range_usize(0, scores.len());
                    scores[idx].clone()
                } else {
                    scores[0].clone()
                }
            }
        }
    }

    pub fn diagnostics(&self) -> Diagnostics {
        let connection_count = self.units.iter().map(|u| u.connections.len()).sum::<usize>();
        let avg_amp = self.units.iter().map(|u| u.amp).sum::<f32>() / self.units.len() as f32;
        Diagnostics {
            unit_count: self.units.len(),
            connection_count,
            pruned_last_step: self.pruned_last_step,
            avg_amp,
        }
    }

    fn learn_hebbian(&mut self) {
        let thr = self.cfg.coactive_threshold;
        let lr = self.cfg.hebb_rate * (1.0 + self.neuromod.max(0.0));

        // Local rule: if i and j are co-active and phase-aligned, strengthen i->j.
        // Otherwise very slight anti-Hebb decay (encourages specialization).
        for i in 0..self.units.len() {
            if !self.learning_enabled[i] {
                continue;
            }
            let a_amp = self.units[i].amp;
            if a_amp <= thr {
                continue;
            }

            // Borrow-safely access unit i mutably and other units immutably.
            let (left, right) = self.units.split_at_mut(i);
            let (unit_i, right_rest) = right
                .split_first_mut()
                .expect("split_at_mut with valid index");

            let a_phase = unit_i.phase;
            for c in &mut unit_i.connections {
                let (b_amp, b_phase) = if c.target < i {
                    let b = &left[c.target];
                    (b.amp, b.phase)
                } else if c.target == i {
                    (unit_i.amp, unit_i.phase)
                } else {
                    let idx = c.target - i - 1;
                    let b = &right_rest[idx];
                    (b.amp, b.phase)
                };

                if b_amp > thr {
                    let align = phase_alignment(a_phase, b_phase);
                    if align > self.cfg.phase_lock_threshold {
                        c.weight += lr * align;
                    } else {
                        c.weight -= lr * 0.05;
                    }
                }

                c.weight = c.weight.clamp(-1.5, 1.5);
            }
        }
    }

    fn forget_and_prune(&mut self) {
        let decay = 1.0 - self.cfg.forget_rate;
        let prune_below = self.cfg.prune_below;

        for u in &mut self.units {
            for c in &mut u.connections {
                c.weight *= decay;
            }
            let before = u.connections.len();
            u.connections.retain(|c| c.weight.abs() >= prune_below);
            self.pruned_last_step += before - u.connections.len();
        }
    }

    fn allocate_units(&mut self, n: usize) -> Vec<UnitId> {
        // Choose from currently unreserved units only.
        let mut idxs: Vec<(UnitId, f32)> = self
            .units
            .iter()
            .enumerate()
            .filter(|(i, _)| !self.reserved[*i])
            .map(|(i, u)| (i, u.amp.abs()))
            .collect();
        idxs.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(core::cmp::Ordering::Equal));

        let chosen: Vec<UnitId> = idxs.into_iter().take(n).map(|(i, _)| i).collect();
        for &id in &chosen {
            self.reserved[id] = true;
        }
        chosen
    }

    fn imprint_if_novel(&mut self, group_units: &[UnitId], strength: f32) {
        // If the stimulus is weak, don't imprint.
        if strength < 0.4 {
            return;
        }

        // Detect novelty by checking whether sensor units already have strong outgoing couplings.
        let mut existing_strength = 0.0;
        for &id in group_units {
            existing_strength += self.units[id]
                .connections
                .iter()
                .map(|c| c.weight.abs())
                .sum::<f32>();
        }

        if existing_strength > 3.0 {
            return;
        }

        // Choose a "concept" unit: the quietest one not in the sensor group.
        let mut candidates: Vec<(UnitId, f32)> = self
            .units
            .iter()
            .enumerate()
            .filter(|(i, _)| !group_units.contains(i))
            .map(|(i, u)| (i, u.amp.abs()))
            .collect();
        candidates.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(core::cmp::Ordering::Equal));

        let Some((concept_id, _)) = candidates.into_iter().next() else {
            return;
        };

        // Connect sensor units to the concept (and back) so it can be recalled.
        for &sid in group_units {
            add_or_bump_connection(&mut self.units[sid].connections, concept_id, self.cfg.imprint_rate);
            add_or_bump_connection(&mut self.units[concept_id].connections, sid, self.cfg.imprint_rate * 0.7);
        }

        // Make the concept slightly excitable.
        self.units[concept_id].bias += 0.04;
    }

    fn intern(&mut self, name: &str) -> SymbolId {
        intern_symbol(&mut self.symbols, &mut self.symbols_rev, name)
    }

    fn symbol_id(&self, name: &str) -> Option<SymbolId> {
        self.symbols.get(name).copied()
    }

    fn note_symbol(&mut self, name: &str) {
        let id = self.intern(name);
        self.active_symbols.push(id);
    }
}

fn intern_symbol(map: &mut HashMap<String, SymbolId>, rev: &mut Vec<String>, name: &str) -> SymbolId {
    if let Some(&id) = map.get(name) {
        return id;
    }
    let id = rev.len() as SymbolId;
    rev.push(name.to_string());
    map.insert(name.to_string(), id);
    id
}

fn add_or_bump_connection(conns: &mut Vec<Connection>, target: UnitId, bump: f32) {
    if let Some(c) = conns.iter_mut().find(|c| c.target == target) {
        c.weight = (c.weight + bump).clamp(-1.5, 1.5);
    } else {
        conns.push(Connection {
            target,
            weight: bump.clamp(-1.5, 1.5),
        });
    }
}

fn wrap_angle(mut x: f32) -> f32 {
    let two_pi = 2.0 * core::f32::consts::PI;
    while x > core::f32::consts::PI {
        x -= two_pi;
    }
    while x < -core::f32::consts::PI {
        x += two_pi;
    }
    x
}

fn angle_diff(a: f32, b: f32) -> f32 {
    wrap_angle(a - b)
}

fn phase_alignment(a: f32, b: f32) -> f32 {
    // 1.0 when aligned, ~0.0 when opposite.
    let d = angle_diff(a, b).abs();
    let x = 1.0 - (d / core::f32::consts::PI);
    x.clamp(0.0, 1.0)
}
