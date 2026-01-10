use braine::{
    storage,
    substrate::{Brain, BrainDelta},
};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::io::{self, Read, Write};

#[derive(Debug, Clone, Copy, Default)]
struct ContextStats {
    first_seen_trial: u32,
    trials_seen: u32,
    reward_fast_ema: f32,
    reward_slow_ema: f32,
    best_slow_ema: f32,
}

impl ContextStats {
    fn note_reward(&mut self, parent_trials: u32, reward: f32) {
        if self.trials_seen == 0 {
            self.first_seen_trial = parent_trials;
            self.trials_seen = 1;
            self.reward_fast_ema = reward;
            self.reward_slow_ema = reward;
            self.best_slow_ema = reward;
            return;
        }

        self.trials_seen = self.trials_seen.saturating_add(1);

        // Two-timescale EMAs for shift/collapse detection.
        // Fast responds within ~5 trials; slow responds within ~20 trials.
        let a_fast = 0.20;
        let a_slow = 0.05;
        self.reward_fast_ema = (1.0 - a_fast) * self.reward_fast_ema + a_fast * reward;
        self.reward_slow_ema = (1.0 - a_slow) * self.reward_slow_ema + a_slow * reward;
        if self.reward_slow_ema > self.best_slow_ema {
            self.best_slow_ema = self.reward_slow_ema;
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ParentLearningPolicy {
    Normal,
    Reduced,
    Holdout,
}

impl ParentLearningPolicy {
    pub fn parse(s: &str) -> Option<Self> {
        match s.trim().to_ascii_lowercase().as_str() {
            "normal" => Some(Self::Normal),
            "reduced" => Some(Self::Reduced),
            "holdout" => Some(Self::Holdout),
            _ => None,
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Normal => "normal",
            Self::Reduced => "reduced",
            Self::Holdout => "holdout",
        }
    }
}

#[derive(Debug, Clone)]
pub struct ExpertPolicy {
    pub parent_learning: ParentLearningPolicy,
    pub max_children: usize,

    /// Scales the reward/neuromod applied to the child brain (>= 1 speeds learning).
    pub child_reward_scale: f32,

    /// Episode length in completed trials.
    pub episode_trials: u32,

    /// Consolidation merges top-K weight deltas.
    pub consolidate_topk: usize,

    /// Per-edge delta clamp applied during consolidation.
    pub consolidate_delta_max: f32,

    /// Spawn trigger: reward regime shift threshold on |EMA_fast - EMA_slow|.
    pub reward_shift_ema_delta_threshold: f32,

    /// Spawn trigger: performance collapse threshold on (best_slow_ema - fast_ema).
    pub performance_collapse_drop_threshold: f32,

    /// Minimum baseline performance required to consider a drop a collapse.
    pub performance_collapse_baseline_min: f32,

    /// Don't spawn experts until we have at least this many trials.
    pub spawn_min_trials: u32,

    /// Per-context cooldown (in trials) after cull/consolidation.
    pub cooldown_trials: u32,

    /// Promotion threshold on the expert's reward EMA.
    pub promote_reward_ema: f32,

    /// Allow experts to spawn nested experts.
    pub allow_nested: bool,

    /// Maximum depth of expert nesting (1 = parent->experts only).
    pub max_depth: u32,
}

impl Default for ExpertPolicy {
    fn default() -> Self {
        Self {
            parent_learning: ParentLearningPolicy::Holdout,
            max_children: 2,
            child_reward_scale: 1.5,
            episode_trials: 32,
            consolidate_topk: 64,
            consolidate_delta_max: 0.02,
            reward_shift_ema_delta_threshold: 0.55,
            performance_collapse_drop_threshold: 0.65,
            performance_collapse_baseline_min: 0.25,
            spawn_min_trials: 20,
            cooldown_trials: 50,
            promote_reward_ema: 0.2,
            allow_nested: false,
            max_depth: 1,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExpertsPersistenceMode {
    /// Persist full expert tree and resume mid-episode.
    Full,
    /// Persist policy/cooldowns but drop active experts on save.
    DropActive,
}

impl ExpertsPersistenceMode {
    pub fn parse(s: &str) -> Option<Self> {
        match s.trim().to_ascii_lowercase().as_str() {
            "full" => Some(Self::Full),
            "drop" | "drop_active" | "dropactive" => Some(Self::DropActive),
            _ => None,
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Full => "full",
            Self::DropActive => "drop_active",
        }
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ExpertsSummary {
    pub active_count: u32,
    #[serde(default)]
    pub total_active_count: u32,
    pub max_children: u32,
    pub last_spawn_reason: String,
    pub last_consolidation: String,

    #[serde(default)]
    pub persistence_mode: String,
    #[serde(default)]
    pub allow_nested: bool,
    #[serde(default)]
    pub max_depth: u32,

    #[serde(default)]
    pub reward_shift_ema_delta_threshold: f32,
    #[serde(default)]
    pub performance_collapse_drop_threshold: f32,
    #[serde(default)]
    pub performance_collapse_baseline_min: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActiveExpertSummary {
    pub id: u32,
    pub context_key: String,
    pub age_steps: u64,
    pub reward_ema: f32,
}

#[derive(Debug, Clone, Default)]
pub struct ControllerRoute {
    pub path: Vec<u32>,
    pub depth: u32,
    pub reward_scale: f32,
}

pub struct ControllerBorrow<'a> {
    pub brain: &'a mut Brain,
    pub route: ControllerRoute,
}

pub struct ControllerBorrowRef<'a> {
    pub brain: &'a Brain,
    #[allow(dead_code)]
    pub route: ControllerRoute,
}

struct Expert {
    id: u32,
    context_key: String,
    brain: Brain,
    fork_point: Brain,

    children: Box<ExpertManager>,

    age_steps: u64,
    completed_trials: u32,
    episode_trials: u32,
    reward_ema: f32,
}

impl Expert {
    fn new(id: u32, context_key: String, parent: &Brain, inherited_policy: &ExpertPolicy) -> Self {
        let brain = parent.clone();
        let fork_point = parent.clone();
        let mut children = ExpertManager::new();
        children.policy = inherited_policy.clone();
        children.enabled = inherited_policy.allow_nested && inherited_policy.max_depth > 1;
        Self {
            id,
            context_key,
            brain,
            fork_point,
            children: Box::new(children),
            age_steps: 0,
            completed_trials: 0,
            episode_trials: 0,
            reward_ema: 0.0,
        }
    }
}

pub struct ExpertManager {
    enabled: bool,
    policy: ExpertPolicy,
    persistence_mode: ExpertsPersistenceMode,
    next_id: u32,
    experts: Vec<Expert>,
    cooldown_by_context: HashMap<String, u32>,

    context_stats: HashMap<String, ContextStats>,

    last_spawn_reason: String,
    last_consolidation: String,
}

impl ExpertManager {
    pub fn new() -> Self {
        Self {
            enabled: false,
            policy: ExpertPolicy::default(),
            persistence_mode: ExpertsPersistenceMode::Full,
            next_id: 1,
            experts: Vec::new(),
            cooldown_by_context: HashMap::new(),
            context_stats: HashMap::new(),
            last_spawn_reason: String::new(),
            last_consolidation: String::new(),
        }
    }

    pub fn enabled(&self) -> bool {
        self.enabled
    }

    pub fn policy(&self) -> &ExpertPolicy {
        &self.policy
    }

    pub fn set_persistence_mode(&mut self, mode: ExpertsPersistenceMode) {
        self.persistence_mode = mode;
    }

    pub fn set_enabled(&mut self, enabled: bool) {
        self.enabled = enabled;
        if !enabled {
            self.experts.clear();
            self.cooldown_by_context.clear();
            self.context_stats.clear();
        }
    }

    pub fn set_policy(&mut self, policy: ExpertPolicy) {
        self.policy = policy;

        // Apply to existing experts (including nested managers).
        for e in &mut self.experts {
            e.children.policy = self.policy.clone();
            e.children.enabled = self.policy.allow_nested && self.policy.max_depth > 1;
        }
    }

    pub fn cull_all_recursive(&mut self) {
        for e in &mut self.experts {
            e.children.cull_all_recursive();
        }
        self.experts.clear();
        self.context_stats.clear();
    }

    /// Record the most recent completed trial for the manager that would own a spawn
    /// under `controller_path`.
    ///
    /// - If `controller_path` is empty: update this manager's per-context stats.
    /// - Otherwise: traverse to the deepest controlling expert and update its child manager.
    pub fn note_trial_for_spawn_target_under_path(
        &mut self,
        context_key: &str,
        controller_path: &[u32],
        parent_trials: u32,
        reward: f32,
    ) {
        if controller_path.is_empty() {
            self.context_stats
                .entry(context_key.to_string())
                .or_default()
                .note_reward(parent_trials, reward);
            return;
        }

        let mut cur: &mut ExpertManager = self;
        for (i, id) in controller_path.iter().enumerate() {
            let Some(idx) = cur.experts.iter().position(|e| e.id == *id) else {
                return;
            };
            let e: &mut Expert = &mut cur.experts[idx];
            let last = i + 1 == controller_path.len();
            if last {
                // Only record stats for nested spawn if nesting is enabled at this level.
                if !cur.policy.allow_nested || cur.policy.max_depth <= (i as u32 + 1) {
                    return;
                }
                e.children
                    .context_stats
                    .entry(context_key.to_string())
                    .or_default()
                    .note_reward(parent_trials, reward);
                return;
            }
            cur = &mut e.children;
        }
    }

    fn should_spawn_for_signals(
        &self,
        context_key: &str,
        parent_trials: u32,
        parent: &Brain,
    ) -> Option<String> {
        if !self.enabled {
            return None;
        }
        if self.experts.len() >= self.policy.max_children {
            return None;
        }
        if self.active_expert_index(context_key).is_some() {
            return None;
        }
        if self.cooldown_by_context.contains_key(context_key) {
            return None;
        }

        // Saturation detection: treat "should grow" as a proxy for saturation / attractor brittleness.
        let saturated = parent.should_grow(0.35);

        let stats = self
            .context_stats
            .get(context_key)
            .copied()
            .unwrap_or_default();
        let novel = stats.trials_seen == 1;

        // Require some burn-in for shift/collapse triggers.
        let min_for_shift = 12;
        let min_for_collapse = 20;

        // Reward regime shift: two-timescale EMA divergence.
        let ema_delta = (stats.reward_fast_ema - stats.reward_slow_ema).abs();
        let sign_flip = stats.reward_fast_ema.abs() > 0.2
            && stats.reward_slow_ema.abs() > 0.2
            && stats.reward_fast_ema.signum() != stats.reward_slow_ema.signum();
        let reward_shift = stats.trials_seen >= min_for_shift
            && (ema_delta >= self.policy.reward_shift_ema_delta_threshold || sign_flip);

        // Performance collapse: fast EMA drops far below prior best slow EMA.
        let perf_collapse = stats.trials_seen >= min_for_collapse
            && stats.best_slow_ema >= self.policy.performance_collapse_baseline_min
            && stats.reward_fast_ema
                <= stats.best_slow_ema - self.policy.performance_collapse_drop_threshold;

        let any_signal = novel || reward_shift || perf_collapse || saturated;
        if !any_signal {
            return None;
        }

        // Respect global warmup unless it's explicit novelty.
        if !novel && parent_trials < self.policy.spawn_min_trials {
            return None;
        }

        let mut reasons: Vec<&'static str> = Vec::new();
        if novel {
            reasons.push("novel_context");
        }
        if reward_shift {
            reasons.push("reward_shift");
        }
        if perf_collapse {
            reasons.push("performance_collapse");
        }
        if saturated {
            reasons.push("saturation");
        }

        Some(reasons.join("+"))
    }

    pub fn maybe_spawn_for_signals(
        &mut self,
        context_key: &str,
        parent_trials: u32,
        parent: &Brain,
    ) {
        let Some(signal_reason) = self.should_spawn_for_signals(context_key, parent_trials, parent)
        else {
            return;
        };

        let id = self.next_id;
        self.next_id += 1;

        self.last_spawn_reason = format!(
            "spawned expert id={} ctx='{}' (signals={}, trials={})",
            id, context_key, signal_reason, parent_trials
        );
        self.experts.push(Expert::new(
            id,
            context_key.to_string(),
            parent,
            &self.policy,
        ));
    }

    /// Spawn under the currently controlling expert chain (nested spawn), using
    /// explicit novelty/shift/collapse/saturation signals.
    ///
    /// If `controller_path` is empty, this behaves like `maybe_spawn_for_signals`.
    pub fn maybe_spawn_for_signals_under_path(
        &mut self,
        context_key: &str,
        controller_path: &[u32],
        parent_trials: u32,
        root_parent: &Brain,
    ) {
        if controller_path.is_empty() {
            self.maybe_spawn_for_signals(context_key, parent_trials, root_parent);
            return;
        }

        let mut cur: &mut ExpertManager = self;
        for (i, id) in controller_path.iter().enumerate() {
            let Some(idx) = cur.experts.iter().position(|e| e.id == *id) else {
                return;
            };

            let e: &mut Expert = &mut cur.experts[idx];
            let last = i + 1 == controller_path.len();
            if last {
                if !cur.policy.allow_nested || cur.policy.max_depth <= (i as u32 + 1) {
                    return;
                }
                e.children
                    .maybe_spawn_for_signals(context_key, parent_trials, &e.brain);
                return;
            }

            cur = &mut e.children;
        }
    }

    pub fn tick_cooldowns(&mut self, completed_trial: bool) {
        if !completed_trial {
            return;
        }
        for v in self.cooldown_by_context.values_mut() {
            *v = v.saturating_sub(1);
        }
        self.cooldown_by_context.retain(|_, v| *v > 0);

        for e in &mut self.experts {
            e.children.tick_cooldowns(completed_trial);
        }
    }

    pub fn active_expert_index(&self, context_key: &str) -> Option<usize> {
        self.experts
            .iter()
            .position(|e| e.context_key == context_key)
    }

    pub fn active_expert_summary(&self, context_key: &str) -> Option<ActiveExpertSummary> {
        let e = self.experts.iter().find(|e| e.context_key == context_key)?;
        if self.policy.allow_nested {
            if let Some(deeper) = e.children.active_expert_summary(context_key) {
                return Some(deeper);
            }
        }
        Some(ActiveExpertSummary {
            id: e.id,
            context_key: e.context_key.clone(),
            age_steps: e.age_steps,
            reward_ema: e.reward_ema,
        })
    }

    pub fn total_active_count_recursive(&self) -> u32 {
        let mut total = self.experts.len() as u32;
        for e in &self.experts {
            total = total.saturating_add(e.children.total_active_count_recursive());
        }
        total
    }

    pub fn summary(&self) -> ExpertsSummary {
        ExpertsSummary {
            active_count: self.experts.len() as u32,
            total_active_count: self.total_active_count_recursive(),
            max_children: self.policy.max_children as u32,
            last_spawn_reason: self.last_spawn_reason.clone(),
            last_consolidation: self.last_consolidation.clone(),
            persistence_mode: self.persistence_mode.as_str().to_string(),
            allow_nested: self.policy.allow_nested,
            max_depth: self.policy.max_depth,

            reward_shift_ema_delta_threshold: self.policy.reward_shift_ema_delta_threshold,
            performance_collapse_drop_threshold: self.policy.performance_collapse_drop_threshold,
            performance_collapse_baseline_min: self.policy.performance_collapse_baseline_min,
        }
    }

    pub fn controller_for_context_mut<'a>(
        &'a mut self,
        context_key: &str,
        parent: &'a mut Brain,
    ) -> ControllerBorrow<'a> {
        let mut route = ControllerRoute {
            reward_scale: 1.0,
            ..Default::default()
        };

        let brain = self.controller_inner(context_key, parent, 0, &mut route);
        ControllerBorrow { brain, route }
    }

    pub fn controller_for_context_ref<'a>(
        &'a self,
        context_key: &str,
        parent: &'a Brain,
    ) -> ControllerBorrowRef<'a> {
        let mut route = ControllerRoute {
            reward_scale: 1.0,
            ..Default::default()
        };

        let brain = self.controller_inner_ref(context_key, parent, 0, &mut route);
        ControllerBorrowRef { brain, route }
    }

    fn controller_inner<'a>(
        &'a mut self,
        context_key: &str,
        parent: &'a mut Brain,
        depth: u32,
        route: &mut ControllerRoute,
    ) -> &'a mut Brain {
        if !self.enabled {
            return parent;
        }

        let Some(idx) = self.active_expert_index(context_key) else {
            return parent;
        };

        // Depth limit: allow routing to this expert, but do not descend further.
        let max_depth = self.policy.max_depth.max(1);
        let at_limit = depth + 1 >= max_depth;

        let e = &mut self.experts[idx];
        e.age_steps = e.age_steps.saturating_add(1);

        route.path.push(e.id);
        route.depth = route.depth.saturating_add(1);
        route.reward_scale *= self.policy.child_reward_scale;

        if at_limit || !self.policy.allow_nested {
            return &mut e.brain;
        }

        e.children
            .controller_inner(context_key, &mut e.brain, depth + 1, route)
    }

    fn controller_inner_ref<'a>(
        &'a self,
        context_key: &str,
        parent: &'a Brain,
        depth: u32,
        route: &mut ControllerRoute,
    ) -> &'a Brain {
        if !self.enabled {
            return parent;
        }

        let Some(idx) = self.active_expert_index(context_key) else {
            return parent;
        };

        // Depth limit: allow routing to this expert, but do not descend further.
        let max_depth = self.policy.max_depth.max(1);
        let at_limit = depth + 1 >= max_depth;

        let e = &self.experts[idx];

        route.path.push(e.id);
        route.depth = route.depth.saturating_add(1);
        route.reward_scale *= self.policy.child_reward_scale;

        if at_limit || !self.policy.allow_nested {
            return &e.brain;
        }

        e.children
            .controller_inner_ref(context_key, &e.brain, depth + 1, route)
    }

    pub fn on_trial_completed_path(&mut self, path: &[u32], reward: f32, root_parent: &mut Brain) {
        self.on_trial_completed_path_inner(path, reward, root_parent);
    }

    fn on_trial_completed_path_inner(
        &mut self,
        path: &[u32],
        reward: f32,
        parent_brain: &mut Brain,
    ) {
        let Some((&id, rest)) = path.split_first() else {
            return;
        };

        let Some(idx) = self.experts.iter().position(|e| e.id == id) else {
            return;
        };

        // First let nested experts process the trial (deepest first).
        {
            let e = &mut self.experts[idx];
            e.children
                .on_trial_completed_path_inner(rest, reward, &mut e.brain);
        }

        // Now update and potentially consolidate/cull this expert.
        // Note: we must re-borrow after the nested call.
        let alpha = 0.15;
        let promote = {
            let e = &mut self.experts[idx];
            e.reward_ema = (1.0 - alpha) * e.reward_ema + alpha * reward;
            e.completed_trials = e.completed_trials.saturating_add(1);
            e.episode_trials = e.episode_trials.saturating_add(1);
            e.episode_trials >= self.policy.episode_trials
                && e.reward_ema >= self.policy.promote_reward_ema
        };

        let episode_done = self.experts[idx].episode_trials >= self.policy.episode_trials;
        if !episode_done {
            return;
        }

        if promote {
            let delta: BrainDelta = self.experts[idx]
                .brain
                .diff_weights_topk(&self.experts[idx].fork_point, self.policy.consolidate_topk);
            if !delta.weight_deltas.is_empty() {
                parent_brain.apply_weight_delta(&delta, self.policy.consolidate_delta_max);
                self.last_consolidation = format!(
                    "consolidated expert id={} ctx='{}' (topk={}, ema={:.3})",
                    self.experts[idx].id,
                    self.experts[idx].context_key,
                    self.policy.consolidate_topk,
                    self.experts[idx].reward_ema
                );
            } else {
                self.last_consolidation = format!(
                    "expert id={} ctx='{}' had no mergeable delta (ema={:.3})",
                    self.experts[idx].id,
                    self.experts[idx].context_key,
                    self.experts[idx].reward_ema
                );
            }
        } else {
            self.last_consolidation = format!(
                "culled expert id={} ctx='{}' (ema={:.3})",
                self.experts[idx].id, self.experts[idx].context_key, self.experts[idx].reward_ema
            );
        }

        let ctx = self.experts[idx].context_key.clone();
        self.experts.remove(idx);
        self.cooldown_by_context
            .insert(ctx, self.policy.cooldown_trials);
    }

    pub fn for_each_brain_mut<F: FnMut(&mut Brain)>(&mut self, f: &mut F) {
        for e in &mut self.experts {
            f(&mut e.brain);
            f(&mut e.fork_point);
            e.children.for_each_brain_mut(f);
        }
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Persistence (binary, deterministic)
    // ─────────────────────────────────────────────────────────────────────────

    pub fn save_state_bytes(&self) -> io::Result<Vec<u8>> {
        let mut out: Vec<u8> = Vec::new();
        self.write_state_to(&mut out)?;
        Ok(out)
    }

    pub fn load_state_bytes(&mut self, bytes: &[u8]) -> io::Result<()> {
        let mut cursor = std::io::Cursor::new(bytes);
        *self = Self::read_state_from(&mut cursor)?;
        Ok(())
    }

    fn write_state_to<W: Write>(&self, w: &mut W) -> io::Result<()> {
        // Version
        storage::write_u32_le(w, 3)?;

        w.write_all(&[self.enabled as u8])?;
        w.write_all(&[match self.persistence_mode {
            ExpertsPersistenceMode::Full => 0,
            ExpertsPersistenceMode::DropActive => 1,
        }])?;

        storage::write_u32_le(w, self.next_id)?;

        // Policy
        let pl = match self.policy.parent_learning {
            ParentLearningPolicy::Normal => 0u8,
            ParentLearningPolicy::Reduced => 1u8,
            ParentLearningPolicy::Holdout => 2u8,
        };
        w.write_all(&[pl])?;
        storage::write_u32_le(w, self.policy.max_children as u32)?;
        storage::write_f32_le(w, self.policy.child_reward_scale)?;
        storage::write_u32_le(w, self.policy.episode_trials)?;
        storage::write_u32_le(w, self.policy.consolidate_topk as u32)?;
        storage::write_f32_le(w, self.policy.consolidate_delta_max)?;
        storage::write_f32_le(w, self.policy.reward_shift_ema_delta_threshold)?;
        storage::write_f32_le(w, self.policy.performance_collapse_drop_threshold)?;
        storage::write_f32_le(w, self.policy.performance_collapse_baseline_min)?;
        storage::write_u32_le(w, self.policy.spawn_min_trials)?;
        storage::write_u32_le(w, self.policy.cooldown_trials)?;
        storage::write_f32_le(w, self.policy.promote_reward_ema)?;
        w.write_all(&[self.policy.allow_nested as u8])?;
        storage::write_u32_le(w, self.policy.max_depth)?;

        storage::write_string(w, &self.last_spawn_reason)?;
        storage::write_string(w, &self.last_consolidation)?;

        // Cooldowns: write deterministically by sorting keys.
        let mut cooldown_items: Vec<(&String, &u32)> = self.cooldown_by_context.iter().collect();
        cooldown_items.sort_by(|a, b| a.0.cmp(b.0));
        storage::write_u32_le(w, cooldown_items.len() as u32)?;
        for (k, v) in cooldown_items {
            storage::write_string(w, k)?;
            storage::write_u32_le(w, *v)?;
        }

        // Context stats (deterministic by sorting keys).
        let mut stat_items: Vec<(&String, &ContextStats)> = self.context_stats.iter().collect();
        stat_items.sort_by(|a, b| a.0.cmp(b.0));
        storage::write_u32_le(w, stat_items.len() as u32)?;
        for (k, s) in stat_items {
            storage::write_string(w, k)?;
            storage::write_u32_le(w, s.first_seen_trial)?;
            storage::write_u32_le(w, s.trials_seen)?;
            storage::write_f32_le(w, s.reward_fast_ema)?;
            storage::write_f32_le(w, s.reward_slow_ema)?;
            storage::write_f32_le(w, s.best_slow_ema)?;
        }

        // Experts
        let persist_experts = self.persistence_mode == ExpertsPersistenceMode::Full;
        let experts_to_write: &[Expert] = if persist_experts { &self.experts } else { &[] };
        storage::write_u32_le(w, experts_to_write.len() as u32)?;
        for e in experts_to_write {
            storage::write_u32_le(w, e.id)?;
            storage::write_string(w, &e.context_key)?;
            storage::write_u64_le(w, e.age_steps)?;
            storage::write_u32_le(w, e.completed_trials)?;
            storage::write_u32_le(w, e.episode_trials)?;
            storage::write_f32_le(w, e.reward_ema)?;

            let mut brain_img: Vec<u8> = Vec::new();
            e.brain.save_image_to(&mut brain_img)?;
            storage::write_bytes(w, &brain_img)?;

            let mut fork_img: Vec<u8> = Vec::new();
            e.fork_point.save_image_to(&mut fork_img)?;
            storage::write_bytes(w, &fork_img)?;

            let child_bytes = e.children.save_state_bytes()?;
            storage::write_bytes(w, &child_bytes)?;
        }

        Ok(())
    }

    fn read_state_from<R: Read>(r: &mut R) -> io::Result<Self> {
        let version = storage::read_u32_le(r)?;
        if version != 1 && version != 2 && version != 3 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "bad experts state version",
            ));
        }

        let enabled = storage::read_exact::<1, _>(r)?[0] != 0;
        let pm = storage::read_exact::<1, _>(r)?[0];
        let persistence_mode = match pm {
            0 => ExpertsPersistenceMode::Full,
            1 => ExpertsPersistenceMode::DropActive,
            _ => ExpertsPersistenceMode::Full,
        };

        let next_id = storage::read_u32_le(r)?;

        let pl = storage::read_exact::<1, _>(r)?[0];
        let parent_learning = match pl {
            0 => ParentLearningPolicy::Normal,
            1 => ParentLearningPolicy::Reduced,
            _ => ParentLearningPolicy::Holdout,
        };
        let max_children = storage::read_u32_le(r)? as usize;
        let child_reward_scale = storage::read_f32_le(r)?;
        let episode_trials = storage::read_u32_le(r)?;
        let consolidate_topk = storage::read_u32_le(r)? as usize;
        let consolidate_delta_max = storage::read_f32_le(r)?;
        // v1/v2 stored a legacy uncertainty-gap threshold here.
        if version <= 2 {
            let _legacy_spawn_confidence_gap = storage::read_f32_le(r)?;
        }
        let (
            reward_shift_ema_delta_threshold,
            performance_collapse_drop_threshold,
            performance_collapse_baseline_min,
        ) = if version >= 3 {
            (
                storage::read_f32_le(r)?,
                storage::read_f32_le(r)?,
                storage::read_f32_le(r)?,
            )
        } else {
            (0.55, 0.65, 0.25)
        };
        let spawn_min_trials = storage::read_u32_le(r)?;
        let cooldown_trials = storage::read_u32_le(r)?;
        let promote_reward_ema = storage::read_f32_le(r)?;
        let allow_nested = storage::read_exact::<1, _>(r)?[0] != 0;
        let max_depth = storage::read_u32_le(r)?;

        let policy = ExpertPolicy {
            parent_learning,
            max_children,
            child_reward_scale,
            episode_trials,
            consolidate_topk,
            consolidate_delta_max,
            reward_shift_ema_delta_threshold,
            performance_collapse_drop_threshold,
            performance_collapse_baseline_min,
            spawn_min_trials,
            cooldown_trials,
            promote_reward_ema,
            allow_nested,
            max_depth,
        };

        let last_spawn_reason = storage::read_string(r)?;
        let last_consolidation = storage::read_string(r)?;

        let cooldown_n = storage::read_u32_le(r)? as usize;
        let mut cooldown_by_context: HashMap<String, u32> = HashMap::new();
        for _ in 0..cooldown_n {
            let k = storage::read_string(r)?;
            let v = storage::read_u32_le(r)?;
            cooldown_by_context.insert(k, v);
        }

        let mut context_stats: HashMap<String, ContextStats> = HashMap::new();
        if version >= 2 {
            let stat_n = storage::read_u32_le(r)? as usize;
            for _ in 0..stat_n {
                let k = storage::read_string(r)?;
                let first_seen_trial = storage::read_u32_le(r)?;
                let trials_seen = storage::read_u32_le(r)?;
                let reward_fast_ema = storage::read_f32_le(r)?;
                let reward_slow_ema = storage::read_f32_le(r)?;
                let best_slow_ema = storage::read_f32_le(r)?;
                context_stats.insert(
                    k,
                    ContextStats {
                        first_seen_trial,
                        trials_seen,
                        reward_fast_ema,
                        reward_slow_ema,
                        best_slow_ema,
                    },
                );
            }
        }

        let expert_n = storage::read_u32_le(r)? as usize;
        let mut experts: Vec<Expert> = Vec::with_capacity(expert_n);
        for _ in 0..expert_n {
            let id = storage::read_u32_le(r)?;
            let context_key = storage::read_string(r)?;
            let age_steps = storage::read_u64_le(r)?;
            let completed_trials = storage::read_u32_le(r)?;
            let episode_trials = storage::read_u32_le(r)?;
            let reward_ema = storage::read_f32_le(r)?;

            let brain_bytes = storage::read_bytes(r)?;
            let fork_bytes = storage::read_bytes(r)?;
            let child_bytes = storage::read_bytes(r)?;

            let mut bcur = std::io::Cursor::new(brain_bytes);
            let brain = Brain::load_image_from(&mut bcur)?;
            let mut fcur = std::io::Cursor::new(fork_bytes);
            let fork_point = Brain::load_image_from(&mut fcur)?;
            let children = {
                let mut ccur = std::io::Cursor::new(child_bytes);
                Box::new(Self::read_state_from(&mut ccur)?)
            };

            experts.push(Expert {
                id,
                context_key,
                brain,
                fork_point,
                children,
                age_steps,
                completed_trials,
                episode_trials,
                reward_ema,
            });
        }

        Ok(Self {
            enabled,
            policy,
            persistence_mode,
            next_id,
            experts,
            cooldown_by_context,
            context_stats,
            last_spawn_reason,
            last_consolidation,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use braine::substrate::BrainConfig;

    fn small_brain() -> Brain {
        let cfg = BrainConfig {
            unit_count: 32,
            connectivity_per_unit: 4,
            seed: Some(1),
            ..BrainConfig::default()
        };
        Brain::new(cfg)
    }

    #[test]
    fn spawns_on_novel_context_after_first_trial() {
        let mut em = ExpertManager::new();
        em.set_enabled(true);
        let brain = small_brain();

        em.note_trial_for_spawn_target_under_path("ctx_a", &[], 1, 0.0);
        em.maybe_spawn_for_signals_under_path("ctx_a", &[], 1, &brain);

        assert_eq!(em.experts.len(), 1);
        assert!(em.last_spawn_reason.contains("novel_context"));
    }

    #[test]
    fn does_not_spawn_without_any_signal() {
        let mut em = ExpertManager::new();
        em.set_enabled(true);
        let brain = small_brain();

        // First trial spawns (novel).
        em.note_trial_for_spawn_target_under_path("ctx_a", &[], 1, 0.2);
        em.maybe_spawn_for_signals_under_path("ctx_a", &[], 1, &brain);
        assert_eq!(em.experts.len(), 1);

        // While active expert exists, should not spawn another.
        em.note_trial_for_spawn_target_under_path("ctx_a", &[], 2, 0.2);
        em.maybe_spawn_for_signals_under_path("ctx_a", &[], 2, &brain);
        assert_eq!(em.experts.len(), 1);
    }

    #[test]
    fn spawns_on_performance_collapse_signal() {
        let mut em = ExpertManager::new();
        em.set_enabled(true);
        let brain = small_brain();

        // Build baseline performance for ctx_b.
        for t in 1..=25 {
            em.note_trial_for_spawn_target_under_path("ctx_b", &[], t, 0.9);
        }

        // No spawn yet (novelty would have applied at t=1, but we didn't call maybe_spawn).
        assert_eq!(em.experts.len(), 0);

        // Sudden collapse.
        for t in 26..=35 {
            em.note_trial_for_spawn_target_under_path("ctx_b", &[], t, -0.9);
        }

        em.maybe_spawn_for_signals_under_path("ctx_b", &[], 35, &brain);
        assert_eq!(em.experts.len(), 1);
        assert!(
            em.last_spawn_reason.contains("performance_collapse")
                || em.last_spawn_reason.contains("reward_shift")
        );
    }
}
