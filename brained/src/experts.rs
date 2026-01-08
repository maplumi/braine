use braine::{
    storage,
    substrate::{Brain, BrainDelta},
};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::io::{self, Read, Write};

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

    /// Spawn trigger: if parent is uncertain (top score gap below this), consider spawning.
    pub spawn_confidence_gap: f32,

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
            spawn_confidence_gap: 0.03,
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

    pub fn maybe_spawn_for_uncertainty(
        &mut self,
        context_key: &str,
        parent_trials: u32,
        confidence_gap: Option<f32>,
        parent: &Brain,
    ) {
        if !self.enabled {
            return;
        }
        if self.experts.len() >= self.policy.max_children {
            return;
        }
        if parent_trials < self.policy.spawn_min_trials {
            return;
        }
        if self.active_expert_index(context_key).is_some() {
            return;
        }
        if self.cooldown_by_context.contains_key(context_key) {
            return;
        }

        let Some(gap) = confidence_gap else {
            return;
        };
        if gap > self.policy.spawn_confidence_gap {
            return;
        }

        let id = self.next_id;
        self.next_id += 1;

        self.last_spawn_reason = format!(
            "spawned expert id={} ctx='{}' (confidence_gap={:.4})",
            id, context_key, gap
        );
        self.experts.push(Expert::new(
            id,
            context_key.to_string(),
            parent,
            &self.policy,
        ));
    }

    /// Spawn under the currently controlling expert chain (nested spawn).
    ///
    /// If `controller_path` is empty, this behaves like `maybe_spawn_for_uncertainty`.
    pub fn maybe_spawn_for_uncertainty_under_path(
        &mut self,
        context_key: &str,
        controller_path: &[u32],
        parent_trials: u32,
        confidence_gap: Option<f32>,
        root_parent: &Brain,
    ) {
        if controller_path.is_empty() {
            self.maybe_spawn_for_uncertainty(
                context_key,
                parent_trials,
                confidence_gap,
                root_parent,
            );
            return;
        }

        // Traverse to the deepest controlling expert and attempt to spawn into its child manager.
        let mut cur: &mut ExpertManager = self;
        for (i, id) in controller_path.iter().enumerate() {
            let Some(idx) = cur.experts.iter().position(|e| e.id == *id) else {
                return;
            };

            let e: &mut Expert = &mut cur.experts[idx];
            let last = i + 1 == controller_path.len();
            if last {
                // Only attempt nested spawn if nesting is enabled at this level.
                if !cur.policy.allow_nested || cur.policy.max_depth <= (i as u32 + 1) {
                    return;
                }
                e.children.maybe_spawn_for_uncertainty(
                    context_key,
                    parent_trials,
                    confidence_gap,
                    &e.brain,
                );
                return;
            }

            cur = &mut e.children;
        }
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
        storage::write_u32_le(w, 1)?;

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
        storage::write_f32_le(w, self.policy.spawn_confidence_gap)?;
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
        if version != 1 {
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
        let spawn_confidence_gap = storage::read_f32_le(r)?;
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
            spawn_confidence_gap,
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
            last_spawn_reason,
            last_consolidation,
        })
    }
}
