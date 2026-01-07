use crate::substrate::{ActionPolicy, Brain, Stimulus};

#[derive(Debug, Clone)]
pub struct ChildSpec {
    pub name: String,
    pub budget_steps: usize,

    // What new sensor signal is this child trying to learn?
    pub stimulus_name: String,

    // What action should this stimulus map to (for this demo objective)?
    pub target_action: String,
}

pub struct ChildBrain {
    pub name: String,
    pub brain: Brain,
    pub remaining: usize,
    pub spec: ChildSpec,

    pub id: u64,
    pub parent_id: u64,

    guidance_action: Option<String>,
    requests: Vec<ChildRequest>,
}

#[derive(Debug, Clone)]
pub enum ChildRequest {
    MoreMilk {
        extra_steps: usize,
        reason: &'static str,
    },
    Guidance,
    SpawnGrandchild {
        extra_steps: usize,
    },
}

#[derive(Debug, Clone, Copy)]
pub struct ConsolidationPolicy {
    pub weight_threshold: f32,
    pub merge_rate: f32,
}

pub struct Supervisor {
    pub parent: Brain,
    pub children: Vec<ChildBrain>,
    pub policy: ConsolidationPolicy,

    max_parallelism: usize,
    milk_pool_steps: usize,
    allow_recursive_spawning: bool,
    next_child_id: u64,
}

impl Supervisor {
    pub fn new(parent: Brain) -> Self {
        Self {
            parent,
            children: Vec::new(),
            policy: ConsolidationPolicy {
                weight_threshold: 0.15,
                merge_rate: 0.35,
            },

            max_parallelism: 1,
            milk_pool_steps: 0,
            allow_recursive_spawning: false,
            next_child_id: 1,
        }
    }

    /// Upper bound on parallel worker threads used when stepping children.
    /// `1` preserves deterministic sequential behavior.
    pub fn set_max_parallelism(&mut self, threads: usize) {
        self.max_parallelism = threads.max(1);
    }

    /// Budget pool the parent can use to sustain children beyond their initial budget.
    /// This models "milk/pocket money" without giving children direct access to parent state.
    pub fn add_milk_pool_steps(&mut self, steps: usize) {
        self.milk_pool_steps = self.milk_pool_steps.saturating_add(steps);
    }

    pub fn set_recursive_spawning(&mut self, enabled: bool) {
        self.allow_recursive_spawning = enabled;
    }

    pub fn spawn_child(&mut self, spec: ChildSpec, seed: u64, cfg_overrides: ChildConfigOverrides) {
        let mut child = self.parent.spawn_child(seed, cfg_overrides);

        // Ensure the child has the new stimulus symbol/group defined.
        // For a true system the Frame would map new signals; here we just allocate a new sensor group.
        child.ensure_sensor(&spec.stimulus_name, 6);

        let id = self.next_child_id;
        self.next_child_id = self.next_child_id.wrapping_add(1);

        self.children.push(ChildBrain {
            name: spec.name.clone(),
            brain: child,
            remaining: spec.budget_steps,
            spec,
            id,
            parent_id: 0,
            guidance_action: None,
            requests: Vec::new(),
        });
    }

    pub fn step_children(&mut self) {
        self.step_children_parallel(self.max_parallelism);
        self.handle_child_requests();
    }

    fn step_children_parallel(&mut self, max_threads: usize) {
        let live = self.children.iter().filter(|c| c.remaining > 0).count();
        if live <= 1 || max_threads <= 1 {
            for child in &mut self.children {
                step_one_child(child);
            }
            return;
        }

        let threads = max_threads.min(self.children.len()).max(1);
        let chunk = (self.children.len() + threads - 1) / threads;

        std::thread::scope(|scope| {
            for segment in self.children.chunks_mut(chunk) {
                scope.spawn(move || {
                    for child in segment {
                        step_one_child(child);
                    }
                });
            }
        });
    }

    fn handle_child_requests(&mut self) {
        // Drain requests and decide whether to extend budgets or provide guidance.
        let mut spawned: Vec<(ChildSpec, u64, ChildConfigOverrides, u64)> = Vec::new();

        for child in &mut self.children {
            if child.requests.is_empty() {
                continue;
            }

            let mut reqs = Vec::new();
            core::mem::swap(&mut reqs, &mut child.requests);

            for req in reqs {
                match req {
                    ChildRequest::MoreMilk { extra_steps, .. } => {
                        let grant = extra_steps.min(self.milk_pool_steps);
                        if grant > 0 {
                            child.remaining = child.remaining.saturating_add(grant);
                            self.milk_pool_steps -= grant;
                        }
                    }
                    ChildRequest::Guidance => {
                        let hint = self.parent.meaning_hint(&child.spec.stimulus_name);
                        child.guidance_action = hint.map(|(a, _)| a);
                    }
                    ChildRequest::SpawnGrandchild { extra_steps } => {
                        if !self.allow_recursive_spawning {
                            continue;
                        }
                        let grant = extra_steps.min(self.milk_pool_steps);
                        if grant == 0 {
                            continue;
                        }
                        self.milk_pool_steps -= grant;

                        let grand_spec = ChildSpec {
                            name: format!("{}_grandchild_{}", child.name, self.next_child_id),
                            budget_steps: grant,
                            stimulus_name: child.spec.stimulus_name.clone(),
                            target_action: child.spec.target_action.clone(),
                        };
                        let seed = 10_000 + self.next_child_id;
                        self.next_child_id = self.next_child_id.wrapping_add(1);

                        // Grandchild exploration: slightly higher noise/plasticity.
                        let overrides = ChildConfigOverrides {
                            noise_amp: 0.05,
                            noise_phase: 0.025,
                            hebb_rate: 0.16,
                            forget_rate: 0.0013,
                        };

                        // The supervisor will spawn this after the loop to avoid borrowing issues.
                        spawned.push((grand_spec, seed, overrides, child.id));
                    }
                }
            }
        }

        for (spec, seed, overrides, parent_id) in spawned {
            let mut grand =
                if let Some(parent_child) = self.children.iter().find(|c| c.id == parent_id) {
                    parent_child.brain.spawn_child(seed, overrides)
                } else {
                    self.parent.spawn_child(seed, overrides)
                };

            grand.ensure_sensor(&spec.stimulus_name, 6);

            let id = self.next_child_id;
            self.next_child_id = self.next_child_id.wrapping_add(1);

            self.children.push(ChildBrain {
                name: spec.name.clone(),
                brain: grand,
                remaining: spec.budget_steps,
                spec,
                id,
                parent_id,
                guidance_action: None,
                requests: Vec::new(),
            });
        }
    }

    /// Score children by how strongly the target action is linked to positive reward
    /// under the stimulus (cheap causality-based fitness).
    pub fn score_children(&self) -> Vec<(usize, f32)> {
        let mut scored: Vec<(usize, f32)> = Vec::new();
        for (idx, c) in self.children.iter().enumerate() {
            let hint = c.brain.meaning_hint(&c.spec.stimulus_name);
            let score = match hint {
                Some((action, s)) if action == c.spec.target_action => s,
                Some((_action, s)) => s * 0.5,
                None => 0.0,
            };
            scored.push((idx, score));
        }
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scored
    }

    pub fn consolidate_best(&mut self) -> Option<(String, f32)> {
        let scored = self.score_children();
        let (best_idx, best_score) = *scored.first()?;

        let child_name = self.children[best_idx].name.clone();
        let child_brain = &self.children[best_idx].brain;

        self.parent.consolidate_from(child_brain, self.policy);

        // Keep only the winner for inspection; drop the rest.
        self.children.clear();

        Some((child_name, best_score))
    }
}

#[derive(Debug, Clone, Copy)]
pub struct ChildConfigOverrides {
    pub noise_amp: f32,
    pub noise_phase: f32,
    pub hebb_rate: f32,
    pub forget_rate: f32,
}

impl Default for ChildConfigOverrides {
    fn default() -> Self {
        Self {
            noise_amp: 0.03,
            noise_phase: 0.015,
            hebb_rate: 0.12,
            forget_rate: 0.0012,
        }
    }
}

fn child_environment_step(spec: &ChildSpec, remaining: usize) -> (Stimulus<'static>, f32) {
    // Minimal "new signal" environment:
    // - Always present the new stimulus.
    // - Reward is positive most of the time.
    //
    // The point is not realism; it's to let the child build a causal link
    // stimulus -> action -> reward in its sandbox.
    let strength = if remaining % 17 == 0 { 0.35 } else { 1.0 };

    // Leak a tiny negative reward sometimes so meaning isn't trivial.
    let reward = if remaining % 41 == 0 { -0.4 } else { 0.7 };

    // We need a 'static str for the demo; this is fine because the spec strings are owned.
    // To avoid lifetime complexity, we map on known demo names.
    // If you add new names, extend this mapping.
    let stim_name: &'static str = if spec.stimulus_name == "vision_new" {
        "vision_new"
    } else {
        "vision_new"
    };

    (Stimulus::new(stim_name, strength), reward)
}

fn step_one_child(child: &mut ChildBrain) {
    if child.remaining == 0 {
        return;
    }

    let target_action = child.spec.target_action.as_str();

    // Run one learning step for this child.
    let (stim, reward) = child_environment_step(&child.spec, child.remaining);

    child.brain.apply_stimulus(stim);
    child.brain.set_neuromodulator(reward);
    child.brain.step();

    let mut policy = ActionPolicy::Deterministic;
    let (action, _score) = child.brain.select_action(&mut policy);
    child.brain.note_action(&action);

    // Reinforce toward the child's target.
    if action == target_action {
        let delta = if child.guidance_action.as_deref() == Some(target_action) {
            0.65
        } else {
            0.6
        };
        child.brain.reinforce_action(target_action, delta);
    } else {
        child.brain.reinforce_action(target_action, 0.2);
    }

    child.brain.commit_observation();
    child.remaining = child.remaining.saturating_sub(1);

    // Child requests: cheap heuristics.
    if child.remaining == 0 {
        return;
    }

    // If it's doing well but about to run out of time, ask for more milk.
    if child.remaining < 40 {
        if let Some((act, score)) = child.brain.meaning_hint(&child.spec.stimulus_name) {
            if act == target_action && score > 0.12 {
                child.requests.push(ChildRequest::MoreMilk {
                    extra_steps: 200,
                    reason: "nearly_solved",
                });
            }
        }
    }

    // Occasionally ask for parent guidance.
    if child.remaining % 120 == 0 {
        child.requests.push(ChildRequest::Guidance);
    }

    // If stuck for a long time, request spawning a grandchild explorer.
    if child.remaining % 300 == 0 {
        if child
            .brain
            .meaning_hint(&child.spec.stimulus_name)
            .is_none()
        {
            child
                .requests
                .push(ChildRequest::SpawnGrandchild { extra_steps: 250 });
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::substrate::BrainConfig;

    fn make_test_brain() -> Brain {
        let cfg = BrainConfig::with_size(64, 8).with_seed(42);
        let mut brain = Brain::new(cfg);
        brain.define_sensor("test_stim", 4);
        brain.define_action("test_act", 4);
        brain
    }

    #[test]
    fn supervisor_spawn_child() {
        let parent = make_test_brain();
        let mut sup = Supervisor::new(parent);

        assert_eq!(sup.children.len(), 0);

        let spec = ChildSpec {
            name: "child1".to_string(),
            budget_steps: 100,
            stimulus_name: "new_signal".to_string(),
            target_action: "test_act".to_string(),
        };

        sup.spawn_child(spec, 123, ChildConfigOverrides::default());

        assert_eq!(sup.children.len(), 1);
        assert_eq!(sup.children[0].name, "child1");
        assert_eq!(sup.children[0].remaining, 100);
    }

    #[test]
    fn supervisor_step_consumes_budget() {
        let parent = make_test_brain();
        let mut sup = Supervisor::new(parent);

        let spec = ChildSpec {
            name: "child1".to_string(),
            budget_steps: 10,
            stimulus_name: "signal".to_string(),
            target_action: "test_act".to_string(),
        };

        sup.spawn_child(spec, 123, ChildConfigOverrides::default());

        let initial_remaining = sup.children[0].remaining;
        sup.step_children();
        
        assert!(sup.children[0].remaining < initial_remaining, 
            "Budget should decrease after step");
    }

    #[test]
    fn supervisor_milk_pool() {
        let parent = make_test_brain();
        let mut sup = Supervisor::new(parent);

        sup.add_milk_pool_steps(500);

        let spec = ChildSpec {
            name: "child1".to_string(),
            budget_steps: 10,
            stimulus_name: "signal".to_string(),
            target_action: "test_act".to_string(),
        };

        sup.spawn_child(spec, 123, ChildConfigOverrides::default());

        // Manually add a milk request
        sup.children[0].requests.push(ChildRequest::MoreMilk {
            extra_steps: 50,
            reason: "test",
        });

        let budget_before = sup.children[0].remaining;
        sup.step_children(); // This handles requests

        // Budget should increase from milk pool
        assert!(sup.children[0].remaining >= budget_before, 
            "Child should receive milk pool steps");
    }

    #[test]
    fn supervisor_consolidation_policy() {
        let parent = make_test_brain();
        let sup = Supervisor::new(parent);

        assert!(sup.policy.weight_threshold > 0.0);
        assert!(sup.policy.merge_rate > 0.0);
        assert!(sup.policy.merge_rate <= 1.0);
    }

    #[test]
    fn supervisor_parallelism_setting() {
        let parent = make_test_brain();
        let mut sup = Supervisor::new(parent);

        sup.set_max_parallelism(4);
        // Setting to 0 should clamp to 1
        sup.set_max_parallelism(0);
        
        // spawn multiple children and step to verify no panic
        for i in 0..3 {
            let spec = ChildSpec {
                name: format!("child{}", i),
                budget_steps: 5,
                stimulus_name: "signal".to_string(),
                target_action: "test_act".to_string(),
            };
            sup.spawn_child(spec, i as u64, ChildConfigOverrides::default());
        }

        sup.set_max_parallelism(2);
        sup.step_children(); // Should work with parallel stepping
    }

    #[test]
    fn supervisor_recursive_spawning_disabled_by_default() {
        let parent = make_test_brain();
        let mut sup = Supervisor::new(parent);
        sup.add_milk_pool_steps(1000);

        let spec = ChildSpec {
            name: "child1".to_string(),
            budget_steps: 10,
            stimulus_name: "signal".to_string(),
            target_action: "test_act".to_string(),
        };

        sup.spawn_child(spec, 123, ChildConfigOverrides::default());

        // Request grandchild spawn
        sup.children[0].requests.push(ChildRequest::SpawnGrandchild {
            extra_steps: 100,
        });

        sup.step_children();

        // Should still be just 1 child (recursive disabled by default)
        assert_eq!(sup.children.len(), 1, 
            "Grandchild should not spawn when recursive is disabled");
    }

    #[test]
    fn supervisor_recursive_spawning_enabled() {
        let parent = make_test_brain();
        let mut sup = Supervisor::new(parent);
        sup.add_milk_pool_steps(1000);
        sup.set_recursive_spawning(true);

        let spec = ChildSpec {
            name: "child1".to_string(),
            budget_steps: 10,
            stimulus_name: "signal".to_string(),
            target_action: "test_act".to_string(),
        };

        sup.spawn_child(spec, 123, ChildConfigOverrides::default());

        // Request grandchild spawn
        sup.children[0].requests.push(ChildRequest::SpawnGrandchild {
            extra_steps: 100,
        });

        sup.step_children();

        // Should now have 2 children (grandchild spawned)
        assert_eq!(sup.children.len(), 2, 
            "Grandchild should spawn when recursive is enabled");
    }

    #[test]
    fn child_config_overrides_default() {
        let overrides = ChildConfigOverrides::default();
        
        // Default overrides should have reasonable exploration values
        assert!(overrides.noise_amp >= 0.0);
        assert!(overrides.hebb_rate >= 0.0);
    }
}
