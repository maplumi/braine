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
        }
    }

    pub fn spawn_child(&mut self, spec: ChildSpec, seed: u64, cfg_overrides: ChildConfigOverrides) {
        let mut child = self.parent.spawn_child(seed, cfg_overrides);

        // Ensure the child has the new stimulus symbol/group defined.
        // For a true system the Frame would map new signals; here we just allocate a new sensor group.
        child.ensure_sensor(&spec.stimulus_name, 6);

        self.children.push(ChildBrain {
            name: spec.name.clone(),
            brain: child,
            remaining: spec.budget_steps,
            spec,
        });
    }

    pub fn step_children(&mut self) {
        let mut i = 0;
        while i < self.children.len() {
            if self.children[i].remaining == 0 {
                i += 1;
                continue;
            }

            let target_action = self.children[i].spec.target_action.clone();

            // Run one learning step for this child.
            let (stim, reward) = child_environment_step(&self.children[i].spec, self.children[i].remaining);

            self.children[i].brain.apply_stimulus(stim);
            self.children[i].brain.set_neuromodulator(reward);
            self.children[i].brain.step();

            let mut policy = ActionPolicy::Deterministic;
            let (action, _score) = self.children[i].brain.select_action(&mut policy);
            self.children[i].brain.note_action(&action);

            // Reinforce toward the child's target.
            if action == target_action {
                self.children[i].brain.reinforce_action(&target_action, 0.6);
            } else {
                self.children[i].brain.reinforce_action(&target_action, 0.2);
            }

            self.children[i].brain.commit_observation();

            self.children[i].remaining -= 1;
            i += 1;
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

