use crate::causality::SymbolId;
use crate::substrate::{Brain, Diagnostics};
use crate::supervisor::Supervisor;

/// A read-only snapshot of what the brain is doing.
///
/// Design intent:
/// - Observers cannot mutate or steer the brain.
/// - Snapshotting is *on-demand* and can allocate; the hot loop stays unchanged.
/// - For per-step details (reinforcements, last committed symbols), enable telemetry with
///   `Brain::set_observer_telemetry(true)`.
#[derive(Debug, Clone)]
pub struct BrainSnapshot {
    pub age_steps: u64,
    pub neuromodulator: f32,
    pub diagnostics: Diagnostics,
    pub causal: crate::causality::CausalStats,

    pub last_stimuli: Vec<String>,
    pub last_actions: Vec<String>,
    pub last_reinforced_actions: Vec<(String, f32)>,
    pub last_committed_symbols: Vec<String>,
}

pub struct BrainAdapter<'a> {
    brain: &'a Brain,
}

impl<'a> BrainAdapter<'a> {
    pub fn new(brain: &'a Brain) -> Self {
        Self { brain }
    }

    pub fn snapshot(&self) -> BrainSnapshot {
        let diagnostics = self.brain.diagnostics();
        let causal = self.brain.causal_stats();

        BrainSnapshot {
            age_steps: self.brain.age_steps(),
            neuromodulator: self.brain.neuromodulator(),
            diagnostics,
            causal,

            last_stimuli: ids_to_names(self.brain, self.brain.last_stimuli_symbols()),
            last_actions: ids_to_names(self.brain, self.brain.last_action_symbols()),
            last_reinforced_actions: self
                .brain
                .last_reinforced_action_symbols()
                .iter()
                .filter_map(|(id, delta)| Some((self.brain.symbol_name(*id)?.to_string(), *delta)))
                .collect(),
            last_committed_symbols: ids_to_names(self.brain, self.brain.last_committed_symbols()),
        }
    }
}

fn ids_to_names(brain: &Brain, ids: &[SymbolId]) -> Vec<String> {
    ids.iter()
        .filter_map(|id| brain.symbol_name(*id).map(|s| s.to_string()))
        .collect()
}

#[derive(Debug, Clone)]
pub struct SupervisorSnapshot {
    pub parent: BrainSnapshot,
    pub children: Vec<ChildSnapshot>,
}

#[derive(Debug, Clone)]
pub struct ChildSnapshot {
    pub name: String,
    pub id: u64,
    pub parent_id: u64,
    pub remaining_steps: usize,
    pub stimulus_name: String,
    pub target_action: String,
}

pub struct SupervisorAdapter<'a> {
    sup: &'a Supervisor,
}

impl<'a> SupervisorAdapter<'a> {
    pub fn new(sup: &'a Supervisor) -> Self {
        Self { sup }
    }

    pub fn snapshot(&self) -> SupervisorSnapshot {
        let parent = BrainAdapter::new(&self.sup.parent).snapshot();
        let children = self
            .sup
            .children
            .iter()
            .map(|c| ChildSnapshot {
                name: c.name.clone(),
                id: c.id,
                parent_id: c.parent_id,
                remaining_steps: c.remaining,
                stimulus_name: c.spec.stimulus_name.clone(),
                target_action: c.spec.target_action.clone(),
            })
            .collect();

        SupervisorSnapshot { parent, children }
    }
}
