use crate::stats::GameStats;
use crate::time::{Duration, Instant};
use serde::{Deserialize, Serialize};

#[cfg(feature = "braine")]
use braine::substrate::{Brain, Stimulus};

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ReplayStimulus {
    pub name: String,
    #[serde(default = "default_strength")]
    pub strength: f32,
}

fn default_strength() -> f32 {
    1.0
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ReplayTrial {
    /// Stimuli presented on this trial.
    #[serde(default)]
    pub stimuli: Vec<ReplayStimulus>,

    /// Allowed actions for this trial.
    #[serde(default)]
    pub allowed_actions: Vec<String>,

    /// The correct action label.
    #[serde(default)]
    pub correct_action: String,

    /// Optional identifier for debugging / provenance.
    #[serde(default)]
    pub id: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ReplayDataset {
    #[serde(default)]
    pub name: String,
    #[serde(default)]
    pub trials: Vec<ReplayTrial>,
}

impl ReplayDataset {
    pub fn builtin_left_right_spot() -> Self {
        // A minimal supervised-style dataset that still exercises the same closed-loop
        // reinforcement update path: reward is emitted based on correctness.
        //
        // Stimuli re-use existing Spot sensor names so the brain already has widths.
        let mut trials = Vec::new();
        for _ in 0..50 {
            trials.push(ReplayTrial {
                id: "L".to_string(),
                stimuli: vec![ReplayStimulus {
                    name: "spot_left".to_string(),
                    strength: 1.0,
                }],
                allowed_actions: vec!["left".to_string(), "right".to_string()],
                correct_action: "left".to_string(),
            });
            trials.push(ReplayTrial {
                id: "R".to_string(),
                stimuli: vec![ReplayStimulus {
                    name: "spot_right".to_string(),
                    strength: 1.0,
                }],
                allowed_actions: vec!["left".to_string(), "right".to_string()],
                correct_action: "right".to_string(),
            });
        }

        Self {
            name: "builtin_spot_lr".to_string(),
            trials,
        }
    }
}

/// Dataset-driven game: each completed trial consumes one record from a list.
///
/// This is designed for validating the LLM/advisor integration boundary:
/// - Braine runs its normal dynamics and selects actions.
/// - Reward is computed from trial correctness.
/// - Advisor can be invoked to nudge configuration, but cannot choose actions.
#[derive(Debug)]
pub struct ReplayGame {
    dataset_name: String,
    trials: Vec<ReplayTrial>,
    idx: usize,

    pub trial_frame: u32,
    pub response_made: bool,
    pub last_action: Option<String>,
    pub stats: GameStats,

    trial_started_at: Instant,
    stimulus_key: String,
}

impl Default for ReplayGame {
    fn default() -> Self {
        Self::new(ReplayDataset::builtin_left_right_spot())
    }
}

impl ReplayGame {
    pub fn new(dataset: ReplayDataset) -> Self {
        let now = Instant::now();
        let mut g = Self {
            dataset_name: if dataset.name.trim().is_empty() {
                "replay".to_string()
            } else {
                dataset.name
            },
            trials: dataset.trials,
            idx: 0,
            trial_frame: 0,
            response_made: false,
            last_action: None,
            stats: GameStats::new(),
            trial_started_at: now,
            stimulus_key: String::new(),
        };
        g.refresh_stimulus_key();
        g
    }

    pub fn set_dataset(&mut self, dataset: ReplayDataset) {
        self.dataset_name = if dataset.name.trim().is_empty() {
            "replay".to_string()
        } else {
            dataset.name
        };
        self.trials = dataset.trials;
        self.idx = 0;
        self.stats = GameStats::new();
        self.response_made = false;
        self.last_action = None;
        self.trial_started_at = Instant::now();
        self.refresh_stimulus_key();
    }

    pub fn dataset_name(&self) -> &str {
        &self.dataset_name
    }

    pub fn stimulus_name(&self) -> &'static str {
        "replay"
    }

    pub fn stimulus_key(&self) -> &str {
        &self.stimulus_key
    }

    pub fn allowed_actions(&self) -> &[String] {
        self.current_trial()
            .map(|t| t.allowed_actions.as_slice())
            .unwrap_or(&[])
    }

    pub fn correct_action(&self) -> &str {
        self.current_trial()
            .map(|t| t.correct_action.as_str())
            .unwrap_or("idle")
    }

    pub fn current_trial(&self) -> Option<&ReplayTrial> {
        if self.trials.is_empty() {
            None
        } else {
            Some(&self.trials[self.idx % self.trials.len()])
        }
    }

    pub fn total_trials(&self) -> usize {
        self.trials.len()
    }

    pub fn index(&self) -> usize {
        self.idx
    }

    pub fn current_trial_id(&self) -> &str {
        self.current_trial().map(|t| t.id.as_str()).unwrap_or("")
    }

    pub fn update_timing(&mut self, trial_period_ms: u32) {
        let trial_period_ms = trial_period_ms.clamp(10, 60_000);
        let trial_period = Duration::from_millis(trial_period_ms as u64);

        let now = Instant::now();
        let elapsed = now.duration_since(self.trial_started_at);
        if elapsed >= trial_period {
            self.response_made = false;
            self.last_action = None;
            self.trial_started_at = now;
        }

        let elapsed = now.duration_since(self.trial_started_at);
        self.trial_frame = elapsed.as_millis().min(u32::MAX as u128) as u32;
    }

    #[cfg(feature = "braine")]
    pub fn apply_stimuli(&self, brain: &mut Brain) {
        if let Some(t) = self.current_trial() {
            for s in &t.stimuli {
                brain.apply_stimulus(Stimulus::new(s.name.as_str(), s.strength));
            }
        }
    }

    pub fn score_action(&mut self, action: &str) -> Option<(f32, bool)> {
        if self.response_made {
            return None;
        }

        let correct = action == self.correct_action();
        let reward = if correct { 1.0 } else { -1.0 };

        self.response_made = true;
        self.last_action = Some(action.to_string());
        self.stats.record_trial(correct);

        // Advance dataset index on completed trial.
        if !self.trials.is_empty() {
            self.idx = self.idx.wrapping_add(1);
        }
        self.refresh_stimulus_key();

        Some((reward, true))
    }

    fn refresh_stimulus_key(&mut self) {
        // Keep the meaning context stable within a dataset.
        // We do not include the trial index to avoid exploding context space.
        self.stimulus_key = format!("replay::{}", self.dataset_name);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn produces_rewards_and_advances() {
        let mut g = ReplayGame::default();
        assert!(g.total_trials() > 0);

        let a0 = g.correct_action().to_string();
        let (r0, done0) = g.score_action(&a0).unwrap();
        assert_eq!(r0, 1.0);
        assert!(done0);
        g.response_made = false;

        let (r1, _done1) = g.score_action("not_a_real_action").unwrap();
        assert_eq!(r1, -1.0);
    }
}
