use crate::stats::GameStats;
use std::time::{Duration, Instant};

// ─────────────────────────────────────────────────────────────────────────
// Spot reversal: the stimulus side stays the same, but the correct action
// mapping flips once after a fixed number of trials.
// ─────────────────────────────────────────────────────────────────────────

#[derive(Debug)]
pub struct SpotReversalGame {
    pub spot_is_left: bool,
    pub reversal_active: bool,
    pub flip_after_trials: u32,

    pub trial_frame: u32,
    pub response_made: bool,
    pub last_action: Option<String>,
    pub stats: GameStats,

    rng_seed: u64,
    trial_started_at: Instant,
}

impl SpotReversalGame {
    pub fn new(flip_after_trials: u32) -> Self {
        let now = Instant::now();
        let mut g = Self {
            spot_is_left: true,
            reversal_active: false,
            flip_after_trials: flip_after_trials.max(1),
            trial_frame: 0,
            response_made: false,
            last_action: None,
            stats: GameStats::new(),
            rng_seed: 0xC0FF_EE12u64,
            trial_started_at: now,
        };
        g.new_trial();
        g
    }

    fn update_reversal(&mut self) {
        if !self.reversal_active && self.stats.trials >= self.flip_after_trials {
            self.reversal_active = true;
        }
    }

    fn new_trial(&mut self) {
        self.trial_frame = 0;
        self.response_made = false;
        self.last_action = None;
        self.trial_started_at = Instant::now();

        // Random side
        self.rng_seed = self
            .rng_seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1);
        let u = (self.rng_seed >> 11) as u32;
        self.spot_is_left = (u & 1) == 0;

        self.update_reversal();
    }

    pub fn update_timing(&mut self, trial_period_ms: u32) {
        let trial_period_ms = trial_period_ms.clamp(10, 60_000);
        let trial_period = Duration::from_millis(trial_period_ms as u64);

        let now = Instant::now();
        let elapsed = now.duration_since(self.trial_started_at);

        if elapsed >= trial_period {
            self.new_trial();
        }

        let now = Instant::now();
        let elapsed = now.duration_since(self.trial_started_at);
        self.trial_frame = elapsed.as_millis().min(u32::MAX as u128) as u32;
    }

    pub fn stimulus_name(&self) -> &'static str {
        if self.spot_is_left {
            "spot_left"
        } else {
            "spot_right"
        }
    }

    pub fn correct_action(&self) -> &'static str {
        let normal = if self.spot_is_left { "left" } else { "right" };
        if !self.reversal_active {
            normal
        } else if normal == "left" {
            "right"
        } else {
            "left"
        }
    }

    pub fn score_action(&mut self, action: &str) -> Option<(f32, bool)> {
        if self.response_made {
            return None;
        }

        let is_correct = action == self.correct_action();
        let reward = if is_correct { 1.0 } else { -1.0 };

        self.response_made = true;
        self.last_action = Some(action.to_string());
        self.stats.record_trial(is_correct);
        self.update_reversal();

        Some((reward, true))
    }
}

impl Default for SpotReversalGame {
    fn default() -> Self {
        // Keep consistent with the daemon's typical default.
        Self::new(200)
    }
}
