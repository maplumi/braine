use crate::stats::GameStats;
use std::time::{Duration, Instant};

// ─────────────────────────────────────────────────────────────────────────
// Bandit game: no stimulus, just left/right arms with fixed reward schedule.
// ─────────────────────────────────────────────────────────────────────────

#[derive(Debug)]
pub struct BanditGame {
    pub trial_frame: u32,
    pub response_made: bool,
    pub last_action: Option<String>,
    pub stats: GameStats,

    pub prob_left: f32,
    pub prob_right: f32,

    rng_seed: u64,
    trial_started_at: Instant,
}

impl BanditGame {
    pub fn new() -> Self {
        let now = Instant::now();
        let mut g = Self {
            trial_frame: 0,
            response_made: false,
            last_action: None,
            stats: GameStats::new(),
            // Simple asymmetric schedule.
            prob_left: 0.8,
            prob_right: 0.2,
            rng_seed: 0xB4A7_1D2Bu64,
            trial_started_at: now,
        };
        g.new_trial();
        g
    }

    fn new_trial(&mut self) {
        self.trial_frame = 0;
        self.response_made = false;
        self.last_action = None;
        self.trial_started_at = Instant::now();
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
        // A constant context sensor to attach meaning/credit assignment.
        "bandit"
    }

    pub fn best_action(&self) -> &'static str {
        if self.prob_left >= self.prob_right {
            "left"
        } else {
            "right"
        }
    }

    fn rng_next_u32(&mut self) -> u32 {
        self.rng_seed = self
            .rng_seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1);
        (self.rng_seed >> 11) as u32
    }

    fn rng_next_f32(&mut self) -> f32 {
        let u = self.rng_next_u32();
        let mantissa = u >> 8; // 24 bits
        (mantissa as f32) / ((1u32 << 24) as f32)
    }

    /// Score exactly one response per trial.
    /// Returns `Some((reward, completed))` if this action ended the trial.
    pub fn score_action(&mut self, action: &str) -> Option<(f32, bool)> {
        if self.response_made {
            return None;
        }

        let is_correct = action == self.best_action();

        let p = if action == "left" {
            self.prob_left
        } else if action == "right" {
            self.prob_right
        } else {
            0.0
        };

        // Binary reward with symmetric +/- signal.
        let rewarded = self.rng_next_f32() < p;
        let reward = if rewarded { 1.0 } else { -1.0 };

        self.response_made = true;
        self.last_action = Some(action.to_string());
        self.stats.record_trial(is_correct);

        Some((reward, true))
    }
}

impl Default for BanditGame {
    fn default() -> Self {
        Self::new()
    }
}
