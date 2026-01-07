//! Spot game implementation for daemon

use std::time::{Duration, Instant};

#[derive(Debug)]
pub struct SpotGame {
    pub spot_is_left: bool,
    pub trial_frame: u32,
    pub response_made: bool,
    pub correct: u32,
    pub incorrect: u32,
    pub trials: u32,
    pub recent: Vec<bool>,
    rng_seed: u64,
    trial_started_at: Instant,
}

impl SpotGame {
    pub fn new() -> Self {
        let now = Instant::now();
        let mut game = Self {
            spot_is_left: true,
            trial_frame: 0,
            response_made: false,
            correct: 0,
            incorrect: 0,
            trials: 0,
            recent: Vec::with_capacity(200),
            rng_seed: 12345,
            trial_started_at: now,
        };
        game.new_trial();
        game
    }

    fn new_trial(&mut self) {
        self.trial_frame = 0;
        self.response_made = false;
        self.trial_started_at = Instant::now();

        // Random side
        self.rng_seed = self
            .rng_seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1);
        let u = (self.rng_seed >> 11) as u32;
        self.spot_is_left = (u & 1) == 0;
    }

    pub fn update_timing(&mut self, trial_period_ms: u32) {
        let trial_period_ms = trial_period_ms.clamp(10, 60_000);
        let trial_period = Duration::from_millis(trial_period_ms as u64);

        let now = Instant::now();
        let elapsed = now.duration_since(self.trial_started_at);

        // Advance trial on a wall-clock schedule (decoupled from daemon FPS).
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

    /// Score exactly one response per trial.
    /// Returns `Some((reward, completed))` if this action ended the trial.
    pub fn score_action(&mut self, action: &str) -> Option<(f32, bool)> {
        if self.response_made {
            return None;
        }

        let correct_action = if self.spot_is_left { "left" } else { "right" };
        let is_correct = action == correct_action;

        // Reward: +1 correct, -1 incorrect to push policy off chance performance.
        let reward = if is_correct { 1.0 } else { -1.0 };

        if is_correct {
            self.correct += 1;
        } else {
            self.incorrect += 1;
        }

        self.recent.push(is_correct);
        if self.recent.len() > 200 {
            self.recent.remove(0);
        }

        self.response_made = true;
        self.trials += 1;

        Some((reward, true))
    }

    pub fn accuracy(&self) -> f32 {
        let total = self.correct + self.incorrect;
        if total == 0 {
            0.5
        } else {
            self.correct as f32 / total as f32
        }
    }

    pub fn recent_rate(&self) -> f32 {
        if self.recent.is_empty() {
            return 0.5;
        }
        let correct_count = self.recent.iter().filter(|&&x| x).count();
        correct_count as f32 / self.recent.len() as f32
    }

    pub fn last_100_rate(&self) -> f32 {
        if self.recent.len() < 10 {
            return self.recent_rate();
        }
        let start = if self.recent.len() > 100 {
            self.recent.len() - 100
        } else {
            0
        };
        let slice = &self.recent[start..];
        let correct_count = slice.iter().filter(|&&x| x).count();
        correct_count as f32 / slice.len() as f32
    }
}
