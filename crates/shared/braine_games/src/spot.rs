use crate::stats::GameStats;
use crate::time::{Duration, Instant};

#[derive(Debug)]
pub struct SpotGame {
    pub spot_is_left: bool,
    pub trial_frame: u32,
    pub response_made: bool,
    pub last_action: Option<String>,
    pub stats: GameStats,
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
            last_action: None,
            stats: GameStats::new(),
            rng_seed: 12345,
            trial_started_at: now,
        };
        game.new_trial();
        game
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

    pub fn correct_action(&self) -> &'static str {
        if self.spot_is_left {
            "left"
        } else {
            "right"
        }
    }

    /// Score exactly one response per trial.
    /// Returns `Some((reward, completed))` if this action ended the trial.
    pub fn score_action(&mut self, action: &str) -> Option<(f32, bool)> {
        if self.response_made {
            return None;
        }

        let is_correct = action == self.correct_action();

        // Reward: +1 correct, -1 incorrect to push policy off chance performance.
        let reward = if is_correct { 1.0 } else { -1.0 };

        self.response_made = true;
        self.last_action = Some(action.to_string());
        self.stats.record_trial(is_correct);

        Some((reward, true))
    }
}

impl Default for SpotGame {
    fn default() -> Self {
        Self::new()
    }
}
