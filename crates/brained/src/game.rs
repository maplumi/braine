//! Spot game implementations for daemon

use braine::substrate::{Brain, Stimulus};
use std::time::{Duration, Instant};

#[derive(Debug, Clone)]
pub struct GameStats {
    pub correct: u32,
    pub incorrect: u32,
    pub trials: u32,
    pub recent: Vec<bool>,
    pub learning_at_trial: Option<u32>,
    pub learned_at_trial: Option<u32>,
    pub mastered_at_trial: Option<u32>,
}

impl GameStats {
    pub fn new() -> Self {
        Self {
            correct: 0,
            incorrect: 0,
            trials: 0,
            recent: Vec::with_capacity(200),
            learning_at_trial: None,
            learned_at_trial: None,
            mastered_at_trial: None,
        }
    }

    fn update_milestones(&mut self) {
        // Keep milestone definitions consistent with the UI labels.
        // Gate on a minimum number of trials to avoid “instant” mastery on tiny samples.
        if self.trials < 20 {
            return;
        }

        let r = self.last_100_rate();
        if self.learning_at_trial.is_none() && r >= 0.70 {
            self.learning_at_trial = Some(self.trials);
        }
        if self.learned_at_trial.is_none() && r >= 0.85 {
            self.learned_at_trial = Some(self.trials);
        }
        if self.mastered_at_trial.is_none() && r >= 0.95 {
            self.mastered_at_trial = Some(self.trials);
        }
    }

    fn record_trial(&mut self, is_correct: bool) {
        if is_correct {
            self.correct += 1;
        } else {
            self.incorrect += 1;
        }

        self.recent.push(is_correct);
        if self.recent.len() > 200 {
            self.recent.remove(0);
        }

        self.trials += 1;
        self.update_milestones();
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

// ─────────────────────────────────────────────────────────────────────────
// SpotXY: 2D position (population-coded) with a 2-action rule on sign(x).
// Includes an optional eval/holdout mode that samples from a held-out x band.
// ─────────────────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub enum SpotXYMode {
    /// Binary left/right classification from sign(x).
    BinaryX,
    /// N×N grid classification over (x,y).
    Grid { n: u32 },
}

#[derive(Debug)]
pub struct SpotXYGame {
    pub pos_x: f32,
    pub pos_y: f32,

    pub trial_frame: u32,
    pub response_made: bool,
    pub last_action: Option<String>,
    pub stats: GameStats,

    pub eval_mode: bool,

    pub mode: SpotXYMode,

    action_names: Vec<String>,
    correct_action: String,

    k: usize,
    sigma: f32,
    centers: Vec<f32>,
    x_names: Vec<String>,
    y_names: Vec<String>,
    x_act: Vec<f32>,
    y_act: Vec<f32>,
    stimulus_key: String,

    holdout_min_abs_x: f32,
    holdout_max_abs_x: f32,

    rng_seed: u64,
    trial_started_at: Instant,
}

// ─────────────────────────────────────────────────────────────────────────
// Pong: minimal closed-loop paddle tracking with 3 actions (up/down/stay).
// Discrete-time update: one action per trial step.
// Sensors are discrete bins (one-hot) for ball_x/ball_y/paddle_y plus velocity
// direction bits.
// ─────────────────────────────────────────────────────────────────────────

#[derive(Debug)]
pub struct PongGame {
    pub ball_x: f32,
    pub ball_y: f32,
    pub ball_vx: f32,
    pub ball_vy: f32,
    pub paddle_y: f32,

    pub paddle_speed: f32,
    pub paddle_half_height: f32,
    pub ball_speed: f32,

    pub trial_frame: u32,
    pub response_made: bool,
    pub last_action: Option<String>,
    pub stats: GameStats,

    action_names: Vec<String>,
    ball_x_names: Vec<String>,
    ball_y_names: Vec<String>,
    paddle_y_names: Vec<String>,
    stimulus_key: String,
    rng_seed: u64,
    trial_started_at: Instant,

    last_step_at: Instant,
    respawn_until: Option<Instant>,
    pending_event_reward: f32,
}

impl PongGame {
    pub fn new() -> Self {
        let now = Instant::now();

        const BINS: u32 = 8;
        let mut ball_x_names = Vec::with_capacity(BINS as usize);
        let mut ball_y_names = Vec::with_capacity(BINS as usize);
        let mut paddle_y_names = Vec::with_capacity(BINS as usize);
        for i in 0..BINS {
            ball_x_names.push(format!("pong_ball_x_{i:02}"));
            ball_y_names.push(format!("pong_ball_y_{i:02}"));
            paddle_y_names.push(format!("pong_paddle_y_{i:02}"));
        }

        let mut g = Self {
            ball_x: 0.0,
            ball_y: 0.0,
            ball_vx: 0.6,
            ball_vy: 0.35,
            paddle_y: 0.0,

            paddle_speed: 1.3,
            paddle_half_height: 0.15,
            ball_speed: 1.0,
            trial_frame: 0,
            response_made: false,
            last_action: None,
            stats: GameStats::new(),
            action_names: vec!["up".to_string(), "down".to_string(), "stay".to_string()],
            ball_x_names,
            ball_y_names,
            paddle_y_names,
            stimulus_key: String::new(),
            rng_seed: 0xB0A7_F00Du64,
            trial_started_at: now,

            last_step_at: now,
            respawn_until: None,
            pending_event_reward: 0.0,
        };
        g.reset_point();
        g
    }

    pub fn ball_visible(&self) -> bool {
        match self.respawn_until {
            Some(t) => Instant::now() >= t,
            None => true,
        }
    }

    pub fn set_param(&mut self, key: &str, value: f32) -> Result<(), String> {
        match key {
            "paddle_speed" => {
                self.paddle_speed = value.clamp(0.1, 5.0);
                Ok(())
            }
            "paddle_half_height" => {
                self.paddle_half_height = value.clamp(0.05, 0.9);
                Ok(())
            }
            "ball_speed" => {
                self.ball_speed = value.clamp(0.1, 3.0);
                Ok(())
            }
            _ => Err(format!(
                "Unknown Pong param '{key}'. Use paddle_speed|paddle_half_height|ball_speed"
            )),
        }
    }

    pub fn stimulus_name(&self) -> &'static str {
        // Base context name.
        "pong"
    }

    pub fn stimulus_key(&self) -> &str {
        &self.stimulus_key
    }

    pub fn allowed_actions(&self) -> &[String] {
        &self.action_names
    }

    pub fn correct_action(&self) -> &'static str {
        // Simple shaping target: move toward the ball's y position.
        // This is not the only viable policy, but it gives the agent a dense
        // supervisory signal while still requiring closed-loop control.
        let dy = self.ball_y - self.paddle_y;
        if dy > 0.12 {
            "up"
        } else if dy < -0.12 {
            "down"
        } else {
            "stay"
        }
    }

    pub fn update_timing(&mut self, trial_period_ms: u32) {
        let trial_period_ms = trial_period_ms.clamp(10, 60_000);
        let trial_period = Duration::from_millis(trial_period_ms as u64);

        let now = Instant::now();

        // Continuous physics step (independent of action cadence).
        let dt = now
            .duration_since(self.last_step_at)
            .as_secs_f32()
            .clamp(0.0, 0.05);
        self.last_step_at = now;

        // Ball can be briefly hidden after a miss to signal a new serve.
        if let Some(t) = self.respawn_until {
            if now < t {
                // Keep paddle centered and don't advance the ball while hidden.
            } else {
                self.respawn_until = None;
            }
        }

        if self.respawn_until.is_none() && dt > 0.0 {
            self.step_physics(dt);
        }

        let elapsed = now.duration_since(self.trial_started_at);
        if elapsed >= trial_period {
            // Allow exactly one action per timestep.
            self.response_made = false;
            self.last_action = None;
            self.trial_started_at = now;
        }

        let elapsed = now.duration_since(self.trial_started_at);
        self.trial_frame = elapsed.as_millis().min(u32::MAX as u128) as u32;
    }

    fn step_physics(&mut self, dt: f32) {
        // Advance ball.
        self.ball_x += self.ball_vx * self.ball_speed * dt;
        self.ball_y += self.ball_vy * self.ball_speed * dt;

        // Bounce on top/bottom.
        if self.ball_y > 1.0 {
            self.ball_y = 2.0 - self.ball_y;
            self.ball_vy = -self.ball_vy;
        } else if self.ball_y < -1.0 {
            self.ball_y = -2.0 - self.ball_y;
            self.ball_vy = -self.ball_vy;
        }

        // Right wall bounce so the ball keeps returning.
        if self.ball_x >= 1.0 {
            self.ball_x = 2.0 - self.ball_x;
            self.ball_vx = -self.ball_vx.abs();
        }

        // Left paddle collision / miss.
        if self.ball_x <= 0.0 {
            let hit = (self.ball_y - self.paddle_y).abs() <= self.paddle_half_height;
            if hit {
                self.pending_event_reward += 1.0;
                self.ball_x = 0.0;
                self.ball_vx = self.ball_vx.abs();
            } else {
                self.pending_event_reward -= 1.0;

                // Hide ball briefly, then respawn a new serve.
                self.respawn_until = Some(Instant::now() + Duration::from_millis(180));
                self.reset_point();
            }
        }

        self.refresh_stimulus_key();
    }

    pub fn apply_stimuli(&self, brain: &mut Brain) {
        // Discrete bins (one-hot). Keep this intentionally small; later we can
        // migrate to graded/population coding without changing the action space.
        let bins = self.ball_x_names.len().max(2) as u32;
        let bx = pong_bin_01(self.ball_x, bins);
        let by = pong_bin_signed(self.ball_y, bins);
        let py = pong_bin_signed(self.paddle_y, bins);

        brain.apply_stimulus(Stimulus::new(self.ball_x_names[bx as usize].as_str(), 1.0));
        brain.apply_stimulus(Stimulus::new(self.ball_y_names[by as usize].as_str(), 1.0));
        brain.apply_stimulus(Stimulus::new(
            self.paddle_y_names[py as usize].as_str(),
            1.0,
        ));

        let vx_name = if self.ball_vx >= 0.0 {
            "pong_vx_pos"
        } else {
            "pong_vx_neg"
        };
        let vy_name = if self.ball_vy >= 0.0 {
            "pong_vy_pos"
        } else {
            "pong_vy_neg"
        };

        brain.apply_stimulus(Stimulus::new(vx_name, 1.0));
        brain.apply_stimulus(Stimulus::new(vy_name, 1.0));
    }

    pub fn score_action(&mut self, action: &str, trial_period_ms: u32) -> Option<(f32, bool)> {
        if self.response_made {
            return None;
        }

        let is_correct = action == self.correct_action();
        let mut reward: f32 = if is_correct { 0.05f32 } else { -0.05f32 };

        // Add any physics event reward (hit/miss) since the last action.
        reward += self.pending_event_reward;
        self.pending_event_reward = 0.0;

        // Apply action: move paddle.
        let dt = (trial_period_ms.clamp(10, 60_000) as f32) / 1000.0;
        match action {
            "up" => self.paddle_y += self.paddle_speed * dt,
            "down" => self.paddle_y -= self.paddle_speed * dt,
            _ => {}
        }
        self.paddle_y = self.paddle_y.clamp(-1.0, 1.0);

        self.refresh_stimulus_key();

        self.response_made = true;
        self.last_action = Some(action.to_string());
        self.stats.record_trial(is_correct);

        // Each action is treated as one “trial” step.
        Some((reward.clamp(-1.0, 1.0), true))
    }

    fn reset_point(&mut self) {
        self.ball_x = 0.5;
        self.ball_y = self.sample_uniform(-0.6, 0.6);
        self.ball_vx = 0.75;
        self.ball_vy = self.sample_uniform(-0.55, 0.55);
        if self.ball_vy.abs() < 0.15 {
            self.ball_vy = if self.ball_vy >= 0.0 { 0.15 } else { -0.15 };
        }
        self.paddle_y = 0.0;
        self.refresh_stimulus_key();
    }

    fn refresh_stimulus_key(&mut self) {
        let bins = self.ball_x_names.len().max(2) as u32;
        let bx = pong_bin_01(self.ball_x, bins);
        let by = pong_bin_signed(self.ball_y, bins);
        let py = pong_bin_signed(self.paddle_y, bins);
        let vx = if self.ball_vx >= 0.0 { "p" } else { "n" };
        let vy = if self.ball_vy >= 0.0 { "p" } else { "n" };
        self.stimulus_key = format!("pong_b{bins:02}_bx{bx:02}_by{by:02}_py{py:02}_vx{vx}_vy{vy}");
    }

    fn sample_uniform(&mut self, lo: f32, hi: f32) -> f32 {
        let u = self.rng_next_f32();
        lo + (hi - lo) * u
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
}

fn pong_bin_signed(v: f32, bins: u32) -> u32 {
    let bins = bins.max(2);
    let t = ((v.clamp(-1.0, 1.0) + 1.0) * 0.5).clamp(0.0, 0.999_999);
    let b = (t * bins as f32).floor() as u32;
    b.min(bins - 1)
}

fn pong_bin_01(v: f32, bins: u32) -> u32 {
    let bins = bins.max(2);
    let t = v.clamp(0.0, 0.999_999);
    let b = (t * bins as f32).floor() as u32;
    b.min(bins - 1)
}

impl SpotXYGame {
    pub fn new(k: usize) -> Self {
        let k = k.max(2);
        let denom = (k - 1) as f32;
        let centers: Vec<f32> = (0..k).map(|i| -1.0 + 2.0 * (i as f32) / denom).collect();
        let sigma = 2.0 / denom;

        let mut x_names: Vec<String> = Vec::with_capacity(k);
        let mut y_names: Vec<String> = Vec::with_capacity(k);
        for i in 0..k {
            x_names.push(format!("pos_x_{i:02}"));
            y_names.push(format!("pos_y_{i:02}"));
        }

        let now = Instant::now();
        let mut g = Self {
            pos_x: 0.0,
            pos_y: 0.0,
            trial_frame: 0,
            response_made: false,
            last_action: None,
            stats: GameStats::new(),
            eval_mode: false,
            mode: SpotXYMode::BinaryX,
            action_names: vec!["left".to_string(), "right".to_string()],
            correct_action: "left".to_string(),
            k,
            sigma,
            centers,
            x_names,
            y_names,
            x_act: vec![0.0; k],
            y_act: vec![0.0; k],
            stimulus_key: String::new(),
            // Held-out band used for evaluation/generalization checks.
            holdout_min_abs_x: 0.25,
            holdout_max_abs_x: 0.45,
            rng_seed: 0x5107_5129u64,
            trial_started_at: now,
        };
        g.new_trial();
        g
    }

    pub fn increase_grid(&mut self) {
        let next = match self.mode {
            SpotXYMode::BinaryX => SpotXYMode::Grid { n: 2 },
            SpotXYMode::Grid { n } => SpotXYMode::Grid { n: (n + 1).min(8) },
        };

        if std::mem::discriminant(&self.mode) == std::mem::discriminant(&next) {
            // Same variant; still might differ in n.
        }

        self.mode = next;
        self.refresh_actions();
        self.stats = GameStats::new();
        self.new_trial();
    }

    pub fn decrease_grid(&mut self) {
        let next = match self.mode {
            SpotXYMode::BinaryX => SpotXYMode::BinaryX,
            SpotXYMode::Grid { n } => {
                if n <= 2 {
                    SpotXYMode::BinaryX
                } else {
                    SpotXYMode::Grid { n: n - 1 }
                }
            }
        };

        self.mode = next;
        self.refresh_actions();
        self.stats = GameStats::new();
        self.new_trial();
    }

    pub fn grid_n(&self) -> u32 {
        match self.mode {
            SpotXYMode::BinaryX => 0,
            SpotXYMode::Grid { n } => n,
        }
    }

    pub fn mode_name(&self) -> &'static str {
        match self.mode {
            SpotXYMode::BinaryX => "binary_x",
            SpotXYMode::Grid { .. } => "grid",
        }
    }

    pub fn allowed_actions(&self) -> &[String] {
        &self.action_names
    }

    pub fn set_eval_mode(&mut self, eval: bool) {
        if self.eval_mode == eval {
            return;
        }
        self.eval_mode = eval;
        // Treat switching as a fresh run so metrics are interpretable.
        self.stats = GameStats::new();
        self.new_trial();
    }

    pub fn stimulus_name(&self) -> &'static str {
        // Base name (not directly used for meaning conditioning).
        "spotxy"
    }

    pub fn stimulus_key(&self) -> &str {
        &self.stimulus_key
    }

    pub fn correct_action(&self) -> &str {
        &self.correct_action
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

    pub fn apply_stimuli(&self, brain: &mut Brain) {
        for i in 0..self.k {
            brain.apply_stimulus(Stimulus::new(self.x_names[i].as_str(), self.x_act[i]));
            brain.apply_stimulus(Stimulus::new(self.y_names[i].as_str(), self.y_act[i]));
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

        Some((reward, true))
    }

    fn new_trial(&mut self) {
        self.trial_frame = 0;
        self.response_made = false;
        self.last_action = None;
        self.trial_started_at = Instant::now();

        self.pos_x = self.sample_x();
        self.pos_y = self.sample_uniform(-1.0, 1.0);

        self.x_act = axis_activations(self.pos_x, &self.centers, self.sigma);
        self.y_act = axis_activations(self.pos_y, &self.centers, self.sigma);

        match self.mode {
            SpotXYMode::BinaryX => {
                let xbin = argmax(&self.x_act);
                self.stimulus_key = format!("spotxy_xbin_{xbin:02}");
                self.correct_action = if self.pos_x < 0.0 {
                    "left".to_string()
                } else {
                    "right".to_string()
                };
            }
            SpotXYMode::Grid { n } => {
                let n = n.clamp(2, 8);

                let ix = grid_bin(self.pos_x, n);
                let iy = grid_bin(self.pos_y, n);
                self.stimulus_key = format!("spotxy_bin_{n:02}_{ix:02}_{iy:02}");
                self.correct_action = format!("spotxy_cell_{n:02}_{ix:02}_{iy:02}");
            }
        }
    }

    fn refresh_actions(&mut self) {
        self.action_names.clear();
        match self.mode {
            SpotXYMode::BinaryX => {
                self.action_names.push("left".to_string());
                self.action_names.push("right".to_string());
            }
            SpotXYMode::Grid { n } => {
                let n = n.clamp(2, 8);
                let cap = (n as usize) * (n as usize);
                self.action_names.reserve(cap);
                for ix in 0..n {
                    for iy in 0..n {
                        self.action_names
                            .push(format!("spotxy_cell_{n:02}_{ix:02}_{iy:02}"));
                    }
                }
            }
        }
    }

    fn sample_uniform(&mut self, lo: f32, hi: f32) -> f32 {
        let u = self.rng_next_f32();
        lo + (hi - lo) * u
    }

    fn sample_x(&mut self) -> f32 {
        // In eval mode we sample only within the holdout band.
        if self.eval_mode {
            let sign = if (self.rng_next_u32() & 1) == 0 {
                -1.0
            } else {
                1.0
            };
            let u = self.rng_next_f32();
            let mag =
                self.holdout_min_abs_x + (self.holdout_max_abs_x - self.holdout_min_abs_x) * u;
            return (sign * mag).clamp(-1.0, 1.0);
        }

        // Training mode: sample from [-1,1] excluding the holdout band.
        loop {
            let x = self.sample_uniform(-1.0, 1.0);
            let ax = x.abs();
            if ax < self.holdout_min_abs_x || ax > self.holdout_max_abs_x {
                return x;
            }
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
}

fn axis_activations(v: f32, centers: &[f32], sigma: f32) -> Vec<f32> {
    let v = v.clamp(-1.0, 1.0);
    let inv_2s2 = 1.0 / (2.0 * sigma * sigma + 1e-9);

    let mut out: Vec<f32> = Vec::with_capacity(centers.len());
    let mut sum = 0.0f32;
    for &c in centers {
        let d = v - c;
        let a = (-d * d * inv_2s2).exp();
        out.push(a);
        sum += a;
    }

    // Per-axis normalization to keep stimulus energy stable.
    if sum > 1e-9 {
        let inv = 1.0 / sum;
        for a in &mut out {
            *a = (*a * inv).clamp(0.0, 1.0);
        }
    }

    out
}

fn argmax(v: &[f32]) -> usize {
    let mut best_i = 0usize;
    let mut best = f32::NEG_INFINITY;
    for (i, &x) in v.iter().enumerate() {
        if x > best {
            best = x;
            best_i = i;
        }
    }
    best_i
}

fn grid_bin(v: f32, n: u32) -> u32 {
    let n = n.max(2);
    // Map [-1,1] to [0,1], then bucket into n bins.
    let t = ((v.clamp(-1.0, 1.0) + 1.0) * 0.5).clamp(0.0, 0.999_999);
    let b = (t * n as f32).floor() as u32;
    b.min(n - 1)
}
