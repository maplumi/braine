//! Game implementations for the daemon.
//!
//! Most deterministic game logic lives in `crates/shared/braine_games`.
//! This module keeps daemon-only glue (e.g. `Brain` stimulus application for Pong).

use braine::substrate::{Brain, Stimulus};
use braine_games::pong::{PongAction, PongEvent, PongSim};
use std::time::{Duration, Instant};

pub use braine_games::bandit::BanditGame;
pub use braine_games::maze::MazeGame;
pub use braine_games::replay::{ReplayDataset, ReplayGame};
pub use braine_games::spot::SpotGame;
pub use braine_games::spot_reversal::SpotReversalGame;
pub use braine_games::spot_xy::SpotXYGame;
pub use braine_games::stats::GameStats;
pub use braine_games::text_next_token::TextNextTokenGame;

// ─────────────────────────────────────────────────────────────────────────
// Pong: minimal closed-loop paddle tracking with 3 actions (up/down/stay).
// Discrete-time update: one action per trial step.
// Sensors are discrete bins (one-hot) for ball_x/ball_y/paddle_y plus velocity
// direction bits.
// ─────────────────────────────────────────────────────────────────────────

#[derive(Debug)]
pub struct PongGame {
    pub sim: PongSim,

    pub trial_frame: u32,
    pub response_made: bool,
    pub last_action: Option<String>,
    pub stats: GameStats,

    action_names: Vec<String>,
    ball_x_names: Vec<String>,
    ball_y_names: Vec<String>,
    ball2_x_names: Vec<String>,
    ball2_y_names: Vec<String>,
    paddle_y_names: Vec<String>,
    target_y_names: Vec<String>,
    stimulus_key: String,
    trial_started_at: Instant,

    hits: u32,
    misses: u32,
    last_event: PongEvent,
    event_flash_ticks: u8,
    pending_credit: Option<PongCredit>,

    last_step_at: Instant,
}

#[derive(Debug, Clone)]
struct PongCredit {
    reward: f32,
    action: String,
    stimulus_key: String,
    event: PongEvent,
}

impl PongGame {
    pub fn new() -> Self {
        let now = Instant::now();

        const BINS: u32 = 8;
        let mut ball_x_names = Vec::with_capacity(BINS as usize);
        let mut ball_y_names = Vec::with_capacity(BINS as usize);
        let mut ball2_x_names = Vec::with_capacity(BINS as usize);
        let mut ball2_y_names = Vec::with_capacity(BINS as usize);
        let mut paddle_y_names = Vec::with_capacity(BINS as usize);
        let mut target_y_names = Vec::with_capacity(BINS as usize);
        for i in 0..BINS {
            ball_x_names.push(format!("pong_ball_x_{i:02}"));
            ball_y_names.push(format!("pong_ball_y_{i:02}"));
            ball2_x_names.push(format!("pong_ball2_x_{i:02}"));
            ball2_y_names.push(format!("pong_ball2_y_{i:02}"));
            paddle_y_names.push(format!("pong_paddle_y_{i:02}"));
            target_y_names.push(format!("pong_target_y_{i:02}"));
        }

        let mut g = Self {
            sim: PongSim::new(0xB0A7_F00Du64),
            trial_frame: 0,
            response_made: false,
            last_action: None,
            stats: GameStats::new(),
            action_names: vec!["up".to_string(), "down".to_string(), "stay".to_string()],
            ball_x_names,
            ball_y_names,
            ball2_x_names,
            ball2_y_names,
            paddle_y_names,
            target_y_names,
            stimulus_key: String::new(),
            trial_started_at: now,

            hits: 0,
            misses: 0,
            last_event: PongEvent::None,
            event_flash_ticks: 0,
            pending_credit: None,

            last_step_at: now,
        };
        g.sim.reset_point();
        g.refresh_stimulus_key();
        g
    }

    pub fn hits(&self) -> u32 {
        self.hits
    }

    pub fn misses(&self) -> u32 {
        self.misses
    }

    pub fn take_pending_credit(&mut self) -> Option<(f32, String, String, PongEvent)> {
        self.pending_credit
            .take()
            .map(|c| (c.reward, c.action, c.stimulus_key, c.event))
    }

    pub fn ball_visible(&self) -> bool {
        self.sim.ball_visible()
    }

    pub fn ball2_visible(&self) -> bool {
        self.sim.ball2_visible()
    }

    pub fn ball2_enabled(&self) -> bool {
        self.sim.distractor_enabled()
    }

    pub fn set_param(&mut self, key: &str, value: f32) -> Result<(), String> {
        match key {
            "paddle_speed" => {
                self.sim.params.paddle_speed = value.clamp(0.1, 5.0);
                Ok(())
            }
            "paddle_half_height" => {
                self.sim.params.paddle_half_height = value.clamp(0.05, 0.9);
                Ok(())
            }
            "ball_speed" => {
                self.sim.params.ball_speed = value.clamp(0.1, 3.0);
                Ok(())
            }
            "paddle_bounce_y" => {
                self.sim.params.paddle_bounce_y = value.clamp(0.0, 2.5);
                Ok(())
            }
            "distractor_enabled" => {
                self.sim.params.distractor_enabled = value >= 0.5;
                Ok(())
            }
            "distractor_speed_scale" => {
                self.sim.params.distractor_speed_scale = value.clamp(0.1, 2.5);
                Ok(())
            }
            _ => Err(format!(
                "Unknown Pong param '{key}'. Use paddle_speed|paddle_half_height|ball_speed|paddle_bounce_y|distractor_enabled|distractor_speed_scale"
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
        // Shaping target: move toward predicted intercept when the ball is approaching.
        // (If the ball is moving away/hidden, avoid thrashing.)
        let target_y = self
            .sim
            .predict_primary_y_at_paddle()
            .unwrap_or(self.sim.state.paddle_y);
        let dy = target_y - self.sim.state.paddle_y;
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

        // Continuous physics step (independent of action cadence).
        let ev = self.sim.update(dt);
        if ev != PongEvent::None {
            self.last_event = ev;
            self.event_flash_ticks = 3;

            match ev {
                PongEvent::Hit => self.hits = self.hits.saturating_add(1),
                PongEvent::Miss => self.misses = self.misses.saturating_add(1),
                PongEvent::None => {}
            }

            // If we have a held action (paddle position held until next trial),
            // credit that action immediately when the physics event occurs.
            if let Some(a) = self.last_action.clone() {
                let reward = self.sim.take_pending_event_reward().clamp(-1.0, 1.0);
                if reward.abs() > 0.0 {
                    self.pending_credit = Some(PongCredit {
                        reward,
                        action: a,
                        stimulus_key: self.stimulus_key.clone(),
                        event: ev,
                    });
                }
            } else {
                // Still clear the pending reward so it doesn't leak into the next scored action.
                let _ = self.sim.take_pending_event_reward();
            }
        }
        self.refresh_stimulus_key();

        let elapsed = now.duration_since(self.trial_started_at);
        if elapsed >= trial_period {
            // Allow exactly one action per timestep.
            self.response_made = false;
            self.last_action = None;
            self.trial_started_at = now;
        }

        let elapsed = now.duration_since(self.trial_started_at);
        self.trial_frame = elapsed.as_millis().min(u32::MAX as u128) as u32;

        if self.event_flash_ticks > 0 {
            self.event_flash_ticks -= 1;
        }
    }

    pub fn apply_stimuli(&self, brain: &mut Brain) {
        // Discrete bins (one-hot). Keep this intentionally small; later we can
        // migrate to graded/population coding without changing the action space.
        let bins = self.ball_x_names.len().max(2) as u32;
        let bx = PongSim::bin_01(self.sim.state.ball_x, bins);
        let by = PongSim::bin_signed(self.sim.state.ball_y, bins);
        let py = PongSim::bin_signed(self.sim.state.paddle_y, bins);

        brain.apply_stimulus_inference(Stimulus::new(self.ball_x_names[bx as usize].as_str(), 1.0));
        brain.apply_stimulus_inference(Stimulus::new(self.ball_y_names[by as usize].as_str(), 1.0));
        brain.apply_stimulus_inference(Stimulus::new(
            self.paddle_y_names[py as usize].as_str(),
            1.0,
        ));

        // Trajectory feature: predicted intercept y at paddle (when approaching).
        if let Some(y) = self.sim.predict_primary_y_at_paddle() {
            let ty = PongSim::bin_signed(y, bins);
            brain.apply_stimulus_inference(Stimulus::new(
                self.target_y_names[ty as usize].as_str(),
                1.0,
            ));
        } else {
            brain.apply_stimulus_inference(Stimulus::new("pong_target_na", 1.0));
        }

        // One-shot event flash (helps the brain "notice" hits/misses).
        if self.event_flash_ticks > 0 {
            match self.last_event {
                PongEvent::Hit => {
                    brain.apply_stimulus_inference(Stimulus::new("pong_evt_hit", 1.0))
                }
                PongEvent::Miss => {
                    brain.apply_stimulus_inference(Stimulus::new("pong_evt_miss", 1.0))
                }
                PongEvent::None => {}
            }
        }

        if self.sim.ball_visible() {
            brain.apply_stimulus_inference(Stimulus::new("pong_ball_visible", 1.0));
        } else {
            brain.apply_stimulus_inference(Stimulus::new("pong_ball_hidden", 1.0));
        }

        let vx_name = if self.sim.state.ball_vx >= 0.0 {
            "pong_vx_pos"
        } else {
            "pong_vx_neg"
        };
        let vy_name = if self.sim.state.ball_vy >= 0.0 {
            "pong_vy_pos"
        } else {
            "pong_vy_neg"
        };

        brain.apply_stimulus_inference(Stimulus::new(vx_name, 1.0));
        brain.apply_stimulus_inference(Stimulus::new(vy_name, 1.0));

        if self.sim.distractor_enabled() {
            let b2x = PongSim::bin_01(self.sim.state.ball2_x, bins);
            let b2y = PongSim::bin_signed(self.sim.state.ball2_y, bins);
            brain.apply_stimulus_inference(Stimulus::new(
                self.ball2_x_names[b2x as usize].as_str(),
                1.0,
            ));
            brain.apply_stimulus_inference(Stimulus::new(
                self.ball2_y_names[b2y as usize].as_str(),
                1.0,
            ));

            if self.sim.ball2_visible() {
                brain.apply_stimulus_inference(Stimulus::new("pong_ball2_visible", 1.0));
            } else {
                brain.apply_stimulus_inference(Stimulus::new("pong_ball2_hidden", 1.0));
            }

            let b2_vx_name = if self.sim.state.ball2_vx >= 0.0 {
                "pong_ball2_vx_pos"
            } else {
                "pong_ball2_vx_neg"
            };
            let b2_vy_name = if self.sim.state.ball2_vy >= 0.0 {
                "pong_ball2_vy_pos"
            } else {
                "pong_ball2_vy_neg"
            };
            brain.apply_stimulus_inference(Stimulus::new(b2_vx_name, 1.0));
            brain.apply_stimulus_inference(Stimulus::new(b2_vy_name, 1.0));
        }
    }

    pub fn score_action(&mut self, action: &str, trial_period_ms: u32) -> Option<(f32, bool)> {
        if self.response_made {
            return None;
        }

        let is_correct = action == self.correct_action();
        let mut reward: f32 = 0.0;

        // Tiny nudge reward just for "moving the right way" (kept small so physics + shaping dominate).
        reward += if is_correct { 0.002 } else { -0.002 };

        // Physics hit/miss rewards are applied immediately in `update_timing()` via
        // `pending_credit` (to avoid crediting the *next* action).

        // Dense shaping: encourage aligning paddle with the (rewarded) ball when it's approaching.
        // This stays within small bounds so hit/miss remains the primary learning signal.
        if self.sim.ball_visible() && self.sim.state.ball_vx < 0.0 {
            let proximity = (1.0 - self.sim.state.ball_x).clamp(0.0, 1.0);
            let hh = self.sim.params.paddle_half_height.max(1.0e-6);
            let err = (self.sim.state.ball_y - self.sim.state.paddle_y).abs();
            let norm = (err / (hh * 1.5)).clamp(0.0, 1.0);
            let aligned = 1.0 - norm; // 0..1
            reward += (aligned - 0.5) * 2.0 * (0.03 * proximity);
        }

        // Small movement penalty to reduce thrash.
        if action == "up" || action == "down" {
            reward -= 0.005;
        }

        // Apply action: move paddle.
        let dt = (trial_period_ms.clamp(10, 60_000) as f32) / 1000.0;
        let a = match action {
            "up" => PongAction::Up,
            "down" => PongAction::Down,
            _ => PongAction::Stay,
        };
        self.sim.apply_action(a, dt);

        self.refresh_stimulus_key();

        self.response_made = true;
        self.last_action = Some(action.to_string());
        self.stats.record_trial(is_correct);

        // Each action is treated as one “trial” step.
        Some((reward.clamp(-1.0, 1.0), true))
    }

    fn refresh_stimulus_key(&mut self) {
        let bins = self.ball_x_names.len().max(2) as u32;
        let bx = PongSim::bin_01(self.sim.state.ball_x, bins);
        let by = PongSim::bin_signed(self.sim.state.ball_y, bins);
        let py = PongSim::bin_signed(self.sim.state.paddle_y, bins);
        let vis = if self.sim.ball_visible() { "v" } else { "h" };
        let vx = if self.sim.state.ball_vx >= 0.0 {
            "p"
        } else {
            "n"
        };
        let vy = if self.sim.state.ball_vy >= 0.0 {
            "p"
        } else {
            "n"
        };

        if self.sim.distractor_enabled() {
            let b2x = PongSim::bin_01(self.sim.state.ball2_x, bins);
            let b2y = PongSim::bin_signed(self.sim.state.ball2_y, bins);
            let vis2 = if self.sim.ball2_visible() { "v" } else { "h" };
            let vx2 = if self.sim.state.ball2_vx >= 0.0 {
                "p"
            } else {
                "n"
            };
            let vy2 = if self.sim.state.ball2_vy >= 0.0 {
                "p"
            } else {
                "n"
            };
            self.stimulus_key = format!(
                "pong_b{bins:02}_vis{vis}_bx{bx:02}_by{by:02}_py{py:02}_vx{vx}_vy{vy}_b2vis{vis2}_b2x{b2x:02}_b2y{b2y:02}_b2vx{vx2}_b2vy{vy2}"
            );
        } else {
            self.stimulus_key =
                format!("pong_b{bins:02}_vis{vis}_bx{bx:02}_by{by:02}_py{py:02}_vx{vx}_vy{vy}");
        }
    }
}
