use braine::substrate::{Brain, Stimulus};
use braine_games::pong::{PongAction, PongSim};
use braine_games::stats::GameStats;
use core::time::Duration;
use web_time::Instant;

#[derive(Debug)]
pub struct PongWebGame {
    pub sim: PongSim,

    pub trial_frame: u32,
    pub response_made: bool,
    pub last_action: Option<String>,
    pub stats: GameStats,

    action_names: [String; 3],
    ball_x_names: [String; 8],
    ball_y_names: [String; 8],
    ball2_x_names: [String; 8],
    ball2_y_names: [String; 8],
    paddle_y_names: [String; 8],

    stimulus_key: String,
    trial_started_at: Instant,
    last_step_at: Instant,
    trial_period_ms: u32,
}

impl PongWebGame {
    pub fn new(seed: u64) -> Self {
        let now = Instant::now();

        let ball_x_names = std::array::from_fn(|i| format!("pong_ball_x_{i:02}"));
        let ball_y_names = std::array::from_fn(|i| format!("pong_ball_y_{i:02}"));
        let ball2_x_names = std::array::from_fn(|i| format!("pong_ball2_x_{i:02}"));
        let ball2_y_names = std::array::from_fn(|i| format!("pong_ball2_y_{i:02}"));
        let paddle_y_names = std::array::from_fn(|i| format!("pong_paddle_y_{i:02}"));

        let mut g = Self {
            sim: PongSim::new(seed),
            trial_frame: 0,
            response_made: false,
            last_action: None,
            stats: GameStats::new(),
            action_names: ["up".to_string(), "down".to_string(), "stay".to_string()],
            ball_x_names,
            ball_y_names,
            ball2_x_names,
            ball2_y_names,
            paddle_y_names,
            stimulus_key: String::new(),
            trial_started_at: now,
            last_step_at: now,
            trial_period_ms: 500,
        };

        g.sim.reset_point();
        g.refresh_stimulus_key();
        g
    }

    pub fn stimulus_name(&self) -> &'static str {
        "pong"
    }

    pub fn stimulus_key(&self) -> &str {
        &self.stimulus_key
    }

    pub fn allowed_actions(&self) -> &[String] {
        &self.action_names
    }

    pub fn correct_action(&self) -> &'static str {
        let dy = self.sim.state.ball_y - self.sim.state.paddle_y;
        if dy > 0.12 {
            "up"
        } else if dy < -0.12 {
            "down"
        } else {
            "stay"
        }
    }

    pub fn ball_visible(&self) -> bool {
        self.sim.ball_visible()
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

    pub fn update_timing(&mut self, trial_period_ms: u32) {
        let trial_period_ms = trial_period_ms.clamp(10, 60_000);
        self.trial_period_ms = trial_period_ms;

        let trial_period = Duration::from_millis(trial_period_ms as u64);
        let now = Instant::now();

        let dt = now
            .duration_since(self.last_step_at)
            .as_secs_f32()
            .clamp(0.0, 0.05);
        self.last_step_at = now;

        self.sim.update(dt);
        self.refresh_stimulus_key();

        let elapsed = now.duration_since(self.trial_started_at);
        if elapsed >= trial_period {
            self.response_made = false;
            self.last_action = None;
            self.trial_started_at = now;
        }

        let elapsed = now.duration_since(self.trial_started_at);
        self.trial_frame = elapsed.as_millis().min(u128::from(u32::MAX)) as u32;
    }

    pub fn apply_stimuli(&self, brain: &mut Brain) {
        let bins = 8u32;
        let bx = PongSim::bin_01(self.sim.state.ball_x, bins);
        let by = PongSim::bin_signed(self.sim.state.ball_y, bins);
        let py = PongSim::bin_signed(self.sim.state.paddle_y, bins);

        brain.apply_stimulus(Stimulus::new(self.ball_x_names[bx as usize].as_str(), 1.0));
        brain.apply_stimulus(Stimulus::new(self.ball_y_names[by as usize].as_str(), 1.0));
        brain.apply_stimulus(Stimulus::new(
            self.paddle_y_names[py as usize].as_str(),
            1.0,
        ));

        if self.sim.ball_visible() {
            brain.apply_stimulus(Stimulus::new("pong_ball_visible", 1.0));
        } else {
            brain.apply_stimulus(Stimulus::new("pong_ball_hidden", 1.0));
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
        brain.apply_stimulus(Stimulus::new(vx_name, 1.0));
        brain.apply_stimulus(Stimulus::new(vy_name, 1.0));

        if self.sim.distractor_enabled() {
            let b2x = PongSim::bin_01(self.sim.state.ball2_x, bins);
            let b2y = PongSim::bin_signed(self.sim.state.ball2_y, bins);
            brain.apply_stimulus(Stimulus::new(
                self.ball2_x_names[b2x as usize].as_str(),
                1.0,
            ));
            brain.apply_stimulus(Stimulus::new(
                self.ball2_y_names[b2y as usize].as_str(),
                1.0,
            ));

            if self.sim.ball2_visible() {
                brain.apply_stimulus(Stimulus::new("pong_ball2_visible", 1.0));
            } else {
                brain.apply_stimulus(Stimulus::new("pong_ball2_hidden", 1.0));
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
            brain.apply_stimulus(Stimulus::new(b2_vx_name, 1.0));
            brain.apply_stimulus(Stimulus::new(b2_vy_name, 1.0));
        }
    }

    pub fn score_action(&mut self, action: &str) -> Option<(f32, bool)> {
        if self.response_made {
            return None;
        }

        let is_correct = action == self.correct_action();
        let mut reward: f32 = 0.0;
        reward += if is_correct { 0.002 } else { -0.002 };
        reward += self.sim.take_pending_event_reward();

        if self.sim.ball_visible() && self.sim.state.ball_vx < 0.0 {
            let proximity = (1.0 - self.sim.state.ball_x).clamp(0.0, 1.0);
            let hh = self.sim.params.paddle_half_height.max(1.0e-6);
            let err = (self.sim.state.ball_y - self.sim.state.paddle_y).abs();
            let norm = (err / (hh * 1.5)).clamp(0.0, 1.0);
            let aligned = 1.0 - norm;
            reward += (aligned - 0.5) * 2.0 * (0.03 * proximity);
        }

        if action == "up" || action == "down" {
            reward -= 0.005;
        }

        let dt = (self.trial_period_ms.clamp(10, 60_000) as f32) / 1000.0;
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

        Some((reward.clamp(-1.0, 1.0), true))
    }

    fn refresh_stimulus_key(&mut self) {
        let bins = 8u32;
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
            let vx2 = if self.sim.state.ball2_vx >= 0.0 { "p" } else { "n" };
            let vy2 = if self.sim.state.ball2_vy >= 0.0 { "p" } else { "n" };
            self.stimulus_key = format!(
                "pong_b{bins:02}_vis{vis}_bx{bx:02}_by{by:02}_py{py:02}_vx{vx}_vy{vy}_b2vis{vis2}_b2x{b2x:02}_b2y{b2y:02}_b2vx{vx2}_b2vy{vy2}"
            );
        } else {
            self.stimulus_key = format!(
                "pong_b{bins:02}_vis{vis}_bx{bx:02}_by{by:02}_py{py:02}_vx{vx}_vy{vy}"
            );
        }
    }
}
