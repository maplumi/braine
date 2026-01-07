//! Spot Game: Minimal visual discrimination test
//!
//! A bright spot appears on either the left or right side.
//! The brain must learn to press the matching direction.
//!
//! This is the simplest possible association learning test:
//! - 2 sensors (spot_left, spot_right)
//! - 2 actions (left, right)
//! - Binary reward (correct = +1, wrong = -1)

use crate::{AppState, ControlMode, MetricsLogger, VisConfig, draw_hud_lines_wrapped};
use braine::substrate::{Brain, Stimulus};
use macroquad::prelude::*;
use std::collections::HashMap;

pub fn configure_brain(brain: &mut Brain) {
    // Minimal sensor set: just left or right
    brain.ensure_sensor("spot_left", 4);
    brain.ensure_sensor("spot_right", 4);

    // Minimal action set: just left or right
    brain.define_action("left", 6);
    brain.define_action("right", 6);
}

fn human_action() -> &'static str {
    let left = is_key_down(KeyCode::Left) || is_key_down(KeyCode::A);
    let right = is_key_down(KeyCode::Right) || is_key_down(KeyCode::D);

    if left && !right {
        "left"
    } else if right && !left {
        "right"
    } else {
        // No input = random for testing
        if macroquad::rand::gen_range(0, 2) == 0 {
            "left"
        } else {
            "right"
        }
    }
}

#[derive(Debug, Clone)]
pub struct SpotUi {
    // Which side has the spot (true = left, false = right)
    spot_is_left: bool,

    // Trial timing
    trial_frame: u32,
    trial_duration: u32, // Frames per trial
    response_made: bool,

    // Stats
    correct: u32,
    incorrect: u32,
    trials: u32,
    recent: Vec<bool>,

    // Learning curve tracking
    last_100_rate: f32,

    // Smoothed action scores for HUD
    ema_action_scores: HashMap<String, f32>,

    // Spot visual properties
    spot_radius: f32,
    spot_x: f32,
    spot_y: f32,
}

impl SpotUi {
    pub fn new(cfg: &VisConfig) -> Self {
        let mut s = Self {
            spot_is_left: true,
            trial_frame: 0,
            trial_duration: 60, // 1 second at 60fps
            response_made: false,
            correct: 0,
            incorrect: 0,
            trials: 0,
            recent: Vec::with_capacity(200),
            last_100_rate: 0.0,
            ema_action_scores: HashMap::new(),
            spot_radius: 40.0,
            spot_x: 0.0,
            spot_y: cfg.world_h * 0.5,
        };
        s.new_trial(cfg);
        s
    }

    fn new_trial(&mut self, cfg: &VisConfig) {
        // Randomly pick left or right
        self.spot_is_left = macroquad::rand::gen_range(0, 2) == 0;

        // Position the spot
        self.spot_x = if self.spot_is_left {
            cfg.world_w * 0.25
        } else {
            cfg.world_w * 0.75
        };
        self.spot_y = cfg.world_h * 0.5;

        self.trial_frame = 0;
        self.response_made = false;
    }

    fn record_outcome(&mut self, correct: bool) {
        self.recent.push(correct);
        if self.recent.len() > 200 {
            self.recent.remove(0);
        }

        // Update last 100 rate
        if self.recent.len() >= 100 {
            let last_100 = &self.recent[self.recent.len() - 100..];
            self.last_100_rate = last_100.iter().filter(|&&b| b).count() as f32 / 100.0;
        }
    }

    pub fn accuracy(&self) -> f32 {
        let total = self.correct + self.incorrect;
        if total == 0 {
            0.5
        } else {
            self.correct as f32 / total as f32
        }
    }

    pub fn flip_countdown(&self) -> usize {
        // Contingency flips every 50 trials
        const SHIFT_EVERY: usize = 50;
        let total = (self.correct + self.incorrect) as usize;
        if total == 0 {
            SHIFT_EVERY
        } else {
            let remaining = SHIFT_EVERY - (total % SHIFT_EVERY);
            if remaining == SHIFT_EVERY {
                0
            } else {
                remaining
            }
        }
    }

    fn recent_rate(&self) -> f32 {
        if self.recent.is_empty() {
            return 0.5;
        }
        let correct = self.recent.iter().filter(|&&b| b).count();
        correct as f32 / self.recent.len() as f32
    }

    pub fn tick_and_render(
        &mut self,
        cfg: &VisConfig,
        app: &AppState,
        brain: &mut Brain,
        logger: &mut MetricsLogger,
        frame: u64,
        _dt: f32,
    ) {
        self.trial_frame += 1;

        // Determine current stimulus
        let ctx = if self.spot_is_left {
            "spot_left"
        } else {
            "spot_right"
        };
        let correct_action = if self.spot_is_left { "left" } else { "right" };

        // Apply stimulus to brain
        brain.apply_stimulus(Stimulus::new(ctx, 1.0));
        brain.step();

        // Get action (only once per trial)
        let reward: f32;
        let action: &str;

        if !self.response_made {
            action = match app.mode {
                ControlMode::Human => human_action(),
                ControlMode::Braine => {
                    // Small exploration
                    if macroquad::rand::gen_range(0.0, 1.0) < 0.08 {
                        if macroquad::rand::gen_range(0, 2) == 0 {
                            "left"
                        } else {
                            "right"
                        }
                    } else {
                        let (a, _) = brain.select_action_with_meaning(ctx, 6.0);
                        if a == "left" { "left" } else { "right" }
                    }
                }
            };

            brain.note_action(action);

            // Check correctness
            let is_correct = action == correct_action;
            reward = if is_correct { 1.0 } else { -1.0 };

            if is_correct {
                self.correct += 1;
            } else {
                self.incorrect += 1;
            }
            self.trials += 1;
            self.record_outcome(is_correct);

            // Reinforce
            brain.set_neuromodulator(reward);
            brain.reinforce_action(action, reward * 0.5);

            self.response_made = true;

            // Log this trial
            if self.trials % 10 == 0 {
                logger.log_event(
                    frame,
                    "Spot",
                    &format!(
                        "trial={} correct={} accuracy={:.3} recent20={:.3} action={} expected={}",
                        self.trials,
                        is_correct,
                        self.accuracy(),
                        self.recent_rate(),
                        action,
                        correct_action
                    ),
                );
            }
        }

        brain.commit_observation();

        // End trial after duration
        if self.trial_frame >= self.trial_duration {
            self.new_trial(cfg);
        }

        // === Rendering ===
        let top = crate::TOP_UI_H;

        // Background
        draw_rectangle(
            0.0,
            top,
            cfg.world_w,
            cfg.world_h,
            Color::new(0.08, 0.08, 0.10, 1.0),
        );
        draw_rectangle_lines(0.0, top, cfg.world_w, cfg.world_h, 2.0, DARKGRAY);

        // Draw the spot
        let spot_color = if self.response_made {
            // Dim after response
            Color::new(0.4, 0.4, 0.5, 0.6)
        } else {
            // Bright yellow spot
            Color::new(1.0, 0.9, 0.3, 1.0)
        };
        draw_circle(self.spot_x, top + self.spot_y, self.spot_radius, spot_color);

        // Draw center line
        let center_x = cfg.world_w * 0.5;
        draw_line(
            center_x,
            top + 20.0,
            center_x,
            top + cfg.world_h - 20.0,
            1.0,
            Color::new(0.3, 0.3, 0.35, 0.5),
        );

        // Labels
        draw_text("LEFT", cfg.world_w * 0.25 - 25.0, top + 30.0, 20.0, GRAY);
        draw_text("RIGHT", cfg.world_w * 0.75 - 30.0, top + 30.0, 20.0, GRAY);

        // Show which key to press
        if !self.response_made {
            let hint = if self.spot_is_left {
                "← Press LEFT"
            } else {
                "Press RIGHT →"
            };
            draw_text(
                hint,
                cfg.world_w * 0.5 - 50.0,
                top + cfg.world_h - 20.0,
                18.0,
                WHITE,
            );
        }

        // Feedback flash
        if self.response_made && self.trial_frame < 30 {
            let outcome = self.recent.last().copied().unwrap_or(false);
            let flash_color = if outcome {
                Color::new(0.2, 0.8, 0.3, 0.3)
            } else {
                Color::new(0.8, 0.2, 0.2, 0.3)
            };
            draw_rectangle(0.0, top, cfg.world_w, cfg.world_h, flash_color);
        }

        // Action scores
        let scored_all = brain.ranked_actions_with_meaning(ctx, 6.0);
        for (name, score) in scored_all.iter() {
            let prev = *self.ema_action_scores.get(name).unwrap_or(&0.0);
            let alpha = 0.15;
            self.ema_action_scores
                .insert(name.clone(), (1.0 - alpha) * prev + alpha * score);
        }

        // HUD
        let hud: Vec<String> = vec![
            format!("Spot Detection Test - Trials: {}", self.trials),
            format!(
                "Correct: {} | Incorrect: {} | Accuracy: {:.1}%",
                self.correct,
                self.incorrect,
                self.accuracy() * 100.0
            ),
            format!(
                "Recent 20: {:.1}% | Last 100: {:.1}%",
                self.recent_rate() * 100.0,
                self.last_100_rate * 100.0
            ),
            format!("Current: {} | Expected action: {}", ctx, correct_action),
            format!(
                "Action scores: left={:+.2} right={:+.2}",
                self.ema_action_scores.get("left").unwrap_or(&0.0),
                self.ema_action_scores.get("right").unwrap_or(&0.0)
            ),
            format!(
                "Trial progress: {}/{} frames",
                self.trial_frame, self.trial_duration
            ),
        ];

        let hud_rect = Rect::new(
            crate::UI_MARGIN,
            crate::HUD_START_Y,
            screen_width() - 2.0 * crate::UI_MARGIN,
            crate::TOP_UI_H - crate::HUD_START_Y - crate::UI_MARGIN,
        );
        draw_hud_lines_wrapped(&hud, hud_rect, WHITE);

        // Milestone markers
        let milestone = if self.last_100_rate >= 0.95 {
            "★★★ MASTERED"
        } else if self.last_100_rate >= 0.85 {
            "★★ LEARNED"
        } else if self.last_100_rate >= 0.70 {
            "★ LEARNING"
        } else if self.trials < 20 {
            "Starting..."
        } else {
            "Not yet learned"
        };
        draw_text(
            milestone,
            cfg.world_w - 150.0,
            top + 30.0,
            18.0,
            if self.last_100_rate >= 0.85 {
                GREEN
            } else {
                YELLOW
            },
        );
    }
}
