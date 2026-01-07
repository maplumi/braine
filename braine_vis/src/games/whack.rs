use crate::{AppState, ControlMode, MetricsLogger, VisConfig, draw_hud_lines_wrapped};
use braine::substrate::{Brain, Stimulus};
use macroquad::prelude::*;
use std::collections::HashMap;

pub fn configure_brain(brain: &mut Brain) {
    // Context combines mapping id and active lane.
    for map in ["0", "1"] {
        for lane in ["0", "1", "2"] {
            let name = format!("whack_ctx_map{map}_lane{lane}");
            brain.ensure_sensor(&name, 4);
        }
    }

    brain.define_action("A", 6);
    brain.define_action("B", 6);
    brain.define_action("C", 6);
}

fn human_action() -> &'static str {
    if is_key_down(KeyCode::Key1) {
        "A"
    } else if is_key_down(KeyCode::Key2) {
        "B"
    } else if is_key_down(KeyCode::Key3) {
        "C"
    } else {
        "A"
    }
}

#[derive(Debug, Clone)]
pub struct WhackUi {
    // Which mapping is active: map0 or map1 (label->lane wiring).
    map_flip: bool,

    // Active target lane.
    active_lane: usize,
    target_t: f32,
    target_visible_for: f32,

    // Delay between targets.
    cooldown_t: f32,

    // Flip schedule.
    shift_every_outcomes: u32,
    outcomes: u32,

    score: f32,
    recent: Vec<bool>,

    ema_action_scores: HashMap<String, f32>,

    last_flip_recent_before: f32,
}

impl WhackUi {
    pub fn new() -> Self {
        Self {
            map_flip: false,
            active_lane: 0,
            target_t: 0.0,
            target_visible_for: 0.9,
            cooldown_t: 0.0,
            shift_every_outcomes: 40,
            outcomes: 0,
            score: 0.0,
            recent: Vec::with_capacity(400),
            ema_action_scores: HashMap::new(),
            last_flip_recent_before: 0.0,
        }
    }

    pub fn flip_countdown(&self) -> u32 {
        let n = self.shift_every_outcomes.max(1);
        let m = self.outcomes % n;
        if m == 0 { n } else { n - m }
    }

    fn record_outcome(&mut self, ok: bool) {
        self.recent.push(ok);
        if self.recent.len() > 400 {
            self.recent.remove(0);
        }
    }

    fn recent_rate_n(&self, n: usize) -> f32 {
        if self.recent.is_empty() {
            return 0.0;
        }
        let n = n.min(self.recent.len());
        let tail = &self.recent[self.recent.len() - n..];
        let good = tail.iter().filter(|&&b| b).count();
        good as f32 / n as f32
    }

    fn label_to_lane(&self, label: &str) -> usize {
        // map0: A->0,B->1,C->2
        // map1: A->2,B->1,C->0 (reversed)
        match (self.map_flip, label) {
            (false, "A") => 0,
            (false, "B") => 1,
            (false, _) => 2,
            (true, "A") => 2,
            (true, "B") => 1,
            (true, _) => 0,
        }
    }

    fn ctx_name(&self) -> &'static str {
        match (self.map_flip, self.active_lane) {
            (false, 0) => "whack_ctx_map0_lane0",
            (false, 1) => "whack_ctx_map0_lane1",
            (false, _) => "whack_ctx_map0_lane2",
            (true, 0) => "whack_ctx_map1_lane0",
            (true, 1) => "whack_ctx_map1_lane1",
            (true, _) => "whack_ctx_map1_lane2",
        }
    }

    fn spawn_target(&mut self) {
        self.active_lane = macroquad::rand::gen_range(0, 3) as usize;
        self.target_t = 0.0;
    }

    fn maybe_flip(&mut self, logger: &mut MetricsLogger, frame: u64) {
        if self.shift_every_outcomes > 0 && (self.outcomes % self.shift_every_outcomes) == 0 {
            self.map_flip = !self.map_flip;
            self.last_flip_recent_before = self.recent_rate_n(50);
            logger.log_event(
                frame,
                "Whack",
                &format!(
                    "FLIP_MARKER outcomes={} recent50_before={:.3} map_flip={}",
                    self.outcomes, self.last_flip_recent_before, self.map_flip
                ),
            );
        }
    }

    pub fn tick_and_render(
        &mut self,
        cfg: &VisConfig,
        app: &AppState,
        brain: &mut Brain,
        logger: &mut MetricsLogger,
        frame: u64,
        dt: f32,
    ) {
        // Progress timers.
        if self.cooldown_t > 0.0 {
            self.cooldown_t = (self.cooldown_t - dt).max(0.0);
            if self.cooldown_t == 0.0 {
                self.spawn_target();
            }
        } else {
            self.target_t += dt;
        }

        let ctx = self.ctx_name();
        brain.apply_stimulus(Stimulus::new(ctx, 1.0));
        brain.step();

        let action: &'static str = match app.mode {
            ControlMode::Human => human_action(),
            ControlMode::Braine => {
                if macroquad::rand::gen_range(0.0, 1.0) < 0.10 {
                    match macroquad::rand::gen_range(0, 3) {
                        0 => "A",
                        1 => "B",
                        _ => "C",
                    }
                } else {
                    let (a, _s) = brain.select_action_with_meaning(ctx, 8.0);
                    if a == "A" {
                        "A"
                    } else if a == "B" {
                        "B"
                    } else {
                        "C"
                    }
                }
            }
        };

        brain.note_action(action);
        brain.note_compound_symbol(&["pair", ctx, action]);

        // Resolve outcome only when a target is visible.
        let mut reward = 0.0f32;
        let mut outcome: Option<bool> = None;

        let target_visible = self.cooldown_t == 0.0;
        if target_visible {
            let chosen_lane = self.label_to_lane(action);
            let correct = chosen_lane == self.active_lane;

            // In Braine mode we always "press" an action each tick; reward only when
            // we're within the reaction window.
            if self.target_t <= self.target_visible_for {
                if correct {
                    reward = 1.0;
                    outcome = Some(true);
                    self.cooldown_t = 0.35;
                } else if macroquad::rand::gen_range(0.0, 1.0) < 0.05 {
                    // Small occasional penalty for wrong hits to keep signal sparse.
                    reward = -0.2;
                    outcome = Some(false);
                }
            } else {
                // Timeout: miss.
                reward = -1.0;
                outcome = Some(false);
                self.cooldown_t = 0.35;
            }

            if self.cooldown_t > 0.0 {
                // Target is resolved; next spawn after cooldown.
                self.outcomes = self.outcomes.wrapping_add(1);
                if let Some(ok) = outcome {
                    self.record_outcome(ok);
                }
                self.score += reward;
                self.maybe_flip(logger, frame);
            }
        }

        if outcome.is_some() {
            brain.reinforce_action(action, reward);
        }
        brain.commit_observation();

        // Draw lanes.
        let top = crate::TOP_UI_H;
        let lane_w = cfg.world_w / 3.0;
        for i in 0..3 {
            let x0 = lane_w * (i as f32);
            draw_rectangle_lines(x0, top, lane_w, cfg.world_h, 2.0, DARKGRAY);
        }

        // Draw mapping labels above lanes.
        let labels = if !self.map_flip {
            ["A", "B", "C"]
        } else {
            ["C", "B", "A"]
        };
        for i in 0..3 {
            let x = lane_w * (i as f32) + 10.0;
            draw_text(labels[i], x, top + 22.0, 22.0, GRAY);
        }

        // Draw target.
        if target_visible {
            let x = lane_w * (self.active_lane as f32) + lane_w * 0.5;
            let y = top + cfg.world_h * 0.55;
            let pulse = 1.0 + 0.2 * (get_time() as f32 * 10.0).sin();
            draw_circle(x, y, 28.0 * pulse, YELLOW);
        }

        // HUD.
        let scored_all = brain.ranked_actions_with_meaning(ctx, 8.0);
        for (name, score) in scored_all.iter().take(6) {
            let prev = *self.ema_action_scores.get(name).unwrap_or(&0.0);
            let alpha = 0.20;
            self.ema_action_scores
                .insert(name.clone(), (1.0 - alpha) * prev + alpha * (*score));
        }
        let mut scored_ema: Vec<(String, f32)> = self
            .ema_action_scores
            .iter()
            .map(|(k, v)| (k.clone(), *v))
            .collect();
        scored_ema.sort_by(|a, b| b.1.total_cmp(&a.1));

        let snap = braine::observer::BrainAdapter::new(brain).snapshot();

        let hud: Vec<String> = vec![
            format!(
                "Whack map={} outcomes={} flip_in={} score={:+.1}",
                if self.map_flip { "REV" } else { "NORM" },
                self.outcomes,
                self.flip_countdown(),
                self.score
            ),
            format!(
                "active_lane={} target_t={:.2} action={} reward={:+.1}",
                self.active_lane, self.target_t, action, reward
            ),
            format!(
                "top_actions(ema): {}",
                scored_ema
                    .iter()
                    .take(3)
                    .map(|(n, s)| format!("{}:{:+.2}", n, s))
                    .collect::<Vec<_>>()
                    .join("  ")
            ),
        ];

        logger.log_snapshot(
            frame,
            "Whack",
            app.mode,
            reward,
            &hud,
            &snap.last_reinforced_actions,
        );

        let hud_rect = Rect::new(
            crate::UI_MARGIN,
            crate::HUD_START_Y,
            screen_width() - 2.0 * crate::UI_MARGIN,
            crate::TOP_UI_H - crate::HUD_START_Y - crate::UI_MARGIN,
        );
        draw_hud_lines_wrapped(&hud, hud_rect, WHITE);

        let hint = "Human: press 1/2/3 (A/B/C labels).";
        draw_text(hint, crate::UI_MARGIN, crate::TOP_UI_H + 18.0, 18.0, GRAY);
    }
}
use crate::{AppState, ControlMode, MetricsLogger, VisConfig, draw_hud_lines_wrapped};
use braine::substrate::{Brain, Stimulus};
use macroquad::prelude::*;
use std::collections::HashMap;

pub fn configure_brain(brain: &mut Brain) {
    // Context combines mapping id and active lane.
    for map in ["0", "1"] {
        for lane in ["0", "1", "2"] {
            let name = format!("whack_ctx_map{map}_lane{lane}");
            brain.ensure_sensor(&name, 4);
        }
    }

    brain.define_action("A", 6);
    brain.define_action("B", 6);
    brain.define_action("C", 6);
}

fn human_action() -> &'static str {
    if is_key_down(KeyCode::Key1) {
        "A"
    } else if is_key_down(KeyCode::Key2) {
        "B"
    } else if is_key_down(KeyCode::Key3) {
        "C"
    } else {
        "A"
    }
}

#[derive(Debug, Clone)]
pub struct WhackUi {
    // Which mapping is active: map0 or map1 (label->lane wiring).
    map_flip: bool,

    // Active target lane.
    active_lane: usize,
    target_t: f32,
    target_visible_for: f32,

    // Delay between targets.
    cooldown_t: f32,

    // Flip schedule.
    shift_every_outcomes: u32,
    outcomes: u32,

    score: f32,
    recent: Vec<bool>,

    ema_action_scores: HashMap<String, f32>,

    last_flip_recent_before: f32,
}

impl WhackUi {
    pub fn new() -> Self {
        Self {
            map_flip: false,
            active_lane: 0,
            target_t: 0.0,
            target_visible_for: 0.9,
            cooldown_t: 0.0,
            shift_every_outcomes: 40,
            outcomes: 0,
            score: 0.0,
            recent: Vec::with_capacity(400),
            ema_action_scores: HashMap::new(),
            last_flip_recent_before: 0.0,
        }
    }

    pub fn flip_countdown(&self) -> u32 {
        let n = self.shift_every_outcomes.max(1);
        let m = self.outcomes % n;
        if m == 0 { n } else { n - m }
    }

    fn record_outcome(&mut self, ok: bool) {
        self.recent.push(ok);
        if self.recent.len() > 400 {
            self.recent.remove(0);
        }
    }

    fn recent_rate_n(&self, n: usize) -> f32 {
        if self.recent.is_empty() {
            return 0.0;
        }
        let n = n.min(self.recent.len());
        let tail = &self.recent[self.recent.len() - n..];
        let good = tail.iter().filter(|&&b| b).count();
        good as f32 / n as f32
    }

    fn label_to_lane(&self, label: &str) -> usize {
        // map0: A->0,B->1,C->2
        // map1: A->2,B->1,C->0 (reversed)
        match (self.map_flip, label) {
            (false, "A") => 0,
            (false, "B") => 1,
            (false, _) => 2,
            (true, "A") => 2,
            (true, "B") => 1,
            (true, _) => 0,
        }
    }

    fn ctx_name(&self) -> &'static str {
        match (self.map_flip, self.active_lane) {
            (false, 0) => "whack_ctx_map0_lane0",
            (false, 1) => "whack_ctx_map0_lane1",
            (false, _) => "whack_ctx_map0_lane2",
            (true, 0) => "whack_ctx_map1_lane0",
            (true, 1) => "whack_ctx_map1_lane1",
            (true, _) => "whack_ctx_map1_lane2",
        }
    }

    fn spawn_target(&mut self) {
        self.active_lane = macroquad::rand::gen_range(0, 3) as usize;
        self.target_t = 0.0;
    }

    fn maybe_flip(&mut self, logger: &mut MetricsLogger, frame: u64) {
        if self.shift_every_outcomes > 0 && (self.outcomes % self.shift_every_outcomes) == 0 {
            self.map_flip = !self.map_flip;
            self.last_flip_recent_before = self.recent_rate_n(50);
            logger.log_event(
                frame,
                "Whack",
                &format!(
                    "FLIP_MARKER outcomes={} recent50_before={:.3} map_flip={}",
                    self.outcomes, self.last_flip_recent_before, self.map_flip
                ),
            );
        }
    }

    pub fn tick_and_render(
        &mut self,
        cfg: &VisConfig,
        app: &AppState,
        brain: &mut Brain,
        logger: &mut MetricsLogger,
        frame: u64,
        dt: f32,
    ) {
        // Progress timers.
        if self.cooldown_t > 0.0 {
            self.cooldown_t = (self.cooldown_t - dt).max(0.0);
            if self.cooldown_t == 0.0 {
                self.spawn_target();
            }
        } else {
            self.target_t += dt;
        }

        let ctx = self.ctx_name();
        brain.apply_stimulus(Stimulus::new(ctx, 1.0));
        brain.step();

        let action: &'static str = match app.mode {
            ControlMode::Human => human_action(),
            ControlMode::Braine => {
                if macroquad::rand::gen_range(0.0, 1.0) < 0.10 {
                    match macroquad::rand::gen_range(0, 3) {
                        0 => "A",
                        1 => "B",
                        _ => "C",
                    }
                } else {
                    let (a, _s) = brain.select_action_with_meaning(ctx, 8.0);
                    if a == "A" {
                        "A"
                    } else if a == "B" {
                        "B"
                    } else {
                        "C"
                    }
                }
            }
        };

        brain.note_action(action);
        brain.note_compound_symbol(&["pair", ctx, action]);

        // Resolve outcome only when a target is visible.
        let mut reward = 0.0f32;
        let mut outcome: Option<bool> = None;

        let target_visible = self.cooldown_t == 0.0;
        if target_visible {
            let chosen_lane = self.label_to_lane(action);
            let correct = chosen_lane == self.active_lane;

            // In Braine mode we always "press" an action each tick; reward only when
            // we're within the reaction window.
            if self.target_t <= self.target_visible_for {
                if correct {
                    reward = 1.0;
                    outcome = Some(true);
                    self.cooldown_t = 0.35;
                } else if macroquad::rand::gen_range(0.0, 1.0) < 0.05 {
                    // Small occasional penalty for wrong hits to keep signal sparse.
                    reward = -0.2;
                    outcome = Some(false);
                }
            } else {
                // Timeout: miss.
                reward = -1.0;
                outcome = Some(false);
                self.cooldown_t = 0.35;
            }

            if self.cooldown_t > 0.0 {
                // Target is resolved; next spawn after cooldown.
                self.outcomes = self.outcomes.wrapping_add(1);
                if let Some(ok) = outcome {
                    self.record_outcome(ok);
                }
                self.score += reward;
                self.maybe_flip(logger, frame);
            }
        }

        if outcome.is_some() {
            brain.reinforce_action(action, reward);
        }
        brain.commit_observation();

        // Draw lanes.
        let top = crate::TOP_UI_H;
        let lane_w = cfg.world_w / 3.0;
        for i in 0..3 {
            let x0 = lane_w * (i as f32);
            draw_rectangle_lines(x0, top, lane_w, cfg.world_h, 2.0, DARKGRAY);
        }

        // Draw mapping labels above lanes.
        let labels = if !self.map_flip {
            ["A", "B", "C"]
        } else {
            ["C", "B", "A"]
        };
        for i in 0..3 {
            let x = lane_w * (i as f32) + 10.0;
            draw_text(labels[i], x, top + 22.0, 22.0, GRAY);
        }

        // Draw target.
        if target_visible {
            let x = lane_w * (self.active_lane as f32) + lane_w * 0.5;
            let y = top + cfg.world_h * 0.55;
            let pulse = 1.0 + 0.2 * (get_time() as f32 * 10.0).sin();
            draw_circle(x, y, 28.0 * pulse, YELLOW);
        }

        // HUD.
        let scored_all = brain.ranked_actions_with_meaning(ctx, 8.0);
        for (name, score) in scored_all.iter().take(6) {
            let prev = *self.ema_action_scores.get(name).unwrap_or(&0.0);
            let alpha = 0.20;
            self.ema_action_scores
                .insert(name.clone(), (1.0 - alpha) * prev + alpha * (*score));
        }
        let mut scored_ema: Vec<(String, f32)> = self
            .ema_action_scores
            .iter()
            .map(|(k, v)| (k.clone(), *v))
            .collect();
        scored_ema.sort_by(|a, b| b.1.total_cmp(&a.1));

        let snap = braine::observer::BrainAdapter::new(brain).snapshot();

        let hud: Vec<String> = vec![
            format!(
                "Whack map={} outcomes={} flip_in={} score={:+.1}",
                if self.map_flip { "REV" } else { "NORM" },
                self.outcomes,
                self.flip_countdown(),
                self.score
            ),
            format!(
                "active_lane={} target_t={:.2} action={} reward={:+.1}",
                self.active_lane, self.target_t, action, reward
            ),
            format!(
                "top_actions(ema): {}",
                scored_ema
                    .iter()
                    .take(3)
                    .map(|(n, s)| format!("{}:{:+.2}", n, s))
                    .collect::<Vec<_>>()
                    .join("  ")
            ),
        ];

        logger.log_snapshot(
            frame,
            "Whack",
            app.mode,
            reward,
            &hud,
            &snap.last_reinforced_actions,
        );

        let hud_rect = Rect::new(
            crate::UI_MARGIN,
            crate::HUD_START_Y,
            screen_width() - 2.0 * crate::UI_MARGIN,
            crate::TOP_UI_H - crate::HUD_START_Y - crate::UI_MARGIN,
        );
        draw_hud_lines_wrapped(&hud, hud_rect, WHITE);

        let hint = "Human: press 1/2/3 (A/B/C labels).";
        draw_text(hint, crate::UI_MARGIN, crate::TOP_UI_H + 18.0, 18.0, GRAY);
    }
}
