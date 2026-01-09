// This file has been deleted.
// All code has been removed.
// (deleted)
use crate::{AppState, ControlMode, MetricsLogger, VisConfig, draw_hud_lines_wrapped};
use braine::substrate::{Brain, Stimulus};
use macroquad::prelude::*;
use std::collections::HashMap;

pub fn configure_brain(brain: &mut Brain) {
    for cue in ["blue", "yellow"] {
        for dx in ["L", "C", "R"] {
            for dy in ["U", "C", "D"] {
                let name = format!("beacon_ctx_{cue}_{dx}{dy}");
                brain.ensure_sensor(&name, 4);
            }
        }
    }

    for dx in ["L", "C", "R"] {
        let name = format!("beacon_distr_dx_{dx}");
        brain.ensure_sensor(&name, 3);
    }
    for dy in ["U", "C", "D"] {
        let name = format!("beacon_distr_dy_{dy}");
        brain.ensure_sensor(&name, 3);
    }

    brain.define_action("up", 6);
    brain.define_action("down", 6);
    brain.define_action("left", 6);
    brain.define_action("right", 6);
    brain.define_action("stay", 6);
}

fn bucket3(v: f32, dead: f32) -> &'static str {
    if v < -dead {
        "L"
    } else if v > dead {
        "R"
    } else {
        "C"
    }
}

fn bucket3y(v: f32, dead: f32) -> &'static str {
    if v < -dead {
        "U"
    } else if v > dead {
        "D"
    } else {
        "C"
    }
}

fn human_action() -> &'static str {
    let left = is_key_down(KeyCode::Left) || is_key_down(KeyCode::A);
    let right = is_key_down(KeyCode::Right) || is_key_down(KeyCode::D);
    let up = is_key_down(KeyCode::Up) || is_key_down(KeyCode::W);
    let down = is_key_down(KeyCode::Down) || is_key_down(KeyCode::S);

    if up && !down {
        "up"
    } else if down && !up {
        "down"
    } else if left && !right {
        "left"
    } else if right && !left {
        "right"
    } else {
        "stay"
    }
}

#[derive(Debug, Clone)]
pub struct BeaconUi {
    agent_x: f32,
    agent_y: f32,
    speed: f32,

    blue_x: f32,
    blue_y: f32,

    yellow_x: f32,
    yellow_y: f32,

    // Which beacon is the current target.
    target_is_blue: bool,

    shift_every_hits: u32,
    hits: u32,

    score: f32,
    recent: Vec<bool>,

    ema_action_scores: HashMap<String, f32>,

    last_flip_recent_before: f32,
}

impl BeaconUi {
    pub fn new(cfg: &VisConfig) -> Self {
        let mut s = Self {
            agent_x: cfg.world_w * 0.5,
            agent_y: cfg.world_h * 0.5,
            speed: 260.0,
            blue_x: cfg.world_w * 0.25,
            blue_y: cfg.world_h * 0.25,
            yellow_x: cfg.world_w * 0.75,
            yellow_y: cfg.world_h * 0.75,
            target_is_blue: true,
            shift_every_hits: 35,
            hits: 0,
            score: 0.0,
            recent: Vec::with_capacity(400),
            ema_action_scores: HashMap::new(),
            last_flip_recent_before: 0.0,
        };
        s.random_walk_targets(cfg);
        s
    }

    pub fn flip_countdown(&self) -> u32 {
        let n = self.shift_every_hits.max(1);
        let m = self.hits % n;
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

    fn random_walk_targets(&mut self, cfg: &VisConfig) {
        let step = 40.0;
        self.blue_x =
            (self.blue_x + macroquad::rand::gen_range(-step, step)).clamp(40.0, cfg.world_w - 40.0);
        self.blue_y =
            (self.blue_y + macroquad::rand::gen_range(-step, step)).clamp(40.0, cfg.world_h - 40.0);
        self.yellow_x = (self.yellow_x + macroquad::rand::gen_range(-step, step))
            .clamp(40.0, cfg.world_w - 40.0);
        self.yellow_y = (self.yellow_y + macroquad::rand::gen_range(-step, step))
            .clamp(40.0, cfg.world_h - 40.0);
    }

    fn ctx_name(&self, dx: &'static str, dy: &'static str) -> &'static str {
        match (self.target_is_blue, dx, dy) {
            (true, "L", "U") => "beacon_ctx_blue_LU",
            (true, "L", "C") => "beacon_ctx_blue_LC",
            (true, "L", "D") => "beacon_ctx_blue_LD",
            (true, "C", "U") => "beacon_ctx_blue_CU",
            (true, "C", "C") => "beacon_ctx_blue_CC",
            (true, "C", "D") => "beacon_ctx_blue_CD",
            (true, "R", "U") => "beacon_ctx_blue_RU",
            (true, "R", "C") => "beacon_ctx_blue_RC",
            (true, "R", "D") => "beacon_ctx_blue_RD",
            (false, "L", "U") => "beacon_ctx_yellow_LU",
            (false, "L", "C") => "beacon_ctx_yellow_LC",
            (false, "L", "D") => "beacon_ctx_yellow_LD",
            (false, "C", "U") => "beacon_ctx_yellow_CU",
            (false, "C", "C") => "beacon_ctx_yellow_CC",
            (false, "C", "D") => "beacon_ctx_yellow_CD",
            (false, "R", "U") => "beacon_ctx_yellow_RU",
            (false, "R", "C") => "beacon_ctx_yellow_RC",
            _ => "beacon_ctx_yellow_RD",
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
        // Move beacons slowly.
        if (frame % 12) == 0 {
            self.random_walk_targets(cfg);
        }

        let (tx, ty) = if self.target_is_blue {
            (self.blue_x, self.blue_y)
        } else {
            (self.yellow_x, self.yellow_y)
        };
        let dx = tx - self.agent_x;
        let dy = ty - self.agent_y;
        let bx = bucket3(dx, 34.0);
        let by = bucket3y(dy, 34.0);
        let ctx = self.ctx_name(bx, by);

        let (dx2, dy2) = if self.target_is_blue {
            (self.yellow_x - self.agent_x, self.yellow_y - self.agent_y)
        } else {
            (self.blue_x - self.agent_x, self.blue_y - self.agent_y)
        };
        let dbx = bucket3(dx2, 34.0);
        let dby = bucket3y(dy2, 34.0);
        let dbx_name = match dbx {
            "L" => "beacon_distr_dx_L",
            "R" => "beacon_distr_dx_R",
            _ => "beacon_distr_dx_C",
        };
        let dby_name = match dby {
            "U" => "beacon_distr_dy_U",
            "D" => "beacon_distr_dy_D",
            _ => "beacon_distr_dy_C",
        };

        brain.apply_stimulus(Stimulus::new(ctx, 1.0));
        brain.apply_stimulus(Stimulus::new(dbx_name, 0.6));
        brain.apply_stimulus(Stimulus::new(dby_name, 0.6));
        brain.step();

        let action: &'static str = match app.mode {
            ControlMode::Human => human_action(),
            ControlMode::Braine => {
                if macroquad::rand::gen_range(0.0, 1.0) < 0.07 {
                    match macroquad::rand::gen_range(0, 5) {
                        0 => "up",
                        1 => "down",
                        2 => "left",
                        3 => "right",
                        _ => "stay",
                    }
                } else {
                    let (a, _s) = brain.select_action_with_meaning(ctx, 7.5);
                    if a == "up" {
                        "up"
                    } else if a == "down" {
                        "down"
                    } else if a == "left" {
                        "left"
                    } else if a == "right" {
                        "right"
                    } else {
                        "stay"
                    }
                }
            }
        };

        brain.note_action(action);
        brain.note_compound_symbol(&["pair", ctx, action]);

        let mut vx: f32 = 0.0;
        let mut vy: f32 = 0.0;
        match action {
            "left" => vx = -1.0,
            "right" => vx = 1.0,
            "up" => vy = -1.0,
            "down" => vy = 1.0,
            _ => {}
        }
        let n: f32 = (vx * vx + vy * vy).sqrt().max(1.0);
        vx /= n;
        vy /= n;

        self.agent_x = (self.agent_x + vx * self.speed * dt).clamp(14.0, cfg.world_w - 14.0);
        self.agent_y = (self.agent_y + vy * self.speed * dt).clamp(14.0, cfg.world_h - 14.0);

        let mut reward = 0.0f32;
        let mut outcome_ok: Option<bool> = None;

        // Reward on entering target radius.
        let r = 22.0;
        let d2 = (self.agent_x - tx).powi(2) + (self.agent_y - ty).powi(2);
        if d2 <= r * r {
            reward = 1.0;
            outcome_ok = Some(true);
            self.hits = self.hits.wrapping_add(1);
            self.score += reward;
            self.record_outcome(true);

            if self.shift_every_hits > 0 && (self.hits % self.shift_every_hits) == 0 {
                self.target_is_blue = !self.target_is_blue;
                self.last_flip_recent_before = self.recent_rate_n(60);
                logger.log_event(
                    frame,
                    "Beacon",
                    &format!(
                        "FLIP_MARKER hits={} recent60_before={:.3} target={}",
                        self.hits,
                        self.last_flip_recent_before,
                        if self.target_is_blue {
                            "BLUE"
                        } else {
                            "YELLOW"
                        }
                    ),
                );
            }

            // Move targets to avoid camping.
            self.random_walk_targets(cfg);
        }

        // Mild penalty for being close to distractor while not target.
        let (dxo, dyo) = if self.target_is_blue {
            (self.agent_x - self.yellow_x, self.agent_y - self.yellow_y)
        } else {
            (self.agent_x - self.blue_x, self.agent_y - self.blue_y)
        };
        if (dxo * dxo + dyo * dyo) <= (r * r) {
            reward += -0.2;
            outcome_ok = outcome_ok.or(Some(false));
        }

        if outcome_ok.is_some() {
            brain.reinforce_action(action, reward);
        }
        brain.commit_observation();

        // Draw arena: border indicates which beacon is relevant.
        let top = crate::TOP_UI_H;
        let border = if self.target_is_blue {
            Color::new(0.15, 0.20, 0.45, 1.0)
        } else {
            Color::new(0.45, 0.40, 0.12, 1.0)
        };
        draw_rectangle_lines(0.0, top, cfg.world_w, cfg.world_h, 6.0, border);

        draw_circle(self.blue_x, top + self.blue_y, 12.0, BLUE);
        draw_circle(self.yellow_x, top + self.yellow_y, 12.0, YELLOW);
        draw_circle(self.agent_x, top + self.agent_y, 10.0, WHITE);

        // HUD.
        let scored_all = brain.ranked_actions_with_meaning(ctx, 7.5);
        for (name, score) in scored_all.iter().take(6) {
            let prev = *self.ema_action_scores.get(name).unwrap_or(&0.0);
            let alpha = 0.18;
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
                "Beacon target={} hits={} flip_in={} score={:+.1}",
                if self.target_is_blue {
                    "BLUE"
                } else {
                    "YELLOW"
                },
                self.hits,
                self.flip_countdown(),
                self.score
            ),
            format!("ctx={} action={} reward={:+.1}", ctx, action, reward),
            format!(
                "top_actions(ema): {}",
                scored_ema
                    .iter()
                    .take(4)
                    .map(|(n, s)| format!("{}:{:+.2}", n, s))
                    .collect::<Vec<_>>()
                    .join("  ")
            ),
        ];

        logger.log_snapshot(
            frame,
            "Beacon",
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

        let hint = "WASD/Arrows move (Human).";
        draw_text(hint, crate::UI_MARGIN, crate::TOP_UI_H + 18.0, 18.0, GRAY);
    }
}
