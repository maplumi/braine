use crate::{AppState, ControlMode, MetricsLogger, VisConfig, draw_hud_lines_wrapped};
use braine::substrate::{Brain, Stimulus};
use macroquad::prelude::*;
use std::collections::HashMap;

#[derive(Debug, Clone, Copy)]
pub struct ForageTickResult {
    pub ctx: &'static str,
    pub action: &'static str,
    pub reward: f32,
    pub just_flipped: bool,
}

pub fn configure_brain(brain: &mut Brain) {
    // Context is a coarse (cue x food-direction) combination.
    for cue in ["green", "red"] {
        for dx in ["L", "C", "R"] {
            for dy in ["U", "C", "D"] {
                let name = format!("forage_ctx_{cue}_{dx}{dy}");
                brain.ensure_sensor(&name, 4);
            }
        }
    }

    for dx in ["L", "C", "R"] {
        let name = format!("forage_poison_dx_{dx}");
        brain.ensure_sensor(&name, 3);
    }
    for dy in ["U", "C", "D"] {
        let name = format!("forage_poison_dy_{dy}");
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
pub struct ForageUi {
    agent_x: f32,
    agent_y: f32,
    speed: f32,

    green_x: f32,
    green_y: f32,

    red_x: f32,
    red_y: f32,

    // If true, green is rewarded and red is punished.
    green_is_good: bool,

    shift_every_outcomes: u32,
    outcomes: u32,

    score: f32,
    recent: Vec<bool>,

    // Flip evaluation metrics.
    last_flip_at_frame: u64,
    last_flip_recent_before: f32,
    last_flip_recovered_in_outcomes: Option<u32>,

    ema_action_scores: HashMap<String, f32>,
}

impl ForageUi {
    pub fn new(cfg: &VisConfig) -> Self {
        let mut s = Self {
            agent_x: cfg.world_w * 0.5,
            agent_y: cfg.world_h * 0.5,
            speed: 240.0,
            green_x: cfg.world_w * 0.25,
            green_y: cfg.world_h * 0.25,
            red_x: cfg.world_w * 0.75,
            red_y: cfg.world_h * 0.75,
            green_is_good: true,
            shift_every_outcomes: 50,
            outcomes: 0,
            score: 0.0,
            recent: Vec::with_capacity(400),
            last_flip_at_frame: 0,
            last_flip_recent_before: 0.0,
            last_flip_recovered_in_outcomes: None,
            ema_action_scores: HashMap::new(),
        };
        s.randomize_targets(cfg);
        s
    }

    pub fn flip_countdown(&self) -> u32 {
        let n = self.shift_every_outcomes.max(1);
        let m = self.outcomes % n;
        if m == 0 { n } else { n - m }
    }

    fn randomize_targets(&mut self, cfg: &VisConfig) {
        self.green_x = macroquad::rand::gen_range(40.0, cfg.world_w - 40.0);
        self.green_y = macroquad::rand::gen_range(40.0, cfg.world_h - 40.0);
        self.red_x = macroquad::rand::gen_range(40.0, cfg.world_w - 40.0);
        self.red_y = macroquad::rand::gen_range(40.0, cfg.world_h - 40.0);
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

    fn tick(
        &mut self,
        cfg: &VisConfig,
        mode: ControlMode,
        brain: &mut Brain,
        dt: f32,
    ) -> ForageTickResult {
        let cue = if self.green_is_good { "green" } else { "red" };

        // Food direction is always "good" target.
        let (good_x, good_y) = if self.green_is_good {
            (self.green_x, self.green_y)
        } else {
            (self.red_x, self.red_y)
        };

        let dx = good_x - self.agent_x;
        let dy = good_y - self.agent_y;

        let bx = bucket3(dx, 32.0);
        let by = bucket3y(dy, 32.0);

        let ctx_name = match (cue, bx, by) {
            ("green", "L", "U") => "forage_ctx_green_LU",
            ("green", "L", "C") => "forage_ctx_green_LC",
            ("green", "L", "D") => "forage_ctx_green_LD",
            ("green", "C", "U") => "forage_ctx_green_CU",
            ("green", "C", "C") => "forage_ctx_green_CC",
            ("green", "C", "D") => "forage_ctx_green_CD",
            ("green", "R", "U") => "forage_ctx_green_RU",
            ("green", "R", "C") => "forage_ctx_green_RC",
            ("green", "R", "D") => "forage_ctx_green_RD",
            ("red", "L", "U") => "forage_ctx_red_LU",
            ("red", "L", "C") => "forage_ctx_red_LC",
            ("red", "L", "D") => "forage_ctx_red_LD",
            ("red", "C", "U") => "forage_ctx_red_CU",
            ("red", "C", "C") => "forage_ctx_red_CC",
            ("red", "C", "D") => "forage_ctx_red_CD",
            ("red", "R", "U") => "forage_ctx_red_RU",
            ("red", "R", "C") => "forage_ctx_red_RC",
            _ => "forage_ctx_red_RD",
        };

        // Poison direction: always the "other" orb.
        let (bad_x, bad_y) = if self.green_is_good {
            (self.red_x, self.red_y)
        } else {
            (self.green_x, self.green_y)
        };
        let pdx = bad_x - self.agent_x;
        let pdy = bad_y - self.agent_y;
        let pdx_b = bucket3(pdx, 32.0);
        let pdy_b = bucket3y(pdy, 32.0);

        let pdx_name = match pdx_b {
            "L" => "forage_poison_dx_L",
            "R" => "forage_poison_dx_R",
            _ => "forage_poison_dx_C",
        };
        let pdy_name = match pdy_b {
            "U" => "forage_poison_dy_U",
            "D" => "forage_poison_dy_D",
            _ => "forage_poison_dy_C",
        };

        brain.apply_stimulus(Stimulus::new(ctx_name, 1.0));
        brain.apply_stimulus(Stimulus::new(pdx_name, 0.7));
        brain.apply_stimulus(Stimulus::new(pdy_name, 0.7));
        brain.step();

        let action: &'static str = match mode {
            ControlMode::Human => human_action(),
            ControlMode::Braine => {
                if macroquad::rand::gen_range(0.0, 1.0) < 0.08 {
                    match macroquad::rand::gen_range(0, 5) {
                        0 => "up",
                        1 => "down",
                        2 => "left",
                        3 => "right",
                        _ => "stay",
                    }
                } else {
                    let (a, _s) = brain.select_action_with_meaning(ctx_name, 7.0);
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
        let pair = format!("pair::{ctx_name}::{action}");
        brain.note_action(&pair);

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
        let mut outcome: Option<bool> = None;

        let r = 14.0;
        let d_green2 =
            (self.agent_x - self.green_x).powi(2) + (self.agent_y - self.green_y).powi(2);
        let d_red2 = (self.agent_x - self.red_x).powi(2) + (self.agent_y - self.red_y).powi(2);

        if d_green2 <= r * r {
            let ok = self.green_is_good;
            reward = if ok { 1.0 } else { -1.0 };
            outcome = Some(ok);
            self.green_x = macroquad::rand::gen_range(40.0, cfg.world_w - 40.0);
            self.green_y = macroquad::rand::gen_range(40.0, cfg.world_h - 40.0);
        } else if d_red2 <= r * r {
            let ok = !self.green_is_good;
            reward = if ok { 1.0 } else { -1.0 };
            outcome = Some(ok);
            self.red_x = macroquad::rand::gen_range(40.0, cfg.world_w - 40.0);
            self.red_y = macroquad::rand::gen_range(40.0, cfg.world_h - 40.0);
        }

        let mut just_flipped = false;
        if let Some(ok) = outcome {
            self.outcomes = self.outcomes.wrapping_add(1);
            self.score += reward;
            self.record_outcome(ok);

            // Flip rule every N outcomes.
            if self.shift_every_outcomes > 0 && (self.outcomes % self.shift_every_outcomes) == 0 {
                self.green_is_good = !self.green_is_good;
                just_flipped = true;
                self.last_flip_recent_before = self.recent_rate_n(50);
                self.last_flip_recovered_in_outcomes = None;
            }

            brain.reinforce_action(action, reward);
            brain.commit_observation();
        } else {
            brain.commit_observation();
        }

        ForageTickResult {
            ctx: ctx_name,
            action,
            reward,
            just_flipped,
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
        let step = self.tick(cfg, app.mode, brain, dt);

        if step.just_flipped {
            self.last_flip_at_frame = frame;
            logger.log_event(
                frame,
                "Forage",
                &format!(
                    "FLIP_MARKER outcomes={} recent50_before={:.3}",
                    self.outcomes, self.last_flip_recent_before
                ),
            );
        }

        // Draw arena. Cue is border color.
        let border = if self.green_is_good {
            Color::new(0.15, 0.35, 0.18, 1.0)
        } else {
            Color::new(0.35, 0.15, 0.18, 1.0)
        };
        let top = crate::TOP_UI_H;
        draw_rectangle_lines(0.0, top, cfg.world_w, cfg.world_h, 6.0, border);

        // Draw targets.
        draw_circle(self.green_x, top + self.green_y, 12.0, GREEN);
        draw_circle(self.red_x, top + self.red_y, 12.0, RED);

        // Draw agent.
        draw_circle(self.agent_x, top + self.agent_y, 10.0, WHITE);

        // HUD.
        let scored_all = brain.ranked_actions_with_meaning(step.ctx, 7.0);
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

        // Snapshot for logging/reinforcement lines.
        let snap = braine::observer::BrainAdapter::new(brain).snapshot();

        let hud: Vec<String> = vec![
            format!(
                "Forage cue={} outcomes={} flip_in={} score={:+.1}",
                if self.green_is_good { "GREEN" } else { "RED" },
                self.outcomes,
                self.flip_countdown(),
                self.score
            ),
            format!("action={} reward={:+.1}", step.action, step.reward),
            "top_actions(ema):".to_string(),
            scored_ema
                .iter()
                .take(4)
                .map(|(n, s)| format!("{}:{:+.2}", n, s))
                .collect::<Vec<_>>()
                .join("  "),
        ];

        logger.log_snapshot(
            frame,
            "Forage",
            app.mode,
            step.reward,
            &hud,
            &snap.last_reinforced_actions,
        );

        let hud_rect = Rect::new(
            crate::UI_MARGIN,
            crate::HUD_START_Y,
            screen_width() - 2.0 * crate::UI_MARGIN,
            crate::TOP_UI_H - crate::HUD_START_Y - crate::UI_MARGIN,
        );

        // Forage doesn't show reinforce lines; keep HUD compact.
        draw_hud_lines_wrapped(&hud, hud_rect, WHITE);

        // Small footer hint.
        let hint = "WASD/Arrows move (Human)";
        draw_text(hint, crate::UI_MARGIN, crate::TOP_UI_H + 18.0, 18.0, GRAY);
    }
}
