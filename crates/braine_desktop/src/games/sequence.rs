use crate::{AppState, ControlMode, MetricsLogger, VisConfig, draw_hud_lines_wrapped};
use braine::substrate::{Brain, Stimulus};
use macroquad::prelude::*;
use std::collections::HashMap;

pub fn configure_brain(brain: &mut Brain) {
    // Current token as context; reward is delayed to the next step.
    brain.ensure_sensor("seq_token_A", 4);
    brain.ensure_sensor("seq_token_B", 4);
    brain.ensure_sensor("seq_token_C", 4);

    // Regime/context so the brain can condition on which pattern is active.
    brain.ensure_sensor("seq_regime_0", 3);
    brain.ensure_sensor("seq_regime_1", 3);

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

fn token_sensor(token: char) -> &'static str {
    match token {
        'A' => "seq_token_A",
        'B' => "seq_token_B",
        _ => "seq_token_C",
    }
}

#[derive(Debug, Clone)]
pub struct SequenceUi {
    // Two simple patterns (both length 4), flipped periodically.
    pattern0: Vec<char>,
    pattern1: Vec<char>,
    use_pattern1: bool,

    idx: usize,

    // Delayed reward bookkeeping: reward previous prediction when the next token arrives.
    prev_prediction: Option<&'static str>,

    // Metrics.
    outcomes: u32,
    shift_every_outcomes: u32,
    score: f32,
    recent: Vec<bool>,
    last_flip_recent_before: f32,

    ema_action_scores: HashMap<String, f32>,
}

impl SequenceUi {
    pub fn new() -> Self {
        Self {
            pattern0: vec!['A', 'B', 'A', 'C'],
            pattern1: vec!['A', 'C', 'B', 'C'],
            use_pattern1: false,
            idx: 0,
            prev_prediction: None,
            outcomes: 0,
            shift_every_outcomes: 60,
            score: 0.0,
            recent: Vec::with_capacity(400),
            last_flip_recent_before: 0.0,
            ema_action_scores: HashMap::new(),
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

    fn current_pattern(&self) -> &Vec<char> {
        if self.use_pattern1 {
            &self.pattern1
        } else {
            &self.pattern0
        }
    }

    pub fn tick_and_render(
        &mut self,
        _cfg: &VisConfig,
        app: &AppState,
        brain: &mut Brain,
        logger: &mut MetricsLogger,
        frame: u64,
        _dt: f32,
    ) {
        let token = self.current_pattern()[self.idx % self.current_pattern().len()];
        let token_name = token_sensor(token);
        let regime = if self.use_pattern1 {
            "seq_regime_1"
        } else {
            "seq_regime_0"
        };

        // Delayed reward: compare prev prediction with current token.
        let mut reward = 0.0f32;
        if let Some(prev) = self.prev_prediction {
            let ok = match (prev, token) {
                ("A", 'A') => true,
                ("B", 'B') => true,
                ("C", 'C') => true,
                _ => false,
            };
            reward = if ok { 1.0 } else { -1.0 };
            self.outcomes = self.outcomes.wrapping_add(1);
            self.score += reward;
            self.record_outcome(ok);

            // Credit/blame the previous predicted action.
            brain.reinforce_action(prev, reward);
        }

        // Present the current token as stimulus.
        brain.apply_stimulus(Stimulus::new(regime, 0.8));
        brain.apply_stimulus(Stimulus::new(token_name, 1.0));
        brain.step();

        let action: &'static str = match app.mode {
            ControlMode::Human => human_action(),
            ControlMode::Braine => {
                // A bit of exploration.
                if macroquad::rand::gen_range(0.0, 1.0) < 0.08 {
                    match macroquad::rand::gen_range(0, 3) {
                        0 => "A",
                        1 => "B",
                        _ => "C",
                    }
                } else {
                    let (a, _s) = brain.select_action_with_meaning(token_name, 7.5);
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
        brain.note_compound_symbol(&["pair", regime, token_name, action]);

        // Save prediction for next-step reward.
        self.prev_prediction = Some(action);
        brain.commit_observation();

        // Flip pattern periodically.
        if self.shift_every_outcomes > 0
            && self.outcomes > 0
            && (self.outcomes % self.shift_every_outcomes) == 0
        {
            self.last_flip_recent_before = self.recent_rate_n(60);
            self.use_pattern1 = !self.use_pattern1;
            logger.log_event(
                frame,
                "Sequence",
                &format!(
                    "FLIP_MARKER outcomes={} recent60_before={:.3} regime={}",
                    self.outcomes,
                    self.last_flip_recent_before,
                    if self.use_pattern1 { 1 } else { 0 }
                ),
            );
        }

        // Advance token.
        self.idx = (self.idx + 1) % self.current_pattern().len();

        // Draw.
        let top = crate::TOP_UI_H;
        draw_rectangle_lines(
            0.0,
            top,
            screen_width(),
            screen_height() - top,
            4.0,
            if self.use_pattern1 {
                Color::new(0.35, 0.30, 0.12, 1.0)
            } else {
                Color::new(0.15, 0.22, 0.40, 1.0)
            },
        );

        let big = token.to_string();
        let font_size = 80.0;
        let m = measure_text(&big, None, font_size as u16, 1.0);
        draw_text(
            &big,
            (screen_width() - m.width) * 0.5,
            top + 140.0,
            font_size,
            WHITE,
        );

        // HUD / interpretability.
        let scored_all = brain.ranked_actions_with_meaning(token_name, 7.5);
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
                "Sequence regime={} outcomes={} flip_in={} score={:+.1}",
                if self.use_pattern1 { 1 } else { 0 },
                self.outcomes,
                self.flip_countdown(),
                self.score
            ),
            format!(
                "token={} predicted_next={} reward={:+.1}",
                token, action, reward
            ),
            format!("recent60={:.2}", self.recent_rate_n(60)),
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
            "Sequence",
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

        draw_text(
            "Human: 1/2/3 predict A/B/C",
            crate::UI_MARGIN,
            crate::TOP_UI_H + 18.0,
            18.0,
            GRAY,
        );
    }
}
