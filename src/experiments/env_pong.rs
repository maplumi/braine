use braine::prng::Prng;
use braine::substrate::{Brain, Stimulus};
use std::io;
use std::io::Write;

#[derive(Debug, Clone, Copy)]
pub struct PongConfig {
    pub width: i32,
    pub paddle_y: i32,
    pub max_steps: usize,
    pub render_every: usize,
    pub retire_window: usize,
    pub retire_hit_rate: f32,
}

impl Default for PongConfig {
    fn default() -> Self {
        Self {
            width: 21,
            paddle_y: 0,
            max_steps: 20_000,
            render_every: 200,
            retire_window: 400,
            retire_hit_rate: 0.92,
        }
    }
}

#[derive(Debug, Clone)]
struct PongState {
    ball_x: i32,
    ball_y: i32,
    vel_y: i32,
    paddle_x: i32,
    steps: usize,
    hits: usize,
    misses: usize,
    cumulative_reward: f32,
    rng: Prng,
}

impl PongState {
    fn new(width: i32, seed: u64) -> Self {
        let mid = width / 2;
        Self {
            ball_x: mid,
            ball_y: 10,
            vel_y: -1,
            paddle_x: mid,
            steps: 0,
            hits: 0,
            misses: 0,
            cumulative_reward: 0.0,
            rng: Prng::new(seed),
        }
    }

    fn reset_ball(&mut self, width: i32) {
        self.ball_x = self.rng.gen_range_usize(0, width as usize) as i32;
        self.ball_y = 10;
        self.vel_y = -1;
    }
}

pub fn run_pong_demo(brain: &mut Brain, cfg: PongConfig) {
    brain.ensure_sensor("pong_ctx_far_left", 4);
    brain.ensure_sensor("pong_ctx_left", 4);
    brain.ensure_sensor("pong_ctx_aligned", 4);
    brain.ensure_sensor("pong_ctx_right", 4);
    brain.ensure_sensor("pong_ctx_far_right", 4);
    brain.ensure_sensor("pong_ball_falling", 3);

    brain.define_action("left", 6);
    brain.define_action("right", 6);
    brain.define_action("stay", 6);

    let epsilon = 0.12;
    let mut s = PongState::new(cfg.width, 2026);
    let bootstrap_steps: usize = 600;
    let mut recent: Vec<bool> = Vec::with_capacity(cfg.retire_window.max(1));

    if out_line(format_args!(
        "pong-demo: width={}, max_steps={}, retire_hit_rate={:.2} (window={})",
        cfg.width, cfg.max_steps, cfg.retire_hit_rate, cfg.retire_window
    ))
    .is_err()
    {
        return;
    }

    while s.steps < cfg.max_steps {
        s.steps += 1;

        let rel = s.ball_x - s.paddle_x;
        let ctx = if rel <= -4 {
            "pong_ctx_far_left"
        } else if rel <= -1 {
            "pong_ctx_left"
        } else if rel == 0 {
            "pong_ctx_aligned"
        } else if rel >= 4 {
            "pong_ctx_far_right"
        } else {
            "pong_ctx_right"
        };

        brain.apply_stimulus(Stimulus::new(ctx, 1.0));
        if s.vel_y < 0 {
            brain.apply_stimulus(Stimulus::new("pong_ball_falling", 0.9));
        }

        brain.step();

        let (action, _score) = if s.steps <= bootstrap_steps {
            if s.steps == bootstrap_steps
                && out_line(format_args!(
                    "switching to autonomous control (bootstrap complete)"
                ))
                .is_err()
            {
                return;
            }
            let a = if rel < 0 {
                "left"
            } else if rel > 0 {
                "right"
            } else {
                "stay"
            };
            (a.to_string(), 0.0)
        } else if s.rng.gen_range_f32(0.0, 1.0) < epsilon {
            let idx = s.rng.gen_range_usize(0, 3);
            match idx {
                0 => ("left".to_string(), 0.0),
                1 => ("right".to_string(), 0.0),
                _ => ("stay".to_string(), 0.0),
            }
        } else {
            brain.select_action_with_meaning(ctx, 6.0)
        };

        brain.note_action(&action);
        let pair = format!("pair::{ctx}::{action}");
        brain.note_action(&pair);

        match action.as_str() {
            "left" => s.paddle_x -= 1,
            "right" => s.paddle_x += 1,
            _ => {}
        }
        s.paddle_x = s.paddle_x.clamp(0, cfg.width - 1);

        s.ball_y += s.vel_y;

        let mut reward = 0.0f32;
        if s.vel_y < 0 {
            let dist = (s.ball_x - s.paddle_x).abs() as f32;
            reward += (-0.008 * dist).clamp(-0.16, 0.0);
            if dist <= 1.0 {
                reward += 0.01;
            }
            reward += match (rel, action.as_str()) {
                (r, "left") if r < 0 => 0.015,
                (r, "right") if r > 0 => 0.015,
                (0, "stay") => 0.015,
                _ => -0.005,
            };
        }

        let mut outcome: Option<bool> = None;
        if s.ball_y <= cfg.paddle_y {
            if (s.ball_x - s.paddle_x).abs() <= 1 {
                s.hits += 1;
                reward += 0.7;
                outcome = Some(true);
                s.reset_ball(cfg.width);
            } else {
                s.misses += 1;
                reward += -0.7;
                outcome = Some(false);
                s.reset_ball(cfg.width);
            }
        }

        reward = reward.clamp(-1.0, 1.0);
        s.cumulative_reward += reward;

        brain.set_neuromodulator(reward);
        if reward > 0.2 {
            brain.reinforce_action(&action, 0.6);
        }
        brain.commit_observation();

        if let Some(ok) = outcome {
            recent.push(ok);
            if recent.len() > cfg.retire_window {
                recent.remove(0);
            }
        }

        if cfg.render_every > 0 && s.steps.is_multiple_of(cfg.render_every) {
            let hit_rate = if s.hits + s.misses == 0 {
                0.0
            } else {
                s.hits as f32 / (s.hits + s.misses) as f32
            };
            let recent_rate = if recent.is_empty() {
                0.0
            } else {
                recent.iter().filter(|&&b| b).count() as f32 / recent.len() as f32
            };

            if out_line(format_args!(
                "age_steps={} hits={} misses={} hit_rate={:.3} recent={:.3} cumulative_reward={:.1}",
                s.steps, s.hits, s.misses, hit_rate, recent_rate, s.cumulative_reward
            ))
            .is_err()
            {
                return;
            }

            let hint = brain.meaning_hint(ctx);
            if out_line(format_args!(
                "  wisdom_probe: ctx={} meaning_hint={:?}",
                ctx, hint
            ))
            .is_err()
            {
                return;
            }
        }

        if recent.len() >= (cfg.retire_window / 2).max(1) {
            let recent_rate = recent.iter().filter(|&&b| b).count() as f32 / recent.len() as f32;
            if recent_rate >= cfg.retire_hit_rate {
                if out_line(format_args!(
                    "self-retire: recent_hit_rate={:.3} over {} outcomes (age_steps={})",
                    recent_rate,
                    recent.len(),
                    s.steps
                ))
                .is_err()
                {
                    return;
                }
                break;
            }
        }
    }

    let final_hit_rate = if s.hits + s.misses == 0 {
        0.0
    } else {
        s.hits as f32 / (s.hits + s.misses) as f32
    };
    if out_line(format_args!("pong-demo done:")).is_err() {
        return;
    }
    if out_line(format_args!("  age_steps={}", s.steps)).is_err() {
        return;
    }
    if out_line(format_args!(
        "  hits={} misses={} hit_rate={:.3}",
        s.hits, s.misses, final_hit_rate
    ))
    .is_err()
    {
        return;
    }
    let _ = out_line(format_args!(
        "  cumulative_reward={:.1}",
        s.cumulative_reward
    ));
}

fn out_line(args: std::fmt::Arguments<'_>) -> io::Result<()> {
    let mut out = io::stdout().lock();
    match out.write_fmt(args) {
        Ok(()) => {}
        Err(e) if e.kind() == io::ErrorKind::BrokenPipe => return Err(e),
        Err(e) => return Err(e),
    }
    match out.write_all(b"\n") {
        Ok(()) => Ok(()),
        Err(e) if e.kind() == io::ErrorKind::BrokenPipe => Err(e),
        Err(e) => Err(e),
    }
}
