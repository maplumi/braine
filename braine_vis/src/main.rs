//! Braine Visualizer - Slint UI
//!
//! Interactive visualization of the brain substrate with Pong and other games.

use braine::substrate::{Brain, BrainConfig, Stimulus};
use slint::{Timer, TimerMode};
use std::cell::RefCell;
use std::rc::Rc;
use std::time::Duration;

slint::include_modules!();

// ═══════════════════════════════════════════════════════════════════════════
// Configuration
// ═══════════════════════════════════════════════════════════════════════════

const WORLD_W: f32 = 400.0;
const WORLD_H: f32 = 520.0;
const PADDLE_W: f32 = 80.0;
const PADDLE_SPEED: f32 = 400.0;
const BALL_R: f32 = 10.0;
const BALL_SPEED_Y: f32 = 200.0;
const CATCH_MARGIN: f32 = 12.0;
const PADDLE_Y: f32 = WORLD_H - 30.0;

// ═══════════════════════════════════════════════════════════════════════════
// Game state
// ═══════════════════════════════════════════════════════════════════════════

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ControlMode {
    Human,
    Braine,
}

#[derive(Debug)]
struct PongGame {
    ball_x: f32,
    ball_y: f32,
    ball_vx: f32,
    ball_vy: f32,
    #[allow(dead_code)]
    decoy_x: f32,
    #[allow(dead_code)]
    decoy_y: f32,
    #[allow(dead_code)]
    decoy_vx: f32,
    #[allow(dead_code)]
    decoy_vy: f32,
    paddle_x: f32,
    
    hits: u32,
    misses: u32,
    score: f32,
    recent: Vec<bool>,
    
    rng_seed: u64,
    
    sensor_axis_flipped: bool,
    shift_every_outcomes: u32,
    
    misbucket_p: f32,
    rel_jitter: f32,
    explore_p: f32,
    decoy_enabled: bool,
    difficulty_level: u8,
}

impl PongGame {
    fn new() -> Self {
        let mut game = Self {
            ball_x: WORLD_W / 2.0,
            ball_y: 40.0,
            ball_vx: 0.0,
            ball_vy: BALL_SPEED_Y,
            decoy_x: WORLD_W / 3.0,
            decoy_y: 60.0,
            decoy_vx: 50.0,
            decoy_vy: BALL_SPEED_Y * 0.92,
            paddle_x: WORLD_W / 2.0,
            
            hits: 0,
            misses: 0,
            score: 0.0,
            recent: Vec::new(),
            
            rng_seed: 12345,
            
            sensor_axis_flipped: false,
            shift_every_outcomes: 160,
            
            misbucket_p: 0.0,
            rel_jitter: 0.0,
            explore_p: 0.12,
            decoy_enabled: false,
            difficulty_level: 0,
        };
        game.apply_difficulty();
        game.reset_ball();
        game
    }
    
    fn apply_difficulty(&mut self) {
        match self.difficulty_level {
            0 => {
                // Tutorial
                self.misbucket_p = 0.0;
                self.rel_jitter = 0.0;
                self.decoy_enabled = false;
                self.explore_p = 0.12;
                self.shift_every_outcomes = 9999;
            }
            1 => {
                // Easy
                self.misbucket_p = 0.0;
                self.rel_jitter = 0.0;
                self.decoy_enabled = false;
                self.explore_p = 0.10;
                self.shift_every_outcomes = self.shift_every_outcomes.max(240);
            }
            2 => {
                // Medium
                self.misbucket_p = 0.04;
                self.rel_jitter = 10.0;
                self.decoy_enabled = false;
                self.explore_p = 0.08;
                self.shift_every_outcomes = self.shift_every_outcomes.max(200);
            }
            3 => {
                // Hard
                self.misbucket_p = 0.08;
                self.rel_jitter = 18.0;
                self.decoy_enabled = true;
                self.explore_p = 0.06;
                self.shift_every_outcomes = self.shift_every_outcomes.max(160);
            }
            _ => {
                // Expert
                self.misbucket_p = 0.12;
                self.rel_jitter = 26.0;
                self.decoy_enabled = true;
                self.explore_p = 0.04;
                self.shift_every_outcomes = self.shift_every_outcomes.max(120);
            }
        }
    }
    
    fn difficulty_name(&self) -> &'static str {
        match self.difficulty_level {
            0 => "Tutorial",
            1 => "Easy",
            2 => "Medium",
            3 => "Hard",
            _ => "Expert",
        }
    }
    
    fn easier(&mut self) {
        if self.difficulty_level > 0 {
            self.difficulty_level -= 1;
            self.apply_difficulty();
        }
    }
    
    fn harder(&mut self) {
        if self.difficulty_level < 4 {
            self.difficulty_level += 1;
            self.apply_difficulty();
        }
    }
    
    fn flip_slower(&mut self) {
        self.shift_every_outcomes = (self.shift_every_outcomes + 40).clamp(40, 400);
    }
    
    fn flip_faster(&mut self) {
        self.shift_every_outcomes = self.shift_every_outcomes.saturating_sub(40).clamp(40, 400);
    }
    
    fn outcomes(&self) -> u32 {
        self.hits + self.misses
    }
    
    fn flip_countdown(&self) -> u32 {
        let shift = self.shift_every_outcomes.max(1);
        let o = self.outcomes();
        if o == 0 { shift } else { shift - (o % shift) }
    }
    
    fn recent_rate(&self) -> f32 {
        if self.recent.is_empty() { return 0.5; }
        let good = self.recent.iter().filter(|&&b| b).count();
        good as f32 / self.recent.len() as f32
    }
    
    fn reset_ball(&mut self) {
        self.rng_seed = self.rng_seed.wrapping_mul(6364136223846793005).wrapping_add(1);
        let u = (self.rng_seed >> 11) as u32;
        let x01 = (u as f32) / (u32::MAX as f32);
        
        if self.difficulty_level == 0 {
            // Tutorial: ball spawns above paddle
            let offset = (x01 - 0.5) * 60.0;
            self.ball_x = (self.paddle_x + offset).clamp(BALL_R, WORLD_W - BALL_R);
            self.ball_vx = 0.0;
        } else {
            self.ball_x = BALL_R + x01 * (WORLD_W - 2.0 * BALL_R);
            self.rng_seed = self.rng_seed.wrapping_mul(6364136223846793005).wrapping_add(1);
            let u2 = (self.rng_seed >> 11) as u32;
            let s01 = (u2 as f32) / (u32::MAX as f32);
            let drift = match self.difficulty_level {
                1 => 90.0,
                2 => 120.0,
                3 => 140.0,
                _ => 170.0,
            };
            self.ball_vx = (s01 * 2.0 - 1.0) * drift;
        }
        self.ball_y = 40.0;
        self.ball_vy = BALL_SPEED_Y;
    }
    
    fn bucket_index(&self, rel: f32) -> usize {
        if rel <= -120.0 { 0 }
        else if rel <= -30.0 { 1 }
        else if rel.abs() <= 18.0 { 2 }
        else if rel >= 120.0 { 4 }
        else { 3 }
    }
    
    fn ctx_name(idx: usize) -> &'static str {
        match idx {
            0 => "pong_ctx_far_left",
            1 => "pong_ctx_left",
            2 => "pong_ctx_aligned",
            3 => "pong_ctx_right",
            _ => "pong_ctx_far_right",
        }
    }
    
    fn tick(&mut self, mode: ControlMode, brain: &mut Brain, dt: f32) -> (f32, bool) {
        // Sensor mapping
        let rel_raw = self.ball_x - self.paddle_x;
        let rel_seen = if self.sensor_axis_flipped { -rel_raw } else { rel_raw };
        let ctx_idx = self.bucket_index(rel_seen);
        let ctx = Self::ctx_name(ctx_idx);
        
        // Stimulus
        brain.apply_stimulus(Stimulus::new(ctx, 1.0));
        brain.step();
        
        // Action
        let action: &'static str = match mode {
            ControlMode::Human => "stay", // Keyboard not available in Slint yet
            ControlMode::Braine => {
                let (a, _) = brain.select_action_with_meaning(ctx, 6.0);
                if a == "left" { "left" }
                else if a == "right" { "right" }
                else { "stay" }
            }
        };
        
        brain.note_action(action);
        let pair = format!("pair::{ctx}::{action}");
        brain.note_action(&pair);
        
        // Move paddle
        let ax = match action {
            "left" => -1.0,
            "right" => 1.0,
            _ => 0.0,
        };
        self.paddle_x += ax * PADDLE_SPEED * dt;
        self.paddle_x = self.paddle_x.clamp(PADDLE_W / 2.0, WORLD_W - PADDLE_W / 2.0);
        
        // Ball physics
        self.ball_x += self.ball_vx * dt;
        self.ball_y += self.ball_vy * dt;
        
        // Bounce off walls
        if self.ball_x <= BALL_R { self.ball_vx = self.ball_vx.abs(); }
        if self.ball_x >= WORLD_W - BALL_R { self.ball_vx = -self.ball_vx.abs(); }
        if self.ball_y <= BALL_R { self.ball_vy = self.ball_vy.abs(); }
        
        // Check paddle collision
        let mut reward = 0.0f32;
        let mut just_flipped = false;
        
        let half = PADDLE_W / 2.0;
        let margin = if self.difficulty_level == 0 { 25.0 } else { CATCH_MARGIN };
        let hit_left = self.paddle_x - half - margin;
        let hit_right = self.paddle_x + half + margin;
        
        if self.ball_vy > 0.0 && self.ball_y + BALL_R >= PADDLE_Y {
            if self.ball_x >= hit_left && self.ball_x <= hit_right {
                // Hit!
                self.hits += 1;
                self.score += 1.0;
                reward = 1.0;
                self.recent.push(true);
                self.ball_vy = -self.ball_vy.abs();
                
                // Reinforce
                brain.set_neuromodulator(0.5);
                brain.reinforce_action(action, 0.08);
                brain.reinforce_action(&pair, 0.15);
            } else if self.ball_y > WORLD_H {
                // Miss
                self.misses += 1;
                self.score -= 0.5;
                reward = -1.0;
                self.recent.push(false);
                self.reset_ball();
                
                // Punish
                brain.set_neuromodulator(-0.3);
                brain.reinforce_action(action, -0.06);
                brain.reinforce_action(&pair, -0.10);
            }
            
            // Check for flip
            let outcomes_now = self.outcomes();
            if self.shift_every_outcomes < 9999 && outcomes_now > 0 && outcomes_now % self.shift_every_outcomes == 0 {
                self.sensor_axis_flipped = !self.sensor_axis_flipped;
                just_flipped = true;
            }
        }
        
        if self.recent.len() > 400 {
            self.recent.drain(0..100);
        }
        
        brain.commit_observation();
        
        (reward, just_flipped)
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Application state
// ═══════════════════════════════════════════════════════════════════════════

struct AppState {
    brain: Brain,
    pong: PongGame,
    mode: ControlMode,
    frame: u64,
    last_pred_ctx: String,
    last_pred_bonus: f32,
}

impl AppState {
    fn new() -> Self {
        let mut brain = Brain::new(BrainConfig {
            unit_count: 160,
            connectivity_per_unit: 8,
            dt: 0.05,
            base_freq: 1.0,
            noise_amp: 0.015,
            noise_phase: 0.008,
            global_inhibition: 0.07,
            hebb_rate: 0.09,
            forget_rate: 0.0015,
            prune_below: 0.0008,
            coactive_threshold: 0.55,
            phase_lock_threshold: 0.6,
            imprint_rate: 0.6,
            seed: Some(123),
            causal_decay: 0.01,
        });
        
        // Register Pong symbols
        brain.define_sensor("pong_ctx_far_left", 4);
        brain.define_sensor("pong_ctx_left", 4);
        brain.define_sensor("pong_ctx_aligned", 4);
        brain.define_sensor("pong_ctx_right", 4);
        brain.define_sensor("pong_ctx_far_right", 4);
        brain.define_action("left", 6);
        brain.define_action("right", 6);
        brain.define_action("stay", 4);
        
        brain.set_observer_telemetry(true);
        
        Self {
            brain,
            pong: PongGame::new(),
            mode: ControlMode::Braine,
            frame: 0,
            last_pred_ctx: String::new(),
            last_pred_bonus: 0.0,
        }
    }
    
    fn tick(&mut self, dt: f32) {
        self.frame += 1;
        let (_reward, _flipped) = self.pong.tick(self.mode, &mut self.brain, dt);
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Main
// ═══════════════════════════════════════════════════════════════════════════

fn main() -> Result<(), slint::PlatformError> {
    let ui = MainWindow::new()?;
    let state = Rc::new(RefCell::new(AppState::new()));
    
    // Initialize UI state
    ui.set_learning(LearningState {
        learning_enabled: true,
        attention_boost: 0.5,
        attention_enabled: false,
        burst_mode_enabled: false,
        auto_dream_on_flip: true,
        auto_burst_on_slump: true,
        prediction_enabled: true,
        prediction_weight: 2.0,
    });
    
    // Wire up callbacks
    {
        let state_clone = state.clone();
        ui.on_mode_changed(move |is_braine| {
            let mut s = state_clone.borrow_mut();
            s.mode = if is_braine { ControlMode::Braine } else { ControlMode::Human };
        });
    }
    
    {
        let state_clone = state.clone();
        ui.on_difficulty_up(move || {
            state_clone.borrow_mut().pong.harder();
        });
    }
    
    {
        let state_clone = state.clone();
        ui.on_difficulty_down(move || {
            state_clone.borrow_mut().pong.easier();
        });
    }
    
    {
        let state_clone = state.clone();
        ui.on_flip_faster(move || {
            state_clone.borrow_mut().pong.flip_faster();
        });
    }
    
    {
        let state_clone = state.clone();
        ui.on_flip_slower(move || {
            state_clone.borrow_mut().pong.flip_slower();
        });
    }
    
    {
        let state_clone = state.clone();
        ui.on_reset_brain(move || {
            let mut s = state_clone.borrow_mut();
            *s = AppState::new();
        });
    }
    
    // Game loop timer (~60 FPS)
    let timer = Timer::default();
    let ui_weak = ui.as_weak();
    let state_clone = state.clone();
    
    timer.start(TimerMode::Repeated, Duration::from_millis(16), move || {
        let ui = ui_weak.unwrap();
        
        if ui.get_paused() {
            return;
        }
        
        let mut s = state_clone.borrow_mut();
        let dt = 0.016;
        s.tick(dt);
        
        // Update Pong state for rendering
        ui.set_pong(PongState {
            ball_x: s.pong.ball_x,
            ball_y: s.pong.ball_y,
            paddle_x: s.pong.paddle_x,
            decoy_x: s.pong.decoy_x,
            decoy_y: s.pong.decoy_y,
            decoy_enabled: s.pong.decoy_enabled,
        });
        
        // Update HUD
        ui.set_hud(HudData {
            frame: s.frame as i32,
            hits: s.pong.hits as i32,
            misses: s.pong.misses as i32,
            score: s.pong.score,
            recent_rate: s.pong.recent_rate(),
            difficulty: s.pong.difficulty_level as i32,
            difficulty_name: s.pong.difficulty_name().into(),
            flip_countdown: s.pong.flip_countdown() as i32,
            sensor_flipped: s.pong.sensor_axis_flipped,
            neuromod: s.brain.neuromodulator(),
            explore_p: s.pong.explore_p,
            decoy_enabled: s.pong.decoy_enabled,
            prediction_enabled: ui.get_learning().prediction_enabled,
            pred_ctx: s.last_pred_ctx.clone().into(),
            pred_bonus: s.last_pred_bonus,
        });
        
        // Update brain stats
        let diag = s.brain.diagnostics();
        let causal = s.brain.causal_stats();
        ui.set_brain_stats(BrainStats {
            unit_count: diag.unit_count as i32,
            connection_count: diag.connection_count as i32,
            avg_amp: diag.avg_amp,
            avg_weight: diag.avg_weight,
            memory_bytes: diag.memory_bytes as i32,
            causal_edges: causal.edges as i32,
            age_steps: s.brain.age_steps() as i32,
        });
    });
    
    ui.run()
}
