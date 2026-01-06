use braine::observer::BrainAdapter;
use braine::storage;
use braine::substrate::{Brain, BrainConfig, Stimulus};
use braine::supervisor::{ChildConfigOverrides, ConsolidationPolicy};
use macroquad::prelude::*;
use std::collections::HashMap;
use std::env;
use std::fs;
use std::fs::File;
use std::io::{self, BufReader, BufWriter, Write};
use std::path::{Path, PathBuf};
use std::sync::mpsc;
use std::thread;

mod games;

// Space reserved for buttons + HUD text (so it doesn't overlap the game area).
const TOP_UI_H: f32 = 220.0;
const UI_MARGIN: f32 = 12.0;
const UI_GAP: f32 = 6.0;
const BTN_H: f32 = 28.0;
const BTN_W_TEST: f32 = 100.0;
const BTN_W_MODE: f32 = 110.0;
const BTN_W_SMALL: f32 = 120.0;
const BTN_FONT_SIZE: f32 = 18.0;
// Start HUD below the 3 header rows.
const HUD_START_Y: f32 = 124.0;

const HUD_FONT_SIZE: u16 = 18;
const HUD_LINE_H: f32 = 20.0;

#[derive(Debug, Clone)]
struct StorageConfig {
    brain_image_path: PathBuf,
    capacity_bytes: Option<usize>,
    autosave_every_secs: Option<f64>,
}

impl StorageConfig {
    fn from_env_and_args() -> Self {
        // Defaults: keep it simple and local.
        let mut brain_dir: Option<PathBuf> = env::var("BRAINE_BRAIN_DIR").ok().map(PathBuf::from);
        let mut brain_image_path: Option<PathBuf> =
            env::var("BRAINE_BRAIN_IMAGE").ok().map(PathBuf::from);
        let mut capacity_bytes: Option<usize> = env::var("BRAINE_BRAIN_CAP_BYTES")
            .ok()
            .and_then(|s| s.parse::<usize>().ok());

        // Default autosave: enabled. Can be overridden by env/args.
        // Set to 0 to disable.
        let mut autosave_every_secs: Option<f64> = Some(2.0);
        if let Ok(v) = env::var("BRAINE_AUTOSAVE_SECS") {
            if let Ok(n) = v.parse::<f64>() {
                autosave_every_secs = if n > 0.0 { Some(n) } else { None };
            }
        }

        let mut args = env::args().skip(1);
        while let Some(a) = args.next() {
            match a.as_str() {
                "--brain-dir" => {
                    if let Some(v) = args.next() {
                        brain_dir = Some(PathBuf::from(v));
                    }
                }
                "--brain-image" => {
                    if let Some(v) = args.next() {
                        brain_image_path = Some(PathBuf::from(v));
                    }
                }
                "--brain-cap" => {
                    if let Some(v) = args.next() {
                        capacity_bytes = v.parse::<usize>().ok();
                    }
                }
                "--autosave" => {
                    if let Some(v) = args.next() {
                        autosave_every_secs = v
                            .parse::<f64>()
                            .ok()
                            .and_then(|n| if n > 0.0 { Some(n) } else { None });
                    }
                }
                _ => {}
            }
        }

        let dir = brain_dir.unwrap_or_else(|| PathBuf::from("data"));
        let brain_image_path = brain_image_path.unwrap_or_else(|| dir.join("brain.bbi"));

        Self {
            brain_image_path,
            capacity_bytes,
            autosave_every_secs,
        }
    }

    fn ensure_parent_dir(&self) -> io::Result<()> {
        if let Some(parent) = self.brain_image_path.parent() {
            fs::create_dir_all(parent)?;
        }
        Ok(())
    }
}

fn encode_brain_image(cap: Option<usize>, brain: &Brain) -> io::Result<Vec<u8>> {
    let mut buf: Vec<u8> = Vec::new();
    if let Some(cap) = cap {
        let mut cw = storage::CapacityWriter::new(&mut buf, cap);
        brain.save_image_to(&mut cw)?;
    } else {
        brain.save_image_to(&mut buf)?;
    }
    Ok(buf)
}

fn write_brain_image_bytes(path: &Path, bytes: &[u8]) -> io::Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let tmp = path.with_extension("bbi.tmp");
    let mut f = File::create(&tmp)?;
    f.write_all(bytes)?;
    f.flush()?;
    drop(f);
    fs::rename(&tmp, path)?;
    Ok(())
}

struct AsyncSaver {
    tx: mpsc::SyncSender<Vec<u8>>,
    rx_done: mpsc::Receiver<io::Result<usize>>,
}

impl AsyncSaver {
    fn new(path: PathBuf) -> Self {
        let (tx, rx) = mpsc::sync_channel::<Vec<u8>>(1);
        let (tx_done, rx_done) = mpsc::channel::<io::Result<usize>>();

        thread::spawn(move || {
            while let Ok(bytes) = rx.recv() {
                let res = write_brain_image_bytes(&path, &bytes).map(|_| bytes.len());
                let _ = tx_done.send(res);
            }
        });

        Self { tx, rx_done }
    }

    fn try_enqueue(&self, bytes: Vec<u8>) -> Result<(), Vec<u8>> {
        self.tx.try_send(bytes).map_err(|e| match e {
            mpsc::TrySendError::Full(v) => v,
            mpsc::TrySendError::Disconnected(v) => v,
        })
    }

    fn poll_done(&self) -> Option<io::Result<usize>> {
        self.rx_done.try_recv().ok()
    }
}

struct AsyncLoader {
    tx: mpsc::SyncSender<()>,
    rx_done: mpsc::Receiver<io::Result<Brain>>,
}

impl AsyncLoader {
    fn new(path: PathBuf) -> Self {
        let (tx, rx) = mpsc::sync_channel::<()>(1);
        let (tx_done, rx_done) = mpsc::channel::<io::Result<Brain>>();

        thread::spawn(move || {
            while rx.recv().is_ok() {
                let res = load_brain_image(&path);
                let _ = tx_done.send(res);
            }
        });

        Self { tx, rx_done }
    }

    fn try_enqueue(&self) -> Result<(), ()> {
        self.tx.try_send(()).map_err(|_| ())
    }

    fn poll_done(&self) -> Option<io::Result<Brain>> {
        self.rx_done.try_recv().ok()
    }
}

fn load_brain_image(path: &Path) -> io::Result<Brain> {
    let f = File::open(path)?;
    let mut br = BufReader::new(f);
    Brain::load_image_from(&mut br)
}

fn draw_panel(rect: Rect) {
    // Subtle panel background for readability.
    draw_rectangle(
        rect.x,
        rect.y,
        rect.w,
        rect.h,
        Color::new(0.10, 0.10, 0.12, 0.88),
    );
    draw_rectangle_lines(
        rect.x,
        rect.y,
        rect.w,
        rect.h,
        1.0,
        Color::new(0.28, 0.28, 0.30, 1.0),
    );
}

fn wrap_text_to_width(text: &str, max_w: f32, font_size: u16) -> Vec<String> {
    // Word-wrap first; if a single "word" is too wide, fall back to char splitting.
    let mut out: Vec<String> = Vec::new();
    let mut cur = String::new();

    fn push_trimmed(out: &mut Vec<String>, s: &str) {
        let t = s.trim();
        if !t.is_empty() {
            out.push(t.to_string());
        }
    }

    for word in text.split_whitespace() {
        let cand = if cur.is_empty() {
            word.to_string()
        } else {
            format!("{cur} {word}")
        };

        if measure_text(&cand, None, font_size, 1.0).width <= max_w {
            cur = cand;
            continue;
        }

        if !cur.is_empty() {
            push_trimmed(&mut out, &cur);
            cur.clear();
        }

        if measure_text(word, None, font_size, 1.0).width <= max_w {
            cur.push_str(word);
            continue;
        }

        // Split very-long tokens by chars.
        let mut chunk = String::new();
        for ch in word.chars() {
            let cand2 = format!("{chunk}{ch}");
            if measure_text(&cand2, None, font_size, 1.0).width <= max_w {
                chunk = cand2;
            } else {
                push_trimmed(&mut out, &chunk);
                chunk.clear();
                chunk.push(ch);
            }
        }
        push_trimmed(&mut out, &chunk);
    }

    push_trimmed(&mut out, &cur);
    out
}

fn draw_hud_lines_wrapped(lines: &[String], rect: Rect, base_color: Color) {
    draw_panel(rect);

    let max_w = (rect.w - 2.0 * UI_GAP).max(1.0);
    let mut y = rect.y + UI_GAP + (HUD_FONT_SIZE as f32);

    for line in lines {
        for seg in wrap_text_to_width(line, max_w, HUD_FONT_SIZE) {
            if y > rect.y + rect.h - UI_GAP {
                return;
            }
            draw_text(&seg, rect.x + UI_GAP, y, HUD_FONT_SIZE as f32, base_color);
            y += HUD_LINE_H;
        }
    }
}

struct MetricsLogger {
    writer: Option<BufWriter<File>>,
    last_periodic_log: u64,
    last_flush_t: f64,
}

impl MetricsLogger {
    fn new(path: &str) -> Self {
        let writer = File::create(path).ok().map(BufWriter::new);
        Self {
            writer,
            last_periodic_log: 0,
            last_flush_t: 0.0,
        }
    }

    fn log_snapshot(
        &mut self,
        frame: u64,
        test: &str,
        mode: ControlMode,
        reward: f32,
        hud_lines: &[String],
        reinforced: &[(String, f32)],
    ) {
        let Some(w) = self.writer.as_mut() else {
            return;
        };

        // Keep logs readable and not enormous:
        // - Always log on terminal reward events.
        // - Always log when reinforcement happened.
        // - Otherwise, log periodically.
        let periodic_every = 30;
        let should_periodic = frame.saturating_sub(self.last_periodic_log) >= periodic_every;
        let should_log = reward.abs() > 0.01 || !reinforced.is_empty() || should_periodic;
        if !should_log {
            return;
        }
        if should_periodic {
            self.last_periodic_log = frame;
        }

        let t = get_time();
        let _ = writeln!(
            w,
            "t={:.3} frame={} test={} mode={:?} reward={:+.3}",
            t, frame, test, mode, reward
        );
        for line in hud_lines {
            let _ = writeln!(w, "  {line}");
        }
        if !reinforced.is_empty() {
            let _ = writeln!(w, "  reinforce:");
            for (name, delta) in reinforced.iter().take(8) {
                let _ = writeln!(w, "    {} {:+.2}", name, delta);
            }
        }
        let _ = writeln!(w, "---");

        if t - self.last_flush_t >= 1.0 {
            let _ = w.flush();
            self.last_flush_t = t;
        }
    }

    fn log_event(&mut self, frame: u64, test: &str, msg: &str) {
        let Some(w) = self.writer.as_mut() else {
            return;
        };

        let t = get_time();
        let _ = writeln!(w, "t={:.3} frame={} test={} EVENT {}", t, frame, test, msg);
        let _ = writeln!(w, "---");
    }
}

fn window_conf() -> Conf {
    Conf {
        window_title: "braine_viz".to_owned(),
        window_width: 800,
        window_height: (520.0 + TOP_UI_H) as i32,
        ..Default::default()
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ControlMode {
    Human,
    Braine,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum UiTestId {
    Pong,
    Bandit,
    Forage,
    Whack,
    Beacon,
    Sequence,
}

#[derive(Clone, Copy, Debug)]
struct VisConfig {
    world_w: f32,
    world_h: f32,

    paddle_w: f32,
    paddle_h: f32,
    paddle_y: f32,
    paddle_speed: f32,

    ball_r: f32,
    ball_speed_y: f32,

    catch_margin: f32,
}

impl Default for VisConfig {
    fn default() -> Self {
        Self {
            world_w: 800.0,
            world_h: 520.0,
            paddle_w: 80.0,
            paddle_h: 14.0,
            paddle_y: 480.0,
            paddle_speed: 520.0,
            ball_r: 10.0,
            ball_speed_y: 260.0,
            catch_margin: 12.0,
        }
    }
}

#[derive(Debug, Clone)]
struct AppState {
    mode: ControlMode,
    test: UiTestId,
}

impl Default for AppState {
    fn default() -> Self {
        Self {
            mode: ControlMode::Human,
            test: UiTestId::Pong,
        }
    }
}

fn mk_brain_for(test: UiTestId) -> Brain {
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

    match test {
        UiTestId::Pong => {
            brain.ensure_sensor("pong_ctx_far_left", 4);
            brain.ensure_sensor("pong_ctx_left", 4);
            brain.ensure_sensor("pong_ctx_aligned", 4);
            brain.ensure_sensor("pong_ctx_right", 4);
            brain.ensure_sensor("pong_ctx_far_right", 4);
            brain.ensure_sensor("pong_ball_falling", 3);

            // Extra distractor sensors (decoy ball).
            brain.ensure_sensor("pong_decoy_ctx_far_left", 3);
            brain.ensure_sensor("pong_decoy_ctx_left", 3);
            brain.ensure_sensor("pong_decoy_ctx_aligned", 3);
            brain.ensure_sensor("pong_decoy_ctx_right", 3);
            brain.ensure_sensor("pong_decoy_ctx_far_right", 3);
            brain.ensure_sensor("pong_decoy_falling", 2);

            brain.define_action("left", 6);
            brain.define_action("right", 6);
            brain.define_action("stay", 6);
        }
        UiTestId::Bandit => {
            brain.ensure_sensor("bandit_ctx", 4);
            brain.define_action("A", 6);
            brain.define_action("B", 6);
        }
        UiTestId::Forage => {
            games::forage::configure_brain(&mut brain);
        }
        UiTestId::Whack => {
            games::whack::configure_brain(&mut brain);
        }
        UiTestId::Beacon => {
            games::beacon::configure_brain(&mut brain);
        }
        UiTestId::Sequence => {
            games::sequence::configure_brain(&mut brain);
        }
    }

    brain.set_observer_telemetry(true);
    brain
}

fn bucket_index_from_rel(rel: f32) -> usize {
    if rel <= -120.0 {
        0
    } else if rel <= -30.0 {
        1
    } else if rel.abs() <= 18.0 {
        2
    } else if rel >= 120.0 {
        4
    } else {
        3
    }
}

fn pong_ctx_name(idx: usize) -> &'static str {
    match idx {
        0 => "pong_ctx_far_left",
        1 => "pong_ctx_left",
        2 => "pong_ctx_aligned",
        3 => "pong_ctx_right",
        _ => "pong_ctx_far_right",
    }
}

fn pong_decoy_ctx_name(idx: usize) -> &'static str {
    match idx {
        0 => "pong_decoy_ctx_far_left",
        1 => "pong_decoy_ctx_left",
        2 => "pong_decoy_ctx_aligned",
        3 => "pong_decoy_ctx_right",
        _ => "pong_decoy_ctx_far_right",
    }
}

fn pong_action_from_human() -> &'static str {
    let left = is_key_down(KeyCode::Left) || is_key_down(KeyCode::A);
    let right = is_key_down(KeyCode::Right) || is_key_down(KeyCode::D);

    if left && !right {
        "left"
    } else if right && !left {
        "right"
    } else {
        "stay"
    }
}

fn bandit_action_from_human() -> &'static str {
    // Human mode: press 1 for A, 2 for B.
    if is_key_down(KeyCode::Key1) {
        "A"
    } else if is_key_down(KeyCode::Key2) {
        "B"
    } else {
        "A"
    }
}

fn button(rect: Rect, label: &str, active: bool) -> bool {
    let (mx, my) = mouse_position();
    let hovered = rect.contains(vec2(mx, my));
    let clicked = hovered && is_mouse_button_pressed(MouseButton::Left);

    let bg = if active {
        Color::new(0.20, 0.45, 0.22, 1.0)
    } else if hovered {
        Color::new(0.25, 0.25, 0.25, 1.0)
    } else {
        Color::new(0.18, 0.18, 0.18, 1.0)
    };

    draw_rectangle(rect.x, rect.y, rect.w, rect.h, bg);
    draw_rectangle_lines(rect.x, rect.y, rect.w, rect.h, 1.0, GRAY);
    draw_text(
        label,
        rect.x + 10.0,
        rect.y + rect.h * 0.72,
        BTN_FONT_SIZE,
        WHITE,
    );

    clicked
}

fn draw_label(rect: Rect, label: &str, active: bool) {
    let bg = if active {
        Color::new(0.20, 0.45, 0.22, 1.0)
    } else {
        Color::new(0.18, 0.18, 0.18, 1.0)
    };

    draw_rectangle(rect.x, rect.y, rect.w, rect.h, bg);
    draw_rectangle_lines(rect.x, rect.y, rect.w, rect.h, 1.0, GRAY);
    draw_text(
        label,
        rect.x + 10.0,
        rect.y + rect.h * 0.72,
        BTN_FONT_SIZE,
        WHITE,
    );
}

#[derive(Debug, Clone)]
struct PongUi {
    ball_x: f32,
    ball_y: f32,
    ball_vx: f32,
    ball_vy: f32,

    decoy_x: f32,
    decoy_y: f32,
    decoy_vx: f32,
    decoy_vy: f32,
    paddle_x: f32,

    hits: u32,
    misses: u32,
    score: f32,
    recent: Vec<bool>,

    rng_seed: u64,

    // Automatic, hidden regime shift: the *sensor mapping* flips every N outcomes.
    sensor_axis_flipped: bool,
    shift_every_outcomes: u32,
    last_flip_outcomes: u32,

    // Harder mode knobs.
    misbucket_p: f32,
    rel_jitter: f32,

    // Exploration (helps avoid stuck policies under sparse reward).
    explore_p: f32,

    // Meta-modulation (temporal/progress): adjust exploration based on
    // time-between-hits relative to an EMA baseline.
    steps_since_hit: u32,
    ema_hit_interval: f32,
    last_r_time: f32,
    meta_explore_add: f32,

    // Option A: stabilize HUD interpretability by smoothing action scores.
    ema_action_scores: HashMap<String, f32>,

    // Decoy enable (distractor).
    decoy_enabled: bool,

    // Negative reinforcement on miss.
    neg_reinforce: bool,
    neg_reinforce_action: f32,
    neg_reinforce_pair: f32,

    // A simple difficulty level that maps to the knobs above.
    difficulty_level: u8,
    // Flip evaluation metrics.
    last_flip_at_frame: u64,
    last_flip_recent_before: f32,
    last_flip_recovered_in_outcomes: Option<u32>,
}

#[derive(Debug, Clone, Copy)]
struct PongTickResult {
    ctx: &'static str,
    action: &'static str,
    reward: f32,
    just_flipped: bool,
}

struct PongChildTrainer {
    brain: Brain,
    env: PongUi,
    steps_left: u32,
    start_hits: u32,
    start_misses: u32,
    total_reward: f32,
    reason: String,
    explore_p: f32,
}

struct PongChildManager {
    trainer: Option<PongChildTrainer>,
    last_flip_outcomes: u32,
    last_child_spawn_outcomes: u32,
    last_child_score: f32,
    last_event: String,
}

impl PongChildManager {
    fn new() -> Self {
        Self {
            trainer: None,
            last_flip_outcomes: 0,
            last_child_spawn_outcomes: 0,
            last_child_score: 0.0,
            last_event: String::new(),
        }
    }

    fn status_line(&self) -> String {
        if let Some(t) = &self.trainer {
            format!(
                "child_sandbox=ON steps_left={} reason={} total_reward={:+.1}",
                t.steps_left, t.reason, t.total_reward
            )
        } else if !self.last_event.is_empty() {
            format!(
                "child_sandbox=OFF last_score={:+.2} last_event={}",
                self.last_child_score, self.last_event
            )
        } else {
            "child_sandbox=OFF".to_string()
        }
    }
}

impl PongUi {
    fn new(cfg: &VisConfig) -> Self {
        let mut s = Self {
            ball_x: cfg.world_w * 0.5,
            ball_y: 40.0,
            ball_vx: 0.0,
            ball_vy: cfg.ball_speed_y,
            decoy_x: cfg.world_w * 0.25,
            decoy_y: 120.0,
            decoy_vx: 0.0,
            decoy_vy: cfg.ball_speed_y * 0.92,
            paddle_x: cfg.world_w * 0.5,
            hits: 0,
            misses: 0,
            score: 0.0,
            recent: Vec::with_capacity(400),
            rng_seed: 2026,
            sensor_axis_flipped: false,
            // Slower by default; can be adjusted in UI.
            shift_every_outcomes: 160,
            last_flip_outcomes: 0,
            misbucket_p: 0.08,
            rel_jitter: 18.0,
            explore_p: 0.06,
            steps_since_hit: 0,
            ema_hit_interval: 0.0,
            last_r_time: 0.0,
            meta_explore_add: 0.0,
            ema_action_scores: HashMap::new(),
            decoy_enabled: true,
            neg_reinforce: true,
            neg_reinforce_action: 0.25,
            neg_reinforce_pair: 0.45,
            difficulty_level: 2,
            last_flip_at_frame: 0,
            last_flip_recent_before: 0.0,
            last_flip_recovered_in_outcomes: None,
        };
        // Ensure initial parameters match the difficulty mapping.
        s.apply_difficulty();
        s.reset_ball(cfg);
        s.reset_decoy(cfg);
        s
    }

    fn apply_difficulty(&mut self) {
        // Difficulty levels: 0 easiest .. 3 hardest.
        match self.difficulty_level {
            0 => {
                self.misbucket_p = 0.00;
                self.rel_jitter = 0.0;
                self.decoy_enabled = false;
                self.explore_p = 0.10;
                self.shift_every_outcomes = self.shift_every_outcomes.max(240);
            }
            1 => {
                self.misbucket_p = 0.04;
                self.rel_jitter = 10.0;
                self.decoy_enabled = false;
                self.explore_p = 0.08;
                self.shift_every_outcomes = self.shift_every_outcomes.max(200);
            }
            2 => {
                self.misbucket_p = 0.08;
                self.rel_jitter = 18.0;
                self.decoy_enabled = true;
                self.explore_p = 0.06;
                self.shift_every_outcomes = self.shift_every_outcomes.max(160);
            }
            _ => {
                self.misbucket_p = 0.12;
                self.rel_jitter = 26.0;
                self.decoy_enabled = true;
                self.explore_p = 0.05;
                self.shift_every_outcomes = self.shift_every_outcomes.max(120);
            }
        }
    }

    fn easier(&mut self) {
        if self.difficulty_level > 0 {
            self.difficulty_level -= 1;
            self.apply_difficulty();
        }
    }

    fn harder(&mut self) {
        if self.difficulty_level < 3 {
            self.difficulty_level += 1;
            self.apply_difficulty();
        }
    }

    fn flip_slower(&mut self) {
        self.shift_every_outcomes = (self.shift_every_outcomes + 40).clamp(40, 400);
        self.apply_difficulty();
    }

    fn flip_faster(&mut self) {
        self.shift_every_outcomes = self.shift_every_outcomes.saturating_sub(40).clamp(40, 400);
        self.apply_difficulty();
    }

    fn outcomes(&self) -> u32 {
        self.hits + self.misses
    }

    fn flip_countdown(&self) -> u32 {
        let shift = self.shift_every_outcomes.max(1);
        let o = self.outcomes();
        if o == 0 {
            shift
        } else {
            let m = o % shift;
            if m == 0 { shift } else { shift - m }
        }
    }

    fn reset_ball(&mut self, cfg: &VisConfig) {
        self.rng_seed = self
            .rng_seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1);
        let u = (self.rng_seed >> 11) as u32;
        let x01 = (u as f32) / (u32::MAX as f32);
        self.ball_x = cfg.ball_r + x01 * (cfg.world_w - 2.0 * cfg.ball_r);
        self.ball_y = 40.0;

        // Randomize horizontal drift a bit so the task isn't trivial.
        self.rng_seed = self
            .rng_seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1);
        let u2 = (self.rng_seed >> 11) as u32;
        let s01 = (u2 as f32) / (u32::MAX as f32);
        // Slightly increase drift at higher difficulties.
        let drift = match self.difficulty_level {
            0 => 90.0,
            1 => 120.0,
            2 => 140.0,
            _ => 170.0,
        };
        self.ball_vx = (s01 * 2.0 - 1.0) * drift;
        self.ball_vy = cfg.ball_speed_y;
    }

    fn reset_decoy(&mut self, cfg: &VisConfig) {
        self.rng_seed = self
            .rng_seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1);
        let u = (self.rng_seed >> 11) as u32;
        let x01 = (u as f32) / (u32::MAX as f32);
        self.decoy_x = cfg.ball_r + x01 * (cfg.world_w - 2.0 * cfg.ball_r);
        self.decoy_y = 40.0;

        self.rng_seed = self
            .rng_seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1);
        let u2 = (self.rng_seed >> 11) as u32;
        let s01 = (u2 as f32) / (u32::MAX as f32);
        let drift = match self.difficulty_level {
            0 => 0.0,
            1 => 0.0,
            2 => 160.0,
            _ => 210.0,
        };
        self.decoy_vx = (s01 * 2.0 - 1.0) * drift;
        self.decoy_vy = cfg.ball_speed_y * 0.92;
    }

    fn noisy_bucket_index(&mut self, rel: f32) -> usize {
        // Add jitter first (harder sensing).
        let jitter = macroquad::rand::gen_range(-self.rel_jitter, self.rel_jitter);
        let mut idx = bucket_index_from_rel(rel + jitter);

        // Occasionally mis-bucket to a neighbor (harder learning).
        let p = macroquad::rand::gen_range(0.0, 1.0);
        if p < self.misbucket_p {
            let dir = if macroquad::rand::gen_range(0, 2) == 0 {
                -1
            } else {
                1
            };
            let i = idx as i32 + dir;
            idx = i.clamp(0, 4) as usize;
        }

        idx
    }

    fn record_outcome(&mut self, ok: bool) {
        self.recent.push(ok);
        if self.recent.len() > 400 {
            self.recent.remove(0);
        }
    }

    fn recent_rate(&self) -> f32 {
        if self.recent.is_empty() {
            return 0.0;
        }
        let good = self.recent.iter().filter(|&&b| b).count();
        good as f32 / self.recent.len() as f32
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
        explore_p: f32,
    ) -> PongTickResult {
        self.steps_since_hit = self.steps_since_hit.saturating_add(1);
        // Decay meta exploration boost slowly so it's a temporary adaptation lever.
        self.meta_explore_add *= 0.995;

        // Sensor mapping shift is hidden from the brain: we flip rel before bucketing.
        let rel_raw = self.ball_x - self.paddle_x;
        let rel_seen = if self.sensor_axis_flipped {
            -rel_raw
        } else {
            rel_raw
        };
        let ctx_idx = self.noisy_bucket_index(rel_seen);
        let ctx = pong_ctx_name(ctx_idx);

        // Distractor (decoy ball) sensors.
        let decoy_rel_raw = self.decoy_x - self.paddle_x;
        let decoy_rel_seen = if self.sensor_axis_flipped {
            -decoy_rel_raw
        } else {
            decoy_rel_raw
        };
        let decoy_idx = self.noisy_bucket_index(decoy_rel_seen);
        let decoy_ctx = pong_decoy_ctx_name(decoy_idx);

        brain.apply_stimulus(Stimulus::new(ctx, 1.0));
        brain.apply_stimulus(Stimulus::new(
            "pong_ball_falling",
            if self.ball_vy > 0.0 { 0.9 } else { 0.0 },
        ));
        brain.apply_stimulus(Stimulus::new(
            decoy_ctx,
            if self.decoy_enabled { 0.9 } else { 0.0 },
        ));
        brain.apply_stimulus(Stimulus::new(
            "pong_decoy_falling",
            if self.decoy_enabled && self.decoy_vy > 0.0 {
                0.7
            } else {
                0.0
            },
        ));
        brain.step();

        let explore_eff = (explore_p + self.meta_explore_add).clamp(0.0, 0.40);

        let action: &'static str = match mode {
            ControlMode::Human => pong_action_from_human(),
            ControlMode::Braine => {
                // Small exploration probability to avoid getting stuck under sparse rewards.
                if macroquad::rand::gen_range(0.0, 1.0) < explore_eff {
                    match macroquad::rand::gen_range(0, 3) {
                        0 => "left",
                        1 => "right",
                        _ => "stay",
                    }
                } else {
                    let (a, _score) = brain.select_action_with_meaning(ctx, 6.0);
                    if a == "left" {
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
        let pair = format!("pair::{ctx}::{action}");
        brain.note_action(&pair);

        let mut ax = 0.0;
        match action {
            "left" => ax = -1.0,
            "right" => ax = 1.0,
            _ => {}
        }
        self.paddle_x += ax * cfg.paddle_speed * dt;
        let half = cfg.paddle_w * 0.5;
        self.paddle_x = self.paddle_x.clamp(half, cfg.world_w - half);

        // Ball physics.
        self.ball_x += self.ball_vx * dt;
        self.ball_y += self.ball_vy * dt;
        if self.ball_x <= cfg.ball_r {
            self.ball_x = cfg.ball_r;
            self.ball_vx = self.ball_vx.abs();
        } else if self.ball_x >= cfg.world_w - cfg.ball_r {
            self.ball_x = cfg.world_w - cfg.ball_r;
            self.ball_vx = -self.ball_vx.abs();
        }
        if self.ball_y <= cfg.ball_r {
            self.ball_y = cfg.ball_r;
            self.ball_vy = self.ball_vy.abs();
        }

        // Decoy physics.
        self.decoy_x += self.decoy_vx * dt;
        self.decoy_y += self.decoy_vy * dt;
        if self.decoy_x <= cfg.ball_r {
            self.decoy_x = cfg.ball_r;
            self.decoy_vx = self.decoy_vx.abs();
        } else if self.decoy_x >= cfg.world_w - cfg.ball_r {
            self.decoy_x = cfg.world_w - cfg.ball_r;
            self.decoy_vx = -self.decoy_vx.abs();
        }
        if self.decoy_y <= cfg.ball_r {
            self.decoy_y = cfg.ball_r;
            self.decoy_vy = self.decoy_vy.abs();
        }

        // Ball-ball collision (main ball vs decoy): simple equal-mass elastic response.
        // This makes the world dynamics a bit richer without changing reward structure.
        if self.decoy_enabled {
            let dx = self.ball_x - self.decoy_x;
            let dy = self.ball_y - self.decoy_y;
            let rr = cfg.ball_r * 2.0;
            let d2 = dx * dx + dy * dy;
            if d2 > 0.0001 && d2 < rr * rr {
                let d = d2.sqrt();
                let nx = dx / d;
                let ny = dy / d;

                // Push them apart so they don't stick.
                let overlap = rr - d;
                let push = overlap * 0.5;
                self.ball_x += nx * push;
                self.ball_y += ny * push;
                self.decoy_x -= nx * push;
                self.decoy_y -= ny * push;

                // Exchange velocity along the normal (equal masses).
                let rvx = self.ball_vx - self.decoy_vx;
                let rvy = self.ball_vy - self.decoy_vy;
                let rel_n = rvx * nx + rvy * ny;
                // Only resolve if moving toward each other.
                if rel_n < 0.0 {
                    self.ball_vx -= rel_n * nx;
                    self.ball_vy -= rel_n * ny;
                    self.decoy_vx += rel_n * nx;
                    self.decoy_vy += rel_n * ny;
                }
            }
        }

        // Harder reward: terminal-only (no shaping).
        let mut reward = 0.0f32;

        // Paddle collision (bounce on hit).
        let paddle_left = self.paddle_x - half;
        let paddle_right = self.paddle_x + half;
        let hit_zone_left = paddle_left - cfg.catch_margin;
        let hit_zone_right = paddle_right + cfg.catch_margin;
        let paddle_top = cfg.paddle_y;

        let ball_reached_paddle = self.ball_vy > 0.0 && (self.ball_y + cfg.ball_r) >= paddle_top;
        let ball_in_x = self.ball_x >= hit_zone_left && self.ball_x <= hit_zone_right;
        if ball_reached_paddle && ball_in_x {
            // Bounce upward.
            self.ball_y = paddle_top - cfg.ball_r;
            self.ball_vy = -self.ball_vy.abs();
            // Add some "spin" from where it hit.
            let offset = (self.ball_x - self.paddle_x) / (half.max(1.0));
            self.ball_vx += offset * 120.0;
            self.ball_vx = self.ball_vx.clamp(-260.0, 260.0);

            self.hits += 1;
            reward = 1.0;
            self.score += 1.0;
            self.record_outcome(true);

            // Meta-modulation (temporal/progress): compute a "faster-than-baseline" signal.
            // Use it to temporarily boost exploration when we're slower than our own past.
            let interval = self.steps_since_hit.max(1) as f32;
            let baseline = if self.ema_hit_interval <= 0.0 {
                interval
            } else {
                self.ema_hit_interval
            };
            let eps = 1e-3;
            let r_time = ((baseline - interval) / (baseline + eps)).clamp(-1.0, 1.0);
            self.last_r_time = r_time;

            // Update baseline after measuring.
            let alpha = 0.08;
            if self.ema_hit_interval <= 0.0 {
                self.ema_hit_interval = interval;
            } else {
                self.ema_hit_interval = (1.0 - alpha) * self.ema_hit_interval + alpha * interval;
            }

            // If we're slower than baseline, explore more for a while.
            if r_time < -0.10 {
                let boost = (-r_time * 0.12).clamp(0.0, 0.18);
                if boost > self.meta_explore_add {
                    self.meta_explore_add = boost;
                }
            } else if r_time > 0.10 {
                // If we're improving, gently reduce the boost.
                self.meta_explore_add *= 0.85;
            }

            self.steps_since_hit = 0;
        }

        // Miss condition: ball passes below paddle.
        if self.ball_y > (cfg.paddle_y + cfg.paddle_h + cfg.ball_r + 6.0) {
            self.misses += 1;
            reward = -1.0;
            self.score *= 0.98;
            self.score -= 0.25;
            self.record_outcome(false);
            self.reset_ball(cfg);

            // Negative reinforcement helps unlearn bad ctx->action associations.
            if self.neg_reinforce {
                brain.reinforce_action(action, -self.neg_reinforce_action);
                brain.reinforce_action(&pair, -self.neg_reinforce_pair);
            }
        }

        // Decoy bounces on paddle too (no reward), and resets if it passes below.
        let decoy_reached_paddle = self.decoy_vy > 0.0 && (self.decoy_y + cfg.ball_r) >= paddle_top;
        let decoy_in_x = self.decoy_x >= hit_zone_left && self.decoy_x <= hit_zone_right;
        if self.decoy_enabled && decoy_reached_paddle && decoy_in_x {
            self.decoy_y = paddle_top - cfg.ball_r;
            self.decoy_vy = -self.decoy_vy.abs();
            let offset = (self.decoy_x - self.paddle_x) / (half.max(1.0));
            self.decoy_vx += offset * 90.0;
            self.decoy_vx = self.decoy_vx.clamp(-280.0, 280.0);
        }
        if self.decoy_enabled && self.decoy_y > (cfg.paddle_y + cfg.paddle_h + cfg.ball_r + 6.0) {
            self.reset_decoy(cfg);
        }

        // Regime shift uses outcome count.
        let outcomes = self.hits + self.misses;
        let mut just_flipped = false;
        if outcomes > 0
            && outcomes % self.shift_every_outcomes.max(1) == 0
            && outcomes != self.last_flip_outcomes
        {
            self.sensor_axis_flipped = !self.sensor_axis_flipped;
            self.last_flip_outcomes = outcomes;
            just_flipped = true;
        }

        reward = reward.clamp(-1.0, 1.0);
        brain.set_neuromodulator(reward);
        if reward > 0.2 {
            brain.reinforce_action(action, 0.6);
        }
        brain.commit_observation();

        PongTickResult {
            ctx,
            action,
            reward,
            just_flipped,
        }
    }

    fn tick_and_render(
        &mut self,
        cfg: &VisConfig,
        app: &AppState,
        brain: &mut Brain,
        logger: &mut MetricsLogger,
        child_mgr: &mut PongChildManager,
        frame: u64,
        dt: f32,
    ) {
        let recent_before = self.recent_rate_n(32);
        let step = self.tick(cfg, app.mode, brain, dt, self.explore_p);
        let outcomes_after = self.outcomes();

        if step.just_flipped {
            child_mgr.last_flip_outcomes = self.outcomes();
            child_mgr.last_event = format!(
                "flip@outcomes={} sensor_axis_flipped={}",
                child_mgr.last_flip_outcomes, self.sensor_axis_flipped
            );
            logger.log_event(frame, "Pong", &child_mgr.last_event);

            // P1 metrics: capture baseline just before the flip.
            self.last_flip_at_frame = frame;
            self.last_flip_recent_before = recent_before;
            self.last_flip_recovered_in_outcomes = None;

            logger.log_event(
                frame,
                "Pong",
                &format!(
                    "FLIP_MARKER outcomes={} recent32_before={:.3}",
                    child_mgr.last_flip_outcomes, self.last_flip_recent_before
                ),
            );
        }

        // P1 metrics: detect recovery after flip.
        if child_mgr.last_flip_outcomes > 0 && self.last_flip_recovered_in_outcomes.is_none() {
            let since_flip = outcomes_after.saturating_sub(child_mgr.last_flip_outcomes);
            if since_flip >= 10 {
                let recent_after = self.recent_rate_n(32);
                // Simple recovery criterion: regain at least 0.45 recent hit-rate.
                if recent_after >= 0.45 {
                    self.last_flip_recovered_in_outcomes = Some(since_flip);
                    logger.log_event(
                        frame,
                        "Pong",
                        &format!(
                            "FLIP_RECOVERY outcomes_since_flip={} recent32_after={:.3} recent32_before={:.3}",
                            since_flip, recent_after, self.last_flip_recent_before
                        ),
                    );
                }
            }
        }

        // If we're post-flip and performance slumps, spawn a child brain sandbox to adapt fast.
        // The child trains on a cloned Pong state for a fixed budget, then consolidates back.
        let outcomes = self.outcomes();
        let recently_flipped = child_mgr.last_flip_outcomes > 0
            && outcomes >= child_mgr.last_flip_outcomes.saturating_add(6)
            && outcomes <= child_mgr.last_flip_outcomes.saturating_add(50);
        let slump = recently_flipped && self.recent_rate_n(24) < 0.30;
        let spawn_cooldown_ok = outcomes.saturating_sub(child_mgr.last_child_spawn_outcomes) > 60;
        if child_mgr.trainer.is_none() && slump && spawn_cooldown_ok {
            let seed = self.rng_seed ^ ((frame as u64) << 1) ^ (outcomes as u64);
            let overrides = ChildConfigOverrides {
                noise_amp: 0.045,
                noise_phase: 0.020,
                hebb_rate: 0.14,
                forget_rate: 0.0013,
            };
            let child_brain = brain.spawn_child(seed, overrides);
            let child_env = self.clone();
            let reason = format!(
                "slump_post_flip recent24={:.2} outcomes={} flip@{}",
                self.recent_rate_n(24),
                outcomes,
                child_mgr.last_flip_outcomes
            );
            child_mgr.trainer = Some(PongChildTrainer {
                brain: child_brain,
                env: child_env,
                steps_left: 900,
                start_hits: self.hits,
                start_misses: self.misses,
                total_reward: 0.0,
                reason: reason.clone(),
                explore_p: (self.explore_p + 0.06).clamp(0.0, 0.35),
            });
            child_mgr.last_child_spawn_outcomes = outcomes;
            child_mgr.last_event = format!("child_spawn seed={} {}", seed, reason);
            logger.log_event(frame, "Pong", &child_mgr.last_event);
        }

        // Step the sandbox child a little each frame so we don't stall rendering.
        if let Some(t) = child_mgr.trainer.as_mut() {
            let sandbox_dt = 1.0 / 60.0;
            let steps_per_frame = 3;
            for _ in 0..steps_per_frame {
                if t.steps_left == 0 {
                    break;
                }
                let r = t.env.tick(
                    cfg,
                    ControlMode::Braine,
                    &mut t.brain,
                    sandbox_dt,
                    t.explore_p,
                );
                t.total_reward += r.reward;
                t.steps_left = t.steps_left.saturating_sub(1);
            }

            if t.steps_left == 0 {
                let delta_hits = t.env.hits.saturating_sub(t.start_hits);
                let delta_misses = t.env.misses.saturating_sub(t.start_misses);
                let child_score = delta_hits as f32 - delta_misses as f32;

                let child_recent32 = t.env.recent_rate_n(32);
                let parent_recent32 = self.recent_rate_n(32);

                brain.consolidate_from(
                    &t.brain,
                    ConsolidationPolicy {
                        weight_threshold: 0.12,
                        merge_rate: 0.45,
                    },
                );

                child_mgr.last_child_score = child_score;
                child_mgr.last_event = format!(
                    "child_consolidate score={:+.1} (Δhits={}, Δmisses={}) child_recent32={:.2} parent_recent32={:.2} child_total_reward={:+.1}",
                    child_score,
                    delta_hits,
                    delta_misses,
                    child_recent32,
                    parent_recent32,
                    t.total_reward
                );
                logger.log_event(frame, "Pong", &child_mgr.last_event);
                child_mgr.trainer = None;
            }
        }

        // Render world below the top UI bar (buttons + status text).
        let oy = TOP_UI_H;
        let half = cfg.paddle_w * 0.5;
        draw_rectangle_lines(0.0, oy, cfg.world_w, cfg.world_h, 2.0, DARKGRAY);
        draw_rectangle(
            self.paddle_x - half,
            cfg.paddle_y + oy,
            cfg.paddle_w,
            cfg.paddle_h,
            Color::new(0.75, 0.75, 0.85, 1.0),
        );
        draw_circle(
            self.ball_x,
            self.ball_y + oy,
            cfg.ball_r,
            Color::new(0.95, 0.45, 0.35, 1.0),
        );

        // Decoy ball (distractor).
        draw_circle(
            self.decoy_x,
            self.decoy_y + oy,
            cfg.ball_r,
            Color::new(0.55, 0.65, 0.90, 1.0),
        );

        let hit_rate = {
            let total = self.hits + self.misses;
            if total == 0 {
                0.0
            } else {
                self.hits as f32 / total as f32
            }
        };

        let ctx = step.ctx;
        let action = step.action;
        let reward = step.reward;
        let hint = brain.meaning_hint(ctx);
        let top_actions = brain.top_actions_with_meaning(ctx, 6.0, 3);
        let scored_all = brain.ranked_actions_with_meaning(ctx, 6.0);

        // Smooth scores to reduce frame-to-frame jitter in the HUD.
        let ema_alpha = 0.10;
        for (a, s) in scored_all.iter() {
            let e = self.ema_action_scores.entry(a.clone()).or_insert(*s);
            *e = (1.0 - ema_alpha) * (*e) + ema_alpha * (*s);
        }
        let mut top_actions_ema: Vec<(String, f32)> = top_actions
            .iter()
            .map(|(a, _)| (a.clone(), *self.ema_action_scores.get(a).unwrap_or(&0.0)))
            .collect();
        top_actions_ema.sort_by(|x, y| y.1.total_cmp(&x.1));

        // Option A: show a couple of causal links for "why".
        let top_links = brain.top_causal_links_from(ctx, 4);
        let pair_sym = format!("pair::{ctx}::{action}");
        let top_pair_links = brain.top_causal_links_from(&pair_sym, 3);
        let diag = brain.diagnostics();
        let snap = BrainAdapter::new(brain).snapshot();

        let hud = [
            format!(
                "test=Pong mode={:?} sensor_axis_flipped={} (auto)",
                app.mode, self.sensor_axis_flipped
            ),
            child_mgr.status_line(),
            {
                if child_mgr.last_flip_outcomes > 0 {
                    let since = self.outcomes().saturating_sub(child_mgr.last_flip_outcomes);
                    let rec = self
                        .last_flip_recovered_in_outcomes
                        .map(|n| format!("recovered_in={}", n))
                        .unwrap_or_else(|| "recovered_in=?".to_string());
                    format!(
                        "flip_metrics: recent32_before={:.2} recent32_now={:.2} since_flip_outcomes={} {}",
                        self.last_flip_recent_before,
                        self.recent_rate_n(32),
                        since,
                        rec
                    )
                } else {
                    "flip_metrics: (no flips yet)".to_string()
                }
            },
            format!(
                "meta: since_hit_steps={} ema_hit_steps={:.1} r_time={:+.2} explore_eff={:.2} (base={:.2} add={:.2})",
                self.steps_since_hit,
                self.ema_hit_interval,
                self.last_r_time,
                (self.explore_p + self.meta_explore_add).clamp(0.0, 0.40),
                self.explore_p,
                self.meta_explore_add,
            ),
            format!(
                "difficulty={} terminal_reward_only misbucket_p={:.2} rel_jitter={:.1} decoy={} explore_p={:.2}",
                self.difficulty_level,
                self.misbucket_p,
                self.rel_jitter,
                self.decoy_enabled,
                self.explore_p
            ),
            format!(
                "regime_flip_every_outcomes={} next_flip_in={} neg_reinforce={} (act={:.2},pair={:.2})",
                self.shift_every_outcomes,
                self.flip_countdown(),
                self.neg_reinforce,
                self.neg_reinforce_action,
                self.neg_reinforce_pair
            ),
            format!("ctx={} action={} reward={:+.3}", ctx, action, reward),
            format!(
                "hits={} misses={} hit_rate={:.3} recent={:.3} score={:.2}",
                self.hits,
                self.misses,
                hit_rate,
                self.recent_rate(),
                self.score
            ),
            format!(
                "brain: age_steps={} conns={} pruned_last_step={} avg_amp={:.3}",
                snap.age_steps, diag.connection_count, diag.pruned_last_step, diag.avg_amp
            ),
            format!(
                "causal: base_symbols={} edges={} last_updates(directed={},cooccur={})",
                snap.causal.base_symbols,
                snap.causal.edges,
                snap.causal.last_directed_edge_updates,
                snap.causal.last_cooccur_edge_updates
            ),
            format!("meaning_hint({})={:?}", ctx, hint),
            format!(
                "top_actions(ctx): {}",
                top_actions
                    .iter()
                    .map(|(a, s)| format!("{}:{:+.2}", a, s))
                    .collect::<Vec<_>>()
                    .join("  ")
            ),
            format!(
                "top_actions_ema: {}",
                top_actions_ema
                    .iter()
                    .map(|(a, s)| format!("{}:{:+.2}", a, s))
                    .collect::<Vec<_>>()
                    .join("  ")
            ),
            format!(
                "causal_from(ctx): {}",
                top_links
                    .iter()
                    .map(|(n, s)| format!("{}:{:+.2}", n, s))
                    .collect::<Vec<_>>()
                    .join("  ")
            ),
            format!(
                "causal_from(pair): {}",
                top_pair_links
                    .iter()
                    .map(|(n, s)| format!("{}:{:+.2}", n, s))
                    .collect::<Vec<_>>()
                    .join("  ")
            ),
        ];

        logger.log_snapshot(
            frame,
            "Pong",
            app.mode,
            reward,
            &hud,
            &snap.last_reinforced_actions,
        );

        // HUD: keep the exact content, but render it in a panel with word-wrapping
        // so it looks more like a real dashboard.
        let hud_rect = Rect::new(
            UI_MARGIN,
            HUD_START_Y,
            screen_width() - 2.0 * UI_MARGIN,
            TOP_UI_H - HUD_START_Y - UI_MARGIN,
        );
        let mut hud_lines: Vec<String> = hud.into_iter().collect();
        if !snap.last_reinforced_actions.is_empty() {
            hud_lines.push("reinforce:".to_string());
            for (name, delta) in snap.last_reinforced_actions.iter().take(5) {
                hud_lines.push(format!("  {} {:+.2}", name, delta));
            }
        }
        draw_hud_lines_wrapped(&hud_lines, hud_rect, WHITE);
    }
}

#[derive(Debug, Clone)]
struct BanditUi {
    t: u32,
    p_a: f32,
    p_b: f32,
    shift_every: u32,
    wins: u32,
    losses: u32,
    recent: Vec<bool>,
    // Flip evaluation metrics.
    last_flip_at_t: u32,
    last_flip_recent_before: f32,
    last_flip_recovered_in_steps: Option<u32>,

    // Option A: stabilize HUD interpretability by smoothing action scores.
    ema_action_scores: HashMap<String, f32>,
}

impl BanditUi {
    fn new() -> Self {
        Self {
            t: 0,
            p_a: 0.8,
            p_b: 0.2,
            shift_every: 220,
            wins: 0,
            losses: 0,
            recent: Vec::with_capacity(400),
            last_flip_at_t: 0,
            last_flip_recent_before: 0.0,
            last_flip_recovered_in_steps: None,

            ema_action_scores: HashMap::new(),
        }
    }

    fn recent_rate(&self) -> f32 {
        if self.recent.is_empty() {
            return 0.0;
        }
        let good = self.recent.iter().filter(|&&b| b).count();
        good as f32 / self.recent.len() as f32
    }

    fn push_recent(&mut self, ok: bool) {
        self.recent.push(ok);
        if self.recent.len() > 400 {
            self.recent.remove(0);
        }
    }

    fn tick_and_render(
        &mut self,
        _cfg: &VisConfig,
        app: &AppState,
        brain: &mut Brain,
        logger: &mut MetricsLogger,
        frame: u64,
        _dt: f32,
    ) {
        self.t = self.t.wrapping_add(1);

        let recent_before = {
            if self.recent.is_empty() {
                0.0
            } else {
                let n = 64.min(self.recent.len());
                let tail = &self.recent[self.recent.len() - n..];
                let good = tail.iter().filter(|&&b| b).count();
                good as f32 / n as f32
            }
        };

        // Automatic hidden regime shift: swap arm probabilities periodically.
        if self.t % self.shift_every == 0 {
            core::mem::swap(&mut self.p_a, &mut self.p_b);

            // P1 marker + baseline.
            self.last_flip_at_t = self.t;
            self.last_flip_recent_before = recent_before;
            self.last_flip_recovered_in_steps = None;

            logger.log_event(
                frame,
                "Bandit",
                &format!(
                    "FLIP_MARKER t={} recent64_before={:.3} pA_now={:.2} pB_now={:.2}",
                    self.t, self.last_flip_recent_before, self.p_a, self.p_b
                ),
            );
        }

        let bandit_flip_countdown = if self.shift_every == 0 {
            0
        } else {
            let m = self.t % self.shift_every;
            if m == 0 {
                self.shift_every
            } else {
                self.shift_every - m
            }
        };

        let ctx = "bandit_ctx";
        brain.apply_stimulus(Stimulus::new(ctx, 1.0));
        brain.step();

        let action = match app.mode {
            ControlMode::Human => bandit_action_from_human().to_string(),
            ControlMode::Braine => {
                let (a, _score) = brain.select_action_with_meaning(ctx, 8.0);
                a
            }
        };

        brain.note_action(&action);
        brain.note_action(&format!("pair::{ctx}::{action}"));

        let p = if action == "A" { self.p_a } else { self.p_b };
        let ok = macroquad::rand::gen_range(0.0, 1.0) < p;
        let reward = if ok { 0.8 } else { -0.6 };

        if ok {
            self.wins += 1;
        } else {
            self.losses += 1;
        }
        self.push_recent(ok);

        // P1 recovery: regain recent win-rate.
        if self.last_flip_at_t > 0 && self.last_flip_recovered_in_steps.is_none() {
            let since = self.t.saturating_sub(self.last_flip_at_t);
            if since >= 20 {
                let recent_now = {
                    let n = 64.min(self.recent.len());
                    let tail = &self.recent[self.recent.len() - n..];
                    let good = tail.iter().filter(|&&b| b).count();
                    good as f32 / n as f32
                };
                if recent_now >= 0.60 {
                    self.last_flip_recovered_in_steps = Some(since);
                    logger.log_event(
                        frame,
                        "Bandit",
                        &format!(
                            "FLIP_RECOVERY steps_since_flip={} recent64_after={:.3} recent64_before={:.3}",
                            since, recent_now, self.last_flip_recent_before
                        ),
                    );
                }
            }
        }

        brain.set_neuromodulator(reward);
        if reward > 0.2 {
            brain.reinforce_action(&action, 0.45);
        }
        brain.commit_observation();

        // Render.
        let w = screen_width();
        let h = screen_height();
        let cx = w * 0.5;
        let cy = h * 0.55;

        draw_text(
            "Bandit (Human): press 1 for A, 2 for B",
            UI_MARGIN,
            HUD_START_Y + 22.0,
            20.0,
            GRAY,
        );

        draw_rectangle(
            cx - 220.0,
            cy - 50.0,
            160.0,
            100.0,
            Color::new(0.14, 0.14, 0.16, 1.0),
        );
        draw_rectangle(
            cx + 60.0,
            cy - 50.0,
            160.0,
            100.0,
            Color::new(0.14, 0.14, 0.16, 1.0),
        );
        draw_text("A", cx - 160.0, cy + 12.0, 48.0, WHITE);
        draw_text("B", cx + 120.0, cy + 12.0, 48.0, WHITE);

        let hint = brain.meaning_hint(ctx);
        let top_actions = brain.top_actions_with_meaning(ctx, 8.0, 2);
        let scored_all = brain.ranked_actions_with_meaning(ctx, 8.0);

        let ema_alpha = 0.12;
        for (a, s) in scored_all.iter() {
            let e = self.ema_action_scores.entry(a.clone()).or_insert(*s);
            *e = (1.0 - ema_alpha) * (*e) + ema_alpha * (*s);
        }
        let mut top_actions_ema: Vec<(String, f32)> = top_actions
            .iter()
            .map(|(a, _)| (a.clone(), *self.ema_action_scores.get(a).unwrap_or(&0.0)))
            .collect();
        top_actions_ema.sort_by(|x, y| y.1.total_cmp(&x.1));

        let top_links = brain.top_causal_links_from(ctx, 4);
        let pair_sym = format!("pair::{ctx}::{action}");
        let top_pair_links = brain.top_causal_links_from(&pair_sym, 3);
        let diag = brain.diagnostics();
        let snap = BrainAdapter::new(brain).snapshot();

        let total = self.wins + self.losses;
        let win_rate = if total == 0 {
            0.0
        } else {
            self.wins as f32 / total as f32
        };

        let hud = [
            format!("test=Bandit mode={:?}", app.mode),
            format!(
                "regime(auto): p(A)={:.2} p(B)={:.2} (flips every {} steps)",
                self.p_a, self.p_b, self.shift_every
            ),
            format!("next_flip_in_steps={}", bandit_flip_countdown),
            {
                if self.last_flip_at_t > 0 {
                    let since = self.t.saturating_sub(self.last_flip_at_t);
                    let recent_now = {
                        let n = 64.min(self.recent.len());
                        let tail = &self.recent[self.recent.len() - n..];
                        let good = tail.iter().filter(|&&b| b).count();
                        good as f32 / n as f32
                    };
                    let rec = self
                        .last_flip_recovered_in_steps
                        .map(|n| format!("recovered_in={}", n))
                        .unwrap_or_else(|| "recovered_in=?".to_string());
                    format!(
                        "flip_metrics: recent64_before={:.2} recent64_now={:.2} since_flip_steps={} {}",
                        self.last_flip_recent_before, recent_now, since, rec
                    )
                } else {
                    "flip_metrics: (no flips yet)".to_string()
                }
            },
            format!("ctx={} action={} reward={:+.3}", ctx, action, reward),
            format!(
                "wins={} losses={} win_rate={:.3} recent={:.3}",
                self.wins,
                self.losses,
                win_rate,
                self.recent_rate()
            ),
            format!(
                "brain: age_steps={} conns={} pruned_last_step={} avg_amp={:.3}",
                snap.age_steps, diag.connection_count, diag.pruned_last_step, diag.avg_amp
            ),
            format!("meaning_hint({})={:?}", ctx, hint),
            format!(
                "top_actions(ctx): {}",
                top_actions
                    .iter()
                    .map(|(a, s)| format!("{}:{:+.2}", a, s))
                    .collect::<Vec<_>>()
                    .join("  ")
            ),
            format!(
                "top_actions_ema: {}",
                top_actions_ema
                    .iter()
                    .map(|(a, s)| format!("{}:{:+.2}", a, s))
                    .collect::<Vec<_>>()
                    .join("  ")
            ),
            format!(
                "causal_from(ctx): {}",
                top_links
                    .iter()
                    .map(|(n, s)| format!("{}:{:+.2}", n, s))
                    .collect::<Vec<_>>()
                    .join("  ")
            ),
            format!(
                "causal_from(pair): {}",
                top_pair_links
                    .iter()
                    .map(|(n, s)| format!("{}:{:+.2}", n, s))
                    .collect::<Vec<_>>()
                    .join("  ")
            ),
        ];

        logger.log_snapshot(
            frame,
            "Bandit",
            app.mode,
            reward,
            &hud,
            &snap.last_reinforced_actions,
        );

        let hud_rect = Rect::new(
            UI_MARGIN,
            HUD_START_Y,
            screen_width() - 2.0 * UI_MARGIN,
            TOP_UI_H - HUD_START_Y - UI_MARGIN,
        );
        let mut hud_lines: Vec<String> = hud.into_iter().collect();
        if !snap.last_reinforced_actions.is_empty() {
            hud_lines.push("reinforce:".to_string());
            for (name, delta) in snap.last_reinforced_actions.iter().take(5) {
                hud_lines.push(format!("  {} {:+.2}", name, delta));
            }
        }
        draw_hud_lines_wrapped(&hud_lines, hud_rect, WHITE);
    }
}

#[macroquad::main(window_conf)]
async fn main() {
    let cfg = VisConfig::default();
    let mut app = AppState::default();

    let storage_cfg = StorageConfig::from_env_and_args();
    let mut last_storage_event: String = String::new();
    let _ = storage_cfg.ensure_parent_dir();

    let async_saver = AsyncSaver::new(storage_cfg.brain_image_path.clone());
    let async_loader = AsyncLoader::new(storage_cfg.brain_image_path.clone());
    let mut last_autosave_at: f64 = 0.0;

    let mut pong = PongUi::new(&cfg);
    let mut bandit = BanditUi::new();

    let mut forage = games::forage::ForageUi::new(&cfg);
    let mut whack = games::whack::WhackUi::new();
    let mut beacon = games::beacon::BeaconUi::new(&cfg);
    let mut sequence = games::sequence::SequenceUi::new();

    let mut pong_child = PongChildManager::new();

    let mut brain = mk_brain_for(app.test);
    let mut logger = MetricsLogger::new("braine_viz_metrics.log");
    let mut frame: u64 = 0;

    loop {
        frame = frame.wrapping_add(1);
        clear_background(Color::new(0.08, 0.08, 0.10, 1.0));

        // Persistence hotkeys (global):
        // - S: save current brain image
        // - L: load brain image (replaces current brain)
        if is_key_pressed(KeyCode::S) {
            match encode_brain_image(storage_cfg.capacity_bytes, &brain) {
                Ok(bytes) => match async_saver.try_enqueue(bytes) {
                    Ok(()) => {
                        last_storage_event =
                            format!("save queued -> {}", storage_cfg.brain_image_path.display());
                        logger.log_event(frame, "Storage", &last_storage_event);
                    }
                    Err(_) => {
                        last_storage_event = "save skipped (busy)".to_string();
                        logger.log_event(frame, "Storage", &last_storage_event);
                    }
                },
                Err(e) => {
                    last_storage_event = format!(
                        "save encode failed: {} (cap={:?})",
                        e, storage_cfg.capacity_bytes
                    );
                    logger.log_event(frame, "Storage", &last_storage_event);
                }
            }
        }
        if is_key_pressed(KeyCode::L) {
            match async_loader.try_enqueue() {
                Ok(()) => {
                    last_storage_event =
                        format!("load queued <- {}", storage_cfg.brain_image_path.display());
                    logger.log_event(frame, "Storage", &last_storage_event);
                }
                Err(_) => {
                    last_storage_event = "load skipped (busy)".to_string();
                    logger.log_event(frame, "Storage", &last_storage_event);
                }
            }
        }

        // Autosave: serialize in the main thread, write in a background thread.
        if let Some(every) = storage_cfg.autosave_every_secs {
            let now = get_time();
            if last_autosave_at == 0.0 {
                last_autosave_at = now;
            }
            if now - last_autosave_at >= every {
                last_autosave_at = now;
                if let Ok(bytes) = encode_brain_image(storage_cfg.capacity_bytes, &brain) {
                    let _ = async_saver.try_enqueue(bytes);
                }
            }
        }

        // Poll async save completion.
        if let Some(done) = async_saver.poll_done() {
            match done {
                Ok(n) => {
                    last_storage_event = format!(
                        "saved {} bytes to {}",
                        n,
                        storage_cfg.brain_image_path.display()
                    );
                    logger.log_event(frame, "Storage", &last_storage_event);
                }
                Err(e) => {
                    last_storage_event = format!(
                        "save write failed: {} (path={})",
                        e,
                        storage_cfg.brain_image_path.display()
                    );
                    logger.log_event(frame, "Storage", &last_storage_event);
                }
            }
        }

        // Poll async load completion.
        if let Some(done) = async_loader.poll_done() {
            match done {
                Ok(b) => {
                    brain = b;
                    brain.set_observer_telemetry(true);
                    last_storage_event =
                        format!("loaded from {}", storage_cfg.brain_image_path.display());
                    logger.log_event(frame, "Storage", &last_storage_event);
                }
                Err(e) => {
                    last_storage_event = format!(
                        "load failed: {} (path={})",
                        e,
                        storage_cfg.brain_image_path.display()
                    );
                    logger.log_event(frame, "Storage", &last_storage_event);
                }
            }
        }

        // Header bar (UI panel).
        draw_rectangle(
            0.0,
            0.0,
            screen_width(),
            TOP_UI_H,
            Color::new(0.07, 0.07, 0.09, 1.0),
        );
        draw_line(0.0, TOP_UI_H, screen_width(), TOP_UI_H, 1.0, DARKGRAY);

        // Test selector (left) + mode selector (right).
        let left_x = UI_MARGIN;
        let top_y = UI_MARGIN;

        let t1 = Rect::new(left_x, top_y, BTN_W_TEST, BTN_H);
        let t2 = Rect::new(left_x + BTN_W_TEST + UI_GAP, top_y, BTN_W_TEST, BTN_H);
        let t3 = Rect::new(
            left_x + 2.0 * (BTN_W_TEST + UI_GAP),
            top_y,
            BTN_W_TEST,
            BTN_H,
        );
        let t4 = Rect::new(
            left_x + 3.0 * (BTN_W_TEST + UI_GAP),
            top_y,
            BTN_W_TEST,
            BTN_H,
        );
        let t5 = Rect::new(
            left_x + 4.0 * (BTN_W_TEST + UI_GAP),
            top_y,
            BTN_W_TEST,
            BTN_H,
        );
        let t6 = Rect::new(
            left_x + 5.0 * (BTN_W_TEST + UI_GAP),
            top_y,
            BTN_W_TEST,
            BTN_H,
        );

        let right_group_w = 2.0 * BTN_W_MODE + UI_GAP;
        let right_x = screen_width() - UI_MARGIN - right_group_w;
        let m1 = Rect::new(right_x, top_y, BTN_W_MODE, BTN_H);
        let m2 = Rect::new(right_x + BTN_W_MODE + UI_GAP, top_y, BTN_W_MODE, BTN_H);

        // Second row controls (difficulty + regime + neg reinforcement).
        let row2_y = top_y + BTN_H + UI_GAP;
        let d1 = Rect::new(left_x, row2_y, BTN_W_SMALL, BTN_H);
        let d_status = Rect::new(left_x + BTN_W_SMALL + UI_GAP, row2_y, BTN_W_SMALL, BTN_H);
        let d2 = Rect::new(
            left_x + 2.0 * (BTN_W_SMALL + UI_GAP),
            row2_y,
            BTN_W_SMALL,
            BTN_H,
        );

        // Third row controls (more breathing room).
        let row3_y = row2_y + BTN_H + UI_GAP;

        let right3_group_w = 3.0 * BTN_W_SMALL + 2.0 * UI_GAP;
        let right3_x = screen_width() - UI_MARGIN - right3_group_w;
        let r3_1 = Rect::new(right3_x, row3_y, BTN_W_SMALL, BTN_H);
        let f_status = Rect::new(right3_x + BTN_W_SMALL + UI_GAP, row3_y, BTN_W_SMALL, BTN_H);
        let r3_2 = Rect::new(
            right3_x + 2.0 * (BTN_W_SMALL + UI_GAP),
            row3_y,
            BTN_W_SMALL,
            BTN_H,
        );

        let mid_group_w = BTN_W_SMALL;
        let mid_x = (screen_width() - mid_group_w) * 0.5;
        let negb = Rect::new(mid_x, row3_y, BTN_W_SMALL, BTN_H);

        if button(t1, "Pong", app.test == UiTestId::Pong) && app.test != UiTestId::Pong {
            app.test = UiTestId::Pong;
            pong = PongUi::new(&cfg);
            pong_child = PongChildManager::new();
            brain = mk_brain_for(app.test);
        }
        if button(t2, "Bandit", app.test == UiTestId::Bandit) && app.test != UiTestId::Bandit {
            app.test = UiTestId::Bandit;
            bandit = BanditUi::new();
            brain = mk_brain_for(app.test);
        }
        if button(t3, "Forage", app.test == UiTestId::Forage) && app.test != UiTestId::Forage {
            app.test = UiTestId::Forage;
            forage = games::forage::ForageUi::new(&cfg);
            brain = mk_brain_for(app.test);
        }
        if button(t4, "Whack", app.test == UiTestId::Whack) && app.test != UiTestId::Whack {
            app.test = UiTestId::Whack;
            whack = games::whack::WhackUi::new();
            brain = mk_brain_for(app.test);
        }
        if button(t5, "Beacon", app.test == UiTestId::Beacon) && app.test != UiTestId::Beacon {
            app.test = UiTestId::Beacon;
            beacon = games::beacon::BeaconUi::new(&cfg);
            brain = mk_brain_for(app.test);
        }
        if button(t6, "Seq", app.test == UiTestId::Sequence) && app.test != UiTestId::Sequence {
            app.test = UiTestId::Sequence;
            sequence = games::sequence::SequenceUi::new();
            brain = mk_brain_for(app.test);
        }
        if button(m1, "Human", app.mode == ControlMode::Human) {
            app.mode = ControlMode::Human;
        }
        if button(m2, "Braine", app.mode == ControlMode::Braine) {
            app.mode = ControlMode::Braine;
        }

        // Apply control only for Pong (Bandit is already fairly interpretable).
        if app.test == UiTestId::Pong {
            if button(d1, "Easier", false) {
                pong.easier();
            }
            draw_label(
                d_status,
                &format!("Difficulty {}", pong.difficulty_level),
                true,
            );
            if button(d2, "Harder", false) {
                pong.harder();
            }

            if button(negb, "Neg Reinforce", pong.neg_reinforce) {
                pong.neg_reinforce = !pong.neg_reinforce;
            }

            if button(r3_1, "Flip Slower", false) {
                pong.flip_slower();
            }
            draw_label(
                f_status,
                &format!(
                    "Flip {} (in {})",
                    pong.shift_every_outcomes,
                    pong.flip_countdown()
                ),
                true,
            );
            if button(r3_2, "Flip Faster", false) {
                pong.flip_faster();
            }
        }

        // Countdown to next flip in the header.
        let status_line = match app.test {
            UiTestId::Pong => format!(
                "Next flip in {} outcomes (every {})",
                pong.flip_countdown(),
                pong.shift_every_outcomes
            ),
            UiTestId::Bandit => {
                let m = if bandit.shift_every == 0 {
                    0
                } else {
                    let mm = bandit.t % bandit.shift_every;
                    if mm == 0 {
                        bandit.shift_every
                    } else {
                        bandit.shift_every - mm
                    }
                };
                format!("Next flip in {} steps (every {})", m, bandit.shift_every)
            }
            UiTestId::Forage => format!("Next flip in {} outcomes", forage.flip_countdown()),
            UiTestId::Whack => format!("Next flip in {} outcomes", whack.flip_countdown()),
            UiTestId::Beacon => format!("Next flip in {} hits", beacon.flip_countdown()),
            UiTestId::Sequence => format!("Next flip in {} outcomes", sequence.flip_countdown()),
        };
        let status_line = if last_storage_event.is_empty() {
            status_line
        } else {
            format!("{} | {}", status_line, last_storage_event)
        };
        // Keep the countdown visible, but tucked in the gap above the HUD panel.
        draw_text(&status_line, UI_MARGIN, HUD_START_Y - 10.0, 18.0, GRAY);

        let dt = get_frame_time();
        match app.test {
            UiTestId::Pong => pong.tick_and_render(
                &cfg,
                &app,
                &mut brain,
                &mut logger,
                &mut pong_child,
                frame,
                dt,
            ),
            UiTestId::Bandit => {
                bandit.tick_and_render(&cfg, &app, &mut brain, &mut logger, frame, dt)
            }
            UiTestId::Forage => {
                forage.tick_and_render(&cfg, &app, &mut brain, &mut logger, frame, dt)
            }
            UiTestId::Whack => {
                whack.tick_and_render(&cfg, &app, &mut brain, &mut logger, frame, dt)
            }
            UiTestId::Beacon => {
                beacon.tick_and_render(&cfg, &app, &mut brain, &mut logger, frame, dt)
            }
            UiTestId::Sequence => {
                sequence.tick_and_render(&cfg, &app, &mut brain, &mut logger, frame, dt)
            }
        }

        next_frame().await;
    }
}
