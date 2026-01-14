//! Braine Visualizer - Slint UI client
//! Connects to the `brained` daemon over TCP (127.0.0.1:9876)

use serde::{Deserialize, Serialize};
use serde_json::Value;
use slint::{ModelRc, Timer, TimerMode, VecModel};
use std::cell::RefCell;
use std::fs;
use std::io::{BufRead, BufReader, Write};
use std::net::TcpStream;
use std::path::PathBuf;
use std::rc::Rc;
use std::sync::{mpsc, Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};

const SAMPLE_REPLAY_DATASET_JSON: &str = include_str!("../../../data/replay/spot_lr_small.json");

fn parse_replay_dataset_json(json: &str) -> Result<Value, String> {
    serde_json::from_str::<Value>(json).map_err(|e| format!("Invalid replay dataset JSON: {e}"))
}

#[derive(Debug, Clone)]
struct GraphHoverNode {
    x01: f32,
    y01: f32,
    label: String,
    value: f32,
    domain: String,
}

#[derive(Debug, Clone)]
struct GraphHoverEdge {
    x1: f32,
    y1: f32,
    x2: f32,
    y2: f32,
    from_label: String,
    to_label: String,
    weight: f32,
}

fn point_segment_distance2(px: f32, py: f32, ax: f32, ay: f32, bx: f32, by: f32) -> f32 {
    let abx = bx - ax;
    let aby = by - ay;
    let apx = px - ax;
    let apy = py - ay;

    let denom = abx * abx + aby * aby;
    if denom <= 1e-12 {
        return apx * apx + apy * apy;
    }

    let t = ((apx * abx + apy * aby) / denom).clamp(0.0, 1.0);
    let cx = ax + t * abx;
    let cy = ay + t * aby;
    let dx = px - cx;
    let dy = py - cy;
    dx * dx + dy * dy
}

#[derive(Debug, Clone, Copy, Default)]
struct GraphPos {
    x01: f32,
    y01: f32,
}

fn clamp01(v: f32) -> f32 {
    v.clamp(0.0, 1.0)
}

fn clamp_pos(p: GraphPos) -> GraphPos {
    GraphPos {
        x01: p.x01.clamp(0.02, 0.98),
        y01: p.y01.clamp(0.02, 0.98),
    }
}

fn classify_domain(kind: &str, label: &str) -> String {
    if kind == "substrate" {
        return "unit".to_string();
    }

    // Causal/symbol graph
    let s = label.trim();
    if s.starts_with("pos_x_") {
        "pos_x".to_string()
    } else if s.starts_with("pos_y_") {
        "pos_y".to_string()
    } else if s.starts_with("pair::") {
        "pair".to_string()
    } else if s.starts_with("action::") {
        "action".to_string()
    } else if s.starts_with("spotxy_") {
        "spotxy".to_string()
    } else {
        "other".to_string()
    }
}

fn apply_view(p: GraphPos, zoom: f32, pan_x01: f32, pan_y01: f32) -> GraphPos {
    // Zoom around the center of the viewport, then pan in normalized space.
    let z = zoom.clamp(0.5, 4.0);
    GraphPos {
        x01: 0.5 + z * (p.x01 - 0.5) + pan_x01,
        y01: 0.5 + z * (p.y01 - 0.5) + pan_y01,
    }
}

fn init_pos_on_circle(i: usize, n: usize) -> GraphPos {
    if n == 0 {
        return GraphPos { x01: 0.5, y01: 0.5 };
    }
    let radius = 0.44f32;
    let denom = n as f32;
    let a = (i as f32 / denom) * std::f32::consts::TAU;
    clamp_pos(GraphPos {
        x01: 0.5 + radius * a.cos(),
        y01: 0.5 + radius * a.sin(),
    })
}

fn force_layout_step(
    ids: &[u32],
    edges: &[(usize, usize, f32)],
    pos: &mut [GraphPos],
    temperature: f32,
) {
    // Fruchterman-Reingold style forces.
    let n = ids.len();
    if n <= 1 {
        return;
    }

    let area = 1.0f32;
    let k = (area / n as f32).sqrt().max(1e-3);
    let k2 = k * k;

    let mut disp: Vec<(f32, f32)> = vec![(0.0, 0.0); n];

    // Repulsion (O(n^2), ok for <=128)
    for i in 0..n {
        for j in (i + 1)..n {
            let dx = pos[i].x01 - pos[j].x01;
            let dy = pos[i].y01 - pos[j].y01;
            let dist2 = (dx * dx + dy * dy).max(1e-6);
            let dist = dist2.sqrt();
            let f = k2 / dist;
            let fx = (dx / dist) * f;
            let fy = (dy / dist) * f;
            disp[i].0 += fx;
            disp[i].1 += fy;
            disp[j].0 -= fx;
            disp[j].1 -= fy;
        }
    }

    // Attraction along edges
    for &(a, b, w) in edges {
        let dx = pos[a].x01 - pos[b].x01;
        let dy = pos[a].y01 - pos[b].y01;
        let dist2 = (dx * dx + dy * dy).max(1e-6);
        let dist = dist2.sqrt();

        // Weight affects spring strength mildly.
        let s = (w.abs()).clamp(0.1, 5.0);
        let f = (dist2 / k).min(5.0) * s;
        let fx = (dx / dist) * f;
        let fy = (dy / dist) * f;

        disp[a].0 -= fx;
        disp[a].1 -= fy;
        disp[b].0 += fx;
        disp[b].1 += fy;
    }

    // Apply displacement with cooling
    for i in 0..n {
        let (dx, dy) = disp[i];
        let mag2 = dx * dx + dy * dy;
        if mag2 <= 1e-12 {
            continue;
        }
        let mag = mag2.sqrt();
        let step = temperature.min(mag);
        pos[i].x01 = clamp01(pos[i].x01 + (dx / mag) * step);
        pos[i].y01 = clamp01(pos[i].y01 + (dy / mag) * step);
    }

    // Keep away from borders a bit
    for p in pos.iter_mut() {
        *p = clamp_pos(*p);
    }
}

slint::include_modules!();

fn exec_tier_pref_path() -> Option<PathBuf> {
    // Minimal XDG config support without extra deps.
    // ~/.config/braine_desktop/exec_tier.txt
    let base = if let Ok(v) = std::env::var("XDG_CONFIG_HOME") {
        PathBuf::from(v)
    } else {
        let home = std::env::var("HOME").ok()?;
        PathBuf::from(home).join(".config")
    };
    Some(base.join("braine_desktop").join("exec_tier.txt"))
}

fn load_exec_tier_pref() -> Option<String> {
    let path = exec_tier_pref_path()?;
    let raw = std::fs::read_to_string(path).ok()?;
    let v = raw.trim().to_string();
    if v.is_empty() {
        None
    } else {
        Some(v)
    }
}

fn save_exec_tier_pref(tier: &str) {
    let Some(path) = exec_tier_pref_path() else {
        return;
    };
    if let Some(parent) = path.parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    let _ = std::fs::write(path, tier);
}

// ═══════════════════════════════════════════════════════════════════════════
// Protocol (mirrors brained daemon)
// ═══════════════════════════════════════════════════════════════════════════

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
enum Request {
    GetState,
    GetGameParams {
        game: String,
    },
    SetView {
        view: String,
    },
    Start,
    Stop,
    SetGame {
        game: String,
    },
    SetSpotXYEval {
        eval: bool,
    },
    SpotXYIncreaseGrid,
    SpotXYDecreaseGrid,
    SetMode {
        mode: String,
    },
    SetGameParam {
        game: String,
        key: String,
        value: f32,
    },
    SetMaxUnits {
        max_units: u32,
    },

    // Storage / snapshots
    SaveSnapshot,
    LoadSnapshot {
        stem: String,
    },

    // Experts (child brains)
    SetExpertsEnabled {
        enabled: bool,
    },
    SetExpertNesting {
        allow_nested: bool,
        max_depth: u32,
    },
    SetExpertPolicy {
        parent_learning: String,
        max_children: u32,
        child_reward_scale: f32,
        episode_trials: u32,
        consolidate_topk: u32,
        reward_shift_ema_delta_threshold: f32,
        performance_collapse_drop_threshold: f32,
        performance_collapse_baseline_min: f32,
        allow_nested: bool,
        max_depth: u32,
        persistence_mode: String,
    },
    CullExperts,
    HumanAction {
        action: String,
    },
    TriggerDream,
    TriggerBurst,
    TriggerSync,
    TriggerImprint,
    SaveBrain,
    LoadBrain,
    ResetBrain,

    MigrateStateFormat {
        target_state_version: u32,
    },
    SetFramerate {
        fps: u32,
    },
    SetTrialPeriodMs {
        ms: u32,
    },

    SetExecutionTier {
        tier: String,
    },

    // Advisor / LLM integration (bounded, slow loop)
    AdvisorContext {
        #[serde(default)]
        include_action_scores: bool,
    },
    AdvisorOnce {
        #[serde(default)]
        apply: bool,
    },

    // Replay dataset (dataset-driven evaluation)
    ReplaySetDataset {
        dataset: Value,
    },
    GetGraph {
        kind: String,
        max_nodes: u32,
        max_edges: u32,
        include_isolated: bool,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
enum Response {
    State(Box<DaemonStateSnapshot>),
    GameParams {
        game: String,
        params: Vec<GameParamDef>,
    },
    AdvisorContext {
        context: Value,
        #[serde(default)]
        action_scores: Vec<DaemonActionScore>,
    },
    AdvisorReport {
        report: Value,
        #[serde(default)]
        applied: bool,
    },
    Success {
        message: String,
    },
    Graph(Box<DaemonGraphSnapshot>),
    Error {
        message: String,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct GameParamDef {
    key: String,
    label: String,
    #[serde(default)]
    description: String,
    min: f32,
    max: f32,
    default: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
struct DaemonGraphSnapshot {
    #[serde(default)]
    kind: String,
    #[serde(default)]
    nodes: Vec<DaemonGraphNode>,
    #[serde(default)]
    edges: Vec<DaemonGraphEdge>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
struct DaemonGraphNode {
    id: u32,
    #[serde(default)]
    label: String,
    #[serde(default)]
    value: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
struct DaemonGraphEdge {
    from: u32,
    to: u32,
    #[serde(default)]
    weight: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
struct DaemonStateSnapshot {
    running: bool,
    mode: String,
    frame: u64,
    #[serde(default)]
    last_error: String,
    #[serde(default)]
    target_fps: u32,
    game: DaemonGameState,
    hud: DaemonHudData,
    brain_stats: DaemonBrainStats,
    #[serde(default)]
    unit_plot: Vec<DaemonUnitPlotPoint>,
    #[serde(default)]
    action_scores: Vec<DaemonActionScore>,
    #[serde(default)]
    meaning: DaemonMeaningSnapshot,

    // Experts (child brains)
    #[serde(default)]
    experts_enabled: bool,
    #[serde(default)]
    experts: DaemonExpertsSummary,
    #[serde(default)]
    active_expert: Option<DaemonActiveExpertSummary>,

    // Storage / snapshots
    #[serde(default)]
    storage: DaemonStorageInfo,

    // Advisor / LLM integration (optional)
    #[serde(default)]
    advisor: DaemonAdvisorSnapshot,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
struct DaemonAdvisorSnapshot {
    #[serde(default)]
    enabled: bool,
    #[serde(default)]
    mode: String,
    #[serde(default)]
    every_trials: u32,
    #[serde(default)]
    last_rationale: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
struct DaemonStorageInfo {
    #[serde(default)]
    data_dir: String,
    #[serde(default)]
    brain_file: String,
    #[serde(default)]
    runtime_file: String,
    #[serde(default)]
    loaded_snapshot: String,
    #[serde(default)]
    brain_bytes: u64,
    #[serde(default)]
    runtime_bytes: u64,

    #[serde(default)]
    state_wrapper_version: u32,
    #[serde(default)]
    snapshots: Vec<DaemonSnapshotEntry>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
struct DaemonSnapshotEntry {
    #[serde(default)]
    stem: String,
    #[serde(default)]
    brain_bytes: u64,
    #[serde(default)]
    runtime_bytes: u64,
    #[serde(default)]
    modified_unix: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
struct DaemonExpertsSummary {
    #[serde(default)]
    active_count: u32,
    #[serde(default)]
    total_active_count: u32,
    #[serde(default)]
    max_children: u32,
    #[serde(default)]
    last_spawn_reason: String,
    #[serde(default)]
    last_consolidation: String,

    #[serde(default)]
    persistence_mode: String,
    #[serde(default)]
    allow_nested: bool,
    #[serde(default)]
    max_depth: u32,

    #[serde(default)]
    reward_shift_ema_delta_threshold: f32,
    #[serde(default)]
    performance_collapse_drop_threshold: f32,
    #[serde(default)]
    performance_collapse_baseline_min: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct DaemonActiveExpertSummary {
    id: u32,
    #[serde(default)]
    context_key: String,
    #[serde(default)]
    age_steps: u64,
    #[serde(default)]
    reward_ema: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
struct DaemonRewardEdges {
    #[serde(default)]
    to_reward_pos: f32,
    #[serde(default)]
    to_reward_neg: f32,
    #[serde(default)]
    meaning: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
struct DaemonMeaningSnapshot {
    #[serde(default)]
    stimulus: String,
    #[serde(default)]
    correct_action: String,

    #[serde(default)]
    action_a_name: String,
    #[serde(default)]
    action_b_name: String,

    #[serde(default)]
    pair_left: DaemonRewardEdges,
    #[serde(default)]
    pair_right: DaemonRewardEdges,

    #[serde(default)]
    action_left: DaemonRewardEdges,
    #[serde(default)]
    action_right: DaemonRewardEdges,

    #[serde(default)]
    pair_gap: f32,
    #[serde(default)]
    global_gap: f32,

    #[serde(default)]
    pair_gap_history: Vec<f32>,
    #[serde(default)]
    global_gap_history: Vec<f32>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
struct DaemonActionScore {
    name: String,
    habit_norm: f32,
    meaning_global: f32,
    meaning_conditional: f32,
    meaning: f32,
    score: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
struct DaemonUnitPlotPoint {
    id: u32,
    amp: f32,
    #[serde(default)]
    amp01: f32,
    rel_age: f32,
    is_reserved: bool,
    is_sensor_member: bool,
    is_group_member: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
struct DaemonGameCommon {
    #[serde(default)]
    reversal_active: bool,
    #[serde(default)]
    chosen_action: String,
    #[serde(default)]
    last_reward: f32,
    #[serde(default)]
    response_made: bool,
    #[serde(default)]
    trial_frame: u32,
    #[serde(default)]
    trial_duration: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
#[serde(tag = "kind")]
enum DaemonGameState {
    #[serde(rename = "spot")]
    Spot {
        #[serde(flatten)]
        common: DaemonGameCommon,
        spot_is_left: bool,
    },
    #[serde(rename = "bandit")]
    Bandit {
        #[serde(flatten)]
        common: DaemonGameCommon,
    },
    #[serde(rename = "spot_reversal")]
    SpotReversal {
        #[serde(flatten)]
        common: DaemonGameCommon,
        spot_is_left: bool,
    },
    #[serde(rename = "spotxy")]
    SpotXY {
        #[serde(flatten)]
        common: DaemonGameCommon,
        pos_x: f32,
        pos_y: f32,
        #[serde(default)]
        spotxy_eval: bool,
        #[serde(default)]
        spotxy_mode: String,
        #[serde(default)]
        spotxy_grid_n: u32,
    },
    #[serde(rename = "pong")]
    Pong {
        #[serde(flatten)]
        common: DaemonGameCommon,
        #[serde(default)]
        pong_ball_x: f32,
        #[serde(default)]
        pong_ball_y: f32,
        #[serde(default)]
        pong_ball_visible: bool,
        #[serde(default)]
        pong_ball2_x: f32,
        #[serde(default)]
        pong_ball2_y: f32,
        #[serde(default)]
        pong_ball2_visible: bool,
        #[serde(default)]
        pong_ball2_enabled: bool,
        pong_paddle_y: f32,
        #[serde(default)]
        pong_paddle_half_height: f32,
        #[serde(default)]
        pong_paddle_speed: f32,
        #[serde(default)]
        pong_ball_speed: f32,
    },
    #[serde(rename = "text")]
    Text {
        #[serde(flatten)]
        common: DaemonGameCommon,
        #[serde(default)]
        text_regime: u32,
        #[serde(default)]
        text_token: String,
        #[serde(default)]
        text_target_next: String,
        #[serde(default)]
        text_outcomes: u32,
        #[serde(default)]
        text_shift_every: u32,
        #[serde(default)]
        text_vocab_size: u32,
    },
    #[serde(rename = "replay")]
    Replay {
        #[serde(flatten)]
        common: DaemonGameCommon,
        #[serde(default)]
        replay_dataset: String,
        #[serde(default)]
        replay_index: u32,
        #[serde(default)]
        replay_total: u32,
        #[serde(default)]
        replay_trial_id: String,
    },
    #[serde(other)]
    #[default]
    Unknown,
}

impl DaemonGameState {
    fn kind(&self) -> &'static str {
        match self {
            Self::Spot { .. } => "spot",
            Self::Bandit { .. } => "bandit",
            Self::SpotReversal { .. } => "spot_reversal",
            Self::SpotXY { .. } => "spotxy",
            Self::Pong { .. } => "pong",
            Self::Text { .. } => "text",
            Self::Replay { .. } => "replay",
            Self::Unknown => "unknown",
        }
    }

    fn common(&self) -> Option<&DaemonGameCommon> {
        match self {
            Self::Spot { common, .. }
            | Self::Bandit { common }
            | Self::SpotReversal { common, .. }
            | Self::SpotXY { common, .. }
            | Self::Pong { common, .. }
            | Self::Text { common, .. }
            | Self::Replay { common, .. } => Some(common),
            Self::Unknown => None,
        }
    }

    fn reversal_active(&self) -> bool {
        self.common().map(|c| c.reversal_active).unwrap_or(false)
    }

    fn chosen_action(&self) -> &str {
        self.common()
            .map(|c| c.chosen_action.as_str())
            .unwrap_or("")
    }

    fn last_reward(&self) -> f32 {
        self.common().map(|c| c.last_reward).unwrap_or(0.0)
    }

    fn response_made(&self) -> bool {
        self.common().map(|c| c.response_made).unwrap_or(false)
    }

    fn trial_frame(&self) -> u32 {
        self.common().map(|c| c.trial_frame).unwrap_or(0)
    }

    fn trial_duration(&self) -> u32 {
        self.common().map(|c| c.trial_duration).unwrap_or(0)
    }

    fn spot_is_left(&self) -> bool {
        match self {
            Self::Spot { spot_is_left, .. } | Self::SpotReversal { spot_is_left, .. } => {
                *spot_is_left
            }
            _ => false,
        }
    }

    fn pos_xy(&self) -> (f32, f32) {
        match self {
            Self::SpotXY { pos_x, pos_y, .. } => (*pos_x, *pos_y),
            // Pong sim uses x in [0,1] and y in [-1,1]. Map x to [-1,1] for the canvas.
            Self::Pong {
                pong_ball_x,
                pong_ball_y,
                ..
            } => (pong_ball_x * 2.0 - 1.0, *pong_ball_y),
            _ => (0.0, 0.0),
        }
    }

    fn pong_ball2_pos_xy(&self) -> (f32, f32) {
        match self {
            Self::Pong {
                pong_ball2_x,
                pong_ball2_y,
                pong_ball2_enabled,
                ..
            } if *pong_ball2_enabled => (pong_ball2_x * 2.0 - 1.0, *pong_ball2_y),
            _ => (0.0, 0.0),
        }
    }

    fn spotxy_eval(&self) -> bool {
        match self {
            Self::SpotXY { spotxy_eval, .. } => *spotxy_eval,
            _ => false,
        }
    }

    fn spotxy_mode(&self) -> &str {
        match self {
            Self::SpotXY { spotxy_mode, .. } => spotxy_mode.as_str(),
            _ => "",
        }
    }

    fn spotxy_grid_n(&self) -> u32 {
        match self {
            Self::SpotXY { spotxy_grid_n, .. } => *spotxy_grid_n,
            _ => 0,
        }
    }

    fn pong_params(&self) -> (f32, f32, f32, f32) {
        match self {
            Self::Pong {
                pong_paddle_y,
                pong_paddle_half_height,
                pong_paddle_speed,
                pong_ball_speed,
                ..
            } => (
                *pong_paddle_y,
                *pong_paddle_half_height,
                *pong_paddle_speed,
                *pong_ball_speed,
            ),
            _ => (0.0, 0.25, 1.3, 1.0),
        }
    }

    fn pong_ball_visible(&self) -> bool {
        match self {
            Self::Pong {
                pong_ball_visible, ..
            } => *pong_ball_visible,
            _ => true,
        }
    }

    fn pong_ball2_visible(&self) -> bool {
        match self {
            Self::Pong {
                pong_ball2_visible,
                pong_ball2_enabled,
                ..
            } => *pong_ball2_enabled && *pong_ball2_visible,
            _ => false,
        }
    }

    fn pong_ball2_enabled(&self) -> bool {
        match self {
            Self::Pong {
                pong_ball2_enabled, ..
            } => *pong_ball2_enabled,
            _ => false,
        }
    }

    fn text_regime(&self) -> u32 {
        match self {
            Self::Text { text_regime, .. } => *text_regime,
            _ => 0,
        }
    }

    fn text_token(&self) -> &str {
        match self {
            Self::Text { text_token, .. } => text_token.as_str(),
            _ => "",
        }
    }

    fn text_target_next(&self) -> &str {
        match self {
            Self::Text {
                text_target_next, ..
            } => text_target_next.as_str(),
            _ => "",
        }
    }

    fn text_outcomes(&self) -> u32 {
        match self {
            Self::Text { text_outcomes, .. } => *text_outcomes,
            _ => 0,
        }
    }

    fn text_shift_every(&self) -> u32 {
        match self {
            Self::Text {
                text_shift_every, ..
            } => *text_shift_every,
            _ => 0,
        }
    }

    fn text_vocab_size(&self) -> u32 {
        match self {
            Self::Text {
                text_vocab_size, ..
            } => *text_vocab_size,
            _ => 0,
        }
    }

    fn replay_dataset(&self) -> &str {
        match self {
            Self::Replay { replay_dataset, .. } => replay_dataset.as_str(),
            _ => "",
        }
    }

    fn replay_index(&self) -> u32 {
        match self {
            Self::Replay { replay_index, .. } => *replay_index,
            _ => 0,
        }
    }

    fn replay_total(&self) -> u32 {
        match self {
            Self::Replay { replay_total, .. } => *replay_total,
            _ => 0,
        }
    }

    fn replay_trial_id(&self) -> &str {
        match self {
            Self::Replay {
                replay_trial_id, ..
            } => replay_trial_id.as_str(),
            _ => "",
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
struct DaemonHudData {
    trials: u32,
    correct: u32,
    incorrect: u32,
    accuracy: f32,
    recent_rate: f32,
    last_100_rate: f32,
    neuromod: f32,
    #[serde(default)]
    learning_at_trial: i32,
    #[serde(default)]
    learned_at_trial: i32,
    #[serde(default)]
    mastered_at_trial: i32,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
struct DaemonBrainStats {
    unit_count: usize,
    #[serde(default)]
    max_units_limit: usize,
    #[serde(default)]
    execution_tier: String,
    #[serde(default)]
    execution_tier_selected: String,
    #[serde(default)]
    execution_tier_effective: String,
    connection_count: usize,
    #[serde(default)]
    pruned_last_step: usize,
    #[serde(default)]
    births_last_step: usize,
    #[serde(default)]
    saturated: bool,
    avg_amp: f32,
    avg_weight: f32,
    #[serde(default)]
    osc_x: f32,
    #[serde(default)]
    osc_y: f32,
    #[serde(default)]
    osc_mag: f32,
    memory_bytes: usize,
    #[serde(default)]
    causal_base_symbols: usize,
    causal_edges: usize,
    #[serde(default)]
    causal_last_directed_edge_updates: usize,
    #[serde(default)]
    causal_last_cooccur_edge_updates: usize,
    age_steps: u64,
}

// ═══════════════════════════════════════════════════════════════════════════
// Daemon client
// ═══════════════════════════════════════════════════════════════════════════

struct DaemonClient {
    tx: mpsc::Sender<Request>,
    snapshot: Arc<Mutex<DaemonStateSnapshot>>, // latest state from daemon
    graph: Arc<Mutex<DaemonGraphSnapshot>>,
    game_params: Arc<Mutex<Vec<GameParamDef>>>,
    advisor_context_json: Arc<Mutex<String>>,
    advisor_report_json: Arc<Mutex<String>>,
    advisor_status: Arc<Mutex<String>>,
    system_error: Arc<Mutex<String>>,
}

impl DaemonClient {
    fn new() -> Self {
        let (tx, rx) = mpsc::channel::<Request>();
        let snapshot = Arc::new(Mutex::new(DaemonStateSnapshot::default()));
        let graph = Arc::new(Mutex::new(DaemonGraphSnapshot::default()));
        let game_params = Arc::new(Mutex::new(Vec::<GameParamDef>::new()));
        let advisor_context_json = Arc::new(Mutex::new(String::new()));
        let advisor_report_json = Arc::new(Mutex::new(String::new()));
        let advisor_status = Arc::new(Mutex::new(String::new()));
        let system_error = Arc::new(Mutex::new(String::new()));
        let snap_clone = Arc::clone(&snapshot);
        let graph_clone = Arc::clone(&graph);
        let params_clone = Arc::clone(&game_params);
        let advisor_ctx_clone = Arc::clone(&advisor_context_json);
        let advisor_report_clone = Arc::clone(&advisor_report_json);
        let advisor_status_clone = Arc::clone(&advisor_status);
        let system_error_clone = Arc::clone(&system_error);

        // Background worker: manages TCP connection and request/response loop
        thread::spawn(move || loop {
            match TcpStream::connect("127.0.0.1:9876") {
                Ok(mut stream) => {
                    if let Ok(mut s) = system_error_clone.lock() {
                        *s = "".to_string();
                    }
                    stream.set_read_timeout(Some(Duration::from_secs(5))).ok();
                    let mut reader = BufReader::new(stream.try_clone().unwrap());
                    loop {
                        let req = match rx.recv() {
                            Ok(r) => r,
                            Err(_) => return, // channel closed
                        };
                        // Send request
                        if let Ok(line) = serde_json::to_string(&req) {
                            if stream.write_all(line.as_bytes()).is_err() {
                                break;
                            }
                            if stream.write_all(b"\n").is_err() {
                                break;
                            }
                        }
                        // Read response
                        let mut resp_line = String::new();
                        if reader.read_line(&mut resp_line).is_err() {
                            break;
                        }
                        if resp_line.trim().is_empty() {
                            continue;
                        }
                        match serde_json::from_str::<Response>(&resp_line) {
                            Ok(Response::State(state)) => {
                                if let Ok(mut s) = snap_clone.lock() {
                                    *s = *state;
                                }
                            }
                            Ok(Response::GameParams { game: _, params }) => {
                                if let Ok(mut p) = params_clone.lock() {
                                    *p = params;
                                }
                            }
                            Ok(Response::AdvisorContext {
                                context,
                                action_scores,
                            }) => {
                                let mut root = serde_json::Map::new();
                                root.insert("context".to_string(), context);
                                root.insert(
                                    "action_scores".to_string(),
                                    serde_json::to_value(action_scores).unwrap_or(Value::Null),
                                );
                                let v = Value::Object(root);
                                if let Ok(mut s) = advisor_ctx_clone.lock() {
                                    *s = serde_json::to_string_pretty(&v)
                                        .unwrap_or_else(|_| "{...}".to_string());
                                }
                                if let Ok(mut s) = advisor_status_clone.lock() {
                                    *s = "AdvisorContext received".to_string();
                                }
                            }
                            Ok(Response::AdvisorReport { report, applied }) => {
                                let mut root = serde_json::Map::new();
                                root.insert("applied".to_string(), Value::Bool(applied));
                                root.insert("report".to_string(), report);
                                let v = Value::Object(root);
                                if let Ok(mut s) = advisor_report_clone.lock() {
                                    *s = serde_json::to_string_pretty(&v)
                                        .unwrap_or_else(|_| "{...}".to_string());
                                }
                                if let Ok(mut s) = advisor_status_clone.lock() {
                                    *s = if applied {
                                        "AdvisorReport received (applied)".to_string()
                                    } else {
                                        "AdvisorReport received".to_string()
                                    };
                                }
                            }
                            Ok(Response::Graph(g)) => {
                                if let Ok(mut gs) = graph_clone.lock() {
                                    *gs = *g;
                                }
                            }
                            Ok(Response::Success { message }) => {
                                if let Ok(mut s) = advisor_status_clone.lock() {
                                    *s = message;
                                }
                                if let Ok(mut s) = system_error_clone.lock() {
                                    *s = "".to_string();
                                }
                            }
                            Ok(Response::Error { message }) => {
                                eprintln!("Daemon error: {}", message);
                                if let Ok(mut s) = advisor_status_clone.lock() {
                                    *s = format!("Error: {}", message);
                                }
                                if let Ok(mut s) = system_error_clone.lock() {
                                    *s = message;
                                }
                            }
                            Err(e) => eprintln!("Bad response: {}", e),
                        }
                    }
                }
                Err(e) => {
                    eprintln!("Unable to connect to brained: {}. Retrying in 1s...", e);
                    if let Ok(mut s) = system_error_clone.lock() {
                        *s = format!("Daemon unreachable: {e}");
                    }
                    thread::sleep(Duration::from_secs(1));
                }
            }
        });

        Self {
            tx,
            snapshot,
            graph,
            game_params,
            advisor_context_json,
            advisor_report_json,
            advisor_status,
            system_error,
        }
    }

    fn send(&self, req: Request) {
        let _ = self.tx.send(req);
    }

    fn snapshot(&self) -> DaemonStateSnapshot {
        self.snapshot.lock().map(|s| s.clone()).unwrap_or_default()
    }

    fn graph_snapshot(&self) -> DaemonGraphSnapshot {
        self.graph.lock().map(|s| s.clone()).unwrap_or_default()
    }

    fn game_params(&self) -> Vec<GameParamDef> {
        self.game_params
            .lock()
            .map(|s| s.clone())
            .unwrap_or_default()
    }

    fn advisor_context_json(&self) -> String {
        self.advisor_context_json
            .lock()
            .map(|s| s.clone())
            .unwrap_or_default()
    }

    fn advisor_report_json(&self) -> String {
        self.advisor_report_json
            .lock()
            .map(|s| s.clone())
            .unwrap_or_default()
    }

    fn advisor_status(&self) -> String {
        self.advisor_status
            .lock()
            .map(|s| s.clone())
            .unwrap_or_default()
    }

    fn system_error(&self) -> String {
        self.system_error
            .lock()
            .map(|s| s.clone())
            .unwrap_or_default()
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Main
// ═══════════════════════════════════════════════════════════════════════════

fn main() -> Result<(), slint::PlatformError> {
    let ui = MainWindow::new()?;
    let client = Rc::new(DaemonClient::new());

    // Apply persisted execution-tier preference early (best-effort).
    if let Some(tier) = load_exec_tier_pref() {
        client.send(Request::SetExecutionTier { tier });
    }

    // Prime knob schema so Game Settings uses daemon-defined ranges.
    client.send(Request::GetGameParams {
        game: ui.get_selected_game_kind().to_string(),
    });

    // Track local control changes so the next poll doesn't immediately overwrite
    // the slider position before the daemon applies the request.
    let pending_fps: Rc<RefCell<Option<(u32, Instant)>>> = Rc::new(RefCell::new(None));
    let pending_trial_ms: Rc<RefCell<Option<(u32, Instant)>>> = Rc::new(RefCell::new(None));
    let last_graph_req: Rc<RefCell<Instant>> =
        Rc::new(RefCell::new(Instant::now() - Duration::from_secs(3600)));

    // Updated whenever we re-render the graph; used for hover hit-testing.
    let graph_hover_nodes: Rc<RefCell<Vec<GraphHoverNode>>> = Rc::new(RefCell::new(Vec::new()));
    let graph_hover_edges: Rc<RefCell<Vec<GraphHoverEdge>>> = Rc::new(RefCell::new(Vec::new()));

    // Persisted node positions across refreshes so the force layout is stable.
    // Keep separate caches per graph kind (substrate vs causal).
    let graph_pos_by_kind: Rc<
        RefCell<std::collections::HashMap<String, std::collections::HashMap<u32, GraphPos>>>,
    > = Rc::new(RefCell::new(std::collections::HashMap::new()));

    // Force layout "temperature" per kind to avoid perpetual motion/flicker.
    let graph_temp_by_kind: Rc<RefCell<std::collections::HashMap<String, f32>>> =
        Rc::new(RefCell::new(std::collections::HashMap::new()));

    // Initial UI data
    ui.set_learning(LearningState {
        learning_enabled: true,
        attention_boost: 0.5,
        attention_enabled: false,
        burst_mode_enabled: false,
        auto_dream_on_flip: false,
        auto_burst_on_slump: false,
        prediction_enabled: false,
        prediction_weight: 0.0,
    });

    // Callbacks to daemon
    {
        let c = client.clone();
        ui.on_mode_changed(move |is_braine| {
            c.send(Request::SetMode {
                mode: if is_braine { "braine" } else { "human" }.to_string(),
            });
        });
    }

    // Graph hover (hit-test nodes/edges and show tooltip).
    {
        let ui_weak = ui.as_weak();
        let nodes = graph_hover_nodes.clone();
        let edges = graph_hover_edges.clone();

        ui.on_graph_hover(move |x01, y01| {
            let Some(ui) = ui_weak.upgrade() else {
                return;
            };

            let nodes = nodes.borrow();
            let edges = edges.borrow();
            if nodes.is_empty() && edges.is_empty() {
                ui.set_graph_hover_visible(false);
                return;
            }

            // Nearest node
            let mut best_node_i: Option<usize> = None;
            let mut best_node_d2: f32 = f32::INFINITY;
            for (i, n) in nodes.iter().enumerate() {
                let dx = n.x01 - x01;
                let dy = n.y01 - y01;
                let d2 = dx * dx + dy * dy;
                if d2 < best_node_d2 {
                    best_node_d2 = d2;
                    best_node_i = Some(i);
                }
            }

            // Nearest edge (distance to segment)
            let mut best_edge_i: Option<usize> = None;
            let mut best_edge_d2: f32 = f32::INFINITY;
            for (i, e) in edges.iter().enumerate() {
                let d2 = point_segment_distance2(x01, y01, e.x1, e.y1, e.x2, e.y2);
                if d2 < best_edge_d2 {
                    best_edge_d2 = d2;
                    best_edge_i = Some(i);
                }
            }

            // Thresholds (normalized coordinates)
            let max_node_r = 0.030f32;
            let max_edge_r = 0.020f32;

            let node_hit = best_node_i.is_some() && best_node_d2 <= max_node_r * max_node_r;
            let edge_hit = best_edge_i.is_some() && best_edge_d2 <= max_edge_r * max_edge_r;

            if node_hit && (!edge_hit || best_node_d2 <= best_edge_d2) {
                let n = &nodes[best_node_i.unwrap()];
                let label = if n.label.trim().is_empty() {
                    "(unlabeled)".to_string()
                } else {
                    n.label.clone()
                };
                let text = format!("{}  [{}]  value={:.3}", label, n.domain, n.value);
                ui.set_graph_hover_text(text.into());
                ui.set_graph_hover_x01(x01);
                ui.set_graph_hover_y01(y01);
                ui.set_graph_hover_visible(true);
                return;
            }

            if edge_hit {
                let e = &edges[best_edge_i.unwrap()];
                let arrow = "->";
                let text = format!(
                    "{} {} {}  w={:.4}",
                    e.from_label.trim(),
                    arrow,
                    e.to_label.trim(),
                    e.weight
                );
                ui.set_graph_hover_text(text.into());
                ui.set_graph_hover_x01(x01);
                ui.set_graph_hover_y01(y01);
                ui.set_graph_hover_visible(true);
                return;
            }

            ui.set_graph_hover_visible(false);
        });
    }

    {
        let c = client.clone();
        ui.on_running_changed(move |is_running| {
            if is_running {
                c.send(Request::Start);
            } else {
                c.send(Request::Stop);
            }
        });
    }

    // Game selection (stop required; daemon enforces too)
    {
        let c = client.clone();
        ui.on_game_changed(move |game| {
            c.send(Request::SetGame {
                game: game.to_string(),
            });

            c.send(Request::GetGameParams {
                game: game.to_string(),
            });
        });
    }

    // Game parameter changes
    {
        let c = client.clone();
        ui.on_set_game_param(move |game, key, value| {
            c.send(Request::SetGameParam {
                game: game.to_string(),
                key: key.to_string(),
                value,
            });
        });
    }

    // View toggle: parent vs active expert
    {
        let c = client.clone();
        let ui_weak = ui.as_weak();
        ui.on_view_mode_changed(move |mode| {
            c.send(Request::SetView {
                view: mode.to_string(),
            });

            // Switching view changes what the daemon serves for plots/graphs.
            // Trigger an immediate graph refresh so the UI reflects the new view.
            if let Some(ui) = ui_weak.upgrade() {
                c.send(Request::GetGraph {
                    kind: ui.get_graph_kind().to_string(),
                    max_nodes: ui.get_graph_max_nodes().max(1) as u32,
                    max_edges: ui.get_graph_max_edges().max(0) as u32,
                    include_isolated: ui.get_graph_include_isolated(),
                });
            }
        });
    }

    // Max units limit (neurogenesis cap)
    {
        let c = client.clone();
        ui.on_set_max_units(move |max_units| {
            let max_units = max_units.max(0) as u32;
            c.send(Request::SetMaxUnits { max_units });
        });
    }

    // Snapshot management
    {
        let c = client.clone();
        ui.on_save_snapshot(move || {
            c.send(Request::SaveSnapshot);
        });
    }
    {
        let c = client.clone();
        ui.on_load_snapshot(move |stem| {
            c.send(Request::LoadSnapshot {
                stem: stem.to_string(),
            });
        });
    }

    // Replay dataset controls
    {
        let c = client.clone();
        let ui_weak = ui.as_weak();
        ui.on_replay_load_dataset(move |path| {
            let Some(ui) = ui_weak.upgrade() else {
                return;
            };

            if ui.get_running() {
                ui.set_replay_dataset_status(
                    "Stop the simulation before loading a replay dataset".into(),
                );
                return;
            }

            let path = path.trim().to_string();
            if path.is_empty() {
                ui.set_replay_dataset_status("Enter a dataset JSON path".into());
                return;
            }

            let json = match fs::read_to_string(&path) {
                Ok(s) => s,
                Err(e) => {
                    ui.set_replay_dataset_status(format!("Failed to read '{path}': {e}").into());
                    return;
                }
            };

            let dataset = match parse_replay_dataset_json(&json) {
                Ok(v) => v,
                Err(msg) => {
                    ui.set_replay_dataset_status(msg.into());
                    return;
                }
            };

            c.send(Request::ReplaySetDataset { dataset });
            ui.set_replay_dataset_status("Dataset sent to daemon (ReplaySetDataset)".into());
        });
    }
    {
        let c = client.clone();
        let ui_weak = ui.as_weak();
        ui.on_replay_load_sample(move || {
            let Some(ui) = ui_weak.upgrade() else {
                return;
            };

            if ui.get_running() {
                ui.set_replay_dataset_status(
                    "Stop the simulation before loading a replay dataset".into(),
                );
                return;
            }

            let dataset = match parse_replay_dataset_json(SAMPLE_REPLAY_DATASET_JSON) {
                Ok(v) => v,
                Err(msg) => {
                    ui.set_replay_dataset_status(msg.into());
                    return;
                }
            };

            c.send(Request::ReplaySetDataset { dataset });
            ui.set_replay_dataset_path("(built-in) spot_lr_small".into());
            ui.set_replay_dataset_status("Sample dataset sent to daemon".into());
        });
    }

    // Advisor / LLM integration controls
    {
        let c = client.clone();
        let ui_weak = ui.as_weak();
        ui.on_advisor_fetch_context(move |include_scores| {
            if let Some(ui) = ui_weak.upgrade() {
                ui.set_advisor_status("Requesting AdvisorContext...".into());
            }
            c.send(Request::AdvisorContext {
                include_action_scores: include_scores,
            });
        });
    }
    {
        let c = client.clone();
        let ui_weak = ui.as_weak();
        ui.on_advisor_once(move |apply| {
            if let Some(ui) = ui_weak.upgrade() {
                ui.set_advisor_status(if apply {
                    "Requesting AdvisorOnce (apply=true)...".into()
                } else {
                    "Requesting AdvisorOnce (apply=false)...".into()
                });
            }
            c.send(Request::AdvisorOnce { apply });
        });
    }

    // Experts (child brains)
    {
        let c_enabled = client.clone();
        ui.on_experts_enabled_changed(move |enabled| {
            c_enabled.send(Request::SetExpertsEnabled { enabled });
        });

        let c_nesting = client.clone();
        ui.on_experts_nesting_changed(move |allow_nested, max_depth| {
            c_nesting.send(Request::SetExpertNesting {
                allow_nested,
                max_depth: (max_depth.max(1) as u32),
            });
        });

        let c_policy = client.clone();
        ui.on_experts_policy_apply(
            move |parent_learning,
                  max_children,
                  child_reward_scale,
                  episode_trials,
                  consolidate_topk,
                  reward_shift_ema_delta_threshold,
                  performance_collapse_drop_threshold,
                  performance_collapse_baseline_min,
                  persistence_mode,
                  allow_nested,
                  max_depth| {
                c_policy.send(Request::SetExpertPolicy {
                    parent_learning: parent_learning.to_string(),
                    max_children: (max_children.max(0) as u32),
                    child_reward_scale,
                    episode_trials: (episode_trials.max(1) as u32),
                    consolidate_topk: (consolidate_topk.max(0) as u32),
                    reward_shift_ema_delta_threshold,
                    performance_collapse_drop_threshold,
                    performance_collapse_baseline_min,
                    allow_nested,
                    max_depth: (max_depth.max(1) as u32),
                    persistence_mode: persistence_mode.to_string(),
                });
            },
        );
    }
    {
        let c = client.clone();
        ui.on_cull_experts(move || {
            c.send(Request::CullExperts);
        });
    }

    // SpotXY eval toggle
    {
        let c = client.clone();
        ui.on_spotxy_eval_changed(move |eval| {
            c.send(Request::SetSpotXYEval { eval });
        });
    }

    // SpotXY grid progression
    {
        let c = client.clone();
        ui.on_spotxy_increase_grid(move || {
            c.send(Request::SpotXYIncreaseGrid);
        });
    }

    // SpotXY grid decrease
    {
        let c = client.clone();
        ui.on_spotxy_decrease_grid(move || {
            c.send(Request::SpotXYDecreaseGrid);
        });
    }

    // Graph request
    {
        let c = client.clone();
        let last = last_graph_req.clone();
        ui.on_graph_request(move |kind, max_nodes, max_edges, include_isolated| {
            *last.borrow_mut() = Instant::now();
            c.send(Request::GetGraph {
                kind: kind.to_string(),
                max_nodes: (max_nodes.max(1) as u32),
                max_edges: (max_edges.max(0) as u32),
                include_isolated,
            });
        });
    }

    // Learning controls
    {
        let c = client.clone();
        ui.on_trigger_dream(move || c.send(Request::TriggerDream));
    }
    {
        let c = client.clone();
        ui.on_trigger_burst(move || c.send(Request::TriggerBurst));
    }
    {
        let c = client.clone();
        ui.on_trigger_sync(move || c.send(Request::TriggerSync));
    }
    {
        let c = client.clone();
        ui.on_trigger_imprint(move || c.send(Request::TriggerImprint));
    }

    // Storage
    {
        let c = client.clone();
        ui.on_save_brain(move || c.send(Request::SaveBrain));
    }
    {
        let c = client.clone();
        ui.on_load_brain(move || c.send(Request::LoadBrain));
    }
    {
        let c = client.clone();
        ui.on_reset_brain(move || c.send(Request::ResetBrain));
    }
    {
        let c = client.clone();
        ui.on_migrate_state_format(move |target| {
            let target_state_version = (target as u32).max(1);
            c.send(Request::MigrateStateFormat {
                target_state_version,
            });
        });
    }

    // Auto snapshot (UI-driven repeating timer)
    {
        let c = client.clone();
        let auto_timer = Timer::default();

        ui.on_auto_snapshot_settings_changed(move |enabled, interval_s| {
            if !enabled {
                auto_timer.stop();
                return;
            }

            let interval_s = (interval_s as i64).max(10) as u64;
            let c_tick = c.clone();
            auto_timer.start(
                TimerMode::Repeated,
                Duration::from_secs(interval_s),
                move || c_tick.send(Request::SaveSnapshot),
            );
        });

        // Apply initial values (start/stop timer immediately)
        ui.invoke_auto_snapshot_settings_changed(
            ui.get_auto_snapshot_enabled(),
            ui.get_auto_snapshot_interval_s(),
        );
    }

    // Framerate control
    {
        let c = client.clone();
        let pending = pending_fps.clone();
        ui.on_set_framerate(move |fps| {
            *pending.borrow_mut() = Some((fps as u32, Instant::now()));
            c.send(Request::SetFramerate { fps: fps as u32 })
        });
    }

    // Trial period control
    {
        let c = client.clone();
        let pending = pending_trial_ms.clone();
        ui.on_set_trial_period_ms(move |ms| {
            *pending.borrow_mut() = Some((ms as u32, Instant::now()));
            c.send(Request::SetTrialPeriodMs { ms: ms as u32 })
        });
    }

    // Execution tier control (CPU/GPU toggle)
    {
        let c = client.clone();
        ui.on_set_execution_tier(move |tier| {
            save_exec_tier_pref(tier.as_str());
            c.send(Request::SetExecutionTier {
                tier: tier.to_string(),
            })
        });
    }

    // Initial graph fetch (small defaults)
    {
        *last_graph_req.borrow_mut() = Instant::now();
        client.send(Request::GetGraph {
            kind: "substrate".to_string(),
            max_nodes: 1000,
            max_edges: 50_000,
            include_isolated: false,
        });
    }

    // Human input
    {
        let c = client.clone();
        let ui_weak = ui.as_weak();
        ui.on_human_key_pressed(move |key| {
            // Human actions are intentionally disabled in "Braine" mode.
            // (We currently hide the Human toggle in the UI.)
            if let Some(ui) = ui_weak.upgrade() {
                if ui.get_is_braine_mode() {
                    return;
                }
            }

            match key.as_str() {
                "left" => c.send(Request::HumanAction {
                    action: "left".into(),
                }),
                "right" => c.send(Request::HumanAction {
                    action: "right".into(),
                }),
                "up" => c.send(Request::HumanAction {
                    action: "up".into(),
                }),
                "down" => c.send(Request::HumanAction {
                    action: "down".into(),
                }),
                "stay" => c.send(Request::HumanAction {
                    action: "stay".into(),
                }),
                _ => {}
            }
        });
    }
    {
        let c = client.clone();
        ui.on_human_key_released(move |_key| {
            // No-op; actions are instantaneous
            let _ = c.clone();
        });
    }

    // Poll daemon for state.
    // This is adaptive: when you slow FPS down, the UI poll slows too so the spot changes remain visible.
    let ui_weak = ui.as_weak();
    let c = client.clone();
    let pending_fps_poll = pending_fps.clone();
    let pending_trial_ms_poll = pending_trial_ms.clone();
    let last_graph_req_poll = last_graph_req.clone();
    let graph_hover_nodes_poll = graph_hover_nodes.clone();
    let graph_hover_edges_poll = graph_hover_edges.clone();
    let graph_pos_by_kind_poll = graph_pos_by_kind.clone();
    let graph_temp_by_kind_poll = graph_temp_by_kind.clone();

    // Rolling oscilloscope buffer (UI plot).
    let osc_samples: Rc<RefCell<Vec<f32>>> = Rc::new(RefCell::new(Vec::new()));
    let osc_samples_poll = osc_samples.clone();

    type PollFn = Rc<dyn Fn(Duration)>;
    let poll_fn: Rc<RefCell<Option<PollFn>>> = Rc::new(RefCell::new(None));
    let poll_fn_setter = poll_fn.clone();

    let poll_impl: PollFn = Rc::new(move |delay: Duration| {
        let ui_weak = ui_weak.clone();
        let c = c.clone();
        let poll_fn = poll_fn_setter.clone();
        let pending_fps_poll = pending_fps_poll.clone();
        let pending_trial_ms_poll = pending_trial_ms_poll.clone();
        let last_graph_req_poll = last_graph_req_poll.clone();
        let graph_hover_nodes_poll = graph_hover_nodes_poll.clone();
        let graph_hover_edges_poll = graph_hover_edges_poll.clone();
        let graph_pos_by_kind_poll = graph_pos_by_kind_poll.clone();
        let graph_temp_by_kind_poll = graph_temp_by_kind_poll.clone();
        let osc_samples_poll = osc_samples_poll.clone();

        Timer::single_shot(delay, move || {
            let now = Instant::now();

            let mut next_delay = Duration::from_millis(100);
            if let Some(ui) = ui_weak.upgrade() {
                // Pause freezes UI updates (daemon continues running).
                if ui.get_paused() {
                    if let Some(poll_fn) = poll_fn.borrow().as_ref() {
                        poll_fn(next_delay);
                    }
                    return;
                }

                c.send(Request::GetState);
                let snap = c.snapshot();
                let g = c.graph_snapshot();
                let params = c.game_params();

                // Update oscillation series (signed waveform uses osc_y).
                {
                    let v = snap.brain_stats.osc_y;
                    let mut buf = osc_samples_poll.borrow_mut();
                    buf.push(v);
                    const MAX: usize = 240;
                    if buf.len() > MAX {
                        let drop = buf.len() - MAX;
                        buf.drain(0..drop);
                    }
                    ui.set_osc_samples(ModelRc::new(VecModel::from(buf.clone())));
                }

                let (spotxy_chosen_ix, spotxy_chosen_iy, spotxy_correct_ix, spotxy_correct_iy) = {
                    let parse = |action: &str, n: u32| -> Option<(i32, i32)> {
                        if n == 0 {
                            return None;
                        }

                        let rest = action.strip_prefix("spotxy_cell_")?;
                        let mut parts = rest.split('_');
                        let nn = parts.next()?.parse::<u32>().ok()?;
                        let ix = parts.next()?.parse::<i32>().ok()?;
                        let iy = parts.next()?.parse::<i32>().ok()?;

                        if nn != n || ix < 0 || iy < 0 {
                            return None;
                        }
                        Some((ix, iy))
                    };

                    let grid_n = snap.game.spotxy_grid_n();
                    let (cx, cy) = parse(snap.game.chosen_action(), grid_n).unwrap_or((-1, -1));
                    let (kx, ky) = parse(&snap.meaning.correct_action, grid_n).unwrap_or((-1, -1));
                    (cx, cy, kx, ky)
                };

                let (pos_x, pos_y) = snap.game.pos_xy();
                let (pong_ball2_x, pong_ball2_y) = snap.game.pong_ball2_pos_xy();
                let (pong_paddle_y, pong_paddle_half_height, pong_paddle_speed, pong_ball_speed) =
                    snap.game.pong_params();

                ui.set_game(GameState {
                    kind: snap.game.kind().into(),
                    reversal_active: snap.game.reversal_active(),
                    chosen_action: snap.game.chosen_action().into(),
                    pos_x,
                    pos_y,
                    pong_ball_visible: snap.game.pong_ball_visible(),
                    pong_ball2_x,
                    pong_ball2_y,
                    pong_ball2_visible: snap.game.pong_ball2_visible(),
                    pong_ball2_enabled: snap.game.pong_ball2_enabled(),
                    pong_paddle_y,
                    pong_paddle_half_height,
                    pong_paddle_speed,
                    pong_ball_speed,
                    spotxy_eval: snap.game.spotxy_eval(),
                    spotxy_mode: snap.game.spotxy_mode().into(),
                    spotxy_grid_n: snap.game.spotxy_grid_n() as i32,
                    spotxy_chosen_ix,
                    spotxy_chosen_iy,
                    spotxy_correct_ix,
                    spotxy_correct_iy,
                    text_regime: snap.game.text_regime() as i32,
                    text_token: snap.game.text_token().into(),
                    text_target_next: snap.game.text_target_next().into(),
                    text_outcomes: snap.game.text_outcomes() as i32,
                    text_shift_every: snap.game.text_shift_every() as i32,
                    text_vocab_size: snap.game.text_vocab_size() as i32,
                    replay_dataset: snap.game.replay_dataset().into(),
                    replay_index: snap.game.replay_index() as i32,
                    replay_total: snap.game.replay_total() as i32,
                    replay_trial_id: snap.game.replay_trial_id().into(),
                    last_reward: snap.game.last_reward(),
                    spot_is_left: snap.game.spot_is_left(),
                    response_made: snap.game.response_made(),
                    trial_frame: snap.game.trial_frame() as i32,
                    trial_duration: snap.game.trial_duration() as i32,
                });

                // Keep the selected (non-running) game in sync with the daemon's active game.
                if !ui.get_running() {
                    ui.set_selected_game_kind(snap.game.kind().into());
                }
                ui.set_hud(HudData {
                    frame: snap.frame as i32,
                    trials: snap.hud.trials as i32,
                    correct: snap.hud.correct as i32,
                    incorrect: snap.hud.incorrect as i32,
                    accuracy: snap.hud.accuracy,
                    recent_rate: snap.hud.recent_rate,
                    last_100_rate: snap.hud.last_100_rate,
                    neuromod: snap.hud.neuromod,
                    learning_at_trial: snap.hud.learning_at_trial,
                    learned_at_trial: snap.hud.learned_at_trial,
                    mastered_at_trial: snap.hud.mastered_at_trial,
                });

                // System error banner: prefer connection/protocol errors; otherwise show daemon-reported error.
                let mut se = c.system_error();
                if se.is_empty() {
                    se = snap.last_error.clone();
                }
                ui.set_system_error(se.into());
                ui.set_brain_stats(BrainStats {
                    unit_count: snap.brain_stats.unit_count as i32,
                    max_units_limit: snap.brain_stats.max_units_limit as i32,
                    execution_tier: snap.brain_stats.execution_tier.clone().into(),
                    execution_tier_selected: snap
                        .brain_stats
                        .execution_tier_selected
                        .clone()
                        .into(),
                    execution_tier_effective: snap
                        .brain_stats
                        .execution_tier_effective
                        .clone()
                        .into(),
                    connection_count: snap.brain_stats.connection_count as i32,
                    pruned_last_step: snap.brain_stats.pruned_last_step as i32,
                    births_last_step: snap.brain_stats.births_last_step as i32,
                    saturated: snap.brain_stats.saturated,
                    avg_amp: snap.brain_stats.avg_amp,
                    avg_weight: snap.brain_stats.avg_weight,
                    memory_bytes: snap.brain_stats.memory_bytes as i32,
                    causal_base_symbols: snap.brain_stats.causal_base_symbols as i32,
                    causal_edges: snap.brain_stats.causal_edges as i32,
                    causal_last_directed_updates: snap.brain_stats.causal_last_directed_edge_updates
                        as i32,
                    causal_last_cooccur_updates: snap.brain_stats.causal_last_cooccur_edge_updates
                        as i32,
                    age_steps: snap.brain_stats.age_steps as i32,
                });

                ui.set_storage_data_dir(snap.storage.data_dir.clone().into());
                ui.set_storage_brain_file(snap.storage.brain_file.clone().into());
                ui.set_storage_brain_bytes(snap.storage.brain_bytes as i32);
                ui.set_storage_runtime_file(snap.storage.runtime_file.clone().into());
                ui.set_storage_runtime_bytes(snap.storage.runtime_bytes as i32);
                ui.set_storage_loaded_snapshot(snap.storage.loaded_snapshot.clone().into());
                ui.set_storage_state_wrapper_version(snap.storage.state_wrapper_version as i32);

                // Advisor (daemon-only)
                ui.set_advisor_enabled(snap.advisor.enabled);
                ui.set_advisor_mode(snap.advisor.mode.clone().into());
                ui.set_advisor_every_trials(snap.advisor.every_trials as i32);
                ui.set_advisor_last_rationale(snap.advisor.last_rationale.clone().into());

                ui.set_advisor_context_json(c.advisor_context_json().into());
                ui.set_advisor_report_json(c.advisor_report_json().into());
                ui.set_advisor_status(c.advisor_status().into());

                let snaps: Vec<SnapshotItem> = snap
                    .storage
                    .snapshots
                    .iter()
                    .map(|s| SnapshotItem {
                        stem: s.stem.clone().into(),
                        modified_unix: s.modified_unix as i32,
                        brain_bytes: s.brain_bytes as i32,
                        runtime_bytes: s.runtime_bytes as i32,
                    })
                    .collect();
                ui.set_storage_snapshots(ModelRc::new(VecModel::from(snaps)));

                let points: Vec<UnitPoint> = snap
                    .unit_plot
                    .iter()
                    .map(|p| UnitPoint {
                        id: p.id as i32,
                        amp: p.amp,
                        amp01: p.amp01,
                        rel_age: p.rel_age,
                        is_reserved: p.is_reserved,
                        is_sensor_member: p.is_sensor_member,
                        is_group_member: p.is_group_member,
                    })
                    .collect();
                ui.set_unit_points(ModelRc::new(VecModel::from(points)));

                let scores: Vec<ActionScore> = snap
                    .action_scores
                    .iter()
                    .map(|s| ActionScore {
                        name: s.name.clone().into(),
                        habit_norm: s.habit_norm,
                        meaning_global: s.meaning_global,
                        meaning_conditional: s.meaning_conditional,
                        meaning: s.meaning,
                        score: s.score,
                    })
                    .collect();
                ui.set_action_scores(ModelRc::new(VecModel::from(scores)));

                ui.set_meaning(MeaningData {
                    stimulus: snap.meaning.stimulus.clone().into(),
                    correct_action: snap.meaning.correct_action.clone().into(),
                    action_a_name: snap.meaning.action_a_name.clone().into(),
                    action_b_name: snap.meaning.action_b_name.clone().into(),
                    pair_left: MeaningEdges {
                        to_reward_pos: snap.meaning.pair_left.to_reward_pos,
                        to_reward_neg: snap.meaning.pair_left.to_reward_neg,
                        meaning: snap.meaning.pair_left.meaning,
                    },
                    pair_right: MeaningEdges {
                        to_reward_pos: snap.meaning.pair_right.to_reward_pos,
                        to_reward_neg: snap.meaning.pair_right.to_reward_neg,
                        meaning: snap.meaning.pair_right.meaning,
                    },
                    action_left: MeaningEdges {
                        to_reward_pos: snap.meaning.action_left.to_reward_pos,
                        to_reward_neg: snap.meaning.action_left.to_reward_neg,
                        meaning: snap.meaning.action_left.meaning,
                    },
                    action_right: MeaningEdges {
                        to_reward_pos: snap.meaning.action_right.to_reward_pos,
                        to_reward_neg: snap.meaning.action_right.to_reward_neg,
                        meaning: snap.meaning.action_right.meaning,
                    },
                    pair_gap: snap.meaning.pair_gap,
                    global_gap: snap.meaning.global_gap,
                });

                fn hist_to_dots(hist: &[f32]) -> Vec<MeaningHistDot> {
                    let n = hist.len();
                    if n == 0 {
                        return Vec::new();
                    }

                    let mut max_abs = 0.0f32;
                    for &v in hist {
                        max_abs = max_abs.max(v.abs());
                    }
                    let inv = if max_abs > 1e-6 { 1.0 / max_abs } else { 0.0 };
                    let denom = (n - 1).max(1) as f32;

                    let mut out = Vec::with_capacity(n);
                    for (i, &raw) in hist.iter().enumerate() {
                        let x01 = (i as f32 / denom).clamp(0.0, 1.0);
                        out.push(MeaningHistDot {
                            x01,
                            v: (raw * inv).clamp(-1.0, 1.0),
                            positive: raw >= 0.0,
                        });
                    }
                    out
                }

                ui.set_meaning_pair_gap_dots(ModelRc::new(VecModel::from(hist_to_dots(
                    &snap.meaning.pair_gap_history,
                ))));
                ui.set_meaning_global_gap_dots(ModelRc::new(VecModel::from(hist_to_dots(
                    &snap.meaning.global_gap_history,
                ))));

                // Read-only experts status line (helps interpret plots when experts are active).
                let experts_status = if snap.experts_enabled {
                    let active = snap.experts.active_count;
                    let total = snap.experts.total_active_count.max(active);
                    let max = snap.experts.max_children;
                    let persist = if snap.experts.persistence_mode.trim().is_empty() {
                        "full"
                    } else {
                        snap.experts.persistence_mode.as_str()
                    };
                    let nested = if snap.experts.allow_nested {
                        format!("nested depth={}", snap.experts.max_depth.max(1))
                    } else {
                        "nested off".to_string()
                    };
                    let active_desc = snap
                        .active_expert
                        .as_ref()
                        .map(|a| {
                            let ctx = if a.context_key.trim().is_empty() {
                                "?".to_string()
                            } else {
                                a.context_key.clone()
                            };
                            format!("active #{} ctx={} r={:.2}", a.id, ctx, a.reward_ema)
                        })
                        .unwrap_or_else(|| "active -".to_string());
                    let deployed = if total > active {
                        format!("{active}/{max} (total {total})")
                    } else {
                        format!("{active}/{max}")
                    };
                    format!(
                        "Experts: {}  {}  persist={}  {}",
                        deployed, active_desc, persist, nested
                    )
                } else {
                    String::new()
                };
                ui.set_experts_status(experts_status.into());
                ui.set_experts_enabled(snap.experts_enabled);
                ui.set_experts_allow_nested(snap.experts.allow_nested);
                ui.set_experts_max_depth(snap.experts.max_depth as i32);
                ui.set_experts_max_children(snap.experts.max_children as i32);
                if !snap.experts.persistence_mode.trim().is_empty() {
                    ui.set_experts_persistence_mode(snap.experts.persistence_mode.clone().into());
                }
                ui.set_experts_reward_shift_ema_delta_threshold(
                    snap.experts.reward_shift_ema_delta_threshold,
                );
                ui.set_experts_performance_collapse_drop_threshold(
                    snap.experts.performance_collapse_drop_threshold,
                );
                ui.set_experts_performance_collapse_baseline_min(
                    snap.experts.performance_collapse_baseline_min,
                );

                // Apply game param schema (Pong + SpotXY).
                if snap.game.kind() == "pong" {
                    for p in &params {
                        match p.key.as_str() {
                            "paddle_speed" => {
                                ui.set_pong_paddle_speed_min(p.min);
                                ui.set_pong_paddle_speed_max(p.max);
                                ui.set_pong_paddle_speed_default(p.default);
                            }
                            "ball_speed" => {
                                ui.set_pong_ball_speed_min(p.min);
                                ui.set_pong_ball_speed_max(p.max);
                                ui.set_pong_ball_speed_default(p.default);
                            }
                            "paddle_half_height" => {
                                ui.set_pong_paddle_half_height_min(p.min);
                                ui.set_pong_paddle_half_height_max(p.max);
                                ui.set_pong_paddle_half_height_default(p.default);
                            }
                            _ => {}
                        }
                    }
                } else if snap.game.kind() == "spotxy" {
                    for p in &params {
                        match p.key.as_str() {
                            "grid_n" => {
                                ui.set_spotxy_grid_n_min(p.min as i32);
                                ui.set_spotxy_grid_n_max(p.max as i32);
                                ui.set_spotxy_grid_n_default(p.default as i32);
                            }
                            "eval" => {
                                // Boolean represented as 0 or 1.
                            }
                            _ => {}
                        }
                    }
                }

                // Graph auto-refresh (independent of the main poll cadence)
                if ui.get_graph_auto_refresh() {
                    let interval_ms = ui.get_graph_interval_ms().max(250) as u64;
                    let since = now.duration_since(*last_graph_req_poll.borrow());
                    if since >= Duration::from_millis(interval_ms) {
                        *last_graph_req_poll.borrow_mut() = now;
                        c.send(Request::GetGraph {
                            kind: ui.get_graph_kind().to_string(),
                            max_nodes: ui.get_graph_max_nodes().max(1) as u32,
                            max_edges: ui.get_graph_max_edges().max(0) as u32,
                            include_isolated: ui.get_graph_include_isolated(),
                        });
                    }
                }

                // Render the last received graph snapshot.
                {
                    use std::collections::HashMap;

                    let n = g.nodes.len();
                    let mut dots: Vec<GraphNodeDot> = Vec::with_capacity(n);

                    let kind_now = g.kind.clone();
                    let layout = ui.get_graph_layout().to_string();

                    // Prepare id list and initial positions.
                    let mut ids: Vec<u32> = Vec::with_capacity(n);
                    let mut pos: Vec<GraphPos> = Vec::with_capacity(n);

                    if layout == "circle" {
                        for (i, node) in g.nodes.iter().enumerate() {
                            ids.push(node.id);
                            pos.push(init_pos_on_circle(i, n));
                        }
                    } else {
                        // force layout: start from cached positions
                        let mut by_kind = graph_pos_by_kind_poll.borrow_mut();
                        let store = by_kind.entry(kind_now.clone()).or_default();
                        for (i, node) in g.nodes.iter().enumerate() {
                            ids.push(node.id);
                            let p = store
                                .get(&node.id)
                                .copied()
                                .unwrap_or_else(|| init_pos_on_circle(i, n));
                            pos.push(p);
                        }
                    }

                    // Build edge list in index space for the layout.
                    let mut index_by_id: HashMap<u32, usize> = HashMap::with_capacity(n);
                    for (i, &id) in ids.iter().enumerate() {
                        index_by_id.insert(id, i);
                    }
                    let mut layout_edges: Vec<(usize, usize, f32)> =
                        Vec::with_capacity(g.edges.len());
                    for e in &g.edges {
                        let Some(&a) = index_by_id.get(&e.from) else {
                            continue;
                        };
                        let Some(&b) = index_by_id.get(&e.to) else {
                            continue;
                        };
                        if a != b {
                            layout_edges.push((a, b, e.weight));
                        }
                    }

                    // Incremental force-directed relaxation.
                    // Keep iteration budget small per UI poll; persistence yields stability.
                    if layout != "circle" {
                        // Use a decaying temperature so the layout converges and then becomes stable.
                        // Reset temperature shortly after a refresh.
                        let since_req = now.duration_since(*last_graph_req_poll.borrow());
                        let reset = since_req < Duration::from_millis(250);

                        let mut temps = graph_temp_by_kind_poll.borrow_mut();
                        let entry = temps.entry(kind_now.clone()).or_insert(0.06f32);
                        if reset {
                            *entry = 0.06f32;
                        } else {
                            *entry = (*entry * 0.97).max(0.0035);
                        }

                        let mut t = *entry;
                        for _ in 0..30 {
                            force_layout_step(&ids, &layout_edges, &mut pos, t);
                            t *= 0.92;
                        }
                    }

                    // Store updated positions (force only) and emit dots for UI.
                    let zoom = ui.get_graph_zoom();
                    let pan_x01 = ui.get_graph_pan_x01();
                    let pan_y01 = ui.get_graph_pan_y01();

                    if layout != "circle" {
                        let mut by_kind = graph_pos_by_kind_poll.borrow_mut();
                        let store = by_kind.entry(kind_now.clone()).or_default();
                        for (i, node) in g.nodes.iter().enumerate() {
                            let p = clamp_pos(pos[i]);
                            store.insert(node.id, p);

                            let pz = apply_view(p, zoom, pan_x01, pan_y01);
                            let domain = classify_domain(&g.kind, &node.label);
                            dots.push(GraphNodeDot {
                                x01: pz.x01,
                                y01: pz.y01,
                                label: node.label.clone().into(),
                                value: node.value,
                                domain: domain.clone().into(),
                            });
                        }
                    } else {
                        for (i, node) in g.nodes.iter().enumerate() {
                            let p = clamp_pos(pos[i]);
                            let pz = apply_view(p, zoom, pan_x01, pan_y01);
                            let domain = classify_domain(&g.kind, &node.label);
                            dots.push(GraphNodeDot {
                                x01: pz.x01,
                                y01: pz.y01,
                                label: node.label.clone().into(),
                                value: node.value,
                                domain: domain.clone().into(),
                            });
                        }
                    }

                    let mut pos_by_id: HashMap<u32, (f32, f32)> = HashMap::with_capacity(n);
                    for (i, id) in ids.iter().enumerate() {
                        let p = clamp_pos(pos[i]);
                        let pz = apply_view(p, zoom, pan_x01, pan_y01);
                        pos_by_id.insert(*id, (pz.x01, pz.y01));
                    }

                    // Update hover hit-test state (kept in sync with rendered layout).
                    {
                        let mut hn = graph_hover_nodes_poll.borrow_mut();
                        hn.clear();
                        hn.reserve(dots.len());
                        for d in &dots {
                            hn.push(GraphHoverNode {
                                x01: d.x01,
                                y01: d.y01,
                                label: d.label.to_string(),
                                value: d.value,
                                domain: d.domain.to_string(),
                            });
                        }
                    }

                    // Build id -> label map for edge tooltips.
                    let mut label_by_id: HashMap<u32, String> =
                        HashMap::with_capacity(g.nodes.len());
                    for node in &g.nodes {
                        label_by_id.insert(node.id, node.label.clone());
                    }

                    let mut max_abs = 0.0f32;
                    for e in &g.edges {
                        max_abs = max_abs.max(e.weight.abs());
                    }
                    let inv = if max_abs > 1e-6 { 1.0 / max_abs } else { 0.0 };

                    let mut segs: Vec<GraphEdgeSeg> = Vec::with_capacity(g.edges.len());
                    let mut hover_edges: Vec<GraphHoverEdge> = Vec::with_capacity(g.edges.len());
                    let mut degree_by_id: HashMap<u32, usize> = HashMap::new();
                    for e in &g.edges {
                        let Some(&(x1, y1)) = pos_by_id.get(&e.from) else {
                            continue;
                        };
                        let Some(&(x2, y2)) = pos_by_id.get(&e.to) else {
                            continue;
                        };

                        *degree_by_id.entry(e.from).or_insert(0) += 1;
                        *degree_by_id.entry(e.to).or_insert(0) += 1;

                        let strength01 = (e.weight.abs() * inv).clamp(0.0, 1.0);
                        let positive = e.weight >= 0.0;

                        // Nonlinear mapping: preserve relative differences without making
                        // most edges effectively invisible.
                        let thickness01 = strength01.powf(1.2).clamp(0.0, 1.0);

                        segs.push(GraphEdgeSeg {
                            x1,
                            y1,
                            x2,
                            y2,
                            positive,
                            strength01,
                            thickness01,
                        });

                        let from_label = label_by_id
                            .get(&e.from)
                            .cloned()
                            .unwrap_or_else(|| format!("{}", e.from));
                        let to_label = label_by_id
                            .get(&e.to)
                            .cloned()
                            .unwrap_or_else(|| format!("{}", e.to));
                        hover_edges.push(GraphHoverEdge {
                            x1,
                            y1,
                            x2,
                            y2,
                            from_label,
                            to_label,
                            weight: e.weight,
                        });
                    }

                    {
                        let mut he = graph_hover_edges_poll.borrow_mut();
                        he.clear();
                        he.extend(hover_edges);
                    }

                    ui.set_graph_nodes(ModelRc::new(VecModel::from(dots)));
                    ui.set_graph_edge_segs(ModelRc::new(VecModel::from(segs)));

                    let node_count = g.nodes.len();
                    let edge_count = degree_by_id
                        .values()
                        .sum::<usize>()
                        .saturating_div(2)
                        .min(g.edges.len());
                    let max_conn = degree_by_id.values().copied().max().unwrap_or(0);
                    let avg_conn = if node_count > 0 {
                        (2.0 * (edge_count as f32)) / (node_count as f32)
                    } else {
                        0.0
                    };
                    ui.set_graph_stats(
                        format!(
                            "Displayed: {} nodes • {} edges • avg {:.2} conn/node • max {}",
                            node_count, edge_count, avg_conn, max_conn
                        )
                        .into(),
                    );
                }
                ui.set_is_braine_mode(snap.mode != "human");
                ui.set_running(snap.running);

                // Keep the UI's displayed FPS in sync with the daemon.
                if snap.target_fps != 0 {
                    let pending = *pending_fps_poll.borrow();
                    let apply = match pending {
                        Some((want, t))
                            if now.duration_since(t) < Duration::from_millis(800)
                                && snap.target_fps != want =>
                        {
                            false
                        }
                        Some((want, _)) if snap.target_fps == want => {
                            pending_fps_poll.borrow_mut().take();
                            true
                        }
                        Some((_, t)) if now.duration_since(t) >= Duration::from_millis(800) => {
                            pending_fps_poll.borrow_mut().take();
                            true
                        }
                        None => true,
                        _ => true,
                    };

                    if apply {
                        ui.set_framerate(snap.target_fps as i32);
                    }
                }

                // Keep the UI's displayed trial period in sync with the daemon.
                let trial_duration = snap.game.trial_duration();
                if trial_duration != 0 {
                    let pending = *pending_trial_ms_poll.borrow();
                    let apply = match pending {
                        Some((want, t))
                            if now.duration_since(t) < Duration::from_millis(800)
                                && trial_duration != want =>
                        {
                            false
                        }
                        Some((want, _)) if trial_duration == want => {
                            pending_trial_ms_poll.borrow_mut().take();
                            true
                        }
                        Some((_, t)) if now.duration_since(t) >= Duration::from_millis(800) => {
                            pending_trial_ms_poll.borrow_mut().take();
                            true
                        }
                        None => true,
                        _ => true,
                    };

                    if apply {
                        ui.set_trial_period_ms(trial_duration as i32);
                    }
                }

                // Choose the next poll rate.
                // - When running: poll ~once per sim step (bounded), so slowing FPS slows visible updates.
                // - When stopped: poll slower.
                if snap.running {
                    let fps = snap.target_fps.max(1);
                    let ms = (1000 / fps).max(16) as u64;
                    next_delay = Duration::from_millis(ms.min(1000));
                } else {
                    next_delay = Duration::from_millis(250);
                }
            }

            if let Some(cb) = poll_fn.borrow().as_ref() {
                cb(next_delay);
            }
        });
    });

    *poll_fn.borrow_mut() = Some(poll_impl.clone());
    poll_impl(Duration::from_millis(100));

    ui.run()
}
