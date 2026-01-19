//! Braine Daemon - Background brain learning service
//!
//! This daemon runs continuously in the background, managing:
//! - Brain state and learning
//! - Game/task execution  
//! - Persistent storage
//! - IPC server for UI clients
//!
//! Storage locations:
//! - Linux: ~/.local/share/braine/
//! - Windows: %APPDATA%\Braine\
//! - MacOS: ~/Library/Application Support/Braine/

use braine::storage;
use braine::substrate::Stimulus;
use braine::substrate::{
    ActionScoreBreakdown, Brain, BrainConfig, BrainDelta, OwnedStimulus, RewardEdges, UnitPlotPoint,
};
use serde::{Deserialize, Serialize};
use std::fs::File;
use std::io::{Read as _, Seek as _, SeekFrom, Write as _};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::net::{TcpListener, TcpStream};
use tokio::sync::RwLock;
use tokio::time;
use tracing::{error, info, warn};

mod advisor;
mod experts;
mod game;
mod paths;
mod state_image;

use experts::{ExpertManager, ExpertsPersistenceMode, ParentLearningPolicy};
use game::{
    BanditGame, PongGame, ReplayDataset, ReplayGame, SpotGame, SpotReversalGame, SpotXYGame,
    TextNextTokenGame,
};
use paths::AppPaths;

fn default_experts_max_depth() -> u32 {
    1
}

fn default_reward_shift_ema_delta_threshold() -> f32 {
    0.55
}

fn default_performance_collapse_drop_threshold() -> f32 {
    0.65
}

fn default_performance_collapse_baseline_min() -> f32 {
    0.25
}

fn default_experts_persistence_mode() -> String {
    "full".to_string()
}

#[allow(clippy::large_enum_variant)]
#[derive(Debug)]
enum ActiveGame {
    Spot(SpotGame),
    Bandit(BanditGame),
    SpotReversal(SpotReversalGame),
    SpotXY(SpotXYGame),
    Pong(PongGame),
    Text(TextNextTokenGame),
    Replay(ReplayGame),
}

impl ActiveGame {
    fn kind(&self) -> &'static str {
        match self {
            ActiveGame::Spot(_) => "spot",
            ActiveGame::Bandit(_) => "bandit",
            ActiveGame::SpotReversal(_) => "spot_reversal",
            ActiveGame::SpotXY(_) => "spotxy",
            ActiveGame::Pong(_) => "pong",
            ActiveGame::Text(_) => "text",
            ActiveGame::Replay(_) => "replay",
        }
    }

    fn update_timing(&mut self, trial_period_ms: u32) {
        match self {
            ActiveGame::Spot(g) => g.update_timing(trial_period_ms),
            ActiveGame::Bandit(g) => g.update_timing(trial_period_ms),
            ActiveGame::SpotReversal(g) => g.update_timing(trial_period_ms),
            ActiveGame::SpotXY(g) => g.update_timing(trial_period_ms),
            ActiveGame::Pong(g) => g.update_timing(trial_period_ms),
            ActiveGame::Text(g) => g.update_timing(trial_period_ms),
            ActiveGame::Replay(g) => g.update_timing(trial_period_ms),
        }
    }

    fn stimulus_name(&self) -> &'static str {
        match self {
            ActiveGame::Spot(g) => g.stimulus_name(),
            ActiveGame::Bandit(g) => g.stimulus_name(),
            ActiveGame::SpotReversal(g) => g.stimulus_name(),
            ActiveGame::SpotXY(g) => g.stimulus_name(),
            ActiveGame::Pong(g) => g.stimulus_name(),
            ActiveGame::Text(g) => g.stimulus_name(),
            ActiveGame::Replay(g) => g.stimulus_name(),
        }
    }

    fn correct_action(&self) -> std::borrow::Cow<'_, str> {
        match self {
            ActiveGame::Spot(g) => std::borrow::Cow::Borrowed(g.correct_action()),
            ActiveGame::Bandit(g) => std::borrow::Cow::Borrowed(g.best_action()),
            ActiveGame::SpotReversal(g) => std::borrow::Cow::Borrowed(g.correct_action()),
            ActiveGame::SpotXY(g) => std::borrow::Cow::Borrowed(g.correct_action()),
            ActiveGame::Pong(g) => std::borrow::Cow::Borrowed(g.correct_action()),
            ActiveGame::Text(g) => std::borrow::Cow::Owned(g.correct_action()),
            ActiveGame::Replay(g) => std::borrow::Cow::Borrowed(g.correct_action()),
        }
    }

    fn allowed_actions(&self) -> &[String] {
        match self {
            ActiveGame::Spot(_) | ActiveGame::Bandit(_) | ActiveGame::SpotReversal(_) => {
                static ACTIONS: std::sync::OnceLock<Vec<String>> = std::sync::OnceLock::new();
                ACTIONS.get_or_init(|| vec!["left".to_string(), "right".to_string()])
            }
            ActiveGame::SpotXY(g) => g.allowed_actions(),
            ActiveGame::Pong(g) => g.allowed_actions(),
            ActiveGame::Text(g) => g.allowed_actions(),
            ActiveGame::Replay(g) => g.allowed_actions(),
        }
    }

    fn response_made(&self) -> bool {
        match self {
            ActiveGame::Spot(g) => g.response_made,
            ActiveGame::Bandit(g) => g.response_made,
            ActiveGame::SpotReversal(g) => g.response_made,
            ActiveGame::SpotXY(g) => g.response_made,
            ActiveGame::Pong(g) => g.response_made,
            ActiveGame::Text(g) => g.response_made,
            ActiveGame::Replay(g) => g.response_made,
        }
    }

    fn trial_frame(&self) -> u32 {
        match self {
            ActiveGame::Spot(g) => g.trial_frame,
            ActiveGame::Bandit(g) => g.trial_frame,
            ActiveGame::SpotReversal(g) => g.trial_frame,
            ActiveGame::SpotXY(g) => g.trial_frame,
            ActiveGame::Pong(g) => g.trial_frame,
            ActiveGame::Text(g) => g.trial_frame,
            ActiveGame::Replay(g) => g.trial_frame,
        }
    }

    fn spot_is_left(&self) -> bool {
        match self {
            ActiveGame::Spot(g) => g.spot_is_left,
            // For Bandit, reuse this field as "best arm is left".
            ActiveGame::Bandit(g) => g.best_action() == "left",
            ActiveGame::SpotReversal(g) => g.spot_is_left,
            // For SpotXY, reuse this field as "correct side is left" (x < 0).
            ActiveGame::SpotXY(g) => g.pos_x < 0.0,
            // For Pong, reuse this field as "ball is above paddle".
            ActiveGame::Pong(g) => g.sim.state.ball_y > g.sim.state.paddle_y,
            // For Text, this field is not meaningful.
            ActiveGame::Text(_) => false,

            // For Replay, this field is not meaningful.
            ActiveGame::Replay(_) => false,
        }
    }

    fn reversal_active(&self) -> bool {
        match self {
            ActiveGame::SpotReversal(g) => g.reversal_active,
            _ => false,
        }
    }

    fn score_action(&mut self, action: &str, trial_period_ms: u32) -> Option<(f32, bool)> {
        match self {
            ActiveGame::Spot(g) => g.score_action(action),
            ActiveGame::Bandit(g) => g.score_action(action),
            ActiveGame::SpotReversal(g) => g.score_action(action),
            ActiveGame::SpotXY(g) => g.score_action(action),
            ActiveGame::Pong(g) => g.score_action(action, trial_period_ms),
            ActiveGame::Text(g) => {
                let _ = trial_period_ms;
                g.score_action(action)
            }
            ActiveGame::Replay(g) => {
                let _ = trial_period_ms;
                g.score_action(action)
            }
        }
    }

    fn stats(&self) -> &game::GameStats {
        match self {
            ActiveGame::Spot(g) => &g.stats,
            ActiveGame::Bandit(g) => &g.stats,
            ActiveGame::SpotReversal(g) => &g.stats,
            ActiveGame::SpotXY(g) => &g.stats,
            ActiveGame::Pong(g) => &g.stats,
            ActiveGame::Text(g) => &g.stats,
            ActiveGame::Replay(g) => &g.stats,
        }
    }

    fn stats_mut(&mut self) -> &mut game::GameStats {
        match self {
            ActiveGame::Spot(g) => &mut g.stats,
            ActiveGame::Bandit(g) => &mut g.stats,
            ActiveGame::SpotReversal(g) => &mut g.stats,
            ActiveGame::SpotXY(g) => &mut g.stats,
            ActiveGame::Pong(g) => &mut g.stats,
            ActiveGame::Text(g) => &mut g.stats,
            ActiveGame::Replay(g) => &mut g.stats,
        }
    }

    fn last_action(&self) -> Option<&str> {
        match self {
            ActiveGame::Spot(g) => g.last_action.as_deref(),
            ActiveGame::Bandit(g) => g.last_action.as_deref(),
            ActiveGame::SpotReversal(g) => g.last_action.as_deref(),
            ActiveGame::SpotXY(g) => g.last_action.as_deref(),
            ActiveGame::Pong(g) => g.last_action.as_deref(),
            ActiveGame::Text(g) => g.last_action.as_deref(),
            ActiveGame::Replay(g) => g.last_action.as_deref(),
        }
    }

    fn stimulus_key(&self) -> Option<&str> {
        match self {
            ActiveGame::SpotXY(g) => Some(g.stimulus_key()),
            ActiveGame::Pong(g) => Some(g.stimulus_key()),
            ActiveGame::Text(g) => Some(g.stimulus_key()),
            ActiveGame::Replay(g) => Some(g.stimulus_key()),
            _ => None,
        }
    }

    fn pos_xy(&self) -> Option<(f32, f32)> {
        match self {
            ActiveGame::SpotXY(g) => Some((g.pos_x, g.pos_y)),
            // Map pong ball_x in [0,1] into [-1,1] for the UI dot.
            ActiveGame::Pong(g) => Some((g.sim.state.ball_x * 2.0 - 1.0, g.sim.state.ball_y)),
            _ => None,
        }
    }

    fn spotxy_eval_mode(&self) -> bool {
        match self {
            ActiveGame::SpotXY(g) => g.eval_mode,
            _ => false,
        }
    }

    fn spotxy_grid_n(&self) -> u32 {
        match self {
            ActiveGame::SpotXY(g) => g.grid_n(),
            _ => 0,
        }
    }

    fn spotxy_mode_name(&self) -> &'static str {
        match self {
            ActiveGame::SpotXY(g) => g.mode_name(),
            _ => "",
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Protocol Messages
// ═══════════════════════════════════════════════════════════════════════════

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
enum Request {
    /// Introspect the daemon API surface (grouped by category).
    ApiCatalog,

    // Diagnostics (read-only)
    DiagGet,

    // Configuration (read/write)
    CfgGet,
    CfgSet {
        #[serde(default)]
        exploration_eps: Option<f32>,
        #[serde(default)]
        meaning_alpha: Option<f32>,
        #[serde(default)]
        target_fps: Option<u32>,
        #[serde(default)]
        trial_period_ms: Option<u32>,
        #[serde(default)]
        max_units: Option<u32>,
    },

    // Synchronization (edge/child -> master)
    SyncGetInfo,
    SyncApplyDelta {
        delta: BrainDelta,
        #[serde(default = "default_sync_delta_max")]
        delta_max: f32,
        expected_weights_len: u32,
        expected_fingerprint: u64,
        #[serde(default = "default_true")]
        autosave: bool,
    },

    // Inference (read-only)
    /// Run a pure inference step on a cloned brain and return action score breakdowns.
    ///
    /// This applies `stimuli` (without imprinting), advances dynamics for `steps`
    /// without learning/forgetting, then computes `action_scores` for `context_key`.
    InferActionScores {
        #[serde(default)]
        context_key: Option<String>,
        #[serde(default)]
        stimuli: Vec<OwnedStimulus>,
        #[serde(default = "default_infer_steps")]
        steps: u32,
        #[serde(default)]
        meaning_alpha: Option<f32>,
    },

    GetState,
    /// Fetch the tunable parameter schema for a game (UI uses this to render knobs).
    GetGameParams {
        game: String,
    },
    SetView {
        view: String,
    },
    SetMaxUnits {
        max_units: u32,
    },

    // Storage / snapshots
    SaveSnapshot,
    LoadSnapshot {
        stem: String,
    },
    GetGraph {
        kind: String,
        max_nodes: u32,
        max_edges: u32,
        #[serde(default)]
        include_isolated: bool,
    },
    Start,
    Stop,
    SetGame {
        game: String,
    },
    SetGameParam {
        game: String,
        key: String,
        value: f32,
    },
    SetSpotXYEval {
        eval: bool,
    },
    SpotXYIncreaseGrid,
    SpotXYDecreaseGrid,
    SetMode {
        mode: String,
    },
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

    /// Rewrite canonical brain file + snapshots to a target daemon-state wrapper format.
    ///
    /// Safety: requires the daemon to be stopped.
    MigrateStateFormat {
        target_state_version: u32,
    },
    Shutdown,
    SetFramerate {
        fps: u32,
    },
    SetTrialPeriodMs {
        ms: u32,
    },

    /// Set the brain execution tier ("scalar", "simd", "parallel", "gpu").
    ///
    /// Note: The effective tier may fall back (e.g. GPU unavailable).
    SetExecutionTier {
        tier: String,
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

        #[serde(default = "default_reward_shift_ema_delta_threshold")]
        reward_shift_ema_delta_threshold: f32,
        #[serde(default = "default_performance_collapse_drop_threshold")]
        performance_collapse_drop_threshold: f32,
        #[serde(default = "default_performance_collapse_baseline_min")]
        performance_collapse_baseline_min: f32,

        #[serde(default)]
        allow_nested: bool,
        #[serde(default = "default_experts_max_depth")]
        max_depth: u32,
        #[serde(default = "default_experts_persistence_mode")]
        persistence_mode: String,
    },
    CullExperts,

    // Advisor / LLM integration (slow loop; bounded config nudges)
    AdvisorGet,
    AdvisorSet {
        enabled: bool,
        #[serde(default)]
        every_trials: Option<u32>,
        #[serde(default)]
        mode: Option<String>,
    },
    AdvisorOnce {
        #[serde(default)]
        apply: bool,
    },

    /// What Braine produces for LLM consumption (bounded, structured context).
    AdvisorContext {
        #[serde(default)]
        include_action_scores: bool,
    },

    /// What an LLM returns (bounded advice). The daemon clamps and applies.
    AdvisorApply {
        advice: advisor::AdvisorAdvice,
    },

    // Replay dataset (dataset-driven evaluation)
    ReplayGetDataset,
    ReplaySetDataset {
        dataset: ReplayDataset,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
#[allow(clippy::large_enum_variant)]
enum Response {
    ApiCatalog {
        categories: Vec<ApiCategory>,
    },
    Diagnostics {
        running: bool,
        frame: u64,
        brain_stats: BrainStats,
        #[serde(default)]
        storage: StorageInfo,
    },
    Config {
        exploration_eps: f32,
        meaning_alpha: f32,
        target_fps: u32,
        trial_period_ms: u32,
        max_units_limit: u32,
    },
    SyncInfo {
        age_steps: u64,
        unit_count: u32,
        weights_len: u32,
        fingerprint: u64,
    },
    SyncApplied {
        applied_edges: u32,
        #[serde(default)]
        saved: bool,
        #[serde(default)]
        save_error: Option<String>,
    },
    InferActionScores {
        context_key: String,
        #[serde(default)]
        action_scores: Vec<ActionScoreBreakdown>,
    },
    State(Box<StateSnapshot>),
    GameParams {
        game: String,
        params: Vec<GameParamDef>,
    },
    Graph(Box<GraphSnapshot>),

    AdvisorStatus {
        config: advisor::AdvisorConfig,
        #[serde(default)]
        last_report: Option<advisor::AdvisorReport>,
    },
    AdvisorReport {
        report: advisor::AdvisorReport,
        #[serde(default)]
        applied: bool,
    },

    AdvisorContext {
        context: advisor::AdvisorContext,
        #[serde(default)]
        action_scores: Vec<ActionScoreBreakdown>,
    },

    ReplayDataset {
        dataset: ReplayDataset,
    },
    Success {
        message: String,
    },
    Error {
        message: String,
    },
}

fn default_true() -> bool {
    true
}

fn default_sync_delta_max() -> f32 {
    0.05
}

fn default_infer_steps() -> u32 {
    1
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
struct ApiCategory {
    name: String,
    #[serde(default)]
    endpoints: Vec<ApiEndpoint>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
struct ApiEndpoint {
    request: String,
    #[serde(default)]
    input: String,
    #[serde(default)]
    output: String,
    #[serde(default)]
    description: String,
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
struct GraphSnapshot {
    #[serde(default)]
    kind: String,
    #[serde(default)]
    nodes: Vec<GraphNode>,
    #[serde(default)]
    edges: Vec<GraphEdge>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
struct GraphNode {
    id: u32,
    #[serde(default)]
    label: String,
    #[serde(default)]
    value: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
struct GraphEdge {
    from: u32,
    to: u32,
    #[serde(default)]
    weight: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct StateSnapshot {
    running: bool,
    mode: String,
    frame: u64,
    #[serde(default)]
    last_error: String,
    target_fps: u32,
    game: GameState,
    hud: HudData,
    brain_stats: BrainStats,
    #[serde(default)]
    unit_plot: Vec<UnitPlotPoint>,
    #[serde(default)]
    action_scores: Vec<ActionScoreBreakdown>,
    #[serde(default)]
    meaning: MeaningSnapshot,

    // Experts (child brains)
    #[serde(default)]
    experts_enabled: bool,
    #[serde(default)]
    experts: experts::ExpertsSummary,
    #[serde(default)]
    active_expert: Option<experts::ActiveExpertSummary>,

    // Storage / snapshots
    #[serde(default)]
    storage: StorageInfo,

    // Advisor / LLM integration (optional)
    #[serde(default)]
    advisor: AdvisorSnapshot,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
struct AdvisorSnapshot {
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
struct StorageInfo {
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

    /// Daemon state wrapper version for the canonical brain file.
    ///
    /// - 0: unknown / missing
    /// - 1: BRSTATE1 (V1)
    /// - 2: BRSTATE2 (V2, LZ4 chunks, optional embedded runtime)
    #[serde(default)]
    state_wrapper_version: u32,
    #[serde(default)]
    snapshots: Vec<SnapshotEntry>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
struct SnapshotEntry {
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
struct MeaningSnapshot {
    #[serde(default)]
    stimulus: String,
    #[serde(default)]
    correct_action: String,

    #[serde(default)]
    action_a_name: String,
    #[serde(default)]
    action_b_name: String,

    #[serde(default)]
    pair_left: RewardEdges,
    #[serde(default)]
    pair_right: RewardEdges,

    #[serde(default)]
    action_left: RewardEdges,
    #[serde(default)]
    action_right: RewardEdges,

    #[serde(default)]
    pair_gap: f32,
    #[serde(default)]
    global_gap: f32,

    /// Trial-sampled history of the correct-vs-wrong pair meaning gap.
    #[serde(default)]
    pair_gap_history: Vec<f32>,
    /// Trial-sampled history of the correct-vs-wrong global action meaning gap.
    #[serde(default)]
    global_gap_history: Vec<f32>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
struct PersistedRuntime {
    #[serde(default)]
    game_kind: String,
    game: PersistedGameStats,

    // Newer format: keep last known stats per game kind.
    // This lets you load a brain and still see how it performed in each task.
    #[serde(default)]
    games: std::collections::HashMap<String, PersistedGameStats>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
struct PersistedGameStats {
    correct: u32,
    incorrect: u32,
    trials: u32,
    recent: Vec<bool>,
    #[serde(default)]
    learning_at_trial: Option<u32>,
    #[serde(default)]
    learned_at_trial: Option<u32>,
    #[serde(default)]
    mastered_at_trial: Option<u32>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
struct GameCommon {
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

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind")]
enum GameState {
    #[serde(rename = "spot")]
    Spot {
        #[serde(flatten)]
        common: GameCommon,
        spot_is_left: bool,
    },
    #[serde(rename = "bandit")]
    Bandit {
        #[serde(flatten)]
        common: GameCommon,
    },
    #[serde(rename = "spot_reversal")]
    SpotReversal {
        #[serde(flatten)]
        common: GameCommon,
        spot_is_left: bool,
    },
    #[serde(rename = "spotxy")]
    SpotXY {
        #[serde(flatten)]
        common: GameCommon,
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
        common: GameCommon,
        pong_ball_x: f32,
        pong_ball_y: f32,
        #[serde(default)]
        pong_ball_visible: bool,
        #[serde(default)]
        pong_hits: u32,
        #[serde(default)]
        pong_misses: u32,
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
        common: GameCommon,
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
        common: GameCommon,
        #[serde(default)]
        replay_dataset: String,
        #[serde(default)]
        replay_index: u32,
        #[serde(default)]
        replay_total: u32,
        #[serde(default)]
        replay_trial_id: String,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct HudData {
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

#[derive(Debug, Clone, Serialize, Deserialize)]
struct BrainStats {
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
    pruned_last_step: usize,
    births_last_step: usize,
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
    causal_base_symbols: usize,
    causal_edges: usize,
    causal_last_directed_edge_updates: usize,
    causal_last_cooccur_edge_updates: usize,
    age_steps: u64,

    // Lightweight learning monitors (optional for clients).
    #[serde(default)]
    plasticity_committed: bool,
    #[serde(default)]
    plasticity_l1: f32,
    #[serde(default)]
    plasticity_edges: u32,
    #[serde(default)]
    plasticity_budget: f32,
    #[serde(default)]
    plasticity_budget_used: f32,
    #[serde(default)]
    eligibility_l1: f32,
    #[serde(default)]
    learning_deadband: f32,
    #[serde(default)]
    homeostasis_rate: f32,
    #[serde(default)]
    homeostasis_bias_l1: f32,
}

// ═══════════════════════════════════════════════════════════════════════════
// Daemon State
// ═══════════════════════════════════════════════════════════════════════════

struct DaemonState {
    brain: Brain,
    experts: ExpertManager,
    game: ActiveGame,
    replay_dataset: ReplayDataset,
    running: bool,
    frame: u64,
    last_error: String,
    last_reward: f32,
    paths: AppPaths,
    exploration_eps: f32,
    meaning_alpha: f32,
    rng_state: u64,
    last_autosave_trial: u32,
    target_fps: u32,
    trial_period_ms: u32,
    pending_neuromod: f32,

    max_units_limit: usize,

    loaded_snapshot_stem: Option<String>,

    persist_state_version: u32,

    view_mode: BrainViewMode,

    meaning_last: MeaningSnapshot,
    meaning_pair_gap_history: Vec<f32>,
    meaning_global_gap_history: Vec<f32>,

    advisor: advisor::AdvisorRuntime,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum BrainViewMode {
    Parent,
    ActiveExpert,
}

impl BrainViewMode {
    fn parse(s: &str) -> Option<Self> {
        match s.trim().to_ascii_lowercase().as_str() {
            "parent" => Some(Self::Parent),
            "active" | "expert" | "active_expert" | "active-expert" => Some(Self::ActiveExpert),
            _ => None,
        }
    }
}

impl DaemonState {
    fn new(paths: AppPaths) -> Self {
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
            ..Default::default()
        });

        brain.define_sensor("spot_left", 4);
        brain.define_sensor("spot_right", 4);
        // Context bit for Spot Reversal (lets the substrate represent regime changes).
        brain.define_sensor("spot_rev_ctx", 2);
        brain.define_sensor("bandit", 4);
        brain.define_action("left", 6);
        brain.define_action("right", 6);
        brain.set_observer_telemetry(true);

        // Execution tier selection.
        // - If BRAINE_EXEC_TIER is set, honor it as a request and warn if unavailable.
        // - Otherwise, auto-select GPU (if compiled+available) with safe fallbacks.
        // Values: scalar|simd|parallel|gpu
        if let Ok(v) = std::env::var("BRAINE_EXEC_TIER") {
            let vv = v.trim().to_ascii_lowercase();
            let requested = match vv.as_str() {
                "scalar" => Some(braine::substrate::ExecutionTier::Scalar),
                "simd" => Some(braine::substrate::ExecutionTier::Simd),
                "parallel" => Some(braine::substrate::ExecutionTier::Parallel),
                "gpu" => Some(braine::substrate::ExecutionTier::Gpu),
                _ => {
                    warn!("Unknown BRAINE_EXEC_TIER value: {}", v);
                    None
                }
            };

            if let Some(t) = requested {
                brain.set_execution_tier(t);
                let effective = brain.effective_execution_tier();
                if effective != t {
                    warn!(
                        "Requested execution tier {:?} but using {:?} (feature/runtime unavailable)",
                        t, effective
                    );
                }
            }
        } else {
            let selected = brain.auto_select_execution_tier();
            tracing::info!("Auto-selected execution tier: {:?}", selected);
        }

        Self {
            brain,
            experts: ExpertManager::new(),
            game: ActiveGame::Spot(SpotGame::new()),
            replay_dataset: ReplayDataset::builtin_left_right_spot(),
            running: false,
            frame: 0,
            last_error: String::new(),
            last_reward: 0.0,
            paths,
            // Exploration controls *random action rate* (epsilon-greedy).
            exploration_eps: 0.2,
            // Meaning weight in action selection; keep stable over time.
            meaning_alpha: 0.2,
            rng_state: 0x9E37_79B9_7F4A_7C15u64 ^ 123u64,
            last_autosave_trial: 0,
            target_fps: 60,
            trial_period_ms: 250,
            pending_neuromod: 0.0,

            max_units_limit: 256,

            loaded_snapshot_stem: None,

            persist_state_version: state_image::VERSION_V1,

            meaning_last: MeaningSnapshot::default(),
            meaning_pair_gap_history: Vec::with_capacity(96),
            meaning_global_gap_history: Vec::with_capacity(96),
            view_mode: BrainViewMode::Parent,

            advisor: advisor::AdvisorRuntime::new_from_env(),
        }
    }

    fn current_stimulus_key<'a>(&'a self) -> std::borrow::Cow<'a, str> {
        let base = self.game.stimulus_name();
        if self.game.kind() == "spot_reversal" && self.game.reversal_active() {
            return std::borrow::Cow::Owned(format!("{}::rev", base));
        }
        if let Some(k) = self.game.stimulus_key() {
            return std::borrow::Cow::Owned(k.to_string());
        }
        std::borrow::Cow::Borrowed(base)
    }

    fn view_brain_for_context<'a>(&'a self, context_key: &str) -> &'a Brain {
        match self.view_mode {
            BrainViewMode::Parent => &self.brain,
            BrainViewMode::ActiveExpert => {
                if !self.experts.enabled() {
                    &self.brain
                } else {
                    self.experts
                        .controller_for_context_ref(context_key, &self.brain)
                        .brain
                }
            }
        }
    }

    fn compute_meaning_snapshot_vs(
        &self,
        brain: &Brain,
        stimulus: &str,
        correct_action: &str,
        wrong_or_chosen_action: &str,
    ) -> MeaningSnapshot {
        let a = correct_action;
        let b = if !wrong_or_chosen_action.is_empty() && wrong_or_chosen_action != correct_action {
            wrong_or_chosen_action
        } else {
            // Pick a second action deterministically.
            // Prefer symmetric pairs for known action sets.
            match correct_action {
                "left" => "right",
                "right" => "left",
                "up" => "down",
                "down" => "up",
                "stay" => "up",
                _ => self
                    .game
                    .allowed_actions()
                    .iter()
                    .find(|x| x.as_str() != correct_action)
                    .map(|s| s.as_str())
                    .unwrap_or("left"),
            }
        };

        let pair_a = brain.pair_reward_edges(stimulus, a);
        let pair_b = brain.pair_reward_edges(stimulus, b);

        let action_a = brain.action_reward_edges(a);
        let action_b = brain.action_reward_edges(b);

        let pair_gap = pair_a.meaning - pair_b.meaning;
        let global_gap = action_a.meaning - action_b.meaning;

        MeaningSnapshot {
            stimulus: stimulus.to_string(),
            correct_action: correct_action.to_string(),

            action_a_name: a.to_string(),
            action_b_name: b.to_string(),

            // UI still expects left/right slots; interpret them as A/B.
            pair_left: pair_a,
            pair_right: pair_b,
            action_left: action_a,
            action_right: action_b,

            pair_gap,
            global_gap,
            pair_gap_history: Vec::new(),
            global_gap_history: Vec::new(),
        }
    }

    fn set_game(&mut self, game: &str) -> Result<(), String> {
        let g = game.trim().to_ascii_lowercase();
        match g.as_str() {
            "spot" => self.game = ActiveGame::Spot(SpotGame::new()),
            "bandit" => self.game = ActiveGame::Bandit(BanditGame::new()),
            "spot_reversal" | "reversal" | "spot-reversal" => {
                self.game = ActiveGame::SpotReversal(SpotReversalGame::new(200))
            }
            "spotxy" | "spot_xy" | "spot-xy" => {
                self.ensure_spotxy_io();
                self.game = ActiveGame::SpotXY(SpotXYGame::new(16));
            }
            "pong" => {
                self.ensure_pong_io();
                self.game = ActiveGame::Pong(PongGame::new());
            }
            "text" | "text_next_token" | "text-next-token" => {
                let gg = TextNextTokenGame::new();
                self.ensure_text_io(&gg);
                self.game = ActiveGame::Text(gg);
            }
            "replay" => {
                let gg = ReplayGame::new(self.replay_dataset.clone());
                self.ensure_replay_io();
                self.game = ActiveGame::Replay(gg);
            }
            _ => {
                return Err(format!(
                    "Unknown game '{game}'. Use spot|bandit|spot_reversal|spotxy|pong|text|replay"
                ))
            }
        }

        // New task => reset meaning history so plots represent the current game.
        self.meaning_last = MeaningSnapshot::default();
        self.meaning_pair_gap_history.clear();
        self.meaning_global_gap_history.clear();
        self.last_reward = 0.0;
        // Prevent autosave underflow if the new game's trial counter resets.
        self.last_autosave_trial = self.game.stats().trials;
        Ok(())
    }

    fn ensure_spotxy_io(&mut self) {
        let k = 16usize;
        for i in 0..k {
            self.brain
                .ensure_sensor_min_width(&format!("pos_x_{i:02}"), 3);
            self.brain
                .ensure_sensor_min_width(&format!("pos_y_{i:02}"), 3);
        }
        self.brain.ensure_action_min_width("left", 6);
        self.brain.ensure_action_min_width("right", 6);

        if let ActiveGame::SpotXY(g) = &self.game {
            for name in g.allowed_actions() {
                self.brain.ensure_action_min_width(name, 6);
            }
        }
    }

    fn ensure_pong_io(&mut self) {
        // Bin sensors (must match PongGame constants).
        let bins = 8u32;
        for i in 0..bins {
            self.brain
                .ensure_sensor_min_width(&format!("pong_ball_x_{i:02}"), 3);
            self.brain
                .ensure_sensor_min_width(&format!("pong_ball_y_{i:02}"), 3);
            self.brain
                .ensure_sensor_min_width(&format!("pong_ball2_x_{i:02}"), 3);
            self.brain
                .ensure_sensor_min_width(&format!("pong_ball2_y_{i:02}"), 3);
            self.brain
                .ensure_sensor_min_width(&format!("pong_paddle_y_{i:02}"), 3);

            self.brain
                .ensure_sensor_min_width(&format!("pong_target_y_{i:02}"), 3);
        }

        self.brain.ensure_sensor_min_width("pong_target_na", 2);
        self.brain.ensure_sensor_min_width("pong_evt_hit", 2);
        self.brain.ensure_sensor_min_width("pong_evt_miss", 2);
        self.brain.ensure_sensor_min_width("pong_ball_visible", 2);
        self.brain.ensure_sensor_min_width("pong_ball_hidden", 2);
        self.brain.ensure_sensor_min_width("pong_ball2_visible", 2);
        self.brain.ensure_sensor_min_width("pong_ball2_hidden", 2);

        self.brain.ensure_sensor_min_width("pong_vx_pos", 2);
        self.brain.ensure_sensor_min_width("pong_vx_neg", 2);
        self.brain.ensure_sensor_min_width("pong_vy_pos", 2);
        self.brain.ensure_sensor_min_width("pong_vy_neg", 2);

        self.brain.ensure_sensor_min_width("pong_ball2_vx_pos", 2);
        self.brain.ensure_sensor_min_width("pong_ball2_vx_neg", 2);
        self.brain.ensure_sensor_min_width("pong_ball2_vy_pos", 2);
        self.brain.ensure_sensor_min_width("pong_ball2_vy_neg", 2);

        self.brain.ensure_action_min_width("up", 6);
        self.brain.ensure_action_min_width("down", 6);
        self.brain.ensure_action_min_width("stay", 6);
    }

    fn ensure_text_io(&mut self, g: &TextNextTokenGame) {
        self.brain.ensure_sensor_min_width("txt_regime_0", 3);
        self.brain.ensure_sensor_min_width("txt_regime_1", 3);
        for name in g.token_sensor_names() {
            self.brain.ensure_sensor_min_width(name, 3);
        }
        for name in g.allowed_actions() {
            self.brain.ensure_action_min_width(name, 6);
        }
    }

    fn ensure_replay_io(&mut self) {
        use std::collections::BTreeSet;

        let mut sensors: BTreeSet<String> = BTreeSet::new();
        let mut actions: BTreeSet<String> = BTreeSet::new();

        for tr in &self.replay_dataset.trials {
            for s in &tr.stimuli {
                sensors.insert(s.name.clone());
            }
            for a in &tr.allowed_actions {
                actions.insert(a.clone());
            }
            if !tr.correct_action.trim().is_empty() {
                actions.insert(tr.correct_action.clone());
            }
        }

        for name in sensors {
            self.brain.ensure_sensor_min_width(&name, 3);
        }
        for name in actions {
            self.brain.ensure_action_min_width(&name, 6);
        }
    }

    fn push_history(buf: &mut Vec<f32>, v: f32, cap: usize) {
        buf.push(v);
        if buf.len() > cap {
            let drop = buf.len() - cap;
            buf.drain(0..drop);
        }
    }

    #[inline]
    fn rng_next_u64(&mut self) -> u64 {
        // xorshift64* (fast, dependency-free; fine for epsilon-greedy exploration)
        let mut x = self.rng_state;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.rng_state = x;
        x.wrapping_mul(2685821657736338717)
    }

    #[inline]
    fn rng_next_f32(&mut self) -> f32 {
        // Uniform-ish in [0,1)
        let x = self.rng_next_u64();
        let mantissa = (x >> 40) as u32; // 24 bits
        (mantissa as f32) / ((1u32 << 24) as f32)
    }

    fn tick(&mut self) {
        if !self.running {
            return;
        }

        // Advance trials on a wall-clock schedule (independent of FPS) so
        // action selection uses the same stimulus the brain will see.
        self.game.update_timing(self.trial_period_ms);

        // SpotXY eval mode is a holdout run: no causal/meaning writes.
        let allow_learning = !self.game.spotxy_eval_mode();

        let base_stimulus = self.game.stimulus_name();
        let stimulus_key_owned: Option<String> =
            if self.game.kind() == "spot_reversal" && self.game.reversal_active() {
                Some(format!("{}::rev", base_stimulus))
            } else {
                self.game.stimulus_key().map(|s| s.to_string())
            };
        let stimulus_key = stimulus_key_owned.as_deref().unwrap_or(base_stimulus);
        let context_key = stimulus_key;

        // Precompute RNG decisions before borrowing the controller brain mutably.
        let need_action = !self.game.response_made();
        let (explore, rand_idx) = if need_action {
            (
                self.rng_next_f32() < self.exploration_eps,
                self.rng_next_u64() as usize,
            )
        } else {
            (false, 0usize)
        };

        // Run the stimulus → step → action-select → reward/commit loop on the controller brain.
        let mut completed = false;
        let mut scored_reward: Option<f32> = None;
        let mut controller_path: Vec<u32> = Vec::new();
        let mut controller_scale: f32 = 1.0;
        let mut controller_is_expert: bool = false;
        {
            // Choose controller brain (supports nested experts).
            let mut ctrl_opt: Option<experts::ControllerBorrow<'_>> = if self.experts.enabled() {
                Some(
                    self.experts
                        .controller_for_context_mut(context_key, &mut self.brain),
                )
            } else {
                None
            };
            let brain: &mut Brain = if let Some(c) = ctrl_opt.as_mut() {
                controller_is_expert = !c.route.path.is_empty();
                controller_scale = c.route.reward_scale;
                controller_path = c.route.path.clone();
                c.brain
            } else {
                &mut self.brain
            };

            // If Pong produced a hit/miss since the last tick, immediately credit the held action.
            // This reduces the "ball hit but paddle didn't learn" effect from delayed reward.
            if allow_learning {
                if let ActiveGame::Pong(g) = &mut self.game {
                    if let Some((reward, action, stimulus_key, _ev)) = g.take_pending_credit() {
                        let r = reward.clamp(-1.0, 1.0);
                        if r.abs() > 0.0 {
                            brain.note_action(action.as_str());
                            brain.note_compound_symbol(&[
                                "pair",
                                stimulus_key.as_str(),
                                action.as_str(),
                            ]);
                            brain.set_neuromodulator(r);
                            brain.reinforce_action(action.as_str(), r);
                            self.pending_neuromod = r;
                            self.last_reward = r;
                        }
                    }
                }
            }

            // Apply last tick's reward as neuromodulation for one step.
            brain.set_neuromodulator(self.pending_neuromod);
            self.pending_neuromod = 0.0;

            // Observe stimulus and advance dynamics.
            match &self.game {
                ActiveGame::SpotXY(g) => {
                    g.apply_stimuli(brain);
                    brain.note_compound_symbol(&[stimulus_key]);
                }
                ActiveGame::Pong(g) => {
                    g.apply_stimuli(brain);
                    brain.note_compound_symbol(&[stimulus_key]);
                }
                ActiveGame::Text(g) => {
                    g.apply_stimuli(brain);
                    brain.note_compound_symbol(&[stimulus_key]);
                }
                ActiveGame::Replay(g) => {
                    g.apply_stimuli(brain);
                    brain.note_compound_symbol(&[stimulus_key]);
                }
                _ => {
                    brain.apply_stimulus(Stimulus::new(base_stimulus, 1.0));
                    if self.game.kind() == "spot_reversal" && self.game.reversal_active() {
                        brain.apply_stimulus(Stimulus::new("spot_rev_ctx", 1.0));
                    }
                    if let Some(ref k) = stimulus_key_owned {
                        brain.note_compound_symbol(&[k.as_str()]);
                    }
                }
            }
            brain.step();

            // Decide and (optionally) score once per trial.
            if !self.game.response_made() {
                // Epsilon-greedy: explore randomly sometimes; otherwise exploit meaning+habit.
                let action_name = if explore {
                    let allowed = self.game.allowed_actions();
                    if allowed.is_empty() {
                        "idle".to_string()
                    } else {
                        allowed[rand_idx % allowed.len()].clone()
                    }
                } else {
                    let ranked = brain.ranked_actions_with_meaning(context_key, self.meaning_alpha);
                    let allowed = self.game.allowed_actions();

                    let mut top1: Option<(String, f32)> = None;
                    for (name, score) in ranked {
                        if !allowed.iter().any(|a| a == &name) {
                            continue;
                        }
                        if top1.is_none() {
                            top1 = Some((name, score));
                        } else {
                            break;
                        }
                    }

                    let picked = top1
                        .as_ref()
                        .map(|(n, _s)| n.clone())
                        .or_else(|| allowed.first().cloned())
                        .unwrap_or_else(|| "idle".to_string());

                    picked
                };

                // Score once per trial.
                if let Some((reward, done)) = self
                    .game
                    .score_action(action_name.as_str(), self.trial_period_ms)
                {
                    completed = done;
                    self.last_reward = reward;
                    scored_reward = Some(reward);

                    if allow_learning {
                        let learn_reward = if controller_is_expert {
                            (reward * controller_scale).clamp(-1.0, 1.0)
                        } else {
                            reward
                        };

                        brain.note_action(action_name.as_str());
                        brain.note_compound_symbol(&["pair", stimulus_key, action_name.as_str()]);

                        brain.set_neuromodulator(learn_reward);
                        brain.reinforce_action(action_name.as_str(), learn_reward);
                        self.pending_neuromod = learn_reward;
                    } else {
                        brain.set_neuromodulator(0.0);
                        self.pending_neuromod = 0.0;
                    }
                } else {
                    if allow_learning {
                        brain.note_action(action_name.as_str());
                        brain.note_compound_symbol(&["pair", stimulus_key, action_name.as_str()]);
                    }
                    brain.set_neuromodulator(0.0);
                }
            } else {
                brain.set_neuromodulator(0.0);
            }

            // Commit or discard perception/action/reward symbols on the controller.
            if allow_learning {
                brain.commit_observation();
            } else {
                brain.discard_observation();
            }
        }

        // Experts may only spawn on explicit novelty/shift/collapse/saturation signals.
        // Evaluate only on trial completion and only when learning is enabled.
        if self.experts.enabled() && completed && allow_learning {
            if let Some(r) = scored_reward {
                let trials = self.game.stats().trials;
                self.experts.note_trial_for_spawn_target_under_path(
                    context_key,
                    &controller_path,
                    trials,
                    r,
                );
                self.experts.maybe_spawn_for_signals_under_path(
                    context_key,
                    &controller_path,
                    trials,
                    &self.brain,
                );
            }
        }

        // Tick expert cooldowns on completed trials.
        self.experts.tick_cooldowns(completed);

        // Update expert evaluation and (optionally) consolidate at trial boundaries.
        if completed {
            if let (false, Some(r)) = (controller_path.is_empty(), scored_reward) {
                self.experts
                    .on_trial_completed_path(&controller_path, r, &mut self.brain);
            }
        }

        if completed {
            let chosen = self.game.last_action().unwrap_or("");
            let controller_brain = if self.experts.enabled() {
                self.experts
                    .controller_for_context_ref(context_key, &self.brain)
                    .brain
            } else {
                &self.brain
            };
            let m = self.compute_meaning_snapshot_vs(
                controller_brain,
                stimulus_key,
                self.game.correct_action().as_ref(),
                chosen,
            );
            Self::push_history(&mut self.meaning_pair_gap_history, m.pair_gap, 96);
            Self::push_history(&mut self.meaning_global_gap_history, m.global_gap, 96);
            self.meaning_last = m;
        }

        if completed {
            // Advisor (LLM integration point): slow-loop, bounded config nudges.
            // Must not directly choose actions; only updates a small set of knobs.
            if allow_learning {
                let trials = self.game.stats().trials;
                if self.advisor.should_invoke(trials) {
                    let text_regime = match &self.game {
                        ActiveGame::Text(g) => Some(g.regime()),
                        _ => None,
                    };
                    let ctx = advisor::AdvisorContext {
                        game: self.game.kind().to_string(),
                        context_key: context_key.to_string(),
                        trials,
                        accuracy: self.game.stats().accuracy(),
                        recent_rate: self.game.stats().recent_rate(),
                        last_reward: self.last_reward,
                        exploration_eps: self.exploration_eps,
                        meaning_alpha: self.meaning_alpha,
                        text_regime,
                    };

                    // Auto-invocation always applies.
                    let report = self.advisor.invoke(ctx, trials, true);
                    self.apply_advice(&report.advice);
                }
            }

            // Anneal exploration but keep a small floor for on-policy correction
            self.exploration_eps = (self.exploration_eps * 0.99).max(0.02);

            if allow_learning {
                // Automatic, bounded neurogenesis: add capacity if the network is saturating.
                // This keeps external influence minimal while preventing long-run brittleness.
                // Tuning notes:
                // - lower threshold => grows earlier
                // - higher max_units => allows more capacity but costs memory/compute
                // When experts are enabled, we avoid topology-changing growth until the merge
                // story is more robust.
                if !self.experts.enabled() {
                    let _grown = self.brain.maybe_neurogenesis(0.35, 1, self.max_units_limit);
                }

                // Auto-save frequently so short sessions still persist.
                let trials = self.game.stats().trials;
                let trials_since_save = trials.saturating_sub(self.last_autosave_trial);
                if trials_since_save >= 10 {
                    match self.save_brain() {
                        Ok(_) => {
                            self.last_autosave_trial = trials;
                        }
                        Err(e) => {
                            let msg = format!("Auto-save FAILED at trial {trials}: {e}");
                            self.last_error = msg.clone();
                            error!("✗ {}", msg);
                        }
                    }
                }
            }
        }

        self.frame += 1;
    }

    fn apply_advice(&mut self, advice: &advisor::AdvisorAdvice) {
        if let Some(v) = advice.exploration_eps {
            self.exploration_eps = v.clamp(0.0, 1.0);
        }
        if let Some(v) = advice.meaning_alpha {
            self.meaning_alpha = v.clamp(0.0, 50.0);
        }
    }

    fn get_snapshot(&self) -> StateSnapshot {
        let stimulus_key = self.current_stimulus_key();
        let stimulus = stimulus_key.as_ref();

        let view_brain = self.view_brain_for_context(stimulus);
        let diag = view_brain.diagnostics();
        let causal = view_brain.causal_stats();
        let active_expert = if self.experts.enabled() {
            self.experts.active_expert_summary(stimulus)
        } else {
            None
        };
        let stats = self.game.stats();

        let common = || GameCommon {
            reversal_active: self.game.reversal_active(),
            chosen_action: self.game.last_action().unwrap_or("").to_string(),
            last_reward: self.last_reward,
            response_made: self.game.response_made(),
            trial_frame: self.game.trial_frame(),
            trial_duration: self.trial_period_ms,
        };

        let game = match &self.game {
            ActiveGame::Spot(_) => GameState::Spot {
                common: common(),
                spot_is_left: self.game.spot_is_left(),
            },
            ActiveGame::Bandit(_) => GameState::Bandit { common: common() },
            ActiveGame::SpotReversal(_) => GameState::SpotReversal {
                common: common(),
                spot_is_left: self.game.spot_is_left(),
            },
            ActiveGame::SpotXY(_) => {
                let (pos_x, pos_y) = self.game.pos_xy().unwrap_or((0.0, 0.0));
                GameState::SpotXY {
                    common: common(),
                    pos_x,
                    pos_y,
                    spotxy_eval: self.game.spotxy_eval_mode(),
                    spotxy_mode: self.game.spotxy_mode_name().to_string(),
                    spotxy_grid_n: self.game.spotxy_grid_n(),
                }
            }
            ActiveGame::Pong(g) => GameState::Pong {
                common: common(),
                pong_ball_x: g.sim.state.ball_x,
                pong_ball_y: g.sim.state.ball_y,
                pong_ball_visible: g.ball_visible(),
                pong_hits: g.hits(),
                pong_misses: g.misses(),
                pong_ball2_x: g.sim.state.ball2_x,
                pong_ball2_y: g.sim.state.ball2_y,
                pong_ball2_visible: g.ball2_visible(),
                pong_ball2_enabled: g.ball2_enabled(),
                pong_paddle_y: g.sim.state.paddle_y,
                pong_paddle_half_height: g.sim.params.paddle_half_height,
                pong_paddle_speed: g.sim.params.paddle_speed,
                pong_ball_speed: g.sim.params.ball_speed,
            },
            ActiveGame::Text(g) => GameState::Text {
                common: common(),
                text_regime: g.regime(),
                text_token: g.current_token().display(),
                text_target_next: g.target_next_token().display(),
                text_outcomes: g.outcomes(),
                text_shift_every: g.shift_every_outcomes(),
                text_vocab_size: g.vocab_size() as u32,
            },
            ActiveGame::Replay(g) => GameState::Replay {
                common: common(),
                replay_dataset: g.dataset_name().to_string(),
                replay_index: g.index() as u32,
                replay_total: g.total_trials() as u32,
                replay_trial_id: g.current_trial_id().to_string(),
            },
        };

        let (osc_x, osc_y, osc_mag) = view_brain.oscillation_sample(512);

        StateSnapshot {
            running: self.running,
            mode: "braine".to_string(),
            frame: self.frame,
            last_error: self.last_error.clone(),
            target_fps: self.target_fps,
            game,
            hud: HudData {
                trials: stats.trials,
                correct: stats.correct,
                incorrect: stats.incorrect,
                accuracy: stats.accuracy(),
                recent_rate: stats.recent_rate(),
                last_100_rate: stats.last_100_rate(),
                neuromod: view_brain.neuromodulator(),
                learning_at_trial: stats.learning_at_trial.map(|v| v as i32).unwrap_or(-1),
                learned_at_trial: stats.learned_at_trial.map(|v| v as i32).unwrap_or(-1),
                mastered_at_trial: stats.mastered_at_trial.map(|v| v as i32).unwrap_or(-1),
            },
            brain_stats: {
                let selected = view_brain.execution_tier();
                let effective = view_brain.effective_execution_tier();
                let learning = view_brain.learning_stats();
                let cfg = view_brain.config();

                BrainStats {
                    unit_count: diag.unit_count,
                    max_units_limit: self.max_units_limit,
                    execution_tier: match effective {
                        braine::substrate::ExecutionTier::Scalar => "Scalar",
                        braine::substrate::ExecutionTier::Simd => "SIMD",
                        braine::substrate::ExecutionTier::Parallel => "Parallel",
                        braine::substrate::ExecutionTier::Gpu => "GPU",
                    }
                    .to_string(),
                    execution_tier_selected: match selected {
                        braine::substrate::ExecutionTier::Scalar => "Scalar",
                        braine::substrate::ExecutionTier::Simd => "SIMD",
                        braine::substrate::ExecutionTier::Parallel => "Parallel",
                        braine::substrate::ExecutionTier::Gpu => "GPU",
                    }
                    .to_string(),
                    execution_tier_effective: match effective {
                        braine::substrate::ExecutionTier::Scalar => "Scalar",
                        braine::substrate::ExecutionTier::Simd => "SIMD",
                        braine::substrate::ExecutionTier::Parallel => "Parallel",
                        braine::substrate::ExecutionTier::Gpu => "GPU",
                    }
                    .to_string(),
                    connection_count: diag.connection_count,
                    pruned_last_step: diag.pruned_last_step,
                    births_last_step: diag.births_last_step,
                    saturated: view_brain.should_grow(0.35),
                    avg_amp: diag.avg_amp,
                    avg_weight: diag.avg_weight,
                    osc_x,
                    osc_y,
                    osc_mag,
                    memory_bytes: diag.memory_bytes,
                    causal_base_symbols: causal.base_symbols,
                    causal_edges: causal.edges,
                    causal_last_directed_edge_updates: causal.last_directed_edge_updates,
                    causal_last_cooccur_edge_updates: causal.last_cooccur_edge_updates,
                    age_steps: view_brain.age_steps(),

                    plasticity_committed: learning.plasticity_committed,
                    plasticity_l1: learning.plasticity_l1,
                    plasticity_edges: learning.plasticity_edges,
                    plasticity_budget: learning.plasticity_budget,
                    plasticity_budget_used: learning.plasticity_budget_used,
                    eligibility_l1: learning.eligibility_l1,
                    learning_deadband: cfg.learning_deadband,
                    homeostasis_rate: cfg.homeostasis_rate,
                    homeostasis_bias_l1: learning.homeostasis_bias_l1,
                }
            },
            unit_plot: view_brain.unit_plot_points(128),
            action_scores: view_brain.action_score_breakdown(stimulus, self.meaning_alpha),
            meaning: {
                let chosen = self.game.last_action().unwrap_or("");
                let mut m = self.compute_meaning_snapshot_vs(
                    view_brain,
                    stimulus,
                    self.game.correct_action().as_ref(),
                    chosen,
                );
                m.pair_gap_history = self.meaning_pair_gap_history.clone();
                m.global_gap_history = self.meaning_global_gap_history.clone();
                m
            },

            experts_enabled: self.experts.enabled(),
            experts: self.experts.summary(),
            active_expert,

            storage: self.storage_info(),

            advisor: {
                let cfg = self.advisor.status();
                let last_rationale = self
                    .advisor
                    .last_report
                    .as_ref()
                    .map(|r| r.advice.rationale.clone())
                    .unwrap_or_default();
                AdvisorSnapshot {
                    enabled: cfg.enabled,
                    mode: cfg.mode,
                    every_trials: cfg.every_trials,
                    last_rationale,
                }
            },
        }
    }

    fn detect_state_wrapper_version(path: &Path) -> u32 {
        let mut file = match File::open(path) {
            Ok(f) => f,
            Err(_) => return 0,
        };
        let mut magic = [0u8; 8];
        if file.read_exact(&mut magic).is_err() {
            return 0;
        }
        if !state_image::is_state_magic(&magic) {
            return 0;
        }
        storage::read_u32_le(&mut file).unwrap_or_default()
    }

    fn snapshots_dir(&self) -> PathBuf {
        self.paths.data_dir().join("snapshots")
    }

    fn brain_snapshot_path(dir: &Path, stem: &str) -> PathBuf {
        dir.join(format!("brain_{stem}.bbi"))
    }

    fn runtime_snapshot_path(dir: &Path, stem: &str) -> PathBuf {
        dir.join(format!("runtime_{stem}.json"))
    }

    fn file_size_bytes(path: &Path) -> u64 {
        std::fs::metadata(path).map(|m| m.len()).unwrap_or(0)
    }

    fn list_snapshots(&self, limit: usize) -> Vec<SnapshotEntry> {
        let dir = self.snapshots_dir();
        let rd = match std::fs::read_dir(&dir) {
            Ok(rd) => rd,
            Err(_) => return Vec::new(),
        };

        // Collect stems by scanning brain_*.bbi files.
        let mut stems: Vec<(String, u64)> = Vec::new();
        for ent in rd.flatten() {
            let path = ent.path();
            if path.extension().and_then(|s| s.to_str()) != Some("bbi") {
                continue;
            }
            let file_name = path.file_name().and_then(|s| s.to_str()).unwrap_or("");
            let Some(stem) = file_name
                .strip_prefix("brain_")
                .and_then(|s| s.strip_suffix(".bbi"))
            else {
                continue;
            };

            let modified_unix = std::fs::metadata(&path)
                .ok()
                .and_then(|m| m.modified().ok())
                .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
                .map(|d| d.as_secs())
                .unwrap_or(0);

            stems.push((stem.to_string(), modified_unix));
        }

        // Newest first.
        stems.sort_by(|a, b| b.1.cmp(&a.1));
        stems.truncate(limit);

        stems
            .into_iter()
            .map(|(stem, modified_unix)| {
                let b = Self::brain_snapshot_path(&dir, &stem);
                let r = Self::runtime_snapshot_path(&dir, &stem);
                SnapshotEntry {
                    stem,
                    brain_bytes: Self::file_size_bytes(&b),
                    runtime_bytes: Self::file_size_bytes(&r),
                    modified_unix,
                }
            })
            .collect()
    }

    fn storage_info(&self) -> StorageInfo {
        let data_dir = self.paths.data_dir().to_string_lossy().to_string();
        let brain_file = self.paths.brain_file().to_string_lossy().to_string();
        let runtime_file = self
            .paths
            .runtime_state_file()
            .to_string_lossy()
            .to_string();
        let loaded_snapshot = self.loaded_snapshot_stem.clone().unwrap_or_default();
        let state_wrapper_version = Self::detect_state_wrapper_version(Path::new(&brain_file));
        StorageInfo {
            data_dir,
            brain_file: brain_file.clone(),
            runtime_file: runtime_file.clone(),
            loaded_snapshot,
            brain_bytes: Self::file_size_bytes(Path::new(&brain_file)),
            runtime_bytes: Self::file_size_bytes(Path::new(&runtime_file)),
            state_wrapper_version,
            snapshots: self.list_snapshots(24),
        }
    }

    fn migrate_state_format(&mut self, target_state_version: u32) -> Result<(), String> {
        if self.running {
            return Err("Stop the daemon before migrating storage format".to_string());
        }
        if target_state_version != state_image::VERSION_V1
            && target_state_version != state_image::VERSION_V2
        {
            return Err("Unsupported target_state_version".to_string());
        }

        // Ensure canonical files are up to date prior to rewrite.
        self.save_brain()?;

        let canonical = self.paths.brain_file();
        self.rewrite_state_file_in_place(
            &canonical,
            Some(self.paths.runtime_state_file()),
            target_state_version,
        )?;

        let snap_dir = self.snapshots_dir();
        if snap_dir.exists() {
            let entries = std::fs::read_dir(&snap_dir)
                .map_err(|e| format!("Failed to read snapshots dir {:?}: {e}", snap_dir))?;
            for ent in entries.flatten() {
                let p = ent.path();
                if !p.is_file() {
                    continue;
                }
                let name = p.file_name().and_then(|s| s.to_str()).unwrap_or("");
                if !name.starts_with("brain_") || !name.ends_with(".bbi") {
                    continue;
                }
                let stem = name
                    .strip_prefix("brain_")
                    .and_then(|s| s.strip_suffix(".bbi"))
                    .unwrap_or("");
                let rt = Self::runtime_snapshot_path(&snap_dir, stem);
                self.rewrite_state_file_in_place(&p, Some(rt), target_state_version)?;
            }
        }

        self.persist_state_version = target_state_version;
        // Ensure subsequent saves stay in the migrated format.
        self.save_brain()?;
        Ok(())
    }

    fn rewrite_state_file_in_place(
        &self,
        path: &Path,
        runtime_path: Option<PathBuf>,
        target_state_version: u32,
    ) -> Result<(), String> {
        let mut file =
            File::open(path).map_err(|e| format!("Failed to open brain file {:?}: {e}", path))?;
        let mut magic = [0u8; 8];
        file.read_exact(&mut magic)
            .map_err(|e| format!("Failed to read file header {:?}: {e}", path))?;
        file.seek(SeekFrom::Start(0))
            .map_err(|e| format!("Failed to seek file {:?}: {e}", path))?;

        let (brain, experts_state, embedded_runtime) = if state_image::is_state_magic(&magic) {
            let loaded = state_image::load_state_from(&mut file)
                .map_err(|e| format!("Failed to load daemon state {:?}: {e}", path))?;
            let ex = loaded.experts_state.unwrap_or_default();
            (loaded.brain, ex, loaded.runtime_state)
        } else {
            // Legacy: brain image only.
            let b = Brain::load_image_from(&mut file)
                .map_err(|e| format!("Failed to load legacy brain image {:?}: {e}", path))?;
            (b, Vec::new(), None)
        };

        let runtime_bytes: Option<Vec<u8>> = runtime_path
            .as_ref()
            .and_then(|p| std::fs::read(p).ok())
            .or(embedded_runtime);

        let tmp = path.with_extension("bbi.tmp");
        {
            let mut out = File::create(&tmp)
                .map_err(|e| format!("Failed to create temp file {:?}: {e}", tmp))?;
            state_image::save_state_to_with_version(
                &mut out,
                &brain,
                &experts_state,
                runtime_bytes.as_deref(),
                target_state_version,
            )
            .map_err(|e| format!("Failed to write migrated state {:?}: {e}", tmp))?;
        }
        std::fs::rename(&tmp, path)
            .map_err(|e| format!("Failed to replace file {:?}: {e}", path))?;
        Ok(())
    }

    fn save_snapshot(&self) -> Result<String, String> {
        // Ensure the canonical files are current, then copy them into snapshots/.
        self.save_brain()?;

        let dir = self.snapshots_dir();
        std::fs::create_dir_all(&dir)
            .map_err(|e| format!("Failed to create snapshots dir {:?}: {e}", dir))?;

        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_err(|e| format!("System time error: {e}"))?;
        let stem = format!("{}_{}", now.as_secs(), now.subsec_millis());

        let src_brain = self.paths.brain_file();
        let src_rt = self.paths.runtime_state_file();

        let dst_brain = Self::brain_snapshot_path(&dir, &stem);
        let dst_rt = Self::runtime_snapshot_path(&dir, &stem);

        std::fs::copy(&src_brain, &dst_brain).map_err(|e| {
            format!(
                "Failed to copy brain snapshot {:?} -> {:?}: {e}",
                src_brain, dst_brain
            )
        })?;

        if src_rt.exists() {
            std::fs::copy(&src_rt, &dst_rt).map_err(|e| {
                format!(
                    "Failed to copy runtime snapshot {:?} -> {:?}: {e}",
                    src_rt, dst_rt
                )
            })?;
        }

        Ok(stem)
    }

    fn load_snapshot(&mut self, stem: &str) -> Result<(), String> {
        let dir = self.snapshots_dir();
        let src_brain = Self::brain_snapshot_path(&dir, stem);
        if !src_brain.exists() {
            return Err(format!("Snapshot not found: {}", stem));
        }
        let src_rt = Self::runtime_snapshot_path(&dir, stem);

        // Copy snapshot files into the canonical locations, then load.
        std::fs::copy(&src_brain, self.paths.brain_file()).map_err(|e| {
            format!(
                "Failed to restore snapshot brain {:?} -> {:?}: {e}",
                src_brain,
                self.paths.brain_file()
            )
        })?;

        if src_rt.exists() {
            std::fs::copy(&src_rt, self.paths.runtime_state_file()).map_err(|e| {
                format!(
                    "Failed to restore snapshot runtime {:?} -> {:?}: {e}",
                    src_rt,
                    self.paths.runtime_state_file()
                )
            })?;
        }

        self.load_brain()?;
        self.loaded_snapshot_stem = Some(stem.to_string());
        Ok(())
    }

    fn get_graph_snapshot(
        &self,
        kind: &str,
        max_nodes: usize,
        max_edges: usize,
        include_isolated: bool,
    ) -> GraphSnapshot {
        let stimulus_key = self.current_stimulus_key();
        let context_key = stimulus_key.as_ref();
        let view_brain = self.view_brain_for_context(context_key);

        match kind {
            "causal" => {
                self.get_causal_graph_snapshot(view_brain, max_nodes, max_edges, include_isolated)
            }
            _ => self.get_substrate_graph_snapshot(
                view_brain,
                max_nodes,
                max_edges,
                include_isolated,
            ),
        }
    }

    fn get_substrate_graph_snapshot(
        &self,
        brain: &Brain,
        max_nodes: usize,
        max_edges: usize,
        include_isolated: bool,
    ) -> GraphSnapshot {
        use std::collections::{HashMap, HashSet};

        let diag = brain.diagnostics();
        let n = diag.unit_count;
        if n == 0 {
            return GraphSnapshot {
                kind: "substrate".to_string(),
                nodes: Vec::new(),
                edges: Vec::new(),
            };
        }

        // Build a connected view deterministically:
        // 1) collect strong candidate edges across all units (no node sampling)
        // 2) keep a pool of strongest edges
        // 3) pick up to max_nodes endpoints by incident strength
        // 4) filter to edges within kept nodes, then truncate to max_edges
        if max_edges == 0 || max_nodes == 0 {
            return GraphSnapshot {
                kind: "substrate".to_string(),
                nodes: Vec::new(),
                edges: Vec::new(),
            };
        }

        let per_node_top = 8usize;
        let mut candidates: Vec<GraphEdge> = Vec::new();
        for from in 0..n {
            let mut local: Vec<(u32, f32)> = Vec::new();
            for (to, w) in brain.neighbors(from) {
                if to == from {
                    continue;
                }
                local.push((to as u32, w));
            }

            local.sort_by(|(to_a, w_a), (to_b, w_b)| {
                w_b.abs().total_cmp(&w_a.abs()).then_with(|| to_a.cmp(to_b))
            });

            for (to, w) in local.into_iter().take(per_node_top) {
                candidates.push(GraphEdge {
                    from: from as u32,
                    to,
                    weight: w,
                });
            }
        }

        candidates.sort_by(|a, b| {
            b.weight
                .abs()
                .total_cmp(&a.weight.abs())
                .then_with(|| a.from.cmp(&b.from))
                .then_with(|| a.to.cmp(&b.to))
        });

        let pool_len = (max_edges.saturating_mul(3)).max(1).min(candidates.len());
        let pool = &candidates[..pool_len];

        // If the graph has no candidate edges, optionally fall back to showing isolated nodes by amplitude.
        if pool.is_empty() {
            if !include_isolated {
                return GraphSnapshot {
                    kind: "substrate".to_string(),
                    nodes: Vec::new(),
                    edges: Vec::new(),
                };
            }

            let amps = brain.unit_amplitudes();
            let want = max_nodes.clamp(1, n);
            let mut ranked: Vec<(u32, f32)> = (0..n as u32)
                .map(|id| {
                    let amp = amps.get(id as usize).copied().unwrap_or(0.0);
                    (id, amp)
                })
                .collect();

            ranked.sort_by(|(id_a, a), (id_b, b)| b.total_cmp(a).then_with(|| id_a.cmp(id_b)));

            let mut nodes: Vec<GraphNode> = ranked
                .into_iter()
                .take(want)
                .map(|(id, amp)| GraphNode {
                    id,
                    label: format!("u{}", id),
                    value: amp,
                })
                .collect();
            nodes.sort_by_key(|n| n.id);

            return GraphSnapshot {
                kind: "substrate".to_string(),
                nodes,
                edges: Vec::new(),
            };
        }

        // Rank nodes by total incident strength in the pool.
        let mut node_score: HashMap<u32, f32> = HashMap::new();
        for e in pool {
            *node_score.entry(e.from).or_insert(0.0) += e.weight.abs();
            *node_score.entry(e.to).or_insert(0.0) += e.weight.abs();
        }

        let mut ranked_nodes: Vec<(u32, f32)> = node_score.into_iter().collect();
        ranked_nodes
            .sort_by(|(id_a, s_a), (id_b, s_b)| s_b.total_cmp(s_a).then_with(|| id_a.cmp(id_b)));

        let keep_n = max_nodes.clamp(1, n);
        let mut keep: HashSet<u32> = HashSet::with_capacity(keep_n);
        for (id, _) in ranked_nodes.into_iter().take(keep_n) {
            keep.insert(id);
        }

        let mut edges: Vec<GraphEdge> = pool
            .iter()
            .filter(|e| keep.contains(&e.from) && keep.contains(&e.to))
            .cloned()
            .collect();

        edges.sort_by(|a, b| {
            b.weight
                .abs()
                .total_cmp(&a.weight.abs())
                .then_with(|| a.from.cmp(&b.from))
                .then_with(|| a.to.cmp(&b.to))
        });
        edges.truncate(max_edges);

        // Pick which nodes to return.
        // - Default: only nodes incident to returned edges.
        // - Include-isolated: keep the ranked node set even if some are degree-0.
        let mut connected: HashSet<u32> = HashSet::new();
        for e in &edges {
            connected.insert(e.from);
            connected.insert(e.to);
        }

        let node_ids: HashSet<u32> = if include_isolated { keep } else { connected };

        let amps = brain.unit_amplitudes();
        let mut nodes: Vec<GraphNode> = node_ids
            .into_iter()
            .map(|id| {
                let amp = amps.get(id as usize).copied().unwrap_or(0.0);
                GraphNode {
                    id,
                    label: format!("u{}", id),
                    value: amp,
                }
            })
            .collect();
        nodes.sort_by_key(|n| n.id);

        GraphSnapshot {
            kind: "substrate".to_string(),
            nodes,
            edges,
        }
    }

    fn get_causal_graph_snapshot(
        &self,
        brain: &Brain,
        max_nodes: usize,
        max_edges: usize,
        include_isolated: bool,
    ) -> GraphSnapshot {
        use std::collections::{HashMap, VecDeque};

        // Seed around current task so the graph is useful immediately.
        let stimulus = self.current_stimulus_key().to_string();
        let pair_left = format!("pair::{}::left", stimulus);
        let pair_right = format!("pair::{}::right", stimulus);

        let mut nodes_by_label: HashMap<String, u32> = HashMap::new();
        let mut next_id: u32 = 0;

        let ensure_node =
            |label: String, nodes_by_label: &mut HashMap<String, u32>, next_id: &mut u32| -> u32 {
                if let Some(&id) = nodes_by_label.get(&label) {
                    return id;
                }
                let id = *next_id;
                *next_id = next_id.wrapping_add(1);
                nodes_by_label.insert(label, id);
                id
            };

        let mut queue: VecDeque<String> = VecDeque::new();
        for s in [
            stimulus.clone(),
            "left".to_string(),
            "right".to_string(),
            pair_left.clone(),
            pair_right.clone(),
            "reward_pos".to_string(),
            "reward_neg".to_string(),
        ] {
            let _ = ensure_node(s.clone(), &mut nodes_by_label, &mut next_id);
            queue.push_back(s);
        }

        let mut edges: Vec<GraphEdge> = Vec::new();

        // Expand graph outward using top outgoing links.
        let per_node_top = 6usize;

        while let Some(from_label) = queue.pop_front() {
            if nodes_by_label.len() >= max_nodes || edges.len() >= max_edges {
                break;
            }

            let from_id = ensure_node(from_label.clone(), &mut nodes_by_label, &mut next_id);
            let links = brain.top_causal_links_from(&from_label, per_node_top);

            for (to_label, w) in links {
                if edges.len() >= max_edges || nodes_by_label.len() >= max_nodes {
                    break;
                }

                let to_id = ensure_node(to_label.clone(), &mut nodes_by_label, &mut next_id);
                edges.push(GraphEdge {
                    from: from_id,
                    to: to_id,
                    weight: w,
                });

                if nodes_by_label.len() < max_nodes {
                    queue.push_back(to_label);
                }
            }
        }

        let mut nodes: Vec<GraphNode> = nodes_by_label
            .iter()
            .map(|(label, &id)| GraphNode {
                id,
                label: label.clone(),
                value: 0.0,
            })
            .collect();
        nodes.sort_by_key(|n| n.id);

        edges.sort_by(|a, b| b.weight.abs().total_cmp(&a.weight.abs()));
        edges.truncate(max_edges);

        if !include_isolated {
            // Only include nodes that are actually connected by the returned edges.
            // This avoids rendering large sets of isolated labels when the edge budget is small.
            if edges.is_empty() {
                return GraphSnapshot {
                    kind: "causal".to_string(),
                    nodes: Vec::new(),
                    edges,
                };
            }

            use std::collections::HashSet;
            let mut connected: HashSet<u32> = HashSet::new();
            for e in &edges {
                connected.insert(e.from);
                connected.insert(e.to);
            }

            nodes.retain(|n| connected.contains(&n.id));
        }

        GraphSnapshot {
            kind: "causal".to_string(),
            nodes,
            edges,
        }
    }

    fn save_brain(&self) -> Result<(), String> {
        let path = self.paths.brain_file();
        info!("Saving brain (braine.bbi)...");

        // Ensure directory exists
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).map_err(|e| {
                let msg = format!("Failed to create directory {:?}: {}", parent, e);
                error!("{}", msg);
                msg
            })?;
        }

        // Persist runtime/task metrics alongside the brain so UI progress doesn't reset.
        let stats = self.game.stats();
        let cur_kind = self.game.kind().to_string();
        let cur_stats = PersistedGameStats {
            correct: stats.correct,
            incorrect: stats.incorrect,
            trials: stats.trials,
            recent: stats.recent.clone(),
            learning_at_trial: stats.learning_at_trial,
            learned_at_trial: stats.learned_at_trial,
            mastered_at_trial: stats.mastered_at_trial,
        };
        let runtime = PersistedRuntime {
            game_kind: cur_kind.clone(),
            game: cur_stats.clone(),
            games: {
                // Preserve older map entries if present.
                let mut m = std::fs::read_to_string(self.paths.runtime_state_file())
                    .ok()
                    .and_then(|s| serde_json::from_str::<PersistedRuntime>(&s).ok())
                    .map(|rt| rt.games)
                    .unwrap_or_default();
                m.insert(cur_kind, cur_stats);
                m
            },
        };
        let rt_path = self.paths.runtime_state_file();
        let json = serde_json::to_vec_pretty(&runtime)
            .map_err(|e| format!("Failed to encode runtime state: {e}"))?;

        let mut file = File::create(&path).map_err(|e| {
            let msg = format!(
                "Failed to create file at {:?}: {} (errno: {})",
                path,
                e,
                e.raw_os_error().unwrap_or(-1)
            );
            error!("{}", msg);
            msg
        })?;

        let experts_state = self
            .experts
            .save_state_bytes()
            .map_err(|e| format!("Failed to serialize experts state: {e}"))?;

        state_image::save_state_to_with_version(
            &mut file,
            &self.brain,
            &experts_state,
            Some(&json),
            self.persist_state_version,
        )
        .map_err(|e| {
            let msg = format!("Failed to serialize daemon state: {}", e);
            error!("{}", msg);
            msg
        })?;

        let mut rt = File::create(&rt_path)
            .map_err(|e| format!("Failed to create runtime state file {:?}: {e}", rt_path))?;
        rt.write_all(&json)
            .map_err(|e| format!("Failed to write runtime state file {:?}: {e}", rt_path))?;

        info!("✓ Brain saved successfully (braine.bbi)");
        Ok(())
    }

    fn load_brain(&mut self) -> Result<(), String> {
        self.loaded_snapshot_stem = None;
        let path = self.paths.brain_file();
        if !path.exists() {
            return Err("Brain file not found (braine.bbi)".to_string());
        }

        let mut file = File::open(&path).map_err(|e| format!("Failed to open file: {}", e))?;
        let mut magic = [0u8; 8];
        file.read_exact(&mut magic)
            .map_err(|e| format!("Failed to read brain file header: {e}"))?;
        file.seek(SeekFrom::Start(0))
            .map_err(|e| format!("Failed to seek brain file: {e}"))?;

        if state_image::is_state_magic(&magic) {
            let loaded = state_image::load_state_from(&mut file)
                .map_err(|e| format!("Failed to load daemon state: {}", e))?;
            self.brain = loaded.brain;
            if let Some(ex_bytes) = loaded.experts_state {
                self.experts
                    .load_state_bytes(&ex_bytes)
                    .map_err(|e| format!("Failed to load experts state: {e}"))?;
            } else {
                self.experts.set_enabled(false);
            }

            // Mirror embedded runtime state back out to runtime.json for existing tooling.
            if let Some(rt_bytes) = loaded.runtime_state {
                let rt_path = self.paths.runtime_state_file();
                if let Some(parent) = rt_path.parent() {
                    let _ = std::fs::create_dir_all(parent);
                }
                if let Err(e) = std::fs::write(&rt_path, &rt_bytes) {
                    warn!(
                        "Failed to write embedded runtime state to {:?}: {e}",
                        rt_path
                    );
                }
            }

            // Track state wrapper version for subsequent saves.
            self.persist_state_version = Self::detect_state_wrapper_version(&path);
        } else {
            // Legacy: brain image only.
            self.brain = Brain::load_image_from(&mut file)
                .map_err(|e| format!("Failed to load brain: {}", e))?;
            self.experts.set_enabled(false);
            self.persist_state_version = state_image::VERSION_V1;
        }

        // Ensure required IO groups exist (for backwards compatibility with older images).
        self.brain.ensure_sensor_min_width("spot_left", 4);
        self.brain.ensure_sensor_min_width("spot_right", 4);
        self.brain.ensure_sensor_min_width("spot_rev_ctx", 2);
        self.brain.ensure_sensor_min_width("bandit", 4);
        self.ensure_spotxy_io();
        if self.game.kind() == "pong" {
            self.ensure_pong_io();
        }
        self.brain.ensure_action_min_width("left", 6);
        self.brain.ensure_action_min_width("right", 6);
        self.brain.set_observer_telemetry(true);

        // Apply the same IO/back-compat fixups to all expert brains and fork points.
        let game_kind = self.game.kind();
        let spotxy_allowed = if let ActiveGame::SpotXY(g) = &self.game {
            Some(g.allowed_actions())
        } else {
            None
        };
        self.experts.for_each_brain_mut(&mut |b: &mut Brain| {
            b.ensure_sensor_min_width("spot_left", 4);
            b.ensure_sensor_min_width("spot_right", 4);
            b.ensure_sensor_min_width("spot_rev_ctx", 2);
            b.ensure_sensor_min_width("bandit", 4);
            // SpotXY IO is derived from current daemon game.
            if game_kind == "spotxy" {
                let k = 16usize;
                for i in 0..k {
                    b.ensure_sensor_min_width(&format!("pos_x_{i:02}"), 3);
                    b.ensure_sensor_min_width(&format!("pos_y_{i:02}"), 3);
                }
                if let Some(names) = spotxy_allowed {
                    for name in names {
                        b.ensure_action_min_width(name, 6);
                    }
                }
            }
            if game_kind == "pong" {
                let bins = 8u32;
                for i in 0..bins {
                    b.ensure_sensor_min_width(&format!("pong_ball_x_{i:02}"), 3);
                    b.ensure_sensor_min_width(&format!("pong_ball_y_{i:02}"), 3);
                    b.ensure_sensor_min_width(&format!("pong_paddle_y_{i:02}"), 3);
                }
                b.ensure_sensor_min_width("pong_vx_pos", 2);
                b.ensure_sensor_min_width("pong_vx_neg", 2);
                b.ensure_sensor_min_width("pong_vy_pos", 2);
                b.ensure_sensor_min_width("pong_vy_neg", 2);
                b.ensure_action_min_width("up", 6);
                b.ensure_action_min_width("down", 6);
                b.ensure_action_min_width("stay", 6);
            }
            b.ensure_action_min_width("left", 6);
            b.ensure_action_min_width("right", 6);
            b.set_observer_telemetry(true);
        });

        // Load runtime/task metrics if present.
        let rt_path = self.paths.runtime_state_file();
        if rt_path.exists() {
            match std::fs::read_to_string(&rt_path)
                .ok()
                .and_then(|s| serde_json::from_str::<PersistedRuntime>(&s).ok())
            {
                Some(rt) => {
                    // Prefer per-game stats map when available.
                    let want_kind = self.game.kind();
                    let picked = rt.games.get(want_kind).cloned().or_else(|| {
                        // Back-compat: old single-game fields.
                        if rt.game_kind.is_empty() || rt.game_kind == want_kind {
                            Some(rt.game.clone())
                        } else {
                            None
                        }
                    });

                    if let Some(p) = picked {
                        let s = self.game.stats_mut();
                        s.correct = p.correct;
                        s.incorrect = p.incorrect;
                        s.trials = p.trials;
                        s.recent = p.recent;
                        s.learning_at_trial = p.learning_at_trial;
                        s.learned_at_trial = p.learned_at_trial;
                        s.mastered_at_trial = p.mastered_at_trial;
                    }
                }
                None => warn!("Failed to parse runtime state file {:?}", rt_path),
            }
        }

        // Align autosave baseline with loaded trial count to avoid underflow.
        self.last_autosave_trial = self.game.stats().trials;
        info!("Brain loaded (braine.bbi)");
        Ok(())
    }

    fn reset_brain(&mut self) {
        *self = Self::new(self.paths.clone());
        info!("Brain reset to initial state");
    }
}

fn parse_execution_tier(s: &str) -> Option<braine::substrate::ExecutionTier> {
    match s.trim().to_ascii_lowercase().as_str() {
        "scalar" | "cpu" => Some(braine::substrate::ExecutionTier::Scalar),
        "simd" => Some(braine::substrate::ExecutionTier::Simd),
        "parallel" => Some(braine::substrate::ExecutionTier::Parallel),
        "gpu" => Some(braine::substrate::ExecutionTier::Gpu),
        _ => None,
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Client Handler
// ═══════════════════════════════════════════════════════════════════════════

async fn handle_client(
    stream: TcpStream,
    state: Arc<RwLock<DaemonState>>,
) -> Result<(), Box<dyn std::error::Error>> {
    let (reader, mut writer) = stream.into_split();
    let mut lines = BufReader::new(reader).lines();

    while let Some(line) = lines.next_line().await? {
        let request: Request = match serde_json::from_str(&line) {
            Ok(req) => req,
            Err(e) => {
                let resp = Response::Error {
                    message: format!("Invalid request: {}", e),
                };
                writer
                    .write_all(serde_json::to_string(&resp)?.as_bytes())
                    .await?;
                writer.write_all(b"\n").await?;
                continue;
            }
        };

        let response = match request {
            Request::ApiCatalog => Response::ApiCatalog {
                categories: vec![
                    ApiCategory {
                        name: "Diagnostics".to_string(),
                        endpoints: vec![ApiEndpoint {
                            request: "DiagGet".to_string(),
                            input: "{}".to_string(),
                            output: "{ type: Diagnostics, ... }".to_string(),
                            description: "Read-only daemon/brain diagnostics snapshot.".to_string(),
                        }],
                    },
                    ApiCategory {
                        name: "Configuration".to_string(),
                        endpoints: vec![
                            ApiEndpoint {
                                request: "CfgGet".to_string(),
                                input: "{}".to_string(),
                                output: "{ type: Config, ... }".to_string(),
                                description: "Read current runtime configuration knobs.".to_string(),
                            },
                            ApiEndpoint {
                                request: "CfgSet".to_string(),
                                input: "{ exploration_eps?, meaning_alpha?, target_fps?, trial_period_ms?, max_units? }"
                                    .to_string(),
                                output: "{ type: Success|Error }".to_string(),
                                description: "Update runtime knobs (safe clamped).".to_string(),
                            },
                        ],
                    },
                    ApiCategory {
                        name: "Synchronization".to_string(),
                        endpoints: vec![
                            ApiEndpoint {
                                request: "SyncGetInfo".to_string(),
                                input: "{}".to_string(),
                                output: "{ type: SyncInfo, ... }".to_string(),
                                description:
                                    "Get sync metadata for edge/child brains (fingerprint + sizes)."
                                        .to_string(),
                            },
                            ApiEndpoint {
                                request: "SyncApplyDelta".to_string(),
                                input: "{ delta, delta_max, expected_weights_len, expected_fingerprint, autosave }"
                                    .to_string(),
                                output: "{ type: SyncApplied|Error }".to_string(),
                                description:
                                    "Apply a sparse BrainDelta (top-K edge weight deltas) from an edge/child brain and optionally persist immediately."
                                        .to_string(),
                            },
                        ],
                    },
                    ApiCategory {
                        name: "Inference".to_string(),
                        endpoints: vec![ApiEndpoint {
                            request: "InferActionScores".to_string(),
                            input: "{ context_key?, stimuli?, steps?, meaning_alpha? }".to_string(),
                            output: "{ type: InferActionScores, context_key, action_scores: [...] }".to_string(),
                            description: "Read-only inference on a cloned brain: apply stimuli (no imprint), advance dynamics (no learning/forget), return action score breakdowns.".to_string(),
                        }],
                    },
                    ApiCategory {
                        name: "Advisor".to_string(),
                        endpoints: vec![
                            ApiEndpoint {
                                request: "AdvisorGet".to_string(),
                                input: "{}".to_string(),
                                output: "{ type: AdvisorStatus, config, last_report? }".to_string(),
                                description: "Get advisor runtime config + last report (if any).".to_string(),
                            },
                            ApiEndpoint {
                                request: "AdvisorSet".to_string(),
                                input: "{ enabled, every_trials?, mode? }".to_string(),
                                output: "{ type: Success|Error }".to_string(),
                                description: "Update advisor runtime knobs.".to_string(),
                            },
                            ApiEndpoint {
                                request: "AdvisorOnce".to_string(),
                                input: "{ apply }".to_string(),
                                output: "{ type: AdvisorReport, report, applied }".to_string(),
                                description: "Invoke built-in advisor once (stub) and optionally apply.".to_string(),
                            },
                            ApiEndpoint {
                                request: "AdvisorContext".to_string(),
                                input: "{ include_action_scores }".to_string(),
                                output: "{ type: AdvisorContext, context, action_scores: [...] }".to_string(),
                                description: "LLM boundary: get structured context for an external advisor.".to_string(),
                            },
                            ApiEndpoint {
                                request: "AdvisorApply".to_string(),
                                input: "{ advice: { exploration_eps?, meaning_alpha?, ttl_trials, rationale } }".to_string(),
                                output: "{ type: Success|Error }".to_string(),
                                description: "LLM boundary: apply external advisor advice (daemon clamps + records).".to_string(),
                            },
                        ],
                    },
                    ApiCategory {
                        name: "Replay".to_string(),
                        endpoints: vec![
                            ApiEndpoint {
                                request: "ReplayGetDataset".to_string(),
                                input: "{}".to_string(),
                                output: "{ type: ReplayDataset, dataset: { name, trials: [...] } }".to_string(),
                                description: "Get current replay dataset (for dataset-driven evaluation).".to_string(),
                            },
                            ApiEndpoint {
                                request: "ReplaySetDataset".to_string(),
                                input: "{ dataset: { name, trials: [...] } }".to_string(),
                                output: "{ type: Success|Error }".to_string(),
                                description: "Set replay dataset (must be stopped); resizes sensors/actions as needed.".to_string(),
                            },
                        ],
                    },
                    ApiCategory {
                        name: "General".to_string(),
                        endpoints: vec![ApiEndpoint {
                            request: "GetState".to_string(),
                            input: "{}".to_string(),
                            output: "{ type: State, ... }".to_string(),
                            description: "Full UI snapshot (stats, HUD, plots, experts, storage)."
                                .to_string(),
                        }],
                    },
                ],
            },
            Request::DiagGet => {
                let s = state.read().await;
                let snap = s.get_snapshot();
                Response::Diagnostics {
                    running: snap.running,
                    frame: snap.frame,
                    brain_stats: snap.brain_stats,
                    storage: snap.storage,
                }
            }
            Request::CfgGet => {
                let s = state.read().await;
                Response::Config {
                    exploration_eps: s.exploration_eps,
                    meaning_alpha: s.meaning_alpha,
                    target_fps: s.target_fps,
                    trial_period_ms: s.trial_period_ms,
                    max_units_limit: s.max_units_limit as u32,
                }
            }
            Request::CfgSet {
                exploration_eps,
                meaning_alpha,
                target_fps,
                trial_period_ms,
                max_units,
            } => {
                let mut s = state.write().await;

                if let Some(v) = exploration_eps {
                    s.exploration_eps = v.clamp(0.0, 1.0);
                }
                if let Some(v) = meaning_alpha {
                    s.meaning_alpha = v.clamp(0.0, 50.0);
                }
                if let Some(v) = target_fps {
                    s.target_fps = v.clamp(1, 240);
                }
                if let Some(v) = trial_period_ms {
                    let ms = v.clamp(10, 10_000);
                    s.trial_period_ms = ms;
                    s.game.update_timing(ms);
                }
                if let Some(max_units) = max_units {
                    let requested = max_units as usize;
                    let current_units = s.brain.diagnostics().unit_count;
                    s.max_units_limit = requested.clamp(current_units, 4096);
                }

                Response::Success {
                    message: "Config updated".to_string(),
                }
            }
            Request::SyncGetInfo => {
                let s = state.read().await;
                Response::SyncInfo {
                    age_steps: s.brain.age_steps(),
                    unit_count: s.brain.diagnostics().unit_count as u32,
                    weights_len: s.brain.weights_len() as u32,
                    fingerprint: s.brain.connections_fingerprint(),
                }
            }
            Request::SyncApplyDelta {
                delta,
                delta_max,
                expected_weights_len,
                expected_fingerprint,
                autosave,
            } => {
                let mut s = state.write().await;

                let weights_len = s.brain.weights_len() as u32;
                if weights_len != expected_weights_len {
                    Response::Error {
                        message: format!(
                            "Sync rejected: weights_len mismatch (local={weights_len}, expected={expected_weights_len})"
                        ),
                    }
                } else {
                    let fp = s.brain.connections_fingerprint();
                    if fp != expected_fingerprint {
                        Response::Error {
                            message: "Sync rejected: fingerprint mismatch (topology differs)".to_string(),
                        }
                    } else {
                        let applied_edges = delta.weight_deltas.len() as u32;
                        let dm = delta_max.clamp(1.0e-6, 0.5);
                        s.brain.apply_weight_delta(&delta, dm);

                        let mut saved = false;
                        let mut save_error: Option<String> = None;
                        if autosave {
                            match s.save_brain() {
                                Ok(()) => saved = true,
                                Err(e) => {
                                    save_error = Some(e);
                                }
                            }
                        }

                        Response::SyncApplied {
                            applied_edges,
                            saved,
                            save_error,
                        }
                    }
                }
            }
            Request::InferActionScores {
                context_key,
                stimuli,
                steps,
                meaning_alpha,
            } => {
                let s = state.read().await;
                let context_key = context_key.unwrap_or_else(|| s.current_stimulus_key().into_owned());
                let alpha = meaning_alpha.unwrap_or(s.meaning_alpha).clamp(0.0, 50.0);

                let mut brain = s.view_brain_for_context(&context_key).clone();
                for stim in &stimuli {
                    brain.apply_stimulus_inference(stim.as_stimulus());
                }
                let steps = steps.clamp(1, 64);
                for _ in 0..steps {
                    brain.step_inference();
                }

                let action_scores = brain.action_score_breakdown(&context_key, alpha);

                Response::InferActionScores {
                    context_key,
                    action_scores,
                }
            }
            Request::GetState => {
                let s = state.read().await;
                Response::State(Box::new(s.get_snapshot()))
            }

            Request::AdvisorGet => {
                let s = state.read().await;
                Response::AdvisorStatus {
                    config: s.advisor.status(),
                    last_report: s.advisor.last_report.clone(),
                }
            }

            Request::AdvisorSet {
                enabled,
                every_trials,
                mode,
            } => {
                let mut s = state.write().await;
                s.advisor.set_enabled(enabled);
                if let Some(n) = every_trials {
                    s.advisor.set_every_trials(n);
                }
                if let Some(m) = mode {
                    s.advisor.set_mode(m);
                }
                Response::Success {
                    message: "Advisor config updated".to_string(),
                }
            }

            Request::AdvisorOnce { apply } => {
                let mut s = state.write().await;
                let trials = s.game.stats().trials;
                let context_key = s.current_stimulus_key();
                let text_regime = match &s.game {
                    ActiveGame::Text(g) => Some(g.regime()),
                    _ => None,
                };

                let ctx = advisor::AdvisorContext {
                    game: s.game.kind().to_string(),
                    context_key: context_key.into_owned(),
                    trials,
                    accuracy: s.game.stats().accuracy(),
                    recent_rate: s.game.stats().recent_rate(),
                    last_reward: s.last_reward,
                    exploration_eps: s.exploration_eps,
                    meaning_alpha: s.meaning_alpha,
                    text_regime,
                };

                let report = s.advisor.invoke(ctx, trials, apply);
                if apply {
                    s.apply_advice(&report.advice);
                }
                Response::AdvisorReport {
                    report,
                    applied: apply,
                }
            }

            Request::AdvisorContext {
                include_action_scores,
            } => {
                let s = state.read().await;
                let trials = s.game.stats().trials;
                let context_key = s.current_stimulus_key();
                let text_regime = match &s.game {
                    ActiveGame::Text(g) => Some(g.regime()),
                    _ => None,
                };

                let ctx = advisor::AdvisorContext {
                    game: s.game.kind().to_string(),
                    context_key: context_key.clone().into_owned(),
                    trials,
                    accuracy: s.game.stats().accuracy(),
                    recent_rate: s.game.stats().recent_rate(),
                    last_reward: s.last_reward,
                    exploration_eps: s.exploration_eps,
                    meaning_alpha: s.meaning_alpha,
                    text_regime,
                };

                let action_scores = if include_action_scores {
                    let brain = s.view_brain_for_context(&context_key);
                    brain.action_score_breakdown(&context_key, s.meaning_alpha)
                } else {
                    Vec::new()
                };

                Response::AdvisorContext {
                    context: ctx,
                    action_scores,
                }
            }

            Request::AdvisorApply { advice } => {
                let mut s = state.write().await;
                let trials = s.game.stats().trials;
                let context_key = s.current_stimulus_key();
                let text_regime = match &s.game {
                    ActiveGame::Text(g) => Some(g.regime()),
                    _ => None,
                };

                let ctx = advisor::AdvisorContext {
                    game: s.game.kind().to_string(),
                    context_key: context_key.into_owned(),
                    trials,
                    accuracy: s.game.stats().accuracy(),
                    recent_rate: s.game.stats().recent_rate(),
                    last_reward: s.last_reward,
                    exploration_eps: s.exploration_eps,
                    meaning_alpha: s.meaning_alpha,
                    text_regime,
                };

                // Clamp + apply. This is the explicit LLM boundary.
                s.apply_advice(&advice);

                // Record for visibility, even when advice did not originate from the built-in stub.
                s.advisor.last_report = Some(advisor::AdvisorReport {
                    at_trials: trials,
                    applied: true,
                    context: ctx,
                    advice,
                });

                Response::Success {
                    message: "Advisor advice applied".to_string(),
                }
            }

            Request::ReplayGetDataset => {
                let s = state.read().await;
                Response::ReplayDataset {
                    dataset: s.replay_dataset.clone(),
                }
            }

            Request::ReplaySetDataset { dataset } => {
                let mut s = state.write().await;
                if s.running {
                    Response::Error {
                        message: "Stop the simulation before setting replay dataset".to_string(),
                    }
                } else {
                    s.replay_dataset = dataset;

                    // Keep I/O sizes in sync, and reset replay game if currently active.
                    s.ensure_replay_io();
                    if matches!(s.game, ActiveGame::Replay(_)) {
                        let gg = ReplayGame::new(s.replay_dataset.clone());
                        s.game = ActiveGame::Replay(gg);
                        s.pending_neuromod = 0.0;
                        s.last_reward = 0.0;
                    }

                    Response::Success {
                        message: "Replay dataset updated".to_string(),
                    }
                }
            }
            Request::SetView { view } => {
                let mut s = state.write().await;
                match BrainViewMode::parse(&view) {
                    Some(m) => {
                        s.view_mode = m;
                        Response::Success {
                            message: format!("View set to {}", view.trim()),
                        }
                    }
                    None => Response::Error {
                        message: format!(
                            "Unknown view '{}'. Use parent|active_expert",
                            view.trim()
                        ),
                    },
                }
            }

            Request::SetMaxUnits { max_units } => {
                let mut s = state.write().await;
                let requested = max_units as usize;
                // Clamp to a sane range; cannot be set below current parent unit count.
                let current_units = s.brain.diagnostics().unit_count;
                let clamped = requested.clamp(current_units, 4096);
                s.max_units_limit = clamped;
                Response::Success {
                    message: format!("Max units limit set to {}", clamped),
                }
            }

            Request::SaveSnapshot => {
                let s = state.read().await;
                match s.save_snapshot() {
                    Ok(stem) => Response::Success {
                        message: format!("Snapshot saved ({stem})"),
                    },
                    Err(e) => Response::Error { message: e },
                }
            }

            Request::LoadSnapshot { stem } => {
                let mut s = state.write().await;
                match s.load_snapshot(stem.trim()) {
                    Ok(_) => Response::Success {
                        message: format!("Snapshot loaded ({})", stem.trim()),
                    },
                    Err(e) => Response::Error { message: e },
                }
            }
            Request::GetGraph {
                kind,
                max_nodes,
                max_edges,
                include_isolated,
            } => {
                let s = state.read().await;
                let max_nodes = (max_nodes as usize).clamp(1, 256);
                let max_edges = (max_edges as usize).clamp(0, 1024);
                Response::Graph(Box::new(s.get_graph_snapshot(
                    &kind,
                    max_nodes,
                    max_edges,
                    include_isolated,
                )))
            }
            Request::Start => {
                let mut s = state.write().await;
                s.running = true;
                Response::Success {
                    message: "Started".to_string(),
                }
            }
            Request::Stop => {
                let mut s = state.write().await;
                s.running = false;
                // Persist on explicit stop to keep online-learned memory
                match s.save_brain() {
                    Ok(_) => Response::Success {
                        message: "Stopped and saved".to_string(),
                    },
                    Err(e) => Response::Error {
                        message: format!("Stopped but save failed: {}", e),
                    },
                }
            }
            Request::GetGameParams { game } => {
                let game = game.trim();

                // The daemon is the source of truth for knob definitions.
                match game {
                    "pong" => {
                        // Keep these in sync with `PongGame::set_param` clamping.
                        const PONG_PADDLE_SPEED_MIN: f32 = 0.1;
                        const PONG_PADDLE_SPEED_MAX: f32 = 5.0;

                        const PONG_BALL_SPEED_MIN: f32 = 0.1;
                        const PONG_BALL_SPEED_MAX: f32 = 3.0;

                        const PONG_PADDLE_HALF_HEIGHT_MIN: f32 = 0.05;
                        const PONG_PADDLE_HALF_HEIGHT_MAX: f32 = 0.9;

                        let defaults = braine_games::pong::PongParams::default();

                        Response::GameParams {
                            game: "pong".to_string(),
                            params: vec![
                                GameParamDef {
                                    key: "paddle_speed".to_string(),
                                    label: "Paddle speed".to_string(),
                                    description: "Paddle movement speed (units per second)."
                                        .to_string(),
                                    min: PONG_PADDLE_SPEED_MIN,
                                    max: PONG_PADDLE_SPEED_MAX,
                                    default: defaults.paddle_speed,
                                },
                                GameParamDef {
                                    key: "ball_speed".to_string(),
                                    label: "Ball speed".to_string(),
                                    description: "Ball movement speed multiplier.".to_string(),
                                    min: PONG_BALL_SPEED_MIN,
                                    max: PONG_BALL_SPEED_MAX,
                                    default: defaults.ball_speed,
                                },
                                GameParamDef {
                                    key: "paddle_half_height".to_string(),
                                    label: "Paddle height".to_string(),
                                    description:
                                        "Half-height of paddle as a fraction of playfield height."
                                            .to_string(),
                                    min: PONG_PADDLE_HALF_HEIGHT_MIN,
                                    max: PONG_PADDLE_HALF_HEIGHT_MAX,
                                    default: defaults.paddle_half_height,
                                },
                            ],
                        }
                    }
                    "spotxy" => {
                        // SpotXY grid range: 0 (binary mode) or 2..=8 grid.
                        Response::GameParams {
                            game: "spotxy".to_string(),
                            params: vec![
                                GameParamDef {
                                    key: "grid_n".to_string(),
                                    label: "Grid size".to_string(),
                                    description:
                                        "Grid dimension (0 = binary left/right; 2-8 = NxN grid)."
                                            .to_string(),
                                    min: 0.0,
                                    max: 8.0,
                                    default: 0.0,
                                },
                                GameParamDef {
                                    key: "eval".to_string(),
                                    label: "Eval mode".to_string(),
                                    description: "Holdout/evaluation mode (0=train, 1=eval)."
                                        .to_string(),
                                    min: 0.0,
                                    max: 1.0,
                                    default: 0.0,
                                },
                            ],
                        }
                    }
                    _ => Response::GameParams {
                        game: game.to_string(),
                        params: Vec::new(),
                    },
                }
            }
            Request::SetGame { game } => {
                let mut s = state.write().await;
                if s.running {
                    Response::Error {
                        message: "Stop the simulation before switching game".to_string(),
                    }
                } else {
                    match s.set_game(&game) {
                        Ok(_) => Response::Success {
                            message: format!("Game set to {}", game),
                        },
                        Err(e) => Response::Error { message: e },
                    }
                }
            }
            Request::SetGameParam { game, key, value } => {
                let mut s = state.write().await;
                let game = game.trim();
                let key = key.trim();

                if s.game.kind() != game {
                    Response::Error {
                        message: format!(
                            "Game param applies to active game only (active={}, requested={})",
                            s.game.kind(),
                            game
                        ),
                    }
                } else {
                    match &mut s.game {
                        ActiveGame::Pong(g) => match g.set_param(key, value) {
                            Ok(_) => Response::Success {
                                message: format!("Set {game}.{key} = {value}"),
                            },
                            Err(e) => Response::Error { message: e },
                        },
                        ActiveGame::SpotXY(g) => {
                            // SpotXY tunable params: grid_n, eval.
                            match key {
                                "grid_n" => {
                                    let n = value.round().clamp(0.0, 8.0) as u32;
                                    // Adapt existing grid state to target size.
                                    while g.grid_n() < n {
                                        g.increase_grid();
                                    }
                                    while g.grid_n() > n {
                                        g.decrease_grid();
                                    }
                                    s.ensure_spotxy_io();
                                    s.pending_neuromod = 0.0;
                                    s.last_reward = 0.0;
                                    Response::Success {
                                        message: format!("Set {game}.{key} = {n}"),
                                    }
                                }
                                "eval" => {
                                    let eval = value >= 0.5;
                                    g.set_eval_mode(eval);
                                    s.pending_neuromod = 0.0;
                                    s.last_reward = 0.0;
                                    Response::Success {
                                        message: format!("Set {game}.{key} = {eval}"),
                                    }
                                }
                                _ => Response::Error {
                                    message: format!(
                                        "Unknown SpotXY param '{key}'. Use grid_n | eval"
                                    ),
                                },
                            }
                        }
                        _ => Response::Error {
                            message: format!("No tunable params implemented for game '{game}'"),
                        },
                    }
                }
            }
            Request::SetSpotXYEval { eval } => {
                let mut s = state.write().await;
                match &mut s.game {
                    ActiveGame::SpotXY(g) => {
                        g.set_eval_mode(eval);
                        s.pending_neuromod = 0.0;
                        s.last_reward = 0.0;
                        Response::Success {
                            message: format!("SpotXY eval mode set to {}", eval),
                        }
                    }
                    _ => Response::Error {
                        message: "SpotXY eval mode is only available in the spotxy game"
                            .to_string(),
                    },
                }
            }
            Request::SpotXYIncreaseGrid => {
                let mut s = state.write().await;
                let mode_and_grid = match &mut s.game {
                    ActiveGame::SpotXY(g) => {
                        g.increase_grid();
                        Some((g.mode_name().to_string(), g.grid_n()))
                    }
                    _ => None,
                };

                if let Some((mode, grid_n)) = mode_and_grid {
                    s.ensure_spotxy_io();
                    s.pending_neuromod = 0.0;
                    s.last_reward = 0.0;
                    Response::Success {
                        message: format!(
                            "SpotXY grid increased (mode={}, grid_n={})",
                            mode, grid_n
                        ),
                    }
                } else {
                    Response::Error {
                        message: "SpotXY grid increase is only available in the spotxy game"
                            .to_string(),
                    }
                }
            }
            Request::SpotXYDecreaseGrid => {
                let mut s = state.write().await;
                let mode_and_grid = match &mut s.game {
                    ActiveGame::SpotXY(g) => {
                        g.decrease_grid();
                        Some((g.mode_name().to_string(), g.grid_n()))
                    }
                    _ => None,
                };

                if let Some((mode, grid_n)) = mode_and_grid {
                    s.ensure_spotxy_io();
                    s.pending_neuromod = 0.0;
                    s.last_reward = 0.0;
                    Response::Success {
                        message: format!(
                            "SpotXY grid decreased (mode={}, grid_n={})",
                            mode, grid_n
                        ),
                    }
                } else {
                    Response::Error {
                        message: "SpotXY grid decrease is only available in the spotxy game"
                            .to_string(),
                    }
                }
            }
            Request::SetMode { .. } => {
                // Spot is Braine-only; treat mode switching as a no-op.
                // The UI may emit a SetMode on startup; returning Success avoids noisy errors.
                Response::Success {
                    message: "Spot is Braine-only; mode unchanged".to_string(),
                }
            }
            Request::HumanAction { .. } => {
                // Not supported in Spot game; ignore (no-op) to avoid UI log spam.
                Response::Success {
                    message: "Spot ignores human actions".to_string(),
                }
            }
            Request::TriggerDream => {
                let mut s = state.write().await;
                s.brain.dream_replay(5, 1.5);
                Response::Success {
                    message: "Dream triggered".to_string(),
                }
            }
            Request::TriggerBurst => {
                let mut s = state.write().await;
                s.brain.set_burst_mode(true, 2.5);
                Response::Success {
                    message: "Burst mode activated".to_string(),
                }
            }
            Request::TriggerSync => {
                let mut s = state.write().await;
                s.brain.force_synchronize_sensors();
                Response::Success {
                    message: "Sensors synchronized".to_string(),
                }
            }
            Request::TriggerImprint => {
                let mut s = state.write().await;
                s.brain.imprint_current_context(0.6);
                Response::Success {
                    message: "Context imprinted".to_string(),
                }
            }
            Request::SaveBrain => {
                let s = state.read().await;
                match s.save_brain() {
                    Ok(_) => Response::Success {
                        message: "Brain saved".to_string(),
                    },
                    Err(e) => Response::Error { message: e },
                }
            }
            Request::LoadBrain => {
                let mut s = state.write().await;
                match s.load_brain() {
                    Ok(_) => Response::Success {
                        message: "Brain loaded".to_string(),
                    },
                    Err(e) => Response::Error { message: e },
                }
            }
            Request::ResetBrain => {
                let mut s = state.write().await;
                s.reset_brain();
                Response::Success {
                    message: "Brain reset".to_string(),
                }
            }
            Request::MigrateStateFormat {
                target_state_version,
            } => {
                let mut s = state.write().await;
                match s.migrate_state_format(target_state_version) {
                    Ok(_) => Response::Success {
                        message: format!(
                            "Storage migrated to daemon state wrapper version {}",
                            target_state_version
                        ),
                    },
                    Err(e) => Response::Error { message: e },
                }
            }
            Request::Shutdown => {
                let s = state.read().await;
                match s.save_brain() {
                    Ok(_) => {
                        info!("Shutdown requested; brain saved");
                        tokio::spawn(async {
                            // Give the response a moment to flush before exiting.
                            time::sleep(Duration::from_millis(50)).await;
                            std::process::exit(0);
                        });
                        Response::Success {
                            message: "Shutting down".to_string(),
                        }
                    }
                    Err(e) => Response::Error {
                        message: format!("Save failed, aborting shutdown: {}", e),
                    },
                }
            }
            Request::SetFramerate { fps } => {
                let mut s = state.write().await;
                let clamped = fps.clamp(1, 1000);
                s.target_fps = clamped;
                info!("Framerate set to {} FPS", clamped);
                Response::Success {
                    message: format!("Framerate set to {} FPS", clamped),
                }
            }
            Request::SetTrialPeriodMs { ms } => {
                let mut s = state.write().await;
                let clamped = ms.clamp(10, 60_000);
                s.trial_period_ms = clamped;
                info!("Trial period set to {} ms", clamped);
                Response::Success {
                    message: format!("Trial period set to {} ms", clamped),
                }
            }

            Request::SetExecutionTier { tier } => {
                match parse_execution_tier(&tier) {
                    None => Response::Error {
                        message: format!(
                            "Invalid execution tier: {} (expected scalar|simd|parallel|gpu)",
                            tier
                        ),
                    },
                    Some(t) => {
                        let mut s = state.write().await;
                        s.brain.set_execution_tier(t);
                        let eff = s.brain.effective_execution_tier();
                        info!("Execution tier set to {:?} (effective {:?})", t, eff);
                        Response::Success {
                            message: format!(
                                "Execution tier set to {:?} (effective {:?})",
                                t, eff
                            ),
                        }
                    }
                }
            }

            Request::SetExpertsEnabled { enabled } => {
                let mut s = state.write().await;
                s.experts.set_enabled(enabled);
                Response::Success {
                    message: format!("Experts enabled = {}", enabled),
                }
            }
            Request::SetExpertNesting {
                allow_nested,
                max_depth,
            } => {
                let mut s = state.write().await;

                let mut p = s.experts.policy().clone();
                p.allow_nested = allow_nested;
                p.max_depth = max_depth.max(1);
                s.experts.set_policy(p);

                Response::Success {
                    message: format!(
                        "Expert nesting updated (allow_nested={}, max_depth={})",
                        allow_nested,
                        max_depth.max(1)
                    ),
                }
            }
            Request::SetExpertPolicy {
                parent_learning,
                max_children,
                child_reward_scale,
                episode_trials,
                consolidate_topk,
                reward_shift_ema_delta_threshold,
                performance_collapse_drop_threshold,
                performance_collapse_baseline_min,
                allow_nested,
                max_depth,
                persistence_mode,
            } => {
                let mut s = state.write().await;

                let pm = ExpertsPersistenceMode::parse(&persistence_mode)
                    .unwrap_or(ExpertsPersistenceMode::Full);
                s.experts.set_persistence_mode(pm);

                let parent_learning =
                    ParentLearningPolicy::parse(&parent_learning).ok_or_else(|| Response::Error {
                        message: "parent_learning must be one of: normal|reduced|holdout"
                            .to_string(),
                    });

                match parent_learning {
                    Ok(parent_learning) => {
                        let mut p = s.experts.policy().clone();
                        p.parent_learning = parent_learning;
                        p.max_children = (max_children as usize).clamp(0, 8);
                        p.child_reward_scale = child_reward_scale.clamp(0.0, 4.0);
                        p.episode_trials = episode_trials.clamp(1, 10_000);
                        p.consolidate_topk = (consolidate_topk as usize).clamp(0, 10_000);
                        p.reward_shift_ema_delta_threshold =
                            reward_shift_ema_delta_threshold.clamp(0.0, 5.0);
                        p.performance_collapse_drop_threshold =
                            performance_collapse_drop_threshold.clamp(0.0, 5.0);
                        p.performance_collapse_baseline_min =
                            performance_collapse_baseline_min.clamp(-1.0, 1.0);
                        p.allow_nested = allow_nested;
                        p.max_depth = max_depth.max(1);
                        s.experts.set_policy(p);

                        Response::Success {
                            message: format!(
                                "Expert policy set (parent_learning={}, max_children={}, child_reward_scale={:.2}, episode_trials={}, consolidate_topk={}, persist={}, allow_nested={}, max_depth={})",
                                parent_learning.as_str(),
                                max_children,
                                child_reward_scale,
                                episode_trials,
                                consolidate_topk,
                                pm.as_str(),
                                allow_nested,
                                max_depth.max(1)
                            ),
                        }
                    }
                    Err(resp) => resp,
                }
            }
            Request::CullExperts => {
                let mut s = state.write().await;
                s.experts.cull_all_recursive();
                Response::Success {
                    message: "Experts culled".to_string(),
                }
            }
        };

        writer
            .write_all(serde_json::to_string(&response)?.as_bytes())
            .await?;
        writer.write_all(b"\n").await?;
    }

    Ok(())
}

// ═══════════════════════════════════════════════════════════════════════════
// Main
// ═══════════════════════════════════════════════════════════════════════════

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize logging
    tracing_subscriber::fmt::init();

    // Setup application paths
    let paths = AppPaths::new()?;
    info!("Persistence initialized (OS data dir; braine.bbi)");

    // Initialize daemon state
    let state = Arc::new(RwLock::new(DaemonState::new(paths)));

    // Save on Ctrl-C so state persists even if the daemon is stopped abruptly.
    {
        let state = Arc::clone(&state);
        tokio::spawn(async move {
            if tokio::signal::ctrl_c().await.is_ok() {
                let s = state.read().await;
                if let Err(e) = s.save_brain() {
                    error!("Ctrl-C save failed: {}", e);
                } else {
                    info!("Ctrl-C: brain saved");
                }
                std::process::exit(0);
            }
        });
    }

    // Try to load existing brain
    {
        let mut s = state.write().await;
        if let Err(e) = s.load_brain() {
            warn!("Could not load brain: {}", e);
            info!("Starting with fresh brain");
        }
    }

    // Start IPC server
    let listener = TcpListener::bind("127.0.0.1:9876").await?;
    info!("Braine daemon listening on 127.0.0.1:9876");

    // Game loop task
    let state_clone = Arc::clone(&state);
    tokio::spawn(async move {
        loop {
            // Read target FPS and calculate delay
            let target_fps = {
                let s = state_clone.read().await;
                s.target_fps
            };
            let frame_millis = (1000 / target_fps).max(1) as u64;

            // Sleep for the calculated duration
            tokio::time::sleep(tokio::time::Duration::from_millis(frame_millis)).await;

            // Execute game tick
            let mut s = state_clone.write().await;
            s.tick();
        }
    });

    // Accept client connections
    loop {
        let (stream, addr) = listener.accept().await?;
        info!("Client connected: {}", addr);
        let state_clone = Arc::clone(&state);

        tokio::spawn(async move {
            if let Err(e) = handle_client(stream, state_clone).await {
                error!("Client handler error: {}", e);
            }
        });
    }
}
