//! CLI client for the `brained` daemon.
//!
//! Examples:
//!   braine-cli status
//!   braine-cli start
//!   braine-cli stop
//!   braine-cli mode human
//!   braine-cli trigger dream
//!   braine-cli action left
//!   braine-cli save
//!
//! By default it talks to 127.0.0.1:9876; override with `--addr host:port`.

use serde::{Deserialize, Serialize};
use std::io::{BufRead, BufReader, Write};
use std::net::TcpStream;
use std::process;
use std::time::Duration;

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
enum Request {
    GetState,
    GetGameParams {
        game: String,
    },
    Start,
    Stop,
    SetGame {
        game: String,
    },
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
    Shutdown,
    SetFramerate {
        fps: u32,
    },
    SetTrialPeriodMs {
        ms: u32,
    },

    SetExecutionTier {
        tier: String,
    },

    // Experts (child brains)
    SetExpertsEnabled {
        enabled: bool,
    },
    SetExpertPolicy {
        parent_learning: String,
        max_children: u32,
        child_reward_scale: f32,
        episode_trials: u32,
        consolidate_topk: u32,

        #[serde(default)]
        reward_shift_ema_delta_threshold: f32,
        #[serde(default)]
        performance_collapse_drop_threshold: f32,
        #[serde(default)]
        performance_collapse_baseline_min: f32,

        #[serde(default)]
        allow_nested: bool,
        #[serde(default)]
        max_depth: u32,
        #[serde(default)]
        persistence_mode: String,
    },
    CullExperts,

    // Advisor / LLM integration
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

    // Explicit LLM boundary
    AdvisorContext {
        #[serde(default)]
        include_action_scores: bool,
    },
    AdvisorApply {
        advice: AdvisorAdvice,
    },

    // Replay dataset (dataset-driven evaluation)
    ReplayGetDataset,
    ReplaySetDataset {
        dataset: ReplayDataset,
    },

    // Newer daemon knob surface (optional)
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
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
enum Response {
    State(Box<StateSnapshot>),
    GameParams {
        game: String,
        params: Vec<GameParamDef>,
    },
    Success {
        message: String,
    },
    Error {
        message: String,
    },

    // Advisor / LLM integration
    AdvisorStatus {
        config: AdvisorConfig,
        #[serde(default)]
        last_report: Option<AdvisorReport>,
    },
    AdvisorReport {
        report: AdvisorReport,
        #[serde(default)]
        applied: bool,
    },

    AdvisorContext {
        context: AdvisorContext,
        #[serde(default)]
        action_scores: Vec<serde_json::Value>,
    },

    ReplayDataset {
        dataset: ReplayDataset,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
struct ReplayStimulus {
    #[serde(default)]
    name: String,
    #[serde(default = "default_replay_strength")]
    strength: f32,
}

fn default_replay_strength() -> f32 {
    1.0
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
struct ReplayTrial {
    #[serde(default)]
    stimuli: Vec<ReplayStimulus>,
    #[serde(default)]
    allowed_actions: Vec<String>,
    #[serde(default)]
    correct_action: String,
    #[serde(default)]
    id: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
struct ReplayDataset {
    #[serde(default)]
    name: String,
    #[serde(default)]
    trials: Vec<ReplayTrial>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
struct AdvisorConfig {
    #[serde(default)]
    enabled: bool,
    #[serde(default)]
    every_trials: u32,
    #[serde(default)]
    mode: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
struct AdvisorContext {
    #[serde(default)]
    game: String,
    #[serde(default)]
    context_key: String,
    #[serde(default)]
    trials: u32,
    #[serde(default)]
    accuracy: f32,
    #[serde(default)]
    recent_rate: f32,
    #[serde(default)]
    last_reward: f32,
    #[serde(default)]
    exploration_eps: f32,
    #[serde(default)]
    meaning_alpha: f32,
    #[serde(default)]
    text_regime: Option<u32>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
struct AdvisorAdvice {
    #[serde(default)]
    ttl_trials: u32,
    #[serde(default)]
    exploration_eps: Option<f32>,
    #[serde(default)]
    meaning_alpha: Option<f32>,
    #[serde(default)]
    rationale: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
struct AdvisorReport {
    #[serde(default)]
    at_trials: u32,
    #[serde(default)]
    applied: bool,
    #[serde(default)]
    context: AdvisorContext,
    #[serde(default)]
    advice: AdvisorAdvice,
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

#[derive(Debug, Clone, Serialize, Deserialize)]
struct StateSnapshot {
    running: bool,
    mode: String,
    frame: u64,
    #[serde(default)]
    target_fps: u32,
    game: GameState,
    hud: HudData,
    brain_stats: BrainStats,

    // Experts (child brains)
    #[serde(default)]
    experts_enabled: bool,
    #[serde(default)]
    experts: ExpertsSummary,
    #[serde(default)]
    active_expert: Option<ActiveExpertSummary>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
struct ExpertsSummary {
    #[serde(default)]
    active_count: u32,
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
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct ActiveExpertSummary {
    id: u32,
    #[serde(default)]
    context_key: String,
    #[serde(default)]
    age_steps: u64,
    #[serde(default)]
    reward_ema: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
struct GameCommon {
    #[serde(default)]
    reversal_active: bool,
    #[serde(default)]
    response_made: bool,
    #[serde(default)]
    trial_frame: u32,
    #[serde(default)]
    trial_duration: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
#[serde(tag = "kind")]
enum GameState {
    #[serde(rename = "spot")]
    Spot {
        #[serde(flatten)]
        common: GameCommon,
        #[serde(default)]
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
        #[serde(default)]
        spot_is_left: bool,
    },
    #[serde(rename = "spotxy")]
    SpotXY {
        #[serde(flatten)]
        common: GameCommon,
    },
    #[serde(rename = "maze")]
    Maze {
        #[serde(flatten)]
        common: GameCommon,
        #[serde(default)]
        maze_mode: String,
        #[serde(default)]
        maze_w: u32,
        #[serde(default)]
        maze_h: u32,
        #[serde(default)]
        maze_player_x: u32,
        #[serde(default)]
        maze_player_y: u32,
        #[serde(default)]
        maze_goal_x: u32,
        #[serde(default)]
        maze_goal_y: u32,
        #[serde(default)]
        maze_steps: u32,
        #[serde(default)]
        maze_event: String,
    },
    #[serde(rename = "pong")]
    Pong {
        #[serde(flatten)]
        common: GameCommon,
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
    #[serde(other)]
    #[default]
    Unknown,
}

impl GameState {
    fn kind(&self) -> &'static str {
        match self {
            Self::Spot { .. } => "spot",
            Self::Bandit { .. } => "bandit",
            Self::SpotReversal { .. } => "spot_reversal",
            Self::SpotXY { .. } => "spotxy",
            Self::Maze { .. } => "maze",
            Self::Pong { .. } => "pong",
            Self::Text { .. } => "text",
            Self::Unknown => "unknown",
        }
    }

    fn common(&self) -> Option<&GameCommon> {
        match self {
            Self::Spot { common, .. }
            | Self::Bandit { common }
            | Self::SpotReversal { common, .. }
            | Self::SpotXY { common }
            | Self::Maze { common, .. }
            | Self::Pong { common }
            | Self::Text { common, .. } => Some(common),
            Self::Unknown => None,
        }
    }

    fn reversal_active(&self) -> bool {
        self.common().map(|c| c.reversal_active).unwrap_or(false)
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

    fn spot_is_left(&self) -> Option<bool> {
        match self {
            Self::Spot { spot_is_left, .. } | Self::SpotReversal { spot_is_left, .. } => {
                Some(*spot_is_left)
            }
            _ => None,
        }
    }

    fn text_summary(&self) -> Option<(u32, &str, &str, u32, u32, u32)> {
        match self {
            Self::Text {
                text_regime,
                text_token,
                text_target_next,
                text_outcomes,
                text_shift_every,
                text_vocab_size,
                ..
            } => Some((
                *text_regime,
                text_token.as_str(),
                text_target_next.as_str(),
                *text_outcomes,
                *text_shift_every,
                *text_vocab_size,
            )),
            _ => None,
        }
    }

    fn maze_summary(&self) -> Option<MazeSummary<'_>> {
        match self {
            Self::Maze {
                maze_mode,
                maze_w,
                maze_h,
                maze_player_x,
                maze_player_y,
                maze_goal_x,
                maze_goal_y,
                maze_steps,
                maze_event,
                ..
            } => Some(MazeSummary {
                mode: maze_mode.as_str(),
                w: *maze_w,
                h: *maze_h,
                player_x: *maze_player_x,
                player_y: *maze_player_y,
                goal_x: *maze_goal_x,
                goal_y: *maze_goal_y,
                steps: *maze_steps,
                event: maze_event.as_str(),
            }),
            _ => None,
        }
    }
}

#[derive(Clone, Copy, Debug)]
struct MazeSummary<'a> {
    mode: &'a str,
    w: u32,
    h: u32,
    player_x: u32,
    player_y: u32,
    goal_x: u32,
    goal_y: u32,
    steps: u32,
    event: &'a str,
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
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct BrainStats {
    unit_count: usize,
    connection_count: usize,
    avg_amp: f32,
    avg_weight: f32,
    memory_bytes: usize,
    causal_edges: usize,
    age_steps: u64,

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

fn usage() -> ! {
    eprintln!("braine-cli (talks to brained @ 127.0.0.1:9876 by default)");
    eprintln!("Usage: braine-cli [--addr host:port] <command> [args]\n");
    eprintln!("Commands:");
    eprintln!("  status                      Show daemon state");
    eprintln!("  start | stop                Control run loop");
    eprintln!(
        "  game <spot|bandit|spot_reversal|spotxy|maze|pong|text|replay>  Switch task/game (stop first)"
    );
    eprintln!("  mode <braine|human>         Switch control mode");
    eprintln!("  action <left|right|up|down|stay>  Send human action");
    eprintln!("  trigger <dream|burst|sync|imprint>  Fire learning helpers");
    eprintln!("  save | load | reset         Persistence controls");
    eprintln!("  shutdown                    Save and exit daemon");
    eprintln!("  fps <1-1000>                Set simulation framerate");
    eprintln!("  trialms <10-60000>          Set trial period in milliseconds");
    eprintln!("  tier <scalar|simd|parallel|gpu>  Set execution tier (effective may fall back)");
    eprintln!("  experts <on|off|cull>        Control expert (child brain) mechanism");
    eprintln!("  experts policy <parent_learning> <max_children> <child_reward_scale> <episode_trials> <consolidate_topk> [allow_nested] [max_depth] [persist_mode]");
    eprintln!("                               allow_nested: true|false (default false)");
    eprintln!("                               max_depth: >=1 (default 1)");
    eprintln!("                               persist_mode: full|drop_active (default full)");
    eprintln!("  paths                       Show data directory and brain file path");
    eprintln!("  advisor status              Show advisor configuration + last report");
    eprintln!("  advisor set <on|off> [every_trials] [mode]  Configure advisor");
    eprintln!("  advisor once [apply]         Invoke advisor once (default apply=false)");
    eprintln!("  advisor context [scores]     Print structured advisor context (Braine -> LLM)");
    eprintln!("  advisor apply [eps] [alpha] [ttl] [rationale...]  Apply advice (LLM -> Braine)");
    eprintln!("  replay get                   Print current replay dataset summary");
    eprintln!("  replay set <dataset.json>    Set replay dataset (stop first)");
    eprintln!("  demo text <trials> [advisor] Run a quick text task demo and print summary");
    eprintln!("  demo replay <trials> [dataset.json] [mock_llm]  Run replay demo; optionally mimic LLM via context/apply");
    eprintln!();
    eprintln!("⚠️  RESEARCH DISCLAIMER: This system was developed with LLM assistance under");
    eprintln!("   human guidance. It is a RESEARCH DEMONSTRATION, NOT PRODUCTION-READY.");
    eprintln!("   Do not use for safety-critical or real-world deployment scenarios.");
    process::exit(1);
}

fn llm_stub_advice(
    ctx: &AdvisorContext,
    prev_text_regime: Option<u32>,
) -> (AdvisorAdvice, Option<u32>) {
    let mut rationale_parts: Vec<String> = Vec::new();
    let mut exploration_target: Option<f32> = None;
    let mut meaning_alpha_target: Option<f32> = None;

    let regime_changed = match (prev_text_regime, ctx.text_regime) {
        (Some(a), Some(b)) => a != b,
        (None, Some(_)) => false,
        _ => false,
    };

    if regime_changed {
        rationale_parts
            .push("detected regime change; increasing exploration for adaptation".to_string());
        exploration_target = Some((ctx.exploration_eps + 0.10).min(0.45));
    } else if ctx.trials >= 20 && ctx.recent_rate < 0.55 {
        rationale_parts.push("recent performance low; increasing exploration".to_string());
        exploration_target = Some((ctx.exploration_eps + 0.05).min(0.40));
    } else if ctx.trials >= 20 && ctx.recent_rate > 0.85 {
        rationale_parts.push("recent performance high; annealing exploration".to_string());
        exploration_target = Some((ctx.exploration_eps * 0.85).max(0.02));
    }

    if ctx.trials >= 40 && ctx.recent_rate < 0.45 {
        rationale_parts
            .push("very low performance; slightly increasing meaning weight".to_string());
        meaning_alpha_target = Some((ctx.meaning_alpha + 0.05).min(1.0));
    }

    let rationale = if rationale_parts.is_empty() {
        "no change".to_string()
    } else {
        rationale_parts.join("; ")
    };

    (
        AdvisorAdvice {
            ttl_trials: 50,
            exploration_eps: exploration_target,
            meaning_alpha: meaning_alpha_target,
            rationale,
        },
        ctx.text_regime,
    )
}

fn parse_bool(s: &str) -> Option<bool> {
    match s.trim().to_ascii_lowercase().as_str() {
        "1" | "true" | "t" | "yes" | "y" | "on" => Some(true),
        "0" | "false" | "f" | "no" | "n" | "off" => Some(false),
        _ => None,
    }
}

fn parse_args() -> (String, Vec<String>) {
    let mut args: Vec<String> = std::env::args().skip(1).collect();
    if args.is_empty() {
        usage();
    }

    let mut addr = "127.0.0.1:9876".to_string();
    if args.len() >= 2 && args[0] == "--addr" {
        addr = args[1].clone();
        args.drain(0..2);
    }

    if args.is_empty() {
        usage();
    }

    (addr, args)
}

fn send_request(addr: &str, req: &Request) -> Result<Response, String> {
    let mut stream = TcpStream::connect(addr).map_err(|e| format!("connect: {e}"))?;
    stream
        .set_read_timeout(Some(Duration::from_secs(5)))
        .map_err(|e| format!("set_read_timeout: {e}"))?;
    let mut reader = BufReader::new(stream.try_clone().map_err(|e| format!("clone: {e}"))?);

    let line = serde_json::to_string(req).map_err(|e| format!("serialize: {e}"))?;
    stream
        .write_all(line.as_bytes())
        .and_then(|_| stream.write_all(b"\n"))
        .map_err(|e| format!("send: {e}"))?;

    let mut resp_line = String::new();
    reader
        .read_line(&mut resp_line)
        .map_err(|e| format!("recv: {e}"))?;
    serde_json::from_str(&resp_line).map_err(|e| format!("parse response: {e}"))
}

fn print_state(s: StateSnapshot) {
    println!(
        "mode={:<6} running={} frame={} trials={} correct={} incorrect={} acc={:.1}% recent={:.1}% last100={:.1}% neuromod={:.3}",
        s.mode,
        s.running,
        s.frame,
        s.hud.trials,
        s.hud.correct,
        s.hud.incorrect,
        s.hud.accuracy * 100.0,
        s.hud.recent_rate * 100.0,
        s.hud.last_100_rate * 100.0,
        s.hud.neuromod,
    );
    println!(
        "brain: units={} conns={} avg_amp={:.3} avg_w={:.3} memory={}B causal_edges={} age_steps={}",
        s.brain_stats.unit_count,
        s.brain_stats.connection_count,
        s.brain_stats.avg_amp,
        s.brain_stats.avg_weight,
        s.brain_stats.memory_bytes,
        s.brain_stats.causal_edges,
        s.brain_stats.age_steps,
    );

    // Optional learning/stability monitors (newer daemons).
    if s.brain_stats.plasticity_committed
        || s.brain_stats.plasticity_l1 != 0.0
        || s.brain_stats.eligibility_l1 != 0.0
        || s.brain_stats.homeostasis_bias_l1 != 0.0
        || s.brain_stats.plasticity_budget != 0.0
        || s.brain_stats.learning_deadband != 0.0
        || s.brain_stats.homeostasis_rate != 0.0
    {
        println!(
            "learn: committed={} elig_l1={:.3} dw_l1={:.3} edges={} budget={:.3} used={:.3} deadband={:.3} homeo_rate={:.3} homeo_dbias_l1={:.3}",
            s.brain_stats.plasticity_committed,
            s.brain_stats.eligibility_l1,
            s.brain_stats.plasticity_l1,
            s.brain_stats.plasticity_edges,
            s.brain_stats.plasticity_budget,
            s.brain_stats.plasticity_budget_used,
            s.brain_stats.learning_deadband,
            s.brain_stats.homeostasis_rate,
            s.brain_stats.homeostasis_bias_l1,
        );
    }
    let spot_is_left = s
        .game
        .spot_is_left()
        .map(|v| v.to_string())
        .unwrap_or_else(|| "-".to_string());
    println!(
        "game: kind={} reversal_active={} spot_is_left={} response_made={} trial_frame={} / {}",
        s.game.kind(),
        s.game.reversal_active(),
        spot_is_left,
        s.game.response_made(),
        s.game.trial_frame(),
        s.game.trial_duration(),
    );

    if let Some((regime, token, target, outcomes, shift_every, vocab_size)) = s.game.text_summary()
    {
        println!(
            "text: regime={} token={} target_next={} outcomes={} shift_every={} vocab_size={}",
            regime, token, target, outcomes, shift_every, vocab_size
        );
    }

    if let Some(m) = s.game.maze_summary() {
        println!(
            "maze: mode={} size={}x{} player=({}, {}) goal=({}, {}) steps={} event={}",
            m.mode,
            m.w,
            m.h,
            m.player_x,
            m.player_y,
            m.goal_x,
            m.goal_y,
            m.steps,
            if m.event.is_empty() { "-" } else { m.event }
        );
    }

    if s.experts_enabled || s.experts.active_count > 0 {
        println!(
            "experts: enabled={} active={}/{} persist={} allow_nested={} max_depth={} last_spawn={} last_consolidation={}",
            s.experts_enabled,
            s.experts.active_count,
            s.experts.max_children,
            if s.experts.persistence_mode.is_empty() {
                "full"
            } else {
                s.experts.persistence_mode.as_str()
            },
            s.experts.allow_nested,
            s.experts.max_depth,
            if s.experts.last_spawn_reason.is_empty() {
                "-"
            } else {
                s.experts.last_spawn_reason.as_str()
            },
            if s.experts.last_consolidation.is_empty() {
                "-"
            } else {
                s.experts.last_consolidation.as_str()
            },
        );
        if let Some(ae) = s.active_expert {
            println!(
                "active_expert: id={} ctx={} age_steps={} reward_ema={:.3}",
                ae.id, ae.context_key, ae.age_steps, ae.reward_ema
            );
        }
    }
}

fn main() {
    let (addr, args) = parse_args();
    let cmd = &args[0];

    let make_error = |msg: &str| -> ! {
        eprintln!("{}", msg);
        process::exit(1);
    };

    let req = match cmd.as_str() {
        "status" => Request::GetState,
        "start" => Request::Start,
        "stop" => Request::Stop,
        "game" => {
            if args.len() < 2 {
                usage();
            }
            let game = args[1].clone();
            Request::SetGame { game }
        }
        "mode" => {
            if args.len() < 2 {
                usage();
            }
            let mode = args[1].clone();
            if mode != "braine" && mode != "human" {
                make_error("mode must be 'braine' or 'human'");
            }
            Request::SetMode { mode }
        }
        "action" => {
            if args.len() < 2 {
                usage();
            }
            let action = args[1].clone();
            if action != "left"
                && action != "right"
                && action != "up"
                && action != "down"
                && action != "stay"
            {
                make_error("action must be left|right|up|down|stay");
            }
            Request::HumanAction { action }
        }
        "trigger" => {
            if args.len() < 2 {
                usage();
            }
            match args[1].as_str() {
                "dream" => Request::TriggerDream,
                "burst" => Request::TriggerBurst,
                "sync" => Request::TriggerSync,
                "imprint" => Request::TriggerImprint,
                _ => make_error("trigger must be dream|burst|sync|imprint"),
            }
        }
        "save" => Request::SaveBrain,
        "load" => Request::LoadBrain,
        "reset" => Request::ResetBrain,
        "shutdown" => Request::Shutdown,
        "fps" => {
            if args.len() < 2 {
                usage();
            }
            let fps: u32 = args[1]
                .parse()
                .unwrap_or_else(|_| make_error("fps must be a number (1-1000)"));
            Request::SetFramerate { fps }
        }
        "trialms" => {
            if args.len() < 2 {
                usage();
            }
            let ms: u32 = args[1]
                .parse()
                .unwrap_or_else(|_| make_error("trialms must be a number (10-60000)"));
            Request::SetTrialPeriodMs { ms }
        }
        "tier" => {
            if args.len() < 2 {
                usage();
            }
            let tier = args[1].clone();
            Request::SetExecutionTier { tier }
        }
        "experts" => {
            if args.len() < 2 {
                usage();
            }
            match args[1].as_str() {
                "on" => Request::SetExpertsEnabled { enabled: true },
                "off" => Request::SetExpertsEnabled { enabled: false },
                "cull" => Request::CullExperts,
                "policy" => {
                    if args.len() < 7 {
                        make_error(
                            "usage: experts policy <parent_learning> <max_children> <child_reward_scale> <episode_trials> <consolidate_topk> [allow_nested] [max_depth] [persistence_mode] [shift_delta] [collapse_drop] [collapse_baseline_min]",
                        );
                    }
                    let parent_learning = args[2].clone();
                    let max_children: u32 = args[3]
                        .parse()
                        .unwrap_or_else(|_| make_error("max_children must be a u32"));
                    let child_reward_scale: f32 = args[4]
                        .parse()
                        .unwrap_or_else(|_| make_error("child_reward_scale must be a float"));
                    let episode_trials: u32 = args[5]
                        .parse()
                        .unwrap_or_else(|_| make_error("episode_trials must be a u32"));
                    let consolidate_topk: u32 = args[6]
                        .parse()
                        .unwrap_or_else(|_| make_error("consolidate_topk must be a u32"));

                    let allow_nested: bool = if args.len() >= 8 {
                        parse_bool(&args[7]).unwrap_or_else(|| {
                            make_error("allow_nested must be true|false (or 1|0)")
                        })
                    } else {
                        false
                    };
                    let max_depth: u32 = if args.len() >= 9 {
                        args[8]
                            .parse()
                            .unwrap_or_else(|_| make_error("max_depth must be a u32 >= 1"))
                    } else {
                        1
                    };
                    let persistence_mode: String = if args.len() >= 10 {
                        args[9].clone()
                    } else {
                        "full".to_string()
                    };

                    // Optional signal thresholds (floats):
                    //   <shift_delta> <collapse_drop> <collapse_baseline_min>
                    // Defaults match daemon policy defaults.
                    let reward_shift_ema_delta_threshold: f32 = if args.len() >= 11 {
                        args[10]
                            .parse()
                            .unwrap_or_else(|_| make_error("shift_delta must be a float"))
                    } else {
                        0.55
                    };
                    let performance_collapse_drop_threshold: f32 = if args.len() >= 12 {
                        args[11]
                            .parse()
                            .unwrap_or_else(|_| make_error("collapse_drop must be a float"))
                    } else {
                        0.65
                    };
                    let performance_collapse_baseline_min: f32 = if args.len() >= 13 {
                        args[12]
                            .parse()
                            .unwrap_or_else(|_| make_error("collapse_baseline_min must be a float"))
                    } else {
                        0.25
                    };
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
                    }
                }
                _ => usage(),
            }
        }
        "paths" => {
            // Special command: doesn't need daemon, just print paths
            #[cfg(unix)]
            {
                if let Ok(home) = std::env::var("HOME") {
                    let data_dir = format!("{}/.local/share/braine", home);
                    println!("Data directory: {}", data_dir);
                    println!("Brain file: {}/braine.bbi", data_dir);
                }
            }
            #[cfg(windows)]
            {
                if let Ok(appdata) = std::env::var("APPDATA") {
                    let data_dir = format!("{}\\Braine", appdata);
                    println!("Data directory: {}", data_dir);
                    println!("Brain file: {}\\braine.bbi", data_dir);
                }
            }
            process::exit(0);
        }
        "advisor" => {
            if args.len() < 2 {
                usage();
            }
            match args[1].as_str() {
                "status" => Request::AdvisorGet,
                "set" => {
                    if args.len() < 3 {
                        make_error("usage: advisor set <on|off> [every_trials] [mode]");
                    }
                    let enabled = parse_bool(&args[2]).unwrap_or_else(|| {
                        make_error("advisor set: <on|off> must be true|false (or on|off)")
                    });
                    let every_trials: Option<u32> = if args.len() >= 4 {
                        Some(
                            args[3]
                                .parse()
                                .unwrap_or_else(|_| make_error("every_trials must be a u32")),
                        )
                    } else {
                        None
                    };
                    let mode: Option<String> = if args.len() >= 5 {
                        Some(args[4].clone())
                    } else {
                        None
                    };
                    Request::AdvisorSet {
                        enabled,
                        every_trials,
                        mode,
                    }
                }
                "once" => {
                    let apply = if args.len() >= 3 {
                        parse_bool(&args[2])
                            .unwrap_or_else(|| make_error("apply must be true|false (or 1|0)"))
                    } else {
                        false
                    };
                    Request::AdvisorOnce { apply }
                }
                "context" => {
                    let include_action_scores = args.len() >= 3 && args[2].as_str() == "scores";
                    Request::AdvisorContext {
                        include_action_scores,
                    }
                }
                "apply" => {
                    // usage: advisor apply [eps] [alpha] [ttl] [rationale...]
                    let exploration_eps: Option<f32> = if args.len() >= 3 {
                        if args[2].trim() == "-" {
                            None
                        } else {
                            Some(
                                args[2]
                                    .parse()
                                    .unwrap_or_else(|_| make_error("eps must be a float or '-'")),
                            )
                        }
                    } else {
                        None
                    };

                    let meaning_alpha: Option<f32> = if args.len() >= 4 {
                        if args[3].trim() == "-" {
                            None
                        } else {
                            Some(
                                args[3]
                                    .parse()
                                    .unwrap_or_else(|_| make_error("alpha must be a float or '-'")),
                            )
                        }
                    } else {
                        None
                    };

                    let ttl_trials: u32 = if args.len() >= 5 {
                        args[4]
                            .parse()
                            .unwrap_or_else(|_| make_error("ttl must be a u32"))
                    } else {
                        50
                    };

                    let rationale: String = if args.len() >= 6 {
                        args[5..].join(" ")
                    } else {
                        "manual".to_string()
                    };

                    Request::AdvisorApply {
                        advice: AdvisorAdvice {
                            ttl_trials,
                            exploration_eps,
                            meaning_alpha,
                            rationale,
                        },
                    }
                }
                _ => usage(),
            }
        }
        "replay" => {
            if args.len() < 2 {
                usage();
            }
            match args[1].as_str() {
                "get" => Request::ReplayGetDataset,
                "set" => {
                    if args.len() < 3 {
                        make_error("usage: replay set <dataset.json>");
                    }
                    let path = args[2].as_str();
                    let raw = std::fs::read_to_string(path)
                        .unwrap_or_else(|e| make_error(&format!("failed to read {path}: {e}")));
                    let dataset: ReplayDataset = serde_json::from_str(&raw).unwrap_or_else(|e| {
                        make_error(&format!("failed to parse dataset json: {e}"))
                    });
                    Request::ReplaySetDataset { dataset }
                }
                _ => usage(),
            }
        }
        "demo" => {
            if args.len() < 3 {
                make_error("usage: demo text <trials> [advisor_on] | demo replay <trials> [dataset.json] [mock_llm]");
            }
            let kind = args[1].as_str();
            let target_trials: u32 = args[2]
                .parse()
                .unwrap_or_else(|_| make_error("trials must be a u32"));

            let advisor_on: bool = if kind == "text" {
                if args.len() >= 4 {
                    parse_bool(&args[3])
                        .unwrap_or_else(|| make_error("advisor_on must be true|false (or 1|0)"))
                } else {
                    false
                }
            } else {
                false
            };

            let dataset_path: Option<String> = if kind == "replay" && args.len() >= 4 {
                Some(args[3].clone())
            } else {
                None
            };

            let mock_llm: bool = if kind == "replay" && args.len() >= 5 {
                parse_bool(&args[4])
                    .unwrap_or_else(|| make_error("mock_llm must be true|false (or 1|0)"))
            } else {
                false
            };

            // Orchestrate via multiple daemon requests.
            let must = |r: &Request| match send_request(&addr, r) {
                Ok(Response::Success { .. })
                | Ok(Response::State(_))
                | Ok(Response::AdvisorStatus { .. })
                | Ok(Response::AdvisorReport { .. }) => {}
                Ok(Response::Error { message }) => make_error(&format!("daemon error: {message}")),
                Ok(_) => {}
                Err(e) => make_error(&format!("daemon request failed: {e}")),
            };

            must(&Request::Stop);
            must(&Request::ResetBrain);
            if kind == "replay" {
                if let Some(p) = &dataset_path {
                    let raw = std::fs::read_to_string(p)
                        .unwrap_or_else(|e| make_error(&format!("failed to read {p}: {e}")));
                    let dataset: ReplayDataset = serde_json::from_str(&raw).unwrap_or_else(|e| {
                        make_error(&format!("failed to parse dataset json: {e}"))
                    });
                    must(&Request::ReplaySetDataset { dataset });
                }
                must(&Request::SetGame {
                    game: "replay".to_string(),
                });
            } else if kind == "text" {
                must(&Request::SetGame {
                    game: "text".to_string(),
                });
            } else {
                make_error("demo: only 'text' and 'replay' are implemented right now");
            }
            must(&Request::CfgSet {
                exploration_eps: Some(0.25),
                meaning_alpha: None,
                target_fps: Some(120),
                trial_period_ms: Some(40),
                max_units: None,
            });
            if kind == "text" {
                must(&Request::AdvisorSet {
                    enabled: advisor_on,
                    every_trials: Some(25),
                    mode: Some("stub".to_string()),
                });
            } else if kind == "replay" {
                // For replay demos we usually want the explicit boundary (mock_llm) rather than the built-in stub.
                must(&Request::AdvisorSet {
                    enabled: false,
                    every_trials: Some(25),
                    mode: Some("off".to_string()),
                });
            }
            must(&Request::Start);

            let start = std::time::Instant::now();
            let mut last_trials: u32 = 0;
            let mut last_acc: f32 = 0.0;
            let mut prev_text_regime: Option<u32> = None;
            let mut last_advised_trials: u32 = 0;
            loop {
                match send_request(&addr, &Request::GetState) {
                    Ok(Response::State(s)) => {
                        let trials = s.hud.trials;
                        if trials != last_trials {
                            last_trials = trials;
                            last_acc = s.hud.last_100_rate;
                        }

                        if kind == "replay" && mock_llm {
                            // Mimic an external LLM: poll context occasionally and push back advice.
                            if trials >= 25 && trials.saturating_sub(last_advised_trials) >= 25 {
                                if let Ok(Response::AdvisorContext { context, .. }) = send_request(
                                    &addr,
                                    &Request::AdvisorContext {
                                        include_action_scores: false,
                                    },
                                ) {
                                    let (advice, next_prev) =
                                        llm_stub_advice(&context, prev_text_regime);
                                    prev_text_regime = next_prev;
                                    let _ = send_request(&addr, &Request::AdvisorApply { advice });
                                    last_advised_trials = trials;
                                }
                            }
                        }
                        if trials >= target_trials {
                            break;
                        }
                    }
                    Ok(Response::Error { message }) => {
                        make_error(&format!("GetState failed: {message}"))
                    }
                    Ok(_) => {}
                    Err(e) => make_error(&format!("GetState failed: {e}")),
                }
                std::thread::sleep(Duration::from_millis(50));
            }

            must(&Request::Stop);

            let elapsed = start.elapsed().as_secs_f32();
            if kind == "text" {
                println!(
                    "demo=text trials={} advisor={} elapsed_s={:.2} last100_rate={:.3}",
                    target_trials, advisor_on, elapsed, last_acc
                );
            } else {
                println!(
                    "demo=replay trials={} dataset={} mock_llm={} elapsed_s={:.2} last100_rate={:.3}",
                    target_trials,
                    dataset_path.as_deref().unwrap_or("(daemon default)"),
                    mock_llm,
                    elapsed,
                    last_acc
                );
            }
            process::exit(0);
        }
        _ => usage(),
    };

    match send_request(&addr, &req) {
        Ok(Response::State(s)) => print_state(*s),
        Ok(Response::GameParams { game, params }) => {
            println!("Game params schema for: {game}");
            for p in params {
                if p.description.trim().is_empty() {
                    println!(
                        "- {} ({}) min={} max={} default={}",
                        p.key, p.label, p.min, p.max, p.default
                    );
                } else {
                    println!(
                        "- {} ({}) min={} max={} default={} — {}",
                        p.key, p.label, p.min, p.max, p.default, p.description
                    );
                }
            }
        }
        Ok(Response::Success { message }) => println!("{message}"),
        Ok(Response::AdvisorStatus {
            config,
            last_report,
        }) => {
            println!(
                "advisor: enabled={} mode={} every_trials={}",
                config.enabled, config.mode, config.every_trials
            );
            if let Some(r) = last_report {
                println!(
                    "last_report: at_trials={} applied={} rationale={}",
                    r.at_trials, r.applied, r.advice.rationale
                );
            }
        }
        Ok(Response::AdvisorReport { report, applied }) => {
            println!(
                "advisor_report: applied={} at_trials={} game={} recent_rate={:.3} eps_before={:.3} advice_eps={:?} rationale={}",
                applied,
                report.at_trials,
                report.context.game,
                report.context.recent_rate,
                report.context.exploration_eps,
                report.advice.exploration_eps,
                report.advice.rationale
            );
        }
        Ok(Response::AdvisorContext {
            context,
            action_scores,
        }) => {
            // Print JSON so external tools can consume it.
            let out = serde_json::json!({
                "context": context,
                "action_scores": action_scores,
            });
            println!(
                "{}",
                serde_json::to_string_pretty(&out)
                    .unwrap_or_else(|_| "{\"error\":\"failed to serialize\"}".to_string())
            );
        }
        Ok(Response::ReplayDataset { dataset }) => {
            println!(
                "replay_dataset: name={} trials={}",
                dataset.name,
                dataset.trials.len()
            );
        }
        Ok(Response::Error { message }) => {
            eprintln!("Error: {message}");
            process::exit(1);
        }
        Err(e) => {
            eprintln!("Failed: {e}");
            process::exit(1);
        }
    }
}
