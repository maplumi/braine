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
    #[serde(rename = "pong")]
    Pong {
        #[serde(flatten)]
        common: GameCommon,
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
            Self::Pong { .. } => "pong",
            Self::Unknown => "unknown",
        }
    }

    fn common(&self) -> Option<&GameCommon> {
        match self {
            Self::Spot { common, .. }
            | Self::Bandit { common }
            | Self::SpotReversal { common, .. }
            | Self::SpotXY { common }
            | Self::Pong { common } => Some(common),
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
}

fn usage() -> ! {
    eprintln!("braine-cli (talks to brained @ 127.0.0.1:9876 by default)");
    eprintln!("Usage: braine-cli [--addr host:port] <command> [args]\n");
    eprintln!("Commands:");
    eprintln!("  status                      Show daemon state");
    eprintln!("  start | stop                Control run loop");
    eprintln!("  game <spot|bandit|spot_reversal>  Switch task/game (stop first)");
    eprintln!("  mode <braine|human>         Switch control mode");
    eprintln!("  action <left|right>         Send human action");
    eprintln!("  trigger <dream|burst|sync|imprint>  Fire learning helpers");
    eprintln!("  save | load | reset         Persistence controls");
    eprintln!("  shutdown                    Save and exit daemon");
    eprintln!("  fps <1-1000>                Set simulation framerate");
    eprintln!("  trialms <10-60000>          Set trial period in milliseconds");
    eprintln!("  experts <on|off|cull>        Control expert (child brain) mechanism");
    eprintln!("  experts policy <parent_learning> <max_children> <child_reward_scale> <episode_trials> <consolidate_topk> [allow_nested] [max_depth] [persist_mode]");
    eprintln!("                               allow_nested: true|false (default false)");
    eprintln!("                               max_depth: >=1 (default 1)");
    eprintln!("                               persist_mode: full|drop_active (default full)");
    eprintln!("  paths                       Show data directory and brain file path");
    process::exit(1);
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
            if action != "left" && action != "right" {
                make_error("action must be 'left' or 'right'");
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
                    println!("Brain file: {}/brain.bbi", data_dir);
                }
            }
            #[cfg(windows)]
            {
                if let Ok(appdata) = std::env::var("APPDATA") {
                    let data_dir = format!("{}\\Braine", appdata);
                    println!("Data directory: {}", data_dir);
                    println!("Brain file: {}\\brain.bbi", data_dir);
                }
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
