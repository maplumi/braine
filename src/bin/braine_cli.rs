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
    Start,
    Stop,
    SetMode { mode: String },
    HumanAction { action: String },
    TriggerDream,
    TriggerBurst,
    TriggerSync,
    TriggerImprint,
    SaveBrain,
    LoadBrain,
    ResetBrain,
    Shutdown,
    SetFramerate { fps: u32 },
    SetTrialPeriodMs { ms: u32 },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
enum Response {
    State(StateSnapshot),
    Success { message: String },
    Error { message: String },
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
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct GameState {
    spot_is_left: bool,
    response_made: bool,
    trial_frame: u32,
    trial_duration: u32,
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
    eprintln!("  mode <braine|human>         Switch control mode");
    eprintln!("  action <left|right>         Send human action");
    eprintln!("  trigger <dream|burst|sync|imprint>  Fire learning helpers");
    eprintln!("  save | load | reset         Persistence controls");
    eprintln!("  shutdown                    Save and exit daemon");
    eprintln!("  fps <1-1000>                Set simulation framerate");
    eprintln!("  trialms <10-60000>          Set trial period in milliseconds");
    eprintln!("  paths                       Show data directory and brain file path");
    process::exit(1);
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
    println!(
        "game: spot_is_left={} response_made={} trial_frame={} / {}",
        s.game.spot_is_left, s.game.response_made, s.game.trial_frame, s.game.trial_duration,
    );
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
        Ok(Response::State(s)) => print_state(s),
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
