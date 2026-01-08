//! Braine Visualizer - Slint UI client
//! Connects to the `brained` daemon over TCP (127.0.0.1:9876)

use serde::{Deserialize, Serialize};
use slint::{ModelRc, Timer, VecModel};
use std::cell::RefCell;
use std::io::{BufRead, BufReader, Write};
use std::net::TcpStream;
use std::rc::Rc;
use std::sync::{mpsc, Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};

slint::include_modules!();

// ═══════════════════════════════════════════════════════════════════════════
// Protocol (mirrors brained daemon)
// ═══════════════════════════════════════════════════════════════════════════

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
    SetFramerate { fps: u32 },
    SetTrialPeriodMs { ms: u32 },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
enum Response {
    State(DaemonStateSnapshot),
    Success { message: String },
    Error { message: String },
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
struct DaemonStateSnapshot {
    running: bool,
    mode: String,
    frame: u64,
    #[serde(default)]
    target_fps: u32,
    game: DaemonGameState,
    hud: DaemonHudData,
    brain_stats: DaemonBrainStats,
    #[serde(default)]
    unit_plot: Vec<DaemonUnitPlotPoint>,
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
struct DaemonGameState {
    spot_is_left: bool,
    response_made: bool,
    trial_frame: u32,
    trial_duration: u32,
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
    connection_count: usize,
    #[serde(default)]
    pruned_last_step: usize,
    #[serde(default)]
    births_last_step: usize,
    #[serde(default)]
    saturated: bool,
    avg_amp: f32,
    avg_weight: f32,
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
}

impl DaemonClient {
    fn new() -> Self {
        let (tx, rx) = mpsc::channel::<Request>();
        let snapshot = Arc::new(Mutex::new(DaemonStateSnapshot::default()));
        let snap_clone = Arc::clone(&snapshot);

        // Background worker: manages TCP connection and request/response loop
        thread::spawn(move || loop {
            match TcpStream::connect("127.0.0.1:9876") {
                Ok(mut stream) => {
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
                                    *s = state;
                                }
                            }
                            Ok(Response::Success { .. }) => {
                                // nothing to store
                            }
                            Ok(Response::Error { message }) => {
                                eprintln!("Daemon error: {}", message);
                            }
                            Err(e) => eprintln!("Bad response: {}", e),
                        }
                    }
                }
                Err(e) => {
                    eprintln!("Unable to connect to brained: {}. Retrying in 1s...", e);
                    thread::sleep(Duration::from_secs(1));
                }
            }
        });

        Self { tx, snapshot }
    }

    fn send(&self, req: Request) {
        let _ = self.tx.send(req);
    }

    fn snapshot(&self) -> DaemonStateSnapshot {
        self.snapshot.lock().map(|s| s.clone()).unwrap_or_default()
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Main
// ═══════════════════════════════════════════════════════════════════════════

fn main() -> Result<(), slint::PlatformError> {
    let ui = MainWindow::new()?;
    let client = Rc::new(DaemonClient::new());

    // Track local control changes so the next poll doesn't immediately overwrite
    // the slider position before the daemon applies the request.
    let pending_fps: Rc<RefCell<Option<(u32, Instant)>>> = Rc::new(RefCell::new(None));
    let pending_trial_ms: Rc<RefCell<Option<(u32, Instant)>>> = Rc::new(RefCell::new(None));

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

    // Human input
    {
        let c = client.clone();
        ui.on_human_key_pressed(move |key| match key.as_str() {
            "left" => c.send(Request::HumanAction {
                action: "left".into(),
            }),
            "right" => c.send(Request::HumanAction {
                action: "right".into(),
            }),
            _ => {}
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

    type PollFn = Rc<dyn Fn(Duration)>;
    let poll_fn: Rc<RefCell<Option<PollFn>>> = Rc::new(RefCell::new(None));
    let poll_fn_setter = poll_fn.clone();

    let poll_impl: PollFn = Rc::new(move |delay: Duration| {
        let ui_weak = ui_weak.clone();
        let c = c.clone();
        let poll_fn = poll_fn_setter.clone();
        let pending_fps_poll = pending_fps_poll.clone();
        let pending_trial_ms_poll = pending_trial_ms_poll.clone();

        Timer::single_shot(delay, move || {
            c.send(Request::GetState);
            let snap = c.snapshot();

            let now = Instant::now();

            let mut next_delay = Duration::from_millis(100);
            if let Some(ui) = ui_weak.upgrade() {
                ui.set_game(GameState {
                    spot_is_left: snap.game.spot_is_left,
                    response_made: snap.game.response_made,
                    trial_frame: snap.game.trial_frame as i32,
                    trial_duration: snap.game.trial_duration as i32,
                });
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
                ui.set_brain_stats(BrainStats {
                    unit_count: snap.brain_stats.unit_count as i32,
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
                ui.set_is_braine_mode(snap.mode != "human");
                ui.set_running(snap.running);

                // Keep the UI's displayed FPS in sync with the daemon.
                if snap.target_fps != 0 {
                    let apply = match pending_fps_poll.borrow().as_ref() {
                        Some((want, t))
                            if now.duration_since(*t) < Duration::from_millis(800)
                                && snap.target_fps != *want =>
                        {
                            false
                        }
                        Some((want, _)) if snap.target_fps == *want => {
                            pending_fps_poll.borrow_mut().take();
                            true
                        }
                        Some((_, t)) if now.duration_since(*t) >= Duration::from_millis(800) => {
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
                if snap.game.trial_duration != 0 {
                    let apply = match pending_trial_ms_poll.borrow().as_ref() {
                        Some((want, t))
                            if now.duration_since(*t) < Duration::from_millis(800)
                                && snap.game.trial_duration != *want =>
                        {
                            false
                        }
                        Some((want, _)) if snap.game.trial_duration == *want => {
                            pending_trial_ms_poll.borrow_mut().take();
                            true
                        }
                        Some((_, t)) if now.duration_since(*t) >= Duration::from_millis(800) => {
                            pending_trial_ms_poll.borrow_mut().take();
                            true
                        }
                        None => true,
                        _ => true,
                    };

                    if apply {
                        ui.set_trial_period_ms(snap.game.trial_duration as i32);
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
