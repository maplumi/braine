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

use braine::substrate::Stimulus;
use braine::substrate::{Brain, BrainConfig, UnitPlotPoint};
use serde::{Deserialize, Serialize};
use std::fs::File;
use std::io::Write as _;
use std::sync::Arc;
use std::time::Duration;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::net::{TcpListener, TcpStream};
use tokio::sync::RwLock;
use tokio::time;
use tracing::{error, info, warn};

mod game;
mod paths;

use game::SpotGame;
use paths::AppPaths;

// ═══════════════════════════════════════════════════════════════════════════
// Protocol Messages
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
    target_fps: u32,
    game: GameState,
    hud: HudData,
    brain_stats: BrainStats,
    #[serde(default)]
    unit_plot: Vec<UnitPlotPoint>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
struct PersistedRuntime {
    game: PersistedGameStats,
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
    connection_count: usize,
    pruned_last_step: usize,
    births_last_step: usize,
    saturated: bool,
    avg_amp: f32,
    avg_weight: f32,
    memory_bytes: usize,
    causal_base_symbols: usize,
    causal_edges: usize,
    causal_last_directed_edge_updates: usize,
    causal_last_cooccur_edge_updates: usize,
    age_steps: u64,
}

// ═══════════════════════════════════════════════════════════════════════════
// Daemon State
// ═══════════════════════════════════════════════════════════════════════════

struct DaemonState {
    brain: Brain,
    game: SpotGame,
    running: bool,
    frame: u64,
    paths: AppPaths,
    exploration_eps: f32,
    meaning_alpha: f32,
    rng_state: u64,
    last_autosave_trial: u32,
    target_fps: u32,
    trial_period_ms: u32,
    pending_neuromod: f32,
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
        });

        brain.define_sensor("spot_left", 4);
        brain.define_sensor("spot_right", 4);
        brain.define_action("left", 6);
        brain.define_action("right", 6);
        brain.set_observer_telemetry(true);

        Self {
            brain,
            game: SpotGame::new(),
            running: false,
            frame: 0,
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

        let stimulus = self.game.stimulus_name();

        // Apply last tick's reward as neuromodulation for one step.
        self.brain.set_neuromodulator(self.pending_neuromod);
        self.pending_neuromod = 0.0;

        // Observe stimulus and advance dynamics.
        self.brain.apply_stimulus(Stimulus::new(stimulus, 1.0));
        self.brain.step();

        // Decide and (optionally) score once per trial.
        let mut completed = false;
        if !self.game.response_made {
            // Epsilon-greedy: explore randomly sometimes; otherwise exploit meaning+habit.
            let explore = self.rng_next_f32() < self.exploration_eps;
            let (action_idx, _conf) = if explore {
                // Spot always has exactly two actions: left/right.
                ((self.rng_next_u64() & 1) as usize, 0.0)
            } else {
                self.brain
                    .select_action_with_meaning_index(stimulus, self.meaning_alpha)
            };

            let action_name = self.brain.action_name(action_idx).unwrap_or("idle");

            // Score once per trial (no brain mutation needed for scoring).
            if let Some((reward, done)) = self.game.score_action(action_name) {
                completed = done;

                // Record action + ctx::action pair for causal/meaning memory.
                self.brain.note_action_index(action_idx);
                self.brain.note_pair_index(stimulus, action_idx);

                // Record reward for meaning/causality *this* tick.
                self.brain.set_neuromodulator(reward);
                self.brain.reinforce_action_index(action_idx, reward);

                // Also apply the reward as neuromodulation on the next dynamics step.
                self.pending_neuromod = reward;
            } else {
                // Even if not scoring (shouldn't happen in Spot), still record action intent.
                self.brain.note_action_index(action_idx);
                self.brain.note_pair_index(stimulus, action_idx);

                // Ensure no stale reward sticks when we're not scoring.
                self.brain.set_neuromodulator(0.0);
            }
        } else {
            // Waiting for next scheduled trial.
            self.brain.set_neuromodulator(0.0);
        }

        // Commit perception/action/reward symbols into causal memory.
        self.brain.commit_observation();

        if completed {
            // Anneal exploration but keep a small floor for on-policy correction
            self.exploration_eps = (self.exploration_eps * 0.99).max(0.02);

            // Automatic, bounded neurogenesis: add capacity if the network is saturating.
            // This keeps external influence minimal while preventing long-run brittleness.
            // Tuning notes:
            // - lower threshold => grows earlier
            // - higher max_units => allows more capacity but costs memory/compute
            let _grown = self.brain.maybe_neurogenesis(0.35, 1, 256);

            // Auto-save frequently so short sessions still persist.
            let trials_since_save = self.game.trials - self.last_autosave_trial;
            if trials_since_save >= 10 {
                info!(
                    "Auto-save triggered: {} trials since last save (trial {})",
                    trials_since_save, self.game.trials
                );
                match self.save_brain() {
                    Ok(_) => {
                        info!("✓ Auto-save succeeded at trial {}", self.game.trials);
                        self.last_autosave_trial = self.game.trials;
                    }
                    Err(e) => {
                        error!("✗ Auto-save FAILED at trial {}: {}", self.game.trials, e);
                    }
                }
            }
        }

        self.frame += 1;
    }

    fn get_snapshot(&self) -> StateSnapshot {
        let diag = self.brain.diagnostics();
        let causal = self.brain.causal_stats();

        StateSnapshot {
            running: self.running,
            mode: "braine".to_string(),
            frame: self.frame,
            target_fps: self.target_fps,
            game: GameState {
                spot_is_left: self.game.spot_is_left,
                response_made: self.game.response_made,
                trial_frame: self.game.trial_frame,
                trial_duration: self.trial_period_ms,
            },
            hud: HudData {
                trials: self.game.trials,
                correct: self.game.correct,
                incorrect: self.game.incorrect,
                accuracy: self.game.accuracy(),
                recent_rate: self.game.recent_rate(),
                last_100_rate: self.game.last_100_rate(),
                neuromod: self.brain.neuromodulator(),
                learning_at_trial: self.game.learning_at_trial.map(|v| v as i32).unwrap_or(-1),
                learned_at_trial: self.game.learned_at_trial.map(|v| v as i32).unwrap_or(-1),
                mastered_at_trial: self.game.mastered_at_trial.map(|v| v as i32).unwrap_or(-1),
            },
            brain_stats: BrainStats {
                unit_count: diag.unit_count,
                connection_count: diag.connection_count,
                pruned_last_step: diag.pruned_last_step,
                births_last_step: diag.births_last_step,
                saturated: self.brain.should_grow(0.35),
                avg_amp: diag.avg_amp,
                avg_weight: diag.avg_weight,
                memory_bytes: diag.memory_bytes,
                causal_base_symbols: causal.base_symbols,
                causal_edges: causal.edges,
                causal_last_directed_edge_updates: causal.last_directed_edge_updates,
                causal_last_cooccur_edge_updates: causal.last_cooccur_edge_updates,
                age_steps: self.brain.age_steps(),
            },
            unit_plot: self.brain.unit_plot_points(128),
        }
    }

    fn save_brain(&self) -> Result<(), String> {
        let path = self.paths.brain_file();
        info!("Saving brain to {:?}", path);

        // Ensure directory exists
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).map_err(|e| {
                let msg = format!("Failed to create directory {:?}: {}", parent, e);
                error!("{}", msg);
                msg
            })?;
        }

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

        self.brain.save_image_to(&mut file).map_err(|e| {
            let msg = format!("Failed to serialize brain: {}", e);
            error!("{}", msg);
            msg
        })?;

        // Persist runtime/task metrics alongside the brain so UI progress doesn't reset.
        let runtime = PersistedRuntime {
            game: PersistedGameStats {
                correct: self.game.correct,
                incorrect: self.game.incorrect,
                trials: self.game.trials,
                recent: self.game.recent.clone(),
                learning_at_trial: self.game.learning_at_trial,
                learned_at_trial: self.game.learned_at_trial,
                mastered_at_trial: self.game.mastered_at_trial,
            },
        };
        let rt_path = self.paths.runtime_state_file();
        let json = serde_json::to_vec_pretty(&runtime)
            .map_err(|e| format!("Failed to encode runtime state: {e}"))?;
        let mut rt = File::create(&rt_path)
            .map_err(|e| format!("Failed to create runtime state file {:?}: {e}", rt_path))?;
        rt.write_all(&json)
            .map_err(|e| format!("Failed to write runtime state file {:?}: {e}", rt_path))?;

        info!("✓ Brain saved successfully to {:?}", path);
        Ok(())
    }

    fn load_brain(&mut self) -> Result<(), String> {
        let path = self.paths.brain_file();
        if !path.exists() {
            return Err(format!("Brain file not found: {:?}", path));
        }
        let mut file = File::open(&path).map_err(|e| format!("Failed to open file: {}", e))?;
        self.brain = Brain::load_image_from(&mut file)
            .map_err(|e| format!("Failed to load brain: {}", e))?;

        // Ensure required IO groups exist (for backwards compatibility with older images).
        self.brain.ensure_sensor("spot_left", 4);
        self.brain.ensure_sensor("spot_right", 4);
        self.brain.ensure_action("left", 6);
        self.brain.ensure_action("right", 6);
        self.brain.set_observer_telemetry(true);

        // Load runtime/task metrics if present.
        let rt_path = self.paths.runtime_state_file();
        if rt_path.exists() {
            match std::fs::read_to_string(&rt_path)
                .ok()
                .and_then(|s| serde_json::from_str::<PersistedRuntime>(&s).ok())
            {
                Some(rt) => {
                    self.game.correct = rt.game.correct;
                    self.game.incorrect = rt.game.incorrect;
                    self.game.trials = rt.game.trials;
                    self.game.recent = rt.game.recent;
                    self.game.learning_at_trial = rt.game.learning_at_trial;
                    self.game.learned_at_trial = rt.game.learned_at_trial;
                    self.game.mastered_at_trial = rt.game.mastered_at_trial;
                }
                None => warn!("Failed to parse runtime state file {:?}", rt_path),
            }
        }
        info!("Brain loaded from {:?}", path);
        Ok(())
    }

    fn reset_brain(&mut self) {
        *self = Self::new(self.paths.clone());
        info!("Brain reset to initial state");
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
            Request::GetState => {
                let s = state.read().await;
                Response::State(s.get_snapshot())
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
            Request::SetMode { .. } => {
                // Spot is Braine-only; mode switching disabled
                Response::Error {
                    message: "Spot game is Braine-only".to_string(),
                }
            }
            Request::HumanAction { .. } => {
                // Not supported in Spot game
                Response::Error {
                    message: "Human control not available for Spot".to_string(),
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
    info!("Data directory: {:?}", paths.data_dir());
    info!("Brain file: {:?}", paths.brain_file());

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
