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

// ═══════════════════════════════════════════════════════════════════════════
// Protocol (mirrors brained daemon)
// ═══════════════════════════════════════════════════════════════════════════

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
enum Request {
    GetState,
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
    SetMaxUnits {
        max_units: u32,
    },

    // Storage / snapshots
    SaveSnapshot,
    LoadSnapshot {
        stem: String,
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
    SetFramerate {
        fps: u32,
    },
    SetTrialPeriodMs {
        ms: u32,
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
    Success { message: String },
    Graph(Box<DaemonGraphSnapshot>),
    Error { message: String },
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
struct DaemonGameState {
    #[serde(default)]
    kind: String,
    #[serde(default)]
    reversal_active: bool,
    #[serde(default)]
    chosen_action: String,
    #[serde(default)]
    pos_x: f32,
    #[serde(default)]
    pos_y: f32,
    #[serde(default)]
    spotxy_eval: bool,
    #[serde(default)]
    spotxy_mode: String,
    #[serde(default)]
    spotxy_grid_n: u32,
    #[serde(default)]
    last_reward: f32,
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
    #[serde(default)]
    max_units_limit: usize,
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
    graph: Arc<Mutex<DaemonGraphSnapshot>>,
}

impl DaemonClient {
    fn new() -> Self {
        let (tx, rx) = mpsc::channel::<Request>();
        let snapshot = Arc::new(Mutex::new(DaemonStateSnapshot::default()));
        let graph = Arc::new(Mutex::new(DaemonGraphSnapshot::default()));
        let snap_clone = Arc::clone(&snapshot);
        let graph_clone = Arc::clone(&graph);

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
                                    *s = *state;
                                }
                            }
                            Ok(Response::Graph(g)) => {
                                if let Ok(mut gs) = graph_clone.lock() {
                                    *gs = *g;
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

        Self {
            tx,
            snapshot,
            graph,
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
        });
    }

    // View toggle: parent vs active expert
    {
        let c = client.clone();
        ui.on_view_mode_changed(move |mode| {
            c.send(Request::SetView {
                view: mode.to_string(),
            });
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

    // Initial graph fetch (small defaults)
    {
        *last_graph_req.borrow_mut() = Instant::now();
        client.send(Request::GetGraph {
            kind: "substrate".to_string(),
            max_nodes: 32,
            max_edges: 64,
            include_isolated: false,
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
    let last_graph_req_poll = last_graph_req.clone();
    let graph_hover_nodes_poll = graph_hover_nodes.clone();
    let graph_hover_edges_poll = graph_hover_edges.clone();
    let graph_pos_by_kind_poll = graph_pos_by_kind.clone();
    let graph_temp_by_kind_poll = graph_temp_by_kind.clone();

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

        Timer::single_shot(delay, move || {
            c.send(Request::GetState);
            let snap = c.snapshot();
            let g = c.graph_snapshot();

            let now = Instant::now();

            let mut next_delay = Duration::from_millis(100);
            if let Some(ui) = ui_weak.upgrade() {
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

                    let (cx, cy) = parse(&snap.game.chosen_action, snap.game.spotxy_grid_n)
                        .unwrap_or((-1, -1));
                    let (kx, ky) = parse(&snap.meaning.correct_action, snap.game.spotxy_grid_n)
                        .unwrap_or((-1, -1));
                    (cx, cy, kx, ky)
                };

                ui.set_game(GameState {
                    kind: snap.game.kind.clone().into(),
                    reversal_active: snap.game.reversal_active,
                    chosen_action: snap.game.chosen_action.clone().into(),
                    pos_x: snap.game.pos_x,
                    pos_y: snap.game.pos_y,
                    spotxy_eval: snap.game.spotxy_eval,
                    spotxy_mode: snap.game.spotxy_mode.clone().into(),
                    spotxy_grid_n: snap.game.spotxy_grid_n as i32,
                    spotxy_chosen_ix,
                    spotxy_chosen_iy,
                    spotxy_correct_ix,
                    spotxy_correct_iy,
                    last_reward: snap.game.last_reward,
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
                    max_units_limit: snap.brain_stats.max_units_limit as i32,
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
                    let active_id = snap
                        .active_expert
                        .as_ref()
                        .map(|a| format!("active #{}", a.id))
                        .unwrap_or_else(|| "active -".to_string());
                    let deployed = if total > active {
                        format!("{active}/{max} (total {total})")
                    } else {
                        format!("{active}/{max}")
                    };
                    format!(
                        "Experts: {}  {}  persist={}  {}",
                        deployed, active_id, persist, nested
                    )
                } else {
                    String::new()
                };
                ui.set_experts_status(experts_status.into());

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
                    for e in &g.edges {
                        let Some(&(x1, y1)) = pos_by_id.get(&e.from) else {
                            continue;
                        };
                        let Some(&(x2, y2)) = pos_by_id.get(&e.to) else {
                            continue;
                        };

                        let strength01 = (e.weight.abs() * inv).clamp(0.0, 1.0);
                        let positive = e.weight >= 0.0;

                        // Make weak edges very thin and strong edges stand out.
                        // Nonlinear mapping gives more visual separation.
                        let thickness01 = strength01.powf(1.8).clamp(0.0, 1.0);

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
                if snap.game.trial_duration != 0 {
                    let pending = *pending_trial_ms_poll.borrow();
                    let apply = match pending {
                        Some((want, t))
                            if now.duration_since(t) < Duration::from_millis(800)
                                && snap.game.trial_duration != want =>
                        {
                            false
                        }
                        Some((want, _)) if snap.game.trial_duration == want => {
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
