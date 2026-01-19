#![allow(clippy::clone_on_copy)]

use braine::substrate::{Brain, CausalGraphViz, ExecutionTier, Stimulus, UnitPlotPoint};
use leptos::prelude::*;
use leptos::task::spawn_local;
use std::sync::Arc;
mod charts;
mod pong_web;
mod sequence_web;
mod text_web;
use charts::RollingHistory;
use text_web::TextWebGame;
mod brain_factory;
mod canvas;
mod files;
mod float_fmt;
mod indexeddb;
mod latex;
mod markdown;
mod math;
mod mermaid;
mod parameter_field;
mod runtime;
mod settings_schema;
mod shell;
mod storage;
mod tokens;
mod tooltip;
mod types;
use float_fmt::{fmt_f32_fixed, fmt_f32_signed_fixed};
use wasm_bindgen::closure::Closure;
use wasm_bindgen::JsCast;
use wasm_bindgen::JsValue;
use wasm_bindgen_futures::JsFuture;

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize, Default)]
struct BrainvizNodeTag {
    color: String, // "#RRGGBB"
    label: String,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize, Default)]
struct BrainvizSymbolTag {
    color: String, // "#RRGGBB"
    label: String,
}

#[derive(Clone, Debug)]
struct InspectTrialEvent {
    step: u64,
    game: GameKind,
    trial: u32,
    action: String,
    reward: f32,
    recent_rate: f32,
}

#[derive(Clone, Copy, Debug)]
struct BrainvizNodeDelta {
    idx: u16,
    /// bit0=amp01, bit1=phase, bit2=salience01
    mask: u8,
    amp01: f32,
    phase: f32,
    salience01: f32,
}

#[derive(Clone, Copy, Debug)]
struct BrainvizEdgeDelta {
    idx: u32,
    weight: f32,
}

#[derive(Clone, Debug)]
struct BrainvizKeyframe {
    step: u64,
    amp01: Vec<f32>,
    phase: Vec<f32>,
    salience01: Vec<f32>,
    edge_weights: Vec<f32>,
}

#[derive(Clone, Debug)]
struct BrainvizDeltaFrame {
    step: u64,
    node_deltas: Vec<BrainvizNodeDelta>,
    edge_deltas: Vec<BrainvizEdgeDelta>,
}

#[derive(Clone, Debug, Default)]
struct BrainvizGameRecording {
    n: usize,
    edges_per_node: usize,
    // Stable metadata (IDs, flags, rel_age). Dynamic fields are reconstructed per frame.
    base_points: Vec<UnitPlotPoint>,
    // Stable edge endpoints for delta encoding.
    edge_pairs: Vec<(usize, usize)>,
    // Current state (for computing deltas while recording).
    cur_amp01: Vec<f32>,
    cur_phase: Vec<f32>,
    cur_salience01: Vec<f32>,
    cur_edge_weights: Vec<f32>,
    // Delta stream + periodic keyframes for random access replay.
    deltas: Vec<BrainvizDeltaFrame>,
    keyframes: Vec<(usize, BrainvizKeyframe)>,
}

const GAME_KIND_COUNT: usize = 9;

fn game_kind_index(kind: crate::ui_model::GameKind) -> usize {
    match kind {
        crate::ui_model::GameKind::Spot => 0,
        crate::ui_model::GameKind::Bandit => 1,
        crate::ui_model::GameKind::SpotReversal => 2,
        crate::ui_model::GameKind::SpotXY => 3,
        crate::ui_model::GameKind::Maze => 4,
        crate::ui_model::GameKind::Pong => 5,
        crate::ui_model::GameKind::Sequence => 6,
        crate::ui_model::GameKind::Text => 7,
        crate::ui_model::GameKind::Replay => 8,
    }
}

fn wrap_delta_phase_to_pi(mut d: f32) -> f32 {
    // UnitPlotPoint.phase is in [0, 2π). For estimating velocity from deltas,
    // wrap to the minimal delta in (-π, π].
    let pi = std::f32::consts::PI;
    let tau = 2.0 * pi;
    while d > pi {
        d -= tau;
    }
    while d <= -pi {
        d += tau;
    }
    d
}

use crate::ui_model::{AnalyticsPanel, DashboardTab, GameKind};

use canvas::{clear_canvas, draw_maze, draw_pong, draw_spotxy};
use files::{download_bytes, read_file_bytes};
use indexeddb::{idb_get_bytes, idb_put_bytes, load_game_accuracies, save_game_accuracies};
use runtime::{AppRuntime, TickConfig, TickResult, WebGame};
use shell::{Sidebar, SystemErrorBanner, ToastStack, Topbar};
use storage::{
    apply_theme_to_document, clear_persisted_stats_state, load_persisted_settings,
    load_persisted_stats_state, local_storage_get_string, local_storage_remove,
    local_storage_set_string, parse_exec_tier_pref, save_persisted_settings,
    save_persisted_stats_state, PersistedGameStats, PersistedSettings, PersistedStatsState,
};
use tokens::{choose_text_token_sensor, token_action_name_from_sensor};
use types::{MazeUiState, PongUiState, ReplayUiState, SequenceUiState, TextUiState};

type ErrorSink = std::cell::RefCell<Option<Box<dyn Fn(String)>>>;

thread_local! {
    static ERROR_SINK: ErrorSink = const { std::cell::RefCell::new(None) };
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ToastLevel {
    Info,
    Success,
    Error,
}

#[derive(Clone, Debug)]
struct Toast {
    id: u64,
    level: ToastLevel,
    message: String,
}

fn report_error(msg: impl Into<String>) {
    let msg = msg.into();
    web_sys::console::error_1(&wasm_bindgen::JsValue::from_str(&msg));
    ERROR_SINK.with(|s| {
        if let Some(cb) = s.borrow().as_ref() {
            cb(msg);
        }
    });
}

use mermaid::{
    apply_theme as apply_mermaid_theme, render_all as render_all_mermaid, MermaidDiagram,
};
use parameter_field::ParameterField;
use settings_schema::ParamSection;
use tooltip::{TooltipPortal, TooltipStore};

const IDB_DB_NAME: &str = "braine";
const IDB_STORE: &str = "kv";
const IDB_KEY_BRAIN_IMAGE: &str = "brain_image";
const IDB_KEY_GAME_ACCURACY: &str = "game_accuracy";

const LOCALSTORAGE_THEME_KEY: &str = "braine_theme";
const LOCALSTORAGE_GAME_STATS_PREFIX: &str = "braine_game_stats_v1.";
const LOCALSTORAGE_SETTINGS_KEY: &str = "braine_settings_v1";
const LOCALSTORAGE_EXEC_TIER_KEY: &str = "braine_exec_tier_v1";
const LOCALSTORAGE_BRAINVIZ_NODE_TAGS_KEY: &str = "braine_brainviz_node_tags_v1";
const LOCALSTORAGE_BRAINVIZ_SYMBOL_TAGS_KEY: &str = "braine_brainviz_symbol_tags_v1";

fn parse_hex_rgb(s: &str) -> Option<(u8, u8, u8)> {
    let s = s.trim();
    let s = s.strip_prefix('#').unwrap_or(s);
    if s.len() != 6 {
        return None;
    }
    let r = u8::from_str_radix(&s[0..2], 16).ok()?;
    let g = u8::from_str_radix(&s[2..4], 16).ok()?;
    let b = u8::from_str_radix(&s[4..6], 16).ok()?;
    Some((r, g, b))
}

fn load_brainviz_node_tags() -> std::collections::HashMap<u32, BrainvizNodeTag> {
    let Some(raw) = local_storage_get_string(LOCALSTORAGE_BRAINVIZ_NODE_TAGS_KEY) else {
        return std::collections::HashMap::new();
    };
    serde_json::from_str::<std::collections::HashMap<u32, BrainvizNodeTag>>(&raw)
        .unwrap_or_else(|_| std::collections::HashMap::new())
}

fn save_brainviz_node_tags(tags: &std::collections::HashMap<u32, BrainvizNodeTag>) {
    if let Ok(raw) = serde_json::to_string(tags) {
        local_storage_set_string(LOCALSTORAGE_BRAINVIZ_NODE_TAGS_KEY, &raw);
    }
}

fn load_brainviz_symbol_tags() -> std::collections::HashMap<u32, BrainvizSymbolTag> {
    let Some(raw) = local_storage_get_string(LOCALSTORAGE_BRAINVIZ_SYMBOL_TAGS_KEY) else {
        return std::collections::HashMap::new();
    };
    serde_json::from_str::<std::collections::HashMap<u32, BrainvizSymbolTag>>(&raw)
        .unwrap_or_else(|_| std::collections::HashMap::new())
}

fn save_brainviz_symbol_tags(tags: &std::collections::HashMap<u32, BrainvizSymbolTag>) {
    if let Ok(raw) = serde_json::to_string(tags) {
        local_storage_set_string(LOCALSTORAGE_BRAINVIZ_SYMBOL_TAGS_KEY, &raw);
    }
}

const STYLE_CARD: &str = "padding: 14px; background: var(--panel); border: 1px solid var(--border); border-radius: 12px;";

// Version strings for display
const VERSION_BRAINE: &str = env!("CARGO_PKG_VERSION");
const VERSION_BRAINE_WEB: &str = "0.1.0";
const VERSION_BBI_FORMAT: u32 = 2;

// Long-form math spec (repo doc) embedded into the web UI.
const DOC_THE_MATH_BEHIND: &str = include_str!("../../../doc/maths/the-math-behind.md");

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
enum Theme {
    #[default]
    Dark,
    Light,
}

impl Theme {
    fn toggle(self) -> Self {
        match self {
            Theme::Dark => Theme::Light,
            Theme::Light => Theme::Dark,
        }
    }

    fn label(self) -> &'static str {
        match self {
            Theme::Dark => "Dark",
            Theme::Light => "Light",
        }
    }

    fn icon(self) -> &'static str {
        match self {
            Theme::Dark => "🌙",
            Theme::Light => "☀️",
        }
    }

    fn as_attr(self) -> &'static str {
        match self {
            Theme::Dark => "dark",
            Theme::Light => "light",
        }
    }

    fn from_attr(v: &str) -> Option<Self> {
        match v {
            "dark" => Some(Theme::Dark),
            "light" => Some(Theme::Light),
            _ => None,
        }
    }
}

/// Sub-tabs for the About page
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
enum AboutSubTab {
    #[default]
    Overview,
    Dynamics,
    Learning,
    Memory,
    MathBehind,
    Architecture,
    Applications,
    LlmIntegration,
    Apis,
}

impl AboutSubTab {
    fn label(self) -> &'static str {
        match self {
            AboutSubTab::Overview => "Overview",
            AboutSubTab::Dynamics => "Dynamics",
            AboutSubTab::Learning => "Learning",
            AboutSubTab::Memory => "Memory",
            AboutSubTab::MathBehind => "The Math Behind",
            AboutSubTab::Architecture => "Architecture",
            AboutSubTab::Applications => "Applications",
            AboutSubTab::LlmIntegration => "LLM Integration",
            AboutSubTab::Apis => "Braine APIs",
        }
    }

    fn all() -> &'static [AboutSubTab] {
        &[
            AboutSubTab::Overview,
            AboutSubTab::Dynamics,
            AboutSubTab::Learning,
            AboutSubTab::Memory,
            AboutSubTab::MathBehind,
            AboutSubTab::Architecture,
            AboutSubTab::Applications,
            AboutSubTab::LlmIntegration,
            AboutSubTab::Apis,
        ]
    }
}

const ABOUT_LLM_DATAFLOW: &str = r#"flowchart TD
    Env[Frame]
    Stim[Stimuli]
    Brain[Brain dynamics]
    Act[Action]
    Reward[Reward]
    Learn[Local learning]

    Env -->|encode| Stim
    Stim --> Brain
    Brain -->|readout| Act
    Act --> Env

    Env -->|reward| Reward
    Reward -->|gates| Learn
    Learn -->|updates| Brain

    Snap[Snapshot]
    Ctx[AdvisorContext]
    LLM[External LLM]
    Advice[AdvisorAdvice]
    Clamp[Daemon clamps]

    Brain -->|snapshot| Snap
    Snap --> Ctx
    Ctx --> LLM
    LLM --> Advice
    Advice --> Clamp
    Clamp -. nudges .-> Brain
"#;

const ABOUT_OVERVIEW_DIAGRAM: &str = r#"flowchart LR
    subgraph Frame[Frame / environment]
        Env[Environment]
        Stim[Stimuli symbols]
        Act[Action symbols]
    end

    subgraph Substrate[Brain substrate]
        Dyn[Wave dynamics]
        Plastic[Local plasticity]
    end

    Reward[Reward / neuromodulator]
    Persist["Persistence<br/>(brain image • BBI)"]
    Viz["Telemetry / viz<br/>(stats • BrainViz • analytics)"]

    Env -->|stimuli| Stim
    Stim --> Dyn
    Dyn -->|actions| Act
    Act --> Env

    Reward -. neuromodulates .-> Plastic
    Dyn --> Plastic --> Dyn

    Dyn -. save/load .-> Persist
    Dyn -. slow side-effects .-> Viz
"#;

const ABOUT_LEARNING_HIERARCHY_DIAGRAM: &str = r#"flowchart LR
    CB["Child brains<br/>(sec–hours)"] --> NG["Neurogenesis<br/>(steps–min)"] --> H["Hebbian<br/>(per-step)"] --> I["Imprinting<br/>(one-shot)"]
"#;

const ABOUT_MEMORY_STRUCTURE_DIAGRAM: &str = r#"flowchart LR
    Persisted[Persisted state]
    Runtime[Runtime state]

    Persisted --> P1[Units and sparse connections]
    Persisted --> P2[Sensor and action groups]
    Persisted --> P3[Symbol table]
    Persisted --> P4[Causal memory edges]

    Runtime --> R1[Pending input]
    Runtime --> R2[Telemetry buffers]
    Runtime --> R3[Transient vectors]
"#;

const ABOUT_CURRENT_ARCHITECTURE_DIAGRAM: &str = r#"flowchart LR
    subgraph Desktop[Desktop / local machine]
        UI[braine_desktop]
        CLI[braine-cli]
    end

    subgraph Daemon[brained daemon]
        BrainD[Brain authoritative]
        File[braine.bbi filesystem]
    end

    subgraph Web[Web WASM]
        BrainW[Brain in-tab]
        IDB[IndexedDB]
    end

    UI <--> Daemon
    CLI <--> Daemon
    BrainD --> File
    BrainW --> IDB
"#;

const ABOUT_FUTURE_ARCHITECTURE_DIAGRAM: &str = r#"flowchart TB
    Central["Central Daemon<br/>Authoritative Brain state<br/>Learning + Persistence"]

    Desktop["Desktop<br/>Local brain copy<br/>(sync: real-time)"]
    Web["Web (WASM)<br/>Local brain copy<br/>(sync: WebSocket)"]
    Mobile["Mobile/IoT<br/>Local brain copy<br/>(sync: MQTT/WS)"]
    Scripts["CLI / Scripts<br/>No local copy<br/>(direct commands)"]

    Desktop <-->|sync| Central
    Web <-->|sync| Central
    Mobile <-->|sync| Central
    Scripts -->|direct| Central
"#;

const ABOUT_LLM_ADVISOR_CONTEXT_REQ: &str =
    r#"{\n  \"type\": \"AdvisorContext\",\n  \"include_action_scores\": true\n}"#;

const ABOUT_LLM_ADVISOR_CONTEXT_RESP: &str = r#"{\n  \"type\": \"AdvisorContext\",\n  \"context\": {\n    \"context_key\": \"replay::spot_lr_small\",\n    \"game\": \"replay\",\n    \"trials\": 200,\n    \"recent_rate\": 0.52,\n    \"last_100_rate\": 0.51,\n    \"exploration_eps\": 0.20,\n    \"meaning_alpha\": 0.60,\n    \"notes\": [\"bounded; no action selection\"]\n  },\n  \"action_scores\": [\n    { \"name\": \"left\",  \"score\": 0.12, \"meaning\": 0.03, \"habit_norm\": 0.09 },\n    { \"name\": \"right\", \"score\": 0.10, \"meaning\": 0.02, \"habit_norm\": 0.08 }\n  ]\n}"#;

const ABOUT_LLM_ADVISOR_APPLY_REQ: &str = r#"{\n  \"type\": \"AdvisorApply\",\n  \"advice\": {\n    \"exploration_eps\": 0.12,\n    \"meaning_alpha\": 0.75,\n    \"ttl_trials\": 200,\n    \"rationale\": \"Increase meaning weight to sharpen context-conditioned discrimination; reduce exploration once stability is improving.\"\n  }\n}"#;

pub fn start() {
    // Convert Rust panics + JS "error" events into a UI-visible banner.
    // This avoids the "lots of console errors but blank UI" failure mode.
    std::panic::set_hook(Box::new(|info| {
        let msg = info.to_string();
        report_error(format!("panic: {msg}"));
    }));

    if let Some(window) = web_sys::window() {
        fn js_value_to_pretty_string(v: &wasm_bindgen::JsValue) -> String {
            if let Some(s) = v.as_string() {
                return s;
            }

            if let Ok(s) = js_sys::Reflect::get(v, &wasm_bindgen::JsValue::from_str("message")) {
                if let Some(s) = s.as_string() {
                    return s;
                }
            }

            if let Ok(s) = js_sys::JSON::stringify(v) {
                if let Some(s) = s.as_string() {
                    if s != "{}" {
                        return s;
                    }
                }
            }

            format!("{v:?}")
        }

        // window.onerror
        let on_error =
            Closure::<dyn FnMut(web_sys::ErrorEvent)>::new(move |e: web_sys::ErrorEvent| {
                let msg = e.message();
                if msg.trim().is_empty() {
                    report_error("window error".to_string());
                } else {
                    report_error(format!("window error: {msg}"));
                }
            });
        let _ = window.add_event_listener_with_callback("error", on_error.as_ref().unchecked_ref());
        on_error.forget();

        // window.onunhandledrejection
        let on_rejection = Closure::<dyn FnMut(web_sys::PromiseRejectionEvent)>::new(
            move |e: web_sys::PromiseRejectionEvent| {
                let reason = e.reason();
                let msg = js_value_to_pretty_string(&reason);

                // Trunk live-reload uses a websocket; embedded/locked-down browsers can fail it.
                // Don't surface that as a fatal app error.
                if msg.contains("/.well-known/trunk/ws") {
                    return;
                }

                report_error(format!("unhandled rejection: {msg}"));
            },
        );
        let _ = window.add_event_listener_with_callback(
            "unhandledrejection",
            on_rejection.as_ref().unchecked_ref(),
        );
        on_rejection.forget();
    }

    if let Some(el) = web_sys::window()
        .and_then(|w| w.document())
        .and_then(|d| d.get_element_by_id("app"))
    {
        if let Ok(html_el) = el.dyn_into::<web_sys::HtmlElement>() {
            mount_to(html_el, || view! { <App /> }).forget();
            return;
        }
    }

    mount_to_body(|| view! { <App /> });
}

#[component]
fn App() -> impl IntoView {
    let runtime = StoredValue::new(AppRuntime::new());

    let tooltip_store: TooltipStore = RwSignal::new(None);
    provide_context(tooltip_store);

    let (system_error, set_system_error) = signal::<Option<String>>(None);

    let toasts: RwSignal<Vec<Toast>> = RwSignal::new(Vec::new());
    let next_toast_id: StoredValue<u64> = StoredValue::new(1);
    let push_toast = {
        move |level: ToastLevel, message: String| {
            let id = {
                let mut id = 0u64;
                next_toast_id.update_value(|n| {
                    id = *n;
                    *n = (*n).saturating_add(1);
                });
                id
            };
            toasts.update(|ts| ts.push(Toast { id, level, message }));

            // Auto-dismiss after a few seconds (except errors, which stay until dismissed).
            if level != ToastLevel::Error {
                if let Some(win) = web_sys::window() {
                    let cb = Closure::wrap(Box::new(move || {
                        toasts.update(|ts| ts.retain(|t| t.id != id));
                    }) as Box<dyn FnMut()>);
                    let _ = win.set_timeout_with_callback_and_timeout_and_arguments_0(
                        cb.as_ref().unchecked_ref(),
                        3500,
                    );
                    cb.forget();
                }
            }
        }
    };

    // Register UI sink for global error reporter.
    ERROR_SINK.with(|s| {
        *s.borrow_mut() = Some(Box::new(move |msg| {
            let msg_toast = msg.clone();
            set_system_error.set(Some(msg));
            push_toast(ToastLevel::Error, msg_toast);
        }));
    });

    let theme0 = local_storage_get_string(LOCALSTORAGE_THEME_KEY)
        .as_deref()
        .map(str::trim)
        .and_then(Theme::from_attr)
        .unwrap_or(Theme::Dark);
    apply_theme_to_document(theme0);
    let (theme, set_theme) = signal(theme0);

    // Persist + apply theme to CSS variables.
    // (CSS consumes :root[data-theme='light'] / default dark.)
    Effect::new(move |_| {
        let t = theme.get();
        apply_theme_to_document(t);
        local_storage_set_string(LOCALSTORAGE_THEME_KEY, t.as_attr());
        apply_mermaid_theme(t.as_attr());
    });

    let (steps, set_steps) = signal(0u64);
    let (diag, set_diag) = signal(runtime.with_value(|r| r.brain.diagnostics()));

    // Load persisted runtime knobs (web-only).
    let settings0: PersistedSettings = load_persisted_settings().unwrap_or_default();

    let (game_kind, set_game_kind) = signal(GameKind::Spot);
    let (dashboard_tab, set_dashboard_tab) = signal(DashboardTab::BrainViz);
    // Mobile responsive: dashboard drawer (hidden by default on small screens)
    let (dashboard_open, set_dashboard_open) = signal(false);
    // Mobile responsive: left sidebar (hidden by default on small screens)
    let (sidebar_open, set_sidebar_open) = signal(false);
    let (spotxy_pos, set_spotxy_pos) = signal::<Option<(f32, f32)>>(None);
    let (_spotxy_stimulus_key, set_spotxy_stimulus_key) = signal(String::new());
    let (spotxy_eval, set_spotxy_eval) = signal(false);
    let (spotxy_mode, set_spotxy_mode) = signal(String::new());
    let (spotxy_grid_n, set_spotxy_grid_n) = signal(0u32);

    let (maze_state, set_maze_state) = signal::<Option<MazeUiState>>(None);

    let (pong_state, set_pong_state) = signal::<Option<PongUiState>>(None);
    let (_pong_stimulus_key, set_pong_stimulus_key) = signal(String::new());
    let (pong_paddle_speed, set_pong_paddle_speed) = signal(0.0f32);
    let (pong_paddle_half_height, set_pong_paddle_half_height) = signal(0.0f32);
    let (pong_ball_speed, set_pong_ball_speed) = signal(0.0f32);
    let (pong_paddle_bounce_y, set_pong_paddle_bounce_y) = signal(0.0f32);
    let (pong_respawn_delay_s, set_pong_respawn_delay_s) = signal(0.0f32);
    let (pong_distractor_enabled, set_pong_distractor_enabled) = signal(false);
    let (_pong_distractor_speed_scale, set_pong_distractor_speed_scale) = signal(0.0f32);

    let (sequence_state, set_sequence_state) = signal::<Option<SequenceUiState>>(None);
    let (text_state, set_text_state) = signal::<Option<TextUiState>>(None);
    let (replay_state, set_replay_state) = signal::<Option<ReplayUiState>>(None);

    // Bandit tuning (editable live in-game).
    let (bandit_prob_left, set_bandit_prob_left) = signal(0.8f32);
    let (bandit_prob_right, set_bandit_prob_right) = signal(0.2f32);

    // Spot/SpotReversal stimulus (left/right) for UI highlighting.
    let (spot_is_left, set_spot_is_left) = signal::<Option<bool>>(None);

    // Status line for the header.
    let (status, set_status) = signal(String::from("ready"));

    // True while we're waiting for a nonblocking GPU step to complete.
    // (Useful for debugging perceived UI stalls.)
    let (gpu_step_pending, set_gpu_step_pending) = signal(false);

    // Core learning/runtime controls.
    let (learning_enabled, set_learning_enabled) = signal(settings0.learning_enabled);
    let (trial_period_ms, set_trial_period_ms) = signal::<u32>(settings0.trial_period_ms);
    let (run_interval_ms, set_run_interval_ms) = signal::<u32>(settings0.run_interval_ms);
    let (reward_scale, set_reward_scale) = signal::<f32>(settings0.reward_scale);
    let (reward_bias, set_reward_bias) = signal::<f32>(settings0.reward_bias);
    let (exploration_eps, set_exploration_eps) = signal::<f32>(0.10);
    let (meaning_alpha, set_meaning_alpha) = signal::<f32>(8.0);

    // Persist settings when they change.
    // (We keep this narrow: only the core knobs in PersistedSettings.)
    Effect::new(move |_| {
        let s = PersistedSettings {
            reward_scale: reward_scale.get(),
            reward_bias: reward_bias.get(),
            learning_enabled: learning_enabled.get(),
            run_interval_ms: run_interval_ms.get(),
            trial_period_ms: trial_period_ms.get(),
            settings_advanced: false,
        };
        save_persisted_settings(&s);
    });

    // Stats signals used across dashboard pages.
    let (trials, set_trials) = signal::<u32>(0);
    let (recent_rate, set_recent_rate) = signal::<f32>(0.0);
    let (last_action, set_last_action) = signal(String::new());
    let (last_reward, set_last_reward) = signal::<f32>(0.0);

    // Analytics panel subtab.
    let (analytics_panel, set_analytics_panel) = signal(AnalyticsPanel::Reward);

    // In-game top dashboard state (replaces right-panel Stats/Analytics tabs).
    let (game_stats_open, set_game_stats_open) = signal(false);
    let (analytics_modal_open, set_analytics_modal_open) = signal(false);

    // Analytics floating window (small, draggable/resizable).
    let (analytics_win_x, set_analytics_win_x) = signal(24.0f64);
    let (analytics_win_y, set_analytics_win_y) = signal(92.0f64);
    let (analytics_drag, set_analytics_drag) = signal::<Option<(i32, f64, f64, f64, f64)>>(None);

    // Defer closing so we don't unmount the window mid-event dispatch.
    let close_analytics_window = {
        let set_analytics_modal_open = set_analytics_modal_open.clone();
        move || {
            if let Some(win) = web_sys::window() {
                let cb = Closure::wrap(Box::new(move || {
                    set_analytics_modal_open.set(false);
                }) as Box<dyn FnMut()>);
                let _ = win.set_timeout_with_callback_and_timeout_and_arguments_0(
                    cb.as_ref().unchecked_ref(),
                    0,
                );
                cb.forget();
            } else {
                set_analytics_modal_open.set(false);
            }
        }
    };

    // Settings UI state.
    let (settings_advanced, set_settings_advanced) = signal(false);
    let settings_specs = StoredValue::new(settings_schema::param_specs());
    let settings_validity_map: RwSignal<std::collections::HashMap<String, bool>> =
        RwSignal::new(std::collections::HashMap::new());
    let (settings_apply_disabled, set_settings_apply_disabled) = signal(false);

    // "Applied" toast / badge state.
    let (config_applied, set_config_applied) = signal(false);

    // Neurogenesis control.
    let (grow_units_n, set_grow_units_n) = signal::<u32>(32);

    // BrainConfig controls (initialized from the live runtime config).
    let cfg0 = runtime.with_value(|r| r.brain.config().clone());
    let (cfg_dt, set_cfg_dt) = signal::<f32>(cfg0.dt);
    let (cfg_base_freq, set_cfg_base_freq) = signal::<f32>(cfg0.base_freq);
    let (cfg_noise_amp, set_cfg_noise_amp) = signal::<f32>(cfg0.noise_amp);
    let (cfg_noise_phase, set_cfg_noise_phase) = signal::<f32>(cfg0.noise_phase);
    let (cfg_global_inhibition, set_cfg_global_inhibition) = signal::<f32>(cfg0.global_inhibition);
    let (cfg_hebb_rate, set_cfg_hebb_rate) = signal::<f32>(cfg0.hebb_rate);
    let (cfg_forget_rate, set_cfg_forget_rate) = signal::<f32>(cfg0.forget_rate);
    let (cfg_prune_below, set_cfg_prune_below) = signal::<f32>(cfg0.prune_below);
    let (cfg_coactive_threshold, set_cfg_coactive_threshold) =
        signal::<f32>(cfg0.coactive_threshold);
    let (cfg_phase_lock_threshold, set_cfg_phase_lock_threshold) =
        signal::<f32>(cfg0.phase_lock_threshold);
    let (cfg_imprint_rate, set_cfg_imprint_rate) = signal::<f32>(cfg0.imprint_rate);
    let (cfg_salience_decay, set_cfg_salience_decay) = signal::<f32>(cfg0.salience_decay);
    let (cfg_salience_gain, set_cfg_salience_gain) = signal::<f32>(cfg0.salience_gain);
    let (cfg_causal_decay, set_cfg_causal_decay) = signal::<f32>(cfg0.causal_decay);

    // Derived "apply" disabled state (any invalid param disables apply).
    Effect::new(move |_| {
        let m = settings_validity_map.get();
        let disabled = m.values().any(|ok| !*ok);
        set_settings_apply_disabled.set(disabled);
    });

    let apply_brain_config = {
        let runtime = runtime.clone();
        move || {
            let dt = cfg_dt.get_untracked();
            let base_freq = cfg_base_freq.get_untracked();
            let noise_amp = cfg_noise_amp.get_untracked();
            let noise_phase = cfg_noise_phase.get_untracked();
            let global_inhibition = cfg_global_inhibition.get_untracked();
            let hebb_rate = cfg_hebb_rate.get_untracked();
            let forget_rate = cfg_forget_rate.get_untracked();
            let prune_below = cfg_prune_below.get_untracked();
            let coactive_threshold = cfg_coactive_threshold.get_untracked();
            let phase_lock_threshold = cfg_phase_lock_threshold.get_untracked();
            let imprint_rate = cfg_imprint_rate.get_untracked();
            let salience_decay = cfg_salience_decay.get_untracked();
            let salience_gain = cfg_salience_gain.get_untracked();
            let causal_decay = cfg_causal_decay.get_untracked();

            let mut err: Option<String> = None;
            runtime.update_value(|r| {
                if let Err(e) = r.brain.update_config(|cfg| {
                    cfg.dt = dt;
                    cfg.base_freq = base_freq;
                    cfg.noise_amp = noise_amp;
                    cfg.noise_phase = noise_phase;
                    cfg.global_inhibition = global_inhibition;
                    cfg.hebb_rate = hebb_rate;
                    cfg.forget_rate = forget_rate;
                    cfg.prune_below = prune_below;
                    cfg.coactive_threshold = coactive_threshold;
                    cfg.phase_lock_threshold = phase_lock_threshold;
                    cfg.imprint_rate = imprint_rate;
                    cfg.salience_decay = salience_decay;
                    cfg.salience_gain = salience_gain;
                    cfg.causal_decay = causal_decay;
                }) {
                    err = Some(e.to_string());
                }
            });

            if let Some(e) = err {
                set_status.set(format!("Config error: {e}"));
                push_toast(ToastLevel::Error, format!("Config error: {e}"));
            } else {
                set_status.set("Braine config applied".to_string());

                push_toast(ToastLevel::Success, "Settings applied".to_string());

                set_config_applied.set(true);
                if let Some(win) = web_sys::window() {
                    let cb = Closure::wrap(Box::new(move || {
                        set_config_applied.set(false);
                    }) as Box<dyn FnMut()>);
                    let _ = win.set_timeout_with_callback_and_timeout_and_arguments_0(
                        cb.as_ref().unchecked_ref(),
                        1500,
                    );
                    cb.forget();
                }
            }
        }
    };

    let reset_brain_config_from_runtime = {
        let runtime = runtime.clone();
        move || {
            runtime.with_value(|r| {
                let cfg = r.brain.config();
                set_cfg_dt.set(cfg.dt);
                set_cfg_base_freq.set(cfg.base_freq);
                set_cfg_noise_amp.set(cfg.noise_amp);
                set_cfg_noise_phase.set(cfg.noise_phase);
                set_cfg_global_inhibition.set(cfg.global_inhibition);
                set_cfg_hebb_rate.set(cfg.hebb_rate);
                set_cfg_forget_rate.set(cfg.forget_rate);
                set_cfg_prune_below.set(cfg.prune_below);
                set_cfg_coactive_threshold.set(cfg.coactive_threshold);
                set_cfg_phase_lock_threshold.set(cfg.phase_lock_threshold);
                set_cfg_imprint_rate.set(cfg.imprint_rate);
                set_cfg_salience_decay.set(cfg.salience_decay);
                set_cfg_salience_gain.set(cfg.salience_gain);
                set_cfg_causal_decay.set(cfg.causal_decay);
            });
            set_status.set("Config reset to current".to_string());
            push_toast(ToastLevel::Info, "Settings reset".to_string());
        }
    };

    // Text prediction "lab" (inference-only on a cloned brain) - currently unused but kept for future
    #[allow(unused_variables)]
    let (_text_prompt, _set_text_prompt) = signal(String::from("hello worl"));
    #[allow(unused_variables)]
    let (_text_prompt_regime, _set_text_prompt_regime) = signal(0u32);
    #[allow(unused_variables)]
    let (_text_temp, _set_text_temp) = signal(1.0f32);
    #[allow(unused_variables)]
    let (_text_preds, _set_text_preds) = signal::<Vec<(String, f32, f32)>>(Vec::new());

    // Text training controls (writes to the live brain)
    let (text_corpus0, set_text_corpus0) = signal(String::from("hello world\n"));
    let (text_corpus1, set_text_corpus1) = signal(String::from("goodbye world\n"));
    let (text_max_vocab, set_text_max_vocab) = signal(32u32);
    let (text_shift_every, set_text_shift_every) = signal(80u32);

    let (text_train_prompt, set_text_train_prompt) = signal(String::from("hello world\n"));
    let (text_train_regime, set_text_train_regime) = signal(0u32);
    let (text_train_epochs, _set_text_train_epochs) = signal(1u32);

    let (reversal_active, set_reversal_active) = signal(false);
    let (reversal_flip_after, set_reversal_flip_after) = signal(0u32);

    let (import_autosave, set_import_autosave) = signal(true);

    // IndexedDB persistence UX
    let (idb_autosave, set_idb_autosave) = signal(true);
    let (idb_loaded, set_idb_loaded) = signal(false);
    let (idb_last_save, set_idb_last_save) = signal(String::new());
    let (brain_dirty, set_brain_dirty) = signal(false);

    let (interval_id, set_interval_id) = signal::<Option<i32>>(None);

    let (is_running, set_is_running) = signal(false);

    // Performance history for charting
    let perf_history = StoredValue::new(RollingHistory::new(200));
    let (perf_version, set_perf_version) = signal(0u32);

    // Neuromodulator (reward) history
    let neuromod_history = StoredValue::new(RollingHistory::new(50));
    let (neuromod_version, set_neuromod_version) = signal(0u32);

    // Learning monitor history (cheap; sampled on the same heavy cadence as diagnostics).
    let (learn_stats, set_learn_stats) = signal(runtime.with_value(|r| r.brain.learning_stats()));
    let learn_elig_history = StoredValue::new(RollingHistory::new(120));
    let learn_plasticity_history = StoredValue::new(RollingHistory::new(120));
    let learn_homeostasis_history = StoredValue::new(RollingHistory::new(120));
    let (learn_version, set_learn_version) = signal(0u32);

    // Action choice history (for choices-over-time chart)
    let choice_events = StoredValue::new(Vec::<String>::new());
    let (choice_window, set_choice_window) = signal(30u32);
    let (choices_version, set_choices_version) = signal(0u32);

    // Persist per-game stats to localStorage (one write per trial per game)
    let persisted_trial_cursor = StoredValue::new((game_kind.get_untracked(), 0u32));

    // Learning milestone tracking
    let (learning_milestone, set_learning_milestone) = signal("Starting...".to_string());
    let (learning_milestone_tone, set_learning_milestone_tone) = signal("starting".to_string());
    let (learned_at_trial, set_learned_at_trial) = signal::<Option<u32>>(None);
    let (mastered_at_trial, set_mastered_at_trial) = signal::<Option<u32>>(None);

    // Correct/incorrect counts
    let (correct_count, set_correct_count) = signal(0u32);
    let (incorrect_count, set_incorrect_count) = signal(0u32);

    // Unit plot data for Graph page
    let (unit_plot, set_unit_plot) = signal::<Vec<UnitPlotPoint>>(Vec::new());

    // Brain age (steps) for stats display
    let (brain_age, set_brain_age) = signal(0u64);
    // Causal memory stats for stats display
    let (causal_symbols, set_causal_symbols) = signal(0usize);
    let (causal_edges, set_causal_edges) = signal(0usize);

    // Left panel mode: show About or show Game
    let (show_about_page, set_show_about_page) = signal(true);
    // About page sub-tab
    let (about_sub_tab, set_about_sub_tab) = signal(AboutSubTab::Overview);

    // Render once; the source is a compile-time embedded repo doc.
    // Use StoredValue so the view closures remain `Fn` (not `FnOnce`).
    let math_behind_html =
        StoredValue::new(markdown::render_markdown_with_mermaid(DOC_THE_MATH_BEHIND));

    // When switching to the Math tab, ask Mermaid to render any fenced diagrams.
    Effect::new(move |_| {
        if about_sub_tab.get() == AboutSubTab::MathBehind {
            spawn_local(async move {
                // Microtask: ensure the DOM has inserted the `.mermaid` nodes.
                let _ = JsFuture::from(js_sys::Promise::resolve(&JsValue::NULL)).await;
                render_all_mermaid();
                latex::render_all();
            });
        }
    });

    // BrainViz uses its own sampling so it can be tuned independently.
    // Snapshot mode: sampling happens only on explicit refresh.
    let (brainviz_points, set_brainviz_points) = signal::<Vec<UnitPlotPoint>>(Vec::new());
    let (brainviz_edges, set_brainviz_edges) = signal::<Vec<(usize, usize, f32)>>(Vec::new());
    let brainviz_base_positions = StoredValue::new(Vec::<(f64, f64, f64)>::new());
    let brainviz_base_positions_n = StoredValue::new(0usize);
    let (brainviz_node_sample, set_brainviz_node_sample) = signal(128u32);
    let (brainviz_edges_per_node, set_brainviz_edges_per_node) = signal(4u32);
    let (brainviz_snapshot_refresh, set_brainviz_snapshot_refresh) = signal(1u32);
    let brainviz_last_snapshot_refresh = StoredValue::new(0u32);

    // BrainViz recording/replay:
    // - While running: BrainViz rendering/sampling is paused (to protect game FPS)
    // - We record per-trial sampled point+edge snapshots per game
    // - When stopped: user can replay those snapshots on demand
    let (brainviz_record_enabled, set_brainviz_record_enabled) = signal(true);
    // Delta encoding knobs (UI-tunable):
    // - classify "active" nodes to capture fine phase motion without recording everything
    // - larger epsilons reduce recorded delta volume and CPU work
    let (brainviz_active_amp_threshold, set_brainviz_active_amp_threshold) = signal(0.5f32);
    let (brainviz_eps_phase_active, set_brainviz_eps_phase_active) = signal(0.02f32);
    let (brainviz_eps_phase_inactive, set_brainviz_eps_phase_inactive) = signal(0.08f32);
    let (brainviz_eps_amp01, _set_brainviz_eps_amp01) = signal(0.01f32);
    let (brainviz_eps_salience01, _set_brainviz_eps_salience01) = signal(0.01f32);
    let (brainviz_record_edges, set_brainviz_record_edges) = signal(true);
    let (brainviz_eps_weight, set_brainviz_eps_weight) = signal(0.005f32);
    let (brainviz_record_every_trials, set_brainviz_record_every_trials) = signal(1u32);
    let (brainviz_record_edges_every_trials, set_brainviz_record_edges_every_trials) = signal(2u32);
    let brainviz_recordings: StoredValue<Vec<BrainvizGameRecording>> =
        StoredValue::new(vec![BrainvizGameRecording::default(); GAME_KIND_COUNT]);
    let (brainviz_replay_active, set_brainviz_replay_active) = signal(false);
    let (brainviz_replay_kind, set_brainviz_replay_kind) = signal(game_kind.get_untracked());
    let (brainviz_replay_idx, set_brainviz_replay_idx) = signal::<usize>(0);
    let (brainviz_is_expanded, set_brainviz_is_expanded) = signal(false);
    let (brainviz_zoom, set_brainviz_zoom) = signal(1.5f32);
    let (brainviz_pan_x, set_brainviz_pan_x) = signal(0.0f32);
    let (brainviz_pan_y, set_brainviz_pan_y) = signal(0.0f32);
    let (_brainviz_auto_rotate, _set_brainviz_auto_rotate) = signal(false); // Disabled by default
    let (brainviz_manual_rotation, set_brainviz_manual_rotation) = signal(0.0f32); // Y-axis rotation (horizontal drag)
    let (brainviz_rotation_x, set_brainviz_rotation_x) = signal(0.0f32); // X-axis rotation (vertical drag)
    let (brainviz_vibration, _set_brainviz_vibration) = signal(0.0f32); // Activity-based vibration
    let (brainviz_idle_time, set_brainviz_idle_time) = signal(0.0f32); // Idle animation time (dreaming mode)
    let (brainviz_view_mode, set_brainviz_view_mode) = signal::<&'static str>("substrate"); // "substrate" or "causal"
    let (brainviz_causal_graph, set_brainviz_causal_graph) =
        signal::<CausalGraphViz>(CausalGraphViz::default());
    let (brainviz_causal_filter_prefix, set_brainviz_causal_filter_prefix) =
        signal::<String>(String::new());
    let (brainviz_causal_focus_selected, set_brainviz_causal_focus_selected) = signal(false);
    let (brainviz_causal_hide_isolates, set_brainviz_causal_hide_isolates) = signal(true);
    let (brainviz_display_nodes, set_brainviz_display_nodes) = signal::<usize>(0);
    let (brainviz_display_edges, set_brainviz_display_edges) = signal::<usize>(0);

    // BrainViz interactivity state (purely view transforms; does not trigger resampling).
    // Tuple: (pointer_id, start_x, start_y, start_pan_x, start_pan_y, start_rot_y, start_rot_x, pan_mode)
    let (brainviz_drag, set_brainviz_drag) =
        signal::<Option<(i32, f64, f64, f32, f32, f32, f32, bool)>>(None);

    // Research disclaimer: shown on every load/refresh (not persisted).
    let (show_research_disclaimer, set_show_research_disclaimer) = signal(true);

    // Game information modal (opened from the left menu).
    let (game_info_modal_kind, set_game_info_modal_kind) = signal::<Option<GameKind>>(None);

    // Defer closing game info modal so we don't unmount the modal mid-event dispatch.
    // This avoids "closure invoked recursively or after being dropped" errors.
    let close_game_info_modal = {
        let set_game_info_modal_kind = set_game_info_modal_kind.clone();
        move || {
            if let Some(win) = web_sys::window() {
                let cb = Closure::wrap(Box::new(move || {
                    set_game_info_modal_kind.set(None);
                }) as Box<dyn FnMut()>);
                let _ = win.set_timeout_with_callback_and_timeout_and_arguments_0(
                    cb.as_ref().unchecked_ref(),
                    0,
                );
                cb.forget();
            } else {
                set_game_info_modal_kind.set(None);
            }
        }
    };

    let (brainviz_display_avg_conn, set_brainviz_display_avg_conn) = signal::<f32>(0.0);
    let (brainviz_display_max_conn, set_brainviz_display_max_conn) = signal::<usize>(0);

    // BrainViz node tagging (manual highlighting):
    // - click a node to select it
    // - assign a color/label (persisted to localStorage)
    let brainviz_hit_nodes = StoredValue::new(Vec::<charts::BrainVizHitNode>::new());
    let brainviz_causal_hit_nodes = StoredValue::new(Vec::<charts::CausalHitNode>::new());
    let (brainviz_selected_node_id, set_brainviz_selected_node_id) = signal::<Option<u32>>(None);
    let (brainviz_selected_tag_color, set_brainviz_selected_tag_color) =
        signal::<String>("#ff4d4d".to_string());
    let (brainviz_selected_tag_label, set_brainviz_selected_tag_label) =
        signal::<String>("".to_string());
    let (brainviz_node_tags, set_brainviz_node_tags) =
        signal::<std::collections::HashMap<u32, BrainvizNodeTag>>(std::collections::HashMap::new());

    // Causal symbol tagging (manual highlighting in causal graph view):
    let (brainviz_selected_symbol_id, set_brainviz_selected_symbol_id) =
        signal::<Option<u32>>(None);
    let (brainviz_selected_symbol_name, set_brainviz_selected_symbol_name) =
        signal::<String>(String::new());
    let (brainviz_selected_symbol_color, set_brainviz_selected_symbol_color) =
        signal::<String>("#22d3ee".to_string());
    let (brainviz_selected_symbol_label, set_brainviz_selected_symbol_label) =
        signal::<String>(String::new());
    let (brainviz_symbol_tags, set_brainviz_symbol_tags) = signal::<
        std::collections::HashMap<u32, BrainvizSymbolTag>,
    >(std::collections::HashMap::new());

    // Load persisted tags once.
    {
        let did_load = StoredValue::new(false);
        Effect::new(move |_| {
            if did_load.get_value() {
                return;
            }
            did_load.set_value(true);
            set_brainviz_node_tags.set(load_brainviz_node_tags());
            set_brainviz_symbol_tags.set(load_brainviz_symbol_tags());
        });
    }

    // Idle state tracking for actual dreaming/sync operations
    let (idle_sync_done, set_idle_sync_done) = signal(false); // Has sync been performed this idle period?
    let (idle_dream_counter, set_idle_dream_counter) = signal(0u32); // Counts idle ticks for dream scheduling
    #[allow(unused_variables)]
    let (idle_status, set_idle_status) = signal("".to_string()); // Status message for idle operations

    // Idle animation and maintenance timer
    // - Increments idle time for visual animation
    // - Runs ACTUAL dreaming/sync on the brain when not running
    // - Sync runs once when entering idle, dreaming runs periodically
    {
        let runtime = runtime.clone();
        let set_idle = set_brainviz_idle_time;
        if let Some(window) = web_sys::window() {
            let cb = Closure::wrap(Box::new(move || {
                // Visual animation: always increment idle time
                set_idle.update(|t| *t += 0.033);

                // Check if brain is running (game loop active)
                let running = is_running.get_untracked();
                let learning = learning_enabled.get_untracked();

                if running {
                    // Brain is running: reset idle state
                    set_idle_sync_done.set(false);
                    set_idle_dream_counter.set(0);
                    set_idle_status.set("".to_string());
                } else {
                    // Brain is idle: perform actual maintenance operations

                    // 1. One-time sync when entering idle (only if not in learning mode)
                    if !idle_sync_done.get_untracked() {
                        let mut synced: Option<usize> = None;
                        runtime.update_value(|r| {
                            // Only sync if not in high-learning state.
                            if !r.brain.is_learning_mode() {
                                synced = Some(r.brain.global_sync());
                            }
                        });
                        set_idle_sync_done.set(true);
                        if synced.is_some() {
                            set_idle_status.set("sync: phases aligned".to_string());
                            set_brain_dirty.set(true);
                        }
                    }

                    // 2. Periodic dreaming while idle (every ~3 seconds, 90 ticks)
                    set_idle_dream_counter.update(|c| *c += 1);
                    let counter = idle_dream_counter.get_untracked();

                    if counter >= 90 && !learning {
                        // Run idle maintenance (micro-dream on inactive clusters)
                        runtime.update_value(|r| {
                            if let Some(processed) = r.brain.idle_maintenance(false) {
                                set_idle_status
                                    .set(format!("dreaming: {} units consolidated", processed));
                                set_brain_dirty.set(true);
                            }
                        });
                        set_idle_dream_counter.set(0);
                    }
                }
            }) as Box<dyn FnMut()>);
            let _ = window.set_interval_with_callback_and_timeout_and_arguments_0(
                cb.as_ref().unchecked_ref(),
                33, // ~30fps
            );
            cb.forget(); // Leak the closure to keep it alive
        }
    }

    // Per-game accuracy persistence (loaded from IDB)
    let (game_accuracies, set_game_accuracies) =
        signal::<std::collections::HashMap<String, f32>>(std::collections::HashMap::new());

    // IndexedDB autosave loop (best-effort).
    // Saves only when the brain has changed (brain_dirty), to avoid excessive writes.
    {
        let runtime = runtime.clone();
        let set_status = set_status.clone();
        if let Some(window) = web_sys::window() {
            let cb = Closure::wrap(Box::new(move || {
                if !idb_autosave.get_untracked() {
                    return;
                }
                if !brain_dirty.get_untracked() {
                    return;
                }

                let bytes = match runtime.with_value(|r| r.brain.save_image_bytes()) {
                    Ok(b) => b,
                    Err(e) => {
                        set_status.set(format!("autosave failed: {e}"));
                        return;
                    }
                };
                let accs = game_accuracies.get_untracked();

                let set_status = set_status.clone();
                spawn_local(async move {
                    match idb_put_bytes(IDB_KEY_BRAIN_IMAGE, &bytes).await {
                        Ok(()) => {
                            let _ = save_game_accuracies(&accs).await;
                            set_brain_dirty.set(false);
                            set_idb_loaded.set(true);
                            let ts = js_sys::Date::new_0()
                                .to_iso_string()
                                .as_string()
                                .unwrap_or_default();
                            set_idb_last_save.set(ts);
                        }
                        Err(e) => {
                            set_status.set(format!("autosave failed: {e}"));
                        }
                    }
                });
            }) as Box<dyn FnMut()>);
            let _ = window.set_interval_with_callback_and_timeout_and_arguments_0(
                cb.as_ref().unchecked_ref(),
                5_000,
            );
            cb.forget();
        }
    }

    // WebGPU availability check.
    // If this build includes the core `gpu` feature, we can select the GPU execution tier
    // when WebGPU is present. Note: today the GPU tier accelerates the dense dynamics
    // update inside `Brain::step()`, while learning/plasticity updates remain CPU.
    let webgpu_available = {
        web_sys::window()
            .and_then(|w| {
                let nav = wasm_bindgen::JsValue::from(w.navigator());
                js_sys::Reflect::get(&nav, &wasm_bindgen::JsValue::from_str("gpu"))
                    .ok()
                    .map(|v| !v.is_undefined())
            })
            .unwrap_or(false)
    };

    // Execution tier preference (persisted): if the user chose CPU/GPU, we should
    // honor it on reload and avoid auto-switching tiers.
    let exec_tier_pref_raw = local_storage_get_string(LOCALSTORAGE_EXEC_TIER_KEY);
    let exec_tier_pref = exec_tier_pref_raw
        .as_deref()
        .and_then(parse_exec_tier_pref)
        .inspect(|t| {
            runtime.update_value(|r| r.brain.set_execution_tier(*t));
        });
    // If an invalid value was stored, clear it so we can fall back to defaults.
    if exec_tier_pref_raw.is_some() && exec_tier_pref.is_none() {
        local_storage_remove(LOCALSTORAGE_EXEC_TIER_KEY);
    }

    // WebGPU support on wasm32:
    // The core GPU backend must be initialized asynchronously (no blocking waits),
    // then the brain can be switched to `ExecutionTier::Gpu`.

    let (gpu_status, set_gpu_status) = signal(if webgpu_available {
        if cfg!(feature = "gpu") {
            match exec_tier_pref {
                Some(ExecutionTier::Scalar) => "WebGPU: detected (CPU selected)",
                Some(ExecutionTier::Gpu) => "WebGPU: detected (initializing…)",
                _ => "WebGPU: detected (initializing…)",
            }
        } else {
            "WebGPU: detected (CPU build; enable the web `gpu` feature)"
        }
    } else {
        "WebGPU: not available (CPU mode)"
    });

    let (exec_tier_selected, set_exec_tier_selected) =
        signal(runtime.with_value(|r| r.brain.execution_tier()));
    let (exec_tier_effective, set_exec_tier_effective) =
        signal(runtime.with_value(|r| r.brain.effective_execution_tier()));

    // UI refresh throttles:
    // - Heavy brain sampling (diagnostics/causal stats/unit plot) is expensive on wasm.
    // - localStorage writes are synchronous and can cause severe jank if done per-trial.
    let ui_last_heavy_refresh_ms = StoredValue::new(0.0f64);
    let ui_last_persist_ms = StoredValue::new(0.0f64);

    // If WebGPU is present and this build has the web `gpu` feature, initialize
    // the GPU context asynchronously. By default we auto-enable GPU on first load,
    // but if the user explicitly selected CPU we do not auto-switch.
    #[cfg(feature = "gpu")]
    {
        let should_try_enable_gpu =
            webgpu_available && exec_tier_pref != Some(ExecutionTier::Scalar);
        if should_try_enable_gpu {
            let runtime = runtime.clone();
            let explicit_gpu_pref = exec_tier_pref == Some(ExecutionTier::Gpu);
            spawn_local(async move {
                set_gpu_status.set("WebGPU: initializing…");
                match braine::gpu::init_gpu_context(65_536).await {
                    Ok(()) => {
                        runtime.update_value(|r| {
                            r.brain.set_execution_tier(ExecutionTier::Gpu);
                        });
                        let eff = runtime.with_value(|r| r.brain.effective_execution_tier());
                        if eff == ExecutionTier::Gpu {
                            set_gpu_status.set("WebGPU: enabled (GPU dynamics tier)");
                            push_toast(ToastLevel::Success, "WebGPU enabled".to_string());
                        } else {
                            set_gpu_status.set("WebGPU: ready (CPU fallback)");
                            push_toast(
                                ToastLevel::Info,
                                "WebGPU detected, but using CPU tier".to_string(),
                            );
                        }
                    }
                    Err(e) => {
                        if explicit_gpu_pref {
                            // Keep the user's preference as "GPU" selected; effective tier will fall back.
                            set_gpu_status.set("WebGPU: init failed (CPU fallback)");
                        } else {
                            runtime.update_value(|r| {
                                r.brain.set_execution_tier(ExecutionTier::Scalar)
                            });
                            set_gpu_status.set("WebGPU: init failed (CPU mode)");
                        }
                        push_toast(ToastLevel::Error, format!("WebGPU init failed: {e}"));
                    }
                }
            });
        }
    }

    let refresh_ui_from_runtime = {
        let runtime = runtime.clone();
        move || {
            let now_ms = js_sys::Date::now();
            let running = is_running.get_untracked();

            set_exec_tier_selected.set(runtime.with_value(|r| r.brain.execution_tier()));
            set_exec_tier_effective.set(runtime.with_value(|r| r.brain.effective_execution_tier()));
            let stats = runtime.with_value(|r| r.game.stats().clone());
            set_trials.set(stats.trials);
            let rate = stats.last_100_rate();
            set_recent_rate.set(rate);
            set_correct_count.set(stats.correct);
            set_incorrect_count.set(stats.incorrect);
            set_learned_at_trial.set(stats.learned_at_trial);
            set_mastered_at_trial.set(stats.mastered_at_trial);

            // Update game accuracy in memory (will be persisted on game switch or save)
            let game_label = game_kind.get_untracked().label().to_string();
            set_game_accuracies.update(|accs| {
                accs.insert(game_label, rate);
            });

            // Update performance history
            perf_history.update_value(|h| h.push(rate));
            set_perf_version.update(|v| *v = v.wrapping_add(1));

            // Update learning milestones
            if rate >= 0.95 {
                set_learning_milestone.set("🏆 MASTERED".to_string());
                set_learning_milestone_tone.set("mastered".to_string());
            } else if rate >= 0.85 {
                set_learning_milestone.set("✓ LEARNED".to_string());
                set_learning_milestone_tone.set("learned".to_string());
            } else if rate >= 0.70 {
                set_learning_milestone.set("📈 Learning...".to_string());
                set_learning_milestone_tone.set("learning".to_string());
            } else if stats.trials < 20 {
                set_learning_milestone.set("⏳ Starting...".to_string());
                set_learning_milestone_tone.set("starting".to_string());
            } else {
                set_learning_milestone.set("🔄 Training".to_string());
                set_learning_milestone_tone.set("training".to_string());
            }

            // Keep game-specific UI state responsive.
            // (This is intentionally lightweight compared to brain-wide sampling.)

            let snap = runtime.with_value(|r| r.game_ui_snapshot());
            set_maze_state.set(snap.maze_state);
            set_spotxy_pos.set(snap.spotxy_pos);
            set_spotxy_stimulus_key.set(snap.spotxy_stimulus_key);
            set_spotxy_eval.set(snap.spotxy_eval);
            set_spotxy_mode.set(snap.spotxy_mode);
            set_spotxy_grid_n.set(snap.spotxy_grid_n);
            set_reversal_active.set(snap.reversal_active);
            set_reversal_flip_after.set(snap.reversal_flip_after_trials);
            set_pong_state.set(snap.pong_state);
            set_pong_stimulus_key.set(snap.pong_stimulus_key);
            set_pong_paddle_speed.set(snap.pong_paddle_speed);
            set_pong_paddle_half_height.set(snap.pong_paddle_half_height);
            set_pong_ball_speed.set(snap.pong_ball_speed);
            set_pong_paddle_bounce_y.set(snap.pong_paddle_bounce_y);
            set_pong_respawn_delay_s.set(snap.pong_respawn_delay_s);
            set_pong_distractor_enabled.set(snap.pong_distractor_enabled);
            set_pong_distractor_speed_scale.set(snap.pong_distractor_speed_scale);
            set_sequence_state.set(snap.sequence_state);
            set_text_state.set(snap.text_state);
            set_replay_state.set(snap.replay_state);
            set_spot_is_left.set(snap.spot_is_left);

            // Read live Bandit parameters when in-bandit.
            runtime.with_value(|r| {
                if let WebGame::Bandit(g) = &r.game {
                    set_bandit_prob_left.set(g.prob_left);
                    set_bandit_prob_right.set(g.prob_right);
                }
            });

            // Heavy brain-wide sampling: throttle aggressively while running.
            // This is the main UI perf lever on wasm.
            let heavy_interval_ms = if running { 1_000.0 } else { 250.0 };
            let should_do_heavy =
                (now_ms - ui_last_heavy_refresh_ms.get_value()) >= heavy_interval_ms;

            // Unit plot sampling is only needed when the UnitPlot analytics panel is visible.
            let need_unit_plot = analytics_modal_open.get_untracked()
                && analytics_panel.get_untracked() == AnalyticsPanel::UnitPlot;

            if should_do_heavy || need_unit_plot {
                ui_last_heavy_refresh_ms.set_value(now_ms);

                set_diag.set(runtime.with_value(|r| r.brain.diagnostics()));
                set_brain_age.set(runtime.with_value(|r| r.brain.age_steps()));

                let ls = runtime.with_value(|r| r.brain.learning_stats());
                set_learn_stats.set(ls);
                learn_elig_history.update_value(|h| h.push(ls.eligibility_l1));
                learn_plasticity_history.update_value(|h| h.push(ls.plasticity_l1));
                learn_homeostasis_history.update_value(|h| h.push(ls.homeostasis_bias_l1));
                set_learn_version.update(|v| *v = v.wrapping_add(1));

                let cstats = runtime.with_value(|r| r.brain.causal_stats());
                set_causal_symbols.set(cstats.base_symbols);
                set_causal_edges.set(cstats.edges);

                if need_unit_plot {
                    let plot_points = runtime.with_value(|r| r.brain.unit_plot_points(128));
                    set_unit_plot.set(plot_points);
                }
            }

            // Persist per-game stats + chart history so refresh restores the current state.
            let kind = game_kind.get_untracked();
            let should_persist =
                persisted_trial_cursor.with_value(|(k, t)| *k != kind || *t != stats.trials);
            if should_persist {
                persisted_trial_cursor.update_value(|(k, t)| {
                    *k = kind;
                    *t = stats.trials;
                });

                // localStorage writes are synchronous and can be very costly.
                // During gameplay, debounce to keep the main thread smooth.
                let persist_interval_ms = if running { 2_000.0 } else { 0.0 };
                let ok_to_persist = persist_interval_ms == 0.0
                    || (now_ms - ui_last_persist_ms.get_value()) >= persist_interval_ms;

                if ok_to_persist {
                    ui_last_persist_ms.set_value(now_ms);
                    let state = PersistedStatsState {
                        version: 1,
                        game: kind.label().to_string(),
                        stats: PersistedGameStats::from(&stats),
                        perf_history: perf_history.with_value(|h| h.data().to_vec()),
                        neuromod_history: neuromod_history.with_value(|h| h.data().to_vec()),
                        choice_events: choice_events.with_value(|v| v.clone()),
                        last_action: last_action.get_untracked(),
                        last_reward: last_reward.get_untracked(),
                    };
                    save_persisted_stats_state(kind, &state);
                }
            }
        }
    };

    // A lightweight refresh used while a GPU step is in-flight. Keeps game canvases (especially
    // Pong) responsive without doing heavy per-tick analytics or brain sampling.
    let refresh_game_ui_from_runtime = {
        let runtime = runtime.clone();
        move || {
            let snap = runtime.with_value(|r| r.game_ui_snapshot());

            set_maze_state.set(snap.maze_state);

            set_spotxy_pos.set(snap.spotxy_pos);
            set_spotxy_stimulus_key.set(snap.spotxy_stimulus_key);
            set_spotxy_eval.set(snap.spotxy_eval);
            set_spotxy_mode.set(snap.spotxy_mode);
            set_spotxy_grid_n.set(snap.spotxy_grid_n);
            set_reversal_active.set(snap.reversal_active);
            set_reversal_flip_after.set(snap.reversal_flip_after_trials);
            set_pong_state.set(snap.pong_state);
            set_pong_stimulus_key.set(snap.pong_stimulus_key);
            set_pong_paddle_speed.set(snap.pong_paddle_speed);
            set_pong_paddle_half_height.set(snap.pong_paddle_half_height);
            set_pong_ball_speed.set(snap.pong_ball_speed);
            set_pong_paddle_bounce_y.set(snap.pong_paddle_bounce_y);
            set_pong_respawn_delay_s.set(snap.pong_respawn_delay_s);
            set_pong_distractor_enabled.set(snap.pong_distractor_enabled);
            set_pong_distractor_speed_scale.set(snap.pong_distractor_speed_scale);
            set_sequence_state.set(snap.sequence_state);
            set_text_state.set(snap.text_state);
            set_replay_state.set(snap.replay_state);
            set_spot_is_left.set(snap.spot_is_left);
        }
    };

    let persist_stats_state: Arc<dyn Fn(GameKind) + Send + Sync> = Arc::new({
        let runtime = runtime.clone();
        move |kind: GameKind| {
            let stats = runtime.with_value(|r| r.game.stats().clone());
            let state = PersistedStatsState {
                version: 1,
                game: kind.label().to_string(),
                stats: PersistedGameStats::from(&stats),
                perf_history: perf_history.with_value(|h| h.data().to_vec()),
                neuromod_history: neuromod_history.with_value(|h| h.data().to_vec()),
                choice_events: choice_events.with_value(|v| v.clone()),
                last_action: last_action.get_untracked(),
                last_reward: last_reward.get_untracked(),
            };
            save_persisted_stats_state(kind, &state);
        }
    });

    let restore_stats_state: Arc<dyn Fn(GameKind) + Send + Sync> = Arc::new({
        let runtime = runtime.clone();
        move |kind: GameKind| {
            if let Some(state) = load_persisted_stats_state(kind) {
                runtime.update_value(|r| r.game.set_stats(state.stats.clone().into_game_stats()));
                perf_history.update_value(|h| h.set_data(state.perf_history.clone()));
                neuromod_history.update_value(|h| h.set_data(state.neuromod_history.clone()));
                set_neuromod_version.update(|v| *v = v.wrapping_add(1));
                choice_events.update_value(|v| *v = state.choice_events.clone());
                set_choices_version.update(|v| *v = v.wrapping_add(1));
                set_last_action.set(state.last_action);
                set_last_reward.set(state.last_reward);
            } else {
                runtime.update_value(|r| r.game.set_stats(braine_games::stats::GameStats::new()));
                perf_history.update_value(|h| h.clear());
                neuromod_history.update_value(|h| h.clear());
                set_neuromod_version.update(|v| *v = v.wrapping_add(1));
                choice_events.update_value(|v| v.clear());
                set_choices_version.update(|v| *v = v.wrapping_add(1));
                set_last_action.set(String::new());
                set_last_reward.set(0.0);
            }
        }
    });

    let set_game: Arc<dyn Fn(GameKind) + Send + Sync> = Arc::new({
        let runtime = runtime.clone();
        let persist_stats_state = Arc::clone(&persist_stats_state);
        let restore_stats_state = Arc::clone(&restore_stats_state);
        move |kind: GameKind| {
            let old_kind = game_kind.get_untracked();
            (persist_stats_state)(old_kind);

            // Match daemon semantics: stop first before switching tasks.
            if let Some(id) = interval_id.get_untracked() {
                if let Some(w) = web_sys::window() {
                    w.clear_interval_with_handle(id);
                }
                set_interval_id.set(None);
                set_is_running.set(false);
            }

            // Save current game's accuracy before switching
            let current_accs = game_accuracies.get_untracked();
            spawn_local(async move {
                let _ = save_game_accuracies(&current_accs).await;
            });

            runtime.update_value(|r| r.set_game(kind));
            set_game_kind.set(kind);

            // Restore per-game stats/chart history (or clear if no saved state).
            (restore_stats_state)(kind);

            // Pong is sensitive to control cadence. If the user hasn't already chosen a
            // relatively fast trial period, switch to a more controllable default.
            if kind == GameKind::Pong {
                let cur = trial_period_ms.get_untracked();
                if cur > 200 {
                    set_trial_period_ms.set(100);
                }
            }

            set_steps.set(0);
            refresh_ui_from_runtime();
            set_status.set(format!("game set: {}", kind.label()));
        }
    });

    let open_docs = Callback::new({
        let set_show_about_page = set_show_about_page.clone();
        let set_about_sub_tab = set_about_sub_tab.clone();
        let set_dashboard_open = set_dashboard_open.clone();
        let set_sidebar_open = set_sidebar_open.clone();
        move |()| {
            set_dashboard_open.set(false);
            set_show_about_page.set(true);
            set_about_sub_tab.set(AboutSubTab::Overview);
            set_sidebar_open.set(false);
        }
    });

    let open_settings = Callback::new({
        let set_show_about_page = set_show_about_page.clone();
        let set_dashboard_tab = set_dashboard_tab.clone();
        let set_dashboard_open = set_dashboard_open.clone();
        let set_sidebar_open = set_sidebar_open.clone();
        move |()| {
            set_show_about_page.set(false);
            set_dashboard_tab.set(DashboardTab::Settings);
            set_dashboard_open.set(true);
            set_sidebar_open.set(false);
        }
    });

    let open_analytics = Callback::new({
        let set_show_about_page = set_show_about_page.clone();
        let set_sidebar_open = set_sidebar_open.clone();
        move |()| {
            set_show_about_page.set(false);
            set_analytics_panel.set(AnalyticsPanel::Reward);
            set_analytics_modal_open.set(true);
            set_sidebar_open.set(false);
        }
    });

    // Inspect: state-change log (structured; render formatting only when Inspect tab is open).
    // Declared before `do_tick` so the tick loop can append cheap structured events.
    let inspect_trial_events = StoredValue::new(Vec::<InspectTrialEvent>::new());
    let (inspect_trial_events_version, set_inspect_trial_events_version) = signal(0u32);

    let do_tick = {
        let runtime = runtime.clone();
        move || {
            let cfg = TickConfig {
                trial_period_ms: trial_period_ms.get_untracked(),
                exploration_eps: exploration_eps.get_untracked(),
                meaning_alpha: meaning_alpha.get_untracked(),
                reward_scale: reward_scale.get_untracked(),
                reward_bias: reward_bias.get_untracked(),
                learning_enabled: learning_enabled.get_untracked(),
            };

            let mut tick_result: TickResult = TickResult::Advanced(None);
            runtime.update_value(|r| {
                tick_result = r.tick(&cfg);
            });

            match tick_result {
                TickResult::PendingGpu => {
                    // A GPU step is in-flight; don't count this as a completed step.
                    // Still refresh *game UI only* so e.g. Pong animation doesn't appear frozen.
                    set_gpu_step_pending.set(true);
                    refresh_game_ui_from_runtime();
                }
                TickResult::Advanced(out) => {
                    set_gpu_step_pending.set(false);
                    set_brain_dirty.set(true);
                    if let Some(out) = out {
                        let last_action = out.last_action.clone();
                        set_last_action.set(last_action.clone());
                        set_last_reward.set(out.reward);

                        // Update choice history (cap to 200)
                        choice_events.update_value(|v| {
                            v.push(last_action);
                            if v.len() > 200 {
                                let extra = v.len() - 200;
                                v.drain(0..extra);
                            }
                        });
                        set_choices_version.update(|v| *v = v.wrapping_add(1));

                        // Track neuromod history
                        neuromod_history.update_value(|h| h.push(out.reward));
                        set_neuromod_version.update(|v| *v = v.wrapping_add(1));

                        // Inspect state-change log (cheap structured append; formatting happens in the view).
                        {
                            let kind = game_kind.get_untracked();
                            let trial = trials.get_untracked();
                            let rate = recent_rate.get_untracked();
                            let ev = InspectTrialEvent {
                                step: steps.get_untracked(),
                                game: kind,
                                trial,
                                action: out.last_action.clone(),
                                reward: out.reward,
                                recent_rate: rate,
                            };
                            inspect_trial_events.update_value(|v| {
                                v.push(ev);
                                if v.len() > 300 {
                                    let extra = v.len() - 300;
                                    v.drain(0..extra);
                                }
                            });
                            set_inspect_trial_events_version.update(|v| *v = v.wrapping_add(1));
                        }

                        // Record state changes per-game for later BrainViz replay.
                        // We record only when a trial completes (i.e., `out` is Some), which
                        // is much cheaper than recording on every interval tick.
                        if brainviz_record_enabled.get_untracked() {
                            // Keep the tick callback cheap: do recording work in a microtask.
                            let runtime = runtime.clone();
                            let kind = game_kind.get_untracked();
                            let n = brainviz_node_sample.get_untracked().clamp(16, 1024) as usize;
                            let edges_per_node =
                                brainviz_edges_per_node.get_untracked().clamp(1, 32) as usize;
                            let step_now = steps.get_untracked();
                            let active_thr = brainviz_active_amp_threshold.get_untracked();
                            let eps_phase_active =
                                brainviz_eps_phase_active.get_untracked().max(0.0);
                            let eps_phase_inactive =
                                brainviz_eps_phase_inactive.get_untracked().max(0.0);
                            let eps_amp01 = brainviz_eps_amp01.get_untracked().max(0.0);
                            let eps_salience01 = brainviz_eps_salience01.get_untracked().max(0.0);
                            let record_edges = brainviz_record_edges.get_untracked();
                            let eps_weight = brainviz_eps_weight.get_untracked().max(0.0);
                            let record_every = brainviz_record_every_trials.get_untracked().max(1);
                            let record_edges_every =
                                brainviz_record_edges_every_trials.get_untracked().max(1);
                            let trial_idx = steps.get_untracked() as u32;

                            if !trial_idx.is_multiple_of(record_every) {
                                return;
                            }

                            spawn_local(async move {
                                const MAX_FRAMES_PER_GAME: usize = 2_000;
                                const KEYFRAME_EVERY: usize = 50;

                                brainviz_recordings.update_value(|recs| {
                                    let rec = &mut recs[game_kind_index(kind)];

                                    // Start a fresh clip if settings changed.
                                    let settings_changed =
                                        rec.n != n || rec.edges_per_node != edges_per_node;
                                    if settings_changed {
                                        *rec = BrainvizGameRecording::default();
                                        rec.n = n;
                                        rec.edges_per_node = edges_per_node;
                                    }

                                    // (Re)initialize stable nodes + stable edge endpoints if needed.
                                    if rec.base_points.is_empty() {
                                        runtime.with_value(|r| {
                                            let pts = r.brain.unit_plot_points(n);

                                            let mut id_to_index: std::collections::HashMap<
                                                usize,
                                                usize,
                                            > = std::collections::HashMap::with_capacity(pts.len());
                                            for (i, p) in pts.iter().enumerate() {
                                                id_to_index.insert(p.id as usize, i);
                                            }

                                            // Stable edge endpoints: top-k neighbors per sampled node.
                                            let mut edge_pairs: Vec<(usize, usize)> = Vec::new();
                                            let mut edge_weights: Vec<f32> = Vec::new();
                                            if record_edges {
                                                for (src_i, p) in pts.iter().enumerate() {
                                                    let src_id = p.id as usize;
                                                    let mut candidates: Vec<(usize, f32)> = r
                                                        .brain
                                                        .neighbors(src_id)
                                                        .filter_map(|(t, w)| {
                                                            id_to_index
                                                                .get(&t)
                                                                .copied()
                                                                .map(|ti| (ti, w))
                                                        })
                                                        .collect();
                                                    candidates.sort_by(|a, b| {
                                                        b.1.abs().total_cmp(&a.1.abs())
                                                    });
                                                    for (ti, w) in
                                                        candidates.into_iter().take(edges_per_node)
                                                    {
                                                        edge_pairs.push((src_i, ti));
                                                        edge_weights.push(w);
                                                    }
                                                }
                                            }

                                            rec.base_points = pts;
                                            rec.edge_pairs = edge_pairs;
                                            rec.cur_amp01 =
                                                rec.base_points.iter().map(|p| p.amp01).collect();
                                            rec.cur_phase =
                                                rec.base_points.iter().map(|p| p.phase).collect();
                                            rec.cur_salience01 = rec
                                                .base_points
                                                .iter()
                                                .map(|p| p.salience01)
                                                .collect();
                                            rec.cur_edge_weights = edge_weights.clone();

                                            // Seed a keyframe at index 0.
                                            rec.keyframes.push((
                                                0,
                                                BrainvizKeyframe {
                                                    step: step_now,
                                                    amp01: rec.cur_amp01.clone(),
                                                    phase: rec.cur_phase.clone(),
                                                    salience01: rec.cur_salience01.clone(),
                                                    edge_weights,
                                                },
                                            ));
                                        });
                                    }

                                    // Compute deltas against current recording state.
                                    let mut node_deltas: Vec<BrainvizNodeDelta> = Vec::new();
                                    let mut edge_deltas: Vec<BrainvizEdgeDelta> = Vec::new();

                                    runtime.with_value(|r| {
                                        let pts = r.brain.unit_plot_points(n);

                                        // If sampling IDs shifted (e.g., brain grew), restart clip.
                                        let ids_match = pts.len() == rec.base_points.len()
                                            && pts
                                                .iter()
                                                .zip(rec.base_points.iter())
                                                .all(|(a, b)| a.id == b.id);
                                        if !ids_match {
                                            *rec = BrainvizGameRecording::default();
                                            rec.n = n;
                                            rec.edges_per_node = edges_per_node;
                                            return;
                                        }

                                        // Node deltas (vibration-relevant fields).
                                        for (i, p) in pts.iter().enumerate() {
                                            let mut mask: u8 = 0;

                                            let da = (p.amp01 - rec.cur_amp01[i]).abs();
                                            if da > eps_amp01 {
                                                mask |= 1;
                                            }

                                            let eps_phase = if p.amp01 >= active_thr {
                                                eps_phase_active
                                            } else {
                                                eps_phase_inactive
                                            };
                                            let dp =
                                                wrap_delta_phase_to_pi(p.phase - rec.cur_phase[i])
                                                    .abs();
                                            if dp > eps_phase {
                                                mask |= 2;
                                            }

                                            let ds = (p.salience01 - rec.cur_salience01[i]).abs();
                                            if ds > eps_salience01 {
                                                mask |= 4;
                                            }

                                            if mask != 0 {
                                                node_deltas.push(BrainvizNodeDelta {
                                                    idx: i as u16,
                                                    mask,
                                                    amp01: p.amp01,
                                                    phase: p.phase,
                                                    salience01: p.salience01,
                                                });
                                                if (mask & 1) != 0 {
                                                    rec.cur_amp01[i] = p.amp01;
                                                }
                                                if (mask & 2) != 0 {
                                                    rec.cur_phase[i] = p.phase;
                                                }
                                                if (mask & 4) != 0 {
                                                    rec.cur_salience01[i] = p.salience01;
                                                }
                                            }
                                        }

                                        // Edge deltas (optional + decimated).
                                        if record_edges
                                            && !rec.edge_pairs.is_empty()
                                            && trial_idx.is_multiple_of(record_edges_every)
                                        {
                                            // Scan neighbors once per source.
                                            let mut by_src: std::collections::HashMap<
                                                usize,
                                                Vec<(usize, usize)>,
                                            > = std::collections::HashMap::new();
                                            for (edge_idx, &(src_i, dst_i)) in
                                                rec.edge_pairs.iter().enumerate()
                                            {
                                                let src_id = pts[src_i].id as usize;
                                                let dst_id = pts[dst_i].id as usize;
                                                by_src
                                                    .entry(src_id)
                                                    .or_default()
                                                    .push((dst_id, edge_idx));
                                            }

                                            let mut new_w: Vec<f32> =
                                                vec![0.0; rec.edge_pairs.len()];
                                            for (src_id, wants) in by_src {
                                                let mut want_map: std::collections::HashMap<
                                                    usize,
                                                    usize,
                                                > = std::collections::HashMap::with_capacity(
                                                    wants.len(),
                                                );
                                                for (dst_id, edge_idx) in wants {
                                                    want_map.insert(dst_id, edge_idx);
                                                }
                                                for (t, w) in r.brain.neighbors(src_id) {
                                                    if let Some(&edge_idx) = want_map.get(&t) {
                                                        new_w[edge_idx] = w;
                                                    }
                                                }
                                            }

                                            for (i, &w) in new_w.iter().enumerate() {
                                                if (w - rec.cur_edge_weights[i]).abs() > eps_weight
                                                {
                                                    edge_deltas.push(BrainvizEdgeDelta {
                                                        idx: i as u32,
                                                        weight: w,
                                                    });
                                                    rec.cur_edge_weights[i] = w;
                                                }
                                            }
                                        }
                                    });

                                    if rec.base_points.is_empty() {
                                        return;
                                    }

                                    rec.deltas.push(BrainvizDeltaFrame {
                                        step: step_now,
                                        node_deltas,
                                        edge_deltas,
                                    });

                                    let idx = rec.deltas.len().saturating_sub(1);
                                    if idx % KEYFRAME_EVERY == 0 {
                                        rec.keyframes.push((
                                            idx,
                                            BrainvizKeyframe {
                                                step: step_now,
                                                amp01: rec.cur_amp01.clone(),
                                                phase: rec.cur_phase.clone(),
                                                salience01: rec.cur_salience01.clone(),
                                                edge_weights: rec.cur_edge_weights.clone(),
                                            },
                                        ));
                                    }

                                    if rec.deltas.len() > MAX_FRAMES_PER_GAME {
                                        let extra = rec.deltas.len() - MAX_FRAMES_PER_GAME;
                                        rec.deltas.drain(0..extra);
                                        rec.keyframes.retain(|(i, _)| *i >= extra);
                                        for (i, _) in rec.keyframes.iter_mut() {
                                            *i = i.saturating_sub(extra);
                                        }
                                    }
                                });
                            });
                        }
                    }

                    set_steps.update(|s| *s += 1);
                    refresh_ui_from_runtime();
                }
            }
        }
    };

    let do_reset = move || {
        let kind = game_kind.get_untracked();
        clear_persisted_stats_state(kind);

        runtime.update_value(|r| r.cancel_pending_tick());
        runtime.set_value(AppRuntime::new());
        runtime.update_value(|r| r.set_game(kind));

        set_steps.set(0);
        set_last_action.set(String::new());
        set_last_reward.set(0.0);
        set_learned_at_trial.set(None);
        set_mastered_at_trial.set(None);
        set_learning_milestone.set("Starting...".to_string());
        set_learning_milestone_tone.set("starting".to_string());
        perf_history.update_value(|h| h.clear());
        neuromod_history.update_value(|h| h.clear());
        choice_events.update_value(|v| v.clear());
        set_choices_version.update(|v| *v = v.wrapping_add(1));
        refresh_ui_from_runtime();
        set_status.set("reset".to_string());
    };

    let do_grow_units = {
        let runtime = runtime.clone();
        let refresh_ui_from_runtime = refresh_ui_from_runtime.clone();
        move || {
            let n = grow_units_n.get_untracked().max(1) as usize;

            if n > 256 {
                let msg = format!(
                    "Grow by {} units? This increases memory/compute and may stutter the UI.",
                    n
                );
                let ok = web_sys::window()
                    .and_then(|w| w.confirm_with_message(&msg).ok())
                    .unwrap_or(true);
                if !ok {
                    return;
                }
            }
            runtime.update_value(|r| {
                let con = r.brain.config().connectivity_per_unit;
                r.brain.grow_units(n, con);
            });
            set_brain_dirty.set(true);
            refresh_ui_from_runtime();
            set_status.set(format!("grew units by {}", n));
        }
    };

    // Accelerated Learning functions (Dream, Burst, Sync, Imprint)
    let do_dream = {
        let runtime = runtime.clone();
        let refresh_ui_from_runtime = refresh_ui_from_runtime.clone();
        move || {
            runtime.update_value(|r| {
                r.brain.dream(100, 2.0, 0.5);
            });
            set_brain_dirty.set(true);
            refresh_ui_from_runtime();
            set_status.set("dream: consolidation complete".to_string());
        }
    };

    let do_burst = {
        let runtime = runtime.clone();
        let refresh_ui_from_runtime = refresh_ui_from_runtime.clone();
        move || {
            runtime.update_value(|r| {
                r.brain.set_burst_mode(true, 3.0);
            });
            set_brain_dirty.set(true);
            refresh_ui_from_runtime();
            set_status.set("burst: learning rate boosted".to_string());
        }
    };

    let do_sync = {
        let runtime = runtime.clone();
        let refresh_ui_from_runtime = refresh_ui_from_runtime.clone();
        move || {
            runtime.update_value(|r| {
                r.brain.force_synchronize_sensors();
            });
            set_brain_dirty.set(true);
            refresh_ui_from_runtime();
            set_status.set("sync: sensor phases aligned".to_string());
        }
    };

    let do_imprint = {
        let runtime = runtime.clone();
        let refresh_ui_from_runtime = refresh_ui_from_runtime.clone();
        move || {
            runtime.update_value(|r| {
                r.brain.imprint_current_context(0.5);
            });
            set_brain_dirty.set(true);
            refresh_ui_from_runtime();
            set_status.set("imprint: context associations created".to_string());
        }
    };

    let do_start = {
        let do_tick = do_tick.clone();
        move || {
            if interval_id.get_untracked().is_some() {
                return;
            }
            let window = match web_sys::window() {
                Some(w) => w,
                None => {
                    set_status.set("no window".to_string());
                    return;
                }
            };

            let cb = Closure::wrap(Box::new(move || {
                do_tick();
            }) as Box<dyn FnMut()>);

            let interval_ms = run_interval_ms.get_untracked().clamp(8, 500) as i32;

            match window.set_interval_with_callback_and_timeout_and_arguments_0(
                cb.as_ref().unchecked_ref(),
                interval_ms,
            ) {
                Ok(id) => {
                    cb.forget();
                    set_interval_id.set(Some(id));
                    set_is_running.set(true);
                    set_status.set("running".to_string());
                }
                Err(_) => set_status.set("failed to start interval".to_string()),
            }
        }
    };

    let do_stop = {
        let persist_stats_state = Arc::clone(&persist_stats_state);
        move || {
            if let Some(id) = interval_id.get_untracked() {
                if let Some(w) = web_sys::window() {
                    w.clear_interval_with_handle(id);
                }
                set_interval_id.set(None);
                set_is_running.set(false);
                (persist_stats_state)(game_kind.get_untracked());
                set_status.set("stopped".to_string());
            }
        }
    };

    let do_tick_sv = StoredValue::new(do_tick.clone());
    let do_reset_sv = StoredValue::new(do_reset.clone());
    let do_start_sv = StoredValue::new(do_start.clone());
    let do_stop_sv = StoredValue::new(do_stop.clone());

    let do_text_apply_corpora: Arc<dyn Fn() + Send + Sync> = Arc::new({
        let runtime = runtime.clone();
        let refresh_ui_from_runtime = refresh_ui_from_runtime.clone();
        let do_stop = do_stop.clone();
        move || {
            do_stop();

            let corpus0 = text_corpus0.get_untracked();
            let corpus1 = text_corpus1.get_untracked();
            let max_vocab = text_max_vocab.get_untracked().clamp(2, 512) as usize;
            let shift_every = text_shift_every.get_untracked().max(1);

            clear_persisted_stats_state(GameKind::Text);

            runtime.update_value(|r| {
                let g = TextWebGame::new_with_corpora(&corpus0, &corpus1, max_vocab, shift_every);
                r.ensure_text_io(&g);
                r.game = WebGame::Text(g);
                r.pending_neuromod = 0.0;
            });

            set_steps.set(0);
            set_last_action.set(String::new());
            set_last_reward.set(0.0);
            set_learned_at_trial.set(None);
            set_mastered_at_trial.set(None);
            set_learning_milestone.set("Starting...".to_string());
            set_learning_milestone_tone.set("starting".to_string());
            perf_history.update_value(|h| h.clear());
            neuromod_history.update_value(|h| h.clear());
            choice_events.update_value(|v| v.clear());
            set_choices_version.update(|v| *v = v.wrapping_add(1));
            refresh_ui_from_runtime();
            set_status.set("text: applied corpora".to_string());
        }
    });

    let do_text_apply_corpora_sv = StoredValue::new(do_text_apply_corpora);

    let do_text_train_prompt: Arc<dyn Fn() + Send + Sync> = Arc::new({
        let runtime = runtime.clone();
        let refresh_ui_from_runtime = refresh_ui_from_runtime.clone();
        let do_stop = do_stop.clone();
        move || {
            do_stop();

            let prompt = text_train_prompt.get_untracked();
            let regime = text_train_regime.get_untracked().clamp(0, 1);
            let epochs = text_train_epochs.get_untracked().max(1);
            let alpha = meaning_alpha.get_untracked();
            let eps = exploration_eps.get_untracked().clamp(0.0, 1.0);
            let reward_scale = reward_scale.get_untracked();
            let reward_bias = reward_bias.get_untracked();
            let allow_learning = learning_enabled.get_untracked();

            let mut trained_pairs: u32 = 0;
            let mut correct: u32 = 0;
            let mut total_reward: f32 = 0.0;

            runtime.update_value(|r| {
                let (known_sensors, known_actions) = match &r.game {
                    WebGame::Text(g) => (
                        g.token_sensor_names().to_vec(),
                        g.allowed_actions().to_vec(),
                    ),
                    _ => return,
                };

                let bytes = prompt.as_bytes();
                if bytes.len() < 2 {
                    return;
                }

                let mut pending_neuromod = 0.0f32;
                for _ in 0..epochs {
                    for w in bytes.windows(2) {
                        let cur = w[0];
                        let nxt = w[1];

                        // Mirror the main tick loop: previous reward modulates the next step.
                        r.brain.set_neuromodulator(pending_neuromod);

                        let cur_sensor = choose_text_token_sensor(Some(cur), &known_sensors);
                        let next_sensor = choose_text_token_sensor(Some(nxt), &known_sensors);
                        let desired_action = token_action_name_from_sensor(&next_sensor);

                        r.brain.apply_stimulus(Stimulus::new("text", 1.0));
                        let regime_sensor = if regime == 1 {
                            "txt_regime_1"
                        } else {
                            "txt_regime_0"
                        };
                        r.brain.apply_stimulus(Stimulus::new(regime_sensor, 0.8));
                        r.brain.apply_stimulus(Stimulus::new(&cur_sensor, 1.0));

                        let ctx = format!(
                            "txt_r{}_c{}",
                            if regime == 1 { 1 } else { 0 },
                            token_action_name_from_sensor(&cur_sensor)
                        );
                        r.brain.note_compound_symbol(&[ctx.as_str()]);
                        r.brain.step();

                        let explore = r.rng_next_f32() < eps;
                        let rand_idx = r.rng_next_u64() as usize;
                        let chosen_action = if explore {
                            if known_actions.is_empty() {
                                "tok_UNK".to_string()
                            } else {
                                known_actions[rand_idx % known_actions.len()].to_string()
                            }
                        } else {
                            let ranked = r.brain.ranked_actions_with_meaning(ctx.as_str(), alpha);
                            ranked
                                .into_iter()
                                .find(|(a, _)| known_actions.iter().any(|k| k == a))
                                .map(|(a, _)| a)
                                .or_else(|| known_actions.first().cloned())
                                .unwrap_or_else(|| "tok_UNK".to_string())
                        };

                        let raw_reward = if chosen_action == desired_action {
                            1.0
                        } else {
                            -1.0
                        };
                        let shaped = ((raw_reward + reward_bias) * reward_scale).clamp(-5.0, 5.0);

                        r.brain.note_action(&chosen_action);
                        r.brain.note_compound_symbol(&[
                            "pair",
                            ctx.as_str(),
                            chosen_action.as_str(),
                        ]);

                        if allow_learning {
                            r.brain.set_neuromodulator(shaped);
                            r.brain.reinforce_action(&chosen_action, shaped);
                            pending_neuromod = shaped;
                            r.brain.commit_observation();
                        } else {
                            r.brain.set_neuromodulator(0.0);
                            pending_neuromod = 0.0;
                            r.brain.discard_observation();
                        }

                        trained_pairs = trained_pairs.wrapping_add(1);
                        if chosen_action == desired_action {
                            correct = correct.wrapping_add(1);
                        }
                        total_reward += shaped;
                    }
                }
            });

            refresh_ui_from_runtime();
            if trained_pairs > 0 {
                let acc = (correct as f32) / (trained_pairs as f32);
                let avg_r = total_reward / (trained_pairs as f32);
                set_status.set(format!(
                    "text: trained {} pairs (acc {}%, avg r {})",
                    trained_pairs,
                    fmt_f32_fixed(acc * 100.0, 1),
                    fmt_f32_fixed(avg_r, 2)
                ));
            } else {
                set_status.set("text: prompt too short".to_string());
            }
        }
    });

    let do_text_train_prompt_sv = StoredValue::new(do_text_train_prompt);

    let do_save = {
        let runtime = runtime.clone();
        move || {
            let bytes = match runtime.with_value(|r| r.brain.save_image_bytes()) {
                Ok(b) => b,
                Err(e) => {
                    set_status.set(format!("save failed: {e}"));
                    return;
                }
            };

            // Also get current accuracies to save
            let accs = game_accuracies.get_untracked();

            let set_status = set_status.clone();
            spawn_local(async move {
                // Save brain
                match idb_put_bytes(IDB_KEY_BRAIN_IMAGE, &bytes).await {
                    Ok(()) => {
                        // Also save game accuracies
                        let _ = save_game_accuracies(&accs).await;
                        set_brain_dirty.set(false);
                        set_idb_loaded.set(true);
                        let ts = js_sys::Date::new_0()
                            .to_iso_string()
                            .as_string()
                            .unwrap_or_default();
                        set_idb_last_save.set(ts);
                        set_status.set(format!("saved {} bytes to IndexedDB", bytes.len()));
                    }
                    Err(e) => set_status.set(format!("save failed: {e}")),
                }
            });
        }
    };

    let do_load = {
        let runtime = runtime.clone();
        move || {
            let set_status = set_status.clone();
            let refresh_ui_from_runtime = refresh_ui_from_runtime.clone();
            spawn_local(async move {
                match idb_get_bytes(IDB_KEY_BRAIN_IMAGE).await {
                    Ok(Some(bytes)) => match Brain::load_image_bytes(&bytes) {
                        Ok(brain) => {
                            runtime.update_value(|r| r.brain = brain);
                            set_steps.set(0);
                            set_last_action.set(String::new());
                            set_last_reward.set(0.0);
                            refresh_ui_from_runtime();
                            set_idb_loaded.set(true);
                            set_brain_dirty.set(false);
                            set_idb_loaded.set(true);
                            set_brain_dirty.set(false);
                            set_status.set(format!("loaded {} bytes from IndexedDB", bytes.len()));
                        }
                        Err(e) => set_status.set(format!("load failed: {e}")),
                    },
                    Ok(None) => set_status.set("no saved brain image in IndexedDB".to_string()),
                    Err(e) => set_status.set(format!("load failed: {e}")),
                }
            });
        }
    };

    // Migration: if IndexedDB contains an older brain image format, load it and re-save
    // using the current serializer (latest version).
    let do_migrate_idb_format = {
        let set_status = set_status.clone();
        move || {
            let set_status = set_status.clone();
            spawn_local(async move {
                match idb_get_bytes(IDB_KEY_BRAIN_IMAGE).await {
                    Ok(Some(bytes)) => match Brain::load_image_bytes(&bytes) {
                        Ok(brain) => match brain.save_image_bytes() {
                            Ok(new_bytes) => {
                                match idb_put_bytes(IDB_KEY_BRAIN_IMAGE, &new_bytes).await {
                                    Ok(()) => set_status.set(format!(
                                        "migrated IndexedDB brain image: {} -> {} bytes",
                                        bytes.len(),
                                        new_bytes.len()
                                    )),
                                    Err(e) => set_status.set(format!("migration save failed: {e}")),
                                }
                            }
                            Err(e) => set_status.set(format!("migration encode failed: {e}")),
                        },
                        Err(e) => set_status.set(format!("migration load failed: {e}")),
                    },
                    Ok(None) => set_status.set("no saved brain image in IndexedDB".to_string()),
                    Err(e) => set_status.set(format!("migration read failed: {e}")),
                }
            });
        }
    };

    // Best-effort auto-load on startup so the web runtime can run without the daemon.
    // This is intentionally quiet unless a saved image exists.
    {
        let runtime = runtime.clone();
        let refresh_ui_from_runtime = refresh_ui_from_runtime.clone();
        let did_autoload = StoredValue::new(false);
        Effect::new(move |_| {
            if did_autoload.get_value() {
                return;
            }
            did_autoload.set_value(true);

            let set_status = set_status.clone();
            spawn_local(async move {
                match idb_get_bytes(IDB_KEY_BRAIN_IMAGE).await {
                    Ok(Some(bytes)) => match Brain::load_image_bytes(&bytes) {
                        Ok(brain) => {
                            runtime.update_value(|r| r.brain = brain);
                            set_steps.set(0);
                            set_last_action.set(String::new());
                            set_last_reward.set(0.0);
                            refresh_ui_from_runtime();
                            set_idb_loaded.set(true);
                            set_brain_dirty.set(false);
                            set_status
                                .set(format!("auto-loaded {} bytes from IndexedDB", bytes.len()));
                        }
                        Err(e) => set_status.set(format!("auto-load failed: {e}")),
                    },
                    Ok(None) => {
                        // No-op
                    }
                    Err(e) => set_status.set(format!("auto-load failed: {e}")),
                }
            });
        });
    }

    let do_export_bbi = {
        let runtime = runtime.clone();
        move || {
            let bytes = match runtime.with_value(|r| r.brain.save_image_bytes()) {
                Ok(b) => b,
                Err(e) => {
                    set_status.set(format!("export failed: {e}"));
                    return;
                }
            };
            match download_bytes("braine.bbi", &bytes) {
                Ok(()) => set_status.set(format!("exported {} bytes (.bbi)", bytes.len())),
                Err(e) => set_status.set(format!("export failed: {e}")),
            }
        }
    };

    let _do_export_bbi_from_idb = move || {
        let set_status = set_status.clone();
        spawn_local(async move {
            match idb_get_bytes(IDB_KEY_BRAIN_IMAGE).await {
                Ok(Some(bytes)) => match download_bytes("brain-indexeddb.bbi", &bytes) {
                    Ok(()) => set_status.set(format!(
                        "exported {} bytes from IndexedDB (.bbi)",
                        bytes.len()
                    )),
                    Err(e) => set_status.set(format!("export failed: {e}")),
                },
                Ok(None) => set_status.set("no saved brain image in IndexedDB".to_string()),
                Err(e) => set_status.set(format!("export failed: {e}")),
            }
        });
    };

    let import_input_ref = NodeRef::<leptos::html::Input>::new();
    let do_import_bbi_click = {
        let import_input_ref = import_input_ref.clone();
        move || {
            if let Some(input) = import_input_ref.get() {
                input.click();
            }
        }
    };

    let do_import_bbi_change = {
        let runtime = runtime.clone();
        let refresh_ui_from_runtime = refresh_ui_from_runtime.clone();
        move |ev: web_sys::Event| {
            let autosave = import_autosave.get_untracked();

            let input: web_sys::HtmlInputElement = match ev.target().and_then(|t| t.dyn_into().ok())
            {
                Some(i) => i,
                None => {
                    set_status.set("import failed: no input".to_string());
                    return;
                }
            };

            let file = input.files().and_then(|fl| fl.get(0));

            let Some(file) = file else {
                set_status.set("import: no file selected".to_string());
                return;
            };

            input.set_value("");

            let set_status = set_status.clone();
            spawn_local(async move {
                match read_file_bytes(file).await {
                    Ok(bytes) => match Brain::load_image_bytes(&bytes) {
                        Ok(brain) => {
                            runtime.update_value(|r| r.brain = brain);
                            set_steps.set(0);
                            set_last_action.set(String::new());
                            set_last_reward.set(0.0);
                            refresh_ui_from_runtime();

                            if autosave {
                                match idb_put_bytes(IDB_KEY_BRAIN_IMAGE, &bytes).await {
                                    Ok(()) => {
                                        let ts = js_sys::Date::new_0()
                                            .to_iso_string()
                                            .as_string()
                                            .unwrap_or_default();
                                        set_idb_last_save.set(ts);
                                        set_status.set(format!(
                                            "imported {} bytes (.bbi); auto-saved to IndexedDB",
                                            bytes.len()
                                        ));
                                    }
                                    Err(e) => set_status.set(format!(
                                        "imported {} bytes (.bbi); auto-save failed: {e}",
                                        bytes.len()
                                    )),
                                }
                            } else {
                                set_status.set(format!("imported {} bytes (.bbi)", bytes.len()));
                            }
                        }
                        Err(e) => set_status.set(format!("import failed: {e}")),
                    },
                    Err(e) => set_status.set(format!("import failed: {e}")),
                }
            });
        }
    };

    let do_pong_set_param = {
        let runtime = runtime.clone();
        move |key: &'static str, value: f32| {
            let mut result: Result<(), String> = Ok(());
            runtime.update_value(|r| {
                result = r.pong_set_param(key, value);
            });
            match result {
                Ok(()) => {
                    refresh_ui_from_runtime();
                }
                Err(e) => {
                    set_status.set(format!("pong param failed: {e}"));
                }
            }
        }
    };

    let do_maze_set_difficulty = {
        let runtime = runtime.clone();
        move |difficulty_param: f32| {
            runtime.update_value(|r| {
                if let WebGame::Maze(g) = &mut r.game {
                    g.set_difficulty(braine_games::maze::MazeDifficulty::from_param(
                        difficulty_param,
                    ));
                }
            });
            refresh_ui_from_runtime();
        }
    };

    let do_bandit_set_probs = {
        let runtime = runtime.clone();
        move |prob_left: f32, prob_right: f32| {
            let pl = prob_left.clamp(0.0, 1.0);
            let pr = prob_right.clamp(0.0, 1.0);
            runtime.update_value(|r| {
                if let WebGame::Bandit(g) = &mut r.game {
                    g.prob_left = pl;
                    g.prob_right = pr;
                }
            });
            set_bandit_prob_left.set(pl);
            set_bandit_prob_right.set(pr);
            refresh_ui_from_runtime();
        }
    };

    let do_reversal_set_flip_after = {
        let runtime = runtime.clone();
        move |flip_after_trials: u32| {
            let n = flip_after_trials.max(1);
            runtime.update_value(|r| {
                if let WebGame::SpotReversal(g) = &mut r.game {
                    g.flip_after_trials = n;
                }
            });
            refresh_ui_from_runtime();
        }
    };

    let do_spotxy_grid_plus = {
        let runtime = runtime.clone();
        move || {
            runtime.update_value(|r| r.spotxy_increase_grid());
            refresh_ui_from_runtime();
        }
    };

    let do_spotxy_grid_minus = {
        let runtime = runtime.clone();
        move || {
            runtime.update_value(|r| r.spotxy_decrease_grid());
            refresh_ui_from_runtime();
        }
    };

    let do_spotxy_toggle_eval = {
        let runtime = runtime.clone();
        move || {
            let next = !spotxy_eval.get_untracked();
            runtime.update_value(|r| r.spotxy_set_eval(next));
            refresh_ui_from_runtime();
        }
    };

    let do_spotxy_set_eval = {
        let runtime = runtime.clone();
        move |eval: bool| {
            runtime.update_value(|r| r.spotxy_set_eval(eval));
            refresh_ui_from_runtime();
        }
    };

    let do_spotxy_set_grid_target = {
        let runtime = runtime.clone();
        move |target: u32| {
            runtime.update_value(|r| {
                let mut guard = 0u32;
                loop {
                    let cur = match &r.game {
                        WebGame::SpotXY(g) => g.grid_n(),
                        _ => return,
                    };
                    if cur == target || guard > 24 {
                        break;
                    }
                    if cur < target {
                        r.spotxy_increase_grid();
                    } else {
                        r.spotxy_decrease_grid();
                    }
                    guard += 1;
                }
            });
            refresh_ui_from_runtime();
        }
    };

    let canvas_ref = NodeRef::<leptos::html::Canvas>::new();
    Effect::new(move |_| {
        let _ = spotxy_grid_n.get();
        let _ = spotxy_eval.get();
        let pos = spotxy_pos.get();
        let action = last_action.get();
        let Some(canvas) = canvas_ref.get() else {
            return;
        };

        let grid_n = spotxy_grid_n.get();
        let accent = if spotxy_eval.get() {
            "#22c55e"
        } else {
            "#7aa2ff"
        };

        let selected = if action.is_empty() {
            None
        } else {
            Some(action.as_str())
        };

        match pos {
            Some((x, y)) => {
                let _ = draw_spotxy(&canvas, x, y, grid_n, accent, selected);
            }
            None => {
                let _ = draw_spotxy(&canvas, 0.0, 0.0, grid_n, accent, selected);
            }
        }
    });

    let maze_canvas_ref = NodeRef::<leptos::html::Canvas>::new();
    Effect::new(move |_| {
        let s = maze_state.get();
        let action = last_action.get();
        let Some(canvas) = maze_canvas_ref.get() else {
            return;
        };

        let selected = if action.is_empty() {
            None
        } else {
            Some(action.as_str())
        };

        if let Some(s) = &s {
            let _ = draw_maze(&canvas, s, selected);
        } else {
            let _ = clear_canvas(&canvas);
        }
    });

    let pong_canvas_ref = NodeRef::<leptos::html::Canvas>::new();
    // Pong canvas is rendered at the browser refresh rate (requestAnimationFrame) and
    // interpolates towards the latest simulation snapshot. This avoids visible "stepping"
    // when the brain tick / UI refresh is lower than the display FPS.
    let pong_anim_started = StoredValue::new(false);
    Effect::new(move |_| {
        let Some(canvas) = pong_canvas_ref.get() else {
            return;
        };
        if pong_anim_started.get_value() {
            return;
        }
        pong_anim_started.set_value(true);

        let window = match web_sys::window() {
            Some(w) => w,
            None => return,
        };
        let window_raf = window.clone();

        let smoothed: std::rc::Rc<std::cell::RefCell<Option<PongUiState>>> =
            std::rc::Rc::new(std::cell::RefCell::new(None));
        let last_ts_ms: std::rc::Rc<std::cell::RefCell<Option<f64>>> =
            std::rc::Rc::new(std::cell::RefCell::new(None));

        type RafCallbackCell = std::rc::Rc<std::cell::RefCell<Option<Closure<dyn FnMut(f64)>>>>;
        let f: RafCallbackCell = std::rc::Rc::new(std::cell::RefCell::new(None));
        let g = f.clone();

        *g.borrow_mut() = Some(Closure::wrap(Box::new(move |ts_ms: f64| {
            // Compute dt
            let dt_s: f32 = {
                let mut last = last_ts_ms.borrow_mut();
                let dt_s = if let Some(prev) = *last {
                    ((ts_ms - prev) / 1000.0).clamp(0.0, 0.05)
                } else {
                    0.016
                };
                *last = Some(ts_ms);
                dt_s as f32
            };

            // Smooth towards latest snapshot
            let target = pong_state.get_untracked();
            if let Some(t) = target {
                let tau: f32 = 0.07; // seconds; lower = snappier, higher = smoother
                let alpha: f32 = 1.0 - (-dt_s / tau).exp();

                let mut cur = smoothed.borrow_mut();
                let mut s = cur.unwrap_or(t);

                s.ball_x += (t.ball_x - s.ball_x) * alpha;
                s.ball_y += (t.ball_y - s.ball_y) * alpha;
                s.paddle_y += (t.paddle_y - s.paddle_y) * alpha;
                s.ball2_x += (t.ball2_x - s.ball2_x) * alpha;
                s.ball2_y += (t.ball2_y - s.ball2_y) * alpha;

                // Non-smoothed fields should snap to keep UI truthful.
                s.paddle_half_height = t.paddle_half_height;
                s.ball_visible = t.ball_visible;
                s.ball2_visible = t.ball2_visible;
                s.hits = t.hits;
                s.misses = t.misses;

                *cur = Some(s);
                let _ = draw_pong(&canvas, &s);
            } else {
                *smoothed.borrow_mut() = None;
                let _ = clear_canvas(&canvas);
            }

            // Queue next frame
            if let Some(cb) = f.borrow().as_ref() {
                let _ = window_raf.request_animation_frame(cb.as_ref().unchecked_ref());
            }
        }) as Box<dyn FnMut(f64)>));

        if let Some(cb) = g.borrow().as_ref() {
            let _ = window.request_animation_frame(cb.as_ref().unchecked_ref());
        }

        // Keep the closure alive for the lifetime of the app.
        // (Leptos will not drop it because we intentionally leak it here.)
        std::mem::forget(g);
    });

    // Neuromod (reward) chart canvas
    let neuromod_chart_ref = NodeRef::<leptos::html::Canvas>::new();
    Effect::new(move |_| {
        if !analytics_modal_open.get() || analytics_panel.get() != AnalyticsPanel::Reward {
            return;
        }
        let _ = neuromod_version.get();
        let Some(canvas) = neuromod_chart_ref.get() else {
            return;
        };
        let data: Vec<f32> = neuromod_history.with_value(|h| h.data().to_vec());
        let _ = charts::draw_neuromod_trace(&canvas, &data, "#0a0f1a");
    });

    // Choices-over-time chart canvas
    let choices_chart_ref = NodeRef::<leptos::html::Canvas>::new();
    Effect::new({
        let runtime = runtime.clone();
        move |_| {
            if !analytics_modal_open.get() || analytics_panel.get() != AnalyticsPanel::Choices {
                return;
            }
            let _ = choices_version.get();
            let _ = choice_window.get();
            let Some(canvas) = choices_chart_ref.get() else {
                return;
            };

            // Limit to a reasonable number of series for readability.
            let mut actions: Vec<String> = runtime.with_value(|r| r.game.allowed_actions_for_ui());
            if actions.len() > 6 {
                actions.truncate(6);
                actions.push("(other)".to_string());
            }

            let events: Vec<String> = choice_events.with_value(|v| v.clone());
            let window = choice_window.get() as usize;

            let mapped: Vec<String> = if actions.iter().any(|a| a == "(other)") {
                let keep: std::collections::HashSet<String> = actions
                    .iter()
                    .filter(|s| s.as_str() != "(other)")
                    .cloned()
                    .collect();
                events
                    .into_iter()
                    .map(|a| {
                        if keep.contains(&a) {
                            a
                        } else {
                            "(other)".to_string()
                        }
                    })
                    .collect()
            } else {
                events
            };

            let _ = charts::draw_choices_over_time(
                &canvas, &actions, &mapped, window, "#0a0f1a", "#1a2540",
            );
        }
    });

    // Unit plot 3D-style canvas for Graph page
    let unit_plot_ref = NodeRef::<leptos::html::Canvas>::new();
    Effect::new(move |_| {
        if !analytics_modal_open.get() || analytics_panel.get() != AnalyticsPanel::UnitPlot {
            return;
        }
        let points = unit_plot.get();
        let Some(canvas) = unit_plot_ref.get() else {
            return;
        };
        let _ = charts::draw_unit_plot_3d(&canvas, &points, "#0a0f1a");
    });

    // Inspect: lightweight sparklines (only draw when Inspect tab is open).
    let inspect_perf_spark_ref = NodeRef::<leptos::html::Canvas>::new();
    let inspect_reward_spark_ref = NodeRef::<leptos::html::Canvas>::new();
    let inspect_learn_elig_spark_ref = NodeRef::<leptos::html::Canvas>::new();
    let inspect_learn_plasticity_spark_ref = NodeRef::<leptos::html::Canvas>::new();
    let inspect_learn_homeostasis_spark_ref = NodeRef::<leptos::html::Canvas>::new();
    let (inspect_neighbor_unit_id, set_inspect_neighbor_unit_id) = signal(0u32);
    let (inspect_neighbors_text, set_inspect_neighbors_text) = signal(String::new());

    // Inspect: oscilloscope (single unit over time; updated only on snapshot refresh / replay step).
    let (inspect_scope_unit_id, set_inspect_scope_unit_id) = signal(0u32);
    let inspect_scope_amp = StoredValue::new(RollingHistory::new(120));
    let inspect_scope_phase_sin = StoredValue::new(RollingHistory::new(120));
    let (inspect_scope_version, set_inspect_scope_version) = signal(0u32);
    let inspect_scope_amp_ref = NodeRef::<leptos::html::Canvas>::new();
    let inspect_scope_phase_ref = NodeRef::<leptos::html::Canvas>::new();
    Effect::new(move |_| {
        if dashboard_tab.get() != DashboardTab::Inspect {
            return;
        }
        let _ = perf_version.get();
        let Some(canvas) = inspect_perf_spark_ref.get() else {
            return;
        };
        let data: Vec<f32> = perf_history.with_value(|h| h.data().to_vec());
        let _ = charts::draw_sparkline(&canvas, &data, "#0a0f1a", "#7aa2ff", "#1a2540");
    });
    Effect::new(move |_| {
        if dashboard_tab.get() != DashboardTab::Inspect {
            return;
        }
        let _ = neuromod_version.get();
        let Some(canvas) = inspect_reward_spark_ref.get() else {
            return;
        };
        let data: Vec<f32> = neuromod_history.with_value(|h| h.data().to_vec());
        let _ = charts::draw_sparkline(&canvas, &data, "#0a0f1a", "#fbbf24", "#1a2540");
    });

    Effect::new(move |_| {
        if dashboard_tab.get() != DashboardTab::Inspect {
            return;
        }
        let _ = learn_version.get();
        let Some(canvas) = inspect_learn_plasticity_spark_ref.get() else {
            return;
        };
        let data: Vec<f32> = learn_plasticity_history.with_value(|h| h.data().to_vec());
        let _ = charts::draw_sparkline(&canvas, &data, "#0a0f1a", "#7aa2ff", "#1a2540");
    });
    Effect::new(move |_| {
        if dashboard_tab.get() != DashboardTab::Inspect {
            return;
        }
        let _ = learn_version.get();
        let Some(canvas) = inspect_learn_elig_spark_ref.get() else {
            return;
        };
        let data: Vec<f32> = learn_elig_history.with_value(|h| h.data().to_vec());
        let _ = charts::draw_sparkline(&canvas, &data, "#0a0f1a", "#4ade80", "#1a2540");
    });
    Effect::new(move |_| {
        if dashboard_tab.get() != DashboardTab::Inspect {
            return;
        }
        let _ = learn_version.get();
        let Some(canvas) = inspect_learn_homeostasis_spark_ref.get() else {
            return;
        };
        let data: Vec<f32> = learn_homeostasis_history.with_value(|h| h.data().to_vec());
        let _ = charts::draw_sparkline(&canvas, &data, "#0a0f1a", "#a78bfa", "#1a2540");
    });

    // Inspect: oscilloscope sampling (only when Inspect is open).
    // Trigger on snapshot refresh, replay idx changes, or unit id changes.
    Effect::new(move |_| {
        if dashboard_tab.get() != DashboardTab::Inspect {
            return;
        }
        let _ = brainviz_snapshot_refresh.get();
        let _ = brainviz_replay_active.get();
        let _ = brainviz_replay_idx.get();
        let unit_id = inspect_scope_unit_id.get();

        if unit_id == 0 {
            return;
        }

        let mut amp01: Option<f32> = None;
        let mut phase_sin: Option<f32> = None;
        brainviz_points.with(|pts| {
            if let Some(p) = pts.iter().find(|p| p.id == unit_id) {
                amp01 = Some(p.amp01);
                phase_sin = Some(p.phase.sin());
            }
        });

        if let (Some(a), Some(s)) = (amp01, phase_sin) {
            inspect_scope_amp.update_value(|h| h.push(a));
            inspect_scope_phase_sin.update_value(|h| h.push(s));
            set_inspect_scope_version.update(|v| *v = v.wrapping_add(1));
        }
    });

    Effect::new(move |_| {
        if dashboard_tab.get() != DashboardTab::Inspect {
            return;
        }
        let _ = inspect_scope_version.get();
        let Some(canvas) = inspect_scope_amp_ref.get() else {
            return;
        };
        let data: Vec<f32> = inspect_scope_amp.with_value(|h| h.data().to_vec());
        let _ = charts::draw_sparkline(&canvas, &data, "#0a0f1a", "#4ade80", "#1a2540");
    });
    Effect::new(move |_| {
        if dashboard_tab.get() != DashboardTab::Inspect {
            return;
        }
        let _ = inspect_scope_version.get();
        let Some(canvas) = inspect_scope_phase_ref.get() else {
            return;
        };
        let data: Vec<f32> = inspect_scope_phase_sin.with_value(|h| h.data().to_vec());
        let _ = charts::draw_sparkline(&canvas, &data, "#0a0f1a", "#a78bfa", "#1a2540");
    });

    // BrainViz snapshot sampler (refresh-driven):
    // - gated by active dashboard tab
    // - never samples while running
    // - replay overrides live sampling
    Effect::new({
        let runtime = runtime.clone();
        move |_| {
            if brainviz_replay_active.get() {
                return;
            }
            if is_running.get() {
                return;
            }

            let active_tab = dashboard_tab.get();
            if active_tab != DashboardTab::BrainViz && active_tab != DashboardTab::Inspect {
                return;
            }

            let refresh = brainviz_snapshot_refresh.get();
            let points_empty = brainviz_points.get().is_empty();
            if !points_empty && refresh == brainviz_last_snapshot_refresh.get_value() {
                return;
            }
            brainviz_last_snapshot_refresh.set_value(refresh);

            let view_mode = brainviz_view_mode.get();
            let n = brainviz_node_sample.get().clamp(16, 1024) as usize;
            let edges_per_node = brainviz_edges_per_node.get().clamp(1, 32) as usize;

            if view_mode == "causal" {
                let causal = runtime.with_value(|r| r.brain.causal_graph_viz(n, n * 2));
                set_brainviz_causal_graph.set(causal);
                return;
            }

            let pts = runtime.with_value(|r| r.brain.unit_plot_points(n));

            if brainviz_base_positions_n.get_value() != pts.len() {
                let npos = pts.len();
                let n_f = npos.max(1) as f64;
                let golden = 2.399_963_229_728_653_5_f64; // ~pi*(3-sqrt(5))
                let mut base: Vec<(f64, f64, f64)> = Vec::with_capacity(npos);
                for i in 0..npos {
                    let i_f = i as f64;
                    let y = 1.0 - 2.0 * ((i_f + 0.5) / n_f);
                    let r = (1.0 - y * y).sqrt();
                    let theta = golden * i_f;
                    let x = r * theta.cos();
                    let z = r * theta.sin();
                    base.push((x, y, z));
                }
                brainviz_base_positions.set_value(base);
                brainviz_base_positions_n.set_value(npos);
            }

            let edges: Vec<(usize, usize, f32)> = runtime.with_value(|r| {
                let mut id_to_index: std::collections::HashMap<usize, usize> =
                    std::collections::HashMap::with_capacity(pts.len());
                for (i, p) in pts.iter().enumerate() {
                    id_to_index.insert(p.id as usize, i);
                }

                let mut edges: Vec<(usize, usize, f32)> = Vec::new();
                for (src_i, p) in pts.iter().enumerate() {
                    let src_id = p.id as usize;
                    let mut candidates: Vec<(usize, f32)> = r
                        .brain
                        .neighbors(src_id)
                        .filter_map(|(t, w)| id_to_index.get(&t).copied().map(|ti| (ti, w)))
                        .collect();
                    candidates.sort_by(|a, b| b.1.abs().total_cmp(&a.1.abs()));
                    for (ti, w) in candidates.into_iter().take(edges_per_node) {
                        edges.push((src_i, ti, w));
                    }
                }
                edges
            });

            let node_count = pts.len();
            let edge_count = edges.len();
            let mut degrees = vec![0usize; node_count.max(1)];
            for (a_i, b_i, _w) in &edges {
                if *a_i < degrees.len() {
                    degrees[*a_i] += 1;
                }
                if *b_i < degrees.len() {
                    degrees[*b_i] += 1;
                }
            }
            let max_conn = degrees.iter().copied().max().unwrap_or(0);
            let avg_conn = if node_count > 0 {
                (2.0 * (edge_count as f32)) / (node_count as f32)
            } else {
                0.0
            };

            set_brainviz_display_nodes.set(node_count);
            set_brainviz_display_edges.set(edge_count);
            set_brainviz_display_avg_conn.set(avg_conn);
            set_brainviz_display_max_conn.set(max_conn);
            set_brainviz_edges.set(edges);
            set_brainviz_points.set(pts);
        }
    });

    // BrainViz: rotating sphere + sampled connectivity (or causal graph)
    let brain_viz_ref = NodeRef::<leptos::html::Canvas>::new();
    Effect::new({
        let _runtime = runtime.clone();
        move |_| {
            if dashboard_tab.get() != DashboardTab::BrainViz {
                return;
            }

            let expanded = brainviz_is_expanded.get();
            let step = steps.get(); // Track for reactivity
            let running = is_running.get();
            let idle_time = brainviz_idle_time.get(); // Idle animation time
            let view_mode = brainviz_view_mode.get();

            // Hard stop: BrainViz does not render while the brain/game is running.
            // We record state changes during running and allow replay while stopped.
            if running {
                return;
            }

            let Some(canvas) = brain_viz_ref.get() else {
                return;
            };

            // Throttle render: keep pinned preview cheap.
            if running {
                let every: u64 = if expanded { 1 } else { 4 };
                if every > 1 && step % every != 0 {
                    return;
                }
            } else {
                let bucket = (idle_time * 30.0) as u32; // ~30fps
                let every: u32 = if expanded { 2 } else { 5 }; // ~15fps vs ~6fps
                #[allow(clippy::manual_is_multiple_of)]
                if bucket % every != 0 {
                    return;
                }
            }

            let zoom = brainviz_zoom.get();
            let pan_x = brainviz_pan_x.get();
            let pan_y = brainviz_pan_y.get();
            let manual_rot = brainviz_manual_rotation.get();
            let rot_x = brainviz_rotation_x.get();
            let _vibration = brainviz_vibration.get(); // Deprecated: vibration is now per-node

            // When idle (not running), add slow "dreaming" rotation and use idle_time for animation.
            // This creates a visual effect of the brain being alive but in a resting/sync state.
            let idle_rot = if running { 0.0 } else { idle_time * 0.05 }; // Slow idle rotation
            let rot_y = manual_rot + idle_rot;

            // Use step-based time when running, idle_time when not running (for dreaming animation)
            let anim_time = if running {
                step as f32
            } else {
                idle_time * 30.0 // Scale up for visible animation speed
            };

            if view_mode == "causal" {
                // Render causal graph with same 3D sphere layout as substrate view
                let causal = brainviz_causal_graph.get();
                let prefix = brainviz_causal_filter_prefix.get();
                let focus_selected = brainviz_causal_focus_selected.get();
                let hide_isolates = brainviz_causal_hide_isolates.get();
                let selected_sym = brainviz_selected_symbol_id.get();

                // Filter in-memory only (no extra calls into the Brain).
                let mut nodes: Vec<braine::substrate::CausalNodeViz> = if prefix.trim().is_empty() {
                    causal.nodes.clone()
                } else {
                    let p = prefix.trim();
                    causal
                        .nodes
                        .iter()
                        .filter(|n| n.name.starts_with(p))
                        .cloned()
                        .collect()
                };

                // Keep edges consistent with filtered nodes.
                let mut allowed: std::collections::HashSet<u32> =
                    std::collections::HashSet::with_capacity(nodes.len() * 2);
                for n in &nodes {
                    allowed.insert(n.id);
                }
                let mut edges: Vec<braine::substrate::CausalEdgeViz> = causal
                    .edges
                    .iter()
                    .copied()
                    .filter(|e| allowed.contains(&e.from) && allowed.contains(&e.to))
                    .collect();

                // Optional focus mode: show only edges incident to selected symbol.
                if focus_selected {
                    if let Some(sel) = selected_sym {
                        edges.retain(|e| e.from == sel || e.to == sel);

                        let mut keep: std::collections::HashSet<u32> =
                            std::collections::HashSet::with_capacity(edges.len() * 2 + 1);
                        keep.insert(sel);
                        for e in &edges {
                            keep.insert(e.from);
                            keep.insert(e.to);
                        }
                        nodes.retain(|n| keep.contains(&n.id));
                    } else {
                        // Focus requires a selection.
                        edges.clear();
                    }
                }

                // Optional cleanup: remove isolated nodes (degree 0) from the filtered view.
                // Keep the selected node visible even if it becomes isolated.
                if hide_isolates {
                    let mut degrees_by_id: std::collections::HashMap<u32, usize> =
                        std::collections::HashMap::with_capacity(edges.len() * 2);
                    for e in &edges {
                        *degrees_by_id.entry(e.from).or_insert(0) += 1;
                        *degrees_by_id.entry(e.to).or_insert(0) += 1;
                    }

                    nodes.retain(|n| {
                        degrees_by_id.get(&n.id).copied().unwrap_or(0) > 0
                            || selected_sym == Some(n.id)
                    });

                    let mut allowed: std::collections::HashSet<u32> =
                        std::collections::HashSet::with_capacity(nodes.len() * 2);
                    for n in &nodes {
                        allowed.insert(n.id);
                    }
                    edges.retain(|e| allowed.contains(&e.from) && allowed.contains(&e.to));
                }

                let sym_tags = brainviz_symbol_tags.get();
                let mut sym_overrides: std::collections::HashMap<u32, (u8, u8, u8)> =
                    std::collections::HashMap::with_capacity(sym_tags.len());
                for (id, tag) in sym_tags.iter() {
                    if let Some(rgb) = parse_hex_rgb(&tag.color) {
                        sym_overrides.insert(*id, rgb);
                    }
                }

                let node_count = nodes.len();
                let edge_count = edges.len();
                let mut id_to_idx: std::collections::HashMap<u32, usize> =
                    std::collections::HashMap::with_capacity(node_count);
                for (i, n) in nodes.iter().enumerate() {
                    id_to_idx.insert(n.id, i);
                }

                let mut degrees = vec![0usize; node_count.max(1)];
                for e in &edges {
                    if let Some(&i) = id_to_idx.get(&e.from) {
                        degrees[i] += 1;
                    }
                    if let Some(&i) = id_to_idx.get(&e.to) {
                        degrees[i] += 1;
                    }
                }
                let max_conn = degrees.iter().copied().max().unwrap_or(0);
                let avg_conn = if node_count > 0 {
                    (2.0 * (edge_count as f32)) / (node_count as f32)
                } else {
                    0.0
                };
                set_brainviz_display_nodes.set(node_count);
                set_brainviz_display_edges.set(edge_count);
                set_brainviz_display_avg_conn.set(avg_conn);
                set_brainviz_display_max_conn.set(max_conn);

                let opts_full = charts::CausalVizRenderOptions {
                    zoom,
                    pan_x,
                    pan_y,
                    rotation: rot_y,
                    rotation_x: rot_x,
                    draw_outline: false,
                    anim_time,
                };
                let hit = charts::draw_causal_graph(
                    &canvas,
                    &nodes,
                    &edges,
                    "#0a0f1a",
                    Some(&sym_overrides),
                    opts_full,
                );
                if let Ok(hit) = hit {
                    brainviz_causal_hit_nodes.set_value(hit);
                }
            } else {
                // Render substrate view
                let is_learning = learning_enabled.get();
                let tags = brainviz_node_tags.get();
                let mut overrides: std::collections::HashMap<u32, (u8, u8, u8)> =
                    std::collections::HashMap::with_capacity(tags.len());
                for (id, tag) in tags.iter() {
                    if let Some(rgb) = parse_hex_rgb(&tag.color) {
                        overrides.insert(*id, rgb);
                    }
                }
                let opts_full = charts::BrainVizRenderOptions {
                    zoom,
                    pan_x,
                    pan_y,
                    draw_outline: false,
                    node_size_scale: 0.5,
                    learning_mode: is_learning,
                    anim_time, // Uses step-based time when running, idle_time when dreaming
                    rotation_y: rot_y,
                    rotation_x: rot_x,
                };
                brainviz_points.with(|points| {
                    brainviz_edges.with(|edges| {
                        brainviz_base_positions.with_value(|base_positions| {
                            let hit = charts::draw_brain_connectivity_sphere(
                                &canvas,
                                points,
                                base_positions.as_slice(),
                                edges.as_slice(),
                                "#0a0f1a",
                                None,
                                Some(&overrides),
                                opts_full,
                            );
                            if let Ok(hit) = hit {
                                brainviz_hit_nodes.set_value(hit);
                            }
                        });
                    });
                });
            }
        }
    });

    // Load game accuracies from IndexedDB on mount
    Effect::new(move |_| {
        spawn_local(async move {
            if let Ok(accs) = load_game_accuracies().await {
                set_game_accuracies.set(accs);
            }
        });
    });

    // BrainViz replay driver: when replay is active, load the selected frame
    // into the same `brainviz_points`/`brainviz_edges` signals used by the renderer.
    Effect::new(move |_| {
        if !brainviz_replay_active.get() {
            return;
        }

        let kind = brainviz_replay_kind.get();
        let idx = brainviz_replay_idx.get();

        let reconstructed = brainviz_recordings.with_value(|recs| {
            let rec = &recs[game_kind_index(kind)];
            if rec.base_points.is_empty() || rec.deltas.is_empty() {
                return None;
            }
            let idx = idx.min(rec.deltas.len().saturating_sub(1));

            // Find nearest keyframe <= idx.
            let mut kf_i = 0usize;
            let mut kf: Option<&BrainvizKeyframe> = None;
            for (i, frame) in &rec.keyframes {
                if *i <= idx {
                    kf_i = *i;
                    kf = Some(frame);
                } else {
                    break;
                }
            }
            let kf = kf.or_else(|| rec.keyframes.first().map(|(_, f)| f))?;

            let mut amp01 = kf.amp01.clone();
            let mut phase = kf.phase.clone();
            let mut salience01 = kf.salience01.clone();
            let mut edge_w = kf.edge_weights.clone();

            for df in rec
                .deltas
                .iter()
                .take(idx.saturating_add(1))
                .skip(kf_i.saturating_add(1))
            {
                for nd in &df.node_deltas {
                    let i = nd.idx as usize;
                    if (nd.mask & 1) != 0 {
                        amp01[i] = nd.amp01;
                    }
                    if (nd.mask & 2) != 0 {
                        phase[i] = nd.phase;
                    }
                    if (nd.mask & 4) != 0 {
                        salience01[i] = nd.salience01;
                    }
                }
                for ed in &df.edge_deltas {
                    let i = ed.idx as usize;
                    if i < edge_w.len() {
                        edge_w[i] = ed.weight;
                    }
                }
            }

            let step = rec.deltas[idx].step;
            let mut points = rec.base_points.clone();
            for (i, p) in points.iter_mut().enumerate() {
                p.amp01 = amp01[i];
                p.phase = phase[i];
                p.salience01 = salience01[i];
                // `amp` isn't used by BrainViz rendering; keep it consistent.
                p.amp = p.amp01;
            }

            let edges: Vec<(usize, usize, f32)> = rec
                .edge_pairs
                .iter()
                .zip(edge_w.iter())
                .map(|(&(a, b), &w)| (a, b, w))
                .collect();

            Some((step, points, edges))
        });

        if let Some((step, points, edges)) = reconstructed {
            let node_count = points.len();
            let edge_count = edges.len();
            let mut degrees = vec![0usize; node_count.max(1)];
            for (a_i, b_i, _w) in &edges {
                if *a_i < degrees.len() {
                    degrees[*a_i] += 1;
                }
                if *b_i < degrees.len() {
                    degrees[*b_i] += 1;
                }
            }
            let max_conn = degrees.iter().copied().max().unwrap_or(0);
            let avg_conn = if node_count > 0 {
                (2.0 * (edge_count as f32)) / (node_count as f32)
            } else {
                0.0
            };

            set_brainviz_display_nodes.set(node_count);
            set_brainviz_display_edges.set(edge_count);
            set_brainviz_display_avg_conn.set(avg_conn);
            set_brainviz_display_max_conn.set(max_conn);

            // For replay, we always show the substrate view.
            set_brainviz_view_mode.set("substrate");
            set_brainviz_points.set(points);
            set_brainviz_edges.set(edges);

            // Keep base positions in sync with the replay sample size.
            // (Renderer falls back if mismatched, but this preserves stability.)
            if brainviz_base_positions_n.get_value() != node_count {
                let npos = node_count;
                let n_f = npos.max(1) as f64;
                let golden = 2.399_963_229_728_653_5_f64;
                let mut base: Vec<(f64, f64, f64)> = Vec::with_capacity(npos);
                for i in 0..npos {
                    let i_f = i as f64;
                    let y = 1.0 - 2.0 * ((i_f + 0.5) / n_f);
                    let r = (1.0 - y * y).sqrt();
                    let theta = golden * i_f;
                    let x = r * theta.cos();
                    let z = r * theta.sin();
                    base.push((x, y, z));
                }
                brainviz_base_positions.set_value(base);
                brainviz_base_positions_n.set_value(npos);
            }

            // Surface the replay step in status for quick sanity checking.
            set_status.set(format!("replay step {step}"));
        }
    });

    // Restore per-game stats (counters + charts) from localStorage once at startup.
    let did_restore_stats = StoredValue::new(false);
    Effect::new(move |_| {
        if did_restore_stats.get_value() {
            return;
        }
        did_restore_stats.set_value(true);
        // This hydrates the runtime + local charts/counters so refresh restores the current session.
        let kind = game_kind.get_untracked();
        (restore_stats_state)(kind);
        refresh_ui_from_runtime();
    });

    let training_health_bar_view = move || {
        let raw_rate = recent_rate.get();
        let rate = if raw_rate.is_finite() { raw_rate } else { 0.0 }.clamp(0.0, 1.0);
        let pct = rate * 100.0;
        let (label, color) = if rate >= 0.95 {
            ("mastered", "#4ade80")
        } else if rate >= 0.85 {
            ("learned", "#7aa2ff")
        } else if rate >= 0.70 {
            ("improving", "#fbbf24")
        } else {
            ("training", "#fb7185")
        };
        view! {
            <div style="margin-top: 10px;">
                <div style="display:flex; align-items:center; justify-content: space-between; gap: 10px;">
                    <div style="font-weight: 800; color: var(--text); font-size: 0.86rem;">"Training health"</div>
                    <div style="font-family: var(--mono); color: var(--muted); font-size: 0.78rem;">{format!("{label} • {}%", fmt_f32_fixed(pct, 0))}</div>
                </div>
                <div style="margin-top: 6px; height: 10px; border-radius: 999px; border: 1px solid var(--border); background: color-mix(in oklab, var(--bg) 70%, transparent); overflow: hidden;">
                    <div
                        style=move || format!("height: 100%; width: {}%; background: {};", fmt_f32_fixed(pct, 2), color)
                    ></div>
                </div>
            </div>
        }
    };

    view! {
        <TooltipPortal store=tooltip_store />
        <div class="app">
            <Show when=move || show_research_disclaimer.get()>
                <div class="launch-modal-overlay">
                    <div class="launch-modal" role="dialog" aria-modal="true">
                        <div class="launch-modal-title">"⚠️ Research Disclaimer"</div>
                        <div class="launch-modal-body">
                            <p style="margin: 0; line-height: 1.7;">
                                "This system was developed with the assistance of Large Language Models (LLMs) under human guidance. "
                                "It is provided as a "<strong>"research demonstration"</strong>" to explore biologically-inspired learning substrates. "
                                "Braine is "<strong>"not production-ready"</strong>" and should not be used for real-world deployment, safety-critical applications, "
                                "or any scenario requiring reliability guarantees. Use at your own discretion for educational and experimental purposes only."
                            </p>
                        </div>
                        <div class="row end wrap" style="justify-content: space-between; gap: 10px;">
                            <div class="subtle">"Shown every refresh."</div>
                            <button
                                class="btn primary"
                                autofocus
                                on:click=move |_| set_show_research_disclaimer.set(false)
                            >
                                "I understand"
                            </button>
                        </div>
                    </div>
                </div>
            </Show>

            <Show when=move || game_info_modal_kind.get().is_some()>
                <div
                    class="modal-overlay"
                    style="z-index: 99999;"
                    on:click={
                        let close = close_game_info_modal.clone();
                        move |_| close()
                    }
                >
                    <div
                        class="modal"
                        role="dialog"
                        aria-modal="true"
                        on:click=move |ev| ev.stop_propagation()
                    >
                        // Modal header with close button - outside the reactive map
                        <div class="modal-head">
                            <div class="modal-title">
                                {move || game_info_modal_kind.get().map(|k| format!("{} {}", k.icon(), k.display_name())).unwrap_or_default()}
                                <span class="subtle" style="margin-left: 10px; font-weight: 700;">
                                    {move || game_info_modal_kind.get().map(|k| k.label()).unwrap_or_default()}
                                </span>
                            </div>
                            <button
                                class="icon-btn"
                                title="Close"
                                on:click={
                                    let close = close_game_info_modal.clone();
                                    move |_| close()
                                }
                            >
                                "×"
                            </button>
                        </div>

                        // Modal content - reactive based on game kind
                        {move || {
                            game_info_modal_kind
                                .get()
                                .map(|kind| {
                                    view! {
                                        <div class="modal-content">
                                            <div class="modal-section">
                                                <div class="modal-section-title">"Overview"</div>
                                                <div class="modal-pre">{kind.description()}</div>
                                            </div>

                                            <div class="modal-section">
                                                <div class="modal-section-title">"What it tests"</div>
                                                <div class="modal-pre">{kind.what_it_tests()}</div>
                                            </div>

                                            <div class="modal-section">
                                                <div class="modal-section-title">"Inputs / Actions"</div>
                                                <div class="modal-pre">{kind.inputs_info()}</div>
                                            </div>

                                            <div class="modal-section">
                                                <div class="modal-section-title">"Reward"</div>
                                                <div class="modal-pre">{kind.reward_info()}</div>
                                            </div>

                                            <div class="modal-section">
                                                <div class="modal-section-title">"Learning objectives"</div>
                                                <div class="modal-pre">{kind.learning_objectives()}</div>
                                            </div>

                                            <Show when=move || kind == GameKind::Text>
                                                <div class="modal-section">
                                                    <div class="modal-section-title">"Text: Task definition"</div>
                                                    <p class="subtle" style="margin: 0 0 10px 0;">
                                                        "These controls rebuild the Text game (vocab + sensors/actions) while keeping the same brain."
                                                    </p>

                                                    <label class="label stack">
                                                        <span>"Corpus 0"</span>
                                                        <textarea
                                                            class="input"
                                                            rows="3"
                                                            prop:value=move || text_corpus0.get()
                                                            on:input=move |ev| set_text_corpus0.set(event_target_value(&ev))
                                                        />
                                                    </label>

                                                    <label class="label stack">
                                                        <span>"Corpus 1"</span>
                                                        <textarea
                                                            class="input"
                                                            rows="3"
                                                            prop:value=move || text_corpus1.get()
                                                            on:input=move |ev| set_text_corpus1.set(event_target_value(&ev))
                                                        />
                                                    </label>

                                                    <div class="row end wrap">
                                                        <label class="label">
                                                            <span>"Max vocab"</span>
                                                            <input
                                                                class="input compact"
                                                                type="number"
                                                                min="2"
                                                                max="512"
                                                                step="1"
                                                                prop:value=move || text_max_vocab.get().to_string()
                                                                on:input=move |ev| {
                                                                    if let Ok(v) = event_target_value(&ev).parse::<u32>() {
                                                                        set_text_max_vocab.set(v.clamp(2, 512));
                                                                    }
                                                                }
                                                            />
                                                        </label>
                                                        <label class="label">
                                                            <span>"Shift every (outcomes)"</span>
                                                            <input
                                                                class="input compact"
                                                                type="number"
                                                                min="1"
                                                                step="1"
                                                                prop:value=move || text_shift_every.get().to_string()
                                                                on:input=move |ev| {
                                                                    if let Ok(v) = event_target_value(&ev).parse::<u32>() {
                                                                        set_text_shift_every.set(v.max(1));
                                                                    }
                                                                }
                                                            />
                                                        </label>

                                                        <button
                                                            class="btn primary"
                                                            on:click=move |_| {
                                                                do_text_apply_corpora_sv.with_value(|f| (f.as_ref())())
                                                            }
                                                        >
                                                            "Apply corpora"
                                                        </button>
                                                    </div>

                                                    <p class="subtle" style="margin: 8px 0 0 0;">
                                                        "After applying, use Run/Step to train on the stream."
                                                    </p>
                                                </div>

                                                <div class="modal-section">
                                                    <div class="modal-section-title">"Text: Prompt training (supervised reward)"</div>
                                                    <p class="subtle" style="margin: 0 0 10px 0;">
                                                        "Walks adjacent byte pairs in the prompt and rewards +1 for predicting the next token, −1 otherwise."
                                                    </p>

                                                    <label class="label stack">
                                                        <span>"Prompt"</span>
                                                        <textarea
                                                            class="input"
                                                            rows="3"
                                                            prop:value=move || text_train_prompt.get()
                                                            on:input=move |ev| set_text_train_prompt.set(event_target_value(&ev))
                                                        />
                                                    </label>

                                                    <div class="row end wrap">
                                                        <label class="label">
                                                            <span>"Regime"</span>
                                                            <input
                                                                class="input compact"
                                                                type="number"
                                                                min="0"
                                                                max="1"
                                                                step="1"
                                                                prop:value=move || text_train_regime.get().to_string()
                                                                on:input=move |ev| {
                                                                    if let Ok(v) = event_target_value(&ev).parse::<u32>() {
                                                                        set_text_train_regime.set(v.clamp(0, 1));
                                                                    }
                                                                }
                                                            />
                                                        </label>

                                                        <button
                                                            class="btn"
                                                            on:click=move |_| {
                                                                do_text_train_prompt_sv.with_value(|f| (f.as_ref())())
                                                            }
                                                        >
                                                            "Train"
                                                        </button>
                                                    </div>
                                                </div>
                                            </Show>
                                        </div>
                                    }
                                    .into_any()
                                })
                                .unwrap_or_else(|| ().into_any())
                        }}
                    </div>
                </div>
            </Show>

            <Topbar
                sidebar_open=sidebar_open
                set_sidebar_open=set_sidebar_open
                gpu_status=gpu_status
                status=status
                gpu_pending=gpu_step_pending
                is_running=is_running
                theme=theme
                set_theme=set_theme
                open_docs=open_docs
                open_analytics=open_analytics
                open_settings=open_settings
            />

            <ToastStack toasts=toasts />

            <div class="content-split lab">
                <Sidebar
                    sidebar_open=sidebar_open
                    set_sidebar_open=set_sidebar_open
                    show_about_page=show_about_page
                    set_show_about_page=set_show_about_page
                    game_kind=game_kind
                    set_game=set_game
                    open_docs=open_docs
                    set_game_info_modal_kind=set_game_info_modal_kind
                />

                <div class="game-area">
                    <SystemErrorBanner system_error=system_error set_system_error=set_system_error />

                    <Show when=move || show_about_page.get()>
                        <div class="stack">
                            <div class="card">
                                <h2 class="card-title">"Docs"</h2>
                                <p class="subtle">"How the substrate works, plus protocol + integration notes."</p>

                                <div class="subtabs" style="margin-top: 12px;">
                                    {AboutSubTab::all()
                                        .iter()
                                        .map(|&tab| {
                                            view! {
                                                <button
                                                    class=move || {
                                                        if about_sub_tab.get() == tab {
                                                            "subtab active"
                                                        } else {
                                                            "subtab"
                                                        }
                                                    }
                                                    on:click=move |_| set_about_sub_tab.set(tab)
                                                >
                                                    {tab.label()}
                                                </button>
                                            }
                                        })
                                        .collect_view()}
                                </div>

                                <div class="stack" style="margin-top: 14px;">
                                    // Overview tab
                                    <Show when=move || about_sub_tab.get() == AboutSubTab::Overview>
                                        <div class="docs-overview-top">
                                            <div style=STYLE_CARD>
                                                <h3 style="margin: 0 0 10px 0; font-size: 1.1rem; color: var(--accent);">"Closed-loop learning substrate"</h3>
                                                <p style="margin: 0; color: var(--text); font-size: 0.92rem; line-height: 1.7;">
                                                    "Braine is a continuously running dynamical system with local plasticity and a scalar reward (neuromodulator). "
                                                    "It is not an LLM: there is no backprop, no global loss, and no token prediction objective."
                                                </p>
                                                <p style="margin: 10px 0 0 0; color: var(--muted); font-size: 0.85rem; line-height: 1.6;">
                                                    {format!("braine v{} • braine_web v{} • bbi format v{}", VERSION_BRAINE, VERSION_BRAINE_WEB, VERSION_BBI_FORMAT)}
                                                </p>
                                            </div>
                                            <div class="docs-overview-stack">
                                                <div style=STYLE_CARD>
                                                    <h3 style="margin: 0 0 10px 0; font-size: 1rem; color: var(--accent);">"What to do"</h3>
                                                    <ol style="margin: 0; padding-left: 20px; color: var(--text); font-size: 0.9rem; line-height: 1.7;">
                                                        <li>"Pick a game (left sidebar)."</li>
                                                        <li>"Press Run; watch last-100 accuracy rise."</li>
                                                        <li>"Tune ε (exploration), α (meaning), and trial ms."</li>
                                                        <li>"Use BrainViz (bottom of dashboard) to inspect structure."</li>
                                                    </ol>
                                                </div>
                                                <div style=STYLE_CARD>
                                                    <h3 style="margin: 0 0 10px 0; font-size: 1rem; color: var(--accent);">"Key idea"</h3>
                                                    <p style="margin: 0; color: var(--text); font-size: 0.9rem; line-height: 1.7;">
                                                        "Learning modifies couplings and attractors; inference reuses the learned dynamics. "
                                                        "Persistence and telemetry are slow side-effects and should never block the step loop."
                                                    </p>
                                                </div>
                                            </div>
                                        </div>

                                        <div class="docs-diagram-wrap docs-wide">
                                            <MermaidDiagram code=ABOUT_OVERVIEW_DIAGRAM max_width_px=920 />
                                        </div>

                                        <p class="docs-diagram-caption">
                                            "The substrate runs continuously: stimuli excite dynamics, actions are read out, and a scalar reward (neuromodulator) gates local learning. Persistence and telemetry are "
                                            <strong>"slow paths"</strong>" that should not block the realtime step loop."
                                        </p>
                                    </Show>

                                // Dynamics tab
                                <Show when=move || about_sub_tab.get() == AboutSubTab::Dynamics>
                                    <div class="docs-masonry">
                                        <div style=STYLE_CARD>
                                            <h3 style="margin: 0 0 10px 0; font-size: 1rem; color: var(--accent);">"Unit State"</h3>
                                            <p style="margin: 0 0 10px 0; color: var(--text); font-size: 0.9rem; line-height: 1.7;">
                                                "Each unit holds only scalars — no vectors or matrices:"
                                            </p>
                                            <div style="background: var(--bg); padding: 12px; border-radius: 8px; font-family: monospace; font-size: 0.85rem; line-height: 1.6; color: var(--text);">
                                                "struct Unit {"<br/>
                                                "  amp: f32,   "<span style="color: var(--muted);">"// activation amplitude"</span><br/>
                                                "  phase: f32, "<span style="color: var(--muted);">"// oscillatory phase [0, 2π)"</span><br/>
                                                "  bias: f32,  "<span style="color: var(--muted);">"// resting potential"</span><br/>
                                                "  decay: f32, "<span style="color: var(--muted);">"// amplitude decay rate"</span><br/>
                                                "}"
                                            </div>
                                        </div>

                                        <div style=STYLE_CARD>
                                            <h3 style="margin: 0 0 10px 0; font-size: 1rem; color: var(--accent);">"Update Rule"</h3>
                                            <p style="margin: 0 0 10px 0; color: var(--text); font-size: 0.9rem; line-height: 1.7;">
                                                "At each tick, each unit updates from:"
                                            </p>
                                            <ul style="margin: 0; padding-left: 20px; color: var(--text); font-size: 0.9rem; line-height: 1.6;">
                                                <li>"Neighbor influence (sparse couplings)"</li>
                                                <li>"External stimulus injection"</li>
                                                <li>"Global inhibition (competition)"</li>
                                                <li>"Decay (forgetting at state level)"</li>
                                                <li>"Bounded noise (exploration)"</li>
                                            </ul>
                                        </div>

                                        <div style=STYLE_CARD>
                                            <h3 style="margin: 0 0 10px 0; font-size: 1rem; color: var(--accent);">"Mathematical Model"</h3>
                                            <div style="background: var(--bg); padding: 12px; border-radius: 8px; font-size: 0.9rem; line-height: 1.8; color: var(--text);">
                                                <div style="margin-bottom: 8px;">
                                                    <strong>"Amplitude update:"</strong>
                                                </div>
                                                <div style="font-family: serif; font-style: italic; padding-left: 12px; margin-bottom: 12px;">
                                                    "amp"<sub>"i"</sub>"(t+1) = amp"<sub>"i"</sub>"(t) · (1 - decay) + input"<sub>"i"</sub>" + Σ"<sub>"j"</sub>" w"<sub>"ij"</sub>" · amp"<sub>"j"</sub>" · cos(Δφ)"
                                                </div>
                                                <div style="margin-bottom: 8px;">
                                                    <strong>"Phase coupling:"</strong>
                                                </div>
                                                <div style="font-family: serif; font-style: italic; padding-left: 12px;">
                                                    "Δφ"<sub>"ij"</sub>" = phase"<sub>"i"</sub>" - phase"<sub>"j"</sub>
                                                </div>
                                            </div>
                                        </div>

                                        <div style=STYLE_CARD>
                                            <h3 style="margin: 0 0 10px 0; font-size: 1rem; color: var(--accent);">"Emergent Behavior"</h3>
                                            <ul style="margin: 0; padding-left: 20px; color: var(--text); font-size: 0.9rem; line-height: 1.7;">
                                                <li><strong>"Stable attractors"</strong>" — habits, identity patterns"</li>
                                                <li><strong>"Context-dependent recall"</strong>" — partial cues trigger full patterns"</li>
                                                <li><strong>"Deterministic far from threshold"</strong>" — reliable actions"</li>
                                                <li><strong>"Stochastic near threshold"</strong>" — exploration"</li>
                                            </ul>
                                        </div>
                                    </div>

                                    // Sparse storage diagram
                                    <div style=STYLE_CARD>
                                        <h3 style="margin: 0 0 10px 0; font-size: 1rem; color: var(--accent);">"Sparse Connection Storage (CSR Format)"</h3>
                                        <p style="margin: 0 0 10px 0; color: var(--text); font-size: 0.9rem; line-height: 1.7;">
                                            "Connections are stored in Compressed Sparse Row (CSR) format for cache efficiency:"
                                        </p>
                                        <div style="background: var(--bg); padding: 12px; border-radius: 8px; font-family: monospace; font-size: 0.85rem; line-height: 1.6; color: var(--text);">
                                            "offsets: Vec<usize>  "<span style="color: var(--muted);">"// index into targets/weights for each unit"</span><br/>
                                            "targets: Vec<UnitId> "<span style="color: var(--muted);">"// target unit IDs"</span><br/>
                                            "weights: Vec<f32>    "<span style="color: var(--muted);">"// connection weights"</span>
                                        </div>
                                    </div>
                                </Show>

                                // Learning tab
                                <Show when=move || about_sub_tab.get() == AboutSubTab::Learning>
                                    <div class="docs-masonry">
                                        <div style=STYLE_CARD>
                                            <h3 style="margin: 0 0 10px 0; font-size: 1rem; color: var(--accent);">"Three-Factor Hebbian Learning"</h3>
                                            <p style="margin: 0 0 10px 0; color: var(--text); font-size: 0.9rem; line-height: 1.7;">
                                                "Learning is local and event-based. Weight changes require:"
                                            </p>
                                            <ol style="margin: 0; padding-left: 20px; color: var(--text); font-size: 0.9rem; line-height: 1.7;">
                                                <li><strong style="color: var(--accent);">"Pre-synaptic activity"</strong>" — source unit active"</li>
                                                <li><strong style="color: var(--accent);">"Post-synaptic activity"</strong>" — target unit active"</li>
                                                <li><strong style="color: var(--accent);">"Neuromodulator"</strong>" — reward/salience signal"</li>
                                            </ol>
                                        </div>

                                        <div style=STYLE_CARD>
                                            <h3 style="margin: 0 0 10px 0; font-size: 1rem; color: var(--accent);">"Learning Rule"</h3>
                                            <div style="background: var(--bg); padding: 12px; border-radius: 8px; font-size: 0.9rem; line-height: 1.8; color: var(--text);">
                                                <div style="margin-bottom: 8px;">
                                                    <strong>"Weight update:"</strong>
                                                </div>
                                                <div style="font-family: serif; font-style: italic; padding-left: 12px; margin-bottom: 12px;">
                                                    "Δw"<sub>"ij"</sub>" = η · neuromod · amp"<sub>"i"</sub>" · amp"<sub>"j"</sub>" · cos(Δφ"<sub>"ij"</sub>")"
                                                </div>
                                                <div style="color: var(--muted); font-size: 0.85rem;">
                                                    "• Phase-aligned → strengthen (LTP)"<br/>
                                                    "• Phase-misaligned → weaken (LTD)"<br/>
                                                    "• High neuromodulator → faster learning"
                                                </div>
                                            </div>
                                        </div>

                                        <div style=STYLE_CARD>
                                            <h3 style="margin: 0 0 10px 0; font-size: 1rem; color: var(--accent);">"Forgetting & Pruning"</h3>
                                            <p style="margin: 0 0 10px 0; color: var(--text); font-size: 0.9rem; line-height: 1.7;">
                                                "Connections decay continuously. This enforces:"
                                            </p>
                                            <ul style="margin: 0; padding-left: 20px; color: var(--text); font-size: 0.9rem; line-height: 1.7;">
                                                <li>"Bounded memory usage"</li>
                                                <li>"Relevance to recent experience"</li>
                                                <li>"Energy efficiency (fewer couplings)"</li>
                                            </ul>
                                            <div style="margin-top: 10px; background: var(--bg); padding: 8px; border-radius: 6px; font-size: 0.85rem; color: var(--muted);">
                                                "Tiny weights (|w| < threshold) are pruned to zero."
                                            </div>
                                        </div>

                                        <div style=STYLE_CARD>
                                            <h3 style="margin: 0 0 10px 0; font-size: 1rem; color: var(--accent);">"Learning Mechanisms"</h3>
                                            <div style="display: grid; grid-template-columns: auto 1fr; gap: 4px 12px; font-size: 0.85rem;">
                                                <span style="color: #4ade80;">"✓"</span><span style="color: var(--text);">"Three-Factor Hebbian"</span>
                                                <span style="color: #4ade80;">"✓"</span><span style="color: var(--text);">"One-Shot Imprinting"</span>
                                                <span style="color: #4ade80;">"✓"</span><span style="color: var(--text);">"Neurogenesis"</span>
                                                <span style="color: #4ade80;">"✓"</span><span style="color: var(--text);">"Structural Pruning"</span>
                                                <span style="color: #4ade80;">"✓"</span><span style="color: var(--text);">"Dream Replay"</span>
                                                <span style="color: #4ade80;">"✓"</span><span style="color: var(--text);">"Attention Gating"</span>
                                                <span style="color: #4ade80;">"✓"</span><span style="color: var(--text);">"Burst-Mode Learning"</span>
                                                <span style="color: var(--muted);">"○"</span><span style="color: var(--muted);">"STDP (proposed)"</span>
                                                <span style="color: var(--muted);">"○"</span><span style="color: var(--muted);">"Meta-Learning (proposed)"</span>
                                            </div>
                                        </div>
                                    </div>

                                    // Learning hierarchy diagram
                                    <div style=STYLE_CARD>
                                        <h3 style="margin: 0 0 12px 0; font-size: 1rem; color: var(--accent);">"Learning Hierarchy"</h3>
                                        <div class="docs-diagram-wrap">
                                            <MermaidDiagram code=ABOUT_LEARNING_HIERARCHY_DIAGRAM max_width_px=500 />
                                        </div>
                                    </div>
                                </Show>

                                // Memory tab
                                <Show when=move || about_sub_tab.get() == AboutSubTab::Memory>
                                    <div class="docs-masonry">
                                        <div style=STYLE_CARD>
                                            <h3 style="margin: 0 0 10px 0; font-size: 1rem; color: var(--accent);">"Causal Memory"</h3>
                                            <p style="margin: 0 0 10px 0; color: var(--text); font-size: 0.9rem; line-height: 1.7;">
                                                "\"Meaning\" is learned links between:"
                                            </p>
                                            <ul style="margin: 0; padding-left: 20px; color: var(--text); font-size: 0.9rem; line-height: 1.7;">
                                                <li>"What was perceived (stimulus symbol)"</li>
                                                <li>"What the system did (action symbol)"</li>
                                                <li>"What happened next (reward/outcome)"</li>
                                            </ul>
                                            <div style="margin-top: 10px; background: var(--bg); padding: 8px; border-radius: 6px; font-size: 0.85rem; color: var(--muted);">
                                                "Cheap temporal causal memory: A → B transition counts"
                                            </div>
                                        </div>

                                        <div style=STYLE_CARD>
                                            <h3 style="margin: 0 0 10px 0; font-size: 1rem; color: var(--accent);">"Symbol Recording"</h3>
                                            <p style="margin: 0 0 10px 0; color: var(--text); font-size: 0.9rem; line-height: 1.7;">
                                                "During each observation, symbols are recorded:"
                                            </p>
                                            <div style="background: var(--bg); padding: 12px; border-radius: 8px; font-family: monospace; font-size: 0.85rem; line-height: 1.6; color: var(--text);">
                                                "note_action(\"left\");"<br/>
                                                "note_pair(\"spot\", \"left\");"<br/>
                                                "note_compound_symbol([\"ctx\", \"action\"]);"
                                            </div>
                                        </div>

                                        <div style=STYLE_CARD>
                                            <h3 style="margin: 0 0 10px 0; font-size: 1rem; color: var(--accent);">"Bounded Context"</h3>
                                            <p style="margin: 0 0 10px 0; color: var(--text); font-size: 0.9rem; line-height: 1.7;">
                                                "For long-running deployments, causal memory is kept bounded:"
                                            </p>
                                            <ul style="margin: 0; padding-left: 20px; color: var(--text); font-size: 0.9rem; line-height: 1.7;">
                                                <li>"Symbol and edge counts decay continuously"</li>
                                                <li>"Near-zero entries are periodically pruned"</li>
                                                <li>"Prevents unbounded hashmap growth"</li>
                                            </ul>
                                        </div>

                                        <div style=STYLE_CARD>
                                            <h3 style="margin: 0 0 10px 0; font-size: 1rem; color: var(--accent);">"Muscle Memory / Savings"</h3>
                                            <p style="margin: 0 0 10px 0; color: var(--text); font-size: 0.9rem; line-height: 1.7;">
                                                "Previously-seen stimuli are re-learned faster:"
                                            </p>
                                            <ul style="margin: 0; padding-left: 20px; color: var(--text); font-size: 0.9rem; line-height: 1.7;">
                                                <li>"Weak non-zero trace for sensor↔concept links"</li>
                                                <li>"Links decay but aren't pruned to zero"</li>
                                                <li>"On re-exposure, associations \"snap back\" faster"</li>
                                            </ul>
                                        </div>
                                    </div>

                                    // Memory structure diagram
                                    <div style=STYLE_CARD>
                                        <h3 style="margin: 0 0 12px 0; font-size: 1rem; color: var(--accent);">"Memory Structure"</h3>
                                        <div class="docs-diagram-wrap">
                                            <MermaidDiagram code=ABOUT_MEMORY_STRUCTURE_DIAGRAM max_width_px=700 />
                                        </div>
                                    </div>
                                </Show>

                                // The Math Behind tab
                                <Show when=move || about_sub_tab.get() == AboutSubTab::MathBehind>
                                    <div class="stack">
                                        <div style=STYLE_CARD>
                                            <h3 style="margin: 0 0 8px 0; font-size: 1rem; color: var(--accent);">"The Math Behind"</h3>
                                            <p style="margin: 0; color: var(--muted); font-size: 0.85rem; line-height: 1.6;">
                                                "This tab renders the repo doc at doc/maths/the-math-behind.md. "
                                                "Markdown is rendered to HTML; fenced mermaid blocks (if any) render as diagrams."
                                            </p>
                                        </div>

                                        <div class="card">
                                            <div class="docs-markdown" inner_html=move || math_behind_html.get_value()></div>
                                        </div>
                                    </div>
                                </Show>

                                // Architecture tab
                                <Show when=move || about_sub_tab.get() == AboutSubTab::Architecture>
                                    <div class="docs-masonry">
                                        <div style=STYLE_CARD>
                                            <h3 style="margin: 0 0 10px 0; font-size: 1rem; color: var(--accent);">"Brain Image Format (BBI)"</h3>
                                            <p style="margin: 0 0 10px 0; color: var(--text); font-size: 0.9rem; line-height: 1.7;">
                                                "Custom, versioned binary format for persistence:"
                                            </p>
                                            <div style="background: var(--bg); padding: 12px; border-radius: 8px; font-family: monospace; font-size: 0.85rem; line-height: 1.6; color: var(--text);">
                                                "Header: BRAINE01 (8 bytes)"<br/>
                                                "Version: u32 (currently 2)"<br/><br/>
                                                "Chunks (LZ4 compressed):"<br/>
                                                "  CFG0 - BrainConfig"<br/>
                                                "  PRNG - RNG state"<br/>
                                                "  STAT - age_steps, neuromod"<br/>
                                                "  UNIT - units + connections"<br/>
                                                "  GRPS - sensor/action groups"<br/>
                                                "  SYMB - symbol table"<br/>
                                                "  CAUS - causal memory"
                                            </div>
                                        </div>

                                        <div style=STYLE_CARD>
                                            <h3 style="margin: 0 0 10px 0; font-size: 1rem; color: var(--accent);">"Design Goals"</h3>
                                            <ul style="margin: 0; padding-left: 20px; color: var(--text); font-size: 0.9rem; line-height: 1.7;">
                                                <li><strong>"Versioned"</strong>" — new chunks can be added"</li>
                                                <li><strong>"Capacity-aware"</strong>" — can enforce byte budgets"</li>
                                                <li><strong>"Compact"</strong>" — LZ4 compression"</li>
                                                <li><strong>"Forward-compatible"</strong>" — unknown chunks skipped"</li>
                                            </ul>
                                        </div>

                                        <div style=STYLE_CARD>
                                            <h3 style="margin: 0 0 10px 0; font-size: 1rem; color: var(--accent);">"Research Landscape"</h3>
                                            <p style="margin: 0 0 10px 0; color: var(--text); font-size: 0.9rem; line-height: 1.7;">
                                                "Braine draws from several established areas:"
                                            </p>
                                            <ul style="margin: 0; padding-left: 20px; color: var(--text); font-size: 0.9rem; line-height: 1.6;">
                                                <li>"Attractor networks"</li>
                                                <li>"Continuous Hopfield dynamics"</li>
                                                <li>"Reservoir computing / LSMs"</li>
                                                <li>"Coupled oscillator models"</li>
                                                <li>"Local plasticity (Hebb/STDP)"</li>
                                                <li>"Embodied cognition"</li>
                                            </ul>
                                        </div>

                                        <div style=STYLE_CARD>
                                            <h3 style="margin: 0 0 10px 0; font-size: 1rem; color: var(--accent);">"How We Differ from LLMs"</h3>
                                            <ul style="margin: 0; padding-left: 20px; color: var(--text); font-size: 0.9rem; line-height: 1.7;">
                                                <li>"No token prediction objective"</li>
                                                <li>"No gradient training on corpora"</li>
                                                <li>"No dense embeddings/matrices"</li>
                                                <li>"Online, local, scalar updates"</li>
                                                <li>"Edge-first (bounded compute/memory)"</li>
                                            </ul>
                                        </div>
                                    </div>

                                    // Component diagram - CURRENT ARCHITECTURE
                                    <div style=STYLE_CARD>
                                        <h3 style="margin: 0 0 12px 0; font-size: 1rem; color: var(--accent);">"Current Architecture"</h3>
                                        <p style="margin: 0 0 12px 0; color: var(--muted); font-size: 0.85rem; line-height: 1.6;">
                                            "The web version hosts Braine entirely in-memory within the browser (WASM). Desktop and CLI connect to a local daemon."
                                        </p>
                                        <div class="docs-diagram-wrap">
                                            <MermaidDiagram code=ABOUT_CURRENT_ARCHITECTURE_DIAGRAM max_width_px=900 />
                                        </div>
                                    </div>

                                    // Future Architecture Vision
                                    <div style=STYLE_CARD>
                                        <h3 style="margin: 0 0 12px 0; font-size: 1rem; color: var(--accent);">"Future Vision: Centralized Brain"</h3>
                                        <p style="margin: 0 0 12px 0; color: var(--muted); font-size: 0.85rem; line-height: 1.6;">
                                            "Planned architecture: a central daemon hosts the authoritative Brain, while edge clients maintain local copies that sync in real-time."
                                        </p>
                                        <div class="docs-diagram-wrap">
                                            <MermaidDiagram code=ABOUT_FUTURE_ARCHITECTURE_DIAGRAM max_width_px=900 />
                                        </div>
                                    </div>
                                </Show>

                                // Applications sub-tab - real-world use cases
                                <Show when=move || about_sub_tab.get() == AboutSubTab::Applications>
                                    <div style="padding: 14px; background: rgba(251, 191, 36, 0.1); border: 1px solid rgba(251, 191, 36, 0.3); border-radius: 12px;">
                                        <h3 style="margin: 0 0 10px 0; font-size: 1rem; color: #fbbf24;">"Important note"</h3>
                                        <p style="margin: 0; color: var(--text); font-size: 0.9rem; line-height: 1.7;">
                                            "These are potential application categories, not claims of deployed or validated systems. Braine is a research substrate; any real-world use would require careful engineering, evaluation, and safety work."
                                        </p>
                                    </div>

                                    <div class="docs-masonry">
                                        <div style=STYLE_CARD>
                                            <h3 style="margin: 0 0 10px 0; font-size: 1rem; color: var(--accent);">"🎛️ Adaptive Control (small loops)"</h3>
                                            <ul style="margin: 0; padding-left: 20px; color: var(--text); font-size: 0.9rem; line-height: 1.7;">
                                                <li>"Online tuning from scalar reward"</li>
                                                <li>"Handles regime shifts (distribution changes)"</li>
                                                <li>"Works with low-dimensional sensors/actions"</li>
                                            </ul>
                                            <div style="margin-top: 8px; color: var(--muted); font-size: 0.85rem; line-height: 1.6;">
                                                "Examples: simple process control, scheduling heuristics, UI latency/throughput tuning."
                                            </div>
                                        </div>

                                        <div style=STYLE_CARD>
                                            <h3 style="margin: 0 0 10px 0; font-size: 1rem; color: var(--accent);">"🎮 Interactive Experiences"</h3>
                                            <ul style="margin: 0; padding-left: 20px; color: var(--text); font-size: 0.9rem; line-height: 1.7;">
                                                <li>"NPC behavior adaptation within a session"</li>
                                                <li>"Persistent state across runs (when persisted)"</li>
                                                <li>"Interpretable structure via BrainViz"</li>
                                            </ul>
                                            <div style="margin-top: 8px; color: var(--muted); font-size: 0.85rem; line-height: 1.6;">
                                                "Examples: adaptive opponents, tutorials that adjust difficulty, toy worlds for learning dynamics."
                                            </div>
                                        </div>

                                        <div style=STYLE_CARD>
                                            <h3 style="margin: 0 0 10px 0; font-size: 1rem; color: var(--accent);">"🌐 Edge Personalization (privacy-first)"</h3>
                                            <ul style="margin: 0; padding-left: 20px; color: var(--text); font-size: 0.9rem; line-height: 1.7;">
                                                <li>"On-device adaptation (no cloud required)"</li>
                                                <li>"Learns from local interaction signals"</li>
                                                <li>"Can run in the browser (WASM)"</li>
                                            </ul>
                                            <div style="margin-top: 8px; color: var(--muted); font-size: 0.85rem; line-height: 1.6;">
                                                "Examples: preference learning, lightweight anomaly flags, personalization for offline-first apps."
                                            </div>
                                        </div>

                                        <div style=STYLE_CARD>
                                            <h3 style="margin: 0 0 10px 0; font-size: 1rem; color: var(--accent);">"🧪 Research + Education"</h3>
                                            <ul style="margin: 0; padding-left: 20px; color: var(--text); font-size: 0.9rem; line-height: 1.7;">
                                                <li>"Inspectable learning (local updates, no backprop)"</li>
                                                <li>"Stress tests with controlled games (Spot/Bandit/SpotXY/Pong)"</li>
                                                <li>"Long-running dynamics + persistence"</li>
                                            </ul>
                                            <div style="margin-top: 8px; color: var(--muted); font-size: 0.85rem; line-height: 1.6;">
                                                "Examples: neuromodulation studies, memory bottleneck experiments, demos for learning systems."
                                            </div>
                                        </div>
                                    </div>

                                    <div style=STYLE_CARD>
                                        <h3 style="margin: 0 0 10px 0; font-size: 1rem; color: var(--accent);">"Capability summary"</h3>
                                        <ul style="margin: 0; padding-left: 20px; color: var(--text); font-size: 0.9rem; line-height: 1.7;">
                                            <li>"Online learning from bounded scalar reward"</li>
                                            <li>"Context-conditioned meaning + causal memory"</li>
                                            <li>"Bounded compute: step-by-step dynamics (no batch training)"</li>
                                            <li>"Persistence of long-lived structure (BBI / IndexedDB)"</li>
                                        </ul>
                                    </div>
                                </Show>

                                // LLM Integration sub-tab
                                <Show when=move || about_sub_tab.get() == AboutSubTab::LlmIntegration>
                                    <div style=STYLE_CARD>
                                        <h3 style="margin: 0 0 10px 0; font-size: 1rem; color: var(--accent);">"Why integrate an LLM at all?"</h3>
                                        <p style="margin: 0; color: var(--text); font-size: 0.9rem; line-height: 1.7;">
                                            "Braine is not an LLM and should not be shaped like one. The integration point exists to let an external planner/analyst "
                                            <strong>"observe"</strong>" bounded state summaries and propose "
                                            <strong>"bounded nudges"</strong>" to a small set of safe knobs. "
                                            "Actions still come from Braine’s learned dynamics."
                                        </p>
                                    </div>

                                    <div class="docs-masonry">
                                        <div style=STYLE_CARD>
                                            <h3 style="margin: 0 0 10px 0; font-size: 1rem; color: var(--accent);">"Contract"</h3>
                                            <ul style="margin: 0; padding-left: 20px; color: var(--text); font-size: 0.9rem; line-height: 1.7;">
                                                <li><strong>"Braine → LLM"</strong>" : structured "<span style="font-family: monospace;">"AdvisorContext"</span>" (bounded)"</li>
                                                <li><strong>"LLM → Braine"</strong>" : structured "<span style="font-family: monospace;">"AdvisorAdvice"</span>" (bounded)"</li>
                                                <li><strong>"No action selection"</strong>" : the LLM never chooses left/right/up/down"</li>
                                                <li><strong>"Daemon clamps"</strong>" : advice is validated and clamped before applying"</li>
                                            </ul>
                                        </div>

                                        <div style=STYLE_CARD>
                                            <h3 style="margin: 0 0 10px 0; font-size: 1rem; color: var(--accent);">"KGF fit"</h3>
                                            <p style="margin: 0; color: var(--text); font-size: 0.9rem; line-height: 1.7;">
                                                "KGF here is the idea of keeping a "
                                                <strong>"symbolic/graph boundary"</strong>" at the frame edge (stimuli/actions/context keys), while the substrate inside remains dynamical and locally plastic. "
                                                "The LLM reads that boundary (context_key + metrics) and proposes safe parameter nudges; Braine consolidates learned structure internally."
                                            </p>
                                        </div>
                                    </div>

                                    <div style=STYLE_CARD>
                                        <h3 style="margin: 0 0 10px 0; font-size: 1rem; color: var(--accent);">"Data Flow"</h3>
                                        <div class="docs-diagram-wrap">
                                            <MermaidDiagram code=ABOUT_LLM_DATAFLOW max_width_px=920 />
                                        </div>
                                    </div>

                                    <div class="docs-masonry">
                                        <div style=STYLE_CARD>
                                            <h3 style="margin: 0 0 10px 0; font-size: 1rem; color: var(--accent);">"Sample: AdvisorContext"</h3>
                                            <div style="color: var(--muted); font-size: 0.85rem; line-height: 1.6; margin-bottom: 8px;">
                                                "Request (client → daemon):"
                                            </div>
                                            <pre style="margin: 0; white-space: pre-wrap; background: var(--bg); padding: 12px; border-radius: 8px; font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, 'Liberation Mono', 'Courier New', monospace; font-size: 0.82rem; line-height: 1.6; color: var(--text);">{ABOUT_LLM_ADVISOR_CONTEXT_REQ}</pre>
                                            <div style="color: var(--muted); font-size: 0.85rem; line-height: 1.6; margin: 12px 0 8px 0;">
                                                "Response (daemon → client):"
                                            </div>
                                            <pre style="margin: 0; white-space: pre-wrap; background: var(--bg); padding: 12px; border-radius: 8px; font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, 'Liberation Mono', 'Courier New', monospace; font-size: 0.82rem; line-height: 1.6; color: var(--text);">{ABOUT_LLM_ADVISOR_CONTEXT_RESP}</pre>
                                        </div>

                                        <div style=STYLE_CARD>
                                            <h3 style="margin: 0 0 10px 0; font-size: 1rem; color: var(--accent);">"Sample: AdvisorApply"</h3>
                                            <p style="margin: 0 0 10px 0; color: var(--text); font-size: 0.9rem; line-height: 1.7;">
                                                "The LLM returns bounded advice. The daemon validates/clamps it and applies it as a temporary configuration change (TTL)."
                                            </p>
                                            <pre style="margin: 0; white-space: pre-wrap; background: var(--bg); padding: 12px; border-radius: 8px; font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, 'Liberation Mono', 'Courier New', monospace; font-size: 0.82rem; line-height: 1.6; color: var(--text);">{ABOUT_LLM_ADVISOR_APPLY_REQ}</pre>
                                            <div style="margin-top: 10px; color: var(--muted); font-size: 0.85rem; line-height: 1.6;">
                                                "Typical evaluation loop: use the Replay game so the context_key is stable (e.g. replay::spot_lr_small), then compare last_100_rate before/after advice over a fixed number of trials."
                                            </div>
                                        </div>
                                    </div>
                                </Show>

                                // Braine APIs sub-tab (high-level, categorized)
                                <Show when=move || about_sub_tab.get() == AboutSubTab::Apis>
                                    <div style=STYLE_CARD>
                                        <h3 style="margin: 0 0 10px 0; font-size: 1rem; color: var(--accent);">"Braine APIs (By Category)"</h3>
                                        <p style="margin: 0; color: var(--muted); font-size: 0.9rem; line-height: 1.7;">
                                            "Braine’s long-lived state is owned by a central daemon (brained). Clients (desktop/web/edge) talk to it over newline-delimited JSON on TCP 127.0.0.1:9876. "
                                            "This section lists the core API categories and the most important request/response shapes."
                                        </p>
                                    </div>

                                    <div class="docs-masonry">
                                        <div style=STYLE_CARD>
                                            <h3 style="margin: 0 0 10px 0; font-size: 1rem; color: var(--accent);">"General"</h3>
                                            <div style="font-family: monospace; font-size: 0.82rem; line-height: 1.7; background: var(--bg); padding: 12px; border-radius: 8px;">
                                                <div><strong>"GetState"</strong>" → StateSnapshot (UI snapshot)"</div>
                                                <div style="color: var(--muted);">"input: { type: \"GetState\" }"</div>
                                                <div style="color: var(--muted);">"output: { type: \"State\", running, game, brain_stats, ... }"</div>
                                                <div style="margin-top: 8px;"><strong>"Start/Stop"</strong>" → Success/Error"</div>
                                                <div style="color: var(--muted);">"input: { type: \"Start\" } / { type: \"Stop\" }"</div>
                                            </div>
                                        </div>

                                        <div style=STYLE_CARD>
                                            <h3 style="margin: 0 0 10px 0; font-size: 1rem; color: var(--accent);">"Inference"</h3>
                                            <p style="margin: 0 0 10px 0; color: var(--muted); font-size: 0.85rem; line-height: 1.6;">
                                                "Read-only action recommendation. Runs on a cloned brain and uses inference-only stepping (no imprinting, no Hebbian learning, no forgetting/pruning)."
                                            </p>
                                            <div style="font-family: monospace; font-size: 0.82rem; line-height: 1.7; background: var(--bg); padding: 12px; border-radius: 8px;">
                                                <div><strong>"InferActionScores"</strong>" → InferActionScores"</div>
                                                <div style="color: var(--muted);">"input: { type: \"InferActionScores\", context_key?, stimuli?: [{ name, strength }], steps?, meaning_alpha? }"</div>
                                                <div style="color: var(--muted);">"output: { type: \"InferActionScores\", context_key, action_scores: [{ name, score, meaning_*... }] }"</div>
                                            </div>
                                        </div>

                                        <div style=STYLE_CARD>
                                            <h3 style="margin: 0 0 10px 0; font-size: 1rem; color: var(--accent);">"Diagnostics"</h3>
                                            <p style="margin: 0 0 10px 0; color: var(--muted); font-size: 0.85rem; line-height: 1.6;">
                                                "Read-only endpoints intended for dashboards/monitoring."</p>
                                            <div style="font-family: monospace; font-size: 0.82rem; line-height: 1.7; background: var(--bg); padding: 12px; border-radius: 8px;">
                                                <div><strong>"DiagGet"</strong>" → Diagnostics"</div>
                                                <div style="color: var(--muted);">"input: { type: \"DiagGet\" }"</div>
                                                <div style="color: var(--muted);">"output: { type: \"Diagnostics\", brain_stats, storage, frame, running }"</div>
                                                <div style="margin-top: 8px;"><strong>"GetGraph"</strong>" → Graph (causal/meaning graphs)"</div>
                                                <div style="color: var(--muted);">"input: { type: \"GetGraph\", kind, max_nodes, max_edges }"</div>
                                            </div>
                                        </div>

                                        <div style=STYLE_CARD>
                                            <h3 style="margin: 0 0 10px 0; font-size: 1rem; color: var(--accent);">"Configuration"</h3>
                                            <p style="margin: 0 0 10px 0; color: var(--muted); font-size: 0.85rem; line-height: 1.6;">
                                                "Write endpoints that change runtime knobs (separated so we can later add auth/rate-limits)."</p>
                                            <div style="font-family: monospace; font-size: 0.82rem; line-height: 1.7; background: var(--bg); padding: 12px; border-radius: 8px;">
                                                <div><strong>"CfgGet"</strong>" → Config"</div>
                                                <div style="color: var(--muted);">"input: { type: \"CfgGet\" }"</div>
                                                <div style="color: var(--muted);">"output: { type: \"Config\", exploration_eps, meaning_alpha, target_fps, trial_period_ms, max_units_limit }"</div>
                                                <div style="margin-top: 8px;"><strong>"CfgSet"</strong>" → Success/Error"</div>
                                                <div style="color: var(--muted);">"input: { type: \"CfgSet\", exploration_eps?, target_fps?, trial_period_ms?, max_units? }"</div>
                                            </div>
                                        </div>

                                        <div style=STYLE_CARD>
                                            <h3 style="margin: 0 0 10px 0; font-size: 1rem; color: var(--accent);">"Sync (Master ↔ Edge/Child)"</h3>
                                            <p style="margin: 0 0 10px 0; color: var(--muted); font-size: 0.85rem; line-height: 1.6;">
                                                "Edge/child brains can propose structural updates back to the master. Today this is a bounded sparse delta over connection weights (top-K)."</p>
                                            <div style="font-family: monospace; font-size: 0.82rem; line-height: 1.7; background: var(--bg); padding: 12px; border-radius: 8px;">
                                                <div><strong>"SyncGetInfo"</strong>" → SyncInfo"</div>
                                                <div style="color: var(--muted);">"input: { type: \"SyncGetInfo\" }"</div>
                                                <div style="color: var(--muted);">"output: { type: \"SyncInfo\", fingerprint, weights_len, unit_count, age_steps }"</div>
                                                <div style="margin-top: 8px;"><strong>"SyncApplyDelta"</strong>" → SyncApplied/Error"</div>
                                                <div style="color: var(--muted);">"input: { type: \"SyncApplyDelta\", delta, expected_weights_len, expected_fingerprint, delta_max?, autosave? }"</div>
                                                <div style="color: var(--muted);">"output: { type: \"SyncApplied\", applied_edges, saved }"</div>
                                            </div>
                                        </div>
                                    </div>

                                    <div style="padding: 14px; background: rgba(251, 191, 36, 0.1); border: 1px solid rgba(251, 191, 36, 0.3); border-radius: 12px;">
                                        <h3 style="margin: 0 0 10px 0; font-size: 1rem; color: #fbbf24;">"Security (Future)"</h3>
                                        <p style="margin: 0; color: var(--text); font-size: 0.85rem; line-height: 1.7;">
                                            "Planned: authentication, per-category ACLs, request budgets, and audit logs for all Config/Sync endpoints."
                                        </p>
                                    </div>
                                </Show>
                            </div>
                        </div>
                    </div>
                    </Show>

                    <Show when=move || !show_about_page.get()>
                        // Controls bar
                        <div class="controls">
                            <button class="btn" on:click=move |_| (do_tick_sv.get_value())()>"⏯ Step"</button>
                            <button class=move || if is_running.get() { "btn" } else { "btn primary" } on:click=move |_| (do_start_sv.get_value())()>"▶ Run"</button>
                            <button class="btn" on:click=move |_| (do_stop_sv.get_value())()>"⏹ Stop"</button>
                            <button class="btn" on:click=move |_| (do_reset_sv.get_value())()>"↺ Reset"</button>
                            <div class="spacer"></div>
                            <label class="label">
                                    <span title="Trial period in milliseconds (one decision/reward per trial). Larger = slower game loop.">"Trial ms"</span>
                                <input
                                    type="number"
                                min="10"
                                max="60000"
                                class="input compact"
                                    title="Trial period in milliseconds (one decision/reward per trial)."
                                prop:value=move || trial_period_ms.get().to_string()
                                on:input=move |ev| {
                                    let v = event_target_value(&ev);
                                    if let Ok(n) = v.parse::<u32>() {
                                        set_trial_period_ms.set(n.clamp(10, 60_000));
                                    }
                                }
                            />
                        </label>
                        <label class="label">
                                <span title="Exploration rate (epsilon). With probability ε, choose a random action instead of the best-scoring one.">"ε"</span>
                            <input
                                type="number"
                                min="0"
                                max="1"
                                step="0.01"
                                class="input compact"
                                    title="Exploration rate (epsilon): probability of taking a random action."
                                prop:value=move || fmt_f32_fixed(exploration_eps.get(), 2)
                                on:input=move |ev| {
                                    let v = event_target_value(&ev);
                                    if let Ok(x) = v.parse::<f32>() {
                                        set_exploration_eps.set(x.clamp(0.0, 1.0));
                                    }
                                }
                            />
                        </label>
                        <label class="label">
                                <span title="Meaning blending (alpha). Higher α puts more weight on learned meaning/association memory when ranking actions.">"α"</span>
                            <input
                                type="number"
                                min="0"
                                max="30"
                                step="0.5"
                                class="input compact"
                                    title="Meaning alpha: weighting for meaning-based action ranking (higher = stronger meaning influence)."
                                prop:value=move || fmt_f32_fixed(meaning_alpha.get(), 1)
                                on:input=move |ev| {
                                    let v = event_target_value(&ev);
                                    if let Ok(x) = v.parse::<f32>() {
                                        set_meaning_alpha.set(x.clamp(0.0, 30.0));
                                    }
                                }
                            />
                        </label>
                    </div>

                    <div class="game-topdash">
                        <div class="game-topdash-bar">
                            <div class="game-topdash-metrics">
                                <div class="metric-chip">
                                    <span class="metric-k">"Steps"</span>
                                    <span class="metric-v mono">{move || steps.get().to_string()}</span>
                                </div>
                                <div class="metric-chip">
                                    <span class="metric-k">"Trials"</span>
                                    <span class="metric-v mono">{move || trials.get().to_string()}</span>
                                </div>
                                <div class="metric-chip">
                                    <span class="metric-k">"Last-100"</span>
                                    <span class=move || {
                                        let r = recent_rate.get();
                                        if r >= 0.85 {
                                            "metric-v good"
                                        } else if r >= 0.70 {
                                            "metric-v warn"
                                        } else {
                                            "metric-v accent"
                                        }
                                    }>
                                        {move || format!("{}%", fmt_f32_fixed(recent_rate.get() * 100.0, 0))}
                                    </span>
                                </div>
                                <div class="metric-chip">
                                    <span class="metric-k">"Action"</span>
                                    <span class="metric-v mono">{move || {
                                        let a = last_action.get();
                                        if a.is_empty() { "—".to_string() } else { a.to_uppercase() }
                                    }}</span>
                                </div>
                                <div class="metric-chip">
                                    <span class="metric-k">"Reward"</span>
                                    <span class=move || {
                                        let r = last_reward.get();
                                        if r > 0.0 {
                                            "metric-v good"
                                        } else if r < 0.0 {
                                            "metric-v bad"
                                        } else {
                                            "metric-v muted"
                                        }
                                    }>
                                        {move || fmt_f32_signed_fixed(last_reward.get(), 2)}
                                    </span>
                                </div>

                                <button
                                    class=move || if game_stats_open.get() { "btn sm primary" } else { "btn sm" }
                                    on:click=move |_| set_game_stats_open.update(|v| *v = !*v)
                                    title="Show/hide full stats + persistence"
                                >
                                    {move || if game_stats_open.get() { "Stats ▴" } else { "Stats ▾" }}
                                </button>
                            </div>

                            <div class="game-topdash-actions">
                                <div class="subtle" style="margin-right: 6px;">"Analytics:"</div>
                                {AnalyticsPanel::all()
                                    .iter()
                                    .copied()
                                    .map(|p| {
                                        view! {
                                            <button
                                                class="btn sm"
                                                on:click=move |_| {
                                                    set_analytics_panel.set(p);
                                                    set_analytics_modal_open.set(true);
                                                }
                                            >
                                                {p.label()}
                                            </button>
                                        }
                                    })
                                    .collect_view()}
                            </div>
                        </div>

                        <Show when=move || game_stats_open.get()>
                            <div class="game-topdash-details">
                                <div class="topdash-grid">
                                    <div class="card topdash-card">
                                        <h3 class="card-title">"📊 Statistics"</h3>
                                        <div class="stat-row">
                                            <span class="stat-label">"Steps"</span>
                                            <span class="stat-value">{move || steps.get().to_string()}</span>
                                        </div>
                                        <div class="stat-row">
                                            <span class="stat-label">"Trials"</span>
                                            <span class="stat-value">{move || trials.get().to_string()}</span>
                                        </div>
                                        <div class="stat-row">
                                            <span class="stat-label">"Correct"</span>
                                            <span class="stat-value good">{move || correct_count.get().to_string()}</span>
                                        </div>
                                        <div class="stat-row">
                                            <span class="stat-label">"Incorrect"</span>
                                            <span class="stat-value bad">{move || incorrect_count.get().to_string()}</span>
                                        </div>
                                        <div class="stat-row">
                                            <span class="stat-label">"Success Rate"</span>
                                            <span class=move || {
                                                let r = recent_rate.get();
                                                if r >= 0.85 {
                                                    "stat-value good value-strong"
                                                } else if r >= 0.70 {
                                                    "stat-value warn value-strong"
                                                } else {
                                                    "stat-value value-strong"
                                                }
                                            }>
                                                {move || format!("{}%", fmt_f32_fixed(recent_rate.get() * 100.0, 1))}
                                            </span>
                                        </div>
                                        <div class="divider"></div>
                                        <div class="stat-row">
                                            <span class="stat-label">"Last Action"</span>
                                            <span class="stat-value accent value-strong">{move || {
                                                let a = last_action.get();
                                                if a.is_empty() { "—".to_string() } else { a.to_uppercase() }
                                            }}</span>
                                        </div>
                                        <div class="stat-row">
                                            <span class="stat-label">"Last Reward"</span>
                                            <span class=move || {
                                                let r = last_reward.get();
                                                if r > 0.0 {
                                                    "stat-value good value-strong"
                                                } else if r < 0.0 {
                                                    "stat-value bad value-strong"
                                                } else {
                                                    "stat-value muted value-strong"
                                                }
                                            }>
                                                {move || fmt_f32_signed_fixed(last_reward.get(), 2)}
                                            </span>
                                        </div>
                                    </div>

                                    <div class="card topdash-card">
                                        <h3 class="card-title">"🧠 Brain Substrate"</h3>
                                        <div class="stat-row">
                                            <span class="stat-label">"Age (steps)"</span>
                                            <span class="stat-value">{move || brain_age.get().to_string()}</span>
                                        </div>
                                        <div class="stat-row">
                                            <span class="stat-label">"Units"</span>
                                            <span class="stat-value">{move || diag.get().unit_count.to_string()}</span>
                                        </div>
                                        <div class="stat-row">
                                            <span class="stat-label">"Connections"</span>
                                            <span class="stat-value">{move || diag.get().connection_count.to_string()}</span>
                                        </div>
                                        <div class="stat-row">
                                            <span class="stat-label">"Avg Weight"</span>
                                            <span class="stat-value">{move || fmt_f32_fixed(diag.get().avg_weight, 4)}</span>
                                        </div>
                                        <div class="stat-row">
                                            <span class="stat-label">"Avg Amplitude"</span>
                                            <span class="stat-value">{move || fmt_f32_fixed(diag.get().avg_amp, 4)}</span>
                                        </div>
                                        <div class="stat-row">
                                            <span class="stat-label">"Pruned (last)"</span>
                                            <span class="stat-value">{move || diag.get().pruned_last_step.to_string()}</span>
                                        </div>
                                        <div class="stat-row">
                                            <span class="stat-label">"Memory"</span>
                                            <span class="stat-value">{move || format!("{}KB", diag.get().memory_bytes / 1024)}</span>
                                        </div>
                                        <div class="divider"></div>
                                        <h4 style="margin: 8px 0 4px 0; font-size: 0.85rem; color: var(--muted);">"Causal Memory"</h4>
                                        <div class="stat-row">
                                            <span class="stat-label">"Symbols"</span>
                                            <span class="stat-value">{move || causal_symbols.get().to_string()}</span>
                                        </div>
                                        <div class="stat-row">
                                            <span class="stat-label">"Edges"</span>
                                            <span class="stat-value">{move || causal_edges.get().to_string()}</span>
                                        </div>
                                    </div>
                                </div>
                            </div>
                        </Show>
                    </div>

                    <Show when=move || analytics_modal_open.get()>
                        <div class="floating-layer">
                            <div
                                class="floating-window"
                                style=move || format!(
                                    "left: {}px; top: {}px;",
                                    analytics_win_x.get(),
                                    analytics_win_y.get()
                                )
                            >
                                <div class="floating-head">
                                    <div class="floating-titlebar">
                                        <div
                                            class="floating-drag"
                                            on:pointerdown=move |ev: web_sys::PointerEvent| {
                                                if ev.button() != 0 {
                                                    return;
                                                }

                                                ev.prevent_default();
                                                let pid = ev.pointer_id();
                                                let sx = ev.client_x() as f64;
                                                let sy = ev.client_y() as f64;

                                                set_analytics_drag.set(Some((
                                                    pid,
                                                    sx,
                                                    sy,
                                                    analytics_win_x.get_untracked(),
                                                    analytics_win_y.get_untracked(),
                                                )));

                                                if let Some(t) = ev.current_target() {
                                                    if let Ok(el) = t.dyn_into::<web_sys::Element>() {
                                                        let _ = el.set_pointer_capture(pid);
                                                    }
                                                }
                                            }
                                            on:pointermove=move |ev: web_sys::PointerEvent| {
                                            let Some((pid, sx, sy, ox, oy)) =
                                                analytics_drag.get_untracked()
                                            else {
                                                return;
                                            };
                                            if ev.pointer_id() != pid {
                                                return;
                                            }

                                            ev.prevent_default();
                                            let mut nx = ox + (ev.client_x() as f64 - sx);
                                            let mut ny = oy + (ev.client_y() as f64 - sy);

                                            // Keep the window reachable (don't allow dragging it fully off-screen).
                                            if let Some(win) = web_sys::window() {
                                                let vw = win
                                                    .inner_width()
                                                    .ok()
                                                    .and_then(|v| v.as_f64())
                                                    .unwrap_or(0.0);
                                                let vh = win
                                                    .inner_height()
                                                    .ok()
                                                    .and_then(|v| v.as_f64())
                                                    .unwrap_or(0.0);
                                                let max_x = (vw - 120.0).max(0.0);
                                                let max_y = (vh - 60.0).max(0.0);
                                                nx = nx.clamp(0.0, max_x);
                                                ny = ny.clamp(0.0, max_y);
                                            } else {
                                                nx = nx.max(0.0);
                                                ny = ny.max(0.0);
                                            }

                                            set_analytics_win_x.set(nx);
                                            set_analytics_win_y.set(ny);
                                            }
                                            on:pointerup=move |ev: web_sys::PointerEvent| {
                                            let Some((pid, ..)) = analytics_drag.get_untracked() else {
                                                return;
                                            };
                                            if ev.pointer_id() != pid {
                                                return;
                                            }

                                            set_analytics_drag.set(None);
                                            if let Some(t) = ev.current_target() {
                                                if let Ok(el) = t.dyn_into::<web_sys::Element>() {
                                                    let _ = el.release_pointer_capture(pid);
                                                }
                                            }
                                            }
                                            on:pointercancel=move |ev: web_sys::PointerEvent| {
                                            let Some((pid, ..)) = analytics_drag.get_untracked() else {
                                                return;
                                            };
                                            if ev.pointer_id() != pid {
                                                return;
                                            }

                                            set_analytics_drag.set(None);
                                            if let Some(t) = ev.current_target() {
                                                if let Ok(el) = t.dyn_into::<web_sys::Element>() {
                                                    let _ = el.release_pointer_capture(pid);
                                                }
                                            }
                                            }
                                        >
                                            <div class="floating-title">
                                                {move || format!("📈 Analytics — {}", analytics_panel.get().label())}
                                            </div>
                                        </div>

                                        <button class="btn sm" on:click=move |_| close_analytics_window()>
                                            "Close"
                                        </button>
                                    </div>

                                    <div class="floating-tabs">
                                        {AnalyticsPanel::all()
                                            .iter()
                                            .copied()
                                            .map(|p| {
                                                view! {
                                                    <button
                                                        class=move || {
                                                            if analytics_panel.get() == p {
                                                                "btn sm primary"
                                                            } else {
                                                                "btn sm"
                                                            }
                                                        }
                                                        on:click=move |_| {
                                                            set_analytics_panel.set(p);
                                                            set_analytics_modal_open.set(true);
                                                        }
                                                    >
                                                        {p.label()}
                                                    </button>
                                                }
                                            })
                                            .collect_view()}
                                    </div>
                                </div>

                                <div class="floating-body">
                                    <Show when=move || analytics_panel.get() == AnalyticsPanel::Reward>
                                        <div class="card">
                                            <h3 class="card-title">"⚡ Reward Trace"</h3>
                                            <p class="subtle">"Last 50 reward signals"</p>
                                            <canvas node_ref=neuromod_chart_ref width="600" height="80" class="canvas"></canvas>
                                        </div>
                                    </Show>

                                    <Show when=move || analytics_panel.get() == AnalyticsPanel::Choices>
                                        <div class="card">
                                            <h3 class="card-title">"🎛️ Choices Over Time"</h3>
                                            <p class="subtle">"Rolling probability of each action (empirical)"</p>

                                            <div class="row end wrap">
                                                <label class="label">
                                                    <span>"Window"</span>
                                                    <input
                                                        class="input compact"
                                                        type="number"
                                                        min="1"
                                                        max="200"
                                                        step="1"
                                                        prop:value=move || choice_window.get().to_string()
                                                        on:input=move |ev| {
                                                            if let Ok(v) = event_target_value(&ev).parse::<u32>() {
                                                                set_choice_window.set(v.clamp(1, 200));
                                                            }
                                                        }
                                                    />
                                                </label>
                                            </div>

                                            <canvas node_ref=choices_chart_ref width="700" height="180" class="canvas"></canvas>
                                            <p class="subtle">"Tip: lower ε to see exploitation; raise ε to see exploration noise."</p>
                                        </div>
                                    </Show>

                                    <Show when=move || analytics_panel.get() == AnalyticsPanel::UnitPlot>
                                        <div class="card">
                                            <h3 class="card-title">"🧬 Unit Activity Plot"</h3>
                                            <p class="subtle">"Sampled unit activations showing amplitude and type distribution"</p>
                                            <canvas node_ref=unit_plot_ref width="800" height="360" class="canvas tall"></canvas>
                                            <div style="margin-top: 8px; display: flex; flex-wrap: wrap; gap: 12px; font-size: 0.75rem;">
                                                <div style="display: flex; align-items: center; gap: 4px;">
                                                    <div style="width: 10px; height: 10px; border-radius: 50%; background: rgb(122, 162, 255);"></div>
                                                    <span style="color: var(--muted);">"Sensors (input)"</span>
                                                </div>
                                                <div style="display: flex; align-items: center; gap: 4px;">
                                                    <div style="width: 10px; height: 10px; border-radius: 50%; background: rgb(74, 222, 128);"></div>
                                                    <span style="color: var(--muted);">"Groups (actions)"</span>
                                                </div>
                                                <div style="display: flex; align-items: center; gap: 4px;">
                                                    <div style="width: 10px; height: 10px; border-radius: 50%; background: rgb(251, 191, 36);"></div>
                                                    <span style="color: var(--muted);">"Regular (free)"</span>
                                                </div>
                                                <div style="display: flex; align-items: center; gap: 4px;">
                                                    <div style="width: 10px; height: 10px; border-radius: 50%; background: rgb(148, 163, 184);"></div>
                                                    <span style="color: var(--muted);">"Concepts (imprinted)"</span>
                                                </div>
                                            </div>
                                        </div>
                                    </Show>
                                </div>
                            </div>
                        </div>
                    </Show>

                    // Game-specific content area
                    <div class="canvas-container">
                        // Spot game - Enhanced with visual arena
                        <Show when=move || game_kind.get() == GameKind::Spot>
                            <div class="game-page">
                                <div class="game-header">
                                    <h2>"Spot Discrimination"</h2>
                                    <p>"Learn to respond LEFT or RIGHT based on stimulus"</p>
                                </div>

                                {training_health_bar_view()}

                                <div class="game-grid">
                                    <div class="game-primary">
                                        <div class="card">
                                            <h3 class="card-title">"Arena"</h3>
                                            <div class="arena-grid" style="margin-top: 10px;">
                                                <div style=move || {
                                                    let active = matches!(spot_is_left.get(), Some(true));
                                                    format!(
                                                        "display: flex; flex-direction: column; align-items: center; justify-content: center; padding: 30px; border-radius: 12px; border: 2px solid {}; background: {};",
                                                        if active { "var(--accent)" } else { "var(--border)" },
                                                        if active { "rgba(122, 162, 255, 0.15)" } else { "rgba(0,0,0,0.2)" },
                                                    )
                                                }>
                                                    <span style="font-size: 3rem;">"⬅️"</span>
                                                    <span style="margin-top: 8px; font-size: 0.9rem; font-weight: 800; color: var(--text);">"LEFT"</span>
                                                    <span style="font-size: 0.75rem; color: var(--muted);">"Press A"</span>
                                                </div>
                                                <div style=move || {
                                                    let active = matches!(spot_is_left.get(), Some(false));
                                                    format!(
                                                        "display: flex; flex-direction: column; align-items: center; justify-content: center; padding: 30px; border-radius: 12px; border: 2px solid {}; background: {};",
                                                        if active { "var(--accent)" } else { "var(--border)" },
                                                        if active { "rgba(122, 162, 255, 0.15)" } else { "rgba(0,0,0,0.2)" },
                                                    )
                                                }>
                                                    <span style="font-size: 3rem;">"➡️"</span>
                                                    <span style="margin-top: 8px; font-size: 0.9rem; font-weight: 800; color: var(--text);">"RIGHT"</span>
                                                    <span style="font-size: 0.75rem; color: var(--muted);">"Press D"</span>
                                                </div>
                                            </div>
                                        </div>

                                        <div class="card">
                                            <h3 class="card-title">"Last Decision"</h3>
                                            <div style="display: flex; align-items: center; gap: 12px; flex-wrap: wrap;">
                                                <span class="subtle">"Braine chose:"</span>
                                                <span style="font-size: 1.1rem; font-weight: 800; color: var(--accent);">
                                                    {move || {
                                                        let a = last_action.get();
                                                        if a.is_empty() { "—".to_string() } else { a.to_uppercase() }
                                                    }}
                                                </span>
                                                <span style=move || format!(
                                                    "padding: 4px 10px; border-radius: 999px; font-size: 0.8rem; font-weight: 800; background: {}; color: #fff;",
                                                    if last_reward.get() > 0.0 { "#22c55e" } else if last_reward.get() < 0.0 { "#ef4444" } else { "#64748b" }
                                                )>
                                                    {move || if last_reward.get() > 0.0 { "✓ Correct" } else if last_reward.get() < 0.0 { "✗ Wrong" } else { "—" }}
                                                </span>
                                            </div>
                                        </div>
                                    </div>

                                    <div class="game-secondary">
                                        <div class="card">
                                            <h3 class="card-title">"Presets"</h3>
                                            <div class="subtle">"Tune cadence + exploration for faster learning."</div>
                                            <div style="display: flex; gap: 8px; flex-wrap: wrap; margin-top: 10px;">
                                                <button class="btn sm" on:click=move |_| {
                                                    set_trial_period_ms.set(450);
                                                    set_exploration_eps.set(0.06);
                                                    set_meaning_alpha.set(8.0);
                                                }>
                                                    "Easy"
                                                </button>
                                                <button class="btn sm" on:click=move |_| {
                                                    set_trial_period_ms.set(300);
                                                    set_exploration_eps.set(0.08);
                                                    set_meaning_alpha.set(6.0);
                                                }>
                                                    "Normal"
                                                </button>
                                                <button class="btn sm" on:click=move |_| {
                                                    set_trial_period_ms.set(160);
                                                    set_exploration_eps.set(0.12);
                                                    set_meaning_alpha.set(4.5);
                                                }>
                                                    "Hard"
                                                </button>
                                            </div>
                                        </div>

                                        <div class="card">
                                            <h3 class="card-title">"Reward & Signals"</h3>
                                            <pre class="codeblock">{GameKind::Spot.reward_info()}</pre>
                                            <div class="subtle" style="margin-top: 8px; line-height: 1.5;">
                                                "Global shaping: shaped = ((raw + bias) × scale) clamped to [−5, +5]"<br/>
                                                {move || format!(
                                                    "Current: bias={}, scale={} ",
                                                    fmt_f32_signed_fixed(reward_bias.get(), 2),
                                                    fmt_f32_fixed(reward_scale.get(), 2)
                                                )}
                                            </div>
                                        </div>

                                        <div class="card">
                                            <h3 class="card-title">"What The Knobs Mean"</h3>
                                            <pre class="pre">{GameKind::Spot.inputs_info()}</pre>
                                            <div class="subtle" style="margin-top: 8px; line-height: 1.5;">
                                                {move || format!(
                                                    "Trial ms={}  ε={}  α={} ",
                                                    trial_period_ms.get(),
                                                    fmt_f32_fixed(exploration_eps.get(), 2),
                                                    fmt_f32_fixed(meaning_alpha.get(), 1)
                                                )}
                                            </div>
                                        </div>
                                    </div>
                                </div>
                            </div>
                        </Show>

                        // Bandit game - Enhanced with arm visualization
                        <Show when=move || game_kind.get() == GameKind::Bandit>
                            <div class="game-page">
                                <div class="game-header">
                                    <h2>"🎰 Two-Armed Bandit"</h2>
                                    <p>"Explore vs exploit: learn which arm pays better"</p>
                                </div>

                                {training_health_bar_view()}

                                <div class="game-grid">
                                    <div class="game-primary">
                                        <div class="card">
                                            <h3 class="card-title">"Arms"</h3>
                                            <div class="arena-grid" style="gap: 16px; margin-top: 10px;">
                                                <div style=move || format!(
                                                    "display: flex; flex-direction: column; align-items: center; padding: 24px; border-radius: 12px; border: 2px solid {}; background: {}; transition: all 0.2s;",
                                                    if last_action.get() == "left" { "#fbbf24" } else { "var(--border)" },
                                                    if last_action.get() == "left" { "rgba(251, 191, 36, 0.1)" } else { "rgba(0,0,0,0.2)" }
                                                )>
                                                    <div style="font-size: 3rem; margin-bottom: 8px;">"🎰"</div>
                                                    <span style="font-size: 1rem; font-weight: 800; color: var(--text);">"ARM A"</span>
                                                    <span style="font-size: 0.75rem; color: var(--muted); margin-top: 4px;">"Press A"</span>
                                                </div>
                                                <div style=move || format!(
                                                    "display: flex; flex-direction: column; align-items: center; padding: 24px; border-radius: 12px; border: 2px solid {}; background: {}; transition: all 0.2s;",
                                                    if last_action.get() == "right" { "#fbbf24" } else { "var(--border)" },
                                                    if last_action.get() == "right" { "rgba(251, 191, 36, 0.1)" } else { "rgba(0,0,0,0.2)" }
                                                )>
                                                    <div style="font-size: 3rem; margin-bottom: 8px;">"🎰"</div>
                                                    <span style="font-size: 1rem; font-weight: 800; color: var(--text);">"ARM B"</span>
                                                    <span style="font-size: 0.75rem; color: var(--muted); margin-top: 4px;">"Press D"</span>
                                                </div>
                                            </div>
                                        </div>

                                        <div class="card">
                                            <h3 class="card-title">"Reward"</h3>
                                            <div style=move || format!(
                                                "display: flex; align-items: center; justify-content: center; gap: 8px; padding: 14px 16px; border-radius: 12px; background: {};",
                                                if last_reward.get() > 0.0 { "rgba(34, 197, 94, 0.2)" } else if last_reward.get() < 0.0 { "rgba(239, 68, 68, 0.2)" } else { "rgba(0,0,0,0.2)" }
                                            )>
                                                <span style=move || format!(
                                                    "font-size: 1.5rem; font-weight: 900; color: {};",
                                                    if last_reward.get() > 0.0 { "#4ade80" } else if last_reward.get() < 0.0 { "#f87171" } else { "var(--muted)" }
                                                )>
                                                    {move || fmt_f32_signed_fixed(last_reward.get(), 1)}
                                                </span>
                                                <span class="subtle">"last reward"</span>
                                            </div>
                                        </div>
                                    </div>

                                    <div class="game-secondary">
                                        <div class="card">
                                            <h3 class="card-title">"Probabilities"</h3>
                                            <div class="subtle">{move || format!(
                                                "P(left)={}  P(right)={} ",
                                                fmt_f32_fixed(bandit_prob_left.get(), 2),
                                                fmt_f32_fixed(bandit_prob_right.get(), 2)
                                            )}</div>
                                            <div style="display: flex; gap: 8px; flex-wrap: wrap; margin-top: 10px;">
                                                <button class="btn sm" on:click=move |_| do_bandit_set_probs(0.90, 0.10)>
                                                    "Easy"
                                                </button>
                                                <button class="btn sm" on:click=move |_| do_bandit_set_probs(0.80, 0.20)>
                                                    "Normal"
                                                </button>
                                                <button class="btn sm" on:click=move |_| do_bandit_set_probs(0.60, 0.40)>
                                                    "Hard"
                                                </button>
                                            </div>
                                            <div style="margin-top: 10px; display: flex; gap: 10px; flex-wrap: wrap;">
                                                <label class="label" style="min-width: 160px;">
                                                    <span title="Probability that choosing LEFT yields +1 (otherwise -1).">"P(left)"</span>
                                                    <input
                                                        type="number"
                                                        min="0"
                                                        max="1"
                                                        step="0.05"
                                                        class="input compact"
                                                        prop:value=move || fmt_f32_fixed(bandit_prob_left.get(), 2)
                                                        on:input=move |ev| {
                                                            if let Ok(x) = event_target_value(&ev).parse::<f32>() {
                                                                do_bandit_set_probs(x, bandit_prob_right.get_untracked());
                                                            }
                                                        }
                                                    />
                                                </label>
                                                <label class="label" style="min-width: 160px;">
                                                    <span title="Probability that choosing RIGHT yields +1 (otherwise -1).">"P(right)"</span>
                                                    <input
                                                        type="number"
                                                        min="0"
                                                        max="1"
                                                        step="0.05"
                                                        class="input compact"
                                                        prop:value=move || fmt_f32_fixed(bandit_prob_right.get(), 2)
                                                        on:input=move |ev| {
                                                            if let Ok(x) = event_target_value(&ev).parse::<f32>() {
                                                                do_bandit_set_probs(bandit_prob_left.get_untracked(), x);
                                                            }
                                                        }
                                                    />
                                                </label>
                                            </div>
                                        </div>

                                        <div class="card">
                                            <h3 class="card-title">"Reward & Signals"</h3>
                                            <pre class="codeblock">{GameKind::Bandit.reward_info()}</pre>
                                            <div class="subtle" style="margin-top: 8px; line-height: 1.5;">
                                                "Global shaping: shaped = ((raw + bias) × scale) clamped to [−5, +5]"<br/>
                                                {move || format!(
                                                    "Current: bias={}, scale={} ",
                                                    fmt_f32_signed_fixed(reward_bias.get(), 2),
                                                    fmt_f32_fixed(reward_scale.get(), 2)
                                                )}
                                            </div>
                                        </div>

                                        <div class="card">
                                            <h3 class="card-title">"Inputs & Actions"</h3>
                                            <pre class="pre">{GameKind::Bandit.inputs_info()}</pre>
                                            <div class="subtle" style="margin-top: 8px; line-height: 1.5;">
                                                {move || format!(
                                                    "Trial ms={}  ε={}  α={} ",
                                                    trial_period_ms.get(),
                                                    fmt_f32_fixed(exploration_eps.get(), 2),
                                                    fmt_f32_fixed(meaning_alpha.get(), 1)
                                                )}
                                            </div>
                                        </div>
                                    </div>
                                </div>
                            </div>
                        </Show>

                        // Spot Reversal - Enhanced with context indicator
                        <Show when=move || game_kind.get() == GameKind::SpotReversal>
                            <div class="game-page">
                                <div class="game-header">
                                    <h2>"Spot Reversal"</h2>
                                    <p>"Rules flip periodically; context bit helps detect reversals"</p>
                                </div>

                                {training_health_bar_view()}

                                <div class="game-grid">
                                    <div class="game-primary">
                                        <div class="card">
                                            <h3 class="card-title">"Current Rule"</h3>
                                            <div style=move || format!(
                                                "display: flex; align-items: center; gap: 12px; padding: 12px 14px; border-radius: 12px; background: {}; border: 1px solid {};",
                                                if reversal_active.get() { "rgba(251, 191, 36, 0.12)" } else { "rgba(122, 162, 255, 0.12)" },
                                                if reversal_active.get() { "rgba(251, 191, 36, 0.25)" } else { "rgba(122, 162, 255, 0.25)" }
                                            )>
                                                <span style="font-size: 1.4rem;">{move || if reversal_active.get() { "🔄" } else { "➡️" }}</span>
                                                <div>
                                                    <div style=move || format!("font-weight: 900; font-size: 1rem; color: {};", if reversal_active.get() { "#fbbf24" } else { "var(--accent)" })>
                                                        {move || if reversal_active.get() { "REVERSED" } else { "NORMAL" }}
                                                    </div>
                                                    <div class="subtle">{move || format!("Flips after {} trials", reversal_flip_after.get())}</div>
                                                </div>
                                            </div>
                                        </div>

                                        <div class="card">
                                            <h3 class="card-title">"Arena"</h3>
                                            <div class="arena-grid" style="margin-top: 10px;">
                                                <div style=move || {
                                                    let active = matches!(spot_is_left.get(), Some(true));
                                                    format!(
                                                        "display: flex; flex-direction: column; align-items: center; justify-content: center; padding: 24px; border-radius: 12px; border: 2px solid {}; background: {};",
                                                        if active { "var(--accent)" } else { "var(--border)" },
                                                        if active { "rgba(122, 162, 255, 0.15)" } else { "rgba(0,0,0,0.2)" },
                                                    )
                                                }>
                                                    <span style="font-size: 2.5rem;">"⬅️"</span>
                                                    <span style="margin-top: 8px; font-weight: 800; color: var(--text);">"LEFT"</span>
                                                </div>
                                                <div style=move || {
                                                    let active = matches!(spot_is_left.get(), Some(false));
                                                    format!(
                                                        "display: flex; flex-direction: column; align-items: center; justify-content: center; padding: 24px; border-radius: 12px; border: 2px solid {}; background: {};",
                                                        if active { "var(--accent)" } else { "var(--border)" },
                                                        if active { "rgba(122, 162, 255, 0.15)" } else { "rgba(0,0,0,0.2)" },
                                                    )
                                                }>
                                                    <span style="font-size: 2.5rem;">"➡️"</span>
                                                    <span style="margin-top: 8px; font-weight: 800; color: var(--text);">"RIGHT"</span>
                                                </div>
                                            </div>
                                        </div>
                                    </div>

                                    <div class="game-secondary">
                                        <div class="card">
                                            <h3 class="card-title">"Flip Timing"</h3>
                                            <div class="subtle">{move || format!("flip_after_trials={}", reversal_flip_after.get())}</div>
                                            <div style="display: flex; gap: 8px; flex-wrap: wrap; margin-top: 10px;">
                                                <button class="btn sm" on:click=move |_| do_reversal_set_flip_after(400)>
                                                    "Easier (slow flip)"
                                                </button>
                                                <button class="btn sm" on:click=move |_| do_reversal_set_flip_after(200)>
                                                    "Normal"
                                                </button>
                                                <button class="btn sm" on:click=move |_| do_reversal_set_flip_after(80)>
                                                    "Hard (fast flip)"
                                                </button>
                                            </div>
                                            <div style="margin-top: 10px; display: flex; gap: 10px; flex-wrap: wrap; align-items: center; justify-content: space-between;">
                                                <label class="label" style="min-width: 200px;">
                                                    <span title="After this many trials, the correct mapping flips.">"flip_after_trials"</span>
                                                    <input
                                                        type="number"
                                                        min="1"
                                                        max="2000"
                                                        step="10"
                                                        class="input compact"
                                                        prop:value=move || reversal_flip_after.get().to_string()
                                                        on:input=move |ev| {
                                                            if let Ok(n) = event_target_value(&ev).parse::<u32>() {
                                                                do_reversal_set_flip_after(n);
                                                            }
                                                        }
                                                    />
                                                </label>
                                                <div class="subtle" style="max-width: 220px;">
                                                    "If this flips too fast, Braine can’t stabilize two rules."
                                                </div>
                                            </div>
                                        </div>

                                        <div class="card">
                                            <h3 class="card-title">"Reward & Signals"</h3>
                                            <pre class="codeblock">{GameKind::SpotReversal.reward_info()}</pre>
                                            <div class="subtle" style="margin-top: 8px; line-height: 1.5;">
                                                "Global shaping: shaped = ((raw + bias) × scale) clamped to [−5, +5]"<br/>
                                                {move || format!(
                                                    "Current: bias={}, scale={}",
                                                    fmt_f32_signed_fixed(reward_bias.get(), 2),
                                                    fmt_f32_fixed(reward_scale.get(), 2)
                                                )}
                                            </div>
                                        </div>

                                        <div class="card">
                                            <h3 class="card-title">"Inputs & Actions"</h3>
                                            <pre class="pre">{GameKind::SpotReversal.inputs_info()}</pre>
                                            <div class="subtle" style="margin-top: 8px; line-height: 1.5;">
                                                {move || format!(
                                                    "Trial ms={}  ε={}  α={} ",
                                                    trial_period_ms.get(),
                                                    fmt_f32_fixed(exploration_eps.get(), 2),
                                                    fmt_f32_fixed(meaning_alpha.get(), 1)
                                                )}
                                            </div>
                                        </div>
                                    </div>
                                </div>
                            </div>
                        </Show>

                        // SpotXY game with canvas - Modern gaming design
                        <Show when=move || game_kind.get() == GameKind::SpotXY>
                            <div class="game-page">
                                <div style="display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 10px; padding: 16px 20px; background: linear-gradient(135deg, rgba(122, 162, 255, 0.1), rgba(0,0,0,0.3)); border: 1px solid var(--border); border-radius: 16px;">
                                    <div>
                                        <h2 style="margin: 0; font-size: 1.2rem; font-weight: 900; color: var(--text); display: flex; align-items: center; gap: 8px;">
                                            <span style="font-size: 1.5rem;">"🎯"</span>
                                            "SpotXY Tracker"
                                        </h2>
                                        <p style="margin: 4px 0 0 0; color: var(--muted); font-size: 0.85rem;">"Predict the dot position in a 2D grid"</p>
                                    </div>
                                    <div style=move || format!(
                                        "padding: 8px 16px; border-radius: 999px; font-size: 0.85rem; font-weight: 900; background: {}; color: {};",
                                        if spotxy_eval.get() { "linear-gradient(135deg, #22c55e, #16a34a)" } else { "rgba(122, 162, 255, 0.2)" },
                                        if spotxy_eval.get() { "#fff" } else { "var(--accent)" }
                                    )>
                                        {move || if spotxy_eval.get() { "🧪 EVAL" } else { "📚 TRAIN" }}
                                    </div>
                                </div>

                                {training_health_bar_view()}

                                <div class="game-grid">
                                    <div class="game-primary">
                                        <div class="card">
                                            <h3 class="card-title">"Canvas"</h3>
                                            <div class="game-canvas-wrap" style="margin-top: 10px;">
                                                <div style=move || format!(
                                                    "position: absolute; width: 340px; height: 340px; border-radius: 50%; filter: blur(70px); opacity: 0.25; pointer-events: none; background: {};",
                                                    if spotxy_eval.get() { "#22c55e" } else { "#7aa2ff" }
                                                )></div>
                                                <div class="game-canvas-frame" style=move || format!(
                                                    "background: {};",
                                                    if spotxy_eval.get() { "linear-gradient(135deg, #22c55e, #16a34a)" } else { "linear-gradient(135deg, #7aa2ff, #5b7dc9)" }
                                                )>
                                                    <canvas
                                                        node_ref=canvas_ref
                                                        width="340"
                                                        height="340"
                                                        class="game-canvas square"
                                                        style="border-radius: 13px; background: #0a0f1a;"
                                                    ></canvas>
                                                </div>
                                            </div>

                                            <div style="display: flex; gap: 12px; justify-content: center; flex-wrap: wrap; margin-top: 12px;">
                                                <div style="display: flex; gap: 4px; padding: 4px; background: rgba(0,0,0,0.3); border-radius: 10px;">
                                                    <button style="padding: 10px 16px; border: none; background: rgba(122, 162, 255, 0.15); color: var(--accent); border-radius: 8px; cursor: pointer; font-size: 0.9rem; font-weight: 800;"
                                                        on:click=move |_| do_spotxy_grid_minus()>"−"</button>
                                                    <div style="padding: 10px 16px; color: var(--text); font-size: 0.9rem; font-weight: 900; min-width: 70px; text-align: center;">
                                                        {move || {
                                                            let n = spotxy_grid_n.get();
                                                            if n == 0 { "1×1".to_string() } else { format!("{n}×{n}") }
                                                        }}
                                                    </div>
                                                    <button style="padding: 10px 16px; border: none; background: rgba(122, 162, 255, 0.15); color: var(--accent); border-radius: 8px; cursor: pointer; font-size: 0.9rem; font-weight: 800;"
                                                        on:click=move |_| do_spotxy_grid_plus()>"+"</button>
                                                </div>
                                                <button style=move || format!(
                                                    "padding: 10px 20px; border: none; border-radius: 10px; cursor: pointer; font-size: 0.9rem; font-weight: 900; background: {}; color: {};",
                                                    if spotxy_eval.get() { "#22c55e" } else { "rgba(122, 162, 255, 0.15)" },
                                                    if spotxy_eval.get() { "#fff" } else { "var(--accent)" }
                                                )
                                                    on:click=move |_| do_spotxy_toggle_eval()>
                                                    {move || if spotxy_eval.get() { "Switch to Train" } else { "Switch to Eval" }}
                                                </button>
                                            </div>

                                            <div style="display: flex; gap: 8px; justify-content: center; margin-top: 10px; flex-wrap: wrap;">
                                                <span style="padding: 6px 14px; background: rgba(122, 162, 255, 0.1); border: 1px solid rgba(122, 162, 255, 0.2); border-radius: 999px; font-size: 0.8rem; color: var(--muted);">
                                                    "Mode: "<span style="color: var(--accent); font-weight: 900;">{move || spotxy_mode.get()}</span>
                                                </span>
                                            </div>
                                        </div>
                                    </div>

                                    <div class="game-secondary">
                                        <div class="card">
                                            <h3 class="card-title">"Difficulty Presets"</h3>
                                            <div class="subtle">"Bigger grid = harder."</div>
                                            <div style="display: flex; gap: 8px; flex-wrap: wrap; margin-top: 10px;">
                                                <button class="btn sm" on:click=move |_| { do_spotxy_set_eval(false); do_spotxy_set_grid_target(0); }>
                                                    "Easy (BinaryX)"
                                                </button>
                                                <button class="btn sm" on:click=move |_| { do_spotxy_set_eval(false); do_spotxy_set_grid_target(4); }>
                                                    "Normal (4×4)"
                                                </button>
                                                <button class="btn sm" on:click=move |_| { do_spotxy_set_eval(false); do_spotxy_set_grid_target(8); }>
                                                    "Hard (8×8)"
                                                </button>
                                            </div>
                                            <div class="subtle" style="margin-top: 10px; line-height: 1.5;">
                                                "Eval mode is holdout (no learning writes). Use it to check generalization."
                                            </div>
                                        </div>

                                        <div class="card">
                                            <h3 class="card-title">"Reward & Signals"</h3>
                                            <pre class="codeblock">{GameKind::SpotXY.reward_info()}</pre>
                                            <div class="subtle" style="margin-top: 8px; line-height: 1.5;">
                                                "Global shaping: shaped = ((raw + bias) × scale) clamped to [−5, +5]"<br/>
                                                {move || format!(
                                                    "Current: bias={}, scale={}",
                                                    fmt_f32_signed_fixed(reward_bias.get(), 2),
                                                    fmt_f32_fixed(reward_scale.get(), 2)
                                                )}
                                            </div>
                                        </div>

                                        <div class="card">
                                            <h3 class="card-title">"Inputs & Actions"</h3>
                                            <pre class="pre">{GameKind::SpotXY.inputs_info()}</pre>
                                        </div>
                                    </div>
                                </div>
                            </div>
                        </Show>

                        // Maze game with canvas
                        <Show when=move || game_kind.get() == GameKind::Maze>
                            <div class="game-page">
                                <div style="display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 10px; padding: 16px 20px; background: linear-gradient(135deg, rgba(34, 197, 94, 0.10), rgba(0,0,0,0.3)); border: 1px solid var(--border); border-radius: 16px;">
                                    <div>
                                        <h2 style="margin: 0; font-size: 1.2rem; font-weight: 900; color: var(--text); display: flex; align-items: center; gap: 8px;">
                                            <span style="font-size: 1.5rem;">"🧭"</span>
                                            "Maze Navigator"
                                        </h2>
                                        <p style="margin: 4px 0 0 0; color: var(--muted); font-size: 0.85rem;">"Learn to reach the goal by trial-and-error"</p>
                                    </div>
                                    <div style="display: flex; gap: 8px; flex-wrap: wrap;">
                                        <kbd style="padding: 4px 10px; background: rgba(255,255,255,0.1); border: 1px solid var(--border); border-radius: 6px; font-size: 0.75rem; color: var(--muted);">"↑ ↓ ← →"</kbd>
                                    </div>
                                </div>

                                {training_health_bar_view()}

                                <div class="game-grid">
                                    <div class="game-primary">
                                        <div class="card">
                                            <h3 class="card-title">"Canvas"</h3>
                                            <div class="game-canvas-wrap" style="margin-top: 10px;">
                                                <div class="game-canvas-frame" style="border: 1px solid var(--border); background: rgba(0,0,0,0.25);">
                                                    <canvas
                                                        node_ref=maze_canvas_ref
                                                        width="420"
                                                        height="420"
                                                        class="game-canvas square"
                                                        style="border-radius: 13px; background: #0a0f1a;"
                                                    ></canvas>
                                                </div>
                                            </div>
                                            <div class="subtle" style="margin-top: 10px; line-height: 1.5;">
                                                {move || {
                                                    if let Some(s) = maze_state.get() {
                                                        format!(
                                                            "difficulty={}  steps={}  event={}",
                                                            s.difficulty, s.steps, s.last_event
                                                        )
                                                    } else {
                                                        "waiting for maze snapshot...".to_string()
                                                    }
                                                }}
                                            </div>
                                        </div>
                                    </div>

                                    <div class="game-secondary">
                                        <div class="card">
                                            <h3 class="card-title">"Difficulty Presets"</h3>
                                            <div class="subtle">"Easy uses stronger shaping; Hard reduces shaping and increases maze size."</div>
                                            <div style="display: flex; gap: 8px; flex-wrap: wrap; margin-top: 10px;">
                                                <button class="btn sm" on:click=move |_| { do_maze_set_difficulty(0.0); set_trial_period_ms.set(80); }>
                                                    "Easy"
                                                </button>
                                                <button class="btn sm" on:click=move |_| { do_maze_set_difficulty(1.0); set_trial_period_ms.set(95); }>
                                                    "Normal"
                                                </button>
                                                <button class="btn sm" on:click=move |_| { do_maze_set_difficulty(2.0); set_trial_period_ms.set(110); }>
                                                    "Hard"
                                                </button>
                                            </div>
                                        </div>

                                        <div class="card">
                                            <h3 class="card-title">"Reward & Signals"</h3>
                                            <pre class="codeblock">{GameKind::Maze.reward_info()}</pre>
                                            <div class="subtle" style="margin-top: 8px; line-height: 1.5;">
                                                "Global shaping: shaped = ((raw + bias) × scale) clamped to [−5, +5]"<br/>
                                                {move || format!(
                                                    "Current: bias={}, scale={}",
                                                    fmt_f32_signed_fixed(reward_bias.get(), 2),
                                                    fmt_f32_fixed(reward_scale.get(), 2)
                                                )}
                                            </div>
                                        </div>

                                        <div class="card">
                                            <h3 class="card-title">"Inputs & Actions"</h3>
                                            <pre class="pre">{GameKind::Maze.inputs_info()}</pre>
                                        </div>
                                    </div>
                                </div>
                            </div>
                        </Show>

                        // Pong game - Modern arcade design
                        <Show when=move || game_kind.get() == GameKind::Pong>
                            <div class="game-page">
                                <div style="display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 10px; padding: 16px 20px; background: linear-gradient(135deg, rgba(251, 191, 36, 0.1), rgba(0,0,0,0.3)); border: 1px solid var(--border); border-radius: 16px;">
                                    <div>
                                        <h2 style="margin: 0; font-size: 1.2rem; font-weight: 900; color: var(--text); display: flex; align-items: center; gap: 8px;">
                                            <span style="font-size: 1.5rem;">"🏓"</span>
                                            "Pong Arena"
                                        </h2>
                                        <p style="margin: 4px 0 0 0; color: var(--muted); font-size: 0.85rem;">"Control the paddle • Intercept the ball"</p>
                                    </div>
                                    <div style="display: flex; gap: 8px; flex-wrap: wrap;">
                                        <kbd style="padding: 4px 10px; background: rgba(255,255,255,0.1); border: 1px solid var(--border); border-radius: 6px; font-size: 0.75rem; color: var(--muted);">"W/S"</kbd>
                                        <kbd style="padding: 4px 10px; background: rgba(255,255,255,0.1); border: 1px solid var(--border); border-radius: 6px; font-size: 0.75rem; color: var(--muted);">"↑/↓"</kbd>
                                    </div>
                                </div>

                                {training_health_bar_view()}

                                <div class="game-grid">
                                    <div class="game-primary">
                                        <div class="card">
                                            <h3 class="card-title">"Arena"</h3>
                                            <div class="game-canvas-wrap" style="margin-top: 10px;">
                                                <div class="game-canvas-frame" style="border: 1px solid var(--border); background: rgba(0,0,0,0.25);">
                                                    <canvas
                                                        node_ref=pong_canvas_ref
                                                        width="540"
                                                        height="320"
                                                        class="game-canvas wide"
                                                        style="border-radius: 13px; background: #0a0f1a;"
                                                    ></canvas>
                                                </div>
                                            </div>
                                            <div class="subtle" style="margin-top: 10px; line-height: 1.5;">
                                                "Tip: If learning is jittery, reduce ε and slow Trial ms a bit."
                                            </div>
                                        </div>
                                    </div>

                                    <div class="game-secondary">
                                        <div class="card">
                                            <h3 class="card-title">"Difficulty Presets"</h3>
                                            <div class="subtle">"Big paddle + slower ball = easier."</div>
                                            <div style="display: flex; gap: 8px; flex-wrap: wrap; margin-top: 10px;">
                                                <button class="btn sm" on:click=move |_| {
                                                    do_pong_set_param("paddle_half_height", 0.28);
                                                    do_pong_set_param("ball_speed", 0.6);
                                                    do_pong_set_param("paddle_speed", 2.0);
                                                    do_pong_set_param("paddle_bounce_y", 0.55);
                                                    do_pong_set_param("respawn_delay_s", 0.25);
                                                    do_pong_set_param("distractor_enabled", 0.0);
                                                    set_trial_period_ms.set(80);
                                                }>
                                                    "Easy"
                                                </button>
                                                <button class="btn sm" on:click=move |_| {
                                                    do_pong_set_param("paddle_half_height", 0.15);
                                                    do_pong_set_param("ball_speed", 1.0);
                                                    do_pong_set_param("paddle_speed", 1.3);
                                                    do_pong_set_param("paddle_bounce_y", 0.9);
                                                    do_pong_set_param("respawn_delay_s", 0.18);
                                                    do_pong_set_param("distractor_enabled", 0.0);
                                                    set_trial_period_ms.set(100);
                                                }>
                                                    "Normal"
                                                </button>
                                                <button class="btn sm" on:click=move |_| {
                                                    do_pong_set_param("paddle_half_height", 0.11);
                                                    do_pong_set_param("ball_speed", 1.4);
                                                    do_pong_set_param("paddle_speed", 1.2);
                                                    do_pong_set_param("paddle_bounce_y", 1.2);
                                                    do_pong_set_param("respawn_delay_s", 0.12);
                                                    set_trial_period_ms.set(120);
                                                }>
                                                    "Hard"
                                                </button>
                                            </div>
                                        </div>

                                        <div class="card">
                                            <h3 class="card-title">"Parameters"</h3>
                                            <div style="display: flex; gap: 14px; justify-content: center; flex-wrap: wrap;">
                                                <div style="display: flex; flex-direction: column; gap: 4px; min-width: 110px;">
                                                    <span style="font-size: 0.7rem; color: var(--muted); text-transform: uppercase; letter-spacing: 1px;">"Paddle Speed"</span>
                                                    <input
                                                        type="range"
                                                        min="0.5"
                                                        max="5"
                                                        step="0.1"
                                                        style="width: 100%; accent-color: #7aa2ff;"
                                                        prop:value=move || fmt_f32_fixed(pong_paddle_speed.get(), 1)
                                                        on:input=move |ev| {
                                                            let v = event_target_value(&ev);
                                                            if let Ok(x) = v.parse::<f32>() {
                                                                do_pong_set_param("paddle_speed", x);
                                                            }
                                                        }
                                                    />
                                                    <span style="font-size: 0.8rem; color: var(--text); font-weight: 800; text-align: center;">{move || fmt_f32_fixed(pong_paddle_speed.get(), 1)}</span>
                                                </div>

                                                <div style="display: flex; flex-direction: column; gap: 4px; min-width: 110px;">
                                                    <span style="font-size: 0.7rem; color: var(--muted); text-transform: uppercase; letter-spacing: 1px;">"Paddle Size"</span>
                                                    <input
                                                        type="range"
                                                        min="0.05"
                                                        max="0.5"
                                                        step="0.01"
                                                        style="width: 100%; accent-color: #7aa2ff;"
                                                        prop:value=move || fmt_f32_fixed(pong_paddle_half_height.get(), 2)
                                                        on:input=move |ev| {
                                                            let v = event_target_value(&ev);
                                                            if let Ok(x) = v.parse::<f32>() {
                                                                do_pong_set_param("paddle_half_height", x);
                                                            }
                                                        }
                                                    />
                                                    <span style="font-size: 0.8rem; color: var(--text); font-weight: 800; text-align: center;">{move || format!("{}%", fmt_f32_fixed(pong_paddle_half_height.get() * 100.0, 0))}</span>
                                                </div>

                                                <div style="display: flex; flex-direction: column; gap: 4px; min-width: 110px;">
                                                    <span style="font-size: 0.7rem; color: var(--muted); text-transform: uppercase; letter-spacing: 1px;">"Ball Speed"</span>
                                                    <input
                                                        type="range"
                                                        min="0.3"
                                                        max="3"
                                                        step="0.1"
                                                        style="width: 100%; accent-color: #fbbf24;"
                                                        prop:value=move || fmt_f32_fixed(pong_ball_speed.get(), 1)
                                                        on:input=move |ev| {
                                                            let v = event_target_value(&ev);
                                                            if let Ok(x) = v.parse::<f32>() {
                                                                do_pong_set_param("ball_speed", x);
                                                            }
                                                        }
                                                    />
                                                    <span style="font-size: 0.8rem; color: var(--text); font-weight: 800; text-align: center;">{move || format!("{}×", fmt_f32_fixed(pong_ball_speed.get(), 1))}</span>
                                                </div>

                                                <div style="display: flex; flex-direction: column; gap: 4px; min-width: 120px;">
                                                    <span style="font-size: 0.7rem; color: var(--muted); text-transform: uppercase; letter-spacing: 1px;">"Bounce Angle"</span>
                                                    <input
                                                        type="range"
                                                        min="0.0"
                                                        max="2.0"
                                                        step="0.05"
                                                        style="width: 100%; accent-color: #fbbf24;"
                                                        prop:value=move || fmt_f32_fixed(pong_paddle_bounce_y.get(), 2)
                                                        on:input=move |ev| {
                                                            let v = event_target_value(&ev);
                                                            if let Ok(x) = v.parse::<f32>() {
                                                                do_pong_set_param("paddle_bounce_y", x);
                                                            }
                                                        }
                                                    />
                                                    <span style="font-size: 0.8rem; color: var(--text); font-weight: 800; text-align: center;">{move || fmt_f32_fixed(pong_paddle_bounce_y.get(), 2)}</span>
                                                </div>

                                                <div style="display: flex; flex-direction: column; gap: 4px; min-width: 130px;">
                                                    <span style="font-size: 0.7rem; color: var(--muted); text-transform: uppercase; letter-spacing: 1px;">"Respawn Delay"</span>
                                                    <input
                                                        type="range"
                                                        min="0.0"
                                                        max="0.6"
                                                        step="0.01"
                                                        style="width: 100%; accent-color: #7aa2ff;"
                                                        prop:value=move || fmt_f32_fixed(pong_respawn_delay_s.get(), 2)
                                                        on:input=move |ev| {
                                                            let v = event_target_value(&ev);
                                                            if let Ok(x) = v.parse::<f32>() {
                                                                do_pong_set_param("respawn_delay_s", x);
                                                            }
                                                        }
                                                    />
                                                    <span style="font-size: 0.8rem; color: var(--text); font-weight: 800; text-align: center;">{move || format!("{}s", fmt_f32_fixed(pong_respawn_delay_s.get(), 2))}</span>
                                                </div>

                                                <div style="display: flex; flex-direction: column; gap: 6px; min-width: 140px; align-items: center; justify-content: center;">
                                                    <span style="font-size: 0.7rem; color: var(--muted); text-transform: uppercase; letter-spacing: 1px;">"Distractor"</span>
                                                    <label class="label" style="flex-direction: row; align-items: center; gap: 10px; padding: 8px 10px; border-radius: 10px; background: rgba(0,0,0,0.25); border: 1px solid var(--border);">
                                                        <input
                                                            type="checkbox"
                                                            prop:checked=move || pong_distractor_enabled.get()
                                                            on:change=move |ev| {
                                                                let v = event_target_checked(&ev);
                                                                do_pong_set_param("distractor_enabled", if v { 1.0 } else { 0.0 });
                                                            }
                                                        />
                                                        <span style="font-size: 0.85rem; font-weight: 900;">{move || if pong_distractor_enabled.get() { "On" } else { "Off" }}</span>
                                                    </label>
                                                    <span class="subtle" style="text-align: center;">"Adds a 2nd ball (no reward)"</span>
                                                </div>
                                            </div>
                                        </div>

                                        <div class="card">
                                            <h3 class="card-title">"Reward & Signals"</h3>
                                            <pre class="codeblock">{GameKind::Pong.reward_info()}</pre>
                                            <div class="subtle" style="margin-top: 8px; line-height: 1.5;">
                                                "Global shaping: shaped = ((raw + bias) × scale) clamped to [−5, +5]"<br/>
                                                {move || format!(
                                                    "Current: bias={}, scale={}",
                                                    fmt_f32_signed_fixed(reward_bias.get(), 2),
                                                    fmt_f32_fixed(reward_scale.get(), 2)
                                                )}
                                            </div>
                                        </div>

                                        <div class="card">
                                            <h3 class="card-title">"Inputs & Actions"</h3>
                                            <pre class="pre">{GameKind::Pong.inputs_info()}</pre>
                                            <div class="subtle" style="margin-top: 8px; line-height: 1.5;">
                                                {move || format!(
                                                    "Trial ms={}  ε={}  α={} ",
                                                    trial_period_ms.get(),
                                                    fmt_f32_fixed(exploration_eps.get(), 2),
                                                    fmt_f32_fixed(meaning_alpha.get(), 1)
                                                )}
                                            </div>
                                        </div>
                                    </div>
                                </div>
                            </div>
                        </Show>

                        // Sequence game - Enhanced
                        <Show when=move || game_kind.get() == GameKind::Sequence>
                            <div class="game-page">
                                <div class="game-header">
                                    <h2>"Sequence Prediction"</h2>
                                    <p>"Learn repeating patterns: A→B→C→A..."</p>
                                </div>

                                {training_health_bar_view()}

                                <div class="game-grid">
                                    <div class="game-primary">
                                        <div class="card">
                                            <h3 class="card-title">"Tokens"</h3>
                                            <div style="display: flex; flex-direction: column; align-items: center; gap: 8px; padding: 10px 0;">
                                                <span class="subtle" style="text-transform: uppercase; letter-spacing: 1px;">"Current Token"</span>
                                                <div style="font-size: 4rem; font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, 'Liberation Mono', 'Courier New', monospace; font-weight: 900; color: var(--accent);">
                                                    {move || sequence_state.get().map(|s| s.token.clone()).unwrap_or_else(|| "?".to_string())}
                                                </div>
                                                <span style="font-size: 1.5rem; color: var(--muted);">"↓"</span>
                                                <span class="subtle" style="text-transform: uppercase; letter-spacing: 1px;">"Target Next"</span>
                                                <div style="font-size: 2.5rem; font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, 'Liberation Mono', 'Courier New', monospace; color: var(--text); opacity: 0.8;">
                                                    {move || sequence_state.get().map(|s| s.target_next.clone()).unwrap_or_else(|| "?".to_string())}
                                                </div>
                                            </div>

                                            <div style="display: flex; gap: 10px; flex-wrap: wrap; justify-content: center; margin-top: 10px; font-size: 0.85rem;">
                                                <span style="padding: 6px 12px; background: rgba(122, 162, 255, 0.15); border-radius: 999px; color: var(--text);">
                                                    {move || sequence_state.get().map(|s| format!("Regime {}", s.regime)).unwrap_or_default()}
                                                </span>
                                                <span style="padding: 6px 12px; background: rgba(122, 162, 255, 0.15); border-radius: 999px; color: var(--text);">
                                                    {move || sequence_state.get().map(|s| format!("Outcomes: {}", s.outcomes)).unwrap_or_default()}
                                                </span>
                                                <span style="padding: 6px 12px; background: rgba(122, 162, 255, 0.15); border-radius: 999px; color: var(--text);">
                                                    {move || sequence_state.get().map(|s| format!("Shift: {}", s.shift_every)).unwrap_or_default()}
                                                </span>
                                            </div>
                                        </div>
                                    </div>

                                    <div class="game-secondary">
                                        <div class="card">
                                            <h3 class="card-title">"Presets"</h3>
                                            <div class="subtle">"Cadence + exploration tuning."</div>
                                            <div style="display: flex; gap: 8px; flex-wrap: wrap; margin-top: 10px;">
                                                <button class="btn sm" on:click=move |_| {
                                                    set_trial_period_ms.set(450);
                                                    set_exploration_eps.set(0.06);
                                                    set_meaning_alpha.set(8.0);
                                                }>
                                                    "Easy"
                                                </button>
                                                <button class="btn sm" on:click=move |_| {
                                                    set_trial_period_ms.set(280);
                                                    set_exploration_eps.set(0.08);
                                                    set_meaning_alpha.set(6.0);
                                                }>
                                                    "Normal"
                                                </button>
                                                <button class="btn sm" on:click=move |_| {
                                                    set_trial_period_ms.set(140);
                                                    set_exploration_eps.set(0.12);
                                                    set_meaning_alpha.set(4.5);
                                                }>
                                                    "Hard"
                                                </button>
                                            </div>
                                        </div>

                                        <div class="card">
                                            <h3 class="card-title">"Reward & Signals"</h3>
                                            <pre class="codeblock">{GameKind::Sequence.reward_info()}</pre>
                                            <div class="subtle" style="margin-top: 8px; line-height: 1.5;">
                                                "Global shaping: shaped = ((raw + bias) × scale) clamped to [−5, +5]"<br/>
                                                {move || format!(
                                                    "Current: bias={}, scale={}",
                                                    fmt_f32_signed_fixed(reward_bias.get(), 2),
                                                    fmt_f32_fixed(reward_scale.get(), 2)
                                                )}
                                            </div>
                                        </div>

                                        <div class="card">
                                            <h3 class="card-title">"Inputs & Actions"</h3>
                                            <pre class="pre">{GameKind::Sequence.inputs_info()}</pre>
                                            <div class="subtle" style="margin-top: 8px; line-height: 1.5;">
                                                {move || format!(
                                                    "Trial ms={}  ε={}  α={} ",
                                                    trial_period_ms.get(),
                                                    fmt_f32_fixed(exploration_eps.get(), 2),
                                                    fmt_f32_fixed(meaning_alpha.get(), 1)
                                                )}
                                            </div>
                                        </div>
                                    </div>
                                </div>
                            </div>
                        </Show>

                        // Text game - Enhanced
                        <Show when=move || game_kind.get() == GameKind::Text>
                            <div class="game-page">
                                <div class="game-header">
                                    <h2>"📝 Text Prediction"</h2>
                                    <p>"Next-token prediction: observe and predict"</p>
                                </div>

                                {training_health_bar_view()}

                                <div class="game-grid">
                                    <div class="game-primary">
                                        <div class="card">
                                            <h3 class="card-title">"Tokens"</h3>
                                            <div style="text-align: center;">
                                                <span class="subtle" style="text-transform: uppercase; letter-spacing: 2px;">"Current Token"</span>
                                                <div style="font-size: 2.5rem; font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, 'Liberation Mono', 'Courier New', monospace; font-weight: 900; color: var(--accent); margin: 12px 0; padding: 16px; background: rgba(0,0,0,0.4); border-radius: 8px;">
                                                    {move || text_state.get().map(|s| format!("\"{}\"", s.token)).unwrap_or_else(|| "\"?\"".to_string())}
                                                </div>
                                                <div style="display: flex; justify-content: center; align-items: center; gap: 12px; margin: 16px 0;">
                                                    <div style="flex: 1; height: 1px; background: var(--border);"></div>
                                                    <span class="subtle">"predict →"</span>
                                                    <div style="flex: 1; height: 1px; background: var(--border);"></div>
                                                </div>
                                                <span class="subtle" style="text-transform: uppercase; letter-spacing: 2px;">"Next Token"</span>
                                                <div style="font-size: 1.8rem; font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, 'Liberation Mono', 'Courier New', monospace; color: var(--text); opacity: 0.85; margin-top: 12px;">
                                                    {move || text_state.get().map(|s| format!("\"{}\"", s.target_next)).unwrap_or_else(|| "\"?\"".to_string())}
                                                </div>
                                            </div>

                                            <div style="display: flex; gap: 8px; flex-wrap: wrap; justify-content: center; margin-top: 14px;">
                                                <span style="padding: 6px 14px; background: rgba(251, 191, 36, 0.15); border: 1px solid rgba(251, 191, 36, 0.3); border-radius: 999px; font-size: 0.8rem; color: #fbbf24;">
                                                    {move || text_state.get().map(|s| format!("Vocab: {}", s.vocab_size)).unwrap_or_default()}
                                                </span>
                                                <span style="padding: 6px 14px; background: rgba(122, 162, 255, 0.15); border: 1px solid rgba(122, 162, 255, 0.3); border-radius: 999px; font-size: 0.8rem; color: var(--accent);">
                                                    {move || text_state.get().map(|s| format!("Regime {}", s.regime)).unwrap_or_default()}
                                                </span>
                                                <span style="padding: 6px 14px; background: rgba(122, 162, 255, 0.12); border: 1px solid rgba(122, 162, 255, 0.25); border-radius: 999px; font-size: 0.8rem; color: var(--text);">
                                                    {move || text_state.get().map(|s| format!("Outcomes: {}", s.outcomes)).unwrap_or_default()}
                                                </span>
                                                <span style="padding: 6px 14px; background: rgba(74, 222, 128, 0.15); border: 1px solid rgba(74, 222, 128, 0.3); border-radius: 999px; font-size: 0.8rem; color: #4ade80;">
                                                    {move || text_state.get().map(|s| format!("Shift: {}", s.shift_every)).unwrap_or_default()}
                                                </span>
                                            </div>
                                        </div>
                                    </div>

                                    <div class="game-secondary">
                                        <div class="card">
                                            <h3 class="card-title">"Training Config"</h3>
                                            <div class="subtle">"Smaller vocab + slower shifts = easier."</div>
                                            <div style="display: flex; gap: 8px; flex-wrap: wrap; margin-top: 10px;">
                                                <button class="btn sm" on:click=move |_| {
                                                    set_text_max_vocab.set(16);
                                                    set_text_shift_every.set(140);
                                                    (do_text_apply_corpora_sv.get_value())();
                                                }>
                                                    "Easy (reset)"
                                                </button>
                                                <button class="btn sm" on:click=move |_| {
                                                    set_text_max_vocab.set(32);
                                                    set_text_shift_every.set(80);
                                                    (do_text_apply_corpora_sv.get_value())();
                                                }>
                                                    "Normal (reset)"
                                                </button>
                                                <button class="btn sm" on:click=move |_| {
                                                    set_text_max_vocab.set(96);
                                                    set_text_shift_every.set(50);
                                                    (do_text_apply_corpora_sv.get_value())();
                                                }>
                                                    "Hard (reset)"
                                                </button>
                                            </div>
                                            <div class="subtle" style="margin-top: 10px; line-height: 1.5;">
                                                {move || format!("Current: max_vocab={}  shift_every={}", text_max_vocab.get(), text_shift_every.get())}
                                            </div>
                                            <div class="subtle" style="margin-top: 8px; line-height: 1.5;">
                                                "Reset buttons rebuild sensors/actions for the Text game but keep the same brain."
                                            </div>
                                        </div>

                                        <div class="card">
                                            <h3 class="card-title">"Reward & Signals"</h3>
                                            <pre class="codeblock">{GameKind::Text.reward_info()}</pre>
                                            <div class="subtle" style="margin-top: 8px; line-height: 1.5;">
                                                "Global shaping: shaped = ((raw + bias) × scale) clamped to [−5, +5]"<br/>
                                                {move || format!(
                                                    "Current: bias={}, scale={}",
                                                    fmt_f32_signed_fixed(reward_bias.get(), 2),
                                                    fmt_f32_fixed(reward_scale.get(), 2)
                                                )}
                                            </div>
                                        </div>

                                        <div class="card">
                                            <h3 class="card-title">"Inputs & Actions"</h3>
                                            <pre class="pre">{GameKind::Text.inputs_info()}</pre>
                                            <div class="subtle" style="margin-top: 8px; line-height: 1.5;">
                                                {move || format!(
                                                    "Trial ms={}  ε={}  α={} ",
                                                    trial_period_ms.get(),
                                                    fmt_f32_fixed(exploration_eps.get(), 2),
                                                    fmt_f32_fixed(meaning_alpha.get(), 1)
                                                )}
                                            </div>
                                        </div>
                                    </div>
                                </div>
                            </div>
                        </Show>

                        // Replay game - Dataset-driven trial inspector
                        <Show when=move || game_kind.get() == GameKind::Replay>
                            <div class="game-page">
                                <div style="display: flex; align-items: center; justify-content: space-between; padding: 16px 20px; background: linear-gradient(135deg, rgba(34, 211, 238, 0.10), rgba(0,0,0,0.30)); border: 1px solid var(--border); border-radius: 16px;">
                                    <div>
                                        <h2 style="margin: 0; font-size: 1.2rem; font-weight: 700; color: var(--text); display: flex; align-items: center; gap: 8px;">
                                            <span style="font-size: 1.4rem;">"📼"</span>
                                            "Replay Dataset"
                                        </h2>
                                        <p style="margin: 4px 0 0 0; color: var(--muted); font-size: 0.8rem;">"Deterministic trials for evaluation + advisor boundary tests"</p>
                                    </div>
                                    <div style="padding: 8px 14px; border-radius: 20px; font-size: 0.85rem; font-weight: 600; background: rgba(34, 211, 238, 0.12); color: #22d3ee; border: 1px solid rgba(34, 211, 238, 0.25);">
                                        {move || {
                                            replay_state
                                                .get()
                                                .map(|s| s.dataset)
                                                .unwrap_or_else(|| "(not running)".to_string())
                                        }}
                                    </div>
                                </div>

                                {training_health_bar_view()}

                                <div class="game-grid">
                                    <div class="game-primary">
                                        <div class="card">
                                            <h3 class="card-title">"Trial Progress"</h3>
                                            <div style="display: flex; justify-content: space-between; gap: 12px; flex-wrap: wrap;">
                                                <div class="subtle">
                                                    "Trial"
                                                    <div style="margin-top: 4px; color: var(--text); font-weight: 900;">
                                                        {move || {
                                                            replay_state
                                                                .get()
                                                                .map(|s| {
                                                                    let idx = s.index.saturating_add(1);
                                                                    format!("{idx} / {}", s.total)
                                                                })
                                                                .unwrap_or_else(|| "—".to_string())
                                                        }}
                                                    </div>
                                                </div>

                                                <div class="subtle">
                                                    "Trial ID"
                                                    <div style="margin-top: 4px; color: var(--text); font-weight: 900; font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, 'Liberation Mono', 'Courier New', monospace;">
                                                        {move || replay_state.get().map(|s| s.trial_id).unwrap_or_else(|| "—".to_string())}
                                                    </div>
                                                </div>
                                            </div>

                                            <div style="margin-top: 12px; height: 10px; border-radius: 999px; background: rgba(148, 163, 184, 0.20); overflow: hidden;">
                                                <div style=move || {
                                                    let pct = replay_state
                                                        .get()
                                                        .and_then(|s| {
                                                            if s.total == 0 {
                                                                None
                                                            } else {
                                                                let idx = s.index.saturating_add(1).min(s.total);
                                                                Some(idx as f32 / s.total as f32)
                                                            }
                                                        })
                                                        .unwrap_or(0.0);
                                                    let width = fmt_f32_fixed(pct.clamp(0.0, 1.0) * 100.0, 1);
                                                    format!(
                                                        "height: 100%; width: {}%; background: linear-gradient(90deg, #22d3ee, #7aa2ff);",
                                                        width
                                                    )
                                                }></div>
                                            </div>

                                            <div style="margin-top: 14px; display: flex; align-items: center; gap: 12px; padding: 12px 14px; background: rgba(0,0,0,0.25); border: 1px solid rgba(148, 163, 184, 0.18); border-radius: 10px; flex-wrap: wrap;">
                                                <div class="subtle">"Braine chose:"</div>
                                                <div style="color: var(--text); font-weight: 900;">
                                                    {move || {
                                                        let a = last_action.get();
                                                        if a.is_empty() { "—".to_string() } else { a }
                                                    }}
                                                </div>
                                                <div style="flex: 1;"></div>
                                                <div style=move || {
                                                    let r = last_reward.get();
                                                    let (bg, fg) = if r > 0.0 {
                                                        ("rgba(34, 197, 94, 0.18)", "#4ade80")
                                                    } else if r < 0.0 {
                                                        ("rgba(239, 68, 68, 0.18)", "#f87171")
                                                    } else {
                                                        ("rgba(100, 116, 139, 0.18)", "#94a3b8")
                                                    };
                                                    format!(
                                                        "padding: 6px 10px; border-radius: 999px; background: {bg}; color: {fg}; font-weight: 900; font-size: 0.85rem;"
                                                    )
                                                }>
                                                    {move || {
                                                        let r = last_reward.get();
                                                        if r > 0.0 { "✓ Correct" } else if r < 0.0 { "✗ Wrong" } else { "—" }
                                                    }}
                                                </div>
                                            </div>

                                            <div class="subtle" style="margin-top: 10px; line-height: 1.5;">
                                                "Tip: If accuracy is stuck, slow the trial, lower ε, and increase α a bit."
                                            </div>
                                        </div>
                                    </div>

                                    <div class="game-secondary">
                                        <div class="card">
                                            <h3 class="card-title">"What Replay Is"</h3>
                                            <div style="color: var(--text); font-size: 0.85rem; line-height: 1.6;">
                                                <div>"Replay is like a stack of flashcards."</div>
                                                <div>"Each flashcard shows some clues (stimuli)."</div>
                                                <div>"Braine picks one button (an action)."</div>
                                                <div>"If it picked the right button, it gets a green point. If not, a red point."</div>
                                                <div style="margin-top: 8px; color: var(--muted);">"Step = do 1 flashcard. Run = keep going."</div>
                                                <div style="margin-top: 6px; color: var(--muted);">"Replay is special because the questions don’t change — it’s great for checking whether a settings change helped."</div>
                                            </div>
                                        </div>

                                        <div class="card">
                                            <h3 class="card-title">"Reward & Signals"</h3>
                                            <pre class="codeblock">{GameKind::Replay.reward_info()}</pre>
                                            <div class="subtle" style="margin-top: 8px; line-height: 1.5;">
                                                "Global shaping: shaped = ((raw + bias) × scale) clamped to [−5, +5]"<br/>
                                                {move || format!(
                                                    "Current: bias={}, scale={}",
                                                    fmt_f32_signed_fixed(reward_bias.get(), 2),
                                                    fmt_f32_fixed(reward_scale.get(), 2)
                                                )}
                                            </div>
                                        </div>

                                        <div class="card">
                                            <h3 class="card-title">"Inputs & Actions"</h3>
                                            <pre class="pre">{GameKind::Replay.inputs_info()}</pre>
                                        </div>
                                    </div>
                                </div>
                            </div>
                        </Show>
                        </div>
                    </Show>
                    </div>

                // Mobile dashboard toggle button (visible only on small screens)
                <button
                    class="dashboard-toggle"
                    on:click=move |_| set_dashboard_open.set(true)
                    title="Open Dashboard"
                >
                    "◀"
                </button>

                // Mobile dashboard overlay (click to close)
                <div
                    class=move || if dashboard_open.get() { "dashboard-overlay open" } else { "dashboard-overlay" }
                    on:click=move |_| set_dashboard_open.set(false)
                ></div>

                // Dashboard (right) - Tabbed panel
                <div class=move || if dashboard_open.get() { "dashboard open" } else { "dashboard" }>
                    <div class="dashboard-tabs">
                        {DashboardTab::all().iter().map(|tab| {
                            let t = *tab;
                            view! {
                                <button
                                    class=move || if dashboard_tab.get() == t { "dashboard-tab active" } else { "dashboard-tab" }
                                    on:click=move |_| set_dashboard_tab.set(t)>
                                    {t.icon()}" "{t.label()}
                                </button>
                            }
                        }).collect::<Vec<_>>()}
                    </div>

                    <div class="dashboard-content">
                        <Show when=move || dashboard_tab.get() == DashboardTab::BrainViz>
                            <div class="stack tight brainviz-stack">
                                <div class="dashboard-pinned brainviz-dock">
                                    <div class="dashboard-pinned-head">
                                        <div class="dashboard-pinned-title">"🧠 BrainViz"</div>
                                        <div class="dashboard-pinned-meta">
                                            {move || {
                                                let n = brainviz_display_nodes.get();
                                                let e = brainviz_display_edges.get();
                                                let avg = brainviz_display_avg_conn.get();
                                                let maxc = brainviz_display_max_conn.get();
                                                format!(
                                                    "{} nodes • {} edges • avg {} • max {}",
                                                    n,
                                                    e,
                                                    fmt_f32_fixed(avg, 2),
                                                    maxc
                                                )
                                            }}
                                        </div>
                                        <button
                                            class="icon-btn"
                                            title=move || if brainviz_is_expanded.get() { "Compact BrainViz" } else { "Expand BrainViz" }
                                            on:click=move |_| {
                                                set_brainviz_is_expanded.update(|v| *v = !*v);
                                            }
                                        >
                                            {move || if brainviz_is_expanded.get() { "⤡" } else { "⤢" }}
                                        </button>
                                    </div>

                                    <p class="subtle">{move || if brainviz_view_mode.get() == "causal" { "Causal view: symbol-to-symbol temporal edges. Node size = frequency, edge color = causal strength." } else { "Substrate view: sampled unit nodes; edges show sparse connection weights." }}</p>
                                    <div class="callout">
                                        <p>"Static data • Drag to rotate • Shift+drag to pan • Wheel to zoom • Refresh to resample • Replay to inspect history"</p>
                                    </div>

                                    <div class="subtle" style="margin-top: 6px;">
                                        "Changes to Nodes/Edges take effect on Refresh."
                                    </div>

                                    <div class="subtle" style="margin-top: 8px;">
                                        {move || {
                                            let src = if idb_loaded.get() { "IndexedDB (brain_image)" } else { "fresh" };
                                            let autosave = if idb_autosave.get() { "on" } else { "off" };
                                            let ts = idb_last_save.get();
                                            if ts.is_empty() {
                                                format!("BBI source: {src} • Autosave: {autosave} • Last save: —")
                                            } else {
                                                format!("BBI source: {src} • Autosave: {autosave} • Last save: {ts}")
                                            }
                                        }}
                                    </div>

                                    <div class="row end wrap" style="margin-top: 10px;">
                                        <label class="label">
                                            <span>"View"</span>
                                            <select
                                                class="input"
                                                on:change=move |ev| {
                                                    let v = event_target_value(&ev);
                                                    set_brainviz_view_mode.set(if v == "causal" { "causal" } else { "substrate" });
                                                }
                                            >
                                                <option value="substrate" selected=move || brainviz_view_mode.get() == "substrate">"Substrate"</option>
                                                <option value="causal" selected=move || brainviz_view_mode.get() == "causal">"Causal"</option>
                                            </select>
                                        </label>
                                        <div class="label" style="display: flex; flex-direction: column; gap: 2px;">
                                            <span>"Nodes"</span>
                                            <div style="display: flex; gap: 2px; align-items: center;">
                                                <button class="btn sm" style="padding: 2px 6px; font-size: 10px;" on:click=move |_| {
                                                    set_brainviz_node_sample.update(|v| *v = (*v).saturating_sub(50).max(16));
                                                }>-50</button>
                                                <button class="btn sm" style="padding: 2px 6px; font-size: 10px;" on:click=move |_| {
                                                    set_brainviz_node_sample.update(|v| *v = (*v).saturating_sub(10).max(16));
                                                }>-10</button>
                                                <input
                                                    class="input compact"
                                                    type="number"
                                                    min="16"
                                                    max="1024"
                                                    step="16"
                                                    style="width: 60px;"
                                                    prop:value=move || brainviz_node_sample.get().to_string()
                                                    on:input=move |ev| {
                                                        if let Ok(v) = event_target_value(&ev).parse::<u32>() {
                                                            set_brainviz_node_sample.set(v.clamp(16, 1024));
                                                        }
                                                    }
                                                />
                                                <button class="btn sm" style="padding: 2px 6px; font-size: 10px;" on:click=move |_| {
                                                    set_brainviz_node_sample.update(|v| *v = (*v + 10).min(1024));
                                                }>+10</button>
                                                <button class="btn sm" style="padding: 2px 6px; font-size: 10px;" on:click=move |_| {
                                                    set_brainviz_node_sample.update(|v| *v = (*v + 50).min(1024));
                                                }>+50</button>
                                            </div>
                                        </div>
                                        <div class="label" style="display: flex; flex-direction: column; gap: 2px;">
                                            <span>"Edges/node"</span>
                                            <div style="display: flex; gap: 2px; align-items: center;">
                                                <button class="btn sm" style="padding: 2px 6px; font-size: 10px;" on:click=move |_| {
                                                    set_brainviz_edges_per_node.update(|v| *v = (*v).saturating_sub(5).max(1));
                                                }>-5</button>
                                                <button class="btn sm" style="padding: 2px 6px; font-size: 10px;" on:click=move |_| {
                                                    set_brainviz_edges_per_node.update(|v| *v = (*v).saturating_sub(1).max(1));
                                                }>-1</button>
                                                <input
                                                    class="input compact"
                                                    type="number"
                                                    min="1"
                                                    max="32"
                                                    step="1"
                                                    style="width: 50px;"
                                                    prop:value=move || brainviz_edges_per_node.get().to_string()
                                                    on:input=move |ev| {
                                                        if let Ok(v) = event_target_value(&ev).parse::<u32>() {
                                                            set_brainviz_edges_per_node.set(v.clamp(1, 32));
                                                        }
                                                    }
                                                />
                                                <button class="btn sm" style="padding: 2px 6px; font-size: 10px;" on:click=move |_| {
                                                    set_brainviz_edges_per_node.update(|v| *v = (*v + 1).min(32));
                                                }>+1</button>
                                                <button class="btn sm" style="padding: 2px 6px; font-size: 10px;" on:click=move |_| {
                                                    set_brainviz_edges_per_node.update(|v| *v = (*v + 5).min(32));
                                                }>+5</button>
                                            </div>
                                        </div>
                                        <button
                                            class="btn sm primary"
                                            on:click=move |_| {
                                                set_brainviz_snapshot_refresh.update(|v| *v = v.wrapping_add(1));
                                            }
                                        >
                                            "Refresh snapshot"
                                        </button>
                                        <button
                                            class="btn sm"
                                            on:click=move |_| {
                                                set_brainviz_zoom.set(1.0);
                                                set_brainviz_pan_x.set(0.0);
                                                set_brainviz_pan_y.set(0.0);
                                                set_brainviz_manual_rotation.set(0.0);
                                                set_brainviz_rotation_x.set(0.0);
                                            }
                                        >
                                            "Reset view"
                                        </button>
                                    </div>

                                    <div class="card" style="margin-top: 10px; padding: 10px;">
                                        <div class="row end wrap" style="gap: 10px;">
                                            <div class="subtle" style="flex: 1; min-width: 220px;">
                                                {move || {
                                                    let running = is_running.get();
                                                    let kind = game_kind.get();
                                                    let frames = brainviz_recordings.with_value(|recs| {
                                                        recs[game_kind_index(kind)].deltas.len()
                                                    });
                                                    if running {
                                                        format!(
                                                            "BrainViz paused while running • recording {} frames ({}).",
                                                            frames,
                                                            kind.label()
                                                        )
                                                    } else if brainviz_replay_active.get() {
                                                        let rk = brainviz_replay_kind.get();
                                                        let ridx = brainviz_replay_idx.get();
                                                        let rframes = brainviz_recordings.with_value(|recs| {
                                                            recs[game_kind_index(rk)].deltas.len()
                                                        });
                                                        let rstep = brainviz_recordings.with_value(|recs| {
                                                            recs[game_kind_index(rk)]
                                                                .deltas
                                                                .get(ridx)
                                                                .map(|f| f.step)
                                                                .unwrap_or(0)
                                                        });
                                                        let kstep = brainviz_recordings.with_value(|recs| {
                                                            let rec = &recs[game_kind_index(rk)];
                                                            rec.keyframes
                                                                .first()
                                                                .map(|(_, k)| k.step)
                                                                .unwrap_or(0)
                                                        });
                                                        format!(
                                                            "Replay: {} frame {}/{} • step {} (k0 step {})",
                                                            rk.display_name(),
                                                            ridx.saturating_add(1),
                                                            rframes,
                                                            rstep,
                                                            kstep
                                                        )
                                                    } else {
                                                        format!(
                                                            "Recorder: {} frames stored for {}",
                                                            frames,
                                                            kind.display_name()
                                                        )
                                                    }
                                                }}
                                            </div>

                                            <label class="label" style="display: flex; align-items: center; gap: 8px;">
                                                <input
                                                    type="checkbox"
                                                    prop:checked=move || brainviz_record_enabled.get()
                                                    on:change=move |ev| {
                                                        set_brainviz_record_enabled.set(event_target_checked(&ev));
                                                    }
                                                />
                                                <span>"Record"</span>
                                            </label>

                                            <button
                                                class="btn sm"
                                                on:click=move |_| {
                                                    let kind = game_kind.get_untracked();
                                                    brainviz_recordings.update_value(|recs| {
                                                        recs[game_kind_index(kind)] = BrainvizGameRecording::default();
                                                    });
                                                    // If we're replaying this game, exit replay.
                                                    if brainviz_replay_active.get_untracked() && brainviz_replay_kind.get_untracked() == kind {
                                                        set_brainviz_replay_active.set(false);
                                                    }
                                                }
                                            >
                                                "Clear clip"
                                            </button>

                                            <button
                                                class=move || if brainviz_replay_active.get() { "btn sm" } else { "btn sm primary" }
                                                on:click=move |_| {
                                                    // Stop the brain before replaying.
                                                    (do_stop_sv.get_value())();

                                                    let kind = game_kind.get_untracked();
                                                    let frames = brainviz_recordings.with_value(|recs| {
                                                        recs[game_kind_index(kind)].deltas.len()
                                                    });

                                                    // UX: if nothing has been recorded yet, make that explicit.
                                                    // Replay only works after the game has completed at least one trial
                                                    // while recording is enabled.
                                                    if frames == 0 {
                                                        set_status.set(
                                                            "BrainViz replay: no frames recorded yet. Run a game with Record enabled (records on trial completion), then Stop and Replay."
                                                                .to_string(),
                                                        );
                                                        // Fall back to a live snapshot so the user sees *something*.
                                                        set_brainviz_snapshot_refresh.update(|v| *v = v.wrapping_add(1));
                                                        set_brainviz_replay_active.set(false);
                                                        return;
                                                    }

                                                    let last = frames.saturating_sub(1);
                                                    set_brainviz_replay_kind.set(kind);
                                                    set_brainviz_replay_idx.set(last);
                                                    set_brainviz_replay_active.set(true);
                                                }
                                            >
                                                "Replay"
                                            </button>

                                            <Show when=move || brainviz_replay_active.get()>
                                                <button
                                                    class="btn sm"
                                                    on:click=move |_| {
                                                        set_brainviz_replay_active.set(false);
                                                    }
                                                >
                                                    "Exit"
                                                </button>
                                            </Show>
                                        </div>

                                        <div class="brainviz-recorder-grid">
                                            <label class="label stack">
                                                <span>"Active amp≥"</span>
                                                <input
                                                    class="input compact"
                                                    type="number"
                                                    min="0"
                                                    max="1"
                                                    step="0.01"
                                                    prop:value=move || fmt_f32_fixed(brainviz_active_amp_threshold.get(), 2)
                                                    on:input=move |ev| {
                                                        if let Ok(v) = event_target_value(&ev).parse::<f32>() {
                                                            set_brainviz_active_amp_threshold.set(v.clamp(0.0, 1.0));
                                                        }
                                                    }
                                                />
                                            </label>

                                            <label class="label stack">
                                                <span>"Phase eps (active)"</span>
                                                <input
                                                    class="input compact"
                                                    type="number"
                                                    min="0"
                                                    max="1"
                                                    step="0.01"
                                                    prop:value=move || fmt_f32_fixed(brainviz_eps_phase_active.get(), 2)
                                                    on:input=move |ev| {
                                                        if let Ok(v) = event_target_value(&ev).parse::<f32>() {
                                                            set_brainviz_eps_phase_active.set(v.max(0.0));
                                                        }
                                                    }
                                                />
                                            </label>

                                            <label class="label stack">
                                                <span>"Phase eps (inactive)"</span>
                                                <input
                                                    class="input compact"
                                                    type="number"
                                                    min="0"
                                                    max="2"
                                                    step="0.01"
                                                    prop:value=move || fmt_f32_fixed(brainviz_eps_phase_inactive.get(), 2)
                                                    on:input=move |ev| {
                                                        if let Ok(v) = event_target_value(&ev).parse::<f32>() {
                                                            set_brainviz_eps_phase_inactive.set(v.max(0.0));
                                                        }
                                                    }
                                                />
                                            </label>

                                            <label class="label stack">
                                                <span>"Record every (trials)"</span>
                                                <input
                                                    class="input compact"
                                                    type="number"
                                                    min="1"
                                                    max="100"
                                                    step="1"
                                                    prop:value=move || brainviz_record_every_trials.get().to_string()
                                                    on:input=move |ev| {
                                                        if let Ok(v) = event_target_value(&ev).parse::<u32>() {
                                                            set_brainviz_record_every_trials.set(v.max(1));
                                                        }
                                                    }
                                                />
                                            </label>

                                            <label class="label brainviz-checkbox">
                                                <input
                                                    type="checkbox"
                                                    prop:checked=move || brainviz_record_edges.get()
                                                    on:change=move |ev| {
                                                        set_brainviz_record_edges.set(event_target_checked(&ev));
                                                    }
                                                />
                                                <span>"Edges"</span>
                                            </label>

                                            <label class="label stack">
                                                <span>"Edge eps"</span>
                                                <input
                                                    class="input compact"
                                                    type="number"
                                                    min="0"
                                                    max="1"
                                                    step="0.001"
                                                    prop:value=move || fmt_f32_fixed(brainviz_eps_weight.get(), 3)
                                                    on:input=move |ev| {
                                                        if let Ok(v) = event_target_value(&ev).parse::<f32>() {
                                                            set_brainviz_eps_weight.set(v.max(0.0));
                                                        }
                                                    }
                                                />
                                            </label>

                                            <label class="label stack">
                                                <span>"Edges every (trials)"</span>
                                                <input
                                                    class="input compact"
                                                    type="number"
                                                    min="1"
                                                    max="100"
                                                    step="1"
                                                    prop:value=move || brainviz_record_edges_every_trials.get().to_string()
                                                    on:input=move |ev| {
                                                        if let Ok(v) = event_target_value(&ev).parse::<u32>() {
                                                            set_brainviz_record_edges_every_trials.set(v.max(1));
                                                        }
                                                    }
                                                />
                                            </label>
                                        </div>

                                        <Show when=move || brainviz_replay_active.get()>
                                            <div class="row end wrap" style="gap: 10px; margin-top: 10px;">
                                                <label class="label">
                                                    <span>"Game"</span>
                                                    <select
                                                        class="input"
                                                        on:change=move |ev| {
                                                            let v = event_target_value(&ev);
                                                            let kind = match v.as_str() {
                                                                "spot" => GameKind::Spot,
                                                                "bandit" => GameKind::Bandit,
                                                                "spot_reversal" => GameKind::SpotReversal,
                                                                "spotxy" => GameKind::SpotXY,
                                                                "pong" => GameKind::Pong,
                                                                "sequence" => GameKind::Sequence,
                                                                "text" => GameKind::Text,
                                                                "replay" => GameKind::Replay,
                                                                _ => GameKind::Spot,
                                                            };
                                                            let last = brainviz_recordings.with_value(|recs| {
                                                                recs[game_kind_index(kind)].deltas.len().saturating_sub(1)
                                                            });
                                                            set_brainviz_replay_kind.set(kind);
                                                            set_brainviz_replay_idx.set(last);
                                                        }
                                                    >
                                                        <option value="spot" selected=move || brainviz_replay_kind.get() == GameKind::Spot>"Spot"</option>
                                                        <option value="bandit" selected=move || brainviz_replay_kind.get() == GameKind::Bandit>"Bandit"</option>
                                                        <option value="spot_reversal" selected=move || brainviz_replay_kind.get() == GameKind::SpotReversal>"Reversal"</option>
                                                        <option value="spotxy" selected=move || brainviz_replay_kind.get() == GameKind::SpotXY>"SpotXY"</option>
                                                        <option value="pong" selected=move || brainviz_replay_kind.get() == GameKind::Pong>"Pong"</option>
                                                        <option value="sequence" selected=move || brainviz_replay_kind.get() == GameKind::Sequence>"Sequence"</option>
                                                        <option value="text" selected=move || brainviz_replay_kind.get() == GameKind::Text>"Text"</option>
                                                        <option value="replay" selected=move || brainviz_replay_kind.get() == GameKind::Replay>"Replay"</option>
                                                    </select>
                                                </label>

                                                <button class="btn sm" on:click=move |_| {
                                                    set_brainviz_replay_idx.update(|i| *i = i.saturating_sub(1));
                                                }>
                                                    "◀"
                                                </button>

                                                <input
                                                    class="input"
                                                    type="range"
                                                    min="0"
                                                    prop:max=move || {
                                                        let k = brainviz_replay_kind.get();
                                                        let n = brainviz_recordings.with_value(|recs| {
                                                                recs[game_kind_index(k)].deltas.len()
                                                        });
                                                        n.saturating_sub(1).to_string()
                                                    }
                                                    prop:value=move || brainviz_replay_idx.get().to_string()
                                                    on:input=move |ev| {
                                                        if let Ok(v) = event_target_value(&ev).parse::<usize>() {
                                                            set_brainviz_replay_idx.set(v);
                                                        }
                                                    }
                                                    style="flex: 1; min-width: 220px;"
                                                />

                                                <button class="btn sm" on:click=move |_| {
                                                    let k = brainviz_replay_kind.get_untracked();
                                                    let n = brainviz_recordings.with_value(|recs| {
                                                        recs[game_kind_index(k)].deltas.len()
                                                    });
                                                    set_brainviz_replay_idx.update(|i| *i = (*i + 1).min(n.saturating_sub(1)));
                                                }>
                                                    "▶"
                                                </button>
                                            </div>
                                        </Show>
                                    </div>

                                    <Show when=move || brainviz_view_mode.get() == "causal">
                                        <div class="card" style="margin-top: 10px; padding: 10px;">
                                            <div class="row end wrap" style="gap: 10px;">
                                                <div style="flex: 1; min-width: 220px;">
                                                    <div style="font-weight: 800; font-size: 0.9rem;">"Causal symbol tags"</div>
                                                    <div class="subtle" style="margin-top: 4px;">
                                                        "Click a node in the causal graph to select a symbol, then assign a color/label."
                                                    </div>
                                                    <div class="subtle" style="margin-top: 6px; font-family: var(--mono);">
                                                        {move || {
                                                            if let Some(id) = brainviz_selected_symbol_id.get() {
                                                                let name = brainviz_selected_symbol_name.get();
                                                                format!("selected: sym {} ({})", id, name)
                                                            } else {
                                                                "selected: (none)".to_string()
                                                            }
                                                        }}
                                                    </div>
                                                </div>

                                                <label class="label" style="display: flex; align-items: center; gap: 8px;">
                                                    <span>"Color"</span>
                                                    <input
                                                        type="color"
                                                        prop:value=move || brainviz_selected_symbol_color.get()
                                                        on:input=move |ev| {
                                                            set_brainviz_selected_symbol_color.set(event_target_value(&ev));
                                                        }
                                                    />
                                                </label>

                                                <label class="label" style="display: flex; align-items: center; gap: 8px;">
                                                    <span>"Label"</span>
                                                    <input
                                                        class="input compact"
                                                        type="text"
                                                        placeholder="e.g. reward link"
                                                        style="width: 220px;"
                                                        prop:value=move || brainviz_selected_symbol_label.get()
                                                        on:input=move |ev| {
                                                            set_brainviz_selected_symbol_label.set(event_target_value(&ev));
                                                        }
                                                    />
                                                </label>

                                                <button
                                                    class="btn sm primary"
                                                    on:click=move |_| {
                                                        let Some(id) = brainviz_selected_symbol_id.get_untracked() else {
                                                            set_status.set("Causal tag: select a symbol first".to_string());
                                                            return;
                                                        };
                                                        let color = brainviz_selected_symbol_color.get_untracked();
                                                        let label = brainviz_selected_symbol_label.get_untracked();

                                                        set_brainviz_symbol_tags.update(|m| {
                                                            m.insert(
                                                                id,
                                                                BrainvizSymbolTag {
                                                                    color: color.clone(),
                                                                    label: label.clone(),
                                                                },
                                                            );
                                                            save_brainviz_symbol_tags(m);
                                                        });
                                                        set_status.set(format!("Causal tag set: sym {id} {color} {label}"));
                                                    }
                                                >
                                                    "Set"
                                                </button>

                                                <button
                                                    class="btn sm"
                                                    on:click=move |_| {
                                                        let Some(id) = brainviz_selected_symbol_id.get_untracked() else {
                                                            return;
                                                        };
                                                        set_brainviz_symbol_tags.update(|m| {
                                                            m.remove(&id);
                                                            save_brainviz_symbol_tags(m);
                                                        });
                                                        set_status.set(format!("Causal tag cleared: sym {id}"));
                                                    }
                                                >
                                                    "Clear"
                                                </button>
                                            </div>
                                        </div>
                                    </Show>

                                    <Show when=move || brainviz_view_mode.get() == "causal">
                                        <div class="card" style="margin-top: 10px; padding: 10px;">
                                            <div class="row end wrap" style="gap: 10px;">
                                                <div style="flex: 1; min-width: 220px;">
                                                    <div style="font-weight: 800; font-size: 0.9rem;">"Causal filters"</div>
                                                    <div class="subtle" style="margin-top: 4px;">
                                                        "Prefix filter matches symbol name starts-with (e.g. pair::spot::). Focus limits to edges touching the selected symbol."
                                                    </div>
                                                    <div class="subtle" style="margin-top: 6px; font-family: var(--mono);">
                                                        {move || {
                                                            format!(
                                                                "filtered: {} nodes • {} edges",
                                                                brainviz_display_nodes.get(),
                                                                brainviz_display_edges.get()
                                                            )
                                                        }}
                                                    </div>
                                                </div>

                                                <label class="label" style="display: flex; align-items: center; gap: 8px;">
                                                    <span>"Prefix"</span>
                                                    <input
                                                        class="input compact"
                                                        type="text"
                                                        placeholder="pair::spot::"
                                                        style="width: 220px;"
                                                        prop:value=move || brainviz_causal_filter_prefix.get()
                                                        on:input=move |ev| {
                                                            set_brainviz_causal_filter_prefix.set(event_target_value(&ev));
                                                        }
                                                    />
                                                </label>

                                                <label class="label" style="display: flex; align-items: center; gap: 8px;">
                                                    <input
                                                        type="checkbox"
                                                        prop:checked=move || brainviz_causal_focus_selected.get()
                                                        on:change=move |ev| {
                                                            set_brainviz_causal_focus_selected.set(event_target_checked(&ev));
                                                        }
                                                    />
                                                    <span>"Focus selected"</span>
                                                </label>

                                                <label class="label" style="display: flex; align-items: center; gap: 8px;">
                                                    <input
                                                        type="checkbox"
                                                        prop:checked=move || brainviz_causal_hide_isolates.get()
                                                        on:change=move |ev| {
                                                            set_brainviz_causal_hide_isolates.set(event_target_checked(&ev));
                                                        }
                                                    />
                                                    <span>"Hide isolates"</span>
                                                </label>

                                                <button
                                                    class="btn sm"
                                                    on:click=move |_| {
                                                        set_brainviz_causal_filter_prefix.set(String::new());
                                                        set_brainviz_causal_focus_selected.set(false);
                                                        set_brainviz_causal_hide_isolates.set(true);
                                                    }
                                                >
                                                    "Clear filters"
                                                </button>
                                            </div>
                                        </div>
                                    </Show>

                                    <div class="card" style="margin-top: 10px; padding: 10px;">
                                        <div class="row end wrap" style="gap: 10px;">
                                            <div style="flex: 1; min-width: 220px;">
                                                <div style="font-weight: 800; font-size: 0.9rem;">"Node tags"</div>
                                                <div class="subtle" style="margin-top: 4px;">
                                                    "Click a node in the BrainViz canvas to select it, then assign a color/label."
                                                </div>
                                                <div class="subtle" style="margin-top: 6px; font-family: var(--mono);">
                                                    {move || {
                                                        if let Some(id) = brainviz_selected_node_id.get() {
                                                            format!("selected: unit {}", id)
                                                        } else {
                                                            "selected: (none)".to_string()
                                                        }
                                                    }}
                                                </div>
                                            </div>

                                            <label class="label" style="display: flex; align-items: center; gap: 8px;">
                                                <span>"Color"</span>
                                                <input
                                                    type="color"
                                                    prop:value=move || brainviz_selected_tag_color.get()
                                                    on:input=move |ev| {
                                                        set_brainviz_selected_tag_color.set(event_target_value(&ev));
                                                    }
                                                />
                                            </label>

                                            <label class="label" style="display: flex; align-items: center; gap: 8px;">
                                                <span>"Label"</span>
                                                <input
                                                    class="input compact"
                                                    type="text"
                                                    placeholder="e.g. spot::left or goal-node"
                                                    style="width: 220px;"
                                                    prop:value=move || brainviz_selected_tag_label.get()
                                                    on:input=move |ev| {
                                                        set_brainviz_selected_tag_label.set(event_target_value(&ev));
                                                    }
                                                />
                                            </label>

                                            <button
                                                class="btn sm primary"
                                                on:click=move |_| {
                                                    let Some(id) = brainviz_selected_node_id.get_untracked() else {
                                                        set_status.set("BrainViz tag: select a node first".to_string());
                                                        return;
                                                    };
                                                    let color = brainviz_selected_tag_color.get_untracked();
                                                    let label = brainviz_selected_tag_label.get_untracked();

                                                    set_brainviz_node_tags.update(|m| {
                                                        m.insert(
                                                            id,
                                                            BrainvizNodeTag {
                                                                color: color.clone(),
                                                                label: label.clone(),
                                                            },
                                                        );
                                                        save_brainviz_node_tags(m);
                                                    });
                                                    set_status.set(format!("BrainViz tag set: unit {id} {color} {label}"));
                                                }
                                            >
                                                "Set"
                                            </button>

                                            <button
                                                class="btn sm"
                                                on:click=move |_| {
                                                    let Some(id) = brainviz_selected_node_id.get_untracked() else {
                                                        return;
                                                    };
                                                    set_brainviz_node_tags.update(|m| {
                                                        m.remove(&id);
                                                        save_brainviz_node_tags(m);
                                                    });
                                                    set_status.set(format!("BrainViz tag cleared: unit {id}"));
                                                }
                                            >
                                                "Clear"
                                            </button>

                                            <button
                                                class="btn sm"
                                                on:click=move |_| {
                                                    set_brainviz_node_tags.set(std::collections::HashMap::new());
                                                    save_brainviz_node_tags(&std::collections::HashMap::new());
                                                    set_status.set("BrainViz tags cleared".to_string());
                                                }
                                            >
                                                "Clear all"
                                            </button>
                                        </div>
                                    </div>

                                    <div style="position: relative;">
                                        <canvas
                                            node_ref=brain_viz_ref
                                            width="900"
                                            height="520"
                                            class=move || {
                                                if brainviz_is_expanded.get() {
                                                    "canvas brainviz brainviz-expanded"
                                                } else {
                                                    "canvas brainviz"
                                                }
                                            }
                                            on:pointerdown=move |ev: web_sys::PointerEvent| {
                                                if ev.button() != 0 {
                                                    return;
                                                }
                                                ev.prevent_default();

                                                let pid = ev.pointer_id();
                                                let sx = ev.client_x() as f64;
                                                let sy = ev.client_y() as f64;

                                                // Shift+drag pans; otherwise rotate.
                                                let pan_mode = ev.shift_key();

                                                set_brainviz_drag.set(Some((
                                                    pid,
                                                    sx,
                                                    sy,
                                                    brainviz_pan_x.get_untracked(),
                                                    brainviz_pan_y.get_untracked(),
                                                    brainviz_manual_rotation.get_untracked(),
                                                    brainviz_rotation_x.get_untracked(),
                                                    pan_mode,
                                                )));

                                                if let Some(t) = ev.current_target() {
                                                    if let Ok(el) = t.dyn_into::<web_sys::Element>() {
                                                        let _ = el.set_pointer_capture(pid);
                                                    }
                                                }
                                            }
                                            on:pointermove=move |ev: web_sys::PointerEvent| {
                                                let Some((pid, sx, sy, ox, oy, ory, orx, pan_mode)) =
                                                    brainviz_drag.get_untracked()
                                                else {
                                                    return;
                                                };
                                                if ev.pointer_id() != pid {
                                                    return;
                                                }

                                                ev.prevent_default();

                                                let dx = (ev.client_x() as f64 - sx) as f32;
                                                let dy = (ev.client_y() as f64 - sy) as f32;

                                                if pan_mode {
                                                    set_brainviz_pan_x.set(ox + dx);
                                                    set_brainviz_pan_y.set(oy + dy);
                                                } else {
                                                    // Rotation sensitivity tuned for both mouse and touch.
                                                    let ry = ory + dx * 0.010;
                                                    let rx = (orx + dy * 0.010).clamp(-1.2, 1.2);
                                                    set_brainviz_manual_rotation.set(ry);
                                                    set_brainviz_rotation_x.set(rx);
                                                }
                                            }
                                            on:pointerup=move |ev: web_sys::PointerEvent| {
                                                let Some((pid, sx, sy, ..)) = brainviz_drag.get_untracked() else {
                                                    return;
                                                };
                                                if ev.pointer_id() != pid {
                                                    return;
                                                }

                                                // Treat a small-movement pointer-up as a click for node selection.
                                                let dx = (ev.client_x() as f64 - sx).abs();
                                                let dy = (ev.client_y() as f64 - sy).abs();
                                                let is_click = dx * dx + dy * dy <= 16.0; // 4px radius

                                                set_brainviz_drag.set(None);
                                                if let Some(t) = ev.current_target() {
                                                    if let Ok(el) = t.dyn_into::<web_sys::Element>() {
                                                        let _ = el.release_pointer_capture(pid);
                                                    }
                                                }

                                                if !is_click {
                                                    return;
                                                }

                                                // Hit-test against the most recently rendered node positions.
                                                let Some(t) = ev.current_target() else {
                                                    return;
                                                };
                                                let Ok(el) = t.dyn_into::<web_sys::Element>() else {
                                                    return;
                                                };
                                                let rect = el.get_bounding_client_rect();
                                                let cx = ev.client_x() as f64 - rect.left();
                                                let cy = ev.client_y() as f64 - rect.top();

                                                if brainviz_view_mode.get_untracked() == "causal" {
                                                    let mut picked: Option<charts::CausalHitNode> = None;
                                                    let mut best_d2 = f64::INFINITY;
                                                    brainviz_causal_hit_nodes.with_value(|hits| {
                                                        for h in hits {
                                                            let dx = h.x - cx;
                                                            let dy = h.y - cy;
                                                            let d2 = dx * dx + dy * dy;
                                                            let r2 = (h.r + 3.0) * (h.r + 3.0);
                                                            if d2 <= r2 && d2 < best_d2 {
                                                                best_d2 = d2;
                                                                picked = Some(h.clone());
                                                            }
                                                        }
                                                    });

                                                    if let Some(h) = picked {
                                                        set_brainviz_selected_symbol_id.set(Some(h.id));
                                                        set_brainviz_selected_symbol_name.set(h.name.clone());

                                                        let tags = brainviz_symbol_tags.get_untracked();
                                                        if let Some(tag) = tags.get(&h.id) {
                                                            set_brainviz_selected_symbol_color
                                                                .set(tag.color.clone());
                                                            set_brainviz_selected_symbol_label
                                                                .set(tag.label.clone());
                                                        } else {
                                                            set_brainviz_selected_symbol_label
                                                                .set(String::new());
                                                        }

                                                        set_status.set(format!("selected sym {} ({})", h.id, h.name));
                                                    }
                                                } else {
                                                    let mut picked: Option<charts::BrainVizHitNode> = None;
                                                    let mut best_d2 = f64::INFINITY;
                                                    brainviz_hit_nodes.with_value(|hits| {
                                                        for h in hits {
                                                            let dx = h.x - cx;
                                                            let dy = h.y - cy;
                                                            let d2 = dx * dx + dy * dy;
                                                            let r2 = (h.r + 3.0) * (h.r + 3.0);
                                                            if d2 <= r2 && d2 < best_d2 {
                                                                best_d2 = d2;
                                                                picked = Some(*h);
                                                            }
                                                        }
                                                    });

                                                    if let Some(h) = picked {
                                                        set_brainviz_selected_node_id.set(Some(h.id));

                                                        // If this node already has a tag, preload its fields.
                                                        let tags = brainviz_node_tags.get_untracked();
                                                        if let Some(tag) = tags.get(&h.id) {
                                                            set_brainviz_selected_tag_color
                                                                .set(tag.color.clone());
                                                            set_brainviz_selected_tag_label
                                                                .set(tag.label.clone());
                                                        } else {
                                                            set_brainviz_selected_tag_label
                                                                .set(String::new());
                                                        }

                                                        set_status.set(format!("selected unit {}", h.id));
                                                    }
                                                }
                                            }
                                            on:pointercancel=move |ev: web_sys::PointerEvent| {
                                                let Some((pid, ..)) = brainviz_drag.get_untracked() else {
                                                    return;
                                                };
                                                if ev.pointer_id() != pid {
                                                    return;
                                                }
                                                set_brainviz_drag.set(None);
                                                if let Some(t) = ev.current_target() {
                                                    if let Ok(el) = t.dyn_into::<web_sys::Element>() {
                                                        let _ = el.release_pointer_capture(pid);
                                                    }
                                                }
                                            }
                                            on:wheel=move |ev: web_sys::WheelEvent| {
                                                ev.prevent_default();
                                                let base = brainviz_zoom.get_untracked();
                                                let speed = if ev.ctrl_key() { 0.002 } else { 0.001 };
                                                let scale = (-(ev.delta_y() as f32) * speed).exp();
                                                let next = (base * scale).clamp(0.25, 8.0);
                                                set_brainviz_zoom.set(next);
                                            }
                                        ></canvas>
                                    </div>

                                    // Legend with node type descriptions
                                    <div style="margin-top: 12px; padding: 10px; background: rgba(10,15,26,0.5); border: 1px solid var(--border); border-radius: 8px;">
                                        <div style="font-size: 0.8rem; color: var(--muted); margin-bottom: 8px; font-weight: 600;">"Node Types"</div>
                                        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 8px; font-size: 0.75rem;">
                                            <div style="display: flex; align-items: flex-start; gap: 8px;">
                                                <div style="width: 12px; height: 12px; border-radius: 50%; background: rgb(255, 153, 102); flex-shrink: 0; margin-top: 2px;"></div>
                                                <div>
                                                    <strong style="color: var(--text);">"Sensors"</strong>
                                                    <div style="color: var(--muted);">"Input units that receive external stimuli from the environment"</div>
                                                </div>
                                            </div>
                                            <div style="display: flex; align-items: flex-start; gap: 8px;">
                                                <div style="width: 12px; height: 12px; border-radius: 50%; background: rgb(74, 222, 128); flex-shrink: 0; margin-top: 2px;"></div>
                                                <div>
                                                    <strong style="color: var(--text);">"Groups"</strong>
                                                    <div style="color: var(--muted);">"Action units that form output groups for behavior/decisions"</div>
                                                </div>
                                            </div>
                                            <div style="display: flex; align-items: flex-start; gap: 8px;">
                                                <div style="width: 12px; height: 12px; border-radius: 50%; background: rgb(251, 191, 36); flex-shrink: 0; margin-top: 2px;"></div>
                                                <div>
                                                    <strong style="color: var(--text);">"Regular"</strong>
                                                    <div style="color: var(--muted);">"Free units that form associations through learning dynamics"</div>
                                                </div>
                                            </div>
                                            <div style="display: flex; align-items: flex-start; gap: 8px;">
                                                <div style="width: 12px; height: 12px; border-radius: 50%; background: rgb(148, 163, 184); flex-shrink: 0; margin-top: 2px;"></div>
                                                <div>
                                                    <strong style="color: var(--text);">"Concepts"</strong>
                                                    <div style="color: var(--muted);">"Reserved units that have formed stable engrams via imprinting"</div>
                                                </div>
                                            </div>
                                        </div>
                                    </div>
                                </div>
                            </div>
                        </Show>


                        <Show when=move || dashboard_tab.get() == DashboardTab::Inspect>
                            <div class="stack">
                                <div class="card">
                                    <div class="row end wrap" style="justify-content: space-between; gap: 10px;">
                                        <div>
                                            <h3 class="card-title">"🔎 Inspect"</h3>
                                            <div class="subtle">
                                                "Lightweight diagnostics from the current BrainViz snapshot."
                                            </div>
                                        </div>

                                        <button
                                            class="btn sm primary"
                                            on:click=move |_| {
                                                set_brainviz_snapshot_refresh.update(|v| *v = v.wrapping_add(1));
                                            }
                                        >
                                            "Refresh snapshot"
                                        </button>
                                    </div>

                                    <div class="subtle" style="margin-top: 10px;">
                                        {move || {
                                            let n = brainviz_display_nodes.get();
                                            let e = brainviz_display_edges.get();
                                            let avg = brainviz_display_avg_conn.get();
                                            let maxc = brainviz_display_max_conn.get();
                                            format!(
                                                "Snapshot: {} nodes • {} edges • avg {} • max {}",
                                                n,
                                                e,
                                                fmt_f32_fixed(avg, 2),
                                                maxc
                                            )
                                        }}
                                    </div>

                                    <div class="subtle" style="margin-top: 6px; font-family: var(--mono);">
                                        {move || {
                                            let d = diag.get();
                                            format!(
                                                "Brain: {} units • {} conns • births(last)={} • pruned(last)={}",
                                                d.unit_count, d.connection_count, d.births_last_step, d.pruned_last_step
                                            )
                                        }}
                                    </div>

                                    <div class="row wrap" style="margin-top: 12px; gap: 14px; align-items: flex-start;">
                                        <div style="min-width: 260px;">
                                            <div class="subtle">{move || format!(
                                                "Trials: {} • recent rate: {}%",
                                                trials.get(),
                                                fmt_f32_fixed(recent_rate.get() * 100.0, 0)
                                            )}</div>
                                            <div class="subtle">{move || {
                                                if game_kind.get() == GameKind::Pong {
                                                    format!(
                                                        "Action match: {} • mismatch: {}",
                                                        correct_count.get(),
                                                        incorrect_count.get()
                                                    )
                                                } else {
                                                    format!(
                                                        "Correct: {} • Incorrect: {}",
                                                        correct_count.get(),
                                                        incorrect_count.get()
                                                    )
                                                }
                                            }}</div>

                                            <Show when=move || game_kind.get() == GameKind::Pong>
                                                <div class="subtle">{move || {
                                                    pong_state
                                                        .get()
                                                        .map(|s| format!("Pong score: Hits {} • Misses {}", s.hits, s.misses))
                                                        .unwrap_or_else(|| "Pong score: Hits 0 • Misses 0".to_string())
                                                }}</div>
                                            </Show>
                                            <div class="subtle">{move || format!(
                                                "Last: action={} • reward={}",
                                                last_action.get(),
                                                fmt_f32_signed_fixed(last_reward.get(), 2)
                                            )}</div>
                                            <div class="subtle" style="margin-top: 6px; font-weight: 800;">
                                                {move || learning_milestone.get()}
                                            </div>

                                            <div class="subtle" style="margin-top: 8px; font-family: var(--mono);">
                                                {move || {
                                                    let ls = learn_stats.get();
                                                    let commit = if ls.plasticity_committed { "y" } else { "n" };
                                                    if ls.plasticity_budget > 0.0 {
                                                        format!(
                                                            "learn: elig_l1={} dw_l1={} edges={} commit={} budget={}/{}",
                                                            fmt_f32_fixed(ls.eligibility_l1, 2),
                                                            fmt_f32_fixed(ls.plasticity_l1, 2),
                                                            ls.plasticity_edges,
                                                            commit,
                                                            fmt_f32_fixed(ls.plasticity_budget_used, 2),
                                                            fmt_f32_fixed(ls.plasticity_budget, 2)
                                                        )
                                                    } else {
                                                        format!(
                                                            "learn: elig_l1={} dw_l1={} edges={} commit={}",
                                                            fmt_f32_fixed(ls.eligibility_l1, 2),
                                                            fmt_f32_fixed(ls.plasticity_l1, 2),
                                                            ls.plasticity_edges,
                                                            commit
                                                        )
                                                    }
                                                }}
                                            </div>
                                        </div>

                                        <div style="min-width: 280px;">
                                            <div class="subtle" style="margin-bottom: 6px;">"Perf (last-100 rate)"</div>
                                            <canvas
                                                node_ref=inspect_perf_spark_ref
                                                width="280"
                                                height="60"
                                                class="canvas"
                                                style="border: 1px solid var(--border); border-radius: 10px;"
                                            ></canvas>
                                        </div>

                                        <div style="min-width: 280px;">
                                            <div class="subtle" style="margin-bottom: 6px;">"Reward"</div>
                                            <canvas
                                                node_ref=inspect_reward_spark_ref
                                                width="280"
                                                height="60"
                                                class="canvas"
                                                style="border: 1px solid var(--border); border-radius: 10px;"
                                            ></canvas>
                                        </div>

                                        <div style="min-width: 280px;">
                                            <div class="subtle" style="margin-bottom: 6px;">"|Δw| (plasticity L1)"</div>
                                            <canvas
                                                node_ref=inspect_learn_plasticity_spark_ref
                                                width="280"
                                                height="60"
                                                class="canvas"
                                                style="border: 1px solid var(--border); border-radius: 10px;"
                                            ></canvas>
                                        </div>

                                        <div style="min-width: 280px;">
                                            <div class="subtle" style="margin-bottom: 6px;">"Eligibility L1"</div>
                                            <canvas
                                                node_ref=inspect_learn_elig_spark_ref
                                                width="280"
                                                height="60"
                                                class="canvas"
                                                style="border: 1px solid var(--border); border-radius: 10px;"
                                            ></canvas>
                                        </div>

                                        <div style="min-width: 280px;">
                                            <div class="subtle" style="margin-bottom: 6px;">"Homeostasis bias L1"</div>
                                            <canvas
                                                node_ref=inspect_learn_homeostasis_spark_ref
                                                width="280"
                                                height="60"
                                                class="canvas"
                                                style="border: 1px solid var(--border); border-radius: 10px;"
                                            ></canvas>
                                        </div>
                                    </div>
                                </div>

                                <div class="card">
                                    <h3 class="card-title">"Phase histogram"</h3>
                                    <pre class="pre">{move || {
                                        // 16-bin histogram over [0, 2π)
                                        let bins = 16usize;
                                        let tau = 2.0 * std::f32::consts::PI;
                                        let mut counts = vec![0u32; bins];
                                        let mut total = 0u32;
                                        brainviz_points.with(|pts| {
                                            for p in pts {
                                                let mut ph = p.phase;
                                                while ph < 0.0 { ph += tau; }
                                                while ph >= tau { ph -= tau; }
                                                let b = (((ph / tau) * (bins as f32)) as usize).min(bins - 1);
                                                counts[b] += 1;
                                                total += 1;
                                            }
                                        });
                                        let maxc = counts.iter().copied().max().unwrap_or(1).max(1);
                                        let mut out = String::new();
                                        out.push_str(&format!("total={} bins={}\n\n", total, bins));
                                        for (i, c) in counts.iter().enumerate() {
                                            let width = (((*c as f32) / (maxc as f32)) * 24.0).round() as usize;
                                            let bar = "▇".repeat(width);
                                            out.push_str(&format!("{:>2}: {:>4} {}\n", i, c, bar));
                                        }
                                        out
                                    }}</pre>
                                </div>

                                <div class="card">
                                    <h3 class="card-title">"Top-K units (by salience01)"</h3>
                                    <pre class="pre">{move || {
                                        let mut rows: Vec<(u32, f32, f32, f32, &'static str)> = Vec::new();
                                        brainviz_points.with(|pts| {
                                            for p in pts {
                                                let kind = if p.is_sensor_member {
                                                    "sensor"
                                                } else if p.is_group_member {
                                                    "group"
                                                } else if p.is_reserved {
                                                    "reserved"
                                                } else {
                                                    "unit"
                                                };
                                                rows.push((p.id, p.salience01, p.amp01, p.rel_age, kind));
                                            }
                                        });
                                        rows.sort_by(|a, b| b.1.total_cmp(&a.1));
                                        let k = 12usize.min(rows.len());
                                        let mut out = String::new();
                                        out.push_str("id       kind      salience  amp01   rel_age\n");
                                        out.push_str("--------------------------------------------\n");
                                        for (id, sal, amp, age, kind) in rows.into_iter().take(k) {
                                            let sal = fmt_f32_fixed(sal, 3);
                                            let amp = fmt_f32_fixed(amp, 2);
                                            let age = fmt_f32_fixed(age, 2);
                                            out.push_str(&format!(
                                                "{:>7}  {:<8}  {:>7}  {:>5}  {:>7}\n",
                                                id, kind, sal, amp, age
                                            ));
                                        }
                                        out
                                    }}</pre>
                                </div>

                                <div class="card">
                                    <h3 class="card-title">"Neighborhood lens"</h3>
                                    <div class="row end wrap" style="gap: 10px;">
                                        <label class="label">
                                            <span>"Unit id"</span>
                                            <input
                                                class="input compact"
                                                type="number"
                                                min="0"
                                                style="width: 120px;"
                                                prop:value=move || inspect_neighbor_unit_id.get().to_string()
                                                on:input=move |ev| {
                                                    if let Ok(v) = event_target_value(&ev).parse::<u32>() {
                                                        set_inspect_neighbor_unit_id.set(v);
                                                    }
                                                }
                                            />
                                        </label>
                                        <button
                                            class="btn sm"
                                            on:click={
                                                let runtime = runtime.clone();
                                                move |_| {
                                                    let id = inspect_neighbor_unit_id.get_untracked() as usize;
                                                    let mut neigh: Vec<(usize, f32)> = runtime
                                                        .with_value(|r| r.brain.neighbors(id).collect());
                                                    neigh.sort_by(|a, b| b.1.abs().total_cmp(&a.1.abs()));
                                                    let mut out = String::new();
                                                    out.push_str(&format!("neighbors({}) top={}\n\n", id, 16));
                                                    for (t, w) in neigh.into_iter().take(16) {
                                                        let w = fmt_f32_signed_fixed(w, 4);
                                                        out.push_str(&format!("{:>6}  {:>9}\n", t, w));
                                                    }
                                                    set_inspect_neighbors_text.set(out);
                                                }
                                            }
                                        >
                                            "Load neighbors"
                                        </button>
                                    </div>
                                    <pre class="pre" style="margin-top: 10px;">{move || inspect_neighbors_text.get()}</pre>
                                    <div class="subtle">"Tip: copy a unit id from Top-K above."</div>
                                </div>

                                <div class="card">
                                    <h3 class="card-title">"Single-unit oscilloscope"</h3>
                                    <div class="row end wrap" style="gap: 10px;">
                                        <label class="label">
                                            <span>"Unit id"</span>
                                            <input
                                                class="input compact"
                                                type="number"
                                                min="0"
                                                style="width: 120px;"
                                                prop:value=move || inspect_scope_unit_id.get().to_string()
                                                on:input=move |ev| {
                                                    if let Ok(v) = event_target_value(&ev).parse::<u32>() {
                                                        set_inspect_scope_unit_id.set(v);
                                                    }
                                                }
                                            />
                                        </label>
                                        <div class="subtle">"Samples on Refresh / Replay step"</div>
                                        <button
                                            class="btn sm"
                                            on:click=move |_| {
                                                inspect_scope_amp.update_value(|h| h.clear());
                                                inspect_scope_phase_sin.update_value(|h| h.clear());
                                                set_inspect_scope_version.update(|v| *v = v.wrapping_add(1));
                                            }
                                        >
                                            "Clear"
                                        </button>
                                    </div>

                                    <div class="row wrap" style="margin-top: 10px; gap: 14px;">
                                        <div style="min-width: 280px;">
                                            <div class="subtle" style="margin-bottom: 6px;">"amp01"</div>
                                            <canvas
                                                node_ref=inspect_scope_amp_ref
                                                width="280"
                                                height="60"
                                                class="canvas"
                                                style="border: 1px solid var(--border); border-radius: 10px;"
                                            ></canvas>
                                        </div>
                                        <div style="min-width: 280px;">
                                            <div class="subtle" style="margin-bottom: 6px;">"sin(phase)"</div>
                                            <canvas
                                                node_ref=inspect_scope_phase_ref
                                                width="280"
                                                height="60"
                                                class="canvas"
                                                style="border: 1px solid var(--border); border-radius: 10px;"
                                            ></canvas>
                                        </div>
                                    </div>
                                </div>

                                <div class="card">
                                    <h3 class="card-title">"State change log (per trial)"</h3>
                                    <pre class="pre">{move || {
                                        let _ = inspect_trial_events_version.get();
                                        let events: Vec<InspectTrialEvent> =
                                            inspect_trial_events.with_value(|v| v.clone());
                                        if events.is_empty() {
                                            return "(no trials yet)".to_string();
                                        }
                                        let tail = events.into_iter().rev().take(60).collect::<Vec<_>>();
                                        let mut out = String::new();
                                        for e in tail.into_iter().rev() {
                                            let rate = fmt_f32_fixed(e.recent_rate * 100.0, 1);
                                            let reward = fmt_f32_signed_fixed(e.reward, 2);
                                            out.push_str(&format!(
                                                "step {:>6}  {}  trial {:>5}  rate {:>5}%  reward {:>6}  action {}\n",
                                                e.step,
                                                e.game.label(),
                                                e.trial,
                                                rate,
                                                reward,
                                                e.action
                                            ));
                                        }
                                        out
                                    }}</pre>
                                </div>

                                <div class="card">
                                    <h3 class="card-title">"Recent actions"</h3>
                                    <pre class="pre">{move || {
                                        let events: Vec<String> = choice_events.with_value(|v| v.clone());
                                        let tail = events.into_iter().rev().take(30).collect::<Vec<_>>();
                                        let mut out = String::new();
                                        for (i, a) in tail.into_iter().rev().enumerate() {
                                            out.push_str(&format!("{:>2}: {}\n", i, a));
                                        }
                                        if out.is_empty() { "(no actions yet)".to_string() } else { out }
                                    }}</pre>
                                    <div class="subtle">"(Next: local neighborhood viz + richer per-trial deltas.)"</div>
                                </div>
                            </div>
                        </Show>


                        <Show when=move || dashboard_tab.get() == DashboardTab::Settings>
                            {
                                let bind = move |k: &str| -> Option<(ReadSignal<f32>, WriteSignal<f32>)> {
                                    match k {
                                        "dt" => Some((cfg_dt, set_cfg_dt)),
                                        "base_freq" => Some((cfg_base_freq, set_cfg_base_freq)),
                                        "global_inhibition" => Some((cfg_global_inhibition, set_cfg_global_inhibition)),
                                        "noise_amp" => Some((cfg_noise_amp, set_cfg_noise_amp)),
                                        "noise_phase" => Some((cfg_noise_phase, set_cfg_noise_phase)),
                                        "hebb_rate" => Some((cfg_hebb_rate, set_cfg_hebb_rate)),
                                        "forget_rate" => Some((cfg_forget_rate, set_cfg_forget_rate)),
                                        "imprint_rate" => Some((cfg_imprint_rate, set_cfg_imprint_rate)),
                                        "salience_decay" => Some((cfg_salience_decay, set_cfg_salience_decay)),
                                        "salience_gain" => Some((cfg_salience_gain, set_cfg_salience_gain)),
                                        "prune_below" => Some((cfg_prune_below, set_cfg_prune_below)),
                                        "coactive_threshold" => Some((cfg_coactive_threshold, set_cfg_coactive_threshold)),
                                        "phase_lock_threshold" => Some((cfg_phase_lock_threshold, set_cfg_phase_lock_threshold)),
                                        "causal_decay" => Some((cfg_causal_decay, set_cfg_causal_decay)),
                                        _ => None,
                                    }
                                };

                                view! {
                                    <div class="stack settings-stack">
                                        <div class="card">
                                            <div class="settings-header">
                                                <div>
                                                    <h3 class="card-title">"⚙️ Brain Configuration"</h3>
                                                    <p class="subtle">
                                                        "Fine-tune substrate dynamics, learning rates, and maintenance thresholds. "
                                                        "All values are validated and clamped to safe ranges. "
                                                        <span style="color: #fbbf24;">"⚠ Warnings"</span>
                                                        " appear when values fall outside recommended ranges."
                                                    </p>
                                                </div>
                                                <label class="label settings-advanced">
                                                    <input
                                                        type="checkbox"
                                                        prop:checked=move || settings_advanced.get()
                                                        on:change=move |ev| {
                                                            set_settings_advanced.set(event_target_checked(&ev));
                                                        }
                                                    />
                                                    <span>"Show advanced"</span>
                                                </label>
                                            </div>
                                        </div>

                                        <div class="card">
                                            <h3 class="card-title">"🧬 Neurogenesis"</h3>
                                            <p class="subtle">"Add new computational units to expand substrate capacity. This increases memory usage and compute requirements proportionally."</p>

                                            <div class="row end wrap" style="gap: 12px; margin-top: 12px;">
                                                <label class="label stack" style="min-width: 180px;">
                                                    <span>"Number of units to add"</span>
                                                    <input
                                                        class="input compact"
                                                        type="number"
                                                        min="1"
                                                        max="1024"
                                                        step="1"
                                                        prop:value=move || grow_units_n.get().to_string()
                                                        on:input=move |ev| {
                                                            if let Ok(v) = event_target_value(&ev).parse::<u32>() {
                                                                set_grow_units_n.set(v.clamp(1, 1024));
                                                            }
                                                        }
                                                        aria-label="Number of units to grow"
                                                    />
                                                </label>

                                                <div class="preset-row">
                                                    <span style="font-size: 0.85rem; color: var(--muted); font-weight: 700; margin-right: 4px;">"Quick presets:"</span>
                                                    <button class="btn sm" on:click=move |_| set_grow_units_n.set(16) title="Add 16 units (small growth)">"16"</button>
                                                    <button class="btn sm" on:click=move |_| set_grow_units_n.set(32) title="Add 32 units">"32"</button>
                                                    <button class="btn sm" on:click=move |_| set_grow_units_n.set(64) title="Add 64 units">"64"</button>
                                                    <button class="btn sm" on:click=move |_| set_grow_units_n.set(128) title="Add 128 units (moderate growth)">"128"</button>
                                                </div>

                                                <button
                                                    class="btn primary"
                                                    on:click=move |_| do_grow_units()
                                                    title="Add the specified number of units to the brain"
                                                >
                                                    "➕ Grow Substrate"
                                                </button>
                                            </div>

                                            <Show when=move || { grow_units_n.get() > 256 }>
                                                <div class="callout" style="margin-top: 12px; border-color: rgba(251, 191, 36, 0.40); background: rgba(251, 191, 36, 0.10);">
                                                    <p style="margin: 0; font-size: 0.88rem; display: flex; align-items: center; gap: 8px;">
                                                        <span style="font-size: 1.2rem;">"⚠"</span>
                                                        <span><strong>"Warning:"</strong>" Large growth (>256 units) may cause UI stuttering and significantly increase memory usage. Consider growing in smaller increments."</span>
                                                    </p>
                                                </div>
                                            </Show>

                                            <div class="callout" style="margin-top: 12px; background: rgba(122, 162, 255, 0.08); border-color: rgba(122, 162, 255, 0.25);">
                                                <p style="margin: 0; font-size: 0.85rem; line-height: 1.5;">
                                                    <strong>"What neurogenesis does:"</strong>
                                                    " Adds new oscillator units to the substrate. These units are initially unconnected and will form associations through local learning dynamics. "
                                                    "Use this when the brain runs out of capacity for new patterns or concepts."
                                                </p>
                                            </div>
                                        </div>

                                        <div class="card">
                                            <h3 class="card-title">"💾 Persistence"</h3>
                                            <p class="subtle">"Save/load your brain locally (IndexedDB) or import/export .bbi."</p>
                                            <div class="stack tight" style="margin-top: 10px;">
                                                <button class="btn" on:click=move |_| do_save()>"💾 Save (IndexedDB)"</button>
                                                <button class="btn" on:click=move |_| do_load()>"📂 Load (IndexedDB)"</button>
                                                <button class="btn" on:click=move |_| do_migrate_idb_format()>"🔁 Migrate stored format"</button>
                                                <button class="btn" on:click=move |_| do_export_bbi()>"📥 Export .bbi"</button>
                                                <button class="btn" on:click=move |_| do_import_bbi_click()>"📤 Import .bbi"</button>
                                            </div>

                                            <div class="row end wrap" style="justify-content: space-between; gap: 12px; margin-top: 10px;">
                                                <label class="checkbox-row" style="margin: 0;">
                                                    <input
                                                        type="checkbox"
                                                        prop:checked=move || import_autosave.get()
                                                        on:change=move |ev| {
                                                            let v = event_target_checked(&ev);
                                                            set_import_autosave.set(v);
                                                        }
                                                    />
                                                    <span>"Auto-save imports"</span>
                                                </label>
                                                <label class="checkbox-row" style="margin: 0;">
                                                    <input
                                                        type="checkbox"
                                                        prop:checked=move || idb_autosave.get()
                                                        on:change=move |ev| {
                                                            let v = event_target_checked(&ev);
                                                            set_idb_autosave.set(v);
                                                        }
                                                    />
                                                    <span>"Auto-save brain (~5s when changed)"</span>
                                                </label>
                                            </div>

                                            <div class="subtle" style="margin-top: 8px;">
                                                {move || {
                                                    if idb_loaded.get() {
                                                        "Source: IndexedDB (brain_image)".to_string()
                                                    } else {
                                                        "Source: fresh (not loaded from IndexedDB yet)".to_string()
                                                    }
                                                }}
                                            </div>
                                            <div class="subtle">
                                                {move || {
                                                    let ts = idb_last_save.get();
                                                    if ts.is_empty() {
                                                        "Last save: —".to_string()
                                                    } else {
                                                        format!("Last save: {ts}")
                                                    }
                                                }}
                                            </div>

                                            <input
                                                node_ref=import_input_ref
                                                type="file"
                                                accept=".bbi,application/octet-stream"
                                                class="hidden"
                                                on:change=do_import_bbi_change
                                            />
                                        </div>

                                        <div class="card">
                                            <h3 class="card-title">"⚡ Execution"</h3>
                                            <p class="subtle">"Choose CPU or GPU tier when available (WebGPU)."</p>
                                            <div class="subtle" style="font-size: 0.80rem; margin-top: 8px;">{move || gpu_status.get()}</div>
                                            <div class="subtle" style="font-size: 0.80rem; margin-top: 4px;">
                                                {move || {
                                                    let sel = exec_tier_selected.get();
                                                    let eff = exec_tier_effective.get();
                                                    format!("Tier: selected={sel:?}  effective={eff:?}")
                                                }}
                                            </div>

                                            <div class="row end wrap" style="justify-content: flex-start; gap: 10px; margin-top: 12px;">
                                                <button
                                                    class=move || {
                                                        if exec_tier_selected.get() == ExecutionTier::Scalar {
                                                            "btn sm primary"
                                                        } else {
                                                            "btn sm"
                                                        }
                                                    }
                                                    on:click=move |_| {
                                                        runtime.update_value(|r| r.brain.set_execution_tier(ExecutionTier::Scalar));
                                                        local_storage_set_string(LOCALSTORAGE_EXEC_TIER_KEY, "scalar");
                                                        if webgpu_available {
                                                            set_gpu_status.set("WebGPU: detected (CPU selected)");
                                                        } else {
                                                            set_gpu_status.set("WebGPU: not available (CPU mode)");
                                                        }
                                                        push_toast(ToastLevel::Info, "Execution tier: CPU".to_string());
                                                    }
                                                >
                                                    "CPU"
                                                </button>

                                                <button
                                                    class=move || {
                                                        if exec_tier_selected.get() == ExecutionTier::Gpu {
                                                            "btn sm primary"
                                                        } else {
                                                            "btn sm"
                                                        }
                                                    }
                                                    on:click=move |_| {
                                                        if !webgpu_available {
                                                            push_toast(ToastLevel::Error, "WebGPU not available in this browser/context".to_string());
                                                            return;
                                                        }

                                                        #[cfg(not(feature = "gpu"))]
                                                        {
                                                            push_toast(
                                                                ToastLevel::Error,
                                                                "This build does not include the web `gpu` feature".to_string(),
                                                            );
                                                        }

                                                        #[cfg(feature = "gpu")]
                                                        {
                                                            local_storage_set_string(LOCALSTORAGE_EXEC_TIER_KEY, "gpu");
                                                            runtime.update_value(|r| {
                                                                r.brain.set_execution_tier(ExecutionTier::Gpu);
                                                            });

                                                            let runtime = runtime.clone();
                                                            spawn_local(async move {
                                                                set_gpu_status.set("WebGPU: initializing…");
                                                                match braine::gpu::init_gpu_context(65_536).await {
                                                                    Ok(()) => {
                                                                        runtime.update_value(|r| {
                                                                            r.brain.set_execution_tier(ExecutionTier::Gpu);
                                                                        });
                                                                        let eff = runtime.with_value(|r| r.brain.effective_execution_tier());
                                                                        if eff == ExecutionTier::Gpu {
                                                                            set_gpu_status.set("WebGPU: enabled (GPU dynamics tier)");
                                                                            push_toast(
                                                                                ToastLevel::Success,
                                                                                "Execution tier: GPU".to_string(),
                                                                            );
                                                                        } else {
                                                                            set_gpu_status.set("WebGPU: ready (CPU fallback)");
                                                                            push_toast(
                                                                                ToastLevel::Info,
                                                                                "WebGPU detected, but using CPU tier".to_string(),
                                                                            );
                                                                        }
                                                                    }
                                                                    Err(e) => {
                                                                        set_gpu_status
                                                                            .set("WebGPU: init failed (CPU fallback)");
                                                                        push_toast(
                                                                            ToastLevel::Error,
                                                                            format!("WebGPU init failed: {e}"),
                                                                        );
                                                                    }
                                                                }
                                                            });
                                                        }
                                                    }
                                                >
                                                    "GPU"
                                                </button>
                                            </div>
                                        </div>

                                        <For
                                            each=move || settings_schema::sections_ordered().into_iter()
                                            key=|s| s.title
                                            children=move |sec| {
                                                if matches!(sec.section, ParamSection::BraineSettings | ParamSection::Neurogenesis) {
                                                    return view! { <span style="display:none;"></span> }.into_any();
                                                }

                                                view! {
                                                    <div class="card">
                                                        <h3 class="card-title">{sec.title}</h3>
                                                        <p class="subtle">{sec.blurb}</p>
                                                        <div class="param-grid">
                                                            <For
                                                                each=move || {
                                                                    let show_adv = settings_advanced.get();
                                                                    settings_specs
                                                                        .get_value()
                                                                        .into_iter()
                                                                        .filter(|p| p.section == sec.section)
                                                                        .filter(move |p| show_adv || !p.advanced)
                                                                        .collect::<Vec<_>>()
                                                                }
                                                                key=|p| p.key
                                                                children=move |p| {
                                                                    let Some((v, set_v)) = bind(p.key) else {
                                                                        return view! { <span style="display:none;"></span> }.into_any();
                                                                    };
                                                                    view! {
                                                                        <ParameterField
                                                                            spec=p
                                                                            value=v
                                                                            set_value=set_v
                                                                            validity_map=settings_validity_map
                                                                        />
                                                                    }
                                                                    .into_any()
                                                                }
                                                            />
                                                        </div>
                                                    </div>
                                                }
                                                .into_any()
                                            }
                                        />

                                        <div class="settings-footer">
                                            <div class="row end wrap" style="justify-content: space-between;">
                                                <div class="settings-footer-left">
                                                    <Show when=move || settings_apply_disabled.get()>
                                                        <div class="param-warn">"Fix invalid values to apply."</div>
                                                    </Show>
                                                    <Show when=move || config_applied.get()>
                                                        <div class="apply-toast">"Applied ✓"</div>
                                                    </Show>
                                                </div>
                                                <div class="row end wrap" style="gap: 10px;">
                                                    <button class="btn" on:click=move |_| reset_brain_config_from_runtime()>
                                                        "Reset"
                                                    </button>
                                                    <button
                                                        class="btn primary"
                                                        disabled=move || settings_apply_disabled.get()
                                                        on:click=move |_| apply_brain_config()
                                                    >
                                                        "Apply"
                                                    </button>
                                                </div>
                                            </div>
                                        </div>
                                    </div>
                                }
                            }
                        </Show>

                        <Show when=move || dashboard_tab.get() == DashboardTab::Learning>
                            <div class="stack tight learning-stack">
                                <div class="card">
                                    <h3 class="card-title">"🧪 Learning"</h3>
                                    <p class="subtle">"Controls for learning writes, accelerated learning, and simulation cadence."</p>
                                </div>

                                <div class="card">
                                    <h3 class="card-title">"📊 Learning Milestones"</h3>

                                    <div class=move || format!("callout tone-{}", learning_milestone_tone.get())>
                                        <p style="margin: 0;"><strong>{move || learning_milestone.get()}</strong></p>
                                        <p style="margin: 6px 0 0 0;">
                                            {move || {
                                                if let Some(t) = mastered_at_trial.get() {
                                                    format!("🏆 Mastered at trial {}", t)
                                                } else if let Some(t) = learned_at_trial.get() {
                                                    format!("✓ Learned at trial {}", t)
                                                } else {
                                                    "Keep training...".to_string()
                                                }
                                            }}
                                        </p>
                                    </div>

                                    <div style="display: grid; grid-template-columns: auto 1fr; gap: 6px 12px; color: var(--text); font-size: 0.85rem; margin-top: 10px;">
                                        <span style="color: var(--muted);">"⏳ Starting"</span>
                                        <span>"First 20 trials — brain is exploring and imprinting"</span>
                                        <span style="color: var(--muted);">"🔄 Training"</span>
                                        <span>"Accuracy < 70% — brain is building associations"</span>
                                        <span style="color: var(--muted);">"📈 Learning"</span>
                                        <span>"Accuracy ≥ 70% — causal links forming, above chance"</span>
                                        <span style="color: #4ade80;">"✓ Learned"</span>
                                        <span>"Accuracy ≥ 85% — reliable performance achieved"</span>
                                        <span style="color: #fbbf24;">"🏆 Mastered"</span>
                                        <span>"Accuracy ≥ 95% — near-optimal behavior"</span>
                                    </div>
                                </div>

                                <div class="card">
                                    <h3 class="card-title">"🧪 Learning Writes"</h3>
                                    <div class="row end wrap">
                                        <label class="label" style="flex-direction: row; align-items: center; gap: 10px;">
                                            <input
                                                type="checkbox"
                                                prop:checked=move || learning_enabled.get()
                                                on:change=move |ev| {
                                                    let v = event_target_checked(&ev);
                                                    set_learning_enabled.set(v);
                                                }
                                            />
                                            <span>"Enable learning (reinforce + commit_observation)"</span>
                                        </label>
                                    </div>
                                    <p class="subtle">"Disable this to run inference-only without updating causal/meaning memory."</p>
                                </div>

                                <div class="card">
                                    <h3 class="card-title">"⚡ Accelerated Learning"</h3>
                                    <div class="row end wrap">
                                        <button class="btn" on:click=move |_| do_dream()>"💭 Dream"</button>
                                        <button class="btn" on:click=move |_| do_burst()>"🔥 Burst"</button>
                                        <button class="btn" on:click=move |_| do_sync()>"🔄 Sync"</button>
                                        <button class="btn" on:click=move |_| do_imprint()>"💡 Imprint"</button>
                                    </div>
                                    <div class="callout" style="margin-top: 10px;">
                                        <p style="margin: 2px 0;"><strong>"Dream"</strong>": Offline replay to consolidate recent structure (stabilizes what just worked)."</p>
                                        <p style="margin: 2px 0;"><strong>"Burst"</strong>": Temporary learning-rate boost (helps rapid adaptation; can increase drift if overused)."</p>
                                        <p style="margin: 2px 0;"><strong>"Sync"</strong>": Aligns sensor phases for cleaner encoding (use after a regime shift / reversal)."</p>
                                        <p style="margin: 2px 0;"><strong>"Imprint"</strong>": One-shot linking of the current context (fast association when the brain is missing a concept)."</p>
                                    </div>
                                </div>

                                <div class="card">
                                    <h3 class="card-title">"⏱ Simulation Speed"</h3>
                                    <div class="row end wrap">
                                        <label class="label">
                                            <span>"Run interval (ms)"</span>
                                            <input
                                                class="input compact"
                                                type="number"
                                                min="8"
                                                max="500"
                                                step="1"
                                                prop:value=move || run_interval_ms.get().to_string()
                                                on:input=move |ev| {
                                                    if let Ok(v) = event_target_value(&ev).parse::<u32>() {
                                                        let v = v.clamp(8, 500);
                                                        set_run_interval_ms.set(v);
                                                        if is_running.get_untracked() {
                                                            let do_stop = do_stop_sv.get_value();
                                                            let do_start = do_start_sv.get_value();
                                                            do_stop();
                                                            do_start();
                                                        }
                                                    }
                                                }
                                            />
                                        </label>
                                        <button class="btn" on:click=move |_| {
                                            set_run_interval_ms.set(33);
                                            if is_running.get_untracked() {
                                                let do_stop = do_stop_sv.get_value();
                                                let do_start = do_start_sv.get_value();
                                                do_stop();
                                                do_start();
                                            }
                                        }>
                                            "Reset"
                                        </button>
                                    </div>
                                    <p class="subtle">"If changed while running, the interval restarts."</p>
                                </div>

                                <div class="card">
                                    <h3 class="card-title">"⚡ Reward Shaping"</h3>
                                    <div class="row end wrap">
                                        <label class="label">
                                            <span>"Scale"</span>
                                            <input
                                                class="input compact"
                                                type="number"
                                                step="0.1"
                                                prop:value=move || fmt_f32_fixed(reward_scale.get(), 2)
                                                on:input=move |ev| {
                                                    if let Ok(v) = event_target_value(&ev).parse::<f32>() {
                                                        set_reward_scale.set(v.clamp(0.0, 10.0));
                                                    }
                                                }
                                            />
                                        </label>
                                        <label class="label">
                                            <span>"Bias"</span>
                                            <input
                                                class="input compact"
                                                type="number"
                                                step="0.1"
                                                prop:value=move || fmt_f32_fixed(reward_bias.get(), 2)
                                                on:input=move |ev| {
                                                    if let Ok(v) = event_target_value(&ev).parse::<f32>() {
                                                        set_reward_bias.set(v.clamp(-2.0, 2.0));
                                                    }
                                                }
                                            />
                                        </label>
                                        <button class="btn" on:click=move |_| {
                                            set_reward_scale.set(1.0);
                                            set_reward_bias.set(0.0);
                                        }>
                                            "Reset"
                                        </button>
                                    </div>
                                    <div class="callout">
                                        <p>
                                            <strong>"Formula:"</strong>
                                            " shaped = clamp((raw + bias) × scale, −5..5)"
                                        </p>
                                        <p class="subtle">"Game scoring/stats still use the raw reward sign."</p>
                                    </div>
                                </div>
                            </div>
                        </Show>

                    </div>

                </div>
            </div>
        </div>
    }
}
