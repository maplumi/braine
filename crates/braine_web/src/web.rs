#![allow(clippy::clone_on_copy)]

use braine::substrate::{Brain, BrainConfig, CausalGraphViz, Stimulus, UnitPlotPoint};
use braine_games::{
    bandit::BanditGame,
    replay::{ReplayDataset, ReplayGame},
    spot::SpotGame,
    spot_reversal::SpotReversalGame,
    spot_xy::SpotXYGame,
};
use leptos::prelude::*;
use leptos::task::spawn_local;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
mod charts;
mod pong_web;
use pong_web::PongWebGame;
mod sequence_web;
use sequence_web::SequenceWebGame;
mod text_web;
use charts::RollingHistory;
use text_web::TextWebGame;
use wasm_bindgen::closure::Closure;
use wasm_bindgen::prelude::*;
use wasm_bindgen::JsCast;

const IDB_DB_NAME: &str = "braine";
const IDB_STORE: &str = "kv";
const IDB_KEY_BRAIN_IMAGE: &str = "brain_image";
const IDB_KEY_GAME_ACCURACY: &str = "game_accuracy";

const LOCALSTORAGE_THEME_KEY: &str = "braine_theme";
const LOCALSTORAGE_GAME_STATS_PREFIX: &str = "braine_game_stats_v1.";
const LOCALSTORAGE_SETTINGS_KEY: &str = "braine_settings_v1";

const STYLE_CARD: &str = "padding: 14px; background: var(--panel); border: 1px solid var(--border); border-radius: 12px;";

// Version strings for display
const VERSION_BRAINE: &str = env!("CARGO_PKG_VERSION");
const VERSION_BRAINE_WEB: &str = "0.1.0";
const VERSION_BBI_FORMAT: u32 = 2;

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
            AboutSubTab::Architecture,
            AboutSubTab::Applications,
            AboutSubTab::LlmIntegration,
            AboutSubTab::Apis,
        ]
    }
}

const ABOUT_LLM_DATAFLOW: &str = r#"Braine (daemon) owns the long-lived Brain.

  (game loop)
  stimuli ──▶ Brain dynamics ──▶ action ──▶ env/reward ──▶ neuromodulator ──▶ local learning

  (advisor loop, slow + bounded)
  Brain+HUD snapshot ──▶ AdvisorContext (Braine → LLM)
                        └──▶ external LLM
                               └──▶ AdvisorAdvice (LLM → Braine)
                                        └──▶ daemon clamps + applies bounded knobs
                                                (exploration_eps, meaning_alpha, ttl)
"#;

const ABOUT_LLM_ADVISOR_CONTEXT_REQ: &str =
    r#"{\n  \"type\": \"AdvisorContext\",\n  \"include_action_scores\": true\n}"#;

const ABOUT_LLM_ADVISOR_CONTEXT_RESP: &str = r#"{\n  \"type\": \"AdvisorContext\",\n  \"context\": {\n    \"context_key\": \"replay::spot_lr_small\",\n    \"game\": \"replay\",\n    \"trials\": 200,\n    \"recent_rate\": 0.52,\n    \"last_100_rate\": 0.51,\n    \"exploration_eps\": 0.20,\n    \"meaning_alpha\": 0.60,\n    \"notes\": [\"bounded; no action selection\"]\n  },\n  \"action_scores\": [\n    { \"name\": \"left\",  \"score\": 0.12, \"meaning\": 0.03, \"habit_norm\": 0.09 },\n    { \"name\": \"right\", \"score\": 0.10, \"meaning\": 0.02, \"habit_norm\": 0.08 }\n  ]\n}"#;

const ABOUT_LLM_ADVISOR_APPLY_REQ: &str = r#"{\n  \"type\": \"AdvisorApply\",\n  \"advice\": {\n    \"exploration_eps\": 0.12,\n    \"meaning_alpha\": 0.75,\n    \"ttl_trials\": 200,\n    \"rationale\": \"Increase meaning weight to sharpen context-conditioned discrimination; reduce exploration once stability is improving.\"\n  }\n}"#;

pub fn start() {
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

    let (theme, set_theme) = signal(Theme::Dark);

    let (steps, set_steps) = signal(0u64);
    let (diag, set_diag) = signal(runtime.with_value(|r| r.brain.diagnostics()));

    let (game_kind, set_game_kind) = signal(GameKind::Spot);
    let (dashboard_tab, set_dashboard_tab) = signal(DashboardTab::Learning);
    // Mobile responsive: dashboard drawer (hidden by default on small screens)
    let (dashboard_open, set_dashboard_open) = signal(false);
    let (analytics_panel, set_analytics_panel) = signal(AnalyticsPanel::Performance);
    let (trial_period_ms, set_trial_period_ms) = signal(500u32);
    let (run_interval_ms, set_run_interval_ms) = signal(33u32);
    let (exploration_eps, set_exploration_eps) = signal(0.08f32);
    let (meaning_alpha, set_meaning_alpha) = signal(6.0f32);
    let (reward_scale, set_reward_scale) = signal(1.0f32);
    let (reward_bias, set_reward_bias) = signal(0.0f32);
    let (learning_enabled, set_learning_enabled) = signal(true);
    let (grow_units_n, set_grow_units_n) = signal(128u32);

    // BrainConfig tuning (live). These mirror `BrainConfig` fields that are safe to
    // update on a running brain. Topology fields are shown read-only.
    let (cfg_dt, set_cfg_dt) = signal(runtime.with_value(|r| r.brain.config().dt));
    let (cfg_base_freq, set_cfg_base_freq) =
        signal(runtime.with_value(|r| r.brain.config().base_freq));
    let (cfg_noise_amp, set_cfg_noise_amp) =
        signal(runtime.with_value(|r| r.brain.config().noise_amp));
    let (cfg_noise_phase, set_cfg_noise_phase) =
        signal(runtime.with_value(|r| r.brain.config().noise_phase));
    let (cfg_global_inhibition, set_cfg_global_inhibition) =
        signal(runtime.with_value(|r| r.brain.config().global_inhibition));
    let (cfg_hebb_rate, set_cfg_hebb_rate) =
        signal(runtime.with_value(|r| r.brain.config().hebb_rate));
    let (cfg_forget_rate, set_cfg_forget_rate) =
        signal(runtime.with_value(|r| r.brain.config().forget_rate));
    let (cfg_prune_below, set_cfg_prune_below) =
        signal(runtime.with_value(|r| r.brain.config().prune_below));
    let (cfg_coactive_threshold, set_cfg_coactive_threshold) =
        signal(runtime.with_value(|r| r.brain.config().coactive_threshold));
    let (cfg_phase_lock_threshold, set_cfg_phase_lock_threshold) =
        signal(runtime.with_value(|r| r.brain.config().phase_lock_threshold));
    let (cfg_imprint_rate, set_cfg_imprint_rate) =
        signal(runtime.with_value(|r| r.brain.config().imprint_rate));
    let (cfg_salience_decay, set_cfg_salience_decay) =
        signal(runtime.with_value(|r| r.brain.config().salience_decay));
    let (cfg_salience_gain, set_cfg_salience_gain) =
        signal(runtime.with_value(|r| r.brain.config().salience_gain));
    let (cfg_causal_decay, set_cfg_causal_decay) =
        signal(runtime.with_value(|r| r.brain.config().causal_decay));

    let (last_action, set_last_action) = signal(String::new());
    let (last_reward, set_last_reward) = signal(0.0f32);

    let (trials, set_trials) = signal(0u32);
    let (recent_rate, set_recent_rate) = signal(0.5f32);
    let (status, set_status) = signal(String::new());

    let (spotxy_pos, set_spotxy_pos) = signal::<Option<(f32, f32)>>(None);
    let (_spotxy_stimulus_key, set_spotxy_stimulus_key) = signal(String::new());
    let (spotxy_eval, set_spotxy_eval) = signal(false);
    let (spotxy_mode, set_spotxy_mode) = signal(String::new());
    let (spotxy_grid_n, set_spotxy_grid_n) = signal(0u32);

    let (pong_state, set_pong_state) = signal::<Option<PongUiState>>(None);
    let (_pong_stimulus_key, set_pong_stimulus_key) = signal(String::new());
    let (pong_paddle_speed, set_pong_paddle_speed) = signal(0.0f32);
    let (pong_paddle_half_height, set_pong_paddle_half_height) = signal(0.0f32);
    let (pong_ball_speed, set_pong_ball_speed) = signal(0.0f32);

    let (sequence_state, set_sequence_state) = signal::<Option<SequenceUiState>>(None);
    let (text_state, set_text_state) = signal::<Option<TextUiState>>(None);
    let (replay_state, set_replay_state) = signal::<Option<ReplayUiState>>(None);

    // Spot/SpotReversal stimulus (left/right) for UI highlighting.
    let (spot_is_left, set_spot_is_left) = signal::<Option<bool>>(None);

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
            } else {
                set_status.set("Braine config applied".to_string());
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
        }
    };

    // Text prediction "lab" (inference-only on a cloned brain)
    let (text_prompt, set_text_prompt) = signal(String::from("hello worl"));
    let (text_prompt_regime, set_text_prompt_regime) = signal(0u32);
    let (text_temp, set_text_temp) = signal(1.0f32);
    let (text_preds, set_text_preds) = signal::<Vec<(String, f32, f32)>>(Vec::new());

    // Text training controls (writes to the live brain)
    let (text_corpus0, set_text_corpus0) = signal(String::from("hello world\n"));
    let (text_corpus1, set_text_corpus1) = signal(String::from("goodbye world\n"));
    let (text_max_vocab, set_text_max_vocab) = signal(32u32);
    let (text_shift_every, set_text_shift_every) = signal(80u32);

    let (text_train_prompt, set_text_train_prompt) = signal(String::from("hello world\n"));
    let (text_train_regime, set_text_train_regime) = signal(0u32);
    let (text_train_epochs, set_text_train_epochs) = signal(1u32);

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
    let (perf_history_version, set_perf_history_version) = signal(0u32);

    // Neuromodulator (reward) history
    let neuromod_history = StoredValue::new(RollingHistory::new(50));
    let (neuromod_version, set_neuromod_version) = signal(0u32);

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

    // BrainViz uses its own sampling so it can be tuned independently.
    let (brainviz_points, set_brainviz_points) = signal::<Vec<UnitPlotPoint>>(Vec::new());
    let (brainviz_node_sample, set_brainviz_node_sample) = signal(128u32);
    let (brainviz_edges_per_node, set_brainviz_edges_per_node) = signal(4u32);
    let (brainviz_zoom, set_brainviz_zoom) = signal(1.5f32);
    let (brainviz_pan_x, set_brainviz_pan_x) = signal(0.0f32);
    let (brainviz_pan_y, set_brainviz_pan_y) = signal(0.0f32);
    let (_brainviz_auto_rotate, _set_brainviz_auto_rotate) = signal(false); // Disabled by default
    let (brainviz_manual_rotation, set_brainviz_manual_rotation) = signal(0.0f32); // Y-axis rotation (horizontal drag)
    let (brainviz_rotation_x, set_brainviz_rotation_x) = signal(0.0f32); // X-axis rotation (vertical drag)
    let (brainviz_vibration, _set_brainviz_vibration) = signal(0.0f32); // Activity-based vibration
    let (brainviz_idle_time, set_brainviz_idle_time) = signal(0.0f32); // Idle animation time (dreaming mode)
    let (brainviz_hover, set_brainviz_hover) = signal::<Option<(u32, f64, f64)>>(None);
    let (brainviz_view_mode, set_brainviz_view_mode) = signal::<&'static str>("substrate"); // "substrate" or "causal"
    let (brainviz_causal_graph, set_brainviz_causal_graph) =
        signal::<CausalGraphViz>(CausalGraphViz::default());
    let brainviz_dragging = StoredValue::new(false);
    let brainviz_last_drag_xy = StoredValue::new((0.0f64, 0.0f64));
    let brainviz_hit_nodes = StoredValue::new(Vec::<charts::BrainVizHitNode>::new());

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

    #[cfg(feature = "gpu")]
    {
        if webgpu_available {
            runtime.update_value(|r| {
                r.brain
                    .set_execution_tier(braine::substrate::ExecutionTier::Gpu)
            });
        }
    }

    let (gpu_status, _set_gpu_status) = signal(if webgpu_available {
        if cfg!(feature = "gpu") {
            "WebGPU: detected (GPU dynamics enabled; learning is CPU)"
        } else {
            "WebGPU: detected (CPU build; enable the web `gpu` feature)"
        }
    } else {
        "WebGPU: not available (CPU mode)"
    });

    let refresh_ui_from_runtime = {
        let runtime = runtime.clone();
        move || {
            set_diag.set(runtime.with_value(|r| r.brain.diagnostics()));
            let stats = runtime.with_value(|r| r.game.stats().clone());
            set_trials.set(stats.trials);
            let rate = stats.last_100_rate();
            set_recent_rate.set(rate);
            set_correct_count.set(stats.correct);
            set_incorrect_count.set(stats.incorrect);
            set_learned_at_trial.set(stats.learned_at_trial);
            set_mastered_at_trial.set(stats.mastered_at_trial);

            // Update unit plot data for Graph page (sample 128 units)
            let plot_points = runtime.with_value(|r| r.brain.unit_plot_points(128));
            set_unit_plot.set(plot_points);

            // Update brain age and causal stats
            set_brain_age.set(runtime.with_value(|r| r.brain.age_steps()));
            let cstats = runtime.with_value(|r| r.brain.causal_stats());
            set_causal_symbols.set(cstats.base_symbols);
            set_causal_edges.set(cstats.edges);

            // Update game accuracy in memory (will be persisted on game switch or save)
            let game_label = game_kind.get_untracked().label().to_string();
            set_game_accuracies.update(|accs| {
                accs.insert(game_label, rate);
            });

            // Update performance history
            perf_history.update_value(|h| h.push(rate));
            set_perf_history_version.update(|v| *v = v.wrapping_add(1));

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

            let snap = runtime.with_value(|r| r.game_ui_snapshot());
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
            set_sequence_state.set(snap.sequence_state);
            set_text_state.set(snap.text_state);
            set_replay_state.set(snap.replay_state);
            set_spot_is_left.set(snap.spot_is_left);

            // Persist per-game stats + chart history so refresh restores the current state.
            let kind = game_kind.get_untracked();
            let should_persist =
                persisted_trial_cursor.with_value(|(k, t)| *k != kind || *t != stats.trials);
            if should_persist {
                persisted_trial_cursor.update_value(|(k, t)| {
                    *k = kind;
                    *t = stats.trials;
                });
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
                set_perf_history_version.update(|v| *v = v.wrapping_add(1));
                neuromod_history.update_value(|h| h.set_data(state.neuromod_history.clone()));
                set_neuromod_version.update(|v| *v = v.wrapping_add(1));
                choice_events.update_value(|v| *v = state.choice_events.clone());
                set_choices_version.update(|v| *v = v.wrapping_add(1));
                set_last_action.set(state.last_action);
                set_last_reward.set(state.last_reward);
            } else {
                runtime.update_value(|r| r.game.set_stats(braine_games::stats::GameStats::new()));
                perf_history.update_value(|h| h.clear());
                set_perf_history_version.update(|v| *v = v.wrapping_add(1));
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

            let mut out: Option<TickOutput> = None;
            runtime.update_value(|r| {
                out = r.tick(&cfg);
            });
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
            }

            set_steps.update(|s| *s += 1);
            refresh_ui_from_runtime();
        }
    };

    let do_reset = move || {
        let kind = game_kind.get_untracked();
        clear_persisted_stats_state(kind);

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
                    "text: trained {} pairs (acc {:.1}%, avg r {:.2})",
                    trained_pairs,
                    acc * 100.0,
                    avg_r
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

    let pong_canvas_ref = NodeRef::<leptos::html::Canvas>::new();
    Effect::new(move |_| {
        let state = pong_state.get();
        let Some(canvas) = pong_canvas_ref.get() else {
            return;
        };

        match state {
            Some(s) => {
                let _ = draw_pong(&canvas, &s);
            }
            None => {
                let _ = clear_canvas(&canvas);
            }
        }
    });

    // Performance chart canvas
    let perf_chart_ref = NodeRef::<leptos::html::Canvas>::new();
    Effect::new(move |_| {
        let _ = perf_history_version.get(); // Subscribe to updates
        let Some(canvas) = perf_chart_ref.get() else {
            return;
        };
        let data: Vec<f32> = perf_history.with_value(|h| h.data().to_vec());
        let _ = charts::draw_line_chart(&canvas, &data, 0.0, 1.0, "#7aa2ff", "#0a0f1a", "#1a2540");
    });

    // Neuromod (reward) chart canvas
    let neuromod_chart_ref = NodeRef::<leptos::html::Canvas>::new();
    Effect::new(move |_| {
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
            if analytics_panel.get() != AnalyticsPanel::Choices {
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

    // Accuracy gauge canvas
    let gauge_ref = NodeRef::<leptos::html::Canvas>::new();
    Effect::new(move |_| {
        let rate = recent_rate.get();
        let Some(canvas) = gauge_ref.get() else {
            return;
        };
        let color = if rate >= 0.85 {
            "#4ade80"
        } else if rate >= 0.70 {
            "#fbbf24"
        } else {
            "#7aa2ff"
        };
        let _ = charts::draw_gauge(&canvas, rate, 0.0, 1.0, "Accuracy", color, "#0a0f1a");
    });

    // Unit plot 3D-style canvas for Graph page
    let unit_plot_ref = NodeRef::<leptos::html::Canvas>::new();
    Effect::new(move |_| {
        let points = unit_plot.get();
        let Some(canvas) = unit_plot_ref.get() else {
            return;
        };
        let _ = charts::draw_unit_plot_3d(&canvas, &points, "#0a0f1a");
    });

    // BrainViz: sample plot points based on UI settings (does not affect learning/perf history).
    // NOTE: edges are re-derived every render; we must refresh nodes periodically too,
    // otherwise node sizes/colors can look "stuck" while edges change.
    Effect::new({
        let runtime = runtime.clone();
        move |_| {
            if analytics_panel.get() != AnalyticsPanel::BrainViz {
                return;
            }

            let view_mode = brainviz_view_mode.get();
            let n = brainviz_node_sample.get().clamp(16, 1024) as usize;

            // Throttle refresh:
            // - while running: refresh every ~2 steps
            // - while idle: refresh every ~2 seconds (idle_time is ~30fps)
            let step = steps.get();
            let running = is_running.get();
            let idle_time = brainviz_idle_time.get();
            if running {
                if step % 2 != 0 {
                    return;
                }
            } else {
                let bucket = (idle_time * 10.0) as u32; // ~10Hz buckets
                #[allow(clippy::manual_is_multiple_of)]
                if bucket % 20 != 0 {
                    return;
                }
            }

            if view_mode == "causal" {
                // Fetch causal graph data
                // (heavier than substrate points; keep it on the same throttle)
                let causal = runtime.with_value(|r| r.brain.causal_graph_viz(n, n * 2));
                set_brainviz_causal_graph.set(causal);
            } else {
                // Fetch substrate unit points
                let pts = runtime.with_value(|r| r.brain.unit_plot_points(n));
                // Global vibration removed - nodes now have local per-node animation
                set_brainviz_points.set(pts);
            }
        }
    });

    // BrainViz: rotating sphere + sampled connectivity (or causal graph)
    let brain_viz_ref = NodeRef::<leptos::html::Canvas>::new();
    let brainviz_causal_hit_nodes = StoredValue::new(Vec::<charts::CausalHitNode>::new());
    Effect::new({
        let runtime = runtime.clone();
        move |_| {
            if analytics_panel.get() != AnalyticsPanel::BrainViz {
                return;
            }

            let step = steps.get(); // Track for reactivity
            let running = is_running.get();
            let idle_time = brainviz_idle_time.get(); // Idle animation time
            let view_mode = brainviz_view_mode.get();
            let Some(canvas) = brain_viz_ref.get() else {
                return;
            };

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
                let opts = charts::CausalVizRenderOptions {
                    zoom,
                    pan_x,
                    pan_y,
                    rotation: rot_y,
                    rotation_x: rot_x,
                    draw_outline: true,
                    anim_time,
                };
                if let Ok(hits) = charts::draw_causal_graph(
                    &canvas,
                    &causal.nodes,
                    &causal.edges,
                    "#0a0f1a",
                    opts,
                ) {
                    brainviz_causal_hit_nodes.set_value(hits);
                }
            } else {
                // Render substrate view
                let points = brainviz_points.get();
                let edges_per_node = brainviz_edges_per_node.get().clamp(1, 32) as usize;

                let mut sampled: std::collections::HashSet<usize> =
                    std::collections::HashSet::new();
                sampled.extend(points.iter().map(|p| p.id as usize));

                let edges: Vec<(u32, u32, f32)> = runtime.with_value(|r| {
                    let mut edges: Vec<(u32, u32, f32)> = Vec::new();
                    for p in &points {
                        let src = p.id as usize;
                        let mut candidates: Vec<(usize, f32)> = r
                            .brain
                            .neighbors(src)
                            .filter(|(t, _w)| sampled.contains(t))
                            .collect();
                        candidates.sort_by(|a, b| b.1.abs().total_cmp(&a.1.abs()));
                        for (t, w) in candidates.into_iter().take(edges_per_node) {
                            edges.push((p.id, t as u32, w));
                        }
                    }
                    edges
                });

                let is_learning = learning_enabled.get();
                let opts = charts::BrainVizRenderOptions {
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
                if let Ok(hits) = charts::draw_brain_connectivity_sphere(
                    &canvas, &points, &edges, rot_y, "#0a0f1a", opts,
                ) {
                    brainviz_hit_nodes.set_value(hits);
                }
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

    // Restore per-game stats (counters + charts) from localStorage once at startup.
    let did_restore_stats = StoredValue::new(false);
    Effect::new(move |_| {
        if did_restore_stats.get_value() {
            return;
        }
        did_restore_stats.set_value(true);

        let kind = game_kind.get_untracked();
        (restore_stats_state)(kind);
        refresh_ui_from_runtime();
    });

    // Settings: load/save reward shaping controls.
    let did_load_settings = StoredValue::new(false);
    Effect::new(move |_| {
        if did_load_settings.get_value() {
            return;
        }
        did_load_settings.set_value(true);

        if let Some(s) = load_persisted_settings() {
            set_reward_scale.set(s.reward_scale);
            set_reward_bias.set(s.reward_bias);
            set_learning_enabled.set(s.learning_enabled);
            set_run_interval_ms.set(s.run_interval_ms.clamp(8, 500));
            set_trial_period_ms.set(s.trial_period_ms.clamp(10, 60_000));
        }
    });
    Effect::new(move |_| {
        let s = PersistedSettings {
            reward_scale: reward_scale.get(),
            reward_bias: reward_bias.get(),
            learning_enabled: learning_enabled.get(),
            run_interval_ms: run_interval_ms.get(),
            trial_period_ms: trial_period_ms.get(),
        };
        save_persisted_settings(&s);
    });

    // Theme: load from localStorage and apply to <html data-theme=...>
    let did_load_theme = StoredValue::new(false);
    Effect::new(move |_| {
        if did_load_theme.get_value() {
            return;
        }
        did_load_theme.set_value(true);

        if let Some(v) = local_storage_get_string(LOCALSTORAGE_THEME_KEY) {
            if let Some(t) = Theme::from_attr(&v) {
                set_theme.set(t);
            }
        }
    });
    Effect::new(move |_| {
        let t = theme.get();
        apply_theme_to_document(t);
        local_storage_set_string(LOCALSTORAGE_THEME_KEY, t.as_attr());
    });

    view! {
        <div class="app">
            // Header
            <header class="app-header">
                <div class="app-header-left">
                    <h1 class="brand">"🧠 Braine"</h1>
                    <span class="subtle">{move || gpu_status.get()}</span>
                </div>
                <div class="app-header-right">
                    <span class="status">{move || status.get()}</span>
                    <Show when=move || is_running.get()>
                        <span class="live-dot"></span>
                    </Show>
                    <button
                        class="btn sm ghost"
                        title=move || format!("Theme: {}", theme.get().label())
                        on:click=move |_| set_theme.set(theme.get().toggle())
                    >
                        {move || theme.get().icon()}" "{move || theme.get().label()}
                    </button>
                </div>
            </header>

            // Game tabs navigation (with About as first tab on left)
            <nav class="app-nav">
                <button
                    class=move || if show_about_page.get() { "tab active" } else { "tab" }
                    on:click=move |_| set_show_about_page.set(true)
                >
                    "ℹ️ About"
                </button>
                {GameKind::all().iter().map(|&kind| {
                    let set_game = Arc::clone(&set_game);
                    view! {
                        <button
                            class=move || if !show_about_page.get() && game_kind.get() == kind { "tab active" } else { "tab" }
                            on:click=move |_| {
                                set_show_about_page.set(false);
                                set_game(kind);
                            }
                        >
                            {kind.display_name()}
                        </button>
                    }
                }).collect_view()}
            </nav>

            // Main content area
            <div class="content-split">
                // Game area (left) - shows About page or game controls/canvas
                <div class="game-area">
                    <Show when=move || show_about_page.get()>
                        // Full-width About page with sub-tabs
                        <div class="about">
                            // Header
                            <div class="about-header">
                                <div style="font-size: 2.5rem;">"🧠"</div>
                                <div>
                                    <h1 style="margin: 0; font-size: 1.4rem; font-weight: 700; color: var(--accent);">"Braine"</h1>
                                    <p style="margin: 2px 0 0 0; color: var(--muted); font-size: 0.85rem;">"Closed-loop learning substrate • v"{VERSION_BRAINE}</p>
                                </div>
                                <div style="flex: 1;"></div>
                                <button class="btn primary" style="padding: 8px 16px;" on:click=move |_| set_show_about_page.set(false)>"🎮 Start Playing"</button>
                            </div>

                            // Sub-tab navigation
                            <div style="display: flex; gap: 4px; margin-bottom: 16px; flex-wrap: wrap;">
                                {AboutSubTab::all().iter().map(|&tab| {
                                    view! {
                                        <button
                                            style=move || if about_sub_tab.get() == tab {
                                                "padding: 8px 14px; background: var(--accent); color: #fff; border: none; border-radius: 6px; cursor: pointer; font-size: 0.85rem; font-weight: 600;"
                                            } else {
                                                "padding: 8px 14px; background: var(--panel); color: var(--text); border: 1px solid var(--border); border-radius: 6px; cursor: pointer; font-size: 0.85rem;"
                                            }
                                            on:click=move |_| set_about_sub_tab.set(tab)
                                        >
                                            {tab.label()}
                                        </button>
                                    }
                                }).collect_view()}
                            </div>

                            // Sub-tab content
                            <div style="display: flex; flex-direction: column; gap: 16px;">
                                // Overview tab
                                <Show when=move || about_sub_tab.get() == AboutSubTab::Overview>
                                    <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 16px;">
                                        <div style=STYLE_CARD>
                                            <h3 style="margin: 0 0 10px 0; font-size: 1rem; color: var(--accent);">"What is Braine?"</h3>
                                            <p style="margin: 0 0 10px 0; color: var(--text); font-size: 0.9rem; line-height: 1.7;">
                                                "Braine is a "<strong>"closed-loop learning substrate"</strong>" — a continuously-running dynamical system with local plasticity and scalar reward. Unlike LLMs, it:"
                                            </p>
                                            <ul style="margin: 0; padding-left: 20px; color: var(--text); font-size: 0.9rem; line-height: 1.7;">
                                                <li>"Uses no backpropagation or gradients"</li>
                                                <li>"Learns online from experience"</li>
                                                <li>"Has bounded memory (forgetting + pruning)"</li>
                                                <li>"Runs on edge devices"</li>
                                            </ul>

                                            <div style="margin-top: 12px; display: flex; gap: 10px; align-items: center; flex-wrap: wrap;">
                                                <span style="color: var(--muted); font-size: 0.85rem;">"Looking for the LLM boundary docs?"</span>
                                                <button class="btn sm" on:click=move |_| set_about_sub_tab.set(AboutSubTab::LlmIntegration)>
                                                    "LLM Integration →"
                                                </button>
                                            </div>
                                        </div>

                                        <div style=STYLE_CARD>
                                            <h3 style="margin: 0 0 10px 0; font-size: 1rem; color: var(--accent);">"Version Info"</h3>
                                            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 8px; font-size: 0.9rem;">
                                                <span style="color: var(--muted);">"Braine Core:"</span>
                                                <span style="color: var(--text); font-weight: 600;">{VERSION_BRAINE}</span>
                                                <span style="color: var(--muted);">"Brain Image (BBI):"</span>
                                                <span style="color: var(--text); font-weight: 600;">{format!("v{}", VERSION_BBI_FORMAT)}</span>
                                                <span style="color: var(--muted);">"Braine Web:"</span>
                                                <span style="color: var(--text); font-weight: 600;">{VERSION_BRAINE_WEB}</span>
                                            </div>
                                        </div>

                                        <div style=STYLE_CARD>
                                            <h3 style="margin: 0 0 10px 0; font-size: 1rem; color: var(--accent);">"Quick Start"</h3>
                                            <ol style="margin: 0; padding-left: 20px; color: var(--text); font-size: 0.9rem; line-height: 1.8;">
                                                <li>"Select a game tab above (e.g., Spot, Bandit)"</li>
                                                <li>"Click "<strong>"▶ Run"</strong>" to start the learning loop"</li>
                                                <li>"Watch the Stats and Analytics panels on the right"</li>
                                                <li>"Observe accuracy climb as the brain learns"</li>
                                            </ol>
                                        </div>

                                        <div style=STYLE_CARD>
                                            <h3 style="margin: 0 0 10px 0; font-size: 1rem; color: var(--accent);">"Key Principles"</h3>
                                            <div style="display: flex; flex-direction: column; gap: 8px; font-size: 0.9rem;">
                                                <div><strong style="color: var(--text);">"Learning modifies state"</strong><span style="color: var(--muted);">" — plasticity + reward updates couplings"</span></div>
                                                <div><strong style="color: var(--text);">"Inference uses state"</strong><span style="color: var(--muted);">" — actions emerge from dynamics"</span></div>
                                                <div><strong style="color: var(--text);">"Closed loop"</strong><span style="color: var(--muted);">" — stimulus → action → reward → repeat"</span></div>
                                            </div>
                                        </div>
                                    </div>

                                    // Architecture diagram
                                    <div style=STYLE_CARD>
                                        <h3 style="margin: 0 0 12px 0; font-size: 1rem; color: var(--accent);">"System Architecture"</h3>
                                        <div style="display: flex; justify-content: center; padding: 16px 0;">
                                            <svg viewBox="0 0 600 200" style="width: 100%; max-width: 600px; height: auto;">
                                                // Frame box
                                                <rect x="10" y="30" width="150" height="140" rx="8" fill="none" stroke="var(--border)" stroke-width="2"/>
                                                <text x="85" y="22" fill="var(--muted)" font-size="12" text-anchor="middle">"Frame"</text>
                                                <text x="85" y="60" fill="var(--text)" font-size="11" text-anchor="middle">"Sensors"</text>
                                                <text x="85" y="100" fill="var(--text)" font-size="11" text-anchor="middle">"Stimulus Encoder"</text>
                                                <text x="85" y="140" fill="var(--text)" font-size="11" text-anchor="middle">"Actuators"</text>

                                                // Brain box
                                                <rect x="220" y="30" width="200" height="140" rx="8" fill="none" stroke="var(--accent)" stroke-width="2"/>
                                                <text x="320" y="22" fill="var(--accent)" font-size="12" text-anchor="middle" font-weight="bold">"Braine"</text>
                                                <text x="320" y="55" fill="var(--text)" font-size="11" text-anchor="middle">"Wave Dynamics"</text>
                                                <text x="320" y="80" fill="var(--text)" font-size="11" text-anchor="middle">"Local Plasticity"</text>
                                                <text x="320" y="105" fill="var(--text)" font-size="11" text-anchor="middle">"Causality Memory"</text>
                                                <text x="320" y="130" fill="var(--text)" font-size="11" text-anchor="middle">"Action Readout"</text>

                                                // Neuromodulator
                                                <rect x="480" y="60" width="110" height="60" rx="8" fill="none" stroke="#4ade80" stroke-width="2"/>
                                                <text x="535" y="85" fill="#4ade80" font-size="11" text-anchor="middle">"Neuromodulator"</text>
                                                <text x="535" y="102" fill="var(--muted)" font-size="10" text-anchor="middle">"(reward signal)"</text>

                                                // Arrows
                                                <line x1="160" y1="70" x2="220" y2="70" stroke="var(--text)" stroke-width="1.5" marker-end="url(#arrowhead)"/>
                                                <line x1="420" y1="100" x2="480" y2="90" stroke="#4ade80" stroke-width="1.5" marker-end="url(#arrowhead)"/>
                                                <line x1="220" y1="130" x2="160" y2="130" stroke="var(--text)" stroke-width="1.5" marker-end="url(#arrowhead)"/>

                                                // Feedback loop
                                                <path d="M535 120 L535 170 L320 170 L320 170" stroke="var(--accent)" stroke-width="1.5" fill="none" stroke-dasharray="4,2" marker-end="url(#arrowhead)"/>

                                                <defs>
                                                    <marker id="arrowhead" markerWidth="10" markerHeight="7" refX="9" refY="3.5" orient="auto">
                                                        <polygon points="0 0, 10 3.5, 0 7" fill="var(--text)"/>
                                                    </marker>
                                                </defs>
                                            </svg>
                                        </div>
                                    </div>

                                    // Research Disclaimer
                                    <div style="padding: 14px; background: rgba(251, 191, 36, 0.1); border: 1px solid rgba(251, 191, 36, 0.3); border-radius: 12px;">
                                        <h3 style="margin: 0 0 10px 0; font-size: 1rem; color: #fbbf24;">"⚠️ Research Disclaimer"</h3>
                                        <p style="margin: 0; color: var(--text); font-size: 0.85rem; line-height: 1.7;">
                                            "This system was developed with the assistance of Large Language Models (LLMs) under human guidance. "
                                            "It is provided as a "<strong>"research demonstration"</strong>" to explore biologically-inspired learning substrates. "
                                            "Braine is "<strong>"not production-ready"</strong>" and should not be used for real-world deployment, safety-critical applications, "
                                            "or any scenario requiring reliability guarantees. Use at your own discretion for educational and experimental purposes only."
                                        </p>
                                    </div>
                                </Show>

                                // Dynamics tab
                                <Show when=move || about_sub_tab.get() == AboutSubTab::Dynamics>
                                    <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 16px;">
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
                                    <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 16px;">
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
                                        <div style="display: flex; justify-content: center; padding: 12px 0;">
                                            <svg viewBox="0 0 500 180" style="width: 100%; max-width: 500px; height: auto;">
                                                // Macro level
                                                <rect x="10" y="10" width="120" height="35" rx="6" fill="rgba(122, 162, 255, 0.2)" stroke="var(--accent)" stroke-width="1.5"/>
                                                <text x="70" y="32" fill="var(--text)" font-size="10" text-anchor="middle" font-weight="bold">"Child Brains"</text>

                                                // Meso level
                                                <rect x="140" y="10" width="120" height="35" rx="6" fill="rgba(74, 222, 128, 0.2)" stroke="#4ade80" stroke-width="1.5"/>
                                                <text x="200" y="32" fill="var(--text)" font-size="10" text-anchor="middle" font-weight="bold">"Neurogenesis"</text>

                                                // Micro level
                                                <rect x="270" y="10" width="120" height="35" rx="6" fill="rgba(251, 191, 36, 0.2)" stroke="#fbbf24" stroke-width="1.5"/>
                                                <text x="330" y="32" fill="var(--text)" font-size="10" text-anchor="middle" font-weight="bold">"Hebbian"</text>

                                                // Nano level
                                                <rect x="400" y="10" width="90" height="35" rx="6" fill="rgba(244, 114, 182, 0.2)" stroke="#f472b6" stroke-width="1.5"/>
                                                <text x="445" y="32" fill="var(--text)" font-size="10" text-anchor="middle" font-weight="bold">"Imprinting"</text>

                                                // Labels
                                                <text x="70" y="60" fill="var(--muted)" font-size="9" text-anchor="middle">"sec-hours"</text>
                                                <text x="200" y="60" fill="var(--muted)" font-size="9" text-anchor="middle">"steps-min"</text>
                                                <text x="330" y="60" fill="var(--muted)" font-size="9" text-anchor="middle">"per-step"</text>
                                                <text x="445" y="60" fill="var(--muted)" font-size="9" text-anchor="middle">"one-shot"</text>

                                                // Descriptions
                                                <text x="70" y="80" fill="var(--text)" font-size="8" text-anchor="middle">"Behavioral"</text>
                                                <text x="70" y="92" fill="var(--text)" font-size="8" text-anchor="middle">"strategies"</text>
                                                <text x="200" y="80" fill="var(--text)" font-size="8" text-anchor="middle">"Fresh capacity"</text>
                                                <text x="200" y="92" fill="var(--text)" font-size="8" text-anchor="middle">"for concepts"</text>
                                                <text x="330" y="80" fill="var(--text)" font-size="8" text-anchor="middle">"Connection"</text>
                                                <text x="330" y="92" fill="var(--text)" font-size="8" text-anchor="middle">"weights"</text>
                                                <text x="445" y="80" fill="var(--text)" font-size="8" text-anchor="middle">"Stimulus-"</text>
                                                <text x="445" y="92" fill="var(--text)" font-size="8" text-anchor="middle">"concept links"</text>

                                                // Arrow connecting them
                                                <line x1="125" y1="27" x2="140" y2="27" stroke="var(--border)" stroke-width="1"/>
                                                <line x1="260" y1="27" x2="270" y2="27" stroke="var(--border)" stroke-width="1"/>
                                                <line x1="390" y1="27" x2="400" y2="27" stroke="var(--border)" stroke-width="1"/>
                                            </svg>
                                        </div>
                                    </div>
                                </Show>

                                // Memory tab
                                <Show when=move || about_sub_tab.get() == AboutSubTab::Memory>
                                    <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 16px;">
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
                                        <div style="display: flex; justify-content: center; padding: 12px 0;">
                                            <svg viewBox="0 0 500 140" style="width: 100%; max-width: 500px; height: auto;">
                                                // Structural state box
                                                <rect x="10" y="20" width="220" height="100" rx="8" fill="rgba(122, 162, 255, 0.1)" stroke="var(--accent)" stroke-width="1.5"/>
                                                <text x="120" y="40" fill="var(--accent)" font-size="11" text-anchor="middle" font-weight="bold">"Structural State (persisted)"</text>
                                                <text x="30" y="60" fill="var(--text)" font-size="10">"• Units + sparse connections"</text>
                                                <text x="30" y="75" fill="var(--text)" font-size="10">"• Sensor/action groups"</text>
                                                <text x="30" y="90" fill="var(--text)" font-size="10">"• Symbol table"</text>
                                                <text x="30" y="105" fill="var(--text)" font-size="10">"• Causal memory edges"</text>

                                                // Runtime state box
                                                <rect x="270" y="20" width="220" height="100" rx="8" fill="rgba(251, 191, 36, 0.1)" stroke="#fbbf24" stroke-width="1.5"/>
                                                <text x="380" y="40" fill="#fbbf24" font-size="11" text-anchor="middle" font-weight="bold">"Runtime State (transient)"</text>
                                                <text x="290" y="60" fill="var(--text)" font-size="10">"• Pending input current"</text>
                                                <text x="290" y="75" fill="var(--text)" font-size="10">"• Telemetry buffers"</text>
                                                <text x="290" y="90" fill="var(--text)" font-size="10">"• Transient vectors"</text>
                                                <text x="290" y="105" fill="var(--muted)" font-size="10">"(can be reset on load)"</text>
                                            </svg>
                                        </div>
                                    </div>
                                </Show>

                                // Architecture tab
                                <Show when=move || about_sub_tab.get() == AboutSubTab::Architecture>
                                    <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 16px;">
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
                                        <div style="display: flex; justify-content: center; padding: 12px 0;">
                                            <svg viewBox="0 0 600 200" style="width: 100%; max-width: 600px; height: auto;">
                                                // DAEMON SIDE (left)
                                                <text x="130" y="18" fill="var(--muted)" font-size="10" text-anchor="middle" font-weight="bold">"DAEMON-BASED"</text>

                                                // Daemon
                                                <rect x="80" y="30" width="100" height="45" rx="6" fill="rgba(122, 162, 255, 0.2)" stroke="var(--accent)" stroke-width="2"/>
                                                <text x="130" y="48" fill="var(--accent)" font-size="9" text-anchor="middle" font-weight="bold">"brained"</text>
                                                <text x="130" y="62" fill="var(--muted)" font-size="8" text-anchor="middle">"Substrate + TCP server"</text>

                                                // Desktop client
                                                <rect x="10" y="100" width="90" height="40" rx="5" fill="rgba(74, 222, 128, 0.15)" stroke="#4ade80" stroke-width="1.5"/>
                                                <text x="55" y="118" fill="var(--text)" font-size="9" text-anchor="middle">"braine_desktop"</text>
                                                <text x="55" y="130" fill="var(--muted)" font-size="7" text-anchor="middle">"(Slint UI)"</text>

                                                // CLI client
                                                <rect x="110" y="100" width="80" height="40" rx="5" fill="rgba(251, 191, 36, 0.15)" stroke="#fbbf24" stroke-width="1.5"/>
                                                <text x="150" y="118" fill="var(--text)" font-size="9" text-anchor="middle">"braine-cli"</text>
                                                <text x="150" y="130" fill="var(--muted)" font-size="7" text-anchor="middle">"(commands)"</text>

                                                // Storage (daemon side)
                                                <rect x="200" y="30" width="70" height="45" rx="5" fill="rgba(100, 116, 139, 0.15)" stroke="var(--muted)" stroke-width="1.5"/>
                                                <text x="235" y="48" fill="var(--text)" font-size="8" text-anchor="middle">"braine.bbi"</text>
                                                <text x="235" y="62" fill="var(--muted)" font-size="7" text-anchor="middle">"(file system)"</text>

                                                // Protocol label
                                                <text x="100" y="88" fill="var(--muted)" font-size="7">"TCP 9876"</text>

                                                // Arrows to daemon
                                                <line x1="55" y1="100" x2="100" y2="75" stroke="var(--border)" stroke-width="1"/>
                                                <line x1="150" y1="100" x2="160" y2="75" stroke="var(--border)" stroke-width="1"/>
                                                <line x1="180" y1="52" x2="200" y2="52" stroke="var(--border)" stroke-width="1"/>

                                                // SEPARATOR
                                                <line x1="295" y1="20" x2="295" y2="180" stroke="var(--border)" stroke-width="1" stroke-dasharray="4,4"/>

                                                // WEB SIDE (right) - STANDALONE
                                                <text x="450" y="18" fill="var(--muted)" font-size="10" text-anchor="middle" font-weight="bold">"STANDALONE (THIS APP)"</text>

                                                // Web app with embedded brain
                                                <rect x="370" y="30" width="160" height="70" rx="8" fill="rgba(244, 114, 182, 0.15)" stroke="#f472b6" stroke-width="2"/>
                                                <text x="450" y="50" fill="#f472b6" font-size="10" text-anchor="middle" font-weight="bold">"braine_web (WASM)"</text>
                                                <text x="450" y="68" fill="var(--text)" font-size="8" text-anchor="middle">"Braine runs IN-MEMORY"</text>
                                                <text x="450" y="82" fill="var(--muted)" font-size="7" text-anchor="middle">"No daemon connection"</text>

                                                // Browser storage
                                                <rect x="400" y="120" width="100" height="40" rx="5" fill="rgba(100, 116, 139, 0.15)" stroke="var(--muted)" stroke-width="1.5"/>
                                                <text x="450" y="138" fill="var(--text)" font-size="8" text-anchor="middle">"IndexedDB"</text>
                                                <text x="450" y="150" fill="var(--muted)" font-size="7" text-anchor="middle">"(browser storage)"</text>

                                                // Arrow from web to storage
                                                <line x1="450" y1="100" x2="450" y2="120" stroke="var(--border)" stroke-width="1"/>
                                            </svg>
                                        </div>
                                    </div>

                                    // Future Architecture Vision
                                    <div style=STYLE_CARD>
                                        <h3 style="margin: 0 0 12px 0; font-size: 1rem; color: var(--accent);">"Future Vision: Centralized Brain"</h3>
                                        <p style="margin: 0 0 12px 0; color: var(--muted); font-size: 0.85rem; line-height: 1.6;">
                                            "Planned architecture: a central daemon hosts the authoritative Brain, while edge clients maintain local copies that sync in real-time."
                                        </p>
                                        <div style="display: flex; justify-content: center; padding: 12px 0;">
                                            <svg viewBox="0 0 550 180" style="width: 100%; max-width: 550px; height: auto;">
                                                // Central daemon (larger, emphasized)
                                                <rect x="200" y="10" width="150" height="55" rx="8" fill="rgba(122, 162, 255, 0.25)" stroke="var(--accent)" stroke-width="2"/>
                                                <text x="275" y="30" fill="var(--accent)" font-size="11" text-anchor="middle" font-weight="bold">"Central Daemon"</text>
                                                <text x="275" y="46" fill="var(--text)" font-size="8" text-anchor="middle">"Authoritative Brain State"</text>
                                                <text x="275" y="58" fill="var(--muted)" font-size="7" text-anchor="middle">"Learning + Persistence"</text>

                                                // Edge clients with local copies
                                                <rect x="10" y="100" width="110" height="55" rx="6" fill="rgba(74, 222, 128, 0.15)" stroke="#4ade80" stroke-width="1.5"/>
                                                <text x="65" y="118" fill="var(--text)" font-size="9" text-anchor="middle">"Desktop"</text>
                                                <text x="65" y="132" fill="var(--muted)" font-size="7" text-anchor="middle">"Local Brain copy"</text>
                                                <text x="65" y="145" fill="var(--muted)" font-size="7" text-anchor="middle">"(sync: real-time)"</text>

                                                <rect x="140" y="100" width="110" height="55" rx="6" fill="rgba(244, 114, 182, 0.15)" stroke="#f472b6" stroke-width="1.5"/>
                                                <text x="195" y="118" fill="var(--text)" font-size="9" text-anchor="middle">"Web (WASM)"</text>
                                                <text x="195" y="132" fill="var(--muted)" font-size="7" text-anchor="middle">"Local Brain copy"</text>
                                                <text x="195" y="145" fill="var(--muted)" font-size="7" text-anchor="middle">"(sync: WebSocket)"</text>

                                                <rect x="270" y="100" width="110" height="55" rx="6" fill="rgba(34, 211, 238, 0.15)" stroke="#22d3ee" stroke-width="1.5"/>
                                                <text x="325" y="118" fill="var(--text)" font-size="9" text-anchor="middle">"Mobile/IoT"</text>
                                                <text x="325" y="132" fill="var(--muted)" font-size="7" text-anchor="middle">"Local Brain copy"</text>
                                                <text x="325" y="145" fill="var(--muted)" font-size="7" text-anchor="middle">"(sync: MQTT/WS)"</text>

                                                <rect x="400" y="100" width="110" height="55" rx="6" fill="rgba(251, 191, 36, 0.15)" stroke="#fbbf24" stroke-width="1.5"/>
                                                <text x="455" y="118" fill="var(--text)" font-size="9" text-anchor="middle">"CLI / Scripts"</text>
                                                <text x="455" y="132" fill="var(--muted)" font-size="7" text-anchor="middle">"No local copy"</text>
                                                <text x="455" y="145" fill="var(--muted)" font-size="7" text-anchor="middle">"(direct commands)"</text>

                                                // Sync arrows (bidirectional)
                                                <line x1="65" y1="100" x2="220" y2="65" stroke="var(--accent)" stroke-width="1" stroke-dasharray="3,2"/>
                                                <line x1="195" y1="100" x2="250" y2="65" stroke="var(--accent)" stroke-width="1" stroke-dasharray="3,2"/>
                                                <line x1="325" y1="100" x2="300" y2="65" stroke="var(--accent)" stroke-width="1" stroke-dasharray="3,2"/>
                                                <line x1="455" y1="100" x2="330" y2="65" stroke="var(--accent)" stroke-width="1"/>

                                                // Label
                                                <text x="275" y="85" fill="var(--muted)" font-size="8" text-anchor="middle">"↕ Real-time sync"</text>
                                            </svg>
                                        </div>
                                    </div>
                                </Show>

                                // Applications sub-tab - real-world use cases
                                <Show when=move || about_sub_tab.get() == AboutSubTab::Applications>
                                    <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 16px;">
                                        <div style=STYLE_CARD>
                                            <h3 style="margin: 0 0 10px 0; font-size: 1rem; color: var(--accent);">"🤖 Embodied Robotics"</h3>
                                            <p style="margin: 0 0 10px 0; color: var(--text); font-size: 0.9rem; line-height: 1.7;">
                                                "Real-time sensorimotor control for robots and drones:"
                                            </p>
                                            <ul style="margin: 0; padding-left: 20px; color: var(--muted); font-size: 0.85rem; line-height: 1.6;">
                                                <li>"Adaptive locomotion (terrain adaptation)"</li>
                                                <li>"Object manipulation and grasping"</li>
                                                <li>"Navigation with obstacle avoidance"</li>
                                                <li>"Multi-sensor fusion (vision, touch, IMU)"</li>
                                            </ul>
                                        </div>

                                        <div style=STYLE_CARD>
                                            <h3 style="margin: 0 0 10px 0; font-size: 1rem; color: var(--accent);">"🏠 Smart Home & IoT"</h3>
                                            <p style="margin: 0 0 10px 0; color: var(--text); font-size: 0.9rem; line-height: 1.7;">
                                                "Edge intelligence for connected devices:"
                                            </p>
                                            <ul style="margin: 0; padding-left: 20px; color: var(--muted); font-size: 0.85rem; line-height: 1.6;">
                                                <li>"Personalized automation (learns your patterns)"</li>
                                                <li>"Energy optimization based on behavior"</li>
                                                <li>"Anomaly detection (security, maintenance)"</li>
                                                <li>"Voice-free intent recognition"</li>
                                            </ul>
                                        </div>

                                        <div style=STYLE_CARD>
                                            <h3 style="margin: 0 0 10px 0; font-size: 1rem; color: var(--accent);">"🎮 Game AI & NPCs"</h3>
                                            <p style="margin: 0 0 10px 0; color: var(--text); font-size: 0.9rem; line-height: 1.7;">
                                                "Believable, adaptive game characters:"
                                            </p>
                                            <ul style="margin: 0; padding-left: 20px; color: var(--muted); font-size: 0.85rem; line-height: 1.6;">
                                                <li>"NPCs that learn from player interactions"</li>
                                                <li>"Adaptive difficulty (genuine skill matching)"</li>
                                                <li>"Emergent behaviors from simple rules"</li>
                                                <li>"Persistent memory across sessions"</li>
                                            </ul>
                                        </div>

                                        <div style=STYLE_CARD>
                                            <h3 style="margin: 0 0 10px 0; font-size: 1rem; color: var(--accent);">"🧠 Cognitive Prosthetics"</h3>
                                            <p style="margin: 0 0 10px 0; color: var(--text); font-size: 0.9rem; line-height: 1.7;">
                                                "Assistive technologies with learning:"
                                            </p>
                                            <ul style="margin: 0; padding-left: 20px; color: var(--muted); font-size: 0.85rem; line-height: 1.6;">
                                                <li>"Adaptive prosthetic limb control"</li>
                                                <li>"Brain-computer interfaces"</li>
                                                <li>"Personalized sensory substitution"</li>
                                                <li>"Rehabilitation assistance"</li>
                                            </ul>
                                        </div>

                                        <div style=STYLE_CARD>
                                            <h3 style="margin: 0 0 10px 0; font-size: 1rem; color: var(--accent);">"📊 Time-Series Control"</h3>
                                            <p style="margin: 0 0 10px 0; color: var(--text); font-size: 0.9rem; line-height: 1.7;">
                                                "Industrial and process control:"
                                            </p>
                                            <ul style="margin: 0; padding-left: 20px; color: var(--muted); font-size: 0.85rem; line-height: 1.6;">
                                                <li>"Manufacturing process optimization"</li>
                                                <li>"HVAC and climate control"</li>
                                                <li>"Traffic signal adaptation"</li>
                                                <li>"Agricultural automation"</li>
                                            </ul>
                                        </div>

                                        <div style=STYLE_CARD>
                                            <h3 style="margin: 0 0 10px 0; font-size: 1rem; color: var(--accent);">"🔬 Research Platform"</h3>
                                            <p style="margin: 0 0 10px 0; color: var(--text); font-size: 0.9rem; line-height: 1.7;">
                                                "Scientific exploration of intelligence:"
                                            </p>
                                            <ul style="margin: 0; padding-left: 20px; color: var(--muted); font-size: 0.85rem; line-height: 1.6;">
                                                <li>"Neuromorphic computing testbed"</li>
                                                <li>"Embodied cognition experiments"</li>
                                                <li>"Emergence and self-organization studies"</li>
                                                <li>"Educational tool for AI concepts"</li>
                                            </ul>
                                        </div>
                                    </div>

                                    // Web-based Edge Computing Applications
                                    <div style=STYLE_CARD>
                                        <h3 style="margin: 0 0 12px 0; font-size: 1.1rem; color: var(--accent);">"🌐 Web-Based Edge Intelligence"</h3>
                                        <p style="margin: 0 0 12px 0; color: var(--text); font-size: 0.9rem; line-height: 1.7;">
                                            "Braine runs natively in the browser via WebAssembly, enabling intelligent applications at the edge without server round-trips:"
                                        </p>
                                        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 12px;">
                                            <div style="padding: 12px; background: rgba(122, 162, 255, 0.08); border-radius: 8px;">
                                                <strong style="color: var(--text);">"📝 Smart Form Assistants"</strong>
                                                <p style="margin: 6px 0 0 0; color: var(--muted); font-size: 0.8rem; line-height: 1.5;">
                                                    "Learn user input patterns to auto-complete, validate, and suggest corrections—all client-side with no data leaving the browser."
                                                </p>
                                            </div>
                                            <div style="padding: 12px; background: rgba(74, 222, 128, 0.08); border-radius: 8px;">
                                                <strong style="color: var(--text);">"🛒 Personalized E-commerce"</strong>
                                                <p style="margin: 6px 0 0 0; color: var(--muted); font-size: 0.8rem; line-height: 1.5;">
                                                    "Real-time product recommendations that adapt to browsing behavior without tracking servers or cookies."
                                                </p>
                                            </div>
                                            <div style="padding: 12px; background: rgba(251, 191, 36, 0.08); border-radius: 8px;">
                                                <strong style="color: var(--text);">"🎨 Adaptive UI/UX"</strong>
                                                <p style="margin: 6px 0 0 0; color: var(--muted); font-size: 0.8rem; line-height: 1.5;">
                                                    "Interfaces that learn user preferences—button placements, color schemes, information density—and adapt in real-time."
                                                </p>
                                            </div>
                                            <div style="padding: 12px; background: rgba(244, 114, 182, 0.08); border-radius: 8px;">
                                                <strong style="color: var(--text);">"🔐 Behavioral Authentication"</strong>
                                                <p style="margin: 6px 0 0 0; color: var(--muted); font-size: 0.8rem; line-height: 1.5;">
                                                    "Continuous authentication via typing rhythm, mouse patterns, and interaction cadence—private and local."
                                                </p>
                                            </div>
                                            <div style="padding: 12px; background: rgba(34, 211, 238, 0.08); border-radius: 8px;">
                                                <strong style="color: var(--text);">"💬 Offline-First Chat Bots"</strong>
                                                <p style="margin: 6px 0 0 0; color: var(--muted); font-size: 0.8rem; line-height: 1.5;">
                                                    "Conversational agents that work without network—perfect for kiosks, field devices, or privacy-sensitive contexts."
                                                </p>
                                            </div>
                                            <div style="padding: 12px; background: rgba(167, 139, 250, 0.08); border-radius: 8px;">
                                                <strong style="color: var(--text);">"📊 Real-Time Analytics"</strong>
                                                <p style="margin: 6px 0 0 0; color: var(--muted); font-size: 0.8rem; line-height: 1.5;">
                                                    "Stream processing in the browser—anomaly detection, trend prediction, and alerts without backend latency."
                                                </p>
                                            </div>
                                            <div style="padding: 12px; background: rgba(251, 113, 133, 0.08); border-radius: 8px;">
                                                <strong style="color: var(--text);">"🎮 Browser-Based Games"</strong>
                                                <p style="margin: 6px 0 0 0; color: var(--muted); font-size: 0.8rem; line-height: 1.5;">
                                                    "NPCs and opponents that learn player strategies in-session, creating personalized challenge without cloud sync."
                                                </p>
                                            </div>
                                            <div style="padding: 12px; background: rgba(100, 116, 139, 0.08); border-radius: 8px;">
                                                <strong style="color: var(--text);">"📱 Progressive Web Apps"</strong>
                                                <p style="margin: 6px 0 0 0; color: var(--muted); font-size: 0.8rem; line-height: 1.5;">
                                                    "PWAs with embedded intelligence—fitness coaches, language tutors, task managers that learn and work offline."
                                                </p>
                                            </div>
                                            <div style="padding: 12px; background: rgba(56, 189, 248, 0.08); border-radius: 8px;">
                                                <strong style="color: var(--text);">"🏥 Medical Triage Assistants"</strong>
                                                <p style="margin: 6px 0 0 0; color: var(--muted); font-size: 0.8rem; line-height: 1.5;">
                                                    "Symptom checkers that run entirely on-device, ensuring patient data never leaves their control."
                                                </p>
                                            </div>
                                            <div style="padding: 12px; background: rgba(163, 230, 53, 0.08); border-radius: 8px;">
                                                <strong style="color: var(--text);">"🌍 Offline Education"</strong>
                                                <p style="margin: 6px 0 0 0; color: var(--muted); font-size: 0.8rem; line-height: 1.5;">
                                                    "Adaptive learning platforms for regions with unreliable connectivity—personalized tutoring without cloud."
                                                </p>
                                            </div>
                                            <div style="padding: 12px; background: rgba(232, 121, 249, 0.08); border-radius: 8px;">
                                                <strong style="color: var(--text);">"🔧 Industrial Dashboards"</strong>
                                                <p style="margin: 6px 0 0 0; color: var(--muted); font-size: 0.8rem; line-height: 1.5;">
                                                    "HMI panels that predict equipment issues locally, reducing latency for time-critical alerts."
                                                </p>
                                            </div>
                                            <div style="padding: 12px; background: rgba(248, 113, 113, 0.08); border-radius: 8px;">
                                                <strong style="color: var(--text);">"🚗 Fleet Management"</strong>
                                                <p style="margin: 6px 0 0 0; color: var(--muted); font-size: 0.8rem; line-height: 1.5;">
                                                    "In-browser driver behavior analysis for logistics—route optimization and safety scoring without cloud upload."
                                                </p>
                                            </div>
                                        </div>
                                    </div>

                                    <div style=STYLE_CARD>
                                        <h3 style="margin: 0 0 10px 0; font-size: 1rem; color: var(--accent);">"Why Braine for These Applications?"</h3>
                                        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 12px; margin-top: 12px;">
                                            <div style="padding: 12px; background: rgba(122, 162, 255, 0.08); border-radius: 8px;">
                                                <strong style="color: var(--text);">"⚡ Real-time"</strong>
                                                <p style="margin: 6px 0 0 0; color: var(--muted); font-size: 0.8rem;">"O(1) step complexity, no batching needed"</p>
                                            </div>
                                            <div style="padding: 12px; background: rgba(74, 222, 128, 0.08); border-radius: 8px;">
                                                <strong style="color: var(--text);">"💾 Edge-friendly"</strong>
                                                <p style="margin: 6px 0 0 0; color: var(--muted); font-size: 0.8rem;">"Runs on MCUs, no cloud dependency"</p>
                                            </div>
                                            <div style="padding: 12px; background: rgba(251, 191, 36, 0.08); border-radius: 8px;">
                                                <strong style="color: var(--text);">"🔄 Online learning"</strong>
                                                <p style="margin: 6px 0 0 0; color: var(--muted); font-size: 0.8rem;">"Adapts continuously, no retraining"</p>
                                            </div>
                                            <div style="padding: 12px; background: rgba(244, 114, 182, 0.08); border-radius: 8px;">
                                                <strong style="color: var(--text);">"🔍 Interpretable"</strong>
                                                <p style="margin: 6px 0 0 0; color: var(--muted); font-size: 0.8rem;">"Inspect dynamics, not black box"</p>
                                            </div>
                                        </div>
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

                                    <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(320px, 1fr)); gap: 16px;">
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
                                        <pre style="margin: 0; white-space: pre-wrap; background: var(--bg); padding: 12px; border-radius: 8px; font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, 'Liberation Mono', 'Courier New', monospace; font-size: 0.82rem; line-height: 1.6; color: var(--text);">{ABOUT_LLM_DATAFLOW}</pre>
                                    </div>

                                    <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(320px, 1fr)); gap: 16px;">
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

                                    <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(320px, 1fr)); gap: 16px;">
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
                                prop:value=move || format!("{:.2}", exploration_eps.get())
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
                                prop:value=move || format!("{:.1}", meaning_alpha.get())
                                on:input=move |ev| {
                                    let v = event_target_value(&ev);
                                    if let Ok(x) = v.parse::<f32>() {
                                        set_meaning_alpha.set(x.clamp(0.0, 30.0));
                                    }
                                }
                            />
                        </label>
                    </div>

                    // Game-specific content area
                    <div class="canvas-container">
                        // Spot game - Enhanced with visual arena
                        <Show when=move || game_kind.get() == GameKind::Spot>
                            <div style="display: flex; flex-direction: column; align-items: center; gap: 16px; width: 100%; max-width: 400px;">
                                <div style="text-align: center; margin-bottom: 8px;">
                                    <h2 style="margin: 0; font-size: 1.1rem; color: var(--text);">"Spot Discrimination"</h2>
                                    <p style="margin: 4px 0 0 0; color: var(--muted); font-size: 0.85rem;">"Learn to respond LEFT or RIGHT based on stimulus"</p>
                                </div>
                                // Visual arena
                                <div class="arena-grid">
                                    <div style=move || {
                                        let active = matches!(spot_is_left.get(), Some(true));
                                        format!(
                                            "display: flex; flex-direction: column; align-items: center; justify-content: center; padding: 30px; border-radius: 12px; border: 2px solid {}; background: {};",
                                            if active { "var(--accent)" } else { "var(--border)" },
                                            if active { "rgba(122, 162, 255, 0.15)" } else { "rgba(0,0,0,0.2)" },
                                        )
                                    }>
                                        <span style="font-size: 3rem;">"⬅️"</span>
                                        <span style="margin-top: 8px; font-size: 0.9rem; font-weight: 600; color: var(--text);">"LEFT"</span>
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
                                        <span style="margin-top: 8px; font-size: 0.9rem; font-weight: 600; color: var(--text);">"RIGHT"</span>
                                        <span style="font-size: 0.75rem; color: var(--muted);">"Press D"</span>
                                    </div>
                                </div>
                                // Response indicator
                                <div style="display: flex; align-items: center; gap: 12px; padding: 12px 20px; background: rgba(0,0,0,0.2); border-radius: 8px;">
                                    <span style="color: var(--muted); font-size: 0.85rem;">"Braine chose:"</span>
                                    <span style="font-size: 1.1rem; font-weight: 600; color: var(--accent);">
                                        {move || { let a = last_action.get(); if a.is_empty() { "—".to_string() } else { a.to_uppercase() } }}
                                    </span>
                                    <span style=move || format!("padding: 4px 10px; border-radius: 4px; font-size: 0.8rem; font-weight: 600; background: {}; color: #fff;",
                                        if last_reward.get() > 0.0 { "#22c55e" } else if last_reward.get() < 0.0 { "#ef4444" } else { "#64748b" })>
                                        {move || if last_reward.get() > 0.0 { "✓ Correct" } else if last_reward.get() < 0.0 { "✗ Wrong" } else { "—" }}
                                    </span>
                                </div>
                            </div>
                        </Show>

                        // Bandit game - Enhanced with arm visualization
                        <Show when=move || game_kind.get() == GameKind::Bandit>
                            <div style="display: flex; flex-direction: column; align-items: center; gap: 16px; width: 100%; max-width: 420px;">
                                <div style="text-align: center; margin-bottom: 8px;">
                                    <h2 style="margin: 0; font-size: 1.1rem; color: var(--text);">"🎰 Two-Armed Bandit"</h2>
                                    <p style="margin: 4px 0 0 0; color: var(--muted); font-size: 0.85rem;">"Explore vs exploit: learn which arm pays better"</p>
                                </div>
                                // Slot machine arms
                                <div class="arena-grid" style="gap: 16px;">
                                    <div style=move || format!("display: flex; flex-direction: column; align-items: center; padding: 24px; border-radius: 12px; border: 2px solid {}; background: {}; transition: all 0.2s;",
                                        if last_action.get() == "left" { "#fbbf24" } else { "var(--border)" },
                                        if last_action.get() == "left" { "rgba(251, 191, 36, 0.1)" } else { "rgba(0,0,0,0.2)" })>
                                        <div style="font-size: 3rem; margin-bottom: 8px;">"🎰"</div>
                                        <span style="font-size: 1rem; font-weight: 700; color: var(--text);">"ARM A"</span>
                                        <span style="font-size: 0.75rem; color: var(--muted); margin-top: 4px;">"Press A"</span>
                                    </div>
                                    <div style=move || format!("display: flex; flex-direction: column; align-items: center; padding: 24px; border-radius: 12px; border: 2px solid {}; background: {}; transition: all 0.2s;",
                                        if last_action.get() == "right" { "#fbbf24" } else { "var(--border)" },
                                        if last_action.get() == "right" { "rgba(251, 191, 36, 0.1)" } else { "rgba(0,0,0,0.2)" })>
                                        <div style="font-size: 3rem; margin-bottom: 8px;">"🎰"</div>
                                        <span style="font-size: 1rem; font-weight: 700; color: var(--text);">"ARM B"</span>
                                        <span style="font-size: 0.75rem; color: var(--muted); margin-top: 4px;">"Press D"</span>
                                    </div>
                                </div>
                                // Reward flash
                                <div style=move || format!("display: flex; align-items: center; justify-content: center; gap: 8px; padding: 16px 24px; border-radius: 8px; background: {}; min-width: 180px;",
                                    if last_reward.get() > 0.0 { "rgba(34, 197, 94, 0.2)" } else if last_reward.get() < 0.0 { "rgba(239, 68, 68, 0.2)" } else { "rgba(0,0,0,0.2)" })>
                                    <span style=move || format!("font-size: 1.5rem; font-weight: 700; color: {};",
                                        if last_reward.get() > 0.0 { "#4ade80" } else if last_reward.get() < 0.0 { "#f87171" } else { "var(--muted)" })>
                                        {move || format!("{:+.1}", last_reward.get())}
                                    </span>
                                    <span style="color: var(--muted); font-size: 0.85rem;">"reward"</span>
                                </div>
                            </div>
                        </Show>

                        // Spot Reversal - Enhanced with context indicator
                        <Show when=move || game_kind.get() == GameKind::SpotReversal>
                            <div style="display: flex; flex-direction: column; align-items: center; gap: 16px; width: 100%; max-width: 420px;">
                                <div style="text-align: center; margin-bottom: 8px;">
                                    <h2 style="margin: 0; font-size: 1.1rem; color: var(--text);">"Spot Reversal"</h2>
                                    <p style="margin: 4px 0 0 0; color: var(--muted); font-size: 0.85rem;">"Rules flip periodically; context bit helps detect reversals"</p>
                                </div>
                                // Reversal status badge
                                <div style=move || format!("display: flex; align-items: center; gap: 12px; padding: 12px 24px; border-radius: 24px; background: {}; border: 2px solid {};",
                                    if reversal_active.get() { "rgba(251, 191, 36, 0.15)" } else { "rgba(122, 162, 255, 0.1)" },
                                    if reversal_active.get() { "#fbbf24" } else { "var(--accent)" })>
                                    <span style="font-size: 1.5rem;">{move || if reversal_active.get() { "🔄" } else { "➡️" }}</span>
                                    <div>
                                        <div style=move || format!("font-weight: 700; font-size: 1rem; color: {};", if reversal_active.get() { "#fbbf24" } else { "var(--accent)" })>
                                            {move || if reversal_active.get() { "REVERSED" } else { "NORMAL" }}
                                        </div>
                                        <div style="font-size: 0.75rem; color: var(--muted);">
                                            {move || format!("Flips after {} trials", reversal_flip_after.get())}
                                        </div>
                                    </div>
                                </div>
                                // Arena (same as Spot but with reversal indicator)
                                <div class="arena-grid">
                                    <div style=move || {
                                        let active = matches!(spot_is_left.get(), Some(true));
                                        format!(
                                            "display: flex; flex-direction: column; align-items: center; justify-content: center; padding: 24px; border-radius: 12px; border: 2px solid {}; background: {};",
                                            if active { "var(--accent)" } else { "var(--border)" },
                                            if active { "rgba(122, 162, 255, 0.15)" } else { "rgba(0,0,0,0.2)" },
                                        )
                                    }>
                                        <span style="font-size: 2.5rem;">"⬅️"</span>
                                        <span style="margin-top: 8px; font-weight: 600; color: var(--text);">"LEFT"</span>
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
                                        <span style="margin-top: 8px; font-weight: 600; color: var(--text);">"RIGHT"</span>
                                    </div>
                                </div>
                            </div>
                        </Show>

                        // SpotXY game with canvas - Modern gaming design
                        <Show when=move || game_kind.get() == GameKind::SpotXY>
                            <div class="game-shell" style="max-width: 480px;">
                                // Header with mode indicator
                                <div style="display: flex; align-items: center; justify-content: space-between; padding: 16px 20px; background: linear-gradient(135deg, rgba(122, 162, 255, 0.1), rgba(0,0,0,0.3)); border: 1px solid var(--border); border-radius: 16px;">
                                    <div>
                                        <h2 style="margin: 0; font-size: 1.2rem; font-weight: 700; color: var(--text); display: flex; align-items: center; gap: 8px;">
                                            <span style="font-size: 1.5rem;">"🎯"</span>
                                            "SpotXY Tracker"
                                        </h2>
                                        <p style="margin: 4px 0 0 0; color: var(--muted); font-size: 0.8rem;">"Predict the dot position in a 2D grid"</p>
                                    </div>
                                    <div style=move || format!("padding: 8px 16px; border-radius: 20px; font-size: 0.85rem; font-weight: 600; background: {}; color: {};",
                                        if spotxy_eval.get() { "linear-gradient(135deg, #22c55e, #16a34a)" } else { "rgba(122, 162, 255, 0.2)" },
                                        if spotxy_eval.get() { "#fff" } else { "var(--accent)" })>
                                        {move || if spotxy_eval.get() { "🧪 EVAL" } else { "📚 TRAIN" }}
                                    </div>
                                </div>

                                // Canvas container with ambient glow
                                <div class="game-canvas-wrap">
                                    <div style=move || format!("position: absolute; width: 280px; height: 280px; border-radius: 50%; filter: blur(60px); opacity: 0.3; pointer-events: none; background: {};",
                                        if spotxy_eval.get() { "#22c55e" } else { "#7aa2ff" })>
                                    </div>
                                    <div class="game-canvas-frame" style=move || format!("background: {};",
                                        if spotxy_eval.get() { "linear-gradient(135deg, #22c55e, #16a34a)" } else { "linear-gradient(135deg, #7aa2ff, #5b7dc9)" })>
                                        <canvas
                                            node_ref=canvas_ref
                                            width="340"
                                            height="340"
                                            class="game-canvas square"
                                            style="border-radius: 13px; background: #0a0f1a;"
                                        ></canvas>
                                    </div>
                                </div>

                                // Controls row
                                <div style="display: flex; gap: 12px; justify-content: center; flex-wrap: wrap;">
                                    <div style="display: flex; gap: 4px; padding: 4px; background: rgba(0,0,0,0.3); border-radius: 10px;">
                                        <button style="padding: 10px 16px; border: none; background: rgba(122, 162, 255, 0.15); color: var(--accent); border-radius: 8px; cursor: pointer; font-size: 0.9rem; font-weight: 600; transition: all 0.15s;"
                                            on:click=move |_| do_spotxy_grid_minus()>"−"</button>
                                        <div style="padding: 10px 16px; color: var(--text); font-size: 0.9rem; font-weight: 600; min-width: 60px; text-align: center;">
                                            {move || {
                                                let n = spotxy_grid_n.get();
                                                if n == 0 {
                                                    "1×1".to_string()
                                                } else {
                                                    format!("{n}×{n}")
                                                }
                                            }}
                                        </div>
                                        <button style="padding: 10px 16px; border: none; background: rgba(122, 162, 255, 0.15); color: var(--accent); border-radius: 8px; cursor: pointer; font-size: 0.9rem; font-weight: 600; transition: all 0.15s;"
                                            on:click=move |_| do_spotxy_grid_plus()>"+"</button>
                                    </div>
                                    <button style=move || format!("padding: 10px 20px; border: none; border-radius: 10px; cursor: pointer; font-size: 0.9rem; font-weight: 600; transition: all 0.15s; background: {}; color: {};",
                                        if spotxy_eval.get() { "#22c55e" } else { "rgba(122, 162, 255, 0.15)" },
                                        if spotxy_eval.get() { "#fff" } else { "var(--accent)" })
                                        on:click=move |_| do_spotxy_toggle_eval()>
                                        {move || if spotxy_eval.get() { "Switch to Train" } else { "Switch to Eval" }}
                                    </button>
                                </div>

                                // Status pills
                                <div style="display: flex; gap: 8px; justify-content: center;">
                                    <span style="padding: 6px 14px; background: rgba(122, 162, 255, 0.1); border: 1px solid rgba(122, 162, 255, 0.2); border-radius: 20px; font-size: 0.8rem; color: var(--muted);">
                                        "Mode: "<span style="color: var(--accent); font-weight: 600;">{move || spotxy_mode.get()}</span>
                                    </span>
                                </div>
                            </div>
                        </Show>

                        // Pong game - Modern arcade design
                        <Show when=move || game_kind.get() == GameKind::Pong>
                            <div class="game-shell" style="max-width: 580px;">
                                // Header
                                <div style="display: flex; align-items: center; justify-content: space-between; padding: 16px 20px; background: linear-gradient(135deg, rgba(251, 191, 36, 0.1), rgba(0,0,0,0.3)); border: 1px solid var(--border); border-radius: 16px;">
                                    <div>
                                        <h2 style="margin: 0; font-size: 1.2rem; font-weight: 700; color: var(--text); display: flex; align-items: center; gap: 8px;">
                                            <span style="font-size: 1.5rem;">"🏓"</span>
                                            "Pong Arena"
                                        </h2>
                                        <p style="margin: 4px 0 0 0; color: var(--muted); font-size: 0.8rem;">"Control the paddle • Intercept the ball"</p>
                                    </div>
                                    <div style="display: flex; gap: 8px;">
                                        <kbd style="padding: 4px 10px; background: rgba(255,255,255,0.1); border: 1px solid var(--border); border-radius: 6px; font-size: 0.75rem; color: var(--muted);">"W/S"</kbd>
                                        <kbd style="padding: 4px 10px; background: rgba(255,255,255,0.1); border: 1px solid var(--border); border-radius: 6px; font-size: 0.75rem; color: var(--muted);">"↑/↓"</kbd>
                                    </div>
                                </div>

                                // Game canvas
                                <div class="game-canvas-wrap">
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

                                // Parameter sliders in a compact card
                                <div style="display: flex; gap: 16px; justify-content: center; flex-wrap: wrap; padding: 16px 20px; background: rgba(0,0,0,0.3); border: 1px solid var(--border); border-radius: 12px;">
                                    <div style="display: flex; flex-direction: column; gap: 4px; min-width: 90px;">
                                        <span style="font-size: 0.7rem; color: var(--muted); text-transform: uppercase; letter-spacing: 1px;">"Paddle Speed"</span>
                                        <input
                                            type="range"
                                            min="0.5"
                                            max="5"
                                            step="0.1"
                                            style="width: 100%; accent-color: #7aa2ff;"
                                            prop:value=move || format!("{:.1}", pong_paddle_speed.get())
                                            on:input=move |ev| {
                                                let v = event_target_value(&ev);
                                                if let Ok(x) = v.parse::<f32>() {
                                                    do_pong_set_param("paddle_speed", x);
                                                }
                                            }
                                        />
                                        <span style="font-size: 0.8rem; color: var(--text); font-weight: 600; text-align: center;">{move || format!("{:.1}", pong_paddle_speed.get())}</span>
                                    </div>
                                    <div style="display: flex; flex-direction: column; gap: 4px; min-width: 90px;">
                                        <span style="font-size: 0.7rem; color: var(--muted); text-transform: uppercase; letter-spacing: 1px;">"Paddle Size"</span>
                                        <input
                                            type="range"
                                            min="0.05"
                                            max="0.5"
                                            step="0.01"
                                            style="width: 100%; accent-color: #7aa2ff;"
                                            prop:value=move || format!("{:.2}", pong_paddle_half_height.get())
                                            on:input=move |ev| {
                                                let v = event_target_value(&ev);
                                                if let Ok(x) = v.parse::<f32>() {
                                                    do_pong_set_param("paddle_half_height", x);
                                                }
                                            }
                                        />
                                        <span style="font-size: 0.8rem; color: var(--text); font-weight: 600; text-align: center;">{move || format!("{:.0}%", pong_paddle_half_height.get() * 100.0)}</span>
                                    </div>
                                    <div style="display: flex; flex-direction: column; gap: 4px; min-width: 90px;">
                                        <span style="font-size: 0.7rem; color: var(--muted); text-transform: uppercase; letter-spacing: 1px;">"Ball Speed"</span>
                                        <input
                                            type="range"
                                            min="0.3"
                                            max="3"
                                            step="0.1"
                                            style="width: 100%; accent-color: #fbbf24;"
                                            prop:value=move || format!("{:.1}", pong_ball_speed.get())
                                            on:input=move |ev| {
                                                let v = event_target_value(&ev);
                                                if let Ok(x) = v.parse::<f32>() {
                                                    do_pong_set_param("ball_speed", x);
                                                }
                                            }
                                        />
                                        <span style="font-size: 0.8rem; color: var(--text); font-weight: 600; text-align: center;">{move || format!("{:.1}×", pong_ball_speed.get())}</span>
                                    </div>
                                </div>
                            </div>
                        </Show>

                        // Sequence game - Enhanced
                        <Show when=move || game_kind.get() == GameKind::Sequence>
                            <div style="display: flex; flex-direction: column; align-items: center; gap: 16px; width: 100%; max-width: 400px;">
                                <div style="text-align: center;">
                                    <h2 style="margin: 0; font-size: 1.1rem; color: var(--text);">"Sequence Prediction"</h2>
                                    <p style="margin: 4px 0 0 0; color: var(--muted); font-size: 0.85rem;">"Learn repeating patterns: A→B→C→A..."</p>
                                </div>
                                // Token display
                                <div style="display: flex; flex-direction: column; align-items: center; gap: 8px; padding: 24px; background: rgba(0,0,0,0.3); border-radius: 16px; width: 100%;">
                                    <span style="color: var(--muted); font-size: 0.8rem; text-transform: uppercase; letter-spacing: 1px;">"Current Token"</span>
                                    <div style="font-size: 4rem; font-family: 'Courier New', monospace; font-weight: 700; color: var(--accent);">
                                        {move || sequence_state.get().map(|s| s.token.clone()).unwrap_or_else(|| "?".to_string())}
                                    </div>
                                    <div style="display: flex; align-items: center; gap: 8px; margin-top: 8px;">
                                        <span style="font-size: 1.5rem; color: var(--muted);">"↓"</span>
                                    </div>
                                    <span style="color: var(--muted); font-size: 0.8rem; text-transform: uppercase; letter-spacing: 1px;">"Target Next"</span>
                                    <div style="font-size: 2.5rem; font-family: 'Courier New', monospace; color: var(--text); opacity: 0.7;">
                                        {move || sequence_state.get().map(|s| s.target_next.clone()).unwrap_or_else(|| "?".to_string())}
                                    </div>
                                </div>
                                // Stats bar
                                <div style="display: flex; gap: 12px; flex-wrap: wrap; justify-content: center; font-size: 0.8rem;">
                                    <span style="padding: 6px 12px; background: rgba(122, 162, 255, 0.15); border-radius: 6px; color: var(--text);">
                                        {move || sequence_state.get().map(|s| format!("Regime {}", s.regime)).unwrap_or_default()}
                                    </span>
                                    <span style="padding: 6px 12px; background: rgba(122, 162, 255, 0.15); border-radius: 6px; color: var(--text);">
                                        {move || sequence_state.get().map(|s| format!("Outcomes: {}", s.outcomes)).unwrap_or_default()}
                                    </span>
                                    <span style="padding: 6px 12px; background: rgba(122, 162, 255, 0.15); border-radius: 6px; color: var(--text);">
                                        {move || sequence_state.get().map(|s| format!("Shift: {}", s.shift_every)).unwrap_or_default()}
                                    </span>
                                </div>
                            </div>
                        </Show>

                        // Text game - Enhanced
                        <Show when=move || game_kind.get() == GameKind::Text>
                            <div style="display: flex; flex-direction: column; align-items: center; gap: 16px; width: 100%; max-width: 500px;">
                                <div style="text-align: center;">
                                    <h2 style="margin: 0; font-size: 1.1rem; color: var(--text);">"📝 Text Prediction"</h2>
                                    <p style="margin: 4px 0 0 0; color: var(--muted); font-size: 0.85rem;">"Next-token prediction: observe and predict"</p>
                                </div>
                                // Token display card
                                <div style="width: 100%; padding: 24px; background: linear-gradient(135deg, rgba(122, 162, 255, 0.1), rgba(0,0,0,0.3)); border: 1px solid var(--border); border-radius: 16px;">
                                    <div style="text-align: center;">
                                        <span style="color: var(--muted); font-size: 0.75rem; text-transform: uppercase; letter-spacing: 2px;">"Current Token"</span>
                                        <div style="font-size: 2.5rem; font-family: 'Courier New', monospace; font-weight: 700; color: var(--accent); margin: 12px 0; padding: 16px; background: rgba(0,0,0,0.4); border-radius: 8px;">
                                            {move || text_state.get().map(|s| format!("\"{}\"", s.token)).unwrap_or_else(|| "\"?\"".to_string())}
                                        </div>
                                        <div style="display: flex; justify-content: center; align-items: center; gap: 12px; margin: 16px 0;">
                                            <div style="flex: 1; height: 1px; background: var(--border);"></div>
                                            <span style="color: var(--muted); font-size: 0.9rem;">"predict →"</span>
                                            <div style="flex: 1; height: 1px; background: var(--border);"></div>
                                        </div>
                                        <span style="color: var(--muted); font-size: 0.75rem; text-transform: uppercase; letter-spacing: 2px;">"Next Token"</span>
                                        <div style="font-size: 1.8rem; font-family: 'Courier New', monospace; color: var(--text); opacity: 0.8; margin-top: 12px;">
                                            {move || text_state.get().map(|s| format!("\"{}\"", s.target_next)).unwrap_or_else(|| "\"?\"".to_string())}
                                        </div>
                                    </div>
                                </div>
                                // Info pills
                                <div style="display: flex; gap: 8px; flex-wrap: wrap; justify-content: center;">
                                    <span style="padding: 6px 14px; background: rgba(251, 191, 36, 0.15); border: 1px solid rgba(251, 191, 36, 0.3); border-radius: 20px; font-size: 0.8rem; color: #fbbf24;">
                                        {move || text_state.get().map(|s| format!("Vocab: {}", s.vocab_size)).unwrap_or_default()}
                                    </span>
                                    <span style="padding: 6px 14px; background: rgba(122, 162, 255, 0.15); border: 1px solid rgba(122, 162, 255, 0.3); border-radius: 20px; font-size: 0.8rem; color: var(--accent);">
                                        {move || text_state.get().map(|s| format!("Regime {}", s.regime)).unwrap_or_default()}
                                    </span>
                                    <span style="padding: 6px 14px; background: rgba(122, 162, 255, 0.12); border: 1px solid rgba(122, 162, 255, 0.25); border-radius: 20px; font-size: 0.8rem; color: var(--text);">
                                        {move || text_state.get().map(|s| format!("Outcomes: {}", s.outcomes)).unwrap_or_default()}
                                    </span>
                                    <span style="padding: 6px 14px; background: rgba(74, 222, 128, 0.15); border: 1px solid rgba(74, 222, 128, 0.3); border-radius: 20px; font-size: 0.8rem; color: #4ade80;">
                                        {move || text_state.get().map(|s| format!("Shift: {}", s.shift_every)).unwrap_or_default()}
                                    </span>
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
                        <Show when=move || dashboard_tab.get() == DashboardTab::GameDetails>
                            <div class="stack">
                                <div class="hero">
                                    <div class="hero-icon">{move || game_kind.get().icon()}</div>
                                    <h2 class="hero-title">{move || game_kind.get().display_name()}</h2>
                                    <p class="hero-desc">{move || game_kind.get().description()}</p>
                                </div>

                                <div class="card">
                                    <h3 class="card-title">"🧪 What This Tests"</h3>
                                    <pre class="pre">{move || game_kind.get().what_it_tests()}</pre>
                                </div>

                                <div class="card">
                                    <h3 class="card-title">"⌨️ Inputs & Actions"</h3>
                                    <pre class="codeblock">{move || game_kind.get().inputs_info()}</pre>
                                </div>

                                <div class="card">
                                    <h3 class="card-title">"🎁 Reward Structure"</h3>
                                    <pre class="codeblock">{move || game_kind.get().reward_info()}</pre>
                                </div>

                                <div class="card">
                                    <h3 class="card-title">"🎯 Learning Objectives"</h3>
                                    <pre class="pre">{move || game_kind.get().learning_objectives()}</pre>
                                </div>

                                <div class="callout">
                                    <p>
                                        "💡 "<strong>"Braine"</strong>" learns via local plasticity + neuromodulation (reward). No backprop. See the About tab for details."
                                    </p>
                                </div>

                                <Show when=move || game_kind.get() == GameKind::Text>
                                    <div class="card">
                                        <h3 class="card-title">"📚 Text Training Data (Task Definition)"</h3>
                                        <p class="subtle">"This rebuilds the Text game (vocab + sensors/actions) but keeps the same brain."</p>

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
                                        <p class="subtle">"After applying, use Run/Step to train on the stream. Disable Learning Writes in Settings to probe without training."</p>
                                    </div>

                                    <div class="card">
                                        <h3 class="card-title">"🏋️ Prompt Training (Supervised Reward)"</h3>
                                        <p class="subtle">"Walks adjacent byte pairs in the prompt and rewards +1 for predicting the next token, −1 otherwise."</p>

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
                                            <label class="label">
                                                <span>"Epochs"</span>
                                                <input
                                                    class="input compact"
                                                    type="number"
                                                    min="1"
                                                    step="1"
                                                    prop:value=move || text_train_epochs.get().to_string()
                                                    on:input=move |ev| {
                                                        if let Ok(v) = event_target_value(&ev).parse::<u32>() {
                                                            set_text_train_epochs.set(v.max(1));
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
                                                "Train prompt"
                                            </button>
                                        </div>

                                        <p class="subtle">"Tip: Use the Interactive Text Prediction table below to probe changes immediately (it runs on a cloned brain)."</p>
                                    </div>

                                    <div class="card">
                                        <h3 class="card-title">"🧪 Interactive Text Prediction (Inference)"</h3>
                                        <p class="subtle">"Web-only inference (no daemon): this does not train; it runs on a cloned brain so it won’t perturb the running game. Predictions come from learned meaning/association memory."</p>

                                        <div class="row end wrap">
                                            <div class="subtle">"Examples:"</div>
                                            <button class="btn sm" on:click=move |_| { set_text_prompt.set("hello worl".to_string()); set_text_prompt_regime.set(0); }>
                                                "hello worl (r0)"
                                            </button>
                                            <button class="btn sm" on:click=move |_| { set_text_prompt.set("goodbye worl".to_string()); set_text_prompt_regime.set(1); }>
                                                "goodbye worl (r1)"
                                            </button>
                                            <button class="btn sm" on:click=move |_| { set_text_prompt.set("hello w".to_string()); set_text_prompt_regime.set(0); }>
                                                "hello w"
                                            </button>
                                        </div>

                                        <label class="label">
                                            <span>"Prompt"</span>
                                            <input
                                                class="input"
                                                type="text"
                                                prop:value=move || text_prompt.get()
                                                on:input=move |ev| set_text_prompt.set(event_target_value(&ev))
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
                                                    prop:value=move || text_prompt_regime.get().to_string()
                                                    on:input=move |ev| {
                                                        if let Ok(v) = event_target_value(&ev).parse::<u32>() {
                                                            set_text_prompt_regime.set(v.clamp(0, 1));
                                                        }
                                                    }
                                                />
                                            </label>
                                            <label class="label">
                                                <span>"Temp"</span>
                                                <input
                                                    class="input compact"
                                                    type="number"
                                                    min="0.1"
                                                    max="10"
                                                    step="0.1"
                                                    prop:value=move || format!("{:.1}", text_temp.get())
                                                    on:input=move |ev| {
                                                        if let Ok(v) = event_target_value(&ev).parse::<f32>() {
                                                            set_text_temp.set(v.clamp(0.1, 10.0));
                                                        }
                                                    }
                                                />
                                            </label>
                                            <button
                                                class="btn"
                                                on:click={
                                                    let runtime = runtime.clone();
                                                    move |_| {
                                                        // Gather current vocab/sensors from the active Text game (so we can map OOV to UNK).
                                                        let (known_sensors, known_actions) = runtime.with_value(|r| {
                                                            match &r.game {
                                                                WebGame::Text(g) => (g.token_sensor_names().to_vec(), g.allowed_actions().to_vec()),
                                                                _ => (Vec::new(), Vec::new()),
                                                            }
                                                        });

                                                        let prompt = text_prompt.get_untracked();
                                                        let regime = text_prompt_regime.get_untracked();
                                                        let temp = text_temp.get_untracked();
                                                        let alpha = meaning_alpha.get_untracked();

                                                        let preds: Vec<(String, f32, f32)> = runtime.with_value(|r| {
                                                            let mut b = r.brain.clone();
                                                            b.set_neuromodulator(0.0);

                                                            // Choose the last byte of the prompt as the current token.
                                                            let cur_byte = prompt.as_bytes().last().copied();
                                                            let tok_sensor = choose_text_token_sensor(cur_byte, &known_sensors);

                                                            // Apply the same inputs that the Text game uses.
                                                            b.apply_stimulus(Stimulus::new("text", 1.0));
                                                            let regime_sensor = if regime == 1 { "txt_regime_1" } else { "txt_regime_0" };
                                                            b.apply_stimulus(Stimulus::new(regime_sensor, 0.8));
                                                            b.apply_stimulus(Stimulus::new(&tok_sensor, 1.0));

                                                            // Context key matches the game's conditioning symbol.
                                                            let ctx = format!(
                                                                "txt_r{}_c{}",
                                                                if regime == 1 { 1 } else { 0 },
                                                                token_action_name_from_sensor(&tok_sensor)
                                                            );
                                                            b.note_compound_symbol(&[ctx.as_str()]);
                                                            b.step();
                                                            b.discard_observation();

                                                            let ranked = b.ranked_actions_with_meaning(ctx.as_str(), alpha);
                                                            let mut top: Vec<(String, f32)> = ranked
                                                                .into_iter()
                                                                .filter(|(a, _)| known_actions.iter().any(|k| k == a))
                                                                .take(8)
                                                                .collect();
                                                            if top.is_empty() {
                                                                top = known_actions
                                                                    .iter()
                                                                    .take(8)
                                                                    .map(|a| (a.clone(), 0.0))
                                                                    .collect();
                                                            }

                                                            // Softmax for a display-probability using temperature.
                                                            let probs = softmax_temp(&top, temp);
                                                            top.into_iter()
                                                                .zip(probs.into_iter())
                                                                .map(|((a, s), p)| (a, s, p))
                                                                .collect()
                                                        });

                                                        set_text_preds.set(preds);
                                                    }
                                                }
                                            >
                                                "Predict next"
                                            </button>
                                        </div>

                                        <div class="subtle" style="margin-top: 10px;">
                                            {move || {
                                                let preds = text_preds.get();
                                                if let Some((a, _s, p)) = preds.first() {
                                                    let disp = display_token_from_action(a);
                                                    format!("Top prediction: {}  ({:.1}% softmax)", disp, p * 100.0)
                                                } else {
                                                    "Top prediction: —".to_string()
                                                }
                                            }}
                                        </div>

                                        <div class="divider"></div>
                                        <div class="subtle">"Top predictions (score + softmax probability)"</div>
                                        <table class="table">
                                            <thead>
                                                <tr>
                                                    <th>"Token"</th>
                                                    <th>"Action"</th>
                                                    <th>"Score"</th>
                                                    <th>"P"</th>
                                                </tr>
                                            </thead>
                                            <tbody>
                                                {move || {
                                                    text_preds
                                                        .get()
                                                        .into_iter()
                                                        .map(|(a, s, p)| {
                                                            let disp = display_token_from_action(&a);
                                                            view! {
                                                                <tr>
                                                                    <td class="mono">{disp}</td>
                                                                    <td class="mono">{a}</td>
                                                                    <td>{format!("{:.3}", s)}</td>
                                                                    <td>{format!("{:.1}%", p * 100.0)}</td>
                                                                </tr>
                                                            }
                                                        })
                                                        .collect_view()
                                                }}
                                            </tbody>
                                        </table>
                                        <p class="subtle">"Note: ‘Temp’ here is a visualization knob (softmax over scores), not Braine’s learning temperature."</p>
                                    </div>
                                </Show>
                            </div>
                        </Show>

                        <Show when=move || dashboard_tab.get() == DashboardTab::Stats>
                            <div class="stack tight">
                                <div class="card">
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
                                            {move || format!("{:.1}%", recent_rate.get() * 100.0)}
                                        </span>
                                    </div>
                                    <div class="divider"></div>
                                    <div class="stat-row">
                                        <span class="stat-label">"Last Action"</span>
                                        <span class="stat-value accent value-strong">{move || { let a = last_action.get(); if a.is_empty() { "—".to_string() } else { a.to_uppercase() } }}</span>
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
                                            {move || format!("{:+.2}", last_reward.get())}
                                        </span>
                                    </div>
                                </div>

                                <div class="card">
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
                                        <span class="stat-value">{move || format!("{:.4}", diag.get().avg_weight)}</span>
                                    </div>
                                    <div class="stat-row">
                                        <span class="stat-label">"Avg Amplitude"</span>
                                        <span class="stat-value">{move || format!("{:.4}", diag.get().avg_amp)}</span>
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

                                <div class="card">
                                    <h3 class="card-title">"💾 Persistence"</h3>
                                    <div class="stack tight">
                                        <button class="btn" on:click=move |_| do_save()>"💾 Save (IndexedDB)"</button>
                                        <button class="btn" on:click=move |_| do_load()>"📂 Load (IndexedDB)"</button>
                                        <button class="btn" on:click=move |_| do_migrate_idb_format()>"🔁 Migrate stored format"</button>
                                        <button class="btn" on:click=move |_| do_export_bbi()>"📥 Export .bbi"</button>
                                        <button class="btn" on:click=move |_| do_import_bbi_click()>"📤 Import .bbi"</button>
                                    </div>
                                    <label class="checkbox-row">
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
                                    <label class="checkbox-row">
                                        <input
                                            type="checkbox"
                                            prop:checked=move || idb_autosave.get()
                                            on:change=move |ev| {
                                                let v = event_target_checked(&ev);
                                                set_idb_autosave.set(v);
                                            }
                                        />
                                        <span>"Auto-save brain (every ~5s when changed)"</span>
                                    </label>
                                    <div class="subtle">
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

                                <div style="padding: 10px; background: rgba(0,0,0,0.2); border: 1px solid var(--border); border-radius: 6px; text-align: center;">
                                    <span style="font-size: 0.75rem; color: var(--muted);">{move || gpu_status.get()}</span>
                                </div>
                            </div>
                        </Show>

                        <Show when=move || dashboard_tab.get() == DashboardTab::Analytics>
                            <div class="stack">
                                <div class="subtabs">
                                    {AnalyticsPanel::all().iter().map(|p| {
                                        let pp = *p;
                                        view! {
                                            <button
                                                class=move || if analytics_panel.get() == pp { "subtab active" } else { "subtab" }
                                                on:click=move |_| set_analytics_panel.set(pp)
                                            >
                                                {pp.label()}
                                            </button>
                                        }
                                    }).collect_view()}
                                </div>

                                <Show when=move || analytics_panel.get() == AnalyticsPanel::Performance>
                                    <div class="card">
                                        <h3 class="card-title">"📈 Performance History"</h3>
                                        <p class="subtle">"Rolling accuracy over last 200 trials"</p>
                                        <canvas node_ref=perf_chart_ref width="500" height="120" class="canvas"></canvas>
                                    </div>

                                    <div class="card">
                                        <h3 class="card-title">"🎯 Current Accuracy"</h3>
                                        <div class="row">
                                            <canvas node_ref=gauge_ref width="100" height="100" class="canvas square"></canvas>
                                            <div>
                                                <div class=move || {
                                                    let r = recent_rate.get();
                                                    if r >= 0.85 {
                                                        "good value-strong value-xxl"
                                                    } else if r >= 0.70 {
                                                        "warn value-strong value-xxl"
                                                    } else {
                                                        "accent value-strong value-xxl"
                                                    }
                                                }>
                                                    {move || format!("{:.0}%", recent_rate.get() * 100.0)}
                                                </div>
                                                <div class="subtle">"last 100 trials"</div>
                                            </div>
                                        </div>
                                    </div>

                                    <div class=move || format!("callout tone-{}", learning_milestone_tone.get())>
                                        <p><strong>{move || learning_milestone.get()}</strong></p>
                                        <p>
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

                                    // Learning milestones explanation
                                    <div style="margin-top: 12px; padding: 12px; background: rgba(10,15,26,0.5); border: 1px solid var(--border); border-radius: 8px; font-size: 0.8rem;">
                                        <div style="font-weight: 600; color: var(--accent); margin-bottom: 8px;">"📊 Learning Milestones"</div>
                                        <div style="display: grid; grid-template-columns: auto 1fr; gap: 6px 12px; color: var(--text);">
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

                                    // Game-specific info
                                    <div style="margin-top: 12px; padding: 12px; background: rgba(10,15,26,0.5); border: 1px solid var(--border); border-radius: 8px; font-size: 0.8rem;">
                                        <div style="font-weight: 600; color: var(--accent); margin-bottom: 8px;">"🎮 Game Information"</div>
                                        {move || {
                                            let kind = game_kind.get();
                                            match kind {
                                                GameKind::Spot => view! {
                                                    <div style="color: var(--text); line-height: 1.6;">
                                                        <div style="margin-bottom: 6px;"><strong>"Spot"</strong>" — Simple left/right discrimination"</div>
                                                        <ul style="margin: 0; padding-left: 16px; color: var(--muted);">
                                                            <li>"Stimulus: 'spot_left' or 'spot_right'"</li>
                                                            <li>"Goal: Match action to stimulus (left→left, right→right)"</li>
                                                            <li>"Chance: 50% — Random guessing baseline"</li>
                                                            <li>"Expected to learn: 20-50 trials"</li>
                                                        </ul>
                                                    </div>
                                                }.into_any(),
                                                GameKind::SpotReversal => view! {
                                                    <div style="color: var(--text); line-height: 1.6;">
                                                        <div style="margin-bottom: 6px;"><strong>"Spot Reversal"</strong>" — Adaptation test"</div>
                                                        <ul style="margin: 0; padding-left: 16px; color: var(--muted);">
                                                            <li>"Stimulus flips periodically (left↔right mapping inverts)"</li>
                                                            <li>"Goal: Quickly adapt to rule changes"</li>
                                                            <li>"Tests: Cognitive flexibility and unlearning"</li>
                                                            <li>"Flip interval configurable via settings"</li>
                                                        </ul>
                                                    </div>
                                                }.into_any(),
                                                GameKind::Bandit => view! {
                                                    <div style="color: var(--text); line-height: 1.6;">
                                                        <div style="margin-bottom: 6px;"><strong>"Bandit"</strong>" — Multi-armed bandit exploration"</div>
                                                        <ul style="margin: 0; padding-left: 16px; color: var(--muted);">
                                                            <li>"Multiple arms with different reward probabilities"</li>
                                                            <li>"Goal: Discover and exploit the best arm"</li>
                                                            <li>"Explore vs exploit trade-off (ε controls exploration)"</li>
                                                            <li>"Expected: Gradual convergence to optimal arm"</li>
                                                        </ul>
                                                    </div>
                                                }.into_any(),
                                                GameKind::SpotXY => view! {
                                                    <div style="color: var(--text); line-height: 1.6;">
                                                        <div style="margin-bottom: 6px;"><strong>"SpotXY"</strong>" — Spatial localization"</div>
                                                        <ul style="margin: 0; padding-left: 16px; color: var(--muted);">
                                                            <li>"Target appears in one of N×N grid cells"</li>
                                                            <li>"BinaryX mode: left/right only (N=1)"</li>
                                                            <li>"Grid mode: select specific cell (N≥2)"</li>
                                                            <li>"Chance: 50% (BinaryX) or 1/N² (Grid)"</li>
                                                            <li>"Tests: Spatial encoding and multi-action learning"</li>
                                                        </ul>
                                                    </div>
                                                }.into_any(),
                                                GameKind::Pong => view! {
                                                    <div style="color: var(--text); line-height: 1.6;">
                                                        <div style="margin-bottom: 6px;"><strong>"Pong"</strong>" — Continuous tracking"</div>
                                                        <ul style="margin: 0; padding-left: 16px; color: var(--muted);">
                                                            <li>"Track ball and move paddle to intercept"</li>
                                                            <li>"Actions: up, down, stay"</li>
                                                            <li>"Reward: +1 hit, -1 miss"</li>
                                                            <li>"Tests: Temporal prediction and motor control"</li>
                                                            <li>"Harder than discrete games — requires coordination"</li>
                                                        </ul>
                                                    </div>
                                                }.into_any(),
                                                GameKind::Sequence => view! {
                                                    <div style="color: var(--text); line-height: 1.6;">
                                                        <div style="margin-bottom: 6px;"><strong>"Sequence"</strong>" — Pattern prediction"</div>
                                                        <ul style="margin: 0; padding-left: 16px; color: var(--muted);">
                                                            <li>"Predict next token in repeating sequence"</li>
                                                            <li>"Tests: Temporal memory and pattern recognition"</li>
                                                            <li>"Shift: Pattern can flip to test relearning"</li>
                                                            <li>"Chance: 1/vocab_size"</li>
                                                        </ul>
                                                    </div>
                                                }.into_any(),
                                                GameKind::Text => view! {
                                                    <div style="color: var(--text); line-height: 1.6;">
                                                        <div style="margin-bottom: 6px;"><strong>"Text"</strong>" — Character prediction"</div>
                                                        <ul style="margin: 0; padding-left: 16px; color: var(--muted);">
                                                            <li>"Predict next character in text corpus"</li>
                                                            <li>"Dual corpus with periodic switching"</li>
                                                            <li>"Tests: Language-like sequence learning"</li>
                                                            <li>"Harder: Large action space (vocab size)"</li>
                                                        </ul>
                                                    </div>
                                                }.into_any(),
                                                GameKind::Replay => view! {
                                                    <div style="color: var(--text); line-height: 1.6;">
                                                        <div style="margin-bottom: 6px;"><strong>"Replay"</strong>" — Dataset-driven trials"</div>
                                                        <ul style="margin: 0; padding-left: 16px; color: var(--muted);">
                                                            <li>"Stimuli and correct actions come from a JSON dataset"</li>
                                                            <li>"Goal: Learn the dataset mapping online"</li>
                                                            <li>"Ideal for deterministic evaluation + LLM boundary tests"</li>
                                                            <li>
                                                                {move || {
                                                                    replay_state
                                                                        .get()
                                                                        .map(|s| {
                                                                            format!(
                                                                                "Dataset: {} (trial {} / {}, id={})",
                                                                                s.dataset,
                                                                                s.index,
                                                                                s.total
                                                                                ,
                                                                                s.trial_id
                                                                            )
                                                                        })
                                                                        .unwrap_or_else(|| "Dataset: (not running)".to_string())
                                                                }}
                                                            </li>
                                                        </ul>
                                                    </div>
                                                }.into_any(),
                                            }
                                        }}
                                    </div>
                                </Show>

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

                                <Show when=move || analytics_panel.get() == AnalyticsPanel::BrainViz>
                                    <div class="card">
                                        <h3 class="card-title">"🧠 Braine Visualization"</h3>
                                        <p class="subtle">{move || if brainviz_view_mode.get() == "causal" { "Causal view: symbol-to-symbol temporal edges. Node size = frequency, edge color = causal strength." } else { "Substrate view: sampled unit nodes; edges show sparse connection weights." }}</p>
                                        <div class="callout">
                                            <p>"Drag to rotate • Shift+drag to pan • Scroll to zoom • Hover for details"</p>
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
                                                    }>"-50"</button>
                                                    <button class="btn sm" style="padding: 2px 6px; font-size: 10px;" on:click=move |_| {
                                                        set_brainviz_node_sample.update(|v| *v = (*v).saturating_sub(10).max(16));
                                                    }>"-10"</button>
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
                                                    }>"+10"</button>
                                                    <button class="btn sm" style="padding: 2px 6px; font-size: 10px;" on:click=move |_| {
                                                        set_brainviz_node_sample.update(|v| *v = (*v + 50).min(1024));
                                                    }>"+50"</button>
                                                </div>
                                            </div>
                                            <div class="label" style="display: flex; flex-direction: column; gap: 2px;">
                                                <span>"Edges/node"</span>
                                                <div style="display: flex; gap: 2px; align-items: center;">
                                                    <button class="btn sm" style="padding: 2px 6px; font-size: 10px;" on:click=move |_| {
                                                        set_brainviz_edges_per_node.update(|v| *v = (*v).saturating_sub(5).max(1));
                                                    }>"-5"</button>
                                                    <button class="btn sm" style="padding: 2px 6px; font-size: 10px;" on:click=move |_| {
                                                        set_brainviz_edges_per_node.update(|v| *v = (*v).saturating_sub(1).max(1));
                                                    }>"-1"</button>
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
                                                    }>"+1"</button>
                                                    <button class="btn sm" style="padding: 2px 6px; font-size: 10px;" on:click=move |_| {
                                                        set_brainviz_edges_per_node.update(|v| *v = (*v + 5).min(32));
                                                    }>"+5"</button>
                                                </div>
                                            </div>
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

                                        <div style="position: relative;">
                                            <canvas
                                                node_ref=brain_viz_ref
                                                width="900"
                                                height="520"
                                                class="canvas brainviz"
                                                style="touch-action: none;"
                                                on:wheel=move |ev| {
                                                    ev.prevent_default();
                                                    let dy = ev.delta_y() as f32;
                                                    let factor = (1.0 + (-dy * 0.001)).clamp(0.85, 1.18);
                                                    set_brainviz_zoom.update(|z| {
                                                        *z = (*z * factor).clamp(0.5, 4.0);
                                                    });
                                                }
                                                on:mousedown=move |ev| {
                                                    let Some(canvas) = brain_viz_ref.get() else { return; };
                                                    let rect = canvas.get_bounding_client_rect();
                                                    let css_x = (ev.client_x() as f64) - rect.left();
                                                    let css_y = (ev.client_y() as f64) - rect.top();
                                                    brainviz_dragging.set_value(true);
                                                    brainviz_last_drag_xy.set_value((css_x, css_y));
                                                }
                                                on:mouseup=move |_| {
                                                    brainviz_dragging.set_value(false);
                                                }
                                                on:mouseleave=move |_| {
                                                    brainviz_dragging.set_value(false);
                                                    set_brainviz_hover.set(None);
                                                }
                                                on:mousemove=move |ev| {
                                                    let Some(canvas) = brain_viz_ref.get() else { return; };
                                                    let rect = canvas.get_bounding_client_rect();
                                                    let css_x = (ev.client_x() as f64) - rect.left();
                                                    let css_y = (ev.client_y() as f64) - rect.top();

                                                    let rw = rect.width().max(1.0);
                                                    let rh = rect.height().max(1.0);
                                                    if css_x < 0.0 || css_y < 0.0 || css_x > rw || css_y > rh {
                                                        set_brainviz_hover.set(None);
                                                        return;
                                                    }

                                                    let sx = (canvas.width() as f64) / rw;
                                                    let sy = (canvas.height() as f64) / rh;
                                                    let x = css_x * sx;
                                                    let y = css_y * sy;

                                                    if brainviz_dragging.get_value() {
                                                        let (lx, ly) = brainviz_last_drag_xy.get_value();
                                                        let dx = (css_x - lx) * sx;
                                                        let dy = (css_y - ly) * sy;

                                                        // Shift+drag = pan, regular drag = rotate (both axes)
                                                        if ev.shift_key() {
                                                            set_brainviz_pan_x.update(|v| *v += dx as f32);
                                                            set_brainviz_pan_y.update(|v| *v += dy as f32);
                                                        } else {
                                                            // Horizontal drag = Y-axis rotation
                                                            set_brainviz_manual_rotation.update(|v| *v += (dx as f32) * 0.01);
                                                            // Vertical drag = X-axis rotation
                                                            set_brainviz_rotation_x.update(|v| *v += (dy as f32) * 0.01);
                                                        }
                                                        brainviz_last_drag_xy.set_value((css_x, css_y));
                                                        return;
                                                    }

                                                    let mut best: Option<(u32, f64, f64)> = None; // (id, css_x, css_y)
                                                    brainviz_hit_nodes.with_value(|hits| {
                                                        let mut best_d2: f64 = f64::INFINITY;
                                                        for hn in hits {
                                                            let dx = hn.x - x;
                                                            let dy = hn.y - y;
                                                            let d2 = dx * dx + dy * dy;
                                                            let r = hn.r + 4.0;
                                                            if d2 <= r * r && d2 < best_d2 {
                                                                best_d2 = d2;
                                                                best = Some((
                                                                    hn.id,
                                                                    hn.x / sx,
                                                                    hn.y / sy,
                                                                ));
                                                            }
                                                        }
                                                    });

                                                    set_brainviz_hover.set(best);
                                                }
                                            ></canvas>

                                            <Show when=move || brainviz_hover.get().is_some()>
                                                <div style=move || {
                                                    let Some((_id, x, y)) = brainviz_hover.get() else { return "display: none;".to_string(); };
                                                    format!(
                                                        "position: absolute; left: {:.0}px; top: {:.0}px; transform: translate(10px, -10px); padding: 8px 10px; background: rgba(10,15,26,0.92); border: 1px solid rgba(122,162,255,0.25); border-radius: 10px; font-size: 12px; color: rgba(232,236,255,0.95); pointer-events: none; max-width: 260px;",
                                                        x,
                                                        y
                                                    )
                                                }>
                                                    {move || {
                                                        let Some((id, _x, _y)) = brainviz_hover.get() else { return "".to_string(); };
                                                        let p = brainviz_points
                                                            .get()
                                                            .into_iter()
                                                            .find(|p| p.id == id);
                                                        if let Some(p) = p {
                                                            let kind = if p.is_sensor_member {
                                                                "sensor"
                                                            } else if p.is_group_member {
                                                                "group"
                                                            } else if p.is_reserved {
                                                                "reserved"
                                                            } else {
                                                                "unit"
                                                            };
                                                            format!("id={}  kind={}  amp01={:.2}  age={:.2}", p.id, kind, p.amp01, p.rel_age)
                                                        } else {
                                                            format!("id={}", id)
                                                        }
                                                    }}
                                                </div>
                                            </Show>
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
                                </Show>
                            </div>
                        </Show>

                        <Show when=move || dashboard_tab.get() == DashboardTab::Settings>
                            <div class="stack">
                                <div class="card">
                                    <h3 class="card-title">"⚙️ Braine Settings"</h3>
                                    <p class="subtle">"Adjust substrate size and how scalar reward is delivered."</p>
                                </div>

                                <div class="card">
                                    <h3 class="card-title">"🧬 Neurogenesis"</h3>
                                    <div class="row end">
                                        <label class="label">
                                            <span>"Grow units by"</span>
                                            <input
                                                class="input compact"
                                                type="number"
                                                title="Adds N new units to the substrate (neurogenesis). This increases capacity without gradients/backprop."
                                                min="1"
                                                step="1"
                                                prop:value=move || grow_units_n.get().to_string()
                                                on:input=move |ev| {
                                                    if let Ok(v) = event_target_value(&ev).parse::<u32>() {
                                                        set_grow_units_n.set(v.max(1));
                                                    }
                                                }
                                            />
                                        </label>
                                        <button class="btn primary" on:click=move |_| do_grow_units()>
                                            "➕ Grow"
                                        </button>
                                    </div>
                                    <p class="subtle">"Adds new units to the substrate without gradients/backprop."</p>
                                </div>

                                <div class="card">
                                    <h3 class="card-title">"🧠 Brain Config (Live)"</h3>
                                    <p class="subtle">"Tunes continuous dynamics/learning parameters. Topology (unit count/connectivity) is not changed here."</p>

                                    <div class="row end wrap">
                                        <label class="label">
                                            <span>"dt"</span>
                                            <input
                                                class="input compact"
                                                type="number"
                                                title="Integration timestep for the continuous dynamics. Smaller dt = more stable/slow updates; larger dt = faster but can destabilize."
                                                min="0.001"
                                                max="1"
                                                step="0.001"
                                                prop:value=move || format!("{:.3}", cfg_dt.get())
                                                on:input=move |ev| {
                                                    if let Ok(v) = event_target_value(&ev).parse::<f32>() {
                                                        set_cfg_dt.set(v.clamp(0.001, 1.0));
                                                    }
                                                }
                                            />
                                        </label>
                                        <label class="label">
                                            <span>"base_freq"</span>
                                            <input
                                                class="input compact"
                                                type="number"
                                                title="Base oscillator frequency for unit dynamics (sets the intrinsic rhythm)."
                                                min="0"
                                                max="10"
                                                step="0.05"
                                                prop:value=move || format!("{:.2}", cfg_base_freq.get())
                                                on:input=move |ev| {
                                                    if let Ok(v) = event_target_value(&ev).parse::<f32>() {
                                                        set_cfg_base_freq.set(v.clamp(0.0, 10.0));
                                                    }
                                                }
                                            />
                                        </label>
                                        <label class="label">
                                            <span>"global_inhibition"</span>
                                            <input
                                                class="input compact"
                                                type="number"
                                                title="Global inhibition strength. Higher values suppress overall activity and can increase selectivity."
                                                min="0"
                                                max="5"
                                                step="0.01"
                                                prop:value=move || format!("{:.2}", cfg_global_inhibition.get())
                                                on:input=move |ev| {
                                                    if let Ok(v) = event_target_value(&ev).parse::<f32>() {
                                                        set_cfg_global_inhibition.set(v.clamp(0.0, 5.0));
                                                    }
                                                }
                                            />
                                        </label>
                                    </div>

                                    <div class="row end wrap">
                                        <label class="label">
                                            <span>"noise_amp"</span>
                                            <input
                                                class="input compact"
                                                type="number"
                                                title="Amplitude of injected noise into dynamics. Useful for exploration; too high can destabilize."
                                                min="0"
                                                max="1"
                                                step="0.001"
                                                prop:value=move || format!("{:.3}", cfg_noise_amp.get())
                                                on:input=move |ev| {
                                                    if let Ok(v) = event_target_value(&ev).parse::<f32>() {
                                                        set_cfg_noise_amp.set(v.clamp(0.0, 1.0));
                                                    }
                                                }
                                            />
                                        </label>
                                        <label class="label">
                                            <span>"noise_phase"</span>
                                            <input
                                                class="input compact"
                                                type="number"
                                                title="Phase noise / jitter applied to oscillators. Adds variability without changing amplitude."
                                                min="0"
                                                max="1"
                                                step="0.001"
                                                prop:value=move || format!("{:.3}", cfg_noise_phase.get())
                                                on:input=move |ev| {
                                                    if let Ok(v) = event_target_value(&ev).parse::<f32>() {
                                                        set_cfg_noise_phase.set(v.clamp(0.0, 1.0));
                                                    }
                                                }
                                            />
                                        </label>
                                        <label class="label">
                                            <span>"causal_decay"</span>
                                            <input
                                                class="input compact"
                                                type="number"
                                                title="Decay rate for causal memory edges. Higher values forget causality faster."
                                                min="0"
                                                max="1"
                                                step="0.001"
                                                prop:value=move || format!("{:.3}", cfg_causal_decay.get())
                                                on:input=move |ev| {
                                                    if let Ok(v) = event_target_value(&ev).parse::<f32>() {
                                                        set_cfg_causal_decay.set(v.clamp(0.0, 1.0));
                                                    }
                                                }
                                            />
                                        </label>
                                    </div>

                                    <div class="row end wrap">
                                        <label class="label">
                                            <span>"hebb_rate"</span>
                                            <input
                                                class="input compact"
                                                type="number"
                                                title="Hebbian learning rate (local plasticity strength). Higher = faster coupling updates."
                                                min="0"
                                                max="1"
                                                step="0.01"
                                                prop:value=move || format!("{:.2}", cfg_hebb_rate.get())
                                                on:input=move |ev| {
                                                    if let Ok(v) = event_target_value(&ev).parse::<f32>() {
                                                        set_cfg_hebb_rate.set(v.clamp(0.0, 1.0));
                                                    }
                                                }
                                            />
                                        </label>
                                        <label class="label">
                                            <span>"forget_rate"</span>
                                            <input
                                                class="input compact"
                                                type="number"
                                                title="Forgetting rate applied to learned couplings. Higher values erase weights faster."
                                                min="0"
                                                max="1"
                                                step="0.001"
                                                prop:value=move || format!("{:.4}", cfg_forget_rate.get())
                                                on:input=move |ev| {
                                                    if let Ok(v) = event_target_value(&ev).parse::<f32>() {
                                                        set_cfg_forget_rate.set(v.clamp(0.0, 1.0));
                                                    }
                                                }
                                            />
                                        </label>
                                        <label class="label">
                                            <span>"prune_below"</span>
                                            <input
                                                class="input compact"
                                                type="number"
                                                title="Pruning threshold: connections with |w| below this may be pruned during maintenance."
                                                min="0"
                                                max="1"
                                                step="0.001"
                                                prop:value=move || format!("{:.3}", cfg_prune_below.get())
                                                on:input=move |ev| {
                                                    if let Ok(v) = event_target_value(&ev).parse::<f32>() {
                                                        set_cfg_prune_below.set(v.clamp(0.0, 1.0));
                                                    }
                                                }
                                            />
                                        </label>
                                    </div>

                                    <div class="row end wrap">
                                        <label class="label">
                                            <span>"coactive_threshold"</span>
                                            <input
                                                class="input compact"
                                                type="number"
                                                title="Co-activity threshold used for forming associations (how strongly two units must co-activate to be considered linked)."
                                                min="0"
                                                max="1"
                                                step="0.01"
                                                prop:value=move || format!("{:.2}", cfg_coactive_threshold.get())
                                                on:input=move |ev| {
                                                    if let Ok(v) = event_target_value(&ev).parse::<f32>() {
                                                        set_cfg_coactive_threshold.set(v.clamp(0.0, 1.0));
                                                    }
                                                }
                                            />
                                        </label>
                                        <label class="label">
                                            <span>"phase_lock_threshold"</span>
                                            <input
                                                class="input compact"
                                                type="number"
                                                title="Threshold for considering oscillators phase-locked (used for binding / stable coupling decisions)."
                                                min="0"
                                                max="1"
                                                step="0.01"
                                                prop:value=move || format!("{:.2}", cfg_phase_lock_threshold.get())
                                                on:input=move |ev| {
                                                    if let Ok(v) = event_target_value(&ev).parse::<f32>() {
                                                        set_cfg_phase_lock_threshold.set(v.clamp(0.0, 1.0));
                                                    }
                                                }
                                            />
                                        </label>
                                        <label class="label">
                                            <span>"imprint_rate"</span>
                                            <input
                                                class="input compact"
                                                type="number"
                                                title="Imprinting rate for creating stable concepts/engrams. Higher = more aggressive one-shot binding."
                                                min="0"
                                                max="1"
                                                step="0.01"
                                                prop:value=move || format!("{:.2}", cfg_imprint_rate.get())
                                                on:input=move |ev| {
                                                    if let Ok(v) = event_target_value(&ev).parse::<f32>() {
                                                        set_cfg_imprint_rate.set(v.clamp(0.0, 1.0));
                                                    }
                                                }
                                            />
                                        </label>
                                    </div>

                                    <div class="row end wrap">
                                        <label class="label">
                                            <span>"salience_decay"</span>
                                            <input
                                                class="input compact"
                                                type="number"
                                                title="Decay rate for salience (how quickly attention/usage fades). Higher = faster fade."
                                                min="0"
                                                max="0.1"
                                                step="0.0001"
                                                prop:value=move || format!("{:.4}", cfg_salience_decay.get())
                                                on:input=move |ev| {
                                                    if let Ok(v) = event_target_value(&ev).parse::<f32>() {
                                                        set_cfg_salience_decay.set(v.clamp(0.0, 0.1));
                                                    }
                                                }
                                            />
                                        </label>
                                        <label class="label">
                                            <span>"salience_gain"</span>
                                            <input
                                                class="input compact"
                                                type="number"
                                                title="Gain applied to salience updates (how quickly frequently used symbols/units become salient)."
                                                min="0"
                                                max="5"
                                                step="0.01"
                                                prop:value=move || format!("{:.2}", cfg_salience_gain.get())
                                                on:input=move |ev| {
                                                    if let Ok(v) = event_target_value(&ev).parse::<f32>() {
                                                        set_cfg_salience_gain.set(v.clamp(0.0, 5.0));
                                                    }
                                                }
                                            />
                                        </label>
                                    </div>

                                    <div class="row end wrap">
                                        <button class="btn" on:click=move |_| reset_brain_config_from_runtime()>
                                            "Reset"
                                        </button>
                                        <button class="btn primary" on:click=move |_| apply_brain_config()>
                                            "Apply"
                                        </button>
                                    </div>
                                </div>
                            </div>
                        </Show>

                        <Show when=move || dashboard_tab.get() == DashboardTab::Learning>
                            <div class="stack">
                                <div class="card">
                                    <h3 class="card-title">"🧪 Learning"</h3>
                                    <p class="subtle">"Controls for learning writes, accelerated learning, and simulation cadence."</p>
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
                                                prop:value=move || format!("{:.2}", reward_scale.get())
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
                                                prop:value=move || format!("{:.2}", reward_bias.get())
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
/// Dashboard tabs for the right panel in split-page layout
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
enum DashboardTab {
    Learning,
    #[default]
    GameDetails,
    Stats,
    Analytics,
    Settings,
}

impl DashboardTab {
    fn label(self) -> &'static str {
        match self {
            DashboardTab::GameDetails => "Game Details",
            DashboardTab::Learning => "Learning",
            DashboardTab::Stats => "Stats",
            DashboardTab::Analytics => "Analytics",
            DashboardTab::Settings => "Settings",
        }
    }
    fn icon(self) -> &'static str {
        match self {
            DashboardTab::GameDetails => "🧩",
            DashboardTab::Learning => "🧠",
            DashboardTab::Stats => "📊",
            DashboardTab::Analytics => "📈",
            DashboardTab::Settings => "⚙️",
        }
    }
    fn all() -> &'static [DashboardTab] {
        &[
            DashboardTab::Learning,
            DashboardTab::GameDetails,
            DashboardTab::Stats,
            DashboardTab::Analytics,
            DashboardTab::Settings,
        ]
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
enum AnalyticsPanel {
    #[default]
    Performance,
    Reward,
    Choices,
    UnitPlot,
    BrainViz,
}

impl AnalyticsPanel {
    fn label(self) -> &'static str {
        match self {
            AnalyticsPanel::Performance => "Performance",
            AnalyticsPanel::Reward => "Reward",
            AnalyticsPanel::Choices => "Choices",
            AnalyticsPanel::UnitPlot => "Unit Plot",
            AnalyticsPanel::BrainViz => "Braine Viz",
        }
    }
    fn all() -> &'static [AnalyticsPanel] {
        &[
            AnalyticsPanel::Performance,
            AnalyticsPanel::Reward,
            AnalyticsPanel::Choices,
            AnalyticsPanel::UnitPlot,
            AnalyticsPanel::BrainViz,
        ]
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum GameKind {
    Spot,
    Bandit,
    SpotReversal,
    SpotXY,
    Pong,
    Sequence,
    Text,
    Replay,
}

impl GameKind {
    fn label(self) -> &'static str {
        match self {
            GameKind::Spot => "spot",
            GameKind::Bandit => "bandit",
            GameKind::SpotReversal => "spot_reversal",
            GameKind::SpotXY => "spotxy",
            GameKind::Pong => "pong",
            GameKind::Sequence => "sequence",
            GameKind::Text => "text",
            GameKind::Replay => "replay",
        }
    }

    fn display_name(self) -> &'static str {
        match self {
            GameKind::Spot => "Spot",
            GameKind::Bandit => "Bandit",
            GameKind::SpotReversal => "Reversal",
            GameKind::SpotXY => "SpotXY",
            GameKind::Pong => "Pong",
            GameKind::Sequence => "Sequence",
            GameKind::Text => "Text",
            GameKind::Replay => "Replay",
        }
    }

    fn icon(self) -> &'static str {
        match self {
            GameKind::Spot => "🎯",
            GameKind::Bandit => "🎰",
            GameKind::SpotReversal => "🔄",
            GameKind::SpotXY => "📍",
            GameKind::Pong => "🏓",
            GameKind::Sequence => "🔢",
            GameKind::Text => "📝",
            GameKind::Replay => "📼",
        }
    }

    fn description(self) -> &'static str {
        match self {
            GameKind::Spot => "Binary discrimination with two stimuli (spot_left/spot_right) and two actions (left/right). One response per trial; reward is +1 for correct, −1 for wrong.",
            GameKind::Bandit => "Two-armed bandit with a constant context stimulus (bandit). Choose left/right once per trial; reward is stochastic with prob_left=0.8 and prob_right=0.2.",
            GameKind::SpotReversal => "Like Spot, but the correct mapping flips once after flip_after_trials=200. Tests adaptation to a distributional shift in reward dynamics.",
            GameKind::SpotXY => "Population-coded 2D position. In BinaryX mode, classify sign(x) into left/right. In Grid mode, choose the correct spotxy_cell_{n}_{ix}_{iy} among n² actions (web control doubles: 2×2 → 4×4 → 8×8).",
            GameKind::Pong => "Discrete-sensor Pong: ball/paddle position and velocity are binned into named sensors; actions are up/down/stay. The sim uses continuous collision detection against the arena walls and is deterministic given a fixed seed (randomness only on post-score serve).",
            GameKind::Sequence => "Next-token prediction over a small alphabet {A,B,C} with a regime shift between two fixed patterns every 60 outcomes.",
            GameKind::Text => "Next-token prediction over a byte vocabulary built from two small corpora (default: 'hello world\\n' vs 'goodbye world\\n') with a regime shift every 80 outcomes.",
            GameKind::Replay => "Dataset-driven replay: each completed trial consumes a record (stimuli + allowed actions + correct action) and emits reward based on correctness. Useful for deterministic evaluation of the advisor boundary.",
        }
    }

    fn what_it_tests(self) -> &'static str {
        match self {
            GameKind::Spot => "• Simple stimulus-action associations\n• Binary classification\n• Basic credit assignment\n• Fastest to learn (~50-100 trials)",
            GameKind::Bandit => "• Exploration vs exploitation trade-off\n• Stochastic reward handling\n• Value estimation under uncertainty\n• Convergence to the better arm (0.8 vs 0.2)",
            GameKind::SpotReversal => "• Behavioral flexibility\n• Rule change detection\n• Context-dependent learning\n• Unlearning old associations\n• Catastrophic forgetting resistance",
            GameKind::SpotXY => "• Multi-class classification (N² classes)\n• Spatial encoding and decoding\n• Scalable representation\n• Train/eval mode separation\n• Generalization testing",
            GameKind::Pong => "• Continuous state representation\n• Real-time motor control\n• Predictive tracking\n• Reward delay handling\n• Sensorimotor coordination",
            GameKind::Sequence => "• Temporal pattern recognition\n• Regime/distribution shifts\n• Sequence prediction over {A,B,C}\n• Phase detection\n• Attractor dynamics",
            GameKind::Text => "• Symbolic next-token prediction (byte tokens)\n• Regime/distribution shifts\n• Online learning without backprop\n• Vocabulary scaling (max_vocab)\n• Credit assignment with scalar reward",
            GameKind::Replay => "• Deterministic evaluation loop\n• Dataset-conditioned correctness reward\n• Context stability (replay::<dataset>)\n• Advisor boundary validation (context → advice)",
        }
    }

    fn inputs_info(self) -> &'static str {
        match self {
            GameKind::Spot => "Stimuli (by name): spot_left or spot_right\nActions: left, right\nTrial timing: controlled by Trial ms",
            GameKind::Bandit => "Stimulus: bandit (constant context)\nActions: left, right\nParameters: prob_left=0.8, prob_right=0.2",
            GameKind::SpotReversal => "Stimuli: spot_left or spot_right (+ reversal context sensor spot_rev_ctx when reversed)\nActions: left, right\nParameter: flip_after_trials=200\nNote: the web runtime also tags the meaning context with ::rev",
            GameKind::SpotXY => "Base stimulus: spotxy (context)\nSensors: pos_x_00..pos_x_15 and pos_y_00..pos_y_15 (population code)\nStimulus key: spotxy_xbin_XX or spotxy_bin_NN_IX_IY\nActions: left/right (BinaryX) OR spotxy_cell_NN_IX_IY (Grid)\nEval mode: holdout band |x| in [0.25..0.45] with learning suppressed",
            GameKind::Pong => "Base stimulus: pong (context)\nSensors: pong_ball_x_00..07, pong_ball_y_00..07, pong_paddle_y_00..07, pong_ball_visible/hidden, pong_vx_pos/neg, pong_vy_pos/neg\nOptional distractor: pong_ball2_x_00..07, pong_ball2_y_00..07, pong_ball2_visible/hidden, pong_ball2_vx_pos/neg, pong_ball2_vy_pos/neg\nStimulus key: pong_b08_vis.._bx.._by.._py.._vx.._vy.. (+ ball2 fields when enabled)\nActions: up, down, stay",
            GameKind::Sequence => "Base stimulus: sequence (context)\nSensors: seq_token_A/B/C and seq_regime_0/1\nStimulus key: seq_r{0|1}_t{A|B|C}\nActions: A, B, C",
            GameKind::Text => "Base stimulus: text (context)\nSensors: txt_regime_0/1 and txt_tok_XX (byte tokens) + txt_tok_UNK\nActions: tok_XX for bytes in vocab + tok_UNK",
            GameKind::Replay => "Stimuli/actions: defined per-trial by the dataset\nStimulus key: replay::<dataset_name>\nReward: +1 on correct_action, −1 otherwise",
        }
    }

    fn reward_info(self) -> &'static str {
        match self {
            GameKind::Spot => "+1.0: Correct response (stimulus matches action)\n−1.0: Incorrect response",
            GameKind::Bandit => "+1.0: Bernoulli reward (win)\n−1.0: No win\nProbabilities: left=0.8, right=0.2 (default)",
            GameKind::SpotReversal => "+1.0: Correct under current mapping\n−1.0: Incorrect\nFlip: once after flip_after_trials=200",
            GameKind::SpotXY => "+1.0: Correct classification\n−1.0: Incorrect\nEval mode: runs dynamics and action selection, but suppresses learning writes",
            GameKind::Pong => "+0.05: Action matches a simple tracking heuristic\n−0.05: Action mismatches heuristic\nEvent reward: +1 on paddle hit, −1 on miss (when the ball reaches the left boundary at x=0)\nAll rewards are clamped to [−1, +1]",
            GameKind::Sequence => "+1.0: Correct next-token prediction\n−1.0: Incorrect\nRegime flips every shift_every_outcomes=60",
            GameKind::Text => "+1.0: Correct next-token prediction\n−1.0: Incorrect\nRegime flips every shift_every_outcomes=80",
            GameKind::Replay => "+1.0: Action matches correct_action\n−1.0: Otherwise\nNotes: no stochasticity unless your dataset includes it",
        }
    }

    fn learning_objectives(self) -> &'static str {
        match self {
            GameKind::Spot => "• Achieve >90% accuracy consistently\n• Learn in <100 trials\n• Demonstrate stable attractor formation",
            GameKind::Bandit => "• Converge to preferred arm selection\n• Maintain ~70% reward rate at optimum\n• Balance exploration early, exploitation late",
            GameKind::SpotReversal => "• Recover accuracy after reversal within ~20 trials\n• Use context bit to accelerate switching\n• Maintain two stable modes",
            GameKind::SpotXY => "• Scale to larger grids (3×3, 4×4, 5×5+)\n• Maintain accuracy in Eval mode\n• Demonstrate spatial generalization",
            GameKind::Pong => "• Track ball trajectory predictively\n• Minimize missed balls over time\n• Develop smooth control policy",
            GameKind::Sequence => "• Predict sequences of length 3-6+\n• Recognize phase within sequence\n• Handle pattern length changes",
            GameKind::Text => "• Build character transition model\n• Adapt to regime shifts\n• Predict based on statistical regularities",
            GameKind::Replay => "• Achieve high accuracy on dataset\n• Use stable context to generalize across repeated trials\n• Validate advisor integration without action selection",
        }
    }

    fn all() -> &'static [GameKind] {
        &[
            GameKind::Spot,
            GameKind::Bandit,
            GameKind::SpotReversal,
            GameKind::SpotXY,
            GameKind::Pong,
            GameKind::Sequence,
            GameKind::Text,
            GameKind::Replay,
        ]
    }
}

struct TickConfig {
    trial_period_ms: u32,
    exploration_eps: f32,
    meaning_alpha: f32,
    reward_scale: f32,
    reward_bias: f32,
    learning_enabled: bool,
}

struct TickOutput {
    last_action: String,
    reward: f32,
}

struct AppRuntime {
    brain: Brain,
    game: WebGame,
    pending_neuromod: f32,
    rng_seed: u64,
}

impl AppRuntime {
    fn new() -> Self {
        Self {
            brain: make_default_brain(),
            game: WebGame::Spot(SpotGame::new()),
            pending_neuromod: 0.0,
            rng_seed: 0xC0FF_EE12u64,
        }
    }

    fn set_game(&mut self, kind: GameKind) {
        self.game = match kind {
            GameKind::Spot => WebGame::Spot(SpotGame::new()),
            GameKind::Bandit => WebGame::Bandit(BanditGame::new()),
            GameKind::SpotReversal => WebGame::SpotReversal(SpotReversalGame::new(200)),
            GameKind::SpotXY => {
                self.ensure_spotxy_io(16);
                let g = SpotXYGame::new(16);
                self.ensure_spotxy_actions(&g);
                WebGame::SpotXY(g)
            }
            GameKind::Pong => {
                self.ensure_pong_io();
                WebGame::Pong(PongWebGame::new(0xB0A7_F00Du64))
            }
            GameKind::Sequence => {
                self.ensure_sequence_io();
                WebGame::Sequence(SequenceWebGame::new())
            }
            GameKind::Text => {
                let g = TextWebGame::new();
                self.ensure_text_io(&g);
                WebGame::Text(g)
            }
            GameKind::Replay => {
                let ds = ReplayDataset::builtin_left_right_spot();
                self.ensure_replay_io(&ds);
                WebGame::Replay(ReplayGame::new(ds))
            }
        };
        self.pending_neuromod = 0.0;
    }

    fn game_ui_snapshot(&self) -> GameUiSnapshot {
        self.game.ui_snapshot()
    }

    fn ensure_spotxy_io(&mut self, k: usize) {
        for i in 0..k {
            self.brain
                .ensure_sensor_min_width(&format!("pos_x_{i:02}"), 3);
            self.brain
                .ensure_sensor_min_width(&format!("pos_y_{i:02}"), 3);
        }
        self.brain.ensure_action_min_width("left", 6);
        self.brain.ensure_action_min_width("right", 6);
    }

    fn ensure_pong_io(&mut self) {
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
        }
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

    fn ensure_sequence_io(&mut self) {
        self.brain.ensure_sensor_min_width("seq_token_A", 4);
        self.brain.ensure_sensor_min_width("seq_token_B", 4);
        self.brain.ensure_sensor_min_width("seq_token_C", 4);

        self.brain.ensure_sensor_min_width("seq_regime_0", 3);
        self.brain.ensure_sensor_min_width("seq_regime_1", 3);

        self.brain.ensure_action_min_width("A", 6);
        self.brain.ensure_action_min_width("B", 6);
        self.brain.ensure_action_min_width("C", 6);
    }

    fn ensure_text_io(&mut self, g: &TextWebGame) {
        self.brain.ensure_sensor_min_width("txt_regime_0", 3);
        self.brain.ensure_sensor_min_width("txt_regime_1", 3);
        for name in g.token_sensor_names() {
            self.brain.ensure_sensor_min_width(name, 3);
        }
        for name in g.allowed_actions() {
            self.brain.ensure_action_min_width(name, 6);
        }
    }

    fn ensure_replay_io(&mut self, dataset: &ReplayDataset) {
        use std::collections::BTreeSet;

        let mut sensors: BTreeSet<String> = BTreeSet::new();
        let mut actions: BTreeSet<String> = BTreeSet::new();

        for tr in &dataset.trials {
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

    fn ensure_spotxy_actions(&mut self, g: &SpotXYGame) {
        self.brain.ensure_action_min_width("left", 6);
        self.brain.ensure_action_min_width("right", 6);
        for name in g.allowed_actions() {
            self.brain.ensure_action_min_width(name, 6);
        }
    }

    fn spotxy_increase_grid(&mut self) {
        let actions = if let WebGame::SpotXY(g) = &mut self.game {
            let cur = g.grid_n();
            let target = if cur == 0 {
                2
            } else if cur.is_power_of_two() {
                (cur.saturating_mul(2)).min(8)
            } else {
                cur.next_power_of_two().min(8)
            };

            // Use the underlying stepwise API, but jump to the next power-of-two size.
            let mut guard = 0u32;
            while g.grid_n() < target && guard < 16 {
                let before = g.grid_n();
                g.increase_grid();
                if g.grid_n() == before {
                    break;
                }
                guard += 1;
            }

            Some(g.allowed_actions().to_vec())
        } else {
            None
        };

        if let Some(actions) = actions {
            self.brain.ensure_action_min_width("left", 6);
            self.brain.ensure_action_min_width("right", 6);
            for name in actions {
                self.brain.ensure_action_min_width(&name, 6);
            }
        }
    }

    fn spotxy_decrease_grid(&mut self) {
        let actions = if let WebGame::SpotXY(g) = &mut self.game {
            let cur = g.grid_n();
            let target = if cur <= 2 {
                0
            } else if cur.is_power_of_two() {
                cur / 2
            } else {
                // Snap down to the closest lower power-of-two.
                1u32 << (31 - cur.leading_zeros())
            };

            let mut guard = 0u32;
            while g.grid_n() > target && guard < 16 {
                let before = g.grid_n();
                g.decrease_grid();
                if g.grid_n() == before {
                    break;
                }
                guard += 1;
            }

            Some(g.allowed_actions().to_vec())
        } else {
            None
        };

        if let Some(actions) = actions {
            self.brain.ensure_action_min_width("left", 6);
            self.brain.ensure_action_min_width("right", 6);
            for name in actions {
                self.brain.ensure_action_min_width(&name, 6);
            }
        }
    }

    fn spotxy_set_eval(&mut self, eval: bool) {
        if let WebGame::SpotXY(g) = &mut self.game {
            g.set_eval_mode(eval);
        }
    }

    fn pong_set_param(&mut self, key: &str, value: f32) -> Result<(), String> {
        match &mut self.game {
            WebGame::Pong(g) => g.set_param(key, value),
            _ => Err("pong_set_param: not in pong".to_string()),
        }
    }

    fn tick(&mut self, cfg: &TickConfig) -> Option<TickOutput> {
        self.game.update_timing(cfg.trial_period_ms);

        // SpotXY eval mode is a holdout run: no causal/meaning writes.
        let allow_learning = cfg.learning_enabled && !self.game.spotxy_eval_mode();

        // Apply last reward as neuromodulation for one step.
        self.brain.set_neuromodulator(self.pending_neuromod);
        self.pending_neuromod = 0.0;

        let base_stimulus = self.game.stimulus_name();
        let stimulus_key_owned: Option<String> = if self.game.reversal_active() {
            Some(format!("{}::rev", base_stimulus))
        } else {
            self.game.stimulus_key().map(|k| k.to_string())
        };
        let context_key_owned = stimulus_key_owned.unwrap_or_else(|| base_stimulus.to_string());
        let context_key = context_key_owned.as_str();

        // Apply stimuli.
        match &self.game {
            WebGame::Spot(_) | WebGame::Bandit(_) | WebGame::SpotReversal(_) => {
                self.brain.apply_stimulus(Stimulus::new(base_stimulus, 1.0));
                if self.game.reversal_active() {
                    self.brain
                        .apply_stimulus(Stimulus::new("spot_rev_ctx", 1.0));
                }
            }
            WebGame::SpotXY(g) => {
                g.apply_stimuli(&mut self.brain);
            }
            WebGame::Pong(g) => {
                g.apply_stimuli(&mut self.brain);
            }
            WebGame::Sequence(g) => {
                g.apply_stimuli(&mut self.brain);
            }
            WebGame::Text(g) => {
                g.apply_stimuli(&mut self.brain);
            }
            WebGame::Replay(g) => {
                g.apply_stimuli(&mut self.brain);
            }
        }

        self.brain.note_compound_symbol(&[context_key]);
        self.brain.step();

        if self.game.response_made() {
            self.brain.set_neuromodulator(0.0);
            if allow_learning {
                self.brain.commit_observation();
            } else {
                self.brain.discard_observation();
            }
            return None;
        }

        let explore = self.rng_next_f32() < cfg.exploration_eps;
        let rand_idx = self.rng_next_u64() as usize;
        let action = match &self.game {
            WebGame::SpotXY(g) => {
                let allowed = g.allowed_actions();
                if allowed.is_empty() {
                    return None;
                }

                if explore {
                    allowed[rand_idx % allowed.len()].to_string()
                } else {
                    let ranked = self
                        .brain
                        .ranked_actions_with_meaning(context_key, cfg.meaning_alpha);
                    ranked
                        .into_iter()
                        .find(|(name, _score)| allowed.iter().any(|a| a == name))
                        .map(|(name, _score)| name)
                        .or_else(|| allowed.first().cloned())
                        .unwrap_or_else(|| "left".to_string())
                }
            }
            WebGame::Pong(g) => {
                let allowed = g.allowed_actions();
                if allowed.is_empty() {
                    return None;
                }

                if explore {
                    allowed[rand_idx % allowed.len()].to_string()
                } else {
                    let ranked = self
                        .brain
                        .ranked_actions_with_meaning(context_key, cfg.meaning_alpha);
                    ranked
                        .into_iter()
                        .find(|(name, _score)| allowed.iter().any(|a| a == name))
                        .map(|(name, _score)| name)
                        .or_else(|| allowed.first().cloned())
                        .unwrap_or_else(|| "stay".to_string())
                }
            }
            WebGame::Sequence(g) => {
                let allowed = g.allowed_actions();
                if allowed.is_empty() {
                    return None;
                }

                if explore {
                    allowed[rand_idx % allowed.len()].to_string()
                } else {
                    let ranked = self
                        .brain
                        .ranked_actions_with_meaning(context_key, cfg.meaning_alpha);
                    ranked
                        .into_iter()
                        .find(|(name, _score)| allowed.iter().any(|a| a == name))
                        .map(|(name, _score)| name)
                        .or_else(|| allowed.first().cloned())
                        .unwrap_or_else(|| "A".to_string())
                }
            }
            WebGame::Text(g) => {
                let allowed = g.allowed_actions();
                if allowed.is_empty() {
                    return None;
                }

                if explore {
                    allowed[rand_idx % allowed.len()].to_string()
                } else {
                    let ranked = self
                        .brain
                        .ranked_actions_with_meaning(context_key, cfg.meaning_alpha);
                    ranked
                        .into_iter()
                        .find(|(name, _score)| allowed.iter().any(|a| a == name))
                        .map(|(name, _score)| name)
                        .or_else(|| allowed.first().cloned())
                        .unwrap_or_else(|| "tok_UNK".to_string())
                }
            }
            WebGame::Replay(g) => {
                let allowed = g.allowed_actions();
                if allowed.is_empty() {
                    return None;
                }

                if explore {
                    allowed[rand_idx % allowed.len()].to_string()
                } else {
                    let ranked = self
                        .brain
                        .ranked_actions_with_meaning(context_key, cfg.meaning_alpha);
                    ranked
                        .into_iter()
                        .find(|(name, _score)| allowed.iter().any(|a| a == name))
                        .map(|(name, _score)| name)
                        .or_else(|| allowed.first().cloned())
                        .unwrap_or_else(|| "stay".to_string())
                }
            }
            _ => {
                let allowed = ["left", "right"];
                if explore {
                    allowed[rand_idx % allowed.len()].to_string()
                } else {
                    let ranked = self
                        .brain
                        .ranked_actions_with_meaning(context_key, cfg.meaning_alpha);
                    ranked
                        .into_iter()
                        .find(|(name, _score)| allowed.iter().any(|a| a == name))
                        .map(|(name, _score)| name)
                        .or_else(|| allowed.first().map(|s| s.to_string()))
                        .unwrap_or_else(|| "left".to_string())
                }
            }
        };

        let (reward, _done) = match self.game.score_action(&action, cfg.trial_period_ms) {
            Some((r, done)) => (r, done),
            None => (0.0, false),
        };

        let shaped_reward = ((reward + cfg.reward_bias) * cfg.reward_scale).clamp(-5.0, 5.0);

        self.brain.note_action(&action);
        self.brain
            .note_compound_symbol(&["pair", context_key, action.as_str()]);

        if allow_learning {
            self.brain.set_neuromodulator(shaped_reward);
            self.brain.reinforce_action(&action, shaped_reward);
            self.pending_neuromod = shaped_reward;
            self.brain.commit_observation();
        } else {
            self.brain.set_neuromodulator(0.0);
            self.pending_neuromod = 0.0;
            self.brain.discard_observation();
        }

        Some(TickOutput {
            last_action: action,
            reward: shaped_reward,
        })
    }

    fn rng_next_u64(&mut self) -> u64 {
        self.rng_seed = self
            .rng_seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1);
        self.rng_seed
    }

    fn rng_next_f32(&mut self) -> f32 {
        let u = (self.rng_next_u64() >> 40) as u32; // 24 bits
        (u as f32) / ((1u32 << 24) as f32)
    }
}

#[allow(clippy::large_enum_variant)]
enum WebGame {
    Spot(SpotGame),
    Bandit(BanditGame),
    SpotReversal(SpotReversalGame),
    SpotXY(SpotXYGame),
    Pong(PongWebGame),
    Sequence(SequenceWebGame),
    Text(TextWebGame),
    Replay(ReplayGame),
}

impl WebGame {
    fn allowed_actions_for_ui(&self) -> Vec<String> {
        match self {
            WebGame::Spot(_) | WebGame::Bandit(_) | WebGame::SpotReversal(_) => {
                vec!["left".to_string(), "right".to_string()]
            }
            WebGame::SpotXY(g) => g.allowed_actions().to_vec(),
            WebGame::Pong(g) => g.allowed_actions().to_vec(),
            WebGame::Sequence(g) => g.allowed_actions().to_vec(),
            WebGame::Text(g) => g.allowed_actions().to_vec(),
            WebGame::Replay(g) => g.allowed_actions().to_vec(),
        }
    }

    fn stimulus_name(&self) -> &'static str {
        match self {
            WebGame::Spot(g) => g.stimulus_name(),
            WebGame::Bandit(g) => g.stimulus_name(),
            WebGame::SpotReversal(g) => g.stimulus_name(),
            WebGame::SpotXY(g) => g.stimulus_name(),
            WebGame::Pong(g) => g.stimulus_name(),
            WebGame::Sequence(g) => g.stimulus_name(),
            WebGame::Text(g) => g.stimulus_name(),
            WebGame::Replay(g) => g.stimulus_name(),
        }
    }

    fn stimulus_key(&self) -> Option<&str> {
        match self {
            WebGame::SpotXY(g) => Some(g.stimulus_key()),
            WebGame::Pong(g) => Some(g.stimulus_key()),
            WebGame::Sequence(g) => Some(g.stimulus_key()),
            WebGame::Text(g) => Some(g.stimulus_key()),
            WebGame::Replay(g) => Some(g.stimulus_key()),
            _ => None,
        }
    }

    fn reversal_active(&self) -> bool {
        match self {
            WebGame::SpotReversal(g) => g.reversal_active,
            _ => false,
        }
    }

    fn spotxy_eval_mode(&self) -> bool {
        match self {
            WebGame::SpotXY(g) => g.eval_mode,
            _ => false,
        }
    }

    fn response_made(&self) -> bool {
        match self {
            WebGame::Spot(g) => g.response_made,
            WebGame::Bandit(g) => g.response_made,
            WebGame::SpotReversal(g) => g.response_made,
            WebGame::SpotXY(g) => g.response_made,
            WebGame::Pong(g) => g.response_made,
            WebGame::Sequence(g) => g.response_made(),
            WebGame::Text(g) => g.response_made(),
            WebGame::Replay(g) => g.response_made,
        }
    }

    fn update_timing(&mut self, trial_period_ms: u32) {
        match self {
            WebGame::Spot(g) => g.update_timing(trial_period_ms),
            WebGame::Bandit(g) => g.update_timing(trial_period_ms),
            WebGame::SpotReversal(g) => g.update_timing(trial_period_ms),
            WebGame::SpotXY(g) => g.update_timing(trial_period_ms),
            WebGame::Pong(g) => g.update_timing(trial_period_ms),
            WebGame::Sequence(g) => g.update_timing(trial_period_ms),
            WebGame::Text(g) => g.update_timing(trial_period_ms),
            WebGame::Replay(g) => g.update_timing(trial_period_ms),
        }
    }

    fn score_action(&mut self, action: &str, trial_period_ms: u32) -> Option<(f32, bool)> {
        match self {
            WebGame::Spot(g) => g.score_action(action),
            WebGame::Bandit(g) => g.score_action(action),
            WebGame::SpotReversal(g) => g.score_action(action),
            WebGame::SpotXY(g) => g.score_action(action),
            WebGame::Pong(g) => {
                let _ = trial_period_ms;
                g.score_action(action)
            }
            WebGame::Sequence(g) => {
                let _ = trial_period_ms;
                g.score_action(action)
            }
            WebGame::Text(g) => {
                let _ = trial_period_ms;
                g.score_action(action)
            }
            WebGame::Replay(g) => {
                let _ = trial_period_ms;
                g.score_action(action)
            }
        }
    }

    fn stats(&self) -> &braine_games::stats::GameStats {
        match self {
            WebGame::Spot(g) => &g.stats,
            WebGame::Bandit(g) => &g.stats,
            WebGame::SpotReversal(g) => &g.stats,
            WebGame::SpotXY(g) => &g.stats,
            WebGame::Pong(g) => &g.stats,
            WebGame::Sequence(g) => &g.game.stats,
            WebGame::Text(g) => &g.game.stats,
            WebGame::Replay(g) => &g.stats,
        }
    }

    fn set_stats(&mut self, stats: braine_games::stats::GameStats) {
        match self {
            WebGame::Spot(g) => g.stats = stats,
            WebGame::Bandit(g) => g.stats = stats,
            WebGame::SpotReversal(g) => g.stats = stats,
            WebGame::SpotXY(g) => g.stats = stats,
            WebGame::Pong(g) => g.stats = stats,
            WebGame::Sequence(g) => g.game.stats = stats,
            WebGame::Text(g) => g.game.stats = stats,
            WebGame::Replay(g) => g.stats = stats,
        }
    }

    fn ui_snapshot(&self) -> GameUiSnapshot {
        match self {
            WebGame::Spot(g) => GameUiSnapshot {
                spot_is_left: Some(g.spot_is_left),
                ..GameUiSnapshot::default()
            },
            WebGame::Bandit(_) => GameUiSnapshot::default(),
            WebGame::SpotReversal(g) => GameUiSnapshot {
                spot_is_left: Some(g.spot_is_left),
                reversal_active: g.reversal_active,
                reversal_flip_after_trials: g.flip_after_trials,
                ..GameUiSnapshot::default()
            },
            WebGame::SpotXY(g) => GameUiSnapshot {
                spotxy_pos: Some((g.pos_x, g.pos_y)),
                spotxy_stimulus_key: g.stimulus_key().to_string(),
                spotxy_eval: g.eval_mode,
                spotxy_mode: g.mode_name().to_string(),
                spotxy_grid_n: g.grid_n(),
                ..GameUiSnapshot::default()
            },
            WebGame::Pong(g) => GameUiSnapshot {
                pong_state: Some(PongUiState {
                    ball_x: g.sim.state.ball_x,
                    ball_y: g.sim.state.ball_y,
                    paddle_y: g.sim.state.paddle_y,
                    paddle_half_height: g.sim.params.paddle_half_height,
                    ball_visible: g.ball_visible(),
                }),
                pong_stimulus_key: g.stimulus_key().to_string(),
                pong_paddle_speed: g.sim.params.paddle_speed,
                pong_paddle_half_height: g.sim.params.paddle_half_height,
                pong_ball_speed: g.sim.params.ball_speed,
                ..GameUiSnapshot::default()
            },
            WebGame::Sequence(g) => GameUiSnapshot {
                sequence_state: Some(SequenceUiState {
                    regime: g.game.regime(),
                    token: g.game.current_token().label().to_string(),
                    target_next: g.game.correct_action().to_string(),
                    outcomes: g.game.outcomes(),
                    shift_every: g.game.shift_every_outcomes(),
                }),
                ..GameUiSnapshot::default()
            },
            WebGame::Text(g) => GameUiSnapshot {
                text_state: Some(TextUiState {
                    regime: g.game.regime(),
                    token: g.game.current_token().display(),
                    target_next: g.game.target_next_token().display(),
                    outcomes: g.game.outcomes(),
                    shift_every: g.game.shift_every_outcomes(),
                    vocab_size: g.game.vocab_size() as u32,
                }),
                ..GameUiSnapshot::default()
            },
            WebGame::Replay(g) => GameUiSnapshot {
                replay_state: Some(ReplayUiState {
                    dataset: g.dataset_name().to_string(),
                    index: g.index() as u32,
                    total: g.total_trials() as u32,
                    trial_id: g.current_trial_id().to_string(),
                }),
                ..GameUiSnapshot::default()
            },
        }
    }
}

#[derive(Default, Clone)]
struct GameUiSnapshot {
    spot_is_left: Option<bool>,

    spotxy_pos: Option<(f32, f32)>,
    spotxy_stimulus_key: String,
    spotxy_eval: bool,
    spotxy_mode: String,
    spotxy_grid_n: u32,

    reversal_active: bool,
    reversal_flip_after_trials: u32,

    pong_state: Option<PongUiState>,
    pong_stimulus_key: String,
    pong_paddle_speed: f32,
    pong_paddle_half_height: f32,
    pong_ball_speed: f32,

    sequence_state: Option<SequenceUiState>,
    text_state: Option<TextUiState>,
    replay_state: Option<ReplayUiState>,
}

#[derive(Clone)]
struct ReplayUiState {
    dataset: String,
    index: u32,
    total: u32,
    trial_id: String,
}

#[derive(Clone)]
struct SequenceUiState {
    regime: u32,
    token: String,
    target_next: String,
    outcomes: u32,
    shift_every: u32,
}

#[derive(Clone)]
struct TextUiState {
    regime: u32,
    token: String,
    target_next: String,
    outcomes: u32,
    shift_every: u32,
    vocab_size: u32,
}

#[derive(Clone, Copy)]
struct PongUiState {
    ball_x: f32,
    ball_y: f32,
    paddle_y: f32,
    paddle_half_height: f32,
    ball_visible: bool,
}

fn make_default_brain() -> Brain {
    let mut brain = Brain::new(BrainConfig {
        seed: Some(2026),
        causal_decay: 0.002,
        ..BrainConfig::default()
    });

    // Actions used by Spot/Bandit.
    brain.define_action("left", 6);
    brain.define_action("right", 6);

    // Context stimuli.
    brain.define_sensor("spot_left", 4);
    brain.define_sensor("spot_right", 4);
    brain.define_sensor("spot_rev_ctx", 2);
    brain.define_sensor("bandit", 4);

    brain
}

fn local_storage() -> Option<web_sys::Storage> {
    web_sys::window().and_then(|w| w.local_storage().ok().flatten())
}

fn local_storage_get_string(key: &str) -> Option<String> {
    local_storage().and_then(|s| s.get_item(key).ok().flatten())
}

fn local_storage_set_string(key: &str, value: &str) {
    if let Some(s) = local_storage() {
        let _ = s.set_item(key, value);
    }
}

fn local_storage_remove(key: &str) {
    if let Some(s) = local_storage() {
        let _ = s.remove_item(key);
    }
}

fn game_stats_storage_key(kind: GameKind) -> String {
    format!("{}{}", LOCALSTORAGE_GAME_STATS_PREFIX, kind.label())
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct PersistedGameStats {
    correct: u32,
    incorrect: u32,
    trials: u32,
    recent: Vec<bool>,
    learning_at_trial: Option<u32>,
    learned_at_trial: Option<u32>,
    mastered_at_trial: Option<u32>,
}

impl From<&braine_games::stats::GameStats> for PersistedGameStats {
    fn from(s: &braine_games::stats::GameStats) -> Self {
        Self {
            correct: s.correct,
            incorrect: s.incorrect,
            trials: s.trials,
            recent: s.recent.clone(),
            learning_at_trial: s.learning_at_trial,
            learned_at_trial: s.learned_at_trial,
            mastered_at_trial: s.mastered_at_trial,
        }
    }
}

impl PersistedGameStats {
    fn into_game_stats(self) -> braine_games::stats::GameStats {
        braine_games::stats::GameStats {
            correct: self.correct,
            incorrect: self.incorrect,
            trials: self.trials,
            recent: self.recent,
            learning_at_trial: self.learning_at_trial,
            learned_at_trial: self.learned_at_trial,
            mastered_at_trial: self.mastered_at_trial,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct PersistedStatsState {
    version: u32,
    game: String,
    stats: PersistedGameStats,
    perf_history: Vec<f32>,
    neuromod_history: Vec<f32>,
    #[serde(default)]
    choice_events: Vec<String>,
    last_action: String,
    last_reward: f32,
}

fn load_persisted_stats_state(kind: GameKind) -> Option<PersistedStatsState> {
    let key = game_stats_storage_key(kind);
    let raw = local_storage_get_string(&key)?;
    serde_json::from_str(&raw).ok()
}

fn save_persisted_stats_state(kind: GameKind, state: &PersistedStatsState) {
    let key = game_stats_storage_key(kind);
    if let Ok(raw) = serde_json::to_string(state) {
        local_storage_set_string(&key, &raw);
    }
}

fn clear_persisted_stats_state(kind: GameKind) {
    let key = game_stats_storage_key(kind);
    local_storage_remove(&key);
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct PersistedSettings {
    reward_scale: f32,
    reward_bias: f32,
    #[serde(default = "default_true")]
    learning_enabled: bool,
    #[serde(default = "default_run_interval_ms")]
    run_interval_ms: u32,
    #[serde(default = "default_trial_period_ms")]
    trial_period_ms: u32,
}

fn default_true() -> bool {
    true
}

fn default_run_interval_ms() -> u32 {
    33
}

fn default_trial_period_ms() -> u32 {
    500
}

fn load_persisted_settings() -> Option<PersistedSettings> {
    let raw = local_storage_get_string(LOCALSTORAGE_SETTINGS_KEY)?;
    serde_json::from_str(&raw).ok()
}

fn save_persisted_settings(settings: &PersistedSettings) {
    if let Ok(raw) = serde_json::to_string(settings) {
        local_storage_set_string(LOCALSTORAGE_SETTINGS_KEY, &raw);
    }
}

fn apply_theme_to_document(theme: Theme) {
    let Some(doc) = web_sys::window().and_then(|w| w.document()) else {
        return;
    };
    let Some(el) = doc.document_element() else {
        return;
    };
    let _ = el.set_attribute("data-theme", theme.as_attr());
}

async fn idb_put_bytes(key: &str, bytes: &[u8]) -> Result<(), String> {
    let db = idb_open().await?;
    let tx = db
        .transaction_with_str_and_mode(IDB_STORE, web_sys::IdbTransactionMode::Readwrite)
        .map_err(|_| "indexeddb: failed to open transaction".to_string())?;
    let store = tx
        .object_store(IDB_STORE)
        .map_err(|_| "indexeddb: failed to open object store".to_string())?;

    let value = js_sys::Uint8Array::from(bytes).into();
    let req = store
        .put_with_key(&value, &JsValue::from_str(key))
        .map_err(|_| "indexeddb: put() threw".to_string())?;
    idb_request_done(req).await?;
    Ok(())
}

async fn idb_get_bytes(key: &str) -> Result<Option<Vec<u8>>, String> {
    let db = idb_open().await?;
    let tx = db
        .transaction_with_str_and_mode(IDB_STORE, web_sys::IdbTransactionMode::Readonly)
        .map_err(|_| "indexeddb: failed to open transaction".to_string())?;
    let store = tx
        .object_store(IDB_STORE)
        .map_err(|_| "indexeddb: failed to open object store".to_string())?;

    let req = store
        .get(&JsValue::from_str(key))
        .map_err(|_| "indexeddb: get() threw".to_string())?;
    let v = idb_request_result(req).await?;
    if v.is_undefined() || v.is_null() {
        return Ok(None);
    }

    let arr = js_sys::Uint8Array::new(&v);
    let mut out = vec![0u8; arr.length() as usize];
    arr.copy_to(&mut out);
    Ok(Some(out))
}

/// Save game accuracies to IndexedDB as JSON
async fn save_game_accuracies(accs: &std::collections::HashMap<String, f32>) -> Result<(), String> {
    let json = serde_json::to_string(accs).map_err(|e| format!("serialize error: {}", e))?;
    idb_put_bytes(IDB_KEY_GAME_ACCURACY, json.as_bytes()).await
}

/// Load game accuracies from IndexedDB
async fn load_game_accuracies() -> Result<std::collections::HashMap<String, f32>, String> {
    match idb_get_bytes(IDB_KEY_GAME_ACCURACY).await? {
        Some(bytes) => {
            let json = String::from_utf8(bytes).map_err(|_| "invalid utf8")?;
            serde_json::from_str(&json).map_err(|e| format!("parse error: {}", e))
        }
        None => Ok(std::collections::HashMap::new()),
    }
}

fn choose_text_token_sensor(last_byte: Option<u8>, known_sensors: &[String]) -> String {
    let preferred = match last_byte {
        Some(b) => format!("txt_tok_{b:02X}"),
        None => "txt_tok_UNK".to_string(),
    };

    if known_sensors.iter().any(|s| s == &preferred) {
        return preferred;
    }

    let unk = "txt_tok_UNK";
    if known_sensors.iter().any(|s| s == unk) {
        return unk.to_string();
    }

    known_sensors.first().cloned().unwrap_or(preferred)
}

fn token_action_name_from_sensor(sensor: &str) -> String {
    match sensor.strip_prefix("txt_tok_") {
        Some(suffix) => format!("tok_{suffix}"),
        None => "tok_UNK".to_string(),
    }
}

fn display_token_from_action(action: &str) -> String {
    let Some(suffix) = action.strip_prefix("tok_") else {
        return "<unk>".to_string();
    };

    if suffix == "UNK" {
        return "<unk>".to_string();
    }

    if suffix.len() != 2 {
        return format!("<{suffix}>");
    }

    let Ok(b) = u8::from_str_radix(suffix, 16) else {
        return format!("<{suffix}>");
    };

    if b == b' ' {
        "<sp>".to_string()
    } else if b == b'\n' {
        "\\n".to_string()
    } else if (0x21..=0x7E).contains(&b) {
        (b as char).to_string()
    } else {
        format!("0x{b:02X}")
    }
}

fn softmax_temp(items: &[(String, f32)], temp: f32) -> Vec<f32> {
    if items.is_empty() {
        return Vec::new();
    }

    let t = temp.max(1.0e-6);

    let mut max_score = f32::NEG_INFINITY;
    for (_name, score) in items {
        if score.is_finite() {
            max_score = max_score.max(*score);
        }
    }
    if !max_score.is_finite() {
        max_score = 0.0;
    }

    let mut exps: Vec<f32> = Vec::with_capacity(items.len());
    let mut sum = 0.0f32;
    for (_name, score) in items {
        let s = if score.is_finite() { *score } else { 0.0 };
        let z = ((s - max_score) / t).clamp(-80.0, 80.0);
        let e = z.exp();
        sum += e;
        exps.push(e);
    }

    if !sum.is_finite() || sum <= 0.0 {
        let p = 1.0 / (items.len() as f32);
        return vec![p; items.len()];
    }

    exps.into_iter().map(|e| e / sum).collect()
}

async fn idb_open() -> Result<web_sys::IdbDatabase, String> {
    let promise = idb_open_promise()?;
    let v = wasm_bindgen_futures::JsFuture::from(promise)
        .await
        .map_err(|_| "indexeddb: open() failed".to_string())?;
    v.dyn_into::<web_sys::IdbDatabase>()
        .map_err(|_| "indexeddb: open() returned unexpected type".to_string())
}

fn idb_open_promise() -> Result<js_sys::Promise, String> {
    let w = web_sys::window().ok_or("no window")?;
    let factory = w
        .indexed_db()
        .map_err(|_| "indexeddb() threw".to_string())?
        .ok_or("indexeddb unavailable".to_string())?;

    let req = factory
        .open_with_u32(IDB_DB_NAME, 1)
        .map_err(|_| "indexeddb: open_with_u32() threw".to_string())?;

    let promise = js_sys::Promise::new(&mut |resolve, reject| {
        let resolve = resolve.clone();
        let reject_upgrade = reject.clone();
        let reject_success = reject.clone();
        let reject_error = reject;

        // Upgrade: create the object store.
        let on_upgrade = Closure::wrap(Box::new(move |ev: web_sys::Event| {
            let Some(target) = ev.target() else {
                let _ = reject_upgrade.call1(
                    &JsValue::UNDEFINED,
                    &JsValue::from_str("indexeddb: no event target"),
                );
                return;
            };
            let Ok(open_req) = target.dyn_into::<web_sys::IdbOpenDbRequest>() else {
                let _ = reject_upgrade.call1(
                    &JsValue::UNDEFINED,
                    &JsValue::from_str("indexeddb: bad upgrade target"),
                );
                return;
            };
            let db = match open_req.result() {
                Ok(v) => match v.dyn_into::<web_sys::IdbDatabase>() {
                    Ok(db) => db,
                    Err(_) => {
                        let _ = reject_upgrade.call1(
                            &JsValue::UNDEFINED,
                            &JsValue::from_str("indexeddb: upgrade result not a db"),
                        );
                        return;
                    }
                },
                Err(_) => {
                    let _ = reject_upgrade.call1(
                        &JsValue::UNDEFINED,
                        &JsValue::from_str("indexeddb: upgrade result() threw"),
                    );
                    return;
                }
            };

            // Creating an existing store throws; ignore if it already exists.
            let _ = db.create_object_store(IDB_STORE);
        }) as Box<dyn FnMut(_)>);
        req.set_onupgradeneeded(Some(on_upgrade.as_ref().unchecked_ref()));
        on_upgrade.forget();

        let on_success = Closure::wrap(Box::new(move |ev: web_sys::Event| {
            let Some(target) = ev.target() else {
                let _ = reject_success.call1(
                    &JsValue::UNDEFINED,
                    &JsValue::from_str("indexeddb: no event target"),
                );
                return;
            };
            let Ok(open_req) = target.dyn_into::<web_sys::IdbOpenDbRequest>() else {
                let _ = reject_success.call1(
                    &JsValue::UNDEFINED,
                    &JsValue::from_str("indexeddb: bad success target"),
                );
                return;
            };
            let db = match open_req.result() {
                Ok(v) => v,
                Err(_) => {
                    let _ = reject_success.call1(
                        &JsValue::UNDEFINED,
                        &JsValue::from_str("indexeddb: result() threw"),
                    );
                    return;
                }
            };
            let _ = resolve.call1(&JsValue::UNDEFINED, &db);
        }) as Box<dyn FnMut(_)>);
        req.set_onsuccess(Some(on_success.as_ref().unchecked_ref()));
        on_success.forget();

        let on_error = Closure::wrap(Box::new(move |_ev: web_sys::Event| {
            let _ = reject_error.call1(
                &JsValue::UNDEFINED,
                &JsValue::from_str("indexeddb: open error"),
            );
        }) as Box<dyn FnMut(_)>);
        req.set_onerror(Some(on_error.as_ref().unchecked_ref()));
        on_error.forget();
    });

    Ok(promise)
}

async fn idb_request_done(req: web_sys::IdbRequest) -> Result<(), String> {
    let promise = idb_request_done_promise(req);
    wasm_bindgen_futures::JsFuture::from(promise)
        .await
        .map(|_| ())
        .map_err(|_| "indexeddb: request failed".to_string())
}

async fn idb_request_result(req: web_sys::IdbRequest) -> Result<JsValue, String> {
    let promise = idb_request_result_promise(req);
    wasm_bindgen_futures::JsFuture::from(promise)
        .await
        .map_err(|_| "indexeddb: request failed".to_string())
}

fn idb_request_done_promise(req: web_sys::IdbRequest) -> js_sys::Promise {
    js_sys::Promise::new(&mut |resolve, reject| {
        let on_success = Closure::wrap(Box::new(move |_ev: web_sys::Event| {
            let _ = resolve.call0(&JsValue::UNDEFINED);
        }) as Box<dyn FnMut(_)>);
        req.set_onsuccess(Some(on_success.as_ref().unchecked_ref()));
        on_success.forget();

        let on_error = Closure::wrap(Box::new(move |_ev: web_sys::Event| {
            let _ = reject.call1(
                &JsValue::UNDEFINED,
                &JsValue::from_str("indexeddb: request error"),
            );
        }) as Box<dyn FnMut(_)>);
        req.set_onerror(Some(on_error.as_ref().unchecked_ref()));
        on_error.forget();
    })
}

fn idb_request_result_promise(req: web_sys::IdbRequest) -> js_sys::Promise {
    js_sys::Promise::new(&mut |resolve, reject| {
        let reject_success = reject.clone();
        let reject_error = reject;
        let on_success = Closure::wrap(Box::new(move |ev: web_sys::Event| {
            let Some(target) = ev.target() else {
                let _ = reject_success.call1(
                    &JsValue::UNDEFINED,
                    &JsValue::from_str("indexeddb: no event target"),
                );
                return;
            };
            let Ok(req) = target.dyn_into::<web_sys::IdbRequest>() else {
                let _ = reject_success.call1(
                    &JsValue::UNDEFINED,
                    &JsValue::from_str("indexeddb: bad request target"),
                );
                return;
            };
            let v = match req.result() {
                Ok(v) => v,
                Err(_) => {
                    let _ = reject_success.call1(
                        &JsValue::UNDEFINED,
                        &JsValue::from_str("indexeddb: result() threw"),
                    );
                    return;
                }
            };
            let _ = resolve.call1(&JsValue::UNDEFINED, &v);
        }) as Box<dyn FnMut(_)>);
        req.set_onsuccess(Some(on_success.as_ref().unchecked_ref()));
        on_success.forget();

        let on_error = Closure::wrap(Box::new(move |_ev: web_sys::Event| {
            let _ = reject_error.call1(
                &JsValue::UNDEFINED,
                &JsValue::from_str("indexeddb: request error"),
            );
        }) as Box<dyn FnMut(_)>);
        req.set_onerror(Some(on_error.as_ref().unchecked_ref()));
        on_error.forget();
    })
}

#[allow(deprecated)]
fn clear_canvas(canvas: &web_sys::HtmlCanvasElement) -> Result<(), String> {
    let ctx = canvas
        .get_context("2d")
        .map_err(|_| "canvas: get_context threw".to_string())?
        .ok_or("canvas: missing 2d context".to_string())?
        .dyn_into::<web_sys::CanvasRenderingContext2d>()
        .map_err(|_| "canvas: context is not 2d".to_string())?;

    // Match the app theme (dark scientific background)
    ctx.set_fill_style(&JsValue::from_str("#0a0f1a"));
    ctx.fill_rect(0.0, 0.0, canvas.width() as f64, canvas.height() as f64);
    Ok(())
}

#[allow(deprecated)]
fn draw_spotxy(
    canvas: &web_sys::HtmlCanvasElement,
    x: f32,
    y: f32,
    grid_n: u32,
    accent: &str,
    selected_action: Option<&str>,
) -> Result<(), String> {
    let ctx = canvas
        .get_context("2d")
        .map_err(|_| "canvas: get_context threw".to_string())?
        .ok_or("canvas: missing 2d context".to_string())?
        .dyn_into::<web_sys::CanvasRenderingContext2d>()
        .map_err(|_| "canvas: context is not 2d".to_string())?;

    let w = canvas.width() as f64;
    let h = canvas.height() as f64;

    // Background
    ctx.set_fill_style(&JsValue::from_str("#0a0f1a"));
    ctx.fill_rect(0.0, 0.0, w, h);

    // Map x,y in [-1,1] to canvas coords.
    let px = ((x.clamp(-1.0, 1.0) as f64 + 1.0) * 0.5) * w;
    let py = (1.0 - (y.clamp(-1.0, 1.0) as f64 + 1.0) * 0.5) * h;

    if grid_n >= 2 {
        // Grid mode: N×N cells
        let eff_grid = grid_n as f64;
        let cell_w = w / eff_grid;
        let cell_h = h / eff_grid;

        // Draw grid lines
        ctx.set_stroke_style(&JsValue::from_str("rgba(122, 162, 255, 0.25)"));
        ctx.set_line_width(1.0);
        for i in 1..grid_n {
            let xf = (i as f64) * cell_w;
            let yf = (i as f64) * cell_h;
            ctx.begin_path();
            ctx.move_to(xf, 0.0);
            ctx.line_to(xf, h);
            ctx.stroke();
            ctx.begin_path();
            ctx.move_to(0.0, yf);
            ctx.line_to(w, yf);
            ctx.stroke();
        }

        // Highlight the correct cell (where dot is)
        let cx = ((px / cell_w).floor()).clamp(0.0, eff_grid - 1.0);
        let cy = ((py / cell_h).floor()).clamp(0.0, eff_grid - 1.0);
        let cell_highlight = if accent == "#22c55e" {
            "rgba(34, 197, 94, 0.12)"
        } else {
            "rgba(122, 162, 255, 0.12)"
        };
        ctx.set_fill_style(&JsValue::from_str(cell_highlight));
        ctx.fill_rect(cx * cell_w, cy * cell_h, cell_w, cell_h);

        // Highlight brain's selected cell if different from correct
        if let Some(action) = selected_action {
            // Parse action: spotxy_cell_{n:02}_{ix:02}_{iy:02}
            if let Some(coords) = parse_spotxy_cell_action(action, grid_n) {
                let (sel_ix, sel_iy) = coords;
                // Invert y for canvas (0,0 is top-left in canvas, but (0,0) in grid should be bottom-left)
                let sel_cy = (grid_n - 1 - sel_iy) as f64;
                let sel_cx = sel_ix as f64;

                // Only draw selection highlight if different from correct cell
                if sel_cx != cx || sel_cy != cy {
                    ctx.set_stroke_style(&JsValue::from_str("rgba(251, 191, 36, 0.8)"));
                    ctx.set_line_width(2.0);
                    ctx.stroke_rect(
                        sel_cx * cell_w + 1.0,
                        sel_cy * cell_h + 1.0,
                        cell_w - 2.0,
                        cell_h - 2.0,
                    );
                }
            }
        }
    } else {
        // BinaryX mode: left/right split
        // Draw center divider line
        ctx.set_stroke_style(&JsValue::from_str("rgba(122, 162, 255, 0.35)"));
        ctx.set_line_width(2.0);
        ctx.begin_path();
        ctx.move_to(w / 2.0, 0.0);
        ctx.line_to(w / 2.0, h);
        ctx.stroke();

        // Highlight the correct half (where dot is)
        let is_left = px < w / 2.0;
        let correct_highlight = if accent == "#22c55e" {
            "rgba(34, 197, 94, 0.10)"
        } else {
            "rgba(122, 162, 255, 0.10)"
        };
        ctx.set_fill_style(&JsValue::from_str(correct_highlight));
        if is_left {
            ctx.fill_rect(0.0, 0.0, w / 2.0, h);
        } else {
            ctx.fill_rect(w / 2.0, 0.0, w / 2.0, h);
        }

        // Highlight brain's selected side if different
        if let Some(action) = selected_action {
            let brain_is_left = action == "left";
            if brain_is_left != is_left {
                // Brain selected wrong side - show orange border
                ctx.set_stroke_style(&JsValue::from_str("rgba(251, 191, 36, 0.8)"));
                ctx.set_line_width(3.0);
                if brain_is_left {
                    ctx.stroke_rect(2.0, 2.0, w / 2.0 - 4.0, h - 4.0);
                } else {
                    ctx.stroke_rect(w / 2.0 + 2.0, 2.0, w / 2.0 - 4.0, h - 4.0);
                }
            }
        }

        // Add "L" and "R" labels
        ctx.set_fill_style(&JsValue::from_str("rgba(178, 186, 210, 0.3)"));
        ctx.set_font("bold 24px sans-serif");
        ctx.set_text_align("center");
        ctx.set_text_baseline("middle");
        let _ = ctx.fill_text("L", w / 4.0, h / 2.0);
        let _ = ctx.fill_text("R", 3.0 * w / 4.0, h / 2.0);
    }

    // Dot
    ctx.set_fill_style(&JsValue::from_str(accent));
    ctx.begin_path();
    let _ = ctx.arc(px, py, 6.0, 0.0, std::f64::consts::PI * 2.0);
    ctx.fill();
    Ok(())
}

/// Parse a grid cell action like "spotxy_cell_02_01_00" into (ix, iy)
fn parse_spotxy_cell_action(action: &str, expected_n: u32) -> Option<(u32, u32)> {
    // Format: spotxy_cell_{n:02}_{ix:02}_{iy:02}
    let parts: Vec<&str> = action.split('_').collect();
    if parts.len() != 5 || parts[0] != "spotxy" || parts[1] != "cell" {
        return None;
    }
    let n: u32 = parts[2].parse().ok()?;
    if n != expected_n {
        return None;
    }
    let ix: u32 = parts[3].parse().ok()?;
    let iy: u32 = parts[4].parse().ok()?;
    Some((ix, iy))
}

#[allow(deprecated)]
fn draw_pong(canvas: &web_sys::HtmlCanvasElement, s: &PongUiState) -> Result<(), String> {
    let ctx = canvas
        .get_context("2d")
        .map_err(|_| "canvas: get_context threw".to_string())?
        .ok_or("canvas: missing 2d context".to_string())?
        .dyn_into::<web_sys::CanvasRenderingContext2d>()
        .map_err(|_| "canvas: context is not 2d".to_string())?;

    let w = canvas.width() as f64;
    let h = canvas.height() as f64;

    // Dark gradient background
    ctx.set_fill_style_str("#0a0f1a");
    ctx.fill_rect(0.0, 0.0, w, h);

    // Subtle grid lines for depth
    ctx.set_stroke_style_str("rgba(122, 162, 255, 0.06)");
    ctx.set_line_width(1.0);
    let grid_spacing = 30.0;
    let mut x = grid_spacing;
    while x < w {
        ctx.begin_path();
        ctx.move_to(x, 0.0);
        ctx.line_to(x, h);
        ctx.stroke();
        x += grid_spacing;
    }
    let mut y = grid_spacing;
    while y < h {
        ctx.begin_path();
        ctx.move_to(0.0, y);
        ctx.line_to(w, y);
        ctx.stroke();
        y += grid_spacing;
    }

    // Field (keep it crisp; no glow)
    let field_inset = 12.0;
    let field_left = field_inset;
    let field_right = (w - field_inset).max(field_left + 1.0);
    let field_top = field_inset;
    let field_bottom = (h - field_inset).max(field_top + 1.0);
    ctx.set_stroke_style_str("rgba(122, 162, 255, 0.28)");
    ctx.set_line_width(2.0);
    ctx.stroke_rect(
        field_left,
        field_top,
        field_right - field_left,
        field_bottom - field_top,
    );

    // Map simulation coordinates to pixels.
    // PongSim uses ball center positions: ball_x in [0,1], ball_y in [-1,1].
    // To make collisions *look* correct, map x=0 to the paddle face, and y=±1 to the walls.
    let ball_r = 6.0;
    let paddle_w = 10.0;
    let paddle_x = field_left; // paddle runs along the left wall

    let play_left = (paddle_x + paddle_w + ball_r).min(field_right - 1.0);
    let play_right = (field_right - ball_r).max(play_left + 1.0);
    let play_top = (field_top + ball_r).min(field_bottom - 1.0);
    let play_bottom = (field_bottom - ball_r).max(play_top + 1.0);

    let play_w = (play_right - play_left).max(1.0);
    let play_h = (play_bottom - play_top).max(1.0);

    let map_x = |x01: f32| play_left + (x01.clamp(0.0, 1.0) as f64) * play_w;
    let map_y = |ys: f32| {
        // ys=+1 is top wall; ys=-1 is bottom wall
        play_top + (1.0 - ((ys.clamp(-1.0, 1.0) as f64 + 1.0) * 0.5)) * play_h
    };

    // Paddle
    let paddle_center_y = map_y(s.paddle_y);
    let paddle_half_px = (s.paddle_half_height.clamp(0.01, 1.0) as f64) * (play_h * 0.5);
    let paddle_height = (paddle_half_px * 2.0).min(play_h);
    let paddle_top =
        (paddle_center_y - paddle_half_px).clamp(play_top, play_bottom - paddle_height);

    ctx.set_fill_style_str("#7aa2ff");
    ctx.fill_rect(paddle_x, paddle_top, paddle_w, paddle_height);
    ctx.set_fill_style_str("rgba(255, 255, 255, 0.25)");
    ctx.fill_rect(
        paddle_x + 1.5,
        paddle_top + 2.0,
        2.0,
        (paddle_height - 4.0).max(0.0),
    );

    // Ball (crisp; no glow)
    if s.ball_visible {
        let bx = map_x(s.ball_x);
        let by = map_y(s.ball_y);

        ctx.set_fill_style_str("#fbbf24");
        ctx.begin_path();
        let _ = ctx.arc(bx, by, ball_r, 0.0, std::f64::consts::PI * 2.0);
        ctx.fill();

        ctx.set_fill_style_str("rgba(255, 255, 255, 0.55)");
        ctx.begin_path();
        let _ = ctx.arc(
            bx - ball_r * 0.35,
            by - ball_r * 0.35,
            ball_r * 0.45,
            0.0,
            std::f64::consts::PI * 2.0,
        );
        ctx.fill();
    }

    // Score zone indicator (right edge)
    ctx.set_fill_style_str("rgba(239, 68, 68, 0.12)");
    ctx.fill_rect(field_right - 6.0, field_top, 6.0, field_bottom - field_top);

    Ok(())
}

fn download_bytes(filename: &str, bytes: &[u8]) -> Result<(), String> {
    let window = web_sys::window().ok_or("no window".to_string())?;
    let document = window.document().ok_or("no document".to_string())?;

    let array = js_sys::Uint8Array::from(bytes);
    let parts = js_sys::Array::new();
    parts.push(&array.buffer());
    let blob = web_sys::Blob::new_with_u8_array_sequence(&parts)
        .map_err(|_| "blob: failed to create".to_string())?;

    let url = web_sys::Url::create_object_url_with_blob(&blob)
        .map_err(|_| "url: create_object_url failed".to_string())?;

    let a = document
        .create_element("a")
        .map_err(|_| "document: create_element failed".to_string())?
        .dyn_into::<web_sys::HtmlAnchorElement>()
        .map_err(|_| "document: anchor cast failed".to_string())?;

    a.set_href(&url);
    a.set_download(filename);
    a.click();

    let _ = web_sys::Url::revoke_object_url(&url);
    Ok(())
}

async fn read_file_bytes(file: web_sys::File) -> Result<Vec<u8>, String> {
    let promise = file_reader_array_buffer_promise(file)?;
    let v = wasm_bindgen_futures::JsFuture::from(promise)
        .await
        .map_err(|_| "file: read failed".to_string())?;

    let buf = v
        .dyn_into::<js_sys::ArrayBuffer>()
        .map_err(|_| "file: expected ArrayBuffer".to_string())?;
    let arr = js_sys::Uint8Array::new(&buf);
    let mut out = vec![0u8; arr.length() as usize];
    arr.copy_to(&mut out);
    Ok(out)
}

fn file_reader_array_buffer_promise(file: web_sys::File) -> Result<js_sys::Promise, String> {
    let reader =
        web_sys::FileReader::new().map_err(|_| "file: FileReader::new failed".to_string())?;
    reader
        .read_as_array_buffer(&file)
        .map_err(|_| "file: read_as_array_buffer failed".to_string())?;

    Ok(js_sys::Promise::new(&mut |resolve, reject| {
        let reject_load = reject.clone();
        let reject_err = reject;
        let reader_ok = reader.clone();
        let onload =
            Closure::wrap(Box::new(
                move |_ev: web_sys::ProgressEvent| match reader_ok.result() {
                    Ok(v) => {
                        if v.is_null() || v.is_undefined() {
                            let _ = reject_load.call1(
                                &JsValue::UNDEFINED,
                                &JsValue::from_str("file: missing result"),
                            );
                        } else {
                            let _ = resolve.call1(&JsValue::UNDEFINED, &v);
                        }
                    }
                    Err(_) => {
                        let _ = reject_load.call1(
                            &JsValue::UNDEFINED,
                            &JsValue::from_str("file: result() threw"),
                        );
                    }
                },
            ) as Box<dyn FnMut(_)>);
        reader.set_onload(Some(onload.as_ref().unchecked_ref()));
        onload.forget();

        let onerror = Closure::wrap(Box::new(move |_ev: web_sys::ProgressEvent| {
            let _ = reject_err.call1(&JsValue::UNDEFINED, &JsValue::from_str("file: read error"));
        }) as Box<dyn FnMut(_)>);
        reader.set_onerror(Some(onerror.as_ref().unchecked_ref()));
        onerror.forget();
    }))
}
