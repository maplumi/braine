use braine::substrate::{Brain, BrainConfig, Stimulus, UnitPlotPoint};
use braine_games::{
    bandit::BanditGame, spot::SpotGame, spot_reversal::SpotReversalGame, spot_xy::SpotXYGame,
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
use text_web::TextWebGame;
use wasm_bindgen::closure::Closure;
use charts::RollingHistory;
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
    let (dashboard_tab, set_dashboard_tab) = signal(DashboardTab::GameDetails);
    let (analytics_panel, set_analytics_panel) = signal(AnalyticsPanel::Performance);
    let (trial_period_ms, set_trial_period_ms) = signal(500u32);
    let (run_interval_ms, set_run_interval_ms) = signal(33u32);
    let (exploration_eps, set_exploration_eps) = signal(0.08f32);
    let (meaning_alpha, set_meaning_alpha) = signal(6.0f32);
    let (reward_scale, set_reward_scale) = signal(1.0f32);
    let (reward_bias, set_reward_bias) = signal(0.0f32);
    let (learning_enabled, set_learning_enabled) = signal(true);
    let (grow_units_n, set_grow_units_n) = signal(128u32);

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

    // Spot/SpotReversal stimulus (left/right) for UI highlighting.
    let (spot_is_left, set_spot_is_left) = signal::<Option<bool>>(None);

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

    // BrainViz uses its own sampling so it can be tuned independently.
    let (brainviz_points, set_brainviz_points) = signal::<Vec<UnitPlotPoint>>(Vec::new());
    let (brainviz_node_sample, set_brainviz_node_sample) = signal(128u32);
    let (brainviz_edges_per_node, set_brainviz_edges_per_node) = signal(4u32);
    let (brainviz_zoom, set_brainviz_zoom) = signal(1.0f32);
    let (brainviz_pan_x, set_brainviz_pan_x) = signal(0.0f32);
    let (brainviz_pan_y, set_brainviz_pan_y) = signal(0.0f32);
    let (brainviz_auto_rotate, set_brainviz_auto_rotate) = signal(true);
    let (brainviz_hover, set_brainviz_hover) = signal::<Option<(u32, f64, f64)>>(None);
    let brainviz_dragging = StoredValue::new(false);
    let brainviz_last_drag_xy = StoredValue::new((0.0f64, 0.0f64));
    let brainviz_hit_nodes = StoredValue::new(Vec::<charts::BrainVizHitNode>::new());
    
    // Per-game accuracy persistence (loaded from IDB)
    let (game_accuracies, set_game_accuracies) = signal::<std::collections::HashMap<String, f32>>(std::collections::HashMap::new());

    // WebGPU availability check (currently: Brain uses CPU Scalar tier always; future: could detect and use WebGPU).
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
    let (gpu_status, _set_gpu_status) = signal(if webgpu_available {
        "WebGPU: available (not yet used by braine)"
    } else {
        "WebGPU: not available (CPU only)"
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
            set_spot_is_left.set(snap.spot_is_left);

            // Persist per-game stats + chart history so refresh restores the current state.
            let kind = game_kind.get_untracked();
            let should_persist = persisted_trial_cursor.with_value(|(k, t)| *k != kind || *t != stats.trials);
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
            refresh_ui_from_runtime();
            set_status.set(format!("grew units by {}", n));
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
                    WebGame::Text(g) => (g.token_sensor_names().to_vec(), g.allowed_actions().to_vec()),
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
                        let regime_sensor = if regime == 1 { "txt_regime_1" } else { "txt_regime_0" };
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

                        let raw_reward = if chosen_action == desired_action { 1.0 } else { -1.0 };
                        let shaped = ((raw_reward + reward_bias) * reward_scale).clamp(-5.0, 5.0);

                        r.brain.note_action(&chosen_action);
                        r.brain
                            .note_compound_symbol(&["pair", ctx.as_str(), chosen_action.as_str()]);

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
            match download_bytes("brain.bbi", &bytes) {
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
                                    Ok(()) => set_status.set(format!(
                                        "imported {} bytes (.bbi); auto-saved to IndexedDB",
                                        bytes.len()
                                    )),
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
        let Some(canvas) = canvas_ref.get() else {
            return;
        };

    let grid_n = spotxy_grid_n.get();
        let accent = if spotxy_eval.get() { "#22c55e" } else { "#7aa2ff" };

        match pos {
            Some((x, y)) => {
                let _ = draw_spotxy(&canvas, x, y, grid_n, accent);
            }
            None => {
                let _ = draw_spotxy(&canvas, 0.0, 0.0, grid_n, accent);
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
        let Some(canvas) = perf_chart_ref.get() else { return; };
        let data: Vec<f32> = perf_history.with_value(|h| h.data().to_vec());
        let _ = charts::draw_line_chart(&canvas, &data, 0.0, 1.0, "#7aa2ff", "#0a0f1a", "#1a2540");
    });
    
    // Neuromod (reward) chart canvas
    let neuromod_chart_ref = NodeRef::<leptos::html::Canvas>::new();
    Effect::new(move |_| {
        let _ = neuromod_version.get();
        let Some(canvas) = neuromod_chart_ref.get() else { return; };
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
            let Some(canvas) = choices_chart_ref.get() else { return; };

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
                    .map(|a| if keep.contains(&a) { a } else { "(other)".to_string() })
                    .collect()
            } else {
                events
            };

            let _ = charts::draw_choices_over_time(
                &canvas,
                &actions,
                &mapped,
                window,
                "#0a0f1a",
                "#1a2540",
            );
        }
    });
    
    // Accuracy gauge canvas
    let gauge_ref = NodeRef::<leptos::html::Canvas>::new();
    Effect::new(move |_| {
        let rate = recent_rate.get();
        let Some(canvas) = gauge_ref.get() else { return; };
        let color = if rate >= 0.85 { "#4ade80" } else if rate >= 0.70 { "#fbbf24" } else { "#7aa2ff" };
        let _ = charts::draw_gauge(&canvas, rate, 0.0, 1.0, "Accuracy", color, "#0a0f1a");
    });
    
    // Unit plot 3D-style canvas for Graph page
    let unit_plot_ref = NodeRef::<leptos::html::Canvas>::new();
    Effect::new(move |_| {
        let points = unit_plot.get();
        let Some(canvas) = unit_plot_ref.get() else { return; };
        let _ = charts::draw_unit_plot_3d(&canvas, &points, "#0a0f1a");
    });

    // BrainViz: sample plot points based on UI settings (does not affect learning/perf history).
    Effect::new({
        let runtime = runtime.clone();
        move |_| {
            if analytics_panel.get() != AnalyticsPanel::BrainViz {
                return;
            }

            let n = brainviz_node_sample.get().clamp(16, 1024) as usize;
            let pts = runtime.with_value(|r| r.brain.unit_plot_points(n));
            set_brainviz_points.set(pts);
        }
    });

    // BrainViz: rotating sphere + sampled connectivity
    let brain_viz_ref = NodeRef::<leptos::html::Canvas>::new();
    Effect::new({
        let runtime = runtime.clone();
        move |_| {
            if analytics_panel.get() != AnalyticsPanel::BrainViz {
                return;
            }

            let steps = steps.get();
            let points = brainviz_points.get();
            let Some(canvas) = brain_viz_ref.get() else { return; };

            let edges_per_node = brainviz_edges_per_node.get().clamp(1, 32) as usize;
            let zoom = brainviz_zoom.get();
            let pan_x = brainviz_pan_x.get();
            let pan_y = brainviz_pan_y.get();
            let auto_rotate = brainviz_auto_rotate.get();

            let mut sampled: std::collections::HashSet<usize> = std::collections::HashSet::new();
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

            let rot = if auto_rotate { (steps as f32) * 0.02 } else { 0.0 };
            let opts = charts::BrainVizRenderOptions {
                zoom,
                pan_x,
                pan_y,
                draw_outline: false,
                node_size_scale: 0.5,
            };
            if let Ok(hits) = charts::draw_brain_connectivity_sphere(
                &canvas,
                &points,
                &edges,
                rot,
                "#0a0f1a",
                opts,
            ) {
                brainviz_hit_nodes.set_value(hits);
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

            // Game tabs navigation
            <nav class="app-nav">
                {GameKind::all().iter().map(|&kind| {
                    let set_game = Arc::clone(&set_game);
                    view! {
                        <button
                            class=move || if game_kind.get() == kind { "tab active" } else { "tab" }
                            on:click=move |_| set_game(kind)
                        >
                            {kind.display_name()}
                        </button>
                    }
                }).collect_view()}
            </nav>

            // Main content area
            <div class="content-split">
                // Game area (left)
                <div class="game-area">
                        // Controls bar
                        <div class="controls">
                            <button class="btn" on:click=move |_| do_tick()>"⏯ Step"</button>
                            <button class=move || if is_running.get() { "btn" } else { "btn primary" } on:click=move |_| do_start()>"▶ Run"</button>
                            <button class="btn" on:click=move |_| do_stop()>"⏹ Stop"</button>
                            <button class="btn" on:click=move |_| do_reset()>"↺ Reset"</button>
                            <div class="spacer"></div>
                            <label class="label">
                                <span>"Trial ms"</span>
                                <input
                                    type="number"
                                min="10"
                                max="60000"
                                class="input"
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
                            <span>"ε"</span>
                            <input
                                type="number"
                                min="0"
                                max="1"
                                step="0.01"
                                class="input"
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
                            <span>"α"</span>
                            <input
                                type="number"
                                min="0"
                                max="30"
                                step="0.5"
                                class="input"
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
                                <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 12px; width: 100%;">
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
                                    <span style="color: var(--muted); font-size: 0.85rem;">"Brain chose:"</span>
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
                                <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 16px; width: 100%;">
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
                                <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 12px; width: 100%;">
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
                            <div style="display: flex; flex-direction: column; gap: 20px; width: 100%; max-width: 480px;">
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
                                <div style="position: relative; display: flex; justify-content: center;">
                                    <div style=move || format!("position: absolute; width: 280px; height: 280px; border-radius: 50%; filter: blur(60px); opacity: 0.3; background: {};",
                                        if spotxy_eval.get() { "#22c55e" } else { "#7aa2ff" })>
                                    </div>
                                    <div style=move || format!("position: relative; padding: 3px; border-radius: 16px; background: {};",
                                        if spotxy_eval.get() { "linear-gradient(135deg, #22c55e, #16a34a)" } else { "linear-gradient(135deg, #7aa2ff, #5b7dc9)" })>
                                        <canvas
                                            node_ref=canvas_ref
                                            width="340"
                                            height="340"
                                            style="border-radius: 13px; background: #0a0f1a; display: block;"
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
                            <div style="display: flex; flex-direction: column; gap: 20px; width: 100%; max-width: 580px;">
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
                                
                                // Game canvas with ambient glow
                                <div style="position: relative; display: flex; justify-content: center;">
                                    <div style="position: absolute; width: 400px; height: 200px; border-radius: 50%; filter: blur(80px); opacity: 0.2; background: linear-gradient(135deg, #7aa2ff, #fbbf24);"></div>
                                    <div style="position: relative; padding: 3px; border-radius: 16px; background: linear-gradient(135deg, #7aa2ff, #5b7dc9);">
                                        <canvas
                                            node_ref=pong_canvas_ref
                                            width="540"
                                            height="320"
                                            style="border-radius: 13px; display: block;"
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
                    </div>

                // Dashboard (right) - Tabbed panel
                <div class="dashboard">
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

                                        <label class="label">
                                            <span>"Corpus 0"</span>
                                            <textarea
                                                class="input"
                                                rows="3"
                                                prop:value=move || text_corpus0.get()
                                                on:input=move |ev| set_text_corpus0.set(event_target_value(&ev))
                                            />
                                        </label>

                                        <label class="label">
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
                                                    class="input"
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
                                                    class="input"
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

                                        <label class="label">
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
                                                    class="input"
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
                                                    class="input"
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
                                        <p class="subtle">"This does not train; it runs on a cloned brain so it won’t perturb the running game."</p>

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
                                                    class="input"
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
                                                    class="input"
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
                                        <span class="stat-label">"Memory"</span>
                                        <span class="stat-value">{move || format!("{}KB", diag.get().memory_bytes / 1024)}</span>
                                    </div>
                                </div>

                                <div class="card">
                                    <h3 class="card-title">"💾 Persistence"</h3>
                                    <div class="stack tight">
                                        <button class="btn" on:click=move |_| do_save()>"💾 Save (IndexedDB)"</button>
                                        <button class="btn" on:click=move |_| do_load()>"📂 Load (IndexedDB)"</button>
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
                                                    class="input"
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
                                        <p class="subtle">"sensor / group / other / reserved"</p>
                                        <canvas node_ref=unit_plot_ref width="800" height="360" class="canvas tall"></canvas>
                                        <p class="subtle">"Sensors (blue), groups (green), other (yellow), reserved (gray)."</p>
                                    </div>
                                </Show>

                                <Show when=move || analytics_panel.get() == AnalyticsPanel::BrainViz>
                                    <div class="card">
                                        <h3 class="card-title">"🧠 Internal Brain Visualization"</h3>
                                        <p class="subtle">"Sampled nodes on a rotating sphere; edges show connection strength."</p>
                                        <div class="callout">
                                            <p>"Drag to pan • Scroll to zoom • Hover for details"</p>
                                        </div>

                                        <div class="row end wrap" style="margin-top: 10px;">
                                            <label class="label">
                                                <span>"Nodes"</span>
                                                <input
                                                    class="input"
                                                    type="number"
                                                    min="16"
                                                    max="1024"
                                                    step="16"
                                                    prop:value=move || brainviz_node_sample.get().to_string()
                                                    on:input=move |ev| {
                                                        if let Ok(v) = event_target_value(&ev).parse::<u32>() {
                                                            set_brainviz_node_sample.set(v.clamp(16, 1024));
                                                        }
                                                    }
                                                />
                                            </label>
                                            <label class="label">
                                                <span>"Edges/node"</span>
                                                <input
                                                    class="input"
                                                    type="number"
                                                    min="1"
                                                    max="32"
                                                    step="1"
                                                    prop:value=move || brainviz_edges_per_node.get().to_string()
                                                    on:input=move |ev| {
                                                        if let Ok(v) = event_target_value(&ev).parse::<u32>() {
                                                            set_brainviz_edges_per_node.set(v.clamp(1, 32));
                                                        }
                                                    }
                                                />
                                            </label>
                                            <label class="label" style="flex-direction: row; align-items: center; gap: 10px;">
                                                <input
                                                    type="checkbox"
                                                    prop:checked=move || brainviz_auto_rotate.get()
                                                    on:change=move |ev| {
                                                        set_brainviz_auto_rotate.set(event_target_checked(&ev));
                                                    }
                                                />
                                                <span>"Auto-rotate"</span>
                                            </label>
                                            <button
                                                class="btn sm"
                                                on:click=move |_| {
                                                    set_brainviz_zoom.set(1.0);
                                                    set_brainviz_pan_x.set(0.0);
                                                    set_brainviz_pan_y.set(0.0);
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
                                                        set_brainviz_pan_x.update(|v| *v += dx as f32);
                                                        set_brainviz_pan_y.update(|v| *v += dy as f32);
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
                                                class="input"
                                                type="number"
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
                            </div>
                        </Show>

                        <Show when=move || dashboard_tab.get() == DashboardTab::Learning>
                            <div class="stack">
                                <div class="card">
                                    <h3 class="card-title">"🧪 Learning"</h3>
                                    <p class="subtle">"Controls for learning writes, reward shaping, and simulation cadence."</p>
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
                                    <h3 class="card-title">"⏱ Simulation Speed"</h3>
                                    <div class="row end wrap">
                                        <label class="label">
                                            <span>"Run interval (ms)"</span>
                                            <input
                                                class="input"
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
                                                class="input"
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
                                                class="input"
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

                        <Show when=move || dashboard_tab.get() == DashboardTab::About>
                            <div style="display: flex; flex-direction: column; gap: 14px;">
                                <div style="text-align: center; padding: 14px; background: linear-gradient(135deg, rgba(122, 162, 255, 0.12), rgba(0,0,0,0.2)); border: 1px solid var(--border); border-radius: 12px;">
                                    <div style="font-size: 1.8rem;">"🧠"</div>
                                    <h2 style="margin: 6px 0 2px 0; font-size: 1.15rem; color: var(--accent);">"Braine — closed-loop learning substrate"</h2>
                                    <p style="margin: 0; color: var(--muted); font-size: 0.85rem;">"Sparse recurrent dynamics • local plasticity • scalar reward (neuromodulation) • no backprop"</p>
                                </div>

                                <div style=STYLE_CARD>
                                    <h3 style="margin: 0 0 10px 0; font-size: 0.95rem; color: var(--accent);">"What you’re looking at"</h3>
                                    <p style="margin: 0; color: var(--text); font-size: 0.9rem; line-height: 1.7;">
                                        "This web lab is designed to run a game (left) while you watch learning signals, indicators, and graphs (right). The goal is interpretability while learning happens — not offline training."
                                    </p>
                                </div>

                                <div style=STYLE_CARD>
                                    <h3 style="margin: 0 0 10px 0; font-size: 0.95rem; color: var(--accent);">"Key principles"</h3>
                                    <div style="display: flex; flex-direction: column; gap: 10px;">
                                        <div>
                                            <strong style="color: var(--text);">"Learning modifies state"</strong>
                                            <div style="color: var(--muted); font-size: 0.85rem; line-height: 1.6;">"Plasticity + reward updates internal couplings and causal memory."</div>
                                        </div>
                                        <div>
                                            <strong style="color: var(--text);">"Inference uses state"</strong>
                                            <div style="color: var(--muted); font-size: 0.85rem; line-height: 1.6;">"Action selection is a readout from the learned dynamics."</div>
                                        </div>
                                        <div>
                                            <strong style="color: var(--text);">"Closed loop"</strong>
                                            <div style="color: var(--muted); font-size: 0.85rem; line-height: 1.6;">"Stimulus → dynamics → action → reward, repeated online."</div>
                                        </div>
                                    </div>
                                </div>

                                <div style=STYLE_CARD>
                                    <h3 style="margin: 0 0 10px 0; font-size: 0.95rem; color: var(--accent);">"Data format (.bbi brain image)"</h3>
                                    <p style="margin: 0 0 10px 0; color: var(--muted); font-size: 0.85rem; line-height: 1.6;">
                                        "Braine persists the substrate as a custom, versioned, chunked binary format (BBI). This is not generic serde; it’s designed to be forward-skippable and capacity-aware."
                                    </p>
                                    <pre class="codeblock">{"BBI header (little-endian)\n- magic: BRAINE01 (8 bytes)\n- version: u32 (currently 2)\n\nChunk layout (v2, compressed)\n- tag: [u8;4] ASCII\n- len: u32 (bytes following)\n- uncompressed_len: u32\n- lz4_payload: bytes\n\nKnown chunk tags\n- CFG0: BrainConfig (unit_count, connectivity_per_unit, dt, base_freq, noise_amp, noise_phase, global_inhibition, hebb_rate, forget_rate, prune_below, coactive_threshold, phase_lock_threshold, imprint_rate, seed?, causal_decay)\n- PRNG: RNG state (u64)\n- STAT: age_steps (u64)\n- UNIT: unit states + CSR sparse connections\n- MASK: reserved[] + learning_enabled[] (bitsets)\n- GRPS: sensor/action group definitions\n- SYMB: symbol string table (rebuilds symbol map)\n- CAUS: causal memory counts/edges\n\nUnknown tags are skipped on load for forward-compatibility."}</pre>
                                </div>

                                <div style=STYLE_CARD>
                                    <h3 style="margin: 0 0 10px 0; font-size: 0.95rem; color: var(--accent);">"Storage mechanisms"</h3>
                                    <pre class="codeblock">{"Web (this app)\n- IndexedDB (db: 'braine', store: 'kv')\n  - key 'brain_image': raw BBI bytes (Brain::save_image_bytes / load_image_bytes)\n  - key 'game_accuracy': JSON map {game -> accuracy}\n- localStorage\n  - 'braine_theme': UI theme\n  - 'braine_settings_v1': JSON {reward_scale, reward_bias, learning_enabled, run_interval_ms, trial_period_ms}\n  - 'braine_game_stats_v1.<Game>': JSON (stats + chart history + choices history)\n- Export/Import\n  - Export downloads the current brain image as a .bbi snapshot\n  - Import loads a .bbi (optionally autosave to IndexedDB)\n\nDaemon mode (brained)\n- Persists to OS data directories (brain.bbi)\n  - Linux: ~/.local/share/braine/brain.bbi\n  - Windows: %APPDATA%\\Braine\\brain.bbi\n  - macOS: ~/Library/Application Support/Braine/brain.bbi\n- Uses newline-delimited JSON over TCP (127.0.0.1:9876) between daemon and clients."}</pre>
                                </div>

                                <div style=STYLE_CARD>
                                    <h3 style="margin: 0 0 10px 0; font-size: 0.95rem; color: var(--accent);">"Configuration parameters"</h3>
                                    <pre class="codeblock">{"BrainConfig (core defaults)\n- unit_count: 256\n- connectivity_per_unit: 12\n- dt: 0.05\n- base_freq: 1.0\n- noise_amp: 0.02\n- noise_phase: 0.01\n- global_inhibition: 0.2\n- hebb_rate: 0.08\n- forget_rate: 0.0005\n- prune_below: 0.01\n- coactive_threshold: 0.3\n- phase_lock_threshold: 0.7\n- imprint_rate: 0.5\n- seed: None\n- causal_decay: 0.002\n\nWeb lab defaults\n- seed override: 2026 (make_default_brain)\n- trial_period_ms: 500\n- run_interval_ms: 33 (≈30Hz)\n- exploration_eps (ε): 0.08\n- meaning_alpha (α): 6.0\n\nGame seeds\n- PongSim seed: 0xB0A7_F00D\n- Web runtime RNG seed: 0xC0FF_EE12"}</pre>
                                </div>

                                <div style=STYLE_CARD>
                                    <h3 style="margin: 0 0 10px 0; font-size: 0.95rem; color: var(--accent);">"Equations (dynamics + learning)"</h3>
                                    <p style="margin: 0 0 10px 0; color: var(--muted); font-size: 0.85rem; line-height: 1.6;">
                                        "These equations mirror the implementation in the core substrate (amp/phase oscillators + sparse couplings + local plasticity + forgetting/pruning + causal memory)."
                                    </p>
                                    <pre class="codeblock">{"State per unit i\n- amplitude a_i(t)\n- phase φ_i(t)\n- bias b_i, decay d_i\n- one-tick input x_i(t) from applied stimuli\n\nSparse recurrent influence (neighbors N(i) with weights w_{ij})\n- amp influence:  S_i(t) = Σ_{j∈N(i)} w_{ij} · a_j(t)\n- phase influence: P_i(t) = Σ_{j∈N(i)} w_{ij} · Δ(φ_j(t), φ_i(t))\n\nGlobal inhibition (competition)\n- ā(t) = (1/N) Σ_k a_k(t)\n- I(t) = g · ā(t)   where g = global_inhibition\n\nNoise\n- η^a_i ~ U[-noise_amp, +noise_amp]\n- η^φ_i ~ U[-noise_phase, +noise_phase]\n\nDynamics update (Euler step, dt)\n- damp term: D_i(t) = d_i · a_i(t)\n- a_i(t+dt) = clip( a_i(t) + (b_i + x_i + S_i − I − D_i + η^a_i) · dt , [-2, 2])\n- φ_i(t+dt) = wrap( φ_i(t) + (base_freq + P_i + η^φ_i) · dt )\n\nHebbian plasticity on existing sparse edges\n- activity threshold θ = coactive_threshold\n- phase alignment A(φ_i, φ_j) ∈ [0,1]\n- phase threshold θ_φ = phase_lock_threshold\n- learning rate: lr = hebb_rate · (1 + max(0, neuromod))\n\nFor each edge i→j (stored in CSR)\n- if a_i > θ and a_j > θ and A(φ_i, φ_j) > θ_φ:\n    Δw_{ij} = lr · A(φ_i, φ_j)\n  else if a_i > θ and a_j > θ and A(φ_i, φ_j) ≤ θ_φ:\n    Δw_{ij} = −lr · 0.05\n  else:\n    Δw_{ij} = 0\n- w_{ij} ← clip(w_{ij} + Δw_{ij}, [-1.5, 1.5])\n\nForgetting + pruning (structural decay)\n- w_{ij} ← (1 − forget_rate) · w_{ij}\n- if |w_{ij}| < prune_below: prune edge (except “engram” sensor↔concept edges keep a minimal trace)\n\nAction scoring (habit + meaning)\n- habit_norm(a) = clamp( (Σ_{u∈action_units(a)} max(0, a_u)) / (|units| · 2), [0,1])\n- meaning(a|stimulus) = (pair_value · 1.0) + (global_value · 0.15)\n  where\n    global_value = causal(a, reward_pos) − causal(a, reward_neg)\n    pair_value = causal(pair(stimulus,a), reward_pos) − causal(pair(stimulus,a), reward_neg)\n- score(a|stimulus) = 0.5 · habit_norm(a) + meaning_alpha · meaning(a|stimulus)\n\nIn the web loop, exploration is ε-gated: with probability ε choose a random allowed action; otherwise choose the top scored action."}</pre>
                                </div>

                                <div style="text-align: center; padding: 12px; color: var(--muted); font-size: 0.8rem;">
                                    "More docs live in the repo: "
                                    <a href="https://github.com/maplumi/braine" target="_blank" style="color: var(--accent);">"github.com/maplumi/braine"</a>
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
    #[default]
    GameDetails,
    Learning,
    Stats,
    Analytics,
    Settings,
    About,
}

impl DashboardTab {
    fn label(self) -> &'static str {
        match self {
            DashboardTab::GameDetails => "Game Details",
            DashboardTab::Learning => "Learning",
            DashboardTab::Stats => "Stats",
            DashboardTab::Analytics => "Analytics",
            DashboardTab::Settings => "Settings",
            DashboardTab::About => "About",
        }
    }
    fn icon(self) -> &'static str {
        match self {
            DashboardTab::GameDetails => "🧩",
            DashboardTab::Learning => "🧠",
            DashboardTab::Stats => "📊",
            DashboardTab::Analytics => "📈",
            DashboardTab::Settings => "⚙️",
            DashboardTab::About => "ℹ️",
        }
    }
    fn all() -> &'static [DashboardTab] {
        &[
            DashboardTab::About,
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
            AnalyticsPanel::BrainViz => "Brain Viz",
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
        }
    }
    
    fn inputs_info(self) -> &'static str {
        match self {
            GameKind::Spot => "Stimuli (by name): spot_left or spot_right\nActions: left, right\nTrial timing: controlled by Trial ms",
            GameKind::Bandit => "Stimulus: bandit (constant context)\nActions: left, right\nParameters: prob_left=0.8, prob_right=0.2",
            GameKind::SpotReversal => "Stimuli: spot_left or spot_right (+ reversal context sensor spot_rev_ctx when reversed)\nActions: left, right\nParameter: flip_after_trials=200\nNote: the web runtime also tags the meaning context with ::rev",
            GameKind::SpotXY => "Base stimulus: spotxy (context)\nSensors: pos_x_00..pos_x_15 and pos_y_00..pos_y_15 (population code)\nStimulus key: spotxy_xbin_XX or spotxy_bin_NN_IX_IY\nActions: left/right (BinaryX) OR spotxy_cell_NN_IX_IY (Grid)\nEval mode: holdout band |x| in [0.25..0.45] with learning suppressed",
            GameKind::Pong => "Base stimulus: pong (context)\nSensors: pong_ball_x_00..07, pong_ball_y_00..07, pong_paddle_y_00..07, pong_vx_pos/neg, pong_vy_pos/neg\nStimulus key: pong_b08_bx.._by.._py.._vx.._vy..\nActions: up, down, stay",
            GameKind::Sequence => "Base stimulus: sequence (context)\nSensors: seq_token_A/B/C and seq_regime_0/1\nStimulus key: seq_r{0|1}_t{A|B|C}\nActions: A, B, C",
            GameKind::Text => "Base stimulus: text (context)\nSensors: txt_regime_0/1 and txt_tok_XX (byte tokens) + txt_tok_UNK\nActions: tok_XX for bytes in vocab + tok_UNK",
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
                .ensure_sensor_min_width(&format!("pong_paddle_y_{i:02}"), 3);
        }
        self.brain.ensure_sensor_min_width("pong_vx_pos", 2);
        self.brain.ensure_sensor_min_width("pong_vx_neg", 2);
        self.brain.ensure_sensor_min_width("pong_vy_pos", 2);
        self.brain.ensure_sensor_min_width("pong_vy_neg", 2);

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
        } else if let Some(k) = self.game.stimulus_key() {
            Some(k.to_string())
        } else {
            None
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

enum WebGame {
    Spot(SpotGame),
    Bandit(BanditGame),
    SpotReversal(SpotReversalGame),
    SpotXY(SpotXYGame),
    Pong(PongWebGame),
    Sequence(SequenceWebGame),
    Text(TextWebGame),
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
        }
    }

    fn stimulus_key(&self) -> Option<&str> {
        match self {
            WebGame::SpotXY(g) => Some(g.stimulus_key()),
            WebGame::Pong(g) => Some(g.stimulus_key()),
            WebGame::Sequence(g) => Some(g.stimulus_key()),
            WebGame::Text(g) => Some(g.stimulus_key()),
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
    web_sys::window()
        .and_then(|w| w.local_storage().ok().flatten())
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

    known_sensors
        .first()
        .cloned()
        .unwrap_or_else(|| preferred)
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
) -> Result<(), String> {
    let ctx = canvas
        .get_context("2d")
        .map_err(|_| "canvas: get_context threw".to_string())?
        .ok_or("canvas: missing 2d context".to_string())?
        .dyn_into::<web_sys::CanvasRenderingContext2d>()
        .map_err(|_| "canvas: context is not 2d".to_string())?;

    let w = canvas.width() as f64;
    let h = canvas.height() as f64;

    let eff_grid = grid_n.max(1) as f64;

    // Background
    ctx.set_fill_style(&JsValue::from_str("#0a0f1a"));
    ctx.fill_rect(0.0, 0.0, w, h);

    // Grid (draw only in Grid mode; BinaryX draws no divider)
    let cell_w = w / eff_grid;
    let cell_h = h / eff_grid;
    if grid_n >= 2 {
        ctx.set_stroke_style(&JsValue::from_str("rgba(122, 162, 255, 0.20)"));
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
    }

    // Map x,y in [-1,1] to canvas coords.
    let px = ((x.clamp(-1.0, 1.0) as f64 + 1.0) * 0.5) * w;
    let py = (1.0 - (y.clamp(-1.0, 1.0) as f64 + 1.0) * 0.5) * h;

    // Highlight the active cell
    let cx = ((px / cell_w).floor()).clamp(0.0, eff_grid - 1.0);
    let cy = ((py / cell_h).floor()).clamp(0.0, eff_grid - 1.0);
    let cell_highlight = if accent == "#22c55e" {
        "rgba(34, 197, 94, 0.10)"
    } else {
        "rgba(122, 162, 255, 0.10)"
    };
    ctx.set_fill_style(&JsValue::from_str(cell_highlight));
    ctx.fill_rect(cx * cell_w, cy * cell_h, cell_w, cell_h);

    // Dot
    ctx.set_fill_style(&JsValue::from_str(accent));
    ctx.begin_path();
    let _ = ctx.arc(px, py, 6.0, 0.0, std::f64::consts::PI * 2.0);
    ctx.fill();
    Ok(())
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

    // Field border with glow
    ctx.set_stroke_style_str("rgba(122, 162, 255, 0.3)");
    ctx.set_line_width(2.0);
    ctx.stroke_rect(8.0, 8.0, w - 16.0, h - 16.0);

    // Top and bottom wall highlights
    ctx.set_stroke_style_str("rgba(122, 162, 255, 0.5)");
    ctx.set_line_width(3.0);
    ctx.begin_path();
    ctx.move_to(8.0, 8.0);
    ctx.line_to(w - 8.0, 8.0);
    ctx.stroke();
    ctx.begin_path();
    ctx.move_to(8.0, h - 8.0);
    ctx.line_to(w - 8.0, h - 8.0);
    ctx.stroke();

    // Map coordinates:
    // ball_x in [0,1] maps to [20, w-20]
    // ball_y, paddle_y in [-1,1] map to [20, h-20] (y downwards)
    let inner_w = (w - 40.0).max(1.0);
    let inner_h = (h - 40.0).max(1.0);
    let map_x = |x01: f32| 20.0 + (x01.clamp(0.0, 1.0) as f64) * inner_w;
    let map_y = |ys: f32| 20.0 + (1.0 - ((ys.clamp(-1.0, 1.0) as f64 + 1.0) * 0.5)) * inner_h;

    // Paddle with glow effect
    let paddle_x = 28.0;
    let paddle_w = 12.0;
    let paddle_y = map_y(s.paddle_y);
    let paddle_h = (s.paddle_half_height.clamp(0.01, 1.0) as f64) * 0.5 * inner_h;
    let paddle_height = (paddle_h * 2.0).min(inner_h);
    let paddle_top = (paddle_y - paddle_h).clamp(20.0, 20.0 + inner_h - paddle_height);
    
    // Paddle glow
    ctx.set_fill_style_str("rgba(122, 162, 255, 0.3)");
    ctx.begin_path();
    ctx.round_rect_with_f64(paddle_x - 4.0, paddle_top - 4.0, paddle_w + 8.0, paddle_height + 8.0, 6.0).ok();
    ctx.fill();
    
    // Paddle body
    ctx.set_fill_style_str("#7aa2ff");
    ctx.begin_path();
    ctx.round_rect_with_f64(paddle_x, paddle_top, paddle_w, paddle_height, 4.0).ok();
    ctx.fill();
    
    // Paddle highlight
    ctx.set_fill_style_str("rgba(255, 255, 255, 0.4)");
    ctx.fill_rect(paddle_x + 2.0, paddle_top + 2.0, 3.0, paddle_height - 4.0);

    // Ball with trail and glow
    if s.ball_visible {
        let bx = map_x(s.ball_x);
        let by = map_y(s.ball_y);
        
        // Ball glow (outer)
        ctx.set_fill_style_str("rgba(251, 191, 36, 0.2)");
        ctx.begin_path();
        let _ = ctx.arc(bx, by, 16.0, 0.0, std::f64::consts::PI * 2.0);
        ctx.fill();
        
        // Ball glow (inner)
        ctx.set_fill_style_str("rgba(251, 191, 36, 0.4)");
        ctx.begin_path();
        let _ = ctx.arc(bx, by, 10.0, 0.0, std::f64::consts::PI * 2.0);
        ctx.fill();
        
        // Ball body
        ctx.set_fill_style_str("#fbbf24");
        ctx.begin_path();
        let _ = ctx.arc(bx, by, 7.0, 0.0, std::f64::consts::PI * 2.0);
        ctx.fill();
        
        // Ball highlight
        ctx.set_fill_style_str("rgba(255, 255, 255, 0.6)");
        ctx.begin_path();
        let _ = ctx.arc(bx - 2.0, by - 2.0, 3.0, 0.0, std::f64::consts::PI * 2.0);
        ctx.fill();
    }

    // Score zone indicator (right edge)
    ctx.set_fill_style_str("rgba(239, 68, 68, 0.15)");
    ctx.fill_rect(w - 20.0, 8.0, 12.0, h - 16.0);

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
