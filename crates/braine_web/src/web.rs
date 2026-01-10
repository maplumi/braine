use braine::substrate::{Brain, BrainConfig, Diagnostics, Stimulus};
use braine_games::{
    bandit::BanditGame, spot::SpotGame, spot_reversal::SpotReversalGame, spot_xy::SpotXYGame,
};
use leptos::prelude::*;
use leptos::task::spawn_local;
mod pong_web;
use pong_web::PongWebGame;
use wasm_bindgen::closure::Closure;
use wasm_bindgen::prelude::*;
use wasm_bindgen::JsCast;

const IDB_DB_NAME: &str = "braine";
const IDB_STORE: &str = "kv";
const IDB_KEY_BRAIN_IMAGE: &str = "brain_image";

#[wasm_bindgen(start)]
pub fn start() {
    mount_to_body(|| view! { <App /> });
}

#[component]
fn App() -> impl IntoView {
    let runtime = StoredValue::new(AppRuntime::new());

    let (steps, set_steps) = signal(0u64);
    let (diag, set_diag) = signal(runtime.with_value(|r| r.brain.diagnostics()));

    let (game_kind, set_game_kind) = signal(GameKind::Spot);
    let (trial_period_ms, set_trial_period_ms) = signal(500u32);
    let (exploration_eps, set_exploration_eps) = signal(0.08f32);
    let (meaning_alpha, set_meaning_alpha) = signal(6.0f32);

    let (last_action, set_last_action) = signal(String::new());
    let (last_reward, set_last_reward) = signal(0.0f32);

    let (trials, set_trials) = signal(0u32);
    let (recent_rate, set_recent_rate) = signal(0.5f32);
    let (status, set_status) = signal(String::new());

    let (spotxy_pos, set_spotxy_pos) = signal::<Option<(f32, f32)>>(None);
    let (spotxy_stimulus_key, set_spotxy_stimulus_key) = signal(String::new());
    let (spotxy_eval, set_spotxy_eval) = signal(false);
    let (spotxy_mode, set_spotxy_mode) = signal(String::new());
    let (spotxy_grid_n, set_spotxy_grid_n) = signal(0u32);

    let (pong_state, set_pong_state) = signal::<Option<PongUiState>>(None);
    let (pong_stimulus_key, set_pong_stimulus_key) = signal(String::new());
    let (pong_paddle_speed, set_pong_paddle_speed) = signal(0.0f32);
    let (pong_paddle_half_height, set_pong_paddle_half_height) = signal(0.0f32);
    let (pong_ball_speed, set_pong_ball_speed) = signal(0.0f32);

    let (reversal_active, set_reversal_active) = signal(false);
    let (reversal_flip_after, set_reversal_flip_after) = signal(0u32);

    let (import_autosave, set_import_autosave) = signal(true);

    let (interval_id, set_interval_id) = signal::<Option<i32>>(None);

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
            set_trials.set(runtime.with_value(|r| r.game.stats().trials));
            set_recent_rate.set(runtime.with_value(|r| r.game.stats().last_100_rate()));

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
        }
    };

    let set_game = {
        let runtime = runtime.clone();
        move |kind: GameKind| {
            runtime.update_value(|r| r.set_game(kind));
            set_game_kind.set(kind);
            set_steps.set(0);
            set_last_action.set(String::new());
            set_last_reward.set(0.0);
            refresh_ui_from_runtime();
            set_status.set(format!("game set: {}", kind.label()));
        }
    };

    let do_tick = {
        let runtime = runtime.clone();
        move || {
            let cfg = TickConfig {
                trial_period_ms: trial_period_ms.get_untracked(),
                exploration_eps: exploration_eps.get_untracked(),
                meaning_alpha: meaning_alpha.get_untracked(),
            };

            let mut out: Option<TickOutput> = None;
            runtime.update_value(|r| {
                out = r.tick(&cfg);
            });
            if let Some(out) = out {
                set_last_action.set(out.last_action);
                set_last_reward.set(out.reward);
            }

            set_steps.update(|s| *s += 1);
            refresh_ui_from_runtime();
        }
    };

    let do_reset = move || {
        runtime.set_value(AppRuntime::new());
        set_steps.set(0);
        set_last_action.set(String::new());
        set_last_reward.set(0.0);
        refresh_ui_from_runtime();
        set_status.set("reset".to_string());
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

            match window.set_interval_with_callback_and_timeout_and_arguments_0(
                cb.as_ref().unchecked_ref(),
                33,
            ) {
                Ok(id) => {
                    cb.forget();
                    set_interval_id.set(Some(id));
                    set_status.set("running".to_string());
                }
                Err(_) => set_status.set("failed to start interval".to_string()),
            }
        }
    };

    let do_stop = move || {
        if let Some(id) = interval_id.get_untracked() {
            if let Some(w) = web_sys::window() {
                w.clear_interval_with_handle(id);
            }
            set_interval_id.set(None);
            set_status.set("stopped".to_string());
        }
    };

    on_cleanup({
        let do_stop = do_stop.clone();
        move || do_stop()
    });

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

            let set_status = set_status.clone();
            spawn_local(async move {
                match idb_put_bytes(IDB_KEY_BRAIN_IMAGE, &bytes).await {
                    Ok(()) => set_status.set(format!("saved {} bytes to IndexedDB", bytes.len())),
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

    let do_export_bbi_from_idb = move || {
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
        let pos = spotxy_pos.get();
        let Some(canvas) = canvas_ref.get() else {
            return;
        };
        match pos {
            Some((x, y)) => {
                let _ = draw_spotxy(&canvas, x, y);
            }
            None => {
                let _ = clear_canvas(&canvas);
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

    view! {
        <main style="font-family: system-ui, -apple-system, Segoe UI, Roboto, sans-serif; padding: 18px; max-width: 880px; margin: 0 auto;">
            <h1 style="margin: 0 0 8px 0;">"braine_web"</h1>
            <p style="margin: 0 0 4px 0; color: #555;">
                "Fully in-browser: Leptos CSR + in-process Brain + shared games."
            </p>
            <p style="margin: 0 0 16px 0; color: #777; font-size: 0.9em;">
                {move || gpu_status.get()}
            </p>

            <section style="display: flex; gap: 10px; flex-wrap: wrap; margin-bottom: 14px;">
                <button on:click=move |_| do_tick()>
                    "Step"
                </button>
                <button on:click=move |_| do_start()>
                    "Run"
                </button>
                <button on:click=move |_| do_stop()>
                    "Stop"
                </button>
                <button on:click=move |_| do_reset()>
                    "Reset"
                </button>
                <button on:click=move |_| do_save()>
                    "Save (IndexedDB)"
                </button>
                <button on:click=move |_| do_load()>
                    "Load (IndexedDB)"
                </button>
                <button on:click=move |_| do_export_bbi()>
                    "Export (.bbi)"
                </button>
                <button on:click=move |_| do_export_bbi_from_idb()>
                    "Export (IndexedDB .bbi)"
                </button>
                <button on:click=move |_| do_import_bbi_click()>
                    "Import (.bbi)"
                </button>
                <label style="display: flex; gap: 8px; align-items: center; color: #333;">
                    <input
                        type="checkbox"
                        prop:checked=move || import_autosave.get()
                        on:change=move |ev| {
                            let v = event_target_checked(&ev);
                            set_import_autosave.set(v);
                        }
                    />
                    <span>"Auto-save import → IndexedDB"</span>
                </label>
                <input
                    node_ref=import_input_ref
                    type="file"
                    accept=".bbi,application/octet-stream"
                    style="display: none;"
                    on:change=do_import_bbi_change
                />
            </section>

            <section style="display: flex; gap: 10px; flex-wrap: wrap; margin-bottom: 14px; align-items: center;">
                <label style="display: flex; gap: 8px; align-items: center;">
                    <span style="min-width: 78px; color: #333;">"Game"</span>
                    <select
                        prop:value=move || game_kind.get().label().to_string()
                        on:change=move |ev| {
                            let v = event_target_value(&ev);
                            if v == GameKind::Spot.label() {
                                set_game(GameKind::Spot);
                            } else if v == GameKind::Bandit.label() {
                                set_game(GameKind::Bandit);
                            } else if v == GameKind::SpotReversal.label() {
                                set_game(GameKind::SpotReversal);
                            } else if v == GameKind::SpotXY.label() {
                                set_game(GameKind::SpotXY);
                            } else if v == GameKind::Pong.label() {
                                set_game(GameKind::Pong);
                            }
                        }
                    >
                        <option value=GameKind::Spot.label()>"spot"</option>
                        <option value=GameKind::Bandit.label()>"bandit"</option>
                        <option value=GameKind::SpotReversal.label()>"spot_reversal"</option>
                        <option value=GameKind::SpotXY.label()>"spotxy"</option>
                        <option value=GameKind::Pong.label()>"pong"</option>
                    </select>
                </label>

                <Show when=move || game_kind.get() == GameKind::SpotReversal>
                    <span style="color: #333;">
                        {move || format!(
                            "reversal_active={} flip_after={}",
                            reversal_active.get(),
                            reversal_flip_after.get()
                        )}
                    </span>
                </Show>

                <Show when=move || game_kind.get() == GameKind::SpotXY>
                    <button on:click=move |_| do_spotxy_grid_minus()>
                        "Grid -"
                    </button>
                    <button on:click=move |_| do_spotxy_grid_plus()>
                        "Grid +"
                    </button>
                    <button on:click=move |_| do_spotxy_toggle_eval()>
                        {move || if spotxy_eval.get() { "Eval: ON" } else { "Eval: OFF" }}
                    </button>
                    <span style="color: #333;">
                        {move || {
                            let mode = spotxy_mode.get();
                            if mode.is_empty() {
                                "".to_string()
                            } else if mode == "grid" {
                                format!("mode=grid n={}", spotxy_grid_n.get())
                            } else {
                                format!("mode={mode}")
                            }
                        }}
                    </span>
                </Show>

                <label style="display: flex; gap: 8px; align-items: center;">
                    <span style="min-width: 78px; color: #333;">"Trial ms"</span>
                    <input
                        type="number"
                        min="10"
                        max="60000"
                        prop:value=move || trial_period_ms.get().to_string()
                        on:input=move |ev| {
                            let v = event_target_value(&ev);
                            if let Ok(n) = v.parse::<u32>() {
                                set_trial_period_ms.set(n.clamp(10, 60_000));
                            }
                        }
                    />
                </label>

                <label style="display: flex; gap: 8px; align-items: center;">
                    <span style="min-width: 78px; color: #333;">"ε"</span>
                    <input
                        type="number"
                        min="0"
                        max="1"
                        step="0.01"
                        prop:value=move || format!("{:.2}", exploration_eps.get())
                        on:input=move |ev| {
                            let v = event_target_value(&ev);
                            if let Ok(x) = v.parse::<f32>() {
                                set_exploration_eps.set(x.clamp(0.0, 1.0));
                            }
                        }
                    />
                </label>

                <label style="display: flex; gap: 8px; align-items: center;">
                    <span style="min-width: 78px; color: #333;">"α"</span>
                    <input
                        type="number"
                        min="0"
                        max="30"
                        step="0.5"
                        prop:value=move || format!("{:.1}", meaning_alpha.get())
                        on:input=move |ev| {
                            let v = event_target_value(&ev);
                            if let Ok(x) = v.parse::<f32>() {
                                set_meaning_alpha.set(x.clamp(0.0, 30.0));
                            }
                        }
                    />
                </label>
            </section>

            <Show when=move || game_kind.get() == GameKind::SpotXY>
                <section style="margin: 10px 0 14px 0; display: grid; gap: 8px;">
                    <div style="color: #444;">
                        {move || {
                            let k = spotxy_stimulus_key.get();
                            if k.is_empty() {
                                "".to_string()
                            } else {
                                format!("stimulus_key={k}")
                            }
                        }}
                    </div>
                    <canvas
                        node_ref=canvas_ref
                        width="260"
                        height="260"
                        style="border: 1px solid #eee; border-radius: 12px; background: #fff;"
                    ></canvas>
                </section>
            </Show>

            <Show when=move || game_kind.get() == GameKind::Pong>
                <section style="margin: 10px 0 14px 0; display: grid; gap: 8px;">
                    <div style="color: #444;">
                        {move || {
                            let k = pong_stimulus_key.get();
                            if k.is_empty() {
                                "".to_string()
                            } else {
                                format!("stimulus_key={k}")
                            }
                        }}
                    </div>
                    <div style="color: #444;">
                        {move || {
                            pong_state
                                .get()
                                .map(|s| {
                                    format!(
                                        "ball_visible={} paddle_y={:.2} ball=({:.2},{:.2})",
                                        s.ball_visible, s.paddle_y, s.ball_x, s.ball_y
                                    )
                                })
                                .unwrap_or_default()
                        }}
                    </div>

                    <section style="display: flex; gap: 10px; flex-wrap: wrap; align-items: center;">
                        <label style="display: flex; gap: 8px; align-items: center;">
                            <span style="min-width: 140px; color: #333;">"Paddle speed"</span>
                            <input
                                type="number"
                                min="0.1"
                                max="5"
                                step="0.1"
                                prop:value=move || format!("{:.2}", pong_paddle_speed.get())
                                on:input=move |ev| {
                                    let v = event_target_value(&ev);
                                    if let Ok(x) = v.parse::<f32>() {
                                        do_pong_set_param("paddle_speed", x);
                                    }
                                }
                            />
                        </label>
                        <label style="display: flex; gap: 8px; align-items: center;">
                            <span style="min-width: 140px; color: #333;">"Paddle half-height"</span>
                            <input
                                type="number"
                                min="0.05"
                                max="0.9"
                                step="0.01"
                                prop:value=move || format!("{:.2}", pong_paddle_half_height.get())
                                on:input=move |ev| {
                                    let v = event_target_value(&ev);
                                    if let Ok(x) = v.parse::<f32>() {
                                        do_pong_set_param("paddle_half_height", x);
                                    }
                                }
                            />
                        </label>
                        <label style="display: flex; gap: 8px; align-items: center;">
                            <span style="min-width: 140px; color: #333;">"Ball speed"</span>
                            <input
                                type="number"
                                min="0.1"
                                max="3"
                                step="0.1"
                                prop:value=move || format!("{:.2}", pong_ball_speed.get())
                                on:input=move |ev| {
                                    let v = event_target_value(&ev);
                                    if let Ok(x) = v.parse::<f32>() {
                                        do_pong_set_param("ball_speed", x);
                                    }
                                }
                            />
                        </label>
                    </section>

                    <canvas
                        node_ref=pong_canvas_ref
                        width="360"
                        height="220"
                        style="border: 1px solid #eee; border-radius: 12px; background: #fff;"
                    ></canvas>
                </section>
            </Show>

            <section style="display: grid; grid-template-columns: 1fr; gap: 8px;">
                <Stat label="Steps" value=move || steps.get().to_string() />
                <Stat label="Trials" value=move || trials.get().to_string() />
                <Stat label="Recent (last ~100)" value=move || format!("{:.3}", recent_rate.get()) />
                <Stat label="Last action" value=move || {
                    let a = last_action.get();
                    if a.is_empty() { "(none)".to_string() } else { a }
                } />
                <Stat label="Last reward" value=move || format!("{:+.3}", last_reward.get()) />
                <Diag diag=move || diag.get() />
                <Stat label="Status" value=move || status.get() />
            </section>

            // About section
            <details style="margin-top: 24px; border: 1px solid #ddd; border-radius: 8px; padding: 16px;">
                <summary style="font-size: 1.1em; font-weight: 600; cursor: pointer; color: #333;">"About Braine"</summary>
                <div style="margin-top: 16px; color: #444; line-height: 1.7;">
                    <h3 style="margin: 0 0 12px 0; color: #222;">"What is Braine?"</h3>
                    <p style="margin: 0 0 12px 0;">
                        "Braine is a closed-loop learning substrate—a fundamentally different approach to artificial intelligence. Unlike Large Language Models (LLMs) that rely on backpropagation and massive datasets, Braine uses sparse recurrent dynamics combined with local plasticity rules and scalar reward signals (neuromodulators). There are no gradients, no backprop—just real-time, online learning."
                    </p>

                    <h3 style="margin: 16px 0 12px 0; color: #222;">"How It Works Internally"</h3>
                    <p style="margin: 0 0 8px 0;">
                        <strong>"Sparse Recurrent Substrate: "</strong>"A network of units with sparse connectivity. Activity propagates through recurrent pathways, allowing temporal integration and context-dependent responses."
                    </p>
                    <p style="margin: 0 0 8px 0;">
                        <strong>"Local Plasticity (Hebbian-style): "</strong>"Connections strengthen when pre- and post-synaptic units co-activate. Learning is local—no global error signal backpropagated through layers."
                    </p>
                    <p style="margin: 0 0 8px 0;">
                        <strong>"Neuromodulator/Reward: "</strong>"A scalar signal that gates plasticity. Positive reward reinforces recent activity patterns; negative reward weakens them. This enables credit assignment without gradients."
                    </p>
                    <p style="margin: 0 0 8px 0;">
                        <strong>"Meaning & Causality Memory: "</strong>"Braine builds semantic associations (meaning memory) and tracks causal relationships between actions and outcomes, enabling context-conditioned behavior."
                    </p>

                    <h3 style="margin: 16px 0 12px 0; color: #222;">"Research Gaps Addressed"</h3>
                    <p style="margin: 0 0 8px 0;">
                        <strong>"1. Online Learning: "</strong>"Most deep learning requires offline batch training. Braine learns continuously from streaming experience."
                    </p>
                    <p style="margin: 0 0 8px 0;">
                        <strong>"2. Embodied/Edge AI: "</strong>"LLMs require cloud infrastructure. Braine's lightweight substrate runs on microcontrollers and in-browser (WASM)."
                    </p>
                    <p style="margin: 0 0 8px 0;">
                        <strong>"3. Sample Efficiency: "</strong>"Learns meaningful behaviors from small numbers of trials, unlike RL methods requiring millions of samples."
                    </p>
                    <p style="margin: 0 0 8px 0;">
                        <strong>"4. Interpretability: "</strong>"Explicit symbol/action associations and causal memory provide transparency into decision-making."
                    </p>

                    <h3 style="margin: 16px 0 12px 0; color: #222;">"Similar Systems"</h3>
                    <p style="margin: 0 0 12px 0;">
                        "Braine shares principles with Hierarchical Temporal Memory (HTM), Spiking Neural Networks (SNNs), Reservoir Computing, and neuromodulated plasticity models from computational neuroscience. It differs by combining sparse dynamics, local Hebbian learning, and explicit semantic memory in a unified, practical substrate."
                    </p>

                    <h3 style="margin: 16px 0 12px 0; color: #222;">"Integration with LLMs"</h3>
                    <p style="margin: 0 0 8px 0;">
                        <strong>"Braine → LLM: "</strong>"Braine's low-level sensorimotor policies and learned action rankings can be used as grounded primitives for LLM-based planners. LLMs provide high-level reasoning; Braine provides embodied skills."
                    </p>
                    <p style="margin: 0 0 12px 0;">
                        <strong>"LLM → Braine: "</strong>"LLM outputs (instructions, goals) can serve as contextual stimuli for Braine, allowing natural language to modulate behavior without retraining the substrate."
                    </p>

                    <h3 style="margin: 16px 0 12px 0; color: #222;">"Novel Use Cases (Edge Focus)"</h3>
                    <p style="margin: 0 0 8px 0;">
                        <strong>"• Adaptive Robotics: "</strong>"Small robots that learn motor coordination on-device without cloud connectivity—adjusting to damage or terrain in real-time."
                    </p>
                    <p style="margin: 0 0 8px 0;">
                        <strong>"• Wearable Health Monitors: "</strong>"Learn user-specific baselines for anomaly detection (heart rate, gait) with privacy-preserving on-device learning."
                    </p>
                    <p style="margin: 0 0 8px 0;">
                        <strong>"• Smart Home Automation: "</strong>"Light/thermostat controllers that adapt to occupant preferences without sending data to the cloud."
                    </p>
                    <p style="margin: 0 0 8px 0;">
                        <strong>"• Agricultural Sensors: "</strong>"Field-deployed devices learning local soil/weather patterns for irrigation control with minimal power."
                    </p>
                    <p style="margin: 0 0 8px 0;">
                        <strong>"• Interactive Toys/Games: "</strong>"Characters that genuinely learn player preferences and adapt difficulty in-device."
                    </p>

                    <hr style="margin: 20px 0; border: none; border-top: 1px solid #ddd;" />
                    <p style="margin: 0; font-size: 0.9em; color: #666;">
                        <strong>"© Maplumi Labs"</strong>" | Developer: Elvis Ayiemba"
                    </p>
                    <p style="margin: 4px 0 0 0; font-size: 0.85em; color: #888;">
                        "This is a research project exploring alternatives to gradient-based learning for embodied and edge AI."
                    </p>
                </div>
            </details>

            <p style="margin-top: 16px; color: #777; font-size: 0.95em;">
                "Persistence: IndexedDB (browser-local)."
            </p>
        </main>
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum GameKind {
    Spot,
    Bandit,
    SpotReversal,
    SpotXY,
    Pong,
}

impl GameKind {
    fn label(self) -> &'static str {
        match self {
            GameKind::Spot => "spot",
            GameKind::Bandit => "bandit",
            GameKind::SpotReversal => "spot_reversal",
            GameKind::SpotXY => "spotxy",
            GameKind::Pong => "pong",
        }
    }
}

struct TickConfig {
    trial_period_ms: u32,
    exploration_eps: f32,
    meaning_alpha: f32,
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

    fn ensure_spotxy_actions(&mut self, g: &SpotXYGame) {
        self.brain.ensure_action_min_width("left", 6);
        self.brain.ensure_action_min_width("right", 6);
        for name in g.allowed_actions() {
            self.brain.ensure_action_min_width(name, 6);
        }
    }

    fn spotxy_increase_grid(&mut self) {
        let actions = if let WebGame::SpotXY(g) = &mut self.game {
            g.increase_grid();
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
            g.decrease_grid();
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
        let allow_learning = !self.game.spotxy_eval_mode();

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

        self.brain.note_action(&action);
        self.brain
            .note_compound_symbol(&["pair", context_key, action.as_str()]);

        if allow_learning {
            self.brain.set_neuromodulator(reward);
            self.brain.reinforce_action(&action, reward);
            self.pending_neuromod = reward;
            self.brain.commit_observation();
        } else {
            self.brain.set_neuromodulator(0.0);
            self.pending_neuromod = 0.0;
            self.brain.discard_observation();
        }

        Some(TickOutput {
            last_action: action,
            reward,
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
}

impl WebGame {
    fn stimulus_name(&self) -> &'static str {
        match self {
            WebGame::Spot(g) => g.stimulus_name(),
            WebGame::Bandit(g) => g.stimulus_name(),
            WebGame::SpotReversal(g) => g.stimulus_name(),
            WebGame::SpotXY(g) => g.stimulus_name(),
            WebGame::Pong(g) => g.stimulus_name(),
        }
    }

    fn stimulus_key(&self) -> Option<&str> {
        match self {
            WebGame::SpotXY(g) => Some(g.stimulus_key()),
            WebGame::Pong(g) => Some(g.stimulus_key()),
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
        }
    }

    fn update_timing(&mut self, trial_period_ms: u32) {
        match self {
            WebGame::Spot(g) => g.update_timing(trial_period_ms),
            WebGame::Bandit(g) => g.update_timing(trial_period_ms),
            WebGame::SpotReversal(g) => g.update_timing(trial_period_ms),
            WebGame::SpotXY(g) => g.update_timing(trial_period_ms),
            WebGame::Pong(g) => g.update_timing(trial_period_ms),
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
        }
    }

    fn stats(&self) -> &braine_games::stats::GameStats {
        match self {
            WebGame::Spot(g) => &g.stats,
            WebGame::Bandit(g) => &g.stats,
            WebGame::SpotReversal(g) => &g.stats,
            WebGame::SpotXY(g) => &g.stats,
            WebGame::Pong(g) => &g.stats,
        }
    }

    fn ui_snapshot(&self) -> GameUiSnapshot {
        match self {
            WebGame::Spot(_) | WebGame::Bandit(_) => GameUiSnapshot::default(),
            WebGame::SpotReversal(g) => GameUiSnapshot {
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
        }
    }
}

#[derive(Default, Clone)]
struct GameUiSnapshot {
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
}

#[derive(Clone, Copy)]
struct PongUiState {
    ball_x: f32,
    ball_y: f32,
    paddle_y: f32,
    paddle_half_height: f32,
    ball_visible: bool,
}

#[component]
fn Stat(label: &'static str, value: impl Fn() -> String + Send + 'static) -> impl IntoView {
    view! {
        <div style="display: flex; justify-content: space-between; border: 1px solid #eee; padding: 10px 12px; border-radius: 10px;">
            <div style="color: #333; font-weight: 600;">{label}</div>
            <div style="color: #111; font-variant-numeric: tabular-nums;">{value}</div>
        </div>
    }
}

#[component]
fn Diag(diag: impl Fn() -> Diagnostics + Send + 'static) -> impl IntoView {
    let value = move || {
        let d = diag();
        format!(
            "units={} conns={} avg_amp={:.4} avg_w={:.5} mem≈{}KB tier={:?} pruned={} births={}",
            d.unit_count,
            d.connection_count,
            d.avg_amp,
            d.avg_weight,
            d.memory_bytes / 1024,
            d.execution_tier,
            d.pruned_last_step,
            d.births_last_step
        )
    };

    view! { <Stat label="Diagnostics" value=value /> }
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

    ctx.set_fill_style(&JsValue::from_str("#ffffff"));
    ctx.fill_rect(0.0, 0.0, canvas.width() as f64, canvas.height() as f64);
    Ok(())
}

#[allow(deprecated)]
fn draw_spotxy(canvas: &web_sys::HtmlCanvasElement, x: f32, y: f32) -> Result<(), String> {
    let ctx = canvas
        .get_context("2d")
        .map_err(|_| "canvas: get_context threw".to_string())?
        .ok_or("canvas: missing 2d context".to_string())?
        .dyn_into::<web_sys::CanvasRenderingContext2d>()
        .map_err(|_| "canvas: context is not 2d".to_string())?;

    let w = canvas.width() as f64;
    let h = canvas.height() as f64;

    // Background
    ctx.set_fill_style(&JsValue::from_str("#ffffff"));
    ctx.fill_rect(0.0, 0.0, w, h);

    // Axes
    ctx.set_stroke_style(&JsValue::from_str("#e6e6e6"));
    ctx.begin_path();
    ctx.move_to(w / 2.0, 0.0);
    ctx.line_to(w / 2.0, h);
    ctx.move_to(0.0, h / 2.0);
    ctx.line_to(w, h / 2.0);
    ctx.stroke();

    // Map x,y in [-1,1] to canvas coords.
    let px = ((x.clamp(-1.0, 1.0) as f64 + 1.0) * 0.5) * w;
    let py = (1.0 - (y.clamp(-1.0, 1.0) as f64 + 1.0) * 0.5) * h;

    // Dot
    ctx.set_fill_style(&JsValue::from_str("#111827"));
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

    // Background
    ctx.set_fill_style(&JsValue::from_str("#ffffff"));
    ctx.fill_rect(0.0, 0.0, w, h);

    // Field border
    ctx.set_stroke_style(&JsValue::from_str("#e5e7eb"));
    ctx.stroke_rect(8.0, 8.0, w - 16.0, h - 16.0);

    // Center line
    ctx.set_stroke_style(&JsValue::from_str("#f3f4f6"));
    ctx.begin_path();
    ctx.move_to(w / 2.0, 12.0);
    ctx.line_to(w / 2.0, h - 12.0);
    ctx.stroke();

    // Map coordinates:
    // ball_x in [0,1] maps to [12, w-12]
    // ball_y, paddle_y in [-1,1] map to [12, h-12] (y downwards)
    let inner_w = (w - 24.0).max(1.0);
    let inner_h = (h - 24.0).max(1.0);
    let map_x = |x01: f32| 12.0 + (x01.clamp(0.0, 1.0) as f64) * inner_w;
    let map_y = |ys: f32| 12.0 + (1.0 - ((ys.clamp(-1.0, 1.0) as f64 + 1.0) * 0.5)) * inner_h;

    // Paddle at left wall
    let paddle_x = 22.0;
    let paddle_w = 10.0;
    let paddle_y = map_y(s.paddle_y);
    let paddle_h = (s.paddle_half_height.clamp(0.01, 1.0) as f64) * 0.5 * inner_h;
    ctx.set_fill_style(&JsValue::from_str("#111827"));
    ctx.fill_rect(
        paddle_x,
        (paddle_y - paddle_h).clamp(12.0, h - 12.0),
        paddle_w,
        (paddle_h * 2.0).min(inner_h),
    );

    // Ball
    if s.ball_visible {
        let bx = map_x(s.ball_x);
        let by = map_y(s.ball_y);
        ctx.set_fill_style(&JsValue::from_str("#2563eb"));
        ctx.begin_path();
        let _ = ctx.arc(bx, by, 6.0, 0.0, std::f64::consts::PI * 2.0);
        ctx.fill();
    }

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
