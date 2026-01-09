use braine::substrate::{Brain, BrainConfig, Diagnostics, Stimulus};
use braine_games::{bandit::BanditGame, spot::SpotGame};
use leptos::prelude::*;
use wasm_bindgen::prelude::*;

const STORAGE_KEY_HEX: &str = "braine.brain_image_hex.v1";

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

    let (interval_id, set_interval_id) = signal::<Option<i32>>(None);

    let refresh_ui_from_runtime = {
        let runtime = runtime.clone();
        move || {
            set_diag.set(runtime.with_value(|r| r.brain.diagnostics()));
            set_trials.set(runtime.with_value(|r| r.game.stats().trials));
            set_recent_rate.set(runtime.with_value(|r| r.game.stats().last_100_rate()));
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

    let do_save = move || {
        let res = runtime.with_value(|r| r.brain.save_image_bytes());
        match res {
            Ok(bytes) => {
                let hex = hex_encode(&bytes);
                match local_storage_set(STORAGE_KEY_HEX, &hex) {
                    Ok(()) => set_status.set(format!(
                        "saved {} bytes ({} chars) to localStorage",
                        bytes.len(),
                        hex.len()
                    )),
                    Err(e) => set_status.set(format!("save failed: {e}")),
                }
            }
            Err(e) => set_status.set(format!("save failed: {e}")),
        }
    };

    let do_load = move || match local_storage_get(STORAGE_KEY_HEX) {
        Ok(Some(hex)) => match hex_decode(&hex) {
            Ok(bytes) => match Brain::load_image_bytes(&bytes) {
                Ok(brain) => {
                    runtime.update_value(|r| r.brain = brain);
                    set_steps.set(0);
                    set_last_action.set(String::new());
                    set_last_reward.set(0.0);
                    refresh_ui_from_runtime();
                    set_status.set(format!("loaded {} bytes from localStorage", bytes.len()));
                }
                Err(e) => set_status.set(format!("load failed: {e}")),
            },
            Err(e) => set_status.set(format!("load failed: {e}")),
        },
        Ok(None) => set_status.set("no saved brain image in localStorage".to_string()),
        Err(e) => set_status.set(format!("load failed: {e}")),
    };

    view! {
        <main style="font-family: system-ui, -apple-system, Segoe UI, Roboto, sans-serif; padding: 18px; max-width: 880px; margin: 0 auto;">
            <h1 style="margin: 0 0 8px 0;">"braine_web"</h1>
            <p style="margin: 0 0 16px 0; color: #555;">
                "Fully in-browser: Leptos CSR + in-process Brain + shared games."
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
                    "Save (localStorage)"
                </button>
                <button on:click=move |_| do_load()>
                    "Load (localStorage)"
                </button>
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
                            }
                        }
                    >
                        <option value=GameKind::Spot.label()>"spot"</option>
                        <option value=GameKind::Bandit.label()>"bandit"</option>
                    </select>
                </label>

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

            <p style="margin-top: 16px; color: #777; font-size: 0.95em;">
                "Next: add canvas rendering and switch persistence to IndexedDB."
            </p>
        </main>
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum GameKind {
    Spot,
    Bandit,
}

impl GameKind {
    fn label(self) -> &'static str {
        match self {
            GameKind::Spot => "spot",
            GameKind::Bandit => "bandit",
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
        };
        self.pending_neuromod = 0.0;
    }

    fn tick(&mut self, cfg: &TickConfig) -> Option<TickOutput> {
        self.game.update_timing(cfg.trial_period_ms);

        // Apply last reward as neuromodulation for one step.
        self.brain.set_neuromodulator(self.pending_neuromod);
        self.pending_neuromod = 0.0;

        let stimulus = self.game.stimulus_name();
        let context_key = stimulus;

        self.brain.apply_stimulus(Stimulus::new(stimulus, 1.0));
        self.brain.note_compound_symbol(&[context_key]);
        self.brain.step();

        if self.game.response_made() {
            self.brain.set_neuromodulator(0.0);
            self.brain.commit_observation();
            return None;
        }

        let allowed = self.game.allowed_actions();
        let explore = self.rng_next_f32() < cfg.exploration_eps;

        let action = if explore {
            allowed[(self.rng_next_u64() as usize) % allowed.len()].to_string()
        } else {
            // Pick the best allowed action by meaning+habit.
            let ranked = self
                .brain
                .ranked_actions_with_meaning(context_key, cfg.meaning_alpha);
            ranked
                .into_iter()
                .find(|(name, _score)| allowed.iter().any(|a| a == name))
                .map(|(name, _score)| name)
                .or_else(|| allowed.first().map(|s| s.to_string()))
                .unwrap_or_else(|| "left".to_string())
        };

        let (reward, _done) = match self.game.score_action(&action) {
            Some((r, done)) => (r, done),
            None => (0.0, false),
        };

        self.brain.note_action(&action);
        self.brain
            .note_compound_symbol(&["pair", context_key, action.as_str()]);

        self.brain.set_neuromodulator(reward);
        self.brain.reinforce_action(&action, reward);
        self.pending_neuromod = reward;

        self.brain.commit_observation();

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
}

impl WebGame {
    fn stimulus_name(&self) -> &'static str {
        match self {
            WebGame::Spot(g) => g.stimulus_name(),
            WebGame::Bandit(g) => g.stimulus_name(),
        }
    }

    fn allowed_actions(&self) -> [&'static str; 2] {
        ["left", "right"]
    }

    fn response_made(&self) -> bool {
        match self {
            WebGame::Spot(g) => g.response_made,
            WebGame::Bandit(g) => g.response_made,
        }
    }

    fn update_timing(&mut self, trial_period_ms: u32) {
        match self {
            WebGame::Spot(g) => g.update_timing(trial_period_ms),
            WebGame::Bandit(g) => g.update_timing(trial_period_ms),
        }
    }

    fn score_action(&mut self, action: &str) -> Option<(f32, bool)> {
        match self {
            WebGame::Spot(g) => g.score_action(action),
            WebGame::Bandit(g) => g.score_action(action),
        }
    }

    fn stats(&self) -> &braine_games::stats::GameStats {
        match self {
            WebGame::Spot(g) => &g.stats,
            WebGame::Bandit(g) => &g.stats,
        }
    }
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
    brain.define_sensor("spot_left", 6);
    brain.define_sensor("spot_right", 6);
    brain.define_sensor("bandit", 6);

    brain
}

fn local_storage_get(key: &str) -> Result<Option<String>, String> {
    let w = web_sys::window().ok_or("no window")?;
    let storage = w.local_storage().map_err(|_| "local_storage() threw")?;
    let Some(storage) = storage else {
        return Err("localStorage unavailable".to_string());
    };
    storage
        .get_item(key)
        .map_err(|_| "get_item() threw".to_string())
}

fn local_storage_set(key: &str, value: &str) -> Result<(), String> {
    let w = web_sys::window().ok_or("no window")?;
    let storage = w.local_storage().map_err(|_| "local_storage() threw")?;
    let Some(storage) = storage else {
        return Err("localStorage unavailable".to_string());
    };
    storage
        .set_item(key, value)
        .map_err(|_| "set_item() threw".to_string())
}

fn hex_encode(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut out = String::with_capacity(bytes.len() * 2);
    for &b in bytes {
        out.push(HEX[(b >> 4) as usize] as char);
        out.push(HEX[(b & 0x0f) as usize] as char);
    }
    out
}

fn hex_decode(hex: &str) -> Result<Vec<u8>, String> {
    let hex = hex.trim();
    if hex.len() % 2 != 0 {
        return Err("hex string must have even length".to_string());
    }

    let mut out = Vec::with_capacity(hex.len() / 2);
    let bytes = hex.as_bytes();
    for i in (0..bytes.len()).step_by(2) {
        let hi = from_hex_digit(bytes[i]).ok_or("invalid hex digit")?;
        let lo = from_hex_digit(bytes[i + 1]).ok_or("invalid hex digit")?;
        out.push((hi << 4) | lo);
    }
    Ok(out)
}

fn from_hex_digit(b: u8) -> Option<u8> {
    match b {
        b'0'..=b'9' => Some(b - b'0'),
        b'a'..=b'f' => Some(b - b'a' + 10),
        b'A'..=b'F' => Some(b - b'A' + 10),
        _ => None,
    }
}
