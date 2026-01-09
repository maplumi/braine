use braine::substrate::{ActionPolicy, Brain, BrainConfig, Diagnostics, Stimulus};
use leptos::prelude::*;
use wasm_bindgen::prelude::*;

const STORAGE_KEY_HEX: &str = "braine.brain_image_hex.v1";

#[wasm_bindgen(start)]
pub fn start() {
    mount_to_body(|| view! { <App /> });
}

#[component]
fn App() -> impl IntoView {
    let brain = StoredValue::new(make_default_brain());

    let (steps, set_steps) = signal(0u64);
    let (diag, set_diag) = signal(brain.with_value(|b| b.diagnostics()));
    let (last_action, set_last_action) = signal(String::new());
    let (status, set_status) = signal(String::new());

    let do_step = move || {
        brain.update_value(|b| {
            // Tiny demo loop: apply one of two stimuli, step dynamics, pick action.
            // This is intentionally minimal scaffolding; we'll replace with a proper
            // UI-driven environment loop once the web shell is in place.
            let t = steps.get_untracked();
            let (stim, reward) = if (t / 40) % 2 == 0 {
                ("vision_food", 0.6)
            } else {
                ("vision_threat", -0.6)
            };

            b.apply_stimulus(Stimulus::new(stim, 1.0));
            b.set_neuromodulator(reward);
            b.step();

            let mut policy = ActionPolicy::EpsilonGreedy { epsilon: 0.10 };
            let (a, _score) = b.select_action(&mut policy);
            set_last_action.set(a);
        });

        set_steps.update(|s| *s += 1);
        set_diag.set(brain.with_value(|b| b.diagnostics()));
    };

    let do_reset = move || {
        brain.set_value(make_default_brain());
        set_steps.set(0);
        set_last_action.set(String::new());
        set_diag.set(brain.with_value(|b| b.diagnostics()));
        set_status.set("reset".to_string());
    };

    let do_save = move || {
        let res = brain.with_value(|b| b.save_image_bytes());
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
                Ok(b) => {
                    let d = b.diagnostics();
                    brain.set_value(b);
                    set_steps.set(0);
                    set_last_action.set(String::new());
                    set_diag.set(d);
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
                "Leptos CSR shell with an in-process Brain. This is the starting point for the real web UI."
            </p>

            <section style="display: flex; gap: 10px; flex-wrap: wrap; margin-bottom: 14px;">
                <button on:click=move |_| do_step()>
                    "Step"
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

            <section style="display: grid; grid-template-columns: 1fr; gap: 8px;">
                <Stat label="Steps" value=move || steps.get().to_string() />
                <Stat label="Last action" value=move || {
                    let a = last_action.get();
                    if a.is_empty() { "(none)".to_string() } else { a }
                } />
                <Diag diag=move || diag.get() />
                <Stat label="Status" value=move || status.get() />
            </section>

            <p style="margin-top: 16px; color: #777; font-size: 0.95em;">
                "Next: add real game loop + canvas, and move persistence to IndexedDB (" <code>"Brain::save_image_bytes"</code> "+ " <code>"Brain::load_image_bytes"</code> ")."
            </p>
        </main>
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

    brain.define_action("approach", 6);
    brain.define_action("avoid", 6);
    brain.define_action("idle", 6);

    brain.define_sensor("vision_food", 6);
    brain.define_sensor("vision_threat", 6);

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
