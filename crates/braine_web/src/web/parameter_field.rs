use leptos::prelude::*;
use std::rc::Rc;

use super::settings_schema::{ParamSpec, RecommendedRange, Risk};

fn decimals_for_step(step: f32) -> usize {
    if step >= 1.0 {
        0
    } else if step >= 0.1 {
        1
    } else if step >= 0.01 {
        2
    } else if step >= 0.001 {
        3
    } else if step >= 0.0001 {
        4
    } else {
        6
    }
}

fn format_float(v: f32, step: f32) -> String {
    let d = decimals_for_step(step);
    // Keep it stable for typical numeric params; do not trim aggressively.
    match d {
        0 => format!("{v:.0}"),
        1 => format!("{v:.1}"),
        2 => format!("{v:.2}"),
        3 => format!("{v:.3}"),
        4 => format!("{v:.4}"),
        _ => format!("{v:.6}"),
    }
}

fn risk_label(r: Risk) -> &'static str {
    match r {
        Risk::Low => "Low risk",
        Risk::Medium => "Medium risk",
        Risk::High => "High risk",
    }
}

fn recommended_hint(rec: Option<RecommendedRange>) -> Option<String> {
    rec.map(|r| format!("Recommended: {:.4} – {:.4}", r.min, r.max))
}

#[component]
pub fn ParameterField(
    spec: ParamSpec,
    value: ReadSignal<f32>,
    set_value: WriteSignal<f32>,
    validity_map: RwSignal<std::collections::HashMap<String, bool>>,
) -> impl IntoView {
    let key = spec.key.to_string();
    let input_id = format!("param-{}", spec.key);
    let tip_id = format!("tip-{}", spec.key);

    let editing = RwSignal::new(false);
    let text = RwSignal::new(format_float(value.get_untracked(), spec.step));

    // Keep text in sync when external changes happen (reset / load), but do not
    // clobber while the user is typing.
    Effect::new({
        let spec = spec.clone();
        move |_| {
            let v = value.get();
            if !editing.get() {
                text.set(format_float(v, spec.step));
            }
        }
    });

    let set_valid: Rc<dyn Fn(bool)> = {
        let key = key.clone();
        Rc::new(move |ok: bool| {
            validity_map.update(|m| {
                m.insert(key.clone(), ok);
            });
        })
    };

    // Initialize validity as true once.
    Effect::new({
        let set_valid = Rc::clone(&set_valid);
        move |_| {
            set_valid(true);
        }
    });

    let warning_text = Memo::new({
        let spec = spec.clone();
        move |_| {
            let v = value.get();
            if v < spec.min || v > spec.max {
                return None;
            }
            let rec = spec.recommended?;
            if v < rec.min || v > rec.max {
                Some(match spec.risk {
                    Risk::High => "May destabilize learning",
                    Risk::Medium => "May slow convergence",
                    Risk::Low => "Non-standard value",
                })
            } else {
                None
            }
        }
    });

    let rec_hint = recommended_hint(spec.recommended);

    view! {
        <div class="param-field">
            <div class="param-label-row">
                <label class="param-label" for=input_id.clone()>
                    {spec.label}
                    {move || {
                        spec.units.map(|u| view!{ <span class="param-units">{format!(" ({u})")}</span> }).into_view()
                    }}
                </label>

                <span class="tooltip-wrap">
                    <button
                        type="button"
                        class="info-btn"
                        aria-label=format!("Info: {}", spec.label)
                        aria-describedby=tip_id.clone()
                    >
                        "i"
                    </button>
                    <div id=tip_id.clone() class="tooltip" role="tooltip">
                        <div class="tooltip-title">{spec.label}</div>
                        <div class="tooltip-body">{spec.description}</div>
                        <div class="tooltip-meta">
                            <div>{format!("Default: {}", format_float(spec.default, spec.step))}</div>
                            {rec_hint.clone().map(|s| view!{ <div>{s}</div> }).into_view()}
                            <div>{format!("Hard limits: {} – {}", format_float(spec.min, spec.step), format_float(spec.max, spec.step))}</div>
                            <div class="tooltip-when">{format!("When to change: {}", spec.when_to_change)}</div>
                            <div class="tooltip-risk">{format!("Risk: {}", risk_label(spec.risk))}</div>
                        </div>
                    </div>
                </span>
            </div>

            <div class="param-input-row">
                <input
                    id=input_id
                    class="input"
                    type="number"
                    inputmode="decimal"
                    min=spec.min
                    max=spec.max
                    step=spec.step
                    prop:value=move || text.get()
                    aria-invalid=move || {
                        let v = value.get();
                        (v < spec.min || v > spec.max).to_string()
                    }
                    on:focus=move |_| editing.set(true)
                    on:input={
                        let set_valid = Rc::clone(&set_valid);
                        move |ev| {
                            let raw = event_target_value(&ev);
                            text.set(raw.clone());
                            let raw_trim = raw.trim();
                            if raw_trim.is_empty() {
                                set_valid(false);
                                return;
                            }
                            match raw_trim.parse::<f32>() {
                                Ok(v) => {
                                    set_value.set(v);
                                    set_valid(v >= spec.min && v <= spec.max);
                                }
                                Err(_) => set_valid(false),
                            }
                        }
                    }
                    on:blur={
                        let set_valid = Rc::clone(&set_valid);
                        move |_| {
                            editing.set(false);
                            let raw = text.get_untracked();
                            let raw_trim = raw.trim();
                            if let Ok(v0) = raw_trim.parse::<f32>() {
                                let v = v0.clamp(spec.min, spec.max);
                                set_value.set(v);
                                text.set(format_float(v, spec.step));
                                set_valid(true);
                            } else {
                                // Revert to the current numeric value.
                                let v = value.get_untracked();
                                text.set(format_float(v, spec.step));
                                set_valid(v >= spec.min && v <= spec.max);
                            }
                        }
                    }
                />

                <button
                    type="button"
                    class="btn link"
                    on:click={
                        let set_valid = Rc::clone(&set_valid);
                        move |_| {
                            set_value.set(spec.default);
                            text.set(format_float(spec.default, spec.step));
                            set_valid(true);
                        }
                    }
                >
                    "Reset"
                </button>
            </div>

            <div class="param-hints">
                {move || {
                    warning_text
                        .get()
                        .map(|w| view!{ <div class="param-warn">{w}</div> })
                        .into_view()
                }}
            </div>
        </div>
    }
}
