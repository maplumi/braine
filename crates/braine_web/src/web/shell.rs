use leptos::ev::KeyboardEvent;
use leptos::prelude::*;
use std::sync::Arc;

use super::{GameKind, Theme, Toast, ToastLevel};

#[component]
pub(super) fn Topbar(
    sidebar_open: ReadSignal<bool>,
    set_sidebar_open: WriteSignal<bool>,
    gpu_status: ReadSignal<&'static str>,
    status: ReadSignal<String>,
    gpu_pending: ReadSignal<bool>,
    is_running: ReadSignal<bool>,
    theme: ReadSignal<Theme>,
    set_theme: WriteSignal<Theme>,
    open_docs: Callback<()>,
    open_analytics: Callback<()>,
    open_settings: Callback<()>,
) -> impl IntoView {
    // These are still wired by the parent for now, but the topbar no longer
    // renders redundant navigation buttons.
    let _ = (open_docs, open_analytics, open_settings);

    view! {
        <header class="app-header">
            <div class="app-header-left">
                <button
                    class="icon-btn sidebar-toggle"
                    title="Menu"
                    on:click=move |_| set_sidebar_open.set(!sidebar_open.get())
                >
                    "☰"
                </button>
                <h1 class="brand">
                    <img class="brand-icon" src="braine-icon.svg" alt="" aria-hidden="true" />
                    "Braine"
                </h1>
                <span class="subtle">{move || gpu_status.get()}</span>
            </div>
            <div class="app-header-right">
                <span class="status">{move || status.get()}</span>
                <Show when=move || gpu_pending.get()>
                    <span class="gpu-pending" title="Waiting for GPU brain step">
                        "GPU busy"
                    </span>
                </Show>
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
    }
}

#[component]
pub(super) fn ToastStack(toasts: RwSignal<Vec<Toast>>) -> impl IntoView {
    view! {
        <div class="toast-stack" aria-live="polite" aria-relevant="additions removals">
            <For
                each=move || toasts.get()
                key=|t| t.id
                children=move |t| {
                    let id = t.id;
                    let class = match t.level {
                        ToastLevel::Info => "toast info",
                        ToastLevel::Success => "toast success",
                        ToastLevel::Error => "toast error",
                    };
                    view! {
                        <div class=class>
                            <div style="flex: 1; white-space: pre-wrap;">{t.message}</div>
                            <button
                                class="toast-close"
                                title="Dismiss"
                                on:click=move |_| toasts.update(|ts| ts.retain(|x| x.id != id))
                            >
                                "×"
                            </button>
                        </div>
                    }
                }
            />
        </div>
    }
}

#[component]
pub(super) fn SystemErrorBanner(
    system_error: ReadSignal<Option<String>>,
    set_system_error: WriteSignal<Option<String>>,
) -> impl IntoView {
    view! {
        <Show when=move || system_error.get().is_some()>
            <div style="margin-bottom: 10px; padding: 10px 12px; background: #ff3b3018; border: 1px solid #ff3b3055; border-radius: 10px;">
                <div style="display:flex; gap: 10px; align-items: center; justify-content: space-between;">
                    <div style="color: #ffb4ad; font-weight: 600;">"Error"</div>
                    <button style="padding: 6px 10px;" on:click=move |_| set_system_error.set(None)>
                        "Dismiss"
                    </button>
                </div>
                <div style="margin-top: 6px; color: var(--text); white-space: pre-wrap; font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, 'Liberation Mono', 'Courier New', monospace; font-size: 12px;">
                    {move || system_error.get().unwrap_or_default()}
                </div>
            </div>
        </Show>
    }
}

#[component]
pub(super) fn Sidebar(
    sidebar_open: ReadSignal<bool>,
    set_sidebar_open: WriteSignal<bool>,
    show_about_page: ReadSignal<bool>,
    set_show_about_page: WriteSignal<bool>,
    game_kind: ReadSignal<GameKind>,
    set_game: Arc<dyn Fn(GameKind) + Send + Sync>,
    open_docs: Callback<()>,
    set_game_info_modal_kind: WriteSignal<Option<GameKind>>,
) -> impl IntoView {
    view! {
        // Sidebar overlay (mobile)
        <div
            class=move || {
                if sidebar_open.get() {
                    "sidebar-overlay open"
                } else {
                    "sidebar-overlay"
                }
            }
            on:click=move |_| set_sidebar_open.set(false)
        ></div>

        // Left game menu
        <aside class=move || if sidebar_open.get() { "sidebar open" } else { "sidebar" }>
            <div class="sidebar-header">
                <div class="sidebar-title">"Games"</div>
            </div>

            <div class="sidebar-section">
                <button
                    class=move || {
                        if show_about_page.get() {
                            "sidebar-item active"
                        } else {
                            "sidebar-item"
                        }
                    }
                    on:click=move |_| open_docs.run(())
                >
                    <span class="sidebar-label">"Docs"</span>
                    <span class="sidebar-ico">"ℹ️"</span>
                </button>
            </div>

            <div class="sidebar-section">
                {GameKind::all()
                    .iter()
                    .map(|&kind| {
                        let set_game_click = Arc::clone(&set_game);
                        let set_game_key = Arc::clone(&set_game);
                        view! {
                            <div
                                class=move || {
                                    if !show_about_page.get() && game_kind.get() == kind {
                                        "sidebar-item active"
                                    } else {
                                        "sidebar-item"
                                    }
                                }
                                role="button"
                                tabindex="0"
                                on:click=move |_| {
                                    set_show_about_page.set(false);
                                    set_game_click(kind);
                                    set_sidebar_open.set(false);
                                }
                                on:keydown=move |ev: KeyboardEvent| {
                                    let key = ev.key();
                                    if key == "Enter" || key == " " {
                                        ev.prevent_default();
                                        set_show_about_page.set(false);
                                        set_game_key(kind);
                                        set_sidebar_open.set(false);
                                    }
                                }
                            >
                                <span class="sidebar-label">{kind.display_name()}</span>
                                <button
                                    class="sidebar-pill sidebar-info-pill"
                                    title="Game information"
                                    on:click=move |ev| {
                                        ev.stop_propagation();
                                        set_game_info_modal_kind.set(Some(kind));
                                    }
                                >
                                    "ⓘ"
                                </button>
                            </div>
                        }
                    })
                    .collect_view()}
            </div>
        </aside>
    }
}
