use leptos::prelude::*;

use super::float_fmt::fmt_f64_fixed;

#[derive(Clone, Debug, PartialEq)]
pub struct TooltipPayload {
    pub id: String,
    pub title: String,
    pub body: String,
    pub meta: Vec<String>,
    pub when_to_change: Option<String>,
    pub risk: Option<String>,
    pub top_px: f64,
    pub left_px: f64,
}

pub type TooltipStore = RwSignal<Option<TooltipPayload>>;

#[component]
pub fn TooltipPortal(store: TooltipStore) -> impl IntoView {
    let payload = Memo::new(move |_| store.get());

    view! {
        <Show when=move || payload.get().is_some() fallback=|| ()>
            {move || {
                let p = payload
                    .get()
                    .expect("Show guarantees payload is Some when rendered");

                let top = fmt_f64_fixed(p.top_px, 0);
                let left = fmt_f64_fixed(p.left_px, 0);
                let style = format!("top: {top}px; left: {left}px;");
                let id = p.id;
                let title = p.title;
                let body = p.body;
                let meta = p.meta;
                let when_to_change = p.when_to_change;
                let risk = p.risk;

                view! {
                    <div id=id class="tooltip tooltip-portal" role="tooltip" style=style>
                        <div class="tooltip-title">{title}</div>
                        <div class="tooltip-body">{body}</div>
                        <div class="tooltip-meta">
                            <For
                                each=move || meta.clone().into_iter().enumerate()
                                key=|(i, _)| *i
                                children=|(_i, line)| view! { <div>{line}</div> }
                            />
                            {when_to_change
                                .clone()
                                .map(|s| view! { <div class="tooltip-when">{format!("When to change: {s}")}</div> })
                                .into_view()}
                            {risk
                                .clone()
                                .map(|s| view! { <div class="tooltip-risk">{format!("Risk: {s}")}</div> })
                                .into_view()}
                        </div>
                    </div>
                }
            }}
        </Show>
    }
}
