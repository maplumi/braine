use leptos::prelude::*;
use wasm_bindgen::prelude::*;
use wasm_bindgen::JsValue;
use wasm_bindgen_futures::{spawn_local, JsFuture};

#[wasm_bindgen]
extern "C" {
    #[wasm_bindgen(js_name = braineMermaidRenderAll)]
    fn braine_mermaid_render_all();

    #[wasm_bindgen(js_name = braineMermaidApplyTheme)]
    fn braine_mermaid_apply_theme(theme_attr: &str);
}

pub fn render_all() {
    // Intentionally fire-and-forget: JS side waits for Mermaid to be ready.
    braine_mermaid_render_all();
}

pub fn apply_theme(theme_attr: &str) {
    // theme_attr is our app's "dark" | "light".
    braine_mermaid_apply_theme(theme_attr);
}

#[component]
pub fn MermaidDiagram(code: &'static str, max_width_px: u32) -> impl IntoView {
    // Leptos 0.7 doesn't expose `on_mount`; schedule a microtask so the DOM has
    // a chance to insert the `.mermaid` node before we ask Mermaid to render.
    Effect::new(move |_| {
        spawn_local(async move {
            let _ = JsFuture::from(js_sys::Promise::resolve(&JsValue::NULL)).await;
            render_all();
        });
    });

    // `data-mermaid-src` lets JS restore + rerender on theme toggles.
    let style = format!("width: 100%; max-width: {max_width_px}px;");

    view! {
        <div class="mermaid docs-diagram" style=style attr:data-mermaid-src=code>
            {code}
        </div>
    }
}
