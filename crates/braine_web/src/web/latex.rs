use wasm_bindgen::prelude::*;

#[wasm_bindgen]
extern "C" {
    #[wasm_bindgen(js_name = braineLatexRenderAll)]
    fn braine_latex_render_all();
}

pub fn render_all() {
    // Intentionally fire-and-forget: JS side waits for KaTeX to be ready.
    braine_latex_render_all();
}
