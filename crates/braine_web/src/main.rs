// WASM entrypoint for Trunk.
//
// Native builds of this crate are intentionally no-ops by default; the real app
// is behind `--features web` and `wasm32`.

fn main() {
    // No-op on native targets.
}

#[cfg(all(feature = "web", target_arch = "wasm32"))]
#[wasm_bindgen::prelude::wasm_bindgen(start)]
pub fn wasm_start() {
    braine_web::start();
}
