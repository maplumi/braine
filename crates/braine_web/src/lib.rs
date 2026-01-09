//! Browser-hosted WASM app (Option A).
//!
//! This crate is intentionally a stub by default so the workspace builds on native/windows
//! targets without requiring wasm toolchains.
//!
//! Enable the real app with: `--features web` (and a wasm32 target).

/// Placeholder function for non-web (or non-wasm) builds.
#[cfg(not(all(feature = "web", target_arch = "wasm32")))]
pub fn placeholder() {
    // No-op.
}

#[cfg(all(feature = "web", target_arch = "wasm32"))]
mod web;

#[cfg(all(feature = "web", target_arch = "wasm32"))]
pub use web::start;
