//! Browser-hosted WASM app (Option A).
//!
//! This crate is intentionally a stub by default so the workspace builds on native/windows
//! targets without requiring wasm toolchains.
//!
//! Enable the real app with: `--features web` (and a wasm32 target).

#[cfg(feature = "web")]
mod app;

#[cfg(not(feature = "web"))]
pub fn placeholder() {
    // No-op.
}
