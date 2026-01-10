//! Re-export wrapper.
//!
//! The actual implementation used by the WASM web app lives in `src/web/sequence_web.rs`.

#![cfg(all(feature = "web", target_arch = "wasm32"))]

#[path = "web/sequence_web.rs"]
mod sequence_web_impl;

pub use sequence_web_impl::*;
