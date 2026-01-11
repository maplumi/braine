//! Re-export wrapper.
//!
//! The actual implementation used by the WASM web app lives in `src/web/text_web.rs`.

#![cfg(all(feature = "web", target_arch = "wasm32"))]

#[path = "web/text_web.rs"]
mod text_web_impl;

pub use text_web_impl::*;
