//! Re-export wrapper.
//!
//! The actual implementation used by the WASM web app lives in `src/web/pong_web.rs`.
//! This file exists only to avoid duplicate implementations if something ever tries
//! to import `crate::pong_web` directly.

#![cfg(all(feature = "web", target_arch = "wasm32"))]

#[path = "web/pong_web.rs"]
mod pong_web_impl;

pub use pong_web_impl::*;
