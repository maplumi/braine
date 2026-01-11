#![cfg(feature = "std")]

pub use core::time::Duration;

// `std::time::Instant::now()` can panic on `wasm32-unknown-unknown` depending on
// how the runtime is configured. `web-time` provides a browser-backed monotonic
// clock via `performance.now()`.
#[cfg(target_arch = "wasm32")]
pub use web_time::Instant;

#[cfg(not(target_arch = "wasm32"))]
pub use std::time::Instant;
