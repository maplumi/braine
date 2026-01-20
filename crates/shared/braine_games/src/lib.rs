#![cfg_attr(not(feature = "std"), no_std)]

#[cfg(not(feature = "std"))]
extern crate alloc;

pub mod pong;

// Explicit helper API for applying stimuli to a `braine::substrate::Brain`.
// Kept behind the optional `braine` feature.
#[cfg(feature = "braine")]
pub mod brain_io;

// WASM-safe monotonic time shim for games that use wall-clock pacing.
#[cfg(feature = "std")]
pub(crate) mod time;

// These games currently rely on wall-clock timing (`std::time::Instant`) and heap
// allocations. Keep them behind `std` so `no_std` consumers can still use the
// lightweight simulations (e.g., Pong).
#[cfg(feature = "std")]
pub mod bandit;
#[cfg(feature = "std")]
pub mod maze;
#[cfg(feature = "std")]
pub mod replay;
#[cfg(feature = "std")]
pub mod sequence;
#[cfg(feature = "std")]
pub mod spot;
#[cfg(feature = "std")]
pub mod spot_reversal;
#[cfg(feature = "std")]
pub mod spot_xy;
#[cfg(feature = "std")]
pub mod stats;
#[cfg(feature = "std")]
pub mod text_next_token;
