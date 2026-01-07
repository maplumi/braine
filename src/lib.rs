//! # braine
//!
//! A brain-like cognitive substrate based on sparse local dynamics.
//!
//! This crate provides a continuously-running dynamical substrate with local
//! Hebbian learning — no matrices, no backprop, no transformers.
//!
//! ## Quick Start
//!
//! ```
//! use braine::prelude::*;
//!
//! // Create a brain with default configuration
//! let cfg = BrainConfig::with_size(256, 12).with_seed(42);
//! let mut brain = Brain::new(cfg);
//!
//! // Define I/O groups
//! brain.define_sensor("vision", 8);
//! brain.define_action("move", 4);
//!
//! // Run a step
//! brain.apply_stimulus(Stimulus::new("vision", 1.0));
//! brain.set_neuromodulator(0.5); // reward signal
//! brain.step();
//!
//! let (action, score) = brain.select_action(&mut ActionPolicy::Deterministic);
//! ```
//!
//! ## Feature Flags
//!
//! - `std` (default): Standard library support
//! - `parallel`: Enable multi-threaded execution via rayon
//! - `simd`: Enable SIMD vectorization via the `wide` crate
//! - `gpu`: Enable GPU compute shaders via wgpu
//! - `serde`: Enable serialization/deserialization
//!
//! ## no_std Support
//!
//! Disable default features for `no_std` environments:
//! ```toml
//! braine = { version = "0.1", default-features = false }
//! ```
//!
//! ## Modules
//!
//! - [`substrate`]: Core brain implementation
//! - [`supervisor`]: Child brain spawning and consolidation
//! - [`causality`]: Temporal causal memory
//! - [`observer`]: Read-only observation adapters

// no_std support
#![cfg_attr(not(feature = "std"), no_std)]

#[cfg(not(feature = "std"))]
extern crate alloc;

#[path = "core/causality.rs"]
pub mod causality;

#[path = "core/prng.rs"]
pub mod prng;

#[path = "core/substrate.rs"]
pub mod substrate;

#[cfg(feature = "std")]
#[path = "core/supervisor.rs"]
pub mod supervisor;

#[cfg(feature = "std")]
#[path = "core/storage.rs"]
pub mod storage;

#[cfg(feature = "gpu")]
#[path = "core/gpu.rs"]
pub mod gpu;

#[cfg(feature = "std")]
pub mod observer;

/// Prelude module for convenient imports.
///
/// ```
/// use braine::prelude::*;
/// ```
pub mod prelude {
    pub use crate::causality::{CausalStats, SymbolId};
    pub use crate::substrate::{
        ActionPolicy, Amplitude, Brain, BrainConfig, Diagnostics, ExecutionTier, Neuromodulator,
        OwnedStimulus, Phase, Stimulus, UnitId, Weight,
    };
    #[cfg(feature = "std")]
    pub use crate::supervisor::{ChildConfigOverrides, ChildSpec, ConsolidationPolicy, Supervisor};
}
