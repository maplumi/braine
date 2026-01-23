// no_std support: use core and alloc when std is not available
#[cfg(not(feature = "std"))]
extern crate alloc;

#[cfg(feature = "std")]
use std::collections::{HashMap, HashSet};
#[cfg(feature = "std")]
use std::io::{self, Read, Write};

#[cfg(not(feature = "std"))]
use alloc::{format, string::String, string::ToString, vec, vec::Vec};
#[cfg(not(feature = "std"))]
use hashbrown::{HashMap, HashSet};

use core::ops::Range;

#[cfg(feature = "parallel")]
use rayon::prelude::*;

#[cfg(feature = "simd")]
use wide::{f32x4, CmpGt, CmpLt};

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

use crate::causality::{CausalMemory, SymbolId};
use crate::prng::Prng;
#[cfg(feature = "std")]
use crate::storage;

pub type UnitId = usize;

/// Type alias for connection weights (range: -1.5 to 1.5).
pub type Weight = f32;

/// Type alias for unit amplitudes (activity level).
pub type Amplitude = f32;

/// Type alias for unit phases (radians, -π to π).
pub type Phase = f32;

/// Type alias for neuromodulator signal (reward/salience scaling).
pub type Neuromodulator = f32;

/// Sentinel value for pruned/invalid connections in CSR storage.
pub const INVALID_UNIT: UnitId = UnitId::MAX;

/// Execution tier for step() and learning updates.
///
/// Allows seamless scaling from edge devices to servers:
/// - `Scalar`: Single-threaded, no SIMD (MCU, WASM, baseline)
/// - `Simd`: Single-threaded with manual SIMD (ARM NEON, x86 SSE/AVX)
/// - `Parallel`: Multi-threaded via rayon (desktop/server)
/// - `Gpu`: GPU compute shaders via wgpu (requires `gpu` feature)
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum ExecutionTier {
    /// Single-threaded scalar execution (default, works everywhere).
    #[default]
    Scalar,
    /// Single-threaded with SIMD vectorization (requires `simd` feature).
    Simd,
    /// Multi-threaded parallel execution (requires `parallel` feature).
    Parallel,
    /// GPU compute shader execution (requires `gpu` feature).
    Gpu,
}

/// Legacy struct kept for API compatibility in some contexts.
/// Internal storage now uses CSR format.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct Connection {
    pub target: UnitId,
    pub weight: Weight,
}

#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct Unit {
    // "Wave" state: amplitude + phase.
    pub amp: Amplitude,
    pub phase: Phase,

    pub bias: f32,
    pub decay: f32,

    /// Salience: cumulative access/activation frequency.
    /// Grows when the unit is active, decays slowly over time.
    /// Used for visualization (node size) and could inform pruning decisions.
    pub salience: f32,
}

/// CSR (Compressed Sparse Row) connection storage for cache-friendly iteration.
///
/// For unit `i`, its connections are stored at indices `conn_offsets[i]..conn_offsets[i+1]`.
/// This layout enables:
/// - Sequential memory access during dynamics updates
/// - SIMD-friendly weight arrays
/// - Efficient parallel iteration
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct CsrConnections {
    /// Flat array of all connection targets.
    pub targets: Vec<UnitId>,
    /// Parallel array of connection weights.
    pub weights: Vec<Weight>,
    /// Offset indices: unit i owns connections at [offsets[i]..offsets[i+1]).
    /// Length = unit_count + 1.
    pub offsets: Vec<usize>,
}

#[derive(Debug, Clone, Copy)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct BrainConfig {
    pub unit_count: usize,
    pub connectivity_per_unit: usize,

    pub dt: f32,
    pub base_freq: f32,

    pub noise_amp: f32,
    pub noise_phase: f32,
    /// Cubic amplitude saturation strength. When >0, adds a smooth -beta * amp^3
    /// term to the amplitude derivative so attractors do not rely solely on
    /// hard clipping. Typical small values: 0.05..0.5.
    pub amp_saturation_beta: f32,

    /// Phase coupling mode for dynamics.
    ///
    /// 0 = linear wrapped angle difference (legacy)
    /// 1 = sinusoidal (Kuramoto-like; bounded)
    /// 2 = tanh(k * delta) (bounded linear-ish)
    pub phase_coupling_mode: u8,

    /// Gain used for tanh phase coupling mode.
    pub phase_coupling_k: f32,

    /// Gain factor applied to phase coupling influence.
    ///
    /// Separates phase coupling strength from amplitude coupling strength.
    /// Typical values: 0.1..1.0 (lower than amplitude weights).
    pub phase_coupling_gain: f32,

    // Competition: subtract proportional inhibition from all units.
    pub global_inhibition: f32,

    /// Inhibition mode: how to compute the global inhibition signal.
    ///
    /// 0 = signed mean (legacy; can cancel out)
    /// 1 = mean absolute amplitude (|a|)
    /// 2 = rectified mean (max(0, a))
    pub inhibition_mode: u8,

    // Local learning/forgetting.
    pub hebb_rate: f32,
    pub forget_rate: f32,
    pub prune_below: f32,

    pub coactive_threshold: f32,

    // If two units are active and phase-aligned, strengthen more.
    // Range ~ [0, 1]: higher means "must be more aligned".
    pub phase_lock_threshold: f32,

    // One-shot concept formation strength (imprinting).
    pub imprint_rate: f32,

    // Salience tracking: nodes accumulate importance when activated.
    // salience_decay: rate at which salience decays each step (e.g., 0.001).
    // salience_gain: how much salience increases when amplitude exceeds threshold.
    pub salience_decay: f32,
    pub salience_gain: f32,

    /// Low-pass activity trace decay used to separate fast activation from
    /// learning/salience traces.
    ///
    /// - 0.0 disables the trace and uses instantaneous activation.
    /// - Higher values track activation more quickly.
    pub activity_trace_decay: f32,

    // ---------------------------------------------------------------------
    // Neurogenesis / Growth policy (refinement item 5)
    // ---------------------------------------------------------------------
    /// 0 = legacy (avg |w| saturation only), 1 = hybrid (includes learning-pressure signals).
    pub growth_policy_mode: u8,
    /// Cooldown in steps between growth events.
    pub growth_cooldown_steps: u32,
    /// EMA smoothing factor for growth signals in [0,1].
    pub growth_signal_alpha: f32,
    /// Minimum EMA of plasticity commits (0..1) to consider "learning pressure" high.
    pub growth_commit_ema_threshold: f32,
    /// Minimum normalized eligibility L1 EMA to consider "learning pressure" high.
    pub growth_eligibility_norm_ema_threshold: f32,
    /// Maximum normalized prune-rate EMA to consider pruning "not relieving saturation".
    pub growth_prune_norm_ema_max: f32,

    // ---------------------------------------------------------------------
    // Meaning/causal temporal structure (refinement item 6)
    // ---------------------------------------------------------------------
    /// Number of lag steps (1 = current behavior). Recommended 1..=16.
    pub causal_lag_steps: u8,
    /// Geometric decay per lag (>0 and <1). lag 2 gets weight=decay, lag 3=decay^2, etc.
    pub causal_lag_decay: f32,
    /// Cap how many symbols per tick participate in lagged updates (keeps bounded work).
    pub causal_symbol_cap: u8,

    // If set, makes behavior reproducible for evaluation.
    pub seed: Option<u64>,

    // Causality/meaning memory decay (0..1). Higher means faster forgetting.
    pub causal_decay: f32,

    // ---------------------------------------------------------------------
    // Stability–Plasticity Control (learning governance)
    // ---------------------------------------------------------------------
    /// Neuromodulator deadband for committing plasticity.
    ///
    /// If `abs(neuromod) <= learning_deadband`, eligibility traces still update,
    /// but weights are not changed.
    pub learning_deadband: f32,

    /// Eligibility trace decay per step in [0,1]. Higher decays faster.
    pub eligibility_decay: f32,

    /// Eligibility trace gain (accumulation rate). Higher accumulates faster.
    pub eligibility_gain: f32,

    /// Smoothness for coactivity thresholding in eligibility.
    ///
    /// 0.0 keeps a hard ReLU at `coactive_threshold`. Higher values make the
    /// transition smoother (softplus temperature, in amp units).
    pub coactive_softness: f32,

    /// Smoothness for phase-gate blending in eligibility.
    ///
    /// 0.0 keeps a hard gate at `phase_lock_threshold`. Higher values increase
    /// blending width (in alignment units).
    pub phase_gate_softness: f32,

    /// Plasticity budget: maximum total `sum(|Δw|)` per step. 0 disables budgeting.
    pub plasticity_budget: f32,

    /// Target activity level for slow homeostasis (uses `abs(amp)`).
    pub homeostasis_target_amp: f32,

    /// Homeostasis rate (bias adaptation step size). 0 disables homeostasis.
    pub homeostasis_rate: f32,

    /// Run homeostasis every N `step()` calls.
    pub homeostasis_every: u32,

    // ---------------------------------------------------------------------
    // Module-local learning + routing (scaling phase 1–2)
    // ---------------------------------------------------------------------
    /// If >0, gate learning to the top-K modules selected from the most recent
    /// committed symbol set.
    ///
    /// 0 disables routing (default; preserves legacy behavior).
    pub module_routing_top_k: u8,

    /// If true, units with no module assignment are excluded from
    /// routing-gated learning.
    pub module_routing_strict: bool,

    /// Weight on per-module reward EMA in routing score.
    pub module_routing_beta: f32,

    /// Per-commit decay rate for module signature association weights in [0,1].
    ///
    /// 0 disables signature decay and reward EMA updates.
    pub module_signature_decay: f32,

    /// Maximum number of symbol associations stored per module.
    pub module_signature_cap: u8,

    /// If >0, skip learning updates for units with learning-activity below this threshold.
    ///
    /// Learning-activity is `max(activity_trace, max(amp, 0))` when
    /// `activity_trace_decay > 0`, otherwise `max(amp, 0)`.
    pub module_learning_activity_threshold: f32,

    /// Optional per-module plasticity budget (L1 of |Δw|) per step.
    ///
    /// 0 disables per-module budgeting.
    pub module_plasticity_budget: f32,

    // ---------------------------------------------------------------------
    // Cross-module coupling governance (scaling phase 3)
    // ---------------------------------------------------------------------
    /// Scale factor applied to committed plasticity (`Δw`) for cross-module edges.
    ///
    /// 1.0 preserves legacy behavior; smaller values make cross-module couplings
    /// harder to form.
    pub cross_module_plasticity_scale: f32,

    /// Additional forgetting applied to cross-module edges.
    ///
    /// Effective decay becomes `1 - forget_rate - cross_module_forget_boost`.
    /// 0.0 disables extra forgetting.
    pub cross_module_forget_boost: f32,

    /// Extra prune threshold applied to cross-module edges.
    ///
    /// Effective prune threshold becomes `prune_below + cross_module_prune_bonus`.
    /// 0.0 disables extra pruning.
    pub cross_module_prune_bonus: f32,

    // ---------------------------------------------------------------------
    // Latent modules: auto-formation + retirement (scaling phase 5)
    // ---------------------------------------------------------------------
    /// If true, allow auto-creating latent modules when routing is enabled but
    /// the router is uninformative and the committed symbol set appears novel.
    pub latent_module_auto_create: bool,

    /// Number of units to reserve for each auto-created latent module.
    ///
    /// 0 disables auto-creation even if `latent_module_auto_create` is true.
    pub latent_module_auto_width: u32,

    /// Minimum steps between auto-created latent modules.
    pub latent_module_auto_cooldown_steps: u32,

    /// Maximum number of active (non-empty) latent modules allowed.
    /// 0 means unlimited.
    pub latent_module_auto_max_active: u32,

    /// Reward magnitude required to allow auto-creation.
    ///
    /// Auto-creation is allowed only when `abs(neuromod) >= threshold`.
    pub latent_module_auto_reward_threshold: f32,

    /// Retire a latent module (freeing its units) if it has not been routed for
    /// this many steps.
    ///
    /// 0 disables retirement.
    pub latent_module_retire_after_steps: u32,

    /// Retire only if `abs(reward_ema) < threshold`.
    pub latent_module_retire_reward_threshold: f32,
}

impl Default for BrainConfig {
    /// Returns a sensible default configuration for edge devices.
    ///
    /// - 256 units with 12 connections each
    /// - Moderate noise for exploration
    /// - Conservative learning/forgetting rates
    fn default() -> Self {
        Self {
            unit_count: 256,
            connectivity_per_unit: 12,
            dt: 0.05,
            base_freq: 1.0,
            noise_amp: 0.02,
            noise_phase: 0.01,
            amp_saturation_beta: 0.1,
            phase_coupling_mode: 1,
            phase_coupling_k: 2.0,
            phase_coupling_gain: 1.0,
            global_inhibition: 0.2,
            inhibition_mode: 0,
            hebb_rate: 0.08,
            forget_rate: 0.0005,
            prune_below: 0.01,
            coactive_threshold: 0.3,
            phase_lock_threshold: 0.7,
            imprint_rate: 0.5,
            salience_decay: 0.001, // Slow decay to preserve importance history
            salience_gain: 0.1,    // Moderate gain when activated
            activity_trace_decay: 0.05,

            growth_policy_mode: 0,
            growth_cooldown_steps: 250,
            growth_signal_alpha: 0.05,
            growth_commit_ema_threshold: 0.2,
            growth_eligibility_norm_ema_threshold: 0.02,
            growth_prune_norm_ema_max: 0.0005,

            causal_lag_steps: 1,
            causal_lag_decay: 0.7,
            causal_symbol_cap: 32,
            seed: None,
            causal_decay: 0.002,

            // Control defaults: tuned for reward-pulse learning (daemon loop)
            // while remaining conservative on edge devices.
            learning_deadband: 0.05,
            eligibility_decay: 0.02,
            eligibility_gain: 0.35,
            coactive_softness: 0.05,
            phase_gate_softness: 0.05,
            plasticity_budget: 0.0,
            homeostasis_target_amp: 0.25,
            homeostasis_rate: 0.0,
            homeostasis_every: 50,

            // Scaling defaults (off by default).
            module_routing_top_k: 0,
            module_routing_strict: false,
            module_routing_beta: 0.2,
            module_signature_decay: 0.01,
            module_signature_cap: 32,
            module_learning_activity_threshold: 0.0,
            module_plasticity_budget: 0.0,

            // Cross-module governance defaults (off by default).
            cross_module_plasticity_scale: 1.0,
            cross_module_forget_boost: 0.0,
            cross_module_prune_bonus: 0.0,

            // Latent module defaults (off by default).
            latent_module_auto_create: false,
            latent_module_auto_width: 8,
            latent_module_auto_cooldown_steps: 500,
            latent_module_auto_max_active: 0,
            latent_module_auto_reward_threshold: 0.2,
            latent_module_retire_after_steps: 0,
            latent_module_retire_reward_threshold: 0.05,
        }
    }
}

impl BrainConfig {
    /// Minimum allowed unit count.
    pub const MIN_UNITS: usize = 4;
    /// Maximum allowed unit count (prevents OOM on 32-bit systems).
    pub const MAX_UNITS: usize = 1 << 24; // 16M units
    /// Maximum connectivity per unit.
    pub const MAX_CONNECTIVITY: usize = 1024;

    /// Create a new config with specified size and connectivity.
    ///
    /// # Panics
    /// Panics if `unit_count` or `connectivity_per_unit` are out of valid range.
    pub fn with_size(unit_count: usize, connectivity_per_unit: usize) -> Self {
        assert!(
            unit_count >= Self::MIN_UNITS,
            "unit_count must be >= {}",
            Self::MIN_UNITS
        );
        assert!(
            unit_count <= Self::MAX_UNITS,
            "unit_count must be <= {}",
            Self::MAX_UNITS
        );
        assert!(
            connectivity_per_unit <= Self::MAX_CONNECTIVITY,
            "connectivity_per_unit must be <= {}",
            Self::MAX_CONNECTIVITY
        );
        assert!(
            connectivity_per_unit < unit_count,
            "connectivity_per_unit must be < unit_count"
        );

        Self {
            unit_count,
            connectivity_per_unit,
            ..Default::default()
        }
    }

    /// Validate the configuration, returning an error message if invalid.
    pub fn validate(&self) -> Result<(), &'static str> {
        if self.unit_count < Self::MIN_UNITS {
            return Err("unit_count too small");
        }
        if self.unit_count > Self::MAX_UNITS {
            return Err("unit_count too large");
        }
        if self.connectivity_per_unit >= self.unit_count {
            return Err("connectivity_per_unit must be < unit_count");
        }
        if self.connectivity_per_unit > Self::MAX_CONNECTIVITY {
            return Err("connectivity_per_unit too large");
        }
        if self.dt <= 0.0 || self.dt > 1.0 {
            return Err("dt must be in (0, 1]");
        }
        if self.phase_coupling_mode > 2 {
            return Err("phase_coupling_mode must be in [0, 2]");
        }
        if !self.phase_coupling_k.is_finite() || self.phase_coupling_k < 0.0 {
            return Err("phase_coupling_k must be finite and >= 0");
        }
        if self.hebb_rate < 0.0 || self.hebb_rate > 1.0 {
            return Err("hebb_rate must be in [0, 1]");
        }
        if self.forget_rate < 0.0 || self.forget_rate > 1.0 {
            return Err("forget_rate must be in [0, 1]");
        }
        if self.causal_decay < 0.0 || self.causal_decay > 1.0 {
            return Err("causal_decay must be in [0, 1]");
        }

        if !self.activity_trace_decay.is_finite() || self.activity_trace_decay < 0.0 {
            return Err("activity_trace_decay must be finite and >= 0");
        }

        if self.growth_policy_mode > 1 {
            return Err("growth_policy_mode must be in [0, 1]");
        }
        if !self.growth_signal_alpha.is_finite()
            || self.growth_signal_alpha < 0.0
            || self.growth_signal_alpha > 1.0
        {
            return Err("growth_signal_alpha must be in [0, 1]");
        }
        if !self.growth_commit_ema_threshold.is_finite()
            || self.growth_commit_ema_threshold < 0.0
            || self.growth_commit_ema_threshold > 1.0
        {
            return Err("growth_commit_ema_threshold must be in [0, 1]");
        }
        if !self.growth_eligibility_norm_ema_threshold.is_finite()
            || self.growth_eligibility_norm_ema_threshold < 0.0
        {
            return Err("growth_eligibility_norm_ema_threshold must be finite and >= 0");
        }
        if !self.growth_prune_norm_ema_max.is_finite() || self.growth_prune_norm_ema_max < 0.0 {
            return Err("growth_prune_norm_ema_max must be finite and >= 0");
        }

        if self.causal_lag_steps == 0 {
            return Err("causal_lag_steps must be >= 1");
        }
        if self.causal_lag_steps > 32 {
            return Err("causal_lag_steps must be <= 32");
        }
        if !(self.causal_lag_decay.is_finite()
            && 0.0 < self.causal_lag_decay
            && self.causal_lag_decay < 1.0)
        {
            return Err("causal_lag_decay must be finite and in (0, 1)");
        }
        if self.causal_symbol_cap == 0 {
            return Err("causal_symbol_cap must be >= 1");
        }

        if self.learning_deadband < 0.0 || self.learning_deadband > 1.0 {
            return Err("learning_deadband must be in [0, 1]");
        }
        if self.eligibility_decay < 0.0 || self.eligibility_decay > 1.0 {
            return Err("eligibility_decay must be in [0, 1]");
        }
        if !self.eligibility_gain.is_finite() || self.eligibility_gain < 0.0 {
            return Err("eligibility_gain must be finite and >= 0");
        }
        if !self.coactive_softness.is_finite() || self.coactive_softness < 0.0 {
            return Err("coactive_softness must be finite and >= 0");
        }
        if !self.phase_gate_softness.is_finite() || self.phase_gate_softness < 0.0 {
            return Err("phase_gate_softness must be finite and >= 0");
        }
        if !self.plasticity_budget.is_finite() || self.plasticity_budget < 0.0 {
            return Err("plasticity_budget must be finite and >= 0");
        }
        if !self.homeostasis_target_amp.is_finite() || self.homeostasis_target_amp < 0.0 {
            return Err("homeostasis_target_amp must be finite and >= 0");
        }
        if !self.homeostasis_rate.is_finite() || self.homeostasis_rate < 0.0 {
            return Err("homeostasis_rate must be finite and >= 0");
        }
        if self.homeostasis_every == 0 {
            return Err("homeostasis_every must be >= 1");
        }

        if !self.module_routing_beta.is_finite() || self.module_routing_beta < 0.0 {
            return Err("module_routing_beta must be finite and >= 0");
        }
        if !self.module_signature_decay.is_finite()
            || self.module_signature_decay < 0.0
            || self.module_signature_decay > 1.0
        {
            return Err("module_signature_decay must be in [0, 1]");
        }
        if self.module_signature_cap == 0 {
            return Err("module_signature_cap must be >= 1");
        }
        if !self.module_learning_activity_threshold.is_finite()
            || self.module_learning_activity_threshold < 0.0
        {
            return Err("module_learning_activity_threshold must be finite and >= 0");
        }
        if !self.module_plasticity_budget.is_finite() || self.module_plasticity_budget < 0.0 {
            return Err("module_plasticity_budget must be finite and >= 0");
        }

        if !self.cross_module_plasticity_scale.is_finite()
            || self.cross_module_plasticity_scale < 0.0
        {
            return Err("cross_module_plasticity_scale must be finite and >= 0");
        }
        if !self.cross_module_forget_boost.is_finite() || self.cross_module_forget_boost < 0.0 {
            return Err("cross_module_forget_boost must be finite and >= 0");
        }
        if !self.cross_module_prune_bonus.is_finite() || self.cross_module_prune_bonus < 0.0 {
            return Err("cross_module_prune_bonus must be finite and >= 0");
        }

        if self.latent_module_auto_width as usize > self.unit_count {
            return Err("latent_module_auto_width must be <= unit_count");
        }
        if !self.latent_module_auto_reward_threshold.is_finite()
            || self.latent_module_auto_reward_threshold < 0.0
        {
            return Err("latent_module_auto_reward_threshold must be finite and >= 0");
        }
        if !self.latent_module_retire_reward_threshold.is_finite()
            || self.latent_module_retire_reward_threshold < 0.0
        {
            return Err("latent_module_retire_reward_threshold must be finite and >= 0");
        }
        Ok(())
    }

    /// Estimated memory usage in bytes for a brain with this config.
    #[must_use]
    pub fn estimated_memory_bytes(&self) -> usize {
        let units_size = self.unit_count * core::mem::size_of::<Unit>();
        let conns = self.unit_count * self.connectivity_per_unit;
        let targets_size = conns * core::mem::size_of::<UnitId>();
        let weights_size = conns * core::mem::size_of::<Weight>();
        let offsets_size = (self.unit_count + 1) * core::mem::size_of::<usize>();

        units_size + targets_size + weights_size + offsets_size
    }

    /// Set the random seed for reproducibility.
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self
    }

    /// Set the learning rate.
    pub fn with_hebb_rate(mut self, rate: f32) -> Self {
        self.hebb_rate = rate;
        self
    }

    /// Set the noise levels for exploration.
    pub fn with_noise(mut self, amp: f32, phase: f32) -> Self {
        self.noise_amp = amp;
        self.noise_phase = phase;
        self
    }
}

#[derive(Debug, Clone, Copy)]
pub struct Stimulus<'a> {
    pub name: &'a str,
    pub strength: f32,
}

/// Owned version of Stimulus for serialization.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct OwnedStimulus {
    pub name: String,
    pub strength: f32,
}

/// A lightweight point for UI visualization of the substrate.
///
/// This is intentionally small and cheap to generate; it is **not** a full brain dump.
#[derive(Debug, Clone, Copy)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct UnitPlotPoint {
    pub id: u32,
    pub amp: f32,
    /// Amplitude normalized to [0,1] relative to the sampled max.
    /// This makes the UI plot readable even when absolute activity is low.
    pub amp01: f32,
    /// Unit's current oscillatory phase in [0, 2π).
    /// Used for pulsing visualization effects.
    pub phase: f32,
    /// Salience normalized to [0,1] relative to the sampled max.
    /// Represents cumulative access frequency - higher salience = more frequently activated.
    pub salience01: f32,
    /// Normalized relative age proxy in [0,1].
    /// Higher means "newer" (later unit IDs), which aligns with neurogenesis appends.
    pub rel_age: f32,
    pub is_reserved: bool,
    pub is_sensor_member: bool,
    pub is_group_member: bool,
}

/// Per-action score breakdown for UI inspection.
///
/// This is *not* a learning signal by itself; it's a snapshot of what the current
/// readout would prefer, and why.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct ActionScoreBreakdown {
    pub name: String,
    pub habit_norm: f32,
    pub meaning_global: f32,
    pub meaning_conditional: f32,
    pub meaning: f32,
    pub score: f32,
}

/// Reward-edge breakdown for a symbol in causal memory.
///
/// This is primarily for UI/diagnostics.
#[derive(Debug, Clone, Copy, Default)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct RewardEdges {
    pub to_reward_pos: f32,
    pub to_reward_neg: f32,
    /// Convenience: `to_reward_pos - to_reward_neg`.
    pub meaning: f32,
}

/// A single node in causal graph visualization.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct CausalNodeViz {
    pub id: SymbolId,
    pub name: String,
    pub base_count: f32,
}

/// A single edge in causal graph visualization.
#[derive(Debug, Clone, Copy)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct CausalEdgeViz {
    pub from: SymbolId,
    pub to: SymbolId,
    pub strength: f32,
}

/// Causal graph data for visualization.
#[derive(Debug, Clone, Default)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct CausalGraphViz {
    pub nodes: Vec<CausalNodeViz>,
    pub edges: Vec<CausalEdgeViz>,
}

impl OwnedStimulus {
    /// Convert to a borrowed Stimulus.
    pub fn as_stimulus(&self) -> Stimulus<'_> {
        Stimulus {
            name: &self.name,
            strength: self.strength,
        }
    }
}

impl<'a> Stimulus<'a> {
    pub fn new(name: &'a str, strength: f32) -> Self {
        Self { name, strength }
    }
}

#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum ActionPolicy {
    Deterministic,
    EpsilonGreedy { epsilon: f32 },
}

/// Runtime diagnostics about the brain's current state.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct Diagnostics {
    /// Total number of units in the substrate.
    pub unit_count: usize,
    /// Total number of active (non-pruned) connections.
    pub connection_count: usize,
    /// Connections pruned in the last step.
    pub pruned_last_step: usize,
    /// Units born via neurogenesis in the last step.
    pub births_last_step: usize,
    /// Average amplitude across all units.
    pub avg_amp: Amplitude,
    /// Average connection weight magnitude (saturation indicator).
    pub avg_weight: Weight,
    /// Estimated memory usage in bytes.
    pub memory_bytes: usize,
    /// Current execution tier.
    pub execution_tier: ExecutionTier,
}

/// Lightweight monitors for learning/stability.
///
/// These are intended for dashboards and debugging: they summarize the most
/// recent step's learning-related activity without exposing internal buffers.
#[derive(Debug, Clone, Copy, Default)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct LearningStats {
    /// Whether plasticity was eligible to be committed this step (neuromod outside deadband).
    pub plasticity_committed: bool,
    /// Sum of absolute weight changes applied this step.
    pub plasticity_l1: f32,
    /// Number of edges updated this step.
    pub plasticity_edges: u32,
    /// Plasticity budget configured (0 means disabled).
    pub plasticity_budget: f32,
    /// Plasticity budget consumed this step (L1).
    pub plasticity_budget_used: f32,
    /// Sum of absolute eligibility values after update.
    pub eligibility_l1: f32,
    /// Sum of absolute bias changes applied by homeostasis this step (0 if not run).
    pub homeostasis_bias_l1: f32,
}

#[derive(Debug, Clone, Copy, Default)]
struct LearningMonitors {
    plasticity_committed: bool,
    plasticity_l1: f32,
    plasticity_edges: u32,
    plasticity_budget_used: f32,
    eligibility_l1: f32,
    homeostasis_bias_l1: f32,
}

#[derive(Debug, Clone)]
struct NamedGroup {
    name: String,
    units: Vec<UnitId>,
}

#[cfg(feature = "std")]
const LATENT_MODULES_CHUNK: [u8; 4] = *b"LMOD";

const NO_MODULE: u16 = u16::MAX;

#[derive(Debug, Clone)]
struct RoutingModule {
    // Base name without kind prefix.
    name: String,
    // Sparse associations between this module and boundary symbols.
    signature: HashMap<SymbolId, f32>,
    // Reward summary to bias routing.
    reward_ema: f32,
    // Last step index when this module was selected by the router.
    last_routed_step: u64,
}

/// A dynamical brain-like substrate with local learning.
///
/// `Brain` is the core cognitive substrate: a collection of oscillating units
/// with sparse connections that learn through local Hebbian plasticity.
///
/// # Key Features
/// - **Sparse local dynamics**: No matrices, no backprop
/// - **Continuous learning**: Hebbian updates every step
/// - **Meaning/causality**: Tracks temporal relationships between symbols
/// - **Tiered execution**: Scalar, SIMD, parallel, or GPU
///
/// # Example
/// ```
/// use braine::substrate::{Brain, BrainConfig, Stimulus, ActionPolicy};
///
/// let cfg = BrainConfig::with_size(256, 12).with_seed(42);
/// let mut brain = Brain::new(cfg);
///
/// brain.define_sensor("vision", 8);
/// brain.define_action("move", 4);
///
/// brain.apply_stimulus(Stimulus::new("vision", 1.0));
/// brain.step();
/// let (action, score) = brain.select_action(&mut ActionPolicy::Deterministic);
/// ```
#[derive(Clone)]
pub struct Brain {
    cfg: BrainConfig,
    units: Vec<Unit>,

    /// Slow activity trace per unit (ephemeral; not persisted).
    ///
    /// Used to decouple fast `amp` from learning/salience gating.
    activity_trace: Vec<f32>,

    // Growth policy signals (ephemeral; not persisted).
    growth_eligibility_norm_ema: f32,
    growth_commit_ema: f32,
    growth_prune_norm_ema: f32,
    growth_last_birth_step: u64,

    // Lagged causal meaning history (ephemeral; not persisted). Stores lag>=2 symbol sets.
    causal_lag_history: Vec<Vec<SymbolId>>,

    /// CSR-format connection storage for cache-friendly iteration.
    connections: CsrConnections,

    /// Eligibility trace per CSR edge (ephemeral; not persisted).
    ///
    /// Length always matches `connections.weights.len()`.
    eligibility: Vec<f32>,

    /// Execution tier for step/learning (Scalar, Simd, or Parallel).
    tier: ExecutionTier,

    rng: Prng,

    reserved: Vec<bool>,

    // Cached membership maps (derived from groups; not serialized).
    // Used for fast, stable “engram” handling.
    sensor_member: Vec<bool>,
    group_member: Vec<bool>,

    // If false, unit's outgoing connections do not undergo learning updates.
    // Used to protect a parent identity subset in child brains.
    learning_enabled: Vec<bool>,

    // External "sensor" input is just injected current to some units.
    sensor_groups: Vec<NamedGroup>,
    action_groups: Vec<NamedGroup>,

    // Persisted latent modules (not sensors/actions).
    latent_groups: Vec<NamedGroup>,

    // Module routing state (ephemeral; not persisted).
    routing_modules: Vec<RoutingModule>,
    routing_module_index: HashMap<String, u16>,
    unit_module: Vec<u16>,
    learning_route_modules: Vec<u16>,

    // Cached unit counts per module (ephemeral; derived from `unit_module`).
    module_unit_counts: Vec<u32>,
    module_unit_counts_dirty: bool,

    // Latent module auto-formation state (ephemeral; not persisted).
    latent_auto_last_create_step: u64,
    latent_auto_seq: u32,

    // Fast lookup: stimulus name -> sensor group index.
    // Derived from `sensor_groups`; not serialized.
    sensor_group_index: HashMap<String, usize>,

    pending_input: Vec<f32>,

    // Neuromodulator scales learning ("reward", "salience").
    neuromod: f32,

    // Boundary symbol table for causality/meaning.
    symbols: HashMap<String, SymbolId>,
    symbols_rev: Vec<String>,
    active_symbols: Vec<SymbolId>,
    causal: CausalMemory,

    reward_pos_symbol: SymbolId,
    reward_neg_symbol: SymbolId,

    pruned_last_step: usize,

    // Count of tombstoned CSR entries (target == INVALID_UNIT).
    // Not serialized; persistence compacts away tombstones.
    csr_tombstones: usize,

    /// Units born via neurogenesis in the last step.
    births_last_step: usize,

    age_steps: u64,

    telemetry: Telemetry,

    learning_monitors: LearningMonitors,
}

/// A bounded, sparse representation of structural changes between two brains.
///
/// This is intentionally minimal for the initial expert/child-brain mechanism:
/// it only carries top-K connection weight deltas.
#[derive(Debug, Clone, Default)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct BrainDelta {
    pub weight_deltas: Vec<(usize, Weight)>,
}

#[derive(Debug, Clone, Default)]
struct Telemetry {
    enabled: bool,

    last_stimuli: Vec<SymbolId>,
    last_actions: Vec<SymbolId>,
    last_reinforced_actions: Vec<(SymbolId, f32)>,
    last_committed_symbols: Vec<SymbolId>,
}

impl Brain {
    /// A coarse fingerprint of the current connection topology.
    ///
    /// Used for safe-ish synchronization between a master brain and edge/child brains.
    /// If this value differs, applying deltas by edge-index is unsafe.
    #[must_use]
    pub fn connections_fingerprint(&self) -> u64 {
        // NOTE: This must work in `no_std` builds (e.g. wasm32-unknown-unknown).
        // We don't need cryptographic security here; we just want a stable-ish
        // topology fingerprint to gate safe delta application.
        #[inline]
        fn mix64(mut h: u64, x: u64) -> u64 {
            // FNV-1a-ish mixing
            h ^= x;
            h = h.wrapping_mul(1099511628211);
            // extra avalanching
            h ^= h >> 33;
            h = h.wrapping_mul(0xff51afd7ed558ccd);
            h ^= h >> 33;
            h
        }

        let mut h = 14695981039346656037u64;
        h = mix64(h, self.units.len() as u64);
        h = mix64(h, self.connections.weights.len() as u64);
        for &t in &self.connections.targets {
            h = mix64(h, t as u64);
        }
        h
    }

    /// Number of connection weights (equals number of edges).
    #[must_use]
    pub fn weights_len(&self) -> usize {
        self.connections.weights.len()
    }

    /// Compute a sparse delta from `base` to `self` by taking the top-K
    /// absolute connection weight changes.
    ///
    /// If the connection topology differs (length mismatch), returns an empty delta.
    #[must_use]
    pub fn diff_weights_topk(&self, base: &Brain, topk: usize) -> BrainDelta {
        if topk == 0 {
            return BrainDelta::default();
        }

        let w_self = &self.connections.weights;
        let w_base = &base.connections.weights;
        if w_self.len() != w_base.len() {
            return BrainDelta::default();
        }

        // If targets differ, merging by index would be incorrect.
        // For now, require identical target arrays.
        if self.connections.targets != base.connections.targets {
            return BrainDelta::default();
        }

        let mut deltas: Vec<(usize, Weight)> = Vec::with_capacity(topk.min(w_self.len()));

        for i in 0..w_self.len() {
            let dw = w_self[i] - w_base[i];
            if dw.abs() > 1.0e-6 {
                deltas.push((i, dw));
            }
        }

        deltas.sort_by(|a, b| {
            b.1.abs()
                .partial_cmp(&a.1.abs())
                .unwrap_or(core::cmp::Ordering::Equal)
        });
        deltas.truncate(topk);

        BrainDelta {
            weight_deltas: deltas,
        }
    }

    /// Apply a sparse delta of connection weight changes.
    ///
    /// The applied delta for each edge is clamped to `[-delta_max, +delta_max]`.
    pub fn apply_weight_delta(&mut self, delta: &BrainDelta, delta_max: Weight) {
        if delta.weight_deltas.is_empty() {
            return;
        }
        if delta_max <= 0.0 {
            return;
        }

        let w = &mut self.connections.weights;
        for (idx, dw) in &delta.weight_deltas {
            if *idx >= w.len() {
                continue;
            }
            let clipped = dw.clamp(-delta_max, delta_max);
            w[*idx] += clipped;
        }
    }

    pub fn new(cfg: BrainConfig) -> Self {
        let mut rng = Prng::new(cfg.seed.unwrap_or(1));

        let mut units = Vec::with_capacity(cfg.unit_count);
        for _ in 0..cfg.unit_count {
            units.push(Unit {
                amp: 0.0,
                phase: rng.gen_range_f32(-core::f32::consts::PI, core::f32::consts::PI),
                bias: 0.0,
                decay: 0.12,
                salience: 0.0,
            });
        }

        // Build CSR connection storage with random sparse wiring.
        // Total connections = unit_count * connectivity_per_unit
        let total_conns = cfg.unit_count * cfg.connectivity_per_unit;
        // Over-allocate a bit to reduce realloc spikes if append_connection is triggered.
        let extra = total_conns / 2;
        let mut conn_targets = Vec::with_capacity(total_conns + extra);
        let mut conn_weights = Vec::with_capacity(total_conns + extra);
        let mut conn_offsets = Vec::with_capacity(cfg.unit_count + 1);

        for i in 0..cfg.unit_count {
            conn_offsets.push(conn_targets.len());
            for _ in 0..cfg.connectivity_per_unit {
                let mut target = rng.gen_range_usize(0, cfg.unit_count);
                if target == i {
                    target = (target + 1) % cfg.unit_count;
                }
                let weight = rng.gen_range_f32(-0.15, 0.15);
                conn_targets.push(target);
                conn_weights.push(weight);
            }
        }
        conn_offsets.push(conn_targets.len()); // sentinel for last unit

        let connections = CsrConnections {
            targets: conn_targets,
            weights: conn_weights,
            offsets: conn_offsets,
        };

        let eligibility = vec![0.0; connections.weights.len()];

        let activity_trace = vec![0.0; cfg.unit_count];

        let pending_input = vec![0.0; cfg.unit_count];
        let reserved = vec![false; cfg.unit_count];
        let learning_enabled = vec![true; cfg.unit_count];
        let sensor_member = vec![false; cfg.unit_count];
        let group_member = vec![false; cfg.unit_count];

        let unit_module = vec![NO_MODULE; cfg.unit_count];

        let mut symbols: HashMap<String, SymbolId> = HashMap::new();
        let mut symbols_rev: Vec<String> = Vec::new();
        let sensor_group_index: HashMap<String, usize> = HashMap::new();

        let routing_modules: Vec<RoutingModule> = Vec::new();
        let routing_module_index: HashMap<String, u16> = HashMap::new();
        let learning_route_modules: Vec<u16> = Vec::new();
        let module_unit_counts: Vec<u32> = Vec::new();
        let module_unit_counts_dirty = true;

        // Reserve reward symbols up front.
        let reward_pos_symbol = intern_symbol(&mut symbols, &mut symbols_rev, "reward_pos");
        let reward_neg_symbol = intern_symbol(&mut symbols, &mut symbols_rev, "reward_neg");

        let causal = CausalMemory::new(cfg.causal_decay);

        Self {
            cfg,
            units,
            activity_trace,
            growth_eligibility_norm_ema: 0.0,
            growth_commit_ema: 0.0,
            growth_prune_norm_ema: 0.0,
            growth_last_birth_step: 0,
            causal_lag_history: Vec::new(),
            connections,
            eligibility,
            tier: ExecutionTier::default(),
            sensor_groups: Vec::new(),
            action_groups: Vec::new(),
            latent_groups: Vec::new(),
            sensor_group_index,

            routing_modules,
            routing_module_index,
            unit_module,
            learning_route_modules,
            module_unit_counts,
            module_unit_counts_dirty,
            latent_auto_last_create_step: 0,
            latent_auto_seq: 0,
            pending_input,
            neuromod: 0.0,
            pruned_last_step: 0,
            births_last_step: 0,
            csr_tombstones: 0,
            rng,
            reserved,
            learning_enabled,

            sensor_member,
            group_member,

            symbols,
            symbols_rev,
            active_symbols: Vec::with_capacity(32),
            causal,
            reward_pos_symbol,
            reward_neg_symbol,

            age_steps: 0,
            telemetry: Telemetry::default(),
            learning_monitors: LearningMonitors::default(),
        }
    }

    fn ensure_routing_module(&mut self, kind: &str, name: &str) -> u16 {
        let prefix = match kind {
            "sensor" => "sensor::",
            "action" => "action::",
            "latent" => "latent::",
            _ => "other::",
        };
        let mut key = String::with_capacity(prefix.len() + name.len());
        key.push_str(prefix);
        key.push_str(name);

        if let Some(&idx) = self.routing_module_index.get(&key) {
            return idx;
        }

        let idx: u16 = self
            .routing_modules
            .len()
            .try_into()
            .unwrap_or(u16::MAX - 1);

        self.routing_modules.push(RoutingModule {
            name: name.to_string(),
            signature: HashMap::new(),
            reward_ema: 0.0,
            last_routed_step: 0,
        });

        // Keep derived caches sized; counts will be recomputed lazily.
        if self.module_unit_counts.len() < self.routing_modules.len() {
            self.module_unit_counts
                .resize(self.routing_modules.len(), 0);
        }
        self.module_unit_counts_dirty = true;

        self.routing_module_index.insert(key, idx);
        idx
    }

    #[inline]
    fn learning_allowed_for_unit(&self, unit: usize) -> bool {
        if self.cfg.module_routing_top_k == 0 {
            return true;
        }
        if self.learning_route_modules.is_empty() {
            // Router is enabled but currently uninformative: do not gate.
            return true;
        }
        let mid = self.unit_module.get(unit).copied().unwrap_or(NO_MODULE);
        if mid == NO_MODULE {
            return !self.cfg.module_routing_strict;
        }
        self.learning_route_modules.contains(&mid)
    }

    fn refresh_module_unit_counts_if_dirty(&mut self) {
        if !self.module_unit_counts_dirty {
            return;
        }
        self.module_unit_counts_dirty = false;

        let module_count = self.routing_modules.len();
        self.module_unit_counts.clear();
        self.module_unit_counts.resize(module_count, 0);

        for &mid in &self.unit_module {
            if mid == NO_MODULE {
                continue;
            }
            let idx = mid as usize;
            if idx < self.module_unit_counts.len() {
                self.module_unit_counts[idx] = self.module_unit_counts[idx].saturating_add(1);
            }
        }
    }

    #[inline]
    fn module_has_units_cached(&self, mid: u16) -> bool {
        self.module_unit_counts
            .get(mid as usize)
            .copied()
            .unwrap_or(0)
            > 0
    }

    fn route_modules_from_symbols(&mut self, symbols: &[SymbolId]) -> Vec<u16> {
        let top_k = self.cfg.module_routing_top_k as usize;
        if top_k == 0 {
            return Vec::new();
        }
        if self.routing_modules.is_empty() {
            return Vec::new();
        }

        self.refresh_module_unit_counts_if_dirty();

        // Build a score per module.
        let beta = self.cfg.module_routing_beta;
        let mut scored: Vec<(u16, f32)> = Vec::with_capacity(self.routing_modules.len());
        for (i, m) in self.routing_modules.iter().enumerate() {
            let mid = i as u16;
            if !self.module_has_units_cached(mid) {
                continue;
            }
            let mut score = beta * m.reward_ema;

            // Seed match: if a boundary symbol name matches module name, prefer it.
            for &sid in symbols {
                if let Some(sym) = self.symbols_rev.get(sid as usize) {
                    if sym.as_str() == m.name.as_str() {
                        score += 1.0;
                    }
                }
                if let Some(v) = m.signature.get(&sid) {
                    score += *v;
                }
            }

            scored.push((mid, score));
        }

        if scored.is_empty() {
            return Vec::new();
        }

        // If all scores are ~zero, do not gate.
        let best = scored
            .iter()
            .map(|(_i, s)| *s)
            .fold(f32::NEG_INFINITY, f32::max);
        if !best.is_finite() || best <= 0.0 {
            return Vec::new();
        }

        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(core::cmp::Ordering::Equal));
        scored.into_iter().take(top_k).map(|(i, _)| i).collect()
    }

    fn update_routing_signatures(&mut self, routed: &[u16], symbols: &[SymbolId], reward: f32) {
        let decay = self.cfg.module_signature_decay;
        let cap = self.cfg.module_signature_cap.max(1) as usize;
        if decay <= 0.0 {
            return;
        }

        for &mid in routed {
            let Some(m) = self.routing_modules.get_mut(mid as usize) else {
                continue;
            };

            // Decay existing association weights.
            for v in m.signature.values_mut() {
                *v *= 1.0 - decay;
            }

            // Add current symbols.
            for &sid in symbols {
                let entry = m.signature.entry(sid).or_insert(0.0);
                *entry += 1.0;
            }

            // Keep bounded size by dropping the weakest entries.
            if m.signature.len() > cap {
                let mut entries: Vec<(SymbolId, f32)> =
                    m.signature.iter().map(|(k, v)| (*k, *v)).collect();
                entries.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(core::cmp::Ordering::Equal));
                entries.truncate(cap);
                m.signature.clear();
                for (k, v) in entries {
                    m.signature.insert(k, v);
                }
            }

            // Reward EMA (bounded, cheap summary).
            m.reward_ema = (1.0 - decay) * m.reward_ema + decay * reward;
        }
    }

    fn routing_symbol_set_is_novel(&self, symbols: &[SymbolId]) -> bool {
        if symbols.is_empty() {
            return false;
        }

        // Consider the set "novel" if any committed symbol has never been
        // associated with any existing module signature.
        for &sid in symbols {
            let mut seen = false;
            for m in &self.routing_modules {
                if m.signature.contains_key(&sid) {
                    seen = true;
                    break;
                }
            }
            if !seen {
                return true;
            }
        }

        false
    }

    fn count_active_latent_modules(&mut self) -> u32 {
        let mids: Vec<u16> = self
            .latent_groups
            .iter()
            .filter_map(|g| {
                let mut key = String::with_capacity("latent::".len() + g.name.len());
                key.push_str("latent::");
                key.push_str(g.name.as_str());
                self.routing_module_index.get(&key).copied()
            })
            .collect();

        self.refresh_module_unit_counts_if_dirty();

        let mut n: u32 = 0;
        for mid in mids {
            if self.module_has_units_cached(mid) {
                n = n.saturating_add(1);
            }
        }
        n
    }

    fn maybe_auto_create_latent_module(&mut self, symbols: &[SymbolId]) -> Option<u16> {
        if !self.cfg.latent_module_auto_create {
            return None;
        }
        if self.cfg.module_routing_top_k == 0 {
            return None;
        }
        if self.cfg.latent_module_auto_width == 0 {
            return None;
        }

        // Require reward signal (positive or negative) to avoid churning modules
        // during low-salience background activity.
        if self.neuromod.abs() < self.cfg.latent_module_auto_reward_threshold {
            return None;
        }

        // Cooldown between auto-created modules.
        let since = self
            .age_steps
            .saturating_sub(self.latent_auto_last_create_step);
        if since < self.cfg.latent_module_auto_cooldown_steps as u64 {
            return None;
        }

        // Respect max active latent modules when configured.
        if self.cfg.latent_module_auto_max_active > 0 {
            let active = self.count_active_latent_modules();
            if active >= self.cfg.latent_module_auto_max_active {
                return None;
            }
        }

        // Only create when routing appears uninformative and the symbol set is novel.
        if !self.routing_symbol_set_is_novel(symbols) {
            return None;
        }

        let width = self.cfg.latent_module_auto_width as usize;
        if width == 0 {
            return None;
        }

        // Generate a unique, non-colliding module name.
        for _ in 0..10_000 {
            self.latent_auto_seq = self.latent_auto_seq.wrapping_add(1);
            let name = format!("auto_latent_{:06}", self.latent_auto_seq);

            let collides = self.sensor_groups.iter().any(|g| g.name == name)
                || self.action_groups.iter().any(|g| g.name == name)
                || self.latent_groups.iter().any(|g| g.name == name);
            if collides {
                continue;
            }

            self.define_module(&name, width);
            self.latent_auto_last_create_step = self.age_steps;

            let mid = self.ensure_routing_module("latent", &name);
            return Some(mid);
        }

        None
    }

    fn maybe_retire_latent_modules(&mut self) {
        let retire_after = self.cfg.latent_module_retire_after_steps;
        if retire_after == 0 {
            return;
        }

        let reward_thr = self.cfg.latent_module_retire_reward_threshold;
        let now = self.age_steps;

        // Iterate from the end so removals preserve stable indices elsewhere.
        for idx in (0..self.latent_groups.len()).rev() {
            let gname = self.latent_groups[idx].name.clone();

            let mut key = String::with_capacity("latent::".len() + gname.len());
            key.push_str("latent::");
            key.push_str(gname.as_str());

            let Some(&mid) = self.routing_module_index.get(&key) else {
                continue;
            };
            let Some(m) = self.routing_modules.get(mid as usize) else {
                continue;
            };

            let since_routed = now.saturating_sub(m.last_routed_step);
            if since_routed < retire_after as u64 {
                continue;
            }
            if m.reward_ema.abs() >= reward_thr {
                continue;
            }

            // Retire: unassign units from the module and drop the latent group.
            let units = core::mem::take(&mut self.latent_groups[idx].units);
            for id in units {
                if id < self.unit_module.len() {
                    self.unit_module[id] = NO_MODULE;
                }
                if id < self.group_member.len() {
                    self.group_member[id] = false;
                }
            }
            self.latent_groups.remove(idx);
            self.module_unit_counts_dirty = true;

            // Clear routing state so the retired module does not bias future routing.
            if let Some(m) = self.routing_modules.get_mut(mid as usize) {
                m.signature.clear();
                m.reward_ema = 0.0;
            }
        }
    }

    #[allow(dead_code)]
    fn rebuild_sensor_group_index(&mut self) {
        self.sensor_group_index.clear();
        for (idx, g) in self.sensor_groups.iter().enumerate() {
            self.sensor_group_index.insert(g.name.clone(), idx);
        }
    }

    #[allow(dead_code)]
    fn rebuild_group_membership(&mut self) {
        self.sensor_member.fill(false);
        self.group_member.fill(false);

        for g in &self.sensor_groups {
            for &id in &g.units {
                if id < self.sensor_member.len() {
                    self.sensor_member[id] = true;
                }
                if id < self.group_member.len() {
                    self.group_member[id] = true;
                }
            }
        }

        for g in &self.action_groups {
            for &id in &g.units {
                if id < self.group_member.len() {
                    self.group_member[id] = true;
                }
            }
        }

        for g in &self.latent_groups {
            for &id in &g.units {
                if id < self.group_member.len() {
                    self.group_member[id] = true;
                }
            }
        }
    }

    #[cfg(feature = "std")]
    fn rebuild_routing_from_groups(&mut self) {
        self.routing_modules.clear();
        self.routing_module_index.clear();
        self.module_unit_counts.clear();
        self.module_unit_counts_dirty = true;

        if self.unit_module.len() != self.units.len() {
            self.unit_module.resize(self.units.len(), NO_MODULE);
        }
        self.unit_module.fill(NO_MODULE);

        // Avoid borrow conflicts by snapshotting group definitions.
        let sensor_groups: Vec<(String, Vec<UnitId>)> = self
            .sensor_groups
            .iter()
            .map(|g| (g.name.clone(), g.units.clone()))
            .collect();
        for (name, units) in sensor_groups {
            let module = self.ensure_routing_module("sensor", name.as_str());
            for id in units {
                if id < self.unit_module.len() {
                    self.unit_module[id] = module;
                }
            }
        }

        let action_groups: Vec<(String, Vec<UnitId>)> = self
            .action_groups
            .iter()
            .map(|g| (g.name.clone(), g.units.clone()))
            .collect();
        for (name, units) in action_groups {
            let module = self.ensure_routing_module("action", name.as_str());
            for id in units {
                if id < self.unit_module.len() {
                    self.unit_module[id] = module;
                }
            }
        }

        let latent_groups: Vec<(String, Vec<UnitId>)> = self
            .latent_groups
            .iter()
            .map(|g| (g.name.clone(), g.units.clone()))
            .collect();
        for (name, units) in latent_groups {
            let module = self.ensure_routing_module("latent", name.as_str());
            for id in units {
                if id < self.unit_module.len() {
                    self.unit_module[id] = module;
                }
            }
        }

        self.module_unit_counts_dirty = true;
    }

    // =========================================================================
    // Execution Tier Configuration
    // =========================================================================

    /// Set the execution tier for step() and learning updates.
    ///
    /// - `Scalar`: Default, works everywhere (MCU, WASM, desktop)
    /// - `Simd`: Single-threaded SIMD (requires `simd` feature)
    /// - `Parallel`: Multi-threaded (requires `parallel` feature)
    pub fn set_execution_tier(&mut self, tier: ExecutionTier) {
        self.tier = tier;
    }

    /// Get the current execution tier.
    pub fn execution_tier(&self) -> ExecutionTier {
        self.tier
    }

    /// Returns the effective execution tier that will actually be used.
    ///
    /// This accounts for compile-time feature gates (e.g. `simd`, `parallel`, `gpu`)
    /// and for runtime GPU availability.
    pub fn effective_execution_tier(&self) -> ExecutionTier {
        match self.tier {
            ExecutionTier::Scalar => ExecutionTier::Scalar,
            ExecutionTier::Simd => {
                #[cfg(feature = "simd")]
                {
                    ExecutionTier::Simd
                }
                #[cfg(not(feature = "simd"))]
                {
                    ExecutionTier::Scalar
                }
            }
            ExecutionTier::Parallel => {
                #[cfg(feature = "parallel")]
                {
                    ExecutionTier::Parallel
                }
                #[cfg(not(feature = "parallel"))]
                {
                    ExecutionTier::Scalar
                }
            }
            ExecutionTier::Gpu => {
                #[cfg(feature = "gpu")]
                {
                    let max_units = self.units.len().max(65_536);
                    if crate::gpu::gpu_available(max_units) {
                        ExecutionTier::Gpu
                    } else {
                        ExecutionTier::Scalar
                    }
                }
                #[cfg(not(feature = "gpu"))]
                {
                    ExecutionTier::Scalar
                }
            }
        }
    }

    /// Select a default execution tier.
    ///
    /// Picks GPU when compiled+available, otherwise falls back to Parallel/Simd/Scalar.
    /// Returns the effective tier selected.
    pub fn auto_select_execution_tier(&mut self) -> ExecutionTier {
        #[cfg(feature = "gpu")]
        {
            let max_units = self.units.len().max(65_536);
            if crate::gpu::gpu_available(max_units) {
                self.tier = ExecutionTier::Gpu;
                return ExecutionTier::Gpu;
            }
        }

        #[cfg(feature = "parallel")]
        {
            self.tier = ExecutionTier::Parallel;
            ExecutionTier::Parallel
        }

        #[cfg(all(not(feature = "parallel"), feature = "simd"))]
        {
            self.tier = ExecutionTier::Simd;
            ExecutionTier::Simd
        }

        #[cfg(all(not(feature = "parallel"), not(feature = "simd")))]
        {
            self.tier = ExecutionTier::Scalar;
            ExecutionTier::Scalar
        }
    }

    // =========================================================================
    // CSR Connection Helpers
    // =========================================================================

    /// Returns an iterator over (target, weight) for unit `i`'s outgoing connections.
    #[inline]
    pub fn neighbors(&self, i: UnitId) -> impl Iterator<Item = (UnitId, f32)> + '_ {
        let start = self.connections.offsets[i];
        let end = self.connections.offsets[i + 1];
        self.connections.targets[start..end]
            .iter()
            .copied()
            .zip(self.connections.weights[start..end].iter().copied())
            .filter(|(t, _)| *t != INVALID_UNIT)
    }

    /// Returns the range of indices in the CSR arrays for unit `i`'s connections.
    #[inline]
    fn conn_range(&self, i: UnitId) -> Range<usize> {
        self.connections.offsets[i]..self.connections.offsets[i + 1]
    }

    /// Count of valid (non-pruned) connections for unit `i`.
    #[allow(dead_code)]
    fn conn_count(&self, i: UnitId) -> usize {
        self.neighbors(i).count()
    }

    /// Total connection count across all units.
    fn total_connection_count(&self) -> usize {
        self.connections
            .targets
            .iter()
            .filter(|&&t| t != INVALID_UNIT)
            .count()
    }

    /// Add or bump a connection from `from` to `target` by `bump`.
    /// If connection exists, bumps weight. Otherwise appends to CSR (may require realloc).
    fn add_or_bump_csr(&mut self, from: UnitId, target: UnitId, bump: f32) {
        let range = self.conn_range(from);

        // First, try to find existing connection or a tombstone slot
        for idx in range.clone() {
            if self.connections.targets[idx] == target {
                // Existing connection: bump weight
                self.connections.weights[idx] =
                    (self.connections.weights[idx] + bump).clamp(-1.5, 1.5);
                return;
            }
        }

        // Try to reuse a tombstone slot within this unit's range
        for idx in range {
            if self.connections.targets[idx] == INVALID_UNIT {
                self.connections.targets[idx] = target;
                self.connections.weights[idx] = bump.clamp(-1.5, 1.5);
                if idx < self.eligibility.len() {
                    self.eligibility[idx] = 0.0;
                }
                self.csr_tombstones = self.csr_tombstones.saturating_sub(1);
                return;
            }
        }

        // No slot available: must append (requires CSR rebuild).
        // This is expensive but rare after initial wiring stabilizes.
        self.append_connection(from, target, bump.clamp(-1.5, 1.5));
    }

    /// Append a new connection (rebuilds CSR structure - expensive, use sparingly).
    fn append_connection(&mut self, from: UnitId, target: UnitId, weight: f32) {
        // Insert at the end of `from`'s segment, shifting later units.
        let insert_pos = self.connections.offsets[from + 1];

        self.connections.targets.insert(insert_pos, target);
        self.connections.weights.insert(insert_pos, weight);
        self.eligibility.insert(insert_pos, 0.0);

        // Update offsets for all units after `from`.
        for i in (from + 1)..self.connections.offsets.len() {
            self.connections.offsets[i] += 1;
        }
    }

    /// Compact the CSR by removing tombstoned entries. Call periodically.
    fn compact_connections(&mut self) {
        let unit_count = self.units.len();
        let mut new_targets = Vec::with_capacity(self.connections.targets.len());
        let mut new_weights = Vec::with_capacity(self.connections.weights.len());
        let mut new_eligibility = Vec::with_capacity(self.eligibility.len());
        let mut new_offsets = Vec::with_capacity(unit_count + 1);

        for i in 0..unit_count {
            new_offsets.push(new_targets.len());
            let range = self.conn_range(i);
            for idx in range {
                let t = self.connections.targets[idx];
                if t != INVALID_UNIT {
                    new_targets.push(t);
                    new_weights.push(self.connections.weights[idx]);
                    new_eligibility.push(self.eligibility.get(idx).copied().unwrap_or(0.0));
                }
            }
        }
        new_offsets.push(new_targets.len());

        self.connections.targets = new_targets;
        self.connections.weights = new_weights;
        self.connections.offsets = new_offsets;
        self.eligibility = new_eligibility;

        // All tombstones are removed by compaction.
        self.csr_tombstones = 0;
    }

    // =========================================================================
    // Public API
    // =========================================================================

    /// Enable/disable observer telemetry.
    /// When enabled, the brain records a small summary of what happened each loop.
    /// Observers read this data without mutating the functional state.
    pub fn set_observer_telemetry(&mut self, enabled: bool) {
        self.telemetry.enabled = enabled;
        if enabled {
            // Pre-allocate small buffers to avoid per-step allocations.
            if self.telemetry.last_stimuli.capacity() < 8 {
                self.telemetry.last_stimuli.reserve(8);
            }
            if self.telemetry.last_actions.capacity() < 8 {
                self.telemetry.last_actions.reserve(8);
            }
            if self.telemetry.last_reinforced_actions.capacity() < 8 {
                self.telemetry.last_reinforced_actions.reserve(8);
            }
            if self.telemetry.last_committed_symbols.capacity() < 16 {
                self.telemetry.last_committed_symbols.reserve(16);
            }
        }
    }

    /// Returns the number of simulation steps since creation.
    #[must_use]
    pub fn age_steps(&self) -> u64 {
        self.age_steps
    }

    /// Returns the current neuromodulator (reward/salience) level.
    ///
    /// Neuromodulator scales learning rate: positive values increase plasticity.
    #[must_use]
    pub fn neuromodulator(&self) -> f32 {
        self.neuromod
    }

    /// Returns statistics about the causal memory.
    #[must_use]
    pub fn causal_stats(&self) -> crate::causality::CausalStats {
        self.causal.stats()
    }

    /// Returns causal graph data for visualization.
    ///
    /// Returns:
    /// - `nodes`: Top N symbols with their names and base counts
    /// - `edges`: Top M causal edges with from/to symbol IDs and strength
    #[must_use]
    pub fn causal_graph_viz(&self, max_nodes: usize, max_edges: usize) -> CausalGraphViz {
        let max_nodes = max_nodes.max(1);

        // Pick edges first; then ensure all edge endpoints are present as nodes.
        // Otherwise the renderer will skip edges whose endpoints aren't in the node set.
        let top_edges = self.causal.top_edges(max_edges);

        let mut endpoint_ids: Vec<SymbolId> = Vec::with_capacity(top_edges.len().saturating_mul(2));
        for (from, to, _strength) in &top_edges {
            endpoint_ids.push(*from);
            endpoint_ids.push(*to);
        }

        endpoint_ids.sort_unstable();
        endpoint_ids.dedup();

        // Order endpoints by base count so the most salient endpoints get kept
        // if we hit the `max_nodes` cap.
        endpoint_ids.sort_by(|&a, &b| {
            let ca = self.causal.base_count(a);
            let cb = self.causal.base_count(b);
            cb.total_cmp(&ca)
        });

        let mut node_ids: Vec<SymbolId> = Vec::with_capacity(max_nodes);
        let mut seen: HashSet<SymbolId> = HashSet::with_capacity(max_nodes * 2);

        for id in endpoint_ids {
            if node_ids.len() >= max_nodes {
                break;
            }
            if seen.insert(id) {
                node_ids.push(id);
            }
        }

        // Fill remaining slots with the most frequent symbols.
        if node_ids.len() < max_nodes {
            for (id, _count) in self.causal.all_symbols_sorted(max_nodes) {
                if node_ids.len() >= max_nodes {
                    break;
                }
                if seen.insert(id) {
                    node_ids.push(id);
                }
            }
        }

        let nodes: Vec<CausalNodeViz> = node_ids
            .into_iter()
            .map(|id| CausalNodeViz {
                id,
                name: self.symbol_name(id).unwrap_or("?").to_string(),
                base_count: self.causal.base_count(id),
            })
            .collect();

        let edges: Vec<CausalEdgeViz> = top_edges
            .into_iter()
            .filter(|(from, to, _strength)| seen.contains(from) && seen.contains(to))
            .map(|(from, to, strength)| CausalEdgeViz { from, to, strength })
            .collect();

        CausalGraphViz { nodes, edges }
    }

    /// Looks up a symbol name by its ID.
    #[must_use]
    pub fn symbol_name(&self, id: SymbolId) -> Option<&str> {
        self.symbols_rev.get(id as usize).map(|s| s.as_str())
    }

    /// Returns the symbol IDs of stimuli applied in the last step.
    ///
    /// Requires telemetry to be enabled via [`set_observer_telemetry`].
    #[must_use]
    pub fn last_stimuli_symbols(&self) -> &[SymbolId] {
        &self.telemetry.last_stimuli
    }

    /// Returns the symbol IDs of actions noted in the last step.
    ///
    /// Requires telemetry to be enabled via [`set_observer_telemetry`].
    #[must_use]
    pub fn last_action_symbols(&self) -> &[SymbolId] {
        &self.telemetry.last_actions
    }

    /// Returns the symbol IDs and delta biases of reinforced actions.
    ///
    /// Requires telemetry to be enabled via [`set_observer_telemetry`].
    #[must_use]
    pub fn last_reinforced_action_symbols(&self) -> &[(SymbolId, f32)] {
        &self.telemetry.last_reinforced_actions
    }

    /// Returns all symbol IDs committed to causal memory in the last observation.
    ///
    /// Requires telemetry to be enabled via [`set_observer_telemetry`].
    #[must_use]
    pub fn last_committed_symbols(&self) -> &[SymbolId] {
        &self.telemetry.last_committed_symbols
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Serialization (std-only)
    // ─────────────────────────────────────────────────────────────────────────

    /// Serialize a versioned, chunked "brain image".
    ///
    /// This is std-only and intended to be capacity-aware when paired with
    /// `storage::CapacityWriter`.
    #[cfg(feature = "std")]
    pub fn save_image_to<W: Write>(&self, w: &mut W) -> io::Result<()> {
        self.save_image_v3_to(w)
    }

    #[cfg(feature = "std")]
    fn save_image_v3_to<W: Write>(&self, w: &mut W) -> io::Result<()> {
        w.write_all(storage::MAGIC)?;
        storage::write_u32_le(w, storage::VERSION_CURRENT)?;

        self.write_cfg_chunk_v2(w)?;
        self.write_prng_chunk_v2(w)?;
        self.write_stat_chunk_v2(w)?;
        self.write_unit_chunk_v2(w)?;
        self.write_mask_chunk_v2(w)?;
        self.write_salience_chunk_v2(w)?;
        self.write_groups_chunk_v2(w)?;
        self.write_latent_modules_chunk_v2(w)?;
        self.write_symbols_chunk_v2(w)?;
        self.write_causality_chunk_v2(w)?;
        Ok(())
    }

    /// Load a versioned, chunked "brain image".
    ///
    /// Unknown chunks are skipped for forward-compatibility.
    #[cfg(feature = "std")]
    pub fn load_image_from<R: Read>(r: &mut R) -> io::Result<Self> {
        let magic = storage::read_exact::<8, _>(r)?;
        if &magic != storage::MAGIC {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "bad brain image magic",
            ));
        }

        let version = storage::read_u32_le(r)?;
        if version != storage::VERSION_CURRENT {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "unsupported brain image version",
            ));
        }

        let mut cfg: Option<BrainConfig> = None;
        let mut rng_state: Option<u64> = None;
        let mut age_steps: Option<u64> = None;
        let mut units: Option<Vec<Unit>> = None;
        let mut connections: Option<CsrConnections> = None;
        let mut reserved: Option<Vec<bool>> = None;
        let mut learning_enabled: Option<Vec<bool>> = None;
        let mut salience: Option<Vec<f32>> = None;
        let mut sensor_groups: Option<Vec<NamedGroup>> = None;
        let mut action_groups: Option<Vec<NamedGroup>> = None;
        let mut latent_groups: Option<Vec<NamedGroup>> = None;
        let mut symbols_rev: Option<Vec<String>> = None;
        let mut causal: Option<CausalMemory> = None;

        loop {
            let (tag, len) = match storage::read_chunk_header(r) {
                Ok(v) => v,
                Err(e) if e.kind() == io::ErrorKind::UnexpectedEof => break,
                Err(e) => return Err(e),
            };

            // Current chunks are LZ4-compressed.
            let payload = {
                let mut take = r.take(len as u64);
                let uncompressed_len = storage::read_u32_le(&mut take)? as usize;
                let mut compressed = Vec::with_capacity((len as usize).saturating_sub(4));
                take.read_to_end(&mut compressed)?;
                let decompressed = storage::decompress_lz4(&compressed, uncompressed_len)?;
                io::copy(&mut take, &mut io::sink())?;
                decompressed
            };

            let mut cursor = io::Cursor::new(payload);
            match &tag {
                b"CFG0" => cfg = Some(Self::read_cfg_payload(&mut cursor)?),
                b"PRNG" => rng_state = Some(storage::read_u64_le(&mut cursor)?),
                b"STAT" => age_steps = Some(storage::read_u64_le(&mut cursor)?),
                b"UNIT" => {
                    let (u, c) = Self::read_unit_payload(&mut cursor)?;
                    units = Some(u);
                    connections = Some(c);
                }
                b"MASK" => {
                    let (rsv, learn) = Self::read_mask_payload(&mut cursor)?;
                    reserved = Some(rsv);
                    learning_enabled = Some(learn);
                }
                b"SALI" => salience = Some(Self::read_salience_payload(&mut cursor)?),
                b"GRPS" => {
                    let (sg, ag) = Self::read_groups_payload(&mut cursor)?;
                    sensor_groups = Some(sg);
                    action_groups = Some(ag);
                }
                b"LMOD" => latent_groups = Some(Self::read_latent_modules_payload(&mut cursor)?),
                b"SYMB" => symbols_rev = Some(Self::read_symbols_payload(&mut cursor)?),
                b"CAUS" => causal = Some(CausalMemory::read_image_payload(&mut cursor)?),
                _ => {
                    // Unknown chunk: skipped.
                }
            }
        }

        let cfg = cfg.ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "missing CFG0"))?;
        let units =
            units.ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "missing UNIT"))?;
        let connections = connections.ok_or_else(|| {
            io::Error::new(io::ErrorKind::InvalidData, "missing UNIT connections")
        })?;
        if cfg.unit_count != units.len() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "CFG0 unit_count mismatch",
            ));
        }
        let unit_count = cfg.unit_count;

        let reserved =
            reserved.ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "missing MASK"))?;
        let learning_enabled = learning_enabled
            .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "missing MASK"))?;
        if reserved.len() != unit_count || learning_enabled.len() != unit_count {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "MASK length mismatch",
            ));
        }

        let sensor_groups = sensor_groups
            .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "missing GRPS"))?;
        let action_groups = action_groups
            .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "missing GRPS"))?;
        let latent_groups = latent_groups.unwrap_or_default();

        let symbols_rev = symbols_rev
            .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "missing SYMB"))?;
        let (symbols, reward_pos_symbol, reward_neg_symbol) =
            Self::rebuild_symbol_tables(&symbols_rev)?;

        let causal =
            causal.ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "missing CAUS"))?;
        let rng_state =
            rng_state.ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "missing PRNG"))?;

        let age_steps = age_steps.unwrap_or(0);

        // Apply salience from SALI chunk if present (backwards compatible).
        // Older images without SALI chunk will have salience = 0.0 (already set in read_unit_payload).
        let mut units = units;
        if let Some(sal) = salience {
            for (i, s) in sal.into_iter().enumerate() {
                if i < units.len() {
                    units[i].salience = s;
                }
            }
        }

        // Initialize activity trace from current activation.
        let activity_trace: Vec<f32> = units.iter().map(|u| u.amp.max(0.0)).collect();

        let eligibility_len = connections.weights.len();

        let mut brain = Self {
            cfg,
            units,
            activity_trace,
            growth_eligibility_norm_ema: 0.0,
            growth_commit_ema: 0.0,
            growth_prune_norm_ema: 0.0,
            growth_last_birth_step: 0,
            causal_lag_history: Vec::new(),
            connections,
            eligibility: vec![0.0; eligibility_len],
            tier: ExecutionTier::default(),
            rng: Prng::from_state(rng_state),
            reserved,
            sensor_member: vec![false; unit_count],
            group_member: vec![false; unit_count],
            learning_enabled,
            sensor_groups,
            sensor_group_index: HashMap::new(),
            action_groups,
            latent_groups,

            routing_modules: Vec::new(),
            routing_module_index: HashMap::new(),
            unit_module: vec![NO_MODULE; unit_count],
            learning_route_modules: Vec::new(),
            module_unit_counts: Vec::new(),
            module_unit_counts_dirty: true,
            latent_auto_last_create_step: 0,
            latent_auto_seq: 0,
            pending_input: vec![0.0; unit_count],
            neuromod: 0.0,
            symbols,
            symbols_rev,
            active_symbols: Vec::with_capacity(32),
            causal,
            reward_pos_symbol,
            reward_neg_symbol,
            pruned_last_step: 0,
            births_last_step: 0,
            csr_tombstones: 0,
            age_steps,
            telemetry: Telemetry::default(),
            learning_monitors: LearningMonitors::default(),
        };

        brain.rebuild_group_membership();
        brain.rebuild_sensor_group_index();
        brain.rebuild_routing_from_groups();
        Ok(brain)
    }

    /// Exact serialized size in bytes for the current brain image.
    #[cfg(feature = "std")]
    pub fn image_size_bytes(&self) -> io::Result<usize> {
        let mut cw = storage::CountingWriter::new();
        self.save_image_to(&mut cw)?;
        Ok(cw.written())
    }

    // -------------------------------------------------------------------------
    // WASM-friendly byte array persistence API
    // -------------------------------------------------------------------------

    /// Serialize the brain image to a byte vector.
    ///
    /// This is the primary API for WASM targets, where the caller can then
    /// persist the bytes to IndexedDB or send over the wire.
    #[cfg(feature = "std")]
    pub fn save_image_bytes(&self) -> io::Result<Vec<u8>> {
        let mut buf = Vec::new();
        self.save_image_to(&mut buf)?;
        Ok(buf)
    }

    /// Load a brain image from a byte slice.
    ///
    /// This is the primary API for WASM targets, where the caller can load
    /// bytes from IndexedDB or receive over the wire.
    #[cfg(feature = "std")]
    pub fn load_image_bytes(bytes: &[u8]) -> io::Result<Self> {
        let mut cursor = io::Cursor::new(bytes);
        Self::load_image_from(&mut cursor)
    }

    #[cfg(feature = "std")]
    fn write_cfg_chunk_v2<W: Write>(&self, w: &mut W) -> io::Result<()> {
        let mut payload: Vec<u8> = Vec::with_capacity(Self::cfg_payload_len_bytes() as usize);
        self.write_cfg_payload(&mut payload)?;
        storage::write_chunk_v2_lz4(w, *b"CFG0", &payload)
    }

    #[cfg(feature = "std")]
    fn cfg_payload_len_bytes() -> u32 {
        // Bytes based on exact write order below.
        4  // unit_count
            + 4  // connectivity_per_unit
            + 4  // dt
            + 4  // base_freq
            + 4  // noise_amp
            + 4  // noise_phase
            + 4  // amp_saturation_beta
            + 4  // phase_coupling_mode
            + 4  // phase_coupling_k
            + 4  // phase_coupling_gain
            + 4  // global_inhibition
            + 4  // inhibition_mode
            + 4  // hebb_rate
            + 4  // forget_rate
            + 4  // prune_below
            + 4  // coactive_threshold
            + 4  // phase_lock_threshold
            + 4  // imprint_rate
            + 4  // salience_decay
            + 4  // salience_gain
            + 4  // seed_present
            + 8  // seed
                + 4 // causal_decay
                + 4 // learning_deadband
                + 4 // eligibility_decay
                + 4 // eligibility_gain
                + 4 // coactive_softness
                + 4 // phase_gate_softness
                + 4 // plasticity_budget
                + 4 // homeostasis_target_amp
                + 4 // homeostasis_rate
                + 4 // homeostasis_every
                + 4 // activity_trace_decay
                + 4 // growth_policy_mode
                + 4 // growth_cooldown_steps
                + 4 // growth_signal_alpha
                + 4 // growth_commit_ema_threshold
                + 4 // growth_eligibility_norm_ema_threshold
                + 4 // growth_prune_norm_ema_max
                + 4 // causal_lag_steps
                + 4 // causal_lag_decay
                + 4 // causal_symbol_cap
                + 4 // module_routing_top_k
                + 4 // module_routing_strict
                + 4 // module_routing_beta
                + 4 // module_signature_decay
                + 4 // module_signature_cap
                + 4 // module_learning_activity_threshold
                + 4 // module_plasticity_budget
                + 4 // cross_module_plasticity_scale
                + 4 // cross_module_forget_boost
                + 4 // cross_module_prune_bonus
                + 4 // latent_module_auto_create
                + 4 // latent_module_auto_width
                + 4 // latent_module_auto_cooldown_steps
                + 4 // latent_module_auto_max_active
                + 4 // latent_module_auto_reward_threshold
                + 4 // latent_module_retire_after_steps
                + 4 // latent_module_retire_reward_threshold
    }

    #[cfg(feature = "std")]
    fn write_cfg_payload<W: Write>(&self, w: &mut W) -> io::Result<()> {
        storage::write_u32_le(w, self.cfg.unit_count as u32)?;
        storage::write_u32_le(w, self.cfg.connectivity_per_unit as u32)?;
        storage::write_f32_le(w, self.cfg.dt)?;
        storage::write_f32_le(w, self.cfg.base_freq)?;
        storage::write_f32_le(w, self.cfg.noise_amp)?;
        storage::write_f32_le(w, self.cfg.noise_phase)?;
        storage::write_f32_le(w, self.cfg.amp_saturation_beta)?;
        storage::write_u32_le(w, self.cfg.phase_coupling_mode as u32)?;
        storage::write_f32_le(w, self.cfg.phase_coupling_k)?;
        storage::write_f32_le(w, self.cfg.phase_coupling_gain)?;
        storage::write_f32_le(w, self.cfg.global_inhibition)?;
        storage::write_u32_le(w, self.cfg.inhibition_mode as u32)?;
        storage::write_f32_le(w, self.cfg.hebb_rate)?;
        storage::write_f32_le(w, self.cfg.forget_rate)?;
        storage::write_f32_le(w, self.cfg.prune_below)?;
        storage::write_f32_le(w, self.cfg.coactive_threshold)?;
        storage::write_f32_le(w, self.cfg.phase_lock_threshold)?;
        storage::write_f32_le(w, self.cfg.imprint_rate)?;
        storage::write_f32_le(w, self.cfg.salience_decay)?;
        storage::write_f32_le(w, self.cfg.salience_gain)?;
        storage::write_u32_le(w, if self.cfg.seed.is_some() { 1 } else { 0 })?;
        storage::write_u64_le(w, self.cfg.seed.unwrap_or(0))?;
        storage::write_f32_le(w, self.cfg.causal_decay)?;

        // Control-loop tuning parameters (appended; backwards compatible on load).
        storage::write_f32_le(w, self.cfg.learning_deadband)?;
        storage::write_f32_le(w, self.cfg.eligibility_decay)?;
        storage::write_f32_le(w, self.cfg.eligibility_gain)?;
        storage::write_f32_le(w, self.cfg.coactive_softness)?;
        storage::write_f32_le(w, self.cfg.phase_gate_softness)?;
        storage::write_f32_le(w, self.cfg.plasticity_budget)?;
        storage::write_f32_le(w, self.cfg.homeostasis_target_amp)?;
        storage::write_f32_le(w, self.cfg.homeostasis_rate)?;
        storage::write_u32_le(w, self.cfg.homeostasis_every)?;
        storage::write_f32_le(w, self.cfg.activity_trace_decay)?;

        // Refinement knobs (appended; backwards compatible on load).
        storage::write_u32_le(w, self.cfg.growth_policy_mode as u32)?;
        storage::write_u32_le(w, self.cfg.growth_cooldown_steps)?;
        storage::write_f32_le(w, self.cfg.growth_signal_alpha)?;
        storage::write_f32_le(w, self.cfg.growth_commit_ema_threshold)?;
        storage::write_f32_le(w, self.cfg.growth_eligibility_norm_ema_threshold)?;
        storage::write_f32_le(w, self.cfg.growth_prune_norm_ema_max)?;
        storage::write_u32_le(w, self.cfg.causal_lag_steps as u32)?;
        storage::write_f32_le(w, self.cfg.causal_lag_decay)?;
        storage::write_u32_le(w, self.cfg.causal_symbol_cap as u32)?;

        // Scaling knobs (appended; backwards compatible on load).
        storage::write_u32_le(w, self.cfg.module_routing_top_k as u32)?;
        storage::write_u32_le(w, if self.cfg.module_routing_strict { 1 } else { 0 })?;
        storage::write_f32_le(w, self.cfg.module_routing_beta)?;
        storage::write_f32_le(w, self.cfg.module_signature_decay)?;
        storage::write_u32_le(w, self.cfg.module_signature_cap as u32)?;
        storage::write_f32_le(w, self.cfg.module_learning_activity_threshold)?;
        storage::write_f32_le(w, self.cfg.module_plasticity_budget)?;

        // Phase 3: cross-module coupling governance (appended; backwards compatible on load).
        storage::write_f32_le(w, self.cfg.cross_module_plasticity_scale)?;
        storage::write_f32_le(w, self.cfg.cross_module_forget_boost)?;
        storage::write_f32_le(w, self.cfg.cross_module_prune_bonus)?;

        // Phase 5: latent module auto-formation + retirement (appended; backwards compatible).
        storage::write_u32_le(
            w,
            if self.cfg.latent_module_auto_create {
                1
            } else {
                0
            },
        )?;
        storage::write_u32_le(w, self.cfg.latent_module_auto_width)?;
        storage::write_u32_le(w, self.cfg.latent_module_auto_cooldown_steps)?;
        storage::write_u32_le(w, self.cfg.latent_module_auto_max_active)?;
        storage::write_f32_le(w, self.cfg.latent_module_auto_reward_threshold)?;
        storage::write_u32_le(w, self.cfg.latent_module_retire_after_steps)?;
        storage::write_f32_le(w, self.cfg.latent_module_retire_reward_threshold)?;
        Ok(())
    }

    #[cfg(feature = "std")]
    fn read_cfg_payload<R: Read>(r: &mut R) -> io::Result<BrainConfig> {
        // Parse from a bounded payload buffer so we can safely branch based on
        // remaining bytes and support legacy layouts.
        let mut payload: Vec<u8> = Vec::new();
        r.read_to_end(&mut payload)?;

        fn remaining(c: &io::Cursor<&[u8]>) -> usize {
            (c.get_ref().len() as u64).saturating_sub(c.position()) as usize
        }

        fn read_f32_default(c: &mut io::Cursor<&[u8]>, default: f32) -> f32 {
            if remaining(c) < 4 {
                return default;
            }
            storage::read_f32_le(c).unwrap_or(default)
        }

        fn read_u32_default(c: &mut io::Cursor<&[u8]>, default: u32) -> u32 {
            if remaining(c) < 4 {
                return default;
            }
            storage::read_u32_le(c).unwrap_or(default)
        }

        fn read_u64_default(c: &mut io::Cursor<&[u8]>, default: u64) -> u64 {
            if remaining(c) < 8 {
                return default;
            }
            storage::read_u64_le(c).unwrap_or(default)
        }

        fn parse_layout(
            payload: &[u8],
            expect_phase_fields: bool,
        ) -> io::Result<(BrainConfig, bool)> {
            let mut c = io::Cursor::new(payload);

            let unit_count = storage::read_u32_le(&mut c)? as usize;
            let connectivity_per_unit = storage::read_u32_le(&mut c)? as usize;

            let dt = storage::read_f32_le(&mut c)?;
            let base_freq = storage::read_f32_le(&mut c)?;
            let noise_amp = storage::read_f32_le(&mut c)?;
            let noise_phase = storage::read_f32_le(&mut c)?;
            let amp_saturation_beta = read_f32_default(&mut c, 0.1);

            // Phase coupling fields were introduced in a later layout.
            let (phase_coupling_mode, phase_coupling_k) = if expect_phase_fields {
                let mode = read_u32_default(&mut c, 0) as u8;
                let k = read_f32_default(&mut c, 2.0);
                (mode, k)
            } else {
                (0u8, 2.0)
            };

            let phase_coupling_gain = read_f32_default(&mut c, 1.0);

            let global_inhibition = storage::read_f32_le(&mut c)?;
            let inhibition_mode = read_u32_default(&mut c, 0) as u8;
            let hebb_rate = storage::read_f32_le(&mut c)?;
            let forget_rate = storage::read_f32_le(&mut c)?;
            let prune_below = storage::read_f32_le(&mut c)?;
            let coactive_threshold = storage::read_f32_le(&mut c)?;
            let phase_lock_threshold = storage::read_f32_le(&mut c)?;
            let imprint_rate = storage::read_f32_le(&mut c)?;

            // Salience fields are optional in older images.
            let salience_decay = read_f32_default(&mut c, 0.001);
            let salience_gain = read_f32_default(&mut c, 0.1);

            let seed_present = read_u32_default(&mut c, 0);
            let seed = read_u64_default(&mut c, 0);
            let causal_decay = read_f32_default(&mut c, 0.002);

            // Control-loop tuning parameters.
            let learning_deadband = read_f32_default(&mut c, 0.05);
            let eligibility_decay = read_f32_default(&mut c, 0.02);
            let eligibility_gain = read_f32_default(&mut c, 0.35);

            // Determine whether smoothing knobs are present.
            // Remaining tail is either:
            // - legacy: plasticity_budget + homeostasis_target_amp + homeostasis_rate + homeostasis_every
            // - with smoothing: coactive_softness + phase_gate_softness + (legacy tail)
            let mut coactive_softness = 0.0;
            let mut phase_gate_softness = 0.0;
            let after_elig_remaining = remaining(&c);
            if after_elig_remaining >= 24 {
                coactive_softness = read_f32_default(&mut c, 0.0);
                phase_gate_softness = read_f32_default(&mut c, 0.0);
            }

            let plasticity_budget = read_f32_default(&mut c, 0.0);
            let homeostasis_target_amp = read_f32_default(&mut c, 0.25);
            let homeostasis_rate = read_f32_default(&mut c, 0.0);
            let homeostasis_every = read_u32_default(&mut c, 50);

            // Activity-trace field is appended at the end; optional.
            let activity_trace_decay = read_f32_default(&mut c, 0.0);

            // Optional appended refinement knobs (safe defaults).
            let growth_policy_mode = read_u32_default(&mut c, 0) as u8;
            let growth_cooldown_steps = read_u32_default(&mut c, 250);
            let growth_signal_alpha = read_f32_default(&mut c, 0.05);
            let growth_commit_ema_threshold = read_f32_default(&mut c, 0.2);
            let growth_eligibility_norm_ema_threshold = read_f32_default(&mut c, 0.02);
            let growth_prune_norm_ema_max = read_f32_default(&mut c, 0.0005);
            let causal_lag_steps = read_u32_default(&mut c, 1) as u8;
            let causal_lag_decay = read_f32_default(&mut c, 0.7);
            let causal_symbol_cap = read_u32_default(&mut c, 32) as u8;

            // Optional appended scaling knobs (safe defaults).
            let module_routing_top_k = read_u32_default(&mut c, 0) as u8;
            let module_routing_strict = read_u32_default(&mut c, 0) != 0;
            let module_routing_beta = read_f32_default(&mut c, 0.2);
            let module_signature_decay = read_f32_default(&mut c, 0.01);
            let module_signature_cap = read_u32_default(&mut c, 32) as u8;
            let module_learning_activity_threshold = read_f32_default(&mut c, 0.0);
            let module_plasticity_budget = read_f32_default(&mut c, 0.0);

            // Optional appended phase 3 knobs (safe defaults).
            let cross_module_plasticity_scale = read_f32_default(&mut c, 1.0);
            let cross_module_forget_boost = read_f32_default(&mut c, 0.0);
            let cross_module_prune_bonus = read_f32_default(&mut c, 0.0);

            // Optional appended phase 5 knobs (safe defaults).
            let latent_module_auto_create = read_u32_default(&mut c, 0) != 0;
            let latent_module_auto_width = read_u32_default(&mut c, 8);
            let latent_module_auto_cooldown_steps = read_u32_default(&mut c, 500);
            let latent_module_auto_max_active = read_u32_default(&mut c, 0);
            let latent_module_auto_reward_threshold = read_f32_default(&mut c, 0.2);
            let latent_module_retire_after_steps = read_u32_default(&mut c, 0);
            let latent_module_retire_reward_threshold = read_f32_default(&mut c, 0.05);

            let cfg = BrainConfig {
                unit_count,
                connectivity_per_unit,
                dt,
                base_freq,
                noise_amp,
                noise_phase,
                amp_saturation_beta,
                phase_coupling_mode,
                phase_coupling_k,
                phase_coupling_gain,
                global_inhibition,
                inhibition_mode,
                hebb_rate,
                forget_rate,
                prune_below,
                coactive_threshold,
                phase_lock_threshold,
                imprint_rate,
                salience_decay,
                salience_gain,
                activity_trace_decay,

                growth_policy_mode,
                growth_cooldown_steps,
                growth_signal_alpha,
                growth_commit_ema_threshold,
                growth_eligibility_norm_ema_threshold,
                growth_prune_norm_ema_max,

                causal_lag_steps,
                causal_lag_decay,
                causal_symbol_cap,
                seed: if seed_present != 0 { Some(seed) } else { None },
                causal_decay,
                learning_deadband,
                eligibility_decay,
                eligibility_gain,
                coactive_softness,
                phase_gate_softness,
                plasticity_budget,
                homeostasis_target_amp,
                homeostasis_rate,
                homeostasis_every,

                module_routing_top_k,
                module_routing_strict,
                module_routing_beta,
                module_signature_decay,
                module_signature_cap,
                module_learning_activity_threshold,
                module_plasticity_budget,

                cross_module_plasticity_scale,
                cross_module_forget_boost,
                cross_module_prune_bonus,

                latent_module_auto_create,
                latent_module_auto_width,
                latent_module_auto_cooldown_steps,
                latent_module_auto_max_active,
                latent_module_auto_reward_threshold,
                latent_module_retire_after_steps,
                latent_module_retire_reward_threshold,
            };

            // Basic sanity: only accept if seed_present looks plausible and cfg validates.
            let seed_ok = seed_present == 0 || seed_present == 1;
            let cfg_ok = cfg.validate().is_ok();
            Ok((cfg, seed_ok && cfg_ok))
        }

        // Try the newer layout first (phase coupling fields present), then fall back.
        let (cfg_new, ok_new) = parse_layout(&payload, true)?;
        if ok_new {
            return Ok(cfg_new);
        }

        let (cfg_old, ok_old) = parse_layout(&payload, false)?;
        if ok_old {
            return Ok(cfg_old);
        }

        // If both heuristics fail, return the "new" parse to preserve information.
        Ok(cfg_new)
    }

    #[cfg(feature = "std")]
    fn write_prng_chunk_v2<W: Write>(&self, w: &mut W) -> io::Result<()> {
        let mut payload: Vec<u8> = Vec::with_capacity(8);
        storage::write_u64_le(&mut payload, self.rng.state())?;
        storage::write_chunk_v2_lz4(w, *b"PRNG", &payload)
    }

    #[cfg(feature = "std")]
    fn write_stat_chunk_v2<W: Write>(&self, w: &mut W) -> io::Result<()> {
        let mut payload: Vec<u8> = Vec::with_capacity(8);
        storage::write_u64_le(&mut payload, self.age_steps)?;
        storage::write_chunk_v2_lz4(w, *b"STAT", &payload)
    }

    #[cfg(feature = "std")]
    fn unit_payload_len_bytes(&self) -> io::Result<u32> {
        let mut len: u64 = 0;
        len += 4; // unit_count
                  // Unit scalar state: amp, phase, bias, decay (4 floats per unit).
        len += (self.units.len() as u64) * 4 * 4;
        // CSR connections: offsets, targets, weights.
        len += 4; // total_connections count
        len += (self.connections.offsets.len() as u64) * 4; // offsets (unit_count + 1)
                                                            // Only count valid (non-tombstoned) connections.
        let valid_count = self.total_connection_count() as u64;
        len += valid_count * 4; // targets
        len += valid_count * 4; // weights
        u32::try_from(len)
            .map_err(|_| io::Error::new(io::ErrorKind::InvalidData, "UNIT chunk too large"))
    }

    #[cfg(feature = "std")]
    fn write_unit_chunk_v2<W: Write>(&self, w: &mut W) -> io::Result<()> {
        let mut payload: Vec<u8> = Vec::with_capacity(self.unit_payload_len_bytes()? as usize);

        // Write unit scalars.
        storage::write_u32_le(&mut payload, self.units.len() as u32)?;
        for u in &self.units {
            storage::write_f32_le(&mut payload, u.amp)?;
            storage::write_f32_le(&mut payload, u.phase)?;
            storage::write_f32_le(&mut payload, u.bias)?;
            storage::write_f32_le(&mut payload, u.decay)?;
        }

        // Write CSR connections (compacted: skip tombstones).
        let mut compact_offsets: Vec<usize> = Vec::with_capacity(self.units.len() + 1);
        let mut compact_targets: Vec<UnitId> = Vec::new();
        let mut compact_weights: Vec<f32> = Vec::new();

        for i in 0..self.units.len() {
            compact_offsets.push(compact_targets.len());
            for (target, weight) in self.neighbors(i) {
                compact_targets.push(target);
                compact_weights.push(weight);
            }
        }
        compact_offsets.push(compact_targets.len());

        storage::write_u32_le(&mut payload, compact_targets.len() as u32)?;
        for &off in &compact_offsets {
            storage::write_u32_le(&mut payload, off as u32)?;
        }
        for &t in &compact_targets {
            storage::write_u32_le(&mut payload, t as u32)?;
        }
        for &wt in &compact_weights {
            storage::write_f32_le(&mut payload, wt)?;
        }

        storage::write_chunk_v2_lz4(w, *b"UNIT", &payload)
    }

    #[cfg(feature = "std")]
    fn read_unit_payload<R: Read>(r: &mut R) -> io::Result<(Vec<Unit>, CsrConnections)> {
        let unit_count = storage::read_u32_le(r)? as usize;

        // Read unit scalars.
        // Note: salience is loaded separately from SALI chunk for backwards compatibility.
        let mut units: Vec<Unit> = Vec::with_capacity(unit_count);
        for _ in 0..unit_count {
            let amp = storage::read_f32_le(r)?;
            let phase = storage::read_f32_le(r)?;
            let bias = storage::read_f32_le(r)?;
            let decay = storage::read_f32_le(r)?;
            units.push(Unit {
                amp,
                phase,
                bias,
                decay,
                salience: 0.0, // Default; will be updated from SALI chunk if present
            });
        }

        // Read CSR connections.
        let total_conns = storage::read_u32_le(r)? as usize;
        let mut offsets: Vec<usize> = Vec::with_capacity(unit_count + 1);
        for _ in 0..(unit_count + 1) {
            offsets.push(storage::read_u32_le(r)? as usize);
        }

        let mut targets: Vec<UnitId> = Vec::with_capacity(total_conns);
        for _ in 0..total_conns {
            targets.push(storage::read_u32_le(r)? as usize);
        }

        let mut weights: Vec<f32> = Vec::with_capacity(total_conns);
        for _ in 0..total_conns {
            weights.push(storage::read_f32_le(r)?);
        }

        let connections = CsrConnections {
            targets,
            weights,
            offsets,
        };
        Ok((units, connections))
    }

    #[cfg(feature = "std")]
    fn mask_payload_len_bytes(&self) -> u32 {
        let n = self.units.len() as u32;
        let bytes_len = (n as usize).div_ceil(8);
        // unit_count (u32) + reserved_len (u32) + reserved_bytes + learn_len (u32) + learn_bytes
        4 + 4 + bytes_len as u32 + 4 + bytes_len as u32
    }

    #[cfg(feature = "std")]
    fn write_mask_chunk_v2<W: Write>(&self, w: &mut W) -> io::Result<()> {
        let mut payload: Vec<u8> = Vec::with_capacity(self.mask_payload_len_bytes() as usize);
        let n = self.units.len();
        storage::write_u32_le(&mut payload, n as u32)?;
        let bytes_len = n.div_ceil(8);
        storage::write_u32_le(&mut payload, bytes_len as u32)?;
        Self::write_bool_bits(&mut payload, &self.reserved)?;
        storage::write_u32_le(&mut payload, bytes_len as u32)?;
        Self::write_bool_bits(&mut payload, &self.learning_enabled)?;
        storage::write_chunk_v2_lz4(w, *b"MASK", &payload)
    }

    #[cfg(feature = "std")]
    fn write_bool_bits<W: Write>(w: &mut W, bits: &[bool]) -> io::Result<()> {
        let mut i = 0usize;
        while i < bits.len() {
            let mut byte = 0u8;
            for b in 0..8 {
                let idx = i + b;
                if idx >= bits.len() {
                    break;
                }
                if bits[idx] {
                    byte |= 1u8 << b;
                }
            }
            w.write_all(&[byte])?;
            i += 8;
        }
        Ok(())
    }

    #[cfg(feature = "std")]
    fn read_mask_payload<R: Read>(r: &mut R) -> io::Result<(Vec<bool>, Vec<bool>)> {
        let n = storage::read_u32_le(r)? as usize;

        let reserved_len = storage::read_u32_le(r)? as usize;
        let reserved_bytes = {
            let mut buf = vec![0u8; reserved_len];
            r.read_exact(&mut buf)?;
            buf
        };

        let learning_len = storage::read_u32_le(r)? as usize;
        let learning_bytes = {
            let mut buf = vec![0u8; learning_len];
            r.read_exact(&mut buf)?;
            buf
        };

        if reserved_len != learning_len {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "MASK byte length mismatch",
            ));
        }
        let expected_len = n.div_ceil(8);
        if reserved_len != expected_len {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "MASK byte length invalid",
            ));
        }

        Ok((
            Self::unpack_bool_bits(n, &reserved_bytes),
            Self::unpack_bool_bits(n, &learning_bytes),
        ))
    }

    #[cfg(feature = "std")]
    fn unpack_bool_bits(n: usize, bytes: &[u8]) -> Vec<bool> {
        let mut out = vec![false; n];
        for i in 0..n {
            let byte = bytes[i / 8];
            let bit = (byte >> (i % 8)) & 1;
            out[i] = bit != 0;
        }
        out
    }

    // -------------------------------------------------------------------------
    // Salience chunk (SALI) - stores per-unit salience values
    // -------------------------------------------------------------------------

    #[cfg(feature = "std")]
    fn salience_payload_len_bytes(&self) -> u32 {
        // unit_count (u32) + unit_count * salience (f32)
        4 + (self.units.len() as u32) * 4
    }

    #[cfg(feature = "std")]
    fn write_salience_chunk_v2<W: Write>(&self, w: &mut W) -> io::Result<()> {
        let mut payload: Vec<u8> = Vec::with_capacity(self.salience_payload_len_bytes() as usize);
        storage::write_u32_le(&mut payload, self.units.len() as u32)?;
        for u in &self.units {
            storage::write_f32_le(&mut payload, u.salience)?;
        }
        storage::write_chunk_v2_lz4(w, *b"SALI", &payload)
    }

    #[cfg(feature = "std")]
    fn read_salience_payload<R: Read>(r: &mut R) -> io::Result<Vec<f32>> {
        let n = storage::read_u32_le(r)? as usize;
        let mut salience = Vec::with_capacity(n);
        for _ in 0..n {
            salience.push(storage::read_f32_le(r)?);
        }
        Ok(salience)
    }

    #[cfg(feature = "std")]
    fn groups_payload_len_bytes(&self) -> io::Result<u32> {
        let mut len: u64 = 0;
        len += 4; // sensor group count
        for g in &self.sensor_groups {
            len += 4 + g.name.len() as u64;
            len += 4; // unit count
            len += 4 * g.units.len() as u64;
        }

        len += 4; // action group count
        for g in &self.action_groups {
            len += 4 + g.name.len() as u64;
            len += 4;
            len += 4 * g.units.len() as u64;
        }

        u32::try_from(len)
            .map_err(|_| io::Error::new(io::ErrorKind::InvalidData, "GRPS chunk too large"))
    }

    #[cfg(feature = "std")]
    fn write_groups_chunk_v2<W: Write>(&self, w: &mut W) -> io::Result<()> {
        let mut payload: Vec<u8> = Vec::with_capacity(self.groups_payload_len_bytes()? as usize);

        storage::write_u32_le(&mut payload, self.sensor_groups.len() as u32)?;
        for g in &self.sensor_groups {
            storage::write_string(&mut payload, &g.name)?;
            storage::write_u32_le(&mut payload, g.units.len() as u32)?;
            for &u in &g.units {
                storage::write_u32_le(&mut payload, u as u32)?;
            }
        }

        storage::write_u32_le(&mut payload, self.action_groups.len() as u32)?;
        for g in &self.action_groups {
            storage::write_string(&mut payload, &g.name)?;
            storage::write_u32_le(&mut payload, g.units.len() as u32)?;
            for &u in &g.units {
                storage::write_u32_le(&mut payload, u as u32)?;
            }
        }

        storage::write_chunk_v2_lz4(w, *b"GRPS", &payload)
    }

    #[cfg(feature = "std")]
    fn latent_modules_payload_len_bytes(&self) -> io::Result<u32> {
        let mut len: u64 = 0;
        len += 4; // group count
        for g in &self.latent_groups {
            len += 4 + g.name.len() as u64;
            len += 4; // unit count
            len += 4 * g.units.len() as u64;
        }

        u32::try_from(len)
            .map_err(|_| io::Error::new(io::ErrorKind::InvalidData, "LMOD chunk too large"))
    }

    #[cfg(feature = "std")]
    fn write_latent_modules_chunk_v2<W: Write>(&self, w: &mut W) -> io::Result<()> {
        let mut payload: Vec<u8> =
            Vec::with_capacity(self.latent_modules_payload_len_bytes()? as usize);

        storage::write_u32_le(&mut payload, self.latent_groups.len() as u32)?;
        for g in &self.latent_groups {
            storage::write_string(&mut payload, &g.name)?;
            storage::write_u32_le(&mut payload, g.units.len() as u32)?;
            for &u in &g.units {
                storage::write_u32_le(&mut payload, u as u32)?;
            }
        }

        storage::write_chunk_v2_lz4(w, LATENT_MODULES_CHUNK, &payload)
    }

    #[cfg(feature = "std")]
    fn read_groups_payload<R: Read>(r: &mut R) -> io::Result<(Vec<NamedGroup>, Vec<NamedGroup>)> {
        let sg_n = storage::read_u32_le(r)? as usize;
        let mut sensor_groups: Vec<NamedGroup> = Vec::with_capacity(sg_n);
        for _ in 0..sg_n {
            let name = storage::read_string(r)?;
            let n = storage::read_u32_le(r)? as usize;
            let mut units: Vec<UnitId> = Vec::with_capacity(n);
            for _ in 0..n {
                units.push(storage::read_u32_le(r)? as usize);
            }
            sensor_groups.push(NamedGroup { name, units });
        }

        let ag_n = storage::read_u32_le(r)? as usize;
        let mut action_groups: Vec<NamedGroup> = Vec::with_capacity(ag_n);
        for _ in 0..ag_n {
            let name = storage::read_string(r)?;
            let n = storage::read_u32_le(r)? as usize;
            let mut units: Vec<UnitId> = Vec::with_capacity(n);
            for _ in 0..n {
                units.push(storage::read_u32_le(r)? as usize);
            }
            action_groups.push(NamedGroup { name, units });
        }

        Ok((sensor_groups, action_groups))
    }

    #[cfg(feature = "std")]
    fn read_latent_modules_payload<R: Read>(r: &mut R) -> io::Result<Vec<NamedGroup>> {
        let n = storage::read_u32_le(r)? as usize;
        let mut groups: Vec<NamedGroup> = Vec::with_capacity(n);
        for _ in 0..n {
            let name = storage::read_string(r)?;
            let k = storage::read_u32_le(r)? as usize;
            let mut units: Vec<UnitId> = Vec::with_capacity(k);
            for _ in 0..k {
                units.push(storage::read_u32_le(r)? as usize);
            }
            groups.push(NamedGroup { name, units });
        }
        Ok(groups)
    }

    #[cfg(feature = "std")]
    fn symbols_payload_len_bytes(&self) -> io::Result<u32> {
        let mut len: u64 = 0;
        len += 4; // count
        for s in &self.symbols_rev {
            len += 4 + s.len() as u64;
        }
        u32::try_from(len)
            .map_err(|_| io::Error::new(io::ErrorKind::InvalidData, "SYMB chunk too large"))
    }

    #[cfg(feature = "std")]
    fn write_symbols_chunk_v2<W: Write>(&self, w: &mut W) -> io::Result<()> {
        let mut payload: Vec<u8> = Vec::with_capacity(self.symbols_payload_len_bytes()? as usize);
        storage::write_u32_le(&mut payload, self.symbols_rev.len() as u32)?;
        for s in &self.symbols_rev {
            storage::write_string(&mut payload, s)?;
        }
        storage::write_chunk_v2_lz4(w, *b"SYMB", &payload)
    }

    #[cfg(feature = "std")]
    fn read_symbols_payload<R: Read>(r: &mut R) -> io::Result<Vec<String>> {
        let n = storage::read_u32_le(r)? as usize;
        let mut out: Vec<String> = Vec::with_capacity(n);
        for _ in 0..n {
            out.push(storage::read_string(r)?);
        }
        Ok(out)
    }

    #[cfg(feature = "std")]
    fn rebuild_symbol_tables(
        symbols_rev: &[String],
    ) -> io::Result<(HashMap<String, SymbolId>, SymbolId, SymbolId)> {
        let mut symbols: HashMap<String, SymbolId> = HashMap::with_capacity(symbols_rev.len());
        let mut reward_pos: Option<SymbolId> = None;
        let mut reward_neg: Option<SymbolId> = None;

        for (i, name) in symbols_rev.iter().enumerate() {
            let id = i as SymbolId;
            if symbols.insert(name.clone(), id).is_some() {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "duplicate symbol name",
                ));
            }
            if name == "reward_pos" {
                reward_pos = Some(id);
            }
            if name == "reward_neg" {
                reward_neg = Some(id);
            }
        }

        let reward_pos = reward_pos.ok_or_else(|| {
            io::Error::new(io::ErrorKind::InvalidData, "missing reward_pos symbol")
        })?;
        let reward_neg = reward_neg.ok_or_else(|| {
            io::Error::new(io::ErrorKind::InvalidData, "missing reward_neg symbol")
        })?;

        Ok((symbols, reward_pos, reward_neg))
    }

    #[cfg(feature = "std")]
    fn write_causality_chunk_v2<W: Write>(&self, w: &mut W) -> io::Result<()> {
        let mut payload: Vec<u8> =
            Vec::with_capacity(self.causal.image_payload_len_bytes() as usize);
        self.causal.write_image_payload(&mut payload)?;
        storage::write_chunk_v2_lz4(w, *b"CAUS", &payload)
    }

    /// Ensure a sensor group exists; if missing, create it.
    pub fn ensure_sensor(&mut self, name: &str, width: usize) {
        if self.sensor_groups.iter().any(|g| g.name == name) {
            return;
        }
        self.define_sensor(name, width);
    }

    /// Ensure a sensor group exists and has at least `min_width` units.
    ///
    /// Unlike [`ensure_sensor`], this will *grow* an existing group by reserving
    /// additional unreserved units and adding them to the group.
    ///
    /// Returns how many units were added.
    pub fn ensure_sensor_min_width(&mut self, name: &str, min_width: usize) -> usize {
        if min_width == 0 {
            return 0;
        }

        if let Some(idx) = self.sensor_groups.iter().position(|g| g.name == name) {
            let module = self.ensure_routing_module("sensor", name);
            let cur = self.sensor_groups[idx].units.len();
            if cur >= min_width {
                return 0;
            }
            let want = min_width - cur;
            let extra = self.allocate_units(want);
            for &id in &extra {
                self.sensor_member[id] = true;
                self.group_member[id] = true;
                if id < self.unit_module.len() {
                    self.unit_module[id] = module;
                }
            }
            self.module_unit_counts_dirty = true;
            self.sensor_groups[idx].units.extend(extra);
            self.intern(name);
            min_width
                .saturating_sub(cur)
                .min(self.sensor_groups[idx].units.len().saturating_sub(cur))
        } else {
            self.define_sensor(name, min_width);
            min_width
        }
    }

    pub fn has_sensor(&self, name: &str) -> bool {
        self.sensor_groups.iter().any(|g| g.name == name)
    }

    /// Ensure an action group exists; if missing, create it.
    pub fn ensure_action(&mut self, name: &str, width: usize) {
        if self.action_groups.iter().any(|g| g.name == name) {
            return;
        }
        self.define_action(name, width);
    }

    /// Ensure an action group exists and has at least `min_width` units.
    ///
    /// Unlike [`ensure_action`], this will *grow* an existing group by reserving
    /// additional unreserved units and adding them to the group.
    ///
    /// Returns how many units were added.
    pub fn ensure_action_min_width(&mut self, name: &str, min_width: usize) -> usize {
        if min_width == 0 {
            return 0;
        }

        if let Some(idx) = self.action_groups.iter().position(|g| g.name == name) {
            let module = self.ensure_routing_module("action", name);
            let cur = self.action_groups[idx].units.len();
            if cur >= min_width {
                return 0;
            }
            let want = min_width - cur;
            let extra = self.allocate_units(want);
            for &id in &extra {
                self.units[id].bias += 0.02;
                self.group_member[id] = true;
                if id < self.unit_module.len() {
                    self.unit_module[id] = module;
                }
            }
            self.module_unit_counts_dirty = true;
            self.action_groups[idx].units.extend(extra);
            self.intern(name);
            min_width
                .saturating_sub(cur)
                .min(self.action_groups[idx].units.len().saturating_sub(cur))
        } else {
            self.define_action(name, min_width);
            min_width
        }
    }

    pub fn has_action(&self, name: &str) -> bool {
        self.action_groups.iter().any(|g| g.name == name)
    }

    /// Return a compact sampling of units for UI visualization.
    ///
    /// Uses evenly-spaced sampling over unit IDs so the plot is stable across frames.
    /// `rel_age` is an ID-based proxy (newer units tend to have higher IDs).
    #[cfg(feature = "std")]
    pub fn unit_plot_points(&self, max_points: usize) -> Vec<UnitPlotPoint> {
        let n = self.units.len();
        if n == 0 || max_points == 0 {
            return Vec::new();
        }

        let take = max_points.min(n);
        let denom = (n - 1).max(1) as f32;

        // First pass: determine max amplitude and max salience across sampled points.
        let mut max_amp = 0.0f32;
        let mut max_salience = 0.0f32;
        for i in 0..take {
            let id = (i * n) / take;
            let a = self.units[id].amp;
            let s = self.units[id].salience;
            if a > max_amp {
                max_amp = a;
            }
            if s > max_salience {
                max_salience = s;
            }
        }
        let inv_max = if max_amp > 1e-6 { 1.0 / max_amp } else { 0.0 };
        let inv_max_salience = if max_salience > 1e-6 {
            1.0 / max_salience
        } else {
            0.0
        };

        let mut out = Vec::with_capacity(take);
        for i in 0..take {
            let id = (i * n) / take;
            let rel_age = (id as f32 / denom).clamp(0.0, 1.0);
            let amp = self.units[id].amp;
            let phase = self.units[id].phase;
            let salience = self.units[id].salience;
            out.push(UnitPlotPoint {
                id: id as u32,
                amp,
                amp01: (amp * inv_max).clamp(0.0, 1.0),
                phase,
                salience01: (salience * inv_max_salience).clamp(0.0, 1.0),
                rel_age,
                is_reserved: self.reserved.get(id).copied().unwrap_or(false),
                is_sensor_member: self.sensor_member.get(id).copied().unwrap_or(false),
                is_group_member: self.group_member.get(id).copied().unwrap_or(false),
            });
        }
        out
    }

    /// Return a lightweight, sampled "global oscillation" vector.
    ///
    /// Interprets each unit as a phasor (amp, phase) and computes the
    /// amplitude-weighted mean of (cos(phase), sin(phase)) over a sampled
    /// subset of units.
    ///
    /// Returns (x, y, mag) where mag ∈ [0, 1] for typical normalized phases.
    #[cfg(feature = "std")]
    pub fn oscillation_sample(&self, max_units: usize) -> (f32, f32, f32) {
        let n = self.units.len();
        if n == 0 || max_units == 0 {
            return (0.0, 0.0, 0.0);
        }

        let take = max_units.min(n).max(1);
        let stride = (n / take).max(1);

        let mut sum_x = 0.0f32;
        let mut sum_y = 0.0f32;
        let mut sum_w = 0.0f32;

        for i in (0..n).step_by(stride).take(take) {
            let u = &self.units[i];
            let w = u.amp.max(0.0);
            if w <= 0.0 {
                continue;
            }
            sum_x += w * u.phase.cos();
            sum_y += w * u.phase.sin();
            sum_w += w;
        }

        if sum_w > 1e-9 {
            sum_x /= sum_w;
            sum_y /= sum_w;
        } else {
            sum_x = 0.0;
            sum_y = 0.0;
        }

        let mag = (sum_x * sum_x + sum_y * sum_y).sqrt();
        (sum_x, sum_y, mag)
    }

    /// Create a sandboxed child brain.
    ///
    /// Design intent:
    /// - child inherits structure (couplings + causal memory)
    /// - child can explore with different noise/plasticity
    /// - child cannot mutate a protected identity subset (action groups by default)
    #[cfg(feature = "std")]
    pub fn spawn_child(
        &self,
        seed: u64,
        overrides: crate::supervisor::ChildConfigOverrides,
    ) -> Brain {
        let mut cfg = self.cfg;
        cfg.seed = Some(seed);
        cfg.noise_amp = overrides.noise_amp;
        cfg.noise_phase = overrides.noise_phase;
        cfg.hebb_rate = overrides.hebb_rate;
        cfg.forget_rate = overrides.forget_rate;

        let mut child = Brain::new(cfg);

        // Copy substrate state.
        child.units = self.units.clone();
        child.connections = self.connections.clone();
        child.eligibility = vec![0.0; child.connections.weights.len()];
        child.sensor_groups = self.sensor_groups.clone();
        child.action_groups = self.action_groups.clone();
        child.latent_groups = self.latent_groups.clone();
        child.reserved = self.reserved.clone();

        // Derived caches depend on groups copied above.
        child.rebuild_sensor_group_index();
        child.rebuild_routing_from_groups();

        // Copy symbol table + causal memory.
        child.symbols = self.symbols.clone();
        child.symbols_rev = self.symbols_rev.clone();
        child.reward_pos_symbol = self.reward_pos_symbol;
        child.reward_neg_symbol = self.reward_neg_symbol;
        child.causal = self.causal.clone();

        // Inherit execution tier from parent.
        child.tier = self.tier;

        // Protect parent identity subset: action-group units.
        let mut mask = vec![true; child.units.len()];
        for g in &child.action_groups {
            for &id in &g.units {
                mask[id] = false;
            }
        }
        child.learning_enabled = mask;

        // Group membership caches depend on groups copied above.
        child.rebuild_group_membership();

        child
    }

    /// Consolidate structural/casual knowledge from a child back into self.
    /// Only merges strong, non-identity couplings.
    #[cfg(feature = "std")]
    pub fn consolidate_from(
        &mut self,
        child: &Brain,
        policy: crate::supervisor::ConsolidationPolicy,
    ) {
        let thr = policy.weight_threshold;
        let rate = policy.merge_rate.clamp(0.0, 1.0);

        // Identity units are action group units.
        let mut protected = vec![false; self.units.len()];
        for g in &self.action_groups {
            for &id in &g.units {
                protected[id] = true;
            }
        }

        // Merge couplings from child into parent.
        for i in 0..self.units.len() {
            if protected[i] {
                continue;
            }

            // Iterate child's connections for unit i.
            for (c_target, c_weight) in child.neighbors(i) {
                if c_weight.abs() < thr {
                    continue;
                }
                if c_target < protected.len() && protected[c_target] {
                    continue;
                }

                // Find parent's connection to same target.
                let parent_range = self.conn_range(i);
                let mut found = false;
                for idx in parent_range.clone() {
                    if self.connections.targets[idx] == c_target {
                        // Blend weights.
                        self.connections.weights[idx] =
                            (1.0 - rate) * self.connections.weights[idx] + rate * c_weight;
                        found = true;
                        break;
                    }
                }

                // If not found, add new connection.
                if !found {
                    self.add_or_bump_csr(i, c_target, c_weight);
                }
            }
        }

        // Merge causal memory: copy any strong edges from child.
        self.causal.merge_from(&child.causal, 0.25);
    }

    /// Define a named sensor group with the specified number of units.
    ///
    /// Sensor groups receive external stimuli via [`apply_stimulus`].
    ///
    /// # Arguments
    /// * `name` - Unique name for this sensor group
    /// * `width` - Number of units in the group
    pub fn define_sensor(&mut self, name: &str, width: usize) {
        let module = self.ensure_routing_module("sensor", name);
        let units = self.allocate_units(width);
        for &id in &units {
            self.sensor_member[id] = true;
            self.group_member[id] = true;
            if id < self.unit_module.len() {
                self.unit_module[id] = module;
            }
        }
        self.module_unit_counts_dirty = true;
        self.sensor_groups.push(NamedGroup {
            name: name.to_string(),
            units,
        });

        // Keep the index map updated.
        let idx = self.sensor_groups.len() - 1;
        self.sensor_group_index.insert(name.to_string(), idx);

        self.intern(name);
    }

    /// Define a named action group with the specified number of units.
    ///
    /// Action groups are used by [`select_action`] to determine behavior.
    ///
    /// # Arguments
    /// * `name` - Unique name for this action group
    /// * `width` - Number of units in the group
    pub fn define_action(&mut self, name: &str, width: usize) {
        let module = self.ensure_routing_module("action", name);
        let units = self.allocate_units(width);
        // Slight positive bias so actions can become stable attractors.
        for &id in &units {
            self.units[id].bias += 0.02;
            self.group_member[id] = true;
            if id < self.unit_module.len() {
                self.unit_module[id] = module;
            }
        }
        self.module_unit_counts_dirty = true;
        self.action_groups.push(NamedGroup {
            name: name.to_string(),
            units,
        });

        self.intern(name);
    }

    /// Define a named latent module with the specified number of units.
    ///
    /// Latent modules are internal learning/routing partitions that are persisted
    /// in the brain image, but are not exposed as sensors or actions.
    ///
    /// If a module with the same name already exists (or collides with an existing
    /// sensor/action group), this is a no-op.
    pub fn define_module(&mut self, name: &str, width: usize) {
        if width == 0 {
            return;
        }
        if self.sensor_groups.iter().any(|g| g.name == name)
            || self.action_groups.iter().any(|g| g.name == name)
            || self.latent_groups.iter().any(|g| g.name == name)
        {
            return;
        }

        let module = self.ensure_routing_module("latent", name);
        let units = self.allocate_units(width);
        for &id in &units {
            self.group_member[id] = true;
            if id < self.unit_module.len() {
                self.unit_module[id] = module;
            }
        }
        self.module_unit_counts_dirty = true;

        self.latent_groups.push(NamedGroup {
            name: name.to_string(),
            units,
        });

        self.intern(name);
    }

    /// Apply a stimulus to the named sensor group.
    ///
    /// This injects input current into the sensor group's units and may trigger
    /// one-shot imprinting if the stimulus is novel.
    ///
    /// # Arguments
    /// * `stimulus` - The stimulus name and strength to apply
    ///
    /// # Example
    /// ```
    /// # use braine::substrate::{Brain, BrainConfig, Stimulus};
    /// # let cfg = BrainConfig::default();
    /// # let mut brain = Brain::new(cfg);
    /// brain.define_sensor("vision", 8);
    /// brain.apply_stimulus(Stimulus::new("vision", 1.0));
    /// ```
    pub fn apply_stimulus(&mut self, stimulus: Stimulus<'_>) {
        // Hot path: avoid allocations.
        // We take a raw slice to the group units to avoid cloning while still
        // being able to mutably update other brain state.
        let idx = match self.sensor_group_index.get(stimulus.name) {
            Some(&i) => i,
            None => match self
                .sensor_groups
                .iter()
                .position(|g| g.name == stimulus.name)
            {
                Some(i) => {
                    // Self-heal if constructed without a rebuild.
                    self.sensor_group_index.insert(stimulus.name.to_string(), i);
                    i
                }
                None => return,
            },
        };

        let (units_ptr, units_len) = match self.sensor_groups.get(idx) {
            Some(g) => (g.units.as_ptr(), g.units.len()),
            None => return,
        };

        // Safety: `sensor_groups` is not mutated during this call, so the pointer
        // remains valid for the duration of the function.
        let group_units: &[UnitId] = unsafe { core::slice::from_raw_parts(units_ptr, units_len) };

        for &id in group_units {
            self.pending_input[id] += stimulus.strength;
        }

        self.note_symbol(stimulus.name);

        if self.telemetry.enabled {
            if let Some(id) = self.symbol_id(stimulus.name) {
                self.telemetry.last_stimuli.push(id);
            }
        }

        // One-shot imprinting: when a stimulus is present, create a new "concept" unit
        // connected to currently active units (including the sensor group itself).
        // This is the simplest "instant learning" mechanism without training loops.
        self.imprint_if_novel(group_units, stimulus.strength);
    }

    /// Apply a stimulus for *inference only*.
    ///
    /// This injects input current into the named sensor group's units, but does **not**:
    /// - create / imprint new concept units
    /// - update symbol/telemetry tables
    ///
    /// Use this when you want a read-only "what would you do?" query without
    /// any structural updates.
    pub fn apply_stimulus_inference(&mut self, stimulus: Stimulus<'_>) {
        let idx = match self.sensor_group_index.get(stimulus.name) {
            Some(&i) => i,
            None => match self
                .sensor_groups
                .iter()
                .position(|g| g.name == stimulus.name)
            {
                Some(i) => {
                    // Self-heal if constructed without a rebuild.
                    self.sensor_group_index.insert(stimulus.name.to_string(), i);
                    i
                }
                None => return,
            },
        };

        let (units_ptr, units_len) = match self.sensor_groups.get(idx) {
            Some(g) => (g.units.as_ptr(), g.units.len()),
            None => return,
        };

        // Safety: `sensor_groups` is not mutated during this call, so the pointer
        // remains valid for the duration of the function.
        let group_units: &[UnitId] = unsafe { core::slice::from_raw_parts(units_ptr, units_len) };

        for &id in group_units {
            self.pending_input[id] += stimulus.strength;
        }
    }

    #[inline]
    fn build_compound_symbol<'a>(buf: &'a mut [u8; 256], parts: &[&str]) -> Option<&'a str> {
        let mut idx: usize = 0;
        for (i, part) in parts.iter().enumerate() {
            if i > 0 {
                let sep = b"::";
                if idx + sep.len() > buf.len() {
                    return None;
                }
                buf[idx..idx + sep.len()].copy_from_slice(sep);
                idx += sep.len();
            }

            let bytes = part.as_bytes();
            if idx + bytes.len() > buf.len() {
                return None;
            }
            buf[idx..idx + bytes.len()].copy_from_slice(bytes);
            idx += bytes.len();
        }

        // Parts come from valid UTF-8 strings; concatenation preserves UTF-8 validity.
        Some(unsafe { core::str::from_utf8_unchecked(&buf[..idx]) })
    }

    /// Record a compound symbol without heap allocation in the hot path.
    ///
    /// Example: `note_compound_symbol(&["pair", ctx, action])`.
    pub fn note_compound_symbol(&mut self, parts: &[&str]) {
        let mut buf = [0u8; 256];
        if let Some(name) = Self::build_compound_symbol(&mut buf, parts) {
            self.note_symbol(name);
        }
    }

    #[inline]
    fn compound_symbol_id(&self, parts: &[&str]) -> Option<SymbolId> {
        let mut buf = [0u8; 256];
        let name = Self::build_compound_symbol(&mut buf, parts)?;
        self.symbol_id(name)
    }

    /// Allocation-free action selection that returns the **action group index**.
    ///
    /// This avoids returning `&str` that would keep borrowing `self` across subsequent
    /// mutation steps (important for tight control loops like `brained`).
    pub fn select_action_with_meaning_index(&self, stimulus: &str, alpha: f32) -> (usize, f32) {
        let alpha = alpha.clamp(0.0, 20.0);
        let stimulus_id = self.symbol_id(stimulus);

        let mut best: Option<(usize, f32)> = None;
        for (idx, g) in self.action_groups.iter().enumerate() {
            let action_name: &str = g.name.as_str();

            // Habit readout: treat negative amplitude as "inactive" and normalize to ~[0,1].
            let habit = g
                .units
                .iter()
                .map(|&id| self.units[id].amp.max(0.0))
                .sum::<f32>();
            let habit_norm = if g.units.is_empty() {
                0.0
            } else {
                (habit / (g.units.len() as f32 * 2.0)).clamp(0.0, 1.0)
            };

            let meaning = if let Some(aid) = self.symbol_id(action_name) {
                let global = self.causal.causal_strength(aid, self.reward_pos_symbol)
                    - self.causal.causal_strength(aid, self.reward_neg_symbol);

                let conditional = if stimulus_id.is_some() {
                    if let Some(pid) = self.compound_symbol_id(&["pair", stimulus, action_name]) {
                        self.causal.causal_strength(pid, self.reward_pos_symbol)
                            - self.causal.causal_strength(pid, self.reward_neg_symbol)
                    } else {
                        0.0
                    }
                } else {
                    0.0
                };

                conditional * 1.0 + global * 0.15
            } else {
                0.0
            };

            let score = habit_norm * 0.5 + alpha * meaning;
            if best.as_ref().map(|b| score > b.1).unwrap_or(true) {
                best = Some((idx, score));
            }
        }

        best.unwrap_or((usize::MAX, 0.0))
    }

    /// Compute the per-action breakdown used by `select_action_with_meaning_index`.
    ///
    /// This is useful for UI: it shows how much of the preference comes from
    /// (a) habit readout and (b) causal/meaning memory.
    #[cfg(feature = "std")]
    pub fn action_score_breakdown(&self, stimulus: &str, alpha: f32) -> Vec<ActionScoreBreakdown> {
        let alpha = alpha.clamp(0.0, 20.0);
        let stimulus_id = self.symbol_id(stimulus);

        let mut out = Vec::with_capacity(self.action_groups.len());
        for g in &self.action_groups {
            let action_name: &str = g.name.as_str();

            let habit = g
                .units
                .iter()
                .map(|&id| self.units[id].amp.max(0.0))
                .sum::<f32>();
            let habit_norm = if g.units.is_empty() {
                0.0
            } else {
                (habit / (g.units.len() as f32 * 2.0)).clamp(0.0, 1.0)
            };

            let (meaning_global, meaning_conditional, meaning) = if let Some(aid) =
                self.symbol_id(action_name)
            {
                let global = self.causal.causal_strength(aid, self.reward_pos_symbol)
                    - self.causal.causal_strength(aid, self.reward_neg_symbol);

                let conditional = if stimulus_id.is_some() {
                    if let Some(pid) = self.compound_symbol_id(&["pair", stimulus, action_name]) {
                        self.causal.causal_strength(pid, self.reward_pos_symbol)
                            - self.causal.causal_strength(pid, self.reward_neg_symbol)
                    } else {
                        0.0
                    }
                } else {
                    0.0
                };

                let m = conditional * 1.0 + global * 0.15;
                (global, conditional, m)
            } else {
                (0.0, 0.0, 0.0)
            };

            let score = habit_norm * 0.5 + alpha * meaning;
            out.push(ActionScoreBreakdown {
                name: action_name.to_string(),
                habit_norm,
                meaning_global,
                meaning_conditional,
                meaning,
                score,
            });
        }

        out
    }

    /// Return causal edge strengths from `pair::<stimulus>::<action>` to `reward_pos/reward_neg`.
    ///
    /// This is allocation-free and intended for UI/debugging.
    #[cfg(feature = "std")]
    pub fn pair_reward_edges(&self, stimulus: &str, action: &str) -> RewardEdges {
        let Some(pid) = self.compound_symbol_id(&["pair", stimulus, action]) else {
            return RewardEdges::default();
        };

        let pos = self.causal.causal_strength(pid, self.reward_pos_symbol);
        let neg = self.causal.causal_strength(pid, self.reward_neg_symbol);
        RewardEdges {
            to_reward_pos: pos,
            to_reward_neg: neg,
            meaning: pos - neg,
        }
    }

    /// Return causal edge strengths from an action symbol to `reward_pos/reward_neg`.
    ///
    /// This is allocation-free and intended for UI/debugging.
    #[cfg(feature = "std")]
    pub fn action_reward_edges(&self, action: &str) -> RewardEdges {
        let Some(aid) = self.symbol_id(action) else {
            return RewardEdges::default();
        };

        let pos = self.causal.causal_strength(aid, self.reward_pos_symbol);
        let neg = self.causal.causal_strength(aid, self.reward_neg_symbol);
        RewardEdges {
            to_reward_pos: pos,
            to_reward_neg: neg,
            meaning: pos - neg,
        }
    }

    /// Borrow the action name for an action group index.
    #[must_use]
    pub fn action_name(&self, index: usize) -> Option<&str> {
        self.action_groups.get(index).map(|g| g.name.as_str())
    }

    /// Record an action event by action-group index (no heap allocation).
    pub fn note_action_index(&mut self, index: usize) {
        // Avoid holding an immutable borrow of self across a mutable call.
        let (ptr, len) = match self.action_groups.get(index) {
            Some(g) => {
                let s = g.name.as_str();
                (s.as_ptr(), s.len())
            }
            None => return,
        };

        let name = unsafe { core::str::from_utf8_unchecked(core::slice::from_raw_parts(ptr, len)) };
        self.note_action(name);
    }

    /// Record a pair::<stimulus>::<action> event by action-group index (no heap allocation).
    pub fn note_pair_index(&mut self, stimulus: &str, index: usize) {
        // Avoid holding an immutable borrow of self across a mutable call.
        let (ptr, len) = match self.action_groups.get(index) {
            Some(g) => {
                let s = g.name.as_str();
                (s.as_ptr(), s.len())
            }
            None => return,
        };

        let action =
            unsafe { core::str::from_utf8_unchecked(core::slice::from_raw_parts(ptr, len)) };
        self.note_compound_symbol(&["pair", stimulus, action]);
    }

    /// Reinforce an action by index (avoids string lookup + allocation).
    pub fn reinforce_action_index(&mut self, index: usize, delta_bias: f32) {
        let Some(group) = self.action_groups.get(index) else {
            return;
        };

        if self.telemetry.enabled {
            if let Some(id) = self.symbol_id(group.name.as_str()) {
                self.telemetry
                    .last_reinforced_actions
                    .push((id, delta_bias));
            }
        }

        for &id in &group.units {
            self.units[id].bias = (self.units[id].bias + delta_bias * 0.01).clamp(-0.5, 0.5);
        }
    }

    /// Record the selected action as an event for causality/meaning.
    pub fn note_action(&mut self, action: &str) {
        self.note_symbol(action);

        if self.telemetry.enabled {
            if let Some(id) = self.symbol_id(action) {
                self.telemetry.last_actions.push(id);
            }
        }
    }

    /// Commit current perception/action/reward events into causal memory.
    /// Call this once per loop after:
    /// - apply_stimulus
    /// - step
    /// - select_action + note_action
    /// - (optional) reinforce_action
    pub fn commit_observation(&mut self) {
        // Map reward scalar to discrete events.
        if self.neuromod > 0.2 {
            self.active_symbols.push(self.reward_pos_symbol);
        } else if self.neuromod < -0.2 {
            self.active_symbols.push(self.reward_neg_symbol);
        }

        // Deduplicate cheaply (small vectors).
        self.active_symbols.sort_unstable();
        self.active_symbols.dedup();

        // Keep bounded work for causal updates.
        let cap = self.cfg.causal_symbol_cap as usize;
        if self.active_symbols.len() > cap {
            self.active_symbols.truncate(cap);
        }

        if self.telemetry.enabled {
            self.telemetry.last_committed_symbols.clear();
            self.telemetry
                .last_committed_symbols
                .extend_from_slice(&self.active_symbols);
        }

        // Phase 2 (routing-as-attention): decide which modules are eligible to learn
        // on the *next* step, based on this committed boundary symbol set.
        if self.cfg.module_routing_top_k > 0 {
            // Clone small vector to avoid borrowing `self` immutably across `&mut self` calls.
            let symbols = self.active_symbols.clone();

            let mut routed = self.route_modules_from_symbols(&symbols);

            // Phase 5 (latent module auto-formation): if routing is enabled but
            // uninformative, optionally create a fresh latent module to capture
            // a novel committed symbol set.
            if routed.is_empty() {
                if let Some(mid) = self.maybe_auto_create_latent_module(&symbols) {
                    routed = vec![mid];
                }
            }

            // Record last routed step for retirement criteria.
            for &mid in &routed {
                if let Some(m) = self.routing_modules.get_mut(mid as usize) {
                    m.last_routed_step = self.age_steps;
                }
            }

            self.learning_route_modules = routed;

            // Update signatures for routed modules to make routing self-organizing.
            // If routing is currently uninformative (empty selection), skip the update.
            if !self.learning_route_modules.is_empty() {
                let reward = self.neuromod.clamp(-1.0, 1.0);
                // Clone small vectors to avoid borrow conflicts with `&mut self`.
                let routed = self.learning_route_modules.clone();
                self.update_routing_signatures(&routed, &symbols, reward);
            }
        } else {
            self.learning_route_modules.clear();
        }

        // Phase 5 (latent module retirement): clean up stale, low-utility latent modules.
        self.maybe_retire_latent_modules();

        // Lagged causal update:
        // - lag 1 is stored inside `CausalMemory.prev_symbols`
        // - we keep lag>=2 history in `self.causal_lag_history`
        let lag_steps = self.cfg.causal_lag_steps.clamp(1, 32) as usize;
        let max_hist = lag_steps.saturating_sub(2);
        if max_hist == 0 {
            self.causal_lag_history.clear();
        } else if self.causal_lag_history.len() > max_hist {
            self.causal_lag_history.truncate(max_hist);
        }

        let prev_lag1: Vec<SymbolId> = self.causal.prev_symbols().to_vec();
        self.causal.observe_lagged(
            &self.active_symbols,
            &self.causal_lag_history,
            self.cfg.causal_lag_decay,
        );

        // Shift history: previous lag1 becomes lag2 for the next tick.
        if max_hist > 0 && !prev_lag1.is_empty() {
            self.causal_lag_history.insert(0, prev_lag1);
            if self.causal_lag_history.len() > max_hist {
                self.causal_lag_history.truncate(max_hist);
            }
        }
        self.active_symbols.clear();
    }

    /// Discard current perception/action/reward events without learning.
    ///
    /// This is useful for evaluation/holdout modes where you want to run the
    /// substrate dynamics and action selection, but you do not want to update
    /// causal/meaning memory.
    pub fn discard_observation(&mut self) {
        // Keep telemetry roughly consistent with commit_observation().
        self.active_symbols.sort_unstable();
        self.active_symbols.dedup();

        if self.telemetry.enabled {
            self.telemetry.last_committed_symbols.clear();
            self.telemetry
                .last_committed_symbols
                .extend_from_slice(&self.active_symbols);
        }

        // Discard should not steer routing state.
        self.learning_route_modules.clear();

        self.active_symbols.clear();
    }

    /// Very small "meaning" query: which action is most causally linked to positive reward
    /// under the last seen stimulus symbol.
    pub fn meaning_hint(&self, stimulus: &str) -> Option<(String, f32)> {
        let s = self.symbol_id(stimulus)?;

        let mut best: Option<(String, f32)> = None;
        for g in &self.action_groups {
            let a = self.symbol_id(&g.name)?;
            let score = self.causal.causal_strength(a, self.reward_pos_symbol)
                - self.causal.causal_strength(a, self.reward_neg_symbol);
            if best.as_ref().map(|b| score > b.1).unwrap_or(true) {
                best = Some((g.name.clone(), score));
            }
        }

        // Also ensure stimulus is at least somewhat connected to the suggested action.
        best.and_then(|(act, sc)| {
            let a = self.symbol_id(&act)?;
            let link = self.causal.causal_strength(s, a);
            Some((act, sc * 0.7 + link * 0.3))
        })
    }

    /// Select an action using both:
    /// - current dynamical readout (habit/attractor)
    /// - learned meaning/causality (goal-directed)
    ///
    /// `alpha` weights meaning vs habit. `alpha=0` => pure habit.
    pub fn select_action_with_meaning(&mut self, stimulus: &str, alpha: f32) -> (String, f32) {
        let (idx, sc) = self.select_action_with_meaning_index(stimulus, alpha);
        let act = self.action_name(idx).unwrap_or("idle");
        (act.to_string(), sc)
    }

    /// Return actions ranked by the same score used by `select_action_with_meaning`.
    ///
    /// Useful for visualization/debugging (e.g. showing top-N candidates in a HUD).
    pub fn ranked_actions_with_meaning(&self, stimulus: &str, alpha: f32) -> Vec<(String, f32)> {
        let alpha = alpha.clamp(0.0, 20.0);
        let stimulus_id = self.symbol_id(stimulus);

        let mut scored: Vec<(String, f32)> = Vec::with_capacity(self.action_groups.len());
        for g in &self.action_groups {
            let habit = g
                .units
                .iter()
                .map(|&id| self.units[id].amp.max(0.0))
                .sum::<f32>();
            let habit_norm = if g.units.is_empty() {
                0.0
            } else {
                (habit / (g.units.len() as f32 * 2.0)).clamp(0.0, 1.0)
            };

            let meaning = if let Some(aid) = self.symbol_id(&g.name) {
                let global = self.causal.causal_strength(aid, self.reward_pos_symbol)
                    - self.causal.causal_strength(aid, self.reward_neg_symbol);

                let conditional = if stimulus_id.is_some() {
                    if let Some(pid) = self.compound_symbol_id(&["pair", stimulus, g.name.as_str()])
                    {
                        self.causal.causal_strength(pid, self.reward_pos_symbol)
                            - self.causal.causal_strength(pid, self.reward_neg_symbol)
                    } else {
                        0.0
                    }
                } else {
                    0.0
                };

                conditional * 1.0 + global * 0.15
            } else {
                0.0
            };

            let score = habit_norm * 0.5 + alpha * meaning;
            scored.push((g.name.clone(), score));
        }

        scored.sort_by(|a, b| b.1.total_cmp(&a.1));
        scored
    }

    pub fn top_actions_with_meaning(
        &self,
        stimulus: &str,
        alpha: f32,
        top_n: usize,
    ) -> Vec<(String, f32)> {
        let mut v = self.ranked_actions_with_meaning(stimulus, alpha);
        v.truncate(top_n);
        v
    }

    /// Explainability helper: strongest outgoing causal links from a named symbol.
    pub fn top_causal_links_from(&self, from: &str, top_n: usize) -> Vec<(String, f32)> {
        let Some(a) = self.symbol_id(from) else {
            return Vec::new();
        };

        self.causal
            .top_outgoing(a, top_n)
            .into_iter()
            .filter_map(|(bid, s)| self.symbol_name(bid).map(|name| (name.to_string(), s)))
            .collect()
    }

    /// Predict the most likely next context symbols given `(stimulus, action)`.
    ///
    /// Uses the `pair::<stimulus>::<action>` symbol's outgoing causal edges to context symbols.
    /// Returns a ranked list of `(context_name, causal_strength)`.
    ///
    /// `ctx_prefix` filters which symbols count as contexts (e.g., `"ctx_"` or `"bucket_"`).
    pub fn predict_next_context(
        &self,
        stimulus: &str,
        action: &str,
        ctx_prefix: &str,
        top_n: usize,
    ) -> Vec<(String, f32)> {
        let Some(pair_id) = self.compound_symbol_id(&["pair", stimulus, action]) else {
            return Vec::new();
        };

        // Gather all known symbols that start with ctx_prefix
        let ctx_ids: Vec<SymbolId> = self
            .symbols
            .iter()
            .filter(|(name, _)| name.starts_with(ctx_prefix))
            .map(|(_, &id)| id)
            .collect();

        self.causal
            .top_outgoing_filtered(pair_id, &ctx_ids, top_n)
            .into_iter()
            .filter_map(|(bid, s)| self.symbol_name(bid).map(|name| (name.to_string(), s)))
            .collect()
    }

    /// Compute a "prediction bonus" for an action: expected value of the predicted next context.
    ///
    /// This can be used as a tie-breaker in action selection.
    /// Returns `Sum_i(P(ctx_i | pair) * value(ctx_i))` where value is reward association.
    fn prediction_bonus(&self, stimulus: &str, action: &str, ctx_prefix: &str) -> f32 {
        let Some(pair_id) = self.compound_symbol_id(&["pair", stimulus, action]) else {
            return 0.0;
        };

        // Gather context symbol IDs
        let ctx_ids: Vec<SymbolId> = self
            .symbols
            .iter()
            .filter(|(name, _)| name.starts_with(ctx_prefix))
            .map(|(_, &id)| id)
            .collect();

        let mut bonus = 0.0f32;
        for &ctx_id in &ctx_ids {
            // How strongly does (stimulus, action) predict this ctx?
            let pred_strength = self.causal.causal_strength(pair_id, ctx_id);
            if pred_strength.abs() < 0.001 {
                continue;
            }
            // What's the value of this ctx? (association with reward)
            let ctx_value = self.causal.causal_strength(ctx_id, self.reward_pos_symbol)
                - self.causal.causal_strength(ctx_id, self.reward_neg_symbol);
            bonus += pred_strength * ctx_value;
        }
        bonus
    }

    /// Select an action using meaning + prediction bonus.
    ///
    /// `pred_weight` controls how much the prediction bonus matters (0 = disabled).
    /// `ctx_prefix` specifies which symbols are contexts for prediction (e.g., `"bucket_"`).
    pub fn select_action_predictive(
        &mut self,
        stimulus: &str,
        alpha: f32,
        pred_weight: f32,
        ctx_prefix: &str,
    ) -> (String, f32, f32) {
        let alpha = alpha.clamp(0.0, 20.0);
        let pred_weight = pred_weight.clamp(0.0, 10.0);
        let stimulus_id = self.symbol_id(stimulus);

        let mut best: Option<(String, f32, f32)> = None; // (action, total_score, pred_bonus)
        for g in &self.action_groups {
            // Habit readout
            let habit = g
                .units
                .iter()
                .map(|&id| self.units[id].amp.max(0.0))
                .sum::<f32>();
            let habit_norm = if g.units.is_empty() {
                0.0
            } else {
                (habit / (g.units.len() as f32 * 2.0)).clamp(0.0, 1.0)
            };

            // Meaning (same as select_action_with_meaning)
            let meaning = if let Some(aid) = self.symbol_id(&g.name) {
                let global = self.causal.causal_strength(aid, self.reward_pos_symbol)
                    - self.causal.causal_strength(aid, self.reward_neg_symbol);
                let conditional = if stimulus_id.is_some() {
                    if let Some(pid) = self.compound_symbol_id(&["pair", stimulus, g.name.as_str()])
                    {
                        self.causal.causal_strength(pid, self.reward_pos_symbol)
                            - self.causal.causal_strength(pid, self.reward_neg_symbol)
                    } else {
                        0.0
                    }
                } else {
                    0.0
                };
                conditional * 1.0 + global * 0.15
            } else {
                0.0
            };

            // Prediction bonus
            let pred_bonus = self.prediction_bonus(stimulus, &g.name, ctx_prefix);

            let score = habit_norm * 0.5 + alpha * meaning + pred_weight * pred_bonus;
            if best.as_ref().map(|b| score > b.1).unwrap_or(true) {
                best = Some((g.name.clone(), score, pred_bonus));
            }
        }

        best.unwrap_or_else(|| ("idle".to_string(), 0.0, 0.0))
    }

    /// Set the neuromodulator (reward/salience) level.
    ///
    /// Positive values increase learning rate; negative values decrease it.
    /// The value is clamped to [-1.0, 1.0].
    ///
    /// # Arguments
    /// * `value` - The neuromodulator level
    pub fn set_neuromodulator(&mut self, value: f32) {
        // Clamp to a reasonable range.
        self.neuromod = value.clamp(-1.0, 1.0);
    }

    /// Reinforce an action by adjusting the bias of its units.
    ///
    /// This provides a direct reward signal to encourage/discourage actions.
    ///
    /// # Arguments
    /// * `action` - Name of the action group to reinforce
    /// * `delta_bias` - Bias adjustment (positive = encourage, negative = discourage)
    pub fn reinforce_action(&mut self, action: &str, delta_bias: f32) {
        if let Some(group) = self.action_groups.iter().find(|g| g.name == action) {
            if self.telemetry.enabled {
                if let Some(id) = self.symbol_id(action) {
                    self.telemetry
                        .last_reinforced_actions
                        .push((id, delta_bias));
                }
            }
            for &id in &group.units {
                self.units[id].bias = (self.units[id].bias + delta_bias * 0.01).clamp(-0.5, 0.5);
            }
        }
    }

    /// Advance the simulation by one timestep.
    ///
    /// This is the main update loop that:
    /// 1. Updates unit dynamics (amp/phase) based on connections and inputs
    /// 2. Applies global inhibition for competition
    /// 3. Runs Hebbian learning on co-active, phase-aligned units
    /// 4. Prunes weak connections (structural forgetting)
    ///
    /// Call this once per control cycle after applying stimuli and setting
    /// the neuromodulator.
    pub fn step(&mut self) {
        self.pruned_last_step = 0;

        // Reset per-step monitors.
        self.learning_monitors = LearningMonitors::default();

        self.age_steps = self.age_steps.wrapping_add(1);
        if self.telemetry.enabled {
            self.telemetry.last_stimuli.clear();
            self.telemetry.last_actions.clear();
            self.telemetry.last_reinforced_actions.clear();
        }

        // Dispatch based on the effective execution tier.
        // This honors compile-time feature gates and runtime GPU availability.
        match self.effective_execution_tier() {
            ExecutionTier::Scalar => self.step_dynamics_scalar(),
            ExecutionTier::Simd => self.step_dynamics_simd(),
            ExecutionTier::Parallel => self.step_dynamics_parallel(),
            ExecutionTier::Gpu => self.step_dynamics_gpu(),
        }

        // Clear one-tick inputs.
        for x in &mut self.pending_input {
            *x = 0.0;
        }

        // Eligibility traces always update (local and cheap).
        self.update_eligibility_scalar();

        // Plasticity is committed only when neuromodulation is present.
        self.apply_plasticity_scalar();

        self.forget_and_prune();

        self.update_growth_signals();

        self.homeostasis_step();
    }

    fn update_growth_signals(&mut self) {
        let alpha = self.cfg.growth_signal_alpha;
        if alpha <= 0.0 {
            return;
        }

        let edge_n = self.connections.weights.len().max(1) as f32;
        let eligibility_norm = (self.learning_monitors.eligibility_l1 / edge_n).clamp(0.0, 10.0);
        let commit = if self.learning_monitors.plasticity_committed {
            1.0
        } else {
            0.0
        };
        let prune_norm = (self.pruned_last_step as f32 / edge_n).clamp(0.0, 1.0);

        self.growth_eligibility_norm_ema =
            (1.0 - alpha) * self.growth_eligibility_norm_ema + alpha * eligibility_norm;
        self.growth_commit_ema = (1.0 - alpha) * self.growth_commit_ema + alpha * commit;
        self.growth_prune_norm_ema =
            (1.0 - alpha) * self.growth_prune_norm_ema + alpha * prune_norm;
    }

    /// Advance the simulation by one timestep, **without learning**.
    ///
    /// This updates unit dynamics and clears one-tick inputs, but does not run
    /// Hebbian learning or forgetting/pruning.
    pub fn step_inference(&mut self) {
        self.pruned_last_step = 0;
        self.age_steps = self.age_steps.wrapping_add(1);

        if self.telemetry.enabled {
            self.telemetry.last_stimuli.clear();
            self.telemetry.last_actions.clear();
            self.telemetry.last_reinforced_actions.clear();
        }

        match self.effective_execution_tier() {
            ExecutionTier::Scalar => self.step_dynamics_scalar(),
            ExecutionTier::Simd => self.step_dynamics_simd(),
            ExecutionTier::Parallel => self.step_dynamics_parallel(),
            ExecutionTier::Gpu => self.step_dynamics_gpu(),
        }

        for x in &mut self.pending_input {
            *x = 0.0;
        }
    }

    /// Compute global inhibition signal based on inhibition_mode.
    fn compute_inhibition(&self) -> f32 {
        let avg = match self.cfg.inhibition_mode {
            0 => {
                // Signed mean (legacy)
                self.units.iter().map(|u| u.amp).sum::<f32>() / self.units.len() as f32
            }
            1 => {
                // Mean absolute amplitude
                self.units.iter().map(|u| u.amp.abs()).sum::<f32>() / self.units.len() as f32
            }
            2 => {
                // Rectified mean (max(0, a))
                self.units.iter().map(|u| u.amp.max(0.0)).sum::<f32>() / self.units.len() as f32
            }
            _ => {
                // Default to signed mean
                self.units.iter().map(|u| u.amp).sum::<f32>() / self.units.len() as f32
            }
        };
        self.cfg.global_inhibition * avg
    }

    /// Scalar (baseline) dynamics update.
    fn step_dynamics_scalar(&mut self) {
        let inhibition = self.compute_inhibition();

        let mut next_amp = vec![0.0; self.units.len()];
        let mut next_phase = vec![0.0; self.units.len()];

        for i in 0..self.units.len() {
            let u = &self.units[i];
            let mut influence_amp = 0.0;
            let mut influence_phase = 0.0;

            for (target, weight) in self.neighbors(i) {
                let v = &self.units[target];
                influence_amp += weight * v.amp;
                influence_phase += weight
                    * self.cfg.phase_coupling_gain
                    * phase_coupling_term(angle_diff(v.phase, u.phase), &self.cfg);
            }

            let noise_a = self
                .rng
                .gen_range_f32(-self.cfg.noise_amp, self.cfg.noise_amp);
            let noise_p = self
                .rng
                .gen_range_f32(-self.cfg.noise_phase, self.cfg.noise_phase);

            let input = self.pending_input[i];
            let damp = u.decay * u.amp;
            let satur = -self.cfg.amp_saturation_beta * u.amp.powi(3);
            let d_amp = (u.bias + input + influence_amp - inhibition - damp + satur + noise_a)
                * self.cfg.dt;
            let d_phase = (self.cfg.base_freq + influence_phase + noise_p) * self.cfg.dt;

            next_amp[i] = (u.amp + d_amp).clamp(-2.0, 2.0);
            next_phase[i] = wrap_angle(u.phase + d_phase);
        }

        // Update units with new amp/phase and update salience.
        // Salience formula: s = (1 - λ) * s + α * max(0, amp - threshold)
        let salience_decay = self.cfg.salience_decay;
        let salience_gain = self.cfg.salience_gain;
        let salience_threshold = self.cfg.coactive_threshold;
        let trace_decay = self.cfg.activity_trace_decay;

        if self.activity_trace.len() != self.units.len() {
            self.activity_trace.resize(self.units.len(), 0.0);
        }

        for i in 0..self.units.len() {
            self.units[i].amp = next_amp[i];
            self.units[i].phase = next_phase[i];

            // Update slow activity trace (derived; not a learned weight).
            let act = next_amp[i].max(0.0);
            let tr = &mut self.activity_trace[i];
            *tr = if trace_decay <= 0.0 {
                act
            } else {
                (1.0 - trace_decay) * (*tr) + trace_decay * act
            };

            // Update salience: decay + gain when active
            let activation = (*tr - salience_threshold).max(0.0);
            self.units[i].salience =
                (1.0 - salience_decay) * self.units[i].salience + salience_gain * activation;
            // Clamp salience to reasonable range
            self.units[i].salience = self.units[i].salience.clamp(0.0, 10.0);
        }
    }

    /// SIMD-optimized dynamics using the wide crate.
    ///
    /// Vectorizes the amplitude/phase update loop while keeping the sparse
    /// neighbor accumulation scalar (irregular memory access patterns).
    #[cfg(feature = "simd")]
    fn step_dynamics_simd(&mut self) {
        let inhibition = self.compute_inhibition();
        let n = self.units.len();

        // Accumulate influences (sparse, hard to vectorize efficiently).
        let mut influence_amp = vec![0.0f32; n];
        let mut influence_phase = vec![0.0f32; n];

        for i in 0..n {
            let u_phase = self.units[i].phase;
            for (target, weight) in self.neighbors(i) {
                let v = &self.units[target];
                influence_amp[i] += weight * v.amp;
                influence_phase[i] += weight
                    * self.cfg.phase_coupling_gain
                    * phase_coupling_term(angle_diff(v.phase, u_phase), &self.cfg);
            }
        }

        // Pre-generate noise.
        let noise_a: Vec<f32> = (0..n)
            .map(|_| {
                self.rng
                    .gen_range_f32(-self.cfg.noise_amp, self.cfg.noise_amp)
            })
            .collect();
        let noise_p: Vec<f32> = (0..n)
            .map(|_| {
                self.rng
                    .gen_range_f32(-self.cfg.noise_phase, self.cfg.noise_phase)
            })
            .collect();

        // Extract SoA views for vectorization.
        let mut amps: Vec<f32> = self.units.iter().map(|u| u.amp).collect();
        let mut phases: Vec<f32> = self.units.iter().map(|u| u.phase).collect();
        let biases: Vec<f32> = self.units.iter().map(|u| u.bias).collect();
        let decays: Vec<f32> = self.units.iter().map(|u| u.decay).collect();

        let dt = f32x4::splat(self.cfg.dt);
        let base_freq = f32x4::splat(self.cfg.base_freq);
        let inhibition_v = f32x4::splat(inhibition);
        let beta_v = f32x4::splat(self.cfg.amp_saturation_beta);
        let min_clamp = f32x4::splat(-2.0);
        let max_clamp = f32x4::splat(2.0);
        let pi = f32x4::splat(std::f32::consts::PI);
        let two_pi = f32x4::splat(std::f32::consts::TAU);

        // Process 4 units at a time.
        let simd_end = n - (n % 4);
        for i in (0..simd_end).step_by(4) {
            let amp = f32x4::from([amps[i], amps[i + 1], amps[i + 2], amps[i + 3]]);
            let phase = f32x4::from([phases[i], phases[i + 1], phases[i + 2], phases[i + 3]]);
            let bias = f32x4::from([biases[i], biases[i + 1], biases[i + 2], biases[i + 3]]);
            let decay = f32x4::from([decays[i], decays[i + 1], decays[i + 2], decays[i + 3]]);
            let inf_a = f32x4::from([
                influence_amp[i],
                influence_amp[i + 1],
                influence_amp[i + 2],
                influence_amp[i + 3],
            ]);
            let inf_p = f32x4::from([
                influence_phase[i],
                influence_phase[i + 1],
                influence_phase[i + 2],
                influence_phase[i + 3],
            ]);
            let n_a = f32x4::from([noise_a[i], noise_a[i + 1], noise_a[i + 2], noise_a[i + 3]]);
            let n_p = f32x4::from([noise_p[i], noise_p[i + 1], noise_p[i + 2], noise_p[i + 3]]);
            let input = f32x4::from([
                self.pending_input[i],
                self.pending_input[i + 1],
                self.pending_input[i + 2],
                self.pending_input[i + 3],
            ]);

            let damp = decay * amp;
            let satur = -beta_v * amp * amp * amp;
            let d_amp = (bias + input + inf_a - inhibition_v - damp + satur + n_a) * dt;
            let d_phase = (base_freq + inf_p + n_p) * dt;

            let new_amp = (amp + d_amp).max(min_clamp).min(max_clamp);
            let mut new_phase = phase + d_phase;

            // wrap_angle for SIMD: normalize to [-PI, PI]
            // while new_phase > PI: new_phase -= TAU
            // while new_phase < -PI: new_phase += TAU
            // Approximation: one iteration usually sufficient for small dt
            let too_high = new_phase.cmp_gt(pi);
            let too_low = new_phase.cmp_lt(-pi);
            new_phase = too_high.blend(new_phase - two_pi, new_phase);
            new_phase = too_low.blend(new_phase + two_pi, new_phase);

            let new_amp_arr = new_amp.to_array();
            let new_phase_arr = new_phase.to_array();
            amps[i..(i + 4)].copy_from_slice(&new_amp_arr);
            phases[i..(i + 4)].copy_from_slice(&new_phase_arr);
        }

        // Handle remainder (tail elements).
        for i in simd_end..n {
            let u = &self.units[i];
            let damp = u.decay * amps[i];
            let d_amp = (u.bias + self.pending_input[i] + influence_amp[i] - inhibition - damp
                + noise_a[i])
                * self.cfg.dt;
            let d_phase = (self.cfg.base_freq + influence_phase[i] + noise_p[i]) * self.cfg.dt;
            amps[i] = (amps[i] + d_amp).clamp(-2.0, 2.0);
            phases[i] = wrap_angle(phases[i] + d_phase);
        }

        // Write back to units and update salience.
        let salience_decay = self.cfg.salience_decay;
        let salience_gain = self.cfg.salience_gain;
        let salience_threshold = self.cfg.coactive_threshold;
        let trace_decay = self.cfg.activity_trace_decay;

        if self.activity_trace.len() != n {
            self.activity_trace.resize(n, 0.0);
        }

        for i in 0..n {
            self.units[i].amp = amps[i];
            self.units[i].phase = phases[i];

            let act = amps[i].max(0.0);
            let tr = &mut self.activity_trace[i];
            *tr = if trace_decay <= 0.0 {
                act
            } else {
                (1.0 - trace_decay) * (*tr) + trace_decay * act
            };

            // Update salience: decay + gain when active
            let activation = (*tr - salience_threshold).max(0.0);
            self.units[i].salience =
                (1.0 - salience_decay) * self.units[i].salience + salience_gain * activation;
            self.units[i].salience = self.units[i].salience.clamp(0.0, 10.0);
        }
    }

    #[cfg(not(feature = "simd"))]
    fn step_dynamics_simd(&mut self) {
        self.step_dynamics_scalar();
    }

    /// Parallel dynamics update using rayon.
    #[cfg(feature = "parallel")]
    fn step_dynamics_parallel(&mut self) {
        let inhibition = self.compute_inhibition();

        // Pre-generate noise (RNG is not thread-safe).
        let noise: Vec<(f32, f32)> = (0..self.units.len())
            .map(|_| {
                (
                    self.rng
                        .gen_range_f32(-self.cfg.noise_amp, self.cfg.noise_amp),
                    self.rng
                        .gen_range_f32(-self.cfg.noise_phase, self.cfg.noise_phase),
                )
            })
            .collect();

        // Parallel computation of next state.
        let units = &self.units;
        let connections = &self.connections;
        let pending_input = &self.pending_input;
        let cfg = &self.cfg;

        let next: Vec<(f32, f32)> = (0..units.len())
            .into_par_iter()
            .map(|i| {
                let u = &units[i];
                let mut influence_amp = 0.0;
                let mut influence_phase = 0.0;

                let start = connections.offsets[i];
                let end = connections.offsets[i + 1];
                for idx in start..end {
                    let target = connections.targets[idx];
                    if target == INVALID_UNIT {
                        continue;
                    }
                    let weight = connections.weights[idx];
                    let v = &units[target];
                    influence_amp += weight * v.amp;
                    influence_phase += weight
                        * cfg.phase_coupling_gain
                        * phase_coupling_term(angle_diff(v.phase, u.phase), cfg);
                }

                let (noise_a, noise_p) = noise[i];
                let input = pending_input[i];
                let damp = u.decay * u.amp;
                let satur = -cfg.amp_saturation_beta * u.amp.powi(3);
                let d_amp =
                    (u.bias + input + influence_amp - inhibition - damp + satur + noise_a) * cfg.dt;
                let d_phase = (cfg.base_freq + influence_phase + noise_p) * cfg.dt;

                (
                    (u.amp + d_amp).clamp(-2.0, 2.0),
                    wrap_angle(u.phase + d_phase),
                )
            })
            .collect();

        // Update units and salience.
        let salience_decay = self.cfg.salience_decay;
        let salience_gain = self.cfg.salience_gain;
        let salience_threshold = self.cfg.coactive_threshold;
        let trace_decay = self.cfg.activity_trace_decay;

        if self.activity_trace.len() != self.units.len() {
            self.activity_trace.resize(self.units.len(), 0.0);
        }

        for (i, (amp, phase)) in next.into_iter().enumerate() {
            self.units[i].amp = amp;
            self.units[i].phase = phase;

            let act = amp.max(0.0);
            let tr = &mut self.activity_trace[i];
            *tr = if trace_decay <= 0.0 {
                act
            } else {
                (1.0 - trace_decay) * (*tr) + trace_decay * act
            };

            // Update salience: decay + gain when active
            let activation = (*tr - salience_threshold).max(0.0);
            self.units[i].salience =
                (1.0 - salience_decay) * self.units[i].salience + salience_gain * activation;
            self.units[i].salience = self.units[i].salience.clamp(0.0, 10.0);
        }
    }

    #[cfg(not(feature = "parallel"))]
    fn step_dynamics_parallel(&mut self) {
        self.step_dynamics_scalar();
    }

    /// GPU compute shader dynamics update.
    ///
    /// The sparse neighbor accumulation is done on CPU (irregular memory access),
    /// then the dense amp/phase update is offloaded to GPU compute shaders.
    /// Only beneficial for very large substrates (10k+ units).
    // On wasm, GPU dynamics must be driven via the nonblocking path.
    // The synchronous GPU implementation blocks on readback and can freeze the browser.
    #[cfg(all(feature = "gpu", target_arch = "wasm32"))]
    fn step_dynamics_gpu(&mut self) {
        self.step_dynamics_scalar();
    }

    #[cfg(all(feature = "gpu", not(target_arch = "wasm32")))]
    fn step_dynamics_gpu(&mut self) {
        use crate::gpu::{GpuInfluence, GpuParams, GpuUnit};

        let max_units = self.units.len().max(65_536);

        let n = self.units.len();

        // CPU: compute influences (sparse graph traversal).
        let avg_amp = self.units.iter().map(|u| u.amp).sum::<f32>() / n as f32;
        let inhibition = self.cfg.global_inhibition * avg_amp;

        let mut influences: Vec<GpuInfluence> = Vec::with_capacity(n);
        for i in 0..n {
            let u_phase = self.units[i].phase;
            let mut inf_amp = 0.0f32;
            let mut inf_phase = 0.0f32;

            for (target, weight) in self.neighbors(i) {
                let v = &self.units[target];
                inf_amp += weight * v.amp;
                inf_phase += weight * phase_coupling_term(angle_diff(v.phase, u_phase), &self.cfg);
            }

            influences.push(GpuInfluence {
                amp: inf_amp,
                phase: inf_phase,
                noise_amp: self
                    .rng
                    .gen_range_f32(-self.cfg.noise_amp, self.cfg.noise_amp),
                noise_phase: self
                    .rng
                    .gen_range_f32(-self.cfg.noise_phase, self.cfg.noise_phase),
            });
        }

        // Prepare unit data for GPU.
        let mut gpu_units: Vec<GpuUnit> = self
            .units
            .iter()
            .map(|u| GpuUnit {
                amp: u.amp,
                phase: u.phase,
                bias: u.bias,
                decay: u.decay,
            })
            .collect();

        let params = GpuParams {
            dt: self.cfg.dt,
            base_freq: self.cfg.base_freq,
            inhibition,
            unit_count: n as u32,
        };

        // Try GPU, fallback to scalar if unavailable or on error.
        let used_gpu = crate::gpu::with_gpu_context(max_units, |ctx| {
            if let Some(gpu) = ctx {
                gpu.step_dynamics(&mut gpu_units, &influences, &self.pending_input, params)
                    .is_ok()
            } else {
                false
            }
        });

        if used_gpu {
            // Write back from GPU and update salience.
            let salience_decay = self.cfg.salience_decay;
            let salience_gain = self.cfg.salience_gain;
            let salience_threshold = self.cfg.coactive_threshold;
            let trace_decay = self.cfg.activity_trace_decay;

            if self.activity_trace.len() != self.units.len() {
                self.activity_trace.resize(self.units.len(), 0.0);
            }

            for (i, gu) in gpu_units.into_iter().enumerate() {
                self.units[i].amp = gu.amp;
                self.units[i].phase = gu.phase;

                let act = gu.amp.max(0.0);
                let tr = &mut self.activity_trace[i];
                *tr = if trace_decay <= 0.0 {
                    act
                } else {
                    (1.0 - trace_decay) * (*tr) + trace_decay * act
                };

                // Update salience: decay + gain when active
                let activation = (*tr - salience_threshold).max(0.0);
                self.units[i].salience =
                    (1.0 - salience_decay) * self.units[i].salience + salience_gain * activation;
                self.units[i].salience = self.units[i].salience.clamp(0.0, 10.0);
            }
        } else {
            // Fallback to scalar if no GPU (scalar already updates salience).
            self.step_dynamics_scalar();
        }
    }

    #[cfg(not(feature = "gpu"))]
    fn step_dynamics_gpu(&mut self) {
        self.step_dynamics_scalar();
    }

    /// Returns true if a WebGPU (wasm) dynamics step is currently in flight.
    #[cfg(all(feature = "gpu", target_arch = "wasm32"))]
    pub fn wasm_gpu_step_in_flight(&self) -> bool {
        self.effective_execution_tier() == ExecutionTier::Gpu
            && crate::gpu::wasm_gpu_step_in_flight()
    }

    #[cfg(not(all(feature = "gpu", target_arch = "wasm32")))]
    pub fn wasm_gpu_step_in_flight(&self) -> bool {
        false
    }

    /// Advance the simulation by one timestep without blocking the browser.
    ///
    /// - On non-wasm targets, this just calls `step()` and returns true.
    /// - On wasm with GPU tier effective, this runs a two-phase GPU step:
    ///   start work on first call (returns false), finish/apply on a later call (returns true).
    #[cfg(all(feature = "gpu", target_arch = "wasm32"))]
    pub fn step_nonblocking(&mut self) -> bool {
        use crate::gpu::{GpuInfluence, GpuParams, GpuUnit};

        if self.effective_execution_tier() != ExecutionTier::Gpu {
            self.step();
            return true;
        }

        // If a GPU step is already in flight, try to finish it.
        if crate::gpu::wasm_gpu_step_in_flight() {
            match crate::gpu::wasm_try_finish_step_dynamics() {
                None => return false,
                Some(Err(_e)) => {
                    crate::gpu::wasm_cancel_pending_step();
                    crate::gpu::disable_gpu_for_session();
                    self.step();
                    return true;
                }
                Some(Ok(gpu_units)) => {
                    self.pruned_last_step = 0;
                    self.age_steps = self.age_steps.wrapping_add(1);
                    self.learning_monitors = LearningMonitors::default();
                    if self.telemetry.enabled {
                        self.telemetry.last_stimuli.clear();
                        self.telemetry.last_actions.clear();
                        self.telemetry.last_reinforced_actions.clear();
                    }

                    let salience_decay = self.cfg.salience_decay;
                    let salience_gain = self.cfg.salience_gain;
                    let salience_threshold = self.cfg.coactive_threshold;
                    let trace_decay = self.cfg.activity_trace_decay;

                    if self.activity_trace.len() != self.units.len() {
                        self.activity_trace.resize(self.units.len(), 0.0);
                    }
                    for (i, gu) in gpu_units.into_iter().enumerate() {
                        self.units[i].amp = gu.amp;
                        self.units[i].phase = gu.phase;

                        let act = gu.amp.max(0.0);
                        let tr = &mut self.activity_trace[i];
                        *tr = if trace_decay <= 0.0 {
                            act
                        } else {
                            (1.0 - trace_decay) * (*tr) + trace_decay * act
                        };

                        let activation = (*tr - salience_threshold).max(0.0);
                        self.units[i].salience = (1.0 - salience_decay) * self.units[i].salience
                            + salience_gain * activation;
                        self.units[i].salience = self.units[i].salience.clamp(0.0, 10.0);
                    }

                    for x in &mut self.pending_input {
                        *x = 0.0;
                    }

                    // Learning stays on CPU.
                    self.update_eligibility_scalar();
                    self.apply_plasticity_scalar();
                    self.forget_and_prune();
                    self.homeostasis_step();
                    return true;
                }
            }
        }

        // Otherwise: start a GPU step for the current state (stimuli already applied).
        let n = self.units.len();
        if n == 0 {
            return true;
        }

        let max_units = n.max(65_536);
        let avg_amp = self.units.iter().map(|u| u.amp).sum::<f32>() / n as f32;
        let inhibition = self.cfg.global_inhibition * avg_amp;

        let mut influences: Vec<GpuInfluence> = Vec::with_capacity(n);
        for i in 0..n {
            let u_phase = self.units[i].phase;
            let mut inf_amp = 0.0f32;
            let mut inf_phase = 0.0f32;

            for (target, weight) in self.neighbors(i) {
                let v = &self.units[target];
                inf_amp += weight * v.amp;
                inf_phase += weight * phase_coupling_term(angle_diff(v.phase, u_phase), &self.cfg);
            }

            influences.push(GpuInfluence {
                amp: inf_amp,
                phase: inf_phase,
                noise_amp: self
                    .rng
                    .gen_range_f32(-self.cfg.noise_amp, self.cfg.noise_amp),
                noise_phase: self
                    .rng
                    .gen_range_f32(-self.cfg.noise_phase, self.cfg.noise_phase),
            });
        }

        let gpu_units: Vec<GpuUnit> = self
            .units
            .iter()
            .map(|u| GpuUnit {
                amp: u.amp,
                phase: u.phase,
                bias: u.bias,
                decay: u.decay,
            })
            .collect();

        let params = GpuParams {
            dt: self.cfg.dt,
            base_freq: self.cfg.base_freq,
            inhibition,
            unit_count: n as u32,
        };

        let started = crate::gpu::with_gpu_context(max_units, |ctx| {
            if let Some(gpu) = ctx {
                gpu.wasm_begin_step_dynamics(&gpu_units, &influences, &self.pending_input, params)
                    .is_ok()
            } else {
                false
            }
        });

        if started {
            false
        } else {
            crate::gpu::disable_gpu_for_session();
            self.step();
            true
        }
    }

    #[cfg(not(all(feature = "gpu", target_arch = "wasm32")))]
    pub fn step_nonblocking(&mut self) -> bool {
        self.step();
        true
    }

    /// Select an action based on current unit activations.
    ///
    /// Returns the action name and its score (sum of unit amplitudes).
    ///
    /// # Arguments
    /// * `policy` - Selection strategy (deterministic or epsilon-greedy)
    ///
    /// # Example
    /// ```
    /// # use braine::substrate::{Brain, BrainConfig, ActionPolicy};
    /// # let cfg = BrainConfig::default();
    /// # let mut brain = Brain::new(cfg);
    /// brain.define_action("move", 4);
    /// let (action, score) = brain.select_action(&mut ActionPolicy::Deterministic);
    /// ```
    #[must_use]
    pub fn select_action(&mut self, policy: &mut ActionPolicy) -> (String, f32) {
        let mut scores: Vec<(String, f32)> = self
            .action_groups
            .iter()
            .map(|g| {
                (
                    g.name.clone(),
                    g.units.iter().map(|&id| self.units[id].amp).sum(),
                )
            })
            .collect();

        scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(core::cmp::Ordering::Equal));

        match policy {
            ActionPolicy::Deterministic => scores[0].clone(),
            ActionPolicy::EpsilonGreedy { epsilon } => {
                if self.rng.gen_range_f32(0.0, 1.0) < *epsilon {
                    let idx = self.rng.gen_range_usize(0, scores.len());
                    scores[idx].clone()
                } else {
                    scores[0].clone()
                }
            }
        }
    }

    /// Returns diagnostic information about the brain's current state.
    ///
    /// Useful for monitoring and visualization.
    #[must_use]
    pub fn diagnostics(&self) -> Diagnostics {
        let connection_count = self.total_connection_count();
        let avg_amp =
            self.units.iter().map(|u| u.amp).sum::<Amplitude>() / self.units.len() as Amplitude;
        let avg_weight = if connection_count > 0 {
            self.connections
                .weights
                .iter()
                .filter(|w| **w != 0.0)
                .map(|w| w.abs())
                .sum::<Weight>()
                / connection_count as Weight
        } else {
            0.0
        };
        let memory_bytes = self.estimate_memory_bytes();
        Diagnostics {
            unit_count: self.units.len(),
            connection_count,
            pruned_last_step: self.pruned_last_step,
            births_last_step: self.births_last_step,
            avg_amp,
            avg_weight,
            memory_bytes,
            execution_tier: self.effective_execution_tier(),
        }
    }

    /// Returns lightweight learning/stability monitors for the most recent step.
    #[must_use]
    pub fn learning_stats(&self) -> LearningStats {
        LearningStats {
            plasticity_committed: self.learning_monitors.plasticity_committed,
            plasticity_l1: self.learning_monitors.plasticity_l1,
            plasticity_edges: self.learning_monitors.plasticity_edges,
            plasticity_budget: self.cfg.plasticity_budget,
            plasticity_budget_used: self.learning_monitors.plasticity_budget_used,
            eligibility_l1: self.learning_monitors.eligibility_l1,
            homeostasis_bias_l1: self.learning_monitors.homeostasis_bias_l1,
        }
    }

    /// Actual memory usage estimate (accounts for neurogenesis growth).
    #[must_use]
    pub fn estimate_memory_bytes(&self) -> usize {
        let units_size = self.units.len() * core::mem::size_of::<Unit>();
        let targets_size = self.connections.targets.len() * core::mem::size_of::<UnitId>();
        let weights_size = self.connections.weights.len() * core::mem::size_of::<Weight>();
        let offsets_size = self.connections.offsets.len() * core::mem::size_of::<usize>();
        let reserved_size = self.reserved.len();
        let learning_size = self.learning_enabled.len();
        let input_size = self.pending_input.len() * core::mem::size_of::<f32>();

        units_size
            + targets_size
            + weights_size
            + offsets_size
            + reserved_size
            + learning_size
            + input_size
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Introspection API for visualization and debugging
    // ─────────────────────────────────────────────────────────────────────────

    /// Returns a slice of all unit amplitudes.
    ///
    /// Useful for heatmap visualization of brain activity.
    #[must_use]
    pub fn unit_amplitudes(&self) -> Vec<f32> {
        self.units.iter().map(|u| u.amp).collect()
    }

    /// Returns a slice of all unit phases.
    ///
    /// Phase values are in radians (-π to π).
    #[must_use]
    pub fn unit_phases(&self) -> Vec<f32> {
        self.units.iter().map(|u| u.phase).collect()
    }

    /// Returns all connection weights as a flat array.
    ///
    /// The weights correspond to the CSR storage order.
    /// Use `connection_matrix()` for a dense representation.
    #[must_use]
    pub fn connection_weights(&self) -> &[f32] {
        &self.connections.weights
    }

    /// Returns all connection targets as a flat array.
    ///
    /// Invalid connections (pruned) have target = `INVALID_UNIT`.
    #[must_use]
    pub fn connection_targets(&self) -> &[UnitId] {
        &self.connections.targets
    }

    /// Returns the CSR offset array for connection indexing.
    ///
    /// For unit `i`, its connections are at indices `offsets[i]..offsets[i+1]`.
    #[must_use]
    pub fn connection_offsets(&self) -> &[usize] {
        &self.connections.offsets
    }

    /// Returns a dense connection weight matrix (unit_count × unit_count).
    ///
    /// **Warning**: O(n²) memory for large brains. Use only for small networks.
    #[must_use]
    pub fn connection_matrix(&self) -> Vec<Vec<f32>> {
        let n = self.units.len();
        let mut matrix = vec![vec![0.0f32; n]; n];

        for (i, row) in matrix.iter_mut().enumerate().take(n) {
            let range = self.conn_range(i);
            for idx in range {
                let target = self.connections.targets[idx];
                if target != INVALID_UNIT {
                    row[target] = self.connections.weights[idx];
                }
            }
        }
        matrix
    }

    /// Returns the top N most active units by amplitude.
    ///
    /// Returns pairs of (unit_id, amplitude), sorted descending.
    #[must_use]
    pub fn top_active_units(&self, n: usize) -> Vec<(UnitId, f32)> {
        let mut indexed: Vec<_> = self
            .units
            .iter()
            .enumerate()
            .map(|(i, u)| (i, u.amp))
            .collect();
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(core::cmp::Ordering::Equal));
        indexed.truncate(n);
        indexed
    }

    /// Returns unit indices for a named sensor group.
    #[must_use]
    pub fn sensor_units(&self, name: &str) -> Option<&[UnitId]> {
        self.sensor_groups
            .iter()
            .find(|g| g.name == name)
            .map(|g| g.units.as_slice())
    }

    /// Returns unit indices for a named action group.
    #[must_use]
    pub fn action_units(&self, name: &str) -> Option<&[UnitId]> {
        self.action_groups
            .iter()
            .find(|g| g.name == name)
            .map(|g| g.units.as_slice())
    }

    /// Returns the current configuration (read-only).
    #[must_use]
    pub fn config(&self) -> &BrainConfig {
        &self.cfg
    }

    /// Update the live configuration.
    ///
    /// This is intended for tuning continuous parameters (dt, noise, learning rates, etc)
    /// from UIs/clients. It does **not** allow changing topology-bearing fields
    /// (`unit_count`, `connectivity_per_unit`) on a running brain.
    ///
    /// Returns an error if validation fails.
    pub fn update_config<F>(&mut self, f: F) -> Result<(), &'static str>
    where
        F: FnOnce(&mut BrainConfig),
    {
        let old = self.cfg;
        let old_seed = old.seed;

        let mut cfg = old;
        f(&mut cfg);

        if cfg.unit_count != old.unit_count {
            return Err("unit_count cannot be changed on a live brain");
        }
        if cfg.connectivity_per_unit != old.connectivity_per_unit {
            return Err("connectivity_per_unit cannot be changed on a live brain");
        }

        cfg.validate()?;
        self.cfg = cfg;

        // If a seed is newly set/changed, reset the PRNG for reproducibility.
        if self.cfg.seed != old_seed {
            if let Some(seed) = self.cfg.seed {
                self.rng = Prng::new(seed);
            }
        }

        Ok(())
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Neurogenesis: Growing New Units
    // ─────────────────────────────────────────────────────────────────────────

    /// Check if neurogenesis is needed based on network saturation.
    ///
    /// Returns true if the average connection weight magnitude exceeds the threshold,
    /// indicating the network may benefit from fresh capacity.
    #[must_use]
    pub fn should_grow(&self, saturation_threshold: f32) -> bool {
        let valid_count = self.total_connection_count();
        if valid_count == 0 {
            return false;
        }

        let avg_weight = self
            .connections
            .weights
            .iter()
            .filter(|w| **w != 0.0)
            .map(|w| w.abs())
            .sum::<f32>()
            / valid_count as f32;

        let legacy = avg_weight > saturation_threshold;
        if self.cfg.growth_policy_mode == 0 {
            return legacy;
        }

        // Enforce a simple cooldown between growth events (hybrid mode only).
        let since_birth = self.age_steps.wrapping_sub(self.growth_last_birth_step);
        if since_birth < self.cfg.growth_cooldown_steps as u64 {
            return false;
        }

        // Hybrid: allow growth when learning pressure stays high and pruning isn't relieving it.
        let learning_pressure = self.growth_commit_ema >= self.cfg.growth_commit_ema_threshold
            && self.growth_eligibility_norm_ema >= self.cfg.growth_eligibility_norm_ema_threshold;

        let pruning_low = self.growth_prune_norm_ema <= self.cfg.growth_prune_norm_ema_max;

        legacy || (learning_pressure && pruning_low)
    }

    /// Grow a single new unit with random wiring to existing units.
    ///
    /// The new unit starts with:
    /// - Zero amplitude (quiet)
    /// - Random phase
    /// - Small positive bias (slightly excitable)
    /// - Random connections to existing units
    ///
    /// Returns the ID of the newly created unit.
    ///
    /// # Arguments
    /// * `connectivity` - Number of outgoing connections to create
    pub fn grow_unit(&mut self, connectivity: usize) -> UnitId {
        let new_id = self.units.len();

        // Create the new unit.
        let new_unit = Unit {
            amp: 0.0,
            phase: self
                .rng
                .gen_range_f32(-core::f32::consts::PI, core::f32::consts::PI),
            bias: 0.05, // Slightly excitable so it can integrate
            decay: 0.12,
            salience: 0.0,
        };
        self.units.push(new_unit);

        // Extend auxiliary arrays.
        self.reserved.push(false);
        self.learning_enabled.push(true);
        self.pending_input.push(0.0);
        self.sensor_member.push(false);
        self.group_member.push(false);
        self.activity_trace.push(0.0);
        self.unit_module.push(NO_MODULE);

        // Append to CSR: add offset for new unit, then add connections.
        let old_end = *self.connections.offsets.last().unwrap_or(&0);
        self.connections.offsets.push(old_end + connectivity);

        // Add connections FROM the new unit TO existing units.
        // This does not create tombstones; it just extends the CSR arrays.
        for _ in 0..connectivity {
            let mut target = self.rng.gen_range_usize(0, new_id);
            // Avoid self-connection (though new_id isn't connected yet)
            if new_id > 0 && target == new_id {
                target = (target + 1) % new_id;
            }
            self.connections.targets.push(target);
            self.connections
                .weights
                .push(self.rng.gen_range_f32(-0.1, 0.1));
            self.eligibility.push(0.0);
        }

        // Also create some INCOMING connections (from random existing units TO new unit).
        // This helps the new unit get activated.
        let incoming = (connectivity / 2).max(1);
        for _ in 0..incoming {
            if new_id == 0 {
                break;
            }
            let source = self.rng.gen_range_usize(0, new_id);
            let weight = self.rng.gen_range_f32(0.05, 0.15); // Slightly positive
            self.add_or_bump_csr(source, new_id, weight);
        }

        self.births_last_step += 1;
        self.growth_last_birth_step = self.age_steps;
        self.cfg.unit_count = self.units.len();
        new_id
    }

    /// Grow multiple units at once, more efficient than calling grow_unit repeatedly.
    ///
    /// # Arguments
    /// * `count` - Number of new units to create
    /// * `connectivity` - Connections per new unit
    ///
    /// Returns the range of new unit IDs.
    pub fn grow_units(&mut self, count: usize, connectivity: usize) -> core::ops::Range<UnitId> {
        let start_id = self.units.len();

        // Reserve all capacity upfront to avoid repeated reallocations.
        self.units.reserve(count);
        self.reserved.reserve(count);
        self.learning_enabled.reserve(count);
        self.pending_input.reserve(count);
        self.sensor_member.reserve(count);
        self.group_member.reserve(count);
        self.activity_trace.reserve(count);
        self.unit_module.reserve(count);

        self.connections.offsets.reserve(count);
        self.connections.targets.reserve(count * connectivity);
        self.connections.weights.reserve(count * connectivity);
        self.eligibility.reserve(count * connectivity);

        for _ in 0..count {
            self.grow_unit(connectivity);
        }
        self.cfg.unit_count = self.units.len();
        start_id..self.units.len()
    }

    /// Automatic neurogenesis: grow units if network is saturated.
    ///
    /// This implements adaptive capacity expansion. When the average connection
    /// weight exceeds the saturation threshold, new units are added to provide
    /// fresh capacity for new concepts.
    ///
    /// # Arguments
    /// * `saturation_threshold` - Trigger growth when avg weight exceeds this (0.3-0.6 typical)
    /// * `growth_count` - Number of units to add when triggered
    /// * `max_units` - Never exceed this total unit count
    ///
    /// Returns the number of units added (0 if no growth needed or at capacity).
    pub fn maybe_neurogenesis(
        &mut self,
        saturation_threshold: f32,
        growth_count: usize,
        max_units: usize,
    ) -> usize {
        self.births_last_step = 0;

        if self.units.len() >= max_units {
            return 0;
        }

        if !self.should_grow(saturation_threshold) {
            return 0;
        }

        let to_add = growth_count.min(max_units - self.units.len());
        let connectivity = self.cfg.connectivity_per_unit;

        self.grow_units(to_add, connectivity);
        to_add
    }

    /// Targeted neurogenesis: grow units specifically connected to a named group.
    ///
    /// Creates new units that are wired to receive from and project to units
    /// in the specified group. Useful for expanding capacity for a specific
    /// sensor or action modality.
    ///
    /// # Arguments
    /// * `group_type` - "sensor" or "action"
    /// * `group_name` - Name of the existing group
    /// * `count` - Number of new units to add
    ///
    /// Returns the new unit IDs, or empty if group not found.
    pub fn grow_for_group(
        &mut self,
        group_type: &str,
        group_name: &str,
        count: usize,
    ) -> Vec<UnitId> {
        let group_units: Vec<UnitId> = match group_type {
            "sensor" => self
                .sensor_groups
                .iter()
                .find(|g| g.name == group_name)
                .map(|g| g.units.clone())
                .unwrap_or_default(),
            "action" => self
                .action_groups
                .iter()
                .find(|g| g.name == group_name)
                .map(|g| g.units.clone())
                .unwrap_or_default(),
            _ => return Vec::new(),
        };

        if group_units.is_empty() {
            return Vec::new();
        }

        let module = self.ensure_routing_module(group_type, group_name);

        let mut new_ids = Vec::with_capacity(count);

        for _ in 0..count {
            let new_id = self.units.len();

            // Create unit.
            let new_unit = Unit {
                amp: 0.0,
                phase: self
                    .rng
                    .gen_range_f32(-core::f32::consts::PI, core::f32::consts::PI),
                bias: 0.08, // Slightly more excitable for targeted growth
                decay: 0.12,
                salience: 0.0,
            };
            self.units.push(new_unit);
            self.reserved.push(false);
            self.learning_enabled.push(true);
            self.pending_input.push(0.0);
            self.sensor_member.push(false);
            self.group_member.push(false);
            self.activity_trace.push(0.0);
            self.unit_module.push(module);
            self.module_unit_counts_dirty = true;
            self.cfg.unit_count = self.units.len();

            // Wire FROM new unit TO group units (new unit can influence the group).
            let outgoing: Vec<UnitId> = group_units
                .iter()
                .take((group_units.len() / 2).max(1))
                .copied()
                .collect();
            let weights: Vec<f32> = outgoing
                .iter()
                .map(|_| self.rng.gen_range_f32(0.05, 0.2))
                .collect();

            let old_end = *self.connections.offsets.last().unwrap_or(&0);
            self.connections.offsets.push(old_end + outgoing.len());
            self.connections.targets.extend(&outgoing);
            self.connections.weights.extend(&weights);
            self.eligibility
                .resize(self.eligibility.len() + outgoing.len(), 0.0);

            // Wire FROM group units TO new unit (group can activate new unit).
            for &source in &group_units {
                let weight = self.rng.gen_range_f32(0.1, 0.25);
                self.add_or_bump_csr(source, new_id, weight);
            }

            new_ids.push(new_id);
            self.births_last_step += 1;
            self.growth_last_birth_step = self.age_steps;
        }

        new_ids
    }

    /// Prune inactive units that have been quiet for too long.
    ///
    /// This is the inverse of neurogenesis: remove units that never became useful.
    /// Only prunes units that are not in any sensor/action group and have
    /// very low activity.
    ///
    /// # Arguments
    /// * `inactivity_threshold` - Prune units with avg amplitude below this
    ///
    /// Note: This is expensive as it requires CSR rebuild. Call sparingly.
    /// Returns the number of units pruned.
    pub fn prune_inactive_units(&mut self, inactivity_threshold: f32) -> usize {
        // Identify protected units (in sensor/action groups).
        let mut protected = vec![false; self.units.len()];
        for g in &self.sensor_groups {
            for &id in &g.units {
                if id < protected.len() {
                    protected[id] = true;
                }
            }
        }
        for g in &self.action_groups {
            for &id in &g.units {
                if id < protected.len() {
                    protected[id] = true;
                }
            }
        }

        // Find units to prune.
        let to_prune: Vec<UnitId> = self
            .units
            .iter()
            .enumerate()
            .filter(|(i, u)| {
                !protected[*i] && u.amp.abs() < inactivity_threshold && u.bias.abs() < 0.02
            })
            .map(|(i, _)| i)
            .collect();

        if to_prune.is_empty() {
            return 0;
        }

        // For now, just mark them as ineffective by zeroing their connections.
        // Full removal would require re-indexing all references.
        for &id in &to_prune {
            let range = self.conn_range(id);
            for idx in range {
                if self.connections.targets[idx] != INVALID_UNIT {
                    self.connections.targets[idx] = INVALID_UNIT;
                    self.connections.weights[idx] = 0.0;
                    if idx < self.eligibility.len() {
                        self.eligibility[idx] = 0.0;
                    }
                    self.csr_tombstones += 1;
                } else {
                    self.connections.weights[idx] = 0.0;
                    if idx < self.eligibility.len() {
                        self.eligibility[idx] = 0.0;
                    }
                }
            }
            // Zero the unit's state.
            self.units[id].amp = 0.0;
            self.units[id].bias = 0.0;
            self.learning_enabled[id] = false;
        }

        to_prune.len()
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Accelerated Learning Mechanisms
    // ─────────────────────────────────────────────────────────────────────────

    /// Apply attention gating: focus learning on most active units.
    ///
    /// This temporarily disables learning for quiet units, allowing the network
    /// to focus plasticity on currently relevant concepts. Call before step().
    ///
    /// # Arguments
    /// * `top_fraction` - Fraction of units to keep learning-enabled (0.0-1.0)
    ///
    /// Returns the number of units with learning enabled after gating.
    ///
    /// # Example
    /// ```
    /// # use braine::substrate::{Brain, BrainConfig};
    /// let mut brain = Brain::new(BrainConfig::default());
    /// brain.attention_gate(0.1); // Only top 10% learn
    /// brain.step();
    /// brain.reset_learning_gates(); // Re-enable all for next cycle
    /// ```
    pub fn attention_gate(&mut self, top_fraction: f32) -> usize {
        let top_fraction = top_fraction.clamp(0.01, 1.0);
        let keep_count = ((self.units.len() as f32) * top_fraction).ceil() as usize;

        // Get amplitudes with indices, excluding reserved units.
        let mut indexed: Vec<(UnitId, f32)> = self
            .units
            .iter()
            .enumerate()
            .filter(|(i, _)| !self.reserved[*i])
            .map(|(i, u)| (i, u.amp.abs()))
            .collect();

        // Sort by amplitude descending.
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(core::cmp::Ordering::Equal));

        // Disable learning for all non-reserved units first.
        for i in 0..self.learning_enabled.len() {
            if !self.reserved[i] {
                self.learning_enabled[i] = false;
            }
        }

        // Enable learning for top N.
        let enabled = keep_count.min(indexed.len());
        for (id, _) in indexed.into_iter().take(enabled) {
            self.learning_enabled[id] = true;
        }

        enabled
    }

    /// Reset all learning gates (re-enable learning for all units).
    ///
    /// Call this after attention_gate() when you want to restore normal learning.
    pub fn reset_learning_gates(&mut self) {
        for enabled in &mut self.learning_enabled {
            *enabled = true;
        }
    }

    /// Dream replay: offline memory consolidation.
    ///
    /// Disconnects from external input and runs internal replay with boosted
    /// learning rate. Patterns that were active during waking will reactivate
    /// and strengthen through noise-driven exploration.
    ///
    /// # Arguments
    /// * `steps` - Number of dream steps to run
    /// * `learning_boost` - Multiplier for learning rate during dreaming (2.0-10.0)
    /// * `noise_boost` - Multiplier for noise during dreaming (2.0-5.0)
    ///
    /// Returns the average amplitude during dreaming (activity level).
    ///
    /// # Example
    /// ```
    /// # use braine::substrate::{Brain, BrainConfig, Stimulus};
    /// let mut brain = Brain::new(BrainConfig::default());
    /// // ... learning during the day ...
    /// brain.dream(100, 5.0, 3.0); // Consolidate memories
    /// ```
    pub fn dream(&mut self, steps: usize, learning_boost: f32, noise_boost: f32) -> f32 {
        // Save original settings.
        let orig_hebb = self.cfg.hebb_rate;
        let orig_noise_amp = self.cfg.noise_amp;
        let orig_noise_phase = self.cfg.noise_phase;
        let orig_neuromod = self.neuromod;

        // Boost learning and noise.
        self.cfg.hebb_rate = (orig_hebb * learning_boost).min(0.5);
        self.cfg.noise_amp = orig_noise_amp * noise_boost;
        self.cfg.noise_phase = orig_noise_phase * noise_boost;
        self.neuromod = 0.7; // High neuromodulator for consolidation

        // Clear any pending input (we're disconnected from the world).
        for x in &mut self.pending_input {
            *x = 0.0;
        }

        // Run dream steps.
        let mut total_amp = 0.0f32;
        for _ in 0..steps {
            // Inject small random activations to trigger memory reactivation.
            let inject_count = (self.units.len() / 20).max(1);
            for _ in 0..inject_count {
                let id = self.rng.gen_range_usize(0, self.units.len());
                if !self.reserved[id] {
                    self.pending_input[id] = self.rng.gen_range_f32(0.2, 0.6);
                }
            }

            self.step();

            // Track activity.
            total_amp += self.units.iter().map(|u| u.amp.abs()).sum::<f32>();
        }

        // Restore original settings.
        self.cfg.hebb_rate = orig_hebb;
        self.cfg.noise_amp = orig_noise_amp;
        self.cfg.noise_phase = orig_noise_phase;
        self.neuromod = orig_neuromod;

        total_amp / (steps * self.units.len()) as f32
    }

    /// Burst-mode learning: apply enhanced plasticity for dramatic activity changes.
    ///
    /// Detects units that had a large amplitude increase since last step and
    /// temporarily boosts their outgoing connection learning. This creates
    /// "flashbulb memories" for surprising or important events.
    ///
    /// Call this before step() with the previous amplitudes.
    ///
    /// # Arguments
    /// * `prev_amps` - Amplitudes from before the current step
    /// * `burst_threshold` - Minimum amplitude jump to trigger burst (0.5-1.5)
    /// * `boost_factor` - Learning rate multiplier for burst units (5.0-20.0)
    ///
    /// Returns the number of units in burst mode.
    pub fn apply_burst_learning(
        &mut self,
        prev_amps: &[f32],
        burst_threshold: f32,
        boost_factor: f32,
    ) -> usize {
        if prev_amps.len() != self.units.len() {
            return 0;
        }

        let mut burst_count = 0;
        let base_lr = self.cfg.hebb_rate;
        let boosted_lr = (base_lr * boost_factor).min(0.5);

        for (i, &prev_amp) in prev_amps.iter().enumerate().take(self.units.len()) {
            let delta = self.units[i].amp - prev_amp;

            // Check for burst: large positive jump AND currently high amplitude.
            if delta > burst_threshold && self.units[i].amp > 1.0 {
                // Apply immediate potentiation to all outgoing connections.
                let range = self.conn_range(i);
                for idx in range {
                    let target = self.connections.targets[idx];
                    if target == INVALID_UNIT {
                        continue;
                    }

                    // Only strengthen if target is also active.
                    if self.units[target].amp > self.cfg.coactive_threshold {
                        let align = phase_alignment(self.units[i].phase, self.units[target].phase);
                        let delta_w = boosted_lr * align;
                        self.connections.weights[idx] =
                            (self.connections.weights[idx] + delta_w).clamp(-1.5, 1.5);
                    }
                }

                burst_count += 1;
            }
        }

        burst_count
    }

    /// Get current amplitudes for burst detection.
    ///
    /// Call this before step() and pass the result to apply_burst_learning() after.
    #[must_use]
    pub fn get_amplitudes(&self) -> Vec<f32> {
        self.units.iter().map(|u| u.amp).collect()
    }

    /// Forced synchronization: one-shot supervised learning.
    ///
    /// Forces specified units into phase alignment and high activation, then
    /// runs a learning step. Creates strong bidirectional associations in one shot.
    /// This is a "teacher mode" for rapid skill acquisition.
    ///
    /// # Arguments
    /// * `group_a` - First group of unit IDs (e.g., stimulus units)
    /// * `group_b` - Second group of unit IDs (e.g., response units)
    /// * `strength` - Connection strength to establish (0.3-0.8)
    ///
    /// # Example
    /// ```
    /// # use braine::substrate::{Brain, BrainConfig};
    /// let mut brain = Brain::new(BrainConfig::default());
    /// brain.define_sensor("cue", 4);
    /// brain.define_action("response", 4);
    /// let cue_ids: Vec<_> = brain.sensor_units("cue").unwrap().to_vec();
    /// let resp_ids: Vec<_> = brain.action_units("response").unwrap().to_vec();
    /// brain.force_associate(&cue_ids, &resp_ids, 0.6);
    /// ```
    pub fn force_associate(&mut self, group_a: &[UnitId], group_b: &[UnitId], strength: f32) {
        let strength = strength.clamp(0.1, 1.0);

        // Force all units in both groups to same phase and high amplitude.
        let forced_phase = 0.0;
        let forced_amp = 1.5;

        for &id in group_a.iter().chain(group_b.iter()) {
            if id < self.units.len() {
                self.units[id].phase = forced_phase;
                self.units[id].amp = forced_amp;
            }
        }

        // Create/strengthen connections A -> B.
        for &a in group_a {
            for &b in group_b {
                if a < self.units.len() && b < self.units.len() && a != b {
                    self.add_or_bump_csr(a, b, strength);
                }
            }
        }

        // Create/strengthen connections B -> A (bidirectional).
        let reverse_strength = strength * 0.7; // Slightly weaker reverse.
        for &b in group_b {
            for &a in group_a {
                if a < self.units.len() && b < self.units.len() && a != b {
                    self.add_or_bump_csr(b, a, reverse_strength);
                }
            }
        }

        // Run a single step with max neuromodulator to cement the association.
        let orig_neuromod = self.neuromod;
        self.neuromod = 1.0;
        self.step();
        self.neuromod = orig_neuromod;
    }

    /// Partial forced association by group names.
    ///
    /// Convenience method that looks up sensor/action groups by name.
    ///
    /// # Arguments
    /// * `sensor_name` - Name of sensor group
    /// * `action_name` - Name of action group
    /// * `strength` - Connection strength (0.3-0.8)
    ///
    /// Returns true if both groups were found and associated.
    pub fn force_associate_groups(
        &mut self,
        sensor_name: &str,
        action_name: &str,
        strength: f32,
    ) -> bool {
        let sensor_ids: Vec<UnitId> = self
            .sensor_groups
            .iter()
            .find(|g| g.name == sensor_name)
            .map(|g| g.units.clone())
            .unwrap_or_default();

        let action_ids: Vec<UnitId> = self
            .action_groups
            .iter()
            .find(|g| g.name == action_name)
            .map(|g| g.units.clone())
            .unwrap_or_default();

        if sensor_ids.is_empty() || action_ids.is_empty() {
            return false;
        }

        self.force_associate(&sensor_ids, &action_ids, strength);
        true
    }

    /// Update eligibility traces for all active, learn-enabled outgoing edges.
    ///
    /// This is the "fast" factor that can run continuously without causing drift.
    fn update_eligibility_scalar(&mut self) {
        if self.activity_trace.len() != self.units.len() {
            self.activity_trace.resize(self.units.len(), 0.0);
        }

        let activity_for_learning = |brain: &Brain, unit: usize| -> f32 {
            let instant = brain.units[unit].amp.max(0.0);
            if brain.cfg.activity_trace_decay <= 0.0 {
                return instant;
            }
            brain.activity_trace[unit].max(instant)
        };
        if self.eligibility.len() != self.connections.weights.len() {
            self.eligibility.resize(self.connections.weights.len(), 0.0);
        }

        let thr = self.cfg.coactive_threshold;
        let phase_thr = self.cfg.phase_lock_threshold;
        let activity_thr = self.cfg.module_learning_activity_threshold;

        let decay = (1.0 - self.cfg.eligibility_decay).clamp(0.0, 1.0);
        let gain = self.cfg.eligibility_gain;
        if gain <= 0.0 {
            // Still apply decay to clear old traces.
            for e in &mut self.eligibility {
                *e *= decay;
            }
            return;
        }

        // Decay all traces first (cheap, linear) and accumulate magnitude.
        let mut l1 = 0.0f32;
        for e in &mut self.eligibility {
            *e *= decay;
            l1 += e.abs();
        }

        for owner in 0..self.units.len() {
            if !self.learning_enabled[owner] {
                continue;
            }

            if !self.learning_allowed_for_unit(owner) {
                continue;
            }

            let a_amp = activity_for_learning(self, owner);
            if activity_thr > 0.0 && a_amp < activity_thr {
                continue;
            }
            if a_amp <= thr {
                continue;
            }

            let a_phase = self.units[owner].phase;
            let range = self.conn_range(owner);

            for idx in range {
                let target = self.connections.targets[idx];
                if target == INVALID_UNIT {
                    continue;
                }

                let b_amp = activity_for_learning(self, target);
                if activity_thr > 0.0 && b_amp < activity_thr {
                    continue;
                }
                if b_amp <= thr {
                    continue;
                }

                let align = phase_alignment(a_phase, self.units[target].phase);

                // Smooth blend between no eligibility (misaligned) and alignment.
                // softness=0 keeps the legacy hard gate.
                let corr = if self.cfg.phase_gate_softness <= 0.0 {
                    if align > phase_thr {
                        align
                    } else {
                        0.0
                    }
                } else {
                    let sigma = sigmoid((align - phase_thr) / self.cfg.phase_gate_softness);
                    sigma * align
                };

                // Co-activity magnitude (soft-thresholded). softness=0 keeps hard ReLU.
                // Apply sqrt to bound the multiplicative term and prevent eligibility saturation.
                let co_raw = smooth_relu(a_amp - thr, self.cfg.coactive_softness)
                    * smooth_relu(b_amp - thr, self.cfg.coactive_softness);
                let co = co_raw.sqrt();

                let de = gain * co * corr;
                let e = &mut self.eligibility[idx];
                let prev = *e;
                let next = (prev + de).clamp(-2.0, 2.0);
                *e = next;
                l1 += next.abs() - prev.abs();
            }
        }

        // Summarize eligibility magnitude for dashboards/debugging.
        self.learning_monitors.eligibility_l1 = l1;
    }

    /// Apply a gated plasticity commit from eligibility traces.
    ///
    /// Weight update is proportional to `hebb_rate * neuromod * eligibility`.
    /// The `learning_deadband` prevents constant drift when neuromod ≈ 0.
    fn apply_plasticity_scalar(&mut self) {
        if self.cfg.hebb_rate <= 0.0 {
            return;
        }
        if self.eligibility.len() != self.connections.weights.len() {
            self.eligibility.resize(self.connections.weights.len(), 0.0);
        }

        let neuromod = self.neuromod;
        if neuromod.abs() <= self.cfg.learning_deadband {
            self.learning_monitors.plasticity_committed = false;
            return;
        }
        self.learning_monitors.plasticity_committed = true;

        // Sign-correct: negative neuromod reduces/undoes recent eligibility.
        let lr = self.cfg.hebb_rate * neuromod;

        // Optional per-step plasticity budget.
        let budget = self.cfg.plasticity_budget;
        let mut remaining_budget = if budget > 0.0 {
            self.cfg.plasticity_budget
        } else {
            f32::INFINITY
        };

        let module_budget = self.cfg.module_plasticity_budget;
        let mut module_remaining: Vec<f32> = if module_budget > 0.0 {
            vec![module_budget; self.routing_modules.len()]
        } else {
            Vec::new()
        };

        let activity_thr = self.cfg.module_learning_activity_threshold;
        let cross_plasticity_scale = self.cfg.cross_module_plasticity_scale;
        let mut l1 = 0.0f32;
        let mut edges = 0u32;

        for owner in 0..self.units.len() {
            if !self.learning_enabled[owner] {
                continue;
            }
            if !self.learning_allowed_for_unit(owner) {
                continue;
            }

            if activity_thr > 0.0 {
                let instant = self.units[owner].amp.max(0.0);
                let a = if self.cfg.activity_trace_decay <= 0.0 {
                    instant
                } else {
                    self.activity_trace
                        .get(owner)
                        .copied()
                        .unwrap_or(0.0)
                        .max(instant)
                };
                if a < activity_thr {
                    continue;
                }
            }

            let mid = self.unit_module.get(owner).copied().unwrap_or(NO_MODULE);
            if module_budget > 0.0 && mid != NO_MODULE {
                let Some(rem) = module_remaining.get(mid as usize) else {
                    continue;
                };
                if *rem <= 0.0 {
                    continue;
                }
            }

            let range = self.conn_range(owner);
            for idx in range {
                if self.connections.targets[idx] == INVALID_UNIT {
                    continue;
                }
                let target = self.connections.targets[idx];
                let e = self.eligibility[idx];
                if e == 0.0 {
                    continue;
                }

                let mut dw = lr * e;
                // Keep single-step changes bounded even under large eligibility.
                dw = dw.clamp(-0.25, 0.25);

                // Phase 3: cross-module coupling should be harder to form.
                // Only apply when both endpoints have module assignments.
                if cross_plasticity_scale != 1.0
                    && cross_plasticity_scale >= 0.0
                    && mid != NO_MODULE
                {
                    let tmid = self.unit_module.get(target).copied().unwrap_or(NO_MODULE);
                    if tmid != NO_MODULE && tmid != mid {
                        dw *= cross_plasticity_scale;
                    }
                }

                // Enforce global and (optional) per-module budgets.
                let mut allowed = remaining_budget;
                if module_budget > 0.0 && mid != NO_MODULE {
                    if let Some(mrem) = module_remaining.get(mid as usize) {
                        allowed = allowed.min(*mrem);
                    }
                }
                if allowed.is_finite() {
                    if allowed <= 0.0 {
                        break;
                    }
                    let cost = dw.abs();
                    if cost > allowed {
                        dw = dw.signum() * allowed;
                    }
                }

                let cost = dw.abs();
                if remaining_budget.is_finite() {
                    remaining_budget = (remaining_budget - cost).max(0.0);
                }
                if module_budget > 0.0 && mid != NO_MODULE {
                    if let Some(mrem) = module_remaining.get_mut(mid as usize) {
                        *mrem = (*mrem - cost).max(0.0);
                    }
                }

                self.connections.weights[idx] =
                    (self.connections.weights[idx] + dw).clamp(-1.5, 1.5);

                l1 += cost;
                edges = edges.saturating_add(1);

                if remaining_budget.is_finite() && remaining_budget <= 0.0 {
                    break;
                }
            }

            if remaining_budget.is_finite() && remaining_budget <= 0.0 {
                break;
            }
        }

        self.learning_monitors.plasticity_l1 = l1;
        self.learning_monitors.plasticity_edges = edges;
        if budget > 0.0 {
            self.learning_monitors.plasticity_budget_used = l1;
        } else {
            self.learning_monitors.plasticity_budget_used = 0.0;
        }
    }

    /// Slow homeostasis: nudges unit biases to keep activity near a target.
    #[allow(clippy::manual_is_multiple_of)]
    fn homeostasis_step(&mut self) {
        let rate = self.cfg.homeostasis_rate;
        if rate <= 0.0 {
            return;
        }

        let every = self.cfg.homeostasis_every as u64;
        // Prefer `%` over `.is_multiple_of()` to keep compatibility with the
        // pinned WASM toolchain used by `braine_web`.
        if every == 0 || (self.age_steps % every) != 0 {
            return;
        }

        let target = self.cfg.homeostasis_target_amp;
        let mut l1 = 0.0f32;
        for i in 0..self.units.len() {
            if self.reserved[i] {
                continue;
            }
            let amp = self.units[i].amp.abs();
            let err = target - amp;
            let prev = self.units[i].bias;
            self.units[i].bias = (prev + rate * err).clamp(-0.5, 0.5);
            l1 += (self.units[i].bias - prev).abs();
        }

        self.learning_monitors.homeostasis_bias_l1 = l1;
    }

    fn forget_and_prune(&mut self) {
        let decay = 1.0 - self.cfg.forget_rate;
        let prune_below = self.cfg.prune_below;
        let cross_forget = self.cfg.cross_module_forget_boost;
        let cross_prune = self.cfg.cross_module_prune_bonus;

        // Optional extra decay for cross-module edges.
        let cross_decay = (1.0 - self.cfg.forget_rate - cross_forget).clamp(0.0, 1.0);

        // Apply decay and prune. For “engrams” (sensor↔concept links), keep a weak trace:
        // allow decay, but do not prune to zero.
        //
        // Concept units are reserved but not members of any group; sensor units are tracked
        // in `sensor_member`. This avoids changing the on-disk image format.
        let unit_count = self.units.len();
        for owner in 0..unit_count {
            let owner_is_concept = self.reserved[owner] && !self.group_member[owner];
            let owner_is_sensor = self.sensor_member[owner];

            let start = self.connections.offsets[owner];
            let end = self.connections.offsets[owner + 1];
            for idx in start..end {
                let target = self.connections.targets[idx];
                if target == INVALID_UNIT {
                    continue;
                }

                let owner_mid = self.unit_module.get(owner).copied().unwrap_or(NO_MODULE);
                let target_mid = self.unit_module.get(target).copied().unwrap_or(NO_MODULE);
                let is_cross_module =
                    owner_mid != NO_MODULE && target_mid != NO_MODULE && owner_mid != target_mid;

                // Decay all active weights (optionally harsher across module boundaries).
                self.connections.weights[idx] *= if is_cross_module { cross_decay } else { decay };

                let target_is_concept = self.reserved[target] && !self.group_member[target];
                let target_is_sensor = self.sensor_member[target];
                let is_engram_edge = (owner_is_sensor && target_is_concept)
                    || (owner_is_concept && target_is_sensor);

                let w = self.connections.weights[idx];
                let abs = w.abs();

                let prune_thr = if is_cross_module {
                    prune_below + cross_prune
                } else {
                    prune_below
                };

                if is_engram_edge {
                    // Keep a minimal, non-zero trace so it can be rapidly re-strengthened
                    // on re-exposure (“savings” / muscle memory).
                    if abs < prune_thr {
                        self.connections.weights[idx] =
                            if w < 0.0 { -prune_thr } else { prune_thr };
                    }
                    continue;
                }

                if abs < prune_thr {
                    self.connections.targets[idx] = INVALID_UNIT;
                    self.connections.weights[idx] = 0.0;
                    if idx < self.eligibility.len() {
                        self.eligibility[idx] = 0.0;
                    }
                    self.pruned_last_step += 1;
                    self.csr_tombstones += 1;
                }
            }
        }

        // Compact to reclaim tombstones.
        // - Periodic compaction keeps things tidy even if tombstones are rare.
        // - Threshold-based compaction prevents long stretches of high waste.
        // NOTE: scripts/dev.sh --web-only builds with Rust 1.84; unsigned `.is_multiple_of()`
        // is unstable there, so we keep the modulo form and silence the clippy lint.
        #[allow(clippy::manual_is_multiple_of)]
        if self.age_steps % 1000 == 0
            || (self.csr_tombstones > 0
                && self.csr_tombstones * 4 > self.connections.targets.len()
                && (self.age_steps & 0x3F) == 0)
        {
            self.compact_connections();
        }
    }

    fn allocate_units(&mut self, n: usize) -> Vec<UnitId> {
        // Choose from currently unreserved units only.
        let mut idxs: Vec<(UnitId, f32)> = self
            .units
            .iter()
            .enumerate()
            .filter(|(i, _)| !self.reserved[*i])
            .map(|(i, u)| (i, u.amp.abs()))
            .collect();
        idxs.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(core::cmp::Ordering::Equal));

        let chosen: Vec<UnitId> = idxs.into_iter().take(n).map(|(i, _)| i).collect();
        for &id in &chosen {
            self.reserved[id] = true;
        }
        chosen
    }

    fn imprint_if_novel(&mut self, group_units: &[UnitId], strength: f32) {
        // If the stimulus is weak, don't imprint.
        if strength < 0.4 {
            return;
        }

        // Detect novelty by checking whether sensor units already have strong outgoing couplings.
        let mut existing_strength = 0.0;
        for &id in group_units {
            for (_, weight) in self.neighbors(id) {
                existing_strength += weight.abs();
            }
        }

        if existing_strength > 3.0 {
            return;
        }

        // Choose a "concept" unit: the quietest one not in the sensor group.
        let mut candidates: Vec<(UnitId, f32)> = self
            .units
            .iter()
            .enumerate()
            .filter(|(i, _)| !group_units.contains(i))
            .map(|(i, u)| (i, u.amp.abs()))
            .collect();
        candidates.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(core::cmp::Ordering::Equal));

        let Some((concept_id, _)) = candidates.into_iter().next() else {
            return;
        };

        // Reserve the concept so future group allocations can't overwrite it.
        // (The concept is not itself a sensor/action group member.)
        if concept_id < self.reserved.len() {
            self.reserved[concept_id] = true;
        }

        // Connect sensor units to the concept (and back) so it can be recalled.
        for &sid in group_units {
            self.add_or_bump_csr(sid, concept_id, self.cfg.imprint_rate);
            self.add_or_bump_csr(concept_id, sid, self.cfg.imprint_rate * 0.7);
        }

        // Make the concept slightly excitable.
        self.units[concept_id].bias += 0.04;
    }

    fn intern(&mut self, name: &str) -> SymbolId {
        intern_symbol(&mut self.symbols, &mut self.symbols_rev, name)
    }

    fn symbol_id(&self, name: &str) -> Option<SymbolId> {
        self.symbols.get(name).copied()
    }

    fn note_symbol(&mut self, name: &str) {
        let id = self.intern(name);
        self.active_symbols.push(id);
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Accelerated Learning Convenience API (for visualizer/experiments)
    // ─────────────────────────────────────────────────────────────────────────

    /// Trigger dream replay: run multiple offline consolidation episodes.
    ///
    /// This is a convenience wrapper around `dream()` that runs multiple episodes
    /// with sensible defaults.
    ///
    /// # Arguments
    /// * `episodes` - Number of dream episodes to run
    /// * `learning_boost` - Learning rate multiplier (1.5-5.0 recommended)
    ///
    /// Returns the average amplitude during dreaming.
    pub fn dream_replay(&mut self, episodes: usize, learning_boost: f32) -> f32 {
        let steps_per_episode = 20;
        let noise_boost = 2.5;
        let mut total_activity = 0.0;

        for _ in 0..episodes {
            total_activity += self.dream(steps_per_episode, learning_boost, noise_boost);
        }

        if episodes > 0 {
            total_activity / episodes as f32
        } else {
            0.0
        }
    }

    /// Force synchronization of all sensor groups.
    ///
    /// Aligns phases of sensor units to enhance coherent encoding.
    /// Call this after regime shifts to help the brain adapt.
    pub fn force_synchronize_sensors(&mut self) {
        let target_phase = 0.0;

        for group in &self.sensor_groups {
            for &unit_id in &group.units {
                if unit_id < self.units.len() {
                    self.units[unit_id].phase = target_phase;
                    self.units[unit_id].amp = (self.units[unit_id].amp + 0.5).min(2.0);
                }
            }
        }
    }

    /// Enable or disable burst-mode learning with a rate multiplier.
    ///
    /// When enabled, Hebbian learning rate is boosted by the given factor.
    /// This is applied on the next step() call.
    ///
    /// # Arguments
    /// * `enabled` - Whether to enable burst mode
    /// * `rate_multiplier` - Hebbian rate multiplier (1.0 = normal, 2.0-5.0 = burst)
    pub fn set_burst_mode(&mut self, enabled: bool, rate_multiplier: f32) {
        // Store in a temporary boost that modifies the effective hebb_rate.
        // We can use neuromodulator as a proxy since it already multiplies learning.
        if enabled {
            // Boost the neuromodulator to increase learning.
            let boosted = self.neuromod + rate_multiplier * 0.3;
            self.set_neuromodulator(boosted);
        }
        // Note: This is a simplified implementation. A more sophisticated version
        // would track burst state explicitly and modify hebb_rate directly.
    }

    /// Set the attention threshold for gating.
    ///
    /// Units below this amplitude threshold will have learning disabled.
    /// Higher thresholds = more selective learning (only most active units).
    ///
    /// # Arguments
    /// * `threshold` - Amplitude threshold (0.0-1.0), default around 0.3
    pub fn set_attention_threshold(&mut self, threshold: f32) {
        let threshold = threshold.clamp(0.0, 1.0);

        // Apply attention gating by disabling learning for low-amplitude units.
        for i in 0..self.units.len() {
            if !self.reserved[i] {
                self.learning_enabled[i] = self.units[i].amp >= threshold;
            }
        }
    }

    /// Imprint the current active context strongly.
    ///
    /// Creates strong associations from currently active sensor units to
    /// the most active non-reserved units. Use sparingly for one-shot learning.
    ///
    /// # Arguments
    /// * `strength` - Imprint strength (0.3-0.8 recommended)
    pub fn imprint_current_context(&mut self, strength: f32) {
        let strength = strength.clamp(0.1, 1.0);

        // Collect currently active sensor units.
        let mut active_sensors: Vec<UnitId> = Vec::new();
        for group in &self.sensor_groups {
            for &unit_id in &group.units {
                if unit_id < self.units.len() && self.units[unit_id].amp > 0.5 {
                    active_sensors.push(unit_id);
                }
            }
        }

        if active_sensors.is_empty() {
            return;
        }

        // Find the most active non-reserved units to associate with.
        let mut candidates: Vec<(UnitId, f32)> = self
            .units
            .iter()
            .enumerate()
            .filter(|(i, _)| !self.reserved[*i] && !active_sensors.contains(i))
            .map(|(i, u)| (i, u.amp))
            .collect();
        candidates.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(core::cmp::Ordering::Equal));

        // Take top 5 most active units.
        let targets: Vec<UnitId> = candidates.into_iter().take(5).map(|(id, _)| id).collect();

        if targets.is_empty() {
            return;
        }

        // Create associations.
        self.force_associate(&active_sensors, &targets, strength);
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Idle Dreaming & Sync API (for background processing when inactive)
    // ─────────────────────────────────────────────────────────────────────────

    /// Check if any units are currently active (amplitude above threshold).
    ///
    /// Returns `true` if active units are detected, `false` if the brain is idle.
    /// Use this to determine if dreaming/sync operations are safe to run.
    ///
    /// # Arguments
    /// * `threshold` - Amplitude threshold for "active" (default: 0.3)
    pub fn is_active(&self, threshold: f32) -> bool {
        self.units.iter().any(|u| u.amp.abs() > threshold)
    }

    /// Get the count of active units above the threshold.
    pub fn active_unit_count(&self, threshold: f32) -> usize {
        self.units
            .iter()
            .filter(|u| u.amp.abs() > threshold)
            .count()
    }

    /// Idle dream: run lightweight consolidation on inactive unit clusters.
    ///
    /// Unlike full `dream()`, this targets only units that have been inactive
    /// for a while (low amplitude), allowing consolidation to run in the background
    /// without disrupting active processing.
    ///
    /// # Arguments
    /// * `steps` - Number of micro-dream steps to run (1-10 recommended for idle)
    /// * `activity_threshold` - Units below this amplitude are eligible for dreaming
    ///
    /// # Returns
    /// The number of units that participated in the idle dream.
    ///
    /// # Example
    /// ```
    /// # use braine::substrate::{Brain, BrainConfig};
    /// let mut brain = Brain::new(BrainConfig::default());
    /// // When idle, run micro-dreams on inactive clusters
    /// if !brain.is_active(0.3) {
    ///     brain.idle_dream(3, 0.2);
    /// }
    /// ```
    pub fn idle_dream(&mut self, steps: usize, activity_threshold: f32) -> usize {
        let steps = steps.clamp(1, 20);
        let activity_threshold = activity_threshold.clamp(0.05, 0.5);

        // Find inactive (but not reserved) units.
        let inactive_ids: Vec<usize> = self
            .units
            .iter()
            .enumerate()
            .filter(|(i, u)| !self.reserved[*i] && u.amp.abs() < activity_threshold)
            .map(|(i, _)| i)
            .collect();

        if inactive_ids.is_empty() {
            return 0;
        }

        // Save original settings.
        let orig_hebb = self.cfg.hebb_rate;
        let orig_neuromod = self.neuromod;

        // Gentle boost for idle consolidation (less aggressive than full dream).
        self.cfg.hebb_rate = (orig_hebb * 1.5).min(0.2);
        self.neuromod = 0.5; // Moderate neuromodulator for gentle consolidation

        // Run micro-dream steps, injecting noise only into inactive units.
        for _ in 0..steps {
            // Inject small activations into a subset of inactive units.
            let inject_count = (inactive_ids.len() / 10).clamp(1, 10);
            for _ in 0..inject_count {
                let idx = self.rng.gen_range_usize(0, inactive_ids.len());
                let id = inactive_ids[idx];
                self.pending_input[id] = self.rng.gen_range_f32(0.1, 0.3);
            }

            // Single step with reduced dynamics.
            self.step();
        }

        // Restore original settings.
        self.cfg.hebb_rate = orig_hebb;
        self.neuromod = orig_neuromod;

        inactive_ids.len()
    }

    /// One-shot global synchronization: align all unit phases for coherence.
    ///
    /// This is a "reset" operation that brings all oscillators into phase alignment.
    /// Unlike `force_synchronize_sensors()` which only affects sensor groups, this
    /// aligns the entire substrate. Use sparingly as it disrupts learned phase
    /// relationships.
    ///
    /// Should only be called once when entering idle state, not repeatedly.
    ///
    /// # Returns
    /// The number of units synchronized.
    pub fn global_sync(&mut self) -> usize {
        let target_phase = 0.0;
        let mut count = 0;

        for i in 0..self.units.len() {
            // Sync all units (including reserved) to a common phase.
            // But only if they have some amplitude (skip completely silent units).
            if self.units[i].amp.abs() > 0.01 {
                self.units[i].phase = target_phase;
                count += 1;
            }
        }

        // Also sync sensor groups with a small amplitude boost.
        for group in &self.sensor_groups {
            for &unit_id in &group.units {
                if unit_id < self.units.len() {
                    self.units[unit_id].phase = target_phase;
                }
            }
        }

        count
    }

    /// Check if the brain is in learning mode (high neuromodulator).
    ///
    /// Returns `true` if neuromodulator is above 0.3 (learning is active).
    pub fn is_learning_mode(&self) -> bool {
        self.neuromod > 0.3
    }

    /// Check if the brain is in inference/reference mode (low neuromodulator).
    ///
    /// Returns `true` if neuromodulator is low, indicating read-only behavior.
    pub fn is_inference_mode(&self) -> bool {
        self.neuromod <= 0.3
    }

    /// Run scheduled idle maintenance: dreaming for inactive clusters.
    ///
    /// This is the main entry point for background consolidation. It checks
    /// if the brain is idle enough for maintenance, and if so, runs micro-dreams.
    ///
    /// # Arguments
    /// * `force` - If true, run regardless of activity level
    ///
    /// # Returns
    /// `Some(units_processed)` if dreaming was performed, `None` if skipped.
    pub fn idle_maintenance(&mut self, force: bool) -> Option<usize> {
        // Only run maintenance if:
        // 1. Brain is not actively processing (or force is true)
        // 2. Brain is not in high-learning mode (would interfere)
        if !force && self.is_active(0.4) {
            return None;
        }

        if self.is_learning_mode() && !force {
            return None;
        }

        // Run a micro-dream cycle on inactive clusters.
        let processed = self.idle_dream(3, 0.25);

        if processed > 0 {
            Some(processed)
        } else {
            None
        }
    }
}

fn intern_symbol(
    map: &mut HashMap<String, SymbolId>,
    rev: &mut Vec<String>,
    name: &str,
) -> SymbolId {
    if let Some(&id) = map.get(name) {
        return id;
    }
    let id = rev.len() as SymbolId;
    rev.push(name.to_string());
    map.insert(name.to_string(), id);
    id
}

fn wrap_angle(mut x: f32) -> f32 {
    let two_pi = 2.0 * core::f32::consts::PI;
    while x > core::f32::consts::PI {
        x -= two_pi;
    }
    while x < -core::f32::consts::PI {
        x += two_pi;
    }
    x
}

fn angle_diff(a: f32, b: f32) -> f32 {
    wrap_angle(a - b)
}

fn phase_coupling_term(delta: f32, cfg: &BrainConfig) -> f32 {
    match cfg.phase_coupling_mode {
        0 => delta,
        1 => delta.sin(),
        2 => {
            let k = cfg.phase_coupling_k.max(0.0);
            (k * delta).tanh()
        }
        _ => delta.sin(),
    }
}

fn sigmoid(x: f32) -> f32 {
    // Stable-ish sigmoid for f32.
    if x >= 0.0 {
        1.0 / (1.0 + (-x).exp())
    } else {
        let ex = x.exp();
        ex / (1.0 + ex)
    }
}

fn softplus(x: f32) -> f32 {
    // Stable softplus: log(1 + exp(x)).
    if x > 20.0 {
        x
    } else if x < -20.0 {
        x.exp()
    } else {
        (x.exp()).ln_1p()
    }
}

fn smooth_relu(x: f32, softness: f32) -> f32 {
    if softness <= 0.0 {
        x.max(0.0)
    } else {
        softness * softplus(x / softness)
    }
}

fn phase_alignment(a: f32, b: f32) -> f32 {
    // 1.0 when aligned, ~0.0 when opposite.
    let d = angle_diff(a, b).abs();
    let x = 1.0 - (d / core::f32::consts::PI);
    x.clamp(0.0, 1.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn neuromodulator_sign_controls_plasticity_direction() {
        let cfg = BrainConfig {
            unit_count: 4,
            connectivity_per_unit: 1,
            noise_amp: 0.0,
            noise_phase: 0.0,
            global_inhibition: 0.0,
            hebb_rate: 0.1,
            coactive_threshold: 0.1,
            phase_lock_threshold: 0.5,
            eligibility_decay: 0.0,
            eligibility_gain: 1.0,
            learning_deadband: 0.0,
            seed: Some(1),
            ..Default::default()
        };

        let mut brain = Brain::new(cfg);

        // Force a known single edge: 0 -> 1 at index 0.
        brain.connections.targets[0] = 1;
        brain.connections.weights[0] = 0.0;
        brain
            .eligibility
            .resize(brain.connections.weights.len(), 0.0);

        // Activate both units and align phases.
        brain.units[0].amp = 1.0;
        brain.units[1].amp = 1.0;
        brain.units[0].phase = 0.0;
        brain.units[1].phase = 0.0;

        brain.update_eligibility_scalar();
        assert!(brain.eligibility[0] > 0.0);

        brain.set_neuromodulator(1.0);
        brain.apply_plasticity_scalar();
        assert!(brain.connections.weights[0] > 0.0);

        // Reset weight; apply negative neuromod and confirm weight decreases.
        brain.connections.weights[0] = 0.0;
        brain.set_neuromodulator(-1.0);
        brain.apply_plasticity_scalar();
        assert!(brain.connections.weights[0] < 0.0);
    }

    #[test]
    fn learning_deadband_prevents_weight_drift_near_zero_neuromod() {
        let cfg = BrainConfig {
            unit_count: 4,
            connectivity_per_unit: 1,
            noise_amp: 0.0,
            noise_phase: 0.0,
            global_inhibition: 0.0,
            hebb_rate: 0.2,
            coactive_threshold: 0.1,
            phase_lock_threshold: 0.5,
            eligibility_decay: 0.0,
            eligibility_gain: 1.0,
            learning_deadband: 0.25,
            seed: Some(2),
            ..Default::default()
        };

        let mut brain = Brain::new(cfg);
        brain.connections.targets[0] = 1;
        brain.connections.weights[0] = 0.0;
        brain
            .eligibility
            .resize(brain.connections.weights.len(), 0.0);

        brain.units[0].amp = 1.0;
        brain.units[1].amp = 1.0;
        brain.units[0].phase = 0.0;
        brain.units[1].phase = 0.0;

        brain.update_eligibility_scalar();
        assert!(brain.eligibility[0] > 0.0);

        // Below deadband: no weight updates.
        brain.set_neuromodulator(0.1);
        brain.apply_plasticity_scalar();
        assert_eq!(brain.connections.weights[0], 0.0);
    }

    #[test]
    fn activity_trace_can_drive_eligibility_when_amp_drops() {
        let cfg = BrainConfig {
            unit_count: 4,
            connectivity_per_unit: 1,
            base_freq: 0.0,
            noise_amp: 0.0,
            noise_phase: 0.0,
            global_inhibition: 0.0,
            hebb_rate: 0.1,
            coactive_threshold: 0.2,
            phase_lock_threshold: 0.5,
            eligibility_decay: 0.0,
            eligibility_gain: 1.0,
            learning_deadband: 0.0,
            activity_trace_decay: 0.5,
            coactive_softness: 0.0,
            phase_gate_softness: 0.0,
            seed: Some(3),
            ..Default::default()
        };

        let mut brain = Brain::new(cfg);
        brain.connections.targets[0] = 1;
        brain.connections.weights[0] = 0.0;
        brain
            .eligibility
            .resize(brain.connections.weights.len(), 0.0);

        // Seed a strong trace directly (tests can access private fields in-module).
        brain.units[0].phase = 0.0;
        brain.units[1].phase = 0.0;
        brain.activity_trace[0] = 1.0;
        brain.activity_trace[1] = 1.0;

        // Drop instantaneous amps below threshold; trace should still drive eligibility.
        brain.units[0].amp = 0.0;
        brain.units[1].amp = 0.0;
        brain.update_eligibility_scalar();
        assert!(brain.eligibility[0] > 0.0);
    }

    #[test]
    fn brain_image_roundtrip_basic() {
        let cfg = BrainConfig {
            unit_count: 32,
            connectivity_per_unit: 4,
            dt: 0.05,
            base_freq: 1.0,
            noise_amp: 0.1,
            noise_phase: 0.05,
            global_inhibition: 0.02,
            hebb_rate: 0.01,
            forget_rate: 0.001,
            prune_below: 0.0001,
            coactive_threshold: 0.2,
            phase_lock_threshold: 0.2,
            imprint_rate: 0.3,
            seed: Some(123),
            causal_decay: 0.01,
            growth_policy_mode: 1,
            growth_cooldown_steps: 17,
            growth_signal_alpha: 0.12,
            growth_commit_ema_threshold: 0.33,
            growth_eligibility_norm_ema_threshold: 0.044,
            growth_prune_norm_ema_max: 0.0009,
            causal_lag_steps: 5,
            causal_lag_decay: 0.62,
            causal_symbol_cap: 21,
            module_routing_top_k: 3,
            module_routing_strict: true,
            module_routing_beta: 0.9,
            module_signature_decay: 0.2,
            module_signature_cap: 9,
            module_learning_activity_threshold: 0.42,
            module_plasticity_budget: 1.23,
            cross_module_plasticity_scale: 0.25,
            cross_module_forget_boost: 0.02,
            cross_module_prune_bonus: 0.003,
            ..Default::default()
        };

        let mut brain = Brain::new(cfg);
        brain.define_sensor("vision", 3);
        brain.define_action("move", 2);
        brain.define_module("ctx", 4);
        let mut bytes: Vec<u8> = Vec::new();
        brain.save_image_to(&mut bytes).unwrap();

        let mut cursor = std::io::Cursor::new(bytes);
        let loaded = Brain::load_image_from(&mut cursor).unwrap();

        assert_eq!(loaded.cfg.unit_count, brain.cfg.unit_count);
        assert_eq!(
            loaded.cfg.connectivity_per_unit,
            brain.cfg.connectivity_per_unit
        );
        assert_eq!(loaded.cfg.growth_policy_mode, brain.cfg.growth_policy_mode);
        assert_eq!(
            loaded.cfg.growth_cooldown_steps,
            brain.cfg.growth_cooldown_steps
        );
        assert!((loaded.cfg.growth_signal_alpha - brain.cfg.growth_signal_alpha).abs() < 1e-6);
        assert!(
            (loaded.cfg.growth_commit_ema_threshold - brain.cfg.growth_commit_ema_threshold).abs()
                < 1e-6
        );
        assert!(
            (loaded.cfg.growth_eligibility_norm_ema_threshold
                - brain.cfg.growth_eligibility_norm_ema_threshold)
                .abs()
                < 1e-6
        );
        assert!(
            (loaded.cfg.growth_prune_norm_ema_max - brain.cfg.growth_prune_norm_ema_max).abs()
                < 1e-6
        );
        assert_eq!(loaded.cfg.causal_lag_steps, brain.cfg.causal_lag_steps);
        assert!((loaded.cfg.causal_lag_decay - brain.cfg.causal_lag_decay).abs() < 1e-6);
        assert_eq!(loaded.cfg.causal_symbol_cap, brain.cfg.causal_symbol_cap);
        assert_eq!(
            loaded.cfg.module_routing_top_k,
            brain.cfg.module_routing_top_k
        );
        assert_eq!(
            loaded.cfg.module_routing_strict,
            brain.cfg.module_routing_strict
        );
        assert!((loaded.cfg.module_routing_beta - brain.cfg.module_routing_beta).abs() < 1e-6);
        assert!(
            (loaded.cfg.module_signature_decay - brain.cfg.module_signature_decay).abs() < 1e-6
        );
        assert_eq!(
            loaded.cfg.module_signature_cap,
            brain.cfg.module_signature_cap
        );
        assert!(
            (loaded.cfg.module_learning_activity_threshold
                - brain.cfg.module_learning_activity_threshold)
                .abs()
                < 1e-6
        );
        assert!(
            (loaded.cfg.module_plasticity_budget - brain.cfg.module_plasticity_budget).abs() < 1e-6
        );
        assert!(
            (loaded.cfg.cross_module_plasticity_scale - brain.cfg.cross_module_plasticity_scale)
                .abs()
                < 1e-6
        );
        assert!(
            (loaded.cfg.cross_module_forget_boost - brain.cfg.cross_module_forget_boost).abs()
                < 1e-6
        );
        assert!(
            (loaded.cfg.cross_module_prune_bonus - brain.cfg.cross_module_prune_bonus).abs() < 1e-6
        );
        assert_eq!(loaded.units.len(), brain.units.len());
        assert_eq!(loaded.reserved.len(), brain.reserved.len());
        assert_eq!(loaded.learning_enabled.len(), brain.learning_enabled.len());
        assert_eq!(loaded.symbols_rev, brain.symbols_rev);
        assert_eq!(loaded.sensor_groups.len(), brain.sensor_groups.len());
        assert_eq!(loaded.action_groups.len(), brain.action_groups.len());
        assert_eq!(loaded.latent_groups.len(), brain.latent_groups.len());
        assert_eq!(loaded.latent_groups[0].name, brain.latent_groups[0].name);
        assert_eq!(
            loaded.latent_groups[0].units.len(),
            brain.latent_groups[0].units.len()
        );

        // Verify CSR connections match.
        assert_eq!(
            loaded.connections.offsets.len(),
            brain.connections.offsets.len()
        );
        assert_eq!(
            loaded.total_connection_count(),
            brain.total_connection_count()
        );
    }

    #[test]
    fn routing_gates_plasticity_by_module() {
        let cfg = BrainConfig {
            unit_count: 12,
            connectivity_per_unit: 1,
            dt: 0.05,
            base_freq: 1.0,
            noise_amp: 0.0,
            noise_phase: 0.0,
            global_inhibition: 0.0,
            hebb_rate: 0.1,
            forget_rate: 0.0,
            prune_below: 0.0,
            coactive_threshold: 0.2,
            phase_lock_threshold: 0.2,
            imprint_rate: 0.0,
            learning_deadband: 0.0,
            module_routing_top_k: 1,
            module_routing_strict: true,
            // Keep activity gating off so we can seed eligibility directly.
            module_learning_activity_threshold: 0.0,
            module_plasticity_budget: 0.0,
            seed: Some(1),
            causal_decay: 0.01,
            ..Default::default()
        };

        let mut brain = Brain::new(cfg);
        brain.define_action("a1", 1);
        brain.define_action("a2", 1);

        let u_a1 = brain.action_units("a1").unwrap()[0];
        let u_a2 = brain.action_units("a2").unwrap()[0];

        let idx_a1 = brain.connections.offsets[u_a1];
        let idx_a2 = brain.connections.offsets[u_a2];

        brain.connections.weights[idx_a1] = 0.0;
        brain.connections.weights[idx_a2] = 0.0;
        brain.eligibility[idx_a1] = 1.0;
        brain.eligibility[idx_a2] = 1.0;

        // Route learning to module "a1".
        brain.note_action("a1");
        brain.commit_observation();
        assert!(!brain.learning_route_modules.is_empty());

        brain.neuromod = 1.0;
        brain.apply_plasticity_scalar();

        assert!(brain.connections.weights[idx_a1].abs() > 0.0);
        assert_eq!(brain.connections.weights[idx_a2], 0.0);
    }

    #[test]
    fn cross_module_plasticity_is_scaled_down() {
        let cfg = BrainConfig {
            unit_count: 12,
            connectivity_per_unit: 1,
            dt: 0.05,
            base_freq: 1.0,
            noise_amp: 0.0,
            noise_phase: 0.0,
            global_inhibition: 0.0,
            hebb_rate: 0.1,
            forget_rate: 0.0,
            prune_below: 0.0,
            coactive_threshold: 0.2,
            phase_lock_threshold: 0.2,
            imprint_rate: 0.0,
            learning_deadband: 0.0,
            // Disable routing so both edges are eligible.
            module_routing_top_k: 0,
            module_routing_strict: false,
            module_learning_activity_threshold: 0.0,
            module_plasticity_budget: 0.0,
            cross_module_plasticity_scale: 0.25,
            seed: Some(2),
            causal_decay: 0.01,
            ..Default::default()
        };

        let mut brain = Brain::new(cfg);
        brain.define_action("a1", 1);
        brain.define_action("a2", 1);

        let u_a1 = brain.action_units("a1").unwrap()[0];
        let u_a2 = brain.action_units("a2").unwrap()[0];

        let idx_a1 = brain.connections.offsets[u_a1];
        let idx_a2 = brain.connections.offsets[u_a2];

        // Force the first outgoing connection of each owner to point to the opposite module.
        brain.connections.targets[idx_a1] = u_a2;
        brain.connections.targets[idx_a2] = u_a1;
        brain.connections.weights[idx_a1] = 0.0;
        brain.connections.weights[idx_a2] = 0.0;
        brain.eligibility[idx_a1] = 1.0;
        brain.eligibility[idx_a2] = 1.0;

        brain.neuromod = 1.0;
        brain.apply_plasticity_scalar();

        // With lr=hebb_rate*neuromod=0.1 and eligibility=1, base dw=0.1.
        // Cross-module scaling at 0.25 yields dw≈0.025.
        assert!((brain.connections.weights[idx_a1] - 0.025).abs() < 1e-6);
        assert!((brain.connections.weights[idx_a2] - 0.025).abs() < 1e-6);
    }

    #[test]
    fn latent_modules_participate_in_routing() {
        let cfg = BrainConfig {
            unit_count: 12,
            connectivity_per_unit: 1,
            dt: 0.05,
            base_freq: 1.0,
            noise_amp: 0.0,
            noise_phase: 0.0,
            global_inhibition: 0.0,
            hebb_rate: 0.1,
            forget_rate: 0.0,
            prune_below: 0.0,
            imprint_rate: 0.0,
            learning_deadband: 0.0,
            module_routing_top_k: 1,
            module_routing_strict: true,
            module_signature_decay: 0.1,
            seed: Some(7),
            causal_decay: 0.01,
            ..Default::default()
        };

        let mut brain = Brain::new(cfg);
        brain.define_module("ctx", 2);

        let ctx_mid = *brain
            .routing_module_index
            .get("latent::ctx")
            .expect("latent module should exist");

        brain.note_compound_symbol(&["ctx"]);
        brain.commit_observation();

        assert_eq!(brain.learning_route_modules, vec![ctx_mid]);
    }

    #[test]
    fn latent_module_auto_create_triggers_on_novel_symbols_when_router_uninformative() {
        let cfg = BrainConfig {
            unit_count: 24,
            connectivity_per_unit: 2,
            dt: 0.05,
            base_freq: 1.0,
            noise_amp: 0.0,
            noise_phase: 0.0,
            global_inhibition: 0.0,
            hebb_rate: 0.1,
            forget_rate: 0.0,
            prune_below: 0.0,
            imprint_rate: 0.0,
            learning_deadband: 0.0,
            module_routing_top_k: 1,
            module_routing_strict: true,
            module_signature_decay: 0.2,
            latent_module_auto_create: true,
            latent_module_auto_width: 4,
            latent_module_auto_cooldown_steps: 0,
            latent_module_auto_reward_threshold: 0.0,
            seed: Some(11),
            causal_decay: 0.01,
            ..Default::default()
        };

        let mut brain = Brain::new(cfg);
        brain.define_sensor("s", 2);

        // Create a novel committed symbol set that has no signature associations yet.
        brain.note_symbol("novel_evt");
        brain.neuromod = 0.8;
        brain.commit_observation();

        assert!(
            brain
                .latent_groups
                .iter()
                .any(|g| g.name.starts_with("auto_latent_")),
            "expected an auto-created latent module"
        );
        assert_eq!(brain.learning_route_modules.len(), 1);
        let mid = brain.learning_route_modules[0] as usize;
        assert!(brain
            .routing_modules
            .get(mid)
            .is_some_and(|m| m.name.starts_with("auto_latent_")));
    }

    #[test]
    fn latent_module_retirement_frees_units_when_stale_and_low_reward() {
        let cfg = BrainConfig {
            unit_count: 24,
            connectivity_per_unit: 2,
            dt: 0.05,
            base_freq: 1.0,
            noise_amp: 0.0,
            noise_phase: 0.0,
            global_inhibition: 0.0,
            hebb_rate: 0.1,
            forget_rate: 0.0,
            prune_below: 0.0,
            imprint_rate: 0.0,
            learning_deadband: 0.0,
            module_routing_top_k: 1,
            latent_module_retire_after_steps: 10,
            latent_module_retire_reward_threshold: 0.05,
            seed: Some(12),
            causal_decay: 0.01,
            ..Default::default()
        };

        let mut brain = Brain::new(cfg);
        brain.age_steps = 100;
        brain.define_module("tmp", 4);
        assert_eq!(brain.latent_groups.len(), 1);
        let units = brain.latent_groups[0].units.clone();

        let mid = *brain
            .routing_module_index
            .get("latent::tmp")
            .expect("latent module should exist") as usize;
        brain.routing_modules[mid].last_routed_step = 0;
        brain.routing_modules[mid].reward_ema = 0.0;

        brain.commit_observation();

        assert!(brain.latent_groups.is_empty(), "expected module to retire");
        for id in units {
            assert_eq!(brain.unit_module[id], NO_MODULE);
            assert!(!brain.group_member[id]);
        }
    }

    #[test]
    fn csr_neighbors_iteration() {
        let cfg = BrainConfig {
            unit_count: 8,
            connectivity_per_unit: 2,
            dt: 0.05,
            base_freq: 1.0,
            noise_amp: 0.0,
            noise_phase: 0.0,
            global_inhibition: 0.0,
            hebb_rate: 0.01,
            forget_rate: 0.0,
            prune_below: 0.0,
            coactive_threshold: 0.2,
            phase_lock_threshold: 0.2,
            imprint_rate: 0.0,
            seed: Some(42),
            causal_decay: 0.01,
            ..Default::default()
        };

        let brain = Brain::new(cfg);

        // Each unit should have exactly 2 connections.
        for i in 0..8 {
            let count = brain.neighbors(i).count();
            assert_eq!(count, 2, "unit {} should have 2 neighbors", i);
        }

        // Total connections = 8 * 2 = 16.
        assert_eq!(brain.total_connection_count(), 16);
    }

    #[test]
    fn update_config_allows_tuning_but_not_topology() {
        let mut brain = Brain::new(BrainConfig::with_size(32, 4));

        // Safe tuning: dt within valid range.
        brain.update_config(|cfg| cfg.dt = 0.1).unwrap();
        assert!((brain.config().dt - 0.1).abs() < 1e-6);

        // Topology changes are rejected.
        let err = brain.update_config(|cfg| cfg.unit_count = 64).unwrap_err();
        assert!(err.contains("unit_count"));
        let err = brain
            .update_config(|cfg| cfg.connectivity_per_unit = 8)
            .unwrap_err();
        assert!(err.contains("connectivity_per_unit"));
    }

    #[test]
    fn ensure_group_min_width_grows_existing_groups() {
        let cfg = BrainConfig {
            unit_count: 32,
            connectivity_per_unit: 2,
            dt: 0.05,
            base_freq: 1.0,
            noise_amp: 0.0,
            noise_phase: 0.0,
            global_inhibition: 0.0,
            hebb_rate: 0.01,
            forget_rate: 0.0,
            prune_below: 0.0,
            coactive_threshold: 0.2,
            phase_lock_threshold: 0.2,
            imprint_rate: 0.0,
            seed: Some(7),
            causal_decay: 0.01,
            ..Default::default()
        };

        let mut brain = Brain::new(cfg);

        brain.define_sensor("s", 2);
        brain.define_action("a", 2);

        assert_eq!(brain.sensor_units("s").unwrap().len(), 2);
        assert_eq!(brain.action_units("a").unwrap().len(), 2);

        let grew_s = brain.ensure_sensor_min_width("s", 5);
        let grew_a = brain.ensure_action_min_width("a", 6);

        assert!(grew_s > 0);
        assert!(grew_a > 0);

        assert_eq!(brain.sensor_units("s").unwrap().len(), 5);
        assert_eq!(brain.action_units("a").unwrap().len(), 6);
    }

    #[test]
    fn execution_tier_switch() {
        let cfg = BrainConfig {
            unit_count: 16,
            connectivity_per_unit: 2,
            dt: 0.05,
            base_freq: 1.0,
            noise_amp: 0.01,
            noise_phase: 0.01,
            global_inhibition: 0.02,
            hebb_rate: 0.01,
            forget_rate: 0.001,
            prune_below: 0.0001,
            coactive_threshold: 0.2,
            phase_lock_threshold: 0.2,
            imprint_rate: 0.0,
            seed: Some(99),
            causal_decay: 0.01,
            ..Default::default()
        };

        let mut brain = Brain::new(cfg);

        // Default tier is Scalar.
        assert_eq!(brain.execution_tier(), ExecutionTier::Scalar);

        // Step with scalar.
        for _ in 0..10 {
            brain.step();
        }
        let _scalar_amp = brain.diagnostics().avg_amp;

        // Switch to SIMD (falls back to scalar if feature not enabled).
        brain.set_execution_tier(ExecutionTier::Simd);
        #[cfg(feature = "simd")]
        assert_eq!(brain.diagnostics().execution_tier, ExecutionTier::Simd);
        #[cfg(not(feature = "simd"))]
        assert_eq!(brain.diagnostics().execution_tier, ExecutionTier::Scalar);

        // Switch to Parallel (falls back to scalar if feature not enabled).
        brain.set_execution_tier(ExecutionTier::Parallel);
        #[cfg(feature = "parallel")]
        assert_eq!(brain.diagnostics().execution_tier, ExecutionTier::Parallel);
        #[cfg(not(feature = "parallel"))]
        assert_eq!(brain.diagnostics().execution_tier, ExecutionTier::Scalar);

        // Step with parallel.
        for _ in 0..10 {
            brain.step();
        }

        // Should still be functioning.
        let parallel_amp = brain.diagnostics().avg_amp;
        assert!(parallel_amp.is_finite());
    }

    #[test]
    fn brain_clone() {
        let cfg = BrainConfig::with_size(32, 4).with_seed(42);
        let mut brain = Brain::new(cfg);
        brain.define_sensor("stim", 4);
        brain.define_action("act", 4);

        // Run a few steps
        brain.apply_stimulus(Stimulus::new("stim", 1.0));
        brain.set_neuromodulator(0.5);
        for _ in 0..10 {
            brain.step();
        }

        // Clone the brain
        let cloned = brain.clone();

        // Verify cloned state matches
        assert_eq!(cloned.age_steps(), brain.age_steps());
        assert_eq!(
            cloned.diagnostics().unit_count,
            brain.diagnostics().unit_count
        );
        assert_eq!(
            cloned.diagnostics().connection_count,
            brain.diagnostics().connection_count
        );
        assert_eq!(cloned.execution_tier(), brain.execution_tier());

        // Verify they evolve independently
        brain.step();
        assert_eq!(cloned.age_steps() + 1, brain.age_steps());
    }

    #[test]
    fn brain_config_default() {
        let cfg = BrainConfig::default();
        assert_eq!(cfg.unit_count, 256);
        assert_eq!(cfg.connectivity_per_unit, 12);
        assert!(cfg.seed.is_none());
    }

    #[test]
    fn brain_config_builder() {
        let cfg = BrainConfig::with_size(512, 16)
            .with_seed(42)
            .with_hebb_rate(0.1)
            .with_noise(0.05, 0.025);

        assert_eq!(cfg.unit_count, 512);
        assert_eq!(cfg.connectivity_per_unit, 16);
        assert_eq!(cfg.seed, Some(42));
        assert_eq!(cfg.hebb_rate, 0.1);
        assert_eq!(cfg.noise_amp, 0.05);
        assert_eq!(cfg.noise_phase, 0.025);
    }

    #[test]
    fn introspection_methods() {
        let cfg = BrainConfig::with_size(64, 8).with_seed(42);
        let mut brain = Brain::new(cfg);
        brain.define_sensor("vision", 4);
        brain.define_action("move", 4);

        // Test unit_amplitudes
        let amps = brain.unit_amplitudes();
        assert_eq!(amps.len(), 64);

        // Test unit_phases
        let phases = brain.unit_phases();
        assert_eq!(phases.len(), 64);

        // Test connection accessors
        assert!(!brain.connection_weights().is_empty());
        assert!(!brain.connection_targets().is_empty());
        assert_eq!(brain.connection_offsets().len(), 65); // unit_count + 1

        // Test connection_matrix
        let matrix = brain.connection_matrix();
        assert_eq!(matrix.len(), 64);
        assert_eq!(matrix[0].len(), 64);

        // Test top_active_units
        brain.apply_stimulus(Stimulus::new("vision", 1.0));
        brain.step();
        let top = brain.top_active_units(5);
        assert!(top.len() <= 5);

        // Test sensor/action unit accessors
        assert!(brain.sensor_units("vision").is_some());
        assert!(brain.action_units("move").is_some());
        assert!(brain.sensor_units("nonexistent").is_none());

        // Test config accessor
        let cfg_ref = brain.config();
        assert_eq!(cfg_ref.unit_count, 64);
    }

    #[test]
    fn inference_step_does_not_modify_weights() {
        let cfg = BrainConfig::with_size(64, 6).with_seed(7);
        let mut brain = Brain::new(cfg);
        brain.define_sensor("stim", 8);
        brain.define_action("left", 4);
        brain.define_action("right", 4);

        let base = brain.clone();

        brain.apply_stimulus_inference(Stimulus::new("stim", 1.0));
        for _ in 0..3 {
            brain.step_inference();
        }

        let delta = brain.diff_weights_topk(&base, 32);
        assert!(
            delta.weight_deltas.is_empty(),
            "inference stepping should not change weights"
        );
    }

    #[test]
    fn config_validation() {
        // Valid config
        let cfg = BrainConfig::with_size(64, 8);
        assert!(cfg.validate().is_ok());

        // Test estimated memory
        let mem = cfg.estimated_memory_bytes();
        assert!(mem > 0);
        assert!(mem < 1024 * 1024); // Should be < 1MB for 64 units

        // Test diagnostics includes new fields
        let brain = Brain::new(cfg);
        let diag = brain.diagnostics();
        assert!(diag.memory_bytes > 0);
        assert_eq!(diag.execution_tier, ExecutionTier::Scalar);
    }

    #[test]
    #[should_panic(expected = "unit_count must be >= 4")]
    fn config_rejects_tiny_network() {
        let _ = BrainConfig::with_size(2, 1);
    }

    #[test]
    #[should_panic(expected = "connectivity_per_unit must be < unit_count")]
    fn config_rejects_overconnected() {
        let _ = BrainConfig::with_size(16, 20);
    }

    // =========================================================================
    // Neurogenesis Tests
    // =========================================================================

    #[test]
    fn neurogenesis_grow_single_unit() {
        let cfg = BrainConfig::with_size(32, 4).with_seed(42);
        let mut brain = Brain::new(cfg);

        let initial_count = brain.units.len();
        let new_id = brain.grow_unit(4);

        assert_eq!(new_id, initial_count);
        assert_eq!(brain.units.len(), initial_count + 1);
        assert_eq!(brain.reserved.len(), brain.units.len());
        assert_eq!(brain.learning_enabled.len(), brain.units.len());
        assert_eq!(brain.pending_input.len(), brain.units.len());

        // New unit should have connections
        let conn_count = brain.neighbors(new_id).count();
        assert_eq!(conn_count, 4, "New unit should have 4 outgoing connections");

        // Check births counter
        assert_eq!(brain.births_last_step, 1);
    }

    #[test]
    fn neurogenesis_grow_multiple_units() {
        let cfg = BrainConfig::with_size(32, 4).with_seed(42);
        let mut brain = Brain::new(cfg);

        let initial_count = brain.units.len();
        let range = brain.grow_units(8, 4);

        assert_eq!(range.start, initial_count);
        assert_eq!(range.end, initial_count + 8);
        assert_eq!(brain.units.len(), initial_count + 8);

        // Each new unit should have connections
        for id in range {
            let conn_count = brain.neighbors(id).count();
            assert!(
                conn_count >= 4,
                "Unit {} should have at least 4 connections",
                id
            );
        }
    }

    #[test]
    fn neurogenesis_maybe_grow_when_saturated() {
        let cfg = BrainConfig::with_size(16, 4).with_seed(42);
        let mut brain = Brain::new(cfg);

        // Initially, weights are small (~0.15), so should NOT grow
        let grown = brain.maybe_neurogenesis(0.5, 4, 100);
        assert_eq!(grown, 0, "Should not grow when not saturated");

        // Artificially saturate the network
        for w in &mut brain.connections.weights {
            *w = 0.8;
        }

        // Now it should grow
        let grown = brain.maybe_neurogenesis(0.5, 4, 100);
        assert_eq!(grown, 4, "Should grow 4 units when saturated");
        assert_eq!(brain.units.len(), 20);
    }

    #[test]
    fn neurogenesis_respects_max_units() {
        let cfg = BrainConfig::with_size(16, 4).with_seed(42);
        let mut brain = Brain::new(cfg);

        // Saturate
        for w in &mut brain.connections.weights {
            *w = 0.9;
        }

        // Try to grow more than allowed
        let grown = brain.maybe_neurogenesis(0.5, 100, 20);
        assert_eq!(grown, 4, "Should only grow up to max_units");
        assert_eq!(brain.units.len(), 20);

        // At max, should not grow further
        let grown = brain.maybe_neurogenesis(0.5, 100, 20);
        assert_eq!(grown, 0, "Should not grow beyond max_units");
    }

    #[test]
    fn neurogenesis_hybrid_policy_respects_cooldown() {
        let mut cfg = BrainConfig::with_size(16, 4).with_seed(42);
        cfg.growth_policy_mode = 1;
        cfg.growth_cooldown_steps = 10;
        cfg.growth_signal_alpha = 1.0; // make EMA updates effectively immediate
        cfg.growth_commit_ema_threshold = 0.5;
        cfg.growth_eligibility_norm_ema_threshold = 0.5;
        cfg.growth_prune_norm_ema_max = 0.5;

        let mut brain = Brain::new(cfg);

        // Avoid legacy saturation trigger; rely solely on hybrid signals.
        let saturation_threshold = 10.0;

        brain.growth_commit_ema = 1.0;
        brain.growth_eligibility_norm_ema = 1.0;
        brain.growth_prune_norm_ema = 0.0;

        brain.age_steps = 100;
        brain.growth_last_birth_step = 100;
        assert!(
            !brain.should_grow(saturation_threshold),
            "Hybrid mode should enforce cooldown"
        );

        brain.age_steps = 111;
        assert!(
            brain.should_grow(saturation_threshold),
            "Hybrid mode should allow growth after cooldown"
        );
    }

    #[test]
    fn neurogenesis_targeted_for_group() {
        let cfg = BrainConfig::with_size(32, 4).with_seed(42);
        let mut brain = Brain::new(cfg);
        brain.define_sensor("vision", 4);

        let initial_count = brain.units.len();
        let new_ids = brain.grow_for_group("sensor", "vision", 2);

        assert_eq!(new_ids.len(), 2);
        assert_eq!(brain.units.len(), initial_count + 2);

        // New units should be connected to vision group
        let vision_units = brain.sensor_units("vision").unwrap();
        for &new_id in &new_ids {
            // Check incoming connections from vision units
            let mut has_incoming = false;
            for &vision_id in vision_units {
                for (target, _) in brain.neighbors(vision_id) {
                    if target == new_id {
                        has_incoming = true;
                        break;
                    }
                }
            }
            assert!(
                has_incoming,
                "New unit should receive connections from vision group"
            );
        }
    }

    #[test]
    fn neurogenesis_diagnostics_include_births() {
        let cfg = BrainConfig::with_size(16, 4).with_seed(42);
        let mut brain = Brain::new(cfg);

        brain.grow_unit(4);
        brain.grow_unit(4);

        let diag = brain.diagnostics();
        assert_eq!(diag.births_last_step, 2);
        assert_eq!(diag.unit_count, 18);
    }

    #[test]
    fn neurogenesis_units_can_integrate() {
        let cfg = BrainConfig::with_size(32, 4).with_seed(42);
        let mut brain = Brain::new(cfg);
        brain.define_sensor("stim", 4);
        brain.define_action("act", 4);

        // Add some new units
        let new_ids = brain.grow_units(4, 4);

        // Run some steps to let them integrate
        for _ in 0..50 {
            brain.apply_stimulus(Stimulus::new("stim", 1.0));
            brain.set_neuromodulator(0.5);
            brain.step();
        }

        // New units should have some activity (not all zero)
        let mut any_active = false;
        for id in new_ids {
            if brain.units[id].amp.abs() > 0.01 {
                any_active = true;
                break;
            }
        }
        assert!(any_active, "At least one new unit should become active");
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Accelerated Learning Mechanism Tests
    // ─────────────────────────────────────────────────────────────────────────

    #[test]
    fn attention_gate_focuses_learning() {
        let cfg = BrainConfig::with_size(32, 4).with_seed(42);
        let mut brain = Brain::new(cfg);

        // Manually set some units to be active
        brain.units[0].amp = 1.5;
        brain.units[1].amp = 1.2;
        brain.units[2].amp = 0.9;
        brain.units[3].amp = 0.1;
        brain.units[4].amp = 0.05;

        // Apply attention gating to top 10%
        let enabled = brain.attention_gate(0.1);

        // Should enable learning for ~3 units (10% of 32 = 3.2, ceil = 4)
        assert!(
            (3..=5).contains(&enabled),
            "Expected 3-5 units enabled, got {}",
            enabled
        );

        // Top amplitude units should have learning enabled
        assert!(
            brain.learning_enabled[0],
            "Highest amp unit should have learning enabled"
        );
        assert!(
            brain.learning_enabled[1],
            "Second highest amp unit should have learning enabled"
        );

        // Low amplitude units should have learning disabled
        assert!(
            !brain.learning_enabled[4],
            "Lowest amp unit should have learning disabled"
        );

        // Reset and verify
        brain.reset_learning_gates();
        for enabled in &brain.learning_enabled {
            assert!(*enabled, "All learning gates should be reset");
        }
    }

    #[test]
    fn dream_consolidates_memories() {
        let cfg = BrainConfig::with_size(32, 4).with_seed(42);
        let mut brain = Brain::new(cfg);
        brain.define_sensor("stim", 4);

        // Learn something during "waking"
        for _ in 0..20 {
            brain.apply_stimulus(Stimulus::new("stim", 1.0));
            brain.set_neuromodulator(0.5);
            brain.step();
        }

        // Get weights before dreaming
        let weights_before: Vec<f32> = brain.connections.weights.clone();

        // Dream
        let avg_amp = brain.dream(50, 3.0, 2.0);

        // Should have had some activity during dreaming
        assert!(avg_amp > 0.0, "Dream should have activity");

        // Weights should have changed during dreaming
        let weights_after = &brain.connections.weights;
        let mut changed = false;
        for (before, after) in weights_before.iter().zip(weights_after.iter()) {
            if (before - after).abs() > 0.001 {
                changed = true;
                break;
            }
        }
        assert!(changed, "Weights should change during dream consolidation");

        // Config should be restored
        assert!(
            brain.cfg.hebb_rate < 0.1,
            "Hebb rate should be restored after dream"
        );
    }

    #[test]
    fn burst_learning_detects_spikes() {
        let cfg = BrainConfig::with_size(16, 4).with_seed(42);
        let mut brain = Brain::new(cfg);

        // Set up "before" state - units are quiet
        let prev_amps = brain.get_amplitudes();

        // Simulate dramatic activity spike
        brain.units[0].amp = 1.8;
        brain.units[1].amp = 1.5;
        brain.units[5].amp = 1.2; // Target for connections

        // Apply burst learning
        let burst_count = brain.apply_burst_learning(&prev_amps, 0.8, 10.0);

        // Should detect bursts
        assert!(
            burst_count >= 2,
            "Should detect at least 2 burst units, got {}",
            burst_count
        );
    }

    #[test]
    fn force_associate_creates_connections() {
        let cfg = BrainConfig::with_size(32, 4).with_seed(42);
        let mut brain = Brain::new(cfg);
        brain.define_sensor("cue", 4);
        brain.define_action("response", 4);

        let cue_ids: Vec<UnitId> = brain.sensor_units("cue").unwrap().to_vec();
        let resp_ids: Vec<UnitId> = brain.action_units("response").unwrap().to_vec();

        // Force association
        brain.force_associate(&cue_ids, &resp_ids, 0.6);

        // Check connections exist from cue to response
        let mut cue_to_resp = 0;
        for &cue_id in &cue_ids {
            for (target, weight) in brain.neighbors(cue_id) {
                if resp_ids.contains(&target) && weight > 0.3 {
                    cue_to_resp += 1;
                }
            }
        }

        assert!(
            cue_to_resp > 0,
            "Should have created cue->response connections"
        );

        // Check reverse connections
        let mut resp_to_cue = 0;
        for &resp_id in &resp_ids {
            for (target, weight) in brain.neighbors(resp_id) {
                if cue_ids.contains(&target) && weight > 0.2 {
                    resp_to_cue += 1;
                }
            }
        }

        assert!(
            resp_to_cue > 0,
            "Should have created response->cue connections"
        );
    }

    #[test]
    fn force_associate_groups_by_name() {
        let cfg = BrainConfig::with_size(32, 4).with_seed(42);
        let mut brain = Brain::new(cfg);
        brain.define_sensor("light", 4);
        brain.define_action("lever", 4);

        // Use convenience method
        let success = brain.force_associate_groups("light", "lever", 0.5);
        assert!(success, "Should find both groups and associate them");

        // Try with non-existent group
        let fail = brain.force_associate_groups("nonexistent", "lever", 0.5);
        assert!(!fail, "Should fail for non-existent groups");
    }

    #[test]
    fn combined_mechanisms_workflow() {
        let cfg = BrainConfig::with_size(64, 8).with_seed(42);
        let mut brain = Brain::new(cfg);
        brain.define_sensor("input", 8);
        brain.define_action("output", 4);

        // 1. Learn with attention gating
        for step in 0..100 {
            let prev_amps = brain.get_amplitudes();

            brain.apply_stimulus(Stimulus::new("input", 1.0));
            brain.set_neuromodulator(0.3);

            // Apply attention gating every 10 steps
            if step % 10 == 0 {
                brain.attention_gate(0.2);
            }

            brain.step();

            // Apply burst learning
            brain.apply_burst_learning(&prev_amps, 0.8, 5.0);

            // Reset gates for next cycle
            if step % 10 == 9 {
                brain.reset_learning_gates();
            }
        }

        // 2. Check for neurogenesis need
        let _grown = brain.maybe_neurogenesis(0.5, 4, 100);

        // 3. Dream to consolidate
        brain.dream(20, 3.0, 2.0);

        // 4. Force a specific association
        brain.force_associate_groups("input", "output", 0.5);

        // Should have a working network
        let diag = brain.diagnostics();
        assert!(diag.unit_count >= 64, "Should have at least initial units");
        assert!(diag.connection_count > 0, "Should have connections");
    }
}
