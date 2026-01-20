//! Helpers for applying stimuli to a `braine::substrate::Brain` from game logic.
//!
//! ## Why this exists
//!
//! Games often have two kinds of inputs:
//!
//! - **Sensor channels**: many low-level feature bits (bins, population codes, wall flags).
//!   These should drive dynamics/action selection, but generally should **not** become symbols
//!   in causal/meaning memory, and should not trigger imprinting.
//! - **Task symbols**: compact context symbols (e.g. a regime token, a discrete cue).
//!   These are intentionally recorded into causal/meaning memory.
//!
//! This module makes that distinction explicit so contributors don’t have to remember the
//! nuance between `apply_stimulus` and `apply_stimulus_inference`.

#[cfg(feature = "braine")]
use braine::substrate::{Brain, Stimulus};

/// Apply a **sensor channel** (input-only).
///
/// This uses `Brain::apply_stimulus_inference`, so it:
/// - injects input current into the sensor group
/// - does **not** record a symbol event
/// - does **not** trigger imprinting
#[cfg(feature = "braine")]
#[inline]
pub fn apply_sensor_channel(brain: &mut Brain, name: &str, strength: f32) {
    brain.apply_stimulus_inference(Stimulus::new(name, strength));
}

/// Apply a **task symbol** (symbol + potential imprinting).
///
/// This uses `Brain::apply_stimulus`, so it:
/// - injects input current into the sensor group
/// - records a symbol event for causal/meaning memory
/// - may trigger one-shot imprinting
#[cfg(feature = "braine")]
#[inline]
pub fn apply_task_symbol(brain: &mut Brain, name: &str, strength: f32) {
    brain.apply_stimulus(Stimulus::new(name, strength));
}

#[cfg(all(test, feature = "braine"))]
mod tests {
    use super::*;
    use braine::substrate::{Brain, BrainConfig};

    #[test]
    fn sensor_channel_does_not_create_causal_symbols() {
        let mut brain = Brain::new(BrainConfig {
            unit_count: 16,
            connectivity_per_unit: 4,
            seed: Some(1),
            ..Default::default()
        });
        brain.define_sensor("s", 1);

        let before = brain.causal_stats().base_symbols;
        apply_sensor_channel(&mut brain, "s", 1.0);
        brain.step();
        brain.commit_observation();
        let after = brain.causal_stats().base_symbols;

        assert_eq!(before, after);
    }

    #[test]
    fn task_symbol_creates_causal_symbols() {
        let mut brain = Brain::new(BrainConfig {
            unit_count: 16,
            connectivity_per_unit: 4,
            seed: Some(2),
            ..Default::default()
        });
        brain.define_sensor("s", 1);

        let before = brain.causal_stats().base_symbols;
        apply_task_symbol(&mut brain, "s", 1.0);
        brain.step();
        brain.commit_observation();
        let after = brain.causal_stats().base_symbols;

        assert!(after > before);
    }
}
