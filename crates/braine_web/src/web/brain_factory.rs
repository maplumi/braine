use braine::substrate::{Brain, BrainConfig};

pub(super) fn make_default_brain() -> Brain {
    let mut brain = Brain::new(BrainConfig {
        seed: Some(2026),
        causal_decay: 0.002,
        ..BrainConfig::default()
    });

    // Actions used by Spot/Bandit.
    brain.define_action("left", 6);
    brain.define_action("right", 6);

    // Context stimuli.
    brain.define_sensor("spot_left", 4);
    brain.define_sensor("spot_right", 4);
    brain.define_sensor("spot_rev_ctx", 2);
    brain.define_sensor("bandit", 4);

    brain
}
