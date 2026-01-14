//! UI models and metadata that should be available on both wasm and native.
//!
//! Keeping these out of the wasm-only `web` module allows us to unit-test the
//! navigation/page inventory on the host.

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum DashboardTab {
    Learning,
    #[default]
    GameDetails,
    Stats,
    Analytics,
    BrainViz,
    Settings,
}

impl DashboardTab {
    pub fn label(self) -> &'static str {
        match self {
            DashboardTab::GameDetails => "Game Details",
            DashboardTab::Learning => "Learning",
            DashboardTab::Stats => "Stats",
            DashboardTab::Analytics => "Analytics",
            DashboardTab::BrainViz => "BrainViz",
            DashboardTab::Settings => "Settings",
        }
    }

    pub fn icon(self) -> &'static str {
        match self {
            DashboardTab::GameDetails => "🧩",
            DashboardTab::Learning => "🧠",
            DashboardTab::Stats => "📊",
            DashboardTab::Analytics => "📈",
            DashboardTab::BrainViz => "🕸️",
            DashboardTab::Settings => "⚙️",
        }
    }

    pub fn all() -> &'static [DashboardTab] {
        &[
            DashboardTab::BrainViz,
            DashboardTab::Learning,
            DashboardTab::GameDetails,
            DashboardTab::Stats,
            DashboardTab::Analytics,
            DashboardTab::Settings,
        ]
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum AnalyticsPanel {
    #[default]
    Performance,
    Reward,
    Choices,
    UnitPlot,
}

impl AnalyticsPanel {
    pub fn label(self) -> &'static str {
        match self {
            AnalyticsPanel::Performance => "Performance",
            AnalyticsPanel::Reward => "Reward",
            AnalyticsPanel::Choices => "Choices",
            AnalyticsPanel::UnitPlot => "Unit Plot",
        }
    }

    pub fn all() -> &'static [AnalyticsPanel] {
        &[
            AnalyticsPanel::Performance,
            AnalyticsPanel::Reward,
            AnalyticsPanel::Choices,
            AnalyticsPanel::UnitPlot,
        ]
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GameKind {
    Spot,
    Bandit,
    SpotReversal,
    SpotXY,
    Pong,
    Sequence,
    Text,
    Replay,
}

impl GameKind {
    pub fn label(self) -> &'static str {
        match self {
            GameKind::Spot => "spot",
            GameKind::Bandit => "bandit",
            GameKind::SpotReversal => "spot_reversal",
            GameKind::SpotXY => "spotxy",
            GameKind::Pong => "pong",
            GameKind::Sequence => "sequence",
            GameKind::Text => "text",
            GameKind::Replay => "replay",
        }
    }

    pub fn display_name(self) -> &'static str {
        match self {
            GameKind::Spot => "Spot",
            GameKind::Bandit => "Bandit",
            GameKind::SpotReversal => "Reversal",
            GameKind::SpotXY => "SpotXY",
            GameKind::Pong => "Pong",
            GameKind::Sequence => "Sequence",
            GameKind::Text => "Text",
            GameKind::Replay => "Replay",
        }
    }

    pub fn icon(self) -> &'static str {
        match self {
            GameKind::Spot => "🎯",
            GameKind::Bandit => "🎰",
            GameKind::SpotReversal => "🔄",
            GameKind::SpotXY => "📍",
            GameKind::Pong => "🏓",
            GameKind::Sequence => "🔢",
            GameKind::Text => "📝",
            GameKind::Replay => "📼",
        }
    }

    pub fn description(self) -> &'static str {
        match self {
            GameKind::Spot => "Binary discrimination with two stimuli (spot_left/spot_right) and two actions (left/right). One response per trial; reward is +1 for correct, −1 for wrong.",
            GameKind::Bandit => "Two-armed bandit with a constant context stimulus (bandit). Choose left/right once per trial; reward is stochastic with prob_left=0.8 and prob_right=0.2.",
            GameKind::SpotReversal => "Like Spot, but the correct mapping flips once after flip_after_trials=200. Tests adaptation to a distributional shift in reward dynamics.",
            GameKind::SpotXY => "Population-coded 2D position. In BinaryX mode, classify sign(x) into left/right. In Grid mode, choose the correct spotxy_cell_{n}_{ix}_{iy} among n² actions (web control doubles: 2×2 → 4×4 → 8×8).",
            GameKind::Pong => "Discrete-sensor Pong: ball/paddle position and velocity are binned into named sensors; actions are up/down/stay. The sim uses continuous collision detection against the arena walls and is deterministic given a fixed seed (randomness only on post-score serve).",
            GameKind::Sequence => "Next-token prediction over a small alphabet {A,B,C} with a regime shift between two fixed patterns every 60 outcomes.",
            GameKind::Text => "Next-token prediction over a byte vocabulary built from two small corpora (default: 'hello world\\n' vs 'goodbye world\\n') with a regime shift every 80 outcomes.",
            GameKind::Replay => "Dataset-driven replay: each completed trial consumes a record (stimuli + allowed actions + correct action) and emits reward based on correctness. Useful for deterministic evaluation of the advisor boundary.",
        }
    }

    pub fn what_it_tests(self) -> &'static str {
        match self {
            GameKind::Spot => "• Simple stimulus-action associations\n• Binary classification\n• Basic credit assignment\n• Fastest to learn (~50-100 trials)",
            GameKind::Bandit => "• Exploration vs exploitation trade-off\n• Stochastic reward handling\n• Value estimation under uncertainty\n• Convergence to the better arm (0.8 vs 0.2)",
            GameKind::SpotReversal => "• Behavioral flexibility\n• Rule change detection\n• Context-dependent learning\n• Unlearning old associations\n• Catastrophic forgetting resistance",
            GameKind::SpotXY => "• Multi-class classification (N² classes)\n• Spatial encoding and decoding\n• Scalable representation\n• Train/eval mode separation\n• Generalization testing",
            GameKind::Pong => "• Continuous state representation\n• Real-time motor control\n• Predictive tracking\n• Reward delay handling\n• Sensorimotor coordination",
            GameKind::Sequence => "• Temporal pattern recognition\n• Regime/distribution shifts\n• Sequence prediction over {A,B,C}\n• Phase detection\n• Attractor dynamics",
            GameKind::Text => "• Symbolic next-token prediction (byte tokens)\n• Regime/distribution shifts\n• Online learning without backprop\n• Vocabulary scaling (max_vocab)\n• Credit assignment with scalar reward",
            GameKind::Replay => "• Deterministic evaluation loop\n• Dataset-conditioned correctness reward\n• Context stability (replay::<dataset>)\n• Advisor boundary validation (context → advice)",
        }
    }

    pub fn inputs_info(self) -> &'static str {
        match self {
            GameKind::Spot => "Stimuli (by name): spot_left or spot_right\nActions: left, right\nTrial timing: controlled by Trial ms",
            GameKind::Bandit => "Stimulus: bandit (constant context)\nActions: left, right\nParameters: prob_left=0.8, prob_right=0.2",
            GameKind::SpotReversal => "Stimuli: spot_left or spot_right (+ reversal context sensor spot_rev_ctx when reversed)\nActions: left, right\nParameter: flip_after_trials=200\nNote: the web runtime also tags the meaning context with ::rev",
            GameKind::SpotXY => "Base stimulus: spotxy (context)\nSensors: pos_x_00..pos_x_15 and pos_y_00..pos_y_15 (population code)\nStimulus key: spotxy_xbin_XX or spotxy_bin_NN_IX_IY\nActions: left/right (BinaryX) OR spotxy_cell_NN_IX_IY (Grid)\nEval mode: holdout band |x| in [0.25..0.45] with learning suppressed",
            GameKind::Pong => "Base stimulus: pong (context)\nSensors: pong_ball_x_00..07, pong_ball_y_00..07, pong_paddle_y_00..07, pong_ball_visible/hidden, pong_vx_pos/neg, pong_vy_pos/neg\nOptional distractor: pong_ball2_x_00..07, pong_ball2_y_00..07, pong_ball2_visible/hidden, pong_ball2_vx_pos/neg, pong_ball2_vy_pos/neg\nStimulus key: pong_b08_vis.._bx.._by.._py.._vx.._vy.. (+ ball2 fields when enabled)\nActions: up, down, stay",
            GameKind::Sequence => "Base stimulus: sequence (context)\nSensors: seq_token_A/B/C and seq_regime_0/1\nStimulus key: seq_r{0|1}_t{A|B|C}\nActions: A, B, C",
            GameKind::Text => "Base stimulus: text (context)\nSensors: txt_regime_0/1 and txt_tok_XX (byte tokens) + txt_tok_UNK\nActions: tok_XX for bytes in vocab + tok_UNK",
            GameKind::Replay => "Stimuli/actions: defined per-trial by the dataset\nStimulus key: replay::<dataset_name>\nReward: +1 on correct_action, −1 otherwise",
        }
    }

    pub fn reward_info(self) -> &'static str {
        match self {
            GameKind::Spot => "+1.0: Correct response (stimulus matches action)\n−1.0: Incorrect response",
            GameKind::Bandit => "+1.0: Bernoulli reward (win)\n−1.0: No win\nProbabilities: left=0.8, right=0.2 (default)",
            GameKind::SpotReversal => "+1.0: Correct under current mapping\n−1.0: Incorrect\nFlip: once after flip_after_trials=200",
            GameKind::SpotXY => "+1.0: Correct classification\n−1.0: Incorrect\nEval mode: runs dynamics and action selection, but suppresses learning writes",
            GameKind::Pong => "+0.05: Action matches a simple tracking heuristic\n−0.05: Action mismatches heuristic\nEvent reward: +1 on paddle hit, −1 on miss (when the ball reaches the left boundary at x=0)\nAll rewards are clamped to [−1, +1]",
            GameKind::Sequence => "+1.0: Correct next-token prediction\n−1.0: Incorrect\nRegime flips every shift_every_outcomes=60",
            GameKind::Text => "+1.0: Correct next-token prediction\n−1.0: Incorrect\nRegime flips every shift_every_outcomes=80",
            GameKind::Replay => "+1.0: Action matches correct_action\n−1.0: Otherwise\nNotes: no stochasticity unless your dataset includes it",
        }
    }

    pub fn learning_objectives(self) -> &'static str {
        match self {
            GameKind::Spot => "• Achieve >90% accuracy consistently\n• Learn in <100 trials\n• Demonstrate stable attractor formation",
            GameKind::Bandit => "• Converge to preferred arm selection\n• Maintain ~70% reward rate at optimum\n• Balance exploration early, exploitation late",
            GameKind::SpotReversal => "• Recover accuracy after reversal within ~20 trials\n• Use context bit to accelerate switching\n• Maintain two stable modes",
            GameKind::SpotXY => "• Scale to larger grids (3×3, 4×4, 5×5+)\n• Maintain accuracy in Eval mode\n• Demonstrate spatial generalization",
            GameKind::Pong => "• Track ball trajectory predictively\n• Minimize missed balls over time\n• Develop smooth control policy",
            GameKind::Sequence => "• Predict sequences of length 3-6+\n• Recognize phase within sequence\n• Handle pattern length changes",
            GameKind::Text => "• Build character transition model\n• Adapt to regime shifts\n• Predict based on statistical regularities",
            GameKind::Replay => "• Achieve high accuracy on dataset\n• Use stable context to generalize across repeated trials\n• Validate advisor integration without action selection",
        }
    }

    pub fn all() -> &'static [GameKind] {
        &[
            GameKind::Spot,
            GameKind::Bandit,
            GameKind::SpotReversal,
            GameKind::SpotXY,
            GameKind::Pong,
            GameKind::Sequence,
            GameKind::Text,
            GameKind::Replay,
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn game_kind_inventory_is_stable() {
        let all = GameKind::all();
        assert_eq!(all.len(), 8);

        let mut labels: Vec<&'static str> = all.iter().copied().map(GameKind::label).collect();
        labels.sort_unstable();
        labels.dedup();
        assert_eq!(labels.len(), 8);

        for k in all {
            assert!(!k.label().trim().is_empty());
            assert!(!k.display_name().trim().is_empty());
            assert!(!k.icon().trim().is_empty());
            assert!(!k.description().trim().is_empty());
        }
    }

    #[test]
    fn dashboard_tabs_include_settings() {
        let all = DashboardTab::all();
        assert!(all.contains(&DashboardTab::Settings));
        assert!(all.contains(&DashboardTab::Analytics));
        assert!(all.contains(&DashboardTab::Stats));
    }

    #[test]
    fn analytics_panels_inventory_is_stable() {
        let all = AnalyticsPanel::all();
        assert!(all.contains(&AnalyticsPanel::Performance));
        assert!(all.contains(&AnalyticsPanel::Reward));
        assert!(all.contains(&AnalyticsPanel::Choices));
        assert!(all.contains(&AnalyticsPanel::UnitPlot));
    }
}
