use braine::substrate::{Brain, Stimulus};
use braine_games::sequence::SequenceGame;

#[derive(Debug)]
pub struct SequenceWebGame {
    pub game: SequenceGame,
}

impl SequenceWebGame {
    pub fn new() -> Self {
        Self {
            game: SequenceGame::new(),
        }
    }

    pub fn stimulus_name(&self) -> &'static str {
        self.game.stimulus_name()
    }

    pub fn stimulus_key(&self) -> &str {
        self.game.stimulus_key()
    }

    pub fn allowed_actions(&self) -> &[String] {
        self.game.allowed_actions()
    }

    pub fn update_timing(&mut self, trial_period_ms: u32) {
        self.game.update_timing(trial_period_ms)
    }

    pub fn apply_stimuli(&self, brain: &mut Brain) {
        // Base context name for meaning conditioning; keep separate from the more specific key.
        brain.apply_stimulus(Stimulus::new(self.stimulus_name(), 1.0));
        self.game.apply_stimuli(brain);
    }

    pub fn response_made(&self) -> bool {
        self.game.response_made
    }

    pub fn score_action(&mut self, action: &str) -> Option<(f32, bool)> {
        self.game.score_action(action)
    }
}
