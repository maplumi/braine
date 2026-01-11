use braine::substrate::{Brain, Stimulus};
use braine_games::text_next_token::TextNextTokenGame;

#[derive(Debug)]
pub struct TextWebGame {
    pub game: TextNextTokenGame,
}

impl TextWebGame {
    pub fn new() -> Self {
        Self {
            game: TextNextTokenGame::new(),
        }
    }

    pub fn new_with_corpora(
        corpus0: &str,
        corpus1: &str,
        max_vocab: usize,
        shift_every: u32,
    ) -> Self {
        let mut game = TextNextTokenGame::new_with_corpora(corpus0, corpus1, max_vocab);
        game.set_shift_every_outcomes(shift_every);
        Self { game }
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

    pub fn token_sensor_names(&self) -> &[String] {
        self.game.token_sensor_names()
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

    #[allow(dead_code)]
    pub fn shift_every_outcomes(&self) -> u32 {
        self.game.shift_every_outcomes()
    }
}
