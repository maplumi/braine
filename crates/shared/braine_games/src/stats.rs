use std::vec::Vec;

#[derive(Debug, Clone)]
pub struct GameStats {
    pub correct: u32,
    pub incorrect: u32,
    pub trials: u32,
    pub recent: Vec<bool>,
    pub learning_at_trial: Option<u32>,
    pub learned_at_trial: Option<u32>,
    pub mastered_at_trial: Option<u32>,
}

impl GameStats {
    pub fn new() -> Self {
        Self {
            correct: 0,
            incorrect: 0,
            trials: 0,
            recent: Vec::with_capacity(200),
            learning_at_trial: None,
            learned_at_trial: None,
            mastered_at_trial: None,
        }
    }

    fn update_milestones(&mut self) {
        // Keep milestone definitions consistent with the UI labels.
        // Gate on a minimum number of trials to avoid “instant” mastery on tiny samples.
        if self.trials < 20 {
            return;
        }

        let r = self.last_100_rate();
        if self.learning_at_trial.is_none() && r >= 0.70 {
            self.learning_at_trial = Some(self.trials);
        }
        if self.learned_at_trial.is_none() && r >= 0.85 {
            self.learned_at_trial = Some(self.trials);
        }
        if self.mastered_at_trial.is_none() && r >= 0.95 {
            self.mastered_at_trial = Some(self.trials);
        }
    }

    pub fn record_trial(&mut self, is_correct: bool) {
        if is_correct {
            self.correct += 1;
        } else {
            self.incorrect += 1;
        }

        self.recent.push(is_correct);
        if self.recent.len() > 200 {
            self.recent.remove(0);
        }

        self.trials += 1;
        self.update_milestones();
    }

    pub fn accuracy(&self) -> f32 {
        let total = self.correct + self.incorrect;
        if total == 0 {
            0.5
        } else {
            self.correct as f32 / total as f32
        }
    }

    pub fn recent_rate(&self) -> f32 {
        if self.recent.is_empty() {
            return 0.5;
        }
        let correct_count = self.recent.iter().filter(|&&x| x).count();
        correct_count as f32 / self.recent.len() as f32
    }

    pub fn last_100_rate(&self) -> f32 {
        if self.recent.len() < 10 {
            return self.recent_rate();
        }
        let start = if self.recent.len() > 100 {
            self.recent.len() - 100
        } else {
            0
        };
        let slice = &self.recent[start..];
        let correct_count = slice.iter().filter(|&&x| x).count();
        correct_count as f32 / slice.len() as f32
    }
}

impl Default for GameStats {
    fn default() -> Self {
        Self::new()
    }
}
