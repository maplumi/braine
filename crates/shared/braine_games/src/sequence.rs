use crate::stats::GameStats;
use std::time::{Duration, Instant};

#[cfg(feature = "braine")]
use braine::substrate::{Brain, Stimulus};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SeqToken {
    A,
    B,
    C,
}

impl SeqToken {
    pub fn label(self) -> &'static str {
        match self {
            SeqToken::A => "A",
            SeqToken::B => "B",
            SeqToken::C => "C",
        }
    }

    pub fn sensor(self) -> &'static str {
        match self {
            SeqToken::A => "seq_token_A",
            SeqToken::B => "seq_token_B",
            SeqToken::C => "seq_token_C",
        }
    }
}

/// Next-token prediction task.
///
/// At each trial the agent observes the current token (and regime), and must choose
/// the *next* token in the active pattern.
#[derive(Debug)]
pub struct SequenceGame {
    pattern0: [SeqToken; 4],
    pattern1: [SeqToken; 4],

    use_pattern1: bool,
    idx: usize,

    pub trial_frame: u32,
    pub response_made: bool,
    pub last_action: Option<String>,
    pub stats: GameStats,

    shift_every_outcomes: u32,
    outcomes: u32,

    action_names: Vec<String>,
    stimulus_key: String,

    trial_started_at: Instant,

    current_token: SeqToken,
    correct_next: SeqToken,
}

impl Default for SequenceGame {
    fn default() -> Self {
        Self::new()
    }
}

impl SequenceGame {
    pub fn new() -> Self {
        let now = Instant::now();
        let mut g = Self {
            pattern0: [SeqToken::A, SeqToken::B, SeqToken::A, SeqToken::C],
            pattern1: [SeqToken::A, SeqToken::C, SeqToken::B, SeqToken::C],
            use_pattern1: false,
            idx: 0,
            trial_frame: 0,
            response_made: false,
            last_action: None,
            stats: GameStats::new(),
            shift_every_outcomes: 60,
            outcomes: 0,
            action_names: vec!["A".to_string(), "B".to_string(), "C".to_string()],
            stimulus_key: String::new(),
            trial_started_at: now,
            current_token: SeqToken::A,
            correct_next: SeqToken::B,
        };
        g.refresh_trial_state();
        g
    }

    pub fn stimulus_name(&self) -> &'static str {
        "sequence"
    }

    pub fn stimulus_key(&self) -> &str {
        &self.stimulus_key
    }

    pub fn allowed_actions(&self) -> &[String] {
        &self.action_names
    }

    pub fn regime(&self) -> u32 {
        if self.use_pattern1 {
            1
        } else {
            0
        }
    }

    pub fn current_token(&self) -> SeqToken {
        self.current_token
    }

    pub fn correct_action(&self) -> &'static str {
        self.correct_next.label()
    }

    pub fn outcomes(&self) -> u32 {
        self.outcomes
    }

    pub fn shift_every_outcomes(&self) -> u32 {
        self.shift_every_outcomes
    }

    pub fn set_shift_every_outcomes(&mut self, n: u32) {
        self.shift_every_outcomes = n.max(1);
    }

    pub fn update_timing(&mut self, trial_period_ms: u32) {
        let trial_period_ms = trial_period_ms.clamp(10, 60_000);
        let trial_period = Duration::from_millis(trial_period_ms as u64);

        let now = Instant::now();
        let elapsed = now.duration_since(self.trial_started_at);
        if elapsed >= trial_period {
            // Allow exactly one action per timestep.
            self.response_made = false;
            self.last_action = None;
            self.trial_started_at = now;
        }

        let elapsed = now.duration_since(self.trial_started_at);
        self.trial_frame = elapsed.as_millis().min(u32::MAX as u128) as u32;
    }

    #[cfg(feature = "braine")]
    pub fn apply_stimuli(&self, brain: &mut Brain) {
        let regime_name = if self.use_pattern1 {
            "seq_regime_1"
        } else {
            "seq_regime_0"
        };

        brain.apply_stimulus(Stimulus::new(regime_name, 0.8));
        brain.apply_stimulus(Stimulus::new(self.current_token.sensor(), 1.0));
    }

    pub fn score_action(&mut self, action: &str) -> Option<(f32, bool)> {
        if self.response_made {
            return None;
        }

        let is_correct = action == self.correct_action();
        let reward = if is_correct { 1.0 } else { -1.0 };

        self.response_made = true;
        self.last_action = Some(action.to_string());
        self.stats.record_trial(is_correct);

        self.outcomes = self.outcomes.wrapping_add(1);
        if self.shift_every_outcomes != 0 && self.outcomes.is_multiple_of(self.shift_every_outcomes)
        {
            self.use_pattern1 = !self.use_pattern1;
        }

        // Advance the token stream immediately, but keep `response_made=true` until the next
        // trial window opens (timing matches the other games).
        self.idx = self.idx.wrapping_add(1);
        self.refresh_trial_state();

        Some((reward, true))
    }

    fn refresh_trial_state(&mut self) {
        // Copy the active pattern to avoid holding a borrow to `self` while mutating fields.
        let p: [SeqToken; 4] = if self.use_pattern1 {
            self.pattern1
        } else {
            self.pattern0
        };
        let n = p.len();
        let i = self.idx % n;
        self.current_token = p[i];
        self.correct_next = p[(i + 1) % n];

        self.stimulus_key = format!(
            "seq_r{}_t{}",
            if self.use_pattern1 { 1 } else { 0 },
            self.current_token.label()
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn correct_action_matches_next_token() {
        let mut g = SequenceGame::new();

        // Pattern0 = A B A C
        assert_eq!(g.regime(), 0);
        assert_eq!(g.current_token(), SeqToken::A);
        assert_eq!(g.correct_action(), "B");

        let _ = g.score_action("B");
        g.response_made = false;
        assert_eq!(g.current_token(), SeqToken::B);
        assert_eq!(g.correct_action(), "A");

        let _ = g.score_action("A");
        g.response_made = false;
        assert_eq!(g.current_token(), SeqToken::A);
        assert_eq!(g.correct_action(), "C");

        let _ = g.score_action("C");
        g.response_made = false;
        assert_eq!(g.current_token(), SeqToken::C);
        assert_eq!(g.correct_action(), "A");
    }

    #[test]
    fn flips_regime_after_shift_every_outcomes() {
        let mut g = SequenceGame::new();
        g.set_shift_every_outcomes(2);

        assert_eq!(g.regime(), 0);
        let _ = g.score_action(g.correct_action());
        g.response_made = false;
        assert_eq!(g.regime(), 0);
        let _ = g.score_action(g.correct_action());
        g.response_made = false;
        assert_eq!(g.regime(), 1);
        let _ = g.score_action(g.correct_action());
        g.response_made = false;
        assert_eq!(g.regime(), 1);
        let _ = g.score_action(g.correct_action());
        g.response_made = false;
        assert_eq!(g.regime(), 0);
    }
}
