use crate::stats::GameStats;
use crate::time::{Duration, Instant};
use std::collections::BTreeMap;

#[cfg(feature = "braine")]
use braine::substrate::{Brain, Stimulus};

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum TextToken {
    Byte(u8),
    Unk,
}

impl TextToken {
    pub fn action_name(self) -> String {
        match self {
            TextToken::Byte(b) => format!("tok_{b:02X}"),
            TextToken::Unk => "tok_UNK".to_string(),
        }
    }

    pub fn sensor_name(self) -> String {
        match self {
            TextToken::Byte(b) => format!("txt_tok_{b:02X}"),
            TextToken::Unk => "txt_tok_UNK".to_string(),
        }
    }

    pub fn display(self) -> String {
        match self {
            TextToken::Byte(b) => {
                if b == b' ' {
                    "<sp>".to_string()
                } else if b == b'\n' {
                    "\\n".to_string()
                } else if (0x21..=0x7E).contains(&b) {
                    (b as char).to_string()
                } else {
                    format!("0x{b:02X}")
                }
            }
            TextToken::Unk => "<unk>".to_string(),
        }
    }
}

/// Next-token prediction task over a byte stream.
///
/// The agent observes the current token (and an optional regime indicator) and must choose the
/// action corresponding to the *next* token.
#[derive(Debug)]
pub struct TextNextTokenGame {
    corpus0: Vec<u8>,
    corpus1: Vec<u8>,
    use_corpus1: bool,
    idx: usize,

    vocab: Vec<TextToken>,
    action_names: Vec<String>,
    sensor_names: Vec<String>,

    trial_allowed_actions: Vec<String>,

    pub trial_frame: u32,
    pub response_made: bool,
    pub last_action: Option<String>,
    pub stats: GameStats,

    shift_every_outcomes: u32,
    outcomes: u32,

    stimulus_key: String,
    trial_started_at: Instant,

    current_token: TextToken,
    correct_next: TextToken,
}

impl Default for TextNextTokenGame {
    fn default() -> Self {
        Self::new()
    }
}

impl TextNextTokenGame {
    pub fn new() -> Self {
        // Keep default vocab intentionally small and stable.
        // Two simple corpora with a mild distributional shift.
        let corpus0 = "hello world\n";
        let corpus1 = "goodbye world\n";
        Self::new_with_corpora(corpus0, corpus1, 32)
    }

    pub fn new_with_corpora(corpus0: &str, corpus1: &str, max_vocab: usize) -> Self {
        let now = Instant::now();
        let corpus0_bytes = corpus0.as_bytes().to_vec();
        let corpus1_bytes = corpus1.as_bytes().to_vec();

        let max_vocab = max_vocab.clamp(2, 512);
        let vocab = Self::build_vocab(&corpus0_bytes, &corpus1_bytes, max_vocab);

        let action_names: Vec<String> = vocab.iter().copied().map(TextToken::action_name).collect();
        let sensor_names: Vec<String> = vocab.iter().copied().map(TextToken::sensor_name).collect();

        let mut g = Self {
            corpus0: if corpus0_bytes.is_empty() {
                b"hello world\n".to_vec()
            } else {
                corpus0_bytes
            },
            corpus1: if corpus1_bytes.is_empty() {
                b"goodbye world\n".to_vec()
            } else {
                corpus1_bytes
            },
            use_corpus1: false,
            idx: 0,
            vocab,
            action_names,
            sensor_names,
            trial_allowed_actions: Vec::new(),
            trial_frame: 0,
            response_made: false,
            last_action: None,
            stats: GameStats::new(),
            // Default shift cadence is intentionally slow; the CLI demo sets its own
            // runtime parameters, and frequent shifts can mask learning in `last100`.
            shift_every_outcomes: 400,
            outcomes: 0,
            stimulus_key: String::new(),
            trial_started_at: now,
            current_token: TextToken::Unk,
            correct_next: TextToken::Unk,
        };

        g.refresh_trial_state();
        g
    }

    fn build_vocab(corpus0: &[u8], corpus1: &[u8], max_vocab: usize) -> Vec<TextToken> {
        let mut freq: BTreeMap<u8, u32> = BTreeMap::new();
        for &b in corpus0.iter().chain(corpus1.iter()) {
            *freq.entry(b).or_default() += 1;
        }

        // Sort by (descending freq, ascending byte) for determinism.
        let mut items: Vec<(u8, u32)> = freq.into_iter().collect();
        items.sort_by(|(a_b, a_f), (b_b, b_f)| b_f.cmp(a_f).then_with(|| a_b.cmp(b_b)));

        let mut vocab: Vec<TextToken> = Vec::new();
        // Reserve a slot for UNK.
        let take_n = max_vocab.saturating_sub(1).min(items.len());
        vocab.extend(
            items
                .into_iter()
                .take(take_n)
                .map(|(b, _)| TextToken::Byte(b)),
        );

        // Always include UNK so any out-of-vocab bytes are representable.
        vocab.push(TextToken::Unk);
        vocab
    }

    fn encode_byte(&self, b: u8) -> TextToken {
        if self
            .vocab
            .iter()
            .any(|t| matches!(t, TextToken::Byte(x) if *x == b))
        {
            TextToken::Byte(b)
        } else {
            TextToken::Unk
        }
    }

    fn active_corpus(&self) -> &[u8] {
        if self.use_corpus1 {
            &self.corpus1
        } else {
            &self.corpus0
        }
    }

    pub fn stimulus_name(&self) -> &'static str {
        "text"
    }

    pub fn stimulus_key(&self) -> &str {
        &self.stimulus_key
    }

    pub fn allowed_actions(&self) -> &[String] {
        &self.trial_allowed_actions
    }

    pub fn all_action_names(&self) -> &[String] {
        &self.action_names
    }

    pub fn token_sensor_names(&self) -> &[String] {
        &self.sensor_names
    }

    pub fn vocab_size(&self) -> usize {
        self.vocab.len()
    }

    pub fn regime(&self) -> u32 {
        if self.use_corpus1 {
            1
        } else {
            0
        }
    }

    pub fn current_token(&self) -> TextToken {
        self.current_token
    }

    pub fn target_next_token(&self) -> TextToken {
        self.correct_next
    }

    pub fn correct_action(&self) -> String {
        self.correct_next.action_name()
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
            self.response_made = false;
            self.last_action = None;
            self.trial_started_at = now;
        }

        let elapsed = now.duration_since(self.trial_started_at);
        self.trial_frame = elapsed.as_millis().min(u32::MAX as u128) as u32;
    }

    #[cfg(feature = "braine")]
    pub fn apply_stimuli(&self, brain: &mut Brain) {
        let regime_name = if self.use_corpus1 {
            "txt_regime_1"
        } else {
            "txt_regime_0"
        };
        brain.apply_stimulus(Stimulus::new(regime_name, 0.8));
        brain.apply_stimulus(Stimulus::new(&self.current_token.sensor_name(), 1.0));
    }

    /// Apply stimuli in **inference-only** mode.
    ///
    /// This avoids repeated imprinting when the daemon is ticking but the task
    /// is waiting for the next trial boundary.
    #[cfg(feature = "braine")]
    pub fn apply_stimuli_inference(&self, brain: &mut Brain) {
        let regime_name = if self.use_corpus1 {
            "txt_regime_1"
        } else {
            "txt_regime_0"
        };
        brain.apply_stimulus_inference(Stimulus::new(regime_name, 0.8));
        brain.apply_stimulus_inference(Stimulus::new(&self.current_token.sensor_name(), 1.0));
    }

    pub fn score_action(&mut self, action: &str) -> Option<(f32, bool)> {
        if self.response_made {
            return None;
        }

        let is_correct = action == self.correct_action();
        // Keep incorrect punishment mild so exploration doesn't constantly drive strong LTD.
        // This task is meant to measure rapid associative formation, not avoidance learning.
        let reward = if is_correct { 1.0 } else { -0.2 };

        self.response_made = true;
        self.last_action = Some(action.to_string());
        self.stats.record_trial(is_correct);

        self.outcomes = self.outcomes.wrapping_add(1);
        // NOTE: wasm build pins Rust 1.84; unsigned `.is_multiple_of()` is unstable there.
        #[allow(clippy::manual_is_multiple_of)]
        if self.shift_every_outcomes != 0 && self.outcomes % self.shift_every_outcomes == 0 {
            self.use_corpus1 = !self.use_corpus1;
        }

        self.idx = self.idx.wrapping_add(1);
        self.refresh_trial_state();

        Some((reward, true))
    }

    fn refresh_trial_state(&mut self) {
        let corpus = self.active_corpus();
        if corpus.is_empty() {
            self.current_token = TextToken::Unk;
            self.correct_next = TextToken::Unk;
        } else {
            let n = corpus.len();
            let i = self.idx % n;
            let cur = corpus[i];
            let nxt = corpus[(i + 1) % n];
            self.current_token = self.encode_byte(cur);
            self.correct_next = self.encode_byte(nxt);
        }

        self.stimulus_key = format!(
            "txt_r{}_c{}",
            if self.use_corpus1 { 1 } else { 0 },
            self.current_token.action_name()
        );

        // Present a small action set per trial (2-way choice): the correct next token + a
        // deterministic distractor. This keeps the task learnable while still requiring a
        // context-conditioned association from current token → next token.
        let correct = self.correct_action();
        let mut distractor_idx = if self.vocab.is_empty() {
            0
        } else {
            // Deterministic pseudo-random, stable across runs.
            let mix = (self.idx as u64)
                .wrapping_mul(6364136223846793005)
                .wrapping_add(if self.use_corpus1 { 1 } else { 0 });
            (mix as usize) % self.vocab.len()
        };

        // Ensure distractor differs from correct action.
        if !self.vocab.is_empty() {
            for _ in 0..self.vocab.len() {
                let cand = self.vocab[distractor_idx].action_name();
                if cand != correct {
                    break;
                }
                distractor_idx = (distractor_idx + 1) % self.vocab.len();
            }
        }
        let distractor = if self.vocab.is_empty() {
            "tok_UNK".to_string()
        } else {
            self.vocab[distractor_idx].action_name()
        };

        let distractor = if distractor == correct {
            // Degenerate case: vocab has only the correct token. Prefer UNK if it differs.
            let unk = TextToken::Unk.action_name();
            if unk != correct {
                unk
            } else {
                distractor
            }
        } else {
            distractor
        };

        self.trial_allowed_actions.clear();
        // Alternate ordering so the correct action is not always in slot 0.
        if (self.idx & 1) == 0 {
            self.trial_allowed_actions.push(correct);
            if distractor != self.trial_allowed_actions[0] {
                self.trial_allowed_actions.push(distractor);
            }
        } else {
            self.trial_allowed_actions.push(distractor);
            if correct != self.trial_allowed_actions[0] {
                self.trial_allowed_actions.push(correct);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn correct_action_matches_next_byte_in_stream() {
        let mut g = TextNextTokenGame::new_with_corpora("ABAC", "ABAC", 8);
        g.set_shift_every_outcomes(10_000);

        assert_eq!(g.regime(), 0);
        assert_eq!(g.current_token().display(), "A");
        assert_eq!(g.correct_action(), TextToken::Byte(b'B').action_name());

        let _ = g.score_action(&TextToken::Byte(b'B').action_name());
        g.response_made = false;
        assert_eq!(g.current_token().display(), "B");
        assert_eq!(g.correct_action(), TextToken::Byte(b'A').action_name());

        let _ = g.score_action(&TextToken::Byte(b'A').action_name());
        g.response_made = false;
        assert_eq!(g.current_token().display(), "A");
        assert_eq!(g.correct_action(), TextToken::Byte(b'C').action_name());
    }

    #[test]
    fn flips_regime_after_shift_every_outcomes() {
        let mut g = TextNextTokenGame::new_with_corpora("AB", "CD", 8);
        g.set_shift_every_outcomes(2);

        assert_eq!(g.regime(), 0);
        let _ = g.score_action(&g.correct_action());
        g.response_made = false;
        assert_eq!(g.regime(), 0);
        let _ = g.score_action(&g.correct_action());
        g.response_made = false;
        assert_eq!(g.regime(), 1);
    }

    #[test]
    fn always_includes_unk_token() {
        let g = TextNextTokenGame::new_with_corpora("XYZ", "", 2);
        assert!(g.all_action_names().iter().any(|a| a.as_str() == "tok_UNK"));
    }

    #[test]
    fn allowed_actions_include_correct_and_distractor() {
        let g = TextNextTokenGame::new_with_corpora("AB", "AB", 8);
        assert!(g.allowed_actions().len() >= 2);
        assert!(g
            .allowed_actions()
            .iter()
            .any(|a| a.as_str() == g.correct_action()));
    }
}
