use braine::substrate::ExecutionTier;
use serde::{Deserialize, Serialize};

use super::types::game_stats_storage_key;
use super::{GameKind, Theme};

fn local_storage() -> Option<web_sys::Storage> {
    web_sys::window().and_then(|w| w.local_storage().ok().flatten())
}

pub(super) fn local_storage_get_string(key: &str) -> Option<String> {
    local_storage().and_then(|s| s.get_item(key).ok().flatten())
}

pub(super) fn local_storage_set_string(key: &str, value: &str) {
    if let Some(s) = local_storage() {
        let _ = s.set_item(key, value);
    }
}

pub(super) fn local_storage_remove(key: &str) {
    if let Some(s) = local_storage() {
        let _ = s.remove_item(key);
    }
}

pub(super) fn parse_exec_tier_pref(v: &str) -> Option<ExecutionTier> {
    match v.trim().to_ascii_lowercase().as_str() {
        "scalar" | "cpu" => Some(ExecutionTier::Scalar),
        "gpu" => Some(ExecutionTier::Gpu),
        _ => None,
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(super) struct PersistedGameStats {
    pub(super) correct: u32,
    pub(super) incorrect: u32,
    pub(super) trials: u32,
    pub(super) recent: Vec<bool>,
    pub(super) learning_at_trial: Option<u32>,
    pub(super) learned_at_trial: Option<u32>,
    pub(super) mastered_at_trial: Option<u32>,
}

impl From<&braine_games::stats::GameStats> for PersistedGameStats {
    fn from(s: &braine_games::stats::GameStats) -> Self {
        Self {
            correct: s.correct,
            incorrect: s.incorrect,
            trials: s.trials,
            recent: s.recent.clone(),
            learning_at_trial: s.learning_at_trial,
            learned_at_trial: s.learned_at_trial,
            mastered_at_trial: s.mastered_at_trial,
        }
    }
}

impl PersistedGameStats {
    pub(super) fn into_game_stats(self) -> braine_games::stats::GameStats {
        braine_games::stats::GameStats {
            correct: self.correct,
            incorrect: self.incorrect,
            trials: self.trials,
            recent: self.recent,
            learning_at_trial: self.learning_at_trial,
            learned_at_trial: self.learned_at_trial,
            mastered_at_trial: self.mastered_at_trial,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(super) struct PersistedStatsState {
    pub(super) version: u32,
    pub(super) game: String,
    pub(super) stats: PersistedGameStats,
    pub(super) perf_history: Vec<f32>,
    pub(super) neuromod_history: Vec<f32>,
    #[serde(default)]
    pub(super) choice_events: Vec<String>,
    pub(super) last_action: String,
    pub(super) last_reward: f32,
}

pub(super) fn load_persisted_stats_state(kind: GameKind) -> Option<PersistedStatsState> {
    let key = game_stats_storage_key(kind);
    let raw = local_storage_get_string(&key)?;
    serde_json::from_str(&raw).ok()
}

pub(super) fn save_persisted_stats_state(kind: GameKind, state: &PersistedStatsState) {
    let key = game_stats_storage_key(kind);
    if let Ok(raw) = serde_json::to_string(state) {
        local_storage_set_string(&key, &raw);
    }
}

pub(super) fn clear_persisted_stats_state(kind: GameKind) {
    let key = game_stats_storage_key(kind);
    local_storage_remove(&key);
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(super) struct PersistedSettings {
    pub(super) reward_scale: f32,
    pub(super) reward_bias: f32,
    #[serde(default = "default_true")]
    pub(super) learning_enabled: bool,
    #[serde(default = "default_run_interval_ms")]
    pub(super) run_interval_ms: u32,
    #[serde(default = "default_trial_period_ms")]
    pub(super) trial_period_ms: u32,
    #[serde(default)]
    pub(super) settings_advanced: bool,
}

fn default_true() -> bool {
    true
}

fn default_run_interval_ms() -> u32 {
    33
}

fn default_trial_period_ms() -> u32 {
    500
}

pub(super) fn load_persisted_settings() -> Option<PersistedSettings> {
    let raw = local_storage_get_string(super::LOCALSTORAGE_SETTINGS_KEY)?;
    serde_json::from_str(&raw).ok()
}

pub(super) fn save_persisted_settings(settings: &PersistedSettings) {
    if let Ok(raw) = serde_json::to_string(settings) {
        local_storage_set_string(super::LOCALSTORAGE_SETTINGS_KEY, &raw);
    }
}

pub(super) fn apply_theme_to_document(theme: Theme) {
    let Some(doc) = web_sys::window().and_then(|w| w.document()) else {
        return;
    };
    let Some(el) = doc.document_element() else {
        return;
    };
    let _ = el.set_attribute("data-theme", theme.as_attr());
}
