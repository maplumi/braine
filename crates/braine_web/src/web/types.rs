use super::GameKind;

#[derive(Default, Clone)]
pub(super) struct GameUiSnapshot {
    pub(super) spot_is_left: Option<bool>,

    pub(super) spotxy_pos: Option<(f32, f32)>,
    pub(super) spotxy_stimulus_key: String,
    pub(super) spotxy_eval: bool,
    pub(super) spotxy_mode: String,
    pub(super) spotxy_grid_n: u32,

    pub(super) reversal_active: bool,
    pub(super) reversal_flip_after_trials: u32,

    pub(super) pong_state: Option<PongUiState>,
    pub(super) pong_stimulus_key: String,
    pub(super) pong_paddle_speed: f32,
    pub(super) pong_paddle_half_height: f32,
    pub(super) pong_ball_speed: f32,
    pub(super) pong_paddle_bounce_y: f32,
    pub(super) pong_respawn_delay_s: f32,
    pub(super) pong_distractor_enabled: bool,
    pub(super) pong_distractor_speed_scale: f32,

    pub(super) sequence_state: Option<SequenceUiState>,
    pub(super) text_state: Option<TextUiState>,
    pub(super) replay_state: Option<ReplayUiState>,
}

#[derive(Clone)]
pub(super) struct ReplayUiState {
    pub(super) dataset: String,
    pub(super) index: u32,
    pub(super) total: u32,
    pub(super) trial_id: String,
}

#[derive(Clone)]
pub(super) struct SequenceUiState {
    pub(super) regime: u32,
    pub(super) token: String,
    pub(super) target_next: String,
    pub(super) outcomes: u32,
    pub(super) shift_every: u32,
}

#[derive(Clone)]
pub(super) struct TextUiState {
    pub(super) regime: u32,
    pub(super) token: String,
    pub(super) target_next: String,
    pub(super) outcomes: u32,
    pub(super) shift_every: u32,
    pub(super) vocab_size: u32,
}

#[derive(Clone, Copy)]
pub(super) struct PongUiState {
    pub(super) ball_x: f32,
    pub(super) ball_y: f32,
    pub(super) paddle_y: f32,
    pub(super) paddle_half_height: f32,
    pub(super) ball_visible: bool,
    pub(super) ball2_x: f32,
    pub(super) ball2_y: f32,
    pub(super) ball2_visible: bool,
}

pub(super) fn game_stats_storage_key(kind: GameKind) -> String {
    format!("{}{}", super::LOCALSTORAGE_GAME_STATS_PREFIX, kind.label())
}
