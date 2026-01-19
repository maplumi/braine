use braine::substrate::{Brain, Stimulus};
use braine_games::{
    bandit::BanditGame,
    maze::MazeGame,
    replay::{ReplayDataset, ReplayGame},
    spot::SpotGame,
    spot_reversal::SpotReversalGame,
    spot_xy::SpotXYGame,
};

use super::brain_factory::make_default_brain;
use super::types::{
    GameUiSnapshot, MazeUiState, PongUiState, ReplayUiState, SequenceUiState, TextUiState,
};
use super::GameKind;

use super::pong_web::PongWebGame;
use super::sequence_web::SequenceWebGame;
use super::text_web::TextWebGame;

#[derive(Clone, Copy)]
pub(super) struct TickConfig {
    pub(super) trial_period_ms: u32,
    pub(super) exploration_eps: f32,
    pub(super) meaning_alpha: f32,
    pub(super) reward_scale: f32,
    pub(super) reward_bias: f32,
    pub(super) learning_enabled: bool,
}

pub(super) struct TickOutput {
    pub(super) last_action: String,
    pub(super) reward: f32,
}

pub(super) enum TickResult {
    PendingGpu,
    Advanced(Option<TickOutput>),
}

struct PendingTick {
    cfg: TickConfig,
    allow_learning: bool,
    context_key: String,
    response_made_at_start: bool,
}

pub(super) struct AppRuntime {
    pub(super) brain: Brain,
    pub(super) game: WebGame,
    pub(super) pending_neuromod: f32,
    pending_tick: Option<PendingTick>,
    rng_seed: u64,
}

impl AppRuntime {
    pub(super) fn new() -> Self {
        Self {
            brain: make_default_brain(),
            game: WebGame::Spot(SpotGame::new()),
            pending_neuromod: 0.0,
            pending_tick: None,
            rng_seed: 0xC0FF_EE12u64,
        }
    }

    pub(super) fn cancel_pending_tick(&mut self) {
        self.pending_tick = None;
        #[cfg(all(feature = "gpu", target_arch = "wasm32"))]
        {
            if self.brain.wasm_gpu_step_in_flight() {
                braine::gpu::wasm_cancel_pending_step();
            }
        }
    }

    pub(super) fn set_game(&mut self, kind: GameKind) {
        self.cancel_pending_tick();
        self.game = match kind {
            GameKind::Spot => WebGame::Spot(SpotGame::new()),
            GameKind::Bandit => WebGame::Bandit(BanditGame::new()),
            GameKind::SpotReversal => WebGame::SpotReversal(SpotReversalGame::new(200)),
            GameKind::SpotXY => {
                self.ensure_spotxy_io(16);
                let g = SpotXYGame::new(16);
                self.ensure_spotxy_actions(&g);
                WebGame::SpotXY(g)
            }
            GameKind::Maze => {
                self.ensure_maze_io();
                WebGame::Maze(MazeGame::new())
            }
            GameKind::Pong => {
                self.ensure_pong_io();
                WebGame::Pong(PongWebGame::new(0xB0A7_F00Du64))
            }
            GameKind::Sequence => {
                self.ensure_sequence_io();
                WebGame::Sequence(SequenceWebGame::new())
            }
            GameKind::Text => {
                let g = TextWebGame::new();
                self.ensure_text_io(&g);
                WebGame::Text(g)
            }
            GameKind::Replay => {
                let ds = ReplayDataset::builtin_left_right_spot();
                self.ensure_replay_io(&ds);
                WebGame::Replay(ReplayGame::new(ds))
            }
        };
        self.pending_neuromod = 0.0;
    }

    fn finish_tick(
        &mut self,
        cfg: TickConfig,
        allow_learning: bool,
        context_key: String,
        response_made_at_start: bool,
    ) -> TickResult {
        let context_key = context_key.as_str();

        // IMPORTANT: For GPU/nonblocking steps, the game state may advance while a step is
        // in-flight (to avoid UI/gameplay appearing "stuck"). Action selection must remain
        // consistent with the state used to apply stimuli for the in-flight step.
        // Therefore, `response_made_at_start` is captured when the step begins and is used
        // here to decide whether we are in the "post-response" commit-only phase.
        if response_made_at_start {
            self.brain.set_neuromodulator(0.0);
            if allow_learning {
                self.brain.commit_observation();
            } else {
                self.brain.discard_observation();
            }
            return TickResult::Advanced(None);
        }

        let explore = self.rng_next_f32() < cfg.exploration_eps;
        let rand_idx = self.rng_next_u64() as usize;
        let action = match &self.game {
            WebGame::SpotXY(g) => {
                let allowed = g.allowed_actions();
                if allowed.is_empty() {
                    return TickResult::Advanced(None);
                }

                if explore {
                    allowed[rand_idx % allowed.len()].to_string()
                } else {
                    let ranked = self
                        .brain
                        .ranked_actions_with_meaning(context_key, cfg.meaning_alpha);
                    ranked
                        .into_iter()
                        .find(|(name, _score)| allowed.iter().any(|a| a == name))
                        .map(|(name, _score)| name)
                        .or_else(|| allowed.first().cloned())
                        .unwrap_or_else(|| "left".to_string())
                }
            }
            WebGame::Pong(g) => {
                let allowed = g.allowed_actions();
                if allowed.is_empty() {
                    return TickResult::Advanced(None);
                }

                if explore {
                    allowed[rand_idx % allowed.len()].to_string()
                } else {
                    let ranked = self
                        .brain
                        .ranked_actions_with_meaning(context_key, cfg.meaning_alpha);
                    ranked
                        .into_iter()
                        .find(|(name, _score)| allowed.iter().any(|a| a == name))
                        .map(|(name, _score)| name)
                        .or_else(|| allowed.first().cloned())
                        .unwrap_or_else(|| "stay".to_string())
                }
            }
            WebGame::Sequence(g) => {
                let allowed = g.allowed_actions();
                if allowed.is_empty() {
                    return TickResult::Advanced(None);
                }

                if explore {
                    allowed[rand_idx % allowed.len()].to_string()
                } else {
                    let ranked = self
                        .brain
                        .ranked_actions_with_meaning(context_key, cfg.meaning_alpha);
                    ranked
                        .into_iter()
                        .find(|(name, _score)| allowed.iter().any(|a| a == name))
                        .map(|(name, _score)| name)
                        .or_else(|| allowed.first().cloned())
                        .unwrap_or_else(|| "A".to_string())
                }
            }
            WebGame::Text(g) => {
                let allowed = g.allowed_actions();
                if allowed.is_empty() {
                    return TickResult::Advanced(None);
                }

                if explore {
                    allowed[rand_idx % allowed.len()].to_string()
                } else {
                    let ranked = self
                        .brain
                        .ranked_actions_with_meaning(context_key, cfg.meaning_alpha);
                    ranked
                        .into_iter()
                        .find(|(name, _score)| allowed.iter().any(|a| a == name))
                        .map(|(name, _score)| name)
                        .or_else(|| allowed.first().cloned())
                        .unwrap_or_else(|| "tok_UNK".to_string())
                }
            }
            WebGame::Replay(g) => {
                let allowed = g.allowed_actions();
                if allowed.is_empty() {
                    return TickResult::Advanced(None);
                }

                if explore {
                    allowed[rand_idx % allowed.len()].to_string()
                } else {
                    let ranked = self
                        .brain
                        .ranked_actions_with_meaning(context_key, cfg.meaning_alpha);
                    ranked
                        .into_iter()
                        .find(|(name, _score)| allowed.iter().any(|a| a == name))
                        .map(|(name, _score)| name)
                        .or_else(|| allowed.first().cloned())
                        .unwrap_or_else(|| "stay".to_string())
                }
            }
            WebGame::Maze(g) => {
                let allowed = g.allowed_actions();
                if allowed.is_empty() {
                    return TickResult::Advanced(None);
                }

                if explore {
                    allowed[rand_idx % allowed.len()].to_string()
                } else {
                    let ranked = self
                        .brain
                        .ranked_actions_with_meaning(context_key, cfg.meaning_alpha);
                    ranked
                        .into_iter()
                        .find(|(name, _score)| allowed.iter().any(|a| a == name))
                        .map(|(name, _score)| name)
                        .or_else(|| allowed.first().cloned())
                        .unwrap_or_else(|| "up".to_string())
                }
            }
            _ => {
                let allowed = ["left", "right"];
                if explore {
                    allowed[rand_idx % allowed.len()].to_string()
                } else {
                    let ranked = self
                        .brain
                        .ranked_actions_with_meaning(context_key, cfg.meaning_alpha);
                    ranked
                        .into_iter()
                        .find(|(name, _score)| allowed.iter().any(|a| a == name))
                        .map(|(name, _score)| name)
                        .or_else(|| allowed.first().map(|s| s.to_string()))
                        .unwrap_or_else(|| "left".to_string())
                }
            }
        };

        let (reward, _done) = match self.game.score_action(&action, cfg.trial_period_ms) {
            Some((r, done)) => (r, done),
            None => (0.0, false),
        };

        let shaped_reward = ((reward + cfg.reward_bias) * cfg.reward_scale).clamp(-5.0, 5.0);

        self.brain.note_action(&action);
        self.brain
            .note_compound_symbol(&["pair", context_key, action.as_str()]);

        if allow_learning {
            self.brain.set_neuromodulator(shaped_reward);
            self.brain.reinforce_action(&action, shaped_reward);
            self.pending_neuromod = shaped_reward;
            self.brain.commit_observation();
        } else {
            self.brain.set_neuromodulator(0.0);
            self.pending_neuromod = 0.0;
            self.brain.discard_observation();
        }

        TickResult::Advanced(Some(TickOutput {
            last_action: action,
            reward: shaped_reward,
        }))
    }

    pub(super) fn game_ui_snapshot(&self) -> GameUiSnapshot {
        self.game.ui_snapshot()
    }

    fn ensure_spotxy_io(&mut self, k: usize) {
        for i in 0..k {
            self.brain
                .ensure_sensor_min_width(&format!("pos_x_{i:02}"), 3);
            self.brain
                .ensure_sensor_min_width(&format!("pos_y_{i:02}"), 3);
        }
        self.brain.ensure_action_min_width("left", 6);
        self.brain.ensure_action_min_width("right", 6);
    }

    fn ensure_maze_io(&mut self) {
        for name in [
            "maze_wall_up",
            "maze_wall_right",
            "maze_wall_down",
            "maze_wall_left",
            "maze_goal_left",
            "maze_goal_right",
            "maze_goal_up",
            "maze_goal_down",
            "maze_goal_here",
            "maze_dist_b0",
            "maze_dist_b1",
            "maze_dist_b2",
            "maze_dist_b3",
            "maze_mode_easy",
            "maze_mode_medium",
            "maze_mode_hard",
            "maze_bump",
            "maze_reached_goal",
        ] {
            self.brain.ensure_sensor_min_width(name, 2);
        }
        for action in ["up", "right", "down", "left"] {
            self.brain.ensure_action_min_width(action, 6);
        }
    }

    fn ensure_pong_io(&mut self) {
        let bins = 8u32;
        for i in 0..bins {
            self.brain
                .ensure_sensor_min_width(&format!("pong_ball_x_{i:02}"), 3);
            self.brain
                .ensure_sensor_min_width(&format!("pong_ball_y_{i:02}"), 3);
            self.brain
                .ensure_sensor_min_width(&format!("pong_ball2_x_{i:02}"), 3);
            self.brain
                .ensure_sensor_min_width(&format!("pong_ball2_y_{i:02}"), 3);
            self.brain
                .ensure_sensor_min_width(&format!("pong_paddle_y_{i:02}"), 3);
        }
        self.brain.ensure_sensor_min_width("pong_ball_visible", 2);
        self.brain.ensure_sensor_min_width("pong_ball_hidden", 2);
        self.brain.ensure_sensor_min_width("pong_ball2_visible", 2);
        self.brain.ensure_sensor_min_width("pong_ball2_hidden", 2);
        self.brain.ensure_sensor_min_width("pong_vx_pos", 2);
        self.brain.ensure_sensor_min_width("pong_vx_neg", 2);
        self.brain.ensure_sensor_min_width("pong_vy_pos", 2);
        self.brain.ensure_sensor_min_width("pong_vy_neg", 2);

        self.brain.ensure_sensor_min_width("pong_ball2_vx_pos", 2);
        self.brain.ensure_sensor_min_width("pong_ball2_vx_neg", 2);
        self.brain.ensure_sensor_min_width("pong_ball2_vy_pos", 2);
        self.brain.ensure_sensor_min_width("pong_ball2_vy_neg", 2);

        self.brain.ensure_action_min_width("up", 6);
        self.brain.ensure_action_min_width("down", 6);
        self.brain.ensure_action_min_width("stay", 6);
    }

    fn ensure_sequence_io(&mut self) {
        self.brain.ensure_sensor_min_width("seq_token_A", 4);
        self.brain.ensure_sensor_min_width("seq_token_B", 4);
        self.brain.ensure_sensor_min_width("seq_token_C", 4);

        self.brain.ensure_sensor_min_width("seq_regime_0", 3);
        self.brain.ensure_sensor_min_width("seq_regime_1", 3);

        self.brain.ensure_action_min_width("A", 6);
        self.brain.ensure_action_min_width("B", 6);
        self.brain.ensure_action_min_width("C", 6);
    }

    pub(crate) fn ensure_text_io(&mut self, g: &TextWebGame) {
        self.brain.ensure_sensor_min_width("txt_regime_0", 3);
        self.brain.ensure_sensor_min_width("txt_regime_1", 3);
        for name in g.token_sensor_names() {
            self.brain.ensure_sensor_min_width(name, 3);
        }
        for name in g.allowed_actions() {
            self.brain.ensure_action_min_width(name, 6);
        }
    }

    fn ensure_replay_io(&mut self, dataset: &ReplayDataset) {
        use std::collections::BTreeSet;

        let mut sensors: BTreeSet<String> = BTreeSet::new();
        let mut actions: BTreeSet<String> = BTreeSet::new();

        for tr in &dataset.trials {
            for s in &tr.stimuli {
                sensors.insert(s.name.clone());
            }
            for a in &tr.allowed_actions {
                actions.insert(a.clone());
            }
            if !tr.correct_action.trim().is_empty() {
                actions.insert(tr.correct_action.clone());
            }
        }

        for name in sensors {
            self.brain.ensure_sensor_min_width(&name, 3);
        }
        for name in actions {
            self.brain.ensure_action_min_width(&name, 6);
        }
    }

    fn ensure_spotxy_actions(&mut self, g: &SpotXYGame) {
        self.brain.ensure_action_min_width("left", 6);
        self.brain.ensure_action_min_width("right", 6);
        for name in g.allowed_actions() {
            self.brain.ensure_action_min_width(name, 6);
        }
    }

    pub(super) fn spotxy_increase_grid(&mut self) {
        let actions = if let WebGame::SpotXY(g) = &mut self.game {
            let cur = g.grid_n();
            let target = if cur == 0 {
                2
            } else if cur.is_power_of_two() {
                (cur.saturating_mul(2)).min(8)
            } else {
                cur.next_power_of_two().min(8)
            };

            // Use the underlying stepwise API, but jump to the next power-of-two size.
            let mut guard = 0u32;
            while g.grid_n() < target && guard < 16 {
                let before = g.grid_n();
                g.increase_grid();
                if g.grid_n() == before {
                    break;
                }
                guard += 1;
            }

            Some(g.allowed_actions().to_vec())
        } else {
            None
        };

        if let Some(actions) = actions {
            self.brain.ensure_action_min_width("left", 6);
            self.brain.ensure_action_min_width("right", 6);
            for name in actions {
                self.brain.ensure_action_min_width(&name, 6);
            }
        }
    }

    pub(super) fn spotxy_decrease_grid(&mut self) {
        let actions = if let WebGame::SpotXY(g) = &mut self.game {
            let cur = g.grid_n();
            let target = if cur <= 2 {
                0
            } else if cur.is_power_of_two() {
                cur / 2
            } else {
                // Snap down to the closest lower power-of-two.
                1u32 << (31 - cur.leading_zeros())
            };

            let mut guard = 0u32;
            while g.grid_n() > target && guard < 16 {
                let before = g.grid_n();
                g.decrease_grid();
                if g.grid_n() == before {
                    break;
                }
                guard += 1;
            }

            Some(g.allowed_actions().to_vec())
        } else {
            None
        };

        if let Some(actions) = actions {
            self.brain.ensure_action_min_width("left", 6);
            self.brain.ensure_action_min_width("right", 6);
            for name in actions {
                self.brain.ensure_action_min_width(&name, 6);
            }
        }
    }

    pub(super) fn spotxy_set_eval(&mut self, eval: bool) {
        if let WebGame::SpotXY(g) = &mut self.game {
            g.set_eval_mode(eval);
        }
    }

    pub(super) fn pong_set_param(&mut self, key: &str, value: f32) -> Result<(), String> {
        match &mut self.game {
            WebGame::Pong(g) => g.set_param(key, value),
            _ => Err("pong_set_param: not in pong".to_string()),
        }
    }

    pub(super) fn tick(&mut self, cfg: &TickConfig) -> TickResult {
        // If a GPU step is already in flight, do not re-apply stimuli or update timing.
        if let Some(pending) = self.pending_tick.as_ref() {
            // Allow the game clock to advance while we're waiting on the GPU IF this tick was
            // already in the post-response phase. This keeps e.g. Pong animation and trial
            // timeout responsive, while avoiding action/state mismatch.
            if pending.response_made_at_start {
                self.game.update_timing(pending.cfg.trial_period_ms);
            }
            if !self.brain.step_nonblocking() {
                return TickResult::PendingGpu;
            }
            let pending = self.pending_tick.take().expect("pending tick disappeared");
            return self.finish_tick(
                pending.cfg,
                pending.allow_learning,
                pending.context_key,
                pending.response_made_at_start,
            );
        }

        let cfg = *cfg;
        self.game.update_timing(cfg.trial_period_ms);

        let response_made_at_start = self.game.response_made();

        // SpotXY eval mode is a holdout run: no causal/meaning writes.
        let allow_learning = cfg.learning_enabled && !self.game.spotxy_eval_mode();

        // Apply last reward as neuromodulation for one step.
        self.brain.set_neuromodulator(self.pending_neuromod);
        self.pending_neuromod = 0.0;

        let base_stimulus = self.game.stimulus_name();
        let stimulus_key_owned: Option<String> = if self.game.reversal_active() {
            Some(format!("{}::rev", base_stimulus))
        } else {
            self.game.stimulus_key().map(|k| k.to_string())
        };
        let context_key_owned = stimulus_key_owned.unwrap_or_else(|| base_stimulus.to_string());
        let context_key = context_key_owned.as_str();

        // Apply stimuli.
        match &self.game {
            WebGame::Spot(_) | WebGame::Bandit(_) | WebGame::SpotReversal(_) => {
                self.brain.apply_stimulus(Stimulus::new(base_stimulus, 1.0));
                if self.game.reversal_active() {
                    self.brain
                        .apply_stimulus(Stimulus::new("spot_rev_ctx", 1.0));
                }
            }
            WebGame::SpotXY(g) => {
                g.apply_stimuli(&mut self.brain);
            }
            WebGame::Maze(g) => {
                g.apply_stimuli(&mut self.brain);
            }
            WebGame::Pong(g) => {
                g.apply_stimuli(&mut self.brain);
            }
            WebGame::Sequence(g) => {
                g.apply_stimuli(&mut self.brain);
            }
            WebGame::Text(g) => {
                g.apply_stimuli(&mut self.brain);
            }
            WebGame::Replay(g) => {
                g.apply_stimuli(&mut self.brain);
            }
        }

        self.brain.note_compound_symbol(&[context_key]);

        // Non-blocking step on wasm+GPU; immediate step on other tiers.
        if !self.brain.step_nonblocking() {
            self.pending_tick = Some(PendingTick {
                cfg,
                allow_learning,
                context_key: context_key_owned,
                response_made_at_start,
            });
            return TickResult::PendingGpu;
        }

        self.finish_tick(
            cfg,
            allow_learning,
            context_key_owned,
            response_made_at_start,
        )
    }

    pub(crate) fn rng_next_u64(&mut self) -> u64 {
        self.rng_seed = self
            .rng_seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1);
        self.rng_seed
    }

    pub(crate) fn rng_next_f32(&mut self) -> f32 {
        let u = (self.rng_next_u64() >> 40) as u32; // 24 bits
        (u as f32) / ((1u32 << 24) as f32)
    }
}

#[allow(clippy::large_enum_variant)]
pub(super) enum WebGame {
    Spot(SpotGame),
    Bandit(BanditGame),
    SpotReversal(SpotReversalGame),
    SpotXY(SpotXYGame),
    Maze(MazeGame),
    Pong(PongWebGame),
    Sequence(SequenceWebGame),
    Text(TextWebGame),
    Replay(ReplayGame),
}

impl WebGame {
    pub(super) fn allowed_actions_for_ui(&self) -> Vec<String> {
        match self {
            WebGame::Spot(_) | WebGame::Bandit(_) | WebGame::SpotReversal(_) => {
                vec!["left".to_string(), "right".to_string()]
            }
            WebGame::SpotXY(g) => g.allowed_actions().to_vec(),
            WebGame::Maze(g) => g.allowed_actions().to_vec(),
            WebGame::Pong(g) => g.allowed_actions().to_vec(),
            WebGame::Sequence(g) => g.allowed_actions().to_vec(),
            WebGame::Text(g) => g.allowed_actions().to_vec(),
            WebGame::Replay(g) => g.allowed_actions().to_vec(),
        }
    }

    pub(super) fn stimulus_name(&self) -> &'static str {
        match self {
            WebGame::Spot(g) => g.stimulus_name(),
            WebGame::Bandit(g) => g.stimulus_name(),
            WebGame::SpotReversal(g) => g.stimulus_name(),
            WebGame::SpotXY(g) => g.stimulus_name(),
            WebGame::Maze(g) => g.stimulus_name(),
            WebGame::Pong(g) => g.stimulus_name(),
            WebGame::Sequence(g) => g.stimulus_name(),
            WebGame::Text(g) => g.stimulus_name(),
            WebGame::Replay(g) => g.stimulus_name(),
        }
    }

    pub(super) fn stimulus_key(&self) -> Option<&str> {
        match self {
            WebGame::SpotXY(g) => Some(g.stimulus_key()),
            WebGame::Maze(g) => Some(g.stimulus_key()),
            WebGame::Pong(g) => Some(g.stimulus_key()),
            WebGame::Sequence(g) => Some(g.stimulus_key()),
            WebGame::Text(g) => Some(g.stimulus_key()),
            WebGame::Replay(g) => Some(g.stimulus_key()),
            _ => None,
        }
    }

    pub(super) fn reversal_active(&self) -> bool {
        match self {
            WebGame::SpotReversal(g) => g.reversal_active,
            _ => false,
        }
    }

    pub(super) fn spotxy_eval_mode(&self) -> bool {
        match self {
            WebGame::SpotXY(g) => g.eval_mode,
            _ => false,
        }
    }

    pub(super) fn response_made(&self) -> bool {
        match self {
            WebGame::Spot(g) => g.response_made,
            WebGame::Bandit(g) => g.response_made,
            WebGame::SpotReversal(g) => g.response_made,
            WebGame::SpotXY(g) => g.response_made,
            WebGame::Maze(g) => g.response_made,
            WebGame::Pong(g) => g.response_made,
            WebGame::Sequence(g) => g.response_made(),
            WebGame::Text(g) => g.response_made(),
            WebGame::Replay(g) => g.response_made,
        }
    }

    pub(super) fn update_timing(&mut self, trial_period_ms: u32) {
        match self {
            WebGame::Spot(g) => g.update_timing(trial_period_ms),
            WebGame::Bandit(g) => g.update_timing(trial_period_ms),
            WebGame::SpotReversal(g) => g.update_timing(trial_period_ms),
            WebGame::SpotXY(g) => g.update_timing(trial_period_ms),
            WebGame::Maze(g) => g.update_timing(trial_period_ms),
            WebGame::Pong(g) => g.update_timing(trial_period_ms),
            WebGame::Sequence(g) => g.update_timing(trial_period_ms),
            WebGame::Text(g) => g.update_timing(trial_period_ms),
            WebGame::Replay(g) => g.update_timing(trial_period_ms),
        }
    }

    pub(super) fn score_action(
        &mut self,
        action: &str,
        trial_period_ms: u32,
    ) -> Option<(f32, bool)> {
        match self {
            WebGame::Spot(g) => g.score_action(action),
            WebGame::Bandit(g) => g.score_action(action),
            WebGame::SpotReversal(g) => g.score_action(action),
            WebGame::SpotXY(g) => g.score_action(action),
            WebGame::Maze(g) => {
                let _ = trial_period_ms;
                g.score_action(action)
            }
            WebGame::Pong(g) => {
                let _ = trial_period_ms;
                g.score_action(action)
            }
            WebGame::Sequence(g) => {
                let _ = trial_period_ms;
                g.score_action(action)
            }
            WebGame::Text(g) => {
                let _ = trial_period_ms;
                g.score_action(action)
            }
            WebGame::Replay(g) => {
                let _ = trial_period_ms;
                g.score_action(action)
            }
        }
    }

    pub(super) fn stats(&self) -> &braine_games::stats::GameStats {
        match self {
            WebGame::Spot(g) => &g.stats,
            WebGame::Bandit(g) => &g.stats,
            WebGame::SpotReversal(g) => &g.stats,
            WebGame::SpotXY(g) => &g.stats,
            WebGame::Maze(g) => &g.stats,
            WebGame::Pong(g) => &g.stats,
            WebGame::Sequence(g) => &g.game.stats,
            WebGame::Text(g) => &g.game.stats,
            WebGame::Replay(g) => &g.stats,
        }
    }

    pub(super) fn set_stats(&mut self, stats: braine_games::stats::GameStats) {
        match self {
            WebGame::Spot(g) => g.stats = stats,
            WebGame::Bandit(g) => g.stats = stats,
            WebGame::SpotReversal(g) => g.stats = stats,
            WebGame::SpotXY(g) => g.stats = stats,
            WebGame::Maze(g) => g.stats = stats,
            WebGame::Pong(g) => g.stats = stats,
            WebGame::Sequence(g) => g.game.stats = stats,
            WebGame::Text(g) => g.game.stats = stats,
            WebGame::Replay(g) => g.stats = stats,
        }
    }

    pub(super) fn ui_snapshot(&self) -> GameUiSnapshot {
        match self {
            WebGame::Spot(g) => GameUiSnapshot {
                spot_is_left: Some(g.spot_is_left),
                ..GameUiSnapshot::default()
            },
            WebGame::Bandit(_) => GameUiSnapshot::default(),
            WebGame::SpotReversal(g) => GameUiSnapshot {
                spot_is_left: Some(g.spot_is_left),
                reversal_active: g.reversal_active,
                reversal_flip_after_trials: g.flip_after_trials,
                ..GameUiSnapshot::default()
            },
            WebGame::SpotXY(g) => GameUiSnapshot {
                spotxy_pos: Some((g.pos_x, g.pos_y)),
                spotxy_stimulus_key: g.stimulus_key().to_string(),
                spotxy_eval: g.eval_mode,
                spotxy_mode: g.mode_name().to_string(),
                spotxy_grid_n: g.grid_n(),
                ..GameUiSnapshot::default()
            },
            WebGame::Maze(g) => GameUiSnapshot {
                maze_state: Some(MazeUiState {
                    w: g.sim.grid.w(),
                    h: g.sim.grid.h(),
                    seed: g.sim.seed,
                    player_x: g.sim.player_x,
                    player_y: g.sim.player_y,
                    goal_x: g.sim.goal_x,
                    goal_y: g.sim.goal_y,
                    steps: g.steps_in_episode,
                    difficulty: g.difficulty_name(),
                    last_event: g.last_event.as_str(),
                }),
                ..GameUiSnapshot::default()
            },
            WebGame::Pong(g) => GameUiSnapshot {
                pong_state: Some(PongUiState {
                    ball_x: g.sim.state.ball_x,
                    ball_y: g.sim.state.ball_y,
                    paddle_y: g.sim.state.paddle_y,
                    paddle_half_height: g.sim.params.paddle_half_height,
                    ball_visible: g.ball_visible(),
                    ball2_x: g.sim.state.ball2_x,
                    ball2_y: g.sim.state.ball2_y,
                    ball2_visible: g.sim.ball2_visible(),
                    hits: g.hits(),
                    misses: g.misses(),
                }),
                pong_stimulus_key: g.stimulus_key().to_string(),
                pong_paddle_speed: g.sim.params.paddle_speed,
                pong_paddle_half_height: g.sim.params.paddle_half_height,
                pong_ball_speed: g.sim.params.ball_speed,
                pong_paddle_bounce_y: g.sim.params.paddle_bounce_y,
                pong_respawn_delay_s: g.sim.params.respawn_delay_s,
                pong_distractor_enabled: g.sim.params.distractor_enabled,
                pong_distractor_speed_scale: g.sim.params.distractor_speed_scale,
                ..GameUiSnapshot::default()
            },
            WebGame::Sequence(g) => GameUiSnapshot {
                sequence_state: Some(SequenceUiState {
                    regime: g.game.regime(),
                    token: g.game.current_token().label().to_string(),
                    target_next: g.game.correct_action().to_string(),
                    outcomes: g.game.outcomes(),
                    shift_every: g.game.shift_every_outcomes(),
                }),
                ..GameUiSnapshot::default()
            },
            WebGame::Text(g) => GameUiSnapshot {
                text_state: Some(TextUiState {
                    regime: g.game.regime(),
                    token: g.game.current_token().display(),
                    target_next: g.game.target_next_token().display(),
                    outcomes: g.game.outcomes(),
                    shift_every: g.game.shift_every_outcomes(),
                    vocab_size: g.game.vocab_size() as u32,
                }),
                ..GameUiSnapshot::default()
            },
            WebGame::Replay(g) => GameUiSnapshot {
                replay_state: Some(ReplayUiState {
                    dataset: g.dataset_name().to_string(),
                    index: g.index() as u32,
                    total: g.total_trials() as u32,
                    trial_id: g.current_trial_id().to_string(),
                }),
                ..GameUiSnapshot::default()
            },
        }
    }
}
