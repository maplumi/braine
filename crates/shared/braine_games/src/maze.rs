use crate::stats::GameStats;
use crate::time::{Duration, Instant};

#[cfg(feature = "braine")]
use braine::substrate::{Brain, Stimulus};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MazeDifficulty {
    Easy,
    Medium,
    Hard,
}

impl MazeDifficulty {
    pub fn name(self) -> &'static str {
        match self {
            MazeDifficulty::Easy => "easy",
            MazeDifficulty::Medium => "medium",
            MazeDifficulty::Hard => "hard",
        }
    }

    pub fn from_param(v: f32) -> Self {
        match v.round().clamp(0.0, 2.0) as u32 {
            0 => MazeDifficulty::Easy,
            1 => MazeDifficulty::Medium,
            _ => MazeDifficulty::Hard,
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub enum MazeAction {
    Up,
    Right,
    Down,
    Left,
}

impl MazeAction {
    pub fn from_action_str(action: &str) -> Option<Self> {
        match action {
            "up" => Some(MazeAction::Up),
            "right" => Some(MazeAction::Right),
            "down" => Some(MazeAction::Down),
            "left" => Some(MazeAction::Left),
            _ => None,
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            MazeAction::Up => "up",
            MazeAction::Right => "right",
            MazeAction::Down => "down",
            MazeAction::Left => "left",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MazeEvent {
    None,
    Moved,
    Bump,
    ReachedGoal,
    Timeout,
}

impl MazeEvent {
    pub fn as_str(self) -> &'static str {
        match self {
            MazeEvent::None => "none",
            MazeEvent::Moved => "moved",
            MazeEvent::Bump => "bump",
            MazeEvent::ReachedGoal => "reached_goal",
            MazeEvent::Timeout => "timeout",
        }
    }
}

// Wall bits per cell.
// 1=up, 2=right, 4=down, 8=left.
const W_UP: u8 = 1;
const W_RIGHT: u8 = 2;
const W_DOWN: u8 = 4;
const W_LEFT: u8 = 8;

#[derive(Debug, Clone)]
pub struct MazeGrid {
    w: u32,
    h: u32,
    cells: Vec<u8>,
}

impl MazeGrid {
    pub fn new(w: u32, h: u32) -> Self {
        let w = w.max(2);
        let h = h.max(2);
        let cells = vec![W_UP | W_RIGHT | W_DOWN | W_LEFT; (w as usize) * (h as usize)];
        Self { w, h, cells }
    }

    pub fn w(&self) -> u32 {
        self.w
    }

    pub fn h(&self) -> u32 {
        self.h
    }

    pub fn walls(&self, x: u32, y: u32) -> u8 {
        // Defensive: never allow a bad (x,y) to take down the whole runtime.
        // Treat out-of-bounds as fully walled.
        if x >= self.w || y >= self.h {
            return W_UP | W_RIGHT | W_DOWN | W_LEFT;
        }
        self.cells[self.idx(x, y)]
    }

    pub fn has_wall(&self, x: u32, y: u32, wall_bit: u8) -> bool {
        self.walls(x, y) & wall_bit != 0
    }

    fn idx(&self, x: u32, y: u32) -> usize {
        (y as usize) * (self.w as usize) + (x as usize)
    }

    fn carve_between(&mut self, x: u32, y: u32, nx: u32, ny: u32) {
        let (a, b) = (self.idx(x, y), self.idx(nx, ny));
        if nx == x && ny + 1 == y {
            // neighbor is up
            self.cells[a] &= !W_UP;
            self.cells[b] &= !W_DOWN;
        } else if nx == x + 1 && ny == y {
            // neighbor is right
            self.cells[a] &= !W_RIGHT;
            self.cells[b] &= !W_LEFT;
        } else if nx == x && ny == y + 1 {
            // neighbor is down
            self.cells[a] &= !W_DOWN;
            self.cells[b] &= !W_UP;
        } else if nx + 1 == x && ny == y {
            // neighbor is left
            self.cells[a] &= !W_LEFT;
            self.cells[b] &= !W_RIGHT;
        }
    }
}

#[derive(Debug, Clone)]
pub struct MazeSim {
    pub grid: MazeGrid,
    pub player_x: u32,
    pub player_y: u32,
    pub goal_x: u32,
    pub goal_y: u32,
    pub seed: u64,
}

impl MazeSim {
    pub fn new(seed: u64, difficulty: MazeDifficulty) -> Self {
        let (w, h) = maze_dims(difficulty);
        Self::new_with_dims(seed, w, h)
    }

    pub fn new_with_dims(seed: u64, w: u32, h: u32) -> Self {
        let mut sim = Self {
            grid: MazeGrid::new(w, h),
            player_x: 0,
            player_y: 0,
            goal_x: w.saturating_sub(1),
            goal_y: h.saturating_sub(1),
            seed,
        };
        sim.regenerate();
        sim
    }

    pub fn regenerate(&mut self) {
        self.grid = MazeGrid::new(self.grid.w(), self.grid.h());
        carve_maze(&mut self.grid, self.seed);
        self.player_x = 0;
        self.player_y = 0;
        self.goal_x = self.grid.w().saturating_sub(1);
        self.goal_y = self.grid.h().saturating_sub(1);
    }

    pub fn manhattan_to_goal(&self) -> u32 {
        self.player_x.abs_diff(self.goal_x) + self.player_y.abs_diff(self.goal_y)
    }

    pub fn try_step(&mut self, action: MazeAction) -> MazeEvent {
        // Keep invariants stable even if some caller ends up with a stale position
        // after a resize/migration.
        if self.player_x >= self.grid.w() || self.player_y >= self.grid.h() {
            self.player_x = 0;
            self.player_y = 0;
        }
        let (x, y) = (self.player_x, self.player_y);
        let walls = self.grid.walls(x, y);

        let (nx, ny, blocked) = match action {
            MazeAction::Up => {
                let blocked = walls & W_UP != 0 || y == 0;
                (x, y.saturating_sub(1), blocked)
            }
            MazeAction::Right => {
                let blocked = walls & W_RIGHT != 0 || x + 1 >= self.grid.w();
                (x.saturating_add(1), y, blocked)
            }
            MazeAction::Down => {
                let blocked = walls & W_DOWN != 0 || y + 1 >= self.grid.h();
                (x, y.saturating_add(1), blocked)
            }
            MazeAction::Left => {
                let blocked = walls & W_LEFT != 0 || x == 0;
                (x.saturating_sub(1), y, blocked)
            }
        };

        if blocked {
            return MazeEvent::Bump;
        }

        self.player_x = nx;
        self.player_y = ny;

        if self.player_x == self.goal_x && self.player_y == self.goal_y {
            MazeEvent::ReachedGoal
        } else {
            MazeEvent::Moved
        }
    }
}

#[derive(Debug)]
pub struct MazeGame {
    pub difficulty: MazeDifficulty,
    pub sim: MazeSim,

    pub trial_frame: u32,
    pub response_made: bool,
    pub last_action: Option<String>,
    pub last_event: MazeEvent,
    pub stats: GameStats,

    pub steps_in_episode: u32,

    /// How many episodes to run on the same maze layout before reseeding.
    ///
    /// Keeping the maze stable for a short curriculum window reduces
    /// non-stationarity and helps learning converge without manual mid-run
    /// config changes.
    episodes_per_maze: u32,
    episode_idx: u32,

    action_names: Vec<String>,
    stimulus_key: String,
    visit_counts: Vec<u16>,
    trial_started_at: Instant,
}

impl MazeGame {
    pub fn new() -> Self {
        Self::new_with_difficulty(MazeDifficulty::Easy)
    }

    pub fn new_with_difficulty(difficulty: MazeDifficulty) -> Self {
        let now = Instant::now();
        let seed = 0x4D41_5A45u64; // "MAZE"

        let sim = MazeSim::new(seed, difficulty);
        let visit_counts = vec![0u16; (sim.grid.w() as usize) * (sim.grid.h() as usize)];

        let mut g = Self {
            difficulty,
            sim,
            trial_frame: 0,
            response_made: false,
            last_action: None,
            last_event: MazeEvent::None,
            stats: GameStats::new(),
            steps_in_episode: 0,
            episodes_per_maze: 8,
            episode_idx: 0,
            action_names: vec![
                "up".to_string(),
                "right".to_string(),
                "down".to_string(),
                "left".to_string(),
            ],
            stimulus_key: String::new(),
            visit_counts,
            trial_started_at: now,
        };
        g.refresh_stimulus_key();
        g
    }

    pub fn episodes_per_maze(&self) -> u32 {
        self.episodes_per_maze
    }

    pub fn set_episodes_per_maze(&mut self, v: u32) {
        self.episodes_per_maze = v.clamp(1, 1_000);
    }

    pub fn stimulus_name(&self) -> &'static str {
        "maze"
    }

    pub fn stimulus_key(&self) -> &str {
        &self.stimulus_key
    }

    pub fn allowed_actions(&self) -> &[String] {
        &self.action_names
    }

    pub fn difficulty_name(&self) -> &'static str {
        self.difficulty.name()
    }

    pub fn set_difficulty(&mut self, difficulty: MazeDifficulty) {
        if self.difficulty == difficulty {
            return;
        }
        self.difficulty = difficulty;
        let seed = self.sim.seed;
        self.sim = MazeSim::new(seed, difficulty);
        self.visit_counts = vec![0u16; (self.sim.grid.w() as usize) * (self.sim.grid.h() as usize)];
        self.stats = GameStats::new();
        self.steps_in_episode = 0;
        self.last_event = MazeEvent::None;
        self.response_made = false;
        self.last_action = None;
        self.trial_started_at = Instant::now();
        self.refresh_stimulus_key();
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
            self.last_event = MazeEvent::None;
        }

        let now = Instant::now();
        let elapsed = now.duration_since(self.trial_started_at);
        self.trial_frame = elapsed.as_millis().min(u32::MAX as u128) as u32;
        self.refresh_stimulus_key();
    }

    #[cfg(feature = "braine")]
    pub fn apply_stimuli(&self, brain: &mut Brain) {
        // Clamp for safety (WASM panics are fatal). If the position is ever stale,
        // keep the game running and allow recovery via normal stepping.
        let w = self.sim.grid.w().max(1);
        let h = self.sim.grid.h().max(1);
        let px = self.sim.player_x.min(w.saturating_sub(1));
        let py = self.sim.player_y.min(h.saturating_sub(1));

        let walls = self.sim.grid.walls(px, py);

        brain.apply_stimulus(Stimulus::new(
            "maze_wall_up",
            if walls & W_UP != 0 { 1.0 } else { 0.0 },
        ));
        brain.apply_stimulus(Stimulus::new(
            "maze_wall_right",
            if walls & W_RIGHT != 0 { 1.0 } else { 0.0 },
        ));
        brain.apply_stimulus(Stimulus::new(
            "maze_wall_down",
            if walls & W_DOWN != 0 { 1.0 } else { 0.0 },
        ));
        brain.apply_stimulus(Stimulus::new(
            "maze_wall_left",
            if walls & W_LEFT != 0 { 1.0 } else { 0.0 },
        ));

        let dx = self.sim.goal_x as i32 - px as i32;
        let dy = self.sim.goal_y as i32 - py as i32;

        brain.apply_stimulus(Stimulus::new(
            "maze_goal_left",
            if dx < 0 { 1.0 } else { 0.0 },
        ));
        brain.apply_stimulus(Stimulus::new(
            "maze_goal_right",
            if dx > 0 { 1.0 } else { 0.0 },
        ));
        brain.apply_stimulus(Stimulus::new(
            "maze_goal_up",
            if dy < 0 { 1.0 } else { 0.0 },
        ));
        brain.apply_stimulus(Stimulus::new(
            "maze_goal_down",
            if dy > 0 { 1.0 } else { 0.0 },
        ));
        brain.apply_stimulus(Stimulus::new(
            "maze_goal_here",
            if dx == 0 && dy == 0 { 1.0 } else { 0.0 },
        ));

        // Distance buckets (0..=3).
        let dist = self.sim.manhattan_to_goal();
        let denom = (self.sim.grid.w() + self.sim.grid.h()).max(1);
        let dist01 = (dist as f32) / (denom as f32);
        let bucket = (dist01 * 4.0).floor().clamp(0.0, 3.0) as u32;
        for b in 0..4 {
            let name = match b {
                0 => "maze_dist_b0",
                1 => "maze_dist_b1",
                2 => "maze_dist_b2",
                _ => "maze_dist_b3",
            };
            brain.apply_stimulus(Stimulus::new(name, if b == bucket { 1.0 } else { 0.0 }));
        }

        brain.apply_stimulus(Stimulus::new(
            "maze_mode_easy",
            if self.difficulty == MazeDifficulty::Easy {
                1.0
            } else {
                0.0
            },
        ));
        brain.apply_stimulus(Stimulus::new(
            "maze_mode_medium",
            if self.difficulty == MazeDifficulty::Medium {
                1.0
            } else {
                0.0
            },
        ));
        brain.apply_stimulus(Stimulus::new(
            "maze_mode_hard",
            if self.difficulty == MazeDifficulty::Hard {
                1.0
            } else {
                0.0
            },
        ));

        brain.apply_stimulus(Stimulus::new(
            "maze_bump",
            if self.last_event == MazeEvent::Bump {
                1.0
            } else {
                0.0
            },
        ));
        brain.apply_stimulus(Stimulus::new(
            "maze_reached_goal",
            if self.last_event == MazeEvent::ReachedGoal {
                1.0
            } else {
                0.0
            },
        ));

        brain.apply_stimulus(Stimulus::new(
            "maze_moved",
            if self.last_event == MazeEvent::Moved {
                1.0
            } else {
                0.0
            },
        ));
        brain.apply_stimulus(Stimulus::new(
            "maze_timeout",
            if self.last_event == MazeEvent::Timeout {
                1.0
            } else {
                0.0
            },
        ));

        // Simple short-term memory to break perceptual aliasing.
        let last = self
            .last_action
            .as_deref()
            .and_then(MazeAction::from_action_str);
        brain.apply_stimulus(Stimulus::new(
            "maze_last_action_none",
            if last.is_none() { 1.0 } else { 0.0 },
        ));
        brain.apply_stimulus(Stimulus::new(
            "maze_last_action_up",
            if matches!(last, Some(MazeAction::Up)) {
                1.0
            } else {
                0.0
            },
        ));
        brain.apply_stimulus(Stimulus::new(
            "maze_last_action_right",
            if matches!(last, Some(MazeAction::Right)) {
                1.0
            } else {
                0.0
            },
        ));
        brain.apply_stimulus(Stimulus::new(
            "maze_last_action_down",
            if matches!(last, Some(MazeAction::Down)) {
                1.0
            } else {
                0.0
            },
        ));
        brain.apply_stimulus(Stimulus::new(
            "maze_last_action_left",
            if matches!(last, Some(MazeAction::Left)) {
                1.0
            } else {
                0.0
            },
        ));

        // Visitation bucket at the current cell (0, 1, 2+).
        let idx = (py as usize) * (self.sim.grid.w() as usize) + (px as usize);
        let visits = self.visit_counts.get(idx).copied().unwrap_or(0);
        let vb = if visits == 0 {
            0
        } else if visits == 1 {
            1
        } else {
            2
        };
        brain.apply_stimulus(Stimulus::new(
            "maze_visit_b0",
            if vb == 0 { 1.0 } else { 0.0 },
        ));
        brain.apply_stimulus(Stimulus::new(
            "maze_visit_b1",
            if vb == 1 { 1.0 } else { 0.0 },
        ));
        brain.apply_stimulus(Stimulus::new(
            "maze_visit_b2",
            if vb == 2 { 1.0 } else { 0.0 },
        ));
    }

    pub fn score_action(&mut self, action: &str) -> Option<(f32, bool)> {
        if self.response_made {
            return None;
        }

        // Sanity: never trust persisted/externally-driven state to be in-bounds.
        if self.sim.player_x >= self.sim.grid.w() || self.sim.player_y >= self.sim.grid.h() {
            self.sim.player_x = 0;
            self.sim.player_y = 0;
        }

        let act = MazeAction::from_action_str(action)?;

        let dist_before = self.sim.manhattan_to_goal() as i32;
        let prev_idx = (self.sim.player_y as usize) * (self.sim.grid.w() as usize)
            + (self.sim.player_x as usize);

        let event = self.sim.try_step(act);
        self.last_event = event;
        self.steps_in_episode = self.steps_in_episode.saturating_add(1);

        let idx = (self.sim.player_y as usize) * (self.sim.grid.w() as usize)
            + (self.sim.player_x as usize);
        if let Some(v) = self.visit_counts.get_mut(idx) {
            *v = v.saturating_add(1);
        }

        let dist_after = self.sim.manhattan_to_goal() as i32;
        let delta = dist_before - dist_after;

        let mut reward = 0.0;

        // Step cost to encourage efficiency.
        reward += match self.difficulty {
            MazeDifficulty::Easy => -0.01,
            MazeDifficulty::Medium => -0.02,
            MazeDifficulty::Hard => -0.03,
        };

        if event == MazeEvent::Bump {
            reward += match self.difficulty {
                MazeDifficulty::Easy => -0.05,
                MazeDifficulty::Medium => -0.08,
                MazeDifficulty::Hard => -0.10,
            };
        }

        // Distance shaping (disabled in hard).
        match self.difficulty {
            MazeDifficulty::Easy => {
                reward += 0.02 * (delta as f32);
            }
            MazeDifficulty::Medium => {
                reward += 0.01 * (delta as f32);
            }
            MazeDifficulty::Hard => {}
        }

        // Mild anti-loop penalty.
        if idx == prev_idx {
            // Only possible if bump or zero movement.
        } else {
            let visits = self.visit_counts[idx] as f32;
            if visits > 1.0 {
                reward += match self.difficulty {
                    MazeDifficulty::Easy => -0.005,
                    MazeDifficulty::Medium => -0.01,
                    MazeDifficulty::Hard => -0.0,
                };
            }
        }

        let reached = event == MazeEvent::ReachedGoal;
        if reached {
            reward += 1.0;
        }

        let max_steps = (self.sim.grid.w() * self.sim.grid.h() * 4).max(8);
        let timed_out = self.steps_in_episode >= max_steps;

        if timed_out && !reached {
            self.last_event = MazeEvent::Timeout;
            reward += match self.difficulty {
                MazeDifficulty::Easy => -0.5,
                MazeDifficulty::Medium => -0.75,
                MazeDifficulty::Hard => -1.0,
            };
        }

        self.response_made = true;
        self.last_action = Some(action.to_string());

        if reached {
            self.stats.record_trial(true);
            self.reset_episode(true);
        } else if timed_out {
            self.stats.record_trial(false);
            self.reset_episode(false);
        }

        Some((reward.clamp(-5.0, 5.0), reached || timed_out))
    }

    fn reset_episode(&mut self, _success: bool) {
        self.steps_in_episode = 0;
        self.episode_idx = self.episode_idx.wrapping_add(1);

        // Curriculum window: keep the same maze for a handful of episodes.
        // Always regenerate on success to avoid overfitting to a single maze.
        // NOTE: Some pinned toolchains treat `u32::is_multiple_of` as unavailable.
        // Keep this portable by using `%` and silencing the corresponding clippy lint.
        #[allow(clippy::manual_is_multiple_of)]
        let should_regen = _success
            || self.episodes_per_maze <= 1
            || (self.episode_idx % self.episodes_per_maze) == 0;

        if should_regen {
            self.sim.seed = self.sim.seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
            self.sim.regenerate();
        } else {
            self.sim.player_x = 0;
            self.sim.player_y = 0;
            self.sim.goal_x = self.sim.grid.w().saturating_sub(1);
            self.sim.goal_y = self.sim.grid.h().saturating_sub(1);
        }
        self.visit_counts.fill(0);
        self.refresh_stimulus_key();
    }

    fn refresh_stimulus_key(&mut self) {
        let walls = self.sim.grid.walls(self.sim.player_x, self.sim.player_y) & 0x0F;
        let dx = (self.sim.goal_x as i32 - self.sim.player_x as i32).signum();
        let dy = (self.sim.goal_y as i32 - self.sim.player_y as i32).signum();
        let dx_tag = if dx < 0 {
            'L'
        } else if dx > 0 {
            'R'
        } else {
            '0'
        };
        let dy_tag = if dy < 0 {
            'U'
        } else if dy > 0 {
            'D'
        } else {
            '0'
        };

        let dist = self.sim.manhattan_to_goal();
        let denom = (self.sim.grid.w() + self.sim.grid.h()).max(1);
        let dist01 = (dist as f32) / (denom as f32);
        let bucket = (dist01 * 4.0).floor().clamp(0.0, 3.0) as u32;

        let idx = (self.sim.player_y as usize) * (self.sim.grid.w() as usize)
            + (self.sim.player_x as usize);
        let visits = self.visit_counts.get(idx).copied().unwrap_or(0);
        let vb = if visits == 0 {
            0
        } else if visits == 1 {
            1
        } else {
            2
        };

        let last = self
            .last_action
            .as_deref()
            .and_then(MazeAction::from_action_str)
            .map(|a| match a {
                MazeAction::Up => 'U',
                MazeAction::Right => 'R',
                MazeAction::Down => 'D',
                MazeAction::Left => 'L',
            })
            .unwrap_or('0');

        self.stimulus_key = format!(
            "maze_{}_w{:01x}_{}{}_b{}_v{}_a{}",
            self.difficulty.name(),
            walls,
            dx_tag,
            dy_tag,
            bucket,
            vb,
            last
        );
    }
}

impl Default for MazeGame {
    fn default() -> Self {
        Self::new()
    }
}

fn maze_dims(d: MazeDifficulty) -> (u32, u32) {
    match d {
        MazeDifficulty::Easy => (7, 7),
        MazeDifficulty::Medium => (11, 11),
        MazeDifficulty::Hard => (17, 17),
    }
}

fn carve_maze(grid: &mut MazeGrid, seed: u64) {
    let w = grid.w();
    let h = grid.h();

    let mut rng = Lcg64::new(seed ^ 0xA5A5_5A5Au64);
    let mut visited = vec![false; (w as usize) * (h as usize)];

    let mut stack: Vec<(u32, u32)> = Vec::new();
    stack.push((0, 0));
    visited[0] = true;

    while let Some(&(x, y)) = stack.last() {
        let mut neighbors: [(u32, u32); 4] = [(x, y), (x, y), (x, y), (x, y)];
        let mut n = 0usize;

        if y > 0 {
            let nx = x;
            let ny = y - 1;
            if !visited[(ny as usize) * (w as usize) + (nx as usize)] {
                neighbors[n] = (nx, ny);
                n += 1;
            }
        }
        if x + 1 < w {
            let nx = x + 1;
            let ny = y;
            if !visited[(ny as usize) * (w as usize) + (nx as usize)] {
                neighbors[n] = (nx, ny);
                n += 1;
            }
        }
        if y + 1 < h {
            let nx = x;
            let ny = y + 1;
            if !visited[(ny as usize) * (w as usize) + (nx as usize)] {
                neighbors[n] = (nx, ny);
                n += 1;
            }
        }
        if x > 0 {
            let nx = x - 1;
            let ny = y;
            if !visited[(ny as usize) * (w as usize) + (nx as usize)] {
                neighbors[n] = (nx, ny);
                n += 1;
            }
        }

        if n == 0 {
            stack.pop();
            continue;
        }

        let pick = (rng.next_u32() as usize) % n;
        let (nx, ny) = neighbors[pick];

        grid.carve_between(x, y, nx, ny);
        visited[(ny as usize) * (w as usize) + (nx as usize)] = true;
        stack.push((nx, ny));
    }
}

#[derive(Debug, Clone, Copy)]
struct Lcg64 {
    state: u64,
}

impl Lcg64 {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_mul(6364136223846793005).wrapping_add(1);
        self.state
    }

    fn next_u32(&mut self) -> u32 {
        (self.next_u64() >> 32) as u32
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn maze_regenerates_deterministically() {
        let mut a = MazeSim::new_with_dims(123, 7, 7);
        let b = MazeSim::new_with_dims(123, 7, 7);
        assert_eq!(a.grid.cells, b.grid.cells);

        a.seed = 124;
        a.regenerate();
        assert_ne!(a.grid.cells, b.grid.cells);
    }

    #[test]
    fn maze_has_openings() {
        let sim = MazeSim::new_with_dims(42, 9, 9);
        // Ensure at least one cell has an opening.
        let any_open = sim.grid.cells.iter().any(|c| {
            (c & (W_UP | W_RIGHT | W_DOWN | W_LEFT)) != (W_UP | W_RIGHT | W_DOWN | W_LEFT)
        });
        assert!(any_open);
    }

    #[test]
    fn maze_can_hold_layout_for_multiple_episodes() {
        let mut g = MazeGame::new_with_difficulty(MazeDifficulty::Easy);
        g.set_episodes_per_maze(8);

        let cells0 = g.sim.grid.cells.clone();

        // Timeout-based reset should usually keep the same layout within the window.
        g.reset_episode(false);
        assert_eq!(g.sim.grid.cells, cells0);

        // Success should regenerate immediately (avoid overfitting one maze).
        g.reset_episode(true);
        assert_ne!(g.sim.grid.cells, cells0);
    }
}
