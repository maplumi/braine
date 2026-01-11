use crate::stats::GameStats;
use crate::time::{Duration, Instant};

#[cfg(feature = "braine")]
use braine::substrate::{Brain, Stimulus};

// ─────────────────────────────────────────────────────────────────────────
// SpotXY: 2D position (population-coded) with a 2-action rule on sign(x).
// Includes an optional eval/holdout mode that samples from a held-out x band.
// ─────────────────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub enum SpotXYMode {
    /// Binary left/right classification from sign(x).
    BinaryX,
    /// N×N grid classification over (x,y).
    Grid { n: u32 },
}

#[derive(Debug)]
pub struct SpotXYGame {
    pub pos_x: f32,
    pub pos_y: f32,

    pub trial_frame: u32,
    pub response_made: bool,
    pub last_action: Option<String>,
    pub stats: GameStats,

    pub eval_mode: bool,

    pub mode: SpotXYMode,

    action_names: Vec<String>,
    correct_action: String,

    k: usize,
    sigma: f32,
    centers: Vec<f32>,
    x_names: Vec<String>,
    y_names: Vec<String>,
    x_act: Vec<f32>,
    y_act: Vec<f32>,
    stimulus_key: String,

    holdout_min_abs_x: f32,
    holdout_max_abs_x: f32,

    rng_seed: u64,
    trial_started_at: Instant,
}

impl SpotXYGame {
    pub fn new(k: usize) -> Self {
        let k = k.max(2);
        let denom = (k - 1) as f32;
        let centers: Vec<f32> = (0..k).map(|i| -1.0 + 2.0 * (i as f32) / denom).collect();
        let sigma = 2.0 / denom;

        let mut x_names: Vec<String> = Vec::with_capacity(k);
        let mut y_names: Vec<String> = Vec::with_capacity(k);
        for i in 0..k {
            x_names.push(format!("pos_x_{i:02}"));
            y_names.push(format!("pos_y_{i:02}"));
        }

        let now = Instant::now();
        let mut g = Self {
            pos_x: 0.0,
            pos_y: 0.0,
            trial_frame: 0,
            response_made: false,
            last_action: None,
            stats: GameStats::new(),
            eval_mode: false,
            mode: SpotXYMode::BinaryX,
            action_names: vec!["left".to_string(), "right".to_string()],
            correct_action: "left".to_string(),
            k,
            sigma,
            centers,
            x_names,
            y_names,
            x_act: vec![0.0; k],
            y_act: vec![0.0; k],
            stimulus_key: String::new(),
            // Held-out band used for evaluation/generalization checks.
            holdout_min_abs_x: 0.25,
            holdout_max_abs_x: 0.45,
            rng_seed: 0x5107_5129u64,
            trial_started_at: now,
        };
        g.new_trial();
        g
    }

    pub fn increase_grid(&mut self) {
        let next = match self.mode {
            SpotXYMode::BinaryX => SpotXYMode::Grid { n: 2 },
            SpotXYMode::Grid { n } => SpotXYMode::Grid { n: (n + 1).min(8) },
        };

        if std::mem::discriminant(&self.mode) == std::mem::discriminant(&next) {
            // Same variant; still might differ in n.
        }

        self.mode = next;
        self.refresh_actions();
        self.stats = GameStats::new();
        self.new_trial();
    }

    pub fn decrease_grid(&mut self) {
        let next = match self.mode {
            SpotXYMode::BinaryX => SpotXYMode::BinaryX,
            SpotXYMode::Grid { n } => {
                if n <= 2 {
                    SpotXYMode::BinaryX
                } else {
                    SpotXYMode::Grid { n: n - 1 }
                }
            }
        };

        self.mode = next;
        self.refresh_actions();
        self.stats = GameStats::new();
        self.new_trial();
    }

    pub fn grid_n(&self) -> u32 {
        match self.mode {
            SpotXYMode::BinaryX => 0,
            SpotXYMode::Grid { n } => n,
        }
    }

    pub fn mode_name(&self) -> &'static str {
        match self.mode {
            SpotXYMode::BinaryX => "binary_x",
            SpotXYMode::Grid { .. } => "grid",
        }
    }

    pub fn allowed_actions(&self) -> &[String] {
        &self.action_names
    }

    pub fn set_eval_mode(&mut self, eval: bool) {
        if self.eval_mode == eval {
            return;
        }
        self.eval_mode = eval;
        // Treat switching as a fresh run so metrics are interpretable.
        self.stats = GameStats::new();
        self.new_trial();
    }

    pub fn stimulus_name(&self) -> &'static str {
        // Base name (not directly used for meaning conditioning).
        "spotxy"
    }

    pub fn stimulus_key(&self) -> &str {
        &self.stimulus_key
    }

    pub fn correct_action(&self) -> &str {
        &self.correct_action
    }

    pub fn update_timing(&mut self, trial_period_ms: u32) {
        let trial_period_ms = trial_period_ms.clamp(10, 60_000);
        let trial_period = Duration::from_millis(trial_period_ms as u64);

        let now = Instant::now();
        let elapsed = now.duration_since(self.trial_started_at);
        if elapsed >= trial_period {
            self.new_trial();
        }

        let now = Instant::now();
        let elapsed = now.duration_since(self.trial_started_at);
        self.trial_frame = elapsed.as_millis().min(u32::MAX as u128) as u32;
    }

    #[cfg(feature = "braine")]
    pub fn apply_stimuli(&self, brain: &mut Brain) {
        for i in 0..self.k {
            brain.apply_stimulus(Stimulus::new(self.x_names[i].as_str(), self.x_act[i]));
            brain.apply_stimulus(Stimulus::new(self.y_names[i].as_str(), self.y_act[i]));
        }
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

        Some((reward, true))
    }

    fn new_trial(&mut self) {
        self.trial_frame = 0;
        self.response_made = false;
        self.last_action = None;
        self.trial_started_at = Instant::now();

        self.pos_x = self.sample_x();
        self.pos_y = self.sample_uniform(-1.0, 1.0);

        self.x_act = axis_activations(self.pos_x, &self.centers, self.sigma);
        self.y_act = axis_activations(self.pos_y, &self.centers, self.sigma);

        match self.mode {
            SpotXYMode::BinaryX => {
                let xbin = argmax(&self.x_act);
                self.stimulus_key = format!("spotxy_xbin_{xbin:02}");
                self.correct_action = if self.pos_x < 0.0 {
                    "left".to_string()
                } else {
                    "right".to_string()
                };
            }
            SpotXYMode::Grid { n } => {
                let n = n.clamp(2, 8);

                let ix = grid_bin(self.pos_x, n);
                let iy = grid_bin(self.pos_y, n);
                self.stimulus_key = format!("spotxy_bin_{n:02}_{ix:02}_{iy:02}");
                self.correct_action = format!("spotxy_cell_{n:02}_{ix:02}_{iy:02}");
            }
        }
    }

    fn refresh_actions(&mut self) {
        self.action_names.clear();
        match self.mode {
            SpotXYMode::BinaryX => {
                self.action_names.push("left".to_string());
                self.action_names.push("right".to_string());
            }
            SpotXYMode::Grid { n } => {
                let n = n.clamp(2, 8);
                let cap = (n as usize) * (n as usize);
                self.action_names.reserve(cap);
                for ix in 0..n {
                    for iy in 0..n {
                        self.action_names
                            .push(format!("spotxy_cell_{n:02}_{ix:02}_{iy:02}"));
                    }
                }
            }
        }
    }

    fn sample_uniform(&mut self, lo: f32, hi: f32) -> f32 {
        let u = self.rng_next_f32();
        lo + (hi - lo) * u
    }

    fn sample_x(&mut self) -> f32 {
        // In eval mode we sample only within the holdout band.
        if self.eval_mode {
            let sign = if (self.rng_next_u32() & 1) == 0 {
                -1.0
            } else {
                1.0
            };
            let u = self.rng_next_f32();
            let mag =
                self.holdout_min_abs_x + (self.holdout_max_abs_x - self.holdout_min_abs_x) * u;
            return (sign * mag).clamp(-1.0, 1.0);
        }

        // Training mode: sample from [-1,1] excluding the holdout band.
        loop {
            let x = self.sample_uniform(-1.0, 1.0);
            let ax = x.abs();
            if ax < self.holdout_min_abs_x || ax > self.holdout_max_abs_x {
                return x;
            }
        }
    }

    fn rng_next_u32(&mut self) -> u32 {
        self.rng_seed = self
            .rng_seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1);
        (self.rng_seed >> 11) as u32
    }

    fn rng_next_f32(&mut self) -> f32 {
        let u = self.rng_next_u32();
        let mantissa = u >> 8; // 24 bits
        (mantissa as f32) / ((1u32 << 24) as f32)
    }
}

impl Default for SpotXYGame {
    fn default() -> Self {
        // Keep consistent with the daemon's typical default.
        Self::new(16)
    }
}

fn axis_activations(v: f32, centers: &[f32], sigma: f32) -> Vec<f32> {
    let v = v.clamp(-1.0, 1.0);
    let inv_2s2 = 1.0 / (2.0 * sigma * sigma + 1e-9);

    let mut out: Vec<f32> = Vec::with_capacity(centers.len());
    let mut sum = 0.0f32;
    for &c in centers {
        let d = v - c;
        let a = (-d * d * inv_2s2).exp();
        out.push(a);
        sum += a;
    }

    // Per-axis normalization to keep stimulus energy stable.
    if sum > 1e-9 {
        let inv = 1.0 / sum;
        for a in &mut out {
            *a = (*a * inv).clamp(0.0, 1.0);
        }
    }

    out
}

fn argmax(v: &[f32]) -> usize {
    let mut best_i = 0usize;
    let mut best = f32::NEG_INFINITY;
    for (i, &x) in v.iter().enumerate() {
        if x > best {
            best = x;
            best_i = i;
        }
    }
    best_i
}

fn grid_bin(v: f32, n: u32) -> u32 {
    let n = n.max(2);
    // Map [-1,1] to [0,1], then bucket into n bins.
    let t = ((v.clamp(-1.0, 1.0) + 1.0) * 0.5).clamp(0.0, 0.999_999);
    let b = (t * n as f32).floor() as u32;
    b.min(n - 1)
}
