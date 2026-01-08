use braine::prng::Prng;
use braine::substrate::{Brain, Stimulus};
use std::io;
use std::io::Write;

#[derive(Debug, Clone, Copy)]
pub struct SpotXYConfig {
    /// Number of population-code bumps per axis.
    pub k: usize,
    /// Total training steps.
    pub max_steps: usize,
    /// Print diagnostics every N steps.
    pub render_every: usize,
    /// Window size for recent accuracy.
    pub retire_window: usize,
    /// Self-retire when recent accuracy reaches this.
    pub retire_accuracy: f32,
    /// Bootstrap with the correct action for the first N steps.
    pub bootstrap_steps: usize,
    /// Epsilon exploration after bootstrap.
    pub epsilon: f32,
    /// Sensor group width per bump.
    pub sensor_width: usize,
    /// Action group width.
    pub action_width: usize,
    /// Meaning weight in `select_action_with_meaning`.
    pub meaning_alpha: f32,
}

impl Default for SpotXYConfig {
    fn default() -> Self {
        Self {
            k: 16,
            max_steps: 30_000,
            render_every: 400,
            retire_window: 500,
            retire_accuracy: 0.90,
            bootstrap_steps: 800,
            epsilon: 0.12,
            sensor_width: 3,
            action_width: 6,
            meaning_alpha: 6.0,
        }
    }
}

#[derive(Debug, Clone)]
struct SpotXYState {
    steps: usize,
    correct: usize,
    wrong: usize,
    cumulative_reward: f32,
    rng: Prng,
    recent: Vec<bool>,
}

impl SpotXYState {
    fn new(seed: u64, retire_window: usize) -> Self {
        Self {
            steps: 0,
            correct: 0,
            wrong: 0,
            cumulative_reward: 0.0,
            rng: Prng::new(seed),
            recent: Vec::with_capacity(retire_window.max(1)),
        }
    }

    fn push_outcome(&mut self, ok: bool, window: usize) {
        self.recent.push(ok);
        if self.recent.len() > window {
            self.recent.remove(0);
        }
    }

    fn lifetime_accuracy(&self) -> f32 {
        let total = self.correct + self.wrong;
        if total == 0 {
            0.0
        } else {
            self.correct as f32 / total as f32
        }
    }

    fn recent_accuracy(&self) -> f32 {
        if self.recent.is_empty() {
            0.0
        } else {
            self.recent.iter().filter(|&&b| b).count() as f32 / self.recent.len() as f32
        }
    }
}

/// Minimal spatial experiment:
/// - emit population-coded sensors for (x,y)
/// - require a 2-action decision based only on the sign of x
/// - keep y as a distractor dimension
///
/// Run: `cargo run -- spotxy-demo`
pub fn run_spotxy_demo(brain: &mut Brain, cfg: SpotXYConfig) {
    let k = cfg.k.max(2);

    // Define sensors: factorized population code per axis.
    // Names are stable and deterministic.
    let mut x_names: Vec<String> = Vec::with_capacity(k);
    let mut y_names: Vec<String> = Vec::with_capacity(k);
    for i in 0..k {
        x_names.push(format!("pos_x_{i:02}"));
        y_names.push(format!("pos_y_{i:02}"));
    }
    for n in &x_names {
        brain.ensure_sensor(n, cfg.sensor_width);
    }
    for n in &y_names {
        brain.ensure_sensor(n, cfg.sensor_width);
    }

    // Define action groups.
    brain.ensure_action("left", cfg.action_width);
    brain.ensure_action("right", cfg.action_width);

    // Context key per x-bin (used for meaning conditioning). We deliberately keep
    // the conditioning key low-cardinality and derived from x only.
    let mut xbin_keys: Vec<String> = Vec::with_capacity(k);
    for i in 0..k {
        xbin_keys.push(format!("spotxy_xbin_{i:02}"));
    }

    // Population code centers and sigma.
    let centers = axis_centers(k);
    let sigma = 2.0 / ((k - 1) as f32);

    let mut s = SpotXYState::new(2026, cfg.retire_window);

    if out_line(format_args!(
        "spotxy-demo: k={k} sigma={sigma:.4} max_steps={} bootstrap={} epsilon={:.2} retire_acc={:.2} window={}",
        cfg.max_steps, cfg.bootstrap_steps, cfg.epsilon, cfg.retire_accuracy, cfg.retire_window
    ))
    .is_err()
    {
        return;
    }

    while s.steps < cfg.max_steps {
        s.steps += 1;

        let x = s.rng.gen_range_f32(-1.0, 1.0);
        let y = s.rng.gen_range_f32(-1.0, 1.0);

        let x_act = axis_activations(x, &centers, sigma);
        let y_act = axis_activations(y, &centers, sigma);

        // Apply factorized stimuli.
        for (i, a) in x_act.iter().enumerate() {
            brain.apply_stimulus(Stimulus::new(x_names[i].as_str(), *a));
        }
        for (i, a) in y_act.iter().enumerate() {
            brain.apply_stimulus(Stimulus::new(y_names[i].as_str(), *a));
        }

        brain.step();

        // Correct rule: depends only on x sign.
        let correct_action = if x < 0.0 { "left" } else { "right" };

        // Conditioning stimulus key: x-bin argmax.
        let xbin = argmax(&x_act);
        let stimulus_key = xbin_keys[xbin].as_str();

        // Ensure the symbol exists for meaning conditioning.
        // (Interns the symbol and records it as an observation event.)
        brain.note_compound_symbol(&[stimulus_key]);

        let (action, _score) = if s.steps <= cfg.bootstrap_steps {
            if s.steps == cfg.bootstrap_steps {
                // Reset the rolling window so "recent" reflects autonomous performance.
                s.recent.clear();
                let _ = out_line(format_args!(
                    "switching to autonomous control (bootstrap complete)"
                ));
            }
            (correct_action.to_string(), 0.0)
        } else if s.rng.gen_range_f32(0.0, 1.0) < cfg.epsilon {
            if s.rng.gen_range_f32(0.0, 1.0) < 0.5 {
                ("left".to_string(), 0.0)
            } else {
                ("right".to_string(), 0.0)
            }
        } else {
            brain.select_action_with_meaning(stimulus_key, cfg.meaning_alpha)
        };

        brain.note_action(&action);
        brain.note_compound_symbol(&["pair", stimulus_key, action.as_str()]);

        let ok = action == correct_action;
        let mut reward: f32 = if ok { 0.7 } else { -0.7 };
        reward = reward.clamp(-1.0, 1.0);

        s.cumulative_reward += reward;
        if ok {
            s.correct += 1;
        } else {
            s.wrong += 1;
        }
        s.push_outcome(ok, cfg.retire_window);

        brain.set_neuromodulator(reward);
        if reward > 0.2 {
            brain.reinforce_action(&action, 0.6);
        }
        brain.commit_observation();

        if cfg.render_every > 0 && s.steps.is_multiple_of(cfg.render_every) {
            let life = s.lifetime_accuracy();
            let recent = s.recent_accuracy();
            if out_line(format_args!(
                "age_steps={} acc={:.3} recent={:.3} cumulative_reward={:.1}  x={:+.3} y={:+.3} xbin={:02} key={}",
                s.steps, life, recent, s.cumulative_reward, x, y, xbin, stimulus_key
            ))
            .is_err()
            {
                return;
            }

            let hint = brain.meaning_hint(stimulus_key);
            if out_line(format_args!("  meaning_hint({})={:?}", stimulus_key, hint)).is_err() {
                return;
            }
        }

        if s.steps > cfg.bootstrap_steps && s.recent.len() >= (cfg.retire_window / 2).max(1) {
            let recent = s.recent_accuracy();
            if recent >= cfg.retire_accuracy {
                let _ = out_line(format_args!(
                    "self-retire: recent_acc={:.3} over {} samples (age_steps={})",
                    recent,
                    s.recent.len(),
                    s.steps
                ));
                break;
            }
        }
    }

    let _ = out_line(format_args!("spotxy-demo done:"));
    let _ = out_line(format_args!("  age_steps={}", s.steps));
    let _ = out_line(format_args!(
        "  correct={} wrong={} acc={:.3}",
        s.correct,
        s.wrong,
        s.lifetime_accuracy()
    ));
    let _ = out_line(format_args!(
        "  cumulative_reward={:.1}",
        s.cumulative_reward
    ));
}

fn axis_centers(k: usize) -> Vec<f32> {
    // Evenly spaced in [-1, +1].
    let denom = (k - 1).max(1) as f32;
    (0..k).map(|i| -1.0 + 2.0 * (i as f32) / denom).collect()
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

    // Per-axis normalization to keep total stimulus energy stable.
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

fn out_line(args: std::fmt::Arguments<'_>) -> io::Result<()> {
    let mut out = io::stdout().lock();
    match out.write_fmt(args) {
        Ok(()) => {}
        Err(e) if e.kind() == io::ErrorKind::BrokenPipe => return Err(e),
        Err(e) => return Err(e),
    }
    match out.write_all(b"\n") {
        Ok(()) => Ok(()),
        Err(e) if e.kind() == io::ErrorKind::BrokenPipe => Err(e),
        Err(e) => Err(e),
    }
}
