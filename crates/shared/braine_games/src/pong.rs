//! Shared Pong simulation.
//!
//! Coordinate system (matches daemon UI assumptions):
//! - `ball_x` in `[0, 1]` (paddle at `x = 0`)
//! - `ball_y` in `[-1, 1]`
//! - `paddle_y` in `[-1, 1]`
//!
//! This module is intentionally `no_std` friendly (no `Instant`, no `Vec`, no `String`).

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PongAction {
    Up,
    Down,
    Stay,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PongEvent {
    None,
    Hit,
    Miss,
}

#[derive(Debug, Clone, Copy)]
pub struct PongParams {
    pub paddle_speed: f32,
    pub paddle_half_height: f32,
    pub ball_speed: f32,
    /// How strongly paddle contact offset influences outgoing vertical velocity.
    /// Higher values create steeper bounce angles.
    pub paddle_bounce_y: f32,
    /// Ball hidden duration after a miss (seconds).
    pub respawn_delay_s: f32,

    /// Optional distractor ball (second ball). This ball is simulated but does not
    /// contribute to reward.
    pub distractor_enabled: bool,
    /// Speed scale applied to the distractor ball relative to `ball_speed`.
    pub distractor_speed_scale: f32,
}

impl Default for PongParams {
    fn default() -> Self {
        Self {
            paddle_speed: 1.3,
            paddle_half_height: 0.15,
            ball_speed: 1.0,
            paddle_bounce_y: 0.9,
            respawn_delay_s: 0.18,

            distractor_enabled: false,
            distractor_speed_scale: 1.0,
        }
    }
}

/// Pong simulation state (physics + positions).
#[derive(Debug, Clone, Copy)]
pub struct PongState {
    pub ball_x: f32,
    pub ball_y: f32,
    pub ball_vx: f32,
    pub ball_vy: f32,

    /// Optional second ball (distractor). Only used when `params.distractor_enabled`.
    pub ball2_x: f32,
    pub ball2_y: f32,
    pub ball2_vx: f32,
    pub ball2_vy: f32,

    pub paddle_y: f32,
}

impl Default for PongState {
    fn default() -> Self {
        Self {
            ball_x: 0.5,
            ball_y: 0.0,
            ball_vx: 0.75,
            ball_vy: 0.15,

            ball2_x: 1.0,
            ball2_y: 0.0,
            ball2_vx: -0.75,
            ball2_vy: -0.15,

            paddle_y: 0.0,
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct PongSim {
    pub state: PongState,
    pub params: PongParams,
    rng_seed: u64,
    respawn_remaining_s: f32,
    respawn2_remaining_s: f32,
    pending_event_reward: f32,
    distractor_active: bool,
}

impl PongSim {
    pub fn new(seed: u64) -> Self {
        let mut sim = Self {
            state: PongState::default(),
            params: PongParams::default(),
            rng_seed: seed,
            respawn_remaining_s: 0.0,
            respawn2_remaining_s: 0.0,
            pending_event_reward: 0.0,
            distractor_active: false,
        };
        sim.reset_point();
        sim
    }

    pub fn ball_visible(&self) -> bool {
        self.respawn_remaining_s <= 0.0
    }

    pub fn ball2_visible(&self) -> bool {
        self.distractor_active && self.respawn2_remaining_s <= 0.0
    }

    pub fn distractor_enabled(&self) -> bool {
        self.distractor_active
    }

    pub fn pending_event_reward(&self) -> f32 {
        self.pending_event_reward
    }

    pub fn take_pending_event_reward(&mut self) -> f32 {
        let r = self.pending_event_reward;
        self.pending_event_reward = 0.0;
        r
    }

    /// Reset into a fresh serve state.
    pub fn reset_point(&mut self) {
        self.sync_distractor_active();
        self.reset_primary_for_serve();
        if self.distractor_active {
            self.reset_distractor_for_serve();
        }
        self.state.paddle_y = 0.0;
        self.respawn_remaining_s = 0.0;
        self.respawn2_remaining_s = 0.0;
    }

    /// Advance simulation by `dt` seconds.
    ///
    /// When a miss occurs, the next serve is chosen immediately but the ball remains hidden
    /// for `respawn_delay_s` seconds.
    pub fn update(&mut self, dt: f32) -> PongEvent {
        let dt = dt.clamp(0.0, 0.05);
        if dt <= 0.0 {
            return PongEvent::None;
        }

        self.sync_distractor_active();

        let ev = self.update_primary(dt);
        if self.distractor_active {
            self.update_distractor(dt);
        }
        ev
    }

    pub fn apply_action(&mut self, action: PongAction, dt: f32) {
        self.sync_distractor_active();
        let dt = dt.clamp(0.0, 60.0);
        match action {
            PongAction::Up => self.state.paddle_y += self.params.paddle_speed * dt,
            PongAction::Down => self.state.paddle_y -= self.params.paddle_speed * dt,
            PongAction::Stay => {}
        }
        self.state.paddle_y = self.state.paddle_y.clamp(-1.0, 1.0);
    }

    fn update_primary(&mut self, dt: f32) -> PongEvent {
        if self.respawn_remaining_s > 0.0 {
            if dt < self.respawn_remaining_s {
                self.respawn_remaining_s -= dt;
                return PongEvent::None;
            }

            let leftover = dt - self.respawn_remaining_s;
            self.respawn_remaining_s = 0.0;
            if leftover <= 0.0 {
                return PongEvent::None;
            }
            return self.step_physics_primary(leftover);
        }

        self.step_physics_primary(dt)
    }

    fn update_distractor(&mut self, dt: f32) {
        if self.respawn2_remaining_s > 0.0 {
            if dt < self.respawn2_remaining_s {
                self.respawn2_remaining_s -= dt;
                return;
            }

            let leftover = dt - self.respawn2_remaining_s;
            self.respawn2_remaining_s = 0.0;
            if leftover <= 0.0 {
                return;
            }
            self.step_physics_distractor(leftover);
            return;
        }

        self.step_physics_distractor(dt);
    }

    fn step_physics_primary(&mut self, dt: f32) -> PongEvent {
        // Use continuous collision detection (CCD) against axis-aligned boundaries.
        // This prevents the ball from "tunneling" through the paddle at larger dt.

        let speed = self.params.ball_speed;
        let mut remaining = dt;
        let mut event = PongEvent::None;

        // With dt <= 0.05 and bounded speeds, a small iteration cap is plenty.
        for _ in 0..8 {
            if remaining <= 0.0 {
                break;
            }

            let vx = self.state.ball_vx * speed;
            let vy = self.state.ball_vy * speed;

            // Find earliest boundary impact within `remaining`.
            let mut t_min = remaining;
            enum Boundary {
                Top,
                Bottom,
                Right,
                Left,
            }
            let mut hit: Option<Boundary> = None;

            if vy > 0.0 {
                let t = (1.0 - self.state.ball_y) / vy;
                if t >= 0.0 && t < t_min {
                    t_min = t;
                    hit = Some(Boundary::Top);
                }
            } else if vy < 0.0 {
                let t = (-1.0 - self.state.ball_y) / vy;
                if t >= 0.0 && t < t_min {
                    t_min = t;
                    hit = Some(Boundary::Bottom);
                }
            }

            if vx > 0.0 {
                let t = (1.0 - self.state.ball_x) / vx;
                if t >= 0.0 && t < t_min {
                    t_min = t;
                    hit = Some(Boundary::Right);
                }
            } else if vx < 0.0 {
                let t = (0.0 - self.state.ball_x) / vx;
                if t >= 0.0 && t < t_min {
                    t_min = t;
                    hit = Some(Boundary::Left);
                }
            }

            // Advance to impact (or to end of remaining time).
            self.state.ball_x += vx * t_min;
            self.state.ball_y += vy * t_min;
            remaining -= t_min;

            let Some(boundary) = hit else {
                break;
            };

            match boundary {
                Boundary::Top => {
                    self.state.ball_y = 1.0;
                    self.state.ball_vy = -self.state.ball_vy;
                }
                Boundary::Bottom => {
                    self.state.ball_y = -1.0;
                    self.state.ball_vy = -self.state.ball_vy;
                }
                Boundary::Right => {
                    self.state.ball_x = 1.0;
                    self.state.ball_vx = -self.state.ball_vx.abs();
                }
                Boundary::Left => {
                    self.state.ball_x = 0.0;
                    // Decide hit/miss at the moment of impact.
                    let hit = (self.state.ball_y - self.state.paddle_y).abs()
                        <= self.params.paddle_half_height;
                    if hit {
                        self.pending_event_reward += 1.0;
                        self.apply_paddle_bounce_primary();
                        event = PongEvent::Hit;
                    } else {
                        self.pending_event_reward -= 1.0;
                        self.respawn_remaining_s = self.params.respawn_delay_s.max(0.0);
                        self.reset_primary_for_serve();
                        return PongEvent::Miss;
                    }
                }
            }

            self.normalize_primary_dir();
        }

        event
    }

    fn step_physics_distractor(&mut self, dt: f32) {
        let speed = (self.params.ball_speed * self.params.distractor_speed_scale).clamp(0.01, 10.0);
        let mut remaining = dt;

        for _ in 0..8 {
            if remaining <= 0.0 {
                break;
            }

            let vx = self.state.ball2_vx * speed;
            let vy = self.state.ball2_vy * speed;

            let mut t_min = remaining;
            enum Boundary {
                Top,
                Bottom,
                Right,
                Left,
            }
            let mut hit: Option<Boundary> = None;

            if vy > 0.0 {
                let t = (1.0 - self.state.ball2_y) / vy;
                if t >= 0.0 && t < t_min {
                    t_min = t;
                    hit = Some(Boundary::Top);
                }
            } else if vy < 0.0 {
                let t = (-1.0 - self.state.ball2_y) / vy;
                if t >= 0.0 && t < t_min {
                    t_min = t;
                    hit = Some(Boundary::Bottom);
                }
            }

            if vx > 0.0 {
                let t = (1.0 - self.state.ball2_x) / vx;
                if t >= 0.0 && t < t_min {
                    t_min = t;
                    hit = Some(Boundary::Right);
                }
            } else if vx < 0.0 {
                let t = (0.0 - self.state.ball2_x) / vx;
                if t >= 0.0 && t < t_min {
                    t_min = t;
                    hit = Some(Boundary::Left);
                }
            }

            self.state.ball2_x += vx * t_min;
            self.state.ball2_y += vy * t_min;
            remaining -= t_min;

            let Some(boundary) = hit else {
                break;
            };

            match boundary {
                Boundary::Top => {
                    self.state.ball2_y = 1.0;
                    self.state.ball2_vy = -self.state.ball2_vy;
                }
                Boundary::Bottom => {
                    self.state.ball2_y = -1.0;
                    self.state.ball2_vy = -self.state.ball2_vy;
                }
                Boundary::Right => {
                    self.state.ball2_x = 1.0;
                    self.state.ball2_vx = -self.state.ball2_vx.abs();
                }
                Boundary::Left => {
                    self.state.ball2_x = 0.0;
                    let hit = (self.state.ball2_y - self.state.paddle_y).abs()
                        <= self.params.paddle_half_height;
                    if hit {
                        self.apply_paddle_bounce_distractor();
                    } else {
                        self.respawn2_remaining_s = self.params.respawn_delay_s.max(0.0);
                        self.reset_distractor_for_serve();
                        return;
                    }
                }
            }

            self.normalize_distractor_dir();
        }
    }

    fn reset_primary_for_serve(&mut self) {
        self.state.ball_x = 1.0;
        self.state.ball_y = self.sample_uniform(-0.6, 0.6);
        self.state.ball_vx = -1.0;
        self.state.ball_vy = self.sample_uniform(-0.9, 0.9);
        if self.state.ball_vy.abs() < 0.12 {
            self.state.ball_vy = if self.state.ball_vy >= 0.0 {
                0.12
            } else {
                -0.12
            };
        }
        self.normalize_primary_dir();
    }

    fn reset_distractor_for_serve(&mut self) {
        self.state.ball2_x = 1.0;
        self.state.ball2_y = self.sample_uniform(-0.8, 0.8);
        self.state.ball2_vx = -1.0;
        self.state.ball2_vy = self.sample_uniform(-0.9, 0.9);
        if self.state.ball2_vy.abs() < 0.12 {
            self.state.ball2_vy = if self.state.ball2_vy >= 0.0 {
                0.12
            } else {
                -0.12
            };
        }
        self.normalize_distractor_dir();
    }

    fn apply_paddle_bounce_primary(&mut self) {
        let hh = self.params.paddle_half_height.max(1.0e-6);
        let offset = ((self.state.ball_y - self.state.paddle_y) / hh).clamp(-1.0, 1.0);
        self.state.ball_vx = 1.0;
        self.state.ball_vy = (self.state.ball_vy * 0.3) + offset * self.params.paddle_bounce_y;
        self.normalize_primary_dir();
    }

    fn apply_paddle_bounce_distractor(&mut self) {
        let hh = self.params.paddle_half_height.max(1.0e-6);
        let offset = ((self.state.ball2_y - self.state.paddle_y) / hh).clamp(-1.0, 1.0);
        self.state.ball2_vx = 1.0;
        self.state.ball2_vy = (self.state.ball2_vy * 0.3) + offset * self.params.paddle_bounce_y;
        self.normalize_distractor_dir();
    }

    fn normalize_primary_dir(&mut self) {
        let len2 =
            self.state.ball_vx * self.state.ball_vx + self.state.ball_vy * self.state.ball_vy;
        if len2 > 1.0e-8 {
            let inv = 1.0 / len2.sqrt();
            self.state.ball_vx *= inv;
            self.state.ball_vy *= inv;
        }
    }

    fn normalize_distractor_dir(&mut self) {
        let len2 =
            self.state.ball2_vx * self.state.ball2_vx + self.state.ball2_vy * self.state.ball2_vy;
        if len2 > 1.0e-8 {
            let inv = 1.0 / len2.sqrt();
            self.state.ball2_vx *= inv;
            self.state.ball2_vy *= inv;
        }
    }

    fn sync_distractor_active(&mut self) {
        let should = self.params.distractor_enabled;
        if should == self.distractor_active {
            return;
        }
        self.distractor_active = should;
        if self.distractor_active {
            self.respawn2_remaining_s = 0.0;
            self.reset_distractor_for_serve();
        }
    }

    pub fn bin_signed(v: f32, bins: u32) -> u32 {
        let bins = bins.max(2);
        let t = ((v.clamp(-1.0, 1.0) + 1.0) * 0.5).clamp(0.0, 0.999_999);
        let b = (t * bins as f32).floor() as u32;
        b.min(bins - 1)
    }

    pub fn bin_01(v: f32, bins: u32) -> u32 {
        let bins = bins.max(2);
        let t = v.clamp(0.0, 0.999_999);
        let b = (t * bins as f32).floor() as u32;
        b.min(bins - 1)
    }

    fn sample_uniform(&mut self, lo: f32, hi: f32) -> f32 {
        let u = self.rng_next_f32();
        lo + (hi - lo) * u
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn miss_hides_ball_and_respawns_from_right() {
        let mut sim = PongSim::new(123);

        // Force a miss at the left boundary on the next update.
        sim.state.ball_x = 0.01;
        sim.state.ball_y = 0.0;
        sim.state.ball_vx = -1.0;
        sim.state.ball_vy = 0.0;
        sim.state.paddle_y = 1.0; // far from ball_y
        sim.params.paddle_half_height = 0.05;
        sim.params.respawn_delay_s = 0.02;

        let ev = sim.update(0.05);
        assert_eq!(ev, PongEvent::Miss);
        assert!(!sim.ball_visible());

        // Ball should already be staged at the right edge for the next serve.
        assert!((sim.state.ball_x - 1.0).abs() < 1.0e-6);
        assert!(sim.state.ball_vx < 0.0);

        // After enough time passes, it becomes visible and starts moving left.
        let _ = sim.update(sim.params.respawn_delay_s);
        assert!(sim.ball_visible());

        let x_before = sim.state.ball_x;
        let _ = sim.update(0.01);
        assert!(sim.state.ball_x < x_before);
    }
}
