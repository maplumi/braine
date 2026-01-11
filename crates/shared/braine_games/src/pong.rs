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
    /// Ball hidden duration after a miss (seconds).
    pub respawn_delay_s: f32,
}

impl Default for PongParams {
    fn default() -> Self {
        Self {
            paddle_speed: 1.3,
            paddle_half_height: 0.15,
            ball_speed: 1.0,
            respawn_delay_s: 0.18,
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
    pub paddle_y: f32,
}

impl Default for PongState {
    fn default() -> Self {
        Self {
            ball_x: 0.5,
            ball_y: 0.0,
            ball_vx: 0.75,
            ball_vy: 0.15,
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
    pending_event_reward: f32,
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

impl PongSim {
    pub fn new(seed: u64) -> Self {
        let mut sim = Self {
            state: PongState::default(),
            params: PongParams::default(),
            rng_seed: seed,
            respawn_remaining_s: 0.0,
            pending_event_reward: 0.0,
        };
        sim.reset_point();
        sim
    }

    pub fn ball_visible(&self) -> bool {
        self.respawn_remaining_s <= 0.0
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
        self.state.ball_x = 0.5;
        self.state.ball_y = self.sample_uniform(-0.6, 0.6);
        self.state.ball_vx = 0.75;
        self.state.ball_vy = self.sample_uniform(-0.55, 0.55);
        if self.state.ball_vy.abs() < 0.15 {
            self.state.ball_vy = if self.state.ball_vy >= 0.0 {
                0.15
            } else {
                -0.15
            };
        }
        self.state.paddle_y = 0.0;
        self.respawn_remaining_s = 0.0;
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

        if self.respawn_remaining_s > 0.0 {
            // While hidden, do not advance ball physics. If the respawn timer
            // elapses during this tick, consume the remaining dt for physics
            // so the ball starts moving immediately upon reappearing.
            if dt < self.respawn_remaining_s {
                self.respawn_remaining_s -= dt;
                return PongEvent::None;
            }

            let leftover = dt - self.respawn_remaining_s;
            self.respawn_remaining_s = 0.0;
            if leftover <= 0.0 {
                return PongEvent::None;
            }
            return self.step_physics(leftover);
        }

        self.step_physics(dt)
    }

    pub fn apply_action(&mut self, action: PongAction, dt: f32) {
        let dt = dt.clamp(0.0, 60.0);
        match action {
            PongAction::Up => self.state.paddle_y += self.params.paddle_speed * dt,
            PongAction::Down => self.state.paddle_y -= self.params.paddle_speed * dt,
            PongAction::Stay => {}
        }
        self.state.paddle_y = self.state.paddle_y.clamp(-1.0, 1.0);
    }

    fn step_physics(&mut self, dt: f32) -> PongEvent {
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
                        self.state.ball_vx = self.state.ball_vx.abs();
                        event = PongEvent::Hit;
                    } else {
                        self.pending_event_reward -= 1.0;
                        self.respawn_remaining_s = self.params.respawn_delay_s.max(0.0);
                        self.reset_point_for_serve();
                        return PongEvent::Miss;
                    }
                }
            }
        }

        event
    }

    fn reset_point_for_serve(&mut self) {
        // Like `reset_point`, but does not clear respawn timer.
        // Respawn from the right edge moving left toward the paddle.
        self.state.ball_x = 1.0;
        self.state.ball_y = self.sample_uniform(-0.6, 0.6);
        self.state.ball_vx = -0.75;
        self.state.ball_vy = self.sample_uniform(-0.55, 0.55);
        if self.state.ball_vy.abs() < 0.15 {
            self.state.ball_vy = if self.state.ball_vy >= 0.0 {
                0.15
            } else {
                -0.15
            };
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
