//! Shared Pong environment/state.
//!
//! This is intentionally kept independent of any UI or networking layer so it can be reused by:
//! - the daemon (native)
//! - a browser-hosted WASM app (edge learning)
//!
//! Implementation will be migrated from the daemon's current Pong logic.

/// Pong game state (physics + ball/paddle positions).
#[derive(Debug, Clone, Copy, Default)]
pub struct PongState {
    pub ball_x: f32,
    pub ball_y: f32,
    pub ball_vx: f32,
    pub ball_vy: f32,
    pub paddle_y: f32,
}
