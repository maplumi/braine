use wasm_bindgen::prelude::*;
use wasm_bindgen::JsCast;

use super::types::{MazeUiState, PongUiState};

use braine_games::maze::MazeSim;

#[allow(deprecated)]
pub(super) fn clear_canvas(canvas: &web_sys::HtmlCanvasElement) -> Result<(), String> {
    let ctx = canvas
        .get_context("2d")
        .map_err(|_| "canvas: get_context threw".to_string())?
        .ok_or("canvas: missing 2d context".to_string())?
        .dyn_into::<web_sys::CanvasRenderingContext2d>()
        .map_err(|_| "canvas: context is not 2d".to_string())?;

    // Match the app theme (dark scientific background)
    ctx.set_fill_style(&JsValue::from_str("#0a0f1a"));
    ctx.fill_rect(0.0, 0.0, canvas.width() as f64, canvas.height() as f64);
    Ok(())
}

#[allow(deprecated)]
pub(super) fn draw_spotxy(
    canvas: &web_sys::HtmlCanvasElement,
    x: f32,
    y: f32,
    grid_n: u32,
    accent: &str,
    selected_action: Option<&str>,
) -> Result<(), String> {
    let ctx = canvas
        .get_context("2d")
        .map_err(|_| "canvas: get_context threw".to_string())?
        .ok_or("canvas: missing 2d context".to_string())?
        .dyn_into::<web_sys::CanvasRenderingContext2d>()
        .map_err(|_| "canvas: context is not 2d".to_string())?;

    let w = canvas.width() as f64;
    let h = canvas.height() as f64;

    // Background
    ctx.set_fill_style(&JsValue::from_str("#0a0f1a"));
    ctx.fill_rect(0.0, 0.0, w, h);

    // Map x,y in [-1,1] to canvas coords.
    let px = ((x.clamp(-1.0, 1.0) as f64 + 1.0) * 0.5) * w;
    let py = (1.0 - (y.clamp(-1.0, 1.0) as f64 + 1.0) * 0.5) * h;

    if grid_n >= 2 {
        // Grid mode: N×N cells
        let eff_grid = grid_n as f64;
        let cell_w = w / eff_grid;
        let cell_h = h / eff_grid;

        // Draw grid lines
        ctx.set_stroke_style(&JsValue::from_str("rgba(122, 162, 255, 0.25)"));
        ctx.set_line_width(1.0);
        for i in 1..grid_n {
            let xf = (i as f64) * cell_w;
            let yf = (i as f64) * cell_h;
            ctx.begin_path();
            ctx.move_to(xf, 0.0);
            ctx.line_to(xf, h);
            ctx.stroke();
            ctx.begin_path();
            ctx.move_to(0.0, yf);
            ctx.line_to(w, yf);
            ctx.stroke();
        }

        // Highlight the correct cell (where dot is)
        let cx = ((px / cell_w).floor()).clamp(0.0, eff_grid - 1.0);
        let cy = ((py / cell_h).floor()).clamp(0.0, eff_grid - 1.0);
        let cell_highlight = if accent == "#22c55e" {
            "rgba(34, 197, 94, 0.12)"
        } else {
            "rgba(122, 162, 255, 0.12)"
        };
        ctx.set_fill_style(&JsValue::from_str(cell_highlight));
        ctx.fill_rect(cx * cell_w, cy * cell_h, cell_w, cell_h);

        // Highlight brain's selected cell if different from correct
        if let Some(action) = selected_action {
            // Parse action: spotxy_cell_{n:02}_{ix:02}_{iy:02}
            if let Some(coords) = parse_spotxy_cell_action(action, grid_n) {
                let (sel_ix, sel_iy) = coords;
                // Invert y for canvas (0,0 is top-left in canvas, but (0,0) in grid should be bottom-left)
                let sel_cy = (grid_n - 1 - sel_iy) as f64;
                let sel_cx = sel_ix as f64;

                // Only draw selection highlight if different from correct cell
                if sel_cx != cx || sel_cy != cy {
                    ctx.set_stroke_style(&JsValue::from_str("rgba(251, 191, 36, 0.8)"));
                    ctx.set_line_width(2.0);
                    ctx.stroke_rect(
                        sel_cx * cell_w + 1.0,
                        sel_cy * cell_h + 1.0,
                        cell_w - 2.0,
                        cell_h - 2.0,
                    );
                }
            }
        }
    } else {
        // BinaryX mode: left/right split
        // Draw center divider line
        ctx.set_stroke_style(&JsValue::from_str("rgba(122, 162, 255, 0.35)"));
        ctx.set_line_width(2.0);
        ctx.begin_path();
        ctx.move_to(w / 2.0, 0.0);
        ctx.line_to(w / 2.0, h);
        ctx.stroke();

        // Highlight the correct half (where dot is)
        let is_left = px < w / 2.0;
        let correct_highlight = if accent == "#22c55e" {
            "rgba(34, 197, 94, 0.10)"
        } else {
            "rgba(122, 162, 255, 0.10)"
        };
        ctx.set_fill_style(&JsValue::from_str(correct_highlight));
        if is_left {
            ctx.fill_rect(0.0, 0.0, w / 2.0, h);
        } else {
            ctx.fill_rect(w / 2.0, 0.0, w / 2.0, h);
        }

        // Highlight brain's selected side if different
        if let Some(action) = selected_action {
            let brain_is_left = action == "left";
            if brain_is_left != is_left {
                // Brain selected wrong side - show orange border
                ctx.set_stroke_style(&JsValue::from_str("rgba(251, 191, 36, 0.8)"));
                ctx.set_line_width(3.0);
                if brain_is_left {
                    ctx.stroke_rect(2.0, 2.0, w / 2.0 - 4.0, h - 4.0);
                } else {
                    ctx.stroke_rect(w / 2.0 + 2.0, 2.0, w / 2.0 - 4.0, h - 4.0);
                }
            }
        }

        // Add "L" and "R" labels
        ctx.set_fill_style(&JsValue::from_str("rgba(178, 186, 210, 0.3)"));
        ctx.set_font("bold 24px sans-serif");
        ctx.set_text_align("center");
        ctx.set_text_baseline("middle");
        let _ = ctx.fill_text("L", w / 4.0, h / 2.0);
        let _ = ctx.fill_text("R", 3.0 * w / 4.0, h / 2.0);
    }

    // Dot
    ctx.set_fill_style(&JsValue::from_str(accent));
    ctx.begin_path();
    let _ = ctx.arc(px, py, 6.0, 0.0, std::f64::consts::PI * 2.0);
    ctx.fill();
    Ok(())
}

/// Parse a grid cell action like "spotxy_cell_02_01_00" into (ix, iy)
fn parse_spotxy_cell_action(action: &str, expected_n: u32) -> Option<(u32, u32)> {
    // Format: spotxy_cell_{n:02}_{ix:02}_{iy:02}
    let parts: Vec<&str> = action.split('_').collect();
    if parts.len() != 5 || parts[0] != "spotxy" || parts[1] != "cell" {
        return None;
    }
    let n: u32 = parts[2].parse().ok()?;
    if n != expected_n {
        return None;
    }
    let ix: u32 = parts[3].parse().ok()?;
    let iy: u32 = parts[4].parse().ok()?;
    Some((ix, iy))
}

#[allow(deprecated)]
pub(super) fn draw_pong(
    canvas: &web_sys::HtmlCanvasElement,
    s: &PongUiState,
) -> Result<(), String> {
    let ctx = canvas
        .get_context("2d")
        .map_err(|_| "canvas: get_context threw".to_string())?
        .ok_or("canvas: missing 2d context".to_string())?
        .dyn_into::<web_sys::CanvasRenderingContext2d>()
        .map_err(|_| "canvas: context is not 2d".to_string())?;

    let w = canvas.width() as f64;
    let h = canvas.height() as f64;

    // Dark gradient background
    ctx.set_fill_style_str("#0a0f1a");
    ctx.fill_rect(0.0, 0.0, w, h);

    // Subtle grid lines for depth
    ctx.set_stroke_style_str("rgba(122, 162, 255, 0.06)");
    ctx.set_line_width(1.0);
    let grid_spacing = 30.0;
    let mut x = grid_spacing;
    while x < w {
        ctx.begin_path();
        ctx.move_to(x, 0.0);
        ctx.line_to(x, h);
        ctx.stroke();
        x += grid_spacing;
    }
    let mut y = grid_spacing;
    while y < h {
        ctx.begin_path();
        ctx.move_to(0.0, y);
        ctx.line_to(w, y);
        ctx.stroke();
        y += grid_spacing;
    }

    // Field (keep it crisp; no glow)
    let field_inset = 12.0;
    let field_left = field_inset;
    let field_right = (w - field_inset).max(field_left + 1.0);
    let field_top = field_inset;
    let field_bottom = (h - field_inset).max(field_top + 1.0);
    ctx.set_stroke_style_str("rgba(122, 162, 255, 0.28)");
    ctx.set_line_width(2.0);
    // Borders: top/bottom/right only (left side open so misses read as an abyss)
    ctx.begin_path();
    ctx.move_to(field_left, field_top);
    ctx.line_to(field_right, field_top);
    ctx.stroke();
    ctx.begin_path();
    ctx.move_to(field_left, field_bottom);
    ctx.line_to(field_right, field_bottom);
    ctx.stroke();
    ctx.begin_path();
    ctx.move_to(field_right, field_top);
    ctx.line_to(field_right, field_bottom);
    ctx.stroke();

    // Score overlay
    ctx.set_fill_style_str("rgba(0, 0, 0, 0.55)");
    ctx.fill_rect(field_left + 10.0, field_top + 8.0, 170.0, 24.0);
    ctx.set_fill_style_str("rgba(255, 255, 255, 0.92)");
    ctx.set_font("12px ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto");
    let _ = ctx.fill_text(
        &format!("Hits {}  Misses {}", s.hits, s.misses),
        field_left + 20.0,
        field_top + 25.0,
    );

    // Map simulation coordinates to pixels.
    // PongSim uses ball center positions: ball_x in [0,1], ball_y in [-1,1].
    // To make collisions *look* correct, map x=0 to the paddle face, and y=±1 to the walls.
    let ball_r = 6.0;
    let paddle_w = 10.0;
    let paddle_x = field_left; // paddle runs along the left wall

    let play_left = (paddle_x + paddle_w + ball_r).min(field_right - 1.0);
    let play_right = (field_right - ball_r).max(play_left + 1.0);
    let play_top = (field_top + ball_r).min(field_bottom - 1.0);
    let play_bottom = (field_bottom - ball_r).max(play_top + 1.0);

    let play_w = (play_right - play_left).max(1.0);
    let play_h = (play_bottom - play_top).max(1.0);

    let map_x = |x01: f32| play_left + (x01.clamp(0.0, 1.0) as f64) * play_w;
    let map_y = |ys: f32| {
        // ys=+1 is top wall; ys=-1 is bottom wall
        play_top + (1.0 - ((ys.clamp(-1.0, 1.0) as f64 + 1.0) * 0.5)) * play_h
    };

    // Paddle
    let paddle_center_y = map_y(s.paddle_y);
    let paddle_half_px = (s.paddle_half_height.clamp(0.01, 1.0) as f64) * (play_h * 0.5);
    let paddle_height = (paddle_half_px * 2.0).min(play_h);
    let paddle_top =
        (paddle_center_y - paddle_half_px).clamp(play_top, play_bottom - paddle_height);

    ctx.set_fill_style_str("#7aa2ff");
    ctx.fill_rect(paddle_x, paddle_top, paddle_w, paddle_height);
    ctx.set_fill_style_str("rgba(255, 255, 255, 0.25)");
    ctx.fill_rect(
        paddle_x + 1.5,
        paddle_top + 2.0,
        2.0,
        (paddle_height - 4.0).max(0.0),
    );

    // Ball (crisp; no glow)
    if s.ball_visible {
        let bx = map_x(s.ball_x);
        let by = map_y(s.ball_y);

        ctx.set_fill_style_str("#fbbf24");
        ctx.begin_path();
        let _ = ctx.arc(bx, by, ball_r, 0.0, std::f64::consts::PI * 2.0);
        ctx.fill();

        ctx.set_fill_style_str("rgba(255, 255, 255, 0.55)");
        ctx.begin_path();
        let _ = ctx.arc(
            bx - ball_r * 0.35,
            by - ball_r * 0.35,
            ball_r * 0.45,
            0.0,
            std::f64::consts::PI * 2.0,
        );
        ctx.fill();
    }

    // Distractor ball (ball2) - different color to distinguish
    if s.ball2_visible {
        let b2x = map_x(s.ball2_x);
        let b2y = map_y(s.ball2_y);

        ctx.set_fill_style_str("#a78bfa"); // purple for distractor
        ctx.begin_path();
        let _ = ctx.arc(b2x, b2y, ball_r, 0.0, std::f64::consts::PI * 2.0);
        ctx.fill();

        ctx.set_fill_style_str("rgba(255, 255, 255, 0.45)");
        ctx.begin_path();
        let _ = ctx.arc(
            b2x - ball_r * 0.35,
            b2y - ball_r * 0.35,
            ball_r * 0.45,
            0.0,
            std::f64::consts::PI * 2.0,
        );
        ctx.fill();
    }

    // Score zone indicator (right edge)
    ctx.set_fill_style_str("rgba(239, 68, 68, 0.12)");
    ctx.fill_rect(field_right - 6.0, field_top, 6.0, field_bottom - field_top);

    Ok(())
}

#[allow(deprecated)]
pub(super) fn draw_maze(
    canvas: &web_sys::HtmlCanvasElement,
    s: &MazeUiState,
    selected_action: Option<&str>,
) -> Result<(), String> {
    let ctx = canvas
        .get_context("2d")
        .map_err(|_| "canvas: get_context threw".to_string())?
        .ok_or("canvas: missing 2d context".to_string())?
        .dyn_into::<web_sys::CanvasRenderingContext2d>()
        .map_err(|_| "canvas: context is not 2d".to_string())?;

    let wpx = canvas.width() as f64;
    let hpx = canvas.height() as f64;

    // Background
    ctx.set_fill_style_str("#0a0f1a");
    ctx.fill_rect(0.0, 0.0, wpx, hpx);

    let mw = s.w.max(2) as f64;
    let mh = s.h.max(2) as f64;

    let inset = 12.0;
    let field_w = (wpx - 2.0 * inset).max(1.0);
    let field_h = (hpx - 2.0 * inset).max(1.0);
    let cell = (field_w / mw).min(field_h / mh).max(1.0);

    let draw_w = cell * mw;
    let draw_h = cell * mh;
    let ox = (wpx - draw_w) * 0.5;
    let oy = (hpx - draw_h) * 0.5;

    // Reconstruct maze deterministically.
    let sim = MazeSim::new_with_dims(s.seed, s.w, s.h);

    // Wall styling
    ctx.set_stroke_style_str("rgba(122, 162, 255, 0.55)");
    ctx.set_line_width((cell * 0.10).clamp(1.0, 3.0));

    // Wall bits: 1=up,2=right,4=down,8=left.
    for y in 0..s.h {
        for x in 0..s.w {
            let wx = ox + (x as f64) * cell;
            let wy = oy + (y as f64) * cell;
            let walls = sim.grid.walls(x, y);

            // Draw per-wall.
            if (walls & 1) != 0 {
                ctx.begin_path();
                ctx.move_to(wx, wy);
                ctx.line_to(wx + cell, wy);
                ctx.stroke();
            }
            if (walls & 2) != 0 {
                ctx.begin_path();
                ctx.move_to(wx + cell, wy);
                ctx.line_to(wx + cell, wy + cell);
                ctx.stroke();
            }
            if (walls & 4) != 0 {
                ctx.begin_path();
                ctx.move_to(wx, wy + cell);
                ctx.line_to(wx + cell, wy + cell);
                ctx.stroke();
            }
            if (walls & 8) != 0 {
                ctx.begin_path();
                ctx.move_to(wx, wy);
                ctx.line_to(wx, wy + cell);
                ctx.stroke();
            }
        }
    }

    // Goal (square)
    let gx = ox + (s.goal_x as f64 + 0.5) * cell;
    let gy = oy + (s.goal_y as f64 + 0.5) * cell;
    let gr = (cell * 0.22).clamp(3.0, 10.0);
    ctx.set_fill_style_str("rgba(34, 197, 94, 0.85)");
    ctx.fill_rect(gx - gr, gy - gr, 2.0 * gr, 2.0 * gr);

    // Player (circle)
    let px = ox + (s.player_x as f64 + 0.5) * cell;
    let py = oy + (s.player_y as f64 + 0.5) * cell;
    let pr = (cell * 0.22).clamp(3.0, 10.0);
    ctx.set_fill_style_str("rgba(251, 191, 36, 0.95)");
    ctx.begin_path();
    let _ = ctx.arc(px, py, pr, 0.0, std::f64::consts::PI * 2.0);
    ctx.fill();

    // Small direction hint (selected action)
    if let Some(a) = selected_action {
        let (dx, dy) = match a {
            "up" => (0.0, -1.0),
            "right" => (1.0, 0.0),
            "down" => (0.0, 1.0),
            "left" => (-1.0, 0.0),
            _ => (0.0, 0.0),
        };
        if dx != 0.0 || dy != 0.0 {
            ctx.set_stroke_style_str("rgba(251, 191, 36, 0.85)");
            ctx.set_line_width((cell * 0.08).clamp(1.0, 3.0));
            ctx.begin_path();
            ctx.move_to(px, py);
            ctx.line_to(px + dx * cell * 0.35, py + dy * cell * 0.35);
            ctx.stroke();
        }
    }

    // Legend text
    ctx.set_fill_style_str("rgba(255, 255, 255, 0.78)");
    ctx.set_font("12px ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto");
    let _ = ctx.fill_text(
        &format!(
            "Maze {}  steps {}  event {}",
            s.difficulty,
            s.steps,
            s.last_event
        ),
        12.0,
        20.0,
    );

    Ok(())
}
