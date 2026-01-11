//! Canvas-based charting and visualization for braine_web.

use wasm_bindgen::JsCast;
use web_sys::{CanvasRenderingContext2d, HtmlCanvasElement};

const SERIES_COLORS: [&str; 8] = [
    "#7aa2ff", // blue
    "#fbbf24", // amber
    "#4ade80", // green
    "#fb7185", // pink/red
    "#a78bfa", // purple
    "#22c55e", // bright green
    "#60a5fa", // light blue
    "#e879f9", // magenta
];

/// A rolling history buffer for chart data.
#[derive(Clone)]
pub struct RollingHistory {
    data: Vec<f32>,
    capacity: usize,
}

impl RollingHistory {
    pub fn new(capacity: usize) -> Self {
        Self {
            data: Vec::with_capacity(capacity),
            capacity,
        }
    }

    pub fn push(&mut self, value: f32) {
        if self.data.len() >= self.capacity {
            self.data.remove(0);
        }
        self.data.push(value);
    }

    pub fn data(&self) -> &[f32] {
        &self.data
    }

    pub fn set_data(&mut self, mut data: Vec<f32>) {
        if data.len() > self.capacity {
            data = data.split_off(data.len() - self.capacity);
        }
        self.data = data;
    }

    #[allow(dead_code)]
    pub fn len(&self) -> usize {
        self.data.len()
    }

    #[allow(dead_code)]
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    #[allow(dead_code)]
    pub fn last(&self) -> Option<f32> {
        self.data.last().copied()
    }

    #[allow(dead_code)]
    pub fn clear(&mut self) {
        self.data.clear();
    }
}

/// Unit point for brain visualization
#[allow(dead_code)]
#[derive(Clone)]
pub struct UnitPoint {
    pub amp01: f32,
    pub is_sensor: bool,
    pub is_action: bool,
}

/// Draw a brain activity visualization - circular layout with unit amplitudes
#[allow(dead_code)]
pub fn draw_brain_activity(
    canvas: &HtmlCanvasElement,
    units: &[UnitPoint],
    bg_color: &str,
) -> Result<(), String> {
    let ctx = canvas
        .get_context("2d")
        .map_err(|_| "get_context failed")?
        .ok_or("no 2d context")?
        .dyn_into::<CanvasRenderingContext2d>()
        .map_err(|_| "cast failed")?;

    let w = canvas.width() as f64;
    let h = canvas.height() as f64;
    let cx = w / 2.0;
    let cy = h / 2.0;
    let radius = (w.min(h) / 2.0) - 20.0;

    // Background
    ctx.set_fill_style_str(bg_color);
    ctx.fill_rect(0.0, 0.0, w, h);

    // Draw center circle (brain outline)
    ctx.set_stroke_style_str("rgba(122, 162, 255, 0.2)");
    ctx.set_line_width(1.0);
    ctx.begin_path();
    ctx.arc(cx, cy, radius, 0.0, std::f64::consts::PI * 2.0).ok();
    ctx.stroke();

    if units.is_empty() {
        return Ok(());
    }

    let n = units.len();
    let angle_step = (2.0 * std::f64::consts::PI) / (n as f64);

    for (i, unit) in units.iter().enumerate() {
        let angle = (i as f64) * angle_step - std::f64::consts::PI / 2.0;
        let r = radius * 0.3 + radius * 0.6 * (unit.amp01 as f64);
        let x = cx + r * angle.cos();
        let y = cy + r * angle.sin();

        let size = 2.0 + 4.0 * (unit.amp01 as f64);
        let alpha = 0.3 + 0.7 * (unit.amp01 as f64);

        let color = if unit.is_sensor {
            format!("rgba(74, 222, 128, {})", alpha) // green for sensors
        } else if unit.is_action {
            format!("rgba(251, 191, 36, {})", alpha) // yellow for actions
        } else {
            format!("rgba(122, 162, 255, {})", alpha) // blue for regular
        };

        ctx.set_fill_style_str(&color);
        ctx.begin_path();
        ctx.arc(x, y, size, 0.0, std::f64::consts::PI * 2.0).ok();
        ctx.fill();
    }

    Ok(())
}

/// Draw a compact connectivity visualization: sampled nodes on a rotating sphere and
/// edges between sampled nodes colored/thickened by connection strength.
pub fn draw_brain_connectivity_sphere(
    canvas: &HtmlCanvasElement,
    nodes: &[braine::substrate::UnitPlotPoint],
    edges: &[(u32, u32, f32)],
    rotation: f32,
    bg_color: &str,
) -> Result<(), String> {
    let ctx = canvas
        .get_context("2d")
        .map_err(|_| "get_context failed")?
        .ok_or("no 2d context")?
        .dyn_into::<CanvasRenderingContext2d>()
        .map_err(|_| "cast failed")?;

    let w = canvas.width() as f64;
    let h = canvas.height() as f64;
    let cx = w / 2.0;
    let cy = h / 2.0;
    let radius = (w.min(h) / 2.0) - 22.0;

    ctx.set_fill_style_str(bg_color);
    ctx.fill_rect(0.0, 0.0, w, h);

    // Sphere outline
    ctx.set_stroke_style_str("rgba(122, 162, 255, 0.20)");
    ctx.set_line_width(1.0);
    ctx.begin_path();
    ctx.arc(cx, cy, radius, 0.0, std::f64::consts::PI * 2.0).ok();
    ctx.stroke();

    if nodes.is_empty() {
        return Ok(());
    }

    // Golden spiral distribution on sphere
    let n = nodes.len() as f64;
    let golden = 2.39996322972865332_f64; // ~pi*(3-sqrt(5))
    let rot = rotation as f64;

    let mut pos: std::collections::HashMap<u32, (f64, f64, f64)> =
        std::collections::HashMap::with_capacity(nodes.len());
    for (i, node) in nodes.iter().enumerate() {
        let i = i as f64;
        let y = 1.0 - 2.0 * ((i + 0.5) / n);
        let r = (1.0 - y * y).sqrt();
        let theta = golden * i;
        let x = r * theta.cos();
        let z = r * theta.sin();

        // Rotate around Y axis
        let xr = x * rot.cos() + z * rot.sin();
        let zr = -x * rot.sin() + z * rot.cos();

        pos.insert(node.id, (xr, y, zr));
    }

    // Draw edges (projected). Limit visual density by weight.
    let mut max_abs_w = 0.0001f32;
    for &(_s, _t, w) in edges {
        max_abs_w = max_abs_w.max(w.abs());
    }

    for &(s, t, w) in edges {
        let Some(&(sx, sy, sz)) = pos.get(&s) else { continue; };
        let Some(&(tx, ty, tz)) = pos.get(&t) else { continue; };

        let depth = ((sz + tz) * 0.5 + 1.0) * 0.5; // 0..1 approx
        let absw = (w.abs() / max_abs_w).clamp(0.0, 1.0) as f64;
        if absw < 0.12 {
            continue;
        }

        // Simple perspective.
        let dist = 2.8;
        let sp = radius / (sz + dist);
        let tp = radius / (tz + dist);
        let x1 = cx + sx * sp;
        let y1 = cy + sy * sp;
        let x2 = cx + tx * tp;
        let y2 = cy + ty * tp;

        let alpha = (0.06 + 0.32 * absw) * (0.35 + 0.65 * depth);
        if w >= 0.0 {
            ctx.set_stroke_style_str(&format!("rgba(122, 162, 255, {:.3})", alpha));
        } else {
            ctx.set_stroke_style_str(&format!("rgba(251, 113, 133, {:.3})", alpha));
        }
        ctx.set_line_width(0.5 + 1.6 * absw);
        ctx.begin_path();
        ctx.move_to(x1, y1);
        ctx.line_to(x2, y2);
        ctx.stroke();
    }

    // Draw nodes on top.
    for node in nodes {
        let Some(&(x, y, z)) = pos.get(&node.id) else { continue; };
        let dist = 2.8;
        let p = radius / (z + dist);
        let px = cx + x * p;
        let py = cy + y * p;

        let amp = (node.amp01 as f64).clamp(0.0, 1.0);
        let base = if node.is_sensor_member {
            2.4
        } else if node.is_group_member {
            2.0
        } else {
            1.7
        };
        let size = base + 5.0 * amp;
        let alpha = 0.35 + 0.65 * amp;

        let color = if node.is_reserved {
            format!("rgba(148, 163, 184, {:.3})", alpha)
        } else if node.is_sensor_member {
            format!("rgba(74, 222, 128, {:.3})", alpha)
        } else if node.is_group_member {
            format!("rgba(251, 191, 36, {:.3})", alpha)
        } else {
            format!("rgba(122, 162, 255, {:.3})", alpha)
        };

        ctx.set_fill_style_str(&color);
        ctx.begin_path();
        ctx.arc(px, py, size, 0.0, std::f64::consts::PI * 2.0).ok();
        ctx.fill();
    }

    Ok(())
}

/// Draw a gauge/meter visualization
pub fn draw_gauge(
    canvas: &HtmlCanvasElement,
    value: f32,
    min_val: f32,
    max_val: f32,
    label: &str,
    color: &str,
    bg_color: &str,
) -> Result<(), String> {
    let ctx = canvas
        .get_context("2d")
        .map_err(|_| "get_context failed")?
        .ok_or("no 2d context")?
        .dyn_into::<CanvasRenderingContext2d>()
        .map_err(|_| "cast failed")?;

    let w = canvas.width() as f64;
    let h = canvas.height() as f64;
    let cx = w / 2.0;
    let cy = h - 10.0;
    let radius = (w.min(h) * 0.8) - 10.0;

    // Background
    ctx.set_fill_style_str(bg_color);
    ctx.fill_rect(0.0, 0.0, w, h);

    // Draw arc background
    ctx.set_stroke_style_str("rgba(122, 162, 255, 0.15)");
    ctx.set_line_width(12.0);
    ctx.begin_path();
    ctx.arc(cx, cy, radius, std::f64::consts::PI, 0.0).ok();
    ctx.stroke();

    // Draw value arc
    let range = (max_val - min_val).max(0.001);
    let norm = ((value - min_val) / range).clamp(0.0, 1.0) as f64;
    let end_angle = std::f64::consts::PI * (1.0 - norm);

    ctx.set_stroke_style_str(color);
    ctx.set_line_width(12.0);
    ctx.begin_path();
    ctx.arc(cx, cy, radius, std::f64::consts::PI, end_angle).ok();
    ctx.stroke();

    // Draw label and value
    ctx.set_fill_style_str("rgba(232, 236, 255, 0.9)");
    ctx.set_font("bold 14px system-ui, sans-serif");
    ctx.set_text_align("center");
    let _ = ctx.fill_text(&format!("{:.1}%", value * 100.0), cx, cy - 20.0);
    
    ctx.set_font("11px system-ui, sans-serif");
    ctx.set_fill_style_str("rgba(170, 180, 230, 0.8)");
    let _ = ctx.fill_text(label, cx, cy - 5.0);

    Ok(())
}

/// Draw action scores as horizontal bars
#[allow(dead_code)]
pub fn draw_action_scores(
    canvas: &HtmlCanvasElement,
    actions: &[(&str, f32)], // (name, score 0-1)
    highlight: Option<&str>,
    bg_color: &str,
) -> Result<(), String> {
    let ctx = canvas
        .get_context("2d")
        .map_err(|_| "get_context failed")?
        .ok_or("no 2d context")?
        .dyn_into::<CanvasRenderingContext2d>()
        .map_err(|_| "cast failed")?;

    let w = canvas.width() as f64;
    let h = canvas.height() as f64;

    ctx.set_fill_style_str(bg_color);
    ctx.fill_rect(0.0, 0.0, w, h);

    if actions.is_empty() {
        return Ok(());
    }

    let bar_height = ((h - 10.0) / actions.len() as f64).min(24.0);
    let label_width = 60.0;
    let bar_area = w - label_width - 20.0;

    ctx.set_font("11px system-ui, sans-serif");
    ctx.set_text_align("right");

    for (i, (name, score)) in actions.iter().enumerate() {
        let y = 5.0 + (i as f64) * (bar_height + 4.0);
        let is_highlight = highlight.map_or(false, |h| h == *name);

        // Label
        ctx.set_fill_style_str(if is_highlight { "rgba(251, 191, 36, 1.0)" } else { "rgba(170, 180, 230, 0.9)" });
        let _ = ctx.fill_text(name, label_width - 5.0, y + bar_height * 0.7);

        // Bar background
        ctx.set_fill_style_str("rgba(122, 162, 255, 0.1)");
        ctx.fill_rect(label_width, y, bar_area, bar_height);

        // Bar value
        let bar_w = bar_area * (*score as f64).clamp(0.0, 1.0);
        let bar_color = if is_highlight { "rgba(251, 191, 36, 0.8)" } else { "rgba(122, 162, 255, 0.6)" };
        ctx.set_fill_style_str(bar_color);
        ctx.fill_rect(label_width, y, bar_w, bar_height);
    }

    Ok(())
}

/// Draw neuromodulator trace (reward history)
pub fn draw_neuromod_trace(
    canvas: &HtmlCanvasElement,
    data: &[f32],
    bg_color: &str,
) -> Result<(), String> {
    let ctx = canvas
        .get_context("2d")
        .map_err(|_| "get_context failed")?
        .ok_or("no 2d context")?
        .dyn_into::<CanvasRenderingContext2d>()
        .map_err(|_| "cast failed")?;

    let w = canvas.width() as f64;
    let h = canvas.height() as f64;

    ctx.set_fill_style_str(bg_color);
    ctx.fill_rect(0.0, 0.0, w, h);

    // Zero line
    let mid_y = h / 2.0;
    ctx.set_stroke_style_str("rgba(122, 162, 255, 0.3)");
    ctx.set_line_width(1.0);
    ctx.begin_path();
    ctx.move_to(0.0, mid_y);
    ctx.line_to(w, mid_y);
    ctx.stroke();

    if data.is_empty() {
        return Ok(());
    }

    let step_x = w / (data.len().max(1) as f64);

    // Draw bars for each value
    for (i, &val) in data.iter().enumerate() {
        let x = (i as f64) * step_x;
        let bar_h = (val.abs() as f64) * (h / 2.0 - 2.0);

        let color = if val > 0.0 {
            "rgba(74, 222, 128, 0.7)" // green for positive
        } else if val < 0.0 {
            "rgba(248, 113, 113, 0.7)" // red for negative
        } else {
            "rgba(122, 162, 255, 0.3)" // neutral
        };

        ctx.set_fill_style_str(color);
        if val >= 0.0 {
            ctx.fill_rect(x, mid_y - bar_h, step_x.max(2.0), bar_h);
        } else {
            ctx.fill_rect(x, mid_y, step_x.max(2.0), bar_h);
        }
    }

    Ok(())
}

/// Draw a simple line chart on a canvas.
pub fn draw_line_chart(
    canvas: &HtmlCanvasElement,
    data: &[f32],
    min_val: f32,
    max_val: f32,
    line_color: &str,
    bg_color: &str,
    grid_color: &str,
) -> Result<(), String> {
    let ctx = canvas
        .get_context("2d")
        .map_err(|_| "get_context failed")?
        .ok_or("no 2d context")?
        .dyn_into::<CanvasRenderingContext2d>()
        .map_err(|_| "cast failed")?;

    let w = canvas.width() as f64;
    let h = canvas.height() as f64;

    // Background
    ctx.set_fill_style_str(bg_color);
    ctx.fill_rect(0.0, 0.0, w, h);

    // Grid lines
    ctx.set_stroke_style_str(grid_color);
    ctx.set_line_width(0.5);
    for i in 1..5 {
        let y = h * (i as f64) / 5.0;
        ctx.begin_path();
        ctx.move_to(0.0, y);
        ctx.line_to(w, y);
        ctx.stroke();
    }

    if data.is_empty() {
        return Ok(());
    }

    let range = (max_val - min_val).max(0.001);
    let step_x = w / (data.len().max(1) as f64);

    ctx.set_stroke_style_str(line_color);
    ctx.set_line_width(2.0);
    ctx.begin_path();

    for (i, &val) in data.iter().enumerate() {
        let norm = ((val - min_val) / range).clamp(0.0, 1.0) as f64;
        let x = (i as f64) * step_x;
        let y = h - norm * h;

        if i == 0 {
            ctx.move_to(x, y);
        } else {
            ctx.line_to(x, y);
        }
    }

    ctx.stroke();
    Ok(())
}

/// Draw action-choice probabilities over time.
///
/// `events` is a sequence of action names (one per trial). The chart plots, for each action in
/// `actions`, the rolling probability of that action in the trailing `window` events.
pub fn draw_choices_over_time(
    canvas: &HtmlCanvasElement,
    actions: &[String],
    events: &[String],
    window: usize,
    bg_color: &str,
    grid_color: &str,
) -> Result<(), String> {
    let ctx = canvas
        .get_context("2d")
        .map_err(|_| "get_context failed")?
        .ok_or("no 2d context")?
        .dyn_into::<CanvasRenderingContext2d>()
        .map_err(|_| "cast failed")?;

    let w = canvas.width() as f64;
    let h = canvas.height() as f64;

    ctx.set_fill_style_str(bg_color);
    ctx.fill_rect(0.0, 0.0, w, h);

    // Grid
    ctx.set_stroke_style_str(grid_color);
    ctx.set_line_width(0.5);
    for i in 1..5 {
        let y = h * (i as f64) / 5.0;
        ctx.begin_path();
        ctx.move_to(0.0, y);
        ctx.line_to(w, y);
        ctx.stroke();
    }

    if events.is_empty() || actions.is_empty() {
        // Title
        ctx.set_font("12px system-ui, sans-serif");
        ctx.set_fill_style_str("rgba(170, 180, 230, 0.8)");
        ctx.set_text_align("left");
        let _ = ctx.fill_text("No action history yet", 10.0, 18.0);
        return Ok(());
    }

    let window = window.max(1);
    let n = events.len();
    let step_x = if n <= 1 { w } else { w / ((n - 1) as f64) };

    // Precompute series (rolling probabilities).
    let mut series: Vec<Vec<f32>> = vec![Vec::with_capacity(n); actions.len()];
    for i in 0..n {
        let start = i.saturating_sub(window - 1);
        let slice = &events[start..=i];
        let denom = (slice.len() as f32).max(1.0);
        for (ai, name) in actions.iter().enumerate() {
            let mut c = 0u32;
            for ev in slice {
                if ev == name {
                    c += 1;
                }
            }
            series[ai].push((c as f32) / denom);
        }
    }

    // Draw series.
    for (ai, probs) in series.iter().enumerate() {
        let color = SERIES_COLORS[ai % SERIES_COLORS.len()];
        ctx.set_stroke_style_str(color);
        ctx.set_line_width(2.0);
        ctx.begin_path();

        for (i, &p) in probs.iter().enumerate() {
            let x = (i as f64) * step_x;
            let y = h - (p as f64).clamp(0.0, 1.0) * h;
            if i == 0 {
                ctx.move_to(x, y);
            } else {
                ctx.line_to(x, y);
            }
        }
        ctx.stroke();
    }

    // Legend
    ctx.set_font("11px system-ui, sans-serif");
    ctx.set_text_align("left");
    let mut lx = 10.0;
    let ly = 16.0;
    for (ai, name) in actions.iter().enumerate() {
        let color = SERIES_COLORS[ai % SERIES_COLORS.len()];
        ctx.set_fill_style_str(color);
        ctx.fill_rect(lx, ly - 9.0, 10.0, 3.0);
        ctx.set_fill_style_str("rgba(232, 236, 255, 0.9)");
        let _ = ctx.fill_text(name, lx + 14.0, ly);
        lx += 14.0 + (name.len() as f64 * 7.0);
        if lx > w - 120.0 {
            // Wrap if it gets too wide.
            lx = 10.0;
        }
    }

    Ok(())
}

/// Draw a bar chart on a canvas (e.g., for action distribution).
#[allow(dead_code)]
pub fn draw_bar_chart(
    canvas: &HtmlCanvasElement,
    labels: &[&str],
    values: &[f32],
    bar_color: &str,
    bg_color: &str,
    text_color: &str,
) -> Result<(), String> {
    let ctx = canvas
        .get_context("2d")
        .map_err(|_| "get_context failed")?
        .ok_or("no 2d context")?
        .dyn_into::<CanvasRenderingContext2d>()
        .map_err(|_| "cast failed")?;

    let w = canvas.width() as f64;
    let h = canvas.height() as f64;

    ctx.set_fill_style_str(bg_color);
    ctx.fill_rect(0.0, 0.0, w, h);

    if labels.is_empty() || values.is_empty() {
        return Ok(());
    }

    let n = labels.len().min(values.len());
    let max_val = values.iter().cloned().fold(0.01f32, f32::max);
    let bar_width = (w / (n as f64)) * 0.7;
    let gap = (w / (n as f64)) * 0.15;

    ctx.set_font("12px system-ui, sans-serif");

    for i in 0..n {
        let x = (i as f64) * (w / (n as f64)) + gap;
        let norm = (values[i] / max_val).clamp(0.0, 1.0) as f64;
        let bar_h = norm * (h - 24.0);
        let y = h - 20.0 - bar_h;

        ctx.set_fill_style_str(bar_color);
        ctx.fill_rect(x, y, bar_width, bar_h);

        ctx.set_fill_style_str(text_color);
        ctx.set_text_align("center");
        let _ = ctx.fill_text(labels[i], x + bar_width / 2.0, h - 4.0);
    }

    Ok(())
}

/// Draw a 3D-style unit activity plot (like desktop Brain Plot).
/// X-axis: rel_age (0=old units, 1=new units)
/// Y-axis: amp01 (normalized amplitude 0-1)
/// Z-depth effect: is_sensor_member units are larger/brighter (foreground)
pub fn draw_unit_plot_3d(
    canvas: &HtmlCanvasElement,
    points: &[braine::substrate::UnitPlotPoint],
    bg_color: &str,
) -> Result<(), String> {
    let ctx = canvas
        .get_context("2d")
        .map_err(|_| "get_context failed")?
        .ok_or("no 2d context")?
        .dyn_into::<CanvasRenderingContext2d>()
        .map_err(|_| "cast failed")?;

    let w = canvas.width() as f64;
    let h = canvas.height() as f64;
    let padding = 40.0;
    let plot_w = w - padding * 2.0;
    let plot_h = h - padding * 2.0;

    // Clear background
    ctx.set_fill_style_str(bg_color);
    ctx.fill_rect(0.0, 0.0, w, h);

    // Draw grid lines
    ctx.set_stroke_style_str("rgba(122, 162, 255, 0.1)");
    ctx.set_line_width(1.0);
    for i in 0..=10 {
        let y = padding + (i as f64) * plot_h / 10.0;
        ctx.begin_path();
        ctx.move_to(padding, y);
        ctx.line_to(w - padding, y);
        ctx.stroke();
        
        let x = padding + (i as f64) * plot_w / 10.0;
        ctx.begin_path();
        ctx.move_to(x, padding);
        ctx.line_to(x, h - padding);
        ctx.stroke();
    }

    // Draw axes
    ctx.set_stroke_style_str("rgba(122, 162, 255, 0.4)");
    ctx.set_line_width(2.0);
    // Y-axis
    ctx.begin_path();
    ctx.move_to(padding, padding);
    ctx.line_to(padding, h - padding);
    ctx.stroke();
    // X-axis
    ctx.begin_path();
    ctx.move_to(padding, h - padding);
    ctx.line_to(w - padding, h - padding);
    ctx.stroke();

    // Axis labels
    ctx.set_fill_style_str("rgba(255, 255, 255, 0.6)");
    ctx.set_font("11px system-ui, sans-serif");
    ctx.set_text_align("center");
    let _ = ctx.fill_text("rel_age (old → new)", w / 2.0, h - 8.0);
    
    ctx.save();
    ctx.translate(12.0, h / 2.0).ok();
    ctx.rotate(-std::f64::consts::PI / 2.0).ok();
    let _ = ctx.fill_text("amplitude", 0.0, 0.0);
    ctx.restore();

    // Draw "0" and "1" labels
    ctx.set_text_align("right");
    let _ = ctx.fill_text("1.0", padding - 5.0, padding + 4.0);
    let _ = ctx.fill_text("0.0", padding - 5.0, h - padding + 4.0);
    ctx.set_text_align("center");
    let _ = ctx.fill_text("0", padding, h - padding + 16.0);
    let _ = ctx.fill_text("1", w - padding, h - padding + 16.0);

    if points.is_empty() {
        // Draw "no data" message
        ctx.set_fill_style_str("rgba(255, 255, 255, 0.4)");
        ctx.set_font("14px system-ui, sans-serif");
        ctx.set_text_align("center");
        let _ = ctx.fill_text("No unit data - start simulation", w / 2.0, h / 2.0);
        return Ok(());
    }

    // Sort points by "depth" - reserved/background first, sensor/group last (foreground)
    let mut sorted_points: Vec<_> = points.iter().collect();
    sorted_points.sort_by(|a, b| {
        let depth_a = if a.is_sensor_member { 3 } else if a.is_group_member { 2 } else if a.is_reserved { 0 } else { 1 };
        let depth_b = if b.is_sensor_member { 3 } else if b.is_group_member { 2 } else if b.is_reserved { 0 } else { 1 };
        depth_a.cmp(&depth_b)
    });

    // Draw points with 3D depth effect
    for point in sorted_points {
        let x = padding + (point.rel_age as f64) * plot_w;
        let y = h - padding - (point.amp01 as f64) * plot_h;
        
        // Size based on depth (sensor = largest, reserved = smallest)
        let base_size = if point.is_sensor_member {
            8.0
        } else if point.is_group_member {
            6.0
        } else if point.is_reserved {
            3.0
        } else {
            5.0
        };
        
        // Color based on unit type
        let (color, glow_color) = if point.is_reserved {
            ("rgba(136, 136, 136, 0.6)", "rgba(136, 136, 136, 0.2)")
        } else if point.is_sensor_member {
            ("rgba(122, 162, 255, 0.9)", "rgba(122, 162, 255, 0.4)")
        } else if point.is_group_member {
            ("rgba(74, 222, 128, 0.85)", "rgba(74, 222, 128, 0.3)")
        } else {
            ("rgba(251, 191, 36, 0.75)", "rgba(251, 191, 36, 0.25)")
        };
        
        // Draw glow effect for foreground units
        if point.is_sensor_member || point.is_group_member {
            ctx.set_fill_style_str(glow_color);
            ctx.begin_path();
            ctx.arc(x, y, base_size * 2.0, 0.0, std::f64::consts::PI * 2.0).ok();
            ctx.fill();
        }
        
        // Draw 3D-style sphere (gradient circle)
        // Outer darker ring
        ctx.set_fill_style_str(color);
        ctx.begin_path();
        ctx.arc(x, y, base_size, 0.0, std::f64::consts::PI * 2.0).ok();
        ctx.fill();
        
        // Inner highlight for 3D effect
        let highlight_color = if point.is_reserved {
            "rgba(180, 180, 180, 0.5)"
        } else if point.is_sensor_member {
            "rgba(180, 200, 255, 0.6)"
        } else if point.is_group_member {
            "rgba(140, 255, 180, 0.5)"
        } else {
            "rgba(255, 220, 140, 0.5)"
        };
        ctx.set_fill_style_str(highlight_color);
        ctx.begin_path();
        ctx.arc(x - base_size * 0.25, y - base_size * 0.25, base_size * 0.5, 0.0, std::f64::consts::PI * 2.0).ok();
        ctx.fill();
    }

    // Draw count label
    ctx.set_fill_style_str("rgba(122, 162, 255, 0.8)");
    ctx.set_font("12px system-ui, sans-serif");
    ctx.set_text_align("right");
    let _ = ctx.fill_text(&format!("{} units", points.len()), w - padding, padding - 8.0);

    Ok(())
}
