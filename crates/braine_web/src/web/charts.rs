//! Canvas-based charting and visualization for braine_web.

use wasm_bindgen::{JsCast, JsValue};
use js_sys::Reflect;
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
    ctx.arc(cx, cy, radius, 0.0, std::f64::consts::PI * 2.0)
        .ok();
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
#[derive(Clone, Copy, Debug)]
pub struct BrainVizHitNode {
    pub id: u32,
    pub x: f64,
    pub y: f64,
    pub r: f64,
}

#[derive(Clone, Copy, Debug)]
pub struct BrainVizRenderOptions {
    pub zoom: f32,
    pub pan_x: f32,
    pub pan_y: f32,
    pub draw_outline: bool,
    pub node_size_scale: f64,
    /// If true, nodes are tinted to indicate learning mode; if false, inference mode.
    pub learning_mode: bool,
    /// Animation time (e.g., step count) for pulsing effects.
    pub anim_time: f32,
    /// Y-axis rotation angle (radians) - horizontal drag.
    pub rotation_y: f32,
    /// X-axis rotation angle (radians) - vertical drag.
    pub rotation_x: f32,
}

impl Default for BrainVizRenderOptions {
    fn default() -> Self {
        Self {
            zoom: 1.0,
            pan_x: 0.0,
            pan_y: 0.0,
            draw_outline: true,
            node_size_scale: 1.0,
            learning_mode: true,
            anim_time: 0.0,
            rotation_y: 0.0,
            rotation_x: 0.0,
        }
    }
}

pub fn draw_brain_connectivity_sphere(
    canvas: &HtmlCanvasElement,
    nodes: &[braine::substrate::UnitPlotPoint],
    edges: &[(u32, u32, f32)],
    _rotation: f32, // DEPRECATED: use opts.rotation_y instead
    bg_color: &str,
    opts: BrainVizRenderOptions,
) -> Result<Vec<BrainVizHitNode>, String> {
    let ctx = canvas
        .get_context("2d")
        .map_err(|_| "get_context failed")?
        .ok_or("no 2d context")?
        .dyn_into::<CanvasRenderingContext2d>()
        .map_err(|_| "cast failed")?;

    let w = canvas.width() as f64;
    let h = canvas.height() as f64;
    let zoom = (opts.zoom as f64).clamp(0.25, 8.0);
    let cx = (w / 2.0) + (opts.pan_x as f64);
    let cy = (h / 2.0) + (opts.pan_y as f64);
    let radius = ((w.min(h) / 2.0) - 22.0) * zoom;

    ctx.set_fill_style_str(bg_color);
    ctx.fill_rect(0.0, 0.0, w, h);

    // Sphere outline (optional)
    if opts.draw_outline {
        ctx.set_stroke_style_str("rgba(122, 162, 255, 0.20)");
        ctx.set_line_width(1.0);
        ctx.begin_path();
        ctx.arc(cx, cy, radius, 0.0, std::f64::consts::PI * 2.0)
            .ok();
        ctx.stroke();
    }

    if nodes.is_empty() {
        return Ok(Vec::new());
    }

    // Golden spiral distribution on sphere
    let n = nodes.len() as f64;
    let golden = 2.39996322972865332_f64; // ~pi*(3-sqrt(5))

    // Use rotation from opts (supports both X and Y axes)
    let rot_y = opts.rotation_y as f64;
    let rot_x = opts.rotation_x as f64;

    let mut pos: std::collections::HashMap<u32, (f64, f64, f64)> =
        std::collections::HashMap::with_capacity(nodes.len());
    for (i, node) in nodes.iter().enumerate() {
        let i = i as f64;
        let y = 1.0 - 2.0 * ((i + 0.5) / n);
        let r = (1.0 - y * y).sqrt();
        let theta = golden * i;
        let x = r * theta.cos();
        let z = r * theta.sin();

        // Rotate around Y axis (horizontal drag)
        let xr = x * rot_y.cos() + z * rot_y.sin();
        let zr = -x * rot_y.sin() + z * rot_y.cos();

        // Rotate around X axis (vertical drag)
        let yr = y * rot_x.cos() - zr * rot_x.sin();
        let zr2 = y * rot_x.sin() + zr * rot_x.cos();

        pos.insert(node.id, (xr, yr, zr2));
    }

    // Precompute projected screen positions for all nodes.
    // Edges MUST use the exact same projection as nodes, otherwise they can appear
    // visually disconnected.
    let dist = 2.8;
    let mut proj: std::collections::HashMap<u32, (f64, f64, f64)> =
        std::collections::HashMap::with_capacity(nodes.len());
    for node in nodes {
        let Some(&(x, y, z)) = pos.get(&node.id) else {
            continue;
        };
        let p = radius / (z + dist);
        let px = cx + x * p;
        let py = cy + y * p;
        proj.insert(node.id, (px, py, z));
    }

    // Fast lookup for per-node coloring (used by gradient edges).
    let node_by_id: std::collections::HashMap<u32, &braine::substrate::UnitPlotPoint> =
        nodes.iter().map(|n| (n.id, n)).collect();

    let node_base_rgb =
        |node: &braine::substrate::UnitPlotPoint, learning_mode: bool| -> (u8, u8, u8) {
            if learning_mode {
                if node.is_sensor_member {
                    (255, 153, 102) // warm orange
                } else if node.is_group_member {
                    (74, 222, 128) // green
                } else if node.is_reserved {
                    (148, 163, 184) // gray
                } else {
                    (251, 191, 36) // amber
                }
            } else {
                if node.is_sensor_member {
                    (122, 162, 255) // blue
                } else if node.is_group_member {
                    (34, 211, 238) // cyan
                } else if node.is_reserved {
                    (148, 163, 184) // gray
                } else {
                    (167, 139, 250) // purple
                }
            }
        };

    // Draw edges (projected). Limit visual density by weight.
    let mut max_abs_w = 0.0001f32;
    for &(_s, _t, w) in edges {
        max_abs_w = max_abs_w.max(w.abs());
    }

    for &(s, t, w) in edges {
        let Some(&(x1, y1, sz)) = proj.get(&s) else {
            continue;
        };
        let Some(&(x2, y2, tz)) = proj.get(&t) else {
            continue;
        };

        let absw = (w.abs() / max_abs_w).clamp(0.0, 1.0) as f64;
        if absw < 0.12 {
            continue;
        }

        // Depth (0..1 approx): 1=front, 0=back.
        let adj_depth = ((sz + tz) * 0.5 + 1.0) * 0.5;

        // Width is primarily based on connection strength, with slight depth influence
        // Strong connections = thick lines, weak connections = thin lines
        let strength_width = 0.5 + 2.5 * absw; // Range: 0.5 to 3.0 (strong emphasis)
        let depth_factor = 0.7 + 0.3 * adj_depth; // Slight boost for front edges
        let edge_width = strength_width * depth_factor;

        // Alpha combines strength (strong = more visible) and depth (front = more visible)
        let alpha = (0.10 + 0.45 * absw) * (0.30 + 0.70 * adj_depth);

        // Edge gradient is derived from endpoint node colors.
        // If either endpoint is missing (shouldn't happen), fall back to sign-based colors.
        let (r1, g1, b1, r2, g2, b2) = match (node_by_id.get(&s), node_by_id.get(&t)) {
            (Some(ns), Some(nt)) => {
                let (r1, g1, b1) = node_base_rgb(ns, opts.learning_mode);
                let (r2, g2, b2) = node_base_rgb(nt, opts.learning_mode);
                (r1, g1, b1, r2, g2, b2)
            }
            _ => {
                if w >= 0.0 {
                    (122, 162, 255, 122, 162, 255)
                } else {
                    (251, 113, 133, 251, 113, 133)
                }
            }
        };

        let grad = ctx.create_linear_gradient(x1, y1, x2, y2);
        let _ = grad.add_color_stop(0.0, &format!("rgba({r1}, {g1}, {b1}, {:.3})", alpha));
        let _ = grad.add_color_stop(1.0, &format!("rgba({r2}, {g2}, {b2}, {:.3})", alpha));
        let _ = Reflect::set(
            ctx.as_ref(),
            &JsValue::from_str("strokeStyle"),
            grad.as_ref(),
        );
        ctx.set_line_width(edge_width);
        ctx.begin_path();
        ctx.move_to(x1, y1);
        ctx.line_to(x2, y2);
        ctx.stroke();

        // Draw a subtle extra pass for very strong edges (they're already pulled forward)
        if absw > 0.6 {
            let tension_alpha = (absw - 0.6) * 0.4 * adj_depth;
            let grad = ctx.create_linear_gradient(x1, y1, x2, y2);
            let _ = grad.add_color_stop(
                0.0,
                &format!("rgba({r1}, {g1}, {b1}, {:.3})", tension_alpha),
            );
            let _ = grad.add_color_stop(
                1.0,
                &format!("rgba({r2}, {g2}, {b2}, {:.3})", tension_alpha),
            );
            let _ = Reflect::set(
                ctx.as_ref(),
                &JsValue::from_str("strokeStyle"),
                grad.as_ref(),
            );
            ctx.set_line_width(edge_width * 1.8);
            ctx.begin_path();
            ctx.move_to(x1, y1);
            ctx.line_to(x2, y2);
            ctx.stroke();
        }
    }

    // Draw nodes on top with pulsing based on amplitude and phase.
    let anim_time = opts.anim_time as f64;
    let mut hit_nodes: Vec<BrainVizHitNode> = Vec::with_capacity(nodes.len());
    for node in nodes {
        let Some(&(x, y, z)) = pos.get(&node.id) else {
            continue;
        };
        let p = radius / (z + dist);
        let px = cx + x * p;
        let py = cy + y * p;

        let amp = (node.amp01 as f64).clamp(0.0, 1.0);
        let phase = node.phase as f64;
        let salience = (node.salience01 as f64).clamp(0.0, 1.0);

        // Pulsing effect: ONLY truly active nodes pulse (amp > 0.5 threshold)
        // Higher threshold ensures only genuinely active nodes animate
        let is_active = amp > 0.5;
        let pulse_freq = 0.15; // Pulse frequency (slower for better visibility)
                               // Per-node local animation: each node pulses independently based on phase
        let pulse = if is_active {
            ((anim_time * pulse_freq + phase).sin() * 0.5 + 0.5) * amp
        } else {
            0.0 // No pulsing for inactive nodes
        };

        // Base size depends on node type (REDUCED from original)
        // Unit type classification:
        // - sensor_member: units allocated to sensor groups (input encoding)
        // - group_member: units allocated to action groups (output selection)
        // - reserved: units with learning disabled (sensor/action or special)
        // - regular: freely available units for concept formation
        let type_base = if node.is_sensor_member {
            1.5
        } else if node.is_group_member {
            1.25
        } else {
            1.0
        };

        // Salience adds to base size: more frequently accessed nodes appear larger
        // Salience contribution: up to +1.5 size for max salience (reduced from +3.0)
        let salience_size = 1.1 * salience;

        // Dynamic amplitude contribution (current activity) - reduced from 4.0
        let amp_size = 1.6 * amp;

        // Combined base (type + salience)
        let base = type_base + salience_size;

        // Scale down nodes (and a touch with zoom) to keep density readable.
        // Only add pulsing to size for ACTIVE nodes
        let pulse_size_factor = if is_active { 1.0 + pulse * 0.25 } else { 1.0 };
        let mut size = (base + amp_size) * opts.node_size_scale * pulse_size_factor;
        size *= zoom.sqrt();

        // Cap node size so high-salience / high-amp nodes don't overwhelm the view.
        // Keep the cap proportional to zoom (so zooming in still feels like zoom).
        let max_size = 9.0 * zoom.sqrt();
        size = size.clamp(1.2, max_size);

        // Alpha: salience contributes to base visibility, amplitude adds dynamic
        // Active nodes are more visible, inactive nodes fade into background
        let alpha = if is_active {
            (0.4 + 0.3 * salience + 0.3 * amp) * (0.85 + 0.15 * pulse)
        } else {
            0.25 + 0.25 * salience // Static low visibility for inactive
        };

        // Color nodes based on type AND learning/inference mode
        // Learning mode: warmer tones (orange/amber accents)
        // Inference mode: cooler tones (cyan/blue accents)
        let color = if opts.learning_mode {
            // Learning mode colors - warmer palette
            if node.is_sensor_member {
                format!("rgba(255, 153, 102, {:.3})", alpha) // Warm orange for sensors
            } else if node.is_group_member {
                format!("rgba(74, 222, 128, {:.3})", alpha) // Green for groups
            } else if node.is_reserved {
                format!("rgba(148, 163, 184, {:.3})", alpha) // Gray for reserved
            } else {
                format!("rgba(251, 191, 36, {:.3})", alpha) // Amber for regular units
            }
        } else {
            // Inference mode colors - cooler palette
            if node.is_sensor_member {
                format!("rgba(122, 162, 255, {:.3})", alpha) // Blue for sensors
            } else if node.is_group_member {
                format!("rgba(34, 211, 238, {:.3})", alpha) // Cyan for groups
            } else if node.is_reserved {
                format!("rgba(148, 163, 184, {:.3})", alpha) // Gray for reserved
            } else {
                format!("rgba(167, 139, 250, {:.3})", alpha) // Purple for regular units
            }
        };

        // Draw glow for highly active nodes only (amp > 0.65 threshold)
        // Only nodes with strong activity get the pulsing glow effect
        if amp > 0.65 {
            let glow_alpha = pulse * 0.35 * amp;
            let glow_color = if opts.learning_mode {
                format!("rgba(255, 200, 100, {:.3})", glow_alpha)
            } else {
                format!("rgba(122, 200, 255, {:.3})", glow_alpha)
            };
            ctx.set_fill_style_str(&glow_color);
            ctx.begin_path();
            ctx.arc(px, py, size * 1.8, 0.0, std::f64::consts::PI * 2.0)
                .ok();
            ctx.fill();
        }

        ctx.set_fill_style_str(&color);
        ctx.begin_path();
        ctx.arc(px, py, size, 0.0, std::f64::consts::PI * 2.0).ok();
        ctx.fill();

        hit_nodes.push(BrainVizHitNode {
            id: node.id,
            x: px,
            y: py,
            r: size.max(1.2),
        });
    }

    Ok(hit_nodes)
}

/// Options for causal graph visualization (unified with substrate view)
pub struct CausalVizRenderOptions {
    pub zoom: f32,
    pub pan_x: f32,
    pub pan_y: f32,
    pub rotation: f32,   // Y-axis rotation (horizontal drag)
    pub rotation_x: f32, // X-axis rotation (vertical drag)
    #[allow(dead_code)]
    pub draw_outline: bool,
    pub anim_time: f32,
}

impl Default for CausalVizRenderOptions {
    fn default() -> Self {
        Self {
            zoom: 1.0,
            pan_x: 0.0,
            pan_y: 0.0,
            rotation: 0.0,
            rotation_x: 0.0,
            draw_outline: true,
            anim_time: 0.0,
        }
    }
}

/// Draw causal graph using the same 3D sphere layout as substrate view.
///
/// Nodes are positioned on a golden-spiral sphere, sized by base count (frequency).
/// Edges show temporal causality with directional arrows.
/// The main difference from substrate view is that this shows symbol-level causal relationships.
pub fn draw_causal_graph(
    canvas: &HtmlCanvasElement,
    nodes: &[braine::substrate::CausalNodeViz],
    edges: &[braine::substrate::CausalEdgeViz],
    bg_color: &str,
    opts: CausalVizRenderOptions,
) -> Result<Vec<CausalHitNode>, String> {
    let ctx = canvas
        .get_context("2d")
        .map_err(|_| "get_context failed")?
        .ok_or("no 2d context")?
        .dyn_into::<CanvasRenderingContext2d>()
        .map_err(|_| "cast failed")?;

    let w = canvas.width() as f64;
    let h = canvas.height() as f64;
    let zoom = (opts.zoom as f64).clamp(0.25, 8.0);
    let cx = (w / 2.0) + (opts.pan_x as f64);
    let cy = (h / 2.0) + (opts.pan_y as f64);
    let radius = ((w.min(h) / 2.0) - 22.0) * zoom;

    ctx.set_fill_style_str(bg_color);
    ctx.fill_rect(0.0, 0.0, w, h);

    // Outline removed for cleaner causal view - nodes and edges speak for themselves
    // The outline is hidden by default in causal mode for a more immersive visualization

    if nodes.is_empty() {
        // Draw placeholder text
        ctx.set_fill_style_str("rgba(178, 186, 210, 0.5)");
        ctx.set_font("14px sans-serif");
        ctx.set_text_align("center");
        let _ = ctx.fill_text("No causal symbols yet. Run some trials.", cx, cy);
        return Ok(Vec::new());
    }

    // Golden spiral distribution on sphere (same algorithm as substrate view)
    let n = nodes.len() as f64;
    let golden = 2.39996322972865332_f64; // ~pi*(3-sqrt(5))
    let rot_y = opts.rotation as f64;
    let rot_x = opts.rotation_x as f64;

    // 3D positions for perspective projection
    let mut pos3d: std::collections::HashMap<u32, (f64, f64, f64)> =
        std::collections::HashMap::with_capacity(nodes.len());
    let mut node_sizes: std::collections::HashMap<u32, f64> = std::collections::HashMap::new();

    // Find max base count for normalization
    let max_count = nodes
        .iter()
        .map(|n| n.base_count)
        .fold(0.0f32, f32::max)
        .max(0.001);

    for (i, node) in nodes.iter().enumerate() {
        let i_f = i as f64;
        let y = 1.0 - 2.0 * ((i_f + 0.5) / n);
        let r = (1.0 - y * y).sqrt();
        let theta = golden * i_f;
        let x = r * theta.cos();
        let z = r * theta.sin();

        // Rotate around Y axis (horizontal drag)
        let xr = x * rot_y.cos() + z * rot_y.sin();
        let zr = -x * rot_y.sin() + z * rot_y.cos();

        // Rotate around X axis (vertical drag)
        let yr = y * rot_x.cos() - zr * rot_x.sin();
        let zr2 = y * rot_x.sin() + zr * rot_x.cos();

        pos3d.insert(node.id, (xr, yr, zr2));

        // Size based on base count (normalized) - same scale as substrate
        let norm_count = (node.base_count / max_count) as f64;
        let size = 4.0 + 8.0 * norm_count.sqrt();
        node_sizes.insert(node.id, size);
    }

    // Draw edges first (under nodes) with perspective projection
    let max_strength = edges
        .iter()
        .map(|e| e.strength.abs())
        .fold(0.0f32, f32::max)
        .max(0.001);

    for edge in edges {
        let Some(&(sx, sy, sz)) = pos3d.get(&edge.from) else {
            continue;
        };
        let Some(&(tx, ty, tz)) = pos3d.get(&edge.to) else {
            continue;
        };

        let depth = ((sz + tz) * 0.5 + 1.0) * 0.5; // 0..1 approx
        let norm_strength = (edge.strength.abs() / max_strength).clamp(0.0, 1.0) as f64;
        if norm_strength < 0.08 {
            continue;
        }

        // Perspective projection (same as substrate view)
        let dist = 2.8;
        let sp = radius / (sz + dist);
        let tp = radius / (tz + dist);
        let x1 = cx + sx * sp;
        let y1 = cy + sy * sp;
        let x2 = cx + tx * tp;
        let y2 = cy + ty * tp;

        let alpha = (0.10 + 0.50 * norm_strength) * (0.35 + 0.65 * depth);
        let width = 0.5 + 2.0 * norm_strength;

        // Color: green for positive causal, red for negative (same as substrate)
        if edge.strength >= 0.0 {
            ctx.set_stroke_style_str(&format!("rgba(74, 222, 128, {:.3})", alpha));
        } else {
            ctx.set_stroke_style_str(&format!("rgba(251, 113, 133, {:.3})", alpha));
        }
        ctx.set_line_width(width);

        ctx.begin_path();
        ctx.move_to(x1, y1);
        ctx.line_to(x2, y2);
        ctx.stroke();

        // Draw arrowhead for directionality
        if norm_strength > 0.2 {
            let angle = (y2 - y1).atan2(x2 - x1);
            let arrow_size = 4.0 + 3.0 * norm_strength;
            let target_size = node_sizes.get(&edge.to).copied().unwrap_or(6.0);
            let ax = x2 - (target_size + 2.0) * angle.cos();
            let ay = y2 - (target_size + 2.0) * angle.sin();

            ctx.begin_path();
            ctx.move_to(ax, ay);
            ctx.line_to(
                ax - arrow_size * (angle - 0.35).cos(),
                ay - arrow_size * (angle - 0.35).sin(),
            );
            ctx.move_to(ax, ay);
            ctx.line_to(
                ax - arrow_size * (angle + 0.35).cos(),
                ay - arrow_size * (angle + 0.35).sin(),
            );
            ctx.stroke();
        }
    }

    // Draw nodes on top with perspective projection
    let anim_time = opts.anim_time as f64;
    let mut hit_nodes: Vec<CausalHitNode> = Vec::with_capacity(nodes.len());

    for node in nodes {
        let Some(&(x3d, y3d, z3d)) = pos3d.get(&node.id) else {
            continue;
        };

        // Perspective projection
        let dist = 2.8;
        let p = radius / (z3d + dist);
        let px = cx + x3d * p;
        let py = cy + y3d * p;

        let norm_count = (node.base_count / max_count) as f64;
        let size = node_sizes.get(&node.id).copied().unwrap_or(6.0);

        // Subtle pulsing based on frequency (more frequent = more visible pulse)
        let pulse_freq = 0.12;
        let pulse = ((anim_time * pulse_freq + (node.id as f64) * 0.5).sin() * 0.5 + 0.5)
            * norm_count.sqrt();
        let pulse_size = 1.0 + pulse * 0.15;
        let final_size = size * pulse_size * zoom.sqrt();

        // Alpha based on depth and frequency
        let depth = (z3d + 1.0) * 0.5;
        let alpha = (0.4 + 0.4 * norm_count) * (0.5 + 0.5 * depth) * (0.9 + 0.1 * pulse);

        // Cyan color for causal symbols (distinguishes from substrate blue)
        let color = format!("rgba(34, 211, 238, {:.3})", alpha);

        ctx.set_fill_style_str(&color);
        ctx.begin_path();
        ctx.arc(px, py, final_size, 0.0, std::f64::consts::PI * 2.0)
            .ok();
        ctx.fill();

        hit_nodes.push(CausalHitNode {
            id: node.id,
            name: node.name.clone(),
            x: px,
            y: py,
            r: final_size.max(3.0),
            base_count: node.base_count,
        });
    }

    Ok(hit_nodes)
}

/// Hit test node for causal graph visualization
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct CausalHitNode {
    pub id: u32,
    pub name: String,
    pub x: f64,
    pub y: f64,
    pub r: f64,
    pub base_count: f32,
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
    ctx.arc(cx, cy, radius, std::f64::consts::PI, end_angle)
        .ok();
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
        ctx.set_fill_style_str(if is_highlight {
            "rgba(251, 191, 36, 1.0)"
        } else {
            "rgba(170, 180, 230, 0.9)"
        });
        let _ = ctx.fill_text(name, label_width - 5.0, y + bar_height * 0.7);

        // Bar background
        ctx.set_fill_style_str("rgba(122, 162, 255, 0.1)");
        ctx.fill_rect(label_width, y, bar_area, bar_height);

        // Bar value
        let bar_w = bar_area * (*score as f64).clamp(0.0, 1.0);
        let bar_color = if is_highlight {
            "rgba(251, 191, 36, 0.8)"
        } else {
            "rgba(122, 162, 255, 0.6)"
        };
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
        let depth_a = if a.is_sensor_member {
            3
        } else if a.is_group_member {
            2
        } else if a.is_reserved {
            0
        } else {
            1
        };
        let depth_b = if b.is_sensor_member {
            3
        } else if b.is_group_member {
            2
        } else if b.is_reserved {
            0
        } else {
            1
        };
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
        let (color, glow_color) = if point.is_sensor_member {
            ("rgba(122, 162, 255, 0.9)", "rgba(122, 162, 255, 0.4)")
        } else if point.is_group_member {
            ("rgba(74, 222, 128, 0.85)", "rgba(74, 222, 128, 0.3)")
        } else if point.is_reserved {
            ("rgba(136, 136, 136, 0.6)", "rgba(136, 136, 136, 0.2)")
        } else {
            ("rgba(251, 191, 36, 0.75)", "rgba(251, 191, 36, 0.25)")
        };

        // Draw glow effect for foreground units
        if point.is_sensor_member || point.is_group_member {
            ctx.set_fill_style_str(glow_color);
            ctx.begin_path();
            ctx.arc(x, y, base_size * 2.0, 0.0, std::f64::consts::PI * 2.0)
                .ok();
            ctx.fill();
        }

        // Draw 3D-style sphere (gradient circle)
        // Outer darker ring
        ctx.set_fill_style_str(color);
        ctx.begin_path();
        ctx.arc(x, y, base_size, 0.0, std::f64::consts::PI * 2.0)
            .ok();
        ctx.fill();

        // Inner highlight for 3D effect
        let highlight_color = if point.is_sensor_member {
            "rgba(180, 200, 255, 0.6)"
        } else if point.is_group_member {
            "rgba(140, 255, 180, 0.5)"
        } else if point.is_reserved {
            "rgba(180, 180, 180, 0.5)"
        } else {
            "rgba(255, 220, 140, 0.5)"
        };
        ctx.set_fill_style_str(highlight_color);
        ctx.begin_path();
        ctx.arc(
            x - base_size * 0.25,
            y - base_size * 0.25,
            base_size * 0.5,
            0.0,
            std::f64::consts::PI * 2.0,
        )
        .ok();
        ctx.fill();
    }

    // Draw count label
    ctx.set_fill_style_str("rgba(122, 162, 255, 0.8)");
    ctx.set_font("12px system-ui, sans-serif");
    ctx.set_text_align("right");
    let _ = ctx.fill_text(
        &format!("{} units", points.len()),
        w - padding,
        padding - 8.0,
    );

    Ok(())
}
