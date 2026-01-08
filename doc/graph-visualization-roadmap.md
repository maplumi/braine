# Graph visualization: current behavior + roadmap

## What it is today

### Layout
- **Force-directed (2D).**
- Node positions are relaxed with a lightweight force-directed step each refresh and persisted across refreshes so the layout stabilizes over time.
- Edges are rendered as straight lines (not orthogonal segments).
- A **layout toggle** is available: Force vs Circle.

### Interaction
- Scroll-wheel zoom (centered zoom) and a simple zoom slider.
- Drag-to-pan for navigating zoomed-in views.

### What nodes/edges mean
- **Substrate graph**
  - Node = substrate unit (label like `u123`)
  - Edge = substrate neighbor connection (synapse)
  - Node `value` = unit amplitude (as sent by daemon)
  - Edge `weight` = connection weight (can be positive or negative)
- **Causal graph**
  - Node = symbol in causal/meaning memory (stimulus keys, pairs, actions)
  - Edge = learned association weight
  - Node label = symbol string

These are intentionally **distinct graphs**: substrate edges are physical connectivity, while causal edges are symbolic association/credit structure. Overlaying them in one view is possible but usually confusing without heavy visual encoding.

### Current rendering cues
- Edge color is based on sign:
  - **Green**: `weight >= 0`
  - **Red**: `weight < 0`
- Edge opacity is based on `abs(weight)` relative to the maximum edge weight in the snapshot.
- Node color is domain-aware (minimal palette): `pos_x_*` and `pos_y_*` are highlighted; other domains use the default node color.
- **Hover-only tooltip** shows label + domain + value.
- A tiny inline legend explains edge sign/strength and domain colors.

## What you asked for (to track)
1. Add richer graph encodings (legend/keys, color gradients for edge strengths, etc.)
2. Domain-aware labeling (e.g., `pos_x_*`, `pos_y_*`, `pair::*`, `action::*`)
3. Force-directed graphs (2D; optionally explore 3D)
4. Axes/value display where it makes sense (e.g., Brain Plot)

## Roadmap (minimal, highest value first)

### 1) Improve readability without changing layout
- Add a small **legend**: edge sign (green/red) + strength mapping.
- Encode edge **strength** by both:
  - opacity (already), and
  - **thickness** (new; still minimal)
- Make tooltip domain-aware:
  - classify causal labels into domains (stimulus/pair/action/pos_x/pos_y/other)

### 2) Domain coloring
- Color nodes by domain/type (especially in causal graph):
  - `pos_x_*` / `pos_y_*`
  - `pair::*`
  - `action::*`
  - everything else
- Keep labels hover-only to avoid clutter.

### 3) Force-directed layout (2D)
- Add a **layout toggle** (circle vs force-directed).
- Use a lightweight 2D simulation with a fixed iteration budget per refresh.
- Keep edge rendering straight-line for force-directed mode (simpler than orthogonal routing).

### 4) 3D exploration (optional)
- True 3D + labels likely requires a rendering layer (e.g., `wgpu`) and camera controls.
- A safer intermediate step is “pseudo-3D”: 2D force-directed with depth as color/size, but still rendered in 2D.

## Notes on axes
- For graphs (node-link diagrams), axes are usually misleading because positions come from a layout algorithm.
- Axes are valuable for true plotted quantities (Brain Plot: age vs amplitude), where we should add ticks/labels.
