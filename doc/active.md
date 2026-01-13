# Active Development Notes

## Idle Dreaming & Sync System (January 2026) ✅

### Overview

Implemented actual (not just visual) dreaming and synchronization operations that run when the brain is idle. This enables background memory consolidation and phase alignment for improved learning outcomes.

### Core API Additions (`crates/core/src/core/substrate.rs`)

```rust
// Activity detection
pub fn is_active(&self, threshold: f32) -> bool;
pub fn active_unit_count(&self, threshold: f32) -> usize;
pub fn is_learning_mode(&self) -> bool;  // neuromod > 0.3
pub fn is_inference_mode(&self) -> bool; // neuromod <= 0.3

// Idle operations
pub fn idle_dream(&mut self, steps: usize, activity_threshold: f32) -> usize;
pub fn global_sync(&mut self) -> usize;
pub fn idle_maintenance(&mut self, force: bool) -> Option<usize>;
```

### Behavior

**Global Sync** (one-time when entering idle):
- Aligns all unit phases to a common reference (phase = 0)
- Only runs once per idle period
- Skipped if brain is in high-learning mode (neuromod > 0.3)
- Creates coherent oscillation for clean restart

**Idle Dreaming** (periodic while idle):
- Runs every ~3 seconds while brain is not running
- Targets only inactive unit clusters (amplitude < 0.25)
- Gentle learning boost (1.5x) with moderate neuromodulator (0.5)
- Injects small noise to trigger memory reactivation
- Consolidates synaptic weights in unused areas
- Skipped if learning mode is enabled

### Web App Integration (`crates/braine_web/src/web.rs`)

The idle maintenance loop runs at ~30fps and:
1. Increments visual animation time (always)
2. When running: resets idle state
3. When idle + not learning:
   - Runs `global_sync()` once on entering idle
   - Runs `idle_maintenance()` every 90 ticks (~3 sec)
4. Tracks status: "sync: phases aligned" or "dreaming: N units consolidated"

### Signals Added

```rust
let (idle_sync_done, set_idle_sync_done) = signal(false);
let (idle_dream_counter, set_idle_dream_counter) = signal(0u32);
let (idle_status, set_idle_status) = signal("".to_string());
```

### Design Rationale

- **Sync once**: Prevents phase drift during idle, but too-frequent sync would erase learned phase relationships
- **Dream continuously**: Inactive clusters benefit from ongoing consolidation; active learning areas are protected
- **Learning mode gate**: High neuromodulator indicates active learning in progress; dreaming would interfere
- **Inference mode safe**: Low neuromodulator means read-only behavior; safe to consolidate inactive regions

---

## Node-Edge Dynamics Enhancement Plan (January 2026)

### Overview

Implement visual and core enhancements for node/edge dynamics based on the analysis in [node-edge-dynamics-analysis.md](node-edge-dynamics-analysis.md).

### Phase 1: Visualization Enhancements ✅ (Completed)

1. ✅ Manual rotation for BrainViz (drag to rotate)
2. ✅ Activity-based vibration from amplitude
3. ✅ Dynamic node coloring (learning=warm, inference=cool palette)
4. ✅ Causal graph visualization with directed edges
5. ✅ Controls hint: "Drag to rotate | Shift+drag to pan | Scroll to zoom"

### Phase 2: Visual Enhancements (Planned)

1. **Force-directed layout for causal view**
   - Use causal_strength for edge attraction
   - Simple damped spring simulation
   - No core changes required
   - Priority: HIGH

2. ✅ **Node size based on activity/frequency**
   - In substrate view: size based on amplitude (and now re-samples periodically so node sizes update live while running/idle)
   - In causal view: already done (base_count)

3. ✅ **Per-edge color gradient**
   - Edges use a canvas linear gradient between endpoint node colors
   - Improves readability vs sign-only coloring

### Phase 3: Core Enhancements (Research)

1. **Add `salience` field to Unit struct**
   ```rust
   pub struct Unit {
       // ... existing fields ...
       pub salience: f32,  // Accumulated importance [0, 1]
   }
   ```
   - Update: `salience = α * salience + (1-α) * (amp * |neuromod|)`
   - α = 0.99 for slow accumulation
   - Priority: MEDIUM
   - Impact: +4 bytes per unit, BBI format version bump

2. **Track last_active_step per unit**
   - Enables recency-based visualization
   - Low overhead (one write per activation)
   - Priority: LOW

3. **Age-dependent decay curves**
   - `decay_rate = base_decay / (1 + log(edge_age))`
   - Makes older memories more resistant (consolidation)
   - Priority: RESEARCH

### Implementation Notes

- Phase 2 changes are visualization-only (no core modifications)
- Phase 3 requires BBI format version bump (currently v2)
- All enhancements should follow existing patterns in substrate.rs

---

## Recent Web UI Improvements (January 2026)

### Unit Plot & BrainViz Interactivity

**Status: Working as designed**

- Both visualizations update when simulation steps change
- They sample from the brain on each render via `refresh_ui_from_runtime()`
- Update cycle: `do_tick()` → `refresh_ui_from_runtime()` → `set_unit_plot.set(...)` → Effect re-renders canvas
- The Unit Plot samples 128 units evenly spaced by ID for stable visualization
- BrainViz only updates when the Analytics panel is visible (performance optimization)

### BrainViz View Switching

**Status: Fully implemented**

- Substrate/Causal dropdown toggle in BrainViz panel
- **Substrate view**: Shows unit nodes and sparse connection edges
  - Node color indicates learning mode (warm) vs inference mode (cool)
  - Edge thickness based on connection weight
- **Causal view**: Shows symbol-to-symbol temporal edges
  - Node size based on base_count (frequency)
  - Edge color: green=positive causal, red=negative
  - Directed edges with arrowheads
  - Symbol name labels below nodes

### BrainViz Interaction

**Status: Implemented**

- **Drag**: Rotate the visualization (Y-axis rotation)
- **Shift+Drag**: Pan the view
- **Scroll**: Zoom in/out
- **Hover**: Show node details tooltip
- Activity-based vibration from average amplitude of sampled units
- Reset View button resets zoom, pan, and rotation

### Persistence / Brain Image (IndexedDB)

**Status: Implemented**

- Web autoloads `brain_image` from IndexedDB (DB: `braine`, store: `kv`) on startup when present
- Autosaves periodically when the brain is marked dirty (best-effort, non-blocking)
- UI shows source + autosave state + last-save timestamp

### Pong Visual Polish

**Status: Implemented**

- Removed in-canvas glow effects and removed the remaining outer “ambient glow” around the Pong canvas container

### Statistics Tab Improvements

**Status: Implemented**

Added the following statistics:
- **Brain Age**: `brain.age_steps()` - total simulation steps
- **Avg Amplitude**: `diagnostics.avg_amp` - mean unit amplitude
- **Pruned Count**: `diagnostics.pruned_last_step` - connections pruned last step
- **Causal Memory**: `brain.causal_stats()` returns:
  - `base_symbols`: Number of recorded symbols
  - `edges`: Number of temporal/causal edges

Refresh mechanism:
- All stats updated in `refresh_ui_from_runtime()` which is called after every tick
- Stats tab shows live values when visible

### WebGPU Dynamic Switching

**Status: Enabled when available (feature-gated)**

- WebGPU availability is detected via `navigator.gpu` check
- When built with the web `gpu` feature, the brain will request the GPU execution tier when WebGPU is present
- If WebGPU is unavailable (or initialization fails), it falls back to CPU automatically
- Current GPU acceleration targets the dense dynamics update; learning/plasticity updates remain CPU

### About Page Reorganization

**Status: Implemented**

- About is now the first tab in left navigation panel (landing page)
- Left panel About includes:
  - Version info (Braine Core, BBI format, Braine Web)
  - "What is this?" description
  - Key Principles (learning modifies state, inference uses state, closed loop)
  - Quick Start guide
  - "Start Playing" button
- Removed About tab from right panel dashboard tabs
- Default right panel tab changed to Learning (was GameDetails)

### Version Display

**Status: Implemented**

Constants defined:
- `VERSION_BRAINE`: From `CARGO_PKG_VERSION` (e.g., "0.3.0")
- `VERSION_BRAINE_WEB`: "0.1.0" (hardcoded)
- `VERSION_BBI_FORMAT`: 2 (brain image format version)

Displayed in left panel About page.

### Responsive Mobile Layout

**Status: Implemented**

Added CSS media queries for screens < 768px:
- Dashboard panel becomes a slide-in drawer from right edge
- Toggle button (◀) visible on right edge of screen
- Overlay click-to-close for drawer
- Navigation tabs scroll horizontally
- Reduced button/input sizes for touch
- Header wraps on small screens

CSS classes added:
- `.dashboard-toggle` - Drawer open button
- `.dashboard-overlay` - Click-outside-to-close backdrop
- `.dashboard.open` - Drawer visible state

### Reserved Nodes

**Status: Not shown because none exist**

- The default brain configuration doesn't call `set_reserved()` on any units
- `unit_plot_points()` and `draw_unit_plot_3d()` properly support reserved nodes (gray color)
- BrainViz also supports reserved nodes with proper coloring
- Reserved nodes would appear if a brain image with reserved units is loaded

### Web-only Persistence, BrainViz, Pong Polish ✅

**Status: Implemented**

- **IndexedDB autoload + autosave** for the brain image, including UI indicators for brain source / autosave / last save
- **BrainViz live node refresh** while edges update (sampling refresh tied to runtime activity)
- **BrainViz connection edges** render with per-edge gradients derived from endpoint node colors
- **Pong visuals** aligned with sim collisions, paddle rendered on the left wall, and all remaining glow removed (in-canvas and DOM)
- **Tooltips** added for Trial timing / ε / α and Settings fields
- **WASM build fix**: removed `std` hashing from `connections_fingerprint()` to keep `wasm32-unknown-unknown` compatible

### Default Values Changed

- `brainviz_zoom`: Changed from 1.0 to 1.5 (zoomed in more by default)
- `dashboard_tab`: Changed from `GameDetails` to `Learning`
- `DashboardTab::About`: Removed from enum

## Architecture Notes

### Signal Flow for Unit Plot Updates

```
do_tick()
  └─> runtime.update_value(|r| r.tick(&cfg))
  └─> set_steps.update()
  └─> refresh_ui_from_runtime()
        └─> brain.unit_plot_points(128)
        └─> set_unit_plot.set(plot_points)
              └─> Effect reacts to unit_plot.get()
                    └─> charts::draw_unit_plot_3d()
```

### Responsive Layout Strategy

Desktop (>= 769px):
- Two-column grid: game area (left) + dashboard (right)
- Dashboard always visible

Mobile (< 768px):
- Single-column layout
- Dashboard slides in from right on toggle
- Overlay prevents interaction with game while dashboard open
