# Active Development Notes

## Recent Web UI Improvements (January 2026)

### Unit Plot & BrainViz Interactivity

**Status: Working as designed**

- Both visualizations update when simulation steps change
- They sample from the brain on each render via `refresh_ui_from_runtime()`
- Update cycle: `do_tick()` → `refresh_ui_from_runtime()` → `set_unit_plot.set(...)` → Effect re-renders canvas
- The Unit Plot samples 128 units evenly spaced by ID for stable visualization
- BrainViz only updates when the Analytics panel is visible (performance optimization)

### BrainViz View Switching

**Status: UI toggle implemented, causal view rendering pending**

- Added substrate/causal dropdown toggle in BrainViz panel
- Default view: Substrate (shows unit nodes and connection edges)
- Causal view: Shows "coming soon" message; would visualize symbol-to-symbol temporal edges from causal memory
- Reference: Desktop Slint implementation uses `graph-kind` toggle between "substrate" and "causal" in main.slint

### Vibrational Frequency Visualization

**Status: Not explicitly visualized**

- Desktop doesn't have explicit frequency visualization
- Phase/amplitude information is shown via the sphere visualization in BrainViz
- Units have oscillating phase values; amplitude determines visibility/size
- Could potentially add frequency spectrum or phase coherence visualization in future

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

**Status: Detected but not used**

- WebGPU availability is detected via `navigator.gpu` check
- Currently displays status: "WebGPU: available (not yet used by braine)" or "WebGPU: not available (CPU only)"
- Brain core uses CPU Scalar tier always
- Dynamic switching not possible without architectural changes to the core substrate
- Future: Could implement WebGPU backend for parallel unit updates

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
