# SpotXY Parameters and WebGPU Status

## Summary
This document covers recent changes to harmonize SpotXY game parameter schemas and clarify GPU usage in the web version.

## SpotXY Parameter Schema

### Protocol Extension
Extended the daemon's `GetGameParams` / `SetGameParam` protocol to include SpotXY parameters:

- **`grid_n`**: Controls grid size (0–8)
  - 0 = binary mode (left/right only)
  - 2–8 = NxN grid size
  - Default: 0
- **`eval`**: Evaluation/holdout mode (0–1, boolean)
  - 0 = training mode (learning enabled)
  - 1 = evaluation mode (dynamics run, learning suppressed)
  - Default: 0

### Implementation Details
- **Daemon** (`crates/brained/src/main.rs`):
  - `Request::GetGameParams` returns schema for "pong" and "spotxy" games
  - `Request::SetGameParam` interprets SpotXY params:
    - `grid_n`: calls `increase_grid`/`decrease_grid` to reach target size
    - `eval`: calls `set_eval_mode(value >= 0.5)`
- **Desktop UI** (`crates/braine_desktop/`):
  - Polling loop applies SpotXY schema to Slint properties
  - Added `spotxy-grid-n-min/max/default` properties for future slider controls
- **CLI** (`crates/core/src/bin/braine_cli.rs`):
  - Protocol already supports viewing/setting SpotXY params

## GPU Usage

### Current Status
- **Native `braine` core**:
  - Optional `gpu` feature (disabled by default) enables `wgpu` compute shaders
  - Targets substrates with 10k+ units for performance gains
  - Uses `pollster::block_on` for synchronous GPU initialization
  - **Not compatible with WASM** (requires `std` + blocking APIs)

- **`braine_web` (WASM)**:
  - **CPU-only** by default (uses `Scalar` execution tier)
  - No GPU acceleration currently enabled
  - Runtime WebGPU detection added:
    - Checks `navigator.gpu` availability on startup
    - Displays status message in UI: "WebGPU: available (not yet used)" or "not available (CPU only)"
  - Canvas rendering uses 2D context (`CanvasRenderingContext2d`), not WebGL/WebGPU

### Future WebGPU Integration
To enable WebGPU in `braine_web`:
1. Adapt `wgpu` initialization for WASM (async via `wasm-bindgen-futures`)
2. Use WASM32 backend features for `wgpu` (no `pollster::block_on`)
3. Optionally integrate WebGPU for canvas rendering (e.g., for visualizations)

See [doc/deployment-web.md](deployment-web.md) "Future Work" section for details.

## Deployment
Comprehensive deployment documentation created in [doc/deployment-web.md](deployment-web.md), covering:
- GitHub Pages hosting
- Build/optimization steps
- Alternative hosting options (Netlify, Vercel, self-hosted)
- Browser compatibility notes

## Related Documentation
- [Interaction Patterns](interaction.md) - Context-conditioned symbols and observation semantics
- [Architecture](architecture.md) - Core substrate design
- [Web Deployment Guide](deployment-web.md) - Detailed deployment instructions
