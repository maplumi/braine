# Dashboard (desktop + web) — MVP spec

This document defines a minimal dashboard for **observing learning** in Braine.

The goal is not a “training UI”; it’s a lightweight readout of:
- whether the substrate is learning (writes are happening)
- whether performance is improving for the active task
- whether the substrate is stable (not saturating / not pruning too aggressively)

## Data sources

### Desktop (daemon-backed)
- Live state via the daemon TCP protocol (`127.0.0.1:9876`), typically polled.
- Persisted snapshot in `runtime.json` (daemon data dir) for post-run inspection.

### Web (in-browser)
- Live state is in-process (WASM `Brain`).
- Persistence is typically IndexedDB.
- MVP should store the *last known* dashboard samples in browser storage so users can compare sessions.

## MVP widgets

### 1) Global status header
- Connection (connected/disconnected, last error)
- Active game + running/paused
- Trial cadence (`trial_period_ms`) and effective FPS (if available)
- Current neuromodulator value

### 2) Learning HUD (per active game)
Show:
- trials, correct, incorrect
- accuracy, recent_rate, last_100_rate
- learning milestones: learning/learned/mastered trial indices

Interpretation:
- `last_100_rate` is the best quick indicator of “is it improving right now?”
- For sparse terminal games (Maze), `trials` grows slowly and windows stabilize late.

### 3) Brain health (substrate diagnostics)
Show:
- unit_count, connection_count, memory_bytes
- avg_weight, saturated
- pruned_last_step, births_last_step
- causal edges + recent causal update counts

Interpretation:
- sustained `saturated=true` suggests capacity pressure (expect neurogenesis or stalled learning)
- frequent high pruning can indicate instability or overly aggressive forgetting

### 4) Per-game table (cross-task memory)
Use the persisted-per-game stats (daemon runtime keeps last-known stats per game kind):
- game kind
- last_100_rate
- trials
- learned/mastered markers

This answers: “did the same brain learn multiple tasks?”

## Nice-to-have (post-MVP)

- Sparklines for `last_100_rate` and/or reward over time
- Expert/child-brain surface: which context spawned an expert, active expert performance, consolidation events
- Meaning-gap monitors (pair vs global gap history) to show whether causal meaning is separating correct vs wrong actions

## Minimal API surface (what the dashboard needs)

For daemon-backed clients, the dashboard only needs:
- a **Status** response containing:
  - HUD stats (the common metrics)
  - brain diagnostics
  - active game kind + a few game-specific state fields

Protocol reference:
- [doc/architecture/daemon-protocol.md](../architecture/daemon-protocol.md)

Tip: the protocol doc includes a “Copy/paste examples (NDJSON)” section that’s convenient for quickly testing requests with `nc`.

For the web app, mirror the same struct shape so the UI can be shared conceptually (even if the internal sources differ).

## Related docs

- Per-game metric meanings: [doc/games/metrics-per-game.md](../games/metrics-per-game.md)
- General metric philosophy: [doc/development/metrics.md](metrics.md)
