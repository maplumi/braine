# Web vs Desktop feature parity

This repo currently has **two different interaction architectures**:

- **Desktop app (daemon + UI/CLI)**: `brained` owns the long-lived `Brain` and exposes state + commands via newline-delimited JSON over TCP (`127.0.0.1:9876`).
- **Web app (WASM)**: `braine_web` runs a `Brain` *in-process* in the browser and persists via IndexedDB.

That means “feature parity” is not just UI work: many desktop features depend on the daemon’s long-lived state and the daemon’s protocol snapshots.

## Parity matrix

| Area | Desktop (daemon + Slint UI + CLI) | Web (WASM / Leptos) |
|---|---|---|
| Runtime ownership | Daemon process owns the `Brain` | Browser tab owns the `Brain` |
| Transport/protocol | Newline-delimited JSON over TCP (`Request`/`Response`) | No daemon bridge (local-only) |
| Persistence | OS data-dir brain image + snapshots | IndexedDB save/load + `.bbi` import/export |
| “Stop-first” game switching enforcement | Yes (daemon enforces) | N/A (local switch in app) |
| Game selection | Daemon games + UI controls | Spot/Bandit/SpotReversal/SpotXY/Pong (plus Sequence added in this repo) |
| Trial cadence | Daemon loop (trial ms, run/stop) | Local loop (trial ms, run/stop) |
| SpotXY eval/holdout mode | Yes (suppresses learning writes) | Yes (suppresses learning writes) |
| Graph visualization (nodes/edges) | Yes (daemon snapshot-driven) | Yes (local BrainViz: substrate + causal views) |
| Brain plots (time series) | Yes | Partial (reward + action-choice traces; sampled unit plot) |
| Meaning inspection | Yes | Partial (limited UI; no deep edge browser yet) |
| Storage/snapshots UI | Yes | Yes (IndexedDB + `.bbi` import/export) |
| Expert/child-brain controls | Yes (policy + nesting knobs) | Not implemented |
| CLI automation | Yes (`braine-cli`) | Not applicable |

## Notes on “missing” web features

The web app can gain parity in two different ways:

1) **Local parity**: re-implement visualizations (graph/meaning/plots) directly over the in-process `Brain` state.
2) **Remote parity**: implement a daemon client in the web app (WebSocket/WebTransport/HTTP bridge) and re-use the daemon’s snapshots and control protocol.

Right now `braine_web` is optimized for **edge-first learning in a browser** (fast iteration + IndexedDB persistence). It includes lightweight local observability (BrainViz + a few charts), while the desktop stack remains the most complete introspection surface.

## Recommended next parity steps

- Add a lightweight “inspect” panel in web (top-k meaning edges; basic unit activity stats).
- Add a graph snapshot endpoint in the daemon that can be streamed efficiently, then decide whether the web should consume it (requires a web-safe transport).
- Keep desktop as the “deep introspection” client; keep web as the “portable learning demo” client.
