# WASM edge hosting feasibility (Leptos UI)

Goal: a browser-hosted “edge” version of the visualizer where:
- the **Brain** runs inside the **WASM process** (no daemon)
- learning happens locally (“on edge”)
- state persists to browser storage (IndexedDB / localStorage)

This document evaluates feasibility and proposes a plan. It does **not** implement it.

## Summary
Feasible with a few constraints:
- The core `braine` crate is already structured for **`no_std`** via `#![no_std]` when `std` feature is disabled.
- Persistence today (`Brain::save_image_to` / `load_image_from`) is **std-only** and uses `Read`/`Write`. For WASM we need either:
  1) a **memory-based** encode/decode API (`Vec<u8>`) that works with `alloc`, or
  2) keep using the current `Read`/`Write` API but implement it over in-memory buffers (still requires `std`-like I/O traits, which are unavailable in pure `no_std`).

So the main work is to add a **WASM-friendly persistence surface** (bytes-in/bytes-out), and then build a UI around it.

## Architecture options

### Option A (recommended): Leptos UI + in-process Brain
- New workspace member: `braine_web` (Leptos, `wasm32-unknown-unknown`).
- `braine` compiled with `default-features = false` (no `std`).
- The web app owns:
  - `Brain` instance
  - game logic (Pong/Bandit/etc.)
  - a render loop (canvas/webgl) + UI controls
- Persistence:
  - serialize `Brain` to bytes
  - store in IndexedDB (preferred) or localStorage (only for tiny brains)

Pros:
- True edge learning; no TCP; simplest story.
Cons:
- Duplicates game logic from daemon unless we refactor games into a shared crate.

### Option B: Browser UI + daemon as “edge service worker”
- Not actually feasible as a service worker cannot host native Rust/Tokio daemon.
- You’d still be running a server somewhere; not the goal.

### Option C: WASM UI embedding the existing daemon protocol
- Emulate the daemon protocol inside the browser (no networking), so the UI can reuse protocol mapping.
- Still requires moving substantial daemon logic into a library.

## WASM constraints you must plan for

### 1) No threads by default
- Rayon/parallel paths are not available unless you opt into web workers + `wasm-bindgen-rayon`.
- Start single-thread.

### 2) File system is not available
- Use browser storage:
  - IndexedDB for large binary blobs (`brain.bbi` bytes)
  - localStorage only for small settings

### 3) Time sources
- `Instant` is supported in WASM, but be careful with determinism.
- For training determinism, the sim should use fixed dt, not wall clock.

### 4) `std` I/O traits
- Current persistence APIs are `std`-only.
- For WASM, add something like:
  - `Brain::save_image_bytes() -> Vec<u8>`
  - `Brain::load_image_bytes(bytes: &[u8]) -> io::Result<Brain>` (or custom error type in no_std)

### 5) Serialization format choice
- The new v2 compressed brain image is binary and friendly to storage as a blob.
- Ensure the compression crate works on wasm (pure Rust). The chosen `lz4_flex` does.

## Plan (high-level)

### Phase 0: Feasibility spike
- Compile `braine` with `default-features = false` for `wasm32-unknown-unknown`.
- Add byte-based persistence API (alloc-only).
- Prove storing/loading bytes from IndexedDB.

### Phase 1: Minimal demo
- Implement one game (Pong) as a web canvas.
- Mirror essential controls: Start/Stop, Trial period, Learning on/off, Reset.
- Display a small HUD (trials, recent accuracy, neuromod, etc.).

### Phase 2: Feature parity subset
- Add Bandit + Spot.
- Add sampled plots (oscillation / unit plot) but keep them lightweight.

### Phase 3: Optional performance
- Consider web workers for stepping + rendering decoupling.

## Risks / open questions
- Avoiding duplicated game logic: ideally refactor games to a shared crate used by both `brained` and `braine_web`.
- Persistence size: large brains will need IndexedDB and compression; localStorage will not work.
- GPU feature: current `wgpu` path is not automatically viable in browser; treat as future.

## Recommendation
Proceed: it’s feasible and aligns with the “learn on edge” goal. Start with Pong only, bytes-based persistence, and IndexedDB.
