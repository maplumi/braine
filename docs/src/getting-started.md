# Getting Started

This repo has three ways to run Braine, depending on what you’re trying to do:

- **Daemon + clients (recommended for long-running learning):** run `brained`, then connect via the desktop UI or CLI.
- **Web UI (in-browser):** run `braine_web` via Trunk; the brain runs in-process in the browser.
- **CLI (automation/debug):** send newline-delimited JSON requests to the daemon.

## Quick start (daemon + desktop UI)

1. Start the daemon:

```bash
cargo run -p brained
```

2. In another terminal, start the desktop UI:

```bash
cargo run -p braine_desktop
```

The desktop UI talks to the daemon over TCP `127.0.0.1:9876` using newline-delimited JSON.

## Quick start (CLI)

With the daemon running:

```bash
cargo run --bin braine-cli -- status
```

If you want to speak the protocol directly, see [Daemon protocol](architecture/daemon-protocol.md).

## Quick start (web UI)

From the repo root:

```bash
cd crates/braine_web
trunk serve --features gpu
```

Then open the URL Trunk prints (usually `http://127.0.0.1:8080/`).

Notes:
- `--features web` enables the real WASM UI.
- `--features gpu` implies `web` and will use WebGPU when available, falling back to CPU.

## Docs

- Published docs (GitHub Pages): `https://maplumi.github.io/braine/docs/`
- Local docs build:

```bash
mdbook build docs
```

## What to read next

- [How It Works](overview/how-it-works.md) for the conceptual model.
- [Architecture](architecture/architecture.md) for the system boundary (daemon/UI/CLI) and substrate layout.
- [Visualizer games](games/visualizer-games.md) to understand the built-in assays.
