# Game testing guide

This guide describes quick, reproducible ways to evaluate learning behavior across the daemon-run games.

The recommended workflow is:

1) Run the daemon + desktop UI.
2) Use a fresh brain (or load a known snapshot).
3) Run each game in **Braine** mode for a fixed number of trials.
4) Use `runtime.json` (and the UI HUD) to compare outcomes across runs.

## Run

- Daemon: `cargo run -p brained`
- UI: `cargo run -p braine_desktop`
- CLI (optional): `cargo run --bin braine-cli -- status`

## Where to read metrics

- The desktop UI shows per-game correctness + recent window stats.
- The daemon persists structured stats to `runtime.json` in the daemon data directory.
  - Get the directory via: `cargo run --bin braine-cli -- paths`

Example (Linux default path):

- `~/.local/share/braine/runtime.json`

Pretty print:

- `jq . ~/.local/share/braine/runtime.json`

## Suggested test protocol (minimal)

Use consistent settings across games when comparing:

- Fix `Trial ms` (except Pong; see below)
- Fix exploration/meaning settings if exposed
- Run each game for 200–500 trials

### Pong special note

Pong is particularly sensitive to control cadence. If hit-rate looks capped, try `Trial ms` in the 40–120ms range first (then adjust paddle/ball speeds).

See [pong-performance.md](pong-performance.md).

## Automated sweep (non-interactive)

For repeatable validation across the daemon games, use the sweep script:

```bash
bash scripts/game_learning_sweep.sh /tmp/game_learning_validation.log
cat /tmp/game_learning_validation.log
```

By default, the script will try to start `brained` if it can’t connect via the CLI.
You can also force a restart (useful when iterating on daemon code):

```bash
RESTART_DAEMON=1 bash scripts/game_learning_sweep.sh /tmp/game_learning_validation.log
```

Per-game durations can be overridden via env vars:

```bash
SPOT_SECS=15 BANDIT_SECS=15 SPOT_REV_SECS=30 SPOTXY_SECS=20 MAZE_SECS=180 PONG_SECS=180 TEXT_SECS=60 REPLAY_SECS=30 \
  bash scripts/game_learning_sweep.sh /tmp/game_learning_validation.log
```

For a strictly non-interactive/remote run, start the daemon detached and keep `RESTART_DAEMON=0`:

```bash
nohup cargo run -p brained >/tmp/brained.log 2>&1 &
RESTART_DAEMON=0 bash scripts/game_learning_sweep.sh /tmp/game_learning_validation.log
cargo run --bin braine-cli -- shutdown
```

## Game checklists

### Spot

- Expect accuracy to rise above chance within tens of trials.
- Verify stability: once learned, performance should remain high without frequent regressions.

### Bandit

- Expect a preference to emerge for the better arm.
- Under drift, expect adaptation rather than permanent lock-in.

### Spot Reversal

- Expect a drop after reversal and recovery as the brain conditions on context.
- Use this to measure flexibility vs habit strength.

### SpotXY

- Increase grid size gradually (if using the UI controls).
- Use **eval/holdout mode** to confirm performance without new learning writes.

### Pong

- Ensure the task is controllable (cadence/speeds).
- Track whether learned behavior generalizes across ball_y/paddle_y bins.

## Experts (child brains) evaluation

If experts are enabled, treat them as a “novelty/shift sandbox” mechanism:

- Keep a baseline run with experts disabled.
- Enable experts and compare: recovery after reversals/drift and stability of the parent.
- When testing holdout behavior, prefer SpotXY eval mode (learning suppressed) to isolate inference from learning.
