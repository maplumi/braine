# Games (daemon + desktop UI)

This repo’s interactive games are run by the `brained` daemon and controlled/inspected via the `braine_desktop` Slint UI.

## Run

- Start daemon: `cargo run -p brained`
- Start UI: `cargo run -p braine_desktop`

Pick a game in the UI, then choose **Human** or **Braine** control mode.

## Persistence and stats

- Brain image: `braine.bbi` in the daemon data dir
- Runtime stats: `runtime.json` in the same directory
- Print paths: `cargo run --bin braine-cli -- paths`

---

## Spot

**Task:** discriminate left vs right spot.

**Measures:** basic association learning; stable stimulus→action mapping.

---

## Bandit

**Task:** two-armed bandit (stochastic reward; drifting best arm).

**Measures:** value learning under drift; adaptation speed.

---

## Spot Reversal

**Task:** spot discrimination with periodic mapping reversal (context bit helps).

**Measures:** context-conditioned learning; re-acquisition after reversals.

---

## SpotXY

**Task:** track the dot in 2D.

**Measures:** scalable spatial encoding; generalization across grid sizes.

**Notes:** SpotXY supports **eval/holdout mode** (dynamics run, learning suppressed). Use this to verify that performance holds without additional writes.

---

## Pong

**Task:** paddle/ball control with binned sensors and 3 actions (up/down/stay).

**Measures:** closed-loop control; sensitivity to trial cadence and encoding.

See [pong-performance.md](pong-performance.md) for common failure modes and what to try.

---

## Web-only games

The in-browser WASM app (`crates/braine_web`) includes additional tasks (e.g., Sequence) that run without the daemon.
