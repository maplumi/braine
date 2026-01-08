# Copilot instructions for `braine`

## Coding Standards
- Produce modular, reusable code
- Follow existing project style and patterns
- Ensure code is efficient and optimized for performance
- Write unit tests for new features and bug fixes
- Write clear comments and documentation
- Adhere to Rust best practices and idioms
- Use meaningful variable and function names
- Avoid wrriting long filed, breakdown into smaller functions and modules
- Ensure proper error handling and reporting
- Maintain backward compatibility when modifying existing functionality
- Always fix errors and warnings from the Rust compiler and Clippy linter

## Big picture (architecture + why)
- This repo is a **closed-loop learning substrate**, not an LLM: sparse recurrent dynamics + local plasticity + scalar reward (**neuromodulator**). No backprop/gradients.
- There is a **service boundary**:
  - `brained/` = daemon (tokio TCP server) that owns the long-lived `Brain` state + task loop + persistence.
  - `braine_vis/` (package name `braine_viz`) = Slint UI client that polls state and sends commands.
  - `src/bin/braine_cli.rs` = CLI client for automation/debugging.
- Cross-component communication is **newline-delimited JSON** over TCP `127.0.0.1:9876`.
  - Canonical protocol types live in `brained/src/main.rs` (`Request`/`Response` with `#[serde(tag = "type")]`).

## Key code locations
- Core substrate + learning rules: `src/core/substrate.rs` (`Brain`, `apply_stimulus`, `step`, `commit_observation`, meaning/causal memory).
- Daemon runtime + protocol handling: `brained/src/main.rs`.
- Daemon tasks/games: `brained/src/game.rs` (e.g., Spot/Bandit/SpotReversal/SpotXY).
- Slint UI: `braine_vis/ui/main.slint` (UI models + tabs/canvases).
- Slint client glue/protocol mapping: `braine_vis/src/main.rs`.

## Project-specific interaction patterns (important)
- “Observation” semantics: the daemon typically does
  - apply stimuli (`Stimulus { name, strength }`), `brain.step()`, choose action,
  - record symbols (e.g., `note_action*`, `note_pair*`, `note_compound_symbol([...])`),
  - set neuromodulator/reinforce, then `commit_observation()`.
- Context-conditioned meaning is usually expressed via symbols like:
  - `pair::<ctx>::<action>` (see docs in `doc/interaction.md` and usage in games/demos).
- **SpotXY eval/holdout mode**: runs dynamics/action selection but suppresses learning writes.
  - Implemented by skipping reinforcement and calling `Brain::discard_observation()` instead of `commit_observation()`.

## When changing the protocol
- If you add/change a daemon `Request`/`Response` or snapshot field in `brained/src/main.rs`, update **all clients**:
  - `braine_vis/src/main.rs` (serde structs + UI mapping)
  - `braine_vis/ui/main.slint` (data model fields)
  - `src/bin/braine_cli.rs` (if the CLI should support it)
- Keep messages backwards-tolerant where feasible by using `#[serde(default)]` on new fields.

## Persistence
- Daemon persists the brain image to OS data dirs (see `brained/README.md` and root `README.md`).
- `Stop`/`Shutdown` trigger saves; there is also autosave logic in the daemon.

## Developer workflows (known-good commands)
- Tests: `cargo test` (unit tests live primarily under `src/core/substrate.rs`, `src/core/causality.rs`, `src/core/supervisor.rs`).
- Local run:
  - daemon: `cargo run -p brained`
  - UI: `cargo run -p braine_viz`
  - CLI: `cargo run --bin braine-cli -- status`
- Packaging/lint script: `./scripts/dev.sh` runs `fmt`, `clippy -D warnings`, release builds, and creates `dist/braine-portable.zip`.
  - Requires `7z` and the `x86_64-pc-windows-gnu` Rust target.

## Conventions / gotchas
- Workspace member directory is `braine_vis/`, but the Cargo package/binary is `braine_viz`.
- Game switching is enforced as “stop first” at the daemon; UI should disable switching while running.


