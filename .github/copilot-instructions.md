# Copilot instructions for `braine`

## Coding Standards
- Produce modular, reusable code
- Follow existing project style and patterns
- Ensure code is efficient and optimized for performance
- Write unit tests for new features and bug fixes
- Write clear comments and documentation
- Adhere to Rust best practices and idioms
- Use meaningful variable and function names
- Avoid writing long files, breakdown into smaller functions and modules
- Ensure proper error handling and reporting
- Maintain backward compatibility when modifying existing functionality
- Always fix errors and warnings from the Rust compiler and Clippy linter

## Big picture (architecture + why)
- This repo is a **closed-loop learning substrate**, not an LLM: sparse recurrent dynamics + local plasticity + scalar reward (**neuromodulator**). No backprop/gradients.
- There is a **service boundary**:
  - `crates/brained/` = daemon (tokio TCP server) that owns the long-lived `Brain` state + task loop + persistence.
  - `crates/braine_desktop/` (package name `braine_desktop`) = Slint UI client that polls state and sends commands.
  - `crates/core/src/bin/braine_cli.rs` = CLI client for automation/debugging.
  - `crates/braine_web/` = (stub) browser-hosted WASM app with Leptos (Option A).
  - `crates/shared/braine_games/` = shared game logic for daemon and WASM.
- Cross-component communication is **newline-delimited JSON** over TCP `127.0.0.1:9876`.
  - Canonical protocol types live in `crates/brained/src/main.rs` (`Request`/`Response` with `#[serde(tag = "type")]`).

## Braine Architecture Direction (non-negotiable)

You are working inside Braine: a continuously running dynamical brain substrate with local learning.
This system is not an LLM and must not be shaped like one.

### Core architectural principles

**Reuse dynamics, do not duplicate learners**
- Scaling a problem must reuse the same recurrent dynamics and learned couplings.
- Do not spawn child brains solely because a task instance is larger, longer, or more detailed.
- Larger or more complex inputs should excite more instances of the same learned patterns, not create new models.

**Child brains exist only for novelty or distributional shift**

Spawn child brains only when:
- A new stimulus symbol or sensor modality appears.
- Reward dynamics change meaningfully.
- Performance collapses despite stable inputs.
- The parent brain reports saturation or inability to form stable attractors.

Child brains are temporary sandboxes for exploration, not permanent sub-solvers.

**Consolidation over composition**
- Children return structural changes (useful couplings, concepts, or abstractions), not raw answers.
- Parents consolidate only high-salience, high-utility changes.
- Avoid hierarchical answer composition; prefer hierarchical representation reuse.

**Learning modifies state; inference uses state**
- Learning updates internal dynamics (couplings, attractors, causal links).
- Inference applies the learned dynamics to new inputs.
- Never mix learning logic into inference paths.

### Task decomposition guidance

Decompose problems by invariants and structure, not by input size.

Prefer:
- Reusable local patterns
- Sparse symbolic anchors at the frame boundary
- Recurrent composition through wave dynamics

Avoid:
- Per-instance expert spawning
- Recursive subproblem trees for self-similar tasks
- Relearning identical contingencies in multiple places

### Problem-set contract (how users define tasks)

Braine consumes problem sets via a minimal contract, not datasets or labels.

A valid problem definition must specify:

**Stimulus symbols**
- Named inputs injected by the Frame
- No assumptions about internal representations

**Action symbols**
- Named outputs the brain can emit
- Chosen via action readout, not hard rules

**Reward / neuromodulation**
- Scalar feedback in a bounded range
- Drives learning rate, not direct supervision

**Temporal structure**
- Trial timing, stimulus duration, and transitions
- Decoupled from simulation framerate

The brain must learn:
- Which stimuli matter
- Which actions tend to help
- Which patterns persist across time

### Scaling expectations

- If a task is a scaled version of a known task, reuse learned dynamics.
- If a task introduces new structure, allow child brain exploration.
- If performance degrades, first adjust representation or encoding, not agent count.

### Anti-patterns (do not implement)

- Explicit backpropagation
- Global loss functions
- Tokenization or sequence prediction (as an LLM objective)
- Expert-per-subproblem architectures
- Fixed-depth recursive solvers

### Design intent reminder

Braine is designed to:
- Learn online
- Forget safely
- Operate indefinitely
- Remain interpretable via structure and dynamics
- Scale through reuse, not duplication

When in doubt, favor dynamic reuse and consolidation over spawning and orchestration.

**One-sentence invariant:** Experts are for novelty; dynamics are for scale.

Optional note (for contributors):
If a design requires many child brains to solve a task, assume the decomposition is wrong and revisit the representation.

## Key code locations
- Core substrate + learning rules: `crates/core/src/core/substrate.rs` (`Brain`, `apply_stimulus`, `step`, `commit_observation`, meaning/causal memory).
- Daemon runtime + protocol handling: `crates/brained/src/main.rs`.
- Daemon tasks/games: `crates/brained/src/game.rs` (e.g., Spot/Bandit/SpotReversal/SpotXY).
- Slint UI: `crates/braine_desktop/ui/main.slint` (UI models + tabs/canvases).
- Slint client glue/protocol mapping: `crates/braine_desktop/src/main.rs`.
- Shared game logic: `crates/shared/braine_games/src/` (Pong, etc.)

## Project-specific interaction patterns (important)
- "Observation" semantics: the daemon typically does
  - apply stimuli (`Stimulus { name, strength }`), `brain.step()`, choose action,
  - record symbols (e.g., `note_action*`, `note_pair*`, `note_compound_symbol([...])`),
  - set neuromodulator/reinforce, then `commit_observation()`.
- Context-conditioned meaning is usually expressed via symbols like:
  - `pair::<ctx>::<action>` (see docs in `doc/interaction.md` and usage in games/demos).
- **SpotXY eval/holdout mode**: runs dynamics/action selection but suppresses learning writes.
  - Implemented by skipping reinforcement and calling `Brain::discard_observation()` instead of `commit_observation()`.

## When changing the protocol
- If you add/change a daemon `Request`/`Response` or snapshot field in `crates/brained/src/main.rs`, update **all clients**:
  - `crates/braine_desktop/src/main.rs` (serde structs + UI mapping)
  - `crates/braine_desktop/ui/main.slint` (data model fields)
  - `crates/core/src/bin/braine_cli.rs` (if the CLI should support it)
- Keep messages backwards-tolerant where feasible by using `#[serde(default)]` on new fields.

## Persistence
- Daemon persists the brain image to OS data dirs (see `crates/brained/README.md` and root `README.md`).
- `Stop`/`Shutdown` trigger saves; there is also autosave logic in the daemon.
- WASM targets use `Brain::save_image_bytes()` / `Brain::load_image_bytes()` for IndexedDB persistence.

## Developer workflows (known-good commands)
- Tests: `cargo test` (unit tests live primarily under `crates/core/src/core/substrate.rs`, `crates/core/src/core/causality.rs`, `crates/core/src/core/supervisor.rs`).
- Local run:
  - daemon: `cargo run -p brained`
  - UI: `cargo run -p braine_desktop`
  - CLI: `cargo run --bin braine-cli -- status`
- Packaging/lint script: `./scripts/dev.sh` runs `fmt`, `clippy -D warnings`, release builds, and creates `dist/braine-portable.zip`.
  - Requires `7z` and the `x86_64-pc-windows-gnu` Rust target.

## Conventions / gotchas
- All crates live under `crates/` directory.
- Game switching is enforced as "stop first" at the daemon; UI should disable switching while running.
