# Daemon protocol (newline-delimited JSON)

Braine’s desktop stack uses a small TCP protocol to control and observe the long-running daemon.

- **Transport**: TCP
- **Address**: `127.0.0.1:9876`
- **Framing**: newline-delimited JSON (**NDJSON**) — each request is exactly one JSON object + `\n`, and each response is exactly one JSON object + `\n`.
- **Serialization**: `serde` tagged enums using `{"type": "..."}`.

This protocol is intentionally simple so it can be used from:
- the Slint desktop UI (`braine_desktop`)
- the CLI (`braine-cli`)
- external “integration hosts” (scripts/services) that want to stream stimuli and rewards

## Compatibility rules

- New fields should be added with `#[serde(default)]` so older clients don’t break.
- Clients should ignore unknown fields.
- Prefer adding new request types over changing semantics of existing ones.

## Message shape

### Request

All requests are tagged:

```json
{"type":"GetState"}
```

Requests may include additional fields:

```json
{"type":"SetTrialPeriodMs","ms":50}
```

### Response

Responses are also tagged:

```json
{"type":"State", ...}
```

Errors return:

```json
{"type":"Error","message":"..."}
```

## Core requests (most useful for dashboards)

These are the minimal calls a dashboard or automation loop usually needs.

### `GetState`
Returns the full daemon snapshot (running state, active game fields, HUD stats, and brain diagnostics).

- Request:
  - `{"type":"GetState"}`
- Response:
  - `{"type":"State", ...}`

### `CfgGet` / `CfgSet`
Get or update runtime knobs shared across games.

- `CfgGet` request: `{"type":"CfgGet"}`
- `CfgSet` request (all fields optional):

```json
{"type":"CfgSet","exploration_eps":0.2,"meaning_alpha":2.5,"reward_symbol_threshold":0.1,"concept_validate_threshold":0.1,"target_fps":60,"trial_period_ms":50,"max_units":4096}
```

Notes:
- `reward_symbol_threshold` controls when scalar reward is converted into discrete `reward_pos` / `reward_neg` symbols during `commit_observation()`.
- `concept_validate_threshold` controls when concept-validation is triggered (during sufficiently strong positive reward).

### `DiagGet`
Lightweight diagnostics: running state, frame counter, brain stats, and storage paths.

- Request: `{"type":"DiagGet"}`
- Response: `{"type":"Diagnostics", ...}`

### `ApiCatalog`
Introspects the daemon’s API surface (grouped by category) so clients can render help or UI affordances.

- Request: `{"type":"ApiCatalog"}`
- Response: `{"type":"ApiCatalog","categories":[...]}`

## Copy/paste examples (NDJSON)

All examples below are **single-line JSON** messages terminated with a newline.

### Quick manual check with netcat

In one terminal, run the daemon:

```bash
cargo run -p brained
```

In another terminal:

```bash
nc 127.0.0.1 9876
```

Then paste a request line (press Enter) and you’ll receive one response line.

### 1) Health check + basic diagnostics

Request:

```json
{"type":"DiagGet"}
```

Response (shape):

```json
{"type":"Diagnostics","running":false,"frame":0,"brain_stats":{...},"storage":{...}}
```

### 2) Poll the full UI snapshot (dashboard polling)

Request:

```json
{"type":"GetState"}
```

Response (shape):

```json
{"type":"State","running":true,"mode":"braine","frame":12345,"target_fps":60,"game":{...},"hud":{...},"brain_stats":{...},"storage":{...}}
```

Notes:
- `GetState` is the “everything snapshot” used by the desktop UI.
- For lightweight polling, `DiagGet` is smaller.

### 3) Start / stop (stop persists brain)

Start:

```json
{"type":"Start"}
```

Response:

```json
{"type":"Success","message":"Started"}
```

Stop (also triggers a save):

```json
{"type":"Stop"}
```

Response (success or error):

```json
{"type":"Success","message":"Stopped and saved"}
```

### 4) Switch games safely (daemon enforces “stop first”)

Stop (required):

```json
{"type":"Stop"}
```

Set game:

```json
{"type":"SetGame","game":"spotxy"}
```

Response:

```json
{"type":"Success","message":"Game set to spotxy"}
```

Then start:

```json
{"type":"Start"}
```

### 5) Read / update runtime config (safe clamped)

Read:

```json
{"type":"CfgGet"}
```

Response (shape):

```json
{"type":"Config","exploration_eps":0.2,"meaning_alpha":2.5,"reward_symbol_threshold":0.2,"concept_validate_threshold":0.2,"target_fps":60,"trial_period_ms":50,"max_units_limit":4096}
```

Update (all fields optional):

```json
{"type":"CfgSet","exploration_eps":0.1,"meaning_alpha":3.0,"reward_symbol_threshold":0.1,"concept_validate_threshold":0.1,"target_fps":60,"trial_period_ms":50,"max_units":4096}
```

Response:

```json
{"type":"Success","message":"Config updated"}
```

### 6) Change cadence (explicit endpoints)

If you prefer dedicated endpoints over `CfgSet`:

```json
{"type":"SetTrialPeriodMs","ms":75}
```

```json
{"type":"SetFramerate","fps":60}
```

Both respond with `{"type":"Success",...}` (or `Error`).

## Manual gates (freeze / paralyze)

The daemon exposes **manual gates** that let clients selectively suppress learning or activity.

- **Freeze**: dynamics still run, but learning updates are skipped for edges incident to gated units.
- **Paralyze**: unit activity is clamped to zero (and learning is skipped).

These are *ephemeral* (not persisted in the brain image) and intended for experimentation/debugging.

### `GatesGetModules`
List routing modules so clients can target module ids.

- Request:

```json
{"type":"GatesGetModules"}
```

- Response (shape):

```json
{"type":"GatesModules","modules":[{"id":0,"name":"latent::0","unit_count":128,"frozen_units":0,"paralyzed_units":0,"reward_ema":0.0,"last_routed_step":1234}, ...]}
```

### `GatesSet`
Set freeze/paralyze for either units or routing modules.

- Freeze module 0:

```json
{"type":"GatesSet","target":"module","ids":[0],"gate":"freeze","enabled":true}
```

- Paralyze units 10 and 11:

```json
{"type":"GatesSet","target":"unit","ids":[10,11],"gate":"paralyze","enabled":true}
```

### `GatesClear`
Clear all gates.

```json
{"type":"GatesClear"}
```

## Programmable reward interface (external trial)

For integration hosts, the daemon supports a simple “external trial” call that applies caller-provided stimuli and reward.

Important:
- The daemon must be stopped (`running=false`) to avoid interfering with an active game loop.

### `Trial`

Request (shape):

```json
{"type":"Trial","context_key":"ctx::0","stimuli":[{"name":"s::x","strength":1.0}],"allowed_actions":["left","right"],"reward":0.25,"learn":true,"steps":1,"meaning_alpha":2.5}
```

Fields:
- `context_key`: stimulus signature string used for meaning queries and pair symbols.
- `stimuli`: list of `{name, strength}` stimuli to apply before stepping.
- `allowed_actions` (optional): if provided, the daemon selects the best-scoring action among those names.
- `forced_action` (optional): if provided, uses this action instead of selecting.
- `reward`: scalar reward in roughly `[-1, 1]`.
- `learn`: if false, the daemon discards the observation (no learning/causal updates).
- `steps`: how many substrate steps to advance after applying stimuli (default 1).
- `meaning_alpha`: optional override for meaning weight.

Response:

```json
{"type":"TrialResult","action":"left","score":0.42,"reward":0.25,"learned":true}
```

### 7) Discover and set game parameters (knobs)

Fetch parameter schema for a game:

```json
{"type":"GetGameParams","game":"maze"}
```

Response (shape):

```json
{"type":"GameParams","game":"maze","params":[{"key":"reward_scale","min":0.0,"max":10.0,"default":1.0}, ...]}
```

Set a knob (applies to the *active* game only):

```json
{"type":"SetGameParam","game":"maze","key":"episodes_per_maze","value":64.0}
```

Response:

```json
{"type":"Success","message":"Set maze.episodes_per_maze = 64"}
```

### 8) Save / load snapshots

Save:

```json
{"type":"SaveSnapshot"}
```

Response (includes the stem in the message):

```json
{"type":"Success","message":"Snapshot saved (2026-01-27_12-34-56)"}
```

Load:

```json
{"type":"LoadSnapshot","stem":"2026-01-27_12-34-56"}
```

Response:

```json
{"type":"Success","message":"Snapshot loaded (2026-01-27_12-34-56)"}
```

### 9) Fetch graphs (substrate vs causal)

Substrate graph (default kind):

```json
{"type":"GetGraph","kind":"substrate","max_nodes":128,"max_edges":512,"include_isolated":false}
```

Causal graph:

```json
{"type":"GetGraph","kind":"causal","max_nodes":128,"max_edges":512,"include_isolated":false}
```

Response (shape):

```json
{"type":"Graph","kind":"causal","nodes":[...],"edges":[...]}
```

### 10) Read-only action score breakdown (debugging meaning-conditioning)

This is useful for dashboards and debugging because it never writes learning state.

Request:

```json
{"type":"InferActionScores","context_key":"pair::maze::up","stimuli":[{"name":"maze_wall_n","strength":1.0}],"steps":1,"meaning_alpha":3.0}
```

Response (shape):

```json
{"type":"InferActionScores","context_key":"pair::maze::up","action_scores":[{"name":"up","habit_norm":0.0,"meaning":0.12,"score":0.12}, ...]}
```

## Control requests

### Run control
- `Start` / `Stop`: start or stop the active task loop.
- `Shutdown`: stop and exit the daemon (also triggers persistence).

### Game selection and parameters
- `SetGame { game }`: switch the active game (daemon enforces “stop first”).
- `GetGameParams { game }`: returns a schema describing game knobs for UI.
- `SetGameParam { game, key, value }`: set a specific game knob.

### Timing / cadence
- `SetFramerate { fps }`: sets daemon tick rate.
- `SetTrialPeriodMs { ms }`: sets trial cadence (game “decision boundary”).

### Mode and manual actions
- `SetMode { mode }`: typically `"braine"` vs `"human"`.
- `HumanAction { action }`: inject a direct action (for human control / debugging).

### Storage
- `SaveBrain` / `LoadBrain` / `ResetBrain`
- `SaveSnapshot` / `LoadSnapshot { stem }`

### View / visualization
- `SetView { view }`: e.g. parent vs active expert.
- `GetGraph { kind, max_nodes, max_edges, include_isolated }`: graph snapshot for visualizers.

## Advanced features

### Inference-only scoring
`InferActionScores` runs a pure inference step on a cloned brain and returns per-action score breakdowns.
This is useful for dashboards, advisors, and debugging meaning-conditioning.

### Experts (child brains)
Requests:
- `SetExpertsEnabled`, `SetExpertNesting`, `SetExpertPolicy`, `CullExperts`

These implement the “experts are for novelty” sandbox mechanism.

### Advisor / LLM boundary
Requests:
- `AdvisorGet`, `AdvisorSet`, `AdvisorOnce`
- `AdvisorContext { include_action_scores }`
- `AdvisorApply { advice }`

The key invariant is that the advisor boundary is **bounded**: the daemon clamps advice and applies it safely.

### Replay dataset
- `ReplayGetDataset`
- `ReplaySetDataset { dataset }`

## Where to find the authoritative definitions

The canonical protocol enums live in:
- [crates/brained/src/main.rs](../../crates/brained/src/main.rs)

Client mirrors:
- [crates/core/src/bin/braine_cli.rs](../../crates/core/src/bin/braine_cli.rs)
- [crates/braine_desktop/src/main.rs](../../crates/braine_desktop/src/main.rs)
