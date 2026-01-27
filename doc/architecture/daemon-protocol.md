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
{"type":"CfgSet","exploration_eps":0.2,"meaning_alpha":2.5,"target_fps":60,"trial_period_ms":50,"max_units":4096}
```

### `DiagGet`
Lightweight diagnostics: running state, frame counter, brain stats, and storage paths.

- Request: `{"type":"DiagGet"}`
- Response: `{"type":"Diagnostics", ...}`

### `ApiCatalog`
Introspects the daemon’s API surface (grouped by category) so clients can render help or UI affordances.

- Request: `{"type":"ApiCatalog"}`
- Response: `{"type":"ApiCatalog","categories":[...]}`

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
