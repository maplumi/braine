# Brain image format (WIP)

This project uses a **custom, versioned “brain image”** format for persistence.

The goal is not generic serialization — it’s a format that matches the project’s core idea:
**memory is structure** (sparse couplings + causal edges + symbol boundaries), with explicit
constraints for embedded/edge usage.

## Design goals
- **Std-only core**: the `braine` crate stays `std`-only.
- **Versioned, forward-compatible**: new versions can add chunks; older readers can skip unknown chunks.
- **Capacity-aware**: storage adapters can enforce a byte budget and fail predictably when the brain outgrows storage.
- **Separation of concerns**:
  - **Structural state** (must persist): units + connections, groups, symbol table, causal memory.
  - **Runtime/scratch** (can be reset): pending input, telemetry buffers, transient vectors.

## Storage adapter concept
A storage adapter is any sink/source implementing `Read`/`Write` (e.g., file, flash, network)
optionally wrapped by a capacity limiter.

Key idea: we can define a **minimum required capacity** for a given brain instance:
- `Brain::image_size_bytes()` → exact serialized size (in bytes)
- `CapacityWriter` → errors if writes exceed a fixed byte budget

This lets us iterate as we improve the brain:
- start with a small capacity target
- measure growth
- add pruning/compression/version bumps deliberately

## Format overview (v1)
Binary, little-endian, chunked.

### Name + file extension
- Format name: **Braine Brain Image (BBI)**
- File extension: **`.bbi`**

### Header
- magic: `BRAINE01` (8 bytes)
- version: `u32` (currently `1`)

There is no flags field in v1.

### Chunks
Each chunk:
- tag: 4 bytes (ASCII)
- len: `u32` payload length
- payload bytes

Planned tags (v1):
- `CFG0` — `BrainConfig`
- `PRNG` — RNG state (for deterministic continuation)
- `STAT` — small runtime counters (e.g., `age_steps`, `neuromod`)
- `UNIT` — units + sparse connections
- `MASK` — `reserved[]` + `learning_enabled[]`
- `GRPS` — sensor/action group definitions
- `SYMB` — `symbols_rev` string table (rebuild `symbols` map from this)
- `CAUS` — causal memory (base counts + directed edge counts)

Unknown tags must be skipped.

## Notes
- This format is intended for **research snapshots**, not as a security boundary.
- The daemon persists the active image as `brain.bbi` alongside `runtime.json`.
- The UI/daemon can also create **timestamped snapshots** under `snapshots/` in the same data
  directory (copies of both the brain image and runtime stats).
