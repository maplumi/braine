# Performance & Scalability

This document covers the performance optimization features in braine, from edge devices to server deployments.

## Connection Storage: CSR Format

Connections are stored in **Compressed Sparse Row (CSR)** format for cache efficiency:

```
offsets:  [0, 3, 5, 8, ...]     // Index into targets/weights per unit
targets:  [2, 5, 7, 1, 9, ...]  // Target unit IDs
weights:  [0.3, -0.1, 0.5, ...] // Connection weights
```

Benefits:
- **Cache-friendly**: Sequential memory access during iteration
- **O(1) neighbor lookup**: Direct index via offsets
- **Pruning**: Tombstone with `INVALID_UNIT`, periodic compaction
- **Serialization**: Flat arrays serialize efficiently

## Execution Tiers

The substrate supports four execution tiers via `ExecutionTier`:

### Scalar (default)
- **Target**: MCU, WASM, any platform
- **Implementation**: Single-threaded, no dependencies
- **Performance**: ~100μs for 512 units
- **When to use**: Embedded, WebAssembly, debugging

### SIMD (`--features simd`)
- **Target**: ARM NEON, x86 SSE/AVX
- **Implementation**: `wide` crate, 4-wide f32x4 vectorization
- **Performance**: ~30% faster than scalar for dense updates
- **When to use**: Single-core but vectorization available

```rust
brain.set_execution_tier(ExecutionTier::Simd);
```

The SIMD path vectorizes the amplitude/phase update loop. Sparse neighbor accumulation remains scalar (irregular memory access patterns).

### Parallel (`--features parallel`)
- **Target**: Desktop/server with multiple cores
- **Implementation**: rayon parallel iterators
- **Performance**: Scales with core count for >1024 units
- **When to use**: Multi-core systems, high connectivity

```rust
brain.set_execution_tier(ExecutionTier::Parallel);
```

Pre-generates noise on main thread (RNG not thread-safe), then parallelizes unit updates.

### GPU (`--features gpu`)
- **Target**: Very large substrates (10k+ units)
- **Implementation**: wgpu compute shaders (WGSL)
- **Performance**: Amortizes at 10k+ units
- **When to use**: Research with massive substrates

```rust
brain.set_execution_tier(ExecutionTier::Gpu);
```

Architecture:
1. **CPU**: Sparse neighbor accumulation (graph traversal)
2. **GPU**: Dense amplitude/phase update (compute shader)
3. **CPU**: Hebbian learning (sparse graph updates)

The GPU context is lazily initialized. Falls back to scalar if no GPU available.

## Tier Selection Guide

| Units | Connectivity | Recommended Tier |
|-------|-------------|------------------|
| <256 | Any | Scalar |
| 256-1024 | Low (<16) | SIMD |
| 256-1024 | High (>16) | Parallel |
| 1024-10k | Any | Parallel |
| >10k | Any | GPU (if available) |

## Benchmarking

### Running Benchmarks

```bash
# Baseline (scalar only)
cargo bench

# With specific features
cargo bench --features simd
cargo bench --features parallel
cargo bench --features "simd,parallel,gpu"

# Specific benchmark group
cargo bench -- "step_tier"
```

### Benchmark Groups

| Group | What it measures |
|-------|-----------------|
| `step_sizes` | step() at 64/128/256/512 units |
| `step_tier` | Scalar vs SIMD vs Parallel vs GPU at 512 units |
| `learning` | Hebbian update performance |
| `serialization` | Save/load round-trip |
| `csr_ops` | CSR neighbor iteration |

### Viewing Results

Benchmark results are saved to `target/criterion/`. Open `target/criterion/report/index.html` for HTML reports (requires gnuplot for graphs, falls back to plotters).

## Memory Usage

The CSR format has predictable memory usage:

```
Per unit:    ~24 bytes (amp, phase, bias, decay, other fields)
Per connection: ~8 bytes (target: 8, weight: 4, but aligned)
Overhead:    ~8 bytes per unit (offsets array)
```

For a 1024-unit brain with 16 connections per unit:
- Units: ~24 KB
- Connections: ~128 KB
- Total: ~152 KB + auxiliary buffers

## Feature Flag Dependencies

| Feature | Crate | Size Impact |
|---------|-------|-------------|
| `parallel` | rayon 1.10 | ~200 KB |
| `simd` | wide 0.7 | ~50 KB |
| `gpu` | wgpu 24.0 + deps | ~2 MB |

For minimal embedded builds, use no features (scalar only).

## Profiling Tips

1. **Release mode**: Always benchmark with `--release`
2. **CPU affinity**: Pin to specific cores for consistent results
3. **Warm-up**: Criterion handles warm-up automatically
4. **Noise isolation**: Close other applications during benchmarks

### Flamegraph

```bash
cargo install flamegraph
cargo flamegraph --features parallel -- pong-demo
```

### perf

```bash
cargo build --release --features parallel
perf record ./target/release/braine pong-demo
perf report
```
