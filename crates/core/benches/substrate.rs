//! Criterion benchmarks for the braine substrate.
//!
//! Run with:
//!   cargo bench
//!   cargo bench --features parallel
//!
//! Results are saved to target/criterion/

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};

use braine::substrate::{Brain, BrainConfig, ExecutionTier, Stimulus};

fn make_brain(unit_count: usize, connectivity: usize, seed: u64) -> Brain {
    Brain::new(BrainConfig {
        unit_count,
        connectivity_per_unit: connectivity,
        dt: 0.05,
        base_freq: 1.0,
        noise_amp: 0.02,
        noise_phase: 0.01,
        global_inhibition: 0.06,
        hebb_rate: 0.08,
        forget_rate: 0.0015,
        prune_below: 0.0008,
        coactive_threshold: 0.55,
        phase_lock_threshold: 0.6,
        imprint_rate: 0.6,
        seed: Some(seed),
        causal_decay: 0.01,
    })
}

/// Benchmark step() with varying substrate sizes.
fn bench_step_sizes(c: &mut Criterion) {
    let mut group = c.benchmark_group("step_size");

    for size in [64, 128, 256, 512, 1024].iter() {
        let connectivity = (*size as f64).sqrt() as usize;
        group.throughput(Throughput::Elements(*size as u64));

        group.bench_with_input(BenchmarkId::new("scalar", size), size, |b, &size| {
            let mut brain = make_brain(size, connectivity, 42);
            brain.set_execution_tier(ExecutionTier::Scalar);
            brain.define_sensor("stim", 6);
            brain.define_action("act", 6);

            b.iter(|| {
                brain.apply_stimulus(Stimulus::new("stim", 1.0));
                brain.set_neuromodulator(0.5);
                brain.step();
                black_box(brain.diagnostics().avg_amp)
            });
        });
    }

    group.finish();
}

/// Benchmark step() comparing execution tiers at a fixed size.
fn bench_step_tiers(c: &mut Criterion) {
    let mut group = c.benchmark_group("step_tier");

    let size = 512;
    let connectivity = 16;
    group.throughput(Throughput::Elements(size as u64));

    // Scalar
    group.bench_function("scalar_512", |b| {
        let mut brain = make_brain(size, connectivity, 42);
        brain.set_execution_tier(ExecutionTier::Scalar);
        brain.define_sensor("stim", 6);
        brain.define_action("act", 6);

        b.iter(|| {
            brain.apply_stimulus(Stimulus::new("stim", 1.0));
            brain.set_neuromodulator(0.5);
            brain.step();
            black_box(brain.diagnostics().avg_amp)
        });
    });

    // SIMD (falls back to scalar if feature not enabled)
    group.bench_function("simd_512", |b| {
        let mut brain = make_brain(size, connectivity, 42);
        brain.set_execution_tier(ExecutionTier::Simd);
        brain.define_sensor("stim", 6);
        brain.define_action("act", 6);

        b.iter(|| {
            brain.apply_stimulus(Stimulus::new("stim", 1.0));
            brain.set_neuromodulator(0.5);
            brain.step();
            black_box(brain.diagnostics().avg_amp)
        });
    });

    // Parallel (falls back to scalar if feature not enabled)
    group.bench_function("parallel_512", |b| {
        let mut brain = make_brain(size, connectivity, 42);
        brain.set_execution_tier(ExecutionTier::Parallel);
        brain.define_sensor("stim", 6);
        brain.define_action("act", 6);

        b.iter(|| {
            brain.apply_stimulus(Stimulus::new("stim", 1.0));
            brain.set_neuromodulator(0.5);
            brain.step();
            black_box(brain.diagnostics().avg_amp)
        });
    });

    // GPU (falls back to scalar if feature not enabled or no GPU available)
    group.bench_function("gpu_512", |b| {
        let mut brain = make_brain(size, connectivity, 42);
        brain.set_execution_tier(ExecutionTier::Gpu);
        brain.define_sensor("stim", 6);
        brain.define_action("act", 6);

        b.iter(|| {
            brain.apply_stimulus(Stimulus::new("stim", 1.0));
            brain.set_neuromodulator(0.5);
            brain.step();
            black_box(brain.diagnostics().avg_amp)
        });
    });

    group.finish();
}

/// Benchmark learning (Hebbian updates) separately.
fn bench_learning(c: &mut Criterion) {
    let mut group = c.benchmark_group("learning");

    for size in [128, 256, 512].iter() {
        let connectivity = (*size as f64).sqrt() as usize;
        group.throughput(Throughput::Elements((*size * connectivity) as u64));

        group.bench_with_input(BenchmarkId::new("hebbian", size), size, |b, &size| {
            let mut brain = make_brain(size, connectivity, 42);
            brain.define_sensor("stim", 6);
            brain.define_action("act", 6);

            // Warm up: get some activity going.
            for _ in 0..50 {
                brain.apply_stimulus(Stimulus::new("stim", 1.0));
                brain.set_neuromodulator(0.7);
                brain.step();
            }

            b.iter(|| {
                brain.apply_stimulus(Stimulus::new("stim", 1.0));
                brain.set_neuromodulator(0.7);
                brain.step();
                black_box(brain.diagnostics().connection_count)
            });
        });
    }

    group.finish();
}

/// Benchmark serialization round-trip.
fn bench_serialization(c: &mut Criterion) {
    let mut group = c.benchmark_group("serialization");

    for size in [128, 256, 512].iter() {
        let connectivity = (*size as f64).sqrt() as usize;

        group.bench_with_input(BenchmarkId::new("save", size), size, |b, &size| {
            let brain = make_brain(size, connectivity, 42);
            let mut buf = Vec::with_capacity(64 * 1024);

            b.iter(|| {
                buf.clear();
                brain.save_image_to(&mut buf).unwrap();
                black_box(buf.len())
            });
        });

        group.bench_with_input(BenchmarkId::new("load", size), size, |b, &size| {
            let brain = make_brain(size, connectivity, 42);
            let mut buf = Vec::new();
            brain.save_image_to(&mut buf).unwrap();

            b.iter(|| {
                let mut cursor = std::io::Cursor::new(&buf);
                let loaded = Brain::load_image_from(&mut cursor).unwrap();
                black_box(loaded.diagnostics().unit_count)
            });
        });
    }

    group.finish();
}

/// Benchmark CSR connection operations.
fn bench_csr_ops(c: &mut Criterion) {
    let mut group = c.benchmark_group("csr_ops");

    let size = 256;
    let connectivity = 16;

    group.bench_function("neighbor_iteration", |b| {
        let brain = make_brain(size, connectivity, 42);

        b.iter(|| {
            let mut total = 0.0f32;
            for i in 0..size {
                for (_, weight) in brain.neighbors(i) {
                    total += weight;
                }
            }
            black_box(total)
        });
    });

    group.bench_function("diagnostics", |b| {
        let brain = make_brain(size, connectivity, 42);

        b.iter(|| black_box(brain.diagnostics()));
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_step_sizes,
    bench_step_tiers,
    bench_learning,
    bench_serialization,
    bench_csr_ops,
);

criterion_main!(benches);
