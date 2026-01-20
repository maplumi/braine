//! GPU compute backend using wgpu for large-scale substrate updates.
//!
//! This module provides compute shader acceleration for Brain dynamics when
//! dealing with very large substrates (10k+ units). The GPU backend parallelizes
//! the amplitude and phase updates across thousands of GPU threads.
//!
//! Enable with the `gpu` feature flag.

use bytemuck::{Pod, Zeroable};
use std::borrow::Cow;
use wgpu::util::DeviceExt;

thread_local! {
    static GPU_CTX: std::cell::OnceCell<Result<GpuContext, String>> =
        const { std::cell::OnceCell::new() };
}

thread_local! {
    static GPU_DISABLED: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// Disable GPU usage for the remainder of this session.
///
/// This is primarily used on wasm/WebGPU when validation errors occur at runtime;
/// we prefer a clean CPU fallback over retrying a broken GPU pipeline every tick.
pub fn disable_gpu_for_session() {
    GPU_DISABLED.with(|d| d.set(true));
}

#[cfg(all(feature = "gpu", target_arch = "wasm32"))]
thread_local! {
    static GPU_PENDING: std::cell::RefCell<Option<PendingGpuReadback>> =
        const { std::cell::RefCell::new(None) };
}

#[cfg(all(feature = "gpu", target_arch = "wasm32"))]
struct PendingGpuReadback {
    n: usize,
    staging: wgpu::Buffer,
    ready: std::rc::Rc<std::cell::RefCell<Option<Result<(), String>>>>,
}

/// Initialize the shared GPU context.
///
/// - On native targets this may block (device/adapter creation).
/// - On `wasm32` this is async and **must not block**.
///
/// Returns `Ok(())` when a usable GPU context is ready; otherwise `Err(msg)`.
#[cfg(feature = "gpu")]
pub async fn init_gpu_context(max_units: usize) -> Result<(), String> {
    #[cfg(target_arch = "wasm32")]
    {
        // If already initialized (success or failure), return current state.
        if let Some(existing) = GPU_CTX.with(|cell| {
            cell.get().map(|res| match res {
                Ok(_) => Ok(()),
                Err(e) => Err(e.clone()),
            })
        }) {
            return existing;
        }

        let ctx_res = GpuContext::new_async(max_units).await;
        let err = ctx_res.as_ref().err().cloned();
        let ok = err.is_none();
        GPU_CTX.with(|cell| {
            // Ignore double-set races; wasm is single-threaded, but be defensive.
            let _ = cell.set(ctx_res);
        });
        if ok {
            GPU_DISABLED.with(|d| d.set(false));
            Ok(())
        } else {
            GPU_DISABLED.with(|d| d.set(true));
            Err(err.unwrap_or_else(|| "WebGPU init failed".to_string()))
        }
    }

    #[cfg(not(target_arch = "wasm32"))]
    {
        let ok = with_gpu_context(max_units, |ctx| ctx.is_some());
        if ok {
            Ok(())
        } else {
            Err("GPU adapter/device unavailable".to_string())
        }
    }
}

/// Access the shared GPU context (lazily initialized).
///
/// The context is cached per-thread for simplicity.
pub fn with_gpu_context<T>(max_units: usize, f: impl FnOnce(Option<&GpuContext>) -> T) -> T {
    #[cfg(target_arch = "wasm32")]
    {
        let _ = max_units;
        if GPU_DISABLED.with(|d| d.get()) {
            return f(None);
        }

        GPU_CTX.with(|cell| match cell.get() {
            Some(Ok(ctx)) => f(Some(ctx)),
            _ => f(None),
        })
    }

    #[cfg(not(target_arch = "wasm32"))]
    {
        GPU_CTX.with(|cell| {
            let ctx = cell.get_or_init(|| {
                GpuContext::new(max_units)
                    .ok_or_else(|| "GPU adapter/device unavailable".to_string())
            });
            f(ctx.as_ref().ok())
        })
    }
}

/// Returns true if a GPU context can be created.
///
/// Note: this may initialize the GPU context and can be expensive.
pub fn gpu_available(max_units: usize) -> bool {
    #[cfg(target_arch = "wasm32")]
    {
        let _ = max_units;
        // On wasm, GPU initialization must be done asynchronously.
        // This function only reports whether the shared context is ready.
        !GPU_DISABLED.with(|d| d.get()) && GPU_CTX.with(|cell| matches!(cell.get(), Some(Ok(_))))
    }

    #[cfg(not(target_arch = "wasm32"))]
    {
        with_gpu_context(max_units, |ctx| ctx.is_some())
    }
}

/// GPU-friendly unit representation (aligned to 16 bytes).
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct GpuUnit {
    pub amp: f32,
    pub phase: f32,
    pub bias: f32,
    pub decay: f32,
}

/// GPU-friendly configuration parameters.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct GpuParams {
    pub dt: f32,
    pub base_freq: f32,
    pub inhibition: f32,
    pub unit_count: u32,
}

/// Influence data computed on CPU (sparse graph traversal is CPU-bound).
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct GpuInfluence {
    pub amp: f32,
    pub phase: f32,
    pub noise_amp: f32,
    pub noise_phase: f32,
}

/// Pending input per unit.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct GpuInput {
    pub value: f32,
    pub _padding: [f32; 3], // Align to 16 bytes
}

/// Error type for GPU operations.
#[derive(Debug)]
pub enum GpuError {
    /// Failed to receive result from GPU.
    ReceiveError,
    /// GPU buffer mapping failed.
    MapError(wgpu::BufferAsyncError),
    /// Size exceeds maximum units.
    SizeExceeded { requested: usize, max: usize },
}

impl std::fmt::Display for GpuError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            GpuError::ReceiveError => write!(f, "Failed to receive GPU result"),
            GpuError::MapError(e) => write!(f, "GPU buffer mapping failed: {:?}", e),
            GpuError::SizeExceeded { requested, max } => {
                write!(f, "Requested {} units exceeds max {}", requested, max)
            }
        }
    }
}

impl std::error::Error for GpuError {}

/// GPU compute context for accelerating Brain dynamics.
pub struct GpuContext {
    device: wgpu::Device,
    queue: wgpu::Queue,
    pipeline: wgpu::ComputePipeline,
    bind_group_layout: wgpu::BindGroupLayout,
    max_units: usize,
}

impl GpuContext {
    /// Create a new GPU context. Blocks until GPU is ready.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn new(max_units: usize) -> Option<Self> {
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            ..Default::default()
        });

        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: None,
            force_fallback_adapter: false,
        }))?;

        let (device, queue) = pollster::block_on(adapter.request_device(
            &wgpu::DeviceDescriptor {
                label: Some("Braine GPU"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::default(),
                memory_hints: wgpu::MemoryHints::Performance,
            },
            None,
        ))
        .ok()?;

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Dynamics Shader"),
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(DYNAMICS_SHADER)),
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Dynamics Pipeline"),
            // Let wgpu derive the layout from WGSL.
            // This avoids WebGPU backend discrepancies around read-only vs read-write storage.
            layout: None,
            module: &shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        let bind_group_layout = pipeline.get_bind_group_layout(0);

        Some(Self {
            device,
            queue,
            pipeline,
            bind_group_layout,
            max_units,
        })
    }

    /// Create a new GPU context asynchronously (WebGPU).
    #[cfg(target_arch = "wasm32")]
    pub async fn new_async(max_units: usize) -> Result<Self, String> {
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends: wgpu::Backends::BROWSER_WEBGPU,
            ..Default::default()
        });

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: None,
                force_fallback_adapter: false,
            })
            .await
            .ok_or_else(|| "WebGPU: request_adapter returned None".to_string())?;

        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: Some("Braine WebGPU"),
                    required_features: wgpu::Features::empty(),
                    required_limits: wgpu::Limits::default(),
                    memory_hints: wgpu::MemoryHints::Performance,
                },
                None,
            )
            .await
            .map_err(|e| format!("WebGPU: request_device failed: {e:?}"))?;

        // Capture validation errors during shader/pipeline/layout setup.
        device.push_error_scope(wgpu::ErrorFilter::Validation);

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Dynamics Shader"),
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(DYNAMICS_SHADER)),
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Dynamics Pipeline"),
            // Let wgpu derive the layout from WGSL.
            // This avoids WebGPU backend discrepancies around read-only vs read-write storage.
            layout: None,
            module: &shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        let bind_group_layout = pipeline.get_bind_group_layout(0);

        // Also validate bind group creation against the layout.
        // This catches common Storage/Uniform mismatches early.
        let dummy_units_in = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Units In Buffer (dummy)"),
            size: std::mem::size_of::<GpuUnit>() as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let dummy_units_out = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Units Out Buffer (dummy)"),
            size: std::mem::size_of::<GpuUnit>() as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let dummy_influences = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Influences Buffer (dummy)"),
            size: std::mem::size_of::<GpuInfluence>() as u64,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        let dummy_inputs = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Inputs Buffer (dummy)"),
            size: std::mem::size_of::<GpuInput>() as u64,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        let dummy_params = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Params Buffer (dummy)"),
            size: std::mem::size_of::<GpuParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM,
            mapped_at_creation: false,
        });
        let _ = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Dynamics Bind Group (dummy)"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: dummy_units_in.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: dummy_influences.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: dummy_inputs.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: dummy_params.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: dummy_units_out.as_entire_binding(),
                },
            ],
        });

        if let Some(e) = device.pop_error_scope().await {
            return Err(format!("WebGPU validation error: {e}"));
        }

        Ok(Self {
            device,
            queue,
            pipeline,
            bind_group_layout,
            max_units,
        })
    }

    /// Execute dynamics update on GPU.
    ///
    /// # Arguments
    /// * `units` - Mutable slice of unit state (amp, phase, bias, decay)
    /// * `influences` - Pre-computed influences from sparse graph traversal
    /// * `inputs` - Pending input per unit
    /// * `params` - Simulation parameters
    ///
    /// # Errors
    /// Returns `GpuError` if the operation fails or size exceeds limits.
    pub fn step_dynamics(
        &self,
        units: &mut [GpuUnit],
        influences: &[GpuInfluence],
        inputs: &[f32],
        params: GpuParams,
    ) -> Result<(), GpuError> {
        let n = units.len();
        if n == 0 {
            return Ok(());
        }
        if n > self.max_units {
            return Err(GpuError::SizeExceeded {
                requested: n,
                max: self.max_units,
            });
        }

        // Create buffers
        let units_in_buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Units In Buffer"),
                contents: bytemuck::cast_slice(units),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            });

        let units_out_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Units Out Buffer"),
            size: std::mem::size_of_val(units) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let influences_buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Influences Buffer"),
                contents: bytemuck::cast_slice(influences),
                usage: wgpu::BufferUsages::STORAGE,
            });

        // Pack inputs with padding
        let inputs_padded: Vec<GpuInput> = inputs
            .iter()
            .map(|&v| GpuInput {
                value: v,
                _padding: [0.0; 3],
            })
            .collect();
        let inputs_buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Inputs Buffer"),
                contents: bytemuck::cast_slice(&inputs_padded),
                usage: wgpu::BufferUsages::STORAGE,
            });

        let params_buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Params Buffer"),
                contents: bytemuck::bytes_of(&params),
                usage: wgpu::BufferUsages::UNIFORM,
            });

        // Staging buffer for readback
        let staging_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Staging Buffer"),
            size: std::mem::size_of_val(units) as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Dynamics Bind Group"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: units_in_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: influences_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: inputs_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: params_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: units_out_buffer.as_entire_binding(),
                },
            ],
        });

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Dynamics Encoder"),
            });

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Dynamics Pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            // Dispatch workgroups (64 threads per group)
            pass.dispatch_workgroups(n.div_ceil(64) as u32, 1, 1);
        }

        // Copy results back
        encoder.copy_buffer_to_buffer(
            &units_out_buffer,
            0,
            &staging_buffer,
            0,
            std::mem::size_of_val(units) as u64,
        );

        self.queue.submit(std::iter::once(encoder.finish()));

        // Read back results
        let buffer_slice = staging_buffer.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = tx.send(result);
        });
        self.device.poll(wgpu::Maintain::Wait);

        let map_result = rx.recv().map_err(|_| GpuError::ReceiveError)?;
        map_result.map_err(GpuError::MapError)?;

        let data = buffer_slice.get_mapped_range();
        let result: &[GpuUnit] = bytemuck::cast_slice(&data);
        units.copy_from_slice(result);
        Ok(())
    }

    /// Begin a GPU dynamics update on wasm **without blocking**.
    ///
    /// This submits work and starts an async mapping request. Call
    /// [`wasm_try_finish_step_dynamics`] later to apply the results.
    #[cfg(target_arch = "wasm32")]
    pub fn wasm_begin_step_dynamics(
        &self,
        units: &[GpuUnit],
        influences: &[GpuInfluence],
        inputs: &[f32],
        params: GpuParams,
    ) -> Result<(), GpuError> {
        let n = units.len();
        if n == 0 {
            return Ok(());
        }
        if n > self.max_units {
            return Err(GpuError::SizeExceeded {
                requested: n,
                max: self.max_units,
            });
        }

        // Create buffers
        let units_in_buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Units In Buffer"),
                contents: bytemuck::cast_slice(units),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            });

        let units_out_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Units Out Buffer"),
            size: std::mem::size_of_val(units) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let influences_buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Influences Buffer"),
                contents: bytemuck::cast_slice(influences),
                usage: wgpu::BufferUsages::STORAGE,
            });

        // Pack inputs with padding
        let inputs_padded: Vec<GpuInput> = inputs
            .iter()
            .map(|&v| GpuInput {
                value: v,
                _padding: [0.0; 3],
            })
            .collect();
        let inputs_buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Inputs Buffer"),
                contents: bytemuck::cast_slice(&inputs_padded),
                usage: wgpu::BufferUsages::STORAGE,
            });

        let params_buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Params Buffer"),
                contents: bytemuck::bytes_of(&params),
                usage: wgpu::BufferUsages::UNIFORM,
            });

        // Staging buffer for readback
        let staging_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Staging Buffer"),
            size: std::mem::size_of_val(units) as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Dynamics Bind Group"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: units_in_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: influences_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: inputs_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: params_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: units_out_buffer.as_entire_binding(),
                },
            ],
        });

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Dynamics Encoder"),
            });

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Dynamics Pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(n.div_ceil(64) as u32, 1, 1);
        }

        encoder.copy_buffer_to_buffer(
            &units_out_buffer,
            0,
            &staging_buffer,
            0,
            std::mem::size_of_val(units) as u64,
        );

        self.queue.submit(std::iter::once(encoder.finish()));

        // Install the pending readback into a thread-local slot.
        let ready: std::rc::Rc<std::cell::RefCell<Option<Result<(), String>>>> =
            std::rc::Rc::new(std::cell::RefCell::new(None));
        let ready_cb = ready.clone();

        let buffer_slice = staging_buffer.slice(..);
        buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
            let mut slot = ready_cb.borrow_mut();
            *slot = Some(result.map_err(|e| format!("GPU map_async failed: {e:?}")));
        });
        // Non-blocking poll.
        self.device.poll(wgpu::Maintain::Poll);

        GPU_PENDING.with(|p| {
            *p.borrow_mut() = Some(PendingGpuReadback {
                n,
                staging: staging_buffer,
                ready,
            });
        });

        Ok(())
    }
}

/// Returns true if a wasm GPU dynamics step is in flight.
#[cfg(all(feature = "gpu", target_arch = "wasm32"))]
pub fn wasm_gpu_step_in_flight() -> bool {
    GPU_PENDING.with(|p| p.borrow().is_some())
}

/// Try to finish the in-flight wasm GPU dynamics step.
///
/// Returns:
/// - `None` if no step is in flight or the mapping hasn't completed yet.
/// - `Some(Ok(vec))` when results are ready.
/// - `Some(Err(msg))` when the GPU operation failed.
#[cfg(all(feature = "gpu", target_arch = "wasm32"))]
pub fn wasm_try_finish_step_dynamics() -> Option<Result<Vec<GpuUnit>, String>> {
    GPU_PENDING.with(|p| {
        let mut guard = p.borrow_mut();
        let status: Result<(), String> = {
            let pending = guard.as_ref()?;
            let ready = pending.ready.borrow();
            match ready.as_ref() {
                None => return None,
                Some(s) => s.clone(),
            }
        };

        // Mapping completed: consume the pending state.
        let PendingGpuReadback { n, staging, .. } = guard.take()?;

        match status {
            Ok(()) => {
                let slice = staging.slice(..);
                let data = slice.get_mapped_range();
                let all: &[GpuUnit] = bytemuck::cast_slice(&data);
                let mut out = vec![
                    GpuUnit {
                        amp: 0.0,
                        phase: 0.0,
                        bias: 0.0,
                        decay: 0.0,
                    };
                    n
                ];
                out.copy_from_slice(&all[..n]);
                drop(data);
                staging.unmap();
                Some(Ok(out))
            }
            Err(e) => {
                staging.unmap();
                Some(Err(e.clone()))
            }
        }
    })
}

/// Cancel any in-flight wasm GPU step.
#[cfg(all(feature = "gpu", target_arch = "wasm32"))]
pub fn wasm_cancel_pending_step() {
    GPU_PENDING.with(|p| {
        if let Some(pending) = p.borrow_mut().take() {
            pending.staging.unmap();
        }
    });
}

/// WGSL compute shader for dynamics update.
const DYNAMICS_SHADER: &str = r#"
struct Unit {
    amp: f32,
    phase: f32,
    bias: f32,
    decay: f32,
}

struct Influence {
    amp: f32,
    phase: f32,
    noise_amp: f32,
    noise_phase: f32,
}

struct Input {
    value: f32,
    _padding: vec3<f32>,
}

struct Params {
    dt: f32,
    base_freq: f32,
    inhibition: f32,
    unit_count: u32,
}

@group(0) @binding(0) var<storage, read> units_in: array<Unit>;
@group(0) @binding(1) var<storage, read> influences: array<Influence>;
@group(0) @binding(2) var<storage, read> inputs: array<Input>;
@group(0) @binding(3) var<uniform> params: Params;
@group(0) @binding(4) var<storage, read_write> units_out: array<Unit>;

const PI: f32 = 3.14159265359;
const TAU: f32 = 6.28318530718;

fn wrap_angle(a: f32) -> f32 {
    var angle = a;
    // Handle angles > PI
    if angle > PI {
        angle = angle - TAU;
    }
    // Handle angles < -PI
    if angle < -PI {
        angle = angle + TAU;
    }
    return angle;
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if i >= params.unit_count {
        return;
    }

    let u = units_in[i];
    let inf = influences[i];
    let input = inputs[i].value;

    let damp = u.decay * u.amp;
    let d_amp = (u.bias + input + inf.amp - params.inhibition - damp + inf.noise_amp) * params.dt;
    let d_phase = (params.base_freq + inf.phase + inf.noise_phase) * params.dt;

    var new_amp = u.amp + d_amp;
    new_amp = clamp(new_amp, -2.0, 2.0);

    let new_phase = wrap_angle(u.phase + d_phase);

    units_out[i].amp = new_amp;
    units_out[i].phase = new_phase;
    units_out[i].bias = u.bias;
    units_out[i].decay = u.decay;
}
"#;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gpu_context_creation() {
        // This test may fail on systems without GPU support
        let ctx = GpuContext::new(1024);
        if ctx.is_some() {
            println!("GPU context created successfully");
        } else {
            println!("No GPU available (expected in some CI environments)");
        }
    }
}
