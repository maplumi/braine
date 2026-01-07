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

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Dynamics Bind Group Layout"),
            entries: &[
                // Units buffer (read-write)
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Influences buffer (read-only)
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Inputs buffer (read-only)
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Params uniform
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Dynamics Pipeline Layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Dynamics Pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        Some(Self {
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
        let units_buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Units Buffer"),
                contents: bytemuck::cast_slice(units),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
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
                    resource: units_buffer.as_entire_binding(),
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
            &units_buffer,
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

@group(0) @binding(0) var<storage, read_write> units: array<Unit>;
@group(0) @binding(1) var<storage, read> influences: array<Influence>;
@group(0) @binding(2) var<storage, read> inputs: array<Input>;
@group(0) @binding(3) var<uniform> params: Params;

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

    let u = units[i];
    let inf = influences[i];
    let input = inputs[i].value;

    let damp = u.decay * u.amp;
    let d_amp = (u.bias + input + inf.amp - params.inhibition - damp + inf.noise_amp) * params.dt;
    let d_phase = (params.base_freq + inf.phase + inf.noise_phase) * params.dt;

    var new_amp = u.amp + d_amp;
    new_amp = clamp(new_amp, -2.0, 2.0);

    let new_phase = wrap_angle(u.phase + d_phase);

    units[i].amp = new_amp;
    units[i].phase = new_phase;
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
