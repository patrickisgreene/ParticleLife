//! GPU state is owned by the render world. Particle data never crosses back to
//! the CPU during normal use; the display renderer consumes the same buffers.
use crate::model::*;
use bevy::{
    core_pipeline::schedule::camera_driver,
    prelude::*,
    render::{
        RenderApp, RenderStartup,
        extract_resource::{ExtractResource, ExtractResourcePlugin},
        render_asset::RenderAssets,
        render_resource::*,
        renderer::{RenderAdapterInfo, RenderContext, RenderDevice, RenderGraph, RenderQueue},
        texture::GpuImage,
    },
};
use std::{borrow::Cow, time::Instant};

#[derive(Resource, Clone, ExtractResource)]
pub struct DisplayTarget(pub Handle<Image>);

pub struct ParticleGpuPlugin;
impl Plugin for ParticleGpuPlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins((
            ExtractResourcePlugin::<Simulation>::default(),
            ExtractResourcePlugin::<Appearance>::default(),
            ExtractResourcePlugin::<Status>::default(),
            ExtractResourcePlugin::<DisplayTarget>::default(),
        ));
        app.sub_app_mut(RenderApp)
            .init_resource::<GpuState>()
            .add_systems(RenderStartup, setup_pipelines)
            .add_systems(RenderGraph, simulate_and_draw.before(camera_driver));
    }
}

#[derive(Clone, ShaderType)]
struct Params {
    counts: UVec4,
    physics: Vec4,
    world: Vec4,
    view: Vec4,
    timing: Vec4,
    flags: UVec4,
    trails: Vec4,
    trail_view: Vec4,
    behavior: Vec4,
    cycle: Vec4,
    sampling: Vec4,
}
impl Params {
    fn new(s: &Simulation, a: &Appearance, alpha: f32, generation: u64) -> Self {
        let (origin, tiles) = a.visible_tiles();
        Self {
            // y carries the vertical tile count; the splat kernel needs both axes,
            // while the raster path infers it from the instance count.
            sampling: Vec4::new(s.behavior.sample_budget as f32, tiles.y as f32, 0.0, 0.0),
            behavior: Vec4::new(
                if s.behavior.preferred_enabled {
                    1.0
                } else {
                    0.0
                },
                if s.behavior.density_enabled {
                    s.behavior.density_strength
                } else {
                    0.0
                },
                s.behavior.density_target,
                0.0,
            ),
            cycle: Vec4::new(
                s.behavior.cycle_mode as f32,
                s.behavior.cycle_seconds,
                s.behavior.cycle_fraction,
                s.behavior.cycle_min_neighbors as f32,
            ),
            trails: Vec4::new(
                if s.trails.enabled { 1.0 } else { 0.0 },
                s.trails.deposit,
                s.trails.diffusion,
                s.trails.half_life,
            ),
            trail_view: Vec4::new(
                s.trails.response,
                s.trails.sensor_distance,
                s.trails.visibility,
                TRAIL_SIDE as f32,
            ),
            counts: UVec4::new(s.count, s.types, s.grid_side(), s.seed),
            physics: Vec4::new(DT, s.radius, s.strength, s.damping),
            world: Vec4::new(WORLD, 0.2, 180.0, a.zoom),
            view: Vec4::new(a.aspect, a.pan.x, a.pan.y, a.size),
            timing: Vec4::new(alpha, a.glow, origin.x as f32, origin.y as f32),
            flags: UVec4::new(
                u32::from(s.exact),
                generation as u32,
                u32::from(s.clustered),
                tiles.x,
            ),
        }
    }
}

#[derive(Resource)]
struct Pipelines {
    compute_layout: BindGroupLayoutDescriptor,
    draw_layout: BindGroupLayoutDescriptor,
    compute: Vec<CachedComputePipelineId>,
    draw: CachedRenderPipelineId,
    background: CachedRenderPipelineId,
    splat: CachedComputePipelineId,
    resolve: CachedRenderPipelineId,
}
fn buffer_entry(
    binding: u32,
    visibility: ShaderStages,
    ty: BufferBindingType,
) -> BindGroupLayoutEntry {
    BindGroupLayoutEntry {
        binding,
        visibility,
        ty: BindingType::Buffer {
            ty,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}
/// Compute entry points, indexed by the constants below. Append new kernels to
/// the end so the existing indices stay valid.
const KERNELS: [&str; 10] = [
    "init",
    "clear",
    "count",
    "prefix",
    "prefix_blocks",
    "scatter",
    "update",
    "clear_trails",
    "diffuse_trails",
    "commit_trails",
];
const KERNEL_INIT: usize = 0;
const KERNEL_GRID: usize = 1;
const KERNEL_UPDATE: usize = 6;
const KERNEL_CLEAR_TRAILS: usize = 7;
const KERNEL_DIFFUSE_TRAILS: usize = 8;
const KERNEL_COMMIT_TRAILS: usize = 9;

fn setup_pipelines(mut commands: Commands, assets: Res<AssetServer>, cache: Res<PipelineCache>) {
    let mut entries = vec![buffer_entry(
        0,
        ShaderStages::COMPUTE,
        BufferBindingType::Uniform,
    )];
    entries.extend((1..=8).map(|i| {
        buffer_entry(
            i,
            ShaderStages::COMPUTE,
            BufferBindingType::Storage { read_only: i == 6 },
        )
    }));
    let compute_layout = BindGroupLayoutDescriptor::new("particle simulation", &entries);
    let shader = assets.load("shaders/simulation.wgsl");
    let compute = KERNELS
        .into_iter()
        .map(|entry| {
            cache.queue_compute_pipeline(ComputePipelineDescriptor {
                label: Some(Cow::Owned(format!("particle {entry}"))),
                layout: vec![compute_layout.clone()],
                shader: shader.clone(),
                entry_point: Some(entry.into()),
                ..default()
            })
        })
        .collect();
    // Shared with the splat compute pipeline, so the two display paths read the
    // same particle buffers through one bind group.
    let stages = ShaderStages::VERTEX_FRAGMENT | ShaderStages::COMPUTE;
    let draw_layout = BindGroupLayoutDescriptor::new(
        "particle display",
        &[
            buffer_entry(0, stages, BufferBindingType::Uniform),
            buffer_entry(1, stages, BufferBindingType::Storage { read_only: true }),
            buffer_entry(2, stages, BufferBindingType::Storage { read_only: true }),
            buffer_entry(3, stages, BufferBindingType::Storage { read_only: true }),
            buffer_entry(4, stages, BufferBindingType::Storage { read_only: true }),
            // Writable storage may not be visible to the vertex stage.
            buffer_entry(
                5,
                ShaderStages::FRAGMENT | ShaderStages::COMPUTE,
                BufferBindingType::Storage { read_only: false },
            ),
        ],
    );
    let shader = assets.load("shaders/particles.wgsl");
    let descriptor = RenderPipelineDescriptor {
        label: Some("soft particle renderer".into()),
        layout: vec![draw_layout.clone()],
        vertex: VertexState {
            shader: shader.clone(),
            entry_point: Some("vertex".into()),
            ..default()
        },
        fragment: Some(FragmentState {
            shader,
            entry_point: Some("fragment".into()),
            targets: vec![Some(ColorTargetState {
                format: TextureFormat::Rgba16Float,
                blend: Some(BlendState {
                    color: BlendComponent {
                        src_factor: BlendFactor::One,
                        dst_factor: BlendFactor::One,
                        operation: BlendOperation::Add,
                    },
                    alpha: BlendComponent::OVER,
                }),
                write_mask: ColorWrites::ALL,
            })],
            ..default()
        }),
        ..default()
    };
    let mut background_descriptor = descriptor.clone();
    background_descriptor.label = Some("infinite background".into());
    background_descriptor.vertex.entry_point = Some("background_vertex".into());
    let fragment = background_descriptor.fragment.as_mut().unwrap();
    fragment.entry_point = Some("background_fragment".into());
    fragment.targets[0].as_mut().unwrap().blend = None;
    let mut resolve_descriptor = descriptor.clone();
    resolve_descriptor.label = Some("splat resolve".into());
    resolve_descriptor.vertex.entry_point = Some("resolve_vertex".into());
    resolve_descriptor
        .fragment
        .as_mut()
        .unwrap()
        .entry_point = Some("resolve_fragment".into());
    let resolve = cache.queue_render_pipeline(resolve_descriptor);
    let splat = cache.queue_compute_pipeline(ComputePipelineDescriptor {
        label: Some("particle splat".into()),
        layout: vec![draw_layout.clone()],
        shader: descriptor.vertex.shader.clone(),
        entry_point: Some("splat".into()),
        ..default()
    });
    let background = cache.queue_render_pipeline(background_descriptor);
    let draw = cache.queue_render_pipeline(descriptor);
    commands.insert_resource(Pipelines {
        compute_layout,
        draw_layout,
        compute,
        draw,
        background,
        splat,
        resolve,
    });
}

struct Buffers {
    particles: [Buffer; 2],
    cells: Buffer,
    neighbors: Buffer,
    sorted: Buffer,
    blocks: Buffer,
    types: Buffer,
    chemicals: Buffer,
    accumulation: Buffer,
    accumulation_pixels: u32,
    trail_revision: u64,
    epoch: u64,
    grid_side: u32,
    rules_revision: u64,
    palette: usize,
}
fn storage(device: &RenderDevice, label: &'static str, size: u64) -> Buffer {
    device.create_buffer(&BufferDescriptor {
        label: Some(label),
        size,
        usage: BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    })
}
impl Buffers {
    fn new(device: &RenderDevice, s: &Simulation) -> Self {
        let n = u64::from(s.count);
        let side = u64::from(s.grid_side());
        Self {
            particles: [
                storage(device, "particles A", n * 32),
                storage(device, "particles B", n * 32),
            ],
            cells: storage(device, "spatial cells", side * side * 16),
            neighbors: storage(device, "packed neighbor records", n * 16),
            sorted: storage(device, "cell-sorted particles", n * 32),
            blocks: storage(device, "prefix blocks", 256 * 4),
            types: storage(
                device,
                "colors and interactions",
                type_data(s).len() as u64 * 4,
            ),
            chemicals: storage(device, "chemical field", TRAIL_BYTES),
            // Sized on first use from the display target; storage buffers may
            // not be zero-length.
            accumulation: storage(device, "splat accumulation", 16),
            accumulation_pixels: 0,
            trail_revision: s.trails.clear_revision,
            epoch: s.epoch,
            grid_side: s.grid_side(),
            rules_revision: 0,
            palette: usize::MAX,
        }
    }
    fn compute_group(
        &self,
        device: &RenderDevice,
        layout: &BindGroupLayout,
        uniform: &UniformBuffer<Params>,
        current: usize,
    ) -> BindGroup {
        device.create_bind_group(
            "simulation bindings",
            layout,
            &BindGroupEntries::sequential((
                uniform,
                self.particles[current].as_entire_binding(),
                self.particles[1 - current].as_entire_binding(),
                self.cells.as_entire_binding(),
                self.neighbors.as_entire_binding(),
                self.blocks.as_entire_binding(),
                self.types.as_entire_binding(),
                self.chemicals.as_entire_binding(),
                self.sorted.as_entire_binding(),
            )),
        )
    }
    /// `update` reorders particles, so `particles[1 - current]` is no longer
    /// index-aligned with `particles[current]`. `sorted` holds the same particles
    /// in the same order as the update's output, so it is the interpolation start.
    fn draw_group(
        &self,
        device: &RenderDevice,
        layout: &BindGroupLayout,
        uniform: &UniformBuffer<Params>,
        current: usize,
    ) -> BindGroup {
        device.create_bind_group(
            "display bindings",
            layout,
            &BindGroupEntries::sequential((
                uniform,
                self.particles[current].as_entire_binding(),
                self.sorted.as_entire_binding(),
                self.types.as_entire_binding(),
                self.chemicals.as_entire_binding(),
                self.accumulation.as_entire_binding(),
            )),
        )
    }
}
#[derive(Resource)]
struct GpuState {
    buffers: Option<Buffers>,
    current: usize,
    generation: u64,
    accumulator: f32,
    last_step: u64,
    report_at: Instant,
    report_steps: u32,
}
impl Default for GpuState {
    fn default() -> Self {
        Self {
            buffers: None,
            current: 0,
            generation: 0,
            accumulator: 0.0,
            last_step: 0,
            report_at: Instant::now(),
            report_steps: 0,
        }
    }
}
fn uniform(device: &RenderDevice, queue: &RenderQueue, params: Params) -> UniformBuffer<Params> {
    let mut buffer = UniformBuffer::from(params);
    buffer.write_buffer(device, queue);
    buffer
}

#[allow(clippy::too_many_arguments)]
fn simulate_and_draw(
    mut context: RenderContext,
    device: Res<RenderDevice>,
    queue: Res<RenderQueue>,
    adapter: Res<RenderAdapterInfo>,
    pipelines: Res<Pipelines>,
    cache: Res<PipelineCache>,
    sim: Res<Simulation>,
    appearance: Res<Appearance>,
    target: Res<DisplayTarget>,
    images: Res<RenderAssets<GpuImage>>,
    status: Res<Status>,
    mut state: ResMut<GpuState>,
) {
    {
        let mut report = status.0.lock().unwrap();
        report.adapter.clone_from(&adapter.name);
        let limits = device.limits();
        report.max_storage = u64::from(limits.max_storage_buffer_binding_size);
        report.max_buffer = limits.max_buffer_size;
        report.max_dispatch = limits.max_compute_workgroups_per_dimension;
        let validation = validate_counts(sim.count, sim.types, &report);
        if let Err(error) = validation {
            report.message = error;
            return;
        }
    }
    for id in &pipelines.compute {
        match cache.get_compute_pipeline_state(*id) {
            CachedPipelineState::Ok(_) => (),
            CachedPipelineState::Err(error) => {
                status.0.lock().unwrap().message = format!("Shader: {error}");
                return;
            }
            _ => {
                status.0.lock().unwrap().message = "Compiling GPU simulation…".into();
                return;
            }
        }
    }
    let Some(draw_pipeline) = cache.get_render_pipeline(pipelines.draw) else {
        status.0.lock().unwrap().message = format!(
            "Preparing particle renderer: {:?}",
            cache.get_render_pipeline_state(pipelines.draw)
        );
        return;
    };
    let Some(background_pipeline) = cache.get_render_pipeline(pipelines.background) else {
        status.0.lock().unwrap().message = format!(
            "Preparing background: {:?}",
            cache.get_render_pipeline_state(pipelines.background)
        );
        return;
    };
    let Some(resolve_pipeline) = cache.get_render_pipeline(pipelines.resolve) else {
        status.0.lock().unwrap().message = format!(
            "Preparing splat resolve: {:?}",
            cache.get_render_pipeline_state(pipelines.resolve)
        );
        return;
    };
    let Some(splat_pipeline) = cache.get_compute_pipeline(pipelines.splat) else {
        status.0.lock().unwrap().message = "Compiling particle splat…".into();
        return;
    };
    let Some(image) = images.get(&target.0) else {
        return;
    };
    let reset = state.buffers.as_ref().is_none_or(|b| b.epoch != sim.epoch);
    if reset {
        state.buffers = Some(Buffers::new(&device, &sim));
        state.current = 0;
        state.generation = 0;
        state.accumulator = 0.0;
        state.last_step = sim.step;
    }
    let buffers = state.buffers.as_mut().unwrap();
    if buffers.grid_side != sim.grid_side() {
        buffers.grid_side = sim.grid_side();
        buffers.cells = storage(
            &device,
            "spatial cells",
            u64::from(sim.grid_side()).pow(2) * 16,
        );
    }
    if buffers.rules_revision != sim.rules_revision || buffers.palette != sim.palette {
        let data = type_data(&sim);
        if buffers.types.size() != data.len() as u64 * 4 {
            buffers.types = storage(&device, "colors and interactions", data.len() as u64 * 4);
        }
        queue.write_buffer(&buffers.types, 0, bytemuck::cast_slice(&data));
        buffers.rules_revision = sim.rules_revision;
        buffers.palette = sim.palette;
    }
    let compute_layout = cache.get_bind_group_layout(&pipelines.compute_layout);
    let draw_layout = cache.get_bind_group_layout(&pipelines.draw_layout);
    let clear_trails =
        reset || state.buffers.as_ref().unwrap().trail_revision != sim.trails.clear_revision;
    if clear_trails {
        state.buffers.as_mut().unwrap().trail_revision = sim.trails.clear_revision;
        let params = uniform(
            &device,
            &queue,
            Params::new(&sim, &appearance, 1.0, state.generation),
        );
        let group = state.buffers.as_ref().unwrap().compute_group(
            &device,
            &compute_layout,
            &params,
            state.current,
        );
        let mut pass = context
            .command_encoder()
            .begin_compute_pass(&ComputePassDescriptor::default());
        pass.set_pipeline(
            cache
                .get_compute_pipeline(pipelines.compute[KERNEL_CLEAR_TRAILS])
                .unwrap(),
        );
        pass.set_bind_group(0, &group, &[]);
        pass.dispatch_workgroups((TRAIL_SIDE * TRAIL_SIDE).div_ceil(256), 1, 1);
    }
    if reset {
        let params = uniform(&device, &queue, Params::new(&sim, &appearance, 1.0, 0));
        let group =
            state
                .buffers
                .as_ref()
                .unwrap()
                .compute_group(&device, &compute_layout, &params, 0);
        let mut pass = context
            .command_encoder()
            .begin_compute_pass(&ComputePassDescriptor::default());
        pass.set_pipeline(
            cache
                .get_compute_pipeline(pipelines.compute[KERNEL_INIT])
                .unwrap(),
        );
        pass.set_bind_group(0, &group, &[]);
        pass.dispatch_workgroups(sim.count.div_ceil(256), 1, 1);
    }
    let manual_steps = sim.step.saturating_sub(state.last_step).min(4) as u32;
    state.last_step = sim.step;
    if sim.paused {
        state.accumulator = 0.0;
    } else {
        state.accumulator = (state.accumulator + sim.frame_dt.min(0.1) * sim.speed).min(4.0 * DT);
    }
    let steps = if sim.paused {
        manual_steps
    } else {
        (state.accumulator / DT) as u32
    };
    // Each step has a distinct uniform allocation: queue writes cannot overwrite
    // uniforms referenced by earlier dispatches in the same submitted frame.
    for _ in 0..steps {
        let params = uniform(
            &device,
            &queue,
            Params::new(&sim, &appearance, 1.0, state.generation),
        );
        let group = state.buffers.as_ref().unwrap().compute_group(
            &device,
            &compute_layout,
            &params,
            state.current,
        );
        for stage in KERNEL_GRID..=KERNEL_UPDATE {
            let dispatch = match stage {
                1 | 3 => (sim.grid_side() * sim.grid_side()).div_ceil(256),
                4 => 1,
                _ => sim.count.div_ceil(256),
            };
            let mut pass = context
                .command_encoder()
                .begin_compute_pass(&ComputePassDescriptor::default());
            pass.set_pipeline(
                cache
                    .get_compute_pipeline(pipelines.compute[stage])
                    .unwrap(),
            );
            pass.set_bind_group(0, &group, &[]);
            pass.dispatch_workgroups(dispatch, 1, 1);
        }
        if sim.trails.enabled {
            for stage in [KERNEL_DIFFUSE_TRAILS, KERNEL_COMMIT_TRAILS] {
                let mut pass = context
                    .command_encoder()
                    .begin_compute_pass(&ComputePassDescriptor::default());
                pass.set_pipeline(
                    cache
                        .get_compute_pipeline(pipelines.compute[stage])
                        .unwrap(),
                );
                pass.set_bind_group(0, &group, &[]);
                pass.dispatch_workgroups((TRAIL_SIDE * TRAIL_SIDE).div_ceil(256), 1, 1);
            }
        }
        state.current = 1 - state.current;
        state.generation += 1;
        if !sim.paused {
            state.accumulator = (state.accumulator - DT).max(0.0);
        }
    }
    let alpha = if sim.paused {
        1.0
    } else {
        (state.accumulator / DT).clamp(0.0, 1.0)
    };
    let size = image.texture_descriptor.size;
    let pixels = size.width * size.height;
    {
        let buffers = state.buffers.as_mut().unwrap();
        if buffers.accumulation_pixels != pixels {
            buffers.accumulation_pixels = pixels;
            // Three u32 channels per pixel, accumulated with atomics.
            buffers.accumulation =
                storage(&device, "splat accumulation", u64::from(pixels) * 12);
        }
    }
    let mut display_params = Params::new(&sim, &appearance, alpha, state.generation);
    // Reserved display-only component: physical target height for a constant pixel stroke.
    display_params.behavior.w = size.height as f32;
    display_params.sampling.z = size.width as f32;
    display_params.sampling.w = size.height as f32;
    // The quad half-extent in pixels. Below about a pixel the rasterizer drops
    // dots that miss a pixel center, and pays six vertices per tile copy to do
    // it, so the splat path is both cheaper and more correct.
    let dot_pixels = appearance.size / WORLD * appearance.zoom * size.height as f32;
    let use_splat = dot_pixels < 1.5;
    let params = uniform(&device, &queue, display_params);
    let group =
        state
            .buffers
            .as_ref()
            .unwrap()
            .draw_group(&device, &draw_layout, &params, state.current);
    if use_splat {
        let accumulation = &state.buffers.as_ref().unwrap().accumulation;
        context
            .command_encoder()
            .clear_buffer(accumulation, 0, None);
        let mut pass = context
            .command_encoder()
            .begin_compute_pass(&ComputePassDescriptor::default());
        pass.set_pipeline(splat_pipeline);
        pass.set_bind_group(0, &group, &[]);
        pass.dispatch_workgroups(sim.count.div_ceil(256), 1, 1);
    }
    {
        let mut pass = context
            .command_encoder()
            .begin_render_pass(&RenderPassDescriptor {
                label: Some("glowing particles"),
                color_attachments: &[Some(RenderPassColorAttachment {
                    view: &image.texture_view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: Operations {
                        load: LoadOp::Clear(LinearRgba::new(0.002, 0.004, 0.012, 1.0).into()),
                        store: StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
        pass.set_bind_group(0, &group, &[]);
        pass.set_pipeline(background_pipeline);
        pass.draw(0..3, 0..1);
        if use_splat {
            pass.set_pipeline(resolve_pipeline);
            pass.draw(0..3, 0..1);
        } else {
            pass.set_pipeline(draw_pipeline);
            let (_, tiles) = appearance.visible_tiles();
            pass.draw(0..6, 0..sim.count * tiles.x * tiles.y);
        }
    }
    state.report_steps += steps;
    let mut report = status.0.lock().unwrap();
    report.generation = state.generation;
    report.message.clear();
    let elapsed = state.report_at.elapsed().as_secs_f32();
    if elapsed >= 1.0 {
        report.steps_per_second = state.report_steps as f32 / elapsed;
        eprintln!("BENCH count={} steps_per_second={:.1} splat={}", sim.count, report.steps_per_second, use_splat);
        state.report_steps = 0;
        state.report_at = Instant::now();
    }
}

#[cfg(test)]
mod shader_tests {
    #[test]
    fn shaders_validate() {
        for source in [
            include_str!("../assets/shaders/particles.wgsl"),
            include_str!("../assets/shaders/simulation.wgsl"),
        ] {
            let module = naga::front::wgsl::parse_str(source).expect("valid WGSL");
            naga::valid::Validator::new(
                naga::valid::ValidationFlags::all(),
                naga::valid::Capabilities::all(),
            )
            .validate(&module)
            .expect("valid shader interfaces and operations");
        }
    }
}
