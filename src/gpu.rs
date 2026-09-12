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
    detect: Vec4,
    outline: Vec4,
    evolution: Vec4,
}
impl Params {
    fn new(
        s: &Simulation,
        a: &Appearance,
        alpha: f32,
        generation: u64,
        detect: Vec4,
        outline: Vec4,
    ) -> Self {
        let (origin, tiles) = a.visible_tiles();
        Self {
            detect,
            outline,
            evolution: Vec4::new(
                if s.evolution.enabled { 1.0 } else { 0.0 },
                s.evolution.mutation,
                s.evolution.cull_free_fraction,
                s.evolution.influence,
            ),
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
            world: Vec4::new(WORLD, CORE_FRACTION, SPEED_LIMIT, a.zoom),
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
    outline_splat: CachedComputePipelineId,
    outline: CachedRenderPipelineId,
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
const KERNELS: [&str; 24] = [
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
    "clear_creatures",
    "bond",
    "resolve",
    "creature_stats",
    "promote",
    "reset_evolution",
    "progeny",
    "creature_frame",
    "promote_frame",
    "count_free",
    "cull_select",
    "cull_release",
    "pick_split",
    "apply_split",
];
const KERNEL_INIT: usize = 0;
const KERNEL_GRID: usize = 1;
const KERNEL_SCATTER: usize = 5;
const KERNEL_UPDATE: usize = 6;
const KERNEL_CLEAR_TRAILS: usize = 7;
const KERNEL_DIFFUSE_TRAILS: usize = 8;
const KERNEL_COMMIT_TRAILS: usize = 9;
const KERNEL_CLEAR_CREATURES: usize = 10;
const KERNEL_BOND: usize = 11;
const KERNEL_RESOLVE: usize = 12;
const KERNEL_STATS: usize = 13;
const KERNEL_PROMOTE: usize = 14;
const KERNEL_RESET_EVOLUTION: usize = 15;
const KERNEL_PROGENY: usize = 16;
const KERNEL_FRAME: usize = 17;
const KERNEL_PROMOTE_FRAME: usize = 18;
const KERNEL_COUNT_FREE: usize = 19;
const KERNEL_CULL_SELECT: usize = 20;
const KERNEL_CULL_RELEASE: usize = 21;
const KERNEL_PICK_SPLIT: usize = 22;
const KERNEL_APPLY_SPLIT: usize = 23;
/// Pointer-jump passes per step. Each doubles the merge depth that collapses in
/// one step; four covers chains of 16, and anything deeper finishes next step.
const RESOLVE_PASSES: usize = 4;

fn setup_pipelines(mut commands: Commands, assets: Res<AssetServer>, cache: Res<PipelineCache>) {
    let mut entries = vec![buffer_entry(
        0,
        ShaderStages::COMPUTE,
        BufferBindingType::Uniform,
    )];
    entries.extend((1..=11).map(|i| {
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
            buffer_entry(
                6,
                ShaderStages::FRAGMENT | ShaderStages::COMPUTE,
                BufferBindingType::Storage { read_only: false },
            ),
            buffer_entry(
                7,
                ShaderStages::FRAGMENT | ShaderStages::COMPUTE,
                BufferBindingType::Storage { read_only: true },
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
    let mut outline_descriptor = descriptor.clone();
    outline_descriptor.label = Some("creature outlines".into());
    outline_descriptor.vertex.entry_point = Some("background_vertex".into());
    outline_descriptor.fragment.as_mut().unwrap().entry_point = Some("outline_fragment".into());
    let outline = cache.queue_render_pipeline(outline_descriptor);
    let outline_splat = cache.queue_compute_pipeline(ComputePipelineDescriptor {
        label: Some("creature outline field".into()),
        layout: vec![draw_layout.clone()],
        shader: descriptor.vertex.shader.clone(),
        entry_point: Some("outline_splat".into()),
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
        outline_splat,
        outline,
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
    creatures: Buffer,
    genomes: Buffer,
    births: Buffer,
    field: Buffer,
    trail_revision: u64,
    epoch: u64,
    grid_side: u32,
    rules_revision: u64,
    palette: usize,
    evolution_revision: u64,
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
            creatures: storage(
                device,
                "creature registry",
                u64::from(MAX_CREATURES) * CREATURE_BYTES,
            ),
            // One vec2 (ratio, orientation) per creature per interaction kind,
            // plus a leading words region; sized for the full rule cache cap.
            genomes: storage(
                device,
                "creature genomes",
                u64::from(MAX_CREATURES) * u64::from(CREATURE_TYPES) * 8,
            ),
            // One marker/id pair per creature id for the split pass; the pair is
            // rewritten every step while detection runs, so zero is just the
            // boot state.
            births: storage(device, "creature birth plans", u64::from(MAX_CREATURES) * 8),
            field: storage(device, "creature occupancy field", FIELD_BYTES),
            trail_revision: s.trails.clear_revision,
            epoch: s.epoch,
            grid_side: s.grid_side(),
            rules_revision: 0,
            palette: usize::MAX,
            evolution_revision: 0,
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
                self.creatures.as_entire_binding(),
                self.genomes.as_entire_binding(),
                self.births.as_entire_binding(),
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
                self.field.as_entire_binding(),
                self.creatures.as_entire_binding(),
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
    let Some(outline_pipeline) = cache.get_render_pipeline(pipelines.outline) else {
        status.0.lock().unwrap().message = format!(
            "Preparing creature outlines: {:?}",
            cache.get_render_pipeline_state(pipelines.outline)
        );
        return;
    };
    let Some(outline_splat_pipeline) = cache.get_compute_pipeline(pipelines.outline_splat) else {
        status.0.lock().unwrap().message = "Compiling creature outlines…".into();
        return;
    };
    let Some(image) = images.get(&target.0) else {
        return;
    };
    // Derived from the rule matrices, so recomputed only as often as the frame.
    let detect = sim.detection.enabled;
    let detect_params = Vec4::new(
        crate::creatures::bond_radius(&sim),
        crate::creatures::min_particles(&sim) as f32,
        sim.detection.promote_steps as f32,
        if detect { 1.0 } else { 0.0 },
    );
    // Stamp radius and isolevel both derive from the inferred rest spacing, so the
    // outline tracks the rule set instead of a tuned constant.
    let (stamp, isolevel) = crate::creatures::outline_geometry(&sim, FIELD_SIDE);
    let outlines = detect && sim.detection.outline > 0.0;
    let outline_params = Vec4::new(
        stamp,
        isolevel,
        if outlines { sim.detection.outline } else { 0.0 },
        FIELD_SIDE as f32,
    );
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
            Params::new(&sim, &appearance, 1.0, state.generation, detect_params, outline_params),
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
        let params = uniform(&device, &queue, Params::new(&sim, &appearance, 1.0, 0, detect_params, outline_params));
        let group =
            state
                .buffers
                .as_ref()
                .unwrap()
                .compute_group(&device, &compute_layout, &params, 0);
        {
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
        // A fresh epoch also clears every genome back to isotropic, along with
        // the newborn history each slot accumulates, so recycled ids do not carry
        // an old body's line into the new world.
        {
            let mut pass = context
                .command_encoder()
                .begin_compute_pass(&ComputePassDescriptor::default());
            pass.set_pipeline(
                cache
                    .get_compute_pipeline(pipelines.compute[KERNEL_RESET_EVOLUTION])
                    .unwrap(),
            );
            pass.set_bind_group(0, &group, &[]);
            pass.dispatch_workgroups(MAX_CREATURES.div_ceil(256), 1, 1);
        }
    }
    let genome_reset = state.buffers.as_ref().unwrap().evolution_revision != sim.evolution.evolution_revision;
    if genome_reset {
        state.buffers.as_mut().unwrap().evolution_revision = sim.evolution.evolution_revision;
        let params = uniform(
            &device,
            &queue,
            Params::new(&sim, &appearance, 1.0, state.generation, detect_params, outline_params),
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
                .get_compute_pipeline(pipelines.compute[KERNEL_RESET_EVOLUTION])
                .unwrap(),
        );
        pass.set_bind_group(0, &group, &[]);
        pass.dispatch_workgroups(MAX_CREATURES.div_ceil(256), 1, 1);
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
            Params::new(&sim, &appearance, 1.0, state.generation, detect_params, outline_params),
        );
        let group = state.buffers.as_ref().unwrap().compute_group(
            &device,
            &compute_layout,
            &params,
            state.current,
        );
        // Ordered explicitly rather than as a contiguous range: detection has to
        // land between the grid build and the force pass, because `update` writes
        // the resolved creature id into each particle.
        let particle_groups = sim.count.div_ceil(256);
        let creature_groups = MAX_CREATURES.div_ceil(256);
        let mut schedule: Vec<(usize, u32)> = (KERNEL_GRID..=KERNEL_SCATTER)
            .map(|stage| {
                let groups = match stage {
                    1 | 3 => (sim.grid_side() * sim.grid_side()).div_ceil(256),
                    4 => 1,
                    _ => particle_groups,
                };
                (stage, groups)
            })
            .collect();
        if detect {
            schedule.push((KERNEL_CLEAR_CREATURES, creature_groups));
            schedule.push((KERNEL_BOND, particle_groups));
            schedule.extend(
                std::iter::repeat_n((KERNEL_RESOLVE, creature_groups), RESOLVE_PASSES),
            );
        }
        schedule.push((KERNEL_UPDATE, particle_groups));
        if detect {
            schedule.push((KERNEL_STATS, particle_groups));
            if sim.evolution.enabled {
                // `progeny` must see a dissolved parent's `flags` while it still
                // reads as tracked, so it has to run before `promote` zeroes the
                // body it left behind. That is what lets a merge pass its line on.
                schedule.push((KERNEL_PROGENY, particle_groups));
            }
            schedule.push((KERNEL_PROMOTE, creature_groups));
            if sim.evolution.enabled {
                // Births, frame measurement and culling all need the same step's
                // counts and centroid, so they stack behind `promote`. The order
                // here is load-bearing: `promote` sets `centroid`, `progeny`
                // feeds lineage, `creature_frame` consumes `centroid`, and
                // `promote_frame` consumes both the accumulated frame and the
                // lineage. Elongated bodies then birth a child that clears and
                // re-tracks on its own, before the free pool is audited.
                schedule.push((KERNEL_FRAME, particle_groups));
                schedule.push((KERNEL_PROMOTE_FRAME, creature_groups));
                schedule.push((KERNEL_PICK_SPLIT, creature_groups));
                schedule.push((KERNEL_APPLY_SPLIT, particle_groups));
                schedule.push((KERNEL_COUNT_FREE, particle_groups));
                schedule.push((KERNEL_CULL_SELECT, creature_groups));
                schedule.push((KERNEL_CULL_RELEASE, particle_groups));
            }
        }
        if sim.trails.enabled {
            let trail_groups = (TRAIL_SIDE * TRAIL_SIDE).div_ceil(256);
            schedule.push((KERNEL_DIFFUSE_TRAILS, trail_groups));
            schedule.push((KERNEL_COMMIT_TRAILS, trail_groups));
        }
        for (stage, groups) in schedule {
            let mut pass = context
                .command_encoder()
                .begin_compute_pass(&ComputePassDescriptor::default());
            pass.set_pipeline(
                cache
                    .get_compute_pipeline(pipelines.compute[stage])
                    .unwrap(),
            );
            pass.set_bind_group(0, &group, &[]);
            pass.dispatch_workgroups(groups, 1, 1);
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
    let mut display_params = Params::new(&sim, &appearance, alpha, state.generation, detect_params, outline_params);
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
    if outlines {
        let field = &state.buffers.as_ref().unwrap().field;
        context.command_encoder().clear_buffer(field, 0, None);
        let mut pass = context
            .command_encoder()
            .begin_compute_pass(&ComputePassDescriptor::default());
        pass.set_pipeline(outline_splat_pipeline);
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
        if outlines {
            pass.set_pipeline(outline_pipeline);
            pass.draw(0..3, 0..1);
        }
    }
    state.report_steps += steps;
    let mut report = status.0.lock().unwrap();
    report.generation = state.generation;
    report.message.clear();
    let elapsed = state.report_at.elapsed().as_secs_f32();
    if elapsed >= 1.0 {
        report.steps_per_second = state.report_steps as f32 / elapsed;
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
