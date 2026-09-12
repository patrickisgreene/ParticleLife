//! Real GPU regression for cluster detection. Drives the production entry points
//! against a hand-placed world, including one body straddling the world seam --
//! the case a naive position mean gets catastrophically wrong -- and one mixed
//! body whose minority fails the per-type promotion minimum.
//!
//! `generations_advance_once_at_birth` checks the evolution wiring behind the
//! outline colour: a child's generation is its parent's plus one, set exactly
//! once, and only for born (lineaged) creatures.
use wgpu::util::DeviceExt;

const WORLD: f32 = 120.0;
const GRID: u32 = 3;
const BOND_RADIUS: f32 = 6.0;
const MAX_CREATURES: usize = 4096;
/// Words per registry record: twelve-u32 base, sixteen per-type counts, three
/// frame accumulators, lineage, mutation flag, pad, and the measured frame vec2.
const CREATURE_U32: usize = 36;

/// Particles of one type in a line, spacing 2, so each is bonded to the ones
/// within three places of it and the cluster is connected transitively.
fn line(x0: f32, y: f32, n: usize, wrap: bool) -> Vec<(f32, f32)> {
    (0..n)
        .map(|i| {
            let x = x0 + i as f32 * 2.0;
            ((if wrap { x.rem_euclid(WORLD) } else { x }), y)
        })
        .collect()
}

#[test]
#[ignore = "requires a Vulkan GPU; run cargo test -- --ignored"]
fn clusters_resolve_to_stable_ids_and_wrap_at_the_seam() {
    pollster::block_on(async {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
        let adapter = instance.request_adapter(&Default::default()).await.unwrap();
        // The force pass alone uses the full baseline allowance of 8 storage
        // buffers per stage, so the tests must ask for what the app asks for:
        // Bevy requests the adapter's limits rather than the defaults.
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                required_limits: adapter.limits(),
                ..Default::default()
            })
            .await
            .unwrap();
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("production simulation"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../assets/shaders/simulation.wgsl").into(),
            ),
        });

        // Compact body away from the seam, plus one centred on x = 0 whose naive
        // mean would land near the middle of the world. The seam body mixes two
        // kinds -- three of type 0 and two of type 1 -- so its total size clears
        // the threshold while its minority does not.
        let mut points = line(18.0, 20.0, 5, false);
        points.extend(line(116.0, 60.0, 5, true));
        let n = points.len() as u32;

        let mut params = [0u32; 56];
        params[..4].copy_from_slice(&[n, 2, GRID, 42]);
        params[4] = 0.001f32.to_bits(); // dt: keep the world essentially static
        params[5] = 10.0f32.to_bits(); // interaction radius
        params[6] = 0.0f32.to_bits(); // no force, so positions hold still
        params[8] = WORLD.to_bits();
        params[9] = 0.2f32.to_bits();
        params[10] = 180.0f32.to_bits();
        params[20] = 1; // exact neighbor sampling
        params[44] = BOND_RADIUS.to_bits();
        params[45] = 3.0f32.to_bits(); // per-type minimum
        params[46] = 2.0f32.to_bits(); // promote steps
        params[47] = 1.0f32.to_bits(); // detection enabled
        params[52] = 0.0f32.to_bits(); // evolution disabled; isotropic rules

        let mut particles = vec![0u32; points.len() * 8];
        for (i, (x, y)) in points.iter().enumerate() {
            particles[i * 8] = x.to_bits();
            particles[i * 8 + 1] = y.to_bits();
            // Both kinds bond to everything; the seam body is three type 0 and
            // two type 1, so the per-type minimum of 3 rejects its minority.
            particles[i * 8 + 4] = if i >= 5 && i % 2 == 1 { 1 } else { 0 };
        }
        // Two types with every interaction nonzero, sized for the two colour
        // slots plus the full rule cache the force pass stages.
        let types = vec![1.0f32; 8 + 16];

        let uniform = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: bytemuck::cast_slice(&params),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        let store = |data: &[u32]| {
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: None,
                contents: bytemuck::cast_slice(data),
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_SRC
                    | wgpu::BufferUsages::COPY_DST,
            })
        };
        let source = store(&particles);
        let destination = store(&vec![0u32; particles.len()]);
        let storages = [
            &source,
            &destination,
            &store(&vec![0u32; (GRID * GRID) as usize * 4]),
            &store(&vec![0u32; points.len() * 4]), // neighbors
            &store(&vec![0u32; 256]),              // blocks
            &store(&types.iter().map(|v| v.to_bits()).collect::<Vec<u32>>()),
            &store(&vec![0u32; 16]), // chemicals, unused with trails off
            &store(&vec![0u32; points.len() * 8]), // sorted
            &store(&vec![0u32; MAX_CREATURES * CREATURE_U32]), // creature registry
            &store(&vec![0u32; MAX_CREATURES * 16 * 2]), // creature genomes
        ];

        // One explicit layout shared by every kernel, mirroring gpu.rs, so a single
        // bind group drives the whole pipeline.
        let entries: Vec<_> = (0..=10u32)
            .map(|i| wgpu::BindGroupLayoutEntry {
                binding: i,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: if i == 0 {
                        wgpu::BufferBindingType::Uniform
                    } else {
                        wgpu::BufferBindingType::Storage { read_only: i == 6 }
                    },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            })
            .collect();
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: None,
            entries: &entries,
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let mut bindings = vec![wgpu::BindGroupEntry {
            binding: 0,
            resource: uniform.as_entire_binding(),
        }];
        bindings.extend(storages.iter().enumerate().map(|(i, b)| wgpu::BindGroupEntry {
            binding: i as u32 + 1,
            resource: b.as_entire_binding(),
        }));
        let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &layout,
            entries: &bindings,
        });
        let stage = |entry: &str| {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(entry),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: Some(entry),
                compilation_options: Default::default(),
                cache: None,
            })
        };
        let creature_groups = MAX_CREATURES as u32 / 256;
        let step: Vec<(wgpu::ComputePipeline, u32)> = [
            ("clear", 1),
            ("count", 1),
            ("prefix", 1),
            ("prefix_blocks", 1),
            ("scatter", 1),
            ("clear_creatures", creature_groups),
            ("bond", 1),
            ("resolve", creature_groups),
            ("resolve", creature_groups),
            ("resolve", creature_groups),
            ("resolve", creature_groups),
            ("update", 1),
            ("creature_stats", 1),
            ("promote", creature_groups),
        ]
        .into_iter()
        .map(|(entry, groups)| (stage(entry), groups))
        .collect();

        // Ids spread one bond-hop per step, so give the labels room to converge and
        // then to satisfy the promotion threshold on top.
        let mut encoder = device.create_command_encoder(&Default::default());
        for _ in 0..12 {
            for (pipeline, groups) in &step {
                let mut pass = encoder.begin_compute_pass(&Default::default());
                pass.set_pipeline(pipeline);
                pass.set_bind_group(0, &group, &[]);
                pass.dispatch_workgroups(*groups, 1, 1);
            }
            // Feed the result back in as the next step's input, which keeps one
            // bind group valid instead of ping-ponging two.
            encoder.copy_buffer_to_buffer(
                &destination,
                0,
                &source,
                0,
                particles.len() as u64 * 4,
            );
        }

        let read = |encoder: &mut wgpu::CommandEncoder, from: &wgpu::Buffer, size: u64| {
            let buffer = device.create_buffer(&wgpu::BufferDescriptor {
                label: None,
                size,
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                mapped_at_creation: false,
            });
            encoder.copy_buffer_to_buffer(from, 0, &buffer, 0, size);
            buffer
        };
        let particles_out = read(&mut encoder, &destination, particles.len() as u64 * 4);
        let creatures_out = read(
            &mut encoder,
            storages[8],
            (MAX_CREATURES * CREATURE_U32) as u64 * 4,
        );
        queue.submit([encoder.finish()]);
        let fetch = |buffer: &wgpu::Buffer| {
            let (tx, rx) = std::sync::mpsc::channel();
            buffer
                .slice(..)
                .map_async(wgpu::MapMode::Read, move |r| tx.send(r).unwrap());
            device
                .poll(wgpu::PollType::Wait {
                    submission_index: None,
                    timeout: Some(std::time::Duration::from_secs(10)),
                })
                .unwrap();
            rx.recv().unwrap().unwrap();
            bytemuck::cast_slice::<u8, u32>(&buffer.slice(..).get_mapped_range()).to_vec()
        };
        let out = fetch(&particles_out);
        let registry = fetch(&creatures_out);

        // Every particle joined a body, and the two bodies stayed distinct.
        let ids: Vec<u32> = (0..points.len()).map(|i| out[i * 8 + 6]).collect();
        assert!(ids.iter().all(|c| *c != 0), "every particle joined: {ids:?}");
        let mut distinct: Vec<u32> = ids.clone();
        distinct.sort_unstable();
        distinct.dedup();
        assert_eq!(distinct.len(), 2, "two bodies, got ids {ids:?}");

        // Group particles by id through their recorded position.
        for id in &distinct {
            let members: Vec<usize> = (0..points.len()).filter(|i| ids[*i] == *id).collect();
            assert_eq!(members.len(), 5, "each body keeps all five particles");
            let c = *id as usize * CREATURE_U32;
            assert_eq!(registry[c], 5, "registry count matches membership");
            let centroid = (
                f32::from_bits(registry[c + 6]),
                f32::from_bits(registry[c + 7]),
            );
            let y = f32::from_bits(out[members[0] * 8 + 1]);
            assert!(
                (centroid.1 - y).abs() < 1.0,
                "centroid y {} should sit on the body at {y}",
                centroid.1
            );
            if y > 40.0 {
                // The seam body is three type 0 and two type 1: its total of five
                // clears the per-type minimum of three, but the type-1 minority
                // does not, so it must not be promoted -- the outline stays off
                // even though the cluster is real.
                assert_eq!(
                    registry[c + 10],
                    0,
                    "mixed body fails the per-type minimum and must not track"
                );
                let wrapped = centroid.0.min(WORLD - centroid.0);
                assert!(
                    wrapped < 3.0,
                    "seam centroid {} collapsed to the naive mean",
                    centroid.0
                );
            } else {
                // The single-type body clears the per-type minimum on its only
                // kind, so it persists long enough to be tracked.
                assert_eq!(
                    registry[c + 10],
                    1,
                    "single-type body clears the per-type minimum and tracks"
                );
                assert!(
                    (centroid.0 - 22.0).abs() < 2.0,
                    "compact body centroid {} should sit at its middle",
                    centroid.0
                );
            }
        }
    });
}

#[test]
#[ignore = "requires a Vulkan GPU; run cargo test -- --ignored"]
fn generations_advance_once_at_birth() {
    pollster::block_on(async {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
        let adapter = instance.request_adapter(&Default::default()).await.unwrap();
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                required_limits: adapter.limits(),
                ..Default::default()
            })
            .await
            .unwrap();
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("production simulation"),
            source: wgpu::ShaderSource::Wgsl(include_str!(
                "../assets/shaders/simulation.wgsl"
            ).into()),
        });

        // One tracked body with a forged line to itself and a pre-seeded
        // generation, plus a control body with no line: the lineaged one must
        // mutate its genomes exactly once and step its generation by one, the
        // control must stay untouched.
        let lineaged = 1usize;
        let control = 2usize;
        let mut params = [0u32; 56];
        params[..4].copy_from_slice(&[0, 2, GRID, 42]);
        params[46] = 1.0f32.to_bits(); // promote steps: promote immediately
        params[47] = 1.0f32.to_bits(); // detection enabled
        params[52] = 1.0f32.to_bits(); // evolution enabled
        params[53] = 0.12f32.to_bits(); // mutation rate

        let mut registry = vec![0u32; MAX_CREATURES * CREATURE_U32];
        for c in [lineaged, control] {
            let w = c * CREATURE_U32;
            registry[w] = 5; // count: viable this step
            registry[w + 10] = 1; // flags: tracked
        }
        registry[lineaged * CREATURE_U32 + 31] = lineaged as u32; // lineage: self
        registry[lineaged * CREATURE_U32 + 32] = 0; // mutated: not yet
        registry[lineaged * CREATURE_U32 + 33] = 5; // generation: pre-seeded
        registry[control * CREATURE_U32 + 31] = 0; // lineage: none

        let uniform = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: bytemuck::cast_slice(&params),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        let store = |data: &[u32]| {
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: None,
                contents: bytemuck::cast_slice(data),
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_SRC
                    | wgpu::BufferUsages::COPY_DST,
            })
        };
        // Genomes start isotropic (ratio 1, orientation 0) for every slot, as
        // `reset_evolution` would leave them.
        let genomes_data: Vec<u32> = (0..MAX_CREATURES * 16)
            .flat_map(|_| [1.0f32.to_bits(), 0u32])
            .collect();
        let genomes = store(&genomes_data);
        let storages = [
            &store(&[0u32; 8]), // source, unused by promote_frame
            &store(&[0u32; 8]), // destination
            &store(&[0u32; 8]), // cells
            &store(&[0u32; 8]), // neighbors
            &store(&[0u32; 8]), // blocks
            &store(&[0u32; 32]), // types
            &store(&[0u32; 8]), // chemicals
            &store(&[0u32; 8]), // sorted
            &store(&registry), // creature registry
            &genomes,          // creature genomes
        ];

        let entries: Vec<_> = (0..=10u32)
            .map(|i| wgpu::BindGroupLayoutEntry {
                binding: i,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: if i == 0 {
                        wgpu::BufferBindingType::Uniform
                    } else {
                        wgpu::BufferBindingType::Storage { read_only: i == 6 }
                    },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            })
            .collect();
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: None,
            entries: &entries,
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let mut bindings = vec![wgpu::BindGroupEntry {
            binding: 0,
            resource: uniform.as_entire_binding(),
        }];
        bindings.extend(storages.iter().enumerate().map(|(i, b)| wgpu::BindGroupEntry {
            binding: i as u32 + 1,
            resource: b.as_entire_binding(),
        }));
        let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &layout,
            entries: &bindings,
        });
        let promote = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("promote_frame"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("promote_frame"),
            compilation_options: Default::default(),
            cache: None,
        });

        let creature_groups = MAX_CREATURES as u32 / 256;
        let mut encoder = device.create_command_encoder(&Default::default());
        {
            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_pipeline(&promote);
            pass.set_bind_group(0, &group, &[]);
            pass.dispatch_workgroups(creature_groups, 1, 1);
        }
        let read = |encoder: &mut wgpu::CommandEncoder, from: &wgpu::Buffer, size: u64| {
            let buffer = device.create_buffer(&wgpu::BufferDescriptor {
                label: None,
                size,
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                mapped_at_creation: false,
            });
            encoder.copy_buffer_to_buffer(from, 0, &buffer, 0, size);
            buffer
        };
        let registry_out =
            read(&mut encoder, storages[8], (MAX_CREATURES * CREATURE_U32) as u64 * 4);
        let genomes_out = read(&mut encoder, &genomes, (MAX_CREATURES * 16 * 2) as u64 * 4);
        queue.submit([encoder.finish()]);
        let fetch = |buffer: &wgpu::Buffer| {
            let (tx, rx) = std::sync::mpsc::channel();
            buffer
                .slice(..)
                .map_async(wgpu::MapMode::Read, move |r| tx.send(r).unwrap());
            device
                .poll(wgpu::PollType::Wait {
                    submission_index: None,
                    timeout: Some(std::time::Duration::from_secs(10)),
                })
                .unwrap();
            rx.recv().unwrap().unwrap();
            bytemuck::cast_slice::<u8, u32>(&buffer.slice(..).get_mapped_range()).to_vec()
        };
        let reg = fetch(&registry_out);
        let genomes = fetch(&genomes_out);

        let l = lineaged * CREATURE_U32;
        let c = control * CREATURE_U32;
        assert_eq!(reg[l + 32], 1, "born creature mutates exactly once");
        assert_eq!(reg[l + 33], 6, "generation advances by one from the parent's");
        // The pre-seeded isotropic kernels (1.0, 0.0) must now differ in at least
        // one of the two recognized kinds for the lineaged creature, while the
        // control's kernels stay untouched.
        let ratio = f32::from_bits(genomes[lineaged * 16 * 2]);
        let orient = f32::from_bits(genomes[lineaged * 16 * 2 + 1]);
        assert!(
            (ratio - 1.0).abs() > 1e-6 || orient.abs() > 1e-6,
            "lineaged creature's genome was mutated ({ratio}, {orient})"
        );
        assert_eq!(reg[c + 32], 0, "unborn creature never mutates");
        assert_eq!(reg[c + 33], 0, "unborn creature stays at generation 0");
        assert_eq!(genomes[control * 16 * 2], 1.0f32.to_bits(), "control ratio unchanged");
        assert_eq!(genomes[control * 16 * 2 + 1], 0, "control orientation unchanged");
    });
}
