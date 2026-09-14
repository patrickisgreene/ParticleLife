//! Real GPU regression for the production field entry points. Kept opt-in so
//! ordinary cargo test also works on machines without a GPU/graphics session.
use wgpu::util::DeviceExt;

#[test]
#[ignore = "requires a Vulkan GPU; run cargo test -- --ignored"]
fn chemical_field_wraps_decays_deposits_once_and_clears() {
    pollster::block_on(async {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::VULKAN,
            ..wgpu::InstanceDescriptor::new_without_display_handle()
        });
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions::default())
            .await
            .expect("Vulkan adapter");
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor::default())
            .await
            .unwrap();
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("production simulation"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../assets/shaders/simulation.wgsl").into(),
            ),
        });
        let mut params = [0u32; 56];
        params[4] = 0.25f32.to_bits(); // dt
        params[25] = 2.0f32.to_bits(); // deposit per second
        params[26] = 2.0f32.to_bits(); // diffusion rate
        params[27] = 1.0f32.to_bits(); // half-life
        params[31] = 4.0f32.to_bits(); // tiny toroidal grid
        let uniform = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: bytemuck::cast_slice(&params),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        let mut initial = [0u32; 256];
        initial[0] = 100.0f32.to_bits();
        initial[5 * 16 + 2] = 3; // three deposits at (1,1)
        initial[4] = 100.0f32.to_bits(); // existing red scent
        initial[5 * 16 + 12] = 255; // one red deposit
        initial[5 * 16 + 14] = 510; // two blue deposits
        let field = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: bytemuck::cast_slice(&initial),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        });
        let stages: Vec<_> = ["diffuse_trails", "commit_trails", "clear_trails"]
            .into_iter()
            .map(|entry| {
                let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: None,
                    layout: None,
                    module: &shader,
                    entry_point: Some(entry),
                    compilation_options: Default::default(),
                    cache: None,
                });
                let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: None,
                    layout: &pipeline.get_bind_group_layout(0),
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: uniform.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 7,
                            resource: field.as_entire_binding(),
                        },
                    ],
                });
                (pipeline, group)
            })
            .collect();
        let run = |passes: &[usize]| -> Vec<u32> {
            let readback = device.create_buffer(&wgpu::BufferDescriptor {
                label: None,
                size: 1024,
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                mapped_at_creation: false,
            });
            let mut encoder = device.create_command_encoder(&Default::default());
            for &stage in passes {
                let mut pass = encoder.begin_compute_pass(&Default::default());
                pass.set_pipeline(&stages[stage].0);
                pass.set_bind_group(0, &stages[stage].1, &[]);
                pass.dispatch_workgroups(1, 1, 1);
            }
            encoder.copy_buffer_to_buffer(&field, 0, &readback, 0, 1024);
            queue.submit([encoder.finish()]);
            let (tx, rx) = std::sync::mpsc::channel();
            readback
                .slice(..)
                .map_async(wgpu::MapMode::Read, move |result| {
                    tx.send(result).unwrap();
                });
            device
                .poll(wgpu::PollType::Wait {
                    submission_index: None,
                    timeout: Some(std::time::Duration::from_secs(10)),
                })
                .unwrap();
            rx.recv().unwrap().unwrap();
            let data = readback.slice(..).get_mapped_range();
            bytemuck::cast_slice(&data).to_vec()
        };
        let first = run(&[0, 1]);
        let density = |data: &[u32], i: usize| f32::from_bits(data[i * 16]);
        let mass = |data: &[u32]| (0..16).map(|i| density(data, i)).sum::<f32>();
        let decay = 2.0f32.powf(-0.25);
        assert!(
            (mass(&first) - 101.5 * decay).abs() < 0.001,
            "mass conserved except deposits and decay"
        );
        assert!(
            density(&first, 3) > 0.0 && density(&first, 12) > 0.0,
            "diffusion crosses both seams"
        );
        assert!((density(&first, 1) - density(&first, 3)).abs() < 0.00001);
        assert!((density(&first, 5) - 1.5 * decay).abs() < 0.00001);
        assert!((0..16).all(|i| first[i * 16 + 2] == 0), "deposits consumed");
        let channel = |data: &[u32], i: usize, c: usize| f32::from_bits(data[i * 16 + 4 + c]);
        assert!((channel(&first, 5, 0) / density(&first, 5) - 1.0 / 3.0).abs() < 0.00001);
        assert!((channel(&first, 5, 2) / density(&first, 5) - 2.0 / 3.0).abs() < 0.00001);
        assert!(
            (channel(&first, 3, 0) - density(&first, 3)).abs() < 0.00001,
            "hue wraps with density"
        );
        assert!(
            (0..16).all(|i| (12..15).all(|c| first[i * 16 + c] == 0)),
            "color deposits consumed"
        );
        let second = run(&[0, 1]);
        assert!(
            (mass(&second) - mass(&first) * decay).abs() < 0.001,
            "deposits not replayed"
        );
        assert!((0..16).all(|i| density(&second, i).is_finite() && density(&second, i) >= 0.0));
        assert!(
            run(&[2]).iter().all(|x| *x == 0),
            "clear resets every field component"
        );
    });
}

#[test]
#[ignore = "requires a Vulkan GPU; run cargo test -- --ignored"]
fn particle_spacing_crowding_and_cycles() {
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
            label: None,
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../assets/shaders/simulation.wgsl").into(),
            ),
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: None,
            layout: None,
            module: &shader,
            entry_point: Some("update"),
            compilation_options: Default::default(),
            cache: None,
        });
        // All test particles lie in spatial cell (1,1), so the cell-sorted slot
        // order the shader reads is identical to the input order.
        let run = |xs: &[f32],
                   attraction: f32,
                   preferred: f32,
                   density_target: f32,
                   mode: u32,
                   age: f32,
                   type_count: u32,
                   next_fraction: f32|
         -> Vec<u32> {
            let n = xs.len() as u32;
            let mut params = [0u32; 56];
            params[..4].copy_from_slice(&[n, type_count, 3, 42]);
            params[4] = 0.02f32.to_bits();
            params[5] = 40.0f32.to_bits();
            params[6] = 1.0f32.to_bits();
            params[8] = 100.0f32.to_bits();
            params[9] = 0.2f32.to_bits();
            params[10] = 180.0f32.to_bits();
            params[20] = 1;
            params[31] = 1.0f32.to_bits();
            params[32] = if preferred > 0.0 { 1.0f32.to_bits() } else { 0 };
            params[33] = if density_target > 0.0 {
                1.0f32.to_bits()
            } else {
                0
            };
            params[34] = density_target.to_bits();
            params[36] = (mode as f32).to_bits();
            params[37] = 1.0f32.to_bits();
            params[38] = next_fraction.to_bits();
            params[39] = 1.0f32.to_bits();
            params[40] = 256.0f32.to_bits(); // sample budget; exact mode ignores it
            let mut particles = vec![0u32; n as usize * 8];
            for (i, x) in xs.iter().enumerate() {
                particles[i * 8] = x.to_bits();
                particles[i * 8 + 1] = 50.0f32.to_bits();
                particles[i * 8 + 4] = i as u32 % type_count;
                particles[i * 8 + 5] = age.to_bits();
            }
            let mut cells = vec![0u32; 9 * 4];
            cells[4 * 4] = n;
            // Packed neighbor records: position, two f16 velocities, kind.
            // Every test particle starts at rest, so the velocity word is zero.
            let neighbors: Vec<u32> = (0..n as usize)
                .flat_map(|i| {
                    [
                        particles[i * 8],
                        particles[i * 8 + 1],
                        0,
                        particles[i * 8 + 4],
                    ]
                })
                .collect();
            let matrix = (type_count * type_count) as usize;
            let mut types = vec![0u32; type_count as usize * 4 + matrix * 4];
            types[type_count as usize * 4..type_count as usize * 4 + matrix]
                .fill(attraction.to_bits());
            types[type_count as usize * 4 + matrix * 3..].fill(preferred.to_bits());
            let contents = [
                params.to_vec(),
                particles.clone(),
                vec![0; particles.len()],
                cells,
                neighbors,
                vec![0],
                types,
                vec![0; 16],
                particles.clone(),
            ];
            let buffers: Vec<_> = contents
                .iter()
                .enumerate()
                .map(|(i, data)| {
                    device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                        label: None,
                        contents: bytemuck::cast_slice(data),
                        usage: if i == 0 {
                            wgpu::BufferUsages::UNIFORM
                        } else {
                            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC
                        },
                    })
                })
                .collect();
            let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &pipeline.get_bind_group_layout(0),
                // `update` reads the cell-sorted copy at binding 8 rather than
                // `source`, so binding 1 is absent from the reflected layout.
                entries: &buffers
                    .iter()
                    .enumerate()
                    .filter(|(i, _)| *i != 1)
                    .map(|(i, b)| wgpu::BindGroupEntry {
                        binding: i as u32,
                        resource: b.as_entire_binding(),
                    })
                    .collect::<Vec<_>>(),
            });
            let size = n as u64 * 32;
            let readback = device.create_buffer(&wgpu::BufferDescriptor {
                label: None,
                size,
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                mapped_at_creation: false,
            });
            let mut encoder = device.create_command_encoder(&Default::default());
            {
                let mut pass = encoder.begin_compute_pass(&Default::default());
                pass.set_pipeline(&pipeline);
                pass.set_bind_group(0, &group, &[]);
                pass.dispatch_workgroups(1, 1, 1);
            }
            encoder.copy_buffer_to_buffer(&buffers[2], 0, &readback, 0, size);
            queue.submit([encoder.finish()]);
            let (tx, rx) = std::sync::mpsc::channel();
            readback
                .slice(..)
                .map_async(wgpu::MapMode::Read, move |result| {
                    tx.send(result).unwrap();
                });
            device
                .poll(wgpu::PollType::Wait {
                    submission_index: None,
                    timeout: Some(std::time::Duration::from_secs(10)),
                })
                .unwrap();
            rx.recv().unwrap().unwrap();
            bytemuck::cast_slice(&readback.slice(..).get_mapped_range()).to_vec()
        };
        let velocity = |data: &[u32]| f32::from_bits(data[2]);
        assert!(
            velocity(&run(&[40., 60.], 1., 0., 0., 0, 0., 2, 0.5)) > 0.,
            "original attraction"
        );
        assert!(
            velocity(&run(&[40., 60.], 1., 0.75, 0., 0, 0., 2, 0.5)) < 0.,
            "too close repels"
        );
        assert!(
            velocity(&run(&[40., 60.], -1., 0.25, 0., 0, 0., 2, 0.5)) > 0.,
            "too far attracts even with negative original rule"
        );
        assert!(
            velocity(&run(&[40., 60.], 1., 0.5, 0., 0, 0., 2, 0.5)).abs() < 0.000001,
            "comfortable spacing is equilibrium"
        );
        assert!(
            velocity(&run(&[40., 50., 60.], 0., 0., 1., 0, 0., 2, 0.5)) < 0.,
            "crowding repels"
        );
        assert!(
            velocity(&run(&[40., 50., 60.], 0., 0., 4., 0, 0., 2, 0.5)) > 0.,
            "sparse neighborhoods attract"
        );
        let timer = run(&[40., 60.], 0., 0., 0., 1, 0.99, 2, 0.5);
        assert_eq!(
            (timer[4], timer[12]),
            (1, 0),
            "cycle wraps and reads the previous generation"
        );
        assert_eq!(f32::from_bits(timer[5]), 0., "age resets on transition");
        assert_eq!(
            run(&[40., 60.], 0., 0., 0., 2, 0.99, 2, 0.5)[4],
            1,
            "successor neighbors trigger"
        );
        assert_eq!(
            run(&[40., 60.], 0., 0., 0., 2, 0.1, 2, 0.5)[4],
            0,
            "neighbor cooldown"
        );
        assert_eq!(
            run(&[40., 50., 60.], 0., 0., 0., 2, 0.99, 2, 0.75)[4],
            0,
            "insufficient successor fraction"
        );
        assert_eq!(
            run(&[40., 60.], 0., 0., 0., 3, 0.49, 2, 0.5)[4],
            1,
            "either allows an early neighbor trigger"
        );
        assert_eq!(
            run(&[40., 60.], 0., 0., 0., 1, 0.99, 1, 0.5)[4],
            0,
            "one type remains stable"
        );
        assert_eq!(
            run(&[40., 60.], 0., 0., 0., 0, 0.99, 2, 0.5)[4],
            0,
            "off preserves type"
        );
    });
}

/// A tight same-type knot ignites: the shell latches an outward direction and
/// burns, while the symmetric core, which has no escape direction, is reseeded
/// somewhere else in the world.
#[test]
#[ignore = "requires a Vulkan GPU; run cargo test -- --ignored"]
fn novae_eject_dense_clumps_and_reseed_the_core() {
    pollster::block_on(async {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::VULKAN,
            ..wgpu::InstanceDescriptor::new_without_display_handle()
        });
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions::default())
            .await
            .expect("Vulkan adapter");
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor::default())
            .await
            .unwrap();
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("production simulation"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../assets/shaders/simulation.wgsl").into(),
            ),
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: None,
            layout: None,
            module: &shader,
            entry_point: Some("update"),
            compilation_options: Default::default(),
            cache: None,
        });
        // One type, no attraction and no damping, so the only forces in play are
        // the unconditional core repulsion and the nova impulse under test.
        let run = |xs: &[f32], threshold: f32| -> Vec<u32> {
            let n = xs.len() as u32;
            let mut params = [0u32; 56];
            params[..4].copy_from_slice(&[n, 1, 3, 42]);
            params[4] = 0.02f32.to_bits(); // dt
            params[5] = 40.0f32.to_bits(); // interaction radius
            params[6] = 1.0f32.to_bits(); // strength
            params[8] = 100.0f32.to_bits(); // world size
            params[9] = 0.2f32.to_bits(); // core fraction
            params[10] = 180.0f32.to_bits(); // speed limit
            params[20] = 1; // exact: count every candidate, no sampling noise
            params[31] = 1.0f32.to_bits();
            params[40] = 256.0f32.to_bits();
            params[52] = threshold.to_bits(); // 0 disables ignition
            params[53] = 0.35f32.to_bits(); // ignition radius fraction
            params[54] = 1800.0f32.to_bits(); // blast strength
            params[55] = 0.3f32.to_bits(); // blast duration
            // All in the middle cell of the 3x3 grid, so `sorted` order is input order.
            let mut particles = vec![0u32; n as usize * 8];
            for (i, x) in xs.iter().enumerate() {
                particles[i * 8] = x.to_bits();
                particles[i * 8 + 1] = 50.0f32.to_bits();
            }
            let mut cells = vec![0u32; 9 * 4];
            cells[4 * 4] = n;
            let neighbors: Vec<u32> = (0..n as usize)
                .flat_map(|i| [particles[i * 8], particles[i * 8 + 1], 0, 0])
                .collect();
            let contents = [
                params.to_vec(),
                particles.clone(),
                vec![0; particles.len()],
                cells,
                neighbors,
                vec![0],
                vec![0u32; 4 + 4],
                vec![0; 16],
                particles.clone(),
            ];
            let buffers: Vec<_> = contents
                .iter()
                .enumerate()
                .map(|(i, data)| {
                    device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                        label: None,
                        contents: bytemuck::cast_slice(data),
                        usage: if i == 0 {
                            wgpu::BufferUsages::UNIFORM
                        } else {
                            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC
                        },
                    })
                })
                .collect();
            let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &pipeline.get_bind_group_layout(0),
                entries: &buffers
                    .iter()
                    .enumerate()
                    .filter(|(i, _)| *i != 1)
                    .map(|(i, b)| wgpu::BindGroupEntry {
                        binding: i as u32,
                        resource: b.as_entire_binding(),
                    })
                    .collect::<Vec<_>>(),
            });
            let size = n as u64 * 32;
            let readback = device.create_buffer(&wgpu::BufferDescriptor {
                label: None,
                size,
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                mapped_at_creation: false,
            });
            let mut encoder = device.create_command_encoder(&Default::default());
            {
                let mut pass = encoder.begin_compute_pass(&Default::default());
                pass.set_pipeline(&pipeline);
                pass.set_bind_group(0, &group, &[]);
                pass.dispatch_workgroups(1, 1, 1);
            }
            encoder.copy_buffer_to_buffer(&buffers[2], 0, &readback, 0, size);
            queue.submit([encoder.finish()]);
            let (tx, rx) = std::sync::mpsc::channel();
            readback
                .slice(..)
                .map_async(wgpu::MapMode::Read, move |result| {
                    tx.send(result).unwrap();
                });
            device
                .poll(wgpu::PollType::Wait {
                    submission_index: None,
                    timeout: Some(std::time::Duration::from_secs(10)),
                })
                .unwrap();
            rx.recv().unwrap().unwrap();
            bytemuck::cast_slice(&readback.slice(..).get_mapped_range()).to_vec()
        };
        let at = |data: &[u32], i: usize| {
            (
                f32::from_bits(data[i * 8]),     // position x
                f32::from_bits(data[i * 8 + 1]), // position y
                f32::from_bits(data[i * 8 + 2]), // velocity x
                f32::from_bits(data[i * 8 + 6]), // remaining burn time
            )
        };
        // Five particles one unit apart: every one has four same-type neighbours
        // well inside the 14-unit ignition radius.
        let clump = [48., 49., 50., 51., 52.];
        let quiet = run(&clump, 100.); // threshold far out of reach
        let burst = run(&clump, 4.);
        let off = run(&clump, 0.); // rule disabled entirely

        let (_, _, quiet_vx, quiet_burn) = at(&quiet, 0);
        let (_, _, burst_vx, burst_burn) = at(&burst, 0);
        assert_eq!(quiet_burn, 0., "below threshold nothing ignites");
        assert!(burst_burn > 0., "the shell is burning");
        assert!(burst_burn < 0.3, "the burn timer counts down");
        assert!(burst_vx < 0., "the left edge is thrown further left");
        assert!(
            burst_vx < quiet_vx * 2.,
            "the blast dominates core repulsion: {burst_vx} vs {quiet_vx}"
        );

        // The middle particle's neighbours cancel exactly, so it is the core.
        let (core_x, core_y, core_vx, core_burn) = at(&burst, 2);
        assert_eq!(core_burn, 0., "a reseeded particle does not burn");
        assert_eq!(core_vx, 0., "a reseeded particle starts at rest");
        assert!(
            (0.0..100.).contains(&core_x) && (0.0..100.).contains(&core_y),
            "reseeded inside the world: ({core_x}, {core_y})"
        );
        assert!(
            (core_x - 50.).hypot(core_y - 50.) > 5.,
            "reseeded away from the clump: ({core_x}, {core_y})"
        );
        assert_eq!(at(&quiet, 2).0, 50., "below threshold the core stays put");

        for i in 0..clump.len() {
            assert_eq!(at(&off, i).3, 0., "disabled leaves the burn timer clear");
            assert_eq!(
                at(&off, i).2,
                at(&quiet, i).2,
                "disabled matches a world that never reaches the threshold"
            );
        }
    });
}
