use bevy::{prelude::*, render::extract_resource::ExtractResource};
use std::sync::{Arc, Mutex};

pub const WORLD: f32 = 2048.0;
pub const TRAIL_SIDE: u32 = 512;
pub const TRAIL_BYTES: u64 = TRAIL_SIDE as u64 * TRAIL_SIDE as u64 * 64;
pub const DT: f32 = 1.0 / 60.0;
/// Fraction of the interaction radius below which every pair repels, regardless
/// of its rule. `force()` in the simulation shader crosses zero exactly here, so
/// it is also the rest separation of a bonded pair. Shared with `creatures.rs`
/// so inferred geometry cannot drift from the shader.
pub const CORE_FRACTION: f32 = 0.2;
/// Creature id 0 means "unassigned", so ids run 1..MAX_CREATURES.
pub const MAX_CREATURES: u32 = 4096;
/// Side of the world-space occupancy field the outline contours. A texel is four
/// world units at the default radius, against a rest spacing near five, so the
/// Gaussian stamp rather than the resolution is what smooths the body.
pub const FIELD_SIDE: u32 = 512;
pub const FIELD_BYTES: u64 = FIELD_SIDE as u64 * FIELD_SIDE as u64 * 8;
/// Types whose per-creature particle count the cluster detector tracks. Mirrors
/// the rule-cache capacity in the simulation shader; kinds above this are not
/// enumerated by the per-type minimum check.
pub const CREATURE_TYPES: u32 = 16;
/// Words per `Creature` in the simulation shader beyond the twelve-u32 base and
/// the per-type counts: three fixed-point body-frame accumulators, a lineage id,
/// a mutation flag, pad alignment, and the measured frame stored as a vec2
/// (angle, aspect). Together with the base and the counts that is 36 words.
const CREATURE_TAIL_WORDS: u64 = 8;
/// Bytes per `Creature` in the simulation shader. The tail keeps the record at a
/// multiple of 16 bytes so the array stride stays WebGPU-aligned.
pub const CREATURE_BYTES: u64 = (12 + CREATURE_TYPES as u64 + CREATURE_TAIL_WORDS) * 4;
pub const SPEED_LIMIT: f32 = 180.0;

pub const PALETTES: [(&str, [u32; 8]); 4] = [
    (
        "Aurora",
        [
            0x70e1ce, 0xa8f0b0, 0x80b9ff, 0xb49aff, 0xef9cda, 0xffcc9c, 0xffedaf, 0x77d2ed,
        ],
    ),
    (
        "Ember",
        [
            0xff725e, 0xffaa70, 0xffd493, 0xe897b2, 0xc78bcc, 0x9da4e0, 0xe8be9b, 0xffe8bb,
        ],
    ),
    (
        "Lagoon",
        [
            0x55cbbb, 0x78e5d5, 0x63b9dc, 0x809ce8, 0xa6b9ed, 0xc3dfce, 0xe6dca8, 0x92d6a0,
        ],
    ),
    (
        "Pastel",
        [
            0xf5a9b8, 0xf5c69b, 0xf1e6a5, 0xb6dba6, 0x9edbd5, 0xa9c7ef, 0xc5b1e9, 0xe2b2d6,
        ],
    ),
];

/// Cluster detection: finds bodies the rules hold together and tracks them.
#[derive(Clone)]
pub struct Detection {
    pub enabled: bool,
    /// Steps a cluster must hold its size before it counts as a creature rather
    /// than a momentary crowd. Higher values trade responsiveness for stability.
    pub promote_steps: u32,
    pub outline: f32,
}
impl Default for Detection {
    fn default() -> Self {
        Self {
            enabled: true,
            promote_steps: 30,
            outline: 0.3,
        }
    }
}

/// Per-creature genomes and the selection pressure that reshapes them. Each body
/// carries an anisotropic kernel (one major/minor ratio and orientation per
/// interaction type) layered on the shared matrices; `promote_frame` in the
/// shader measures the body's actual frame and blends it into the encoded one.
/// Reproduction is a tracked cluster splitting into two viable halves, and
/// culling dissolves the least durable body when free particles run short.
#[derive(Clone)]
pub struct Evolution {
    pub enabled: bool,
    /// How strongly a child's per-kind ratios and orientations wander from its
    /// parent's at birth.
    pub mutation: f32,
    /// Fraction of the particle budget that may be free before the weakest
    /// tracked creature is dissolved back into free material.
    pub cull_free_fraction: f32,
    /// How strongly creature genomes warp the shared rules: 0 keeps the rules
    /// exactly as written, 1 applies the fully mutated kernel.
    pub influence: f32,
    /// Bumped to ask the GPU to reset every genome to a fresh isotropic state.
    pub evolution_revision: u64,
}
impl Default for Evolution {
    fn default() -> Self {
        Self {
            enabled: true,
            mutation: 0.12,
            cull_free_fraction: 0.02,
            influence: 1.0,
            evolution_revision: 0,
        }
    }
}

/// A shared, toroidal chemical field. Disabled by default to preserve worlds.
#[derive(Clone)]
pub struct Trails {
    pub enabled: bool,
    pub deposit: f32,
    pub diffusion: f32,
    pub half_life: f32,
    pub response: f32,
    pub sensor_distance: f32,
    pub visibility: f32,
    pub clear_revision: u64,
}
impl Default for Trails {
    fn default() -> Self {
        Self {
            enabled: true,
            deposit: 0.2,
            diffusion: 50.0,
            half_life: 0.5,
            response: 90.0,
            sensor_distance: 32.0,
            visibility: 0.4,
            clear_revision: 0,
        }
    }
}

#[derive(Clone)]
pub struct Behavior {
    pub preferred_enabled: bool,
    pub density_enabled: bool,
    pub density_target: f32,
    pub density_strength: f32,
    /// 0 off, 1 timer, 2 neighbors, 3 either.
    pub cycle_mode: u32,
    pub cycle_seconds: f32,
    pub cycle_fraction: f32,
    pub cycle_min_neighbors: u32,
    /// Neighbors sampled per particle per step. The force estimator's variance
    /// scales as 1/budget regardless of local density, but the visible effect of
    /// that noise shrinks as neighbor counts rise, so dense worlds tolerate far
    /// fewer samples. Lowering this is the cheapest way to buy particle count.
    pub sample_budget: u32,
}
impl Default for Behavior {
    fn default() -> Self {
        Self {
            preferred_enabled: false,
            density_enabled: true,
            density_target: 430.0,
            density_strength: 0.30,
            cycle_mode: 1,
            cycle_seconds: 1.0,
            cycle_fraction: 0.35,
            cycle_min_neighbors: 3,
            sample_budget: 100,
        }
    }
}

/// Every matrix is directional: row is the affected type, column its neighbor.
#[derive(Clone, Copy, Default, PartialEq, Eq)]
pub enum RuleKind {
    #[default]
    Attraction,
    Swirl,
    Alignment,
    Distance,
}
impl RuleKind {
    pub const ALL: [Self; 4] = [
        Self::Attraction,
        Self::Swirl,
        Self::Alignment,
        Self::Distance,
    ];
    pub fn label(self) -> &'static str {
        match self {
            Self::Attraction => "Attraction",
            Self::Swirl => "Swirl",
            Self::Alignment => "Alignment",
            Self::Distance => "Distance",
        }
    }
    pub fn hint(self) -> &'static str {
        match self {
            Self::Attraction => "− repel · + attract",
            Self::Swirl => "− clockwise · + counterclockwise",
            Self::Alignment => "− oppose neighbor velocity · + match neighbor velocity",
            Self::Distance => {
                "Fraction of interaction radius · zero uses the original attraction rule"
            }
        }
    }
    fn salt(self) -> u32 {
        match self {
            Self::Attraction => 0,
            Self::Swirl => 0x91e10da5,
            Self::Alignment => 0xd192ed03,
            Self::Distance => 0x7a2c9b41,
        }
    }
}

#[derive(Clone, Resource, ExtractResource)]
pub struct Simulation {
    pub count: u32,
    pub types: u32,
    pub seed: u32,
    pub epoch: u64,
    pub rules_revision: u64,
    pub rules: Arc<Vec<f32>>,
    pub swirl: Arc<Vec<f32>>,
    pub alignment: Arc<Vec<f32>>,
    pub distance: Arc<Vec<f32>>,
    pub behavior: Behavior,
    pub palette: usize,
    pub paused: bool,
    pub step: u64,
    pub speed: f32,
    pub radius: f32,
    pub strength: f32,
    pub damping: f32,
    pub exact: bool,
    pub clustered: bool,
    pub frame_dt: f32,
    pub trails: Trails,
    pub detection: Detection,
    pub evolution: Evolution,
}
impl Default for Simulation {
    fn default() -> Self {
        Self {
            count: 60_000,
            types: 32,
            seed: 42,
            epoch: 1,
            rules_revision: 1,
            rules: Arc::new(random_rules(32, 64)),
            swirl: Arc::new(vec![0.0; 32 * 32]),
            alignment: Arc::new(vec![0.0; 32 * 32]),
            distance: Arc::new(vec![0.5; 32 * 32]),
            behavior: Behavior::default(),
            palette: 0,
            paused: false,
            step: 0,
            speed: 1.0,
            radius: 24.0,
            strength: 90.0,
            damping: 2.0,
            exact: false,
            clustered: true,
            frame_dt: DT,
            trails: Trails::default(),
            detection: Detection::default(),
            evolution: Evolution::default(),
        }
    }
}
impl Simulation {
    pub fn randomize(&mut self) {
        self.rules = Arc::new(random_rules(self.types, self.seed));
        self.swirl = Arc::new(vec![0.0; self.rules.len()]);
        self.alignment = Arc::new(random_rules(self.types, self.seed));
        self.distance = Arc::new(random_rules(self.types, self.seed));
        self.rules_revision += 1;
    }
    pub fn matrix(&self, kind: RuleKind) -> &Arc<Vec<f32>> {
        match kind {
            RuleKind::Attraction => &self.rules,
            RuleKind::Swirl => &self.swirl,
            RuleKind::Alignment => &self.alignment,
            RuleKind::Distance => &self.distance,
        }
    }
    fn matrix_mut(&mut self, kind: RuleKind) -> &mut Arc<Vec<f32>> {
        match kind {
            RuleKind::Attraction => &mut self.rules,
            RuleKind::Swirl => &mut self.swirl,
            RuleKind::Alignment => &mut self.alignment,
            RuleKind::Distance => &mut self.distance,
        }
    }
    pub fn set_rule(&mut self, kind: RuleKind, index: usize, value: f32) {
        Arc::make_mut(self.matrix_mut(kind))[index] = if kind == RuleKind::Distance {
            value.clamp(0.0, 0.95)
        } else {
            value.clamp(-1.0, 1.0)
        };
        self.rules_revision += 1;
    }
    pub fn randomize_matrix(&mut self, kind: RuleKind) {
        let mut values = random_rules(self.types, self.seed ^ kind.salt());
        if kind == RuleKind::Distance {
            for value in &mut values {
                *value = 0.25 + (*value + 1.0) * 0.275;
            }
        }
        *self.matrix_mut(kind) = Arc::new(values);
        self.rules_revision += 1;
    }
    pub fn clear_matrix(&mut self, kind: RuleKind) {
        Arc::make_mut(self.matrix_mut(kind)).fill(0.0);
        self.rules_revision += 1;
    }
    pub fn grid_side(&self) -> u32 {
        ((WORLD / self.radius).floor() as u32).clamp(3, 256)
    }
}

#[derive(Clone, Resource, ExtractResource)]
pub struct Appearance {
    pub size: f32,
    pub glow: f32,
    pub zoom: f32,
    pub pan: Vec2,
    pub aspect: f32,
}
impl Default for Appearance {
    fn default() -> Self {
        Self {
            size: 0.5,
            glow: 0.25,
            zoom: 1.0,
            pan: Vec2::ZERO,
            aspect: 16.0 / 9.0,
        }
    }
}

impl Appearance {
    /// Cursor and viewport size must use the same units (logical pixels in the UI).
    pub fn zoom_at_cursor(&mut self, cursor: Vec2, viewport: Vec2, scroll: f32) {
        if !viewport.is_finite()
            || viewport.min_element() <= 0.0
            || !cursor.is_finite()
            || !scroll.is_finite()
        {
            return;
        }
        let new_zoom = (self.zoom * (scroll * 0.002).exp()).clamp(0.25, 20.0);
        if new_zoom == self.zoom {
            return;
        }
        let normalized = cursor / viewport * 2.0 - Vec2::ONE;
        let offset = normalized * Vec2::new(self.aspect, -1.0);
        self.pan += offset * (self.zoom.recip() - new_zoom.recip());
        self.zoom = new_zoom;
        self.wrap_pan();
    }

    /// Camera coordinates have period two, matching the shader's [-1, 1] world.
    pub fn wrap_pan(&mut self) {
        self.pan = (self.pan + Vec2::ONE).rem_euclid(Vec2::splat(2.0)) - Vec2::ONE;
    }

    /// Include the particle radius so soft discs crossing a seam remain intact.
    pub fn visible_tiles(&self) -> (IVec2, UVec2) {
        let extent = Vec2::new(self.aspect, 1.0) / self.zoom + Vec2::splat(self.size * 2.0 / WORLD);
        let first = ((self.pan - extent + Vec2::ONE) * 0.5).floor().as_ivec2();
        let last = ((self.pan + extent + Vec2::ONE) * 0.5).floor().as_ivec2();
        (first, (last - first + IVec2::ONE).as_uvec2())
    }
}

#[derive(Default)]
pub struct GpuStatus {
    pub adapter: String,
    pub message: String,
    pub max_storage: u64,
    pub max_buffer: u64,
    pub max_dispatch: u32,
    pub generation: u64,
    pub steps_per_second: f32,
}
#[derive(Clone, Resource, Default, ExtractResource)]
pub struct Status(pub Arc<Mutex<GpuStatus>>);

// A conservative working-set budget, in addition to actual device binding limits.
// This leaves room for render targets, Bevy, the desktop, and in-flight old buffers.
pub fn validate_counts(count: u32, types: u32, status: &GpuStatus) -> Result<(), String> {
    if count == 0 || types == 0 {
        return Err("Particle and type counts must be positive.".into());
    }
    let particles = u64::from(count) * 32;
    let rules = (u64::from(types) * u64::from(types))
        .checked_mul(RuleKind::ALL.len() as u64)
        .and_then(|size| size.checked_add(u64::from(types) * 4))
        .and_then(|size| size.checked_mul(4))
        .ok_or("Type matrix size overflows the allocation limit.")?;
    let limit = status.max_storage.min(status.max_buffer);
    if limit == 0 {
        return Err("Waiting for the GPU to initialize.".into());
    }
    if particles > limit
        || rules > limit
        || TRAIL_BYTES > limit
        || count.div_ceil(256) > status.max_dispatch
    {
        return Err("Requested counts exceed this GPU's buffer or dispatch limits.".into());
    }
    // Per particle: two 32-byte ping-pong buffers the renderer interpolates
    // between, the 32-byte cell-sorted copy, and the 16-byte packed neighbor
    // record the force loop reads. The flat allocation also covers the enlarged
    // creature registry and the per-creature genome array.
    if u64::from(count) * PARTICLE_BYTES + rules + TRAIL_BYTES + 4_000_000 > SIMULATION_BUDGET {
        return Err(
            "Requested world exceeds the simulation memory budget. Reduce particles or types."
                .into(),
        );
    }
    Ok(())
}

/// Bytes of GPU storage each particle occupies across every simulation buffer:
/// two 32-byte ping-pong copies, the 32-byte cell-sorted copy, and the 16-byte
/// packed neighbor record. Cluster labels ride in the sorted copy's spare field.
pub const PARTICLE_BYTES: u64 = 32 + 32 + 32 + 16;
/// Working-set ceiling, in addition to actual device binding limits. Leaves room
/// for render targets, the splat accumulator, Bevy, the desktop, and in-flight
/// old buffers on a 4 GB card.
pub const SIMULATION_BUDGET: u64 = 512 * 1024 * 1024;

pub fn hash(mut x: u32) -> u32 {
    x = (x ^ (x >> 16)).wrapping_mul(0x7feb352d);
    x = (x ^ (x >> 15)).wrapping_mul(0x846ca68b);
    x ^ (x >> 16)
}
pub fn random_rules(types: u32, seed: u32) -> Vec<f32> {
    (0..u64::from(types) * u64::from(types))
        .map(|i| (hash((i as u32).wrapping_add(seed)) & 0xffffff) as f32 / 8388607.5 - 1.0)
        .collect()
}
pub fn palette_color(palette: usize, index: u32, types: u32) -> [f32; 3] {
    let anchors = PALETTES[palette].1;
    let unpack = |hex: u32| Color::srgb_u8((hex >> 16) as u8, (hex >> 8) as u8, hex as u8);
    if types <= 8 {
        return unpack(anchors[index as usize % 8])
            .to_linear()
            .to_f32_array()[..3]
            .try_into()
            .unwrap();
    }
    let t = index as f32 * 8.0 / types as f32;
    let a = Oklaba::from(unpack(anchors[t as usize % 8]));
    let b = Oklaba::from(unpack(anchors[(t as usize + 1) % 8]));
    let c = LinearRgba::from(a.mix(&b, t.fract()));
    [c.red.max(0.0), c.green.max(0.0), c.blue.max(0.0)]
}

pub fn type_data(sim: &Simulation) -> Vec<f32> {
    let mut data =
        Vec::with_capacity(sim.types as usize * 4 + sim.rules.len() * RuleKind::ALL.len());
    for i in 0..sim.types {
        let c = palette_color(sim.palette, i, sim.types);
        data.extend([c[0], c[1], c[2], 1.0]);
    }
    // GPU layout: RGBA colors, then attraction, swirl, alignment, distance matrices.
    for kind in RuleKind::ALL {
        data.extend(sim.matrix(kind).iter());
    }
    data
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn new_rules_are_opt_in_and_gpu_layout_matches_type_count() {
        let mut sim = Simulation::default();
        assert!(
            sim.swirl
                .iter()
                .chain(sim.alignment.iter())
                .all(|x| *x == 0.0)
        );
        sim.set_rule(RuleKind::Swirl, 1, 0.75);
        sim.set_rule(RuleKind::Alignment, 8, -0.5);
        let data = type_data(&sim);
        let colors = sim.types as usize * 4;
        let matrix = sim.types as usize * sim.types as usize;
        assert_eq!(data.len(), colors + RuleKind::ALL.len() * matrix);
        assert_eq!(&data[colors..colors + matrix], sim.rules.as_slice());
        assert_eq!(data[colors + matrix + 1], 0.75);
        assert_eq!(data[colors + 2 * matrix + 8], -0.5);
        assert_eq!(data[colors + matrix + 8], 0.0);
        sim.types = 13;
        sim.randomize();
        assert_eq!(
            type_data(&sim).len(),
            13 * 4 + RuleKind::ALL.len() * 13 * 13
        );
        assert!(
            sim.swirl
                .iter()
                .chain(sim.alignment.iter())
                .all(|x| *x == 0.0)
        );
    }
    #[test]
    fn editing_and_randomizing_matrices_are_independent() {
        let mut sim = Simulation::default();
        let old_snapshot = sim.clone();
        sim.randomize_matrix(RuleKind::Swirl);
        assert_eq!(sim.rules, old_snapshot.rules);
        assert_eq!(sim.alignment, old_snapshot.alignment);
        assert_ne!(sim.swirl, old_snapshot.swirl);
        let first = sim.swirl.clone();
        sim.randomize_matrix(RuleKind::Swirl);
        assert_eq!(first, sim.swirl);
        sim.randomize_matrix(RuleKind::Alignment);
        assert_ne!(sim.swirl, sim.alignment);
        sim.set_rule(RuleKind::Swirl, 0, -0.25);
        assert_eq!(old_snapshot.swirl[0], 0.0);
        let alignment = sim.alignment.clone();
        sim.clear_matrix(RuleKind::Swirl);
        assert!(sim.swirl.iter().all(|x| *x == 0.0));
        assert_eq!(sim.alignment, alignment);
        assert!(sim.rules_revision > old_snapshot.rules_revision);
    }
    #[test]
    fn distance_rules_and_optional_behaviors() {
        let mut sim = Simulation::default();
        assert!(!sim.behavior.preferred_enabled && !sim.behavior.density_enabled);
        assert_eq!(sim.behavior.cycle_mode, 0);
        sim.randomize_matrix(RuleKind::Distance);
        assert!(sim.distance.iter().all(|v| (0.25..=0.8).contains(v)));
        sim.set_rule(RuleKind::Distance, 0, -1.0);
        assert_eq!(sim.distance[0], 0.0);
        sim.set_rule(RuleKind::Distance, 1, 1.0);
        assert_eq!(sim.distance[1], 0.95);
        let data = type_data(&sim);
        assert_eq!(
            &data[sim.types as usize * 4 + sim.rules.len() * 3..],
            sim.distance.as_slice()
        );
    }
    #[test]
    fn seeded_directional_rules() {
        let a = random_rules(8, 42);
        assert_eq!(a, random_rules(8, 42));
        assert_ne!(a, random_rules(8, 43));
        assert_ne!(a[1], a[8]);
        assert!(a.iter().all(|x| (-1.0..=1.0).contains(x)));
    }
    #[test]
    fn allocation_validation() {
        let s = GpuStatus {
            max_storage: 128 << 20,
            max_buffer: 256 << 20,
            max_dispatch: 65535,
            ..default()
        };
        assert!(validate_counts(200_000, 8, &s).is_ok());
        assert!(validate_counts(1_000_000, 8, &s).is_ok(), "1M must fit");
        assert!(
            validate_counts(8_000_000, 8, &s).is_err(),
            "oversized worlds still rejected"
        );
        for (n, t) in [(0, 8), (1, 0), (u32::MAX, 8), (100_000, u32::MAX)] {
            assert!(validate_counts(n, t, &s).is_err());
        }
    }
    #[test]
    fn evolution_defaults_are_sane_and_revision_bumps() {
        let mut s = Simulation::default();
        assert!(s.evolution.enabled);
        assert_eq!(s.evolution.mutation, 0.12);
        assert_eq!(s.evolution.cull_free_fraction, 0.02);
        assert_eq!(s.evolution.influence, 1.0);
        let revision = s.evolution.evolution_revision;
        s.evolution.evolution_revision += 1;
        assert_ne!(s.evolution.evolution_revision, revision);
    }
    #[test]
    fn extended_palettes_are_finite() {
        for p in 0..4 {
            for i in 0..129 {
                assert!(
                    palette_color(p, i, 129)
                        .iter()
                        .all(|v| v.is_finite() && *v >= 0.0)
                );
            }
        }
    }
}

#[cfg(test)]
mod wrapping_tests {
    use super::*;

    #[test]
    fn wrapping_camera_preserves_world_phase() {
        for pan in [
            Vec2::new(103.25, -88.75),
            Vec2::new(-1.001, 1.001),
            Vec2::ONE,
        ] {
            let mut view = Appearance { pan, ..default() };
            view.wrap_pan();
            assert!(view.pan.cmpge(Vec2::splat(-1.0)).all());
            assert!(view.pan.cmplt(Vec2::ONE).all());
            let periods = (pan - view.pan) / 2.0;
            assert!((periods - periods.round()).length() < 0.00001);
        }
    }

    #[test]
    fn tiles_cover_the_viewport_including_soft_edges() {
        for zoom in [0.25, 1.0, 20.0] {
            for aspect in [0.4, 1.0, 16.0 / 9.0, 4.0] {
                for pan in [Vec2::ZERO, Vec2::new(-0.999, 0.999)] {
                    let view = Appearance {
                        zoom,
                        aspect,
                        pan,
                        ..default()
                    };
                    let (first, count) = view.visible_tiles();
                    let extent =
                        Vec2::new(aspect, 1.0) / zoom + Vec2::splat(view.size * 2.0 / WORLD);
                    let left = first.as_vec2() * 2.0 - Vec2::ONE;
                    let right = (first.as_vec2() + count.as_vec2()) * 2.0 - Vec2::ONE;
                    assert!(left.cmple(pan - extent).all());
                    assert!(right.cmpge(pan + extent).all());
                    assert!(count.x > 0 && count.y > 0);
                }
            }
        }
    }
}

#[cfg(test)]
mod zoom_tests {
    use super::*;

    #[test]
    fn cursor_world_point_is_fixed_modulo_wrapping() {
        for viewport in [Vec2::new(1600.0, 900.0), Vec2::new(600.0, 1000.0)] {
            for scale in [1.0, 1.5, 2.0] {
                for fraction in [Vec2::ZERO, Vec2::ONE, Vec2::splat(0.5), Vec2::new(0.2, 0.8)] {
                    for scroll in [-800.0, 400.0] {
                        let mut view = Appearance {
                            pan: Vec2::new(0.99, -0.99),
                            aspect: viewport.x / viewport.y,
                            ..default()
                        };
                        let offset = (fraction * 2.0 - Vec2::ONE) * Vec2::new(view.aspect, -1.0);
                        let before = view.pan + offset / view.zoom;
                        view.zoom_at_cursor(fraction * viewport / scale, viewport / scale, scroll);
                        let after = view.pan + offset / view.zoom;
                        let difference =
                            (after - before + Vec2::ONE).rem_euclid(Vec2::splat(2.0)) - Vec2::ONE;
                        assert!(difference.length() < 0.00001, "cursor moved: {difference}");
                    }
                }
            }
        }
    }

    #[test]
    fn zoom_limits_and_invalid_viewports_do_not_move_camera() {
        for (zoom, scroll) in [(0.25, -100.0), (20.0, 100.0), (1.0, 0.0)] {
            let mut view = Appearance {
                zoom,
                pan: Vec2::new(0.25, -0.75),
                ..default()
            };
            let before = view.clone();
            view.zoom_at_cursor(Vec2::new(20.0, 90.0), Vec2::splat(100.0), scroll);
            assert_eq!(view.pan, before.pan);
            assert_eq!(view.zoom, before.zoom);
        }
        let mut view = Appearance::default();
        view.zoom_at_cursor(Vec2::ONE, Vec2::ZERO, 100.0);
        assert_eq!(view.zoom, 1.0);
        assert_eq!(view.pan, Vec2::ZERO);
    }
}
