struct Particle { position: vec2<f32>, velocity: vec2<f32>, kind: u32, age: f32, pad1: u32, pad2: u32 }
// Compact per-neighbor record consumed by the force loop: 16 bytes, one aligned
// vec4 load, against 4 + 32 bytes for the old indices -> source indirection.
// Velocity is two f16s; speeds are capped at world.z (180), where f16 spacing is
// about 0.1 and the value only feeds soft alignment steering. `kindid` packs the
// kind in the low byte and the creature id above it, keeping the record 16 bytes.
struct Neighbor { position: vec2<f32>, velocity: u32, kindid: u32 }
struct Cell { count: atomic<u32>, offset: u32, cursor: atomic<u32>, pad: u32 }
struct Params {
    counts: vec4<u32>, // particles, types, grid side, seed
    physics: vec4<f32>, // dt, radius, strength, damping
    world: vec4<f32>, // size, core fraction, speed limit, zoom
    view: vec4<f32>, // aspect, pan x, pan y, dot radius
    timing: vec4<f32>, // interpolation, glow, unused, unused
    flags: vec4<u32>, // exact, generation, clustered, unused
    trails: vec4<f32>, // enabled, deposit per second, diffusion rate, half-life
    trail_view: vec4<f32>, // signed response, sensing distance, visibility, field side
    behavior: vec4<f32>, // preferred enabled, density gain, target count, unused
    cycle: vec4<f32>, // mode, interval/cooldown, successor fraction, minimum neighbors
    sampling: vec4<f32>, // neighbor sample budget, tile rows, viewport w, viewport h
    detect: vec4<f32>, // bond radius, per-type minimum, promote steps, enabled
    outline: vec4<f32>, // stamp radius, isolevel, visibility, field side
    evolution: vec4<f32>, // enabled, mutation, cull free fraction, influence
}
@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read_write> source: array<Particle>;
@group(0) @binding(2) var<storage, read_write> destination: array<Particle>;
@group(0) @binding(3) var<storage, read_write> cells: array<Cell>;
@group(0) @binding(4) var<storage, read_write> neighbors: array<Neighbor>;
@group(0) @binding(5) var<storage, read_write> blocks: array<u32>;
@group(0) @binding(6) var<storage, read> types: array<f32>;
struct Chemical {
    density: f32, next: f32, deposits: atomic<u32>, pad: u32,
    pigment: vec4<f32>, next_pigment: vec4<f32>,
    red: atomic<u32>, green: atomic<u32>, blue: atomic<u32>, color_pad: u32,
}
@group(0) @binding(7) var<storage, read_write> chemicals: array<Chemical>;
// Particles reordered into cell order every step. Holds the pre-update state, so
// the renderer also uses it as the interpolation start that matches destination.
@group(0) @binding(8) var<storage, read_write> sorted: array<Particle>;
// Cluster detection. Labels are indexed by sorted slot; `parent` is a union-find
// over creature ids, reset every step so stale merges cannot accumulate.
// Union-find lives in the record it merges, and per-particle labels live in the
// sorted copy's spare `pad2`, so detection costs exactly one extra binding. The
// baseline WebGPU limit is 8 storage buffers per stage and the force pass already
// used all 8; keeping the count low is what stops that becoming a hard wall.
struct Creature {
    count: atomic<u32>,
    parent: atomic<u32>,
    cos_x: atomic<u32>, sin_x: atomic<u32>,
    cos_y: atomic<u32>, sin_y: atomic<u32>,
    centroid: vec2<f32>,
    persistence: u32, peak: u32, flags: u32, pad0: u32,
    // Per-type counts, one per kind up to MAX_CREATURE_TYPES. The promotion
    // threshold is per type, so a body mixing kinds must hold its minimum in
    // every one of them rather than just in the total. Appended at the end so
    // the display shader's read-only view of the fields above keeps its offsets.
    type_counts: array<atomic<u32>, 16>,
    // The measured body frame: fixed-point sums of unit centroid offsets are
    // accumulated each step, then `promote_frame` turns them into the vec2
    // (principal axis angle, length aspect) used by the anisotropic forces and
    // the outline stamp.
    frame_xx: atomic<u32>, frame_xy: atomic<u32>, frame_yy: atomic<u32>,
    // A split records the previous id as this body's lineage; `mutated` keeps a
    // recycled id from re-mutating on the same ancestry. `generation` is how many
    // splits deep this body is -- 0 for a body assembled from free material,
    // parent + 1 for a split child -- and is what the outline colour keys on.
    // `frame` is the last measured shape, which survives the per-step accumulator
    // resets.
    lineage: atomic<u32>, mutated: u32, generation: u32,
    frame: vec2<f32>,
}
@group(0) @binding(9) var<storage, read_write> creatures: array<Creature>;
// One anisotropic kernel per creature id per recognized kind, as vec2(ratio,
// orientation). Reset to isotropic whenever the world restarts.
@group(0) @binding(10) var<storage, read_write> genomes: array<vec2<f32>>;
// One word-pair per creature id: a birth marker (the creature's own id while it
// is scheduled to split this step, else 0) and the fresh id reserved for its
// child. `pick_split` writes both; `apply_split` consumes them, so the child
// halves of a body leave this step already labelled.
@group(0) @binding(11) var<storage, read_write> births: array<u32>;
const MAX_CREATURES: u32 = 4096u;
// Types the per-type promotion check enumerates per creature. Matches the rule
// cache capacity so a world is never taller than the check it can run.
const MAX_CREATURE_TYPES: u32 = 16u;
// Types stay in the low byte of the packed neighbor kind while the creature id
// lives above it, so the record stays 16 bytes wide.
const KIND_MASK: u32 = 0xffu;
// A tracked body this elongated (measured length aspect) is deep in an unstable
// stretch, so it splits into two viable halves instead of thinning to a string.
const SPLIT_ASPECT: f32 = 3.0;
// Mirrors BOND_THRESHOLD in src/creatures.rs; the two must agree or the detector
// will look for bodies the inference never predicted.
const BOND_THRESHOLD: f32 = 0.15;
const CIRCULAR_SCALE: f32 = 4096.0;
const TAU: f32 = 6.283185307;
fn chemical_index(xy: vec2<i32>) -> u32 {
    let side = i32(p.trail_view.w);
    let wrapped = ((xy % side) + side) % side;
    return u32(wrapped.y * side + wrapped.x);
}
fn scent(position: vec2<f32>) -> f32 {
    let coord = position / p.world.x * p.trail_view.w - 0.5;
    let base = vec2<i32>(floor(coord));
    let f = fract(coord);
    let a = mix(chemicals[chemical_index(base)].density, chemicals[chemical_index(base + vec2(1, 0))].density, f.x);
    let b = mix(chemicals[chemical_index(base + vec2(0, 1))].density, chemicals[chemical_index(base + vec2(1, 1))].density, f.x);
    return mix(a, b, f.y);
}
@compute @workgroup_size(256)
fn clear_trails(@builtin(global_invocation_id) id: vec3<u32>) {
    if id.x >= u32(p.trail_view.w * p.trail_view.w) { return; }
    chemicals[id.x].density = 0.0;
    chemicals[id.x].next = 0.0;
    chemicals[id.x].pigment = vec4(0.0);
    chemicals[id.x].next_pigment = vec4(0.0);
    atomicStore(&chemicals[id.x].deposits, 0u);
    atomicStore(&chemicals[id.x].red, 0u);
    atomicStore(&chemicals[id.x].green, 0u);
    atomicStore(&chemicals[id.x].blue, 0u);
}
@compute @workgroup_size(256)
fn diffuse_trails(@builtin(global_invocation_id) id: vec3<u32>) {
    let side = u32(p.trail_view.w);
    if id.x >= side * side { return; }
    let xy = vec2<i32>(i32(id.x % side), i32(id.x / side));
    let average = (chemicals[chemical_index(xy + vec2(1, 0))].density
        + chemicals[chemical_index(xy + vec2(-1, 0))].density
        + chemicals[chemical_index(xy + vec2(0, 1))].density
        + chemicals[chemical_index(xy + vec2(0, -1))].density) * 0.25;
    // Convex diffusion remains nonnegative and stable for every UI setting.
    // Read only density here; commit next in a separate dispatch to avoid races.
    let diffused = mix(chemicals[id.x].density, average, 1.0 - exp(-p.trails.z * p.physics.x));
    let deposit = f32(atomicLoad(&chemicals[id.x].deposits)) * p.trails.y * p.physics.x;
    let decay = exp(-0.69314718 * p.physics.x / p.trails.w);
    let density = (diffused + deposit) * decay;
    chemicals[id.x].next = min(100.0, density);
    let average_color = (chemicals[chemical_index(xy + vec2(1, 0))].pigment
        + chemicals[chemical_index(xy + vec2(-1, 0))].pigment
        + chemicals[chemical_index(xy + vec2(0, 1))].pigment
        + chemicals[chemical_index(xy + vec2(0, -1))].pigment) * 0.25;
    let color_deposit = vec4(f32(atomicLoad(&chemicals[id.x].red)),
        f32(atomicLoad(&chemicals[id.x].green)), f32(atomicLoad(&chemicals[id.x].blue)), 0.0)
        / 255.0 * p.trails.y * p.physics.x;
    let pigment = (mix(chemicals[id.x].pigment, average_color, 1.0 - exp(-p.trails.z * p.physics.x)) + color_deposit) * decay;
    // Cap pigment and concentration together so saturated cells keep their hue.
    chemicals[id.x].next_pigment = pigment * min(1.0, 100.0 / max(density, 0.000001));
}
@compute @workgroup_size(256)
fn commit_trails(@builtin(global_invocation_id) id: vec3<u32>) {
    if id.x >= u32(p.trail_view.w * p.trail_view.w) { return; }
    chemicals[id.x].density = chemicals[id.x].next;
    chemicals[id.x].pigment = chemicals[id.x].next_pigment;
    atomicStore(&chemicals[id.x].deposits, 0u);
    atomicStore(&chemicals[id.x].red, 0u);
    atomicStore(&chemicals[id.x].green, 0u);
    atomicStore(&chemicals[id.x].blue, 0u);
}
fn hash(input: u32) -> u32 {
    var x = input;
    x = (x ^ (x >> 16u)) * 0x7feb352du;
    x = (x ^ (x >> 15u)) * 0x846ca68bu;
    return x ^ (x >> 16u);
}
fn random(input: u32) -> f32 { return f32(hash(input) & 0xffffffu) / 16777216.0; }
fn cell_id(pos: vec2<f32>) -> u32 {
    let xy = min(vec2<u32>(pos / p.world.x * f32(p.counts.z)), vec2<u32>(p.counts.z - 1u));
    return xy.y * p.counts.z + xy.x;
}
fn pack_neighbor(particle: Particle) -> Neighbor {
    return Neighbor(particle.position, pack2x16float(particle.velocity), particle.kind | (particle.pad1 << 8u));
}
@compute @workgroup_size(256)
fn init(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x; if i >= p.counts.x { return; }
    var particle = Particle(vec2(0.0), vec2(0.0), 0u, 0.0, 0u, 0u);
    var pos = vec2(random(i * 3u + p.counts.w), random(i * 3u + p.counts.w + 1u)) * p.world.x;
    if p.flags.z != 0u { pos = p.world.x * 0.5 + (pos - p.world.x * 0.5) * 0.06; }
    particle = Particle(pos, vec2(0.0), i % p.counts.y, random(i * 3u + p.counts.w + 2u) * p.cycle.y, 0u, 0u);
    source[i] = particle; destination[i] = particle;
    // Seed the sorted copy too: the renderer interpolates from it, and a reset
    // frame can display before any step has run.
    sorted[i] = particle;
    neighbors[i] = pack_neighbor(particle);
}
@compute @workgroup_size(256)
fn clear(@builtin(global_invocation_id) id: vec3<u32>) {
    if id.x >= p.counts.z * p.counts.z { return; }
    atomicStore(&cells[id.x].count, 0u); atomicStore(&cells[id.x].cursor, 0u);
}
@compute @workgroup_size(256)
fn count(@builtin(global_invocation_id) id: vec3<u32>) {
    if id.x < p.counts.x { atomicAdd(&cells[cell_id(source[id.x].position)].count, 1u); }
}
var<workgroup> scan: array<u32, 256>;
@compute @workgroup_size(256)
fn prefix(@builtin(global_invocation_id) id: vec3<u32>, @builtin(local_invocation_id) local: vec3<u32>, @builtin(workgroup_id) group: vec3<u32>) {
    var value = 0u;
    if id.x < p.counts.z * p.counts.z { value = atomicLoad(&cells[id.x].count); }
    scan[local.x] = value; workgroupBarrier();
    for (var stride = 1u; stride < 256u; stride *= 2u) {
        var add = 0u; if local.x >= stride { add = scan[local.x - stride]; }
        workgroupBarrier(); scan[local.x] += add; workgroupBarrier();
    }
    if id.x < p.counts.z * p.counts.z { cells[id.x].offset = scan[local.x] - value; }
    if local.x == 255u { blocks[group.x] = scan[255]; }
}
@compute @workgroup_size(256)
fn prefix_blocks(@builtin(local_invocation_id) local: vec3<u32>) {
    let block_count = (p.counts.z * p.counts.z + 255u) / 256u;
    var value = 0u; if local.x < block_count { value = blocks[local.x]; }
    scan[local.x] = value; workgroupBarrier();
    for (var stride = 1u; stride < 256u; stride *= 2u) {
        var add = 0u; if local.x >= stride { add = scan[local.x - stride]; }
        workgroupBarrier(); scan[local.x] += add; workgroupBarrier();
    }
    if local.x < block_count { blocks[local.x] = scan[local.x] - value; }
}
// Reorders particles into cell order instead of writing an index. The force loop
// then reads a cell's members as one contiguous run, and the threads of a
// workgroup share a neighborhood instead of scattering across the whole world.
@compute @workgroup_size(256)
fn scatter(@builtin(global_invocation_id) id: vec3<u32>) {
    if id.x >= p.counts.x { return; }
    let particle = source[id.x];
    let c = cell_id(particle.position);
    let slot = cells[c].offset + blocks[c / 256u] + atomicAdd(&cells[c].cursor, 1u);
    sorted[slot] = particle;
    neighbors[slot] = pack_neighbor(particle);
}
fn force(r: f32, coefficient: f32) -> f32 {
    if r < p.world.y { return r / p.world.y - 1.0; }
    return coefficient * (1.0 - abs(2.0 * r - 1.0 - p.world.y) / (1.0 - p.world.y));
}
fn preferred_force(r: f32, separation: f32) -> f32 {
    let d = clamp(separation, 0.2, 0.95);
    if r < d { return (r - d) / d; }
    return 4.0 * (r - d) * (1.0 - r) / ((1.0 - d) * (1.0 - d));
}
// The four rule matrices, staged once per workgroup. Every interaction reads four
// of these; from workgroup memory they stop competing with neighbor data for L1.
// Capacity covers 16 types (4 * 16 * 16); larger worlds fall back to global reads.
const RULE_CAPACITY: u32 = 1024u;
var<workgroup> rules: array<f32, 1024>;
fn rule(index: u32, cached: bool) -> f32 {
    if cached { return rules[index]; }
    return types[p.counts.y * 4u + index];
}
// The per-particle anisotropic metric. `scale` stretches the interaction radius
// along the creature's rotated major axis; a scale of 1 is the shared rules
// exactly as written. cos/sin rotate that axis into world space.
struct Metric { cos: f32, sin: f32, scale: f32 }
fn id_metric() -> Metric { return Metric(1.0, 0.0, 1.0); }
fn metric_length(d: vec2<f32>, m: Metric) -> f32 {
    let u = m.cos * d.x + m.sin * d.y;
    let v = -m.sin * d.x + m.cos * d.y;
    let inv = 1.0 / max(m.scale, 0.0001);
    let q = u * u * inv * inv + v * v * m.scale * m.scale;
    return sqrt(q);
}
fn creature_metric(c: u32, kind: u32) -> Metric {
    let frame = creatures[c].frame;
    let measured = max(frame.y, 1.0);
    let genome = genomes[c * MAX_CREATURE_TYPES + kind];
    // A round body has no measurable frame, so the genome only steers once the
    // body actually elongates; influence scales how far the encoded kernel wins
    // over the shared rules.
    let influence = clamp(p.evolution.w, 0.0, 1.0);
    let blend = clamp((measured - 1.0) / 2.0, 0.0, 1.0) * influence;
    let angle = mix(genome.y, frame.x, blend);
    let scale = max(mix(1.0, genome.x, influence), 1.0);
    return Metric(cos(angle), sin(angle), scale);
}
fn wrap_half(a: f32) -> f32 {
    var t = fract(a / TAU) * TAU;
    if t >= TAU * 0.5 { t -= TAU; }
    return t;
}
// Every neighbor uses the affected particle's current local rules.
struct Response {
    acceleration: vec2<f32>,
    density: vec2<f32>,
    alignment: vec2<f32>,
    weight: f32,
    nearby: f32,
    successors: f32,
}
fn empty_response() -> Response {
    return Response(vec2(0.0), vec2(0.0), vec2(0.0), 0.0, 0.0, 0.0);
}
fn merge(x: Response, y: Response) -> Response {
    return Response(x.acceleration + y.acceleration, x.density + y.density,
        x.alignment + y.alignment, x.weight + y.weight, x.nearby + y.nearby, x.successors + y.successors);
}
fn interact(a: Particle, b: Neighbor, row: u32, successor: u32, cached: bool,
    metric: Metric, aniso: bool) -> Response {
    var out = empty_response();
    var delta = b.position - a.position;
    delta -= round(delta / p.world.x) * p.world.x;
    let euclid = length(delta);
    let b_kind = b.kindid & KIND_MASK;
    // The anisotropic metric only applies within one body: between bodies a
    // shared frame is meaningless, so those interactions stay isotropic.
    var distance = euclid;
    if aniso && (b.kindid >> 8u) == a.pad1 { distance = metric_length(delta, metric); }
    if distance >= p.physics.y { return out; }
    out.nearby = 1.0;
    if b_kind == successor { out.successors = 1.0; }
    if euclid <= 0.00001 { return out; }
    let matrix_size = p.counts.y * p.counts.y;
    let pair = row + b_kind;
    let coefficient = rule(pair, cached);
    let swirl = rule(pair + matrix_size, cached);
    let align = rule(pair + 2u * matrix_size, cached);
    let preferred = rule(pair + 3u * matrix_size, cached);
    let direction = delta / euclid;
    // Short-range repulsion stays isotropic even inside a body: warping the core
    // is what lets an elongated body collapse in on itself.
    let core_r = euclid / p.physics.y;
    if core_r < p.world.y {
        out.acceleration = direction * force(core_r, 0.0);
        return out;
    }
    let r = distance / p.physics.y;
    var radial = force(r, coefficient);
    if p.behavior.x > 0.0 && preferred > 0.0 {
        radial = preferred_force(r, preferred);
        if r < p.world.y { radial = min(radial, force(r, 0.0)); }
    }
    out.acceleration = direction * radial;
    // A clockwise rotation of the vector toward the neighbor produces
    // counterclockwise motion around that neighbor.
    let envelope = select(0.0, max(0.0, 1.0 - abs(2.0 * r - 1.0 - p.world.y) / (1.0 - p.world.y)), r >= p.world.y);
    out.acceleration += vec2(direction.y, -direction.x) * swirl * envelope;
    out.density = direction * envelope;
    // Signed alignment steers toward matching or opposite velocity.
    // Normalize by neighborhood weight so dense clusters cannot multiply
    // the steering rate; magnitude still controls each pair's influence.
    let weight = (1.0 - r) * (1.0 - r);
    out.alignment = (align * unpack2x16float(b.velocity) - abs(align) * a.velocity) * weight;
    out.weight = weight;
    return out;
}
@compute @workgroup_size(256)
fn update(@builtin(global_invocation_id) id: vec3<u32>, @builtin(local_invocation_index) lane: u32) {
    // Uniform across the workgroup, so the barrier below stays in uniform control
    // flow and must sit ahead of the out-of-range return.
    let matrix_size = p.counts.y * p.counts.y;
    let rule_floats = matrix_size * 4u;
    let cached = rule_floats <= RULE_CAPACITY;
    if cached {
        let colors = p.counts.y * 4u;
        for (var k = lane; k < rule_floats; k += 256u) { rules[k] = types[colors + k]; }
    }
    workgroupBarrier();
    let i = id.x; if i >= p.counts.x { return; }
    let a = sorted[i];
    let row = a.kind * p.counts.y;
    let center = vec2<i32>(a.position / p.world.x * f32(p.counts.z));
    // Hoist each cell's run base and length: the sampling loop below used to
    // reload both from storage on every sample.
    var starts: array<u32, 9>;
    var lengths: array<u32, 9>;
    var total = 0u;
    for (var k = 0u; k < 9u; k++) {
        let xy = (center + vec2<i32>(i32(k % 3u) - 1, i32(k / 3u) - 1) + i32(p.counts.z)) % i32(p.counts.z);
        let c = u32(xy.y) * p.counts.z + u32(xy.x);
        let run = atomicLoad(&cells[c].count);
        starts[k] = cells[c].offset + blocks[c / 256u];
        lengths[k] = run;
        total += run;
    }
    let cap = max(16u, u32(p.sampling.x));
    let budget = select(min(total, cap), total, p.flags.x != 0u);
    var foreign = empty_response();
    let successor = (a.kind + 1u) % p.counts.y;
    // One metric per particle is hoisted out of the neighbor loop: every pair
    // this body samples then shares the same anisotropy, so per-pair genome
    // reads and trig are paid once instead of per interaction.
    var metric = id_metric();
    let aniso = p.evolution.x > 0.0 && a.pad1 != 0u && a.pad1 < MAX_CREATURES;
    if aniso { metric = creature_metric(a.pad1, a.kind); }
    var neighbor_slot = 0u; var base = 0u;
    // Stratified sampling across the concatenated cell ranges, weighted back to
    // the full population. Rotation changes every step; no cell is privileged.
    let jitter = random(i + p.flags.y * 1664525u);
    for (var sample = 0u; sample < budget; sample++) {
        let rank = min(u32((f32(sample) + jitter) * f32(total) / f32(budget)), total - 1u);
        loop {
            if rank < base + lengths[neighbor_slot] || neighbor_slot == 8u { break; }
            base += lengths[neighbor_slot]; neighbor_slot++;
        }
        let slot = starts[neighbor_slot] + rank - base;
        if slot == i { continue; }
        foreign = merge(foreign, interact(a, neighbors[slot], row, successor, cached, metric, aniso));
    }
    let sample_scale = f32(total) / max(1.0, f32(budget));
    let nearby = foreign.nearby;
    let successors = foreign.successors;
    let estimated_neighbors = nearby * sample_scale;
    let crowding = clamp(1.0 - estimated_neighbors / max(1.0, p.behavior.z), -1.0, 1.0);
    var acceleration = (foreign.acceleration + foreign.density * crowding * p.behavior.y) * sample_scale;
    var alignment = foreign.alignment * sample_scale;
    let alignment_weight = foreign.weight * sample_scale;
    if alignment_weight > 0.0 { alignment /= alignment_weight; }
    var velocity = (a.velocity + acceleration * p.physics.z * p.physics.x + alignment * (1.0 - exp(-4.0 * p.physics.x))) * exp(-p.physics.w * p.physics.x);
    if p.trails.x > 0.0 {
        let sensor = p.trail_view.y;
        let gradient = vec2(scent(a.position + vec2(sensor, 0.0)) - scent(a.position - vec2(sensor, 0.0)),
            scent(a.position + vec2(0.0, sensor)) - scent(a.position - vec2(0.0, sensor)));
        // Saturated steering bounds acceleration even in a crowded scent hotspot.
        velocity += gradient / (1.0 + length(gradient)) * p.trail_view.x * p.physics.x;
    }
    let speed = length(velocity);
    if speed > p.world.z { velocity *= p.world.z / speed; }
    var pos = a.position + velocity * p.physics.x;
    pos -= floor(pos / p.world.x) * p.world.x;
    var kind = a.kind;
    var age = a.age;
    if p.cycle.x > 0.0 && p.counts.y > 1u {
        age = min(age + p.physics.x, p.cycle.y);
        let ready = age >= p.cycle.y;
        let timer_trigger = p.cycle.x == 1.0 || p.cycle.x == 3.0;
        let neighbor_trigger = (p.cycle.x == 2.0 || p.cycle.x == 3.0)
            && estimated_neighbors >= p.cycle.w && nearby > 0.0
            && successors / nearby >= p.cycle.z;
        if (ready && timer_trigger) || (age >= select(p.cycle.y, min(0.5, p.cycle.y), p.cycle.x == 3.0) && neighbor_trigger) { kind = successor; age = 0.0; }
    }
    // Carry the creature id forward, and stash the previous id in `pad2` so a
    // split (a tracked body handing one particle a new id) can be attributed as
    // a birth by `progeny`.
    var creature = a.pad1;
    if p.detect.w != 0.0 {
        let label = a.pad2;
        creature = select(0u, atomicLoad(&creatures[label].parent), label != 0u);
    }
    destination[i] = Particle(pos, velocity, kind, age, creature, a.pad1);
    if p.trails.x > 0.0 {
        let cell = chemical_index(vec2<i32>(floor(pos / p.world.x * p.trail_view.w)));
        let c = kind * 4u;
        let rgb = vec3<u32>(round(clamp(vec3(types[c], types[c + 1u], types[c + 2u]), vec3(0.0), vec3(1.0)) * 255.0));
        atomicAdd(&chemicals[cell].deposits, 1u);
        atomicAdd(&chemicals[cell].red, rgb.x);
        atomicAdd(&chemicals[cell].green, rgb.y);
        atomicAdd(&chemicals[cell].blue, rgb.z);
    }
}

// ---- Cluster detection ----------------------------------------------------
// Creature identity has to survive across steps, but the cell sort renumbers
// every particle each step, so labels cannot be particle indices. Each particle
// carries a persistent id in `pad1`; bonds found this step are unioned through
// `parent`, and `resolve` flattens the chains. Convergence is amortized over a
// few steps, which is the right granularity for a continuous simulation.

// Mirrors `bonds()` in src/creatures.rs. Both directions must pull: a pair where
// one side attracts and the other flees is a chase, which produces motion rather
// than a body. Preferred distances override the attraction curve outright.
fn types_bond(a: u32, b: u32) -> bool {
    let t = p.counts.y;
    let m = t * t;
    let colors = t * 4u;
    if p.behavior.x > 0.0 {
        if types[colors + 3u * m + a * t + b] > 0.0 || types[colors + 3u * m + b * t + a] > 0.0 {
            return true;
        }
    }
    return types[colors + a * t + b] + types[colors + b * t + a] > BOND_THRESHOLD;
}
@compute @workgroup_size(256)
fn clear_creatures(@builtin(global_invocation_id) id: vec3<u32>) {
    let c = id.x;
    if c >= MAX_CREATURES { return; }
    // Slot 0 is never a creature -- id 0 means unassigned -- so its count field
    // is reused as the monotonic id allocator, and must survive the reset or
    // fresh clusters would collide with live ids every step. Its `parent` and
    // `cos_x` are repurposed as the free-particle tally and the cull candidate
    // for the low-durability dissolve, and start each step clean here.
    if c == 0u {
        atomicStore(&creatures[0].parent, 0u);
        atomicStore(&creatures[0].cos_x, 0xffffffffu);
        return;
    }
    // Only the per-step accumulators reset. Persistence, peak size and tracked
    // state have to survive the step or nothing could ever be promoted.
    atomicStore(&creatures[c].count, 0u);
    atomicStore(&creatures[c].cos_x, 0u);
    atomicStore(&creatures[c].sin_x, 0u);
    atomicStore(&creatures[c].cos_y, 0u);
    atomicStore(&creatures[c].sin_y, 0u);
    // Bonds are re-derived every step, so the union-find starts flat.
    atomicStore(&creatures[c].parent, c);
    // Per-type counts are per-step accumulators like `count`, so they reset too.
    for (var t = 0u; t < min(p.counts.y, MAX_CREATURE_TYPES); t++) {
        atomicStore(&creatures[c].type_counts[t], 0u);
    }
    // The body frame is re-measured every step the body is visible.
    atomicStore(&creatures[c].frame_xx, 0u);
    atomicStore(&creatures[c].frame_xy, 0u);
    atomicStore(&creatures[c].frame_yy, 0u);
}
@compute @workgroup_size(256)
fn bond(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    if i >= p.counts.x || p.detect.w == 0.0 { return; }
    let a = sorted[i];
    let reach = p.detect.x * p.detect.x;
    let center = vec2<i32>(a.position / p.world.x * f32(p.counts.z));
    var best = a.pad1;
    var claimant = i;
    var bonded = false;
    for (var k = 0u; k < 9u; k++) {
        let xy = (center + vec2<i32>(i32(k % 3u) - 1, i32(k / 3u) - 1) + i32(p.counts.z)) % i32(p.counts.z);
        let c = u32(xy.y) * p.counts.z + u32(xy.x);
        let start = cells[c].offset + blocks[c / 256u];
        let run = atomicLoad(&cells[c].count);
        for (var n = 0u; n < run; n++) {
            let slot = start + n;
            if slot == i { continue; }
            // Position and kind come from the packed record, which is the
            // cache-friendly array; only the id needs the wider particle.
            let nb = neighbors[slot];
            var delta = nb.position - a.position;
            delta -= round(delta / p.world.x) * p.world.x;
            if dot(delta, delta) > reach { continue; }
            if !types_bond(a.kind, nb.kindid & KIND_MASK) { continue; }
            bonded = true;
            claimant = min(claimant, slot);
            let other = sorted[slot].pad1;
            if other != 0u && (best == 0u || other < best) { best = other; }
        }
    }
    if !bonded {
        // A particle with no bonded neighbour is free material, not a creature.
        sorted[i].pad2 = 0u;
        return;
    }
    if best == 0u {
        // Nothing nearby carries an id yet. Exactly one particle claims a fresh
        // one; the lowest sorted slot is a deterministic tie-break that needs no
        // extra communication. The rest inherit it on the next step.
        if claimant != i {
            sorted[i].pad2 = 0u;
            return;
        }
        best = atomicAdd(&creatures[0].count, 1u) % (MAX_CREATURES - 1u) + 1u;
        // Ids wrap, so a recycled slot must drop the previous occupant's history,
        // both the tracking fields and the inherited line.
        creatures[best].persistence = 0u;
        creatures[best].peak = 0u;
        creatures[best].flags = 0u;
        atomicStore(&creatures[best].lineage, 0u);
        creatures[best].mutated = 0u;
        creatures[best].generation = 0u;
        creatures[best].frame = vec2(0.0);
    }
    sorted[i].pad2 = best;
    // Taking a smaller id than the one carried in means two creatures just met:
    // record the merge so every other member follows on the next step. No second
    // neighbourhood scan is needed to discover it.
    if a.pad1 != 0u && best < a.pad1 { atomicMin(&creatures[a.pad1].parent, best); }
}
// One pointer jump per dispatch. The host runs several, flattening merge chains
// in log steps rather than cluster-diameter steps.
@compute @workgroup_size(256)
fn resolve(@builtin(global_invocation_id) id: vec3<u32>) {
    let c = id.x;
    if c >= MAX_CREATURES { return; }
    let up = atomicLoad(&creatures[c].parent);
    atomicMin(&creatures[c].parent, atomicLoad(&creatures[up].parent));
}
@compute @workgroup_size(256)
fn creature_stats(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    if i >= p.counts.x || p.detect.w == 0.0 { return; }
    let d = destination[i];
    let c = d.pad1;
    if c == 0u { return; }
    // Tally regardless of the count-carry so the promotion pass can enforce the
    // per-type minimum on every kind the body actually contains.
    if d.kind < MAX_CREATURE_TYPES {
        atomicAdd(&creatures[c].type_counts[d.kind], 1u);
    }
    atomicAdd(&creatures[c].count, 1u);
    // Circular mean. The world is a torus, so averaging raw coordinates puts the
    // centroid of a body straddling the seam on the opposite side of the world.
    // Float atomics do not exist, so the unit vectors are biased into u32.
    let angle = d.position / p.world.x * TAU;
    atomicAdd(&creatures[c].cos_x, u32((cos(angle.x) + 1.0) * CIRCULAR_SCALE));
    atomicAdd(&creatures[c].sin_x, u32((sin(angle.x) + 1.0) * CIRCULAR_SCALE));
    atomicAdd(&creatures[c].cos_y, u32((cos(angle.y) + 1.0) * CIRCULAR_SCALE));
    atomicAdd(&creatures[c].sin_y, u32((sin(angle.y) + 1.0) * CIRCULAR_SCALE));
}
@compute @workgroup_size(256)
fn promote(@builtin(global_invocation_id) id: vec3<u32>) {
    let c = id.x;
    if c == 0u || c >= MAX_CREATURES { return; }
    let n = atomicLoad(&creatures[c].count);
    if n == 0u {
        creatures[c].persistence = 0u;
        creatures[c].peak = 0u;
        creatures[c].flags = 0u;
        return;
    }
    let inv = 1.0 / (f32(n) * CIRCULAR_SCALE);
    let mean = vec4(f32(atomicLoad(&creatures[c].cos_x)), f32(atomicLoad(&creatures[c].sin_x)),
        f32(atomicLoad(&creatures[c].cos_y)), f32(atomicLoad(&creatures[c].sin_y))) * inv - 1.0;
    let angle = vec2(atan2(mean.y, mean.x), atan2(mean.w, mean.z));
    creatures[c].centroid = fract(angle / TAU + 1.0) * p.world.x;
    creatures[c].peak = max(creatures[c].peak, n);
    // A body counts only while every kind present in it clears the per-type
    // minimum: a cluster that is mostly one kind with a sprinkle of another is
    // an accidental crowd, not the mixed body the rules call for. Kinds beyond
    // the enumeration cap are invisible here, so the total-size bar still gates
    // bodies made entirely of those kinds.
    var per_type = true;
    let kinds = min(p.counts.y, MAX_CREATURE_TYPES);
    for (var t = 0u; t < kinds; t++) {
        let here = atomicLoad(&creatures[c].type_counts[t]);
        if here != 0u && here < u32(p.detect.y) { per_type = false; }
    }
    if per_type && n >= u32(p.detect.y) {
        creatures[c].persistence += 1u;
    } else {
        creatures[c].persistence = 0u;
    }
    // Tracked only after surviving long enough to be a body rather than a
    // momentary crowd. This is what stops outlines flickering on and off.
    creatures[c].flags = select(0u, 1u, creatures[c].persistence >= u32(p.detect.z));
}

// ---- Evolution ------------------------------------------------------------
// Each creature carries one anisotropic kernel (ratio, orientation) per kind in
// `genomes`, layered on the shared matrices. Reproduction is a tracked body
// splitting into two viable halves: when detection hands a particle a new id and
// that id's body is viable this step, the old id becomes its lineage.
// `promote_frame` measures each body's principal frame and mutates a newborn
// exactly once, at birth. When the free-particle pool runs short, the least
// durable tracked body is dissolved back into free material as a pressure valve.

@compute @workgroup_size(256)
fn reset_evolution(@builtin(global_invocation_id) id: vec3<u32>) {
    let c = id.x;
    if c >= MAX_CREATURES { return; }
    for (var k = 0u; k < MAX_CREATURE_TYPES; k++) {
        genomes[c * MAX_CREATURE_TYPES + k] = vec2(1.0, 0.0);
    }
    births[2u * c] = 0u;
    births[2u * c + 1u] = 0u;
    if c != 0u {
        atomicStore(&creatures[c].lineage, 0u);
        creatures[c].mutated = 0u;
        creatures[c].generation = 0u;
        creatures[c].frame = vec2(0.0);
    }
}
@compute @workgroup_size(256)
fn progeny(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    if i >= p.counts.x || p.detect.w == 0.0 || p.evolution.x == 0.0 { return; }
    let d = destination[i];
    let prev = d.pad2;
    let child = d.pad1;
    if prev == 0u || child == 0u || prev == child { return; }
    if prev >= MAX_CREATURES || child >= MAX_CREATURES { return; }
    // A tracked parent plus a child already clearing the viability bar means the
    // body split into two viable halves; the child inherits the parent's line.
    if (creatures[prev].flags & 1u) != 0u && atomicLoad(&creatures[child].count) >= u32(p.detect.y) {
        atomicMax(&creatures[child].lineage, prev);
    }
}
@compute @workgroup_size(256)
fn creature_frame(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    if i >= p.counts.x || p.detect.w == 0.0 || p.evolution.x == 0.0 { return; }
    let d = destination[i];
    let c = d.pad1;
    if c == 0u { return; }
    // Unit offset from the centroid measured this step, accumulated in the same
    // fixed-point scheme as the circular mean: the products of unit vectors stay
    // within [-1, 1] and the cross term is biased up.
    let centroid = creatures[c].centroid;
    var delta = d.position - centroid;
    delta -= round(delta / p.world.x) * p.world.x;
    let span = max(length(delta), 0.00001);
    let u = delta / span;
    let scale = 1024.0;
    atomicAdd(&creatures[c].frame_xx, u32(u.x * u.x * scale));
    atomicAdd(&creatures[c].frame_yy, u32(u.y * u.y * scale));
    atomicAdd(&creatures[c].frame_xy, u32((u.x * u.y + 1.0) * scale));
}
@compute @workgroup_size(256)
fn promote_frame(@builtin(global_invocation_id) id: vec3<u32>) {
    let c = id.x;
    if c == 0u || c >= MAX_CREATURES || p.detect.w == 0.0 || p.evolution.x == 0.0 { return; }
    let n = atomicLoad(&creatures[c].count);
    if n == 0u { return; }
    // Eigen-decomposition of the second-moment matrix [[xx, xy], [xy, yy]].
    let scale = 1024.0;
    let inv = 1.0 / (f32(n) * scale);
    let xx = f32(atomicLoad(&creatures[c].frame_xx)) * inv;
    let xy = f32(atomicLoad(&creatures[c].frame_xy)) * inv - 1.0;
    let yy = f32(atomicLoad(&creatures[c].frame_yy)) * inv;
    let trace = xx + yy;
    let diff = xx - yy;
    let radius = sqrt(diff * diff + 4.0 * xy * xy);
    let angle = 0.5 * atan2(2.0 * xy, diff);
    let major = 0.5 * (trace + radius);
    let minor = max(0.5 * (trace - radius), 0.00001);
    // Aspect falls out in the length domain, where 1 is a round body.
    creatures[c].frame = vec2(angle, clamp(sqrt(major / minor), 1.0, 12.0));
    // A slot that just inherited a line mutates exactly once, at birth, layering
    // noise on the parent's kernels so selection has variation to sort by.
    let parent = atomicLoad(&creatures[c].lineage);
    if parent != 0u && parent < MAX_CREATURES
        && (creatures[c].flags & 1u) != 0u && creatures[c].mutated == 0u {
        for (var k = 0u; k < min(p.counts.y, MAX_CREATURE_TYPES); k++) {
            let child_index = c * MAX_CREATURE_TYPES + k;
            let g = genomes[parent * MAX_CREATURE_TYPES + k];
            let noise = random(hash(c * 97u + k * 13u + p.flags.y)) * 2.0 - 1.0;
            genomes[child_index] = vec2(
                clamp(g.x * (1.0 + p.evolution.y * noise), 1.0, 10.0),
                wrap_half(g.y + p.evolution.y * noise * TAU * 0.5),
            );
        }
        creatures[c].mutated = 1u;
        creatures[c].generation = creatures[parent].generation + 1u;
    }
}
@compute @workgroup_size(256)
fn count_free(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    if i >= p.counts.x || p.evolution.x == 0.0 { return; }
    if destination[i].pad1 == 0u { atomicAdd(&creatures[0].parent, 1u); }
}
@compute @workgroup_size(256)
fn cull_select(@builtin(global_invocation_id) id: vec3<u32>) {
    let c = id.x;
    if c == 0u || c >= MAX_CREATURES || p.detect.w == 0.0 || p.evolution.x == 0.0 { return; }
    if (creatures[c].flags & 1u) == 0u { return; }
    let free = atomicLoad(&creatures[0].parent);
    if f32(free) / max(1.0, f32(p.counts.x)) >= p.evolution.z { return; }
    // Durability is persistence times current share of the peak, so a body in
    // decline is weak and an expanding one is strong. Quantized durability in
    // the high bits and the id below give a deterministic atomic-min winner.
    let n = atomicLoad(&creatures[c].count);
    let share = f32(n) / max(1.0, f32(creatures[c].peak));
    let durability = f32(creatures[c].persistence) * share;
    let packed = (u32(min(durability, 1048575.0)) << 12u) | c;
    atomicMin(&creatures[0].cos_x, packed);
}
@compute @workgroup_size(256)
fn cull_release(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    if i >= p.counts.x || p.detect.w == 0.0 || p.evolution.x == 0.0 { return; }
    let target_full = atomicLoad(&creatures[0].cos_x);
    if target_full == 0xffffffffu { return; }
    let doomed = target_full & 0xfffu;
    if doomed == 0u || destination[i].pad1 != doomed { return; }
    // Dissolve the weakest body back into free material: unlinking the label is
    // not enough, the cluster would just reform under the same rules, so the
    // particles get a kick that breaks the body apart.
    destination[i].pad1 = 0u;
    let seed = hash(i * 31u + p.flags.y * 805306457u);
    let direction_angle = random(seed) * TAU;
    let momentum = p.world.z * (0.4 + 0.3 * random(seed ^ 0x9e3779b9u));
    destination[i].velocity = vec2(cos(direction_angle), sin(direction_angle)) * momentum;
}
// Reproduction: a tracked body stretched beyond `SPLIT_ASPECT` detaches its
// leading half as a new creature rather than thinning into free material. The
// pair of kernels keeps both halves labelled by the end of the same step, so the
// bond pass never has to re-split between two halves that share a label.
@compute @workgroup_size(256)
fn pick_split(@builtin(global_invocation_id) id: vec3<u32>) {
    let c = id.x;
    if c >= MAX_CREATURES { return; }
    // Clear the marker for every slot, not just the ones about to split, so a
    // stale marker from an earlier step can never send particles flying.
    births[2u * c] = 0u;
    if c == 0u || p.detect.w == 0.0 || p.evolution.x == 0.0 { return; }
    let n = atomicLoad(&creatures[c].count);
    if n == 0u || (creatures[c].flags & 1u) == 0u { return; }
    // Both halves must clear the viability bar or the birth is just a cull.
    if n < 2u * u32(p.detect.y) { return; }
    // Stagger births across the population: the split threshold carries a
    // per-creature offset, so neighbours do not all trip it in the same step and
    // the whole world does not change colour together.
    let threshold = SPLIT_ASPECT + random(hash(c * 2654435761u + p.flags.y)) * 1.5;
    let aspect = max(creatures[c].frame.y, 1.0);
    if aspect <= threshold { return; }
    births[2u * c] = c;
    let child = atomicAdd(&creatures[0].count, 1u) % (MAX_CREATURES - 1u) + 1u;
    births[2u * c + 1u] = child;
    // Seed the child sheet the way `bond` seeds a fresh id, but enough running
    // history to be tracked immediately: a newly split half has nothing to prove
    // before it can inherit its parent's appearance and genes.
    creatures[child].persistence = u32(p.detect.z);
    creatures[child].peak = 0u;
    creatures[child].flags = 0u;
    atomicStore(&creatures[child].lineage, c);
    creatures[child].mutated = 0u;
    creatures[child].generation = creatures[c].generation + 1u;
    creatures[child].frame = vec2(0.0);
}
@compute @workgroup_size(256)
fn apply_split(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    if i >= p.counts.x || p.detect.w == 0.0 || p.evolution.x == 0.0 { return; }
    let parent = destination[i].pad1;
    if parent == 0u || parent >= MAX_CREATURES { return; }
    if births[2u * parent] != parent { return; }
    let frame = creatures[parent].frame;
    let axis = vec2(cos(frame.x), sin(frame.x));
    let centroid = creatures[parent].centroid;
    var rel = destination[i].position - centroid;
    rel -= round(rel / p.world.x) * p.world.x;
    // Everything ahead of the centroid along the measured major axis leaves as
    // the child; the body behind keeps the parent's id.
    if dot(rel, axis) > 0.0 {
        destination[i].pad1 = births[2u * parent + 1u];
        // Open a real gap across the cut this step: two halves that stay within
        // each other's bond reach would fold back together and the re-merge
        // would hand out another generation, marching every body's colour up in
        // lockstep. A positional shift wider than the bond radius makes the
        // detachment permanent; the velocity is just the visual loft of a birth.
        let gap = p.detect.x * 1.5;
        destination[i].position += axis * gap;
        destination[i].velocity += axis * (0.5 * p.world.z);
    }
}
