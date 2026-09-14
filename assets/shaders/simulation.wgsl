struct Particle { position: vec2<f32>, velocity: vec2<f32>, kind: u32, age: f32, pad1: u32, pad2: u32 }
// Compact per-neighbor record consumed by the force loop: 16 bytes, one aligned
// vec4 load, against 4 + 32 bytes for the old indices -> source indirection.
// Velocity is two f16s; speeds are capped at world.z (180), where f16 spacing is
// about 0.1 and the value only feeds soft alignment steering.
struct Neighbor { position: vec2<f32>, velocity: u32, kind: u32 }
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
    brush: vec4<f32>, // world center, radius, signed acceleration
    interaction: vec4<f32>, // preview mode (attract 1, repel -1, dump 2), unused
    nova: vec4<f32>, // ignition threshold (0 disables), radius fraction, impulse, burn seconds
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
    return Neighbor(particle.position, pack2x16float(particle.velocity), particle.kind);
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
// Below this share of the nova radius a particle's neighbors surround it evenly
// enough that no escape direction exists: it is the star's core, and it reseeds.
const NOVA_CORE_FRACTION: f32 = 0.12;
var<workgroup> rules: array<f32, 1024>;
fn rule(index: u32, cached: bool) -> f32 {
    if cached { return rules[index]; }
    return types[p.counts.y * 4u + index];
}
// Every neighbor uses the affected particle's current local rules.
struct Response {
    acceleration: vec2<f32>,
    density: vec2<f32>,
    alignment: vec2<f32>,
    // Summed wrapped offsets to same-type neighbors inside the nova radius. The
    // mean points at the local same-type centroid, so its length grows with
    // distance from the center: that is what separates a knot's shell from its core.
    kin_offset: vec2<f32>,
    weight: f32,
    nearby: f32,
    successors: f32,
    kin: f32,
}
fn empty_response() -> Response {
    return Response(vec2(0.0), vec2(0.0), vec2(0.0), vec2(0.0), 0.0, 0.0, 0.0, 0.0);
}
fn merge(x: Response, y: Response) -> Response {
    return Response(x.acceleration + y.acceleration, x.density + y.density,
        x.alignment + y.alignment, x.kin_offset + y.kin_offset, x.weight + y.weight,
        x.nearby + y.nearby, x.successors + y.successors, x.kin + y.kin);
}
fn interact(a: Particle, b: Neighbor, row: u32, successor: u32, cached: bool) -> Response {
    var out = empty_response();
    var delta = b.position - a.position;
    delta -= round(delta / p.world.x) * p.world.x;
    let euclid = length(delta);
    let b_kind = b.kind;
    let distance = euclid;
    if distance >= p.physics.y { return out; }
    out.nearby = 1.0;
    if b_kind == successor { out.successors = 1.0; }
    // Counted ahead of the core early-return below, which fires for exactly the
    // tightly packed pairs a nova cares about.
    if b_kind == a.kind && distance < p.physics.y * p.nova.y {
        out.kin = 1.0;
        out.kin_offset = delta;
    }
    if euclid <= 0.00001 { return out; }
    let matrix_size = p.counts.y * p.counts.y;
    let pair = row + b_kind;
    let coefficient = rule(pair, cached);
    let swirl = rule(pair + matrix_size, cached);
    let align = rule(pair + 2u * matrix_size, cached);
    let preferred = rule(pair + 3u * matrix_size, cached);
    let direction = delta / euclid;
    // Short-range repulsion applies regardless of the pair rule.
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
        foreign = merge(foreign, interact(a, neighbors[slot], row, successor, cached));
    }
    let sample_scale = f32(total) / max(1.0, f32(budget));
    let nearby = foreign.nearby;
    let successors = foreign.successors;
    let estimated_neighbors = nearby * sample_scale;
    let crowding = clamp(1.0 - estimated_neighbors / max(1.0, p.behavior.z), -1.0, 1.0);
    // A knot of one type dense enough to pass the threshold ignites: the shell
    // latches an outward direction and burns for nova.w seconds, while the core
    // is reseeded at random, returning its particles to the wider world.
    var burn = bitcast<f32>(a.pad1);
    var escape = unpack2x16float(a.pad2);
    var reseed = false;
    if p.nova.x > 0.0 && burn <= 0.0 && foreign.kin * sample_scale >= p.nova.x {
        let mean = foreign.kin_offset / max(1.0, foreign.kin);
        let reach = length(mean);
        if reach < NOVA_CORE_FRACTION * p.physics.y * p.nova.y {
            reseed = true;
        } else {
            escape = -mean / reach;
            burn = p.nova.w;
        }
    }
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
    if p.brush.w != 0.0 {
        var toward = p.brush.xy - a.position;
        toward -= round(toward / p.world.x) * p.world.x;
        let distance = length(toward);
        if distance > 0.00001 && distance < p.brush.z {
            let falloff = 1.0 - distance / p.brush.z;
            velocity += toward / distance * p.brush.w * falloff * falloff * p.physics.x;
        }
    }
    if p.nova.x > 0.0 && burn > 0.0 {
        velocity += escape * p.nova.z * p.physics.x;
        burn = max(0.0, burn - p.physics.x);
    }
    let speed = length(velocity);
    // A burning particle outruns the normal limit by exactly what its own impulse
    // can deliver. It crosses more than one grid cell per step and so interacts
    // sparsely while in flight, which is the point of an ejection.
    let speed_cap = select(p.world.z, p.world.z + p.nova.z * p.nova.w, burn > 0.0);
    if speed > speed_cap { velocity *= speed_cap / speed; }
    var pos = a.position + velocity * p.physics.x;
    pos -= floor(pos / p.world.x) * p.world.x;
    if reseed {
        let salt = i * 7u + p.counts.w + p.flags.y * 2654435761u;
        pos = vec2(random(salt), random(salt + 1u)) * p.world.x;
        velocity = vec2(0.0);
        burn = 0.0;
        escape = vec2(0.0);
    }
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
    destination[i] = Particle(pos, velocity, kind, age, bitcast<u32>(burn), pack2x16float(escape));
    // Without this a reseeded particle renders as a one-frame streak, because the
    // display interpolates from sorted[] toward destination[]. Only invocation i
    // reads sorted[i]; the force loop reads neighbors[]. So this write is race-free.
    if reseed { sorted[i].position = pos; }
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
