struct Particle { position: vec2<f32>, velocity: vec2<f32>, kind: u32, pad0: u32, pad1: u32, pad2: u32 }
struct Params { counts: vec4<u32>, physics: vec4<f32>, world: vec4<f32>, view: vec4<f32>, timing: vec4<f32>, flags: vec4<u32>, trails: vec4<f32>, trail_view: vec4<f32>, behavior: vec4<f32>, cycle: vec4<f32>, sampling: vec4<f32>, detect: vec4<f32>, outline: vec4<f32>, evolution: vec4<f32> }
@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> current: array<Particle>;
// The cell-sorted pre-update copy: same particle order as `current`, so the two
// buffers stay index-aligned even though the simulation reorders every step.
@group(0) @binding(2) var<storage, read> previous: array<Particle>;
@group(0) @binding(3) var<storage, read> types: array<f32>;
struct Chemical {
    density: f32, next: f32, deposits: u32, pad: u32,
    pigment: vec4<f32>, next_pigment: vec4<f32>,
    red: u32, green: u32, blue: u32, color_pad: u32,
}
@group(0) @binding(4) var<storage, read> chemicals: array<Chemical>;
fn chemical_index(xy: vec2<i32>) -> u32 {
    let side = i32(p.trail_view.w);
    let wrapped = ((xy % side) + side) % side;
    return u32(wrapped.y * side + wrapped.x);
}
fn trail_sample(xy: vec2<i32>) -> vec4<f32> {
    let cell = chemicals[chemical_index(xy)];
    return vec4(cell.pigment.rgb, cell.density);
}
fn scent(position: vec2<f32>) -> vec4<f32> {
    let coord = position / p.world.x * p.trail_view.w - 0.5;
    let base = vec2<i32>(floor(coord)); let f = fract(coord);
    return mix(mix(trail_sample(base), trail_sample(base + vec2(1, 0)), f.x),
        mix(trail_sample(base + vec2(0, 1)), trail_sample(base + vec2(1, 1)), f.x), f.y);
}
struct VertexOut { @builtin(position) position: vec4<f32>, @location(0) uv: vec2<f32>, @location(1) color: vec3<f32> }
@vertex
fn vertex(@builtin(vertex_index) v: u32, @builtin(instance_index) instance: u32) -> VertexOut {
    let corners = array(vec2(-1.0,-1.0), vec2(1.0,-1.0), vec2(-1.0,1.0), vec2(-1.0,1.0), vec2(1.0,-1.0), vec2(1.0,1.0));
    let i = instance % p.counts.x;
    var out: VertexOut;
    out.uv = corners[v];
    out.color = vec3(0.0);
    let tile = instance / p.counts.x;
    let tile_offset = p.timing.zw + vec2<f32>(f32(tile % p.flags.w), f32(tile / p.flags.w));
    let a = previous[i]; let b = current[i];
    var delta = b.position - a.position;
    delta -= round(delta / p.world.x) * p.world.x;
    var pos = a.position + delta * p.timing.x;
    pos -= floor(pos / p.world.x) * p.world.x;
    let center = (pos / p.world.x * 2.0 - 1.0 + tile_offset * 2.0 - p.view.yz) * p.world.w;
    let offset = corners[v] * p.view.w * 2.0 / p.world.x * p.world.w;
    out.position = vec4((center + offset) / vec2(p.view.x, 1.0), 0.0, 1.0);
    let c = b.kind * 4u;
    var color = vec3(types[c], types[c+1u], types[c+2u]);
    out.color = color;
    return out;
}
@fragment
fn fragment(in: VertexOut) -> @location(0) vec4<f32> {
    let r2 = dot(in.uv, in.uv);
    let alpha = exp(-3.5 * r2) * (1.0 - smoothstep(0.7, 1.0, r2));
    return vec4(in.color * (1.0 + p.timing.y * 2.0) * alpha, alpha);
}


// One fullscreen triangle gives the toroidal world a continuous backdrop.
// Grid and color fields are world-anchored and periodic at the world boundary.
@vertex
fn background_vertex(@builtin(vertex_index) v: u32) -> VertexOut {
    let corners = array(vec2(-1.0, -1.0), vec2(3.0, -1.0), vec2(-1.0, 3.0));
    var out: VertexOut;
    out.position = vec4(corners[v], 0.0, 1.0);
    out.uv = corners[v];
    out.color = vec3(0.0);
    return out;
}
fn grid_lines(world: vec2<f32>, spacing: f32) -> f32 {
    let coord = world / spacing;
    let footprint = max(fwidth(coord), vec2(0.00001));
    let distance = abs(fract(coord - 0.5) - 0.5) / footprint;
    // Fade fine lines before they become subpixel and shimmer while zooming.
    let fade = 1.0 - smoothstep(0.08, 0.3, max(footprint.x, footprint.y));
    return (1.0 - smoothstep(0.35, 1.1, min(distance.x, distance.y))) * fade;
}
@fragment
fn background_fragment(in: VertexOut) -> @location(0) vec4<f32> {
    let world = (in.uv * vec2(p.view.x, 1.0) / p.world.w + p.view.yz + 1.0) * p.world.x * 0.5;
    let phase = world / p.world.x * 6.283185307;
    let haze = 0.5 + 0.5 * sin(phase.x + 0.55 * sin(phase.y)) * cos(phase.y);
    var color = mix(vec3(0.0025, 0.004, 0.010), vec3(0.005, 0.010, 0.018), haze);
    let minor = grid_lines(world, 64.0);
    let major = grid_lines(world, 256.0);
    color += vec3(0.003, 0.006, 0.008) * minor * 0.35;
    color += vec3(0.004, 0.007, 0.010) * major * 0.45;
    if p.trails.x > 0.0 && p.trail_view.z > 0.0 {
        let trail = scent(world);
        let ink = 1.0 - exp(-trail.a * 0.8);
        let hue = trail.rgb / max(trail.a, 0.000001);
        color += hue * 0.3 * ink * p.trail_view.z;
    }
    let vignette = 1.0 - 0.16 * smoothstep(0.25, 1.65, length(in.uv));
    return vec4(color * vignette, 1.0);
}


// ---- Compute splat path ---------------------------------------------------
// Below roughly one pixel per dot the instanced quad path is both wasteful and
// wrong: wasteful because every particle costs six vertices x 64 bytes of
// storage fetch, multiplied by the visible tile count; wrong because a quad
// smaller than a pixel produces no fragments at all unless it happens to cover
// a pixel center, so particles blink out as they drift. Splatting deposits each
// particle's total light directly, which is stable and bandwidth-cheap.
@group(0) @binding(5) var<storage, read_write> accumulation: array<atomic<u32>>;

// Integral of the fragment falloff over the quad's [-1,1]^2 domain. Scaling it
// by radius^2 gives the same total light the rasterizer deposits, so brightness
// is continuous across the switch between the two paths.
const FALLOFF_INTEGRAL: f32 = 0.8505024;
const SPLAT_SCALE: f32 = 4096.0;
const STAMP_REACH: i32 = 3;

fn falloff(r2: f32) -> f32 {
    if r2 >= 1.0 { return 0.0; }
    return exp(-3.5 * r2) * (1.0 - smoothstep(0.7, 1.0, r2));
}
fn deposit(xy: vec2<i32>, value: vec3<f32>, viewport: vec2<f32>) {
    if xy.x < 0 || xy.y < 0 || xy.x >= i32(viewport.x) || xy.y >= i32(viewport.y) { return; }
    let base = (u32(xy.y) * u32(viewport.x) + u32(xy.x)) * 3u;
    let q = vec3<u32>(max(value, vec3(0.0)) * SPLAT_SCALE);
    if q.x > 0u { atomicAdd(&accumulation[base], q.x); }
    if q.y > 0u { atomicAdd(&accumulation[base + 1u], q.y); }
    if q.z > 0u { atomicAdd(&accumulation[base + 2u], q.z); }
}
// Deposits `energy` in total, distributed over the covered pixels by the same
// falloff the fragment stage uses but renormalized, so no light is lost to the
// gaps between pixel centers.
fn stamp(pixel: vec2<f32>, radius: f32, energy: vec3<f32>, viewport: vec2<f32>) {
    let base = vec2<i32>(floor(pixel));
    if radius < 0.5 {
        // Sub-pixel: the whole dot lands inside one pixel. This is the 1M case,
        // and it costs a single atomic per channel.
        deposit(base, energy, viewport);
        return;
    }
    let reach = min(i32(ceil(radius)), STAMP_REACH);
    var total = 0.0;
    for (var dy = -reach; dy <= reach; dy++) {
        for (var dx = -reach; dx <= reach; dx++) {
            let offset = vec2<f32>(base + vec2(dx, dy)) + 0.5 - pixel;
            total += falloff(dot(offset, offset) / (radius * radius));
        }
    }
    if total <= 0.0 {
        deposit(base, energy, viewport);
        return;
    }
    for (var dy = -reach; dy <= reach; dy++) {
        for (var dx = -reach; dx <= reach; dx++) {
            let xy = base + vec2(dx, dy);
            let offset = vec2<f32>(xy) + 0.5 - pixel;
            let weight = falloff(dot(offset, offset) / (radius * radius));
            if weight > 0.0 { deposit(xy, energy * weight / total, viewport); }
        }
    }
}
@compute @workgroup_size(256)
fn splat(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    if i >= p.counts.x { return; }
    let a = previous[i]; let b = current[i];
    var delta = b.position - a.position;
    delta -= round(delta / p.world.x) * p.world.x;
    var pos = a.position + delta * p.timing.x;
    pos -= floor(pos / p.world.x) * p.world.x;
    let c = b.kind * 4u;
    let tint = vec3(types[c], types[c + 1u], types[c + 2u]) * (1.0 + p.timing.y * 2.0);
    let viewport = vec2(p.sampling.z, p.sampling.w);
    // Same half-extent the vertex stage builds, expressed in pixels: NDC y spans
    // 2.0 across `viewport.y` rows.
    let radius = max(p.view.w / p.world.x * p.world.w * viewport.y, 0.0001);
    let energy = max(tint, vec3(0.0)) * radius * radius * FALLOFF_INTEGRAL;
    // Toroidal tiling costs almost nothing here: a particle's copies are a few
    // rejected bounds compares instead of a full extra instance each.
    let tiles_y = u32(p.sampling.y);
    for (var ty = 0u; ty < tiles_y; ty++) {
        for (var tx = 0u; tx < p.flags.w; tx++) {
            let tile_offset = p.timing.zw + vec2(f32(tx), f32(ty));
            let center = (pos / p.world.x * 2.0 - 1.0 + tile_offset * 2.0 - p.view.yz) * p.world.w;
            let clip = center / vec2(p.view.x, 1.0);
            stamp((clip * vec2(0.5, -0.5) + 0.5) * viewport, radius, energy, viewport);
        }
    }
}
@vertex
fn resolve_vertex(@builtin(vertex_index) v: u32) -> VertexOut {
    let corners = array(vec2(-1.0, -1.0), vec2(3.0, -1.0), vec2(-1.0, 3.0));
    var out: VertexOut;
    out.position = vec4(corners[v], 0.0, 1.0);
    out.uv = corners[v];
    out.color = vec3(0.0);
    return out;
}
@fragment
fn resolve_fragment(in: VertexOut) -> @location(0) vec4<f32> {
    let xy = vec2<u32>(in.position.xy);
    let base = (xy.y * u32(p.sampling.z) + xy.x) * 3u;
    let rgb = vec3(f32(atomicLoad(&accumulation[base])),
        f32(atomicLoad(&accumulation[base + 1u])),
        f32(atomicLoad(&accumulation[base + 2u]))) / SPLAT_SCALE;
    // Alpha stays zero: the additive blend leaves the background's alpha intact.
    return vec4(rgb, 0.0);
}


// ---- Creature outlines ----------------------------------------------------
// A density isocontour rather than a hull or a covariance ellipse: it follows
// concave bodies, wraps across the world seam for free, and needs no per-creature
// geometry. Each particle of a tracked body stamps a normalized Gaussian into a
// world-space field; the outline is drawn where that field crosses the isolevel
// implied by the inferred rest spacing.
struct FieldCell { occupancy: atomic<u32>, owner: atomic<u32> }
@group(0) @binding(6) var<storage, read_write> creature_field: array<FieldCell>;
// Mirrors `Creature` in simulation.wgsl; read-only here, so no atomics. The
// tail fields -- per-type counts, frame accumulators, lineage, mutation flag
// and the measured frame -- pad each record to its real stride; the frame is
// read for the outline stamp and the generation for its colour.
struct CreatureView {
    count: u32, parent: u32,
    cos_x: u32, sin_x: u32, cos_y: u32, sin_y: u32,
    centroid: vec2<f32>,
    persistence: u32, peak: u32, flags: u32, pad0: u32,
    type_counts: array<u32, 16>,
    frame_xx: u32, frame_xy: u32, frame_yy: u32,
    lineage: u32, mutated: u32, generation: u32,
    frame: vec2<f32>,
}
@group(0) @binding(7) var<storage, read> creature_registry: array<CreatureView>;

const OCCUPANCY_SCALE: f32 = 4096.0;
const FIELD_REACH: i32 = 3;

fn field_index(xy: vec2<i32>) -> u32 {
    let side = i32(p.outline.w);
    let wrapped = ((xy % side) + side) % side;
    return u32(wrapped.y * side + wrapped.x);
}
fn field_occupancy(xy: vec2<i32>) -> f32 {
    return f32(atomicLoad(&creature_field[field_index(xy)].occupancy)) / OCCUPANCY_SCALE;
}
@compute @workgroup_size(256)
fn outline_splat(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    if i >= p.counts.x || p.outline.z <= 0.0 { return; }
    let creature = current[i].pad1;
    // Only bodies that have held together long enough to be promoted get drawn,
    // which is what keeps outlines from flickering on transient crowds.
    if creature == 0u || (creature_registry[creature].flags & 1u) == 0u { return; }
    let side = p.outline.w;
    let coord = current[i].position / p.world.x * side;
    let radius = max(p.outline.x / p.world.x * side, 0.35);
    // Stretch the stamp along the creature's measured major axis so the outline
    // tracks the same anisotropic frame the forces use. frame = (angle, aspect).
    let measured = creature_registry[creature].frame;
    let aspect = max(measured.y, 1.0);
    let axis = vec2(cos(measured.x), sin(measured.x));
    let rx = radius * aspect;
    let ry = radius / aspect;
    let base = vec2<i32>(floor(coord));
    let reach = min(i32(ceil(rx)), FIELD_REACH);
    // Normalize over the sampled texels so each particle contributes exactly one
    // unit of density however it lands between them; the field then reads
    // directly as particles per texel and the isolevel is scale free.
    var total = 0.0;
    for (var dy = -reach; dy <= reach; dy++) {
        for (var dx = -reach; dx <= reach; dx++) {
            let offset = vec2<f32>(base + vec2(dx, dy)) + 0.5 - coord;
            let px = dot(offset, axis);
            let py = dot(offset, vec2(-axis.y, axis.x));
            total += exp(-2.0 * (px * px / (rx * rx) + py * py / (ry * ry)));
        }
    }
    if total <= 0.0 { return; }
    let packed_id = creature & 0xfffu;
    for (var dy = -reach; dy <= reach; dy++) {
        for (var dx = -reach; dx <= reach; dx++) {
            let xy = base + vec2(dx, dy);
            let offset = vec2<f32>(xy) + 0.5 - coord;
            let px = dot(offset, axis);
            let py = dot(offset, vec2(-axis.y, axis.x));
            let weight = exp(-2.0 * (px * px / (rx * rx) + py * py / (ry * ry))) / total;
            if weight <= 0.0001 { continue; }
            let cell = field_index(xy);
            let quantized = u32(weight * OCCUPANCY_SCALE);
            atomicAdd(&creature_field[cell].occupancy, quantized);
            // Ownership goes to the strongest single contributor, so a texel on a
            // boundary takes the colour of the body that actually fills it.
            atomicMax(&creature_field[cell].owner, (quantized << 12u) | packed_id);
        }
    }
}
// Generation colours step a golden-ratio hue so consecutive generations never
// collide, siblings share a colour, and the mapping stays bounded even as the
// count of generations grows without limit.
fn creature_tint(generation: u32) -> vec3<f32> {
    let hue = fract(f32(generation) * 0.618033988749895) * 6.0;
    let sector = u32(hue);
    let f = hue - f32(sector);
    let p = 0.0;
    let q = 1.0 - f;
    let t = f;
    var rgb = vec3(1.0, t, p);
    switch sector {
        case 1u { rgb = vec3(q, 1.0, p); }
        case 2u { rgb = vec3(p, 1.0, t); }
        case 3u { rgb = vec3(p, q, 1.0); }
        case 4u { rgb = vec3(t, p, 1.0); }
        case 5u { rgb = vec3(1.0, p, q); }
        default {}
    }
    return rgb * 0.9;
}
@fragment
fn outline_fragment(in: VertexOut) -> @location(0) vec4<f32> {
    if p.outline.z <= 0.0 { return vec4(0.0); }
    let world = (in.uv * vec2(p.view.x, 1.0) / p.world.w + p.view.yz + 1.0) * p.world.x * 0.5;
    let coord = world / p.world.x * p.outline.w - 0.5;
    let base = vec2<i32>(floor(coord));
    let f = fract(coord);
    let density = mix(
        mix(field_occupancy(base), field_occupancy(base + vec2(1, 0)), f.x),
        mix(field_occupancy(base + vec2(0, 1)), field_occupancy(base + vec2(1, 1)), f.x),
        f.y);
    let signed = density - p.outline.y;
    // Constant pixel width at any zoom, the same screen-space derivative trick
    // `grid_lines` uses for the background grid.
    let footprint = max(fwidth(signed), 0.000001);
    let edge = 1.0 - smoothstep(0.0, 1.6, abs(signed) / footprint);
    if edge <= 0.0 { return vec4(0.0); }
    let owner = atomicLoad(&creature_field[field_index(vec2<i32>(round(coord)))].owner) & 0xfffu;
    let generation = creature_registry[owner].generation;
    return vec4(creature_tint(generation) * edge * p.outline.z, 0.0);
}
