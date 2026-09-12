//! Infers what creatures the current rule set should be able to hold together.
//!
//! Nothing here reads the simulation. It reads the rule matrices and derives the
//! geometry a stable body would have to have, which gives the detector its
//! thresholds (bond radius, minimum size) and the outline its stamp radius and
//! isosurface level. Treat the sizes as priors that scale the detector, not as a
//! forecast of what the simulation will actually produce.

use crate::model::*;

/// A set of types the rules let cohere, plus the geometry implied by that set.
#[derive(Clone, Debug, PartialEq)]
pub struct CreatureTemplate {
    /// Types that bond into one body, ascending.
    pub types: Vec<u32>,
    /// Rest separation of a bonded pair, in world units.
    pub spacing: f32,
    /// Characteristic body radius, in world units.
    pub expected_radius: f32,
    /// Per-type minimum: every member type of the body must hold at least this
    /// many particles or the cluster is treated as debris rather than a creature.
    pub min_per_type: u32,
}

/// Mutual attraction below this is treated as no bond. Rule values span [-1, 1]
/// and a pair contributes both directions, so this is a deliberately low bar: the
/// detector would rather over-group and be filtered by size than miss a body.
const BOND_THRESHOLD: f32 = 0.15;

/// Every matrix is directional — row is the affected type, column its neighbor —
/// so a bond needs both directions to pull. A pair where one side attracts and the
/// other flees is a chase relationship: it produces motion, not a body, and
/// summing the two directions is what rejects it.
pub fn bond_strength(sim: &Simulation, a: u32, b: u32) -> f32 {
    let t = sim.types as usize;
    let (a, b) = (a as usize, b as usize);
    sim.rules[a * t + b] + sim.rules[b * t + a]
}

/// Whether a pair holds together at rest.
///
/// Preferred distances override the attraction curve entirely: `preferred_force`
/// is attractive above its zero whatever the attraction coefficient says, so an
/// active preferred distance bonds the pair on its own.
pub fn bonds(sim: &Simulation, a: u32, b: u32) -> bool {
    if preferred_distance(sim, a, b).is_some() {
        return true;
    }
    bond_strength(sim, a, b) > BOND_THRESHOLD
}

/// The preferred separation for a pair as a fraction of the interaction radius,
/// if preferred distances are enabled and either direction sets one.
///
/// The two directions can disagree, in which case no single separation satisfies
/// both and the body settles between them; the mean is the best cheap estimate.
fn preferred_distance(sim: &Simulation, a: u32, b: u32) -> Option<f32> {
    if !sim.behavior.preferred_enabled {
        return None;
    }
    let t = sim.types as usize;
    let (a, b) = (a as usize, b as usize);
    // `preferred_force` clamps its argument the same way; mirror it so the
    // inferred zero matches the one the shader actually produces.
    let clamped: Vec<f32> = [sim.distance[a * t + b], sim.distance[b * t + a]]
        .iter()
        .filter(|d| **d > 0.0)
        .map(|d| d.clamp(0.2, 0.95))
        .collect();
    if clamped.is_empty() {
        return None;
    }
    Some(clamped.iter().sum::<f32>() / clamped.len() as f32)
}

/// Rest separation of a bonded pair, in world units.
///
/// `force()` is repulsive below `CORE_FRACTION` and attractive above it, and
/// crosses zero exactly at the boundary, so an attraction-only pair rests there.
/// This is exact rather than a numerical solve. An active preferred distance
/// moves the zero to its own separation.
pub fn pair_spacing(sim: &Simulation, a: u32, b: u32) -> f32 {
    preferred_distance(sim, a, b).unwrap_or(CORE_FRACTION) * sim.radius
}

/// Type sets the bond graph connects, as ascending lists.
pub fn compositions(sim: &Simulation) -> Vec<Vec<u32>> {
    let t = sim.types;
    let mut parent: Vec<u32> = (0..t).collect();
    fn root(parent: &mut [u32], mut x: u32) -> u32 {
        while parent[x as usize] != x {
            parent[x as usize] = parent[parent[x as usize] as usize];
            x = parent[x as usize];
        }
        x
    }
    for a in 0..t {
        for b in a..t {
            if bonds(sim, a, b) {
                let (ra, rb) = (root(&mut parent, a), root(&mut parent, b));
                if ra != rb {
                    parent[rb.max(ra) as usize] = rb.min(ra);
                }
            }
        }
    }
    let mut groups: Vec<Vec<u32>> = vec![Vec::new(); t as usize];
    for a in 0..t {
        let r = root(&mut parent, a);
        groups[r as usize].push(a);
    }
    groups.retain(|g| !g.is_empty());
    groups
}

/// Templates for every composition the current rules support.
pub fn templates(sim: &Simulation) -> Vec<CreatureTemplate> {
    compositions(sim)
        .into_iter()
        .map(|types| {
            // Average over the bonded pairs inside the group, self-pairs included:
            // a single cohesive type is the most common body there is.
            let mut total = 0.0;
            let mut pairs = 0.0;
            for (i, a) in types.iter().enumerate() {
                for b in &types[i..] {
                    if bonds(sim, *a, *b) {
                        total += pair_spacing(sim, *a, *b);
                        pairs += 1.0;
                    }
                }
            }
            let spacing = if pairs > 0.0 {
                total / pairs
            } else {
                CORE_FRACTION * sim.radius
            };
            // Particles cannot feel each other past the interaction radius, so a
            // body cannot stay cohesive much wider than that; it is the natural
            // scale for a saturated blob.
            let expected_radius = sim.radius;
            // Hexagonal packing puts 2*pi/sqrt(3) particles per unit area in units
            // of the spacing squared.
            let packed = 3.6276 * (expected_radius / spacing.max(0.0001)).powi(2);
            let members = types.len().max(1) as f32;
            CreatureTemplate {
                types,
                spacing,
                expected_radius,
                // A saturated body of k types carries roughly k equal shares, so a
                // quarter-saturated body leaves each type `packed * 0.25 / k`
                // particles. The body counts only if every member type clears
                // that bar -- a mostly-one-kind blob with a sprinkle of another
                // is an accidental crowd, not the mixed body the rules call for.
                // Never trust the estimate down past a handful: floor the share
                // at four.
                min_per_type: (packed * 0.25 / members).max(4.0) as u32,
            }
        })
        .collect()
}

/// Rest separation averaged over every inferred template.
pub fn mean_spacing(sim: &Simulation) -> f32 {
    let templates = templates(sim);
    if templates.is_empty() {
        return CORE_FRACTION * sim.radius;
    }
    templates.iter().map(|t| t.spacing).sum::<f32>() / templates.len() as f32
}

/// Distance within which the detector treats two particles as bonded. Wider than
/// the rest separation so a body under strain stays one cluster, but far short of
/// the interaction radius so distinct bodies do not merge.
pub fn bond_radius(sim: &Simulation) -> f32 {
    (mean_spacing(sim) * 1.5).min(sim.radius)
}

/// Stamp radius (world units) and isolevel (particles per texel) for the outline
/// field. Both follow from the inferred rest spacing, which is what ties the
/// drawn outline back to the rule set rather than to a hand-tuned constant.
///
/// A body at rest packs one particle per `spacing^2` of area, so a texel holds
/// `texel^2 / spacing^2` of them; half that is the natural edge of the body.
pub fn outline_geometry(sim: &Simulation, field_side: u32) -> (f32, f32) {
    let spacing = mean_spacing(sim).max(0.0001);
    let texel = WORLD / field_side as f32;
    let interior = (texel * texel) / (spacing * spacing);
    (spacing * 2.0, interior * 0.5)
}

/// Smallest per-type count any inferred template accepts. The detector applies a
/// single threshold to each type present in a cluster, so the most permissive
/// template sets it.
pub fn min_particles(sim: &Simulation) -> u32 {
    templates(sim)
        .iter()
        .map(|t| t.min_per_type)
        .min()
        .unwrap_or(4)
        .max(4)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    /// Types, attraction matrix; everything else default.
    fn sim(types: u32, rules: Vec<f32>) -> Simulation {
        Simulation {
            types,
            rules: Arc::new(rules),
            swirl: Arc::new(vec![0.0; (types * types) as usize]),
            alignment: Arc::new(vec![0.0; (types * types) as usize]),
            distance: Arc::new(vec![0.0; (types * types) as usize]),
            ..Simulation::default()
        }
    }

    #[test]
    fn one_sided_attraction_is_a_chase_not_a_bond() {
        // 0 chases 1, 1 flees 0. Summing the directions cancels them out.
        let s = sim(2, vec![0.0, 1.0, -1.0, 0.0]);
        assert!(!bonds(&s, 0, 1), "a chase must not read as a body");
        assert_eq!(compositions(&s).len(), 2, "chase leaves two compositions");
        // Mutual attraction on the same pair does bond.
        let s = sim(2, vec![0.0, 1.0, 1.0, 0.0]);
        assert!(bonds(&s, 0, 1));
        assert_eq!(compositions(&s), vec![vec![0, 1]]);
    }

    #[test]
    fn compositions_group_transitively_and_cover_every_type() {
        // 0-1 bond, 2 is isolated, 3 is self-cohesive only.
        let mut rules = vec![0.0; 16];
        rules[1] = 1.0;
        rules[4] = 1.0; // 0 <-> 1
        rules[15] = 1.0; // 3 self
        let s = sim(4, rules);
        let groups = compositions(&s);
        assert_eq!(groups, vec![vec![0, 1], vec![2], vec![3]]);
        let covered: usize = groups.iter().map(Vec::len).sum();
        assert_eq!(covered, 4, "every type lands in exactly one composition");
    }

    /// The shader's force curve, so the inferred rest separation is checked
    /// against the real thing rather than against a restatement of itself.
    fn shader_force(r: f32, coefficient: f32) -> f32 {
        if r < CORE_FRACTION {
            return r / CORE_FRACTION - 1.0;
        }
        coefficient * (1.0 - (2.0 * r - 1.0 - CORE_FRACTION).abs() / (1.0 - CORE_FRACTION))
    }
    fn shader_preferred_force(r: f32, separation: f32) -> f32 {
        let d = separation.clamp(0.2, 0.95);
        if r < d {
            return (r - d) / d;
        }
        4.0 * (r - d) * (1.0 - r) / ((1.0 - d) * (1.0 - d))
    }

    #[test]
    fn inferred_spacing_sits_on_the_force_curve_zero() {
        let s = sim(2, vec![1.0; 4]);
        let spacing = pair_spacing(&s, 0, 1);
        let r = spacing / s.radius;
        assert!(
            shader_force(r, 1.0).abs() < 1e-6,
            "attraction-only rest separation must be a zero of force()"
        );
        // Stable, not just stationary: repel below, attract above.
        assert!(shader_force(r - 0.05, 1.0) < 0.0);
        assert!(shader_force(r + 0.05, 1.0) > 0.0);
    }

    #[test]
    fn preferred_distance_moves_the_zero_and_overrides_attraction() {
        let mut s = sim(2, vec![-1.0; 4]); // every pair repels
        s.behavior.preferred_enabled = true;
        s.distance = Arc::new(vec![0.6; 4]);
        assert!(
            bonds(&s, 0, 1),
            "preferred distance bonds despite repulsive attraction"
        );
        let r = pair_spacing(&s, 0, 1) / s.radius;
        assert!((r - 0.6).abs() < 1e-6);
        assert!(shader_preferred_force(r, 0.6).abs() < 1e-6);
        // Disabling the behavior falls back to the core fraction.
        s.behavior.preferred_enabled = false;
        assert!((pair_spacing(&s, 0, 1) / s.radius - CORE_FRACTION).abs() < 1e-6);
    }

    #[test]
    fn disagreeing_directions_settle_between_them() {
        let mut s = sim(2, vec![0.0; 4]);
        s.behavior.preferred_enabled = true;
        let mut d = vec![0.0; 4];
        d[1] = 0.4; // 0 wants 0.4 from 1
        d[2] = 0.8; // 1 wants 0.8 from 0
        s.distance = Arc::new(d);
        let r = pair_spacing(&s, 0, 1) / s.radius;
        assert!((r - 0.6).abs() < 1e-6, "mean of the two demands");
    }

    #[test]
    fn templates_scale_with_radius_and_stay_sane() {
        let s = sim(2, vec![1.0; 4]);
        let t = &templates(&s)[0];
        assert_eq!(t.types, vec![0, 1]);
        assert!((t.spacing - CORE_FRACTION * s.radius).abs() < 1e-6);
        assert!(t.min_per_type >= 4);
        // Doubling the radius doubles spacing but leaves the packing ratio alone,
        // so the size threshold is scale free.
        let mut wide = s.clone();
        wide.radius = s.radius * 2.0;
        let w = &templates(&wide)[0];
        assert!((w.spacing - 2.0 * t.spacing).abs() < 1e-4);
        assert_eq!(w.min_per_type, t.min_per_type);
    }

    #[test]
    fn per_type_minimum_splits_across_types_and_floors_at_four() {
        // A single cohesive type carries the whole body-size estimate, so it is
        // gated on real mass rather than a token handful.
        let s = sim(1, vec![1.0]);
        let t = &templates(&s)[0];
        assert_eq!(t.types, vec![0]);
        assert!(t.min_per_type > 4, "single-type bodies need a real mass");
        // Eight mutually bonded types split the same estimate eight ways, which
        // collapses the share to the four-particle floor.
        let s = sim(8, vec![1.0; 64]);
        let t = &templates(&s)[0];
        assert_eq!(t.types.len(), 8);
        assert_eq!(t.min_per_type, 4, "every member type needs at least four");
        // The detector threshold follows the same most-permissive, floored rule.
        assert!(min_particles(&s) >= 4);
    }

    #[test]
    fn bond_radius_sits_between_the_spacing_and_the_interaction_radius() {
        for preferred in [false, true] {
            let mut s = sim(3, vec![1.0; 9]);
            s.behavior.preferred_enabled = preferred;
            s.distance = Arc::new(vec![0.9; 9]);
            let r = bond_radius(&s);
            let spacing = templates(&s)[0].spacing;
            assert!(r >= spacing, "must not split a body at rest");
            assert!(r <= s.radius, "must not merge bodies that never interact");
        }
    }
}
