# Particle Life

A native Bevy playground with GPU particle simulation, directional attraction,
swirl, velocity alignment, preferred spacing, crowding response, type cycles, and chemical trails.

Run from this directory with `cargo run --release`. The world and background tile
seamlessly as you pan and zoom. Scroll to zoom, drag the background to pan, Space
to pause, Right Arrow to step, R to reset positions, Tab on the background to hide/show controls, and F11
for fullscreen.

## Chemical trails

Open **Chemical trails** and check **Enable chemical trails**. All particle types
deposit into one shared 512 × 512 chemical field:

- **Follow / avoid**: positive follows increasing concentration; negative avoids
  it. Zero keeps deposition and visualization without affecting motion.
- **Deposit rate**: chemical left per particle per simulation second. Zero lets
  existing trails fade without adding more.
- **Half-life**: seconds of simulation time for concentration to halve.
- **Diffusion**: how quickly scent spreads into neighboring cells.
- **Sensing distance**: how far particles sample to choose a direction.
- **Trail visibility**: brightness of the particle-colored overlay. Zero hides the field
  without turning off its effect on motion.
- **Clear trails**: removes the field without resetting particles, even paused.

Use the Enable chemical trails checkbox to switch trails on or off. Enabling/disabling or resetting the world clears
old trails. Pause freezes deposition, diffusion, and decay; stepping advances all
three with the particles. Trail following works alongside the interaction rules.
Deposits inherit the particle color at deposition time; overlapping colors mix,
and old colors fade naturally after palette changes.
The field and sensing wrap at the world edges. Chemical concentration is capped
at 100 to keep dense deposits bounded. Trail updates stay on the GPU and add
about 16 MiB of storage, independent of particle/type count.

Try the default trail settings first. Change Follow / avoid to a negative value
for particles that move away from their previous paths, or increase the half-life
to leave a longer memory of motion.

## Checks

`cargo test` checks settings, data layout, wrapping, and WGSL validation.
`cargo test -- --ignored` additionally runs a Vulkan GPU regression that verifies
chemical diffusion across both seams, decay, one-time deposition, and clearing.

## Spacing, density & type cycles

All three new behaviors are initially off. Open **Spacing, density & type cycles**:

- **Preferred pair distances** enables the **Distance** matrix under Interaction
  rules. Values are fractions of the interaction radius: 0.5 means half the radius.
  Nonzero pairs replace their original attraction curve with repulsion below the
  preferred distance and attraction above it. Zero restores that pair's original
  rule. The short-range core remains repulsive; distances below 0.2 are treated as
  0.2. Randomize selected creates distances between 0.25 and 0.8. Other forces can
  shift the final equilibrium.
- **Density-dependent attraction / repulsion** adds a radial response based on
  neighbors inside the interaction radius, excluding the particle itself. Below
  Target neighbors it attracts, above it repels, and at the target it adds no force.
  Crowding strength controls how strongly this supplements the other rules.
  Fast mode estimates crowding and successor fractions from its neighbor sample;
  Exact mode checks every nearby particle.
- **Type cycling** follows 1 → 2 → … → 1. Timer advances each particle after its
  interval, with seeded staggered initial phases. Neighbors advances when enough
  nearby particles belong to the next type, after the configured cooldown. Either
  permits that neighbor trigger after 0.5 seconds or advances when the full timer
  expires. Minimum neighbors and Next-type fraction control the neighbor trigger.
  A single-type world stays unchanged. Pausing freezes timers; stepping advances
  them. Resets restore initial types and seeded phases. Changing the interval
  applies live to existing particle ages.

Type changes retain position and velocity. Particle color and new trail deposits
use the new type; already deposited trails keep their historical colors and fade.
Count regeneration resets distance entries to 0.5 and swirl/alignment to zero;
behavior switches and other settings remain as selected.

The opt-in GPU tests also verify spacing equilibrium, dense/sparse responses,
cycle thresholds, cooldowns, and last-to-first type transitions.

## Feathers controls

The controls use Bevy Feathers’ dark pane and subpane widgets in a reserved left
sidebar. The joined tabs select **World** (counts and motion), **Appearance**
(color, light, and chemical trails), or **Rules** (behavior and interaction rules).
Playback, speed, and simulation status stay above the tabs. Scroll within a tab
or drag its scrollbar to reach the remaining settings.

The rule matrix scrolls in both directions; use Shift/Ctrl + mouse wheel or its
horizontal scrollbar to reach additional columns. Click a cell to edit its value.
Only visible cells are created, including for large type counts.

Click the simulation background to focus simulation shortcuts. Tab then hides or
shows the sidebar; inside controls, Tab navigates widgets instead. Hide and Show
controls buttons also work with the mouse. The simulation fills the space freed
when controls are hidden. Numeric fields accept typed integers, including the full
32-bit unsigned seed range; invalid count input blocks regeneration.
