use crate::model::*;
use bevy::scene::prelude::*;
use bevy::text::{EditableText, FontSourceTemplate, TextEditChange};
use bevy::{
    feathers::{
        FeathersPlugins,
        constants::fonts,
        containers::*,
        controls::*,
        dark_theme::create_dark_theme,
        font_styles::InheritableFont,
        rounded_corners::RoundedCorners,
        theme::{ThemedText, UiTheme},
    },
    input::keyboard::KeyboardInput,
    input::mouse::{AccumulatedMouseMotion, MouseScrollUnit, MouseWheel},
    input_focus::{FocusCause, FocusedInput, InputFocus, tab_navigation::TabGroup},
    picking::hover::HoverMap,
    prelude::*,
    ui::{Checked, InteractionDisabled},
    ui_widgets::{Activate, SliderPrecision, SliderRange, SliderValue, ValueChange},
    window::{MonitorSelection, PrimaryWindow, WindowMode},
};

pub struct ControlsPlugin;
#[derive(SystemSet, Debug, Clone, PartialEq, Eq, Hash)]
pub struct UiSet;
impl Plugin for ControlsPlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins(FeathersPlugins)
            .insert_resource(UiTheme(create_dark_theme()))
            .init_resource::<Controls>()
            .add_systems(Startup, setup)
            .add_systems(Update, (input, sync, matrix).chain().in_set(UiSet))
            .add_observer(activate)
            .add_observer(float_change)
            .add_observer(integer_change)
            .add_observer(bool_change)
            .add_observer(scroll)
            .add_observer(validate_integer_text);
    }
}
#[derive(Resource)]
pub struct Controls {
    count: u32,
    types: u32,
    selected: (u32, u32),
    rule_kind: RuleKind,
    pub visible: bool,
    tab: usize,
    error: String,
    fps: f32,
    dragging: bool,
    invalid: [bool; 3],
}
impl Default for Controls {
    fn default() -> Self {
        Self {
            count: std::env::var("PL_COUNT").ok().and_then(|v| v.parse().ok()).unwrap_or(100_000),
            types: 8,
            selected: (0, 0),
            rule_kind: RuleKind::Attraction,
            visible: true,
            tab: 0,
            error: String::new(),
            fps: 60.,
            dragging: false,
            invalid: [false; 3],
        }
    }
}
/// Physical pixel bounds shared by rendering and pointer conversion.
pub fn viewport_geometry(size: UVec2, scale: f32, visible: bool) -> (UVec2, UVec2) {
    let size = size.max(UVec2::ONE);
    let left = if visible {
        (360. * scale).round().max(0.) as u32
    } else {
        0
    };
    let left = left.min(size.x / 2);
    (UVec2::new(left, 0), UVec2::new(size.x - left, size.y))
}
#[derive(Component)]
struct BackgroundFocus;
#[derive(Component)]
struct Sidebar;
#[derive(Component)]
struct ShowButton;
#[derive(Component)]
struct TabBody(usize);
#[derive(Component, Clone, Copy)]
enum Action {
    Tab(usize),
    Hide,
    Show,
    Play,
    Step,
    Reset,
    Apply,
    ResetView,
    ClearTrails,
    Randomize,
    Zero,
    Universe,
    Kind(RuleKind),
    Palette(usize),
    Cycle(u32),
    Exact(bool),
    Cell(u32, u32),
}
#[derive(Component, Clone, Copy)]
enum Integer {
    Count,
    Types,
    Seed,
}
#[derive(Component, Clone, Copy)]
enum Toggle {
    Clustered,
    Preferred,
    Density,
    Trails,
}
#[derive(Component, Clone, Copy)]
enum Info {
    Status,
    Adapter,
    Message,
    Error,
    Active,
    Rule,
    Selection,
    Hover,
    Palette,
    Cycle,
    Exact,
}
#[derive(Component)]
struct Swatch(u32);
#[derive(Component)]
struct MatrixViewport;
#[derive(Component)]
struct MatrixContent;
#[derive(Component)]
struct MatrixCell(u32, u32);
#[derive(Component, Clone, Copy)]
enum Gate {
    Density,
    Cycle,
    Neighbors,
    Trails,
    Step,
}

fn text(commands: &mut Commands, parent: Entity, value: impl Into<String>) -> Entity {
    let value = value.into();
    commands.spawn_scene(bsn! { Text(value) ThemedText TextFont { font: FontSourceTemplate::Handle(fonts::REGULAR), font_size: bevy::text::FontSize::Px(13.0) } }).insert(ChildOf(parent)).id()
}
fn info(commands: &mut Commands, parent: Entity, kind: Info) {
    let id = text(commands, parent, "");
    commands.entity(id).insert(kind);
    if matches!(kind, Info::Error | Info::Message) {
        commands.entity(id).remove::<ThemedText>().insert(TextColor(
            if matches!(kind, Info::Error) {
                Color::srgb(1., 0.45, 0.4)
            } else {
                Color::srgb(1., 0.8, 0.35)
            },
        ));
    }
}
fn row(commands: &mut Commands, parent: Entity) -> Entity {
    commands
        .spawn((
            Node {
                flex_direction: FlexDirection::Row,
                flex_wrap: FlexWrap::Wrap,
                column_gap: px(3),
                row_gap: px(3),
                ..default()
            },
            ChildOf(parent),
        ))
        .id()
}
fn button(
    commands: &mut Commands,
    parent: Entity,
    caption: &'static str,
    action: Action,
    corners: RoundedCorners,
) -> Entity {
    let mut entity = commands.spawn_scene(bsn! {
        @FeathersButton { @caption: bsn! { Text(caption) ThemedText }, @corners: corners }
    });
    entity.insert(action);
    if parent != Entity::PLACEHOLDER {
        entity.insert(ChildOf(parent));
    }
    entity.id()
}
fn sub(commands: &mut Commands, parent: Entity, title: &'static str) -> Entity {
    let root = commands
        .spawn_scene(subpane())
        .insert((
            ChildOf(parent),
            Node {
                flex_direction: FlexDirection::Column,
                flex_shrink: 0.,
                ..default()
            },
        ))
        .id();
    let header = commands
        .spawn_scene(subpane_header())
        .insert(ChildOf(root))
        .id();
    text(commands, header, title);
    commands
        .spawn_scene(subpane_body())
        .insert(ChildOf(root))
        .id()
}
fn check(commands: &mut Commands, parent: Entity, caption: &'static str, kind: Toggle) {
    commands
        .spawn_scene(bsn! { @FeathersCheckbox { @caption: bsn! { Text(caption) ThemedText } } })
        .insert((ChildOf(parent), kind));
}
fn integer(commands: &mut Commands, parent: Entity, caption: &'static str, kind: Integer) {
    text(commands, parent, caption);
    commands
        .spawn_scene(bsn! { @FeathersNumberInput { @number_format: NumberFormat::I64 } })
        .insert((ChildOf(parent), kind));
}
fn slider(commands: &mut Commands, parent: Entity, field: Field) -> Entity {
    text(commands, parent, field.label());
    let (min, max) = field.range(RuleKind::Attraction);
    let id = commands
        .spawn_scene(bsn! { @FeathersSlider { @min: min, @max: max } Node { flex_grow: 0.0, flex_shrink: 0.0 } })
        .insert((
            ChildOf(parent),
            field,
            SliderPrecision(if matches!(field, Field::Rule) {
                3
            } else if matches!(field, Field::CycleMin) {
                0
            } else {
                2
            }),
        ))
        .id();
    if let Some(gate) = field.gate() {
        commands.entity(id).insert(gate);
    }
    id
}
fn menu(commands: &mut Commands, parent: Entity, kind: Info, choices: Vec<(&'static str, Action)>) {
    let root = commands
        .spawn_scene(bsn! { @FeathersMenu })
        .insert(ChildOf(parent))
        .id();
    let label = commands
        .spawn_scene(bsn! { @FeathersMenuButton })
        .insert(ChildOf(root))
        .id();
    info(commands, label, kind);
    let popup = commands
        .spawn_scene(bsn! { @FeathersMenuPopup })
        .insert(ChildOf(root))
        .id();
    for (caption, action) in choices {
        commands
            .spawn_scene(bsn! { @FeathersMenuItem { @caption: bsn! { Text(caption) ThemedText } } })
            .insert((ChildOf(popup), action));
    }
}
fn setup(mut commands: Commands, mut focus: ResMut<InputFocus>) {
    let background = commands.spawn(BackgroundFocus).observe(shortcut).id();
    focus.set(background, FocusCause::Navigated);
    let root=commands.spawn_scene(bsn! { pane() InheritableFont { font: fonts::REGULAR, font_size: bevy::text::FontSize::Px(13.0) } }).insert((Sidebar, TabGroup::default(), Node {
        position_type: PositionType::Absolute, left:px(0), top:px(0), width:px(360), height:percent(100),
        flex_direction:FlexDirection::Column, padding:UiRect::all(px(6)), row_gap:px(5), overflow:Overflow::clip(), ..default()
    }, BackgroundColor(Color::srgb(0.035,0.04,0.05)))).id();
    let header = commands
        .spawn_scene(pane_header())
        .insert(ChildOf(root))
        .id();
    text(&mut commands, header, "PARTICLE LIFE");
    button(
        &mut commands,
        header,
        "Hide",
        Action::Hide,
        RoundedCorners::All,
    );
    info(&mut commands, root, Info::Status);
    info(&mut commands, root, Info::Adapter);
    info(&mut commands, root, Info::Message);
    let playback = row(&mut commands, root);
    button(
        &mut commands,
        playback,
        "Pause",
        Action::Play,
        RoundedCorners::All,
    );
    let step = button(
        &mut commands,
        playback,
        "Step",
        Action::Step,
        RoundedCorners::All,
    );
    commands.entity(step).insert(Gate::Step);
    button(
        &mut commands,
        playback,
        "Reset positions",
        Action::Reset,
        RoundedCorners::All,
    );
    slider(&mut commands, root, Field::Speed);
    let modes = row(&mut commands, root);
    button(
        &mut commands,
        modes,
        "Fast · 256 samples",
        Action::Exact(false),
        RoundedCorners::Left,
    );
    button(
        &mut commands,
        modes,
        "Exact",
        Action::Exact(true),
        RoundedCorners::Right,
    );
    info(&mut commands, root, Info::Exact);
    let tabs = row(&mut commands, root);
    for (i, (name, corners)) in [
        ("World", RoundedCorners::Left),
        ("Appearance", RoundedCorners::None),
        ("Rules", RoundedCorners::Right),
    ]
    .into_iter()
    .enumerate()
    {
        let id = button(&mut commands, tabs, name, Action::Tab(i), corners);
        commands.entity(id).insert(Node {
            flex_grow: 1.,
            height: px(28),
            align_items: AlignItems::Center,
            justify_content: JustifyContent::Center,
            border_radius: corners.to_border_radius(4.),
            ..default()
        });
    }
    let mut bodies = Vec::new();
    for tab in 0..3 {
        let frame = commands
            .spawn((
                ChildOf(root),
                TabBody(tab),
                Node {
                    flex_direction: FlexDirection::Row,
                    flex_grow: 1.,
                    flex_basis: px(0),
                    min_height: px(0),
                    column_gap: px(3),
                    ..default()
                },
            ))
            .id();
        let body = commands
            .spawn_scene(pane_body())
            .insert((
                ChildOf(frame),
                ScrollPosition::default(),
                Node {
                    flex_direction: FlexDirection::Column,
                    flex_grow: 1.,
                    min_width: px(0),
                    min_height: px(0),
                    row_gap: px(8),
                    padding: UiRect::all(px(4)),
                    overflow: Overflow::scroll_y(),
                    ..default()
                },
            ))
            .id();
        scrollbar(&mut commands, frame, body, true);
        bodies.push(body);
    }
    let world = sub(&mut commands, bodies[0], "World & particle types");
    integer(&mut commands, world, "Particles", Integer::Count);
    integer(&mut commands, world, "Types", Integer::Types);
    integer(&mut commands, world, "Seed", Integer::Seed);
    check(
        &mut commands,
        world,
        "Clustered initialization",
        Toggle::Clustered,
    );
    button(
        &mut commands,
        world,
        "Apply counts & regenerate",
        Action::Apply,
        RoundedCorners::All,
    );
    info(&mut commands, world, Info::Error);
    info(&mut commands, world, Info::Active);
    text(
        &mut commands,
        world,
        "Count changes regenerate interaction rules. Seed and clustering apply on reset.",
    );
    let motion = sub(&mut commands, bodies[0], "Motion");
    for f in [Field::Radius, Field::Strength, Field::Damping, Field::SampleBudget] {
        slider(&mut commands, motion, f);
    }
    let color = sub(&mut commands, bodies[1], "Color & light");
    menu(
        &mut commands,
        color,
        Info::Palette,
        PALETTES
            .iter()
            .enumerate()
            .map(|(i, (name, _))| (*name, Action::Palette(i)))
            .collect(),
    );
    let swatches = row(&mut commands, color);
    for i in 0..16 {
        commands.spawn((
            Node {
                width: px(14),
                height: px(14),
                border_radius: BorderRadius::all(px(7)),
                ..default()
            },
            BackgroundColor(Color::WHITE),
            Swatch(i),
            ChildOf(swatches),
        ));
    }
    slider(&mut commands, color, Field::Size);
    slider(&mut commands, color, Field::Glow);
    button(
        &mut commands,
        color,
        "Reset view",
        Action::ResetView,
        RoundedCorners::All,
    );
    let trails = sub(&mut commands, bodies[1], "Chemical trails");
    check(
        &mut commands,
        trails,
        "Enable chemical trails",
        Toggle::Trails,
    );
    text(
        &mut commands,
        trails,
        "A shared scent field: all types leave trails and sense the same chemical. Positive follows trails; negative avoids them; zero only leaves trails.",
    );
    for f in [
        Field::Response,
        Field::Deposit,
        Field::HalfLife,
        Field::Diffusion,
        Field::Sensor,
        Field::Visibility,
    ] {
        slider(&mut commands, trails, f);
    }
    let clear = button(
        &mut commands,
        trails,
        "Clear trails",
        Action::ClearTrails,
        RoundedCorners::All,
    );
    commands.entity(clear).insert(Gate::Trails);
    text(
        &mut commands,
        trails,
        "Trails wrap with the world and pause with the simulation. Reset clears them.",
    );
    let behavior = sub(&mut commands, bodies[2], "Spacing, density & type cycles");
    check(
        &mut commands,
        behavior,
        "Preferred pair distances",
        Toggle::Preferred,
    );
    text(
        &mut commands,
        behavior,
        "Edit Distance in the rule matrix. Replaces attraction for nonzero pairs; close-range repulsion remains.",
    );
    check(
        &mut commands,
        behavior,
        "Density-dependent attraction / repulsion",
        Toggle::Density,
    );
    slider(&mut commands, behavior, Field::DensityTarget);
    slider(&mut commands, behavior, Field::DensityStrength);
    text(
        &mut commands,
        behavior,
        "Below target attracts; above target repels. Fast mode estimates neighbor counts.",
    );
    menu(
        &mut commands,
        behavior,
        Info::Cycle,
        ["Off", "Timer", "Neighbors", "Either"]
            .into_iter()
            .enumerate()
            .map(|(i, n)| (n, Action::Cycle(i as u32)))
            .collect(),
    );
    for f in [Field::CycleSeconds, Field::CycleFraction, Field::CycleMin] {
        slider(&mut commands, behavior, f);
    }
    text(
        &mut commands,
        behavior,
        "Types advance 1 → 2 → … → 1. Either allows neighbor changes after 0.5 s; the timer sets the maximum wait.",
    );
    let rules = sub(&mut commands, bodies[2], "Interaction rules");
    for (name, action) in [
        ("Randomize selected", Action::Randomize),
        ("Zero selected", Action::Zero),
        ("New universe (all rules)", Action::Universe),
    ] {
        button(&mut commands, rules, name, action, RoundedCorners::All);
    }
    let kinds = row(&mut commands, rules);
    for kind in RuleKind::ALL {
        button(
            &mut commands,
            kinds,
            kind.label(),
            Action::Kind(kind),
            RoundedCorners::All,
        );
    }
    text(
        &mut commands,
        rules,
        "Row feels column: blue is positive, coral is negative. Shift + scroll moves horizontally.",
    );
    info(&mut commands, rules, Info::Rule);
    let frame = commands
        .spawn((
            ChildOf(rules),
            Node {
                display: Display::Grid,
                height: px(232),
                min_height: px(232),
                flex_shrink: 0.,
                grid_template_columns: vec![
                    RepeatedGridTrack::flex(1, 1.),
                    RepeatedGridTrack::px(1, 8.),
                ],
                grid_template_rows: vec![
                    RepeatedGridTrack::flex(1, 1.),
                    RepeatedGridTrack::px(1, 8.),
                ],
                column_gap: px(2),
                row_gap: px(2),
                ..default()
            },
        ))
        .id();
    let view = commands
        .spawn((
            MatrixViewport,
            ScrollPosition::default(),
            Node {
                min_width: px(0),
                min_height: px(0),
                overflow: Overflow::scroll(),
                grid_row: GridPlacement::start(1),
                grid_column: GridPlacement::start(1),
                ..default()
            },
            ChildOf(frame),
        ))
        .id();
    scrollbar(&mut commands, frame, view, true);
    scrollbar(&mut commands, frame, view, false);
    commands.spawn((
        MatrixContent,
        Node {
            position_type: PositionType::Relative,
            flex_shrink: 0.,
            ..default()
        },
        ChildOf(view),
    ));
    info(&mut commands, rules, Info::Hover);
    info(&mut commands, rules, Info::Selection);
    slider(&mut commands, rules, Field::Rule);
    text(
        &mut commands,
        root,
        "Background: scroll to zoom · drag to pan\nSpace pause · → step · R reset · F11 fullscreen\nTab: hide/show on background; navigate in controls",
    );
    let show = button(
        &mut commands,
        Entity::PLACEHOLDER,
        "Show controls · Tab",
        Action::Show,
        RoundedCorners::All,
    );
    commands.entity(show).remove::<ChildOf>().insert((
        ShowButton,
        Node {
            position_type: PositionType::Absolute,
            left: px(12),
            top: px(12),
            height: px(28),
            padding: UiRect::horizontal(px(8)),
            align_items: AlignItems::Center,
            ..default()
        },
    ));
}

#[derive(Component, Clone, Copy)]
enum Field {
    Speed,
    Radius,
    Strength,
    Damping,
    Size,
    Glow,
    DensityTarget,
    DensityStrength,
    SampleBudget,
    CycleSeconds,
    CycleFraction,
    CycleMin,
    Response,
    Deposit,
    HalfLife,
    Diffusion,
    Sensor,
    Visibility,
    Rule,
}
impl Field {
    fn label(self) -> &'static str {
        match self {
            Self::Speed => "Speed",
            Self::Radius => "Interaction radius",
            Self::Strength => "Force strength",
            Self::Damping => "Damping",
            Self::Size => "Particle radius",
            Self::Glow => "Glow",
            Self::DensityTarget => "Target neighbors",
            Self::DensityStrength => "Crowding strength",
            Self::SampleBudget => "Sample budget",
            Self::CycleSeconds => "Interval / cooldown (s)",
            Self::CycleFraction => "Next-type fraction",
            Self::CycleMin => "Minimum neighbors",
            Self::Response => "Follow / avoid",
            Self::Deposit => "Deposit rate",
            Self::HalfLife => "Half-life (seconds)",
            Self::Diffusion => "Diffusion",
            Self::Sensor => "Sensing distance",
            Self::Visibility => "Trail visibility",
            Self::Rule => "Selected rule value",
        }
    }
    fn range(self, kind: RuleKind) -> (f32, f32) {
        match self {
            Self::Speed => (0.1, 3.0),
            Self::Radius => (8.0, 128.0),
            Self::Strength => (0.0, 250.0),
            Self::Damping => (0.1, 10.0),
            Self::Size => (0.5, 8.0),
            Self::Glow => (0.0, 1.0),
            Self::DensityTarget => (1.0, 512.0),
            Self::DensityStrength => (0.0, 2.0),
            Self::SampleBudget => (16.0, 256.0),
            Self::CycleSeconds => (0.5, 30.0),
            Self::CycleFraction => (0.05, 1.0),
            Self::CycleMin => (1.0, 128.0),
            Self::Response => (-200.0, 200.0),
            Self::Deposit => (0.0, 10.0),
            Self::HalfLife => (0.5, 30.0),
            Self::Diffusion => (0.0, 30.0),
            Self::Sensor => (4.0, 96.0),
            Self::Visibility => (0.0, 1.0),
            Self::Rule => {
                if kind == RuleKind::Distance {
                    (0., 0.95)
                } else {
                    (-1., 1.)
                }
            }
        }
    }
    fn gate(self) -> Option<Gate> {
        match self {
            Self::DensityTarget => Some(Gate::Density),
            Self::DensityStrength => Some(Gate::Density),
            Self::CycleSeconds => Some(Gate::Cycle),
            Self::CycleFraction => Some(Gate::Neighbors),
            Self::CycleMin => Some(Gate::Neighbors),
            Self::Response => Some(Gate::Trails),
            Self::Deposit => Some(Gate::Trails),
            Self::HalfLife => Some(Gate::Trails),
            Self::Diffusion => Some(Gate::Trails),
            Self::Sensor => Some(Gate::Trails),
            Self::Visibility => Some(Gate::Trails),
            _ => None,
        }
    }
    fn value(self, s: &Simulation, a: &Appearance, c: &Controls) -> f32 {
        match self {
            Self::Speed => s.speed,
            Self::Radius => s.radius,
            Self::Strength => s.strength,
            Self::Damping => s.damping,
            Self::Size => a.size,
            Self::Glow => a.glow,
            Self::DensityTarget => s.behavior.density_target,
            Self::DensityStrength => s.behavior.density_strength,
            Self::SampleBudget => s.behavior.sample_budget as f32,
            Self::CycleSeconds => s.behavior.cycle_seconds,
            Self::CycleFraction => s.behavior.cycle_fraction,
            Self::CycleMin => s.behavior.cycle_min_neighbors as f32,
            Self::Response => s.trails.response,
            Self::Deposit => s.trails.deposit,
            Self::HalfLife => s.trails.half_life,
            Self::Diffusion => s.trails.diffusion,
            Self::Sensor => s.trails.sensor_distance,
            Self::Visibility => s.trails.visibility,
            Self::Rule => s.matrix(c.rule_kind)[(c.selected.0 * s.types + c.selected.1) as usize],
        }
    }
    fn set(self, value: f32, s: &mut Simulation, a: &mut Appearance, c: &Controls) {
        let (min, max) = self.range(c.rule_kind);
        let value = value.clamp(min, max);
        match self {
            Self::Speed => s.speed = value,
            Self::Radius => s.radius = value,
            Self::Strength => s.strength = value,
            Self::Damping => s.damping = value,
            Self::Size => a.size = value,
            Self::Glow => a.glow = value,
            Self::DensityTarget => s.behavior.density_target = value,
            Self::DensityStrength => s.behavior.density_strength = value,
            Self::SampleBudget => s.behavior.sample_budget = value.round() as u32,
            Self::CycleSeconds => s.behavior.cycle_seconds = value,
            Self::CycleFraction => s.behavior.cycle_fraction = value,
            Self::CycleMin => s.behavior.cycle_min_neighbors = value.round() as u32,
            Self::Response => s.trails.response = value,
            Self::Deposit => s.trails.deposit = value,
            Self::HalfLife => s.trails.half_life = value,
            Self::Diffusion => s.trails.diffusion = value,
            Self::Sensor => s.trails.sensor_distance = value,
            Self::Visibility => s.trails.visibility = value,
            Self::Rule => s.set_rule(
                c.rule_kind,
                (c.selected.0 * s.types + c.selected.1) as usize,
                value,
            ),
        }
    }
}
fn enabled(gate: Gate, s: &Simulation) -> bool {
    match gate {
        Gate::Density => s.behavior.density_enabled,
        Gate::Cycle => s.behavior.cycle_mode != 0,
        Gate::Neighbors => s.behavior.cycle_mode >= 2,
        Gate::Trails => s.trails.enabled,
        Gate::Step => s.paused,
    }
}
#[allow(clippy::too_many_arguments)]
fn activate(
    event: On<Activate>,
    actions: Query<(&Action, Has<InteractionDisabled>)>,
    mut s: ResMut<Simulation>,
    mut a: ResMut<Appearance>,
    mut c: ResMut<Controls>,
    status: Res<Status>,
    mut focus: ResMut<InputFocus>,
    background: Single<Entity, With<BackgroundFocus>>,
) {
    let Ok((action, disabled)) = actions.get(event.entity) else {
        return;
    };
    if disabled {
        return;
    }
    match *action {
        Action::Tab(tab) => {
            c.tab = tab;
        }
        Action::Hide => {
            c.visible = false;
            focus.set(*background, FocusCause::Navigated);
        }
        Action::Show => {
            c.visible = true;
            focus.set(*background, FocusCause::Navigated);
        }
        Action::Play => s.paused = !s.paused,
        Action::Step => {
            if s.paused {
                s.step += 1;
            }
        }
        Action::Reset => s.epoch += 1,
        Action::Apply if c.invalid.iter().any(|v| *v) => {}
        Action::Apply => match validate_counts(c.count, c.types, &status.0.lock().unwrap()) {
            Ok(()) => {
                s.count = c.count;
                s.types = c.types;
                s.randomize();
                s.epoch += 1;
                c.selected = (0, 0);
                c.error.clear();
            }
            Err(e) => c.error = e,
        },
        Action::ResetView => {
            a.pan = Vec2::ZERO;
            a.zoom = 1.;
        }
        Action::ClearTrails => s.trails.clear_revision += 1,
        Action::Randomize => {
            s.seed = hash(s.seed);
            s.randomize_matrix(c.rule_kind);
        }
        Action::Zero => s.clear_matrix(c.rule_kind),
        Action::Universe => {
            s.seed = hash(s.seed);
            for k in RuleKind::ALL {
                s.randomize_matrix(k);
            }
            s.epoch += 1;
        }
        Action::Kind(k) => c.rule_kind = k,
        Action::Palette(i) => s.palette = i,
        Action::Cycle(i) => s.behavior.cycle_mode = i,
        Action::Exact(v) => s.exact = v,
        Action::Cell(r, col) => c.selected = (r, col),
    }
}
fn float_change(
    event: On<ValueChange<f32>>,
    fields: Query<(&Field, Has<InteractionDisabled>)>,
    mut s: ResMut<Simulation>,
    mut a: ResMut<Appearance>,
    c: Res<Controls>,
) {
    if let Ok((field, false)) = fields.get(event.source)
        && event.value.is_finite()
    {
        field.set(event.value, &mut s, &mut a, &c);
    }
}
fn valid_integer(value: i64, kind: Integer) -> Option<u32> {
    u32::try_from(value)
        .ok()
        .filter(|v| *v > 0 || matches!(kind, Integer::Seed))
}
fn integer_change(
    event: On<ValueChange<i64>>,
    fields: Query<&Integer>,
    mut s: ResMut<Simulation>,
    mut c: ResMut<Controls>,
) {
    let Ok(kind) = fields.get(event.source) else {
        return;
    };
    if let Some(v) = valid_integer(event.value, *kind) {
        match kind {
            Integer::Count => c.count = v,
            Integer::Types => c.types = v,
            Integer::Seed => s.seed = v,
        };
        if !c.invalid.iter().any(|v| *v) {
            c.error.clear();
        }
    } else {
        c.error =
            "Enter an integer within the allowed range: counts 1–4294967295; seed 0–4294967295."
                .into();
    }
}
fn bool_change(event: On<ValueChange<bool>>, fields: Query<&Toggle>, mut s: ResMut<Simulation>) {
    let Ok(kind) = fields.get(event.source) else {
        return;
    };
    match kind {
        Toggle::Clustered => s.clustered = event.value,
        Toggle::Preferred => s.behavior.preferred_enabled = event.value,
        Toggle::Density => s.behavior.density_enabled = event.value,
        Toggle::Trails => {
            if s.trails.enabled != event.value {
                s.trails.enabled = event.value;
                s.trails.clear_revision += 1;
            }
        }
    }
}
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
fn sync(
    mut commands: Commands,
    s: Res<Simulation>,
    a: Res<Appearance>,
    mut c: ResMut<Controls>,
    status: Res<Status>,
    time: Res<Time>,
    window: Single<&Window, With<PrimaryWindow>>,
    mut nodes: Query<
        (&mut Node, Option<&TabBody>, Has<Sidebar>, Has<ShowButton>),
        Or<(With<TabBody>, With<Sidebar>, With<ShowButton>)>,
    >,
    sliders: Query<(Entity, &Field, &SliderValue, &SliderRange)>,
    integers: Query<(Entity, &Integer)>,
    toggles: Query<(Entity, &Toggle, Has<Checked>)>,
    gates: Query<(Entity, &Gate, Has<InteractionDisabled>)>,
    mut actions: Query<(&Action, &mut ButtonVariant, &Children)>,
    mut texts: Query<(&mut Text, Option<&Info>)>,
    mut swatches: Query<
        (&Swatch, &mut Node, &mut BackgroundColor),
        (Without<TabBody>, Without<Sidebar>, Without<ShowButton>),
    >,
    mut last_integers: Local<Option<[u32; 3]>>,
) {
    c.fps += (1. / time.delta_secs().max(0.001) - c.fps) * 0.05;
    for (mut node, tab, sidebar, show) in &mut nodes {
        node.display = if if let Some(tab) = tab {
            c.visible && c.tab == tab.0
        } else if sidebar {
            c.visible
        } else {
            show && !c.visible
        } {
            Display::Flex
        } else {
            Display::None
        };
        if sidebar {
            let (origin, _) =
                viewport_geometry(window.physical_size(), window.scale_factor(), c.visible);
            node.width = px(origin.x as f32 / window.scale_factor());
        }
    }
    for (entity, field, value, range) in &sliders {
        let v = field.value(&s, &a, &c);
        if value.0 != v {
            commands.entity(entity).insert(SliderValue(v));
        }
        let (min, max) = field.range(c.rule_kind);
        if range.start() != min || range.end() != max {
            commands.entity(entity).insert(SliderRange::new(min, max));
        }
    }
    let values = [c.count, c.types, s.seed];
    for (entity, kind) in &integers {
        let i = match kind {
            Integer::Count => 0,
            Integer::Types => 1,
            Integer::Seed => 2,
        };
        if last_integers.is_none_or(|old| old[i] != values[i]) {
            commands.trigger(UpdateNumberInput {
                entity,
                value: NumberInputValue::I64(i64::from(values[i])),
            });
        }
    }
    *last_integers = Some(values);
    for (entity, kind, checked) in &toggles {
        let value = match kind {
            Toggle::Clustered => s.clustered,
            Toggle::Preferred => s.behavior.preferred_enabled,
            Toggle::Density => s.behavior.density_enabled,
            Toggle::Trails => s.trails.enabled,
        };
        if value != checked {
            if value {
                commands.entity(entity).insert(Checked);
            } else {
                commands.entity(entity).remove::<Checked>();
            }
        }
    }
    for (entity, gate, disabled) in &gates {
        let value = !enabled(*gate, &s);
        if value != disabled {
            if value {
                commands.entity(entity).insert(InteractionDisabled);
            } else {
                commands.entity(entity).remove::<InteractionDisabled>();
            }
        }
    }
    for (action, mut variant, children) in &mut actions {
        let selected = match action {
            Action::Tab(i) => c.tab == *i,
            Action::Kind(k) => c.rule_kind == *k,
            Action::Exact(v) => s.exact == *v,
            _ => false,
        };
        let next = if selected {
            ButtonVariant::Primary
        } else {
            ButtonVariant::Normal
        };
        if *variant != next {
            *variant = next;
        }
        if matches!(action, Action::Play) {
            for child in children {
                if let Ok((mut t, _)) = texts.get_mut(*child) {
                    let next = if s.paused { "Play" } else { "Pause" };
                    if t.0 != next {
                        t.0 = next.into();
                    }
                }
            }
        }
    }
    let report = status.0.lock().unwrap();
    for (mut t, kind) in &mut texts {
        let Some(kind) = kind else { continue };
        let next = match kind {
            Info::Status => format!(
                "{:.0} FPS · {:.0} steps/s · generation {}",
                c.fps, report.steps_per_second, report.generation
            ),
            Info::Adapter => report.adapter.clone(),
            Info::Message => report.message.clone(),
            Info::Error => c.error.clone(),
            Info::Active => format!("Active: {} particles · {} types", s.count, s.types),
            Info::Rule => c.rule_kind.hint().into(),
            Info::Selection => {
                let v = Field::Rule.value(&s, &a, &c);
                let mut t = format!(
                    "Type {} feels type {}: {v:+.3}",
                    c.selected.0 + 1,
                    c.selected.1 + 1
                );
                if c.rule_kind == RuleKind::Distance && v > 0. {
                    t += &format!(
                        "\nPreferred separation: {:.1} world units (at least the repulsion core).",
                        v.max(0.2) * s.radius
                    );
                }
                t
            }
            Info::Palette => format!("Palette: {}", PALETTES[s.palette].0),
            Info::Cycle => format!(
                "Type cycling: {}",
                ["Off", "Timer", "Neighbors", "Either"][s.behavior.cycle_mode as usize]
            ),
            Info::Exact => {
                if s.exact {
                    "Exact evaluates every nearby particle; dense clusters cost more.".into()
                } else {
                    String::new()
                }
            }
            Info::Hover => continue,
        };
        if t.0 != next {
            t.0 = next;
        }
    }
    for (Swatch(i), mut node, mut color) in &mut swatches {
        node.display = if *i < s.types {
            Display::Flex
        } else {
            Display::None
        };
        color.0 = type_color(&s, *i);
    }
}
fn type_color(s: &Simulation, i: u32) -> Color {
    let c = palette_color(s.palette, i, s.types);
    Color::linear_rgb(c[0], c[1], c[2])
}

#[derive(EntityEvent)]
#[entity_event(propagate, auto_propagate)]
struct Scroll {
    entity: Entity,
    delta: Vec2,
}
fn scroll(mut event: On<Scroll>, mut nodes: Query<(&Node, &ComputedNode, &mut ScrollPosition)>) {
    let Ok((node, computed, mut pos)) = nodes.get_mut(event.entity) else {
        return;
    };
    let max = ((computed.content_size() - computed.size()) * computed.inverse_scale_factor())
        .max(Vec2::ZERO);
    for axis in 0..2 {
        let overflow = if axis == 0 {
            node.overflow.x
        } else {
            node.overflow.y
        };
        if overflow == OverflowAxis::Scroll {
            let before = pos.0[axis];
            pos.0[axis] = (before + event.delta[axis]).clamp(0., max[axis]);
            event.delta[axis] -= pos.0[axis] - before;
        }
    }
    if event.delta.length_squared() < 0.01 {
        event.propagate(false);
    }
}
#[allow(clippy::too_many_arguments)]
fn input(
    mut wheel: MessageReader<MouseWheel>,
    motion: Res<AccumulatedMouseMotion>,
    keys: Res<ButtonInput<KeyCode>>,
    buttons: Res<ButtonInput<MouseButton>>,
    hover: Res<HoverMap>,
    window: Single<&Window, With<PrimaryWindow>>,
    mut focus: ResMut<InputFocus>,
    mut c: ResMut<Controls>,
    mut a: ResMut<Appearance>,
    mut commands: Commands,
    parents: Query<&ChildOf>,
    ui: Query<(), With<Node>>,
    background: Single<Entity, With<BackgroundFocus>>,
) {
    let (origin, size) =
        viewport_geometry(window.physical_size(), window.scale_factor(), c.visible);
    let origin = origin.as_vec2() / window.scale_factor();
    let size = size.as_vec2() / window.scale_factor();
    let cursor = window.cursor_position();
    let over_ui = hover
        .values()
        .any(|map| map.keys().any(|e| ui.contains(*e)));
    let on_background = cursor
        .is_some_and(|p| p.x >= origin.x && p.y >= 0. && p.x < origin.x + size.x && p.y < size.y)
        && !over_ui;
    if buttons.any_just_pressed([MouseButton::Left, MouseButton::Middle]) {
        c.dragging = on_background;
        if on_background {
            focus.set(*background, FocusCause::Navigated);
        }
    }
    if !buttons.any_pressed([MouseButton::Left, MouseButton::Middle]) {
        c.dragging = false;
    }
    for w in wheel.read() {
        let delta = Vec2::new(w.x, w.y)
            * if w.unit == MouseScrollUnit::Line {
                24.
            } else {
                1.
            };
        if on_background {
            if let Some(cursor) = cursor {
                a.zoom_at_cursor(cursor - origin, size, delta.y);
            }
        } else {
            let mut delta = -delta;
            if keys.any_pressed([
                KeyCode::ShiftLeft,
                KeyCode::ShiftRight,
                KeyCode::ControlLeft,
                KeyCode::ControlRight,
            ]) {
                std::mem::swap(&mut delta.x, &mut delta.y);
            }
            // Send one bubbling event per pointer, to its deepest UI hit.
            for map in hover.values() {
                if let Some(entity) = map
                    .keys()
                    .filter(|e| ui.contains(**e))
                    .max_by_key(|e| parents.iter_ancestors(**e).count())
                {
                    commands.trigger(Scroll {
                        entity: *entity,
                        delta,
                    });
                }
            }
        }
    }
    if c.dragging && on_background {
        let scale = 2. / size.y.max(1.) / a.zoom;
        a.pan += Vec2::new(-motion.delta.x, motion.delta.y) * scale;
        a.wrap_pan();
    }
    if !window.focused {
        c.dragging = false;
    }
}
/// Inclusive headers at index zero; only allocate cells intersecting the viewport.
fn visible_cells(offset: Vec2, size: Vec2, types: u32) -> (UVec2, UVec2) {
    let limit = UVec2::splat(types.saturating_add(1));
    let first = (offset.max(Vec2::ZERO) / 30.).floor().as_uvec2().min(limit);
    let end = ((offset.max(Vec2::ZERO) + size.max(Vec2::ZERO)) / 30.)
        .ceil()
        .as_uvec2()
        .saturating_add(UVec2::ONE)
        .min(limit);
    (first, end)
}
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
fn matrix(
    mut commands: Commands,
    s: Res<Simulation>,
    c: Res<Controls>,
    viewport: Single<(&ScrollPosition, &ComputedNode), With<MatrixViewport>>,
    mut content: Single<(Entity, &mut Node), With<MatrixContent>>,
    cells: Query<(Entity, &MatrixCell)>,
    hover: Res<HoverMap>,
    mut labels: Query<(&Info, &mut Text)>,
    mut previous: Local<Option<(UVec2, UVec2, u64, u32, usize, RuleKind, (u32, u32))>>,
) {
    if !c.visible || c.tab != 2 {
        return;
    }
    let (pos, computed) = *viewport;
    let (first, end) = visible_cells(
        pos.0,
        computed.size() * computed.inverse_scale_factor(),
        s.types,
    );
    let key = (
        first,
        end,
        s.rules_revision,
        s.types,
        s.palette,
        c.rule_kind,
        c.selected,
    );
    let mut hover_text = String::new();
    for map in hover.values() {
        for entity in map.keys() {
            if let Ok((_, MatrixCell(y, x))) = cells.get(*entity)
                && *x > 0
                && *y > 0
            {
                let v = s.matrix(c.rule_kind)[((*y - 1) * s.types + *x - 1) as usize];
                hover_text = format!("{} feels {}: {v:+.3}", y, x);
            }
        }
    }
    for (kind, mut t) in &mut labels {
        if matches!(kind, Info::Hover) && t.0 != hover_text {
            t.0 = hover_text.clone();
        }
    }
    if previous.as_ref() == Some(&key) {
        return;
    }
    *previous = Some(key);
    for (entity, _) in &cells {
        commands.entity(entity).despawn();
    }
    let (parent, node) = &mut *content;
    node.width = px((s.types as f32 + 1.) * 30.);
    node.height = node.width;
    for y in first.y..end.y {
        for x in first.x..end.x {
            let mut node = Node {
                position_type: PositionType::Absolute,
                left: px(x as f32 * 30.),
                top: px(y as f32 * 30.),
                width: px(28),
                height: px(28),
                align_items: AlignItems::Center,
                justify_content: JustifyContent::Center,
                border_radius: BorderRadius::all(px(4)),
                ..default()
            };
            if x == 0 || y == 0 {
                let id = commands
                    .spawn((node, MatrixCell(y, x), ChildOf(*parent)))
                    .id();
                if x + y > 0 {
                    let label = text(&mut commands, id, x.max(y).to_string());
                    commands
                        .entity(label)
                        .remove::<ThemedText>()
                        .insert(TextColor(type_color(&s, x.max(y) - 1)));
                }
                continue;
            }
            let v = s.matrix(c.rule_kind)[((y - 1) * s.types + x - 1) as usize];
            let intensity = (v.abs() * 180.) as u8;
            let color = if v >= 0. {
                Color::srgb_u8(28, 55 + intensity / 2, 65 + intensity)
            } else {
                Color::srgb_u8(65 + intensity, 35 + intensity / 3, 40)
            };
            if c.selected == (y - 1, x - 1) {
                node.border = UiRect::all(px(2));
            }
            commands
                .spawn((
                    node,
                    BackgroundColor(color),
                    BorderColor::all(Color::WHITE),
                    MatrixCell(y, x),
                    Action::Cell(y - 1, x - 1),
                    ChildOf(*parent),
                ))
                .observe(|event: On<Pointer<Click>>, mut commands: Commands| {
                    commands.trigger(Activate {
                        entity: event.entity,
                    });
                });
        }
    }
}

fn shortcut(
    mut event: On<FocusedInput<KeyboardInput>>,
    mut c: ResMut<Controls>,
    mut s: ResMut<Simulation>,
    mut window: Single<&mut Window, With<PrimaryWindow>>,
) {
    // Consume background keys before they reach Feathers' window-level Tab navigation.
    event.propagate(false);
    if !event.input.state.is_pressed() || event.input.repeat {
        return;
    }
    match event.input.key_code {
        KeyCode::Tab => c.visible = !c.visible,
        KeyCode::Space => s.paused = !s.paused,
        KeyCode::ArrowRight if s.paused => s.step += 1,
        KeyCode::KeyR => s.epoch += 1,
        KeyCode::F11 => {
            window.mode = if window.mode == WindowMode::Windowed {
                WindowMode::BorderlessFullscreen(MonitorSelection::Current)
            } else {
                WindowMode::Windowed
            }
        }
        _ => {}
    }
}

// NumberInput emits no value event for incomplete or overflowing text. Track those
// states as well so Apply never silently uses a previous, valid count.
fn validate_integer_text(
    event: On<TextEditChange>,
    parents: Query<&ChildOf>,
    fields: Query<&Integer>,
    texts: Query<&EditableText>,
    mut c: ResMut<Controls>,
) {
    let Ok(parent) = parents.get(event.event_target()) else {
        return;
    };
    let Ok(kind) = fields.get(parent.parent()) else {
        return;
    };
    let Ok(text) = texts.get(event.event_target()) else {
        return;
    };
    let index = match kind {
        Integer::Count => 0,
        Integer::Types => 1,
        Integer::Seed => 2,
    };
    c.invalid[index] = text
        .value()
        .to_string()
        .parse::<i64>()
        .ok()
        .and_then(|v| valid_integer(v, *kind))
        .is_none();
    if c.invalid.iter().any(|v| *v) {
        c.error =
            "Enter an integer within the allowed range: counts 1–4294967295; seed 0–4294967295."
                .into();
    } else {
        c.error.clear();
    }
}

fn scrollbar(commands: &mut Commands, parent: Entity, target: Entity, vertical: bool) {
    let orientation = if vertical {
        bevy::ui_widgets::ControlOrientation::Vertical
    } else {
        bevy::ui_widgets::ControlOrientation::Horizontal
    };
    commands
        .spawn_scene(bsn! { @FeathersScrollbar { @target: target, @orientation: orientation } })
        .insert((
            ChildOf(parent),
            Node {
                min_width: px(8),
                min_height: px(8),
                flex_shrink: 0.,
                grid_row: GridPlacement::start(if vertical { 1 } else { 2 }),
                grid_column: GridPlacement::start(if vertical { 2 } else { 1 }),
                ..default()
            },
        ));
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn feathers_tree_switches_tabs_and_dispatches_keyboard_input() {
        let mut app = App::new();
        app.add_plugins(
            DefaultPlugins
                .build()
                .disable::<bevy::winit::WinitPlugin>()
                .disable::<bevy::render::RenderPlugin>()
                .disable::<bevy::render::pipelined_rendering::PipelinedRenderingPlugin>()
                .disable::<bevy::log::LogPlugin>(),
        )
        .init_resource::<Simulation>()
        .init_resource::<Appearance>()
        .init_resource::<Status>()
        .add_plugins(ControlsPlugin)
        .add_systems(Startup, |mut commands: Commands| {
            commands.spawn((Camera2d, IsDefaultUiCamera));
        });
        app.finish();
        app.cleanup();
        for _ in 0..4 {
            app.update();
        }
        let window = app
            .world_mut()
            .query_filtered::<Entity, With<PrimaryWindow>>()
            .single(app.world())
            .unwrap();
        let press = |app: &mut App, key_code, logical_key| {
            app.world_mut().write_message(KeyboardInput {
                key_code,
                logical_key,
                state: bevy::input::ButtonState::Pressed,
                text: None,
                repeat: false,
                window,
            });
            app.update();
        };
        press(&mut app, KeyCode::Tab, bevy::input::keyboard::Key::Tab);
        assert!(
            !app.world().resource::<Controls>().visible,
            "Background Tab must hide controls before Feathers navigation"
        );
        press(&mut app, KeyCode::Tab, bevy::input::keyboard::Key::Tab);
        assert!(app.world().resource::<Controls>().visible);
        for tab in 0..3 {
            let entity = app
                .world_mut()
                .query::<(Entity, &Action)>()
                .iter(app.world())
                .find_map(|(e, a)| matches!(a,Action::Tab(i) if *i==tab).then_some(e))
                .unwrap();
            app.world_mut().trigger(Activate { entity });
            app.update();
            assert_eq!(app.world().resource::<Controls>().tab, tab);
            for (body, node) in app
                .world_mut()
                .query::<(&TabBody, &Node)>()
                .iter(app.world())
            {
                assert_eq!(node.display == Display::Flex, body.0 == tab);
            }
        }
        let speed = app
            .world_mut()
            .query::<(Entity, &Field)>()
            .iter(app.world())
            .find_map(|(e, f)| matches!(f, Field::Speed).then_some(e))
            .unwrap();
        app.world_mut()
            .resource_mut::<InputFocus>()
            .set(speed, FocusCause::Navigated);
        let paused = app.world().resource::<Simulation>().paused;
        press(&mut app, KeyCode::Space, bevy::input::keyboard::Key::Space);
        assert_eq!(
            app.world().resource::<Simulation>().paused,
            paused,
            "Focused widgets must not trigger simulation shortcuts"
        );
        press(&mut app, KeyCode::Tab, bevy::input::keyboard::Key::Tab);
        assert!(
            app.world().resource::<Controls>().visible,
            "Widget Tab must navigate without hiding controls"
        );
    }
    fn controls_app() -> App {
        let mut app = App::new();
        app.init_resource::<Simulation>()
            .init_resource::<Appearance>()
            .init_resource::<Controls>()
            .init_resource::<Status>()
            .init_resource::<InputFocus>()
            .add_observer(activate)
            .add_observer(bool_change)
            .add_observer(float_change);
        app.world_mut().spawn(BackgroundFocus);
        {
            let status = app.world().resource::<Status>();
            let mut report = status.0.lock().unwrap();
            report.max_storage = 256 * 1024 * 1024;
            report.max_buffer = 256 * 1024 * 1024;
            report.max_dispatch = 65535;
        }
        app
    }
    #[test]
    fn playback_and_trail_actions_preserve_semantics() {
        let mut app = controls_app();
        let play = app.world_mut().spawn(Action::Play).id();
        let step = app.world_mut().spawn(Action::Step).id();
        app.world_mut().trigger(Activate { entity: step });
        assert_eq!(app.world().resource::<Simulation>().step, 0);
        app.world_mut().trigger(Activate { entity: play });
        app.world_mut().trigger(Activate { entity: step });
        assert_eq!(app.world().resource::<Simulation>().step, 1);
        let toggle = app.world_mut().spawn(Toggle::Trails).id();
        let before = app.world().resource::<Simulation>().trails.clear_revision;
        let value = !app.world().resource::<Simulation>().trails.enabled;
        for _ in 0..2 {
            app.world_mut().trigger(ValueChange {
                source: toggle,
                value,
                is_final: true,
            });
        }
        assert_eq!(
            app.world().resource::<Simulation>().trails.clear_revision,
            before + 1
        );
        let field = app
            .world_mut()
            .spawn((Field::Response, InteractionDisabled))
            .id();
        let before = app.world().resource::<Simulation>().trails.response;
        app.world_mut().trigger(ValueChange {
            source: field,
            value: 100f32,
            is_final: true,
        });
        assert_eq!(app.world().resource::<Simulation>().trails.response, before);
    }
    #[test]
    fn regeneration_validates_before_mutating_the_world() {
        let mut app = controls_app();
        let apply = app.world_mut().spawn(Action::Apply).id();
        let before = app.world().resource::<Simulation>().epoch;
        app.world_mut().resource_mut::<Controls>().types = u32::MAX;
        app.world_mut().trigger(Activate { entity: apply });
        assert_eq!(app.world().resource::<Simulation>().epoch, before);
        assert!(!app.world().resource::<Controls>().error.is_empty());
        {
            let mut c = app.world_mut().resource_mut::<Controls>();
            c.count = 1000;
            c.types = 4;
            c.invalid[0] = true;
        }
        app.world_mut().trigger(Activate { entity: apply });
        assert_eq!(app.world().resource::<Simulation>().epoch, before);
        app.world_mut().resource_mut::<Controls>().invalid[0] = false;
        app.world_mut().trigger(Activate { entity: apply });
        let s = app.world().resource::<Simulation>();
        assert_eq!((s.count, s.types, s.epoch), (1000, 4, before + 1));
        assert_eq!(app.world().resource::<Controls>().selected, (0, 0));
    }
    #[test]
    fn sidebar_bounds_follow_dpi_and_visibility() {
        assert_eq!(
            viewport_geometry(UVec2::new(1440, 900), 1., true),
            (UVec2::new(360, 0), UVec2::new(1080, 900))
        );
        assert_eq!(
            viewport_geometry(UVec2::new(2880, 1800), 2., true),
            (UVec2::new(720, 0), UVec2::new(2160, 1800))
        );
        assert_eq!(
            viewport_geometry(UVec2::new(400, 300), 1.5, true),
            (UVec2::new(200, 0), UVec2::new(200, 300))
        );
        assert_eq!(
            viewport_geometry(UVec2::new(1440, 900), 1., false),
            (UVec2::ZERO, UVec2::new(1440, 900))
        );
        assert_eq!(
            viewport_geometry(UVec2::ZERO, 2., true),
            (UVec2::ZERO, UVec2::ONE)
        );
    }
    #[test]
    fn sidebar_zoom_keeps_world_point_under_cursor() {
        for scale in [1., 1.5, 2.] {
            let (origin, size) =
                viewport_geometry((Vec2::new(1440., 900.) * scale).as_uvec2(), scale, true);
            let size = size.as_vec2() / scale;
            let cursor = origin.as_vec2() / scale + size * Vec2::new(0.37, 0.62);
            let relative = cursor - origin.as_vec2() / scale;
            let mut view = Appearance {
                aspect: size.x / size.y,
                ..default()
            };
            let normalized = relative / size * 2. - Vec2::ONE;
            let offset = normalized * Vec2::new(view.aspect, -1.);
            let before = view.pan + offset / view.zoom;
            view.zoom_at_cursor(relative, size, 24.);
            let after = view.pan + offset / view.zoom;
            assert!((before - after).length() < 1e-5);
        }
    }
    #[test]
    fn integer_entry_preserves_full_unsigned_range() {
        assert_eq!(valid_integer(4_294_967_295, Integer::Seed), Some(u32::MAX));
        assert_eq!(valid_integer(16_777_217, Integer::Count), Some(16_777_217));
        assert_eq!(valid_integer(0, Integer::Seed), Some(0));
        for value in [-1, 4_294_967_296, i64::MAX] {
            assert_eq!(valid_integer(value, Integer::Count), None);
        }
        assert_eq!(valid_integer(0, Integer::Types), None);
    }
    #[test]
    fn matrix_virtualization_is_bounded_and_clamps_edges() {
        let (first, end) = visible_cells(Vec2::new(9000., 12000.), Vec2::new(320., 220.), 100_000);
        assert_eq!(first, UVec2::new(300, 400));
        assert!((end.x - first.x) * (end.y - first.y) <= 12 * 10);
        assert_eq!(
            visible_cells(Vec2::ZERO, Vec2::new(320., 220.), 1),
            (UVec2::ZERO, UVec2::splat(2))
        );
        assert_eq!(
            visible_cells(Vec2::splat(10000.), Vec2::splat(100.), 8),
            (UVec2::splat(9), UVec2::splat(9))
        );
    }
    #[test]
    fn rules_edit_selected_matrix_and_keep_distance_nonnegative() {
        let mut s = Simulation::default();
        let mut a = Appearance::default();
        let c = Controls {
            rule_kind: RuleKind::Distance,
            selected: (1, 2),
            ..default()
        };
        let revision = s.rules_revision;
        Field::Rule.set(-0.5, &mut s, &mut a, &c);
        assert_eq!(s.matrix(RuleKind::Distance)[10], 0.);
        assert_eq!(s.rules_revision, revision + 1);
        Field::Rule.set(1., &mut s, &mut a, &c);
        assert_eq!(s.matrix(RuleKind::Distance)[10], 0.95);
    }
}
