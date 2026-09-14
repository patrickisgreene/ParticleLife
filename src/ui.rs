use crate::interaction::*;
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
    input_focus::{
        FocusCause, FocusedInput, InputFocus,
        tab_navigation::{TabGroup, TabIndex},
    },
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
            .init_resource::<InteractionTools>()
            .add_systems(Startup, setup)
            .add_systems(
                Update,
                (input, update_dumps, sync, matrix, sync_tools)
                    .chain()
                    .in_set(UiSet),
            )
            .add_observer(activate)
            .add_observer(float_change)
            .add_observer(brush_change)
            .add_observer(integer_change)
            .add_observer(bool_change)
            .add_observer(scroll)
            .add_observer(validate_integer_text)
            .add_observer(section_change)
            .add_systems(Update, sync_hidden_focus.after(UiSet))
            .add_systems(
                PostUpdate,
                compact_fonts
                    .after(bevy::ui::UiSystems::Propagate)
                    .before(bevy::ui::UiSystems::Content),
            );
    }
}
#[derive(Resource)]
pub struct Controls {
    count: u32,
    types: u32,
    selected: [(u32, u32); 4],
    pub visible: bool,
    error: String,
    fps: f32,
    dragging: bool,
    pointer_start: Option<Vec2>,
    pointer_moved: bool,
    brushing: bool,
    invalid: [bool; 3],
}
impl Default for Controls {
    fn default() -> Self {
        Self {
            count: 100_000,
            types: 8,
            selected: [(0, 0); 4],
            visible: true,
            error: String::new(),
            fps: 60.,
            dragging: false,
            pointer_start: None,
            pointer_moved: false,
            brushing: false,
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
#[derive(Component, Clone, Copy)]
enum Action {
    Tool(Tool),
    Hide,
    Show,
    Play,
    Step,
    Reset,
    Apply,
    ResetView,
    ClearTrails,
    Randomize(RuleKind),
    Zero(RuleKind),
    Palette(usize),
    Cycle(u32),
    Exact(bool),
    Cell(RuleKind, u32, u32),
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
    Nova,
    Trails,
}
#[derive(Component, Clone, Copy)]
enum Info {
    Status,
    Adapter,
    Message,
    Error,
    Active,
    Rule(RuleKind),
    Selection(RuleKind),
    Hover(RuleKind),
    Palette,
    Cycle,
    Exact,
}
#[derive(Component)]
struct Swatch(u32);
#[derive(Component)]
struct MatrixViewport(RuleKind);
#[derive(Component)]
struct MatrixContent(RuleKind);
#[derive(Component)]
struct MatrixCell(RuleKind, u32, u32);
#[derive(Component, Clone, Copy)]
enum Gate {
    Density,
    Nova,
    Cycle,
    Neighbors,
    Trails,
    Step,
}

fn text(commands: &mut Commands, parent: Entity, value: impl Into<String>) -> Entity {
    let value = value.into();
    commands.spawn_scene(bsn! { Text(value) ThemedText TextFont { font: FontSourceTemplate::Handle(fonts::REGULAR), font_size: bevy::text::FontSize::Px(11.0) } }).insert(ChildOf(parent)).id()
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
#[derive(Component)]
struct SectionBody;
#[derive(Component)]
struct SectionToggle(Entity);
#[derive(Component)]
struct HiddenTabIndex(i32);

// Feathers has font overrides inside controls as well as on containers.
fn compact_fonts(mut fonts: Query<&mut TextFont, Changed<TextFont>>) {
    for mut font in &mut fonts {
        if font.font_size != bevy::text::FontSize::Px(11.0) {
            font.font_size = bevy::text::FontSize::Px(11.0);
        }
    }
}

fn section_change(
    event: On<ValueChange<bool>>,
    toggles: Query<&SectionToggle>,
    mut bodies: Query<&mut Node, With<SectionBody>>,
    mut commands: Commands,
    parents: Query<&ChildOf>,
    mut focus: ResMut<InputFocus>,
) {
    let Ok(toggle) = toggles.get(event.source) else {
        return;
    };
    let Ok(mut body) = bodies.get_mut(toggle.0) else {
        return;
    };
    body.display = if event.value {
        Display::Flex
    } else {
        Display::None
    };
    if event.value {
        commands.entity(event.source).insert(Checked);
    } else {
        commands.entity(event.source).remove::<Checked>();
        if focus
            .get()
            .is_some_and(|entity| parents.iter_ancestors(entity).any(|e| e == toggle.0))
        {
            focus.set(event.source, FocusCause::Navigated);
        }
    }
}

// Bevy tab navigation does not filter Display::None ancestors itself.
fn sync_hidden_focus(
    mut commands: Commands,
    parents: Query<&ChildOf>,
    nodes: Query<&Node>,
    mut indices: Query<(Entity, &mut TabIndex, Option<&HiddenTabIndex>)>,
) {
    for (entity, mut index, saved) in &mut indices {
        let hidden = std::iter::once(entity)
            .chain(parents.iter_ancestors(entity))
            .any(|e| nodes.get(e).is_ok_and(|n| n.display == Display::None));
        if hidden && saved.is_none() {
            commands.entity(entity).insert(HiddenTabIndex(index.0));
            index.0 = -1;
        } else if !hidden && let Some(saved) = saved {
            index.0 = saved.0;
            commands.entity(entity).remove::<HiddenTabIndex>();
        }
    }
}

fn sub(commands: &mut Commands, parent: Entity, title: &'static str, expanded: bool) -> Entity {
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
    commands
        .entity(header)
        .entry::<Node>()
        .and_modify(|mut node| {
            node.min_height = px(24);
            node.justify_content = JustifyContent::Start;
        });
    let toggle = commands
        .spawn_scene(bsn! { @FeathersDisclosureToggle })
        .insert(ChildOf(header))
        .id();
    text(commands, header, title);
    let body = commands
        .spawn_scene(subpane_body())
        .insert((ChildOf(root), SectionBody))
        .id();
    commands
        .entity(body)
        .entry::<Node>()
        .and_modify(move |mut node| {
            node.display = if expanded {
                Display::Flex
            } else {
                Display::None
            };
        });
    commands.entity(toggle).insert(SectionToggle(body));
    if expanded {
        commands.entity(toggle).insert(Checked);
    }
    commands.entity(header).observe(
        move |event: On<Pointer<Click>>,
              parents: Query<&ChildOf>,
              checked: Query<Has<Checked>>,
              mut commands: Commands| {
            // The native toggle handles its own clicks; the rest of the header forwards activation.
            if event.entity != toggle && !parents.iter_ancestors(event.entity).any(|e| e == toggle)
            {
                commands.trigger(ValueChange {
                    source: toggle,
                    value: !checked.get(toggle).unwrap_or(false),
                    is_final: true,
                });
            }
        },
    );
    body
}

fn control_row(commands: &mut Commands, parent: Entity, caption: &'static str) -> Entity {
    let row = commands
        .spawn((
            Node {
                width: percent(100),
                flex_direction: FlexDirection::Row,
                align_items: AlignItems::Center,
                column_gap: px(6),
                flex_shrink: 0.,
                ..default()
            },
            ChildOf(parent),
        ))
        .id();
    let label = text(commands, row, caption);
    commands.entity(label).insert(Node {
        width: percent(45),
        min_width: px(0),
        flex_shrink: 0.,
        ..default()
    });
    let slot = commands
        .spawn((
            Node {
                flex_grow: 1.,
                flex_basis: px(0),
                min_width: px(0),
                ..default()
            },
            ChildOf(row),
        ))
        .id();
    slot
}
fn check(commands: &mut Commands, parent: Entity, caption: &'static str, kind: Toggle) {
    commands
        .spawn_scene(bsn! { @FeathersCheckbox { @caption: bsn! { Text(caption) ThemedText } } })
        .insert((ChildOf(parent), kind));
}
fn integer(commands: &mut Commands, parent: Entity, caption: &'static str, kind: Integer) {
    let parent = control_row(commands, parent, caption);
    commands
        .spawn_scene(bsn! { @FeathersNumberInput { @number_format: NumberFormat::I64 } Node { width: percent(100), min_width: px(0) } })
        .insert((ChildOf(parent), kind));
}
fn slider(commands: &mut Commands, parent: Entity, field: Field) -> Entity {
    let parent = control_row(commands, parent, field.label());
    let (min, max) = field.range();
    let id = commands
        .spawn_scene(bsn! { @FeathersSlider { @min: min, @max: max } Node { width: percent(100), min_width: px(0), flex_grow: 1.0 } })
        .insert((
            ChildOf(parent),
            field,
            SliderPrecision(if matches!(field, Field::Rule(_)) {
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
    let parent = control_row(
        commands,
        parent,
        if matches!(kind, Info::Palette) {
            "Palette"
        } else {
            "Type cycle"
        },
    );
    let root = commands
        .spawn_scene(bsn! { @FeathersMenu Node { width: percent(100), min_width: px(0) } })
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
#[derive(Component, Clone, Copy)]
enum BrushControl {
    Radius,
    Strength,
    Rate,
}
#[derive(Component)]
struct BrushSettings;
#[derive(Component)]
struct ToolHint;

fn brush_change(
    event: On<ValueChange<f32>>,
    fields: Query<&BrushControl>,
    mut tools: ResMut<InteractionTools>,
) {
    if !event.value.is_finite() {
        return;
    }
    match fields.get(event.source) {
        Ok(BrushControl::Radius) => tools.radius = event.value.clamp(8., 512.),
        Ok(BrushControl::Strength) => tools.strength = event.value.clamp(0., 5000.),
        Ok(BrushControl::Rate) => tools.rate = event.value.clamp(100., 20000.),
        _ => (),
    }
}
fn update_dumps(
    time: Res<Time>,
    status: Res<Status>,
    mut tools: ResMut<InteractionTools>,
    mut sim: ResMut<Simulation>,
    mut controls: ResMut<Controls>,
    mut remainder: Local<f32>,
) {
    if tools.epoch != sim.epoch {
        tools.pending.clear();
        tools.epoch = sim.epoch;
        tools.active = false;
        tools.error.clear();
        *remainder = 0.;
    }
    let report = status.0.lock().unwrap();
    if report.dump_ack.0 == sim.epoch {
        tools
            .pending
            .retain(|batch| batch.serial > report.dump_ack.1);
    }
    if tools.tool != Tool::Dump || !tools.active {
        *remainder = 0.;
        return;
    }
    let Some(center) = tools.cursor else {
        return;
    };
    if tools.pending.len() >= 120 {
        tools.error = "Waiting for the GPU to catch up.".into();
        return;
    }
    *remainder += tools.rate * time.delta_secs().min(0.1);
    let count = remainder.floor() as u32;
    if count == 0 {
        return;
    }
    *remainder -= count as f32;
    let Some(next) = sim.count.checked_add(count) else {
        tools.error = "Particle count limit reached.".into();
        return;
    };
    if let Err(error) = validate_counts(next, sim.types, &report) {
        tools.error = error;
        return;
    }
    tools.error.clear();
    tools.serial += 1;
    let batch = ParticleDump {
        serial: tools.serial,
        start: sim.count,
        count,
        center,
        radius: tools.radius,
    };
    tools.pending.push(batch);
    sim.count = next;
    controls.count = next;
}
fn sync_tools(
    mut commands: Commands,
    tools: Res<InteractionTools>,
    mut actions: Query<(&Action, &mut ButtonVariant)>,
    mut panels: Query<&mut Node, With<BrushSettings>>,
    sliders: Query<(Entity, &BrushControl, &SliderValue)>,
    mut labels: Query<&mut Text, With<ToolHint>>,
) {
    for (action, mut variant) in &mut actions {
        let Action::Tool(tool) = action else { continue };
        let next = if tools.tool == *tool {
            ButtonVariant::Primary
        } else {
            ButtonVariant::Normal
        };
        if *variant != next {
            *variant = next;
        }
    }
    for mut node in &mut panels {
        node.display = if tools.tool != Tool::Navigate {
            Display::Flex
        } else {
            Display::None
        };
    }
    for (entity, field, value) in &sliders {
        let next = match field {
            BrushControl::Radius => tools.radius,
            BrushControl::Strength => tools.strength,
            BrushControl::Rate => tools.rate,
        };
        if value.0 != next {
            commands.entity(entity).insert(SliderValue(next));
        }
    }
    for mut text in &mut labels {
        let hint = match tools.tool {
            Tool::Navigate => "Drag to pan · wheel to zoom · middle drag works with every tool",
            Tool::Attract => "Hold left to attract · Shift + wheel changes radius",
            Tool::Repel => "Hold left to repel · Shift + wheel changes radius",
            Tool::Dump => "Hold left to add particles · mixed types · Shift + wheel changes radius",
        };
        let next = if tools.error.is_empty() {
            hint.to_string()
        } else {
            format!("{hint}\n{}", tools.error)
        };
        if text.0 != next {
            text.0 = next;
        }
    }
}
fn setup(mut commands: Commands, mut focus: ResMut<InputFocus>) {
    let background = commands.spawn(BackgroundFocus).observe(shortcut).id();
    focus.set(background, FocusCause::Navigated);
    let root=commands.spawn_scene(bsn! { pane() InheritableFont { font: fonts::REGULAR, font_size: bevy::text::FontSize::Px(11.0) } }).insert((Sidebar, TabGroup::default(), Node {
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
    let tools = row(&mut commands, root);
    for (label, tool) in [
        ("Pan", Tool::Navigate),
        ("Attract", Tool::Attract),
        ("Repel", Tool::Repel),
        ("Dump", Tool::Dump),
    ] {
        button(
            &mut commands,
            tools,
            label,
            Action::Tool(tool),
            RoundedCorners::All,
        );
    }
    let hint = text(&mut commands, root, "");
    commands.entity(hint).insert(ToolHint);
    let brush = commands
        .spawn((
            Node {
                flex_direction: FlexDirection::Column,
                row_gap: px(4),
                ..default()
            },
            BrushSettings,
            ChildOf(root),
        ))
        .id();
    for (caption, field, min, max) in [
        ("Brush radius", BrushControl::Radius, 8_f32, 512_f32),
        ("Brush strength", BrushControl::Strength, 0_f32, 5000_f32),
        ("Particles / second", BrushControl::Rate, 100_f32, 20000_f32),
    ] {
        let slot = control_row(&mut commands, brush, caption);
        commands.spawn_scene(bsn! { @FeathersSlider { @min: min, @max: max } Node { width: percent(100), min_width: px(0) } }).insert((ChildOf(slot), field, SliderPrecision(0)));
    }
    let frame = commands
        .spawn((
            ChildOf(root),
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
    let world = sub(&mut commands, body, "World & particle types", true);
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
    let motion = sub(&mut commands, body, "Motion", false);
    for f in [
        Field::Radius,
        Field::Strength,
        Field::Damping,
        Field::SampleBudget,
    ] {
        slider(&mut commands, motion, f);
    }
    let color = sub(&mut commands, body, "Color & light", false);
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
    let trails = sub(&mut commands, body, "Chemical trails", false);
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
    let behavior = sub(&mut commands, body, "Spacing, density & type cycles", false);
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
    let nova = sub(&mut commands, body, "Stars & novae", false);
    check(
        &mut commands,
        nova,
        "Ignite dense same-type clumps",
        Toggle::Nova,
    );
    for f in [
        Field::NovaThreshold,
        Field::NovaRadius,
        Field::NovaImpulse,
        Field::NovaDuration,
    ] {
        slider(&mut commands, nova, f);
    }
    text(
        &mut commands,
        nova,
        "Too many particles of one type inside the ignition radius blow apart. The core is reseeded at random across the world; fast mode estimates the count, so exact mode ignites more precisely.",
    );
    for kind in RuleKind::ALL {
        let rules = sub(&mut commands, body, kind.label(), false);
        text(
            &mut commands,
            rules,
            match kind {
                RuleKind::Attraction => {
                    "Pulls particles together or pushes them apart. Positive attracts; negative repels. Very close particles always repel."
                }
                RuleKind::Swirl => {
                    "Adds sideways motion around neighboring particles. Positive turns counterclockwise; negative turns clockwise."
                }
                RuleKind::Alignment => {
                    "Steers particles toward their neighbors’ velocity. Positive matches their motion; negative steers against it."
                }
                RuleKind::Distance => {
                    "Sets a preferred separation as a fraction of the interaction radius. Enable Preferred pair distances above to use it. Nonzero values replace radial attraction for that pair; zero keeps the attraction rule."
                }
            },
        );
        let actions = row(&mut commands, rules);
        button(
            &mut commands,
            actions,
            "Randomize",
            Action::Randomize(kind),
            RoundedCorners::All,
        );
        button(
            &mut commands,
            actions,
            "Zero",
            Action::Zero(kind),
            RoundedCorners::All,
        );
        text(
            &mut commands,
            rules,
            "Row feels column. Select a cell to edit that pair. Shift + scroll moves horizontally.",
        );
        info(&mut commands, rules, Info::Rule(kind));
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
                MatrixViewport(kind),
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
            MatrixContent(kind),
            Node {
                position_type: PositionType::Relative,
                flex_shrink: 0.,
                ..default()
            },
            ChildOf(view),
        ));
        info(&mut commands, rules, Info::Hover(kind));
        info(&mut commands, rules, Info::Selection(kind));
        slider(&mut commands, rules, Field::Rule(kind));
    }
    text(
        &mut commands,
        root,
        "Background: wheel zoom · middle drag pan\nSpace pause · → step · R reset · F11 fullscreen\nTab: hide/show on background; navigate in controls",
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

#[derive(Component, Clone, Copy, PartialEq)]
enum Field {
    Speed,
    Radius,
    Strength,
    Damping,
    Size,
    Glow,
    DensityTarget,
    DensityStrength,
    NovaThreshold,
    NovaRadius,
    NovaImpulse,
    NovaDuration,
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
    Rule(RuleKind),
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
            Self::NovaThreshold => "Ignition threshold",
            Self::NovaRadius => "Ignition radius",
            Self::NovaImpulse => "Blast strength",
            Self::NovaDuration => "Blast duration (s)",
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
            Self::Rule(_) => "Selected rule value",
        }
    }
    fn range(self) -> (f32, f32) {
        match self {
            Self::Speed => (0.1, 3.0),
            Self::Radius => (8.0, 128.0),
            Self::Strength => (0.0, 250.0),
            Self::Damping => (0.1, 10.0),
            Self::Size => (0.5, 8.0),
            Self::Glow => (0.0, 1.0),
            Self::DensityTarget => (1.0, 512.0),
            Self::DensityStrength => (0.0, 2.0),
            Self::NovaThreshold => (4.0, 256.0),
            Self::NovaRadius => (0.05, 1.0),
            Self::NovaImpulse => (200.0, 6000.0),
            Self::NovaDuration => (0.05, 1.0),
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
            Self::Rule(kind) => {
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
            Self::NovaThreshold | Self::NovaRadius => Some(Gate::Nova),
            Self::NovaImpulse | Self::NovaDuration => Some(Gate::Nova),
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
            Self::NovaThreshold => s.behavior.nova_threshold,
            Self::NovaRadius => s.behavior.nova_radius,
            Self::NovaImpulse => s.behavior.nova_impulse,
            Self::NovaDuration => s.behavior.nova_duration,
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
            Self::Rule(kind) => {
                let (row, col) = c.selected[kind as usize];
                s.matrix(kind)[(row * s.types + col) as usize]
            }
        }
    }
    fn set(self, value: f32, s: &mut Simulation, a: &mut Appearance, c: &Controls) {
        let (min, max) = self.range();
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
            Self::NovaThreshold => s.behavior.nova_threshold = value,
            Self::NovaRadius => s.behavior.nova_radius = value,
            Self::NovaImpulse => s.behavior.nova_impulse = value,
            Self::NovaDuration => s.behavior.nova_duration = value,
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
            Self::Rule(kind) => {
                let (row, col) = c.selected[kind as usize];
                s.set_rule(kind, (row * s.types + col) as usize, value);
            }
        }
    }
}
fn enabled(gate: Gate, s: &Simulation) -> bool {
    match gate {
        Gate::Density => s.behavior.density_enabled,
        Gate::Nova => s.behavior.nova_enabled,
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
    mut tools: ResMut<InteractionTools>,
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
        Action::Tool(tool) => {
            tools.tool = tool;
            tools.active = false;
            c.brushing = false;
            c.dragging = false;
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
                c.selected = [(0, 0); 4];
                c.error.clear();
            }
            Err(e) => c.error = e,
        },
        Action::ResetView => {
            a.pan = Vec2::ZERO;
            a.zoom = 1.;
        }
        Action::ClearTrails => s.trails.clear_revision += 1,
        Action::Randomize(kind) => {
            s.seed = hash(s.seed);
            s.randomize_matrix(kind);
        }
        Action::Zero(kind) => s.clear_matrix(kind),
        Action::Palette(i) => s.palette = i,
        Action::Cycle(i) => s.behavior.cycle_mode = i,
        Action::Exact(v) => s.exact = v,
        Action::Cell(kind, r, col) => c.selected[kind as usize] = (r, col),
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
        Toggle::Nova => s.behavior.nova_enabled = event.value,
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
        (&mut Node, Has<Sidebar>, Has<ShowButton>),
        Or<(With<Sidebar>, With<ShowButton>)>,
    >,
    sliders: Query<(Entity, &Field, &SliderValue, &SliderRange)>,
    integers: Query<(Entity, &Integer)>,
    toggles: Query<(Entity, &Toggle, Has<Checked>)>,
    gates: Query<(Entity, &Gate, Has<InteractionDisabled>)>,
    mut actions: Query<(&Action, &mut ButtonVariant, &Children)>,
    mut texts: Query<(&mut Text, Option<&Info>)>,
    mut swatches: Query<
        (&Swatch, &mut Node, &mut BackgroundColor),
        (Without<Sidebar>, Without<ShowButton>),
    >,
    mut last_integers: Local<Option<[u32; 3]>>,
) {
    c.fps += (1. / time.delta_secs().max(0.001) - c.fps) * 0.05;
    for (mut node, sidebar, show) in &mut nodes {
        node.display = if if sidebar {
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
        let (min, max) = field.range();
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
            Toggle::Nova => s.behavior.nova_enabled,
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
            Action::Tool(_) => continue,
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
            Info::Rule(kind) => kind.hint().into(),
            Info::Selection(kind) => {
                let selected = c.selected[*kind as usize];
                let v = Field::Rule(*kind).value(&s, &a, &c);
                let mut t = format!(
                    "Type {} feels type {}: {v:+.3}",
                    selected.0 + 1,
                    selected.1 + 1
                );
                if *kind == RuleKind::Distance && v > 0. {
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
            Info::Hover(_) => continue,
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
    mut tools: ResMut<InteractionTools>,
    sim: Res<Simulation>,
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
    let on_background = on_background && window.focused;
    if buttons.just_pressed(MouseButton::Middle) {
        c.dragging = on_background;
        c.brushing = false;
        c.pointer_start = None;
        if on_background {
            focus.set(*background, FocusCause::Navigated);
        }
    }
    if buttons.just_pressed(MouseButton::Left) && !buttons.pressed(MouseButton::Middle) {
        c.pointer_start = cursor.filter(|_| on_background);
        c.pointer_moved = false;
        c.brushing = on_background && tools.tool != Tool::Navigate;
        c.dragging = on_background && tools.tool == Tool::Navigate;
        if on_background {
            focus.set(*background, FocusCause::Navigated);
        }
    }
    if let (Some(start), Some(cursor)) = (c.pointer_start, cursor) {
        c.pointer_moved |= cursor.distance(start) > 4.;
    }
    if buttons.just_released(MouseButton::Left) {
        c.pointer_start = None;
        c.brushing = false;
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
                if tools.tool != Tool::Navigate
                    && keys.any_pressed([KeyCode::ShiftLeft, KeyCode::ShiftRight])
                {
                    tools.radius = (tools.radius * (delta.y * 0.004).exp()).clamp(8., 512.);
                } else {
                    a.zoom_at_cursor(cursor - origin, size, delta.y);
                }
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
    if c.dragging && on_background && (buttons.pressed(MouseButton::Middle) || c.pointer_moved) {
        let scale = 2. / size.y.max(1.) / a.zoom;
        a.pan += Vec2::new(-motion.delta.x, motion.delta.y) * scale;
        a.wrap_pan();
    }
    tools.cursor = cursor
        .filter(|_| on_background)
        .map(|cursor| cursor_world(&a, cursor - origin, size));
    tools.active = c.brushing
        && buttons.pressed(MouseButton::Left)
        && !buttons.pressed(MouseButton::Middle)
        && on_background
        && (!sim.paused || tools.tool == Tool::Dump);
    if !window.focused {
        c.dragging = false;
        c.brushing = false;
        c.pointer_start = None;
        tools.active = false;
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
    viewports: Query<(&MatrixViewport, &ScrollPosition, &ComputedNode)>,
    mut contents: Query<(Entity, &MatrixContent, &mut Node)>,
    parents: Query<&ChildOf>,
    sections: Query<&Node, (With<SectionBody>, Without<MatrixContent>)>,
    cells: Query<(Entity, &MatrixCell)>,
    hover: Res<HoverMap>,
    mut labels: Query<(&Info, &mut Text)>,
    mut previous: Local<[Option<(UVec2, UVec2, u64, u32, usize, RuleKind, (u32, u32))>; 4]>,
) {
    if !c.visible {
        return;
    }
    for (parent, content, mut node) in &mut contents {
        let kind = content.0;
        let selected = c.selected[kind as usize];
        if parents.iter_ancestors(parent).any(|entity| {
            sections
                .get(entity)
                .is_ok_and(|node| node.display == Display::None)
        }) {
            continue;
        }
        let Some((_, pos, computed)) = viewports.iter().find(|(view, _, _)| view.0 == kind) else {
            continue;
        };
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
            kind,
            selected,
        );
        let mut hover_text = String::new();
        for map in hover.values() {
            for entity in map.keys() {
                if let Ok((_, MatrixCell(cell_kind, y, x))) = cells.get(*entity)
                    && *cell_kind == kind
                    && *x > 0
                    && *y > 0
                {
                    let v = s.matrix(kind)[((*y - 1) * s.types + *x - 1) as usize];
                    hover_text = format!("{} feels {}: {v:+.3}", y, x);
                }
            }
        }
        for (info, mut t) in &mut labels {
            if matches!(info, Info::Hover(k) if *k == kind) && t.0 != hover_text {
                t.0 = hover_text.clone();
            }
        }
        if previous[kind as usize].as_ref() == Some(&key) {
            continue;
        }
        previous[kind as usize] = Some(key);
        for (entity, cell) in &cells {
            if cell.0 == kind {
                commands.entity(entity).despawn();
            }
        }
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
                        .spawn((node, MatrixCell(kind, y, x), ChildOf(parent)))
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
                let v = s.matrix(kind)[((y - 1) * s.types + x - 1) as usize];
                let intensity = (v.abs() * 180.) as u8;
                let color = if v >= 0. {
                    Color::srgb_u8(28, 55 + intensity / 2, 65 + intensity)
                } else {
                    Color::srgb_u8(65 + intensity, 35 + intensity / 3, 40)
                };
                if selected == (y - 1, x - 1) {
                    node.border = UiRect::all(px(2));
                }
                commands
                    .spawn((
                        node,
                        BackgroundColor(color),
                        BorderColor::all(Color::WHITE),
                        MatrixCell(kind, y, x),
                        Action::Cell(kind, y - 1, x - 1),
                        ChildOf(parent),
                    ))
                    .observe(|event: On<Pointer<Click>>, mut commands: Commands| {
                        commands.trigger(Activate {
                            entity: event.entity,
                        });
                    });
            }
        }
    }
}

fn shortcut(
    mut event: On<FocusedInput<KeyboardInput>>,
    mut c: ResMut<Controls>,
    mut tools: ResMut<InteractionTools>,
    mut s: ResMut<Simulation>,
    mut window: Single<&mut Window, With<PrimaryWindow>>,
) {
    // Consume background keys before they reach Feathers' window-level Tab navigation.
    event.propagate(false);
    if !event.input.state.is_pressed() || event.input.repeat {
        return;
    }
    match event.input.key_code {
        KeyCode::Escape => {
            tools.tool = Tool::Navigate;
            tools.active = false;
            c.brushing = false;
            c.dragging = false;
            c.pointer_start = None;
        }
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
    fn feathers_sections_preserve_state_and_dispatch_keyboard_input() {
        let mut app = App::new();
        app.add_plugins(AssetPlugin::default())
            .init_asset::<bevy::shader::Shader>();
        app.add_plugins(
            DefaultPlugins
                .build()
                .disable::<AssetPlugin>()
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
        let sections: Vec<_> = app
            .world_mut()
            .query::<(Entity, &SectionToggle, Has<Checked>)>()
            .iter(app.world())
            .map(|(e, section, checked)| (e, section.0, checked))
            .collect();
        assert_eq!(sections.len(), 10);
        assert_eq!(
            sections.iter().filter(|(_, _, checked)| *checked).count(),
            1
        );
        for (_, body, checked) in &sections {
            assert_eq!(
                app.world().get::<Node>(*body).unwrap().display == Display::Flex,
                *checked
            );
        }
        // All section roots share one scrollable body. Only the first starts open.
        let mut section_parents = Vec::new();
        for (_, body, _) in &sections {
            let root = app.world().get::<ChildOf>(*body).unwrap().parent();
            section_parents.push(app.world().get::<ChildOf>(root).unwrap().parent());
        }
        assert!(
            section_parents
                .iter()
                .all(|parent| *parent == section_parents[0])
        );
        assert!(
            app.world()
                .get::<ScrollPosition>(section_parents[0])
                .is_some()
        );
        // Keyboard activation closes an expanded section.
        let section_for = |app: &mut App, field: Field| {
            let entity = app
                .world_mut()
                .query::<(Entity, &Field)>()
                .iter(app.world())
                .find_map(|(e, f)| (*f == field).then_some(e))
                .unwrap();
            let mut ancestor = entity;
            while app.world().get::<SectionBody>(ancestor).is_none() {
                ancestor = app.world().get::<ChildOf>(ancestor).unwrap().parent();
            }
            *sections
                .iter()
                .find(|(_, body, _)| *body == ancestor)
                .unwrap()
        };
        let (toggle, body, _) = section_for(&mut app, Field::DensityTarget);
        app.world_mut().trigger(ValueChange {
            source: toggle,
            value: true,
            is_final: true,
        });
        app.update();
        app.world_mut()
            .resource_mut::<InputFocus>()
            .set(toggle, FocusCause::Navigated);
        press(&mut app, KeyCode::Space, bevy::input::keyboard::Key::Space);
        assert_eq!(
            app.world().get::<Node>(body).unwrap().display,
            Display::None
        );
        assert!(app.world().get::<Checked>(toggle).is_none());
        // Reopening a different section leaves the first one collapsed.
        let (other_toggle, other_body, _) =
            section_for(&mut app, Field::Rule(RuleKind::Attraction));
        app.world_mut().trigger(ValueChange {
            source: other_toggle,
            value: true,
            is_final: true,
        });
        app.update();
        assert_eq!(
            app.world().get::<Node>(other_body).unwrap().display,
            Display::Flex
        );
        assert_eq!(
            app.world().get::<Node>(body).unwrap().display,
            Display::None
        );
        let rule = app
            .world_mut()
            .query::<(Entity, &Field)>()
            .iter(app.world())
            .find_map(|(e, f)| matches!(f, Field::Rule(RuleKind::Attraction)).then_some(e))
            .unwrap();
        app.world_mut().trigger(ValueChange {
            source: rule,
            value: 0.25f32,
            is_final: true,
        });
        app.update();
        app.world_mut()
            .resource_mut::<InputFocus>()
            .set(rule, FocusCause::Navigated);
        app.world_mut().trigger(ValueChange {
            source: other_toggle,
            value: false,
            is_final: true,
        });
        app.update();
        assert_eq!(
            app.world().resource::<InputFocus>().get(),
            Some(other_toggle)
        );
        assert_eq!(app.world().get::<TabIndex>(rule).unwrap().0, -1);
        app.world_mut().resource_mut::<Controls>().visible = false;
        app.update();
        app.world_mut().resource_mut::<Controls>().visible = true;
        app.update();
        assert_eq!(
            app.world().get::<Node>(other_body).unwrap().display,
            Display::None
        );
        app.world_mut().trigger(ValueChange {
            source: other_toggle,
            value: true,
            is_final: true,
        });
        app.update();
        assert_eq!(app.world().get::<SliderValue>(rule).unwrap().0, 0.25);
        assert_eq!(app.world().get::<TabIndex>(rule).unwrap().0, 0);
        for kind in RuleKind::ALL {
            let (toggle, _, _) = section_for(&mut app, Field::Rule(kind));
            app.world_mut().trigger(ValueChange {
                source: toggle,
                value: true,
                is_final: true,
            });
        }
        app.update();
        app.update();
        for kind in RuleKind::ALL {
            assert!(
                app.world_mut()
                    .query::<&MatrixCell>()
                    .iter(app.world())
                    .any(|cell| cell.0 == kind)
            );
        }
        for font in app.world_mut().query::<&TextFont>().iter(app.world()) {
            assert_eq!(font.font_size, bevy::text::FontSize::Px(11.0));
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
            .init_resource::<InteractionTools>()
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
        assert_eq!(app.world().resource::<Controls>().selected, [(0, 0); 4]);
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
    fn rule_panels_edit_and_clear_only_their_own_matrix() {
        let mut app = controls_app();
        for (index, kind) in RuleKind::ALL.into_iter().enumerate() {
            let cell = app
                .world_mut()
                .spawn(Action::Cell(kind, index as u32, 1))
                .id();
            app.world_mut().trigger(Activate { entity: cell });
            let field = app.world_mut().spawn(Field::Rule(kind)).id();
            app.world_mut().trigger(ValueChange {
                source: field,
                value: 0.1 * (index + 1) as f32,
                is_final: true,
            });
        }
        for (index, kind) in RuleKind::ALL.into_iter().enumerate() {
            let s = app.world().resource::<Simulation>();
            let c = app.world().resource::<Controls>();
            assert_eq!(c.selected[index], (index as u32, 1));
            assert_eq!(
                Field::Rule(kind).value(s, &Appearance::default(), c),
                0.1 * (index + 1) as f32
            );
        }
        let before: Vec<_> = RuleKind::ALL
            .into_iter()
            .map(|kind| app.world().resource::<Simulation>().matrix(kind).to_vec())
            .collect();
        let zero = app.world_mut().spawn(Action::Zero(RuleKind::Swirl)).id();
        app.world_mut().trigger(Activate { entity: zero });
        assert!(
            app.world()
                .resource::<Simulation>()
                .matrix(RuleKind::Swirl)
                .iter()
                .all(|v| *v == 0.)
        );
        let randomize = app
            .world_mut()
            .spawn(Action::Randomize(RuleKind::Swirl))
            .id();
        app.world_mut().trigger(Activate { entity: randomize });
        assert!(
            app.world()
                .resource::<Simulation>()
                .matrix(RuleKind::Swirl)
                .iter()
                .any(|v| *v != 0.)
        );
        for kind in [
            RuleKind::Attraction,
            RuleKind::Alignment,
            RuleKind::Distance,
        ] {
            assert_eq!(
                app.world().resource::<Simulation>().matrix(kind).as_slice(),
                before[kind as usize].as_slice()
            );
        }
    }
    #[test]
    fn rules_edit_selected_matrix_and_keep_distance_nonnegative() {
        let mut s = Simulation::default();
        let mut a = Appearance::default();
        let c = Controls {
            selected: [(1, 2); 4],
            ..default()
        };
        let revision = s.rules_revision;
        let index = (c.selected[RuleKind::Distance as usize].0 * s.types
            + c.selected[RuleKind::Distance as usize].1) as usize;
        Field::Rule(RuleKind::Distance).set(-0.5, &mut s, &mut a, &c);
        assert_eq!(s.matrix(RuleKind::Distance)[index], 0.);
        assert_eq!(s.rules_revision, revision + 1);
        Field::Rule(RuleKind::Distance).set(1., &mut s, &mut a, &c);
        assert_eq!(s.matrix(RuleKind::Distance)[index], 0.95);
    }
}
