mod gpu;
mod interaction;
mod model;
#[cfg(test)]
mod trail_tests;
mod ui;

use bevy::{
    camera::{CameraOutputMode, Hdr, Viewport, visibility::RenderLayers},
    post_process::bloom::Bloom,
    prelude::*,
    render::render_resource::{BlendState, TextureFormat, TextureUsages},
    window::{PresentMode, PrimaryWindow},
};
use gpu::{DisplayTarget, ParticleGpuPlugin};
use model::*;

fn main() {
    App::new()
        .insert_resource(ClearColor(Color::srgb(0.008, 0.012, 0.025)))
        .init_resource::<Simulation>()
        .init_resource::<Appearance>()
        .init_resource::<Status>()
        .add_plugins(DefaultPlugins.set(WindowPlugin {
            primary_window: Some(Window {
                title: "Particle Life · GPU playground".into(),
                resolution: (1440, 900).into(),
                present_mode: PresentMode::AutoVsync,
                ..default()
            }),
            ..default()
        }))
        .add_plugins((ui::ControlsPlugin, ParticleGpuPlugin))
        .add_systems(Startup, setup)
        .add_systems(Update, (resize.after(ui::UiSet), clock))
        .run();
}
#[derive(Component)]
struct Canvas;
#[derive(Component)]
struct SimulationCamera;
fn setup(
    mut commands: Commands,
    mut images: ResMut<Assets<Image>>,
    window: Single<&Window, With<PrimaryWindow>>,
) {
    commands.spawn((
        SimulationCamera,
        Hdr,
        Camera2d,
        Bloom {
            intensity: 0.15,
            ..Bloom::NATURAL
        },
    ));
    commands.spawn((
        Camera2d,
        Camera {
            order: 1,
            clear_color: ClearColorConfig::Custom(Color::NONE),
            output_mode: CameraOutputMode::Write {
                blend_state: Some(BlendState::PREMULTIPLIED_ALPHA_BLENDING),
                clear_color: ClearColorConfig::None,
            },
            ..default()
        },
        RenderLayers::layer(1),
        IsDefaultUiCamera,
    ));
    let image = images.add(target_image(
        window.physical_width().max(1),
        window.physical_height().max(1),
    ));
    commands.spawn((
        Sprite {
            image: image.clone(),
            custom_size: Some(Vec2::new(window.width(), window.height())),
            ..default()
        },
        Canvas,
    ));
    commands.insert_resource(DisplayTarget(image));
}
fn target_image(width: u32, height: u32) -> Image {
    let mut image = Image::new_target_texture(width, height, TextureFormat::Rgba16Float, None);
    image.texture_descriptor.usage =
        TextureUsages::RENDER_ATTACHMENT | TextureUsages::TEXTURE_BINDING | TextureUsages::COPY_DST;
    image
}
#[allow(clippy::too_many_arguments)]
fn resize(
    window: Single<&Window, With<PrimaryWindow>>,
    mut images: ResMut<Assets<Image>>,
    target: Res<DisplayTarget>,
    mut canvas: Single<&mut Sprite, With<Canvas>>,
    mut appearance: ResMut<Appearance>,
    mut bloom: Single<&mut Bloom>,
    mut camera: Single<&mut Camera, With<SimulationCamera>>,
    controls: Res<ui::Controls>,
) {
    let (origin, size) = ui::viewport_geometry(
        window.physical_size(),
        window.scale_factor(),
        controls.visible,
    );
    camera.viewport = Some(Viewport {
        physical_position: origin,
        physical_size: size,
        ..default()
    });
    let width = size.x;
    let height = size.y;
    if images
        .get(&target.0)
        .is_some_and(|image| image.width() != width || image.height() != height)
        && let Some(mut image) = images.get_mut(&target.0)
    {
        *image = target_image(width, height);
    }
    canvas.custom_size = Some(size.as_vec2() / window.scale_factor());
    appearance.aspect = width as f32 / height as f32;
    bloom.intensity = appearance.glow * 0.6;
}
fn clock(time: Res<Time>, mut sim: ResMut<Simulation>) {
    sim.frame_dt = time.delta_secs();
}
