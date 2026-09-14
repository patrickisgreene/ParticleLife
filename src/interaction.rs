//! Direct particle brushes and queued additions to the GPU population.
use crate::model::{Appearance, WORLD};
use bevy::{prelude::*, render::extract_resource::ExtractResource};

#[derive(Clone, Copy, Default, PartialEq, Eq, Debug)]
pub enum Tool {
    #[default]
    Navigate,
    Attract,
    Repel,
    Dump,
}
impl Tool {
    pub fn brush_sign(self) -> f32 {
        match self {
            Self::Attract => 1.,
            Self::Repel => -1.,
            _ => 0.,
        }
    }
}

#[derive(Clone)]
pub struct ParticleDump {
    pub serial: u64,
    pub start: u32,
    pub count: u32,
    pub center: Vec2,
    pub radius: f32,
}
impl ParticleDump {
    pub fn particles(&self, seed: u32, types: u32) -> Vec<u32> {
        let mut words = Vec::with_capacity(self.count as usize * 8);
        for i in 0..self.count {
            let key = crate::model::hash(seed ^ (self.serial as u32).wrapping_mul(1664525) ^ i);
            let angle = (key as f64 / (u32::MAX as f64 + 1.)) as f32 * std::f32::consts::TAU;
            let radius = ((crate::model::hash(key) as f64 / (u32::MAX as f64 + 1.)) as f32).sqrt()
                * self.radius;
            let position = (self.center + Vec2::new(angle.cos(), angle.sin()) * radius)
                .rem_euclid(Vec2::splat(WORLD));
            words.extend_from_slice(&[
                position.x.to_bits(),
                position.y.to_bits(),
                0,
                0,
                crate::model::hash(key ^ 0x9134abcd) % types,
                0,
                0,
                0,
            ]);
        }
        words
    }
}

#[derive(Resource, Clone, ExtractResource)]
pub struct InteractionTools {
    pub tool: Tool,
    pub radius: f32,
    pub strength: f32,
    pub cursor: Option<Vec2>,
    pub active: bool,
    pub rate: f32,
    pub pending: Vec<ParticleDump>,
    pub serial: u64,
    pub epoch: u64,
    pub error: String,
}
impl Default for InteractionTools {
    fn default() -> Self {
        Self {
            tool: Tool::Navigate,
            radius: 64.,
            strength: 1200.,
            cursor: None,
            active: false,
            rate: 2000.,
            pending: Vec::new(),
            serial: 0,
            epoch: 0,
            error: String::new(),
        }
    }
}
#[cfg(test)]
pub fn wrapped_delta(delta: Vec2) -> Vec2 {
    delta - (delta / WORLD).round() * WORLD
}
pub fn cursor_world(view: &Appearance, cursor: Vec2, viewport: Vec2) -> Vec2 {
    let normalized = cursor / viewport.max(Vec2::ONE) * 2. - Vec2::ONE;
    let camera = normalized * Vec2::new(view.aspect, -1.) / view.zoom + view.pan;
    ((camera + Vec2::ONE) * (WORLD * 0.5)).rem_euclid(Vec2::splat(WORLD))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn pointer_coordinates_match_camera_and_wrap() {
        let mut view = Appearance {
            aspect: 2.,
            ..default()
        };
        assert_eq!(
            cursor_world(&view, Vec2::new(200., 100.), Vec2::new(400., 200.)),
            Vec2::splat(WORLD / 2.)
        );
        view.pan = Vec2::new(0.9, -0.4);
        view.zoom = 3.;
        let cursor = Vec2::new(390., 20.);
        let size = Vec2::new(400., 200.);
        let before = cursor_world(&view, cursor, size);
        view.zoom_at_cursor(cursor, size, 24.);
        assert!(wrapped_delta(cursor_world(&view, cursor, size) - before).length() < 0.001);
    }
    #[test]
    fn dumped_particles_wrap_and_start_unassigned() {
        let dump = ParticleDump {
            serial: 3,
            start: 50,
            count: 1000,
            center: Vec2::new(WORLD - 1., 1.),
            radius: 20.,
        };
        let particles = dump.particles(42, 7);
        assert_eq!(particles.len(), 8000);
        for particle in particles.chunks_exact(8) {
            let position = Vec2::new(f32::from_bits(particle[0]), f32::from_bits(particle[1]));
            assert!(position.cmpge(Vec2::ZERO).all() && position.cmplt(Vec2::splat(WORLD)).all());
            assert!(wrapped_delta(position - dump.center).length() <= dump.radius + 0.001);
            assert!(particle[4] < 7);
            assert_eq!(&particle[5..], &[0, 0, 0]);
            assert_eq!(&particle[2..4], &[0, 0]);
        }
    }
}
