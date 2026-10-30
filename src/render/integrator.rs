use super::image::Image;
use super::ray::intersect;
use crate::view_scene::ViewScene;
use crate::viewport::Camera;
use glam::{Vec2, Vec3};
use rayon::prelude::*;

#[derive(Clone, Copy, Debug)]
pub struct RenderSettings {
    pub width: u32,
    pub height: u32,
    pub samples: u32,
    pub max_depth: u32, // Max number of bounces for indirect lighting
}

// Simple pseudo-random number generator (PRNG) using xorshift
fn rand_f32(seed: &mut u32) -> f32 {
    *seed ^= *seed << 13;
    *seed ^= *seed >> 17;
    *seed ^= *seed << 5;
    (*seed as f32) / (u32::MAX as f32)
}

// A random unit vector in 3D space
fn random_unit_vector(seed: &mut u32) -> Vec3 {
    let theta = 2.0 * std::f32::consts::PI * rand_f32(seed);
    let z = 2.0 * rand_f32(seed) - 1.0;
    let r = (1.0 - z * z).sqrt();
    Vec3::new(r * theta.cos(), r * theta.sin(), z)
}

// Recursive function for Path Tracing
fn trace_path(scene: &ViewScene, ray: &super::ray::Ray, depth: u32, seed: &mut u32) -> Vec3 {
    if depth == 0 {
        return Vec3::ZERO;
    }

    if let Some(hit) = intersect(scene, ray) {
        let mut color = Vec3::ZERO;

        // 1. Direct lighting (hardcoded point light)
        let light_pos = Vec3::new(3.0, 5.0, 3.0);
        let light_color = Vec3::new(1.0, 1.0, 1.0);
        let to_light = (light_pos - hit.point).normalize();
        let diffuse = hit.normal.dot(to_light).max(0.0);
        color += hit.albedo * light_color * diffuse;

        // 2. Indirect lighting (bounces)
        let mut next_dir = random_unit_vector(seed);
        if next_dir.dot(hit.normal) < 0.0 {
            next_dir = -next_dir; // Secure the next direction to be in the hemisphere of the normal
        }
        let next_ray = super::ray::Ray {
            origin: hit.point,
            direction: next_dir,
        };

        // Avoid infinite recursion by probabilistically terminating the path based on the albedo
        let p = hit.albedo.max_element().clamp(0.1, 1.0);
        if rand_f32(seed) < p {
            color += hit.albedo * trace_path(scene, &next_ray, depth - 1, seed) / p;
        }

        return color;
    }

    // Sky color background when the ray doesn't hit anything
    Vec3::new(0.05, 0.05, 0.1)
}

pub fn render(scene: &ViewScene, camera: &Camera, settings: RenderSettings) -> Image {
    let mut image = Image::new(settings.width, settings.height);
    let aspect = settings.width as f32 / settings.height as f32;

    // Parallelise the rows of the image using Rayon
    let rows: Vec<Vec<Vec3>> = (0..settings.height)
        .into_par_iter()
        .map(|y| {
            let mut row = vec![Vec3::ZERO; settings.width as usize];
            let mut seed = 12345 + y; // Unique seed per row

            for _ in 0..settings.samples {
                for x in 0..settings.width {
                    let jitter_x = rand_f32(&mut seed) - 0.5;
                    let jitter_y = rand_f32(&mut seed) - 0.5;

                    let ndc = Vec2::new(
                        2.0 * (x as f32 + 0.5 + jitter_x) / settings.width as f32 - 1.0,
                        1.0 - 2.0 * (y as f32 + 0.5 + jitter_y) / settings.height as f32,
                    );

                    let ray = match camera.ray(ndc, aspect) {
                        Ok(ray) => ray,
                        Err(_) => continue,
                    };

                    let color = trace_path(scene, &ray, settings.max_depth, &mut seed);
                    row[x as usize] += color;
                }
            }
            row
        })
        .collect();

    // Knock over the rows into the image pixels
    for (y, row) in rows.into_iter().enumerate() {
        for (x, color) in row.into_iter().enumerate() {
            image.pixels[y * settings.width as usize + x] = color;
        }
    }

    image
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::native_scene::NativeScene;
    use crate::scene::Primitive;
    use std::path::PathBuf;

    #[test]
    fn render_demo_cube() {
        let mut scene = NativeScene::default();

        // 1. Create the floor and scale it to 10x10
        let floor_id = scene.create(Primitive::Plane, "floor").unwrap();
        let floor_transform = crate::native_scene::NativeTransform {
            scale: [10.0, 1.0, 10.0],
            ..Default::default()
        };
        scene.set_transform(floor_id, floor_transform).unwrap();

        // 2. Create the cube and elevate it so it rests on the floor
        let cube_id = scene.create(Primitive::Cube, "my_cube").unwrap();
        let cube_transform = crate::native_scene::NativeTransform {
            translation: [0.0, 0.5, 0.0],
            ..Default::default()
        };
        scene.set_transform(cube_id, cube_transform).unwrap();

        // 3. Convert to ViewScene
        let view_scene = ViewScene::from_native(&scene, 1024 * 1024).unwrap();

        // 4. Configure the camera (closer to see the detail)
        let mut camera = Camera::default();
        // Move the camera to see the shadow well
        camera
            .set_pose(Vec3::new(0.0, 1.0, 0.0), 6.0, 0.5, 0.3)
            .unwrap();

        // 5. Render at 512x512 (taking advantage of Rayon)
        let settings = RenderSettings {
            width: 512,
            height: 512,
            samples: 50,
            max_depth: 3,
        };

        let image = render(&view_scene, &camera, settings);
        let path = PathBuf::from("demo_render_path_traced.ppm");
        image.save_ppm(&path, settings.samples).unwrap();

        println!(
            "Imagen Path Traced guardada en: {:?}",
            path.canonicalize().unwrap()
        );
    }
}
