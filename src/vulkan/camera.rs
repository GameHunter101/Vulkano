use na::Vector3;
use nalgebra as na;

use crate::Buffer;

pub struct Camera {
    pub view_matrix: na::Matrix4<f32>,
    position: na::Vector3<f32>,
    down_direction: na::Unit<na::Vector3<f32>>,
    left_direction: na::Unit<na::Vector3<f32>>,
    rotation_matrix: na::Matrix3<f32>,
    yaw: f32,
    pitch: f32,
}

impl Default for Camera {
    fn default() -> Self {
        let mut cam = Camera {
            view_matrix: na::Matrix4::identity(),
            position: na::Vector3::new(0.0, 0.0, -5.0),
            down_direction: na::Unit::new_normalize(na::Vector3::new(0.0, 0.0, -1.0)),
            left_direction: na::Unit::new_normalize(na::Vector3::new(1.0, 0.0, 0.0)),
            rotation_matrix: na::Matrix3::identity(),
            pitch: 0.0,
            yaw: 0.0,
        };
        cam.update_view_matrix();
        cam
    }
}

#[allow(dead_code)]
impl Camera {
    pub fn update_buffer(&self, buffer: &mut Buffer, screen_width: u32, screen_height: u32) {
        let data: [[[f32; 4]; 4]; 2] = [
            self.view_matrix.into(),
            [[screen_width as f32, screen_height as f32, 0.0, 0.0]; 4],
        ];
        // dbg!(data);
        buffer.fill(&data);
    }

    fn update_view_matrix(&mut self) {
        let rotation_matrix = self.rotation_matrix.to_homogeneous();

        let translation_matrix = na::Matrix4::from(na::Translation3::from(self.position));
        let affine_matrix = translation_matrix * rotation_matrix;

        /* self.left_direction = na::Unit::new_normalize(
            na::Vector3::new(0.0, -1.0, 0.0).cross(self.view_direction.as_ref()),
        );
        self.down_direction =
            na::Unit::new_normalize(self.left_direction.cross(self.view_direction.as_ref())); */
        self.view_matrix = affine_matrix;
    }

    pub fn update_rotation(&mut self, mouse_delta: [f64; 2]) {
        let sensitvity = 0.01;
        self.yaw += mouse_delta[0] as f32 * sensitvity;
        self.pitch -= mouse_delta[1] as f32 * sensitvity;

        /* let lr_axis_rotation = na::Rotation3::from_axis_angle(
            &na::Unit::new_normalize(na::Vector3::new(1.0, 0.0, 0.0)),
            self.pitch,
        );
        let up_axis_rotation = na::Rotation3::from_axis_angle(
            &na::Unit::new_normalize(na::Vector3::new(0.0, 1.0, 0.0)),
            self.yaw,
        ); */

        let yaw_rotation = na::Matrix3::new(
            self.yaw.cos(),
            0.0,
            self.yaw.sin(),
            0.0,
            1.0,
            0.0,
            -self.yaw.sin(),
            0.0,
            self.yaw.cos(),
        );

        let pitch_rotation = na::Matrix3::new(
            1.0,
            0.0,
            0.0,
            0.0,
            self.pitch.cos(),
            -self.pitch.sin(),
            0.0,
            self.pitch.sin(),
            self.pitch.cos(),
        );

        self.rotation_matrix = yaw_rotation * pitch_rotation;

        // dbg!(self.matrix_to_euler_angles(self.rotation_matrix));

        self.update_view_matrix();
    }

    fn matrix_to_euler_angles(&self, matrix: na::Matrix3<f32>) -> [f32; 3] {
        let pitch = matrix[(2, 0)].asin();
        let roll = matrix[(2, 1)].atan2(matrix[(2, 2)]);
        let yaw = matrix[(1, 0)].atan2(matrix[(0, 0)]);

        [pitch, roll, yaw]
    }

    pub fn move_forward(&mut self, distance: f32) {
        // self.position += distance * self.view_direction.as_ref();
        self.position += distance * (self.rotation_matrix * na::Vector3::new(0.0, 0.0, 1.0));
        self.update_view_matrix();
    }

    pub fn move_backward(&mut self, distance: f32) {
        self.move_forward(-distance);
    }

    pub fn move_left(&mut self, distance: f32) {
        let rotation_matrix = na::Matrix3::new(0.0, 0.0, -1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0);
        /* let rotated_view = rotation_matrix * self.rotation_matrix;
        self.position += distance * rotated_view; */
        self.position +=
            distance * (rotation_matrix * self.rotation_matrix * na::Vector3::new(0.0, 0.0, 1.0));
        self.update_view_matrix();
    }

    pub fn move_right(&mut self, distance: f32) {
        self.move_left(-distance);
    }

    pub fn turn_right(&mut self, angle: f32) {
        // let rotation = na::Rotation3::from_axis_angle(&self.down_direction, angle);
        let rotation_matrix = na::Matrix3::new(
            angle.cos(),
            0.0,
            angle.sin(),
            0.0,
            1.0,
            0.0,
            -angle.sin(),
            0.0,
            angle.cos(),
        );
        self.rotation_matrix *= rotation_matrix;
        self.update_view_matrix();
    }

    pub fn turn_left(&mut self, angle: f32) {
        self.turn_right(-angle);
    }

    pub fn turn_up(&mut self, angle: f32) {
        let rotation_matrix = na::Matrix3::new(
            1.0,
            0.0,
            0.0,
            0.0,
            angle.cos(),
            -angle.sin(),
            0.0,
            angle.sin(),
            angle.cos(),
        );
        self.rotation_matrix *= rotation_matrix;
        self.update_view_matrix();
    }

    pub fn turn_down(&mut self, angle: f32) {
        self.turn_up(-angle);
    }
}
