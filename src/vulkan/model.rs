use std::mem::size_of;

use ash::vk;
use nalgebra as na;

use crate::Buffer;

use super::pipeline;

pub struct Model<V, I> {
    vertex_data: Vec<V>,
    index_data: Vec<u32>,
    handle_to_index: std::collections::HashMap<usize, usize>,
    handles: Vec<usize>,
    pub instances: Vec<I>,
    first_invisible: usize,
    next_handle: usize,
    pub vertex_buffer: Option<Buffer>,
    pub index_buffer: Option<Buffer>,
    pub instance_buffer: Option<Buffer>,
}

#[allow(dead_code)]
impl<V, I> Model<V, I> {
    fn get(&self, handle: usize) -> Option<&I> {
        if let Some(&index) = self.handle_to_index.get(&handle) {
            self.instances.get(index)
        } else {
            None
        }
    }

    fn get_mut(&mut self, handle: usize) -> Option<&mut I> {
        if let Some(&index) = self.handle_to_index.get(&handle) {
            self.instances.get_mut(index)
        } else {
            None
        }
    }

    fn swap_by_handle(&mut self, handle_1: usize, handle_2: usize) -> Result<(), InvalidHandle> {
        if handle_1 == handle_2 {
            return Ok(());
        }
        if let (Some(&index_1), Some(&index_2)) = (
            self.handle_to_index.get(&handle_1),
            self.handle_to_index.get(&handle_2),
        ) {
            self.handles.swap(index_1, index_2);
            self.instances.swap(index_1, index_2);
            self.handle_to_index.insert(index_1, handle_2);
            self.handle_to_index.insert(index_2, handle_1);
            Ok(())
        } else {
            Err(InvalidHandle)
        }
    }

    fn swap_by_index(&mut self, index_1: usize, index_2: usize) {
        if index_1 == index_2 {
            return;
        }
        let handle_1 = self.handles[index_1];
        let handle_2 = self.handles[index_2];
        self.handles.swap(index_1, index_2);
        self.instances.swap(index_1, index_2);
        self.handle_to_index.insert(index_1, handle_2);
        self.handle_to_index.insert(index_2, handle_1);
    }

    fn is_visible(&self, handle: usize) -> Result<bool, InvalidHandle> {
        if let Some(index) = self.handle_to_index.get(&handle) {
            Ok(index < &self.first_invisible)
        } else {
            Err(InvalidHandle)
        }
    }

    fn make_visible(&mut self, handle: usize) -> Result<(), InvalidHandle> {
        if let Some(&index) = self.handle_to_index.get(&handle) {
            if index < self.first_invisible {
                return Ok(());
            }
            self.swap_by_index(index, self.first_invisible);
            self.first_invisible += 1;
            Ok(())
        } else {
            Err(InvalidHandle)
        }
    }

    fn make_invisible(&mut self, handle: &usize) -> Result<(), InvalidHandle> {
        if let Some(&index) = self.handle_to_index.get(&handle) {
            if index >= self.first_invisible {
                return Ok(());
            }
            self.swap_by_index(index, self.first_invisible - 1);
            self.first_invisible -= 1;
            Ok(())
        } else {
            Err(InvalidHandle)
        }
    }

    fn insert(&mut self, element: I) -> usize {
        let handle = self.next_handle;
        self.next_handle += 1;
        let index = self.instances.len();
        self.instances.push(element);
        self.handles.push(handle);
        self.handle_to_index.insert(handle, index);
        handle
    }

    pub fn insert_visibly(&mut self, element: I) -> usize {
        let new_handle = self.insert(element);
        self.make_visible(new_handle).ok();
        new_handle
    }

    fn remove(&mut self, handle: usize) -> Result<I, InvalidHandle> {
        if let Some(&index) = self.handle_to_index.get(&handle) {
            if index < self.first_invisible {
                self.swap_by_index(index, self.first_invisible - 1);
                self.first_invisible -= 1;
            }
            self.swap_by_index(self.first_invisible, self.instances.len() - 1);
            self.handles.pop();
            self.handle_to_index.remove(&handle);

            Ok(self.instances.pop().unwrap())
        } else {
            Err(InvalidHandle)
        }
    }

    pub fn update_vertex_buffer(
        &mut self,
        logical_device: &ash::Device,
        allocator: &mut gpu_allocator::vulkan::Allocator,
    ) -> Result<(), gpu_allocator::AllocationError> {
        if let Some(buffer) = &mut self.vertex_buffer {
            buffer.fill(&self.vertex_data);
            Ok(())
        } else {
            let bytes = (size_of::<V>() * self.vertex_data.len()) as u64;
            let mut buffer = Buffer::new(
                logical_device,
                allocator,
                bytes,
                vk::BufferUsageFlags::VERTEX_BUFFER,
            );
            buffer.fill(&self.vertex_data);
            self.vertex_buffer = Some(buffer);
            Ok(())
        }
    }

    pub fn update_index_buffer(
        &mut self,
        logical_device: &ash::Device,
        allocator: &mut gpu_allocator::vulkan::Allocator,
    ) -> Result<(), gpu_allocator::AllocationError> {
        if let Some(buffer) = &mut self.index_buffer {
            buffer.fill(&self.index_data);
            Ok(())
        } else {
            let bytes = (self.index_data.len() * size_of::<u32>()) as u64;
            let mut buffer = Buffer::new(
                logical_device,
                allocator,
                bytes,
                vk::BufferUsageFlags::INDEX_BUFFER,
            );
            buffer.fill(&self.index_data);
            self.index_buffer = Some(buffer);
            Ok(())
        }
    }

    pub fn update_instance_buffer(
        &mut self,
        logical_device: &ash::Device,
        allocator: &mut gpu_allocator::vulkan::Allocator,
    ) -> Result<(), gpu_allocator::AllocationError> {
        if let Some(buffer) = &mut self.instance_buffer {
            buffer.fill(&self.instances[0..self.first_invisible]);
            Ok(())
        } else {
            let bytes = (self.first_invisible * size_of::<I>()) as u64;
            let mut buffer = Buffer::new(
                logical_device,
                allocator,
                bytes,
                vk::BufferUsageFlags::VERTEX_BUFFER,
            );
            buffer.fill(&self.instances[0..self.first_invisible]);
            self.instance_buffer = Some(buffer);
            Ok(())
        }
    }

    pub fn draw(&self, logical_device: &ash::Device, command_buffer: vk::CommandBuffer) {
        if let Some(vertex_buffer) = &self.vertex_buffer {
            if let Some(instance_buffer) = &self.instance_buffer {
                if let Some(index_buffer) = &self.index_buffer {
                    if self.first_invisible > 0 {
                        unsafe {
                            logical_device.cmd_bind_vertex_buffers(
                                command_buffer,
                                0,
                                &[vertex_buffer.buffer],
                                &[0],
                            );
                            logical_device.cmd_bind_vertex_buffers(
                                command_buffer,
                                1,
                                &[instance_buffer.buffer],
                                &[0],
                            );
                            logical_device.cmd_bind_index_buffer(
                                command_buffer,
                                index_buffer.buffer,
                                0,
                                vk::IndexType::UINT32,
                            );
                            logical_device.cmd_draw_indexed(
                                command_buffer,
                                self.index_data.len() as u32,
                                self.first_invisible as u32,
                                0,
                                0,
                                0,
                            );
                        }
                    }
                }
            }
        }
    }

    pub fn cleanup(
        &mut self,
        logical_device: &ash::Device,
        allocator: &mut gpu_allocator::vulkan::Allocator,
    ) {
        if let Some(vertex_buffer) = &mut self.vertex_buffer {
            vertex_buffer.cleanup(logical_device, allocator);
        }
        if let Some(index_buffer) = &mut self.instance_buffer {
            index_buffer.cleanup(logical_device, allocator);
        }
        if let Some(index_buffer) = &mut self.index_buffer {
            index_buffer.cleanup(logical_device, allocator);
        }
    }
}

#[repr(C)]
pub struct InstanceData {
    pub model_matrix: [[f32; 4]; 4],
    pub inverse_model_matrix: [[f32; 4]; 4],
    pub color: [f32; 3],
    pub metallic: f32,
    pub roughness: f32,
}

#[allow(dead_code)]
impl InstanceData {
    pub fn from_matrix_and_properties(
        model_matrix: na::Matrix4<f32>,
        color: [f32; 3],
        metallic: f32,
        roughness: f32,
    ) -> InstanceData {
        InstanceData {
            model_matrix: model_matrix.into(),
            inverse_model_matrix: model_matrix.try_inverse().unwrap().into(),
            color,
            metallic,
            roughness,
        }
    }

    pub fn screen_quad(
        view_matrix: na::Matrix4<f32>,
    ) -> InstanceData {
        InstanceData {
            model_matrix: na::Matrix4::identity().into(),
            inverse_model_matrix: na::Matrix4::identity().into(),
            color: [0.0, 0.0, 0.0],
            metallic: 0.0,
            roughness: 0.0,
        }
    }
}

#[repr(C)]
#[derive(Copy, Clone, Debug)]
pub struct VertexData {
    pub position: [f32; 3],
    pub normal: [f32; 3],
}

impl VertexData {
    fn midpoint(a: &VertexData, b: &VertexData) -> VertexData {
        VertexData {
            position: [
                0.5 * (a.position[0] + b.position[0]),
                0.5 * (a.position[1] + b.position[1]),
                0.5 * (a.position[2] + b.position[2]),
            ],
            normal: normalize([
                0.5 * (a.normal[0] + b.normal[0]),
                0.5 * (a.normal[1] + b.normal[1]),
                0.5 * (a.normal[2] + b.normal[2]),
            ]),
        }
    }
}

fn normalize(v: [f32; 3]) -> [f32; 3] {
    let length = (v[0].powi(2) + v[1].powi(2) + v[2].powi(2)).sqrt();
    let normalized = [v[0] / length, v[1] / length, v[2] / length];
    normalized
}

#[allow(dead_code)]
impl Model<VertexData, InstanceData> {
    pub fn cube() -> Model<[f32; 3], InstanceData> {
        let lbf = [-1.0, 1.0, 0.0]; //lbf: left-bottom-front
        let lbb = [-1.0, 1.0, 2.0];
        let ltf = [-1.0, -1.0, 0.0];
        let ltb = [-1.0, -1.0, 2.0];
        let rbf = [1.0, 1.0, 0.0];
        let rbb = [1.0, 1.0, 2.0];
        let rtf = [1.0, -1.0, 0.0];
        let rtb = [1.0, -1.0, 2.0];
        Model {
            vertex_data: vec![lbf, lbb, ltf, ltb, rbf, rbb, rtf, rtb],
            index_data: vec![
                0, 1, 5, 0, 5, 4, //bottom
                2, 7, 3, 2, 6, 7, //top
                0, 6, 2, 0, 4, 6, //front
                1, 3, 7, 1, 7, 5, //back
                0, 2, 1, 1, 2, 3, //left
                4, 5, 6, 5, 7, 6, //right
            ],
            handle_to_index: std::collections::HashMap::new(),
            handles: Vec::new(),
            instances: Vec::new(),
            first_invisible: 0,
            next_handle: 0,
            vertex_buffer: None,
            index_buffer: None,
            instance_buffer: None,
        }
    }

    pub fn icosahedron() -> Model<VertexData, InstanceData> {
        let phi = (1.0 + 5.0_f32.sqrt()) / 2.0;
        let darkgreen_front_top = VertexData {
            position: [phi, -1.0, 0.0],
            normal: normalize([phi, -1.0, 0.0]),
        }; //0
        let darkgreen_front_bottom = VertexData {
            position: [phi, 1.0, 0.0],
            normal: normalize([phi, 1.0, 0.0]),
        }; //1
        let darkgreen_back_top = VertexData {
            position: [-phi, -1.0, 0.0],
            normal: normalize([-phi, -1.0, 0.0]),
        }; //2
        let darkgreen_back_bottom = VertexData {
            position: [-phi, 1.0, 0.0],
            normal: normalize([-phi, 1.0, 0.0]),
        }; //3
        let lightgreen_front_right = VertexData {
            position: [1.0, 0.0, -phi],
            normal: normalize([1.0, 0.0, -phi]),
        }; //4
        let lightgreen_front_left = VertexData {
            position: [-1.0, 0.0, -phi],
            normal: normalize([-1.0, 0.0, -phi]),
        }; //5
        let lightgreen_back_right = VertexData {
            position: [1.0, 0.0, phi],
            normal: normalize([1.0, 0.0, phi]),
        }; //6
        let lightgreen_back_left = VertexData {
            position: [-1.0, 0.0, phi],
            normal: normalize([-1.0, 0.0, phi]),
        }; //7
        let purple_top_left = VertexData {
            position: [0.0, -phi, -1.0],
            normal: normalize([0.0, -phi, -1.0]),
        }; //8
        let purple_top_right = VertexData {
            position: [0.0, -phi, 1.0],
            normal: normalize([0.0, -phi, 1.0]),
        }; //9
        let purple_bottom_left = VertexData {
            position: [0.0, phi, -1.0],
            normal: normalize([0.0, phi, -1.0]),
        }; //10
        let purple_bottom_right = VertexData {
            position: [0.0, phi, 1.0],
            normal: normalize([0.0, phi, 1.0]),
        }; //11
        Model {
            vertex_data: vec![
                darkgreen_front_top,
                darkgreen_front_bottom,
                darkgreen_back_top,
                darkgreen_back_bottom,
                lightgreen_front_right,
                lightgreen_front_left,
                lightgreen_back_right,
                lightgreen_back_left,
                purple_top_left,
                purple_top_right,
                purple_bottom_left,
                purple_bottom_right,
            ],
            index_data: vec![
                0, 9, 8, //
                0, 8, 4, //
                0, 4, 1, //
                0, 1, 6, //
                0, 6, 9, //
                8, 9, 2, //
                8, 2, 5, //
                8, 5, 4, //
                4, 5, 10, //
                4, 10, 1, //
                1, 10, 11, //
                1, 11, 6, //
                2, 3, 5, //
                2, 7, 3, //
                2, 9, 7, //
                5, 3, 10, //
                3, 11, 10, //
                3, 7, 11, //
                6, 7, 9, //
                6, 11, 7, //
            ],
            handle_to_index: std::collections::HashMap::new(),
            handles: Vec::new(),
            instances: Vec::new(),
            first_invisible: 0,
            next_handle: 0,
            vertex_buffer: None,
            index_buffer: None,
            instance_buffer: None,
        }
    }

    pub fn sphere(refinements: u32) -> Model<VertexData, InstanceData> {
        let mut model = Model::icosahedron();
        for _ in 0..refinements {
            model.refine();
        }
        for vert in &mut model.vertex_data {
            vert.position = normalize(vert.position);
        }
        model
    }

    pub fn refine(&mut self) {
        let mut new_indices = vec![];
        let mut midpoints = std::collections::HashMap::<(u32, u32), u32>::new();
        for triangle in self.index_data.chunks(3) {
            let a = triangle[0];
            let b = triangle[1];
            let c = triangle[2];
            let vertex_a = self.vertex_data[a as usize];
            let vertex_b = self.vertex_data[b as usize];
            let vertex_c = self.vertex_data[c as usize];
            let mab = if let Some(ab) = midpoints.get(&(a, b)) {
                *ab
            } else {
                let vertex_ab = VertexData::midpoint(&vertex_a, &vertex_b);
                let mab = self.vertex_data.len() as u32;
                self.vertex_data.push(vertex_ab);
                midpoints.insert((a, b), mab);
                midpoints.insert((b, a), mab);
                mab
            };
            let mbc = if let Some(bc) = midpoints.get(&(b, c)) {
                *bc
            } else {
                let vertex_bc = VertexData::midpoint(&vertex_b, &vertex_c);
                let mbc = self.vertex_data.len() as u32;
                midpoints.insert((b, c), mbc);
                midpoints.insert((c, b), mbc);
                self.vertex_data.push(vertex_bc);
                mbc
            };
            let mca = if let Some(ca) = midpoints.get(&(c, a)) {
                *ca
            } else {
                let vertex_ca = VertexData::midpoint(&vertex_c, &vertex_a);
                let mca = self.vertex_data.len() as u32;
                midpoints.insert((c, a), mca);
                midpoints.insert((a, c), mca);
                self.vertex_data.push(vertex_ca);
                mca
            };
            new_indices.extend_from_slice(&[mca, a, mab, mab, b, mbc, mbc, c, mca, mab, mbc, mca]);
        }
        self.index_data = new_indices;
    }

    pub fn screen_quad() -> Model<VertexData, InstanceData> {
        let lb = VertexData {
            position: [-1.0, 1.0, 0.0],
            normal: [-1.0, 1.0, 0.0],
        }; //lb: left-bottom
        let lt = VertexData {
            position: [-1.0, -1.0, 0.0],
            normal: [-1.0, -1.0, 0.0],
        };
        let rb = VertexData {
            position: [1.0, 1.0, 0.0],
            normal: normalize([1.0, 1.0, 0.0]),
        };
        let rt = VertexData {
            position: [1.0, -1.0, 0.0],
            normal: [1.0, -1.0, 0.0],
        };
        Model {
            vertex_data: vec![lb, lt, rb, rt],
            index_data: vec![0, 2, 1, 1, 2, 3],
            handle_to_index: std::collections::HashMap::new(),
            handles: Vec::new(),
            instances: Vec::new(),
            first_invisible: 0,
            next_handle: 0,
            vertex_buffer: None,
            index_buffer: None,
            instance_buffer: None,
        }
    }
}

#[derive(Debug, Clone)]
struct InvalidHandle;
impl std::fmt::Display for InvalidHandle {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Invalid handle")
    }
}
impl std::error::Error for InvalidHandle {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        None
    }
}

#[derive(Clone, Copy, Debug)]
#[repr(C)]
pub struct TexturedVertexData {
    pub position: [f32; 3],
    pub tex_coord: [f32; 2],
}

#[repr(C)]
#[derive(Debug)]
pub struct TexturedInstanceData {
    pub model_matrix: [[f32; 4]; 4],
    pub inverse_model_matrix: [[f32; 4]; 4],
    pub texture_id: u32,
}

#[allow(dead_code)]
impl TexturedInstanceData {
    pub fn from_matrix(model_matrix: na::Matrix4<f32>) -> TexturedInstanceData {
        TexturedInstanceData {
            model_matrix: model_matrix.into(),
            inverse_model_matrix: model_matrix.try_inverse().unwrap().into(),
            texture_id: 0,
        }
    }

    pub fn from_matrix_and_texture(
        model_matrix: na::Matrix4<f32>,
        texture_id: usize,
    ) -> TexturedInstanceData {
        TexturedInstanceData {
            model_matrix: model_matrix.into(),
            inverse_model_matrix: model_matrix.try_inverse().unwrap().into(),
            texture_id: texture_id as u32,
        }
    }
}

impl Model<TexturedVertexData, TexturedInstanceData> {
    pub fn quad() -> Self {
        let lb = TexturedVertexData {
            position: [-1.0, 1.0, 0.0],
            tex_coord: [0.0, 1.0],
        }; //lb: left-bottom
        let lt = TexturedVertexData {
            position: [-1.0, -1.0, 0.0],
            tex_coord: [0.0, 0.0],
        };
        let rb = TexturedVertexData {
            position: [1.0, 1.0, 0.0],
            tex_coord: [1.0, 1.0],
        };
        let rt = TexturedVertexData {
            position: [1.0, -1.0, 0.0],
            tex_coord: [1.0, 0.0],
        };
        Model {
            vertex_data: vec![lb, lt, rb, rt],
            index_data: vec![0, 2, 1, 1, 2, 3],
            handle_to_index: std::collections::HashMap::new(),
            handles: Vec::new(),
            instances: Vec::new(),
            first_invisible: 0,
            next_handle: 0,
            vertex_buffer: None,
            index_buffer: None,
            instance_buffer: None,
        }
    }
}

pub enum ModelTypes {
    Textured(Model<TexturedVertexData, TexturedInstanceData>),
    Normal(Model<VertexData, InstanceData>),
}
