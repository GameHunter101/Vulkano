use ash::{vk, Entry, Instance};
use nalgebra as na;

use crate::Buffer;
use crate::Debug;
use crate::Pipeline;
use crate::Pools;
use crate::QueueFamilies;
use crate::Queues;
use crate::Surface;
use crate::Swapchain;
use crate::{
    create_commandbuffers, init_device_and_queues, init_instance,
    init_physical_device_and_properties, init_renderpass,
};

use crate::Model;

use super::model::InstanceData;
use super::model::ModelTypes;
use super::model::TexturedInstanceData;
use super::model::TexturedVertexData;
use super::model::VertexData;
use super::pipeline;
use super::text::AllText;
use super::texture::TextureStorage;

pub struct Vulkano {
    // pub window: winit::window::Window,
    _entry: Entry,
    pub instance: Instance,
    pub debug: std::mem::ManuallyDrop<Debug>,
    pub surfaces: std::mem::ManuallyDrop<Surface>,
    pub physical_device: vk::PhysicalDevice,
    _physical_device_properties: vk::PhysicalDeviceProperties,
    _physical_device_features: vk::PhysicalDeviceFeatures,
    queue_families: QueueFamilies,
    pub queues: Queues,
    pub device: ash::Device,
    pub swapchain: Swapchain,
    pub renderpass: vk::RenderPass,
    pub pipeline: Pipeline,
    pub pools: Pools,
    pub commandbuffers: Vec<vk::CommandBuffer>,
    pub models: Vec<ModelTypes>,
    pub screen_quad: Option<Model<VertexData, InstanceData>>,
    pub allocator: std::mem::ManuallyDrop<gpu_allocator::vulkan::Allocator>,
    pub uniform_buffer: Buffer,
    descriptor_pool: vk::DescriptorPool,
    camera_descriptor_sets: Vec<vk::DescriptorSet>,
    pub lights_descriptor_sets: Vec<vk::DescriptorSet>,
    pub descriptor_sets_texture: Vec<vk::DescriptorSet>,
    pub light_buffer: Buffer,
    pub texture_storage: TextureStorage,
    pub text: AllText,
}

impl Vulkano {
    pub fn init(window: &winit::window::Window) -> Result<Vulkano, Box<dyn std::error::Error>> {
        let entry = unsafe { Entry::load()? };

        let layer_names = vec!["VK_LAYER_KHRONOS_validation"];
        let instance = init_instance(&entry, &layer_names)?;
        let debug = Debug::init(&entry, &instance)?;
        let surfaces = Surface::init(window, &entry, &instance)?;

        let (physical_device, physical_device_properties, physical_device_features) =
            init_physical_device_and_properties(&instance)?;

        let queue_families = QueueFamilies::init(&instance, physical_device, &surfaces)?;

        let (logical_device, queues) =
            init_device_and_queues(&instance, physical_device, &queue_families, &layer_names)?;
        let mut swapchain = Swapchain::init(
            &instance,
            &surfaces,
            physical_device,
            &logical_device,
            &queue_families,
        )?;
        let renderpass = init_renderpass(&logical_device, swapchain.surface_format.format)?;
        swapchain.create_framebuffers(&logical_device, renderpass)?;

        // let pipeline = Pipeline::init(&logical_device, &swapchain, &renderpass)?;
        let pools = Pools::init(&logical_device, &queue_families)?;

        let allocator_create_description = gpu_allocator::vulkan::AllocatorCreateDesc {
            instance: instance.clone(),
            device: logical_device.clone(),
            physical_device,
            debug_settings: Default::default(),
            buffer_device_address: false,
        };

        let mut allocator = gpu_allocator::vulkan::Allocator::new(&allocator_create_description)
            .expect("Failed to create allocator");

        let commandbuffers =
            create_commandbuffers(&logical_device, &pools, swapchain.framebuffers.len())?;

        let mut uniform_buffer = Buffer::new(
            &logical_device,
            &mut allocator,
            192,
            vk::BufferUsageFlags::UNIFORM_BUFFER,
        );
        let mut light_buffer = Buffer::new(
            &logical_device,
            &mut allocator,
            144,
            vk::BufferUsageFlags::STORAGE_BUFFER,
        );

        let camera_transform: [[[f32; 4]; 4]; 3] = [
            na::Matrix4::identity().into(),
            na::Matrix4::identity().into(),
            [[
                swapchain.extent.width as f32,
                swapchain.extent.height as f32,
                0.0,
                0.0,
            ]; 4],
        ];

        uniform_buffer.fill(&camera_transform);
        light_buffer.fill(&[0.0, 0.0]);

        let pool_sizes = [
            vk::DescriptorPoolSize {
                ty: vk::DescriptorType::UNIFORM_BUFFER,
                descriptor_count: swapchain.amount_of_images,
            },
            vk::DescriptorPoolSize {
                ty: vk::DescriptorType::STORAGE_BUFFER,
                descriptor_count: swapchain.amount_of_images,
            },
            /* vk::DescriptorPoolSize {
                ty: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
                descriptor_count: pipeline::MAXIMAL_NUMBER_OF_TEXTURES * swapchain.amount_of_images,
            }, */
        ];

        let descriptor_pool_info = vk::DescriptorPoolCreateInfo::builder()
            .flags(vk::DescriptorPoolCreateFlags::UPDATE_AFTER_BIND_EXT)
            .max_sets(pipeline::MAXIMAL_NUMBER_OF_TEXTURES * pool_sizes.len() as u32)
            .pool_sizes(&pool_sizes);
        let descriptor_pool =
            unsafe { logical_device.create_descriptor_pool(&descriptor_pool_info, None) }?;

        let pipeline = Pipeline::init(&logical_device, &swapchain, &renderpass)?;

        let desc_layouts_camera =
            vec![pipeline.descriptor_set_layouts[0]; swapchain.amount_of_images as usize];
        let descriptor_set_allocate_info_camera = vk::DescriptorSetAllocateInfo::builder()
            .descriptor_pool(descriptor_pool)
            .set_layouts(&desc_layouts_camera);
        let camera_descriptor_sets = unsafe {
            logical_device.allocate_descriptor_sets(&descriptor_set_allocate_info_camera)
        }?;

        for descset in &camera_descriptor_sets {
            let buffer_infos = [vk::DescriptorBufferInfo {
                buffer: uniform_buffer.buffer,
                offset: 0,
                range: 192,
            }];
            let desc_sets_write = [vk::WriteDescriptorSet::builder()
                .dst_set(*descset)
                .dst_binding(0)
                .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                .buffer_info(&buffer_infos)
                .build()];
            unsafe { logical_device.update_descriptor_sets(&desc_sets_write, &[]) };
        }

        let desc_layouts_light =
            vec![pipeline.descriptor_set_layouts[1]; swapchain.amount_of_images as usize];
        let descriptor_set_allocate_info_light = vk::DescriptorSetAllocateInfo::builder()
            .descriptor_pool(descriptor_pool)
            .set_layouts(&desc_layouts_light);
        let lights_descriptor_sets = unsafe {
            logical_device.allocate_descriptor_sets(&descriptor_set_allocate_info_light)
        }?;

        for descset in &lights_descriptor_sets {
            let buffer_infos = [vk::DescriptorBufferInfo {
                buffer: light_buffer.buffer,
                offset: 0,
                range: 8,
            }];
            let desc_sets_write = [vk::WriteDescriptorSet::builder()
                .dst_set(*descset)
                .dst_binding(0)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .buffer_info(&buffer_infos)
                .build()];
            unsafe { logical_device.update_descriptor_sets(&desc_sets_write, &[]) };
        }

        /* let desc_layouts_texture =
            vec![pipeline.descriptor_set_layouts[1]; swapchain.amount_of_images as usize];
        let descriptor_counts =
            vec![pipeline::MAXIMAL_NUMBER_OF_TEXTURES; swapchain.amount_of_images as usize];
        let mut count_info = vk::DescriptorSetVariableDescriptorCountAllocateInfoEXT::builder()
            .descriptor_counts(&descriptor_counts);
        let descriptor_set_allocate_info_texture = vk::DescriptorSetAllocateInfo::builder()
            .descriptor_pool(descriptor_pool)
            .set_layouts(&desc_layouts_texture)
            .push_next(&mut count_info);

        let descriptor_sets_texture = unsafe {
            logical_device.allocate_descriptor_sets(&descriptor_set_allocate_info_texture)
        }?; */

        let text = AllText::new("./fonts/ARIAL.TTF")?;

        Ok(Vulkano {
            // window,
            _entry: entry,
            instance,
            debug: std::mem::ManuallyDrop::new(debug),
            surfaces: std::mem::ManuallyDrop::new(surfaces),
            physical_device,
            _physical_device_properties: physical_device_properties,
            _physical_device_features: physical_device_features,
            queue_families,
            queues,
            device: logical_device,
            swapchain,
            renderpass,
            pipeline,
            pools,
            commandbuffers,
            models: vec![],
            screen_quad: None,
            allocator: std::mem::ManuallyDrop::new(allocator),
            uniform_buffer,
            descriptor_pool,
            camera_descriptor_sets,
            lights_descriptor_sets,
            descriptor_sets_texture: vec![],
            light_buffer,
            texture_storage: TextureStorage::new(),
            text,
        })
    }

    pub fn update_command_buffer(&mut self, index: usize) -> Result<(), vk::Result> {
        let command_buffer = self.commandbuffers[index];
        let command_buffer_begin_info = vk::CommandBufferBeginInfo::builder();
        unsafe {
            self.device
                .begin_command_buffer(command_buffer, &command_buffer_begin_info)?;
        }
        let clear_values = [
            vk::ClearValue {
                color: vk::ClearColorValue {
                    float32: [0.2, 0.2, 0.2, 1.0],
                },
            },
            vk::ClearValue {
                depth_stencil: vk::ClearDepthStencilValue {
                    depth: 1.0,
                    stencil: 0,
                },
            },
        ];

        let renderpass_being_info = vk::RenderPassBeginInfo::builder()
            .render_pass(self.renderpass)
            .framebuffer(self.swapchain.framebuffers[index])
            .render_area(vk::Rect2D {
                offset: vk::Offset2D { x: 0, y: 0 },
                extent: self.swapchain.extent,
            })
            .clear_values(&clear_values);
        unsafe {
            self.device.cmd_begin_render_pass(
                command_buffer,
                &renderpass_being_info,
                vk::SubpassContents::INLINE,
            );
            self.device.cmd_bind_pipeline(
                command_buffer,
                vk::PipelineBindPoint::GRAPHICS,
                self.pipeline.pipeline,
            );
            self.device.cmd_bind_descriptor_sets(
                command_buffer,
                vk::PipelineBindPoint::GRAPHICS,
                self.pipeline.layout,
                0,
                &[
                    self.camera_descriptor_sets[index],
                    self.lights_descriptor_sets[index],
                    // self.descriptor_sets_texture[index],
                ],
                &[],
            );
            for model in &self.models {
                match model {
                    ModelTypes::Normal(normal) => normal.draw(&self.device, command_buffer),
                    ModelTypes::Textured(textured) => textured.draw(&self.device, command_buffer),
                }
            }
            if let Some(quad) = &self.screen_quad {
                quad.draw(&self.device, command_buffer);
            }
            // self.text.draw(&self.device, command_buffer, index);
            self.device.cmd_end_render_pass(command_buffer);
            self.device.end_command_buffer(command_buffer)?;
        }
        Ok(())
    }

    pub fn recreate_swapchain(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        unsafe {
            self.device
                .device_wait_idle()
                .expect("Something wrong while waiting");
        }
        unsafe {
            self.swapchain.cleanup(&self.device);
        }
        self.swapchain = Swapchain::init(
            &self.instance,
            &self.surfaces,
            self.physical_device,
            &self.device,
            &self.queue_families,
        )?;
        self.swapchain
            .create_framebuffers(&self.device, self.renderpass)?;
        self.pipeline.cleanup(&self.device);
        self.pipeline = Pipeline::init(&self.device, &self.swapchain, &self.renderpass)?;

        /* unsafe {
            self.device.reset_descriptor_pool(
                self.descriptor_pool,
                ash::vk::DescriptorPoolResetFlags::empty(),
            )?;
        }

        let desc_layouts_camera =
            vec![self.pipeline.descriptor_set_layouts[0]; self.swapchain.amount_of_images as usize];
        let descriptor_set_allocate_info_camera = vk::DescriptorSetAllocateInfo::builder()
            .descriptor_pool(self.descriptor_pool)
            .set_layouts(&desc_layouts_camera);
        let camera_descriptor_sets = unsafe {
            self.device.allocate_descriptor_sets(&descriptor_set_allocate_info_camera)
        }?;

        for descset in &camera_descriptor_sets {
            let buffer_infos = [vk::DescriptorBufferInfo {
                buffer: self.uniform_buffer.buffer,
                offset: 0,
                range: 192,
            }];
            let desc_sets_write = [vk::WriteDescriptorSet::builder()
                .dst_set(*descset)
                .dst_binding(0)
                .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                .buffer_info(&buffer_infos)
                .build()];
            unsafe { self.device.update_descriptor_sets(&desc_sets_write, &[]) };
        }

        let desc_layouts_light =
            vec![self.pipeline.descriptor_set_layouts[1]; self.swapchain.amount_of_images as usize];
        let descriptor_set_allocate_info_light = vk::DescriptorSetAllocateInfo::builder()
            .descriptor_pool(self.descriptor_pool)
            .set_layouts(&desc_layouts_light);
        let lights_descriptor_sets = unsafe {
            self.device.allocate_descriptor_sets(&descriptor_set_allocate_info_light)
        }?;

        for descset in &lights_descriptor_sets {
            let buffer_infos = [vk::DescriptorBufferInfo {
                buffer: self.light_buffer.buffer,
                offset: 0,
                range: 8,
            }];
            let desc_sets_write = [vk::WriteDescriptorSet::builder()
                .dst_set(*descset)
                .dst_binding(0)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .buffer_info(&buffer_infos)
                .build()];
            unsafe { self.device.update_descriptor_sets(&desc_sets_write, &[]) };
        } */

        /* let desc_layouts_texture =
            vec![self.pipeline.descriptor_set_layouts[1]; self.swapchain.amount_of_images as usize];
        let descriptor_counts =
            vec![pipeline::MAXIMAL_NUMBER_OF_TEXTURES; self.swapchain.amount_of_images as usize];
        let mut count_info = vk::DescriptorSetVariableDescriptorCountAllocateInfoEXT::builder()
            .descriptor_counts(&descriptor_counts);
        let descriptor_set_allocate_info_texture = vk::DescriptorSetAllocateInfo::builder()
            .descriptor_pool(self.descriptor_pool)
            .set_layouts(&desc_layouts_texture)
            .push_next(&mut count_info);

        self.descriptor_sets_texture = unsafe {
            self.device
                .allocate_descriptor_sets(&descriptor_set_allocate_info_texture)
        }?; */

        /* self.text.clear_pipeline(&self.device);
        self.text
            .update_textures(&self.swapchain, &self.device, &self.renderpass)?; */
        Ok(())
    }

    pub fn new_texture_from_file<P: AsRef<std::path::Path>>(
        &mut self,
        path: P,
    ) -> Result<usize, Box<dyn std::error::Error>> {
        self.texture_storage.new_texture_from_file(
            path,
            &self.instance,
            &self.device,
            self.physical_device,
            &mut self.allocator,
            self.pools.command_pool_graphics,
            self.queues.graphics_queue,
        )
    }
    fn cleanup_model<V, I>(&mut self, model: &mut Model<V, I>) {
        if let Some(vertex_buffer) = &mut model.vertex_buffer {
            vertex_buffer.cleanup(&self.device, &mut self.allocator);
        }
        if let Some(index_buffer) = &mut model.instance_buffer {
            index_buffer.cleanup(&self.device, &mut self.allocator);
        }
        if let Some(index_buffer) = &mut model.index_buffer {
            index_buffer.cleanup(&self.device, &mut self.allocator);
        }
    }
}

impl Drop for Vulkano {
    fn drop(&mut self) {
        unsafe {
            self.device
                .device_wait_idle()
                .expect("Something went wrong while waiting");
            self.text.cleanup(&self.device, &mut self.allocator);
            self.texture_storage.cleanup(&self.device);
            self.device
                .destroy_descriptor_pool(self.descriptor_pool, None);
            self.light_buffer.cleanup(&self.device, &mut self.allocator);
            self.uniform_buffer
                .cleanup(&self.device, &mut self.allocator);
            for model in &mut self.models {
                match model {
                    ModelTypes::Normal(normal) => normal.cleanup(&self.device, &mut self.allocator),
                    ModelTypes::Textured(textured) => {
                        textured.cleanup(&self.device, &mut self.allocator)
                    }
                }
            }
            if let Some(quad) = &mut self.screen_quad {
                quad.cleanup(&self.device, &mut self.allocator);
            }
            std::mem::ManuallyDrop::drop(&mut self.allocator);
            self.pools.cleanup(&self.device);
            self.pipeline.cleanup(&self.device);
            self.device.destroy_render_pass(self.renderpass, None);
            self.swapchain.cleanup(&self.device);
            self.device.destroy_device(None);
            std::mem::ManuallyDrop::drop(&mut self.surfaces);
            std::mem::ManuallyDrop::drop(&mut self.debug);
            self.instance.destroy_instance(None)
        };
    }
}
