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
use crate::{InstanceData, Model, VertexData};

pub struct Vulkano {
    pub window: winit::window::Window,
    _entry: Entry,
    pub instance: Instance,
    pub debug: std::mem::ManuallyDrop<Debug>,
    pub surfaces: std::mem::ManuallyDrop<Surface>,
    pub physical_device: vk::PhysicalDevice,
    _physical_device_properties: vk::PhysicalDeviceProperties,
    _physical_device_features: vk::PhysicalDeviceFeatures,
    _queue_families: QueueFamilies,
    pub queues: Queues,
    pub device: ash::Device,
    pub swapchain: Swapchain,
    pub renderpass: vk::RenderPass,
    pub pipeline: Pipeline,
    pub pools: Pools,
    pub commandbuffers: Vec<vk::CommandBuffer>,
    pub models: Vec<Model<VertexData, InstanceData>>,
    pub allocator: std::mem::ManuallyDrop<gpu_allocator::vulkan::Allocator>,
    pub uniform_buffer: Buffer,
    descriptor_pool: vk::DescriptorPool,
    descriptor_sets: Vec<vk::DescriptorSet>,
}

impl Vulkano {
    pub fn init(window: winit::window::Window) -> Result<Vulkano, Box<dyn std::error::Error>> {
        let entry = unsafe { Entry::load()? };

        let layer_names = vec!["VK_LAYER_KHRONOS_validation"];
        let instance = init_instance(&entry, &layer_names)?;
        let debug = Debug::init(&entry, &instance)?;
        let surfaces = Surface::init(&window, &entry, &instance)?;

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

        let pipeline = Pipeline::init(&logical_device, &swapchain, &renderpass)?;
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
            128,
            vk::BufferUsageFlags::UNIFORM_BUFFER,
        );
        let camera_transform: [[[f32; 4]; 4]; 2] = [
            na::Matrix4::identity().into(),
            na::Matrix4::identity().into(),
        ];
        uniform_buffer.fill(&camera_transform);

        let pool_sizes = [vk::DescriptorPoolSize {
            ty: vk::DescriptorType::UNIFORM_BUFFER,
            descriptor_count: swapchain.amount_of_images,
        }];
        let descriptor_pool_info = vk::DescriptorPoolCreateInfo::builder()
            .max_sets(swapchain.amount_of_images)
            .pool_sizes(&pool_sizes);
        let descriptor_pool =
            unsafe { logical_device.create_descriptor_pool(&descriptor_pool_info, None) }?;

        let desc_layouts =
            vec![pipeline.descriptor_set_layouts[0]; swapchain.amount_of_images as usize];
        let descriptor_set_allocate_info = vk::DescriptorSetAllocateInfo::builder()
            .descriptor_pool(descriptor_pool)
            .set_layouts(&desc_layouts);
        let descriptor_sets =
            unsafe { logical_device.allocate_descriptor_sets(&descriptor_set_allocate_info) }?;

        for descriptor_set in descriptor_sets.iter() {
            let buffer_infos = [vk::DescriptorBufferInfo {
                buffer: uniform_buffer.buffer,
                offset: 0,
                range: 128,
            }];
            let desc_sets_write = [vk::WriteDescriptorSet::builder()
                .dst_set(*descriptor_set)
                .dst_binding(0)
                .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                .buffer_info(&buffer_infos)
                .build()];
            unsafe { logical_device.update_descriptor_sets(&desc_sets_write, &[]) };
        }

        Ok(Vulkano {
            window,
            _entry: entry,
            instance,
            debug: std::mem::ManuallyDrop::new(debug),
            surfaces: std::mem::ManuallyDrop::new(surfaces),
            physical_device,
            _physical_device_properties: physical_device_properties,
            _physical_device_features: physical_device_features,
            _queue_families: queue_families,
            queues,
            device: logical_device,
            swapchain,
            renderpass,
            pipeline,
            pools,
            commandbuffers,
            models: vec![],
            allocator: std::mem::ManuallyDrop::new(allocator),
            uniform_buffer,
            descriptor_pool,
            descriptor_sets,
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
                &[self.descriptor_sets[index]],
                &[],
            );
            for model in &self.models {
                model.draw(&self.device, command_buffer);
            }
            self.device.cmd_end_render_pass(command_buffer);
            self.device.end_command_buffer(command_buffer)?;
        }
        Ok(())
    }
}

impl Drop for Vulkano {
    fn drop(&mut self) {
        unsafe {
            self.device
                .device_wait_idle()
                .expect("Something went wrong while waiting");
            self.device
                .destroy_descriptor_pool(self.descriptor_pool, None);
            self.uniform_buffer
                .cleanup(&self.device, &mut self.allocator);
            for model in &mut self.models {
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
