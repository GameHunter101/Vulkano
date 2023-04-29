use ash::vk;

use super::{buffer::Buffer, utils::create_image};

pub struct Texture {
    pub image: image::RgbaImage,
    pub vk_image: vk::Image,
    pub vk_image_memory: vk::DeviceMemory,
    pub image_view: vk::ImageView,
    pub sampler: vk::Sampler,
}

impl Texture {
    pub fn from_file<P: AsRef<std::path::Path>>(
        path: P,
        instance: &ash::Instance,
        logical_device: &ash::Device,
        physical_device: vk::PhysicalDevice,
        allocator: &mut gpu_allocator::vulkan::Allocator,
        command_pool_graphics: vk::CommandPool,
        graphics_queue: vk::Queue,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let image = image::open(path)
            .map(|img| img.to_rgba8())
            .expect("unable to open image");
        let (width, height) = image.dimensions();

        let memory_properties =
            unsafe { instance.get_physical_device_memory_properties(physical_device) };
        let (vk_image, vk_image_memory) = create_image(
            logical_device,
            width,
            height,
            1,
            vk::SampleCountFlags::TYPE_1,
            vk::Format::R8G8B8A8_SRGB,
            vk::ImageTiling::OPTIMAL,
            vk::ImageUsageFlags::TRANSFER_DST | vk::ImageUsageFlags::SAMPLED,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
            &memory_properties,
        );
        let view_create_info = vk::ImageViewCreateInfo::builder()
            .image(vk_image)
            .view_type(vk::ImageViewType::TYPE_2D)
            .format(vk::Format::R8G8B8A8_SRGB)
            .subresource_range(vk::ImageSubresourceRange {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                level_count: 1,
                layer_count: 1,
                ..Default::default()
            });
        let image_view = unsafe { logical_device.create_image_view(&view_create_info, None) }
            .expect("Image view creation");

        let sampler_info = vk::SamplerCreateInfo::builder()
            .mag_filter(vk::Filter::LINEAR)
            .min_filter(vk::Filter::LINEAR);
        let sampler = unsafe { logical_device.create_sampler(&sampler_info, None) }
            .expect("Sampler creation");

        let data = image.clone().into_raw();
        let mut buffer = Buffer::new(
            logical_device,
            allocator,
            data.len() as u64,
            vk::BufferUsageFlags::TRANSFER_SRC,
        );
        buffer.fill(&data);

        let command_buffer_allocate_info = vk::CommandBufferAllocateInfo::builder()
            .command_pool(command_pool_graphics)
            .command_buffer_count(1);
        let copy_command_buffer =
            unsafe { logical_device.allocate_command_buffers(&command_buffer_allocate_info) }
                .unwrap()[0];

        let command_begin_info = vk::CommandBufferBeginInfo::builder()
            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
        unsafe { logical_device.begin_command_buffer(copy_command_buffer, &command_begin_info) }?;

        let barrier = vk::ImageMemoryBarrier::builder()
            .image(vk_image)
            .src_access_mask(vk::AccessFlags::empty())
            .dst_access_mask(vk::AccessFlags::TRANSFER_WRITE)
            .old_layout(vk::ImageLayout::UNDEFINED)
            .new_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
            .subresource_range(vk::ImageSubresourceRange {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                base_mip_level: 0,
                level_count: 1,
                base_array_layer: 0,
                layer_count: 1,
            })
            .build();
        unsafe {
            logical_device.cmd_pipeline_barrier(
                copy_command_buffer,
                vk::PipelineStageFlags::TOP_OF_PIPE,
                vk::PipelineStageFlags::TRANSFER,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &[barrier],
            )
        };

        let image_subresource = vk::ImageSubresourceLayers {
            aspect_mask: vk::ImageAspectFlags::COLOR,
            mip_level: 0,
            base_array_layer: 0,
            layer_count: 1,
        };
        let region = vk::BufferImageCopy {
            buffer_offset: 0,
            buffer_row_length: 0,
            buffer_image_height: 0,
            image_offset: vk::Offset3D { x: 0, y: 0, z: 0 },
            image_extent: vk::Extent3D {
                width,
                height,
                depth: 1,
            },
            image_subresource,
            ..Default::default()
        };
        unsafe {
            logical_device.cmd_copy_buffer_to_image(
                copy_command_buffer,
                buffer.buffer,
                vk_image,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                &[region],
            );
        }

        let barrier = vk::ImageMemoryBarrier::builder()
            .image(vk_image)
            .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
            .dst_access_mask(vk::AccessFlags::SHADER_READ)
            .old_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
            .new_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
            .subresource_range(vk::ImageSubresourceRange {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                base_mip_level: 0,
                level_count: 1,
                base_array_layer: 0,
                layer_count: 1,
            })
            .build();
        unsafe {
            logical_device.cmd_pipeline_barrier(
                copy_command_buffer,
                vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::FRAGMENT_SHADER,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &[barrier],
            )
        };

        unsafe { logical_device.end_command_buffer(copy_command_buffer) }?;
        let submit_infos = [vk::SubmitInfo::builder()
            .command_buffers(&[copy_command_buffer])
            .build()];
        let fence = unsafe { logical_device.create_fence(&vk::FenceCreateInfo::default(), None) }?;
        unsafe { logical_device.queue_submit(graphics_queue, &submit_infos, fence) }?;

        unsafe { logical_device.wait_for_fences(&[fence], true, std::u64::MAX) }?;
        unsafe { logical_device.destroy_fence(fence, None) };
        buffer.cleanup(logical_device, allocator);
        unsafe {
            logical_device.free_command_buffers(command_pool_graphics, &[copy_command_buffer])
        };

        Ok(Texture {
            image,
            vk_image,
            image_view,
            vk_image_memory,
            sampler,
        })
    }
}

pub struct TextureStorage {
    textures: Vec<Texture>,
}

impl TextureStorage {
    pub fn new() -> Self {
        TextureStorage { textures: vec![] }
    }
    pub unsafe fn cleanup(&mut self, logical_device: &ash::Device) {
        for texture in &self.textures {
            logical_device.destroy_image(texture.vk_image, None);
            logical_device.free_memory(texture.vk_image_memory, None);
            logical_device.destroy_image_view(texture.image_view, None);
            logical_device.destroy_sampler(texture.sampler, None);
        }
    }
    pub fn new_texture_from_file<P: AsRef<std::path::Path>>(
        &mut self,
        path: P,
        instance: &ash::Instance,
        logical_device: &ash::Device,
        physical_device: vk::PhysicalDevice,
        allocator: &mut gpu_allocator::vulkan::Allocator,
        command_pool_graphics: vk::CommandPool,
        graphics_queue: vk::Queue,
    ) -> Result<usize, Box<dyn std::error::Error>> {
        let new_texture = Texture::from_file(
            path,
            instance,
            logical_device,
            physical_device,
            allocator,
            command_pool_graphics,
            graphics_queue,
        )?;
        let new_id = self.textures.len();
        self.textures.push(new_texture);
        Ok(new_id)
    }
    pub fn get(&self, index: usize) -> Option<&Texture> {
        self.textures.get(index)
    }
    pub fn get_mut(&mut self, index: usize) -> Option<&mut Texture> {
        self.textures.get_mut(index)
    }

    pub fn get_descriptor_image_info(&self) -> Vec<vk::DescriptorImageInfo> {
        self.textures
            .iter()
            .map(|t| vk::DescriptorImageInfo {
                image_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                image_view: t.image_view,
                sampler: t.sampler,
                ..Default::default()
            })
            .collect()
    }
}
