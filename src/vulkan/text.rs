use std::collections::HashMap;

use ash::vk;

use super::{buffer::Buffer, pipeline::Pipeline, swapchain::Swapchain, utils::create_image};

pub struct TextTexture {
    vk_image: vk::Image,
    pub image_view: vk::ImageView,
    pub sampler: vk::Sampler,
    vk_image_memory: vk::DeviceMemory,
}

impl TextTexture {
    pub fn from_u8s(
        data: &[u8],
        width: u32,
        height: u32,
        instance: &ash::Instance,
        logical_device: &ash::Device,
        physical_device: vk::PhysicalDevice,
        allocator: &mut gpu_allocator::vulkan::Allocator,
        command_pool_graphics: vk::CommandPool,
        graphics_queue: vk::Queue,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        if data.len() == 0 {
            return Err(Box::new(TextError::EmptyData));
        }
        let memory_properties =
            unsafe { instance.get_physical_device_memory_properties(physical_device) };
        let (vk_image, vk_image_memory) = create_image(
            logical_device,
            width,
            height,
            1,
            vk::SampleCountFlags::TYPE_1,
            vk::Format::R8_SRGB,
            vk::ImageTiling::OPTIMAL,
            vk::ImageUsageFlags::TRANSFER_DST | vk::ImageUsageFlags::SAMPLED,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
            &memory_properties,
        );
        let view_create_info = vk::ImageViewCreateInfo::builder()
            .image(vk_image)
            .view_type(vk::ImageViewType::TYPE_2D)
            .format(vk::Format::R8_SRGB)
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

        let mut buffer = Buffer::new(
            logical_device,
            allocator,
            (data.len() * 4) as u64,
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
        Ok(TextTexture {
            vk_image,
            image_view,
            sampler,
            vk_image_memory,
        })
    }
}

pub struct AllText {
    fonts: Vec<fontdue::Font>,
    vertex_data: Vec<TextVertexData>,
    vertex_buffer: Option<Buffer>,
    textures: Vec<TextTexture>,
    texture_ids: HashMap<fontdue::layout::GlyphRasterConfig, u32>,
    pipeline: Option<Pipeline>,
    desc_pool: Option<vk::DescriptorPool>,
    descriptor_sets: Vec<vk::DescriptorSet>,
    texture_amount: usize,
}

impl AllText {
    pub fn new<P: AsRef<std::path::Path>>(
        standard_font: P,
    ) -> Result<AllText, Box<dyn std::error::Error>> {
        let mut all_text = AllText {
            fonts: vec![],
            vertex_data: vec![],
            vertex_buffer: None,
            textures: vec![],
            texture_ids: HashMap::new(),
            pipeline: None,
            desc_pool: None,
            descriptor_sets: vec![],
            texture_amount: 0,
        };
        all_text.load_font(standard_font)?;
        Ok(all_text)
    }

    pub fn load_font<P: AsRef<std::path::Path>>(
        &mut self,
        path: P,
    ) -> Result<usize, Box<dyn std::error::Error>> {
        let font_data = std::fs::read(path)?;
        let font = fontdue::Font::from_bytes(font_data, fontdue::FontSettings::default())?;
        self.fonts.push(font);
        Ok(self.fonts.len() - 1)
    }

    pub fn create_letters(
        &self,
        styles: &[&fontdue::layout::TextStyle],
        color: [f32; 3],
    ) -> Vec<Letter> {
        let mut layout: fontdue::layout::Layout<()> =
            fontdue::layout::Layout::new(fontdue::layout::CoordinateSystem::PositiveYUp);
        let settings = fontdue::layout::LayoutSettings {
            ..fontdue::layout::LayoutSettings::default()
        };
        layout.reset(&settings);
        for style in styles {
            layout.append(&self.fonts, style);
        }
        let output = layout.glyphs();
        let mut letters: Vec<Letter> = vec![];
        for glyph in output {
            letters.push(Letter {
                color,
                position_and_shape: glyph.clone(),
            });
        }
        letters
    }

    pub fn create_vertex_data(
        &mut self,
        letters: Vec<Letter>,
        position: (u32, u32),
        window: &winit::window::Window,
        instance: &ash::Instance,
        logical_device: &ash::Device,
        physical_device: vk::PhysicalDevice,
        allocator: &mut gpu_allocator::vulkan::Allocator,
        command_pool_graphics: vk::CommandPool,
        graphics_queue: vk::Queue,
        swapchain: &Swapchain,
        renderpass: &vk::RenderPass,
    ) {
        let screensize = window.inner_size();
        let mut need_texture_update = false;
        let mut vertexdata = Vec::with_capacity(6 * letters.len());
        for l in letters {
            let id_option = self.texture_ids.get(&l.position_and_shape.key);
            let id;
            if id_option.is_none() {
                let (metrics, bitmap) = self.fonts[l.position_and_shape.font_index]
                    .rasterize(l.position_and_shape.parent, l.position_and_shape.key.px);
                need_texture_update = true;
                id = match self
                    .new_texture_from_u8s(
                        &bitmap,
                        metrics.width as u32,
                        metrics.height as u32,
                        instance,
                        logical_device,
                        physical_device,
                        allocator,
                        command_pool_graphics,
                        graphics_queue,
                    ) {
                        Ok(the_id) => the_id as u32,
                        Err(e) => {
                            if let Some(err) = e.downcast_ref::<TextError>() {
                                match err {
                                    TextError::EmptyData => {
                                        continue;
                                    }
                                }
                            } else {
                                panic!();
                            }
                        }
                    };
                self.texture_ids.insert(l.position_and_shape.key, id);
            } else {
                id = *id_option.unwrap() as u32;
            }
            let left =
                2. * (l.position_and_shape.x + position.0 as f32) / screensize.width as f32 - 1.0;
            let right = 2.
                * (l.position_and_shape.x + position.0 as f32 + l.position_and_shape.width as f32)
                / screensize.width as f32
                - 1.0;
            let top = 2.
                * (-l.position_and_shape.y + position.1 as f32
                    - l.position_and_shape.height as f32)
                / screensize.height as f32
                - 1.0;
            let bottom =
                2. * (-l.position_and_shape.y + position.1 as f32) / screensize.height as f32 - 1.0;
            let v1 = TextVertexData {
                position: [left, top, 0.],
                tex_coord: [0., 0.],
                color: l.color,
                texture_id: id,
            };
            let v2 = TextVertexData {
                position: [left, bottom, 0.],
                tex_coord: [0., 1.],
                color: l.color,
                texture_id: id,
            };
            let v3 = TextVertexData {
                position: [right, top, 0.],
                tex_coord: [1., 0.],
                color: l.color,
                texture_id: id,
            };
            let v4 = TextVertexData {
                position: [right, bottom, 0.],
                tex_coord: [1., 1.],
                color: l.color,
                texture_id: id,
            };
            vertexdata.push(v1);
            vertexdata.push(v2);
            vertexdata.push(v3);
            vertexdata.push(v3);
            vertexdata.push(v2);
            vertexdata.push(v4);
        }
        self.vertex_data.append(&mut vertexdata);
        if need_texture_update {
            self.update_textures(swapchain, logical_device, renderpass)
                .unwrap();
        }
    }

    pub fn new_texture_from_u8s(
        &mut self,
        data: &[u8],
        width: u32,
        height: u32,
        instance: &ash::Instance,
        logical_device: &ash::Device,
        physical_device: vk::PhysicalDevice,
        allocator: &mut gpu_allocator::vulkan::Allocator,
        command_pool_graphics: vk::CommandPool,
        graphics_queue: vk::Queue,
    ) -> Result<usize, Box<dyn std::error::Error>> {
        let new_texture = TextTexture::from_u8s(
            data,
            width,
            height,
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

    pub fn update_textures(
        &mut self,
        swapchain: &Swapchain,
        logical_device: &ash::Device,
        renderpass: &vk::RenderPass,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let amount = self.textures.len();
        if amount > self.texture_amount {
            self.texture_amount = amount;
            let p = self.pipeline.take();
            if let Some(mut pip) = p {
                pip.cleanup(logical_device);
            }
        }
        if self.pipeline.is_none() {
            let pip = Pipeline::init_text(logical_device, swapchain, renderpass, amount as u32)?;
            self.pipeline = Some(pip);
        }
        if let Some(pool) = self.desc_pool {
            unsafe {
                logical_device.destroy_descriptor_pool(pool, None);
            }
        }
        let pool_sizes = [vk::DescriptorPoolSize {
            ty: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
            descriptor_count: amount as u32 * swapchain.amount_of_images,
        }];
        let descriptor_pool_info = vk::DescriptorPoolCreateInfo::builder()
            .flags(vk::DescriptorPoolCreateFlags::UPDATE_AFTER_BIND_EXT)
            .max_sets(swapchain.amount_of_images)
            .pool_sizes(&pool_sizes);
        let descriptor_pool =
            unsafe { logical_device.create_descriptor_pool(&descriptor_pool_info, None) }?;
        self.desc_pool = Some(descriptor_pool);

        let descriptor_layouts_text = vec![
            self.pipeline.as_ref().unwrap().descriptor_set_layouts[0];
            swapchain.amount_of_images as usize
        ];
        let descriptor_counts = vec![amount as u32; swapchain.amount_of_images as usize];
        let mut count_info = vk::DescriptorSetVariableDescriptorCountAllocateInfoEXT::builder()
            .descriptor_counts(&descriptor_counts);
        let descriptor_set_allocate_info_text = vk::DescriptorSetAllocateInfo::builder()
            .descriptor_pool(descriptor_pool)
            .set_layouts(&descriptor_layouts_text)
            .push_next(&mut count_info);
        let descriptor_sets_text =
            unsafe { logical_device.allocate_descriptor_sets(&descriptor_set_allocate_info_text) }?;

        for i in 0..swapchain.amount_of_images {
            let image_infos: Vec<vk::DescriptorImageInfo> = self
                .textures
                .iter()
                .map(|t| vk::DescriptorImageInfo {
                    image_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                    image_view: t.image_view,
                    sampler: t.sampler,
                    ..Default::default()
                })
                .collect();

            let descriptor_write_image = vk::WriteDescriptorSet::builder()
                .dst_set(descriptor_sets_text[i as usize])
                .dst_binding(0)
                .dst_array_element(0)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .image_info(&image_infos)
                .build();
            unsafe {
                logical_device.update_descriptor_sets(&[descriptor_write_image], &[]);
            }
        }
        self.descriptor_sets = descriptor_sets_text;
        Ok(())
    }

    pub fn update_vertex_buffer(
        &mut self,
        logical_device: &ash::Device,
        allocator: &mut gpu_allocator::vulkan::Allocator,
    ) {
        if self.vertex_data.is_empty() {
            return;
        }
        if let Some(buffer) = &mut self.vertex_buffer {
            buffer.fill(&self.vertex_data);
            return;
        } else {
            let bytes = (self.vertex_data.len() * std::mem::size_of::<TextVertexData>()) as u64;
            let mut buffer = Buffer::new(
                logical_device,
                allocator,
                bytes,
                vk::BufferUsageFlags::VERTEX_BUFFER,
            );
            buffer.fill(&self.vertex_data);
            self.vertex_buffer = Some(buffer);
        }
    }

    pub fn draw(
        &self,
        logical_device: &ash::Device,
        command_buffer: vk::CommandBuffer,
        index: usize,
    ) {
        if let Some(pipeline) = &self.pipeline {
            if self.descriptor_sets.len() > index {
                unsafe {
                    logical_device.cmd_bind_pipeline(
                        command_buffer,
                        vk::PipelineBindPoint::GRAPHICS,
                        pipeline.pipeline,
                    );

                    logical_device.cmd_bind_descriptor_sets(
                        command_buffer,
                        vk::PipelineBindPoint::GRAPHICS,
                        pipeline.layout,
                        0,
                        &[self.descriptor_sets[index]],
                        &[],
                    );
                }

                if let Some(vertex_buffer) = &self.vertex_buffer {
                    unsafe {
                        logical_device.cmd_bind_vertex_buffers(
                            command_buffer,
                            0,
                            &[vertex_buffer.buffer],
                            &[0],
                        );
                        logical_device.cmd_draw(
                            command_buffer,
                            self.vertex_data.len() as u32,
                            1,
                            0,
                            0,
                        );
                    }
                }
            }
        }
    }

    pub fn clear_pipeline(&mut self, logical_device: &ash::Device) {
        let p = self.pipeline.take();
        if let Some(mut pip) = p {
            pip.cleanup(logical_device);
        }
    }

    pub fn cleanup(
        &mut self,
        logical_device: &ash::Device,
        allocator: &mut gpu_allocator::vulkan::Allocator,
    ) {
        self.clear_pipeline(logical_device);
        let p = self.desc_pool.take();
        if let Some(pool) = p {
            unsafe {
                logical_device.destroy_descriptor_pool(pool, None);
            }
        }
        let b = self.vertex_buffer.take();
        if let Some(mut buf) = b {
            buf.cleanup(logical_device, allocator);
        }
        for texture in &self.textures {
            unsafe {
                logical_device.destroy_sampler(texture.sampler, None);
                logical_device.destroy_image_view(texture.image_view, None);
                logical_device.destroy_image(texture.vk_image, None);
                logical_device.free_memory(texture.vk_image_memory, None);
            }
        }
    }
}

#[derive(Debug)]
pub struct Letter {
    color: [f32; 3],
    position_and_shape: fontdue::layout::GlyphPosition,
}

#[derive(Clone, Copy, Debug)]
#[repr(C)]
pub struct TextVertexData {
    pub position: [f32; 3],
    pub tex_coord: [f32; 2],
    pub color: [f32; 3],
    pub texture_id: u32,
}

#[derive(Debug)]
pub enum TextError {
    EmptyData,
}
impl std::fmt::Display for TextError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TextError::EmptyData => {
                write!(f, "Empty Data")?;
            }
        }
        Ok(())
    }
}
impl std::error::Error for TextError {}