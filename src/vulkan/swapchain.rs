use ash::{vk, Instance};

use crate::{QueueFamilies, Surface};

pub struct Swapchain {
    pub swapchain_loader: ash::extensions::khr::Swapchain,
    pub swapchain: vk::SwapchainKHR,
    _images: Vec<vk::Image>,
    image_views: Vec<vk::ImageView>,
    pub framebuffers: Vec<vk::Framebuffer>,
    pub surface_format: vk::SurfaceFormatKHR,
    pub extent: vk::Extent2D,
    pub image_available: Vec<vk::Semaphore>,
    pub rendering_finished: Vec<vk::Semaphore>,
    pub may_begin_drawing: Vec<vk::Fence>,
    pub amount_of_images: u32,
    pub current_image: usize,
    depth_data: DepthData,
}

impl Swapchain {
    pub fn init(
        instance: &Instance,
        surfaces: &Surface,
        physical_device: vk::PhysicalDevice,
        logical_device: &ash::Device,
        queue_families: &QueueFamilies,
    ) -> Result<Swapchain, vk::Result> {
        let surface_capabilities = surfaces.get_capabilities(physical_device)?;
        let surface_format = *surfaces.get_formats(physical_device)?.first().unwrap();
        let queue_family_graphics = [queue_families.graphics_queue_index.unwrap()];
        let extent = surface_capabilities.current_extent;
        let _surface_present_mode = surfaces.get_present_modes(physical_device)?;
        let swapchain_create_info = vk::SwapchainCreateInfoKHR::builder()
            .surface(surfaces.surface)
            .min_image_count(
                3.max(surface_capabilities.min_image_count)
                    .min(surface_capabilities.max_image_count),
            )
            .image_format(surface_format.format)
            .image_color_space(surface_format.color_space)
            .image_extent(extent)
            .image_array_layers(1)
            .image_usage(vk::ImageUsageFlags::COLOR_ATTACHMENT)
            .image_sharing_mode(vk::SharingMode::EXCLUSIVE)
            .queue_family_indices(&queue_family_graphics)
            .pre_transform(surface_capabilities.current_transform)
            .composite_alpha(vk::CompositeAlphaFlagsKHR::OPAQUE)
            .present_mode(vk::PresentModeKHR::FIFO);
        let swapchain_loader = ash::extensions::khr::Swapchain::new(&instance, &logical_device);
        let swapchain = unsafe { swapchain_loader.create_swapchain(&swapchain_create_info, None)? };
        let swapchain_images = unsafe { swapchain_loader.get_swapchain_images(swapchain)? };
        let amount_of_images = swapchain_images.len() as u32;
        let mut swapchain_imageviews: Vec<vk::ImageView> =
            Vec::with_capacity(swapchain_images.len());
        for image in &swapchain_images {
            let subresource_range = vk::ImageSubresourceRange::builder()
                .aspect_mask(vk::ImageAspectFlags::COLOR)
                .base_mip_level(0)
                .level_count(1)
                .base_array_layer(0)
                .layer_count(1);
            let imageview_create_info = vk::ImageViewCreateInfo::builder()
                .image(*image)
                .view_type(vk::ImageViewType::TYPE_2D)
                .format(vk::Format::B8G8R8A8_UNORM)
                .subresource_range(*subresource_range);
            let imageview =
                unsafe { logical_device.create_image_view(&imageview_create_info, None)? };
            swapchain_imageviews.push(imageview);
        }

        let mut image_available = vec![];
        let mut rendering_finished = vec![];
        let mut may_begin_drawing = vec![];
        let semaphore_info = vk::SemaphoreCreateInfo::builder();
        let fence_info = vk::FenceCreateInfo::builder().flags(vk::FenceCreateFlags::SIGNALED);
        for _ in 0..amount_of_images {
            let semaphore_available =
                unsafe { logical_device.create_semaphore(&semaphore_info, None) }?;
            let semaphore_finished =
                unsafe { logical_device.create_semaphore(&semaphore_info, None) }?;
            image_available.push(semaphore_available);
            rendering_finished.push(semaphore_finished);
            let fence = unsafe { logical_device.create_fence(&fence_info, None) }?;
            may_begin_drawing.push(fence);
        }

        let mut depth_data = DepthData::init();
        let memory_properties =
            unsafe { instance.get_physical_device_memory_properties(physical_device) };
        depth_data.create_depth_resources(
            instance,
            logical_device,
            physical_device,
            extent,
            &memory_properties,
        );

        Ok(Swapchain {
            swapchain_loader,
            swapchain,
            _images: swapchain_images,
            image_views: swapchain_imageviews,
            framebuffers: vec![],
            surface_format,
            extent,
            image_available,
            rendering_finished,
            may_begin_drawing,
            amount_of_images,
            current_image: 0,
            depth_data,
        })
    }

    pub fn create_framebuffers(
        &mut self,
        logical_device: &ash::Device,
        renderpass: vk::RenderPass,
    ) -> Result<(), vk::Result> {
        for image_view in &self.image_views {
            let view = [*image_view, self.depth_data.depth_image_view.unwrap()];
            let framebuffer_info = vk::FramebufferCreateInfo::builder()
                .render_pass(renderpass)
                .attachments(&view)
                .width(self.extent.width)
                .height(self.extent.height)
                .layers(1);
            let buffer = unsafe { logical_device.create_framebuffer(&framebuffer_info, None)? };
            self.framebuffers.push(buffer);
        }
        Ok(())
    }

    pub unsafe fn cleanup(&self, logical_device: &ash::Device) {
        logical_device.destroy_image_view(self.depth_data.depth_image_view.unwrap(), None);
        logical_device.destroy_image(self.depth_data.depth_image.unwrap(), None);
        logical_device.free_memory(self.depth_data.depth_image_memory.unwrap(), None);

        for fence in &self.may_begin_drawing {
            logical_device.destroy_fence(*fence, None);
        }
        for semaphore in &self.image_available {
            logical_device.destroy_semaphore(*semaphore, None);
        }
        for semaphore in &self.rendering_finished {
            logical_device.destroy_semaphore(*semaphore, None);
        }
        for framebuffer in &self.framebuffers {
            logical_device.destroy_framebuffer(*framebuffer, None);
        }
        for image_view in &self.image_views {
            logical_device.destroy_image_view(*image_view, None);
        }
        self.swapchain_loader
            .destroy_swapchain(self.swapchain, None)
    }
}

struct DepthData {
    depth_image: Option<vk::Image>,
    depth_image_view: Option<vk::ImageView>,
    depth_image_memory: Option<vk::DeviceMemory>,
}

impl DepthData {
    fn init() -> DepthData {
        DepthData {
            depth_image: None,
            depth_image_view: None,
            depth_image_memory: None,
        }
    }

    fn create_depth_resources(
        &mut self,
        instance: &Instance,
        logical_device: &ash::Device,
        physical_device: vk::PhysicalDevice,
        swapchain_extent: vk::Extent2D,
        device_memory_properties: &vk::PhysicalDeviceMemoryProperties,
    ) {
        let depth_format = self.find_supported_format(
            instance,
            physical_device,
            &[
                vk::Format::D32_SFLOAT,
                vk::Format::D32_SFLOAT_S8_UINT,
                vk::Format::D24_UNORM_S8_UINT,
            ],
            vk::ImageTiling::OPTIMAL,
            vk::FormatFeatureFlags::DEPTH_STENCIL_ATTACHMENT,
        );
        let (depth_image, depth_image_memory) = self.create_image(
            logical_device,
            swapchain_extent.width,
            swapchain_extent.height,
            1,
            vk::SampleCountFlags::TYPE_1,
            depth_format,
            vk::ImageTiling::OPTIMAL,
            vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
            device_memory_properties,
        );
        let depth_image_view = self.create_image_view(
            logical_device,
            depth_image,
            depth_format,
            vk::ImageAspectFlags::DEPTH,
            1,
        );

        self.depth_image = Some(depth_image);
        self.depth_image_view = Some(depth_image_view);
        self.depth_image_memory = Some(depth_image_memory);
    }

    fn find_supported_format(
        &self,
        instance: &Instance,
        physical_device: vk::PhysicalDevice,
        candidate_formats: &[vk::Format],
        tiling: vk::ImageTiling,
        features: vk::FormatFeatureFlags,
    ) -> vk::Format {
        for &format in candidate_formats.iter() {
            let format_properties =
                unsafe { instance.get_physical_device_format_properties(physical_device, format) };
            if tiling == vk::ImageTiling::LINEAR
                && format_properties.linear_tiling_features.contains(features)
            {
                return format.clone();
            } else if tiling == vk::ImageTiling::OPTIMAL
                && format_properties.optimal_tiling_features.contains(features)
            {
                return format.clone();
            }
        }
        panic!("Failed to find supported format")
    }

    fn create_image(
        &self,
        logical_device: &ash::Device,
        width: u32,
        height: u32,
        mip_levels: u32,
        num_samples: vk::SampleCountFlags,
        format: vk::Format,
        tiling: vk::ImageTiling,
        usage: vk::ImageUsageFlags,
        required_memory_properties: vk::MemoryPropertyFlags,
        device_memory_properties: &vk::PhysicalDeviceMemoryProperties,
    ) -> (vk::Image, vk::DeviceMemory) {
        let image_create_info = vk::ImageCreateInfo {
            s_type: vk::StructureType::IMAGE_CREATE_INFO,
            p_next: std::ptr::null(),
            flags: vk::ImageCreateFlags::empty(),
            image_type: vk::ImageType::TYPE_2D,
            format,
            mip_levels,
            array_layers: 1,
            samples: num_samples,
            tiling,
            usage,
            sharing_mode: vk::SharingMode::EXCLUSIVE,
            queue_family_index_count: 0,
            p_queue_family_indices: std::ptr::null(),
            initial_layout: vk::ImageLayout::UNDEFINED,
            extent: vk::Extent3D {
                width,
                height,
                depth: 1,
            },
        };

        let texture_image = unsafe {
            logical_device
                .create_image(&image_create_info, None)
                .expect("Faield to create texture image")
        };

        let image_memory_requirement =
            unsafe { logical_device.get_image_memory_requirements(texture_image) };
        let memory_allocate_info = vk::MemoryAllocateInfo {
            s_type: vk::StructureType::MEMORY_ALLOCATE_INFO,
            p_next: std::ptr::null(),
            allocation_size: image_memory_requirement.size,
            memory_type_index: self.find_memory_type(
                image_memory_requirement.memory_type_bits,
                required_memory_properties,
                device_memory_properties,
            ),
        };

        let texture_image_memory = unsafe {
            logical_device
                .allocate_memory(&memory_allocate_info, None)
                .expect("Failed to allocate texture image memory")
        };

        unsafe {
            logical_device
                .bind_image_memory(texture_image, texture_image_memory, 0)
                .expect("Failed to bind image memory")
        };

        (texture_image, texture_image_memory)
    }

    fn find_memory_type(
        &self,
        type_filter: u32,
        required_properties: vk::MemoryPropertyFlags,
        memory_properties: &vk::PhysicalDeviceMemoryProperties,
    ) -> u32 {
        for (i, memory_type) in memory_properties.memory_types.iter().enumerate() {
            if (type_filter & (1 << i)) > 0
                && memory_type.property_flags.contains(required_properties)
            {
                return i as u32;
            }
        }
        panic!("Failed to find suitable memory type")
    }

    fn create_image_view(
        &self,
        logical_device: &ash::Device,
        image: vk::Image,
        format: vk::Format,
        aspect_flags: vk::ImageAspectFlags,
        mip_levels: u32,
    ) -> vk::ImageView {
        let image_view_create_info = vk::ImageViewCreateInfo::builder()
            .view_type(vk::ImageViewType::TYPE_2D)
            .format(format)
            .components(
                vk::ComponentMapping::builder()
                    .r(vk::ComponentSwizzle::IDENTITY)
                    .g(vk::ComponentSwizzle::IDENTITY)
                    .b(vk::ComponentSwizzle::IDENTITY)
                    .a(vk::ComponentSwizzle::IDENTITY)
                    .build(),
            )
            .subresource_range(
                vk::ImageSubresourceRange::builder()
                    .aspect_mask(aspect_flags)
                    .base_mip_level(0)
                    .level_count(mip_levels)
                    .base_array_layer(0)
                    .layer_count(1)
                    .build(),
            )
            .image(image);
        unsafe {
            logical_device
                .create_image_view(&image_view_create_info, None)
                .expect("Failed to create image view")
        }
    }
}
