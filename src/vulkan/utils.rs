use std::ffi;

use crate::Pools;
use crate::QueueFamilies;
use crate::Queues;
use ash::{vk, Entry, Instance};

use super::debug::vulkan_debug_utils_callback;
use super::vulkano::Vulkano;

pub fn init_renderpass(
    logical_device: &ash::Device,
    format: vk::Format,
) -> Result<vk::RenderPass, vk::Result> {
    let attachments = [
        vk::AttachmentDescription::builder()
            .format(format)
            .load_op(vk::AttachmentLoadOp::CLEAR)
            .store_op(vk::AttachmentStoreOp::STORE)
            .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
            .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
            .initial_layout(vk::ImageLayout::UNDEFINED)
            .final_layout(vk::ImageLayout::PRESENT_SRC_KHR)
            .samples(vk::SampleCountFlags::TYPE_1)
            .build(),
        vk::AttachmentDescription::builder()
            .format(vk::Format::D32_SFLOAT)
            .load_op(vk::AttachmentLoadOp::CLEAR)
            .store_op(vk::AttachmentStoreOp::DONT_CARE)
            .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
            .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
            .initial_layout(vk::ImageLayout::UNDEFINED)
            .final_layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL)
            .samples(vk::SampleCountFlags::TYPE_1)
            .build(),
    ];
    let color_attachment_references = [vk::AttachmentReference {
        attachment: 0,
        layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
    }];
    let depth_attachment_references = vk::AttachmentReference {
        attachment: 1,
        layout: vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
    };
    let subpasses = [vk::SubpassDescription::builder()
        .color_attachments(&color_attachment_references)
        .depth_stencil_attachment(&depth_attachment_references)
        .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)
        .build()];
    let subpass_dependencies = [vk::SubpassDependency::builder()
        .src_subpass(vk::SUBPASS_EXTERNAL)
        .src_stage_mask(vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT)
        .dst_subpass(0)
        .dst_stage_mask(vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT)
        .dst_access_mask(
            vk::AccessFlags::COLOR_ATTACHMENT_READ | vk::AccessFlags::COLOR_ATTACHMENT_WRITE,
        )
        .build()];
    let renderpass_info = vk::RenderPassCreateInfo::builder()
        .attachments(&attachments)
        .subpasses(&subpasses)
        .dependencies(&subpass_dependencies);
    let renderpass = unsafe { logical_device.create_render_pass(&renderpass_info, None)? };
    Ok(renderpass)
}

pub fn init_instance(entry: &Entry, layer_names: &[&str]) -> Result<Instance, vk::Result> {
    let engine_name = ffi::CString::new("MyEngine").unwrap();
    let app_name = ffi::CString::new("My App").unwrap();
    let app_info = vk::ApplicationInfo::builder()
        .application_name(&app_name)
        .engine_name(&engine_name)
        .engine_version(vk::make_api_version(0, 0, 1, 0))
        .application_version(vk::make_api_version(0, 0, 1, 0))
        .api_version(vk::make_api_version(0, 0, 1, 0));
    let layer_names_c: Vec<ffi::CString> = layer_names
        .iter()
        .map(|&name| std::ffi::CString::new(name).unwrap())
        .collect();
    let layer_name_pointers: Vec<*const i8> = layer_names_c
        .iter()
        .map(|layer_name| layer_name.as_ptr())
        .collect();
    let extension_name_pointers: Vec<*const i8> = vec![
        ash::extensions::ext::DebugUtils::name().as_ptr(),
        ash::extensions::khr::Surface::name().as_ptr(),
        ash::extensions::khr::Win32Surface::name().as_ptr(),
    ];
    let mut debug_create_info = vk::DebugUtilsMessengerCreateInfoEXT::builder()
        .message_severity(
            vk::DebugUtilsMessageSeverityFlagsEXT::WARNING
                | vk::DebugUtilsMessageSeverityFlagsEXT::VERBOSE
                | vk::DebugUtilsMessageSeverityFlagsEXT::INFO
                | vk::DebugUtilsMessageSeverityFlagsEXT::ERROR,
        )
        .message_type(
            vk::DebugUtilsMessageTypeFlagsEXT::GENERAL
                | vk::DebugUtilsMessageTypeFlagsEXT::PERFORMANCE
                | vk::DebugUtilsMessageTypeFlagsEXT::VALIDATION,
        )
        .pfn_user_callback(Some(vulkan_debug_utils_callback));

    let instance_create_info = vk::InstanceCreateInfo::builder()
        .push_next(&mut debug_create_info)
        .application_info(&app_info)
        .enabled_layer_names(&layer_name_pointers)
        .enabled_extension_names(&extension_name_pointers);

    unsafe { entry.create_instance(&instance_create_info, None) }
}

pub fn init_physical_device_and_properties(
    instance: &Instance,
) -> Result<
    (
        vk::PhysicalDevice,
        vk::PhysicalDeviceProperties,
        vk::PhysicalDeviceFeatures,
    ),
    vk::Result,
> {
    let physical_devices = unsafe { instance.enumerate_physical_devices()? };
    let mut chosen = None;
    for device in physical_devices {
        let properties = unsafe { instance.get_physical_device_properties(device) };
        let features = unsafe { instance.get_physical_device_features(device) };
        if properties.device_type == vk::PhysicalDeviceType::DISCRETE_GPU {
            chosen = Some((device, properties, features));
        }
    }
    Ok(chosen.unwrap())
}

pub fn init_device_and_queues(
    instance: &Instance,
    physical_device: vk::PhysicalDevice,
    queue_families: &QueueFamilies,
    layer_names: &[&str],
) -> Result<(ash::Device, Queues), vk::Result> {
    let layer_names_c: Vec<ffi::CString> = layer_names
        .iter()
        .map(|&name| ffi::CString::new(name).unwrap())
        .collect();
    let layer_name_pointers: Vec<*const i8> = layer_names_c
        .iter()
        .map(|layer_name| layer_name.as_ptr())
        .collect();

    let priorities = [1.0f32];
    let queue_infos = [
        vk::DeviceQueueCreateInfo::builder()
            .queue_family_index(queue_families.graphics_queue_index.unwrap())
            .queue_priorities(&priorities)
            .build(),
        vk::DeviceQueueCreateInfo::builder()
            .queue_family_index(queue_families.transfer_queue_index.unwrap())
            .queue_priorities(&priorities)
            .build(),
    ];
    let features = vk::PhysicalDeviceFeatures::builder().fill_mode_non_solid(true);

    let device_extension_name_pointers: Vec<*const i8> =
        vec![ash::extensions::khr::Swapchain::name().as_ptr()];
    let device_create_info = vk::DeviceCreateInfo::builder()
        .queue_create_infos(&queue_infos)
        .enabled_extension_names(&device_extension_name_pointers)
        .enabled_layer_names(&layer_name_pointers)
        .enabled_features(&features);
    let logical_device =
        unsafe { instance.create_device(physical_device, &device_create_info, None)? };
    let graphics_queue =
        unsafe { logical_device.get_device_queue(queue_families.graphics_queue_index.unwrap(), 0) };
    let transfer_queue =
        unsafe { logical_device.get_device_queue(queue_families.transfer_queue_index.unwrap(), 0) };
    Ok((
        logical_device,
        Queues {
            graphics_queue,
            transfer_queue,
        },
    ))
}

pub fn create_commandbuffers(
    logical_device: &ash::Device,
    pools: &Pools,
    amount: usize,
) -> Result<Vec<vk::CommandBuffer>, vk::Result> {
    let commandbuffer_allocate_info = vk::CommandBufferAllocateInfo::builder()
        .command_pool(pools.command_pool_graphics)
        .command_buffer_count(amount as u32);
    unsafe { logical_device.allocate_command_buffers(&commandbuffer_allocate_info) }
}

pub fn create_image(
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
        memory_type_index: find_memory_type(
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

pub fn find_memory_type(
    type_filter: u32,
    required_properties: vk::MemoryPropertyFlags,
    memory_properties: &vk::PhysicalDeviceMemoryProperties,
) -> u32 {
    for (i, memory_type) in memory_properties.memory_types.iter().enumerate() {
        if (type_filter & (1 << i)) > 0 && memory_type.property_flags.contains(required_properties)
        {
            return i as u32;
        }
    }
    panic!("Failed to find suitable memory type")
}

pub fn get_image_mapped_pointer(
    logical_device: &ash::Device,
    image_memory: vk::DeviceMemory,
    memory_requirements: vk::MemoryRequirements,
) -> *mut ffi::c_void {
    let size = memory_requirements.size;

    let mapped_ptr =
        unsafe { logical_device.map_memory(image_memory, 0, size, vk::MemoryMapFlags::empty()) }
            .expect("Failed to map image memory");

    mapped_ptr
}

pub fn screenshot(vulkano: &Vulkano) -> Result<(), Box<dyn std::error::Error>> {
    let command_buffer_allocate_info = vk::CommandBufferAllocateInfo::builder()
        .command_pool(vulkano.pools.command_pool_graphics)
        .command_buffer_count(1);
    let copy_buffer = unsafe {
        vulkano
            .device
            .allocate_command_buffers(&command_buffer_allocate_info)
    }
    .unwrap()[0];

    let command_begin_info =
        vk::CommandBufferBeginInfo::builder().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
    unsafe {
        vulkano
            .device
            .begin_command_buffer(copy_buffer, &command_begin_info)
    }?;
    let memory_properties = unsafe {
        vulkano
            .instance
            .get_physical_device_memory_properties(vulkano.physical_device)
    };

    let (destination_image, destination_image_memory) = create_image(
        &vulkano.device,
        vulkano.swapchain.extent.width,
        vulkano.swapchain.extent.height,
        1,
        vk::SampleCountFlags::TYPE_1,
        vk::Format::R8G8B8A8_UNORM,
        vk::ImageTiling::LINEAR,
        vk::ImageUsageFlags::TRANSFER_DST,
        vk::MemoryPropertyFlags::HOST_VISIBLE,
        &memory_properties,
    );

    let barrier = vk::ImageMemoryBarrier::builder()
        .image(destination_image)
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
        vulkano.device.cmd_pipeline_barrier(
            copy_buffer,
            vk::PipelineStageFlags::TRANSFER,
            vk::PipelineStageFlags::TRANSFER,
            vk::DependencyFlags::empty(),
            &[],
            &[],
            &[barrier],
        );
    };

    let source_image = vulkano.swapchain.images[vulkano.swapchain.current_image];
    let barrier = vk::ImageMemoryBarrier::builder()
        .image(source_image)
        .src_access_mask(vk::AccessFlags::MEMORY_READ)
        .dst_access_mask(vk::AccessFlags::TRANSFER_READ)
        .old_layout(vk::ImageLayout::PRESENT_SRC_KHR)
        .new_layout(vk::ImageLayout::TRANSFER_SRC_OPTIMAL)
        .subresource_range(vk::ImageSubresourceRange {
            aspect_mask: vk::ImageAspectFlags::COLOR,
            base_mip_level: 0,
            level_count: 1,
            base_array_layer: 0,
            layer_count: 1,
        })
        .build();

    unsafe {
        vulkano.device.cmd_pipeline_barrier(
            copy_buffer,
            vk::PipelineStageFlags::TRANSFER,
            vk::PipelineStageFlags::TRANSFER,
            vk::DependencyFlags::empty(),
            &[],
            &[],
            &[barrier],
        );
    };

    let zero_offset = vk::Offset3D::default();
    let copy_area = vk::ImageCopy::builder()
        .src_subresource(vk::ImageSubresourceLayers {
            aspect_mask: vk::ImageAspectFlags::COLOR,
            mip_level: 0,
            base_array_layer: 0,
            layer_count: 1,
        })
        .src_offset(zero_offset)
        .dst_subresource(vk::ImageSubresourceLayers {
            aspect_mask: vk::ImageAspectFlags::COLOR,
            mip_level: 0,
            base_array_layer: 0,
            layer_count: 1,
        })
        .dst_offset(zero_offset)
        .extent(vk::Extent3D {
            width: vulkano.swapchain.extent.width,
            height: vulkano.swapchain.extent.height,
            depth: 1,
        })
        .build();

    unsafe {
        vulkano.device.cmd_copy_image(
            copy_buffer,
            source_image,
            vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
            destination_image,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            &[copy_area],
        )
    };

    let barrier = vk::ImageMemoryBarrier::builder()
        .image(destination_image)
        .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
        .dst_access_mask(vk::AccessFlags::MEMORY_READ)
        .old_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
        .new_layout(vk::ImageLayout::GENERAL)
        .subresource_range(vk::ImageSubresourceRange {
            aspect_mask: vk::ImageAspectFlags::COLOR,
            base_mip_level: 0,
            level_count: 1,
            base_array_layer: 0,
            layer_count: 1,
        })
        .build();

    unsafe {
        vulkano.device.cmd_pipeline_barrier(
            copy_buffer,
            vk::PipelineStageFlags::TRANSFER,
            vk::PipelineStageFlags::TRANSFER,
            vk::DependencyFlags::empty(),
            &[],
            &[],
            &[barrier],
        );
    };

    let barrier = vk::ImageMemoryBarrier::builder()
        .image(source_image)
        .src_access_mask(vk::AccessFlags::TRANSFER_READ)
        .dst_access_mask(vk::AccessFlags::MEMORY_READ)
        .old_layout(vk::ImageLayout::TRANSFER_SRC_OPTIMAL)
        .new_layout(vk::ImageLayout::PRESENT_SRC_KHR)
        .subresource_range(vk::ImageSubresourceRange {
            aspect_mask: vk::ImageAspectFlags::COLOR,
            base_mip_level: 0,
            level_count: 1,
            base_array_layer: 0,
            layer_count: 1,
        })
        .build();

    unsafe {
        vulkano.device.cmd_pipeline_barrier(
            copy_buffer,
            vk::PipelineStageFlags::TRANSFER,
            vk::PipelineStageFlags::TRANSFER,
            vk::DependencyFlags::empty(),
            &[],
            &[],
            &[barrier],
        )
    };

    unsafe { vulkano.device.end_command_buffer(copy_buffer) }?;

    let submit_infos = [vk::SubmitInfo::builder()
        .command_buffers(&[copy_buffer])
        .build()];
    let fence = unsafe {
        vulkano
            .device
            .create_fence(&vk::FenceCreateInfo::default(), None)
    }?;

    unsafe {
        vulkano
            .device
            .queue_submit(vulkano.queues.graphics_queue, &submit_infos, fence)
    }?;

    unsafe {
        vulkano
            .device
            .wait_for_fences(&[fence], true, std::u64::MAX)
    }?;

    unsafe { vulkano.device.destroy_fence(fence, None) };
    unsafe {
        vulkano
            .device
            .free_command_buffers(vulkano.pools.command_pool_graphics, &[copy_buffer])
    };

    let source_ptr = get_image_mapped_pointer(&vulkano.device, destination_image_memory, unsafe {
        vulkano
            .device
            .get_image_memory_requirements(destination_image)
    });

    let subresource_layout = unsafe {
        vulkano.device.get_image_subresource_layout(
            destination_image,
            vk::ImageSubresource {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                mip_level: 0,
                array_layer: 0,
            },
        )
    };

    let mut data = Vec::<u8>::with_capacity(subresource_layout.size as usize);
    unsafe {
        std::ptr::copy(
            source_ptr,
            data.as_mut_ptr() as *mut std::ffi::c_void,
            subresource_layout.size as usize,
        );
        data.set_len(subresource_layout.size as usize);
    }

    unsafe {
        vulkano.device.free_memory(destination_image_memory, None);
        vulkano.device.destroy_image(destination_image, None);
    };

    let mut buffer: image::ImageBuffer<image::Rgba<u8>, _> = image::ImageBuffer::from_raw(
        vulkano.swapchain.extent.width,
        vulkano.swapchain.extent.height,
        data,
    )
    .expect("ImageBuffer creation");

    for (_, _, pixel) in buffer.enumerate_pixels_mut() {
        let blue = pixel[0];
        let green = pixel[1];
        let red = pixel[2];
        let alpha = pixel[3];
        pixel.0 = [red, green, blue, alpha];
    }

    let screen_image = image::DynamicImage::ImageRgba8(buffer);
    screen_image.save("screenshot.jpg")?;

    Ok(())
}
