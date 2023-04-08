use std::ptr::copy_nonoverlapping as memcpy;
use std::{ffi, mem::size_of_val};

use ash::{vk, Entry, Instance};
use winit::{dpi::PhysicalSize, window::WindowBuilder};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let event_loop = winit::event_loop::EventLoop::new();
    let window = WindowBuilder::new()
        .with_inner_size(PhysicalSize::<u32>::from((800, 600)))
        .with_title("Vulkano")
        .build(&event_loop)?;
    let mut vulkano = Vulkano::init(window)?;

    use winit::event::{Event, WindowEvent};
    event_loop.run(move |event, _, control_flow| match event {
        Event::WindowEvent {
            event: WindowEvent::CloseRequested,
            ..
        } => {
            *control_flow = winit::event_loop::ControlFlow::Exit;
        }
        Event::RedrawRequested(_) => {
            let (image_index, _) = unsafe {
                vulkano
                    .swapchain
                    .swapchain_loader
                    .acquire_next_image(
                        vulkano.swapchain.swapchain,
                        std::u64::MAX,
                        vulkano.swapchain.image_available[vulkano.swapchain.current_image],
                        vk::Fence::null(),
                    )
                    .expect("Error acquiring image")
            };
            unsafe {
                vulkano
                    .device
                    .wait_for_fences(
                        &[vulkano.swapchain.may_begin_drawing[vulkano.swapchain.current_image]],
                        true,
                        std::u64::MAX,
                    )
                    .expect("Waiting for fence");
                vulkano
                    .device
                    .reset_fences(&[
                        vulkano.swapchain.may_begin_drawing[vulkano.swapchain.current_image]
                    ])
                    .expect("Resetting fences");
            }
            let semaphores_available =
                [vulkano.swapchain.image_available[vulkano.swapchain.current_image]];
            let waiting_stages = [vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT];
            let semaphores_finished =
                [vulkano.swapchain.rendering_finished[vulkano.swapchain.current_image]];
            let commandbuffers = [vulkano.commandbuffers[image_index as usize]];
            let submit_info = [vk::SubmitInfo::builder()
                .wait_semaphores(&semaphores_available)
                .wait_dst_stage_mask(&waiting_stages)
                .command_buffers(&commandbuffers)
                .signal_semaphores(&semaphores_finished)
                .build()];
            unsafe {
                vulkano
                    .device
                    .queue_submit(
                        vulkano.queues.graphics_queue,
                        &submit_info,
                        vulkano.swapchain.may_begin_drawing[vulkano.swapchain.current_image],
                    )
                    .expect("Submitting to queue");
            }
            let swapchains = [vulkano.swapchain.swapchain];
            let indices = [image_index];
            let present_info = vk::PresentInfoKHR::builder()
                .wait_semaphores(&semaphores_finished)
                .swapchains(&swapchains)
                .image_indices(&indices);
            unsafe {
                vulkano
                    .swapchain
                    .swapchain_loader
                    .queue_present(vulkano.queues.graphics_queue, &present_info)
                    .expect("Presenting to queue");
            }
            vulkano.swapchain.current_image =
                (vulkano.swapchain.current_image + 1) % vulkano.swapchain.amount_of_images as usize;
        }
        Event::MainEventsCleared => {
            vulkano.window.request_redraw();
        }
        _ => {}
    });
}

unsafe extern "system" fn vulkan_debug_utils_callback(
    message_severity: vk::DebugUtilsMessageSeverityFlagsEXT,
    message_type: vk::DebugUtilsMessageTypeFlagsEXT,
    p_callback_data: *const vk::DebugUtilsMessengerCallbackDataEXT,
    _p_user_data: *mut ffi::c_void,
) -> vk::Bool32 {
    let message = ffi::CStr::from_ptr((*p_callback_data).p_message);
    let severity = format!("{:?}", message_severity).to_lowercase();
    let ty = format!("{:?}", message_type).to_lowercase();
    println!("[Debug][{}][{}] {:?}", severity, ty, message);
    vk::FALSE
}

fn init_instance(entry: &Entry, layer_names: &[&str]) -> Result<Instance, vk::Result> {
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

fn init_physical_device_and_properties(
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

struct Queues {
    graphics_queue: vk::Queue,
    transfer_queue: vk::Queue,
}

fn init_device_and_queues(
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

fn init_renderpass(
    logical_device: &ash::Device,
    physical_device: vk::PhysicalDevice,
    format: vk::Format,
) -> Result<vk::RenderPass, vk::Result> {
    let attachments = [vk::AttachmentDescription::builder()
        .format(format)
        .load_op(vk::AttachmentLoadOp::CLEAR)
        .store_op(vk::AttachmentStoreOp::STORE)
        .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
        .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
        .initial_layout(vk::ImageLayout::UNDEFINED)
        .final_layout(vk::ImageLayout::PRESENT_SRC_KHR)
        .samples(vk::SampleCountFlags::TYPE_1)
        .build()];
    let color_attachment_references = [vk::AttachmentReference {
        attachment: 0,
        layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
    }];
    let subpasses = [vk::SubpassDescription::builder()
        .color_attachments(&color_attachment_references)
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

fn create_commandbuffers(
    logical_device: &ash::Device,
    pools: &Pools,
    amount: usize,
) -> Result<Vec<vk::CommandBuffer>, vk::Result> {
    let commandbuffer_allocate_info = vk::CommandBufferAllocateInfo::builder()
        .command_pool(pools.command_pool_graphics)
        .command_buffer_count(amount as u32);
    unsafe { logical_device.allocate_command_buffers(&commandbuffer_allocate_info) }
}

fn fill_commandbuffers(
    logical_device: &ash::Device,
    commandbuffers: &Vec<vk::CommandBuffer>,
    renderpass: &vk::RenderPass,
    swapchain: &Swapchain,
    pipeline: &Pipeline,
    vertex_buffer_1: &vk::Buffer,
    vertex_buffer_2: &vk::Buffer,
) -> Result<(), vk::Result> {
    for (i, &commandbuffer) in commandbuffers.iter().enumerate() {
        let commandbuffer_begin_info = vk::CommandBufferBeginInfo::builder();
        unsafe {
            logical_device.begin_command_buffer(commandbuffer, &commandbuffer_begin_info)?;
        }
        let clearvalues = [vk::ClearValue {
            color: vk::ClearColorValue {
                float32: [0.2, 0.2, 0.2, 1.0],
            },
        }];
        let renderpass_being_info = vk::RenderPassBeginInfo::builder()
            .render_pass(*renderpass)
            .framebuffer(swapchain.framebuffers[i])
            .render_area(vk::Rect2D {
                offset: vk::Offset2D { x: 0, y: 0 },
                extent: swapchain.extent,
            })
            .clear_values(&clearvalues);
        unsafe {
            logical_device.cmd_begin_render_pass(
                commandbuffer,
                &renderpass_being_info,
                vk::SubpassContents::INLINE,
            );
            logical_device.cmd_bind_pipeline(
                commandbuffer,
                vk::PipelineBindPoint::GRAPHICS,
                pipeline.pipeline,
            );
            logical_device.cmd_bind_vertex_buffers(commandbuffer, 0, &[*vertex_buffer_1], &[0]);
            logical_device.cmd_bind_vertex_buffers(commandbuffer, 1, &[*vertex_buffer_2], &[0]);
            logical_device.cmd_draw(commandbuffer, 6, 1, 0, 0);
            logical_device.cmd_end_render_pass(commandbuffer);
            logical_device.end_command_buffer(commandbuffer)?;
        }
    }
    Ok(())
}

struct Debug {
    loader: ash::extensions::ext::DebugUtils,
    messenger: vk::DebugUtilsMessengerEXT,
}

impl Debug {
    fn init(entry: &Entry, instance: &Instance) -> Result<Debug, vk::Result> {
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
        let loader = ash::extensions::ext::DebugUtils::new(&entry, &instance);
        let messenger = unsafe { loader.create_debug_utils_messenger(&debug_create_info, None)? };
        Ok(Debug { loader, messenger })
    }
}

impl Drop for Debug {
    fn drop(&mut self) {
        unsafe {
            self.loader
                .destroy_debug_utils_messenger(self.messenger, None)
        };
    }
}

struct Surface {
    surface: vk::SurfaceKHR,
    surface_loader: ash::extensions::khr::Surface,
}

impl Surface {
    fn init(
        window: &winit::window::Window,
        entry: &Entry,
        instance: &Instance,
    ) -> Result<Surface, vk::Result> {
        use raw_window_handle::{HasRawDisplayHandle, HasRawWindowHandle};
        let surface = unsafe {
            ash_window::create_surface(
                &entry,
                &instance,
                window.raw_display_handle(),
                window.raw_window_handle(),
                None,
            )?
        };
        let surface_loader = ash::extensions::khr::Surface::new(&entry, &instance);
        Ok(Surface {
            surface,
            surface_loader,
        })
    }

    fn get_capabilities(
        &self,
        physical_device: vk::PhysicalDevice,
    ) -> Result<vk::SurfaceCapabilitiesKHR, vk::Result> {
        unsafe {
            self.surface_loader
                .get_physical_device_surface_capabilities(physical_device, self.surface)
        }
    }

    fn get_present_modes(
        &self,
        physical_device: vk::PhysicalDevice,
    ) -> Result<Vec<vk::PresentModeKHR>, vk::Result> {
        unsafe {
            self.surface_loader
                .get_physical_device_surface_present_modes(physical_device, self.surface)
        }
    }

    fn get_formats(
        &self,
        physical_device: vk::PhysicalDevice,
    ) -> Result<Vec<vk::SurfaceFormatKHR>, vk::Result> {
        unsafe {
            self.surface_loader
                .get_physical_device_surface_formats(physical_device, self.surface)
        }
    }

    fn get_physical_device_surface_support(
        &self,
        physical_device: vk::PhysicalDevice,
        queue_family_index: usize,
    ) -> Result<bool, vk::Result> {
        unsafe {
            self.surface_loader.get_physical_device_surface_support(
                physical_device,
                queue_family_index as u32,
                self.surface,
            )
        }
    }
}

impl Drop for Surface {
    fn drop(&mut self) {
        unsafe {
            self.surface_loader.destroy_surface(self.surface, None);
        }
    }
}

struct QueueFamilies {
    graphics_queue_index: Option<u32>,
    transfer_queue_index: Option<u32>,
}

impl QueueFamilies {
    fn init(
        instance: &Instance,
        physical_device: vk::PhysicalDevice,
        surfaces: &Surface,
    ) -> Result<QueueFamilies, vk::Result> {
        let queue_family_properties =
            unsafe { instance.get_physical_device_queue_family_properties(physical_device) };
        let mut found_graphics_queue_index = None;
        let mut found_transfer_queue_index = None;
        for (index, queue_family) in queue_family_properties.iter().enumerate() {
            if queue_family.queue_count > 0
                && queue_family.queue_flags.contains(vk::QueueFlags::GRAPHICS)
                && surfaces.get_physical_device_surface_support(physical_device, index)?
            {
                found_graphics_queue_index = Some(index as u32);
            }
            if queue_family.queue_count > 0
                && queue_family.queue_flags.contains(vk::QueueFlags::TRANSFER)
            {
                if found_transfer_queue_index.is_none()
                    || !queue_family.queue_flags.contains(vk::QueueFlags::GRAPHICS)
                {
                    found_transfer_queue_index = Some(index as u32);
                }
            }
        }
        Ok(QueueFamilies {
            graphics_queue_index: found_graphics_queue_index,
            transfer_queue_index: found_transfer_queue_index,
        })
    }
}

struct Swapchain {
    swapchain_loader: ash::extensions::khr::Swapchain,
    swapchain: vk::SwapchainKHR,
    images: Vec<vk::Image>,
    image_views: Vec<vk::ImageView>,
    framebuffers: Vec<vk::Framebuffer>,
    surface_format: vk::SurfaceFormatKHR,
    extent: vk::Extent2D,
    image_available: Vec<vk::Semaphore>,
    rendering_finished: Vec<vk::Semaphore>,
    may_begin_drawing: Vec<vk::Fence>,
    amount_of_images: u32,
    current_image: usize,
}

impl Swapchain {
    fn init(
        instance: &Instance,
        surfaces: &Surface,
        physical_device: vk::PhysicalDevice,
        logical_device: &ash::Device,
        queue_families: &QueueFamilies,
        queues: &Queues,
    ) -> Result<Swapchain, vk::Result> {
        let surface_capabilities = surfaces.get_capabilities(physical_device)?;
        let surface_format = *surfaces.get_formats(physical_device)?.first().unwrap();
        let queue_family_graphics = [queue_families.graphics_queue_index.unwrap()];
        let extent = surface_capabilities.current_extent;
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

        Ok(Swapchain {
            swapchain_loader,
            swapchain,
            images: swapchain_images,
            image_views: swapchain_imageviews,
            framebuffers: vec![],
            surface_format,
            extent,
            image_available,
            rendering_finished,
            may_begin_drawing,
            amount_of_images,
            current_image: 0,
        })
    }

    fn create_framebuffers(
        &mut self,
        logical_device: &ash::Device,
        renderpass: vk::RenderPass,
    ) -> Result<(), vk::Result> {
        for image_view in &self.image_views {
            let view = [*image_view];
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

    unsafe fn cleanup(&self, logical_device: &ash::Device) {
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

struct Pipeline {
    pipeline: vk::Pipeline,
    layout: vk::PipelineLayout,
}
impl Pipeline {
    fn cleanup(&mut self, logical_device: &ash::Device) {
        unsafe {
            logical_device.destroy_pipeline(self.pipeline, None);
            logical_device.destroy_pipeline_layout(self.layout, None);
        }
    }
    fn init(
        logical_device: &ash::Device,
        swapchain: &Swapchain,
        renderpass: &vk::RenderPass,
    ) -> Result<Pipeline, vk::Result> {
        let vertex_shader_create_info = vk::ShaderModuleCreateInfo::builder().code(
            vk_shader_macros::include_glsl!("./shaders/shader.vert", kind: vert),
        );
        let vertex_shader_module =
            unsafe { logical_device.create_shader_module(&vertex_shader_create_info, None)? };

        let fragment_shader_create_info = vk::ShaderModuleCreateInfo::builder().code(
            vk_shader_macros::include_glsl!("./shaders/shader.frag", kind: frag),
        );
        let fragment_shader_module =
            unsafe { logical_device.create_shader_module(&fragment_shader_create_info, None)? };
        let main_function_name = ffi::CString::new("main").unwrap();
        let vertex_shader_stage = vk::PipelineShaderStageCreateInfo::builder()
            .stage(vk::ShaderStageFlags::VERTEX)
            .module(vertex_shader_module)
            .name(&main_function_name);
        let fragment_shader_stage = vk::PipelineShaderStageCreateInfo::builder()
            .stage(vk::ShaderStageFlags::FRAGMENT)
            .module(fragment_shader_module)
            .name(&main_function_name);
        let shader_stages = vec![vertex_shader_stage.build(), fragment_shader_stage.build()];

        let vertex_attribute_descriptions = [
            vk::VertexInputAttributeDescription {
                binding: 0,
                location: 0,
                offset: 0,
                format: vk::Format::R32G32B32A32_SFLOAT,
            },
            vk::VertexInputAttributeDescription {
                binding: 1,
                location: 1,
                offset: 0,
                format: vk::Format::R32_SFLOAT,
            },
            vk::VertexInputAttributeDescription {
                binding: 1,
                location: 2,
                offset: 4,
                format: vk::Format::R32G32B32A32_SFLOAT,
            },
        ];

        let vertex_binding_descriptions = [
            vk::VertexInputBindingDescription {
                binding: 0,
                stride: 16,
                input_rate: vk::VertexInputRate::VERTEX,
            },
            vk::VertexInputBindingDescription {
                binding: 1,
                stride: 20,
                input_rate: vk::VertexInputRate::VERTEX,
            },
        ];

        let vertex_input_info = vk::PipelineVertexInputStateCreateInfo::builder()
            .vertex_attribute_descriptions(&vertex_attribute_descriptions)
            .vertex_binding_descriptions(&vertex_binding_descriptions);
        let input_assembly_info = vk::PipelineInputAssemblyStateCreateInfo::builder()
            .topology(vk::PrimitiveTopology::TRIANGLE_LIST);

        let viewports = [vk::Viewport {
            x: 0.0,
            y: 0.0,
            width: swapchain.extent.width as f32,
            height: swapchain.extent.height as f32,
            min_depth: 0.0,
            max_depth: 1.0,
        }];
        let scissors = [vk::Rect2D {
            offset: vk::Offset2D { x: 0, y: 0 },
            extent: swapchain.extent,
        }];
        let viewport_info = vk::PipelineViewportStateCreateInfo::builder()
            .viewports(&viewports)
            .scissors(&scissors);
        let rasterizer_info = vk::PipelineRasterizationStateCreateInfo::builder()
            .line_width(1.0)
            .front_face(vk::FrontFace::COUNTER_CLOCKWISE)
            .cull_mode(vk::CullModeFlags::NONE)
            .polygon_mode(vk::PolygonMode::FILL);
        let multisampler_info = vk::PipelineMultisampleStateCreateInfo::builder()
            .rasterization_samples(vk::SampleCountFlags::TYPE_1);
        let colorblend_attachments = [vk::PipelineColorBlendAttachmentState::builder()
            .blend_enable(true)
            .src_color_blend_factor(vk::BlendFactor::SRC_ALPHA)
            .dst_color_blend_factor(vk::BlendFactor::ONE_MINUS_SRC_ALPHA)
            .color_blend_op(vk::BlendOp::ADD)
            .src_alpha_blend_factor(vk::BlendFactor::SRC_ALPHA)
            .dst_alpha_blend_factor(vk::BlendFactor::ONE_MINUS_SRC_ALPHA)
            .alpha_blend_op(vk::BlendOp::ADD)
            .color_write_mask(
                vk::ColorComponentFlags::R
                    | vk::ColorComponentFlags::G
                    | vk::ColorComponentFlags::B
                    | vk::ColorComponentFlags::A,
            )
            .build()];
        let colorblend_info =
            vk::PipelineColorBlendStateCreateInfo::builder().attachments(&colorblend_attachments);
        let pipeline_layout_info = vk::PipelineLayoutCreateInfo::builder();
        let pipeline_layout =
            unsafe { logical_device.create_pipeline_layout(&pipeline_layout_info, None)? };
        let pipeline_info = vk::GraphicsPipelineCreateInfo::builder()
            .stages(&shader_stages)
            .vertex_input_state(&vertex_input_info)
            .input_assembly_state(&input_assembly_info)
            .viewport_state(&viewport_info)
            .rasterization_state(&rasterizer_info)
            .multisample_state(&multisampler_info)
            .color_blend_state(&colorblend_info)
            .layout(pipeline_layout)
            .render_pass(*renderpass)
            .subpass(0);
        let graphics_pipeline = unsafe {
            logical_device
                .create_graphics_pipelines(
                    vk::PipelineCache::null(),
                    &[pipeline_info.build()],
                    None,
                )
                .expect("There was a problem creating the graphics pipeline")
        }[0];
        unsafe {
            logical_device.destroy_shader_module(fragment_shader_module, None);
            logical_device.destroy_shader_module(vertex_shader_module, None);
        }
        Ok(Pipeline {
            pipeline: graphics_pipeline,
            layout: pipeline_layout,
        })
    }
}

struct Pools {
    command_pool_graphics: vk::CommandPool,
    command_pool_transfer: vk::CommandPool,
}

impl Pools {
    fn init(
        logical_device: &ash::Device,
        queue_families: &QueueFamilies,
    ) -> Result<Pools, vk::Result> {
        let graphics_command_pool_info = vk::CommandPoolCreateInfo::builder()
            .queue_family_index(queue_families.graphics_queue_index.unwrap())
            .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER);
        let command_pool_graphics =
            unsafe { logical_device.create_command_pool(&graphics_command_pool_info, None) }?;
        let transfer_command_pool_info = vk::CommandPoolCreateInfo::builder()
            .queue_family_index(queue_families.transfer_queue_index.unwrap())
            .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER);
        let command_pool_transfer =
            unsafe { logical_device.create_command_pool(&transfer_command_pool_info, None) }?;
        Ok(Pools {
            command_pool_graphics,
            command_pool_transfer,
        })
    }
    fn cleanup(&self, logical_device: &ash::Device) {
        unsafe {
            logical_device.destroy_command_pool(self.command_pool_graphics, None);
            logical_device.destroy_command_pool(self.command_pool_transfer, None);
        }
    }
}

struct Buffer {
    buffer: vk::Buffer,
    allocation: gpu_allocator::vulkan::Allocation,
}

impl Buffer {
    fn new(
        logical_device: &ash::Device,
        allocator: &mut gpu_allocator::vulkan::Allocator,
        size: u64,
    ) -> Buffer {
        let buffer_create_info = vk::BufferCreateInfo::builder()
            .size(size)
            .usage(vk::BufferUsageFlags::VERTEX_BUFFER)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);

        let buffer = unsafe {
            logical_device
                .create_buffer(&buffer_create_info, None)
                .expect("Failed to create buffer")
        };

        let memory_requirements = unsafe { logical_device.get_buffer_memory_requirements(buffer) };
        let location = gpu_allocator::MemoryLocation::CpuToGpu;

        let allocation_create_info = gpu_allocator::vulkan::AllocationCreateDesc {
            requirements: memory_requirements,
            location,
            linear: true,
            name: "Buffer",
            allocation_scheme: gpu_allocator::vulkan::AllocationScheme::GpuAllocatorManaged,
        };

        let allocation = allocator
            .allocate(&allocation_create_info)
            .expect("Failed to allocate memory for buffer");

        unsafe {
            logical_device
                .bind_buffer_memory(buffer, allocation.memory(), allocation.offset())
                .expect("Failed to bind buffer");
        };

        Buffer { buffer, allocation }
    }

    fn fill(&mut self, data: &[f32]) {
        let destination = self.allocation.mapped_ptr().unwrap().cast().as_ptr();
        unsafe {
            memcpy::<f32>(data.as_ptr(), destination, data.len());
        };
    }

    fn cleanup(
        &mut self,
        logical_device: &ash::Device,
        allocator: &mut gpu_allocator::vulkan::Allocator,
    ) {
        unsafe {
            logical_device.destroy_buffer(self.buffer, None);
        }
        allocator
            .free(std::mem::take(&mut self.allocation))
            .expect("Failed to free buffer memory");
    }
}


struct Vulkano {
    window: winit::window::Window,
    entry: Entry,
    instance: Instance,
    debug: std::mem::ManuallyDrop<Debug>,
    surfaces: std::mem::ManuallyDrop<Surface>,
    physical_device: vk::PhysicalDevice,
    physical_device_properties: vk::PhysicalDeviceProperties,
    physical_device_features: vk::PhysicalDeviceFeatures,
    queue_families: QueueFamilies,
    queues: Queues,
    device: ash::Device,
    swapchain: Swapchain,
    renderpass: vk::RenderPass,
    pipeline: Pipeline,
    pools: Pools,
    commandbuffers: Vec<vk::CommandBuffer>,
    buffers: Vec<Buffer>,
    allocator: std::mem::ManuallyDrop<gpu_allocator::vulkan::Allocator>
}

impl Vulkano {
    fn init(window: winit::window::Window) -> Result<Vulkano, Box<dyn std::error::Error>> {
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
            &queues,
        )?;
        let renderpass = init_renderpass(
            &logical_device,
            physical_device,
            swapchain.surface_format.format,
        )?;
        swapchain.create_framebuffers(&logical_device, renderpass)?;

        let pipeline = Pipeline::init(&logical_device, &swapchain, &renderpass)?;
        let pools = Pools::init(&logical_device, &queue_families)?;

        let mut allocator_create_description = gpu_allocator::vulkan::AllocatorCreateDesc {
            instance: instance.clone(),
            device: logical_device.clone(),
            physical_device,
            debug_settings: Default::default(),
            buffer_device_address: false,
        };

        let mut allocator = gpu_allocator::vulkan::Allocator::new(&allocator_create_description)
            .expect("Failed to create allocator");

        let data_1 = [
            0.5f32, 0.0f32, 0.0f32, 1.0f32, 0.0f32, 0.2f32, 0.0f32, 1.0f32, -0.5f32, 0.0f32,
            0.0f32, 1.0f32, -0.9f32, -0.9f32, 0.0f32, 1.0f32, 0.3f32, -0.8f32, 0.0f32, 1.0f32,
            0.0f32, -0.6f32, 0.0f32, 1.0f32,
        ];
        let mut buffer_1 = Buffer::new(&logical_device, &mut allocator, size_of_val(&data_1) as u64);
        buffer_1.fill(&data_1);

        let data_2 = [
            15.0f32, 0.0f32, 1.0f32, 0.0f32, 1.0f32, 15.0f32, 0.0f32, 1.0f32, 0.0f32, 1.0f32,
            15.0f32, 0.0f32, 1.0f32, 0.0f32, 1.0f32, 1.0f32, 0.8f32, 0.7f32, 0.0f32, 1.0f32,
            1.0f32, 0.8f32, 0.7f32, 0.0f32, 1.0f32, 1.0f32, 0.0f32, 0.0f32, 1.0f32, 1.0f32,
        ];
        let mut buffer_2 = Buffer::new(&logical_device, &mut allocator, size_of_val(&data_2) as u64);
        buffer_2.fill(&data_2);

        let commandbuffers =
            create_commandbuffers(&logical_device, &pools, swapchain.framebuffers.len())?;
        fill_commandbuffers(
            &logical_device,
            &commandbuffers,
            &renderpass,
            &swapchain,
            &pipeline,
            &buffer_1.buffer,
            &buffer_2.buffer,
        )?;

        Ok(Vulkano {
            window,
            entry,
            instance,
            debug: std::mem::ManuallyDrop::new(debug),
            surfaces: std::mem::ManuallyDrop::new(surfaces),
            physical_device,
            physical_device_properties,
            physical_device_features,
            queue_families,
            queues,
            device: logical_device,
            swapchain,
            renderpass,
            pipeline,
            pools,
            commandbuffers,
            buffers: vec![buffer_1, buffer_2],
            allocator: std::mem::ManuallyDrop::new(allocator)
        })
    }
}

impl Drop for Vulkano {
    fn drop(&mut self) {
        unsafe {
            self.device
                .device_wait_idle()
                .expect("Something went wrong while waiting");
            for buf in &mut self.buffers {
                buf.cleanup(&self.device, &mut self.allocator);
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
