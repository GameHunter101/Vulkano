use ash::vk;
use nalgebra as na;
use winit::{dpi::PhysicalSize, window::WindowBuilder};

mod vulkan {
    pub mod buffer;
    pub mod camera;
    pub mod debug;
    pub mod light;
    pub mod model;
    pub mod pipeline;
    pub mod pools;
    pub mod queues;
    pub mod surface;
    pub mod swapchain;
    pub mod utils;
    pub mod vulkano;
}

use vulkan::buffer::Buffer;
use vulkan::camera::Camera;
use vulkan::debug::Debug;
use vulkan::light::*;
use vulkan::model::*;
use vulkan::pipeline::Pipeline;
use vulkan::pools::Pools;
use vulkan::queues::*;
use vulkan::surface::Surface;
use vulkan::swapchain::Swapchain;
use vulkan::utils::*;
use vulkan::vulkano::Vulkano;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let event_loop = winit::event_loop::EventLoop::new();
    let window = WindowBuilder::new()
        .with_inner_size(PhysicalSize::<u32>::from((800, 600)))
        .with_title("Vulkano")
        .build(&event_loop)?;

    let mut vulkano = Vulkano::init(window)?;

    let mut camera = Camera::default();

    let mut sphere = Model::sphere(3);
    let mut lights = LightManager::default();

    for i in 0..10 {
        for j in 0..10 {
            sphere.insert_visibly(InstanceData::from_matrix_and_properties(
                na::Matrix4::new_translation(&na::Vector3::new(i as f32 - 5., j as f32 + 5., 10.0))
                    * na::Matrix4::new_scaling(0.5),
                [0., 0., 0.8],
                i as f32 * 0.1,
                j as f32 * 0.1,
            ));
        }
    }

    lights.add_light(DirectionalLight {
        direction: na::Vector3::new(-1., -1., 0.),
        illuminance: [10.1, 10.1, 10.1],
    });
    lights.add_light(PointLight {
        position: na::Point3::new(0.1, -3.0, -3.0),
        luminous_flux: [100.0, 100.0, 100.0],
    });
    lights.add_light(PointLight {
        position: na::Point3::new(0.1, -3.0, -3.0),
        luminous_flux: [100.0, 100.0, 100.0],
    });
    lights.add_light(PointLight {
        position: na::Point3::new(0.1, -3.0, -3.0),
        luminous_flux: [100.0, 100.0, 100.0],
    });

    lights.update_buffer(
        &vulkano.device,
        &mut vulkano.light_buffer,
        &mut vulkano.descriptor_sets_light,
    );

    sphere
        .update_vertex_buffer(&vulkano.device, &mut vulkano.allocator)
        .unwrap();
    sphere
        .update_index_buffer(&vulkano.device, &mut vulkano.allocator)
        .unwrap();
    sphere
        .update_instance_buffer(&vulkano.device, &mut vulkano.allocator)
        .unwrap();
    vulkano.models = vec![sphere];

    let mut mouse_pos: Option<winit::dpi::PhysicalPosition<f64>> = None;

    use winit::event::{Event, WindowEvent};
    event_loop.run(move |event, _, control_flow| match event {
        Event::WindowEvent {
            event: WindowEvent::CursorLeft { .. },
            ..
        } => {
            mouse_pos = None;
        }

        Event::WindowEvent {
            event: WindowEvent::CursorMoved { position, .. },
            ..
        } => {
            if let Some(mouse_pos) = mouse_pos {
                camera.turn_right(((position.x - mouse_pos.x) * 0.01) as f32);
                camera.turn_up(((mouse_pos.y - position.y) * 0.01) as f32);
            }
            mouse_pos = Some(position);
        }

        Event::WindowEvent {
            event: WindowEvent::KeyboardInput { input, .. },
            ..
        } => {
            if let winit::event::KeyboardInput {
                state: winit::event::ElementState::Pressed,
                virtual_keycode: Some(keycode),
                ..
            } = input
            {
                let distance = 0.1;
                match keycode {
                    winit::event::VirtualKeyCode::W => {
                        camera.move_forward(distance);
                    }
                    winit::event::VirtualKeyCode::S => {
                        camera.move_backward(distance);
                    }
                    winit::event::VirtualKeyCode::A => {
                        camera.move_left(distance);
                    }
                    winit::event::VirtualKeyCode::D => {
                        camera.move_right(distance);
                    }
                    winit::event::VirtualKeyCode::F11 => {
                        screenshot(&vulkano).expect("Trouble taking screenshot");
                    }
                    _ => {}
                }
            }
        }

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
                camera.update_buffer(&mut vulkano.uniform_buffer);
                for model in &mut vulkano.models {
                    model
                        .update_instance_buffer(&vulkano.device, &mut vulkano.allocator)
                        .unwrap();
                }
                vulkano
                    .update_command_buffer(image_index as usize)
                    .expect("Updating the command buffer");
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
                match vulkano
                    .swapchain
                    .swapchain_loader
                    .queue_present(vulkano.queues.graphics_queue, &present_info)
                {
                    Ok(..) => {}
                    Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => {
                        vulkano.recreate_swapchain().expect("Swapchain recreation");
                        camera.set_aspect(
                            vulkano.swapchain.extent.width as f32
                                / vulkano.swapchain.extent.height as f32,
                        );
                        camera.update_buffer(&mut vulkano.uniform_buffer);
                    }
                    _ => {
                        panic!("Unhandled queue presentation error");
                    }
                }
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
