#![allow(unused)]

use std::collections::HashMap;
use std::sync::{
    atomic::{AtomicBool, Ordering},
    mpsc, Arc, Mutex,
};
use std::thread;
use std::time::{Duration, Instant};

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
    pub mod text;
    pub mod texture;
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
    // window.set_cursor_visible(false);
    // window.set_cursor_grab(winit::window::CursorGrabMode::Locked)?;

    let mut vulkano = Vulkano::init(&window)?;

    let mut camera = Arc::new(Mutex::new(Camera::default()));

    let mut screen_quad = Model::screen_quad();

    let mut lights = LightManager::default();

    screen_quad.insert_visibly(InstanceData::screen_quad(
        camera.lock().unwrap().view_matrix,
    ));

    lights.update_buffer(
        &vulkano.device,
        &mut vulkano.light_buffer,
        &mut vulkano.lights_descriptor_sets,
    );

    screen_quad
        .update_vertex_buffer(&vulkano.device, &mut vulkano.allocator)
        .unwrap();
    screen_quad
        .update_index_buffer(&vulkano.device, &mut vulkano.allocator)
        .unwrap();
    screen_quad
        .update_instance_buffer(&vulkano.device, &mut vulkano.allocator)
        .unwrap();

    vulkano.models = vec![];
    vulkano.screen_quad = Some(screen_quad);

    let mut mouse_pos: Option<winit::dpi::PhysicalPosition<f64>> = None;
    let mut thread_handle: Option<thread::JoinHandle<()>> = None;
    let mut key_pressed: Option<winit::event::VirtualKeyCode> = None;
    // let (tx, rx) = mpsc::channel::<Option<()>>();
    // let rx = Arc::new(Mutex::new(rx));
    let mut map =
        HashMap::<winit::event::VirtualKeyCode, (thread::JoinHandle<()>, Arc<AtomicBool>)>::new();

    use winit::event::{Event, WindowEvent};
    event_loop.run(move |event, _, control_flow| match event {
        /* Event::WindowEvent { event: WindowEvent::MouseInput { state,button, ..},.. } => {
            if button == winit::event::MouseButton::Left && state == winit::event::ElementState::Pressed {
                window.set_cursor_visible(false);
                window.set_cursor_grab(winit::window::CursorGrabMode::Locked);
            }
        } */
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
                camera
                    .lock()
                    .unwrap()
                    .update_rotation([position.x - mouse_pos.x, mouse_pos.y - position.y]);
            }
            mouse_pos = Some(position);
        }

        Event::WindowEvent {
            event:
                WindowEvent::KeyboardInput {
                    input:
                        winit::event::KeyboardInput {
                            state: key_state,
                            virtual_keycode: Some(virtual_keycode),
                            ..
                        },
                    ..
                },
            ..
        } => match key_state {
            winit::event::ElementState::Pressed => {
                if !map.contains_key(&virtual_keycode) {
                    let flag = Arc::new(AtomicBool::new(true));
                    let flag_clone = flag.clone();
                    let distance = 0.005;
                    let camera_clone = Arc::clone(&camera);
                    let handle = thread::spawn(move || {
                        let mut last_frame_time = Instant::now();
                        loop {
                            if !flag_clone.load(Ordering::SeqCst) {
                                break;
                            }

                            let elapsed = last_frame_time.elapsed();
                            last_frame_time = Instant::now();
                            let frame_duration = elapsed.as_millis() as f32;

                            match virtual_keycode {
                                winit::event::VirtualKeyCode::W => {
                                    camera_clone.lock().unwrap().move_forward(distance * frame_duration);
                                }
                                winit::event::VirtualKeyCode::S => {
                                    camera_clone.lock().unwrap().move_backward(distance * frame_duration);
                                }
                                winit::event::VirtualKeyCode::A => {
                                    camera_clone.lock().unwrap().move_left(distance * frame_duration);
                                }
                                winit::event::VirtualKeyCode::D => {
                                    camera_clone.lock().unwrap().move_right(distance * frame_duration);
                                }
                                winit::event::VirtualKeyCode::Space => {
                                    camera_clone.lock().unwrap().move_up(distance * frame_duration);
                                }
                                winit::event::VirtualKeyCode::LControl => {
                                    camera_clone.lock().unwrap().move_down(distance * frame_duration);
                                }
                                _ => {}
                            };
                            thread::sleep(Duration::from_micros(10));
                        }
                    });
                    map.insert(virtual_keycode, (handle, flag));
                }
            }
            winit::event::ElementState::Released => {
                if let Some((handle, flag)) = map.remove(&virtual_keycode) {
                    flag.store(false, Ordering::SeqCst);
                    // thread::spawn(move || {
                    handle.join().expect("Failed to join thread");
                    // });
                }
            }
        },
        /* let distance = 0.1;
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
            winit::event::VirtualKeyCode::Escape => {
                window.set_cursor_visible(true);
                window.set_cursor_grab(winit::window::CursorGrabMode::None);
            }
            _ => {}
        } */
        Event::WindowEvent {
            event: WindowEvent::CloseRequested,
            ..
        } => {
            if let Some(handle) = thread_handle.take() {
                handle.join().unwrap();
            }
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
                camera.lock().unwrap().update_buffer(
                    &mut vulkano.uniform_buffer,
                    vulkano.swapchain.extent.width,
                    vulkano.swapchain.extent.height,
                );
                for model in &mut vulkano.models {
                    match model {
                        ModelTypes::Normal(normal) => normal
                            .update_instance_buffer(&vulkano.device, &mut vulkano.allocator)
                            .unwrap(),
                        ModelTypes::Textured(textured) => textured
                            .update_instance_buffer(&vulkano.device, &mut vulkano.allocator)
                            .unwrap(),
                    }
                }

                /* let imageinfos = vulkano.texture_storage.get_descriptor_image_info();
                let descriptorwrite_image = vk::WriteDescriptorSet::builder()
                    .dst_set(vulkano.descriptor_sets_texture[image_index as usize])
                    .dst_binding(0)
                    .dst_array_element(0)
                    .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                    .image_info(&imageinfos)
                    .build();

                vulkano
                    .device
                    .update_descriptor_sets(&[descriptorwrite_image], &[]); */

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
                        camera.lock().unwrap().update_buffer(
                            &mut vulkano.uniform_buffer,
                            vulkano.swapchain.extent.width,
                            vulkano.swapchain.extent.height,
                        );
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
            window.request_redraw();
        }
        _ => {}
    });
}
