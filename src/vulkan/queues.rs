use ash::{vk, Instance};

use crate::Surface;

pub struct QueueFamilies {
    pub graphics_queue_index: Option<u32>,
    pub transfer_queue_index: Option<u32>,
}

impl QueueFamilies {
    pub fn init(
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

pub struct Queues {
    pub graphics_queue: vk::Queue,
    pub transfer_queue: vk::Queue,
}
