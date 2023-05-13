use std::ptr::copy_nonoverlapping;

use ash::vk;

pub struct Buffer {
    pub buffer: vk::Buffer,
    allocation: gpu_allocator::vulkan::Allocation,
}

impl Buffer {
    pub fn new(
        logical_device: &ash::Device,
        allocator: &mut gpu_allocator::vulkan::Allocator,
        size: u64,
        usage: vk::BufferUsageFlags,
    ) -> Buffer {
        let buffer_create_info = vk::BufferCreateInfo::builder()
            .size(size)
            .usage(usage)
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

    pub fn fill<T>(&mut self, data: &[T]) {
        let destination = self.allocation.mapped_ptr().unwrap().cast().as_ptr();
        unsafe {
            copy_nonoverlapping::<T>(data.as_ptr(), destination, data.len());
        };
    }

    pub fn cleanup(
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
