pub mod index_buffer;
pub mod vertex_buffer;

//why use and not just mod?
pub use index_buffer::*;
use std::marker::PhantomData;
pub use vertex_buffer::*;

use crate::{error::*, vulkan_abstraction};
use ash::vk;
use ash::vk::Handle;
use std::rc::Rc;
//TODO divide into a trait gpuonly and hostmemoery accessible and then do custom new functions with custom flags by leveraging strong typing

pub fn get_memory_type_index(
    core: &vulkan_abstraction::Core,
    mem_prop_flags: vk::MemoryPropertyFlags,
    mem_requirements: &vk::MemoryRequirements,
) -> SrResult<u32> {
    type BitsType = u32;
    let bits: BitsType = mem_requirements.memory_type_bits;
    assert_ne!(bits, 0);

    let mem_types = core.device().memory_properties().memory_types;
    let mut idx = -1;
    for i in 0..(8 * size_of::<BitsType>()) {
        let mem_type_is_supported = bits & (1 << i) != 0;
        if mem_type_is_supported {
            if mem_types[i].property_flags & mem_prop_flags == mem_prop_flags {
                idx = i as isize;
                break;
            }
        }
    }
    if idx < 0 {
        return Err(SrError::new_custom("Vertex Buffer Memory Type not supported!".to_string()));
    }

    Ok(idx as u32)
}
pub trait Buffer {
    fn inner(&self) -> vk::Buffer;

    fn raw(&self) -> &vulkan_abstraction::RawBuffer;

    fn raw_mut(&mut self) -> &mut vulkan_abstraction::RawBuffer;

    fn byte_size(&self) -> vk::DeviceSize;
    fn is_null(&self) -> bool;
    fn get_device_address(&self) -> vk::DeviceAddress;
    fn new_null(core: Rc<vulkan_abstraction::Core>) -> Self
    where
        Self: Sized;
}

/// Exclusive trait for host-visible buffers (CpuToGpu or GpuToCpu) that can be mapped.
pub trait HostAccessibleBuffer<T>: Buffer {
    fn map_mut(&mut self) -> SrResult<&mut [T]>;

    fn map(& self) -> SrResult<& [T]>;

    fn len(&self) -> usize ;
}

pub(crate) struct RawBuffer {
    core: Rc<vulkan_abstraction::Core>,
    buffer: vk::Buffer,
    allocation: gpu_allocator::vulkan::Allocation,
    byte_size: u64,
}

impl RawBuffer {
    pub fn new_aligned(
        core: Rc<vulkan_abstraction::Core>,
        byte_size: vk::DeviceSize,
        alignment: u64,
        memory_location: gpu_allocator::MemoryLocation,
        buffer_usage_flags: vk::BufferUsageFlags,
        name: &'static str,
    ) -> SrResult<Self> {
        if byte_size == 0 {
            return Ok(Self::new_null(core));
        }

        let device = core.device().inner();
        let buffer = {
            let buf_info = vk::BufferCreateInfo::default()
                .size(byte_size)
                .usage(buffer_usage_flags)
                .sharing_mode(vk::SharingMode::EXCLUSIVE);

            unsafe { device.create_buffer(&buf_info, None) }?
        };

        let mem_requirements = unsafe { device.get_buffer_memory_requirements(buffer) };
        let mem_requirements = mem_requirements.alignment(mem_requirements.alignment.max(alignment));

        let allocation = core.allocator_mut().allocate(&gpu_allocator::vulkan::AllocationCreateDesc {
            name,
            requirements: mem_requirements,
            location: memory_location,
            linear: true,
            allocation_scheme: gpu_allocator::vulkan::AllocationScheme::GpuAllocatorManaged,
        })?;

        unsafe { device.bind_buffer_memory(buffer, allocation.memory(), allocation.offset()) }?;

        Ok(Self {
            core,
            buffer,
            allocation,
            byte_size,
        })
    }

    pub fn new_null(core: Rc<vulkan_abstraction::Core>) -> Self {
        Self {
            core,
            buffer: vk::Buffer::null(),
            allocation: gpu_allocator::vulkan::Allocation::default(),
            byte_size: 0,
        }
    }

    pub fn map<V: Sized>(& self) -> SrResult<& [V]> {
        if !self.buffer.is_null() {
            let slice = self.allocation.mapped_slice().unwrap();
            let ret = unsafe { std::slice::from_raw_parts_mut(slice.as_ptr() as *mut V, slice.len() / std::mem::size_of::<V>()) };
            Ok(ret)
        } else {
            Ok(&mut [])
        }
    }

    pub fn map_mut<V: Sized>(&mut self) -> SrResult<&mut [V]> {
        if !self.buffer.is_null() {
            let slice = self.allocation.mapped_slice_mut().unwrap();
            let ret = unsafe { std::slice::from_raw_parts_mut(slice.as_ptr() as *mut V, slice.len() / std::mem::size_of::<V>()) };
            Ok(ret)
        } else {
            Ok(&mut [])
        }
    }
}

impl Drop for RawBuffer {
    fn drop(&mut self) {
        if self.buffer != vk::Buffer::null() {
            let device = self.core.device().inner();
            unsafe {
                device.destroy_buffer(self.buffer, None);
            }
            let allocation = std::mem::replace(&mut self.allocation, gpu_allocator::vulkan::Allocation::default());
            if let Err(e) = self.core.allocator_mut().free(allocation) {
                log::error!("Allocator::free returned {e} in RawBuffer::drop");
            }
        }
    }
}
macro_rules! impl_buffer_trait {
    ($type:ident < $first_gen:ident $(, $rest_gens:ident)* >) => {
        impl < $first_gen $(, $rest_gens)* > Buffer for $type < $first_gen $(, $rest_gens)* > {
            fn inner(&self) -> vk::Buffer {
                self.raw.buffer
            }

            fn raw(&self) -> &vulkan_abstraction::RawBuffer {
                &self.raw
            }

             fn raw_mut(&mut self) -> &mut vulkan_abstraction::RawBuffer {
                &mut self.raw
            }


            fn byte_size(&self) -> vk::DeviceSize {
                self.raw.byte_size
            }

            fn is_null(&self) -> bool {
                self.raw.buffer == vk::Buffer::null()
            }

            fn get_device_address(&self) -> vk::DeviceAddress {
                if self.is_null() {
                    return 0;
                }
                let info = vk::BufferDeviceAddressInfo::default().buffer(self.raw.buffer);
                unsafe { self.raw.core.device().inner().get_buffer_device_address(&info) }
            }

            fn new_null(core: Rc<vulkan_abstraction::Core>) -> Self {
                Self {
                    raw: RawBuffer::new_null(core),
                    _marker: Default::default(),
                }
            }
        }
    };

    ($type:ident) => {
        impl Buffer for $type {
            fn inner(&self) -> vk::Buffer {
                self.raw.buffer
            }

             fn raw(&self) -> &vulkan_abstraction::RawBuffer {
                &self.raw
            }

             fn raw_mut(&mut self) -> &mut vulkan_abstraction::RawBuffer {
                &mut self.raw
            }


            fn byte_size(&self) -> vk::DeviceSize {
                self.raw.byte_size
            }

            fn is_null(&self) -> bool {
                self.raw.buffer == vk::Buffer::null()
            }

            fn get_device_address(&self) -> vk::DeviceAddress {
                if self.is_null() {
                    return 0;
                }
                let info = vk::BufferDeviceAddressInfo::default().buffer(self.raw.buffer);
                unsafe { self.raw.core.device().inner().get_buffer_device_address(&info) }
            }

            fn new_null(core: Rc<vulkan_abstraction::Core>) -> Self {
                Self {
                    raw: RawBuffer::new_null(core),
                }
            }
        }
    };
}

// --- 1. Staging Buffer (CpuToGpu, Mappable) ---

pub struct StagingBuffer<T> {
    raw: RawBuffer,
    _marker: PhantomData<T>,
}

impl_buffer_trait!(StagingBuffer<T>);

impl<T> StagingBuffer<T> {
    pub fn new_temp(core: Rc<vulkan_abstraction::Core>, len: usize) -> SrResult<Self> {
        let byte_size = (len * std::mem::size_of::<T>()) as vk::DeviceSize;
        let raw = RawBuffer::new_aligned(
            core,
            byte_size,
            1,
            gpu_allocator::MemoryLocation::CpuToGpu,
            vk::BufferUsageFlags::TRANSFER_SRC | vk::BufferUsageFlags::TRANSFER_DST,
            "staging buffer",
        )?;
        Ok(Self {
            raw,
            _marker: PhantomData,
        })
    }

    pub fn new(core: Rc<vulkan_abstraction::Core>, len: usize ,buffer_usage_flags: vk::BufferUsageFlags, name: &'static str ) -> SrResult<Self> {
        let byte_size = (len * std::mem::size_of::<T>()) as vk::DeviceSize;
        let raw = RawBuffer::new_aligned(
            core,
            byte_size,
            1,
            gpu_allocator::MemoryLocation::CpuToGpu,
            buffer_usage_flags,
            name,
        )?;
        Ok(Self {
            raw,
            _marker: PhantomData,
        })
    }
    

    pub fn new_from_data(core: Rc<vulkan_abstraction::Core>, data: &[T]) -> SrResult<Self>
    where
        T: Copy,
    {
        if data.len() == 0 {
            return Ok(Self::new_null(core));
        }

        let mut staging_buffer = Self::new_temp(core, data.len())?;

        let mapped_memory = staging_buffer.map_mut()?;
        mapped_memory[0..data.len()].copy_from_slice(data);

        Ok(staging_buffer)
    }

    pub fn new_cloned_to_gpu_only_buffer(&self, usage: vk::BufferUsageFlags, name: &'static str) -> SrResult<GpuOnlyBuffer> {
        let mut dst = GpuOnlyBuffer::new::<T>(self.raw.core.clone(), self.len(), usage, name)?;
        self.clone_into_gpu_only_buffer(&mut dst)?;
        Ok(dst)
    }

    pub fn clone_into_gpu_only_buffer(&self, dst: &mut GpuOnlyBuffer) -> SrResult<()> {
        if self.is_null() {
            return Ok(());
        }
        if dst.is_null() {
            return Err(SrError::new_custom(
                "attempted to clone from a non-null buffer to a null buffer".to_string(),
            ));
        }

        let device = self.raw.core.device().inner();
        let cmd_buf =
            vulkan_abstraction::cmd_buffer::new_command_buffer(self.raw.core.cmd_pool(), self.raw.core.device().inner())?;

        let begin_info = vk::CommandBufferBeginInfo::default().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);

        unsafe { device.begin_command_buffer(cmd_buf, &begin_info) }?;

        debug_assert!(self.byte_size() <= dst.byte_size());

        //copy src.byte_size() bytes, from position 0 in src buffer to position 0 in dst buffer
        let regions = [vk::BufferCopy::default().size(self.byte_size()).src_offset(0).dst_offset(0)];

        unsafe { device.cmd_copy_buffer(cmd_buf, self.inner(), dst.inner(), &regions) };

        unsafe { device.end_command_buffer(cmd_buf) }?;

        self.raw.core.queue().submit_sync(cmd_buf)?;

        unsafe { device.free_command_buffers(self.raw.core.cmd_pool().inner(), &[cmd_buf]) };

        Ok(())
    }
}

impl<T> HostAccessibleBuffer<T> for StagingBuffer<T> {
    fn map_mut(&mut self) -> SrResult<&mut [T]> {
        self.raw.map_mut::<T>()
    }

    fn map(& self) -> SrResult<& [T]> {
        self.raw.map::<T>()
    }

    fn len(&self) -> usize {
        (self.raw.byte_size as usize) / std::mem::size_of::<T>()
    }
}

// --- 2. Uniform Buffer (CpuToGpu, Mappable) ---

pub struct UniformBuffer<T> {
    raw: RawBuffer,
    _marker: PhantomData<T>,
}
impl_buffer_trait!(UniformBuffer<T>);

impl<T> UniformBuffer<T> {
    pub fn new(core: Rc<vulkan_abstraction::Core>, len: usize) -> SrResult<Self> {
        let byte_size = (len * std::mem::size_of::<T>()) as vk::DeviceSize;
        let raw = RawBuffer::new_aligned(
            core,
            byte_size,
            1,
            gpu_allocator::MemoryLocation::CpuToGpu,
            vk::BufferUsageFlags::UNIFORM_BUFFER,
            "uniform buffer",
        )?;
        Ok(Self {
            raw,
            _marker: PhantomData,
        })
    }
}

impl<T> HostAccessibleBuffer<T> for UniformBuffer<T> {
    fn map_mut(&mut self) -> SrResult<&mut [T]> {
        self.raw.map_mut::<T>()
    }

    fn map(& self) -> SrResult<& [T]> {
        self.raw.map::<T>()
    }

    fn len(&self) -> usize {
        (self.raw.byte_size as usize) / std::mem::size_of::<T>()
    }
}

// --- 3. Gpu Only Buffer (GpuOnly, Non-Mappable) ---

pub struct GpuOnlyBuffer {
    raw: RawBuffer,
}
impl_buffer_trait!(GpuOnlyBuffer);

impl GpuOnlyBuffer {
    pub fn new<T>(
        core: Rc<vulkan_abstraction::Core>,
        len: usize,
        usage: vk::BufferUsageFlags,
        name: &'static str,
    ) -> SrResult<Self> {
        let byte_size = (len * std::mem::size_of::<T>()) as vk::DeviceSize;
        let raw = RawBuffer::new_aligned(
            core,
            byte_size,
            1,
            gpu_allocator::MemoryLocation::GpuOnly,
            usage | vk::BufferUsageFlags::TRANSFER_DST | vk::BufferUsageFlags::TRANSFER_SRC,
            name,
        )?;
        Ok(Self {
            raw,
        })
    }

    pub fn new_aligned<T>(
        core: Rc<vulkan_abstraction::Core>,
        len: usize,
        alignment: u64,
        usage: vk::BufferUsageFlags,
        name: &'static str,
    ) -> SrResult<Self> {
        let byte_size = (len * std::mem::size_of::<T>()) as vk::DeviceSize;
        let raw = RawBuffer::new_aligned(
            core,
            byte_size,
            alignment,
            gpu_allocator::MemoryLocation::GpuOnly,
            usage | vk::BufferUsageFlags::TRANSFER_DST | vk::BufferUsageFlags::TRANSFER_SRC,
            name,
        )?;
        Ok(Self {
            raw,
        })
    }
    

    pub fn len<T>(&self) -> usize {
        (self.raw.byte_size as usize) / std::mem::size_of::<T>()
    }

    pub fn new_from_data<T>(
        core: Rc<vulkan_abstraction::Core>,
        data: &[T],
        buffer_usage_flags: vk::BufferUsageFlags,
        name: &'static str,
    ) -> SrResult<Self>
    where
        T: Copy,
    {
        if data.len() == 0 {
            return Ok(Self::new_null(core));
        }

        let staging_buffer = StagingBuffer::new_from_data(Rc::clone(&core), data)?;
        let gpu_buffer =
            staging_buffer.new_cloned_to_gpu_only_buffer(buffer_usage_flags | vk::BufferUsageFlags::TRANSFER_DST, name)?;

        Ok(gpu_buffer)
    }
}

// --- 4. Arena Indexed Buffer ---

/// High-level manager for GPU data that changes at runtime.
/// Keeps a CPU-side staging buffer to allow granular updates and a GPU-side buffer for shaders.
pub struct ArenaIndexedBuffer<T> {
    pub staging: StagingBuffer<T>,
    pub gpu_only: GpuOnlyBuffer,
    capacity: usize,
    free_slots: Vec<usize>, // A simple LIFO stack of available indices
}

impl<T: Copy> ArenaIndexedBuffer<T> { //TODO handle reqeust to grow
    pub fn new(
        core: Rc<vulkan_abstraction::Core>,
        capacity: usize,
        usage: vk::BufferUsageFlags,
        name: &'static str,
    ) -> SrResult<Self> {
        let staging = StagingBuffer::new(core.clone(), capacity, usage, name)?; //TODO flags
        let gpu_only = GpuOnlyBuffer::new::<T>(core.clone(), capacity, usage, name)?;

        // Populate the free list with all available indices
        let free_slots = (0..capacity).rev().collect();

        Ok(Self {
            staging,
            gpu_only,
            capacity,
            free_slots,
        })
    }




    /// Allocates a slot for new data. Returns the assigned index and the BufferCopy region
    /// that needs to be submitted to a CommandBuffer for GPU synchronization.
    pub fn allocate_and_update(&mut self, data: T) -> SrResult<(usize, vk::BufferCopy)> {
        let index = self
            .free_slots
            .pop()
            .ok_or_else(|| SrError::new_custom("Arena out of capacity!".to_string()))?;

        // Map staging buffer and write data to the specific slot
        let mapped = self.staging.map_mut()?;
        mapped[index] = data;

        // Calculate byte-offset and size for the GPU copy command
        let offset = (index * std::mem::size_of::<T>()) as vk::DeviceSize;
        let size = std::mem::size_of::<T>() as vk::DeviceSize;

        let region = vk::BufferCopy::default().src_offset(offset).dst_offset(offset).size(size);

        Ok((index, region))
    }

    /// Frees an index so it can be reused by future allocations.
    pub fn free_index(&mut self, index: usize) {
        if index < self.capacity {
            self.free_slots.push(index);
        }
    }
}
