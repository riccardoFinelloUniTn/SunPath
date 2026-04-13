pub mod index_buffer;
pub mod vertex_buffer;

use std::collections::VecDeque;
//why use and not just mod?
pub use index_buffer::*;
use std::marker::PhantomData;
pub use vertex_buffer::*;

use crate::vulkan_abstraction::{Core};
use crate::{error::*, vulkan_abstraction, MAX_FRAMES_IN_FLIGHT};
use ash::vk;
use ash::vk::{BufferUsageFlags, DeviceAddress, DeviceSize, Handle};
use std::rc::Rc;
use log::error;
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

    fn map(&self) -> SrResult<&[T]>;

    fn len(&self) -> usize;
}

pub struct RawBuffer {
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

    pub fn map<V: Sized>(&self) -> SrResult<&[V]> {
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
    pub fn new_temp(core: Rc<vulkan_abstraction::Core>, len: usize) -> SrResult<Self> { //TODO this gets used for new from data and it has no flags
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

    pub fn new(
        core: Rc<vulkan_abstraction::Core>,
        len: usize,
        buffer_usage_flags: vk::BufferUsageFlags,
        name: &'static str,
    ) -> SrResult<Self> {
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


    pub fn new_from_data_with_custom_length(core: Rc<vulkan_abstraction::Core>, data: &[T] , len : usize) -> SrResult<Self>
    where
        T: Copy,
    {
        if  len < data.len()  {
            return Err(SrError::new_custom(
                format!("attempted to create an insufficiently sized buffer, src : {} bytes , dst : {len} bytes" , data.len() )
            ));
        }

        let mut staging_buffer = Self::new_temp(core, len)?;

        let mapped_memory = staging_buffer.map_mut()?;
        mapped_memory[0..data.len()].copy_from_slice(data);

        Ok(staging_buffer)
    }


    pub fn new_cloned_to_gpu_only_buffer(&self, usage: vk::BufferUsageFlags, name: &'static str) -> SrResult<GpuOnlyBuffer> {
        let mut dst = GpuOnlyBuffer::new::<T>(self.raw.core.clone(), self.len(), usage, name)?;
        self.clone_section_into_gpu_only_buffer(0 , self.byte_size() ,&mut dst )?;
        Ok(dst)
    }


    pub fn clone_section_into_gpu_only_buffer(&self , src_offset : DeviceSize ,  src_section_length : DeviceSize  , dst: &mut GpuOnlyBuffer) -> SrResult<()> {
        if self.is_null() {
            return Ok(());
        }
        if dst.is_null() {
            return Err(SrError::new_custom(
                "attempted to clone from a non-null buffer to a null buffer".to_string(),
            ));
        }

        if self.byte_size() < src_section_length + src_offset {
            return Err(SrError::new_custom(
                format!( "attempted to clone from outside the src buffer, src : {src_section_length} bytes , dst : {} bytes", dst.byte_size()),
            ));
        }
        if dst.byte_size() < src_section_length {
            return Err(SrError::new_custom(
                format!("attempted to clone into an insufficiently sized buffer, src : {src_section_length} bytes , dst : {} bytes" , dst.byte_size())
            ));
        }

        let device = self.raw.core.device().inner();
        let cmd_buf =
            vulkan_abstraction::cmd_buffer::new_command_buffer(self.raw.core.graphics_cmd_pool(), self.raw.core.device().inner())?;

        let begin_info = vk::CommandBufferBeginInfo::default().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);

        unsafe { device.begin_command_buffer(cmd_buf, &begin_info) }?;


        let regions = [vk::BufferCopy::default().size(src_section_length).src_offset(src_offset).dst_offset(0)];

        unsafe { device.cmd_copy_buffer(cmd_buf, self.inner(), dst.inner(), &regions) };

        unsafe { device.end_command_buffer(cmd_buf) }?;

        self.raw.core.graphics_queue().submit_sync(cmd_buf)?;

        unsafe { device.free_command_buffers(self.raw.core.graphics_cmd_pool().inner(), &[cmd_buf]) };

        Ok(())
    }

    pub fn clone_into_gpu_only_buffer(&self, dst: &mut GpuOnlyBuffer) -> SrResult<()> {
        self.clone_section_into_gpu_only_buffer(0 , self.byte_size() , dst )?;

        Ok(())
    }
}

impl<T> HostAccessibleBuffer<T> for StagingBuffer<T> {
    fn map_mut(&mut self) -> SrResult<&mut [T]> {
        self.raw.map_mut::<T>()
    }

    fn map(&self) -> SrResult<&[T]> {
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

    fn map(&self) -> SrResult<&[T]> {
        self.raw.map::<T>()
    }

    fn len(&self) -> usize {
        (self.raw.byte_size as usize) / std::mem::size_of::<T>()
    }
}

// --- 3. Gpu Only Buffer (GpuOnly, Non-Mappable) ---

pub struct GpuOnlyBuffer { //TODO flatten it into a trait,so index and vertex buffer are the impl 
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
        Ok(Self { raw })
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
        Ok(Self { raw })
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
pub struct ArenaIndexedWithRingStagingBuffer<T> {
    staging: StagingBuffer<T>,
    gpu_only: GpuOnlyBuffer,
    capacity: usize,
    free_slots: Vec<usize>, // A simple LIFO stack of available indices
    ///these are the freed slot of this frame,which will be available the next one
    pending_free_slots: VecDeque<(u64, usize)>,

}


impl<T> Buffer for ArenaIndexedWithRingStagingBuffer<T> {
    ///This returns the gpu one 
    fn inner(&self) -> vk::Buffer {
        self.gpu_only.inner()
    }
    ///This returns the gpu one
    fn raw(&self) -> &RawBuffer {
        self.gpu_only.raw()
    }
    ///This returns the gpu one
    fn raw_mut(&mut self) -> &mut RawBuffer {
        self.gpu_only.raw_mut()
    }
    ///This returns the gpu one
    fn byte_size(&self) -> DeviceSize {
       self.gpu_only.byte_size()
    }
    ///This returns the gpu one 
    fn is_null(&self) -> bool {
        self.gpu_only.raw.buffer == vk::Buffer::null()
    }

    fn get_device_address(&self) -> vk::DeviceAddress {
        if self.is_null() {
            return 0;
        }
        let info = vk::BufferDeviceAddressInfo::default().buffer(self.gpu_only.raw.buffer);
        unsafe { self.gpu_only.raw.core.device().inner().get_buffer_device_address(&info) }
    }

    fn new_null(core: Rc<vulkan_abstraction::Core>) -> Self {
        Self {
            staging: StagingBuffer::new_null(core.clone()),
            gpu_only: GpuOnlyBuffer::new_null(core),
            capacity: 0,
            free_slots: vec![],
            pending_free_slots: Default::default(),
        }
    }
}

impl<T: Copy> ArenaIndexedWithRingStagingBuffer<T> {
    pub(crate) fn new_into_gpu_from_data(core: Rc<Core>, data: &[T], buffer_usage_flags: BufferUsageFlags, name: &'static str) -> SrResult<Self>
    {
        let capacity = data.len();

        if capacity == 0 {
            return Ok(Self::new_null(core));
        }

        let staging_buffer = StagingBuffer::new_from_data_with_custom_length(Rc::clone(&core), data , data.len() * MAX_FRAMES_IN_FLIGHT )?;

        let mut gpu_buffer = GpuOnlyBuffer::new::<T>(core.clone(), data.len(), buffer_usage_flags  | vk::BufferUsageFlags::TRANSFER_DST, name )?;


            staging_buffer.clone_section_into_gpu_only_buffer(0, data.len() as DeviceSize, &mut gpu_buffer)?;
        Ok(Self{
            staging:staging_buffer,
            gpu_only: gpu_buffer,
            capacity,
            free_slots: vec![],
            pending_free_slots: Default::default(),
        })
    }



    //TODO handle reqeust to grow and compact near indexes in a ranges
    pub fn new(
        core: Rc<vulkan_abstraction::Core>,
        capacity: usize,
        usage: vk::BufferUsageFlags,
        name: &'static str,
    ) -> SrResult<Self> {
        let staging = StagingBuffer::new(core.clone(), capacity * MAX_FRAMES_IN_FLIGHT , usage  | vk::BufferUsageFlags::TRANSFER_SRC, name)?; //TODO flags
        let gpu_only = GpuOnlyBuffer::new::<T>(core.clone(), capacity, usage  | vk::BufferUsageFlags::TRANSFER_DST, name)?;

        // Populate the free list with all available indices
        let free_slots = (0..capacity).rev().collect();

        Ok(Self {
            staging,
            gpu_only,
            capacity,
            free_slots,
            pending_free_slots: Default::default(),
        })
    }

    /// Frees an index so it can be reused by future allocations.
    pub fn free_index(&mut self, index: usize) {
       let current_frame = *self.raw().core.absolute_frame_count.borrow() as u64;
        self.pending_free_slots.push_back((current_frame, index));
    }

    pub fn process_pending_frees(&mut self, current_frame: u64) {
        while let Some(&(frame_freed, index)) = self.pending_free_slots.front() {
            if current_frame >= frame_freed + MAX_FRAMES_IN_FLIGHT as u64 {
                self.free_slots.push(index);
                self.pending_free_slots.pop_front();
            } else {
                break;
            }
        }
    }


    pub fn inner_staging(&self) -> vk::Buffer {
        self.staging.inner()
    }
    pub fn raw_staging(&self) -> &RawBuffer {
        self.staging.raw()
    }

    fn raw_mut_staging(&mut self) -> &mut RawBuffer {
        self.staging.raw_mut()
    }
    fn byte_size_staging(&self) -> DeviceSize {
        self.staging.byte_size()
    }
    fn is_staging_null(&self) -> bool {
        self.staging.raw.buffer == vk::Buffer::null()
    }

    fn get_staging_address(&self) -> vk::DeviceAddress {
        if self.is_null() {
            return 0;
        }
        let info = vk::BufferDeviceAddressInfo::default().buffer(self.staging.raw.buffer);
        unsafe { self.staging.raw.core.device().inner().get_buffer_device_address(&info) }
    }
    



    /// Allocates a slot for new data. Returns the assigned index and the BufferCopy region
    /// that needs to be submitted to a CommandBuffer for GPU synchronization.
    pub fn allocate_and_update(&mut self, data: &T  ) -> SrResult<usize> { //TODO this does not actually allocate or delegate correctly yet and with a vector of data

        let index = self
            .free_slots
            .pop()
            .ok_or_else(|| SrError::new_custom("Arena out of capacity!".to_string()))?;

        let elements_per_frame = self.capacity;
        let frame_module = *self.raw().core.absolute_frame_count.borrow() % MAX_FRAMES_IN_FLIGHT;

        let staging_index = index + (elements_per_frame * frame_module);

        let mapped = self.staging.map_mut()?;
        mapped[staging_index] = *data;

        let size = std::mem::size_of::<T>() as vk::DeviceSize;
        let dst_offset = (index as vk::DeviceSize) * size;
        let src_offset = (staging_index as vk::DeviceSize) * size;

        let regions = [vk::BufferCopy::default().src_offset(src_offset).dst_offset(dst_offset).size(size)];

        //TODO temp one time allocation

        let device = self.raw().core.device().inner();
        let cmd_buf =
            vulkan_abstraction::cmd_buffer::new_command_buffer(self.raw().core.graphics_cmd_pool(), self.raw().core.device().inner())?;

        let begin_info = vk::CommandBufferBeginInfo::default().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);

        unsafe { device.begin_command_buffer(cmd_buf, &begin_info) }?;


        unsafe { device.cmd_copy_buffer(cmd_buf, self.inner_staging(), self.inner(), &regions) };

        unsafe { device.end_command_buffer(cmd_buf) }?;

        self.raw().core.transfer_queue().submit_sync(cmd_buf)?;

        unsafe { device.free_command_buffers(self.raw().core.graphics_cmd_pool().inner(), &[cmd_buf]) };


        Ok(index)
    }


}
