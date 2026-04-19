pub mod arena_core;
pub mod arena_host;
pub mod arena_gpu;
pub mod arena_keyed;
pub mod index_buffer;
pub mod vertex_buffer;

//why use and not just mod?
pub use arena_core::*;
pub use arena_host::*;
pub use arena_gpu::*;
pub use arena_keyed::*;
pub use index_buffer::*;
use std::marker::PhantomData;
pub use vertex_buffer::*;

use crate::vulkan_abstraction::Core;
use crate::{error::*, vulkan_abstraction};
use ash::vk;
use ash::vk::{BufferUsageFlags, BufferUsageFlags2KHR, DeviceAddress, DeviceSize, Handle};
use log::{error, info};
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

    fn usage(&self) -> BufferUsageFlags;

    fn raw(&self) -> &vulkan_abstraction::RawBuffer;

    fn raw_mut(&mut self) -> &mut vulkan_abstraction::RawBuffer;

    fn byte_size(&self) -> vk::DeviceSize;
    fn is_null(&self) -> bool;
    fn get_device_address(&self) -> vk::DeviceAddress;
    fn new_null(core: Rc<vulkan_abstraction::Core>) -> Self
    where
        Self: Sized;
}

/// Marker + accessor trait for buffers whose data lives in GPU-only memory
/// and are updated via a staging buffer.
pub trait GpuSideBuffer: Buffer {
    /// Returns the staging buffer used for CPU→GPU transfers.
    /// Types without a persistent staging buffer return `vk::Buffer::null()`.
    fn inner_staging(&self) -> vk::Buffer {
        vk::Buffer::null()
    }
}

/// Exclusive trait for host-visible buffers (CpuToGpu or GpuToCpu) that can be mapped.
pub trait HostAccessibleBuffer<T>: Buffer {
    fn map_mut(&mut self) -> SrResult<&mut [T]>;

    fn map(&self) -> SrResult<&[T]>;

    fn get(&self) -> SrResult<&T> {
        self.map().map(|s| &s[0])
    }

    fn get_mut(&mut self) -> SrResult<&mut T> {
        self.map_mut().map(|s| &mut s[0])
    }

    fn len(&self) -> usize;
}


pub struct RawBuffer {
    core: Rc<vulkan_abstraction::Core>,
    buffer: vk::Buffer,
    allocation: gpu_allocator::vulkan::Allocation,
    byte_size: u64,
    usage: BufferUsageFlags,
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

        let queue_family_indices = [
            core.graphics_queue().queue_family_index(),
            core.transfer_queue().queue_family_index(),
        ];

        let device = core.device().inner();

        let buffer = if queue_family_indices[0] != queue_family_indices[1] {
            //TODO add parameters to choose concurrency or not
            let buf_info = vk::BufferCreateInfo::default()
                .size(byte_size)
                .usage(buffer_usage_flags)
                .sharing_mode(vk::SharingMode::CONCURRENT)
                .queue_family_indices(&queue_family_indices);

            unsafe { device.create_buffer(&buf_info, None) }?
        } else {
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
            usage: buffer_usage_flags,
        })
    }

    pub fn new_null(core: Rc<vulkan_abstraction::Core>) -> Self {
        Self {
            core,
            buffer: vk::Buffer::null(),
            allocation: gpu_allocator::vulkan::Allocation::default(),
            byte_size: 0,
            usage: BufferUsageFlags::empty(),
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

            fn usage(&self) -> vk::BufferUsageFlags {
                self.raw().usage.clone()
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

            fn usage(&self) -> vk::BufferUsageFlags {
                self.raw().usage.clone()
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
    pub fn new_temp(core: Rc<vulkan_abstraction::Core>, len: vk::DeviceSize) -> SrResult<Self> {
        //TODO this gets used for new from data and it has no flags
        let byte_size = (len * std::mem::size_of::<T>() as vk::DeviceSize) as vk::DeviceSize;
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
        len: vk::DeviceSize,
        buffer_usage_flags: vk::BufferUsageFlags,
        name: &'static str,
    ) -> SrResult<Self> {
        let byte_size = len * std::mem::size_of::<T>() as vk::DeviceSize;
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

    pub fn new_temp_from_data(core: Rc<vulkan_abstraction::Core>, data: &[T]) -> SrResult<Self>
    where
        T: Copy,
    {
        if data.len() == 0 {
            return Ok(Self::new_null(core));
        }

        let mut staging_buffer = Self::new_temp(core, data.len() as vk::DeviceSize)?;

        let mapped_memory = staging_buffer.map_mut()?;
        mapped_memory[0..data.len()].copy_from_slice(data);

        Ok(staging_buffer)
    }

    pub fn new_from_data(
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

        let mut staging_buffer = Self::new(core, data.len() as vk::DeviceSize, buffer_usage_flags, name)?;

        let mapped_memory = staging_buffer.map_mut()?;
        mapped_memory[0..data.len()].copy_from_slice(data);

        Ok(staging_buffer)
    }

    pub fn new_from_data_with_custom_length(
        core: Rc<vulkan_abstraction::Core>,
        data: &[T],
        len: vk::DeviceSize,
        buffer_usage_flags: vk::BufferUsageFlags,
        name: &'static str,
    ) -> SrResult<Self>
    where
        T: Copy,
    {
        if len < (data.len() as vk::DeviceSize) {
            return Err(SrError::new_custom(format!(
                "attempted to create an insufficiently sized buffer, src : {} bytes , dst : {len} bytes",
                data.len()
            )));
        }

        let mut staging_buffer = Self::new(core, len, buffer_usage_flags, name)?;

        let mapped_memory = staging_buffer.map_mut()?;
        mapped_memory[0..data.len()].copy_from_slice(data);

        Ok(staging_buffer)
    }

    pub fn new_cloned_to_gpu_only_buffer(&self, usage: vk::BufferUsageFlags, name: &'static str) -> SrResult<GpuOnlyBuffer> {
        let mut dst = GpuOnlyBuffer::new::<T>(self.raw.core.clone(), self.len() as vk::DeviceSize, usage, name)?;
        self.clone_section_into_gpu_only_buffer(0, self.byte_size(), &mut dst)?;
        Ok(dst)
    }

    pub fn clone_section_into_gpu_only_buffer(
        &self,
        src_offset: DeviceSize,
        src_section_length: DeviceSize,
        dst: &mut GpuOnlyBuffer,
    ) -> SrResult<()> {
        if self.is_null() {
            return Ok(());
        }
        if dst.is_null() {
            return Err(SrError::new_custom(
                "attempted to clone from a non-null buffer to a null buffer".to_string(),
            ));
        }

        if self.byte_size() < src_section_length + src_offset {
            return Err(SrError::new_custom(format!(
                "attempted to clone from outside the src buffer, src : {src_section_length} bytes , dst : {} bytes",
                dst.byte_size()
            )));
        }
        if dst.byte_size() < src_section_length {
            return Err(SrError::new_custom(format!(
                "attempted to clone into an insufficiently sized buffer, src : {src_section_length} bytes , dst : {} bytes",
                dst.byte_size()
            )));
        }

        let device = self.raw.core.device().inner();
        let cmd_buf = vulkan_abstraction::cmd_buffer::new_command_buffer(
            self.raw.core.transfer_cmd_pool(),
            self.raw.core.device().inner(),
        )?;

        let begin_info = vk::CommandBufferBeginInfo::default().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);

        unsafe { device.begin_command_buffer(cmd_buf, &begin_info) }?;

        let regions = [vk::BufferCopy::default()
            .size(src_section_length)
            .src_offset(src_offset)
            .dst_offset(0)];

        unsafe { device.cmd_copy_buffer(cmd_buf, self.inner(), dst.inner(), &regions) };

        // 1. Dynamically infer what is going to read this destination buffer
        let dst_usage = dst.usage(); // Assuming you track this in GpuOnlyBuffer!
        let (dst_stage_mask, dst_access_mask) = infer_read_masks_from_usage(dst_usage);

        // 2. Point the barrier at the DESTINATION buffer, not the source!
        let buffer_barrier = vk::BufferMemoryBarrier2::default()
            .src_stage_mask(vk::PipelineStageFlags2::TRANSFER)
            .src_access_mask(vk::AccessFlags2::TRANSFER_WRITE)
            .dst_stage_mask(dst_stage_mask)
            .dst_access_mask(dst_access_mask)
            .buffer(dst.inner()) // ✅ Fixed!
            .offset(0) // ✅ Granular: only barrier what we copied
            .size(src_section_length); // ✅ Granular: only barrier what we copied

        let dependency_info = vk::DependencyInfo::default().buffer_memory_barriers(std::slice::from_ref(&buffer_barrier));

        unsafe { device.cmd_pipeline_barrier2(cmd_buf, &dependency_info) };

        unsafe { device.end_command_buffer(cmd_buf) }?;

        self.raw.core.transfer_queue().submit_sync(cmd_buf)?;

        unsafe { device.free_command_buffers(self.raw.core.transfer_cmd_pool().inner(), &[cmd_buf]) };

        Ok(())
    }

    pub fn clone_into_gpu_only_buffer(&self, dst: &mut GpuOnlyBuffer) -> SrResult<()> {
        self.clone_section_into_gpu_only_buffer(0, self.byte_size(), dst)?;

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
    pub fn new(core: Rc<vulkan_abstraction::Core>, len: vk::DeviceSize) -> SrResult<Self> {
        let byte_size = len * std::mem::size_of::<T>() as vk::DeviceSize;
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

pub struct GpuOnlyBuffer {
    //TODO flatten it into a trait,so index and vertex buffer are the impl
    raw: RawBuffer,
}
impl_buffer_trait!(GpuOnlyBuffer);

impl GpuSideBuffer for GpuOnlyBuffer {}

impl GpuOnlyBuffer {
    pub fn new<T>(
        core: Rc<vulkan_abstraction::Core>,
        len: vk::DeviceSize,
        usage: vk::BufferUsageFlags,
        name: &'static str,
    ) -> SrResult<Self> {
        let byte_size = len * std::mem::size_of::<T>() as vk::DeviceSize;
        let raw = RawBuffer::new_aligned(
            core,
            byte_size,
            1,
            gpu_allocator::MemoryLocation::GpuOnly,
            usage | vk::BufferUsageFlags::TRANSFER_DST,
            name,
        )?;
        log::debug!("New Gpu Buffer with these usage flags {usage:?}");
        Ok(Self { raw })
    }

    pub fn new_aligned<T>(
        core: Rc<vulkan_abstraction::Core>,
        len: vk::DeviceSize,
        alignment: u64,
        usage: vk::BufferUsageFlags,
        name: &'static str,
    ) -> SrResult<Self> {
        let byte_size = len * std::mem::size_of::<T>() as vk::DeviceSize;
        let raw = RawBuffer::new_aligned(
            core,
            byte_size,
            alignment,
            gpu_allocator::MemoryLocation::GpuOnly,
            usage | vk::BufferUsageFlags::TRANSFER_DST,
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

        let staging_buffer = StagingBuffer::new_temp_from_data(Rc::clone(&core), data)?;
        let gpu_buffer =
            staging_buffer.new_cloned_to_gpu_only_buffer(buffer_usage_flags | vk::BufferUsageFlags::TRANSFER_DST, name)?;

        Ok(gpu_buffer)
    }
}


/// Derives the appropriate pipeline stages and access flags for reading a buffer
/// based on the usage flags it was created with.
pub fn infer_read_masks_from_usage(usage: vk::BufferUsageFlags) -> (vk::PipelineStageFlags2, vk::AccessFlags2) {
    let mut stage_mask = vk::PipelineStageFlags2::empty();
    let mut access_mask = vk::AccessFlags2::empty();

    if usage.contains(vk::BufferUsageFlags::VERTEX_BUFFER) {
        stage_mask |= vk::PipelineStageFlags2::VERTEX_INPUT;
        access_mask |= vk::AccessFlags2::VERTEX_ATTRIBUTE_READ;
    }

    if usage.contains(vk::BufferUsageFlags::INDEX_BUFFER) {
        stage_mask |= vk::PipelineStageFlags2::VERTEX_INPUT;
        access_mask |= vk::AccessFlags2::INDEX_READ;
    }

    if usage.contains(vk::BufferUsageFlags::UNIFORM_BUFFER) {
        // UBOs can be read in almost any shader stage
        stage_mask |= vk::PipelineStageFlags2::ALL_GRAPHICS | vk::PipelineStageFlags2::COMPUTE_SHADER;
        access_mask |= vk::AccessFlags2::UNIFORM_READ;
    }

    if usage.contains(vk::BufferUsageFlags::STORAGE_BUFFER) {
        // SSBOs can be read in compute or graphics
        stage_mask |= vk::PipelineStageFlags2::ALL_GRAPHICS | vk::PipelineStageFlags2::COMPUTE_SHADER;
        access_mask |= vk::AccessFlags2::SHADER_READ;
    }

    if usage.contains(vk::BufferUsageFlags::INDIRECT_BUFFER) {
        stage_mask |= vk::PipelineStageFlags2::DRAW_INDIRECT;
        access_mask |= vk::AccessFlags2::INDIRECT_COMMAND_READ;
    }

    // Raytracing specific flags
    if usage.contains(vk::BufferUsageFlags::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR) {
        stage_mask |= vk::PipelineStageFlags2::ACCELERATION_STRUCTURE_BUILD_KHR;
        access_mask |= vk::AccessFlags2::ACCELERATION_STRUCTURE_READ_KHR;
    }
    if usage.contains(vk::BufferUsageFlags::ACCELERATION_STRUCTURE_STORAGE_KHR) {
        stage_mask |= vk::PipelineStageFlags2::RAY_TRACING_SHADER_KHR;
        access_mask |= vk::AccessFlags2::ACCELERATION_STRUCTURE_READ_KHR;
    }

    // Fallback if somehow it's none of the above
    if stage_mask.is_empty() {
        stage_mask = vk::PipelineStageFlags2::ALL_COMMANDS;
        access_mask = vk::AccessFlags2::MEMORY_READ;
    }

    (stage_mask, access_mask)
}
