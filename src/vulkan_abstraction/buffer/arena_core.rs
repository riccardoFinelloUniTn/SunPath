use std::collections::VecDeque;

use std::rc::Rc;

use ash::vk;

use crate::MAX_FRAMES_IN_FLIGHT;
use crate::error::*;
use crate::vulkan_abstraction;

use super::{Buffer, GpuOnlyBuffer, HostAccessibleBuffer, StagingBuffer};

pub trait ArenaBuffer: Buffer {
    ///the capacity of the gpu arena buffer
    fn capacity(&self) -> usize;
    fn process_pending_frees(&mut self, current_frame: u64);
}

/// Shared core for arena buffers that use a ring-buffered staging buffer
/// and a GPU-only destination buffer.
///
/// The staging buffer has `capacity * MAX_FRAMES_IN_FLIGHT` elements so that
/// each in-flight frame gets its own staging region. Writes land in the
/// current frame's region and a `BufferCopy` is returned for the caller to
/// submit on a command buffer.
pub(crate) struct ArenaRingCore<T: Copy> {
    staging: StagingBuffer<T>,
    gpu_only: GpuOnlyBuffer,
    capacity: usize,
    free_slots: Vec<usize>,
    pending_free_slots: VecDeque<(u64, usize)>,
    core: Rc<vulkan_abstraction::Core>,
}

impl<T: Copy> ArenaRingCore<T> {
    pub fn new(
        core: Rc<vulkan_abstraction::Core>,
        capacity: usize,
        usage: vk::BufferUsageFlags,
        name: &'static str,
    ) -> SrResult<Self> {
        let staging = StagingBuffer::new(
            core.clone(),
            (capacity * MAX_FRAMES_IN_FLIGHT) as vk::DeviceSize,
            usage | vk::BufferUsageFlags::TRANSFER_SRC,
            name,
        )?;
        let gpu_only = GpuOnlyBuffer::new::<T>(
            core.clone(),
            capacity as vk::DeviceSize,
            usage | vk::BufferUsageFlags::TRANSFER_SRC | vk::BufferUsageFlags::TRANSFER_DST,
            name,
        )?;
        let free_slots = (0..capacity).rev().collect();

        Ok(Self {
            staging,
            gpu_only,
            capacity,
            free_slots,
            pending_free_slots: VecDeque::new(),
            core,
        })
    }

    pub fn new_from_data(
        core: Rc<vulkan_abstraction::Core>,
        data: &[T],
        usage: vk::BufferUsageFlags,
        name: &'static str,
    ) -> SrResult<Self> {
        let capacity = data.len();

        if capacity == 0 {
            return Self::new_null(core);
        }

        let staging = StagingBuffer::new_from_data_with_custom_length(
            core.clone(),
            data,
            (capacity * MAX_FRAMES_IN_FLIGHT) as vk::DeviceSize,
            usage | vk::BufferUsageFlags::TRANSFER_SRC,
            name,
        )?;

        let mut gpu_only = GpuOnlyBuffer::new::<T>(
            core.clone(),
            capacity as vk::DeviceSize,
            usage | vk::BufferUsageFlags::TRANSFER_SRC | vk::BufferUsageFlags::TRANSFER_DST,
            name,
        )?;

        staging.clone_section_into_gpu_only_buffer(
            0,
            (capacity * std::mem::size_of::<T>()) as vk::DeviceSize,
            &mut gpu_only,
        )?;

        Ok(Self {
            staging,
            gpu_only,
            capacity,
            free_slots: vec![],
            pending_free_slots: VecDeque::new(),
            core,
        })
    }

    fn new_null(core: Rc<vulkan_abstraction::Core>) -> SrResult<Self> {
        Ok(Self {
            staging: StagingBuffer::new_null(core.clone()),
            gpu_only: GpuOnlyBuffer::new_null(core.clone()),
            capacity: 0,
            free_slots: vec![],
            pending_free_slots: VecDeque::new(),
            core,
        })
    }

    /// Write data to a specific slot in the ring-buffered staging area.
    /// Returns the slot and a `BufferCopy` region to submit on a command buffer.
    pub fn write_to_slot(&mut self, slot: usize, data: &T) -> SrResult<(usize, vk::BufferCopy)> {
        let frame_module = *self.core.absolute_frame_count.borrow() % MAX_FRAMES_IN_FLIGHT;
        let staging_index = slot + (self.capacity * frame_module);

        let mapped = self.staging.map_mut()?;
        mapped[staging_index] = *data;

        let size = std::mem::size_of::<T>() as vk::DeviceSize;
        let dst_offset = (slot as vk::DeviceSize) * size;
        let src_offset = (staging_index as vk::DeviceSize) * size;

        Ok((
            slot,
            vk::BufferCopy::default()
                .src_offset(src_offset)
                .dst_offset(dst_offset)
                .size(size),
        ))
    }

    /// Pop a free slot from the stack.
    pub fn allocate_slot(&mut self) -> SrResult<usize> {
        self.free_slots
            .pop()
            .ok_or_else(|| SrError::new_custom("Arena out of capacity!".to_string()))
    }

    /// Allocate a slot, write data, and return (slot, BufferCopy).
    pub fn allocate_and_write(&mut self, data: &T) -> SrResult<(usize, vk::BufferCopy)> {
        let slot = self.allocate_slot()?;
        self.write_to_slot(slot, data)
    }

    /// Mark a slot for deferred freeing.
    pub fn free_slot(&mut self, slot: usize) {
        let current_frame = *self.core.absolute_frame_count.borrow() as u64;
        self.pending_free_slots.push_back((current_frame, slot));
    }

    pub fn process_pending_frees(&mut self, current_frame: u64) {
        while let Some(&(frame_freed, slot)) = self.pending_free_slots.front() {
            if current_frame >= frame_freed + MAX_FRAMES_IN_FLIGHT as u64 {
                self.free_slots.push(slot);
                self.pending_free_slots.pop_front();
            } else {
                break;
            }
        }
    }

    pub fn inner_gpu(&self) -> vk::Buffer {
        self.gpu_only.inner()
    }

    pub fn inner_staging(&self) -> vk::Buffer {
        self.staging.inner()
    }

    pub fn gpu_only(&self) -> &GpuOnlyBuffer {
        &self.gpu_only
    }

    pub fn gpu_only_mut(&mut self) -> &mut GpuOnlyBuffer {
        &mut self.gpu_only
    }

    pub fn capacity(&self) -> usize {
        self.capacity
    }
}
