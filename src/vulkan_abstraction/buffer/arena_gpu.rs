use std::rc::Rc;

use ash::vk;
use ash::vk::{BufferUsageFlags, DeviceSize};

use crate::error::*;
use crate::vulkan_abstraction;
use crate::vulkan_abstraction::RawBuffer;

use super::{ArenaBuffer, ArenaRingCore, Buffer};

/// Arena buffer with anonymous sequential indices.
/// Keeps a ring-buffered staging buffer for per-frame writes and a GPU-only
/// buffer for shader access. Slots are allocated from a free-list and freed
/// with deferred deallocation.
pub struct ArenaGpuBuffer<T: Copy> {
    inner: ArenaRingCore<T>,
}

impl<T: Copy> Buffer for ArenaGpuBuffer<T> {
    fn inner(&self) -> vk::Buffer {
        self.inner.inner_gpu()
    }

    fn usage(&self) -> BufferUsageFlags {
        self.inner.gpu_only().usage()
    }

    fn raw(&self) -> &RawBuffer {
        self.inner.gpu_only().raw()
    }

    fn raw_mut(&mut self) -> &mut RawBuffer {
        self.inner.gpu_only_mut().raw_mut()
    }

    fn byte_size(&self) -> DeviceSize {
        self.inner.gpu_only().byte_size()
    }

    fn is_null(&self) -> bool {
        self.inner.gpu_only().is_null()
    }

    fn get_device_address(&self) -> vk::DeviceAddress {
        self.inner.gpu_only().get_device_address()
    }

    fn new_null(core: Rc<vulkan_abstraction::Core>) -> Self {
        Self {
            inner: ArenaRingCore::new(core, 0, BufferUsageFlags::empty(), "null")
                .expect("null arena should not fail"),
        }
    }
}

impl<T: Copy> ArenaBuffer for ArenaGpuBuffer<T> {
    fn capacity(&self) -> usize {
        self.inner.capacity()
    }

   

    fn process_pending_frees(&mut self, current_frame: u64) {
        self.inner.process_pending_frees(current_frame);
    }
}

impl<T: Copy> ArenaGpuBuffer<T> {
    pub fn new(
        core: Rc<vulkan_abstraction::Core>,
        capacity: usize,
        usage: vk::BufferUsageFlags,
        name: &'static str,
    ) -> SrResult<Self> {
        Ok(Self {
            inner: ArenaRingCore::new(core, capacity, usage, name)?,
        })
    }

    pub(crate) fn new_from_data(
        core: Rc<vulkan_abstraction::Core>,
        data: &[T],
        usage: vk::BufferUsageFlags,
        name: &'static str,
    ) -> SrResult<Self> {
        let len = data.len();
        Ok(Self {
            inner: ArenaRingCore::new_from_data(core, data, usage, name)?,
        })
    }

    /// Allocates a slot for new data. Returns the assigned index and the
    /// `BufferCopy` region that needs to be submitted on a command buffer.
    pub fn allocate_and_update(&mut self, data: &T) -> SrResult<(usize, vk::BufferCopy)> {
        let result = self.inner.allocate_and_write(data)?;
        Ok(result)
    }

    /// Frees an index so it can be reused by future allocations.
    pub fn free_index(&mut self, index: usize) {
        self.inner.free_slot(index);
    }

    pub fn inner_staging(&self) -> vk::Buffer {
        self.inner.inner_staging()
    }
}
