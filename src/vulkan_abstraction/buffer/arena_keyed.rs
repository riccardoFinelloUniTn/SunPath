use std::collections::HashMap;
use std::collections::VecDeque;
use std::rc::Rc;

use ash::vk;

use crate::MAX_FRAMES_IN_FLIGHT;
use crate::error::*;
use crate::vulkan_abstraction;
use crate::vulkan_abstraction::{Buffer, HostAccessibleBuffer};

use super::{GpuOnlyBuffer, StagingBuffer, UniformBuffer};

/// Core arena logic shared by all keyed arena variants.
/// Manages slot allocation with arbitrary u64 keys and deferred frees.
/// The data buffer always lives on GPU (GpuOnlyBuffer), writes go through a ring-buffered staging buffer.
struct ArenaKeyedCore<T: Copy> { //TODO bring together implementations
    staging: StagingBuffer<T>,
    gpu_only: GpuOnlyBuffer,
    capacity: usize,
    id_map: HashMap<u64, usize>,
    free_slots: Vec<usize>,
    pending_free_slots: VecDeque<(u64, usize)>, // (frame_freed, slot)
    core: Rc<vulkan_abstraction::Core>,
}

impl<T: Copy> ArenaKeyedCore<T> {
    fn new(
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
            id_map: HashMap::new(),
            free_slots,
            pending_free_slots: VecDeque::new(),
            core,
        })
    }

    /// Insert or update data for the given key. Returns the physical slot and a BufferCopy
    /// region that must be submitted to a command buffer to sync to GPU.
    fn insert(&mut self, id: u64, data: &T) -> SrResult<(usize, vk::BufferCopy)> {
        let slot = if let Some(&existing_slot) = self.id_map.get(&id) {
            existing_slot
        } else {
            let slot = self
                .free_slots
                .pop()
                .ok_or_else(|| SrError::new_custom("ArenaKeyed out of capacity!".to_string()))?;
            self.id_map.insert(id, slot);
            slot
        };

        self.write_to_slot(slot, data)
    }

    fn write_to_slot(&mut self, slot: usize, data: &T) -> SrResult<(usize, vk::BufferCopy)> {
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

    /// Mark a key for removal. The slot is deferred until MAX_FRAMES_IN_FLIGHT frames have passed.
    fn remove(&mut self, id: u64) {
        if let Some(slot) = self.id_map.remove(&id) {
            let current_frame = *self.core.absolute_frame_count.borrow() as u64;
            self.pending_free_slots.push_back((current_frame, slot));
        }
    }

    fn get_slot(&self, id: u64) -> Option<usize> {
        self.id_map.get(&id).copied()
    }

    fn contains(&self, id: u64) -> bool {
        self.id_map.contains_key(&id)
    }

    fn process_pending_frees(&mut self, current_frame: u64) {
        while let Some(&(frame_freed, slot)) = self.pending_free_slots.front() {
            if current_frame >= frame_freed + MAX_FRAMES_IN_FLIGHT as u64 {
                self.free_slots.push(slot);
                self.pending_free_slots.pop_front();
            } else {
                break;
            }
        }
    }

    fn inner_gpu(&self) -> vk::Buffer {
        self.gpu_only.inner()
    }

    fn inner_staging(&self) -> vk::Buffer {
        self.staging.inner()
    }

    fn len(&self) -> usize {
        self.id_map.len()
    }
}

// ─── Variant 1: CPU-only mapping ────────────────────────────────────────────

/// Arena buffer with arbitrary u64 keys. The id→slot mapping lives only on CPU.
/// The GPU sees a flat storage buffer indexed by physical slot.
pub struct ArenaKeyedCpuOnly<T: Copy> {
    inner: ArenaKeyedCore<T>,
}

impl<T: Copy> ArenaKeyedCpuOnly<T> {
    pub fn new(
        core: Rc<vulkan_abstraction::Core>,
        capacity: usize,
        usage: vk::BufferUsageFlags,
        name: &'static str,
    ) -> SrResult<Self> {
        Ok(Self {
            inner: ArenaKeyedCore::new(core, capacity, usage, name)?,
        })
    }

    pub fn insert(&mut self, id: u64, data: &T) -> SrResult<(usize, vk::BufferCopy)> {
        self.inner.insert(id, data)
    }

    pub fn remove(&mut self, id: u64) {
        self.inner.remove(id);
    }

    pub fn get_slot(&self, id: u64) -> Option<usize> {
        self.inner.get_slot(id)
    }

    pub fn contains(&self, id: u64) -> bool {
        self.inner.contains(id)
    }

    pub fn process_pending_frees(&mut self, current_frame: u64) {
        self.inner.process_pending_frees(current_frame);
    }

    pub fn inner_gpu(&self) -> vk::Buffer {
        self.inner.inner_gpu()
    }

    pub fn inner_staging(&self) -> vk::Buffer {
        self.inner.inner_staging()
    }

    pub fn capacity(&self) -> usize {
        self.inner.capacity
    }

    pub fn len(&self) -> usize {
        self.inner.len()
    }
}

// ─── Variant 2: GPU-only mapping ────────────────────────────────────────────

/// Arena buffer with arbitrary u64 keys. The id→slot mapping is also available
/// on GPU as a storage buffer (`mapping_gpu[slot] = external_id_u32`).
pub struct ArenaKeyedGpuMapped<T: Copy> {
    inner: ArenaKeyedCore<T>,
    mapping_gpu: GpuOnlyBuffer,
    mapping_staging: StagingBuffer<u32>,
}

impl<T: Copy> ArenaKeyedGpuMapped<T> {
    const EMPTY_SLOT: u32 = u32::MAX;

    pub fn new(
        core: Rc<vulkan_abstraction::Core>,
        capacity: usize,
        usage: vk::BufferUsageFlags,
        name: &'static str,
    ) -> SrResult<Self> {
        let mapping_staging = StagingBuffer::new_from_data(
            core.clone(),
            &vec![Self::EMPTY_SLOT; capacity],
            vk::BufferUsageFlags::TRANSFER_SRC | vk::BufferUsageFlags::STORAGE_BUFFER,
            "arena keyed mapping staging",
        )?;
        let mapping_gpu =
            mapping_staging.new_cloned_to_gpu_only_buffer(vk::BufferUsageFlags::STORAGE_BUFFER, "arena keyed mapping gpu")?;

        Ok(Self {
            inner: ArenaKeyedCore::new(core, capacity, usage, name)?,
            mapping_gpu,
            mapping_staging,
        })
    }

    /// Insert or update. Returns (slot, data_copy, mapping_copy) — both BufferCopy
    /// regions must be submitted to sync both the data and the mapping to GPU.
    pub fn insert(&mut self, id: u64, data: &T) -> SrResult<(usize, vk::BufferCopy, vk::BufferCopy)> {
        let (slot, data_copy) = self.inner.insert(id, data)?;

        let mapping_copy = self.write_mapping_slot(slot, id as u32)?;

        Ok((slot, data_copy, mapping_copy))
    }

    pub fn remove(&mut self, id: u64) -> Option<vk::BufferCopy> {
        let slot = self.inner.get_slot(id);
        self.inner.remove(id);
        slot.and_then(|s| self.write_mapping_slot(s, Self::EMPTY_SLOT).ok())
    }

    fn write_mapping_slot(&mut self, slot: usize, value: u32) -> SrResult<vk::BufferCopy> {
        let mapped = self.mapping_staging.map_mut()?;
        mapped[slot] = value;

        let size = std::mem::size_of::<u32>() as vk::DeviceSize;
        Ok(vk::BufferCopy::default()
            .src_offset((slot as vk::DeviceSize) * size)
            .dst_offset((slot as vk::DeviceSize) * size)
            .size(size))
    }

    pub fn get_slot(&self, id: u64) -> Option<usize> {
        self.inner.get_slot(id)
    }

    pub fn contains(&self, id: u64) -> bool {
        self.inner.contains(id)
    }

    pub fn process_pending_frees(&mut self, current_frame: u64) {
        self.inner.process_pending_frees(current_frame);
    }

    pub fn inner_gpu(&self) -> vk::Buffer {
        self.inner.inner_gpu()
    }

    pub fn inner_staging(&self) -> vk::Buffer {
        self.inner.inner_staging()
    }

    pub fn mapping_buffer(&self) -> vk::Buffer {
        self.mapping_gpu.inner()
    }

    pub fn capacity(&self) -> usize {
        self.inner.capacity
    }

    pub fn len(&self) -> usize {
        self.inner.len()
    }
}

// ─── Variant 3: Uniform (CPU+GPU) mapping ───────────────────────────────────

/// Arena buffer with arbitrary u64 keys. The id→slot mapping is stored in a
/// UniformBuffer, accessible from both CPU and GPU.
//TODO document better of inner workings and correct use
pub struct ArenaKeyedUniformMapped<T: Copy> {
    inner: ArenaKeyedCore<T>,
    mapping_staging: StagingBuffer<u32>,
}

impl<T: Copy> ArenaKeyedUniformMapped<T> {
    const EMPTY_SLOT: u32 = u32::MAX;

    pub fn new(
        core: Rc<vulkan_abstraction::Core>,
        capacity: usize,
        usage: vk::BufferUsageFlags,
        name: &'static str,
    ) -> SrResult<Self> {
        let mapping_staging = StagingBuffer::new(core.clone(), capacity as vk::DeviceSize,usage, name)?;

        let mut arena = Self {
            inner: ArenaKeyedCore::new(core, capacity, usage, name)?,
            mapping_staging,
        };

        // Initialize all mapping slots to EMPTY
        let mapped = arena.mapping_staging.map_mut()?;
        mapped.iter_mut().for_each(|v| *v = Self::EMPTY_SLOT);

        Ok(arena)
    }

    /// Insert or update. Returns (slot, data_copy). The mapping is updated in-place
    /// on the uniform buffer (no copy command needed for the mapping).
    pub fn insert(&mut self, id: u64, data: &T) -> SrResult<(usize, vk::BufferCopy)> { //TODO this lacks synchronization?
        let (slot, data_copy) = self.inner.insert(id, data)?;

        let mapped = self.mapping_staging.map_mut()?;
        mapped[slot] = id as u32;

        Ok((slot, data_copy))
    }

    pub fn remove(&mut self, id: u64) {
        if let Some(slot) = self.inner.get_slot(id) {
            if let Ok(mapped) = self.mapping_staging.map_mut() {
                mapped[slot] = Self::EMPTY_SLOT;
            }
        }
        self.inner.remove(id);
    }

    pub fn get_slot(&self, id: u64) -> Option<usize> {
        self.inner.get_slot(id)
    }

    pub fn contains(&self, id: u64) -> bool {
        self.inner.contains(id)
    }

    pub fn process_pending_frees(&mut self, current_frame: u64) {
        self.inner.process_pending_frees(current_frame);
    }

    pub fn inner_gpu(&self) -> vk::Buffer {
        self.inner.inner_gpu()
    }

    pub fn inner_staging(&self) -> vk::Buffer {
        self.inner.inner_staging()
    }

    pub fn mapping_buffer(&self) -> vk::Buffer {
        self.mapping_staging.inner()
    }

    pub fn capacity(&self) -> usize {
        self.inner.capacity
    }

    pub fn len(&self) -> usize {
        self.inner.len()
    }
}
