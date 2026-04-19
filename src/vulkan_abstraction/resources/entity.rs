use ash::vk;
use crate::vulkan_abstraction::Material;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct EntityId(pub u64);

#[derive(Copy , Clone)]
pub struct Entity {
    pub id: EntityId,
    /// Index into Renderer's blases vec (shared geometry).
    pub blas_index: usize,
    /// Physical slot in the meshes_info arena buffer (= gl_InstanceCustomIndexEXT).
    pub arena_slot: usize,
    /// Instance transform (3x4 row-major, as Vulkan expects for ray tracing).
    pub transform: vk::TransformMatrixKHR,
}


#[derive(Clone, Copy)]
#[repr(C, packed)]
pub(crate) struct EntityGpuData {
    pub(crate) vertex_buffer: vk::DeviceAddress,
    pub(crate) index_buffer: vk::DeviceAddress,
    pub(crate) material: Material,
}
