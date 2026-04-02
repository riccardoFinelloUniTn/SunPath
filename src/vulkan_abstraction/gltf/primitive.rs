use crate::vulkan_abstraction;

pub type PrimitiveUniqueKey = (usize, usize);

pub struct PrimitiveData {
    pub vertex_buffer: vulkan_abstraction::VertexBuffer<vulkan_abstraction::gltf::Vertex>,
    pub index_buffer: vulkan_abstraction::IndexBuffer<u32>,
}

pub struct Primitive {
    pub unique_key: PrimitiveUniqueKey,
    pub material: vulkan_abstraction::gltf::Material,
    pub local_emissive_triangles: Vec<[nalgebra::Vector4<f32>; 3]>,
}
