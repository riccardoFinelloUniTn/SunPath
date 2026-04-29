use crate::vulkan_abstraction;

#[derive(Clone, Copy)]
#[repr(C, packed)]
pub struct Material {
    base_color_value: [f32; 4],
    base_color_texture_index: u32,

    metallic_factor: f32,
    roughness_factor: f32,
    metallic_roughness_texture_index: u32,

    normal_texture_index: u32,
    occlusion_texture_index: u32,

    _padding: [f32; 2],

    //rgb + strength
    emissive_factor: [f32; 4],
    emissive_texture_index: u32,

    pub alpha_mode: u32,
    pub alpha_cutoff: f32,

    pub transmission_factor: f32,
    pub ior: f32,

    pub _end_padding: [u32; 3],
}
impl Material {
    const NULL_TEXTURE_INDEX: u32 = u32::MAX;
}

impl From<&vulkan_abstraction::gltf::Material> for Material {
    fn from(material: &vulkan_abstraction::gltf::Material) -> Self {
        let to_texture_index = |i: Option<usize>| -> u32 {
            match i {
                Some(i) => i as u32,
                None => Self::NULL_TEXTURE_INDEX,
            }
        };

        Self {
            base_color_value: material.pbr_metallic_roughness_properties.base_color_factor,
            base_color_texture_index: to_texture_index(material.pbr_metallic_roughness_properties.base_color_texture_index),

            metallic_factor: material.pbr_metallic_roughness_properties.metallic_factor,
            roughness_factor: material.pbr_metallic_roughness_properties.roughness_factor,
            metallic_roughness_texture_index: to_texture_index(
                material.pbr_metallic_roughness_properties.base_color_texture_index,
            ),

            normal_texture_index: to_texture_index(material.normal_texture_index),
            occlusion_texture_index: to_texture_index(material.occlusion_texture_index),

            emissive_factor: [
                material.emissive_factor[0],
                material.emissive_factor[1],
                material.emissive_factor[2],
                material.emissive_strength,
            ],
            emissive_texture_index: to_texture_index(material.emissive_texture_index),

            alpha_mode: 0,
            alpha_cutoff: 0.0,
            transmission_factor: material.transmission_factor,
            ior: material.ior,
            _end_padding: [0; 3],
            _padding: [0.0; 2],
        }
    }
}
