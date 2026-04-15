use std::rc::Rc;

use ash::vk;
use nalgebra as na;

use crate::{CameraMatrices, error::SrResult, vulkan_abstraction};
use crate::vulkan_abstraction::Buffer;

#[derive(Clone, Copy)]
#[repr(C, packed)]
struct MatricesBufferContents {
    pub view_inverse: na::Matrix4<f32>,
    pub proj_inverse: na::Matrix4<f32>,
    pub view_proj: na::Matrix4<f32>,
    pub prev_view_proj: na::Matrix4<f32>,
}

#[derive(Clone, Copy)]
#[repr(C, packed)]
struct Material {
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

    pub alpha_mode: u32,   // 4 bytes
    pub alpha_cutoff: f32, // 4 bytes

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
                material.emissive_strength
            ],
            emissive_texture_index: to_texture_index(material.emissive_texture_index),

            alpha_mode: 0,
            alpha_cutoff: 0.0,
            transmission_factor: material.transmission_factor,
            ior: material.ior,
            _end_padding: [0;3],
            _padding: [0.0; 2],
        }
    }
}

#[derive(Clone, Copy)]
#[repr(C, packed)]
struct MeshesInfoBufferContents {
    vertex_buffer: vk::DeviceAddress,
    index_buffer: vk::DeviceAddress,

    material: Material,
}

pub(crate) struct ShaderDataBuffers {
    matrices_uniform_buffer: vulkan_abstraction::UniformBuffer<MatricesBufferContents>,
    meshes_info_storage_buffer: vulkan_abstraction::ArenaIndexedWithRingStagingBuffer<MeshesInfoBufferContents>,
    emissive_triangles_storage_buffer: vulkan_abstraction::GpuOnlyBuffer,
    textures: Vec<(vk::Sampler, vk::ImageView)>,

    core: Rc<vulkan_abstraction::Core>,
}

impl ShaderDataBuffers {
    pub const NUMBER_OF_SAMPLERS: usize = 1024;

    pub fn new_empty(core: Rc<vulkan_abstraction::Core>) -> SrResult<Self> {
        let matrices_uniform_buffer = vulkan_abstraction::UniformBuffer::new(Rc::clone(&core), 1 as vk::DeviceSize)?;

        Ok(Self {
            matrices_uniform_buffer,
            meshes_info_storage_buffer: vulkan_abstraction::Buffer::new_null(Rc::clone(&core)),
            emissive_triangles_storage_buffer: vulkan_abstraction::Buffer::new_null(Rc::clone(&core)),
            textures: Vec::new(),
            core,
        })
    }

    pub fn set_matrices(
        &mut self,
        CameraMatrices {
            view_inverse,
            proj_inverse,
            view_proj,
            prev_view_proj
        }: CameraMatrices,
    ) -> SrResult<()> {

        let mem = self.matrices_uniform_buffer.raw_mut().map_mut::<MatricesBufferContents>()?;
        mem[0] = MatricesBufferContents {
            view_inverse,
            proj_inverse,
            view_proj,
            prev_view_proj
        };

        Ok(())
    }
    //TODO fix reallocation on update 
    pub fn update(
        &mut self,
        blas_instances: &[vulkan_abstraction::BlasInstance],
        materials: &[vulkan_abstraction::gltf::Material],
        images: &[vulkan_abstraction::Image],
        samplers: &[vulkan_abstraction::Sampler],
        textures: &[vulkan_abstraction::gltf::Texture],
        fallback: vulkan_abstraction::Texture,
        default_sampler: &vulkan_abstraction::Sampler,
        emissive_triangles: &[vulkan_abstraction::gltf::EmissiveTriangle],
    ) -> SrResult<()> {
        self.create_meshes_info(blas_instances, materials)?;
        self.set_textures(images, samplers, textures, fallback, default_sampler);
        self.set_emissive_triangles(emissive_triangles)?;

        Ok(())
    }



    fn set_emissive_triangles(
        &mut self,
        emissive_triangles: &[vulkan_abstraction::gltf::EmissiveTriangle],
    ) -> SrResult<()> {
        if emissive_triangles.is_empty() {
            //1-element dummy buffer of zeroes if there are no lights
            let dummy = [vulkan_abstraction::gltf::EmissiveTriangle {
                v0: [0.0; 4],
                v1: [0.0; 4],
                v2: [0.0; 4],
                emission: [0.0; 4],
            }];
            self.emissive_triangles_storage_buffer = vulkan_abstraction::GpuOnlyBuffer::new_from_data(
                Rc::clone(&self.core),
                &dummy,
                vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_SRC,
                "emissive triangles dummy storage buffer",
            )?;
        } else {
            self.emissive_triangles_storage_buffer = vulkan_abstraction::GpuOnlyBuffer::new_from_data(
                Rc::clone(&self.core),
                emissive_triangles,
                vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_SRC,
                "emissive triangles storage buffer",
            )?;
        }
        Ok(())
    }

    pub fn add_meshes_info( //TODO this is the blocking impl
        &mut self,
        blas_instances: &[vulkan_abstraction::BlasInstance],
        materials: &[vulkan_abstraction::gltf::Material],
    ) -> SrResult<()> {
        let mut copy_buffers = Vec::with_capacity(materials.len());
        for (blas_instance, material) in std::iter::zip(blas_instances.iter(), materials.iter()) {
            let mesh_info = MeshesInfoBufferContents {
                vertex_buffer: blas_instance.blas.vertex_buffer().get_device_address(),
                index_buffer: blas_instance.blas.index_buffer().get_device_address(),
                material: Material::from(material),
            };
            let (index, copy_buffer) = self.meshes_info_storage_buffer.allocate_and_update(&mesh_info)?;//TODO use the one for &[T]
            copy_buffers.push(copy_buffer);
        }
        if copy_buffers.is_empty() {
            return Ok(());
        }

        let device = self.core.device().inner();
        let transfer_queue = self.core.transfer_queue();
        let cmd_pool = self.core.transfer_cmd_pool();

        let cmd_buf = vulkan_abstraction::cmd_buffer::new_command_buffer(cmd_pool, device)?;
        let begin_info = vk::CommandBufferBeginInfo::default()
            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);

        unsafe {
            device.begin_command_buffer(cmd_buf, &begin_info)?;

            // 3. Submit the Batch Copy
            // This single command executes all regions you collected in the loop
            device.cmd_copy_buffer(
                cmd_buf,
                self.meshes_info_storage_buffer.inner_staging(),
                self.meshes_info_storage_buffer.inner(),
                &copy_buffers,
            );
            // 4. Memory Barrier (Availability -> Visibility)
            // We must ensure the TRANSFER writes are visible before shaders try to read them.
            let buffer_barrier = vk::BufferMemoryBarrier2::default()
                .src_stage_mask(vk::PipelineStageFlags2::TRANSFER)
                .src_access_mask(vk::AccessFlags2::TRANSFER_WRITE)
                // TODO Adjust dst_stage_mask based on where you read this!
                // e.g., RAY_TRACING_SHADER_KHR or FRAGMENT_SHADER
                .dst_stage_mask(vk::PipelineStageFlags2::RAY_TRACING_SHADER_KHR | vk::PipelineStageFlags2::COMPUTE_SHADER)
                .dst_access_mask(vk::AccessFlags2::SHADER_READ)
                .buffer(self.meshes_info_storage_buffer.inner())
                .offset(0)
                .size(vk::WHOLE_SIZE); // You can be more granular here if you want

            let dependency_info = vk::DependencyInfo::default()
                .buffer_memory_barriers(std::slice::from_ref(&buffer_barrier));

            device.cmd_pipeline_barrier2(cmd_buf, &dependency_info);

            device.end_command_buffer(cmd_buf)?;
        }

        // 5. Submit and Wait (Blocking)
        // NOTE: submit_sync blocks the CPU until the GPU finishes.
        transfer_queue.submit_sync(cmd_buf)?;

        // 6. Cleanup
        unsafe { device.free_command_buffers(cmd_pool.inner(), &[cmd_buf]) };

        Ok(())
    }

    fn create_meshes_info(
        &mut self,
        blas_instances: &[vulkan_abstraction::BlasInstance],
        materials: &[vulkan_abstraction::gltf::Material],
    ) -> SrResult<()> {
        let meshes_info_storage_buffer_contents = std::iter::zip(blas_instances.iter(), materials.iter())
            .map(|(blas_instance, material)| MeshesInfoBufferContents {
                vertex_buffer: blas_instance.blas.vertex_buffer().get_device_address(),
                index_buffer: blas_instance.blas.index_buffer().get_device_address(),
                material: Material::from(material),
            })
            .collect::<Vec<_>>();
                
        self.meshes_info_storage_buffer = vulkan_abstraction::ArenaIndexedWithRingStagingBuffer::new_into_gpu_from_data(
            Rc::clone(&self.core),
            &meshes_info_storage_buffer_contents,
            vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_SRC,
            "meshes info storage buffer",
        )?;

        Ok(())
    }

    fn set_textures(
        &mut self,
        images: &[vulkan_abstraction::Image],
        samplers: &[vulkan_abstraction::Sampler],
        textures: &[vulkan_abstraction::gltf::Texture],
        fallback: vulkan_abstraction::Texture,
        default_sampler: &vulkan_abstraction::Sampler,
    ) {
        self.textures.clear();

        self.textures.reserve_exact(Self::NUMBER_OF_SAMPLERS);

        for tex in textures {
            let sampler = match tex.sampler {
                Some(i) => &samplers[i],
                None => default_sampler,
            };
            let image = &images[tex.source];
            self.textures.push((sampler.inner(), image.image_view()));
        }

        while self.textures.len() < Self::NUMBER_OF_SAMPLERS {
            self.textures.push((fallback.1.inner(), fallback.0.image_view()));
        }

        assert_eq!(self.textures.len(), Self::NUMBER_OF_SAMPLERS);
    }

    pub fn get_matrices_uniform_buffer(&self) -> vk::Buffer {
        self.matrices_uniform_buffer.inner()
    }

    pub fn get_meshes_info_storage_buffer(&self) -> vk::Buffer {
        self.meshes_info_storage_buffer.inner()
    }

    pub fn get_emissive_triangles_storage_buffer(&self) -> vk::Buffer {
        self.emissive_triangles_storage_buffer.inner()
    }

    pub fn get_textures(&self) -> &[(vk::Sampler, vk::ImageView)] {
        &self.textures
    }
}
