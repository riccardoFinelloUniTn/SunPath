use std::collections::HashMap;
use std::rc::Rc;

use ash::vk;
use nalgebra as na;

use crate::vulkan_abstraction::{Buffer, HostAccessibleBuffer};
use crate::{CameraMatrices, error::SrResult, vulkan_abstraction};

// ─── GPU-side structs ────────────────────────────────────────────────────────

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

#[derive(Clone, Copy)]
#[repr(C, packed)]
struct MeshesInfoBufferContents { //TODO I want to extract this from the entity so that there is a strong correlation
    vertex_buffer: vk::DeviceAddress,
    index_buffer: vk::DeviceAddress,
    material: Material,
}


// ─── Resource Manager ────────────────────────────────────────────────────────

const ARENA_CAPACITY: usize = 4096;

pub(crate) struct ShaderDataBuffers {
    //TODO this struct lacks a lot of update and remove methods also indexing into an hashset is kind of a problem especially for gltf
    //TODO this lacks the *tlas and there needs to be a wider entity approach in the project

    // Camera
    matrices_uniform_buffer: vulkan_abstraction::UniformBuffer<MatricesBufferContents>,

    // Per-entity mesh info (vertex/index addresses + material), indexed by arena slot
    meshes_info_storage_buffer: vulkan_abstraction::ArenaIndexedWithRingStagingBuffer<MeshesInfoBufferContents>,

    // Entity management
    entities: HashMap<u64, vulkan_abstraction::Entity>,
    next_entity_id: u64,

    // Entity transforms for shader access (indexed by arena slot, CPU-mapped storage buffer)
    entity_transforms: vulkan_abstraction::StagingBuffer<vk::TransformMatrixKHR>,

    // Emissive lighting — local-space triangles stored per-BLAS
    //TODO wtf are this two separate and not an arena buffer
    blas_emissive_triangles_cpu: Vec<vulkan_abstraction::gltf::EmissiveTriangle>,
    blas_emissive_triangles_gpu: vulkan_abstraction::GpuOnlyBuffer,
    // Dense indirection buffer for NEE sampling: (blas_tri_index, entity_arena_slot) pairs
    //TODO should this be an arena as well
    emissive_indirection_gpu: vulkan_abstraction::GpuOnlyBuffer,

    // Textures
    textures: Vec<(vk::Sampler, vk::ImageView)>,

    // Owned images with unique IDs (for runtime destruction)
    owned_images: HashMap<u64, vulkan_abstraction::Image>,
    next_image_id: u64,

    core: Rc<vulkan_abstraction::Core>,
}

impl ShaderDataBuffers {
    pub const NUMBER_OF_SAMPLERS: usize = 1024;

    pub fn new_empty(core: Rc<vulkan_abstraction::Core>) -> SrResult<Self> {
        let matrices_uniform_buffer = vulkan_abstraction::UniformBuffer::new(Rc::clone(&core), 1 as vk::DeviceSize)?;
        let meshes_info_storage_buffer = vulkan_abstraction::ArenaIndexedWithRingStagingBuffer::new(
            core.clone(),
            ARENA_CAPACITY,
            vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_SRC,
            "Meshes info storage buffer",
        )?;
        let entity_transforms = vulkan_abstraction::StagingBuffer::new(
            core.clone(),
            ARENA_CAPACITY as vk::DeviceSize,
            vk::BufferUsageFlags::STORAGE_BUFFER,
            "Entity transforms buffer",
        )?;

        Ok(Self {
            matrices_uniform_buffer,
            meshes_info_storage_buffer,

            entities: HashMap::new(),
            next_entity_id: 0,

            entity_transforms,

            blas_emissive_triangles_cpu: Vec::new(),
            blas_emissive_triangles_gpu: vulkan_abstraction::Buffer::new_null(Rc::clone(&core)),
            emissive_indirection_gpu: vulkan_abstraction::Buffer::new_null(Rc::clone(&core)),

            textures: Vec::new(),

            owned_images: HashMap::new(),
            next_image_id: 0,

            core,
        })
    }

    // ─── Camera ──────────────────────────────────────────────────────────────

    pub fn set_matrices(
        &mut self,
        CameraMatrices {
            view_inverse,
            proj_inverse,
            view_proj,
            prev_view_proj,
        }: CameraMatrices,
    ) -> SrResult<()> {
        let mem = self.matrices_uniform_buffer.map_mut()?;
        mem[0] = MatricesBufferContents {
            view_inverse,
            proj_inverse,
            view_proj,
            prev_view_proj,
        };
        Ok(())
    }

    // ─── Entity management ───────────────────────────────────────────────────

    /// Create an entity with the given BLAS, material, and transform.
    /// Returns the EntityId. The caller is responsible for TLAS updates.
    pub fn create_entity(
        &mut self,
        blas: &vulkan_abstraction::BLAS,
        blas_index: usize,
        material: &vulkan_abstraction::gltf::Material,
        transform: vk::TransformMatrixKHR,
    ) -> SrResult<vulkan_abstraction::EntityId> {
        let id = self.next_entity_id;
        self.next_entity_id += 1;

        let mesh_info = MeshesInfoBufferContents {
            vertex_buffer: blas.vertex_buffer().get_device_address(),
            index_buffer: blas.index_buffer().get_device_address(),
            material: Material::from(material),
        };

        let (arena_slot, copy_region) = self.meshes_info_storage_buffer.allocate_and_update(&mesh_info)?;

        // Write transform to the CPU-mapped transforms buffer
        let transforms = self.entity_transforms.map_mut()?;
        transforms[arena_slot] = transform;

        // Flush meshes_info to GPU (blocking)
        self.flush_single_copy(
            self.meshes_info_storage_buffer.inner_staging(),
            self.meshes_info_storage_buffer.inner(),
            &[copy_region],
        )?;

        let entity = vulkan_abstraction::Entity {
            id: vulkan_abstraction::EntityId(id),
            blas_index,
            arena_slot,
            transform,
        };
        self.entities.insert(id, entity);

        Ok(vulkan_abstraction::EntityId(id))
    }

    /// Destroy an entity. The arena slot is deferred-freed.
    pub fn destroy_entity(&mut self, id: vulkan_abstraction::EntityId) {
        if let Some(entity) = self.entities.remove(&id.0) {
            self.meshes_info_storage_buffer.free_index(entity.arena_slot);
        }
    }

    /// Update an entity's transform. Does NOT rebuild the TLAS — caller must do that.
    pub fn set_entity_transform(&mut self, id: vulkan_abstraction::EntityId, transform: vk::TransformMatrixKHR) -> SrResult<()> {
        if let Some(entity) = self.entities.get_mut(&id.0) {
            entity.transform = transform;
            let transforms = self.entity_transforms.map_mut()?;
            transforms[entity.arena_slot] = transform;
        }
        Ok(())
    }

    pub fn get_entity(&self, id: vulkan_abstraction::EntityId) -> Option<&vulkan_abstraction::Entity> {
        self.entities.get(&id.0)
    }

    pub fn entities(&self) -> &HashMap<u64, vulkan_abstraction::Entity> {
        &self.entities
    }

    // ─── Emissive triangles (per-BLAS, local-space) ──────────────────────────

    /// Append local-space emissive triangles for a BLAS. Returns the start index
    /// into the global buffer so the BLAS can record its range.
    pub fn add_blas_emissive_triangles(&mut self, triangles: &[vulkan_abstraction::gltf::EmissiveTriangle]) -> u32 {
        let start = self.blas_emissive_triangles_cpu.len() as u32;
        self.blas_emissive_triangles_cpu.extend_from_slice(triangles);
        start
    }

    /// Upload the accumulated blas_emissive_triangles_cpu to GPU.
    pub fn upload_blas_emissive_triangles(&mut self) -> SrResult<()> {
        if self.blas_emissive_triangles_cpu.is_empty() {
            let dummy = [vulkan_abstraction::gltf::EmissiveTriangle {
                v0: [0.0; 4],
                v1: [0.0; 4],
                v2: [0.0; 4],
                emission: [0.0; 4],
            }];
            self.blas_emissive_triangles_gpu = vulkan_abstraction::GpuOnlyBuffer::new_from_data(
                Rc::clone(&self.core),
                &dummy,
                vk::BufferUsageFlags::STORAGE_BUFFER,
                "blas emissive triangles dummy",
            )?;
        } else {
            self.blas_emissive_triangles_gpu = vulkan_abstraction::GpuOnlyBuffer::new_from_data(
                Rc::clone(&self.core),
                &self.blas_emissive_triangles_cpu,
                vk::BufferUsageFlags::STORAGE_BUFFER,
                "blas emissive triangles",
            )?;
        }
        Ok(())
    }

    /// Rebuild the dense emissive indirection buffer from all live entities and their BLASes' ranges.
    pub fn rebuild_emissive_indirection(&mut self, blases: &[vulkan_abstraction::BLAS]) -> SrResult<()> {
        let mut entries = Vec::new();

        for entity in self.entities.values() {
            let blas = &blases[entity.blas_index];
            for range in &blas.emissive_triangle_ranges {
                for tri_idx in range.clone() {
                    entries.push(vulkan_abstraction::gltf::EmissiveIndirectionEntry {
                        blas_tri_index: tri_idx,
                        entity_id: entity.arena_slot as u32,
                    });
                }
            }
        }

        if entries.is_empty() {
            let dummy = [vulkan_abstraction::gltf::EmissiveIndirectionEntry {
                blas_tri_index: 0,
                entity_id: 0,
            }];
            self.emissive_indirection_gpu = vulkan_abstraction::GpuOnlyBuffer::new_from_data(
                Rc::clone(&self.core),
                &dummy,
                vk::BufferUsageFlags::STORAGE_BUFFER,
                "emissive indirection dummy",
            )?;
        } else {
            self.emissive_indirection_gpu = vulkan_abstraction::GpuOnlyBuffer::new_from_data(
                Rc::clone(&self.core),
                &entries,
                vk::BufferUsageFlags::STORAGE_BUFFER,
                "emissive indirection",
            )?;
        }

        Ok(())
    }

    // ─── Image storage ───────────────────────────────────────────────────────

    /// Take ownership of an image and return a unique ID for it.
    pub fn add_image(&mut self, image: vulkan_abstraction::Image) -> u64 {
        let id = self.next_image_id;
        self.next_image_id += 1;
        self.owned_images.insert(id, image);
        id
    }

    /// Remove and destroy an image by its ID. No-op if the ID doesn't exist.
    pub fn remove_image(&mut self, id: u64) {
        self.owned_images.remove(&id);
        // Image is dropped here, which triggers Vulkan cleanup via Drop impl
    }

    pub fn get_image(&self, id: u64) -> Option<&vulkan_abstraction::Image> {
        self.owned_images.get(&id)
    }

    // ─── Legacy API (used by current load_scene, will be refactored) ─────────

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
        self.add_meshes_info(blas_instances, materials)?;
        self.set_textures(images, samplers, textures, fallback, default_sampler);
        self.set_emissive_triangles(emissive_triangles)?;
        Ok(())
    }

    fn set_emissive_triangles(&mut self, emissive_triangles: &[vulkan_abstraction::gltf::EmissiveTriangle]) -> SrResult<()> {
        if emissive_triangles.is_empty() {
            let dummy = [vulkan_abstraction::gltf::EmissiveTriangle {
                v0: [0.0; 4],
                v1: [0.0; 4],
                v2: [0.0; 4],
                emission: [0.0; 4],
            }];
            self.blas_emissive_triangles_gpu = vulkan_abstraction::GpuOnlyBuffer::new_from_data(
                Rc::clone(&self.core),
                &dummy,
                vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_SRC,
                "emissive triangles dummy storage buffer",
            )?;
        } else {
            self.blas_emissive_triangles_gpu = vulkan_abstraction::GpuOnlyBuffer::new_from_data(
                Rc::clone(&self.core),
                emissive_triangles,
                vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_SRC,
                "emissive triangles storage buffer",
            )?;
        }
        Ok(())
    }

    pub fn add_meshes_info(
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
            let (_index, copy_buffer) = self.meshes_info_storage_buffer.allocate_and_update(&mesh_info)?;
            copy_buffers.push(copy_buffer);
        }
        if copy_buffers.is_empty() {
            return Ok(());
        }

        self.flush_single_copy(
            self.meshes_info_storage_buffer.inner_staging(),
            self.meshes_info_storage_buffer.inner(),
            &copy_buffers,
        )
    }

    pub fn set_textures(
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

    // ─── Descriptor set accessors ────────────────────────────────────────────

    pub fn get_matrices_uniform_buffer(&self) -> vk::Buffer {
        self.matrices_uniform_buffer.inner()
    }

    pub fn get_meshes_info_storage_buffer(&self) -> vk::Buffer {
        self.meshes_info_storage_buffer.inner()
    }

    pub fn get_emissive_triangles_storage_buffer(&self) -> vk::Buffer {
        self.blas_emissive_triangles_gpu.inner()
    }

    pub fn get_emissive_indirection_buffer(&self) -> vk::Buffer {
        self.emissive_indirection_gpu.inner()
    }

    pub fn get_entity_transforms_buffer(&self) -> vk::Buffer {
        self.entity_transforms.inner()
    }

    pub fn get_textures(&self) -> &[(vk::Sampler, vk::ImageView)] {
        &self.textures
    }

    // ─── Internal helpers ────────────────────────────────────────────────────

    fn flush_single_copy(&self, src: vk::Buffer, dst: vk::Buffer, copy_regions: &[vk::BufferCopy]) -> SrResult<()> {
        if copy_regions.is_empty() {
            return Ok(());
        }

        let device = self.core.device().inner();
        let transfer_queue = self.core.transfer_queue();
        let cmd_pool = self.core.transfer_cmd_pool();

        let cmd_buf = vulkan_abstraction::cmd_buffer::new_command_buffer(cmd_pool, device)?;
        let begin_info = vk::CommandBufferBeginInfo::default().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);

        unsafe {
            device.begin_command_buffer(cmd_buf, &begin_info)?;

            device.cmd_copy_buffer(cmd_buf, src, dst, copy_regions);

            let buffer_barrier = vk::BufferMemoryBarrier2::default()
                .src_stage_mask(vk::PipelineStageFlags2::TRANSFER)
                .src_access_mask(vk::AccessFlags2::TRANSFER_WRITE)
                .dst_stage_mask(vk::PipelineStageFlags2::RAY_TRACING_SHADER_KHR | vk::PipelineStageFlags2::COMPUTE_SHADER)
                .dst_access_mask(vk::AccessFlags2::SHADER_READ)
                .buffer(dst)
                .offset(0)
                .size(vk::WHOLE_SIZE);

            let dependency_info = vk::DependencyInfo::default().buffer_memory_barriers(std::slice::from_ref(&buffer_barrier));
            device.cmd_pipeline_barrier2(cmd_buf, &dependency_info);

            device.end_command_buffer(cmd_buf)?;
        }

        transfer_queue.submit_sync(cmd_buf)?;
        unsafe { device.free_command_buffers(cmd_pool.inner(), &[cmd_buf]) };

        Ok(())
    }
}
