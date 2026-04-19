use std::collections::HashMap;
use std::rc::Rc;

use ash::vk;
use nalgebra as na;

use crate::vulkan_abstraction::{BlasInstance, BlasMetaData, Buffer, HostAccessibleBuffer};
use crate::{CameraMatrices, MAX_TLAS_INSTANCES, error::SrResult, vulkan_abstraction};

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
struct EntityGpuData {
    vertex_buffer: vk::DeviceAddress,
    index_buffer: vk::DeviceAddress,
    material: Material,
}


/// Represents a single light candidate reservoir for Spatiotemporal Reservoir Resampling (ReSTIR).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub(crate) struct Reservoir {
    /// The index of the winning light candidate in the emissive triangles array.
    pub light_idx: u32,
    pub _pad0: [u32; 3],        // std430 vec3 alignment forces offset to 16

    /// The exact 3D world position on the light source that was sampled.
    pub light_pos: [f32; 3],
    pub _pad1: f32,             // std430 vec3 alignment forces offset to 32

    /// The 3D world normal of the light source at the sampled position.
    pub light_normal: [f32; 3],
    /// The sum of all candidate weights evaluated so far.
    pub w_sum: f32,
    /// The number of light candidates that have been processed to get this winner.
    pub m: f32,
    /// The final unbiased probabilistic weight of this reservoir, used to scale the final shadow ray.
    pub w: f32,
    pub _pad2: [u32; 2],        // Pad out to exactly 64 bytes total
}


// ─── Resource Manager ────────────────────────────────────────────────────────

const ARENA_CAPACITY: vk::DeviceSize = 4096;

pub(crate) struct ResourceManager {
    // Camera
    matrices_uniform_buffer: vulkan_abstraction::UniformBuffer<MatricesBufferContents>,

    // Per-entity mesh info (vertex/index addresses + material), indexed by arena slot
    meshes_info_storage_buffer: vulkan_abstraction::ArenaGpuBuffer<EntityGpuData>,

    // Entity management
    entities: HashMap<u64, vulkan_abstraction::Entity>,
    next_entity_id: u64,

    // Entity transforms for shader access (indexed by arena slot, CPU-mapped storage buffer)
    entity_transforms: vulkan_abstraction::StagingBuffer<vk::TransformMatrixKHR>,

    // Acceleration structures
    blases: Vec<vulkan_abstraction::BLAS>,
    tlas: vulkan_abstraction::TLAS,
    instances_buffer: vulkan_abstraction::StagingBuffer<vk::AccelerationStructureInstanceKHR>,
    cpu_instances_data: Vec<BlasMetaData>,

    // Emissive lighting — local-space triangles stored per-BLAS (arena ring buffer)
    blas_emissive_triangles: vulkan_abstraction::ArenaGpuBuffer<vulkan_abstraction::gltf::EmissiveTriangle>,
    // Dense indirection buffer for NEE sampling: (blas_tri_index, entity_arena_slot) pairs
    emissive_indirection_gpu: vulkan_abstraction::GpuOnlyBuffer,

    // Textures
    textures: Vec<(vk::Sampler, vk::ImageView)>,

    // Scene-owned images and samplers
    scene_images: Vec<vulkan_abstraction::Image>,
    scene_samplers: Vec<vulkan_abstraction::Sampler>,

    // Owned images with unique IDs (for runtime destruction)
    owned_images: HashMap<u64, vulkan_abstraction::Image>,
    next_image_id: u64,

    // Fallback and default textures/samplers
    fallback_texture_image: vulkan_abstraction::Image,
    fallback_texture_sampler: vulkan_abstraction::Sampler,
    default_sampler: vulkan_abstraction::Sampler,

    core: Rc<vulkan_abstraction::Core>,
}

impl ResourceManager {
    pub const NUMBER_OF_SAMPLERS: usize = 1024;

    pub fn new_empty(core: Rc<vulkan_abstraction::Core>) -> SrResult<Self> {
        let matrices_uniform_buffer = vulkan_abstraction::UniformBuffer::new(Rc::clone(&core), 1 as vk::DeviceSize)?;
        let meshes_info_storage_buffer = vulkan_abstraction::ArenaGpuBuffer::new(
            core.clone(),
            ARENA_CAPACITY,
            vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_SRC,
            "Meshes info storage buffer",
        )?;
        let entity_transforms = vulkan_abstraction::StagingBuffer::new(
            core.clone(),
            ARENA_CAPACITY,
            vk::BufferUsageFlags::STORAGE_BUFFER,
            "Entity transforms buffer",
        )?;

        let mut instances_buffer = vulkan_abstraction::StagingBuffer::new(
            Rc::clone(&core),
            MAX_TLAS_INSTANCES as vk::DeviceSize,
            vk::BufferUsageFlags::STORAGE_BUFFER
                | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS
                | vk::BufferUsageFlags::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR,
            "Cpu side instances of blases",
        )?;
        let tlas = vulkan_abstraction::TLAS::new(Rc::clone(&core), &[], &mut instances_buffer)?;

        let fallback_texture_image = {
            const RESOLUTION: u32 = 64;
            let image_data = crate::utils::iterate_image_extent(RESOLUTION, RESOLUTION)
                .map(|(x, y)| {
                    if (x + y).is_multiple_of(2) { 0xff000000u32 } else { 0xffff00ffu32 }
                })
                .map(u32::to_be_bytes)
                .flatten()
                .collect::<Vec<u8>>();

            vulkan_abstraction::Image::new_from_data(
                Rc::clone(&core),
                image_data,
                vk::Extent3D { width: RESOLUTION, height: RESOLUTION, depth: 1 },
                vk::Format::R8G8B8A8_UNORM,
                vk::ImageTiling::OPTIMAL,
                gpu_allocator::MemoryLocation::GpuOnly,
                vk::ImageUsageFlags::SAMPLED,
                "fallback texture image",
            )?
        };
        let fallback_texture_sampler = vulkan_abstraction::Sampler::new(
            Rc::clone(&core),
            vk::Filter::NEAREST,
            vk::Filter::NEAREST,
            vk::SamplerAddressMode::REPEAT,
            vk::SamplerAddressMode::REPEAT,
            vk::SamplerAddressMode::REPEAT,
            vk::SamplerMipmapMode::LINEAR,
        )?;
        let default_sampler = vulkan_abstraction::Sampler::new(
            Rc::clone(&core),
            vk::Filter::LINEAR,
            vk::Filter::LINEAR,
            vk::SamplerAddressMode::CLAMP_TO_EDGE,
            vk::SamplerAddressMode::CLAMP_TO_EDGE,
            vk::SamplerAddressMode::CLAMP_TO_EDGE,
            vk::SamplerMipmapMode::LINEAR,
        )?;

        Ok(Self {
            matrices_uniform_buffer,
            meshes_info_storage_buffer,

            entities: HashMap::new(),
            next_entity_id: 0,

            entity_transforms,

            blases: Vec::new(),
            tlas,
            instances_buffer,
            cpu_instances_data: Vec::new(),

            blas_emissive_triangles: vulkan_abstraction::ArenaGpuBuffer::new(
                core.clone(),
                ARENA_CAPACITY,
                vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_SRC,
                "blas emissive triangles",
            )?,
            emissive_indirection_gpu: vulkan_abstraction::Buffer::new_null(Rc::clone(&core)),

            textures: Vec::new(),

            scene_images: Vec::new(),
            scene_samplers: Vec::new(),

            owned_images: HashMap::new(),
            next_image_id: 0,

            fallback_texture_image,
            fallback_texture_sampler,
            default_sampler,

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

    fn entity_gpu_data(blas: &vulkan_abstraction::BLAS, material: &vulkan_abstraction::gltf::Material) -> EntityGpuData {
        EntityGpuData {
            vertex_buffer: blas.vertex_buffer().get_device_address(),
            index_buffer: blas.index_buffer().get_device_address(),
            material: Material::from(material),
        }
    }

    /// Create an entity with the given BLAS index, material, and transform.
    pub fn create_entity(
        &mut self,
        blas_index: usize,
        material: &vulkan_abstraction::gltf::Material,
        transform: vk::TransformMatrixKHR,
    ) -> SrResult<vulkan_abstraction::EntityId> {
        let id = self.next_entity_id;
        self.next_entity_id += 1;

        let gpu_data = Self::entity_gpu_data(&self.blases[blas_index], material);

        let (arena_slot, copy_region) = self.meshes_info_storage_buffer.allocate_and_update(&gpu_data)?;

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

    /// Append local-space emissive triangles for a BLAS into the arena ring buffer.
    /// Allocates slots and flushes the staging copies to GPU.
    pub fn add_blas_emissive_triangles(&mut self, triangles: &[vulkan_abstraction::gltf::EmissiveTriangle]) -> SrResult<()> {
        if triangles.is_empty() {
            return Ok(());
        }

        let mut copy_regions = Vec::with_capacity(triangles.len());
        for tri in triangles {
            let (_slot, copy_region) = self.blas_emissive_triangles.allocate_and_update(tri)?;
            copy_regions.push(copy_region);
        }

        self.flush_single_copy(
            self.blas_emissive_triangles.inner_staging(),
            self.blas_emissive_triangles.inner(),
            &copy_regions,
        )
    }

    /// Rebuild the dense emissive indirection buffer from all live entities and their BLASes' ranges.
    pub fn rebuild_emissive_indirection(&mut self) -> SrResult<()> {
        let mut entries = Vec::new();

        for entity in self.entities.values() {
            let blas = &self.blases[entity.blas_index];
            for range in blas.emissive_triangle_ranges() {
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

    // ─── Acceleration structures ───────────────────────────────────────────

    pub fn tlas(&self) -> &vulkan_abstraction::TLAS {
        &self.tlas
    }

    pub fn blases(&self) -> &[vulkan_abstraction::BLAS] {
        &self.blases
    }

    pub fn blases_mut(&mut self) -> &mut Vec<vulkan_abstraction::BLAS> {
        &mut self.blases
    }

    pub fn cpu_instances_data(&self) -> &[BlasMetaData] {
        &self.cpu_instances_data
    }

    pub fn cpu_instances_data_mut(&mut self) -> &mut Vec<BlasMetaData> {
        &mut self.cpu_instances_data
    }

    pub fn rebuild_tlas(&mut self) -> SrResult<()> {
        let blas_instances: Vec<_> = self
            .cpu_instances_data
            .iter()
            .enumerate()
            .map(|(index, cpu_instance)| BlasInstance {
                blas: &self.blases[index],
                transform: cpu_instance.transform,
                blas_instance_index: cpu_instance.blas_instance_index,
            })
            .collect();

        self.tlas.rebuild(&blas_instances, &mut self.instances_buffer)?;
        Ok(())
    }

    pub fn update_tlas(&mut self) -> SrResult<()> {
        let blas_instances: Vec<_> = self
            .cpu_instances_data
            .iter()
            .enumerate()
            .map(|(index, cpu_instance)| BlasInstance {
                blas: &self.blases[index],
                transform: cpu_instance.transform,
                blas_instance_index: cpu_instance.blas_instance_index,
            })
            .collect();

        self.tlas.update(&blas_instances, &mut self.instances_buffer)?;
        Ok(())
    }

    // ─── Textures ───────────────────────────────────────────────────────────

    pub fn default_sampler(&self) -> &vulkan_abstraction::Sampler {
        &self.default_sampler
    }

    pub fn set_textures(
        &mut self,
        images: &[vulkan_abstraction::Image],
        samplers: &[vulkan_abstraction::Sampler],
        textures: &[vulkan_abstraction::gltf::Texture],
    ) {
        self.textures.clear();
        self.textures.reserve_exact(Self::NUMBER_OF_SAMPLERS);

        for tex in textures {
            let sampler = match tex.sampler {
                Some(i) => &samplers[i],
                None => &self.default_sampler,
            };
            let image = &images[tex.source];
            self.textures.push((sampler.inner(), image.image_view()));
        }

        while self.textures.len() < Self::NUMBER_OF_SAMPLERS {
            self.textures.push((self.fallback_texture_sampler.inner(), self.fallback_texture_image.image_view()));
        }

        assert_eq!(self.textures.len(), Self::NUMBER_OF_SAMPLERS);
    }

    // ─── Scene loading ───────────────────────────────────────────────────────

    pub fn load_scene(&mut self, scene: &crate::Scene, scene_data: crate::SceneData) -> SrResult<()> {
        let (blas_instances, blas_indices, materials, textures, samplers, images, emissive_triangles) =
            scene.load_into_gpu(&self.core, &mut self.blases, scene_data)?;

        // TLAS rebuild (needs blas_instances which borrows self.blases)
        self.tlas.rebuild(&blas_instances, &mut self.instances_buffer)?;

        // Collect entity creation data and consume blas_instances (drops borrow on self.blases)
        let entity_creation_data: Vec<_> = blas_instances
            .iter()
            .zip(blas_indices.iter())
            .zip(materials.iter())
            .map(|((bi, &blas_idx), mat)| (blas_idx, mat.clone(), bi.transform))
            .collect();

        self.cpu_instances_data = blas_instances
            .into_iter()
            .map(|blas_instance| BlasMetaData {
                transform: blas_instance.transform,
                blas_instance_index: blas_instance.blas_instance_index,
            })
            .collect();

        // Now self.blases borrow is free — set textures, feed emissive data, create entities
        self.set_textures(&images, &samplers, &textures);
        self.add_blas_emissive_triangles(&emissive_triangles)?;

        for (blas_idx, material, transform) in &entity_creation_data {
            self.create_entity(*blas_idx, material, *transform)?;
        }

        self.rebuild_emissive_indirection()?;

        self.scene_images = images;
        self.scene_samplers = samplers;

        Ok(())
    }

    // ─── Descriptor set accessors ────────────────────────────────────────────

    pub fn get_matrices_uniform_buffer(&self) -> vk::Buffer {
        self.matrices_uniform_buffer.inner()
    }

    pub fn get_meshes_info_storage_buffer(&self) -> vk::Buffer {
        self.meshes_info_storage_buffer.inner()
    }

    pub fn get_emissive_triangles_storage_buffer(&self) -> vk::Buffer {
        self.blas_emissive_triangles.inner()
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
