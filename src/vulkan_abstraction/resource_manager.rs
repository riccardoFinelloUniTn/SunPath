use std::collections::HashMap;
use std::rc::Rc;

use crate::vulkan_abstraction::{Buffer, EntityGpuData, HostAccessibleBuffer, Material, MatricesBufferContents, BLAS};
use crate::{error::SrResult, vulkan_abstraction, CameraMatrices, MAX_TLAS_INSTANCES};
use ash::vk;
use rand::{RngExt};

const ARENA_CAPACITY: vk::DeviceSize = 4096;

pub(crate) struct ResourceManager { //TODO ring buffer for cameras and instances_buffer or uniform or something and this would be needed for cpu stuff too if they were ever uses outside of gpu data build
    // Camera
    matrices_uniform_buffer: vulkan_abstraction::UniformBuffer<vulkan_abstraction::MatricesBufferContents>,

    // Entity system — single source of truth
    // GPU-side: EntityGpuData per slot (vertex/index addresses, material, transform)
    entities: vulkan_abstraction::ArenaKeyMappedBuffer<vulkan_abstraction::EntityGpuData>,
    // GPU-side: dedicated transform buffer (stride = 48 bytes = VkTransformMatrixKHR), indexed by the same arena slot.
    // Binding 12 reads this as entity_transform_t[slot] in shaders.
    transforms: vulkan_abstraction::ArenaKeyMappedBuffer<vk::TransformMatrixKHR>,
    // CPU-side metadata per entity (blas_index, transform — needed for TLAS rebuild & emissive indirection)
    entity_data: HashMap<u64, vulkan_abstraction::Entity>,


    // Acceleration structures
    blases: HashMap<u64 ,vulkan_abstraction::BLAS>,
    tlas: vulkan_abstraction::TLAS,

    instances_buffer: vulkan_abstraction::StagingBuffer<vk::AccelerationStructureInstanceKHR>,

    /// instance index to entity this is needed to get O(1) reverse search on blas instance removal
    instance_to_entity: HashMap<u64, u64 >,


    // Emissive lighting — local-space triangles stored per-BLAS (arena ring buffer)
    blas_emissive_triangles: vulkan_abstraction::ArenaGpuBuffer<vulkan_abstraction::gltf::EmissiveTriangle>,
    // Dense indirection buffer for NEE sampling: (blas_tri_index, entity_arena_slot) pairs
    emissive_indirection_gpu: vulkan_abstraction::GpuOnlyBuffer,


    // Textures
    textures: Vec<(vk::Sampler, vk::ImageView)>,

    // Samplers loaded from scene
    samplers: Vec<vulkan_abstraction::Sampler>,

    // Owned images with unique IDs (includes scene images)
    images: HashMap<u64, vulkan_abstraction::Image>,

    // Fallback and default textures/samplers
    fallback_texture_image: vulkan_abstraction::Image,
    fallback_texture_sampler: vulkan_abstraction::Sampler,
    default_sampler: vulkan_abstraction::Sampler,

    //these are action to be done at the start or end of frame together with queued free slots for arena buffers
    buffer_copies_queued : Vec<(vk::Buffer,vk::Buffer, vk::BufferCopy)>,

    core: Rc<vulkan_abstraction::Core>,
}

impl ResourceManager {
    pub const NUMBER_OF_SAMPLERS: usize = 1024;

    pub fn new_empty(core: Rc<vulkan_abstraction::Core>) -> SrResult<Self> {
        let matrices_uniform_buffer = vulkan_abstraction::UniformBuffer::new(Rc::clone(&core), 1 as vk::DeviceSize)?;

        let entities = vulkan_abstraction::ArenaKeyMappedBuffer::new(
            core.clone(),
            ARENA_CAPACITY,
            vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_SRC,
            "Entities GPU buffer",
        )?;

        let transforms = vulkan_abstraction::ArenaKeyMappedBuffer::new(
            core.clone(),
            ARENA_CAPACITY,
            vk::BufferUsageFlags::STORAGE_BUFFER,
            "Entity transforms GPU buffer",
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
                    if (x + y).is_multiple_of(2) {
                        0xff000000u32
                    } else {
                        0xffff00ffu32
                    }
                })
                .map(u32::to_be_bytes)
                .flatten()
                .collect::<Vec<u8>>();

            vulkan_abstraction::Image::new_from_data(
                Rc::clone(&core),
                image_data,
                vk::Extent3D {
                    width: RESOLUTION,
                    height: RESOLUTION,
                    depth: 1,
                },
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

            entities,
            transforms,
            entity_data: HashMap::new(),

            blases:Default::default(),
            tlas,
            instances_buffer,

            instance_to_entity: Default::default(),
            blas_emissive_triangles: vulkan_abstraction::ArenaGpuBuffer::new(
                core.clone(),
                ARENA_CAPACITY,
                vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_SRC,
                "blas emissive triangles",
            )?,

            emissive_indirection_gpu: vulkan_abstraction::Buffer::new_null(Rc::clone(&core)),
            textures: Vec::new(),

            samplers: Vec::new(),

            images: HashMap::new(),

            fallback_texture_image,
            fallback_texture_sampler,
            default_sampler,

            buffer_copies_queued: vec![],
            core,
        })
    }


    pub fn empty_out(self) -> SrResult<Self> {
        Self::new_empty(self.core)
    }


    pub fn start_of_frame(&mut self) -> SrResult<()> {
        if self.buffer_copies_queued.is_empty() {
            return Ok(());
        }

        let copies = std::mem::take(&mut self.buffer_copies_queued);

        let device = self.core.device().inner();
        let graphics_queue = self.core.graphics_queue();
        let cmd_pool = self.core.graphics_cmd_pool();

        let cmd_buf = vulkan_abstraction::cmd_buffer::new_command_buffer(cmd_pool, device)?;
        let begin_info = vk::CommandBufferBeginInfo::default().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);

        unsafe {
            device.begin_command_buffer(cmd_buf, &begin_info)?;

            for (src, dst, region) in &copies {
                device.cmd_copy_buffer(cmd_buf, *src, *dst, std::slice::from_ref(region));
            }

            let unique_dsts: std::collections::HashSet<vk::Buffer> = copies.iter().map(|(_, dst, _)| *dst).collect();
            let barriers: Vec<vk::BufferMemoryBarrier2> = unique_dsts
                .into_iter()
                .map(|buf| {
                    vk::BufferMemoryBarrier2::default()
                        .src_stage_mask(vk::PipelineStageFlags2::TRANSFER)
                        .src_access_mask(vk::AccessFlags2::TRANSFER_WRITE)
                        .dst_stage_mask(
                            vk::PipelineStageFlags2::RAY_TRACING_SHADER_KHR | vk::PipelineStageFlags2::COMPUTE_SHADER,
                        )
                        .dst_access_mask(vk::AccessFlags2::SHADER_READ)
                        .buffer(buf)
                        .offset(0)
                        .size(vk::WHOLE_SIZE)
                })
                .collect();

            let dependency_info = vk::DependencyInfo::default().buffer_memory_barriers(&barriers);
            device.cmd_pipeline_barrier2(cmd_buf, &dependency_info);

            device.end_command_buffer(cmd_buf)?;
        }

        graphics_queue.submit_sync(cmd_buf)?;
        unsafe { device.free_command_buffers(cmd_pool.inner(), &[cmd_buf]) };

        Ok(())
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

    /// Build the GPU data for an entity from its BLAS, material, and transform.
    fn build_entity_gpu_data(
        blas: &vulkan_abstraction::BLAS,
        material: &vulkan_abstraction::gltf::Material,
        transform: vk::TransformMatrixKHR,
    ) -> EntityGpuData {
        EntityGpuData {
            vertex_buffer: blas.vertex_buffer().get_device_address(),
            index_buffer: blas.index_buffer().get_device_address(),
            material: Material::from(material),
            transform,
        }
    }

    /// Create an entity with the given BLAS index, material, and transform.
    pub fn create_entity(
        &mut self,
        blas_index: u64,
        material: &vulkan_abstraction::gltf::Material,
        transform: vk::TransformMatrixKHR,
    ) -> SrResult<vulkan_abstraction::EntityId> {
        let id = Self::generate(&self.entity_data);

        let gpu_data = Self::build_entity_gpu_data(&self.blases[&blas_index], material, transform);
        let (slot, copy_region) = self.entities.insert(id, &gpu_data)?;
        self.queue_copy(self.entities.inner_staging(), self.entities.inner(), copy_region);

        let (_, xform_copy) = self.transforms.insert(id, &transform)?;
        self.queue_copy(self.transforms.inner_staging(), self.transforms.inner(), xform_copy);

        let entity = vulkan_abstraction::Entity {
            id: vulkan_abstraction::EntityId(id),
            blas_index,
            transform,
            material: Material::from(material),
            blas_instance_index: slot as u64,
        };
        self.entity_data.insert(id, entity);

        Ok(vulkan_abstraction::EntityId(id))
    }

    /// Destroy an entity. The arena slot is deferred-freed.
    pub fn destroy_entity(&mut self, id: vulkan_abstraction::EntityId) {
        self.entities.remove(id.0);
        self.entity_data.remove(&id.0);
    }

    /// Update an entity's transform. Does NOT rebuild the TLAS — caller must do that.
    pub fn set_entity_transform(&mut self, id: vulkan_abstraction::EntityId, transform: vk::TransformMatrixKHR) -> SrResult<()> {
        if let Some(entity) = self.entity_data.get_mut(&id.0) {
            entity.transform = transform;

            let blas = &self.blases[&entity.blas_index];
            let gpu_data = EntityGpuData {
                vertex_buffer: blas.vertex_buffer().get_device_address(),
                index_buffer: blas.index_buffer().get_device_address(),
                material: entity.material,
                transform,
            };
            let (_slot, copy_region) = self.entities.insert(id.0, &gpu_data)?;
            self.queue_copy(self.entities.inner_staging(), self.entities.inner(), copy_region);

            let (_, xform_copy) = self.transforms.insert(id.0, &transform)?;
            self.queue_copy(self.transforms.inner_staging(), self.transforms.inner(), xform_copy);
        }
        Ok(())
    }

    pub fn get_entity(&self, id: vulkan_abstraction::EntityId) -> Option<&vulkan_abstraction::Entity> {
        self.entity_data.get(&id.0)
    }

    pub fn entity_data(&self) -> &HashMap<u64, vulkan_abstraction::Entity> {
        &self.entity_data
    }

    // ─── Emissive triangles (per-BLAS, local-space) ──────────────────────────

    /// Append local-space emissive triangles for a BLAS into the arena ring buffer.
    /// Allocates slots and flushes the staging copies to GPU.
    pub fn add_blas_emissive_triangles(&mut self, triangles: &[vulkan_abstraction::gltf::EmissiveTriangle]) -> SrResult<()> {
        if triangles.is_empty() {
            return Ok(());
        }

        for tri in triangles {
            let (_slot, copy_region) = self.blas_emissive_triangles.allocate_and_update(tri)?;
            self.queue_copy(self.blas_emissive_triangles.inner_staging(), self.blas_emissive_triangles.inner(), copy_region);
        }

        Ok(())
    }

    /// Rebuild the dense emissive indirection buffer from all live entities and their BLASes' ranges.
    pub fn rebuild_emissive_indirection(&mut self) -> SrResult<()> {
        let mut entries = Vec::new();

        for (&entity_id, entity) in &self.entity_data {
            let blas = &self.blases[&entity.blas_index];
            let arena_slot = self.entities.get_slot(entity_id).unwrap_or(0);
            for range in blas.emissive_triangle_ranges() {
                for tri_idx in range.clone() {
                    entries.push(vulkan_abstraction::gltf::EmissiveIndirectionEntry {
                        blas_tri_index: tri_idx,
                        entity_id: arena_slot as u32, //TODO this is putting a u64 into a u32 this is collision at its finest
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
        let id = Self::generate(&self.images);
        self.images.insert(id, image);
        id
    }

    /// Remove and destroy an image by its ID. No-op if the ID doesn't exist.
    pub fn remove_image(&mut self, id: u64) {
        self.images.remove(&id);
    }

    pub fn get_image(&self, id: u64) -> Option<&vulkan_abstraction::Image> {
        self.images.get(&id)
    }

    // ─── Acceleration structures ───────────────────────────────────────────

    pub fn tlas(&self) -> &vulkan_abstraction::TLAS {
        &self.tlas
    }

    pub fn blases(&self) -> &HashMap<u64, BLAS> {
        &self.blases
    }

    pub fn blases_mut(&mut self) -> &mut HashMap<u64, BLAS> {
        &mut self.blases
    }

    pub fn rebuild_tlas(&mut self) -> SrResult<()> {
        self.tlas.rebuild_from_entities(&self.entity_data, &self.blases, &mut self.instances_buffer)
    }

    pub fn update_tlas(&mut self) -> SrResult<()> {
        self.tlas.update_from_entities(&self.entity_data, &self.blases, &mut self.instances_buffer)
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
            self.textures.push((
                self.fallback_texture_sampler.inner(),
                self.fallback_texture_image.image_view(),
            ));
        }

        assert_eq!(self.textures.len(), Self::NUMBER_OF_SAMPLERS);
    }

    // ─── Scene loading ───────────────────────────────────────────────────────

    pub fn load_scene(&mut self, scene: &crate::Scene, scene_data: crate::SceneData) -> SrResult<()> {
        let mut blases = vec![];
        let (blas_instances, blas_indices, materials, textures, samplers, images, emissive_triangles) =
            scene.load_into_gpu(&self.core, &mut blases , scene_data)?;




        // TLAS rebuild (needs blas_instances which borrows self.blases)
        self.tlas.rebuild(&blas_instances, &mut self.instances_buffer)?;


        // Collect entity creation data and consume blas_instances (drops borrow on self.blases)
        let entity_creation_data: Vec<_> = blas_instances
            .iter()
            .zip(blas_indices.iter())
            .zip(materials.iter())
            .map(|((bi, &blas_idx), mat)| (blas_idx, mat.clone(), bi.transform))
            .collect();

        // Set textures, feed emissive data, create entities
        self.set_textures(&images, &samplers, &textures);
        self.add_blas_emissive_triangles(&emissive_triangles)?;

        blases.into_iter().enumerate().for_each(|(id, blas_i)| {
            self.blases.insert(id as u64, blas_i);
        });

        for (blas_idx, material, transform) in &entity_creation_data {
            self.create_entity(*blas_idx as u64, material, *transform)?;
        }

        self.rebuild_emissive_indirection()?;

        // Take ownership of scene images into the images HashMap
        for image in images {
            self.add_image(image);
        }
        self.samplers = samplers;



        Ok(())
    }

    // ─── Descriptor set accessors ────────────────────────────────────────────

    pub fn get_matrices_uniform_buffer(&self) -> vk::Buffer {
        self.matrices_uniform_buffer.inner()
    }

    /// The entities GPU buffer serves as the meshes info storage buffer.
    /// Shader reads EntityGpuData (vertex/index addresses, material, transform) per slot.
    pub fn get_meshes_info_storage_buffer(&self) -> vk::Buffer {
        self.entities.inner()
    }

    pub fn get_emissive_triangles_storage_buffer(&self) -> vk::Buffer {
        self.blas_emissive_triangles.inner()
    }

    pub fn get_emissive_indirection_buffer(&self) -> vk::Buffer {
        self.emissive_indirection_gpu.inner()
    }

    pub fn get_entity_transforms_buffer(&self) -> vk::Buffer {
        self.transforms.inner()
    }

    pub fn get_textures(&self) -> &[(vk::Sampler, vk::ImageView)] {
        &self.textures
    }

    // ─── Internal helpers ────────────────────────────────────────────────────

    fn queue_copy(&mut self, src: vk::Buffer, dst: vk::Buffer, region: vk::BufferCopy) {
        self.buffer_copies_queued.push((src, dst, region));
    }

    pub(crate) fn generate<T>(hash_map: &HashMap<u64, T>) -> u64 {
        loop {
            let mut rng = rand::rng();
            let key = rng.random::<u64>();
            if !hash_map.contains_key(&key) {
                return key;
            }
        }
    }
}
