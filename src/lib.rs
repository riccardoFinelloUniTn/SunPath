pub mod camera;
pub mod error;
pub mod scene;
pub mod utils;
pub mod vulkan_abstraction;

pub use camera::*;
use error::*;
pub use scene::*;

use std::{collections::HashMap, rc::Rc};

use ash::vk;

use crate::utils::env_var_as_bool;
use crate::vulkan_abstraction::DenoiseDescriptorSetLayout;

struct ImageDependentData {
    pub raytracing_cmd_buf: vulkan_abstraction::CmdBuffer,
    pub blit_cmd_buf: vulkan_abstraction::CmdBuffer,
    #[allow(unused)]
    raytrace_result_image: vulkan_abstraction::Image,
    #[allow(unused)]
    denoise_result_image: vulkan_abstraction::Image,

    #[allow(unused)]
    pub raytracing_descriptor_sets: vulkan_abstraction::RaytracingDescriptorSets,
    #[allow(unused)]
    pub denoise_descriptor_sets: vulkan_abstraction::DenoiseDescriptorSets,
}

pub type CreateSurfaceFn = dyn Fn(&ash::Entry, &ash::Instance) -> SrResult<vk::SurfaceKHR>;

pub struct Renderer {
    image_dependant_data: HashMap<vk::Image, ImageDependentData>,

    shader_data_buffers: vulkan_abstraction::ShaderDataBuffers,

    blases: Vec<vulkan_abstraction::BLAS>,
    tlas: vulkan_abstraction::TLAS,

    scene_images: Vec<vulkan_abstraction::Image>,
    scene_samplers: Vec<vulkan_abstraction::Sampler>,

    shader_binding_table: vulkan_abstraction::ShaderBindingTable,
    ray_tracing_pipeline: vulkan_abstraction::RayTracingPipeline,
    ray_tracing_descriptor_set_layout: vulkan_abstraction::RaytracingDescriptorSetLayout,
    denoise_descriptor_set_layout: DenoiseDescriptorSetLayout,
    image_extent: vk::Extent3D,
    image_format: vk::Format,

    denoise_pipeline: vulkan_abstraction::DenoisePipeline,

    fallback_texture_image: vulkan_abstraction::Image,
    fallback_texture_sampler: vulkan_abstraction::Sampler,
    default_sampler: vulkan_abstraction::Sampler,

    core: Rc<vulkan_abstraction::Core>,

    //used for temporal denoising/antialiasing
    //2 images to avoid race conditions when reading/writing
    pub accumulation_images: [vulkan_abstraction::Image; 2],
    pub frame_count: u32,
}

impl Renderer {
    pub fn new(image_extent: (u32, u32), image_format: vk::Format) -> SrResult<Self> {
        Ok(Self::new_impl(image_extent, image_format, &[], None)?.0)
    }

    // It's necessary to pass a fn to create the surface, because it depends on instance, device depends on it (if present), and both device and
    // instance are created and owned inside Renderer (in Core) so this was deemed a good approach to allow the user to build their own surface
    pub fn new_with_surface(
        image_extent: (u32, u32),
        image_format: vk::Format,
        instance_exts: &'static [*const i8],
        create_surface: &CreateSurfaceFn,
    ) -> SrResult<(Self, vk::SurfaceKHR)> {
        let (r, s) = Self::new_impl(image_extent, image_format, instance_exts, Some(create_surface))?;
        return Ok((r, s.unwrap()));
    }

    fn new_impl(
        image_extent: (u32, u32),
        image_format: vk::Format,
        instance_exts: &'static [*const i8],
        create_surface: Option<&CreateSurfaceFn>,
    ) -> SrResult<(Self, Option<vk::SurfaceKHR>)> {
        let with_validation_layer = env_var_as_bool(ENABLE_VALIDATION_LAYER_ENV_VAR).unwrap_or(IS_DEBUG_BUILD);
        let with_gpuav = env_var_as_bool(ENABLE_GPUAV_ENV_VAR_NAME).unwrap_or(false);
        let (core, surface) = vulkan_abstraction::Core::new_with_surface(
            with_validation_layer,
            with_gpuav,
            image_format,
            instance_exts,
            create_surface,
        )?;
        let core = Rc::new(core);

        let image_extent = utils::tuple_to_extent3d(image_extent);

        let blases = vec![];
        let tlas = vulkan_abstraction::TLAS::new(Rc::clone(&core), &[])?;

        //must be filled by loading a scene
        let shader_data_buffers = vulkan_abstraction::ShaderDataBuffers::new_empty(Rc::clone(&core))?;

        let ray_tracing_descriptor_set_layout = vulkan_abstraction::RaytracingDescriptorSetLayout::new(Rc::clone(&core))?;
        let denoise_descriptor_set_layout = vulkan_abstraction::DenoiseDescriptorSetLayout::new(Rc::clone(&core))?;

        let ray_tracing_pipeline = vulkan_abstraction::RayTracingPipeline::new(
            Rc::clone(&core),
            &ray_tracing_descriptor_set_layout,
            env_var_as_bool(ENABLE_SHADER_DEBUG_SYMBOLS_ENV_VAR).unwrap_or(IS_DEBUG_BUILD),
        )?;

        let denoise_pipeline = vulkan_abstraction::DenoisePipeline::new(
            Rc::clone(&core),
        )?;

        let shader_binding_table = vulkan_abstraction::ShaderBindingTable::new(&core, &ray_tracing_pipeline)?;

        let image_dependant_data = HashMap::new();

        let create_accum_image = |name: &'static str| -> SrResult<vulkan_abstraction::Image> {
            vulkan_abstraction::Image::new(
                core.clone(),
                image_extent, // <--- USE THIS (it's already a vk::Extent3D)
                vk::Format::R32G32B32A32_SFLOAT,
                vk::ImageTiling::OPTIMAL,
                gpu_allocator::MemoryLocation::GpuOnly,
                vk::ImageUsageFlags::STORAGE | vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::TRANSFER_SRC | vk::ImageUsageFlags::TRANSFER_DST,
                name
            )
        };

        let accumulation_images = [
            create_accum_image("Accumulation_Ping")?,
            create_accum_image("Accumulation_Pong")?,
        ];

        let fallback_texture_image = {
            const RESOLUTION: u32 = 64;
            let image_data = utils::iterate_image_extent(RESOLUTION, RESOLUTION)
                .map(|(x, y)| {
                    // black/fucsia checkboard pattern
                    if (x + y).is_multiple_of(2) { 0xff000000 } else { 0xffff00ff }
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
            vk::SamplerAddressMode::REPEAT,
            vk::SamplerAddressMode::REPEAT,
            vk::SamplerAddressMode::REPEAT,
            vk::SamplerMipmapMode::LINEAR,
        )?;

        Ok((
            Self {
                image_dependant_data,

                shader_binding_table,
                ray_tracing_pipeline,
                ray_tracing_descriptor_set_layout,
                denoise_descriptor_set_layout,

                blases,
                tlas,
                scene_images: Vec::new(),
                scene_samplers: Vec::new(),

                image_extent,
                image_format,

                denoise_pipeline,

                accumulation_images,
                frame_count: 0,

                fallback_texture_image,
                fallback_texture_sampler,
                default_sampler,

                shader_data_buffers,

                core,
            },
            surface,
        ))
    }

    pub fn resize(&mut self, image_extent: (u32, u32)) -> SrResult<()> {


        let new_extent = utils::tuple_to_extent3d(image_extent);
        if new_extent == self.image_extent {
            return Ok(());
        }
        self.clear_image_dependent_data();
        self.image_extent = new_extent;

        let create_accum_image = |name: &'static str| -> SrResult<vulkan_abstraction::Image> {
            vulkan_abstraction::Image::new(
                self.core.clone(),
                new_extent,
                vk::Format::R32G32B32A32_SFLOAT,
                vk::ImageTiling::OPTIMAL,
                gpu_allocator::MemoryLocation::GpuOnly,
                vk::ImageUsageFlags::STORAGE | vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::TRANSFER_SRC | vk::ImageUsageFlags::TRANSFER_DST,
                name
            )
        };


        self.accumulation_images = [
            create_accum_image("Accumulation_1")?,
            create_accum_image("Accumulation_2")?,
        ];

        self.frame_count = 0;

        //self.image_dependant_data
        Ok(())
    }

    pub fn clear_image_dependent_data(&mut self) {
        self.image_dependant_data = HashMap::new();
    }

    pub fn build_image_dependent_data(&mut self, images: &[vk::Image]) -> SrResult<()> {
        for post_blit_image in images {
            let raytrace_result_image = vulkan_abstraction::Image::new(
                Rc::clone(&self.core),
                self.image_extent,
                vk::Format::R32G32B32A32_SFLOAT,
                vk::ImageTiling::OPTIMAL,
                gpu_allocator::MemoryLocation::GpuOnly,
                vk::ImageUsageFlags::STORAGE | vk::ImageUsageFlags::TRANSFER_SRC,
                "sunray (internal, pre-blit) raytrace result image",
            )?;

            let denoise_result_image = vulkan_abstraction::Image::new(
                Rc::clone(&self.core),
                self.image_extent,
                vk::Format::R32G32B32A32_SFLOAT,
                vk::ImageTiling::OPTIMAL,
                gpu_allocator::MemoryLocation::GpuOnly,
                vk::ImageUsageFlags::STORAGE | vk::ImageUsageFlags::TRANSFER_SRC,
                "sunray (internal, pre-blit) denoise result image",
            )?;

            let raytracing_descriptor_sets = vulkan_abstraction::RaytracingDescriptorSets::new(
                Rc::clone(&self.core),
                &self.ray_tracing_descriptor_set_layout,
                &self.tlas,
                raytrace_result_image.image_view(),
                &self.shader_data_buffers,
            )?;

            let denoise_descriptor_sets = vulkan_abstraction::DenoiseDescriptorSets::new(Rc::clone(&self.core), &self.denoise_descriptor_set_layout, &raytrace_result_image, &denoise_result_image)?;

            raytracing_descriptor_sets.update_accumulation_images(&self.accumulation_images, self.default_sampler.inner());

            let blit_cmd_buf = vulkan_abstraction::CmdBuffer::new(Rc::clone(&self.core))?;
            let raytracing_cmd_buf = vulkan_abstraction::CmdBuffer::new(Rc::clone(&self.core))?;
            

            //record blit
            {
                let cmd_buf_begin_info =
                    vk::CommandBufferBeginInfo::default().flags(vk::CommandBufferUsageFlags::SIMULTANEOUS_USE);
                unsafe {
                    self.core
                        .device()
                        .inner()
                        .begin_command_buffer(blit_cmd_buf.inner(), &cmd_buf_begin_info)
                }?;

                Self::cmd_blit_image(
                    &self.core,
                    blit_cmd_buf.inner(),
                    denoise_result_image.inner(),
                    denoise_result_image.extent(),
                    *post_blit_image,
                    denoise_result_image.image_subresource_range(),
                )?;

                unsafe { self.core.device().inner().end_command_buffer(blit_cmd_buf.inner()) }?;
            }

            self.image_dependant_data.insert(
                *post_blit_image,
                ImageDependentData {
                    raytrace_result_image,
                    denoise_result_image,
                    raytracing_cmd_buf,
                    blit_cmd_buf,
                    raytracing_descriptor_sets,
                    denoise_descriptor_sets,
                },
            );
        }

        Ok(())
    }


    pub fn load_gltf(&mut self, path: &str) -> SrResult<()> {
        let gltf = vulkan_abstraction::gltf::Gltf::new(Rc::clone(&self.core), path)?;
        let (default_scene, scene_data) = gltf.create_default_scene()?;

        self.load_scene(&default_scene, scene_data)?;
        Ok(())
    }

    pub fn load_scene(&mut self, scene: &crate::Scene, scene_data: crate::SceneData) -> SrResult<()> {
        let (blas_instances, materials, textures, samplers, images) =
            scene.load_into_gpu(&self.core, &mut self.blases, scene_data)?;

        let fallback_texture = vulkan_abstraction::Texture(&self.fallback_texture_image, &self.fallback_texture_sampler);
        self.tlas.rebuild(&blas_instances)?;
        self.shader_data_buffers.update(
            &blas_instances,
            &materials,
            &images,
            &samplers,
            &textures,
            fallback_texture,
            &self.default_sampler,
        )?;

        self.scene_images = images;
        self.scene_samplers = samplers;

        self.clear_image_dependent_data();

        Ok(())
    }

    pub fn set_camera(&mut self, camera: crate::Camera) -> SrResult<()> {
        self.shader_data_buffers.set_matrices(camera.as_matrices(self.image_extent))
    }

    /// Render to dst_image. the user may also pass a Semaphore which the user should signal when the image is
    /// ready to be written to (for example after being acquired from a swapchain) and a Fence will be returned
    /// that will be signaled when the rendering is finished (which can be used to know when the Semaphore has no pending operations left).
    pub fn render_to_image(&mut self, dst_image: vk::Image, wait_sem: vk::Semaphore) -> SrResult<vk::Fence> {


        if !self.image_dependant_data.contains_key(&dst_image) {
            self.build_image_dependent_data(&[dst_image])?;
        }

        let this_ptr = self as *mut Self;

        let img_dependent_data = self.image_dependant_data.get_mut(&dst_image).unwrap();

        // Raytracing
        img_dependent_data.raytracing_cmd_buf.fence_mut().wait()?;
        img_dependent_data.blit_cmd_buf.fence_mut().wait()?;

        let cmd_buf = img_dependent_data.raytracing_cmd_buf.inner();
        let result_image = img_dependent_data.raytrace_result_image.inner();
        let denoised_image = img_dependent_data.denoise_result_image.inner();
        let result_extent = img_dependent_data.raytrace_result_image.extent();

        let raytracing_descriptor_sets_ptr = &img_dependent_data.raytracing_descriptor_sets as *const vulkan_abstraction::RaytracingDescriptorSets;
        let denoise_descriptor_sets_ptr = &img_dependent_data.denoise_descriptor_sets as *const vulkan_abstraction::DenoiseDescriptorSets;


        unsafe {
            // Use (*this_ptr).core because 'self.core' is locked by 'img_dependent_data'
            let device = (*this_ptr).core.device().inner();

            let begin_info = vk::CommandBufferBeginInfo::default()
                .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);

            device.begin_command_buffer(cmd_buf, &begin_info)?;

            (*this_ptr).cmd_raytracing_render(
                cmd_buf,
                &*raytracing_descriptor_sets_ptr, // Dereference back to reference
                result_image,
                result_extent,
            )?;

            let memory_barrier = vk::MemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::SHADER_WRITE) // Wait for RT to finish writing
                .dst_access_mask(vk::AccessFlags::SHADER_READ); // Make sure Denoise can see the data

            device.cmd_pipeline_barrier(
                cmd_buf,
                vk::PipelineStageFlags::RAY_TRACING_SHADER_KHR, // Source Stage
                vk::PipelineStageFlags::COMPUTE_SHADER,         // Destination Stage (Denoising is usually compute)
                vk::DependencyFlags::empty(),
                &[memory_barrier], // Use a global memory barrier for simplicity
                &[],
                &[],
            );

            (*this_ptr).cmd_denoise_image(
                cmd_buf,
                &*denoise_descriptor_sets_ptr,
                result_extent.width,
                result_extent.height,
                result_image,
                denoised_image,
            )?;



            device.end_command_buffer(cmd_buf)?;


            // Submit using (*this_ptr)
            (*this_ptr).core.queue().submit_async(
                img_dependent_data.raytracing_cmd_buf.inner(),
                &[],
                &[],
                &[],
                img_dependent_data.raytracing_cmd_buf.fence_mut().submit()?,
            )?;
        }


        // Blitting
        let (wait_sems, wait_dst_stages) = ([wait_sem], [vk::PipelineStageFlags::ALL_GRAPHICS]);
        let (wait_sems, wait_dst_stages) = if wait_sem == vk::Semaphore::null() {
            ([].as_slice(), [].as_slice())
        } else {
            (wait_sems.as_slice(), wait_dst_stages.as_slice())
        };

        let signal_fence = img_dependent_data.blit_cmd_buf.fence_mut().submit()?;

        unsafe {
            // Again, use (*this_ptr) because img_dependent_data is still alive here
            (*this_ptr).core.queue().submit_async(
                img_dependent_data.blit_cmd_buf.inner(),
                &wait_sems,
                &wait_dst_stages,
                &[],
                signal_fence,
            )?;
        }

        Ok(signal_fence)
    }

    pub fn render_to_host_memory(&mut self) -> SrResult<Vec<u8>> {
        let mut dst_image = vulkan_abstraction::Image::new(
            Rc::clone(&self.core),
            self.image_extent,
            self.image_format,
            vk::ImageTiling::LINEAR,
            gpu_allocator::MemoryLocation::GpuToCpu,
            vk::ImageUsageFlags::STORAGE | vk::ImageUsageFlags::TRANSFER_DST,
            "mapped sunray output image",
        )?;

        let wait_fence = self.render_to_image(dst_image.inner(), vk::Semaphore::null())?;
        vulkan_abstraction::wait_fence(self.core.device(), wait_fence)?;

        Ok(dst_image.get_raw_image_data_with_no_padding()?)
    }

    fn cmd_raytracing_render(
        &mut self,
        cmd_buf: vk::CommandBuffer,
        descriptor_sets: &vulkan_abstraction::RaytracingDescriptorSets,
        image: vk::Image,
        extent: vk::Extent3D,
    ) -> SrResult<()> {

        let device = self.core.device().inner();

        //ping pong to avoid errors when using accumulation images
        let history_idx = (self.frame_count % 2) as usize;
        let accum_idx = ((self.frame_count + 1) % 2) as usize;


        // Use GENERAL for everything to rule out layout mismatches
        let (old_layout, src_stage, src_access) = if self.frame_count == 0 {
            (vk::ImageLayout::UNDEFINED, vk::PipelineStageFlags::TOP_OF_PIPE, vk::AccessFlags::empty())
        } else {
            (vk::ImageLayout::GENERAL, vk::PipelineStageFlags::RAY_TRACING_SHADER_KHR, vk::AccessFlags::SHADER_WRITE | vk::AccessFlags::SHADER_READ)
        };

        // Initializing push constant values
        let push_constants = vulkan_abstraction::PushConstant {
            frame_count: self.frame_count,
            use_srgb: self.image_format == vk::Format::R8G8B8A8_SRGB,
            _padding: [0; 3],
        };


        self.frame_count += 1;
        unsafe {

            let subresource_range = vk::ImageSubresourceRange::default()
                .aspect_mask(vk::ImageAspectFlags::COLOR)
                .base_mip_level(0)
                .level_count(vk::REMAINING_MIP_LEVELS)
                .base_array_layer(0)
                .layer_count(vk::REMAINING_ARRAY_LAYERS);

            let make_barrier = |img: vk::Image, old: vk::ImageLayout, new: vk::ImageLayout, src_a, dst_a| {
                vk::ImageMemoryBarrier::default()
                    .src_access_mask(src_a)
                    .dst_access_mask(dst_a)
                    .old_layout(old)
                    .new_layout(new)
                    .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .image(img)
                    .subresource_range(subresource_range)
            };

            // Barrier 1: Swapchain
            let b_swap = make_barrier(
                image,
                vk::ImageLayout::UNDEFINED,
                vk::ImageLayout::GENERAL,
                vk::AccessFlags::empty(),
                vk::AccessFlags::SHADER_WRITE
            );

            let (hist_old, hist_src) = if self.frame_count == 1 {
                (vk::ImageLayout::UNDEFINED, vk::AccessFlags::empty())
            } else {
                (vk::ImageLayout::GENERAL, vk::AccessFlags::SHADER_WRITE)
            };

            // Barrier 2: History (Targeting GENERAL)
            let b_hist = make_barrier(
                self.accumulation_images[history_idx].inner(),
                hist_old,
                vk::ImageLayout::GENERAL,
                hist_src,
                vk::AccessFlags::SHADER_READ
            );

            let (accum_old, accum_src) = if self.frame_count == 1 {
                (vk::ImageLayout::UNDEFINED, vk::AccessFlags::empty())
            } else {
                (vk::ImageLayout::GENERAL, vk::AccessFlags::SHADER_READ)
            };

            // Barrier 3: Accum (Targeting GENERAL)
            let b_accum = make_barrier(
                self.accumulation_images[accum_idx].inner(),
                accum_old,
                vk::ImageLayout::GENERAL,
                accum_src,
                vk::AccessFlags::SHADER_WRITE
            );

            device.cmd_pipeline_barrier(
                cmd_buf,
                // Frame 1: Wait for nothing. Frame 2+: Wait for previous Ray Tracing.
                if self.frame_count == 1 { vk::PipelineStageFlags::TOP_OF_PIPE } else { vk::PipelineStageFlags::RAY_TRACING_SHADER_KHR },
                vk::PipelineStageFlags::RAY_TRACING_SHADER_KHR,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &[b_swap, b_hist, b_accum],
            );


            device.cmd_bind_pipeline(
                cmd_buf,
                vk::PipelineBindPoint::RAY_TRACING_KHR,
                self.ray_tracing_pipeline.inner(),
            );
            device.cmd_bind_descriptor_sets(
                cmd_buf,
                vk::PipelineBindPoint::RAY_TRACING_KHR,
                self.ray_tracing_pipeline.layout(),
                0,
                descriptor_sets.inner(),
                &[],
            );
            device.cmd_push_constants(
                cmd_buf,
                self.ray_tracing_pipeline.layout(),
                vk::ShaderStageFlags::RAYGEN_KHR | vk::ShaderStageFlags::CLOSEST_HIT_KHR | vk::ShaderStageFlags::MISS_KHR,
                0,
                &std::mem::transmute::<
                    vulkan_abstraction::PushConstant,
                    [u8; std::mem::size_of::<vulkan_abstraction::PushConstant>()],
                >(push_constants), //TODO: comment this transmute
            );
            self.core.rt_pipeline_device().cmd_trace_rays(
                cmd_buf,
                self.shader_binding_table.raygen_region(),
                self.shader_binding_table.miss_region(),
                self.shader_binding_table.hit_region(),
                self.shader_binding_table.callable_region(),
                extent.width,
                extent.height,
                extent.depth, //for now it's one because of the Extent2D.into()
            );
        }

        Ok(())
    }

    fn cmd_denoise_image(
        &self,
        cmd_buf: vk::CommandBuffer,
        descriptor_sets: &vulkan_abstraction::DenoiseDescriptorSets,
        width: u32,
        height: u32,
        input_image:  vk::Image,
        output_image:  vk::Image,
    ) -> SrResult<()> {
        let device = self.core.device().inner();

        let input_barrier = vk::ImageMemoryBarrier::default()
            .src_access_mask(vk::AccessFlags::SHADER_WRITE) // RT wrote to it
            .dst_access_mask(vk::AccessFlags::SHADER_READ)  // Denoise reads it
            .old_layout(vk::ImageLayout::GENERAL)
            .new_layout(vk::ImageLayout::GENERAL)
            .image(input_image)
            .subresource_range(vk::ImageSubresourceRange {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                base_mip_level: 0,
                level_count: 1,
                base_array_layer: 0,
                layer_count: 1,
            });

        // Ensure the output image is ready to be written to
        let output_barrier = vk::ImageMemoryBarrier::default()
            .src_access_mask(vk::AccessFlags::empty())      // Don't care what it was before
            .dst_access_mask(vk::AccessFlags::SHADER_WRITE) // Denoise writes to it
            .old_layout(vk::ImageLayout::UNDEFINED)         // Discard previous content
            .new_layout(vk::ImageLayout::GENERAL)
            .image(output_image)
            .subresource_range(vk::ImageSubresourceRange {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                base_mip_level: 0,
                level_count: 1,
                base_array_layer: 0,
                layer_count: 1,
            });



        unsafe{
            device.cmd_pipeline_barrier(
                cmd_buf,
                vk::PipelineStageFlags::RAY_TRACING_SHADER_KHR, // Wait for RT
                vk::PipelineStageFlags::COMPUTE_SHADER,         // Block Compute
                vk::DependencyFlags::empty(),
                &[], &[],
                &[input_barrier, output_barrier]
            );

            device.cmd_bind_pipeline(
                cmd_buf,
                vk::PipelineBindPoint::COMPUTE,
                self.denoise_pipeline.inner(),
            );

            device.cmd_bind_descriptor_sets(
                cmd_buf,
                vk::PipelineBindPoint::COMPUTE,
                self.denoise_pipeline.layout(),
                0,
                descriptor_sets.inner(),
                &[],
            );
        }




        let group_x = (width + 15) / 16;
        let group_y = (height + 15) / 16;
        unsafe {
            device.cmd_dispatch(cmd_buf, group_x, group_y, 1);
        }


        let post_barrier = vk::ImageMemoryBarrier::default()
            .src_access_mask(vk::AccessFlags::SHADER_WRITE) // Denoise wrote it
            .dst_access_mask(vk::AccessFlags::TRANSFER_READ) // Blit reads it
            .old_layout(vk::ImageLayout::GENERAL)
            .new_layout(vk::ImageLayout::GENERAL)
            .image(output_image)
            .subresource_range(vk::ImageSubresourceRange {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                base_mip_level: 0,
                level_count: 1,
                base_array_layer: 0,
                layer_count: 1,
            });

        unsafe {
            device.cmd_pipeline_barrier(
                cmd_buf,
                vk::PipelineStageFlags::COMPUTE_SHADER, // Wait for Denoise
                vk::PipelineStageFlags::TRANSFER,       // Block Transfer (Blit)
                vk::DependencyFlags::empty(),
                &[], &[],
                &[post_barrier]
            );
        }


        Ok(())

    }
    fn cmd_blit_image(
        core: &vulkan_abstraction::Core,
        cmd_buf: vk::CommandBuffer,
        src_image: vk::Image,
        extent: vk::Extent3D,
        dst_image: vk::Image,
        image_subresource_range: &vk::ImageSubresourceRange,
    ) -> SrResult<()> {
        let device = core.device().inner();

        let image_subresource_layer = vk::ImageSubresourceLayers::default()
            .aspect_mask(image_subresource_range.aspect_mask)
            .base_array_layer(image_subresource_range.base_array_layer)
            .layer_count(image_subresource_range.layer_count)
            .mip_level(image_subresource_range.base_mip_level);
        let zero_offset = vk::Offset3D { x: 0, y: 0, z: 0 };
        let src_whole_image_offset = vk::Offset3D::default()
            .x(extent.width as i32)
            .y(extent.height as i32)
            .z(extent.depth as i32);
        let dst_whole_image_offset = vk::Offset3D::default()
            .x(extent.width as i32)
            .y(extent.height as i32)
            .z(extent.depth as i32);
        let src_offsets = [zero_offset, src_whole_image_offset];
        let dst_offsets = [zero_offset, dst_whole_image_offset];
        let image_blit = vk::ImageBlit::default()
            .src_subresource(image_subresource_layer)
            .src_offsets(src_offsets)
            .dst_subresource(image_subresource_layer)
            .dst_offsets(dst_offsets);

        unsafe {
            //transition src_image from general to transfer source layout
            vulkan_abstraction::cmd_image_memory_barrier(
                core,
                cmd_buf,
                src_image,
                vk::PipelineStageFlags::RAY_TRACING_SHADER_KHR,
                vk::PipelineStageFlags::TRANSFER,
                vk::AccessFlags::SHADER_WRITE,
                vk::AccessFlags::TRANSFER_READ,
                vk::ImageLayout::GENERAL,
                vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
            );

            //transition dst_image to transfer destination layout
            vulkan_abstraction::cmd_image_memory_barrier(
                core,
                cmd_buf,
                dst_image,
                vk::PipelineStageFlags::TOP_OF_PIPE,
                vk::PipelineStageFlags::TRANSFER,
                vk::AccessFlags::empty(),
                vk::AccessFlags::TRANSFER_WRITE,
                vk::ImageLayout::UNDEFINED,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            );

            device.cmd_blit_image(
                cmd_buf,
                src_image,
                vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                dst_image,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                &[image_blit],
                vk::Filter::NEAREST,
            );

            //transition dst_image to general layout which is required for mapping the image
            vulkan_abstraction::cmd_image_memory_barrier(
                core,
                cmd_buf,
                dst_image,
                vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::ALL_GRAPHICS, // the image should already be transitioned when the user makes use of it
                vk::AccessFlags::TRANSFER_WRITE,
                vk::AccessFlags::MEMORY_READ,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                vk::ImageLayout::GENERAL,
            );

            //transition back src_image to general layout
            vulkan_abstraction::cmd_image_memory_barrier(
                core,
                cmd_buf,
                src_image,
                vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::BOTTOM_OF_PIPE,
                vk::AccessFlags::TRANSFER_READ,
                vk::AccessFlags::empty(),
                vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                vk::ImageLayout::GENERAL,
            );
        }

        Ok(())
    }

    pub fn core(&self) -> &Rc<vulkan_abstraction::Core> {
        &self.core
    }
}

// useful environment variables, set to 1 or 0
const ENABLE_VALIDATION_LAYER_ENV_VAR: &'static str = "ENABLE_VALIDATION_LAYER"; // defaults to 0 in debug build, to 1 in release build
const ENABLE_GPUAV_ENV_VAR_NAME: &'static str = "ENABLE_GPUAV"; // does nothing unless validation layer is enabled, defaults to 0
const ENABLE_SHADER_DEBUG_SYMBOLS_ENV_VAR: &'static str = "ENABLE_SHADER_DEBUG_SYMBOLS"; // defaults to 0 in debug build, to 1 in release build
const IS_DEBUG_BUILD: bool = cfg!(debug_assertions);

impl Drop for Renderer {
    fn drop(&mut self) {
        match self.core().queue().wait_idle() {
            Ok(()) => {}
            Err(e) => match e.get_source() {
                ErrorSource::Vulkan(e) => {
                    log::warn!("VkQueueWaitIdle s returned {e:?} in sunray::Renderer::drop")
                }
                _ => log::warn!("VkQueueWaitIdle returned {e} in sunray::Renderer::drop"),
            },
        }
    }
}
