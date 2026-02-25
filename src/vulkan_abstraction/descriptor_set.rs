use std::rc::Rc;

use ash::vk;

use crate::{error::*, vulkan_abstraction};

use vulkan_abstraction::TLAS;

pub struct RaytracingDescriptorSetLayout {
    descriptor_set_layout: vk::DescriptorSetLayout,

    core: Rc<vulkan_abstraction::Core>,
}

impl RaytracingDescriptorSetLayout {
    pub const TLAS_BINDING: u32 = 0;
    pub const OUTPUT_IMAGE_BINDING: u32 = 1;
    pub const MATRICES_UNIFORM_BUFFER_BINDING: u32 = 2;
    pub const MESHES_INFO_STORAGE_BUFFER_BINDING: u32 = 3;
    pub const SAMPLERS_BINDING: u32 = 4;
    pub const DEPTH_BINDING: u32 = 5;
    pub const NORMAL_BINDING: u32 = 6;
    pub const MOTION_VECTOR_BINDING: u32 = 7;
    pub const NUMBER_OF_BINDINGS: usize = 8;


    pub const NUMBER_OF_SAMPLERS: u32 = vulkan_abstraction::ShaderDataBuffers::NUMBER_OF_SAMPLERS as u32;

    pub fn new(core: Rc<vulkan_abstraction::Core>) -> SrResult<Self> {
        let device = core.device().inner();

        // TODO: check the stage_flags for each binding
        let descriptor_set_layout_bindings = [
            // TLAS layout binding
            vk::DescriptorSetLayoutBinding::default()
                .binding(Self::TLAS_BINDING)
                .descriptor_type(vk::DescriptorType::ACCELERATION_STRUCTURE_KHR)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::RAYGEN_KHR | vk::ShaderStageFlags::CLOSEST_HIT_KHR),
            // output image layout binding
            vk::DescriptorSetLayoutBinding::default()
                .binding(Self::OUTPUT_IMAGE_BINDING)
                .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::RAYGEN_KHR),
            // matrices uniform buffer layout binding
            vk::DescriptorSetLayoutBinding::default()
                .binding(Self::MATRICES_UNIFORM_BUFFER_BINDING)
                .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::ALL),
            // meshes info uniform buffer layout binding
            vk::DescriptorSetLayoutBinding::default()
                .binding(Self::MESHES_INFO_STORAGE_BUFFER_BINDING)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::ALL),
            // samplers buffer layout binding
            vk::DescriptorSetLayoutBinding::default()
                .binding(Self::SAMPLERS_BINDING)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .descriptor_count(Self::NUMBER_OF_SAMPLERS)
                .stage_flags(vk::ShaderStageFlags::ALL),
            // Depth layout binding
            vk::DescriptorSetLayoutBinding::default()
                .binding(Self::DEPTH_BINDING)
                .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::RAYGEN_KHR),
            // normal layout binding
            vk::DescriptorSetLayoutBinding::default()
                .binding(Self::NORMAL_BINDING)
                .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::RAYGEN_KHR),
            // motion vectors binding
            vk::DescriptorSetLayoutBinding::default()
                .binding(Self::MOTION_VECTOR_BINDING)
                .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::RAYGEN_KHR),
        ];

        let descriptor_set_layout_create_info =
            vk::DescriptorSetLayoutCreateInfo::default().bindings(&descriptor_set_layout_bindings);

        let descriptor_set_layout = unsafe { device.create_descriptor_set_layout(&descriptor_set_layout_create_info, None) }?;

        Ok(Self {
            descriptor_set_layout,
            core,
        })
    }

    pub fn inner(&self) -> vk::DescriptorSetLayout {
        self.descriptor_set_layout
    }
}
impl Drop for RaytracingDescriptorSetLayout {
    fn drop(&mut self) {
        unsafe {
            self.core
                .device()
                .inner()
                .destroy_descriptor_set_layout(self.descriptor_set_layout, None)
        };
    }
}

pub struct DenoiseDescriptorSetLayout {
    descriptor_set_layout: vk::DescriptorSetLayout,
    core: Rc<vulkan_abstraction::Core>,
}

impl DenoiseDescriptorSetLayout {
    pub const RAW_COLOR_BINDING: u32 = 0;
    pub const OUTPUT_IMAGE_BINDING: u32 = 1;
    pub const DEPTH_BINDING: u32 = 2;
    pub const NORMAL_BINDING: u32 = 3;
    pub const MOTION_VECTOR_BINDING: u32 = 4;
    pub const HISTORY_BINDING: u32 = 5;
    pub const ACCUMULATION_BINDING: u32 = 6;

    pub const NUMBER_OF_BINDINGS: usize = 7;

    pub fn new(core: Rc<vulkan_abstraction::Core>) -> SrResult<Self> {
        let device = core.device().inner();

        let bindings = [
            vk::DescriptorSetLayoutBinding::default()
                .binding(Self::RAW_COLOR_BINDING)
                .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE),

            vk::DescriptorSetLayoutBinding::default()
                .binding(Self::OUTPUT_IMAGE_BINDING)
                .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE),

            vk::DescriptorSetLayoutBinding::default()
                .binding(Self::DEPTH_BINDING)
                .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE),

            vk::DescriptorSetLayoutBinding::default()
                .binding(Self::NORMAL_BINDING)
                .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE),

            vk::DescriptorSetLayoutBinding::default()
                .binding(Self::MOTION_VECTOR_BINDING)
                .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE),

            vk::DescriptorSetLayoutBinding::default()
                .binding(Self::HISTORY_BINDING)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .descriptor_count(2)
                .stage_flags(vk::ShaderStageFlags::COMPUTE),

            // Accumulation layout binding (Write-only, uses imageStore)
            vk::DescriptorSetLayoutBinding::default()
                .binding(Self::ACCUMULATION_BINDING)
                .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                .descriptor_count(2)
                .stage_flags(vk::ShaderStageFlags::COMPUTE),


        ];

        let create_info = vk::DescriptorSetLayoutCreateInfo::default()
            .bindings(&bindings);

        let descriptor_set_layout = unsafe {
            device.create_descriptor_set_layout(&create_info, None)?
        };

        Ok(Self {
            descriptor_set_layout,
            core,
        })
    }

    pub fn inner(&self) -> vk::DescriptorSetLayout {
        self.descriptor_set_layout
    }
}

impl Drop for DenoiseDescriptorSetLayout {
    fn drop(&mut self) {
        unsafe {
            self.core
            .device()
            .inner()
            .destroy_descriptor_set_layout(self.descriptor_set_layout, None);
        }
    }
}

pub(crate) struct RaytracingDescriptorSets {
    descriptor_sets: Vec<vk::DescriptorSet>,
    descriptor_pool: vk::DescriptorPool,

    core: Rc<vulkan_abstraction::Core>,
}

impl RaytracingDescriptorSets {
    pub fn new(
        core: Rc<vulkan_abstraction::Core>,
        descriptor_set_layout: &RaytracingDescriptorSetLayout,
        tlas: &TLAS,
        output_image: &vulkan_abstraction::Image,       // Changed from ImageView to Image
        depth_image: &vulkan_abstraction::Image,
        normal_image: &vulkan_abstraction::Image,
        motion_vector_image: &vulkan_abstraction::Image,
        shader_data: &vulkan_abstraction::ShaderDataBuffers,
    ) -> SrResult<Self> {
        let device = core.device().inner();
        let descriptor_pool_sizes = [
            vk::DescriptorPoolSize::default()
                .ty(vk::DescriptorType::ACCELERATION_STRUCTURE_KHR)
                .descriptor_count(1),
            vk::DescriptorPoolSize::default()
                .ty(vk::DescriptorType::STORAGE_IMAGE)
                .descriptor_count(4), // 1 Output + 1 Depth + 1 Normal + 1 Motion Vector
            vk::DescriptorPoolSize::default()
                .ty(vk::DescriptorType::UNIFORM_BUFFER)
                .descriptor_count(1),
            vk::DescriptorPoolSize::default()
                .ty(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(1),
            vk::DescriptorPoolSize::default()
                .ty(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .descriptor_count(RaytracingDescriptorSetLayout::NUMBER_OF_SAMPLERS),
        ];

        let descriptor_pool_create_info = vk::DescriptorPoolCreateInfo::default()
            .pool_sizes(&descriptor_pool_sizes)
            .max_sets(1);

        let descriptor_pool = unsafe { device.create_descriptor_pool(&descriptor_pool_create_info, None) }?;

        let descriptor_set_layouts = [descriptor_set_layout.inner()];

        let descriptor_set_allocate_info = vk::DescriptorSetAllocateInfo::default()
            .descriptor_pool(descriptor_pool)
            .set_layouts(&descriptor_set_layouts);

        let descriptor_sets = unsafe { device.allocate_descriptor_sets(&descriptor_set_allocate_info) }?;

        let mut descriptor_writes = Vec::new();

        // write TLAS to descriptor set
        let tlases = [tlas.inner()];
        let mut write_descriptor_set_acceleration_structure =
            vk::WriteDescriptorSetAccelerationStructureKHR::default().acceleration_structures(&tlases);

        // Using push() for writes, just like your original code
        let mut push_write = |write| descriptor_writes.push(write);

        push_write(
            vk::WriteDescriptorSet::default()
                .descriptor_type(vk::DescriptorType::ACCELERATION_STRUCTURE_KHR)
                .push_next(&mut write_descriptor_set_acceleration_structure)
                .dst_set(descriptor_sets[0])
                .dst_binding(RaytracingDescriptorSetLayout::TLAS_BINDING)
                .descriptor_count(1), // Added descriptor_count
        );

        // --- NEW: Helper for images to keep code clean ---
        let create_info = |img: &vulkan_abstraction::Image| {
            [vk::DescriptorImageInfo::default()
                .image_view(img.image_view())
                .image_layout(vk::ImageLayout::GENERAL)]
        };

        let output_info = create_info(output_image);
        let depth_info = create_info(depth_image);
        let normal_info = create_info(normal_image);
        let mv_info = create_info(motion_vector_image);

        // Write Output Image
        push_write(
            vk::WriteDescriptorSet::default()
                .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                .image_info(&output_info)
                .dst_set(descriptor_sets[0])
                .dst_binding(RaytracingDescriptorSetLayout::OUTPUT_IMAGE_BINDING),
        );

        // Write Depth Image
        push_write(
            vk::WriteDescriptorSet::default()
                .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                .image_info(&depth_info)
                .dst_set(descriptor_sets[0])
                .dst_binding(RaytracingDescriptorSetLayout::DEPTH_BINDING), // Make sure this constant exists! (e.g., 7)
        );

        // Write Normal Image
        push_write(
            vk::WriteDescriptorSet::default()
                .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                .image_info(&normal_info)
                .dst_set(descriptor_sets[0])
                .dst_binding(RaytracingDescriptorSetLayout::NORMAL_BINDING), // Make sure this constant exists! (e.g., 8)
        );

        // Write Motion Vector Image
        push_write(
            vk::WriteDescriptorSet::default()
                .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                .image_info(&mv_info)
                .dst_set(descriptor_sets[0])
                .dst_binding(RaytracingDescriptorSetLayout::MOTION_VECTOR_BINDING), // Make sure this constant exists! (e.g., 9)
        );


        // write matrices uniform buffer to descriptor set
        let descriptor_buffer_infos = [vk::DescriptorBufferInfo::default()
            .buffer(shader_data.get_matrices_uniform_buffer())
            .range(vk::WHOLE_SIZE)];
        push_write(
            vk::WriteDescriptorSet::default()
                .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                .buffer_info(&descriptor_buffer_infos)
                .dst_set(descriptor_sets[0])
                .dst_binding(RaytracingDescriptorSetLayout::MATRICES_UNIFORM_BUFFER_BINDING),
        );

        // write meshes info uniform buffer to descriptor set
        let descriptor_buffer_infos = [vk::DescriptorBufferInfo::default()
            .buffer(shader_data.get_meshes_info_storage_buffer())
            .range(vk::WHOLE_SIZE)];
        push_write(
            vk::WriteDescriptorSet::default()
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .buffer_info(&descriptor_buffer_infos)
                .dst_set(descriptor_sets[0])
                .dst_binding(RaytracingDescriptorSetLayout::MESHES_INFO_STORAGE_BUFFER_BINDING),
        );

        // write samplers to descriptor set
        assert_eq!(
            shader_data.get_textures().len(),
            RaytracingDescriptorSetLayout::NUMBER_OF_SAMPLERS as usize
        );

        let descriptor_sampler_infos = shader_data
            .get_textures()
            .iter()
            .map(|(sampler, image_view)| {
                vk::DescriptorImageInfo::default()
                    .sampler(*sampler)
                    .image_view(*image_view)
                    .image_layout(vk::ImageLayout::GENERAL)
            })
            .collect::<Vec<_>>();

        push_write(
            vk::WriteDescriptorSet::default()
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .image_info(&descriptor_sampler_infos)
                .dst_set(descriptor_sets[0])
                .dst_binding(RaytracingDescriptorSetLayout::SAMPLERS_BINDING),
        );

        unsafe { device.update_descriptor_sets(&descriptor_writes, &[]) };

        Ok(Self {
            core,
            descriptor_sets,
            descriptor_pool,
        })
    }

    /*
    pub fn update_accumulation_images(
        &self,
        images: &[vulkan_abstraction::Image; 2],
        sampler: vk::Sampler,
    ) {
        let device = self.core.device().inner();

        // 1. Prepare Infos for History (Binding 5) - Read Only
        // We act as if we are binding an array of 2 textures.
        let history_infos = images
            .iter()
            .map(|img| {
                vk::DescriptorImageInfo::default()
                    .sampler(sampler)
                    .image_view(img.image_view())
                    .image_layout(vk::ImageLayout::GENERAL)
            })
            .collect::<Vec<_>>();

        // 2. Prepare Infos for Accumulation (Binding 6) - General/Write
        // We act as if we are binding an array of 2 storage images.
        let accum_infos = images
            .iter()
            .map(|img| {
                vk::DescriptorImageInfo::default()
                    .image_view(img.image_view())
                    .image_layout(vk::ImageLayout::GENERAL)
            })
            .collect::<Vec<_>>();

        let mut writes = Vec::new();


        for set in &self.descriptor_sets {
            // Write Binding 5 (Array of 2 Textures)
            writes.push(
                vk::WriteDescriptorSet::default()
                    .dst_set(*set)
                    .dst_binding(RaytracingDescriptorSetLayout::HISTORY_BINDING)
                    .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                    .dst_array_element(0) // Start at index 0 of the array
                    .image_info(&history_infos), // Contains 2 items
            );


            // Write Binding 6 (Array of 2 Images)
            writes.push(
                vk::WriteDescriptorSet::default()
                    .dst_set(*set)
                    .dst_binding(RaytracingDescriptorSetLayout::ACCUMULATION_BINDING)
                    .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                    .dst_array_element(0) // Start at index 0 of the array
                    .image_info(&accum_infos), // Contains 2 items
            );
        }

        unsafe {
            device.update_descriptor_sets(&writes, &[]);
        }
    }

     */

    pub fn inner(&self) -> &[vk::DescriptorSet] {
        &self.descriptor_sets
    }
}

impl Drop for RaytracingDescriptorSets {
    fn drop(&mut self) {
        //only do this if you set VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT
        //unsafe { self.core.device().free_descriptor_sets(self.descriptor_pool, &self.descriptor_sets) }.unwrap();

        unsafe { self.core.device().inner().destroy_descriptor_pool(self.descriptor_pool, None) };
    }
}



/// Wrapper struct for the denoise pass descriptor set
pub(crate) struct DenoiseDescriptorSets {
    descriptor_sets: Vec<vk::DescriptorSet>,
    descriptor_pool: vk::DescriptorPool,
    core: Rc<vulkan_abstraction::Core>,
}

impl DenoiseDescriptorSets {
    pub fn new(
        core: Rc<vulkan_abstraction::Core>,
        layout: &DenoiseDescriptorSetLayout,
        input_image: &vulkan_abstraction::Image,
        output_image: &vulkan_abstraction::Image,
        depth_image: &vulkan_abstraction::Image,
        normal_image: &vulkan_abstraction::Image,
        motion_vector_image: &vulkan_abstraction::Image,
        history_images: &[vulkan_abstraction::Image; 2],
        accumulation_images: &[vulkan_abstraction::Image; 2],
        history_sampler: vk::Sampler,
    ) -> SrResult<Self> {
        let device = core.device().inner();

        let pool_sizes = [
            vk::DescriptorPoolSize::default()
                .ty(vk::DescriptorType::STORAGE_IMAGE)
                .descriptor_count(7),
            vk::DescriptorPoolSize::default()
                .ty(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .descriptor_count(2),
        ];

        let pool_info = vk::DescriptorPoolCreateInfo::default()
            .pool_sizes(&pool_sizes)
            .max_sets(1);

        let descriptor_pool = unsafe { device.create_descriptor_pool(&pool_info, None)? };

        let set_layouts = [layout.inner()];

        let alloc_info = vk::DescriptorSetAllocateInfo::default()
            .descriptor_pool(descriptor_pool)
            .set_layouts(&set_layouts);

        let descriptor_sets = unsafe { device.allocate_descriptor_sets(&alloc_info)? };
        let set = descriptor_sets[0];

        let create_info = |img: &vulkan_abstraction::Image| {
            vk::DescriptorImageInfo::default()
                .image_layout(vk::ImageLayout::GENERAL)
                .image_view(img.image_view())
        };

        let input_info = create_info(input_image);
        let output_info = create_info(output_image);
        let depth_info = create_info(depth_image);
        let normal_info = create_info(normal_image);
        let mv_info = create_info(motion_vector_image);

        // 3. Create Image Infos for the Ping-Pong Arrays
        let history_infos = [
            vk::DescriptorImageInfo::default()
                .image_layout(vk::ImageLayout::GENERAL)
                .image_view(history_images[0].image_view())
                .sampler(history_sampler),
            vk::DescriptorImageInfo::default()
                .image_layout(vk::ImageLayout::GENERAL)
                .image_view(history_images[1].image_view())
                .sampler(history_sampler),
        ];

        let accumulation_infos = [
            create_info(&accumulation_images[0]),
            create_info(&accumulation_images[1]),
        ];

        // 4. Write to the Descriptor Set
        let writes = [
            // Binding 0: Raw Color
            vk::WriteDescriptorSet::default()
                .dst_set(set)
                .dst_binding(DenoiseDescriptorSetLayout::RAW_COLOR_BINDING)
                .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                .image_info(std::slice::from_ref(&input_info)),

            // Binding 1: Output
            vk::WriteDescriptorSet::default()
                .dst_set(set)
                .dst_binding(DenoiseDescriptorSetLayout::OUTPUT_IMAGE_BINDING)
                .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                .image_info(std::slice::from_ref(&output_info)),

            // Binding 2: Depth
            vk::WriteDescriptorSet::default()
                .dst_set(set)
                .dst_binding(DenoiseDescriptorSetLayout::DEPTH_BINDING)
                .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                .image_info(std::slice::from_ref(&depth_info)),

            // Binding 3: Normal
            vk::WriteDescriptorSet::default()
                .dst_set(set)
                .dst_binding(DenoiseDescriptorSetLayout::NORMAL_BINDING)
                .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                .image_info(std::slice::from_ref(&normal_info)),

            // Binding 4: Motion Vectors
            vk::WriteDescriptorSet::default()
                .dst_set(set)
                .dst_binding(DenoiseDescriptorSetLayout::MOTION_VECTOR_BINDING)
                .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                .image_info(std::slice::from_ref(&mv_info)),

            // Binding 5: History (Array of 2, Samplers)
            vk::WriteDescriptorSet::default()
                .dst_set(set)
                .dst_binding(DenoiseDescriptorSetLayout::HISTORY_BINDING)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .image_info(&history_infos), // Pass the whole array!

            // Binding 6: Accumulation (Array of 2, Storage)
            vk::WriteDescriptorSet::default()
                .dst_set(set)
                .dst_binding(DenoiseDescriptorSetLayout::ACCUMULATION_BINDING)
                .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                .image_info(&accumulation_infos), // Pass the whole array!
        ];

        unsafe { device.update_descriptor_sets(&writes, &[]) };

        Ok(Self {
            core,
            descriptor_sets,
            descriptor_pool,
        })
    }

    pub fn inner(&self) -> &[vk::DescriptorSet] {
        &self.descriptor_sets
    }
}

impl Drop for DenoiseDescriptorSets {
    fn drop(&mut self) {
        unsafe {
            self.core.device().inner().destroy_descriptor_pool(self.descriptor_pool, None);
        }
    }
}
