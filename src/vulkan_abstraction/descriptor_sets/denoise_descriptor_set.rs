use std::rc::Rc;
use ash::vk;
use crate::error::SrResult;
use crate::vulkan_abstraction;

pub struct DenoiseDescriptorSetLayout {
    descriptor_set_layout: vk::DescriptorSetLayout,
    core: Rc<vulkan_abstraction::Core>,
}

impl DenoiseDescriptorSetLayout {
    pub const TEMPORAL_RESULT_BINDING: u32 = 0; // Input from Temporal Pass
    pub const DEPTH_BINDING: u32 = 1;           // Edge preservation
    pub const NORMAL_BINDING: u32 = 2;          // Edge preservation
    pub const FINAL_OUTPUT_BINDING: u32 = 3;    // Final image to the screen

    pub const NUMBER_OF_BINDINGS: usize = 4;

    pub fn new(core: Rc<vulkan_abstraction::Core>) -> SrResult<Self> {
        let device = core.device().inner();

        let bindings = [
            vk::DescriptorSetLayoutBinding::default()
                .binding(Self::TEMPORAL_RESULT_BINDING)
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
                .binding(Self::FINAL_OUTPUT_BINDING)
                .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                .descriptor_count(1)
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



/// Wrapper struct for the denoise pass descriptor set
pub struct DenoiseDescriptorSets {
    descriptor_sets: Vec<vk::DescriptorSet>,
    descriptor_pool: vk::DescriptorPool,
    core: Rc<vulkan_abstraction::Core>,
}

impl DenoiseDescriptorSets {
    pub fn new(
        core: Rc<vulkan_abstraction::Core>,
        layout: &DenoiseDescriptorSetLayout,
        temporal_results: &[vulkan_abstraction::Image; 2], // Pass the accumulation array from the Temporal pass here
        depth_image: &vulkan_abstraction::Image,
        normal_image: &vulkan_abstraction::Image,
        output_image: &vulkan_abstraction::Image, // The final output image
    ) -> SrResult<Self> {
        let device = core.device().inner();

        // 1. Pool Sizes (4 Storage Images per set * 2 Sets = 8 Storage Images total)
        let pool_sizes = [
            vk::DescriptorPoolSize::default()
                .ty(vk::DescriptorType::STORAGE_IMAGE)
                .descriptor_count(8),
        ];

        let pool_info = vk::DescriptorPoolCreateInfo::default()
            .pool_sizes(&pool_sizes)
            .max_sets(2); // Allocate 2 sets!

        let descriptor_pool = unsafe { device.create_descriptor_pool(&pool_info, None)? };

        // 2. Allocate Descriptor Sets
        let set_layouts = [layout.inner(), layout.inner()]; // Layout used twice

        let alloc_info = vk::DescriptorSetAllocateInfo::default()
            .descriptor_pool(descriptor_pool)
            .set_layouts(&set_layouts);

        let descriptor_sets = unsafe { device.allocate_descriptor_sets(&alloc_info)? };

        // Helper
        let create_info = |img: &vulkan_abstraction::Image| {
            vk::DescriptorImageInfo::default()
                .image_layout(vk::ImageLayout::GENERAL)
                .image_view(img.image_view())
        };

        // 3. Image Infos
        let depth_info = create_info(depth_image);
        let normal_info = create_info(normal_image);
        let output_info = create_info(output_image);

        let temporal_info_0 = create_info(&temporal_results[0]);
        let temporal_info_1 = create_info(&temporal_results[1]);

        let writes = [
            // ---------------- SET 0 ----------------
            vk::WriteDescriptorSet::default()
                .dst_set(descriptor_sets[0])
                .dst_binding(DenoiseDescriptorSetLayout::TEMPORAL_RESULT_BINDING)
                .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                .image_info(std::slice::from_ref(&temporal_info_0)),

            vk::WriteDescriptorSet::default()
                .dst_set(descriptor_sets[0])
                .dst_binding(DenoiseDescriptorSetLayout::DEPTH_BINDING)
                .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                .image_info(std::slice::from_ref(&depth_info)),

            vk::WriteDescriptorSet::default()
                .dst_set(descriptor_sets[0])
                .dst_binding(DenoiseDescriptorSetLayout::NORMAL_BINDING)
                .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                .image_info(std::slice::from_ref(&normal_info)),

            vk::WriteDescriptorSet::default()
                .dst_set(descriptor_sets[0])
                .dst_binding(DenoiseDescriptorSetLayout::FINAL_OUTPUT_BINDING)
                .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                .image_info(std::slice::from_ref(&output_info)),

            // ---------------- SET 1 ----------------
            vk::WriteDescriptorSet::default()
                .dst_set(descriptor_sets[1])
                .dst_binding(DenoiseDescriptorSetLayout::TEMPORAL_RESULT_BINDING)
                .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                .image_info(std::slice::from_ref(&temporal_info_1)),

            vk::WriteDescriptorSet::default()
                .dst_set(descriptor_sets[1])
                .dst_binding(DenoiseDescriptorSetLayout::DEPTH_BINDING)
                .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                .image_info(std::slice::from_ref(&depth_info)),

            vk::WriteDescriptorSet::default()
                .dst_set(descriptor_sets[1])
                .dst_binding(DenoiseDescriptorSetLayout::NORMAL_BINDING)
                .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                .image_info(std::slice::from_ref(&normal_info)),

            vk::WriteDescriptorSet::default()
                .dst_set(descriptor_sets[1])
                .dst_binding(DenoiseDescriptorSetLayout::FINAL_OUTPUT_BINDING)
                .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                .image_info(std::slice::from_ref(&output_info)),
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
