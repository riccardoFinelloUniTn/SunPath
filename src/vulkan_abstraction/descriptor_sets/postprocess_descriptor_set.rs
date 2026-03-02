use std::rc::Rc;
use ash::vk;
use crate::error::SrResult;
use crate::vulkan_abstraction;

pub struct PostprocessDescriptorSetLayout {
    descriptor_set_layout: vk::DescriptorSetLayout,
    core: Rc<vulkan_abstraction::Core>,
}

impl PostprocessDescriptorSetLayout {
    pub const INPUT_IMAGE: u32 = 0;
    pub const OUTPUT_IMAGE: u32 = 1;
    pub const NUMBER_OF_BINDINGS: usize = 2;

    pub fn new(core: Rc<vulkan_abstraction::Core>) -> SrResult<Self> {
        let device = core.device().inner();

        let bindings = [
            vk::DescriptorSetLayoutBinding::default()
                .binding(Self::INPUT_IMAGE)
                .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE),

            vk::DescriptorSetLayoutBinding::default()
                .binding(Self::OUTPUT_IMAGE)
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

pub struct PostProcessDescriptorSets {
    descriptor_sets: Vec<vk::DescriptorSet>,
    descriptor_pool: vk::DescriptorPool,

    core: Rc<vulkan_abstraction::Core>,
}

impl PostProcessDescriptorSets {
    pub fn new(
        core: Rc<vulkan_abstraction::Core>,
        layout: &PostprocessDescriptorSetLayout,
        input_image: &vulkan_abstraction::Image,
        output_image: &vulkan_abstraction::Image, // The final output image
    ) -> SrResult<Self> {
        let device = core.device().inner();

        let pool_sizes = [
            vk::DescriptorPoolSize::default()
                .ty(vk::DescriptorType::STORAGE_IMAGE)
                .descriptor_count(2),
        ];

        let pool_info = vk::DescriptorPoolCreateInfo::default()
            .pool_sizes(&pool_sizes)
            .max_sets(1);

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
        let input_info = create_info(input_image);
        let output_info = create_info(output_image);

        let writes = [

            vk::WriteDescriptorSet::default()
                .dst_binding(PostprocessDescriptorSetLayout::INPUT_IMAGE)
                .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                .image_info(std::slice::from_ref(&input_info)),

            vk::WriteDescriptorSet::default()
                .dst_binding(PostprocessDescriptorSetLayout::OUTPUT_IMAGE)
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

impl Drop for PostProcessDescriptorSets {
    fn drop(&mut self) {
        unsafe {
            self.core.device().inner().destroy_descriptor_pool(self.descriptor_pool, None);
        }
    }
}



