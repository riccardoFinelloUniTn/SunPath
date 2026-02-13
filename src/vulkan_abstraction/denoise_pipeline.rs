use std::{ffi::CStr, rc::Rc};
use ash::vk;
use crate::error::SrResult;
use crate::vulkan_abstraction::{self, Core};

const SHADER_ENTRY_POINT: &CStr = c"main";

pub struct DenoisePipeline {
    core: Rc<Core>,
    pipeline: vk::Pipeline,
    pipeline_layout: vk::PipelineLayout,
    descriptor_set_layout: vk::DescriptorSetLayout,
}

impl DenoisePipeline {
    pub fn new(core: Rc<Core>) -> SrResult<Self> {
        let device = core.device().inner();

        // --- 1. Shader Loading Helper (Your Syntax) ---
        let make_shader_stage_create_info =
            |stage: vk::ShaderStageFlags, spirv: &[u8]| -> SrResult<vk::PipelineShaderStageCreateInfo> {

                let spirv_u32 = bytemuck::cast_slice(spirv);

                let module_create_info = vk::ShaderModuleCreateInfo::default()
                    .flags(vk::ShaderModuleCreateFlags::empty())
                    .code(spirv_u32);

                let module = unsafe { device.create_shader_module(&module_create_info, None) }?;

                let stage_create_info = vk::PipelineShaderStageCreateInfo::default()
                    .name(SHADER_ENTRY_POINT)
                    .module(module)
                    .stage(stage);

                Ok(stage_create_info)
            };

        // --- 2. Load Denoise Shader ---
        // Make sure your build script compiles 'denoise.glsl' to this location!
        let denoise_stage_create_info = make_shader_stage_create_info(
            vk::ShaderStageFlags::COMPUTE,
            include_bytes_align_as!(u32, concat!(env!("OUT_DIR"), "/denoise.spirv")),
        )?;

        // --- 3. Descriptor Set Layout ---
        let layout_bindings = [
            // Binding 0: Input Image (Noisy)
            vk::DescriptorSetLayoutBinding::default()
                .binding(0)
                .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE),
            // Binding 1: Output Image (Clean)
            vk::DescriptorSetLayoutBinding::default()
                .binding(1)
                .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE),
        ];

        let layout_info = vk::DescriptorSetLayoutCreateInfo::default().bindings(&layout_bindings);
        let descriptor_set_layout = unsafe { device.create_descriptor_set_layout(&layout_info, None)? };

        // --- 4. Pipeline Layout ---
        let pipeline_layout_info = vk::PipelineLayoutCreateInfo::default()
            .set_layouts(std::slice::from_ref(&descriptor_set_layout));

        let pipeline_layout = unsafe { device.create_pipeline_layout(&pipeline_layout_info, None)? };

        // --- 5. Create Compute Pipeline ---
        let pipeline_info = vk::ComputePipelineCreateInfo::default()
            .stage(denoise_stage_create_info)
            .layout(pipeline_layout);

        let pipelines = unsafe {
            device.create_compute_pipelines(vk::PipelineCache::null(), &[pipeline_info], None)
                .map_err(|(_, err)| err)?
        };
        let pipeline = pipelines[0];

        // --- 6. Cleanup Shader Module ---
        // The pipeline now owns the shader logic, so we can destroy the module.
        unsafe {
            device.destroy_shader_module(denoise_stage_create_info.module, None);
        }

        Ok(Self {
            core,
            pipeline,
            pipeline_layout,
            descriptor_set_layout,
        })
    }

    // Getters for usage in the command buffer
    pub fn inner(&self) -> vk::Pipeline {
        self.pipeline
    }

    pub fn layout(&self) -> vk::PipelineLayout {
        self.pipeline_layout
    }

    pub fn descriptor_set_layout(&self) -> vk::DescriptorSetLayout {
        self.descriptor_set_layout
    }
}

impl Drop for DenoisePipeline {
    fn drop(&mut self) {
        let device = self.core.device().inner();
        unsafe {
            device.destroy_pipeline(self.pipeline, None);
            device.destroy_pipeline_layout(self.pipeline_layout, None);
            device.destroy_descriptor_set_layout(self.descriptor_set_layout, None);
        }
    }
}