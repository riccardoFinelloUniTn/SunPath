use std::{ffi::CStr, rc::Rc};
use ash::vk;
use crate::error::SrResult;
use crate::vulkan_abstraction::{self, Core};

const SHADER_ENTRY_POINT: &CStr = c"main";


pub trait ComputeTypeDef {
    type PushConstant;
    fn spirv_bytes() -> &'static [u8];
}

pub struct DenoisePass;
pub struct TemporalPass;

impl ComputeTypeDef for DenoisePass {
    type PushConstant = DenoisePushConstant;     //TODO change type
    fn spirv_bytes() -> &'static [u8] {
        include_bytes_align_as!(u32, concat!(env!("OUT_DIR"), "/denoise.spirv"))
    }
}

impl ComputeTypeDef for TemporalPass {
    type PushConstant = DenoisePushConstant;    //TODO change type
    fn spirv_bytes() -> &'static [u8] {
        include_bytes_align_as!(u32, concat!(env!("OUT_DIR"), "/temporal.spirv"))
    }
}


#[allow(dead_code)] // read by the gpu
#[repr(C, packed)]
#[derive(Debug)]
pub struct DenoisePushConstant {
    pub frame_count: u32,
}

pub struct ComputePipeline<T: ComputeTypeDef> {
    core: Rc<Core>,
    pipeline: vk::Pipeline,
    pipeline_layout: vk::PipelineLayout,
    descriptor_set_layout: vk::DescriptorSetLayout,
    _marker: T,
}

impl ComputePipeline<DenoisePass> {
    pub fn new(
        core: Rc<Core>,
        descriptor_set_layout: &vulkan_abstraction::DenoiseDescriptorSetLayout
    ) -> SrResult<Self> {
        let device = core.device().inner();

        //Shader Loading Helper
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

        // Load Denoise Shader
        let denoise_stage_create_info = make_shader_stage_create_info(
            vk::ShaderStageFlags::COMPUTE,
            include_bytes_align_as!(u32, concat!(env!("OUT_DIR"), "/denoise.spirv")),
        )?;

        // create push constants
        let push_constant_ranges = [vk::PushConstantRange::default()
            .stage_flags(vk::ShaderStageFlags::COMPUTE)
            .offset(0)
            .size(std::mem::size_of::<DenoisePushConstant>() as u32)];

        // Descriptor Set Layout
        let set_layouts = [descriptor_set_layout.inner()];

        let pipeline_layout_info = vk::PipelineLayoutCreateInfo::default()
            .set_layouts(&set_layouts)
            .push_constant_ranges(&push_constant_ranges);

        let pipeline_layout = unsafe { device.create_pipeline_layout(&pipeline_layout_info, None)? };

        // Create Compute Pipeline
        let pipeline_info = vk::ComputePipelineCreateInfo::default()
            .stage(denoise_stage_create_info)
            .layout(pipeline_layout);

        let pipelines = unsafe {
            device.create_compute_pipelines(vk::PipelineCache::null(), &[pipeline_info], None)
                .map_err(|(_, err)| err)?
        };
        let pipeline = pipelines[0];

        // Cleanup Shader Module
        unsafe {
            device.destroy_shader_module(denoise_stage_create_info.module, None);
        }

        Ok(Self {
            core,
            pipeline,
            pipeline_layout,
            descriptor_set_layout: descriptor_set_layout.inner(),       //TODO this could be redundant
            _marker: DenoisePass,
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

impl<T: ComputeTypeDef> Drop for ComputePipeline<T> {
    fn drop(&mut self) {
        let device = self.core.device().inner();
        unsafe {
            device.destroy_pipeline(self.pipeline, None);
            device.destroy_pipeline_layout(self.pipeline_layout, None);

        }
    }
}