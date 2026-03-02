use std::{ffi::CStr, rc::Rc};
use std::marker::PhantomData;
use ash::vk;
use crate::error::SrResult;
use crate::vulkan_abstraction::{self, Core, DenoiseDescriptorSetLayout, PostProcessDescriptorSetLayout};
use crate::vulkan_abstraction::TemporalAccumulationDescriptorSetLayout;

const SHADER_ENTRY_POINT: &CStr = c"main";


pub trait ComputeTypeDef {
    type PushConstant;
    type DescriptorSetLayout;
    fn spirv_bytes() -> &'static [u8];
}

pub struct DenoisePass;
pub struct TemporalPass;
pub struct PostprocessPass;

impl ComputeTypeDef for DenoisePass {
    type PushConstant = DenoisePushConstant;
    type DescriptorSetLayout = DenoiseDescriptorSetLayout;
    fn spirv_bytes() -> &'static [u8] {
        include_bytes_align_as!(u32, concat!(env!("OUT_DIR"), "/denoise.spirv"))
    }
}

impl ComputeTypeDef for TemporalPass {
    type PushConstant = TemporalAccumulationPushConstant;
    type DescriptorSetLayout = TemporalAccumulationDescriptorSetLayout;
    fn spirv_bytes() -> &'static [u8] {
        include_bytes_align_as!(u32, concat!(env!("OUT_DIR"), "/temporal_accumulation.spirv"))
    }
}

impl ComputeTypeDef for PostprocessPass {
    type PushConstant = PostprocessPushConstant;
    type DescriptorSetLayout = PostProcessDescriptorSetLayout;

    fn spirv_bytes() -> &'static [u8] {
        include_bytes_align_as!(u32, concat!(env!("OUT_DIR"), "/postprocess.spirv"))
    }
}

///Push Constant for the denoiser pass.
/// Frame count is self explicative.
/// Step width references the distance between each pixel used as a sample during the a-trous filtering.
#[allow(dead_code)] // read by the gpu
#[repr(C, packed)]
#[derive(Debug)]
pub struct DenoisePushConstant {
    pub frame_count: u32,
    pub step_width: u32,
}

#[allow(dead_code)] // read by the gpu
#[repr(C, packed)]
#[derive(Debug)]
pub struct TemporalAccumulationPushConstant {
    pub frame_count: u32,
}

#[allow(dead_code)] // read by the gpu
#[repr(C, packed)]
#[derive(Debug)]
pub struct PostprocessPushConstant;

pub struct ComputePipeline<T: ComputeTypeDef> {
    core: Rc<Core>,
    pipeline: vk::Pipeline,
    pipeline_layout: vk::PipelineLayout,
    descriptor_set_layout: vk::DescriptorSetLayout,
    _marker: PhantomData<T>,
}

impl<T:ComputeTypeDef> ComputePipeline<T> {
    pub fn new(
        core: Rc<Core>,
        descriptor_set_layout: vk::DescriptorSetLayout,
    ) -> SrResult<Self> {
        let device = core.device().inner();

        // 1. Get the SPIR-V bytes from the trait implementation
        let spirv_bytes = T::spirv_bytes();
        let spirv_u32 = bytemuck::cast_slice(spirv_bytes);

        // 2. Create the Shader Module
        let module_create_info = vk::ShaderModuleCreateInfo::default()
            .code(spirv_u32);

        let shader_module = unsafe { device.create_shader_module(&module_create_info, None) }?;

        // 3. Set up the stage info
        let shader_stage_create_info = vk::PipelineShaderStageCreateInfo::default()
            .name(SHADER_ENTRY_POINT)
            .module(shader_module)
            .stage(vk::ShaderStageFlags::COMPUTE);

        // 4. Use the generic PushConstant type for size
        let size = std::mem::size_of::<T::PushConstant>() as u32;

        // 1. Only create the range if the size is actually greater than 0
        let push_constant_ranges = if size > 0 {
            vec![vk::PushConstantRange::default()
                .stage_flags(vk::ShaderStageFlags::COMPUTE)
                .offset(0)
                .size(size)]
        } else {
            // If it's a ZST, we provide an empty Vec
            Vec::new()
        };

        let set_layouts = [descriptor_set_layout];

        let pipeline_layout_info = vk::PipelineLayoutCreateInfo::default()
            .set_layouts(&set_layouts)
            .push_constant_ranges(&push_constant_ranges);

        let pipeline_layout = unsafe { device.create_pipeline_layout(&pipeline_layout_info, None)? };

        // 5. Create the Pipeline
        let pipeline_info = vk::ComputePipelineCreateInfo::default()
            .stage(shader_stage_create_info)
            .layout(pipeline_layout);

        let pipelines = unsafe {
            device.create_compute_pipelines(vk::PipelineCache::null(), &[pipeline_info], None)
                .map_err(|(_, err)| {
                    // Clean up layout and module if creation fails
                    device.destroy_pipeline_layout(pipeline_layout, None);
                    device.destroy_shader_module(shader_module, None);
                    err
                })?
        };
        let pipeline = pipelines[0];

        // 6. Cleanup Shader Module (it is no longer needed once the pipeline is created)
        unsafe {
            device.destroy_shader_module(shader_module, None);
        }

        Ok(Self {
            core,
            pipeline,
            pipeline_layout,
            descriptor_set_layout,
            _marker: PhantomData,
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