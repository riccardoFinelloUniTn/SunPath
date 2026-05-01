use crate::error::SrResult;
use crate::vulkan_abstraction::{AccelerationStructure, Buffer, CmdBuffer, Core, Image, RawBuffer};
use ash::vk;
use ash::vk::{CommandBuffer};
use std::collections::HashMap;
use std::sync::Arc;
use derive_builder::Builder;
use vk_sync_fork as vk_sync;

trait Resource {
    type Desc: ResourceDesc;
    fn borrow_resource(res: &AnyRenderResource) -> &Self;
}
trait ResourceDesc {}

pub(crate) struct RawResourceHandle {
    pub(crate) id: u32,
    pub(crate) version: u32,
}

pub enum AnyRenderResource {
    OwnedImage(Image),
    ImportedImage(Arc<Image>),
    OwnedBuffer(RawBuffer),
    ImportedBuffer(Arc<dyn Buffer>),
    ImportedRayTracingAcceleration(Arc<AccelerationStructure>),
}
#[derive(Clone,Copy)]
struct Handle {}

type DynRenderFn = dyn FnOnce(&mut CommandBuffer, &mut TransientResources) -> SrResult<()>; //TODO TransientResources here is intended to be a way to dereference the resources,but this implies it handles also external ones
#[derive(Clone,Copy)]
pub struct ResourceRef {
    handle: Handle,
    access_type : PassResourceAccessType
}
#[derive(Builder)]
#[builder(pattern = "owned")]
pub(crate) struct RenderPass {
    pub read: Vec<ResourceRef>,
    pub write: Vec<ResourceRef>,
    #[builder( setter(strip_option))]
    pub render_fn: Option<Box<DynRenderFn>>,
    #[builder(setter(skip))]
    pub name: String,
    #[builder(setter(skip))]
    idx: usize,
}




#[allow(dead_code)]
fn global_barrier(
    core: &Core,
    cb: &CmdBuffer,
    previous_accesses: &[vk_sync::AccessType],
    next_accesses: &[vk_sync::AccessType],
) {

    vk_sync::cmd::pipeline_barrier(
        core.device().inner(),
        cb.inner(),
        Some(vk_sync::GlobalBarrier {
            previous_accesses,
            next_accesses,
        }),
        &[],
        &[],
    );
}


pub struct TransientResources {
    //TODO this struct needs to be emptied after the next frame creation so that resources can be reused
}

#[derive(Clone)]

pub(crate) enum GraphResourceImportInfo {
    Image {
        resource: Arc<Image>,
        access_type: vk_sync::AccessType,
    },
    Buffer {
        resource: Arc<RawBuffer>,
        access_type: vk_sync::AccessType,
    },
    RayTracingAcceleration {
        resource: Arc<AccelerationStructure>,
        access_type: vk_sync::AccessType,
    },
    SwapchainImage,
}

pub struct ImageDesc {}

pub struct BufferDesc {}
pub struct RaytracingASDesc {}

pub enum GraphResourceDesc {
    Image(ImageDesc),
    Buffer(BufferDesc),
    RaytracingAS(RaytracingASDesc),
}

pub enum GraphResourceInfo {
    //this is description of what I need to allocate to satisfy the request pof the render pass
    Created(GraphResourceDesc),
    Imported(GraphResourceImportInfo),
}

struct PipelineCache {}

pub(crate) trait RenderGraphState {}
#[derive(Default)]
pub(crate) struct Setup {}
impl RenderGraphState for Setup {}

pub struct RenderGraph<K, State: RenderGraphState> {
    //TODO
    passes: Vec<RenderPass>,
    resources: Vec<GraphResourceInfo>,
    transient_resources: TransientResources,
    borrowed_resources: HashMap<K, ResourceRef>,
    frame_descriptor_set: vk::DescriptorSet, //for the internal data?
    state_data: State,
}

pub(crate) struct PassResourceRef {
    pub handle: RawResourceHandle,
    pub access: PassResourceAccessType,
}

#[derive(Copy, Clone)]
pub enum PassResourceAccessSyncType {
    AlwaysSync,
    SkipSyncIfSameAccessType,
}

#[derive(Copy, Clone)]
pub struct PassResourceAccessType {
    access_type: vk_sync::AccessType,
    sync_type: PassResourceAccessSyncType,
}



pub(crate) struct Render {}
impl<K> RenderGraph<K, Setup> {
    pub fn new() -> SrResult<Self> {
        Ok(RenderGraph {
            passes: vec![],
            resources: vec![],
            transient_resources: TransientResources {},
            borrowed_resources: HashMap::default(),
            frame_descriptor_set: Default::default(),
            state_data: Setup::default(),
        })
    }

    pub fn create_render_pass(&mut self, name: impl Into<String>)-> RenderPass  {
        RenderPass{
            read: vec![],
            write: vec![],
            render_fn: None,
            name: name.into(),
            idx: 0,
        }
    }

    pub fn add_render_pass(&mut self, pass : RenderPass){
        for written_res  in pass.write.iter(){
        }
    }
}

pub struct BuiltRenderGraph {
    //ready to execute
}
