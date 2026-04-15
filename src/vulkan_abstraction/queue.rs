use std::rc::Rc;

use crate::{error::*, vulkan_abstraction};
use ash::vk;

pub struct Queue {
    queue: vk::Queue,
    queue_family_index: u32,
    queue_index: u32,

    device: Rc<vulkan_abstraction::Device>,
}
impl Queue {
    pub fn new(device: Rc<vulkan_abstraction::Device>, queue_index: u32, queue_family_index: u32) -> SrResult<Self> {
        let queue = unsafe { device.inner().get_device_queue(queue_family_index, queue_index) };
        Ok(Self {
            queue,
            queue_family_index,
            queue_index,
            device,
        })
    }

    pub fn wait_idle(&self) -> SrResult<()> {
        unsafe { self.device.inner().queue_wait_idle(self.queue) }?;
        Ok(())
    }

    pub fn submit_async(
        &self,
        command_buffer: vk::CommandBuffer,
        wait_semaphores: &[vk::Semaphore],
        wait_dst_stages: &[vk::PipelineStageFlags],
        signal_semaphores: &[vk::Semaphore],
        signal_fence: vk::Fence,
    ) -> SrResult<()> {
        // NOTE: consider using VkQueueSubmit2 from the extension VK_KHR_synchronization2 which adds more dst stages (VkPipelineStageFlags2) like BLIT
        if cfg!(debug_assertions) && wait_semaphores.len() != wait_dst_stages.len() {
            return Err(SrError::new_custom(
                "Incorrect parameters to Queue::submit_async: wait_semaphores.len() != wait_dst_stages.len()".to_string(),
            ));
        }

        let command_buffers = [command_buffer];
        let submit_info = vk::SubmitInfo::default()
            .wait_semaphores(wait_semaphores)
            .wait_dst_stage_mask(&wait_dst_stages)
            .command_buffers(&command_buffers)
            .signal_semaphores(signal_semaphores);

        unsafe { self.device.inner().queue_submit(self.queue, &[submit_info], signal_fence) }?;

        Ok(())
    }

    pub fn submit_sync(&self, command_buffer: vk::CommandBuffer) -> SrResult<()> {
        let command_buffers = [command_buffer];
        let submit_info = vk::SubmitInfo::default()
            .wait_semaphores(&[])
            .wait_dst_stage_mask(&[])
            .command_buffers(&command_buffers)
            .signal_semaphores(&[]);

        let mut fence = vulkan_abstraction::Fence::new_unsignaled(Rc::clone(&self.device))?;

        unsafe { self.device.inner().queue_submit(self.queue, &[submit_info], fence.submit()?) }?;
        fence.wait()?;

        Ok(())
    }

    pub fn queue_family_index(&self) -> u32 {
        self.queue_family_index
    }


    pub fn queue_index(&self) -> u32 {
        self.queue_index
    }

    #[allow(dead_code)]
    pub fn inner(&self) -> vk::Queue {
        self.queue
    }
}
