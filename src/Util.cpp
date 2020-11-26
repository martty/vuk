#include "vuk/Context.hpp"
#include "vuk/RenderGraph.hpp"

bool vuk::execute_submit_and_present_to_one(PerThreadContext& ptc, RenderGraph& rg, SwapchainRef swapchain) {
	auto present_rdy = ptc.acquire_semaphore();
	uint32_t image_index = (uint32_t)-1;
	VkResult acq_result = vkAcquireNextImageKHR(ptc.ctx.device, swapchain->swapchain, UINT64_MAX, present_rdy, VK_NULL_HANDLE, &image_index);
	if (acq_result != VK_SUCCESS) {
		VkSubmitInfo si { .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO };
		si.commandBufferCount = 0;
		si.pCommandBuffers = nullptr;
		si.waitSemaphoreCount = 1;
		si.pWaitSemaphores = &present_rdy;
		VkPipelineStageFlags flags = (VkPipelineStageFlags)vuk::PipelineStageFlagBits::eTopOfPipe;
		si.pWaitDstStageMask = &flags;
        ptc.ctx.submit_graphics(si, VK_NULL_HANDLE);
		return false;
	}

	auto render_complete = ptc.acquire_semaphore();
	std::vector<std::pair<SwapChainRef, size_t>> swapchains_with_indexes = { { swapchain, image_index } };

	auto cb = rg.execute(ptc, swapchains_with_indexes);

	VkSubmitInfo si { .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO };
	si.commandBufferCount = 1;
	si.pCommandBuffers = &cb;
	si.pSignalSemaphores = &render_complete;
	si.signalSemaphoreCount = 1;
	si.waitSemaphoreCount = 1;
	si.pWaitSemaphores = &present_rdy;
	VkPipelineStageFlags flags = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
	si.pWaitDstStageMask = &flags;
	auto fence = ptc.acquire_fence();
	ptc.ctx.submit_graphics(si, fence);

	VkPresentInfoKHR pi{ .sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR };
	pi.swapchainCount = 1;
	pi.pSwapchains = &swapchain->swapchain;
	pi.pImageIndices = &image_index;
	pi.waitSemaphoreCount = 1;
	pi.pWaitSemaphores = &render_complete;
	auto present_result = vkQueuePresentKHR(ptc.ctx.graphics_queue, &pi);
	return present_result == VK_SUCCESS;
}

void vuk::execute_submit_and_wait(PerThreadContext& ptc, RenderGraph& rg) {
	auto cbuf = rg.execute(ptc, {});
	// get an unpooled fence
	VkFenceCreateInfo fci{ .sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO };
	VkFence fence;
	vkCreateFence(ptc.ctx.device, &fci, nullptr, &fence);
	VkSubmitInfo si{ .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO };
	si.commandBufferCount = 1;
	si.pCommandBuffers = &cbuf;

	ptc.ctx.submit_graphics(si, fence);
	vkWaitForFences(ptc.ctx.device, 1, &fence, VK_TRUE, UINT64_MAX);
	vkDestroyFence(ptc.ctx.device, fence, nullptr);
}

