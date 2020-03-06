#include "Context.hpp"
#include "RenderGraph.hpp"

void vuk::execute_submit_and_present_to_one(PerThreadContext& ptc, RenderGraph& rg, SwapchainRef swapchain) {
	auto render_complete = ptc.semaphore_pool.acquire(1)[0];
	auto present_rdy = ptc.semaphore_pool.acquire(1)[0];
	auto acq_result = ptc.ctx.device.acquireNextImageKHR(swapchain->swapchain, UINT64_MAX, present_rdy, vk::Fence{});
	auto index = acq_result.value;

	std::vector<std::pair<SwapChainRef, size_t>> swapchains_with_indexes = { { swapchain, index } };

	auto cb = rg.execute(ptc, swapchains_with_indexes);

	vk::SubmitInfo si;
	si.commandBufferCount = 1;
	si.pCommandBuffers = &cb;
	si.pSignalSemaphores = &render_complete;
	si.signalSemaphoreCount = 1;
	si.waitSemaphoreCount = 1;
	si.pWaitSemaphores = &present_rdy;
	vk::PipelineStageFlags flags = vk::PipelineStageFlagBits::eColorAttachmentOutput;
	si.pWaitDstStageMask = &flags;
	auto fence = ptc.fence_pool.acquire(1)[0];
	ptc.ctx.graphics_queue.submit(si, fence);
	vk::PresentInfoKHR pi;
	pi.swapchainCount = 1;
	pi.pSwapchains = &swapchain->swapchain;
	pi.pImageIndices = &acq_result.value;
	pi.waitSemaphoreCount = 1;
	pi.pWaitSemaphores = &render_complete;
	ptc.ctx.graphics_queue.presentKHR(pi);
}
