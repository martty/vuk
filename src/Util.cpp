#include "vuk/Context.hpp"
#include "vuk/RenderGraph.hpp"
#include <vuk/FrameAllocator.hpp>

bool vuk::execute_submit_and_present_to_one(Context& ctx, ThreadLocalFrameAllocator& fa, ExecutableRenderGraph&& rg, Swapchain& swapchain) {
	auto present_rdy = fa.allocate_semaphore();
	uint32_t image_index = (uint32_t)-1;
	VkResult acq_result = vkAcquireNextImageKHR(ctx.device, swapchain.swapchain, UINT64_MAX, present_rdy, VK_NULL_HANDLE, &image_index);
	if (acq_result != VK_SUCCESS) {
		VkSubmitInfo si{ .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO };
		si.commandBufferCount = 0;
		si.pCommandBuffers = nullptr;
		si.waitSemaphoreCount = 1;
		si.pWaitSemaphores = &present_rdy;
		VkPipelineStageFlags flags = (VkPipelineStageFlags)vuk::PipelineStageFlagBits::eTopOfPipe;
		si.pWaitDstStageMask = &flags;
		ctx.submit_graphics(si, VK_NULL_HANDLE);
		return false;
	}

	auto render_complete = fa.allocate_semaphore();
	std::vector<std::pair<vuk::Swapchain*, size_t>> swapchains_with_indexes = { { &swapchain, image_index } };

	auto cbws = rg.execute(fa, swapchains_with_indexes);

	cbws.wait_semaphores.push_back(present_rdy);
	cbws.wait_values.push_back(0);
	cbws.wait_stages.push_back(VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT);

	VkSubmitInfo si{ .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO };
	si.commandBufferCount = (uint32_t)cbws.command_buffers.size();
	si.pCommandBuffers = cbws.command_buffers.data();
	VkTimelineSemaphoreSubmitInfo tssi{ .sType = VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO };

	si.pSignalSemaphores = &render_complete;
	si.signalSemaphoreCount = 1;
	si.pWaitSemaphores = cbws.wait_semaphores.data();
	si.waitSemaphoreCount = (uint32_t)cbws.wait_semaphores.size();
	si.pWaitDstStageMask = cbws.wait_stages.data();
	tssi.pWaitSemaphoreValues = cbws.wait_values.data();
	tssi.waitSemaphoreValueCount = (uint32_t)cbws.wait_semaphores.size();
	uint64_t v = 1;
	tssi.pSignalSemaphoreValues = &v;
	si.pNext = &tssi;

	auto fence = fa.allocate_fence();
	ctx.submit_graphics(si, fence);
	//printf("%d", image_index);
	VkPresentInfoKHR pi{ .sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR };
	pi.swapchainCount = 1;
	pi.pSwapchains = &swapchain.swapchain;
	pi.pImageIndices = &image_index;
	pi.waitSemaphoreCount = 1;
	pi.pWaitSemaphores = &render_complete;
	auto present_result = vkQueuePresentKHR(ctx.graphics_queue, &pi);
	return present_result == VK_SUCCESS;
}

void vuk::execute_submit_and_wait(Context& ctx, ThreadLocalFrameAllocator& fa, ExecutableRenderGraph&& rg) {
	auto cbws = rg.execute(fa, {});
	// get an unpooled fence
	VkFenceCreateInfo fci{ .sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO };
	VkFence fence;
	vkCreateFence(ctx.device, &fci, nullptr, &fence);
	VkSubmitInfo si{ .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO };
	si.commandBufferCount = cbws.command_buffers.size();
	si.pCommandBuffers = cbws.command_buffers.data();

	ctx.submit_graphics(si, fence);
	vkWaitForFences(ctx.device, 1, &fence, VK_TRUE, UINT64_MAX);
	vkDestroyFence(ctx.device, fence, nullptr);
}

