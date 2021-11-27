#include "vuk/Context.hpp"
#include "vuk/RenderGraph.hpp"

/*
template<class T>
Result<NUnique<T>, AllocateException> allocate_unique_semaphores(NAllocator allocator, SourceLocationAtFrame loc) {
	NUnique<T> semas(allocator);
	if (auto res = allocator.allocate_semaphores(*semas, loc); !res) {
		return { expected_error, res.error() };
	}
	return { expected_value, semas };
}


NUnique<std::array<VkSemaphore, 2>> semas = allocate_unique_semaphores<std::array<VkSemaphore, 2>>(allocator, VUK_HERE_AND_NOW());
if (!semas) {
	return { expected_error, semas.error() };
}
auto [present_rdy, render_complete] = *semas.value();
*/

namespace vuk {

	Result<void> execute_submit_and_present_to_one(PerThreadContext& ptc, NAllocator allocator, ExecutableRenderGraph&& rg, SwapchainRef swapchain) {
		NUnique<std::array<VkSemaphore, 2>> semas(allocator);
		VUK_DO_OR_RETURN(allocator.allocate_semaphores(*semas, VUK_HERE_AND_NOW()));
		auto [present_rdy, render_complete] = *semas;

		uint32_t image_index = (uint32_t)-1;
		VkResult acq_result = vkAcquireNextImageKHR(ptc.ctx.device, swapchain->swapchain, UINT64_MAX, present_rdy, VK_NULL_HANDLE, &image_index);
		if (acq_result != VK_SUCCESS) {
			VkSubmitInfo si{ .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO };
			si.commandBufferCount = 0;
			si.pCommandBuffers = nullptr;
			si.waitSemaphoreCount = 1;
			si.pWaitSemaphores = &present_rdy;
			VkPipelineStageFlags flags = (VkPipelineStageFlags)PipelineStageFlagBits::eTopOfPipe;
			si.pWaitDstStageMask = &flags;
			ptc.ctx.submit_graphics(si, VK_NULL_HANDLE);
			return { expected_error, PresentException{acq_result} };
		}

		std::vector<std::pair<SwapChainRef, size_t>> swapchains_with_indexes = { { swapchain, image_index } };

		auto cb = rg.execute(ptc, swapchains_with_indexes);

		VkSubmitInfo si{ .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO };
		si.commandBufferCount = 1;
		si.pCommandBuffers = &cb;
		si.pSignalSemaphores = &render_complete;
		si.signalSemaphoreCount = 1;
		si.waitSemaphoreCount = 1;
		si.pWaitSemaphores = &present_rdy;
		VkPipelineStageFlags flags = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
		si.pWaitDstStageMask = &flags;

		NUnique<VkFence> fence(allocator);
		VUK_DO_OR_RETURN(allocator.allocate_fences({ &*fence, 1 }, VUK_HERE_AND_NOW()));

		ptc.ctx.submit_graphics(si, *fence);

		VkPresentInfoKHR pi{ .sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR };
		pi.swapchainCount = 1;
		pi.pSwapchains = &swapchain->swapchain;
		pi.pImageIndices = &image_index;
		pi.waitSemaphoreCount = 1;
		pi.pWaitSemaphores = &render_complete;
		auto present_result = vkQueuePresentKHR(ptc.ctx.graphics_queue, &pi);
		if (present_result == VK_SUCCESS) {
			return { expected_value };
		} else {
			return { expected_error, PresentException{present_result} };
		}
	}

	Result<void> execute_submit_and_wait(PerThreadContext& ptc, NAllocator allocator, ExecutableRenderGraph&& rg) {
		auto cbuf = rg.execute(ptc, {});
		// get an unpooled fence
		NUnique<VkFence> fence(allocator);
		VUK_DO_OR_RETURN(allocator.allocate_fences({ &*fence, 1 }, VUK_HERE_AND_NOW()));
		VkSubmitInfo si{
		.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO
		};
		si.commandBufferCount = 1;
		si.pCommandBuffers = &cbuf;

		ptc.ctx.submit_graphics(si, *fence);
		vkWaitForFences(ptc.ctx.device, 1, &*fence, VK_TRUE, UINT64_MAX);
	}
}