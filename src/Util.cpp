#include "vuk/Context.hpp"
#include "vuk/RenderGraph.hpp"

namespace vuk {
	Result<void> execute_submit_and_present_to_one(NAllocator& allocator, ExecutableRenderGraph&& rg, SwapchainRef swapchain) {
		Context& ctx = allocator.get_context();
		Unique<std::array<VkSemaphore, 2>> semas(allocator);
		VUK_DO_OR_RETURN(allocator.allocate_semaphores(*semas, VUK_HERE_AND_NOW()));
		auto [present_rdy, render_complete] = *semas;

		uint32_t image_index = (uint32_t)-1;
		VkResult acq_result = vkAcquireNextImageKHR(ctx.device, swapchain->swapchain, UINT64_MAX, present_rdy, VK_NULL_HANDLE, &image_index);
		if (acq_result != VK_SUCCESS) {
			VkSubmitInfo si{ .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO };
			si.commandBufferCount = 0;
			si.pCommandBuffers = nullptr;
			si.waitSemaphoreCount = 1;
			si.pWaitSemaphores = &present_rdy;
			VkPipelineStageFlags flags = (VkPipelineStageFlags)PipelineStageFlagBits::eTopOfPipe;
			si.pWaitDstStageMask = &flags;
			ctx.submit_graphics(si, VK_NULL_HANDLE);
			return { expected_error, PresentException{acq_result} };
		}

		std::vector<std::pair<SwapChainRef, size_t>> swapchains_with_indexes = { { swapchain, image_index } };

		auto cb = rg.execute(ctx, allocator, swapchains_with_indexes);
		if (!cb) {
			return { expected_error, cb.error() };
		}
		auto& hl_cbuf = *cb;

		VkSubmitInfo si{ .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO };
		si.commandBufferCount = 1;
		si.pCommandBuffers = &hl_cbuf->command_buffer;
		si.pSignalSemaphores = &render_complete;
		si.signalSemaphoreCount = 1;
		si.waitSemaphoreCount = 1;
		si.pWaitSemaphores = &present_rdy;
		VkPipelineStageFlags flags = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
		si.pWaitDstStageMask = &flags;

		Unique<VkFence> fence(allocator);
		VUK_DO_OR_RETURN(allocator.allocate_fences({ &*fence, 1 }, VUK_HERE_AND_NOW()));

		ctx.submit_graphics(si, *fence);

		VkPresentInfoKHR pi{ .sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR };
		pi.swapchainCount = 1;
		pi.pSwapchains = &swapchain->swapchain;
		pi.pImageIndices = &image_index;
		pi.waitSemaphoreCount = 1;
		pi.pWaitSemaphores = &render_complete;
		auto present_result = vkQueuePresentKHR(ctx.graphics_queue, &pi);
		if (present_result == VK_SUCCESS) {
			return { expected_value };
		} else {
			return { expected_error, PresentException{present_result} };
		}
	}

	Result<void> execute_submit_and_wait(NAllocator& allocator, ExecutableRenderGraph&& rg) {
		Context& ctx = allocator.get_context();
		auto cb = rg.execute(ctx, allocator, {});
		if (!cb) {
			return { expected_error, cb.error() };
		}
		auto& hl_cbuf = *cb;
		Unique<VkFence> fence(allocator);
		VUK_DO_OR_RETURN(allocator.allocate_fences({ &*fence, 1 }, VUK_HERE_AND_NOW()));
		VkSubmitInfo si{ .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO };
		si.commandBufferCount = 1;
		si.pCommandBuffers = &hl_cbuf->command_buffer;

		ctx.submit_graphics(si, *fence);
		vkWaitForFences(ctx.device, 1, &*fence, VK_TRUE, UINT64_MAX);
		return { expected_value };
	}

	Unique<ImageView>::~Unique() noexcept {
		if (allocator && payload.payload != VK_NULL_HANDLE) {
			allocator->deallocate(payload);
		}
	}

	void Unique<ImageView>::reset(ImageView value) noexcept {
		if (payload != value) {
			if (allocator && payload != ImageView{}) {
				allocator->deallocate(std::move(payload));
			}
			payload = std::move(value);
		}
	}
}