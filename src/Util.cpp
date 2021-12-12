#include "vuk/Context.hpp"
#include "vuk/RenderGraph.hpp"
#include "vuk/AllocatorHelpers.hpp"

namespace vuk {
	Result<void> execute_submit_and_present_to_one(Allocator& allocator, ExecutableRenderGraph&& rg, SwapchainRef swapchain) {
		Context& ctx = allocator.get_context();
		Unique<std::array<VkSemaphore, 2>> semas(allocator);
		VUK_DO_OR_RETURN(allocator.allocate_semaphores(*semas));
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
			VUK_DO_OR_RETURN(ctx.submit_graphics(si, VK_NULL_HANDLE));
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
		VUK_DO_OR_RETURN(allocator.allocate_fences({ &*fence, 1 }));

		VUK_DO_OR_RETURN(ctx.submit_graphics(si, *fence));

		VkPresentInfoKHR pi{ .sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR };
		pi.swapchainCount = 1;
		pi.pSwapchains = &swapchain->swapchain;
		pi.pImageIndices = &image_index;
		pi.waitSemaphoreCount = 1;
		pi.pWaitSemaphores = &render_complete;
		auto present_result = vkQueuePresentKHR(ctx.graphics_queue, &pi);
		if (present_result != VK_SUCCESS) {
			return { expected_error, PresentException{present_result} };
		}
		return { expected_value };
	}

	Result<void> execute_submit_and_wait(Allocator& allocator, ExecutableRenderGraph&& rg) {
		Context& ctx = allocator.get_context();
		auto cb = rg.execute(ctx, allocator, {});
		if (!cb) {
			return { expected_error, cb.error() };
		}
		auto& hl_cbuf = *cb;
		Unique<VkFence> fence(allocator);
		VUK_DO_OR_RETURN(allocator.allocate_fences({ &*fence, 1 }));
		VkSubmitInfo si{ .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO };
		si.commandBufferCount = 1;
		si.pCommandBuffers = &hl_cbuf->command_buffer;

		VUK_DO_OR_RETURN(ctx.submit_graphics(si, *fence));
		VkResult result = vkWaitForFences(ctx.device, 1, &*fence, VK_TRUE, UINT64_MAX);
		if (result != VK_SUCCESS) {
			return { expected_error, VkException{result} };
		}
		return { expected_value };
	}

	SampledImage make_sampled_image(ImageView iv, SamplerCreateInfo sci) {
		return { SampledImage::Global{ iv, sci, ImageLayout::eShaderReadOnlyOptimal } };
	}

	SampledImage make_sampled_image(Name n, SamplerCreateInfo sci) {
		return{ SampledImage::RenderGraphAttachment{ n, sci, {}, ImageLayout::eShaderReadOnlyOptimal } };
	}

	SampledImage make_sampled_image(Name n, ImageViewCreateInfo ivci, SamplerCreateInfo sci) {
		return { SampledImage::RenderGraphAttachment{ n, sci, ivci, ImageLayout::eShaderReadOnlyOptimal } };
	}

	Unique<ImageView>::~Unique() noexcept {
		if (allocator && payload.payload != VK_NULL_HANDLE) {
			deallocate(*allocator, payload);
		}
	}

	void Unique<ImageView>::reset(ImageView value) noexcept {
		if (payload != value) {
			if (allocator && payload != ImageView{}) {
				deallocate(*allocator, std::move(payload));
			}
			payload = std::move(value);
		}
	}
}