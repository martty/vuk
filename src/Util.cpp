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
			VUK_DO_OR_RETURN(ctx.submit_graphics(std::span{ &si, 1 }, VK_NULL_HANDLE));
			return { expected_error, PresentException{acq_result} };
		}

		std::vector<std::pair<SwapChainRef, size_t>> swapchains_with_indexes = { { swapchain, image_index } };

		auto sbundle = rg.execute(allocator, swapchains_with_indexes);
		if (!sbundle) {
			return { expected_error, sbundle.error() };
		}

		auto domain_to_queue_index = [](Domain domain) -> uint64_t {
			if (domain == Domain::eGraphicsQueue) {
				return 0;
			} else {
				return 2; // TODO:
			}
		};

		auto domain_to_queue = [](Context& ctx, Domain domain) -> Queue& {
			if (domain == Domain::eGraphicsQueue) {
				return *ctx.graphics_queue;
			} else {
				return *ctx.transfer_queue; // TODO:
			}
		};

		std::array<uint64_t, 3> queue_progress_references;
		queue_progress_references[domain_to_queue_index(Domain::eGraphicsQueue)] = *ctx.graphics_queue->submit_sync.value;
		queue_progress_references[domain_to_queue_index(Domain::eTransferQueue)] = *ctx.transfer_queue->submit_sync.value;

		if(sbundle->batches.size() > 1)
			std::swap(sbundle->batches[0], sbundle->batches[1]);

		for (auto& batch : sbundle->batches) {
			auto domain = batch.domain;
			// TODO: proper q sel
			Queue& queue = domain_to_queue(ctx, domain);

			for (auto& submit_info : batch.submits) {
				Unique<VkFence> fence(allocator);
				VUK_DO_OR_RETURN(allocator.allocate_fences({ &*fence, 1 }));
				auto& hl_cbuf = submit_info.command_buffers.back();

				VkSubmitInfo2KHR si{ .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO_2_KHR };
				si.commandBufferInfoCount = 1;
				VkCommandBufferSubmitInfoKHR cbufsi{ VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO_KHR };
				cbufsi.commandBuffer = hl_cbuf;
				si.pCommandBufferInfos = &cbufsi;

				std::vector<VkSemaphoreSubmitInfoKHR> wait_semas;
				for (auto& w : submit_info.relative_waits) {
					VkSemaphoreSubmitInfoKHR ssi{ VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO_KHR };
					auto& wait_queue = domain_to_queue(ctx, w.first).submit_sync;
					ssi.semaphore = wait_queue.semaphore;
					ssi.value = queue_progress_references[domain_to_queue_index(w.first)] + w.second;
					ssi.stageMask = (VkPipelineStageFlagBits2KHR)PipelineStageFlagBits::eAllCommands;
					wait_semas.emplace_back(ssi);
				}
				if (domain == Domain::eGraphicsQueue) { // TODO: for final? only
					VkSemaphoreSubmitInfoKHR ssi{ VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO_KHR };
					ssi.semaphore = present_rdy;
					ssi.stageMask = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT_KHR;
					//wait_semas.emplace_back(ssi);
				}
				si.pWaitSemaphoreInfos = wait_semas.data();
				si.waitSemaphoreInfoCount = wait_semas.size();

				std::vector<VkSemaphoreSubmitInfoKHR> signal_semas;
				VkSemaphoreSubmitInfoKHR ssi{ VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO_KHR };
				ssi.semaphore = queue.submit_sync.semaphore;
				*queue.submit_sync.value += 1; // TODO: this should be atomic or we should lock the queue sooner
				ssi.value = *queue.submit_sync.value;
				ssi.stageMask = (VkPipelineStageFlagBits2KHR)PipelineStageFlagBits::eAllCommands;
				signal_semas.emplace_back(ssi);
				if (domain == Domain::eGraphicsQueue) { // TODO: for final? only
					ssi.semaphore = render_complete;
					ssi.value = 0; // binary sema
					signal_semas.emplace_back(ssi);
				}
				si.pSignalSemaphoreInfos = signal_semas.data();
				si.signalSemaphoreInfoCount = signal_semas.size();
				queue.submit(std::span{ &si, 1 }, *fence);
				
				for (auto& bufsig : submit_info.buf_signals) {
					bufsig->status = Future<Buffer>::Status::eSubmitted;
				}

				for (auto& bufsig : submit_info.image_signals) {
					bufsig->status = Future<Image>::Status::eSubmitted;
				}
			}
		}

		VkPresentInfoKHR pi{ .sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR };
		pi.swapchainCount = 1;
		pi.pSwapchains = &swapchain->swapchain;
		pi.pImageIndices = &image_index;
		pi.waitSemaphoreCount = 1;
		pi.pWaitSemaphores = &render_complete;
		auto present_result = vkQueuePresentKHR(ctx.graphics_queue->queue, &pi);
		if (present_result != VK_SUCCESS) {
			return { expected_error, PresentException{present_result} };
		}
		return { expected_value };
	}

	Result<void> execute_submit_and_wait(Allocator& allocator, ExecutableRenderGraph&& rg) {
		Context& ctx = allocator.get_context();
		auto sbundle = rg.execute(allocator, {});
		if (!sbundle) {
			return { expected_error, sbundle.error() };
		}

		Unique<VkFence> fence(allocator);
		VUK_DO_OR_RETURN(allocator.allocate_fences({ &*fence, 1 }));
		for (auto& batch : sbundle->batches) {
			auto domain = batch.domain;
			for (auto& si : batch.submits) {
				auto& hl_cbuf = si.command_buffers.back();
				VkSubmitInfo si{ .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO };
				si.commandBufferCount = 1;
				si.pCommandBuffers = &hl_cbuf;
				if (domain == Domain::eGraphicsQueue) {
					VUK_DO_OR_RETURN(ctx.submit_graphics(std::span{ &si, 1 }, *fence));
				} else {
					VUK_DO_OR_RETURN(ctx.submit_transfer(std::span{ &si, 1 }, *fence));
				}
			}
		}
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

	FutureBase::FutureBase(Allocator& alloc) : allocator(&alloc) {}

	template<class T>
	Future<T>::Future(Allocator& alloc, struct RenderGraph& rg, Name output_binding) : FutureBase(alloc), rg(&rg), output_binding(output_binding) {
		status = Status::eRenderGraphBound;
	}

	template<class T>
	Future<T>::Future(Allocator& alloc, std::unique_ptr<struct RenderGraph> org, Name output_binding) : FutureBase(alloc), owned_rg(std::move(org)), rg(owned_rg.get()), output_binding(output_binding) {
		status = Status::eRenderGraphBound;
	}

	template<class T>
	Future<T>::Future(Allocator& alloc, T&& value) : FutureBase(alloc), result(std::move(value)) {
		status = Status::eHostAvailable;
	}

	template<>
	Result<Buffer> Future<Buffer>::get() {
		if (status == Status::eInputAttached || status == Status::eInitial) {
			return { expected_error };
		} else if (status == Status::eHostAvailable) {
			return { expected_value, result };
		} else if (status == Status::eSubmitted) {
			//allocator->get_context().wait_for_queues();
		} else {
			VUK_DO_OR_RETURN(execute_submit_and_wait(*allocator, std::move(*rg).link(allocator->get_context(), {})));
			status = Status::eHostAvailable;
			return { expected_value, result };
		}
	}

	template struct Future<Image>;
	template struct Future<Buffer>;
	template struct Future<BufferGPU>;
	template struct Future<BufferCrossDevice>;
}