#include "vuk/AllocatorHelpers.hpp"
#include "vuk/Context.hpp"
#include "vuk/RenderGraph.hpp"

namespace vuk {
	Result<void> link_execute_submit(Allocator& allocator, std::span<std::pair<Allocator*, RenderGraph*>> rgs) {
		std::vector<ExecutableRenderGraph> ergs;
		std::vector<std::pair<Allocator*, ExecutableRenderGraph*>> ptrvec;
		ergs.reserve(rgs.size());
		for (auto& [alloc, rg] : rgs) {
			ergs.emplace_back(std::move(*rg).link(alloc->get_context(), {}));
			ptrvec.emplace_back(alloc, &ergs.back());
		}

		return execute_submit(allocator, std::span(ptrvec), {}, {}, {});
	}

	Result<std::vector<SubmitBundle>> execute(std::span<std::pair<Allocator*, ExecutableRenderGraph*>> ergs,
	                                          std::vector<std::pair<SwapchainRef, size_t>> swapchains_with_indexes) {
		std::vector<SubmitBundle> bundles;
		for (auto& [alloc, rg] : ergs) {
			auto sbundle = rg->execute(*alloc, swapchains_with_indexes);
			if (!sbundle) {
				return { expected_error, sbundle.error() };
			}
			bool has_waits = false;
			for (auto& batch : sbundle->batches) {
				for (auto& s : batch.submits) {
					if (s.relative_waits.size() > 0) {
						has_waits = true;
					}
				}
			}
			// in the case where there are no waits in the entire bundle, we can merge all the submits together
			if (!has_waits && bundles.size() > 0) {
				auto& last = bundles.back();
				for (auto& batch : sbundle->batches) {
					auto tgt_domain = batch.domain;
					auto it = std::find_if(last.batches.begin(), last.batches.end(), [=](auto& batch) { return batch.domain == tgt_domain; });
					if (it != last.batches.end()) {
						it->submits.insert(it->submits.end(), batch.submits.begin(), batch.submits.end());
					} else {
						last.batches.emplace_back(batch);
					}
				}
			} else {
				bundles.push_back(*sbundle);
			}
		}
		return { expected_value, bundles };
	}

	Result<void> submit(Allocator& allocator, SubmitBundle bundle, VkSemaphore present_rdy, VkSemaphore render_complete) {
		Context& ctx = allocator.get_context();

		vuk::DomainFlags used_domains;
		for (auto& batch : bundle.batches) {
			used_domains |= batch.domain;
		}

		std::array<uint64_t, 3> queue_progress_references;
		std::unique_lock<std::mutex> gfx_lock;
		if (used_domains & DomainFlagBits::eGraphicsQueue) {
			queue_progress_references[ctx.domain_to_queue_index(DomainFlagBits::eGraphicsQueue)] = *ctx.graphics_queue->submit_sync.value;
			gfx_lock = std::unique_lock{ ctx.graphics_queue->queue_lock };
		}
		std::unique_lock<std::mutex> compute_lock;
		if (used_domains & DomainFlagBits::eComputeQueue) {
			queue_progress_references[ctx.domain_to_queue_index(DomainFlagBits::eComputeQueue)] = *ctx.compute_queue->submit_sync.value;
			compute_lock = std::unique_lock{ ctx.compute_queue->queue_lock };
		}
		std::unique_lock<std::mutex> transfer_lock;
		if (used_domains & DomainFlagBits::eTransferQueue) {
			queue_progress_references[ctx.domain_to_queue_index(DomainFlagBits::eTransferQueue)] = *ctx.transfer_queue->submit_sync.value;
			transfer_lock = std::unique_lock{ ctx.transfer_queue->queue_lock };
		}

		if (bundle.batches.size() > 1) {
			std::swap(bundle.batches[0], bundle.batches[1]); // FIXME: silence some false positive validation
		}
		for (SubmitBatch& batch : bundle.batches) {
			auto domain = batch.domain;
			Queue& queue = ctx.domain_to_queue(domain);
			Unique<VkFence> fence(allocator);
			VUK_DO_OR_RETURN(allocator.allocate_fences({ &*fence, 1 }));

			uint64_t num_cbufs = 0;
			uint64_t num_waits = 1; // 1 extra for present_rdy
			for (uint64_t i = 0; i < batch.submits.size(); i++) {
				SubmitInfo& submit_info = batch.submits[i];
				num_cbufs += submit_info.command_buffers.size();
				num_waits += submit_info.relative_waits.size();
			}

			std::vector<VkSubmitInfo2KHR> sis;
			std::vector<VkCommandBufferSubmitInfoKHR> cbufsis;
			cbufsis.reserve(num_cbufs);
			std::vector<VkSemaphoreSubmitInfoKHR> wait_semas;
			wait_semas.reserve(num_waits);
			std::vector<VkSemaphoreSubmitInfoKHR> signal_semas;
			signal_semas.reserve(batch.submits.size() + 1); // 1 extra for render_complete

			for (uint64_t i = 0; i < batch.submits.size(); i++) {
				SubmitInfo& submit_info = batch.submits[i];

				for (auto& fut : submit_info.future_signals) {
					fut->status = FutureBase::Status::eSubmitted;
				}

				if (submit_info.command_buffers.size() == 0) {
					continue;
				}

				for (uint64_t i = 0; i < submit_info.command_buffers.size(); i++) {
					cbufsis.emplace_back(
					    VkCommandBufferSubmitInfoKHR{ .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO_KHR, .commandBuffer = submit_info.command_buffers[i] });
				}

				uint32_t wait_sema_count = 0;
				for (auto& w : submit_info.relative_waits) {
					VkSemaphoreSubmitInfoKHR ssi{ VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO_KHR };
					auto& wait_queue = ctx.domain_to_queue(w.first).submit_sync;
					ssi.semaphore = wait_queue.semaphore;
					ssi.value = queue_progress_references[ctx.domain_to_queue_index(w.first)] + w.second;
					ssi.stageMask = (VkPipelineStageFlagBits2KHR)PipelineStageFlagBits::eAllCommands;
					wait_semas.emplace_back(ssi);
					wait_sema_count++;
				}
				if (domain == DomainFlagBits::eGraphicsQueue && i == 0 && present_rdy != VK_NULL_HANDLE) { // TODO: for first cbuf only that refs the swapchain attment
					VkSemaphoreSubmitInfoKHR ssi{ VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO_KHR };
					ssi.semaphore = present_rdy;
					ssi.stageMask = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT_KHR;
					// wait_semas.emplace_back(ssi);
					// wait_sema_count++;
				}

				VkSemaphoreSubmitInfoKHR ssi{ VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO_KHR };
				ssi.semaphore = queue.submit_sync.semaphore;
				ssi.value = ++(*queue.submit_sync.value);

				ssi.stageMask = (VkPipelineStageFlagBits2KHR)PipelineStageFlagBits::eAllCommands;

				for (auto& fut : submit_info.future_signals) {
					fut->status = FutureBase::Status::eSubmitted;
					fut->initial_domain = domain;
					fut->initial_visibility = ssi.value;
				}

				uint32_t signal_sema_count = 1;
				signal_semas.emplace_back(ssi);
				if (domain == DomainFlagBits::eGraphicsQueue && i == batch.submits.size() - 1 &&
				    render_complete != VK_NULL_HANDLE) { // TODO: for final cbuf only that refs the swapchain attment
					ssi.semaphore = render_complete;
					ssi.value = 0; // binary sema
					signal_semas.emplace_back(ssi);
					signal_sema_count++;
				}

				VkSubmitInfo2KHR& si = sis.emplace_back(VK_STRUCTURE_TYPE_SUBMIT_INFO_2_KHR);
				VkCommandBufferSubmitInfoKHR* p_cbuf_infos = &cbufsis.back() - (submit_info.command_buffers.size() - 1);
				VkSemaphoreSubmitInfoKHR* p_wait_semas = wait_sema_count > 0 ? &wait_semas.back() - (wait_sema_count - 1) : nullptr;
				VkSemaphoreSubmitInfoKHR* p_signal_semas = &signal_semas.back() - (signal_sema_count - 1);

				si.pWaitSemaphoreInfos = p_wait_semas;
				si.waitSemaphoreInfoCount = wait_sema_count;
				si.pCommandBufferInfos = p_cbuf_infos;
				si.commandBufferInfoCount = (uint32_t)submit_info.command_buffers.size();
				si.pSignalSemaphoreInfos = p_signal_semas;
				si.signalSemaphoreInfoCount = signal_sema_count;
			}

			VUK_DO_OR_RETURN(queue.submit(std::span{ sis }, *fence));
		}

		return { expected_value };
	}

	// assume rgs are independent - they don't reference eachother
	Result<void> execute_submit(Allocator& allocator,
	                            std::span<std::pair<Allocator*, ExecutableRenderGraph*>> rgs,
	                            std::vector<std::pair<SwapchainRef, size_t>> swapchains_with_indexes,
	                            VkSemaphore present_rdy,
	                            VkSemaphore render_complete) {
		auto bundles = execute(rgs, swapchains_with_indexes);
		if (!bundles) {
			return { expected_error, bundles.error() };
		}
		assert(bundles->size() < 2); // can't handle this yet
		for (auto& bundle : *bundles) {
			VUK_DO_OR_RETURN(submit(allocator, bundle, present_rdy, render_complete));
		}

		return { expected_value };
	}

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
			return { expected_error, PresentException{ acq_result } };
		}

		std::vector<std::pair<SwapchainRef, size_t>> swapchains_with_indexes = { { swapchain, image_index } };

		std::pair v = { &allocator, &rg };
		VUK_DO_OR_RETURN(execute_submit(allocator, std::span{ &v, 1 }, swapchains_with_indexes, present_rdy, render_complete));

		VkPresentInfoKHR pi{ .sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR };
		pi.swapchainCount = 1;
		pi.pSwapchains = &swapchain->swapchain;
		pi.pImageIndices = &image_index;
		pi.waitSemaphoreCount = 1;
		pi.pWaitSemaphores = &render_complete;
		auto present_result = vkQueuePresentKHR(ctx.graphics_queue->queue, &pi);
		if (present_result != VK_SUCCESS) {
			return { expected_error, PresentException{ present_result } };
		}

		return { expected_value };
	}

	Result<void> execute_submit_and_wait(Allocator& allocator, ExecutableRenderGraph&& rg) {
		Context& ctx = allocator.get_context();
		std::pair v = { &allocator, &rg };
		VUK_DO_OR_RETURN(execute_submit(allocator, std::span{ &v, 1 }, {}, {}, {}));
		ctx.wait_idle(); // TODO:
		return { expected_value };
	}

	SampledImage make_sampled_image(ImageView iv, SamplerCreateInfo sci) {
		return { SampledImage::Global{ iv, sci, ImageLayout::eShaderReadOnlyOptimal } };
	}

	SampledImage make_sampled_image(Name n, SamplerCreateInfo sci) {
		return { SampledImage::RenderGraphAttachment{ n, sci, {}, ImageLayout::eShaderReadOnlyOptimal } };
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
	Future<T>::Future(Allocator& alloc, struct RenderGraph& rg, Name output_binding, DomainFlags dst_domain) : control(std::make_unique<FutureBase>(alloc)), rg(&rg), output_binding(output_binding) {
		control->status = FutureBase::Status::eRenderGraphBound;
		this->rg->attach_out(output_binding, *this, dst_domain);
	}

	template<class T>
	Future<T>::Future(Allocator& alloc, std::unique_ptr<struct RenderGraph> org, Name output_binding, DomainFlags dst_domain) : control(std::make_unique<FutureBase>(alloc)), owned_rg(std::move(org)), rg(owned_rg.get()), output_binding(output_binding) {
		control->status = FutureBase::Status::eRenderGraphBound;
		rg->attach_out(output_binding, *this, dst_domain);
	}

	template<class T>
	Future<T>::Future(Allocator& alloc, T&& value) : control(std::make_unique<FutureBase>(alloc)) {
		control->get_result<T>() = std::move(value);
		control->status = FutureBase::Status::eHostAvailable;
	}

	template<class T>
	Result<T> Future<T>::get() {
		if (control->status == FutureBase::Status::eInputAttached || control->status == FutureBase::Status::eInitial) {
			return { expected_error }; // can't get result of future that has not been attached anything or has been attached into a rendergraph
		} else if (control->status == FutureBase::Status::eHostAvailable) {
			return { expected_value, control->get_result<T>() };
		} else if (control->status == FutureBase::Status::eSubmitted) {
			std::pair w = { (DomainFlags)control->initial_domain, control->initial_visibility };
			control->allocator->get_context().wait_for_domains(std::span{ &w, 1 });
			return { expected_value, control->get_result<T>() };
		} else {
			auto erg = std::move(*rg).link(control->allocator->get_context(), {});
			std::pair v = { control->allocator, &erg };
			VUK_DO_OR_RETURN(execute_submit(*control->allocator, std::span{ &v, 1 }, {}, {}, {}));
			std::pair w = { (DomainFlags)control->initial_domain, control->initial_visibility };
			control->allocator->get_context().wait_for_domains(std::span{ &w, 1 });
			control->status = FutureBase::Status::eHostAvailable;
			return { expected_value, control->get_result<T>() };
		}
	}

	template<class T>
	Result<void> Future<T>::submit() {
		if (control->status == FutureBase::Status::eInputAttached || control->status == FutureBase::Status::eInitial) {
			return { expected_error };
		} else if (control->status == FutureBase::Status::eHostAvailable || control->status == FutureBase::Status::eSubmitted) {
			return { expected_value }; // nothing to do
		} else {
			control->status = FutureBase::Status::eSubmitted;
			auto erg = std::move(*rg).link(control->allocator->get_context(), {});
			std::pair v = { control->allocator, &erg };
			VUK_DO_OR_RETURN(execute_submit(*control->allocator, std::span{ &v, 1 }, {}, {}, {}));
			return { expected_value };
		}
	}

	template struct Future<ImageAttachment>;
	template struct Future<Buffer>;
} // namespace vuk
