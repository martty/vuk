#include "vuk/Util.hpp"
#include "vuk/AllocatorHelpers.hpp"
#include "vuk/Context.hpp"
#include "vuk/Future.hpp"
#include "vuk/RenderGraph.hpp"
#include "vuk/SampledImage.hpp"

#include <atomic>
#include <doctest/doctest.h>
#include <mutex>
#include <sstream>
#include <utility>

namespace vuk {
	struct QueueImpl {
		// TODO: this recursive mutex should be changed to better queue handling
		std::recursive_mutex queue_lock;
		PFN_vkQueueSubmit queueSubmit;
		PFN_vkQueueSubmit2KHR queueSubmit2KHR;
		TimelineSemaphore submit_sync;
		VkQueue queue;
		std::array<std::atomic<uint64_t>, 3> last_device_waits;
		std::atomic<uint64_t> last_host_wait;
		uint32_t family_index;

		QueueImpl(PFN_vkQueueSubmit fn1, PFN_vkQueueSubmit2KHR fn2, VkQueue queue, uint32_t queue_family_index, TimelineSemaphore ts) :
		    queueSubmit(fn1),
		    queueSubmit2KHR(fn2),
		    submit_sync(ts),
		    queue(queue),
		    family_index(queue_family_index) {}
	};

	Queue::Queue(PFN_vkQueueSubmit fn1, PFN_vkQueueSubmit2KHR fn2, VkQueue queue, uint32_t queue_family_index, TimelineSemaphore ts) :
	    impl(new QueueImpl(fn1, fn2, queue, queue_family_index, ts)) {}
	Queue::~Queue() {
		delete impl;
	}

	Queue::Queue(Queue&& o) noexcept : impl(std::exchange(o.impl, nullptr)) {}

	Queue& Queue::operator=(Queue&& o) noexcept {
		impl = std::exchange(o.impl, nullptr);
		return *this;
	}

	TimelineSemaphore& Queue::get_submit_sync() {
		return impl->submit_sync;
	}

	std::recursive_mutex& Queue::get_queue_lock() {
		return impl->queue_lock;
	}

	Result<void> Queue::submit(std::span<VkSubmitInfo2KHR> sis, VkFence fence) {
		VkResult result = impl->queueSubmit2KHR(impl->queue, (uint32_t)sis.size(), sis.data(), fence);
		if (result != VK_SUCCESS) {
			return { expected_error, VkException{ result } };
		}
		return { expected_value };
	}

	Result<void> Queue::submit(std::span<VkSubmitInfo> sis, VkFence fence) {
		std::lock_guard _(impl->queue_lock);
		VkResult result = impl->queueSubmit(impl->queue, (uint32_t)sis.size(), sis.data(), fence);
		if (result != VK_SUCCESS) {
			return { expected_error, VkException{ result } };
		}
		return { expected_value };
	}

	Result<void> Context::wait_for_domains(std::span<std::pair<DomainFlags, uint64_t>> queue_waits) {
		std::array<uint32_t, 3> domain_to_sema_index = { ~0u, ~0u, ~0u };
		std::array<VkSemaphore, 3> queue_timeline_semaphores;
		std::array<uint64_t, 3> values = {};

		uint32_t count = 0;
		for (auto [domain, v] : queue_waits) {
			auto idx = domain_to_queue_index(domain);
			auto& mapping = domain_to_sema_index[idx];
			if (mapping == -1) {
				mapping = count++;
			}
			auto& q = domain_to_queue(domain);
			queue_timeline_semaphores[mapping] = q.impl->submit_sync.semaphore;
			values[mapping] = values[mapping] > v ? values[mapping] : v;
		}

		VkSemaphoreWaitInfo swi{ .sType = VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO };
		swi.pSemaphores = queue_timeline_semaphores.data();
		swi.pValues = values.data();
		swi.semaphoreCount = count;
		VkResult result = this->vkWaitSemaphores(device, &swi, UINT64_MAX);
		for (auto [domain, v] : queue_waits) {
			auto& q = domain_to_queue(domain);
			q.impl->last_host_wait.store(v);
		}
		if (result != VK_SUCCESS) {
			return { expected_error, VkException{ result } };
		}
		return { expected_value };
	}

	Result<void> link_execute_submit(Allocator& allocator, Compiler& compiler, std::span<std::shared_ptr<RenderGraph>> rgs) {
		auto erg = compiler.link(rgs, {});
		if (!erg) {
			return erg;
		}
		std::pair erg_and_alloc = std::pair{ &allocator, &*erg };
		return execute_submit(allocator, std::span(&erg_and_alloc, 1), {}, {}, {});
	}

	Result<std::vector<SubmitBundle>> execute(std::span<std::pair<Allocator*, ExecutableRenderGraph*>> ergs,
	                                          std::vector<std::pair<SwapchainRef, size_t>> swapchains_with_indexes) {
		std::vector<SubmitBundle> bundles;
		for (auto& [alloc, rg] : ergs) {
			auto sbundle = rg->execute(*alloc, swapchains_with_indexes);
			if (!sbundle) {
				return Result<std::vector<SubmitBundle>>(std::move(sbundle));
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

	std::string_view to_name(vuk::DomainFlagBits d) {
		switch (d) {
		case DomainFlagBits::eTransferQueue:
			return "Transfer";
		case DomainFlagBits::eGraphicsQueue:
			return "Graphics";
		case DomainFlagBits::eComputeQueue:
			return "Compute";
		default:
			return "Unknown";
		}
	}

	std::string to_dot(SubmitBundle& bundle) {
		std::stringstream ss;
		ss << "digraph {";
		for (auto& batch : bundle.batches) {
			ss << "subgraph cluster_" << to_name(batch.domain) << " {";
			char name = 'A';

			for (size_t i = 0; i < batch.submits.size(); i++) {
				ss << to_name(batch.domain)[0] << name << ";";
				name++;
			}
			ss << "}";
		}

		for (auto& batch : bundle.batches) {
			char name = 'A';

			for (auto& sub : batch.submits) {
				for (auto& wait : sub.relative_waits) {
					char dst_name = wait.second == 0 ? 'X' : 'A' + (char)wait.second - 1;
					ss << to_name(batch.domain)[0] << name << "->" << to_name(wait.first)[0] << dst_name << ";";
				}
				name++;
			}
		}

		ss << "}";
		return ss.str();
	}

	void flatten_transfer_and_compute_onto_graphics(SubmitBundle& bundle) {
		if (bundle.batches.empty()) {
			return;
		}
		auto domain_to_index = [](vuk::DomainFlagBits d) {
			switch (d) {
			case DomainFlagBits::eTransferQueue:
				return 2;
			case DomainFlagBits::eGraphicsQueue:
				return 0;
			case DomainFlagBits::eComputeQueue:
				return 1;
			default:
				assert(0);
				return 4;
			}
		};
		size_t num_submits = 0;
		for (auto& batch : bundle.batches) {
			num_submits += batch.submits.size();
		}
		SubmitBatch dst_batch{ .domain = DomainFlagBits::eGraphicsQueue };
		uint64_t progress[3] = {};
		while (true) {
			for (auto& batch : bundle.batches) {
				auto queue = (DomainFlagBits)(batch.domain & DomainFlagBits::eQueueMask).m_mask;
				auto our_id = domain_to_index(queue);
				for (size_t i = progress[our_id]; i < batch.submits.size(); i++) {
					auto b = batch.submits[i];
					bool all_waits_satisfied = true;
					// check if all waits can be satisfied for this submit
					for (auto& [queue, wait_id] : b.relative_waits) {
						auto q_id = domain_to_index((DomainFlagBits)(queue & DomainFlagBits::eQueueMask).m_mask);
						auto& progress_on_wait_queue = progress[q_id];
						if (progress_on_wait_queue < wait_id) {
							all_waits_satisfied = false;
							break;
						}
					}
					if (all_waits_satisfied) {
						if (!b.relative_waits.empty()) {
							b.relative_waits = { { DomainFlagBits::eGraphicsQueue, dst_batch.submits.size() } }; // collapse into a single wait
						}
						dst_batch.submits.emplace_back(b);
						progress[our_id]++; // retire this batch
					} else {
						// couldn't make progress
						// break here is not correct, because there might be multiple waits with the same rank
						// TODO: we need to break here anyways for unsorted - we need to sort
						break;
					}
				}
			}
			if (dst_batch.submits.size() == num_submits) { // we have moved all the submits to the dst_batch
				break;
			}
		}
		bundle.batches = { dst_batch };
	}

	TEST_CASE("testing flattening submit graphs") {
		{
			SubmitBundle empty{};
			auto before = to_dot(empty);
			flatten_transfer_and_compute_onto_graphics(empty);
			auto after = to_dot(empty);
			CHECK(before == after);
		}
		{
			// transfer : TD -> TC -> TB -> TA
			// everything moved to graphics
			SubmitBundle only_transfer{ .batches = { SubmitBatch{ .domain = vuk::DomainFlagBits::eTransferQueue,
				                                                    .submits = { { .relative_waits = {} },
				                                                                 { .relative_waits = { { vuk::DomainFlagBits::eTransferQueue, 1 } } },
				                                                                 { .relative_waits = { { vuk::DomainFlagBits::eTransferQueue, 2 } } },
				                                                                 { .relative_waits = { { vuk::DomainFlagBits::eTransferQueue, 3 } } } } } } };

			auto before = to_dot(only_transfer);
			flatten_transfer_and_compute_onto_graphics(only_transfer);
			auto after = to_dot(only_transfer);
			CHECK(after == "digraph {subgraph cluster_Graphics {GA;GB;GC;GD;}GB->GA;GC->GB;GD->GC;}");
		}
		{
			// transfer : TD  TC -> TB  TA
			//			   v  ^     v
			// graphics : GD->GC    GB->GA
			// flattens to
			// graphics : TD -> GD -> GC -> TC -> TB -> GB -> GA TA
			SubmitBundle two_queue{ .batches = { SubmitBatch{ .domain = vuk::DomainFlagBits::eTransferQueue,
				                                                .submits = { { .relative_waits = {} },
				                                                             { .relative_waits = { { vuk::DomainFlagBits::eGraphicsQueue, 2 } } },
				                                                             { .relative_waits = { { vuk::DomainFlagBits::eTransferQueue, 2 } } },
				                                                             { .relative_waits = { { vuk::DomainFlagBits::eGraphicsQueue, 4 } } } } },
				                                   SubmitBatch{ .domain = vuk::DomainFlagBits::eGraphicsQueue,
				                                                .submits = { { .relative_waits = {} },
				                                                             { .relative_waits = { { vuk::DomainFlagBits::eGraphicsQueue, 1 } } },
				                                                             { .relative_waits = { { vuk::DomainFlagBits::eTransferQueue, 3 } } },
				                                                             { .relative_waits = { { vuk::DomainFlagBits::eGraphicsQueue, 3 } } } } } } };

			auto before = to_dot(two_queue);
			flatten_transfer_and_compute_onto_graphics(two_queue);
			auto after = to_dot(two_queue);
			CHECK(after == "digraph {subgraph cluster_Graphics {GA;GB;GC;GD;GE;GF;GG;GH;}GC->GB;GD->GC;GE->GD;GF->GE;GG->GF;GH->GG;}");
		}
	}

	Result<void> submit(Allocator& allocator, SubmitBundle bundle, VkSemaphore present_rdy, VkSemaphore render_complete) {
		Context& ctx = allocator.get_context();

		vuk::DomainFlags used_domains;
		for (auto& batch : bundle.batches) {
			used_domains |= batch.domain;
		}

		std::array<uint64_t, 3> queue_progress_references;
		std::unique_lock<std::recursive_mutex> gfx_lock;
		if (used_domains & DomainFlagBits::eGraphicsQueue) {
			queue_progress_references[ctx.domain_to_queue_index(DomainFlagBits::eGraphicsQueue)] = *ctx.graphics_queue->impl->submit_sync.value;
			gfx_lock = std::unique_lock{ ctx.graphics_queue->impl->queue_lock };
		}
		std::unique_lock<std::recursive_mutex> compute_lock;
		if (used_domains & DomainFlagBits::eComputeQueue) {
			queue_progress_references[ctx.domain_to_queue_index(DomainFlagBits::eComputeQueue)] = *ctx.compute_queue->impl->submit_sync.value;
			compute_lock = std::unique_lock{ ctx.compute_queue->impl->queue_lock };
		}
		std::unique_lock<std::recursive_mutex> transfer_lock;
		if (used_domains & DomainFlagBits::eTransferQueue) {
			queue_progress_references[ctx.domain_to_queue_index(DomainFlagBits::eTransferQueue)] = *ctx.transfer_queue->impl->submit_sync.value;
			transfer_lock = std::unique_lock{ ctx.transfer_queue->impl->queue_lock };
		}
		bool needs_flatten = ((used_domains & DomainFlagBits::eTransferQueue) &&
		                      (ctx.domain_to_queue_index(DomainFlagBits::eTransferQueue) == ctx.domain_to_queue_index(DomainFlagBits::eGraphicsQueue) ||
		                       ctx.domain_to_queue_index(DomainFlagBits::eTransferQueue) == ctx.domain_to_queue_index(DomainFlagBits::eComputeQueue))) ||
		                     ((used_domains & DomainFlagBits::eComputeQueue) &&
		                      (ctx.domain_to_queue_index(DomainFlagBits::eComputeQueue) == ctx.domain_to_queue_index(DomainFlagBits::eGraphicsQueue)));
		if (needs_flatten) {
			bool needs_transfer_compute_flatten =
			    ctx.domain_to_queue_index(DomainFlagBits::eTransferQueue) == ctx.domain_to_queue_index(DomainFlagBits::eGraphicsQueue) &&
			    ctx.domain_to_queue_index(DomainFlagBits::eComputeQueue) == ctx.domain_to_queue_index(DomainFlagBits::eGraphicsQueue);
			if (needs_transfer_compute_flatten) {
				flatten_transfer_and_compute_onto_graphics(bundle);
			} else {
				assert(false && "NYI");
			}
		} else {
			if (bundle.batches.size() > 1) {
				std::swap(bundle.batches[0], bundle.batches[1]); // FIXME: silence some false positive validation
			}
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
				num_waits += submit_info.absolute_waits.size();
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
					auto& wait_queue = ctx.domain_to_queue(w.first).impl->submit_sync;
					ssi.semaphore = wait_queue.semaphore;
					ssi.value = queue_progress_references[ctx.domain_to_queue_index(w.first)] + w.second;
					ssi.stageMask = (VkPipelineStageFlagBits2KHR)PipelineStageFlagBits::eAllCommands;
					wait_semas.emplace_back(ssi);
					wait_sema_count++;
				}
				for (auto& w : submit_info.absolute_waits) {
					VkSemaphoreSubmitInfoKHR ssi{ VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO_KHR };
					auto& wait_queue = ctx.domain_to_queue(w.first).impl->submit_sync;
					ssi.semaphore = wait_queue.semaphore;
					ssi.value = w.second;
					ssi.stageMask = (VkPipelineStageFlagBits2KHR)PipelineStageFlagBits::eAllCommands;
					wait_semas.emplace_back(ssi);
					wait_sema_count++;
				}
				if (domain == DomainFlagBits::eGraphicsQueue && i == 0 && present_rdy != VK_NULL_HANDLE) { // TODO: for first cbuf only that refs the swapchain attment
					VkSemaphoreSubmitInfoKHR ssi{ VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO_KHR };
					ssi.semaphore = present_rdy;
					ssi.stageMask = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT_KHR;
					wait_semas.emplace_back(ssi);
					wait_sema_count++;
				}

				VkSemaphoreSubmitInfoKHR ssi{ VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO_KHR };
				ssi.semaphore = queue.impl->submit_sync.semaphore;
				ssi.value = ++(*queue.impl->submit_sync.value);

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

				VkSubmitInfo2KHR& si = sis.emplace_back(VkSubmitInfo2KHR{ VK_STRUCTURE_TYPE_SUBMIT_INFO_2_KHR });
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
			return bundles;
		}

		for (auto& bundle : *bundles) {
			VUK_DO_OR_RETURN(submit(allocator, bundle, present_rdy, render_complete));
		}

		return { expected_value };
	}

	Result<VkResult> present_to_one(Context& ctx, SingleSwapchainRenderBundle&& bundle) {
		VkPresentInfoKHR pi{ .sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR };
		pi.swapchainCount = 1;
		pi.pSwapchains = &bundle.swapchain->swapchain;
		pi.pImageIndices = &bundle.image_index;
		pi.waitSemaphoreCount = 1;
		pi.pWaitSemaphores = &bundle.render_complete;
		auto present_result = ctx.vkQueuePresentKHR(ctx.graphics_queue->impl->queue, &pi);
		if (present_result != VK_SUCCESS && present_result != VK_SUBOPTIMAL_KHR) {
			return { expected_error, VkException{ present_result } };
		}
		if (present_result == VK_SUBOPTIMAL_KHR || bundle.acquire_result == VK_SUBOPTIMAL_KHR) {
			return { expected_value, VK_SUBOPTIMAL_KHR };
		}
		return { expected_value, VK_SUCCESS };
	}

	Result<SingleSwapchainRenderBundle> acquire_one(Allocator& allocator, SwapchainRef swapchain) {
		Context& ctx = allocator.get_context();
		Unique<std::array<VkSemaphore, 2>> semas(allocator);
		VUK_DO_OR_RETURN(allocator.allocate_semaphores(*semas));
		auto [present_rdy, render_complete] = *semas;

		uint32_t image_index = (uint32_t)-1;
		VkResult acq_result = ctx.vkAcquireNextImageKHR(ctx.device, swapchain->swapchain, UINT64_MAX, present_rdy, VK_NULL_HANDLE, &image_index);
		// VK_SUBOPTIMAL_KHR shouldn't stop presentation; it is handled at the end
		if (acq_result != VK_SUCCESS && acq_result != VK_SUBOPTIMAL_KHR) {
			return { expected_error, VkException{ acq_result } };
		}

		return { expected_value, SingleSwapchainRenderBundle{ swapchain, image_index, present_rdy, render_complete, acq_result } };
	}

	Result<SingleSwapchainRenderBundle> acquire_one(Context& ctx, SwapchainRef swapchain, VkSemaphore present_ready, VkSemaphore render_complete) {
		uint32_t image_index = (uint32_t)-1;
		VkResult acq_result = ctx.vkAcquireNextImageKHR(ctx.device, swapchain->swapchain, UINT64_MAX, present_ready, VK_NULL_HANDLE, &image_index);
		// VK_SUBOPTIMAL_KHR shouldn't stop presentation; it is handled at the end
		if (acq_result != VK_SUCCESS && acq_result != VK_SUBOPTIMAL_KHR) {
			return { expected_error, VkException{ acq_result } };
		}

		return { expected_value, SingleSwapchainRenderBundle{ swapchain, image_index, present_ready, render_complete, acq_result } };
	}

	Result<SingleSwapchainRenderBundle> execute_submit(Allocator& allocator, ExecutableRenderGraph&& rg, SingleSwapchainRenderBundle&& bundle) {
		std::vector<std::pair<SwapchainRef, size_t>> swapchains_with_indexes = { { bundle.swapchain, bundle.image_index } };

		std::pair v = { &allocator, &rg };
		VUK_DO_OR_RETURN(execute_submit(allocator, std::span{ &v, 1 }, swapchains_with_indexes, bundle.present_ready, bundle.render_complete));

		return { expected_value, std::move(bundle) };
	}

	Result<VkResult> execute_submit_and_present_to_one(Allocator& allocator, ExecutableRenderGraph&& rg, SwapchainRef swapchain) {
		auto bundle = acquire_one(allocator, swapchain);
		if (!bundle) {
			return bundle;
		}
		auto bundle2 = execute_submit(allocator, std::move(rg), std::move(*bundle));
		if (!bundle2) {
			return bundle2;
		}
		return present_to_one(allocator.get_context(), std::move(*bundle2));
	}

	Result<void> execute_submit_and_wait(Allocator& allocator, ExecutableRenderGraph&& rg) {
		Context& ctx = allocator.get_context();
		std::pair v = { &allocator, &rg };
		VUK_DO_OR_RETURN(execute_submit(allocator, std::span{ &v, 1 }, {}, {}, {}));
		ctx.wait_idle(); // TODO:
		return { expected_value };
	}

	Result<VkResult> present(Allocator& allocator, Compiler& compiler, SwapchainRef swapchain, Future&& future, RenderGraphCompileOptions compile_options) {
		auto ptr = future.get_render_graph();
		auto erg = compiler.link(std::span{ &ptr, 1 }, compile_options);
		if (!erg) {
			return erg;
		}
		return execute_submit_and_present_to_one(allocator, std::move(*erg), swapchain);
	}

	SampledImage make_sampled_image(ImageView iv, SamplerCreateInfo sci) {
		return { SampledImage::Global{ iv, sci, ImageLayout::eReadOnlyOptimalKHR } };
	}

	SampledImage make_sampled_image(NameReference n, SamplerCreateInfo sci) {
		return { SampledImage::RenderGraphAttachment{ n, sci, {}, ImageLayout::eReadOnlyOptimalKHR } };
	}

	SampledImage make_sampled_image(NameReference n, ImageViewCreateInfo ivci, SamplerCreateInfo sci) {
		return { SampledImage::RenderGraphAttachment{ n, sci, ivci, ImageLayout::eReadOnlyOptimalKHR } };
	}

	Future::Future(std::shared_ptr<struct RenderGraph> org, Name output_binding, DomainFlags dst_domain) :
	    output_binding(QualifiedName{ {}, output_binding }),
	    rg(std::move(org)),
	    control(std::make_shared<FutureBase>()) {
		rg->attach_out(QualifiedName{ {}, output_binding }, *this, dst_domain);
	}

	Future::Future(std::shared_ptr<struct RenderGraph> org, QualifiedName output_binding, DomainFlags dst_domain) :
	    output_binding(output_binding),
	    rg(std::move(org)),
	    control(std::make_shared<FutureBase>()) {
		rg->attach_out(output_binding, *this, dst_domain);
	}

	Future::Future(const Future& o) noexcept : output_binding(o.output_binding), rg(o.rg), control(o.control) {}

	Future& Future::operator=(const Future& o) noexcept {
		control = o.control;
		rg = o.rg;
		output_binding = o.output_binding;
		return *this;
	}

	Future::Future(Future&& o) noexcept :
	    output_binding{ std::exchange(o.output_binding, QualifiedName{}) },
	    rg{ std::exchange(o.rg, nullptr) },
	    control{ std::exchange(o.control, nullptr) } {}

	Future& Future::operator=(Future&& o) noexcept {
		control = std::exchange(o.control, nullptr);
		rg = std::exchange(o.rg, nullptr);
		output_binding = std::exchange(o.output_binding, QualifiedName{});

		return *this;
	}

	Future::~Future() {
		if (rg && rg->impl) {
			rg->detach_out(output_binding, *this);
		}
	}

	Result<void> Future::wait(Allocator& allocator, Compiler& compiler) {
		if (control->status == FutureBase::Status::eInitial && !rg) {
			return { expected_error,
				       RenderGraphException{} }; // can't get wait for future that has not been attached anything or has been attached into a rendergraph
		} else if (control->status == FutureBase::Status::eHostAvailable) {
			return { expected_value };
		} else if (control->status == FutureBase::Status::eSubmitted) {
			std::pair w = { (DomainFlags)control->initial_domain, control->initial_visibility };
			allocator.get_context().wait_for_domains(std::span{ &w, 1 });
			return { expected_value };
		} else {
			auto erg = compiler.link(std::span{ &rg, 1 }, {});
			if (!erg) {
				return erg;
			}
			std::pair v = { &allocator, &*erg };
			VUK_DO_OR_RETURN(execute_submit(allocator, std::span{ &v, 1 }, {}, {}, {}));
			std::pair w = { (DomainFlags)control->initial_domain, control->initial_visibility };
			allocator.get_context().wait_for_domains(std::span{ &w, 1 });
			control->status = FutureBase::Status::eHostAvailable;
			return { expected_value };
		}
	}

	template<class T>
	Result<T> Future::get(Allocator& allocator, Compiler& compiler) {
		if (auto result = wait(allocator, compiler)) {
			return { expected_value, get_result<T>() };
		} else {
			return result;
		}
	}

	Result<void> Future::submit(Allocator& allocator, Compiler& compiler) {
		if (control->status == FutureBase::Status::eInitial && !rg) {
			return { expected_error, RenderGraphException{} };
		} else if (control->status == FutureBase::Status::eHostAvailable || control->status == FutureBase::Status::eSubmitted) {
			return { expected_value }; // nothing to do
		} else {
			control->status = FutureBase::Status::eSubmitted;
			auto erg = compiler.link(std::span{ &rg, 1 }, {});
			if (!erg) {
				return erg;
			}
			std::pair v = { &allocator, &*erg };
			VUK_DO_OR_RETURN(execute_submit(allocator, std::span{ &v, 1 }, {}, {}, {}));
			return { expected_value };
		}
	}

	template Result<Buffer> Future::get(Allocator&, Compiler&);
	template Result<ImageAttachment> Future::get(Allocator&, Compiler&);

	std::string_view image_view_type_to_sv(ImageViewType view_type) noexcept {
		switch (view_type) {
		case ImageViewType::e1D:
			return "1D";
		case ImageViewType::e2D:
			return "2D";
		case ImageViewType::e3D:
			return "3D";
		case ImageViewType::eCube:
			return "Cube";
		case ImageViewType::e1DArray:
			return "1DArray";
		case ImageViewType::e2DArray:
			return "2DArray";
		case ImageViewType::eCubeArray:
			return "CubeArray";
		default:
			assert(0 && "not reached.");
			return "";
		}
	}
} // namespace vuk
