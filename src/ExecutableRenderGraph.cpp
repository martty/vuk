#include "Cache.hpp"
#include "RenderGraphImpl.hpp"
#include "vuk/AllocatorHelpers.hpp"
#include "vuk/CommandBuffer.hpp"
#include "vuk/Context.hpp"
#include "vuk/Future.hpp"
#include "vuk/Hash.hpp" // for create
#include "vuk/RenderGraph.hpp"
#include "vuk/Util.hpp"
#include "vuk/runtime/vk/VulkanQueueExecutor.hpp"

#include <fmt/format.h>
#include <mutex>
#include <sstream>
#include <unordered_set>
#include <vector>

namespace vuk {
	ExecutableRenderGraph::ExecutableRenderGraph(Compiler& rg) : impl(rg.impl) {}

	ExecutableRenderGraph::ExecutableRenderGraph(ExecutableRenderGraph&& o) noexcept : impl(std::exchange(o.impl, nullptr)) {}
	ExecutableRenderGraph& ExecutableRenderGraph::operator=(ExecutableRenderGraph&& o) noexcept {
		impl = std::exchange(o.impl, nullptr);
		return *this;
	}

	ExecutableRenderGraph::~ExecutableRenderGraph() {}

	void begin_render_pass(Context& ctx, vuk::RenderPassInfo& rpass, VkCommandBuffer& cbuf, bool use_secondary_command_buffers) {
		VkRenderPassBeginInfo rbi{ .sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO };
		rbi.renderPass = rpass.handle;
		rbi.framebuffer = rpass.framebuffer;
		rbi.renderArea = VkRect2D{ vuk::Offset2D{}, vuk::Extent2D{ rpass.fbci.width, rpass.fbci.height } };
		rbi.clearValueCount = 0;

		ctx.vkCmdBeginRenderPass(cbuf, &rbi, use_secondary_command_buffers ? VK_SUBPASS_CONTENTS_SECONDARY_COMMAND_BUFFERS : VK_SUBPASS_CONTENTS_INLINE);
	}
	/*
	[[nodiscard]] bool resolve_image_barrier(const Context& ctx, VkImageMemoryBarrier2KHR& dep, const AttachmentInfo& bound, vuk::DomainFlagBits current_domain) {
	  dep.image = bound.attachment.image.image;
	  // turn base_{layer, level} into absolute values wrt the image
	  dep.subresourceRange.baseArrayLayer += bound.attachment.base_layer;
	  dep.subresourceRange.baseMipLevel += bound.attachment.base_level;
	  // clamp arrays and levels to actual accessible range in image, return false if the barrier would refer to no levels or layers
	  assert(bound.attachment.layer_count != VK_REMAINING_ARRAY_LAYERS);
	  if (dep.subresourceRange.layerCount != VK_REMAINING_ARRAY_LAYERS) {
	    if (dep.subresourceRange.baseArrayLayer + dep.subresourceRange.layerCount > bound.attachment.base_layer + bound.attachment.layer_count) {
	      int count = static_cast<int32_t>(bound.attachment.layer_count) - (dep.subresourceRange.baseArrayLayer - bound.attachment.base_layer);
	      if (count < 1) {
	        return false;
	      }
	      dep.subresourceRange.layerCount = static_cast<uint32_t>(count);
	    }
	  } else {
	    if (dep.subresourceRange.baseArrayLayer > bound.attachment.base_layer + bound.attachment.layer_count) {
	      return false;
	    }
	    dep.subresourceRange.layerCount = bound.attachment.layer_count;
	  }
	  assert(bound.attachment.level_count != VK_REMAINING_MIP_LEVELS);
	  if (dep.subresourceRange.levelCount != VK_REMAINING_MIP_LEVELS) {
	    if (dep.subresourceRange.baseMipLevel + dep.subresourceRange.levelCount > bound.attachment.base_level + bound.attachment.level_count) {
	      int count = static_cast<int32_t>(bound.attachment.level_count) - (dep.subresourceRange.baseMipLevel - bound.attachment.base_level);
	      if (count < 1)
	        return false;
	      dep.subresourceRange.levelCount = static_cast<uint32_t>(count);
	    }
	  } else {
	    if (dep.subresourceRange.baseMipLevel > bound.attachment.base_level + bound.attachment.level_count) {
	      return false;
	    }
	    dep.subresourceRange.levelCount = bound.attachment.level_count;
	  }

	  if (dep.srcQueueFamilyIndex != VK_QUEUE_FAMILY_IGNORED) {
	    assert(dep.dstQueueFamilyIndex != VK_QUEUE_FAMILY_IGNORED);
	    bool transition = dep.dstQueueFamilyIndex != dep.srcQueueFamilyIndex;
	    auto src_domain = static_cast<vuk::DomainFlagBits>(dep.srcQueueFamilyIndex);
	    auto dst_domain = static_cast<vuk::DomainFlagBits>(dep.dstQueueFamilyIndex);
	    dep.srcQueueFamilyIndex = ctx.domain_to_queue_family_index(static_cast<vuk::DomainFlags>(dep.srcQueueFamilyIndex));
	    dep.dstQueueFamilyIndex = ctx.domain_to_queue_family_index(static_cast<vuk::DomainFlags>(dep.dstQueueFamilyIndex));
	    if (dep.srcQueueFamilyIndex == dep.dstQueueFamilyIndex && transition) {
	      if (dst_domain != current_domain) {
	        return false; // discard release barriers if they map to the same queue
	      }
	    }
	  }

	  return true;
	}*/

	void ExecutableRenderGraph::fill_render_pass_info(vuk::RenderPassInfo& rpass, const size_t& i, vuk::CommandBuffer& cobuf) {
		if (rpass.handle == VK_NULL_HANDLE) {
			cobuf.ongoing_render_pass = {};
			return;
		}
		vuk::CommandBuffer::RenderPassInfo rpi;
		rpi.render_pass = rpass.handle;
		rpi.subpass = (uint32_t)i;
		rpi.extent = vuk::Extent2D{ rpass.fbci.width, rpass.fbci.height };
		auto& spdesc = rpass.rpci.subpass_descriptions[i];
		rpi.color_attachments = std::span<const VkAttachmentReference>(spdesc.pColorAttachments, spdesc.colorAttachmentCount);
		rpi.samples = rpass.fbci.sample_count.count;
		rpi.depth_stencil_attachment = spdesc.pDepthStencilAttachment;
		for (uint32_t i = 0; i < spdesc.colorAttachmentCount; i++) {
			rpi.color_attachment_ivs[i] = rpass.fbci.attachments[i];
		}
		cobuf.color_blend_attachments.resize(spdesc.colorAttachmentCount);
		cobuf.ongoing_render_pass = rpi;
	}
	/*
	void RGCImpl::emit_barriers(Context& ctx,
	                            VkCommandBuffer cbuf,
	                            vuk::DomainFlagBits domain,
	                            RelSpan<VkMemoryBarrier2KHR> mem_bars,
	                            RelSpan<VkImageMemoryBarrier2KHR> im_bars) {
	  // resolve and compact image barriers in place
	  auto im_span = im_bars.to_span(image_barriers);

	  uint32_t imbar_dst_index = 0;
	  for (auto src_index = 0; src_index < im_bars.size(); src_index++) {
	    auto dep = im_span[src_index];
	    int32_t def_pass_idx;
	    std::memcpy(&def_pass_idx, &dep.pNext, sizeof(def_pass_idx));
	    dep.pNext = 0;
	    auto& bound = get_bound_attachment(def_pass_idx);
	    if (bound.parent_attachment < 0) {
	      if (!resolve_image_barrier(ctx, dep, get_bound_attachment(bound.parent_attachment), domain)) {
	        continue;
	      }
	    } else {
	      if (!resolve_image_barrier(ctx, dep, bound, domain)) {
	        continue;
	      }
	    }
	    im_span[imbar_dst_index++] = dep;
	  }

	  auto mem_span = mem_bars.to_span(mem_barriers);

	  VkDependencyInfoKHR dependency_info{ .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO_KHR,
	                                       .memoryBarrierCount = (uint32_t)mem_span.size(),
	                                       .pMemoryBarriers = mem_span.data(),
	                                       .imageMemoryBarrierCount = imbar_dst_index,
	                                       .pImageMemoryBarriers = im_span.data() };

	  if (mem_bars.size() > 0 || imbar_dst_index > 0) {
	    ctx.vkCmdPipelineBarrier2KHR(cbuf, &dependency_info);
	  }
	}*/
	/*
	Result<SubmitInfo> ExecutableRenderGraph::record_single_submit(Allocator& alloc, std::span<ScheduledItem*> items, vuk::DomainFlagBits domain) {
	  assert(items.size() > 0);

	  auto& ctx = alloc.get_context();
	  SubmitInfo si;

	  Unique<CommandPool> cpool(alloc);
	  VkCommandPoolCreateInfo cpci{ VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO };
	  cpci.flags = VkCommandPoolCreateFlagBits::VK_COMMAND_POOL_CREATE_TRANSIENT_BIT;
	  cpci.queueFamilyIndex = ctx.domain_to_queue_family_index(domain); // currently queue family idx = queue idx

	  VUK_DO_OR_RETURN(alloc.allocate_command_pools(std::span{ &*cpool, 1 }, std::span{ &cpci, 1 }));

	  robin_hood::unordered_set<SwapchainRef> used_swapchains;

	  Unique<CommandBufferAllocation> hl_cbuf(alloc);
	  CommandBufferAllocationCreateInfo ci{ .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY, .command_pool = *cpool };
	  VUK_DO_OR_RETURN(alloc.allocate_command_buffers(std::span{ &*hl_cbuf, 1 }, std::span{ &ci, 1 }));
	  si.command_buffers.emplace_back(*hl_cbuf);

	  VkCommandBuffer cbuf = hl_cbuf->command_buffer;

	  VkCommandBufferBeginInfo cbi{ .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO, .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT };
	  ctx.vkBeginCommandBuffer(cbuf, &cbi);

	  void* cbuf_profile_data = nullptr;
	  if (callbacks.on_begin_command_buffer)
	    cbuf_profile_data = callbacks.on_begin_command_buffer(callbacks.user_data, cbuf);

	  uint64_t command_buffer_index = items[0]->command_buffer_index;
	  int32_t render_pass_index = -1;
	  for (size_t i = 0; i < items.size(); i++) {
	    auto& item = items[i];

	    for (auto& ref : item->referenced_swapchains.to_span(impl->swapchain_references)) {
	      used_swapchains.emplace(impl->get_bound_attachment(ref).swapchain);
	    }

	    if (item->command_buffer_index != command_buffer_index) { // end old cb and start new one
	      if (auto result = ctx.vkEndCommandBuffer(cbuf); result != VK_SUCCESS) {
	        return { expected_error, VkException{ result } };
	      }

	      VUK_DO_OR_RETURN(alloc.allocate_command_buffers(std::span{ &*hl_cbuf, 1 }, std::span{ &ci, 1 }));
	      si.command_buffers.emplace_back(*hl_cbuf);

	      cbuf = hl_cbuf->command_buffer;

	      VkCommandBufferBeginInfo cbi{ .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO, .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT };
	      ctx.vkBeginCommandBuffer(cbuf, &cbi);

	      if (callbacks.on_begin_command_buffer)
	        cbuf_profile_data = callbacks.on_begin_command_buffer(callbacks.user_data, cbuf);
	    }

	    // if we had a render pass running, but now it changes
	    if (item->render_pass_index != render_pass_index && render_pass_index != -1) {
	      ctx.vkCmdEndRenderPass(cbuf);
	    }

	    if (i > 1) {
	      // insert post-barriers
	      impl->emit_barriers(ctx, cbuf, domain, items[i - 1]->post_memory_barriers, items[i - 1]->post_image_barriers);
	    }
	    // insert pre-barriers
	    impl->emit_barriers(ctx, cbuf, domain, item->pre_memory_barriers, item->pre_image_barriers);

	    // if render pass is changing and new pass uses one
	    if (item->render_pass_index != render_pass_index && item->render_pass_index != -1) {
	      begin_render_pass(ctx, impl->rpis[item->render_pass_index], cbuf, false);
	    }

	    render_pass_index = item->render_pass_index;

	    for (auto& w : item->relative_waits.to_span(impl->waits)) {
	      si.relative_waits.emplace_back(w);
	    }

	    for (auto& w : item->absolute_waits.to_span(impl->absolute_waits)) {
	      si.absolute_waits.emplace_back(w);
	    }

	    CommandBuffer cobuf(*this, ctx, alloc, cbuf);
	    if (render_pass_index >= 0) {
	      fill_render_pass_info(impl->rpis[item->render_pass_index], 0, cobuf);
	    } else {
	      cobuf.ongoing_render_pass = {};
	    }

	    // propagate signals onto SI
	    auto pass_fut_signals = item->future_signals.to_span(impl->future_signals);
	    si.future_signals.insert(si.future_signals.end(), pass_fut_signals.begin(), pass_fut_signals.end());

	    if (!item->qualified_name.is_invalid()) {
	      ctx.begin_region(cobuf.command_buffer, pass->qualified_name.name);
	    }
	    if (pass->pass->execute) {
	      cobuf.current_pass = pass;
	      std::vector<void*> list;
	      for (auto& r : pass->resources.to_span(impl->resources)) {
	        auto res = *get_resource_image(vuk::NameReference::direct(r.original_name), cobuf.current_pass);
	        list.emplace_back((void*)&res->attachment);
	      }
	      void* pass_profile_data = nullptr;
	      if (callbacks.on_begin_pass)
	        pass_profile_data = callbacks.on_begin_pass(callbacks.user_data, pass->pass->name, cbuf, (DomainFlagBits)pass->domain.m_mask);
	      pass->pass->execute(cobuf);
	      if (callbacks.on_end_pass)
	        callbacks.on_end_pass(callbacks.user_data, pass_profile_data);
	    }
	    if (!item->execable->debug_info qualified_name.is_invalid()) {
	      ctx.end_region(cobuf.command_buffer);
	    }

	    if (auto res = cobuf.result(); !res) {
	      return res;
	    }
	  }

	  if (render_pass_index != -1) {
	    ctx.vkCmdEndRenderPass(cbuf);
	  }

	  // insert post-barriers
	  impl->emit_barriers(ctx, cbuf, domain, items.back()->post_memory_barriers, items.back()->post_image_barriers);

	  if (callbacks.on_end_command_buffer)
	    callbacks.on_end_command_buffer(callbacks.user_data, cbuf_profile_data);
	  if (auto result = ctx.vkEndCommandBuffer(cbuf); result != VK_SUCCESS) {
	    return { expected_error, VkException{ result } };
	  }

	  si.used_swapchains.insert(si.used_swapchains.end(), used_swapchains.begin(), used_swapchains.end());

	  return { expected_value, std::move(si) };
	}*/

	struct VkQueueStream : public Stream {
		Context& ctx;
		vuk::rtvk::QueueExecutor* executor;

		std::vector<SubmitInfo> batch;
		std::deque<Signal> signals;
		SubmitInfo si;
		Unique<CommandPool> cpool;
		Unique<CommandBufferAllocation> hl_cbuf;
		VkCommandBuffer cbuf = VK_NULL_HANDLE;
		ProfilingCallbacks* callbacks;
		bool is_recording = false;
		void* cbuf_profile_data = nullptr;

		RenderPassInfo rp = {};
		std::vector<VkImageMemoryBarrier2KHR> im_bars;
		std::vector<VkImageMemoryBarrier2KHR> half_im_bars;
		std::vector<VkMemoryBarrier2KHR> mem_bars;
		std::vector<VkMemoryBarrier2KHR> half_mem_bars;

		VkQueueStream(Allocator alloc, vuk::rtvk::QueueExecutor* qe, ProfilingCallbacks* callbacks) :
		    Stream(alloc, qe),
		    ctx(alloc.get_context()),
		    executor(qe),
		    callbacks(callbacks) {
			domain = qe->tag.domain;
		}

		void add_dependency(Stream* dep) override {
			if (is_recording) {
				end_cbuf();
				batch.emplace_back();
			}
			dependencies.push_back(dep);
		}

		void sync_deps() override {
			if (batch.empty()) {
				batch.emplace_back();
			}
			for (auto dep : dependencies) {
				auto res = *dep->submit();
				if (res.signal) {
					batch.back().waits.push_back(res.signal);
				}
				if (res.sema_wait != VK_NULL_HANDLE) {
					batch.back().pres_wait.push_back(res.sema_wait);
				}
			}
			dependencies.clear();
			if (!is_recording) {
				begin_cbuf();
			}
			flush_barriers();
		}

		Result<SubmitResult> submit(Signal* signal = nullptr) override {
			sync_deps();
			end_cbuf();
			if (!signal) {
				signal = &signals.emplace_back();
			}
			signal->source.executor = executor;
			batch.back().signals.emplace_back(signal);
			executor->submit_batch(batch);
			batch.clear();
			return { expected_value, signal };
		}

		Result<VkResult> present(Swapchain& swp) {
			sync_deps();
			end_cbuf();
			batch.back().pres_signal.emplace_back(swp.semaphores[swp.linear_index * 2 + 1]);
			executor->submit_batch(batch);
			batch.clear();
			VkPresentInfoKHR pi{ .sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR };
			pi.swapchainCount = 1;
			pi.pSwapchains = &swp.swapchain;
			pi.pImageIndices = &swp.image_index;
			pi.waitSemaphoreCount = 1;
			pi.pWaitSemaphores = &swp.semaphores[swp.linear_index * 2 + 1];
			auto res = executor->queue_present(pi);
			if (res.value() && swp.acquire_result == VK_SUBOPTIMAL_KHR) {
				return { expected_value, VK_SUBOPTIMAL_KHR };
			}
			return res;
		}

		Result<void> begin_cbuf() {
			assert(!is_recording);
			is_recording = true;
			domain = domain;
			if (cpool->command_pool == VK_NULL_HANDLE) {
				cpool = Unique<CommandPool>(alloc);
				VkCommandPoolCreateInfo cpci{ VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO };
				cpci.flags = VkCommandPoolCreateFlagBits::VK_COMMAND_POOL_CREATE_TRANSIENT_BIT;
				cpci.queueFamilyIndex = executor->get_queue_family_index(); // currently queue family idx = queue idx

				VUK_DO_OR_RETURN(alloc.allocate_command_pools(std::span{ &*cpool, 1 }, std::span{ &cpci, 1 }));
			}
			hl_cbuf = Unique<CommandBufferAllocation>(alloc);
			CommandBufferAllocationCreateInfo ci{ .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY, .command_pool = *cpool };
			VUK_DO_OR_RETURN(alloc.allocate_command_buffers(std::span{ &*hl_cbuf, 1 }, std::span{ &ci, 1 }));

			si.command_buffers.emplace_back(*hl_cbuf);

			cbuf = hl_cbuf->command_buffer;

			VkCommandBufferBeginInfo cbi{ .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO, .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT };
			alloc.get_context().vkBeginCommandBuffer(cbuf, &cbi);

			cbuf_profile_data = nullptr;
			if (callbacks->on_begin_command_buffer)
				cbuf_profile_data = callbacks->on_begin_command_buffer(callbacks->user_data, cbuf);
		}

		Result<void> end_cbuf() {
			flush_barriers();
			is_recording = false;
			if (callbacks->on_end_command_buffer)
				callbacks->on_end_command_buffer(callbacks->user_data, cbuf_profile_data);
			if (auto result = ctx.vkEndCommandBuffer(hl_cbuf->command_buffer); result != VK_SUCCESS) {
				return { expected_error, VkException{ result } };
			}
			batch.back().command_buffers.push_back(hl_cbuf->command_buffer);
			cbuf = VK_NULL_HANDLE;
			return { expected_value };
		};

		void flush_barriers() {
			VkDependencyInfoKHR dependency_info{ .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO_KHR,
				                                   .memoryBarrierCount = (uint32_t)mem_bars.size(),
				                                   .pMemoryBarriers = mem_bars.data(),
				                                   .imageMemoryBarrierCount = (uint32_t)im_bars.size(),
				                                   .pImageMemoryBarriers = im_bars.data() };

			if (mem_bars.size() > 0 || im_bars.size() > 0) {
				ctx.vkCmdPipelineBarrier2KHR(cbuf, &dependency_info);
			}

			mem_bars.clear();
			im_bars.clear();
		}

		void synch_image(ImageAttachment& img_att, StreamResourceUse src_use, StreamResourceUse dst_use, void* tag) {
			auto aspect = format_to_aspect(img_att.format);

			// if we start an RP and we have LOAD_OP_LOAD (currently always), then we must upgrade access with an appropriate READ
			if (is_framebuffer_attachment(dst_use)) {
				if ((aspect & ImageAspectFlagBits::eColor) == ImageAspectFlags{}) { // not color -> depth or depth/stencil
					dst_use.access |= vuk::AccessFlagBits::eDepthStencilAttachmentRead;
				} else {
					dst_use.access |= vuk::AccessFlagBits::eColorAttachmentRead;
				}
			}

			Subrange::Image subrange = {};

			DomainFlagBits src_domain = src_use.stream ? src_use.stream->domain : DomainFlagBits::eNone;
			DomainFlagBits dst_domain = dst_use.stream ? dst_use.stream->domain : DomainFlagBits::eNone;

			scope_to_domain((VkPipelineStageFlagBits2KHR&)src_use.stages, src_domain & DomainFlagBits::eQueueMask);
			scope_to_domain((VkPipelineStageFlagBits2KHR&)dst_use.stages, dst_domain & DomainFlagBits::eQueueMask);

			// compute image barrier for this access -> access
			VkImageMemoryBarrier2KHR barrier{ .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2_KHR };
			barrier.srcAccessMask = is_read_access(src_use) ? 0 : (VkAccessFlags)src_use.access;
			barrier.dstAccessMask = (VkAccessFlags)dst_use.access;
			barrier.oldLayout = (VkImageLayout)src_use.layout;
			barrier.newLayout = (VkImageLayout)dst_use.layout;
			barrier.subresourceRange.aspectMask = (VkImageAspectFlags)aspect;
			barrier.subresourceRange.baseArrayLayer = subrange.base_layer;
			barrier.subresourceRange.baseMipLevel = subrange.base_level;
			barrier.subresourceRange.layerCount = subrange.layer_count;
			barrier.subresourceRange.levelCount = subrange.level_count;

			if (src_domain == DomainFlagBits::eAny || src_domain == DomainFlagBits::eHost) {
				src_domain = dst_domain;
			}
			if (dst_domain == DomainFlagBits::eAny) {
				dst_domain = src_domain;
			}

			barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
			barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
			if (src_use.stream && dst_use.stream && src_use.stream != dst_use.stream) { // cross-stream
				if (src_use.stream->executor && src_use.stream->executor->type == Executor::Type::eVulkanDeviceQueue && dst_use.stream->executor &&
				    dst_use.stream->executor->type == Executor::Type::eVulkanDeviceQueue) { // cross queue
					auto src_queue = static_cast<rtvk::QueueExecutor*>(src_use.stream->executor);
					auto dst_queue = static_cast<rtvk::QueueExecutor*>(dst_use.stream->executor);
					if (src_queue->get_queue_family_index() != dst_queue->get_queue_family_index()) { // cross queue family
						barrier.srcQueueFamilyIndex = src_queue->get_queue_family_index();
						barrier.dstQueueFamilyIndex = dst_queue->get_queue_family_index();
					}
				}
			}

			if (src_use.stages == PipelineStageFlags{}) {
				barrier.srcAccessMask = {};
			}
			if (dst_use.stages == PipelineStageFlags{}) {
				barrier.dstAccessMask = {};
			}

			barrier.srcStageMask = (VkPipelineStageFlags2)src_use.stages.m_mask;
			barrier.dstStageMask = (VkPipelineStageFlags2)dst_use.stages.m_mask;

			barrier.image = img_att.image.image;

			if (dst_domain == DomainFlagBits::eNone) {
				barrier.pNext = tag;
				half_im_bars.push_back(barrier);
			} else if (src_domain == DomainFlagBits::eNone) {
				auto it = std::find_if(half_im_bars.begin(), half_im_bars.end(), [=](auto& mb) { return mb.pNext == tag; });
				assert(it != half_im_bars.end());
				barrier.pNext = nullptr;
				barrier.srcAccessMask = it->srcAccessMask;
				barrier.srcStageMask = it->srcStageMask;
				barrier.srcQueueFamilyIndex = it->srcQueueFamilyIndex;
				barrier.oldLayout = it->oldLayout;
				im_bars.push_back(barrier);
				half_im_bars.erase(it);
				img_att.layout = (ImageLayout)barrier.newLayout;

				if (barrier.oldLayout != VK_IMAGE_LAYOUT_UNDEFINED) {
					assert(barrier.newLayout != VK_IMAGE_LAYOUT_UNDEFINED);
				}
			} else {
				im_bars.push_back(barrier);
				img_att.layout = (ImageLayout)barrier.newLayout;

				if (barrier.oldLayout != VK_IMAGE_LAYOUT_UNDEFINED) {
					assert(barrier.newLayout != VK_IMAGE_LAYOUT_UNDEFINED);
				}
			}

			if (is_framebuffer_attachment(dst_use)) {
				prepare_render_pass_attachment(alloc, img_att);
			}
		};

		void synch_memory(StreamResourceUse src_use, StreamResourceUse dst_use, void* tag) {
			VkMemoryBarrier2KHR barrier{ .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2_KHR };

			DomainFlagBits src_domain = src_use.stream ? src_use.stream->domain : DomainFlagBits::eNone;
			DomainFlagBits dst_domain = dst_use.stream ? dst_use.stream->domain : DomainFlagBits::eNone;

			if (src_domain == DomainFlagBits::eAny || dst_domain == DomainFlagBits::eHost) {
				src_domain = dst_domain;
			}
			if (dst_domain == DomainFlagBits::eAny) {
				dst_domain = src_domain;
			}

			scope_to_domain((VkPipelineStageFlagBits2KHR&)src_use.stages, src_domain & DomainFlagBits::eQueueMask);
			scope_to_domain((VkPipelineStageFlagBits2KHR&)dst_use.stages, dst_domain & DomainFlagBits::eQueueMask);

			barrier.srcAccessMask = is_read_access(src_use) ? 0 : (VkAccessFlags)src_use.access;
			barrier.dstAccessMask = (VkAccessFlags)dst_use.access;
			barrier.srcStageMask = (VkPipelineStageFlagBits2)src_use.stages.m_mask;
			barrier.dstStageMask = (VkPipelineStageFlagBits2)dst_use.stages.m_mask;
			if (barrier.srcStageMask == 0) {
				barrier.srcStageMask = (VkPipelineStageFlagBits2)PipelineStageFlagBits::eNone;
				barrier.srcAccessMask = {};
			}
			if (dst_domain == DomainFlagBits::eNone) {
				barrier.pNext = tag;
				half_mem_bars.push_back(barrier);
			} else if (src_domain == DomainFlagBits::eNone) {
				auto it = std::find_if(half_mem_bars.begin(), half_mem_bars.end(), [=](auto& mb) { return mb.pNext == tag; });
				assert(it != half_mem_bars.end());
				barrier.pNext = nullptr;
				barrier.srcAccessMask = it->srcAccessMask;
				barrier.srcStageMask = it->srcStageMask;
				mem_bars.push_back(barrier);
				half_mem_bars.erase(it);
			} else {
				mem_bars.push_back(barrier);
			}
		};

		void prepare_render_pass_attachment(Allocator alloc, ImageAttachment img_att) {
			auto aspect = format_to_aspect(img_att.format);
			VkAttachmentReference attref{};

			attref.attachment = (uint32_t)rp.rpci.attachments.size();

			auto& descr = rp.rpci.attachments.emplace_back();
			// no layout changed by RPs currently
			descr.initialLayout = (VkImageLayout)img_att.layout;
			descr.finalLayout = (VkImageLayout)img_att.layout;
			attref.layout = (VkImageLayout)img_att.layout;

			descr.loadOp = VK_ATTACHMENT_LOAD_OP_LOAD;
			descr.storeOp = VK_ATTACHMENT_STORE_OP_STORE;

			descr.format = (VkFormat)img_att.format;
			descr.samples = (VkSampleCountFlagBits)img_att.sample_count.count;

			if ((aspect & ImageAspectFlagBits::eColor) == ImageAspectFlags{}) { // not color -> depth or depth/stencil
				rp.rpci.ds_ref = attref;
			} else {
				rp.rpci.color_refs.push_back(attref);
			}

			if (img_att.image_view.payload == VK_NULL_HANDLE) {
				auto iv = allocate_image_view(alloc, img_att); // TODO: dropping error
				img_att.image_view = **iv;

				auto name = std::string("ImageView: RenderTarget ");
				alloc.get_context().set_name(img_att.image_view.payload, Name(name));
			}
			rp.framebuffer_ivs.push_back(img_att.image_view.payload);
			rp.fbci.width = img_att.extent.extent.width;
			rp.fbci.height = img_att.extent.extent.height;
			rp.fbci.layers = img_att.layer_count;
			assert(img_att.level_count == 1);
			rp.fbci.sample_count = img_att.sample_count;
			rp.fbci.attachments.push_back(img_att.image_view);
		}

		Result<void> prepare_render_pass() {
			SubpassDescription sd;
			size_t color_count = 0;
			sd.colorAttachmentCount = (uint32_t)rp.rpci.color_refs.size();
			sd.pColorAttachments = rp.rpci.color_refs.data();

			sd.pDepthStencilAttachment = rp.rpci.ds_ref ? &*rp.rpci.ds_ref : nullptr;
			sd.flags = {};
			sd.inputAttachmentCount = 0;
			sd.pInputAttachments = nullptr;
			sd.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
			sd.preserveAttachmentCount = 0;
			sd.pPreserveAttachments = nullptr;

			rp.rpci.subpass_descriptions.push_back(sd);

			rp.rpci.subpassCount = (uint32_t)rp.rpci.subpass_descriptions.size();
			rp.rpci.pSubpasses = rp.rpci.subpass_descriptions.data();

			// we use barriers
			rp.rpci.dependencyCount = 0;
			rp.rpci.pDependencies = nullptr;

			rp.rpci.attachmentCount = (uint32_t)rp.rpci.attachments.size();
			rp.rpci.pAttachments = rp.rpci.attachments.data();

			auto result = alloc.allocate_render_passes(std::span{ &rp.handle, 1 }, std::span{ &rp.rpci, 1 });

			rp.fbci.renderPass = rp.handle;
			rp.fbci.pAttachments = rp.framebuffer_ivs.data();
			rp.fbci.attachmentCount = (uint32_t)rp.framebuffer_ivs.size();

			Unique<VkFramebuffer> fb(alloc);
			VUK_DO_OR_RETURN(alloc.allocate_framebuffers(std::span{ &*fb, 1 }, std::span{ &rp.fbci, 1 }));
			rp.framebuffer = *fb; // queue framebuffer for destruction
			// drop render pass immediately
			if (result) {
				alloc.deallocate(std::span{ &rp.handle, 1 });
			}
			begin_render_pass(alloc.get_context(), rp, cbuf, false);

			return { expected_value };
		}

		void end_render_pass() {
			alloc.get_context().vkCmdEndRenderPass(cbuf);
			rp = {};
		}
	};

	struct HostStream : Stream {
		HostStream(Allocator alloc) : Stream(alloc, nullptr) {
			domain = DomainFlagBits::eHost;
		}

		void add_dependency(Stream* dep) override {
			dependencies.push_back(dep);
		}
		void sync_deps() override {
			assert(false);
		}

		void synch_image(ImageAttachment& img_att, StreamResourceUse src_use, StreamResourceUse dst_use, void* tag) override {
			/* host -> host and host -> device not needed, device -> host inserts things on the device side */
			return;
		}
		void synch_memory(StreamResourceUse src_use, StreamResourceUse dst_use, void* tag) override {
			/* host -> host and host -> device not needed, device -> host inserts things on the device side */
			return;
		}

		Result<SubmitResult> submit(Signal* signal = nullptr) override {
			return { expected_value, signal };
		}
	};

	struct VkPEStream : Stream {
		VkPEStream(Allocator alloc, Swapchain& swp) : Stream(alloc, nullptr), swp(&swp) {
			domain = DomainFlagBits::ePE;
		}
		Swapchain* swp;

		void add_dependency(Stream* dep) override {
			dependencies.push_back(dep);
		}
		void sync_deps() override {
			assert(false);
		}

		void synch_image(ImageAttachment& img_att, StreamResourceUse src_use, StreamResourceUse dst_use, void* tag) override {}

		void synch_memory(StreamResourceUse src_use, StreamResourceUse dst_use, void* tag) override { /* PE doesn't do memory */
			assert(false);
		}

		Result<SubmitResult> submit(Signal* signal = nullptr) override {
			assert(swp);
			assert(signal == nullptr);
			SubmitResult sr{ .sema_wait = swp->semaphores[2 * swp->linear_index] };
			return { expected_value, sr };
		}
	};

	enum class RW { eRead, eWrite };
	struct ExecutionInfo {
		Stream* stream;
		size_t naming_index;
	};

	struct Scheduler {
		Scheduler(Allocator all, std::vector<ScheduledItem>& scheduled_execables, DefUseMap& res_to_links, std::vector<Ref>& pass_reads) :
		    allocator(all),
		    scheduled_execables(scheduled_execables),
		    res_to_links(res_to_links),
		    pass_reads(pass_reads) {
			// these are the items that were determined to run
			for (auto& i : scheduled_execables) {
				scheduled.emplace(i.execable);
				work_queue.emplace_back(i);
			}
		}

		Allocator allocator;
		DefUseMap& res_to_links;
		std::vector<Ref>& pass_reads;
		std::vector<ScheduledItem>& scheduled_execables;

		std::deque<ScheduledItem> work_queue;

		size_t naming_index_counter = 0;
		std::unordered_map<Node*, ExecutionInfo> executed;
		std::unordered_set<Node*> scheduled;

		void schedule_new(Node* node) {
			assert(node);
			if (scheduled.count(node)) { // we have scheduling info for this
				auto it = std::find_if(scheduled_execables.begin(), scheduled_execables.end(), [=](ScheduledItem& item) { return item.execable == node; });
				if (it != scheduled_execables.end()) {
					work_queue.push_front(*it);
				}
			} else { // no info, just schedule it as-is
				work_queue.push_front(ScheduledItem{ .execable = node });
			}
		}

		// returns true if the item is ready
		bool process(ScheduledItem& item) {
			if (item.ready) {
				return true;
			} else {
				item.ready = true;
				work_queue.push_front(item); // requeue this item
				return false;
			}
		}

		void schedule_dependency(Ref parm, RW access) {
			auto it = res_to_links.find(parm);
			// some items (like constants) don't have links, so they also don't need to be scheduled
			if (it == res_to_links.end()) {
				return;
			}
			auto& link = it->second;

			if (access == RW::eWrite) { // synchronize against writing
				// we are going to write here, so schedule all reads or the def, if no read
				if (link.reads.size() > 0) {
					// all reads
					for (auto& r : link.reads.to_span(pass_reads)) {
						schedule_new(r.node);
					}
				} else {
					// just the def
					schedule_new(link.def.node);
				}
			} else { // just reading, so don't synchronize with reads
				// just the def
				schedule_new(link.def.node);
			}
		}

		void done(Node* node, Stream* stream) {
			auto counter = naming_index_counter;
			naming_index_counter += node->type.size();
			executed.emplace(node, ExecutionInfo{ stream, counter });
		}

		template<class T>
		T& get_value(Ref parm) {
			auto& link = res_to_links[parm];
			if (link.urdef.node->kind == Node::AALLOC) {
				return reinterpret_cast<T&>(link.urdef.node->aalloc.args[0].node->constant.value);
			} else {
				return *reinterpret_cast<T*>(get_value(parm));
			}
		};

		void* get_value(Ref parm) {
			auto& link = res_to_links[parm];
			return get_constant_value(link.urdef.node);
		}

		template<class T>
		T& get_array_elem_value_arg(Ref parm, size_t index) {
			auto& link = res_to_links[parm];
			auto& arg = link.urdef.node->aalloc.args[index + 1];
			return get_value<T>(arg);
		};

		template<class T>
		T& get_array_elem_value(Ref parm, size_t index) {
			auto& link = res_to_links[parm];
			return get_value<T*>(parm)[index];
		};

		Type* base_type(Ref parm) {
			return Type::stripped(parm.type());
		}

		struct DependencyInfo {
			StreamResourceUse src_use;
			StreamResourceUse dst_use;
		};

		DependencyInfo
		get_dependency_info(Ref parm, Type* arg_ty, RW type, Stream* dst_stream, Access src_access = Access::eNone, Access dst_access = Access::eNone) {
			auto parm_ty = parm.type();
			auto& link = res_to_links[parm];

			StreamResourceUse src_use = {};
			bool sync_against_def = type == RW::eRead || link.reads.size() == 0; // def -> *, src = def
			if (sync_against_def) {
				if (arg_ty->kind == Type::IMBUED_TY) {
					if (parm_ty->kind == Type::ALIASED_TY) { // this is coming from an output annotated, so we know the source access
						auto src_arg = parm.node->call.args[parm_ty->aliased.ref_idx];
						auto call_ty = parm.node->call.fn.type()->opaque_fn.args[parm_ty->aliased.ref_idx];
						if (call_ty->kind == Type::IMBUED_TY) {
							src_access = call_ty->imbued.access;
						} else {
							// TODO: handling unimbued aliased
							src_access = Access::eNone;
						}
					} else if (parm_ty->kind == Type::IMBUED_TY) {
						assert(0);
					} else { // there is no need to sync (eg. declare)
					}
				} else if (parm_ty->kind == Type::ALIASED_TY) { // this is coming from an output annotated, so we know the source access
					auto src_arg = parm.node->call.args[parm_ty->aliased.ref_idx];
					auto call_ty = parm.node->call.fn.type()->opaque_fn.args[parm_ty->aliased.ref_idx];
					if (call_ty->kind == Type::IMBUED_TY) {
						src_access = call_ty->imbued.access;
					} else {
						// TODO: handling unimbued aliased
					}
				} else {
					/* no src access */
				}
				src_use = { to_use(src_access), executed.at(parm.node).stream };
			} else {                       // read* -> undef, src = reads
				if (link.reads.size() > 0) { // we need to emit: def -> reads, RAW or nothing (before first read)
					// to avoid R->R deps, we emit a single dep for all the reads
					// for this we compute a merged layout (TRANSFER_SRC_OPTIMAL / READ_ONLY_OPTIMAL / GENERAL)
					ResourceUse use;
					auto reads = link.reads.to_span(pass_reads);

					bool need_read_only = false;
					bool need_transfer = false;
					bool need_general = false;
					src_use.stream = nullptr;
					src_use.layout = ImageLayout::eReadOnlyOptimalKHR;
					for (int read_idx = 0; read_idx < reads.size(); read_idx++) {
						auto& r = reads[read_idx];
						if (r.node->kind == Node::CALL) {
							arg_ty = r.node->call.fn.type()->opaque_fn.args[r.index];
							parm = r.node->call.args[r.index];
						} else {
							assert(0);
						}

						if (arg_ty->kind == Type::IMBUED_TY) {
							if (parm_ty->kind == Type::ALIASED_TY) { // this is coming from an output annotated, so we know the source access
								auto src_arg = parm.node->call.args[parm_ty->aliased.ref_idx];
								auto call_ty = parm.node->call.fn.type()->opaque_fn.args[parm_ty->aliased.ref_idx];
								if (call_ty->kind == Type::IMBUED_TY) {
									src_access = call_ty->imbued.access;
								} else {
									// TODO: handling unimbued aliased
									src_access = Access::eNone;
								}
							} else if (parm_ty->kind == Type::IMBUED_TY) {
								assert(0);
							} else { // there is no need to sync (eg. declare)
							}
						} else if (parm_ty->kind == Type::ALIASED_TY) { // this is coming from an output annotated, so we know the source access
							auto src_arg = parm.node->call.args[parm_ty->aliased.ref_idx];
							auto call_ty = parm.node->call.fn.type()->opaque_fn.args[parm_ty->aliased.ref_idx];
							if (call_ty->kind == Type::IMBUED_TY) {
								src_access = call_ty->imbued.access;
							} else {
								// TODO: handling unimbued aliased
							}
						} else {
							/* no src access */
						}

						StreamResourceUse use = { to_use(src_access), executed.at(parm.node).stream };
						if (use.stream == nullptr) {
							use.stream = src_use.stream;
						} else if (use.stream != src_use.stream && src_use.stream) {
							// there are multiple stream in this read group
							// this is okay - but in this case we can't synchronize against all of them together
							// so we synchronize against them individually by setting last use and ending the read gather
							assert(false); // we should've handled this by now
						}

						if (is_transfer_access(dst_access)) {
							need_transfer = true;
						}
						if (is_storage_access(dst_access)) {
							need_general = true;
						}
						if (is_readonly_access(dst_access)) {
							need_read_only = true;
						}

						src_use.access |= use.access;
						src_use.stages |= use.stages;
						src_use.stream = use.stream;
					}

					// compute barrier and waits for the merged reads

					if (need_transfer && !need_read_only) {
						src_use.layout = ImageLayout::eTransferSrcOptimal;
					}

					if (need_general || (need_transfer && need_read_only)) {
						src_use.layout = ImageLayout::eGeneral;
					}
				}
			}

			StreamResourceUse dst_use = {};
			if (type == RW::eWrite) { // * -> undef, dst = undef
				if (arg_ty->kind == Type::IMBUED_TY) {
					dst_access = arg_ty->imbued.access;
				} else {
					/* no dst access */
				}
				dst_use = { to_use(dst_access), dst_stream };
			} else if (type == RW::eRead) { // def -> read, dst = sum(read)
				if (link.reads.size() > 0) {  // we need to emit: def -> reads, RAW or nothing (before first read)
					// to avoid R->R deps, we emit a single dep for all the reads
					// for this we compute a merged layout (TRANSFER_SRC_OPTIMAL / READ_ONLY_OPTIMAL / GENERAL)
					ResourceUse use;
					auto reads = link.reads.to_span(pass_reads);

					bool need_read_only = false;
					bool need_transfer = false;
					bool need_general = false;
					dst_use.stream = nullptr;
					dst_use.layout = ImageLayout::eReadOnlyOptimalKHR;
					for (int read_idx = 0; read_idx < reads.size(); read_idx++) {
						auto& r = reads[read_idx];
						if (r.node->kind == Node::CALL) {
							arg_ty = r.node->call.fn.type()->opaque_fn.args[r.index];
							parm = r.node->call.args[r.index];
						} else {
							assert(0);
						}

						if (arg_ty->kind == Type::IMBUED_TY) {
							dst_access = arg_ty->imbued.access;
						} else {
							assert(0);
						}

						StreamResourceUse use = { to_use(dst_access), dst_stream };
						if (use.stream == nullptr) {
							use.stream = dst_use.stream;
						} else if (use.stream != dst_use.stream && dst_use.stream) {
							// there are multiple stream in this read group
							// this is okay - but in this case we can't synchronize against all of them together
							// so we synchronize against them individually by setting last use and ending the read gather
							assert(false); // we should've handled this by now
						}

						if (is_transfer_access(dst_access)) {
							need_transfer = true;
						}
						if (is_storage_access(dst_access)) {
							need_general = true;
						}
						if (is_readonly_access(dst_access)) {
							need_read_only = true;
						}

						dst_use.access |= use.access;
						dst_use.stages |= use.stages;
						dst_use.stream = use.stream;
					}

					// compute barrier and waits for the merged reads

					if (need_transfer && !need_read_only) {
						dst_use.layout = ImageLayout::eTransferSrcOptimal;
					}

					if (need_general || (need_transfer && need_read_only)) {
						dst_use.layout = ImageLayout::eGeneral;
					}
				}
			}
			return { src_use, dst_use };
		}
	};

	struct Recorder {
		Recorder(Allocator alloc, ProfilingCallbacks* callbacks, std::vector<Ref>& pass_reads) :
		    ctx(alloc.get_context()),
		    alloc(alloc),
		    callbacks(callbacks),
		    pass_reads(pass_reads) {}
		Context& ctx;
		Allocator alloc;
		ProfilingCallbacks* callbacks;
		std::vector<Ref>& pass_reads;

		std::unordered_map<DomainFlagBits, std::unique_ptr<Stream>> streams;

		// start recording if needed
		// all dependant domains flushed
		// all pending sync to be flushed
		void synchronize_stream(Stream* stream) {
			stream->sync_deps();
		}

		Stream* stream_for_domain(DomainFlagBits domain) {
			auto it = streams.find(domain);
			if (it != streams.end()) {
				return it->second.get();
			}
			return nullptr;
		}

		Stream* stream_for_executor(Executor* executor) {
			for (auto& [domain, stream] : streams) {
				if (stream->executor == executor) {
					return stream.get();
				}
			}
			assert(0);
			return nullptr;
		}

		void flush_domain(vuk::DomainFlagBits domain, Signal* signal) {
			auto& stream = streams.at(domain);

			stream->submit(signal);
		};

		void add_sync(Type* base_ty, Scheduler::DependencyInfo di, void* value) {
			StreamResourceUse src_use = di.src_use;
			StreamResourceUse dst_use = di.dst_use;
			auto src_stream = src_use.stream;
			auto dst_stream = dst_use.stream;
			bool has_src = src_stream;
			bool has_dst = dst_stream;
			bool has_both = has_src && has_dst;
			bool cross = has_both && (src_stream != dst_stream);
			bool only_src = has_src && !has_dst;

			if (cross) {
				dst_stream->add_dependency(src_stream);
			}

			if (base_ty->is_image()) {
				auto& img_att = *reinterpret_cast<ImageAttachment*>(value);
				if (has_dst) {
					dst_stream->synch_image(img_att, src_use, dst_use, value);
				}
				if (only_src || cross) {
					src_stream->synch_image(img_att, src_use, dst_use, value);
				}
			} else if (base_ty->is_buffer()) {
				// buffer needs no cross
				if (has_dst) {
					dst_stream->synch_memory(src_use, dst_use, value);
				} else if (has_src) {
					src_stream->synch_memory(src_use, dst_use, value);
				}
			} else if (base_ty->kind == Type::ARRAY_TY) {
				auto elem_ty = base_ty->array.T;
				auto size = base_ty->array.size;
				if (elem_ty->is_image()) {
					auto img_atts = reinterpret_cast<ImageAttachment**>(value);
					for (int i = 0; i < size; i++) {
						if (has_dst) {
							dst_stream->synch_image(*img_atts[i], src_use, dst_use, img_atts[i]);
						}
						if (only_src || cross) {
							src_stream->synch_image(*img_atts[i], src_use, dst_use, img_atts[i]);
						}
					}
				} else if (elem_ty->is_buffer()) {
					for (int i = 0; i < size; i++) {
						// buffer needs no cross
						auto bufs = reinterpret_cast<Buffer**>(value);
						if (has_dst) {
							dst_stream->synch_memory(src_use, dst_use, bufs[i]);
						} else if (has_src) {
							src_stream->synch_memory(src_use, dst_use, bufs[i]);
						}
					}
				} else {
					assert(0);
				}
			} else {
				assert(0);
			}
		}
	};

	std::string_view domain_to_string(DomainFlagBits domain) {
		domain = (DomainFlagBits)(domain & DomainFlagBits::eDomainMask).m_mask;

		switch (domain) {
		case DomainFlagBits::eNone:
			return "None";
		case DomainFlagBits::eHost:
			return "Host";
		case DomainFlagBits::ePE:
			return "PE";
		case DomainFlagBits::eGraphicsQueue:
			return "Graphics";
		case DomainFlagBits::eComputeQueue:
			return "Compute";
		case DomainFlagBits::eTransferQueue:
			return "Transfer";
		}
		assert(false);
		return "";
	}

#define VUK_DUMP_EXEC

	Result<void> ExecutableRenderGraph::execute(Allocator& alloc) {
		Context& ctx = alloc.get_context();

		Recorder recorder(alloc, &impl->callbacks, impl->pass_reads);
		recorder.streams.emplace(DomainFlagBits::eHost, std::make_unique<HostStream>(alloc));
		if (auto exe = ctx.get_executor(DomainFlagBits::eGraphicsQueue)) {
			recorder.streams.emplace(DomainFlagBits::eGraphicsQueue,
			                         std::make_unique<VkQueueStream>(alloc, static_cast<rtvk::QueueExecutor*>(exe), &impl->callbacks));
		}
		if (auto exe = ctx.get_executor(DomainFlagBits::eComputeQueue)) {
			recorder.streams.emplace(DomainFlagBits::eComputeQueue,
			                         std::make_unique<VkQueueStream>(alloc, static_cast<rtvk::QueueExecutor*>(exe), &impl->callbacks));
		}
		if (auto exe = ctx.get_executor(DomainFlagBits::eTransferQueue)) {
			recorder.streams.emplace(DomainFlagBits::eTransferQueue,
			                         std::make_unique<VkQueueStream>(alloc, static_cast<rtvk::QueueExecutor*>(exe), &impl->callbacks));
		}
		auto host_stream = recorder.streams.at(DomainFlagBits::eHost).get();

		std::deque<VkPEStream> pe_streams;

		for (auto& item : impl->scheduled_execables) {
			item.scheduled_stream = recorder.stream_for_domain(item.scheduled_domain);
		}

		Scheduler sched(alloc, impl->scheduled_execables, impl->res_to_links, impl->pass_reads);

		// DYNAMO
		// loop through scheduled items
		// for each scheduled item, schedule deps
		// generate barriers and batch breaks as needed by deps
		// allocate images/buffers as declares are encountered
		// start/end RPs as needed
		// inference will still need to run in the beginning, as that is compiletime

		auto print_results = [&](Node* node) {
			for (size_t i = 0; i < node->type.size(); i++) {
				if (i > 0) {
					fmt::print(", ");
				}
				if (node->debug_info) {
					fmt::print("%{}", node->debug_info->result_names[i]);
				} else {
					fmt::print("%{}_{}", node->kind_to_sv(), sched.naming_index_counter + i);
				}
			}
		};
		auto print_args = [&](std::span<Ref> args) {
			for (size_t i = 0; i < args.size(); i++) {
				if (i > 0) {
					fmt::print(", ");
				}
				auto& parm = args[i];

				if (parm.node->debug_info) {
					fmt::print("%{}", parm.node->debug_info->result_names[parm.index]);
				} else {
					fmt::print("%{}_{}", parm.node->kind_to_sv(), sched.executed.at(parm.node).naming_index + parm.index);
				}
			}
		};

		while (!sched.work_queue.empty()) {
			auto item = sched.work_queue.front();
			sched.work_queue.pop_front();
			auto& node = item.execable;
			if (sched.executed.count(node)) { // only going execute things once
				continue;
			}
			// we run nodes twice - first time we reenqueue at the front and then put all deps before it
			// second time we see it, we know that all deps have run, so we can run the node itself
			switch (node->kind) {
			case Node::VALLOC: { // when encountering a DECLARE, allocate the thing if needed
				if (node->type[0]->kind == Type::BUFFER_TY) {
					auto& bound = constant<Buffer>(node->valloc.args[0]);
					bound.size = eval<size_t>(node->valloc.args[1]); // collapse inferencing
#ifdef VUK_DUMP_EXEC
					print_results(node);
					fmt::print(" = declare<buffer>\n");
#endif
					if (bound.buffer == VK_NULL_HANDLE) {
						assert(bound.size != ~(0u));
						BufferCreateInfo bci{ .mem_usage = bound.memory_usage, .size = bound.size, .alignment = 1 }; // TODO: alignment?
						auto allocator = node->valloc.allocator ? *node->valloc.allocator : alloc;
						auto buf = allocate_buffer(allocator, bci);
						if (!buf) {
							return buf;
						}
						bound = **buf;
					}
				} else if (node->type[0]->kind == Type::IMAGE_TY) {
					auto& attachment = *reinterpret_cast<ImageAttachment*>(node->valloc.args[0].node->constant.value);
					// collapse inferencing
					attachment.extent.extent.width = eval<uint32_t>(node->valloc.args[1]);
					attachment.extent.extent.height = eval<uint32_t>(node->valloc.args[2]);
					attachment.extent.extent.depth = eval<uint32_t>(node->valloc.args[3]);
					attachment.extent.sizing = Sizing::eAbsolute;
					attachment.format = constant<Format>(node->valloc.args[4]);
					attachment.sample_count = constant<Samples>(node->valloc.args[5]);
					attachment.base_layer = eval<uint32_t>(node->valloc.args[6]);
					attachment.layer_count = eval<uint32_t>(node->valloc.args[7]);
					attachment.base_level = eval<uint32_t>(node->valloc.args[8]);
					attachment.level_count = eval<uint32_t>(node->valloc.args[9]);
#ifdef VUK_DUMP_EXEC
					print_results(node);
					fmt::print(" = declare<image>\n");
#endif
					if (!attachment.image) {
						auto allocator = node->valloc.allocator ? *node->valloc.allocator : alloc;
						attachment.usage = impl->compute_usage(&impl->res_to_links[first(node)]);
						assert(attachment.usage != ImageUsageFlags{});
						auto img = allocate_image(allocator, attachment);
						if (!img) {
							return img;
						}
						attachment.image = **img;
						// ctx.set_name(attachment.image.image, bound.name.name);
					}
				} else if (node->type[0]->kind == Type::SWAPCHAIN_TY) {
#ifdef VUK_DUMP_EXEC
					print_results(node);
					fmt::print(" = declare<swapchain>\n");
#endif
					/* no-op */
				} else {
					assert(0);
				}
				sched.done(node, host_stream); // declarations execute on the host
				break;
			}
			case Node::AALLOC: {
				assert(node->type[0]->kind == Type::ARRAY_TY);
				if (sched.process(item)) {
					for (size_t i = 1; i < node->aalloc.args.size(); i++) {
						auto arg_ty = node->aalloc.args[i].type();
						auto& parm = node->aalloc.args[i];
						auto& link = impl->res_to_links[parm];

						recorder.add_sync(sched.base_type(parm), sched.get_dependency_info(parm, arg_ty, RW::eWrite, nullptr), sched.get_value(parm));
					}

#ifdef VUK_DUMP_EXEC
					print_results(node);
					auto size = node->type[0]->array.size;
					auto elem_ty = node->type[0]->array.T;
					assert(elem_ty->kind == Type::BUFFER_TY || elem_ty->kind == Type::IMAGE_TY);
					fmt::print(" = declare<{}[{}]> ", elem_ty->kind == Type::BUFFER_TY ? "buffer" : "image", size);
					print_args(node->valloc.args.subspan(1));
					assert(node->valloc.args[0].type()->kind == Type::MEMORY_TY);
					if (elem_ty->kind == Type::BUFFER_TY) {
						auto arr_mem = new Buffer[size];
						for (auto i = 0; i < size; i++) {
							auto& elem = node->valloc.args[i + 1];
							assert(Type::stripped(elem.type())->kind == Type::BUFFER_TY);

							memcpy(&arr_mem[i], sched.get_value(elem), sizeof(Buffer));
						}
						node->valloc.args[0].node->constant.value = arr_mem;
					} else if (elem_ty->kind == Type::IMAGE_TY) {
						auto arr_mem = new ImageAttachment[size];
						for (auto i = 0; i < size; i++) {
							auto& elem = node->valloc.args[i + 1];
							assert(Type::stripped(elem.type())->kind == Type::IMAGE_TY);

							memcpy(&arr_mem[i], sched.get_value(elem), sizeof(ImageAttachment));
						}
						node->valloc.args[0].node->constant.value = arr_mem;
					}

					fmt::print("\n");
#endif
					sched.done(node, host_stream); // declarations execute on the host
				} else {
					for (auto& parm : node->valloc.args.subspan(1)) {
						sched.schedule_dependency(parm, RW::eWrite);
					}
				}
				break;
			}
			case Node::CALL: {
				if (sched.process(item)) {                    // we have executed every dep, so execute ourselves too
					Stream* dst_stream = item.scheduled_stream; // the domain this call will execute on

					// run all the barriers here!

					for (size_t i = 0; i < node->call.args.size(); i++) {
						auto& arg_ty = node->call.fn.type()->opaque_fn.args[i];
						auto& parm = node->call.args[i];
						auto& link = impl->res_to_links[parm];

						if (arg_ty->kind == Type::IMBUED_TY) {
							auto dst_access = arg_ty->imbued.access;

							// here: figuring out which allocator to use to make image views for the RP and then making them
							if (is_framebuffer_attachment(dst_access)) {
								auto urdef = link.urdef.node;
								auto allocator = urdef->valloc.allocator ? *urdef->valloc.allocator : alloc;
								auto& img_att = sched.get_value<ImageAttachment>(parm);
								if (img_att.view_type == ImageViewType::eInfer) {// framebuffers need 2D or 2DArray views
									if (img_att.layer_count > 1) {
										img_att.view_type = ImageViewType::e2DArray;
									} else {
										img_att.view_type = ImageViewType::e2D;
									}
								}
								if (img_att.image_view.payload == VK_NULL_HANDLE) {
									auto iv = allocate_image_view(allocator, img_att); // TODO: dropping error
									img_att.image_view = **iv;

									auto name = std::string("ImageView: RenderTarget ");
									alloc.get_context().set_name(img_att.image_view.payload, Name(name));
								}
							}
						}
						auto value = sched.get_value(parm);
						if (arg_ty->kind == Type::IMBUED_TY) {
							auto access = arg_ty->imbued.access;
							// Write and ReadWrite
							RW sync_access = (is_write_access(access) || access == Access::eConsume) ? RW::eWrite : RW::eRead;
							recorder.add_sync(sched.base_type(parm), sched.get_dependency_info(parm, arg_ty, sync_access, dst_stream), sched.get_value(parm));
						} else {
							assert(0);
						}
					}

					// make the renderpass if needed!
					recorder.synchronize_stream(dst_stream);
					auto vk_rec = dynamic_cast<VkQueueStream*>(dst_stream); // TODO: change this into dynamic dispatch on the Stream
					assert(vk_rec);
					// run the user cb!
					if (node->call.fn.type()->kind == Type::OPAQUE_FN_TY) {
						CommandBuffer cobuf(*this, ctx, alloc, vk_rec->cbuf);
						if (vk_rec->rp.rpci.attachments.size() > 0) {
							vk_rec->prepare_render_pass();
							fill_render_pass_info(vk_rec->rp, 0, cobuf);
						}

						std::vector<void*> opaque_args;
						std::vector<void*> opaque_meta;
						std::vector<void*> opaque_rets;
						for (size_t i = 0; i < node->call.args.size(); i++) {
							auto& parm = node->call.args[i];
							auto& link = impl->res_to_links[parm];
							assert(link.urdef);
							opaque_args.push_back(sched.get_value(parm));
							opaque_meta.push_back(&link.urdef);
						}
						opaque_rets.resize(node->call.fn.type()->opaque_fn.return_types.size());
						(*node->call.fn.type()->opaque_fn.callback)(cobuf, opaque_args, opaque_meta, opaque_rets);

						if (vk_rec->rp.handle) {
							vk_rec->end_render_pass();
						}
					} else {
						assert(0);
					}
#ifdef VUK_DUMP_EXEC
					print_results(node);
					fmt::print(" = call ${} ", domain_to_string(dst_stream->domain));
					if (node->call.fn.type()->debug_info) {
						fmt::print("<{}> ", node->call.fn.type()->debug_info->name);
					}
					print_args(node->call.args);
					fmt::print("\n");
#endif
					sched.done(node, dst_stream);
				} else { // execute deps
					for (size_t i = 0; i < node->call.args.size(); i++) {
						auto& arg_ty = node->call.fn.type()->opaque_fn.args[i];
						auto& parm = node->call.args[i];
						auto& link = impl->res_to_links[parm];

						if (arg_ty->kind == Type::IMBUED_TY) {
							auto access = arg_ty->imbued.access;
							// Write and ReadWrite
							RW sync_access = (is_write_access(access) || access == Access::eConsume) ? RW::eWrite : RW::eRead;
							sched.schedule_dependency(parm, sync_access);
						} else {
							assert(0);
						}
					}
				}
				break;
			}
			case Node::ACQUIRE: {
				auto acq = node->acquire.acquire;
				auto src_stream = recorder.stream_for_executor(acq->source.executor);
				Stream* dst_stream = item.scheduled_stream;

				Scheduler::DependencyInfo di;
				di.src_use = { acq->last_use, src_stream };
				di.dst_use = { to_use(Access::eNone), dst_stream };
				recorder.add_sync(node->type[0], di, sched.get_value(first(node)));

				if (node->type[0]->kind == Type::BUFFER_TY) {
#ifdef VUK_DUMP_EXEC
					print_results(node);
					fmt::print(" = acquire<buffer>\n");
#endif
				} else if (node->type[0]->kind == Type::IMAGE_TY) {
#ifdef VUK_DUMP_EXEC
					print_results(node);
					fmt::print(" = acquire<image>\n");
#endif
				}

				sched.done(node, dst_stream);
				break;
			}
			case Node::RELEASE:
				if (sched.process(item)) {
					// release is to execute: we need to flush current queue -> end current batch and add signal
					auto parm = node->release.src;
					auto src_stream = sched.executed.at(parm.node).stream;
					auto& link = impl->res_to_links[parm];
					DomainFlagBits src_domain = src_stream->domain;
					Stream* dst_stream;
					if (node->release.dst_domain == DomainFlagBits::ePE) {
						auto& link = sched.res_to_links[node->release.src];
						auto& swp = sched.get_value<Swapchain>(link.urdef.node->acquire_next_image.swapchain);
						auto it = std::find_if(pe_streams.begin(), pe_streams.end(), [&](auto& pe_stream) { return pe_stream.swp == &swp; });
						assert(it != pe_streams.end());
						dst_stream = &*it;
					} else if (node->release.dst_domain == DomainFlagBits::eAny) {
						dst_stream = src_stream;
					} else {
						dst_stream = recorder.stream_for_domain(node->release.dst_domain);
					}
					assert(dst_stream);
					DomainFlagBits dst_domain = dst_stream->domain;

					Type* parm_ty = parm.type();
					auto di = sched.get_dependency_info(parm, parm_ty, RW::eWrite, dst_stream, Access::eNone, node->release.dst_access);
					if (node->release.dst_access != Access::eNone) {
						recorder.add_sync(sched.base_type(parm), di, sched.get_value(parm));
					}
#ifdef VUK_DUMP_EXEC
					print_results(node);
					fmt::print("release ${}->${} ", domain_to_string(src_domain), domain_to_string(dst_domain));
					print_args(std::span{ &node->release.src, 1 });
					fmt::print("\n");
#endif
					auto& acqrel = node->release.release;
					acqrel->last_use = di.src_use;
					if (src_domain == DomainFlagBits::eHost) {
						acqrel->status = Signal::Status::eHostAvailable;
					}
					if (dst_domain == DomainFlagBits::ePE) {
						auto& link = sched.res_to_links[node->release.src];
						auto& swp = sched.get_value<Swapchain>(link.urdef.node->acquire_next_image.swapchain);
						assert(src_stream->domain & DomainFlagBits::eDevice);
						auto result = dynamic_cast<VkQueueStream*>(src_stream)->present(swp);
						// TODO: do something with the result here
					}

					recorder.flush_domain(src_domain, acqrel);
					fmt::print("");
					sched.done(node, src_stream);
				} else {
					sched.schedule_dependency(node->release.src, RW::eWrite);
				}
				break;

			case Node::ACQUIRE_NEXT_IMAGE: {
				if (sched.process(item)) {
					auto& swp = sched.get_value<Swapchain>(node->acquire_next_image.swapchain);
					swp.linear_index = (swp.linear_index + 1) % swp.images.size();
					swp.acquire_result =
					    ctx.vkAcquireNextImageKHR(ctx.device, swp.swapchain, UINT64_MAX, swp.semaphores[2 * swp.linear_index], VK_NULL_HANDLE, &swp.image_index);
					// VK_SUBOPTIMAL_KHR shouldn't stop presentation; it is handled at the end
					if (swp.acquire_result != VK_SUCCESS && swp.acquire_result != VK_SUBOPTIMAL_KHR) {
						return { expected_error, VkException{ swp.acquire_result } };
					}

					auto pe_stream = &pe_streams.emplace_back(alloc, swp);
#ifdef VUK_DUMP_EXEC
					print_results(node);
					fmt::print(" = acquire_next_image ");
					print_args(std::span{ &node->acquire_next_image.swapchain, 1 });
					fmt::print("\n");
#endif
					sched.done(node, pe_stream);
				} else {
					sched.schedule_dependency(node->acquire_next_image.swapchain, RW::eWrite);
				}
				break;
			}
			case Node::INDEXING:
				if (sched.process(item)) {
					// half sync
					std::vector<Buffer*> bufs;
					auto& link = sched.res_to_links[node->indexing.array];
					assert(link.urdef.node->kind == Node::AALLOC);
					auto size = link.urdef.type()->array.size;
					for (auto i = 0; i < size; i++) {
						bufs.push_back(&sched.get_value<Buffer>(link.urdef.node->aalloc.args[i + 1]));
					}

					recorder.add_sync(sched.base_type(node->indexing.array),
					                  sched.get_dependency_info(node->indexing.array, node->indexing.array.type(), RW::eWrite, nullptr),
					                  bufs.data());
#ifdef VUK_DUMP_EXEC
					print_results(node);
					fmt::print(" = ");
					print_args(std::span{ &node->indexing.array, 1 });
					fmt::print("[{}]", constant<uint64_t>(node->indexing.index));
					fmt::print("\n");
#endif
					sched.done(node, nullptr); // indexing doesn't execute
				} else {
					sched.schedule_dependency(node->indexing.array, RW::eWrite);
					sched.schedule_dependency(node->indexing.index, RW::eRead);
				}
				break;
			default:
				assert(0);
			}
		}

		/* INFERENCE
		// pre-inference: which IAs are in which FBs?
		for (auto& rp : impl->rpis) {
		  for (auto& rp_att : rp.attachments.to_span(impl->rp_infos)) {
		    auto& att = *rp_att.attachment_info;

		    att.rp_uses.append(impl->attachment_rp_references, &rp);
		    auto& ia = att.attachment;
		    ia.image_type = ia.image_type == ImageType::eInfer ? vuk::ImageType::e2D : ia.image_type;

		    ia.base_layer = ia.base_layer == VK_REMAINING_ARRAY_LAYERS ? 0 : ia.base_layer;
		    ia.layer_count =
		        ia.layer_count == VK_REMAINING_ARRAY_LAYERS ? 1 : ia.layer_count; // TODO: this prevents inference later on, so this means we are doing it too
		early ia.base_level = ia.base_level == VK_REMAINING_MIP_LEVELS ? 0 : ia.base_level;

		    if (ia.view_type == ImageViewType::eInfer) {
		      if (ia.layer_count > 1) {
		        ia.view_type = ImageViewType::e2DArray;
		      } else {
		        ia.view_type = ImageViewType::e2D;
		      }
		    }

		    ia.level_count = 1; // can only render to a single mip level
		    ia.extent.extent.depth = 1;

		    // we do an initial IA -> FB, because we won't process complete IAs later, but we need their info
		    if (ia.sample_count != Samples::eInfer && !rp_att.is_resolve_dst) {
		      rp.fbci.sample_count = ia.sample_count;
		    }

		    if (ia.extent.sizing == Sizing::eAbsolute && ia.extent.extent.width > 0 && ia.extent.extent.height > 0) {
		      rp.fbci.width = ia.extent.extent.width;
		      rp.fbci.height = ia.extent.extent.height;
		    }

		    // resolve images are always sample count 1
		    if (rp_att.is_resolve_dst) {
		      att.attachment.sample_count = Samples::e1;
		    }
		  }
		}

		decltype(impl->ia_inference_rules) ia_resolved_rules;
		for (auto& [n, rules] : impl->ia_inference_rules) {
		  ia_resolved_rules.emplace(impl->resolve_name(n), std::move(rules));
		}

		decltype(impl->buf_inference_rules) buf_resolved_rules;
		for (auto& [n, rules] : impl->buf_inference_rules) {
		  buf_resolved_rules.emplace(impl->resolve_name(n), std::move(rules));
		}

		std::vector<std::pair<AttachmentInfo*, IAInferences*>> attis_to_infer;
		for (auto& bound : impl->bound_attachments) {
		  if (bound.type == AttachmentInfo::Type::eInternal && bound.parent_attachment == 0) {
		    // compute usage if it is to be inferred
		    if (bound.attachment.usage == ImageUsageFlagBits::eInfer) {
		      bound.attachment.usage = {};
		      for (auto& chain : bound.use_chains.to_span(impl->attachment_use_chain_references)) {
		        bound.attachment.usage |= impl->compute_usage(chain);
		      }
		    }
		    // if there is no image, then we will infer the base mip and layer to be 0
		    if (!bound.attachment.image) {
		      bound.attachment.base_layer = 0;
		      bound.attachment.base_level = 0;
		    }
		    if (bound.attachment.image_view == ImageView{}) {
		      if (bound.attachment.view_type == ImageViewType::eInfer && bound.attachment.layer_count != VK_REMAINING_ARRAY_LAYERS) {
		        if (bound.attachment.image_type == ImageType::e1D) {
		          if (bound.attachment.layer_count == 1) {
		            bound.attachment.view_type = ImageViewType::e1D;
		          } else {
		            bound.attachment.view_type = ImageViewType::e1DArray;
		          }
		        } else if (bound.attachment.image_type == ImageType::e2D) {
		          if (bound.attachment.layer_count == 1) {
		            bound.attachment.view_type = ImageViewType::e2D;
		          } else {
		            bound.attachment.view_type = ImageViewType::e2DArray;
		          }
		        } else if (bound.attachment.image_type == ImageType::e3D) {
		          if (bound.attachment.layer_count == 1) {
		            bound.attachment.view_type = ImageViewType::e3D;
		          } else {
		            bound.attachment.view_type = ImageViewType::e2DArray;
		          }
		        }
		      }
		    }
		    IAInferences* rules_ptr = nullptr;
		    auto rules_it = ia_resolved_rules.find(bound.name);
		    if (rules_it != ia_resolved_rules.end()) {
		      rules_ptr = &rules_it->second;
		    }

		    attis_to_infer.emplace_back(&bound, rules_ptr);
		  }
		}

		std::vector<std::pair<BufferInfo*, BufferInferences*>> bufis_to_infer;
		for (auto& bound : impl->bound_buffers) {
		  if (bound.buffer.size != ~(0u))
		    continue;

		  BufferInferences* rules_ptr = nullptr;
		  auto rules_it = buf_resolved_rules.find(bound.name);
		  if (rules_it != buf_resolved_rules.end()) {
		    rules_ptr = &rules_it->second;
		  }

		  bufis_to_infer.emplace_back(&bound, rules_ptr);
		}

		InferenceContext inf_ctx{ this };
		bool infer_progress = true;
		std::stringstream msg;

		// we provide an upper bound of 100 inference iterations to catch infinite loops that don't converge to a fixpoint
		for (size_t i = 0; i < 100 && !attis_to_infer.empty() && infer_progress; i++) {
		  infer_progress = false;
		  for (auto ia_it = attis_to_infer.begin(); ia_it != attis_to_infer.end();) {
		    auto& atti = *ia_it->first;
		    auto& ia = atti.attachment;
		    auto prev = ia;
		    // infer FB -> IA
		    if (ia.sample_count == Samples::eInfer || (ia.extent.extent.width == 0 && ia.extent.extent.height == 0) ||
		        ia.extent.sizing == Sizing::eRelative) { // this IA can potentially take inference from an FB
		      for (auto* rpi : atti.rp_uses.to_span(impl->attachment_rp_references)) {
		        auto& fbci = rpi->fbci;
		        Samples fb_samples = fbci.sample_count;
		        bool samples_known = fb_samples != Samples::eInfer;

		        // an extent is known if it is not 0
		        // 0 sized framebuffers are illegal
		        Extent3D fb_extent = { fbci.width, fbci.height };
		        bool extent_known = !(fb_extent.width == 0 || fb_extent.height == 0);

		        if (samples_known && ia.sample_count == Samples::eInfer) {
		          ia.sample_count = fb_samples;
		        }

		        if (extent_known) {
		          if (ia.extent.extent.width == 0 && ia.extent.extent.height == 0) {
		            ia.extent.extent.width = fb_extent.width;
		            ia.extent.extent.height = fb_extent.height;
		          } else if (ia.extent.sizing == Sizing::eRelative) {
		            ia.extent.extent.width = static_cast<uint32_t>(ia.extent._relative.width * fb_extent.width);
		            ia.extent.extent.height = static_cast<uint32_t>(ia.extent._relative.height * fb_extent.height);
		            ia.extent.extent.depth = static_cast<uint32_t>(ia.extent._relative.depth * fb_extent.depth);
		          }
		          ia.extent.sizing = Sizing::eAbsolute;
		        }
		      }
		    }
		    // infer custom rule -> IA
		    if (ia_it->second) {
		      inf_ctx.prefix = ia_it->second->prefix;
		      for (auto& rule : ia_it->second->rules) {
		        rule(inf_ctx, ia);
		      }
		    }
		    if (prev != ia) { // progress made
		      // check for broken constraints
		      if (prev.base_layer != ia.base_layer && prev.base_layer != VK_REMAINING_ARRAY_LAYERS) {
		        msg << "Rule broken for attachment[" << atti.name.name.c_str() << "] :\n ";
		        msg << " base layer was previously known to be " << prev.base_layer << ", but now set to " << ia.base_layer;
		        return { expected_error, RenderGraphException{ msg.str() } };
		      }
		      if (prev.layer_count != ia.layer_count && prev.layer_count != VK_REMAINING_ARRAY_LAYERS) {
		        msg << "Rule broken for attachment[" << atti.name.name.c_str() << "] :\n ";
		        msg << " layer count was previously known to be " << prev.layer_count << ", but now set to " << ia.layer_count;
		        return { expected_error, RenderGraphException{ msg.str() } };
		      }
		      if (prev.base_level != ia.base_level && prev.base_level != VK_REMAINING_MIP_LEVELS) {
		        msg << "Rule broken for attachment[" << atti.name.name.c_str() << "] :\n ";
		        msg << " base level was previously known to be " << prev.base_level << ", but now set to " << ia.base_level;
		        return { expected_error, RenderGraphException{ msg.str() } };
		      }
		      if (prev.level_count != ia.level_count && prev.level_count != VK_REMAINING_MIP_LEVELS) {
		        msg << "Rule broken for attachment[" << atti.name.name.c_str() << "] :\n ";
		        msg << " level count was previously known to be " << prev.level_count << ", but now set to " << ia.level_count;
		        return { expected_error, RenderGraphException{ msg.str() } };
		      }
		      if (prev.format != ia.format && prev.format != Format::eUndefined) {
		        msg << "Rule broken for attachment[" << atti.name.name.c_str() << "] :\n ";
		        msg << " format was previously known to be " << format_to_sv(prev.format) << ", but now set to " << format_to_sv(ia.format);
		        return { expected_error, RenderGraphException{ msg.str() } };
		      }
		      if (prev.sample_count != ia.sample_count && prev.sample_count != SampleCountFlagBits::eInfer) {
		        msg << "Rule broken for attachment[" << atti.name.name.c_str() << "] :\n ";
		        msg << " sample count was previously known to be " << static_cast<uint32_t>(prev.sample_count.count) << ", but now set to "
		            << static_cast<uint32_t>(ia.sample_count.count);
		        return { expected_error, RenderGraphException{ msg.str() } };
		      }
		      if (prev.extent.extent.width != ia.extent.extent.width && prev.extent.extent.width != 0) {
		        msg << "Rule broken for attachment[" << atti.name.name.c_str() << "] :\n ";
		        msg << " extent.width was previously known to be " << prev.extent.extent.width << ", but now set to " << ia.extent.extent.width;
		        return { expected_error, RenderGraphException{ msg.str() } };
		      }
		      if (prev.extent.extent.height != ia.extent.extent.height && prev.extent.extent.height != 0) {
		        msg << "Rule broken for attachment[" << atti.name.name.c_str() << "] :\n ";
		        msg << " extent.height was previously known to be " << prev.extent.extent.height << ", but now set to " << ia.extent.extent.height;
		        return { expected_error, RenderGraphException{ msg.str() } };
		      }
		      if (prev.extent.extent.depth != ia.extent.extent.depth && prev.extent.extent.depth != 0) {
		        msg << "Rule broken for attachment[" << atti.name.name.c_str() << "] :\n ";
		        msg << " extent.depth was previously known to be " << prev.extent.extent.depth << ", but now set to " << ia.extent.extent.depth;
		        return { expected_error, RenderGraphException{ msg.str() } };
		      }
		      if (ia.may_require_image_view() && prev.view_type != ia.view_type && prev.view_type != ImageViewType::eInfer) {
		        msg << "Rule broken for attachment[" << atti.name.name.c_str() << "] :\n ";
		        msg << " view type was previously known to be " << image_view_type_to_sv(prev.view_type) << ", but now set to "
		            << image_view_type_to_sv(ia.view_type);
		        return { expected_error, RenderGraphException{ msg.str() } };
		      }

		      infer_progress = true;
		      // infer IA -> FB
		      if (ia.sample_count == Samples::eInfer && (ia.extent.extent.width == 0 && ia.extent.extent.height == 0)) { // this IA is not helpful for FB
		inference continue;
		      }
		      for (auto* rpi : atti.rp_uses.to_span(impl->attachment_rp_references)) {
		        auto& fbci = rpi->fbci;
		        Samples fb_samples = fbci.sample_count;
		        bool samples_known = fb_samples != Samples::eInfer;

		        // an extent is known if it is not 0
		        // 0 sized framebuffers are illegal
		        Extent2D fb_extent = Extent2D{ fbci.width, fbci.height };
		        bool extent_known = !(fb_extent.width == 0 || fb_extent.height == 0);

		        if (samples_known && extent_known) {
		          continue;
		        }

		        if (!samples_known && ia.sample_count != Samples::eInfer) {
		          auto attachments = rpi->attachments.to_span(impl->rp_infos);
		          auto it = std::find_if(attachments.begin(), attachments.end(), [attip = &atti](auto& rp_att) { return rp_att.attachment_info == attip; });
		          assert(it != attachments.end());
		          if (!it->is_resolve_dst) {
		            fbci.sample_count = ia.sample_count;
		          }
		        }

		        if (ia.extent.sizing == vuk::Sizing::eAbsolute && ia.extent.extent.width > 0 && ia.extent.extent.height > 0) {
		          fbci.width = ia.extent.extent.width;
		          fbci.height = ia.extent.extent.height;
		        }
		      }
		    }
		    if (ia.is_fully_known()) {
		      ia_it = attis_to_infer.erase(ia_it);
		    } else {
		      ++ia_it;
		    }
		  }
		}

		for (auto& [atti, iaref] : attis_to_infer) {
		  msg << "Could not infer attachment [" << atti->name.name.c_str() << "]:\n";
		  auto& ia = atti->attachment;
		  if (ia.sample_count == Samples::eInfer) {
		    msg << "- sample count unknown\n";
		  }
		  if (ia.extent.sizing == Sizing::eRelative) {
		    msg << "- relative sizing could not be resolved\n";
		  }
		  if (ia.extent.extent.width == 0) {
		    msg << "- extent.width unknown\n";
		  }
		  if (ia.extent.extent.height == 0) {
		    msg << "- extent.height unknown\n";
		  }
		  if (ia.extent.extent.depth == 0) {
		    msg << "- extent.depth unknown\n";
		  }
		  if (ia.format == Format::eUndefined) {
		    msg << "- format unknown\n";
		  }
		  if (ia.may_require_image_view() && ia.view_type == ImageViewType::eInfer) {
		    msg << "- view type unknown\n";
		  }
		  if (ia.base_layer == VK_REMAINING_ARRAY_LAYERS) {
		    msg << "- base layer unknown\n";
		  }
		  if (ia.layer_count == VK_REMAINING_ARRAY_LAYERS) {
		    msg << "- layer count unknown\n";
		  }
		  if (ia.base_level == VK_REMAINING_MIP_LEVELS) {
		    msg << "- base level unknown\n";
		  }
		  if (ia.level_count == VK_REMAINING_MIP_LEVELS) {
		    msg << "- level count unknown\n";
		  }
		  msg << "\n";
		}

		if (attis_to_infer.size() > 0) {
		  return { expected_error, RenderGraphException{ msg.str() } };
		}

		infer_progress = true;
		// we provide an upper bound of 100 inference iterations to catch infinite loops that don't converge to a fixpoint
		for (size_t i = 0; i < 100 && !bufis_to_infer.empty() && infer_progress; i++) {
		  infer_progress = false;
		  for (auto bufi_it = bufis_to_infer.begin(); bufi_it != bufis_to_infer.end();) {
		    auto& bufi = *bufi_it->first;
		    auto& buff = bufi.buffer;
		    auto prev = buff;

		    // infer custom rule -> IA
		    if (bufi_it->second) {
		      inf_ctx.prefix = bufi_it->second->prefix;
		      for (auto& rule : bufi_it->second->rules) {
		        rule(inf_ctx, buff);
		      }
		    }
		    if (prev != buff) { // progress made
		      // check for broken constraints
		      if (prev.size != buff.size && prev.size != ~(0u)) {
		        msg << "Rule broken for buffer[" << bufi.name.name.c_str() << "] :\n ";
		        msg << " size was previously known to be " << prev.size << ", but now set to " << buff.size;
		        return { expected_error, RenderGraphException{ msg.str() } };
		      }

		      infer_progress = true;
		    }
		    if (buff.size != ~(0u)) {
		      bufi_it = bufis_to_infer.erase(bufi_it);
		    } else {
		      ++bufi_it;
		    }
		  }
		}

		for (auto& [buff, bufinfs] : bufis_to_infer) {
		  msg << "Could not infer buffer [" << buff->name.name.c_str() << "]:\n";
		  if (buff->buffer.size == ~(0u)) {
		    msg << "- size unknown\n";
		  }
		  msg << "\n";
		}

		if (bufis_to_infer.size() > 0) {
		  return { expected_error, RenderGraphException{ msg.str() } };
		}
		*/

		return { expected_value };
	}
} // namespace vuk
