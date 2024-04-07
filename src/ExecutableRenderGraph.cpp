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

//#define VUK_DUMP_EXEC
//#define VUK_DEBUG_IMBAR

namespace vuk {
	std::string format_source_location(Node* node) {
		if (node->debug_info) {
			std::string msg = "";
			for (int i = (int)node->debug_info->trace.size() - 1; i >= 0; i--) {
				auto& source = node->debug_info->trace[i];
				msg += fmt::format("{}({}): ", source.file_name(), source.line());
				if (i > 0) {
					msg += "\n";
				}
			}
			return msg;
		} else {
			return "?: ";
		}
	}
} // namespace vuk

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

	std::optional<Subrange::Image> intersect(Subrange::Image a, Subrange::Image b) {
		Subrange::Image result;
		result.base_layer = std::max(a.base_layer, b.base_layer);
		int count;
		if (a.layer_count == VK_REMAINING_ARRAY_LAYERS) {
			count = static_cast<int32_t>(b.layer_count) + std::min(0, static_cast<int32_t>(result.base_layer) - static_cast<int32_t>(b.base_layer));
		} else if (b.layer_count == VK_REMAINING_ARRAY_LAYERS) {
			count = static_cast<int32_t>(a.layer_count) + std::min(0, static_cast<int32_t>(result.base_layer) - static_cast<int32_t>(a.base_layer));
		} else {
			count = static_cast<int32_t>(std::min(a.base_layer + a.layer_count, b.base_layer + b.layer_count)) - static_cast<int32_t>(result.base_layer);
		}
		if (count < 1) {
			return {};
		}
		result.layer_count = static_cast<uint32_t>(count);

		result.base_level = std::max(a.base_level, b.base_level);
		if (a.level_count == VK_REMAINING_MIP_LEVELS) {
			count = static_cast<int32_t>(b.level_count) + std::min(0, static_cast<int32_t>(result.base_level) - static_cast<int32_t>(b.base_level));
		} else if (b.level_count == VK_REMAINING_MIP_LEVELS) {
			count = static_cast<int32_t>(a.level_count) + std::min(0, static_cast<int32_t>(result.base_level) - static_cast<int32_t>(a.base_level));
		} else {
			count = static_cast<int32_t>(std::min(a.base_level + a.level_count, b.base_level + b.level_count)) - static_cast<int32_t>(result.base_level);
		}
		if (count < 1) {
			return {};
		}
		result.level_count = static_cast<uint32_t>(count);

		return result;
	}

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

		Signal* make_signal() override {
			return &signals.emplace_back();
		}

		void sync_deps() override {
			if (batch.empty()) {
				batch.emplace_back();
			}
			for (auto dep : dependencies) {
				auto signal = dep->make_signal();
				if (signal) {
					dep->add_dependent_signal(signal);
				}
				auto res = *dep->submit();
				if (signal) {
					batch.back().waits.push_back(signal);
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

		Result<SubmitResult> submit() override {
			sync_deps();
			end_cbuf();
			for (auto& signal : dependent_signals) {
				signal->source.executor = executor;
				batch.back().signals.emplace_back(signal);
			}
			executor->submit_batch(batch);
			for (auto& item : batch) {
				for (auto& signal : item.signals) {
					alloc.wait_sync_points(std::span{ &signal->source, 1 });
				}
			}
			batch.clear();
			dependent_signals.clear();
			return { expected_value };
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

			return { expected_value };
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

		void print_ib(VkImageMemoryBarrier2KHR ib, std::string extra = "") {
			auto layout_to_str = [](VkImageLayout l) {
				switch (l) {
				case VK_IMAGE_LAYOUT_UNDEFINED:
					return "UND";
				case VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL:
					return "SRC";
				case VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL:
					return "DST";
				case VK_IMAGE_LAYOUT_GENERAL:
					return "GEN";
				case VK_IMAGE_LAYOUT_READ_ONLY_OPTIMAL:
					return "ROO";
				case VK_IMAGE_LAYOUT_ATTACHMENT_OPTIMAL:
					return "ATT";
				case VK_IMAGE_LAYOUT_PRESENT_SRC_KHR:
					return "PRS";
				}
				assert(0);
			};
			fmt::println("[{}][m{}:{}][l{}:{}][{}->{}]{}",
			             fmt::ptr(ib.image),
			             ib.subresourceRange.baseMipLevel,
			             ib.subresourceRange.baseMipLevel + ib.subresourceRange.levelCount - 1,
			             ib.subresourceRange.baseArrayLayer,
			             ib.subresourceRange.baseArrayLayer + ib.subresourceRange.layerCount - 1,
			             layout_to_str(ib.oldLayout),
			             layout_to_str(ib.newLayout),
			             extra);
		}

		bool is_readonly_layout(VkImageLayout l) {
			switch (l) {
			case VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL:
			case VK_IMAGE_LAYOUT_READ_ONLY_OPTIMAL:
				return true;
			}
			return false;
		}

		void synch_image(ImageAttachment& img_att, StreamResourceUse src_use, StreamResourceUse dst_use, void* tag, bool init_allowed) override {
			auto aspect = format_to_aspect(img_att.format);

			// if we start an RP and we have LOAD_OP_LOAD (currently always), then we must upgrade access with an appropriate READ
			if (is_framebuffer_attachment(dst_use)) {
				if ((aspect & ImageAspectFlagBits::eColor) == ImageAspectFlags{}) { // not color -> depth or depth/stencil
					dst_use.access |= vuk::AccessFlagBits::eDepthStencilAttachmentRead;
				} else {
					dst_use.access |= vuk::AccessFlagBits::eColorAttachmentRead;
				}
			}

			if (src_use == ResourceUse{} && is_readonly_access(dst_use) && img_att.layout == dst_use.layout) { // an already synchronized read
				return;
			}

			DomainFlagBits src_domain = src_use.stream ? src_use.stream->domain : DomainFlagBits::eNone;
			DomainFlagBits dst_domain = dst_use.stream ? dst_use.stream->domain : DomainFlagBits::eNone;

			scope_to_domain((VkPipelineStageFlagBits2KHR&)src_use.stages, src_domain & DomainFlagBits::eQueueMask);
			scope_to_domain((VkPipelineStageFlagBits2KHR&)dst_use.stages, dst_domain & DomainFlagBits::eQueueMask);

			// compute image barrier for this access -> access
			VkImageMemoryBarrier2KHR barrier{ .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2_KHR };
			barrier.srcAccessMask = is_readonly_access(src_use) ? 0 : (VkAccessFlags)src_use.access;
			barrier.dstAccessMask = (VkAccessFlags)dst_use.access;
			barrier.oldLayout = (VkImageLayout)src_use.layout;
			barrier.newLayout = (VkImageLayout)dst_use.layout;
			barrier.subresourceRange.aspectMask = (VkImageAspectFlags)aspect;
			barrier.subresourceRange.baseArrayLayer = img_att.base_layer;
			barrier.subresourceRange.baseMipLevel = img_att.base_level;
			barrier.subresourceRange.layerCount = img_att.layer_count;
			barrier.subresourceRange.levelCount = img_att.level_count;

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

#ifdef VUK_DEBUG_IMBAR
			fmt::println("----------------------------");
#endif
			barrier.image = img_att.image.image;

			auto add_half = [this, dst_use](VkImageMemoryBarrier2 barrier, ImageAttachment& img_att) {
				Subrange::Image a_range{ img_att.base_level, img_att.level_count, img_att.base_layer, img_att.layer_count };
				for (auto& b : half_im_bars) {
					auto b_range = Subrange::Image{
						b.subresourceRange.baseMipLevel, b.subresourceRange.levelCount, b.subresourceRange.baseArrayLayer, b.subresourceRange.layerCount
					};
					// duplicate
					if (b.image == barrier.image && intersect(a_range, b_range)) {
#ifdef VUK_DEBUG_IMBAR
						print_ib(barrier, "!!");
#endif
						return;
					}
				}
#ifdef VUK_DEBUG_IMBAR
				print_ib(barrier, "+");
#endif
				half_im_bars.push_back(barrier);
			};

			auto add_full = [this, init_allowed](VkImageMemoryBarrier2 barrier, ImageAttachment& img_att) {
				assert(img_att.layout == ImageLayout::eUndefined || barrier.oldLayout != VK_IMAGE_LAYOUT_UNDEFINED || init_allowed);
				assert(barrier.oldLayout != VK_IMAGE_LAYOUT_UNDEFINED || !is_readonly_layout(barrier.newLayout));
				im_bars.push_back(barrier);

				img_att.layout = (ImageLayout)barrier.newLayout;
				if (barrier.oldLayout != VK_IMAGE_LAYOUT_UNDEFINED) {
					assert(barrier.newLayout != VK_IMAGE_LAYOUT_UNDEFINED);
				}
			};

			if (dst_domain == DomainFlagBits::eNone) {
				add_half(barrier, img_att);
			} else if (src_domain == DomainFlagBits::eNone) {
				VkImageMemoryBarrier2KHR found;
				std::vector<Subrange::Image> work_queue;
				work_queue.emplace_back(Subrange::Image{ img_att.base_level, img_att.level_count, img_att.base_layer, img_att.layer_count });

				while (work_queue.size() > 0) {
					Subrange::Image b = work_queue.back();
					work_queue.pop_back();
					auto it = half_im_bars.rbegin();
					vuk::Subrange::Image a, isection;
					// locate matching half barrier
					do {
						it = std::find_if(it, half_im_bars.rend(), [&](auto& mb) { return mb.image == img_att.image.image; });
						// if there isn't a matching first bar, but this readonly, then this sync can be skipped
						// we might in the wrong layout for reconverged resources (but this might be bug)
						// lets hope for the best!
						if (it == half_im_bars.rend() && is_readonly_access(dst_use)) {
							// assert(img_att.layout == dst_use.layout);
							img_att.layout = dst_use.layout;
							if (is_framebuffer_attachment(dst_use)) {
								prepare_render_pass_attachment(alloc, img_att);
							}
							return;
						}
						assert(it != half_im_bars.rend());
						found = *it;
						a = {
							found.subresourceRange.baseMipLevel, found.subresourceRange.levelCount, found.subresourceRange.baseArrayLayer, found.subresourceRange.layerCount
						};

						// we want to make a barrier for the intersection of the existing and incoming
						auto isection_opt = intersect(a, b);
						if (isection_opt) {
							isection = *isection_opt;
							break;
						}
						++it;
					} while (it != half_im_bars.rend());
					assert(it != half_im_bars.rend());
#ifdef VUK_DEBUG_IMBAR
					for (auto it2 = half_im_bars.rbegin(); it2 != half_im_bars.rend(); ++it2) {
						if (it != it2) {
							print_ib(*it2, "");
						}
					}
#endif
					half_im_bars.erase(std::next(it).base());

					barrier.subresourceRange.baseArrayLayer = isection.base_layer;
					barrier.subresourceRange.baseMipLevel = isection.base_level;
					barrier.subresourceRange.layerCount = isection.layer_count;
					barrier.subresourceRange.levelCount = isection.level_count;

					// then we want to remove the existing barrier from the candidates and put barrier(s) that are the difference
					auto difference = [](Subrange::Image a, Subrange::Image isection) {
						std::vector<Subrange::Image> new_srs;
						// before, mips
						if (isection.base_level > a.base_level) {
							new_srs.push_back(
							    { .base_level = a.base_level, .level_count = isection.base_level - a.base_level, .base_layer = a.base_layer, .layer_count = a.layer_count });
						}
						// after, mips
						if (a.base_level + a.level_count > isection.base_level + isection.level_count) {
							new_srs.push_back({ .base_level = isection.base_level + isection.level_count,
							                    .level_count = a.base_level + a.level_count - (isection.base_level + isection.level_count),
							                    .base_layer = a.base_layer,
							                    .layer_count = a.layer_count });
						}
						// before, layers
						if (isection.base_layer > a.base_layer) {
							new_srs.push_back(
							    { .base_level = a.base_level, .level_count = a.level_count, .base_layer = a.base_layer, .layer_count = isection.base_layer - a.base_layer });
						}
						// after, layers
						if (a.base_layer + a.layer_count > isection.base_layer + isection.layer_count) {
							new_srs.push_back({
							    .base_level = a.base_level,
							    .level_count = a.level_count,
							    .base_layer = isection.base_layer + isection.layer_count,
							    .layer_count = a.base_layer + a.layer_count - (isection.base_layer + isection.layer_count),
							});
						}

						return new_srs;
					};

					auto new_srs = difference(a, isection);
					// push the splintered src half barrier(s)
					for (auto& nb : new_srs) {
						auto barrier = found;

						barrier.subresourceRange.baseArrayLayer = nb.base_layer;
						barrier.subresourceRange.baseMipLevel = nb.base_level;
						barrier.subresourceRange.layerCount = nb.layer_count;
						barrier.subresourceRange.levelCount = nb.level_count;

						add_half(barrier, img_att);
					}

					// splinter the dst barrier, and push into the work queue
					auto new_dst = difference(b, isection);
					work_queue.insert(work_queue.end(), new_dst.begin(), new_dst.end());

					// fill out src part of the barrier based on the sync half we found
					barrier.pNext = nullptr;
					barrier.srcAccessMask = found.srcAccessMask;
					barrier.srcStageMask = found.srcStageMask;
					barrier.srcQueueFamilyIndex = found.srcQueueFamilyIndex;
					barrier.oldLayout = found.oldLayout;
#ifdef VUK_DEBUG_IMBAR
					print_ib(barrier, "*");
#endif
					add_full(barrier, img_att);
				}
			} else {
#ifdef VUK_DEBUG_IMBAR
				print_ib(barrier, "$");
#endif
				add_full(barrier, img_att);
			}
		};

		void synch_memory(StreamResourceUse src_use, StreamResourceUse dst_use, void* tag) override {
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

			barrier.srcAccessMask = is_readonly_access(src_use) ? 0 : (VkAccessFlags)src_use.access;
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
			descr.storeOp = is_readonly_layout((VkImageLayout)img_att.layout) ? VK_ATTACHMENT_STORE_OP_NONE_KHR : VK_ATTACHMENT_STORE_OP_STORE;

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
			rp.fbci.width = img_att.extent.width;
			rp.fbci.height = img_att.extent.height;
			rp.fbci.layers = img_att.layer_count;
			assert(img_att.level_count == 1);
			rp.fbci.sample_count = img_att.sample_count;
			rp.fbci.attachments.push_back(img_att.image_view);
		}

		Result<void> prepare_render_pass() {
			SubpassDescription sd;
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

		void synch_image(ImageAttachment& img_att, StreamResourceUse src_use, StreamResourceUse dst_use, void* tag, bool) override {
			/* host -> host and host -> device not needed, device -> host inserts things on the device side */
			return;
		}
		void synch_memory(StreamResourceUse src_use, StreamResourceUse dst_use, void* tag) override {
			/* host -> host and host -> device not needed, device -> host inserts things on the device side */
			return;
		}

		Signal* make_signal() override {
			return nullptr;
		}

		Result<SubmitResult> submit() override {
			return { expected_value };
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

		void synch_image(ImageAttachment& img_att, StreamResourceUse src_use, StreamResourceUse dst_use, void* tag, bool) override {}

		void synch_memory(StreamResourceUse src_use, StreamResourceUse dst_use, void* tag) override { /* PE doesn't do memory */
			assert(false);
		}

		Signal* make_signal() override {
			return nullptr;
		}

		Result<SubmitResult> submit() override {
			assert(swp);
			SubmitResult sr{ .sema_wait = swp->semaphores[2 * swp->linear_index] };
			return { expected_value, sr };
		}
	};

	enum class RW { eRead, eWrite };

	struct Scheduler {
		Scheduler(Allocator all, RGCImpl* impl) :
		    allocator(all),
		    scheduled_execables(impl->scheduled_execables),
		    pass_reads(impl->pass_reads),
		    executed(impl->executed),
		    impl(impl) {
			// these are the items that were determined to run
			for (auto& i : scheduled_execables) {
				scheduled.emplace(i.execable);
				work_queue.emplace_back(i);
			}
		}

		Allocator allocator;
		std::vector<Ref>& pass_reads;
		std::vector<ScheduledItem>& scheduled_execables;

		InlineArena<std::byte, 4 * 1024> arena;

		RGCImpl* impl;

		std::deque<ScheduledItem> work_queue;

		size_t naming_index_counter = 0;
		size_t instr_counter = 0;
		std::unordered_map<Node*, ExecutionInfo>& executed;
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
			if (parm.node->kind == Node::CONSTANT || parm.node->kind == Node::PLACEHOLDER || parm.node->kind == Node::MATH_BINARY) {
				return;
			}
			auto link = parm.link();

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

		template<class T>
		  requires(!std::is_same_v<T, void*> && !std::is_same_v<T, std::span<void*>>)
		void done(Node* node, Stream* stream, T value) {
			auto counter = naming_index_counter;
			naming_index_counter += node->type.size();
			auto value_ptr = static_cast<void*>(new (arena.ensure_space(sizeof(T))) T{ value });
			auto values = new (arena.ensure_space(sizeof(void* [1]))) void*[1];
			values[0] = value_ptr;
			executed.emplace(node, ExecutionInfo{ stream, counter, std::span{ values, 1 } });
		}

		void done(Node* node, Stream* stream, void* value_ptr) {
			auto counter = naming_index_counter;
			naming_index_counter += node->type.size();
			auto values = new (arena.ensure_space(sizeof(void* [1]))) void*[1];
			values[0] = value_ptr;
			executed.emplace(node, ExecutionInfo{ stream, counter, std::span{ values, 1 } });
		}

		void done(Node* node, Stream* stream, std::span<void*> values) {
			auto counter = naming_index_counter;
			naming_index_counter += node->type.size();
			auto v = new (arena.ensure_space(sizeof(void*) * values.size())) void*[values.size()];
			std::copy(values.begin(), values.end(), v);
			executed.emplace(node, ExecutionInfo{ stream, counter, std::span{ v, values.size() } });
		}

		template<class T>
		T& get_value(Ref parm) {
			return *reinterpret_cast<T*>(get_value(parm));
		};

		void* get_value(Ref parm) {
			auto v = impl->get_value(parm);
			return v;
		}

		std::span<void*> get_values(Node* node) {
			auto v = impl->get_values(node);
			return v;
		}

		Type* base_type(Ref parm) {
			return Type::stripped(parm.type());
		}

		struct DependencyInfo {
			bool init_permitted;
			StreamResourceUse src_use;
			StreamResourceUse dst_use;
		};

		DependencyInfo
		get_dependency_info(Ref parm, Type* arg_ty, RW type, Stream* dst_stream, Access src_access = Access::eNone, Access dst_access = Access::eNone) {
			auto parm_ty = parm.type();
			auto& link = parm.link();

			bool init_permitted = parm.node->kind == Node::CONSTRUCT && parm.node->type[0]->kind != Type::ARRAY_TY || parm.node->kind == Node::ACQUIRE_NEXT_IMAGE;

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
						src_access = arg_ty->imbued.access;
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
			} else { // read* -> undef, src = reads
				auto arg_ty_copy = arg_ty;
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
							arg_ty_copy = r.node->call.fn.type()->opaque_fn.args[r.index];
							parm = r.node->call.args[r.index];
						} else if (r.node->kind == Node::CONVERGE) {
							continue;
						} else {
							assert(0);
						}

						if (arg_ty_copy->kind == Type::IMBUED_TY) {
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
								src_access = arg_ty->imbued.access;
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

						if (is_transfer_access(src_access)) {
							need_transfer = true;
						}
						if (is_storage_access(src_access)) {
							need_general = true;
						}
						if (is_readonly_access(src_access)) {
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
				auto arg_ty_copy = arg_ty;
				if (link.reads.size() > 0) { // we need to emit: def -> reads, RAW or nothing (before first read)
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
							arg_ty_copy = r.node->call.fn.type()->opaque_fn.args[r.index];
							parm = r.node->call.args[r.index];
						} else if (r.node->kind == Node::CONVERGE) {
							continue;
						} else {
							assert(0);
						}

						if (arg_ty_copy->kind == Type::IMBUED_TY) {
							dst_access = arg_ty_copy->imbued.access;
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
			return { init_permitted, src_use, dst_use };
		}
	};

	struct Recorder {
		Recorder(Allocator alloc, ProfilingCallbacks* callbacks, std::vector<Ref>& pass_reads, std::shared_ptr<RG> cg_module) :
		    ctx(alloc.get_context()),
		    alloc(alloc),
		    callbacks(callbacks),
		    pass_reads(pass_reads),
		    cg_module(cg_module) {}
		Context& ctx;
		Allocator alloc;
		ProfilingCallbacks* callbacks;
		std::vector<Ref>& pass_reads;
		std::shared_ptr<RG> cg_module;

		std::unordered_map<DomainFlagBits, std::unique_ptr<Stream>> streams;
		std::unordered_map<VkImage, StreamResourceUse> pending_image_syncs;
		std::unordered_map<void*, Stream*> pending_buffer_syncs;

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

		void add_sync(Type* base_ty, Scheduler::DependencyInfo di, void* value) {
			StreamResourceUse src_use = di.src_use;
			StreamResourceUse dst_use = di.dst_use;
			if ((ResourceUse)src_use == ResourceUse{} && !di.init_permitted) {
				src_use.stream = nullptr;
			}
			auto src_stream = src_use.stream;
			auto dst_stream = dst_use.stream;
			bool has_src = src_stream;
			bool has_dst = dst_stream;
			bool has_both = has_src && has_dst;
			bool cross = has_both && (src_stream != dst_stream);
			bool only_src = has_src && !has_dst;
			bool only_dst = !has_src && has_dst;

			if (cross) {
				dst_stream->add_dependency(src_stream);
			}

			if ((ResourceUse)src_use == ResourceUse{} && (ResourceUse)dst_use == ResourceUse{} && !di.init_permitted) {
				return;
			}

			if (base_ty == cg_module->builtin_image) {
				auto& img_att = *reinterpret_cast<ImageAttachment*>(value);
				if (has_dst) {
					if (only_dst) {
						auto it = pending_image_syncs.find(img_att.image.image);
						if (it != pending_image_syncs.end()) {
							if (it->second.stream != dst_stream) {
								it->second.stream->synch_image(img_att, {}, dst_use, value, di.init_permitted); // synchronize dst onto first stream
								dst_stream->synch_image(img_att, it->second, {}, value, di.init_permitted);     // synchronize src onto second stream
							}
							pending_image_syncs.erase(it);
						}
					}
					dst_stream->synch_image(img_att, src_use, dst_use, value, di.init_permitted);
				}
				if (only_src || cross) {
					src_stream->synch_image(img_att, src_use, dst_use, value, di.init_permitted);
					if (only_src) {
						pending_image_syncs.insert_or_assign(img_att.image.image, src_use);
					}
				}
			} else if (base_ty == cg_module->builtin_buffer) {
				// buffer needs no cross
				if (has_dst) {
					if (only_dst) {
						auto it = pending_buffer_syncs.find(value);
						if (it != pending_buffer_syncs.end()) {
							if (it->second != dst_stream) {
								it->second->synch_memory(src_use, dst_use, value);
							}
							pending_buffer_syncs.erase(it);
						}
					}
					dst_stream->synch_memory(src_use, dst_use, value);
				} else if (has_src) {
					src_stream->synch_memory(src_use, dst_use, value);
					if (only_src) {
						pending_buffer_syncs.insert_or_assign(value, src_stream);
					}
				}
			} else if (base_ty->kind == Type::ARRAY_TY) {
				auto elem_ty = base_ty->array.T;
				auto size = base_ty->array.count;
				if (elem_ty == cg_module->builtin_image) {
					auto img_atts = reinterpret_cast<ImageAttachment*>(value);
					for (int i = 0; i < size; i++) {
						if (has_dst) {
							if (only_dst) {
								auto it = pending_image_syncs.find(img_atts[i].image.image);
								if (it != pending_image_syncs.end()) {
									if (it->second.stream != dst_stream) {
										it->second.stream->synch_image(img_atts[i], {}, dst_use, &img_atts[i], di.init_permitted); // synchronize dst onto first stream
										dst_stream->synch_image(img_atts[i], it->second, {}, &img_atts[i], di.init_permitted);     // synchronize dst onto second stream
									}
									pending_image_syncs.erase(it);
								}
							}
							dst_stream->synch_image(img_atts[i], src_use, dst_use, &img_atts[i], di.init_permitted);
						}
						if (only_src || cross) {
							src_stream->synch_image(img_atts[i], src_use, dst_use, &img_atts[i], di.init_permitted);
							if (only_src) {
								pending_image_syncs.insert_or_assign(img_atts[i].image.image, src_use);
							}
						}
					}
				} else if (elem_ty == cg_module->builtin_buffer) {
					for (int i = 0; i < size; i++) {
						// buffer needs no cross
						auto bufs = reinterpret_cast<Buffer*>(value);
						if (has_dst) {
							auto it = pending_buffer_syncs.find(value);
							if (it != pending_buffer_syncs.end()) {
								if (it->second != dst_stream) {
									it->second->synch_memory(src_use, dst_use, &bufs[i]);
								}
								pending_buffer_syncs.erase(it);
							}
							dst_stream->synch_memory(src_use, dst_use, &bufs[i]);
						} else if (has_src) {
							src_stream->synch_memory(src_use, dst_use, &bufs[i]);
							if (only_src) {
								pending_buffer_syncs.insert_or_assign(&bufs[i], src_stream);
							}
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

	Result<void> ExecutableRenderGraph::execute(Allocator& alloc) {
		Context& ctx = alloc.get_context();

		Recorder recorder(alloc, &impl->callbacks, impl->pass_reads, impl->cg_module);
		recorder.streams.emplace(DomainFlagBits::eHost, std::make_unique<HostStream>(alloc));
		if (auto exe = ctx.get_executor(DomainFlagBits::eGraphicsQueue)) {
			recorder.streams.emplace(DomainFlagBits::eGraphicsQueue,
			                         std::make_unique<VkQueueStream>(alloc, static_cast<rtvk::QueueExecutor*>(exe), &impl->callbacks));
		}
		if (auto exe = ctx.get_executor(DomainFlagBits::eComputeQueue)) {
			recorder.streams.emplace(DomainFlagBits::eComputeQueue, std::make_unique<VkQueueStream>(alloc, static_cast<rtvk::QueueExecutor*>(exe), &impl->callbacks));
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

		Scheduler sched(alloc, impl);

		// DYNAMO
		// loop through scheduled items
		// for each scheduled item, schedule deps
		// generate barriers and batch breaks as needed by deps
		// allocate images/buffers as declares are encountered
		// start/end RPs as needed
		// inference will still need to run in the beginning, as that is compiletime

		auto print_results_to_string = [&](Node* node) {
			std::string msg = "";
			for (size_t i = 0; i < node->type.size(); i++) {
				if (i > 0) {
					msg += fmt::format(", ");
				}
				if (node->debug_info && !node->debug_info->result_names.empty()) {
					msg += fmt::format("%{}", node->debug_info->result_names[i]);
				} else {
					msg += fmt::format("%{}_{}", node->kind_to_sv(), sched.naming_index_counter + i);
				}
			}
			return msg;
		};
		auto print_results = [&](Node* node) {
			fmt::print("{}", print_results_to_string(node));
		};

		auto parm_to_string = [&](Ref parm) {
			if (parm.node->debug_info && !parm.node->debug_info->result_names.empty()) {
				return fmt::format("%{}", parm.node->debug_info->result_names[parm.index]);
			} else if (parm.node->kind == Node::CONSTANT) {
				Type* ty = parm.node->type[0];
				if (ty->kind == Type::INTEGER_TY) {
					switch (ty->integer.width) {
					case 32:
						return fmt::format("{}", constant<uint32_t>(parm));
					case 64:
						return fmt::format("{}", constant<uint64_t>(parm));
					}
				} else if (ty->kind == Type::MEMORY_TY) {
					return std::string("<mem>");
				}
			} else if (parm.node->kind == Node::PLACEHOLDER) {
				return std::string("?");
			} else {
				return fmt::format("%{}_{}", parm.node->kind_to_sv(), sched.executed.at(parm.node).naming_index + parm.index);
			}
			assert(0);
			return std::string("???");
		};

		auto print_args_to_string = [&](std::span<Ref> args) {
			std::string msg = "";
			for (size_t i = 0; i < args.size(); i++) {
				if (i > 0) {
					msg += fmt::format(", ");
				}
				auto& parm = args[i];

				msg += parm_to_string(parm);
			}
			return msg;
		};
		auto print_args = [&](std::span<Ref> args) {
			fmt::print("{}", print_args_to_string(args));
		};

		auto node_to_string = [](Node* node) {
			if (node->kind == Node::CONSTRUCT) {
				return fmt::format("construct<{}> ", Type::to_string(node->type[0]));
			} else {
				return fmt::format("{} ", node->kind_to_sv());
			}
		};

		enum class Level { eError };

		auto format_message = [&](Level level, Node* node, std::span<Ref> args, std::string err) {
			std::string msg = "";
			msg += format_source_location(node);
			msg += fmt::format("{}: '", level == Level::eError ? "error" : "other");
			msg += print_results_to_string(node);
			msg += fmt::format(" = {}", node_to_string(node));
			msg += print_args_to_string(args);
			msg += err;
			return msg;
		};

		while (!sched.work_queue.empty()) {
			auto item = sched.work_queue.front();
			sched.work_queue.pop_front();
			auto& node = item.execable;
			if (sched.executed.count(node)) { // only going execute things once
				continue;
			}
			if (item.ready) {
				sched.instr_counter++;
#ifdef VUK_DUMP_EXEC
				fmt::print("[{:#06x}] ", sched.instr_counter);
#endif
			}
			// we run nodes twice - first time we reenqueue at the front and then put all deps before it
			// second time we see it, we know that all deps have run, so we can run the node itself
			switch (node->kind) {
			case Node::MATH_BINARY: {
				if (sched.process(item)) {
					auto do_op = [&]<class T>(T, Node* node) -> T {
						T& a = sched.get_value<T>(node->math_binary.a);
						T& b = sched.get_value<T>(node->math_binary.b);
						switch (node->math_binary.op) {
						case Node::BinOp::MUL:
							return a * b;
						}
						assert(0);
					};
					switch (node->type[0]->kind) {
					case Type::INTEGER_TY: {
						switch (node->type[0]->integer.width) {
						case 32:
							sched.done(node, nullptr, do_op(uint32_t{}, node));
							break;
						case 64:
							sched.done(node, nullptr, do_op(uint64_t{}, node));
							break;
						default:
							assert(0);
						}
						break;
					}
					default:
						assert(0);
					}
				} else {
					for (auto i = 0; i < node->fixed_node.arg_count; i++) {
						sched.schedule_dependency(node->fixed_node.args[i], RW::eRead);
					}
				}
				break;
			}
			case Node::CONSTRUCT: { // when encountering a CONSTRUCT, allocate the thing if needed
				if (sched.process(item)) {
					if (node->type[0] == impl->cg_module->builtin_buffer) {
						auto& bound = constant<Buffer>(node->construct.args[0]);
						try {
							bound.size = eval<size_t>(node->construct.args[1]); // collapse inferencing
						} catch (CannotBeConstantEvaluated& err) {
							if (err.ref.node->kind == Node::PLACEHOLDER) {
								return { expected_error,
									       RenderGraphException(format_message(Level::eError, node, node->construct.args.subspan(1), "': argument not set or inferrable\n")) };
							} else {
								return { expected_error,
									       RenderGraphException(format_message(Level::eError, node, node->construct.args.subspan(1), "': argument not constant evaluatable\n")) };
							}
						}
#ifdef VUK_DUMP_EXEC
						print_results(node);
						fmt::print(" = construct<buffer> ");
						print_args(node->construct.args.subspan(1));
						fmt::print("\n");
#endif
						if (bound.buffer == VK_NULL_HANDLE) {
							assert(bound.size != ~(0u));
							BufferCreateInfo bci{ .mem_usage = bound.memory_usage, .size = bound.size, .alignment = 1 }; // TODO: alignment?
							auto allocator = node->construct.allocator ? *node->construct.allocator : alloc;
							auto buf = allocate_buffer(allocator, bci);
							if (!buf) {
								return buf;
							}
							bound = **buf;
						}
						sched.done(node, host_stream, bound);
					} else if (node->type[0] == impl->cg_module->builtin_image) {
						auto& attachment = *reinterpret_cast<ImageAttachment*>(node->construct.args[0].node->constant.value);
						// collapse inferencing
						try {
							attachment.extent.width = eval<uint32_t>(node->construct.args[1]);
							attachment.extent.height = eval<uint32_t>(node->construct.args[2]);
							attachment.extent.depth = eval<uint32_t>(node->construct.args[3]);
							attachment.format = eval<Format>(node->construct.args[4]);
							attachment.sample_count = eval<Samples>(node->construct.args[5]);
							attachment.base_layer = eval<uint32_t>(node->construct.args[6]);
							attachment.layer_count = eval<uint32_t>(node->construct.args[7]);
							attachment.base_level = eval<uint32_t>(node->construct.args[8]);
							attachment.level_count = eval<uint32_t>(node->construct.args[9]);

							if (attachment.image_view == ImageView{}) {
								if (attachment.view_type == ImageViewType::eInfer && attachment.layer_count != VK_REMAINING_ARRAY_LAYERS) {
									if (attachment.image_type == ImageType::e1D) {
										if (attachment.layer_count == 1) {
											attachment.view_type = ImageViewType::e1D;
										} else {
											attachment.view_type = ImageViewType::e1DArray;
										}
									} else if (attachment.image_type == ImageType::e2D) {
										if (attachment.layer_count == 1) {
											attachment.view_type = ImageViewType::e2D;
										} else {
											attachment.view_type = ImageViewType::e2DArray;
										}
									} else if (attachment.image_type == ImageType::e3D) {
										if (attachment.layer_count == 1) {
											attachment.view_type = ImageViewType::e3D;
										} else {
											attachment.view_type = ImageViewType::e2DArray;
										}
									}
								}
							}

						} catch (CannotBeConstantEvaluated& err) {
							if (err.ref.node->kind == Node::PLACEHOLDER) {
								return { expected_error,
									       RenderGraphException(format_message(Level::eError, node, node->construct.args.subspan(1), "': argument not set or inferrable\n")) };
							} else {
								return { expected_error,
									       RenderGraphException(format_message(Level::eError, node, node->construct.args.subspan(1), "': argument not constant evaluatable\n")) };
							}
						}
#ifdef VUK_DUMP_EXEC
						print_results(node);
						fmt::print(" = construct<image> ");
						print_args(node->construct.args.subspan(1));
						fmt::print("\n");
#endif
						if (!attachment.image) {
							auto allocator = node->construct.allocator ? *node->construct.allocator : alloc;
							attachment.usage = impl->compute_usage(&first(node).link());
							assert(attachment.usage != ImageUsageFlags{});
							auto img = allocate_image(allocator, attachment);
							if (!img) {
								return img;
							}
							attachment.image = **img;
							if (node->debug_info && node->debug_info->result_names.size() > 0 && !node->debug_info->result_names[0].empty()) {
								ctx.set_name(attachment.image.image, node->debug_info->result_names[0]);
							}
						}
						sched.done(node, host_stream, attachment);
					} else if (node->type[0] == impl->cg_module->builtin_swapchain) {
#ifdef VUK_DUMP_EXEC
						print_results(node);
						fmt::print(" = construct<swapchain>\n");
#endif
						/* no-op */
						sched.done(node, host_stream, sched.get_value(node->construct.args[0]));
					} else if (node->type[0]->kind == Type::ARRAY_TY) {
						for (size_t i = 1; i < node->construct.args.size(); i++) {
							auto arg_ty = node->construct.args[i].type();
							auto& parm = node->construct.args[i];

							recorder.add_sync(sched.base_type(parm), sched.get_dependency_info(parm, arg_ty, RW::eWrite, nullptr), sched.get_value(parm));
						}

						auto size = node->type[0]->array.count;
						auto elem_ty = node->type[0]->array.T;
#ifdef VUK_DUMP_EXEC
						print_results(node);
						assert(elem_ty == impl->cg_module->builtin_buffer || elem_ty == impl->cg_module->builtin_image);
						fmt::print(" = construct<{}[{}]> ", elem_ty == impl->cg_module->builtin_buffer ? "buffer" : "image", size);
						print_args(node->construct.args.subspan(1));
						fmt::print("\n");
#endif
						assert(node->construct.args[0].type()->kind == Type::MEMORY_TY);
						if (elem_ty == impl->cg_module->builtin_buffer) {
							auto arr_mem = new (sched.arena.ensure_space(sizeof(Buffer) * size)) Buffer[size];
							for (auto i = 0; i < size; i++) {
								auto& elem = node->construct.args[i + 1];
								assert(Type::stripped(elem.type()) == impl->cg_module->builtin_buffer);

								memcpy(&arr_mem[i], sched.get_value(elem), sizeof(Buffer));
							}
							node->construct.args[0].node->constant.value = arr_mem;
							sched.done(node, host_stream, (void*)arr_mem);
						} else if (elem_ty == impl->cg_module->builtin_image) {
							auto arr_mem = new (sched.arena.ensure_space(sizeof(ImageAttachment) * size)) ImageAttachment[size];
							for (auto i = 0; i < size; i++) {
								auto& elem = node->construct.args[i + 1];
								assert(Type::stripped(elem.type()) == impl->cg_module->builtin_image);

								memcpy(&arr_mem[i], sched.get_value(elem), sizeof(ImageAttachment));
							}
							node->construct.args[0].node->constant.value = arr_mem;
							sched.done(node, host_stream, (void*)arr_mem);
						}
					} else {
						assert(0);
					}
				} else {
					for (auto& parm : node->construct.args.subspan(1)) {
						sched.schedule_dependency(parm, RW::eWrite);
					}
				}
				break;
			}
			case Node::CALL: {
				if (sched.process(item)) {                    // we have executed every dep, so execute ourselves too
					Stream* dst_stream = item.scheduled_stream; // the domain this call will execute on

					auto vk_rec = dynamic_cast<VkQueueStream*>(dst_stream); // TODO: change this into dynamic dispatch on the Stream
					assert(vk_rec);
					// run all the barriers here!

					for (size_t i = 0; i < node->call.args.size(); i++) {
						auto& arg_ty = node->call.fn.type()->opaque_fn.args[i];
						auto& parm = node->call.args[i];
						auto& link = parm.link();

						if (arg_ty->kind == Type::IMBUED_TY) {
							auto access = arg_ty->imbued.access;

							// here: figuring out which allocator to use to make image views for the RP and then making them
							if (is_framebuffer_attachment(access)) {
								auto urdef = link.urdef.node;
								auto allocator = urdef->kind == Node::CONSTRUCT && urdef->construct.allocator ? *urdef->construct.allocator : alloc;
								auto& img_att = sched.get_value<ImageAttachment>(parm);
								if (img_att.view_type == ImageViewType::eInfer || img_att.view_type == ImageViewType::eCube) { // framebuffers need 2D or 2DArray views
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

							// Write and ReadWrite
							RW sync_access = (is_write_access(access) || access == Access::eConsume) ? RW::eWrite : RW::eRead;
							recorder.add_sync(sched.base_type(parm), sched.get_dependency_info(parm, arg_ty, sync_access, dst_stream), sched.get_value(parm));
							
							if (is_framebuffer_attachment(access)) {
								auto& img_att = sched.get_value<ImageAttachment>(parm);
								vk_rec->prepare_render_pass_attachment(alloc, img_att);
							}
						} else {
							assert(0);
						}
					}

					// make the renderpass if needed!
					recorder.synchronize_stream(dst_stream);
					// run the user cb!
					std::vector<void*> opaque_rets;
					if (node->call.fn.type()->kind == Type::OPAQUE_FN_TY) {
						CommandBuffer cobuf(*this, ctx, alloc, vk_rec->cbuf);
						if (node->call.fn.type()->debug_info) {
							ctx.begin_region(vk_rec->cbuf, node->call.fn.type()->debug_info->name);
						}
						if (vk_rec->rp.rpci.attachments.size() > 0) {
							vk_rec->prepare_render_pass();
							fill_render_pass_info(vk_rec->rp, 0, cobuf);
						}

						std::vector<void*> opaque_args;
						std::vector<void*> opaque_meta;
						for (size_t i = 0; i < node->call.args.size(); i++) {
							auto& parm = node->call.args[i];
							auto& link = parm.link();
							assert(link.urdef);
							opaque_args.push_back(sched.get_value(parm));
							opaque_meta.push_back(&link.urdef);
						}
						opaque_rets.resize(node->call.fn.type()->opaque_fn.return_types.size());
						(*node->call.fn.type()->opaque_fn.callback)(cobuf, opaque_args, opaque_meta, opaque_rets);
						if (vk_rec->rp.handle) {
							vk_rec->end_render_pass();
						}
						if (node->call.fn.type()->debug_info) {
							ctx.end_region(vk_rec->cbuf);
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
					sched.done(node, dst_stream, std::span(opaque_rets));
				} else { // execute deps
					for (size_t i = 0; i < node->call.args.size(); i++) {
						auto& arg_ty = node->call.fn.type()->opaque_fn.args[i];
						auto& parm = node->call.args[i];

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
			case Node::RELACQ: {
				if (sched.process(item)) {
					auto acqrel = node->relacq.rel_acq;
					Stream* dst_stream = item.scheduled_stream;
					auto values = new void*[node->relacq.src.size()];
					// if acq is nullptr, then this degenerates to a NOP, sync and skip
					for (size_t i = 0; i < node->relacq.src.size(); i++) {
						auto parm = node->relacq.src[i];
						auto arg_ty = node->type[i];
						auto di = sched.get_dependency_info(parm, arg_ty, RW::eWrite, dst_stream);
						di.src_use.stream = di.dst_use.stream; // TODO: not sure this is valid, but we can't handle split
						di.dst_use.stream = nullptr;
						auto value = sched.get_value(parm);
						auto storage = new std::byte[parm.type()->size];
						memcpy(storage, impl->get_value(parm), parm.type()->size);
						values[i] = storage;
						recorder.add_sync(sched.base_type(parm), di, storage);

						acqrel->last_use.push_back(di.src_use);
						if (i == 0) {
							di.src_use.stream->add_dependent_signal(acqrel);
						}
					}
					if (!acqrel) { // (we should've handled this before this moment)
						fmt::print("???");
						assert(false);
					} else {
						switch (acqrel->status) {
						case Signal::Status::eDisarmed: // means we have to signal this
							node->relacq.values = std::span{ values, node->relacq.src.size() };
							break;
						case Signal::Status::eSynchronizable: // means this is an acq instead (we should've handled this before this moment)
						case Signal::Status::eHostAvailable:
							fmt::print("???");
							assert(false);
							break;
						}
					}
#ifdef VUK_DUMP_EXEC
					print_results(node);
					fmt::print(" <- ");
					print_args(node->relacq.src);
					fmt::print("\n");
#endif
					sched.done(node, item.scheduled_stream, std::span{ values, node->relacq.src.size() });
				} else {
					for (size_t i = 0; i < node->relacq.src.size(); i++) {
						sched.schedule_dependency(node->relacq.src[i], RW::eWrite);
					}
				}
				break;
			}
			case Node::ACQUIRE: {
				auto acq = node->acquire.acquire;
				auto src_stream = recorder.stream_for_executor(acq->source.executor);

				Scheduler::DependencyInfo di;
				di.src_use = { acq->last_use[node->acquire.index], src_stream };
				di.dst_use = { to_use(Access::eNone), nullptr };
				recorder.add_sync(node->type[0], di, sched.get_value({ node, node->acquire.index }));

				if (node->type[0] == impl->cg_module->builtin_buffer) {
#ifdef VUK_DUMP_EXEC
					print_results(node);
					fmt::print(" = acquire<buffer>\n");
#endif
				} else if (node->type[0] == impl->cg_module->builtin_image) {
#ifdef VUK_DUMP_EXEC
					print_results(node);
					fmt::print(" = acquire<image>\n");
#endif
				}

				sched.done(node, nullptr, sched.get_value({ node, node->acquire.index }));
				break;
			}
			case Node::RELEASE:
				if (sched.process(item)) {
					// release is to execute: we need to flush current queue -> end current batch and add signal
					auto parm = node->release.src;
					auto src_stream = item.scheduled_stream;
					DomainFlagBits src_domain = src_stream->domain;
					Stream* dst_stream;
					if (node->release.dst_domain == DomainFlagBits::ePE) {
						auto& link = node->release.src.link();
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
					acqrel->last_use.push_back(di.src_use);
					if (src_domain == DomainFlagBits::eHost) {
						acqrel->status = Signal::Status::eHostAvailable;
					}
					if (dst_domain == DomainFlagBits::ePE) {
						auto& link = node->release.src.link();
						auto& swp = sched.get_value<Swapchain>(link.urdef.node->acquire_next_image.swapchain);
						assert(src_stream->domain & DomainFlagBits::eDevice);
						auto result = dynamic_cast<VkQueueStream*>(src_stream)->present(swp);
						// TODO: do something with the result here
					}
					src_stream->add_dependent_signal(acqrel);
					src_stream->submit();

					auto storage = new std::byte[parm.type()->size];
					memcpy(storage, impl->get_value(parm), parm.type()->size);
					node->release.value = storage;
					sched.done(node, src_stream, sched.get_value(parm));
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
					sched.done(node, pe_stream, swp.images[swp.image_index]);
				} else {
					sched.schedule_dependency(node->acquire_next_image.swapchain, RW::eWrite);
				}
				break;
			}
			case Node::EXTRACT: {
				if (sched.process(item)) {
					// no sync - currently no extract composite needs sync
					/* recorder.add_sync(sched.base_type(node->extract.composite),
					                  sched.get_dependency_info(node->extract.composite, node->extract.composite.type(), RW::eWrite, nullptr),
					                  sched.get_value(node->extract.composite));*/
#ifdef VUK_DUMP_EXEC
					print_results(node);
					fmt::print(" = ");
					print_args(std::span{ &node->extract.composite, 1 });
					fmt::print("[{}]", constant<uint64_t>(node->extract.index));
					fmt::print("\n");
#endif
					sched.done(node, nullptr, sched.get_value(first(node))); // extract doesn't execute
				} else {
					sched.schedule_dependency(node->extract.composite, RW::eWrite);
					sched.schedule_dependency(node->extract.index, RW::eRead);
				}
				break;
			}
			case Node::SLICE: {
				if (sched.process(item)) {
					Subrange::Image r = { constant<uint32_t>(node->slice.base_level),
						                    constant<uint32_t>(node->slice.level_count),
						                    constant<uint32_t>(node->slice.base_layer),
						                    constant<uint32_t>(node->slice.layer_count) };

					// half sync
					recorder.add_sync(sched.base_type(node->slice.image),
					                  sched.get_dependency_info(node->slice.image, node->slice.image.type(), RW::eRead, nullptr),
					                  sched.get_value(node->slice.image));

#ifdef VUK_DUMP_EXEC
					print_results(node);
					fmt::print(" = ");
					print_args(std::span{ &node->slice.image, 1 });
					if (r.base_level > 0 || r.level_count != VK_REMAINING_MIP_LEVELS) {
						fmt::print("[m{}:{}]", r.base_level, r.base_level + r.level_count - 1);
					}
					if (r.base_layer > 0 || r.layer_count != VK_REMAINING_ARRAY_LAYERS) {
						fmt::print("[l{}:{}]", r.base_layer, r.base_layer + r.layer_count - 1);
					}
					fmt::print("\n");
#endif

					// assert(elem_ty == impl->cg_module->builtin_image);
					auto sliced = ImageAttachment(*(ImageAttachment*)sched.get_value(node->slice.image));
					sliced.base_level += r.base_level;
					if (r.level_count != VK_REMAINING_MIP_LEVELS) {
						sliced.level_count = r.level_count;
					}
					sliced.base_layer += r.base_layer;
					if (r.layer_count != VK_REMAINING_ARRAY_LAYERS) {
						sliced.layer_count = r.layer_count;
					}

					if (!(node->debug_info && node->debug_info->result_names.size() > 0 && !node->debug_info->result_names[0].empty())) {
						auto decl = node->slice.image.link().urdef.node;
						if (decl->debug_info && decl->debug_info->result_names.size() > 0 && !decl->debug_info->result_names[0].empty()) {
							std::string name = fmt::format("{}[m{}:{}][l{}:{}]",
							                               decl->debug_info->result_names[0],
							                               sliced.base_level,
							                               sliced.base_level + sliced.level_count - 1,
							                               sliced.base_layer,
							                               sliced.base_layer + sliced.layer_count - 1);
							impl->cg_module->name_output(first(node), name);
						}
					}

					sched.done(node, sched.executed.at(node->slice.image.node).stream, sliced); // slice doesn't execute
				} else {
					sched.schedule_dependency(node->slice.image, RW::eRead);
					sched.schedule_dependency(node->slice.base_level, RW::eRead);
					sched.schedule_dependency(node->slice.level_count, RW::eRead);
					sched.schedule_dependency(node->slice.base_layer, RW::eRead);
					sched.schedule_dependency(node->slice.layer_count, RW::eRead);
				}
				break;
			}
			case Node::CONVERGE: {
				if (sched.process(item)) {
					auto base = node->converge.ref_and_diverged[0];

					// half sync
					for (size_t i = 1; i < node->converge.ref_and_diverged.size(); i++) {
						auto& item = node->converge.ref_and_diverged[i];
						recorder.add_sync(sched.base_type(item),
						                  sched.get_dependency_info(item, item.type(), node->converge.write[i - 1] ? RW::eWrite : RW::eRead, nullptr),
						                  sched.get_value(item));
					}

#ifdef VUK_DUMP_EXEC
					print_results(node);
					fmt::print(" = ");
					print_args(node->converge.ref_and_diverged.subspan(0, 1));
					fmt::print("{{");
					print_args(node->converge.ref_and_diverged.subspan(1));
					fmt::print("}}");
					fmt::print("\n");
#endif

					sched.done(node, item.scheduled_stream, sched.get_value(base)); // converge doesn't execute
				} else {
					sched.schedule_dependency(node->converge.ref_and_diverged[0], RW::eRead);
					for (size_t i = 1; i < node->converge.ref_and_diverged.size(); i++) {
						sched.schedule_dependency(node->converge.ref_and_diverged[i], node->converge.write[i - 1] ? RW::eWrite : RW::eRead);
					}
				}
				break;
			}
			case Node::INDIRECT_DEPEND: {
				auto rref = node->indirect_depend.rref;
				Ref true_ref;
				auto count = rref.node->generic_node.arg_count;
				if (count != (uint8_t)~0u) {
					true_ref = rref.node->fixed_node.args[rref.index];
				} else {
					true_ref = rref.node->variable_node.args[rref.index];
				}

				if (sched.process(item)) {
#ifdef VUK_DUMP_EXEC
					print_results(node);
					fmt::print(" = ");
					fmt::print("{{");
					print_args(std::span{ &true_ref, 1 });
					fmt::print("}}*");
					fmt::print("\n");
#endif
					// half sync
					// recorder.add_sync(sched.base_type(true_ref), sched.get_dependency_info(true_ref, true_ref.type(), RW::eWrite, nullptr), sched.get_value(true_ref));

					sched.done(node, sched.executed.at(rref.node).stream, sched.get_value(true_ref)); // indirect depend doesn't execute
				} else {
					sched.schedule_dependency(true_ref, RW::eWrite);
					sched.schedule_new(rref.node);
				}
				break;
			}
			default:
				assert(0);
			}
		}

		// restore acquire types
		for (auto& [node, t] : impl->type_restore) {
			node->type[0] = t;
		}

		return { expected_value };
	}
} // namespace vuk
