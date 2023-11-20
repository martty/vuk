#include "Cache.hpp"
#include "RenderGraphImpl.hpp"
#include "vuk/AllocatorHelpers.hpp"
#include "vuk/CommandBuffer.hpp"
#include "vuk/Context.hpp"
#include "vuk/Future.hpp"
#include "vuk/Hash.hpp" // for create
#include "vuk/RenderGraph.hpp"
#include "vuk/Util.hpp"

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
		auto attachments = rpass.attachments.to_span(impl->rp_infos);
		for (uint32_t i = 0; i < spdesc.colorAttachmentCount; i++) {
			rpi.color_attachment_names[i] = attachments[spdesc.pColorAttachments[i].attachment].attachment_info->name;
		}
		cobuf.color_blend_attachments.resize(spdesc.colorAttachmentCount);
		cobuf.ongoing_render_pass = rpi;
	}

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
	}

	Result<SubmitInfo> ExecutableRenderGraph::record_single_submit(Allocator& alloc, std::span<PassInfo*> passes, vuk::DomainFlagBits domain) {
		assert(passes.size() > 0);

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
		if (this->impl->callbacks.on_begin_command_buffer)
			cbuf_profile_data = this->impl->callbacks.on_begin_command_buffer(this->impl->callbacks.user_data, cbuf);

		uint64_t command_buffer_index = passes[0]->command_buffer_index;
		int32_t render_pass_index = -1;
		for (size_t i = 0; i < passes.size(); i++) {
			auto& pass = passes[i];

			for (auto& ref : pass->referenced_swapchains.to_span(impl->swapchain_references)) {
				used_swapchains.emplace(impl->get_bound_attachment(ref).swapchain);
			}

			if (pass->command_buffer_index != command_buffer_index) { // end old cb and start new one
				if (auto result = ctx.vkEndCommandBuffer(cbuf); result != VK_SUCCESS) {
					return { expected_error, VkException{ result } };
				}

				VUK_DO_OR_RETURN(alloc.allocate_command_buffers(std::span{ &*hl_cbuf, 1 }, std::span{ &ci, 1 }));
				si.command_buffers.emplace_back(*hl_cbuf);

				cbuf = hl_cbuf->command_buffer;

				VkCommandBufferBeginInfo cbi{ .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO, .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT };
				ctx.vkBeginCommandBuffer(cbuf, &cbi);

				if (this->impl->callbacks.on_begin_command_buffer)
					cbuf_profile_data = this->impl->callbacks.on_begin_command_buffer(this->impl->callbacks.user_data, cbuf);
			}

			// if we had a render pass running, but now it changes
			if (pass->render_pass_index != render_pass_index && render_pass_index != -1) {
				ctx.vkCmdEndRenderPass(cbuf);
			}

			if (i > 1) {
				// insert post-barriers
				impl->emit_barriers(ctx, cbuf, domain, passes[i - 1]->post_memory_barriers, passes[i - 1]->post_image_barriers);
			}
			// insert pre-barriers
			impl->emit_barriers(ctx, cbuf, domain, pass->pre_memory_barriers, pass->pre_image_barriers);

			// if render pass is changing and new pass uses one
			if (pass->render_pass_index != render_pass_index && pass->render_pass_index != -1) {
				begin_render_pass(ctx, impl->rpis[pass->render_pass_index], cbuf, false);
			}

			render_pass_index = pass->render_pass_index;

			for (auto& w : pass->relative_waits.to_span(impl->waits)) {
				si.relative_waits.emplace_back(w);
			}

			for (auto& w : pass->absolute_waits.to_span(impl->absolute_waits)) {
				si.absolute_waits.emplace_back(w);
			}

			CommandBuffer cobuf(*this, ctx, alloc, cbuf);
			if (render_pass_index >= 0) {
				fill_render_pass_info(impl->rpis[pass->render_pass_index], 0, cobuf);
			} else {
				cobuf.ongoing_render_pass = {};
			}

			// propagate signals onto SI
			auto pass_fut_signals = pass->future_signals.to_span(impl->future_signals);
			si.future_signals.insert(si.future_signals.end(), pass_fut_signals.begin(), pass_fut_signals.end());

			if (!pass->qualified_name.is_invalid()) {
				ctx.begin_region(cobuf.command_buffer, pass->qualified_name.name);
			}
			if (pass->pass->execute) {
				cobuf.current_pass = pass;
				void* pass_profile_data = nullptr;
				if (this->impl->callbacks.on_begin_pass)
					pass_profile_data = this->impl->callbacks.on_begin_pass(this->impl->callbacks.user_data, pass->pass->name, cbuf, (DomainFlagBits)pass->domain.m_mask);
				pass->pass->execute(cobuf);
				if (this->impl->callbacks.on_end_pass)
					this->impl->callbacks.on_end_pass(this->impl->callbacks.user_data, pass_profile_data);
			}
			if (!pass->qualified_name.is_invalid()) {
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
		impl->emit_barriers(ctx, cbuf, domain, passes.back()->post_memory_barriers, passes.back()->post_image_barriers);

		if (this->impl->callbacks.on_end_command_buffer)
			this->impl->callbacks.on_end_command_buffer(this->impl->callbacks.user_data, cbuf_profile_data);
		if (auto result = ctx.vkEndCommandBuffer(cbuf); result != VK_SUCCESS) {
			return { expected_error, VkException{ result } };
		}

		si.used_swapchains.insert(si.used_swapchains.end(), used_swapchains.begin(), used_swapchains.end());

		return { expected_value, std::move(si) };
	}

	Result<SubmitBundle> ExecutableRenderGraph::execute(Allocator& alloc, std::vector<std::pair<SwapchainRef, size_t>> swp_with_index) {
		Context& ctx = alloc.get_context();

		// bind swapchain attachment images & ivs
		for (auto& bound : impl->bound_attachments) {
			if (bound.type == AttachmentInfo::Type::eSwapchain && bound.parent_attachment == 0) {
				auto it = std::find_if(swp_with_index.begin(), swp_with_index.end(), [boundb = &bound](auto& t) { return t.first == boundb->swapchain; });
				bound.attachment.image_view = it->first->image_views[it->second];
				bound.attachment.image = it->first->images[it->second];
				bound.attachment.extent = Dimension3D::absolute(it->first->extent);
				bound.attachment.sample_count = vuk::Samples::e1;
			}
		}

		// pre-inference: which IAs are in which FBs?
		for (auto& rp : impl->rpis) {
			for (auto& rp_att : rp.attachments.to_span(impl->rp_infos)) {
				auto& att = *rp_att.attachment_info;

				att.rp_uses.append(impl->attachment_rp_references, &rp);
				auto& ia = att.attachment;
				ia.image_type = ia.image_type == ImageType::eInfer ? vuk::ImageType::e2D : ia.image_type;

				ia.base_layer = ia.base_layer == VK_REMAINING_ARRAY_LAYERS ? 0 : ia.base_layer;
				ia.layer_count =
				    ia.layer_count == VK_REMAINING_ARRAY_LAYERS ? 1 : ia.layer_count; // TODO: this prevents inference later on, so this means we are doing it too early
				ia.base_level = ia.base_level == VK_REMAINING_MIP_LEVELS ? 0 : ia.base_level;

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
					if (ia.sample_count == Samples::eInfer && (ia.extent.extent.width == 0 && ia.extent.extent.height == 0)) { // this IA is not helpful for FB inference
						continue;
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

		// acquire the render passes
		for (auto& rp : impl->rpis) {
			if (rp.attachments.size() == 0) {
				continue;
			}

			for (auto& attrpinfo : rp.attachments.to_span(impl->rp_infos)) {
				attrpinfo.description.format = (VkFormat)attrpinfo.attachment_info->attachment.format;
				attrpinfo.description.samples = (VkSampleCountFlagBits)attrpinfo.attachment_info->attachment.sample_count.count;
				rp.rpci.attachments.push_back(attrpinfo.description);
			}

			rp.rpci.attachmentCount = (uint32_t)rp.rpci.attachments.size();
			rp.rpci.pAttachments = rp.rpci.attachments.data();

			auto result = alloc.allocate_render_passes(std::span{ &rp.handle, 1 }, std::span{ &rp.rpci, 1 });
			// drop render pass immediately
			if (result) {
				alloc.deallocate(std::span{ &rp.handle, 1 });
			}
		}

		// create buffers
		for (auto& bound : impl->bound_buffers) {
			if (bound.buffer.buffer == VK_NULL_HANDLE) {
				BufferCreateInfo bci{ .mem_usage = bound.buffer.memory_usage, .size = bound.buffer.size, .alignment = 1 }; // TODO: alignment?
				auto allocator = bound.allocator ? *bound.allocator : alloc;
				auto buf = allocate_buffer(allocator, bci);
				if (!buf) {
					return buf;
				}
				bound.buffer = **buf;
			}
		}

		// create non-attachment images
		for (auto& bound : impl->bound_attachments) {
			if (!bound.attachment.image && bound.parent_attachment == 0) {
				auto allocator = bound.allocator ? *bound.allocator : alloc;
				assert(bound.attachment.usage != ImageUsageFlags{});
				auto img = allocate_image(allocator, bound.attachment);
				if (!img) {
					return img;
				}
				bound.attachment.image = **img;
				ctx.set_name(bound.attachment.image.image, bound.name.name);
			}
		}

		// create framebuffers, create & bind attachments
		for (auto& rp : impl->rpis) {
			if (rp.attachments.size() == 0)
				continue;

			auto& ivs = rp.fbci.attachments;
			std::vector<VkImageView> vkivs;

			Extent2D fb_extent = Extent2D{ rp.fbci.width, rp.fbci.height };

			// create internal attachments; bind attachments to fb
			std::optional<uint32_t> fb_layer_count;
			for (auto& attrpinfo : rp.attachments.to_span(impl->rp_infos)) {
				auto& bound = *attrpinfo.attachment_info;
				uint32_t base_layer = bound.attachment.base_layer + bound.image_subrange.base_layer;
				uint32_t layer_count = bound.image_subrange.layer_count == VK_REMAINING_ARRAY_LAYERS ? bound.attachment.layer_count : bound.image_subrange.layer_count;
				assert(base_layer + layer_count <= bound.attachment.base_layer + bound.attachment.layer_count);
				fb_layer_count = layer_count;

				auto specific_attachment = bound.attachment;
				if (bound.parent_attachment < 0) {
					specific_attachment = impl->get_bound_attachment(bound.parent_attachment).attachment;
					specific_attachment.image_view = {};
				}
				if (specific_attachment.image_view == ImageView{}) {
					specific_attachment.base_layer = base_layer;
					if (specific_attachment.view_type == ImageViewType::eCube) {
						if (layer_count > 1) {
							specific_attachment.view_type = ImageViewType::e2DArray;
						} else {
							specific_attachment.view_type = ImageViewType::e2D;
						}
					}
					specific_attachment.layer_count = layer_count;
					assert(specific_attachment.level_count == 1);

					auto allocator = bound.allocator ? *bound.allocator : alloc;
					auto iv = allocate_image_view(allocator, specific_attachment);
					if (!iv) {
						return iv;
					}
					specific_attachment.image_view = **iv;
					auto name = std::string("ImageView: RenderTarget ") + std::string(bound.name.name.to_sv());
					ctx.set_name(specific_attachment.image_view.payload, Name(name));
				}

				ivs.push_back(specific_attachment.image_view);
				vkivs.push_back(specific_attachment.image_view.payload);
			}

			rp.fbci.renderPass = rp.handle;
			rp.fbci.pAttachments = &vkivs[0];
			rp.fbci.width = fb_extent.width;
			rp.fbci.height = fb_extent.height;
			assert(fb_extent.width > 0);
			assert(fb_extent.height > 0);
			rp.fbci.attachmentCount = (uint32_t)vkivs.size();
			rp.fbci.layers = *fb_layer_count;

			Unique<VkFramebuffer> fb(alloc);
			VUK_DO_OR_RETURN(alloc.allocate_framebuffers(std::span{ &*fb, 1 }, std::span{ &rp.fbci, 1 }));
			rp.framebuffer = *fb; // queue framebuffer for destruction
		}

		for (auto& attachment_info : impl->bound_attachments) {
			if (attachment_info.attached_future && attachment_info.parent_attachment == 0) {
				ImageAttachment att = attachment_info.attachment;
				attachment_info.attached_future->result = att;
			}
		}

		for (auto& buffer_info : impl->bound_buffers) {
			if (buffer_info.attached_future) {
				Buffer buf = buffer_info.buffer;
				buffer_info.attached_future->result = buf;
			}
		}

		SubmitBundle sbundle;

		auto record_batch = [&alloc, this](std::span<PassInfo*> passes, DomainFlagBits domain) -> Result<SubmitBatch> {
			SubmitBatch sbatch{ .domain = domain };
			auto partition_it = passes.begin();
			while (partition_it != passes.end()) {
				auto batch_index = (*partition_it)->batch_index;
				auto new_partition_it = std::partition_point(partition_it, passes.end(), [batch_index](PassInfo* rpi) { return rpi->batch_index == batch_index; });
				auto partition_span = std::span(partition_it, new_partition_it);
				auto si = record_single_submit(alloc, partition_span, domain);
				if (!si) {
					return si;
				}
				sbatch.submits.emplace_back(*si);
				partition_it = new_partition_it;
			}
			for (auto& rel : impl->final_releases) {
				if (rel.dst_use.domain & domain) {
					sbatch.submits.back().future_signals.push_back(rel.signal);
				}
			}
			
			std::erase_if(impl->final_releases, [=](auto& rel) { return rel.dst_use.domain & domain; });
			return { expected_value, sbatch };
		};

		// record cbufs
		// assume that rpis are partitioned wrt batch_index

		if (impl->graphics_passes.size() > 0) {
			auto batch = record_batch(impl->graphics_passes, DomainFlagBits::eGraphicsQueue);
			if (!batch) {
				return batch;
			}
			sbundle.batches.emplace_back(std::move(*batch));
		}

		if (impl->compute_passes.size() > 0) {
			auto batch = record_batch(impl->compute_passes, DomainFlagBits::eComputeQueue);
			if (!batch) {
				return batch;
			}
			sbundle.batches.emplace_back(std::move(*batch));
		}

		if (impl->transfer_passes.size() > 0) {
			auto batch = record_batch(impl->transfer_passes, DomainFlagBits::eTransferQueue);
			if (!batch) {
				return batch;
			}
			sbundle.batches.emplace_back(std::move(*batch));
		}

		return { expected_value, std::move(sbundle) };
	}

	Result<BufferInfo, RenderGraphException> ExecutableRenderGraph::get_resource_buffer(const NameReference& name_ref, PassInfo* pass_info) {
		for (auto& r : pass_info->resources.to_span(impl->resources)) {
			if (r.type == Resource::Type::eBuffer && r.original_name == name_ref.name.name && r.foreign == name_ref.rg) {
				auto& att = impl->get_bound_buffer(r.reference);
				return { expected_value, att };
			}
		}

		return { expected_error, errors::make_cbuf_references_undeclared_resource(*pass_info, Resource::Type::eImage, name_ref.name.name) };
	}

	Result<AttachmentInfo, RenderGraphException> ExecutableRenderGraph::get_resource_image(const NameReference& name_ref, PassInfo* pass_info) {
		for (auto& r : pass_info->resources.to_span(impl->resources)) {
			if (r.type == Resource::Type::eImage && r.original_name == name_ref.name.name && r.foreign == name_ref.rg) {
				auto& att = impl->get_bound_attachment(r.reference);
				auto parent_idx = att.parent_attachment;
				vuk::AttachmentInfo* parent = nullptr;
				if (parent_idx < 0) {
					while (parent_idx < 0) {
						parent = &impl->get_bound_attachment(parent_idx);
						parent_idx = parent->parent_attachment;
					}
					att.attachment.image = parent->attachment.image;
					att.attachment.base_layer = att.image_subrange.base_layer;
					att.attachment.base_level = att.image_subrange.base_level;
					att.attachment.layer_count =
					    att.image_subrange.layer_count == VK_REMAINING_ARRAY_LAYERS ? att.attachment.layer_count : att.image_subrange.layer_count;
					att.attachment.level_count = att.image_subrange.level_count == VK_REMAINING_MIP_LEVELS ? att.attachment.level_count : att.image_subrange.level_count;
					att.attachment.view_type = parent->attachment.view_type;
					att.attachment.image_view = {};
				}
				return { expected_value, att };
			}
		}

		return { expected_error, errors::make_cbuf_references_undeclared_resource(*pass_info, Resource::Type::eImage, name_ref.name.name) };
	}

	Result<bool, RenderGraphException> ExecutableRenderGraph::is_resource_image_in_general_layout(const NameReference& name_ref, PassInfo* pass_info) {
		for (auto& r : pass_info->resources.to_span(impl->resources)) {
			if (r.type == Resource::Type::eImage && r.original_name == name_ref.name.name && r.foreign == name_ref.rg) {
				return { expected_value, r.promoted_to_general };
			}
		}

		return { expected_error, errors::make_cbuf_references_undeclared_resource(*pass_info, Resource::Type::eImage, name_ref.name.name) };
	}

	QualifiedName ExecutableRenderGraph::resolve_name(Name name, PassInfo* pass_info) const noexcept {
		auto qualified_name = QualifiedName{ pass_info->qualified_name.prefix, name };
		return impl->resolve_name(qualified_name);
	}

	const ImageAttachment& InferenceContext::get_image_attachment(Name name) const {
		auto fqname = QualifiedName{ prefix, name };
		auto resolved_name = erg->impl->resolve_name(fqname);

		auto link = &erg->impl->res_to_links.at(resolved_name); // TODO: no error signaling
		while (link->def->pass > 0) {
			link = link->prev;
		}
		return erg->impl->get_bound_attachment(link->def->pass).attachment;
	}

	const Buffer& InferenceContext::get_buffer(Name name) const {
		auto fqname = QualifiedName{ prefix, name };
		auto resolved_name = erg->impl->resolve_name(fqname);

		auto link = &erg->impl->res_to_links.at(resolved_name); // TODO: no error signaling
		while (link->def->pass > 0) {
			link = link->prev;
		}
		return erg->impl->get_bound_buffer(link->def->pass).buffer;
	}
} // namespace vuk
