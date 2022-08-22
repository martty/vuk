#include "Cache.hpp"
#include "RenderGraphImpl.hpp"
#include "vuk/AllocatorHelpers.hpp"
#include "vuk/CommandBuffer.hpp"
#include "vuk/Context.hpp"
#include "vuk/Future.hpp"
#include "vuk/Hash.hpp" // for create
#include "vuk/RenderGraph.hpp"
#include <sstream>
#include <unordered_set>

namespace vuk {
	ExecutableRenderGraph::ExecutableRenderGraph(Compiler& rg) : impl(rg.impl) {}

	ExecutableRenderGraph::ExecutableRenderGraph(ExecutableRenderGraph&& o) noexcept : impl(std::exchange(o.impl, nullptr)) {}
	ExecutableRenderGraph& ExecutableRenderGraph::operator=(ExecutableRenderGraph&& o) noexcept {
		impl = std::exchange(o.impl, nullptr);
		return *this;
	}

	ExecutableRenderGraph::~ExecutableRenderGraph() {}

	void ExecutableRenderGraph::create_attachment(Context& ctx, AttachmentInfo& attachment_info) {
		if (attachment_info.type == AttachmentInfo::Type::eInternal) {
			vuk::ImageUsageFlags usage = {};
			for (auto& void_chain : attachment_info.use_chains) {
				auto& chain = *reinterpret_cast<std::vector<UseRef, short_alloc<UseRef, 64>>*>(void_chain);
				usage |= Compiler::compute_usage(std::span(chain));
			}

			vuk::ImageCreateInfo ici;
			ici.usage = usage;
			ici.arrayLayers = 1;
			ici.flags = attachment_info.attachment.image_flags;
			assert(attachment_info.attachment.extent.sizing != Sizing::eRelative);
			ici.extent = static_cast<vuk::Extent3D>(attachment_info.attachment.extent.extent);
			ici.imageType = attachment_info.attachment.image_type;
			ici.format = attachment_info.attachment.format;
			ici.mipLevels = attachment_info.attachment.level_count;
			ici.arrayLayers = attachment_info.attachment.layer_count;
			ici.initialLayout = vuk::ImageLayout::eUndefined;
			ici.samples = attachment_info.attachment.sample_count.count;
			ici.sharingMode = vuk::SharingMode::eExclusive;
			ici.tiling = attachment_info.attachment.tiling;

			RGCI rgci;
			rgci.name = attachment_info.name;
			rgci.ici = ici;

			auto rg = ctx.acquire_rendertarget(rgci, ctx.get_frame_count());
			attachment_info.attachment.image = rg.image;
		}
	}

	void begin_renderpass(vuk::RenderPassInfo& rpass, VkCommandBuffer& cbuf, bool use_secondary_command_buffers) {
		VkRenderPassBeginInfo rbi{ .sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO };
		rbi.renderPass = rpass.handle;
		rbi.framebuffer = rpass.framebuffer;
		rbi.renderArea = VkRect2D{ vuk::Offset2D{}, vuk::Extent2D{ rpass.fbci.width, rpass.fbci.height } };
		std::vector<VkClearValue> clears(rpass.attachments.size());
		for (size_t i = 0; i < rpass.attachments.size(); i++) {
			auto& att = rpass.attachments[i];
			if (att.clear_value) {
				clears[i] = att.clear_value->c;
			}
		}
		rbi.pClearValues = clears.data();
		rbi.clearValueCount = (uint32_t)clears.size();

		vkCmdBeginRenderPass(cbuf, &rbi, use_secondary_command_buffers ? VK_SUBPASS_CONTENTS_SECONDARY_COMMAND_BUFFERS : VK_SUBPASS_CONTENTS_INLINE);
	}

	[[nodiscard]] bool resolve_image_barrier(const Context& ctx, ImageBarrier& dep, const AttachmentInfo& bound, vuk::DomainFlagBits current_domain) {
		dep.barrier.image = bound.attachment.image;
		// turn base_{layer, level} into absolute values wrt the image
		dep.barrier.subresourceRange.baseArrayLayer += bound.attachment.base_layer;
		dep.barrier.subresourceRange.baseMipLevel += bound.attachment.base_level;
		// clamp arrays and levels to actual accessible range in image, return false if the barrier would refer to no levels or layers
		assert(bound.attachment.layer_count != VK_REMAINING_ARRAY_LAYERS);
		if (dep.barrier.subresourceRange.layerCount != VK_REMAINING_ARRAY_LAYERS) {
			if (dep.barrier.subresourceRange.baseArrayLayer + dep.barrier.subresourceRange.layerCount > bound.attachment.base_layer + bound.attachment.layer_count) {
				int count = static_cast<int32_t>(bound.attachment.layer_count) - (dep.barrier.subresourceRange.baseArrayLayer - bound.attachment.base_layer);
				if (count < 1) {
					return false;
				}
				dep.barrier.subresourceRange.layerCount = static_cast<uint32_t>(count);
			}
		} else {
			if (dep.barrier.subresourceRange.baseArrayLayer > bound.attachment.base_layer + bound.attachment.layer_count) {
				return false;
			}
		}
		assert(bound.attachment.level_count != VK_REMAINING_MIP_LEVELS);
		if (dep.barrier.subresourceRange.levelCount != VK_REMAINING_MIP_LEVELS) {
			if (dep.barrier.subresourceRange.baseMipLevel + dep.barrier.subresourceRange.levelCount > bound.attachment.base_level + bound.attachment.level_count) {
				int count = static_cast<int32_t>(bound.attachment.level_count) - (dep.barrier.subresourceRange.baseMipLevel - bound.attachment.base_level);
				if (count < 1)
					return false;
				dep.barrier.subresourceRange.levelCount = static_cast<uint32_t>(count);
			}
		} else {
			if (dep.barrier.subresourceRange.baseMipLevel > bound.attachment.base_level + bound.attachment.level_count) {
				return false;
			}
		}

		if (dep.barrier.srcQueueFamilyIndex != VK_QUEUE_FAMILY_IGNORED) {
			assert(dep.barrier.dstQueueFamilyIndex != VK_QUEUE_FAMILY_IGNORED);
			bool transition = dep.barrier.dstQueueFamilyIndex != dep.barrier.srcQueueFamilyIndex;
			auto dst_domain = static_cast<vuk::DomainFlagBits>(dep.barrier.dstQueueFamilyIndex);
			dep.barrier.srcQueueFamilyIndex = ctx.domain_to_queue_family_index(static_cast<vuk::DomainFlags>(dep.barrier.srcQueueFamilyIndex));
			dep.barrier.dstQueueFamilyIndex = ctx.domain_to_queue_family_index(static_cast<vuk::DomainFlags>(dep.barrier.dstQueueFamilyIndex));
			if (dep.barrier.srcQueueFamilyIndex == dep.barrier.dstQueueFamilyIndex && transition) {
				if (dst_domain != current_domain) {
					return false; // discard release barriers if they map to the same queue
				}
			}
		}

		return true;
	}

	// TODO: refactor to return RenderPassInfo
	void ExecutableRenderGraph::fill_renderpass_info(vuk::RenderPassInfo& rpass, const size_t& i, vuk::CommandBuffer& cobuf) {
		if (rpass.handle == VK_NULL_HANDLE) {
			cobuf.ongoing_renderpass = {};
			return;
		}
		vuk::CommandBuffer::RenderPassInfo rpi;
		rpi.renderpass = rpass.handle;
		rpi.subpass = (uint32_t)i;
		rpi.extent = vuk::Extent2D{ rpass.fbci.width, rpass.fbci.height };
		auto& spdesc = rpass.rpci.subpass_descriptions[i];
		rpi.color_attachments = std::span<const VkAttachmentReference>(spdesc.pColorAttachments, spdesc.colorAttachmentCount);
		rpi.samples = rpass.fbci.sample_count.count;
		rpi.depth_stencil_attachment = spdesc.pDepthStencilAttachment;
		for (uint32_t i = 0; i < spdesc.colorAttachmentCount; i++) {
			rpi.color_attachment_names[i] = rpass.attachments[spdesc.pColorAttachments[i].attachment].attachment_info->name;
		}
		cobuf.color_blend_attachments.resize(spdesc.colorAttachmentCount);
		cobuf.ongoing_renderpass = rpi;
	}

	Result<SubmitInfo> ExecutableRenderGraph::record_single_submit(Allocator& alloc, std::span<RenderPassInfo> rpis, vuk::DomainFlagBits domain) {
		assert(rpis.size() > 0);

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
		vkBeginCommandBuffer(cbuf, &cbi);

		uint64_t command_buffer_index = rpis[0].command_buffer_index;
		for (auto& rpass : rpis) {
			if (rpass.command_buffer_index != command_buffer_index) { // end old cb and start new one
				if (auto result = vkEndCommandBuffer(cbuf); result != VK_SUCCESS) {
					return { expected_error, VkException{ result } };
				}

				VUK_DO_OR_RETURN(alloc.allocate_command_buffers(std::span{ &*hl_cbuf, 1 }, std::span{ &ci, 1 }));
				si.command_buffers.emplace_back(*hl_cbuf);

				cbuf = hl_cbuf->command_buffer;

				VkCommandBufferBeginInfo cbi{ .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO, .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT };
				vkBeginCommandBuffer(cbuf, &cbi);
			}

			bool use_secondary_command_buffers = rpass.subpasses[0].use_secondary_command_buffers;
			bool is_single_pass = rpass.subpasses.size() == 1 && rpass.subpasses[0].passes.size() == 1;
			if (is_single_pass && !rpass.subpasses[0].passes[0]->qualified_name.is_invalid() && rpass.subpasses[0].passes[0]->pass->execute) {
				ctx.debug.begin_region(cbuf, rpass.subpasses[0].passes[0]->qualified_name);
			}

			for (auto dep : rpass.pre_barriers) {
				auto& bound = impl->bound_attachments[dep.image];
				dep.barrier.image = bound.attachment.image;
				if (!resolve_image_barrier(ctx, dep, bound, domain)) {
					continue;
				}
				vkCmdPipelineBarrier(cbuf, (VkPipelineStageFlags)dep.src, (VkPipelineStageFlags)dep.dst, 0, 0, nullptr, 0, nullptr, 1, &dep.barrier);
			}
			for (const auto& dep : rpass.pre_mem_barriers) {
				vkCmdPipelineBarrier(cbuf, (VkPipelineStageFlags)dep.src, (VkPipelineStageFlags)dep.dst, 0, 1, &dep.barrier, 0, nullptr, 0, nullptr);
			}

			if (rpass.handle != VK_NULL_HANDLE) {
				begin_renderpass(rpass, cbuf, use_secondary_command_buffers);
			}

			for (auto& att : rpass.attachments) {
				if (att.attachment_info->type == AttachmentInfo::Type::eSwapchain) {
					used_swapchains.emplace(att.attachment_info->swapchain);
				}
			}

			for (auto& w : rpass.waits) {
				si.relative_waits.emplace_back(w);
			}

			for (size_t i = 0; i < rpass.subpasses.size(); i++) {
				auto& sp = rpass.subpasses[i];
				// insert image pre-barriers
				if (rpass.handle == VK_NULL_HANDLE) {
					for (auto dep : sp.pre_barriers) {
						auto& bound = impl->bound_attachments[dep.image];
						if (!resolve_image_barrier(ctx, dep, bound, domain)) {
							continue;
						}
						vkCmdPipelineBarrier(cbuf, (VkPipelineStageFlags)dep.src, (VkPipelineStageFlags)dep.dst, 0, 0, nullptr, 0, nullptr, 1, &dep.barrier);
					}
					for (auto& dep : sp.pre_mem_barriers) {
						vkCmdPipelineBarrier(cbuf, (VkPipelineStageFlags)dep.src, (VkPipelineStageFlags)dep.dst, 0, 1, &dep.barrier, 0, nullptr, 0, nullptr);
					}
				} else {
					assert(sp.pre_barriers.empty());
					assert(sp.pre_mem_barriers.empty());
				}
				for (auto& p : sp.passes) {
					CommandBuffer cobuf(*this, ctx, alloc, cbuf);
					fill_renderpass_info(rpass, i, cobuf);
					// propagate signals onto SI
					si.future_signals.insert(si.future_signals.end(), p->future_signals.begin(), p->future_signals.end());

					if (p->pass->execute) {
						cobuf.current_pass = p;
						if (!p->qualified_name.is_invalid() && !is_single_pass) {
							ctx.debug.begin_region(cobuf.command_buffer, p->qualified_name);
							p->pass->execute(cobuf);
							ctx.debug.end_region(cobuf.command_buffer);
						} else {
							p->pass->execute(cobuf);
						}
					}

					if (auto res = cobuf.result(); !res) {
						return res;
					}
				}
				if (i < rpass.subpasses.size() - 1 && rpass.handle != VK_NULL_HANDLE) {
					use_secondary_command_buffers = rpass.subpasses[i + 1].use_secondary_command_buffers;
					vkCmdNextSubpass(cbuf, use_secondary_command_buffers ? VK_SUBPASS_CONTENTS_SECONDARY_COMMAND_BUFFERS : VK_SUBPASS_CONTENTS_INLINE);
				}

				// insert image post-barriers
				if (rpass.handle == VK_NULL_HANDLE) {
					for (auto dep : sp.post_barriers) {
						auto& bound = impl->bound_attachments[dep.image];
						if (!resolve_image_barrier(ctx, dep, bound, domain)) {
							continue;
						}
						vkCmdPipelineBarrier(cbuf, (VkPipelineStageFlags)dep.src, (VkPipelineStageFlags)dep.dst, 0, 0, nullptr, 0, nullptr, 1, &dep.barrier);
					}
					for (const auto& dep : sp.post_mem_barriers) {
						vkCmdPipelineBarrier(cbuf, (VkPipelineStageFlags)dep.src, (VkPipelineStageFlags)dep.dst, 0, 1, &dep.barrier, 0, nullptr, 0, nullptr);
					}
				} else {
					assert(sp.post_barriers.empty());
					assert(sp.post_mem_barriers.empty());
				}
			}
			if (is_single_pass && !rpass.subpasses[0].passes[0]->qualified_name.is_invalid() && rpass.subpasses[0].passes[0]->pass->execute) {
				ctx.debug.end_region(cbuf);
			}
			if (rpass.handle != VK_NULL_HANDLE) {
				vkCmdEndRenderPass(cbuf);
			}
			for (auto dep : rpass.post_barriers) {
				auto& bound = impl->bound_attachments[dep.image];
				if (!resolve_image_barrier(ctx, dep, bound, domain)) {
					continue;
				}
				vkCmdPipelineBarrier(cbuf, (VkPipelineStageFlags)dep.src, (VkPipelineStageFlags)dep.dst, 0, 0, nullptr, 0, nullptr, 1, &dep.barrier);
			}
			for (const auto& dep : rpass.post_mem_barriers) {
				vkCmdPipelineBarrier(cbuf, (VkPipelineStageFlags)dep.src, (VkPipelineStageFlags)dep.dst, 0, 1, &dep.barrier, 0, nullptr, 0, nullptr);
			}
		}

		if (auto result = vkEndCommandBuffer(cbuf); result != VK_SUCCESS) {
			return { expected_error, VkException{ result } };
		}

		si.used_swapchains.insert(si.used_swapchains.end(), used_swapchains.begin(), used_swapchains.end());

		return { expected_value, std::move(si) };
	}

	Result<SubmitBundle> ExecutableRenderGraph::execute(Allocator& alloc, std::vector<std::pair<SwapchainRef, size_t>> swp_with_index) {
		Context& ctx = alloc.get_context();

		// bind swapchain attachment images & ivs
		for (auto& [name, bound] : impl->bound_attachments) {
			if (bound.type == AttachmentInfo::Type::eSwapchain) {
				auto it = std::find_if(swp_with_index.begin(), swp_with_index.end(), [boundb = &bound](auto& t) { return t.first == boundb->swapchain; });
				bound.attachment.image_view = it->first->image_views[it->second];
				bound.attachment.image = it->first->images[it->second];
				bound.attachment.extent = Dimension3D::absolute(it->first->extent);
				bound.attachment.sample_count = vuk::Samples::e1;
			}
		}

		// pre-inference: which IAs are in which FBs?
		for (auto& rp : impl->rpis) {
			for (auto& rp_att : rp.attachments) {
				auto& att = *rp_att.attachment_info;

				if (!rp.framebufferless) { // framebuffers get extra inference
					att.rp_uses.emplace_back(&rp);
					auto& ia = att.attachment;
					ia.image_type = ia.image_type == ImageType::eInfer ? vuk::ImageType::e2D : ia.image_type;

					ia.base_layer = ia.base_layer == VK_REMAINING_ARRAY_LAYERS ? 0 : ia.base_layer;
					ia.layer_count = ia.layer_count == VK_REMAINING_ARRAY_LAYERS ? 1 : ia.layer_count; // TODO: this prevents inference later on, so this means we are doing it too early
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

					rp.fbci.layers = rp.layer_count;
				}

				// resolve images are always sample count 1
				if (rp_att.is_resolve_dst) {
					att.attachment.sample_count = Samples::e1;
				}
			}
		}

		decltype(impl->ia_inference_rules) resolved_rules;
		for (auto& [n, rules] : impl->ia_inference_rules) {
			resolved_rules.emplace(impl->resolve_name(n), std::move(rules));
		}

		std::vector<std::pair<AttachmentInfo*, IAInferences*>> attis_to_infer;
		for (auto& [name, bound] : impl->bound_attachments) {
			if (bound.type == AttachmentInfo::Type::eInternal) {
				// compute usage if it is to be inferred
				if (bound.attachment.usage == ImageUsageFlagBits::eInfer) {
					bound.attachment.usage = {};
					for (auto& void_chain : bound.use_chains) {
						auto& chain = *reinterpret_cast<std::vector<UseRef, short_alloc<UseRef, 64>>*>(void_chain);
						bound.attachment.usage |= Compiler::compute_usage(std::span(chain));
					}
				}
				// if there is no image, then we will infer the base mip and layer to be 0
				if (bound.attachment.image == VK_NULL_HANDLE) {
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
				auto rules_it = resolved_rules.find(name);
				if (rules_it != resolved_rules.end()) {
					rules_ptr = &rules_it->second;
				}

				attis_to_infer.emplace_back(&bound, rules_ptr);
			}
		}

		InferenceContext inf_ctx{ this };
		bool infer_progress = true;
		std::stringstream msg;

		// we provide an upper bound of 100 inference to iteration to catch infinite loops that don't converge to a fixpoint
		for (size_t i = 0; i < 100 && !attis_to_infer.empty() && infer_progress; i++) {
			infer_progress = false;
			for (auto ia_it = attis_to_infer.begin(); ia_it != attis_to_infer.end();) {
				auto& atti = *ia_it->first;
				auto& ia = atti.attachment;
				auto prev = ia;
				// infer FB -> IA
				if (ia.sample_count == Samples::eInfer || (ia.extent.extent.width == 0 && ia.extent.extent.height == 0) ||
				    ia.extent.sizing == Sizing::eRelative) { // this IA can potentially take inference from an FB
					for (auto* rpi : atti.rp_uses) {
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
						msg << "Rule broken for attachment[" << atti.name.c_str() << "] :\n ";
						msg << " base layer was previously known to be " << prev.base_layer << ", but now set to " << ia.base_layer;
						return { expected_error, RenderGraphException{ msg.str() } };
					}
					if (prev.layer_count != ia.layer_count && prev.layer_count != VK_REMAINING_ARRAY_LAYERS) {
						msg << "Rule broken for attachment[" << atti.name.c_str() << "] :\n ";
						msg << " layer count was previously known to be " << prev.layer_count << ", but now set to " << ia.layer_count;
						return { expected_error, RenderGraphException{ msg.str() } };
					}
					if (prev.base_level != ia.base_level && prev.base_level != VK_REMAINING_MIP_LEVELS) {
						msg << "Rule broken for attachment[" << atti.name.c_str() << "] :\n ";
						msg << " base level was previously known to be " << prev.base_level << ", but now set to " << ia.base_level;
						return { expected_error, RenderGraphException{ msg.str() } };
					}
					if (prev.level_count != ia.level_count && prev.level_count != VK_REMAINING_MIP_LEVELS) {
						msg << "Rule broken for attachment[" << atti.name.c_str() << "] :\n ";
						msg << " level count was previously known to be " << prev.level_count << ", but now set to " << ia.level_count;
						return { expected_error, RenderGraphException{ msg.str() } };
					}
					if (prev.format != ia.format && prev.format != Format::eUndefined) {
						msg << "Rule broken for attachment[" << atti.name.c_str() << "] :\n ";
						msg << " format was previously known to be " << format_to_sv(prev.format) << ", but now set to " << format_to_sv(ia.format);
						return { expected_error, RenderGraphException{ msg.str() } };
					}
					if (prev.sample_count != ia.sample_count && prev.sample_count != SampleCountFlagBits::eInfer) {
						msg << "Rule broken for attachment[" << atti.name.c_str() << "] :\n ";
						msg << " sample count was previously known to be " << static_cast<uint32_t>(prev.sample_count.count) << ", but now set to "
						    << static_cast<uint32_t>(ia.sample_count.count);
						return { expected_error, RenderGraphException{ msg.str() } };
					}
					if (prev.extent.extent.width != ia.extent.extent.width && prev.extent.extent.width != 0) {
						msg << "Rule broken for attachment[" << atti.name.c_str() << "] :\n ";
						msg << " extent.width was previously known to be " << prev.extent.extent.width << ", but now set to " << ia.extent.extent.width;
						return { expected_error, RenderGraphException{ msg.str() } };
					}
					if (prev.extent.extent.height != ia.extent.extent.height && prev.extent.extent.height != 0) {
						msg << "Rule broken for attachment[" << atti.name.c_str() << "] :\n ";
						msg << " extent.height was previously known to be " << prev.extent.extent.height << ", but now set to " << ia.extent.extent.height;
						return { expected_error, RenderGraphException{ msg.str() } };
					}
					if (prev.extent.extent.depth != ia.extent.extent.depth && prev.extent.extent.depth != 0) {
						msg << "Rule broken for attachment[" << atti.name.c_str() << "] :\n ";
						msg << " extent.depth was previously known to be " << prev.extent.extent.depth << ", but now set to " << ia.extent.extent.depth;
						return { expected_error, RenderGraphException{ msg.str() } };
					}
					if (ia.may_require_image_view() && prev.view_type != ia.view_type && prev.view_type != ImageViewType::eInfer) {
						msg << "Rule broken for attachment[" << atti.name.c_str() << "] :\n ";
						msg << " view type was previously known to be " << image_view_type_to_sv(prev.view_type) << ", but now set to "
						    << image_view_type_to_sv(ia.view_type);
						return { expected_error, RenderGraphException{ msg.str() } };
					}

					infer_progress = true;
					// infer IA -> FB
					if (ia.sample_count == Samples::eInfer && (ia.extent.extent.width == 0 && ia.extent.extent.height == 0)) { // this IA is not helpful for FB inference
						continue;
					}
					for (auto* rpi : atti.rp_uses) {
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
							auto it =
							    std::find_if(rpi->attachments.begin(), rpi->attachments.end(), [attip = &atti](auto& rp_att) { return rp_att.attachment_info == attip; });
							assert(it != rpi->attachments.end());
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
			msg << "Could not infer attachment [" << atti->name.c_str() << "]:\n";
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
			// TODO: error harmonization
#ifndef NDEBUG
			fprintf(stderr, "%s", msg.str().c_str());
			assert(false);
#endif
			return { expected_error, RenderGraphException{ msg.str() } };
		}

		// acquire the renderpasses
		for (auto& rp : impl->rpis) {
			if (rp.attachments.size() == 0) {
				continue;
			}

			for (auto& attrpinfo : rp.attachments) {
				attrpinfo.description.format = (VkFormat)attrpinfo.attachment_info->attachment.format;
				attrpinfo.description.samples = (VkSampleCountFlagBits)attrpinfo.attachment_info->attachment.sample_count.count;
				rp.rpci.attachments.push_back(attrpinfo.description);
			}

			rp.rpci.attachmentCount = (uint32_t)rp.rpci.attachments.size();
			rp.rpci.pAttachments = rp.rpci.attachments.data();

			rp.handle = ctx.acquire_renderpass(rp.rpci, ctx.get_frame_count());
		}

		// create non-attachment images
		for (auto& [name, bound] : impl->bound_attachments) {
			if (bound.attachment.image == Image{}) {
				if (bound.rp_uses.size() > 0) { // its an FB attachment
					create_attachment(ctx, bound);
				} else {
					auto image = *allocate_image(alloc, bound.attachment); // TODO: dropping error
					ctx.debug.set_name(*image, bound.name);
					bound.attachment.image = *image;
				}
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
			for (auto& attrpinfo : rp.attachments) {
				auto& bound = *attrpinfo.attachment_info;
				std::optional<uint32_t> base_layer;
				std::optional<uint32_t> layer_count;
				for (auto& sp : rp.subpasses) {
					auto& pi = *sp.passes[0]; // all passes should be using the same fb, so we can pick the first
					for (auto& res : pi.resources) {
						auto resolved_name = impl->resolve_name(res.name);
						auto whole_name = impl->whole_name(resolved_name);
						if (whole_name == bound.name) {
							auto& sr = impl->use_chains.at(resolved_name)[1].subrange;
							base_layer = sr.image.base_layer;
							layer_count = sr.image.layer_count;
						}
					}
				}
				assert(base_layer);
				assert(layer_count);
				fb_layer_count = bound.attachment.layer_count;

				auto specific_attachment = bound.attachment;
				if (specific_attachment.image_view == ImageView{}) {
					specific_attachment.base_layer = *base_layer;
					if (specific_attachment.view_type == ImageViewType::eCube) {
						if (*layer_count > 1) {
							specific_attachment.view_type = ImageViewType::e2DArray;
						} else {
							specific_attachment.view_type = ImageViewType::e2D;
						}
					}
					specific_attachment.layer_count = *layer_count;
					assert(specific_attachment.level_count == 1);

					auto iv = *allocate_image_view(alloc, specific_attachment); // TODO: dropping error
					specific_attachment.image_view = *iv;
					auto name = std::string("ImageView: RenderTarget ") + std::string(bound.name.to_sv());
					ctx.debug.set_name(specific_attachment.image_view.payload, Name(name));
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
			if (rp.fbci.layers == VK_REMAINING_ARRAY_LAYERS) {
				rp.fbci.layers = *fb_layer_count;
			}

			Unique<VkFramebuffer> fb(alloc);
			VUK_DO_OR_RETURN(alloc.allocate_framebuffers(std::span{ &*fb, 1 }, std::span{ &rp.fbci, 1 }));
			rp.framebuffer = *fb; // queue framebuffer for destruction
		}

		for (auto& [name, attachment_info] : impl->bound_attachments) {
			if (attachment_info.attached_future) {
				ImageAttachment att = attachment_info.attachment;
				attachment_info.attached_future->result = att;
			}
		}

		SubmitBundle sbundle;

		auto record_batch = [&alloc, this](std::span<RenderPassInfo> rpis, DomainFlagBits domain) {
			SubmitBatch sbatch{ .domain = domain };
			auto partition_it = rpis.begin();
			while (partition_it != rpis.end()) {
				auto batch_index = partition_it->batch_index;
				auto new_partition_it =
				    std::partition_point(partition_it, rpis.end(), [batch_index](const RenderPassInfo& rpi) { return rpi.batch_index == batch_index; });
				auto partition_span = std::span(partition_it, new_partition_it);
				auto si = record_single_submit(alloc, partition_span, domain);
				sbatch.submits.emplace_back(*si); // TODO: error handling
				partition_it = new_partition_it;
			}
			return sbatch;
		};

		// record cbufs
		// assume that rpis are partitioned wrt batch_index

		auto graphics_rpis = std::span(impl->rpis.begin(), impl->rpis.begin() + impl->num_graphics_rpis);
		if (graphics_rpis.size() > 0) {
			sbundle.batches.emplace_back(record_batch(graphics_rpis, DomainFlagBits::eGraphicsQueue));
		}

		auto compute_rpis = std::span(impl->rpis.begin() + impl->num_graphics_rpis, impl->rpis.begin() + impl->num_graphics_rpis + impl->num_compute_rpis);
		if (compute_rpis.size() > 0) {
			sbundle.batches.emplace_back(record_batch(compute_rpis, DomainFlagBits::eComputeQueue));
		}

		auto transfer_rpis = std::span(impl->rpis.begin() + impl->num_graphics_rpis + impl->num_compute_rpis, impl->rpis.end());
		if (transfer_rpis.size() > 0) {
			sbundle.batches.emplace_back(record_batch(transfer_rpis, DomainFlagBits::eTransferQueue));
		}

		return { expected_value, std::move(sbundle) };
	}

	Result<BufferInfo, RenderGraphException> ExecutableRenderGraph::get_resource_buffer(Name n, PassInfo* pass_info) {
		auto resolved = resolve_name(n, pass_info);
		auto whole = impl->whole_name(resolved);
		auto it = impl->bound_buffers.find(whole);
		if (it == impl->bound_buffers.end()) {
			return { expected_error, RenderGraphException{ "Buffer not found" } };
		}
		return { expected_value, it->second };
	}

	Result<AttachmentInfo, RenderGraphException> ExecutableRenderGraph::get_resource_image(Name n, PassInfo* pass_info) {
		auto resolved = resolve_name(n, pass_info);
		auto whole = impl->whole_name(resolved);
		auto it = impl->bound_attachments.find(whole);
		if (it == impl->bound_attachments.end()) {
			return { expected_error, RenderGraphException{ "Image not found" } };
		}
		auto uc_it = impl->use_chains.find(resolved);
		if (uc_it == impl->use_chains.end()) {
			return { expected_error, RenderGraphException{ "Resource not found" } };
		}
		auto& chain = uc_it->second;
		for (auto& elem : chain) {
			if (elem.pass == pass_info) {
				// TODO: make this less expensive
				auto attI = it->second;
				attI.attachment.base_layer = elem.subrange.image.base_layer;
				attI.attachment.base_level = elem.subrange.image.base_level;
				attI.attachment.layer_count =
				    elem.subrange.image.layer_count == VK_REMAINING_ARRAY_LAYERS ? attI.attachment.layer_count : elem.subrange.image.layer_count;
				attI.attachment.level_count =
				    elem.subrange.image.level_count == VK_REMAINING_MIP_LEVELS ? attI.attachment.level_count : elem.subrange.image.level_count;
				return { expected_value, attI };
			}
		}
		return { expected_value, it->second };
	}

	Result<bool, RenderGraphException> ExecutableRenderGraph::is_resource_image_in_general_layout(Name n, PassInfo* pass_info) {
		auto resolved = resolve_name(n, pass_info);

		auto it = impl->use_chains.find(resolved);
		if (it == impl->use_chains.end()) {
			return { expected_error, RenderGraphException{ "Resource not found" } };
		}
		auto& chain = it->second;
		for (auto& elem : chain) {
			if (elem.pass == pass_info) {
				return { expected_value, elem.use.layout == vuk::ImageLayout::eGeneral };
			}
		}
		assert(false && "Image resource was not declared to be used in this pass, but was referred to.");
		return { expected_error, RenderGraphException{ "Image resourced was not declared to be used in this pass, but was referred to." } };
	}

	Name ExecutableRenderGraph::resolve_name(Name name, PassInfo* pass_info) const noexcept {
		auto qualified_name = pass_info->prefix.is_invalid() ? name : pass_info->prefix.append(name.to_sv());
		return impl->resolve_name(qualified_name);
	}

	const ImageAttachment& InferenceContext::get_image_attachment(Name name) const {
		auto fqname = prefix.append(name.to_sv());
		auto resolved_name = erg->impl->resolve_name(fqname);
		auto whole_name = erg->impl->whole_name(resolved_name);
		return erg->impl->bound_attachments.at(whole_name).attachment;
	}
} // namespace vuk
