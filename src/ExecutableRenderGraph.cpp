#include "Cache.hpp"
#include "RenderGraphImpl.hpp"
#include "vuk/CommandBuffer.hpp"
#include "vuk/Context.hpp"
#include "vuk/Future.hpp"
#include "vuk/Hash.hpp" // for create
#include "vuk/RenderGraph.hpp"
#include <unordered_set>

namespace vuk {
	ExecutableRenderGraph::ExecutableRenderGraph(RenderGraph&& rg) : impl(rg.impl) {
		rg.impl = nullptr; // pilfered
	}

	ExecutableRenderGraph::ExecutableRenderGraph(ExecutableRenderGraph&& o) noexcept : impl(std::exchange(o.impl, nullptr)) {}
	ExecutableRenderGraph& ExecutableRenderGraph::operator=(ExecutableRenderGraph&& o) noexcept {
		impl = std::exchange(o.impl, nullptr);
		return *this;
	}

	ExecutableRenderGraph::~ExecutableRenderGraph() {
		delete impl;
	}

	void ExecutableRenderGraph::create_attachment(Context& ctx,
	                                              Name name,
	                                              AttachmentRPInfo& attachment_info,
	                                              vuk::Extent2D fb_extent,
	                                              vuk::SampleCountFlagBits samples) {
		auto& chain = impl->use_chains.at(name);
		if (attachment_info.type == AttachmentRPInfo::Type::eInternal) {
			vuk::ImageUsageFlags usage = RenderGraph::compute_usage(std::span(chain));

			vuk::ImageCreateInfo ici;
			ici.usage = usage;
			ici.arrayLayers = 1;
			// compute extent
			if (attachment_info.attachment.extent.sizing == Sizing::eRelative) {
				assert(fb_extent.width > 0 && fb_extent.height > 0);
				ici.extent = vuk::Extent3D{ static_cast<uint32_t>(attachment_info.attachment.extent._relative.width * fb_extent.width),
					                          static_cast<uint32_t>(attachment_info.attachment.extent._relative.height * fb_extent.height),
					                          1u };
			} else {
				ici.extent = static_cast<vuk::Extent3D>(attachment_info.attachment.extent.extent);
			}
			// concretize attachment size
			attachment_info.attachment.extent = Dimension2D::absolute(ici.extent.width, ici.extent.height);
			ici.imageType = vuk::ImageType::e2D;
			ici.format = vuk::Format(attachment_info.description.format);
			ici.mipLevels = 1;
			ici.initialLayout = vuk::ImageLayout::eUndefined;
			ici.samples = samples;
			ici.sharingMode = vuk::SharingMode::eExclusive;
			ici.tiling = vuk::ImageTiling::eOptimal;

			vuk::ImageViewCreateInfo ivci;
			ivci.image = vuk::Image{};
			ivci.format = vuk::Format(attachment_info.description.format);
			ivci.viewType = vuk::ImageViewType::e2D;
			vuk::ImageSubresourceRange isr;

			isr.aspectMask = format_to_aspect(ici.format);
			isr.baseArrayLayer = 0;
			isr.layerCount = 1;
			isr.baseMipLevel = 0;
			isr.levelCount = 1;
			ivci.subresourceRange = isr;

			RGCI rgci;
			rgci.name = name;
			rgci.ici = ici;
			rgci.ivci = ivci;

			auto rg = ctx.acquire_rendertarget(rgci, ctx.get_frame_count());
			attachment_info.attachment.image_view = rg.image_view;
			attachment_info.attachment.layer_count = 1;
			attachment_info.attachment.level_count = 1;
			attachment_info.attachment.image = rg.image;
		}
	}

	Result<void> create_image(Allocator& alloc, ImageAttachment& attachment, vuk::Extent2D fb_extent, vuk::SampleCountFlagBits samples) {
		ImageCreateInfo ici;
		ici.format = vuk::Format(attachment.format);
		ici.imageType = attachment.imageType;
		ici.flags = attachment.image_flags;
		ici.arrayLayers = attachment.layer_count;
		if (attachment.extent.sizing == Sizing::eRelative) {
			assert(fb_extent.width > 0 && fb_extent.height > 0);
			ici.extent = vuk::Extent3D{ static_cast<uint32_t>(attachment.extent._relative.width * fb_extent.width),
				                          static_cast<uint32_t>(attachment.extent._relative.height * fb_extent.height),
				                          1u };
		} else {
			ici.extent = static_cast<vuk::Extent3D>(attachment.extent.extent);
		}

		return alloc.allocate_images(std::span{ &attachment.image, 1 }, std::span{ &ici, 1 });
	}

	Result<void> create_image_view(Allocator& alloc, ImageAttachment& attachment) {
		ImageViewCreateInfo ivci;
		ivci.image = attachment.image;
		ivci.format = vuk::Format(attachment.format);
		ivci.viewType = attachment.viewType;
		ivci.flags = attachment.image_view_flags;
		ivci.viewType = attachment.viewType;

		ImageSubresourceRange isr;
		isr.aspectMask = format_to_aspect(ivci.format);
		isr.baseArrayLayer = attachment.base_layer;
		isr.layerCount = attachment.layer_count;
		isr.baseMipLevel = attachment.base_level;
		isr.levelCount = attachment.level_count;
		ivci.subresourceRange = isr;

		return alloc.allocate_image_views(std::span{ &attachment.image_view, 1 }, std::span{ &ivci, 1 });
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

	[[nodiscard]] bool resolve_image_barrier(const Context& ctx, ImageBarrier& dep, const AttachmentRPInfo& bound) {
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
			dep.barrier.srcQueueFamilyIndex = ctx.domain_to_queue_family_index(static_cast<vuk::DomainFlags>(dep.barrier.srcQueueFamilyIndex));
		}
		if (dep.barrier.dstQueueFamilyIndex != VK_QUEUE_FAMILY_IGNORED) {
			dep.barrier.dstQueueFamilyIndex = ctx.domain_to_queue_family_index(static_cast<vuk::DomainFlags>(dep.barrier.dstQueueFamilyIndex));
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
			rpi.color_attachment_names[i] = rpass.attachments[spdesc.pColorAttachments[i].attachment].name;
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
			if (is_single_pass && !rpass.subpasses[0].passes[0]->pass.name.is_invalid() && rpass.subpasses[0].passes[0]->pass.execute) {
				ctx.debug.begin_region(cbuf, rpass.subpasses[0].passes[0]->pass.name);
			}

			for (auto dep : rpass.pre_barriers) {
				auto& bound = impl->bound_attachments[dep.image];
				dep.barrier.image = bound.attachment.image;
				if (!resolve_image_barrier(ctx, dep, bound)) {
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
				if (att.type == AttachmentRPInfo::Type::eSwapchain) {
					used_swapchains.emplace(impl->bound_attachments.at(att.name).swapchain);
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
						if (!resolve_image_barrier(ctx, dep, bound)) {
							continue;
						}
						vkCmdPipelineBarrier(cbuf, (VkPipelineStageFlags)dep.src, (VkPipelineStageFlags)dep.dst, 0, 0, nullptr, 0, nullptr, 1, &dep.barrier);
					}
					for (auto& dep : sp.pre_mem_barriers) {
						vkCmdPipelineBarrier(cbuf, (VkPipelineStageFlags)dep.src, (VkPipelineStageFlags)dep.dst, 0, 1, &dep.barrier, 0, nullptr, 0, nullptr);
					}
				}
				for (auto& p : sp.passes) {
					CommandBuffer cobuf(*this, ctx, alloc, cbuf);
					fill_renderpass_info(rpass, i, cobuf);
					// propagate waits & signals onto SI
					if (p->pass.signal) {
						si.future_signals.emplace_back(p->pass.signal);
					}

					if (p->pass.execute) {
						cobuf.current_pass = p;
						if (!p->pass.name.is_invalid() && !is_single_pass) {
							ctx.debug.begin_region(cobuf.command_buffer, p->pass.name);
							p->pass.execute(cobuf);
							ctx.debug.end_region(cobuf.command_buffer);
						} else {
							p->pass.execute(cobuf);
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
						if (!resolve_image_barrier(ctx, dep, bound)) {
							continue;
						}
						vkCmdPipelineBarrier(cbuf, (VkPipelineStageFlags)dep.src, (VkPipelineStageFlags)dep.dst, 0, 0, nullptr, 0, nullptr, 1, &dep.barrier);
					}
					for (const auto& dep : sp.post_mem_barriers) {
						vkCmdPipelineBarrier(cbuf, (VkPipelineStageFlags)dep.src, (VkPipelineStageFlags)dep.dst, 0, 1, &dep.barrier, 0, nullptr, 0, nullptr);
					}
				}
			}
			if (is_single_pass && !rpass.subpasses[0].passes[0]->pass.name.is_invalid() && rpass.subpasses[0].passes[0]->pass.execute) {
				ctx.debug.end_region(cbuf);
			}
			if (rpass.handle != VK_NULL_HANDLE) {
				vkCmdEndRenderPass(cbuf);
			}
			for (auto dep : rpass.post_barriers) {
				auto& bound = impl->bound_attachments[dep.image];
				if (!resolve_image_barrier(ctx, dep, bound)) {
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

		// perform sample count inference for framebuffers
		// loop through all renderpasses, and attempt to infer any sample count we
		// can then loop again, stopping if we have inferred all or have not made
		// progress resolve images are always sample count 1 and are excluded from
		// the inference
		bool infer_progress = false;
		bool any_fb_incomplete = false;
		do {
			any_fb_incomplete = false;
			infer_progress = false;
			for (auto& rp : impl->rpis) {
				if (rp.attachments.size() == 0) {
					continue;
				}

				Samples fb_samples = rp.fbci.sample_count;
				bool samples_known = fb_samples != Samples::eInfer;

				if (samples_known) {
					continue;
				}

				// see if any attachment has a set sample count
				for (auto& attrpinfo : rp.attachments) {
					auto& bound = impl->bound_attachments[attrpinfo.name];

					if (bound.attachment.sample_count != Samples::eInfer && !attrpinfo.is_resolve_dst) {
						fb_samples = bound.attachment.sample_count;
						samples_known = true;
						break;
					}
				}

				// propagate known sample count onto attachments
				if (samples_known) {
					for (auto& attrpinfo : rp.attachments) {
						auto& bound = impl->bound_attachments[attrpinfo.name];
						if (!attrpinfo.is_resolve_dst) {
							bound.attachment.sample_count = fb_samples;
						}
					}
					rp.fbci.sample_count = fb_samples;
					infer_progress = true; // progress made
				} else {
					any_fb_incomplete = true;
				}
			}
		} while (any_fb_incomplete && infer_progress); // stop looping if all attachment have been sized
		                                               // or we made no progress

		assert(!any_fb_incomplete && "Failed to infer sample count for all attachments.");

		// finish by acquiring the renderpasses
		for (auto& rp : impl->rpis) {
			if (rp.attachments.size() == 0) {
				continue;
			}

			for (auto& attrpinfo : rp.attachments) {
				if (attrpinfo.is_resolve_dst) {
					attrpinfo.description.samples = VK_SAMPLE_COUNT_1_BIT;
				} else {
					attrpinfo.description.samples = (VkSampleCountFlagBits)rp.fbci.sample_count.count;
				}
				rp.rpci.attachments.push_back(attrpinfo.description);
			}

			rp.rpci.attachmentCount = (uint32_t)rp.rpci.attachments.size();
			rp.rpci.pAttachments = rp.rpci.attachments.data();

			rp.handle = ctx.acquire_renderpass(rp.rpci, ctx.get_frame_count());
		}

		// bind swapchain attachment images & ivs
		for (auto& [name, bound] : impl->bound_attachments) {
			if (bound.type == AttachmentRPInfo::Type::eSwapchain) {
				auto it = std::find_if(swp_with_index.begin(), swp_with_index.end(), [boundb = &bound](auto& t) { return t.first == boundb->swapchain; });
				bound.attachment.image_view = it->first->image_views[it->second];
				bound.attachment.image = it->first->images[it->second];
				bound.attachment.extent = Dimension2D::absolute(it->first->extent);
				bound.attachment.sample_count = vuk::Samples::e1;
			}
		}

		// perform size inference for framebuffers (we need to do this here due to swapchain attachments)
		// loop through all renderpasses, and attempt to infer any size we can
		// then loop again, stopping if we have inferred all or have not made progress
		do {
			any_fb_incomplete = false;
			infer_progress = false;
			for (auto& rp : impl->rpis) {
				if (rp.attachments.size() == 0) {
					continue;
				}

				// an extent is known if it is not 0
				// 0 sized framebuffers are illegal
				Extent2D fb_extent = Extent2D{ rp.fbci.width, rp.fbci.height };
				bool extent_known = !(fb_extent.width == 0 || fb_extent.height == 0);

				if (extent_known) {
					continue;
				}

				// see if any attachment has an absolute size
				for (auto& attrpinfo : rp.attachments) {
					auto& bound = impl->bound_attachments[attrpinfo.name];

					if (bound.attachment.extent.sizing == vuk::Sizing::eAbsolute && bound.attachment.extent.extent.width > 0 &&
					    bound.attachment.extent.extent.height > 0) {
						fb_extent = bound.attachment.extent.extent;
						extent_known = true;
						break;
					}
				}

				if (extent_known) {
					rp.fbci.width = fb_extent.width;
					rp.fbci.height = fb_extent.height;

					for (auto& attrpinfo : rp.attachments) {
						auto& bound = impl->bound_attachments[attrpinfo.name];
						bound.attachment.extent = Dimension2D::absolute(fb_extent);
					}

					infer_progress = true; // progress made
				}

				if (!extent_known) {
					any_fb_incomplete = true;
				}
			}
		} while (any_fb_incomplete && infer_progress); // stop looping if all attachment have been sized or we made no progress

		assert(!any_fb_incomplete && "Failed to infer size for all attachments.");

		// create framebuffers, create & bind attachments
		for (auto& rp : impl->rpis) {
			if (rp.attachments.size() == 0)
				continue;

			auto& ivs = rp.fbci.attachments;
			std::vector<VkImageView> vkivs;

			Extent2D fb_extent = Extent2D{ rp.fbci.width, rp.fbci.height };

			// create internal attachments; bind attachments to fb
			for (auto& attrpinfo : rp.attachments) {
				auto resolved_name = impl->resolve_name(attrpinfo.name);
				auto& bound = impl->bound_attachments[resolved_name];
				if (bound.type == AttachmentRPInfo::Type::eInternal) {
					create_attachment(ctx, resolved_name, bound, fb_extent, (vuk::SampleCountFlagBits)attrpinfo.description.samples);
				}

				ivs.push_back(bound.attachment.image_view);
				vkivs.push_back(bound.attachment.image_view.payload);
			}
			rp.fbci.renderPass = rp.handle;
			rp.fbci.pAttachments = &vkivs[0];
			rp.fbci.width = fb_extent.width;
			rp.fbci.height = fb_extent.height;
			rp.fbci.attachmentCount = (uint32_t)vkivs.size();
			rp.fbci.layers = 1;

			Unique<VkFramebuffer> fb(alloc);
			VUK_DO_OR_RETURN(alloc.allocate_framebuffers(std::span{ &*fb, 1 }, std::span{ &rp.fbci, 1 }));
			rp.framebuffer = *fb; // queue framebuffer for destruction
		}

		// create non-attachment images
		for (auto& [name, bound] : impl->bound_attachments) {
			if (bound.type == AttachmentRPInfo::Type::eInternal && bound.attachment.image == VK_NULL_HANDLE) {
				create_attachment(ctx, name, bound, vuk::Extent2D{ 0, 0 }, bound.attachment.sample_count.count);
			}
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
		auto it = impl->bound_buffers.find(resolved);
		if (it == impl->bound_buffers.end()) {
			return { expected_error, RenderGraphException{ "Buffer not found" } };
		}
		return { expected_value, it->second };
	}

	Result<AttachmentRPInfo, RenderGraphException> ExecutableRenderGraph::get_resource_image(Name n, PassInfo* pass_info) {
		auto resolved = resolve_name(n, pass_info);
		auto it = impl->bound_attachments.find(resolved);
		if (it == impl->bound_attachments.end()) {
			return { expected_error, RenderGraphException{ "Image not found" } };
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
		auto qualified_name = pass_info->prefix.is_invalid() ? name : pass_info->prefix.append(name);
		return impl->resolve_name(qualified_name);
	}
} // namespace vuk
