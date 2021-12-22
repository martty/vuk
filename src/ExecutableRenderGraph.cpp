#include "vuk/RenderGraph.hpp"
#include "vuk/Hash.hpp" // for create
#include "Cache.hpp"
#include "vuk/Context.hpp"
#include "vuk/CommandBuffer.hpp"
#include "RenderGraphImpl.hpp"
#include <unordered_set>

namespace vuk {
	ExecutableRenderGraph::ExecutableRenderGraph(RenderGraph&& rg) : impl(rg.impl) {
		rg.impl = nullptr; //pilfered
	}

	ExecutableRenderGraph::ExecutableRenderGraph(ExecutableRenderGraph&& o) noexcept : impl(std::exchange(o.impl, nullptr)) {}
	ExecutableRenderGraph& ExecutableRenderGraph::operator=(ExecutableRenderGraph&& o) noexcept {
		impl = std::exchange(o.impl, nullptr);
		return *this;
	}

	ExecutableRenderGraph::~ExecutableRenderGraph() {
		delete impl;
	}

	void ExecutableRenderGraph::create_attachment(Context& ctx, Name name, AttachmentRPInfo& attachment_info, vuk::Extent2D fb_extent, vuk::SampleCountFlagBits samples) {
		auto& chain = impl->use_chains.at(name);
		if (attachment_info.type == AttachmentRPInfo::Type::eInternal) {
			vuk::ImageUsageFlags usage = RenderGraph::compute_usage(std::span(chain));

			vuk::ImageCreateInfo ici;
			ici.usage = usage;
			ici.arrayLayers = 1;
			// compute extent
			if (attachment_info.extents.sizing == Sizing::eRelative) {
				assert(fb_extent.width > 0 && fb_extent.height > 0);
				ici.extent = vuk::Extent3D{ static_cast<uint32_t>(attachment_info.extents._relative.width * fb_extent.width), static_cast<uint32_t>(attachment_info.extents._relative.height * fb_extent.height), 1u };
			} else {
				ici.extent = static_cast<vuk::Extent3D>(attachment_info.extents.extent);
			}
			// concretize attachment size
			attachment_info.extents = Dimension2D::absolute(ici.extent.width, ici.extent.height);
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

			auto rg = ctx.acquire_rendertarget(rgci, ctx.frame_counter);
			attachment_info.iv = rg.image_view;
			attachment_info.image = rg.image;
		}
	}

	void begin_renderpass(vuk::RenderPassInfo& rpass, VkCommandBuffer& cbuf, bool use_secondary_command_buffers) {
		VkRenderPassBeginInfo rbi{ .sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO };
		rbi.renderPass = rpass.handle;
		rbi.framebuffer = rpass.framebuffer;
		rbi.renderArea = VkRect2D{ vuk::Offset2D{}, vuk::Extent2D{ rpass.fbci.width, rpass.fbci.height } };
		std::vector<VkClearValue> clears;
		for (size_t i = 0; i < rpass.attachments.size(); i++) {
			auto& att = rpass.attachments[i];
			if (att.should_clear)
				clears.push_back(att.clear_value.c);
		}
		rbi.pClearValues = clears.data();
		rbi.clearValueCount = (uint32_t)clears.size();

		vkCmdBeginRenderPass(cbuf, &rbi, use_secondary_command_buffers ? VK_SUBPASS_CONTENTS_SECONDARY_COMMAND_BUFFERS : VK_SUBPASS_CONTENTS_INLINE);
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

	Result<Unique<CommandBufferAllocation>> ExecutableRenderGraph::execute(Context& ctx, Allocator& alloc, std::vector<std::pair<SwapChainRef, size_t>> swp_with_index) {
		// bind swapchain attachment images & ivs
		for (auto& [name, bound] : impl->bound_attachments) {
			if (bound.type == AttachmentRPInfo::Type::eSwapchain) {
				auto it = std::find_if(swp_with_index.begin(), swp_with_index.end(), [boundb = &bound](auto& t) { return t.first == boundb->swapchain; });
				bound.iv = it->first->image_views[it->second];
				bound.image = it->first->images[it->second];
				bound.extents = Dimension2D::absolute(it->first->extent);
				bound.samples = vuk::Samples::e1;
			}
		}

		// perform size inference for framebuffers (we need to do this here due to swapchain attachments)
		// loop through all renderpasses, and attempt to infer any size we can
		// then loop again, stopping if we have inferred all or have not made progress
		bool infer_progress = false;
		bool any_fb_incomplete = false;
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

					if (bound.extents.sizing == vuk::Sizing::eAbsolute && bound.extents.extent.width > 0 && bound.extents.extent.height > 0) {
						fb_extent = bound.extents.extent;
						extent_known = true;
						break;
					}
				}

				if (extent_known) {
					rp.fbci.width = fb_extent.width;
					rp.fbci.height = fb_extent.height;

					for (auto& attrpinfo : rp.attachments) {
						auto& bound = impl->bound_attachments[attrpinfo.name];
						bound.extents = Dimension2D::absolute(fb_extent);
					}

					infer_progress = true; // progress made
				}

				if (!extent_known) {
					any_fb_incomplete = true;
				}
			}
		} while (any_fb_incomplete || infer_progress); // stop looping if all attachment have been sized or we made no progress

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
				auto& bound = impl->bound_attachments[attrpinfo.name];
				if (bound.type == AttachmentRPInfo::Type::eInternal) {
					create_attachment(ctx, attrpinfo.name, bound, fb_extent, (vuk::SampleCountFlagBits)attrpinfo.description.samples);
				}

				ivs.push_back(bound.iv);
				vkivs.push_back(bound.iv.payload);
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
			if (bound.type == AttachmentRPInfo::Type::eInternal && bound.image == VK_NULL_HANDLE) {
				create_attachment(ctx, name, bound, vuk::Extent2D{ 0,0 }, bound.samples.count);
			}
		}

		// actual execution
		Unique<VkCommandPool> cpool(alloc);
		VkCommandPoolCreateInfo cpci{ VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO };
		cpci.flags = VkCommandPoolCreateFlagBits::VK_COMMAND_POOL_CREATE_TRANSIENT_BIT;
		cpci.queueFamilyIndex = ctx.graphics_queue_family_index; // TODO: hardcoding queue family index
		VUK_DO_OR_RETURN(alloc.allocate_command_pools(std::span{ &*cpool, 1 }, std::span{ &cpci, 1 }));
		Unique<CommandBufferAllocation> hl_cbuf(alloc);
		CommandBufferAllocationCreateInfo ci{ .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY, .command_pool = *cpool};
		VUK_DO_OR_RETURN(alloc.allocate_command_buffers(std::span{ &*hl_cbuf, 1 }, std::span{ &ci, 1 }));

		VkCommandBuffer cbuf = hl_cbuf->command_buffer;

		VkCommandBufferBeginInfo cbi{ .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO, .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT };
		vkBeginCommandBuffer(cbuf, &cbi);

		for (auto& rpass : impl->rpis) {
			bool use_secondary_command_buffers = rpass.subpasses[0].use_secondary_command_buffers;
			bool is_single_pass = rpass.subpasses.size() == 1 && rpass.subpasses[0].passes.size() == 1;
			if (is_single_pass) {
				ctx.debug.begin_region(cbuf, rpass.subpasses[0].passes[0]->pass.name);
			}

			for (auto dep : rpass.pre_barriers) {
				dep.barrier.image = impl->bound_attachments[dep.image].image;
				vkCmdPipelineBarrier(cbuf, (VkPipelineStageFlags)dep.src, (VkPipelineStageFlags)dep.dst, 0, 0, nullptr, 0, nullptr, 1, &dep.barrier);
			}
			for (const auto& dep : rpass.pre_mem_barriers) {
				vkCmdPipelineBarrier(cbuf, (VkPipelineStageFlags)dep.src, (VkPipelineStageFlags)dep.dst, 0, 1, &dep.barrier, 0, nullptr, 0, nullptr);
			}

			if (rpass.handle != VK_NULL_HANDLE) {
				begin_renderpass(rpass, cbuf, use_secondary_command_buffers);
			}

			for (size_t i = 0; i < rpass.subpasses.size(); i++) {
				auto& sp = rpass.subpasses[i];
				// insert image pre-barriers
				if (rpass.handle == VK_NULL_HANDLE) {
					for (auto dep : sp.pre_barriers) {
						dep.barrier.image = impl->bound_attachments[dep.image].image;
						vkCmdPipelineBarrier(cbuf, (VkPipelineStageFlags)dep.src, (VkPipelineStageFlags)dep.dst, 0, 0, nullptr, 0, nullptr, 1, &dep.barrier);
					}
					for (const auto& dep : sp.pre_mem_barriers) {
						vkCmdPipelineBarrier(cbuf, (VkPipelineStageFlags)dep.src, (VkPipelineStageFlags)dep.dst, 0, 1, &dep.barrier, 0, nullptr, 0, nullptr);
					}
				}
				for (auto& p : sp.passes) {
					CommandBuffer cobuf(*this, ctx, alloc, cbuf);
					fill_renderpass_info(rpass, i, cobuf);
					// if pass requested no secondary cbufs, but due to subpass merging that is what we got
					if (p->pass.use_secondary_command_buffers == false && use_secondary_command_buffers == true) {
						auto res = cobuf.begin_secondary();
						if (!res) {
							return { expected_error, res.error() };
						}
						auto secondary = *res;
						if (p->pass.execute) {
							secondary.current_pass = p;
							if (!p->pass.name.is_invalid() && !is_single_pass) {
								ctx.debug.begin_region(cobuf.command_buffer, p->pass.name);
								p->pass.execute(secondary);
								ctx.debug.end_region(cobuf.command_buffer);
							} else {
								p->pass.execute(secondary);
							}
						}
						if (secondary.has_error()) {
							return { expected_error, secondary.error() };
						}
						auto secondary_cbuf = secondary.get_buffer();
						cobuf.execute({ &secondary_cbuf, 1 });
					} else {
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
					}
					if (cobuf.has_error()) {
						return { expected_error, cobuf.error() };
					}
				}
				if (i < rpass.subpasses.size() - 1 && rpass.handle != VK_NULL_HANDLE) {
					use_secondary_command_buffers = rpass.subpasses[i + 1].use_secondary_command_buffers;
					vkCmdNextSubpass(cbuf, use_secondary_command_buffers ? VK_SUBPASS_CONTENTS_SECONDARY_COMMAND_BUFFERS : VK_SUBPASS_CONTENTS_INLINE);
				}

				// insert image post-barriers
				if (rpass.handle == VK_NULL_HANDLE) {
					for (auto dep : sp.post_barriers) {
						dep.barrier.image = impl->bound_attachments[dep.image].image;
						vkCmdPipelineBarrier(cbuf, (VkPipelineStageFlags)dep.src, (VkPipelineStageFlags)dep.dst, 0, 0, nullptr, 0, nullptr, 1, &dep.barrier);
					}
					for (const auto& dep : sp.post_mem_barriers) {
						vkCmdPipelineBarrier(cbuf, (VkPipelineStageFlags)dep.src, (VkPipelineStageFlags)dep.dst, 0, 1, &dep.barrier, 0, nullptr, 0, nullptr);
					}
				}
			}
			if (is_single_pass) {
				ctx.debug.end_region(cbuf);
			}
			if (rpass.handle != VK_NULL_HANDLE) {
				vkCmdEndRenderPass(cbuf);
			}
			for (auto dep : rpass.post_barriers) {
				dep.barrier.image = impl->bound_attachments[dep.image].image;
				vkCmdPipelineBarrier(cbuf, (VkPipelineStageFlags)dep.src, (VkPipelineStageFlags)dep.dst, 0, 0, nullptr, 0, nullptr, 1, &dep.barrier);
			}
			for (const auto& dep : rpass.post_mem_barriers) {
				vkCmdPipelineBarrier(cbuf, (VkPipelineStageFlags)dep.src, (VkPipelineStageFlags)dep.dst, 0, 1, &dep.barrier, 0, nullptr, 0, nullptr);
			}
		}
		auto result = vkEndCommandBuffer(cbuf);
		if (result != VK_SUCCESS) {
			return { expected_error, VkException{result} };
		}
		return { expected_value, std::move(hl_cbuf) };
	}

	Result<BufferInfo, RenderGraphException> ExecutableRenderGraph::get_resource_buffer(Name n) {
		auto resolved = impl->resolve_name(n);
		auto it = impl->bound_buffers.find(resolved);
		if (it == impl->bound_buffers.end()) {
			return { expected_error, RenderGraphException{"Buffer not found"} };
		}
		return { expected_value, it->second };
	}

	Result<AttachmentRPInfo, RenderGraphException> ExecutableRenderGraph::get_resource_image(Name n) {
		auto resolved = impl->resolve_name(n);
		auto it = impl->bound_attachments.find(resolved);
		if (it == impl->bound_attachments.end()) {
			return { expected_error, RenderGraphException{"Buffer not found"} };
		}
		return { expected_value, it->second };
	}

	Result<bool, RenderGraphException> ExecutableRenderGraph::is_resource_image_in_general_layout(Name n, PassInfo* pass_info) {
		auto resolved = impl->resolve_name(n);
		auto it = impl->use_chains.find(resolved);
		if (it == impl->use_chains.end()) {
			return { expected_error, RenderGraphException{"Resource not found"} };
		}
		auto& chain = it->second;
		for (auto& elem : chain) {
			if (elem.pass == pass_info) {
				return { expected_value, elem.use.layout == vuk::ImageLayout::eGeneral };
			}
		}
		assert(false && "Image resourced was not declared to be used in this pass, but was referred to.");
		return { expected_error, RenderGraphException{ "Image resourced was not declared to be used in this pass, but was referred to." } };
	}

	Name ExecutableRenderGraph::resolve_name(Name name) const noexcept {
		return impl->resolve_name(name);
	}
} // namespace vuk
