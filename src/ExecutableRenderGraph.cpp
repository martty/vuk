#include "vuk/RenderGraph.hpp"
#include "vuk/Hash.hpp" // for create
#include "Cache.hpp"
#include "vuk/Context.hpp"
#include "vuk/CommandBuffer.hpp"
#include "Allocator.hpp"
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

	void ExecutableRenderGraph::create_attachment(PerThreadContext& ptc, Name name, AttachmentRPInfo& attachment_info, vuk::Extent2D fb_extent, vuk::SampleCountFlagBits samples) {
		auto& chain = impl->use_chains.at(name);
		if (attachment_info.type == AttachmentRPInfo::Type::eInternal) {
			vuk::ImageUsageFlags usage = compute_usage(chain);

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

			auto rg = ptc.acquire_rendertarget(rgci);
			attachment_info.iv = rg.image_view;
			attachment_info.image = rg.image;
		}
	}

	void begin_renderpass(vuk::RenderPassInfo& rpass, VkCommandBuffer& cbuf, bool use_secondary_command_buffers) {
		if (rpass.handle == VK_NULL_HANDLE) {
			return;
		}

		VkRenderPassBeginInfo rbi{.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO};
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
		for (auto& ca : rpi.color_attachments) {
			auto& att = rpass.attachments[ca.attachment];
			if (!att.samples.infer)
				rpi.samples = att.samples.count;
		}
		// TODO: depth could be msaa too
		if (rpi.color_attachments.size() == 0) { // depth only pass, samples == 1
			rpi.samples = vuk::SampleCountFlagBits::e1;
		}
		cobuf.ongoing_renderpass = rpi;
	}

	VkCommandBuffer ExecutableRenderGraph::execute(vuk::PerThreadContext& ptc, std::vector<std::pair<SwapChainRef, size_t>> swp_with_index) {
		// create framebuffers, create & bind attachments
		for (auto& rp : impl->rpis) {
			if (rp.attachments.size() == 0)
				continue;

			vuk::Extent2D fb_extent;
			bool extent_known = false;

			// bind swapchain attachments, deduce framebuffer size & sample count
			for (auto& attrpinfo : rp.attachments) {
				auto& bound = impl->bound_attachments[attrpinfo.name];

				if (bound.type == AttachmentRPInfo::Type::eSwapchain) {
					auto it = std::find_if(swp_with_index.begin(), swp_with_index.end(), [&](auto& t) { return t.first == bound.swapchain; });
					bound.iv = it->first->image_views[it->second];
					bound.image = it->first->images[it->second];
					fb_extent = it->first->extent;
					bound.extents = Dimension2D::absolute(it->first->extent);
					extent_known = true;
				} else {
					if (bound.extents.sizing == Sizing::eAbsolute) {
						fb_extent = bound.extents.extent;
						extent_known = true;
					}
				}
			}

			if (extent_known) {
				rp.fbci.width = fb_extent.width;
				rp.fbci.height = fb_extent.height;
			}

			for (auto& attrpinfo : rp.attachments) {
				auto& bound = impl->bound_attachments[attrpinfo.name];
				if (extent_known) {
					bound.extents = Dimension2D::absolute(fb_extent);
				}
			}
		}

		for (auto& [name, bound] : impl->bound_attachments) {
			if (bound.type == AttachmentRPInfo::Type::eSwapchain) {
				auto it = std::find_if(swp_with_index.begin(), swp_with_index.end(), [boundb = &bound](auto& t) { return t.first == boundb->swapchain; });
				bound.iv = it->first->image_views[it->second];
				bound.image = it->first->images[it->second];
			}
		}
	
		for (auto& rp : impl->rpis) {
			if (rp.attachments.size() == 0)
				continue;

			auto& ivs = rp.fbci.attachments;
			std::vector<VkImageView> vkivs;

			Extent2D fb_extent = Extent2D{rp.fbci.width, rp.fbci.height};
			
			// do a second pass so that we can infer from the attachments we previously inferred from
			// TODO: we should allow arbitrary number of passes
			if (fb_extent.width == 0 || fb_extent.height == 0) {
				for (auto& attrpinfo : rp.attachments) {
					auto& bound = impl->bound_attachments[attrpinfo.name];
					if (bound.extents.extent.width > 0 && bound.extents.extent.height > 0) {
						fb_extent = bound.extents.extent;
					}
				}
			}
			// TODO: check here if all attachments have been sized

			// create internal attachments; bind attachments to fb
			for (auto& attrpinfo : rp.attachments) {
				auto& bound = impl->bound_attachments[attrpinfo.name];
				if (bound.type == AttachmentRPInfo::Type::eInternal) {
					create_attachment(ptc, attrpinfo.name, bound, fb_extent, (vuk::SampleCountFlagBits)attrpinfo.description.samples);
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
			rp.framebuffer = ptc.acquire_framebuffer(rp.fbci);
		}

		// create non-attachment images
		for (auto& [name, bound] : impl->bound_attachments) {
			if (bound.type == AttachmentRPInfo::Type::eInternal && bound.image == VK_NULL_HANDLE) {
				create_attachment(ptc, name, bound, vuk::Extent2D{0,0}, bound.samples.count);
			}
		}

		// actual execution
		auto cbuf = ptc.acquire_command_buffer(VK_COMMAND_BUFFER_LEVEL_PRIMARY);

		VkCommandBufferBeginInfo cbi{.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO, .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT };
		vkBeginCommandBuffer(cbuf, &cbi);

		CommandBuffer cobuf(*this, ptc, cbuf);
		for (auto& rpass : impl->rpis) {
            bool use_secondary_command_buffers = rpass.subpasses[0].use_secondary_command_buffers;
            begin_renderpass(rpass, cbuf, use_secondary_command_buffers);
			for (size_t i = 0; i < rpass.subpasses.size(); i++) {
				auto& sp = rpass.subpasses[i];
				fill_renderpass_info(rpass, i, cobuf);
				// insert image pre-barriers
				if (rpass.handle == VK_NULL_HANDLE) {
					for (auto dep : sp.pre_barriers) {
						dep.barrier.image = impl->bound_attachments[dep.image].image;
						vkCmdPipelineBarrier(cbuf, (VkPipelineStageFlags)dep.src, (VkPipelineStageFlags)dep.dst, 0, 0, nullptr, 0, nullptr, 1, &dep.barrier);
					}
					for (auto dep : sp.pre_mem_barriers) {
						vkCmdPipelineBarrier(cbuf, (VkPipelineStageFlags)dep.src, (VkPipelineStageFlags)dep.dst, 0, 1, &dep.barrier, 0, nullptr, 0, nullptr);
					}
				}
                for(auto& p: sp.passes) {
					// if pass requested no secondary cbufs, but due to subpass merging that is what we got
					if (p->pass.use_secondary_command_buffers == false && use_secondary_command_buffers == true) {
                        auto secondary = cobuf.begin_secondary();
                        if(p->pass.execute) {
                            secondary.current_pass = p;
                            if(!p->pass.name.empty()) {
                                //ptc.ctx.debug.begin_region(cobuf.command_buffer, sp.pass->pass.name);
                                p->pass.execute(secondary);
                                //ptc.ctx.debug.end_region(cobuf.command_buffer);
                            } else {
                                p->pass.execute(secondary);
                            }
                        }
                        auto result = secondary.get_buffer();
                        cobuf.execute({&result, 1});
                    } else {
                        if(p->pass.execute) {
                            cobuf.current_pass = p;
                            if(!p->pass.name.empty()) {
                                //ptc.ctx.debug.begin_region(cobuf.command_buffer, sp.pass->pass.name);
                                p->pass.execute(cobuf);
                                //ptc.ctx.debug.end_region(cobuf.command_buffer);
                            } else {
                                p->pass.execute(cobuf);
                            }
                        }

                        cobuf.attribute_descriptions.clear();
                        cobuf.binding_descriptions.clear();
                        cobuf.set_bindings = {};
                        cobuf.sets_used = {};
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
					for (auto dep : sp.post_mem_barriers) {
						vkCmdPipelineBarrier(cbuf, (VkPipelineStageFlags)dep.src, (VkPipelineStageFlags)dep.dst, 0, 1, &dep.barrier, 0, nullptr, 0, nullptr);
					}
				}
			}
			if (rpass.handle != VK_NULL_HANDLE) {
				vkCmdEndRenderPass(cbuf);
			}
		}
		vkEndCommandBuffer(cbuf);
		return cbuf;
	}
	
	BufferInfo ExecutableRenderGraph::get_resource_buffer(Name n) {
		return impl->bound_buffers.at(n);
	}
	
	AttachmentRPInfo ExecutableRenderGraph::get_resource_image(Name n) {
		return impl->bound_attachments.at(n);
	}

	bool ExecutableRenderGraph::is_resource_image_in_general_layout(Name n, PassInfo* pass_info) {
		auto& chain = impl->use_chains.at(n);
		for (auto& elem : chain) {
			if (elem.pass == pass_info) {
				return elem.use.layout == vuk::ImageLayout::eGeneral;
			}
		}
		assert(0);
		return false;
	}
} // namespace vuk
