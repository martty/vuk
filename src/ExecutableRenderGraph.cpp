#include "vuk/RenderGraph.hpp"
#include "vuk/Hash.hpp" // for create
#include "Cache.hpp"
#include "vuk/Context.hpp"
#include "vuk/CommandBuffer.hpp"
#include "Allocator.hpp"
#include "RenderGraphImpl.hpp"
#include "ResourceBundle.hpp"
#include "ContextImpl.hpp"
#include <vuk/FrameAllocator.hpp>
#include <RenderPass.hpp>

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

	template<class Allocator>
	void ExecutableRenderGraph::create_attachment(Allocator& allocator, Name name, AttachmentRPInfo& attachment_info, vuk::Extent2D fb_extent, vuk::SampleCountFlagBits samples) {
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

			auto rg = allocator.allocate_rendertarget(rgci);
			attachment_info.iv = rg.image_view;
			attachment_info.image = rg.image;
		}
	}

	void begin_renderpass(vuk::RenderPassInfo& rpass, VkCommandBuffer& cbuf, bool use_secondary_command_buffers) {
		if (rpass.handle == VK_NULL_HANDLE) {
			return;
		}

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
		cobuf.ongoing_renderpass = rpi;
	}

	void ExecutableRenderGraph::size_attachments(std::vector<std::pair<Swapchain*, size_t>> swp_with_index) {
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
	}

	void ExecutableRenderGraph::bind_swapchain_images(std::vector<std::pair<Swapchain*, size_t>> swp_with_index) {
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
	}

	template<class Allocator>
	void ExecutableRenderGraph::bind_attachments(Allocator& allocator) {
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
					create_attachment(allocator, attrpinfo.name, bound, fb_extent, (vuk::SampleCountFlagBits)attrpinfo.description.samples);
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
			rp.framebuffer = allocator.allocate_framebuffer(rp.fbci);
		}

		// create non-attachment images
		for (auto& [name, bound] : impl->bound_attachments) {
			if (bound.type == AttachmentRPInfo::Type::eInternal && bound.image == VK_NULL_HANDLE) {
				create_attachment(allocator, name, bound, vuk::Extent2D{ 0,0 }, bound.samples.count);
			}
		}
	}

	template<class Allocator>
	SubmitInfo ExecutableRenderGraph::run_passes(Allocator& allocator) {
		// actual execution
		// TODO: queue family index
		VkCommandBuffer cbuf = allocator.allocate_command_buffer(VK_COMMAND_BUFFER_LEVEL_PRIMARY, allocator.ctx.graphics_queue_family_index);

		VkCommandBufferBeginInfo cbi{ .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO, .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT };
		vkBeginCommandBuffer(cbuf, &cbi);

		CommandBufferImpl<Allocator> cobuf(*this, allocator, cbuf);
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
				for (auto& p : sp.passes) {
					// if pass requested no secondary cbufs, but due to subpass merging that is what we got
					bool make_secondary_cbuf = p->pass.use_secondary_command_buffers == false && use_secondary_command_buffers == true;
					auto&& cbuf_to_use = make_secondary_cbuf ? cobuf.begin_secondary() : cobuf;
					if (p->pass.execute) {
						cbuf_to_use.current_pass = p;
						if (!p->pass.name.is_invalid()) {
							//ptc.ctx.debug.begin_region(cobuf.command_buffer, sp.pass->pass.name);
							p->pass.execute(cbuf_to_use);
							//ptc.ctx.debug.end_region(cobuf.command_buffer);
						} else {
							p->pass.execute(cbuf_to_use);
						}
					}

					cobuf.pcrs.clear();
					cobuf.attribute_descriptions.clear();
					cobuf.binding_descriptions.clear();
					cobuf.set_bindings = {};
					cobuf.sets_used = {};
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
				for (auto dep : rpass.post_barriers) {
					dep.barrier.image = impl->bound_attachments[dep.image].image;
					vkCmdPipelineBarrier(cbuf, (VkPipelineStageFlags)dep.src, (VkPipelineStageFlags)dep.dst, 0, 0, nullptr, 0, nullptr, 1, &dep.barrier);
				}
				for (auto dep : rpass.post_mem_barriers) {
					vkCmdPipelineBarrier(cbuf, (VkPipelineStageFlags)dep.src, (VkPipelineStageFlags)dep.dst, 0, 1, &dep.barrier, 0, nullptr, 0, nullptr);
				}
			}
		}
		vkEndCommandBuffer(cbuf);

		SubmitInfo cbws{ .command_buffers = {cbuf} };
		for (auto& passinfo : impl->passes) {
			for (auto token : passinfo.pass.waits) {
				// TODO: proper wait/signal values
				cbws.wait_values.push_back(1);
				auto* resources = allocator.get_token_data(token).resources;
				auto sema = resources->sema;
				cbws.wait_semaphores.push_back(sema);
				cbws.wait_stages.push_back(VK_PIPELINE_STAGE_ALL_COMMANDS_BIT);
			}
		}
		return cbws;
	}

	SubmitInfo ExecutableRenderGraph::execute(ThreadLocalFrameAllocator& fa, std::vector<std::pair<Swapchain*, size_t>> swp_with_index) {
		// acquire the renderpasses
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

			rp.handle = fa.allocate_renderpass(rp.rpci);
		}

		bind_swapchain_images(swp_with_index);
		size_attachments(swp_with_index);
		bind_attachments(fa);

		return run_passes(fa);
	}

	SubmitInfo ExecutableRenderGraph::execute(LinearResourceAllocator<Allocator>& bundle, std::vector<std::pair<Swapchain*, size_t>> swp_with_index) {
		bind_swapchain_images(swp_with_index);
		size_attachments(swp_with_index);
		bind_attachments(bundle);

		return run_passes(bundle);
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
		assert(false && "Image resourced was not declared to be used in this pass, but was referred to.");
		return false;
	}
} // namespace vuk
