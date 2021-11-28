#include "vuk/RenderGraph.hpp"
#include "RenderGraphUtil.hpp"
#include "RenderGraphImpl.hpp"
#include "vuk/Context.hpp"
#include "vuk/Exception.hpp"
#include <unordered_set>

namespace vuk {
	RenderGraph::RenderGraph() : impl(new RGImpl) {
	}

	RenderGraph::RenderGraph(RenderGraph&& o) noexcept : impl(std::exchange(o.impl, nullptr)) {}
	RenderGraph& RenderGraph::operator=(RenderGraph&& o) noexcept {
		impl = std::exchange(o.impl, nullptr);
		return *this;
	}

	RenderGraph::~RenderGraph() {
		delete impl;
	}

	void RenderGraph::add_pass(Pass p) {
		impl->passes.emplace_back(*impl->arena_, std::move(p));
	}

	void RenderGraph::append(RenderGraph other) {
		// TODO:
		// this code is written weird because of wonky allocators
		for (auto& p : other.impl->passes) {
			impl->passes.emplace_back(*impl->arena_, Pass{}) = p;
		}

		impl->bound_attachments.insert(std::make_move_iterator(other.impl->bound_attachments.begin()), std::make_move_iterator(other.impl->bound_attachments.end()));
		impl->bound_buffers.insert(std::make_move_iterator(other.impl->bound_buffers.begin()), std::make_move_iterator(other.impl->bound_buffers.end()));
	}

	void RenderGraph::add_alias(Name new_name, Name old_name) {
		if (new_name != old_name) {
			impl->aliases[new_name] = old_name;
		}
	}

	// determine rendergraph inputs and outputs, and resources that are neither
	void RenderGraph::build_io() {
		for (auto& pif : impl->passes) {
			for (auto& res : pif.pass.resources) {
				if (is_read_access(res.ia)) {
					pif.inputs.emplace_back(res);
					auto resolved_name = resolve_name(res.name, impl->aliases);
					auto hashed_resolved_name = ::hash::fnv1a::hash(resolved_name.to_sv().data(), resolved_name.to_sv().size(), hash::fnv1a::default_offset_basis);
					pif.resolved_input_name_hashes.emplace_back(hashed_resolved_name);
					pif.bloom_resolved_inputs |= hashed_resolved_name;
				}
				if (is_write_access(res.ia)) {
					auto hashed_name = ::hash::fnv1a::hash(res.name.to_sv().data(), res.name.to_sv().size(), hash::fnv1a::default_offset_basis);
					pif.bloom_outputs |= hashed_name;
					pif.output_name_hashes.emplace_back(hashed_name);
					pif.outputs.emplace_back(res);
				}
			}
		}
	}

	void RenderGraph::compile(const vuk::RenderGraph::CompileOptions& compile_options) {
		// find which reads are graph inputs (not produced by any pass) & outputs (not consumed by any pass)
		build_io();

		// sort passes if requested
		if (impl->passes.size() > 1 && compile_options.reorder_passes) {
			topological_sort(impl->passes.begin(), impl->passes.end(), [](const auto& p1, const auto& p2) {
				bool could_execute_after = false;
				bool could_execute_before = false;

				if ((p1.bloom_outputs & p2.bloom_resolved_inputs) != 0) {
					for (auto& o : p1.output_name_hashes) {
						for (auto& i : p2.resolved_input_name_hashes) {
							if (o == i) {
								could_execute_after = true;
								break;
							}
						}
					}
				}

				if ((p2.bloom_outputs & p1.bloom_resolved_inputs) != 0) {
					for (auto& o : p2.output_name_hashes) {
						for (auto& i : p1.resolved_input_name_hashes) {
							if (o == i) {
								could_execute_before = true;
								break;
							}
						}
					}
				}
				if (!could_execute_after && !could_execute_before && p1.outputs == p2.outputs) {
					return p1.pass.auxiliary_order < p2.pass.auxiliary_order;
				}

				if (could_execute_after && could_execute_before) {
					return p1.pass.auxiliary_order < p2.pass.auxiliary_order;
				} else if (could_execute_after) {
					return true;
				} else
					return false;
				});
		}

		if (compile_options.check_pass_ordering) {
			for (unsigned i = 0; i < impl->passes.size() - 1; i++) {
				for (unsigned j = i; j < impl->passes.size(); j++) {
					auto& p1 = impl->passes[i];
					auto& p2 = impl->passes[j];

					bool could_execute_after = false;
					bool could_execute_before = false;

					if ((p1.bloom_outputs & p2.bloom_resolved_inputs) != 0) {
						for (auto& o : p1.output_name_hashes) {
							for (auto& i : p2.resolved_input_name_hashes) {
								if (o == i) {
									could_execute_after = true;
									break;
								}
							}
						}
					}

					if ((p2.bloom_outputs & p1.bloom_resolved_inputs) != 0) {
						for (auto& o : p2.output_name_hashes) {
							for (auto& i : p1.resolved_input_name_hashes) {
								if (o == i) {
									could_execute_before = true;
									break;
								}
							}
						}
					}
					// unambiguously wrong ordering found
					if (could_execute_before && !could_execute_after) {
						throw RenderGraphException{ "Pass ordering violates resource constraints." };
					}
				}
			}
		}

		impl->use_chains.clear();
		// assemble use chains
		for (auto& passinfo : impl->passes) {
			for (auto& res : passinfo.pass.resources) {
				auto it = impl->use_chains.find(resolve_name(res.name, impl->aliases));
				if (it == impl->use_chains.end()) {
					it = impl->use_chains.emplace(resolve_name(res.name, impl->aliases), std::vector<UseRef, short_alloc<UseRef, 64>>{short_alloc<UseRef, 64>{*impl->arena_}}).first;
				}
				it->second.emplace_back(UseRef{ to_use(res.ia), &passinfo });
			}
		}

		// we need to collect passes into framebuffers, which will determine the renderpasses
		using attachment_set = std::unordered_set<Resource, std::hash<Resource>, std::equal_to<Resource>, short_alloc<Resource, 16>>;
		using passinfo_vec = std::vector<PassInfo*, short_alloc<PassInfo*, 16>>;
		std::vector<std::pair<attachment_set, passinfo_vec>, short_alloc<std::pair<attachment_set, passinfo_vec>, 8>> attachment_sets{ *impl->arena_ };
		for (auto& passinfo : impl->passes) {
			attachment_set atts{ *impl->arena_ };

			for (auto& res : passinfo.pass.resources) {
				if (is_framebuffer_attachment(res))
					atts.insert(res);
			}

			if (auto p = attachment_sets.size() > 0 && attachment_sets.back().first == atts ? &attachment_sets.back() : nullptr) {
				p->second.push_back(&passinfo);
			} else {
				passinfo_vec pv{ *impl->arena_ };
				pv.push_back(&passinfo);
				attachment_sets.emplace_back(atts, pv);
			}
		}

		impl->rpis.clear();
		// renderpasses are uniquely identified by their index from now on
		// tell passes in which renderpass/subpass they will execute
		impl->rpis.reserve(attachment_sets.size());
		for (auto& [attachments, passes] : attachment_sets) {
			RenderPassInfo rpi{ *impl->arena_ };
			auto rpi_index = impl->rpis.size();

			int32_t subpass = -1;
			for (auto& p : passes) {
				p->render_pass_index = rpi_index;
				if (rpi.subpasses.size() > 0) {
					auto& last_pass = rpi.subpasses.back().passes[0];
					// if the pass has the same inputs and outputs, we execute them on the same subpass
					if (last_pass->inputs == p->inputs && last_pass->outputs == p->outputs) {
						p->subpass = last_pass->subpass;
						rpi.subpasses.back().passes.push_back(p);
						// potentially upgrade to secondary cbufs
						rpi.subpasses.back().use_secondary_command_buffers |= p->pass.use_secondary_command_buffers;
						continue;
					}
				}
				SubpassInfo si{ *impl->arena_ };
				si.passes = { p };
				si.use_secondary_command_buffers = p->pass.use_secondary_command_buffers;
				p->subpass = ++subpass;
				rpi.subpasses.push_back(si);
			}
			for (auto& att : attachments) {
				AttachmentRPInfo info;
				info.name = resolve_name(att.name, impl->aliases);
				rpi.attachments.push_back(info);
			}

			if (attachments.size() == 0) {
				rpi.framebufferless = true;
			}

			impl->rpis.push_back(rpi);
		}
	}

	void RenderGraph::resolve_resource_into(Name resolved_name, Name ms_name) {
		add_pass({
			.resources = {
				vuk::Resource{ms_name, vuk::Resource::Type::eImage, vuk::eColorResolveRead},
				vuk::Resource{resolved_name, vuk::Resource::Type::eImage, vuk::eColorResolveWrite}
			},
			.resolves = {{ms_name, resolved_name}}
			});
	}

	void RenderGraph::attach_swapchain(Name name, SwapchainRef swp, Clear c) {
		AttachmentRPInfo attachment_info;
		attachment_info.extents = vuk::Dimension2D::absolute(swp->extent);
		attachment_info.iv = {};
		// directly presented
		attachment_info.description.format = (VkFormat)swp->format;
		attachment_info.samples = Samples::e1;

		attachment_info.type = AttachmentRPInfo::Type::eSwapchain;
		attachment_info.swapchain = swp;
		attachment_info.should_clear = true;
		attachment_info.clear_value = c;

		Resource::Use& initial = attachment_info.initial;
		Resource::Use & final = attachment_info.final;
		// for WSI, we want to wait for colourattachmentoutput
		// we don't care about any writes, we will clear
		initial.access = vuk::AccessFlags{};
		initial.stages = vuk::PipelineStageFlagBits::eColorAttachmentOutput;
		// clear
		initial.layout = vuk::ImageLayout::ePreinitialized;
		/* Normally, we would need an external dependency at the end as well since we are changing layout in finalLayout,
   but since we are signalling a semaphore, we can rely on Vulkan's default behavior,
   which injects an external dependency here with
   dstStageMask = VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
   dstAccessMask = 0. */
		final.access = vuk::AccessFlagBits{};
		final.layout = vuk::ImageLayout::ePresentSrcKHR;
		final.stages = vuk::PipelineStageFlagBits::eBottomOfPipe;

		impl->bound_attachments.emplace(name, attachment_info);
	}

	void RenderGraph::attach_managed(Name name, vuk::Format format, vuk::Dimension2D extent, vuk::Samples samp, Clear c) {
		AttachmentRPInfo attachment_info;
		attachment_info.extents = extent;
		attachment_info.iv = {};

		attachment_info.type = AttachmentRPInfo::Type::eInternal;
		attachment_info.description.format = (VkFormat)format;
		attachment_info.samples = samp;

		attachment_info.should_clear = true;
		attachment_info.clear_value = c;
		Resource::Use& initial = attachment_info.initial;
		Resource::Use & final = attachment_info.final;
		initial.access = vuk::AccessFlags{};
		initial.stages = vuk::PipelineStageFlagBits::eTopOfPipe;
		// for internal attachments we don't want to preserve previous data
		initial.layout = vuk::ImageLayout::ePreinitialized;

		// with an undefined final layout, there will be no final sync
		final.layout = vuk::ImageLayout::eUndefined;
		final.access = vuk::AccessFlagBits{};
		final.stages = vuk::PipelineStageFlagBits::eBottomOfPipe;

		impl->bound_attachments.emplace(name, attachment_info);
	}

	void RenderGraph::attach_buffer(Name name, vuk::Buffer buf, Access initial, Access final) {
		BufferInfo buf_info{ .name = name, .initial = to_use(initial), .final = to_use(final), .buffer = buf };
		impl->bound_buffers.emplace(name, buf_info);
	}

	void RenderGraph::attach_image(Name name, ImageAttachment att, Access initial_acc, Access final_acc) {
		AttachmentRPInfo attachment_info;
		attachment_info.extents = vuk::Dimension2D::absolute(att.extent);
		attachment_info.image = att.image;
		attachment_info.iv = att.image_view;

		attachment_info.type = AttachmentRPInfo::Type::eExternal;
		attachment_info.description.format = (VkFormat)att.format;
		attachment_info.samples = att.sample_count;

		attachment_info.should_clear = initial_acc == Access::eClear; // if initial access was clear, we will clear
		attachment_info.clear_value = att.clear_value;
		Resource::Use& initial = attachment_info.initial;
		Resource::Use & final = attachment_info.final;
		initial = to_use(initial_acc);
		final = to_use(final_acc);
		impl->bound_attachments.emplace(name, attachment_info);
	}

	void sync_bound_attachment_to_renderpass(vuk::AttachmentRPInfo& rp_att, vuk::AttachmentRPInfo& attachment_info) {
		rp_att.description.format = attachment_info.description.format;
		rp_att.samples = attachment_info.samples;
		rp_att.description.samples = (VkSampleCountFlagBits)attachment_info.samples.count;
		rp_att.iv = attachment_info.iv;
		rp_att.extents = attachment_info.extents;
		rp_att.clear_value = attachment_info.clear_value;
		rp_att.should_clear = attachment_info.should_clear;
		rp_att.type = attachment_info.type;
	}

	void RenderGraph::validate() {
		// check if all resourced are attached
		for (const auto& [n, v] : impl->use_chains) {
			if (!impl->bound_attachments.contains(n) && !impl->bound_buffers.contains(n)) {
				throw RenderGraphException{ std::string("Missing resource: \"") + std::string(n.to_sv()) + "\". Did you forget to attach it?" };
			}
		}
	}

	ExecutableRenderGraph RenderGraph::link(Context& ctx, const vuk::RenderGraph::CompileOptions& compile_options)&& {
		compile(compile_options);

		// at this point the graph is built, we know of all the resources and everything should have been attached
		// perform checking if this indeed the case
		validate();

		for (auto& [raw_name, attachment_info] : impl->bound_attachments) {
			auto name = resolve_name(raw_name, impl->aliases);
			auto& chain = impl->use_chains.at(name);
			chain.insert(chain.begin(), UseRef{ std::move(attachment_info.initial), nullptr });
			chain.emplace_back(UseRef{ attachment_info.final, nullptr });

			vuk::ImageAspectFlags aspect = format_to_aspect((vuk::Format)attachment_info.description.format);

			for (size_t i = 0; i < chain.size() - 1; i++) {
				auto& left = chain[i];
				auto& right = chain[i + 1];

				bool crosses_rpass = (left.pass == nullptr || right.pass == nullptr || left.pass->render_pass_index != right.pass->render_pass_index);
				if (crosses_rpass) {
					if (left.pass) { // RenderPass ->
						auto& left_rp = impl->rpis[left.pass->render_pass_index];
						// if this is an attachment, we specify layout
						if (is_framebuffer_attachment(left.use)) {
							assert(!left_rp.framebufferless);
							auto& rp_att = *contains_if(left_rp.attachments, [name](auto& att) {return att.name == name; });

							sync_bound_attachment_to_renderpass(rp_att, attachment_info);
							// we keep last use as finalLayout
							rp_att.description.finalLayout = (VkImageLayout)left.use.layout;
	
							// compute attachment store
							if (right.use.layout == vuk::ImageLayout::eUndefined) {
								rp_att.description.storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
							} else {
								rp_att.description.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
							}
						}
						// emit barrier for final resource state
                        if(!right.pass && right.use.layout != vuk::ImageLayout::eUndefined && (left.use.layout != right.use.layout || (is_write_access(left.use) || is_write_access(right.use)))) { // different layouts, need to have dependency
							VkImageMemoryBarrier barrier{ .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER };
							ImageBarrier ib{};
							barrier.dstAccessMask = (VkAccessFlags)right.use.access;
                            barrier.srcAccessMask = is_read_access(left.use) ? 0 : (VkAccessFlags)left.use.access;
							barrier.oldLayout = (VkImageLayout)left.use.layout;
							barrier.newLayout = (VkImageLayout)right.use.layout;
							barrier.subresourceRange.aspectMask = (VkImageAspectFlags)aspect;
							barrier.subresourceRange.baseArrayLayer = 0;
							barrier.subresourceRange.baseMipLevel = 0;
							barrier.subresourceRange.layerCount = VK_REMAINING_ARRAY_LAYERS;
							barrier.subresourceRange.levelCount = VK_REMAINING_MIP_LEVELS;
							barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
							barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
							ib.src = left.use.stages;
							ib.dst = right.use.stages;
							ib.barrier = barrier;
							ib.image = name;
                            // attach this barrier to the end of subpass or end of renderpass
							if (left_rp.framebufferless) {
								left_rp.subpasses[left.pass->subpass].post_barriers.push_back(ib);
							}
							else {
								left_rp.post_barriers.push_back(ib);
							}
						}
					}

					if (right.pass) { // -> RenderPass
						auto& right_rp = impl->rpis[right.pass->render_pass_index];
						// if this is an attachment, we specify layout
						if (is_framebuffer_attachment(right.use)) {
							assert(!right_rp.framebufferless);
							auto& rp_att = *contains_if(right_rp.attachments, [name](auto& att) {return att.name == name; });

							sync_bound_attachment_to_renderpass(rp_att, attachment_info);
							// 
							
							rp_att.description.initialLayout = (VkImageLayout)right.use.layout;
                            assert(rp_att.description.initialLayout != (VkImageLayout)vuk::ImageLayout::eUndefined);
							
							// compute attachment load
							if (left.use.layout == vuk::ImageLayout::eUndefined) {
								rp_att.description.loadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
							} else if (left.use.layout == vuk::ImageLayout::ePreinitialized) {
								// preinit means clear
								rp_att.description.initialLayout = (VkImageLayout)vuk::ImageLayout::eUndefined;
								rp_att.description.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
							} else {
								rp_att.description.loadOp = VK_ATTACHMENT_LOAD_OP_LOAD;
							}
						}
						// we are keeping this weird logic until this is configurable
						// emit a barrier for now instead of an external subpass dep
                        if(right.use.layout != left.use.layout ||
                            (is_write_access(left.use) || is_write_access(right.use))) { // different layouts, need to have dependency
							VkImageMemoryBarrier barrier{ .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER };
							barrier.dstAccessMask = (VkAccessFlags)right.use.access;
							barrier.srcAccessMask = is_read_access(left.use) ? 0 : (VkAccessFlags)left.use.access;
							barrier.oldLayout = left.use.layout == vuk::ImageLayout::ePreinitialized ? (VkImageLayout)vuk::ImageLayout::eUndefined : (VkImageLayout)left.use.layout;
							barrier.newLayout = (VkImageLayout)right.use.layout;
							barrier.subresourceRange.aspectMask = (VkImageAspectFlags)aspect;
							barrier.subresourceRange.baseArrayLayer = 0;
							barrier.subresourceRange.baseMipLevel = 0;
							barrier.subresourceRange.layerCount = VK_REMAINING_ARRAY_LAYERS;
							barrier.subresourceRange.levelCount = VK_REMAINING_MIP_LEVELS;
							barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
							barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
							ImageBarrier ib{ .image = name, .barrier = barrier, .src = left.use.stages, .dst = right.use.stages };
                            if(right_rp.framebufferless) {
                                right_rp.subpasses[right.pass->subpass].pre_barriers.push_back(ib);
                            } else {
                                right_rp.pre_barriers.push_back(ib);
                            }
						}
					}
				} else { // subpass-subpass link -> subpass - subpass dependency
					// WAW, WAR, RAW accesses need sync

					// if we merged the passes into a subpass, no sync is needed
					if (left.pass->subpass == right.pass->subpass)
						continue;
					if (is_framebuffer_attachment(left.use) && (is_write_access(left.use) || is_write_access(right.use))) {
						assert(left.pass->render_pass_index == right.pass->render_pass_index);
						auto& rp = impl->rpis[right.pass->render_pass_index];
						VkSubpassDependency sd{};
						sd.dstAccessMask = (VkAccessFlags)right.use.access;
						sd.dstStageMask = (VkPipelineStageFlags)right.use.stages;
						sd.dstSubpass = right.pass->subpass;
                        sd.srcAccessMask = is_read_access(left.use) ? 0 : (VkAccessFlags)left.use.access;
						sd.srcStageMask = (VkPipelineStageFlags)left.use.stages;
						sd.srcSubpass = left.pass->subpass;
						rp.rpci.subpass_dependencies.push_back(sd);
					}
					auto& left_rp = impl->rpis[left.pass->render_pass_index];
                    if(left_rp.framebufferless && (is_write_access(left.use) || is_write_access(right.use))) {
						// right layout == Undefined means the chain terminates, no transition/barrier
						if (right.use.layout == vuk::ImageLayout::eUndefined)
							continue;
						VkImageMemoryBarrier barrier{ .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER };
						barrier.dstAccessMask = (VkAccessFlags)right.use.access;
                        barrier.srcAccessMask = is_read_access(left.use) ? 0 : (VkAccessFlags)left.use.access;
						barrier.newLayout = (VkImageLayout)right.use.layout;
						barrier.oldLayout = (VkImageLayout)left.use.layout;
						barrier.subresourceRange.aspectMask = (VkImageAspectFlags)aspect;
						barrier.subresourceRange.baseArrayLayer = 0;
						barrier.subresourceRange.baseMipLevel = 0;
						barrier.subresourceRange.layerCount = VK_REMAINING_ARRAY_LAYERS;
						barrier.subresourceRange.levelCount = VK_REMAINING_MIP_LEVELS;
						ImageBarrier ib{ .image = name, .barrier = barrier, .src = left.use.stages, .dst = right.use.stages };
						left_rp.subpasses[left.pass->subpass].post_barriers.push_back(ib);
					}
				}
			}
		}

		for (auto& [raw_name, buffer_info] : impl->bound_buffers) {
			auto name = resolve_name(raw_name, impl->aliases);
			auto& chain = impl->use_chains.at(name);
			chain.insert(chain.begin(), UseRef{ std::move(buffer_info.initial), nullptr });
			chain.emplace_back(UseRef{ buffer_info.final, nullptr });

			for (size_t i = 0; i < chain.size() - 1; i++) {
				auto& left = chain[i];
				auto& right = chain[i + 1];

				bool crosses_rpass = (left.pass == nullptr || right.pass == nullptr || left.pass->render_pass_index != right.pass->render_pass_index);
				if (crosses_rpass) {
                    if(left.pass && right.use.layout != vuk::ImageLayout::eUndefined && (is_write_access(left.use) || is_write_access(right.use))) { // RenderPass ->
						auto& left_rp = impl->rpis[left.pass->render_pass_index];

						VkMemoryBarrier barrier{ .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER };
						barrier.dstAccessMask = (VkAccessFlags)right.use.access;
                        barrier.srcAccessMask = is_read_access(left.use) ? 0 : (VkAccessFlags)left.use.access;
						MemoryBarrier mb{ .barrier = barrier, .src = left.use.stages, .dst = right.use.stages };
						left_rp.subpasses[left.pass->subpass].post_mem_barriers.push_back(mb);
					}

					if(right.pass && left.use.layout != vuk::ImageLayout::eUndefined && (is_write_access(left.use) || is_write_access(right.use))) { // -> RenderPass
						auto& right_rp = impl->rpis[right.pass->render_pass_index];

						VkMemoryBarrier barrier{ .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER };
						barrier.dstAccessMask = (VkAccessFlags)right.use.access;
                        barrier.srcAccessMask = is_read_access(left.use) ? 0 : (VkAccessFlags)left.use.access;
						MemoryBarrier mb{ .barrier = barrier, .src = left.use.stages, .dst = right.use.stages };
						if (mb.src == vuk::PipelineStageFlags{}) {
							mb.src = vuk::PipelineStageFlagBits::eTopOfPipe;
							mb.barrier.srcAccessMask = {};
						}
						right_rp.subpasses[right.pass->subpass].pre_mem_barriers.push_back(mb);
					}
				} else { // subpass-subpass link -> subpass - subpass dependency
					if (left.pass->subpass == right.pass->subpass)
						continue;
					auto& left_rp = impl->rpis[left.pass->render_pass_index];
                    if(left_rp.framebufferless && (is_write_access(left.use) || is_write_access(right.use))) {
						VkMemoryBarrier barrier{ .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER };
						barrier.dstAccessMask = (VkAccessFlags)right.use.access;
                        barrier.srcAccessMask = is_read_access(left.use) ? 0 : (VkAccessFlags)left.use.access;
						MemoryBarrier mb{ .barrier = barrier, .src = left.use.stages, .dst = right.use.stages };
						left_rp.subpasses[left.pass->subpass].post_mem_barriers.push_back(mb);
					}
				}
			}
		}


		for (auto& rp : impl->rpis) {
			rp.rpci.color_ref_offsets.resize(rp.subpasses.size());
			rp.rpci.ds_refs.resize(rp.subpasses.size());
		}

		// we now have enough data to build VkRenderPasses and VkFramebuffers
		
		// compile attachments
		// we have to assign the proper attachments to proper slots
		// the order is given by the resource binding order

		size_t previous_rp = -1;
		uint32_t previous_sp = -1;
		for (auto& pass : impl->passes) {
			auto& rp = impl->rpis[pass.render_pass_index];
			auto subpass_index = pass.subpass;
			auto& color_attrefs = rp.rpci.color_refs;
			auto& resolve_attrefs = rp.rpci.resolve_refs;
			auto& color_ref_offsets = rp.rpci.color_ref_offsets;
			auto& ds_attrefs = rp.rpci.ds_refs;

			// do not process merged passes
			if (previous_rp != -1 && previous_rp == pass.render_pass_index && previous_sp == pass.subpass) {
				continue;
			} else {
				previous_rp = pass.render_pass_index;
				previous_sp = pass.subpass;
			}

			for (auto& res : pass.pass.resources) {
				if (!is_framebuffer_attachment(res))
					continue;
				if (res.ia == vuk::Access::eColorResolveWrite) // resolve attachment are added when processing the color attachment
					continue;
				VkAttachmentReference attref{};

				auto name = resolve_name(res.name, impl->aliases);
				auto& chain = impl->use_chains.find(name)->second;
				auto cit = std::find_if(chain.begin(), chain.end(), [&](auto& useref) { return useref.pass == &pass; });
				assert(cit != chain.end());
				attref.layout = (VkImageLayout)cit->use.layout;
				attref.attachment = (uint32_t)std::distance(rp.attachments.begin(), std::find_if(rp.attachments.begin(), rp.attachments.end(), [&](auto& att) { return name == att.name; }));

				if (attref.layout != (VkImageLayout)vuk::ImageLayout::eColorAttachmentOptimal) {
					if (attref.layout == (VkImageLayout)vuk::ImageLayout::eDepthStencilAttachmentOptimal) {
						ds_attrefs[subpass_index] = attref;
					}
				} else {
					VkAttachmentReference rref{};
					rref.attachment = VK_ATTACHMENT_UNUSED;
					if (auto it = pass.pass.resolves.find(res.name); it != pass.pass.resolves.end()) {
						// this a resolve src attachment
						// get the dst attachment
						auto& dst_name = it->second;
						rref.layout = (VkImageLayout)vuk::ImageLayout::eColorAttachmentOptimal; // the only possible layout for resolves
						rref.attachment = (uint32_t)std::distance(rp.attachments.begin(), std::find_if(rp.attachments.begin(), rp.attachments.end(), [&](auto& att) { return dst_name == att.name; }));
						rp.attachments[rref.attachment].samples = vuk::Samples::e1; // resolve dst must be sample count = 1
						rp.attachments[rref.attachment].is_resolve_dst = true;
					}

					// we insert the new attachment at the end of the list for current subpass index
					if (subpass_index < rp.subpasses.size() - 1) {
						auto next_start = color_ref_offsets[subpass_index + 1];
						color_attrefs.insert(color_attrefs.begin() + next_start, attref);
						resolve_attrefs.insert(resolve_attrefs.begin() + next_start, rref);
					} else {
						color_attrefs.push_back(attref);
						resolve_attrefs.push_back(rref);
					}
					for (size_t i = subpass_index + 1; i < rp.subpasses.size(); i++) {
						color_ref_offsets[i]++;
					}
				}
			}
		}

		// compile subpass description structures

		for (auto& rp : impl->rpis) {
			if (rp.attachments.size() == 0) {
				continue;
			}

			auto& subp = rp.rpci.subpass_descriptions;
			auto& color_attrefs = rp.rpci.color_refs;
			auto& color_ref_offsets = rp.rpci.color_ref_offsets;
			auto& resolve_attrefs = rp.rpci.resolve_refs;
			auto& ds_attrefs = rp.rpci.ds_refs;

			// subpasses
			for (size_t i = 0; i < rp.subpasses.size(); i++) {
				vuk::SubpassDescription sd;
				size_t color_count = 0;
				if (i < rp.subpasses.size() - 1) {
					color_count = color_ref_offsets[i + 1] - color_ref_offsets[i];
				} else {
					color_count = color_attrefs.size() - color_ref_offsets[i];
				}
				{
					auto first = color_attrefs.data() + color_ref_offsets[i];
					sd.colorAttachmentCount = (uint32_t)color_count;
					sd.pColorAttachments = first;
				}

				sd.pDepthStencilAttachment = ds_attrefs[i] ? &*ds_attrefs[i] : nullptr;
				sd.flags = {};
				sd.inputAttachmentCount = 0;
				sd.pInputAttachments = nullptr;
				sd.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
				sd.preserveAttachmentCount = 0;
				sd.pPreserveAttachments = nullptr;
				{
					auto first = resolve_attrefs.data() + color_ref_offsets[i];
					sd.pResolveAttachments = first;
				}

				subp.push_back(sd);
			}

			rp.rpci.subpassCount = (uint32_t)rp.rpci.subpass_descriptions.size();
			rp.rpci.pSubpasses = rp.rpci.subpass_descriptions.data();

			rp.rpci.dependencyCount = (uint32_t)rp.rpci.subpass_dependencies.size();
			rp.rpci.pDependencies = rp.rpci.subpass_dependencies.data();
		}

		// perform sample count inference for framebuffers
		// loop through all renderpasses, and attempt to infer any sample count we can
		// then loop again, stopping if we have inferred all or have not made progress
		// resolve images are always sample count 1 and are excluded from the inference

		bool infer_progress = false;
		bool any_fb_incomplete = false;
		do {
			any_fb_incomplete = false;
			infer_progress = false;
			for (auto& rp : impl->rpis) {
				if (rp.attachments.size() == 0) {
					continue;
				}

				vuk::Samples fb_samples = rp.fbci.sample_count;
				bool samples_known = fb_samples != vuk::Samples::eInfer;

				if (samples_known) {
					continue;
				}

				// see if any attachment has a set sample count
				for (auto& attrpinfo : rp.attachments) {
					auto& bound = impl->bound_attachments[attrpinfo.name];

					if (bound.samples != vuk::Samples::eInfer && !attrpinfo.is_resolve_dst) {
						fb_samples = bound.samples;
						samples_known = true;
						break;
					}
				}

				// propagate known sample count onto attachments
				if (samples_known) {
					for (auto& attrpinfo : rp.attachments) {
						auto& bound = impl->bound_attachments[attrpinfo.name];
						if (!attrpinfo.is_resolve_dst) {
							bound.samples = fb_samples;
						}
					}
					rp.fbci.sample_count = fb_samples;
					infer_progress = true; // progress made
				} else {
					any_fb_incomplete = true;
				}
			}
		} while (any_fb_incomplete || infer_progress); // stop looping if all attachment have been sized or we made no progress

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

			rp.handle = ctx.acquire_renderpass(rp.rpci, ctx.frame_counter);
		}

		return { std::move(*this) };
	}

	MapProxy<Name, std::span<const UseRef>> RenderGraph::get_use_chains() {
		return &impl->use_chains;
	}

	MapProxy<Name, const AttachmentRPInfo&> RenderGraph::get_bound_attachments() {
		return &impl->bound_attachments;
	}

	vuk::ImageUsageFlags RenderGraph::compute_usage(std::span<const UseRef> chain) {
		vuk::ImageUsageFlags usage;
		for (const auto& c : chain) {
			switch (c.use.layout) {
			case vuk::ImageLayout::eDepthStencilAttachmentOptimal:
				usage |= vuk::ImageUsageFlagBits::eDepthStencilAttachment; break;
			case vuk::ImageLayout::eShaderReadOnlyOptimal: // TODO: more complex analysis
				usage |= vuk::ImageUsageFlagBits::eSampled; break;
			case vuk::ImageLayout::eColorAttachmentOptimal:
				usage |= vuk::ImageUsageFlagBits::eColorAttachment; break;
			case vuk::ImageLayout::eTransferSrcOptimal:
				usage |= vuk::ImageUsageFlagBits::eTransferSrc; break;
			case vuk::ImageLayout::eTransferDstOptimal:
				usage |= vuk::ImageUsageFlagBits::eTransferDst; break;
			default:;
			}
			// TODO: this isn't conservative enough, we need more information
			if (c.use.layout == vuk::ImageLayout::eGeneral) {
				if (c.use.stages & (vuk::PipelineStageFlagBits::eComputeShader | vuk::PipelineStageFlagBits::eVertexShader |
					vuk::PipelineStageFlagBits::eTessellationControlShader | vuk::PipelineStageFlagBits::eTessellationEvaluationShader |
					vuk::PipelineStageFlagBits::eGeometryShader | vuk::PipelineStageFlagBits::eFragmentShader)) {
					usage |= vuk::ImageUsageFlagBits::eStorage;
				}
			}
		}

		return usage;
	}
} // namespace vuk
