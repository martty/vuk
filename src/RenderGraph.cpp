#include "RenderGraph.hpp"
#include "Hash.hpp" // for create
#include "Cache.hpp"
#include "Context.hpp"
#include "CommandBuffer.hpp"
#include "Allocator.hpp"

namespace vuk {
	template<class T, class F>
	T* contains_if(std::vector<T>& v, F&& f) {
		auto it = std::find_if(v.begin(), v.end(), f);
		if (it != v.end()) return &(*it);
		else return nullptr;
	}

	template<class T, class F>
	T const* contains_if(const std::vector<T>& v, F&& f) {
		auto it = std::find_if(v.begin(), v.end(), f);
		if (it != v.end()) return &(*it);
		else return nullptr;
	}

	template<class T>
	T const* contains(const std::vector<T>& v, const T& f) {
		auto it = std::find(v.begin(), v.end(), f);
		if (it != v.end()) return &(*it);
		else return nullptr;
	}

	template <typename Iterator, typename Compare>
	void topological_sort(Iterator begin, Iterator end, Compare cmp) {
		while (begin != end) {
			auto const new_begin = std::partition(begin, end, [&](auto const& a) {
				return std::none_of(begin, end, [&](auto const& b) { return cmp(b, a); });
				});
			assert(new_begin != begin && "not a partial ordering");
			begin = new_begin;
		}
	}



	bool is_write_access(ImageAccess ia) {
		switch (ia) {
		case eColorResolveWrite:
		case eColorWrite:
		case eColorRW:
		case eDepthStencilRW:
		case eFragmentWrite:
			return true;
		default:
			return false;
		}
	}

	bool is_read_access(ImageAccess ia) {
		switch (ia) {
		case eColorResolveRead:
		case eColorRead:
		case eColorRW:
		case eDepthStencilRead:
		case eFragmentRead:
		case eFragmentSampled:
			return true;
		default:
			return false;
		}
	}

	Resource::Use to_use(ImageAccess ia) {
		switch (ia) {
		case eColorResolveWrite:
		case eColorWrite: return { vk::PipelineStageFlagBits::eColorAttachmentOutput, vk::AccessFlagBits::eColorAttachmentWrite, vk::ImageLayout::eColorAttachmentOptimal };
		case eColorRW: return { vk::PipelineStageFlagBits::eColorAttachmentOutput, vk::AccessFlagBits::eColorAttachmentWrite | vk::AccessFlagBits::eColorAttachmentRead, vk::ImageLayout::eColorAttachmentOptimal };
		case eColorResolveRead:
		case eColorRead: return { vk::PipelineStageFlagBits::eColorAttachmentOutput, vk::AccessFlagBits::eColorAttachmentRead, vk::ImageLayout::eColorAttachmentOptimal };
		case eDepthStencilRW : return { vk::PipelineStageFlagBits::eEarlyFragmentTests | vk::PipelineStageFlagBits::eLateFragmentTests, vk::AccessFlagBits::eDepthStencilAttachmentRead | vk::AccessFlagBits::eDepthStencilAttachmentWrite, vk::ImageLayout::eDepthStencilAttachmentOptimal };

		case eFragmentSampled: return { vk::PipelineStageFlagBits::eFragmentShader, vk::AccessFlagBits::eShaderRead, vk::ImageLayout::eShaderReadOnlyOptimal };
		default:
			assert(0 && "NYI");
			return {};
		}

	}

	bool is_framebuffer_attachment(const Resource& r) {
		if (r.type == Resource::Type::eBuffer) return false;
		switch (r.ia) {
		case eColorWrite:
		case eColorRW:
		case eDepthStencilRW:
		case eColorRead:
		case eDepthStencilRead:
		case eColorResolveRead:
		case eColorResolveWrite:
			return true;
		default:
			return false;
		}
	}

	bool is_framebuffer_attachment(Resource::Use u) {
		switch (u.layout) {
		case vk::ImageLayout::eColorAttachmentOptimal:
		case vk::ImageLayout::eDepthStencilAttachmentOptimal:
			return true;
		default:
			return false;
		}
	}

	bool is_write_access(Resource::Use u) {
		if (u.access & vk::AccessFlagBits::eColorAttachmentWrite) return true;
		if (u.access & vk::AccessFlagBits::eDepthStencilAttachmentWrite) return true;
		if (u.access & vk::AccessFlagBits::eShaderWrite) return true;
		assert(0 && "NYI");
		return false;
	}

	bool is_read_access(Resource::Use u) {
		return !is_write_access(u);
	}

	size_t format_to_size(vk::Format format) {
		switch (format) {
		case vk::Format::eR32G32B32A32Sfloat:
			return sizeof(float) * 4;
		case vk::Format::eR32G32B32Sfloat:
			return sizeof(float) * 3;
		case vk::Format::eR32G32Sfloat:
			return sizeof(float) * 2;
		case vk::Format::eR8G8B8A8Unorm:
			return sizeof(char) * 4;
		default:
			assert(0);
			return 0;
		}
	}

	uint32_t Ignore::to_size() {
		if (bytes != 0) return bytes;
		return (uint32_t)format_to_size(format);
	}

	FormatOrIgnore::FormatOrIgnore(vk::Format format) : ignore(false), format(format), size((uint32_t)format_to_size(format)) {
	}
	FormatOrIgnore::FormatOrIgnore(Ignore ign) : ignore(true), format(ign.format), size(ign.to_size()) {
	}

	// determine rendergraph inputs and outputs, and resources that are neither
	void RenderGraph::build_io() {
		std::unordered_set<Resource> inputs;
		std::unordered_set<Resource> outputs;

		for (auto& pif : passes) {
			for (auto& res : pif.pass.resources) {
				if (is_read_access(res.ia)) {
					pif.inputs.insert(res);
				}
				if (is_write_access(res.ia)) {
					pif.outputs.insert(res);
				}
			}
		
			for (auto& i : pif.inputs) {
				if (global_outputs.erase(i) == 0) {
					pif.global_inputs.insert(i);
				}
			}
			for (auto& i : pif.outputs) {
				if (global_inputs.erase(i) == 0) {
					pif.global_outputs.insert(i);
				}
			}

			global_inputs.insert(pif.global_inputs.begin(), pif.global_inputs.end());
			global_outputs.insert(pif.global_outputs.begin(), pif.global_outputs.end());

			inputs.insert(pif.inputs.begin(), pif.inputs.end());
			outputs.insert(pif.outputs.begin(), pif.outputs.end());
		}

		std::copy_if(outputs.begin(), outputs.end(), std::back_inserter(tracked), [&](auto& needle) { return !global_outputs.contains(needle); });
		global_io.insert(global_io.end(), global_inputs.begin(), global_inputs.end());
		global_io.insert(global_io.end(), global_outputs.begin(), global_outputs.end());
		global_io.erase(std::unique(global_io.begin(), global_io.end()), global_io.end());
	}

	Name resolve_name(Name in, const std::unordered_map<Name, Name> aliases) {
		auto it = aliases.find(in);
		if (it == aliases.end())
			return in;
		else
			return resolve_name(it->second, aliases);
	};

	void RenderGraph::build() {
		// compute sync
		// find which reads are graph inputs (not produced by any pass) & outputs (not consumed by any pass)
		build_io();
		// sort passes
		if (passes.size() > 1) {
			topological_sort(passes.begin(), passes.end(), [](const auto& p1, const auto& p2) {
				bool could_execute_after = false;
				bool could_execute_before = false;
				for (auto& o : p1.outputs) {
					for (auto& i : p2.inputs){
						if(i.src_name == o.use_name)
							could_execute_after = true;
					}
				}
				for (auto& o : p2.outputs) {
					for (auto& i : p1.inputs){
						if(i.src_name == o.use_name)
							could_execute_before = true;
					}
				}
				if (could_execute_after && could_execute_before) {
					return p1.pass.auxiliary_order < p2.pass.auxiliary_order;
				} else if (could_execute_after) {
					return true;
				} else 
					return false;
				});
		}
		// determine which passes are "head" and "tail" -> ones that can execute in the beginning or the end of the RG
		// -> the ones that only have global io
		for (auto& p : passes) {
			if (p.global_inputs.size() == p.inputs.size()) {
				head_passes.push_back(&p);
				p.is_head_pass = true;
			}
			if (p.global_outputs.size() == p.outputs.size()) {
				tail_passes.push_back(&p);
				p.is_tail_pass = true;
			}
		}

		// assemble use chains
		for (auto& passinfo : passes) {
			for (auto& res : passinfo.pass.resources) {
				if (res.src_name != res.use_name) {
					aliases[res.use_name] = res.src_name;
				}
				use_chains[resolve_name(res.use_name, aliases)].emplace_back(UseRef{ to_use(res.ia), &passinfo });
			}
		}

		// we need to collect passes into framebuffers, which will determine the renderpasses
		std::vector<std::pair<std::unordered_set<Resource>, std::vector<PassInfo*>>> attachment_sets;
		for (auto& passinfo : passes) {
			std::unordered_set<Resource> atts;

			for (auto& res : passinfo.pass.resources) {
				if(is_framebuffer_attachment(res))
					atts.insert(res);
			}
		
			if (auto p = contains_if(attachment_sets, [&](auto& t) { return t.first == atts; })) {
				p->second.push_back(&passinfo);
			} else {
				attachment_sets.emplace_back(atts, std::vector{ &passinfo });
			}
		}

		// renderpasses are uniquely identified by their index from now on
		// tell passes in which renderpass/subpass they will execute
		rpis.reserve(attachment_sets.size());
		for (auto& [attachments, passes] : attachment_sets) {
			RenderPassInfo rpi;
			auto rpi_index = rpis.size();

			uint32_t subpass = 0;
			for (auto& p : passes) {
				p->render_pass_index = rpi_index;
				p->subpass = subpass++;
				SubpassInfo si;
				si.pass = p;
				rpi.subpasses.push_back(si);
			}
			for (auto& att : attachments) {
				AttachmentRPInfo info;
				info.name = resolve_name(att.use_name, aliases);
				rpi.attachments.push_back(info);
			}
			rpis.push_back(rpi);
		}
	}

	void RenderGraph::bind_attachment_to_swapchain(Name name, SwapchainRef swp, Clear c) {
		AttachmentRPInfo attachment_info;
		attachment_info.extents = swp->extent;
		attachment_info.iv = {};
		// directly presented
		attachment_info.description.format = swp->format;
		attachment_info.samples = Samples::e1;

		attachment_info.type = AttachmentRPInfo::Type::eSwapchain;
		attachment_info.swapchain = swp;
		attachment_info.should_clear = true;
		attachment_info.clear_value = c;
		
		Resource::Use& initial = attachment_info.initial;
		Resource::Use& final = attachment_info.final;
		// for WSI, we want to wait for colourattachmentoutput
		// we don't care about any writes, we will clear
		initial.access = vk::AccessFlags{};
		initial.stages = vk::PipelineStageFlagBits::eColorAttachmentOutput;
		// clear
		initial.layout = vk::ImageLayout::ePreinitialized;
		/* Normally, we would need an external dependency at the end as well since we are changing layout in finalLayout,
   but since we are signalling a semaphore, we can rely on Vulkan's default behavior,
   which injects an external dependency here with
   dstStageMask = VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
   dstAccessMask = 0. */
		final.access = vk::AccessFlagBits{};
		final.layout = vk::ImageLayout::ePresentSrcKHR;
		final.stages = vk::PipelineStageFlagBits::eBottomOfPipe;

		bound_attachments.emplace(name, attachment_info);
	}

	void RenderGraph::mark_attachment_internal(Name name, vk::Format format, vuk::Extent2D extent, vuk::Samples samp, Clear c) {
		AttachmentRPInfo attachment_info;
		attachment_info.sizing = AttachmentRPInfo::Sizing::eAbsolute;
		attachment_info.extents = extent;
		attachment_info.iv = {};

		attachment_info.type = AttachmentRPInfo::Type::eInternal;
		attachment_info.description.format = format;
		attachment_info.samples = samp;

		attachment_info.should_clear = true;
		attachment_info.clear_value = c;
		Resource::Use& initial = attachment_info.initial;
		Resource::Use& final = attachment_info.final;
		initial.access = vk::AccessFlags{};
		initial.stages = vk::PipelineStageFlagBits::eTopOfPipe;
		// for internal attachments we don't want to preserve previous data
		initial.layout = vk::ImageLayout::ePreinitialized;

		// with an undefined final layout, there will be no final sync
		final.layout = vk::ImageLayout::eUndefined;
		final.access = vk::AccessFlagBits{};
		final.stages = vk::PipelineStageFlagBits::eBottomOfPipe;

		bound_attachments.emplace(name, attachment_info);
	}

	void RenderGraph::mark_attachment_internal(Name name, vk::Format format, vuk::Extent2D::Framebuffer fbrel, vuk::Samples samp, Clear c) {
		AttachmentRPInfo attachment_info;
		attachment_info.sizing = AttachmentRPInfo::Sizing::eFramebufferRelative;
		attachment_info.fb_relative = fbrel;
		attachment_info.iv = {};

		attachment_info.type = AttachmentRPInfo::Type::eInternal;
		attachment_info.description.format = format;
		attachment_info.samples = samp;

		attachment_info.should_clear = true;
		attachment_info.clear_value = c;
		Resource::Use& initial = attachment_info.initial;
		Resource::Use& final = attachment_info.final;
		initial.access = vk::AccessFlags{};
		initial.stages = vk::PipelineStageFlagBits::eTopOfPipe;
		// for internal attachments we don't want to preserve previous data
		initial.layout = vk::ImageLayout::ePreinitialized;

		// with an undefined final layout, there will be no final sync
		final.layout = vk::ImageLayout::eUndefined;
		final.access = vk::AccessFlagBits{};
		final.stages = vk::PipelineStageFlagBits::eBottomOfPipe;

		bound_attachments.emplace(name, attachment_info);
	}

	void RenderGraph::mark_attachment_resolve(Name resolved_name, Name ms_name) {
		add_pass({
			.resources = {
				vuk::Resource{ms_name, ms_name, vuk::Resource::Type::eImage, vuk::eColorResolveRead},
				vuk::Resource{resolved_name, resolved_name, vuk::Resource::Type::eImage, vuk::eColorResolveWrite}
			},
			.resolves = {{ms_name, resolved_name}}
		});
	}

	void RenderGraph::build(vuk::PerThreadContext& ptc) {
		for (auto& [raw_name, attachment_info] : bound_attachments) {
			auto name = resolve_name(raw_name, aliases);
			auto& chain = use_chains.at(name);
			chain.insert(chain.begin(), UseRef{ std::move(attachment_info.initial), nullptr });
			chain.emplace_back(UseRef{ attachment_info.final, nullptr });
		
			for (size_t i = 0; i < chain.size() - 1; i++) {
				auto& left = chain[i];
				auto& right = chain[i + 1];

				bool crosses_rpass = (left.pass == nullptr || right.pass == nullptr || left.pass->render_pass_index != right.pass->render_pass_index);
				if (crosses_rpass) {
					if (left.pass) { // RenderPass ->
						auto& left_rp = rpis[left.pass->render_pass_index];
						// if this is an attachment, we specify layout
						if (is_framebuffer_attachment(left.use)) {
							auto& rp_att = *contains_if(left_rp.attachments, [name](auto& att) {return att.name == name; });

							sync_bound_attachment_to_renderpass(rp_att, attachment_info);
							// if there is a "right" rp
							// or if this attachment has a required end layout
							// then we transition for it
							if (right.pass || right.use.layout != vk::ImageLayout::eUndefined) {
								rp_att.description.finalLayout = right.use.layout;
							} else {
								// we keep last use as finalLayout
								rp_att.description.finalLayout = left.use.layout;
							}
							// compute attachment store
							if (right.use.layout == vk::ImageLayout::eUndefined) {
								rp_att.description.storeOp = vk::AttachmentStoreOp::eDontCare;
							} else {
								rp_att.description.storeOp = vk::AttachmentStoreOp::eStore;
							}
						}
						// TODO: we need a dep here if there is a write or there is a layout transition
						if (/*left.use.layout != right.use.layout &&*/ right.use.layout != vk::ImageLayout::eUndefined) { // different layouts, need to have dependency
							vk::SubpassDependency sd;
							sd.dstAccessMask = right.use.access;
							sd.dstStageMask = right.use.stages;
							sd.srcSubpass = left.pass->subpass;
							sd.srcAccessMask = left.use.access;
							sd.srcStageMask = left.use.stages;
							sd.dstSubpass = VK_SUBPASS_EXTERNAL;
							left_rp.rpci.subpass_dependencies.push_back(sd);
						}
					}

					if (right.pass) { // -> RenderPass
						auto& right_rp = rpis[right.pass->render_pass_index];
						// if this is an attachment, we specify layout
						if (is_framebuffer_attachment(right.use)) {
							auto& rp_att = *contains_if(right_rp.attachments, [name](auto& att) {return att.name == name; });

							sync_bound_attachment_to_renderpass(rp_att, attachment_info);
							// we will have "left" transition for us
							if (left.pass) {
								rp_att.description.initialLayout = right.use.layout;
							} else {
								// if there is no "left" renderpass, then we take the initial layout
								rp_att.description.initialLayout = left.use.layout;
							}
							// compute attachment load
							if (left.use.layout == vk::ImageLayout::eUndefined) {
								rp_att.description.loadOp = vk::AttachmentLoadOp::eDontCare;
							} else if (left.use.layout == vk::ImageLayout::ePreinitialized) {
								// preinit means clear
								rp_att.description.initialLayout = vk::ImageLayout::eUndefined;
								rp_att.description.loadOp = vk::AttachmentLoadOp::eClear;
							} else {
								rp_att.description.loadOp = vk::AttachmentLoadOp::eLoad;
							}
						}
						// TODO: we need a dep here if there is a write or there is a layout transition
						if (/*right.use.layout != left.use.layout &&*/ left.use.layout != vk::ImageLayout::eUndefined) { // different layouts, need to have dependency
							vk::SubpassDependency sd;
							sd.dstAccessMask = right.use.access;
							sd.dstStageMask = right.use.stages;
							sd.dstSubpass = right.pass->subpass;
							sd.srcAccessMask = left.use.access;
							sd.srcStageMask = left.use.stages;
							sd.srcSubpass = VK_SUBPASS_EXTERNAL;
							right_rp.rpci.subpass_dependencies.push_back(sd);
						}
					}
				} else { // subpass-subpass link -> subpass - subpass dependency
					// WAW, WAR, RAW accesses need sync
					if (is_framebuffer_attachment(left.use) && (is_write_access(left.use) || (is_read_access(left.use) && is_write_access(right.use)))) {
						assert(left.pass->render_pass_index == right.pass->render_pass_index);
						auto& rp = rpis[right.pass->render_pass_index];
						vk::SubpassDependency sd;
						sd.dstAccessMask = right.use.access;
						sd.dstStageMask = right.use.stages;
						sd.dstSubpass = right.pass->subpass;
						sd.srcAccessMask = left.use.access;
						sd.srcStageMask = left.use.stages;
						sd.srcSubpass = left.pass->subpass;
						rp.rpci.subpass_dependencies.push_back(sd);
					}
				}
			}
		}

		for (auto& rp : rpis) {
			rp.rpci.color_ref_offsets.resize(rp.subpasses.size());
			rp.rpci.ds_refs.resize(rp.subpasses.size());
		}
	
		// we now have enough data to build vk::RenderPasses and vk::Framebuffers
		// we have to assign the proper attachments to proper slots
		// the order is given by the resource binding order
		for (auto& pass : passes) {
			auto& rp = rpis[pass.render_pass_index];
			auto subpass_index = pass.subpass;
			auto& color_attrefs = rp.rpci.color_refs;
			auto& resolve_attrefs = rp.rpci.resolve_refs;
			auto& color_ref_offsets = rp.rpci.color_ref_offsets;
			auto& ds_attrefs = rp.rpci.ds_refs;

			for (auto& res : pass.pass.resources) {
				if (!is_framebuffer_attachment(res))
					continue;
				if (res.ia == vuk::ImageAccess::eColorResolveWrite) // resolve attachment are added when processing the color attachment
					continue;
				vk::AttachmentReference attref;

				auto name = resolve_name(res.use_name, aliases);
				auto& chain = use_chains[name];
				auto cit = std::find_if(chain.begin(), chain.end(), [&](auto& useref) { return useref.pass == &pass; });
				assert(cit != chain.end());
				attref.layout = cit->use.layout;
				attref.attachment = (uint32_t)std::distance(rp.attachments.begin(), std::find_if(rp.attachments.begin(), rp.attachments.end(), [&](auto& att) { return name == att.name; }));

				if (attref.layout != vk::ImageLayout::eColorAttachmentOptimal) {
					if (attref.layout == vk::ImageLayout::eDepthStencilAttachmentOptimal) {
						ds_attrefs[subpass_index] = attref;
					}
				} else {

					vk::AttachmentReference rref;
					rref.attachment = VK_ATTACHMENT_UNUSED;
					if (auto it = pass.pass.resolves.find(res.use_name); it != pass.pass.resolves.end()) {
						// this a resolve src attachment
						// get the dst attachment
						auto& dst_name = it->second;
						rref.layout = vk::ImageLayout::eColorAttachmentOptimal; // the only possible layout for resolves
						rref.attachment = (uint32_t)std::distance(rp.attachments.begin(), std::find_if(rp.attachments.begin(), rp.attachments.end(), [&](auto& att) { return dst_name == att.name; }));
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

		for (auto& rp : rpis) {
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
				sd.pipelineBindPoint = vk::PipelineBindPoint::eGraphics;
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

			// attachments
			vuk::Samples samples(vk::SampleCountFlagBits::e1);
			for (auto& attrpinfo : rp.attachments) {
				auto& bound = bound_attachments.at(attrpinfo.name);
				if (!bound.samples.infer) {
					samples = bound.samples;
				}
			}
			for (auto& attrpinfo : rp.attachments) {
				if (attrpinfo.samples.infer) {
					attrpinfo.description.samples = samples.count;
				} else {
					attrpinfo.description.samples = attrpinfo.samples.count;
				}
				rp.rpci.attachments.push_back(attrpinfo.description);
			}
			
			rp.rpci.attachmentCount = (uint32_t)rp.rpci.attachments.size();
			rp.rpci.pAttachments = rp.rpci.attachments.data();

			rp.handle = ptc.renderpass_cache.acquire(rp.rpci);
		}
	}

	void sync_bound_attachment_to_renderpass(vuk::RenderGraph::AttachmentRPInfo& rp_att, vuk::RenderGraph::AttachmentRPInfo& attachment_info) {
		rp_att.description.format = attachment_info.description.format;
		rp_att.samples = attachment_info.samples;
		rp_att.description.samples = attachment_info.samples.count;
		rp_att.iv = attachment_info.iv;
		rp_att.extents = attachment_info.extents;
		rp_att.clear_value = attachment_info.clear_value;
		rp_att.should_clear = attachment_info.should_clear;
		rp_att.type = attachment_info.type;
	}

	vk::ImageUsageFlags RenderGraph::compute_usage(std::vector<vuk::RenderGraph::UseRef>& chain) {
		vk::ImageUsageFlags usage;
		for (auto& c : chain) {
			switch (c.use.layout) {
			case vk::ImageLayout::eDepthStencilAttachmentOptimal:
				usage |= vk::ImageUsageFlagBits::eDepthStencilAttachment; break;
			case vk::ImageLayout::eShaderReadOnlyOptimal:
				usage |= vk::ImageUsageFlagBits::eSampled; break;
			case vk::ImageLayout::eColorAttachmentOptimal:
				usage |= vk::ImageUsageFlagBits::eColorAttachment; break;
			default:;
			}
		}

		return usage;
	}

	void RenderGraph::create_attachment(PerThreadContext& ptc, Name name, RenderGraph::AttachmentRPInfo& attachment_info, vuk::Extent2D fb_extent, vk::SampleCountFlagBits samples) {
		auto& chain = use_chains.at(name);
		if (attachment_info.type == AttachmentRPInfo::Type::eInternal) {
			vk::ImageUsageFlags usage = compute_usage(chain);

			vk::ImageCreateInfo ici;
			ici.usage = usage;
			ici.arrayLayers = 1;
			// compute extent
			if (attachment_info.sizing == AttachmentRPInfo::Sizing::eFramebufferRelative) {
				assert(fb_extent.width > 0 && fb_extent.height > 0);
				ici.extent = vk::Extent3D(static_cast<uint32_t>(attachment_info.fb_relative.width * fb_extent.width), static_cast<uint32_t>(attachment_info.fb_relative.height * fb_extent.height), 1);
			} else {
				ici.extent = vk::Extent3D(attachment_info.extents, 1);
			}
			ici.imageType = vk::ImageType::e2D;
			ici.format = attachment_info.description.format;
			ici.mipLevels = 1;
			ici.initialLayout = vk::ImageLayout::eUndefined;
			ici.samples = samples;
			ici.sharingMode = vk::SharingMode::eExclusive;
			ici.tiling = vk::ImageTiling::eOptimal;

			vk::ImageViewCreateInfo ivci;
			ivci.image = vk::Image{};
			ivci.format = attachment_info.description.format;
			ivci.viewType = vk::ImageViewType::e2D;
			vk::ImageSubresourceRange isr;
			vk::ImageAspectFlagBits aspect;
			if (ici.format == vk::Format::eD32Sfloat) {
				aspect = vk::ImageAspectFlagBits::eDepth;
			} else {
				aspect = vk::ImageAspectFlagBits::eColor;
			}
			isr.aspectMask = aspect;
			isr.baseArrayLayer = 0;
			isr.layerCount = 1;
			isr.baseMipLevel = 0;
			isr.levelCount = 1;
			ivci.subresourceRange = isr;

			RGCI rgci;
			rgci.name = name;
			rgci.ici = ici;
			rgci.ivci = ivci;

			auto rg = ptc.transient_images.acquire(rgci);
			attachment_info.iv = rg.image_view;
			attachment_info.image = rg.image;
		}
	}

	vk::CommandBuffer RenderGraph::execute(vuk::PerThreadContext& ptc, std::vector<std::pair<SwapChainRef, size_t>> swp_with_index) {		
		// create framebuffers, create & bind attachments
		for (auto& rp : rpis) {
			vk::Extent2D fb_extent;
			bool extent_known = false;

			// bind swapchain attachments, deduce framebuffer size & sample count
			for (auto& attrpinfo : rp.attachments) {
				auto& bound = bound_attachments[attrpinfo.name];

				if (bound.type == AttachmentRPInfo::Type::eSwapchain) {
					auto it = std::find_if(swp_with_index.begin(), swp_with_index.end(), [&](auto& t) { return t.first == bound.swapchain; });
					bound.iv = it->first->image_views[it->second];
					fb_extent = it->first->extent;
					extent_known = true;
				} else {
					if (bound.sizing == AttachmentRPInfo::Sizing::eAbsolute) {
						fb_extent = bound.extents;
						extent_known = true;
					}
				}
			}

			for (auto& attrpinfo : rp.attachments) {
				auto& bound = bound_attachments[attrpinfo.name];
				if (extent_known) {
					bound.extents = fb_extent;
				}
			}
		}

		for (auto& rp : rpis) {
			auto& ivs = rp.fbci.attachments;
			std::vector<vk::ImageView> vkivs;
			vk::Extent2D fb_extent;

			for (auto& attrpinfo : rp.attachments) {
				auto& bound = bound_attachments[attrpinfo.name];
				if (bound.extents.width > 0 && bound.extents.height > 0)
					fb_extent = bound.extents;
			}
			
			// create internal attachments; bind attachments to fb
			for (auto& attrpinfo : rp.attachments) {
				auto& bound = bound_attachments[attrpinfo.name];
				if (bound.type == AttachmentRPInfo::Type::eInternal) {
					create_attachment(ptc, attrpinfo.name, bound, fb_extent, attrpinfo.description.samples);
				}

				ivs.push_back(bound.iv);
				vkivs.push_back(bound.iv.payload);
			}
			rp.fbci.renderPass = rp.handle;
			rp.fbci.width = fb_extent.width;
			rp.fbci.height = fb_extent.height;
			rp.fbci.pAttachments = &vkivs[0];
			rp.fbci.attachmentCount = (uint32_t)vkivs.size();
			rp.fbci.layers = 1;
			rp.framebuffer = ptc.framebuffer_cache.acquire(rp.fbci);
		}
		// actual execution
		auto cbufs = ptc.commandbuffer_pool.acquire(vk::CommandBufferLevel::ePrimary, 1);
		auto& cbuf = cbufs[0];

		vk::CommandBufferBeginInfo cbi;
		cbi.flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit;
		cbuf.begin(cbi);

		CommandBuffer cobuf(*this, ptc, cbuf);
		for (auto& rpass : rpis) {
			vk::RenderPassBeginInfo rbi;
			rbi.renderPass = rpass.handle;
			rbi.framebuffer = rpass.framebuffer;
			rbi.renderArea = vk::Rect2D(vk::Offset2D{}, vk::Extent2D{ rpass.fbci.width, rpass.fbci.height });
			std::vector<vk::ClearValue> clears;
			for (size_t i = 0; i < rpass.attachments.size(); i++) {
				auto& att = rpass.attachments[i];
				if (att.should_clear)
					clears.push_back(att.clear_value.c);
			}
			rbi.pClearValues = clears.data();
			rbi.clearValueCount = (uint32_t)clears.size();
			cbuf.beginRenderPass(rbi, vk::SubpassContents::eSecondaryCommandBuffers);
			for (size_t i = 0; i < rpass.subpasses.size(); i++) {
				auto& sp = rpass.subpasses[i];
				vuk::CommandBuffer::RenderPassInfo rpi;
				rpi.renderpass = rpass.handle;
				rpi.subpass = (uint32_t)i;
				rpi.extent = vk::Extent2D(rpass.fbci.width, rpass.fbci.height);
				auto& spdesc = rpass.rpci.subpass_descriptions[i];
				rpi.color_attachments = std::span<const vk::AttachmentReference>(spdesc.pColorAttachments, spdesc.colorAttachmentCount);
				for (auto& ca : rpi.color_attachments) {
					auto& att = rpass.attachments[ca.attachment];
					if (!att.samples.infer)
						rpi.samples = att.samples.count;
				}
				if (rpi.color_attachments.size() == 0) { // depth only pass, samples == 1
                    rpi.samples = vk::SampleCountFlagBits::e1;
				}
				cobuf.ongoing_renderpass = rpi;
				if (sp.pass->pass.execute) {
					if (!sp.pass->pass.name.empty()) {
						//ptc.ctx.debug.begin_region(cobuf.command_buffer, sp.pass->pass.name);
						sp.pass->pass.execute(cobuf);
						//ptc.ctx.debug.end_region(cobuf.command_buffer);
					} else {
						sp.pass->pass.execute(cobuf);
					}
				}
				cobuf.attribute_descriptions.clear();
				cobuf.binding_descriptions.clear();
				cobuf.set_bindings = {};
				cobuf.sets_used = {};
				if (i < rpass.subpasses.size() - 1)
					cbuf.nextSubpass(vk::SubpassContents::eSecondaryCommandBuffers);
			}
			cbuf.endRenderPass();
		}
		cbuf.end();
		return cbuf;
	}
}
