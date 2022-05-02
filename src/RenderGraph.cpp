#include "vuk/RenderGraph.hpp"
#include "RenderGraphImpl.hpp"
#include "RenderGraphUtil.hpp"
#include "vuk/Context.hpp"
#include "vuk/Exception.hpp"
#include "vuk/Future.hpp"

#include <set>
#include <unordered_set>

// intrinsics
namespace {
	void diverge(vuk::CommandBuffer&) {}
	void converge(vuk::CommandBuffer&) {}
} // namespace

namespace vuk {
	Name Resource::Subrange::Image::combine_name(Name prefix) const {
		std::string suffix = std::string(prefix.to_sv());
		suffix += "[" + std::to_string(base_layer) + ":" + std::to_string(base_layer + layer_count - 1) + "]";
		suffix += "[" + std::to_string(base_level) + ":" + std::to_string(base_level + level_count - 1) + "]";
		return Name(suffix.c_str());
	}

	RenderGraph::RenderGraph() : impl(new RGImpl) {
		name = Name(std::to_string(reinterpret_cast<uintptr_t>(impl)));
	}

	RenderGraph::RenderGraph(Name name) : impl(new RGImpl), name(name) {}

	RenderGraph::RenderGraph(RenderGraph&& o) noexcept : impl(std::exchange(o.impl, nullptr)) {}
	RenderGraph& RenderGraph::operator=(RenderGraph&& o) noexcept {
		impl = std::exchange(o.impl, nullptr);
		return *this;
	}

	RenderGraph::~RenderGraph() {
		delete impl;
	}

	void RenderGraph::add_pass(Pass p) {
		for (auto& r : p.resources) {
			if (r.is_create && r.type == Resource::Type::eImage) {
				attach_managed(r.name, (Format)r.ici.description.format, r.ici.attachment.extent, r.ici.attachment.sample_count, r.ici.attachment.clear_value);
			}
		}
		impl->passes.emplace_back(*impl->arena_, std::move(p));
	}

	void RenderGraph::append(Name subgraph_name, RenderGraph other) {
		Name joiner = subgraph_name.append("::");
		// TODO:
		// this code is written weird because of wonky allocators
		for (auto& p : other.impl->passes) {
			p.prefix = p.prefix.is_invalid() ? joiner : joiner.append(p.prefix);
			p.pass.name = joiner.append(p.pass.name);
			for (auto& r : p.pass.resources) {
				r.name = joiner.append(r.name);
				r.out_name = r.out_name.is_invalid() ? Name{} : joiner.append(r.out_name);
			}

			decltype(p.pass.resolves) resolves;
			for (auto& [n1, n2] : p.pass.resolves) {
				resolves.emplace(joiner.append(n1), joiner.append(n2));
			}
			p.pass.resolves = resolves;
			impl->passes.emplace_back(*impl->arena_, Pass{}) = std::move(p);
		}

		for (auto& [name, att] : other.impl->bound_attachments) {
			impl->bound_attachments.emplace(joiner.append(name), std::move(att));
		}
		for (auto& [name, buf] : other.impl->bound_buffers) {
			impl->bound_buffers.emplace(joiner.append(name), std::move(buf));
		}
	}

	void RenderGraph::add_alias(Name new_name, Name old_name) {
		if (new_name != old_name) {
			impl->aliases[new_name] = old_name;
		}
	}

	void RenderGraph::converge_image(Name pre_diverge, Name post_diverge) {
		// pass that consumes pre_diverge name
		add_pass({ .name = pre_diverge.append("_DIVERGE"),
		           .resources = { Resource{ pre_diverge, Resource::Type::eImage, Access::eConsume, pre_diverge.append("d") } },
		           .execute = diverge });

		add_pass({ .name = post_diverge.append("_CONVERGE"),
		           .resources = { Resource{ pre_diverge.append("d"), Resource::Type::eImage, Access::eConverge, post_diverge },  },
		           .execute = converge });
	}

	// determine rendergraph inputs and outputs, and resources that are neither
	void RenderGraph::build_io() {
		for (auto& pif : impl->passes) {
			for (Resource& res : pif.pass.resources) {
				Name in_name;
				Name out_name;
				if (res.type == Resource::Type::eImage && res.subrange.image != Resource::Subrange::Image{}) {
					in_name = res.subrange.image.combine_name(res.name);
					out_name = res.out_name.is_invalid() ? Name{} : res.subrange.image.combine_name(res.out_name);
				} else {
					in_name = res.name;
					out_name = res.out_name;
				}

				auto hashed_in_name = ::hash::fnv1a::hash(in_name.to_sv().data(), res.name.to_sv().size(), hash::fnv1a::default_offset_basis);
				auto hashed_out_name = ::hash::fnv1a::hash(out_name.to_sv().data(), res.out_name.to_sv().size(), hash::fnv1a::default_offset_basis);

				pif.input_names.emplace_back(in_name);
				pif.bloom_resolved_inputs |= hashed_in_name;

				if (!res.out_name.is_invalid()) {
					pif.bloom_outputs |= hashed_out_name;
					pif.output_names.emplace_back(out_name);
				}

				if (is_write_access(res.ia) || is_acquire(res.ia) || is_release(res.ia) || res.ia == Access::eConsume || res.ia == Access::eConverge) {
					assert(!impl->poisoned_names.contains(in_name)); // we have poisoned this name because a write has already consumed it
					pif.bloom_write_inputs |= hashed_in_name;
					pif.write_input_names.emplace_back(in_name);
					impl->poisoned_names.emplace(in_name);
				}

				// for image subranges, we additionally add a dependency on the diverged original resource
				// this resource is created by the diverged pass and consumed by the converge pass, thereby constraining all the passes who refer to these
				if (res.type == Resource::Type::eImage && res.subrange.image != Resource::Subrange::Image{}) {
					auto hashed_name = ::hash::fnv1a::hash(res.name.append("d").to_sv().data(), res.name.to_sv().size(), hash::fnv1a::default_offset_basis);

					pif.input_names.emplace_back(res.name.append("d"));
					pif.bloom_resolved_inputs |= hashed_name;
				}
			}
		}
	}

	void RenderGraph::schedule_intra_queue(std::span<PassInfo> passes, const RenderGraph::CompileOptions& compile_options) {
		// sort passes if requested
		// printf("-------------");
		if (passes.size() > 1 && compile_options.reorder_passes) {
			topological_sort(passes.begin(), passes.end(), [](const auto& p1, const auto& p2) {
				if (&p1 == &p2) {
					return false;
				}
				// p2 uses an input of p1 -> p2 after p1
				if ((p1.bloom_outputs & p2.bloom_resolved_inputs) != 0) {
					for (auto& o : p1.output_names) {
						for (auto& i : p2.input_names) {
							if (o == i) {
								// printf("\"%s\" -> \"%s\" [label=\"%s\"];\n",
								// p1.pass.name.c_str(), p2.pass.name.c_str(), i.c_str());
								return true; // p2 is ordered after p1
							}
						}
					}
				}
				// p2 writes to an input and p1 reads from the same input -> p2
				// after p1
				if ((p1.bloom_resolved_inputs & p2.bloom_write_inputs) != 0) {
					for (auto& o : p1.input_names) {
						for (auto& i : p2.write_input_names) {
							if (o == i) {
								// printf("\"%s\" -> \"%s\" [label=\"%s\"];\n",
								// p1.pass.name.c_str(), p2.pass.name.c_str(), i.c_str());
								return true; // p2 is ordered after p1
							}
						}
					}
				}

				return false;
			});
		}

		if (compile_options.check_pass_ordering) {
			for (auto it0 = passes.begin(); it0 != passes.end() - 1; ++it0) {
				for (auto it1 = it0; it1 < passes.end(); it1++) {
					auto& p1 = *it0;
					auto& p2 = *it1;

					bool could_execute_after = false;
					bool could_execute_before = false;

					if ((p1.bloom_outputs & p2.bloom_resolved_inputs) != 0) {
						for (auto& o : p1.output_names) {
							for (auto& in : p2.input_names) {
								if (o == in) {
									could_execute_after = true;
									break;
								}
							}
						}
					}

					if ((p2.bloom_outputs & p1.bloom_resolved_inputs) != 0) {
						for (auto& o : p2.output_names) {
							for (auto& in : p1.input_names) {
								if (o == in) {
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
	}

	void RenderGraph::compile(const RenderGraph::CompileOptions& compile_options) {
		// find which reads are graph inputs (not produced by any pass) & outputs
		// (not consumed by any pass)
		build_io();

		// run global pass ordering - once we split per-queue we don't see enough
		// inputs to order within a queue
		schedule_intra_queue(impl->passes, compile_options);

		// gather name alias info now - once we partition, we might encounter
		// unresolved aliases
		for (auto& passinfo : impl->passes) {
			for (auto& res : passinfo.pass.resources) {
				// for read or write, we add source to use chain
				if (!res.out_name.is_invalid()) {
					add_alias(res.out_name, res.name);
				}
			}
		}

		// for now, just use what the passes requested as domain
		for (auto& p : impl->passes) {
			p.domain = p.pass.execute_on;
		}

		// use chains pass
		impl->use_chains.clear();
		for (PassInfo& passinfo : impl->passes) {
			for (Resource& res : passinfo.pass.resources) {
				// for read or write, we add source to use chain
				auto resolved_name = impl->resolve_name(res.name);
				auto it = impl->use_chains.find(resolved_name);
				if (it == impl->use_chains.end()) {
					it = impl->use_chains.emplace(resolved_name, std::vector<UseRef, short_alloc<UseRef, 64>>{ short_alloc<UseRef, 64>{ *impl->arena_ } }).first;
				}
				auto& chain = it->second;

				if (chain.size() > 0 && is_acquire(chain.back().original)) { // acquire of resource - this must happen on the next use domain
					// propagate subsequent use back onto the acquire
					chain.back().pass->domain = passinfo.domain;
					chain.back().high_level_access = Access::eNone;
					chain.emplace_back(UseRef{ res.name, res.out_name, res.ia, res.ia, {}, res.type, res.subrange, &passinfo });
				} else if (chain.size() > 0 && is_release(chain.back().original) && is_acquire(res.ia)) {
					// release-acquire pair
					// mark these passes to be excluded
					chain.back().pass->domain = DomainFlagBits::eNone;
					passinfo.domain = DomainFlagBits::eNone;
					chain.pop_back();              // remove both from use chain
				} else if (is_release(res.ia)) { // release of resource - this must happen on the previous use domain
					assert(chain.size() > 0);      // release cannot head a use chain
					// only release -> propagate previous use onto release
					chain.emplace_back(UseRef{ res.name, res.out_name, res.ia, chain.back().high_level_access, {}, res.type, res.subrange, &passinfo });
				} else if (res.ia == Access::eConsume) { // is not a use
				} else {
					chain.emplace_back(UseRef{ res.name, res.out_name, res.ia, res.ia, {}, res.type, res.subrange, &passinfo });
				}
			}
		}

		// queue inference pass
		for (auto& [name, chain] : impl->use_chains) {
			DomainFlags last_domain = DomainFlagBits::eDevice;

			// forward inference
			for (uint64_t i = 0; i < chain.size(); i++) {
				auto& use_ref = chain[i];
				auto domain = use_ref.pass->domain;
				if (domain != last_domain && domain != DomainFlagBits::eDevice && domain != DomainFlagBits::eAny) {
					last_domain = use_ref.pass->domain;
				}
				if ((last_domain != DomainFlagBits::eDevice && last_domain != DomainFlagBits::eAny) &&
				    (domain == DomainFlagBits::eDevice || domain == DomainFlagBits::eAny)) {
					use_ref.pass->domain = last_domain;
				}
			}
			last_domain = DomainFlagBits::eDevice;
			// backward inference
			for (int64_t i = chain.size() - 1; i > 0; i--) {
				auto& use_ref = chain[i];
				auto domain = use_ref.pass->domain;

				if (domain != last_domain && domain != DomainFlagBits::eDevice && domain != DomainFlagBits::eAny) {
					last_domain = use_ref.pass->domain;
				}
				if ((last_domain != DomainFlagBits::eDevice && last_domain != DomainFlagBits::eAny) &&
				    (domain == DomainFlagBits::eDevice || domain == DomainFlagBits::eAny)) {
					use_ref.pass->domain = last_domain;
				}
			}
		}

		// queue inference failure fixup pass
		// we also prepare for pass sorting
		impl->ordered_passes.reserve(impl->passes.size());
		for (auto& p : impl->passes) {
			if (p.domain == DomainFlagBits::eDevice || p.domain == DomainFlagBits::eAny) { // couldn't infer, set pass as graphics
				p.domain = DomainFlagBits::eGraphicsOnGraphics;
			}
			impl->ordered_passes.push_back(&p);
		}

		// partition passes into different queues
		// TODO: queue inference
		auto transfer_begin = impl->ordered_passes.begin();
		auto transfer_end = std::stable_partition(
		    impl->ordered_passes.begin(), impl->ordered_passes.end(), [](const PassInfo* p) { return p->domain & DomainFlagBits::eTransferQueue; });
		auto compute_begin = transfer_end;
		auto compute_end =
		    std::stable_partition(transfer_end, impl->ordered_passes.end(), [](const PassInfo* p) { return p->domain & DomainFlagBits::eComputeQueue; });
		auto graphics_begin = compute_end;
		auto graphics_end =
		    std::stable_partition(compute_end, impl->ordered_passes.end(), [](const PassInfo* p) { return p->domain & DomainFlagBits::eGraphicsQueue; });
		std::span transfer_passes = { transfer_begin, transfer_end };
		std::span compute_passes = { compute_begin, compute_end };
		std::span graphics_passes = { graphics_begin, graphics_end };
		impl->ordered_passes.erase(graphics_end, impl->ordered_passes.end());

		// graphics: assemble renderpasses based on framebuffers
		// we need to collect passes into framebuffers, which will determine the renderpasses
		using attachment_set = std::unordered_set<Resource, std::hash<Resource>, std::equal_to<Resource>, short_alloc<Resource, 16>>;
		using passinfo_vec = std::vector<PassInfo*, short_alloc<PassInfo*, 16>>;
		std::vector<std::pair<attachment_set, passinfo_vec>, short_alloc<std::pair<attachment_set, passinfo_vec>, 8>> attachment_sets{ *impl->arena_ };
		for (auto& passinfo : graphics_passes) {
			attachment_set atts{ *impl->arena_ };

			for (auto& res : passinfo->pass.resources) {
				if (is_framebuffer_attachment(res))
					atts.insert(res);
			}

			if (auto p = attachment_sets.size() > 0 && attachment_sets.back().first == atts ? &attachment_sets.back() : nullptr) {
				p->second.push_back(passinfo);
			} else {
				passinfo_vec pv{ *impl->arena_ };
				pv.push_back(passinfo);
				attachment_sets.emplace_back(atts, pv);
			}
		}

		impl->num_graphics_rpis = attachment_sets.size();
		impl->num_compute_rpis = compute_passes.size();
		impl->num_transfer_rpis = transfer_passes.size();

		impl->rpis.clear();
		// renderpasses are uniquely identified by their index from now on
		// tell passes in which renderpass/subpass they will execute
		impl->rpis.reserve(impl->num_graphics_rpis + impl->num_compute_rpis + impl->num_transfer_rpis);
		for (auto& [attachments, passes] : attachment_sets) {
			RenderPassInfo rpi{ *impl->arena_ };
			auto rpi_index = impl->rpis.size();

			int32_t subpass = -1;
			for (auto& p : passes) {
				p->render_pass_index = rpi_index;
				if (rpi.subpasses.size() > 0) {
					auto& last_pass = rpi.subpasses.back().passes[0];
					// if the pass has the same inputs and outputs, we execute them on the
					// same subpass
					if (last_pass->input_names == p->input_names && last_pass->output_names == p->output_names) {
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
				info.name = impl->resolve_name(att.name);
				rpi.attachments.push_back(info);
			}

			if (attachments.size() == 0) {
				rpi.framebufferless = true;
			}

			impl->rpis.push_back(rpi);
		}

		// compute: just make rpis
		for (auto& passinfo : compute_passes) {
			RenderPassInfo rpi{ *impl->arena_ };
			auto rpi_index = impl->rpis.size();

			passinfo->render_pass_index = rpi_index;
			passinfo->subpass = 0;
			rpi.framebufferless = true;

			SubpassInfo si{ *impl->arena_ };
			si.passes.emplace_back(passinfo);
			si.use_secondary_command_buffers = false;
			rpi.subpasses.push_back(si);

			impl->rpis.push_back(rpi);
		}

		// transfer: just make rpis
		for (auto& passinfo : transfer_passes) {
			RenderPassInfo rpi{ *impl->arena_ };
			auto rpi_index = impl->rpis.size();

			passinfo->render_pass_index = rpi_index;
			passinfo->subpass = 0;
			rpi.framebufferless = true;

			SubpassInfo si{ *impl->arena_ };
			si.passes.emplace_back(passinfo);
			si.use_secondary_command_buffers = false;
			rpi.subpasses.push_back(si);

			impl->rpis.push_back(rpi);
		}
	}

	void RenderGraph::resolve_resource_into(Name resolved_name_src, Name resolved_name_dst, Name ms_name) {
		add_pass({ .resources = { Resource{ ms_name, Resource::Type::eImage, eColorResolveRead, {} },
		                          Resource{ resolved_name_src, Resource::Type::eImage, eColorResolveWrite, resolved_name_dst } },
		           .resolves = { { ms_name, resolved_name_src } } });
	}

	void RenderGraph::attach_swapchain(Name name, SwapchainRef swp, Clear c) {
		AttachmentRPInfo attachment_info;
		attachment_info.attachment.extent = Dimension2D::absolute(swp->extent);
		// directly presented
		attachment_info.description.format = (VkFormat)swp->format;
		attachment_info.attachment.sample_count = Samples::e1;

		attachment_info.type = AttachmentRPInfo::Type::eSwapchain;
		attachment_info.swapchain = swp;
		attachment_info.should_clear = true;
		attachment_info.attachment.clear_value = c;

		ResourceUse& initial = attachment_info.initial;
		ResourceUse& final = attachment_info.final;
		// for WSI, we want to wait for colourattachmentoutput
		// we don't care about any writes, we will clear
		initial.access = AccessFlags{};
		initial.stages = PipelineStageFlagBits::eColorAttachmentOutput;
		// clear
		initial.layout = ImageLayout::ePreinitialized;
		/* Normally, we would need an external dependency at the end as well since
	 we are changing layout in finalLayout, but since we are signalling a
	 semaphore, we can rely on Vulkan's default behavior, which injects an
	 external dependency here with dstStageMask =
	 VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, dstAccessMask = 0. */
		final.access = AccessFlagBits{};
		final.layout = ImageLayout::ePresentSrcKHR;
		final.stages = PipelineStageFlagBits::eBottomOfPipe;

		impl->bound_attachments.emplace(name, attachment_info);
	}

	void RenderGraph::attach_managed(Name name, Format format, Dimension2D extent, Samples samp, Clear c) {
		AttachmentRPInfo attachment_info;
		attachment_info.attachment.extent = extent;

		attachment_info.type = AttachmentRPInfo::Type::eInternal;
		attachment_info.description.format = (VkFormat)format;
		attachment_info.attachment.sample_count = samp;

		attachment_info.should_clear = true;
		attachment_info.attachment.clear_value = c;
		ResourceUse& initial = attachment_info.initial;
		ResourceUse& final = attachment_info.final;
		initial.access = AccessFlags{};
		initial.stages = PipelineStageFlagBits::eTopOfPipe;
		// for internal attachments we don't want to preserve previous data
		initial.layout = ImageLayout::ePreinitialized;

		// with an undefined final layout, there will be no final sync
		final.layout = ImageLayout::eUndefined;
		final.access = AccessFlagBits{};
		final.stages = PipelineStageFlagBits::eBottomOfPipe;

		impl->bound_attachments.emplace(name, attachment_info);
	}

	void RenderGraph::attach_buffer(Name name, Buffer buf, Access initial, Access final) {
		BufferInfo buf_info{ .name = name, .initial = to_use(initial), .final = to_use(final), .buffer = buf };
		impl->bound_buffers.emplace(name, buf_info);
	}

	void RenderGraph::attach_image(Name name, ImageAttachment att, Access initial_acc, Access final_acc) {
		AttachmentRPInfo attachment_info;
		attachment_info.attachment = att;

		attachment_info.type = AttachmentRPInfo::Type::eExternal;
		attachment_info.description.format = (VkFormat)att.format;

		attachment_info.should_clear = initial_acc == Access::eClear; // if initial access was clear, we will clear
		ResourceUse& initial = attachment_info.initial;
		ResourceUse& final = attachment_info.final;
		initial = to_use(initial_acc);
		final = to_use(final_acc);
		impl->bound_attachments.emplace(name, attachment_info);
	}

	void RenderGraph::attach_in(Name name, Future<ImageAttachment>&& fimg) {
		if (fimg.get_status() == FutureBase::Status::eSubmitted || fimg.get_status() == FutureBase::Status::eHostAvailable) {
			auto att = fimg.control->get_result<ImageAttachment>();
			AttachmentRPInfo attachment_info;
			attachment_info.attachment = att;

			attachment_info.type = AttachmentRPInfo::Type::eExternal;
			attachment_info.description.format = (VkFormat)att.format;

			attachment_info.should_clear = false;
			attachment_info.initial = { fimg.control->last_use.stages, fimg.control->last_use.access, fimg.control->last_use.layout };
			attachment_info.final = to_use(vuk::Access::eNone);

			add_pass({ .name = fimg.output_binding.append("_FUTURE_ACQUIRE"),
			           .resources = { Resource{ name.append("_"), Resource::Type::eImage, eAcquire, name } },
			           .wait = std::move(fimg.control) });

			impl->bound_attachments.emplace(name.append("_"), attachment_info);
		} else if (fimg.get_status() == FutureBase::Status::eRenderGraphBound || fimg.get_status() == FutureBase::Status::eOutputAttached) {
			fimg.get_status() = FutureBase::Status::eInputAttached;
			if (fimg.rg->impl) {
				append(this->name, std::move(*fimg.rg));
			}
			add_pass({ .name = fimg.output_binding.append("_FUTURE_ACQUIRE"),
			           .resources = { Resource{ this->name.append("::").append(fimg.output_binding).append("+"), Resource::Type::eImage, eAcquire, name } } });
			add_alias(name, this->name.append("::").append(fimg.output_binding));
		} else {
			assert(0);
		}
	}

	void RenderGraph::attach_in(Name name, Future<Buffer>&& fimg) {
		if (fimg.get_status() == FutureBase::Status::eSubmitted || fimg.get_status() == FutureBase::Status::eHostAvailable) {
			BufferInfo buf_info{ .name = name,
				                   .initial = { fimg.control->last_use.stages, fimg.control->last_use.access, fimg.control->last_use.layout },
				                   .final = to_use(vuk::Access::eNone),
				                   .buffer = fimg.control->get_result<Buffer>() };
			add_pass({ .name = fimg.output_binding.append("_FUTURE_ACQUIRE"),
			           .resources = { Resource{ name.append("_"), Resource::Type::eBuffer, eAcquire, name } },
			           .wait = std::move(fimg.control) });
			impl->bound_buffers.emplace(name.append("_"), buf_info);
		} else if (fimg.get_status() == FutureBase::Status::eRenderGraphBound || fimg.get_status() == FutureBase::Status::eOutputAttached) {
			if (fimg.rg->impl) {
				append(this->name, std::move(*fimg.rg));
			}
			fimg.get_status() = FutureBase::Status::eInputAttached;
			add_pass({ .name = fimg.output_binding.append("_FUTURE_ACQUIRE"),
			           .resources = { Resource{ this->name.append("::").append(fimg.output_binding).append("+"), Resource::Type::eBuffer, eAcquire, name } } });
			add_alias(name, this->name.append("::").append(fimg.output_binding));
		} else {
			assert(0);
		}
	}

	void RenderGraph::attach_out(Name name, Future<ImageAttachment>& fimg, DomainFlags dst_domain) {
		fimg.get_status() = FutureBase::Status::eOutputAttached;
		add_pass({ .name = name.append("_RELEASE"),
		           .execute_on = DomainFlagBits::eDevice,
		           .resources = { Resource{ name, Resource::Type::eImage, domain_to_release_access(dst_domain), name.append("+") } },
		           .signal = fimg.control.get() });
	}

	void RenderGraph::attach_out(Name name, Future<Buffer>& fbuf, DomainFlags dst_domain) {
		fbuf.get_status() = FutureBase::Status::eOutputAttached;
		add_pass({ .name = name.append("_RELEASE"),
		           .execute_on = DomainFlagBits::eDevice,
		           .resources = { Resource{ name, Resource::Type::eBuffer, domain_to_release_access(dst_domain), name.append("+") } },
		           .signal = fbuf.control.get() });
	}

	void sync_bound_attachment_to_renderpass(AttachmentRPInfo& rp_att, AttachmentRPInfo& attachment_info) {
		rp_att.description.format = attachment_info.description.format;
		rp_att.attachment = attachment_info.attachment;
		rp_att.description.samples = (VkSampleCountFlagBits)attachment_info.attachment.sample_count.count;
		rp_att.should_clear = attachment_info.should_clear;
		rp_att.type = attachment_info.type;
	}

	void RenderGraph::validate() {
		// check if all resourced are attached
		for (const auto& [n, v] : impl->use_chains) {
			auto name = impl->resolve_name(n);
			if (!impl->bound_attachments.contains(name) && !impl->bound_buffers.contains(name)) {
				throw RenderGraphException{ std::string("Missing resource: \"") + std::string(n.to_sv()) + "\". Did you forget to attach it?" };
			}
		}
	}

	ExecutableRenderGraph RenderGraph::link(Context& ctx, const RenderGraph::CompileOptions& compile_options) && {
		compile(compile_options);

		// at this point the graph is built, we know of all the resources and
		// everything should have been attached perform checking if this indeed the
		// case
		validate();

		for (auto& [raw_name, attachment_info] : impl->bound_attachments) {
			auto name = impl->resolve_name(raw_name);
			auto chain_it = impl->use_chains.find(name);
			if (chain_it == impl->use_chains.end()) {
				// TODO: warning here, if turned on
				continue;
			}
			auto& chain = chain_it->second;
			if (is_acquire(chain[0].original)) {
			} else {
				chain.insert(chain.begin(), UseRef{ {}, {}, vuk::eManual, vuk::eManual, attachment_info.initial, Resource::Type::eImage, {}, nullptr });
			}
			if (is_release(chain.back().original)) {
				assert(chain.size() > 2); // single chain of release is not valid
				auto& last_use_before_release = *(chain.end() - 1);
				if (attachment_info.final.layout != ImageLayout::eUndefined) {
					chain.insert(chain.end() - 1,
					             UseRef{ {}, {}, vuk::eManual, vuk::eManual, attachment_info.final, Resource::Type::eImage, {}, last_use_before_release.pass });
				}
			} else {
				chain.emplace_back(UseRef{ {}, {}, vuk::eManual, vuk::eManual, attachment_info.final, Resource::Type::eImage, {}, nullptr });
			}

			ImageAspectFlags aspect = format_to_aspect((Format)attachment_info.description.format);

			bool is_diverged = false;
			ResourceUse original;
			for (size_t i = 0; i < chain.size() - 1; i++) {
				auto* left = &chain[i];
				auto& right = chain[i + 1];

				DomainFlags left_domain = left->pass ? left->pass->domain : DomainFlagBits::eNone;
				DomainFlags right_domain = right.pass ? right.pass->domain : DomainFlagBits::eNone;

				if (right.high_level_access == Access::eConverge) {
					continue;
				}
				if (left->high_level_access == Access::eConverge) {
					auto dst_use = to_use(right.high_level_access);
					auto& left_rp = impl->rpis[left->pass->render_pass_index];
					ResourceUse original;
					// we need to reconverge this diverged image
					// to do this we will walk backwards, and any use we find that is diverged, we will converge it into the dst_access
					std::unordered_set<Resource::Subrange::Image> layer_level_visited;
					for (int64_t j = i - 1; j >= 0; j--) {
						auto& ch = chain[j];
						if (ch.subrange.image != Resource::Subrange::Image{}) { // diverged
							// check if we have already visited these layer x level combinations
							if (auto [iter, new_elem] = layer_level_visited.emplace(ch.subrange.image); !new_elem) {
								continue;
							}
							// if not, then emit convergence barrier
							VkImageMemoryBarrier barrier{ .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER };
							ImageBarrier ib{};
							barrier.srcAccessMask = is_read_access(ch.use) ? 0 : (VkAccessFlags)ch.use.access;
							barrier.dstAccessMask = (VkAccessFlags)dst_use.access;
							barrier.oldLayout = (VkImageLayout)ch.use.layout;
							barrier.newLayout = (VkImageLayout)dst_use.layout;
							barrier.subresourceRange.aspectMask = (VkImageAspectFlags)aspect;
							barrier.subresourceRange.baseArrayLayer = ch.subrange.image.base_layer;
							barrier.subresourceRange.baseMipLevel = ch.subrange.image.base_level;
							barrier.subresourceRange.layerCount = ch.subrange.image.layer_count;
							barrier.subresourceRange.levelCount = ch.subrange.image.level_count;
							barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
							barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
							ib.src = ch.use.stages;
							ib.dst = dst_use.stages;
							ib.barrier = barrier;
							ib.image = name;
							// attach this barrier to the end of subpass or end of renderpass
							if (left_rp.framebufferless) {
								left_rp.subpasses[left->pass->subpass].post_barriers.push_back(ib);
							} else {
								left_rp.post_barriers.push_back(ib);
							}
						} else {
							// found the last converged use, remember the use there and stop looping
							original = ch.use;
							break;
						}
					}
					std::vector<Resource::Subrange::Image> ll_vec;
					std::copy(layer_level_visited.begin(), layer_level_visited.end(), std::back_inserter(ll_vec));
					// sort layers/levels
					std::sort(ll_vec.begin(), ll_vec.end());
					if (ll_vec[0].base_layer > 0) {
						// emit a global pre-barrier: all full layers before the first visited
						assert("NYI"); // TODO: non-zero based layers
					}
					// merge into ranges
					for (uint64_t i = 1; i < ll_vec.size();) {
						auto& prev = ll_vec[i - 1];
						auto& curr = ll_vec[i];
						// TODO: not doing layer merging
						assert(prev.base_layer == curr.base_layer && prev.layer_count == curr.layer_count);
						if (prev.base_level + prev.level_count == curr.base_level) { // merge
							prev.level_count += curr.level_count;
							ll_vec.erase(ll_vec.begin() + i);
						} else { // no merge
							i++;
						}
					}

					for (auto& ll : ll_vec) {
						// TODO: not emitting pre-post barriers yet
						continue;
						if (ll.base_level > 0) { // pre-barrier: all mips before first
							VkImageMemoryBarrier barrier{ .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER };
							ImageBarrier ib{};
							barrier.srcAccessMask = is_read_access(original) ? 0 : (VkAccessFlags)original.access;
							barrier.dstAccessMask = (VkAccessFlags)dst_use.access;
							barrier.oldLayout = (VkImageLayout)original.layout;
							barrier.newLayout = (VkImageLayout)dst_use.layout;
							barrier.subresourceRange.aspectMask = (VkImageAspectFlags)aspect;
							barrier.subresourceRange.baseArrayLayer = ll.base_layer;
							barrier.subresourceRange.baseMipLevel = 0;
							barrier.subresourceRange.layerCount = ll.layer_count;
							barrier.subresourceRange.levelCount = ll.base_level;
							barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
							barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
							ib.src = original.stages;
							ib.dst = dst_use.stages;
							ib.barrier = barrier;
							ib.image = name;
							// attach this barrier to the end of subpass or end of renderpass
							if (left_rp.framebufferless) {
								left_rp.subpasses[left->pass->subpass].post_barriers.push_back(ib);
							} else {
								left_rp.post_barriers.push_back(ib);
							}
						}
						// post-barrier: remaining mips after last
						VkImageMemoryBarrier barrier{ .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER };
						ImageBarrier ib{};
						barrier.srcAccessMask = is_read_access(original) ? 0 : (VkAccessFlags)original.access;
						barrier.dstAccessMask = (VkAccessFlags)dst_use.access;
						barrier.oldLayout = (VkImageLayout)original.layout;
						barrier.newLayout = (VkImageLayout)dst_use.layout;
						barrier.subresourceRange.aspectMask = (VkImageAspectFlags)aspect;
						barrier.subresourceRange.baseArrayLayer = ll.base_layer;
						barrier.subresourceRange.baseMipLevel = ll.level_count;
						barrier.subresourceRange.layerCount = ll.layer_count;
						barrier.subresourceRange.levelCount = VK_REMAINING_MIP_LEVELS;
						barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
						barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
						ib.src = original.stages;
						ib.dst = dst_use.stages;
						ib.barrier = barrier;
						ib.image = name;
						// attach this barrier to the end of subpass or end of renderpass
						if (left_rp.framebufferless) {
							left_rp.subpasses[left->pass->subpass].post_barriers.push_back(ib);
						} else {
							left_rp.post_barriers.push_back(ib);
						}
					}
					// emit a global post-barrier: remaining full layers after the last visited
					if (ll_vec[ll_vec.size() - 1].layer_count != VK_REMAINING_ARRAY_LAYERS) {
						assert("NYI"); // TODO: non-full layers
					}
					is_diverged = true;
					continue;
				}

				// first divergence
				if (left->subrange.image == Resource::Subrange::Image{} && right.subrange.image != Resource::Subrange::Image{}) {
					is_diverged = true;
				}

				ResourceUse prev_use;
				prev_use = left->use = left->high_level_access == Access::eManual ? left->use : to_use(left->high_level_access);
				ResourceUse next_use;
				next_use = right.use = right.high_level_access == Access::eManual ? right.use : to_use(right.high_level_access);
				auto subrange = right.subrange.image;

				// if the image is diverged, then we need to find a matching previous use in the chain - either a use whose range intersects or the undiverged use
				if (is_diverged) {
					for (int64_t j = i; j >= 0; j--) {
						auto& ch = chain[j];
						if (ch.subrange.image == right.subrange.image) { // TODO: we want subset not equality
							prev_use = ch.use;
							left = &ch;
							break;
						}
						if (ch.subrange.image == Resource::Subrange::Image{}) { // TODO: subset covers this case
							prev_use = ch.use;
							left = &ch;
							break;
						}
					}
				}

				auto src_stages = prev_use.stages;
				auto dst_stages = next_use.stages;

				if (is_acquire(left->original)) {
					// acquire without release - must be first in chain
					assert(i == 0);
					// we are acquiring from a future
					auto wait_fut = left->pass->pass.wait.get();
					if (wait_fut) {
						left->pass->absolute_waits.emplace_back(wait_fut->initial_domain, wait_fut->initial_visibility);

						left_domain = wait_fut->initial_domain;
						src_stages = wait_fut->last_use.stages;
						prev_use.layout = wait_fut->last_use.layout;
						prev_use.access = wait_fut->last_use.access;
					} else {
						// we are acquiring from a queue
						DomainFlags src_domain;
						switch (left->original) {
						case eAcquireFromGraphics:
							src_domain = DomainFlagBits::eGraphicsQueue;
							break;
						case eAcquireFromCompute:
							src_domain = DomainFlagBits::eComputeQueue;
							break;
						case eAcquireFromTransfer:
							src_domain = DomainFlagBits::eTransferQueue;
							break;
						default:
							assert(0 && "Acquire from queue without Queue");
						}

						left_domain = src_domain;
					}
				}

				scope_to_domain(src_stages, left_domain & DomainFlagBits::eQueueMask);
				scope_to_domain(dst_stages, right_domain & DomainFlagBits::eQueueMask);

				bool crosses_queue = (left_domain != DomainFlagBits::eNone && right_domain != DomainFlagBits::eNone &&
				                      (left_domain & DomainFlagBits::eQueueMask) != (right_domain & DomainFlagBits::eQueueMask));

				if (is_acquire(left->original)) {
					if (crosses_queue) {
						ImageBarrier acquire_barrier;
						VkImageMemoryBarrier barrier{ .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER };
						barrier.srcAccessMask = 0; // ignored
						barrier.dstAccessMask = (VkAccessFlags)next_use.access;
						barrier.oldLayout = (VkImageLayout)prev_use.layout;
						barrier.newLayout = (VkImageLayout)next_use.layout;
						barrier.subresourceRange.aspectMask = (VkImageAspectFlags)aspect;
						barrier.subresourceRange.baseArrayLayer = 0;
						barrier.subresourceRange.baseMipLevel = 0;
						barrier.subresourceRange.layerCount = VK_REMAINING_ARRAY_LAYERS;
						barrier.subresourceRange.levelCount = VK_REMAINING_MIP_LEVELS;
						barrier.srcQueueFamilyIndex = ctx.domain_to_queue_family_index(left_domain);
						barrier.dstQueueFamilyIndex = ctx.domain_to_queue_family_index(right_domain);
						acquire_barrier.src = PipelineStageFlagBits::eTopOfPipe; // NONE
						acquire_barrier.dst = dst_stages;
						acquire_barrier.barrier = barrier;
						acquire_barrier.image = name;
						auto& right_rp = impl->rpis[right.pass->render_pass_index];
						right_rp.pre_barriers.emplace_back(acquire_barrier);
					}
				}

				if (is_release(right.original)) {
					// release without acquire - must be last in chain
					assert(i + 1 == (chain.size() - 1));
					if (right.pass->pass.signal) {
						auto& fut = *right.pass->pass.signal;
						fut.last_use = QueueResourceUse{
							left->original, prev_use.stages, prev_use.access, prev_use.layout, (DomainFlagBits)(left_domain & DomainFlagBits::eQueueMask).m_mask
						};
						attachment_info.attached_future = &fut;
					}

					DomainFlags dst_domain;
					switch (right.original) {
					case eReleaseToGraphics:
						dst_domain = DomainFlagBits::eGraphicsQueue;
						break;
					case eReleaseToCompute:
						dst_domain = DomainFlagBits::eComputeQueue;
						break;
					case eReleaseToTransfer:
						dst_domain = DomainFlagBits::eTransferQueue;
						break;
					default:
						dst_domain = left->pass->domain & DomainFlagBits::eQueueMask; // no domain change
					}

					bool release_to_different_queue = (left->pass->domain & DomainFlagBits::eQueueMask) != dst_domain;

					if (release_to_different_queue) { // release half of QFOT
						ImageBarrier release_barrier;
						VkImageMemoryBarrier barrier{ .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER };
						barrier.srcAccessMask = is_read_access(prev_use) ? 0 : (VkAccessFlags)prev_use.access;
						barrier.dstAccessMask = 0;                          // ignored
						barrier.oldLayout = (VkImageLayout)prev_use.layout; // no layout transition - we don't know
						barrier.newLayout = (VkImageLayout)prev_use.layout;
						barrier.subresourceRange.aspectMask = (VkImageAspectFlags)aspect;
						barrier.subresourceRange.baseArrayLayer = 0;
						barrier.subresourceRange.baseMipLevel = 0;
						barrier.subresourceRange.layerCount = VK_REMAINING_ARRAY_LAYERS;
						barrier.subresourceRange.levelCount = VK_REMAINING_MIP_LEVELS;
						barrier.srcQueueFamilyIndex = ctx.domain_to_queue_family_index(left_domain);
						barrier.dstQueueFamilyIndex = ctx.domain_to_queue_family_index(dst_domain);
						release_barrier.src = src_stages;
						release_barrier.dst = PipelineStageFlagBits::eBottomOfPipe; // NONE
						release_barrier.barrier = barrier;
						release_barrier.image = name;
						auto& left_rp = impl->rpis[left->pass->render_pass_index];
						left_rp.post_barriers.emplace_back(release_barrier);
					}

					auto& left_rp = impl->rpis[left->pass->render_pass_index];
					if (is_framebuffer_attachment(prev_use)) {
						assert(!left_rp.framebufferless);
						auto& rp_att = *contains_if(left_rp.attachments, [name](auto& att) { return att.name == name; });

						sync_bound_attachment_to_renderpass(rp_att, attachment_info);
						// we keep last use as finalLayout
						assert(prev_use.layout != ImageLayout::eUndefined && prev_use.layout != ImageLayout::ePreinitialized);
						rp_att.description.finalLayout = (VkImageLayout)prev_use.layout;

						// compute attachment store
						if (next_use.layout == ImageLayout::eUndefined) {
							rp_att.description.storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
						} else {
							rp_att.description.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
						}
					}

					continue;
				}

				if (crosses_queue) {
					left->pass->is_waited_on = true;
					right.pass->waits.emplace_back((DomainFlagBits)(left_domain & DomainFlagBits::eQueueMask).m_mask, left->pass);

					assert(prev_use.layout != ImageLayout::ePreinitialized);
					assert(next_use.layout != ImageLayout::eUndefined);
					// all the images are exclusive -> QFOT

					{
						ImageBarrier release_barrier;
						VkImageMemoryBarrier barrier{ .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER };
						barrier.srcAccessMask = is_read_access(prev_use) ? 0 : (VkAccessFlags)prev_use.access;
						barrier.dstAccessMask = 0; // ignored
						barrier.oldLayout = (VkImageLayout)prev_use.layout;
						barrier.newLayout = (VkImageLayout)next_use.layout;
						barrier.subresourceRange.aspectMask = (VkImageAspectFlags)aspect;
						barrier.subresourceRange.baseArrayLayer = subrange.base_layer;
						barrier.subresourceRange.baseMipLevel = subrange.base_level;
						barrier.subresourceRange.layerCount = subrange.layer_count;
						barrier.subresourceRange.levelCount = subrange.level_count;
						barrier.srcQueueFamilyIndex = ctx.domain_to_queue_family_index(left_domain);
						barrier.dstQueueFamilyIndex = ctx.domain_to_queue_family_index(right_domain);
						release_barrier.src = src_stages;
						release_barrier.dst = PipelineStageFlagBits::eBottomOfPipe; // NONE
						release_barrier.barrier = barrier;
						release_barrier.image = name;
						auto& left_rp = impl->rpis[left->pass->render_pass_index];
						left_rp.post_barriers.emplace_back(release_barrier);
					}
					{
						ImageBarrier acquire_barrier;
						VkImageMemoryBarrier barrier{ .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER };
						barrier.srcAccessMask = 0; // ignored
						barrier.dstAccessMask = (VkAccessFlags)next_use.access;
						barrier.oldLayout = (VkImageLayout)prev_use.layout;
						barrier.newLayout = (VkImageLayout)next_use.layout;
						barrier.subresourceRange.aspectMask = (VkImageAspectFlags)aspect;
						barrier.subresourceRange.baseArrayLayer = subrange.base_layer;
						barrier.subresourceRange.baseMipLevel = subrange.base_level;
						barrier.subresourceRange.layerCount = subrange.layer_count;
						barrier.subresourceRange.levelCount = subrange.level_count;
						barrier.srcQueueFamilyIndex = ctx.domain_to_queue_family_index(left_domain);
						barrier.dstQueueFamilyIndex = ctx.domain_to_queue_family_index(right_domain);
						acquire_barrier.src = PipelineStageFlagBits::eTopOfPipe; // NONE
						acquire_barrier.dst = dst_stages;
						acquire_barrier.barrier = barrier;
						acquire_barrier.image = name;
						auto& right_rp = impl->rpis[right.pass->render_pass_index];
						right_rp.pre_barriers.emplace_back(acquire_barrier);
					}

					continue;
				}

				bool crosses_rpass = (left->pass == nullptr || right.pass == nullptr || left->pass->render_pass_index != right.pass->render_pass_index);
				if (crosses_rpass) {
					if (left->pass) { // RenderPass ->
						auto& left_rp = impl->rpis[left->pass->render_pass_index];
						// if this is an attachment, we specify layout
						if (is_framebuffer_attachment(prev_use)) {
							assert(!left_rp.framebufferless);
							auto& rp_att = *contains_if(left_rp.attachments, [name](auto& att) { return att.name == name; });

							sync_bound_attachment_to_renderpass(rp_att, attachment_info);
							// we keep last use as finalLayout
							assert(prev_use.layout != ImageLayout::eUndefined && prev_use.layout != ImageLayout::ePreinitialized);
							rp_att.description.finalLayout = (VkImageLayout)prev_use.layout;

							// compute attachment store
							if (next_use.layout == ImageLayout::eUndefined) {
								rp_att.description.storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
							} else {
								rp_att.description.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
							}
						}
						// emit barrier for final resource state
						if (!right.pass && next_use.layout != ImageLayout::eUndefined &&
						    (prev_use.layout != next_use.layout || (is_write_access(prev_use) || is_write_access(next_use)))) { // different layouts, need to
							                                                                                                      // have dependency
							VkImageMemoryBarrier barrier{ .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER };
							ImageBarrier ib{};
							barrier.srcAccessMask = is_read_access(prev_use) ? 0 : (VkAccessFlags)prev_use.access;
							barrier.dstAccessMask = (VkAccessFlags)next_use.access;
							barrier.oldLayout = (VkImageLayout)prev_use.layout;
							barrier.newLayout = (VkImageLayout)next_use.layout;
							barrier.subresourceRange.aspectMask = (VkImageAspectFlags)aspect;
							barrier.subresourceRange.baseArrayLayer = subrange.base_layer;
							barrier.subresourceRange.baseMipLevel = subrange.base_level;
							barrier.subresourceRange.layerCount = subrange.layer_count;
							barrier.subresourceRange.levelCount = subrange.level_count;
							barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
							barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
							ib.src = prev_use.stages;
							ib.dst = next_use.stages;
							ib.barrier = barrier;
							ib.image = name;
							// attach this barrier to the end of subpass or end of renderpass
							if (left_rp.framebufferless) {
								left_rp.subpasses[left->pass->subpass].post_barriers.push_back(ib);
							} else {
								left_rp.post_barriers.push_back(ib);
							}
						}
					}

					if (right.pass) { // -> RenderPass
						auto& right_rp = impl->rpis[right.pass->render_pass_index];
						// if this is an attachment, we specify layout
						if (is_framebuffer_attachment(next_use)) {
							assert(!right_rp.framebufferless);
							auto& rp_att = *contains_if(right_rp.attachments, [name](auto& att) { return att.name == name; });

							sync_bound_attachment_to_renderpass(rp_att, attachment_info);
							//

							rp_att.description.initialLayout = (VkImageLayout)next_use.layout;
							assert(rp_att.description.initialLayout != (VkImageLayout)ImageLayout::eUndefined);

							// compute attachment load
							if (prev_use.layout == ImageLayout::eUndefined) {
								rp_att.description.loadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
							} else if (prev_use.layout == ImageLayout::ePreinitialized) {
								// preinit means clear
								rp_att.description.initialLayout = (VkImageLayout)ImageLayout::eUndefined;
								rp_att.description.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
							} else {
								rp_att.description.loadOp = VK_ATTACHMENT_LOAD_OP_LOAD;
							}
						}
						// we are keeping this weird logic until this is configurable
						// emit a barrier for now instead of an external subpass dep
						if (next_use.layout != prev_use.layout || (is_write_access(prev_use) || is_write_access(next_use))) { // different layouts, need to
							                                                                                                    // have dependency
							VkImageMemoryBarrier barrier{ .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER };
							barrier.srcAccessMask = is_read_access(prev_use) ? 0 : (VkAccessFlags)prev_use.access;
							barrier.dstAccessMask = (VkAccessFlags)next_use.access;
							barrier.oldLayout = prev_use.layout == ImageLayout::ePreinitialized ? (VkImageLayout)ImageLayout::eUndefined : (VkImageLayout)prev_use.layout;
							barrier.newLayout = (VkImageLayout)next_use.layout;
							barrier.subresourceRange.aspectMask = (VkImageAspectFlags)aspect;
							barrier.subresourceRange.baseArrayLayer = subrange.base_layer;
							barrier.subresourceRange.baseMipLevel = subrange.base_level;
							barrier.subresourceRange.layerCount = subrange.layer_count;
							barrier.subresourceRange.levelCount = subrange.level_count;
							barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
							barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;

							if (src_stages == PipelineStageFlags{}) {
								barrier.srcAccessMask = {};
							}
							if (dst_stages == PipelineStageFlags{}) {
								barrier.dstAccessMask = {};
							}
							ImageBarrier ib{ .image = name, .barrier = barrier, .src = src_stages, .dst = dst_stages };
							if (right_rp.framebufferless) {
								right_rp.subpasses[right.pass->subpass].pre_barriers.push_back(ib);
							} else {
								right_rp.pre_barriers.push_back(ib);
							}
						}
					}
				} else { // subpass-subpass link -> subpass - subpass dependency
					// WAW, WAR, RAW accesses need sync

					// if we merged the passes into a subpass, no sync is needed
					if (left->pass->subpass == right.pass->subpass)
						continue;
					if (is_framebuffer_attachment(prev_use) && (is_write_access(prev_use) || is_write_access(next_use))) {
						assert(left->pass->render_pass_index == right.pass->render_pass_index);
						auto& rp = impl->rpis[right.pass->render_pass_index];
						VkSubpassDependency sd{};
						sd.dstAccessMask = (VkAccessFlags)next_use.access;
						sd.dstStageMask = (VkPipelineStageFlags)next_use.stages;
						sd.dstSubpass = right.pass->subpass;
						sd.srcAccessMask = is_read_access(prev_use) ? 0 : (VkAccessFlags)prev_use.access;
						sd.srcStageMask = (VkPipelineStageFlags)prev_use.stages;
						sd.srcSubpass = left->pass->subpass;
						rp.rpci.subpass_dependencies.push_back(sd);
					}
					auto& left_rp = impl->rpis[left->pass->render_pass_index];
					auto& right_rp = impl->rpis[right.pass->render_pass_index];
					if (left_rp.framebufferless && (next_use.layout != prev_use.layout || is_write_access(prev_use) || is_write_access(next_use))) {
						// right layout == Undefined means the chain terminates, no
						// transition/barrier
						if (next_use.layout == ImageLayout::eUndefined)
							continue;
						VkImageMemoryBarrier barrier{ .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER };
						barrier.srcAccessMask = is_read_access(prev_use) ? 0 : (VkAccessFlags)prev_use.access;
						barrier.dstAccessMask = (VkAccessFlags)next_use.access;
						barrier.newLayout = (VkImageLayout)next_use.layout;
						barrier.oldLayout = (VkImageLayout)prev_use.layout;
						barrier.subresourceRange.aspectMask = (VkImageAspectFlags)aspect;
						barrier.subresourceRange.baseArrayLayer = subrange.base_layer;
						barrier.subresourceRange.baseMipLevel = subrange.base_level;
						barrier.subresourceRange.layerCount = subrange.layer_count;
						barrier.subresourceRange.levelCount = subrange.level_count;
						ImageBarrier ib{ .image = name, .barrier = barrier, .src = prev_use.stages, .dst = next_use.stages };
						right_rp.subpasses[right.pass->subpass].pre_barriers.push_back(ib);
					}
				}
			}
		}

		for (auto& [raw_name, buffer_info] : impl->bound_buffers) {
			auto name = impl->resolve_name(raw_name);
			auto chain_it = impl->use_chains.find(name);
			if (chain_it == impl->use_chains.end()) {
				// TODO: warning here, if turned on
				continue;
			}
			auto& chain = chain_it->second;
			chain.insert(chain.begin(),
			             UseRef{ {}, {}, vuk::eManual, vuk::eManual, buffer_info.initial, Resource::Type::eBuffer, Resource::Subrange{ .buffer = {} }, nullptr });
			chain.emplace_back(UseRef{ {}, {}, vuk::eManual, vuk::eManual, buffer_info.final, Resource::Type::eBuffer, Resource::Subrange{ .buffer = {} }, nullptr });

			for (size_t i = 0; i < chain.size() - 1; i++) {
				auto& left = chain[i];
				auto& right = chain[i + 1];

				left.use = left.high_level_access == Access::eManual ? left.use : to_use(left.high_level_access);
				right.use = right.high_level_access == Access::eManual ? right.use : to_use(right.high_level_access);

				DomainFlags left_domain = left.pass ? left.pass->domain : DomainFlagBits::eNone;
				DomainFlags right_domain = right.pass ? right.pass->domain : DomainFlagBits::eNone;

				// release - acquire pair
				if (left.original == eRelease && right.original == eAcquire) {
					// noop
				} else if (right.original == eRelease && (i + 1 == (chain.size() - 2))) {
					// release without acquire - must be last in chain
					auto& fut = *right.pass->pass.signal;
					fut.last_use = QueueResourceUse{
						left.original, left.use.stages, left.use.access, left.use.layout, (DomainFlagBits)(left_domain & DomainFlagBits::eQueueMask).m_mask
					};
					fut.get_result<Buffer>() = buffer_info.buffer; // TODO: when we have managed buffers, then this is too soon to attach
				}

				bool crosses_queue = (left_domain != DomainFlagBits::eNone && right_domain != DomainFlagBits::eNone &&
				                      (left_domain & DomainFlagBits::eQueueMask) != (right_domain & DomainFlagBits::eQueueMask));
				if (crosses_queue) {
					left.pass->is_waited_on = true;
					right.pass->waits.emplace_back((DomainFlagBits)(left_domain & DomainFlagBits::eQueueMask).m_mask, left.pass);

					continue;
				}

				bool crosses_rpass = (left.pass == nullptr || right.pass == nullptr || left.pass->render_pass_index != right.pass->render_pass_index);
				if (crosses_rpass) {
					if (left.pass && right.use.layout != ImageLayout::eUndefined && (is_write_access(left.use) || is_write_access(right.use))) { // RenderPass ->
						auto& left_rp = impl->rpis[left.pass->render_pass_index];

						VkMemoryBarrier barrier{ .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER };
						barrier.srcAccessMask = is_read_access(left.use) ? 0 : (VkAccessFlags)left.use.access;
						barrier.dstAccessMask = (VkAccessFlags)right.use.access;
						MemoryBarrier mb{ .barrier = barrier, .src = left.use.stages, .dst = right.use.stages };
						left_rp.subpasses[left.pass->subpass].post_mem_barriers.push_back(mb);
					}

					if (right.pass && left.use.layout != ImageLayout::eUndefined && (is_write_access(left.use) || is_write_access(right.use))) { // -> RenderPass
						auto& right_rp = impl->rpis[right.pass->render_pass_index];

						VkMemoryBarrier barrier{ .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER };
						barrier.srcAccessMask = is_read_access(left.use) ? 0 : (VkAccessFlags)left.use.access;
						barrier.dstAccessMask = (VkAccessFlags)right.use.access;
						MemoryBarrier mb{ .barrier = barrier, .src = left.use.stages, .dst = right.use.stages };
						if (mb.src == PipelineStageFlags{}) {
							mb.src = PipelineStageFlagBits::eTopOfPipe;
							mb.barrier.srcAccessMask = {};
						}
						right_rp.subpasses[right.pass->subpass].pre_mem_barriers.push_back(mb);
					}
				} else { // subpass-subpass link -> subpass - subpass dependency
					if (left.pass->subpass == right.pass->subpass)
						continue;
					auto& left_rp = impl->rpis[left.pass->render_pass_index];
					if (left_rp.framebufferless && (is_write_access(left.use) || is_write_access(right.use))) {
						VkMemoryBarrier barrier{ .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER };
						barrier.srcAccessMask = is_read_access(left.use) ? 0 : (VkAccessFlags)left.use.access;
						barrier.dstAccessMask = (VkAccessFlags)right.use.access;
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

		// assign passes to command buffers and batches (within a single queue)
		uint32_t batch_index = -1;
		DomainFlags current_domain = DomainFlagBits::eNone;
		bool needs_split = false;
		for (auto& rp : impl->rpis) {
			bool needs_split_next = false;
			for (auto& sp : rp.subpasses) {
				for (auto& passinfo : sp.passes) {
					if (passinfo->domain == DomainFlagBits::eNone) {
						continue;
					}

					if ((passinfo->domain & DomainFlagBits::eQueueMask) != (current_domain & DomainFlagBits::eQueueMask)) { // if we go into a new queue,
						                                                                                                      // reset batch index
						current_domain = passinfo->domain & DomainFlagBits::eQueueMask;
						batch_index = -1;
					}
					if (passinfo->waits.size() > 0) {
						needs_split = true;
					}
					if (passinfo->is_waited_on) {
						needs_split_next = true;
					}
				}
			}
			rp.command_buffer_index = 0; // we don't split command buffers within batches, for now
			rp.batch_index = (needs_split || (batch_index == -1)) ? ++batch_index : batch_index;
			needs_split = needs_split_next;
		}

		// build waits, now that we have fixed the batches
		for (auto& rp : impl->rpis) {
			for (auto& sp : rp.subpasses) {
				for (auto& passinfo : sp.passes) {
					for (auto& wait : passinfo->waits) {
						rp.waits.emplace_back(wait.first,
						                      impl->rpis[wait.second->render_pass_index].batch_index + 1); // 0 = means previous
					}
				}
			}
		}

		// we now have enough data to build VkRenderPasses and VkFramebuffers

		// compile attachments
		// we have to assign the proper attachments to proper slots
		// the order is given by the resource binding order

		size_t previous_rp = -1;
		uint32_t previous_sp = -1;
		for (auto& pass_p : impl->ordered_passes) {
			auto& pass = *pass_p;
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
				if (res.ia == Access::eColorResolveWrite) // resolve attachment are added when
				                                          // processing the color attachment
					continue;
				VkAttachmentReference attref{};

				auto name = impl->resolve_name(res.name);
				auto& chain = impl->use_chains.find(name)->second;
				auto cit = std::find_if(chain.begin(), chain.end(), [&](auto& useref) { return useref.pass == &pass; });
				assert(cit != chain.end());
				attref.layout = (VkImageLayout)cit->use.layout;
				attref.attachment = (uint32_t)std::distance(rp.attachments.begin(),
				                                            std::find_if(rp.attachments.begin(), rp.attachments.end(), [&](auto& att) { return name == att.name; }));

				if (attref.layout != (VkImageLayout)ImageLayout::eColorAttachmentOptimal) {
					if (attref.layout == (VkImageLayout)ImageLayout::eDepthStencilAttachmentOptimal) {
						ds_attrefs[subpass_index] = attref;
					}
				} else {
					VkAttachmentReference rref{};
					rref.attachment = VK_ATTACHMENT_UNUSED;
					if (auto it = pass.pass.resolves.find(res.name); it != pass.pass.resolves.end()) {
						// this a resolve src attachment
						// get the dst attachment
						auto dst_name = impl->resolve_name(it->second);
						rref.layout = (VkImageLayout)ImageLayout::eColorAttachmentOptimal; // the only possible
						                                                                   // layout for resolves
						rref.attachment = (uint32_t)std::distance(
						    rp.attachments.begin(), std::find_if(rp.attachments.begin(), rp.attachments.end(), [&](auto& att) { return dst_name == att.name; }));
						rp.attachments[rref.attachment].attachment.sample_count = Samples::e1; // resolve dst must be sample count = 1
						rp.attachments[rref.attachment].is_resolve_dst = true;
					}

					// we insert the new attachment at the end of the list for current
					// subpass index
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
				SubpassDescription sd;
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

		return { std::move(*this) };
	}

	MapProxy<Name, std::span<const UseRef>> RenderGraph::get_use_chains() {
		return &impl->use_chains;
	}

	MapProxy<Name, const AttachmentRPInfo&> RenderGraph::get_bound_attachments() {
		return &impl->bound_attachments;
	}

	MapProxy<Name, const BufferInfo&> RenderGraph::get_bound_buffers() {
		return &impl->bound_buffers;
	}

	ImageUsageFlags RenderGraph::compute_usage(std::span<const UseRef> chain) {
		ImageUsageFlags usage;
		for (const auto& c : chain) {
			if (c.high_level_access != Access::eManual) {
				switch (to_use(c.high_level_access).layout) {
				case ImageLayout::eDepthStencilAttachmentOptimal:
					usage |= ImageUsageFlagBits::eDepthStencilAttachment;
					break;
				case ImageLayout::eShaderReadOnlyOptimal: // TODO: more complex analysis
					usage |= ImageUsageFlagBits::eSampled;
					break;
				case ImageLayout::eColorAttachmentOptimal:
					usage |= ImageUsageFlagBits::eColorAttachment;
					break;
				case ImageLayout::eTransferSrcOptimal:
					usage |= ImageUsageFlagBits::eTransferRead;
					break;
				case ImageLayout::eTransferDstOptimal:
					usage |= ImageUsageFlagBits::eTransferWrite;
					break;
				default:
					break;
				}
				// TODO: this isn't conservative enough, we need more information
				if (c.use.layout == ImageLayout::eGeneral) {
					if (c.use.stages &
					    (PipelineStageFlagBits::eComputeShader | PipelineStageFlagBits::eVertexShader | PipelineStageFlagBits::eTessellationControlShader |
					     PipelineStageFlagBits::eTessellationEvaluationShader | PipelineStageFlagBits::eGeometryShader | PipelineStageFlagBits::eFragmentShader)) {
						usage |= ImageUsageFlagBits::eStorage;
					}
				}
			}
		}

		return usage;
	}
} // namespace vuk
