#include "vuk/RenderGraph.hpp"
#include "RenderGraphImpl.hpp"
#include "RenderGraphUtil.hpp"
#include "vuk/CommandBuffer.hpp"
#include "vuk/Context.hpp"
#include "vuk/Exception.hpp"
#include "vuk/Future.hpp"

#include <set>
#include <sstream>
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

	RenderGraph::RenderGraph(Name name) : name(name), impl(new RGImpl) {}

	RenderGraph::RenderGraph(RenderGraph&& o) noexcept : name(o.name), impl(std::exchange(o.impl, nullptr)) {}
	RenderGraph& RenderGraph::operator=(RenderGraph&& o) noexcept {
		impl = std::exchange(o.impl, nullptr);
		name = o.name;
		return *this;
	}

	RenderGraph::~RenderGraph() {
		delete impl;
	}

	void RenderGraph::add_pass(Pass p) {
		for (auto& r : p.resources) {
			if (r.is_create && r.type == Resource::Type::eImage) {
				attach_image(r.name, r.ici, vuk::Access::eNone, vuk::Access::eNone);
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
			att.name = joiner.append(name);
			impl->bound_attachments.emplace(joiner.append(name), std::move(att));
		}
		for (auto& [name, buf] : other.impl->bound_buffers) {
			buf.name = joiner.append(name);
			impl->bound_buffers.emplace(joiner.append(name), std::move(buf));
		}

		for (auto& [name, iainf] : other.impl->ia_inference_rules) {
			iainf.prefix = joiner.append(iainf.prefix);
			impl->ia_inference_rules.emplace(joiner.append(name), std::move(iainf));
		}

		for (auto& [new_name, old_name] : other.impl->aliases) {
			new_name = joiner.append(new_name);
			impl->aliases.emplace(new_name, joiner.append(old_name));
		}

		for (auto& [name, v] : other.impl->acquires) {
			impl->acquires.emplace(joiner.append(name), std::move(v));
		}

		for (auto& [name, v] : other.impl->releases) {
			impl->releases.emplace(joiner.append(name), std::move(v));
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
	void RGImpl::build_io() {
		poisoned_names.clear();
		for (auto& pif : passes) {
			pif.input_names.clear();
			pif.output_names.clear();
			pif.write_input_names.clear();
			pif.bloom_write_inputs = {};
			pif.bloom_outputs = {};
			pif.bloom_resolved_inputs = {};

			for (Resource& res : pif.pass.resources) {
				Name in_name;
				Name out_name;
				if (res.type == Resource::Type::eImage && res.subrange.image != Resource::Subrange::Image{}) {
					in_name = res.subrange.image.combine_name(resolve_alias(res.name));
					out_name = res.out_name.is_invalid() ? Name{} : res.subrange.image.combine_name(resolve_alias(res.out_name));
				} else {
					in_name = resolve_alias(res.name);
					out_name = resolve_alias(res.out_name);
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
					assert(!poisoned_names.contains(in_name)); // we have poisoned this name because a write has already consumed it
					pif.bloom_write_inputs |= hashed_in_name;
					pif.write_input_names.emplace_back(in_name);
					poisoned_names.emplace(in_name);
				}

				// for image subranges, we additionally add a dependency on the diverged original resource
				// this resource is created by the diverged pass and consumed by the converge pass, thereby constraining all the passes who refer to these
				if (res.type == Resource::Type::eImage && res.subrange.image != Resource::Subrange::Image{}) {
					Name dep = res.name.append("d");
					auto hashed_name = ::hash::fnv1a::hash(dep.to_sv().data(), dep.to_sv().size(), hash::fnv1a::default_offset_basis);

					pif.input_names.emplace_back(dep);
					pif.bloom_resolved_inputs |= hashed_name;
				}
			}
		}
	}

	void RenderGraph::schedule_intra_queue(std::span<PassInfo> passes, const RenderGraphCompileOptions& compile_options) {
		// sort passes if requested
		if (passes.size() > 1 && compile_options.reorder_passes) {
			topological_sort(passes.begin(), passes.end(), [this](const auto& p1, const auto& p2) {
				if (&p1 == &p2) {
					return false;
				}
				// p2 uses an input of p1 -> p2 after p1
				if ((p1.bloom_outputs & p2.bloom_resolved_inputs) != 0) {
					for (auto& o : p1.output_names) {
						for (auto& i : p2.input_names) {
							if (o == impl->resolve_alias(i)) {
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
							if (impl->resolve_alias(o) == impl->resolve_alias(i)) {
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

	std::string RenderGraph::dump_graph() {
		std::stringstream ss;
		ss << "digraph vuk {\n";
		for (auto i = 0; i < impl->passes.size(); i++) {
			for (auto j = 0; j < impl->passes.size(); j++) {
				if (i == j)
					continue;
				auto& p1 = impl->passes[i];
				auto& p2 = impl->passes[j];
				for (auto& o : p1.output_names) {
					for (auto& i : p2.input_names) {
						if (o == impl->resolve_alias(i)) {
							ss << "\"" << p1.pass.name.c_str() << "\" -> \"" << p2.pass.name.c_str() << "\" [label=\"" << impl->resolve_alias(i).c_str() << "\"];\n";
							// p2 is ordered after p1
						}
					}
				}
				for (auto& o : p1.input_names) {
					for (auto& i : p2.write_input_names) {
						if (impl->resolve_alias(o) == impl->resolve_alias(i)) {
							ss << "\"" << p1.pass.name.c_str() << "\" -> \"" << p2.pass.name.c_str() << "\" [label=\"" << impl->resolve_alias(i).c_str() << "\"];\n";
							// p2 is ordered after p1
						}
					}
				}
			}
		}
		ss << "}\n";
		return ss.str();
	}

	void RenderGraph::inline_subgraphs() {
		for (auto& [sg_ptr, sg_info] : impl->subgraphs) {
			if (sg_info.count > 0) {
				assert(sg_ptr->impl && "Don't support this yet");
				sg_ptr->inline_subgraphs();
				Name sg_name = sg_ptr->name;
				if (auto& counter = ++impl->sg_name_counter[sg_name]; counter > 1) {
					sg_name = sg_name.append(Name(std::string("_") + std::to_string(counter - 1)));
				}
				append(sg_name, std::move(*sg_ptr));
				for (auto& [name_in_parent, name_in_sg] : sg_info.exported_names) {
					add_alias(name_in_parent, sg_name.append("::").append(name_in_sg));
				}
			}
		}
		impl->subgraphs.clear();
	}

	void RenderGraph::compile(const RenderGraphCompileOptions& compile_options) {
		// inline all the subgraphs into us
		inline_subgraphs();

		// find which reads are graph inputs (not produced by any pass) & outputs
		// (not consumed by any pass)
		impl->build_io();

		// run global pass ordering - once we split per-queue we don't see enough
		// inputs to order within a queue
		schedule_intra_queue(impl->passes, compile_options);

		// gather name alias info now - once we partition, we might encounter unresolved aliases
		robin_hood::unordered_flat_map<Name, Name> name_map;
		name_map.insert(impl->aliases.begin(), impl->aliases.end());

		for (auto& passinfo : impl->passes) {
			for (auto& res : passinfo.pass.resources) {
				// for read or write, we add source to use chain
				if (!res.out_name.is_invalid()) {
					name_map.emplace(res.out_name, res.name);
				}
			}
		}

		// populate resource name -> attachment map
		for (auto& [k, v] : name_map) {
			auto it = name_map.find(v);
			Name res = v;
			while (it != name_map.end()) {
				res = it->second;
				it = name_map.find(res);
			}
			impl->assigned_names.emplace(k, res);
		}

		// for now, just use what the passes requested as domain
		for (auto& p : impl->passes) {
			p.domain = p.pass.execute_on;
		}

		decltype(impl->acquires) res_acqs;
		for (auto& [k, v] : impl->acquires) {
			res_acqs.emplace(impl->resolve_name(k), std::move(v));
		}

		for (auto& [name, att] : impl->bound_attachments) {
			att.use_chains.clear();
		}
		for (auto& [name, att] : impl->bound_buffers) {
			att.use_chains.clear();
		}

		// use chains pass
		impl->use_chains.clear();
		std::unordered_multimap<Name, Name> diverged_chains;
		for (PassInfo& passinfo : impl->passes) {
			for (Resource& res : passinfo.pass.resources) {
				Name resolved_name;
				Name undiverged_name;
				bool diverged_subchain = false;
				if (res.type == Resource::Type::eImage && res.subrange.image != Resource::Subrange::Image{}) {
					undiverged_name = impl->resolve_name(res.name);
					resolved_name = res.subrange.image.combine_name(undiverged_name);
					// assign diverged resources to attachments
					impl->assigned_names.emplace(resolved_name, undiverged_name);
					diverged_subchain = true;
				} else {
					undiverged_name = resolved_name = impl->resolve_name(res.name);
				}

				bool skip = false;
				// for read or write, we add source to use chain
				auto it = impl->use_chains.find(resolved_name);
				if (it == impl->use_chains.end()) {
					it = impl->use_chains.emplace(resolved_name, std::vector<UseRef, short_alloc<UseRef, 64>>{ short_alloc<UseRef, 64>{ *impl->arena_ } }).first;
					auto& chain = it->second;

					// we attach this use chain to the appropriate attachment
					if (res.type == Resource::Type::eImage) {
						auto undiv_att_it = impl->bound_attachments.find(undiverged_name);
						assert(undiv_att_it != impl->bound_attachments.end());
						undiv_att_it->second.use_chains.emplace_back(&chain);
					} else {
						auto undiv_att_it = impl->bound_buffers.find(undiverged_name);
						assert(undiv_att_it != impl->bound_buffers.end());
						undiv_att_it->second.use_chains.emplace_back(&chain);
					}

					// figure what is appropriate chain header
					// 1. if this is direct attachment use, we want to put here initial_use
					bool is_direct_attachment_use = false;
					QueueResourceUse initial;
					if (res.type == Resource::Type::eImage) {
						auto att_it = impl->bound_attachments.find(resolved_name);
						is_direct_attachment_use = att_it != impl->bound_attachments.end();
						if (is_direct_attachment_use) {
							initial = att_it->second.initial;
						}
					} else {
						auto att_it = impl->bound_buffers.find(resolved_name);
						is_direct_attachment_use = att_it != impl->bound_buffers.end();
						if (is_direct_attachment_use) {
							initial = att_it->second.initial;
						}
					}
					// 2. if this is an acquire from a future, we want to put the acquire here
					// if this attachment is acquired, we will perform the acquire as initial use
					// otherwise we take initial use from the attachment
					// insert initial usage if we need to synchronize against it
					if (auto it = res_acqs.find(resolved_name); it != res_acqs.end()) {
						RGImpl::Acquire* acquire = &it->second;
						chain.emplace_back(UseRef{ {}, {}, vuk::eManual, vuk::eManual, acquire->src_use, res.type, {}, nullptr });
						chain.emplace_back(UseRef{ res.name, res.out_name, res.ia, res.ia, {}, res.type, res.subrange, &passinfo });
						chain.back().use.domain = passinfo.domain;
						// make the actually executed first pass wait on the future
						chain.back().pass->absolute_waits.emplace_back(acquire->initial_domain, acquire->initial_visibility);
						skip = true; // we added to the chain here already
					} else if (is_direct_attachment_use) {
						chain.insert(chain.begin(), UseRef{ {}, {}, vuk::eManual, vuk::eManual, initial, res.type, {}, nullptr });
					} else if (diverged_subchain) { // 3. if this is a diverged subchain, we want to put the last undiverged use here (or nothing, if this was first use)
						auto undiv_name = impl->resolve_name(res.name);
						auto& undiverged_chain = impl->use_chains.at(undiv_name);
						if (undiverged_chain.size() > 0) {
							chain.emplace_back(undiverged_chain.back());
						}
						diverged_chains.emplace(undiv_name, resolved_name);
					}
				}
				auto& chain = it->second;

				// we don't want to put the convergence onto the divergence use chains
				// what we want to put is the next use after convergence
				// if there is no next use after convergence, we'll attempt to pick up a last use from final layout or release, or error
				if (!chain.empty() && chain.back().high_level_access == Access::eConverge) {
					auto range = diverged_chains.equal_range(resolved_name);
					for (auto it = range.first; it != range.second; ++it) {
						auto& subchain = impl->use_chains.at(it->second);
						subchain.emplace_back(UseRef{ res.name, res.out_name, res.ia, res.ia, {}, res.type, subchain.back().subrange, &passinfo });
					}
					// TODO: we need to handle layer/level tails here by inspecting which diverged chains live
				}

				if (res.ia != Access::eConsume && !skip) { // eConsume is not a use
					chain.emplace_back(UseRef{ res.name, res.out_name, res.ia, res.ia, {}, res.type, res.subrange, &passinfo });
					chain.back().use.domain = passinfo.domain;
				}
			}
		}

		// queue inference pass
		for (auto& [name, chain] : impl->use_chains) {
			DomainFlags last_domain = DomainFlagBits::eDevice;

			// forward inference
			for (uint64_t i = 0; i < chain.size(); i++) {
				auto& use_ref = chain[i];
				auto domain = use_ref.use.domain;
				if (domain != last_domain && domain != DomainFlagBits::eDevice && domain != DomainFlagBits::eAny) {
					last_domain = use_ref.use.domain;
				}
				if ((last_domain != DomainFlagBits::eDevice && last_domain != DomainFlagBits::eAny) &&
				    (domain == DomainFlagBits::eDevice || domain == DomainFlagBits::eAny)) {
					if (use_ref.pass) {
						use_ref.pass->domain = last_domain;
					}
					use_ref.use.domain = last_domain;
				}
			}
			last_domain = DomainFlagBits::eDevice;
			// backward inference
			for (int64_t i = chain.size() - 1; i >= 0; i--) {
				auto& use_ref = chain[i];
				auto domain = use_ref.use.domain;

				if (domain != last_domain && domain != DomainFlagBits::eDevice && domain != DomainFlagBits::eAny) {
					last_domain = use_ref.use.domain;
				}
				if ((last_domain != DomainFlagBits::eDevice && last_domain != DomainFlagBits::eAny) &&
				    (domain == DomainFlagBits::eDevice || domain == DomainFlagBits::eAny)) {
					if (use_ref.pass) {
						use_ref.pass->domain = last_domain;
					}
					use_ref.use.domain = last_domain;
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

		for (auto& [name, chain] : impl->use_chains) {
			for (uint64_t i = 0; i < chain.size(); i++) {
				auto& use_ref = chain[i];
				// inference failure fixup
				if (use_ref.use.domain == DomainFlagBits::eDevice || use_ref.use.domain == DomainFlagBits::eAny) {
					use_ref.use.domain = DomainFlagBits::eGraphicsOnGraphics;
				}
			}
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
				auto name = impl->resolve_name(att.name);
				rpi.attachments.push_back(AttachmentRPInfo{ &impl->bound_attachments.at(name) });
				rpi.layer_count = att.subrange.image.layer_count;
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

	void RenderGraph::clear_image(Name image_name, Name image_name_out, Clear clear_value, Resource::Subrange::Image subrange) {
		std::vector<std::byte> args(sizeof(Clear));
		std::memcpy(args.data(), &clear_value, sizeof(Clear));
		Resource res{ image_name, Resource::Type::eImage, eClear, image_name_out };
		res.subrange.image = subrange;
		add_pass({ .name = image_name.append("_CLEAR"),
		           .resources = { std::move(res) },
		           .execute = [image_name, clear_value](CommandBuffer& cbuf) { cbuf.clear_image(image_name, clear_value); },
		           .arguments = std::move(args),
		           .type = Pass::Type::eClear });
	}

	void RenderGraph::attach_swapchain(Name name, SwapchainRef swp) {
		AttachmentInfo attachment_info;
		attachment_info.name = name;
		attachment_info.attachment.extent = Dimension3D::absolute(swp->extent);
		// directly presented
		attachment_info.attachment.format = swp->format;
		attachment_info.attachment.sample_count = Samples::e1;
		attachment_info.attachment.base_layer = 0;
		attachment_info.attachment.base_level = 0;
		attachment_info.attachment.layer_count = 1;
		attachment_info.attachment.level_count = 1;

		attachment_info.type = AttachmentInfo::Type::eSwapchain;
		attachment_info.swapchain = swp;

		QueueResourceUse& initial = attachment_info.initial;
		QueueResourceUse& final = attachment_info.final;
		// for WSI, we want to wait for colourattachmentoutput
		// we don't care about any writes, we will clear
		initial.access = AccessFlags{};
		initial.stages = PipelineStageFlagBits::eColorAttachmentOutput;
		// discard
		initial.layout = ImageLayout::eUndefined;
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

	void RenderGraph::attach_buffer(Name name, Buffer buf, Access initial, Access final) {
		BufferInfo buf_info{ .name = name, .initial = to_use(initial, DomainFlagBits::eAny), .final = to_use(final, DomainFlagBits::eAny), .buffer = buf };
		impl->bound_buffers.emplace(name, buf_info);
	}

	void RenderGraph::attach_image(Name name, ImageAttachment att, Access initial_acc, Access final_acc) {
		AttachmentInfo attachment_info;
		attachment_info.name = name;
		attachment_info.attachment = att;
		if (att.has_concrete_image() && att.has_concrete_image_view()) {
			attachment_info.type = AttachmentInfo::Type::eExternal;
		} else {
			attachment_info.type = AttachmentInfo::Type::eInternal;
		}
		attachment_info.attachment.format = att.format;

		QueueResourceUse& initial = attachment_info.initial;
		QueueResourceUse& final = attachment_info.final;
		initial = to_use(initial_acc, DomainFlagBits::eAny);
		final = to_use(final_acc, DomainFlagBits::eAny);
		impl->bound_attachments.emplace(name, attachment_info);
	}

	void RenderGraph::attach_and_clear_image(Name name, ImageAttachment att, Clear clear_value, Access initial_acc, Access final_acc) {
		Name tmp_name = name.append(get_temporary_name());
		attach_image(tmp_name, att, initial_acc, final_acc);
		clear_image(tmp_name, name, clear_value);
	}

	void RenderGraph::attach_in(Name name, Future fimg) {
		if (fimg.get_status() == FutureBase::Status::eSubmitted || fimg.get_status() == FutureBase::Status::eHostAvailable) {
			if (fimg.is_image()) {
				auto att = fimg.get_result<ImageAttachment>();
				AttachmentInfo attachment_info;
				attachment_info.name = name;
				attachment_info.attachment = att;

				attachment_info.type = AttachmentInfo::Type::eExternal;

				attachment_info.initial = fimg.control->last_use;
				attachment_info.final = to_use(vuk::Access::eNone, DomainFlagBits::eAny);

				impl->bound_attachments.emplace(name, attachment_info);
			} else {
				BufferInfo buf_info{
					.name = name, .initial = fimg.control->last_use, .final = to_use(vuk::Access::eNone, DomainFlagBits::eAny), .buffer = fimg.get_result<Buffer>()
				};
				impl->bound_buffers.emplace(name, buf_info);
			}
			impl->acquires.emplace(name, RGImpl::Acquire{ fimg.control->last_use, fimg.control->initial_domain, fimg.control->initial_visibility });
			impl->imported_names.emplace(name);
		} else if (fimg.get_status() == FutureBase::Status::eInitial) {
			Name sg_name = fimg.rg->name;
			// an unsubmitted RG is being attached, we remove the release from that RG, and we allow the name to be found in us
			if (fimg.rg->impl) {
				fimg.rg->impl->releases.erase(fimg.get_bound_name());
				impl->subgraphs[fimg.rg].count++;
			} else {
				impl->releases.erase(sg_name.append("::").append(fimg.get_bound_name()));
			}
			impl->subgraphs[fimg.rg].exported_names.emplace(name, fimg.get_bound_name());
			impl->imported_names.emplace(name);
			fimg.rg.reset();
		} else {
			assert(0);
		}
	}

	void RenderGraph::attach_in(std::span<Future> futures) {
		for (auto& f : futures) {
			auto name = f.get_bound_name();
			attach_in(name, std::move(f));
		}
	}

	void RenderGraph::inference_rule(Name target, std::function<void(const struct InferenceContext&, ImageAttachment&)> rule) {
		impl->ia_inference_rules[target].prefix = "";
		impl->ia_inference_rules[target].rules.push_back(std::move(rule));
	}

	robin_hood::unordered_flat_set<Name> RGImpl::get_available_resources() {
		build_io();
		// seed the available names with the names we imported from subgraphs
		robin_hood::unordered_flat_set<Name> outputs = imported_names;
		for (auto& pif : passes) {
			for (auto& in : pif.input_names) {
				outputs.erase(in);
			}
			for (auto& in : pif.write_input_names) {
				outputs.erase(in);
			}
			for (auto& in : pif.output_names) {
				outputs.emplace(in);
			}
		}
		return outputs;
	}

	std::vector<Future> RenderGraph::split() {
		robin_hood::unordered_flat_set<Name> outputs = impl->get_available_resources();
		std::vector<Future> futures;
		for (auto& elem : outputs) {
			futures.emplace_back(this->shared_from_this(), elem);
		}
		return futures;
	}

	void RenderGraph::attach_out(Name name, Future& fimg, DomainFlags dst_domain) {
		impl->releases.emplace(name, RGImpl::Release{ to_use(Access::eNone, dst_domain), fimg.control.get() });
	}

	void RenderGraph::detach_out(Name name, Future& fimg) {
		for (auto it = impl->releases.begin(); it != impl->releases.end(); ++it) {
			if (it->first == name && it->second.signal == fimg.control.get()) {
				impl->releases.erase(it);
				return;
			}
		}
	}

	Name RenderGraph::get_temporary_name() {
		return impl->temporary_name.append(Name(std::to_string(impl->temporary_name_counter++)));
	}

	IARule same_extent_as(Name n) {
		return [=](const InferenceContext& ctx, ImageAttachment& ia) {
			ia.extent = ctx.get_image_attachment(n).extent;
		};
	}

	IARule same_format_as(Name n) {
		return [=](const InferenceContext& ctx, ImageAttachment& ia) {
			ia.format = ctx.get_image_attachment(n).format;
		};
	}

	IARule same_shape_as(Name n) {
		return [=](const InferenceContext& ctx, ImageAttachment& ia) {
			auto& src = ctx.get_image_attachment(n);
			if (src.base_layer != VK_REMAINING_ARRAY_LAYERS)
				ia.base_layer = src.base_layer;
			if (src.layer_count != VK_REMAINING_ARRAY_LAYERS)
				ia.layer_count = src.layer_count;
			if (src.base_level != VK_REMAINING_MIP_LEVELS)
				ia.base_level = src.base_level;
			if (src.level_count != VK_REMAINING_MIP_LEVELS)
				ia.level_count = src.level_count;
			if (src.extent.extent.width != 0 && src.extent.extent.height != 0)
				ia.extent = src.extent;
		};
	}

	IARule similar_to(Name n) {
		return [=](const InferenceContext& ctx, ImageAttachment& ia) {
			auto& src = ctx.get_image_attachment(n);
			if (src.base_layer != VK_REMAINING_ARRAY_LAYERS)
				ia.base_layer = src.base_layer;
			if (src.layer_count != VK_REMAINING_ARRAY_LAYERS)
				ia.layer_count = src.layer_count;
			if (src.base_level != VK_REMAINING_MIP_LEVELS)
				ia.base_level = src.base_level;
			if (src.level_count != VK_REMAINING_MIP_LEVELS)
				ia.level_count = src.level_count;
			if (src.extent.extent.width != 0 && src.extent.extent.height != 0)
				ia.extent = src.extent;
			if (src.format != Format::eUndefined)
				ia.format = src.format;
			if (src.sample_count != Samples::eInfer)
				ia.sample_count = src.sample_count;
		};
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

	ExecutableRenderGraph RenderGraph::link(const RenderGraphCompileOptions& compile_options) && {
		compile(compile_options);

		// at this point the graph is built, we know of all the resources and
		// everything should have been attached perform checking if this indeed the
		// case
		validate();

		// handle clears
		// if we are going into a Clear, see if the subsequent use is a framebuffer use
		// in this case we propagate the clear onto the renderpass of the next use, and we drop the clear pass
		// if we can't do this (subsequent use not fb use), then we will emit CmdClear when recording
		for (auto& [raw_name, attachment_info] : impl->bound_attachments) {
			auto name = impl->resolve_name(raw_name);
			auto chain_it = impl->use_chains.find(name);
			if (chain_it == impl->use_chains.end()) {
				// TODO: warning here, if turned on
				continue;
			}
			auto& chain = chain_it->second;
			for (int i = 0; i < (int)chain.size() - 1; i++) {
				auto& left = chain[i];
				auto& right = chain[i + 1];
				if (left.original == eClear && left.pass->pass.type == Pass::Type::eClear) {
					// next use is as fb attachment
					if ((i < chain.size() - 1) && is_framebuffer_attachment(to_use(right.original, right.use.domain))) {
						auto& next_rpi = impl->rpis[right.pass->render_pass_index];
						for (auto& rpi_att : next_rpi.attachments) {
							if (rpi_att.attachment_info == &attachment_info) {
								assert(left.pass->pass.arguments.size() == sizeof(Clear));
								rpi_att.clear_value = Clear{};
								std::memcpy(&rpi_att.clear_value, left.pass->pass.arguments.data(), sizeof(Clear));
								break;
							}
						}
						auto& sp = impl->rpis[left.pass->render_pass_index].subpasses[left.pass->subpass];
						sp.passes.erase(std::find(sp.passes.begin(), sp.passes.end(), left.pass));
						impl->ordered_passes.erase(std::find(impl->ordered_passes.begin(), impl->ordered_passes.end(), left.pass));
						left.pass->render_pass_index = static_cast<size_t>(-1);
						chain.erase(chain.begin() + i); // remove the clear from the use chain
					}
				}
			}
		}

		decltype(impl->releases) res_rels;
		for (auto& [k, v] : impl->releases) {
			res_rels.emplace(impl->resolve_name(k), std::move(v));
		}

		for (auto& [raw_name, chain] : impl->use_chains) {
			auto name = impl->resolve_name(raw_name);
			auto att_it = impl->bound_attachments.find(name);
			if (att_it == impl->bound_attachments.end()) {
				continue;
			}

			auto& attachment_info = att_it->second;

			RGImpl::Release* release = nullptr;
			if (auto it = res_rels.find(name); it != res_rels.end()) {
				release = &it->second;
				if (attachment_info.final.layout != ImageLayout::eUndefined) {
					chain.insert(chain.end(), UseRef{ {}, {}, vuk::eManual, vuk::eManual, attachment_info.final, Resource::Type::eImage, {}, nullptr });
					// propagate domain
					auto& last_use = chain[chain.size() - 1].use;
					if (last_use.domain == DomainFlagBits::eAny || last_use.domain == DomainFlagBits::eDevice) {
						last_use.domain = chain[chain.size() - 2].use.domain;
					}
				} else if (chain.back().high_level_access == Access::eConverge) {
					assert(it->second.dst_use.layout != ImageLayout::eUndefined); // convergence into no use = disallowed
				}

				chain.emplace_back(UseRef{ {}, {}, vuk::eManual, vuk::eManual, it->second.dst_use, Resource::Type::eImage, {}, nullptr });
			} else {
				chain.emplace_back(UseRef{ {}, {}, vuk::eManual, vuk::eManual, attachment_info.final, Resource::Type::eImage, {}, nullptr });
			}

			{
				// if the last use did not specify a domain, we'll take domain from previous use
				// previous use must always have a concrete domain
				auto& last_use = chain[chain.size() - 1].use;
				if (last_use.domain == DomainFlagBits::eAny || last_use.domain == DomainFlagBits::eDevice) {
					last_use.domain = chain[chain.size() - 2].use.domain;
				}
			}

			ImageAspectFlags aspect = format_to_aspect(attachment_info.attachment.format);

			QueueResourceUse original;
			PassInfo* last_executing_pass = nullptr;
			for (size_t i = 0; i < chain.size() - 1; i++) {
				auto* left = &chain[i];
				auto& right = chain[i + 1];

				if (left->pass) {
					last_executing_pass = left->pass;
				}

				if (right.high_level_access == Access::eConverge || left->high_level_access == Access::eConverge) {
					continue;
				}

				QueueResourceUse prev_use;
				prev_use = left->use = left->high_level_access == Access::eManual ? left->use : to_use(left->high_level_access, left->use.domain);
				QueueResourceUse next_use;
				next_use = right.use = right.high_level_access == Access::eManual ? right.use : to_use(right.high_level_access, right.use.domain);
				auto subrange = right.subrange.image;

				auto& src_stages = prev_use.stages;
				auto& dst_stages = next_use.stages;

				scope_to_domain(src_stages, left->use.domain & DomainFlagBits::eQueueMask);
				scope_to_domain(dst_stages, right.use.domain & DomainFlagBits::eQueueMask);

				bool crosses_queue = (left->use.domain != DomainFlagBits::eNone && right.use.domain != DomainFlagBits::eNone &&
				                      (left->use.domain & DomainFlagBits::eQueueMask) != (right.use.domain & DomainFlagBits::eQueueMask));

				// compute image barrier for this access -> access
				VkImageMemoryBarrier barrier{ .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER };
				barrier.srcAccessMask = is_read_access(prev_use) ? 0 : (VkAccessFlags)prev_use.access;
				barrier.dstAccessMask = (VkAccessFlags)next_use.access;
				barrier.oldLayout = (VkImageLayout)prev_use.layout;
				barrier.newLayout = (VkImageLayout)next_use.layout;
				barrier.subresourceRange.aspectMask = (VkImageAspectFlags)aspect;
				barrier.subresourceRange.baseArrayLayer = subrange.base_layer;
				barrier.subresourceRange.baseMipLevel = subrange.base_level;
				barrier.subresourceRange.layerCount = subrange.layer_count;
				barrier.subresourceRange.levelCount = subrange.level_count;
				assert(left->use.domain.m_mask != 0);
				assert(right.use.domain.m_mask != 0);
				barrier.srcQueueFamilyIndex = static_cast<uint32_t>(left->use.domain.m_mask);
				barrier.dstQueueFamilyIndex = static_cast<uint32_t>(right.use.domain.m_mask);

				if (src_stages == PipelineStageFlags{}) {
					barrier.srcAccessMask = {};
				}
				if (dst_stages == PipelineStageFlags{}) {
					barrier.dstAccessMask = {};
				}
				ImageBarrier ib{ .image = name, .barrier = barrier, .src = src_stages, .dst = dst_stages };

				// handle signaling & waiting
				if (right.pass && right.pass->pass.signal) {
					auto& fut = *right.pass->pass.signal;
					fut.last_use = QueueResourceUse{ src_stages, prev_use.access, prev_use.layout, (left->use.domain & DomainFlagBits::eQueueMask) };
					attachment_info.attached_future = &fut;
				}

				// the use chain is ending and we have to release this resource
				// we ignore the last element in the chain if it doesn't specify a valid use
				if (i + 2 == chain.size() && release) {
					assert(last_executing_pass != nullptr);
					auto& fut = *release->signal;
					fut.last_use = QueueResourceUse{ src_stages, prev_use.access, prev_use.layout, (left->use.domain & DomainFlagBits::eQueueMask) };
					attachment_info.attached_future = &fut;
					last_executing_pass->future_signals.push_back(&fut); // last executing pass gets to signal this future too
				}

				if (crosses_queue && left->pass && right.pass) {
					left->pass->is_waited_on = true;
					right.pass->waits.emplace_back((DomainFlagBits)(left->use.domain & DomainFlagBits::eQueueMask).m_mask, left->pass);
				}

				bool crosses_rpass = (left->pass == nullptr || right.pass == nullptr || left->pass->render_pass_index != right.pass->render_pass_index);
				if (crosses_rpass) {
					if (left->pass) { // RenderPass ->
						auto& left_rp = impl->rpis[left->pass->render_pass_index];
						// if this is an attachment, we specify layout
						if (is_framebuffer_attachment(prev_use)) {
							assert(!left_rp.framebufferless);
							auto& rp_att = *contains_if(left_rp.attachments, [&](auto& att) { return att.attachment_info == &attachment_info; });

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
						if (crosses_queue ||
						    (!right.pass && next_use.layout != ImageLayout::eUndefined &&
						     (prev_use.layout != next_use.layout || (is_write_access(prev_use) || is_write_access(next_use))))) { // different layouts, need to
							                                                                                                        // have dependency
							// attach this barrier to the end of subpass or end of renderpass
							auto ib_l = ib;
							scope_to_domain(ib_l.src, left->use.domain & DomainFlagBits::eQueueMask);
							scope_to_domain(ib_l.dst, left->use.domain & DomainFlagBits::eQueueMask);
							if (left_rp.framebufferless) {
								left_rp.subpasses[left->pass->subpass].post_barriers.push_back(ib_l);
							} else {
								left_rp.post_barriers.push_back(ib_l);
							}
						}
					}

					if (right.pass) { // -> RenderPass
						auto& right_rp = impl->rpis[right.pass->render_pass_index];
						// if this is an attachment, we specify layout
						if (is_framebuffer_attachment(next_use)) {
							assert(!right_rp.framebufferless);
							auto& rp_att = *contains_if(right_rp.attachments, [&](auto& att) { return att.attachment_info == &attachment_info; });

							rp_att.description.initialLayout = (VkImageLayout)next_use.layout;
							assert(rp_att.description.initialLayout != (VkImageLayout)ImageLayout::eUndefined);

							// compute attachment load
							if (rp_att.clear_value) {
								rp_att.description.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
							} else if (prev_use.layout == ImageLayout::eUndefined) {
								rp_att.description.loadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
							} else {
								rp_att.description.loadOp = VK_ATTACHMENT_LOAD_OP_LOAD;
							}
						}
						// we are keeping this weird logic until this is configurable
						// emit a barrier for now instead of an external subpass dep
						if (crosses_queue || next_use.layout != prev_use.layout || (is_write_access(prev_use) || is_write_access(next_use))) { // different layouts, need
							                                                                                                                     // to have dependency
							scope_to_domain(ib.src, right.use.domain & DomainFlagBits::eQueueMask);
							scope_to_domain(ib.dst, right.use.domain & DomainFlagBits::eQueueMask);
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
						sd.dstStageMask = (VkPipelineStageFlags)dst_stages;
						sd.dstSubpass = right.pass->subpass;
						sd.srcAccessMask = is_read_access(prev_use) ? 0 : (VkAccessFlags)prev_use.access;
						sd.srcStageMask = (VkPipelineStageFlags)src_stages;
						sd.srcSubpass = left->pass->subpass;
						rp.rpci.subpass_dependencies.push_back(sd);
					}
					auto& left_rp = impl->rpis[left->pass->render_pass_index];
					auto& right_rp = impl->rpis[right.pass->render_pass_index];
					// cross-queue is impossible here
					assert(!crosses_queue);
					if (left_rp.framebufferless && (next_use.layout != prev_use.layout || is_write_access(prev_use) || is_write_access(next_use))) {
						// right layout == Undefined means the chain terminates, no transition/barrier
						if (next_use.layout == ImageLayout::eUndefined)
							continue;
						right_rp.subpasses[right.pass->subpass].pre_barriers.push_back(ib);
					}
				}
			}
		}

		for (auto& rp : impl->rpis) {
			for (auto& att : rp.attachments) {
				assert(att.description.finalLayout != (VkImageLayout)ImageLayout::eUndefined);
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

			RGImpl::Release* release = nullptr;
			if (auto it = res_rels.find(name); it != res_rels.end()) {
				release = &it->second;
				chain.emplace_back(
				    UseRef{ {}, {}, vuk::eManual, vuk::eManual, it->second.dst_use, Resource::Type::eBuffer, Resource::Subrange{ .buffer = {} }, nullptr });
			} else {
				chain.emplace_back(
				    UseRef{ {}, {}, vuk::eManual, vuk::eManual, buffer_info.final, Resource::Type::eBuffer, Resource::Subrange{ .buffer = {} }, nullptr });
			}
			{
				// if the last use did not specify a domain, we'll take domain from previous use
				// previous use must always have a concrete domain
				auto& last_use = chain[chain.size() - 1].use;
				if (last_use.domain == DomainFlagBits::eAny || last_use.domain == DomainFlagBits::eDevice) {
					last_use.domain = chain[chain.size() - 2].use.domain;
				}
			}

			for (size_t i = 0; i < chain.size() - 1; i++) {
				auto& left = chain[i];
				auto& right = chain[i + 1];

				DomainFlags left_domain = left.use.domain;
				DomainFlags right_domain = right.use.domain;

				left.use = left.high_level_access == Access::eManual ? left.use : to_use(left.high_level_access, left_domain);
				right.use = right.high_level_access == Access::eManual ? right.use : to_use(right.high_level_access, right_domain);

				scope_to_domain(left.use.stages, left_domain & DomainFlagBits::eQueueMask);
				scope_to_domain(right.use.stages, right_domain & DomainFlagBits::eQueueMask);

				bool crosses_queue = (left_domain != DomainFlagBits::eNone && right_domain != DomainFlagBits::eNone &&
				                      (left_domain & DomainFlagBits::eQueueMask) != (right_domain & DomainFlagBits::eQueueMask));

				if (right.pass && right.pass->pass.signal) {
					auto& fut = *right.pass->pass.signal;
					fut.last_use = QueueResourceUse{ left.use.stages, left.use.access, left.use.layout, (left_domain & DomainFlagBits::eQueueMask) };
					fut.result = buffer_info.buffer; // TODO: when we have managed buffers, then this is too soon to attach
				}

				if (i + 2 == chain.size() && release) {
					auto& fut = *release->signal;
					fut.last_use = QueueResourceUse{ left.use.stages, left.use.access, left.use.layout, (left_domain & DomainFlagBits::eQueueMask) };
					fut.result = buffer_info.buffer;           // TODO: when we have managed buffers, then this is too soon to attach
					left.pass->future_signals.push_back(&fut); // last executing pass gets to signal this future too
				}

				if (crosses_queue && left.pass && right.pass) {
					left.pass->is_waited_on = true;
					right.pass->waits.emplace_back((DomainFlagBits)(left_domain & DomainFlagBits::eQueueMask).m_mask, left.pass);

					continue;
				}

				VkMemoryBarrier barrier{ .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER };
				barrier.srcAccessMask = is_read_access(left.use) ? 0 : (VkAccessFlags)left.use.access;
				barrier.dstAccessMask = (VkAccessFlags)right.use.access;
				MemoryBarrier mb{ .barrier = barrier, .src = left.use.stages, .dst = right.use.stages };
				if (mb.src == PipelineStageFlags{}) {
					mb.src = PipelineStageFlagBits::eTopOfPipe;
					mb.barrier.srcAccessMask = {};
				}

				bool crosses_rpass = (left.pass == nullptr || right.pass == nullptr || left.pass->render_pass_index != right.pass->render_pass_index);
				if (crosses_rpass) {
					if (left.pass && right.use.layout != ImageLayout::eUndefined && (is_write_access(left.use) || is_write_access(right.use))) { // RenderPass ->
						auto& left_rp = impl->rpis[left.pass->render_pass_index];
						left_rp.subpasses[left.pass->subpass].post_mem_barriers.push_back(mb);
					}

					if (right.pass && left.use.layout != ImageLayout::eUndefined && (is_write_access(left.use) || is_write_access(right.use))) { // -> RenderPass
						auto& right_rp = impl->rpis[right.pass->render_pass_index];
						right_rp.subpasses[right.pass->subpass].pre_mem_barriers.push_back(mb);
					}
				} else { // subpass-subpass link -> subpass - subpass dependency
					if (left.pass->subpass == right.pass->subpass)
						continue;
					auto& left_rp = impl->rpis[left.pass->render_pass_index];
					if (left_rp.framebufferless && (is_write_access(left.use) || is_write_access(right.use))) {
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

				Name attachment_name = impl->resolve_name(res.name);
				Name subresource_name = attachment_name;
				if (res.type == Resource::Type::eImage && res.subrange.image != Resource::Subrange::Image{}) {
					subresource_name = res.subrange.image.combine_name(impl->resolve_name(res.name));
				}
				auto& attachment_info = impl->bound_attachments.at(attachment_name);
				auto& chain = impl->use_chains.find(subresource_name)->second;
				auto cit = std::find_if(chain.begin(), chain.end(), [&](auto& useref) { return useref.pass == &pass; });
				assert(cit != chain.end());
				attref.layout = (VkImageLayout)cit->use.layout;
				attref.attachment = (uint32_t)std::distance(rp.attachments.begin(), std::find_if(rp.attachments.begin(), rp.attachments.end(), [&](auto& att) {
					                                            return att.attachment_info == &attachment_info;
				                                            }));

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
						auto& dst_att = impl->bound_attachments.at(dst_name);
						rref.layout = (VkImageLayout)ImageLayout::eColorAttachmentOptimal; // the only possible
						                                                                   // layout for resolves
						rref.attachment = (uint32_t)std::distance(
						    rp.attachments.begin(), std::find_if(rp.attachments.begin(), rp.attachments.end(), [&](auto& att) { return att.attachment_info == &dst_att; }));
						auto& att_rp_info = rp.attachments[rref.attachment];

						att_rp_info.attachment_info->attachment.sample_count = Samples::e1; // resolve dst must be sample count = 1
						rp.attachments[rref.attachment].is_resolve_dst = true;
						rp.attachments[rref.attachment].resolve_src = &attachment_info;
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

		return { std::move(*this) };
	}

	MapProxy<Name, std::span<const UseRef>> RenderGraph::get_use_chains() {
		return &impl->use_chains;
	}

	MapProxy<Name, const AttachmentInfo&> RenderGraph::get_bound_attachments() {
		return &impl->bound_attachments;
	}

	MapProxy<Name, const BufferInfo&> RenderGraph::get_bound_buffers() {
		return &impl->bound_buffers;
	}

	ImageUsageFlags RenderGraph::compute_usage(std::span<const UseRef> chain) {
		ImageUsageFlags usage;
		for (const auto& c : chain) {
			if (c.high_level_access != Access::eManual && c.high_level_access != Access::eConverge) {
				switch (to_use(c.high_level_access, DomainFlagBits::eAny).layout) {
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
					usage |= ImageUsageFlagBits::eTransferSrc;
					break;
				case ImageLayout::eTransferDstOptimal:
					usage |= ImageUsageFlagBits::eTransferDst;
					break;
				default:
					break;
				}
				// TODO: this isn't conservative enough, we need more information
				if (c.use.layout == ImageLayout::eGeneral) {
					if (c.use.stages & (PipelineStageFlagBits::eComputeShader | PipelineStageFlagBits::eVertexShader | PipelineStageFlagBits::eTessellationControlShader |
					                    PipelineStageFlagBits::eTessellationEvaluationShader | PipelineStageFlagBits::eGeometryShader |
					                    PipelineStageFlagBits::eFragmentShader | PipelineStageFlagBits::eRayTracingShaderKHR)) {
						usage |= ImageUsageFlagBits::eStorage;
					}
				}
			}
		}

		return usage;
	}
} // namespace vuk
