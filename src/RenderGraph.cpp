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
	Name Subrange::Image::combine_name(Name prefix) const {
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
		PassWrapper pw;
		pw.name = p.name;
		pw.use_secondary_command_buffers = p.use_secondary_command_buffers;
		pw.arguments = p.arguments;
		pw.execute = std::move(p.execute);
		pw.execute_on = p.execute_on;
		pw.resolves =
		    std::span{ reinterpret_cast<decltype(pw.resolves)::value_type*>(impl->arena_->allocate(sizeof(p.resolves[0]) * p.resolves.size())), p.resolves.size() };
		std::copy(p.resolves.begin(), p.resolves.end(), pw.resolves.begin());
		pw.resources = std::span{ reinterpret_cast<decltype(pw.resources)::value_type*>(impl->arena_->allocate(sizeof(p.resources[0]) * p.resources.size())),
			                        p.resources.size() };
		std::copy(p.resources.begin(), p.resources.end(), pw.resources.begin());
		pw.type = p.type;
		impl->passes.emplace_back(std::move(pw));
	}

	void RGCImpl::append(Name subgraph_name, const RenderGraph& other) {
		Name joiner = subgraph_name.is_invalid() ? Name("") : subgraph_name.append("::");

		for (auto [new_name, old_name] : other.impl->aliases) {
			computed_aliases.emplace(joiner.append(new_name.to_sv()), old_name);
		}

		// TODO: this code is written weird because of wonky allocators
		computed_passes.reserve(computed_passes.size() + other.impl->passes.size());
		for (auto& p : other.impl->passes) {
			PassInfo pi{ *arena_, p };
			pi.prefix = joiner;
			pi.qualified_name = joiner.append(p.name.to_sv());
			for (auto r : p.resources) {
				if (!r.name.is_invalid()) {
					r.name = resolve_alias_rec(joiner.append(r.name.to_sv()));
				}
				r.out_name = r.out_name.is_invalid() ? Name{} : resolve_alias_rec(joiner.append(r.out_name.to_sv()));
				pi.resources.emplace_back(std::move(r));
			}

			for (auto& [n1, n2] : p.resolves) {
				pi.resolves.emplace_back(joiner.append(n1.to_sv()), joiner.append(n2.to_sv()));
			}
			computed_passes.emplace_back(std::move(pi));
		}

		for (auto [name, att] : other.impl->bound_attachments) {
			att.name = joiner.append(name.to_sv());
			bound_attachments.emplace(joiner.append(name.to_sv()), std::move(att));
		}
		for (auto [name, buf] : other.impl->bound_buffers) {
			buf.name = joiner.append(name.to_sv());
			bound_buffers.emplace(joiner.append(name.to_sv()), std::move(buf));
		}

		for (auto [name, prefix, iainf] : other.impl->ia_inference_rules) {
			prefix = joiner.append(prefix.to_sv());
			auto& rule = ia_inference_rules[joiner.append(name.to_sv())];
			rule.prefix = prefix;
			rule.rules.emplace_back(iainf);
		}

		for (auto& [name, v] : other.impl->acquires) {
			acquires.emplace(joiner.append(name.to_sv()), v);
		}

		for (auto& [name, v] : other.impl->releases) {
			releases.emplace(joiner.append(name.to_sv()), v);
		}

		for (auto [name, v] : other.impl->diverged_subchain_headers) {
			v.first = joiner.append(v.first.to_sv());
			diverged_subchain_headers.emplace(joiner.append(name.to_sv()), v);
		}
	}

	void RenderGraph::add_alias(Name new_name, Name old_name) {
		if (new_name != old_name) {
			impl->aliases.emplace_back(new_name, old_name);
		}
	}

	void RenderGraph::diverge_image(Name whole_name, Subrange::Image subrange, Name subrange_name) {
		impl->diverged_subchain_headers.emplace_back(subrange_name, std::pair{ whole_name, subrange });
		auto it = std::find(impl->whole_names_consumed.begin(), impl->whole_names_consumed.end(), whole_name);
		bool new_divergence = it == impl->whole_names_consumed.end();

		if (new_divergence) {
			impl->whole_names_consumed.emplace_back(whole_name);
			add_pass({ .name = whole_name.append("_DIVERGE"),
			           .resources = { Resource{ whole_name, Resource::Type::eImage, Access::eConsume, whole_name.append("__diverged") } },
			           .execute = diverge });
		}
	}

	void RenderGraph::converge_image(Name pre_diverge, Name post_diverge) {
		add_pass({ .name = post_diverge.append("_CONVERGE"),
		           .resources = { Resource{ pre_diverge.append("__diverged"), Resource::Type::eImage, Access::eConverge, post_diverge },  },
		           .execute = converge, .type = PassType::eConverge });
	}

	void RenderGraph::converge_image_explicit(std::span<Name> pre_diverge, Name post_diverge) {
		Pass post{ .name = post_diverge.append("_CONVERGE"), .execute = converge, .type = PassType::eConvergeExplicit };
		for (auto& name : pre_diverge) {
			post.resources.emplace_back(Resource{ name, Resource::Type::eImage, Access::eConsume });
		}
		post.resources.emplace_back(Resource{ Name{}, Resource::Type::eImage, Access::eConverge, post_diverge });
		add_pass(std::move(post));
	}

	// determine rendergraph inputs and outputs, and resources that are neither
	std::vector<PassInfo, short_alloc<PassInfo, 64>> RGImpl::build_io(std::span<PassWrapper> passes_used) {
		robin_hood::unordered_flat_set<Name> poisoned_names;

		auto pis = std::vector<PassInfo, short_alloc<PassInfo, 64>>(*arena_);
		for (auto& pass : passes_used) {
			pis.emplace_back(*arena_, pass);
		}

		for (auto& pif : pis) {
			pif.input_names.clear();
			pif.output_names.clear();
			pif.write_input_names.clear();
			pif.bloom_write_inputs = {};
			pif.bloom_outputs = {};
			pif.bloom_resolved_inputs = {};

			for (Resource& res : pif.resources) {
				Name in_name = resolve_alias(res.name);
				Name out_name = resolve_alias(res.out_name);

				auto hashed_in_name = ::hash::fnv1a::hash(in_name.to_sv().data(), res.name.to_sv().size(), hash::fnv1a::default_offset_basis);
				auto hashed_out_name = ::hash::fnv1a::hash(out_name.to_sv().data(), res.out_name.to_sv().size(), hash::fnv1a::default_offset_basis);

				pif.input_names.emplace_back(in_name);
				pif.bloom_resolved_inputs |= hashed_in_name;

				if (!res.out_name.is_invalid()) {
					pif.bloom_outputs |= hashed_out_name;
					pif.output_names.emplace_back(out_name);
				}

				if (is_write_access(res.ia) || is_acquire(res.ia) || is_release(res.ia) || res.ia == Access::eConsume || res.ia == Access::eConverge ||
				    pif.pass->type == PassType::eForcedAccess) {
					assert(!poisoned_names.contains(in_name)); // we have poisoned this name because a write has already consumed it
					pif.bloom_write_inputs |= hashed_in_name;
					pif.write_input_names.emplace_back(in_name);
					poisoned_names.emplace(in_name);
				}

				// if this resource use is the first in a diverged subchain, we additionally add a dependency onto the undiverged subchain
				if (auto it = std::find_if(diverged_subchain_headers.begin(), diverged_subchain_headers.end(), [=](auto& item) { return item.first == in_name; });
				    it != diverged_subchain_headers.end()) {
					auto& sch_info = it->second;
					auto dep = sch_info.first.append("__diverged");
					auto hashed_name = ::hash::fnv1a::hash(dep.to_sv().data(), dep.to_sv().size(), hash::fnv1a::default_offset_basis);

					pif.input_names.emplace_back(dep);
					pif.bloom_resolved_inputs |= hashed_name;
				}
			}
		}

		return pis;
	}

	// determine rendergraph inputs and outputs, and resources that are neither
	void RGCImpl::build_io(std::span<struct PassInfo> passes_used) {
		robin_hood::unordered_flat_set<Name> poisoned_names;

		for (auto& pif : passes_used) {
			pif.input_names.clear();
			pif.output_names.clear();
			pif.write_input_names.clear();
			pif.bloom_write_inputs = {};
			pif.bloom_outputs = {};
			pif.bloom_resolved_inputs = {};

			for (Resource& res : pif.resources) {
				Name in_name = res.name; // these names have been alias-resolved already
				Name out_name = res.out_name;

				auto hashed_in_name = ::hash::fnv1a::hash(in_name.to_sv().data(), res.name.to_sv().size(), hash::fnv1a::default_offset_basis);
				auto hashed_out_name = ::hash::fnv1a::hash(out_name.to_sv().data(), res.out_name.to_sv().size(), hash::fnv1a::default_offset_basis);

				pif.input_names.emplace_back(in_name);
				pif.bloom_resolved_inputs |= hashed_in_name;

				if (!res.out_name.is_invalid()) {
					pif.bloom_outputs |= hashed_out_name;
					pif.output_names.emplace_back(out_name);
				}

				if (is_write_access(res.ia) || is_acquire(res.ia) || is_release(res.ia) || res.ia == Access::eConsume || res.ia == Access::eConverge ||
				    pif.pass->type == PassType::eForcedAccess) {
					assert(!poisoned_names.contains(in_name)); // we have poisoned this name because a write has already consumed it
					pif.bloom_write_inputs |= hashed_in_name;
					pif.write_input_names.emplace_back(in_name);
					poisoned_names.emplace(in_name);
				}

				// if this resource use is the first in a diverged subchain, we additionally add a dependency onto the undiverged subchain
				if (auto it = diverged_subchain_headers.find(in_name); it != diverged_subchain_headers.end()) {
					auto& sch_info = it->second;
					auto dep = sch_info.first.append("__diverged");
					auto hashed_name = ::hash::fnv1a::hash(dep.to_sv().data(), dep.to_sv().size(), hash::fnv1a::default_offset_basis);

					pif.input_names.emplace_back(dep);
					pif.bloom_resolved_inputs |= hashed_name;
				}
			}
		}
	}

	void RGCImpl::schedule_intra_queue(std::span<PassInfo> passes, const RenderGraphCompileOptions& compile_options) {
		// sort passes if requested
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

	std::string Compiler::dump_graph() {
		std::stringstream ss;
		ss << "digraph vuk {\n";
		for (auto i = 0; i < impl->computed_passes.size(); i++) {
			for (auto j = 0; j < impl->computed_passes.size(); j++) {
				if (i == j)
					continue;
				auto& p1 = impl->computed_passes[i];
				auto& p2 = impl->computed_passes[j];
				for (auto& o : p1.output_names) {
					for (auto& i : p2.input_names) {
						if (o == impl->resolve_alias(i)) {
							ss << "\"" << p1.pass->name.c_str() << "\" -> \"" << p2.pass->name.c_str() << "\" [label=\"" << impl->resolve_alias(i).c_str() << "\"];\n";
							// p2 is ordered after p1
						}
					}
				}
				for (auto& o : p1.input_names) {
					for (auto& i : p2.write_input_names) {
						if (impl->resolve_alias(o) == impl->resolve_alias(i)) {
							ss << "\"" << p1.pass->name.c_str() << "\" -> \"" << p2.pass->name.c_str() << "\" [label=\"" << impl->resolve_alias(i).c_str() << "\"];\n";
							// p2 is ordered after p1
						}
					}
				}
			}
		}
		ss << "}\n";
		return ss.str();
	}

	std::unordered_map<std::shared_ptr<RenderGraph>, std::string> RGCImpl::compute_prefixes(const RenderGraph& rg, bool do_prefix) {
		std::unordered_map<std::shared_ptr<RenderGraph>, std::string> sg_prefixes;
		for (auto& [sg_ptr, sg_info] : rg.impl->subgraphs) {
			if (sg_info.count > 0) {
				Name sg_name = sg_ptr->name;
				assert(sg_ptr->impl);

				auto prefixes = compute_prefixes(*sg_ptr, true);
				sg_prefixes.merge(prefixes);
				if (auto& counter = ++sg_name_counter[sg_name]; counter > 1) {
					sg_name = sg_name.append(std::string("_") + std::to_string(counter - 1));
				}
				sg_prefixes.emplace(sg_ptr, std::string(sg_name.to_sv()));
			}
		}
		if (do_prefix && sg_prefixes.size() > 0) {
			for (auto& [k, v] : sg_prefixes) {
				v = std::string(rg.name.to_sv()) + "::" + v;
			}
		}
		return sg_prefixes;
	}

	void RGCImpl::inline_subgraphs(const std::shared_ptr<RenderGraph>& rg,
	                               const std::unordered_map<std::shared_ptr<RenderGraph>, std::string>& sg_prefixes,
	                               std::unordered_set<std::shared_ptr<RenderGraph>>& consumed_rgs) {
		auto our_prefix = sg_prefixes.at(rg);
		for (auto& [sg_ptr, sg_info] : rg->impl->subgraphs) {
			if (sg_info.count > 0) {
				auto& prefix = sg_prefixes.at(sg_ptr);
				assert(sg_ptr->impl);
				for (auto& [name_in_parent, name_in_sg] : sg_info.exported_names) {
					auto old_name = Name(prefix).append("::").append(name_in_sg.to_sv());
					auto new_name = our_prefix.empty() ? name_in_parent : Name(our_prefix).append("::").append(name_in_parent.to_sv());
					if (name_in_parent != old_name) {
						computed_aliases[new_name] = old_name;
					}
				}
				if (!consumed_rgs.contains(sg_ptr)) {
					inline_subgraphs(sg_ptr, sg_prefixes, consumed_rgs);
					append(Name(prefix), *sg_ptr);
					consumed_rgs.emplace(sg_ptr);
				}
			}
		}
	}

	Compiler::Compiler() : impl(new RGCImpl) {}
	Compiler::~Compiler() {
		delete impl;
	}

	void Compiler::compile(std::span<std::shared_ptr<RenderGraph>> rgs, const RenderGraphCompileOptions& compile_options) {
		auto arena = impl->arena_.release();
		delete impl;
		arena->reset();
		impl = new RGCImpl(arena);

		// inline all the subgraphs into us
		std::unordered_map<std::shared_ptr<RenderGraph>, std::string> sg_pref;
		for (auto& rg : rgs) {
			auto prefs = impl->compute_prefixes(*rg, true);
			auto rg_name = rg->name;
			if (auto& counter = ++impl->sg_name_counter[rg_name]; counter > 1) {
				rg_name = rg_name.append(std::string("_") + std::to_string(counter - 1));
			}
			prefs.emplace(rg, std::string{ rg_name.c_str() });
			sg_pref.merge(prefs);
			std::unordered_set<std::shared_ptr<RenderGraph>> consumed_rgs = {};
			impl->inline_subgraphs(rg, sg_pref, consumed_rgs);
		}

		for (auto& rg : rgs) {
			impl->append(Name{ sg_pref.at(rg).c_str() }, *rg);
		}

		// gather name alias info now - once we partition, we might encounter unresolved aliases
		robin_hood::unordered_flat_map<Name, Name> name_map;

		name_map.insert(impl->computed_aliases.begin(), impl->computed_aliases.end());

		impl->computed_aliases.clear();
		// follow aliases and resolve them into a single lookup
		for (auto& [k, v] : name_map) {
			auto it = name_map.find(v);
			Name res = v;
			while (it != name_map.end()) {
				res = it->second;
				it = name_map.find(res);
			}
			assert(!res.is_invalid());
			impl->computed_aliases.emplace(k, res);
		}

		for (auto& passinfo : impl->computed_passes) {
			for (auto& res : passinfo.resources) {
				// for read or write, we add source to use chain
				if (!res.name.is_invalid() && !res.out_name.is_invalid()) {
					auto [iter, succ] = name_map.emplace(res.out_name, res.name);
					assert(iter->second == res.name);
				}
			}
		}

		impl->assigned_names.clear();
		// populate resource name -> use chain map
		for (auto& [k, v] : name_map) {
			auto it = name_map.find(v);
			Name res = v;
			while (it != name_map.end()) {
				res = it->second;
				it = name_map.find(res);
			}
			assert(!res.is_invalid());
			impl->assigned_names.emplace(k, res);
		}

		// find which reads are graph inputs (not produced by any pass) & outputs
		// (not consumed by any pass)
		impl->build_io(impl->computed_passes);

		// run global pass ordering - once we split per-queue we don't see enough
		// inputs to order within a queue
		impl->schedule_intra_queue(impl->computed_passes, compile_options);

		// auto dumped_graph = dump_graph();

		// for now, just use what the passes requested as domain
		for (auto& p : impl->computed_passes) {
			p.domain = p.pass->execute_on;
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

		// prepare converge passes
		for (auto& passinfo : impl->computed_passes) {
			if (passinfo.pass->type != PassType::eConvergeExplicit) {
				continue;
			}

			auto post_diverge_name = passinfo.resources.back().out_name;

			// TODO: does this really work or make sense for >1 undiv_names?
			std::unordered_set<Name> undiv_names;
			for (int i = 0; i < passinfo.resources.size() - 1; i++) {
				auto& res = passinfo.resources[i];
				auto resolved_name = impl->resolve_name(res.name);
				auto& undiv_name = impl->diverged_subchain_headers.at(resolved_name).first;
				undiv_names.emplace(undiv_name);
			}

			passinfo.resources.pop_back();

			for (auto& name : undiv_names) {
				passinfo.resources.push_back(Resource{ name.append("__diverged"), Resource::Type::eImage, Access::eConverge, post_diverge_name });
				name_map.emplace(passinfo.resources.back().out_name, passinfo.resources.back().name);
			}
		}

		impl->assigned_names.clear();
		// populate resource name -> use chain map
		for (auto& [k, v] : name_map) {
			auto it = name_map.find(v);
			Name res = v;
			while (it != name_map.end()) {
				res = it->second;
				it = name_map.find(res);
			}
			assert(!res.is_invalid());
			impl->assigned_names.emplace(k, res);
		}

		// use chains pass
		impl->use_chains.clear();
		std::unordered_multimap<Name, Name> diverged_chains;
		for (PassInfo& passinfo : impl->computed_passes) {
			for (Resource& res : passinfo.resources) {
				Name resolved_name = impl->resolve_name(res.name);

				bool skip = false;
				// for read or write, we add source to use chain
				auto it = impl->use_chains.find(resolved_name);
				if (it == impl->use_chains.end()) {
					it = impl->use_chains.emplace(resolved_name, std::vector<UseRef, short_alloc<UseRef, 64>>{ short_alloc<UseRef, 64>{ *impl->arena_ } }).first;
					auto& chain = it->second;

					bool diverged_subchain = false;
					Name undiverged_name = resolved_name;

					Subrange sr{};
					if (auto it = impl->diverged_subchain_headers.find(resolved_name); it != impl->diverged_subchain_headers.end()) {
						auto& sch_info = it->second;
						diverged_subchain = true;
						sr.image = sch_info.second;
						// find undiverged name
						undiverged_name = impl->resolve_name(sch_info.first);
					}

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
						Acquire* acquire = &it->second;
						chain.emplace_back(UseRef{ {}, {}, vuk::eManual, vuk::eManual, acquire->src_use, res.type, {}, nullptr });
						chain.emplace_back(UseRef{ res.name, res.out_name, res.ia, res.ia, {}, res.type, sr, &passinfo });
						chain.back().use.domain = passinfo.domain;
						// make the actually executed first pass wait on the future
						chain.back().pass->absolute_waits.emplace_back(acquire->initial_domain, acquire->initial_visibility);
						skip = true; // we added to the chain here already
					} else if (is_direct_attachment_use) {
						chain.insert(chain.begin(), UseRef{ {}, {}, vuk::eManual, vuk::eManual, initial, res.type, {}, nullptr });
					} else if (diverged_subchain) { // 3. if this is a diverged subchain, we want to put the last undiverged use here (or nothing, if this was first use)
						auto& undiverged_chain = impl->use_chains.at(undiverged_name);
						if (undiverged_chain.size() > 0) {
							chain.emplace_back(undiverged_chain.back());
						}
						chain.emplace_back(UseRef{ res.name, res.out_name, res.ia, res.ia, {}, res.type, sr, &passinfo });
						chain.back().use.domain = passinfo.domain;
						skip = true;
						diverged_chains.emplace(undiverged_name, resolved_name);
					}
				}
				auto& chain = it->second;

				// we don't want to put the convergence onto the divergent use chains
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
					Subrange sr{};
					if (!chain.empty()) {
						sr = chain.back().subrange;
					}
					chain.emplace_back(UseRef{ res.name, res.out_name, res.ia, res.ia, {}, res.type, sr, &passinfo });
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
		impl->ordered_passes.reserve(impl->computed_passes.size());
		for (auto& p : impl->computed_passes) {
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

			for (auto& res : passinfo->resources) {
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
						rpi.subpasses.back().use_secondary_command_buffers |= p->pass->use_secondary_command_buffers;
						continue;
					}
				}
				SubpassInfo si{ *impl->arena_ };
				si.passes = { p };
				si.use_secondary_command_buffers = p->pass->use_secondary_command_buffers;
				p->subpass = ++subpass;
				rpi.subpasses.push_back(si);
			}
			for (auto& att : attachments) {
				auto res_name = impl->resolve_name(att.name);
				auto whole_name = impl->whole_name(res_name);
				rpi.attachments.push_back(AttachmentRPInfo{ &impl->bound_attachments.at(whole_name) });
				auto& ch = impl->use_chains.at(res_name);
				rpi.layer_count = ch[1].subrange.image.layer_count;
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

	void RenderGraph::clear_image(Name image_name, Name image_name_out, Clear clear_value) {
		auto arg_ptr = impl->arena_->allocate(sizeof(Clear));
		std::memcpy(arg_ptr, &clear_value, sizeof(Clear));
		Resource res{ image_name, Resource::Type::eImage, eClear, image_name_out };
		add_pass({ .name = image_name.append("_CLEAR"),
		           .resources = { std::move(res) },
		           .execute = [image_name, clear_value](CommandBuffer& cbuf) { cbuf.clear_image(image_name, clear_value); },
		           .arguments = reinterpret_cast<std::byte*>(arg_ptr),
		           .type = PassType::eClear });
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
		Name tmp_name = name.append(get_temporary_name().to_sv());
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
			impl->acquires.emplace_back(name, Acquire{ fimg.control->last_use, fimg.control->initial_domain, fimg.control->initial_visibility });
			impl->imported_names.emplace_back(name);
		} else if (fimg.get_status() == FutureBase::Status::eInitial) {
			// an unsubmitted RG is being attached, we remove the release from that RG, and we allow the name to be found in us
			assert(fimg.rg->impl);
			std::erase_if(fimg.rg->impl->releases, [name = fimg.get_bound_name()](auto& item) { return item.first == name; });
			auto sg_info_it = std::find_if(impl->subgraphs.begin(), impl->subgraphs.end(), [&](auto& it) { return it.first == fimg.rg; });
			if (sg_info_it == impl->subgraphs.end()) {
				impl->subgraphs.emplace_back(std::pair{ fimg.rg, RGImpl::SGInfo{} });
				sg_info_it = impl->subgraphs.end() - 1;
			}
			auto& sg_info = sg_info_it->second;
			sg_info.count++;
			auto old_exported_names = sg_info.exported_names;
			auto current_exported_name_size = sg_info.exported_names.size_bytes();
			sg_info.exported_names = std::span{ reinterpret_cast<decltype(sg_info.exported_names)::value_type*>(
				                                      impl->arena_->allocate(current_exported_name_size + sizeof(sg_info.exported_names[0]))),
				                                  sg_info.exported_names.size() + 1 };
			std::copy(old_exported_names.begin(), old_exported_names.end(), sg_info.exported_names.begin());
			sg_info.exported_names.back() = std::pair{ name, fimg.get_bound_name() };
			impl->imported_names.emplace_back(name);
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
		impl->ia_inference_rules.emplace_back(IAInference{ target, Name(""), std::move(rule) });
	}

	robin_hood::unordered_flat_set<Name> RGImpl::get_available_resources() {
		auto pass_infos = build_io(passes);
		// seed the available names with the names we imported from subgraphs
		robin_hood::unordered_flat_set<Name> outputs;
		outputs.insert(imported_names.begin(), imported_names.end());

		for (auto& [name, _] : bound_attachments) {
			outputs.insert(name);
		}
		for (auto& [name, _] : bound_buffers) {
			outputs.insert(name);
		}

		for (auto& pif : pass_infos) {
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

	void RenderGraph::attach_out(Name name, Future& fimg, DomainFlags dst_domain, Subrange subrange) {
		impl->releases.emplace_back(name, Release{ to_use(Access::eNone, dst_domain), subrange, fimg.control.get() });
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
		return impl->temporary_name.append(std::to_string(impl->temporary_name_counter++));
	}

	IARule same_extent_as(Name n) {
		return [=](const InferenceContext& ctx, ImageAttachment& ia) {
			ia.extent = ctx.get_image_attachment(n).extent;
		};
	}

	IARule same_2D_extent_as(Name n) {
		return [=](const InferenceContext& ctx, ImageAttachment& ia) {
			auto& o = ctx.get_image_attachment(n);
			ia.extent.sizing = o.extent.sizing;
			ia.extent.extent.width = o.extent.extent.width;
			ia.extent.extent.height = o.extent.extent.height;
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

	void RGCImpl::validate() {
		// check if all resourced are attached
		for (const auto& [n, v] : use_chains) {
			auto resolved_name = resolve_name(n);
			auto att_name = whole_name(resolved_name);
			if (!bound_attachments.contains(att_name) && !bound_buffers.contains(att_name)) {
				// TODO: error handling
				throw RenderGraphException{ std::string("Missing resource: \"") + std::string(n.to_sv()) + "\". Did you forget to attach it?" };
			}
		}
	}

	ExecutableRenderGraph Compiler::link(std::span<std::shared_ptr<RenderGraph>> rgs, const RenderGraphCompileOptions& compile_options) {
		compile(rgs, compile_options);

		// at this point the graph is built, we know of all the resources and
		// everything should have been attached perform checking if this indeed the
		// case
		impl->validate();

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
				if (left.original == eClear && left.pass->pass->type == PassType::eClear) {
					// next use is as fb attachment
					if ((i < chain.size() - 1) && is_framebuffer_attachment(to_use(right.original, right.use.domain))) {
						auto& next_rpi = impl->rpis[right.pass->render_pass_index];
						for (auto& rpi_att : next_rpi.attachments) {
							if (rpi_att.attachment_info == &attachment_info) {
								rpi_att.clear_value = Clear{};
								std::memcpy(&rpi_att.clear_value, left.pass->pass->arguments, sizeof(Clear));
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
			auto resolved_name = impl->resolve_name(raw_name);
			auto whole_name = impl->whole_name(resolved_name);
			auto att_it = impl->bound_attachments.find(whole_name);
			if (att_it == impl->bound_attachments.end()) {
				continue;
			}

			auto& attachment_info = att_it->second;

			Release* release = nullptr;
			if (auto it = res_rels.find(resolved_name); it != res_rels.end()) {
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
				barrier.srcQueueFamilyIndex = static_cast<uint32_t>((left->use.domain & DomainFlagBits::eQueueMask).m_mask);
				barrier.dstQueueFamilyIndex = static_cast<uint32_t>((right.use.domain & DomainFlagBits::eQueueMask).m_mask);

				if (src_stages == PipelineStageFlags{}) {
					barrier.srcAccessMask = {};
				}
				if (dst_stages == PipelineStageFlags{}) {
					barrier.dstAccessMask = {};
				}
				ImageBarrier ib{ .image = whole_name, .barrier = barrier, .src = src_stages, .dst = dst_stages };

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

			Release* release = nullptr;
			if (auto it = res_rels.find(name); it != res_rels.end()) {
				release = &it->second;
				chain.emplace_back(UseRef{ {}, {}, vuk::eManual, vuk::eManual, it->second.dst_use, Resource::Type::eBuffer, Subrange{ .buffer = {} }, nullptr });
			} else {
				chain.emplace_back(UseRef{ {}, {}, vuk::eManual, vuk::eManual, buffer_info.final, Resource::Type::eBuffer, Subrange{ .buffer = {} }, nullptr });
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
					if (left.pass && (crosses_queue || (!right.pass && right.use.layout != ImageLayout::eUndefined &&
					                                    (is_write_access(left.use) || is_write_access(right.use))))) { // RenderPass ->
						auto& left_rp = impl->rpis[left.pass->render_pass_index];
						left_rp.subpasses[left.pass->subpass].post_mem_barriers.push_back(mb);
					}

					if (right.pass && left.use.layout != ImageLayout::eUndefined && (is_write_access(left.use) || is_write_access(right.use))) { // -> RenderPass
						auto& right_rp = impl->rpis[right.pass->render_pass_index];
						if (right_rp.framebufferless) {
							right_rp.subpasses[right.pass->subpass].pre_mem_barriers.push_back(mb);
						} else {
							right_rp.pre_mem_barriers.push_back(mb);
						}
					}
				} else { // subpass-subpass link -> subpass - subpass dependency
					if (left.pass->subpass == right.pass->subpass)
						continue;
					auto& left_rp = impl->rpis[left.pass->render_pass_index];
					auto& right_rp = impl->rpis[right.pass->render_pass_index];
					// cross-queue is impossible here
					assert(!crosses_queue);
					if (left_rp.framebufferless && (is_write_access(left.use) || is_write_access(right.use))) {
						// right layout == Undefined means the chain terminates, no transition/barrier
						if (right.use.layout == ImageLayout::eUndefined)
							continue;
						right_rp.subpasses[right.pass->subpass].pre_mem_barriers.push_back(mb);
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

			for (auto& res : pass.resources) {
				if (!is_framebuffer_attachment(res))
					continue;
				if (res.ia == Access::eColorResolveWrite) // resolve attachment are added when
				                                          // processing the color attachment
					continue;
				VkAttachmentReference attref{};

				Name resolved_name = impl->resolve_name(res.name);
				Name attachment_name = impl->whole_name(resolved_name);
				auto& attachment_info = impl->bound_attachments.at(attachment_name);
				auto& chain = impl->use_chains.find(resolved_name)->second;
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
					if (auto it = std::find_if(pass.resolves.begin(), pass.resolves.end(), [=](auto& p) { return p.first == res.name; }); it != pass.resolves.end()) {
						// this a resolve src attachment
						// get the dst attachment
						auto dst_name = impl->resolve_name(it->second);
						auto whole_name = impl->whole_name(dst_name);
						auto& dst_att = impl->bound_attachments.at(whole_name);
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

		return { *this };
	}

	MapProxy<Name, std::span<const UseRef>> Compiler::get_use_chains() {
		return &impl->use_chains;
	}

	MapProxy<Name, const AttachmentInfo&> Compiler::get_bound_attachments() {
		return &impl->bound_attachments;
	}

	MapProxy<Name, const BufferInfo&> Compiler::get_bound_buffers() {
		return &impl->bound_buffers;
	}

	ImageUsageFlags Compiler::compute_usage(std::span<const UseRef> chain) {
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
