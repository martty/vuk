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

	void RenderGraph::add_pass(Pass p, source_location source) {
		PassWrapper pw;
		pw.name = p.name;
		pw.arguments = p.arguments;
		pw.execute = std::move(p.execute);
		pw.execute_on = p.execute_on;
		pw.resources.offset0 = impl->resources.size();
		impl->resources.insert(impl->resources.end(), p.resources.begin(), p.resources.end());
		pw.resources.offset1 = impl->resources.size();
		pw.type = p.type;
		pw.source = std::move(source);
		impl->passes.emplace_back(std::move(pw));
	}

	void RGCImpl::append(Name subgraph_name, const RenderGraph& other) {
		Name joiner = subgraph_name.is_invalid() ? Name("") : subgraph_name;

		for (auto [new_name, old_name] : other.impl->aliases) {
			computed_aliases.emplace(QualifiedName{ joiner, new_name }, QualifiedName{ Name{}, old_name });
		}

		/* for (auto old_name : other.impl->imported_names) {
		  computed_aliases.emplace(QualifiedName{ joiner, old_name }, QualifiedName{ Name{}, old_name });
		}*/

		for (auto& p : other.impl->passes) {
			PassInfo& pi = computed_passes.emplace_back(*arena_, p);
			pi.qualified_name = { joiner, p.name };
			pi.resources.offset0 = resources.size();
			for (auto r : p.resources.to_span(other.impl->resources)) {
				r.original_name = r.name.name;
				if (r.foreign) {
					auto prefix = Name{ sg_prefixes.at(r.foreign) };
					auto res_name = resolve_alias_rec({ prefix, r.name.name });
					auto res_out_name = r.out_name.name.is_invalid() ? QualifiedName{} : resolve_alias_rec({ prefix, r.out_name.name });
					computed_aliases.emplace(QualifiedName{ joiner, r.name.name }, res_name);
					r.name = res_name;
					if (!r.out_name.is_invalid()) {
						computed_aliases.emplace(QualifiedName{ joiner, r.out_name.name }, res_out_name);
						r.out_name = res_out_name;
					}
				} else {
					if (!r.name.name.is_invalid()) {
						r.name = resolve_alias_rec({ joiner, r.name.name });
					}
					r.out_name = r.out_name.name.is_invalid() ? QualifiedName{} : resolve_alias_rec({ joiner, r.out_name.name });
				}
				resources.emplace_back(std::move(r));
			}
			pi.resources.offset1 = resources.size();
		}

		for (auto [name, att] : other.impl->bound_attachments) {
			att.name = { joiner, name.name };
			bound_attachments.emplace_back(std::move(att));
		}
		for (auto [name, buf] : other.impl->bound_buffers) {
			buf.name = { joiner, name.name };
			bound_buffers.emplace_back(std::move(buf));
		}

		for (auto [name, iainf] : other.impl->ia_inference_rules) {
			auto& rule = ia_inference_rules[resolve_alias_rec(QualifiedName{ joiner, name.name })];
			rule.prefix = joiner;
			rule.rules.emplace_back(iainf);
		}

		for (auto [name, bufinf] : other.impl->buf_inference_rules) {
			auto& rule = buf_inference_rules[resolve_alias_rec(QualifiedName{ joiner, name.name })];
			rule.prefix = joiner;
			rule.rules.emplace_back(bufinf);
		}

		for (auto& [name, v] : other.impl->releases) {
			auto res_name = resolve_alias_rec({ joiner, name.name });
			releases.emplace_back(res_name, v);
		}

		for (auto [name, v] : other.impl->diverged_subchain_headers) {
			v.first.prefix = joiner;
			diverged_subchain_headers.emplace(QualifiedName{ joiner, name.name }, v);
		}
	}

	void RenderGraph::add_alias(Name new_name, Name old_name) {
		if (new_name != old_name) {
			impl->aliases.emplace_back(new_name, old_name);
		}
	}

	void RenderGraph::diverge_image(Name whole_name, Subrange::Image subrange, Name subrange_name) {
		impl->diverged_subchain_headers.emplace_back(QualifiedName{ Name{}, subrange_name }, std::pair{ QualifiedName{ Name{}, whole_name }, subrange });

		add_pass({ .name = whole_name.append("_DIVERGE"),
		           .resources = { Resource{ whole_name, Resource::Type::eImage, Access::eConsume, subrange_name } },
		           .execute = diverge,
		           .type = PassType::eDiverge });
	}

	void RenderGraph::converge_image_explicit(std::span<Name> pre_diverge, Name post_diverge) {
		Pass post{ .name = post_diverge.append("_CONVERGE"), .execute = converge, .type = PassType::eConverge };
		post.resources.emplace_back(Resource{ Name{}, Resource::Type::eImage, Access::eConverge, post_diverge });
		for (auto& name : pre_diverge) {
			post.resources.emplace_back(Resource{ name, Resource::Type::eImage, Access::eConsume });
		}
		add_pass(std::move(post));
	}

	void RGCImpl::merge_diverge_passes(std::vector<PassInfo, short_alloc<PassInfo, 64>>& passes) {
		std::unordered_map<QualifiedName, PassInfo*> merge_passes;
		for (auto& pass : passes) {
			if (pass.pass->type == PassType::eDiverge) {
				auto& pi = merge_passes[pass.qualified_name];
				if (!pi) {
					pi = &pass;
					auto res = pass.resources.to_span(resources)[0];
					res.name = {};
					pass.resources.append(resources, res);
					pass.resources.to_span(resources)[0].out_name = {};
				} else {
					auto o = pass.resources.to_span(resources);
					assert(o.size() == 1);
					o[0].name = {};
					pi->resources.append(resources, o[0]);
					pass.resources = {};
				}
			}
		}
		std::erase_if(passes, [](auto& pass) { return pass.resources.size() == 0; });
	}

	// determine rendergraph inputs and outputs, and resources that are neither
	std::vector<PassInfo, short_alloc<PassInfo, 64>> RGImpl::build_io(std::span<PassWrapper> passes_used) {
		robin_hood::unordered_flat_set<Name> poisoned_names;

		auto pis = std::vector<PassInfo, short_alloc<PassInfo, 64>>(*arena_);
		for (auto& pass : passes_used) {
			pis.emplace_back(*arena_, pass);
		}

		input_names.clear();
		output_names.clear();
		write_input_names.clear();

		for (auto& pif : pis) {
			pif.bloom_write_inputs = {};
			pif.bloom_outputs = {};
			pif.bloom_resolved_inputs = {};
			pif.input_names.offset0 = input_names.size();
			pif.output_names.offset0 = output_names.size();
			pif.write_input_names.offset0 = write_input_names.size();

			for (Resource& res : pif.resources.to_span(resources)) {
				if (res.foreign) {
					continue;
				}
				Name in_name = resolve_alias(res.name.name);
				Name out_name = resolve_alias(res.out_name.name);

				auto hashed_in_name = ::hash::fnv1a::hash(in_name.to_sv().data(), res.name.name.to_sv().size(), hash::fnv1a::default_offset_basis);
				auto hashed_out_name = ::hash::fnv1a::hash(out_name.to_sv().data(), res.out_name.name.to_sv().size(), hash::fnv1a::default_offset_basis);

				input_names.emplace_back(QualifiedName{ Name{}, in_name });
				pif.bloom_resolved_inputs |= hashed_in_name;

				if (!res.out_name.name.is_invalid()) {
					pif.bloom_outputs |= hashed_out_name;
					output_names.emplace_back(QualifiedName{ Name{}, out_name });
				}

				if (is_write_access(res.ia) || res.ia == Access::eConsume || res.ia == Access::eConverge || pif.pass->type == PassType::eForcedAccess) {
					assert(!poisoned_names.contains(in_name)); // we have poisoned this name because a write has already consumed it
					pif.bloom_write_inputs |= hashed_in_name;
					write_input_names.emplace_back(QualifiedName{ Name{}, in_name });
					poisoned_names.emplace(in_name);
				}

				// if this resource use is the first in a diverged subchain, we additionally add a dependency onto the undiverged subchain
				if (auto it = std::find_if(diverged_subchain_headers.begin(), diverged_subchain_headers.end(), [=](auto& item) { return item.first.name == in_name; });
				    it != diverged_subchain_headers.end()) {
					auto& sch_info = it->second;
					auto dep = sch_info.first.name.append("__diverged");
					auto hashed_name = ::hash::fnv1a::hash(dep.to_sv().data(), dep.to_sv().size(), hash::fnv1a::default_offset_basis);

					input_names.emplace_back(QualifiedName{ Name{}, dep });
					pif.bloom_resolved_inputs |= hashed_name;
				}
			}

			pif.input_names.offset1 = input_names.size();
			pif.output_names.offset1 = output_names.size();
			pif.write_input_names.offset1 = write_input_names.size();
		}

		return pis;
	}

	void RGCImpl::schedule_intra_queue(std::span<PassInfo> passes, const RenderGraphCompileOptions& compile_options) {
		// build edges into link map
		// reserving here to avoid rehashing map
		// TODO: we know the upper bound on size, use that
		res_to_links.clear();
		res_to_links.reserve(1000);
		bound_attachments.reserve(1000);
		bound_buffers.reserve(1000);

		for (auto pass_idx = 0; pass_idx < passes.size(); pass_idx++) {
			auto& pif = passes[pass_idx];
			for (Resource& res : pif.resources.to_span(resources)) {
				bool is_undef = false;
				bool is_def = !res.out_name.is_invalid();
				int32_t res_idx = static_cast<int32_t>(&res - &*pif.resources.to_span(resources).begin());
				if (!res.name.is_invalid()) {
					auto& r_io = res_to_links[res.name];
					r_io.type = res.type;
					if (!is_write_access(res.ia) && pif.pass->type != PassType::eForcedAccess && res.ia != Access::eConsume) {
						r_io.reads.append(pass_idx_helper, { pass_idx, res_idx });
					}
					if (is_write_access(res.ia) || res.ia == Access::eConsume || pif.pass->type == PassType::eForcedAccess) {
						r_io.undef = { pass_idx, res_idx };
						is_undef = true;
						if (is_def) {
							r_io.next = &res_to_links[res.out_name];
						}
					}
				}
				if (is_def) {
					auto& w_io = res_to_links[res.out_name];
					w_io.def = { pass_idx, res_idx };
					w_io.type = res.type;
					if (is_undef) {
						w_io.prev = &res_to_links[res.name];
					}
				}
			}
		}

		// introduce chain links with inputs (acquired and attached buffers & images) and outputs (releases)
		// these are denoted with negative indices
		for (auto& bound : bound_attachments) {
			res_to_links[bound.name].def = { .pass = static_cast<int32_t>(-1 * (&bound - &*bound_attachments.begin() + 1)) };
		}

		for (auto& bound : bound_buffers) {
			res_to_links[bound.name].def = { .pass = static_cast<int32_t>(-1 * (&bound - &*bound_buffers.begin() + 1)) };
		}

		for (auto& bound : releases) {
			res_to_links[bound.first].undef = { .pass = static_cast<int32_t>(-1 * (&bound - &*releases.begin() + 1)) };
		}

		// collect chains by looking at links without a prev
		for (auto& [name, link] : res_to_links) {
			if (!link.prev) {
				chains.push_back(&link);
			}
		}

		// diagnose unheaded chains at this point
		for (auto& chp : chains) {
			assert(chp->def && "Unheaded chain.");
			// return { expected_error, errors::make_unattached_resource_exception(passinfo, res, undiverged_name) };
		}

		// calculate indegrees for all passes & build adjacency
		std::vector<size_t> indegrees(passes.size());
		std::vector<uint8_t> adjacency_matrix(passes.size() * passes.size());
		for (auto& [qfname, link] : res_to_links) {
			if (link.undef && (link.undef->pass >= 0) &&
			    ((link.def && link.def->pass >= 0) || link.reads.size() > 0)) { // we only care about an undef if the def or reads are in the graph
				indegrees[link.undef->pass]++;
				if (link.def) {
					adjacency_matrix[link.def->pass * passes.size() + link.undef->pass]++; // def -> undef
				}
			}
			for (auto& read : link.reads.to_span(pass_idx_helper)) {
				if ((link.def && link.def->pass >= 0)) {
					indegrees[read.pass]++;                                         // this only counts as a dep if there is a def before
					adjacency_matrix[link.def->pass * passes.size() + read.pass]++; // def -> read
				}
				if ((link.undef && link.undef->pass >= 0)) {
					indegrees[link.undef->pass]++;
					adjacency_matrix[read.pass * passes.size() + link.undef->pass]++; // read -> undef
				}
			}
		}

		// enqueue all indegree == 0 passes
		std::vector<size_t> process_queue;
		for (auto i = 0; i < indegrees.size(); i++) {
			if (indegrees[i] == 0)
				process_queue.push_back(i);
		}
		// dequeue indegree = 0 pass, add it to the ordered list, then decrement adjacent pass indegrees and push indegree == 0 to queue
		computed_pass_idx_to_ordered_idx.resize(passes.size());
		ordered_idx_to_computed_pass_idx.resize(passes.size());
		while (process_queue.size() > 0) {
			auto pop_idx = process_queue.back();
			computed_pass_idx_to_ordered_idx[pop_idx] = ordered_passes.size();
			ordered_idx_to_computed_pass_idx[ordered_passes.size()] = pop_idx;
			ordered_passes.emplace_back(&passes[pop_idx]);
			process_queue.pop_back();
			for (auto i = 0; i < passes.size(); i++) { // all the outgoing from this pass
				if (i == pop_idx) {
					continue;
				}
				auto adj_value = adjacency_matrix[pop_idx * passes.size() + i];
				if (adj_value > 0) {
					if (indegrees[i] -= adj_value; indegrees[i] == 0) {
						process_queue.push_back(i);
					}
				}
			}
		}
		assert(ordered_passes.size() == passes.size());

		// subchain fixup pass

		// connect subchains
		// diverging subchains are chains where def->pass >= 0 AND def->pass type is eDiverge
		// reconverged subchains are chains where def->pass >= 0 AND def->pass type is eConverge
		for (auto& head : chains) {
			if (head->def->pass >= 0 && head->type == Resource::Type::eImage) { // no Buffer divergence
				auto& pass = get_pass(*head->def);
				if (pass.pass->type == PassType::eDiverge) { // diverging subchain
					auto& whole_res = pass.resources.to_span(resources)[0].name;
					head->source = &res_to_links[whole_res];
				} else if (pass.pass->type == PassType::eConverge) { // reconverged subchain
					// take first resource which guaranteed to be diverged
					auto div_resources = pass.resources.to_span(resources).subspan(1);
					for (auto& res : div_resources) {
						res_to_links[res.name].destination = head;
					}
				}
			}
		}

		helper_links.resize(100);
		// fixup diverge subchains by copying first use on the converge subchain to their end
		for (auto& head : chains) {
			if (head->source) { // a diverged subchain
				// seek to the end
				ChainLink* chain;
				for (chain = head; chain->destination == nullptr; chain = chain->next)
					;
				auto last_chain_on_diverged = chain;
				auto first_chain_on_converged = chain->destination;
				if (first_chain_on_converged->reads.size() > 0) { // first use is reads
					// take the reads
					for (auto& r : first_chain_on_converged->reads.to_span(pass_idx_helper)) {
						last_chain_on_diverged->reads.append(pass_idx_helper, r);
					}
					// remove the undef from the diverged chain
					last_chain_on_diverged->undef = {};
				} else if (first_chain_on_converged->undef) {
					// take the undef from the converged chain and put it on the diverged chain undef
					last_chain_on_diverged->undef = first_chain_on_converged->undef;
					// we need to add a release to nothing to preserve the last undef
					ChainLink helper;
					helper.prev = last_chain_on_diverged;
					helper.def = last_chain_on_diverged->undef;
					helper.type = last_chain_on_diverged->type;
					auto& rel = releases.emplace_back(QualifiedName{}, Release{});
					helper.undef = { .pass = static_cast<int32_t>(-1 * (&rel - &*releases.begin() + 1)) };
					auto& new_link = helper_links.emplace_back(helper);
					last_chain_on_diverged->next = &new_link;
				}
			}
		}

		// take all diverged subchains and replace their def with a new attachment that has the proper acq and subrange
		for (auto& head : chains) {
			if (head->def->pass >= 0 && head->type == Resource::Type::eImage) { // no Buffer divergence
				auto& pass = get_pass(*head->def);
				if (pass.pass->type == PassType::eDiverge) { // diverging subchain
					// whole resource is always first resource
					auto& whole_res = pass.resources.to_span(resources)[0].name;
					auto link = &res_to_links[whole_res];
					while (link->prev) { // seek to the head of the original chain
						link = link->prev;
					}
					auto att = get_bound_attachment(link->def->pass); // the original chain attachment, make copy
					auto& our_res = get_resource(*head->def);

					att.image_subrange = diverged_subchain_headers.at(our_res.out_name).second; // look up subrange referenced by this subchain
					att.name = QualifiedName{ Name{}, att.name.name.append(our_res.out_name.name.to_sv()) };
					att.parent_attachment = link->def->pass;
					auto new_bound = bound_attachments.emplace_back(att);
					// replace def with new attachment
					head->def = { .pass = static_cast<int32_t>(-1 * bound_attachments.size()) };
				}
			}
		}

		// fixup converge subchains by removing the first use
		for (auto& head : chains) {
			if (head->def->pass >= 0 && head->type == Resource::Type::eImage) { // no Buffer divergence
				auto& pass = get_pass(*head->def);
				if (pass.pass->type == PassType::eConverge) { // converge subchain
					if (head->reads.size() > 0) {               // first use is reads
						head->reads = {};                         // remove the reads
					} else if (head->undef) {                   // first use is undef
						head = head->next;                        // drop link from chain
						head->prev = nullptr;
					}
					// reconverged resource is always first resource
					auto& whole_res = pass.resources.to_span(resources)[0];
					// take first resource which guaranteed to be diverged
					auto div_resources = pass.resources.to_span(resources).subspan(1);
					auto& div_res = div_resources[1];
					// TODO: we actually need to walk all converging resources here to find the scope of the convergence
					// walk this resource to convergence
					auto link = &res_to_links[div_res.name];
					while (link->prev) { // seek to the head of the diverged chain
						link = link->prev;
					}
					assert(link->source);
					link = link->source;
					head->source = link; // set the source for this subchain the original undiv chain
					while (link->prev) { // seek to the head of the original chain
						link = link->prev;
					}
					auto whole_att = get_bound_attachment(link->def->pass); // the whole attachment
					whole_att.name = QualifiedName{ Name{}, whole_att.name.name.append(whole_res.out_name.name.to_sv()) };
					whole_att.acquire.unsynchronized = true;
					whole_att.parent_attachment = link->def->pass;
					auto new_bound = bound_attachments.emplace_back(whole_att);
					// replace head->def with new attachments (the converged resource)
					head->def = { .pass = static_cast<int32_t>(-1 * bound_attachments.size()) };
				}
			}
		}

		// TODO: validate incorrect convergence
		/* else if (chain.back().high_level_access == Access::eConverge) {
		  assert(it->second.dst_use.layout != ImageLayout::eUndefined); // convergence into no use = disallowed
		}*/
		// return { expected_error, errors::make_unattached_resource_exception(passinfo, res, undiverged_name) };
	}

	std::string Compiler::dump_graph() {
		std::stringstream ss;
		/* ss << "digraph vuk {\n";
		for (auto i = 0; i < impl->computed_passes.size(); i++) {
		  for (auto j = 0; j < impl->computed_passes.size(); j++) {
		    if (i == j)
		      continue;
		    auto& p1 = impl->computed_passes[i];
		    auto& p2 = impl->computed_passes[j];
		    for (auto& o : p1.output_names.to_span(impl->output_names)) {
		      for (auto& i : p2.input_names.to_span(impl->input_names)) {
		        if (o == impl->resolve_alias(i)) {
		          ss << "\"" << p1.pass->name.c_str() << "\" -> \"" << p2.pass->name.c_str() << "\" [label=\"" << impl->resolve_alias(i).name.c_str() << "\"];\n";
		          // p2 is ordered after p1
		        }
		      }
		    }
		    for (auto& o : p1.input_names.to_span(impl->input_names)) {
		      for (auto& i : p2.write_input_names.to_span(impl->write_input_names)) {
		        if (impl->resolve_alias(o) == impl->resolve_alias(i)) {
		          ss << "\"" << p1.pass->name.c_str() << "\" -> \"" << p2.pass->name.c_str() << "\" [label=\"" << impl->resolve_alias(i).name.c_str() << "\"];\n";
		          // p2 is ordered after p1
		        }
		      }
		    }
		  }
		}
		ss << "}\n";*/
		return ss.str();
	}

	std::unordered_map<RenderGraph*, std::string> RGCImpl::compute_prefixes(const RenderGraph& rg, bool do_prefix) {
		std::unordered_map<RenderGraph*, std::string> sg_prefixes;
		for (auto& [sg_ptr, sg_info] : rg.impl->subgraphs) {
			if (sg_info.count > 0) {
				Name sg_name = sg_ptr->name;
				assert(sg_ptr->impl);

				auto prefixes = compute_prefixes(*sg_ptr, true);
				sg_prefixes.merge(prefixes);
				if (auto& counter = ++sg_name_counter[sg_name]; counter > 1) {
					sg_name = sg_name.append(std::string("_") + std::to_string(counter - 1));
				}
				sg_prefixes.emplace(sg_ptr.get(), std::string(sg_name.to_sv()));
			}
		}
		if (do_prefix && sg_prefixes.size() > 0) {
			for (auto& [k, v] : sg_prefixes) {
				v = std::string(rg.name.to_sv()) + "::" + v;
			}
		}
		return sg_prefixes;
	}

	void RGCImpl::inline_subgraphs(const std::shared_ptr<RenderGraph>& rg, std::unordered_set<std::shared_ptr<RenderGraph>>& consumed_rgs) {
		auto our_prefix = sg_prefixes.at(rg.get());
		for (auto& [sg_ptr, sg_info] : rg->impl->subgraphs) {
			if (sg_info.count > 0) {
				auto& prefix = sg_prefixes.at(sg_ptr.get());
				assert(sg_ptr->impl);
				for (auto& [name_in_parent, name_in_sg] : sg_info.exported_names) {
					auto old_name = QualifiedName{ Name(prefix), name_in_sg };
					auto new_name = QualifiedName{ our_prefix.empty() ? Name{} : Name(our_prefix), name_in_parent };
					computed_aliases[new_name] = old_name;
				}
				if (!consumed_rgs.contains(sg_ptr)) {
					inline_subgraphs(sg_ptr, consumed_rgs);
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

	Result<void> Compiler::inline_rgs(std::span<std::shared_ptr<RenderGraph>> rgs) {
		// inline all the subgraphs into us
		for (auto& rg : rgs) {
			impl->sg_prefixes.merge(impl->compute_prefixes(*rg, true));
			auto rg_name = rg->name;
			if (auto& counter = ++impl->sg_name_counter[rg_name]; counter > 1) {
				rg_name = rg_name.append(std::string("_") + std::to_string(counter - 1));
			}
			impl->sg_prefixes.emplace(rg.get(), std::string{ rg_name.c_str() });
			std::unordered_set<std::shared_ptr<RenderGraph>> consumed_rgs = {};
			impl->inline_subgraphs(rg, consumed_rgs);
		}

		for (auto& rg : rgs) {
			impl->append(Name{ impl->sg_prefixes.at(rg.get()).c_str() }, *rg);
		}

		return { expected_value };
	}

	void RGCImpl::compute_assigned_names_1(robin_hood::unordered_flat_map<QualifiedName, QualifiedName>& name_map) {
		name_map.insert(computed_aliases.begin(), computed_aliases.end());

		computed_aliases.clear();
		// follow aliases and resolve them into a single lookup
		for (auto& [k, v] : name_map) {
			auto it = name_map.find(v);
			auto res = v;
			while (it != name_map.end()) {
				res = it->second;
				it = name_map.find(res);
			}
			assert(!res.is_invalid());
			computed_aliases.emplace(k, res);
		}

		for (auto& passinfo : computed_passes) {
			for (auto& res : passinfo.resources.to_span(resources)) {
				// for read or write, we add source to use chain
				if (!res.name.is_invalid() && !res.out_name.is_invalid()) {
					auto [iter, succ] = name_map.emplace(res.out_name, res.name);
					assert(iter->second == res.name);
				}
			}
		}

		assigned_names.clear();
		// populate resource name -> use chain map
		for (auto& [k, v] : name_map) {
			auto it = name_map.find(v);
			auto res = v;
			while (it != name_map.end()) {
				res = it->second;
				it = name_map.find(res);
			}
			assert(!res.is_invalid());
			assigned_names.emplace(k, res);
		}
	}

	void Compiler::queue_inference() {
		// queue inference pass
		// prepopulate run domain with requested domain
		for (auto& p : impl->computed_passes) {
			p.domain = p.pass->execute_on;
		}

		for (auto& head : impl->chains) {
			DomainFlags last_domain = DomainFlagBits::eDevice;
			bool is_image = head->type == Resource::Type::eImage;
			auto propagate_domain = [&last_domain](auto& domain) {
				if (domain != last_domain && domain != DomainFlagBits::eDevice && domain != DomainFlagBits::eAny) {
					last_domain = domain;
				}
				if ((last_domain != DomainFlagBits::eDevice && last_domain != DomainFlagBits::eAny) &&
				    (domain == DomainFlagBits::eDevice || domain == DomainFlagBits::eAny)) {
					domain = last_domain;
				}
			};

			// forward inference
			ChainLink* chain;
			for (chain = head; chain != nullptr; chain = chain->next) {
				if (chain->def->pass >= 0) {
					propagate_domain(impl->get_pass(*chain->def).domain);
				} else {
					DomainFlagBits att_dom;
					if (is_image) {
						att_dom = impl->get_bound_attachment(chain->def->pass).acquire.initial_domain;
					} else {
						att_dom = impl->get_bound_buffer(chain->def->pass).acquire.initial_domain;
					}
					if (att_dom != last_domain && att_dom != DomainFlagBits::eDevice && att_dom != DomainFlagBits::eAny) {
						last_domain = att_dom;
					}
				}
				for (auto& r : chain->reads.to_span(impl->pass_idx_helper)) {
					propagate_domain(impl->get_pass(r).domain);
				}
				if (chain->undef && chain->undef->pass >= 0) {
					propagate_domain(impl->get_pass(*chain->undef).domain);
				}
			}
		}

		// backward inference
		for (auto& head : impl->chains) {
			DomainFlags last_domain = DomainFlagBits::eDevice;
			// queue inference pass
			auto propagate_domain = [&last_domain](auto& domain) {
				if (domain != last_domain && domain != DomainFlagBits::eDevice && domain != DomainFlagBits::eAny) {
					last_domain = domain;
				}
				if ((last_domain != DomainFlagBits::eDevice && last_domain != DomainFlagBits::eAny) &&
				    (domain == DomainFlagBits::eDevice || domain == DomainFlagBits::eAny)) {
					domain = last_domain;
				}
			};

			ChainLink* chain;
			// wind chain to the end
			for (chain = head; chain->next != nullptr; chain = chain->next)
				;
			for (; chain != nullptr; chain = chain->prev) {
				if (chain->undef) {
					if (chain->undef->pass < 0) {
						last_domain = impl->get_release(chain->undef->pass).dst_use.domain;
					} else {
						propagate_domain(impl->get_pass(*chain->undef).domain);
					}
				}
				for (auto& r : chain->reads.to_span(impl->pass_idx_helper)) {
					propagate_domain(impl->get_pass(r).domain);
				}
				if (chain->def->pass >= 0) {
					propagate_domain(impl->get_pass(*chain->def).domain);
				}
			}
		}

		// queue inference failure fixup pass
		for (auto& p : impl->ordered_passes) {
			if (p->domain == DomainFlagBits::eDevice || p->domain == DomainFlagBits::eAny) { // couldn't infer, set pass as graphics
				p->domain = DomainFlagBits::eGraphicsQueue;
			}
		}
	}
	
	// partition passes into different queues
	void Compiler::pass_partitioning() {
		impl->partitioned_passes.reserve(impl->ordered_passes.size());
		impl->computed_pass_idx_to_partitioned_idx.resize(impl->ordered_passes.size());
		for (size_t i = 0; i < impl->ordered_passes.size(); i++) {
			auto& p = impl->ordered_passes[i];
			if (p->domain & DomainFlagBits::eTransferQueue) {
				impl->computed_pass_idx_to_partitioned_idx[impl->ordered_idx_to_computed_pass_idx[i]] = impl->partitioned_passes.size();
				impl->partitioned_passes.push_back(p);
			}
		}
		impl->transfer_passes = { impl->partitioned_passes.begin(), impl->partitioned_passes.size() };
		for (size_t i = 0; i < impl->ordered_passes.size(); i++) {
			auto& p = impl->ordered_passes[i];
			if (p->domain & DomainFlagBits::eComputeQueue) {
				impl->computed_pass_idx_to_partitioned_idx[impl->ordered_idx_to_computed_pass_idx[i]] = impl->partitioned_passes.size();
				impl->partitioned_passes.push_back(p);
			}
		}
		impl->compute_passes = { impl->partitioned_passes.begin() + impl->transfer_passes.size(), impl->partitioned_passes.size() - impl->transfer_passes.size() };
		for (size_t i = 0; i < impl->ordered_passes.size(); i++) {
			auto& p = impl->ordered_passes[i];
			if (p->domain & DomainFlagBits::eGraphicsQueue) {
				impl->computed_pass_idx_to_partitioned_idx[impl->ordered_idx_to_computed_pass_idx[i]] = impl->partitioned_passes.size();
				impl->partitioned_passes.push_back(p);
			}
		}
		impl->graphics_passes = { impl->partitioned_passes.begin() + impl->transfer_passes.size() + impl->compute_passes.size(),
			                        impl->partitioned_passes.size() - impl->transfer_passes.size() - impl->compute_passes.size() };
	}

	// resource linking pass
	// populate swapchain and resource -> bound references
	void Compiler::resource_linking() {
		for (auto head : impl->chains) {
			bool is_image = head->type == Resource::Type::eImage;
			bool is_swapchain = false;
			if (is_image) {
				auto& att = impl->get_bound_attachment(head->def->pass);
				att.use_chains.append(impl->attachment_use_chain_references, head);
				is_swapchain = att.type == AttachmentInfo::Type::eSwapchain;
			} else {
				auto& att = impl->get_bound_buffer(head->def->pass);
				att.use_chains.append(impl->attachment_use_chain_references, head);
			}

			if (head->source) { // propagate use onto previous chain def
				ChainLink* link = head;

				while (link->source) {
					for (link = link->source; link->def->pass > 0; link = link->prev)
						;
				}
				for (; link->def->pass >= 0; link = link->prev)
					;

				if (link->type == Resource::Type::eImage) {
					auto& att = impl->get_bound_attachment(link->def->pass);
					att.use_chains.append(impl->attachment_use_chain_references, head);
					is_swapchain = att.type == AttachmentInfo::Type::eSwapchain;
				} else {
					auto& att = impl->get_bound_buffer(link->def->pass);
					att.use_chains.append(impl->attachment_use_chain_references, head);
				}
			}

			for (ChainLink* link = head; link != nullptr; link = link->next) {
				if (link->def->pass >= 0) {
					auto& pass = impl->get_pass(*link->def);
					auto& def_res = impl->get_resource(*link->def);
					if (is_swapchain) {
						pass.referenced_swapchains.append(impl->swapchain_references, head->def->pass);
					}
					def_res.reference = head->def->pass;
				}
				for (auto& r : link->reads.to_span(impl->pass_idx_helper)) {
					auto& pass = impl->get_pass(r);
					auto& def_res = impl->get_resource(r);
					if (is_swapchain) {
						pass.referenced_swapchains.append(impl->swapchain_references, head->def->pass);
					}
					def_res.reference = head->def->pass;
				}
				if (link->undef && link->undef->pass >= 0) {
					auto& pass = impl->get_pass(*link->undef);
					auto& def_res = impl->get_resource(*link->undef);
					if (is_swapchain) {
						pass.referenced_swapchains.append(impl->swapchain_references, head->def->pass);
					}
					def_res.reference = head->def->pass;
				}
			}
		}
	}

	void Compiler::renderpass_assignment() {
		// graphics: assemble renderpasses based on framebuffers
		// we need to collect passes into framebuffers, which will determine the renderpasses
		using attachment_set = std::unordered_set<Resource, std::hash<Resource>, std::equal_to<Resource>, short_alloc<Resource, 16>>;
		using passinfo_vec = std::vector<PassInfo*, short_alloc<PassInfo*, 16>>;
		std::vector<std::pair<attachment_set, passinfo_vec>, short_alloc<std::pair<attachment_set, passinfo_vec>, 8>> attachment_sets{ *impl->arena_ };
		for (auto& passinfo : impl->graphics_passes) {
			attachment_set atts{ *impl->arena_ };

			for (auto& res : passinfo->resources.to_span(impl->resources)) {
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

		// renderpasses are uniquely identified by their index from now on
		// tell passes in which renderpass/subpass they will execute
		impl->rpis.reserve(attachment_sets.size());
		for (auto& [attachments, passes] : attachment_sets) {
			RenderPassInfo rpi{ *impl->arena_ };
			auto rpi_index = impl->rpis.size();

			int32_t subpass = -1;

			if (attachments.size() == 0) {
				continue;
			} else {
				for (auto& p : passes) {
					p->render_pass_index = (int32_t)rpi_index;
					p->subpass = ++subpass;
				}
				for (auto& atts : attachments) {
					auto& bound_att = impl->get_bound_attachment(atts.reference);
					auto& att = rpi.attachments.emplace_back(AttachmentRPInfo{ &bound_att });
					att.description.loadOp = VK_ATTACHMENT_LOAD_OP_LOAD;
					att.description.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
				}
			}

			impl->rpis.push_back(rpi);
		}
	}

	Result<void> Compiler::compile(std::span<std::shared_ptr<RenderGraph>> rgs, const RenderGraphCompileOptions& compile_options) {
		auto arena = impl->arena_.release();
		delete impl;
		arena->reset();
		impl = new RGCImpl(arena);

		VUK_DO_OR_RETURN(inline_rgs(rgs));

		// gather name alias info now - once we partition, we might encounter unresolved aliases
		robin_hood::unordered_flat_map<QualifiedName, QualifiedName> name_map;

		impl->compute_assigned_names_1(name_map);

		impl->merge_diverge_passes(impl->computed_passes);

		// run global pass ordering - once we split per-queue we don't see enough
		// inputs to order within a queue
		impl->schedule_intra_queue(impl->computed_passes, compile_options);

		// auto dumped_graph = dump_graph();

		// TODO: inference code relies on assigned names
		impl->assigned_names.clear();
		// populate resource name -> use chain map
		for (auto& [k, v] : name_map) {
			auto it = name_map.find(v);
			auto res = v;
			while (it != name_map.end()) {
				res = it->second;
				it = name_map.find(res);
			}
			assert(!res.is_invalid());
			impl->assigned_names.emplace(k, res);
		}

		queue_inference();
		pass_partitioning();
		resource_linking();
		renderpass_assignment();

		return { expected_value };
	}

	void RenderGraph::resolve_resource_into(Name resolved_name_src, Name resolved_name_dst, Name ms_name) {
		add_pass({ .resources = { Resource{ ms_name, Resource::Type::eImage, eTransferRead, {} },
		                          Resource{ resolved_name_src, Resource::Type::eImage, eTransferWrite, resolved_name_dst } },
		           .execute = [ms_name, resolved_name_src](CommandBuffer& cbuf) { cbuf.resolve_image(ms_name, resolved_name_src); },
		           .type = PassType::eResolve });
		inference_rule(resolved_name_src, same_shape_as(ms_name));
		inference_rule(ms_name, same_shape_as(resolved_name_src));
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
		attachment_info.name = { Name{}, name };
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

		QueueResourceUse& initial = attachment_info.acquire.src_use;
		// for WSI, we want to wait for colourattachmentoutput
		// we don't care about any writes, we will clear
		initial.access = AccessFlags{};
		initial.stages = PipelineStageFlagBits::eColorAttachmentOutput;
		// discard
		initial.layout = ImageLayout::eUndefined;

		impl->bound_attachments.emplace(attachment_info.name, attachment_info);
	}

	void RenderGraph::attach_buffer(Name name, Buffer buf, Access initial) {
		BufferInfo buf_info{ .name = { Name{}, name }, .buffer = buf, .acquire = { .src_use = to_use(initial, DomainFlagBits::eAny) } };
		impl->bound_buffers.emplace(buf_info.name, buf_info);
	}

	void RenderGraph::attach_buffer_from_allocator(Name name, Buffer buf, Allocator allocator, Access initial) {
		BufferInfo buf_info{ .name = { Name{}, name }, .buffer = buf, .acquire = { .src_use = to_use(initial, DomainFlagBits::eAny) }, .allocator = allocator };
		impl->bound_buffers.emplace(buf_info.name, buf_info);
	}

	void RenderGraph::attach_image(Name name, ImageAttachment att, Access initial_acc) {
		AttachmentInfo attachment_info;
		attachment_info.name = { Name{}, name };
		attachment_info.attachment = att;
		if (att.has_concrete_image() && att.has_concrete_image_view()) {
			attachment_info.type = AttachmentInfo::Type::eExternal;
		} else {
			attachment_info.type = AttachmentInfo::Type::eInternal;
		}
		attachment_info.attachment.format = att.format;

		QueueResourceUse& initial = attachment_info.acquire.src_use;
		initial = to_use(initial_acc, DomainFlagBits::eAny);
		impl->bound_attachments.emplace(attachment_info.name, attachment_info);
	}

	void RenderGraph::attach_image_from_allocator(Name name, ImageAttachment att, Allocator allocator, Access initial_acc) {
		AttachmentInfo attachment_info;
		attachment_info.allocator = allocator;
		attachment_info.name = { Name{}, name };
		attachment_info.attachment = att;
		if (att.has_concrete_image() && att.has_concrete_image_view()) {
			attachment_info.type = AttachmentInfo::Type::eExternal;
		} else {
			attachment_info.type = AttachmentInfo::Type::eInternal;
		}
		attachment_info.attachment.format = att.format;

		QueueResourceUse& initial = attachment_info.acquire.src_use;
		initial = to_use(initial_acc, DomainFlagBits::eAny);
		impl->bound_attachments.emplace(attachment_info.name, attachment_info);
	}

	void RenderGraph::attach_and_clear_image(Name name, ImageAttachment att, Clear clear_value, Access initial_acc) {
		Name tmp_name = name.append(get_temporary_name().to_sv());
		attach_image(tmp_name, att, initial_acc);
		clear_image(tmp_name, name, clear_value);
	}

	void RenderGraph::attach_in(Name name, Future fimg) {
		if (fimg.get_status() == FutureBase::Status::eSubmitted || fimg.get_status() == FutureBase::Status::eHostAvailable) {
			if (fimg.is_image()) {
				auto att = fimg.get_result<ImageAttachment>();
				AttachmentInfo attachment_info;
				attachment_info.name = { Name{}, name };
				attachment_info.attachment = att;

				attachment_info.type = AttachmentInfo::Type::eExternal;

				attachment_info.acquire = { fimg.control->last_use, fimg.control->initial_domain, fimg.control->initial_visibility };

				impl->bound_attachments.emplace(attachment_info.name, attachment_info);
			} else {
				BufferInfo buf_info{ .name = { Name{}, name }, .buffer = fimg.get_result<Buffer>() };
				buf_info.acquire = { fimg.control->last_use, fimg.control->initial_domain, fimg.control->initial_visibility };
				impl->bound_buffers.emplace(buf_info.name, buf_info);
			}
			impl->imported_names.emplace_back(name);
		} else if (fimg.get_status() == FutureBase::Status::eInitial) {
			// an unsubmitted RG is being attached, we remove the release from that RG, and we allow the name to be found in us
			assert(fimg.rg->impl);
			std::erase_if(fimg.rg->impl->releases, [name = fimg.get_bound_name()](auto& item) { return item.first.name == name; });
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
		impl->ia_inference_rules.emplace_back(IAInference{ QualifiedName{ Name{}, target }, std::move(rule) });
	}

	void RenderGraph::inference_rule(Name target, std::function<void(const struct InferenceContext&, Buffer&)> rule) {
		impl->buf_inference_rules.emplace_back(BufferInference{ QualifiedName{ Name{}, target }, std::move(rule) });
	}

	robin_hood::unordered_flat_set<Name> RGImpl::get_available_resources() {
		auto pass_infos = build_io(passes);
		// seed the available names with the names we imported from subgraphs
		robin_hood::unordered_flat_set<Name> outputs;
		outputs.insert(imported_names.begin(), imported_names.end());

		for (auto& [name, _] : bound_attachments) {
			outputs.insert(name.name);
		}
		for (auto& [name, _] : bound_buffers) {
			outputs.insert(name.name);
		}

		for (auto& pif : pass_infos) {
			for (auto& in : pif.input_names.to_span(input_names)) {
				outputs.erase(in.name);
			}
			for (auto& in : pif.write_input_names.to_span(write_input_names)) {
				outputs.erase(in.name);
			}
			for (auto& in : pif.output_names.to_span(output_names)) {
				outputs.emplace(in.name);
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
		impl->releases.emplace_back(QualifiedName{ Name{}, name }, Release{ to_use(Access::eNone, dst_domain), fimg.control.get() });
	}

	void RenderGraph::detach_out(Name name, Future& fimg) {
		for (auto it = impl->releases.begin(); it != impl->releases.end(); ++it) {
			if (it->first.name == name && it->second.signal == fimg.control.get()) {
				impl->releases.erase(it);
				return;
			}
		}
	}

	void RenderGraph::release(Name name, Access final) {
		impl->releases.emplace_back(QualifiedName{ Name{}, name }, Release{ to_use(final, DomainFlagBits::eAny) });
	}

	void RenderGraph::release_for_present(Name name) {
		/*
		Normally, we would need an external dependency at the end as well since
		we are changing layout in finalLayout, but since we are signalling a
		semaphore, we can rely on Vulkan's default behavior, which injects an
		external dependency here with dstStageMask =
		VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, dstAccessMask = 0.
		*/
		QueueResourceUse final{
			.stages = PipelineStageFlagBits::eAllCommands, .access = AccessFlagBits{}, .layout = ImageLayout::ePresentSrcKHR, .domain = DomainFlagBits::eAny
		};
		impl->releases.emplace_back(QualifiedName{ Name{}, name }, Release{ .dst_use = final });
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
			if (src.view_type != ImageViewType::eInfer)
				ia.view_type = src.view_type;
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

	BufferRule same_size_as(Name inference_source) {
		return [=](const InferenceContext& ctx, Buffer& buf) {
			auto& src = ctx.get_buffer(inference_source);
			buf.size = src.size;
		};
	}

	bool crosses_queue(QueueResourceUse last_use, QueueResourceUse current_use) {
		return (last_use.domain != DomainFlagBits::eNone && last_use.domain != DomainFlagBits::eAny && current_use.domain != DomainFlagBits::eNone &&
		        current_use.domain != DomainFlagBits::eAny && (last_use.domain & DomainFlagBits::eQueueMask) != (current_use.domain & DomainFlagBits::eQueueMask));
	}

	void RGCImpl::emit_image_barrier(RelSpan<VkImageMemoryBarrier2KHR>& barriers,
	                                 int32_t bound_attachment,
	                                 QueueResourceUse last_use,
	                                 QueueResourceUse current_use,
	                                 Subrange::Image& subrange,
	                                 ImageAspectFlags aspect,
	                                 bool is_release) {
		scope_to_domain((VkPipelineStageFlagBits2KHR&)last_use.stages, is_release ? last_use.domain : current_use.domain & DomainFlagBits::eQueueMask);
		scope_to_domain((VkPipelineStageFlagBits2KHR&)current_use.stages, is_release ? last_use.domain : current_use.domain & DomainFlagBits::eQueueMask);

		// compute image barrier for this access -> access
		VkImageMemoryBarrier2KHR barrier{ .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2_KHR };
		barrier.srcAccessMask = is_read_access(last_use) ? 0 : (VkAccessFlags)last_use.access;
		barrier.dstAccessMask = (VkAccessFlags)current_use.access;
		barrier.oldLayout = (VkImageLayout)last_use.layout;
		barrier.newLayout = (VkImageLayout)current_use.layout;
		barrier.subresourceRange.aspectMask = (VkImageAspectFlags)aspect;
		barrier.subresourceRange.baseArrayLayer = subrange.base_layer;
		barrier.subresourceRange.baseMipLevel = subrange.base_level;
		barrier.subresourceRange.layerCount = subrange.layer_count;
		barrier.subresourceRange.levelCount = subrange.level_count;
		assert(last_use.domain.m_mask != 0);
		assert(current_use.domain.m_mask != 0);
		if (last_use.domain == DomainFlagBits::eAny) {
			last_use.domain = current_use.domain;
		}
		if (current_use.domain == DomainFlagBits::eAny) {
			current_use.domain = last_use.domain;
		}
		barrier.srcQueueFamilyIndex = static_cast<uint32_t>((last_use.domain & DomainFlagBits::eQueueMask).m_mask);
		barrier.dstQueueFamilyIndex = static_cast<uint32_t>((current_use.domain & DomainFlagBits::eQueueMask).m_mask);

		if (last_use.stages == PipelineStageFlags{}) {
			barrier.srcAccessMask = {};
		}
		if (current_use.stages == PipelineStageFlags{}) {
			barrier.dstAccessMask = {};
		}

		std::memcpy(&barrier.pNext, &bound_attachment, sizeof(int32_t));
		barrier.srcStageMask = (VkPipelineStageFlags2)last_use.stages.m_mask;
		barrier.dstStageMask = (VkPipelineStageFlags2)current_use.stages.m_mask;

		barriers.append(image_barriers, barrier);
	}

	void RGCImpl::emit_memory_barrier(RelSpan<VkMemoryBarrier2KHR>& barriers, QueueResourceUse last_use, QueueResourceUse current_use) {
		scope_to_domain((VkPipelineStageFlagBits2KHR&)last_use.stages, last_use.domain & DomainFlagBits::eQueueMask);
		scope_to_domain((VkPipelineStageFlagBits2KHR&)current_use.stages, current_use.domain & DomainFlagBits::eQueueMask);

		VkMemoryBarrier2KHR barrier{ .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2_KHR };
		barrier.srcAccessMask = is_read_access(last_use) ? 0 : (VkAccessFlags)last_use.access;
		barrier.dstAccessMask = (VkAccessFlags)current_use.access;
		barrier.srcStageMask = (VkPipelineStageFlagBits2)last_use.stages.m_mask;
		barrier.dstStageMask = (VkPipelineStageFlagBits2)current_use.stages.m_mask;
		if (barrier.srcStageMask == 0) {
			barrier.srcStageMask = (VkPipelineStageFlagBits2)PipelineStageFlagBits::eNone;
			barrier.srcAccessMask = {};
		}

		barriers.append(mem_barriers, barrier);
	}

	Result<ExecutableRenderGraph> Compiler::link(std::span<std::shared_ptr<RenderGraph>> rgs, const RenderGraphCompileOptions& compile_options) {
		VUK_DO_OR_RETURN(compile(rgs, compile_options));

		// we need to handle chains in order of dependency
		std::vector<ChainLink*> work_queue;
		for (auto head : impl->chains) {
			bool is_image = head->type == Resource::Type::eImage;
			if (is_image) {
				if (!head->source) {
					work_queue.push_back(head);
				}
			} else {
				work_queue.push_back(head);
			}
		}

		// handle head (queue wait, initial use) -> emit barriers -> handle tail (signal, final use)
		while (work_queue.size() > 0) {
			ChainLink* head = work_queue.back();
			work_queue.pop_back();
			// handle head
			ImageAspectFlags aspect;
			Subrange::Image image_subrange;
			bool is_image = head->type == Resource::Type::eImage;
			if (is_image) {
				auto& att = impl->get_bound_attachment(head->def->pass);
				aspect = format_to_aspect(att.attachment.format);
				image_subrange = att.image_subrange;
			}
			bool is_subchain = head->source != nullptr;

			// initial waits are handled by the common chain code
			// last use on the chain
			QueueResourceUse last_use;
			ChainLink* link;
			for (link = head; link != nullptr; link = link->next) {
				// populate last use from def or attach
				if (link->def->pass >= 0) {
					auto& def_pass = impl->get_pass(*link->def);
					auto& def_res = impl->get_resource(*link->def);
					last_use = to_use(def_res.ia, def_pass.domain);
				} else {
					last_use = is_image ? impl->get_bound_attachment(head->def->pass).acquire.src_use : impl->get_bound_buffer(head->def->pass).acquire.src_use;
				}

				// handle chain
				// we need to emit: def -> reads, RAW or nothing
				//					reads -> undef, WAR
				//					if there were no reads, then def -> undef, which is either WAR or WAW

				// we need to sometimes emit a barrier onto the last thing that happened, find that pass here
				// it is either last read pass, or if there were no reads, the def pass, or if the def pass doesn't exist, then we search parent chains
				int32_t last_executing_pass_idx = link->def->pass >= 0 ? (int32_t)impl->computed_pass_idx_to_ordered_idx[link->def->pass] : -1;

				if (link->reads.size() > 0) { // we need to emit: def -> reads, RAW or nothing (before first read)
					// to avoid R->R deps, we emit a single dep for all the reads
					// for this we compute a merged layout (TRANSFER_SRC_OPTIMAL / READ_ONLY_OPTIMAL / GENERAL)
					QueueResourceUse use;
					auto reads = link->reads.to_span(impl->pass_idx_helper);
					size_t read_idx = 0;
					size_t start_of_reads = 0;

					// this is where we stick the source dep, it is either def or undef in the parent chain
					int32_t last_use_source = link->def->pass >= 0 ? link->def->pass : -1;

					while (read_idx < reads.size()) {
						int32_t first_pass_idx = INT32_MAX;
						bool need_read_only = false;
						bool need_transfer = false;
						bool need_general = false;
						use.domain = DomainFlagBits::eNone;
						use.layout = ImageLayout::eReadOnlyOptimalKHR;
						for (; read_idx < reads.size(); read_idx++) {
							auto& r = reads[read_idx];
							auto& pass = impl->get_pass(r);
							auto& res = impl->get_resource(r);

							auto use2 = to_use(res.ia, pass.domain);
							if (use.domain == DomainFlagBits::eNone) {
								use.domain = use2.domain;
							} else if (use.domain != use2.domain) {
								// there are multiple domains in this read group
								// this is okay - but in this case we can't synchronize against all of them together
								// so we synchronize against them individually by setting last use and ending the read gather
								break;
							}
							// this read can be merged, so merge it
							int32_t order_idx = (int32_t)impl->computed_pass_idx_to_ordered_idx[r.pass];
							if (order_idx < first_pass_idx) {
								first_pass_idx = (int32_t)order_idx;
							}
							if (order_idx > last_executing_pass_idx) {
								last_executing_pass_idx = (int32_t)order_idx;
							}

							if (is_transfer_access(res.ia)) {
								need_transfer = true;
							}
							if (is_storage_access(res.ia)) {
								need_general = true;
							}
							if (is_readonly_access(res.ia)) {
								need_read_only = true;
							}

							use.access |= use2.access;
							use.stages |= use2.stages;
						}

						// compute barrier and waits for the merged reads

						if (need_transfer && !need_read_only) {
							use.layout = ImageLayout::eTransferSrcOptimal;
						}

						if (need_general || (need_transfer && need_read_only)) {
							use.layout = ImageLayout::eGeneral;
							for (auto& r : reads.subspan(start_of_reads, read_idx - start_of_reads)) {
								auto& res = impl->get_resource(r);
								res.promoted_to_general = true;
							}
						}

						if (last_use_source < 0 && is_subchain) {
							assert(link->source->undef && link->source->undef->pass >= 0);
							last_use_source = link->source->undef->pass;
						}

						// TODO: do not emit this if dep is a read and the layouts match
						auto& dst = impl->get_pass(first_pass_idx);
						if (is_image) {
							if (crosses_queue(last_use, use)) {
								impl->emit_image_barrier(impl->get_pass((int32_t)impl->computed_pass_idx_to_ordered_idx[last_use_source]).post_image_barriers,
								                         head->def->pass,
								                         last_use,
								                         use,
								                         image_subrange,
								                         aspect,
								                         true);
							}
							impl->emit_image_barrier(dst.pre_image_barriers, head->def->pass, last_use, use, image_subrange, aspect);
						} else {
							impl->emit_memory_barrier(dst.pre_memory_barriers, last_use, use);
						}
						if (crosses_queue(last_use, use)) {
							// in this case def was on a different queue the subsequent reads
							// we stick the wait on the first read pass in order
							impl->get_pass(first_pass_idx)
							    .relative_waits.append(
							        impl->waits,
							        { (DomainFlagBits)(last_use.domain & DomainFlagBits::eQueueMask).m_mask, impl->computed_pass_idx_to_ordered_idx[last_use_source] });
							impl->get_pass((int32_t)impl->computed_pass_idx_to_ordered_idx[last_use_source]).is_waited_on = true;
						}
						last_use = use;
						last_use_source = reads[read_idx - 1].pass;
						start_of_reads = read_idx;
					}
				}

				// process tails outside
				if (link->next == nullptr) {
					break;
				}

				// if there are no intervening reads, emit def -> undef, otherwise emit reads -> undef
				//  def -> undef, which is either WAR or WAW (before undef)
				//	reads -> undef, WAR (before undef)
				if (link->undef) {
					auto& pass = impl->get_pass(*link->undef);
					auto& res = impl->get_resource(*link->undef);
					QueueResourceUse use = to_use(res.ia, pass.domain);
					if (use.layout == ImageLayout::eGeneral) {
						res.promoted_to_general = true;
					}

					// handle renderpass details
					// all renderpass write-attachments are entered via an undef (because of it being a write)
					if (pass.render_pass_index >= 0) {
						auto& rpi = impl->rpis[pass.render_pass_index];
						auto& bound_att = impl->get_bound_attachment(head->def->pass);
						for (auto& att : rpi.attachments) {
							if (att.attachment_info == &bound_att) {
								// if the last use was discard, then downgrade load op
								if (last_use.layout == ImageLayout::eUndefined) {
									att.description.loadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
								} else if (use.access & AccessFlagBits::eColorAttachmentWrite) { // add CA read, because of LOAD_OP_LOAD
									use.access |= AccessFlagBits::eColorAttachmentRead;
								} else if (use.access & AccessFlagBits::eDepthStencilAttachmentWrite) { // add DSA read, because of LOAD_OP_LOAD
									use.access |= AccessFlagBits::eDepthStencilAttachmentRead;
								}
							}
						}
					}

					// if the def pass doesn't exist, and there were no reads
					if (last_executing_pass_idx == -1) {
						if (is_subchain) { // if subchain, search parent chain for pass
							assert(link->source->undef && link->source->undef->pass >= 0);
							last_executing_pass_idx = (int32_t)impl->computed_pass_idx_to_ordered_idx[link->source->undef->pass];
						}
					}

					if (is_image) {
						if (crosses_queue(last_use, use)) { // release barrier
							if (last_executing_pass_idx !=
							    -1) { // if last_executing_pass_idx is -1, then there is release in this rg, so we don't emit the release (single-sided acq)
								impl->emit_image_barrier(
								    impl->get_pass(last_executing_pass_idx).post_image_barriers, head->def->pass, last_use, use, image_subrange, aspect, true);
							}
						}
						impl->emit_image_barrier(impl->get_pass(*link->undef).pre_image_barriers, head->def->pass, last_use, use, image_subrange, aspect);
					} else {
						impl->emit_memory_barrier(impl->get_pass(*link->undef).pre_memory_barriers, last_use, use);
					}

					if (crosses_queue(last_use, use)) {
						// we wait on either def or the last read if there was one
						if (last_executing_pass_idx != -1) {
							impl->get_pass(*link->undef)
							    .relative_waits.append(impl->waits, { (DomainFlagBits)(last_use.domain & DomainFlagBits::eQueueMask).m_mask, last_executing_pass_idx });
							impl->get_pass(last_executing_pass_idx).is_waited_on = true;
						} else {
							auto& acquire = is_image ? impl->get_bound_attachment(link->def->pass).acquire : impl->get_bound_buffer(link->def->pass).acquire;
							impl->get_pass(*link->undef).absolute_waits.append(impl->absolute_waits, { acquire.initial_domain, acquire.initial_visibility });
						}
					}
					last_use = use;
				}
			}

			// tail can be either a release or nothing
			if (link->undef && link->undef->pass < 0) { // a release
				// what if last pass is a read:
				// we loop through the read passes and select the one executing last based on the ordering
				// that pass can perform the signal and the barrier post-pass
				auto& release = impl->get_release(link->undef->pass);
				int32_t last_pass_idx = 0;
				if (link->reads.size() > 0) {
					for (auto& r : link->reads.to_span(impl->pass_idx_helper)) {
						auto order_idx = impl->computed_pass_idx_to_ordered_idx[r.pass];
						if (order_idx > last_pass_idx) {
							last_pass_idx = (int32_t)order_idx;
						}
					}
				} else { // no intervening read, we put it directly on def
					if (link->def->pass >= 0) {
						last_pass_idx = (int32_t)impl->computed_pass_idx_to_ordered_idx[link->def->pass];
					} else { // no passes using this resource, just acquired and released -> put the dep on last pass
						// TODO: should this be last pass on domain?
						last_pass_idx = (int32_t)impl->ordered_passes.size() - 1;
					}
				}

				// if the release has a bound future to signal, record that here
				auto& pass = impl->get_pass(last_pass_idx);
				if (auto* fut = release.signal) {
					fut->last_use = last_use;
					if (is_image) {
						impl->get_bound_attachment(head->def->pass).attached_future = fut;
					} else {
						impl->get_bound_buffer(head->def->pass).attached_future = fut;
					}
					pass.future_signals.append(impl->future_signals, fut);
				}

				QueueResourceUse use = release.dst_use;
				if (use.layout != ImageLayout::eUndefined) {
					if (is_image) {
						// single sided release barrier
						impl->emit_image_barrier(impl->get_pass(last_pass_idx).post_image_barriers, head->def->pass, last_use, use, image_subrange, aspect, true);
					} else {
						impl->emit_memory_barrier(impl->get_pass(last_pass_idx).post_memory_barriers, last_use, use);
					}
				}
			} else if (!link->undef) {
				// no release on this end, so if def belongs to an RP and there were no reads, we can downgrade the store
				if (link->def && link->def->pass >= 0 && link->reads.size() == 0) {
					auto& pass = impl->get_pass(link->def->pass);
					if (pass.render_pass_index >= 0) {
						auto& rpi = impl->rpis[pass.render_pass_index];
						auto& bound_att = impl->get_bound_attachment(head->def->pass);
						for (auto& att : rpi.attachments) {
							if (att.attachment_info == &bound_att) {
								att.description.storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
							}
						}
					}
				}
				// TODO: we can also downgrade if the reads are in the same RP, but this is less likely
			}

			// we have processed this chain, lets see if we can unblock more chains
			// TODO: this is O(n^2)
			if (is_image) {
				for (auto new_head : impl->chains) {
					if (new_head->source == link) {
						auto& new_att = impl->get_bound_attachment(new_head->def->pass);
						new_att.acquire.src_use = last_use;
						work_queue.push_back(new_head);
					}
				}
			}
		}

		// assign passes to batches (within a single queue)
		uint32_t batch_index = -1;
		DomainFlags current_queue = DomainFlagBits::eNone;
		bool needs_split = false;
		bool needs_split_next = false;
		for (size_t i = 0; i < impl->partitioned_passes.size(); i++) {
			auto& current_pass = impl->partitioned_passes[i];
			auto queue = (DomainFlagBits)(current_pass->domain & DomainFlagBits::eQueueMask).m_mask;

			if (queue != current_queue) { // if we go into a new queue, reset batch index
				current_queue = queue;
				batch_index = -1;
				needs_split = false;
			}

			if (current_pass->relative_waits.size() > 0) {
				needs_split = true;
			}
			if (current_pass->is_waited_on) {
				needs_split_next = true;
			}

			current_pass->batch_index = (needs_split || (batch_index == -1)) ? ++batch_index : batch_index;
			needs_split = needs_split_next;
			needs_split_next = false;
		}

		// build waits, now that we have fixed the batches
		for (size_t i = 0; i < impl->partitioned_passes.size(); i++) {
			auto& current_pass = impl->partitioned_passes[i];
			auto waits = current_pass->relative_waits.to_span(impl->waits);
			for (auto& wait : waits) {
				wait.second = impl->ordered_passes[wait.second]->batch_index + 1; // 0 = means previous
			}
		}

		// we now have enough data to build VkRenderPasses and VkFramebuffers

		// compile attachments
		// we have to assign the proper attachments to proper slots
		// the order is given by the resource binding order

		for (auto& rp : impl->rpis) {
			rp.rpci.color_ref_offsets.resize(1);
			rp.rpci.ds_refs.resize(1);
		}

		size_t previous_rp = -1;
		uint32_t previous_sp = -1;
		for (auto& pass_p : impl->partitioned_passes) {
			auto& pass = *pass_p;
			if (pass.render_pass_index < 0) {
				continue;
			}
			auto& rpi = impl->rpis[pass.render_pass_index];
			auto subpass_index = pass.subpass;
			auto& color_attrefs = rpi.rpci.color_refs;
			auto& ds_attrefs = rpi.rpci.ds_refs;

			// do not process merged passes
			if (previous_rp != -1 && previous_rp == pass.render_pass_index && previous_sp == pass.subpass) {
				continue;
			} else {
				previous_rp = pass.render_pass_index;
				previous_sp = pass.subpass;
			}

			for (auto& res : pass.resources.to_span(impl->resources)) {
				if (!is_framebuffer_attachment(res))
					continue;
				VkAttachmentReference attref{};

				auto& attachment_info = impl->get_bound_attachment(res.reference);
				auto aspect = format_to_aspect(attachment_info.attachment.format);
				ImageLayout layout;
				if (res.promoted_to_general) {
					layout = ImageLayout::eGeneral;
				} else if (!is_write_access(res.ia)) {
					layout = ImageLayout::eReadOnlyOptimalKHR;
				} else {
					layout = ImageLayout::eAttachmentOptimalKHR;
				}

				for (auto& att : rpi.attachments) {
					if (att.attachment_info == &attachment_info) {
						att.description.initialLayout = (VkImageLayout)layout;
						att.description.finalLayout = (VkImageLayout)layout;
					}
				}

				attref.layout = (VkImageLayout)layout;
				attref.attachment = (uint32_t)std::distance(rpi.attachments.begin(), std::find_if(rpi.attachments.begin(), rpi.attachments.end(), [&](auto& att) {
					                                            return att.attachment_info == &attachment_info;
				                                            }));
				if ((aspect & ImageAspectFlagBits::eColor) == ImageAspectFlags{}) { // not color -> depth or depth/stencil
					ds_attrefs[subpass_index] = attref;
				} else {
					color_attrefs.push_back(attref);
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
			for (size_t i = 0; i < 1; i++) {
				SubpassDescription sd;
				size_t color_count = 0;
				if (i < 0) {
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

		return { expected_value, *this };
	}

	std::span<ChainLink*> Compiler::get_use_chains() const {
		return std::span(impl->chains);
	}

	MapProxy<QualifiedName, const AttachmentInfo&> Compiler::get_bound_attachments() {
		return &impl->bound_attachments;
	}

	MapProxy<QualifiedName, const BufferInfo&> Compiler::get_bound_buffers() {
		return &impl->bound_buffers;
	}

	ImageUsageFlags Compiler::compute_usage(const ChainLink* head) {
		return impl->compute_usage(head);
	}

	ImageUsageFlags RGCImpl::compute_usage(const ChainLink* head) {
		ImageUsageFlags usage = {};

		constexpr auto access_to_usage = [](ImageUsageFlags& usage, Access acc) {
			if (acc & (eMemoryRW | eColorResolveRead | eColorResolveWrite | eColorRW)) {
				usage |= ImageUsageFlagBits::eColorAttachment;
			}
			if (acc & (eMemoryRW | eFragmentSampled | eComputeSampled | eRayTracingSampled | eVertexSampled)) {
				usage |= ImageUsageFlagBits::eSampled;
			}
			if (acc & (eMemoryRW | eDepthStencilRW)) {
				usage |= ImageUsageFlagBits::eDepthStencilAttachment;
			}
			if (acc & (eMemoryRW | eTransferRead)) {
				usage |= ImageUsageFlagBits::eTransferSrc;
			}
			if (acc & (eMemoryRW | eTransferWrite | eClear)) {
				usage |= ImageUsageFlagBits::eTransferDst;
			}
			if (acc & (eMemoryRW | eFragmentRW | eComputeRW | eRayTracingRW)) {
				usage |= ImageUsageFlagBits::eStorage;
			}
		};

		for (auto chain = head; chain != nullptr; chain = chain->next) {
			if (chain->def->pass >= 0) {
				auto ia = get_resource(*chain->def).ia;
				access_to_usage(usage, ia);
			}
			for (auto& r : chain->reads.to_span(pass_idx_helper)) {
				auto ia = get_resource(r).ia;
				access_to_usage(usage, ia);
			}
			if (chain->undef && chain->undef->pass >= 0) {
				auto ia = get_resource(*chain->undef).ia;
				access_to_usage(usage, ia);
			} else if (chain->undef) { // TODO: add release here

				// access_to_usage(usage, impl->get_release(chain->undef->pass));
			}
		}

		return usage;
	}

	const struct AttachmentInfo& Compiler::get_chain_attachment(const ChainLink* head) {
		const ChainLink* link = head;

		while (link->def->pass >= 0 || link->source) {
			for (; link->def->pass >= 0; link = link->prev)
				;
			while (link->source) {
				for (link = link->source; link->def->pass > 0; link = link->prev)
					;
			}
		}

		assert(link->def->pass < 0);
		return impl->get_bound_attachment(link->def->pass);
	}
} // namespace vuk
