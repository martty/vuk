#include "vuk/RenderGraph.hpp"
#include "RenderGraphImpl.hpp"
#include "RenderGraphUtil.hpp"
#include "vuk/CommandBuffer.hpp"
#include "vuk/Context.hpp"
#include "vuk/Exception.hpp"
#include "vuk/Future.hpp"

#include <bit>
#include <charconv>
#include <fmt/printf.h>
#include <set>
#include <sstream>
#include <unordered_set>

// intrinsics
namespace {
	void diverge(vuk::CommandBuffer&) {}
	void converge(vuk::CommandBuffer&) {}
} // namespace

namespace vuk {
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
		pw.make_argument_tuple = p.make_argument_tuple;
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
			PassInfo& pi = computed_passes.emplace_back(p);
			pi.qualified_name = { joiner, p.name };
			pi.resources.offset0 = resources.size();
			for (auto r : p.resources.to_span(other.impl->resources)) {
				r.original_name = r.name.name;
				if (r.foreign) {
					auto prefix = Name{ std::find_if(sg_prefixes.begin(), sg_prefixes.end(), [=](auto& kv) { return kv.first == r.foreign; })->second };
					auto full_src_prefix = !r.name.prefix.is_invalid() ? prefix.append(r.name.prefix.to_sv()) : prefix;
					auto res_name = resolve_alias_rec({ full_src_prefix, r.name.name });
					auto res_out_name = r.out_name.name.is_invalid() ? QualifiedName{} : resolve_alias_rec({ full_src_prefix, r.out_name.name });
					auto full_dst_prefix = !r.name.prefix.is_invalid() ? joiner.append(r.name.prefix.to_sv()) : joiner;
					computed_aliases.emplace(QualifiedName{ full_dst_prefix, r.name.name }, res_name);
					r.name = res_name;
					if (!r.out_name.is_invalid()) {
						computed_aliases.emplace(QualifiedName{ full_dst_prefix, r.out_name.name }, res_out_name);
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

		final_releases.insert(final_releases.end(), other.impl->final_releases.begin(), other.impl->final_releases.end());
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
		std::erase_if(passes, [](auto& pass) { return pass.pass->type == PassType::eDiverge && pass.resources.size() == 0; });
	}

	Result<void> build_links(std::span<PassInfo> passes, ResourceLinkMap& res_to_links, std::vector<Resource>& resources, std::vector<ChainAccess>& pass_reads) {
		// build edges into link map
		// reserving here to avoid rehashing map
		res_to_links.clear();
		res_to_links.reserve(passes.size() * 10);

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
						r_io.reads.append(pass_reads, { pass_idx, res_idx });
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

		return { expected_value };
	}

	Result<void> RGCImpl::terminate_chains() {
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

		return { expected_value };
	}

	Result<void> collect_chains(ResourceLinkMap& res_to_links, std::vector<ChainLink*>& chains) {
		chains.clear();
		// collect chains by looking at links without a prev
		for (auto& [name, link] : res_to_links) {
			if (!link.prev) {
				chains.push_back(&link);
			}
		}

		return { expected_value };
	}

	Result<void> RGCImpl::diagnose_unheaded_chains() {
		// diagnose unheaded chains at this point
		for (auto& chp : chains) {
			if (!chp->def) {
				if (chp->reads.size() > 0) {
					auto& pass = get_pass(chp->reads.to_span(pass_reads)[0]);
					auto& res = get_resource(chp->reads.to_span(pass_reads)[0]);

					return { expected_error, errors::make_unattached_resource_exception(pass, res) };
				}
				if (chp->undef && chp->undef->pass >= 0) {
					auto& pass = get_pass(*chp->undef);
					auto& res = get_resource(*chp->undef);

					return { expected_error, errors::make_unattached_resource_exception(pass, res) };
				}
			}
		}

		return { expected_value };
	}

	Result<void> RGCImpl::schedule_intra_queue(std::span<PassInfo> passes, const RenderGraphCompileOptions& compile_options) {
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
			for (auto& read : link.reads.to_span(pass_reads)) {
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

		return { expected_value };
	}

	Result<void> RGCImpl::relink_subchains() {
		child_chains.clear();
		// connect subchains
		// diverging subchains are chains where def->pass >= 0 AND def->pass type is eDiverge
		// reconverged subchains are chains where def->pass >= 0 AND def->pass type is eConverge
		for (auto& head : chains) {
			if (head->def->pass >= 0 && head->type == Resource::Type::eImage) { // no Buffer divergence
				auto& pass = get_pass(*head->def);
				if (pass.pass->type == PassType::eDiverge) { // diverging subchain
					auto& whole_res = pass.resources.to_span(resources)[0].name;
					auto parent_chain_end = &res_to_links[whole_res];
					head->source = parent_chain_end;
					parent_chain_end->child_chains.append(child_chains, head);
				} else if (pass.pass->type == PassType::eConverge) { // reconverged subchain
					// take first resource which guaranteed to be diverged
					auto div_resources = pass.resources.to_span(resources).subspan(1);
					for (auto& res : div_resources) {
						res_to_links[res.name].destination = head;
					}
				}
			}
		}

		conv_subchains.clear();
		// take all converged subchains and replace their def with a new attachment that has the original and subrange, and unsynch acq
		for (auto& head : chains) {
			if (head->def->pass >= 0 && head->type == Resource::Type::eImage) { // no Buffer divergence
				auto& pass = get_pass(*head->def);
				if (pass.pass->type == PassType::eConverge) { // converge subchain
					conv_subchains.push_back(&head);
					// reconverged resource is always first resource
					auto& whole_res = pass.resources.to_span(resources)[0];
					// take first resource which guaranteed to be diverged
					auto div_resources = pass.resources.to_span(resources).subspan(1);
					auto& div_res = div_resources[0];
					// TODO: we actually need to walk all converging resources here to find the scope of the convergence
					// walk this resource to convergence
					ChainLink* link = &res_to_links[div_res.name];
					while (link->prev) { // seek to the head of the diverged chain
						link = link->prev;
					}
					assert(link->source);
					link = link->source;
					head->source = link; // set the source for this subchain the original undiv chain
				}
			}
		}

		for (auto& head : conv_subchains) {
			auto& pass = get_pass(*(*head)->def);
			// reconverged resource is always first resource
			auto& whole_res = pass.resources.to_span(resources)[0];
			auto link = *head;
			// here we must climb back to a node where the def->pass is < 0
			while (link->def->pass >= 0) {
				while (link->prev) { // seek to the head of chain
					link = link->prev;
				}
				if (link->source) {
					link = link->source;
				}
			}
			auto whole_att = get_bound_attachment(link->def->pass); // the whole attachment
			whole_att.name = QualifiedName{ Name{}, whole_res.out_name.name };
			whole_att.acquire.unsynchronized = true;
			whole_att.parent_attachment = link->def->pass;
			auto new_bound = bound_attachments.emplace_back(whole_att);
			// replace head->def with new attachments (the converged resource)
			(*head)->def = { .pass = static_cast<int32_t>(-1 * bound_attachments.size()) };
		}

		div_subchains.clear();
		// take all diverged subchains and replace their def with a new attachment that has the proper acq and subrange
		for (auto& head : chains) {
			if (head->def->pass >= 0 && head->type == Resource::Type::eImage) { // no Buffer divergence
				auto& pass = get_pass(*head->def);
				if (pass.pass->type == PassType::eDiverge) { // diverging subchain
					div_subchains.push_back(head);
					// whole resource is always first resource
					auto& whole_res = pass.resources.to_span(resources)[0].name;
					auto link = &res_to_links[whole_res];
					while (link->prev) { // seek to the head of the original chain
						link = link->prev;
					}
					auto att = get_bound_attachment(link->def->pass); // the original chain attachment, make copy
					auto& our_res = get_resource(*head->def);

					att.image_subrange = diverged_subchain_headers.at(our_res.out_name).second; // look up subrange referenced by this subchain
					att.name = QualifiedName{ Name{}, our_res.out_name.name };
					att.parent_attachment = link->def->pass;
					auto new_bound = bound_attachments.emplace_back(att);
					// replace def with new attachment
					head->def = { .pass = static_cast<int32_t>(-1 * bound_attachments.size()) };
				}
			}
		}

		return { expected_value };
	}

	// subchain fixup pass
	Result<void> RGCImpl::fix_subchains() {
		// fixup diverge subchains by copying first use on the converge subchain to their end
		for (auto& head : div_subchains) {
			// seek to the end
			ChainLink* chain;
			for (chain = head; chain->destination == nullptr && chain->next != nullptr; chain = chain->next)
				;
			auto last_chain_on_diverged = chain;
			auto first_chain_on_converged = chain->destination;
			if (!chain->destination) {
				continue;
			}
			if (first_chain_on_converged->reads.size() > 0) { // first use is reads
				// take the reads
				for (auto& r : first_chain_on_converged->reads.to_span(pass_reads)) {
					last_chain_on_diverged->reads.append(pass_reads, r);
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

		// fixup converge subchains by removing the first use
		for (auto& head : conv_subchains) {
			auto old_head = *head;
			if ((*head)->reads.size() > 0) { // first use is reads
				(*head)->reads = {};           // remove the reads
			} else if ((*head)->undef) {     // first use is undef
				// if we had this as a child chain, then lets replace it
				auto it = std::find(child_chains.begin(), child_chains.end(), old_head);
				while (it != child_chains.end()) {
					*it = old_head->next;
					it = std::find(child_chains.begin(), child_chains.end(), old_head);
				}
				if (old_head->next) { // drop link from chain
					// repoint subchain while preserving def and source
					*head = old_head->next;
					(*head)->prev = nullptr;
					(*head)->def = old_head->def;
					(*head)->source = old_head->source;
				} else { // there is no next use, just drop chain
					// copy here to avoid reallocating the vector while going through
					std::vector<ChainLink*> child_ch =
					    std::vector<ChainLink*>((*head)->child_chains.to_span(child_chains).begin(), (*head)->child_chains.to_span(child_chains).end());
					for (auto& cc : child_ch) {
						(*head)->source->child_chains.append(child_chains, cc);
					}
					*head = nullptr;
				}
			}
			if (*head) {
				old_head->source->child_chains.append(child_chains, *head);
			}
		}
		/*
		for (auto& head : chains) {
		  for (ChainLink* link = head; link != nullptr; link = link->next) {
		    if (link && link->source) {
		      link->source->child_chains.offset0 = link->source->child_chains.offset1 = 0;
		    }
		  }
		}

		child_chains.clear();

		for (auto& head : chains) {
		  for (ChainLink* link = head; link != nullptr; link = link->next) {
		    if (link && link->source) {
		      link->source->child_chains.append(child_chains, head);
		    }
		  }
		}*/

		return { expected_value };

		// TODO: validate incorrect convergence
		/* else if (chain.back().high_level_access == Access::eConverge) {
		  assert(it->second.dst_use.layout != ImageLayout::eUndefined); // convergence into no use = disallowed
		}*/
		//
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

	void RGCImpl::compute_prefixes(const RenderGraph& rg, std::string& prefix) {
		auto prefix_current_size = prefix.size();
		prefix.append(rg.name.c_str());

		if (auto& counter = ++sg_name_counter[rg.name]; counter > 1) {
			prefix.append("#");
			prefix.resize(prefix.size() + 10);
			auto [ptr, ec] = std::to_chars(prefix.data() + prefix.size() - 10, prefix.data() + prefix.size(), counter - 1);
			assert(ec == std::errc());
			prefix.resize(ptr - prefix.data());
		}

		sg_prefixes.emplace(&rg, prefix);

		prefix.append("::");

		for (auto& [sg_ptr, sg_info] : rg.impl->subgraphs) {
			if (sg_info.count > 0) {
				assert(sg_ptr->impl);

				compute_prefixes(*sg_ptr, prefix);
			}
		}

		prefix.resize(prefix_current_size); // rewind prefix
	}

	void RGCImpl::inline_subgraphs(const RenderGraph& rg, robin_hood::unordered_flat_set<RenderGraph*>& consumed_rgs) {
		auto our_prefix = sg_prefixes.at(&rg);
		for (auto& [sg_ptr, sg_info] : rg.impl->subgraphs) {
			auto sg_raw_ptr = sg_ptr.get();
			if (sg_info.count > 0) {
				auto prefix = sg_prefixes.at(sg_raw_ptr);
				assert(sg_raw_ptr->impl);
				for (auto& [name_in_parent, name_in_sg] : sg_info.exported_names) {
					QualifiedName old_name;
					if (!name_in_sg.prefix.is_invalid()) { // unfortunately, prefix + name_in_sg.prefix duplicates the name of the sg, so remove it
						std::string fixed_prefix = prefix.substr(0, prefix.size() - sg_raw_ptr->name.to_sv().size());
						fixed_prefix.append(name_in_sg.prefix.to_sv());
						old_name = QualifiedName{ Name(fixed_prefix), name_in_sg.name };
					} else {
						old_name = QualifiedName{ Name(prefix), name_in_sg.name };
					}

					auto new_name = QualifiedName{ our_prefix.empty() ? Name{} : Name(our_prefix), name_in_parent };
					computed_aliases[new_name] = old_name;
				}
				if (!consumed_rgs.contains(sg_raw_ptr)) {
					inline_subgraphs(*sg_raw_ptr, consumed_rgs);
					append(Name(prefix), *sg_raw_ptr);
					consumed_rgs.emplace(sg_raw_ptr);
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

		robin_hood::unordered_flat_set<RenderGraph*> consumed_rgs = {};
		std::string prefix = "";
		for (auto& rg : rgs) {
			impl->compute_prefixes(*rg, prefix);
			consumed_rgs.clear();
			impl->inline_subgraphs(*rg, consumed_rgs);
		}

		for (auto& rg : rgs) {
			auto our_prefix = std::find_if(impl->sg_prefixes.begin(), impl->sg_prefixes.end(), [rgp = rg.get()](auto& kv) { return kv.first == rgp; })->second;
			impl->append(Name{ our_prefix.c_str() }, *rg);
		}

		return { expected_value };
	}

	void RGCImpl::compute_assigned_names() {
		// gather name alias info now - once we partition, we might encounter unresolved aliases
		robin_hood::unordered_flat_map<QualifiedName, QualifiedName> name_map;
		name_map.insert(computed_aliases.begin(), computed_aliases.end());

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
					if ((DomainFlags)att_dom != last_domain && att_dom != DomainFlagBits::eDevice && att_dom != DomainFlagBits::eAny) {
						last_domain = att_dom;
					}
				}
				for (auto& r : chain->reads.to_span(impl->pass_reads)) {
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
				for (auto& r : chain->reads.to_span(impl->pass_reads)) {
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
				impl->last_ordered_pass_idx_in_domain_array[2] = impl->ordered_idx_to_computed_pass_idx[i];
				impl->partitioned_passes.push_back(p);
			}
		}
		impl->transfer_passes = { impl->partitioned_passes.begin(), impl->partitioned_passes.size() };
		for (size_t i = 0; i < impl->ordered_passes.size(); i++) {
			auto& p = impl->ordered_passes[i];
			if (p->domain & DomainFlagBits::eComputeQueue) {
				impl->computed_pass_idx_to_partitioned_idx[impl->ordered_idx_to_computed_pass_idx[i]] = impl->partitioned_passes.size();
				impl->last_ordered_pass_idx_in_domain_array[1] = impl->ordered_idx_to_computed_pass_idx[i];
				impl->partitioned_passes.push_back(p);
			}
		}
		impl->compute_passes = { impl->partitioned_passes.begin() + impl->transfer_passes.size(), impl->partitioned_passes.size() - impl->transfer_passes.size() };
		for (size_t i = 0; i < impl->ordered_passes.size(); i++) {
			auto& p = impl->ordered_passes[i];
			if (p->domain & DomainFlagBits::eGraphicsQueue) {
				impl->computed_pass_idx_to_partitioned_idx[impl->ordered_idx_to_computed_pass_idx[i]] = impl->partitioned_passes.size();
				impl->last_ordered_pass_idx_in_domain_array[0] = impl->ordered_idx_to_computed_pass_idx[i];
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
				for (auto& r : link->reads.to_span(impl->pass_reads)) {
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

	void Compiler::render_pass_assignment() {
		// graphics: assemble renderpasses based on framebuffers
		// we need to collect passes into framebuffers, which will determine the renderpasses

		// renderpasses are uniquely identified by their index from now on
		// tell passes in which renderpass/subpass they will execute
		impl->rpis.reserve(impl->graphics_passes.size());
		for (auto& passinfo : impl->graphics_passes) {
			int32_t rpi_index = -1;
			RenderPassInfo* rpi = nullptr;

			for (auto& res : passinfo->resources.to_span(impl->resources)) {
				if (is_framebuffer_attachment(res)) {
					if (rpi == nullptr) {
						rpi_index = (int32_t)impl->rpis.size();
						rpi = &impl->rpis.emplace_back();
					}
					auto& bound_att = impl->get_bound_attachment(res.reference);
					AttachmentRPInfo rp_info{ &bound_att };
					rp_info.description.loadOp = VK_ATTACHMENT_LOAD_OP_LOAD;
					rp_info.description.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
					rpi->attachments.append(impl->rp_infos, rp_info);
				}
			}

			passinfo->render_pass_index = (int32_t)rpi_index;
			passinfo->subpass = 0;
		}
	}

	Result<void> Compiler::compile(std::span<std::shared_ptr<RenderGraph>> rgs, const RenderGraphCompileOptions& compile_options) {
		auto arena = impl->arena_.release();
		delete impl;
		arena->reset();
		impl = new RGCImpl(arena);
		impl->callbacks = compile_options.callbacks;

		VUK_DO_OR_RETURN(inline_rgs(rgs));

		impl->compute_assigned_names();

		impl->merge_diverge_passes(impl->computed_passes);

		// run global pass ordering - once we split per-queue we don't see enough
		// inputs to order within a queue

		VUK_DO_OR_RETURN(build_links(impl->computed_passes, impl->res_to_links, impl->resources, impl->pass_reads));
		VUK_DO_OR_RETURN(impl->terminate_chains());
		VUK_DO_OR_RETURN(collect_chains(impl->res_to_links, impl->chains));
		VUK_DO_OR_RETURN(impl->diagnose_unheaded_chains());
		VUK_DO_OR_RETURN(impl->schedule_intra_queue(impl->computed_passes, compile_options));

		VUK_DO_OR_RETURN(impl->relink_subchains());
		resource_linking();
		VUK_DO_OR_RETURN(impl->fix_subchains());
		// fix subchains might remove chains, so drop those now
		std::erase(impl->chains, nullptr);
		// auto dumped_graph = dump_graph();

		queue_inference();
		pass_partitioning();
		render_pass_assignment();

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

	ImageAttachment& RenderGraph::attach_image(Name name, ImageAttachment att, Access initial_acc) {
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

		// TODO: this is heavyhanded - we add this because of swapchain sync
		if (format_to_aspect(att.format) & ImageAspectFlagBits::eDepth) {
			initial.stages |= PipelineStageFlagBits::eEarlyFragmentTests | PipelineStageFlagBits::eLateFragmentTests;
			initial.access |= vuk::AccessFlagBits::eDepthStencilAttachmentWrite;
		}

		if (format_to_aspect(att.format) & ImageAspectFlagBits::eColor) {
			initial.stages |= PipelineStageFlagBits::eColorAttachmentOutput;
			initial.access |= AccessFlagBits::eColorAttachmentWrite;
		}

		impl->bound_attachments.emplace(attachment_info.name, attachment_info).first->second.attachment;
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

		// TODO: this is heavyhanded - we add this because of swapchain sync
		if (format_to_aspect(att.format) & ImageAspectFlagBits::eDepth) {
			initial.stages |= PipelineStageFlagBits::eEarlyFragmentTests | PipelineStageFlagBits::eLateFragmentTests;
			initial.access |= vuk::AccessFlagBits::eDepthStencilAttachmentWrite;
		}

		if (format_to_aspect(att.format) & ImageAspectFlagBits::eColor) {
			initial.stages |= PipelineStageFlagBits::eColorAttachmentOutput;
			initial.access |= AccessFlagBits::eColorAttachmentWrite;
		}

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
			impl->imported_names.emplace_back(QualifiedName{ {}, name });
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
			impl->imported_names.emplace_back(QualifiedName{ {}, name });
			fimg.rg.reset();
		} else {
			assert(0);
		}
	}

	void RenderGraph::attach_in(std::span<Future> futures) {
		for (auto& f : futures) {
			auto name = f.get_bound_name();
			attach_in(name.name, std::move(f));
		}
	}

	void RenderGraph::inference_rule(Name target, std::function<void(const struct InferenceContext&, ImageAttachment&)> rule) {
		impl->ia_inference_rules.emplace_back(IAInference{ QualifiedName{ Name{}, target }, std::move(rule) });
	}

	void RenderGraph::inference_rule(Name target, std::function<void(const struct InferenceContext&, Buffer&)> rule) {
		impl->buf_inference_rules.emplace_back(BufferInference{ QualifiedName{ Name{}, target }, std::move(rule) });
	}

	robin_hood::unordered_flat_set<QualifiedName> RGImpl::get_available_resources() {
		std::vector<PassInfo> pass_infos;
		pass_infos.reserve(passes.size());
		for (auto& pass : passes) {
			pass_infos.emplace_back(pass).resources = pass.resources;
		}
		std::vector<Resource> resolved_resources;
		for (auto res : resources) {
			res.name.name = res.name.name.is_invalid() ? Name{} : resolve_alias(res.name.name);
			res.out_name.name = res.out_name.name.is_invalid() ? Name{} : resolve_alias(res.out_name.name);
			resolved_resources.emplace_back(res);
		}
		ResourceLinkMap res_to_links;
		std::vector<ChainAccess> pass_reads;
		std::vector<ChainLink*> chains;
		build_links(pass_infos, res_to_links, resolved_resources, pass_reads);

		for (auto& bound : bound_attachments) {
			res_to_links[bound.first].def = { .pass = static_cast<int32_t>(-1 * (&bound - &*bound_attachments.begin() + 1)) };
		}

		for (auto& bound : bound_buffers) {
			res_to_links[bound.first].def = { .pass = static_cast<int32_t>(-1 * (&bound - &*bound_buffers.begin() + 1)) };
		}

		for (auto& bound : releases) {
			res_to_links[bound.first].undef = { .pass = static_cast<int32_t>(-1 * (&bound - &*releases.begin() + 1)) };
		}

		collect_chains(res_to_links, chains);

		robin_hood::unordered_flat_set<QualifiedName> outputs;
		outputs.insert(imported_names.begin(), imported_names.end());
		for (auto& head : chains) {
			ChainLink* link;
			for (link = head; link->next != nullptr; link = link->next)
				;

			if (link->reads.size()) {
				auto r = link->reads.to_span(pass_reads)[0];
				outputs.emplace(get_resource(pass_infos, r).name);
			}
			if (link->undef) {
				if (link->undef->pass >= 0) {
					outputs.emplace(get_resource(pass_infos, *link->undef).out_name);
				} else {
					if (link->def->pass >= 0) {
						outputs.emplace(get_resource(pass_infos, *link->def).out_name);
					} else {
						// tailed by release and def unusable
					}
				}
			}
			if (link->def) {
				if (link->def->pass >= 0) {
					outputs.emplace(get_resource(pass_infos, *link->def).out_name);
				} else {
					QualifiedName name = link->type == Resource::Type::eImage
					                         ? (&*bound_attachments.begin() + (-1 * (link->def->pass) - 1))->first
					                         : (&*bound_buffers.begin() + (-1 * (link->def->pass) - 1))->first; // the only def we have is the binding
					outputs.emplace(name);
				}
			}
		}
		return outputs;
	}

	std::vector<Future> RenderGraph::split() {
		robin_hood::unordered_flat_set<QualifiedName> outputs = impl->get_available_resources();
		std::vector<Future> futures;
		for (auto& elem : outputs) {
			futures.emplace_back(this->shared_from_this(), elem);
		}
		return futures;
	}

	void RenderGraph::add_final_release(Future& future, DomainFlags src_domain) {
		impl->final_releases.emplace_back(Release{ Access::eNone, to_use(Access::eNone, src_domain), future.control.get() });
	}

	void RenderGraph::remove_final_release(Future& future) {
		for (auto it = impl->final_releases.begin(); it != impl->final_releases.end(); ++it) {
			if (it->signal == future.control.get()) {
				impl->final_releases.erase(it);
				return;
			}
		}
	}

	void RenderGraph::attach_out(QualifiedName name, Future& fimg, DomainFlags dst_domain) {
		impl->releases.emplace_back(name, Release{ Access::eNone, to_use(Access::eNone, dst_domain), fimg.control.get() });
	}

	void RenderGraph::detach_out(QualifiedName name, Future& fimg) {
		for (auto it = impl->releases.begin(); it != impl->releases.end(); ++it) {
			if (it->first == name && it->second.signal == fimg.control.get()) {
				impl->releases.erase(it);
				return;
			}
		}
	}

	void RenderGraph::release(Name name, Access final) {
		impl->releases.emplace_back(QualifiedName{ Name{}, name }, Release{ final, to_use(final, DomainFlagBits::eAny) });
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
		impl->releases.emplace_back(QualifiedName{ Name{}, name }, Release{ .original = ePresent, .dst_use = final });
	}

	Name RenderGraph::get_temporary_name() {
		return impl->temporary_name.append(std::to_string(impl->temporary_name_counter++));
	}

	IARule same_extent_as(Name n) {
		return [=](const InferenceContext& ctx, ImageAttachment& ia) {
			ia.extent = ctx.get_image_attachment(n).extent;
		};
	}

	IARule same_extent_as(TypedFuture<Image> inference_source) {
		return [=](const InferenceContext& ctx, ImageAttachment& ia) {
			ia.extent = inference_source.attachment->extent;
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
		if (last_use.stages == vuk::PipelineStageFlagBits{}) {
			return;
		}

		// for now we only emit pre- memory barriers, so the executing domain is always 'current_use.domain'
		scope_to_domain((VkPipelineStageFlagBits2KHR&)last_use.stages, current_use.domain & DomainFlagBits::eQueueMask);
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

	Result<void> RGCImpl::generate_barriers_and_waits() {
#ifdef VUK_DUMP_USE
		fmt::printf("------------------------\n");
		fmt::printf("digraph vuk {\n");
#endif

		// we need to handle chains in order of dependency
		std::vector<ChainLink*> work_queue;
		for (auto head : chains) {
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
		std::vector<ChainLink*> seen_chains;
		while (work_queue.size() > 0) {
			ChainLink* head = work_queue.back();
			work_queue.pop_back();
			seen_chains.push_back(head);
			// handle head
			ImageAspectFlags aspect;
			Subrange::Image image_subrange;
			bool is_image = head->type == Resource::Type::eImage;
			if (is_image) {
				auto& att = get_bound_attachment(head->def->pass);
				aspect = format_to_aspect(att.attachment.format);
				image_subrange = att.image_subrange;
#ifdef VUK_DUMP_USE
				if (head->source) {
					fmt::print("\"{}\" [shape=diamond,label=\" {} \"];\n", att.name.name.c_str(), att.name.name.c_str());
					auto& last_ud = get_resource(*head->source->undef);
					fmt::print("\"{}\" -> \"{}\" [color = red];\n", last_ud.name.name.c_str(), att.name.name.c_str());
				} else {
					fmt::print("\"{}\" [shape=box,label=\" {} \"];\n", att.name.name.c_str(), att.name.name.c_str());
				}
#endif
			} else {
#ifdef VUK_DUMP_USE
				auto& buf = get_bound_buffer(head->def->pass);
				fmt::print("\"{}\" [shape=box,label=\" {} \"];\n", buf.name.name.c_str(), buf.name.name.c_str());
#endif
			}
			bool is_subchain = head->source != nullptr;

			// last use on the chain
			QueueResourceUse last_use;
			ChainLink* link;

			for (link = head; link != nullptr; link = link->next) {
#ifdef VUK_DUMP_USE
				QualifiedName def_name;
				if (link->def->pass >= 0) {
					auto& def_res = get_resource(*link->def);
					def_name = def_res.out_name;
				} else {
					if (is_image) {
						def_name = get_bound_attachment(head->def->pass).name;
					} else {
						def_name = get_bound_buffer(head->def->pass).name;
					}
				}
#endif

				// populate last use from def or attach
				if (link->def->pass >= 0) {
					auto& def_pass = get_pass(*link->def);
					auto& def_res = get_resource(*link->def);
					last_use = to_use(def_res.ia, def_pass.domain);
#ifdef VUK_DUMP_USE
					fmt::print("\"{}\" [label=\" {} \"];\n", def_res.name.name.c_str(), def_res.name.name.c_str());
#endif
				} else {
					last_use = is_image ? get_bound_attachment(head->def->pass).acquire.src_use : get_bound_buffer(head->def->pass).acquire.src_use;
				}
				if (last_use.stages == PipelineStageFlagBits{}) {
					last_use.stages = PipelineStageFlagBits::eAllCommands;
				}

				// handle chain
				// we need to emit: def -> reads, RAW or nothing
				//					reads -> undef, WAR
				//					if there were no reads, then def -> undef, which is either WAR or WAW

				// we need to sometimes emit a barrier onto the last thing that happened, find that pass here
				// it is either last read pass, or if there were no reads, the def pass, or if the def pass doesn't exist, then we search parent chains
				int32_t last_executing_pass_idx = link->def->pass >= 0 ? (int32_t)computed_pass_idx_to_ordered_idx[link->def->pass] : -1;

				if (link->reads.size() > 0) { // we need to emit: def -> reads, RAW or nothing (before first read)
					// to avoid R->R deps, we emit a single dep for all the reads
					// for this we compute a merged layout (TRANSFER_SRC_OPTIMAL / READ_ONLY_OPTIMAL / GENERAL)
					QueueResourceUse use;
					auto reads = link->reads.to_span(pass_reads);
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
							auto& pass = get_pass(r);
							auto& res = get_resource(r);
#ifdef VUK_DUMP_USE
							fmt::print("\"{}{}\" [label=\" {} \"];\n", def_name.name.c_str(), 10000 + r.pass, "R");
							fmt::print(
							    "\"{}\" -> \"{}{}\" [label=\" {} \"];\n", def_name.name.c_str(), def_name.name.c_str(), 10000 + r.pass, pass.qualified_name.name.c_str());
#endif

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
							int32_t order_idx = (int32_t)computed_pass_idx_to_ordered_idx[r.pass];
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
								auto& res = get_resource(r);
								res.promoted_to_general = true;
							}
						}

						if (last_use_source < 0 && is_subchain) {
							assert(link->source->undef && link->source->undef->pass >= 0);
							last_use_source = link->source->undef->pass;
						}

						// TODO: do not emit this if dep is a read and the layouts match
						auto& dst = get_pass(first_pass_idx);
						if (is_image) {
							if (crosses_queue(last_use, use)) {
								emit_image_barrier(get_pass((int32_t)computed_pass_idx_to_ordered_idx[last_use_source]).post_image_barriers,
								                   head->def->pass,
								                   last_use,
								                   use,
								                   image_subrange,
								                   aspect,
								                   true);
							}
							emit_image_barrier(dst.pre_image_barriers, head->def->pass, last_use, use, image_subrange, aspect);
						} else {
							emit_memory_barrier(dst.pre_memory_barriers, last_use, use);
						}
						if (crosses_queue(last_use, use)) {
							// in this case def was on a different queue the subsequent reads
							// we stick the wait on the first read pass in order
							get_pass(first_pass_idx)
							    .relative_waits.append(
							        waits, { (DomainFlagBits)(last_use.domain & DomainFlagBits::eQueueMask).m_mask, computed_pass_idx_to_ordered_idx[last_use_source] });
							get_pass((int32_t)computed_pass_idx_to_ordered_idx[last_use_source]).is_waited_on++;
						}
						last_use = use;
						last_use_source = reads[read_idx - 1].pass;
						start_of_reads = read_idx;
					}
				}

				// if there are no intervening reads, emit def -> undef, otherwise emit reads -> undef
				//  def -> undef, which is either WAR or WAW (before undef)
				//	reads -> undef, WAR (before undef)
				if (link->undef && link->undef->pass >= 0) {
					auto& pass = get_pass(*link->undef);
					auto& res = get_resource(*link->undef);
#ifdef VUK_DUMP_USE
					fmt::print("\"{}\" -> \"{}\" [style=bold, color=blue, label=\" {} \"];\n",
					           res.name.name.c_str(),
					           res.out_name.name.c_str(),
					           pass.qualified_name.name.c_str());
#endif
					QueueResourceUse use = to_use(res.ia, pass.domain);
					if (use.layout == ImageLayout::eGeneral) {
						res.promoted_to_general = true;
					}

					// handle renderpass details
					// all renderpass write-attachments are entered via an undef (because of it being a write)
					if (pass.render_pass_index >= 0) {
						auto& rpi = rpis[pass.render_pass_index];
						auto& bound_att = get_bound_attachment(head->def->pass);
						for (auto& att : rpi.attachments.to_span(rp_infos)) {
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
							last_executing_pass_idx = (int32_t)computed_pass_idx_to_ordered_idx[link->source->undef->pass];
						}
					}

					if (res.ia != eConsume) {
						if (is_image) {
							if (crosses_queue(last_use, use)) { // release barrier
								if (last_executing_pass_idx !=
								    -1) { // if last_executing_pass_idx is -1, then there is release in this rg, so we don't emit the release (single-sided acq)
									emit_image_barrier(get_pass(last_executing_pass_idx).post_image_barriers, head->def->pass, last_use, use, image_subrange, aspect, true);
								}
							}
							emit_image_barrier(get_pass(*link->undef).pre_image_barriers, head->def->pass, last_use, use, image_subrange, aspect);
						} else {
							emit_memory_barrier(get_pass(*link->undef).pre_memory_barriers, last_use, use);
						}

						if (crosses_queue(last_use, use)) {
							// we wait on either def or the last read if there was one
							if (last_executing_pass_idx != -1) {
								get_pass(*link->undef)
								    .relative_waits.append(waits, { (DomainFlagBits)(last_use.domain & DomainFlagBits::eQueueMask).m_mask, last_executing_pass_idx });
								get_pass(last_executing_pass_idx).is_waited_on++;
							} else {
								auto& acquire = is_image ? get_bound_attachment(link->def->pass).acquire : get_bound_buffer(link->def->pass).acquire;
								get_pass(*link->undef).absolute_waits.append(absolute_waits, { acquire.initial_domain, acquire.initial_visibility });
							}
						}
						last_use = use;
					}
				}

				// process tails outside
				if (link->next == nullptr) {
					break;
				}
			}

			// tail can be either a release or nothing
			if (link->undef && link->undef->pass < 0) { // a release
#ifdef VUK_DUMP_USE
				QualifiedName def_name;
				if (link->def->pass >= 0) {
					auto& def_res = get_resource(*link->def);
					def_name = def_res.out_name;
				} else {
					if (is_image) {
						def_name = get_bound_attachment(head->def->pass).name;
					} else {
						def_name = get_bound_buffer(head->def->pass).name;
					}
				}
				fmt::print("\"{}\" -> \"R+\" [style=bold, color=green];\n", def_name.name.c_str());
#endif
				// what if last pass is a read:
				// we loop through the read passes and select the one executing last based on the ordering
				// that pass can perform the signal and the barrier post-pass
				auto& release = get_release(link->undef->pass);
				int32_t last_pass_idx = 0;
				if (link->reads.size() > 0) {
					for (auto& r : link->reads.to_span(pass_reads)) {
						auto order_idx = computed_pass_idx_to_ordered_idx[r.pass];
						if (order_idx > last_pass_idx) {
							last_pass_idx = (int32_t)order_idx;
						}
					}
				} else { // no intervening read, we put it directly on def
					if (link->def->pass >= 0) {
						last_pass_idx = (int32_t)computed_pass_idx_to_ordered_idx[link->def->pass];
					} else { // no passes using this resource, just acquired and released -> put the dep on last pass
						// in case neither acquire nor release specify a pass we just put it on the last pass
						last_pass_idx = (int32_t)ordered_passes.size() - 1;
						// if release specifies a domain, use that
						if (release.dst_use.domain != DomainFlagBits::eAny && release.dst_use.domain != DomainFlagBits::eDevice) {
							last_pass_idx = last_ordered_pass_idx_in_domain((DomainFlagBits)(release.dst_use.domain & DomainFlagBits::eQueueMask).m_mask);
						}
					}
				}

				// if the release has a bound future to signal, record that here
				auto& pass = get_pass(last_pass_idx);
				if (auto* fut = release.signal) {
					fut->last_use = last_use;
					if (is_image) {
						get_bound_attachment(head->def->pass).attached_future = fut;
					} else {
						get_bound_buffer(head->def->pass).attached_future = fut;
					}
					pass.future_signals.append(future_signals, fut);
				}

				QueueResourceUse use = release.dst_use;
				if (use.layout != ImageLayout::eUndefined) {
					if (is_image) {
						// single sided release barrier
						emit_image_barrier(get_pass(last_pass_idx).post_image_barriers, head->def->pass, last_use, use, image_subrange, aspect, true);
					} else {
						emit_memory_barrier(get_pass(last_pass_idx).post_memory_barriers, last_use, use);
					}
				}
			} else {
				// no release on this end, so if def belongs to an RP and there were no reads,
				// or the chain ends in an RP undef, we can downgrade the store
				bool def_only = link->def && link->def->pass >= 0 && link->reads.size() == 0;
				bool undef = link->undef.has_value();
				if (undef || def_only) {
					auto& pass = get_pass(undef ? link->undef->pass : link->def->pass);
					if (pass.render_pass_index >= 0) {
						auto& rpi = rpis[pass.render_pass_index];
						auto& bound_att = get_bound_attachment(head->def->pass);
						for (auto& att : rpi.attachments.to_span(rp_infos)) {
							if (att.attachment_info == &bound_att) {
								att.description.storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
							}
						}
					}
				}
				// TODO: we can also downgrade if the reads are in the same RP, but this is less likely
			}

			// we have processed this chain, lets see if we can unblock more chains
			for (auto new_head : link->child_chains.to_span(child_chains)) {
				if (!new_head) {
					continue;
				}
				auto& new_att = get_bound_attachment(new_head->def->pass);
				new_att.acquire.src_use = last_use;
				work_queue.push_back(new_head);
			}
		}

		if (seen_chains.size() != chains.size()) {
			std::vector<ChainLink*> missing_chains;
			std::sort(chains.begin(), chains.end());
			std::sort(seen_chains.begin(), seen_chains.end());
			std::set_difference(chains.begin(), chains.end(), seen_chains.begin(), seen_chains.end(), std::back_inserter(missing_chains));
			assert(false);
		}

#ifdef VUK_DUMP_USE
		fmt::printf("};\n");
		fmt::printf("------------------------\n");
#endif

		return { expected_value };
	}

	Result<void> RGCImpl::merge_rps() {
		// this is only done on gfx passes
		if (graphics_passes.size() == 0) {
			return { expected_value };
		}
		// we run this after barrier gen
		// loop through gfx passes in order
		for (size_t i = 0; i < graphics_passes.size() - 1; i++) {
			auto& pass0 = *graphics_passes[i];
			auto& pass1 = *graphics_passes[i + 1];

			// two gfx passes can be merged if they are
			// - adjacent and both have rps
			if (pass0.render_pass_index == -1 || pass1.render_pass_index == -1) {
				continue;
			}
			// - have the same ordered set of attachments
			bool can_merge = true;
			auto p0res = pass0.resources.to_span(resources);
			auto p1res = pass1.resources.to_span(resources);
			size_t k = 0;
			for (size_t j = 0; j < p0res.size(); j++) {
				auto& res0 = p0res[j];
				auto& link0 = res_to_links.at(res0.name);
				if (!is_framebuffer_attachment(res0)) {
					continue;
				}
				// advance attachments in p1 until we get a match or run out
				for (; k < p1res.size(); k++) {
					auto& res1 = p1res[k];
					auto& link1 = res_to_links.at(res1.name);
					// TODO: we only handle some cases here (too conservative)
					bool same_access = res0.ia == res1.ia;
					if (same_access && link0.next == &link1) {
						break;
					}
				}
				if (k == p1res.size()) {
					can_merge = false;
				}
			}
			if (!can_merge) {
				continue;
			}
			// - contain only color and ds deps between them
			if (pass0.post_memory_barriers.size() > 0 || pass1.pre_memory_barriers.size() > 0) {
				continue;
			}
			for (auto& bar : pass0.post_image_barriers.to_span(image_barriers)) {
				if ((bar.srcAccessMask & ~(VK_ACCESS_2_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT |
				                           VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_READ_BIT | VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT)) != 0) {
					can_merge = false;
				}
				if ((bar.dstAccessMask & ~(VK_ACCESS_2_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT |
				                           VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_READ_BIT | VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT)) != 0) {
					can_merge = false;
				}
			}
			for (auto& bar : pass1.pre_image_barriers.to_span(image_barriers)) {
				if ((bar.srcAccessMask & ~(VK_ACCESS_2_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT |
				                           VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_READ_BIT | VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT)) != 0) {
					can_merge = false;
				}
				if ((bar.dstAccessMask & ~(VK_ACCESS_2_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT |
				                           VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_READ_BIT | VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT)) != 0) {
					can_merge = false;
				}
			}

			if (can_merge) {
				rpis[pass1.render_pass_index].attachments = {};
				pass1.render_pass_index = pass0.render_pass_index;
				pass0.post_image_barriers = {};
				pass1.pre_image_barriers = {};
			}
		}

		return { expected_value };
	}

	Result<void> RGCImpl::assign_passes_to_batches() {
		// cull waits
		{
			DomainFlags current_queue = DomainFlagBits::eNone;
			std::array<int64_t, 3> last_passes_waited = { -1, -1, -1 };
			// loop through all passes
			for (size_t i = 0; i < partitioned_passes.size(); i++) {
				auto& current_pass = partitioned_passes[i];
				auto queue = (DomainFlagBits)(current_pass->domain & DomainFlagBits::eQueueMask).m_mask;

				if ((DomainFlags)queue != current_queue) { // if we go into a new queue, reset wait indices
					last_passes_waited = { -1, -1, -1 };
					current_queue = queue;
				}

				RelSpan<std::pair<DomainFlagBits, uint64_t>> new_waits = {};
				auto sp = current_pass->relative_waits.to_span(waits);
				for (auto i = 0; i < sp.size(); i++) {
					auto [queue, pass_idx] = sp[i];
					auto queue_idx = std::countr_zero((uint32_t)queue) - 1;
					auto& last_amt = last_passes_waited[queue_idx];
					if ((int64_t)pass_idx > last_amt) {
						last_amt = pass_idx;
						new_waits.append(waits, std::pair{ queue, pass_idx });
						sp = current_pass->relative_waits.to_span(waits);
					} else {
						ordered_passes[pass_idx]->is_waited_on--;
					}
				}

				current_pass->relative_waits = new_waits;
			}
		}

		// assign passes to batches (within a single queue)
		uint32_t batch_index = -1;
		DomainFlags current_queue = DomainFlagBits::eNone;
		bool needs_split = false;
		bool needs_split_next = false;
		for (size_t i = 0; i < partitioned_passes.size(); i++) {
			auto& current_pass = partitioned_passes[i];
			auto queue = (DomainFlagBits)(current_pass->domain & DomainFlagBits::eQueueMask).m_mask;

			if ((DomainFlags)queue != current_queue) { // if we go into a new queue, reset batch index
				current_queue = queue;
				batch_index = -1;
				needs_split = false;
			}

			if (current_pass->relative_waits.size() > 0) {
				needs_split = true;
			}
			if (current_pass->is_waited_on > 0) {
				needs_split_next = true;
			}

			current_pass->batch_index = (needs_split || (batch_index == -1)) ? ++batch_index : batch_index;
			needs_split = needs_split_next;
			needs_split_next = false;
		}

		return { expected_value };
	}

	Result<void> RGCImpl::build_waits() {
		// build waits, now that we have fixed the batches
		for (size_t i = 0; i < partitioned_passes.size(); i++) {
			auto& current_pass = partitioned_passes[i];
			auto rel_waits = current_pass->relative_waits.to_span(waits);
			for (auto& wait : rel_waits) {
				wait.second = ordered_passes[wait.second]->batch_index + 1; // 0 = means previous
			}
		}

		return { expected_value };
	}

	Result<void> RGCImpl::build_renderpasses() {
		// compile attachments
		// we have to assign the proper attachments to proper slots
		// the order is given by the resource binding order

		for (auto& rp : rpis) {
			rp.rpci.color_ref_offsets.resize(1);
			rp.rpci.ds_refs.resize(1);
		}

		size_t previous_rp = -1;
		uint32_t previous_sp = -1;
		for (auto& pass_p : partitioned_passes) {
			auto& pass = *pass_p;
			if (pass.render_pass_index < 0) {
				continue;
			}
			auto& rpi = rpis[pass.render_pass_index];
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

			for (auto& res : pass.resources.to_span(resources)) {
				if (!is_framebuffer_attachment(res))
					continue;
				VkAttachmentReference attref{};

				auto& attachment_info = get_bound_attachment(res.reference);
				auto aspect = format_to_aspect(attachment_info.attachment.format);
				ImageLayout layout;
				if (res.promoted_to_general) {
					layout = ImageLayout::eGeneral;
				} else if (!is_write_access(res.ia)) {
					layout = ImageLayout::eReadOnlyOptimalKHR;
				} else {
					layout = ImageLayout::eAttachmentOptimalKHR;
				}

				for (auto& att : rpi.attachments.to_span(rp_infos)) {
					if (att.attachment_info == &attachment_info) {
						att.description.initialLayout = (VkImageLayout)layout;
						att.description.finalLayout = (VkImageLayout)layout;
					}
				}

				attref.layout = (VkImageLayout)layout;
				auto attachments = rpi.attachments.to_span(rp_infos);
				attref.attachment = (uint32_t)std::distance(
				    attachments.begin(), std::find_if(attachments.begin(), attachments.end(), [&](auto& att) { return att.attachment_info == &attachment_info; }));
				if ((aspect & ImageAspectFlagBits::eColor) == ImageAspectFlags{}) { // not color -> depth or depth/stencil
					ds_attrefs[subpass_index] = attref;
				} else {
					color_attrefs.push_back(attref);
				}
			}
		}

		// compile subpass description structures

		for (auto& rp : rpis) {
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

		return { expected_value };
	}

	Result<ExecutableRenderGraph> Compiler::link(std::span<std::shared_ptr<RenderGraph>> rgs, const RenderGraphCompileOptions& compile_options) {
		VUK_DO_OR_RETURN(compile(rgs, compile_options));

		VUK_DO_OR_RETURN(impl->generate_barriers_and_waits());

		VUK_DO_OR_RETURN(impl->merge_rps());

		VUK_DO_OR_RETURN(impl->assign_passes_to_batches());

		VUK_DO_OR_RETURN(impl->build_waits());

		// we now have enough data to build VkRenderPasses and VkFramebuffers
		VUK_DO_OR_RETURN(impl->build_renderpasses());

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
			Access ia = Access::eNone;
			if (chain->def->pass >= 0) {
				ia = get_resource(*chain->def).ia;
				access_to_usage(usage, ia);
			}
			for (auto& r : chain->reads.to_span(pass_reads)) {
				ia = get_resource(r).ia;
				access_to_usage(usage, ia);
			}
			if (chain->undef && chain->undef->pass >= 0) {
				ia = get_resource(*chain->undef).ia;
				access_to_usage(usage, ia);
			} else if (chain->undef) {
				ia = get_release(chain->undef->pass).original;
				access_to_usage(usage, ia);
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

	std::optional<QualifiedName> Compiler::get_last_use_name(const ChainLink* head) {
		const ChainLink* link = head;
		for (; link->next != nullptr; link = link->next)
			;
		if (link->reads.size()) {
			auto r = link->reads.to_span(impl->pass_reads)[0];
			return impl->get_resource(r).name;
		}
		if (link->undef) {
			if (link->undef->pass >= 0) {
				return impl->get_resource(*link->undef).out_name;
			} else {
				if (link->def->pass >= 0) {
					return impl->get_resource(*link->def).out_name;
				} else {
					return {}; // tailed by release and def unusable
				}
			}
		}
		if (link->def) {
			if (link->def->pass >= 0) {
				return impl->get_resource(*link->def).out_name;
			} else {
				return link->type == Resource::Type::eImage ? impl->get_bound_attachment(link->def->pass).name
				                                            : impl->get_bound_buffer(link->def->pass).name; // the only def we have is the binding
			}
		}

		return {};
	}
} // namespace vuk
