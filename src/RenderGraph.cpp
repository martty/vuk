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

namespace vuk {
	/*
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
	}*/

	Result<void> build_links(std::span<std::shared_ptr<RG>> rgs, DefUseMap& res_to_links, std::vector<Node*>& pass_reads) {
		// build edges into link map
		// reserving here to avoid rehashing map
		res_to_links.clear();
		res_to_links.reserve(100);

		for (auto& rg : rgs) {
			// in each RG module, look at the nodes
			// declare -> clear -> call(R) -> call(W) -> release
			//   A     ->  B    ->  B      ->   C     -> X
			// declare: def A -> new entry
			// clear: undef A, def B
			// call(R): read B
			// call(W): undef B, def C
			// release: undef C
			for (auto& node : rg->op_arena) {
				switch (node.kind) {
				case Node::NOP:
				case Node::CONSTANT:
					break;
				case Node::VALLOC:
					res_to_links[first(&node)].def = first(&node);
					res_to_links[first(&node)].type = first(&node).type();
					break;
				case Node::AALLOC:
					for (size_t i = 0; i < node.aalloc.args.size(); i++) {
						auto& parm = node.aalloc.args[i];
						res_to_links[parm].undef = &node;
					}

					res_to_links[first(&node)].def = first(&node);
					res_to_links[first(&node)].type = first(&node).type();
					break;
				case Node::CALL: {
					// args
					for (size_t i = 0; i < node.call.args.size(); i++) {
						auto& arg_ty = node.call.fn.type()->opaque_fn.args[i];
						auto& parm = node.call.args[i];
						// TODO: assert same type when imbuement is stripped
						if (arg_ty->kind == Type::IMBUED_TY) {
							auto access = arg_ty->imbued.access;
							if (is_write_access(access) || access == Access::eConsume) { // Write and ReadWrite
								res_to_links[parm].undef = &node;
							}
							if (!is_write_access(access) && access != Access::eConsume) { // Read and ReadWrite
								res_to_links[parm].reads.append(pass_reads, &node);
							}
						} else {
							assert(0);
						}
					}
					size_t index = 0;
					for (auto& ret_t : node.type) {
						assert(ret_t->kind == Type::ALIASED_TY);
						auto ref_idx = ret_t->aliased.ref_idx;
						res_to_links[{ &node, index }].def = { &node, index };
						res_to_links[node.call.args[ref_idx]].next = &res_to_links[{ &node, index }];
						res_to_links[{ &node, index }].prev = &res_to_links[node.call.args[ref_idx]];
						index++;
					}
					break;
				}
				case Node::RELEASE:
					res_to_links[node.release.src].undef = &node;
					break;

				case Node::INDEXING:
					res_to_links[first(&node)].def = first(&node);
					res_to_links[first(&node)].type = first(&node).type()->array.T;
					break;
				default:
					assert(0);
				}
			}
		}

		for (auto& [ref, link] : res_to_links) {
			if (link.urdef)
				continue;
			if (!link.prev) { // from head to tails, propagate
				auto l = &link;
				do {
					l->urdef = link.def;
					l = l->next;
				} while (l);
			}
		}

		// second pass - resolve composite urdefs
		for (auto& rg : rgs) {
			for (auto& node : rg->op_arena) {
				switch (node.kind) {
				case Node::INDEXING:
					auto array_def = res_to_links[node.indexing.array].urdef;
					auto index_v = constant<uint64_t>(node.indexing.index);
					auto array_arg = array_def.node->aalloc.args[index_v + 1];
					assert(res_to_links[array_arg].urdef);
					res_to_links[first(&node)].urdef = res_to_links[array_arg].urdef;
				}
			}
		}

		return { expected_value };
	}

	Result<void> collect_chains(DefUseMap& res_to_links, std::vector<ChainLink*>& chains) {
		chains.clear();
		// collect chains by looking at links without a prev
		for (auto& [name, link] : res_to_links) {
			if (!link.prev) {
				chains.push_back(&link);
			}
		}

		return { expected_value };
	}

	/*
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
	}*/

	DomainFlagBits pick_first_domain(DomainFlags f) { // TODO: make this work
		return (DomainFlagBits)f.m_mask;
	}

	Result<void> RGCImpl::schedule_intra_queue(std::span<std::shared_ptr<RG>> rgs, const RenderGraphCompileOptions& compile_options) {
		// we need to schedule all execables that run
		std::vector<Node*> schedule_items;
		std::unordered_map<Node*, size_t> node_to_schedule;

		for (auto& rg : rgs) {
			for (auto& node : rg->op_arena) {
				switch (node.kind) {
				case Node::VALLOC:
				case Node::CALL:
				case Node::CLEAR:
				case Node::RELEASE:
					node_to_schedule[&node] = schedule_items.size();
					schedule_items.emplace_back(&node);
					break;
				}
			}
		}
		// calculate indegrees for all passes & build adjacency
		const size_t size = schedule_items.size();
		std::vector<size_t> indegrees(size);
		std::vector<uint8_t> adjacency_matrix(size * size);

		for (auto& [ref, link] : res_to_links) {
			if (link.undef && node_to_schedule.count(link.undef) && node_to_schedule.count(link.def.node)) {
				indegrees[node_to_schedule[link.undef]]++;
				adjacency_matrix[node_to_schedule[link.def.node] * size + node_to_schedule[link.undef]]++; // def -> undef
			}
			for (auto& read : link.reads.to_span(pass_reads)) {
				if (!node_to_schedule.count(read)) {
					continue;
				}

				if (node_to_schedule.count(link.def.node)) {
					indegrees[node_to_schedule[read]]++;                                                 // this only counts as a dep if there is a def before
					adjacency_matrix[node_to_schedule[link.def.node] * size + node_to_schedule[read]]++; // def -> read
				}

				if (link.undef && node_to_schedule.count(link.undef)) {
					indegrees[node_to_schedule[link.undef]]++;
					adjacency_matrix[node_to_schedule[read] * size + node_to_schedule[link.undef]]++; // read -> undef
				}
			}
		}

		// enqueue all indegree == 0 execables
		std::vector<size_t> process_queue;
		for (auto i = 0; i < indegrees.size(); i++) {
			if (indegrees[i] == 0)
				process_queue.push_back(i);
		}
		// dequeue indegree = 0 execables, add it to the ordered list, then decrement adjacent execables indegrees and push indegree == 0 to queue
		while (process_queue.size() > 0) {
			auto pop_idx = process_queue.back();
			auto& execable = schedule_items[pop_idx];
			ScheduledItem item{ .execable = execable,
				                  .scheduled_domain =
				                      execable->scheduling_info ? pick_first_domain(execable->scheduling_info->required_domain) : vuk::DomainFlagBits::eAny };
			if (execable->kind != Node::VALLOC) { // we use def nodes for deps, but we don't want to schedule them later as their ordering doesn't matter
				scheduled_execables.push_back(item);
			}
			process_queue.pop_back();
			for (auto i = 0; i < schedule_items.size(); i++) { // all the outgoing from this pass
				if (i == pop_idx) {
					continue;
				}
				auto adj_value = adjacency_matrix[pop_idx * size + i];
				if (adj_value > 0) {
					if (indegrees[i] -= adj_value; indegrees[i] == 0) {
						process_queue.push_back(i);
					}
				}
			}
		}
		return { expected_value };
	}

	/* Result<void> RGCImpl::relink_subchains() {
	  child_chains.clear();
	  // connect subchains
	  // diverging subchains are chains where def->pass >= 0 AND def->pass type is eDiverge
	  // reconverged subchains are chains where def->pass >= 0 AND def->pass type is eConverge
	  for (auto& head : chains) {
	    if (head->def->pass >= 0 && head->type->is_image()) { // no Buffer divergence
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
	  }*/
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

	// return { expected_value };

	// TODO: validate incorrect convergence
	/* else if (chain.back().high_level_access == Access::eConverge) {
	  assert(it->second.dst_use.layout != ImageLayout::eUndefined); // convergence into no use = disallowed
	}*/
	//
	//}

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

	Compiler::Compiler() : impl(new RGCImpl) {}
	Compiler::~Compiler() {
		delete impl;
	}
	void Compiler::queue_inference() {
		// queue inference pass
		DomainFlags last_domain = DomainFlagBits::eDevice;
		auto propagate_domain = [&last_domain](Node* node) {
			if (!node->scheduling_info) {
				return;
			}
			auto& domain = node->scheduling_info->required_domain;
			if (domain != last_domain && domain != DomainFlagBits::eDevice && domain != DomainFlagBits::eAny) {
				last_domain = domain;
			}
			if ((last_domain != DomainFlagBits::eDevice && last_domain != DomainFlagBits::eAny) &&
			    (domain == DomainFlagBits::eDevice || domain == DomainFlagBits::eAny)) {
				domain = last_domain;
			}
		};

		for (auto& head : impl->chains) {
			bool is_image = head->type->is_image();

			// forward inference
			ChainLink* chain;
			for (chain = head; chain != nullptr; chain = chain->next) {
				propagate_domain(chain->def.node);
				for (auto& r : chain->reads.to_span(impl->pass_reads)) {
					propagate_domain(r);
				}
				if (chain->undef) {
					propagate_domain(chain->undef);
				}
			}
		}

		// backward inference
		for (auto& head : impl->chains) {
			last_domain = DomainFlagBits::eDevice;

			ChainLink* chain;
			// wind chain to the end
			for (chain = head; chain->next != nullptr; chain = chain->next)
				;
			for (; chain != nullptr; chain = chain->prev) {
				if (chain->undef) {
					propagate_domain(chain->undef);
				}
				for (auto& r : chain->reads.to_span(impl->pass_reads)) {
					propagate_domain(r);
				}
				propagate_domain(chain->def.node);
			}
		}

		// queue inference failure fixup pass
		for (auto& p : impl->scheduled_execables) {
			if (p.scheduled_domain == DomainFlagBits::eDevice || p.scheduled_domain == DomainFlagBits::eAny) { // couldn't infer, set pass as graphics
				p.scheduled_domain = DomainFlagBits::eGraphicsQueue;
			}
		}
	}

	// partition passes into different queues
	void Compiler::pass_partitioning() {
		impl->partitioned_execables.reserve(impl->scheduled_execables.size());
		impl->scheduled_idx_to_partitioned_idx.resize(impl->scheduled_execables.size());
		for (size_t i = 0; i < impl->scheduled_execables.size(); i++) {
			auto& p = impl->scheduled_execables[i];
			if (p.scheduled_domain & DomainFlagBits::eTransferQueue) {
				impl->scheduled_idx_to_partitioned_idx[i] = impl->partitioned_execables.size();
				// impl->last_ordered_pass_idx_in_domain_array[2] = impl->ordered_idx_to_computed_pass_idx[i];
				impl->partitioned_execables.push_back(&p);
			}
		}
		impl->transfer_passes = { impl->partitioned_execables.begin(), impl->partitioned_execables.size() };
		for (size_t i = 0; i < impl->scheduled_execables.size(); i++) {
			auto& p = impl->scheduled_execables[i];
			if (p.scheduled_domain & DomainFlagBits::eComputeQueue) {
				impl->scheduled_idx_to_partitioned_idx[i] = impl->partitioned_execables.size();
				// impl->last_ordered_pass_idx_in_domain_array[1] = impl->ordered_idx_to_computed_pass_idx[i];
				impl->partitioned_execables.push_back(&p);
			}
		}
		impl->compute_passes = { impl->partitioned_execables.begin() + impl->transfer_passes.size(),
			                       impl->partitioned_execables.size() - impl->transfer_passes.size() };
		for (size_t i = 0; i < impl->scheduled_execables.size(); i++) {
			auto& p = impl->scheduled_execables[i];
			if (p.scheduled_domain & DomainFlagBits::eGraphicsQueue) {
				impl->scheduled_idx_to_partitioned_idx[i] = impl->partitioned_execables.size();
				// impl->last_ordered_pass_idx_in_domain_array[0] = impl->ordered_idx_to_computed_pass_idx[i];
				impl->partitioned_execables.push_back(&p);
			}
		}
		impl->graphics_passes = { impl->partitioned_execables.begin() + impl->transfer_passes.size() + impl->compute_passes.size(),
			                        impl->partitioned_execables.size() - impl->transfer_passes.size() - impl->compute_passes.size() };
	}
	/*
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
	}*/

	Result<void> Compiler::compile(std::span<std::shared_ptr<RG>> rgs, const RenderGraphCompileOptions& compile_options) {
		auto arena = impl->arena_.release();
		delete impl;
		arena->reset();
		impl = new RGCImpl(arena);
		impl->callbacks = compile_options.callbacks;

		std::vector<std::shared_ptr<RG>> work_queue(rgs.begin(), rgs.end());
		std::vector<std::shared_ptr<RG>> all_rgs;

		while (!work_queue.empty()) {
			auto item = work_queue.back();
			work_queue.pop_back();
			work_queue.insert(work_queue.end(), item->subgraphs.begin(), item->subgraphs.end());
			all_rgs.push_back(item);
		}
		// TODO:
		// impl->merge_diverge_passes(impl->computed_passes);

		// run global pass ordering - once we split per-queue we don't see enough
		// inputs to order within a queue

		VUK_DO_OR_RETURN(build_links(all_rgs, impl->res_to_links, impl->pass_reads));
		// VUK_DO_OR_RETURN(collect_chains(impl->res_to_links, impl->chains));

		// VUK_DO_OR_RETURN(impl->diagnose_unheaded_chains());
		VUK_DO_OR_RETURN(impl->schedule_intra_queue(all_rgs, compile_options));
		/*
		VUK_DO_OR_RETURN(impl->relink_subchains());
		resource_linking();
		VUK_DO_OR_RETURN(impl->fix_subchains());
		// fix subchains might remove chains, so drop those now
		std::erase(impl->chains, nullptr);
		// auto dumped_graph = dump_graph();
		*/

		queue_inference();
		pass_partitioning();
		/*render_pass_assignment();*/

		return { expected_value };
	}

	IARule same_extent_as(Name n) {
		return [=](const InferenceContext& ctx, ImageAttachment& ia) {
			// ia.extent = ctx.get_image_attachment(n).extent;
		};
	}

	IARule same_extent_as(TypedFuture<Image> inference_source) {
		return [=](const InferenceContext& ctx, ImageAttachment& ia) {
			// ia.extent = inference_source.attachment->extent;
		};
	}

	IARule same_2D_extent_as(Name n) {
		return [=](const InferenceContext& ctx, ImageAttachment& ia) {
			/* auto& o = ctx.get_image_attachment(n);
			ia.extent.sizing = o.extent.sizing;
			ia.extent.extent.width = o.extent.extent.width;
			ia.extent.extent.height = o.extent.extent.height;*/
		};
	}

	IARule same_format_as(Name n) {
		return [=](const InferenceContext& ctx, ImageAttachment& ia) {
			// ia.format = ctx.get_image_attachment(n).format;
		};
	}

	IARule same_shape_as(Name n) {
		return [=](const InferenceContext& ctx, ImageAttachment& ia) {
			/* auto& src = ctx.get_image_attachment(n);
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
			  ia.view_type = src.view_type;*/
		};
	}

	IARule similar_to(Name n) {
		return [=](const InferenceContext& ctx, ImageAttachment& ia) {
			/* auto& src = ctx.get_image_attachment(n);
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
			  ia.sample_count = src.sample_count;*/
		};
	}

	BufferRule same_size_as(Name inference_source) {
		return [=](const InferenceContext& ctx, Buffer& buf) {
			/* auto& src = ctx.get_buffer(inference_source);
			buf.size = src.size;*/
		};
	}

	bool crosses_queue(QueueResourceUse last_use, QueueResourceUse current_use) {
		return (last_use.domain != DomainFlagBits::eNone && last_use.domain != DomainFlagBits::eAny && current_use.domain != DomainFlagBits::eNone &&
		        current_use.domain != DomainFlagBits::eAny && (last_use.domain & DomainFlagBits::eQueueMask) != (current_use.domain & DomainFlagBits::eQueueMask));
	}

	VkImageMemoryBarrier2KHR RGCImpl::emit_image_barrier(QueueResourceUse last_use,
	                                                     QueueResourceUse current_use,
	                                                     const Subrange::Image& subrange,
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
		if (last_use.domain == DomainFlagBits::eAny || last_use.domain == DomainFlagBits::eHost) {
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

		barrier.srcStageMask = (VkPipelineStageFlags2)last_use.stages.m_mask;
		barrier.dstStageMask = (VkPipelineStageFlags2)current_use.stages.m_mask;

		return barrier;
	}

	VkImageMemoryBarrier2KHR emit_av_image_barrier(QueueResourceUse last_use,
	                                               QueueResourceUse current_use,
	                                               const Subrange::Image& subrange,
	                                               Image image,
	                                               ImageAspectFlags aspect,
	                                               bool is_release) {
		scope_to_domain((VkPipelineStageFlagBits2KHR&)last_use.stages, is_release ? last_use.domain : current_use.domain & DomainFlagBits::eQueueMask);

		// compute image barrier for this access -> access
		VkImageMemoryBarrier2KHR barrier{ .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2_KHR };
		barrier.srcAccessMask = is_read_access(last_use) ? 0 : (VkAccessFlags)last_use.access;
		barrier.dstAccessMask = (VkAccessFlags)current_use.access;
		barrier.oldLayout = (VkImageLayout)last_use.layout;
		barrier.newLayout = (VkImageLayout)last_use.layout;
		barrier.subresourceRange.aspectMask = (VkImageAspectFlags)aspect;
		barrier.subresourceRange.baseArrayLayer = subrange.base_layer;
		barrier.subresourceRange.baseMipLevel = subrange.base_level;
		barrier.subresourceRange.layerCount = subrange.layer_count;
		barrier.subresourceRange.levelCount = subrange.level_count;
		assert(last_use.domain.m_mask != 0);
		assert(current_use.domain.m_mask != 0);
		if (last_use.domain == DomainFlagBits::eAny || last_use.domain == DomainFlagBits::eHost) {
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

		barrier.srcStageMask = (VkPipelineStageFlags2)last_use.stages.m_mask;
		barrier.dstStageMask = (VkPipelineStageFlags2)current_use.stages.m_mask;

		return barrier;
	}
	/*
	Result<void> RGCImpl::generate_barriers_and_waits() {
	#ifdef VUK_DUMP_USE
	  fmt::printf("------------------------\n");
	  fmt::printf("digraph vuk {\n");
	#endif

	  // we need to handle chains in order of dependency
	  std::vector<ChainLink*> work_queue;
	  for (auto head : chains) {
	    bool is_image = head->type->is_image();
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
	    bool is_image = head->type->is_image();
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
	}*/

	Result<ExecutableRenderGraph> Compiler::link(std::span<std::shared_ptr<RG>> rgs, const RenderGraphCompileOptions& compile_options) {
		VUK_DO_OR_RETURN(compile(rgs, compile_options));

		/* VUK_DO_OR_RETURN(impl->generate_barriers_and_waits());

		VUK_DO_OR_RETURN(impl->merge_rps());

		VUK_DO_OR_RETURN(impl->assign_passes_to_batches());

		VUK_DO_OR_RETURN(impl->build_waits());

		// we now have enough data to build VkRenderPasses and VkFramebuffers
		VUK_DO_OR_RETURN(impl->build_renderpasses());*/

		return { expected_value, *this };
	}

	std::span<ChainLink*> Compiler::get_use_chains() const {
		return std::span(impl->chains);
	}

	ImageUsageFlags Compiler::compute_usage(const ChainLink* head) {
		return impl->compute_usage(head);
	}

	ImageUsageFlags RGCImpl::compute_usage(const ChainLink* head) {
		ImageUsageFlags usage = {};
		return usage;
	}
	/*
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
	}*/
} // namespace vuk
