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
	Result<void> RGCImpl::build_nodes() {
		nodes.clear();

		// type unification

		std::vector<Node*, short_alloc<Node*>> work_queue(*arena_);
		for (auto& ref : refs) {
			auto node = ref->get_node();
			if (node->flag == 0) {
				node->flag = 1;
				work_queue.push_back(node);
			}
		}

		type_map[Type::hash(cg_module->get_builtin_buffer())] = cg_module->get_builtin_buffer();
		type_map[Type::hash(cg_module->get_builtin_image())] = cg_module->get_builtin_image();
		type_map[Type::hash(cg_module->get_builtin_swapchain())] = cg_module->get_builtin_swapchain();

		while (!work_queue.empty()) {
			auto node = work_queue.back();
			work_queue.pop_back();

			auto count = node->generic_node.arg_count;
			if (count != (uint8_t)~0u) {
				for (int i = 0; i < count; i++) {
					auto arg = node->fixed_node.args[i].node;
					if (arg->flag == 0) {
						arg->flag = 1;
						work_queue.push_back(arg);
					}
				}
			} else {
				for (int i = 0; i < node->variable_node.args.size(); i++) {
					auto arg = node->variable_node.args[i].node;
					if (arg->flag == 0) {
						arg->flag = 1;
						work_queue.push_back(arg);
					}
				}
			}

			nodes.push_back(node);
			auto unify_type = [&](Type*& t) {
				auto [v, succ] = type_map.try_emplace(Type::hash(t), t);
				t = v->second;
			};
			for (auto& t : node->type) {
				if (t->kind == Type::ALIASED_TY) {
					unify_type(t->aliased.T);
				} else if (t->kind == Type::IMBUED_TY) {
					unify_type(t->imbued.T);
				} else if (t->kind == Type::ARRAY_TY) {
					unify_type(t->array.T);
				} else if (t->kind == Type::COMPOSITE_TY) {
					for (auto& elem_ty : t->composite.types) {
						unify_type(elem_ty);
					}
				}
				unify_type(t);
			}
		}

		cg_module->builtin_buffer = type_map[Type::hash(cg_module->builtin_buffer)];
		cg_module->builtin_image = type_map[Type::hash(cg_module->builtin_image)];
		cg_module->builtin_swapchain = type_map[Type::hash(cg_module->builtin_swapchain)];

		for (auto& node : nodes) {
			node->flag = 0;
		}

		return { expected_value };
	}

	Result<void> RGCImpl::build_links() {
		// build edges into link map
		// reserving here to avoid rehashing map
		pass_reads.clear();

		// in each RG module, look at the nodes
		// declare -> clear -> call(R) -> call(W) -> release
		//   A     ->  B    ->  B      ->   C     -> X
		// declare: def A -> new entry
		// clear: undef A, def B
		// call(R): read B
		// call(W): undef B, def C
		// release: undef C
		for (auto& node : nodes) {
			size_t result_count = node->type.size();
			if (result_count > 0) {
				node->links = new (cg_module->payload_arena.ensure_space(sizeof(ChainLink) * result_count)) ChainLink[result_count];
			}
		}

		for (auto& node : nodes) {
			switch (node->kind) {
			case Node::NOP:
			case Node::CONSTANT:
			case Node::PLACEHOLDER:
			case Node::MATH_BINARY:
				break;
			case Node::CONSTRUCT:
				for (size_t i = 0; i < node->construct.args.size(); i++) {
					auto& parm = node->construct.args[i];
					parm.link().undef = { node, i };
				}

				first(node).link().def = first(node);
				break;
			case Node::RELACQ: // ~~ write joiner
				for (size_t i = 0; i < node->relacq.src.size(); i++) {
					node->relacq.src[i].link().undef = { node, i };
					Ref{ node, i }.link().def = { node, i };
					node->relacq.src[i].link().next = &Ref{ node, i }.link();
					Ref{ node, i }.link().prev = &node->relacq.src[i].link();
				}
				break;
			case Node::ACQUIRE:
				first(node).link().def = first(node);
				break;
			case Node::CALL: {
				// args
				for (size_t i = 0; i < node->call.args.size(); i++) {
					auto& arg_ty = node->call.fn.type()->opaque_fn.args[i];
					auto& parm = node->call.args[i];
					// TODO: assert same type when imbuement is stripped
					if (arg_ty->kind == Type::IMBUED_TY) {
						auto access = arg_ty->imbued.access;
						if (is_write_access(access) || access == Access::eConsume) { // Write and ReadWrite
							parm.link().undef = { node, i };
						}
						if (!is_write_access(access) && access != Access::eConsume) { // Read and ReadWrite
							parm.link().reads.append(pass_reads, { node, i });
						}
					} else {
						assert(0);
					}
				}
				size_t index = 0;
				for (auto& ret_t : node->type) {
					assert(ret_t->kind == Type::ALIASED_TY);
					auto ref_idx = ret_t->aliased.ref_idx;
					Ref{ node, index }.link().def = { node, index };
					node->call.args[ref_idx].link().next = &Ref{ node, index }.link();
					Ref{ node, index }.link().prev = &node->call.args[ref_idx].link();
					index++;
				}
				break;
			}
			case Node::RELEASE:
				node->release.src.link().undef = { node, 0 };
				break;

			case Node::EXTRACT:
				first(node).link().def = first(node);
				break;

			case Node::SLICE:
				first(node).link().def = first(node);
				node->slice.image.link().child_chains.append(child_chains, &first(node).link());
				break;

			case Node::CONVERGE:
				node->converge.ref_and_diverged[0].link().undef = { node, 0 };
				first(node).link().def = first(node);
				node->converge.ref_and_diverged[0].link().next = &first(node).link();
				first(node).link().prev = &node->converge.ref_and_diverged[0].link();
				for (size_t i = 1; i < node->converge.ref_and_diverged.size(); i++) {
					auto& parm = node->converge.ref_and_diverged[i];
					parm.link().reads.append(pass_reads, { node, i });
				}
				break;

			case Node::INDIRECT_DEPEND: {
				auto rref = node->indirect_depend.rref;
				Ref true_ref;
				auto count = rref.node->generic_node.arg_count;
				if (count != (uint8_t)~0u) {
					true_ref = rref.node->fixed_node.args[rref.index];
				} else {
					true_ref = rref.node->variable_node.args[rref.index];
				}
				first(node).link().def = first(node);
				assert(true_ref.link().next == nullptr);
				true_ref.link().next = &first(node).link();
				first(node).link().prev = &true_ref.link();
				break;
			}

			case Node::ACQUIRE_NEXT_IMAGE:
				first(node).link().def = first(node);
				break;

			default:
				assert(0);
			}
		}

		for (auto& node : nodes) {
			size_t result_count = node->type.size();
			for (auto i = 0; i < result_count; i++) {
				auto& link = node->links[i];
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
		}

		// TODO:
		// we need a pass that walks through links
		// an incompatible read group contains multiple domains
		// in this case they can't be together - so we linearize them into domain groups
		// so def -> {r1, r2} becomes def -> r1 -> undef{g0} -> def{g0} -> r2

		return { expected_value };
	}

	Result<void> RGCImpl::reify_inference() {
		auto is_placeholder = [](Ref r) {
			return r.node->kind == Node::PLACEHOLDER;
		};

		auto placeholder_to_constant = [this]<class T>(Ref r, T value) {
			if (r.node->kind == Node::PLACEHOLDER) {
				r.node->kind = Node::CONSTANT;
				r.node->constant.value = new (cg_module->payload_arena.ensure_space(sizeof(T))) T(value);
			}
		};

		auto placeholder_to_ptr = []<class T>(Ref r, T* ptr) {
			if (r.node->kind == Node::PLACEHOLDER) {
				r.node->kind = Node::CONSTANT;
				r.node->constant.value = ptr;
			}
		};

		// valloc reification - if there were later setting of fields, then remove placeholders
		for (auto node : nodes) {
			switch (node->kind) {
			case Node::CONSTRUCT:
				auto args_ptr = node->construct.args.data();
				if (node->type[0] == cg_module->builtin_image) {
					auto ptr = &constant<ImageAttachment>(args_ptr[0]);
					auto& value = constant<ImageAttachment>(args_ptr[0]);
					if (value.extent.width > 0) {
						placeholder_to_ptr(args_ptr[1], &ptr->extent.width);
					}
					if (value.extent.height > 0) {
						placeholder_to_ptr(args_ptr[2], &ptr->extent.height);
					}
					if (value.extent.depth > 0) {
						placeholder_to_ptr(args_ptr[3], &ptr->extent.depth);
					}
					if (value.format != Format::eUndefined) {
						placeholder_to_ptr(args_ptr[4], &ptr->format);
					}
					if (value.sample_count != Samples::eInfer) {
						placeholder_to_ptr(args_ptr[5], &ptr->sample_count);
					}
					if (value.base_layer != VK_REMAINING_ARRAY_LAYERS) {
						placeholder_to_ptr(args_ptr[6], &ptr->base_layer);
					}
					if (value.layer_count != VK_REMAINING_ARRAY_LAYERS) {
						placeholder_to_ptr(args_ptr[7], &ptr->layer_count);
					}
					if (value.base_level != VK_REMAINING_MIP_LEVELS) {
						placeholder_to_ptr(args_ptr[8], &ptr->base_level);
					}
					if (value.level_count != VK_REMAINING_MIP_LEVELS) {
						placeholder_to_ptr(args_ptr[9], &ptr->level_count);
					}
				} else if (node->type[0] == cg_module->builtin_buffer) {
					auto ptr = &constant<Buffer>(args_ptr[0]);
					auto& value = constant<Buffer>(args_ptr[0]);
					if (value.size != ~(0u)) {
						placeholder_to_ptr(args_ptr[1], &ptr->size);
					}
				}
			}
		}

		for (auto node : nodes) {
			switch (node->kind) {
			case Node::CALL: {
				// args
				std::optional<Extent2D> extent;
				std::optional<Samples> samples;
				std::optional<uint32_t> layer_count;
				for (size_t i = 0; i < node->call.args.size(); i++) {
					auto& arg_ty = node->call.fn.type()->opaque_fn.args[i];
					auto& parm = node->call.args[i];
					if (arg_ty->kind == Type::IMBUED_TY) {
						auto access = arg_ty->imbued.access;
						auto& link = parm.link();
						if (link.urdef.node->kind == Node::CONSTRUCT) {
							auto& args = link.urdef.node->construct.args;
							if (is_framebuffer_attachment(access)) {
								if (is_placeholder(args[9])) {
									placeholder_to_constant(args[9], 1U); // can only render to a single mip level
								}
								if (is_placeholder(args[3])) {
									placeholder_to_constant(args[3], 1U); // depth must be 1
								}
								if (!samples && !is_placeholder(args[5])) { // known sample count
									samples = constant<Samples>(args[5]);
								} else if (samples && is_placeholder(args[5])) {
									placeholder_to_constant(args[5], *samples);
								}
								if (!extent && !is_placeholder(args[1]) && !is_placeholder(args[2])) { // known extent2D
									extent = Extent2D{ eval<uint32_t>(args[1]), eval<uint32_t>(args[2]) };
								} else if (extent && is_placeholder(args[1]) && is_placeholder(args[2])) {
									placeholder_to_constant(args[1], extent->width);
									placeholder_to_constant(args[2], extent->height);
								}
								if (!layer_count && !is_placeholder(args[7])) { // known layer count
									layer_count = eval<uint32_t>(args[7]);
								} else if (layer_count && is_placeholder(args[7])) {
									placeholder_to_constant(args[7], *layer_count);
								}
							}
						} else if (link.urdef.node->kind == Node::ACQUIRE_NEXT_IMAGE) {
							Swapchain& swp = *reinterpret_cast<Swapchain*>(link.urdef.node->acquire_next_image.swapchain.node->construct.args[0].node->constant.value);
							extent = Extent2D{ swp.images[0].extent.width, swp.images[0].extent.height };
							layer_count = swp.images[0].layer_count;
							samples = Samples::e1;
						}
					} else {
						assert(0);
					}
				}
				break;
			}
			case Node::CONSTRUCT: {
				auto& args = node->construct.args;
				if (node->type[0] == cg_module->builtin_image) {
					if (constant<ImageAttachment>(args[0]).image.image == VK_NULL_HANDLE) { // if there is no image, we will use base layer 0 and base mip 0
						placeholder_to_constant(args[6], 0U);
						placeholder_to_constant(args[8], 0U);
					}
				}
				break;
			}
			}
		}

		return { expected_value };
	}

	Result<void> RGCImpl::collect_chains() {
		chains.clear();
		// collect chains by looking at links without a prev
		for (auto& node : nodes) {
			size_t result_count = node->type.size();
			for (auto i = 0; i < result_count; i++) {
				auto& link = node->links[i];
				if (!link.prev) {
					chains.push_back(&link);
				}
			}
		}

		return { expected_value };
	}

	DomainFlagBits pick_first_domain(DomainFlags f) { // TODO: make this work
		return (DomainFlagBits)f.m_mask;
	}

	Result<void> RGCImpl::schedule_intra_queue(const RenderGraphCompileOptions& compile_options) {
		// we need to schedule all execables that run
		std::vector<Node*, short_alloc<Node*>> schedule_items(*arena_);
		robin_hood::unordered_flat_map<Node*, size_t> node_to_schedule;

		for (auto node : nodes) {
			switch (node->kind) {
			case Node::CONSTRUCT:
			case Node::CALL:
			case Node::CLEAR:
			case Node::ACQUIRE:
			case Node::MATH_BINARY:
			case Node::RELACQ:
			case Node::RELEASE:
			case Node::CONVERGE:
				node_to_schedule[node] = schedule_items.size();
				schedule_items.emplace_back(node);
				break;
			}
		}
		// calculate indegrees for all passes & build adjacency
		const size_t size = schedule_items.size();
		std::vector<size_t, short_alloc<size_t>> indegrees(size, *arena_);
		std::vector<uint8_t, short_alloc<uint8_t>> adjacency_matrix(size * size, *arena_);

		for (auto& node : nodes) {
			size_t result_count = node->type.size();
			for (auto i = 0; i < result_count; i++) {
				auto& link = node->links[i];
				if (link.undef && node_to_schedule.count(link.undef.node) && node_to_schedule.count(link.def.node)) {
					indegrees[node_to_schedule[link.undef.node]]++;
					adjacency_matrix[node_to_schedule[link.def.node] * size + node_to_schedule[link.undef.node]]++; // def -> undef
				}
				for (auto& read : link.reads.to_span(pass_reads)) {
					if (!node_to_schedule.count(read.node)) {
						continue;
					}

					if (node_to_schedule.count(link.def.node)) {
						indegrees[node_to_schedule[read.node]]++;                                                 // this only counts as a dep if there is a def before
						adjacency_matrix[node_to_schedule[link.def.node] * size + node_to_schedule[read.node]]++; // def -> read
					}

					if (link.undef && node_to_schedule.count(link.undef.node)) {
						indegrees[node_to_schedule[link.undef.node]]++;
						adjacency_matrix[node_to_schedule[read.node] * size + node_to_schedule[link.undef.node]]++; // read -> undef
					}
				}
			}
		}

		// enqueue all indegree == 0 execables
		std::vector<size_t, short_alloc<size_t>> process_queue(*arena_);
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
				                      execable->scheduling_info ? pick_first_domain(execable->scheduling_info->required_domains) : vuk::DomainFlagBits::eAny };
			if (execable->kind != Node::CONSTRUCT) { // we use def nodes for deps, but we don't want to schedule them later as their ordering doesn't matter
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

	Compiler::Compiler() : impl(new RGCImpl) {}
	Compiler::~Compiler() {
		delete impl;
	}
	void Compiler::queue_inference() {
		// queue inference pass
		DomainFlags last_domain = DomainFlagBits::eDevice;
		auto propagate_domain = [&last_domain](Node* node) {
			if (!node) {
				return;
			}
			if (!node->scheduling_info) {
				return;
			}
			auto& domain = node->scheduling_info->required_domains;
			if (domain != last_domain && domain != DomainFlagBits::eDevice && domain != DomainFlagBits::eAny) {
				last_domain = domain;
			}
			if ((last_domain != DomainFlagBits::eDevice && last_domain != DomainFlagBits::eAny) &&
			    (domain == DomainFlagBits::eDevice || domain == DomainFlagBits::eAny)) {
				domain = last_domain;
			}
		};

		for (auto& head : impl->chains) {
			// forward inference
			ChainLink* chain;
			for (chain = head; chain != nullptr; chain = chain->next) {
				propagate_domain(chain->def.node);
				for (auto& r : chain->reads.to_span(impl->pass_reads)) {
					propagate_domain(r.node);
				}
				if (chain->undef) {
					propagate_domain(chain->undef.node);
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
					propagate_domain(chain->undef.node);
				}
				for (auto& r : chain->reads.to_span(impl->pass_reads)) {
					propagate_domain(r.node);
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

		for (auto& head : impl->chains) {
			// forward inference
			ChainLink* chain;
			for (chain = head; chain != nullptr; chain = chain->next) {
				propagate_domain(chain->def.node);
				for (auto& r : chain->reads.to_span(impl->pass_reads)) {
					propagate_domain(r.node);
				}
				if (chain->undef) {
					propagate_domain(chain->undef.node);
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
					propagate_domain(chain->undef.node);
				}
				for (auto& r : chain->reads.to_span(impl->pass_reads)) {
					propagate_domain(r.node);
				}
				propagate_domain(chain->def.node);
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

	Result<void> Compiler::compile(std::span<std::shared_ptr<ExtNode>> nodes, const RenderGraphCompileOptions& compile_options) {
		auto arena = impl->arena_.release();
		delete impl;
		arena->reset();
		impl = new RGCImpl(arena);
		impl->callbacks = compile_options.callbacks;

		impl->cg_module = nodes[0]->module;

		impl->refs.assign(nodes.begin(), nodes.end());

		auto replace_refs = [&](Ref needle, Ref replace_with) {
			for (auto node : impl->nodes) {
				auto count = node->generic_node.arg_count;
				if (count != (uint8_t)~0u) {
					for (int i = 0; i < count; i++) {
						if (node->fixed_node.args[i] == needle) {
							node->fixed_node.args[i] = replace_with;
						}
					}
				} else {
					for (int i = 0; i < node->variable_node.args.size(); i++) {
						if (node->variable_node.args[i] == needle) {
							node->variable_node.args[i] = replace_with;
						}
					}
				}
			}
		};

		// implicit convergence: this has to be done on the full node set
		// insert converge nodes
		std::unordered_map<Ref, std::vector<Ref>> slices;

		// linked-sea-of-nodes to list of nodes
		std::vector<RG*, short_alloc<RG*>> work_queue(*impl->arena_);
		std::unordered_set<RG*> visited;
		for (auto& ref : impl->refs) {
			auto mod = ref->module.get();
			if (!visited.count(mod)) {
				work_queue.push_back(mod);
				visited.emplace(mod);
			}
		}

		while (!work_queue.empty()) {
			auto mod = work_queue.back();
			work_queue.pop_back();
			for (auto& sg : mod->subgraphs) {
				auto sg_mod = sg.get();
				if (!visited.count(sg_mod)) {
					work_queue.push_back(sg_mod);
					visited.emplace(sg_mod);
				}
			}
			for (auto& n : mod->op_arena) {
				auto node = &n;
				if (node->kind != Node::CONVERGE) {
					impl->nodes.push_back(node);
				}
				switch (node->kind) {
				case Node::SLICE: {
					slices[node->slice.image].push_back(first(node));
					break;
				}
				}
			}
		}

		// build links for the full node set
		VUK_DO_OR_RETURN(impl->build_links());

		// insert converge nodes
		auto in_module = [](RG& module, Node* node) {
			auto it = std::find_if(module.op_arena.begin(), module.op_arena.end(), [=](auto& n) { return &n == node; });
			if (it != module.op_arena.end()) {
				return true;
			}
			return false;
		};

		auto before_module = [](RG& module, Node* a, Node* b) {
			auto it_a = std::find_if(module.op_arena.begin(), module.op_arena.end(), [=](auto& n) { return &n == a; });
			auto it_b = std::find_if(module.op_arena.begin(), module.op_arena.end(), [=](auto& n) { return &n == b; });
			return it_a < it_b;
		};

		for (auto& [base, sliced] : slices) {
			std::vector<Ref, short_alloc<Ref>> tails(*impl->arena_);
			for (auto& s : sliced) {
				auto r = &s.link();
				while (r->next) {
					r = r->next;
				}
				if (r->undef.node) { // depend on undefs indirectly via INDIRECT_DEPEND
					auto idep = impl->cg_module->make_indirect_depend(r->undef.node, r->undef.index);
					tails.push_back(idep);
				} else {
					tails.push_back(r->def); // depend on defs directly
				}
			}
			auto converged_base = impl->cg_module->make_converge(base, tails);
			for (auto node : impl->nodes) {
				if (node->kind == Node::SLICE) {
					continue;
				}

				auto count = node->generic_node.arg_count;
				if (count != (uint8_t)~0u) {
					for (int i = 0; i < count; i++) {
						if (node->fixed_node.args[i] == base) {
							for (auto& t : tails) {
								// TODO: multimodule dominance
								/*
								assert(in_module(*impl->cg_module, t.node));
								if (!before_module(*impl->cg_module, t.node, node)) {
								  return { expected_error, RenderGraphException{ "Convergence not dominated" } };
								}
								*/
							}
							node->fixed_node.args[i] = converged_base;
						}
					}
				} else {
					for (int i = 0; i < node->variable_node.args.size(); i++) {
						if (node->variable_node.args[i] == base) {
							for (auto& t : tails) {
								// TODO: multimodule dominance
								/* assert(in_module(*impl->cg_module, t.node));
								if (!before_module(*impl->cg_module, t.node, node)) {
								  return { expected_error, RenderGraphException{ "Convergence not dominated" } };
								}*/
							}
							node->variable_node.args[i] = converged_base;
						}
					}
				}
			}
		}

		VUK_DO_OR_RETURN(impl->build_nodes());

		// eliminate useless relacqs

		for (auto node : impl->nodes) {
			switch (node->kind) {
			case Node::RELACQ: {
				if (node->relacq.rel_acq == nullptr) {
					for (size_t i = 0; i < node->relacq.src.size(); i++) {
						auto needle = Ref{ node, i };
						auto replace_with = node->relacq.src[i];

						replace_refs(needle, replace_with);
					}
				} else {
					switch (node->relacq.rel_acq->status) {
					case Signal::Status::eDisarmed: // means we have to signal this, keep
						break;
					case Signal::Status::eSynchronizable: // means this is an acq instead
					case Signal::Status::eHostAvailable:
						for (size_t i = 0; i < node->relacq.src.size(); i++) {
							auto new_ref = impl->cg_module->make_acquire(node->type[i], node->relacq.rel_acq, i, node->relacq.values[i]);
							replace_refs(Ref{ node, i }, new_ref);
						}
						break;
					}
				}
				break;
			}
			}
		}

		VUK_DO_OR_RETURN(impl->build_nodes());

		VUK_DO_OR_RETURN(impl->build_links());
		VUK_DO_OR_RETURN(impl->reify_inference());
		VUK_DO_OR_RETURN(impl->collect_chains());

		VUK_DO_OR_RETURN(impl->schedule_intra_queue(compile_options));

		queue_inference();
		pass_partitioning();

		return { expected_value };
	}

	Result<ExecutableRenderGraph> Compiler::link(std::span<std::shared_ptr<ExtNode>> nodes, const RenderGraphCompileOptions& compile_options) {
		VUK_DO_OR_RETURN(compile(nodes, compile_options));

		return { expected_value, *this };
	}

	std::span<ChainLink*> Compiler::get_use_chains() const {
		return std::span(impl->chains);
	}

	void* Compiler::get_value(Ref parm) {
		return impl->get_value(parm);
	}

	ImageUsageFlags Compiler::compute_usage(const ChainLink* head) {
		return impl->compute_usage(head);
	}

	constexpr void access_to_usage(ImageUsageFlags& usage, Access acc) {
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

	ImageUsageFlags RGCImpl::compute_usage(const ChainLink* head) {
		ImageUsageFlags usage = {};

		for (auto chain = head; chain != nullptr; chain = chain->next) {
			for (auto& r : chain->reads.to_span(pass_reads)) {
				switch (r.node->kind) {
				case Node::CALL: {
					auto& arg_ty = r.node->call.fn.type()->opaque_fn.args[r.index];
					auto& parm = r.node->call.args[r.index];
					if (arg_ty->kind == Type::IMBUED_TY) {
						auto access = arg_ty->imbued.access;
						access_to_usage(usage, access);
					}
				}
				}
			}
			if (chain->undef) {
				switch (chain->undef.node->kind) {
				case Node::CALL: {
					auto& arg_ty = chain->undef.node->call.fn.type()->opaque_fn.args[chain->undef.index];
					auto& parm = chain->undef.node->call.args[chain->undef.index];
					if (arg_ty->kind == Type::IMBUED_TY) {
						auto access = arg_ty->imbued.access;
						access_to_usage(usage, access);
					}
				}
				}
			}

			for (auto& child_chain : chain->child_chains.to_span(child_chains)) {
				usage |= compute_usage(child_chain);
			}
		}

		return usage;
	}
} // namespace vuk
