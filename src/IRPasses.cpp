#include "GraphDumper.hpp"
#include "vuk/Exception.hpp"
#include "vuk/IRProcess.hpp"
#include "vuk/RenderGraph.hpp"
#include "vuk/SyncLowering.hpp"
#include "vuk/Value.hpp"
#include "vuk/runtime/CommandBuffer.hpp"
#include "vuk/runtime/vk/VkRuntime.hpp"

#include <bit>
#include <charconv>
#include <fmt/printf.h>
#include <memory_resource>
#include <random>
#include <set>
#include <unordered_set>

namespace {
	using Generator = std::mt19937;
	thread_local static std::random_device rd;

	thread_local static Generator _random_generator = []() {
		std::array<typename Generator::result_type, Generator::state_size> data;
		std::generate(std::begin(data), std::end(data), std::ref(rd));
		auto seq = std::seed_seq(std::begin(data), std::end(data));
		return Generator(seq);
	}();
} // namespace

#define VUK_ENABLE_ICE

#ifndef VUK_ENABLE_ICE
#define VUK_ICE(expression) (void)((!!(expression)) || assert(expression))
#else
#define VUK_ICE(expression) (void)((!!(expression)) || (GraphDumper::end_cluster(), GraphDumper::end_graph(), false) || (assert(expression), false))
#endif

namespace vuk {
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

	void IRModule::collect_garbage() {
		std::pmr::monotonic_buffer_resource mbr;
		std::pmr::polymorphic_allocator<std::byte> allocator(&mbr);
		collect_garbage(allocator);
	}

	void IRModule::collect_garbage(std::pmr::polymorphic_allocator<std::byte> allocator) {
		std::pmr::vector<Node*> liveness_work_queue(allocator);
		std::pmr::unordered_set<Node*> live_set(allocator);

		// initial set of live nodes
		for (auto it = op_arena.begin(); it != op_arena.end();) {
			auto node = &*it;
			// if the node is garbage, just collect it now
			if (node->kind == Node::GARBAGE) {
				it = op_arena.erase(it);
				continue;
			}
			if (node->kind == Node::SPLICE) {
				// dropped splices not in the initial set
				if(!node->splice.held) {
					++it;
					continue;
				} else { // held splices are always live
					live_set.emplace(node);
				}
			}
			if (node->index < (module_id << 32 | link_frontier)) {
				if (node->kind != Node::SPLICE) { // non-splice nodes before the link frontier are not in the initial set
					++it;
					continue;
				}
				// splice nodes that are not potential garbage are in the initial set even before the link frontier
			}
			// everything else is in the initial set
			liveness_work_queue.emplace_back(node);
			++it;
		}

		// compute live set
		while (!liveness_work_queue.empty()) {
			auto node = liveness_work_queue.back();
			liveness_work_queue.pop_back();
			apply_generic_args([&](Ref parm) { liveness_work_queue.push_back(parm.node); }, node);
			live_set.emplace(node);
		}

		// GC the module
		for (auto it = op_arena.begin(); it != op_arena.end(); ++it) {
			auto node = &*it;
			if (!live_set.contains(node)) {
				garbage.push_back(node);
			}
		}
		for (auto& node : garbage) {
			destroy_node(node);
		}
		garbage.clear();
	}

	Compiler::Compiler() : impl(new RGCImpl) {}
	Compiler::~Compiler() {
		delete impl;
	}

	void Compiler::reset() {
		auto arena = impl->arena_.release();
		delete impl;
		arena->reset();
		impl = new RGCImpl(arena);
	}

	template<class It>
	std::pmr::vector<Node*> collect_dependents(It start, It end, std::pmr::polymorphic_allocator<std::byte> allocator) {
		std::pmr::vector<Node*> work_queue(allocator);
		std::pmr::vector<Node*> nodes(allocator);
		for (auto it = start; it != end; ++it) {
			auto node = *it;
			if (node->flag == 0) {
				node->flag = 1;
				work_queue.push_back(node);
			}
		}

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
		}

		for (auto& node : nodes) {
			node->flag = 0;
		}

		return nodes;
	}

	Result<void> RGCImpl::build_nodes() {
		nodes.clear();

		std::vector<Node*, short_alloc<Node*>> work_queue(*arena_);
		for (auto& node : ref_nodes) {
			if (node->flag == 0) {
				node->flag = 1;
				work_queue.push_back(node);
			}
		}

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
		}

		for (auto& node : nodes) {
			node->flag = 0;
		}

		return { expected_value };
	}

	void allocate_node_links(Node* node, std::pmr::polymorphic_allocator<std::byte> allocator) {
		size_t result_count = node->type.size();
		if (result_count > 0) {
			node->links = new (allocator.allocate_bytes(sizeof(ChainLink) * result_count)) ChainLink[result_count];
		}
	}

	void collect_tails(Ref head, std::pmr::vector<Ref>& tails, std::pmr::vector<Ref>& pass_reads) {
		auto link = &head.link();

		if (link->next) {
			do {
				if (link->undef && link->undef.node->kind == Node::SLICE) {
					collect_tails(nth(link->undef.node, 0), tails, pass_reads);
					collect_tails(nth(link->undef.node, 1), tails, pass_reads);
				}
				link = link->next;
			} while (link->next);
		}

		if (link->undef && link->undef.node->kind == Node::SLICE) {
			collect_tails(nth(link->undef.node, 0), tails, pass_reads);
			collect_tails(nth(link->undef.node, 1), tails, pass_reads);
		} else if (link->undef) {
			tails.push_back(link->undef); // TODO: RREF
		} else if (link->reads.size() > 0) {
			for (auto read : link->reads.to_span(pass_reads)) {
				tails.push_back(read);
			}
		} else if (link->def.node->kind != Node::SLICE) {
			tails.push_back(link->def);
		}
	}

	void RGCImpl::process_node_links(IRModule* module,
	                                 Node* node,
	                                 std::pmr::vector<Ref>& pass_reads,
	                                 std::pmr::vector<Ref>& pass_nops,
	                                 std::pmr::vector<ChainLink*>& child_chains,
	                                 std::pmr::vector<Node*>& new_nodes,
	                                 std::pmr::polymorphic_allocator<std::byte> allocator,
	                                 bool do_ssa) {
		auto walk_writes = [&](Ref parm, Subrange::Image requested) -> Ref {
			auto link = &parm.link();
			Ref last_write;

			MultiSubrange current_range = MultiSubrange::all();

			do {
				if (link->undef && link->undef.node->kind == Node::SLICE) {
					auto& slice = link->undef.node->slice;
					// TODO: support const eval here
					Subrange::Image existing_slice_range = { constant<uint32_t>(slice.base_level),
						                                       constant<uint32_t>(slice.level_count),
						                                       constant<uint32_t>(slice.base_layer),
						                                       constant<uint32_t>(slice.layer_count) };
					auto left = current_range.set_intersect(existing_slice_range);
					if (auto isection = left.set_intersect(requested)) {        // requested range overlaps with split -> we might need to converge
						if (!MultiSubrange(requested).set_difference(isection)) { // if fully contained in the left -> no converge needed
							link = &nth(link->undef.node, 0).link();
							current_range = left;
						} else { // requested range is partially in left and in right -> converge needed of the tails
							std::pmr::vector<Ref> tails;
							// walk left and walk right
							collect_tails(nth(link->undef.node, 0), tails, pass_reads);
							collect_tails(nth(link->undef.node, 1), tails, pass_reads);
							std::pmr::vector<char> ws(tails.size(), true);

							last_write = module->make_converge(tails, ws);
							garbage_nodes.push_back(last_write.node);
							last_write.node->index = node->index - 1;
							allocate_node_links(last_write.node, allocator);
							link->undef = last_write;
							link->next = &last_write.link();
							last_write.link().prev = link;
							last_write.link().def = last_write;
							new_nodes.push_back(last_write.node);
							break;
						}
					} else { // requested range is fully contained in rest, switch to rest
						link = &nth(link->undef.node, 1).link();
						auto right = current_range.set_difference(left);
						current_range = right;
					}
				} else if (link->undef && link->undef.node->kind == Node::CONVERGE) {
					// TODO: this does not support walking converges properly yet!
					current_range = MultiSubrange::all();
				}
				if (link->next) {
					link = link->next;
				}
			} while (link->next || link->child_chains.size() > 0);

			if (!last_write.node) {
				assert(!link->undef);
				last_write = link->def;
			}

			return last_write;
		};

		auto add_breaking_result = [&](Node* node, size_t output_idx) {
			Ref out = Ref{ node, output_idx };
			out.link().def = { node, output_idx };
		};

		auto add_result = [&](Node* node, size_t output_idx, Ref parm) {
			Ref out = Ref{ node, output_idx };
			out.link().def = { node, output_idx };

			bool see_through_splice = parm.node->kind == Node::SPLICE && parm.node->splice.dst_access == Access::eNone &&
			                          parm.node->splice.dst_domain == DomainFlagBits::eAny &&
			                          (!parm.node->splice.rel_acq || parm.node->splice.rel_acq->status == Signal::Status::eDisarmed);
			auto& st_parm = see_through_splice ? parm.node->splice.src[parm.index] : parm;
			if (!st_parm.node->links) {
				assert(do_ssa);
				return;
			}

			auto link = &st_parm.link();
			auto& prev = out.link().prev;
			if (!do_ssa) {
				VUK_ICE(!link->next);
				assert(!prev);
			}
			link->next = &out.link();
			prev = link;
		};

		auto add_write = [&](Node* node, Ref& parm, size_t index, Subrange::Image requested = {}) -> void {
			assert(parm.node->kind != Node::GARBAGE);
			bool see_through_splice = parm.node->kind == Node::SPLICE && parm.node->splice.dst_access == Access::eNone &&
			                          parm.node->splice.dst_domain == DomainFlagBits::eAny &&
			                          (!parm.node->splice.rel_acq || parm.node->splice.rel_acq->status == Signal::Status::eDisarmed);
			auto& st_parm = see_through_splice ? parm.node->splice.src[parm.index] : parm;
			if (!st_parm.node->links) {
				assert(do_ssa);
				// external node -> init
				allocate_node_links(st_parm.node, allocator);
				for (size_t i = 0; i < st_parm.node->type.size(); i++) {
					Ref{ st_parm.node, 0 }.link().def = { st_parm.node, i };
				}
			}
			auto link = &st_parm.link();
			if (link->undef.node != nullptr) { // there is already a write -> do SSA rewrite
				assert(do_ssa);
				auto old_ref = link->undef;                 // this is an rref
				assert(node->index >= old_ref.node->index); // we are after the existing write

				// attempt to find the final revision of this
				// this could be either the last write on the main chain, or the last write on a child chain
				auto last_write = walk_writes(see_through_splice ? parm.node->splice.src[parm.index] : parm, requested);
				parm = last_write;
				link = &parm.link();
			}
			link->undef = { node, index };
		};

		auto add_read = [&](Node* node, Ref& parm, size_t index) {
			assert(parm.node->kind != Node::GARBAGE);
			bool see_through_splice = parm.node->kind == Node::SPLICE && parm.node->splice.dst_access == Access::eNone &&
			                          parm.node->splice.dst_domain == DomainFlagBits::eAny &&
			                          (!parm.node->splice.rel_acq || parm.node->splice.rel_acq->status == Signal::Status::eDisarmed);
			auto& st_parm = see_through_splice ? parm.node->splice.src[parm.index] : parm;
			if (!st_parm.node->links) {
				assert(do_ssa);
				// external node -> init
				allocate_node_links(st_parm.node, allocator);
				for (size_t i = 0; i < st_parm.node->type.size(); i++) {
					Ref{ st_parm.node, 0 }.link().def = { st_parm.node, i };
				}
			}
			auto link = &st_parm.link();
			if (link->undef.node != nullptr && node->index > link->undef.node->index) { // there is already a write and it is earlier than us
				assert(do_ssa);
				auto last_write = walk_writes(see_through_splice ? parm.node->splice.src[parm.index] : parm, {});
				parm = last_write;
				link = &parm.link();
			}
			link->reads.append(pass_reads, { node, index });
		};

		switch (node->kind) {
		case Node::CONSTANT:
		case Node::PLACEHOLDER:
			break;
		case Node::CONSTRUCT:
			first(node).link().def = first(node);

			for (size_t i = 0; i < node->construct.args.size(); i++) {
				auto& parm = node->construct.args[i];
				parm.link().undef = { node, i };
			}

			if (node->type[0]->kind == Type::ARRAY_TY || node->type[0]->hash_value == current_module->types.builtin_sampled_image) {
				for (size_t i = 1; i < node->construct.args.size(); i++) {
					auto& parm = node->construct.args[i];
					parm.link().next = &first(node).link();
				}
			}

			break;
		case Node::MATH_BINARY:
			add_read(node, node->math_binary.a, 0);
			add_read(node, node->math_binary.b, 1);
			add_breaking_result(node, 0);
			break;
		case Node::SPLICE: { // ~~ nop joiner
			                   /*
                 " results must be through splices "
                 a -> splice
			             -> b (r or w)
                 must rewrite into
                 a -> splice -> b
			             
                 " dependencies must not bypass splices "
                 a -> splice -> b
                 must not rewrite into
                 a -> b
			             
                 splice -> a -> splice
                 -> b
                 must not rewrite into
                 splice -> a -> splice -> b
			                 */
			for (size_t i = 0; i < node->type.size(); i++) {
				if (!node->splice.rel_acq || (node->splice.rel_acq && node->splice.rel_acq->status == Signal::Status::eDisarmed)) {
					if (node->splice.dst_access == Access::eNone && node->splice.dst_domain == DomainFlagBits::eAny) {
						if (do_ssa) {
							add_write(node, node->splice.src[i], i);
							add_result(node, i, node->splice.src[i]);
						} else {
							node->splice.src[i].link().nops.append(pass_nops, { node, i });
							Ref{ node, i }.link().def = { node, i };
							Ref{ node, i }.link().prev = &node->splice.src[i].link();
						}

					} else { // releases are still writes...
						add_write(node, node->splice.src[i], i);
						add_result(node, i, node->splice.src[i]);
					}
				} else { // acquire
					Ref{ node, i }.link().def = { node, i };
				}
			}
			break;
		}
		case Node::CALL: {
			// args
			auto fn_type = node->call.args[0].type();
			size_t first_parm = fn_type->kind == Type::OPAQUE_FN_TY ? 1 : 4;
			auto& args = fn_type->kind == Type::OPAQUE_FN_TY ? fn_type->opaque_fn.args : fn_type->shader_fn.args;
			for (size_t i = first_parm; i < node->call.args.size(); i++) {
				auto& arg_ty = args[i - first_parm];
				auto& parm = node->call.args[i];
				// TODO: assert same type when imbuement is stripped
				if (arg_ty->kind == Type::IMBUED_TY) {
					auto access = arg_ty->imbued.access;
					if (is_write_access(access)) { // Write and ReadWrite
						add_write(node, parm, i);
					}
					if (!is_write_access(access)) { // Read and ReadWrite
						add_read(node, parm, i);
					}
					auto& base = *arg_ty->imbued.T;
					if (do_ssa && base->hash_value == current_module->types.builtin_image) {
						auto def = get_def2(parm);
						if (def && def->node->kind == Node::CONSTRUCT) {
							auto& ia = *reinterpret_cast<ImageAttachment*>(def->node->construct.args[0].node->constant.value);
							if (!ia.image) {
								access_to_usage(ia.usage, access);
							}
						} else if (def && (def->node->kind == Node::SPLICE || def->node->kind == Node::ACQUIRE_NEXT_IMAGE)) { // nop
						} else if (!def) {
						} else {
							assert(0);
						}
					}
				} else {
					assert(0);
				}
			}
			size_t index = 0;
			for (auto& ret_t : node->type) {
				assert(ret_t->kind == Type::ALIASED_TY);
				auto ref_idx = ret_t->aliased.ref_idx;
				auto& arg_ty = args[ref_idx - first_parm];
				if (arg_ty->kind == Type::IMBUED_TY) {
					auto access = arg_ty->imbued.access;
					if (is_write_access(access)) {
						add_result(node, index, node->call.args[ref_idx]);
					} else {
						Ref{ node, index }.link().def = { node, index };
						Ref{ node, index }.link().prev = &node->call.args[ref_idx].link();
					}
				} else {
					assert(0);
				}
				index++;
			}
			break;
		}

		case Node::EXTRACT:
			first(node).link().def = first(node);
			break;

		case Node::SLICE: {
			Subrange::Image slice_range = { constant<uint32_t>(node->slice.base_level),
				                              constant<uint32_t>(node->slice.level_count),
				                              constant<uint32_t>(node->slice.base_layer),
				                              constant<uint32_t>(node->slice.layer_count) };
			add_write(node, node->slice.image, 0, slice_range);
			nth(node, 0).link().def = nth(node, 0); // we introduce the slice image def
			nth(node, 1).link().def = nth(node, 1); // we introduce the rest image def
			if (node->slice.image.node->links) {
				node->slice.image.link().child_chains.append(child_chains, &nth(node, 0).link()); // add child chain for sliced
			} else {
				assert(do_ssa);
			}

			break;
		}
		case Node::CONVERGE:
			first(node).link().def = first(node);
			node->converge.diverged[0].link().next = &first(node).link();
			first(node).link().prev = &node->converge.diverged[0].link();
			for (size_t i = 0; i < node->converge.diverged.size(); i++) {
				auto& parm = node->converge.diverged[i];
				auto write = node->converge.write[i];
				if (write) {
					add_write(node, parm, i);
				} else {
					add_read(node, parm, i);
				}
			}
			break;

		case Node::ACQUIRE_NEXT_IMAGE:
			first(node).link().def = first(node);
			break;

		case Node::GARBAGE:
			break;

		default:
			assert(0);
		}
	}

	void build_urdef(Node* node) {
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

	Result<void> RGCImpl::build_links(std::vector<Node*>& working_set, std::pmr::polymorphic_allocator<std::byte> allocator) {
		pass_reads.clear();
		pass_nops.clear();
		child_chains.clear();

		// in each IRModule module, look at the nodes
		// declare -> clear -> call(R) -> call(W) -> release
		//   A     ->  B    ->  B      ->   C     -> X
		// declare: def A -> new entry
		// clear: undef A, def B
		// call(R): read B
		// call(W): undef B, def C
		// release: undef C
		for (auto& node : working_set) {
			allocate_node_links(node, allocator);
		}

		std::pmr::vector<Node*> new_nodes;
		for (auto& node : working_set) {
			process_node_links(current_module.get(), node, pass_reads, pass_nops, child_chains, new_nodes, allocator, false);
		}

		// fixup splice links - they are bridged
		for (auto& node : working_set) {
			switch (node->kind) {
			case Node::SPLICE: {
				for (size_t i = 0; i < node->type.size(); i++) {
					if (!node->splice.rel_acq || (node->splice.rel_acq && node->splice.rel_acq->status == Signal::Status::eDisarmed)) {
						if (node->splice.dst_access == Access::eNone && node->splice.dst_domain == DomainFlagBits::eAny) {
							Ref{ node, i }.link() = node->splice.src[i].link();
						}
					}
				}
			}
			}
		}

		working_set.insert(working_set.end(), new_nodes.begin(), new_nodes.end());

		// build URDEF
		// TODO: remove?, replace with get_def
		for (auto& node : working_set) {
			build_urdef(node);
		}

		// TODO:
		// we need a pass that walks through links
		// an incompatible read group contains multiple domains
		// in this case they can't be together - so we linearize them into domain groups
		// so def -> {r1, r2} becomes def -> r1 -> undef{g0} -> def{g0} -> r2

		return { expected_value };
	}

	template<class It>
	Result<void> RGCImpl::build_links(IRModule* module,
	                                  It start,
	                                  It end,
	                                  std::pmr::vector<Ref>& pass_reads,
	                                  std::pmr::vector<Ref>& pass_nops,
	                                  std::pmr::vector<ChainLink*>& child_chains,
	                                  std::pmr::polymorphic_allocator<std::byte> allocator) {
		std::pmr::vector<Node*> new_nodes(allocator);
		for (auto it = start; it != end; ++it) {
			allocate_node_links(*it, allocator);
		}
		for (auto it = start; it != end; ++it) {
			process_node_links(module, *it, pass_reads, pass_nops, child_chains, new_nodes, allocator, true);
		}
		for (auto it = start; it != end; ++it) {
			build_urdef(*it);
		}

		return { expected_value };
	}

	Result<void> RGCImpl::reify_inference() {
		auto is_placeholder = [](Ref r) {
			return r.node->kind == Node::PLACEHOLDER;
		};

		bool progress = false;

		auto placeholder_to_constant = [&progress]<class T>(Ref r, T value) {
			if (r.node->kind == Node::PLACEHOLDER) {
				r.node->kind = Node::CONSTANT;
				assert(sizeof(T) == r.type()->size);
				r.node->constant.value = new char[sizeof(T)];
				new (r.node->constant.value) T(value);
				r.node->constant.owned = true;
				progress = true;
			}
		};

		auto placeholder_to_ptr = []<class T>(Ref r, T* ptr) {
			if (r.node->kind == Node::PLACEHOLDER) {
				r.node->kind = Node::CONSTANT;
				r.node->constant.value = ptr;
				r.node->constant.owned = false;
			}
		};

		// valloc reification - if there were later setting of fields, then remove placeholders
		for (auto node : nodes) {
			switch (node->kind) {
			case Node::CONSTRUCT: {
				auto args_ptr = node->construct.args.data();
				if (node->type[0]->hash_value == current_module->types.builtin_image) {
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
				} else if (node->type[0]->hash_value == current_module->types.builtin_buffer) {
					auto ptr = &constant<Buffer>(args_ptr[0]);
					auto& value = constant<Buffer>(args_ptr[0]);
					if (value.size != ~(0u)) {
						placeholder_to_ptr(args_ptr[1], &ptr->size);
					}
				}
			} break;
			default:
				break;
			}
		}

		// framebuffer inference
		do {
			progress = false;
			for (auto node : nodes) {
				switch (node->kind) {
				case Node::CALL: {
					if (node->call.args[0].type()->kind != Type::OPAQUE_FN_TY) {
						continue;
					}

					// args
					std::optional<Extent2D> extent;
					std::optional<Samples> samples;
					std::optional<uint32_t> layer_count;
					for (size_t i = 1; i < node->call.args.size(); i++) {
						auto& arg_ty = node->call.args[0].type()->opaque_fn.args[i - 1];
						auto& parm = node->call.args[i];
						if (arg_ty->kind == Type::IMBUED_TY) {
							auto access = arg_ty->imbued.access;
							auto& link = parm.link();
							auto def = get_def2(parm);
							if (def && def->node->kind == Node::CONSTRUCT) {
								auto& args = def->node->construct.args;
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
										auto e1 = eval<uint32_t>(args[1]);
										auto e2 = eval<uint32_t>(args[2]);
										if (e1.holds_value() && e2.holds_value()) {
											extent = Extent2D{ *e1, *e2 };
										}
									} else if (extent && is_placeholder(args[1]) && is_placeholder(args[2])) {
										placeholder_to_constant(args[1], extent->width);
										placeholder_to_constant(args[2], extent->height);
									}
									if (!layer_count && !is_placeholder(args[7])) { // known layer count
										auto e = eval<uint32_t>(args[7]);
										if (e.holds_value()) {
											layer_count = *e;
										}
									} else if (layer_count && is_placeholder(args[7])) {
										placeholder_to_constant(args[7], *layer_count);
									}
								}
							} else if (def && def->node->kind == Node::ACQUIRE_NEXT_IMAGE) {
								auto e = eval<Swapchain**>(def->node->acquire_next_image.swapchain);
								if (e.holds_value()) {
									Swapchain& swp = ***e;
									extent = Extent2D{ swp.images[0].extent.width, swp.images[0].extent.height };
									layer_count = swp.images[0].layer_count;
									samples = Samples::e1;
								}
							}
						} else {
							assert(0);
						}
					}
					break;
				}
				case Node::CONSTRUCT: {
					auto& args = node->construct.args;
					if (node->type[0]->hash_value == current_module->types.builtin_image) {
						if (constant<ImageAttachment>(args[0]).image.image == VK_NULL_HANDLE) { // if there is no image, we will use base layer 0 and base mip 0
							placeholder_to_constant(args[6], 0U);
							placeholder_to_constant(args[8], 0U);
						}
					}
					break;
				}
				default:
					break;
				}
			}
		} while (progress);

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

	// build required synchronization for nodes
	// at this point we know everything
	Result<void> RGCImpl::build_sync() {
		for (auto node : nodes) {
			switch (node->kind) {
			case Node::CALL: {
				auto fn_type = node->call.args[0].type();
				size_t first_parm = fn_type->kind == Type::OPAQUE_FN_TY ? 1 : 4;
				auto& args = fn_type->kind == Type::OPAQUE_FN_TY ? fn_type->opaque_fn.args : fn_type->shader_fn.args;
				for (size_t i = first_parm; i < node->call.args.size(); i++) {
					auto& arg_ty = args[i - first_parm];
					auto& parm = node->call.args[i];
					auto& link = parm.link();

					if (arg_ty->kind == Type::IMBUED_TY) {
						auto access = arg_ty->imbued.access;
						if (is_write_access(access)) { // Write and ReadWrite
							assert(!link.undef_sync);
							auto dst_access = arg_ty->imbued.access;
							link.undef_sync = to_use(dst_access);
						} else if (!link.read_sync) { // generate Read sync, if we haven't before
							// to avoid R->R deps, we emit a single dep for all the reads
							// for this we compute a merged layout (TRANSFER_SRC_OPTIMAL / READ_ONLY_OPTIMAL / GENERAL)
							ResourceUse dst_use;
							auto reads = link.reads.to_span(pass_reads);
							Type* arg_ty;

							bool need_read_only = false;
							bool need_transfer = false;
							bool need_general = false;

							dst_use.layout = ImageLayout::eReadOnlyOptimalKHR;
							for (int read_idx = 0; read_idx < reads.size(); read_idx++) {
								auto& r = reads[read_idx];
								if (r.node->kind == Node::CALL) {
									if (r.node->call.args[0].type()->kind == Type::OPAQUE_FN_TY) {
										arg_ty = r.node->call.args[0].type()->opaque_fn.args[r.index - first_parm].get(); // TODO: insert casts instead
										parm = r.node->call.args[r.index];
									} else if (r.node->call.args[0].type()->kind == Type::SHADER_FN_TY) {
										arg_ty = r.node->call.args[0].type()->shader_fn.args[r.index - first_parm].get(); // TODO: insert casts instead
										parm = r.node->call.args[r.index];
									} else {
										assert(0);
									}
								} else if (r.node->kind == Node::CONVERGE) {
									continue;
								} else if (r.node->kind == Node::SPLICE) {
									continue;
								} else {
									assert(0);
								}

								assert(arg_ty->kind == Type::IMBUED_TY); // TODO: handle discharged CALLs
								Access dst_access = arg_ty->imbued.access;

								if (is_transfer_access(dst_access)) {
									need_transfer = true;
								}
								if (is_storage_access(dst_access)) {
									need_general = true;
								}
								if (is_readonly_access(dst_access)) {
									need_read_only = true;
								}
								auto use = to_use(dst_access);

								dst_use.access |= use.access;
								dst_use.stages |= use.stages;
							}

							// compute barrier and waits for the merged reads

							if (need_transfer && !need_read_only) {
								dst_use.layout = ImageLayout::eTransferSrcOptimal;
							}

							if (need_general || (need_transfer && need_read_only)) {
								dst_use.layout = ImageLayout::eGeneral;
							}

							link.read_sync = dst_use;
						}
					}
				}
			} break;
			case Node::SPLICE: {
				auto& node_si = *node->scheduled_item;

				for (size_t i = 0; i < node->splice.src.size(); i++) {
					auto& parm = node->splice.src[i];
					auto& link = parm.link();

					if (node->splice.dst_access != Access::eNone) {
						link.undef_sync = to_use(node->splice.dst_access);
					} else {
						if (parm.node->scheduled_item) {
							auto& parm_si = *parm.node->scheduled_item;
							if (parm_si.scheduled_domain != node_si.scheduled_domain) { // parameters are scheduled on different domain
								// we don't know anything about future use, so put "anything"
								parm.link().undef_sync = to_use(Access::eMemoryRW);
							}
						}
					}
				}
			} break;
			default: {
				if (node->scheduled_item) {
					auto& node_si = *node->scheduled_item;

					// SANITY: parameters on the same domain as node
					apply_generic_args([&](Ref parm) { assert(!parm.node->scheduled_item || parm.node->scheduled_item->scheduled_domain == node_si.scheduled_domain); },
					                   node);
				}
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
			case Node::MATH_BINARY:
			case Node::SPLICE:
			case Node::CONVERGE:
				node_to_schedule[node] = schedule_items.size();
				schedule_items.emplace_back(node);
				break;
			default:
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
			ScheduledItem item{ .execable = execable, .scheduled_domain = vuk::DomainFlagBits::eAny };
			if (execable->kind != Node::CONSTRUCT) { // we use def nodes for deps, but we don't want to schedule them later as their ordering doesn't matter
				auto it = scheduled_execables.emplace(item);
				it->execable->scheduled_item = &*it;
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

		for (auto& ind : indegrees) {
			if (ind > 0) {
				std::vector<Node*> unschedulables;
				for (auto i = 0; i < indegrees.size(); i++) {
					if (indegrees[i] > 0) {
						unschedulables.push_back(schedule_items[i]);
					}
				}
				assert(false);
			}
		}

		return { expected_value };
	}

	void Compiler::queue_inference() {
		// queue inference pass
		DomainFlagBits last_domain = DomainFlagBits::eDevice;
		auto propagate_domain = [&last_domain](Node* node) {
			if (!node || !node->scheduled_item) {
				return;
			}
			auto& sched_domain = node->scheduled_item->scheduled_domain;

			// this node has not yet been scheduled
			if (sched_domain == DomainFlagBits::eAny) {
				if ((last_domain != DomainFlagBits::eDevice && last_domain != DomainFlagBits::eAny) &&
				    !node->scheduling_info) { // we have prop info and no scheduling info
					sched_domain = last_domain;
				} else if ((last_domain == DomainFlagBits::eDevice || last_domain == DomainFlagBits::eAny) &&
				           node->scheduling_info) { // we have scheduling info but no prop info
					sched_domain = pick_first_domain(node->scheduling_info->required_domains);
				} else if ((last_domain != DomainFlagBits::eDevice && last_domain != DomainFlagBits::eAny) && node->scheduling_info) { // we have both
					auto intersection = last_domain & node->scheduling_info->required_domains;
					if (intersection.m_mask == 0) { // no intersection, we pick required
						sched_domain = pick_first_domain(node->scheduling_info->required_domains);
					} else { // there was intersection, pick that
						sched_domain = (DomainFlagBits)intersection.m_mask;
					}
				}
			} else { // we have already scheduled this -> propagate
				last_domain = sched_domain;
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
		for (auto& p : impl->scheduled_execables) {
			if (p.scheduled_domain & DomainFlagBits::eTransferQueue) {
				impl->partitioned_execables.push_back(&p);
			}
		}
		impl->transfer_passes = { impl->partitioned_execables.begin(), impl->partitioned_execables.size() };
		for (auto& p : impl->scheduled_execables) {
			if (p.scheduled_domain & DomainFlagBits::eComputeQueue) {
				impl->partitioned_execables.push_back(&p);
			}
		}
		impl->compute_passes = { impl->partitioned_execables.begin() + impl->transfer_passes.size(),
			                       impl->partitioned_execables.size() - impl->transfer_passes.size() };
		for (auto& p : impl->scheduled_execables) {
			if (p.scheduled_domain & DomainFlagBits::eGraphicsQueue) {
				impl->partitioned_execables.push_back(&p);
			}
		}
		impl->graphics_passes = { impl->partitioned_execables.begin() + impl->transfer_passes.size() + impl->compute_passes.size(),
			                        impl->partitioned_execables.size() - impl->transfer_passes.size() - impl->compute_passes.size() };
	}

	Result<void> Compiler::validate_read_undefined() {
		for (auto node : impl->nodes) {
			switch (node->kind) {
			case Node::CONSTRUCT: { // CONSTRUCT discards -
				// TODO: arrays!
				if (node->type[0]->kind != Type::ARRAY_TY && node->links->reads.size() > 0 &&
				    node->type[0]->hash_value != current_module->types.builtin_sampled_image) { // we are trying to read from it :(
					auto reads = node->links->reads.to_span(impl->pass_reads);
					for (auto offender : reads) {
						if (offender.node->kind == Node::SPLICE) { // TODO: not actually a read
							continue;
						}

						auto message0 = format_graph_message(Level::eError, offender.node, "tried to read something that was never written:\n");
						std::string message1;
						if (node->debug_info && node->debug_info->result_names.size() > 0) {
							message1 = fmt::format("	{} was declared/discarded on {}\n", node->debug_info->result_names[0], format_source_location(node));
						} else {
							message1 = fmt::format("	declared/discarded on {}\n", format_source_location(node));
						}
						if (offender.node->kind == Node::CALL) {
							auto fn_type = offender.node->call.args[0].type();
							size_t first_parm = fn_type->kind == Type::OPAQUE_FN_TY ? 1 : 4;
							offender.index -= first_parm;
						}
						auto message2 = fmt::format("	tried to be read as {}th argument", offender.index);
						return { expected_error, RenderGraphException{ message0 + message1 + message2 } };
					}
				} else if (!node->links->undef) {
					// TODO: DCE
					break;
				}
				// in case we have CONSTRUCT -> (SPLICE ->)* READ
				// there is an undef and no read - unravel splices that are never read
				auto undef = node;
				while (undef->links->reads.size() == 0 && undef->links->undef && undef->links->undef.node->kind == Node::SPLICE) {
					undef = undef->links->undef.node;
				}
				// it is either not splice or there are reads
				if (undef->links->reads.size() > 0) {
					auto reads = undef->links->reads.to_span(impl->pass_reads);
					for (auto offender : reads) {
						if (offender.node->kind == Node::SPLICE) {
							continue;
						}
						return { expected_error,
							       RenderGraphException{ format_graph_message(Level::eError, offender.node, "tried to read something that was never written.") } };
					}
				}
				break;
			}
			default:
				break;
			}
		}

		return { expected_value };
	}

	Result<void> Compiler::validate_duplicated_resource_ref() {
		std::unordered_set<Buffer> bufs;
		std::unordered_set<ImageAttachment> ias;
		std::unordered_set<Swapchain*> swps;
		for (auto node : impl->nodes) {
			switch (node->kind) {
			case Node::CONSTRUCT: {
				bool s = true;
				if (node->type[0]->hash_value == current_module->types.builtin_image) {
					auto ia = reinterpret_cast<ImageAttachment*>(node->construct.args[0].node->constant.value);
					if (ia->image) {
						auto [_, succ] = ias.emplace(*ia);
						s = succ;
					}
				} else if (node->type[0]->hash_value == current_module->types.builtin_buffer) {
					auto buf = reinterpret_cast<Buffer*>(node->construct.args[0].node->constant.value);
					if (buf->buffer != VK_NULL_HANDLE) {
						auto [_, succ] = bufs.emplace(*buf);
						s = succ;
					}
				} else if (node->type[0]->hash_value == current_module->types.builtin_swapchain) {
					auto [_, succ] = swps.emplace(reinterpret_cast<Swapchain*>(node->construct.args[0].node->constant.value));
					s = succ;
				} else { // TODO: it is an array, no val yet
				}
				if (!s) {
					return { expected_error, RenderGraphException{ format_graph_message(Level::eError, node, "tried to acquire something that was already known.") } };
				}
			} break;
			case Node::SPLICE: {
				if (!node->splice.rel_acq || node->splice.rel_acq->status == Signal::Status::eDisarmed) {
					break;
				}
				bool s = true;
				assert(node->type.size() == node->splice.values.size());
				for (auto i = 0; i < node->type.size(); i++) {
					// is this ever used?
					auto& link = node->links[i];
					if (!link.undef && link.reads.size() == 0 && !link.next) { // it is never used
						continue;
					}
					if (node->type[i]->hash_value == current_module->types.builtin_image) {
						auto [_, succ] = ias.emplace(*reinterpret_cast<ImageAttachment*>(node->splice.values[i]));
						s = succ;
					} else if (node->type[i]->hash_value == current_module->types.builtin_buffer) {
						auto [_, succ] = bufs.emplace(*reinterpret_cast<Buffer*>(node->splice.values[i]));
						s = succ;
					} else if (node->type[i]->hash_value == current_module->types.builtin_swapchain) {
						auto [_, succ] = swps.emplace(reinterpret_cast<Swapchain*>(node->splice.values[i]));
						s = succ;
					} else { // TODO: it is an array, no val yet
					}
					if (!s) {
						break;
					}
				}
				if (!s) {
					return { expected_error, RenderGraphException{ format_graph_message(Level::eError, node, "tried to acquire something that was already known.") } };
				}
			} break;
			default:
				break;
			}
		}

		return { expected_value };
	}

	Result<void> RGCImpl::implicit_linking(IRModule* module, std::pmr::polymorphic_allocator<std::byte> allocator) {
		std::pmr::vector<Node*> nodes(allocator);

		for (auto& node : module->op_arena) {
			nodes.push_back(&node);
		}

		std::pmr::vector<Ref> pass_reads(allocator);
		std::pmr::vector<Ref> pass_nops(allocator);
		std::pmr::vector<ChainLink*> child_chains(allocator);

		std::sort(nodes.begin(), nodes.end(), [](Node* a, Node* b) { return a->index < b->index; });
		// link with SSA
		build_links(module, nodes.begin(), nodes.end(), pass_reads, pass_nops, child_chains, allocator);
		module->link_frontier = module->node_counter;
		return { expected_value };
	}

	auto format_as(Ref f) {
		return std::string("\"") + fmt::to_string(fmt::ptr(f.node)) + "@" + fmt::to_string(f.index) + std::string("\"");
	}

	struct Replace {
		Ref needle;
		Ref value;
	};

	auto format_as(Replace f) {
		return fmt::to_string(f.needle) + "->" + fmt::to_string(f.value);
	}

	// the issue with multiple replaces is that if there are two replaces link: eg, a->b and b->c
	// in this case the order of replaces / args after replacement will determine the outcome and we might leave b's, despite wanting to get rid of them all
	// to prevent this, we form replace chains when adding replaces
	// if we already b->c:
	//   - and we want to add a->b, then we add a->c and keep b->c (search value in needles)
	//   - and we want to add c->d, then we add c->d and change b->c to b->d (search needle in values)

	// for efficient replacing we can sort both replaces and args with the same sort predicate
	// we loop over replaces and keep a persistent iterator into args, that we increment
	struct Replacer {
		Replacer(std::vector<Replace, short_alloc<Replace>>& v) : replaces(v) {}

		// we keep replaces sorted by needle
		std::vector<Replace, short_alloc<Replace>>& replaces;

		void replace(Ref needle, Ref value) {
			Ref value2 = value;
			// search value in needles -> this will be the end we use
			// 0 or 1 hits
			auto iit =
			    std::lower_bound(replaces.begin(), replaces.end(), Replace{ value, value }, [](const Replace& a, const Replace& b) { return a.needle < b.needle; });
			if (iit != replaces.end() && iit->needle == value) { // 1 hit
				value2 = iit->value;
			}

			// search needle in values (extend chains longer)
			iit = std::find_if(replaces.begin(), replaces.end(), [=](Replace& a) { return a.value == needle; });
			while (iit != replaces.end()) { //
				iit->value = value2;
				// fmt::print("{}", *iit);
				iit = std::find_if(iit, replaces.end(), [=](Replace& a) { return a.value == needle; });
			}

			// sorted insert of new replace
			auto it =
			    std::upper_bound(replaces.begin(), replaces.end(), Replace{ needle, value }, [](const Replace& a, const Replace& b) { return a.needle < b.needle; });
			replaces.insert(it, { needle, value2 });
			// fmt::print("{}\n", Replace{ needle, value2 });
		}
	};

	template<class Pred>
	Result<void> Compiler::rewrite(Pred pred) {
		std::vector<Replace, short_alloc<Replace>> replaces(*impl->arena_);
		Replacer rr(replaces);

		for (auto node : impl->nodes) {
			pred(node, rr);
		}

		/* fmt::print("[");
		    for (auto& r : replaces) {
		      fmt::print("{}, ", r);
		    }
		    fmt::print("]\n");*/

		std::vector<Ref*, short_alloc<Ref*>> args(*impl->arena_);
		// collect all args
		for (auto node : impl->nodes) {
			auto count = node->generic_node.arg_count;
			if (count != (uint8_t)~0u) {
				for (int i = 0; i < count; i++) {
					auto arg = &node->fixed_node.args[i];
					args.push_back(arg);
				}
			} else {
				for (int i = 0; i < node->variable_node.args.size(); i++) {
					auto arg = &(*(Ref**)&node->variable_node.args)[i];
					args.push_back(arg);
				}
			}
		}

		std::sort(args.begin(), args.end(), [](Ref* a, Ref* b) { return a->node < b->node || (a->node == b->node && a->index < b->index); });

		// do the replaces
		auto arg_it = args.begin();
		auto arg_end = args.end();
		for (auto replace_it = replaces.begin(); replace_it != replaces.end(); ++replace_it) {
			auto& replace = *replace_it;
			while (arg_it != arg_end && **arg_it < replace.needle) {
				++arg_it;
			}
			while (arg_it != arg_end && **arg_it == replace.needle) {
				**arg_it = replace.value;
				++arg_it;
			}
		}

		return { expected_value };
	}

	Result<void> Compiler::compile(std::span<std::shared_ptr<ExtNode>> nodes, const RenderGraphCompileOptions& compile_options) {
		reset();
		impl->callbacks = compile_options.callbacks;
		GraphDumper::begin_graph(compile_options.dump_graph, compile_options.graph_label);

		impl->refs.assign(nodes.begin(), nodes.end());
		// tail nodes
		for (auto& r : impl->refs) {
			impl->ref_nodes.emplace_back(r->get_node());
		}

		std::vector<std::shared_ptr<ExtNode>, short_alloc<std::shared_ptr<ExtNode>>> extnode_work_queue(*impl->arena_);
		extnode_work_queue.assign(nodes.begin(), nodes.end());

		std::unordered_set<IRModule*> modules;
		modules.emplace(current_module.get());

		while (!extnode_work_queue.empty()) {
			auto enode = extnode_work_queue.back();
			extnode_work_queue.pop_back();
			extnode_work_queue.insert(extnode_work_queue.end(), std::make_move_iterator(enode->deps.begin()), std::make_move_iterator(enode->deps.end()));
			enode->deps.clear();

			modules.emplace(enode->source_module.get());
			impl->depnodes.push_back(std::move(enode));
		}

		GraphDumper::begin_cluster("fragments");
		std::pmr::polymorphic_allocator<std::byte> allocator(&impl->mbr);

		for (auto& m : modules) {
			// gc the module
			m->collect_garbage(allocator);

			// implicit link the module
			GraphDumper::begin_cluster(std::string("fragments_") + std::to_string(m->module_id));
			GraphDumper::dump_graph_op(m->op_arena, false, false);
			GraphDumper::end_cluster();
			VUK_DO_OR_RETURN(impl->implicit_linking(m, allocator));
			for (auto& op : m->op_arena) {
				op.links = nullptr;
			}
		}
		GraphDumper::next_cluster("fragments", "modules");
		for (auto& m : modules) {
			GraphDumper::begin_cluster(std::string("modules_") + std::to_string(m->module_id));
			GraphDumper::dump_graph_op(m->op_arena, false, false);
			GraphDumper::end_cluster();
		}

		std::sort(impl->depnodes.begin(), impl->depnodes.end());
		impl->depnodes.erase(std::unique(impl->depnodes.begin(), impl->depnodes.end()), impl->depnodes.end());

		VUK_DO_OR_RETURN(impl->build_nodes());

		std::shuffle(impl->nodes.begin(), impl->nodes.end(), _random_generator);
		VUK_DO_OR_RETURN(impl->build_links(impl->nodes, allocator));
		GraphDumper::next_cluster("modules", "full");
		GraphDumper::dump_graph(impl->nodes, false, false);

		// eliminate useless splices & bridge multiple slices
		rewrite([&](Node* node, auto& replaces) {
			switch (node->kind) {
			case Node::SPLICE: {
				// splice elimination
				// a release - must be kept
				if (!(node->splice.dst_access == Access::eNone && node->splice.dst_domain == DomainFlagBits::eAny)) {
					break;
				}

				// an acquire - must be kept
				if (node->splice.rel_acq != nullptr && node->splice.rel_acq->status != Signal::Status::eDisarmed) {
					break;
				}

				if (node->splice.rel_acq != nullptr) {
					node->splice.values = { new void*[node->splice.src.size()], node -> splice.src.size() };
					node->splice.rel_acq->last_use.resize(node->splice.src.size());
				}

				for (size_t i = 0; i < node->splice.src.size(); i++) {
					auto needle = Ref{ node, i };
					auto parm = node->splice.src[i];

					// a splice that requires signalling -> defer it
					if (node->splice.rel_acq != nullptr) {
						node->splice.values[i] = new std::byte[parm.type()->size];

						// find last use that is not splice that we defer away
						auto link = &parm.link();
						while (link->next) {
							link = link->next;
						}
						Node* last_use = nullptr;
						while (link) {
							if (link->reads.size() > 0) { // splices never read
								last_use = link->reads.to_span(impl->pass_reads)[0].node;
								break;
							}
							if (link->def.node->kind == Node::SPLICE && (node->splice.rel_acq == nullptr || node->splice.rel_acq->status == Signal::Status::eDisarmed)) {
								;
							} else {
								last_use = link->def.node;
								break;
							}
							link = link->prev;
						}
						assert(last_use);
						impl->deferred_splices[last_use].push_back(needle);
						impl->pending_splice_sigs[needle.node] = 0;
					}
				}
			} break;
			case Node::SLICE: {
				auto& slice = node->slice;
				Subrange::Image our_slice_range = { constant<uint32_t>(slice.base_level),
					                                  constant<uint32_t>(slice.level_count),
					                                  constant<uint32_t>(slice.base_layer),
					                                  constant<uint32_t>(slice.layer_count) };
				// walk up
				auto link = &node->slice.image.link();
				do {
					if (link->def.node->kind == Node::SLICE) { // it is a slice
						Subrange::Image their_slice_range = { constant<uint32_t>(link->def.node->slice.base_level),
							                                    constant<uint32_t>(link->def.node->slice.level_count),
							                                    constant<uint32_t>(link->def.node->slice.base_layer),
							                                    constant<uint32_t>(link->def.node->slice.layer_count) };
						if (link->def.index == 0) { //  and we took left
							auto isect = intersect_one(our_slice_range, their_slice_range);
							if (isect == our_slice_range) {
								replaces.replace(first(node), node->slice.image);
								replaces.replace(nth(node, 1), node->slice.image);
								break;
							}
						} else { //  and we took right
							auto isect = intersect_one(our_slice_range, their_slice_range);
							if (isect == our_slice_range) {
								replaces.replace(first(node), node->slice.image);
								replaces.replace(nth(node, 1), node->slice.image);
								break;
							}
						}
					}
					if (!link->prev) {
						break;
					}
					link = link->prev;
				} while (link->prev);
				if (link->def.node->kind == Node::SLICE) { // it is a slice
					Subrange::Image their_slice_range = { constant<uint32_t>(link->def.node->slice.base_level),
						                                    constant<uint32_t>(link->def.node->slice.level_count),
						                                    constant<uint32_t>(link->def.node->slice.base_layer),
						                                    constant<uint32_t>(link->def.node->slice.layer_count) };
					if (link->def.index == 0) { //  and we took left
						auto isect = intersect_one(our_slice_range, their_slice_range);
						if (isect == our_slice_range) {
							replaces.replace(first(node), node->slice.image);
							replaces.replace(nth(node, 1), node->slice.image);
							break;
						}
					} else { //  and we took right
						auto isect = intersect_one(our_slice_range, their_slice_range);
						if (isect == our_slice_range) {
							replaces.replace(first(node), node->slice.image);
							replaces.replace(nth(node, 1), node->slice.image);
							break;
						}
					}
				}
			} break;
			default:
				break;
			}
		});

		VUK_DO_OR_RETURN(impl->build_nodes());
		// post replace
		//_dump_graph(impl->nodes, false, false);
		VUK_DO_OR_RETURN(impl->build_links(impl->nodes, allocator));

		// FINAL GRAPH
		GraphDumper::next_cluster("final");
		GraphDumper::dump_graph(impl->nodes, false, true);
		GraphDumper::end_cluster();
		GraphDumper::end_graph();
		//_dump_graph(impl->nodes, false, false);

		VUK_DO_OR_RETURN(validate_read_undefined());
		VUK_DO_OR_RETURN(validate_duplicated_resource_ref());

		VUK_DO_OR_RETURN(impl->collect_chains());
		VUK_DO_OR_RETURN(impl->reify_inference());

		VUK_DO_OR_RETURN(impl->schedule_intra_queue(compile_options));

		queue_inference();
		pass_partitioning();

		VUK_DO_OR_RETURN(impl->build_sync());

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

	ImageUsageFlags RGCImpl::compute_usage(const ChainLink* head) {
		ImageUsageFlags usage = {};

		for (auto chain = head; chain != nullptr; chain = chain->next) {
			for (auto& r : chain->reads.to_span(pass_reads)) {
				switch (r.node->kind) {
				case Node::CALL: {
					auto fn_type = r.node->call.args[0].type();
					size_t first_parm = fn_type->kind == Type::OPAQUE_FN_TY ? 1 : 4;
					auto& args = fn_type->kind == Type::OPAQUE_FN_TY ? fn_type->opaque_fn.args : fn_type->shader_fn.args;

					auto& arg_ty = args[r.index - first_parm];
					if (arg_ty->kind == Type::IMBUED_TY) {
						auto access = arg_ty->imbued.access;
						access_to_usage(usage, access);
					}
					break;
				}
				default:
					break;
				}
			}
			if (chain->undef) {
				switch (chain->undef.node->kind) {
				case Node::CALL: {
					auto fn_type = chain->undef.node->call.args[0].type();
					size_t first_parm = fn_type->kind == Type::OPAQUE_FN_TY ? 1 : 4;
					auto& args = fn_type->kind == Type::OPAQUE_FN_TY ? fn_type->opaque_fn.args : fn_type->shader_fn.args;

					auto& arg_ty = args[chain->undef.index - first_parm];
					if (arg_ty->kind == Type::IMBUED_TY) {
						auto access = arg_ty->imbued.access;
						access_to_usage(usage, access);
					}
					break;
				}
				default:
					break;
				}
			}

			for (auto& child_chain : chain->child_chains.to_span(child_chains)) {
				usage |= compute_usage(child_chain);
			}
		}

		return usage;
	}
} // namespace vuk
