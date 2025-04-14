#include "GraphDumper.hpp"
#include "vuk/Exception.hpp"
#include "vuk/IRProcess.hpp"
#include "vuk/RenderGraph.hpp"
#include "vuk/SyncLowering.hpp"
#include "vuk/runtime/CommandBuffer.hpp"

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
	auto format_as(Ref f) {
		return std::string("\"") + fmt::to_string(fmt::ptr(f.node)) + "@" + fmt::to_string(f.index) + std::string("\"");
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

	void IRModule::collect_garbage() {
		std::pmr::monotonic_buffer_resource mbr;
		std::pmr::polymorphic_allocator<std::byte> allocator(&mbr);
		collect_garbage(allocator);
	}

	void IRModule::collect_garbage(std::pmr::polymorphic_allocator<std::byte> allocator) {
		enum { DEAD = 1, ALIVE = 2, ALIVE_REC = 3 };

		// initial set of live nodes
		for (auto it = op_arena.begin(); it != op_arena.end();) {
			auto node = &*it;
			node->flag = DEAD;
			// if the node is garbage, just collect it now
			if (node->kind == Node::GARBAGE) {
				it = op_arena.erase(it);
				continue;
			}
			// nodes which have been linked before and are no longer held can be dropped from the initial set
			if (node->index < (module_id << 32 | link_frontier) && !node->held) {
				++it;
				continue;
			}
			// everything else is in the initial set
			node->flag = ALIVE;
			++it;
		}

		int outer_loops = 0;
		int inner_loops = 0;
		int steps = 0;
		// compute live set
		bool change = false;
		do {
			outer_loops++;
			change = false;
			for (auto it = op_arena.begin(), end = op_arena.end(); it != end; ++it) {
				inner_loops++;
				auto orig_node = &*it;
				if (orig_node->flag != ALIVE) {
					continue;
				}
				while (orig_node->flag != ALIVE_REC) {
					auto node = orig_node;
					// if node is ALIVE then we make all children ALIVE
					while (node->flag == ALIVE) {
						bool step = false;
						auto count = node->generic_node.arg_count;
						if (count != (uint8_t)~0u) {
							for (int i = 0; i < count; i++) {
								auto snode = node->fixed_node.args[i].node;
								if (snode->flag == DEAD) { // turn it ALIVE and start from there
									node = snode;
									node->flag = ALIVE;
									step = true;
									change = true;
									steps++;
									break;
								}
							}
							if (step) {
								continue;
							}
						} else {
							for (int i = 0; i < node->variable_node.args.size(); i++) {
								auto snode = node->variable_node.args[i].node;
								if (snode->flag == DEAD) { // turn it ALIVE and start from there
									node = snode;
									node->flag = ALIVE;
									step = true;
									change = true;
									steps++;
									break;
								}
							}
							if (step) {
								continue;
							}
						}
						// we got here so all children must be ALIVE or ALIVE_REC
						node->flag = ALIVE_REC;
					}
				}
			}
		} while (change);

		// GC the module
		for (auto it = op_arena.begin(); it != op_arena.end();) {
			auto node = &*it;
			if (node->flag == DEAD) {
				it = *destroy_node(node);
			} else {
				++it;
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
		auto pool = std::move(impl->pool);
		auto pass_r = std::move(impl->pass_reads);
		auto arena = impl->arena_.release();
		delete impl;
		arena->reset();
		impl = new RGCImpl(arena, std::move(pool));
		impl->pass_reads = std::move(pass_r);
		impl->pass_reads.clear();
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

	// #define VUK_DUMP_SSA

	struct NodeContext {
		IRModule* module;
		std::pmr::vector<Ref>& pass_reads;
		std::pmr::vector<ChainLink*>& child_chains;
		std::pmr::vector<Node*>& new_nodes;
		std::pmr::polymorphic_allocator<std::byte> allocator;
		std::vector<std::pair<Buffer, ChainLink*>> bufs;
		bool do_ssa;

		std::vector<std::string> debug_stack;

		void print_ctx() {
			if (debug_stack.empty()) {
				return;
			}
			fmt::print("[");
			for (auto& s : debug_stack) {
				fmt::print("{} ", s);
			}
			fmt::print("]: ");
		}

		Ref walk_writes(Node* node, Ref parm) {
			auto link = &parm.link();
			Ref last_write;

			do {
				if (link->next) {
					link = link->next;
				}
			} while (link->next);

			if (link->undef) {
				print_ctx();
				// this was consumed by a slice S
				// if we are a slice S' ourselves, we might get to elide a convergence if
				// 1. the cut introduced by slice S' is contained in the cut introduced by slice S (shrinking)
				// 2. the cut introduced by slice S' is equal to the cut introduced by slice S (identity)
				// 3. the cut introduced by slice S' is contained by S^ \ S (shrinking the remainder)
				if (link->undef.node->kind == Node::SLICE) {
					auto& slice_node = link->undef.node;
					auto type_kind = nth(slice_node, 2).type()->kind;
					bool forbid_elision = type_kind == Type::UNION_TY;
					if (node->kind == Node::SLICE && !forbid_elision) {
						auto scope_S = Cut{ slice_node->slice.axis, *eval<uint64_t>(slice_node->slice.start), *eval<uint64_t>(slice_node->slice.count) };
						auto scope_Sp = Cut{ node->slice.axis, *eval<uint64_t>(node->slice.start), *eval<uint64_t>(node->slice.count) };

						if (scope_Sp.shrinks(scope_S)) { // cases 1 and 2, we can elide the convergence
#ifdef VUK_DUMP_SSA
							fmt::println("shrinking or identity - eliding convergence");
#endif
							auto new_start = scope_Sp.range.offset - scope_S.range.offset;
							if (new_start == 0 && scope_Sp.range.count == 1 && node->slice.axis == Node::NamedAxis::FIELD) {
								auto src = node->slice.src;
								node->kind = Node::LOGICAL_COPY;
								node->logical_copy = {};
								node->logical_copy.src = src;
								node->type = std::span(node->type.data(), 1);
								return walk_writes(node, nth(slice_node, 0));
							} else {
								node->slice.start = current_module->make_constant<uint64_t>(new_start);
								node->slice.count = current_module->make_constant<uint64_t>(scope_Sp.range.count);
								return walk_writes(node, nth(slice_node, 0));
							}
						} else if (!scope_Sp.intersects(scope_S)) { // case 3, we can elide the convergence
#ifdef VUK_DUMP_SSA
							fmt::println("remainder - eliding convergence");
#endif
							return walk_writes(node, nth(slice_node, 1));
						}
					}
#ifdef VUK_DUMP_SSA
					fmt::println("slice - emitting convergence");
#endif
					std::array tails{ nth(slice_node, 2), nth(slice_node, 0), nth(slice_node, 1) };
					last_write = module->make_converge(slice_node->slice.src.type(), tails);
					last_write.node->index = node->index;
					allocate_node_links(last_write.node, allocator);
					process_node_links(last_write.node);
					new_nodes.push_back(last_write.node);
					// this was consumed by a converge - this means we need to replicate the slice
				} else if (link->undef.node->kind == Node::CONVERGE) {
#ifdef VUK_DUMP_SSA
					fmt::println("convergence - replicating slice");
#endif
					last_write = module->make_slice(parm.node->type[0], first(link->undef.node), parm.node->slice.axis, parm.node->slice.start, parm.node->slice.count);
					last_write.node->index = node->index;
					allocate_node_links(last_write.node, allocator);
					process_node_links(last_write.node);
					new_nodes.push_back(last_write.node);
				} else if (link->undef.node->kind == Node::CONSTRUCT && first(link->undef.node).type()->kind == Type::UNION_TY) {
#ifdef VUK_DUMP_SSA
					fmt::println("construct - replicating extract");
#endif
					last_write = module->make_extract(first(link->undef.node), link->undef.index - 1);
					last_write.node->index = node->index;
					allocate_node_links(last_write.node->slice.start.node, allocator);
					process_node_links(last_write.node->slice.start.node);
					allocate_node_links(last_write.node->slice.count.node, allocator);
					process_node_links(last_write.node->slice.count.node);
					allocate_node_links(last_write.node, allocator);
					process_node_links(last_write.node);
					new_nodes.push_back(last_write.node);
				} else {
					VUK_ICE(false);
				}
			} else {
				last_write = link->def;
			}

			return last_write;
		};

		void add_write(Node* node, Ref& parm, size_t index) {
			VUK_ICE(parm.node->kind != Node::GARBAGE);
			if (!parm.node->links) {
				VUK_ICE(do_ssa);
				// external node -> init
				allocate_node_links(parm.node, allocator);
				for (size_t i = 0; i < parm.node->type.size(); i++) {
					Ref{ parm.node, 0 }.link().def = { parm.node, i };
				}
			}
			auto link = &parm.link();
			if (link->undef.node == node) {
				return; // we are already writing this
			}
			if (link->undef.node != nullptr) { // there is already a write -> do SSA rewrite
#ifdef VUK_DUMP_SSA
				print_ctx();
				fmt::println("have to SSA rewrite param({}@{}), at input index {}", Node::kind_to_sv(parm.node->kind), parm.index, index);
#endif
				VUK_ICE(do_ssa);
				auto old_ref = link->undef;                  // this is an rref
				VUK_ICE(node->index >= old_ref.node->index); // we are after the existing write
				// attempt to find the final revision of this
				// this could be either the last write on the main chain, or the last write on a child chain
				auto last_write = walk_writes(node, parm);
				parm = last_write;
				link = &parm.link();
			}
			link->undef = { node, index };
		};

		void add_breaking_result(Node* node, size_t output_idx) {
			Ref out = Ref{ node, output_idx };
			out.link().def = { node, output_idx };
		};

		void add_result(Node* node, size_t output_idx, Ref parm) {
			if (!node->links) {
				VUK_ICE(do_ssa);
				// external node -> init
				allocate_node_links(node, allocator);
			}
			Ref out = Ref{ node, output_idx };
			out.link().def = { node, output_idx };

			if (!parm.node->links) {
				VUK_ICE(do_ssa);
				return;
			}

			auto link = &parm.link();
			auto& prev = out.link().prev;
			if (!do_ssa) {
				VUK_ICE(!link->next);
				VUK_ICE(!prev);
			}
			link->next = &out.link();
			prev = link;
		};

		void add_read(Node* node, Ref& parm, size_t index) {
			VUK_ICE(parm.node->kind != Node::GARBAGE);
			auto& st_parm = parm;
			if (!st_parm.node->links) {
				VUK_ICE(do_ssa);
				// external node -> init
				allocate_node_links(st_parm.node, allocator);
				for (size_t i = 0; i < st_parm.node->type.size(); i++) {
					Ref{ st_parm.node, 0 }.link().def = { st_parm.node, i };
				}
			}
			auto link = &st_parm.link();
			if (link->undef.node != nullptr && node->index > link->undef.node->index) { // there is already a write and it is earlier than us
				VUK_ICE(do_ssa);
				auto last_write = walk_writes(node, parm);
				parm = last_write;
				link = &parm.link();
			}
			link->reads.append(pass_reads, { node, index });
		};

		void process_node_links(Node* node) {
#ifdef VUK_DUMP_SSA
			debug_stack.push_back(std::string(node->kind_to_sv(node->kind)));
			print_ctx();
			fmt::println("entering");
#endif
			switch (node->kind) {
			case Node::CONSTANT:
			case Node::PLACEHOLDER:
				add_breaking_result(node, 0);
				break;
			case Node::CONSTRUCT:
				first(node).link().def = first(node);

				for (size_t i = 0; i < node->construct.args.size(); i++) {
					auto& parm = node->construct.args[i];
					if (node->type[0]->kind == Type::ARRAY_TY || node->type[0]->kind == Type::UNION_TY) {
						add_write(node, parm, i);
					} else {
						add_read(node, parm, i);
					}
				}

				if (node->type[0]->kind == Type::ARRAY_TY || node->type[0]->hash_value == current_module->types.builtin_sampled_image) {
					for (size_t i = 1; i < node->construct.args.size(); i++) {
						auto& parm = node->construct.args[i];
						auto& st_parm = parm;
						st_parm.link().next = &first(node).link();
					}
				}

				break;
			case Node::MATH_BINARY:
				add_read(node, node->math_binary.a, 0);
				add_read(node, node->math_binary.b, 1);
				add_breaking_result(node, 0);
				break;
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
							} else if (def && (def->node->kind == Node::ACQUIRE_NEXT_IMAGE || def->node->kind == Node::ACQUIRE || def->node->kind == Node::CONSTANT)) { // nop
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
			case Node::RELEASE:
				for (size_t i = 0; i < node->release.src.size(); i++) {
					add_write(node, node->release.src[i], i);
					add_result(node, i, node->release.src[i]);
				}
				break;
			case Node::ACQUIRE:
				for (size_t out = 0; out < node->type.size(); out++) {
					add_breaking_result(node, out);
					if (do_ssa && node->type[out]->hash_value == current_module->types.builtin_buffer) {
						auto& buf = *reinterpret_cast<Buffer*>(node->acquire.values[out]);
						assert(buf.buffer != VK_NULL_HANDLE);
						bool found = false;
						for (auto& [existing_buf, link] : bufs) {
							if (buf.buffer == existing_buf.buffer && Range{ buf.offset, buf.size }.intersect({ existing_buf.offset, existing_buf.size })) {
								// we will need to union the buffers

								std::array args = { Ref{ node, out }, link->def };
								auto con_union = module->make_declare_union(args);
								con_union.node->index = node->index;
								allocate_node_links(con_union.node, allocator);
								process_node_links(con_union.node);
								new_nodes.push_back(con_union.node);
								found = true;
								break;
							}
						}
						if (!found) {
							bufs.emplace_back(buf, &nth(node, out).link());
						}
					}
				}
				break;

			case Node::SLICE: {
				add_read(node, node->slice.start, 1);
				add_read(node, node->slice.count, 2);
				add_write(node, node->slice.src, 0);
				if (node->kind == Node::LOGICAL_COPY){ // if we rewrote this to a copy
					add_result(node, 0, node->slice.src);
					break;
				}
				nth(node, 0).link().def = nth(node, 0); // we introduce the slice image def
				nth(node, 1).link().def = nth(node, 1); // we introduce the rest image def
				add_breaking_result(node, 2);
				if (node->slice.src.node->links) {
					node->slice.src.link().child_chains.append(child_chains, &nth(node, 0).link()); // add child chain for sliced
				} else {
					assert(do_ssa);
				}

				break;
			}
			case Node::CONVERGE:
				if (node->converge.diverged[0].node->kind == Node::SLICE) {
					add_result(node->converge.diverged[0].node, 2, node->converge.diverged[0].node->slice.src);
				}
				add_result(node, 0, node->converge.diverged[0]);
				for (size_t i = 0; i < node->converge.diverged.size(); i++) {
					auto& parm = node->converge.diverged[i];
					add_write(node, parm, i);
				}
				break;

			case Node::ACQUIRE_NEXT_IMAGE:
				add_breaking_result(node, 0);
				break;

			case Node::GARBAGE:
				break;

			case Node::USE:
				add_result(node, 0, node->use.src);
				add_write(node, node->use.src, 0);
				break;

			case Node::LOGICAL_COPY:
				add_result(node, 0, node->logical_copy.src);
				add_read(node, node->logical_copy.src, 0);
				break;

			default:
				assert(0);
			}
#ifdef VUK_DUMP_SSA
			print_ctx();
			fmt::println("exiting");
			debug_stack.pop_back();
#endif
		}
	};

	Result<void> RGCImpl::build_links(std::vector<Node*>& working_set, std::pmr::polymorphic_allocator<std::byte> allocator) {
		pass_reads.clear();
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
		NodeContext nc{ current_module.get(), pass_reads, child_chains, new_nodes, allocator, bufs, false };

		for (auto& node : working_set) {
			nc.process_node_links(node);
		}

		working_set.insert(working_set.end(), new_nodes.begin(), new_nodes.end());

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
	                                  std::pmr::vector<ChainLink*>& child_chains,
	                                  std::pmr::polymorphic_allocator<std::byte> allocator) {
		std::pmr::vector<Node*> new_nodes(allocator);
		for (auto it = start; it != end; ++it) {
			allocate_node_links(*it, allocator);
		}
		NodeContext nc{ current_module.get(), pass_reads, child_chains, new_nodes, allocator, bufs, true };

		for (auto it = start; it != end; ++it) {
			nc.process_node_links(*it);
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
							if (is_framebuffer_attachment(access)) {
								auto& link = parm.link();
								auto def = get_def2(parm);
								if (def && def->node->kind == Node::CONSTRUCT) {
									auto& args = def->node->construct.args;
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
								} else if (def && def->node->kind == Node::ACQUIRE_NEXT_IMAGE) {
									auto e = eval<Swapchain**>(def->node->acquire_next_image.swapchain);
									if (e.holds_value()) {
										Swapchain& swp = ***e;
										extent = Extent2D{ swp.images[0].extent.width, swp.images[0].extent.height };
										layer_count = swp.images[0].layer_count;
										samples = Samples::e1;
									}
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
		live_ranges.clear();
		// collect chains by looking at links without a prev
		for (auto& node : nodes) {
			size_t result_count = node->type.size();
			for (auto i = 0; i < result_count; i++) {
				auto link = &node->links[i];
				if (!link->prev) { // head, add to chains
					chains.push_back(link);
					LiveRange& lr = live_ranges[link];
					lr.def_link = link;
					while (link->next) { // tail
						link = link->next;
					}
					lr.undef_link = link;
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
								} else if (r.node->kind == Node::CONVERGE || r.node->kind == Node::CONSTRUCT) {
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
			case Node::RELEASE: {
				assert(node->scheduled_item);
				auto& node_si = *node->scheduled_item;
				for (size_t i = 0; i < node->release.src.size(); i++) {
					auto& parm = node->release.src[i];
					auto& link = parm.link();
					assert(!link.undef_sync);
					if (node->release.dst_access != Access::eNone) {
						link.undef_sync = to_use(node->release.dst_access);
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
			case Node::USE: {
				auto& parm = node->use.src;
				auto type = parm.type()->hash_value;
				if (parm.type()->kind == Type::ARRAY_TY) {
					type = (*parm.type()->array.T)->hash_value;
				}
				if (type != current_module->types.builtin_buffer && type != current_module->types.builtin_image) {
					break;
				}
				auto& link = parm.link();
				assert(!link.undef_sync);
				if (node->use.access != Access::eNone) {
					link.undef_sync = to_use(node->use.access);
				} else {
					assert(node->use.src.node->kind == Node::CONVERGE);
					auto& converge = node->use.src.node->converge;
					// find something with sync and broadcast that onto the convergence
					// it is possible we find nothing - in this case no sync is needed
					for (size_t i = 1; i < converge.diverged.size(); i++) {
						ChainLink* use_link = &converge.diverged[i].link();
						while (!use_link->read_sync && !use_link->undef_sync && use_link->prev) {
							use_link = use_link->prev;
						}
						if (!use_link->read_sync && !use_link->undef_sync) {
							continue;
						}
						link.undef_sync = use_link->undef_sync ? use_link->undef_sync : use_link->read_sync;
						break;
					}
				}
			} break;
			default:
				break;
			}
		}

		return { expected_value };
	}

	DomainFlagBits pick_first_domain(DomainFlags f) { // TODO: make this work
		return (DomainFlagBits)f.m_mask;
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
				break;
			}
			default:
				break;
			}
		}

		return { expected_value };
	}

	Result<void> Compiler::validate_duplicated_resource_ref() {
		std::unordered_map<Buffer, Node*> bufs;
		std::unordered_map<ImageAttachment, Node*> ias;
		std::unordered_map<Swapchain*, Node*> swps;
		auto add_one = [&](Type* type, Node* node, void* value) -> std::optional<Node*> {
			if (type->kind == Type::ARRAY_TY || type->kind == Type::UNION_TY) {
				return {};
			}
			if (type->hash_value == current_module->types.builtin_image) {
				auto ia = reinterpret_cast<ImageAttachment*>(value);
				if (ia->image) {
					auto [_, succ] = ias.emplace(*ia, node);
					if (!succ) {
						return ias.at(*ia);
					}
				}
			} else if (type->hash_value == current_module->types.builtin_buffer) {
				auto buf = reinterpret_cast<Buffer*>(value);
				if (buf->buffer != VK_NULL_HANDLE) {
					auto [_, succ] = bufs.emplace(*buf, node);
					if (!succ) {
						return bufs.at(*buf);
					}
				}
			} else if (type->hash_value == current_module->types.builtin_swapchain) {
				auto swp = reinterpret_cast<Swapchain*>(value);
				auto [_, succ] = swps.emplace();
				if (!succ) {
					return swps.at(swp);
				}
			} else { // TODO: it is an array, no val yet
			}

			return {};
		};
		for (auto node : impl->nodes) {
			std::optional<Node*> fail = {};
			switch (node->kind) {
			case Node::CONSTRUCT: {
				fail = add_one(node->type[0].get(), node, node->construct.args[0].node->constant.value);
			} break;
			case Node::ACQUIRE: {
				for (size_t i = 0; i < node->type.size(); i++) {
					auto as_ref = nth(node, i);
					auto& link = as_ref.link();
					if (link.reads.size() == 0 && !link.undef && !link.next) { // if not used, we don't care about it
						continue;
					}
					fail = add_one(node->type[i].get(), node, node->acquire.values[i]);
					if (fail && node->type[i].get()->hash_value == current_module->types.builtin_buffer &&
					    fail.value()->kind == Node::ACQUIRE) { // an acq-acq for buffers, this is allowed
						fail = {};
					}
				}
			} break;
			default:
				break;
			}
			if (fail) {
				auto loc = format_source_location(*fail);
				auto msg = fmt::format("tried to acquire something that was already known. Previously acquired by {} with callstack:\n{}", node_to_string(*fail), loc);
				return { expected_error, RenderGraphException{ format_graph_message(Level::eError, node, msg) } };
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
		std::pmr::vector<ChainLink*> child_chains(allocator);

		std::sort(nodes.begin(), nodes.end(), [](Node* a, Node* b) { return a->index < b->index; });
		// link with SSA
		build_links(module, nodes.begin(), nodes.end(), pass_reads, child_chains, allocator);
		module->link_frontier = module->node_counter;
		return { expected_value };
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

	Result<void> Compiler::linearize() {
		impl->naming_index_counter = 0;
		impl->scheduled.clear();
		impl->item_list.clear();

		// these are the items that were determined to run
		for (auto& i : impl->scheduled_execables) {
			impl->work_queue.emplace_back(i.execable, false);
		}

		while (!impl->work_queue.empty()) {
			auto item = impl->work_queue.front();
			impl->work_queue.pop_front();
			auto& node = item.node;
			if (impl->scheduled.contains(node)) { // only going schedule things once
				continue;
			}

			if (impl->work_queue.size() > (10 * impl->nodes.size())) {
				return { expected_error, RenderGraphException{ "Too many iterations in linearization, something is wrong" } };
			}

			// we run nodes twice - first time we reenqueue at the front and then put all deps before it
			// second time we see it, we know that all deps have run, so we can run the node itself
			if (impl->process(item)) {
				impl->scheduled.emplace(node);
				node->scheduled_item->naming_index = impl->naming_index_counter;
				impl->item_list.push_back(node->scheduled_item);
				impl->naming_index_counter += node->type.size();
			} else {
				switch (node->kind) {
				case Node::MATH_BINARY: {
					for (auto i = 0; i < node->fixed_node.arg_count; i++) {
						impl->schedule_dependency(node->fixed_node.args[i], RW::eRead);
					}
				} break;
				case Node::CONSTRUCT: {
					for (auto& parm : node->construct.args.subspan(1)) {
						impl->schedule_dependency(parm, RW::eRead);
					}

				} break;
				case Node::CALL: {
					auto fn_type = node->call.args[0].type();
					size_t first_parm = fn_type->kind == Type::OPAQUE_FN_TY ? 1 : 4;
					auto& args = fn_type->kind == Type::OPAQUE_FN_TY ? fn_type->opaque_fn.args : fn_type->shader_fn.args;

					for (size_t i = first_parm; i < node->call.args.size(); i++) {
						auto& arg_ty = args[i - first_parm];
						auto& parm = node->call.args[i];

						if (arg_ty->kind == Type::IMBUED_TY) {
							auto access = arg_ty->imbued.access;
							// Write and ReadWrite
							RW sync_access = (is_write_access(access)) ? RW::eWrite : RW::eRead;
							impl->schedule_dependency(parm, sync_access);
						} else {
							assert(0);
						}
					}
				} break;
				case Node::RELEASE: {
					auto acqrel = node->rel_acq;
					if (!acqrel || acqrel->status == Signal::Status::eDisarmed) {
						for (size_t i = 0; i < node->release.src.size(); i++) {
							impl->schedule_dependency(node->release.src[i], RW::eWrite);
						}
					}
				} break;
				case Node::ACQUIRE: {
					// ACQUIRE does not have any deps
				} break;
				case Node::ACQUIRE_NEXT_IMAGE: {
					impl->schedule_dependency(node->acquire_next_image.swapchain, RW::eWrite);
				} break;
				case Node::SLICE: {
					impl->schedule_dependency(node->slice.src, RW::eWrite);
					impl->schedule_dependency(node->slice.start, RW::eRead);
					impl->schedule_dependency(node->slice.count, RW::eRead);
				} break;
				case Node::CONVERGE: {
					for (size_t i = 0; i < node->converge.diverged.size(); i++) {
						impl->schedule_dependency(node->converge.diverged[i], RW::eWrite);
					}
				} break;
				case Node::USE: {
					impl->schedule_dependency(node->use.src, RW::eWrite);
				} break;
				case Node::LOGICAL_COPY: {
					impl->schedule_dependency(node->logical_copy.src, RW::eRead);
				} break;
				default:
					VUK_ICE(false);
					break;
				}
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
		for (auto& m : modules) {
			for (auto& op : m->op_arena) {
				op.flag = 0;
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

		VUK_DO_OR_RETURN(impl->build_nodes());
		// post replace
		VUK_DO_OR_RETURN(impl->build_links(impl->nodes, allocator));

		VUK_DO_OR_RETURN(validate_read_undefined());
		VUK_DO_OR_RETURN(validate_duplicated_resource_ref());

		VUK_DO_OR_RETURN(impl->collect_chains());

		// do forced convergence here
		std::pmr::vector<Node*> new_nodes;
		NodeContext nc{ current_module.get(), impl->pass_reads, impl->child_chains, new_nodes, allocator, impl->bufs, true };
		for (auto& [def, lr] : impl->live_ranges) {
			if (lr.def_link->def.node->kind == Node::SLICE) { // subchains - not important
				continue;
			}
			while (lr.undef_link->next) {
				lr.undef_link = lr.undef_link->next;
			}
			if (lr.undef_link->undef && lr.undef_link->undef.node->kind == Node::SLICE &&
			    nth(lr.undef_link->undef.node, 2).type()->kind != Type::UNION_TY) { // main chain that ends in SLICE..
				// make force reconvergence node
				auto slice_node = lr.undef_link->undef.node;
				std::array tails{ nth(slice_node, 2), nth(slice_node, 0), nth(slice_node, 1) };
				auto f_converge = nc.module->make_converge(slice_node->slice.src.type(), tails);
				allocate_node_links(f_converge.node, allocator);
				nc.process_node_links(f_converge.node);
				new_nodes.push_back(f_converge.node);
				// add use node
				auto use_node = nc.module->make_use(f_converge, Access::eNone);
				allocate_node_links(use_node.node, allocator);
				nc.process_node_links(use_node.node);
				new_nodes.push_back(use_node.node);
				// make the ref node depend on it
				assert(impl->ref_nodes.back()->kind == Node::RELEASE);
				auto& release_node = impl->ref_nodes.back();
				std::array wrapping{ release_node->release.src[0], use_node };
				release_node->release.src[0].link().undef = {};
				release_node->release.src[0].link().next = {};
				release_node->release.src[0] = nc.module->make_converge(release_node->release.src[0].type(), wrapping);
				release_node->release.src[0].node->index = release_node->index;
				allocate_node_links(release_node->release.src[0].node, allocator);
				nc.process_node_links(release_node->release.src[0].node);
				new_nodes.push_back(release_node->release.src[0].node);
				allocate_node_links(release_node, allocator);
				nc.process_node_links(release_node);
			}
		}
		impl->nodes.insert(impl->nodes.end(), new_nodes.begin(), new_nodes.end());
		new_nodes.clear();

		VUK_DO_OR_RETURN(impl->collect_chains());
		VUK_DO_OR_RETURN(impl->reify_inference());

		for (auto& node : impl->ref_nodes) {
			ScheduledItem item{ .execable = node, .scheduled_domain = vuk::DomainFlagBits::eAny };
			auto it = impl->scheduled_execables.emplace(item);
			it->execable->scheduled_item = &*it;
		}

		for (auto& node : impl->nodes) {
			switch (node->kind) {
			case Node::SLICE:
			case Node::CALL: {
				ScheduledItem item{ .execable = node, .scheduled_domain = vuk::DomainFlagBits::eAny };
				auto it = impl->scheduled_execables.emplace(item);
				it->execable->scheduled_item = &*it;
			} break;

			default:
				break;
			}
		}

		queue_inference();
		pass_partitioning();

		VUK_DO_OR_RETURN(impl->build_sync());

		// FINAL GRAPH
		GraphDumper::next_cluster("final");
		GraphDumper::dump_graph(impl->nodes, false, false);
		GraphDumper::end_cluster();
		GraphDumper::end_graph();

		VUK_DO_OR_RETURN(linearize());

		return { expected_value };
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

	std::span<struct ScheduledItem*> Compiler::get_scheduled_nodes() const {
		return std::span(impl->item_list);
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
