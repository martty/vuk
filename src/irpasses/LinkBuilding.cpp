#include "vuk/ir/GraphDumper.hpp"
#include "vuk/ir/IRPasses.hpp"
#include "vuk/SyncLowering.hpp"
#include <array>

// #define VUK_DUMP_SSA

namespace vuk {
	void IRPass::print_ctx() {
		if (debug_stack.empty()) {
			return;
		}
		fmt::print("[");
		for (auto& s : debug_stack) {
			fmt::print("{} ", s);
		}
		fmt::print("]: ");
	}

	Ref IRPass::walk_writes(Node* node, Ref parm) {
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
					auto scope_S = Cut{ slice_node->slice.axis, constant<uint64_t>(slice_node->slice.start), constant<uint64_t>(slice_node->slice.count) };
					auto scope_Sp = Cut{ node->slice.axis, constant<uint64_t>(node->slice.start), constant<uint64_t>(node->slice.count) };

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
				last_write = current_module->make_converge(slice_node->slice.src.type(), tails);
				last_write.node->index = node->index;
				add_node(last_write.node);
				// this was consumed by a converge - this means we need to replicate the slice
			} else if (link->undef.node->kind == Node::CONVERGE) {
#ifdef VUK_DUMP_SSA
				fmt::println("convergence - replicating slice");
#endif
				last_write =
				    current_module->make_slice(parm.node->type[0], first(link->undef.node), parm.node->slice.axis, parm.node->slice.start, parm.node->slice.count);
				last_write.node->index = node->index;
				add_node(last_write.node);
			} else if (link->undef.node->kind == Node::CONSTRUCT && first(link->undef.node).type()->kind == Type::UNION_TY) {
#ifdef VUK_DUMP_SSA
				fmt::println("construct - replicating extract");
#endif
				last_write = current_module->make_extract(first(link->undef.node), link->undef.index - 1);
				last_write.node->index = node->index;
				allocate_node_links(last_write.node->slice.start.node);
				process_node_links(last_write.node->slice.start.node);
				allocate_node_links(last_write.node->slice.count.node);
				process_node_links(last_write.node->slice.count.node);
				add_node(last_write.node);
			} else {
				VUK_ICE(false);
			}
		} else {
			last_write = link->def;
		}

		return last_write;
	};

	void IRPass::add_write(Node* node, Ref& parm, size_t index) {
		VUK_ICE(parm.node->kind != Node::GARBAGE);
		if (!parm.node->links) {
			VUK_ICE(do_ssa);
			// external node -> init
			allocate_node_links(parm.node);
			for (size_t i = 0; i < parm.node->type.size(); i++) {
				Ref{ parm.node, i }.link().def = { parm.node, i };
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

	void IRPass::add_breaking_result(Node* node, size_t output_idx) {
		Ref out = Ref{ node, output_idx };
		out.link().def = { node, output_idx };
	};

	void IRPass::add_result(Node* node, size_t output_idx, Ref parm) {
		if (!node->links) {
			VUK_ICE(do_ssa);
			// external node -> init
			allocate_node_links(node);
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

	void IRPass::add_read(Node* node, Ref& parm, size_t index, bool needs_ssa = true) {
		VUK_ICE(parm.node->kind != Node::GARBAGE);
		auto& st_parm = parm;
		if (!st_parm.node->links) {
			VUK_ICE(do_ssa);
			// external node -> init
			allocate_node_links(st_parm.node);
			for (size_t i = 0; i < st_parm.node->type.size(); i++) {
				Ref{ st_parm.node, i }.link().def = { st_parm.node, i };
			}
		}
		auto link = &st_parm.link();
		if (link->undef.node != nullptr && node->index > link->undef.node->index && needs_ssa) { // there is already a write and it is earlier than us
			VUK_ICE(do_ssa);
			auto last_write = walk_writes(node, parm);
			parm = last_write;
			link = &parm.link();
		}
		link->reads.append(impl.pass_reads, { node, index });
	};

	void IRPass::process_node_links(Node* node) {
#ifdef VUK_DUMP_SSA
		debug_stack.push_back(std::string(node->kind_to_sv(node->kind)));
		print_ctx();
		fmt::println("entering");
#endif
		switch (node->kind) {
		case Node::SET: // not a real node
			break;
		case Node::CONSTANT:
		case Node::PLACEHOLDER:
			add_breaking_result(node, 0);
			break;
		case Node::CONSTRUCT:
			first(node).link().def = first(node);

			for (size_t i = 0; i < node->construct.args.size(); i++) {
				auto& parm = node->construct.args[i];
				if (node->type[0]->kind == Type::ARRAY_TY || node->type[0]->kind == Type::UNION_TY || parm.type()->kind == Type::POINTER_TY) {
					add_write(node, parm, i);
				} else {
					add_read(node, parm, i);
				}
			}
			for (size_t i = 1; i < node->construct.args.size(); i++) {
				auto& parm = node->construct.args[i];
				if (node->type[0]->kind == Type::ARRAY_TY || node->type[0]->hash_value == current_module->types.builtin_sampled_image ||
				    parm.type()->kind == Type::POINTER_TY) {
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
					if (do_ssa && base->is_imageview()) {
						auto def = eval<ImageView<>>(parm);
						/* if (def.holds_value() && !def->image) { // TODO : PAV : we need observe only allocates here..
						  access_to_usage(def->usage, access);
						}*/
					}
				} else {
					assert(0); // not handling non-imbued yet
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
				if (do_ssa && node->type[out]->is_bufferlike_view()) {
					auto& buf = *reinterpret_cast<Buffer<>*>(node->acquire.values[out]);
					auto bo = runtime.ptr_to_buffer_offset(buf.ptr);
					assert(bo.buffer != VK_NULL_HANDLE);
					bool found = false;
					for (auto& [existing_buf, link] : impl.bufs) {
						if (bo.buffer == existing_buf.buffer && Range{ bo.offset, buf.sz_bytes }.intersect({ existing_buf.offset, existing_buf.size })) {
							// we will need to union the buffers

							std::array args = { Ref{ node, out }, link->def };
							auto con_union = current_module->make_declare_union(args);
							con_union.node->index = node->index;
							allocate_node_links(con_union.node);
							process_node_links(con_union.node);
							new_nodes.push_back(con_union.node);
							found = true;
							break;
						}
					}
					if (!found) {
						impl.bufs.emplace_back(Runtime::Resolver::BufferWithOffsetAndSize{ runtime.ptr_to_buffer_offset(buf.ptr), buf.sz_bytes }, &nth(node, out).link());
					}
				}
			}
			break;

		case Node::SLICE: {
			add_read(node, node->slice.start, 1);
			add_read(node, node->slice.count, 2);
			if (node->type[0]->kind == Type::INTEGER_TY) {
				add_read(node, node->slice.src, 0, false);
			} else {
				add_write(node, node->slice.src, 0);
			}
			if (node->kind == Node::LOGICAL_COPY) { // if we rewrote this to a copy
				add_result(node, 0, node->slice.src);
				break;
			}
			nth(node, 0).link().def = nth(node, 0); // we introduce the slice image def
			nth(node, 1).link().def = nth(node, 1); // we introduce the rest image def
			add_breaking_result(node, 2);
			if (node->slice.src.node->links) {
				node->slice.src.link().child_chains.append(impl.child_chains, &nth(node, 0).link()); // add child chain for sliced
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

		case Node::COMPILE_PIPELINE:
			add_breaking_result(node, 0);
			add_read(node, node->compile_pipeline.src, 0);
			break;
		case Node::GET_ALLOCATION_SIZE:
			add_read(node, node->get_allocation_size.ptr, 0);
			add_breaking_result(node, 0);
			break;
		case Node::GET_IV_META:
			add_read(node, node->get_iv_meta.imageview, 0);
			add_breaking_result(node, 0);
			break;
		case Node::ALLOCATE:
			add_read(node, node->allocate.src, 0);
			add_result(node, 0, node->allocate.src);
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

	Result<void> link_building::implicit_linking(std::pmr::vector<Node*>& nodes) {
		do_ssa = true;
		impl.pass_reads.clear();
		impl.child_chains.clear();

		for (auto& node : nodes) {
			allocate_node_links(node);
		}

		for (auto& node : nodes) {
			process_node_links(node);
		}

		return { expected_value };
	}

	Result<void> link_building::operator()() {
		do_ssa = false;
		impl.pass_reads.clear();
		impl.child_chains.clear();

		// in each IRModule module, look at the nodes
		// declare -> clear -> call(R) -> call(W) -> release
		//   A     ->  B    ->  B      ->   C     -> X
		// declare: def A -> new entry
		// clear: undef A, def B
		// call(R): read B
		// call(W): undef B, def C
		// release: undef C
		for (auto& node : impl.nodes) {
			allocate_node_links(node);
		}

		for (auto& node : impl.nodes) {
			process_node_links(node);
		}
		// TODO:
		// we need a pass that walks through links
		// an incompatible read group contains multiple domains
		// in this case they can't be together - so we linearize them into domain groups
		// so def -> {r1, r2} becomes def -> r1 -> undef{g0} -> def{g0} -> r2

		return { expected_value };
	};
} // namespace vuk