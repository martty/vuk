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
#include <set>
#include <sstream>
#include <unordered_set>

namespace vuk {
	void RGCImpl::dump_graph() {
		std::stringstream ss;
		ss << "digraph vuk {\n";
		ss << "rankdir=\"TB\"\nnewrank = true\nnode[shape = rectangle width = 0 height = 0 margin = 0]\n";
		for (auto node : nodes) {
			if (node->kind == Node::CONSTANT) {
				if (node->type[0]->kind == Type::INTEGER_TY || node->type[0]->kind == Type::MEMORY_TY) {
					continue;
				}
			}
			if (node->kind == Node::PLACEHOLDER || node->kind == Node::SPLICE || node->kind == Node::SLICE || node->kind == Node::INDIRECT_DEPEND) {
				continue;
			}

			auto arg_count = node->generic_node.arg_count == (uint8_t)~0u ? node->variable_node.args.size() : node->generic_node.arg_count;
			auto result_count = node->type.size();
			ss << uintptr_t(node) << " [label=<\n";
			ss << "<TABLE BORDER=\"0\" CELLBORDER=\"1\" CELLSPACING=\"0\"";
			ss << "><TR>\n ";

			if (node->debug_info) {
				for (auto& n : node->debug_info->result_names) {
					ss << "<TD>";
					ss << "%" << n;
					ss << "</TD>";
				}
			}

			for (size_t i = 0; i < result_count; i++) {
				ss << "<TD PORT= \"r" << i << "\">";
				ss << "<FONT FACE=\"Courier New\">";
				ss << Type::to_string(node->type[i]);
				ss << "</FONT>";
				ss << "</TD>";
			}
			ss << "<TD>";
			ss << node->kind_to_sv();
			if (node->kind == Node::CALL) {
				auto opaque_fn_ty = node->call.args[0].type()->opaque_fn;

				if (node->call.args[0].type()->debug_info) {
					ss << " <B>";
					ss << node->call.args[0].type()->debug_info->name;
					ss << "</B>";
				}
			}
			ss << "</TD>";

			for (size_t i = 0; i < arg_count; i++) {
				auto arg = node->generic_node.arg_count != (uint8_t)~0u ? node->fixed_node.args[i] : node->variable_node.args[i];

				ss << "<TD PORT= \"a" << i << "\">";
				if (arg.node->kind == Node::CONSTANT) {
					if (arg.type()->kind == Type::INTEGER_TY) {
						if (arg.type()->integer.width == 32) {
							ss << constant<uint32_t>(arg);
						} else {
							ss << constant<uint64_t>(arg);
						}
					} else if (arg.type()->kind == Type::MEMORY_TY) {
						ss << "&lt;mem&gt;";
					}
				} else if (arg.node->kind == Node::PLACEHOLDER) {
					ss << "?";
				} else {
					if (node->kind == Node::CALL) {
						auto opaque_fn_ty = node->call.args[0].type()->opaque_fn;
						if (opaque_fn_ty.args[i - 1]->kind == Type::IMBUED_TY) {
							ss << "<FONT FACE=\"Courier New\">";
							ss << ":" << Type::to_sv(opaque_fn_ty.args[i - 1]->imbued.access);
							ss << "</FONT>";
						}
					} else {
						ss << "&bull;";
					}
				}
				ss << "</TD>";
			}

			ss << "</TR></TABLE>>];\n";

			// connections
			for (size_t i = 0; i < arg_count; i++) {
				auto arg = node->generic_node.arg_count != (uint8_t)~0u ? node->fixed_node.args[i] : node->variable_node.args[i];
				if (arg.node->kind == Node::CONSTANT) {
					if (arg.type()->kind == Type::INTEGER_TY || arg.type()->kind == Type::MEMORY_TY) {
						continue;
					}
				}
				if (arg.node->kind == Node::PLACEHOLDER) {
					continue;
				}
				if (arg.node->kind == Node::SPLICE && arg.node->splice.rel_acq && arg.node->splice.rel_acq->status == Signal::Status::eDisarmed) { // bridge splices
					auto bridged_arg = arg.node->splice.src[arg.index];
					ss << uintptr_t(bridged_arg.node) << " :r" << bridged_arg.index << " -> " << uintptr_t(node) << " :a" << i << " :n [color=blue]\n";
				} else if (arg.node->kind == Node::INDIRECT_DEPEND) { // bridge indirect depends (connect to node)
					auto bridged_arg = arg.node->indirect_depend.rref;
					ss << uintptr_t(bridged_arg.node) << " -> " << uintptr_t(node) << " :a" << i << " :n [color=blue]\n";
				} else if (arg.node->kind == Node::SLICE) { // bridge slices
					auto bridged_arg = arg.node->slice.image;
					if (bridged_arg.node->kind == Node::SPLICE) {
						bridged_arg = bridged_arg.node->splice.src[arg.index];
					}
					Subrange::Image r = { constant<uint32_t>(arg.node->slice.base_level),
						                    constant<uint32_t>(arg.node->slice.level_count),
						                    constant<uint32_t>(arg.node->slice.base_layer),
						                    constant<uint32_t>(arg.node->slice.layer_count) };
					ss << uintptr_t(bridged_arg.node) << " :r" << bridged_arg.index << " -> " << uintptr_t(node) << " :a" << i << " :n [color=green, label=\"";
					if (r.base_level > 0 || r.level_count != VK_REMAINING_MIP_LEVELS) {
						ss << fmt::format("[m{}:{}]", r.base_level, r.base_level + r.level_count - 1);
					}
					if (r.base_layer > 0 || r.layer_count != VK_REMAINING_ARRAY_LAYERS) {
						ss << fmt::format("[l{}:{}]", r.base_layer, r.base_layer + r.layer_count - 1);
					}
					ss << "\"]\n";
				} else {
					ss << uintptr_t(arg.node) << " :r" << arg.index << " -> " << uintptr_t(node) << " :a" << i << " :n\n";
				}
			}
		}
		ss << "}\n";
		printf("\n\n%s\n\n", ss.str().c_str());
		printf("");
	}

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

	Result<void> RGCImpl::build_links() {
		// build edges into link map
		// reserving here to avoid rehashing map
		pass_reads.clear();

		// in each IRModule module, look at the nodes
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
				node->links = new (arena_->allocate(sizeof(ChainLink) * result_count)) ChainLink[result_count];
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
				first(node).link().def = first(node);

				for (size_t i = 0; i < node->construct.args.size(); i++) {
					auto& parm = node->construct.args[i];
					parm.link().undef = { node, i };
				}

				if (node->type[0]->kind == Type::ARRAY_TY) {
					for (size_t i = 1; i < node->construct.args.size(); i++) {
						auto& parm = node->construct.args[i];
						parm.link().next = &first(node).link();
					}
				}

				break;
			case Node::SPLICE: { // ~~ write joiner
				for (size_t i = 0; i < node->type.size(); i++) {
					Ref{ node, i }.link().def = { node, i };
					if (!node->splice.rel_acq || node->splice.rel_acq && node->splice.rel_acq->status == Signal::Status::eDisarmed) {
						assert(node->splice.src[i].link().undef.node == nullptr);
						node->splice.src[i].link().undef = { node, i };
						node->splice.src[i].link().next = &Ref{ node, i }.link();
						Ref{ node, i }.link().prev = &node->splice.src[i].link();
					}
				}
				break;
			}
			case Node::ACQUIRE:
				first(node).link().def = first(node);
				break;
			case Node::CALL: {
				// args
				for (size_t i = 1; i < node->call.args.size(); i++) {
					auto& arg_ty = node->call.args[0].type()->opaque_fn.args[i - 1];
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
				if (node->release.arg_count == 1) {
					node->release.src.link().undef = { node, 0 };
					first(node).link().prev = &node->release.src.link();
				}
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
					auto write = node->converge.write[i - 1];
					if (write) {
						parm.link().undef = { node, i };
					} else {
						parm.link().reads.append(pass_reads, { node, i });
					}
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

		bool progress = false;

		auto placeholder_to_constant = [this, &progress]<class T>(Ref r, T value) {
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
			case Node::CONSTRUCT:
				auto args_ptr = node->construct.args.data();
				if (node->type[0] == current_module.builtin_image) {
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
				} else if (node->type[0] == current_module.builtin_buffer) {
					auto ptr = &constant<Buffer>(args_ptr[0]);
					auto& value = constant<Buffer>(args_ptr[0]);
					if (value.size != ~(0u)) {
						placeholder_to_ptr(args_ptr[1], &ptr->size);
					}
				}
			}
		}

		// framebuffer inference
		try {
			do {
				progress = false;
				for (auto node : nodes) {
					switch (node->kind) {
					case Node::CALL: {
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
									Swapchain& swp = *eval<Swapchain*>(link.urdef.node->acquire_next_image.swapchain);
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
						if (node->type[0] == current_module.builtin_image) {
							if (constant<ImageAttachment>(args[0]).image.image == VK_NULL_HANDLE) { // if there is no image, we will use base layer 0 and base mip 0
								placeholder_to_constant(args[6], 0U);
								placeholder_to_constant(args[8], 0U);
							}
						}
						break;
					}
					}
				}
			} while (progress);
		} catch (vuk::CannotBeConstantEvaluated& exc) {
			return { expected_error, exc };
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

	Result<void> RGCImpl::build_sync() {
		for (auto node : nodes) {
			switch (node->kind) {
			case Node::CALL: {
				// args
				for (size_t i = 1; i < node->call.args.size(); i++) {
					auto& arg_ty = node->call.args[0].type()->opaque_fn.args[i - 1];
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
									arg_ty = r.node->call.args[0].type()->opaque_fn.args[r.index - 1]; // TODO: insert casts instead
									parm = r.node->call.args[r.index];
								} else if (r.node->kind == Node::CONVERGE) {
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
			case Node::SPLICE:
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

	void Compiler::reset() {
		auto arena = impl->arena_.release();
		delete impl;
		arena->reset();
		impl = new RGCImpl(arena);
	}

	Result<void> Compiler::compile(std::span<std::shared_ptr<ExtNode>> nodes, const RenderGraphCompileOptions& compile_options) {
		reset();
		impl->callbacks = compile_options.callbacks;

		impl->refs.assign(nodes.begin(), nodes.end());

		std::vector<std::shared_ptr<ExtNode>, short_alloc<std::shared_ptr<ExtNode>>> extnode_work_queue(*impl->arena_);
		extnode_work_queue.assign(nodes.begin(), nodes.end());

		while (!extnode_work_queue.empty()) {
			auto enode = extnode_work_queue.back();
			extnode_work_queue.pop_back();
			extnode_work_queue.insert(extnode_work_queue.end(), std::make_move_iterator(enode->deps.begin()), std::make_move_iterator(enode->deps.end()));
			enode->deps.clear();
			impl->depnodes.push_back(std::move(enode));
		}

		std::sort(impl->depnodes.begin(), impl->depnodes.end());
		impl->depnodes.erase(std::unique(impl->depnodes.begin(), impl->depnodes.end()), impl->depnodes.end());

		// implicit convergence: this has to be done on the full node set
		// insert converge nodes
		std::unordered_map<Ref, std::vector<Ref>> slices;

		// linked-sea-of-nodes to list of nodes
		for (auto& n : current_module.op_arena) {
			auto node = &n;
			if (node->kind != Node::CONVERGE) {
				impl->nodes.push_back(node);
			}
			switch (node->kind) {
			case Node::NOP: {
				// impl->garbage_nodes.push_back(node);
				break;
			}
			case Node::SLICE: {
				assert(node->slice.image.node->kind != Node::NOP);
				slices[node->slice.image].push_back(first(node));
				break;
			}
			}
		}

		// build links for the full node set
		VUK_DO_OR_RETURN(impl->build_links());

		// insert converge nodes
		auto in_module = [](IRModule& module, Node* node) {
			auto it = std::find_if(module.op_arena.begin(), module.op_arena.end(), [=](auto& n) { return &n == node; });
			if (it != module.op_arena.end()) {
				return true;
			}
			return false;
		};

		auto before_module = [](IRModule& module, Node* a, Node* b) {
			auto it_a = std::find_if(module.op_arena.begin(), module.op_arena.end(), [=](auto& n) { return &n == a; });
			auto it_b = std::find_if(module.op_arena.begin(), module.op_arena.end(), [=](auto& n) { return &n == b; });
			return it_a < it_b;
		};

		for (auto& [base, sliced] : slices) {
			std::vector<Ref, short_alloc<Ref>> tails(*impl->arena_);
			std::vector<char, short_alloc<char>> write(*impl->arena_);
			for (auto& s : sliced) {
				auto r = &s.link();
				while (r->next) {
					r = r->next;
				}
				if (r->undef.node) { // depend on undefs indirectly via INDIRECT_DEPEND
					auto idep = current_module.make_indirect_depend(r->undef.node, r->undef.index);
					tails.push_back(idep);
					write.push_back(false);
				} else if (r->reads.size() > 0) { // depend on reads indirectly via INDIRECT_DEPEND
					tails.push_back(r->def);
					write.push_back(true);
				} else {
					tails.push_back(r->def); // depend on def directly (via a read)
					write.push_back(false);
				}
			}
			auto converged_base = current_module.make_converge(base, tails, write);
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
								assert(in_module(*current_module., t.node));
								if (!before_module(*current_module., t.node, node)) {
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
								/* assert(in_module(*current_module., t.node));
								if (!before_module(*current_module., t.node, node)) {
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
		std::erase_if(impl->depnodes, [](std::shared_ptr<ExtNode>& sp) { return sp.use_count() == 1 && sp->acqrel->status == Signal::Status::eDisarmed; });

		// eliminate useless splices
		struct Replace {
			Ref needle;
			Ref value;
		};
		std::vector<Replace, short_alloc<Replace>> replaces(*impl->arena_);
		std::vector<Ref*, short_alloc<Ref*>> args(*impl->arena_);

		for (auto node : impl->nodes) {
			switch (node->kind) {
			case Node::SPLICE: {
				if (node->splice.rel_acq == nullptr) {
					for (size_t i = 0; i < node->splice.src.size(); i++) {
						auto needle = Ref{ node, i };
						auto replace_with = node->splice.src[i];

						replaces.emplace_back(Replace{needle, replace_with});
					}
					impl->garbage_nodes.push_back(node);
				} else {
					switch (node->splice.rel_acq->status) {
					case Signal::Status::eDisarmed: // means we have to signal this, keep
						break;
					case Signal::Status::eSynchronizable: // means this is an acq instead
					case Signal::Status::eHostAvailable:
						for (size_t i = 0; i < node->type.size(); i++) {
							auto new_ref = current_module.make_acquire(node->type[i], node->splice.rel_acq, i, node->splice.values[i]);
							replaces.emplace_back(Replace{ Ref{ node, i }, new_ref });
							impl->garbage_nodes.emplace_back(new_ref.node);
						}
						break;
					}
				}
				break;
			}
			case Node::RELEASE: {
				if (node->links[0].reads.size() > 0 || node->links[0].undef) {
					auto needle = Ref{ node, 0 };
					if (node->release.release->status == Signal::Status::eDisarmed) {
						auto new_node = current_module.make_splice(node->release.src.node, node->release.release);
						replaces.emplace_back(needle, first(new_node));
						impl->garbage_nodes.emplace_back(new_node);
					} else {
						Node acq_node{ .kind = Node::ACQUIRE,
							             .type = { new Type*[1](node->type[0]), 1 },
							             .acquire = { .value = node->release.value, .acquire = node->release.release, .index = 0 } };
						auto new_node = current_module.emplace_op(acq_node);
						replaces.emplace_back(needle, first(new_node));
						impl->garbage_nodes.emplace_back(new_node);
					}
				}
			}
			}
		}

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
		std::sort(replaces.begin(), replaces.end(), [](Replace& a, Replace& b) {
			return a.needle.node < b.needle.node || (a.needle.node == b.needle.node && a.needle.index < b.needle.index);
		});

		// do the replaces
		auto arg_it = args.begin();
		for (auto& r : replaces) {
			while (arg_it != args.end() && **arg_it < r.needle) {
				++arg_it;
			}
			while (arg_it != args.end() && **arg_it == r.needle) {
				**arg_it = r.value;
				++arg_it;
			}
		}

		VUK_DO_OR_RETURN(impl->build_nodes());
		// impl->dump_graph();
		VUK_DO_OR_RETURN(impl->build_links());

		VUK_DO_OR_RETURN(impl->reify_inference());
		VUK_DO_OR_RETURN(impl->collect_chains());

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
					auto& arg_ty = r.node->call.args[0].type()->opaque_fn.args[r.index - 1];
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
					auto& arg_ty = chain->undef.node->call.args[0].type()->opaque_fn.args[chain->undef.index - 1];
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
