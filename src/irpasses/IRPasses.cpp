#include "vuk/Exception.hpp"
#include "vuk/ir/GraphDumper.hpp"
#include "vuk/ir/IRPasses.hpp"
#include "vuk/ir/IRProcess.hpp"
#include "vuk/RenderGraph.hpp"
#include "vuk/runtime/CommandBuffer.hpp"
#include "vuk/SyncLowering.hpp"

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

namespace vuk {
	Result<void> RGCImpl::run_passes(Runtime& runtime, std::pmr::polymorphic_allocator<std::byte> allocator) {
		Result<void> result = { expected_value };
		for (auto& pass_factory : ir_passes) {
			auto pass = pass_factory(*this, runtime, allocator);
			result = (*pass)();
			if (!result) {
				// we failed. lets see what we can do.
				// we are always before linearization here, so lets try to linearize
				auto linear_res = linearize(runtime, allocator);
				if (!linear_res) {
					// we also failed at linearization, so we are going to dump the graph and bail out
					// TODO: dump graph
					return result;
				}
				// we succeeded at linearizing, so lets dump the IR
				fmt::println("IR listing");
				size_t instr_counter = 0;
				for (auto& pitem : item_list) {
					auto& item = *pitem;
					auto node = item.execable;
					instr_counter++;
					fmt::println("[{:#06x}] {}", instr_counter, exec_to_string(item));
				}
				return result;
			}
			(void)result;
			if (pass->node_set_modified() || new_nodes.size() > 0) {
				nodes.insert(nodes.end(), new_nodes.begin(), new_nodes.end());
				new_nodes.clear();
				VUK_DO_OR_RETURN(build_nodes());
			}
			if (pass->node_set_modified() || new_nodes.size() > 0 || pass->node_connections_modified()) {
				VUK_DO_OR_RETURN(build_links(runtime, nodes, allocator));
			}
		}

		return { expected_value };
	}

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

	Result<void> RGCImpl::build_nodes() {
		nodes.clear();

		std::vector<Node*, short_alloc<Node*>> work_queue(*arena_);
		for (auto& node : ref_nodes) {
			if (node->flag == 0) {
				node->flag = 1;
				work_queue.push_back(node);
			}
		}
		for (auto& node : set_nodes) {
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
				if (!parm.type()->is_bufferlike_view() && type != current_module->types.builtin_image) {
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
			case Node::ALLOCATE: {                 // ALLOCATE discards
				if (node->links->reads.size() > 0) { // we are trying to read from it :(
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

	Result<void> Compiler::validate_same_argument_different_access() {
		std::unordered_map<Ref, size_t> arg_set;
		for (auto node : impl->nodes) {
			switch (node->kind) {
			case Node::CALL: {
				arg_set.clear();
				auto fn_type = node->call.args[0].type();
				size_t first_parm = fn_type->kind == Type::OPAQUE_FN_TY ? 1 : 4;
				auto& args = fn_type->kind == Type::OPAQUE_FN_TY ? fn_type->opaque_fn.args : fn_type->shader_fn.args;
				for (size_t i = first_parm; i < node->call.args.size(); i++) {
					auto& arg_ty = args[i - first_parm];
					auto& parm = node->call.args[i];

					auto [it, succ] = arg_set.try_emplace(parm, i);
					if (!succ) {
						auto other_arg_ty = args[it->second - first_parm];
						assert(arg_ty->kind == Type::IMBUED_TY);
						assert(other_arg_ty->kind == Type::IMBUED_TY);
						if (arg_ty->imbued.access == other_arg_ty->imbued.access) { // same is okay
							continue;
						}
						auto msg = fmt::format("tried to pass the same value through #{}({}) and #{}({}) with different access",
						                       it->second - first_parm,
						                       Type::to_sv(other_arg_ty->imbued.access),
						                       i - first_parm,
						                       Type::to_sv(arg_ty->imbued.access));
						return { expected_error, RenderGraphException{ format_graph_message(Level::eError, node, msg) } };
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

	Result<void> RGCImpl::implicit_linking(Allocator& alloc, IRModule* module, std::pmr::polymorphic_allocator<std::byte> allocator) {
		std::pmr::vector<Node*> nodes(allocator);

		for (auto& node : module->op_arena) {
			if (node.index < (module->module_id << 32 | module->link_frontier) && node.kind != Node::ACQUIRE) { // already linked
				continue;
			}

			if (node.kind == Node::SET) {
				set_nodes.push_back(&node);
			} else if (node.kind == Node::CALL && node.call.args[0].type()->kind == Type::MEMORY_TY) { // we need to compile this PBCI
				auto& pbci = constant<PipelineBaseCreateInfo>(node.call.args[0]);
				auto pipeline = alloc.get_context().get_pipeline(pbci);
				auto& flat_bindings = pipeline->reflection_info.flat_bindings;

				std::vector<std::shared_ptr<Type>> arg_types;
				std::vector<std::shared_ptr<Type>> ret_types;
				std::shared_ptr<Type> base_ty;
				size_t i = 0;
				for (auto& [set_index, b] : flat_bindings) {
					Access acc = Access::eNone;
					switch (b->type) {
					case DescriptorType::eSampledImage:
						acc = Access::eComputeSampled;
						base_ty = current_module->types.get_builtin_image();
						break;
					case DescriptorType::eCombinedImageSampler:
						acc = Access::eComputeSampled;
						base_ty = current_module->types.get_builtin_sampled_image();
						break;
					case DescriptorType::eStorageImage:
						acc = b->non_writable ? Access::eComputeRead : (b->non_readable ? Access::eComputeWrite : Access::eComputeRW);
						base_ty = current_module->types.get_builtin_image();
						break;
					case DescriptorType::eUniformBuffer:
					case DescriptorType::eStorageBuffer:
						acc = b->non_writable ? Access::eComputeRead : (b->non_readable ? Access::eComputeWrite : Access::eComputeRW);
						base_ty = to_IR_type<Buffer<>>();
						break;
					case DescriptorType::eSampler:
						acc = Access::eNone;
						base_ty = current_module->types.get_builtin_sampler();
						break;
					default:
						assert(0);
					}

					arg_types.push_back(current_module->types.make_imbued_ty(base_ty, acc));
					ret_types.emplace_back(current_module->types.make_aliased_ty(base_ty, i + 4));
					i++;
				}
				auto shader_fn_ty = current_module->types.make_shader_fn_ty(arg_types, ret_types, vuk::DomainFlagBits::eAny, pipeline, pipeline->pipeline_name.c_str());
				node.call.args[0] = current_module->make_declare_fn(shader_fn_ty);
				delete[] node.type.data();
				node.type = { new std::shared_ptr<Type>[ret_types.size()], ret_types.size() };
				std::copy(ret_types.begin(), ret_types.end(), node.type.data());
				nodes.push_back(&node);
			} else {
				nodes.push_back(&node);
			}
		}

		std::sort(nodes.begin(), nodes.end(), [](Node* a, Node* b) { return a->index < b->index; });
		// link with SSA
		build_links_implicit(alloc.get_context(), nodes, allocator);
		module->link_frontier = module->node_counter;
		return { expected_value };
	}

	Result<void> RGCImpl::build_links_implicit(Runtime& runtime, std::pmr::vector<Node*>& working_set, std::pmr::polymorphic_allocator<std::byte> allocator) {
		return link_building(*this, runtime, allocator).implicit_linking(working_set);
	}

	Result<void> RGCImpl::build_links(Runtime& runtime, std::vector<Node*>& working_set, std::pmr::polymorphic_allocator<std::byte> allocator) {
		return link_building(*this, runtime, allocator)();
	}

	Result<void> RGCImpl::linearize(Runtime& runtime, std::pmr::polymorphic_allocator<std::byte> allocator) {
		return linearization(*this, runtime, allocator)();
	}

	Result<void> Compiler::compile(Allocator& alloc, std::span<std::shared_ptr<ExtNode>> nodes, const RenderGraphCompileOptions& compile_options) {
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
			VUK_DO_OR_RETURN(impl->implicit_linking(alloc, m, allocator));
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
		VUK_DO_OR_RETURN(impl->build_links(alloc.get_context(), impl->nodes, allocator));
		GraphDumper::next_cluster("modules", "full");
		GraphDumper::dump_graph(impl->nodes, false, false);

		// apply SET nodes
		for (auto& s : impl->set_nodes) {
			auto link = &s->set.dst.link();
			if (!link) {
				continue;
			}
			while (link->prev) {
				link = link->prev;
			}
			auto def_node = link->def.node;
			if (def_node->kind == Node::CONSTRUCT) {
				def_node->construct.args[s->set.index + 1] = s->set.value;
			}
		}
		impl->set_nodes.clear();

		VUK_DO_OR_RETURN(impl->build_nodes());
		VUK_DO_OR_RETURN(impl->build_links(alloc.get_context(), impl->nodes, allocator));
		impl->ir_passes = {
			{ make_ir_pass<constant_folding>(), make_ir_pass<reify_inference>(), make_ir_pass<constant_folding>(), make_ir_pass<validate_duplicated_resource_ref>() }
		};
		impl->run_passes(alloc.get_context(), allocator);
		VUK_DO_OR_RETURN(validate_read_undefined());
		VUK_DO_OR_RETURN(validate_same_argument_different_access());

		VUK_DO_OR_RETURN(impl->collect_chains());
		impl->ir_passes = { { make_ir_pass<forced_convergence>() } };
		impl->run_passes(alloc.get_context(), allocator);
		VUK_DO_OR_RETURN(impl->collect_chains());

		impl->scheduled_execables.clear();

		for (auto& node : impl->ref_nodes) {
			assert(node);
			ScheduledItem item{ .execable = node, .scheduled_domain = vuk::DomainFlagBits::eAny };
			auto it = impl->scheduled_execables.emplace(item);
			it->execable->scheduled_item = &*it;
		}

		for (auto& node : impl->nodes) {
			assert(node);
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

		VUK_DO_OR_RETURN(impl->linearize(alloc.get_context(), allocator));

		// we have added some nodes to the current module - but these are considered to be linked
		// so we advanced the frontier of the current module to improve GC
		current_module->link_frontier = current_module->node_counter;

		return { expected_value };
	}

	std::span<ChainLink*> Compiler::get_use_chains() const {
		return std::span(impl->chains);
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
