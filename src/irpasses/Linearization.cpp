#include "vuk/ir/GraphDumper.hpp"
#include "vuk/ir/IRPasses.hpp"
#include "vuk/SyncLowering.hpp"

namespace vuk {
	Result<void> linearization::operator()() {
		if (impl.scheduled_execables.empty()) {
			for (auto& node : impl.ref_nodes) {
				assert(node);
				ScheduledItem item{ .execable = node, .scheduled_domain = vuk::DomainFlagBits::eAny };
				auto it = impl.scheduled_execables.emplace(item);
				it->execable->scheduled_item = &*it;
			}

			for (auto& node : impl.nodes) {
				assert(node);
				switch (node->kind) {
				case Node::SLICE:
				case Node::CALL: {
					ScheduledItem item{ .execable = node, .scheduled_domain = vuk::DomainFlagBits::eAny };
					auto it = impl.scheduled_execables.emplace(item);
					it->execable->scheduled_item = &*it;
				} break;

				default:
					break;
				}
			}
		}

		impl.naming_index_counter = 0;
		impl.scheduled.clear();
		impl.item_list.clear();
		std::vector<ScheduledItem> initial_set(impl.scheduled_execables.begin(), impl.scheduled_execables.end());

		// these are the items that were determined to run
		for (auto& i : initial_set) {
			impl.work_queue.emplace_back(RGCImpl::Sched{ i.execable, false });
			impl.expanded.clear();

			while (!impl.work_queue.empty()) {
				RGCImpl::Sched item = impl.work_queue.front();
				impl.work_queue.pop_front();

				auto& node = item.node;
				assert(node);
				if (impl.scheduled.contains(node)) { // only going schedule things once
					continue;
				}

				// we run nodes twice - first time we reenqueue at the front and then put all deps before it
				// second time we see it, we know that all deps have run, so we can run the node itself
				if (impl.process(item)) {
					impl.scheduled.emplace(node);
					node->scheduled_item->naming_index = impl.naming_index_counter;
					impl.item_list.push_back(node->scheduled_item);
					impl.naming_index_counter += node->type.size();
				} else {
					switch (node->kind) {
					case Node::MATH_BINARY: {
						for (auto i = 0; i < node->fixed_node.arg_count; i++) {
							impl.schedule_dependency(node->fixed_node.args[i], RW::eRead);
						}
					} break;
					case Node::CONSTRUCT: {
						for (auto& parm : node->construct.args.subspan(1)) {
							impl.schedule_dependency(parm, RW::eRead);
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
								impl.schedule_dependency(parm, sync_access);
							} else {
								assert(0);
							}
						}
					} break;
					case Node::RELEASE: {
						auto acqrel = node->rel_acq;
						if (!acqrel || acqrel->status == Signal::Status::eDisarmed) {
							for (size_t i = 0; i < node->release.src.size(); i++) {
								impl.schedule_dependency(node->release.src[i], RW::eWrite);
							}
						}
					} break;
					case Node::ACQUIRE: {
						// ACQUIRE does not have any deps
					} break;
					case Node::ACQUIRE_NEXT_IMAGE: {
						impl.schedule_dependency(node->acquire_next_image.swapchain, RW::eWrite);
					} break;
					case Node::SLICE: {
						if (!node->type[0]->is_synchronized()) {
							impl.schedule_dependency(node->slice.src, RW::eRead);
						} else {
							impl.schedule_dependency(node->slice.src, RW::eWrite);
						}
						impl.schedule_dependency(node->slice.start, RW::eRead);
						impl.schedule_dependency(node->slice.count, RW::eRead);
					} break;
					case Node::CONVERGE: {
						for (size_t i = 0; i < node->converge.diverged.size(); i++) {
							impl.schedule_dependency(node->converge.diverged[i], RW::eWrite);
						}
					} break;
					case Node::USE: {
						impl.schedule_dependency(node->use.src, RW::eWrite);
					} break;
					case Node::LOGICAL_COPY: {
						impl.schedule_dependency(node->logical_copy.src, RW::eRead);
					} break;
					case Node::COMPILE_PIPELINE: {
						impl.schedule_dependency(node->compile_pipeline.src, RW::eRead);
					} break;
					case Node::ALLOCATE: {
						impl.schedule_dependency(node->allocate.src, RW::eRead);
					} break;
					case Node::GET_ALLOCATION_SIZE: {
						impl.schedule_dependency(node->get_allocation_size.ptr, RW::eRead);
					} break;
					default:
						VUK_ICE(false);
						break;
					}
					impl.expanded.emplace(item.node);
				}
			}
		}

		return { expected_value };
	}
} // namespace vuk