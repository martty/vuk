#include "vuk/Exception.hpp"
#include "vuk/ir/IRPasses.hpp"
#include "vuk/ir/IRProcess.hpp"
#include "vuk/RadixTree.hpp"

namespace vuk {
	Result<void> validate_duplicated_resource_ref::operator()() {
		RadixTree<void*> memory;
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
			} else if (node->type[0]->is_bufferlike_view()) { // bufferlike views
				auto& buf = *reinterpret_cast<Buffer<>*>(value);
				bool succ = !memory.insert_unaligned(buf.ptr.device_address, buf.sz_bytes, node);
				if (!succ) {
					// TODO: PAV: search entire range
					auto node2 = memory.find(buf.ptr.device_address);
					assert(node2);
					return reinterpret_cast<Node*>(*node2);
				}
			} else if (type->hash_value == current_module->types.builtin_swapchain) {
				auto swp = reinterpret_cast<Swapchain*>(value);
				auto [_, succ] = swps.emplace();
				if (!succ) {
					return swps.at(swp);
				}
			} else { // TODO: no val yet for arrays
			}

			return {};
		};
		for (auto node : impl.nodes) {
			std::optional<Node*> fail = {};
			switch (node->kind) {
			case Node::CONSTANT:
			case Node::CONSTRUCT: {
				auto value = eval(first(node));
				if (!value) { // cannot be constant evaluated -> we are going to allocate it, therefore it can't alias
					(void)value.error();
					break;
				}
				fail = add_one(node->type[0].get(), node, *value);
			} break;
			case Node::ACQUIRE: {
				for (size_t i = 0; i < node->type.size(); i++) {
					auto as_ref = nth(node, i);
					auto& link = as_ref.link();
					if (link.reads.size() == 0 && !link.undef && !link.next) { // if not used, we don't care about it
						continue;
					}
					fail = add_one(node->type[i].get(), node, node->acquire.values[i]);
					if (fail && node->type[i]->is_bufferlike_view() && fail.value()->kind == Node::ACQUIRE) { // an acq-acq for buffers, this is allowed
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
} // namespace vuk