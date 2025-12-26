#include "vuk/RenderGraph.hpp"
#include <fmt/format.h>
#include <unordered_set>
#include <vector>
#include <vuk/ir/IR.hpp>
#include <vuk/ir/IRPass.hpp>
#include <vuk/Value.hpp>

namespace vuk {
	thread_local std::shared_ptr<IRModule> current_module = std::make_shared<IRModule>();

	std::shared_ptr<Type> IRModule::Types::make_imageview_ty() {
		using T = Format;
		size_t tag = typeid(T).hash_code();
		auto format_callback = [](void* v, std::string& dst) {
			if constexpr (requires { format_as(*reinterpret_cast<T*>(v)); }) {
				auto formatted = format_as(*reinterpret_cast<T*>(v));
				dst.append(formatted);
			} else {
				// Fallback: format as underlying integer type
				fmt::format_to(std::back_inserter(dst), "{}", static_cast<std::underlying_type_t<T>>(*reinterpret_cast<T*>(v)));
			}
		};
		// Extract type name and create enum type with debug info
		constexpr auto type_name = get_type_name<T>();
		auto enum_type = std::shared_ptr<Type>(new Type{ .kind = Type::ENUM_TY,
		                                                 .size = sizeof(T),
		                                                 .debug_info = current_module->types.allocate_type_debug_info(std::string(type_name)),
		                                                 .enumt = { .format_to = format_callback, .tag = tag } });
		auto enum_ty = emplace_type(enum_type);
		auto pt_ty = make_enum_value_ty(enum_ty, (uint64_t)Format::eUndefined);
		return make_pointer_ty(pt_ty);
	}

	std::shared_ptr<Type> IRModule::Types::make_imageview_ty(std::shared_ptr<Type> pt_ty) {
		return make_pointer_ty(pt_ty);
	}

	namespace {
		// Helper to format a constant value inline
		std::string format_constant_inline(Node* node) {
			Type* ty = node->type[0].get();
			if (ty->kind == Type::INTEGER_TY) {
				switch (ty->scalar.width) {
				case 32:
					return fmt::format("{}", constant<uint32_t>(first(node)));
				case 64:
					return fmt::format("{}", constant<uint64_t>(first(node)));
				}
			} else if (ty->kind == Type::ENUM_VALUE_TY) {
				std::string formatted;
				if (ty->enum_value.enum_type->get()->enumt.format_to) {
					ty->enum_value.enum_type->get()->enumt.format_to((void*)&ty->enum_value.value, formatted);
					return formatted;
				} else {
					return fmt::format("{}", ty->enum_value.value);
				}
			} else if (ty->kind == Type::COMPOSITE_TY) {
				if (ty->composite.format_to) {
					std::string result;
					ty->composite.format_to(node->constant.value, result);
					return result;
				}
			}
			return "";
		}

		// Forward declaration for recursion
		std::string format_ref_inline(Ref ref);

		// Helper to format a slice inline with dot/bracket notation
		std::string format_slice_inline(Node* node) {
			auto src_str = format_ref_inline(node->slice.src);

			// Get the index value if it's a constant
			std::string index_str;
			if (node->slice.start.node->kind == Node::CONSTANT) {
				index_str = format_constant_inline(node->slice.start.node);
			} else {
				// Non-constant index, use variable reference
				if (node->slice.start.node->debug_info && node->slice.start.node->debug_info->result_names.size() > node->slice.start.index) {
					index_str = fmt::format("%{}", node->slice.start.node->debug_info->result_names[node->slice.start.index]);
				} else {
					index_str = fmt::format("[{:#x}:{}]", (uintptr_t)node->slice.start.node, node->slice.start.index);
				}
			}

			switch (node->slice.axis) {
			case Node::NamedAxis::FIELD: {
				// Field access using dot notation
				// Try to get field name from the composite type
				auto src_type = Type::stripped(node->slice.src.type());
				if (src_type->kind == Type::COMPOSITE_TY && !src_type->member_names.empty()) {
					uint64_t field_index = 0;
					if (node->slice.start.node->kind == Node::CONSTANT) {
						field_index = constant<uint64_t>(node->slice.start);
					}
					if (field_index < src_type->member_names.size()) {
						return fmt::format("{}.{}", src_str, src_type->member_names[field_index]);
					}
				}
				// Fallback to numeric field access
				return fmt::format("{}.{}", src_str, index_str);
			}
			case Node::NamedAxis::MIP:
				// Mip level access using m[]
				return fmt::format("{}m[{}]", src_str, index_str);
			case Node::NamedAxis::LAYER:
				// Layer access using l[]
				return fmt::format("{}l[{}]", src_str, index_str);
			case Node::NamedAxis::COMPONENT:
				// Component access
				return fmt::format("{}c[{}]", src_str, index_str);
			default:
				// Array access using []
				return fmt::format("{}[{}]", src_str, index_str);
			}
		}

		// Format a reference inline (recursively handle constants and slices)
		std::string format_ref_inline(Ref ref) {
			if (ref.node->kind == Node::CONSTANT) {
				auto inline_val = format_constant_inline(ref.node);
				if (!inline_val.empty()) {
					return inline_val;
				}
			} else if (ref.node->kind == Node::SLICE) {
				return format_slice_inline(ref.node);
			}

			// Default: use variable name or node reference
			if (ref.node->debug_info && ref.node->debug_info->result_names.size() > ref.index) {
				return fmt::format("%{}", ref.node->debug_info->result_names[ref.index]);
			} else {
				return fmt::format("[{:#x}:{}]", (uintptr_t)ref.node, ref.index);
			}
		}
	} // namespace

	void UntypedValue::dump_ir() const {
		std::vector<Node*> nodes;
		std::unordered_set<Node*> visited;
		std::unordered_set<Node*> inlined; // Track nodes that will be displayed inline

		// Start from the head node
		auto head = get_head();
		std::vector<Node*> work_queue;
		work_queue.push_back(head.node);

		// First pass: identify which nodes should be inlined
		auto should_inline = [](Node* node) {
			return node->kind == Node::CONSTANT || node->kind == Node::SLICE;
		};

		// Traverse the IR graph reachable from this value
		while (!work_queue.empty()) {
			auto node = work_queue.back();
			work_queue.pop_back();

			if (visited.find(node) != visited.end()) {
				continue;
			}
			visited.insert(node);

			// Check if this node should be inlined
			if (should_inline(node) && node != head.node) {
				inlined.insert(node);
				// Still need to traverse arguments of inlined nodes to find dependencies
			} else {
				nodes.push_back(node);
			}

			// Traverse arguments regardless of whether the node is inlined
			apply_generic_args(
			    [&](Ref arg) {
				    if (visited.find(arg.node) == visited.end()) {
					    work_queue.push_back(arg.node);
				    }
			    },
			    node);
		}

		// Reverse to get dependencies before uses (depth-first traversal naturally gives us reverse order)
		std::reverse(nodes.begin(), nodes.end());

		// Dump the collected nodes in order (head node will be last)
		for (auto node : nodes) {
			fmt::print("[{:#x}] ", (uintptr_t)node);

			fmt::print("(");
			for (size_t i = 0; i < node->type.size(); i++) {
				if (i > 0)
					fmt::print(", ");
				if (node->debug_info && node->debug_info->result_names.size() > i) {
					fmt::print("%{}:", node->debug_info->result_names[i]);
				}
				fmt::print("{}", Type::to_string(node->type[i].get()));
			}
			fmt::print(")");

			fmt::print(" = {} ", Node::kind_to_sv(node->kind));

			// Print args using apply_generic_args, but inline constants and slices
			bool first = true;
			apply_generic_args(
			    [&](Ref arg) {
				    if (first) {
					    first = false;
				    } else {
					    fmt::print(", ");
				    }

				    // Format the argument inline if possible
				    fmt::print("{}", format_ref_inline(arg));
			    },
			    node);

			fmt::println("");
		}
	}

	Ref IRModule::make_get_iv_meta(Ref ptr) {
		auto ty = new std::shared_ptr<Type>[1]{ to_IR_type<ImageViewEntry>() };
		return first(emplace_op(Node{ .kind = Node::GET_IV_META, .type = std::span{ ty, 1 }, .get_iv_meta = { .imageview = ptr } }));
	}
} // namespace vuk

namespace std {
	size_t hash<vuk::Ref>::operator()(vuk::Ref const& x) const noexcept {
		size_t h = 0;
		hash_combine(h, x.node, x.index);
		return h;
	}
} // namespace std
