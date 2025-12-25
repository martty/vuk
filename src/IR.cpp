#include "vuk/RenderGraph.hpp"
#include <fmt/format.h>
#include <unordered_set>
#include <vector>
#include <vuk/ir/IR.hpp>
#include <vuk/ir/IRPass.hpp>
#include <vuk/Value.hpp>

namespace vuk {
	thread_local std::shared_ptr<IRModule> current_module = std::make_shared<IRModule>();

	struct AllocaCtx : IREvalContext {
		std::vector<void*> allocated;

		void* allocate_host_memory(size_t size) override {
			void* ptr = malloc(size);
			allocated.push_back(ptr);
			return ptr;
		}

		~AllocaCtx() {
			for (auto ptr : allocated) {
				free(ptr);
			}
		}
	};

	Result<void*, CannotBeConstantEvaluated> eval(Ref ref) {
		AllocaCtx ctx;
		return ctx.eval(ref);
	}

	void UntypedValue::dump_ir() const {
		std::vector<Node*> nodes;
		std::unordered_set<Node*> visited;

		// Start from the head node
		auto head = get_head();
		std::vector<Node*> work_queue;
		work_queue.push_back(head.node);

		// Traverse the IR graph reachable from this value
		while (!work_queue.empty()) {
			auto node = work_queue.back();
			work_queue.pop_back();

			if (visited.find(node) != visited.end()) {
				continue;
			}
			visited.insert(node);
			nodes.push_back(node);

			apply_generic_args(
			    [&](Ref arg) {
				    if (visited.find(arg.node) == visited.end()) {
					    work_queue.push_back(arg.node);
				    }
			    },
			    node);
		}

		// Sort nodes topologically so dependencies come before uses
		topological_sort(nodes.begin(), nodes.end(), [](Node* a, Node* b) {
			bool b_depends_on_a = false;
			apply_generic_args(
			    [&](Ref arg) {
				    if (arg.node == a) {
					    b_depends_on_a = true;
				    }
			    },
			    b);
			return b_depends_on_a;
		});

		// Dump the collected nodes in order (head node will be last)
		fmt::println("IR dump for value (reachable nodes: {}):", nodes.size());
		for (auto node : nodes) {
			fmt::print("  [{:#x}] {} -> ", (uintptr_t)node, Node::kind_to_sv(node->kind));

			// Print result types
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

			// Print args using apply_generic_args
			bool first = true;
			apply_generic_args(
			    [&](Ref arg) {
				    if (first) {
					    fmt::print(" <- ");
					    first = false;
				    } else {
					    fmt::print(", ");
				    }
				    fmt::print("[{:#x}:{}]", (uintptr_t)arg.node, arg.index);
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
