#pragma once

#include "vuk/ir/IRProcess.hpp"
#include <fmt/format.h>
#include <string>
#include <vector>

namespace vuk {
	struct IREvalContext {
		virtual void* allocate_host_memory(size_t size) = 0;

		Result<void*, CannotBeConstantEvaluated> evaluate_construct(Node* node) {
			// TODO : PAV : ImageAttachments?
			assert(node->type[0]->kind != Type::POINTER_TY);
			assert(node->type[0]->hash_value != current_module->types.builtin_swapchain);
			if (node->type[0]->kind == Type::ARRAY_TY) {
				auto array_size = node->type[0]->array.count;
				auto elem_ty = *node->type[0]->array.T;

				char* arr_mem = static_cast<char*>(allocate_host_memory(elem_ty->size * array_size));
				for (auto i = 0; i < array_size; i++) {
					auto& elem = node->construct.args[i + 1];
					assert(Type::stripped(elem.type())->hash_value == elem_ty->hash_value);

					auto v = eval(elem);
					if (!v) {
						return v;
					}
					memcpy(arr_mem + i * elem_ty->size, *v, elem_ty->size);
				}
				if (array_size == 0) { // zero-len arrays
					arr_mem = nullptr;
				}
				return { expected_value, arr_mem };
			} else if (node->type[0]->kind == Type::UNION_TY) {
				for (size_t i = 1; i < node->construct.args.size(); i++) {
					auto arg_ty = node->construct.args[i].type();
					auto& parm = node->construct.args[i];
				}

				char* arr_mem = static_cast<char*>(allocate_host_memory(node->type[0]->size));
				size_t offset = 0;
				for (auto i = 0; i < node->construct.args.size() - 1; i++) {
					auto sz = node->type[0]->composite.types[i]->size;
					auto& elem = node->construct.args[i + 1];

					auto v = eval(elem);
					if (!v) {
						return v;
					}
					memcpy(arr_mem + offset, *v, sz);
					offset += sz;
				}

				return { expected_value, arr_mem };
			} else {
				auto result_ty = node->type[0].get();
				// allocate type
				void* result = allocate_host_memory(result_ty->size);
				// loop args and resolve them
				std::vector<void*> argvals;
				for (size_t i = 1; i < node->construct.args.size(); i++) {
					auto& parm = node->construct.args[i];
					auto v = eval(parm);
					if (!v) {
						return v;
					}
					argvals.push_back(*v);
				}

				result_ty->composite.construct(result, argvals);
				return { expected_value, result };
			}
		}

		void evaluate_slice(Ref composite, uint8_t axis, uint64_t start, uint64_t count, void* composite_v, void* dst) {
			auto t = Type::stripped(composite.type());
			if (axis == Node::NamedAxis::FIELD) {
				assert(t->kind == Type::COMPOSITE_TY || t->kind == Type::UNION_TY);
				assert(count == 1);
				auto sliced = static_cast<std::byte*>(composite_v);
				auto offset = t->offsets[start];
				memcpy(dst, sliced + offset, t->composite.types[start]->size);
				return;
			}
			if (t->kind == Type::ARRAY_TY) {
				assert(axis == 0);
				assert(count == 1);
				auto sliced = static_cast<std::byte*>(composite_v);
				memcpy(dst, sliced + t->array.stride * start, (*t->array.T)->size);
				return;
			}
			memcpy(dst, composite_v, t->size);
			if (t->is_imageview()) {
				assert(false);
				/*/ if (axis == Node::NamedAxis::MIP) {
				  auto& sliced = *static_cast<ImageAttachment*>(dst);
				  sliced.base_level += start;
				  if (count != Range::REMAINING) {
				    sliced.level_count = count;
				  }
				} else if (axis == Node::NamedAxis::LAYER) {
				  auto& sliced = *static_cast<ImageAttachment*>(dst);
				  sliced.base_layer += start;
				  if (count != Range::REMAINING) {
				    sliced.layer_count = count;
				  }
				} else {
				  assert(0);
				}*/
			} else if (t->is_bufferlike_view()) {
				if (axis == 0) {
					auto& sliced = *static_cast<Buffer<>*>(dst);
					sliced.ptr += start;
					if (count != Range::REMAINING) {
						sliced.sz_bytes = count;
					}
				} else {
					assert(0);
				}
			} else {
				assert(0);
			}
		}

		template<class F, class... Args>
		auto eval_with_type(const std::shared_ptr<Type>& t, F&& f, Args... args) {
			switch (t->kind) {
			case Type::INTEGER_TY: {
				switch (t->scalar.width) {
				case 32:
					return f(*reinterpret_cast<uint32_t*>(args)...);
					break;
				case 64:
					return f(*reinterpret_cast<uint64_t*>(args)...);
					break;
				default:
					assert(0);
				}
				break;
			}
			default:
				break;
			}
		}

		void* eval_binop(Node::BinOp op, const std::shared_ptr<Type>& t, void* a, void* b) {
			auto result = allocate_host_memory(t->size);
			switch (op) {
			case Node::BinOp::ADD: {
				eval_with_type(
				    t,
				    [&](auto a, auto b) {
					    auto c = a + b;
					    memcpy(result, &c, sizeof(c));
				    },
				    a,
				    b);
			} break;
			case Node::BinOp::SUB: {
				eval_with_type(
				    t,
				    [&](auto a, auto b) {
					    auto c = a - b;
					    memcpy(result, &c, sizeof(c));
				    },
				    a,
				    b);
			} break;
			case Node::BinOp::MUL: {
				eval_with_type(
				    t,
				    [&](auto a, auto b) {
					    auto c = a * b;
					    memcpy(result, &c, sizeof(c));
				    },
				    a,
				    b);
			} break;
			case Node::BinOp::DIV: {
				eval_with_type(
				    t,
				    [&](auto a, auto b) {
					    auto c = a / b;
					    memcpy(result, &c, sizeof(c));
				    },
				    a,
				    b);
			} break;
			case Node::BinOp::MOD: {
				eval_with_type(
				    t,
				    [&](auto a, auto b) {
					    auto c = a % b;
					    memcpy(result, &c, sizeof(c));
				    },
				    a,
				    b);
			} break;
			}
			return result;
		}

		Result<void*, CannotBeConstantEvaluated> eval(Ref ref) {
			// can always operate on defs, values are ~immutable
			if (ref.node->links) {
				auto link = &ref.link();
				if (link->def) {
					while (link->prev && link->prev->def) {
						link = link->prev;
					}
					ref = link->def;
				}
			}

			switch (ref.node->kind) {
			case Node::CONSTANT: {
				return { expected_value, ref.node->constant.value };
			}
			case Node::CONSTRUCT: {
				return evaluate_construct(ref.node);
			}
			case Node::ACQUIRE_NEXT_IMAGE: {
				auto swp_ = eval(ref.node->acquire_next_image.swapchain);
				if (!swp_) {
					return swp_;
				}
				return swp_;
			}
			case Node::ACQUIRE:
				return { expected_value, ref.node->acquire.values[ref.index] };
			case Node::CALL: {
				auto t = ref.type();
				if (t->kind != Type::ALIASED_TY) {
					return { expected_control, CannotBeConstantEvaluated{ ref } };
				}
				return eval(ref.node->call.args[t->aliased.ref_idx]);
			}
			case Node::MATH_BINARY: {
				auto& math_binary = ref.node->math_binary;

				auto a_ = eval(math_binary.a);
				if (!a_) {
					return a_;
				}
				auto& a = *a_;

				auto b_ = eval(math_binary.b);
				if (!b_) {
					return b_;
				}
				auto& b = *b_;
				return { expected_value, eval_binop(math_binary.op, ref.type(), a, b) };

			} break;
			case Node::GET_CI: {
				auto src = ref.node->get_ci.src;
				auto src_ = eval(src);
				if (!src_) {
					return src_;
				}
				if (src.type()->is_imageview()) {
					auto& iv = *static_cast<ImageView<>*>(*src_);
					return { expected_value, &iv.get_ci() };
				} else if (src.type()->kind == Type::IMAGE_TY) {
					auto& i = *static_cast<Image<>*>(*src_);
					return { expected_value, &i.get_ci() };
				} else {
					assert(false);
				}
			} break;
			case Node::SLICE: {
				if (ref.index == 1) {
					return eval(ref.node->slice.src);
				}
				auto composite_ = eval(ref.node->slice.src);
				if (!composite_) {
					return composite_;
				}
				auto& composite = *composite_;
				auto start_ = eval(ref.node->slice.start);
				if (!start_) {
					return start_;
				}
				auto& start = *start_;
				auto index = *static_cast<uint64_t*>(start);
				auto count_ = eval(ref.node->slice.count);
				if (!count_) {
					return count_;
				}
				auto& count = *count_;
				auto countv = *static_cast<uint64_t*>(count);

				auto& slice = ref.node->slice;
				auto type = Type::stripped(ref.node->slice.src.type());

				auto retv = allocate_host_memory(ref.node->type[0]->size);
				evaluate_slice(ref.node->slice.src, slice.axis, index, countv, composite, retv);
				return { expected_value, std::move(retv) };
			}
			default:
				return { expected_control, CannotBeConstantEvaluated{ ref } };
			}
			assert(0);
		}

		template<class T>
		Result<T, CannotBeConstantEvaluated> eval(Ref ref) {
			auto res = eval(ref);
			if (!res) {
				return res;
			}
			return { expected_value, *reinterpret_cast<T*>(*res) };
		}
	};

	struct IRPass : IREvalContext {
		IRPass(RGCImpl& impl, Runtime& runtime, std::pmr::polymorphic_allocator<std::byte> allocator) : impl(impl), runtime(runtime), allocator(allocator) {}
		virtual ~IRPass() {}

		RGCImpl& impl;
		Runtime& runtime;
		std::pmr::polymorphic_allocator<std::byte> allocator;
		std::pmr::vector<Node*> new_nodes;

		virtual Result<void> operator()() = 0;

		virtual bool node_set_modified() {
			return true;
		}
		virtual bool node_connections_modified() {
			return true;
		}

		void* allocate_host_memory(size_t s) override {
			return allocator.allocate_bytes(s);
		}

		void allocate_node_links(Node* node) {
			size_t result_count = node->type.size();
			if (result_count > 0) {
				node->links = new (allocate_host_memory(sizeof(ChainLink) * result_count)) ChainLink[result_count];
			}
		}

		void add_node(Node* node) {
			if (!node->rel_acq) {
				node->rel_acq = new AcquireRelease;
			}
			allocate_node_links(node);
			process_node_links(node);
			new_nodes.push_back(node);
		}

		bool do_ssa;

		std::vector<std::string> debug_stack;

		void print_ctx();
		Ref walk_writes(Node* node, Ref parm);
		void add_write(Node* node, Ref& parm, size_t index);
		void add_read(Node* node, Ref& parm, size_t index, bool needs_ssa);
		void add_breaking_result(Node* node, size_t output_idx);
		void add_result(Node* node, size_t output_idx, Ref parm);
		void process_node_links(Node* node);

		/// @brief Visits all nodes in preorder (parent first) and applies the given function to each element.
		/// @param f A callable object to be applied to each element during the preorder traversal.
		template<class F>
		void visit_all_preorder(F&& f) {
			std::vector<Node*, short_alloc<Node*>> work_queue(*impl.arena_);
			for (auto& node : impl.ref_nodes) {
				if (node->flag == 0) {
					node->flag = 1;
					work_queue.push_back(node);
				}
			}

			while (!work_queue.empty()) {
				auto node = work_queue.back();
				f(node);
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
			}

			for (auto& node : impl.nodes) {
				node->flag = 0;
			}
		}

		/// @brief Visits all nodes in postorder (child first) and applies the given function to each element.
		/// @param f A callable object to be invoked for each element during postorder traversal.
		template<class F>
		void visit_all_postorder(F&& f) {
			std::vector<Node*, short_alloc<Node*>> work_queue(*impl.arena_);
			for (auto& node : impl.nodes) {
				node->flag = 0;
			}
			impl.nodes.clear();
			for (auto& node : impl.ref_nodes) {
				work_queue.push_back(node);
			}
			for (auto& node : impl.set_nodes) {
				work_queue.push_back(node);
			}

			while (!work_queue.empty()) {
				auto node = work_queue.back();

				auto sz = work_queue.size();
				auto count = node->generic_node.arg_count;
				if (count != (uint8_t)~0u) {
					for (int i = 0; i < count; i++) {
						auto arg = node->fixed_node.args[i].node;
						if (arg->flag == 0) {
							work_queue.push_back(arg);
						}
					}
				} else {
					for (int i = 0; i < node->variable_node.args.size(); i++) {
						auto arg = node->variable_node.args[i].node;
						if (arg->flag == 0) {
							work_queue.push_back(arg);
						}
					}
				}
				if (work_queue.size() == sz) { // leaf node or all children processed, process
					if (node->flag == 0) {
						node->flag = 2;
						f(node);
						impl.nodes.push_back(node);
					}
					work_queue.pop_back();
				}
			}

			for (auto& node : impl.nodes) {
				node->flag = 0;
			}
		}

		void for_each_use(Ref r, auto fn) {
			auto link = &r.node->links[r.index];
			while (link) {
				for (auto read : link->reads.to_span(impl.pass_reads)) {
					fn(read);
				}
				if (link->undef) {
					fn(link->undef);
				}
				link = link->next;
			}
		};

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
				auto it = std::upper_bound(
				    replaces.begin(), replaces.end(), Replace{ needle, value }, [](const Replace& a, const Replace& b) { return a.needle < b.needle; });
				replaces.insert(it, { needle, value2 });
				// fmt::print("{}\n", Replace{ needle, value2 });
			}
		};

		/// @brief Rewrites the render graph using the provided predicate.
		/// @tparam Pred The type of the predicate function or functor.
		/// @param pred A predicate to apply to each node in the render graph.
		/// @return Result<void> indicating success or failure of the rewrite operation.
		template<class Pred>
		Result<void> rewrite(Pred pred) {
			std::vector<Replace, short_alloc<Replace>> replaces(*impl.arena_);
			Replacer rr(replaces);

			// impl.nodes might grow during pred calls
			for (size_t i = 0; i < impl.nodes.size(); i++) {
				auto& node = impl.nodes[i];
				pred(node, rr);
				if (new_nodes.size() > 0) {
					impl.nodes.insert(impl.nodes.end(), new_nodes.begin(), new_nodes.end());
					new_nodes.clear();
				}
			}

			/* fmt::print("[");
			    for (auto& r : replaces) {
			      fmt::print("{}, ", r);
			    }
			    fmt::print("]\n");*/

			std::vector<Ref*, short_alloc<Ref*>> args(*impl.arena_);
			// collect all args
			for (auto node : impl.nodes) {
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
	};

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
} // namespace vuk