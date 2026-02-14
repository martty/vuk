#pragma once

#include "vuk/ir/IRProcess.hpp"
#include "vuk/runtime/CommandBuffer.hpp"
#include "vuk/runtime/vk/Program.hpp"
#include <fmt/format.h>
#include <string>
#include <vector>

namespace vuk {
	struct IREvalContext {
		virtual void* allocate_host_memory(size_t size) = 0;

		template<class T>
		class temporary_alloc {
		public:
			using value_type = T;

			temporary_alloc(IREvalContext& ctx) noexcept : ctx_(&ctx) {}

			template<class U>
			temporary_alloc(const temporary_alloc<U>& other) noexcept : ctx_(other.ctx_) {}

			T* allocate(size_t n) {
				return static_cast<T*>(ctx_->allocate_host_memory(n * sizeof(T)));
			}

			void deallocate(T*, size_t) noexcept {
				// No-op: arena-style allocation, no individual deallocation
			}

			template<class U>
			bool operator==(const temporary_alloc<U>& other) const noexcept {
				return ctx_ == other.ctx_;
			}

			template<class U>
			bool operator!=(const temporary_alloc<U>& other) const noexcept {
				return ctx_ != other.ctx_;
			}

			template<class U>
			friend class temporary_alloc;

		private:
			IREvalContext* ctx_;
		};

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
				auto& sliced = *static_cast<ImageView<>*>(dst);
				if (axis == Node::NamedAxis::MIP) {
					sliced = sliced.mip_range(start, count);
				} else if (axis == Node::NamedAxis::LAYER) {
					sliced = sliced.layer_range(start, count);
				} else if (axis == Node::NamedAxis::X || axis == Node::NamedAxis::Y || axis == Node::NamedAxis::Z) {
					// X, Y, Z axes: modify offset and extent
					auto& ve = sliced.get_meta();
					if (axis == Node::NamedAxis::X) {
						sliced = sliced.subregion({ (int32_t)start, 0, 0 }, { (uint32_t)count, ve.extent.height, ve.extent.depth });
					} else if (axis == Node::NamedAxis::Y) {
						sliced = sliced.subregion({ 0, (int32_t)start, 0 }, { ve.extent.width, (uint32_t)count, ve.extent.depth });
					} else if (axis == Node::NamedAxis::Z) {
						sliced = sliced.subregion({ 0, 0, (int32_t)start }, { ve.extent.width, ve.extent.height, (uint32_t)count });
					}
				} else {
					assert(0);
				}
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
				case 1:
					return f(*reinterpret_cast<bool*>(args)...);
					break;
				case 8:
					return f(*reinterpret_cast<uint8_t*>(args)...);
					break;
				case 16:
					return f(*reinterpret_cast<uint16_t*>(args)...);
					break;
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
			case Type::FLOAT_TY: {
				switch (t->scalar.width) {
				case 32:
					return f(*reinterpret_cast<float*>(args)...);
					break;
				case 64:
					return f(*reinterpret_cast<double*>(args)...);
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
					    if constexpr (!std::is_same_v<decltype(a), bool>) {
						    auto c = a / b;
						    memcpy(result, &c, sizeof(c));
					    } else {
						    assert(false && "DIV operation not supported for non-arithmetic types");
					    }
				    },
				    a,
				    b);
			} break;
			case Node::BinOp::MOD: {
				eval_with_type(
				    t,
				    [&](auto a, auto b) {
					    if constexpr (!std::is_same_v<decltype(a), bool> && !std::is_floating_point_v<decltype(a)>) {
						    auto c = a % b;
						    memcpy(result, &c, sizeof(c));
					    } else {
						    assert(false && "MOD operation not supported for non-arithmetic types");
					    }
				    },
				    a,
				    b);
			} break;
			}
			return result;
		}

		void* eval_logical_binop(Node::LogicalOp op, const std::shared_ptr<Type>& t, void* a, void* b) {
			auto result = allocate_host_memory(sizeof(bool));
			switch (op) {
			case Node::LogicalOp::AND: {
				eval_with_type(
				    t,
				    [&](auto a, auto b) {
					    bool c = (a && b);
					    memcpy(result, &c, sizeof(c));
				    },
				    a,
				    b);
			} break;
			case Node::LogicalOp::OR: {
				eval_with_type(
				    t,
				    [&](auto a, auto b) {
					    bool c = (a || b);
					    memcpy(result, &c, sizeof(c));
				    },
				    a,
				    b);
			} break;
			case Node::LogicalOp::XOR: {
				eval_with_type(
				    t,
				    [&](auto a, auto b) {
					    bool c = ((a != 0) != (b != 0));
					    memcpy(result, &c, sizeof(c));
				    },
				    a,
				    b);
			} break;
			case Node::LogicalOp::EQ: {
				eval_with_type(
				    t,
				    [&](auto a, auto b) {
					    bool c = a == b;
					    memcpy(result, &c, sizeof(c));
				    },
				    a,
				    b);
			} break;
			case Node::LogicalOp::NE: {
				eval_with_type(
				    t,
				    [&](auto a, auto b) {
					    bool c = a != b;
					    memcpy(result, &c, sizeof(c));
				    },
				    a,
				    b);
			} break;
			case Node::LogicalOp::LT: {
				eval_with_type(
				    t,
				    [&](auto a, auto b) {
					    bool c = a < b;
					    memcpy(result, &c, sizeof(c));
				    },
				    a,
				    b);
			} break;
			case Node::LogicalOp::LE: {
				eval_with_type(
				    t,
				    [&](auto a, auto b) {
					    bool c = a <= b;
					    memcpy(result, &c, sizeof(c));
				    },
				    a,
				    b);
			} break;
			case Node::LogicalOp::GT: {
				eval_with_type(
				    t,
				    [&](auto a, auto b) {
					    bool c = a > b;
					    memcpy(result, &c, sizeof(c));
				    },
				    a,
				    b);
			} break;
			case Node::LogicalOp::GE: {
				eval_with_type(
				    t,
				    [&](auto a, auto b) {
					    bool c = a >= b;
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
				auto fn_type = ref.node->call.args[0].type();
				if (t->kind != Type::ALIASED_TY && fn_type->kind == Type::OPAQUE_FN_TY && fn_type->opaque_fn.execute_on == DomainFlagBits::eConstant) {
					auto results = execute_user_callback(ref.node->call.args, nullptr);
					if (results.holds_value()) {
						return { expected_value, (*results)[ref.index] };
					} else {
						return { expected_control, CannotBeConstantEvaluated{ ref } };
					}
				} else {
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
			case Node::LOGICAL_BINARY: {
				auto& logical_binary = ref.node->logical_binary;

				auto a_ = eval(logical_binary.a);
				if (!a_) {
					return a_;
				}
				auto& a = *a_;

				auto b_ = eval(logical_binary.b);
				if (!b_) {
					return b_;
				}
				auto& b = *b_;
				return { expected_value, eval_logical_binop(logical_binary.op, logical_binary.a.type(), a, b) };

			} break;
			case Node::SELECT: {
				auto& select = ref.node->select;

				// Evaluate condition as size_t
				auto cond = eval_as_size_t(select.condition);
				if (!cond) {
					return { expected_control, CannotBeConstantEvaluated{ ref } };
				}

				// If condition is non-zero, return a, otherwise return b
				if (*cond != 0) {
					return eval(select.a);
				} else {
					return eval(select.b);
				}
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
			assert(to_IR_type<T>() == ref.type());
			auto res = eval(ref);
			if (!res) {
				return res;
			}
			return { expected_value, *reinterpret_cast<T*>(*res) };
		}

		Result<size_t, CannotBeConstantEvaluated> eval_as_size_t(Ref ref) {
			auto base_ty = Type::stripped(ref.type());
			if (base_ty->kind == Type::INTEGER_TY) {
				auto res = eval(ref);
				if (!res) {
					return { expected_control, CannotBeConstantEvaluated{ ref } };
				}
				void* value = *res;
				switch (base_ty->scalar.width) {
				case 1:
					return { expected_value, static_cast<size_t>(*static_cast<bool*>(value)) };
				case 8:
					return { expected_value, static_cast<size_t>(*static_cast<uint8_t*>(value)) };
				case 16:
					return { expected_value, static_cast<size_t>(*static_cast<uint16_t*>(value)) };
				case 32:
					return { expected_value, static_cast<size_t>(*static_cast<uint32_t*>(value)) };
				case 64:
					return { expected_value, static_cast<size_t>(*static_cast<uint64_t*>(value)) };
				default:
					assert(0 && "Unsupported integer width");
					return { expected_control, CannotBeConstantEvaluated{ ref } };
				}
			}
			assert(0 && "Expected integer type");
			return { expected_control, CannotBeConstantEvaluated{ ref } };
		}

		Result<std::vector<void*, temporary_alloc<void*>>> execute_user_callback(std::span<Ref> call_args, CommandBuffer* cobuf) {
			auto fn_type = call_args[0].type();
			size_t first_parm = fn_type->kind == Type::OPAQUE_FN_TY ? 1 : 4;
			std::vector<void*, temporary_alloc<void*>> opaque_rets(temporary_alloc<void*>(*this));

			if (fn_type->kind == Type::OPAQUE_FN_TY) {
				std::vector<void*, temporary_alloc<void*>> opaque_args(temporary_alloc<void*>(*this));
				std::vector<void*, temporary_alloc<void*>> opaque_meta(temporary_alloc<void*>(*this));
				for (size_t i = first_parm; i < call_args.size(); i++) {
					auto& parm = call_args[i];
					auto parm_ = eval(parm);
					if (!parm_) {
						return parm_;
					}
					opaque_args.push_back(*parm_);
					opaque_meta.push_back(&parm);
				}
				opaque_rets.resize(fn_type->opaque_fn.return_types.size());

				// Allocate memory for unsynchronized return types
				for (size_t i = 0; i < fn_type->opaque_fn.return_types.size(); i++) {
					auto& ret_type = fn_type->opaque_fn.return_types[i];
					if (ret_type->kind != Type::ALIASED_TY) {
						opaque_rets[i] = allocate_host_memory(ret_type->size);
					}
				}

				(*fn_type->callback)(cobuf, opaque_args, opaque_meta, opaque_rets);
			} else if (fn_type->kind == Type::SHADER_FN_TY) {
				assert(cobuf);
				opaque_rets.resize(fn_type->shader_fn.return_types.size());
				auto pbi = reinterpret_cast<PipelineBaseInfo*>(fn_type->shader_fn.shader);

				cobuf->bind_compute_pipeline(pbi);

				auto& flat_bindings = pbi->reflection_info.flat_bindings;
				for (size_t i = first_parm; i < (first_parm + flat_bindings.size()); i++) {
					auto& parm = call_args[i];
					if (parm.type()->kind != Type::POINTER_TY) {
						auto binding_idx = i - first_parm;
						auto& [set, binding] = flat_bindings[binding_idx];
						auto val = get_value(parm);
						switch (binding->type) {
						case DescriptorType::eSampledImage:
						case DescriptorType::eStorageImage:
							cobuf->bind_image(set, binding->binding, *reinterpret_cast<ImageView<>*>(val));
							break;
						case DescriptorType::eUniformBuffer:
						case DescriptorType::eStorageBuffer: {
							auto& v = *reinterpret_cast<Buffer<>*>(val);
							cobuf->bind_buffer(set, binding->binding, v);
							break;
						}
						case DescriptorType::eSampler:
							cobuf->bind_sampler(set, binding->binding, *reinterpret_cast<SamplerCreateInfo*>(val));
							break;
						case DescriptorType::eCombinedImageSampler: {
							auto& si = *reinterpret_cast<SampledImage*>(val);
							cobuf->bind_image(set, binding->binding, si.ia);
							cobuf->bind_sampler(set, binding->binding, si.sci);
							break;
						}
						default:
							assert(0);
						}

						opaque_rets[binding_idx] = val;
					}
				}
				// remaining arguments as push constants
				size_t pc_offset = 0;
				for (size_t i = (first_parm + flat_bindings.size()); i < call_args.size(); i++) {
					auto& parm = call_args[i];
					cobuf->push_constants(ShaderStageFlagBits::eCompute, pc_offset, get_value(parm), parm.type()->size);
					pc_offset += parm.type()->size;
				}

				cobuf->dispatch(constant<uint32_t>(call_args[1]), constant<uint32_t>(call_args[2]), constant<uint32_t>(call_args[3]));
			} else {
				assert(0);
			}

			return { expected_value, opaque_rets };
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

		/// @brief Creates a new SLICE node with a different source, preserving other arguments from an existing SLICE node.
		/// @param slice_node The existing SLICE node to clone
		/// @param new_src The new source Ref to use in the cloned SLICE node
		/// @return A new Ref to the first output of the created SLICE node
		Ref make_slice_with_new_src(Node* slice_node, Ref new_src) {
			assert(slice_node->kind == Node::SLICE);

			auto& original_slice = slice_node->slice;
			auto stripped = Type::stripped(new_src.type());
			auto ty = new std::shared_ptr<Type>[3];

			// Copy the type information, updating based on new source
			if (slice_node->type.size() >= 1) {
				ty[0] = slice_node->type[0]; // result type
			}
			ty[1] = ty[2] = stripped; // rest and original types based on new source

			return first(current_module->emplace_op(
			    Node{ .kind = Node::SLICE,
			          .type = std::span{ ty, 3 },
			          .slice = { .src = new_src, .start = original_slice.start, .count = original_slice.count, .axis = original_slice.axis } }));
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

		/// @brief Walks back through the link chain to find the definition of a Ref.
		/// @param ref The Ref to find the definition for
		/// @return The Ref pointing to the definition
		static Ref to_def(Ref ref) {
			auto link = &ref.link();
			while (link->prev && link->prev->def) {
				link = link->prev;
			}
			return link->def;
		}

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

				if (value2 == needle) { // cycle
					// fmt::print("Skipping replace {} -> {} (cycle)\n", fmt::to_string(needle), fmt::to_string(value));
					return;
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

			// Handle held nodes by mutating them to LOGICAL_COPY
			// Remove those replaces since we've handled them via mutation
			auto replaces_end = std::remove_if(replaces.begin(), replaces.end(), [](const Replace& r) {
				if (r.needle.node->held) {
					// Mutate the held node to be a logical copy of the replacement value
					Node* held_node = r.needle.node;

					// Clean up existing node data
					if (held_node->generic_node.arg_count == (uint8_t)~0u) {
						delete[] held_node->variable_node.args.data();
					}

					// Convert to LOGICAL_COPY
					held_node->kind = Node::LOGICAL_COPY;
					held_node->fixed_node.arg_count = 1;
					held_node->logical_copy.src = r.value;

					// Return true to remove this replace from the list
					// (we've handled it by mutation, don't need to replace uses)
					return true;
				}
				return false;
			});

			replaces.erase(replaces_end, replaces.end());

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

		/// @brief Capture a snapshot of the current IR graph state for visualization
		/// @param label Optional label for this snapshot. If empty, auto-generates "Snapshot N"
		///
		/// Snapshots are collected only if RGCImpl::enable_html_graph_snapshots is true.
		/// Each snapshot is named hierarchically as "{PassName}/{label}" and can be viewed
		/// in the generated HTML file showing graph evolution through the pass.
		///
		/// Example usage:
		/// @code
		/// capture_snapshot("Initial State");
		/// // ... perform transformations ...
		/// capture_snapshot("After optimization");
		/// @endcode
		void capture_snapshot(std::string label = "");

		/// @brief Get the name of this pass (extracted from RTTI)
		/// @return Pass name as a string (e.g., "constant_folding")
		std::string get_pass_name() const;

	private:
		size_t snapshot_counter = 0;
		bool snapshots_initialized = false;
		std::string pass_name;
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