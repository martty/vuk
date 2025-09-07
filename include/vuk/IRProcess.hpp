#pragma once

#include "ResourceUse.hpp"
#include "vuk/IR.hpp"
#include "vuk/RelSpan.hpp"
#include "vuk/ShortAlloc.hpp"
#include "vuk/SourceLocation.hpp"

#include <deque>
#include <memory_resource>
#include <robin_hood.h>
#include <unordered_set>

namespace vuk {

	using DefUseMap = robin_hood::unordered_node_map<Ref, ChainLink>;

	enum class RW { eRead, eWrite };

#define INIT(x) x(decltype(x)::allocator_type(*arena_))

	struct RGCImpl {
		RGCImpl() : arena_(new arena(4 * 1024 * 1024)), pool(std::make_unique<std::pmr::unsynchronized_pool_resource>()), mbr(pool.get()) {}
		RGCImpl(arena* a, std::unique_ptr<std::pmr::unsynchronized_pool_resource> pool) : arena_(a), pool(std::move(pool)), mbr(this->pool.get()) {}
		std::unique_ptr<arena> arena_;
		std::unique_ptr<std::pmr::unsynchronized_pool_resource> pool;
		std::pmr::monotonic_buffer_resource mbr;

		std::vector<ScheduledItem*> partitioned_execables;

		std::pmr::vector<Ref> pass_reads;

		std::vector<std::shared_ptr<ExtNode>> refs;
		std::vector<Node*> ref_nodes;
		std::vector<Node*> set_nodes;
		std::vector<std::shared_ptr<ExtNode>> depnodes;
		std::vector<Node*> nodes;
		std::vector<Node*> garbage_nodes;
		std::vector<ChainLink*> chains;
		std::pmr::vector<ChainLink*> child_chains;

		std::vector<std::pair<Buffer, ChainLink*>> bufs;

		std::span<ScheduledItem*> transfer_passes, compute_passes, graphics_passes;

		struct LiveRange {
			ChainLink* def_link;
			ChainLink* undef_link;
			void* last_value;
			AcquireRelease* acqrel;
			StreamResourceUse last_use;
		};

		std::unordered_map<ChainLink*, LiveRange> live_ranges;

		plf::colony<ScheduledItem> scheduled_execables;
		struct Sched {
			Node* node;
			bool ready;
		};
		std::deque<Sched> work_queue;
		robin_hood::unordered_flat_set<Node*> scheduled;
		std::unordered_set<Node*> expanded;
		std::vector<ScheduledItem*> item_list;

		size_t naming_index_counter = 0;
		void schedule_new(Node* node) {
			if (scheduled.contains(node)) {
				return;
			}
			assert(!expanded.contains(node)); // TODO: cycle detected
			assert(node);
			if (node->scheduled_item) { // we have scheduling info for this
				work_queue.emplace_front(Sched{ node, false });
			} else { // no info, just schedule it as-is
				auto it = scheduled_execables.emplace(ScheduledItem{ .execable = node });
				node->scheduled_item = &*it;
				work_queue.emplace_front(Sched{ node, false });
			}
		}

		// returns true if the item is ready
		bool process(Sched& item) {
			if (item.ready) {
				return true;
			} else {
				item.ready = true;
				work_queue.push_front(item); // requeue this item
				return false;
			}
		}

		void schedule_dependency(Ref parm, RW access) {
			if (parm.node->kind == Node::CONSTANT || parm.node->kind == Node::PLACEHOLDER) {
				return;
			}
			auto link = parm.link();

			if (access == RW::eWrite) { // synchronize against writing
				// we are going to write here, so schedule all reads or the def, if no read
				if (link.reads.size() > 0) {
					// all reads
					for (auto& r : link.reads.to_span(pass_reads)) {
						schedule_new(r.node);
					}
				} else {
					// just the def
					schedule_new(link.def.node);
				}
			} else { // just reading, so don't synchronize with reads -> just the def
				schedule_new(link.def.node);
			}
		}

		template<class T>
		T& get_value(Ref parm) {
			return *reinterpret_cast<T*>(get_value(parm));
		};

		void* get_value(Ref parm) {
			switch (parm.node->kind) {
			case Node::CONSTANT:
				return parm.node->constant.value;
			case Node::ACQUIRE:
				return parm.node->acquire.values[parm.index];
			default:
				assert(0);
				return nullptr;
			}
		}

		std::span<void*> get_values(Node* node) {
			assert(node->kind == Node::ACQUIRE);
			return node->acquire.values;
		}

		Result<void> build_nodes();
		Result<void> build_links(std::vector<Node*>& working_set, std::pmr::polymorphic_allocator<std::byte> allocator);
		template<class It>
		Result<void> build_links(IRModule* module,
		                         It start,
		                         It end,
		                         std::pmr::vector<Ref>& pass_reads,
		                         std::pmr::vector<ChainLink*>& child_chains,
		                         std::pmr::polymorphic_allocator<std::byte> allocator);
		Result<void> implicit_linking(Allocator& alloc, IRModule* module, std::pmr::polymorphic_allocator<std::byte> allocator);
		Result<void> build_sync();
		Result<void> reify_inference();
		Result<void> collect_chains();

		ImageUsageFlags compute_usage(const ChainLink* head);

		ProfilingCallbacks callbacks;
	};
#undef INIT

	template<class T, class A, class F>
	T* contains_if(std::vector<T, A>& v, F&& f) {
		auto it = std::find_if(v.begin(), v.end(), f);
		if (it != v.end())
			return &(*it);
		else
			return nullptr;
	}

	template<class T, class A, class F>
	T const* contains_if(const std::vector<T, A>& v, F&& f) {
		auto it = std::find_if(v.begin(), v.end(), f);
		if (it != v.end())
			return &(*it);
		else
			return nullptr;
	}

	template<class T, class A>
	T const* contains(const std::vector<T, A>& v, const T& f) {
		auto it = std::find(v.begin(), v.end(), f);
		if (it != v.end())
			return &(*it);
		else
			return nullptr;
	}

	template<typename Iterator, typename Compare>
	void topological_sort(Iterator begin, Iterator end, Compare cmp) {
		while (begin != end) {
			auto const new_begin = std::partition(begin, end, [&](auto const& a) { return std::none_of(begin, end, [&](auto const& b) { return cmp(b, a); }); });
			assert(new_begin != begin && "not a partial ordering");
			begin = new_begin;
		}
	}

	template<class F>
	auto apply_generic_args(F&& f, vuk::Node* node) {
		auto count = node->generic_node.arg_count;
		if (count != (uint8_t)~0u) {
			for (int i = 0; i < count; i++) {
				f(node->fixed_node.args[i]);
			}
		} else {
			for (int i = 0; i < node->variable_node.args.size(); i++) {
				f(node->variable_node.args[i]);
			}
		}
	}

	inline std::optional<Subrange::Image> intersect_one(Subrange::Image a, Subrange::Image b) {
		Subrange::Image result;
		int64_t count;

		result.base_layer = std::max(a.base_layer, b.base_layer);
		int64_t end_layer = std::min(a.base_layer + (int64_t)a.layer_count, b.base_layer + (int64_t)b.layer_count);
		count = end_layer - result.base_layer;

		if (count < 1) {
			return {};
		}
		result.layer_count = static_cast<uint32_t>(count);

		result.base_level = std::max(a.base_level, b.base_level);
		int64_t end_level = std::min(a.base_level + (int64_t)a.level_count, b.base_level + (int64_t)b.level_count);
		count = end_level - result.base_level;

		if (count < 1) {
			return {};
		}
		result.level_count = static_cast<uint32_t>(count);

		return result;
	}

	inline std::optional<Subrange::Buffer> intersect_one(Subrange::Buffer a, Subrange::Buffer b) {
		Subrange::Buffer result;
		int64_t size;

		result.offset = std::max(a.offset, b.offset);
		int64_t end = std::min(a.offset + (int64_t)a.size, b.offset + (int64_t)b.size);
		size = end - result.offset;

		if (size < 1) {
			return {};
		}
		result.size = static_cast<uint32_t>(size);

		return result;
	}

	template<class F>
	void difference_one(Subrange::Image a, Subrange::Image isection, F&& func) {
		if (!intersect_one(a, isection)) {
			func(a);
			return;
		}
		// before, mips
		if (isection.base_level > a.base_level) {
			func({ .base_level = a.base_level, .level_count = isection.base_level - a.base_level, .base_layer = a.base_layer, .layer_count = a.layer_count });
		}
		// after, mips
		if (a.base_level + (int64_t)a.level_count > isection.base_level + (int64_t)isection.level_count) {
			func({ .base_level = isection.base_level + isection.level_count,
			       .level_count = a.level_count == VK_REMAINING_MIP_LEVELS ? VK_REMAINING_MIP_LEVELS
			                                                               : a.base_level + a.level_count - (isection.base_level + isection.level_count),
			       .base_layer = a.base_layer,
			       .layer_count = a.layer_count });
		}
		// before, layers
		if (isection.base_layer > a.base_layer) {
			func({ .base_level = a.base_level, .level_count = a.level_count, .base_layer = a.base_layer, .layer_count = isection.base_layer - a.base_layer });
		}
		// after, layers
		if (a.base_layer + (int64_t)a.layer_count > isection.base_layer + (int64_t)isection.layer_count) {
			func({
			    .base_level = a.base_level,
			    .level_count = a.level_count,
			    .base_layer = isection.base_layer + isection.layer_count,
			    .layer_count = a.layer_count == VK_REMAINING_ARRAY_LAYERS ? VK_REMAINING_ARRAY_LAYERS
			                                                              : a.base_layer + a.layer_count - (isection.base_layer + isection.layer_count),
			});
		}
	};

	template<class F>
	void difference_one(Subrange::Buffer a, Subrange::Buffer isection, F&& func) {
		if (!intersect_one(a, isection)) {
			func(a);
			return;
		}
		// before
		if (isection.offset > a.offset) {
			func({ .offset = a.offset, .size = isection.offset - a.offset });
		}
		// after
		if (a.offset + (int64_t)a.size > isection.offset + (int64_t)isection.size) {
			func({ .offset = isection.offset + isection.size,
			       .size = a.size == VK_REMAINING_MIP_LEVELS ? VK_REMAINING_MIP_LEVELS : a.offset + a.size - (isection.offset + isection.size) });
		}
	};
	struct Cut {
		uint8_t axis;
		Range range;

		constexpr bool shrinks(Cut& other) const noexcept {
			return axis == other.axis && range <= other.range;
		}

		constexpr bool intersects(Cut& other) const noexcept {
			bool same_axis = axis == other.axis;
			if (range.count == Range::REMAINING) {
				return same_axis && range.offset <= other.range.offset;
			}
			if (other.range.count == Range::REMAINING) {
				return same_axis && other.range.offset <= range.offset;
			}
			bool a_in_b = range.offset < other.range.offset && range.offset + range.count > other.range.offset;
			bool b_in_a = other.range.offset < range.offset && other.range.offset + other.range.count > range.offset;
			return same_axis && (a_in_b || b_in_a);
		}
	};

	inline Result<RefOrValue, CannotBeConstantEvaluated> eval(Ref ref);

	inline void set_if_available(void* dst, Ref& ref) {
		auto res = eval(ref);
		if (res.holds_value() && !res->is_ref) {
			if (ref.type()->size == 8) {
				uint64_t t;
				memcpy(&t, res->value, 8);
				if (t == ~(0ULL)) {
					return;
				}
			} else if (ref.type()->size == 4) {
				uint32_t t;
				memcpy(&t, res->value, 4);
				if (t == ~(0u)) {
					return;
				}
			} else if (ref.type()->size == 2) {
				uint16_t t;
				memcpy(&t, res->value, 2);
				if (t == USHRT_MAX) {
					return;
				}
			} else if (ref.type()->size == 1) {
				uint8_t t;
				memcpy(&t, res->value, 1);
				if (t == UCHAR_MAX) {
					return;
				}
			} else {
				assert(0);
			}
			memcpy(dst, res->value, ref.type()->size);
			if (ref.node->kind != Node::CONSTANT) {
				ref = current_module->make_constant(ref.type(), res->value);
			}
		}
	}

	inline void evaluate_slice(Ref composite, uint8_t axis, uint64_t start, uint64_t count, void* composite_v, void* dst) {
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
		if (t->hash_value == current_module->types.builtin_image) {
			if (axis == Node::NamedAxis::MIP) {
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
			}
		} else if (t->hash_value == current_module->types.builtin_buffer) {
			if (axis == 0) {
				auto& sliced = *static_cast<Buffer*>(dst);
				sliced.offset += start;
				if (count != Range::REMAINING) {
					sliced.size = count;
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
			switch (t->integer.width) {
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

	inline void* eval_binop(Node::BinOp op, const std::shared_ptr<Type>& t, void* a, void* b) {
		auto result = (void*)new char[t->size];
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

	inline Result<RefOrValue, CannotBeConstantEvaluated> eval(Ref ref) {
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
			return { expected_value, RefOrValue::from_value(ref.node->constant.value) };
		}
		case Node::CONSTRUCT: {
			if (ref.type()->hash_value == current_module->types.builtin_buffer) {
				auto& bound = constant<Buffer>(ref.node->construct.args[0]);
				set_if_available(&bound.size, ref.node->construct.args[1]);
				return { expected_value, RefOrValue::from_value(&bound, ref) };
			} else if (ref.type()->hash_value == current_module->types.builtin_image) {
				auto& attachment = constant<ImageAttachment>(ref.node->construct.args[0]);
				set_if_available(&attachment.extent.width, ref.node->construct.args[1]);
				set_if_available(&attachment.extent.height, ref.node->construct.args[2]);
				set_if_available(&attachment.extent.depth, ref.node->construct.args[3]);
				set_if_available(&attachment.format, ref.node->construct.args[4]);
				set_if_available(&attachment.sample_count, ref.node->construct.args[5]);
				set_if_available(&attachment.base_layer, ref.node->construct.args[6]);
				set_if_available(&attachment.layer_count, ref.node->construct.args[7]);
				set_if_available(&attachment.base_level, ref.node->construct.args[8]);
				set_if_available(&attachment.level_count, ref.node->construct.args[9]);
				return { expected_value, RefOrValue::from_value(&attachment, ref) };
			} else {
				return { expected_value, RefOrValue::from_ref(ref) };
			}
		}
		case Node::ACQUIRE_NEXT_IMAGE: {
			auto swp_ = eval(ref.node->acquire_next_image.swapchain);
			if (!swp_) {
				return swp_;
			}
			auto& swp = *swp_;
			assert(swp.is_ref && swp.ref.node->kind == Node::CONSTRUCT);
			auto arr = swp.ref.node->construct.args[1]; // array of images
			assert(arr.node->kind == Node::CONSTRUCT);
			auto elem = arr.node->construct.args[1]; // first image
			return eval(elem);
		}
		case Node::ACQUIRE:
			return { expected_value, RefOrValue::from_value(ref.node->acquire.values[ref.index]) };
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
			auto& a_rov = *a_;
			if (a_rov.is_ref) {
				return { expected_control, CannotBeConstantEvaluated{ ref } };
			}
			auto a = a_rov.value;

			auto b_ = eval(math_binary.b);
			if (!b_) {
				return b_;
			}
			auto& b_rov = *b_;
			if (b_rov.is_ref) {
				return { expected_control, CannotBeConstantEvaluated{ ref } };
			}
			auto b = b_rov.value;
			return { expected_value, RefOrValue::adopt_value(eval_binop(math_binary.op, ref.type(), a, b)) };

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
			if (start.is_ref) {
				return { expected_control, CannotBeConstantEvaluated{ ref } };
			}
			auto index = *static_cast<uint64_t*>(start.value);
			auto count_ = eval(ref.node->slice.count);
			if (!count_) {
				return count_;
			}
			auto& count = *count_;
			if (count.is_ref) {
				return { expected_control, CannotBeConstantEvaluated{ ref } };
			}
			auto countv = *static_cast<uint64_t*>(count.value);

			auto& slice = ref.node->slice;
			auto type = Type::stripped(ref.node->slice.src.type());

			if (composite.is_ref) {
				if (slice.axis == Node::NamedAxis::FIELD) {
					if (composite.ref.node->kind == Node::CONSTRUCT) {
						return eval(composite.ref.node->construct.args[index + 1]);
					} else {
						return { expected_control, CannotBeConstantEvaluated{ ref } };
					}
				} else {
					if (composite.ref.node->kind == Node::CONSTRUCT && type->kind == Type::ARRAY_TY) {
						return eval(composite.ref.node->construct.args[index + 1]);
					}
				}
			} else {
				auto retv = RefOrValue::adopt_value(new char[ref.node->type[0]->size], composite.ref);
				evaluate_slice(ref.node->slice.src, slice.axis, index, countv, composite.value, retv.value);
				return { expected_value, std::move(retv) };
			}

			return { expected_control, CannotBeConstantEvaluated{ ref } };
		}
		default:
			return { expected_control, CannotBeConstantEvaluated{ ref } };
		}
		assert(0);
	}

	// errors and printing
	enum class Level { eError };

	std::string format_source_location(Node* node);

	void parm_to_string(Ref parm, std::string& msg);
	void print_args_to_string(std::span<Ref> args, std::string& msg);
	void print_args(std::span<Ref> args);
	std::string print_args_to_string_with_arg_names(std::span<const std::string_view> arg_names, std::span<Ref> args);
	std::string node_to_string(Node* node);
	std::vector<std::string_view> arg_names(Type* t);

	std::string format_graph_message(Level level, Node* node, std::string err);

	uint64_t value_identity(Type* base_ty, void* value);

	namespace errors { /*
		RenderGraphException make_unattached_resource_exception(PassInfo& pass_info, Resource& resource);
		RenderGraphException make_cbuf_references_unknown_resource(PassInfo& pass_info, Resource::Type type, Name name);
		RenderGraphException make_cbuf_references_undeclared_resource(PassInfo& pass_info, Resource::Type type, Name name);*/
	} // namespace errors
}; // namespace vuk