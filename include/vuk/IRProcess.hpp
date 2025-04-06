#pragma once

#include "ResourceUse.hpp"
#include "vuk/IR.hpp"
#include "vuk/RelSpan.hpp"
#include "vuk/ShortAlloc.hpp"
#include "vuk/SourceLocation.hpp"

#include <deque>
#include <gch/small_vector.hpp>
#include <memory_resource>
#include <robin_hood.h>

namespace vuk {

	using DefUseMap = robin_hood::unordered_node_map<Ref, ChainLink>;

	struct ScheduledItem {
		Node* execable;
		DomainFlagBits scheduled_domain;
		Stream* scheduled_stream;

		bool ready = false; // for DYNAMO
	};

	struct ExecutionInfo {
		Stream* stream;
		size_t naming_index;
		Node::Kind kind;
	};

#define INIT(x) x(decltype(x)::allocator_type(*arena_))

	struct RGCImpl {
		RGCImpl() : arena_(new arena(4 * 1024 * 1024)), pool(std::make_unique<std::pmr::unsynchronized_pool_resource>()), mbr(pool.get()) {}
		RGCImpl(arena* a, std::unique_ptr<std::pmr::unsynchronized_pool_resource> pool) : arena_(a), pool(std::move(pool)), mbr(this->pool.get()) {}
		std::unique_ptr<arena> arena_;
		std::unique_ptr<std::pmr::unsynchronized_pool_resource> pool;
		std::pmr::monotonic_buffer_resource mbr;

		plf::colony<ScheduledItem> scheduled_execables;
		std::vector<ScheduledItem*> partitioned_execables;

		std::pmr::vector<Ref> pass_reads;

		std::vector<std::shared_ptr<ExtNode>> refs;
		std::vector<Node*> ref_nodes;
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

		template<class T>
		T& get_value(Ref parm) {
			return *reinterpret_cast<T*>(get_value(parm));
		};

		void* get_value(Ref parm) {
			if (parm.node->kind == Node::ACQUIRE_NEXT_IMAGE) {
				Swapchain* swp = *reinterpret_cast<Swapchain**>(get_value(parm.node->acquire_next_image.swapchain));
				return &swp->images[swp->image_index];
			} else {
				if (parm.node->kind == Node::ACQUIRE) {
					return parm.node->acquire.values[parm.index];
				} else {
					return *eval<void*>(parm);
				}
			}
			assert(0);
			return nullptr;
		}

		std::span<void*> get_values(Node* node) {
			assert(node->kind == Node::ACQUIRE);
			return node->acquire.values;
		}

		void process_node_links(IRModule* module,
		                        Node* node,
		                        std::pmr::vector<Ref>& pass_reads,
		                        std::pmr::vector<ChainLink*>& child_chains,
		                        std::pmr::vector<Node*>& new_nodes,
		                        std::pmr::polymorphic_allocator<std::byte> allocator,
		                        bool do_ssa);

		Result<void> build_nodes();
		Result<void> build_links(std::vector<Node*>& working_set, std::pmr::polymorphic_allocator<std::byte> allocator);
		template<class It>
		Result<void> build_links(IRModule* module,
		                         It start,
		                         It end,
		                         std::pmr::vector<Ref>& pass_reads,
		                         std::pmr::vector<ChainLink*>& child_chains,
		                         std::pmr::polymorphic_allocator<std::byte> allocator);
		Result<void> implicit_linking(IRModule* module, std::pmr::polymorphic_allocator<std::byte> allocator);
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

	// errors and printing
	enum class Level { eError };

	std::string format_source_location(Node* node);

	std::string parm_to_string(Ref parm);
	std::string print_args_to_string(std::span<Ref> args);
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