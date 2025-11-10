#pragma once

#include "vuk/ir/IR.hpp"
#include "vuk/RelSpan.hpp"
#include "vuk/ResourceUse.hpp"
#include "vuk/ShortAlloc.hpp"
#include "vuk/SourceLocation.hpp"

#include <deque>
#include <fmt/format.h>
#include <memory_resource>
#include <robin_hood.h>
#include <unordered_set>

namespace vuk {

	using DefUseMap = robin_hood::unordered_node_map<Ref, ChainLink>;

	enum class RW { eRead, eWrite };

#define INIT(x) x(decltype(x)::allocator_type(*arena_))

	struct IRPass;

	using ir_pass_factory = std::unique_ptr<IRPass> (*)(struct RGCImpl& impl, Runtime& runtime, std::pmr::polymorphic_allocator<std::byte> allocator);
	template<class T>
	ir_pass_factory make_ir_pass() {
		return +[](struct RGCImpl& impl, Runtime& runtime, std::pmr::polymorphic_allocator<std::byte> allocator) {
			return std::unique_ptr<IRPass>(new T(impl, runtime, allocator));
		};
	}

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

		std::vector<std::pair<Resolver::BufferWithOffsetAndSize, ChainLink*>> bufs;
		std::pmr::vector<Node*> new_nodes;

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

		Result<void> build_nodes();
		Result<void> build_links_implicit(Runtime& runtime, std::pmr::vector<Node*>& working_set, std::pmr::polymorphic_allocator<std::byte> allocator);
		Result<void> build_links(Runtime& runtime, std::vector<Node*>& working_set, std::pmr::polymorphic_allocator<std::byte> allocator);
		Result<void> implicit_linking(Allocator& alloc, IRModule* module, std::pmr::polymorphic_allocator<std::byte> allocator);
		Result<void> build_sync();
		Result<void> collect_chains();
		Result<void> linearize(Runtime& runtime, std::pmr::polymorphic_allocator<std::byte> allocator);

		ImageUsageFlags compute_usage(const ChainLink* head);

		ProfilingCallbacks callbacks;

		std::vector<ir_pass_factory> ir_passes;

		Result<void> run_passes(Runtime& runtime, std::pmr::polymorphic_allocator<std::byte> allocator);
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

	inline auto format_as(Ref f) {
		return std::string("\"") + fmt::to_string(fmt::ptr(f.node)) + "@" + fmt::to_string(f.index) + std::string("\"");
	}

	inline constexpr void access_to_usage(ImageUsageFlags& usage, Access acc) {
		if (acc & (eMemoryRW | eColorRW)) {
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
		if (acc & (eMemoryRW | eTransferWrite)) {
			usage |= ImageUsageFlagBits::eTransferDst;
		}
		if (acc & (eMemoryRW | eFragmentRW | eComputeRW | eRayTracingRW)) {
			usage |= ImageUsageFlagBits::eStorage;
		}
	};

	// errors and printing
	enum class Level { eError };

	std::string format_source_location(Node* node);
	std::string domain_to_string(DomainFlagBits domain);

	void parm_to_string(Ref parm, std::string& msg);
	void print_args_to_string(std::span<Ref> args, std::string& msg);
	void print_args(std::span<Ref> args);
	std::string print_args_to_string_with_arg_names(std::span<const std::string_view> arg_names, std::span<Ref> args);
	std::string node_to_string(Node* node);
	std::vector<std::string_view> arg_names(Type* t);

	std::string format_graph_message(Level level, Node* node, std::string err);
	std::string format_message(Level level, ScheduledItem& item, std::string err);

	namespace errors { /*
		RenderGraphException make_unattached_resource_exception(PassInfo& pass_info, Resource& resource);
		RenderGraphException make_cbuf_references_unknown_resource(PassInfo& pass_info, Resource::Type type, Name name);
		RenderGraphException make_cbuf_references_undeclared_resource(PassInfo& pass_info, Resource::Type type, Name name);*/
	} // namespace errors
}; // namespace vuk

#define VUK_ENABLE_ICE

#ifndef VUK_ENABLE_ICE
#define VUK_ICE(expression) (void)((!!(expression)) || assert(expression))
#else
#define VUK_ICE(expression) (void)((!!(expression)) || (GraphDumper::end_cluster(), GraphDumper::end_graph(), false) || (assert(expression), false))
#endif