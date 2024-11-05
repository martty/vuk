#pragma once

#include "vuk/IR.hpp"
#include "vuk/RelSpan.hpp"
#include "vuk/ShortAlloc.hpp"
#include "vuk/SourceLocation.hpp"

#include <deque>
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
		std::span<void*> values;
	};

#define INIT(x) x(decltype(x)::allocator_type(*arena_))

	struct RGCImpl {
		RGCImpl() : arena_(new arena(4 * 1024 * 1024)) {}
		RGCImpl(arena* a) : arena_(a) {}
		std::unique_ptr<arena> arena_;
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

		std::unordered_map<Node*, std::vector<Ref>> deferred_splices; // Node: the node that needs to signal the splice, Ref: ref to a splice result
		std::unordered_map<Node*, size_t> pending_splice_sigs;        // Node: splice node, size_t: number of splice srcs that have been processed

		std::span<ScheduledItem*> transfer_passes, compute_passes, graphics_passes;

		template<class T>
		T& get_value(Ref parm) {
			return *reinterpret_cast<T*>(get_value(parm));
		};

		void* get_value(Ref parm) {
			if (parm.node->kind == Node::EXTRACT) {
				auto& composite = parm.node->extract.composite;
				auto index_v = get_value<uint64_t>(parm.node->extract.index);
				void* composite_v = get_value(parm.node->extract.composite);
				auto t = Type::stripped(composite.type());
				if (t->kind == Type::COMPOSITE_TY) {
					auto offset = t->offsets[index_v];
					return reinterpret_cast<std::byte*>(composite_v) + offset;
				} else if (t->kind == Type::ARRAY_TY) {
					return reinterpret_cast<std::byte*>(composite_v) + t->array.stride * index_v;
				} else {
					assert(0);
				}
			} else if (parm.node->kind == Node::ACQUIRE_NEXT_IMAGE) {
				Swapchain* swp = reinterpret_cast<Swapchain*>(get_value(parm.node->acquire_next_image.swapchain));
				return &swp->images[swp->image_index];
			} else {
				if (parm.node->execution_info) {
					return parm.node->execution_info->values[parm.index];
				} else {
					return *eval<void*>(parm);
				}
			}
			assert(0);
			return nullptr;
		}

		std::span<void*> get_values(Node* node) {
			assert(node->execution_info);
			return node->execution_info->values;
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
		Result<void> schedule_intra_queue(const RenderGraphCompileOptions& compile_options);
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

	struct MultiSubrange {
		static MultiSubrange all() {
			MultiSubrange msr;
			msr.ranges.resize(1);
			return msr;
		}

		static MultiSubrange none() {
			MultiSubrange msr;
			return msr;
		}

		MultiSubrange() = default;
		MultiSubrange(Subrange::Image r) {
			ranges.emplace_back(r);
		}

		MultiSubrange(std::pmr::vector<Subrange::Image> r) {
			if (r.size() == 1) {
				ranges = std::move(r);
				return;
			}
			for (auto i = 0; i < ((int)r.size()) - 1; i++) {
				for (auto j = i + 1; j < r.size(); j++) {
					auto& ri = r[i];
					auto& rj = r[j];
					if (auto isection = intersect_one(ri, rj)) {
						difference_one(ri, *isection, [&](Subrange::Image i) { ranges.emplace_back(i); });
					} else {
						ranges.push_back(ri);
					}
					ranges.push_back(rj);
				}
			}
		}

		MultiSubrange set_intersect(Subrange::Image b) {
			std::pmr::vector<Subrange::Image> new_ranges;
			for (auto& a : ranges) {
				if (auto i = intersect_one(a, b)) {
					new_ranges.emplace_back(*i);
				}
			}
			return MultiSubrange(std::move(new_ranges));
		}

		MultiSubrange set_difference(Subrange::Image b) {
			std::pmr::vector<Subrange::Image> new_ranges;
			for (auto& a : ranges) {
				difference_one(a, b, [&](Subrange::Image i) { new_ranges.emplace_back(i); });
			}
			return MultiSubrange(std::move(new_ranges));
		}

		MultiSubrange set_difference(MultiSubrange& b) {
			std::pmr::vector<Subrange::Image> new_ranges;
			for (auto& a : ranges) {
				for (auto& b : b.ranges) {
					difference_one(a, b, [&](Subrange::Image i) { new_ranges.emplace_back(i); });
				}
			}
			return MultiSubrange(std::move(new_ranges));
		}

		Subrange::Image& operator[](size_t index) {
			return ranges[index];
		}

		const Subrange::Image& operator[](size_t index) const {
			return ranges[index];
		}

		size_t size() const {
			return ranges.size();
		}

		explicit operator bool() {
			return ranges.size() > 0;
		}

	private:
		std::pmr::vector<Subrange::Image> ranges;
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

	namespace errors { /*
		RenderGraphException make_unattached_resource_exception(PassInfo& pass_info, Resource& resource);
		RenderGraphException make_cbuf_references_unknown_resource(PassInfo& pass_info, Resource::Type type, Name name);
		RenderGraphException make_cbuf_references_undeclared_resource(PassInfo& pass_info, Resource::Type type, Name name);*/
	} // namespace errors
}; // namespace vuk