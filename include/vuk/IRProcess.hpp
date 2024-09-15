#pragma once

#include "vuk/IR.hpp"
#include "vuk/RelSpan.hpp"
#include "vuk/ShortAlloc.hpp"
#include "vuk/SourceLocation.hpp"

#include <deque>
#include <robin_hood.h>
#include <memory_resource>

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
		std::vector<std::shared_ptr<ExtNode>> depnodes;
		std::vector<Node*> nodes;
		std::pmr::vector<Node*> all_nodes;
		std::vector<Node*> garbage_nodes;
		std::vector<ChainLink*> chains;
		std::pmr::vector<ChainLink*> child_chains;

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
				if (composite.type()->kind == Type::COMPOSITE_TY) {
					auto offset = composite.type()->composite.offsets[index_v];
					return reinterpret_cast<std::byte*>(composite_v) + offset;
				} else if (composite.type()->kind == Type::ARRAY_TY) {
					return reinterpret_cast<std::byte*>(composite_v) + parm.node->extract.composite.type()->array.stride * index_v;
				}
			} else if (parm.node->kind == Node::ACQUIRE_NEXT_IMAGE) {
				Swapchain* swp = reinterpret_cast<Swapchain*>(get_value(parm.node->acquire_next_image.swapchain));
				return &swp->images[swp->image_index];
			} else {
				if (parm.node->execution_info) {
					return parm.node->execution_info->values[parm.index];
				} else {
					return eval<void*>(parm);
				}
			}
		}

		std::span<void*> get_values(Node* node) {
			assert(node->execution_info);
			return node->execution_info->values;
		}

		Result<void> build_nodes();
		Result<void> build_links(const std::vector<Node*>& working_set, std::pmr::polymorphic_allocator<std::byte> allocator);
		Result<void> build_sync();
		Result<void> reify_inference();
		Result<void> schedule_intra_queue(const RenderGraphCompileOptions& compile_options);
		Result<void> collect_chains();

		ImageUsageFlags compute_usage(const ChainLink* head);

		void dump_graph();

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

	namespace errors { /*
		RenderGraphException make_unattached_resource_exception(PassInfo& pass_info, Resource& resource);
		RenderGraphException make_cbuf_references_unknown_resource(PassInfo& pass_info, Resource::Type type, Name name);
		RenderGraphException make_cbuf_references_undeclared_resource(PassInfo& pass_info, Resource::Type type, Name name);*/
	}                  // namespace errors
};                   // namespace vuk