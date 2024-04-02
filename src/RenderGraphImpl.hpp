#pragma once

#include "RenderGraphUtil.hpp"
#include "RenderPass.hpp"
#include "vuk/RelSpan.hpp"
#include "vuk/RenderGraphReflection.hpp"
#include "vuk/ShortAlloc.hpp"
#include "vuk/SourceLocation.hpp"

#include <deque>
#include <robin_hood.h>

namespace vuk {
	struct RenderPassInfo {
		std::vector<VkImageView> framebuffer_ivs;
		RenderPassCreateInfo rpci;
		FramebufferCreateInfo fbci;
		VkRenderPass handle = {};
		VkFramebuffer framebuffer;
	};

#define INIT(x) x(decltype(x)::allocator_type(*arena_))

	using DefUseMap = robin_hood::unordered_node_map<Ref, ChainLink>;

	struct Stream {
		Stream(Allocator alloc, Executor* executor) : alloc(alloc), executor(executor) {}
		virtual ~Stream() {}
		Allocator alloc;
		Executor* executor = nullptr;
		DomainFlagBits domain;
		std::vector<Stream*> dependencies;
		std::vector<Signal*> dependent_signals;

		virtual void add_dependency(Stream* dep) = 0;
		virtual void sync_deps() = 0;

		virtual Signal* make_signal() = 0;

		void add_dependent_signal(Signal* signal) {
			dependent_signals.push_back(signal);
		}

		virtual void synch_image(ImageAttachment& img_att, StreamResourceUse src_use, StreamResourceUse dst_use, void* tag) = 0;
		virtual void synch_memory(StreamResourceUse src_use, StreamResourceUse dst_use, void* tag) = 0;

		struct SubmitResult {
			VkSemaphore sema_wait;
		};

		virtual Result<SubmitResult> submit() = 0;
	};

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

	struct RGCImpl {
		RGCImpl() : arena_(new arena(4 * 1024 * 1024)), INIT(rpis) {}
		RGCImpl(arena* a) : arena_(a), INIT(rpis) {}
		std::unique_ptr<arena> arena_;

		std::vector<std::pair<DomainFlagBits, uint64_t>> waits;
		std::vector<std::pair<DomainFlagBits, uint64_t>> absolute_waits;

		std::vector<ScheduledItem> scheduled_execables;
		std::vector<ScheduledItem*> partitioned_execables;
		std::vector<size_t> scheduled_idx_to_partitioned_idx;

		std::vector<VkImageMemoryBarrier2KHR> image_barriers;
		std::vector<VkMemoryBarrier2KHR> mem_barriers;

		std::vector<Ref> pass_reads;

		std::shared_ptr<RG> cg_module;
		robin_hood::unordered_flat_map<uint32_t, Type*> type_map;
		std::vector<std::shared_ptr<ExtNode>> refs;
		std::vector<Node*> nodes;
		std::vector<ChainLink*> chains;
		std::vector<ChainLink*> child_chains;
		std::deque<ChainLink> helper_links;
		std::vector<int32_t> swapchain_references;

		std::vector<ChainLink*> attachment_use_chain_references;
		std::vector<RenderPassInfo*> attachment_rp_references;

		std::vector<RenderPassInfo, short_alloc<RenderPassInfo>> rpis;
		std::span<ScheduledItem*> transfer_passes, compute_passes, graphics_passes;

		std::unordered_map<Node*, ExecutionInfo> executed;

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
				auto it = executed.find(parm.node);
				if (it != executed.end()) {
					return it->second.values[parm.index];
				} else {
					return eval<void*>(parm);
				}
			}
		}

		std::span<void*> get_values(Node* node) {
			auto it = executed.find(node);
			assert(it != executed.end());
			return it->second.values;
		}

		Result<void> build_nodes();
		Result<void> build_links();
		Result<void> reify_inference();
		Result<void> schedule_intra_queue(const RenderGraphCompileOptions& compile_options);
		Result<void> collect_chains();

		std::vector<ChainLink*> div_subchains;
		std::vector<ChainLink**> conv_subchains;
		Result<void> relink_subchains();
		Result<void> fix_subchains();

		// opt passes
		Result<void> merge_rps();

		// link passes
		Result<void> generate_barriers_and_waits();
		Result<void> assign_passes_to_batches();
		Result<void> build_waits();
		Result<void> build_renderpasses();

		void emit_barriers(Context& ctx,
		                   VkCommandBuffer cbuf,
		                   vuk::DomainFlagBits domain,
		                   RelSpan<VkMemoryBarrier2KHR> mem_bars,
		                   RelSpan<VkImageMemoryBarrier2KHR> im_bars);

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

	namespace errors { /*
		RenderGraphException make_unattached_resource_exception(PassInfo& pass_info, Resource& resource);
		RenderGraphException make_cbuf_references_unknown_resource(PassInfo& pass_info, Resource::Type type, Name name);
		RenderGraphException make_cbuf_references_undeclared_resource(PassInfo& pass_info, Resource::Type type, Name name);*/
	}                  // namespace errors
};                   // namespace vuk