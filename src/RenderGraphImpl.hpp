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
		RelSpan<AttachmentRPInfo> attachments;
		vuk::RenderPassCreateInfo rpci;
		vuk::FramebufferCreateInfo fbci;
		VkRenderPass handle = {};
		VkFramebuffer framebuffer;
	};

	using IARule = std::function<void(const struct InferenceContext&, ImageAttachment&)>;
	using BufferRule = std::function<void(const struct InferenceContext&, Buffer&)>;

	struct IAInference {
		QualifiedName resource;
		IARule rule;
	};

	struct IAInferences {
		Name prefix;
		std::vector<IARule> rules;
	};

	struct BufferInference {
		QualifiedName resource;
		BufferRule rule;
	};

	struct BufferInferences {
		Name prefix;
		std::vector<BufferRule> rules;
	};

	struct Release {
		Access original;
		QueueResourceUse dst_use;
		FutureBase* signal = nullptr;
	};

#define INIT(x) x(decltype(x)::allocator_type(*arena_))

	using DefUseMap = std::unordered_map<Ref, ChainLink>;

	struct ScheduledItem {
		Node* execable;
		DomainFlags scheduled_domain;

		int32_t is_waited_on = 0;
		size_t batch_index;
		size_t command_buffer_index = 0;
		int32_t render_pass_index = -1;
		uint32_t subpass;
		
		RelSpan<VkImageMemoryBarrier2KHR> pre_image_barriers, post_image_barriers;
		RelSpan<VkMemoryBarrier2KHR> pre_memory_barriers, post_memory_barriers;
		RelSpan<std::pair<DomainFlagBits, uint64_t>> relative_waits;
		RelSpan<std::pair<DomainFlagBits, uint64_t>> absolute_waits;
		RelSpan<FutureBase*> future_signals;
		RelSpan<int32_t> referenced_swapchains; // TODO: maybe not the best place for it
	};

	struct RGCImpl {
		RGCImpl() : arena_(new arena(4 * 1024 * 1024)), INIT(rpis) {}
		RGCImpl(arena* a) : arena_(a), INIT(rpis) {}
		std::unique_ptr<arena> arena_;

		std::vector<std::pair<DomainFlagBits, uint64_t>> waits;
		std::vector<std::pair<DomainFlagBits, uint64_t>> absolute_waits;
		std::vector<FutureBase*> future_signals;

		std::vector<ScheduledItem> scheduled_execables;
		std::vector<ScheduledItem*> partitioned_execables;
		std::vector<size_t> scheduled_idx_to_partitioned_idx;

		std::vector<VkImageMemoryBarrier2KHR> image_barriers;
		std::vector<VkMemoryBarrier2KHR> mem_barriers;

		DefUseMap res_to_links;
		std::vector<Node*> pass_reads;

		std::vector<ChainLink*> chains;
		std::vector<ChainLink*> child_chains;
		std::deque<ChainLink> helper_links;
		std::vector<int32_t> swapchain_references;
		std::vector<AttachmentRPInfo> rp_infos;
		/* std::array<size_t, 3> last_ordered_pass_idx_in_domain_array;
		
		 int32_t last_ordered_pass_idx_in_domain(DomainFlagBits queue) {
			uint32_t idx;
			if (queue == DomainFlagBits::eGraphicsQueue) {
				idx = 0;
			} else if (queue == DomainFlagBits::eComputeQueue) {
				idx = 1;
			} else {
				idx = 2;
			}
			return (int32_t)last_ordered_pass_idx_in_domain_array[idx];
		}*/

		std::vector<ChainLink*> attachment_use_chain_references;
		std::vector<RenderPassInfo*> attachment_rp_references;

		std::unordered_map<QualifiedName, IAInferences> ia_inference_rules;
		std::unordered_map<QualifiedName, BufferInferences> buf_inference_rules;

		std::vector<RenderPassInfo, short_alloc<RenderPassInfo, 64>> rpis;
		std::span<ScheduledItem*> transfer_passes, compute_passes, graphics_passes;

		void merge_diverge_passes(std::vector<PassInfo, short_alloc<PassInfo, 64>>& passes);

		Result<void> diagnose_unheaded_chains();
		Result<void> schedule_intra_queue(std::span<std::shared_ptr<RG>> rgs, const RenderGraphCompileOptions& compile_options);

		std::vector<ChainLink*> div_subchains;
		std::vector<ChainLink**> conv_subchains;
		Result<void> relink_subchains();
		Result<void> fix_subchains();

		void emit_image_barrier(RelSpan<VkImageMemoryBarrier2KHR>&,
		                        int32_t bound_attachment,
		                        QueueResourceUse last_use,
		                        QueueResourceUse current_use,
		                        Subrange::Image& subrange,
		                        ImageAspectFlags aspect,
		                        bool is_release = false);
		void emit_memory_barrier(RelSpan<VkMemoryBarrier2KHR>&, QueueResourceUse last_use, QueueResourceUse current_use);

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
	} // namespace errors
};  // namespace vuk