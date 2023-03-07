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

	struct PassWrapper {
		Name name;
		DomainFlags execute_on = DomainFlagBits::eAny;

		RelSpan<Resource> resources;

		std::function<void(CommandBuffer&)> execute;
		void* (*make_argument_tuple)(CommandBuffer&, std::span<void*>);

		std::byte* arguments; // internal use
		PassType type;
		source_location source;
	};

	struct PassInfo {
		PassInfo(PassWrapper&);

		PassWrapper* pass;

		QualifiedName qualified_name;

		size_t batch_index;
		size_t command_buffer_index = 0;
		int32_t render_pass_index = -1;
		uint32_t subpass;
		DomainFlags domain = DomainFlagBits::eAny;

		RelSpan<Resource> resources;

		RelSpan<VkImageMemoryBarrier2KHR> pre_image_barriers, post_image_barriers;
		RelSpan<VkMemoryBarrier2KHR> pre_memory_barriers, post_memory_barriers;
		RelSpan<std::pair<DomainFlagBits, uint64_t>> relative_waits;
		RelSpan<std::pair<DomainFlagBits, uint64_t>> absolute_waits;
		RelSpan<FutureBase*> future_signals;
		RelSpan<int32_t> referenced_swapchains; // TODO: maybe not the best place for it

		int32_t is_waited_on = 0;
	};

#define INIT(x) x(decltype(x)::allocator_type(*arena_))
	struct RGImpl {
		std::unique_ptr<arena> arena_;
		std::vector<PassWrapper, short_alloc<PassWrapper, 64>> passes;

		std::vector<QualifiedName, short_alloc<QualifiedName, 64>> imported_names;          // names coming from subgraphs
		std::vector<std::pair<Name, Name>, short_alloc<std::pair<Name, Name>, 64>> aliases; // maps resource names to resource names
		std::vector<std::pair<QualifiedName, std::pair<QualifiedName, Subrange::Image>>,
		            short_alloc<std::pair<QualifiedName, std::pair<QualifiedName, Subrange::Image>>, 64>>
		    diverged_subchain_headers;

		robin_hood::unordered_flat_map<QualifiedName, AttachmentInfo> bound_attachments;
		robin_hood::unordered_flat_map<QualifiedName, BufferInfo> bound_buffers;

		std::vector<IAInference, short_alloc<IAInference, 64>> ia_inference_rules;
		std::vector<BufferInference, short_alloc<BufferInference, 64>> buf_inference_rules;

		std::vector<Resource> resources;

		struct SGInfo {
			uint64_t count = 0;
			std::span<std::pair<Name, QualifiedName>> exported_names = {};
		};

		std::vector<std::pair<std::shared_ptr<RenderGraph>, SGInfo>, short_alloc<std::pair<std::shared_ptr<RenderGraph>, SGInfo>, 64>> subgraphs;

		std::vector<std::pair<QualifiedName, Acquire>, short_alloc<std::pair<QualifiedName, Acquire>, 64>> acquires;
		std::vector<std::pair<QualifiedName, Release>, short_alloc<std::pair<QualifiedName, Release>, 64>> releases;

		RGImpl() :
		    arena_(new arena(sizeof(Pass) * 64)),
		    INIT(passes),
		    INIT(imported_names),
		    INIT(aliases),
		    INIT(diverged_subchain_headers),
		    INIT(ia_inference_rules),
		    INIT(buf_inference_rules),
		    INIT(subgraphs),
		    INIT(acquires),
		    INIT(releases) {}

		Name resolve_alias(Name in) {
			auto it = std::find_if(aliases.begin(), aliases.end(), [=](auto& v) { return v.first == in; });
			if (it == aliases.end()) {
				return in;
			} else {
				return resolve_alias(it->second);
			}
		};

		robin_hood::unordered_flat_set<QualifiedName> get_available_resources();

		Resource& get_resource(std::vector<PassInfo>& pass_infos, ChainAccess& ca) {
			return resources[pass_infos[ca.pass].resources.offset0 + ca.resource];
		}

		size_t temporary_name_counter = 0;
		Name temporary_name = "_temporary";
	};

	using ResourceLinkMap = robin_hood::unordered_node_map<QualifiedName, ChainLink>;

	struct RGCImpl {
		RGCImpl() : arena_(new arena(4 * 1024 * 1024)), INIT(computed_passes), INIT(ordered_passes), INIT(partitioned_passes), INIT(rpis) {}
		RGCImpl(arena* a) : arena_(a), INIT(computed_passes), INIT(ordered_passes), INIT(partitioned_passes), INIT(rpis) {}
		std::unique_ptr<arena> arena_;

		// per PassInfo
		std::vector<Resource> resources;

		std::vector<std::pair<DomainFlagBits, uint64_t>> waits;
		std::vector<std::pair<DomainFlagBits, uint64_t>> absolute_waits;
		std::vector<FutureBase*> future_signals;
		std::deque<QualifiedName> qfname_references;
		// /per PassInfo

		std::vector<PassInfo, short_alloc<PassInfo, 64>> computed_passes;
		std::vector<PassInfo*, short_alloc<PassInfo*, 64>> ordered_passes;
		std::vector<size_t> computed_pass_idx_to_ordered_idx;
		std::vector<size_t> ordered_idx_to_computed_pass_idx;
		std::vector<PassInfo*, short_alloc<PassInfo*, 64>> partitioned_passes;
		std::vector<size_t> computed_pass_idx_to_partitioned_idx;

		robin_hood::unordered_flat_map<QualifiedName, QualifiedName> computed_aliases; // maps resource names to resource names
		robin_hood::unordered_flat_map<QualifiedName, QualifiedName> assigned_names;   // maps resource names to attachment names
		robin_hood::unordered_flat_map<Name, uint64_t> sg_name_counter;
		robin_hood::unordered_flat_map<const RenderGraph*, std::string> sg_prefixes;

		std::vector<VkImageMemoryBarrier2KHR> image_barriers;
		std::vector<VkMemoryBarrier2KHR> mem_barriers;

		ResourceLinkMap res_to_links;
		std::vector<ChainAccess> pass_reads;
		Resource& get_resource(ChainAccess& ca) {
			return resources[computed_passes[ca.pass].resources.offset0 + ca.resource];
		}

		const Resource& get_resource(const ChainAccess& ca) const {
			return resources[computed_passes[ca.pass].resources.offset0 + ca.resource];
		}

		PassInfo& get_pass(ChainAccess& ca) {
			return computed_passes[ca.pass];
		}

		PassInfo& get_pass(int32_t ordered_pass_idx) {
			assert(ordered_pass_idx >= 0);
			return *ordered_passes[ordered_pass_idx];
		}

		std::vector<ChainLink*> chains;
		std::vector<ChainLink*> child_chains;
		std::deque<ChainLink> helper_links;
		std::vector<int32_t> swapchain_references;
		std::vector<AttachmentRPInfo> rp_infos;
		std::array<size_t, 3> last_ordered_pass_idx_in_domain_array;
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
		}

		std::vector<AttachmentInfo> bound_attachments;
		std::vector<BufferInfo> bound_buffers;
		AttachmentInfo& get_bound_attachment(int32_t idx) {
			assert(idx < 0);
			return bound_attachments[(-1 * idx - 1)];
		}
		BufferInfo& get_bound_buffer(int32_t idx) {
			assert(idx < 0);
			return bound_buffers[(-1 * idx - 1)];
		}

		std::vector<ChainLink*> attachment_use_chain_references;
		std::vector<RenderPassInfo*> attachment_rp_references;

		std::vector<std::pair<QualifiedName, Release>> releases;
		Release& get_release(int64_t idx) {
			return releases[-1 * (idx)-1].second;
		}

		std::unordered_map<QualifiedName, IAInferences> ia_inference_rules;
		std::unordered_map<QualifiedName, BufferInferences> buf_inference_rules;

		robin_hood::unordered_flat_map<QualifiedName, std::pair<QualifiedName, Subrange::Image>> diverged_subchain_headers;

		QualifiedName resolve_name(QualifiedName in) {
			auto it = assigned_names.find(in);
			if (it == assigned_names.end()) {
				return in;
			} else {
				return it->second;
			}
		};

		QualifiedName resolve_alias(QualifiedName in) {
			auto it = computed_aliases.find(in);
			if (it == computed_aliases.end()) {
				return in;
			} else {
				return it->second;
			}
		};

		QualifiedName resolve_alias_rec(QualifiedName in) {
			auto it = computed_aliases.find(in);
			if (it == computed_aliases.end()) {
				return in;
			} else {
				return resolve_alias_rec(it->second);
			}
		};

		void compute_assigned_names();

		std::vector<RenderPassInfo, short_alloc<RenderPassInfo, 64>> rpis;
		std::span<PassInfo*> transfer_passes, compute_passes, graphics_passes;

		void append(Name subgraph_name, const RenderGraph& other);

		void merge_diverge_passes(std::vector<PassInfo, short_alloc<PassInfo, 64>>& passes);

		void compute_prefixes(const RenderGraph& rg, std::string& prefix);
		void inline_subgraphs(const RenderGraph& rg, robin_hood::unordered_flat_set<RenderGraph*>& consumed_rgs);

		Result<void> terminate_chains();
		Result<void> diagnose_unheaded_chains();
		Result<void> schedule_intra_queue(std::span<struct PassInfo> passes, const RenderGraphCompileOptions& compile_options);
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
	};
#undef INIT

	inline PassInfo::PassInfo(PassWrapper& p) : pass(&p) {}

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

	namespace errors {
		RenderGraphException make_unattached_resource_exception(PassInfo& pass_info, Resource& resource);
		RenderGraphException make_cbuf_references_unknown_resource(PassInfo& pass_info, Resource::Type type, Name name);
		RenderGraphException make_cbuf_references_undeclared_resource(PassInfo& pass_info, Resource::Type type, Name name);
	} // namespace errors
};  // namespace vuk