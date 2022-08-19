#pragma once

#include "RenderGraphUtil.hpp"
#include "RenderPass.hpp"
#include "vuk/ShortAlloc.hpp"

#include <robin_hood.h>

namespace vuk {
	struct RenderPassInfo {
		RenderPassInfo(arena&);
		uint32_t command_buffer_index;
		uint32_t batch_index;
		std::vector<SubpassInfo, short_alloc<SubpassInfo, 64>> subpasses;
		std::vector<AttachmentRPInfo, short_alloc<AttachmentRPInfo, 16>> attachments;
		uint32_t layer_count;
		vuk::RenderPassCreateInfo rpci;
		vuk::FramebufferCreateInfo fbci;
		bool framebufferless = false;
		VkRenderPass handle = {};
		VkFramebuffer framebuffer;
		std::vector<ImageBarrier, short_alloc<ImageBarrier, 64>> pre_barriers, post_barriers;
		std::vector<MemoryBarrier, short_alloc<MemoryBarrier, 64>> pre_mem_barriers, post_mem_barriers;
		std::vector<std::pair<DomainFlagBits, uint32_t>, short_alloc<std::pair<DomainFlagBits, uint32_t>, 64>> waits;
	};

	using IARule = std::function<void(const struct InferenceContext&, ImageAttachment&)>;
	struct IAInference {
		Name resource;
		Name prefix;
		IARule rule;
	};

	struct IAInferences {
		Name prefix;
		std::vector<IARule> rules;
	};

	struct Release {
		QueueResourceUse dst_use;
		Subrange subrange;
		FutureBase* signal = nullptr;
	};

	struct Acquire {
		QueueResourceUse src_use;
		DomainFlagBits initial_domain;
		uint64_t initial_visibility;
	};

	struct PassWrapper {
		Name name;
		DomainFlags execute_on;

		bool use_secondary_command_buffers;

		std::span<Resource> resources;
		std::span<std::pair<Name, Name>> resolves; // src -> dst

		std::function<void(CommandBuffer&)> execute;
		std::byte* arguments; // internal use
		PassType type;
	};

	struct PassInfo {
		PassInfo(arena&, PassWrapper&);

		PassWrapper* pass;

		Name qualified_name;

		std::vector<Resource, short_alloc<Resource, 64>> resources;
		std::vector<std::pair<Name, Name>, short_alloc<std::pair<Name, Name>, 64>> resolves; // src -> dst

		size_t render_pass_index;
		uint32_t subpass;
		DomainFlags domain;

		Name prefix;

		std::vector<std::pair<DomainFlagBits, PassInfo*>, short_alloc<std::pair<DomainFlagBits, PassInfo*>, 16>> waits;
		std::vector<std::pair<DomainFlagBits, uint64_t>, short_alloc<std::pair<DomainFlagBits, uint64_t>, 16>> absolute_waits;
		bool is_waited_on = false;
		uint32_t bloom_resolved_inputs = 0;
		std::vector<Name, short_alloc<Name, 16>> input_names;
		uint32_t bloom_outputs = 0;
		uint32_t bloom_write_inputs = 0;
		std::vector<Name, short_alloc<Name, 16>> output_names;
		std::vector<Name, short_alloc<Name, 16>> write_input_names;

		std::vector<FutureBase*, short_alloc<FutureBase*, 16>> future_signals;

		bool is_head_pass = false;
		bool is_tail_pass = false;
	};

#define INIT(x) x(decltype(x)::allocator_type(*arena_))
	struct RGImpl {
		std::unique_ptr<arena> arena_;
		std::vector<PassWrapper, short_alloc<PassWrapper, 64>> passes;

		std::vector<Name, short_alloc<Name, 64>> imported_names;                            // names coming from subgraphs
		std::vector<std::pair<Name, Name>, short_alloc<std::pair<Name, Name>, 64>> aliases; // maps resource names to resource names
		std::vector<Name, short_alloc<Name, 64>> whole_names_consumed;
		std::vector<std::pair<Name, std::pair<Name, Subrange::Image>>, short_alloc<std::pair<Name, std::pair<Name, Subrange::Image>>, 64>>
		    diverged_subchain_headers;

		robin_hood::unordered_flat_map<Name, AttachmentInfo> bound_attachments;
		robin_hood::unordered_flat_map<Name, BufferInfo> bound_buffers;

		std::vector<IAInference, short_alloc<IAInference, 64>> ia_inference_rules;

		struct SGInfo {
			uint64_t count = 0;
			std::span<std::pair<Name, Name>> exported_names = {};
		};

		std::vector<std::pair<std::shared_ptr<RenderGraph>, SGInfo>, short_alloc<std::pair<std::shared_ptr<RenderGraph>, SGInfo>, 64>> subgraphs;

		std::vector<std::pair<Name, Acquire>, short_alloc<std::pair<Name, Acquire>, 64>> acquires;
		std::vector<std::pair<Name, Release>, short_alloc<std::pair<Name, Release>, 64>> releases;

		RGImpl() :
		    arena_(new arena(sizeof(Pass) * 64)),
		    INIT(passes),
		    INIT(imported_names),
		    INIT(aliases),
		    INIT(whole_names_consumed),
		    INIT(diverged_subchain_headers),
		    INIT(ia_inference_rules),
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

		// determine rendergraph inputs and outputs, and resources that are neither
		std::vector<PassInfo, short_alloc<PassInfo, 64>> build_io(std::span<PassWrapper> passes);

		robin_hood::unordered_flat_set<Name> get_available_resources();

		size_t temporary_name_counter = 0;
		Name temporary_name = "_temporary";
	};

	struct RGCImpl {
		RGCImpl() : arena_(new arena(4 * 1024 * 1024)), INIT(computed_passes), INIT(ordered_passes), INIT(rpis) {}
		RGCImpl(arena* a) : arena_(a), INIT(computed_passes), INIT(ordered_passes), INIT(rpis) {}
		std::unique_ptr<arena> arena_;

		std::vector<PassInfo, short_alloc<PassInfo, 64>> computed_passes;
		std::vector<PassInfo*, short_alloc<PassInfo*, 64>> ordered_passes;
		robin_hood::unordered_flat_map<Name, Name> computed_aliases; // maps resource names to resource names
		robin_hood::unordered_flat_map<Name, Name> assigned_names;   // maps resource names to attachment names
		robin_hood::unordered_flat_map<Name, uint64_t> sg_name_counter;
		robin_hood::unordered_node_map<Name, std::vector<UseRef, short_alloc<UseRef, 64>>> use_chains;

		robin_hood::unordered_flat_map<Name, AttachmentInfo> bound_attachments;
		robin_hood::unordered_flat_map<Name, BufferInfo> bound_buffers;

		std::unordered_map<Name, Acquire> acquires;
		std::unordered_multimap<Name, Release> releases;

		std::unordered_map<Name, IAInferences> ia_inference_rules;

		robin_hood::unordered_flat_map<Name, std::pair<Name, Subrange::Image>> diverged_subchain_headers;

		Name resolve_name(Name in) {
			auto it = assigned_names.find(in);
			if (it == assigned_names.end()) {
				return in;
			} else {
				return it->second;
			}
		};

		// note : call it on resolved names only
		Name whole_name(Name in) {
			if (auto it = diverged_subchain_headers.find(in); it != diverged_subchain_headers.end()) {
				auto& sch_info = it->second;
				return resolve_name(sch_info.first);
			} else {
				return in;
			}
		}

		Name resolve_alias(Name in) {
			auto it = computed_aliases.find(in);
			if (it == computed_aliases.end()) {
				return in;
			} else {
				return it->second;
			}
		};

		Name resolve_alias_rec(Name in) {
			auto it = computed_aliases.find(in);
			if (it == computed_aliases.end()) {
				return in;
			} else {
				return resolve_alias_rec(it->second);
			}
		};

		std::vector<RenderPassInfo, short_alloc<RenderPassInfo, 64>> rpis;
		size_t num_graphics_rpis = 0;
		size_t num_compute_rpis = 0;
		size_t num_transfer_rpis = 0;

		void append(Name subgraph_name, const RenderGraph& other);

		// determine rendergraph inputs and outputs, and resources that are neither
		void build_io(std::span<struct PassInfo> passes);

		std::unordered_map<std::shared_ptr<RenderGraph>, std::string> compute_prefixes(const RenderGraph& rg, bool do_prefix);
		void inline_subgraphs(const std::shared_ptr<RenderGraph>& rg,
		                      const std::unordered_map<std::shared_ptr<RenderGraph>, std::string>& prefixes,
		                      std::unordered_set<std::shared_ptr<RenderGraph>>& consumed_rgs);

		/// @brief Check if this rendergraph is valid.
		/// \throws RenderGraphException
		void validate();

		void schedule_intra_queue(std::span<struct PassInfo> passes, const RenderGraphCompileOptions& compile_options);
	};
#undef INIT

#define INIT2(x) x(decltype(x)::allocator_type(arena_))
	inline PassInfo::PassInfo(arena& arena_, PassWrapper& p) :
	    pass(&p),
	    INIT2(resources),
	    INIT2(resolves),
	    INIT2(waits),
	    INIT2(absolute_waits),
	    INIT2(input_names),
	    INIT2(output_names),
	    INIT2(write_input_names),
	    INIT2(future_signals) {}
#undef INIT2
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
}; // namespace vuk