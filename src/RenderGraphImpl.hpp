#pragma once

#include "RelSpan.hpp"
#include "RenderGraphUtil.hpp"
#include "RenderPass.hpp"
#include "vuk/ShortAlloc.hpp"
#include "vuk/SourceLocation.hpp"

#include <deque>
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
		std::vector<VkImageMemoryBarrier2KHR, short_alloc<VkImageMemoryBarrier2KHR, 64>> pre_barriers, post_barriers;
		std::vector<VkMemoryBarrier2KHR, short_alloc<VkMemoryBarrier2KHR, 64>> pre_mem_barriers, post_mem_barriers;
		std::vector<std::pair<DomainFlagBits, uint32_t>, short_alloc<std::pair<DomainFlagBits, uint32_t>, 64>> waits;
	};

	using IARule = std::function<void(const struct InferenceContext&, ImageAttachment&)>;
	using BufferRule = std::function<void(const struct InferenceContext&, Buffer&)>;

	struct IAInference {
		Name resource;
		Name prefix;
		IARule rule;
	};

	struct IAInferences {
		Name prefix;
		std::vector<IARule> rules;
	};

	struct BufferInference {
		Name resource;
		Name prefix;
		BufferRule rule;
	};

	struct BufferInferences {
		Name prefix;
		std::vector<BufferRule> rules;
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
		source_location source;
	};

	struct PassInfo {
		PassInfo(arena&, PassWrapper&);

		PassWrapper* pass;

		QualifiedName qualified_name;

		size_t render_pass_index;
		uint32_t subpass;
		DomainFlags domain;

		RelSpan<Resource> resources;
		RelSpan<std::pair<QualifiedName, QualifiedName>> resolves; // src -> dst
		RelSpan<QualifiedName> input_names;
		RelSpan<QualifiedName> output_names;
		RelSpan<QualifiedName> write_input_names;

		RelSpan<std::pair<DomainFlagBits, PassInfo*>> waits;
		RelSpan<std::pair<DomainFlagBits, uint64_t>> absolute_waits;
		RelSpan<FutureBase*> future_signals;

		bool is_waited_on = false;
		uint32_t bloom_resolved_inputs = 0;

		uint32_t bloom_outputs = 0;
		uint32_t bloom_write_inputs = 0;

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
		std::vector<std::pair<QualifiedName, std::pair<QualifiedName, Subrange::Image>>,
		            short_alloc<std::pair<QualifiedName, std::pair<QualifiedName, Subrange::Image>>, 64>>
		    diverged_subchain_headers;

		robin_hood::unordered_flat_map<QualifiedName, AttachmentInfo> bound_attachments;
		robin_hood::unordered_flat_map<QualifiedName, BufferInfo> bound_buffers;

		std::vector<IAInference, short_alloc<IAInference, 64>> ia_inference_rules;
		std::vector<BufferInference, short_alloc<BufferInference, 64>> buf_inference_rules;

		std::vector<Resource> resources;
		std::vector<std::pair<Name, Name>> resolves; // src -> dst
		std::vector<QualifiedName> input_names;
		std::vector<QualifiedName> output_names;
		std::vector<QualifiedName> write_input_names;

		struct SGInfo {
			uint64_t count = 0;
			std::span<std::pair<Name, Name>> exported_names = {};
		};

		std::vector<std::pair<std::shared_ptr<RenderGraph>, SGInfo>, short_alloc<std::pair<std::shared_ptr<RenderGraph>, SGInfo>, 64>> subgraphs;

		std::vector<std::pair<QualifiedName, Acquire>, short_alloc<std::pair<QualifiedName, Acquire>, 64>> acquires;
		std::vector<std::pair<QualifiedName, Release>, short_alloc<std::pair<QualifiedName, Release>, 64>> releases;

		RGImpl() :
		    arena_(new arena(sizeof(Pass) * 64)),
		    INIT(passes),
		    INIT(imported_names),
		    INIT(aliases),
		    INIT(whole_names_consumed),
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

		// per PassInfo
		std::vector<Resource> resources;
		std::vector<std::pair<QualifiedName, QualifiedName>> resolves; // src -> dst
		std::vector<QualifiedName> input_names;
		std::vector<QualifiedName> output_names;
		std::vector<QualifiedName> write_input_names;

		std::vector<std::pair<DomainFlagBits, PassInfo*>> waits;
		std::vector<std::pair<DomainFlagBits, uint64_t>> absolute_waits;
		std::vector<FutureBase*> future_signals;
		std::deque<QualifiedName> qfname_references;
		// /per PassInfo

		std::vector<PassInfo, short_alloc<PassInfo, 64>> computed_passes;
		std::vector<PassInfo*, short_alloc<PassInfo*, 64>> ordered_passes;
		robin_hood::unordered_flat_map<QualifiedName, QualifiedName> computed_aliases; // maps resource names to resource names
		robin_hood::unordered_flat_map<QualifiedName, QualifiedName> assigned_names;   // maps resource names to attachment names
		robin_hood::unordered_flat_map<Name, uint64_t> sg_name_counter;
		std::unordered_map<RenderGraph*, std::string> sg_prefixes;
		robin_hood::unordered_node_map<QualifiedName, std::vector<UseRef, short_alloc<UseRef, 64>>> use_chains;

		robin_hood::unordered_flat_map<QualifiedName, AttachmentInfo> bound_attachments;
		robin_hood::unordered_flat_map<QualifiedName, BufferInfo> bound_buffers;
		std::vector<void*> attachment_use_chain_references;
		std::vector<RenderPassInfo*> attachment_rp_references;

		std::unordered_map<QualifiedName, Acquire> acquires;
		std::unordered_multimap<QualifiedName, Release> releases;

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

		// note : call it on resolved names only
		QualifiedName whole_name(QualifiedName in) {
			if (auto it = diverged_subchain_headers.find(in); it != diverged_subchain_headers.end()) {
				auto& sch_info = it->second;
				return resolve_name(sch_info.first);
			} else {
				return in;
			}
		}

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

		std::vector<RenderPassInfo, short_alloc<RenderPassInfo, 64>> rpis;
		size_t num_graphics_rpis = 0;
		size_t num_compute_rpis = 0;
		size_t num_transfer_rpis = 0;

		void append(Name subgraph_name, const RenderGraph& other);

		// determine rendergraph inputs and outputs, and resources that are neither
		void build_io(std::span<struct PassInfo> passes);

		std::unordered_map<RenderGraph*, std::string> compute_prefixes(const RenderGraph& rg, bool do_prefix);
		void inline_subgraphs(const std::shared_ptr<RenderGraph>& rg, std::unordered_set<std::shared_ptr<RenderGraph>>& consumed_rgs);

		void schedule_intra_queue(std::span<struct PassInfo> passes, const RenderGraphCompileOptions& compile_options);
	};
#undef INIT

#define INIT2(x) x(decltype(x)::allocator_type(arena_))
	inline PassInfo::PassInfo(arena& arena_, PassWrapper& p) : pass(&p) {}
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

	namespace errors {
		RenderGraphException make_unattached_resource_exception(PassInfo& pass_info, Resource& resource, QualifiedName undiverged_name);
		RenderGraphException make_cbuf_references_unknown_resource(PassInfo& pass_info, Resource::Type type, Name name);
		RenderGraphException make_cbuf_references_undeclared_resource(PassInfo& pass_info, Resource::Type type, Name name);
	} // namespace errors
};  // namespace vuk