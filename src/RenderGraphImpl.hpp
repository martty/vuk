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
		std::vector<ImageBarrier> pre_barriers, post_barriers;
		std::vector<MemoryBarrier> pre_mem_barriers, post_mem_barriers;
		std::vector<std::pair<DomainFlagBits, uint32_t>> waits;
	};

	using IARule = std::function<void(const struct InferenceContext&, ImageAttachment&)>;
	struct IAInference {
		Name prefix;
		std::vector<IARule> rules;
	};

#define INIT(x) x(decltype(x)::allocator_type(*arena_))
	struct RGImpl {
		std::unique_ptr<arena> arena_;
		std::vector<PassInfo, short_alloc<PassInfo, 64>> passes;
		std::vector<PassInfo*, short_alloc<PassInfo*, 64>> ordered_passes;

		robin_hood::unordered_flat_set<Name> imported_names;       // names coming from subgraphs
		robin_hood::unordered_flat_map<Name, Name> aliases;        // maps resource names to resource names
		robin_hood::unordered_flat_map<Name, Name> assigned_names; // maps resource names to attachment names
		robin_hood::unordered_flat_set<Name> poisoned_names;
		robin_hood::unordered_flat_map<Name, uint64_t> sg_name_counter;

		robin_hood::unordered_node_map<Name, std::vector<UseRef, short_alloc<UseRef, 64>>> use_chains;

		std::vector<RenderPassInfo, short_alloc<RenderPassInfo, 64>> rpis;
		size_t num_graphics_rpis = 0;
		size_t num_compute_rpis = 0;
		size_t num_transfer_rpis = 0;

		robin_hood::unordered_flat_map<Name, AttachmentInfo> bound_attachments;
		robin_hood::unordered_flat_map<Name, BufferInfo> bound_buffers;

		std::unordered_map<Name, IAInference> ia_inference_rules;

		struct SGInfo {
			uint64_t count = 0;
			std::unordered_map<Name, Name> exported_names;
		};

		std::unordered_map<std::shared_ptr<RenderGraph>, SGInfo> subgraphs;

		struct Release {
			QueueResourceUse dst_use;
			FutureBase* signal = nullptr;
		};

		struct Acquire {
			QueueResourceUse src_use;
			DomainFlagBits initial_domain;
			uint64_t initial_visibility;
		};

		std::unordered_map<Name, Acquire> acquires;

		std::unordered_multimap<Name, Release> releases;

		RGImpl() : arena_(new arena(1024 * 1024)), INIT(passes), INIT(ordered_passes), INIT(rpis) {}

		Name resolve_name(Name in) {
			auto it = assigned_names.find(in);
			if (it == assigned_names.end()) {
				return in;
			} else {
				return it->second;
			}
		};

		Name resolve_alias(Name in) {
			auto it = aliases.find(in);
			if (it == aliases.end()) {
				return in;
			} else {
				return resolve_alias(it->second);
			}
		};

		// determine rendergraph inputs and outputs, and resources that are neither
		void build_io();

		robin_hood::unordered_flat_set<Name> get_available_resources();


		size_t temporary_name_counter = 0;
		Name temporary_name = "_temporary";
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
}; // namespace vuk