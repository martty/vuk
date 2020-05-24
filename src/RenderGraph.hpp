#pragma once

#include <stdio.h>
#include <vector>
#include <unordered_map>
#include <vulkan/vulkan.hpp>
#include <unordered_set>
#include <algorithm>
#include <iostream>
#include <variant>
#include <string_view>
#include <optional>
#include <functional>
#include "Hash.hpp"
#include "vuk_fwd.hpp"
#include "RenderPass.hpp"
#include "unordered_map.hpp"

namespace vuk {
	struct Preserve {};
	struct ClearColor {
		ClearColor(uint32_t r, uint32_t g, uint32_t b, uint32_t a) {
			ccv.setUint32({ r,g,b,a });
		}
		ClearColor(float r, float g, float b, float a) {
			ccv.setFloat32({ r,g,b,a });
		}
		vk::ClearColorValue ccv;
	};

	struct ClearDepthStencil {
		ClearDepthStencil(float depth, uint32_t stencil) {
			cdsv.depth = depth;
			cdsv.stencil = stencil;
		}
		vk::ClearDepthStencilValue cdsv;
	};


	struct PreserveOrClear {
		PreserveOrClear(ClearColor cc) : clear(true) { c.color = cc.ccv; }
		PreserveOrClear(ClearDepthStencil cc) : clear(true) { c.depthStencil = cc.cdsv; }
		PreserveOrClear(Preserve) : clear(false) {}

		bool clear;
		vk::ClearValue c;
	};

	struct Clear {
		Clear() = default;
		Clear(ClearColor cc) { c.color = cc.ccv; }
		Clear(ClearDepthStencil cc) { c.depthStencil = cc.cdsv; }
	
		vk::ClearValue c;
	};

	enum ImageAccess {
		eColorRW,
		eColorWrite,
		eColorRead,
		eColorResolveRead, // special op to mark renderpass resolve read
		eColorResolveWrite, // special op to mark renderpass resolve write
		eDepthStencilRW,
		eDepthStencilRead,
		eInputRead,
		eVertexSampled,
		eVertexRead,
		eFragmentSampled,
		eFragmentRead,
		eFragmentWrite // written using image store
	};

	struct Samples {
		vk::SampleCountFlagBits count;
		bool infer;

		struct Framebuffer {};

		Samples() : count(vk::SampleCountFlagBits::e1), infer(false) {}
		Samples(vk::SampleCountFlagBits samples) : count(samples), infer(false) {}
		Samples(Framebuffer) : infer(true) {}

		constexpr static auto e1 = vk::SampleCountFlagBits::e1;
		constexpr static auto e2 = vk::SampleCountFlagBits::e2;
		constexpr static auto e4 = vk::SampleCountFlagBits::e4;
		constexpr static auto e8 = vk::SampleCountFlagBits::e8;
		constexpr static auto e16 = vk::SampleCountFlagBits::e16;
		constexpr static auto e32 = vk::SampleCountFlagBits::e32;
		constexpr static auto e64 = vk::SampleCountFlagBits::e64;
	};

	struct Resource;
	struct BufferResource {};
	struct ImageResource {
		Name name;

		Resource operator()(ImageAccess ia);
	};
}

inline vuk::ImageResource operator "" _image(const char* name, size_t) {
	return { name };
}

namespace vuk {
	struct Resource {
		Name src_name;
		Name use_name;
		enum class Type { eBuffer, eImage } type;
		ImageAccess ia;
		struct Use {
			vk::PipelineStageFlags stages;
			vk::AccessFlags access;
			vk::ImageLayout layout; // ignored for buffers
		};

		Resource(Name n, Type t, ImageAccess ia) : src_name(n), use_name(n), type(t), ia(ia) {}
		Resource(Name src, Name use, Type t, ImageAccess ia) : src_name(src), use_name(use), type(t), ia(ia) {}

		bool operator==(const Resource& o) const {
			return (use_name == o.use_name && src_name == o.src_name);// || use_name == o.src_name || src_name == o.use_name;
		}
	};

	inline Resource ImageResource::operator()(ImageAccess ia) {
		return Resource{name, Resource::Type::eImage, ia};
	}

	struct Pass {
		Name name;
		Name executes_on;
		float auxiliary_order = 0.f;

		std::vector<Resource> resources;
		std::unordered_map<Name, Name> resolves; // src -> dst

		std::function<void(vuk::CommandBuffer&)> execute;
	};
}

namespace std {
	template<> struct hash<vuk::Resource> {
		std::size_t operator()(vuk::Resource const& s) const noexcept {
			size_t h = 0;
			hash_combine(h, s.src_name, s.use_name, s.type);
			return h;
		}
	};
}

#include <ShortAlloc.hpp>

namespace vuk {
	struct Extent2D : public vk::Extent2D {
		using vk::Extent2D::Extent2D;

		Extent2D(vk::Extent2D e) : vk::Extent2D(e) {}

		struct Framebuffer {
			float width = 1.0f;
			float height = 1.0f;
		};
	};

	struct RenderGraph {
        arena arena_;
        RenderGraph();

		struct PassInfo {
            PassInfo(arena&);
            Pass pass;

            size_t render_pass_index;
            uint32_t subpass;

            ska::unordered_set<Resource, std::hash<Resource>, std::equal_to<Resource>, short_alloc<Resource, 16>> inputs;
            ska::unordered_set<Resource, std::hash<Resource>, std::equal_to<Resource>, short_alloc<Resource, 16>> outputs;

            ska::unordered_set<Resource, std::hash<Resource>, std::equal_to<Resource>, short_alloc<Resource, 16>> global_inputs;
            ska::unordered_set<Resource, std::hash<Resource>, std::equal_to<Resource>, short_alloc<Resource, 16>> global_outputs;

            bool is_head_pass = false;
            bool is_tail_pass = false;
        };

		std::vector<PassInfo, short_alloc<PassInfo, 64>> passes;

		std::vector<PassInfo*, short_alloc<PassInfo*, 8>> head_passes;
		std::vector<PassInfo*, short_alloc<PassInfo*, 8>> tail_passes;

		ska::unordered_map<Name, Name, std::hash<Name>, std::equal_to<Name>, short_alloc<std::pair<const Name, Name>, 64>> aliases;

		ska::unordered_set<Resource, std::hash<Resource>, std::equal_to<Resource>, short_alloc<Resource, 16>> global_inputs;
		ska::unordered_set<Resource, std::hash<Resource>, std::equal_to<Resource>, short_alloc<Resource, 16>> global_outputs;
        std::vector<Resource, short_alloc<Resource, 16>> global_io;

		struct UseRef {
			Resource::Use use;
			PassInfo* pass = nullptr;
		};

		ska::unordered_map<Name, std::vector<UseRef, short_alloc<UseRef, 64>>, std::hash<Name>, std::equal_to<Name>,
                           short_alloc<std::pair<const Name, std::vector<UseRef, short_alloc<UseRef, 64>>>, 64>>
            use_chains;

		struct AttachmentSInfo {
			vk::ImageLayout layout;
			vk::AccessFlags access;
			vk::PipelineStageFlags stage;
		};

		struct AttachmentRPInfo {
			Name name;

			enum class Sizing {
				eAbsolute, eFramebufferRelative
			} sizing;
			vuk::Extent2D::Framebuffer fb_relative;
			vuk::Extent2D extents;
			vuk::Samples samples;

			vk::AttachmentDescription description;

			Resource::Use initial, final;

			enum class Type {
				eInternal, eExternal, eSwapchain
			} type;

			vuk::ImageView iv;
			vk::Image image = {};
			// swapchain for swapchain
			Swapchain* swapchain;

			// optionally set
			bool should_clear = false;
			Clear clear_value;
		};

		struct SubpassInfo {
            SubpassInfo(arena&);
			std::vector<PassInfo*, short_alloc<PassInfo*, 16>> passes;
		};

		struct RenderPassInfo {
            RenderPassInfo(arena&);
			std::vector<SubpassInfo, short_alloc<SubpassInfo, 64>> subpasses;
			std::vector<AttachmentRPInfo, short_alloc<AttachmentRPInfo, 16>> attachments;
			vuk::RenderPassCreateInfo rpci;
			vuk::FramebufferCreateInfo fbci;
			vk::RenderPass handle;
			vk::Framebuffer framebuffer;
		};
		std::vector<RenderPassInfo, short_alloc<RenderPassInfo, 64>> rpis;

		void add_pass(Pass p) {
			PassInfo pi(arena_);
			pi.pass = std::move(p);
			passes.push_back(pi);
		}

		// determine rendergraph inputs and outputs, and resources that are neither
		void build_io();

		void build();

		// RGscaffold
		ska::unordered_map<Name, AttachmentRPInfo> bound_attachments;
		void bind_attachment_to_swapchain(Name name, Swapchain* swp, Clear);
		void mark_attachment_internal(Name, vk::Format, vuk::Extent2D, vuk::Samples, Clear);
		void mark_attachment_internal(Name, vk::Format, vuk::Extent2D::Framebuffer, vuk::Samples, Clear);
		void mark_attachment_resolve(Name resolved_name, Name ms_name);
		vk::ImageUsageFlags compute_usage(std::vector<vuk::RenderGraph::UseRef, short_alloc<UseRef, 64>>& chain);

		// RG
		void build(vuk::PerThreadContext&);
		void create_attachment(vuk::PerThreadContext&, Name name, RenderGraph::AttachmentRPInfo& attachment_info, vuk::Extent2D extents, vk::SampleCountFlagBits);
		vk::CommandBuffer execute(vuk::PerThreadContext&, std::vector<std::pair<Swapchain*, size_t>> swp_with_index, bool use_secondary_command_buffers);
	};
	void sync_bound_attachment_to_renderpass(vuk::RenderGraph::AttachmentRPInfo& rp_att, vuk::RenderGraph::AttachmentRPInfo& attachment_info);
}
