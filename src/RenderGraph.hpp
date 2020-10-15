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

	enum Access {
		eNone,
		eClear,
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
		eAttributeRead,
		eFragmentSampled,
		eFragmentRead,
		eFragmentWrite, // written using image store
		eTransferSrc,
		eTransferDst,
		eComputeRead,
		eComputeWrite,
		eComputeRW,
	};

	struct Resource;
	struct BufferResource {
		Name name;

		Resource operator()(Access ba);
	};
	struct ImageResource {
		Name name;

		Resource operator()(Access ia);
	};
}

inline vuk::ImageResource operator "" _image(const char* name, size_t) {
	return { name };
}

inline vuk::BufferResource operator "" _buffer(const char* name, size_t) {
	return { name };
}

namespace vuk {
	struct Resource {
		Name src_name;
		Name use_name;
		enum class Type { eBuffer, eImage } type;
		Access ia;
		struct Use {
			vk::PipelineStageFlags stages;
			vk::AccessFlags access;
			vk::ImageLayout layout; // ignored for buffers
		};

		Resource(Name n, Type t, Access ia) : src_name(n), use_name(n), type(t), ia(ia) {}
		Resource(Name src, Name use, Type t, Access ia) : src_name(src), use_name(use), type(t), ia(ia) {}

		bool operator==(const Resource& o) const {
			return (use_name == o.use_name && src_name == o.src_name);
		}
	};

	inline Resource ImageResource::operator()(Access ia) {
		return Resource{name, Resource::Type::eImage, ia};
	}

	inline Resource BufferResource::operator()(Access ba) {
		return Resource{ name, Resource::Type::eBuffer, ba };
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

	struct Attachment {
        vk::Image image;
        vuk::ImageView image_view;
        
		vk::Extent2D extent;
        vk::Format format;
        vuk::Samples sample_count = vuk::Samples::e1;
        Clear clear_value;

		static Attachment from_texture(const vuk::Texture& t, Clear clear_value) {
            return Attachment{
                .image = t.image.get(), .image_view = t.view.get(), .extent = {t.extent.width, t.extent.height}, .format = t.format, .sample_count = {t.sample_count}, .clear_value = clear_value};
        }
    };

	struct RenderGraph {
        std::unique_ptr<arena> arena_;
        RenderGraph();

		struct PassInfo {
            PassInfo(arena&);
			
            Pass pass;

            size_t render_pass_index;
            uint32_t subpass;

            std::unordered_set<Resource, std::hash<Resource>, std::equal_to<Resource>, short_alloc<Resource, 16>> inputs;
            std::unordered_set<Resource, std::hash<Resource>, std::equal_to<Resource>, short_alloc<Resource, 16>> outputs;

            std::unordered_set<Resource, std::hash<Resource>, std::equal_to<Resource>, short_alloc<Resource, 16>> global_inputs;
            std::unordered_set<Resource, std::hash<Resource>, std::equal_to<Resource>, short_alloc<Resource, 16>> global_outputs;

            bool is_head_pass = false;
            bool is_tail_pass = false;
        };

		std::vector<PassInfo> passes;

		std::vector<PassInfo*, short_alloc<PassInfo*, 8>> head_passes;
		std::vector<PassInfo*, short_alloc<PassInfo*, 8>> tail_passes;

		ska::unordered_map<Name, Name, std::hash<Name>, std::equal_to<Name>, short_alloc<std::pair<const Name, Name>, 64>> aliases;

		std::unordered_set<Resource, std::hash<Resource>, std::equal_to<Resource>, short_alloc<Resource, 16>> global_inputs;
		std::unordered_set<Resource, std::hash<Resource>, std::equal_to<Resource>, short_alloc<Resource, 16>> global_outputs;
        std::vector<Resource, short_alloc<Resource, 16>> global_io;

		struct UseRef {
			Resource::Use use;
			PassInfo* pass = nullptr;
		};

		std::unordered_map<Name, std::vector<UseRef, short_alloc<UseRef, 64>>, std::hash<Name>, std::equal_to<Name>,
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

		struct BufferInfo {
			Name name;

			Resource::Use initial;
			Resource::Use final;

			vuk::Buffer buffer;
		};


		struct ImageBarrier {
			Name image;
			vk::ImageMemoryBarrier barrier;
			vk::PipelineStageFlags src;
			vk::PipelineStageFlags dst;
		};

		struct MemoryBarrier {
			vk::MemoryBarrier barrier;
			vk::PipelineStageFlags src;
			vk::PipelineStageFlags dst;
		};

		struct SubpassInfo {
            SubpassInfo(arena&);
			std::vector<PassInfo*, short_alloc<PassInfo*, 16>> passes;
			std::vector<ImageBarrier> pre_barriers;
			std::vector<ImageBarrier> post_barriers;
			std::vector<MemoryBarrier> pre_mem_barriers, post_mem_barriers;
		};

		struct RenderPassInfo {
            RenderPassInfo(arena&);
			std::vector<SubpassInfo, short_alloc<SubpassInfo, 64>> subpasses;
			std::vector<AttachmentRPInfo, short_alloc<AttachmentRPInfo, 16>> attachments;
			vuk::RenderPassCreateInfo rpci;
			vuk::FramebufferCreateInfo fbci;
			bool framebufferless = false;
			vk::RenderPass handle = {};
			vk::Framebuffer framebuffer;
		};
		std::vector<RenderPassInfo, short_alloc<RenderPassInfo, 64>> rpis;

		void add_pass(Pass p) {
			PassInfo pi(*arena_);
			pi.pass = std::move(p);
			passes.push_back(pi);
		}

		// determine rendergraph inputs and outputs, and resources that are neither
		void build_io();

		void build();

		// RGscaffold
		std::unordered_map<Name, AttachmentRPInfo> bound_attachments;
		std::unordered_map<Name, BufferInfo> bound_buffers;
		void bind_attachment_to_swapchain(Name, Swapchain* swp, Clear);
		void mark_attachment_internal(Name, vk::Format, vuk::Extent2D, vuk::Samples, Clear);
		void mark_attachment_internal(Name, vk::Format, vuk::Extent2D::Framebuffer, vuk::Samples, Clear);
		void mark_attachment_resolve(Name resolved_name, Name ms_name);
		void bind_buffer(Name, vuk::Buffer);
        void bind_attachment(Name, Attachment, Access initial, Access final);
		vk::ImageUsageFlags compute_usage(std::vector<vuk::RenderGraph::UseRef, short_alloc<UseRef, 64>>& chain);

		// RG
		void build(vuk::PerThreadContext&);
		void create_attachment(vuk::PerThreadContext&, Name name, RenderGraph::AttachmentRPInfo& attachment_info, vuk::Extent2D extents, vk::SampleCountFlagBits);
		vk::CommandBuffer execute(vuk::PerThreadContext&, std::vector<std::pair<Swapchain*, size_t>> swp_with_index, bool use_secondary_command_buffers);

		BufferInfo get_resource_buffer(Name);

		private:
			void fill_renderpass_info(vuk::RenderGraph::RenderPassInfo& rpass, const size_t& i, vuk::CommandBuffer& cobuf);
	};
}
