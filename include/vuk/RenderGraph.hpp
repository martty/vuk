#pragma once

#include <stdio.h>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <algorithm>
#include <iostream>
#include <string_view>
#include <optional>
#include <functional>
#include "vuk/Hash.hpp"
#include "vuk/vuk_fwd.hpp"
#include "RenderPass.hpp"
#include "vuk/Buffer.hpp"
#include "vuk/Image.hpp"
#include <vuk/ShortAlloc.hpp>
#include "unordered_map.hpp"

namespace vuk {
	struct Preserve {};
	struct ClearColor {
		ClearColor(uint32_t r, uint32_t g, uint32_t b, uint32_t a) {
			ccv.uint32[0] = r;
			ccv.uint32[1] = g;
			ccv.uint32[2] = b;
			ccv.uint32[3] = a;
		}
		ClearColor(float r, float g, float b, float a) {
			ccv.float32[0] = r;
			ccv.float32[1] = g;
			ccv.float32[2] = b;
			ccv.float32[3] = a;
		}
		VkClearColorValue ccv;
	};

	struct ClearDepthStencil {
		ClearDepthStencil(float depth, uint32_t stencil) {
			cdsv.depth = depth;
			cdsv.stencil = stencil;
		}
		VkClearDepthStencilValue cdsv;
	};


	struct PreserveOrClear {
		PreserveOrClear(ClearColor cc) : clear(true) { c.color = cc.ccv; }
		PreserveOrClear(ClearDepthStencil cc) : clear(true) { c.depthStencil = cc.cdsv; }
		PreserveOrClear(Preserve) : clear(false) {}

		bool clear;
		VkClearValue c;
	};

	struct Clear {
		Clear() = default;
		Clear(ClearColor cc) { c.color = cc.ccv; }
		Clear(ClearDepthStencil cc) { c.depthStencil = cc.cdsv; }
	
		VkClearValue c;
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
		eMemoryRead,
		eMemoryWrite,
		eMemoryRW
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

	enum class AccessFlagBits : VkAccessFlags {
		eIndirectCommandRead = VK_ACCESS_INDIRECT_COMMAND_READ_BIT,
		eIndexRead = VK_ACCESS_INDEX_READ_BIT,
		eVertexAttributeRead = VK_ACCESS_VERTEX_ATTRIBUTE_READ_BIT,
		eUniformRead = VK_ACCESS_UNIFORM_READ_BIT,
		eInputAttachmentRead = VK_ACCESS_INPUT_ATTACHMENT_READ_BIT,
		eShaderRead = VK_ACCESS_SHADER_READ_BIT,
		eShaderWrite = VK_ACCESS_SHADER_WRITE_BIT,
		eColorAttachmentRead = VK_ACCESS_COLOR_ATTACHMENT_READ_BIT,
		eColorAttachmentWrite = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
		eDepthStencilAttachmentRead = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT,
		eDepthStencilAttachmentWrite = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
		eTransferRead = VK_ACCESS_TRANSFER_READ_BIT,
		eTransferWrite = VK_ACCESS_TRANSFER_WRITE_BIT,
		eHostRead = VK_ACCESS_HOST_READ_BIT,
		eHostWrite = VK_ACCESS_HOST_WRITE_BIT,
		eMemoryRead = VK_ACCESS_MEMORY_READ_BIT,
		eMemoryWrite = VK_ACCESS_MEMORY_WRITE_BIT,
		eTransformFeedbackWriteEXT = VK_ACCESS_TRANSFORM_FEEDBACK_WRITE_BIT_EXT,
		eTransformFeedbackCounterReadEXT = VK_ACCESS_TRANSFORM_FEEDBACK_COUNTER_READ_BIT_EXT,
		eTransformFeedbackCounterWriteEXT = VK_ACCESS_TRANSFORM_FEEDBACK_COUNTER_WRITE_BIT_EXT,
		eConditionalRenderingReadEXT = VK_ACCESS_CONDITIONAL_RENDERING_READ_BIT_EXT,
		eColorAttachmentReadNoncoherentEXT = VK_ACCESS_COLOR_ATTACHMENT_READ_NONCOHERENT_BIT_EXT,
		eAccelerationStructureReadKHR = VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR,
		eAccelerationStructureWriteKHR = VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR,
		eShadingRateImageReadNV = VK_ACCESS_SHADING_RATE_IMAGE_READ_BIT_NV,
		eFragmentDensityMapReadEXT = VK_ACCESS_FRAGMENT_DENSITY_MAP_READ_BIT_EXT,
		eCommandPreprocessReadNV = VK_ACCESS_COMMAND_PREPROCESS_READ_BIT_NV,
		eCommandPreprocessWriteNV = VK_ACCESS_COMMAND_PREPROCESS_WRITE_BIT_NV,
		eAccelerationStructureReadNV = VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_NV,
		eAccelerationStructureWriteNV = VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_NV
	};

	using AccessFlags = Flags<AccessFlagBits>;

	inline constexpr AccessFlags operator|(AccessFlagBits bit0, AccessFlagBits bit1) noexcept {
		return AccessFlags(bit0) | bit1;
	}

	inline constexpr AccessFlags operator&(AccessFlagBits bit0, AccessFlagBits bit1) noexcept {
		return AccessFlags(bit0) & bit1;
	}

	inline constexpr AccessFlags operator^(AccessFlagBits bit0, AccessFlagBits bit1) noexcept {
		return AccessFlags(bit0) ^ bit1;
	}
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
			vuk::PipelineStageFlags stages;
			vuk::AccessFlags access;
			vuk::ImageLayout layout; // ignored for buffers
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

namespace vuk {
	struct Attachment {
        vuk::Image image;
        vuk::ImageView image_view;
        
		vuk::Extent2D extent;
        vuk::Format format;
        vuk::Samples sample_count = vuk::Samples::e1;
        Clear clear_value;

		static Attachment from_texture(const vuk::Texture& t, Clear clear_value) {
            return Attachment{
                .image = t.image.get(), .image_view = t.view.get(), .extent = {t.extent.width, t.extent.height}, .format = t.format, .sample_count = {t.sample_count}, .clear_value = clear_value};
        }
		static Attachment from_texture(const vuk::Texture& t) {
            return Attachment{
                .image = t.image.get(), .image_view = t.view.get(), .extent = {t.extent.width, t.extent.height}, .format = t.format, .sample_count = {t.sample_count}};
		}
	};

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


	struct RenderGraph {
		std::unique_ptr<arena> arena_;
		RenderGraph();

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
			vuk::ImageLayout layout;
			vuk::AccessFlags access;
			vuk::PipelineStageFlags stage;
		};

		struct AttachmentRPInfo {
			Name name;

			enum class Sizing {
				eAbsolute, eFramebufferRelative
			} sizing;
			vuk::Extent2D::Framebuffer fb_relative;
			vuk::Extent2D extents;
			vuk::Samples samples;

			VkAttachmentDescription description = {};

			Resource::Use initial, final;

			enum class Type {
				eInternal, eExternal, eSwapchain
			} type;

			vuk::ImageView iv;
			vuk::Image image = {};
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
			VkImageMemoryBarrier barrier = {};
			vuk::PipelineStageFlags src;
			vuk::PipelineStageFlags dst;
		};

		struct MemoryBarrier {
			VkMemoryBarrier barrier = {};
			vuk::PipelineStageFlags src;
			vuk::PipelineStageFlags dst;
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
			VkRenderPass handle = {};
			VkFramebuffer framebuffer;
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
		void mark_attachment_internal(Name, vuk::Format, vuk::Extent2D, vuk::Samples, Clear);
		void mark_attachment_internal(Name, vuk::Format, vuk::Extent2D::Framebuffer, vuk::Samples, Clear);
		void mark_attachment_resolve(Name resolved_name, Name ms_name);
		void bind_buffer(Name, vuk::Buffer);
        void bind_attachment(Name, Attachment, Access initial, Access final);
		vuk::ImageUsageFlags compute_usage(const std::vector<vuk::RenderGraph::UseRef, short_alloc<UseRef, 64>>& chain);

		// RG
		void build(vuk::PerThreadContext&);
		void create_attachment(vuk::PerThreadContext&, Name name, RenderGraph::AttachmentRPInfo& attachment_info, vuk::Extent2D extents, vuk::SampleCountFlagBits);
		VkCommandBuffer execute(vuk::PerThreadContext&, std::vector<std::pair<Swapchain*, size_t>> swp_with_index, bool use_secondary_command_buffers);

		BufferInfo get_resource_buffer(Name);
		bool is_resource_image_in_general_layout(Name n, PassInfo* pass_info);

	private:
		void fill_renderpass_info(vuk::RenderGraph::RenderPassInfo& rpass, const size_t& i, vuk::CommandBuffer& cobuf);
	};
}
