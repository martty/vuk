#pragma once

#include <stdio.h>
#include <vector>
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
#include "robin_hood.h"

namespace vuk {
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
        unsigned hash_src_name;
        unsigned hash_use_name;
		enum class Type { eBuffer, eImage } type;
		Access ia;
		struct Use {
			vuk::PipelineStageFlags stages;
			vuk::AccessFlags access;
			vuk::ImageLayout layout; // ignored for buffers
		};

		Resource(Name n, Type t, Access ia): src_name(n), use_name(n), type(t), ia(ia) {
            hash_src_name = hash_use_name = hash::fnv1a::hash(src_name.data(), src_name.size(), hash::fnv1a::default_offset_basis);
		}
		Resource(Name src, Name use, Type t, Access ia) : src_name(src), use_name(use), type(t), ia(ia) {
            hash_src_name = hash::fnv1a::hash(src_name.data(), src_name.size(), hash::fnv1a::default_offset_basis);
            hash_use_name = hash::fnv1a::hash(use_name.data(), use_name.size(), hash::fnv1a::default_offset_basis);
		}

		bool operator==(const Resource& o) const noexcept {
			return (hash_use_name == o.hash_use_name && hash_src_name == o.hash_src_name);
		}
	};

	Resource::Use to_use(Access acc);

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
        bool use_secondary_command_buffers = false;

		std::vector<Resource> resources;
		robin_hood::unordered_flat_map<Name, Name> resolves; // src -> dst

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
		PassInfo(arena&, Pass&&);

		Pass pass;

		size_t render_pass_index;
		uint32_t subpass;

		robin_hood::unordered_flat_set<Resource> inputs;
		robin_hood::unordered_flat_set<Resource> outputs;

		robin_hood::unordered_flat_set<Resource> global_inputs;
		robin_hood::unordered_flat_set<Resource> global_outputs;

		bool is_head_pass = false;
		bool is_tail_pass = false;
	};


	struct RenderGraph {
		std::unique_ptr<arena> arena_;
		RenderGraph();

		std::vector<PassInfo> passes;

		std::vector<PassInfo*, short_alloc<PassInfo*, 8>> head_passes;
		std::vector<PassInfo*, short_alloc<PassInfo*, 8>> tail_passes;

		robin_hood::unordered_flat_map<Name, Name> aliases;

		robin_hood::unordered_flat_set<Resource> global_inputs;
		robin_hood::unordered_flat_set<Resource> global_outputs;
        std::vector<Resource, short_alloc<Resource, 16>> global_io;

		struct UseRef {
			Resource::Use use;
			PassInfo* pass = nullptr;
		};

		robin_hood::unordered_flat_map<Name, std::vector<UseRef, short_alloc<UseRef, 64>>> use_chains;

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
            bool use_secondary_command_buffers;
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
			passes.emplace_back(*arena_, std::move(p));
		}

		// determine rendergraph inputs and outputs, and resources that are neither
		void build_io();

		void build();

		// RGscaffold
		robin_hood::unordered_flat_map<Name, AttachmentRPInfo> bound_attachments;
		robin_hood::unordered_flat_map<Name, BufferInfo> bound_buffers;
		void bind_attachment_to_swapchain(Name, Swapchain* swp, Clear);
		void mark_attachment_internal(Name, vuk::Format, vuk::Extent2D, vuk::Samples, Clear);
		void mark_attachment_internal(Name, vuk::Format, vuk::Extent2D::Framebuffer, vuk::Samples, Clear);
		void mark_attachment_resolve(Name resolved_name, Name ms_name);
		void bind_buffer(Name, Buffer, Access initial, Access final);
        void bind_attachment(Name, Attachment, Access initial, Access final);
		vuk::ImageUsageFlags compute_usage(const std::vector<vuk::RenderGraph::UseRef, short_alloc<UseRef, 64>>& chain);

		// RG
		void build(vuk::PerThreadContext&);
		void create_attachment(vuk::PerThreadContext&, Name name, RenderGraph::AttachmentRPInfo& attachment_info, vuk::Extent2D extents, vuk::SampleCountFlagBits);
		VkCommandBuffer execute(vuk::PerThreadContext&, std::vector<std::pair<Swapchain*, size_t>> swp_with_index);

		BufferInfo get_resource_buffer(Name);
		bool is_resource_image_in_general_layout(Name n, PassInfo* pass_info);

	private:
		void fill_renderpass_info(vuk::RenderGraph::RenderPassInfo& rpass, const size_t& i, vuk::CommandBuffer& cobuf);
	};
}
