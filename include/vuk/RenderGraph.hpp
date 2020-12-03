#pragma once

#include <vector>
#include <string_view>
#include <optional>
#include <functional>
#include "vuk/Hash.hpp"
#include "vuk/vuk_fwd.hpp"
#include "RenderPass.hpp"
#include "vuk/Buffer.hpp"
#include "vuk/Image.hpp"
#include "vuk/Swapchain.hpp"
#include "vuk/MapProxy.hpp"

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

	struct Pass {
		Name name;
		Name executes_on;
		float auxiliary_order = 0.f;
        bool use_secondary_command_buffers = false;

		std::vector<Resource> resources;
		robin_hood::unordered_flat_map<Name, Name> resolves; // src -> dst

		std::function<void(vuk::CommandBuffer&)> execute;
	};

	struct RenderGraph {
		RenderGraph();
		~RenderGraph();

		RenderGraph(RenderGraph&&) noexcept;
		RenderGraph& operator=(RenderGraph&&) noexcept;

		void add_pass(Pass p);

		void append(RenderGraph other);

		void mark_attachment_resolve(Name resolved_name, Name ms_name);
		void bind_attachment_to_swapchain(Name, SwapchainRef swp, Clear);
		void mark_attachment_internal(Name, Format, Extent2D, Samples, Clear);
		void mark_attachment_internal(Name, Format, Extent2D::Framebuffer, Samples, Clear);
		void bind_buffer(Name, Buffer, Access initial, Access final);
		void bind_attachment(Name, Attachment, Access initial, Access final);

		struct ExecutableRenderGraph link(PerThreadContext&) &&;

		// reflection functions
		MapProxy<Name, std::span<const struct UseRef>> get_use_chains();
		MapProxy<Name, struct AttachmentRPInfo&> get_bound_attachments();
		vuk::ImageUsageFlags compute_usage(std::span<const UseRef> chain);

	private:
		struct RGImpl* impl;
		friend struct ExecutableRenderGraph;

		// determine rendergraph inputs and outputs, and resources that are neither
		void build_io();
		// build the graph, assign framebuffers, renderpasses and subpass
		void compile();
	};

	struct ExecutableRenderGraph {
		ExecutableRenderGraph(RenderGraph&&);
		~ExecutableRenderGraph();

		VkCommandBuffer execute(vuk::PerThreadContext&, std::vector<std::pair<Swapchain*, size_t>> swp_with_index);

		struct BufferInfo get_resource_buffer(Name);
		struct AttachmentRPInfo get_resource_image(Name);

		bool is_resource_image_in_general_layout(Name n, struct PassInfo* pass_info);
	private:
		struct RGImpl* impl;

		void create_attachment(PerThreadContext& ptc, Name name, struct AttachmentRPInfo& attachment_info, Extent2D fb_extent, SampleCountFlagBits samples);
		void fill_renderpass_info(struct RenderPassInfo& rpass, const size_t& i, class CommandBuffer& cobuf);
	};
}


inline vuk::ImageResource operator "" _image(const char* name, size_t) {
	return { name };
}

inline vuk::BufferResource operator "" _buffer(const char* name, size_t) {
	return { name };
}

