#pragma once

#include <vector>
#include <string_view>
#include <optional>
#include <functional>
#include "vuk/Hash.hpp"
#include "vuk/vuk_fwd.hpp"
#include "../src/RenderPass.hpp"
#include "vuk/Buffer.hpp"
#include "vuk/Image.hpp"
#include "vuk/Swapchain.hpp"
#include "vuk/MapProxy.hpp"
#include "vuk/Result.hpp"
#include "vuk/Allocator.hpp"

namespace vuk {
	struct Resource;

	namespace detail {
		struct BufferResource {
			Name name;

			Resource operator()(Access ba);
		};
		struct ImageResource {
			Name name;

			Resource operator()(Access ia);
		};
	}

	struct Resource {
		Name name;
		enum class Type { eBuffer, eImage } type;
		Access ia;
		struct Use {
			vuk::PipelineStageFlags stages;
			vuk::AccessFlags access;
			vuk::ImageLayout layout; // ignored for buffers
		};

		Resource(Name n, Type t, Access ia) : name(n), type(t), ia(ia) {}

		bool operator==(const Resource& o) const noexcept {
			return name == o.name;
		}
	};

	Resource::Use to_use(Access acc);

	// TODO: infer this from a smart IV
	struct ImageAttachment {
		vuk::Image image;
		vuk::ImageView image_view;

		vuk::Extent2D extent;
		vuk::Format format;
		vuk::Samples sample_count = vuk::Samples::e1;
		Clear clear_value;

		static ImageAttachment from_texture(const vuk::Texture& t, Clear clear_value) {
			return ImageAttachment{
				.image = t.image.get(), .image_view = t.view.get(), .extent = {t.extent.width, t.extent.height}, .format = t.format, .sample_count = {t.sample_count}, .clear_value = clear_value };
		}
		static ImageAttachment from_texture(const vuk::Texture& t) {
			return ImageAttachment{
				.image = t.image.get(), .image_view = t.view.get(), .extent = {t.extent.width, t.extent.height}, .format = t.format, .sample_count = {t.sample_count} };
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

	// declare these specializations for GCC
	template<> ConstMapIterator<Name, std::span<const struct UseRef>>::~ConstMapIterator();
	template<> ConstMapIterator<Name, const struct AttachmentRPInfo&>::~ConstMapIterator();

	struct RenderGraph {
		RenderGraph();
		~RenderGraph();

		RenderGraph(const RenderGraph&) = delete;
		RenderGraph& operator=(const RenderGraph&) = delete;

		RenderGraph(RenderGraph&&) noexcept;
		RenderGraph& operator=(RenderGraph&&) noexcept;

		/// @brief 
		/// @param the Pass to add to the RenderGraph 
		void add_pass(Pass);

		// append the other RenderGraph onto this one
		// will copy or move passes and attachments
		void append(RenderGraph other);

		/// @brief Add an alias for a resource
		/// @param new_name 
		/// @param old_name 
		void add_alias(Name new_name, Name old_name);

		/// @brief Add a resolve operation from the image resource `ms_name` to image_resource `resolved_name`
		/// @param resolved_name 
		/// @param ms_name 
		void resolve_resource_into(Name resolved_name, Name ms_name);

		void attach_swapchain(Name, SwapchainRef swp, Clear);
		void attach_buffer(Name, Buffer, Access initial, Access final);
		void attach_image(Name, ImageAttachment, Access initial, Access final);

		void attach_managed(Name, Format, Dimension2D, Samples, Clear);

		struct CompileOptions {
			bool reorder_passes = true; // reorder passes according to resources
			bool check_pass_ordering = false; // check that pass ordering does not violate resource constraints (not needed when reordering passes)
		};

		/// @brief Consume this RenderGraph and create an ExecutableRenderGraph
		struct ExecutableRenderGraph link(Context& ctx, const CompileOptions& compile_options)&&;

		// reflection functions
 
		/// @brief Build the graph, assign framebuffers, renderpasses and subpasses
		///	link automatically calls this, only needed if you want to use the reflection functions
		void compile(const CompileOptions& compile_options);

		MapProxy<Name, std::span<const struct UseRef>> get_use_chains();
		MapProxy<Name, const struct AttachmentRPInfo&> get_bound_attachments();
		static vuk::ImageUsageFlags compute_usage(std::span<const UseRef> chain);
	private:
		struct RGImpl* impl;
		friend struct ExecutableRenderGraph;

		/// @brief Check if this rendergraph is valid.
		/// \throws RenderGraphException
		void validate();

		// determine rendergraph inputs and outputs, and resources that are neither
		void build_io();
	};

	struct ExecutableRenderGraph {
		ExecutableRenderGraph(RenderGraph&&);
		~ExecutableRenderGraph();

		ExecutableRenderGraph(const ExecutableRenderGraph&) = delete;
		ExecutableRenderGraph& operator=(const ExecutableRenderGraph&) = delete;

		ExecutableRenderGraph(ExecutableRenderGraph&&) noexcept;
		ExecutableRenderGraph& operator=(ExecutableRenderGraph&&) noexcept;

		Result<Unique<struct HLCommandBuffer>> execute(Context&, class Allocator&, std::vector<std::pair<Swapchain*, size_t>> swp_with_index);

		Result<struct BufferInfo, RenderGraphException> get_resource_buffer(Name);
		Result<struct AttachmentRPInfo, RenderGraphException> get_resource_image(Name);

		Result<bool, RenderGraphException> is_resource_image_in_general_layout(Name n, struct PassInfo* pass_info);
	private:
		struct RGImpl* impl;

		void create_attachment(Context& ptc, Name name, struct AttachmentRPInfo& attachment_info, Extent2D fb_extent, SampleCountFlagBits samples);
		void fill_renderpass_info(struct RenderPassInfo& rpass, const size_t& i, class CommandBuffer& cobuf);
	};
}

inline vuk::detail::ImageResource operator "" _image(const char* name, size_t) {
	return { name };
}

inline vuk::detail::BufferResource operator "" _buffer(const char* name, size_t) {
	return { name };
}
