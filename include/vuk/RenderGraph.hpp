#pragma once

#include "../src/RenderPass.hpp"
#include "vuk/Allocator.hpp"
#include "vuk/Buffer.hpp"
#include "vuk/Context.hpp"
#include "vuk/Hash.hpp"
#include "vuk/Image.hpp"
#include "vuk/MapProxy.hpp"
#include "vuk/Result.hpp"
#include "vuk/Swapchain.hpp"
#include "vuk/vuk_fwd.hpp"
#include <functional>
#include <optional>
#include <string_view>
#include <vector>

namespace vuk {
	template<class T>
	struct Future;
	struct Resource;

	namespace detail {
		struct BufferResourceInputOnly;

		struct BufferResource {
			Name name;

			BufferResourceInputOnly operator>>(Access ba);
		};

		struct ImageResourceInputOnly;

		struct ImageResource {
			Name name;

			Resource operator()(Access ia, Format, Dimension2D, Samples, Clear);
			ImageResourceInputOnly operator>>(Access ia);
		};

		struct ImageResourceInputOnly {
			Name name;
			Access ba;

			Resource operator>>(Name output);
			operator Resource();
		};

		struct BufferResourceInputOnly {
			Name name;
			Access ba;

			Resource operator>>(Name output);
			operator Resource();
		};
	} // namespace detail

	struct ResourceUse {
		vuk::PipelineStageFlags stages;
		vuk::AccessFlags access;
		vuk::ImageLayout layout; // ignored for buffers
	};

	struct AttachmentRPInfo {
		Name name;

		ImageAttachment attachment = {};

		VkAttachmentDescription description = {};

		ResourceUse initial, final;

		enum class Type { eInternal, eExternal, eSwapchain } type;

		// swapchain for swapchain
		Swapchain* swapchain = nullptr;

		// optionally set
		bool should_clear = false;

		bool is_resolve_dst = false;
		FutureBase* attached_future = nullptr;
	};

	struct PartialImageAlias {
		Name src;
		Name dst;
		uint32_t base_level;
		uint32_t level_count;
		uint32_t base_layer;
		uint32_t layer_count;
	};

	struct BufferInfo {
		Name name;

		ResourceUse initial;
		ResourceUse final;

		Buffer buffer;
		FutureBase* attached_future = nullptr;
	};

	struct Resource {
		Name name;
		enum class Type { eBuffer, eImage } type;
		Access ia;
		Name out_name;
		bool is_create = false;
		AttachmentRPInfo ici;
		BufferCreateInfo bci;
		union Subrange {
			struct Image {
				uint32_t base_layer = 0;
				uint32_t base_level = 0;

				uint32_t layer_count = VK_REMAINING_ARRAY_LAYERS;
				uint32_t level_count = VK_REMAINING_MIP_LEVELS;

				constexpr bool operator==(const Image& o) const {
					return base_level == o.base_level && level_count == o.level_count && base_layer == o.base_layer && layer_count == o.layer_count;
				}

				Name combine_name(Name prefix) {
					std::string suffix = std::string(prefix.to_sv());
					suffix += "[" + std::to_string(base_layer) + ":" + std::to_string(base_layer + layer_count - 1) + "]";
					suffix += "[" + std::to_string(base_level) + ":" + std::to_string(base_level + level_count - 1) + "]";
					return Name(suffix.c_str());
				}

				bool operator<(const Image& o) const {
					return std::tie(base_layer, base_level, layer_count, level_count) < std::tie(o.base_layer, o.base_level, o.layer_count, o.level_count);
				}
			} image = {};
			struct Buffer {
				uint64_t offset = 0;
				uint64_t size = VK_WHOLE_SIZE;
			} buffer;
		} subrange = {};

		Resource(Name n, Type t, Access ia) : name(n), type(t), ia(ia) {}
		Resource(Name n, Type t, Access ia, Name out_name) : name(n), type(t), ia(ia), out_name(out_name) {}
		Resource(Name n, Type t, Access ia, Format fmt, Dimension2D dim, Samples samp, Clear cv, Name out_name) :
		    name(n),
		    type(t),
		    ia(ia),
		    is_create(true),
		    ici{ .attachment = { .extent = dim, .format = fmt, .sample_count = samp, .clear_value = cv }, .description{ .format = (VkFormat)fmt } },
		    out_name(out_name) {}

		bool operator==(const Resource& o) const noexcept {
			return name == o.name;
		}
	};
} // namespace vuk

namespace std {
	template<>
	struct hash<vuk::Resource::Subrange::Image> {
		size_t operator()(vuk::Resource::Subrange::Image const& x) const noexcept {
			size_t h = 0;
			hash_combine(h, x.base_layer, x.base_level, x.layer_count, x.level_count);
			return h;
		}
	};
}; // namespace std

namespace vuk {
	ResourceUse to_use(Access acc);

	struct Pass {
		Name name;
		DomainFlags execute_on = DomainFlagBits::eGraphicsOnGraphics; // TODO: default or not?

		bool use_secondary_command_buffers = false;

		std::vector<Resource> resources;
		robin_hood::unordered_flat_map<Name, Name> resolves; // src -> dst

		std::unique_ptr<FutureBase> wait;
		FutureBase* signal;

		std::function<void(vuk::CommandBuffer&)> execute;
	};

	// declare these specializations for GCC
	template<>
	ConstMapIterator<Name, std::span<const struct UseRef>>::~ConstMapIterator();
	template<>
	ConstMapIterator<Name, const struct AttachmentRPInfo&>::~ConstMapIterator();
	template<>
	ConstMapIterator<Name, const struct BufferInfo&>::~ConstMapIterator();

	struct RenderGraph {
		RenderGraph();
		~RenderGraph();

		RenderGraph(const RenderGraph&) = delete;
		RenderGraph& operator=(const RenderGraph&) = delete;

		RenderGraph(RenderGraph&&) noexcept;
		RenderGraph& operator=(RenderGraph&&) noexcept;

		/// @brief Add a pass to the rendergraph
		/// @param the Pass to add to the RenderGraph
		void add_pass(Pass);

		// append the other RenderGraph onto this one
		// will copy or move passes and attachments
		void append(Name subgraph_name, RenderGraph other);

		/// @brief Add an alias for a resource
		/// @param new_name
		/// @param old_name
		void add_alias(Name new_name, Name old_name);

		/// @brief Reconverge image. Prevents diverged use moving before pre_diverge or after post_diverge.
		void converge_image(Name pre_diverge, Name post_diverge);

		/// @brief Add a resolve operation from the image resource `ms_name` that consumes `resolved_name_src` and produces `resolved_name_dst`
		/// @param resolved_name_src Image resource name consumed (single-sampled)
		/// @param resolved_name_dst Image resource name created (single-sampled)
		/// @param ms_name Image resource to resolve from (multisampled)
		void resolve_resource_into(Name resolved_name_src, Name resolved_name_dst, Name ms_name);

		/// @brief Attach a swapchain to the given name
		void attach_swapchain(Name name, SwapchainRef swp, Clear);
		
		/// @brief Attach a buffer to the given name
		void attach_buffer(Name name, Buffer, Access initial, Access final);

		/// @brief Attach an image to the given name
		void attach_image(Name name, ImageAttachment, Access initial, Access final);

		/// @brief Attach a future of an image to the given name
		void attach_in(Name name, Future<ImageAttachment>&& fimg, Access final);

		/// @brief Attach a future of a buffer to the given name
		void attach_in(Name name, Future<Buffer>&& fimg, Access final);

		/// @brief Request the rendergraph 
		void attach_managed(Name name, Format format, Dimension2D dimension, Samples samples, Clear clear_value);

		/// @brief Control compilation options when compiling the rendergraph
		struct CompileOptions {
			bool reorder_passes = true;       // reorder passes according to resources
			bool check_pass_ordering = false; // check that pass ordering does not violate resource constraints (not needed when reordering passes)
		};

		/// @brief Consume this RenderGraph and create an ExecutableRenderGraph
		struct ExecutableRenderGraph link(Context& ctx, const CompileOptions& compile_options) &&;

		// reflection functions

		/// @brief Build the graph, assign framebuffers, renderpasses and subpasses
		///	link automatically calls this, only needed if you want to use the reflection functions
		void compile(const CompileOptions& compile_options);

		MapProxy<Name, std::span<const struct UseRef>> get_use_chains();
		MapProxy<Name, const struct AttachmentRPInfo&> get_bound_attachments();
		MapProxy<Name, const struct BufferInfo&> get_bound_buffers();
		static vuk::ImageUsageFlags compute_usage(std::span<const UseRef> chain);

	private:
		struct RGImpl* impl;
		friend struct ExecutableRenderGraph;

		/// @brief Check if this rendergraph is valid.
		/// \throws RenderGraphException
		void validate();

		// determine rendergraph inputs and outputs, and resources that are neither
		void build_io();

		void schedule_intra_queue(std::span<struct PassInfo> passes, const vuk::RenderGraph::CompileOptions& compile_options);

		// future support functions
		friend struct Future<ImageAttachment>;
		friend struct Future<Buffer>;
		void attach_out(Name, Future<ImageAttachment>& fimg, DomainFlags dst_domain);
		void attach_out(Name, Future<Buffer>& fbuf, DomainFlags dst_domain);
	};

	struct SubmitInfo {
		std::vector<std::pair<DomainFlagBits, uint64_t>> relative_waits;
		std::vector<VkCommandBuffer> command_buffers;
		std::vector<FutureBase*> future_signals;
		std::vector<SwapchainRef> used_swapchains;
	};

	struct SubmitBatch {
		vuk::DomainFlagBits domain;
		std::vector<SubmitInfo> submits;
	};

	struct SubmitBundle {
		std::vector<SubmitBatch> batches;
	};

	struct ExecutableRenderGraph {
		ExecutableRenderGraph(RenderGraph&&);
		~ExecutableRenderGraph();

		ExecutableRenderGraph(const ExecutableRenderGraph&) = delete;
		ExecutableRenderGraph& operator=(const ExecutableRenderGraph&) = delete;

		ExecutableRenderGraph(ExecutableRenderGraph&&) noexcept;
		ExecutableRenderGraph& operator=(ExecutableRenderGraph&&) noexcept;

		Result<SubmitBundle> execute(Allocator&, std::vector<std::pair<Swapchain*, size_t>> swp_with_index);

		Result<struct BufferInfo, RenderGraphException> get_resource_buffer(Name, struct PassInfo*);
		Result<struct AttachmentRPInfo, RenderGraphException> get_resource_image(Name, struct PassInfo*);

		Result<bool, RenderGraphException> is_resource_image_in_general_layout(Name n, struct PassInfo* pass_info);

		Name resolve_name(Name, struct PassInfo*) const noexcept;

	private:
		struct RGImpl* impl;

		void create_attachment(Context& ptc, Name name, struct AttachmentRPInfo& attachment_info, Extent2D fb_extent, SampleCountFlagBits samples);
		void fill_renderpass_info(struct RenderPassInfo& rpass, const size_t& i, class CommandBuffer& cobuf);
		Result<SubmitInfo> record_single_submit(Allocator&, std::span<RenderPassInfo> rpis, DomainFlagBits domain);
	};
} // namespace vuk

inline vuk::detail::ImageResource operator"" _image(const char* name, size_t) {
	return { name };
}

inline vuk::detail::BufferResource operator"" _buffer(const char* name, size_t) {
	return { name };
}
