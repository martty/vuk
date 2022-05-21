#pragma once

#include "vuk/Buffer.hpp"
#include "vuk/Future.hpp"
#include "vuk/Hash.hpp"
#include "vuk/Image.hpp"
#include "vuk/ImageAttachment.hpp"
#include "vuk/MapProxy.hpp"
#include "vuk/Result.hpp"
#include "vuk/Swapchain.hpp"
#include "vuk/vuk_fwd.hpp"

#include <functional>
#include <optional>
#include <span>
#include <string_view>
#include <vector>

namespace vuk {
	struct FutureBase;
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

			Resource operator()(Access ia, Format, Dimension2D, Samples);
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
		PipelineStageFlags stages;
		AccessFlags access;
		ImageLayout layout; // ignored for buffers
	};

	struct AttachmentRPInfo {
		Name name;

		ImageAttachment attachment = {};

		VkAttachmentDescription description = {};

		ResourceUse initial, final;

		enum class Type { eInternal, eExternal, eSwapchain } type;

		// swapchain for swapchain
		Swapchain* swapchain = nullptr;

		std::optional<Clear> clear_value;

		bool is_resolve_dst = false;
		FutureBase* attached_future = nullptr;
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
		ImageAttachment ici;
		BufferCreateInfo bci;
		union Subrange {
			struct Image {
				uint32_t base_layer = 0;
				uint32_t base_level = 0;

				uint32_t layer_count = VK_REMAINING_ARRAY_LAYERS;
				uint32_t level_count = VK_REMAINING_MIP_LEVELS;

				constexpr bool operator==(const Image& o) const noexcept {
					return base_level == o.base_level && level_count == o.level_count && base_layer == o.base_layer && layer_count == o.layer_count;
				}

				Name combine_name(Name prefix) const;

				bool operator<(const Image& o) const noexcept {
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
		Resource(Name n, Type t, Access ia, Format fmt, Dimension2D dim, Samples samp, Name out_name) :
		    name(n),
		    type(t),
		    ia(ia),
		    out_name(out_name),
		    is_create(true),
		    ici{ .extent = dim, .format = fmt, .sample_count = samp } {}

		bool operator==(const Resource& o) const noexcept {
			return name == o.name;
		}
	};

	ResourceUse to_use(Access acc);

	/// @brief Fundamental unit of execution and scheduling. Refers to resources
	struct Pass {
		Name name;
		DomainFlags execute_on = DomainFlagBits::eDevice;

		bool use_secondary_command_buffers = false;

		std::vector<Resource> resources;
		std::unordered_map<Name, Name> resolves; // src -> dst

		std::unique_ptr<FutureBase> wait;
		FutureBase* signal = nullptr;

		std::function<void(CommandBuffer&)> execute;
		std::vector<std::byte> arguments; // internal use
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
		RenderGraph(Name name);
		~RenderGraph();

		RenderGraph(const RenderGraph&) = delete;
		RenderGraph& operator=(const RenderGraph&) = delete;

		RenderGraph(RenderGraph&&) noexcept;
		RenderGraph& operator=(RenderGraph&&) noexcept;

		/// @brief Add a pass to the rendergraph
		/// @param pass the Pass to add to the RenderGraph
		void add_pass(Pass pass);

		/// @brief Append the other RenderGraph onto this one (by copy or move of passes and attachments)
		/// @param subgraph_name the prefix used for the names in other
		void append(Name subgraph_name, RenderGraph other);

		/// @brief Add an alias for a resource
		/// @param new_name Additional name to refer to the resource
		/// @param old_name Old name used to refere to the resource
		void add_alias(Name new_name, Name old_name);

		/// @brief Reconverge image. Prevents diverged use moving before pre_diverge or after post_diverge.
		void converge_image(Name pre_diverge, Name post_diverge);

		/// @brief Add a resolve operation from the image resource `ms_name` that consumes `resolved_name_src` and produces `resolved_name_dst`
		/// @param resolved_name_src Image resource name consumed (single-sampled)
		/// @param resolved_name_dst Image resource name created (single-sampled)
		/// @param ms_name Image resource to resolve from (multisampled)
		void resolve_resource_into(Name resolved_name_src, Name resolved_name_dst, Name ms_name);

		/// @brief Clear image attachment
		/// @param image_name_in Name of the image resource to clear
		/// @param image_name_out Name of the cleared image resource
		/// @param clear_value Value used for the clear
		void clear_image(Name image_name_in, Name image_name_out, Clear clear_value);

		/// @brief Attach a swapchain to the given name
		/// @param name Name of the resource to attach to
		void attach_swapchain(Name name, SwapchainRef swp);

		/// @brief Attach a buffer to the given name
		/// @param name Name of the resource to attach to
		/// @param buffer Buffer to attach
		/// @param initial Access to the resource prior to this rendergraph
		/// @param final Desired Access to the resource after this rendergraph
		void attach_buffer(Name name, Buffer buffer, Access initial = eNone, Access final = eNone);

		/// @brief Attach an image to the given name
		/// @param name Name of the resource to attach to
		/// @param image_attachment ImageAttachment to attach
		/// @param initial Access to the resource prior to this rendergraph
		/// @param final Desired Access to the resource after this rendergraph
		void attach_image(Name name, ImageAttachment image_attachment, Access initial = eNone, Access final = eNone);

		/// @brief Attach a future to the given name
		/// @param name Name of the resource to attach to
		/// @param future Future to be consumed into this rendergraph
		void attach_in(Name name, Future&& future);

		/// @brief Attach multiple futures - the names are matched to future bound names
		/// @param futures Futures to be consumed into this rendergraph
		void attach_in(std::span<Future> futures);

		/// @brief Control compilation options when compiling the rendergraph
		struct CompileOptions {
			/// @brief reorder passes according to dependencies
			bool reorder_passes = true;
			/// @brief check that pass ordering does not violate resource constraints (not needed when reordering passes)
			bool check_pass_ordering = false;
		};

		/// @brief Consume this RenderGraph and create an ExecutableRenderGraph
		/// @param compile_options CompileOptions controlling compilation behaviour
		struct ExecutableRenderGraph link(const CompileOptions& compile_options) &&;

		// reflection functions

		/// @brief Build the graph, assign framebuffers, renderpasses and subpasses
		///	link automatically calls this, only needed if you want to use the reflection functions
		/// @param compile_options CompileOptions controlling compilation behaviour
		void compile(const CompileOptions& compile_options);

		/// @brief retrieve usages of resource in the RenderGraph
		MapProxy<Name, std::span<const struct UseRef>> get_use_chains();
		/// @brief retrieve bound image attachments in the RenderGraph
		MapProxy<Name, const struct AttachmentRPInfo&> get_bound_attachments();
		/// @brief retrieve bound buffers in the RenderGraph
		MapProxy<Name, const struct BufferInfo&> get_bound_buffers();
		/// @brief compute ImageUsageFlags for given use chains
		static ImageUsageFlags compute_usage(std::span<const UseRef> chain);

	private:
		struct RGImpl* impl;
		Name name;
		friend struct ExecutableRenderGraph;

		/// @brief Check if this rendergraph is valid.
		/// \throws RenderGraphException
		void validate();

		// determine rendergraph inputs and outputs, and resources that are neither
		void build_io();

		void schedule_intra_queue(std::span<struct PassInfo> passes, const RenderGraph::CompileOptions& compile_options);

		// future support functions
		friend class Future;
		void attach_out(Name, Future& fimg, DomainFlags dst_domain);
	};

	struct SubmitInfo {
		std::vector<std::pair<DomainFlagBits, uint64_t>> relative_waits;
		std::vector<VkCommandBuffer> command_buffers;
		std::vector<FutureBase*> future_signals;
		std::vector<SwapchainRef> used_swapchains;
	};

	struct SubmitBatch {
		DomainFlagBits domain;
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
