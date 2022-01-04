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
#include "vuk/Context.hpp"

namespace vuk {
	template<class T> struct Future;
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
	}

	struct ResourceUse {
		vuk::Access original = vuk::eNone;
		vuk::PipelineStageFlags stages;
		vuk::AccessFlags access;
		vuk::ImageLayout layout; // ignored for buffers
	};

	struct AttachmentRPInfo {
		Name name;

		vuk::Dimension2D extents;
		vuk::Samples samples;

		VkAttachmentDescription description = {};

		ResourceUse initial, final;

		enum class Type {
			eInternal, eExternal, eSwapchain
		} type;

		vuk::ImageView iv;
		vuk::Image image = {};
		// swapchain for swapchain
		Swapchain* swapchain = nullptr;

		// optionally set
		bool should_clear = false;
		Clear clear_value;

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
		AttachmentRPInfo ici;
		BufferCreateInfo bci;

		Resource(Name n, Type t, Access ia) : name(n), type(t), ia(ia) {}
		Resource(Name n, Type t, Access ia, Name out_name) : name(n), type(t), ia(ia), out_name(out_name) {}
		Resource(Name n, Type t, Access ia, Format fmt, Dimension2D dim, Samples samp, Clear cv, Name out_name) : name(n), type(t), ia(ia), is_create(true), ici{ .extents = dim, .samples = samp, .description{.format = (VkFormat)fmt}, .clear_value = cv}, out_name(out_name) {}

		bool operator==(const Resource& o) const noexcept {
			return name == o.name;
		}
	};

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
	template<> ConstMapIterator<Name, std::span<const struct UseRef>>::~ConstMapIterator();
	template<> ConstMapIterator<Name, const struct AttachmentRPInfo&>::~ConstMapIterator();
	template<> ConstMapIterator<Name, const struct BufferInfo&>::~ConstMapIterator();

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
		void append(Name subgraph_name, RenderGraph other);

		/// @brief Add an alias for a resource
		/// @param new_name 
		/// @param old_name 
		void add_alias(Name new_name, Name old_name);

		/// @brief Add a resolve operation from the image resource `ms_name` to image_resource `resolved_name`
		/// @param resolved_name 
		/// @param ms_name 
		// TODO: docs
		void resolve_resource_into(Name resolved_name_src, Name resolved_name_dst, Name ms_name);

		void attach_swapchain(Name, SwapchainRef swp, Clear);
		void attach_buffer(Name, Buffer, Access initial, Access final);
		void attach_image(Name, ImageAttachment, Access initial, Access final);

		void attach_in(Name, Future<ImageAttachment>&& fimg, Access final);
		void attach_in(Name, Future<Buffer>&& fimg, Access final);

		void attach_out(Name, Future<ImageAttachment>& fimg);
		void attach_out(Name, Future<Buffer>& fbuf);

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
}

inline vuk::detail::ImageResource operator "" _image(const char* name, size_t) {
	return { name };
}

inline vuk::detail::BufferResource operator "" _buffer(const char* name, size_t) {
	return { name };
}
