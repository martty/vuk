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
#include <unordered_set>
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

	struct Resource {
		Name name;
		enum class Type { eBuffer, eImage } type;
		Access ia;
		Name out_name;

		Resource(Name n, Type t, Access ia) : name(n), type(t), ia(ia) {}
		Resource(Name n, Type t, Access ia, Name out_name) : name(n), type(t), ia(ia), out_name(out_name) {}

		bool operator==(const Resource& o) const noexcept {
			return name == o.name;
		}
	};

	QueueResourceUse to_use(Access acc, DomainFlags domain);

	enum class PassType { eUserPass, eClear, eConverge, eConvergeExplicit, eForcedAccess };

	/// @brief Fundamental unit of execution and scheduling. Refers to resources
	struct Pass {
		Name name;
		DomainFlags execute_on = DomainFlagBits::eDevice;

		bool use_secondary_command_buffers = false;

		std::vector<Resource> resources;
		std::vector<std::pair<Name, Name>> resolves; // src -> dst

		std::function<void(CommandBuffer&)> execute;
		std::byte* arguments; // internal use
		PassType type = PassType::eUserPass;
	};

	// declare these specializations for GCC
	template<>
	ConstMapIterator<Name, std::span<const struct UseRef>>::~ConstMapIterator();
	template<>
	ConstMapIterator<Name, const struct AttachmentInfo&>::~ConstMapIterator();
	template<>
	ConstMapIterator<Name, const struct BufferInfo&>::~ConstMapIterator();

	struct RenderGraph : std::enable_shared_from_this<RenderGraph> {
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

		/// @brief Add an alias for a resource
		/// @param new_name Additional name to refer to the resource
		/// @param old_name Old name used to refere to the resource
		void add_alias(Name new_name, Name old_name);

		/// @brief Diverge image. subrange is available as subrange_name afterwards.
		void diverge_image(Name whole_name, Subrange::Image subrange, Name subrange_name);

		/// @brief Reconverge image. Prevents diverged use moving before pre_diverge or after post_diverge.
		void converge_image(Name pre_diverge, Name post_diverge);

		/// @brief Reconverge image from named parts. Prevents diverged use moving before pre_diverge or after post_diverge.
		void converge_image_explicit(std::span<Name> pre_diverge, Name post_diverge);

		/// @brief Add a resolve operation from the image resource `ms_name` that consumes `resolved_name_src` and produces `resolved_name_dst`
		/// This is only supported for color images.
		/// @param resolved_name_src Image resource name consumed (single-sampled)
		/// @param resolved_name_dst Image resource name created (single-sampled)
		/// @param ms_name Image resource to resolve from (multisampled)
		void resolve_resource_into(Name resolved_name_src, Name resolved_name_dst, Name ms_name);

		/// @brief Clear image attachment
		/// @param image_name_in Name of the image resource to clear
		/// @param image_name_out Name of the cleared image resource
		/// @param clear_value Value used for the clear
		/// @param subrange Range of image cleared
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

		/// @brief Attach an image to the given name
		/// @param name Name of the resource to attach to
		/// @param image_attachment ImageAttachment to attach
		/// @param clear_value Value used for the clear
		/// @param initial Access to the resource prior to this rendergraph
		/// @param final Desired Access to the resource after this rendergraph
		void attach_and_clear_image(Name name, ImageAttachment image_attachment, Clear clear_value, Access initial = eNone, Access final = eNone);

		/// @brief Attach a future to the given name
		/// @param name Name of the resource to attach to
		/// @param future Future to be attached into this rendergraph
		void attach_in(Name name, Future future);

		/// @brief Attach multiple futures - the names are matched to future bound names
		/// @param futures Futures to be attached into this rendergraph
		void attach_in(std::span<Future> futures);

		void inference_rule(Name target, std::function<void(const struct InferenceContext& ctx, ImageAttachment& ia)>);
		void inference_rule(Name target, std::function<void(const struct InferenceContext& ctx, Buffer& ia)>);

		/// @brief Compute all the unconsumed resource names and return them as Futures
		std::vector<Future> split();

		Name name;

	private:
		struct RGImpl* impl;
		friend struct ExecutableRenderGraph;
		friend struct Compiler;
		friend struct RGCImpl;

		// future support functions
		friend class Future;
		void attach_out(Name, Future& fimg, DomainFlags dst_domain, Subrange subrange);

		void detach_out(Name, Future& fimg);

		Name get_temporary_name();
	};

	struct InferenceContext {
		const ImageAttachment& get_image_attachment(Name name) const;
		const Buffer& get_buffer(Name name) const;

		struct ExecutableRenderGraph* erg;
		Name prefix;
	};

	using IARule = std::function<void(const struct InferenceContext& ctx, ImageAttachment& ia)>;

	// builtin inference rules for convenience

	/// @brief Inference target has the same extent as the source
	IARule same_extent_as(Name inference_source);

	/// @brief Inference target has the same width & height as the source
	IARule same_2D_extent_as(Name inference_source);

	/// @brief Inference target has the same format as the source
	IARule same_format_as(Name inference_source);

	/// @brief Inference target has the same shape(extent, layers, levels) as the source
	IARule same_shape_as(Name inference_source);

	/// @brief Inference target is similar to(same shape, same format, same sample count) the source
	IARule image_similar_to(Name inference_source);

	struct Compiler {
		Compiler();
		~Compiler();

		/// @brief Build the graph, assign framebuffers, renderpasses and subpasses
		///	link automatically calls this, only needed if you want to use the reflection functions
		/// @param compile_options CompileOptions controlling compilation behaviour
		void compile(std::span<std::shared_ptr<RenderGraph>> rgs, const RenderGraphCompileOptions& compile_options);

		/// @brief Use this RenderGraph and create an ExecutableRenderGraph
		/// @param compile_options CompileOptions controlling compilation behaviour
		struct ExecutableRenderGraph link(std::span<std::shared_ptr<RenderGraph>> rgs, const RenderGraphCompileOptions& compile_options);

		// reflection functions

		/// @brief retrieve usages of resource in the RenderGraph
		MapProxy<Name, std::span<const struct UseRef>> get_use_chains();
		/// @brief retrieve bound image attachments in the RenderGraph
		MapProxy<Name, const struct AttachmentInfo&> get_bound_attachments();
		/// @brief retrieve bound buffers in the RenderGraph
		MapProxy<Name, const struct BufferInfo&> get_bound_buffers();
		/// @brief compute ImageUsageFlags for given use chain
		static ImageUsageFlags compute_usage(std::span<const UseRef> chain);

		/// @brief Dump the pass dependency graph in graphviz format
		std::string dump_graph();

	private:
		struct RGCImpl* impl;

		friend struct ExecutableRenderGraph;
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
		ExecutableRenderGraph(Compiler&);
		~ExecutableRenderGraph();

		ExecutableRenderGraph(const ExecutableRenderGraph&) = delete;
		ExecutableRenderGraph& operator=(const ExecutableRenderGraph&) = delete;

		ExecutableRenderGraph(ExecutableRenderGraph&&) noexcept;
		ExecutableRenderGraph& operator=(ExecutableRenderGraph&&) noexcept;

		Result<SubmitBundle> execute(Allocator&, std::vector<std::pair<Swapchain*, size_t>> swp_with_index);

		Result<struct BufferInfo, RenderGraphException> get_resource_buffer(Name, struct PassInfo*);
		Result<struct AttachmentInfo, RenderGraphException> get_resource_image(Name, struct PassInfo*);

		Result<bool, RenderGraphException> is_resource_image_in_general_layout(Name n, struct PassInfo* pass_info);

		Name resolve_name(Name, struct PassInfo*) const noexcept;

	private:
		struct RGCImpl* impl;

		void create_attachment(Context& ptc, struct AttachmentInfo& attachment_info);
		void fill_renderpass_info(struct RenderPassInfo& rpass, const size_t& i, class CommandBuffer& cobuf);
		Result<SubmitInfo> record_single_submit(Allocator&, std::span<RenderPassInfo> rpis, DomainFlagBits domain);

		friend struct InferenceContext;
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
	struct hash<vuk::Subrange::Image> {
		size_t operator()(vuk::Subrange::Image const& x) const noexcept {
			size_t h = 0;
			hash_combine(h, x.base_layer, x.base_level, x.layer_count, x.level_count);
			return h;
		}
	};
}; // namespace std
