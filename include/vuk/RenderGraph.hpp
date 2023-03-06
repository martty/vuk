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
		uint32_t id = 0;
		QualifiedName name;
		Name original_name;
		enum class Type { eBuffer, eImage } type;
		Access ia;
		QualifiedName out_name;
		struct RenderGraph* foreign = nullptr;
		int32_t reference = 0;
		bool promoted_to_general = false;

		Resource(Name n, Type t, Access ia) : name{ Name{}, n }, type(t), ia(ia) {}
		Resource(Name n, Type t, Access ia, Name out_name) : name{ Name{}, n }, type(t), ia(ia), out_name{ Name{}, out_name } {}
		Resource(RenderGraph* foreign, QualifiedName n, Type t, Access ia) : name{ n }, type(t), ia(ia), foreign(foreign) {}
		Resource(uint32_t id, Type t, Access ia) : id(id), type(t), ia(ia) {}

		bool operator==(const Resource& o) const noexcept {
			return name == o.name;
		}
	};

	QueueResourceUse to_use(Access acc, DomainFlags domain);

	enum class PassType { eUserPass, eClear, eResolve, eDiverge, eConverge, eForcedAccess };

	/// @brief Fundamental unit of execution and scheduling. Refers to resources
	struct Pass {
		Name name;
		DomainFlags execute_on = DomainFlagBits::eDevice;

		std::vector<Resource> resources;

		std::function<void(CommandBuffer&)> execute;
		std::byte* arguments; // internal use
		PassType type = PassType::eUserPass;
	};

	// declare these specializations for GCC
	template<>
	ConstMapIterator<QualifiedName, const struct AttachmentInfo&>::~ConstMapIterator();
	template<>
	ConstMapIterator<QualifiedName, const struct BufferInfo&>::~ConstMapIterator();

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
		void add_pass(Pass pass, source_location location = source_location::current());

		/// @brief Add an alias for a resource
		/// @param new_name Additional name to refer to the resource
		/// @param old_name Old name used to refere to the resource
		void add_alias(Name new_name, Name old_name);

		/// @brief Diverge image. subrange is available as subrange_name afterwards.
		void diverge_image(Name whole_name, Subrange::Image subrange, Name subrange_name);

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
		void attach_buffer(Name name, Buffer buffer, Access initial = eNone);

		/// @brief Attach a buffer to be allocated from the specified allocator
		/// @param name Name of the resource to attach to
		/// @param buffer Buffer to attach
		/// @param allocator Allocator the Buffer will be allocated from
		/// @param initial Access to the resource prior to this rendergraph
		void attach_buffer_from_allocator(Name name, Buffer buffer, Allocator allocator, Access initial = eNone);

		/// @brief Attach an image to the given name
		/// @param name Name of the resource to attach to
		/// @param image_attachment ImageAttachment to attach
		/// @param initial Access to the resource prior to this rendergraph
		void attach_image(Name name, ImageAttachment image_attachment, Access initial = eNone);

		/// @brief Attach an image to be allocated from the specified allocator
		/// @param name Name of the resource to attach to
		/// @param image_attachment ImageAttachment to attach
		/// @param buffer Buffer to attach
		/// @param initial Access to the resource prior to this rendergraph
		void attach_image_from_allocator(Name name, ImageAttachment image_attachment, Allocator allocator, Access initial = eNone);

		/// @brief Attach an image to the given name
		/// @param name Name of the resource to attach to
		/// @param image_attachment ImageAttachment to attach
		/// @param clear_value Value used for the clear
		/// @param initial Access to the resource prior to this rendergraph
		void attach_and_clear_image(Name name, ImageAttachment image_attachment, Clear clear_value, Access initial = eNone);

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

		/// @brief Mark resources to be released from the rendergraph with future access
		/// @param name Name of the resource to be released
		/// @param final Access after the rendergraph
		void release(Name name, Access final);

		/// @brief Mark resource to be released from the rendergraph for presentation
		/// @param name Name of the resource to be released
		void release_for_present(Name name);

		/// @brief Name of the rendergraph
		Name name;

		uint32_t id = 0;

	private:
		struct RGImpl* impl;
		friend struct ExecutableRenderGraph;
		friend struct Compiler;
		friend struct RGCImpl;

		// future support functions
		friend class Future;
		void attach_out(QualifiedName, Future& fimg, DomainFlags dst_domain);

		void detach_out(QualifiedName, Future& fimg);

		Name get_temporary_name();
	};

	template<size_t N>
	struct StringLiteral {
		constexpr StringLiteral(const char (&str)[N]) {
			std::copy_n(str, N, value);
		}

		char value[N];
	};

	template<Access acc, StringLiteral N, class T = void>
	struct IA {
		static constexpr Access access = acc;
		using base = Image;
		using attach = ImageAttachment;
		static constexpr StringLiteral identifier = N;

		Name name;
	};

	// from: https://stackoverflow.com/a/28213747
	template<typename T>
	struct closure_traits {};

#define REM_CTOR(...) __VA_ARGS__
#define SPEC(cv, var, is_var)                                                                                                                                  \
	template<typename C, typename R, typename... Args>                                                                                                           \
	struct closure_traits<R (C::*)(Args... REM_CTOR var) cv> {                                                                                                   \
		using arity = std::integral_constant<std::size_t, sizeof...(Args)>;                                                                                        \
		using is_variadic = std::integral_constant<bool, is_var>;                                                                                                  \
		using is_const = std::is_const<int cv>;                                                                                                                    \
                                                                                                                                                               \
		using result_type = R;                                                                                                                                     \
                                                                                                                                                               \
		template<std::size_t i>                                                                                                                                    \
		using arg = typename std::tuple_element<i, std::tuple<Args...>>::type;                                                                                     \
		static constexpr size_t count = sizeof...(Args);                                                                                                           \
		using types = std::tuple<Args...>;                                                                                                                         \
	};

	SPEC(const, (, ...), 1)
	SPEC(const, (), 0)
	SPEC(, (, ...), 1)
	SPEC(, (), 0)

#undef SPEC
#undef REM_CTOR

	template<int i, typename T>
	struct drop {
		using type = T;
	};

	template<int i, typename T>
	using drop_t = typename drop<i, T>::type;

	template<typename T>
	struct drop<0, T> {
		using type = T;
	};

	template<int i, typename T, typename... Ts>
	requires(i > 0)
	struct drop<i, std::tuple<T, Ts...>> {
		using type = drop_t<i - 1, std::tuple<Ts...>>;
	};

	template<class T>
	struct TypedFuture {
		Future future;
	};

	template<typename Tuple>
	struct TupleMap;

	template<class IA>
	Resource to_resource() {
		return Resource(Name(typeid(IA).name()), Resource::Type::eImage, IA::access);
	}

	template<class IA>
	Resource to_resource_out() {
		return Resource(Name{}, Resource::Type::eImage, IA::access, Name(typeid(IA).name()));
	}

	template<class U, class T>
	void attach_one(std::shared_ptr<RenderGraph>& rg, T&& arg) {
		rg->attach_in(Name(typeid(U).name()), arg.future);
	}

	template<typename... T>
	struct TupleMap<std::tuple<T...>> {

		using ret_tuple = std::tuple<TypedFuture<typename T::base>...>;

		template<class Ret, class F>
		static auto make_lam(Name name, F&& body) {
			std::shared_ptr<RenderGraph> rg = std::make_shared<RenderGraph>();
			Pass p;
			p.name = name;
			// we need to walk return types and match them to input types
			p.resources = { to_resource<T>()... };
			TupleMap<Ret>::fill_out(p);
			// we need TE execution for this
			// p.execute = std::function(body);
			rg->add_pass(p);
			return [=](TypedFuture<typename T::base>&&...args) mutable -> typename TupleMap<Ret>::ret_tuple{
				(attach_one<T, TypedFuture<typename T::base>>(rg, std::move(args)), ...);
				return TupleMap<Ret>::make_ret(rg);
			};
		}

		static auto make_ret(std::shared_ptr<RenderGraph> rg) {
			return std::make_tuple(TypedFuture<typename T::base>{ Future{ rg, Name(typeid(T).name()).append("+") } }...);
		}

		static auto fill_out(Pass& p ) {
			auto out_res_names = { Name(typeid(T).name())... };
			for (auto& n : out_res_names) {
				for (auto& r : p.resources) {
					if (n == r.name.name) {
						r.out_name.name = n.append("+");
						break;
					}
				}
			}
		}
	};
	

	template<class F>
	auto make_pass(Name name, F&& body) {
		using traits = closure_traits<decltype(&F::operator())>;
		return TupleMap<drop_t<1, typename traits::types>>::template make_lam<typename traits::result_type, F>(name, std::forward<F>(body));
	}

	struct InferenceContext {
		const ImageAttachment& get_image_attachment(Name name) const;
		const Buffer& get_buffer(Name name) const;

		struct ExecutableRenderGraph* erg;
		Name prefix;
	};

	using IARule = std::function<void(const struct InferenceContext& ctx, ImageAttachment& ia)>;
	using BufferRule = std::function<void(const struct InferenceContext& ctx, Buffer& buffer)>;

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
	IARule similar_to(Name inference_source);

	/// @brief Inference target is the same size as the source
	BufferRule same_size_as(Name inference_source);

	struct Compiler {
		Compiler();
		~Compiler();

		/// @brief Build the graph, assign framebuffers, renderpasses and subpasses
		///	link automatically calls this, only needed if you want to use the reflection functions
		/// @param compile_options CompileOptions controlling compilation behaviour
		Result<void> compile(std::span<std::shared_ptr<RenderGraph>> rgs, const RenderGraphCompileOptions& compile_options);

		/// @brief Use this RenderGraph and create an ExecutableRenderGraph
		/// @param compile_options CompileOptions controlling compilation behaviour
		Result<struct ExecutableRenderGraph> link(std::span<std::shared_ptr<RenderGraph>> rgs, const RenderGraphCompileOptions& compile_options);

		// reflection functions

		/// @brief retrieve usages of resources in the RenderGraph
		std::span<struct ChainLink*> get_use_chains() const;
		/// @brief retrieve bound image attachments in the RenderGraph
		MapProxy<QualifiedName, const struct AttachmentInfo&> get_bound_attachments();
		/// @brief retrieve bound buffers in the RenderGraph
		MapProxy<QualifiedName, const struct BufferInfo&> get_bound_buffers();

		/// @brief compute ImageUsageFlags for given use chain
		ImageUsageFlags compute_usage(const struct ChainLink* chain);
		/// @brief Get the image attachment heading this use chain
		const struct AttachmentInfo& get_chain_attachment(const struct ChainLink* chain);
		/// @brief Get the last name that references this chain (may not exist)
		std::optional<QualifiedName> get_last_use_name(const struct ChainLink* chain);

		/// @brief Dump the pass dependency graph in graphviz format
		std::string dump_graph();

	private:
		struct RGCImpl* impl;

		// internal passes
		Result<void> inline_rgs(std::span<std::shared_ptr<RenderGraph>> rgs);
		void queue_inference();
		void pass_partitioning();
		void resource_linking();
		void renderpass_assignment();

		friend struct ExecutableRenderGraph;
	};

	struct SubmitInfo {
		std::vector<std::pair<DomainFlagBits, uint64_t>> relative_waits;
		std::vector<std::pair<DomainFlagBits, uint64_t>> absolute_waits;
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

		QualifiedName resolve_name(Name, struct PassInfo*) const noexcept;

	private:
		struct RGCImpl* impl;

		void create_attachment(Context& ptc, struct AttachmentInfo& attachment_info);
		void fill_renderpass_info(struct RenderPassInfo& rpass, const size_t& i, class CommandBuffer& cobuf);
		Result<SubmitInfo> record_single_submit(Allocator&, std::span<PassInfo*> passes, DomainFlagBits domain);

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
