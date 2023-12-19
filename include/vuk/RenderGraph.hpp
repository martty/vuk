#pragma once

#include "vuk/Buffer.hpp"
#include "vuk/Future.hpp"
#include "vuk/Hash.hpp"
#include "vuk/IR.hpp"
#include "vuk/Image.hpp"
#include "vuk/ImageAttachment.hpp"
#include "vuk/MapProxy.hpp"
#include "vuk/Result.hpp"
#include "vuk/Swapchain.hpp"
#include "vuk/vuk_fwd.hpp"

#include <deque>
#include <functional>
#include <optional>
#include <span>
#include <string_view>
#include <type_traits>
#include <unordered_set>
#include <vector>

#if defined(__clang__) or defined(__GNUC__)
#define VUK_IA(access, ...)        vuk::Arg<vuk::ImageAttachment, access, decltype([]() {}), __VA_ARGS__>
#define VUK_BA(access, ...)        vuk::Arg<vuk::Buffer, access, decltype([]() {}), __VA_ARGS__>
#define VUK_ARG(type, access, ...) vuk::Arg<type, access, decltype([]() {}), __VA_ARGS__>
#else
namespace vuk {
	template<size_t I>
	struct tag_type {};
}; // namespace vuk
#define VUK_IA(access, ...)        vuk::Arg<vuk::ImageAttachment, access, vuk::tag_type<__COUNTER__>, __VA_ARGS__>
#define VUK_BA(access, ...)        vuk::Arg<vuk::Buffer, access, vuk::tag_type<__COUNTER__>, __VA_ARGS__>
#define VUK_ARG(type, access, ...) vuk::Arg<type, access, vuk::tag_type<__COUNTER__>, __VA_ARGS__>
#endif

namespace vuk {
	QueueResourceUse to_use(Access acc, DomainFlags domain);

	// declare these specializations for GCC
	template<>
	ConstMapIterator<QualifiedName, const struct AttachmentInfo&>::~ConstMapIterator();
	template<>
	ConstMapIterator<QualifiedName, const struct BufferInfo&>::~ConstMapIterator();

	template<size_t N>
	struct StringLiteral {
		constexpr StringLiteral(const char (&str)[N]) {
			std::copy_n(str, N, value);
		}

		char value[N];
	};

	template<class Type, Access acc, class UniqueT, StringLiteral N = "">
	struct Arg {
		using type = Type;
		static constexpr Access access = acc;

		static constexpr StringLiteral identifier = N;

		Type* ptr;

		Ref src;
		Ref def;

		operator const Type&() const noexcept
		  requires(!std::is_array_v<Type>)
		{
			return *ptr;
		}

		const Type* operator->() const noexcept
		  requires(!std::is_array_v<Type>)
		{
			return ptr;
		}

		size_t size() const noexcept
		  requires std::is_array_v<Type>
		{
			return def.type()->array.size;
		}

		auto operator[](size_t index) const noexcept
		  requires std::is_array_v<Type>
		{
			return (*ptr)[index];
		}
	};
	/*
	template<class Arg>
	struct arg_kind {};

	template<Access acc, class UniqueT, StringLiteral N>
	struct arg_kind<Arg<ImageAttachment, acc, UniqueT, N>>{
	  static constexpr Type::TypeKind kind = Type::IMAGE_TY;
	};

	template<Access acc, class UniqueT, StringLiteral N>
	struct arg_kind<Arg<Buffer, acc, UniqueT, N>> {
	  static constexpr Type::TypeKind kind = Type::BUFFER_TY;
	};*/

	using IARule = std::function<void(const struct InferenceContext& ctx, ImageAttachment& ia)>;
	using BufferRule = std::function<void(const struct InferenceContext& ctx, Buffer& buffer)>;

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

#define DELAYED_ERROR(error)                                                                                                                                   \
private:                                                                                                                                                       \
	template<typename T>                                                                                                                                         \
	struct Error {                                                                                                                                               \
		static_assert(std::is_same_v<T, T> == false, error);                                                                                                       \
		using type = T;                                                                                                                                            \
	};                                                                                                                                                           \
	using error_t = typename Error<void>::type;                                                                                                                  \
                                                                                                                                                               \
public:

	// END OF DELAYED_ERROR

	template<typename... Ts>
	struct type_list {
		static constexpr size_t length = sizeof...(Ts);
		using type = type_list<Ts...>;
	};

	template<typename>
	struct tuple_to_typelist {
		DELAYED_ERROR("tuple_to_typelist expects a std::tuple");
	};

	template<typename... Ts>
	struct tuple_to_typelist<std::tuple<Ts...>> {
		using type = type_list<Ts...>;
	};

	template<typename T>
	using tuple_to_typelist_t = typename tuple_to_typelist<T>::type;

	template<typename>
	struct typelist_to_tuple {
		DELAYED_ERROR("typelist_to_tuple expects a type_list");
	};

	template<typename... Ts>
	struct typelist_to_tuple<type_list<Ts...>> {
		using type = std::tuple<Ts...>;
	};

	template<typename T>
	using typelist_to_tuple_t = typename typelist_to_tuple<T>::type;

	template<typename T, typename List>
	struct prepend_type;

	template<typename T, typename... Ts>
	struct prepend_type<T, type_list<Ts...>> {
		using type = type_list<T, Ts...>;
	};

	template<size_t... N>
	struct index_list {
		static constexpr size_t length = sizeof...(N);
		using type = index_list<N...>;
	};

	template<size_t N, typename List>
	struct prepend_index;

	template<size_t N, size_t... Ns>
	struct prepend_index<N, index_list<Ns...>> {
		using type = index_list<N, Ns...>;
	};

	template<typename T1, typename T2, size_t N = 0>
	struct intersect_type_lists;

	template<typename... T1, size_t N>
	struct intersect_type_lists<type_list<T1...>, type_list<>, N> {
		using type = type_list<>;
		using indices = index_list<>;
	};

	template<typename... T1, typename T2_Head, typename... T2, size_t N>
	struct intersect_type_lists<type_list<T1...>, type_list<T2_Head, T2...>, N> {
	private:
		using rest = intersect_type_lists<type_list<T1...>, type_list<T2...>, N + 1>;
		using rest_type = typename rest::type;
		using rest_indices = typename rest::indices;

		static constexpr bool condition = (std::is_same_v<T1, T2_Head> || ...);

	public:
		using type = std::conditional_t<condition, prepend_type<T2_Head, rest_type>, rest_type>;
		using indices = std::conditional_t<condition, prepend_index<N, rest_indices>, rest_indices>;
	};

	template<typename Tuple, typename... Ts>
	auto make_subtuple(const Tuple& tuple, type_list<Ts...>) -> typelist_to_tuple_t<type_list<Ts...>> {
		return typelist_to_tuple_t<type_list<Ts...>>(std::get<Ts>(tuple)...);
	}

	template<size_t... Ns>
	auto make_indices(index_list<Ns...> indices) {
		return std::array{ Ns... };
	}

	template<typename T1, typename T2>
	auto intersect_tuples(const T1& tuple) {
		using T1_list = tuple_to_typelist_t<T1>;
		using T2_list = tuple_to_typelist_t<T2>;
		using intersection = intersect_type_lists<T1_list, T2_list>;
		using intersection_type = typename intersection::type;
		using intersection_indices = typename intersection::indices;
		auto subtuple = make_subtuple(tuple, intersection_type{});
		auto indices = make_indices(intersection_indices{});
		return std::pair{ indices, subtuple };
	}

	template<typename Tuple>
	struct TupleMap;

	template<typename... T>
	void pack_typed_tuple(std::span<void*> src, std::span<void*> meta, CommandBuffer& cb, void* dst) {
		std::tuple<CommandBuffer*, T...>& tuple = *new (dst) std::tuple<CommandBuffer*, T...>;
		std::get<0>(tuple) = &cb;
#define X(n)                                                                                                                                                   \
	if constexpr ((sizeof...(T)) > n) {                                                                                                                          \
		auto& elem = std::get<n + 1>(tuple);                                                                                                                       \
		elem.ptr = reinterpret_cast<decltype(elem.ptr)>(src[n]);                                                                                                   \
		elem.def = *reinterpret_cast<Ref*>(meta[n]);                                                                                                                \
	}
		X(0)
		X(1)
		X(2)
		X(3)
		X(4)
		X(5)
		X(6)
		X(7)
		X(8)
		X(9)
		X(10)
		X(11)
		X(12)
		X(13)
		X(14)
		X(15)
		static_assert(sizeof...(T) <= 16);
#undef X
	}; // namespace vuk

	template<typename... T>
	void unpack_typed_tuple(const std::tuple<T...>& src, std::span<void*> dst) {
#define X(n)                                                                                                                                                   \
	if constexpr ((sizeof...(T)) > n) {                                                                                                                          \
		dst[n] = std::get<n>(src).ptr;                                                                                                                             \
	}
		X(0)
		X(1)
		X(2)
		X(3)
		X(4)
		X(5)
		X(6)
		X(7)
		X(8)
		X(9)
		X(10)
		X(11)
		X(12)
		X(13)
		X(14)
		X(15)
		static_assert(sizeof...(T) <= 16);
#undef X
	}

	template<typename>
	struct is_tuple : std::false_type {};

	template<typename... T>
	struct is_tuple<std::tuple<T...>> : std::true_type {};

	template<typename... T>
	static auto make_ret(std::shared_ptr<RG> rg, Node* node, const std::tuple<T...>& us) {
		if constexpr (sizeof...(T) > 0) {
			size_t i = 0;
			return std::make_tuple(TypedFuture<typename T::type>{ rg, { node, i++ }, std::get<T>(us).def }...);
		}
	}

	template<typename... T>
	static auto fill_arg_ty(RG& rg, const std::tuple<T...>& args, std::vector<Type*>& arg_types) {
		(arg_types.emplace_back(rg.make_imbued_ty(std::get<T>(args).src.type(), T::access)), ...);
	}

	template<typename... T>
	static auto fill_ret_ty(RG& rg, std::array<size_t, sizeof...(T)> idxs, const std::tuple<T...>& args, std::vector<Type*>& ret_types) {
		(ret_types.emplace_back(rg.make_aliased_ty(std::get<T>(args).src.type(), 0)), ...);
		for (auto i = 0; i < ret_types.size(); i++) {
			ret_types[i]->aliased.ref_idx = idxs[i];
		}
	}

	template<typename... T>
	struct TupleMap<std::tuple<T...>> {
		using ret_tuple = std::tuple<TypedFuture<typename T::type>...>;

		template<class Ret, class F>
		static auto make_lam(Name name, F&& body) {
			auto callback = [typed_cb = std::move(body)](CommandBuffer& cb, std::span<void*> args, std::span<void*> meta, std::span<void*> rets) {
				// we do type recovery here -> convert untyped args to typed ones
				alignas(alignof(std::tuple<CommandBuffer&, T...>)) char storage[sizeof(std::tuple<CommandBuffer&, T...>)];
				pack_typed_tuple<T...>(args, meta, cb, storage);
				auto typed_ret = std::apply(typed_cb, *reinterpret_cast<std::tuple<CommandBuffer&, T...>*>(storage));
				// now we erase these types
				if constexpr (!is_tuple<Ret>::value) {
					rets[0] = typed_ret.ptr;
				} else {
					unpack_typed_tuple(typed_ret, rets);
				}
			};

			// when this function is called, we weave in this call into the IR
			return [untyped_cb = std::move(callback), name](TypedFuture<typename T::type>... args) mutable {
				auto& first = [](auto& first, auto&...) -> auto& {
					return first;
				}(args...);
				auto& rgp = first.get_render_graph();
				RG& rg = *rgp.get();
				[](auto& first, auto&... rest) {
					(first.get_render_graph()->subgraphs.push_back(rest.get_render_graph()), ...);
				}(args...);

				std::vector<Type*> arg_types;
				std::tuple arg_tuple_as_a = { T{ args.operator->(), args.get_head(), args.get_def() }... };
				fill_arg_ty(rg, arg_tuple_as_a, arg_types);

				std::vector<Type*> ret_types;
				if constexpr (is_tuple<Ret>::value) {
					auto [idxs, ret_tuple] = intersect_tuples<std::tuple<T...>, Ret>(arg_tuple_as_a);
					fill_ret_ty(rg, idxs, ret_tuple, ret_types);
				} else if constexpr (!std::is_same_v<Ret, void>) {
					auto [idxs, ret_tuple] = intersect_tuples<std::tuple<T...>, std::tuple<Ret>>(arg_tuple_as_a);
					fill_ret_ty(rg, idxs, ret_tuple, ret_types);
				}
				auto opaque_fn_ty = rg.make_opaque_fn_ty(arg_types, ret_types, vuk::DomainFlagBits::eAny, untyped_cb);
				opaque_fn_ty->debug_info = new TypeDebugInfo{ .name = name.c_str() };
				auto opaque_fn = rg.make_declare_fn(opaque_fn_ty);
				Node* node = rg.make_call(opaque_fn, args.get_head()...);
				if constexpr (is_tuple<Ret>::value) {
					auto [idxs, ret_tuple] = intersect_tuples<std::tuple<T...>, Ret>(arg_tuple_as_a);
					return make_ret(rgp, node, ret_tuple);
				} else if constexpr (!std::is_same_v<Ret, void>) {
					auto [idxs, ret_tuple] = intersect_tuples<std::tuple<T...>, std::tuple<Ret>>(arg_tuple_as_a);
					return std::get<0>(make_ret(rgp, node, ret_tuple));
				}
			};
		}
	};

	template<class F>
	[[nodiscard]] auto make_pass(Name name, F&& body) {
		using traits = closure_traits<decltype(&F::operator())>;
		return TupleMap<drop_t<1, typename traits::types>>::template make_lam<typename traits::result_type, F>(name, std::forward<F>(body));
	}

	[[nodiscard]] inline TypedFuture<ImageAttachment> declare_ia(Name name, ImageAttachment ia = {}) {
		std::shared_ptr<RG> rg = std::make_shared<RG>();
		Ref ref = rg->make_declare_image(ia);
		rg->name_outputs(ref.node, { name.c_str() });
		return { rg, ref, ref };
	}

	[[nodiscard]] inline TypedFuture<Buffer> declare_buf(Name name, Buffer buf = {}) {
		std::shared_ptr<RG> rg = std::make_shared<RG>();
		Ref ref = rg->make_declare_buffer(buf);
		rg->name_outputs(ref.node, { name.c_str() });
		return { rg, ref, ref };
	}

	template<class T, class... Args>
	[[nodiscard]] inline TypedFuture<T[]> declare_array(Name name, const TypedFuture<T>& arg, Args... args) {
		auto rg = arg.get_render_graph();
		[&rg](auto&... rest) {
			(rg->subgraphs.push_back(rest.get_render_graph()), ...);
		}(args...);
		std::array refs = { arg.get_head(), args.get_head()... };
		std::array defs = { arg.get_def(), args.get_def()... };
		Ref ref = rg->make_declare_array(refs[0].type(), refs, defs);
		rg->name_outputs(ref.node, { name.c_str() });
		return { rg, ref, ref };
	}

	[[nodiscard]] inline TypedFuture<ImageAttachment> clear(TypedFuture<ImageAttachment> in, Clear clear_value) {
		auto& rg = in.get_render_graph();
		return in.transmute(rg->make_clear_image(in.get_head(), clear_value));
	}

	[[nodiscard]] inline TypedFuture<SwapchainRenderBundle> import_swapchain(SwapchainRenderBundle bundle) {
		std::shared_ptr<RG> rg = std::make_shared<RG>();
		Ref ref = rg->make_import_swapchain(bundle);
		return { rg, ref, ref };
	}

	[[nodiscard]] inline TypedFuture<ImageAttachment> acquire_next_image(Name name, TypedFuture<SwapchainRenderBundle> in) {
		auto& rg = in.get_render_graph();
		Ref ref = rg->make_acquire_next_image(in.get_head());
		rg->name_outputs(ref.node, { name.c_str() });
		return in.transmute<ImageAttachment>(ref);
	}

	struct InferenceContext {
		const ImageAttachment& get_image_attachment(Name name) const;
		const Buffer& get_buffer(Name name) const;

		struct ExecutableRenderGraph* erg;
		Name prefix;
	};

	// builtin inference rules for convenience

	/// @brief Inference target has the same extent as the source
	IARule same_extent_as(Name inference_source);
	IARule same_extent_as(TypedFuture<Image> inference_source);

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

		/// @brief Build the graph, assign framebuffers, render passes and subpasses
		///	link automatically calls this, only needed if you want to use the reflection functions
		/// @param compile_options CompileOptions controlling compilation behaviour
		Result<void> compile(std::span<std::shared_ptr<RG>> rgs, const RenderGraphCompileOptions& compile_options);

		/// @brief Use this RenderGraph and create an ExecutableRenderGraph
		/// @param compile_options CompileOptions controlling compilation behaviour
		Result<struct ExecutableRenderGraph> link(std::span<std::shared_ptr<RG>> rgs, const RenderGraphCompileOptions& compile_options);

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
		void queue_inference();
		void pass_partitioning();
		void resource_linking();
		void render_pass_assignment();

		friend struct ExecutableRenderGraph;
	};

	struct SubmitInfo {
		std::vector<std::pair<DomainFlagBits, uint64_t>> relative_waits;
		std::vector<std::pair<DomainFlagBits, uint64_t>> absolute_waits;
		std::vector<VkCommandBuffer> command_buffers;
		std::vector<Signal*> future_signals;
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

		Result<bool, RenderGraphException> is_resource_image_in_general_layout(const NameReference&, struct PassInfo* pass_info);

	private:
		struct RGCImpl* impl;

		void fill_render_pass_info(struct RenderPassInfo& rpass, const size_t& i, class CommandBuffer& cobuf);
		Result<SubmitInfo> record_single_submit(Allocator&, std::span<struct ScheduledItem*> passes, DomainFlagBits domain);

		friend struct InferenceContext;
	};

} // namespace vuk

namespace std {
	template<>
	struct hash<vuk::Subrange::Image> {
		size_t operator()(vuk::Subrange::Image const& x) const noexcept {
			size_t h = 0;
			hash_combine(h, x.base_layer, x.base_level, x.layer_count, x.level_count);
			return h;
		}
	};

	template<>
	struct hash<vuk::Ref> {
		size_t operator()(vuk::Ref const& x) const noexcept {
			size_t h = 0;
			hash_combine(h, x.node, x.index);
			return h;
		}
	};
}; // namespace std
