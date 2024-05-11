#pragma once

#include "vuk/Buffer.hpp"
#include "vuk/Hash.hpp"
#include "vuk/IR.hpp"
#include "vuk/ImageAttachment.hpp"
#include "vuk/Result.hpp"
#include "vuk/SourceLocation.hpp"
#include "vuk/Value.hpp"
#include "vuk/runtime/vk/Image.hpp"
#include "vuk/runtime/vk/VkSwapchain.hpp"
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
#define VUK_IA(access, ...)        vuk::Arg<vuk::ImageAttachment, access, decltype([]() {}) __VA_OPT__(,) __VA_ARGS__>
#define VUK_BA(access, ...)        vuk::Arg<vuk::Buffer, access, decltype([]() {}) __VA_OPT__(,) __VA_ARGS__>
#define VUK_ARG(type, access, ...) vuk::Arg<type, access, decltype([]() {}) __VA_OPT__(,) __VA_ARGS__>
#else
namespace vuk {
	template<size_t I>
	struct tag_type {};
}; // namespace vuk
#define VUK_IA(access, ...)        vuk::Arg<vuk::ImageAttachment, access, vuk::tag_type<__COUNTER__>, __VA_ARGS__>
#define VUK_BA(access, ...)        vuk::Arg<vuk::Buffer, access, vuk::tag_type<__COUNTER__>, __VA_ARGS__>
#define VUK_ARG(type, access, ...) vuk::Arg<type, access, vuk::tag_type<__COUNTER__>, __VA_ARGS__>
#endif

#define VUK_CALLSTACK SourceLocationAtFrame _pscope = VUK_HERE_AND_NOW(), SourceLocationAtFrame _scope = VUK_HERE_AND_NOW()
#define VUK_CALL      (_pscope != _scope ? _scope.parent = &_pscope, _scope : _scope)

namespace vuk {
	ResourceUse to_use(Access acc);

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

	template<typename T1, typename T2>
	struct intersect_type_lists;

	template<typename... T1>
	struct intersect_type_lists<type_list<T1...>, type_list<>> {
		using type = type_list<>;
	};

	template<typename... T1, typename T2_Head, typename... T2>
	struct intersect_type_lists<type_list<T1...>, type_list<T2_Head, T2...>> {
	private:
		using rest = intersect_type_lists<type_list<T1...>, type_list<T2...>>;
		using rest_type = typename rest::type;

		static constexpr bool condition = (std::is_same_v<T1, T2_Head> || ...);

	public:
		using type = std::conditional_t<condition, typename prepend_type<T2_Head, rest_type>::type, rest_type>;
	};

	template<typename Tuple, typename... Ts>
	auto make_subtuple(const Tuple& tuple, type_list<Ts...>) -> typelist_to_tuple_t<type_list<Ts...>> {
		return typelist_to_tuple_t<type_list<Ts...>>(std::get<Ts>(tuple)...);
	}

	//	https://devblogs.microsoft.com/oldnewthing/20200629-00/?p=103910
	template<typename T, typename Tuple>
	struct tuple_element_index_helper;

	template<typename T>
	struct tuple_element_index_helper<T, std::tuple<>> {
		static constexpr std::size_t value = 0;
	};

	template<typename T, typename... Rest>
	struct tuple_element_index_helper<T, std::tuple<T, Rest...>> {
		static constexpr std::size_t value = 0;
		using RestTuple = std::tuple<Rest...>;
		static_assert(tuple_element_index_helper<T, RestTuple>::value == std::tuple_size_v<RestTuple>, "type appears more than once in tuple");
	};

	template<typename T, typename First, typename... Rest>
	struct tuple_element_index_helper<T, std::tuple<First, Rest...>> {
		using RestTuple = std::tuple<Rest...>;
		static constexpr std::size_t value = 1 + tuple_element_index_helper<T, RestTuple>::value;
	};

	template<typename T, typename Tuple>
	struct tuple_element_index {
		static constexpr std::size_t value = tuple_element_index_helper<T, Tuple>::value;
		static_assert(value < std::tuple_size_v<Tuple>, "type does not appear in tuple");
	};

	template<typename T, typename Tuple>
	inline constexpr std::size_t tuple_element_index_v = tuple_element_index<T, Tuple>::value;

	template<typename Tuple, typename... Ts>
	auto make_indices(const Tuple& tuple, type_list<Ts...>) {
		return std::array{ tuple_element_index_v<Ts, Tuple>... };
	}

	template<typename T1, typename T2>
	auto intersect_tuples(const T1& tuple) {
		using T1_list = tuple_to_typelist_t<T1>;
		using T2_list = tuple_to_typelist_t<T2>;
		using intersection = intersect_type_lists<T1_list, T2_list>;
		using intersection_type = typename intersection::type;
		auto subtuple = make_subtuple(tuple, intersection_type{});
		auto indices = make_indices(tuple, intersection_type{});
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
		elem.def = *reinterpret_cast<Ref*>(meta[n]);                                                                                                               \
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
	static auto make_ret(std::shared_ptr<ExtNode> extnode, const std::tuple<T...>& us) {
		if constexpr (sizeof...(T) > 0) {
			size_t i = 0;
			// FIXME: I think this is well defined but seems like compilers don't agree on the result
#if VUK_COMPILER_MSVC
			return std::tuple{ Value<typename T::type>{ ExtRef{ extnode, Ref{ extnode->get_node(), sizeof...(T) - (++i) } } }... };
#else
			return std::tuple{ Value<typename T::type>{ ExtRef{ extnode, Ref{ extnode->get_node(), i++ } } }... };
#endif
		}
	}

	template<typename... T>
	static auto fill_arg_ty(const std::tuple<T...>& args, std::vector<Type*>& arg_types) {
		(arg_types.emplace_back(current_module.make_imbued_ty(std::get<T>(args).src.type(), T::access)), ...);
	}

	template<typename... T>
	static auto fill_ret_ty(std::array<size_t, sizeof...(T)> idxs, const std::tuple<T...>& args, std::vector<Type*>& ret_types) {
		(ret_types.emplace_back(current_module.make_aliased_ty(Type::stripped(std::get<T>(args).src.type()), 0)), ...);
		for (auto i = 0; i < ret_types.size(); i++) {
			ret_types[i]->aliased.ref_idx = idxs[i];
		}
	}

	inline auto First = [](auto& first, auto&...) -> auto& {
		return first;
	};

	template<typename... T>
	struct TupleMap<std::tuple<T...>> {
		using ret_tuple = std::tuple<Value<typename T::type>...>;

		template<class Ret, class F>
		static auto make_lam(Name name, F&& body, SchedulingInfo scheduling_info, VUK_CALLSTACK) {
			auto callback = [typed_cb = std::move(body)](CommandBuffer& cb, std::span<void*> args, std::span<void*> meta, std::span<void*> rets) mutable {
				// we do type recovery here -> convert untyped args to typed ones
				alignas(alignof(std::tuple<CommandBuffer&, T...>)) char storage[sizeof(std::tuple<CommandBuffer&, T...>)];
				pack_typed_tuple<T...>(args, meta, cb, storage);
				if constexpr (!std::is_same_v<void, decltype(std::apply(typed_cb, *reinterpret_cast<std::tuple<CommandBuffer&, T...>*>(storage)))>) {
					auto typed_ret = std::apply(typed_cb, *reinterpret_cast<std::tuple<CommandBuffer&, T...>*>(storage));
					// now we erase these types
					if constexpr (!is_tuple<Ret>::value) {
						rets[0] = typed_ret.ptr;
					} else {
						unpack_typed_tuple(typed_ret, rets);
					}
				} else {
					std::apply(typed_cb, *reinterpret_cast<std::tuple<CommandBuffer&, T...>*>(storage));
				}
			};

			// when this function is called, we weave in this call into the IR
			return [untyped_cb = std::move(callback), name, scheduling_info, inner_scope = VUK_CALL](Value<typename T::type>... args, VUK_CALLSTACK) mutable {
				auto& first = First(args...);

				bool reuse_node =
				    first.node.use_count() == 1 && first.node->get_node()->kind != Node::ACQUIRE && first.node->acqrel->status == Signal::Status::eDisarmed;

				std::vector<Type*> arg_types;
				std::tuple arg_tuple_as_a = { T{ nullptr, args.get_peeled_head() }... };
				fill_arg_ty(arg_tuple_as_a, arg_types);

				std::vector<Type*> ret_types;
				if constexpr (is_tuple<Ret>::value) {
					auto [idxs, ret_tuple] = intersect_tuples<std::tuple<T...>, Ret>(arg_tuple_as_a);
					fill_ret_ty(idxs, ret_tuple, ret_types);
				} else if constexpr (!std::is_same_v<Ret, void>) {
					auto [idxs, ret_tuple] = intersect_tuples<std::tuple<T...>, std::tuple<Ret>>(arg_tuple_as_a);
					fill_ret_ty(idxs, ret_tuple, ret_types);
				}
				auto opaque_fn_ty = current_module.make_opaque_fn_ty(arg_types, ret_types, vuk::DomainFlagBits::eAny, untyped_cb);
				opaque_fn_ty->debug_info = current_module.allocate_type_debug_info(name.c_str());
				auto opaque_fn = current_module.make_declare_fn(opaque_fn_ty);
				Node* node = current_module.make_call(opaque_fn, args.peel_head()...);
				node->scheduling_info = new SchedulingInfo(scheduling_info);
				inner_scope.parent = &_scope;
				current_module.set_source_location(node, inner_scope);

				std::vector<std::shared_ptr<ExtNode>> dependent_nodes;
				[reuse_node, &dependent_nodes](auto& first, auto&... rest) {
					if (!reuse_node) {
						dependent_nodes.push_back(std::move(first.node));
					}
					(dependent_nodes.push_back(std::move(rest.node)), ...);
				}(args...);

				if (reuse_node) {
					first.node->mutate(node);
				}
				auto extnode = reuse_node ? std::move(first.node) : std::make_shared<ExtNode>(ExtNode(node, std::move(dependent_nodes)));
				if (reuse_node) {
					extnode->deps.insert(extnode->deps.end(), std::make_move_iterator(dependent_nodes.begin()), std::make_move_iterator(dependent_nodes.end()));
				}

				if constexpr (is_tuple<Ret>::value) {
					auto [idxs, ret_tuple] = intersect_tuples<std::tuple<T...>, Ret>(arg_tuple_as_a);
					return make_ret(std::move(extnode), ret_tuple);
				} else if constexpr (!std::is_same_v<Ret, void>) {
					auto [idxs, ret_tuple] = intersect_tuples<std::tuple<T...>, std::tuple<Ret>>(arg_tuple_as_a);
					return std::get<0>(make_ret(std::move(extnode), ret_tuple));
				}
			};
		}
	};

	template<class F>
	[[nodiscard]] auto make_pass(Name name, F&& body, SchedulingInfo scheduling_info = SchedulingInfo(DomainFlagBits::eAny), VUK_CALLSTACK) {
		using traits = closure_traits<decltype(&F::operator())>;
		return TupleMap<drop_t<1, typename traits::types>>::template make_lam<typename traits::result_type, F>(
		    name, std::forward<F>(body), scheduling_info, VUK_CALL);
	}

	inline ExtRef make_ext_ref(Ref ref, std::vector<std::shared_ptr<ExtNode>> deps = {}) {
		return ExtRef(std::make_shared<ExtNode>(ref.node, std::move(deps)), ref);
	}

	[[nodiscard]] inline Value<ImageAttachment> declare_ia(Name name, ImageAttachment ia = {}, VUK_CALLSTACK) {
		Ref ref = current_module.make_declare_image(ia);
		current_module.name_output(ref, name.c_str());
		current_module.set_source_location(ref.node, VUK_CALL);
		return { make_ext_ref(ref) };
	}

	[[nodiscard]] inline Value<ImageAttachment> acquire_ia(Name name, ImageAttachment ia, Access access, VUK_CALLSTACK) {
		Ref ref = current_module.make_acquire(current_module.get_builtin_image(), nullptr, ia);
		auto ext_ref = make_ext_ref(ref);
		ext_ref.node->owned_acqrel = std::make_unique<AcquireRelease>();
		ext_ref.node->owned_acqrel->status = Signal::Status::eHostAvailable;
		ext_ref.node->owned_acqrel->last_use.resize(1);
		ext_ref.node->owned_acqrel->last_use[0] = to_use(access);
		ext_ref.node->acqrel = ext_ref.node->owned_acqrel.get();
		ref.node->acquire.acquire = ext_ref.node->owned_acqrel.get();
		current_module.name_output(ref, name.c_str());
		current_module.set_source_location(ref.node, VUK_CALL);
		return { std::move(ext_ref) };
	}

	[[nodiscard]] inline Value<Buffer> declare_buf(Name name, Buffer buf = {}, VUK_CALLSTACK) {
		Ref ref = current_module.make_declare_buffer(buf);
		current_module.name_output(ref, name.c_str());
		current_module.set_source_location(ref.node, VUK_CALL);
		return { make_ext_ref(ref) };
	}

	[[nodiscard]] inline Value<Buffer> acquire_buf(Name name, Buffer buf, Access access, VUK_CALLSTACK) {
		Ref ref = current_module.make_acquire(current_module.get_builtin_buffer(), nullptr, buf);
		auto ext_ref = make_ext_ref(ref);
		ext_ref.node->owned_acqrel = std::make_unique<AcquireRelease>();
		ext_ref.node->acqrel = ext_ref.node->owned_acqrel.get();
		ext_ref.node->owned_acqrel->status = Signal::Status::eHostAvailable;
		ext_ref.node->owned_acqrel->last_use.resize(1);
		ext_ref.node->owned_acqrel->last_use[0] = to_use(access);
		ref.node->acquire.acquire = ext_ref.node->owned_acqrel.get();
		current_module.name_output(ref, name.c_str());
		current_module.set_source_location(ref.node, VUK_CALL);
		return { std::move(ext_ref) };
	}

	// TODO: due to the pack, we can't do the source_location::current() trick
	template<class T, class... Args>
	[[nodiscard]] inline Value<T[]> declare_array(Name name, Value<T> arg, Args... args) {
		std::vector<std::shared_ptr<ExtNode>> deps;
		std::array refs = { arg.get_head(), args.get_head()... };
		deps = { arg.node, args.node... };
		Ref ref = current_module.make_declare_array(Type::stripped(refs[0].type()), refs);
		current_module.name_output(ref, name.c_str());
		return { make_ext_ref(ref, deps) };
	}

	template<class T>
	[[nodiscard]] inline Value<T[]> declare_array(Name name, std::span<const Value<T>> args, VUK_CALLSTACK) {
		std::vector<Ref> refs;
		std::vector<std::shared_ptr<ExtNode>> deps;
		for (size_t i = 0; i < args.size(); i++) {
			auto& arg = args[i];
			refs.push_back(arg.get_head());
			deps.push_back(arg.node);
		}
		Type* t;
		if constexpr (std::is_same_v<T, vuk::ImageAttachment>) {
			t = current_module.get_builtin_image();
		} else if constexpr (std::is_same_v<T, vuk::Buffer>) {
			t = current_module.get_builtin_buffer();
		}
		Ref ref = current_module.make_declare_array(t, refs);
		current_module.name_output(ref, name.c_str());
		current_module.set_source_location(ref.node, VUK_CALL);
		return { make_ext_ref(ref, std::move(deps)) };
	}

	template<class T>
	[[nodiscard]] inline Value<T[]> declare_array(Name name, std::span<Value<T>> args, VUK_CALLSTACK) {
		std::vector<Ref> refs;
		std::vector<std::shared_ptr<ExtNode>> deps;
		for (size_t i = 0; i < args.size(); i++) {
			auto& arg = args[i];
			refs.push_back(arg.get_head());
			deps.push_back(arg.node);
		}
		Type* t;
		if constexpr (std::is_same_v<T, vuk::ImageAttachment>) {
			t = current_module.get_builtin_image();
		} else if constexpr (std::is_same_v<T, vuk::Buffer>) {
			t = current_module.get_builtin_buffer();
		}
		Ref ref = current_module.make_declare_array(t, refs);
		current_module.name_output(ref, name.c_str());
		current_module.set_source_location(ref.node, VUK_CALL);
		return { make_ext_ref(ref, std::move(deps)) };
	}

	[[nodiscard]] inline Value<Swapchain> declare_swapchain(Swapchain& bundle, VUK_CALLSTACK) {
		Ref ref = current_module.make_declare_swapchain(bundle);
		current_module.set_source_location(ref.node, VUK_CALL);
		return { make_ext_ref(ref) };
	}

	[[nodiscard]] inline Value<ImageAttachment> acquire_next_image(Name name, Value<Swapchain> in, VUK_CALLSTACK) {
		Ref ref = current_module.make_acquire_next_image(in.get_head());
		current_module.name_output(ref, name.c_str());
		current_module.set_source_location(ref.node, VUK_CALL);
		return std::move(in).transmute<ImageAttachment>(ref);
	}

	[[nodiscard]] inline Value<void> enqueue_presentation(Value<ImageAttachment> in) {
		return std::move(in).as_released<void>(Access::ePresent, DomainFlagBits::ePE);
	}

	struct Compiler {
		Compiler();
		~Compiler();

		/// @brief Build the graph, assign framebuffers, render passes and subpasses
		///	link automatically calls this, only needed if you want to use the reflection functions
		/// @param compile_options CompileOptions controlling compilation behaviour
		Result<void> compile(std::span<std::shared_ptr<ExtNode>> rgs, const RenderGraphCompileOptions& compile_options);

		/// @brief Use this RenderGraph and create an ExecutableRenderGraph
		/// @param compile_options CompileOptions controlling compilation behaviour
		Result<struct ExecutableRenderGraph> link(std::span<std::shared_ptr<ExtNode>> rgs, const RenderGraphCompileOptions& compile_options);

		// reflection functions

		/// @brief retrieve usages of resources in the RenderGraph
		std::span<struct ChainLink*> get_use_chains() const;

		/// @brief compute ImageUsageFlags for given use chain
		ImageUsageFlags compute_usage(const struct ChainLink* chain);

		/// @brief Dump the pass dependency graph in graphviz format
		std::string dump_graph();

		template<class T>
		T& get_value(Ref parm) {
			return *reinterpret_cast<T*>(get_value(parm));
		};

		void* get_value(Ref parm);

	private:
		struct RGCImpl* impl;

		// internal passes
		void queue_inference();
		void pass_partitioning();
		void resource_linking();
		void render_pass_assignment();

		friend struct ExecutableRenderGraph;
	};

	struct ExecutableRenderGraph {
		ExecutableRenderGraph(Compiler&);
		~ExecutableRenderGraph();

		ExecutableRenderGraph(const ExecutableRenderGraph&) = delete;
		ExecutableRenderGraph& operator=(const ExecutableRenderGraph&) = delete;

		ExecutableRenderGraph(ExecutableRenderGraph&&) noexcept;
		ExecutableRenderGraph& operator=(ExecutableRenderGraph&&) noexcept;

		Result<void> execute(Allocator& allocator);

	private:
		struct RGCImpl* impl;

		void fill_render_pass_info(struct RenderPassInfo& rpass, const size_t& i, class CommandBuffer& cobuf);

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
