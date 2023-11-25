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

#include <deque>
#include <functional>
#include <optional>
#include <span>
#include <string_view>
#include <unordered_set>
#include <vector>

#if defined(__clang__) or defined(__GNUC__)
#define VUK_IA(access) vuk::IA<access, decltype([]() {})>
#define VUK_BA(access) vuk::BA<access, decltype([]() {})>
#else
namespace vuk {
	template<size_t I>
	struct tag_type {};
}; // namespace vuk
#define VUK_IA(access) vuk::IA<access, vuk::tag_type<__COUNTER__>>
#define VUK_BA(access) vuk::BA<access, vuk::tag_type<__COUNTER__>>
#endif

namespace vuk {
	struct FutureBase;
	struct Resource;

	struct TypeDebugInfo {
		std::string name;
	};

	struct Type {
		enum TypeKind { IMAGE_TY, BUFFER_TY, IMBUED_TY, CONNECTED_TY, OPAQUE_FN_TY } kind;

		TypeDebugInfo* debug_info = nullptr;

		union {
			struct {
				Type* T;
				Access access;
			} imbued;
			struct {
				Type* T;
				size_t ref_idx;
			} connected;
			struct {
				std::span<Type* const> args;
				std::span<Type* const> return_types;
				int execute_on;
				std::function<void(CommandBuffer&, std::span<void*>, std::span<void*>)>* callback;
			} opaque_fn;
		};

		bool is_image() {
			return kind == IMAGE_TY;
		}

		bool is_buffer() {
			return kind == BUFFER_TY;
		}

		~Type() {}
	};

	struct RG;

	struct Node;

	struct Ref {
		Node* node = nullptr;
		size_t index;

		Type* type();

		explicit constexpr operator bool() const noexcept {
			return node != nullptr;
		}

		constexpr std::strong_ordering operator<=>(const Ref&) const noexcept = default;
	};

	struct SchedulingInfo {
		DomainFlags required_domain;
	};

	struct NodeDebugInfo {
		std::vector<std::string> result_names;
	};

	struct Node {
		enum Kind { DECLARE, IMPORT, CALL, CLEAR, DIVERGE, CONVERGE, RESOLVE, RELEASE } kind;
		std::span<Type* const> type;
		NodeDebugInfo* debug_info = nullptr;
		SchedulingInfo* scheduling_info = nullptr;
		union {
			struct {
				void* value;
				std::optional<Allocator> allocator;
			} declare;
			struct {
				void* value;
			} import;
			struct {
				std::span<Ref> args;
				Type* fn_ty;
			} call;
			struct {
				const Ref dst;
				Clear* cv;
			} clear;
			struct {
				const Ref initial;
				Subrange::Image subrange;
			} diverge;
			struct {
				std::span<Ref> diverged;
			} converge;
			struct {
				const Ref source_ms;
				const Ref source_ss;
				const Ref dst_ss;
			} resolve;
			struct {
				const Ref src;
			} release;
		};

		std::string_view kind_to_sv() {
			switch (kind) {
			case DECLARE:
				return "declare";
			case CALL:
				return "call";
			}
			assert(0);
			return "";
		}
	};

	inline Ref first(Node* node) {
		assert(node->type.size() > 0);
		return { node, 0 };
	}

	inline Ref nth(Node* node, size_t idx) {
		assert(node->type.size() > idx);
		return { node, idx };
	}

	inline Type* Ref::type() {
		return node->type[index];
	}

	struct RG {
		RG() {
			builtin_image = &types.emplace_back(Type{ .kind = Type::IMAGE_TY });
			builtin_buffer = &types.emplace_back(Type{ .kind = Type::BUFFER_TY });
		}

		std::deque<Node> op_arena;
		char* debug_arena;

		std::deque<Type> types;
		Type* builtin_image;
		Type* builtin_buffer;

		std::vector<std::shared_ptr<RG>> subgraphs;
		// uint64_t current_hash = 0;

		void* ensure_space(size_t size) {
			return &op_arena.emplace_back(Node{});
		}

		Node* emplace_op(Node v, NodeDebugInfo = {}) {
			return new (ensure_space(sizeof Node)) Node(v);
		}

		Type* emplace_type(Type&& t, TypeDebugInfo = {}) {
			return &types.emplace_back(std::move(t));
		}

		void reference_RG(std::shared_ptr<RG> other) {
			subgraphs.emplace_back(other);
		}

		// TYPES
		Type* make_imbued_ty(Type* ty, Access access) {
			return emplace_type(Type{ .kind = Type::IMBUED_TY, .imbued = { .T = ty, .access = access } });
		}

		Type* make_connected_ty(Type* ty, size_t ref_idx) {
			return emplace_type(Type{ .kind = Type::CONNECTED_TY, .connected = { .T = ty, .ref_idx = ref_idx } });
		}

		// OPS

		void name_outputs(Node* node, std::vector<std::string> names) {
			if (!node->debug_info) {
				node->debug_info = new NodeDebugInfo;
			}
			node->debug_info->result_names.assign(names.begin(), names.end());
		}

		void name_output(Ref ref, std::string name) {
			if (!ref.node->debug_info) {
				ref.node->debug_info = new NodeDebugInfo;
			}
			auto& names = ref.node->debug_info->result_names;
			if (names.size() <= ref.index) {
				names.resize(ref.index + 1);
			}
			names[ref.index] = name;
		}

		Ref make_declare_image(ImageAttachment value) {
			return first(emplace_op(Node{ .kind = Node::DECLARE, .type = std::span{ &builtin_image, 1 }, .declare = { .value = new ImageAttachment(value) } }));
		}

		Ref make_declare_buffer(Buffer value) {
			return first(emplace_op(Node{ .kind = Node::DECLARE, .type = std::span{ &builtin_buffer, 1 }, .declare = { .value = new Buffer(value) } }));
		}

		Ref make_clear_image(Ref dst, Clear cv) {
			return first(emplace_op(Node{ .kind = Node::CLEAR, .type = std::span{ &builtin_image, 1 }, .clear = { .dst = dst, .cv = new Clear(cv) } }));
		}

		Type* make_opaque_fn_ty(std::span<Type* const> args,
		                        std::span<Type* const> ret_types,
		                        DomainFlags execute_on,
		                        std::function<void(CommandBuffer&, std::span<void*>, std::span<void*>)> callback) {
			auto arg_ptr = new Type*[args.size()];
			std::copy(args.begin(), args.end(), arg_ptr);
			auto ret_ty_ptr = new Type*[ret_types.size()];
			std::copy(ret_types.begin(), ret_types.end(), ret_ty_ptr);
			return emplace_type(Type{ .kind = Type::OPAQUE_FN_TY,
			                          .opaque_fn = { .args = std::span(arg_ptr, args.size()),
			                                         .return_types = std::span(ret_ty_ptr, ret_types.size()),
			                                         .execute_on = execute_on.m_mask,
			                                         .callback = new std::function<void(CommandBuffer&, std::span<void*>, std::span<void*>)>(callback) } });
		}

		template<class... Refs>
		Node* make_call(Type* fn, Refs... args) {
			Ref* args_ptr = new Ref[sizeof...(args)]{ args... };
			return emplace_op(Node{ .kind = Node::CALL, .type = fn->opaque_fn.return_types, .call = { .args = std::span(args_ptr, sizeof...(args)), .fn_ty = fn } });
		}

		void make_release(Ref src) {
			emplace_op(Node{ .kind = Node::RELEASE, .release = { .src = src } });
		}
	};

	QueueResourceUse to_use(Access acc, DomainFlags domain);

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
		ImageAttachment& attach_image(Name name, ImageAttachment image_attachment, Access initial = eNone);

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

		void add_final_release(Future& future, DomainFlags src_domain);
		void remove_final_release(Future& future);

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

	template<Access acc, class T, StringLiteral N = "">
	struct IA {
		static constexpr Access access = acc;
		using base = Image;
		using attach = ImageAttachment;
		static constexpr StringLiteral identifier = N;
		static constexpr Type::TypeKind kind = Type::IMAGE_TY;
		ImageAttachment* ptr;

		operator ImageAttachment() {
			return *ptr;
		}
	};

	template<Access acc, class T, StringLiteral N = "">
	struct BA {
		static constexpr Access access = acc;
		using base = Buffer;
		using attach = Buffer;
		static constexpr StringLiteral identifier = N;
		static constexpr Type::TypeKind kind = Type::BUFFER_TY;

		Buffer* ptr;

		operator Buffer() {
			return *ptr;
		}
	};

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

	template<class T>
	struct TypedFuture {};

	template<>
	struct TypedFuture<Image> {
		std::shared_ptr<RG> rg;
		Ref head;
		ImageAttachment* value;

		ImageAttachment* operator->() {
			return value;
		}
	};

	template<>
	struct TypedFuture<Buffer> {
		std::shared_ptr<RG> rg;
		Ref head;
		Buffer* value;

		Buffer* operator->() {
			return value;
		}
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

	struct Nil;

	template<typename...>
	struct type_list {
		static constexpr size_t length = 0u;
		using head = Nil;
	};

	template<typename Head_t, typename... Tail_ts>
	struct type_list<Head_t, Tail_ts...> {
		static constexpr size_t length = sizeof...(Tail_ts) + 1u;
		using head = Head_t;
		using tail = type_list<Tail_ts...>;
	};

	template<typename, typename>
	struct cons {
		DELAYED_ERROR("cons<T, U> expects a type_list as its U");
	};

	template<typename T, typename... Ts>
	struct cons<T, type_list<Ts...>> {
		using type = type_list<T, Ts...>;
	};

	template<typename T, typename... Ts>
	using cons_t = typename cons<T, Ts...>::type;

	template<typename T>
	struct head {
		using type = typename T::head;
	};

	template<typename T>
	using head_t = typename head<T>::type;

	template<typename T>
	struct tail {
		using type = typename T::tail;
	};

	template<typename T>
	using tail_t = typename tail<T>::type;

	template<typename T>
	struct length {
		static constexpr size_t value = T::length;
	};

	template<typename T>
	constexpr size_t length_v = length<T>::value;

	template<template<typename> typename, typename>
	struct map {
		DELAYED_ERROR("map<Metafunction_t, T> expects a type_list as its T");
	};

	template<template<typename> typename Metafunction_t>
	struct map<Metafunction_t, type_list<>> {
		using type = type_list<>;
	};

	template<template<typename> typename Metafunction_t, typename T, typename... Ts>
	struct map<Metafunction_t, type_list<T, Ts...>> {
	private:
		using tail_mapped = typename map<Metafunction_t, type_list<Ts...>>::type;
		using head_mapped = typename Metafunction_t<T>::type;

	public:
		using type = cons_t<head_mapped, tail_mapped>;
	};

	template<template<typename> typename Metafunction_t, typename T>
	using map_t = typename map<Metafunction_t, T>::type;

	namespace details {
		template<template<typename> typename, template<typename, typename> typename, bool, typename>
		struct reduce {
			DELAYED_ERROR("reduce<Predicate_t, BinaryOp_t, Default, T> expects a type_list as its T");
		};

		template<template<typename> typename Predicate_t, template<typename, typename> typename BinaryOp_t, bool Default>
		struct reduce<Predicate_t, BinaryOp_t, Default, type_list<>> {
			static constexpr bool value = Default;
		};

		template<template<typename> typename Predicate_t, template<typename, typename> typename BinaryOp_t, bool Default, typename T, typename... Ts>
		struct reduce<Predicate_t, BinaryOp_t, Default, type_list<T, Ts...>> {
		private:
			using applied_head = Predicate_t<T>;
			using applied_tail = reduce<Predicate_t, BinaryOp_t, Default, type_list<Ts...>>;

		public:
			static constexpr bool value = BinaryOp_t<applied_head, applied_tail>::value;
		};

		template<typename T, typename U>
		struct OR {
			static constexpr bool value = T::value || U::value;
		};

		template<typename T, typename U>
		struct AND {
			static constexpr bool value = T::value && U::value;
		};

		template<typename T>
		struct NOT {
			static constexpr bool value = !T::value;
		};
	} // namespace details

	template<template<typename> typename Predicate_t, typename List_t>
	struct any : details::reduce<Predicate_t, details::OR, false, List_t> {};

	template<template<typename> typename Predicate_t, typename T>
	constexpr bool any_v = any<Predicate_t, T>::value;

	template<template<typename> typename, typename>
	struct filter {
		DELAYED_ERROR("filter<Predicate_t, T> expects a type_list as its T");
	};

	template<template<typename> typename Predicate_t>
	struct filter<Predicate_t, type_list<>> {
		using type = type_list<>;
	};

	template<template<typename> typename Predicate_t, typename T, typename... Ts>
	struct filter<Predicate_t, type_list<T, Ts...>> {
	private:
		using rest = typename filter<Predicate_t, type_list<Ts...>>::type;

	public:
		using type = std::conditional_t<Predicate_t<T>::value, cons_t<T, rest>, rest>;
	};

	template<template<typename> typename Predicate_t, typename List_t>
	using filter_t = typename filter<Predicate_t, List_t>::type;

	namespace details {
		template<template<typename, size_t> typename, typename, size_t>
		struct map_indexed_impl {
			DELAYED_ERROR("map_indexed expects a type_list as its T");
		};

		template<template<typename, size_t> typename Metafunction_t, size_t I, typename T, typename... Ts>
		struct map_indexed_impl<Metafunction_t, type_list<T, Ts...>, I> {
		private:
			using applied_head = typename Metafunction_t<T, I>::type;
			using applied_tail = typename map_indexed_impl<Metafunction_t, type_list<Ts...>, I + 1>::type;

		public:
			using type = cons_t<applied_head, applied_tail>;
		};

		template<template<typename, size_t> typename Metafunction_t, size_t I, typename T>
		struct map_indexed_impl<Metafunction_t, type_list<T>, I> {
		private:
			using applied_head = typename Metafunction_t<T, I>::type;

		public:
			using type = type_list<applied_head>;
		};

		template<template<typename, size_t> typename Metafunction_t, size_t I>
		struct map_indexed_impl<Metafunction_t, type_list<>, I> {
			using type = Nil;
		};
	} // namespace details

	template<template<typename, size_t> typename Metafunction_t, typename List_t>
	struct map_indexed {
		using type = typename details::map_indexed_impl<Metafunction_t, List_t, 0u>::type;
	};

	template<template<typename, size_t> typename Metafunction_t, typename List_t>
	using map_indexed_t = typename map_indexed<Metafunction_t, List_t>::type;

	struct Good1 {};
	struct Good2 {};
	struct Bad1 {};
	struct Bad2 {};

	template<typename Spec1, typename Spec2>
	struct same_specialization : std::false_type {};

	template<template<typename> typename Spec1, template<typename> typename Spec2, typename T>
	struct same_specialization<Spec1<T>, Spec2<T>> : std::true_type {};

	template<typename T>
	struct AT {
		float x;
	};

	template<typename T>
	struct BT {
		float x;
	};

	using ListA = type_list<AT<Good1>, AT<Good2>, AT<Bad1>>;
	using ListB = type_list<BT<Bad2>, BT<Good2>, BT<Good1>>;

	template<typename T, typename U>
	static constexpr bool same_specialization_v = same_specialization<T, U>::value;
	static_assert(same_specialization_v<AT<Good1>, BT<Good1>>);
	static_assert(!same_specialization_v<BT<Good1>, AT<Bad1>>);

	template<typename T>
	struct unwrap {
		using type = typename T::type;
	};

	template<typename T>
	using unwrap_t = typename T::type;

	template<typename TypeList, typename T>
	struct any_is_same_specialization_unwrapped {
		template<typename U>
		struct partial {
			static constexpr bool value = std::is_same_v<unwrap_t<T>, U>;
		};

		static constexpr bool value = any_v<partial, TypeList>;
	};

	template<typename T, size_t I>
	struct Indexed {
		using type = T;
		static constexpr size_t Index = I;
	};

	template<typename T, size_t I>
	struct wrap_indexed {
		using type = Indexed<T, I>;
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

	template<typename ListA_t, typename ListB_t>
	struct filtered_indexed {
		template<typename Elem>
		struct partial {
			static constexpr bool value = any_is_same_specialization_unwrapped<ListB_t, Elem>::value;
		};

		using indexed_listA = map_indexed_t<wrap_indexed, ListA_t>;

		using type = filter_t<partial, indexed_listA>;
	};

	template<typename IndexedList_t, size_t I, typename ToFill_t, typename Filled_t>
	void fill_tuple(const ToFill_t& fill, Filled_t& filled) {
		using head = typename IndexedList_t::head;
		if constexpr (std::is_same_v<head, Nil>) {
			return;
		} else {
			std::get<I>(filled) = std::get<head::Index>(fill);
			fill_tuple<typename IndexedList_t::tail, I + 1>(fill, filled);
		}
	};

	template<typename T1, typename T2>
	auto intersect_tuples(const T1& t1) {
		using T1List = tuple_to_typelist_t<T1>;
		using T2List = tuple_to_typelist_t<T2>;
		using indices = typename filtered_indexed<T1List, T2List>::type;
		typelist_to_tuple_t<map_t<unwrap, indices>> t3;

		fill_tuple<indices, 0>(t1, t3);

		return t3;
	}

	template<typename Tuple>
	struct TupleMap;

	template<typename... T>
	void pack_typed_tuple(std::span<void*> src, CommandBuffer& cb, void* dst) {
		std::tuple<CommandBuffer*, T...>& tuple = *reinterpret_cast<std::tuple<CommandBuffer*, T...>*>(dst);
		std::get<0>(tuple) = &cb;
#define X(n)                                                                                                                                                   \
	if constexpr ((sizeof...(T)) > n) {                                                                                                                          \
		auto& ptr = std::get<n>(tuple).ptr;                                                                                                                        \
		ptr = reinterpret_cast<decltype(ptr)>(src[n]);                                                                                                             \
	}
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
	};

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
			return std::make_tuple(TypedFuture<typename T::base>{ rg, { node, i++ }, std::get<T>(us).ptr }...);
		} else if constexpr (sizeof...(T) == 0) {
			return TypedFuture<typename T::base>{ rg, first(node), std::get<0>(us).ptr };
		}
	}

	template<typename... T>
	struct TupleMap<std::tuple<T...>> {
		using ret_tuple = std::tuple<TypedFuture<typename T::base>...>;

		template<class Ret, class F>
		static auto make_lam(Name name, F&& body) {
			auto callback = [typed_cb = std::move(body)](CommandBuffer& cb, std::span<void*> args, std::span<void*> rets) {
				// we do type recovery here -> convert untyped args to typed ones
				alignas(alignof(std::tuple<CommandBuffer&, T...>)) char storage[sizeof(std::tuple<CommandBuffer&, T...>)];
				pack_typed_tuple(args, cb, storage);
				auto typed_ret = std::apply(typed_cb, *reinterpret_cast<std::tuple<CommandBuffer&, T...>*>(storage));
				// now we erase these types
				if constexpr (!is_tuple<Ret>::value) {
					rets[0] = typed_ret.ptr;
				} else {
					unpack_typed_tuple(typed_ret, rets);
				}
			};

			// when this function is called, we weave in this call into the IR
			return [untyped_cb = std::move(callback)](TypedFuture<typename T::base>... args) mutable {
				auto& first = [](auto& first, auto&...) -> auto& {
					return first;
				}(args...);
				auto& rgp = first.rg;
				RG& rg = *rgp.get();

				std::vector<Type*> arg_types;
				fill_arg_ty(rg, arg_types);

				std::tuple arg_tuple = { T(args.value)... };

				std::vector<Type*> ret_types;
				if constexpr (is_tuple<Ret>::value) {
					TupleMap<Ret>::fill_ret_ty(rg, ret_types);
				} else if constexpr (!std::is_same_v<Ret, void>) {
					ret_types.emplace_back(rg.make_imbued_ty(rg.builtin_image, Ret::access));
				}
				auto opaque_fn_ty = rg.make_opaque_fn_ty(arg_types, ret_types, vuk::DomainFlagBits::eAny, untyped_cb);

				Node* node = rg.make_call(opaque_fn_ty, args.head...);
				if constexpr (is_tuple<Ret>::value) {
					std::tuple ret_tuple = intersect_tuples<std::tuple<T...>, Ret>(arg_tuple);
					return make_ret(rgp, node, ret_tuple);
				} else if constexpr (!std::is_same_v<Ret, void>) {
					return std::remove_reference_t<decltype(first)>{ rgp, vuk::first(node), first.value };
				}
			};
		}

		static auto fill_arg_ty(RG& rg, std::vector<Type*>& arg_types) {
			(arg_types.emplace_back(rg.make_imbued_ty(rg.builtin_image, T::access)), ...);
		}

		static auto fill_ret_ty(RG& rg, std::vector<Type*>& ret_types) {
			(ret_types.emplace_back(rg.make_imbued_ty(rg.builtin_image, T::access)), ...);
		}
	};

	template<class F>
	[[nodiscard]] auto make_pass(Name name, F&& body) {
		using traits = closure_traits<decltype(&F::operator())>;
		return TupleMap<drop_t<1, typename traits::types>>::template make_lam<typename traits::result_type, F>(name, std::forward<F>(body));
	}

	[[nodiscard]] inline TypedFuture<Image> declare_ia(Name name, ImageAttachment ia = {}) {
		std::shared_ptr<RG> rg = std::make_shared<RG>();
		Ref ref = rg->make_declare_image(ia);
		rg->name_outputs(ref.node, { name.c_str() });
		return { rg, ref, reinterpret_cast<ImageAttachment*>(ref.node->declare.value) };
	}

	[[nodiscard]] inline TypedFuture<Buffer> declare_buf(Name name, Buffer buf = {}) {
		std::shared_ptr<RG> rg = std::make_shared<RG>();
		Ref ref = rg->make_declare_buffer(buf);
		rg->name_outputs(ref.node, { name.c_str() });
		return { rg, ref, reinterpret_cast<Buffer*>(ref.node->declare.value) };
	}

	[[nodiscard]] inline TypedFuture<Image> clear(TypedFuture<Image> in, Clear clear_value) {
		auto& rg = in.rg;
		return { rg, rg->make_clear_image(in.head, clear_value), in.value };
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

		Result<struct BufferInfo*, RenderGraphException> get_resource_buffer(const NameReference&, struct PassInfo*);
		Result<struct AttachmentInfo*, RenderGraphException> get_resource_image(const NameReference&, struct PassInfo*);

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
