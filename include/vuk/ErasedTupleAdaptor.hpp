#pragma once

#include <type_traits>

namespace vuk {
	namespace detail {
		template<class T>
		struct member_type_helper;

		template<class C, class T>
		struct member_type_helper<T C::*> {
			using type = T;
		};

		template<class T>
		struct member_type : member_type_helper<typename std::remove_cvref<T>::type> {};

		// Helper type
		template<class T>
		using member_type_t = member_type<T>::type;
	} // namespace detail

	template<class T>
	struct erased_tuple_adaptor : std::false_type {};

// https://stackoverflow.com/a/44479664
#define EVAL(...)                                                                                                          __VA_ARGS__
#define VARCOUNT(...)                                                                                                      EVAL(VARCOUNT_I(__VA_ARGS__, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, ))
#define VARCOUNT_I(_, _21, _20, _19, _18, _17, _16, _15, _14, _13, _12, _11, _10, _9, _8, _7, _6, _5, _4, _3, _2, X_, ...) X_
#define STR(X)                                                                                                             STR_I(X)
#define STR_I(X)                                                                                                           #X
#define GLUE(X, Y)                                                                                                         GLUE_I(X, Y)
#define GLUE_I(X, Y)                                                                                                       X##Y
#define FIRST(...)                                                                                                         EVAL(FIRST_I(__VA_ARGS__, ))
#define FIRST_I(X, ...)                                                                                                    X
#define TUPLE_TAIL(...)                                                                                                    EVAL(TUPLE_TAIL_I(__VA_ARGS__))
#define TUPLE_TAIL_I(X, ...)                                                                                               (__VA_ARGS__)

#define TRANSFORM(NAME_, ARGS_)    (GLUE(TRANSFORM_, VARCOUNT ARGS_)(NAME_, ARGS_))
#define TRANSFORM_1(NAME_, ARGS_)  NAME_ ARGS_
#define TRANSFORM_2(NAME_, ARGS_)  NAME_(FIRST ARGS_), TRANSFORM_1(NAME_, TUPLE_TAIL ARGS_)
#define TRANSFORM_3(NAME_, ARGS_)  NAME_(FIRST ARGS_), TRANSFORM_2(NAME_, TUPLE_TAIL ARGS_)
#define TRANSFORM_4(NAME_, ARGS_)  NAME_(FIRST ARGS_), TRANSFORM_3(NAME_, TUPLE_TAIL ARGS_)
#define TRANSFORM_5(NAME_, ARGS_)  NAME_(FIRST ARGS_), TRANSFORM_4(NAME_, TUPLE_TAIL ARGS_)
#define TRANSFORM_6(NAME_, ARGS_)  NAME_(FIRST ARGS_), TRANSFORM_5(NAME_, TUPLE_TAIL ARGS_)
#define TRANSFORM_7(NAME_, ARGS_)  NAME_(FIRST ARGS_), TRANSFORM_6(NAME_, TUPLE_TAIL ARGS_)
#define TRANSFORM_8(NAME_, ARGS_)  NAME_(FIRST ARGS_), TRANSFORM_7(NAME_, TUPLE_TAIL ARGS_)
#define TRANSFORM_9(NAME_, ARGS_)  NAME_(FIRST ARGS_), TRANSFORM_8(NAME_, TUPLE_TAIL ARGS_)
#define TRANSFORM_10(NAME_, ARGS_) NAME_(FIRST ARGS_), TRANSFORM_9(NAME_, TUPLE_TAIL ARGS_)
#define TRANSFORM_11(NAME_, ARGS_) NAME_(FIRST ARGS_), TRANSFORM_10(NAME_, TUPLE_TAIL ARGS_)
#define TRANSFORM_12(NAME_, ARGS_) NAME_(FIRST ARGS_), TRANSFORM_11(NAME_, TUPLE_TAIL ARGS_)
#define TRANSFORM_13(NAME_, ARGS_) NAME_(FIRST ARGS_), TRANSFORM_12(NAME_, TUPLE_TAIL ARGS_)
#define TRANSFORM_14(NAME_, ARGS_) NAME_(FIRST ARGS_), TRANSFORM_13(NAME_, TUPLE_TAIL ARGS_)
#define TRANSFORM_15(NAME_, ARGS_) NAME_(FIRST ARGS_), TRANSFORM_14(NAME_, TUPLE_TAIL ARGS_)
#define TRANSFORM_16(NAME_, ARGS_) NAME_(FIRST ARGS_), TRANSFORM_15(NAME_, TUPLE_TAIL ARGS_)
#define TRANSFORM_17(NAME_, ARGS_) NAME_(FIRST ARGS_), TRANSFORM_16(NAME_, TUPLE_TAIL ARGS_)
#define TRANSFORM_18(NAME_, ARGS_) NAME_(FIRST ARGS_), TRANSFORM_17(NAME_, TUPLE_TAIL ARGS_)
#define TRANSFORM_19(NAME_, ARGS_) NAME_(FIRST ARGS_), TRANSFORM_18(NAME_, TUPLE_TAIL ARGS_)
#define TRANSFORM_20(NAME_, ARGS_) NAME_(FIRST ARGS_), TRANSFORM_19(NAME_, TUPLE_TAIL ARGS_)
#define TRANSFORM_21(NAME_, ARGS_) NAME_(FIRST ARGS_), TRANSFORM_20(NAME_, TUPLE_TAIL ARGS_)

#define TRANSFORMSC(NAME_, ARGS_)   (GLUE(TRANSFORMSC_, VARCOUNT ARGS_)(NAME_, ARGS_))
#define TRANSFORMSC_1(NAME_, ARGS_) NAME_ ARGS_
#define TRANSFORMSC_2(NAME_, ARGS_)                                                                                                                            \
	NAME_(FIRST ARGS_);                                                                                                                                          \
	TRANSFORMSC_1(NAME_, TUPLE_TAIL ARGS_)
#define TRANSFORMSC_3(NAME_, ARGS_)                                                                                                                            \
	NAME_(FIRST ARGS_);                                                                                                                                          \
	TRANSFORMSC_2(NAME_, TUPLE_TAIL ARGS_)
#define TRANSFORMSC_4(NAME_, ARGS_)                                                                                                                            \
	NAME_(FIRST ARGS_);                                                                                                                                          \
	TRANSFORMSC_3(NAME_, TUPLE_TAIL ARGS_)
#define TRANSFORMSC_5(NAME_, ARGS_)                                                                                                                            \
	NAME_(FIRST ARGS_);                                                                                                                                          \
	TRANSFORMSC_4(NAME_, TUPLE_TAIL ARGS_)
#define TRANSFORMSC_6(NAME_, ARGS_)                                                                                                                            \
	NAME_(FIRST ARGS_);                                                                                                                                          \
	TRANSFORMSC_5(NAME_, TUPLE_TAIL ARGS_)
#define TRANSFORMSC_7(NAME_, ARGS_)                                                                                                                            \
	NAME_(FIRST ARGS_);                                                                                                                                          \
	TRANSFORMSC_6(NAME_, TUPLE_TAIL ARGS_)
#define TRANSFORMSC_8(NAME_, ARGS_)                                                                                                                            \
	NAME_(FIRST ARGS_);                                                                                                                                          \
	TRANSFORMSC_7(NAME_, TUPLE_TAIL ARGS_)
#define TRANSFORMSC_9(NAME_, ARGS_)                                                                                                                            \
	NAME_(FIRST ARGS_);                                                                                                                                          \
	TRANSFORMSC_8(NAME_, TUPLE_TAIL ARGS_)
#define TRANSFORMSC_10(NAME_, ARGS_)                                                                                                                           \
	NAME_(FIRST ARGS_);                                                                                                                                          \
	TRANSFORMSC_9(NAME_, TUPLE_TAIL ARGS_)
#define TRANSFORMSC_11(NAME_, ARGS_)                                                                                                                           \
	NAME_(FIRST ARGS_);                                                                                                                                          \
	TRANSFORMSC_10(NAME_, TUPLE_TAIL ARGS_)
#define TRANSFORMSC_11(NAME_, ARGS_)                                                                                                                           \
	NAME_(FIRST ARGS_);                                                                                                                                          \
	TRANSFORMSC_10(NAME_, TUPLE_TAIL ARGS_)
#define TRANSFORMSC_12(NAME_, ARGS_)                                                                                                                           \
	NAME_(FIRST ARGS_);                                                                                                                                          \
	TRANSFORMSC_11(NAME_, TUPLE_TAIL ARGS_)
#define TRANSFORMSC_13(NAME_, ARGS_)                                                                                                                           \
	NAME_(FIRST ARGS_);                                                                                                                                          \
	TRANSFORMSC_12(NAME_, TUPLE_TAIL ARGS_)
#define TRANSFORMSC_14(NAME_, ARGS_)                                                                                                                           \
	NAME_(FIRST ARGS_);                                                                                                                                          \
	TRANSFORMSC_13(NAME_, TUPLE_TAIL ARGS_)
#define TRANSFORMSC_15(NAME_, ARGS_)                                                                                                                           \
	NAME_(FIRST ARGS_);                                                                                                                                          \
	TRANSFORMSC_14(NAME_, TUPLE_TAIL ARGS_)
#define TRANSFORMSC_16(NAME_, ARGS_)                                                                                                                           \
	NAME_(FIRST ARGS_);                                                                                                                                          \
	TRANSFORMSC_15(NAME_, TUPLE_TAIL ARGS_)
#define TRANSFORMSC_17(NAME_, ARGS_)                                                                                                                           \
	NAME_(FIRST ARGS_);                                                                                                                                          \
	TRANSFORMSC_16(NAME_, TUPLE_TAIL ARGS_)
#define TRANSFORMSC_18(NAME_, ARGS_)                                                                                                                           \
	NAME_(FIRST ARGS_);                                                                                                                                          \
	TRANSFORMSC_17(NAME_, TUPLE_TAIL ARGS_)
#define TRANSFORMSC_19(NAME_, ARGS_)                                                                                                                           \
	NAME_(FIRST ARGS_);                                                                                                                                          \
	TRANSFORMSC_18(NAME_, TUPLE_TAIL ARGS_)
#define TRANSFORMSC_20(NAME_, ARGS_)                                                                                                                           \
	NAME_(FIRST ARGS_);                                                                                                                                          \
	TRANSFORMSC_19(NAME_, TUPLE_TAIL ARGS_)
#define TRANSFORMSC_21(NAME_, ARGS_)                                                                                                                           \
	NAME_(FIRST ARGS_);                                                                                                                                          \
	TRANSFORMSC_20(NAME_, TUPLE_TAIL ARGS_)

#define MEM_OBJ_ARG(X)  &T::X
#define STRIFY(X)       STR(X)
#define MEM_OBJ_TYPE(X) detail::member_type_t<decltype(&T::X)>
#define OFFSETIFY(X)    offsetof(T, X)
#define WRAPPED_T(X)    Value<detail::member_type_t<decltype(&T::X)>> X

#define MAKE_INITIALIZER(...)   { __VA_ARGS__ }
#define MAKE_TEMPLATE_LIST(...) <__VA_ARGS__>
#define MAKE_EXPAND(...)        __VA_ARGS__

#define ADAPT_STRUCT_FOR_IR(type, ...)                                                                                                                         \
	template<>                                                                                                                                                   \
	struct erased_tuple_adaptor<type> : std::true_type {                                                                                                         \
		using T = type;                                                                                                                                            \
		static constexpr std::tuple members = EVAL(MAKE_INITIALIZER TRANSFORM(MEM_OBJ_ARG, (__VA_ARGS__)));                                                        \
		static constexpr std::array member_names = EVAL(MAKE_INITIALIZER TRANSFORM(STRIFY, (__VA_ARGS__)));                                                        \
		static constexpr std::tuple EVAL(MAKE_TEMPLATE_LIST TRANSFORM(MEM_OBJ_TYPE, (__VA_ARGS__))) member_types;                                                  \
		inline static std::array offsets = EVAL(MAKE_INITIALIZER TRANSFORM(OFFSETIFY, (__VA_ARGS__)));                                                             \
		static void construct(void* dst, std::span<void*> parts) {                                                                                                 \
			T& v = *new (dst) T;                                                                                                                                     \
			size_t i = 0;                                                                                                                                            \
			std::apply(                                                                                                                                              \
			    [&](auto... member_obj_tys) { ((v.*member_obj_tys = *reinterpret_cast<detail::member_type_t<decltype(member_obj_tys)>*>(parts[i++])), ...); },       \
			    members);                                                                                                                                            \
		}                                                                                                                                                          \
		static void* get(void* value, size_t index) {                                                                                                              \
			T& v = *reinterpret_cast<T*>(value);                                                                                                                     \
			return std::apply(                                                                                                                                       \
			    [&](auto... member_obj_tys) {                                                                                                                        \
				    std::array results = { static_cast<void*>(&(v.*member_obj_tys))... };                                                                              \
				    return results[index];                                                                                                                             \
			    },                                                                                                                                                   \
			    members);                                                                                                                                            \
		}                                                                                                                                                          \
		template<size_t I>                                                                                                                                         \
		static bool check_member_at_index(const T& v) {                                                                                                            \
			constexpr auto member = std::get<I>(members);                                                                                                            \
			return (v.*member) == member_placeholder<member>::value;                                                                                                 \
		}                                                                                                                                                          \
		static bool is_default(void* value, size_t index) {                                                                                                        \
			T& v = *reinterpret_cast<T*>(value);                                                                                                                     \
			bool result = false;                                                                                                                                     \
			[&]<size_t... I>(std::index_sequence<I...>) {                                                                                                            \
				((result = (I == index ? check_member_at_index<I>(v) : result)), ...);                                                                                 \
			}(std::make_index_sequence<std::tuple_size_v<decltype(members)>>{});                                                                                     \
			return result;                                                                                                                                           \
		}                                                                                                                                                          \
		static void destroy(void* value) {                                                                                                                         \
			T& v = *reinterpret_cast<T*>(value);                                                                                                                     \
			v.~T();                                                                                                                                                  \
		}                                                                                                                                                          \
		static constexpr const char* name = #type;                                                                                                                 \
		struct proxy {                                                                                                                                             \
			EVAL(MAKE_EXPAND TRANSFORMSC(WRAPPED_T, (__VA_ARGS__)));                                                                                                 \
			proxy* operator->() {                                                                                                                                    \
				return this;                                                                                                                                           \
			}                                                                                                                                                        \
		};                                                                                                                                                         \
	}

#define ADAPT_TEMPLATED_STRUCT_FOR_IR(ttype, type, ...)                                                                                                        \
	template<ttype Type>                                                                                                                                         \
	struct erased_tuple_adaptor<type<Type>> : std::true_type {                                                                                                   \
		using T = type<Type>;                                                                                                                                      \
		static constexpr std::tuple members = EVAL(MAKE_INITIALIZER TRANSFORM(MEM_OBJ_ARG, (__VA_ARGS__)));                                                        \
		static constexpr std::array member_names = EVAL(MAKE_INITIALIZER TRANSFORM(STRIFY, (__VA_ARGS__)));                                                        \
		static constexpr std::tuple EVAL(MAKE_TEMPLATE_LIST TRANSFORM(MEM_OBJ_TYPE, (__VA_ARGS__))) member_types;                                                  \
		inline static std::array offsets = EVAL(MAKE_INITIALIZER TRANSFORM(OFFSETIFY, (__VA_ARGS__)));                                                             \
		static void construct(void* dst, std::span<void*> parts) {                                                                                                 \
			T& v = *new (dst) T;                                                                                                                                     \
			size_t i = 0;                                                                                                                                            \
			std::apply(                                                                                                                                              \
			    [&](auto... member_obj_tys) { ((v.*member_obj_tys = *reinterpret_cast<detail::member_type_t<decltype(member_obj_tys)>*>(parts[i++])), ...); },       \
			    members);                                                                                                                                            \
		}                                                                                                                                                          \
		static void* get(void* value, size_t index) {                                                                                                              \
			T& v = *reinterpret_cast<T*>(value);                                                                                                                     \
			return std::apply(                                                                                                                                       \
			    [&](auto... member_obj_tys) {                                                                                                                        \
				    std::array results = { static_cast<void*>(&(v.*member_obj_tys))... };                                                                              \
				    return results[index];                                                                                                                             \
			    },                                                                                                                                                   \
			    members);                                                                                                                                            \
		}                                                                                                                                                          \
		template<size_t I>                                                                                                                                         \
		static bool check_member_at_index(const T& v) {                                                                                                            \
			constexpr auto member = std::get<I>(members);                                                                                                            \
			return (v.*member) == member_placeholder<member>::value;                                                                                                 \
		}                                                                                                                                                          \
		static bool is_default(void* value, size_t index) {                                                                                                        \
			T& v = *reinterpret_cast<T*>(value);                                                                                                                     \
			bool result = false;                                                                                                                                     \
			[&]<size_t... I>(std::index_sequence<I...>) {                                                                                                            \
				((result = (I == index ? check_member_at_index<I>(v) : result)), ...);                                                                                 \
			}(std::make_index_sequence<std::tuple_size_v<decltype(members)>>{});                                                                                     \
			return result;                                                                                                                                           \
		}                                                                                                                                                          \
		static void destroy(void* value) {                                                                                                                         \
			T& v = *reinterpret_cast<T*>(value);                                                                                                                     \
			v.~T();                                                                                                                                                  \
		}                                                                                                                                                          \
		static constexpr const char* name = #type;                                                                                                                 \
		struct proxy {                                                                                                                                             \
			EVAL(MAKE_EXPAND TRANSFORMSC(WRAPPED_T, (__VA_ARGS__)));                                                                                                 \
			proxy* operator->() {                                                                                                                                    \
				return this;                                                                                                                                           \
			}                                                                                                                                                        \
		};                                                                                                                                                         \
	}

#define ADAPT_TEMPLATED_PACK_STRUCT_FOR_IR(type, ...)                                                                                                          \
	template<class... Args>                                                                                                                                      \
	struct erased_tuple_adaptor<type<Args...>> : std::true_type {                                                                                                \
		using T = type<Args...>;                                                                                                                                   \
		static constexpr std::tuple members = EVAL(MAKE_INITIALIZER TRANSFORM(MEM_OBJ_ARG, (__VA_ARGS__)));                                                        \
		static constexpr std::array member_names = EVAL(MAKE_INITIALIZER TRANSFORM(STRIFY, (__VA_ARGS__)));                                                        \
		static constexpr std::tuple EVAL(MAKE_TEMPLATE_LIST TRANSFORM(MEM_OBJ_TYPE, (__VA_ARGS__))) member_types;                                                  \
		inline static std::array offsets = EVAL(MAKE_INITIALIZER TRANSFORM(OFFSETIFY, (__VA_ARGS__)));                                                             \
		static void construct(void* dst, std::span<void*> parts) {                                                                                                 \
			T& v = *new (dst) T;                                                                                                                                     \
			size_t i = 0;                                                                                                                                            \
			std::apply(                                                                                                                                              \
			    [&](auto... member_obj_tys) { ((v.*member_obj_tys = *reinterpret_cast<detail::member_type_t<decltype(member_obj_tys)>*>(parts[i++])), ...); },       \
			    members);                                                                                                                                            \
		}                                                                                                                                                          \
		static void* get(void* value, size_t index) {                                                                                                              \
			T& v = *reinterpret_cast<T*>(value);                                                                                                                     \
			return std::apply(                                                                                                                                       \
			    [&](auto... member_obj_tys) {                                                                                                                        \
				    std::array results = { static_cast<void*>(&(v.*member_obj_tys))... };                                                                              \
				    return results[index];                                                                                                                             \
			    },                                                                                                                                                   \
			    members);                                                                                                                                            \
		}                                                                                                                                                          \
		template<size_t I>                                                                                                                                         \
		static bool check_member_at_index(const T& v) {                                                                                                            \
			constexpr auto member = std::get<I>(members);                                                                                                            \
			return (v.*member) == member_placeholder<member>::value;                                                                                                 \
		}                                                                                                                                                          \
		static bool is_default(void* value, size_t index) {                                                                                                        \
			T& v = *reinterpret_cast<T*>(value);                                                                                                                     \
			bool result = false;                                                                                                                                     \
			[&]<size_t... I>(std::index_sequence<I...>) {                                                                                                            \
				((result = (I == index ? check_member_at_index<I>(v) : result)), ...);                                                                                 \
			}(std::make_index_sequence<std::tuple_size_v<decltype(members)>>{});                                                                                     \
			return result;                                                                                                                                           \
		}                                                                                                                                                          \
		static void destroy(void* value) {                                                                                                                         \
			T& v = *reinterpret_cast<T*>(value);                                                                                                                     \
			v.~T();                                                                                                                                                  \
		}                                                                                                                                                          \
		static constexpr const char* name = #type;                                                                                                                 \
		struct proxy {                                                                                                                                             \
			EVAL(MAKE_EXPAND TRANSFORMSC(WRAPPED_T, (__VA_ARGS__)));                                                                                                 \
			proxy* operator->() {                                                                                                                                    \
				return this;                                                                                                                                           \
			}                                                                                                                                                        \
		};                                                                                                                                                         \
	}
} // namespace vuk