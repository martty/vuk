#pragma once

#include <tuple>
#include <type_traits>

namespace vuk {
	struct CommandBuffer;

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

	template<class T, class... Ts>
	constexpr std::size_t index_of(const std::tuple<Ts...>&) {
		int found{}, count{};
		((!found ? (++count, found = std::is_same_v<T, Ts>) : 0), ...);
		return found ? count - 1 : count;
	}
} // namespace vuk