#pragma once
#include <stdint.h>
#include <span>

//https://gist.github.com/filsinger/1255697/21762ea83a2d3c17561c8e6a29f44249a4626f9e

namespace hash {
	template <typename S> struct fnv_internal;
	template <typename S> struct fnv1a_tpl;

	template <> struct fnv_internal<uint32_t> {
		constexpr static uint32_t default_offset_basis = 0x811C9DC5;
		constexpr static uint32_t prime = 0x01000193;
	};

	template <> struct fnv1a_tpl<uint32_t> : public fnv_internal<uint32_t> {
		constexpr static inline uint32_t hash(char const*const aString, const uint32_t val = default_offset_basis) {
			return (aString[0] == '\0') ? val : hash(&aString[1], (val ^ uint32_t(aString[0])) * prime);
		}

		constexpr static inline uint32_t hash(char const*const aString, const size_t aStrlen, const uint32_t val) {
			return (aStrlen == 0) ? val : hash(aString + 1, aStrlen - 1, (val ^ uint32_t(aString[0])) * prime);
		}
	};

	using fnv1a = fnv1a_tpl<uint32_t>;
} // namespace hash

inline constexpr uint32_t operator "" _fnv1a(const char* aString, const size_t aStrlen) {
	typedef hash::fnv1a_tpl<uint32_t> hash_type;
	return hash_type::hash(aString, aStrlen, hash_type::default_offset_basis);
}

template <typename T>
inline void hash_combine(size_t& seed, const T& v) {
	std::hash<T> hasher;
	seed ^= hasher(v) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
}
template <typename T, typename... Rest>
inline void hash_combine(size_t& seed, const T& v, Rest&&... rest) {
	std::hash<T> hasher;
	seed ^= hasher(v) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
	hash_combine(seed, std::forward<Rest>(rest)...);
}

template<typename E>
constexpr auto to_integral(E e) -> typename std::underlying_type<E>::type {
	return static_cast<typename std::underlying_type<E>::type>(e);
}

namespace std {
	template <class T, size_t E>
	struct hash<std::span<T, E>> {
		size_t operator()(std::span<T, E> const& x) const noexcept {
			size_t h = 0;
			for (auto& e : x) {
				hash_combine(h, e);
			}
			return h;
		}
	};
};
