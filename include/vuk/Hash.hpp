#pragma once
#include <stdint.h>
#include <string_view>
#include <utility>

// https://gist.github.com/filsinger/1255697/21762ea83a2d3c17561c8e6a29f44249a4626f9e

namespace hash {
	template<typename S>
	struct fnv_internal;
	template<typename S>
	struct fnv1a_tpl;

	template<>
	struct fnv_internal<uint32_t> {
		constexpr static uint32_t default_offset_basis = 0x811C9DC5;
		constexpr static uint32_t prime = 0x01000193;
	};

	template<>
	struct fnv1a_tpl<uint32_t> : public fnv_internal<uint32_t> {
		constexpr static inline uint32_t hash(char const* const aString, const uint32_t val = default_offset_basis) {
			return (aString[0] == '\0') ? val : hash(&aString[1], (val ^ uint32_t(aString[0])) * prime);
		}

		constexpr static inline uint32_t hash(char const* const aString, const size_t aStrlen, const uint32_t val) {
			return (aStrlen == 0) ? val : hash(aString + 1, aStrlen - 1, (val ^ uint32_t(aString[0])) * prime);
		}
	};

	using fnv1a = fnv1a_tpl<uint32_t>;
} // namespace hash

inline constexpr uint32_t operator"" _fnv1a(const char* aString, const size_t aStrlen) {
	typedef hash::fnv1a_tpl<uint32_t> hash_type;
	return hash_type::hash(aString, aStrlen, hash_type::default_offset_basis);
}

template<typename T>
inline void hash_combine(size_t& seed, const T& v) noexcept {
	std::hash<T> hasher;
	seed ^= hasher(v) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
}

inline constexpr void hash_combine_direct(uint32_t& seed, uint32_t v) noexcept {
	seed ^= v + 0x9e3779b9 + (seed << 6) + (seed >> 2);
}

#define FWD(x) (static_cast<decltype(x)&&>(x))

template<typename T, typename... Rest>
inline void hash_combine(size_t& seed, const T& v, Rest&&... rest) noexcept {
	std::hash<T> hasher;
	seed ^= hasher(v) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
	hash_combine(seed, FWD(rest)...);
}

#undef FWD
