#pragma once

#include <vuk/Name.hpp>

namespace vuk {
	class Context;
	class InflightContext;
	class PerThreadContext;

	class CommandBuffer;

	struct Swapchain;
	using SwapChainRef = Swapchain*;

	class Allocator;

	struct ShaderSource;

	// temporary
	struct RGImage;
	struct RGCI;

	// 0b00111 -> 3
	inline uint32_t num_leading_ones(uint32_t mask) noexcept {
#ifdef __has_builtin
#if __has_builtin(__builtin_clz)
		return (31 ^ __builtin_clz(mask)) + 1;
#else
#error "__builtin_clz not available"
#endif
#else
		unsigned long lz;
		if (!_BitScanReverse(&lz, mask))
			return 0;
		return lz + 1;
#endif
	}

	// return a/b rounded to infinity
	constexpr uint64_t idivceil(uint64_t a, uint64_t b) noexcept {
		return (a + b - 1) / b;
	}
}
