#pragma once

#include <string_view>

namespace vuk {
	class Context;
	class InflightContext;
	class PerThreadContext;
	
	using Name = std::string_view;

	class CommandBuffer;

	struct Swapchain;
	using SwapChainRef = Swapchain *;

	// 0b00111 -> 3
	inline uint32_t num_leading_ones(uint32_t mask) {
#ifdef __builtin_clz
		return (31 ^ __builtin_clz(mask)) + 1;
#else
		unsigned long lz;
		if (!_BitScanReverse(&lz, mask))
			return 0;
		return lz + 1;
#endif
	}
}
