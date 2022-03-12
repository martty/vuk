#pragma once

#include "vuk/Name.hpp"

namespace vuk {
	class Context;
	class Allocator;

	class CommandBuffer;

	struct Swapchain;
	using SwapchainRef = Swapchain*;

	class LegacyGPUAllocator;

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

	struct Exception;
	struct ShaderCompilationException;
	struct RenderGraphException;
	struct AllocateException;
	struct PresentException;
	struct VkException;

	template<class V, class E = Exception>
	struct Result;

	template<class T>
	class Unique;

	struct FramebufferCreateInfo;

	struct BufferCreateInfo;

	struct Buffer;
	struct BufferGPU;
	struct BufferCrossDevice;

	struct Query;
	struct TimestampQuery;
	struct TimestampQueryPool;
	struct TimestampQueryCreateInfo;

	struct CommandBufferAllocationCreateInfo;
	struct CommandBufferAllocation;

	struct SetBinding;
	struct DescriptorSet;
	struct PersistentDescriptorSetCreateInfo;
	struct PersistentDescriptorSet;

	struct ShaderModule;
	struct PipelineBaseCreateInfo;
	struct PipelineBaseInfo;
	struct Program;

	struct PersistentDescriptorSet;
	
	struct FutureBase;
	template<class T>
	class Future;
} // namespace vuk
