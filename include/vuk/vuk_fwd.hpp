#pragma once

#include "vuk/Name.hpp"

namespace vuk {
	class Allocator;
	class CommandBuffer;

	class Runtime;
	struct Swapchain;

	struct Query;
	struct TimestampQuery;
	struct TimestampQueryPool;

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

	struct GraphicsPipelineInfo;
	struct GraphicsPipelineInstanceCreateInfo;
	struct ComputePipelineInfo;
	struct ComputePipelineInstanceCreateInfo;
	struct RayTracingPipelineInfo;
	struct RayTracingPipelineInstanceCreateInfo;

	struct ShaderSource;

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

	struct BufferCreateInfo;
	template<class T>
	struct BufferLike;

	struct ImageAttachment;

	template<class Type, size_t Extent>
	struct view;

	using byte = std::byte;
	inline constexpr size_t dynamic_extent = -1;

	template<class Type = byte, size_t Extent = dynamic_extent>
	using Buffer = view<BufferLike<Type>, Extent>;

	struct VirtualAllocation;
	struct VirtualAddressSpace;
	struct VirtualAllocationCreateInfo;
	struct VirtualAddressSpaceCreateInfo;

	struct Compiler;
} // namespace vuk
