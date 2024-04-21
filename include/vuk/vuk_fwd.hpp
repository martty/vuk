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
	struct Buffer;

	struct ImageAttachment;

	struct Compiler;
} // namespace vuk
