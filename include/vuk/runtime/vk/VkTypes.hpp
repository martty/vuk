#pragma once

namespace vuk {
	enum class IndexType {
		eUint16 = VK_INDEX_TYPE_UINT16,
		eUint32 = VK_INDEX_TYPE_UINT32,
		eNoneKHR = VK_INDEX_TYPE_NONE_KHR,
		eUint8EXT = VK_INDEX_TYPE_UINT8_EXT,
		eNoneNV = VK_INDEX_TYPE_NONE_NV
	};

	enum class ShaderStageFlagBits : VkShaderStageFlags {
		eVertex = VK_SHADER_STAGE_VERTEX_BIT,
		eTessellationControl = VK_SHADER_STAGE_TESSELLATION_CONTROL_BIT,
		eTessellationEvaluation = VK_SHADER_STAGE_TESSELLATION_EVALUATION_BIT,
		eGeometry = VK_SHADER_STAGE_GEOMETRY_BIT,
		eFragment = VK_SHADER_STAGE_FRAGMENT_BIT,
		eCompute = VK_SHADER_STAGE_COMPUTE_BIT,
		eAllGraphics = VK_SHADER_STAGE_ALL_GRAPHICS,
		eAll = VK_SHADER_STAGE_ALL,
		eRaygenKHR = VK_SHADER_STAGE_RAYGEN_BIT_KHR,
		eAnyHitKHR = VK_SHADER_STAGE_ANY_HIT_BIT_KHR,
		eClosestHitKHR = VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR,
		eMissKHR = VK_SHADER_STAGE_MISS_BIT_KHR,
		eIntersectionKHR = VK_SHADER_STAGE_INTERSECTION_BIT_KHR,
		eCallableKHR = VK_SHADER_STAGE_CALLABLE_BIT_KHR,
		eTaskNV = VK_SHADER_STAGE_TASK_BIT_NV,
		eMeshNV = VK_SHADER_STAGE_MESH_BIT_NV,
		eTaskEXT = VK_SHADER_STAGE_TASK_BIT_EXT,
		eMeshEXT = VK_SHADER_STAGE_MESH_BIT_EXT,
		eAnyHitNV = VK_SHADER_STAGE_ANY_HIT_BIT_NV,
		eCallableNV = VK_SHADER_STAGE_CALLABLE_BIT_NV,
		eClosestHitNV = VK_SHADER_STAGE_CLOSEST_HIT_BIT_NV,
		eIntersectionNV = VK_SHADER_STAGE_INTERSECTION_BIT_NV,
		eMissNV = VK_SHADER_STAGE_MISS_BIT_NV,
		eRaygenNV = VK_SHADER_STAGE_RAYGEN_BIT_NV
	};

	using ShaderStageFlags = Flags<ShaderStageFlagBits>;
	inline constexpr ShaderStageFlags operator|(ShaderStageFlagBits bit0, ShaderStageFlagBits bit1) noexcept {
		return ShaderStageFlags(bit0) | bit1;
	}

	inline constexpr ShaderStageFlags operator&(ShaderStageFlagBits bit0, ShaderStageFlagBits bit1) noexcept {
		return ShaderStageFlags(bit0) & bit1;
	}

	inline constexpr ShaderStageFlags operator^(ShaderStageFlagBits bit0, ShaderStageFlagBits bit1) noexcept {
		return ShaderStageFlags(bit0) ^ bit1;
	}

	enum class PipelineStageFlagBits : VkFlags64 {
		eNone = VK_PIPELINE_STAGE_NONE,
		eTopOfPipe = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
		eDrawIndirect = VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT,
		eVertexInput = VK_PIPELINE_STAGE_VERTEX_INPUT_BIT,
		eVertexShader = VK_PIPELINE_STAGE_VERTEX_SHADER_BIT,
		eTessellationControlShader = VK_PIPELINE_STAGE_TESSELLATION_CONTROL_SHADER_BIT,
		eTessellationEvaluationShader = VK_PIPELINE_STAGE_TESSELLATION_EVALUATION_SHADER_BIT,
		eGeometryShader = VK_PIPELINE_STAGE_GEOMETRY_SHADER_BIT,
		eFragmentShader = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
		eEarlyFragmentTests = VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT,
		eLateFragmentTests = VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT,
		eColorAttachmentOutput = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
		eComputeShader = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
		eTransfer = VK_PIPELINE_STAGE_TRANSFER_BIT,
		eBottomOfPipe = VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
		eHost = VK_PIPELINE_STAGE_HOST_BIT,
		eAllGraphics = VK_PIPELINE_STAGE_ALL_GRAPHICS_BIT,
		eAllCommands = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
		eTransformFeedbackEXT = VK_PIPELINE_STAGE_TRANSFORM_FEEDBACK_BIT_EXT,
		eConditionalRenderingEXT = VK_PIPELINE_STAGE_CONDITIONAL_RENDERING_BIT_EXT,
		eRayTracingShaderKHR = VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR,
		eAccelerationStructureBuildKHR = VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
		eShadingRateImageNV = VK_PIPELINE_STAGE_SHADING_RATE_IMAGE_BIT_NV,
		eTaskShaderNV = VK_PIPELINE_STAGE_TASK_SHADER_BIT_NV,
		eMeshShaderNV = VK_PIPELINE_STAGE_MESH_SHADER_BIT_NV,
		eTaskShaderEXT = VK_PIPELINE_STAGE_TASK_SHADER_BIT_EXT,
		eMeshShaderEXT = VK_PIPELINE_STAGE_MESH_SHADER_BIT_EXT,
		eFragmentDensityProcessEXT = VK_PIPELINE_STAGE_FRAGMENT_DENSITY_PROCESS_BIT_EXT,
		eCommandPreprocessNV = VK_PIPELINE_STAGE_COMMAND_PREPROCESS_BIT_NV,
		eAccelerationStructureBuildNV = VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_NV,
		eRayTracingShaderNV = VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_NV,
		eCopy = VK_PIPELINE_STAGE_2_COPY_BIT_KHR,
		eBlit = VK_PIPELINE_STAGE_2_BLIT_BIT_KHR,
		eResolve = VK_PIPELINE_STAGE_2_RESOLVE_BIT_KHR,
		eClear = VK_PIPELINE_STAGE_2_CLEAR_BIT_KHR
	};

	using PipelineStageFlags = Flags<PipelineStageFlagBits>;
	inline constexpr PipelineStageFlags operator|(PipelineStageFlagBits bit0, PipelineStageFlagBits bit1) noexcept {
		return PipelineStageFlags(bit0) | bit1;
	}

	inline constexpr PipelineStageFlags operator&(PipelineStageFlagBits bit0, PipelineStageFlagBits bit1) noexcept {
		return PipelineStageFlags(bit0) & bit1;
	}

	inline constexpr PipelineStageFlags operator^(PipelineStageFlagBits bit0, PipelineStageFlagBits bit1) noexcept {
		return PipelineStageFlags(bit0) ^ bit1;
	}

	enum class AccessFlagBits : VkFlags64 {
		eNone = VK_ACCESS_NONE_KHR,
		eIndirectCommandRead = VK_ACCESS_INDIRECT_COMMAND_READ_BIT,
		eIndexRead = VK_ACCESS_INDEX_READ_BIT,
		eVertexAttributeRead = VK_ACCESS_VERTEX_ATTRIBUTE_READ_BIT,
		eUniformRead = VK_ACCESS_UNIFORM_READ_BIT,
		eInputAttachmentRead = VK_ACCESS_INPUT_ATTACHMENT_READ_BIT,
		eShaderRead = VK_ACCESS_SHADER_READ_BIT,
		eShaderWrite = VK_ACCESS_SHADER_WRITE_BIT,
		eColorAttachmentRead = VK_ACCESS_COLOR_ATTACHMENT_READ_BIT,
		eColorAttachmentWrite = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
		eDepthStencilAttachmentRead = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT,
		eDepthStencilAttachmentWrite = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
		eTransferRead = VK_ACCESS_TRANSFER_READ_BIT,
		eTransferWrite = VK_ACCESS_TRANSFER_WRITE_BIT,
		eHostRead = VK_ACCESS_HOST_READ_BIT,
		eHostWrite = VK_ACCESS_HOST_WRITE_BIT,
		eMemoryRead = VK_ACCESS_MEMORY_READ_BIT,
		eMemoryWrite = VK_ACCESS_MEMORY_WRITE_BIT,
		eTransformFeedbackWriteEXT = VK_ACCESS_TRANSFORM_FEEDBACK_WRITE_BIT_EXT,
		eTransformFeedbackCounterReadEXT = VK_ACCESS_TRANSFORM_FEEDBACK_COUNTER_READ_BIT_EXT,
		eTransformFeedbackCounterWriteEXT = VK_ACCESS_TRANSFORM_FEEDBACK_COUNTER_WRITE_BIT_EXT,
		eConditionalRenderingReadEXT = VK_ACCESS_CONDITIONAL_RENDERING_READ_BIT_EXT,
		eColorAttachmentReadNoncoherentEXT = VK_ACCESS_COLOR_ATTACHMENT_READ_NONCOHERENT_BIT_EXT,
		eAccelerationStructureReadKHR = VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR,
		eAccelerationStructureWriteKHR = VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR,
		eShadingRateImageReadNV = VK_ACCESS_SHADING_RATE_IMAGE_READ_BIT_NV,
		eFragmentDensityMapReadEXT = VK_ACCESS_FRAGMENT_DENSITY_MAP_READ_BIT_EXT,
		eCommandPreprocessReadNV = VK_ACCESS_COMMAND_PREPROCESS_READ_BIT_NV,
		eCommandPreprocessWriteNV = VK_ACCESS_COMMAND_PREPROCESS_WRITE_BIT_NV,
		eAccelerationStructureReadNV = VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_NV,
		eAccelerationStructureWriteNV = VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_NV,
		eShaderSampledRead = VK_ACCESS_2_SHADER_SAMPLED_READ_BIT_KHR,
		eShaderStorageRead = VK_ACCESS_2_SHADER_STORAGE_READ_BIT_KHR,
		eShaderStorageWrite = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT_KHR
	};

	using AccessFlags = Flags<AccessFlagBits>;

	inline constexpr AccessFlags operator|(AccessFlagBits bit0, AccessFlagBits bit1) noexcept {
		return AccessFlags(bit0) | bit1;
	}

	inline constexpr AccessFlags operator&(AccessFlagBits bit0, AccessFlagBits bit1) noexcept {
		return AccessFlags(bit0) & bit1;
	}

	inline constexpr AccessFlags operator^(AccessFlagBits bit0, AccessFlagBits bit1) noexcept {
		return AccessFlags(bit0) ^ bit1;
	}

	struct CommandPool {
		VkCommandPool command_pool = VK_NULL_HANDLE;
		uint32_t queue_family_index;

		constexpr bool operator==(const CommandPool& other) const noexcept {
			return command_pool == other.command_pool;
		}
	};

	struct CommandBufferAllocationCreateInfo {
		VkCommandBufferLevel level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
		CommandPool command_pool;
	};

	struct CommandBufferAllocation {
		CommandBufferAllocation() = default;
		CommandBufferAllocation(VkCommandBuffer command_buffer, CommandPool command_pool) noexcept : command_buffer(command_buffer), command_pool(command_pool) {}

		VkCommandBuffer command_buffer = VK_NULL_HANDLE;
		CommandPool command_pool = {};

		operator VkCommandBuffer() noexcept {
			return command_buffer;
		}
	};

	enum class BufferUsageFlagBits : VkBufferUsageFlags {
		eTransferRead = VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
		eTransferWrite = VK_BUFFER_USAGE_TRANSFER_DST_BIT,
		eUniformTexelBuffer = VK_BUFFER_USAGE_UNIFORM_TEXEL_BUFFER_BIT,
		eStorageTexelBuffer = VK_BUFFER_USAGE_STORAGE_TEXEL_BUFFER_BIT,
		eUniformBuffer = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
		eStorageBuffer = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
		eIndexBuffer = VK_BUFFER_USAGE_INDEX_BUFFER_BIT,
		eVertexBuffer = VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
		eIndirectBuffer = VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT,
		eShaderDeviceAddress = VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
		eTransformFeedbackBufferEXT = VK_BUFFER_USAGE_TRANSFORM_FEEDBACK_BUFFER_BIT_EXT,
		eTransformFeedbackCounterBufferEXT = VK_BUFFER_USAGE_TRANSFORM_FEEDBACK_COUNTER_BUFFER_BIT_EXT,
		eConditionalRenderingEXT = VK_BUFFER_USAGE_CONDITIONAL_RENDERING_BIT_EXT,
		eShaderDeviceAddressEXT = VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT_EXT,
		eShaderDeviceAddressKHR = VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT_KHR,
		eAccelerationStructureBuildInputReadOnlyKHR = VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR,
		eAccelerationStructureStorageKHR = VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR,
		eShaderBindingTableKHR = VK_BUFFER_USAGE_SHADER_BINDING_TABLE_BIT_KHR
	};

	using BufferUsageFlags = Flags<BufferUsageFlagBits>;

	inline constexpr BufferUsageFlags operator|(BufferUsageFlagBits bit0, BufferUsageFlagBits bit1) noexcept {
		return BufferUsageFlags(bit0) | bit1;
	}

	inline constexpr BufferUsageFlags operator&(BufferUsageFlagBits bit0, BufferUsageFlagBits bit1) noexcept {
		return BufferUsageFlags(bit0) & bit1;
	}

	inline constexpr BufferUsageFlags operator^(BufferUsageFlagBits bit0, BufferUsageFlagBits bit1) noexcept {
		return BufferUsageFlags(bit0) ^ bit1;
	}

	enum class ImageLayout {
		eUndefined = VK_IMAGE_LAYOUT_UNDEFINED,
		eGeneral = VK_IMAGE_LAYOUT_GENERAL,
		eColorAttachmentOptimal = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
		eDepthStencilAttachmentOptimal = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
		eDepthStencilReadOnlyOptimal = VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL,
		eShaderReadOnlyOptimal = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
		eTransferSrcOptimal = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
		eTransferDstOptimal = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
		ePreinitialized = VK_IMAGE_LAYOUT_PREINITIALIZED,
		eDepthReadOnlyStencilAttachmentOptimal = VK_IMAGE_LAYOUT_DEPTH_READ_ONLY_STENCIL_ATTACHMENT_OPTIMAL,
		eDepthAttachmentStencilReadOnlyOptimal = VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_STENCIL_READ_ONLY_OPTIMAL,
		eDepthAttachmentOptimal = VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL,
		eDepthReadOnlyOptimal = VK_IMAGE_LAYOUT_DEPTH_READ_ONLY_OPTIMAL,
		eStencilAttachmentOptimal = VK_IMAGE_LAYOUT_STENCIL_ATTACHMENT_OPTIMAL,
		eStencilReadOnlyOptimal = VK_IMAGE_LAYOUT_STENCIL_READ_ONLY_OPTIMAL,
		ePresentSrcKHR = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
		eSharedPresentKHR = VK_IMAGE_LAYOUT_SHARED_PRESENT_KHR,
		eShadingRateOptimalNV = VK_IMAGE_LAYOUT_SHADING_RATE_OPTIMAL_NV,
		eFragmentDensityMapOptimalEXT = VK_IMAGE_LAYOUT_FRAGMENT_DENSITY_MAP_OPTIMAL_EXT,
		eDepthAttachmentOptimalKHR = VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL_KHR,
		eDepthAttachmentStencilReadOnlyOptimalKHR = VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_STENCIL_READ_ONLY_OPTIMAL_KHR,
		eDepthReadOnlyOptimalKHR = VK_IMAGE_LAYOUT_DEPTH_READ_ONLY_OPTIMAL_KHR,
		eDepthReadOnlyStencilAttachmentOptimalKHR = VK_IMAGE_LAYOUT_DEPTH_READ_ONLY_STENCIL_ATTACHMENT_OPTIMAL_KHR,
		eStencilAttachmentOptimalKHR = VK_IMAGE_LAYOUT_STENCIL_ATTACHMENT_OPTIMAL_KHR,
		eStencilReadOnlyOptimalKHR = VK_IMAGE_LAYOUT_STENCIL_READ_ONLY_OPTIMAL_KHR,
		eReadOnlyOptimalKHR = VK_IMAGE_LAYOUT_READ_ONLY_OPTIMAL_KHR,
		eReadOnlyOptimal = VK_IMAGE_LAYOUT_READ_ONLY_OPTIMAL,
		eAttachmentOptimalKHR = VK_IMAGE_LAYOUT_ATTACHMENT_OPTIMAL_KHR,
		eAttachmentOptimal = VK_IMAGE_LAYOUT_ATTACHMENT_OPTIMAL
	};
} // namespace vuk