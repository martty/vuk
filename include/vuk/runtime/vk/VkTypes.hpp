#pragma once

#include "vuk/Config.hpp"
#include "vuk/Flags.hpp"
#include <cstdint>

namespace vuk {
	enum class IndexType : uint32_t { eUint16 = 0, eUint32 = 1, eNoneKHR = 1000165000, eUint8EXT = 1000265000, eNoneNV = 1000165000 };

	enum class ShaderStageFlagBits : uint32_t {
		eVertex = 0x00000001,
		eTessellationControl = 0x00000002,
		eTessellationEvaluation = 0x00000004,
		eGeometry = 0x00000008,
		eFragment = 0x00000010,
		eCompute = 0x00000020,
		eAllGraphics = 0x0000001F,
		eAll = 0x7FFFFFFF,
		eRaygenKHR = 0x00000100,
		eAnyHitKHR = 0x00000200,
		eClosestHitKHR = 0x00000400,
		eMissKHR = 0x00000800,
		eIntersectionKHR = 0x00001000,
		eCallableKHR = 0x00002000,
		eTaskNV = 0x00000040,
		eMeshNV = 0x00000080,
		eAnyHitNV = 0x00000200,
		eCallableNV = 0x00002000,
		eClosestHitNV = 0x00000400,
		eIntersectionNV = 0x00001000,
		eMissNV = 0x00000800,
		eRaygenNV = 0x00000100
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

	enum class PipelineStageFlagBits : uint64_t {
		eNone = 0,
		eTopOfPipe = 0x00000001,
		eDrawIndirect = 0x00000002,
		eVertexInput = 0x00000004,
		eVertexShader = 0x00000008,
		eTessellationControlShader = 0x00000010,
		eTessellationEvaluationShader = 0x00000020,
		eGeometryShader = 0x00000040,
		eFragmentShader = 0x00000080,
		eEarlyFragmentTests = 0x00000100,
		eLateFragmentTests = 0x00000200,
		eColorAttachmentOutput = 0x00000400,
		eComputeShader = 0x00000800,
		eTransfer = 0x00001000,
		eBottomOfPipe = 0x00002000,
		eHost = 0x00004000,
		eAllGraphics = 0x00008000,
		eAllCommands = 0x00010000,
		eTransformFeedbackEXT = 0x01000000,
		eConditionalRenderingEXT = 0x00040000,
		eRayTracingShaderKHR = 0x00200000,
		eAccelerationStructureBuildKHR = 0x02000000,
		eShadingRateImageNV = 0x00400000,
		eTaskShaderNV = 0x00080000,
		eMeshShaderNV = 0x00100000,
		eFragmentDensityProcessEXT = 0x00800000,
		eCommandPreprocessNV = 0x00020000,
		eAccelerationStructureBuildNV = 0x02000000,
		eRayTracingShaderNV = 0x00200000,
		eCopy = 0x100000000ULL,
		eBlit = 0x200000000ULL,
		eResolve = 0x400000000ULL,
		eClear = 0x800000000ULL
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

	enum class AccessFlagBits : uint64_t {
		eNone = 0,
		eIndirectCommandRead = 0x00000001,
		eIndexRead = 0x00000002,
		eVertexAttributeRead = 0x00000004,
		eUniformRead = 0x00000008,
		eInputAttachmentRead = 0x00000010,
		eShaderRead = 0x00000020,
		eShaderWrite = 0x00000040,
		eColorAttachmentRead = 0x00000080,
		eColorAttachmentWrite = 0x00000100,
		eDepthStencilAttachmentRead = 0x00000200,
		eDepthStencilAttachmentWrite = 0x00000400,
		eTransferRead = 0x00000800,
		eTransferWrite = 0x00001000,
		eHostRead = 0x00002000,
		eHostWrite = 0x00004000,
		eMemoryRead = 0x00008000,
		eMemoryWrite = 0x00010000,
		eTransformFeedbackWriteEXT = 0x02000000,
		eTransformFeedbackCounterReadEXT = 0x04000000,
		eTransformFeedbackCounterWriteEXT = 0x08000000,
		eConditionalRenderingReadEXT = 0x00100000,
		eColorAttachmentReadNoncoherentEXT = 0x00080000,
		eAccelerationStructureReadKHR = 0x00200000,
		eAccelerationStructureWriteKHR = 0x00400000,
		eShadingRateImageReadNV = 0x00800000,
		eFragmentDensityMapReadEXT = 0x01000000,
		eCommandPreprocessReadNV = 0x00020000,
		eCommandPreprocessWriteNV = 0x00040000,
		eAccelerationStructureReadNV = 0x00200000,
		eAccelerationStructureWriteNV = 0x00400000,
		eShaderSampledRead = 0x100000000ULL,
		eShaderStorageRead = 0x200000000ULL,
		eShaderStorageWrite = 0x400000000ULL
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
		VkCommandBufferLevel level = (VkCommandBufferLevel)0;
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

	enum class BufferUsageFlagBits : uint32_t {
		eTransferRead = 0x00000001,
		eTransferWrite = 0x00000002,
		eUniformTexelBuffer = 0x00000004,
		eStorageTexelBuffer = 0x00000008,
		eUniformBuffer = 0x00000010,
		eStorageBuffer = 0x00000020,
		eIndexBuffer = 0x00000040,
		eVertexBuffer = 0x00000080,
		eIndirectBuffer = 0x00000100,
		eShaderDeviceAddress = 0x00020000,
		eTransformFeedbackBufferEXT = 0x00000800,
		eTransformFeedbackCounterBufferEXT = 0x00001000,
		eConditionalRenderingEXT = 0x00000200,
		eShaderDeviceAddressEXT = 0x00020000,
		eShaderDeviceAddressKHR = 0x00020000,
		eAccelerationStructureBuildInputReadOnlyKHR = 0x00080000,
		eAccelerationStructureStorageKHR = 0x00100000,
		eShaderBindingTableKHR = 0x00000400
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

	enum class ImageLayout : uint32_t {
		eUndefined = 0,
		eGeneral = 1,
		eColorAttachmentOptimal = 2,
		eDepthStencilAttachmentOptimal = 3,
		eDepthStencilReadOnlyOptimal = 4,
		eShaderReadOnlyOptimal = 5,
		eTransferSrcOptimal = 6,
		eTransferDstOptimal = 7,
		ePreinitialized = 8,
		eDepthReadOnlyStencilAttachmentOptimal = 1000117000,
		eDepthAttachmentStencilReadOnlyOptimal = 1000117001,
		eDepthAttachmentOptimal = 1000241000,
		eDepthReadOnlyOptimal = 1000241001,
		eStencilAttachmentOptimal = 1000241002,
		eStencilReadOnlyOptimal = 1000241003,
		ePresentSrcKHR = 1000001002,
		eSharedPresentKHR = 1000111000,
		eShadingRateOptimalNV = 1000164003,
		eFragmentDensityMapOptimalEXT = 1000218000,
		eDepthAttachmentOptimalKHR = 1000241000,
		eDepthAttachmentStencilReadOnlyOptimalKHR = 1000117001,
		eDepthReadOnlyOptimalKHR = 1000241001,
		eDepthReadOnlyStencilAttachmentOptimalKHR = 1000117000,
		eStencilAttachmentOptimalKHR = 1000241002,
		eStencilReadOnlyOptimalKHR = 1000241003,
		eReadOnlyOptimalKHR = 1000314000,
		eReadOnlyOptimal = 1000314000,
		eAttachmentOptimalKHR = 1000314001,
		eAttachmentOptimal = 1000314001
	};
} // namespace vuk