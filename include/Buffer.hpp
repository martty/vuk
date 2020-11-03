#pragma once

#include "Types.hpp"

namespace vuk {
    enum class BufferUsageFlagBits : VkBufferUsageFlags {
        eTransferSrc = VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
        eTransferDst = VK_BUFFER_USAGE_TRANSFER_DST_BIT,
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
        eRayTracingKHR = VK_BUFFER_USAGE_RAY_TRACING_BIT_KHR,
        eRayTracingNV = VK_BUFFER_USAGE_RAY_TRACING_BIT_NV,
        eShaderDeviceAddressEXT = VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT_EXT,
        eShaderDeviceAddressKHR = VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT_KHR
    };

    using BufferUsageFlags = Flags<BufferUsageFlagBits>;

    inline constexpr BufferUsageFlags operator|(BufferUsageFlagBits bit0, BufferUsageFlags bit1) noexcept {
        return BufferUsageFlags(bit0) | bit1;
    }

    inline constexpr BufferUsageFlags operator&(BufferUsageFlagBits bit0, BufferUsageFlagBits bit1) noexcept {
        return BufferUsageFlags(bit0) & bit1;
    }

    inline constexpr BufferUsageFlags operator^(BufferUsageFlagBits bit0, BufferUsageFlagBits bit1) noexcept {
        return BufferUsageFlags(bit0) ^ bit1;
    }

	struct Buffer {
		VkDeviceMemory device_memory;
		VkBuffer buffer;
		size_t offset;
		size_t size;
		void* mapped_ptr;

		bool operator==(const Buffer& o) const {
			return std::tie(device_memory, buffer, offset, size) ==
				std::tie(o.device_memory, o.buffer, o.offset, o.size);
		}
	};
}
