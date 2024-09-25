#pragma once

#include "vuk/Types.hpp"
#include <assert.h>

namespace vuk {
	/// @brief A contiguous portion of GPU-visible memory that can be used for storing buffer-type data
	struct Buffer {
		void* allocation = nullptr;
		VkBuffer buffer = VK_NULL_HANDLE;
		size_t offset = 0;
		size_t size = ~(0u);
		uint64_t device_address = 0;
		std::byte* mapped_ptr = nullptr;
		MemoryUsage memory_usage;

		constexpr bool operator==(const Buffer& o) const noexcept {
			return buffer == o.buffer && offset == o.offset && size == o.size;
		}

		constexpr explicit operator bool() const noexcept {
			return buffer != VK_NULL_HANDLE;
		}

		/// @brief Create a new Buffer by offsetting
		[[nodiscard]] Buffer add_offset(VkDeviceSize offset_to_add) {
			assert(offset_to_add <= size);
			return { allocation,
				       buffer,
				       offset + offset_to_add,
				       size - offset_to_add,
				       device_address != 0 ? device_address + offset_to_add : 0,
				       mapped_ptr != nullptr ? mapped_ptr + offset_to_add : nullptr,
				       memory_usage };
		}

		/// @brief Create a new Buffer that is a subset of the original
		[[nodiscard]] Buffer subrange(VkDeviceSize new_offset, VkDeviceSize new_size) {
			assert(new_offset + new_size <= size);
			return { allocation,
				       buffer,
				       offset + new_offset,
				       new_size,
				       device_address != 0 ? device_address + new_offset : 0,
				       mapped_ptr != nullptr ? mapped_ptr + new_offset : nullptr,
				       memory_usage };
		}
	};

	/// @brief Buffer creation parameters
	struct BufferCreateInfo {
		/// @brief Memory usage to determine which heap to allocate the memory from
		MemoryUsage mem_usage;
		/// @brief Size of the Buffer in bytes
		VkDeviceSize size;
		/// @brief Alignment of the allocated Buffer in bytes
		VkDeviceSize alignment = 1;
	};
} // namespace vuk

namespace std {
	template<>
	struct hash<vuk::Buffer> {
		size_t operator()(vuk::Buffer const& x) const noexcept;
	};
} // namespace std